"""Bounded capture for explicitly allowed executables, not a command or filesystem sandbox."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass

from .plugin_contracts import PluginResourcePolicy
from .plugin_process import PluginProcessError, _WindowsJob


@dataclass(frozen=True)
class CapturedProcess:
    """Actual process exit and bounded raw byte streams; nonzero exit is not hidden."""

    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessCaptureCancelled(RuntimeError):
    """The owning caller cancelled; process cleanup completes before this escapes."""


# 功能：
#   逐项回收本次拥有的进程组、Job 和管道；单项失败不阻止其他资源释放。
#   清理使用独立的有界等待；无法确认退出或通信线程仍存活时抛出汇总错误。
# 输入：
#   process：本次创建的宿主进程，不接受通过名称搜索的其他进程。
#   job：该进程绑定的 Windows Job；其他平台为空。
#   threads：本次已成功启动的通信线程。
# 输出：
#   None：不返回业务数据。
def _cleanup_capture(process, job: _WindowsJob | None, threads: list[threading.Thread]) -> None:
    errors: list[Exception] = []
    try:
        if os.name != "nt":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except OSError as error:
        errors.append(error)
    if job is not None:
        try:
            job.close()
        except Exception as error:
            errors.append(error)
    try:
        process.wait(timeout=5)
    except Exception as error:
        errors.append(error)
    for thread in threads:
        try:
            thread.join(timeout=2)
        except Exception as error:
            errors.append(error)
    if any(thread.is_alive() for thread in threads):
        # 活跃线程可能持有缓冲管道锁，不能在主线程强行 close 而再次无限阻塞。
        errors.append(TimeoutError("PROCESS_CAPTURE_CHANNEL_CLEANUP_TIMEOUT"))
    else:
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except BrokenPipeError:
                    pass
                except Exception as error:
                    errors.append(error)
    if errors:
        # 公开错误类型沿用调用方已处理的进程失败契约，完整清理原因保留在异常链。
        raise PluginProcessError("PROCESS_CAPTURE_CLEANUP_FAILED") from ExceptionGroup(
            "Owned process cleanup failures", errors
        )


# 功能：
#   1. 同时输送输入并收集两个输出流，以共享字节预算和总时限约束子进程通信。
#   2. 退出时清理本次创建的进程组；Windows 附加资源 Job，POSIX 不提供内存配额。
#   3. 可执行文件及参数的授权由调用方负责，本函数不构成文件系统或命令沙箱。
# 输入：
#   command：调用方已授权的可执行文件及独立参数序列。
#   stdin：提供给子进程的原始输入字节。
#   maximum_bytes：输入上限及标准输出、错误输出合计的字节上限。
#   timeout：进程运行与通信共用的总秒数预算。
#   environment：明确传给子进程的环境变量。
#   resource_policy：Windows Job 的内存、处理器时间与进程数限制。
#   cancel_event：调用方发出的线程安全取消信号，未提供时仅使用总时限。
# 输出：
#   result：实际退出码以及分别保存的标准输出、错误输出字节。
def capture_process(
    command: Sequence[str],
    *,
    stdin: bytes,
    maximum_bytes: int,
    timeout: float,
    environment: Mapping[str, str],
    resource_policy: PluginResourcePolicy,
    cancel_event: threading.Event | None = None,
) -> CapturedProcess:
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise ValueError("PROCESS_CAPTURE_CANCEL_EVENT_INVALID")
    if (
        type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= 16 * 1024 * 1024
        or type(stdin) is not bytes
        or len(stdin) > maximum_bytes
    ):
        raise ValueError("PROCESS_CAPTURE_SIZE_INVALID")
    # 先做范围判断，超大整数无需转换为浮点，统一返回配置错误而不是溢出异常。
    if type(timeout) not in (int, float) or not 0 < timeout <= 3600 or not math.isfinite(timeout):
        raise ValueError("PROCESS_CAPTURE_TIMEOUT_INVALID")
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
    ):
        raise ValueError("PROCESS_CAPTURE_COMMAND_INVALID")
    resource_policy = PluginResourcePolicy.model_validate(resource_policy.model_dump())
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessCaptureCancelled("PROCESS_CAPTURE_CANCELLED")
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=dict(environment),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    job = None
    threads: list[threading.Thread] = []
    streams = [bytearray(), bytearray()]
    failures: list[BaseException] = []
    lock = threading.Lock()
    total = 0

    # 功能：
    #   持续读取单个输出管道，使用共享锁原子核算两个输出流的合计预算。
    # 输入：
    #   pipe：子进程标准输出或错误输出管道。
    #   output：该管道对应的有界字节缓冲区。
    # 输出：
    #   None：不返回业务数据。
    def drain(pipe, output):
        nonlocal total
        try:
            while chunk := pipe.read(4096):
                with lock:
                    if total + len(chunk) > maximum_bytes:
                        raise ValueError("PROCESS_CAPTURE_OUTPUT_LIMIT")
                    total += len(chunk)
                    output.extend(chunk)
        except Exception as error:
            with lock:
                failures.append(error)
            if process.poll() is None:
                with suppress(OSError):
                    process.kill()

    # 功能：
    #   在独立线程输送输入，防止子进程不读取输入时阻塞负责超时处理的主线程。
    # 输入：
    #   无显式参数；使用外层的 process、stdin、failures 和 lock。
    # 输出：
    #   None：不返回业务数据。
    def feed():
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except BrokenPipeError:
            # Programs may legitimately exit without reading optional stdin;
            # their actual exit status/output remains visible to the caller.
            pass
        except Exception as error:
            with lock:
                failures.append(error)
            if process.poll() is None:
                with suppress(OSError):
                    process.kill()

    try:
        if os.name == "nt":
            job = _WindowsJob(process.pid, resource_policy)
        for target, args in (
            (drain, (process.stdout, streams[0])),
            (drain, (process.stderr, streams[1])),
            (feed, ()),
        ):
            thread = threading.Thread(
                target=target, args=args, name="broker-process-channel", daemon=True
            )
            thread.start()
            threads.append(thread)
        # 不能一次阻塞 wait 完整预算：协程取消需要在一个短轮询窗口内传到子进程。
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessCaptureCancelled("PROCESS_CAPTURE_CANCELLED")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            if process.poll() is None:
                try:
                    process.wait(timeout=min(0.05, remaining))
                except subprocess.TimeoutExpired:
                    continue
            elif all(not thread.is_alive() for thread in threads):
                break
            else:
                # 进程退出不代表后代已关闭输出管道，通信排空仍受同一时限和取消信号约束。
                for thread in threads:
                    thread.join(timeout=min(0.01, max(0, timeout - (time.monotonic() - started))))
        if failures:
            raise failures[0]
        result = CapturedProcess(process.returncode, bytes(streams[0]), bytes(streams[1]))
        return result
    finally:
        _cleanup_capture(process, job, threads)
