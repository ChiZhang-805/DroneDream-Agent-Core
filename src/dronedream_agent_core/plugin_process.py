"""Isolated MCP stdio process boundary for untrusted user tool plugins."""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from dronedream_plugin_sdk.protocol import (
    copy_json,
    decode_json,
    encode_json,
    read_frame,
    valid_request_id,
)

from .hashing import sha256_json
from .plugin_contracts import PluginResourcePolicy


class PluginProcessError(RuntimeError):
    """A plugin process failed startup, protocol validation, or a tool call."""


class PluginHostServices(Protocol):
    # 功能：
    #   定义反向调用宿主的接口，具体权限检查由宿主代理按每次操作执行。
    # 输入：
    #   method：宿主方法名称。
    #   params：有限 JSON 参数对象。
    # 输出：
    #   result：宿主提供的可序列化结果。
    def __call__(self, method: str, params: dict[str, Any]) -> object: ...


# 功能：
#   只继承启动所需路径与区域设置；强制 UTF-8，网络请求由宿主代理处理。
# 输入：
#   permissions：插件声明的权限，不直接转换为子进程网络权限。
#   resource_policy：宿主允许的网络目标与资源策略。
# 输出：
#   environment：传给子进程的独立环境变量字典。
def _safe_environment(
    *, permissions: list[str], resource_policy: PluginResourcePolicy
) -> dict[str, str]:
    # AppContainer 需要 LOCALAPPDATA 定位隔离配置；仅传目录名，文件访问仍受令牌和 ACL 限制。
    allowed = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "LOCALAPPDATA",
        "LANG",
        "LC_ALL",
    )
    environment = {key: value for key in allowed if (value := os.environ.get(key))}
    # Windows 系统语言不能改变协议编码，这些变量不赋予额外系统权限。
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:strict"
    environment["DRONEDREAM_PLUGIN_SANDBOX"] = "1"
    environment["DRONEDREAM_PLUGIN_NETWORK_BROKER_ONLY"] = "1"
    environment["DRONEDREAM_ALLOWED_NETWORK_HOSTS"] = ",".join(
        resource_policy.allowed_network_hosts
    )
    # 代理变量只能约束遵守代理的库；正式隔离依靠 AppContainer，不把环境变量当系统沙箱。
    blocked_proxy = "http://127.0.0.1:9"
    environment.update(
        {
            "HTTP_PROXY": blocked_proxy,
            "HTTPS_PROXY": blocked_proxy,
            "ALL_PROXY": blocked_proxy,
            "NO_PROXY": "",
        }
    )
    return environment


# 功能：
#   解析包内可执行文件，在要求系统隔离时组装受支持的 Windows 隔离器命令。
# 输入：
#   plugin_root：已校验的插件安装根目录。
#   command：包内程序名及不经 shell 展开的参数。
#   require_os_isolation：是否必须使用系统级隔离。
#   isolator_path：可选的受信任隔离器路径。
#   resource_policy：子进程资源限制；省略时使用默认策略。
# 输出：
#   resolved：直接执行参数，或带资源配额的隔离器参数列表。
def _isolated_command(
    *,
    plugin_root: Path,
    command: list[str],
    require_os_isolation: bool,
    isolator_path: Path | None,
    resource_policy: PluginResourcePolicy | None = None,
) -> list[str]:
    resource_policy = resource_policy or PluginResourcePolicy()
    resolved = resolve_plugin_command(plugin_root, command)
    if not require_os_isolation:
        return resolved
    if os.name == "nt":
        candidate = isolator_path
        if candidate is None:
            configured = os.environ.get("DRONEDREAM_PLUGIN_ISOLATOR", "")
            candidate = (
                Path(configured)
                if configured
                else Path(sys.executable).with_name("dronedream-plugin-isolator.exe")
            )
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise PluginProcessError("PLUGIN_OS_ISOLATOR_UNAVAILABLE")
        profile = "DroneDream.Plugin." + sha256_json(str(plugin_root.resolve()))[:32]
        resolved = [
            str(candidate),
            "--profile",
            profile,
            "--root",
            str(plugin_root.resolve()),
            "--memory-mb",
            str(resource_policy.memory_limit_mb),
            "--cpu-seconds",
            str(resource_policy.cpu_time_limit_seconds),
            "--process-limit",
            str(resource_policy.process_limit),
            "--",
            *resolved,
        ]
        return resolved
    raise PluginProcessError("PLUGIN_OS_ISOLATOR_UNAVAILABLE")


# 功能：
#   构建 POSIX 子进程资源限额回调，不提供文件系统或网络隔离；Windows 不需要该回调。
# 输入：
#   policy：地址空间、处理器时间和进程数限制。
# 输出：
#   apply_limits：POSIX 启动前回调；Windows 返回 None。
def _posix_resource_limiter(policy: PluginResourcePolicy) -> Callable[[], None] | None:
    if os.name == "nt":
        return None

    # 功能：
    #   在 fork 后、exec 前仅设置系统资源限额，不调用业务逻辑或其他应用回调。
    # 输入：
    #   无显式参数；使用外层 policy。
    # 输出：
    #   None：不返回业务数据。
    def apply_limits() -> None:
        import resource

        memory_bytes = policy.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (policy.cpu_time_limit_seconds, policy.cpu_time_limit_seconds),
        )
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (policy.process_limit, policy.process_limit))

    return apply_limits


class _WindowsJob:
    """Assign a child to a kill-on-close Windows Job with hard resource ceilings."""

    # 功能：
    #   为指定子进程创建资源受限、关闭时终止成员的 Windows Job，绑定成功后才持有句柄。
    # 输入：
    #   process_id：本次创建的子进程标识。
    #   policy：内存、处理器时间和进程数上限。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, process_id: int, policy: PluginResourcePolicy) -> None:
        self._handle: int | None = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # HANDLE 必须保持指针宽度，否则 ctypes 默认整数转换会截断 64 位句柄。
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        # 绑定 Job 只申请所需 SET_QUOTA 与 TERMINATE 权限，不能误用查询或修改信息权限。
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, process_id)
        if not job or not process:
            if job:
                kernel32.CloseHandle(job)
            if process:
                kernel32.CloseHandle(process)
            raise PluginProcessError("PLUGIN_RESOURCE_BROKER_START_FAILED")
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x8 | 0x2
        information.BasicLimitInformation.PerProcessUserTimeLimit = (
            policy.cpu_time_limit_seconds * 10_000_000
        )
        information.BasicLimitInformation.ActiveProcessLimit = policy.process_limit
        information.ProcessMemoryLimit = policy.memory_limit_mb * 1024 * 1024
        ok = kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(information), ctypes.sizeof(information)
        ) and kernel32.AssignProcessToJobObject(job, process)
        kernel32.CloseHandle(process)
        if not ok:
            kernel32.CloseHandle(job)
            raise PluginProcessError("PLUGIN_RESOURCE_LIMIT_ASSIGN_FAILED")
        self._handle = int(job)

    # 功能：
    #   关闭当前拥有的 Job 句柄并终止其剩余成员；失败时保留句柄以便重试。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if self._handle is not None:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            if not close_handle(self._handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = None


# 功能：
#   将可执行文件约束在插件根目录内，保留参数的字面值，不通过 shell 执行。
# 输入：
#   plugin_root：插件安装根目录。
#   command：包内相对程序名与参数列表。
# 输出：
#   resolved_command：包含绝对可执行路径的命令列表。
def resolve_plugin_command(plugin_root: Path, command: list[str]) -> list[str]:
    if not command:
        raise PluginProcessError("PLUGIN_COMMAND_MISSING")
    executable = Path(command[0])
    if executable.is_absolute():
        raise PluginProcessError("PLUGIN_EXECUTABLE_MUST_BE_BUNDLED")
    resolved = (plugin_root / executable).resolve()
    root = plugin_root.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise PluginProcessError("PLUGIN_EXECUTABLE_INVALID")
    resolved_command = [str(resolved), *command[1:]]
    return resolved_command


class McpStdioClient:
    """Small strict MCP client used only for tools/list and tools/call."""

    # 功能：
    #   1. 启动受配额约束的插件，建立有界通信队列并完成协议握手。
    #   2. 启动或握手任一步失败时回收已创建的进程、Job 和通信通道。
    # 输入：
    #   plugin_root：已核验的插件根目录。
    #   command：包内程序名及参数。
    #   protocol_version：握手要求的精确协议标识。
    #   startup_timeout_seconds：初始化请求秒数预算。
    #   call_timeout_seconds：普通请求秒数预算。
    #   configuration：插件配置对象；省略时为空对象。
    #   permissions：宿主已批准的权限集合。
    #   resource_policy：进程与消息资源限制。
    #   host_services：可选的反向宿主调用代理。
    #   require_os_isolation：是否强制系统隔离。
    #   isolator_path：受信任隔离器路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        plugin_root: Path,
        command: list[str],
        protocol_version: str,
        startup_timeout_seconds: float,
        call_timeout_seconds: float,
        configuration: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
        resource_policy: PluginResourcePolicy | None = None,
        host_services: PluginHostServices | None = None,
        require_os_isolation: bool = False,
        isolator_path: Path | None = None,
    ) -> None:
        for timeout in (startup_timeout_seconds, call_timeout_seconds):
            if (
                type(timeout) not in (int, float)
                or not 0 < timeout <= threading.TIMEOUT_MAX
                or not math.isfinite(timeout)
            ):
                raise PluginProcessError("PLUGIN_TIMEOUT_INVALID")
        configuration = copy_json({} if configuration is None else configuration)
        if not isinstance(configuration, dict):
            raise PluginProcessError("PLUGIN_CONFIGURATION_INVALID")
        permissions = permissions or []
        resource_policy = PluginResourcePolicy.model_validate(
            (resource_policy or PluginResourcePolicy()).model_dump()
        )
        resolved_command = _isolated_command(
            plugin_root=plugin_root,
            command=command,
            require_os_isolation=require_os_isolation,
            isolator_path=isolator_path,
            resource_policy=resource_policy,
        )
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                resolved_command,
                cwd=plugin_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                env=_safe_environment(permissions=permissions, resource_policy=resource_policy),
                creationflags=creation_flags,
                preexec_fn=_posix_resource_limiter(resource_policy),
            )
        except OSError as error:
            raise PluginProcessError("PLUGIN_PROCESS_START_FAILED") from error
        self._job = None
        self._pending: dict[str, queue.Queue[dict[str, Any] | BaseException | None]] = {}
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2_000)
        self._stderr: list[str] = []
        self._write_lock = threading.Lock()
        self._closed = False
        self._cleanup_complete = False
        self._close_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._maximum_message_bytes = resource_policy.maximum_message_bytes
        self._host_services = host_services
        self._host_slots = threading.BoundedSemaphore(4)
        self._host_request_ids: set[str | int] = set()
        self._read_failure: BaseException | None = None
        self._writes: queue.Queue[tuple[str, threading.Event, list[BaseException]]] = queue.Queue(
            maxsize=32
        )
        try:
            self._job = (
                None
                if os.name == "nt" and require_os_isolation
                else _WindowsJob(self._process.pid, resource_policy)
            )
            assert self._process.stdout is not None and self._process.stderr is not None
            for target, name in (
                (self._write_stdin, "dronedream-plugin-writer"),
                (self._read_stdout, "dronedream-plugin-reader"),
                (self._read_stderr, "dronedream-plugin-stderr"),
            ):
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                self._threads.append(thread)
            initialization = self._request(
                "initialize",
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "DroneDream-AGENT", "version": "1.0.0"},
                    "initializationOptions": {"configuration": configuration or {}},
                },
                timeout=startup_timeout_seconds,
            )
            if (
                not isinstance(initialization, dict)
                or initialization.get("protocolVersion") != protocol_version
            ):
                raise PluginProcessError("PLUGIN_INITIALIZATION_PROTOCOL_MISMATCH")
            self._notify("notifications/initialized", {})
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise
        self._call_timeout_seconds = call_timeout_seconds

    # 功能：
    #   解析有界协议帧，严格区分回复、通知和反向请求，错误时唤醒全部等待者。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = read_frame(self._process.stdout, limit=self._maximum_message_bytes)
                if line is None:
                    break
                if not line.strip():
                    continue
                value = decode_json(line, limit=self._maximum_message_bytes)
                if isinstance(value, dict) and value.get("jsonrpc") == "2.0":
                    request_id = value.get("id")
                    if "method" in value:
                        if (
                            not isinstance(value["method"], str)
                            or "result" in value
                            or "error" in value
                        ):
                            raise PluginProcessError("PLUGIN_PROTOCOL_MESSAGE_INVALID")
                        if request_id is not None:
                            if not valid_request_id(request_id):
                                raise PluginProcessError("PLUGIN_PROTOCOL_ID_INVALID")
                            self._start_host_request(value)
                            continue
                        try:
                            self._notifications.put_nowait(value)
                        except queue.Full:
                            self._broadcast_error(
                                PluginProcessError("PLUGIN_NOTIFICATION_OVERFLOW")
                            )
                            return
                    else:
                        if not valid_request_id(request_id) or ("result" in value) == (
                            "error" in value
                        ):
                            raise PluginProcessError("PLUGIN_PROTOCOL_MESSAGE_INVALID")
                        # 本客户端请求使用字符串；整数回复不能强制转成字符串以匹配其他请求。
                        with self._pending_lock:
                            target = self._pending.get(request_id)
                        if target is not None:
                            try:
                                target.put_nowait(value)
                            except queue.Full as error:
                                raise PluginProcessError("PLUGIN_DUPLICATE_RESPONSE") from error
                        # 超时后的迟到回复已不拥有请求槽，不能重新激活过期调用。
                else:
                    raise PluginProcessError("PLUGIN_PROTOCOL_MESSAGE_INVALID")
        except BaseException as error:
            self._broadcast_error(error)
        finally:
            self._broadcast_error(PluginProcessError("PLUGIN_PROTOCOL_CLOSED"))

    # 功能：
    #   约束反向调用的并发数与标识唯一性，回调未结束前不提前归还槽位。
    # 输入：
    #   value：已校验协议外层的反向请求。
    # 输出：
    #   None：不返回业务数据。
    def _start_host_request(self, value: dict[str, Any]) -> None:
        request_id = value["id"]
        with self._pending_lock:
            if request_id in self._host_request_ids:
                raise PluginProcessError("PLUGIN_DUPLICATE_HOST_REQUEST")
            if not self._host_slots.acquire(blocking=False):
                raise PluginProcessError("PLUGIN_HOST_REQUEST_OVERFLOW")
            self._host_request_ids.add(request_id)

        # 功能：
        #   执行宿主回调并在任何退出路径释放该请求标识与并发槽，管道关闭也不例外。
        # 输入：
        #   无显式参数；使用外层 value 和 request_id。
        # 输出：
        #   None：不返回业务数据。
        def serve() -> None:
            try:
                self._serve_host_request(value)
            except Exception as error:
                self._broadcast_error(error)
            finally:
                with self._pending_lock:
                    self._host_request_ids.discard(request_id)
                self._host_slots.release()

        try:
            threading.Thread(target=serve, name="dronedream-plugin-host-rpc", daemon=True).start()
        except BaseException:
            with self._pending_lock:
                self._host_request_ids.discard(request_id)
            self._host_slots.release()
            raise

    # 功能：
    #   仅向已配置宿主代理转发 DroneDream 方法，返回有界错误代码而非敏感异常详情。
    # 输入：
    #   value：包含方法、标识和参数的反向请求。
    # 输出：
    #   None：不返回业务数据。
    def _serve_host_request(self, value: dict[str, Any]) -> None:
        request_id = value.get("id")
        method = value.get("method")
        params = value.get("params", {})
        if (
            self._host_services is None
            or not isinstance(method, str)
            or not method.startswith("dronedream/")
            or not isinstance(params, dict)
        ):
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "HOST_METHOD_NOT_ALLOWED"},
                }
            )
            return
        try:
            result = self._host_services(method, params)
            self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as error:
            issue = getattr(error, "issue_code", type(error).__name__)
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32001, "message": str(issue)[:160]},
                }
            )

    # 功能：
    #   保存首个读取错误并非阻塞地唤醒所有请求，不让单个满队列拖住其他请求。
    # 输入：
    #   error：通道失败原因或无消息退出标记。
    # 输出：
    #   None：不返回业务数据。
    def _broadcast_error(self, error: BaseException | None) -> None:
        with self._pending_lock:
            if error is not None and self._read_failure is None:
                self._read_failure = error
            targets = list(self._pending.values())
        for target in targets:
            try:
                target.put_nowait(error)
            except queue.Full:
                continue

    # 功能：
    #   保存最多二十段、每段五百字符的诊断尾部，避免无换行输出占用无界内存。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        try:
            while line := self._process.stderr.readline(501):
                self._stderr.append(line.rstrip()[:500])
                if len(self._stderr) > 20:
                    self._stderr.pop(0)
        except (OSError, ValueError):
            # 关闭管道或非法编码只结束诊断读取，stderr 不参与业务结果判定。
            return

    # 功能：
    #   独占管道写线程并通知每项写入完成，关闭后为所有已排队发送者报告失败。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _write_stdin(self) -> None:
        assert self._process.stdin is not None
        while not self._closed:
            try:
                payload, completed, errors = self._writes.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._write_lock:
                    self._process.stdin.write(payload)
                    self._process.stdin.flush()
            except (OSError, ValueError) as error:
                errors.append(error)
            finally:
                completed.set()
        # 明确关闭后立即通知排队者，不让其继续等待原本的完整发送超时。
        while True:
            try:
                _, completed, errors = self._writes.get_nowait()
            except queue.Empty:
                break
            errors.append(PluginProcessError("PLUGIN_PROCESS_CLOSED"))
            completed.set()

    # 功能：
    #   1. 限制消息字节数、排队容量和管道写入等待时间。
    #   2. 写超时时终止本次子进程，避免已部分写出的过期命令之后继续执行。
    # 输入：
    #   value：需要发送的 JSON 对象。
    #   timeout：本次写入等待的秒数预算。
    # 输出：
    #   None：不返回业务数据。
    def _send(self, value: dict[str, Any], *, timeout: float = 5.0) -> None:
        if self._closed or self._process.poll() is not None:
            raise PluginProcessError("PLUGIN_PROCESS_NOT_RUNNING")
        payload = encode_json(value, limit=self._maximum_message_bytes - 1) + "\n"
        completed = threading.Event()
        errors: list[BaseException] = []
        try:
            self._writes.put_nowait((payload, completed, errors))
        except queue.Full as error:
            raise PluginProcessError("PLUGIN_PROTOCOL_WRITE_OVERFLOW") from error
        if not completed.wait(timeout):
            failure = PluginProcessError("PLUGIN_PROTOCOL_WRITE_TIMEOUT")
            self._broadcast_error(failure)
            with suppress(OSError):
                self._process.kill()
            raise failure
        if errors:
            raise PluginProcessError("PLUGIN_PROTOCOL_WRITE_FAILED") from errors[0]

    # 功能：
    #   为一次请求持有独立关联槽，以同一截止时间约束发送和等待，结束后丢弃迟到回复。
    # 输入：
    #   method：协议方法名称。
    #   params：方法参数对象。
    #   timeout：发送与回复等待共用的秒数预算。
    # 输出：
    #   result：经协议外层检查的返回值，业务结构仍由调用入口校验。
    def _request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        request_id = f"request-{uuid4().hex}"
        messages: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue(maxsize=1)
        with self._pending_lock:
            if self._read_failure is not None:
                raise PluginProcessError("PLUGIN_PROTOCOL_READ_FAILED") from self._read_failure
            self._pending[request_id] = messages
        try:
            self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                timeout=min(timeout, 5.0),
            )
            try:
                message = messages.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as error:
                self.cancel(request_id)
                raise PluginProcessError(f"PLUGIN_TIMEOUT:{method}") from error
            if message is None:
                # 子进程 stderr 不可信且可能包含敏感数据，不拼接进公开错误消息。
                raise PluginProcessError("PLUGIN_PROCESS_EXITED")
            if isinstance(message, BaseException):
                raise PluginProcessError("PLUGIN_PROTOCOL_READ_FAILED") from message
            if message.get("jsonrpc") != "2.0":
                raise PluginProcessError("PLUGIN_PROTOCOL_VERSION_INVALID")
            if "error" in message:
                raise PluginProcessError("PLUGIN_RPC_ERROR")
            if "result" not in message:
                raise PluginProcessError("PLUGIN_RPC_RESULT_MISSING")
            result = message["result"]
            return result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    # 功能：
    #   发送不需要应答的协议通知，不创建等待回复的关联槽。
    # 输入：
    #   method：通知名称。
    #   params：通知参数。
    # 输出：
    #   None：不返回业务数据。
    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # 功能：
    #   请求插件协作取消指定调用，不将通知发送成功当作执行已经停止的证据。
    # 输入：
    #   request_id：原请求关联标识。
    # 输出：
    #   None：不返回业务数据。
    def cancel(self, request_id: str) -> None:
        self._notify("notifications/cancelled", {"requestId": request_id})

    # 功能：
    #   单独探测进程协议响应，不要求正在运行的慢工具先返回。
    # 输入：
    #   无。
    # 输出：
    #   responsive：回复是否为协议预期的对象结构。
    def ping(self) -> bool:
        result = self._request("ping", {}, timeout=min(self._call_timeout_seconds, 5.0))
        responsive = isinstance(result, dict)
        return responsive

    # 功能：
    #   非阻塞地取走当前积压的有界通知，每条通知只交付一次。
    # 输入：
    #   无。
    # 输出：
    #   values：本次取出的通知列表。
    def drain_notifications(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        while True:
            try:
                values.append(self._notifications.get_nowait())
            except queue.Empty:
                return values

    # 功能：
    #   读取并检查工具目录的容器类型，工具与安装清单的绑定仍由调用方核验。
    # 输入：
    #   无。
    # 输出：
    #   tools：插件报告的工具描述列表。
    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {}, timeout=self._call_timeout_seconds)
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise PluginProcessError("PLUGIN_TOOL_CATALOG_INVALID")
        tools = result["tools"]
        if any(not isinstance(item, dict) for item in tools):
            raise PluginProcessError("PLUGIN_TOOL_CATALOG_INVALID")
        return tools

    # 功能：
    #   调用工具并优先取结构化结果，兼容文本结果时仅接受有限 JSON 对象。
    # 输入：
    #   name：插件内的工具名称。
    #   arguments：工具参数对象。
    # 输出：
    #   structured：原生结构化结果，或从兼容文本中解析得到的结果对象。
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=self._call_timeout_seconds,
        )
        if not isinstance(result, dict):
            raise PluginProcessError("PLUGIN_TOOL_RESULT_INVALID")
        if result.get("isError"):
            raise PluginProcessError("PLUGIN_TOOL_REPORTED_ERROR")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    parsed = decode_json(text, limit=self._maximum_message_bytes)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    structured = parsed
                    return structured
        raise PluginProcessError("PLUGIN_TOOL_STRUCTURED_RESULT_MISSING")

    # 功能：
    #   读取资源描述目录，不将列出 URI 等同于拥有本地文件访问权限。
    # 输入：
    #   无。
    # 输出：
    #   resources：已检查容器类型的资源描述列表。
    def list_resources(self) -> list[dict[str, Any]]:
        result = self._request("resources/list", {}, timeout=self._call_timeout_seconds)
        if not isinstance(result, dict) or not isinstance(result.get("resources"), list):
            raise PluginProcessError("PLUGIN_RESOURCE_CATALOG_INVALID")
        resources = result["resources"]
        if any(not isinstance(item, dict) for item in resources):
            raise PluginProcessError("PLUGIN_RESOURCE_CATALOG_INVALID")
        return resources

    # 功能：
    #   经插件协议读取资源内容，不直接在宿主打开插件给出的 URI。
    # 输入：
    #   uri：插件资源标识。
    # 输出：
    #   contents：已检查容器类型的内容项列表。
    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        result = self._request("resources/read", {"uri": uri}, timeout=self._call_timeout_seconds)
        if not isinstance(result, dict) or not isinstance(result.get("contents"), list):
            raise PluginProcessError("PLUGIN_RESOURCE_CONTENT_INVALID")
        contents = result["contents"]
        if any(not isinstance(item, dict) for item in contents):
            raise PluginProcessError("PLUGIN_RESOURCE_CONTENT_INVALID")
        return contents

    # 功能：
    #   请求插件发布资源变化通知，不另建资源轮询线程。
    # 输入：
    #   uri：需要订阅的插件资源标识。
    # 输出：
    #   None：不返回业务数据。
    def subscribe_resource(self, uri: str) -> None:
        self._request("resources/subscribe", {"uri": uri}, timeout=self._call_timeout_seconds)

    # 功能：
    #   取消单个资源订阅，保留共享进程及其他订阅。
    # 输入：
    #   uri：需要取消订阅的插件资源标识。
    # 输出：
    #   None：不返回业务数据。
    def unsubscribe_resource(self, uri: str) -> None:
        self._request("resources/unsubscribe", {"uri": uri}, timeout=self._call_timeout_seconds)

    # 功能：
    #   1. 禁止新调用并唤醒等待者，逐项回收本次进程、Job、通信线程和管道。
    #   2. 单项失败仍尝试其他清理；线程未退出时不强行关闭可能持锁的管道，允许重试。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        with self._close_lock:
            if self._cleanup_complete:
                return
            self._closed = True
            self._broadcast_error(PluginProcessError("PLUGIN_PROCESS_CLOSED"))
            errors: list[Exception] = []
            try:
                if self._process.poll() is None:
                    self._process.terminate()
            except OSError as error:
                errors.append(error)
            if self._job is not None:
                try:
                    self._job.close()
                except Exception as error:
                    errors.append(error)
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                    self._process.wait(timeout=2)
                except Exception as error:
                    errors.append(error)
            except Exception as error:
                errors.append(error)
            deadline = time.monotonic() + 2.0
            for thread in self._threads:
                if thread is threading.current_thread():
                    continue
                try:
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                except Exception as error:
                    errors.append(error)
            if any(thread.is_alive() for thread in self._threads):
                # 主线程 close 一个正在 read/write 的缓冲流，可能再次无限等待其内部锁。
                errors.append(TimeoutError("PLUGIN_CHANNEL_CLEANUP_TIMEOUT"))
            else:
                for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except BrokenPipeError:
                            pass
                        except Exception as error:
                            errors.append(error)
            self._cleanup_complete = not errors
            if errors:
                raise PluginProcessError("PLUGIN_PROCESS_CLEANUP_FAILED") from ExceptionGroup(
                    "Plugin process cleanup failures", errors
                )

    # 功能：
    #   将已初始化的客户端交给上下文使用，不再次启动插件。
    # 输入：
    #   无。
    # 输出：
    #   self：当前客户端。
    def __enter__(self) -> McpStdioClient:
        return self

    # 功能：
    #   上下文退出时回收客户端，不吞掉业务异常。
    # 输入：
    #   _type：上下文异常类型。
    #   _value：上下文异常对象。
    #   _traceback：上下文异常回溯。
    # 输出：
    #   None：不返回业务数据。
    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class McpSessionFailure:
    """Bind asynchronous failure to the exact package/configuration that stopped responding."""

    plugin_id: str
    package_sha256: str
    configuration_sha256: str
    issue_code: str


class McpSessionPool:
    """Persistent, concurrent-safe MCP sessions keyed by immutable runtime identity."""

    # 功能：
    #   创建按运行身份隔离的会话池，启动有界周期的后台存活探测。
    # 输入：
    #   heartbeat_interval_seconds：存活探测周期，必须是系统等待接口支持的有限正数。
    #   on_unhealthy：精确会话失效后的可选报告回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        heartbeat_interval_seconds: float = 15.0,
        on_unhealthy: Callable[[McpSessionFailure], None] | None = None,
    ) -> None:
        if (
            type(heartbeat_interval_seconds) not in (int, float)
            or not 0 < heartbeat_interval_seconds <= threading.TIMEOUT_MAX
            or not math.isfinite(heartbeat_interval_seconds)
        ):
            raise ValueError("PLUGIN_HEARTBEAT_INTERVAL_INVALID")
        self._lock = threading.RLock()
        self._sessions: dict[tuple[str, str, str, str], McpStdioClient] = {}
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._on_unhealthy = on_unhealthy
        self._stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="dronedream-plugin-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    # 功能：
    #   检查会话响应并只淘汰精确失败实例，不误删并发安装的新实例；报告失败不终止其他检查。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval_seconds):
            with self._lock:
                sessions = list(self._sessions.items())
            for key, client in sessions:
                if self._stop.is_set():
                    break
                issue = ""
                try:
                    if not client.ping():
                        issue = "PLUGIN_HEARTBEAT_INVALID"
                except BaseException as error:
                    issue = f"PLUGIN_HEARTBEAT_FAILED:{type(error).__name__}"
                if not issue:
                    continue
                removed = False
                with self._lock:
                    if self._sessions.get(key) is client:
                        self._sessions.pop(key, None)
                        removed = True
                if not removed:
                    continue
                close = getattr(client, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
                if self._on_unhealthy is not None:
                    try:
                        self._on_unhealthy(McpSessionFailure(key[0], key[1], key[2], issue))
                    except Exception:
                        # 单个遥测回调失败不能使所有其他插件的存活探测停止。
                        continue

    # 功能：
    #   仅复用包内容、位置、配置、宿主权限域和运行策略完全相同的会话，关闭后拒绝新建。
    # 输入：
    #   plugin_id：插件标识。
    #   package_sha256：实际安装包摘要。
    #   plugin_root：插件安装目录。
    #   command：包内启动命令。
    #   protocol_version：握手协议标识。
    #   startup_timeout_seconds：初始化预算。
    #   call_timeout_seconds：调用预算。
    #   configuration：创建前冻结的配置对象。
    #   permissions：获准权限列表。
    #   resource_policy：创建前重新校验的运行配额。
    #   host_scope_id：宿主资源授权作用域标识。
    #   host_services：该作用域对应的宿主代理。
    #   require_os_isolation：是否要求系统隔离。
    #   isolator_path：可选隔离器路径。
    #   client_factory：创建协议客户端的工厂。
    # 输出：
    #   client：精确身份匹配的现有客户端或刚创建的客户端。
    def get(
        self,
        *,
        plugin_id: str,
        package_sha256: str,
        plugin_root: Path,
        command: list[str],
        protocol_version: str,
        startup_timeout_seconds: float,
        call_timeout_seconds: float,
        configuration: dict[str, Any],
        permissions: list[str],
        resource_policy: PluginResourcePolicy,
        host_scope_id: str = "none",
        host_services: PluginHostServices | None = None,
        require_os_isolation: bool = False,
        isolator_path: Path | None = None,
        client_factory: Callable[..., McpStdioClient] = McpStdioClient,
    ) -> McpStdioClient:
        configuration = copy_json(configuration)
        command = list(command)
        permissions = list(permissions)
        resource_policy = PluginResourcePolicy.model_validate(resource_policy.model_dump())
        identity = sha256_json(
            {
                "package_sha256": package_sha256,
                "plugin_root": str(plugin_root.resolve()),
                "startup_timeout_seconds": startup_timeout_seconds,
                "call_timeout_seconds": call_timeout_seconds,
                "command": command,
                "protocol_version": protocol_version,
                "configuration": configuration,
                "permissions": permissions,
                "resource_policy": resource_policy.model_dump(mode="json"),
                "host_scope_id": host_scope_id,
                "require_os_isolation": require_os_isolation,
                "isolator_path": str(isolator_path) if isolator_path else None,
            }
        )
        key = (plugin_id, package_sha256, sha256_json(configuration), identity)
        with self._lock:
            if self._stop.is_set():
                raise PluginProcessError("PLUGIN_SESSION_POOL_CLOSED")
            client = self._sessions.get(key)
            if client is not None:
                return client
            client = client_factory(
                plugin_root=plugin_root,
                command=command,
                protocol_version=protocol_version,
                startup_timeout_seconds=startup_timeout_seconds,
                call_timeout_seconds=call_timeout_seconds,
                configuration=configuration,
                permissions=permissions,
                resource_policy=resource_policy,
                host_services=host_services,
                require_os_isolation=require_os_isolation,
                isolator_path=isolator_path,
            )
            # close 在等待池锁前就设置停止标记，不能把握手期间创建的会话继续发布出去。
            if self._stop.is_set():
                client.close()
                raise PluginProcessError("PLUGIN_SESSION_POOL_CLOSED")
            self._sessions[key] = client
            return client

    # 功能：
    #   在临时宿主资源消失前卸下并关闭指定客户端，不按插件名称误关其他作用域。
    # 输入：
    #   client：调用方需要归还的精确客户端实例。
    # 输出：
    #   None：不返回业务数据。
    def release(self, client: McpStdioClient) -> None:
        with self._lock:
            keys = [key for key, owned in self._sessions.items() if owned is client]
            for key in keys:
                self._sessions.pop(key)
        if keys:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    # 功能：
    #   在池锁内卸下指定插件的全部会话，然后逐项关闭，单个失败不跳过其他实例。
    # 输入：
    #   plugin_id：需要失效的插件标识。
    # 输出：
    #   None：不返回业务数据。
    def invalidate(self, plugin_id: str) -> None:
        with self._lock:
            keys = [key for key in self._sessions if key[0] == plugin_id]
            clients = [self._sessions.pop(key) for key in keys]
        self._close_clients(clients)

    # 功能：
    #   尝试关闭所有已卸下的会话，最后统一报告错误，不因第一个错误遗留后续实例。
    # 输入：
    #   clients：本次需要回收的客户端列表。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _close_clients(clients: list[McpStdioClient]) -> None:
        errors: list[Exception] = []
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise PluginProcessError("PLUGIN_SESSION_CLEANUP_FAILED") from ExceptionGroup(
                "Plugin session cleanup failures", errors
            )

    # 功能：
    #   先禁止新建会话，再卸下全部实例；即使关闭实例失败，也尝试等待探测线程结束。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._stop.set()
        with self._lock:
            clients = list(self._sessions.values())
            self._sessions.clear()
        try:
            self._close_clients(clients)
        finally:
            if (
                self._heartbeat_thread.is_alive()
                and threading.current_thread() is not self._heartbeat_thread
            ):
                self._heartbeat_thread.join(timeout=1.0)
