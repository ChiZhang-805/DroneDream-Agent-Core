"""Bounded read-only process probes for a stalled simulation source.

Only explicitly registered children are inspected. No command lines, process
signals, thread priorities, clock changes or control actions are involved.
Results are diagnostic evidence, not a replacement for missing sensor samples.
"""

import math
import os
import sys
import threading
import time
from collections import Counter, deque
from copy import deepcopy
from itertools import islice
from pathlib import Path

from .native_cpu_sampling import sample_owned_thread


# 功能：
#   限制实际 proc 文件读取长度，而不是先完整读取后再截断。
# 输入：
#   path：当前进程或线程的诊断文件路径。
#   maximum_characters：本次读取的正整数字符上限，不超过四千零九十六。
# 输出：
#   prefix：最多指定字符数的 UTF-8 文本前缀。
def _read_prefix(path: Path, maximum_characters: int) -> str:
    if type(maximum_characters) is not int or not 1 <= maximum_characters <= 4096:
        raise ValueError("SOURCE_PROBE_READ_LIMIT_INVALID")
    with path.open(encoding="utf-8") as stream:
        prefix = stream.read(maximum_characters)
    return prefix


# 功能：
#   有界读取一个明确 PID 的 I/O、线程等待及调度信息；读取失败不能伪装成零活动。
# 输入：
#   pid：调用方明确选择的正整数进程标识。
#   proc_root：proc 根目录，测试可替换为隔离目录。
# 输出：
#   result：进程及有限线程诊断、截断标记和读取错误；不输出控制许可。
def sample_process_waits(pid: int, *, proc_root: Path = Path("/proc")) -> dict:
    if type(pid) is not int or not 0 < pid <= 2**31 - 1:
        raise ValueError("SOURCE_PROBE_PROCESS_ID_INVALID")
    root = proc_root / str(pid)
    result = {"pid": pid}
    try:
        result["io"] = _read_prefix(root / "io", 2048)
    except (OSError, UnicodeError) as error:
        result["io_read_error"] = type(error).__name__
    try:
        # 目录枚举也有预算；超过预算时总量未知，不能把截取数量冒充完整线程数。
        tasks = list(islice((root / "task").iterdir(), 4097))
        truncated = len(tasks) > 4096
        result["thread_count"] = None if truncated else len(tasks)
        tasks = sorted(tasks[:4096], key=lambda path: int(path.name))
        counts = Counter()
        active = []
        scheduling = []
        for task in tasks[:128]:
            try:
                stat = _read_prefix(task / "stat", 4096)
                # Linux comm can contain spaces and parentheses; fields start
                # only after its final closing parenthesis.
                comm = stat[stat.index("(") + 1 : stat.rindex(")")]
                fields = stat[stat.rindex(")") + 2 :].split()
                try:
                    wait = _read_prefix(task / "wchan", 128).strip()
                except (OSError, UnicodeError):
                    wait = "unavailable"
                state = fields[0]
                counts[(comm, state, wait)] += 1
                # Linux schedstat reports cumulative nanoseconds actually
                # running and waiting for CPU. An R state alone cannot tell
                # computation from a runnable thread starved by scheduling.
                # Keep bounded per-thread samples so adjacent probes can be
                # differenced after landing, without attaching a profiler.
                try:
                    counters = [int(value) for value in _read_prefix(
                        task / "schedstat", 256).split()]
                    if len(counters) != 3 or any(value < 0 for value in counters):
                        raise ValueError("invalid scheduling counters")
                    scheduling.append({"tid": int(task.name), "name": comm,
                        "runtime_ns": counters[0], "runqueue_wait_ns": counters[1],
                        "timeslices": counters[2]})
                except (OSError, ValueError):
                    pass  # Optional diagnostics; absence does not mean zero.
                if state in {"R", "D"}:
                    active.append(
                        {"tid": int(task.name), "name": comm, "state": state, "wait_channel": wait}
                    )
            except (OSError, ValueError, IndexError):
                continue  # A worker may exit between the independent reads.
        result["wait_counts"] = [
            {"name": name, "state": state, "wait_channel": wait, "count": count}
            for (name, state, wait), count in sorted(counts.items())
        ]
        result["active_threads"] = active
        result["thread_scheduling"] = scheduling
        result["thread_inventory_truncated"] = truncated or len(tasks) > 128
    except (OSError, ValueError) as error:
        result["read_error"] = type(error).__name__
    return result


class NativeSourceStallProbe:
    """Observe registered simulator children off-thread without controlling them."""
    # 功能：
    #   启动独立诊断线程，仅保存最近三十二条停滞样本，不在传感器回调里读取 proc。
    # 输入：
    #   self：新建的停滞探针。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self._last_source = None
        self._processes = {}
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._rows = deque(maxlen=32)
        self._count = 0
        self._failure = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="native-source-stall-probe"
        )
        self._thread.start()

    # 功能：
    #   登记调用方拥有的模拟器子进程；允许角色清单不是独立的进程所有权证明。
    # 输入：
    #   self：当前探针。
    #   name：Gazebo、PX4 或渲染副本的标准角色。
    #   pid：调用方负责保证归属的正整数进程标识。
    # 输出：
    #   None：不返回业务数据。
    def register(self, name: str, pid: int) -> None:
        if (not isinstance(name, str) or name not in {"gazebo", "px4", "render-replica"}
                or type(pid) is not int or not 0 < pid <= 2**31 - 1):
            raise ValueError("SOURCE_PROBE_REQUIRES_OWN_SIMULATOR_CHILD")
        with self._lock:
            self._processes[name] = pid

    # 功能：
    #   记录有效有序的主机接收时刻，忽略回退时刻以免制造虚假的来源停滞。
    # 输入：
    #   self：当前探针。
    #   received_monotonic：非负有限的主机单调秒数。
    # 输出：
    #   None：不返回业务数据。
    def source_received(self, received_monotonic: float) -> None:
        if (type(received_monotonic) not in (float, int)
                or not 0 <= received_monotonic <= sys.float_info.max):
            raise ValueError("SOURCE_PROBE_RECEIPT_TIME_INVALID")
        with self._lock:
            if self._last_source is not None and received_monotonic < self._last_source:
                return
            self._last_source = received_monotonic

    # 功能：
    #   捕获诊断线程终止并保留有限失败类型，防止后台已退出却报告普通成功统计。
    # 输入：
    #   self：当前探针。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        try:
            self._run_loop()
        except BaseException as error:
            self._failure = type(error).__name__

    # 功能：
    #   仅在来源间隔明显增大时检查已登记进程，可选 CPU 采样必须显式开启。
    # 输入：
    #   self：当前探针及停止事件。
    # 输出：
    #   None：不返回业务数据。
    def _run_loop(self):
        while not self._stop.wait(0.05):
            with self._lock:
                previous, children = self._last_source, dict(self._processes)
            started = time.monotonic()
            if type(started) not in (int, float) or not 0 <= started <= sys.float_info.max:
                raise ValueError("SOURCE_PROBE_CLOCK_INVALID")
            if previous is None:
                continue
            if started < previous:
                raise ValueError("SOURCE_PROBE_CLOCK_REGRESSED")
            gap_ms = (started - previous) * 1000
            if not math.isfinite(gap_ms):
                raise ValueError("SOURCE_PROBE_CLOCK_INVALID")
            if gap_ms < 120:
                continue
            row = {
                "started_at_unix_ms": int(time.time() * 1000),
                "source_gap_ms": gap_ms,
                "processes": {name: sample_process_waits(pid) for name, pid in children.items()},
            }
            if os.environ.get("DRONEDREAM_SIMULATION_CPU_SAMPLES") == "1":
                gazebo = row["processes"].get("gazebo", {})
                running = {item["tid"] for item in gazebo.get("active_threads", [])
                           if item["state"] == "R"}
                candidates = [item for item in gazebo.get("thread_scheduling", [])
                              if item["tid"] in running]
                if candidates:
                    selected = max(candidates, key=lambda item: item["runtime_ns"])
                    row["cpu_sample"] = sample_owned_thread(
                        gazebo["pid"], selected["tid"], self._stop)
            finished = time.monotonic()
            if (type(finished) not in (int, float) or not started <= finished <= sys.float_info.max
                    or not math.isfinite((finished - started) * 1000)):
                raise ValueError("SOURCE_PROBE_CLOCK_INVALID")
            row["probe_duration_ms"] = (finished - started) * 1000
            self._rows.append(row)
            self._count += 1

    # 功能：
    #   有界等待诊断线程退出并返回独立回执，超时或线程失败均不得作为飞行资格。
    # 输入：
    #   self：当前探针。
    # 输出：
    #   receipt：累计样本、最近停滞及可选读取错误，始终不授予飞行资格。
    def close(self) -> dict:
        if self._thread is threading.current_thread():
            raise RuntimeError("SOURCE_PROBE_CANNOT_JOIN_SELF")
        self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            return {"read_error": "probe-shutdown-timeout", "qualification_granted": False}
        receipt = {
            "sample_count": self._count,
            "latest_stalls": deepcopy(list(self._rows)),
            "qualification_granted": False,
            "purpose": "source-stall-diagnostics-only",
        }
        if self._failure is not None:
            receipt["read_error"] = self._failure
        return receipt
