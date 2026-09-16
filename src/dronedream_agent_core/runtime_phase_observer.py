"""Non-actuating, bounded lifecycle observation outside the control thread.

This is not a per-frame context ticket and never skips a sensor/control tick.
Task identity and sensor deadlines are still checked by the normal actor and
executor. Missing, stale or broken context is UNKNOWN, never landing proof.
Only an observed ending phase is latched until this episode's observer closes.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

from .runtime_phase import ENDING_PHASES, phase_context, runtime_phase_context


# 功能：
#   在浮点换算前检查本地单调钟类型与有限范围，避免巨大整数或布尔时钟被接受。
# 输入：
#   value：候选单调秒数。
# 输出：
#   valid：合法非负有限时间为 True。
def _valid_clock(value) -> bool:
    valid = type(value) in (float, int) and 0 <= value <= sys.float_info.max
    return valid


class RuntimePhaseObserver:
    """Cache bounded lifecycle context off-thread; ending remains latched within one episode."""
    MAXIMUM_READ_AGE_SECONDS = .05
    POLL_SECONDS = .005

    # 功能：
    #   检查回调后启动唯一后台读取器，控制线程只查询独立内存缓存。
    # 输入：
    #   self：本回合新建的阶段观察器。
    #   path：本回合阶段文件路径。
    #   reader：返回有限阶段上下文的读取函数。
    #   clock：本地单调钟秒数回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path, *, reader=runtime_phase_context, clock=time.monotonic):
        if not callable(reader) or not callable(clock):
            raise ValueError("RUNTIME_PHASE_OBSERVER_CALLBACK_INVALID")
        self._path, self._reader, self._clock = path, reader, clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._context = phase_context(None)
        self._started_at = None
        self._ending = False
        self._failed = False
        self._closed = False
        self._reads = 0
        self._maximum_read_ms = 0.
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="runtime-phase-observer")
        self._thread.start()

    # 功能：
    #   从读取开始计龄并校验返回标签，记录有限诊断后发布独立缓存，关闭后不再更新。
    # 输入：
    #   self：后台独占采样、与控制线程共享缓存的观察器。
    # 输出：
    #   None：不返回业务数据。
    def _sample(self):
        started = self._clock()
        context = self._reader(self._path)
        finished = self._clock()
        if (not _valid_clock(started) or not _valid_clock(finished)
                or finished < started):
            raise ValueError("RUNTIME_PHASE_OBSERVER_CLOCK_INVALID")
        duration_ms = (finished - started) * 1000
        if not math.isfinite(duration_ms):
            raise ValueError("RUNTIME_PHASE_OBSERVER_CLOCK_INVALID")
        # 保留读取开始时刻，挂载文件系统耗时不会被结束时刻掩盖；不缓存任何权限字段。
        context = phase_context({
            "phase": context["phase"],
            "checkpoint_id": context["checkpoint_id"],
            "enclosing_executor_state": {"phase": context["executor_phase"]},
        })
        with self._lock:
            if self._closed or self._ending:
                return
            if self._started_at is not None and started < self._started_at:
                raise ValueError("RUNTIME_PHASE_OBSERVER_CLOCK_REGRESSED")
            self._reads += 1
            self._maximum_read_ms = max(self._maximum_read_ms, duration_ms)
            self._started_at, self._context = started, context
            self._ending = context["executor_phase"] in ENDING_PHASES

    # 功能：
    #   周期采样直到本回合结束或读取失败，故障清空活跃上下文，终止标签只用于停止流程。
    # 输入：
    #   self：当前后台观察器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        try:
            while not self._stop.is_set():
                self._sample()
                # 本回合观察到的结束不可逆，不影响其他回合的新观察器。
                with self._lock:
                    if self._ending:
                        break
                self._stop.wait(self.POLL_SECONDS)
        except BaseException:
            with self._lock:
                self._failed = True
                if not self._ending:
                    self._context = phase_context(None)

    # 功能：
    #   只读内存返回有限阶段副本，关闭、过期或时钟异常时返回未知，不发起磁盘补读。
    # 输入：
    #   self：当前观察器。
    # 输出：
    #   context：当前可用上下文，或无权限的未知上下文。
    def latest(self) -> dict[str, object]:
        try:
            now = self._clock()
        except Exception:
            context = phase_context(None)
            return context
        with self._lock:
            if self._closed or not _valid_clock(now):
                return phase_context(None)
            if self._ending:
                context = dict(self._context)
                return context
            if (self._failed or self._started_at is None
                    or not 0 <= now - self._started_at <= self.MAXIMUM_READ_AGE_SECONDS):
                return phase_context(None)
            context = dict(self._context)
            return context

    # 功能：
    #   停止读取并限时等待线程退出，超时明确报错，成功回执也不代表实际飞机已经落地。
    # 输入：
    #   self：当前观察器。
    #   timeout_seconds：不超过两秒的关闭等待时间。
    # 输出：
    #   summary：读取次数、最大耗时及后台关闭状态。
    def close(self, *, timeout_seconds: float = 2.) -> dict[str, object]:
        if type(timeout_seconds) not in (float, int) or not 0 < timeout_seconds <= 2.:
            raise ValueError("RUNTIME_PHASE_OBSERVER_CLOSE_TIMEOUT_INVALID")
        if threading.current_thread() is self._thread:
            raise ValueError("RUNTIME_PHASE_OBSERVER_SELF_CLOSE_FORBIDDEN")
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("RUNTIME_PHASE_OBSERVER_DID_NOT_STOP")
        with self._lock:
            summary = {"reads": self._reads, "maximum_read_ms": self._maximum_read_ms,
                    "reader_failed": self._failed, "ending_observed": self._ending,
                    "thread_stopped": True, "maximum_context_read_age_seconds":
                    self.MAXIMUM_READ_AGE_SECONDS, "confirms_physical_landing": False}
        return summary
