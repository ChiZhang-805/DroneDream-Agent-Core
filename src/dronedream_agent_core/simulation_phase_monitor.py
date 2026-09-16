"""Bounded phase reads for the independent simulator monitor, not actuation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .runtime_file_reader import PinnedRuntimeObjectReader
from .runtime_phase import MAXIMUM_PHASE_BYTES, phase_context, runtime_phase_context
from .runtime_phase_channel import MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS


class SimulationPhaseMonitor:
    """A missing read must not relabel an airborne vehicle as preflight.

    File reads/cache gaps have a 100 ms budget. Run-scoped phase communication
    has a separate 500 ms liveness bound; neither is an actuation lease.
    Original source/read-start times are retained, never renewed by polling.
    A phase is neither evidence of physical landing nor permission to move.
    """

    MAXIMUM_REUSE_SECONDS = .1

    # 功能：
    #   为一次独立仿真监测建立阶段缓存，验证读取及单调钟回调，不继承其他回合状态。
    # 输入：
    #   self：新监视器。
    #   reader：阶段文件读取函数。
    #   clock：本地单调秒数回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, reader=runtime_phase_context, clock=time.monotonic, channel=None):
        if not callable(reader) or not callable(clock):
            raise ValueError("LIVE_RUNTIME_PHASE_CALLBACK_INVALID")
        self._reader, self._clock = reader, clock
        self._phase: str | None = None
        self._read_started: float | None = None
        self._previous_time: float | None = None
        self._had_phase = False
        self._file_reader = None
        self._closed = False
        self._channel_path = channel
        self._channel = None
        self._terminal_phase = None

    # 功能：
    #   在执行器启动前建立专属接收端，文件兼容模式不创建通道。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def open(self) -> None:
        if self._closed:
            raise ValueError("LIVE_RUNTIME_PHASE_MONITOR_CLOSED")
        if self._channel_path is not None and self._channel is None:
            from .runtime_phase_channel import RuntimePhaseReceiver

            self._channel = RuntimePhaseReceiver(self._channel_path)

    # 功能：
    #   为真实运行固定阶段文件的目录身份；测试或专用读取回调保持原有接口。
    # 输入：
    #   path：当前运行的唯一阶段文件路径。
    # 输出：
    #   context：当前阶段上下文，不包含飞行授权。
    def _read_context(self, path: Path) -> dict:
        if self._closed:
            raise ValueError("LIVE_RUNTIME_PHASE_MONITOR_CLOSED")
        if self._channel_path is not None:
            self.open()
            return self._channel.read_context(path)
        if self._reader is not runtime_phase_context:
            return self._reader(path)
        if self._file_reader is None:
            self._file_reader = PinnedRuntimeObjectReader(path, maximum_bytes=MAXIMUM_PHASE_BYTES)
        elif path.absolute() != self._file_reader.path:
            raise ValueError("LIVE_RUNTIME_PHASE_PATH_CHANGED")
        try:
            context = phase_context(self._file_reader.read())
        except FileNotFoundError:
            # 执行器首次发布之前文件可以不存在；已有阶段失去后仍受原始复用期限约束。
            context = phase_context(None)
        return context

    # 功能：
    #   释放监视器持有的目录句柄；关闭不表示无人机已落地。
    # 输入：
    #   self：当前运行监视器。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._closed = True
        self._terminal_phase = None
        self._phase, self._read_started = None, None
        if self._file_reader is not None:
            self._file_reader.close()
        if self._channel is not None:
            self._channel.close()

    # 功能：
    #   读取并验证有限非负单调秒数，先检查类型避免布尔和巨大整数转换问题。
    # 输入：
    #   self：当前监视器。
    # 输出：
    #   value：可用于年龄比较的本地时刻。
    def _read_clock(self):
        value = self._clock()
        if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
            raise ValueError("LIVE_RUNTIME_PHASE_CLOCK_INVALID")
        return value

    # 功能：
    #   1. 将读取耗时计入原始年龄；文件复用限制为一百毫秒，阶段通道失联为五百毫秒。
    #   2. 时钟异常清除缓存，曾有阶段而后丢失时明确报错，不将飞行中状态改为起飞前。
    # 输入：
    #   self：当前独立仿真监视器。
    #   path：当前回合阶段文件路径。
    # 输出：
    #   result：有限阶段、是否复用、原始年龄及错误码，不包含运动或落地授权。
    def read(self, path: Path) -> dict[str, object]:
        # 结束声明仅终止本回合阶段观察；真正落地仍由独立原生传感器确认。
        # LANDING 不是终态，不能锁存它来隐藏仍在进行的降落失联。
        if not self._closed and self._terminal_phase is not None:
            return {"phase": self._terminal_phase, "reused": True,
                    "read_age_seconds": None, "issue": None}
        read_failed = False
        try:
            started = self._read_clock()
            try:
                context = self._read_context(path)
                if not isinstance(context, dict):
                    raise ValueError("LIVE_RUNTIME_PHASE_READER_INVALID")
            except Exception:
                context, read_failed = {}, True
            finished = self._read_clock()
            if (finished < started
                    or (self._previous_time is not None and started < self._previous_time)):
                raise ValueError("LIVE_RUNTIME_PHASE_CLOCK_INVALID")
        except Exception:
            self._phase, self._read_started = None, None
            result = {"phase": None, "reused": False, "read_age_seconds": None,
                      "issue": "LIVE_RUNTIME_PHASE_CLOCK_INVALID"}
            return result
        self._previous_time = finished
        maximum_age = (MAXIMUM_PHASE_HEARTBEAT_AGE_SECONDS if self._channel_path is not None
                       else self.MAXIMUM_REUSE_SECONDS)
        phase = phase_context({"phase": context.get("executor_phase")})["phase"]
        source_age = context.get("source_age_seconds", 0.)
        age_valid = (type(source_age) in (float, int)
                     and 0 <= source_age <= maximum_age)
        valid = (phase != "UNKNOWN" and age_valid
                 and finished - started <= self.MAXIMUM_REUSE_SECONDS
                 and finished - started + source_age <= maximum_age)
        if valid:
            self._phase, self._read_started = phase, started - source_age
            self._had_phase = True
            if self._channel is not None and phase in {"COMPLETE", "LANDED", "FAILED"}:
                self._terminal_phase = phase
        # 重复失败不刷新时间；时钟错误也不能通过恢复后的轮询重新激活旧缓存。
        age = None if self._read_started is None else finished - self._read_started
        usable = age is not None and 0 <= age <= maximum_age
        result = {"phase": self._phase if usable else None,
                "reused": usable and not valid, "read_age_seconds": age,
                "issue": ("LIVE_RUNTIME_PHASE_UNAVAILABLE"
                          if self._had_phase and not usable else
                          "LIVE_RUNTIME_PHASE_READ_FAILED" if read_failed and not usable else None)}
        return result
