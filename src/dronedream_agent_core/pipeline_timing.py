"""Cheap phase diagnostics; these never grant control or alter source clocks."""

import math
import sys
import time
from collections.abc import Callable

from .sensor_frame_clock import SensorFrameTime

_MAXIMUM_NS = (1 << 63) - 1


# 功能：
#   1. 根据保留的来源与接收时刻，分别计算传输、排队等待和累计处理前年龄。
#   2. 拒绝非法时间及换算溢出；只生成诊断，不重写来源、截断负延迟或授予控制权限。
# 输入：
#   frame：传感器接入时保留的完整来源时钟。
#   processing_monotonic：实际开始处理时的本地单调钟秒数。
# 输出：
#   result：原始来源标识与三类毫秒延迟组成的独立诊断字典。
def sensor_processing_timing(frame: SensorFrameTime, *, processing_monotonic: float) -> dict:
    if not isinstance(frame, SensorFrameTime):
        raise ValueError("SENSOR_PROCESSING_DIAGNOSTIC_CLOCK_INVALID")
    wall_times = frame.source_unix_ns, frame.received_unix_ns
    local_times = (frame.sample_monotonic_seconds,
                   frame.received_monotonic_seconds, processing_monotonic)
    # 先检查原始类型和可表示范围，避免 math.isfinite 对超大整数的转换自行溢出。
    if (any(type(value) is not int or not 0 <= value <= _MAXIMUM_NS for value in wall_times)
            or any(type(value) not in (int, float) or not 0 <= value <= sys.float_info.max
                   for value in local_times)
            or frame.source_unix_ns > frame.received_unix_ns
            or not frame.sample_monotonic_seconds
            <= frame.received_monotonic_seconds <= processing_monotonic):
        raise ValueError("SENSOR_PROCESSING_DIAGNOSTIC_CLOCK_INVALID")
    transport = frame.age_at_receipt_ns / 1_000_000
    wait = (processing_monotonic - frame.received_monotonic_seconds) * 1_000
    total = transport + wait
    if not math.isfinite(total):
        raise ValueError("SENSOR_PROCESSING_DIAGNOSTIC_CLOCK_INVALID")
    result = {
        "source_unix_ns": frame.source_unix_ns,
        "received_unix_ns": frame.received_unix_ns,
        "source_sequence": frame.sequence,
        "publisher_simulation_time_ns": frame.simulation_ns,
        "scene_epoch": frame.epoch,
        "scene_sha256": frame.scene_sha256,
        "clock_kind": frame.clock_kind,
        "source_to_receipt_ms": transport,
        "receipt_to_processing_ms": wait,
        "source_to_processing_ms": total,
    }
    return result


# 功能：
#   读取有符号 64 位整数纳秒，保留任意单调钟原点并拒绝浮点、布尔或超预算时刻。
# 输入：
#   clock_ns：已确认可调用的同一阶段计时函数。
# 输出：
#   tick：可安全相减、换算成毫秒的整数时刻。
def _phase_clock_ns(clock_ns: Callable[[], int]) -> int:
    tick = clock_ns()
    if type(tick) is not int or not -_MAXIMUM_NS - 1 <= tick <= _MAXIMUM_NS:
        raise ValueError("PIPELINE_TIMING_CLOCK_INVALID")
    return tick


class PhaseTimings:
    """Bounded sequential CPU/wall-time diagnostics for one processing pass."""

    # 功能：
    #   校验计时函数并固定初始纳秒基线，为一次处理流程建立有界阶段统计。
    # 输入：
    #   self：待初始化的计时器。
    #   clock_ns：同一单调纳秒时钟，默认使用 perf_counter_ns。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        if not callable(clock_ns):
            raise ValueError("PIPELINE_TIMING_CLOCK_INVALID")
        self._clock = clock_ns
        self._last = _phase_clock_ns(clock_ns)
        self._values: dict[str, float] = {}

    # 功能：
    #   结束上一阶段并记录毫秒耗时；名称、数量或时钟失败均不写入结果或推进基线。
    # 输入：
    #   self：本次处理流程的计时器。
    #   name：不超过 64 字符的非空阶段名；一次流程最多 32 个不同名称。
    # 输出：
    #   None：不返回业务数据。
    def mark(self, name: str) -> None:
        if (type(name) is not str or not 1 <= len(name) <= 64 or not name.strip()
                or name in self._values or len(self._values) >= 32):
            raise ValueError("PIPELINE_TIMING_PHASE_INVALID")
        now = _phase_clock_ns(self._clock)
        if now < self._last:
            raise ValueError("PIPELINE_TIMING_CLOCK_REGRESSED")
        # 先完成全部校验再推进基线；重试成功时仍涵盖从上一成功阶段以来的真实间隔。
        self._values[name] = (now - self._last) / 1_000_000
        self._last = now

    # 功能：
    #   返回已完成阶段的独立统计副本，不将阶段耗时解释为观测时间或控制有效期。
    # 输入：
    #   self：本次处理流程的计时器。
    # 输出：
    #   values：阶段名称到毫秒耗时的独立字典。
    def snapshot(self) -> dict[str, float]:
        values = dict(self._values)
        return values


class PhaseTimingSummary:
    """Constant-space diagnostics for completed, waiting and rejected processing passes."""

    # 功能：
    #   初始化最多 32 个阶段的累计诊断，不保存无限增长的逐帧列表。
    # 输入：
    #   self：待初始化的统计器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self._phases: dict[str, tuple[int, float, float]] = {}
        self._outcomes = {"awaiting-target": 0, "control": 0, "rejected": 0}

    # 功能：
    #   1. 累计真实已测阶段的次数、总时长及最大值，不将缺失阶段记作零延迟。
    #   2. 分开记录等待任务、控制周期和拒绝周期，不修改传感器时钟或授予飞行资格。
    # 输入：
    #   self：当前统计器。
    #   timings：一次流程的阶段计时器。
    #   outcome：该次流程实际结束于等待任务、控制或拒绝。
    # 输出：
    #   None：不返回业务数据。
    def record(self, timings: PhaseTimings, *, outcome: str) -> None:
        if (not isinstance(timings, PhaseTimings) or type(outcome) is not str
                or outcome not in self._outcomes):
            raise ValueError("PIPELINE_TIMING_SUMMARY_INPUT_INVALID")
        values = timings.snapshot()
        if len(self._phases.keys() | values.keys()) > 32:
            raise ValueError("PIPELINE_TIMING_SUMMARY_PHASE_LIMIT")
        for name, duration in values.items():
            count, total, maximum = self._phases.get(name, (0, 0., 0.))
            self._phases[name] = (count + 1, total + duration, max(maximum, duration))
        self._outcomes[outcome] += 1

    # 功能：
    #   生成独立且可序列化的累计统计，不把最大值或平均值冒充 P99，也不声称飞行通过。
    # 输入：
    #   self：当前统计器。
    # 输出：
    #   result：周期分类及各阶段实际采样次数、平均毫秒和最大毫秒。
    def snapshot(self) -> dict:
        result = {
            "schema_version": "dronedream.perception-phase-timing-summary.v1",
            "cycle_outcomes": dict(self._outcomes),
            "phase_ms": {name: {"count": count, "mean": total / count, "max": maximum}
                         for name, (count, total, maximum) in self._phases.items()},
            "qualification_granted": False,
        }
        return result
