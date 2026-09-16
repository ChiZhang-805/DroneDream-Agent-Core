"""Process-local Python thread handoff budget for the sensor worker.

This only bounds the interpreter's nominal thread quantum. It grants neither
OS real-time priority nor a latency guarantee; actual joint latency is measured
under rendering/inference/evidence load. Control leases remain unchanged.
"""

import gc
import heapq
import math
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

from .control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS, LOCAL_CONTROL_PERIOD_SECONDS

_baseline_retention_active = False


# 功能：
#   检查普通有限数值，不把布尔值当数值，也不在巨大整数转浮点时抛出溢出异常。
# 输入：
#   value：当前时钟或频率字段。
# 输出：
#   valid：精确数值类型且可安全参与有限浮点计算时为 True。
def _finite_number(value: object) -> bool:
    valid = (type(value) in (int, float)
             and not (type(value) is int and value.bit_length() > 64)
             and math.isfinite(value))
    return valid


# 功能：
#   1. 在专属工作进程入口暂时保留启动对象图，避免每次完整 GC 重扫固定基线。
#   2. 不关闭新对象的循环回收，不更改 GC 阈值；已有其他所有者的冻结对象不能被解冻。
#   3. 仅在创建运行流之前由单个入口使用，不在共享应用解释器或每帧内反复启用。
# 输入：
#   无。
# 输出：
#   None：上下文管理器不提供业务值。
@contextmanager
def retained_interpreter_baseline():
    global _baseline_retention_active
    if _baseline_retention_active or not gc.isenabled():
        yield
        return
    existing_baseline = gc.get_freeze_count()
    gc.collect()
    gc.freeze()
    _baseline_retention_active = True
    try:
        yield
    finally:
        _baseline_retention_active = False
        if not existing_baseline:
            gc.unfreeze()


# 功能：
#   缩短专属进程的 Python 线程切换间隔，不授予操作系统实时优先级或截止时间保证。
# 输入：
#   无。
# 输出：
#   interval：设置后的解释器切换间隔秒数。
def configure_sensor_thread_handoff() -> float:
    interval = min(sys.getswitchinterval(), .001)
    sys.setswitchinterval(interval)
    return interval


# 功能：
#   为部署策略与显式教师的连续控制共用低延迟输入调度，不为云端或仅记录任务授予控制。
# 输入：
#   provider：已选的模型调用提供者。
#   output_mode：当前真实输出模式。
#   simulation_teacher_control：是否由启动参数明确启用教师操纵。
# 输出：
#   enabled：是否启用新帧唤醒及短解释器交接周期。
def local_input_cadence_enabled(provider, output_mode, *, simulation_teacher_control: bool) -> bool:
    if type(simulation_teacher_control) is not bool:
        raise ValueError("SENSOR_CADENCE_TEACHER_FLAG_INVALID")
    enabled = (output_mode == "normalized-body-velocity" and (
        provider in ("local-policy", "simulation-training")
        or (provider is None and simulation_teacher_control)))
    return enabled


# 功能：
#   允许新传感器帧在维护周期之间有限唤醒，本地连续控制最多使用半个控制周期的间隔。
# 输入：
#   maintenance_rate_hz：周期维护频率。
#   continuous_local_control：是否采用连续本地控制。
# 输出：
#   rate：允许的输入处理上限频率，不代表实际帧率或模型请求频率。
def sensor_input_maximum_rate_hz(*, maintenance_rate_hz: float,
                                 continuous_local_control: bool) -> float:
    if (not _finite_number(maintenance_rate_hz)
            or maintenance_rate_hz <= 0 or type(continuous_local_control) is not bool):
        raise ValueError("SENSOR_ARRIVAL_RATE_INVALID")
    rate = (max(maintenance_rate_hz, 2 / LOCAL_CONTROL_PERIOD_SECONDS)
            if continuous_local_control else maintenance_rate_hz)
    if not math.isfinite(1.0 / rate):
        raise ValueError("SENSOR_ARRIVAL_RATE_INVALID")
    return rate


class ReadyControlScheduler:
    """Allow one ready-action tick between independently ingested depth frames.

    This never processes stale data as a new scan. The caller must retain the
    original geometry timestamps and revalidate current health before output.
    One use per depth sequence prevents fast model replies starving perception.
    """

    # 功能：
    #   初始化每个独立深度序列最多一次的控制交接机会。
    # 输入：
    #   self：当前调度器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self._last_priority_sequence = -1
        self._wait_sequence = -1
        self._wait_until = 0.0
        self._last_wait_clock = -math.inf

    # 功能：
    #   为已在运行的本地推理保留最多 35 毫秒交接机会，不阻塞等待 Future 或续期控制来源。
    # 输入：
    #   self：当前调度器。
    #   now_monotonic、depth_sequence：当前单调钟和独立深度序列。
    #   continuous_pending、result_ready：推理是否仍在运行、是否已有结果。
    #   perception_healthy、features_retain_dispatch_budget：健康与剩余来源预算状态。
    #   depth_processing_failed、fault_active：深度失败和故障状态。
    # 输出：
    #   should_wait：当前仍可短暂让出处理机会时为 True。
    def await_pending_result(self, *, now_monotonic: float, depth_sequence: int,
                             continuous_pending: bool, result_ready: bool,
                             perception_healthy: bool, features_retain_dispatch_budget: bool,
                             depth_processing_failed: bool, fault_active: bool) -> bool:
        if (not _finite_number(now_monotonic) or now_monotonic < 0
                or now_monotonic < self._last_wait_clock
                or type(depth_sequence) is not int or not 0 <= depth_sequence <= 2**63 - 1
                or any(type(flag) is not bool for flag in (continuous_pending, result_ready,
                    perception_healthy, features_retain_dispatch_budget,
                    depth_processing_failed, fault_active))):
            self._wait_until = 0.0
            return False
        self._last_wait_clock = now_monotonic
        eligible = (continuous_pending and not result_ready and perception_healthy
                    and features_retain_dispatch_budget and not depth_processing_failed
                    and not fault_active and depth_sequence > self._last_priority_sequence
                    and depth_sequence > 0)
        if not eligible:
            if self._wait_sequence == depth_sequence:
                self._wait_until = min(self._wait_until, now_monotonic)
            return False
        if depth_sequence > self._wait_sequence:
            self._wait_sequence = depth_sequence
            self._wait_until = now_monotonic + .035
        should_wait = depth_sequence == self._wait_sequence and now_monotonic < self._wait_until
        return should_wait

    # 功能：
    #   检查当前独立深度序列是否可给就绪结果一次交接机会，不代替下游动作验收。
    # 输入：
    #   self：当前调度器。
    #   depth_sequence：当前独立深度序列。
    #   result_ready、perception_healthy、features_retain_dispatch_budget：结果、健康和预算状态。
    #   depth_processing_failed、fault_active：处理失败和故障状态。
    # 输出：
    #   eligible：所有精确类型和调度条件同时满足时为 True。
    def eligible(self, *, depth_sequence: int, result_ready: bool,
                 perception_healthy: bool, features_retain_dispatch_budget: bool,
                 depth_processing_failed: bool, fault_active: bool) -> bool:
        if (type(depth_sequence) is not int or not 0 <= depth_sequence <= 2**63 - 1
                or any(type(flag) is not bool for flag in (result_ready, perception_healthy,
                    features_retain_dispatch_budget, depth_processing_failed, fault_active))):
            return False
        eligible = bool(depth_sequence > self._last_priority_sequence and depth_sequence > 0
                    and result_ready and perception_healthy and features_retain_dispatch_budget
                    and not depth_processing_failed and not fault_active)
        return eligible

    # 功能：
    #   消耗当前深度序列的交接机会，拒绝重复、倒退或非整数序列。
    # 输入：
    #   self：当前调度器。
    #   depth_sequence：本次已使用的独立深度序列。
    # 输出：
    #   None：不返回业务数据。
    def consume(self, depth_sequence: int) -> None:
        if (type(depth_sequence) is not int or not 0 < depth_sequence <= 2**63 - 1
                or depth_sequence <= self._last_priority_sequence):
            raise ValueError("CONTROL_PRIORITY_REQUIRES_INDEPENDENT_DEPTH_SEQUENCE")
        self._last_priority_sequence = depth_sequence


class SensorArrivalScheduler:
    """Let a new sensor sample wake an idle periodic maintenance loop.

    Maintenance ticks must not postpone a pending independent sample until the
    next timer boundary. Heavy input work still starts at most once per input
    period, with no catch-up bursts. This is a scheduling hint only: source
    alignment, validity, and action admission are the caller's responsibility.
    """

    # 功能：
    #   设置独立新帧的处理间隔，限制失败重试和追赶突发。
    # 输入：
    #   self：当前传感器唤醒调度器。
    #   rate_hz：允许的最大输入处理频率。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, rate_hz: float):
        if not _finite_number(rate_hz) or rate_hz <= 0 or not math.isfinite(1.0 / rate_hz):
            raise ValueError("SENSOR_ARRIVAL_RATE_INVALID")
        self._period = 1.0 / rate_hz
        self._last_attempt = -math.inf

    # 功能：
    #   仅为未处理、仍在原始来源期限内的新帧提供唤醒提示，不使旧样本重新变新。
    # 输入：
    #   self：当前传感器调度器。
    #   now_monotonic、sample_monotonic：当前时间与来源原始时间。
    #   processed_sample_monotonic：已处理来源时间。
    #   fault_active：当前是否存在禁止处理的故障。
    # 输出：
    #   eligible：新帧、频率与故障条件都满足时为 True。
    def eligible(self, *, now_monotonic: float, sample_monotonic: float | None,
                 processed_sample_monotonic: float, fault_active: bool) -> bool:
        if (type(fault_active) is not bool or fault_active or sample_monotonic is None
                or any(not _finite_number(x)
                       for x in (now_monotonic, sample_monotonic,
                                 processed_sample_monotonic))
                or now_monotonic < 0 or sample_monotonic < 0):
            return False
        eligible = bool(
            0 <= now_monotonic - sample_monotonic <= LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
            and sample_monotonic > processed_sample_monotonic
            and now_monotonic >= self._last_attempt + self._period
        )
        return eligible

    # 功能：
    #   记录包括失败处理在内的尝试时间，不允许时间倒退恢复处理预算。
    # 输入：
    #   self：当前传感器调度器。
    #   now_monotonic：本次真实处理尝试的单调钟。
    # 输出：
    #   None：不返回业务数据。
    def record_attempt(self, *, now_monotonic: float) -> None:
        if (not _finite_number(now_monotonic)
                or now_monotonic < 0 or now_monotonic < self._last_attempt):
            raise ValueError("SENSOR_ARRIVAL_CLOCK_INVALID")
        self._last_attempt = now_monotonic


# 功能：
#   1. 交接周期只完成已有结果，不以同一旧几何立即启动替代推理而饿死等待中的感知。
#   2. 新帧完成接入后可正常提交；没有等待图像时仍可使用通过独立时效门控的既有几何。
#   3. 此处仅决定工作顺序，不授予动作权限、不增加模型调用频率。
# 输入：
#   continuous_control：是否采用连续控制。
#   handoff_tick：当前是否是控制结果交接周期。
#   new_depth_frame：本周期是否完成新深度接入。
#   newer_depth_waiting：是否还有更新的深度帧等待处理。
# 输出：
#   allowed：当前可以开始输入准备时为 True。
def model_input_work_allowed(*, continuous_control: bool, handoff_tick: bool,
                             new_depth_frame: bool, newer_depth_waiting: bool) -> bool:
    if any(type(flag) is not bool for flag in (
        continuous_control, handoff_tick, new_depth_frame, newer_depth_waiting
    )):
        return False
    if not continuous_control:
        return True
    allowed = not handoff_tick and (new_depth_frame or not newer_depth_waiting)
    return allowed


class InterpreterPauseMonitor:
    """Bounded measurement of cyclic-GC pauses; never disables collection."""

    # 功能：
    #   注册 GC 暂停诊断回调，仅保留有限条目，不关闭或替换垃圾回收机制。
    # 输入：
    #   self：当前暂停诊断器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self._closed = False
        self._starts = {}
        self._recent = deque(maxlen=64)
        self._longest = []
        self._by_generation = {}
        self._count = 0
        self._total_ms = self._maximum_ms = 0.
        self._callback = self._observe
        gc.callbacks.append(self._callback)

    # 功能：
    #   配对合法的 GC 开始与结束事件，累计总量并有界保留最近和最长暂停。
    # 输入：
    #   self：当前诊断器。
    #   phase：原生回调的 start 或 stop 阶段。
    #   info：包含代数与已回收对象数的原生统计。
    # 输出：
    #   None：不返回业务数据。
    def _observe(self, phase, info):
        if (self._closed or type(info) is not dict or type(info.get("generation")) is not int
                or info["generation"] not in (0, 1, 2)):
            return
        generation = info["generation"]
        if phase == "start":
            now, wall = time.perf_counter(), time.time()
            if _finite_number(now) and _finite_number(wall) and 0 <= wall <= (2**63 - 1) / 1000:
                self._starts[generation] = now, int(wall * 1000)
        elif phase == "stop":
            started = self._starts.pop(generation, None)
            if started is None:
                return
            now = time.perf_counter()
            collected = info.get("collected", 0)
            if (not _finite_number(now) or type(collected) is not int
                    or not 0 <= collected <= 2**63 - 1):
                return
            elapsed = (now - started[0])*1000
            if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(
                self._total_ms + elapsed
            ):
                return
            self._count += 1
            self._total_ms += elapsed
            self._maximum_ms = max(self._maximum_ms, elapsed)
            counts = self._by_generation.setdefault(generation, {"collections": 0,
                "total_pause_ms": 0., "collected": 0})
            counts["collections"] += 1
            counts["total_pause_ms"] += elapsed
            counts["collected"] += collected
            if elapsed >= 1.:
                row = {"started_at_unix_ms": started[1],
                    "duration_ms": elapsed, "generation": generation,
                    "collected": collected}
                self._recent.append(row)
                entry = elapsed, self._count, row
                if len(self._longest) < 16:
                    heapq.heappush(self._longest, entry)
                else:
                    heapq.heappushpop(self._longest, entry)

    # 功能：
    #   撤销回调、拒绝迟到事件并返回独立统计，不将诊断摘要暴露为可修改的内部容器。
    # 输入：
    #   self：当前诊断器。
    # 输出：
    #   summary：收集次数、总暂停及有界明细。
    def close(self):
        self._closed = True
        self._starts.clear()
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
        summary = {"collection_count": self._count, "total_pause_ms": self._total_ms,
                "maximum_pause_ms": self._maximum_ms,
                "startup_frozen_objects": gc.get_freeze_count(),
                "generations": {str(key): dict(value)
                                for key, value in self._by_generation.items()},
                "longest_pauses": [dict(row) for _, _, row in sorted(self._longest, reverse=True)],
                "latest_pauses_at_least_one_ms": [dict(row) for row in self._recent]}
        return summary


class SimulationSourceIntervals:
    """Bounded diagnostics comparing receive gaps with the publisher's sim clock.

    This does not synchronize clocks or grant freshness. A slow simulation and
    a delayed callback may have different clock deltas; neither excuses a stale
    aircraft command or fills an independently unobserved flight interval.
    """

    # 功能：
    #   初始化线程安全的接收间隔诊断，不参与来源同步或控制许可。
    # 输入：
    #   self：当前间隔诊断器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self._lock = threading.Lock()
        self._previous = None
        self._count = self._clock_regressions = 0
        self._longest = []

    # 功能：
    #   分别累计接收间隔和模拟时钟变化，拒绝非法接收事件而不推进基线。
    # 输入：
    #   self：当前间隔诊断器。
    #   received_monotonic：本机单调接收时间。
    #   received_unix_ms：本机接收纪元毫秒。
    #   simulation_time_ns：发布者模拟时间，未提供时保持 None。
    # 输出：
    #   None：不返回业务数据。
    def observe(self, received_monotonic: float, received_unix_ms: int,
                simulation_time_ns: int | None) -> None:
        if (not _finite_number(received_monotonic) or received_monotonic < 0
                or type(received_unix_ms) is not int or not 0 <= received_unix_ms <= 2**63 - 1
                or simulation_time_ns is not None and (
                    type(simulation_time_ns) is not int
                    or not 0 <= simulation_time_ns <= 2**63 - 1)):
            raise ValueError("SIMULATION_SOURCE_INTERVAL_CLOCK_INVALID")
        with self._lock:
            if self._previous is not None:
                previous_received, previous_simulation = self._previous
                gap_ms = (received_monotonic - previous_received) * 1000
                if not math.isfinite(gap_ms) or gap_ms < 0:
                    raise ValueError("SIMULATION_SOURCE_INTERVAL_CLOCK_INVALID")
                simulation_delta_ms = (
                    (simulation_time_ns - previous_simulation) / 1_000_000
                    if simulation_time_ns is not None and previous_simulation is not None
                    else None
                )
                if simulation_delta_ms is not None and simulation_delta_ms < 0:
                    self._clock_regressions += 1
                entry = (gap_ms, self._count + 1, {
                    "received_at_unix_ms": received_unix_ms,
                    "receive_gap_ms": gap_ms,
                    "publisher_simulation_delta_ms": simulation_delta_ms,
                    "received_sequence": self._count + 1,
                })
                if len(self._longest) < 20:
                    heapq.heappush(self._longest, entry)
                else:
                    heapq.heappushpop(self._longest, entry)
            self._count += 1
            self._previous = received_monotonic, simulation_time_ns

    # 功能：
    #   在锁内复制有限接收间隔统计，明确模拟时钟诊断不能替代实际来源期限。
    # 输入：
    #   self：当前间隔诊断器。
    # 输出：
    #   summary：接收数、模拟时钟回退数及最长间隔明细。
    def summary(self):
        with self._lock:
            summary = {
                "received_count": self._count,
                "publisher_clock_regressions": self._clock_regressions,
                "longest_receive_intervals": [dict(row) for _, _, row in sorted(
                    self._longest, reverse=True)],
                "purpose": "diagnostics-only; no freshness or control authority",
            }
            return summary
