from dataclasses import replace

import pytest

from dronedream_agent_core.pipeline_timing import (
    PhaseTimings,
    PhaseTimingSummary,
    sensor_processing_timing,
)
from dronedream_agent_core.sensor_frame_clock import SensorFrameTime
from dronedream_plugin_sdk.protocol import encode_json


# 功能：
#   对照等待、控制和拒绝周期的聚合结果，确保缺失阶段不参与均值且结果不共享可变对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_summary_counts_waiting_and_rejected_passes_without_zero_filling():
    summary = PhaseTimingSummary()
    for outcome, elapsed in [("awaiting-target", 1), ("control", 3), ("rejected", 8)]:
        ticks = iter([0, elapsed * 1_000_000])
        timings = PhaseTimings(ticks.__next__)
        timings.mark("fusion" if outcome != "rejected" else "cycle_tail")
        summary.record(timings, outcome=outcome)
    result = summary.snapshot()
    assert result["phase_ms"] == {
        "fusion": {"count": 2, "mean": 2., "max": 3.},
        "cycle_tail": {"count": 1, "mean": 8., "max": 8.},
    }
    assert result["cycle_outcomes"] == {"awaiting-target": 1, "control": 1, "rejected": 1}
    encode_json(result)
    result["cycle_outcomes"]["control"] = 100
    result["phase_ms"]["fusion"]["max"] = 0
    assert summary.snapshot()["phase_ms"]["fusion"]["max"] == 3.
    assert summary.snapshot()["cycle_outcomes"]["control"] == 1


# 功能：
#   验证跨周期的阶段名称也有固定预算，拒绝后不会留下部分计数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_summary_phase_budget_is_atomic_across_passes():
    summary = PhaseTimingSummary()
    for index in range(32):
        ticks = iter([0, 1_000_000])
        timings = PhaseTimings(ticks.__next__)
        timings.mark(str(index))
        summary.record(timings, outcome="rejected")
    before = summary.snapshot()
    ticks = iter([0, 1_000_000])
    extra = PhaseTimings(lambda: next(ticks))
    extra.mark("extra")
    with pytest.raises(ValueError, match="SUMMARY_PHASE_LIMIT"):
        summary.record(extra, outcome="control")
    assert summary.snapshot() == before


# 功能：
#   确认非法周期分类被稳定拒绝且不修改累计统计。
# 输入：
#   outcome：错误类型或未知的周期分类。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("outcome", [None, [], {}, True, "qualified"])
def test_summary_invalid_outcome_does_not_change_counts(outcome):
    summary = PhaseTimingSummary()
    before = summary.snapshot()
    with pytest.raises(ValueError, match="SUMMARY_INPUT_INVALID"):
        summary.record(PhaseTimings(), outcome=outcome)
    assert summary.snapshot() == before


# 功能：
#   验证阶段差值按毫秒记录、快照不会随后续写入变化，并拒绝重复名称。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_timings_are_detached_monotonic_and_not_source_time():
    ticks = iter([500, 1000500, 3000500])
    clock = PhaseTimings(lambda: next(ticks))
    clock.mark("native")
    first = clock.snapshot()
    clock.mark("fusion")
    assert first == {"native": 1.}
    assert clock.snapshot() == {"native": 1., "fusion": 2.}
    with pytest.raises(ValueError, match="PHASE_INVALID"):
        clock.mark("native")


# 功能：
#   确认时钟回退明确失败，不裁成零耗时，也不使下一次采样从错误基线开始。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_regressed_monotonic_clock_is_not_hidden():
    ticks = iter([10, 9, 1_000_010])
    clock = PhaseTimings(lambda: next(ticks))
    with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
        clock.mark("native")
    assert clock.snapshot() == {}
    clock.mark("native")
    assert clock.snapshot() == {"native": 1.}


# 功能：
#   对照独立传输与排队时间，确认诊断保留来源标识且不改写原始时钟。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_transport_and_worker_wait_remain_distinct_and_keep_source_identity():
    frame = SensorFrameTime(source_unix_ns=1_000_000_000, received_unix_ns=1_084_000_000,
        sample_monotonic_seconds=10., received_monotonic_seconds=10.084,
        clock_kind="scene-source", sequence=805, simulation_ns=123456789,
        epoch="a" * 64, scene_sha256="b" * 64)
    result = sensor_processing_timing(frame, processing_monotonic=10.127)
    assert result["source_to_receipt_ms"] == pytest.approx(84.)
    assert result["receipt_to_processing_ms"] == pytest.approx(43.)
    assert result["source_to_processing_ms"] == pytest.approx(127.)
    assert result["source_sequence"] == 805 and result["source_unix_ns"] == 1_000_000_000
    assert frame.source_unix_ns == 1_000_000_000
    assert result["publisher_simulation_time_ns"] == 123456789
    assert result["scene_epoch"] == "a" * 64 and result["scene_sha256"] == "b" * 64
    for now in (10., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="DIAGNOSTIC_CLOCK_INVALID"):
            sensor_processing_timing(frame, processing_monotonic=now)


# 功能：
#   确认处理时刻的错误类型、超大整数和毫秒换算溢出不会成为可发布的诊断。
# 输入：
#   now：需要拒绝的处理单调钟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now", [True, None, "11", 10 ** 400, 1e308])
def test_invalid_processing_clock_is_rejected_consistently(now):
    frame = SensorFrameTime(1, 2, 0., 1., "host-receipt")
    with pytest.raises(ValueError, match="SENSOR_PROCESSING_DIAGNOSTIC_CLOCK_INVALID"):
        sensor_processing_timing(frame, processing_monotonic=now)


# 功能：
#   验证诊断边界拒绝畸形来源钟、负传输年龄及不可能的本地接收顺序，不修改原始记录。
# 输入：
#   changes：在正常时钟夹具上逐项注入的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"source_unix_ns": True}, {"source_unix_ns": -1},
    {"source_unix_ns": 3}, {"received_unix_ns": 10 ** 400},
    {"received_unix_ns": "2"}, {"received_unix_ns": 2.},
    {"received_monotonic_seconds": float("nan")},
    {"received_monotonic_seconds": None}, {"received_monotonic_seconds": -1.},
    {"sample_monotonic_seconds": True}, {"sample_monotonic_seconds": 2.},
])
def test_malformed_frame_clock_does_not_produce_diagnostics(changes):
    frame = replace(SensorFrameTime(1, 2, 0., 1., "host-receipt"), **changes)
    with pytest.raises(ValueError, match="SENSOR_PROCESSING_DIAGNOSTIC_CLOCK_INVALID"):
        sensor_processing_timing(frame, processing_monotonic=2.)


# 功能：
#   确认错误的时钟函数及非整数纳秒在初始化时拒绝，不延后成误导性的阶段耗时。
# 输入：
#   clock_ns：非法时钟函数或返回错误类型的可调用对象。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("clock_ns", [None, 5, lambda: True, lambda: float("nan"),
                                    lambda: 1.5, lambda: 10 ** 400])
def test_phase_clock_requires_callable_integer_nanoseconds(clock_ns):
    with pytest.raises(ValueError, match="PIPELINE_TIMING_CLOCK_INVALID"):
        PhaseTimings(clock_ns)


# 功能：
#   证明非法阶段时刻不写入结果也不推进基线，随后的合法采样仍覆盖完整间隔。
# 输入：
#   bad_tick：非法中间时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad_tick", [True, None, float("nan"), 1.5, 10 ** 400])
def test_invalid_mark_does_not_advance_phase_state(bad_tick):
    ticks = iter([0, bad_tick, 2_000_000])
    clock = PhaseTimings(lambda: next(ticks))
    with pytest.raises(ValueError, match="PIPELINE_TIMING_CLOCK_INVALID"):
        clock.mark("native")
    assert clock.snapshot() == {}
    clock.mark("native")
    assert clock.snapshot() == {"native": 2.}


# 功能：
#   拒绝非字符串、过长及空白阶段名，且在拒绝名称时不消耗下一次时钟采样。
# 输入：
#   name：非法阶段名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name", [True, 5, [], {}, " ", "x" * 65])
def test_phase_name_budget_is_checked_before_clock_read(name):
    ticks = iter([0, 1_000_000])
    clock = PhaseTimings(lambda: next(ticks))
    with pytest.raises(ValueError, match="PIPELINE_TIMING_PHASE_INVALID"):
        clock.mark(name)
    clock.mark("native")
    assert clock.snapshot() == {"native": 1.}


# 功能：
#   保留合法的任意单调钟原点和零耗时阶段，并验证阶段数预算失败不改动快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_signed_clock_origin_zero_intervals_and_phase_capacity():
    clock = PhaseTimings(lambda: -500)
    for index in range(32):
        clock.mark(f"phase-{index}")
    before = clock.snapshot()
    assert len(before) == 32 and set(before.values()) == {0.}
    with pytest.raises(ValueError, match="PIPELINE_TIMING_PHASE_INVALID"):
        clock.mark("excess")
    assert clock.snapshot() == before
