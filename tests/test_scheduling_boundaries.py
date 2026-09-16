"""Fail-closed scheduling hints and bounded diagnostics, not action authorization."""

from types import SimpleNamespace

import pytest
from test_ready_control_scheduling import conditions, waiting

from dronedream_agent_core import runtime_scheduling as scheduling


# 功能：
#   非布尔就绪字段不能因为非空而放行控制调度。
# 输入：
#   field：需要保持精确布尔类型的调度字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["result_ready", "perception_healthy",
                                  "features_retain_dispatch_budget"])
def test_ready_scheduler_rejects_truthy_nonbooleans(field):
    scheduler = scheduling.ReadyControlScheduler()
    assert not scheduler.eligible(**{**conditions(), field: "false"})


# 功能：
#   等待中的单调钟回退应终止本次机会，不能让短暂等待再次生效。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_wait_budget_cannot_reopen_after_clock_regression():
    scheduler = scheduling.ReadyControlScheduler()
    assert scheduler.await_pending_result(**waiting())
    assert not scheduler.await_pending_result(**waiting(now_monotonic=0.5))
    assert not scheduler.await_pending_result(**waiting(now_monotonic=1.01))


# 功能：
#   非整数深度序列不能消耗控制机会，并且失败不改变最后消费序列。
# 输入：
#   sequence：非法深度序列。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("sequence", [True, 1.5, "2", 1 << 80])
def test_consume_requires_exact_sequence(sequence):
    scheduler = scheduling.ReadyControlScheduler()
    with pytest.raises(ValueError):
        scheduler.consume(sequence)
    assert scheduler.eligible(**conditions())


# 功能：
#   非数值采样率须明确拒绝，不能触发未归类的数学类型或溢出异常。
# 输入：
#   rate：非法频率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rate", [None, "20", 1 << 20000], ids=["none", "string", "integer"])
def test_sensor_rate_validation_is_type_safe(rate):
    with pytest.raises(ValueError):
        scheduling.SensorArrivalScheduler(rate_hz=rate)
    with pytest.raises(ValueError):
        scheduling.sensor_input_maximum_rate_hz(maintenance_rate_hz=rate,
                                                 continuous_local_control=True)


# 功能：
#   损坏时间和非布尔故障状态不得唤醒传感器处理，也不抛类型异常终止循环。
# 输入：
#   change：损坏的唤醒字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [{"now_monotonic": "1"}, {"sample_monotonic": []},
    {"fault_active": 0}, {"processed_sample_monotonic": None}])
def test_arrival_hint_is_fail_closed(change):
    scheduler = scheduling.SensorArrivalScheduler(rate_hz=20.)
    hint = dict(now_monotonic=1., sample_monotonic=.99,
                processed_sample_monotonic=.9, fault_active=False)
    assert not scheduler.eligible(**{**hint, **change})


# 功能：
#   工作负载提示的每个开关都只能是精确布尔值。
# 输入：
#   field：被误传为整数的开关。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["continuous_control", "handoff_tick", "new_depth_frame",
                                  "newer_depth_waiting"])
def test_model_work_hint_rejects_nonbooleans(field):
    flags = dict(continuous_control=True, handoff_tick=False,
                 new_depth_frame=True, newer_depth_waiting=False)
    assert not scheduling.model_input_work_allowed(**{**flags, field: 0})


# 功能：
#   错误接收时钟或模拟时钟不能污染诊断计数和后续间隔基线。
# 输入：
#   row：非法接收事件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("row", [(float("nan"), 1200, 0), (.5, 1200, 0),
    (1.2, True, 0), (1.2, 1200, True), (1.2, 1200, -1)])
def test_source_interval_rejects_bad_event_without_state_change(row):
    monitor = scheduling.SimulationSourceIntervals()
    monitor.observe(1., 1000, 0)
    before = monitor.summary()
    with pytest.raises(ValueError):
        monitor.observe(*row)
    assert monitor.summary() == before


# 功能：
#   已关闭的 GC 诊断器不接收迟到回调，返回记录不与内部计数或条目共享可变引用。
# 输入：
#   monkeypatch：为手工 GC 事件设置独立时钟。
# 输出：
#   None：不返回业务数据。
def test_gc_monitor_close_freezes_owned_summary(monkeypatch):
    now = [1.]
    monkeypatch.setattr(scheduling, "time", SimpleNamespace(
        perf_counter=lambda: now[0], time=lambda: now[0]))
    monitor = scheduling.InterpreterPauseMonitor()
    scheduling.gc.callbacks.remove(monitor._callback)
    monitor._observe("start", {"generation": 2})
    now[0] = 1.01
    monitor._observe("stop", {"generation": 2, "collected": 1})
    result = monitor.close()
    result["longest_pauses"][0]["duration_ms"] = -1
    monitor._observe("start", {"generation": 2})
    now[0] = 1.02
    monitor._observe("stop", {"generation": 2, "collected": 1})
    summary = monitor.close()
    assert summary["collection_count"] == 1
    assert summary["longest_pauses"][0]["duration_ms"] > 0
