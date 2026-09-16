import pytest

from dronedream_agent_core import runtime_scheduling as module


# 功能：
#   线程交接配置只能缩短当前量子，不把已有更短量子变慢。
# 输入：
#   monkeypatch：局部解释器接口替身。
#   existing、expected：现有量子与预期量子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("existing,expected", [(0.005, 0.001), (0.0005, 0.0005)])
def test_sensor_handoff_budget_never_slows_an_existing_shorter_quantum(
    monkeypatch,
    existing,
    expected,
):
    calls = []
    monkeypatch.setattr(module.sys, "getswitchinterval", lambda: existing)
    monkeypatch.setattr(module.sys, "setswitchinterval", calls.append)
    assert module.configure_sensor_thread_handoff() == expected
    assert calls == [expected]


# 功能：
#   诊断分别记录模拟进展和本机接收间隔，不用模拟时间掩盖真实回调延迟。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_publisher_clock_distinguishes_simulation_progress_without_hiding_receive_gaps():
    from dronedream_agent_core.runtime_scheduling import SimulationSourceIntervals

    monitor = SimulationSourceIntervals()
    monitor.observe(1.0, 1000, 0)
    monitor.observe(1.5, 1500, 20_000_000)
    monitor.observe(2.0, 2000, 520_000_000)
    monitor.observe(2.1, 2100, None)
    monitor.observe(2.2, 2200, 500_000_000)
    monitor.observe(2.3, 2300, 400_000_000)
    summary = monitor.summary()
    assert summary["publisher_clock_regressions"] == 1
    assert summary["received_count"] == 6
    rows = summary["longest_receive_intervals"]
    assert rows[0]["receive_gap_ms"] == rows[1]["receive_gap_ms"] == 500
    assert {r["publisher_simulation_delta_ms"] for r in rows[:2]} == {20, 500}
    assert any(r["publisher_simulation_delta_ms"] is None for r in rows)
    for index in range(100):
        monitor.observe(3.0 + index * 0.02, 3000 + index * 20, index * 20_000_000)
    assert len(monitor.summary()["longest_receive_intervals"]) == 20
