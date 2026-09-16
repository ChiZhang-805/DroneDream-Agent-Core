from unittest.mock import Mock

import pytest

from dronedream_agent_core.runtime_phase import phase_context
from dronedream_agent_core.simulation_phase_monitor import SimulationPhaseMonitor


# 功能：
#   丢失阶段读取只能短暂复用原始状态，不能续期或将空中状态重命名为起飞前。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_missing_read_does_not_relabel_airborne_phase_or_renew_its_age(tmp_path):
    clock = Mock(return_value=10.)
    reader = Mock(return_value=phase_context({"phase": "TRACK"}))
    monitor = SimulationPhaseMonitor(reader=reader, clock=clock)
    path = tmp_path / "phase.json"
    assert monitor.read(path)["phase"] == "TRACK"
    reader.return_value = phase_context(None)
    clock.return_value = 10.04
    assert monitor.read(path) == {"phase": "TRACK", "reused": True,
        "read_age_seconds": pytest.approx(.04), "issue": None}
    clock.return_value = 10.101
    assert monitor.read(path)["issue"] == "LIVE_RUNTIME_PHASE_UNAVAILABLE"
    assert monitor.read(path)["phase"] is None
    reader.return_value = phase_context({"phase": "WAYPOINT_SETTLE"})
    assert monitor.read(path)["phase"] == "WAYPOINT_SETTLE"


# 功能：
#   初次没有阶段时保持未知，暂停记录则保留外层真实任务阶段。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_startup_missing_phase_is_unknown_and_holds_use_actual_enclosing_phase(tmp_path):
    path = tmp_path / "phase.json"
    monitor = SimulationPhaseMonitor()
    try:
        initial = monitor.read(path)
        assert initial["phase"] is None and initial["issue"] is None
        path.write_text('{"phase":"PERCEPTION_REFRESH_HOLD",'
                        '"enclosing_executor_state":{"phase":"TAKEOFF"}}', encoding="utf-8")
        assert monitor.read(path)["phase"] == "TAKEOFF"
    finally:
        monitor.close()
    assert monitor.read(path)["phase"] is None


# 功能：
#   读取期间时钟回退或变成非有限值时，不给出可使用的阶段。
# 输入：
#   tmp_path：测试私有目录。
#   times：注入的读取开始及结束时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("times", [[10., 9.], [float('nan'), 10.], [10., float('inf')]])
def test_invalid_clock_does_not_grant_a_phase(tmp_path, times):
    monitor = SimulationPhaseMonitor(reader=Mock(return_value=phase_context({"phase":"TRACK"})),
        clock=Mock(side_effect=times))
    assert monitor.read(tmp_path / "phase.json")["issue"] == "LIVE_RUNTIME_PHASE_CLOCK_INVALID"


# 功能：
#   磁盘读取耗时计入阶段年龄，不能用结束时刻把慢读取标成新鲜。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_read_latency_is_not_hidden_by_completion_timestamp(tmp_path):
    clock = Mock(side_effect=[10., 10., 10.02, 10.15])
    monitor = SimulationPhaseMonitor(
        reader=Mock(return_value=phase_context({"phase":"TRACK"})), clock=clock)
    assert monitor.read(tmp_path / "phase.json")["phase"] == "TRACK"
    result = monitor.read(tmp_path / "phase.json")
    assert result["phase"] is None and result["issue"] == "LIVE_RUNTIME_PHASE_UNAVAILABLE"
