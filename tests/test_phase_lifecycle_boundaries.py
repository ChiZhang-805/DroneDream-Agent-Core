"""Mission labels guide lifecycle only; they never prove physical landing."""

from unittest.mock import Mock

import pytest
from test_runtime_phase_observer import manual

from dronedream_agent_core.runtime_phase import phase_context, runtime_phase_context
from dronedream_agent_core.runtime_phase_observer import RuntimePhaseObserver
from dronedream_agent_core.simulation_phase_monitor import SimulationPhaseMonitor


# 功能：
#   重复键、非有限 JSON 和非法阶段文字不能被解释为有效任务阶段。
# 输入：
#   tmp_path：测试私有目录。
#   content：无效阶段文件内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", ['{"phase":"TRACK","phase":"COMPLETE"}',
    '{"phase":"TRACK","bad":NaN}', '{"phase":"   "}', '{"phase":"TRACK\\u0000"}'])
def test_phase_wire_rejects_ambiguous_or_invalid_content(tmp_path, content):
    path = tmp_path / "phase.json"
    path.write_text(content, encoding="utf-8")
    assert runtime_phase_context(path) == phase_context(None)


# 功能：
#   阶段观察器的两个回调必须可调用，不能启动一个注定失败的读取线程。
# 输入：
#   tmp_path：测试私有目录。
#   kind：同步仿真监视器或后台阶段观察器。
#   field：被破坏的回调参数名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["monitor", "observer"])
@pytest.mark.parametrize("field", ["reader", "clock"])
def test_phase_callbacks_validated_before_start(tmp_path, kind, field):
    owner = None
    try:
        with pytest.raises(ValueError):
            if kind == "monitor":
                owner = SimulationPhaseMonitor(**{field: None})
            else:
                owner = RuntimePhaseObserver(tmp_path / "phase.json", **{field: None})
    finally:
        if isinstance(owner, RuntimePhaseObserver):
            owner.close()


# 功能：
#   错误类型和巨大时钟应返回不可用，而不触发转换异常或将布尔值当成秒数。
# 输入：
#   tmp_path：测试私有目录。
#   value：非法时钟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "10", 2**4096],
                         ids=["bool", "none", "text", "huge"])
def test_monitor_rejects_invalid_clock_without_raising(tmp_path, value):
    monitor = SimulationPhaseMonitor(reader=Mock(return_value=phase_context({"phase": "TRACK"})),
                                      clock=Mock(return_value=value))
    assert monitor.read(tmp_path / "phase.json")["issue"] == "LIVE_RUNTIME_PHASE_CLOCK_INVALID"


# 功能：
#   时钟出错必须清除复用状态，随后恢复的时钟不能使之前的飞行阶段再次变新。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_clock_failure_invalidates_cached_phase(tmp_path):
    clock = Mock(return_value=10.)
    reader = Mock(return_value=phase_context({"phase": "TRACK"}))
    monitor = SimulationPhaseMonitor(reader=reader, clock=clock)
    path = tmp_path / "phase.json"
    assert monitor.read(path)["phase"] == "TRACK"
    clock.return_value = float("nan")
    assert monitor.read(path)["phase"] is None
    reader.return_value = phase_context(None)
    clock.return_value = 10.01
    result = monitor.read(path)
    assert result["phase"] is None and result["issue"] == "LIVE_RUNTIME_PHASE_UNAVAILABLE"


# 功能：
#   读取器异常只能在原始短期限内复用，期限结束后必须报告不可用而非抛出原始异常。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_monitor_reader_failure_has_bounded_reuse(tmp_path):
    clock = Mock(return_value=10.)
    reader = Mock(return_value=phase_context({"phase": "TRACK"}))
    monitor = SimulationPhaseMonitor(reader=reader, clock=clock)
    path = tmp_path / "phase.json"
    assert monitor.read(path)["phase"] == "TRACK"
    reader.side_effect = OSError("synthetic sensitive path")
    clock.return_value = 10.01
    assert monitor.read(path)["reused"] is True
    clock.return_value = 10.11
    result = monitor.read(path)
    assert result["phase"] is None and result["issue"]
    assert "synthetic sensitive path" not in str(result)


# 功能：
#   高频内存读取遇到时钟回调异常时返回未知，不让异常中断调用方控制循环。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程替身。
# 输出：
#   None：不返回业务数据。
def test_observer_latest_contains_clock_callback_error(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    try:
        observer._sample()
        clock.side_effect = OSError("bad clock")
        assert observer.latest() == phase_context(None)
    finally:
        observer.close()


# 功能：
#   时钟差值换算为毫秒溢出时，不能把无限大诊断保存为正常观测。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：安装手动线程及可控时钟。
# 输出：
#   None：不返回业务数据。
def test_observer_rejects_overflowed_duration(tmp_path, monkeypatch):
    observer, reader, clock = manual(tmp_path, monkeypatch)
    try:
        clock.side_effect = [0., 1e308]
        with pytest.raises(ValueError):
            observer._sample()
    finally:
        result = observer.close()
    assert result["maximum_read_ms"] == 0.
