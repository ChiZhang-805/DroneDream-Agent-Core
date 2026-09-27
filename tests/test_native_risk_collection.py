"""Offline checks for the explicitly non-qualifying native risk observation campaign."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "risk_collection",
    Path(__file__).resolve().parents[1] / "scripts/collect_native_risk_context.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# 功能：
#   检查已达目标高度时不发送平移或危险探针，只产生明确的小幅转向。
# 输入：
#   elapsed：跨越多个阶段边界的秒数。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("elapsed", [0, 2.999, 3, 5.999, 6, 29.99])
def test_collection_never_executes_counterfactual_translations(elapsed):
    action = module.collection_action(elapsed, 0.6, 0.4, 0.6)
    assert action.mode == "pilot-control"
    assert action.axes[:3] == [0, 0, 0]
    assert abs(action.axes[3]) == 0.2


# 功能：
#   拒绝非有限、负数与布尔时钟，防止错误时间生成运动指令。
# 输入：
#   elapsed：非法时间值。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("elapsed", [True, -1, float("nan"), float("inf"), "3"])
def test_collection_rejects_invalid_time(elapsed):
    with pytest.raises(ValueError, match="ELAPSED_INVALID"):
        module.collection_action(elapsed, 0.6, 0.4, 0.6)


# 功能：
#   检查采集失败也会关闭端口并保留失败，不能凭关闭成功授予飞行资格。
# 输入：
#   tmp_path：独立回执目录。
#   monkeypatch：替换仿真端口的测试工具，不启动飞行。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("close_failure", [False, True])
def test_collection_reset_failure_still_closes(tmp_path, monkeypatch, close_failure):
    route = tmp_path / "route.json"
    route.write_text('{"positions_m":[{"z":0.6}]}', encoding="utf-8")
    config = SimpleNamespace(initial_collection_mode="stream-imitation", episode_steps=100,
                             route=route, speed_limit_mps=0.4,
                             asset_sha256={"route": hashlib.sha256(route.read_bytes()).hexdigest()},
                             expert_role="local-navigation-policy", output_root=tmp_path,
                             model_dump=lambda **kwargs: {"fixture": True})
    env = Mock(episode_path=None)
    env.reset.side_effect = RuntimeError("sensor-not-ready")
    if close_failure:
        env.close.side_effect = RuntimeError("landing-not-confirmed")
    monkeypatch.setattr(module, "Px4GazeboTrainingEnvironment", lambda config: env)
    report = module.collect(config, 10)
    env.close.assert_called_once()
    env.finalize_stream_captures.assert_not_called()
    assert report["error"] == "RuntimeError: sensor-not-ready"
    assert report["safely_closed"] is not close_failure
    assert report["qualified_for_flight"] is False
    assert (tmp_path / "collection-report.json").is_file()


# 功能：
#   检查高度偏差只生成有限速度，过高时下降、过低时上升，不发送位置目标。
# 输入：
#   altitude、limit：当前高度和物理速度限额。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("altitude", [0.3, 0.6, 1.0, 2.0])
@pytest.mark.parametrize("limit", [0.1, 0.4, 2.0])
def test_altitude_adjustment_is_slow_and_directional(altitude, limit):
    action = module.collection_action(0, altitude, limit, 0.6)
    assert action.axes[:2] == [0, 0]
    assert abs(action.axes[2] * limit) <= 0.2
    assert action.axes[2] * (0.6 - altitude) >= 0
    assert abs(action.axes[2]) <= 0.5


# 功能：
#   限制采集时长和端口模式，禁止将此采集器当作无限时或其他专家飞行入口。
# 输入：
#   change：非法配置字段和值。
# 输出：
#   None：断言不符合时测试失败。
@pytest.mark.parametrize("change", [("initial_collection_mode", "reward-step"),
                                    ("episode_steps", 256),
                                    ("expert_role", "recovery-policy")])
def test_collection_mode_is_bounded(change):
    config = SimpleNamespace(initial_collection_mode="stream-imitation", episode_steps=100,
                             expert_role="local-navigation-policy", speed_limit_mps=0.4)
    setattr(config, *change)
    with pytest.raises(ValueError, match="COLLECTION_MODE_INVALID"):
        module.validate_campaign(config, 10)


# 功能：
#   在启动仿真前检查实际运行器的速度范围，避免宽松端口配置直到子进程启动才报错。
# 输入：
#   speed：超范围、布尔或非有限速度。
# 输出：
#   None：非法速度被明确拒绝，否则测试失败。
@pytest.mark.parametrize('speed', [0.09, 1.21, True, float('nan'), float('inf')])
def test_campaign_rejects_runner_speed_mismatch(speed):
    config = SimpleNamespace(initial_collection_mode='stream-imitation', episode_steps=100,
                             expert_role='local-navigation-policy', speed_limit_mps=speed)
    with pytest.raises(ValueError, match='RUNNER_SPEED_LIMIT_INVALID'):
        module.validate_campaign(config, 10)


# 功能：
#   检查增加方向覆盖仍不引入水平平移，非法课程在启动前拒绝。
# 输入：
#   无。
# 输出：
#   None：扫描课程限幅与拒绝断言通过。
def test_yaw_sweep_is_explicit_and_translation_free():
    for elapsed in (0, 10, 29.9):
        action = module.collection_action(elapsed, 1.5, 1.2, 1.5, "sweep")
        assert action.axes == [0, 0, 0, 0.5]
    with pytest.raises(ValueError, match="YAW_PROFILE_INVALID"):
        module.collection_action(0, 1.5, 1.2, 1.5, "unbounded")


# 功能：临时无新帧时继续等待，不能提交旧控制或重置总截止；输入：假端口；输出：新观测。
def test_bounded_wait_recovers_without_replaying_control(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.)
    env = Mock()
    fresh = object()
    env.next_stream_observation.side_effect = [
        TimeoutError("PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT"), fresh]
    report = {"observation_wait_timeouts": 0}
    assert module.await_fresh_observation(env, deadline=11., report=report) is fresh
    assert report["observation_wait_timeouts"] == 1
    deadlines = [call.kwargs['deadline'] for call in env.next_stream_observation.call_args_list]
    assert deadlines == [11., 11.]
    env.submit_stream_action.assert_not_called()


# 功能：总时限或其他错误必须终止等待，不能被通用重试吞掉；输入：截止与错误；输出：退出。
def test_bounded_wait_preserves_deadline_and_other_errors(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.)
    env = Mock()
    report = {"observation_wait_timeouts": 0}
    assert module.await_fresh_observation(env, deadline=10., report=report) is None
    env.next_stream_observation.assert_not_called()
    env.next_stream_observation.side_effect = TimeoutError("TRANSPORT_BROKEN")
    with pytest.raises(TimeoutError, match="TRANSPORT_BROKEN"):
        module.await_fresh_observation(env, deadline=11., report=report)
    env.next_stream_observation.side_effect = TimeoutError("PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT")
    with pytest.raises(TimeoutError, match="RISK_CONTEXT_WAIT_RETRY_LIMIT"):
        module.await_fresh_observation(env, deadline=11., report=report)
