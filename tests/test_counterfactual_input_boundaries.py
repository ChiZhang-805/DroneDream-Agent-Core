"""Nominal teacher input snapshots; these tests do not qualify physical flight."""

from dataclasses import replace

import pytest
from test_counterfactual_teacher import action, state, teacher

from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig, _approach


# 功能：
#   验证教师构造后不会继续共享调用方的动力学配置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teacher_owns_config():
    initial = teacher()
    config = CounterfactualConfig(acceleration_mps2=2.0, braking_acceleration_mps2=1.0)
    evaluator = type(initial)(
        [{"center_x": 8, "center_y": 0, "center_z": 1, "size_x": 1, "size_y": 1, "size_z": 1}],
        initial.envelope,
        config,
    )
    config.braking_acceleration_mps2 = 0.1
    assert evaluator.config.braking_acceleration_mps2 == 1.0


# 功能：
#   验证有限性检查保留实际速度增量，不能误把判断结果 True 当成一米每秒的变化量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_approach_preserves_requested_change_magnitude():
    assert _approach((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.02) == (0.98, 0.0, 0.0)
    assert _approach((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        _approach((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), True)


# 功能：
#   验证绕过契约验证写入的非法四元数和重复动态物体不会进入反事实积分。
# 输入：
#   fault：状态破坏类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["quaternion", "duplicate"])
def test_teacher_revalidates_state(fault):
    current = state()
    if fault == "quaternion":
        current.orientation.__dict__["w"] = float("nan")
    else:
        obstacle = DynamicObstacleObservation(
            obstacle_id="same",
            position_m=Vector3(x=8, y=8, z=8),
            velocity_mps=Vector3(x=0, y=0, z=0),
            radius_m=0.1,
            height_m=0.1,
            confidence=1.0,
            age_seconds=0.0,
        )
        current = replace(current, dynamic_obstacles=(obstacle, obstacle))
    with pytest.raises(ValueError):
        teacher().evaluate(current, action(), PilotControlLimits(1, 1, 20))


# 功能：
#   模拟几何查询期间外部改写提案和目标，核对回执仍绑定真正参与计算的输入。
# 输入：
#   monkeypatch：包装几何查询以触发修改。
# 输出：
#   None：不返回业务数据。
def test_receipt_binds_frozen_action_and_state(monkeypatch):
    evaluator, current, proposal = teacher(wall_x=8), state(), action()
    original = evaluator.geometry.clearance

    # 功能：
    #   修改原始输入后仍执行真实几何计算，用于检测回执是否晚读了调用方对象。
    # 输入：
    #   positions：教师积分得到的位置序列。
    # 输出：
    #   clearance：真实几何查询结果。
    def mutate(positions):
        proposal.axes[0] = 0.1
        current.goal.x = 6
        clearance = original(positions)
        return clearance

    monkeypatch.setattr(evaluator.geometry, "clearance", mutate)
    receipt = evaluator.evaluate(current, proposal, PilotControlLimits(1, 1, 20))
    assert receipt["action"]["axes"][0] == receipt["physical_request"][0] == 1.0
    assert receipt["state"]["goal"]["x"] == 5.0
