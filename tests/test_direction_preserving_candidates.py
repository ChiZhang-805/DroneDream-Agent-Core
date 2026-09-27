"""Keep slower three-axis intentions available to the full safety search."""

import math

import pytest

from dronedream_agent_core.contracts import LocalPlannerRequest, Vector3
from dronedream_agent_core.dynamic_safety import _candidate_velocities, _reachable_velocity, predictive_safety_decision


# 功能：
#   构造高余量的合成运动请求，将测试限制在候选方向而非动力学饱和。
# 输入：
#   velocity：要保持方向的 ENU 速度三元组。
#   explicit：是否使用本地模型的显式速度提议。
# 输出：
#   request：用于候选生成与完整安全预测的请求。
def request_for(velocity, explicit):
    request = LocalPlannerRequest(current_position_m=Vector3(x=0., y=0., z=5.),
        current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=velocity[0], y=velocity[1], z=5. + velocity[2]),
        requested_velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]) if explicit else None,
        vehicle_radius_m=.2, vehicle_height_m=.3, max_speed_mps=1., max_acceleration_mps2=10.,
        max_jerk_mps3=100., required_clearance_m=.3, prediction_horizon_seconds=2.,
        prediction_step_seconds=.2)
    return request


# 功能：
#   验证爬升、下降、斜飞和水平动作均保留原方向的分级减速，不强迫增加横移。
# 输入：
#   velocity：参数化的三维意图。
#   explicit：模型速度或路线方向两类入口。
# 输出：
#   None：通过断言验证方向保持候选。
@pytest.mark.parametrize('velocity', [(0., 0., .8), (0., 0., -.8), (.3, .4, .5), (.6, .2, 0.)])
@pytest.mark.parametrize('explicit', [True, False])
def test_candidates_include_proportional_slowing(velocity, explicit):
    candidates = _candidate_velocities(request_for(velocity, explicit), include_level_flight=True)
    for ratio in (1., .7, .4):
        expected = tuple(ratio * value for value in velocity)
        assert any(math.dist(candidate, expected) < 1e-9 for candidate in candidates)


# 功能：
#   复现竖直窄通道中全速过快但慢速可行的情况，完整预测不得只能制动或横移。
# 输入：
#   explicit：两种控制意图来源。
# 输出：
#   None：所选动作仍通过真实几何预测和附加候选门禁。
@pytest.mark.parametrize('explicit', [True, False])
def test_slow_climb_survives_full_safety_search(explicit):
    request = request_for((0., 0., .8), explicit)
    # 功能：
    #   模拟窄通道的附加横移否决，碰撞预测和最终复核仍由生产实现执行。
    # 输入：
    #   _request、velocity、_path：本次生产预测请求、候选速度和预测路径。
    # 输出：
    #   issues：横移时的覆盖拒绝原因。
    def corridor_check(_request, velocity, _path):
        issues = [] if abs(velocity[0]) < 1e-9 and abs(velocity[1]) < 1e-9 else ['SIDEWAYS_NOT_OBSERVED']
        return issues

    decision = predictive_safety_decision(request, [], motion_check=corridor_check,
        candidate_check=lambda velocity, clearance, yaw: velocity[2] <= .6)
    assert decision.action != 'hold'
    assert decision.selected_velocity_mps.x == pytest.approx(0.)
    assert decision.selected_velocity_mps.y == pytest.approx(0.)
    assert decision.selected_velocity_mps.z == pytest.approx(.56)


# 功能：
#   新的减速候选仍保留当前惯性并执行同一加速度和加加速度投影，不能直接覆盖成期望速度。
# 输入：
#   explicit：模型或路线意图入口。
# 输出：
#   None：通过逐候选物理增量断言验证。
@pytest.mark.parametrize('explicit', [True, False])
def test_direction_candidates_keep_inertia_and_jerk_limits(explicit):
    request = request_for((0., 0., .8), explicit)
    request.current_velocity_mps = Vector3(x=.3, y=.1, z=-.2)
    request.current_acceleration_mps2 = Vector3(x=.1, y=-.2, z=.3)
    request.max_acceleration_mps2 = .8
    request.max_jerk_mps3 = 1.
    candidates = _candidate_velocities(request)
    for ratio in (1., .7, .4):
        expected = _reachable_velocity(request, (0., 0., .8 * ratio))
        assert any(math.dist(candidate, expected) < 1e-9 for candidate in candidates)
    current = (.3, .1, -.2)
    for velocity in candidates:
        acceleration = tuple((velocity[i] - current[i]) / .2 for i in range(3))
        assert math.hypot(*velocity) <= request.max_speed_mps
        assert math.hypot(*acceleration) <= .8 + 1e-9
        assert math.dist(acceleration, (.1, -.2, .3)) <= .2 + 1e-9

