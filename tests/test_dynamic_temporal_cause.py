"""Synthetic obstacle-attribution boundaries, not physical training data."""

import math

import pytest

from dronedream_agent_core.contracts import DynamicObstacleObservation, LocalPlannerRequest, Vector3
from dronedream_agent_core.dynamic_safety import predictive_safety_decision
from dronedream_agent_core.observation_validity import observation_validity


# 功能：
#   创建几何净空尚未越限但可能需要提前减速的合成请求，不使用仿真真值标签。
# 输入：
#   无。
# 输出：
#   request：前方两米处有动态物体的低速控制请求。
def request_with_budget_obstacle():
    request = LocalPlannerRequest(current_position_m=Vector3(x=0, y=0, z=1),
        current_velocity_mps=Vector3(x=0, y=0, z=0), target_position_m=Vector3(x=5, y=0, z=1),
        dynamic_obstacles=[DynamicObstacleObservation(obstacle_id='budget-obstacle',
            position_m=Vector3(x=2, y=0, z=1), velocity_mps=Vector3(x=0, y=0, z=0),
            radius_m=.2, height_m=1., confidence=.9, age_seconds=0.)],
        vehicle_radius_m=.2, vehicle_height_m=.3, max_speed_mps=.4,
        max_acceleration_mps2=10, required_clearance_m=.3,
        prediction_horizon_seconds=2, prediction_step_seconds=.2)
    return request


# 功能：
#   验证物体耗尽时效净空余量造成的提前减速属于真实避让起因，不等到接近碰撞才标注。
# 输入：
#   无。
# 输出：
#   None：合成决策必须仍通过原净空门槛，并绑定限制该余量的物体。
def test_dynamic_margin_causing_budget_rejection_is_recorded():
    decision = predictive_safety_decision(request_with_budget_obstacle(), [],
        candidate_check=lambda velocity, clearance, yaw: clearance >= 1.)
    assert decision.action in {'continue', 'slow', 'replan'}
    assert decision.minimum_predicted_clearance_m >= 1.
    assert decision.selected_velocity_mps.x < .4
    assert decision.avoidance_obstacle_id == 'budget-obstacle'


# 功能：
#   验证与净空无关的时钟过期不能因为附近存在物体而生成动态恢复标签。
# 输入：
#   无。
# 输出：
#   None：全部时效无效时必须刹停且不归因于最近物体。
def test_expired_budget_does_not_become_an_obstacle_cause():
    decision = predictive_safety_decision(request_with_budget_obstacle(), [],
        candidate_check=lambda velocity, clearance, yaw: False)
    assert decision.action == 'hold'
    assert decision.avoidance_obstacle_id is None


# 功能：
#   用实际自适应时效算法区分近障余量不足与单纯陈旧；对照预测不能放宽实际动作期限。
# 输入：
#   age_ms：原始观测已过去的毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('age_ms', [60, 100, 230, 251])
def test_cause_uses_actual_observation_budget(age_ms):
    request = request_with_budget_obstacle()

    # 功能：
    #   在固定原始时钟、定位余量及发令预算下校验候选，不续租同一帧。
    # 输入：
    #   velocity、clearance、yaw：已预测速度、净空及实际偏航速度。
    # 输出：
    #   allowed：实际时效判定是否允许候选继续接受安全检查。
    def current(velocity, clearance, yaw):
        validity = observation_validity(source_observed_at_unix_ms=1000,
            now_unix_ms=1000 + age_ms, inherited_deadline_unix_ms=1600,
            clearance_margin_m=max(0., clearance - request.required_clearance_m),
            ego_speed_bound_mps=math.hypot(*velocity), obstacle_speed_bound_mps=0.,
            acceleration_bound_mps2=.8, uncertainty_margin_m=.68,
            downstream_reserve_ms=70, angular_speed_bound_rad_s=abs(math.radians(yaw)))
        allowed = validity.disposition == 'control-eligible'
        return allowed

    decision = predictive_safety_decision(request, [], candidate_check=current)
    if age_ms < 180:
        assert decision.avoidance_obstacle_id == 'budget-obstacle'
        assert decision.action != 'hold'
        velocity = decision.selected_velocity_mps
        assert current((velocity.x, velocity.y, velocity.z),
                       decision.minimum_predicted_clearance_m, decision.selected_yaw_rate_dps)
    else:
        assert decision.action == 'hold'
        assert decision.avoidance_obstacle_id is None
