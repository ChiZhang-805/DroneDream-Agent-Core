"""Numerical teacher tests; never counted as physical counterfactual flights."""

from dataclasses import replace

import pytest
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.training.counterfactual_teacher import (
    CounterfactualConfig,
    CounterfactualState,
    CounterfactualTeacher,
)
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.outcome_verifier import OutcomeEnvelope


# 功能：
#   用单面墙、固定飞机包络和可调制动加速度构造名义离线教师。
# 输入：
#   wall_x：墙在 ENU x 轴上的位置。
#   braking：名义制动加速度。
# 输出：
#   evaluator：只做数值预测、不执行飞行的测试教师。
def teacher(*, wall_x=1.6, braking=1.0):
    geometry = [
        {
            "center_x": wall_x,
            "center_y": 0.0,
            "center_z": 1.0,
            "size_x": 0.02,
            "size_y": 10.0,
            "size_z": 4.0,
        }
    ]
    evaluator = CounterfactualTeacher(
        geometry,
        OutcomeEnvelope(0.2, 0.3, (-10.0, -10.0, -5.0), (10.0, 10.0, 8.0)),
        CounterfactualConfig(
            acceleration_mps2=2.0,
            braking_acceleration_mps2=braking,
            position_uncertainty_m=0.03,
            dynamic_acceleration_bound_mps2=0.5,
        ),
    )
    return evaluator


# 功能：
#   构造原点附近、目标位于前方且具有明确空动态见证的源状态。
# 输入：
#   speed：初始沿世界 x 轴的速度。
# 输出：
#   current：测试用反事实状态。
def state(*, speed=0.8):
    current = CounterfactualState(
        Vector3(x=0.0, y=0.0, z=1.0),
        Vector3(x=speed, y=0.0, z=0.0),
        QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0),
        Vector3(x=5.0, y=0.0, z=1.0),
        (),
        "a" * 64,
    )
    return current


# 功能：
#   构造有独立前向幅度和转向幅度的连续控制提案。
# 输入：
#   forward：归一化前向轴。
#   yaw：归一化转向轴。
# 输出：
#   proposal：显式四轴控制。
def action(forward=1.0, yaw=0.0):
    proposal = PilotAction(mode="pilot-control", axes=[forward, 0.0, 0.0, yaw])
    return proposal


# 功能：
#   核对风险确实受提案幅度和制动能力影响，并保留不能授予飞行资格的声明。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_risk_depends_on_braking_strength_and_student_action_magnitude():
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    evaluator = teacher(wall_x=1.05)
    speeding = evaluator.evaluate(state(), action(), limits)
    braking = evaluator.evaluate(state(), action(-0.5), limits)
    assert speeding["risk_target"] > braking["risk_target"]
    assert speeding["clearance_lower_bound_m"] < braking["clearance_lower_bound_m"]
    weak_brakes = teacher(wall_x=1.05, braking=0.5).evaluate(state(), action(), limits)
    assert weak_brakes["risk_target"] >= speeding["risk_target"]
    assert weak_brakes["times_seconds"][-1] > speeding["times_seconds"][-1]
    assert not speeding["qualification_granted"]


# 功能：
#   核对 FRU 转向符号及指令过期后的制动过程，不把右转误作世界正 y 移动。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rate_turns_velocity_and_expired_command_is_followed_by_braking():
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    receipt = teacher(wall_x=8.0).evaluate(state(speed=0.0), action(yaw=1.0), limits)
    assert receipt["positions_m"][-1][1] < 0  # FRU right is world -Y at identity attitude
    assert receipt["final_speed_mps"] < 1e-8
    assert receipt["physical_request"] == [1.0, 0.0, 0.0, 20.0]
    straight = teacher(wall_x=8.0).evaluate(state(speed=0.0), action(yaw=0.0), limits)
    assert straight["positions_m"][-1][1] == pytest.approx(0.0)


# 功能：
#   验证移动障碍、围栏和过期动态见证均参与风险结果或拒绝判定。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_moving_obstacle_and_fence_participate_in_counterfactual():
    evaluator = teacher(wall_x=8.0)
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    obstacle = DynamicObstacleObservation(
        obstacle_id="crossing",
        position_m=Vector3(x=0.8, y=1, z=1),
        velocity_mps=Vector3(x=0, y=-2, z=0),
        radius_m=0.2,
        height_m=1.8,
        confidence=1.0,
        age_seconds=0.0,
    )
    clear = evaluator.evaluate(state(), action(), limits)
    crossing = evaluator.evaluate(replace(state(), dynamic_obstacles=(obstacle,)), action(), limits)
    assert crossing["risk_target"] > clear["risk_target"]
    outside = evaluator.evaluate(
        replace(state(), position=Vector3(x=-9.95, y=0, z=1)), action(), limits
    )
    assert outside["risk_target"] == 1.0
    with pytest.raises(ValueError, match="WITNESS_UNCERTAIN"):
        evaluator.evaluate(
            replace(state(), dynamic_obstacles=(obstacle.model_copy(update={"age_seconds": 0.3}),)),
            action(),
            limits,
        )


# 功能：
#   核对学生风险和教师纠正分别绑定原提案及观测，悬停模式不能冒充零速度指令。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_correction_is_separate_from_student_risk_and_stop_is_not_velocity():
    observation = UnitEnvironment().reset(seed=1)
    evaluator = teacher(wall_x=1.3)
    correction, receipt = evaluator.correction(observation, state())
    risk, student_receipt = evaluator.risk(observation, state(), action())
    assert correction.observation_sha256 == risk.observation_sha256 == sha256_json(observation)
    assert correction.verifier_receipt_sha256 == sha256_json(receipt)
    assert risk.verifier_receipt_sha256 == sha256_json(student_receipt)
    assert risk.proposed_action_sha256 == sha256_json(action())
    assert evaluator.risk(observation, state(), PilotAction(mode="hold", axes=[0.0] * 4)) == (
        None,
        None,
    )


# 功能：
#   验证到达目标时的零速度纠正仍保留四轴标签，而非偷偷改成位置锁定模式。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_zero_velocity_correction_remains_a_velocity_label_not_position_hold():
    from dronedream_agent_core.training.dagger import corrected_samples

    observation = UnitEnvironment().reset(seed=1)
    neutral = replace(state(speed=0.0), goal=state().position)
    correction, _ = teacher(wall_x=8.0).correction(observation, neutral)
    assert correction.action.mode == "pilot-control"
    assert correction.action.axes == [0.0] * 4
    label, _ = corrected_samples(observation, action(), correction, None)
    assert label.target_action_index == 11 and label.target_pilot_control == [0.0] * 4
    with pytest.raises(ValueError, match="four explicit control axes"):
        type(label).model_validate({**label.model_dump(), "target_pilot_control": []})


# 功能：
#   验证邻近目标的连续搜索优于粗粒度种子、满足明确预算且可重复，不改变在线控制器。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_continuous_correction_refines_close_goal_without_changing_online_control():
    observation = UnitEnvironment().reset(seed=1)
    nearby = replace(state(speed=0.0), goal=Vector3(x=0.17, y=0.06, z=1.02))
    evaluator = teacher(wall_x=8.0)
    correction, receipt = evaluator.correction(observation, nearby)
    search = receipt["correction_search"]
    assert search["selected_score"] > search["seed_score"] + 0.001
    assert search["evaluated_actions"] <= search["maximum_evaluations"] == 52
    assert search["online_controller"] is False
    assert correction.action.axes[:3] != [0.0, -0.5, 0.0]
    assert all(-1 <= value <= 1 for value in correction.action.axes)
    assert correction.verifier_receipt_sha256 == sha256_json(receipt)
    repeated, repeated_receipt = evaluator.correction(observation, nearby)
    assert repeated == correction and repeated_receipt == receipt


# 功能：
#   记录全部候选的实际风险，确认最终纠正只从独立通过风险检查的候选中选出。
# 输入：
#   monkeypatch：包装教师评估调用。
# 输出：
#   None：不返回业务数据。
def test_refined_action_is_selected_only_from_individually_verified_candidates(monkeypatch):
    evaluator = teacher(wall_x=1.05)
    evaluate = evaluator.evaluate
    candidates = {}

    # 功能：
    #   委托真实计算并记录每个候选四轴与风险，不替换教师结论。
    # 输入：
    #   current：源世界状态。
    #   candidate：候选四轴提案。
    #   limits：物理幅度限额。
    # 输出：
    #   receipt：原教师计算回执。
    def record(current, candidate, limits):
        receipt = evaluate(current, candidate, limits)
        candidates[tuple(candidate.axes)] = receipt["risk_target"]
        return receipt

    monkeypatch.setattr(evaluator, "evaluate", record)
    correction, receipt = evaluator.correction(UnitEnvironment().reset(seed=1), state())
    assert any(risk > 0 for risk in candidates.values())
    assert candidates[tuple(correction.action.axes)] == 0
    assert receipt["correction_search"]["evaluated_actions"] == len(candidates)
    assert (
        receipt["correction_search"]["selected_score"]
        >= (receipt["correction_search"]["seed_score"])
    )


# 功能：
#   验证搜索轮数必须是范围内的整数，布尔与小数不能隐式转换成轮数。
# 输入：
#   passes：待拒绝的搜索轮数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("passes", [0, 9, True, 1.5])
def test_continuous_correction_has_a_finite_explicit_search_budget(passes):
    with pytest.raises(ValueError):
        CounterfactualConfig(
            acceleration_mps2=1.0,
            braking_acceleration_mps2=1.0,
            correction_refinement_passes=passes,
        )


# 功能：
#   验证无类型见证字段在几何计算前被拒绝，避免布尔转换或巨大整数溢出。
# 输入：
#   field：要破坏的状态字段。
#   error：期望的拒绝原因。
#   values：该字段的不合法值集合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,error,values",
    [
        ("context_sha256", "CONTEXT_IDENTITY_REQUIRED", [None, 1, []]),
        (
            "additional_position_uncertainty_m",
            "POSITION_UNCERTAINTY_INVALID",
            [True, None, "0.1", 10**400],
        ),
        ("observed_action_latency_seconds", "OBSERVED_LATENCY_INVALID", [True, 10**400]),
    ],
)
def test_malformed_witness_is_rejected_before_geometry(field, error, values):
    evaluator = teacher()
    for value in values:
        with pytest.raises(ValueError, match=error):
            evaluator.evaluate(
                replace(state(), **{field: value}), action(), PilotControlLimits(1.0, 1.0, 20.0)
            )
