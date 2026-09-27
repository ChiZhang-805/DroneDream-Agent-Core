"""Decision-time latency bounds; synthetic nominal tests, not flight evidence."""

from dataclasses import replace

import pytest
from test_counterfactual_teacher import action, state, teacher

from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.risk_latency_contract import (
    decision_latency_contract,
    validate_decision_latency_prediction,
)


# 功能：证明旧回执身份不变，新预测策略必须显式改变配置身份。
# 输入：无；仅合成教师配置。输出：身份或版本串混用时断言失败。
def test_latency_policy_is_explicit_and_legacy_identity_survives():
    config = CounterfactualConfig(acceleration_mps2=2.0, braking_acceleration_mps2=1.0)
    legacy = config.model_dump()
    assert "decision_latency_policy" not in legacy
    assert sha256_json(CounterfactualConfig.model_validate(legacy)) == sha256_json(legacy)
    config.decision_latency_policy = "bounded-source-age-v1"
    assert config.model_dump()["decision_latency_policy"] == "bounded-source-age-v1"
    assert sha256_json(config) != sha256_json(legacy)


# 功能：保持决策时状态不变、仅改变未来接收耗时，要求新版目标和预测轨迹完全相同。
# 输入：later：合法的未来执行延迟。输出：预测不得偷看执行后耗时；原耗时仍留在回执中。
@pytest.mark.parametrize("later", [0.0, 0.07, 0.109, 0.19, 0.25])
def test_future_acceptance_time_cannot_change_decision_target(later):
    oracle = teacher(wall_x=1.3)
    oracle.config.decision_latency_policy = "bounded-source-age-v1"
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    first = oracle.evaluate(state(), action(), limits)
    other = oracle.evaluate(
        replace(state(), observed_action_latency_seconds=later), action(), limits
    )
    assert first["positions_m"] == other["positions_m"]
    assert first["risk_target"] == other["risk_target"]
    assert first["clearance_lower_bound_m"] == other["clearance_lower_bound_m"]
    assert other["state"]["observed_action_latency_seconds"] == later
    assert (
        other["decision_latency_contract"]["uses_future_actuation_latency_for_prediction"] is False
    )


# 功能：包含向墙、离墙及侧向初速，检查名义延迟包络覆盖多个实际延迟，不能假定迟到总更危险。
# 输入：vx、vy：初速；forward：动作。输出：有界预测不得比任何采样延迟的旧名义预测更乐观。
@pytest.mark.parametrize("vx,vy", [(0.8, 0.0), (-0.8, 0.0), (0.0, 0.7), (0.0, 0.0)])
@pytest.mark.parametrize("forward", [-1.0, 0.0, 1.0])
def test_interval_allowance_covers_nominal_timing_samples(vx, vy, forward):
    base = teacher(wall_x=1.3)
    base.config.latency_seconds = 0.0
    bounded = teacher(wall_x=1.3)
    bounded.config.decision_latency_policy = "bounded-source-age-v1"
    current = replace(state(), velocity=Vector3(x=vx, y=vy, z=0.0))
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    conservative = bounded.evaluate(current, action(forward), limits)
    for latency in (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25):
        actual = base.evaluate(
            replace(current, observed_action_latency_seconds=latency), action(forward), limits
        )
        assert conservative["clearance_lower_bound_m"] <= actual["clearance_lower_bound_m"] + 1e-9


# 功能：新策略也拒绝超时记录，不能用延迟包络让过期动作成为合法训练数据。
# 输入：无。输出：超过共同源年龄上界的执行记录必须报错。
def test_bounded_policy_does_not_admit_expired_execution():
    oracle = teacher()
    oracle.config.decision_latency_policy = "bounded-source-age-v1"
    with pytest.raises(ValueError, match="OBSERVED_LATENCY_INVALID"):
        oracle.evaluate(
            replace(state(), observed_action_latency_seconds=0.251),
            action(),
            PilotControlLimits(1.0, 1.0, 20.0),
        )


# 功能：核验实际新回执及故意损坏的延迟字段，避免版本混用和布尔替代数值。
# 输入：field、value：变造方式。输出：所有变造均被拒绝，原始回执可接纳。
@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_seconds", 0.26),
        ("minimum_seconds", False),
        ("additional_swept_uncertainty_m", 0.0),
        ("nominal_dynamics_only", 1),
        ("uses_future_actuation_latency_for_prediction", True),
    ],
)
def test_prediction_contract_cannot_be_relabelled(field, value):
    oracle = teacher()
    oracle.config.decision_latency_policy = "bounded-source-age-v1"
    receipt = oracle.evaluate(state(), action(), PilotControlLimits(1.0, 1.0, 20.0))
    validate_decision_latency_prediction(receipt, oracle.config)
    receipt["decision_latency_contract"][field] = value
    with pytest.raises(ValueError, match="LATENCY_CONTRACT_MISMATCH"):
        validate_decision_latency_prediction(receipt, oracle.config)


# 功能：非法速度或积分周期不能生成可被误认为合法的延迟包络。
# 输入：bad：非法数值。输出：输入异常必须被明确拒绝。
@pytest.mark.parametrize("bad", [True, -1.0, float("nan"), float("inf"), "1"])
def test_invalid_speed_bound_is_rejected(bad):
    with pytest.raises(ValueError, match="LATENCY_BOUND_INPUT_INVALID"):
        decision_latency_contract(bad, 1.0, 0.0, 0.02)


# 功能：覆盖有运动障碍的相对时序，保证原见证不被改写且延迟区间确实包含障碍位移。
# 输入：无；独立运动圆柱障碍。输出：新版名义净空不大于采样延迟的预测净空。
def test_dynamic_obstacle_latency_is_part_of_relative_motion_allowance():
    obstacle = DynamicObstacleObservation(
        obstacle_id="synthetic-moving-cylinder",
        position_m=Vector3(x=1.4, y=0.2, z=1.0),
        velocity_mps=Vector3(x=-0.4, y=0.0, z=0.0),
        radius_m=0.2,
        height_m=1.0,
        age_seconds=0.02,
        confidence=1.0,
    )
    current = replace(state(speed=0.3), dynamic_obstacles=(obstacle,))
    original = obstacle.model_dump()
    base, bounded = teacher(wall_x=8.0), teacher(wall_x=8.0)
    base.config.latency_seconds = 0.0
    bounded.config.decision_latency_policy = "bounded-source-age-v1"
    limits = PilotControlLimits(1.0, 1.0, 20.0)
    result = bounded.evaluate(current, action(-0.5), limits)
    assert (
        result["decision_latency_contract"]["additional_swept_uncertainty_m"] >= (0.3 + 0.4) * 0.25
    )
    for delay in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
        observed = base.evaluate(
            replace(current, observed_action_latency_seconds=delay), action(-0.5), limits
        )
        assert result["clearance_lower_bound_m"] <= observed["clearance_lower_bound_m"] + 1e-9
    assert obstacle.model_dump() == original


# 功能：旧策略不接受新版声明，新策略缺少或更换实际延迟字段也不能接纳。
# 输入：无。输出：缺失契约、混合版本及错误实际边界均被拒绝。
def test_missing_mixed_or_wrong_effective_contract_rejected():
    oracle = teacher()
    oracle.config.decision_latency_policy = "bounded-source-age-v1"
    result = oracle.evaluate(state(), action(), PilotControlLimits(1.0, 1.0, 20.0))
    legacy = teacher().config
    with pytest.raises(ValueError, match="LATENCY_POLICY_MIXED"):
        validate_decision_latency_prediction(result, legacy)
    result["effective_latency_seconds"] = 0.07
    with pytest.raises(ValueError, match="LATENCY_CONTRACT_MISMATCH"):
        validate_decision_latency_prediction(result, oracle.config)
    result["effective_latency_seconds"] = 0.25
    del result["decision_latency_contract"]
    with pytest.raises(ValueError, match="LATENCY_CONTRACT_MISMATCH"):
        validate_decision_latency_prediction(result, oracle.config)
