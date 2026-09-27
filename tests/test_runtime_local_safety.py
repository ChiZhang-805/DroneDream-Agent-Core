import math

import pytest

from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety


# 功能：
#   建立固定几何和动力学包络的测试载具，不冒充真实硬件资格。
# 输入：
#   无：采用本文件明确声明的测试常量。
# 输出：
#   vehicle：供安全评估测试使用的载具模型。
def _vehicle() -> VehicleAsset:
    vehicle = VehicleAsset(
        asset_id="test-x500",
        name="Test X500",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=3.0,
        body_radius_m=0.38,
        body_height_m=0.43,
        max_speed_mps=2.0,
        max_acceleration_mps2=4.0,
        qualified_range_m=100.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=0.2,
        sensors=["depth-camera", "stereo-vio"],
    )
    return vehicle


# 功能：
#   横穿动态障碍触发受限动作，指令保留原观测、预测起点和短租期，不把目标当作直接执行许可。
# 输入：
#   无：使用固定交叉运动场景。
# 输出：
#   None：不返回业务数据。
def test_command_is_short_lived_hash_bound_and_avoids_crossing_track() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=7,
        observed_at_unix_ms=10_000,
        source="simulation-ground-truth",
        stream_healthy=True,
        stream_age_seconds=0.02,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.5),
        current_velocity_mps=Vector3(x=1.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=5.0, y=0.0, z=1.5),
        dynamic_obstacles=[
            DynamicObstacleObservation(
                obstacle_id="person-crossing",
                position_m=Vector3(x=2.0, y=-1.0, z=1.5),
                velocity_mps=Vector3(x=0.0, y=1.0, z=0.0),
                radius_m=0.35,
                height_m=1.7,
                confidence=0.98,
                age_seconds=0.02,
            )
        ],
    )
    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_010,
    )
    assert command.observation_sequence == 7
    assert command.evaluated_target_position_m == observation.target_position_m
    assert command.tracking_recovery_active is False
    assert command.estimator_to_world_position_offset_m == Vector3(x=0.0, y=0.0, z=0.0)
    # 新筛选器可以找到仍有时效余量的替代动作；不能再固定期待旧的制动租期。
    assert 10_010 < command.valid_until_unix_ms <= observation.observed_at_unix_ms + 250
    assert command.observation_budget.disposition == "control-eligible"
    assert command.decision.action in {"slow", "replan", "hold"}
    assert command.decision.threat_obstacle_id == "person-crossing"
    assert command.command_position_m.x == pytest.approx(
        observation.current_position_m.x + command.decision.selected_velocity_mps.x * 0.8
    )
    forecast_velocity = (
        command.decision.braking_prediction_velocity_mps or command.decision.selected_velocity_mps
    )
    assert command.decision.predicted_path_m[0].x == pytest.approx(
        observation.current_position_m.x + forecast_velocity.x * 0.2
    )


# 功能：
#   恢复指令绑定当前目标与恢复模式，输出速度不突破恢复阶段上限。
# 输入：
#   无：使用从静止开始的合成机载观测。
# 输出：
#   None：不返回业务数据。
def test_command_binds_tracking_recovery_mode_and_target() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=8,
        observed_at_unix_ms=10_100,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=-1.0, y=2.0, z=1.0),
    )
    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_110,
        tracking_recovery_active=True,
    )
    assert command.tracking_recovery_active is True
    assert command.evaluated_target_position_m == observation.target_position_m
    velocity = command.decision.selected_velocity_mps
    assert math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) <= 0.200001


# 功能：
#   不健康观测触发零运动制动，仍保留惯性风险预测；训练标签只承认真实零速度执行回执。
# 输入：
#   无：使用带侧向速度和加速度的失效观测。
# 输出：
#   None：不返回业务数据。
def test_unhealthy_hold_has_no_position_carrot_and_compiles_actual_zero_feedforward():
    from dronedream_agent_core.control_execution_evidence import control_application_record
    from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
    from dronedream_agent_core.training.executed_control import executed_training_action

    position = Vector3(x=-0.36, y=0.09, z=0.5)
    observation = RuntimeLocalSafetyObservation(
        sequence=1,
        observed_at_unix_ms=10_000,
        source="onboard",
        stream_healthy=False,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=position,
        current_velocity_mps=Vector3(x=0.01, y=-0.15, z=0.003),
        current_acceleration_world_enu_mps2=Vector3(x=0.3, y=-2.0, z=0.05),
        target_position_m=Vector3(x=-2.0, y=2.0, z=1.5),
    )
    vehicle = _vehicle().model_copy(update={"max_acceleration_mps2": 0.2})
    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=vehicle,
        static_primitives=[],
        required_clearance_m=0.25,
        generated_at_unix_ms=10_020,
        model_call_id="expired-native-call",
        navigation_control_authority="model-required",
        model_authority_reason="control-intent-dispatch-budget-exhausted",
    )
    assert command.command_position_m == position
    assert command.decision.braking_prediction_velocity_mps.y < -0.1
    application = control_application_record(
        command,
        sequence=1,
        accepted_at_unix_ms=10_040,
        transport="position-velocity-ned",
        velocity_ned_mps=(0.0, 0.0, 0.0),
        position_ned_m=(position.y, position.x, -position.z),
        yaw_heading_deg=95.0,
    )
    action, expired = executed_training_action(
        {}, command, application, limits=PilotControlLimits(0.4, 0.4, 20.0)
    )
    assert not expired and action.mode == "hold" and action.axes == [0.0] * 4
    with pytest.raises(ValueError, match="HOLD_RECEIPT_NOT_VERIFIABLE"):
        executed_training_action(
            {},
            command,
            application.model_copy(update={"velocity_ned_mps": (0.0001, 0.0, 0.0)}),
            limits=PilotControlLimits(0.4, 0.4, 20.0),
        )


# 功能：
#   模型必需模式缺少租约身份时拒绝，完整绑定目标、调用和任务摘要后才生成相应指令。
# 输入：
#   无：使用同一观测分别测试不完整与完整模型授权。
# 输出：
#   None：不返回业务数据。
def test_model_required_command_requires_fully_bound_authority() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=10,
        observed_at_unix_ms=10_300,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.12, y=0.0, z=1.0),
    )
    with pytest.raises(ValueError, match="fully bound model lease"):
        evaluate_runtime_local_safety(
            observation=observation,
            vehicle=_vehicle(),
            static_primitives=[],
            required_clearance_m=0.35,
            generated_at_unix_ms=10_310,
            navigation_control_authority="model-required",
            model_navigation_authorized=True,
        )

    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_310,
        navigation_goal_id="source-waypoint-0007",
        navigation_control_authority="model-required",
        model_navigation_authorized=True,
        model_call_id="call-model-authority",
        model_selected_candidate_id="candidate-safe-corridor",
        model_path_sha256="a" * 64,
        model_authority_reason="model-path-lease-active",
    )
    assert command.model_navigation_authorized is True
    assert command.navigation_goal_id == "source-waypoint-0007"
    assert command.model_call_id == "call-model-authority"
    assert command.model_path_sha256 == "a" * 64


# 功能：
#   侧向速度较大时恢复首先减速，不因靠近目标就忽略已有惯性直接重入航迹。
# 输入：
#   无：使用横向速度每秒零点六米的观测。
# 输出：
#   None：不返回业务数据。
def test_tracking_recovery_brakes_high_lateral_speed_before_rejoining() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=9,
        observed_at_unix_ms=10_200,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.6, z=0.0),
        target_position_m=Vector3(x=0.4, y=0.0, z=1.0),
    )

    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_210,
        tracking_recovery_active=True,
    )

    velocity = command.decision.selected_velocity_mps
    assert math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) < 0.6
    assert abs(velocity.y) < 0.6


# 功能：
#   局部避障同时受任务速度限制，不能仅凭载具硬件支持更快速度就自行加速。
# 输入：
#   无：使用硬件上限高于任务上限的测试载具。
# 输出：
#   None：不返回业务数据。
def test_mission_speed_limit_bounds_local_avoidance_below_vehicle_capability() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=13,
        observed_at_unix_ms=10_600,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=5.0, y=0.0, z=1.0),
    )

    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_610,
        maximum_speed_mps=0.34,
    )

    velocity = command.decision.selected_velocity_mps
    assert math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) <= 0.340001


# 功能：
#   明确拒绝零任务速度，避免非法配置被用作合法运动上限。
# 输入：
#   无：使用真实评估入口及零速度参数。
# 输出：
#   None：不返回业务数据。
def test_mission_speed_limit_must_be_positive_and_finite() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=14,
        observed_at_unix_ms=10_700,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=1.0, y=0.0, z=1.0),
    )

    with pytest.raises(ValueError, match="maximum speed"):
        evaluate_runtime_local_safety(
            observation=observation,
            vehicle=_vehicle(),
            static_primitives=[],
            required_clearance_m=0.35,
            generated_at_unix_ms=10_710,
            maximum_speed_mps=0.0,
        )


# 功能：
#   接近恢复目标时按距离降低速度，使位置接近的同时也能满足稳定速度门槛。
# 输入：
#   无：距目标零点二米且静止的合成观测。
# 输出：
#   None：不返回业务数据。
def test_tracking_recovery_tapers_below_settle_speed_near_target() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=11,
        observed_at_unix_ms=10_400,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.2, y=0.0, z=1.0),
    )

    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_410,
        tracking_recovery_active=True,
    )

    velocity = command.decision.selected_velocity_mps
    assert math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) <= 0.090001
    assert velocity.x > 0.0
    assert command.command_position_m.x == pytest.approx(velocity.x * 0.8)


# 功能：
#   靠近目标但速度仍高时保留短位置参照时域，不提前启用低速收敛延长策略。
# 输入：
#   无：距目标零点三米且仍以每秒零点四米运动的观测。
# 输出：
#   None：不返回业务数据。
def test_tracking_recovery_keeps_short_carrot_while_speed_is_high() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=12,
        observed_at_unix_ms=10_500,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.4, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.3, y=0.0, z=1.0),
    )

    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=10_510,
        command_horizon_seconds=0.2,
        tracking_recovery_active=True,
    )

    assert command.command_position_m.x == pytest.approx(
        command.decision.selected_velocity_mps.x * 0.2
    )


# 功能：
#   定位协方差超出允许范围时保持制动，并在指令中记录定位不确定原因。
# 输入：
#   无：使用较大定位误差的机载观测。
# 输出：
#   None：不返回业务数据。
def test_command_holds_when_localization_is_unbounded() -> None:
    observation = RuntimeLocalSafetyObservation(
        sequence=1,
        observed_at_unix_ms=1_000,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.9,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=0.2, y=0.0, z=0.0),
        target_position_m=Vector3(x=2.0, y=0.0, z=1.0),
    )
    command = evaluate_runtime_local_safety(
        observation=observation,
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=1_010,
    )
    assert command.decision.action == "hold"
    assert "LOCALIZATION_UNCERTAIN" in command.decision.issue_codes
