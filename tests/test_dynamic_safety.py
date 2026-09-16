import math

import pytest

import dronedream_agent_core.dynamic_safety as dynamic_safety
from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    LocalPlannerRequest,
    PredictiveSafetyDecision,
    Vector3,
)
from dronedream_agent_core.dynamic_safety import (
    _candidate_velocities,
    predictive_safety_decision,
)


# 功能：
#   构造静止机体朝前方目标运动的合成预测请求，供各安全分支复用。
# 输入：
#   obstacles：可选动态障碍列表。
# 输出：
#   request：包含明确机体尺寸、时间步及安全余量的请求。
def _request(*, obstacles=None) -> LocalPlannerRequest:
    request = LocalPlannerRequest(
        current_position_m=Vector3(x=0, y=0, z=1),
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        target_position_m=Vector3(x=5, y=0, z=1),
        dynamic_obstacles=obstacles or [],
        vehicle_radius_m=0.2,
        vehicle_height_m=0.3,
        max_speed_mps=2,
        max_acceleration_mps2=10,
        required_clearance_m=0.3,
        prediction_horizon_seconds=2,
        prediction_step_seconds=0.2,
    )
    return request


# 功能：
#   验证无障碍通道保留朝向目标的可行速度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_clear_corridor_continues_toward_target() -> None:
    decision = predictive_safety_decision(_request(), [])
    assert decision.action == "continue"
    assert decision.selected_velocity_mps.x > 1.5


# 功能：
#   验证加速度限制造成的暂时方向偏差不被误当作绕障理由。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_acceleration_limited_heading_error_does_not_replace_clear_global_track() -> None:
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(x=0.0, y=1.0, z=0.0),
            "max_acceleration_mps2": 0.2,
        }
    )

    decision = predictive_safety_decision(request, [])

    assert decision.action == "continue"
    assert decision.threat_obstacle_id is None


# 功能：
#   验证直行预测撞墙时选择满足余量的侧向候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_static_wall_rejects_direct_velocity_before_impact() -> None:
    wall = {
        "name": "corridor-wall",
        "center_x": 2.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 0.3,
        "size_y": 2.0,
        "size_z": 3.0,
    }
    decision = predictive_safety_decision(_request(), [wall])
    assert decision.action in {"slow", "replan"}
    assert abs(decision.selected_velocity_mps.y) > 0.1
    assert decision.minimum_predicted_clearance_m >= 0.3


# 功能：
#   验证避障替换模型动作时真实标记确定性覆盖来源，并取消原偏航提议。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_static_wall_marks_model_velocity_as_deterministically_overridden() -> None:
    wall = {
        "name": "corridor-wall",
        "center_x": 2.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 0.3,
        "size_y": 2.0,
        "size_z": 3.0,
    }
    request = _request().model_copy(
        update={
            "requested_velocity_mps": Vector3(x=2.0, y=0.0, z=0.0),
            "requested_yaw_rate_dps": 15.0,
        }
    )

    decision = predictive_safety_decision(request, [wall])

    assert decision.action in {"slow", "replan"}
    assert decision.control_source in {
        "deterministic-safety-override",
        "deterministic-brake",
    }
    assert decision.control_source != "local-model-body-control"
    assert decision.selected_yaw_rate_dps == 0.0


# 功能：
#   验证水平候选无安全前进路径时增加竖直绕行候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_vertical_bypass_is_considered_when_primary_candidates_cannot_progress() -> None:
    wall = {
        "name": "wide-low-wall",
        "center_x": 1.5,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 0.4,
        "size_y": 100.0,
        "size_z": 0.6,
    }

    decision = predictive_safety_decision(_request(), [wall])

    assert decision.evaluated_candidate_count > 25
    assert decision.selected_velocity_mps.x > 0.0
    assert abs(decision.selected_velocity_mps.z) > 0.1
    assert decision.minimum_predicted_clearance_m >= 0.3


# 功能：
#   验证斜向和竖直候选合成后的整体速度仍满足机体上限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_three_axis_candidates_never_exceed_the_speed_envelope() -> None:
    request = _request().model_copy(
        update={
            "target_position_m": Vector3(x=5.0, y=0.0, z=6.0),
            "max_acceleration_mps2": 30.0,
        }
    )

    candidates = _candidate_velocities(
        request,
        vertical_biases=(0.35, -0.35),
    )

    assert candidates
    assert all(
        math.sqrt(candidate[0] ** 2 + candidate[1] ** 2 + candidate[2] ** 2)
        <= request.max_speed_mps + 1e-9
        for candidate in candidates
    )


# 功能：
#   验证模型速度提议逐步受加速度和加加速度限制，而非瞬间成为实测速度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_velocity_is_limited_by_acceleration_and_jerk_per_tick() -> None:
    request = _request().model_copy(
        update={
            "max_acceleration_mps2": 4.0,
            "max_jerk_mps3": 1.0,
            "requested_velocity_mps": Vector3(x=2.0, y=0.0, z=0.0),
            "current_acceleration_mps2": Vector3(x=0.0, y=0.0, z=0.0),
        }
    )

    decision = predictive_safety_decision(request, [])

    assert decision.control_source == "local-model-body-control"
    assert decision.selected_velocity_mps.x == pytest.approx(0.04)
    assert decision.maximum_jerk_mps3 == 1.0


# 功能：
#   验证三轴正负超速下的保持命令为零，但制动预测保留真实惯性和前方墙体威胁。
# 输入：
#   axis：超速所在轴。
#   sign：速度正负方向。
#   healthy：感知健康标志。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("healthy", [False, True])
def test_overspeed_brake_does_not_erase_inertia_at_the_command_speed_limit(axis, sign, healthy):
    current = [0.0, 0.0, 0.0]
    current[axis] = sign * 4.0
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(**dict(zip("xyz", current, strict=True))),
            "max_acceleration_mps2": 0.2,
            "max_jerk_mps3": 1.0,
            "perception_stream_healthy": healthy,
            "requested_velocity_mps": Vector3(x=2.0, y=0.0, z=0.0),
            "requested_yaw_rate_dps": 30.0,
        }
    )
    wall = dict(
        name="inertial-wall",
        center_x=0.0,
        center_y=0.0,
        center_z=1.0,
        size_x=20.0,
        size_y=20.0,
        size_z=20.0,
    )
    wall["center_" + "xyz"[axis]] += sign * 6.0
    wall["size_" + "xyz"[axis]] = 0.1
    decision = predictive_safety_decision(request, [wall])
    assert decision.action == "hold" and decision.control_source == "deterministic-brake"
    assert decision.selected_velocity_mps == Vector3(x=0.0, y=0.0, z=0.0)
    assert decision.selected_yaw_rate_dps == 0.0
    assert "MEASURED_SPEED_OUTSIDE_CONTROL_ENVELOPE" in decision.issue_codes
    forecast = decision.braking_prediction_velocity_mps
    assert getattr(forecast, "xyz"[axis]) == pytest.approx(sign * 3.96)
    assert decision.minimum_predicted_clearance_m < 0.0
    assert decision.threat_obstacle_id == "static:inertial-wall"
    assert _candidate_velocities(request) == []  # No impossible clipped candidates.


# 功能：
#   验证接近限速时向外加速仍会使下一步超限，不能靠速度裁剪掩盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_outward_acceleration_near_speed_limit_is_not_magically_cancelled():
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(x=1.95, y=0.0, z=0.0),
            "current_acceleration_mps2": Vector3(x=2.0, y=0.0, z=0.0),
            "max_acceleration_mps2": 3.0,
            "max_jerk_mps3": 0.1,
            "requested_velocity_mps": Vector3(x=2.0, y=0.0, z=0.0),
        }
    )
    decision = predictive_safety_decision(request, [])
    assert decision.action == "hold"
    assert decision.selected_velocity_mps == Vector3(x=0.0, y=0.0, z=0.0)
    assert "NEXT_STEP_SPEED_OUTSIDE_CONTROL_ENVELOPE" in decision.issue_codes
    # 1.95 + (2 - .1 * .2) * .2: the prediction can exceed the command cap.
    assert decision.braking_prediction_velocity_mps.x == pytest.approx(2.346)
    assert _candidate_velocities(request) == []


# 功能：
#   验证限速内可行的模型减速提议保留模型控制归因。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_feasible_braking_within_speed_limit_remains_model_controlled():
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(x=1.9, y=0.0, z=0.0),
            "current_acceleration_mps2": Vector3(x=-0.5, y=0.0, z=0.0),
            "max_acceleration_mps2": 1.0,
            "max_jerk_mps3": 1.0,
            "requested_velocity_mps": Vector3(x=0.2, y=0.0, z=0.0),
        }
    )
    decision = predictive_safety_decision(request, [])
    assert decision.action == "continue" and decision.control_source == "local-model-body-control"
    assert decision.selected_velocity_mps.x == pytest.approx(1.76)


# 功能：
#   验证添加远处静态几何不会改变当前获选动作及其真实威胁证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_far_geometry_culling_preserves_full_geometry_decision_evidence() -> None:
    wall = {
        "name": "corridor-wall",
        "center_x": 2.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 0.3,
        "size_y": 2.0,
        "size_z": 3.0,
    }
    far_geometry = [
        {
            "name": f"far-wall-{index}",
            "center_x": 100.0 + index,
            "center_y": 100.0,
            "center_z": 1.0,
            "size_x": 1.0,
            "size_y": 1.0,
            "size_z": 2.0,
        }
        for index in range(100)
    ]

    expected = predictive_safety_decision(_request(), [wall])
    actual = predictive_safety_decision(_request(), [wall, *far_geometry])

    assert actual.action == expected.action
    assert actual.selected_velocity_mps == expected.selected_velocity_mps
    assert actual.minimum_predicted_clearance_m == expected.minimum_predicted_clearance_m
    assert actual.threat_obstacle_id == expected.threat_obstacle_id


# 功能：
#   验证横向穿行目标触发局部避让，并保留该目标的威胁身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_crossing_obstacle_changes_local_motion() -> None:
    crossing = DynamicObstacleObservation(
        obstacle_id="person-1",
        position_m=Vector3(x=2, y=-1, z=1),
        velocity_mps=Vector3(x=0, y=1, z=0),
        radius_m=0.3,
        height_m=1.7,
        confidence=0.95,
        age_seconds=0.05,
    )
    decision = predictive_safety_decision(_request(obstacles=[crossing]), [])
    assert decision.action in {"slow", "replan"}
    assert decision.threat_obstacle_id == "person-1"
    assert decision.minimum_predicted_clearance_m >= 0.3


# 功能：
#   验证被障碍包围而无安全速度时输出保持和重规划原因。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_safe_velocity_fails_closed_to_hold() -> None:
    enclosing_volume = {
        "name": "invalid-start-volume",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
    }
    decision = predictive_safety_decision(_request(), [enclosing_volume])
    assert decision.action == "hold"
    assert "NO_SAFE_LOCAL_VELOCITY" in decision.issue_codes


# 功能：
#   验证超过观测期限的感知不能授权继续前进。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stale_perception_brakes_and_holds() -> None:
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(x=1.0, y=0.0, z=0.0),
            "perception_stream_age_seconds": 0.8,
            "maximum_perception_age_seconds": 0.5,
        }
    )
    decision = predictive_safety_decision(request, [])
    assert decision.action == "hold"
    assert decision.selected_velocity_mps.x < request.current_velocity_mps.x
    assert "PERCEPTION_STREAM_STALE" in decision.issue_codes


# 功能：
#   验证各类保护保持均区分零控制目标与保留惯性的未来位置预测。
# 输入：
#   cause：触发保持的证据问题或无路可走场景。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("cause", ["unhealthy", "stale", "uncertain", "no-safe-velocity"])
def test_hold_target_is_zero_but_braking_forecast_retains_inertia(cause):
    request = _request().model_copy(
        update={
            "current_velocity_mps": Vector3(x=1.0, y=0.1, z=0.05),
            "max_acceleration_mps2": 0.2,
            "perception_stream_healthy": cause != "unhealthy",
            "perception_stream_age_seconds": 1.0 if cause == "stale" else 0.0,
            "localization_covariance_m2": 1.0 if cause == "uncertain" else 0.0,
        }
    )
    geometry = [
        {
            "name": "enclosure",
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 1.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
        }
    ]
    decision = predictive_safety_decision(request, geometry if cause == "no-safe-velocity" else [])
    assert decision.action == "hold"
    assert decision.selected_velocity_mps == Vector3(x=0.0, y=0.0, z=0.0)
    assert decision.selected_yaw_rate_dps == 0
    forecast = decision.braking_prediction_velocity_mps
    assert 0.9 < forecast.x < 1.0
    assert decision.predicted_path_m[0].x == pytest.approx(forecast.x * 0.2)
    assert decision.predicted_path_m[-1].x > 1.0
    if cause == "no-safe-velocity":
        assert decision.minimum_predicted_clearance_m < 0
        assert decision.threat_obstacle_id == "static:enclosure"
    assert PredictiveSafetyDecision.model_validate(decision.model_dump()) == decision


# 功能：
#   验证保持合同拒绝任何残余速度、偏航以及错误的模型控制归因。
# 输入：
#   update：非法保持字段覆盖。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update",
    [
        {"selected_velocity_mps": {"x": 1e-16, "y": 0.0, "z": 0.0}},
        {"selected_yaw_rate_dps": 0.01},
        {"control_source": "local-model-body-control"},
    ],
)
def test_hold_rejects_even_tiny_residual_motion_at_contract_boundary(update):
    request = _request().model_copy(update={"perception_stream_healthy": False})
    payload = predictive_safety_decision(request, []).model_dump()
    with pytest.raises(ValueError, match="hold requires zero"):
        PredictiveSafetyDecision.model_validate({**payload, **update})


# 功能：
#   验证普通运动不冒用保持预测，未提供保持字段时仍保留原有线协议身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_motion_cannot_reuse_hold_forecast_and_absent_field_preserves_wire_identity():
    decision = predictive_safety_decision(_request(), [])
    payload = decision.model_dump()
    assert "braking_prediction_velocity_mps" not in payload
    with pytest.raises(ValueError, match="cannot label a motion command"):
        PredictiveSafetyDecision.model_validate(
            {**payload, "braking_prediction_velocity_mps": {"x": 0.0, "y": 0.0, "z": 0.0}}
        )


# 功能：
#   验证已经在安全位置稳定悬停时，零速度不被误标为安全故障而卡住全局任务。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_settled_safe_hold_sample_allows_global_schedule_to_advance() -> None:
    launch_pad = {
        "name": "launch-pad",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "radius_m": 0.85,
        "height_m": 0.08,
    }
    request = LocalPlannerRequest(
        current_position_m=Vector3(x=0.0, y=0.0, z=0.77),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.0, y=0.0, z=0.77),
        vehicle_radius_m=0.38,
        vehicle_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        required_clearance_m=0.34,
        prediction_horizon_seconds=3.0,
        prediction_step_seconds=0.2,
    )

    decision = predictive_safety_decision(request, [launch_pad])

    assert decision.action == "continue"
    assert decision.selected_velocity_mps == Vector3(x=0.0, y=0.0, z=0.0)
    assert decision.minimum_predicted_clearance_m >= request.required_clearance_m
    assert decision.issue_codes == []


# 功能：
#   验证起点尚未碰撞但余量不足时，只采用保留正净空且最终达到余量的恢复上升。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_positive_static_clearance_deficit_uses_monotonic_escape() -> None:
    launch_pad = {
        "name": "launch-pad",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "radius_m": 0.85,
        "height_m": 0.08,
    }
    request = LocalPlannerRequest(
        current_position_m=Vector3(x=0.0, y=0.0, z=0.35),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        vehicle_radius_m=0.38,
        vehicle_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        required_clearance_m=0.2,
        prediction_horizon_seconds=3.0,
        prediction_step_seconds=0.2,
    )
    decision = predictive_safety_decision(request, [launch_pad])
    assert decision.action == "slow"
    assert decision.selected_velocity_mps.z > 0.0
    assert decision.minimum_predicted_clearance_m > 0.0
    assert decision.issue_codes == ["STATIC_CLEARANCE_RECOVERY"]


# 功能：
#   验证很小的余量不足可以按实际缺口恢复，而非必须额外移动固定五厘米。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_small_static_clearance_deficit_does_not_require_fixed_five_cm_gain() -> None:
    launch_pad = {
        "name": "near-margin-pad",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "radius_m": 0.85,
        "height_m": 0.08,
    }
    request = LocalPlannerRequest(
        current_position_m=Vector3(x=0.0, y=0.0, z=0.45),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.0, y=0.0, z=0.46),
        vehicle_radius_m=0.38,
        vehicle_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        required_clearance_m=0.2,
        prediction_horizon_seconds=3.0,
        prediction_step_seconds=0.2,
    )

    decision = predictive_safety_decision(request, [launch_pad])

    assert decision.action == "slow"
    assert decision.selected_velocity_mps.z > 0.0
    assert 0.0 < decision.minimum_predicted_clearance_m < request.required_clearance_m
    assert decision.predicted_path_m[-1].z > request.current_position_m.z
    assert decision.issue_codes == ["STATIC_CLEARANCE_RECOVERY"]


# 功能：
#   故意令宽阶段漏掉头顶障碍，验证最终完整几何重验仍能否决恢复动作。
# 输入：
#   monkeypatch：替换宽阶段筛选函数。
# 输出：
#   None：不返回业务数据。
def test_static_clearance_recovery_is_reproved_against_complete_geometry(
    monkeypatch,
) -> None:
    launch_pad = {
        "name": "launch-pad",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "radius_m": 0.85,
        "height_m": 0.08,
    }
    unexpected_overhead_obstacle = {
        "name": "overhead-obstacle",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.85,
        "size_x": 3.0,
        "size_y": 3.0,
        "size_z": 0.1,
    }
    request = LocalPlannerRequest(
        current_position_m=Vector3(x=0.0, y=0.0, z=0.35),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        vehicle_radius_m=0.38,
        vehicle_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        required_clearance_m=0.2,
        prediction_horizon_seconds=3.0,
        prediction_step_seconds=0.2,
    )
    monkeypatch.setattr(
        dynamic_safety,
        "_candidate_search_primitives",
        lambda _request, _primitives: [launch_pad],
    )

    decision = predictive_safety_decision(
        request,
        [launch_pad, unexpected_overhead_obstacle],
    )

    assert decision.action == "hold"
    assert decision.issue_codes == [
        "FULL_GEOMETRY_RECHECK_FAILED",
        "HOLD_AND_REQUEST_REPLAN",
    ]
