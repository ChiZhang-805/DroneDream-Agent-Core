"""Hash-bound adapter from live observations to short-lived local safety commands."""

from __future__ import annotations

import math
from typing import Any

from .contracts import (
    BodyFrameControlIntent,
    ControlObservationBudget,
    HybridControlLease,
    LocalPlannerRequest,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from .control_timing import LOCAL_DISPATCH_RESERVE_MS
from .control_uncertainty import localization_uncertainty_margin_m
from .dynamic_safety import predictive_braking_decision, predictive_safety_decision
from .hashing import sha256_json
from .motion_envelope import RUNTIME_PREDICTION_HORIZON_SECONDS, prediction_query_radius_m
from .observation_validity import observation_validity, publication_deadline
from .realtime_feature_encoders import body_to_world_enu
from .vertical_navigation import VerticalMotionGuard
from .yaw_command_envelope import YawCommandEnvelope


# 功能：
#   1. 在改写生成时刻前检查既有租期，慢计算只会消耗预算，不能制造负寿命或续期。
#   2. 有效命令重新验证定位偏移和完整合同；失效结果不进入实时命令通道。
# 输入：
#   evaluated：保留原始传感器期限的已评估命令。
#   published_at_unix_ms：实际准备发布的 Unix 毫秒时刻。
#   estimator_offset：本次已测量的估计器与世界坐标偏移。
# 输出：
#   command：仍满足传输预算的命令，否则为 None。
def prepare_safety_publication(
    evaluated: RuntimeLocalSafetyCommand, published_at_unix_ms: int, estimator_offset: Vector3,
) -> RuntimeLocalSafetyCommand | None:
    if (type(published_at_unix_ms) is not int
            or published_at_unix_ms < evaluated.generated_at_unix_ms):
        raise ValueError("SAFETY_PUBLICATION_CLOCK_REGRESSED")
    # 定位约束即使在结果过期时也要验证，不能用过期路径掩盖异常偏移。
    payload = {**evaluated.model_dump(mode="python"),
               "estimator_to_world_position_offset_m": estimator_offset}
    checked = RuntimeLocalSafetyCommand.model_validate(payload)
    if publication_deadline(
        evaluated_deadline_unix_ms=checked.valid_until_unix_ms,
        published_at_unix_ms=published_at_unix_ms, reserve_ms=LOCAL_DISPATCH_RESERVE_MS,
    ) is None:
        return None
    command = RuntimeLocalSafetyCommand.model_validate({
        **payload, "generated_at_unix_ms": published_at_unix_ms,
    })
    return command


# 功能：
#   判断物理标量是否为有限实数，排除布尔值、字符串和无法表示为浮点数的整数。
# 输入：
#   value：准备参与运动计算的数值。
# 输出：
#   valid：数值类型与有限性是否有效。
def _finite_scalar(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   根据当前观测、载具包络与保留惯性给出几何粗筛半径，覆盖超速制动时仍会经过的空间。
# 输入：
#   observation：本次安全评估使用的实时观测。
#   vehicle：当前载具几何与动力学上限。
#   required_clearance_m：需要保留的净空距离，单位米。
#   maximum_speed_mps：当前任务允许的速度上限，单位米每秒。
# 输出：
#   radius_m：以当前观测位置为中心、至少六米的查询半径。
def runtime_safety_query_radius_m(
    *,
    observation: RuntimeLocalSafetyObservation,
    vehicle: VehicleAsset,
    required_clearance_m: float,
    maximum_speed_mps: float,
) -> float:
    if not _finite_scalar(maximum_speed_mps) or maximum_speed_mps <= 0.0:
        raise ValueError("maximum speed must be finite and positive")
    if not _finite_scalar(required_clearance_m) or not 0 <= required_clearance_m <= 20:
        raise ValueError("required clearance must be finite and within [0, 20] metres")
    observation = RuntimeLocalSafetyObservation.model_validate(
        observation.model_dump(mode="python"), strict=True
    )
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    acceleration = observation.current_acceleration_world_enu_mps2
    # The safety evaluator permits steps up to .25 s and clamps its starting
    # acceleration to the vehicle envelope. Bound the same retained inertia
    # BEFORE querying geometry, including when braking from overspeed. A later
    # planner cannot recover a wall omitted by an undersized broad phase.
    retained_acceleration = (
        min(
            vehicle.max_acceleration_mps2,
            math.hypot(acceleration.x, acceleration.y, acceleration.z),
        )
        if acceleration is not None
        else 0.0
    )
    radius_m = max(
        6.0,
        prediction_query_radius_m(
            current_velocity_mps=tuple(
                getattr(observation.current_velocity_mps, axis) for axis in "xyz"
            ),
            maximum_speed_mps=min(vehicle.max_speed_mps, maximum_speed_mps),
            horizon_seconds=RUNTIME_PREDICTION_HORIZON_SECONDS,
            vehicle_radius_m=vehicle.body_radius_m,
            vehicle_height_m=vehicle.body_height_m,
            required_clearance_m=required_clearance_m,
            acceleration_speed_margin_mps=retained_acceleration * 0.25,
        ),
    )
    return radius_m


# 功能：
#   1. 固定并重验观测、载具和可选模型控制意图，在原授权内预测碰撞、速度、加速度与制动。
#   2. 以真实源时间、运动速度、几何余量和定位误差收紧指令期限，证据耗尽时保留威胁并制动。
#   3. 输出绑定原观测与控制来源的短时指令，最终执行器仍须核对时效、任务及物理限制。
# 输入：
#   observation：当前传感器融合观测。
#   vehicle：当前载具的几何和运动包络。
#   static_primitives：粗筛取得的当前静态碰撞几何。
#   required_clearance_m：最小净空，单位米。
#   generated_at_unix_ms：本次生成指令的整数 Unix 毫秒时刻。
#   command_horizon_seconds：将已验证速度转换为短时位置参照的时域。
#   prediction_step_seconds：碰撞预测的时间步长。
#   validity_milliseconds：指令最大租期，仍受原始观测和模型期限约束。
#   maximum_speed_mps：可选的任务速度上限，不能突破载具上限。
#   tracking_recovery_active：明确的航迹恢复模式开关。
#   navigation_goal_id：当前语义目标身份。
#   navigation_control_authority：模型必需模式或明确的航迹回退模式。
#   model_navigation_authorized：当前是否持有模型导航授权。
#   model_call_id：产生当前控制的模型调用身份。
#   model_selected_candidate_id：候选兼容模式的已授权候选身份。
#   model_path_sha256：当前模型任务参照摘要。
#   model_navigation_snapshot_sha256：模型消费的导航快照摘要。
#   model_authority_reason：控制授权或拒绝原因。
#   requested_control_intent：可选的已绑定四轴控制意图。
#   route_yaw_rate_dps：仅航迹回退模式使用的偏航角速度，单位度每秒。
#   motion_guard：与观测摘要绑定的本周期竖直运动、覆盖及负载约束。
#   yaw_envelope：可选的同源偏航变化包络，仅调整安全裁决前的模型提案，不延迟后续制动。
# 输出：
#   command：带观测摘要、风险判断、时效预算与执行参照的短时控制指令。
def evaluate_runtime_local_safety(
    *,
    observation: RuntimeLocalSafetyObservation,
    vehicle: VehicleAsset,
    static_primitives: list[dict[str, Any]],
    required_clearance_m: float,
    generated_at_unix_ms: int,
    command_horizon_seconds: float = 0.8,
    prediction_step_seconds: float = 0.2,
    validity_milliseconds: int = 500,
    maximum_speed_mps: float | None = None,
    tracking_recovery_active: bool = False,
    navigation_goal_id: str | None = None,
    navigation_control_authority: str = "route-fallback",
    hybrid_lease: HybridControlLease | None = None,
    model_navigation_authorized: bool = False,
    model_call_id: str | None = None,
    model_selected_candidate_id: str | None = None,
    model_path_sha256: str | None = None,
    model_navigation_snapshot_sha256: str | None = None,
    model_authority_reason: str | None = None,
    requested_control_intent: BodyFrameControlIntent | None = None,
    route_yaw_rate_dps: float = 0.0,
    motion_guard: VerticalMotionGuard | None = None,
    yaw_envelope: YawCommandEnvelope | None = None,
) -> RuntimeLocalSafetyCommand:
    if not _finite_scalar(command_horizon_seconds) or not 0.05 <= command_horizon_seconds <= 1.0:
        raise ValueError("command horizon must be in [0.05, 1.0] seconds")
    if not _finite_scalar(prediction_step_seconds) or not 0.05 <= prediction_step_seconds <= 0.25:
        raise ValueError("prediction step must be in [0.05, 0.25] seconds")
    if type(validity_milliseconds) is not int or not 50 <= validity_milliseconds <= 2_000:
        raise ValueError("command validity must be in [50, 2000] milliseconds")
    if maximum_speed_mps is not None and (
        not _finite_scalar(maximum_speed_mps) or maximum_speed_mps <= 0.0
    ):
        raise ValueError("maximum speed must be finite and positive")
    if not _finite_scalar(required_clearance_m) or not 0 <= required_clearance_m <= 20:
        raise ValueError("required clearance must be finite and within [0, 20] metres")
    if type(generated_at_unix_ms) is not int or not 0 <= generated_at_unix_ms <= 2**63 - 2001:
        raise ValueError("command generation time must be a bounded non-negative integer")
    if any(
        type(flag) is not bool for flag in (tracking_recovery_active, model_navigation_authorized)
    ):
        raise ValueError("control authority and recovery flags must be explicit booleans")
    if navigation_control_authority not in {"route-fallback", "model-required", "bounded-hybrid"}:
        raise ValueError("unknown navigation control authority")
    if navigation_control_authority == "bounded-hybrid":
        if hybrid_lease is None or motion_guard is None:
            raise ValueError("HYBRID_REQUIRES_LEASE_AND_MOTION_GUARD")
        hybrid_lease = HybridControlLease.model_validate(hybrid_lease.model_dump())
        if (hybrid_lease.navigation_goal_id != navigation_goal_id or model_navigation_authorized
                or requested_control_intent is not None
                or generated_at_unix_ms < hybrid_lease.started_at_unix_ms
                or hybrid_lease.expires_at_unix_ms - generated_at_unix_ms < 50):
            raise ValueError("HYBRID_EVALUATION_CONTEXT_INVALID")
        maximum_speed_mps = min(maximum_speed_mps or .25, hybrid_lease.maximum_speed_mps)
        validity_milliseconds = min(validity_milliseconds,
            hybrid_lease.expires_at_unix_ms - generated_at_unix_ms)
    elif hybrid_lease is not None:
        raise ValueError("HYBRID_LEASE_REQUIRES_HYBRID_AUTHORITY")
    # 私有重验副本让几何计算与最后摘要消费同一观测，外部修改不能篡改已生成的目标或姿态。
    observation = RuntimeLocalSafetyObservation.model_validate(
        observation.model_dump(mode="python"), strict=True
    )
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    if observation.observed_at_unix_ms > generated_at_unix_ms:
        raise ValueError("observation is newer than command generation time")
    if requested_control_intent is not None:
        requested_control_intent = BodyFrameControlIntent.model_validate(
            requested_control_intent.model_dump(mode="python"), strict=True
        )
    requested_world_velocity = None
    if not _finite_scalar(route_yaw_rate_dps) or not -45 <= route_yaw_rate_dps <= 45:
        raise ValueError("route yaw rate must be finite and within [-45, 45] deg/s")
    if route_yaw_rate_dps and (
        requested_control_intent is not None or navigation_control_authority != "route-fallback"
    ):
        raise ValueError("route yaw and model control authorities cannot be mixed")
    requested_yaw_rate_dps = route_yaw_rate_dps
    if yaw_envelope is not None and requested_control_intent is None:
        raise ValueError('YAW_ENVELOPE_REQUIRES_MODEL_INTENT')
    planner_max_acceleration_mps2 = vehicle.max_acceleration_mps2
    planner_max_jerk_mps3 = 100.0
    if requested_control_intent is not None:
        if not model_navigation_authorized:
            raise ValueError("body-frame control requires model navigation authority")
        if observation.body_orientation_world_from_body is None:
            raise ValueError("body-frame control requires current body orientation")
        if requested_control_intent.model_call_id != model_call_id:
            raise ValueError("body-frame control model call does not match active lease")
        if requested_control_intent.task_reference_sha256 != model_path_sha256:
            raise ValueError("body-frame control path does not match active lease")
        if requested_control_intent.navigation_snapshot_sha256 != model_navigation_snapshot_sha256:
            raise ValueError("body-frame control input snapshot does not match active lease")
        if not (
            requested_control_intent.generated_at_unix_ms
            <= generated_at_unix_ms
            < requested_control_intent.valid_until_unix_ms
        ):
            raise ValueError("body-frame control intent is not live for this control tick")
        requested_world_velocity = body_to_world_enu(
            observation.body_orientation_world_from_body,
            Vector3(
                x=requested_control_intent.forward_velocity_mps,
                y=requested_control_intent.right_velocity_mps,
                z=requested_control_intent.up_velocity_mps,
            ),
        )
        requested_yaw_rate_dps = requested_control_intent.yaw_rate_dps
        if yaw_envelope is not None:
            from .yaw_command_envelope import shape_model_yaw

            requested_yaw_rate_dps = shape_model_yaw(requested_control_intent, yaw_envelope,
                now_unix_ms=generated_at_unix_ms)
        planner_max_acceleration_mps2 = min(
            vehicle.max_acceleration_mps2,
            requested_control_intent.maximum_acceleration_mps2,
        )
        planner_max_jerk_mps3 = requested_control_intent.maximum_jerk_mps3
    # Recovery owns a fixed corridor-rejoin target.  A constant 0.2 m/s cap
    # still commands 0.2 m/s at the 0.2 m position tolerance, so the aircraft
    # cannot also satisfy a 0.1 m/s settle gate: it overshoots, reverses and
    # repeats.  Taper the admissible speed as distance closes while retaining
    # acceleration limiting in the predictive planner.  The small positive
    # floor is only a schema-safe upper bound; the planner's target-distance
    # limit still converges to zero at the target.
    target_distance_m = math.dist(
        (
            observation.current_position_m.x,
            observation.current_position_m.y,
            observation.current_position_m.z,
        ),
        (
            observation.target_position_m.x,
            observation.target_position_m.y,
            observation.target_position_m.z,
        ),
    )
    # The vehicle asset describes a hardware envelope, not permission to
    # exceed the active mission track's qualified speed.  The runtime passes
    # the tighter track/profile bound here so an avoidance manoeuvre cannot
    # turn a 0.34 m/s precision route into a 1.2 m/s local correction.
    planner_max_speed_mps = min(
        vehicle.max_speed_mps,
        maximum_speed_mps if maximum_speed_mps is not None else vehicle.max_speed_mps,
    )
    if tracking_recovery_active:
        planner_max_speed_mps = min(
            planner_max_speed_mps,
            0.2,
            # A 0.45 proportional cap remains below 0.1 m/s at the 0.2 m
            # position gate, while avoiding an unnecessarily long crawl just
            # outside that gate under the contract's bounded settle timeout.
            max(0.04, target_distance_m * 0.45),
        )
    if (observation.motion_context_sha256
            != (motion_guard.sha256 if motion_guard is not None else None)):
        raise ValueError("RUNTIME_MOTION_CONTEXT_BINDING_MISMATCH")
    if motion_guard is not None:
        if motion_guard.context["position_m"] != [
                observation.current_position_m.x, observation.current_position_m.y,
                observation.current_position_m.z]:
            raise ValueError("RUNTIME_MOTION_POSITION_BINDING_MISMATCH")
        if motion_guard.context["geometry_sha256"] != sha256_json(static_primitives):
            raise ValueError("RUNTIME_MOTION_GEOMETRY_BINDING_MISMATCH")
        if motion_guard.clearance != required_clearance_m:
            raise ValueError("RUNTIME_MOTION_CLEARANCE_BINDING_MISMATCH")
        planner_max_acceleration_mps2 = motion_guard.acceleration_limit(
            planner_max_acceleration_mps2)
    request = LocalPlannerRequest(
        current_position_m=observation.current_position_m,
        current_velocity_mps=observation.current_velocity_mps,
        current_acceleration_mps2=observation.current_acceleration_world_enu_mps2,
        target_position_m=observation.target_position_m,
        dynamic_obstacles=observation.dynamic_obstacles,
        vehicle_radius_m=motion_guard.radius if motion_guard is not None else vehicle.body_radius_m,
        vehicle_height_m=motion_guard.height if motion_guard is not None else vehicle.body_height_m,
        max_speed_mps=planner_max_speed_mps,
        max_acceleration_mps2=planner_max_acceleration_mps2,
        max_jerk_mps3=planner_max_jerk_mps3,
        requested_velocity_mps=requested_world_velocity,
        requested_yaw_rate_dps=requested_yaw_rate_dps,
        required_clearance_m=required_clearance_m,
        prediction_horizon_seconds=RUNTIME_PREDICTION_HORIZON_SECONDS,
        prediction_step_seconds=prediction_step_seconds,
        perception_stream_healthy=observation.stream_healthy,
        perception_stream_age_seconds=observation.stream_age_seconds,
        localization_covariance_m2=observation.localization_covariance_m2,
    )
    source_ms = min(observation.observed_at_unix_ms,
                    generated_at_unix_ms - math.ceil(observation.stream_age_seconds * 1000))
    uncertainty_margin_m = localization_uncertainty_margin_m(observation.localization_covariance_m2)
    ego_speed = math.hypot(observation.current_velocity_mps.x,
                          observation.current_velocity_mps.y, observation.current_velocity_mps.z)
    obstacle_speed = max((math.hypot(o.velocity_mps.x, o.velocity_mps.y, o.velocity_mps.z)
                          for o in observation.dynamic_obstacles), default=0.)
    inherited_deadline = (requested_control_intent.valid_until_unix_ms
                          if requested_control_intent is not None
                          else generated_at_unix_ms + validity_milliseconds)

    # 功能：
    #   用同一原始时钟和不确定性检查每个候选，避免几何可行但根本没有发令时间的候选获胜。
    # 输入：
    #   velocity：预测 ENU 速度三元组；clearance：完整预测净空；yaw：实际输出偏航度每秒。
    # 输出：
    #   validity：不延长来源寿命的候选时效结论。
    def candidate_validity(velocity, clearance, yaw):
        validity = observation_validity(
            source_observed_at_unix_ms=source_ms, now_unix_ms=generated_at_unix_ms,
            inherited_deadline_unix_ms=inherited_deadline,
            clearance_margin_m=max(0., clearance - required_clearance_m),
            ego_speed_bound_mps=max(ego_speed, math.hypot(*velocity)),
            obstacle_speed_bound_mps=obstacle_speed,
            acceleration_bound_mps2=vehicle.max_acceleration_mps2,
            uncertainty_margin_m=uncertainty_margin_m,
            downstream_reserve_ms=LOCAL_DISPATCH_RESERVE_MS,
            angular_speed_bound_rad_s=abs(math.radians(yaw)))
        return validity

    first_rejected_budget = None

    # 功能：
    #   在候选排序之前拒绝时效不足的动作，保留首个被拒动作的诊断，最后输出仍重新检查。
    # 输入：
    #   velocity、clearance、yaw：候选动作及其几何证据。
    # 输出：
    #   allowed：候选当前具有实际控制预算时为真。
    def candidate_current(velocity, clearance, yaw):
        nonlocal first_rejected_budget
        validity = candidate_validity(velocity, clearance, yaw)
        allowed = validity.disposition == "control-eligible"
        if not allowed and first_rejected_budget is None:
            first_rejected_budget = (validity, max(0., clearance - required_clearance_m))
        return allowed

    # 已知负载资格不满足时无需枚举几十个必败候选，直接走同一条保留惯性证据的制动路径。
    if motion_guard is None:
        decision = predictive_safety_decision(request, static_primitives, candidate_check=candidate_current)
    elif motion_guard.issues:
        decision = predictive_braking_decision(request, static_primitives,
                                               issue_codes=motion_guard.issues)
    else:
        decision = predictive_safety_decision(request, static_primitives,
                                              motion_check=motion_guard.check, candidate_check=candidate_current)
    adaptive_deadline_ms = None
    observation_budget = None
    if (decision.action == "hold" and first_rejected_budget is not None
            and "CANDIDATE_TIME_BUDGET_EXHAUSTED" in decision.issue_codes):
        # 这是被拒运动的预算，不是给紧急制动重新授予运动资格。
        # 保留原刹停预测、候选数和其他拒绝原因，不能用另一次预测覆盖证据。
        validity, clearance_margin_m = first_rejected_budget
        observation_budget = ControlObservationBudget(
            source_observed_at_unix_ms=validity.source_observed_at_unix_ms,
            control_deadline_unix_ms=validity.control_deadline_unix_ms,
            disposition=validity.disposition, reason=validity.reason,
            clearance_margin_m=clearance_margin_m, uncertainty_margin_m=uncertainty_margin_m,
            downstream_reserve_ms=LOCAL_DISPATCH_RESERVE_MS)
        budget_issues = ["OBSERVATION_CONTROL_BUDGET_EXHAUSTED",
                         "OBSERVATION_" + validity.reason.upper().replace("-", "_")]
        decision = decision.model_copy(update={"issue_codes": list(dict.fromkeys(
            [*budget_issues, *decision.issue_codes]))[:32]})
    if decision.action != "hold":
        # Reuse the FULL action-specific collision forecast, not an encoder's
        # nearest range or "free reach" (neither proves a free swept volume).
        # This is a temporal-error veto in addition to that forecast. It does
        # not promote the planner's max acceleration to guaranteed braking.
        clearance_margin_m = max(0.0, decision.minimum_predicted_clearance_m - required_clearance_m)
        velocity = decision.selected_velocity_mps
        validity = candidate_validity((velocity.x, velocity.y, velocity.z),
            decision.minimum_predicted_clearance_m, decision.selected_yaw_rate_dps)
        observation_budget = ControlObservationBudget(
            source_observed_at_unix_ms=validity.source_observed_at_unix_ms,
            control_deadline_unix_ms=validity.control_deadline_unix_ms,
            disposition=validity.disposition,
            reason=validity.reason,
            clearance_margin_m=clearance_margin_m,
            uncertainty_margin_m=uncertainty_margin_m,
            downstream_reserve_ms=LOCAL_DISPATCH_RESERVE_MS,
        )
        if validity.disposition == "control-eligible":
            adaptive_deadline_ms = validity.control_deadline_unix_ms
        else:
            # Expired evidence cannot justify forward motion. Retain geometry
            # and its threat forecast when issuing the independent brake; do
            # not erase obstacles or claim this emergency action is risk-free.
            decision = predictive_braking_decision(
                request,
                static_primitives,
                issue_codes=[
                    "OBSERVATION_CONTROL_BUDGET_EXHAUSTED",
                    "OBSERVATION_" + validity.reason.upper().replace("-", "_"),
                ],
            )
    velocity = decision.selected_velocity_mps
    position = observation.current_position_m
    current_speed_mps = math.hypot(
        observation.current_velocity_mps.x,
        observation.current_velocity_mps.y,
        observation.current_velocity_mps.z,
    )
    effective_command_horizon_seconds = command_horizon_seconds
    if tracking_recovery_active and target_distance_m <= 0.5 and current_speed_mps <= 0.25:
        # The executor's telemetry wait repeats this position as a PX4
        # keepalive.  A 0.2 s carrot becomes only 1--2 cm after the recovery
        # speed taper and can consume the whole signed settle timeout even
        # though both gates are about to pass.  The selected velocity already
        # owns a collision-checked 3 s prediction; a bounded 0.8 s position
        # carrot lets PX4 finish the last few centimetres without changing the
        # early high-speed braking envelope or its velocity limit.
        effective_command_horizon_seconds = max(command_horizon_seconds, 0.8)
    command_position = Vector3(
        x=position.x + velocity.x * effective_command_horizon_seconds,
        y=position.y + velocity.y * effective_command_horizon_seconds,
        z=position.z + velocity.z * effective_command_horizon_seconds,
    )
    command_deadline_ms = generated_at_unix_ms + validity_milliseconds
    if adaptive_deadline_ms is not None:
        command_deadline_ms = min(command_deadline_ms, adaptive_deadline_ms)
    if requested_control_intent is not None:
        command_deadline_ms = min(command_deadline_ms, requested_control_intent.valid_until_unix_ms)
    command = RuntimeLocalSafetyCommand(
        observation_sha256=sha256_json(observation),
        observation_sequence=observation.sequence,
        generated_at_unix_ms=generated_at_unix_ms,
        valid_until_unix_ms=command_deadline_ms,
        source=observation.source,
        evaluated_target_position_m=observation.target_position_m,
        evaluated_body_orientation_world_from_body=observation.body_orientation_world_from_body,
        navigation_goal_id=navigation_goal_id,
        tracking_recovery_active=tracking_recovery_active,
        navigation_control_authority=navigation_control_authority,
        hybrid_lease=hybrid_lease,
        model_navigation_authorized=model_navigation_authorized,
        model_call_id=model_call_id,
        model_selected_candidate_id=model_selected_candidate_id,
        model_path_sha256=model_path_sha256,
        model_navigation_snapshot_sha256=model_navigation_snapshot_sha256,
        model_authority_reason=model_authority_reason,
        requested_control_intent=requested_control_intent,
        command_position_m=command_position,
        decision=decision,
        observation_budget=observation_budget,
    )
    return command
