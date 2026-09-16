"""Deterministic short-horizon avoidance for static and tracked moving obstacles."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .collision import _box_half_sizes, _validated_primitive, vehicle_clearance
from .collision_batch import static_point_clearances
from .contracts import LocalPlannerRequest, PredictiveSafetyDecision, Vector3
from .motion_envelope import (
    SAFETY_SCORING_CLEARANCE_BONUS_M,
    interval_clearance_lower_bound,
    prediction_query_radius_m,
)

Point = tuple[float, float, float]


# 功能：
#   在公共入口严格重验请求并独立保存几何，避免评估期间外部更新改变控制依据。
# 输入：
#   request：当前机体状态、模型提议和预测限制。
#   static_primitives：最多十万个完整静态基元。
# 输出：
#   inputs：已固定的请求与几何二元组。
def _owned_inputs(request: LocalPlannerRequest, static_primitives: list[dict[str, Any]]):
    request = LocalPlannerRequest.model_validate(request.model_dump(mode="python"), strict=True)
    if not isinstance(static_primitives, list) or len(static_primitives) > 100_000:
        raise ValueError("LOCAL_SAFETY_GEOMETRY_INPUT_INVALID")
    primitives = [deepcopy(_validated_primitive(primitive)) for primitive in static_primitives]
    inputs = request, primitives
    return inputs


# 功能：
#   提取已验证向量的世界 ENU 分量，不转换单位。
# 输入：
#   value：三轴向量对象。
# 输出：
#   point：不可变的三轴数值元组。
def _point(value: Vector3) -> Point:
    point = value.x, value.y, value.z
    return point


# 功能：
#   稳定计算三维模长，无法表示时拒绝而非让无穷值把限幅方向归零。
# 输入：
#   value：有限三轴物理向量。
# 输出：
#   magnitude：向量模长。
def _magnitude(value: Point) -> float:
    magnitude = math.hypot(*value)
    if not math.isfinite(magnitude):
        raise ValueError("LOCAL_SAFETY_VECTOR_OVERFLOW")
    return magnitude


# 功能：
#   计算同一坐标系下的逐轴差，用于目标方向及速度变化。
# 输入：
#   first、second：同单位的三轴向量。
# 输出：
#   difference：前者减后者的三元组。
def _subtract(first: Point, second: Point) -> Point:
    difference = tuple(first[index] - second[index] for index in range(3))
    return difference


# 功能：
#   沿请求变化方向限制单步向量增量，不逐轴独立限幅而破坏整体加加速度约束。
# 输入：
#   current、requested：当前与请求向量。
#   maximum_delta：本步允许的最大向量变化模长。
# 输出：
#   limited：位于当前向量与请求向量之间的受限结果。
def _limit_delta(current: Point, requested: Point, maximum_delta: float) -> Point:
    delta = _subtract(requested, current)
    magnitude = _magnitude(delta)
    if magnitude <= maximum_delta or magnitude <= 1e-12:
        limited = requested
        return limited
    scale = maximum_delta / magnitude
    limited = tuple(current[index] + delta[index] * scale for index in range(3))
    return limited


# 功能：
#   由限速请求和加速度／加加速度约束计算下一步预测速度，保留实测超速造成的惯性。
# 输入：
#   request：已固定的机体状态、时间步及物理上限。
#   requested：三轴目标速度，单位米每秒。
# 输出：
#   reachable：下一时间步的预测速度，不代表已实际执行或保证无碰撞。
def _reachable_velocity(request: LocalPlannerRequest, requested: Point) -> Point:
    current = _point(request.current_velocity_mps)
    step = request.prediction_step_seconds
    requested = _limit_magnitude(requested, request.max_speed_mps)
    desired_acceleration = tuple((requested[index] - current[index]) / step for index in range(3))
    desired_acceleration = _limit_magnitude(desired_acceleration, request.max_acceleration_mps2)
    current_acceleration = (
        _point(request.current_acceleration_mps2)
        if request.current_acceleration_mps2 is not None
        else (0.0, 0.0, 0.0)
    )
    # Derivative estimates can spike when telemetry resumes.  A value outside
    # the vehicle envelope is not a physically reachable state for the command
    # generator, so clamp it before applying the jerk delta.  Clamping after
    # the jerk step would silently violate the very jerk bound being enforced.
    current_acceleration = _limit_magnitude(
        current_acceleration,
        request.max_acceleration_mps2,
    )
    reachable_acceleration = _limit_delta(
        current_acceleration,
        desired_acceleration,
        request.max_jerk_mps3 * step,
    )
    reachable_acceleration = _limit_magnitude(reachable_acceleration, request.max_acceleration_mps2)
    reachable = tuple(current[index] + reachable_acceleration[index] * step for index in range(3))
    return reachable


# 功能：
#   保持向量方向并约束整体模长，不将三轴各自上限误当作整体限速。
# 输入：
#   requested：请求的三轴向量。
#   maximum_magnitude：允许的正模长上限。
# 输出：
#   limited：原向量或等比例缩小后的向量。
def _limit_magnitude(requested: Point, maximum_magnitude: float) -> Point:
    magnitude = _magnitude(requested)
    if magnitude <= maximum_magnitude or magnitude <= 1e-12:
        limited = requested
        return limited
    scale = maximum_magnitude / magnitude
    limited = tuple(component * scale for component in requested)
    return limited


# 功能：
#   合并水平和竖直有符号净空，两个方向均分离时采用欧氏距离。
# 输入：
#   horizontal、vertical：水平与竖直净空，单位米。
# 输出：
#   clearance：保守组合净空。
def _orthogonal_clearance(horizontal: float, vertical: float) -> float:
    clearance = (
        math.hypot(horizontal, vertical)
        if horizontal > 0 and vertical > 0
        else max(horizontal, vertical)
    )
    return clearance


# 功能：
#   外推目标中心并按置信度和原始年龄膨胀障碍；丢失证据不能作为空闲空间。
# 输入：
#   point：待预测机体中心。
#   elapsed：从当前时刻开始的预测秒数。
#   request：机体包围尺寸。
#   obstacle：当前已验证的动态目标及原始年龄。
# 输出：
#   clearance：机体到动态包围体的净空；数值无效时为阻止放行的负值。
def _tracked_clearance(point, *, elapsed, request, obstacle):
    position, velocity = _point(obstacle.position_m), _point(obstacle.velocity_mps)
    predicted = tuple(
        position[index] + velocity[index] * (elapsed + obstacle.age_seconds) for index in range(3)
    )
    uncertainty = (1.0 - obstacle.confidence) * 0.5
    lost = obstacle.confidence < 0.35 or obstacle.age_seconds > 1.0
    # Lost evidence is not free space. Cover the motion back to the last known
    # location rather than letting an unreliable extrapolation move it away.
    uncertainty += obstacle.age_seconds * _magnitude(velocity) * (1.0 if lost else 0.25)
    horizontal = math.hypot(point[0] - predicted[0], point[1] - predicted[1])
    horizontal -= request.vehicle_radius_m + obstacle.radius_m + uncertainty
    vertical = abs(point[2] - predicted[2])
    vertical -= (request.vehicle_height_m + obstacle.height_m) / 2 + uncertainty
    if not math.isfinite(horizontal) or not math.isfinite(vertical):
        clearance = -1_000_000.0  # Unknown numerical envelope, never clear-space proof.
        return clearance
    clearance = _orthogonal_clearance(horizontal, vertical)
    return clearance


# 功能：
#   逐个比较当前预测点与动态目标包围体，保留最小净空对应的身份。
# 输入：
#   point：预测机体中心。
#   elapsed：预测相对秒数。
#   request：动态目标集合和机体尺寸。
# 输出：
#   result：最小净空与目标身份；没有目标时使用有限远距哨兵和空身份。
def _dynamic_clearance(
    point: Point,
    *,
    elapsed: float,
    request: LocalPlannerRequest,
) -> tuple[float, str | None]:
    minimum = 999.0
    threat: str | None = None
    for obstacle in request.dynamic_obstacles:
        clearance = _tracked_clearance(point, elapsed=elapsed, request=request, obstacle=obstacle)
        if clearance < minimum:
            minimum = clearance
            threat = obstacle.obstacle_id
    result = minimum, threat
    return result


# 功能：
#   依据两个端点和相对行程给出区间净空下界，避免只检查采样点而漏过穿行目标。
# 输入：
#   start、end：区间首尾机体坐标。
#   start_time、end_time：区间首尾预测秒数。
#   velocity：该段机体预测速度。
#   request：已校验的动态目标与机体包络。
# 输出：
#   result：该区间的最小保守净空及威胁身份。
def _dynamic_interval_clearance(start, end, *, start_time, end_time, velocity, request):
    minimum, threat = 999.0, None
    for obstacle in request.dynamic_obstacles:
        left = _tracked_clearance(start, elapsed=start_time, request=request, obstacle=obstacle)
        right = _tracked_clearance(end, elapsed=end_time, request=request, obstacle=obstacle)
        travel = math.dist(velocity, _point(obstacle.velocity_mps)) * (end_time - start_time)
        lower = interval_clearance_lower_bound(left, right, travel)
        if lower < minimum:
            minimum, threat = lower, obstacle.obstacle_id
    result = minimum, threat
    return result


# 功能：
#   生成有限组方向与速度候选，逐个应用物理限制并丢弃仍超速的下一步状态。
# 输入：
#   request：机体状态、目标或模型控制提议与上限。
#   vertical_biases：候选竖直方向偏置比例。
# 输出：
#   velocities：去重后处于速度包络内的候选速度。
def _candidate_velocities(
    request: LocalPlannerRequest,
    *,
    vertical_biases: tuple[float, ...] = (0.0,),
) -> list[Point]:
    position = _point(request.current_position_m)
    target = _point(request.target_position_m)
    to_target = _subtract(target, position)
    horizontal_distance = math.hypot(to_target[0], to_target[1])
    heading = math.atan2(to_target[1], to_target[0]) if horizontal_distance > 1e-9 else 0.0
    target_distance = max(_magnitude(to_target), 1e-9)
    requested_velocity = (
        _point(request.requested_velocity_mps)
        if request.requested_velocity_mps is not None
        else None
    )
    if requested_velocity is not None and _magnitude(requested_velocity) > 1e-9:
        heading = math.atan2(requested_velocity[1], requested_velocity[0])
        desired_speed = min(request.max_speed_mps, _magnitude(requested_velocity))
        desired_vertical = requested_velocity[2]
    else:
        desired_speed = min(request.max_speed_mps, target_distance)
        desired_vertical = desired_speed * to_target[2] / target_distance
    candidates: list[Point] = []
    for speed_ratio in (1.0, 0.7, 0.4):
        horizontal_speed = desired_speed * speed_ratio
        for angle_degrees in (0, -30, 30, -60, 60, -90, 90, 180):
            angle = heading + math.radians(angle_degrees)
            for vertical_bias in vertical_biases:
                requested = _limit_magnitude(
                    (
                        horizontal_speed * math.cos(angle),
                        horizontal_speed * math.sin(angle),
                        max(
                            -request.max_speed_mps,
                            min(
                                request.max_speed_mps,
                                desired_vertical + vertical_bias * desired_speed,
                            ),
                        ),
                    ),
                    request.max_speed_mps,
                )
                candidates.append(_reachable_velocity(request, requested))
    candidates.append(_reachable_velocity(request, (0.0, 0.0, 0.0)))
    # A speed projection here would violate the acceleration / jerk constraints
    # just calculated. Infeasible combinations cannot become motion candidates.
    velocities = [
        velocity
        for velocity in dict.fromkeys(candidates)
        if _magnitude(velocity) <= request.max_speed_mps
    ]
    return velocities


# 功能：
#   对完整静态基元集合计算机体包围体的最小净空与对应威胁。
# 输入：
#   point：预测机体中心。
#   request：机体半径及全高。
#   static_primitives：当前使用的静态几何。
# 输出：
#   result：最小静态净空及静态威胁身份。
def _static_clearance(
    point: Point,
    *,
    request: LocalPlannerRequest,
    static_primitives: list[dict[str, Any]],
) -> tuple[float, str | None]:
    static_clearance = 999.0
    static_name: str | None = None
    for primitive in static_primitives:
        clearance = vehicle_clearance(
            point,
            primitive,
            radius_m=request.vehicle_radius_m,
            half_height_m=request.vehicle_height_m / 2,
        )
        if clearance < static_clearance:
            static_clearance = clearance
            static_name = f"static:{primitive.get('name', 'unknown')}"
    result = static_clearance, static_name
    return result


# 功能：
#   在同一机体位置和时刻比较静态与动态威胁，不让较安全的一类遮盖另一类。
# 输入：
#   point：预测中心。
#   elapsed：预测秒数。
#   request：动态观测与机体参数。
#   static_primitives：静态几何集合。
# 输出：
#   result：两类威胁中较小的净空及身份。
def _point_clearance(point, *, elapsed, request, static_primitives):
    static_clearance, static_name = _static_clearance(
        point, request=request, static_primitives=static_primitives
    )
    dynamic_clearance, dynamic_name = _dynamic_clearance(point, elapsed=elapsed, request=request)
    result = min(
        (static_clearance, static_name),
        (dynamic_clearance, dynamic_name),
        key=lambda item: item[0],
    )
    return result


# 功能：
#   计算覆盖实际碰撞保守包围体的外包球，供搜索宽阶段使用。
# 输入：
#   primitive：已验证的静态基元。
# 输出：
#   radius：该基元保守外包球的半径，单位米。
def _primitive_enclosing_radius(primitive: dict[str, Any]) -> float:
    if "size_x" in primitive:
        radius = math.hypot(*_box_half_sizes(primitive))
        return radius
    radius = float(primitive["radius_m"])
    if "length_m" in primitive:
        radius += float(primitive["length_m"]) / 2.0
        return radius
    if "height_m" in primitive:
        radius += float(primitive["height_m"]) / 2.0
        return radius
    return radius


# 功能：
#   用可达距离与净空评分上限剔除不可能影响候选的远处几何；获选路径仍须全量重验。
# 输入：
#   request：当前状态、预测时域及机体包络。
#   static_primitives：完整静态基元。
# 输出：
#   relevant：候选评分需要考虑的几何子集。
def _candidate_search_primitives(
    request: LocalPlannerRequest,
    static_primitives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    origin = _point(request.current_position_m)
    maximum_lower_bound = prediction_query_radius_m(
        current_velocity_mps=_point(request.current_velocity_mps),
        maximum_speed_mps=request.max_speed_mps,
        horizon_seconds=request.prediction_horizon_seconds,
        vehicle_radius_m=request.vehicle_radius_m,
        vehicle_height_m=request.vehicle_height_m,
        required_clearance_m=request.required_clearance_m,
    )
    relevant: list[dict[str, Any]] = []
    for primitive in static_primitives:
        center = (
            float(primitive["center_x"]),
            float(primitive["center_y"]),
            float(primitive["center_z"]),
        )
        lower_bound = math.dist(origin, center) - _primitive_enclosing_radius(primitive)
        if lower_bound <= maximum_lower_bound:
            relevant.append(primitive)
    return relevant


# 功能：
#   检查当前点和每个采样区间的静态／动态净空下界，区间中点仅用于关联而非实测撞击时间。
# 输入：
#   request：机体参数与动态目标。
#   velocity：该预测的固定速度。
#   points、times：包含当前点的预测位置与相对时间。
#   static_samples：逐点静态净空和身份。
# 输出：
#   prediction：未来点列、最小净空、关联时间及威胁身份。
def _prediction_from_samples(request, velocity, points, times, static_samples):
    minimum, threat = static_samples[0]
    dynamic, dynamic_name = _dynamic_clearance(points[0], elapsed=0.0, request=request)
    if dynamic < minimum:
        minimum, threat = dynamic, dynamic_name
    time_to_minimum = 0.0
    speed = _magnitude(velocity)
    for index in range(1, len(points)):
        left, right = static_samples[index - 1], static_samples[index]
        clearance = (
            interval_clearance_lower_bound(
                left[0], right[0], speed * (times[index] - times[index - 1])
            )
            if left[1] is not None or right[1] is not None
            else 999.0
        )
        name = left[1] if left[0] <= right[0] else right[1]
        dynamic, dynamic_name = _dynamic_interval_clearance(
            points[index - 1],
            points[index],
            start_time=times[index - 1],
            end_time=times[index],
            velocity=velocity,
            request=request,
        )
        if dynamic < clearance:
            clearance, name = dynamic, dynamic_name
        if clearance < minimum:
            minimum, threat = clearance, name
            # An interval bound has no exact closest-approach time. Associate
            # it with the interval midpoint, never report it as measured impact.
            time_to_minimum = (times[index - 1] + times[index]) / 2
    prediction = points[1:], minimum, time_to_minimum, threat
    return prediction


# 功能：
#   构造含当前时刻的有限采样时间，末点精确截在预测窗口边界。
# 输入：
#   request：已校验的预测时域及步长。
# 输出：
#   times：从零开始递增的相对秒数。
def _prediction_times(request):
    count = max(1, math.ceil(request.prediction_horizon_seconds / request.prediction_step_seconds))
    times = [
        0.0,
        *(
            min(request.prediction_horizon_seconds, i * request.prediction_step_seconds)
            for i in range(1, count + 1)
        ),
    ]
    return times


# 功能：
#   按固定下一步速度投影未来位置，并以完整几何检查全部采样区间；不冒充实际动力学轨迹。
# 输入：
#   request：当前状态、时域、动态障碍和包络。
#   velocity：预测速度。
#   static_primitives：待验收的静态基元。
# 输出：
#   prediction：未来点列、最小净空、关联时间及威胁。
def _predict(
    request: LocalPlannerRequest,
    velocity: Point,
    static_primitives: list[dict[str, Any]],
) -> tuple[list[Point], float, float, str | None]:
    origin = _point(request.current_position_m)
    times = _prediction_times(request)
    points = [
        tuple(origin[axis] + velocity[axis] * elapsed for axis in range(3)) for elapsed in times
    ]
    samples = [
        _static_clearance(point, request=request, static_primitives=static_primitives)
        for point in points
    ]
    prediction = _prediction_from_samples(request, velocity, points, times, samples)
    return prediction


# 功能：
#   批量计算候选轨迹的静态净空，再分别叠加动态区间检查；获选候选另做全量验证。
# 输入：
#   request：固定机体状态与预测参数。
#   velocities：有界候选速度列表。
#   static_primitives：宽阶段筛选的静态基元。
# 输出：
#   predictions：与候选顺序一一对应的预测结果。
def _predict_candidates(request, velocities, static_primitives):
    origin = _point(request.current_position_m)
    times = _prediction_times(request)
    count = len(times)
    paths = [
        [tuple(origin[axis] + velocity[axis] * elapsed for axis in range(3)) for elapsed in times]
        for velocity in velocities
    ]
    predictions = []
    if not paths:
        return predictions
    clearances, nearest = static_point_clearances(
        [point for path in paths for point in path],
        static_primitives,
        radius_m=request.vehicle_radius_m,
        half_height_m=request.vehicle_height_m / 2,
    )
    for index, path in enumerate(paths):
        samples = []
        for step in range(count):
            offset = index * count + step
            clearance = float(clearances[offset])
            name = (
                f"static:{static_primitives[nearest[offset]].get('name', 'unknown')}"
                if nearest[offset] >= 0
                else None
            )
            samples.append((clearance, name))
        predictions.append(
            _prediction_from_samples(request, velocities[index], path, times, samples)
        )
    return predictions


# 功能：
#   生成零速度前馈保持指令，并单独报告保留惯性的预测；没有宣称飞机瞬间停下或必然无碰撞。
# 输入：
#   request：已验证的机体状态与控制限制。
#   static_primitives：固定的静态几何。
#   candidate_count：此前评估的候选数量。
#   issue_codes：触发保护保持的原因。
# 输出：
#   decision：零控制输出、制动预测和威胁信息组成的保护决策。
def _braking_hold(
    request: LocalPlannerRequest,
    static_primitives: list[dict[str, Any]],
    *,
    candidate_count: int,
    issue_codes: list[str],
) -> PredictiveSafetyDecision:
    velocity = _reachable_velocity(request, (0.0, 0.0, 0.0))
    path, clearance, minimum_time, threat = _predict(request, velocity, static_primitives)
    decision = PredictiveSafetyDecision(
        action="hold",
        selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        braking_prediction_velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
        selected_yaw_rate_dps=0.0,
        maximum_acceleration_mps2=request.max_acceleration_mps2,
        maximum_jerk_mps3=request.max_jerk_mps3,
        control_source="deterministic-brake",
        predicted_path_m=[Vector3(x=x, y=y, z=z) for x, y, z in path],
        minimum_predicted_clearance_m=clearance,
        time_to_minimum_clearance_seconds=minimum_time,
        threat_obstacle_id=threat,
        evaluated_candidate_count=max(1, candidate_count),  # The brake is evaluated too.
        issue_codes=issue_codes,
    )
    return decision


# 功能：
#   对显式安全否决生成保护保持，不伪造感知流故障作为原因。
# 输入：
#   request：待重验并固定的当前预测请求。
#   static_primitives：当前静态几何。
#   issue_codes：调用方提供的真实否决原因。
# 输出：
#   decision：无运动授权的保持指令及惯性预测。
def predictive_braking_decision(
    request: LocalPlannerRequest,
    static_primitives: list[dict[str, Any]],
    *,
    issue_codes: list[str],
) -> PredictiveSafetyDecision:
    request, static_primitives = _owned_inputs(request, static_primitives)
    decision = _braking_hold(request, static_primitives, candidate_count=1, issue_codes=issue_codes)
    return decision


# 功能：
#   1. 固定输入，检查感知、动态证据和速度包络，异常时保护保持。
#   2. 优先保留可行模型提议；危险附近搜索有界替代速度，并用完整几何重验最终结果。
#   3. 区分模型控制、确定性覆盖与保持，有限匀速预测不代表真实飞行或执行许可。
# 输入：
#   request：当前机体状态、局部模型提议、目标与安全限制。
#   static_primitives：当前完整静态基元集合。
# 输出：
#   decision：动作、速度、来源归因及可追踪的预测证据。
def predictive_safety_decision(
    request: LocalPlannerRequest,
    static_primitives: list[dict[str, Any]],
) -> PredictiveSafetyDecision:
    request, static_primitives = _owned_inputs(request, static_primitives)
    unhealthy_codes: list[str] = []
    if not request.perception_stream_healthy:
        unhealthy_codes.append("PERCEPTION_STREAM_UNHEALTHY")
    if request.perception_stream_age_seconds > request.maximum_perception_age_seconds:
        unhealthy_codes.append("PERCEPTION_STREAM_STALE")
    if request.localization_covariance_m2 > request.maximum_localization_covariance_m2:
        unhealthy_codes.append("LOCALIZATION_UNCERTAIN")
    if any(o.confidence < 0.35 or o.age_seconds > 1.0 for o in request.dynamic_obstacles):
        # Revalidate at this public boundary too: callers other than fusion must
        # not bypass lost-track handling with a cached stream_healthy=True flag.
        unhealthy_codes.append("DYNAMIC_TRACK_EVIDENCE_UNRELIABLE")
    current = _point(request.current_velocity_mps)
    if _magnitude(current) > request.max_speed_mps:
        unhealthy_codes.append("MEASURED_SPEED_OUTSIDE_CONTROL_ENVELOPE")
    if unhealthy_codes:
        decision = _braking_hold(
            request,
            static_primitives,
            candidate_count=1,
            issue_codes=[*unhealthy_codes, "HOLD_UNTIL_FRESH_LOCAL_WORLD"],
        )
        return decision

    position = _point(request.current_position_m)
    target = _point(request.target_position_m)
    target_delta = _subtract(target, position)
    target_distance = max(_magnitude(target_delta), 1e-9)
    target_unit = tuple(component / target_distance for component in target_delta)
    desired_speed = min(request.max_speed_mps, target_distance)
    requested_nominal = (
        _point(request.requested_velocity_mps)
        if request.requested_velocity_mps is not None
        else tuple(component * desired_speed for component in target_unit)
    )
    nominal_velocity = _reachable_velocity(request, requested_nominal)
    if _magnitude(nominal_velocity) > request.max_speed_mps:
        decision = _braking_hold(
            request,
            static_primitives,
            candidate_count=1,
            issue_codes=["NEXT_STEP_SPEED_OUTSIDE_CONTROL_ENVELOPE", "HOLD_UNTIL_REACHABLE_SPEED"],
        )
        return decision
    nominal_control_source = (
        "local-model-body-control" if request.requested_velocity_mps is not None else "route-target"
    )
    nominal_path, nominal_clearance, nominal_time, nominal_threat = _predict(
        request,
        nominal_velocity,
        static_primitives,
    )
    if nominal_clearance >= request.required_clearance_m + 0.08:
        # The active model proposal (or explicitly authorized route fallback)
        # owns nominal motion. In a clearly safe
        # corridor, evaluating dozens of alternative 3-D velocities cannot
        # change the only authorized result (continue), but it can consume a
        # substantial fraction of the short-lived command lease. Preserve the
        # full candidate search near any safety margin or predicted obstacle;
        # take this fast path only with an explicit 8 cm clearance buffer.
        decision = PredictiveSafetyDecision(
            action="continue",
            selected_velocity_mps=Vector3(
                x=nominal_velocity[0],
                y=nominal_velocity[1],
                z=nominal_velocity[2],
            ),
            selected_yaw_rate_dps=request.requested_yaw_rate_dps,
            maximum_acceleration_mps2=request.max_acceleration_mps2,
            maximum_jerk_mps3=request.max_jerk_mps3,
            control_source=nominal_control_source,
            predicted_path_m=[Vector3(x=x, y=y, z=z) for x, y, z in nominal_path],
            minimum_predicted_clearance_m=nominal_clearance,
            time_to_minimum_clearance_seconds=nominal_time,
            threat_obstacle_id=nominal_threat,
            evaluated_candidate_count=1,
            issue_codes=[],
        )
        return decision
    origin_clearance, origin_threat = _point_clearance(
        position,
        elapsed=0.0,
        request=request,
        static_primitives=static_primitives,
    )
    evaluations: list[tuple[float, Point, list[Point], float, float, str | None, bool]] = []
    candidates = _candidate_velocities(request)
    search_primitives = _candidate_search_primitives(request, static_primitives)

    # 功能：
    #   评分安全候选；静态近边界恢复需保持正净空、满足起点容差且最终达到目标余量。
    # 输入：
    #   candidate_set：本次待评分的候选速度。
    # 输出：
    #   None：将通过检查的候选追加到当前调用的 evaluations。
    def evaluate_candidates(candidate_set: list[Point]) -> None:
        predictions = _predict_candidates(request, candidate_set, search_primitives)
        for velocity, prediction in zip(candidate_set, predictions, strict=True):
            path, clearance, minimum_time, threat = prediction
            recovering_static_clearance = False
            if clearance < request.required_clearance_m:
                # A vehicle can begin a controller cycle inside the preferred
                # margin without being in collision (for example immediately
                # above its launch pad). Freezing there is a deadlock. Permit
                # only a monotonic, collision-free escape that reaches the full
                # margin within the prediction horizon; never apply this
                # exception while a moving obstacle is present.
                end_clearance, end_threat = _point_clearance(
                    path[-1],
                    elapsed=request.prediction_horizon_seconds,
                    request=request,
                    static_primitives=search_primitives,
                )
                recovering_static_clearance = bool(
                    origin_clearance > 0.0
                    and origin_clearance < request.required_clearance_m
                    and origin_threat is not None
                    and origin_threat.startswith("static:")
                    and not request.dynamic_obstacles
                    and clearance > 0.0
                    and clearance >= max(0.001, origin_clearance - 0.005)
                    and end_clearance >= request.required_clearance_m
                    and (end_threat is None or end_threat.startswith("static:"))
                )
                if not recovering_static_clearance:
                    continue
            progress = sum(velocity[index] * target_unit[index] for index in range(3))
            smoothness = _magnitude(_subtract(velocity, current))
            clearance_reward = min(
                clearance, request.required_clearance_m + SAFETY_SCORING_CLEARANCE_BONUS_M
            )
            score = progress * 3.0 + clearance_reward * 0.8 - smoothness * 0.25
            evaluations.append(
                (
                    score,
                    velocity,
                    path,
                    clearance,
                    minimum_time,
                    threat,
                    recovering_static_clearance,
                )
            )

    evaluate_candidates(candidates)
    primary_has_progressing_safe_candidate = any(
        sum(evaluation[1][index] * target_unit[index] for index in range(3)) > 0.01
        for evaluation in evaluations
    )
    if not primary_has_progressing_safe_candidate:
        expanded = _candidate_velocities(
            request,
            vertical_biases=(0.35, -0.35),
        )
        existing = set(candidates)
        expanded = [velocity for velocity in expanded if velocity not in existing]
        candidates.extend(expanded)
        evaluate_candidates(expanded)

    if not evaluations:
        decision = _braking_hold(
            request,
            static_primitives,
            candidate_count=len(candidates),
            issue_codes=["NO_SAFE_LOCAL_VELOCITY", "HOLD_AND_REQUEST_REPLAN"],
        )
        return decision

    _score, velocity, path, clearance, minimum_time, threat, recovering_clearance = max(
        evaluations, key=lambda item: item[0]
    )
    # Re-evaluate the selected path with the full semantic geometry.  The
    # broad phase above is conservative, so this should not change safety, but
    # it keeps the reported minimum clearance and threat tied to all source
    # primitives rather than to an optimization subset.
    path, clearance, minimum_time, threat = _predict(
        request,
        velocity,
        static_primitives,
    )
    if clearance < request.required_clearance_m and recovering_clearance:
        end_clearance, end_threat = _point_clearance(
            path[-1],
            elapsed=request.prediction_horizon_seconds,
            request=request,
            static_primitives=static_primitives,
        )
        recovering_clearance = bool(
            origin_clearance > 0.0
            and origin_clearance < request.required_clearance_m
            and origin_threat is not None
            and origin_threat.startswith("static:")
            and not request.dynamic_obstacles
            and clearance > 0.0
            and clearance >= max(0.001, origin_clearance - 0.005)
            and end_clearance >= request.required_clearance_m
            and (end_threat is None or end_threat.startswith("static:"))
        )
    if clearance < request.required_clearance_m and not recovering_clearance:
        # A future primitive type with an incorrect enclosing bound must fail
        # closed instead of leaking an unsafe candidate through the fast path.
        decision = _braking_hold(
            request,
            static_primitives,
            candidate_count=len(candidates),
            issue_codes=["FULL_GEOMETRY_RECHECK_FAILED", "HOLD_AND_REQUEST_REPLAN"],
        )
        return decision
    speed = _magnitude(velocity)
    nominal_velocity_preserved = math.dist(velocity, nominal_velocity) <= 1e-9
    nominal_path_hazard = nominal_clearance < request.required_clearance_m
    forward = sum(velocity[index] * target_unit[index] for index in range(3))
    cosine = forward / max(speed, 1e-9)
    near_required_margin = clearance < request.required_clearance_m + 0.08
    dynamic_threat = threat is not None and not threat.startswith("static:")
    if recovering_clearance:
        action = "slow"
    elif speed < 0.05 and (nominal_path_hazard or dynamic_threat):
        decision = _braking_hold(
            request,
            static_primitives,
            candidate_count=len(candidates),
            issue_codes=["LOCAL_SAFETY_HOLD"],
        )
        return decision
    elif cosine < math.cos(math.radians(20)) and nominal_path_hazard:
        action = "replan"
    elif near_required_margin or (dynamic_threat and speed < desired_speed * 0.75):
        action = "slow"
    else:
        # A low reachable velocity caused only by the current acceleration
        # state is not itself a hazard or permission to replace the proposal;
        # doing so creates a self-reinforcing crawl.  The same applies when a
        # hold sample has already settled at its target: zero velocity in a
        # clear local world is the expected nominal state, not a repair that
        # should freeze schedule time until the bounded repair timer expires.
        # Local control overrides only for an actual clearance constraint or
        # a moving obstacle.
        action = "continue"
    decision = PredictiveSafetyDecision(
        action=action,
        selected_velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
        selected_yaw_rate_dps=(
            request.requested_yaw_rate_dps
            if action in {"continue", "slow"} and nominal_velocity_preserved
            else 0.0
        ),
        maximum_acceleration_mps2=request.max_acceleration_mps2,
        maximum_jerk_mps3=request.max_jerk_mps3,
        control_source=(
            nominal_control_source
            if action in {"continue", "slow"} and nominal_velocity_preserved
            else (
                "deterministic-safety-override"
                if action in {"continue", "slow"}
                else "deterministic-brake"
            )
        ),
        predicted_path_m=[Vector3(x=x, y=y, z=z) for x, y, z in path],
        minimum_predicted_clearance_m=clearance,
        time_to_minimum_clearance_seconds=minimum_time,
        threat_obstacle_id=threat,
        evaluated_candidate_count=len(candidates),
        issue_codes=(
            ["STATIC_CLEARANCE_RECOVERY"]
            if recovering_clearance
            else ([] if action == "continue" else [f"LOCAL_SAFETY_{action.upper()}"])
        ),
    )
    return decision
