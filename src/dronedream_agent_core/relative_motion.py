"""Analytic constant-velocity contact features, not actuator authorization."""

from __future__ import annotations

import math

from .contracts import Vector3
from .control_uncertainty import finite_positive_number


# 功能：
#   固定并严格重验相对运动向量及预测时域，避免被修改的合同进入数值计算。
# 输入：
#   position：世界 ENU 相对位置，单位米。
#   velocity：世界 ENU 相对速度，单位米每秒。
#   horizon_seconds：正且有限的预测窗口。
# 输出：
#   vectors：独立持有的相对位置与速度对象二元组。
def _motion_vectors(position: Vector3, velocity: Vector3, horizon_seconds: float):
    if not finite_positive_number(horizon_seconds):
        raise ValueError("RELATIVE_MOTION_HORIZON_INVALID")
    vectors = (
        Vector3.model_validate(position.model_dump(mode="python"), strict=True),
        Vector3.model_validate(velocity.model_dump(mode="python"), strict=True),
    )
    return vectors


# 功能：
#   计算一维匀速到达边界的时间；先相减保留近表面精度，仅溢出时改为分别相除。
# 输入：
#   boundary：目标边界坐标。
#   position：当前坐标。
#   velocity：非零相对速度。
# 输出：
#   time_seconds：有符号到达时间，超出浮点范围的时间由调用方裁剪至预测窗口。
def _boundary_time(boundary: float, position: float, velocity: float) -> float:
    displacement = boundary - position
    time_seconds = (
        displacement / velocity
        if math.isfinite(displacement)
        else boundary / velocity - position / velocity
    )
    return time_seconds


# 功能：
#   计算相对点匀速进入直立圆柱的首个时刻；水平和竖直必须同时重叠，不代替载具安全检查。
# 输入：
#   relative_position：世界 ENU 相对位置，单位米。
#   relative_velocity：世界 ENU 相对速度，单位米每秒。
#   radius_m：圆柱正半径，单位米。
#   height_m：圆柱正全高，单位米。
#   horizon_seconds：正预测窗口，单位秒。
# 输出：
#   contact_seconds：首次接触秒数；窗口内无接触时为窗口长度。
def cylinder_contact_seconds(
    relative_position: Vector3,
    relative_velocity: Vector3,
    *,
    radius_m: float,
    height_m: float,
    horizon_seconds: float = 30.0,
) -> float:
    p, v = _motion_vectors(relative_position, relative_velocity, horizon_seconds)
    if not finite_positive_number(radius_m) or not finite_positive_number(height_m):
        raise ValueError("RELATIVE_MOTION_CYLINDER_INVALID")
    contact_seconds = horizon_seconds
    speed = math.hypot(v.x, v.y)
    if not math.isfinite(speed):
        raise ValueError("RELATIVE_MOTION_SPEED_OVERFLOW")
    if speed == 0.0:
        if math.hypot(p.x, p.y) > radius_m:
            return contact_seconds
        xy_start, xy_end = 0.0, horizon_seconds
    else:
        ux, uy = v.x / speed, v.y / speed
        try:
            along = math.fsum((p.x * ux, p.y * uy))
            perpendicular = abs(math.fsum((p.x * uy, -p.y * ux)))
        except OverflowError as error:
            raise ValueError("RELATIVE_MOTION_PROJECTION_OVERFLOW") from error
        if perpendicular > radius_m:
            return contact_seconds
        # 用弦长求交代替 b²-4ac，避免远处小目标的半径被大数相减抹掉。
        ratio = perpendicular / radius_m
        half_chord = radius_m * math.sqrt((1.0 - ratio) * (1.0 + ratio))
        xy_start = _boundary_time(-half_chord, along, speed)
        xy_end = _boundary_time(half_chord, along, speed)
    half_height = height_m / 2
    if v.z == 0.0:
        if abs(p.z) > half_height:
            return contact_seconds
        z_start, z_end = 0.0, horizon_seconds
    else:
        z_start, z_end = sorted(
            (_boundary_time(-half_height, p.z, v.z), _boundary_time(half_height, p.z, v.z))
        )
    # Collision needs horizontal and vertical overlap at the same time. A
    # horizontal crossing alone is not contact with a finite-height cylinder.
    start, end = max(0.0, xy_start, z_start), min(horizon_seconds, xy_end, z_end)
    contact_seconds = start if start <= end else horizon_seconds
    return contact_seconds


# 功能：
#   将相对位置投影到匀速轨迹，并在当前时刻与预测终点之间求最近中心距离。
# 输入：
#   relative_position：世界 ENU 相对位置，单位米。
#   relative_velocity：世界 ENU 相对速度，单位米每秒。
#   horizon_seconds：正预测窗口，单位秒。
# 输出：
#   distance_m：预测窗口内的最近中心距离，不是表面净空或运动许可。
def closest_center_distance_m(
    relative_position: Vector3,
    relative_velocity: Vector3,
    *,
    horizon_seconds: float = 30.0,
) -> float:
    p, v = _motion_vectors(relative_position, relative_velocity, horizon_seconds)
    speed = math.hypot(v.x, v.y, v.z)
    if not math.isfinite(speed):
        raise ValueError("RELATIVE_MOTION_SPEED_OVERFLOW")
    closest_time = 0.0
    if speed > 0.0:
        try:
            projection = math.fsum((p.x * (v.x / speed), p.y * (v.y / speed), p.z * (v.z / speed)))
        except OverflowError as error:
            raise ValueError("RELATIVE_MOTION_PROJECTION_OVERFLOW") from error
        closest_time = min(horizon_seconds, max(0.0, -projection / speed))
    distance_m = math.hypot(
        p.x + v.x * closest_time, p.y + v.y * closest_time, p.z + v.z * closest_time
    )
    if not math.isfinite(distance_m):
        raise ValueError("RELATIVE_MOTION_DISTANCE_OVERFLOW")
    return distance_m
