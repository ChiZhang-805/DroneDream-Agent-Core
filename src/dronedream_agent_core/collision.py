"""Continuous conservative vehicle-envelope checks against real SDF semantics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from dronedream_plugin_sdk.protocol import MAX_JSON_BYTES, decode_json

from .contracts import GraphRoute, RouteClearanceReport, RouteCollision, Vector3
from .control_uncertainty import (
    PLANNING_LOCALIZATION_ALLOWANCE_M,
    LOCALIZATION_SIGMA_MULTIPLIER,
    finite_positive_number,
    localization_uncertainty_margin_m,
)
from .hashing import sha256_json
from .plugin_files import read_plugin_file

Point = tuple[float, float, float]

# Planning and validation must inflate the vehicle by the same amount.  Keep
# this value in the geometry authority instead of duplicating a magic number
# across route generators and safety plug-ins.
CONSERVATIVE_VEHICLE_ENVELOPE_SCALE = 1.2
PREFERRED_TRANSIT_CLEARANCE_M = 0.35
_MAX_ROUTE_SAMPLES = 250_000


# 功能：按共享预算反求足够的搜索净空，保守预留最大本地余量，避免搜索与验收使用两套假设。
# 输入：variance_m2：已核验的定位方差上界（平方米），不能缺失或用零表示未知。
# 输出：clearance_m：包络之外的最小搜索净空；仍须验证整条路线且不授予运动权限。
def localization_required_clearance_m(variance_m2: float) -> float:
    if not finite_positive_number(variance_m2) or variance_m2 > 10_000.0:
        raise ValueError("LOCALIZATION_VARIANCE_BOUND_INVALID")
    uncertainty = max(PLANNING_LOCALIZATION_ALLOWANCE_M,
                      localization_uncertainty_margin_m(variance_m2))
    # 0.15 是预算允许的本地避障余量上限；微小数值余量避免边界舍入导致反复失败。
    clearance_m = uncertainty + 0.15 + 0.04 + 1e-9
    return clearance_m


# 功能：
#   用同一公式拆分路线净空预算，失败也返回缺口；预算只是条件评估，不授予起飞权限。
# 输入：
#   clearance_m：机体包络之外的净空；variance_m2：可选原生定位方差上界。
# 输出：
#   assessment：规划假设、实测余量、缺口及该路线能够容纳的最大定位方差。
def assess_tracking_corridor_budget(clearance_m: float, variance_m2: float | None = None) -> dict:
    if not finite_positive_number(clearance_m):
        raise ValueError("route has no positive tracking corridor")
    local = max(0.05, min(0.15, clearance_m * 0.4))
    uncertainty = max(PLANNING_LOCALIZATION_ALLOWANCE_M,
                      localization_uncertainty_margin_m(variance_m2)
                      if variance_m2 is not None else PLANNING_LOCALIZATION_ALLOWANCE_M)
    available = clearance_m - local - uncertainty
    allowance = max(0.0, clearance_m - local - 0.04)
    try:
        maximum_variance = (allowance / LOCALIZATION_SIGMA_MULTIPLIER) ** 2
    except OverflowError as exc:
        raise ValueError("route tracking budget overflow") from exc
    if not math.isfinite(maximum_variance):
        raise ValueError("route tracking budget overflow")
    return {"minimum_route_clearance_m": clearance_m, "required_local_clearance_m": local,
            "reserved_uncertainty_m": uncertainty, "minimum_tracking_allowance_m": 0.04,
            "available_tracking_m": available, "clearance_deficit_m": max(0.0, 0.04 - available),
            "maximum_localization_variance_m2": maximum_variance,
            "uncertainty_basis": "live-native-localization" if variance_m2 is not None else "planning-assumption",
            "live_localization_checked": variance_m2 is not None, "funded": available >= 0.04}


# 功能：
#   将已证明的通道净空分配给安全余量、定位误差及跟踪偏差；没有实时协方差时只是规划假设。
# 输入：
#   minimum_route_clearance_m：已验证的路线最小净空，单位米。
#   localization_covariance_m2：可选实时定位协方差，单位平方米。
# 输出：
#   budget：本地净空、定位预留、跟踪偏差和恢复容差的共享预算。
def build_tracking_corridor_budget(
    minimum_route_clearance_m: float,
    *,
    localization_covariance_m2: float | None = None,
) -> dict[str, float]:
    assessment = assess_tracking_corridor_budget(minimum_route_clearance_m, localization_covariance_m2)
    required_clearance_m = assessment["required_local_clearance_m"]
    reserved_uncertainty_m = assessment["reserved_uncertainty_m"]
    available_tracking_m = assessment["available_tracking_m"]
    if available_tracking_m < 0.04:
        raise ValueError("route clearance cannot fund local safety and tracking margins")
    lag_limit_m = min(0.25, available_tracking_m * 0.80)
    rejoin_tolerance_m = max(0.025, lag_limit_m * 0.90)
    budget = {
        "minimum_route_clearance_m": minimum_route_clearance_m,
        "required_local_clearance_m": required_clearance_m,
        "reserved_uncertainty_m": reserved_uncertainty_m,
        "tracking_lag_limit_m": lag_limit_m,
        "tracking_rejoin_tolerance_m": rejoin_tolerance_m,
        "tracking_sample_rate_hz": 10.0,
    }
    return budget


# 功能：
#   有界读取当前普通语义文件，解析与摘要消费同一字节，拒绝重复键与非标准数值。
# 输入：
#   path：本次待验收的地图语义文件。
# 输出：
#   result：语义对象及实际读取字节的 SHA-256 二元组。
def _load(path: Path) -> tuple[dict[str, Any], str]:
    content = read_plugin_file(path, limit=MAX_JSON_BYTES)
    value = decode_json(content, limit=MAX_JSON_BYTES, node_limit=2_000_000)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    result = value, hashlib.sha256(content).hexdigest()
    return result


# 功能：
#   判断几何标量是否为可表示的有限实数，排除真假开关和隐式字符串转换。
# 输入：
#   value：准备参与几何计算的值。
# 输出：
#   valid：数值类型与有限性是否通过。
def _finite(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   验证唯一且完整的基础几何形状，保留名称等描述字段，独立保存参与计算的有限物理量。
# 输入：
#   primitive：箱体、球体、圆柱或胶囊的几何字典。
# 输出：
#   validated：坐标、尺寸和角度已验证的独立基元字典。
def _validated_primitive(primitive: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(primitive, dict):
        raise ValueError("collision primitive must be an object")
    fields = {
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "radius_m",
        "height_m",
        "length_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    }
    if not {"center_x", "center_y", "center_z"} <= primitive.keys() or any(
        not _finite(primitive[name]) for name in fields & primitive.keys()
    ):
        raise ValueError("collision primitive contains invalid physical values")
    sizes = {"size_x", "size_y", "size_z"}
    if sizes & primitive.keys():
        if not sizes <= primitive.keys() or {"radius_m", "height_m", "length_m"} & primitive.keys():
            raise ValueError("collision box dimensions are incomplete or ambiguous")
        dimensions = sizes
    else:
        if "radius_m" not in primitive or {"height_m", "length_m"} <= primitive.keys():
            raise ValueError("collision round primitive is incomplete or ambiguous")
        dimensions = {"radius_m", "height_m", "length_m"} & primitive.keys()
    if any(primitive[name] <= 0 for name in dimensions):
        raise ValueError("collision primitive dimensions must be positive")
    validated = dict(primitive)
    for name in fields & primitive.keys():
        validated[name] = float(primitive[name])
    return validated


# 功能：
#   合并同一地图快照中的静态和 Runtime 碰撞层，避免计划或复核漏掉运行端独有障碍。
# 输入：
#   semantic：已验证来源的地图语义对象。
# 输出：
#   primitives：保留两层全部障碍、尺寸已验证且数量有界的独立基元列表。
def planning_collision_primitives(semantic: dict[str, Any]) -> list[dict[str, Any]]:
    static = semantic.get("collision_primitives")
    runtime = semantic.get("runtime_collision_primitives", [])
    if not isinstance(static, list) or not 1 <= len(static) <= 100_000:
        raise ValueError("semantic artifact has no collision primitives")
    if not isinstance(runtime, list) or len(runtime) > 100_000:
        raise ValueError("runtime collision primitives invalid")
    if len(static) + len(runtime) > 100_000:
        raise ValueError("planning collision primitive budget exceeded")
    # 不按名称覆盖同名图元：同名但扩大了的物理碰撞包络也必须保留。
    primitives = [_validated_primitive(primitive) for primitive in (*static, *runtime)]
    return primitives


# 功能：
#   在去掉偏航的坐标系内计算倾斜箱体的保守半尺寸，覆盖滚转和俯仰而不缩小真实障碍。
# 输入：
#   primitive：尺寸与旋转角均已验证的箱体。
# 输出：
#   half_sizes：偏航局部坐标系内的三轴包围盒半尺寸，单位米。
def _box_half_sizes(primitive: dict[str, Any]) -> Point:
    x, y, z = (primitive[f"size_{axis}"] / 2 for axis in "xyz")
    roll, pitch = primitive.get("roll_rad", 0.0), primitive.get("pitch_rad", 0.0)
    if roll == 0.0 and pitch == 0.0:
        half_sizes = x, y, z
        return half_sizes
    sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
    # SDF 的旋转为 Rz(yaw) Ry(pitch) Rx(roll)；绝对旋转矩阵乘半尺寸得到外包络。
    half_sizes = (
        abs(cp) * x + abs(sp * sr) * y + abs(sp * cr) * z,
        abs(cr) * y + abs(sr) * z,
        abs(sp) * x + abs(cp * sr) * y + abs(cp * cr) * z,
    )
    if not all(math.isfinite(value) for value in half_sizes):
        raise ValueError("collision box envelope overflow")
    return half_sizes


# 功能：
#   校验基元并计算保守世界包围盒，统一运行时与跟踪索引的三轴旋转和膨胀规则。
# 输入：
#   primitive：包含中心、尺寸与可选 SDF 旋转的几何对象。
#   expansion_m：额外非负包围余量，单位米。
# 输出：
#   bounds：世界 ENU 坐标的最小点与最大点二元组。
def primitive_bounds(primitive: dict[str, Any], expansion_m: float = 0.0) -> tuple[Point, Point]:
    primitive = _validated_primitive(primitive)
    if not _finite(expansion_m) or expansion_m < 0:
        raise ValueError("collision bounds expansion must be finite and nonnegative")
    center = tuple(primitive[f"center_{axis}"] for axis in "xyz")
    if "size_x" in primitive:
        half_x, half_y, half_z = _box_half_sizes(primitive)
        yaw = primitive.get("yaw_rad", 0.0)
        extents = (
            abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y,
            abs(math.sin(yaw)) * half_x + abs(math.cos(yaw)) * half_y,
            half_z,
        )
    else:
        # 曲面使用外包球：不依赖轴方向，宁可多选也不能在宽阶段排除真实碰撞。
        radius = primitive["radius_m"]
        axial = primitive.get("length_m", primitive.get("height_m", 0.0)) / 2.0
        extents = (radius + axial, radius + axial, radius + axial)
    expanded = tuple(value + expansion_m for value in extents)
    bounds = (
        tuple(center[index] - expanded[index] for index in range(3)),
        tuple(center[index] + expanded[index] for index in range(3)),
    )
    if not all(math.isfinite(value) for point in bounds for value in point):
        raise ValueError("collision bounds overflow")
    return bounds


# 功能：
#   合并正交方向的有符号间隙，双方向均分离时取欧氏距离，重叠时保留保守负间隙。
# 输入：
#   first：水平方向净空，单位米。
#   second：竖直方向净空，单位米。
# 输出：
#   distance：合并后的有符号净空。
def _orthogonal(first: float, second: float) -> float:
    distance = math.hypot(first, second) if first > 0 and second > 0 else max(first, second)
    return distance


# 功能：
#   将 SDF 的滚转、俯仰、偏航旋转作用于局部 Z 轴，得到圆柱或胶囊的中心轴端点。
# 输入：
#   primitive：包含中心与姿态的已验证基元。
#   length：轴线总长，单位米。
# 输出：
#   endpoints：世界坐标中的轴线起点与终点。
def _axis_endpoints(primitive: dict[str, Any], length: float) -> tuple[Point, Point]:
    roll = float(primitive.get("roll_rad", 0.0))
    pitch = float(primitive.get("pitch_rad", 0.0))
    yaw = float(primitive.get("yaw_rad", 0.0))
    axis = (
        math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll),
        math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll),
        math.cos(pitch) * math.cos(roll),
    )
    center = (
        float(primitive["center_x"]),
        float(primitive["center_y"]),
        float(primitive["center_z"]),
    )
    half = length / 2
    endpoints = (
        tuple(center[index] - axis[index] * half for index in range(3)),
        tuple(center[index] + axis[index] * half for index in range(3)),
    )
    if not all(math.isfinite(value) for point in endpoints for value in point):
        raise ValueError("collision primitive axis overflow")
    return endpoints


# 功能：
#   求点到有限线段的最短距离，退化轴按点处理，投影比例限制在线段内。
# 输入：
#   point：需要查询的三维点。
#   start：线段起点。
#   end：线段终点。
# 输出：
#   distance：点至线段的非负距离，单位米。
def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    delta = tuple(end[index] - start[index] for index in range(3))
    length = math.hypot(*delta)
    if not math.isfinite(length):
        raise ValueError("collision axis length overflow")
    if length <= 1e-9:
        distance = math.dist(point, start)
    else:
        unit = tuple(component / length for component in delta)
        projection = sum((point[index] - start[index]) * unit[index] for index in range(3))
        if not math.isfinite(projection):
            raise ValueError("collision axis projection overflow")
        along = max(0.0, min(length, projection))
        closest = tuple(start[index] + along * unit[index] for index in range(3))
        distance = math.dist(point, closest)
    return distance


# 功能：
#   计算已验证基元与无人机保守包络的有符号净空；倾斜箱体外包，倾斜圆柱按胶囊外包。
# 输入：
#   point：无人机中心的有限世界坐标。
#   primitive：当前已验证障碍基元。
#   radius_m：无人机水平包络半径。
#   half_height_m：无人机包络半高。
# 输出：
#   clearance：保守有符号距离，负值表示包络重叠。
def _clearance(
    point: Point,
    primitive: dict[str, Any],
    *,
    radius_m: float,
    half_height_m: float,
) -> float:
    center = (
        float(primitive["center_x"]),
        float(primitive["center_y"]),
        float(primitive["center_z"]),
    )
    if "size_x" in primitive:
        half_x, half_y, half_z = _box_half_sizes(primitive)
        delta_x = point[0] - center[0]
        delta_y = point[1] - center[1]
        yaw = float(primitive.get("yaw_rad", 0.0))
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        outside_x = max(abs(local_x) - half_x, 0.0)
        outside_y = max(abs(local_y) - half_y, 0.0)
        horizontal = math.hypot(outside_x, outside_y) - radius_m
        vertical = abs(point[2] - center[2]) - half_z - half_height_m
        clearance = _orthogonal(horizontal, vertical)
        return clearance

    primitive_radius = float(primitive["radius_m"])
    conservative_radius = math.hypot(radius_m, half_height_m)
    if "length_m" in primitive:
        start, end = _axis_endpoints(primitive, float(primitive["length_m"]))
        clearance = (
            _point_segment_distance(point, start, end) - primitive_radius - conservative_radius
        )
        return clearance
    if "height_m" in primitive:
        roll = abs(float(primitive.get("roll_rad", 0.0)))
        pitch = abs(float(primitive.get("pitch_rad", 0.0)))
        height = float(primitive["height_m"])
        if roll <= 1e-12 and pitch <= 1e-12:
            horizontal = math.hypot(point[0] - center[0], point[1] - center[1])
            horizontal -= primitive_radius + radius_m
            vertical = abs(point[2] - center[2]) - height / 2 - half_height_m
            clearance = _orthogonal(horizontal, vertical)
            return clearance
        start, end = _axis_endpoints(primitive, height)
        clearance = (
            _point_segment_distance(point, start, end) - primitive_radius - conservative_radius
        )
        return clearance
    clearance = math.dist(point, center) - primitive_radius - conservative_radius
    return clearance


# 功能：
#   验证运行时点查询的坐标、尺寸及几何，返回有限保守净空，不授予运动授权。
# 输入：
#   point：无人机中心的三维米制坐标。
#   primitive：待检查的真实障碍基元。
#   radius_m：无人机水平包络半径，必须为正。
#   half_height_m：无人机包络半高，必须为正。
# 输出：
#   clearance：该位置与障碍之间的有限有符号净空。
def vehicle_clearance(
    point: Point,
    primitive: dict[str, Any],
    *,
    radius_m: float,
    half_height_m: float,
) -> float:
    if not all(finite_positive_number(value) for value in (radius_m, half_height_m)):
        raise ValueError("vehicle envelope dimensions must be positive")
    if (
        not isinstance(point, (tuple, list))
        or len(point) != 3
        or not all(_finite(v) for v in point)
    ):
        raise ValueError("collision point must contain three finite coordinates")
    primitive = _validated_primitive(primitive)
    clearance = _clearance(
        point,
        primitive,
        radius_m=radius_m,
        half_height_m=half_height_m,
    )
    if not math.isfinite(clearance):
        raise ValueError("collision clearance overflow")
    return clearance


# 功能：
#   一次验证并固定整批位置和障碍，用原标量几何逐点求最小净空，保留全部障碍与首个同分身份。
#   相邻位置完全相同时复用本次静态结果，不省略动态预测的时间点，也不跨快照复用。
# 输入：
#   points：有界的三维米制位置序列。
#   primitives：当前快照的完整障碍列表。
#   radius_m、half_height_m：机体水平半径与半高，单位米且严格为正。
# 输出：
#   samples：与位置顺序一致的（净空米数，障碍索引）列表；无近处障碍为（999.0，-1）。
def minimum_vehicle_clearances(points, primitives, *, radius_m: float, half_height_m: float) -> list[tuple[float, int]]:
    if not all(finite_positive_number(value) for value in (radius_m, half_height_m)):
        raise ValueError("vehicle envelope dimensions must be positive")
    if not isinstance(points, (tuple, list)) or len(points) > _MAX_ROUTE_SAMPLES:
        raise ValueError("collision point budget exceeded")
    if not isinstance(primitives, list) or len(primitives) > 100_000:
        raise ValueError("collision primitive budget exceeded")
    frozen_points = []
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 3 or not all(_finite(v) for v in point):
            raise ValueError("collision point must contain three finite coordinates")
        frozen_points.append(tuple(point))
    frozen_primitives = [_validated_primitive(primitive) for primitive in primitives]
    samples = []
    previous_point = None
    for point in frozen_points:
        if point == previous_point:
            samples.append(samples[-1])
            continue
        minimum, nearest = 999.0, -1
        for index, primitive in enumerate(frozen_primitives):
            # 只消除重复校验，不换用近似形状、减少预测点或改变标量运算顺序。
            clearance = _clearance(point, primitive, radius_m=radius_m, half_height_m=half_height_m)
            if not math.isfinite(clearance):
                raise ValueError("collision clearance overflow")
            if clearance < minimum:
                minimum, nearest = clearance, index
        samples.append((minimum, nearest))
        previous_point = point
    return samples


# 功能：
#   有界采样单段并保留两个端点，拒绝会导致无限分配的极小间隔或异常跨度。
# 输入：
#   start：已经验证的起点米制坐标。
#   end：已经验证的终点米制坐标。
#   interval_m：严格正且有限的最大采样间隔。
# 输出：
#   samples：包含两个端点的有限位置列表。
def _segment_samples(start: Point, end: Point, interval_m: float) -> list[Point]:
    if not finite_positive_number(interval_m):
        raise ValueError("sample interval must be finite and positive")
    proposed = math.dist(start, end) / interval_m
    if not math.isfinite(proposed) or proposed > _MAX_ROUTE_SAMPLES - 1:
        raise ValueError("route sample budget exceeded")
    count = max(1, math.ceil(proposed))
    samples = [
        tuple(start[axis] + (end[axis] - start[axis]) * index / count for axis in range(3))
        for index in range(count + 1)
    ]
    if any(not math.isfinite(value) for point in samples for value in point):
        raise ValueError("route sample coordinate overflow")
    return samples


# 功能：
#   1. 固定并验证路线及实际地图字节，对全部基元逐段计算保守净空下界。
#   2. 从采样距离扣除半个真实步距，覆盖采样点之间的空间；可能保守拒绝，不宣称观测到真实碰撞。
#   3. 保留全局及每段最小下界、见证采样点和最多一百条违规记录，拒绝无界工作量。
# 输入：
#   route：待检查的路线。
#   semantic_path：本次碰撞几何文件。
#   vehicle_diameter_m：无人机包络水平直径。
#   vehicle_height_m：无人机包络高度。
#   sample_interval_m：相邻检查点的最大间隔。
#   penetration_tolerance_m：明确允许的非负数值穿透容差。
# 输出：
#   report：绑定实际路线与地图摘要的保守验收报告，非飞行资格证明。
def validate_route_clearance(
    route: GraphRoute,
    semantic_path: Path,
    *,
    vehicle_diameter_m: float,
    vehicle_height_m: float,
    sample_interval_m: float = 0.1,
    penetration_tolerance_m: float = 0.001,
) -> RouteClearanceReport:
    if not all(finite_positive_number(value) for value in (vehicle_diameter_m, vehicle_height_m)):
        raise ValueError("vehicle envelope dimensions must be positive")
    if not finite_positive_number(sample_interval_m):
        raise ValueError("sample interval must be positive")
    if not _finite(penetration_tolerance_m) or penetration_tolerance_m < 0:
        raise ValueError("penetration tolerance must be finite and non-negative")
    route = GraphRoute.model_validate(route.model_dump(mode="python"), strict=True)
    semantic, semantic_sha256 = _load(semantic_path)
    primitives = planning_collision_primitives(semantic)
    route_points = [(point.x, point.y, point.z) for point in route.positions_m]
    segments = list(zip(route_points, route_points[1:], strict=False))
    if not segments:
        segments = [(route_points[0], route_points[0])]
    sample_count = 0
    minimum = math.inf
    minimum_point = route_points[0]
    minimum_primitive = ""
    collision_count = 0
    collisions: list[RouteCollision] = []
    segment_minimum_clearances_m: list[float] = []
    from time import monotonic
    from .model_harness.progress import progress_sink, report_progress
    observing = progress_sink.get() is not None
    started = monotonic()
    next_progress = started + 4
    if observing:
        report_progress("clearance",
            f"开始检查一条候选路线：共 {len(segments)} 个连续线段，对照 {len(primitives)} 个地图碰撞基元。无人机按直径 {vehicle_diameter_m:.2f} 米、高度 {vehicle_height_m:.2f} 米的三维包络计算，不把无人机当作没有体积的点。\n\n检查将沿路线以不大于 {sample_interval_m:.2f} 米的间隔采样，并从净空结果扣除半个实际采样步距，为采样点之间的空间保留保守余量。因此，两个路径点本身没有碰到障碍，并不意味着中间这一段可以穿过墙壁或天花板。\n\n本阶段会保留各段最小间隙、对应位置和潜在冲突。多条候选路线需要分别检查；尚未检查的部分不会提前计为通过。此处使用地图中的静态几何，不能代替起飞前和飞行中的实时传感器检查。",
            f"Checking {len(segments)} continuous segments against {len(primitives)} collision primitives, using a {vehicle_diameter_m:.2f} m wide and {vehicle_height_m:.2f} m high vehicle envelope.\n\nSamples are at most {sample_interval_m:.2f} m apart; half the actual sample spacing is deducted from clearance to conservatively cover gaps between samples. Clear endpoints alone do not establish a clear segment.\n\nEach candidate is checked separately, retaining minimum gaps and potential conflicts. Unchecked segments are not passed. Static map checks do not replace live sensor checks.")
    for segment_index, (start, end) in enumerate(segments):
        samples = _segment_samples(start, end, sample_interval_m)
        if (
            sample_count + len(samples) > _MAX_ROUTE_SAMPLES
            or (sample_count + len(samples)) * len(primitives) > 50_000_000
        ):
            raise ValueError("route collision query budget exceeded")
        # 有符号保守距离是 1-Lipschitz；任意段内点到最近端点最多半步距。
        penalty_m = math.dist(start, end) / (2 * (len(samples) - 1))
        segment_minimum = math.inf
        for point in samples:
            for primitive in primitives:
                clearance = (
                    _clearance(
                        point,
                        primitive,
                        radius_m=vehicle_diameter_m / 2,
                        half_height_m=vehicle_height_m / 2,
                    )
                    - penalty_m
                )
                if not math.isfinite(clearance):
                    raise ValueError("route clearance overflow")
                segment_minimum = min(segment_minimum, clearance)
                if clearance < minimum:
                    minimum, minimum_point = clearance, point
                    minimum_primitive = str(primitive.get("name", "unknown"))
                if clearance < -penetration_tolerance_m:
                    collision_count += 1
                    if len(collisions) < 100:
                        collisions.append(
                            RouteCollision(
                                sample_index=sample_count,
                                position_m=Vector3(x=point[0], y=point[1], z=point[2]),
                                primitive_name=str(primitive.get("name", "unknown")),
                                clearance_m=clearance,
                            )
                        )
            sample_count += 1
            # 只在观察器存在时低频读取时钟，不增加每个障碍物距离查询的开销。
            if observing and sample_count % 32 == 0 and monotonic() >= next_progress:
                elapsed = monotonic() - started
                next_progress = monotonic() + 4
                report_progress("clearance",
                    f"本条候选路线已检查 {sample_count} 个采样位置，正在处理第 {segment_index + 1}/{len(segments)} 段，计算已用 {elapsed:.1f} 秒。当前最小保守间隙为 {minimum:.3f} 米，发现 {collision_count} 次超出容差的包络冲突记录；这些是扫描至今的中间结果，最终结论须等待整条路线完成。",
                    f"Checked {sample_count} sample positions; processing segment {segment_index + 1}/{len(segments)}, elapsed {elapsed:.1f} s. Minimum conservative gap so far: {minimum:.3f} m; {collision_count} envelope conflicts beyond tolerance. These are intermediate observations, not the final route result.")
        if len(route_points) > 1:
            segment_minimum_clearances_m.append(segment_minimum)
    report = RouteClearanceReport(
        accepted=collision_count == 0,
        route_sha256=sha256_json(route),
        semantic_sha256=semantic_sha256,
        sample_interval_m=sample_interval_m,
        sample_count=sample_count,
        primitive_count=len(primitives),
        collision_count=collision_count,
        minimum_clearance_m=minimum,
        minimum_clearance_point=Vector3(x=minimum_point[0], y=minimum_point[1], z=minimum_point[2]),
        minimum_clearance_primitive=minimum_primitive,
        segment_minimum_clearances_m=segment_minimum_clearances_m,
        collisions=collisions,
    )
    if observing:
        report_progress("clearance", f"本条路线净空计算完成，用时 {monotonic() - started:.1f} 秒，共检查 {sample_count} 个位置。静态净空检查{'通过' if report.accepted else '未通过'}；接下来还要结合操作余量和候选路线评价确定是否采用。", f"Clearance calculation completed in {monotonic() - started:.1f} s for {sample_count} positions. Static check {'passed' if report.accepted else 'did not pass'}; operational margins and candidate evaluation still determine selection.")
    return report
