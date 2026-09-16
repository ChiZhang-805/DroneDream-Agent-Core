"""Vectorized point queries with the scalar vehicle-envelope geometry semantics.

No map or path samples are removed. Primitive ordering (including equal-clearance
ties) is preserved; callers still independently recheck any selected control.
"""

from __future__ import annotations

import math

import numpy as np

from .collision import _axis_endpoints, _box_half_sizes, _validated_primitive
from .control_uncertainty import finite_positive_number


# 功能：
#   逐障碍分块计算全部查询点的保守净空，保留原顺序及同分时首个障碍，不省略地图内容。
# 输入：
#   points：三维有限米制坐标列表或数值矩阵。
#   primitives：当前地图中的几何字典列表。
#   radius_m：无人机水平包络半径。
#   half_height_m：无人机包络半高。
# 输出：
#   result：逐点最小净空与障碍索引；999 米和 -1 表示没有障碍改善初始上限。
def static_point_clearances(points, primitives, *, radius_m: float, half_height_m: float):
    if not isinstance(primitives, list) or len(primitives) > 100_000:
        raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
    primitives = [_validated_primitive(primitive) for primitive in primitives]
    # 先检查原类型，避免字符串、复数或布尔在转换 float64 时被静默接受。
    if isinstance(points, np.ndarray):
        valid_points = points.dtype.kind in "fiu" and points.ndim == 2 and points.shape[1] == 3
    else:
        valid_points = isinstance(points, (list, tuple)) and all(
            isinstance(point, (list, tuple, np.ndarray))
            and len(point) == 3
            and all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in point
            )
            for point in points
        )
    if not valid_points or len(points) > 250_000:
        raise ValueError("BATCH_COLLISION_INPUT_INVALID")
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or len(values) > 250_000
        or not all(finite_positive_number(v) for v in (radius_m, half_height_m))
    ):
        raise ValueError("BATCH_COLLISION_INPUT_INVALID")
    minimum = np.full(len(values), 999.0, dtype=np.float64)
    indices = np.full(len(values), -1, dtype=np.int64)
    enclosing = math.hypot(radius_m, half_height_m)
    # Bound intermediate arrays for unusually long paths as well as ordinary
    # short-horizon control. Only one primitive is broadcast at a time.
    for first in range(0, len(values), 1024):
        block = values[first : first + 1024]
        distances, nearest = minimum[first : first + 1024], indices[first : first + 1024]
        for index, primitive in enumerate(primitives):
            center = np.asarray([float(primitive[f"center_{axis}"]) for axis in "xyz"])
            if not np.isfinite(center).all():
                raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
            delta = block - center
            if "size_x" in primitive:
                half_sizes = _box_half_sizes(primitive)
                yaw = float(primitive.get("yaw_rad", 0.0))
                if not all(math.isfinite(v) and v > 0 for v in half_sizes) or not math.isfinite(
                    yaw
                ):
                    raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
                cosine, sine = math.cos(yaw), math.sin(yaw)
                x = cosine * delta[:, 0] + sine * delta[:, 1]
                y = -sine * delta[:, 0] + cosine * delta[:, 1]
                horizontal = (
                    np.hypot(
                        np.maximum(np.abs(x) - half_sizes[0], 0.0),
                        np.maximum(np.abs(y) - half_sizes[1], 0.0),
                    )
                    - radius_m
                )
                vertical = np.abs(delta[:, 2]) - half_sizes[2] - half_height_m
                clearance = _orthogonal(horizontal, vertical)
            else:
                radius = float(primitive["radius_m"])
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
                if "length_m" in primitive or "height_m" in primitive:
                    length = float(primitive.get("length_m", primitive.get("height_m")))
                    roll, pitch = (
                        float(primitive.get(name, 0.0)) for name in ("roll_rad", "pitch_rad")
                    )
                    if (
                        not math.isfinite(length)
                        or length <= 0
                        or not math.isfinite(roll)
                        or not math.isfinite(pitch)
                    ):
                        raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
                    if "length_m" not in primitive and abs(roll) <= 1e-12 and abs(pitch) <= 1e-12:
                        horizontal = np.hypot(delta[:, 0], delta[:, 1]) - (radius + radius_m)
                        vertical = np.abs(delta[:, 2]) - length / 2 - half_height_m
                        clearance = _orthogonal(horizontal, vertical)
                    else:
                        start, end = (np.asarray(p) for p in _axis_endpoints(primitive, length))
                        axis = end - start
                        squared = float(np.sum(axis * axis))
                        ratio = (
                            np.zeros(len(block))
                            if squared <= 1e-18
                            else np.clip(np.sum((block - start) * axis, axis=1) / squared, 0.0, 1.0)
                        )
                        clearance = np.linalg.norm(block - (start + ratio[:, None] * axis), axis=1)
                        clearance -= radius + enclosing
                else:
                    clearance = np.linalg.norm(delta, axis=1) - radius - enclosing
            if not np.isfinite(clearance).all():
                raise ValueError("BATCH_COLLISION_GEOMETRY_INVALID")
            better = clearance < distances
            distances[better] = clearance[better]
            nearest[better] = index
    result = minimum, indices
    return result


# 功能：
#   按标量净空的相同规则合并水平与竖直间隙，双正时用欧氏距离，其余保留最大有符号间隙。
# 输入：
#   horizontal：每个查询点的水平间隙数组。
#   vertical：每个查询点的竖直间隙数组。
# 输出：
#   clearance：逐点合并后的有符号净空数组。
def _orthogonal(horizontal, vertical):
    clearance = np.where(
        (horizontal > 0) & (vertical > 0),
        np.hypot(horizontal, vertical),
        np.maximum(horizontal, vertical),
    )
    return clearance
