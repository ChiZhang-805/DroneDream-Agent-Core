"""Immutable semantic broad phase; compile bounds once, never truncate near hazards."""

import copy

import numpy as np

from .collision import _finite


# 功能：
#   验证三元数值向量并固定为独立双精度数组，拒绝多余维度和隐式字符串或布尔转换。
# 输入：
#   value：Python 三元序列或 NumPy 三元数值数组。
# 输出：
#   vector：有限 float64 三元向量。
def _vector(value):
    if isinstance(value, np.ndarray):
        if value.shape != (3,) or value.dtype.kind not in "fiu":
            raise ValueError("STATIC_GEOMETRY_VECTOR_INVALID")
    elif (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, float, np.integer, np.floating))
            for item in value
        )
    ):
        raise ValueError("STATIC_GEOMETRY_VECTOR_INVALID")
    try:
        with np.errstate(over="raise", invalid="raise"):
            vector = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError("STATIC_GEOMETRY_VECTOR_INVALID") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("STATIC_GEOMETRY_VECTOR_INVALID")
    return vector


class StaticGeometryIndex:
    """Cache conservative axis-aligned bounds without borrowing caller state."""

    # 功能：
    #   固定最多十万个基元，并且只计算一次经过严格校验的包围盒，不借用回调返回的数组。
    # 输入：
    #   primitives：本次使用的基元列表。
    #   bounds：为一个基元返回保守世界包围盒的回调。
    # 输出：
    #   None：初始化独立几何与索引数组。
    def __init__(self, primitives, *, bounds):
        if not isinstance(primitives, list) or len(primitives) > 100_000 or not callable(bounds):
            raise ValueError("STATIC_GEOMETRY_INPUT_INVALID")
        self._primitives = copy.deepcopy(primitives)
        pairs = []
        try:
            for primitive in self._primitives:
                low, high = bounds(primitive)
                pairs.append((_vector(low), _vector(high)))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("STATIC_GEOMETRY_BOUNDS_INVALID") from error
        self._low = np.stack([low for low, _ in pairs]) if pairs else np.empty((0, 3))
        self._high = np.stack([high for _, high in pairs]) if pairs else np.empty((0, 3))
        if (self._low > self._high).any():
            raise ValueError("STATIC_GEOMETRY_BOUNDS_INVALID")

    # 功能：
    #   按到包围盒的真实距离筛选候选，保留大障碍及稳定顺序；超容量拒绝而非截断。
    # 输入：
    #   point：世界三轴米坐标。
    #   radius_m：由调用方运动范围确定的正查询半径，单位米。
    #   maximum_primitives：允许返回的最多基元数量。
    # 输出：
    #   nearby：独立复制的候选基元列表，不是碰撞结论或运动许可。
    def nearby(self, point, *, radius_m: float = 6.0, maximum_primitives: int = 256):
        try:
            position = _vector(point)
        except ValueError as error:
            raise ValueError("STATIC_GEOMETRY_QUERY_INVALID") from error
        if (
            not _finite(radius_m)
            or radius_m <= 0
            or type(maximum_primitives) is not int
            or not 0 < maximum_primitives <= 100_000
        ):
            raise ValueError("STATIC_GEOMETRY_QUERY_INVALID")
        # hypot 避免有限距离的平方溢出。仅极远距离相减可能溢出为正无穷，
        # 它严格大于任何已验证的有限半径，不会被选成近邻。
        with np.errstate(over="ignore", invalid="raise"):
            outside = np.maximum(np.maximum(self._low - position, position - self._high), 0.0)
            distance = np.hypot.reduce(outside, axis=1)
        selected = np.flatnonzero(distance <= radius_m)
        if len(selected) > maximum_primitives:
            # The caller must reduce its domain or decline motion. Silently
            # taking the first N obstacles is not a conservative safety query.
            raise ValueError("STATIC_GEOMETRY_LOCAL_CAPACITY_EXCEEDED")
        # Stable ties keep replay deterministic; copies keep callers from
        # changing geometry behind the already-compiled bounds.
        selected = selected[np.argsort(distance[selected], kind="stable")]
        nearby = [copy.deepcopy(self._primitives[index]) for index in selected]
        return nearby
