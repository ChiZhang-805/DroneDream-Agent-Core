import math
from unittest.mock import Mock

import numpy as np
import pytest
from test_collision_batch import mixed_shapes

from dronedream_agent_core.static_geometry_index import StaticGeometryIndex
from scripts.runtime_depth_safety_worker import _primitive_bounds


# 功能：
#   验证空间索引与逐个包围盒距离筛选一致，查询不重建包围盒也不借用外部几何。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_fixed_bounds_match_scalar_selection_and_are_not_recompiled():
    shapes = mixed_shapes(805)
    bounds = Mock(side_effect=_primitive_bounds)
    index = StaticGeometryIndex(shapes, bounds=bounds)
    assert bounds.call_count == len(shapes)
    for point in ((0, 0, 0), (1, -4, 3), (-5, -1, 2)):
        expected = []
        for primitive in shapes:
            low, high = _primitive_bounds(primitive)
            distance = math.sqrt(
                sum(max(low[a] - point[a], point[a] - high[a], 0.0) ** 2 for a in range(3))
            )
            if distance <= 3.0:
                expected.append((distance, primitive))
        expected.sort(key=lambda pair: pair[0])
        assert index.nearby(point, radius_m=3.0) == [p for _, p in expected]
    assert bounds.call_count == len(shapes)
    original = index.nearby((0, 0, 0))
    shapes.clear()
    query = index.nearby((0, 0, 0))
    query[0]["center_x"] = 1000
    assert index.nearby((0, 0, 0)) == original


# 功能：
#   验证附近障碍超出容量时拒绝整个查询，而非截断后放行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nearby_obstacle_capacity_never_discards_unchecked_geometry():
    shapes = [dict(center_x=0.0, center_y=0.0, center_z=0.0, radius_m=1.0)] * 257
    index = StaticGeometryIndex(shapes, bounds=_primitive_bounds)
    with pytest.raises(ValueError, match="CAPACITY_EXCEEDED"):
        index.nearby((0, 0, 0))
    assert len(index.nearby((0, 0, 0), maximum_primitives=257)) == 257
    assert index.nearby((100, 100, 100)) == []
    with pytest.raises(ValueError, match="QUERY_INVALID"):
        index.nearby((float("nan"), 0, 0))


# 功能：
#   验证有限大距离比较不会因平方溢出，把远方障碍当成半径内障碍。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_large_finite_distance_does_not_collapse_to_infinity_comparison():
    shape = {"center_x": 1.0e200, "center_y": 0.0, "center_z": 0.0, "radius_m": 1.0}
    index = StaticGeometryIndex([shape], bounds=_primitive_bounds)
    assert index.nearby((0.0, 0.0, 0.0), radius_m=1.0e199) == []


# 功能：
#   验证运行时索引能选中旋转后实际进入查询半径的长箱体。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_bounds_include_tilted_long_box():
    shape = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 1.0,
        "size_y": 1.0,
        "size_z": 12.0,
        "pitch_rad": math.pi / 2,
    }
    index = StaticGeometryIndex([shape], bounds=_primitive_bounds)
    assert index.nearby((5.0, 0.0, 0.0), radius_m=0.1) == [shape]


# 功能：
#   验证查询点的布尔与字符串不能通过数组强制转换冒充坐标。
# 输入：
#   point：畸形查询向量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("point", [(True, 0.0, 0.0), ("0", 0.0, 0.0)])
def test_query_position_is_not_coerced(point):
    index = StaticGeometryIndex([], bounds=_primitive_bounds)
    with pytest.raises(ValueError, match="QUERY_INVALID"):
        index.nearby(point)


# 功能：
#   验证半径开关值不能冒充实际一米距离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_boolean_radius_rejected():
    with pytest.raises(ValueError, match="QUERY_INVALID"):
        StaticGeometryIndex([], bounds=_primitive_bounds).nearby((0.0, 0.0, 0.0), radius_m=True)


# 功能：
#   验证包围盒坐标必须正好是三元数值向量，不能将多余维度展平或强制转换。
# 输入：
#   low：错误的包围盒下界。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("low", [[[0.0, 0.0, 0.0]], ["0", 0.0, 0.0], [False, 0.0, 0.0]])
def test_bounds_are_exact_numeric_vectors(low):
    with pytest.raises(ValueError, match="BOUNDS_INVALID"):
        StaticGeometryIndex([{}], bounds=lambda primitive: (low, (1.0, 1.0, 1.0)))


# 功能：
#   验证回调反复使用同一数组时，每个包围盒仍绑定自己的数值快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reused_callback_arrays_are_copied_per_primitive():
    low, high = np.zeros(3), np.zeros(3)

    # 功能：
    #   以复用缓冲区模拟原地工作的包围盒提供者。
    # 输入：
    #   primitive：本次计算中心。
    # 输出：
    #   bounds：下一次调用将被覆盖的两份数组。
    def reuse_bounds(primitive):
        low[:] = (primitive["x"] - 0.1, -0.1, -0.1)
        high[:] = (primitive["x"] + 0.1, 0.1, 0.1)
        bounds = low, high
        return bounds

    shapes = [{"x": 0.0}, {"x": 20.0}]
    index = StaticGeometryIndex(shapes, bounds=reuse_bounds)
    low[:] = high[:] = 1000.0
    assert index.nearby(np.array([0.0, 0.0, 0.0], dtype=np.float32), radius_m=1.0) == [shapes[0]]
    assert index.nearby((20.0, 0.0, 0.0), radius_m=1.0) == [shapes[1]]


# 功能：
#   验证等距离障碍保持输入顺序，以及 NumPy 真假值或复数向量不能通过类型检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stable_ties_and_numpy_type_boundaries():
    shapes = [
        {"center_x": x, "center_y": 0.0, "center_z": 0.0, "radius_m": 0.5} for x in (2.0, -2.0)
    ]
    index = StaticGeometryIndex(shapes, bounds=_primitive_bounds)
    assert index.nearby((0.0, 0.0, 0.0)) == shapes
    for point in (np.array([False, False, False]), np.array([0.0j, 0.0j, 0.0j])):
        with pytest.raises(ValueError, match="QUERY_INVALID"):
            index.nearby(point)
