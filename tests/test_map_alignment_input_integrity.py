"""Raw observation types must survive validation before numeric fitting."""

import numpy as np
import pytest
from test_local_map_alignment import box, wall_points

from dronedream_agent_core.local_map_alignment import (
    MapAlignmentLimits,
    MapSurfaceIndex,
    _points,
    fit_map_translation,
)
from dronedream_agent_core.local_pose_alignment import fit_map_pose, rotation_exp_and_left_jacobian


# 功能：
#   布尔、文本和复数不是有效测距位置，不能在转成 float64 后才检查。
# 输入：
#   points：错误原始类型的点阵。
# 输出：
#   None：原始类型被拒绝时通过。
@pytest.mark.parametrize(
    "points",
    [
        [[True, 0.0, 0.0]],
        [["1", 0, 0]],
        [[1 + 0j, 0, 0]],
        np.ones((2, 3), dtype=bool),
        np.array([["1", "0", "0"]]),
        np.array([[1, 0, 0]], dtype=object),
    ],
)
def test_points_reject_lossy_type_coercion(points):
    with pytest.raises(ValueError, match="POINTS_INVALID"):
        _points(points)


# 功能：
#   计算必须使用独立点快照，不能在调用者后来修改数组后改变既有输入。
# 输入：
#   无；构造普通浮点数组。
# 输出：
#   None：快照没有共享输入内存时通过。
def test_points_detach_existing_numpy_storage():
    source = np.array([[1.0, 2.0, 3.0]])
    frozen = _points(source)
    source[0, 0] = 99.0
    assert frozen[0, 0] == 1.0
    assert not np.shares_memory(source, frozen)


# 功能：
#   射线起点与点云一样需要严格数值类型，不能通过提前强转隐去非法来源。
# 输入：
#   origin：非法的共享射线起点。
# 输出：
#   None：平移和姿态两条入口都拒绝时通过。
@pytest.mark.parametrize("origin", [[False, 0.0, 0.0], ["0", 0, 0], [0j, 0, 0]])
def test_origins_preserve_original_type_validation(origin):
    points = wall_points()
    index = MapSurfaceIndex([box()])
    with pytest.raises(ValueError, match="ORIGINS_INVALID"):
        fit_map_translation(points, index, sensor_origins_world_m=origin)
    with pytest.raises(ValueError, match="ORIGINS_INVALID"):
        fit_map_pose(
            points, index, sensor_origins_world_m=origin, reference_position_world_m=[0.0, 0.0, 0.0]
        )


# 功能：
#   验证公开旋转接口不会把布尔、文本或复数解释为真实旋转向量。
# 输入：
#   vector：非法的旋转输入。
# 输出：
#   None：旋转计算前已拒绝时通过。
@pytest.mark.parametrize("vector", [[True, 0.0, 0.0], ["0", 0, 0], [0j, 0, 0]])
def test_rotation_requires_actual_scalar_values(vector):
    with pytest.raises(ValueError, match="ROTATION_VECTOR_INVALID"):
        rotation_exp_and_left_jacobian(vector)


# 功能：
#   即使冻结配置被绕过而篡改，也不能把无限迭代或非法距离传进实际拟合。
# 输入：
#   无；仅对测试配置使用 object.__setattr__ 注入错误。
# 输出：
#   None：匹配入口和平移入口都重新拒绝时通过。
def test_changed_limit_object_is_revalidated_at_entry():
    limits = MapAlignmentLimits()
    object.__setattr__(limits, "maximum_iterations", True)
    index = MapSurfaceIndex([box()])
    points = wall_points()
    with pytest.raises(ValueError, match="LIMITS_INVALID"):
        index.match(points, limits, sensor_origins_world_m=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="LIMITS_INVALID"):
        fit_map_translation(points, index, sensor_origins_world_m=[0.0, 0.0, 0.0], limits=limits)
