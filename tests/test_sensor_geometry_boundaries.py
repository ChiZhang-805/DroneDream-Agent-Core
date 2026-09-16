"""Rotation and occupancy boundary regressions; no physical-flight qualification."""

import math

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, Vector3
from dronedream_agent_core.occupancy_collision import (
    bounded_occupied_cell_boxes,
    box_fully_inside,
    merged_occupied_cell_boxes,
)
from dronedream_agent_core.realtime_feature_encoders import (
    _rotate as feature_rotate,
)
from dronedream_agent_core.realtime_feature_encoders import body_to_world_enu, world_enu_to_body
from dronedream_agent_core.runtime_sensor_contracts import _rotate_vector as camera_rotate
from dronedream_agent_core.sensor_bridge import _rotate as bridge_rotate


# 功能：
#   核对三条实际调用入口接受微小四元数归一化误差时，仍保持旋转方向和米制长度。
# 输入：
#   rotate：桥接、特征或相机方向判断所使用的旋转入口。
#   scale：契约容差内的四元数缩放，负值同时验证等价符号。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rotate", [bridge_rotate, feature_rotate, camera_rotate])
@pytest.mark.parametrize("scale", [.9992, 1., 1.0008, -1.0008])
def test_accepted_quaternion_tolerance_preserves_metric_rotation(rotate, scale):
    half = math.sqrt(.5) * scale
    orientation = QuaternionWxyz(w=half, x=0., y=0., z=half)
    original = orientation.model_dump()
    result = rotate(orientation, (10., 2., 3.))
    assert result == pytest.approx((-2., 10., 3.), abs=1e-12)
    assert math.hypot(*result) == pytest.approx(math.hypot(10., 2., 3.), abs=1e-12)
    assert orientation.model_dump() == original


# 功能：
#   核对旋转可表示的大数向量时，不因中间交叉积放大而溢出。
# 输入：
#   rotate：三条生产旋转入口之一。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rotate", [bridge_rotate, feature_rotate, camera_rotate])
def test_rotation_avoids_intermediate_overflow(rotate):
    orientation = QuaternionWxyz(w=0., x=0., y=0., z=1.)
    result = rotate(orientation, (1e308, 0., 0.))
    assert result == (-1e308, 0., 0.)


# 功能：
#   拒绝绕过模型构造器后产生的非法姿态，不能把零四元数当作无旋转。
# 输入：
#   rotate：三条生产旋转入口之一。
#   component：替换单位四元数标量分量的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rotate", [bridge_rotate, feature_rotate, camera_rotate])
@pytest.mark.parametrize("component", [0., 2., float("nan"), float("inf")])
def test_rotation_rejects_mutated_invalid_orientation(rotate, component):
    orientation = QuaternionWxyz(w=1., x=0., y=0., z=0.).model_copy(update={"w": component})
    with pytest.raises(ValueError, match="QUATERNION"):
        rotate(orientation, (1., 0., 0.))


# 功能：
#   验证公共前右上控制轴经世界 ENU 往返转换后不改变幅度或左右方向。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_public_control_round_trip_preserves_scaled_orientation():
    half = math.sqrt(.5) * 1.0008
    orientation = QuaternionWxyz(w=half, x=0., y=0., z=half)
    body = Vector3(x=.8, y=-.2, z=.1)
    world = body_to_world_enu(orientation, body)
    recovered = world_enu_to_body(orientation, world)
    assert world.model_dump() == pytest.approx({"x": -.2, "y": .8, "z": .1})
    assert recovered.model_dump() == pytest.approx(body.model_dump(), abs=1e-12)


# 功能：
#   在极大但有限的中心坐标下核对占用盒排序不崩溃、不丢掉任何输入占用格。
# 输入：
#   center：足以使旧平方距离溢出的查询中心。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("center", [(1e200, 0., 0.), (-1e308, 1e308, 0.)])
def test_finite_far_center_keeps_a_conservative_occupancy_cover(center):
    source = [(0, 0, 0), (4, 0, 0), (8, 0, 0)]
    result = bounded_occupied_cell_boxes(source, center_in_cells=center, limit=2)
    assert len(result) == 2
    for cell in source:
        assert any(all(bounds[a] <= cell[a] < bounds[a + 3] for a in range(3))
                   for bounds, _ in result)


# 功能：
#   拒绝非三维整数格索引，防止分数、布尔或畸形数据被静默合并成错误占用区域。
# 输入：
#   cell：不满足整数格契约的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("cell", [(.5, 0, 0), (True, 0, 0), (0, 0), (0, 0, 0, 0),
                                 ("0", 0, 0), (float("nan"), 0, 0), (2**53, 0, 0)])
def test_occupancy_rejects_noninteger_or_unrepresentable_cells(cell):
    with pytest.raises(ValueError, match="cell"):
        merged_occupied_cell_boxes([cell])


# 功能：
#   检查布尔几何不能作为删除感知证据的完整包含证明。
# 输入：
#   field：被布尔值替换的几何分量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["center_x", "size_x", "yaw_rad"])
def test_boolean_geometry_never_proves_redundancy(field):
    inner = {"shape": "box", "center_x": 0., "center_y": 0., "center_z": 0.,
             "size_x": 1., "size_y": 1., "size_z": 1., "yaw_rad": 0.}
    outer = {**inner, "size_x": 4., "size_y": 4., "size_z": 4.}
    inner[field] = True
    assert not box_fully_inside(inner, outer)
