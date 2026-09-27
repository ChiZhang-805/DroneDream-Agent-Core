"""Synthetic camera geometry tests, not physical visibility or flight qualification."""

import math

import pytest

from dronedream_agent_core.contracts import QuaternionWxyz, Vector3
from dronedream_agent_core.training.observation_envelope import box_frustum_margin


# 功能：
#   构造相机前方的小盒，供视锥边界测试使用。
# 输入：
#   overrides：替换单项几何或标定参数。
# 输出：
#   margin：待测试的有符号覆盖余量。
def margin(**overrides):
    parameters = dict(camera_position=Vector3(x=0., y=0., z=0.),
        orientation_world_from_sensor=QuaternionWxyz(w=1., x=0., y=0., z=0.),
        box_minimum=Vector3(x=4., y=-.3, z=-.5), box_maximum=Vector3(x=5., y=.3, z=.5),
        horizontal_fov_rad=math.pi/2, vertical_fov_rad=math.pi/2,
        minimum_range_m=.2, maximum_range_m=10.)
    parameters.update(overrides)
    result = box_frustum_margin(**parameters)
    return result


# 功能：
#   验证检查对象是完整盒而非中心，并区分左右、上下、后方和远面的越界。
# 输入：
#   low、high、visible：米制上下界和预期是否完整位于视锥内。
# 输出：
#   None：几何判定与解析值不符时失败。
@pytest.mark.parametrize('low,high,visible', [
    ((4., -.3, -.5), (5., .3, .5), True),
    ((4., -.3, 0.), (5., .3, 4.8), False),
    ((4., -.3, -4.8), (5., .3, 0.), False),
    ((4., -4.8, -.5), (5., 0., .5), False),
    ((-5., -.3, -.5), (-4., .3, .5), False),
    ((9., -.3, -.5), (11., .3, .5), False),
])
def test_complete_box_not_only_center(low, high, visible):
    value = margin(box_minimum=Vector3(**dict(zip('xyz', low, strict=True))),
                   box_maximum=Vector3(**dict(zip('xyz', high, strict=True))))
    assert (value > 0) is visible


# 功能：
#   验证整体平移和相机旋转后保持同一米制覆盖余量。
# 输入：
#   无。
# 输出：
#   None：坐标变换或前向轴约定错误时失败。
def test_world_pose_and_sensor_orientation():
    first = margin()
    rotated = margin(camera_position=Vector3(x=10., y=20., z=30.),
        orientation_world_from_sensor=QuaternionWxyz(w=math.sqrt(.5), x=0., y=0., z=math.sqrt(.5)),
        box_minimum=Vector3(x=9.7, y=24., z=29.5), box_maximum=Vector3(x=10.3, y=25., z=30.5))
    assert rotated == pytest.approx(first)


# 功能：
#   拒绝非法相机标定与反向盒边界，不通过数值截断伪造覆盖。
# 输入：
#   overrides：非法字段。
# 输出：
#   None：非法输入明确报错。
@pytest.mark.parametrize('overrides', [
    {'horizontal_fov_rad': True}, {'vertical_fov_rad': float('nan')},
    {'horizontal_fov_rad': math.pi}, {'maximum_range_m': .1},
    {'maximum_range_m': 10**400}, {'minimum_range_m': 0.},
    {'box_minimum': Vector3(x=6., y=0., z=0.)},
])
def test_invalid_bounds(overrides):
    with pytest.raises(ValueError, match='FRUSTUM_BOUND_INVALID'):
        margin(**overrides)
