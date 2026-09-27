"""Synthetic teacher attention checks; not evidence of a learned control policy."""

import pytest

from dronedream_agent_core.contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from dronedream_agent_core.simulation_teacher import teacher_heading_rate, teacher_observation_target


# 功能：
#   构造仅供单元测试的动态目标，不替代真实运动见证。
# 输入：
#   identity：目标标识；x、y：水平米坐标；age：原始观测年龄。
# 输出：
#   obstacle：低速、有效置信度的合成目标。
def obstacle(identity='person', x=4., y=0., age=0.):
    obstacle = DynamicObstacleObservation(obstacle_id=identity, position_m=Vector3(x=x, y=y, z=1.),
        velocity_mps=Vector3(x=0., y=.1, z=0.), radius_m=.3, height_m=1., confidence=.9, age_seconds=age)
    return obstacle


# 功能：
#   验证路线向左转时示范仍观察前方活动目标，原导航目的地不被替换或修改。
# 输入：
#   无。
# 输出：
#   None：断言偏航参考与导航目标分别保留。
def test_route_turn_does_not_turn_camera_away_from_tracked_obstacle():
    position, goal = Vector3(x=0., y=0., z=1.), Vector3(x=0., y=4., z=1.)
    measured = obstacle()
    original = measured.model_dump()
    target = teacher_observation_target(position=position, goal=goal, obstacles=[measured])
    yaw = teacher_heading_rate(orientation=QuaternionWxyz(w=1., x=0., y=0., z=0.),
        position=position, goal=target, maximum_rate_dps=20.)
    assert yaw == 0.
    assert goal == Vector3(x=0., y=4., z=1.)
    target.x = 8.
    assert measured.model_dump() == original


# 功能：
#   验证无动态目标的左右转弯仍服从路线且保留原连续角速度幅度。
# 输入：
#   side：原路线转弯方向。
# 输出：
#   None：断言正负偏航与坐标约定一致。
@pytest.mark.parametrize('side', [-1., 1.])
def test_empty_dynamic_scene_keeps_route_turn(side):
    position, goal = Vector3(x=0., y=0., z=1.), Vector3(x=1., y=side, z=1.)
    target = teacher_observation_target(position=position, goal=goal, obstacles=[])
    assert teacher_heading_rate(orientation=QuaternionWxyz(w=1., x=0., y=0., z=0.),
        position=position, goal=target, maximum_rate_dps=20.) == -20. * side


# 功能：
#   验证陈旧或无置信度目标不触发盲目朝路线转头，源数据保持原年龄。
# 输入：
#   age、confidence：需要抑制主动偏航的边界条件。
# 输出：
#   None：零偏航参考不得更新目标或赋予控制权限。
@pytest.mark.parametrize('age,confidence', [(0.251, .9), (26., .9), (0., 0.)])
def test_unusable_target_does_not_rejuvenate_or_turn_to_route(age, confidence):
    position = Vector3(x=0., y=0., z=1.)
    measured = obstacle(age=age).model_copy(update={'confidence': confidence})
    assert teacher_observation_target(position=position, goal=Vector3(x=0., y=5., z=1.),
                                      obstacles=[measured]) == position
    assert measured.age_seconds == age


# 功能：
#   验证以表面距离选择目标，等距离按身份稳定选择，不依赖输入顺序。
# 输入：
#   reverse：是否颠倒输入目标顺序。
# 输出：
#   None：输出必须指向同一个最近目标。
@pytest.mark.parametrize('reverse', [False, True])
def test_nearest_target_and_ties_are_stable(reverse):
    rows = [obstacle('a', 2., 1.), obstacle('b', 2., -1.), obstacle('far', 9., 0.)]
    if reverse:
        rows.reverse()
    target = teacher_observation_target(position=Vector3(x=0., y=0., z=1.),
        goal=Vector3(x=0., y=5., z=1.), obstacles=rows)
    assert target == Vector3(x=2., y=1., z=1.)


# 功能：
#   验证重复身份、超容量和绕过构造器篡改的年龄均被拒绝。
# 输入：
#   kind：非法目标集合类型。
# 输出：
#   None：无效输入不能成为示范方向。
@pytest.mark.parametrize('kind', ['duplicate', 'capacity', 'mutated'])
def test_invalid_dynamic_inputs_rejected(kind):
    rows = [obstacle(), obstacle()] if kind == 'duplicate' else (
        [obstacle(str(i)) for i in range(257)] if kind == 'capacity' else
        [obstacle().model_copy(update={'age_seconds': float('nan')})])
    with pytest.raises(ValueError):
        teacher_observation_target(position=Vector3(x=0., y=0., z=1.),
            goal=Vector3(x=0., y=5., z=1.), obstacles=rows)


# 功能：
#   验证布尔值、非有限数及非法幅度不能伪装成合法偏航限制。
# 输入：
#   limit：待拒绝的参数。
# 输出：
#   None：非法边界必须明确拒绝，不生成角速度。
@pytest.mark.parametrize('limit', [True, '20', None, 0., -1., 46., float('nan'), float('inf'), 10**400])
def test_invalid_yaw_limit_rejected(limit):
    with pytest.raises(ValueError, match='SIMULATION_TEACHER_YAW_LIMIT_INVALID'):
        teacher_heading_rate(orientation=QuaternionWxyz(w=1., x=0., y=0., z=0.),
            position=Vector3(x=0., y=0., z=1.), goal=Vector3(x=0., y=5., z=1.), maximum_rate_dps=limit)
