"""Synthetic observation contracts, not evidence of learned flight capability."""

import math

import pytest

from dronedream_agent_core.contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from dronedream_agent_core.heading_observation import encode_heading_observation, heading_observation_from_snapshot
from dronedream_agent_core.hashing import sha256_json
from control_fixtures import complete_feature_snapshot
from dronedream_agent_core.realtime_feature_encoders import encode_dynamic_targets

IDENTITY = QuaternionWxyz(w=1., x=0., y=0., z=0.)
POSITION = Vector3(x=0., y=0., z=1.)
GOAL = Vector3(x=4., y=0., z=1.)


# 功能：
#   创建可控的静止目标，便于隔离距离相同而左右方向不同的输入。
# 输入：
#   y：目标左右世界坐标；identity：目标标识；age：原始年龄。
# 输出：
#   target：合成感知目标。
def target(y, identity='person', age=0.):
    target = DynamicObstacleObservation(obstacle_id=identity, position_m=Vector3(x=4., y=y, z=1.),
        velocity_mps=Vector3(x=0., y=0., z=0.), radius_m=.3, height_m=1., confidence=.9, age_seconds=age)
    return target


# 功能：
#   为测试调用无控制权限的方位编码。
# 输入：
#   obstacles：感知目标；kwargs：测试覆盖的几何参数。
# 输出：
#   features：八维观测。
def encode(obstacles=(), **kwargs):
    parameters = dict(position=POSITION, goal=GOAL, orientation=IDENTITY,
                      local_radius_m=8., obstacles=obstacles)
    parameters.update(kwargs)
    features = encode_heading_observation(**parameters)
    return features


# 功能：
#   复现旧粗分区编码无法区分同格左右目标，验证精确方位编码消除该信息损失。
# 输入：
#   无。
# 输出：
#   None：旧输入相同但新输入方向相反，标签和执行量不参与编码。
def test_same_cell_opposite_bearings_are_distinguishable():
    targets = [target(.3), target(-.3)]
    coarse = [encode_dynamic_targets([item], body_position_world_enu_m=POSITION,
        body_orientation_world_from_body=IDENTITY, body_velocity_world_enu_mps=Vector3(x=0., y=0., z=0.),
        observed_at_unix_ms=1000, encoded_at_unix_ms=1000) for item in targets]
    assert coarse[0].features == coarse[1].features
    left, right = (encode([item]) for item in targets)
    assert left[3] < 0 < right[3]
    assert left[3] == pytest.approx(-right[3])
    assert left[:3] == right[:3]
    assert left[4:] == right[4:]


# 功能：
#   纯爬升不引入因姿态倾斜而虚构的水平转向，保留目标高度用于其他既有输入。
# 输入：
#   angle：机体俯仰角。
# 输出：
#   None：水平目标重合时方位为未定义而非任意正负方向。
@pytest.mark.parametrize('angle', [-.2, 0., .2])
def test_vertical_goal_has_no_artificial_heading(angle):
    features = encode_heading_observation(position=POSITION, goal=Vector3(x=0., y=0., z=9.),
        orientation=QuaternionWxyz(w=math.cos(angle/2), x=0., y=math.sin(angle/2), z=0.),
        obstacles=[], local_radius_m=8.)
    assert features == (0., 0., 0., 0., 0., 0., 0., 0.)


# 功能：
#   验证旋转、最近目标选择、原年龄和目标排列不改变几何语义。
# 输入：
#   无。
# 输出：
#   None：选择稳定且陈旧标志保留，不刷新源对象。
def test_rotated_pose_stable_target_selection_and_age():
    rows = [target(-.3, 'b', .251), target(.3, 'a')]
    before = [item.model_dump() for item in rows]
    features = encode(rows)
    assert features == encode(list(reversed(rows)))
    assert features[3] < 0 and features[-2:] == (1., 1.)
    assert before == [item.model_dump() for item in rows]
    rotated = encode_heading_observation(position=POSITION, goal=GOAL,
        orientation=QuaternionWxyz(w=math.sqrt(.5), x=0., y=0., z=math.sqrt(.5)),
        obstacles=[], local_radius_m=8.)
    assert rotated[0] == pytest.approx(.5)
    assert rotated[1] == pytest.approx(.5)


# 功能：
#   拒绝非法尺度、重复目标和绕过构造器的非有限年龄。
# 输入：
#   kind：无效输入类别。
# 输出：
#   None：输入不能静默变成合法控制观测。
@pytest.mark.parametrize('kind', ['radius', 'duplicate', 'age'])
def test_invalid_heading_inputs_rejected(kind):
    rows = [target(.3)]
    radius = float('nan') if kind == 'radius' else 8.
    if kind == 'duplicate':
        rows.append(target(-.3))
    if kind == 'age':
        rows[0] = rows[0].model_copy(update={'age_seconds': float('nan')})
    with pytest.raises(ValueError):
        encode_heading_observation(position=POSITION, goal=GOAL, orientation=IDENTITY,
                                   obstacles=rows, local_radius_m=radius)


# 功能：
#   验证实时入口和离线编码一致，修改内容、过期状态及旧的距离摘要都不能伪造精确观测。
# 输入：
#   kind：正常、篡改、过期或缺失几何的输入类别。
# 输出：
#   None：正常输出相同，坏输入被明确拒绝。
@pytest.mark.parametrize('kind', ['valid', 'tampered', 'expired', 'legacy', 'future'])
def test_snapshot_heading_geometry_and_live_deadline(kind):
    obstacle = target(.3)
    row = obstacle.model_dump(mode='json')
    row['observation_age_seconds'] = row.pop('age_seconds')
    if kind == 'legacy':
        row.pop('position_m')
    snapshot = dict(current_position_m=POSITION.model_dump(), goal_position_m=GOAL.model_dump(),
        dynamic_obstacles=[row], local_radius_m=8., control_reference_observed_at_unix_ms=1000,
        realtime_feature_snapshot=complete_feature_snapshot().model_dump(mode='json'))
    snapshot['snapshot_sha256'] = sha256_json(snapshot)
    if kind == 'tampered':
        snapshot['goal_position_m']['x'] = 99.
    now = 1250 if kind == 'expired' else 999 if kind == 'future' else 1000
    if kind == 'valid':
        assert heading_observation_from_snapshot(snapshot, now_unix_ms=now) == encode([obstacle])
    else:
        with pytest.raises(ValueError, match='HEADING_OBSERVATION'):
            heading_observation_from_snapshot(snapshot, now_unix_ms=now)


# 功能：
#   验证实际导航快照生产器保留感知几何，在线提取与离线编码使用相同实测信息。
# 输入：
#   无。
# 输出：
#   None：编译后的几何和八维特征一致，调用方后续修改不会改变已冻结的快照。
def test_navigation_compiler_preserves_measured_heading_geometry():
    from dataclasses import replace
    from test_learning_observation_recorder import request_fixture
    from dronedream_agent_core.navigation_snapshot import compile_navigation_snapshot

    request, _ = request_fixture()
    obstacle = target(.3)
    request = replace(request, frame=request.frame.model_copy(update={'dynamic_obstacles': [obstacle]}))
    snapshot = compile_navigation_snapshot(request)
    features = heading_observation_from_snapshot(snapshot, now_unix_ms=1000)
    expected = encode_heading_observation(position=request.frame.localization_position_m,
        goal=request.goal_position_m, orientation=IDENTITY, obstacles=[obstacle],
        local_radius_m=snapshot['local_radius_m'])
    assert features == expected
    assert snapshot['dynamic_obstacles'][0]['position_m'] == obstacle.position_m.model_dump()
    obstacle.position_m.x = 99.
    assert heading_observation_from_snapshot(snapshot, now_unix_ms=1000) == expected


# 功能：
#   巨大整数、布尔和无限尺度不能触发隐式浮点转换或绕过输入边界。
# 输入：
#   radius：非法距离归一化尺度。
# 输出：
#   None：所有非法尺度均被明确拒绝。
@pytest.mark.parametrize('radius', [True, float('inf'), -(10**400), 10**400, 0., '8'])
def test_heading_radius_has_bounded_numeric_type(radius):
    with pytest.raises(ValueError, match='HEADING_OBSERVATION_RADIUS_INVALID'):
        encode(local_radius_m=radius)
