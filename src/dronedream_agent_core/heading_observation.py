"""Measured heading geometry, not a heading controller or an executed action.

Both offline reconstruction and live inference use this encoder. No simulator
truth, route lookahead, teacher command or future observation is an input.
"""

import math
from collections.abc import Sequence

from .contracts import DynamicObstacleObservation, QuaternionWxyz, Vector3
from .hashing import sha256_json
from .realtime_feature_encoders import RealtimeFeatureSnapshot, world_enu_to_body

HEADING_OBSERVATION_WIDTH = 8
HEADING_OBSERVATION_CONTRACT = {
    'width': HEADING_OBSERVATION_WIDTH,
    'coordinates': 'world ENU horizontal displacement rotated to body FRU',
    'goal': ['atan2(right,forward)/pi', 'horizontal-body-distance/local-radius capped 4',
             'horizontal-body-distance >= 0.1m'],
    'nearest_track': ['same bearing', 'same distance', 'same defined-heading flag'],
    'flags': ['any track present', 'any track age >0.25s or confidence <=0'],
    'selection': 'minimum measured 3D center distance minus max(radius,height/2), then id',
    'timing': 'original synchronous measured track; no prediction or age refresh',
    'missing_track': 'zero bearing, distance and defined-heading flag; presence distinguishes absence',
    'authority': 'perception geometry only; no yaw command or control limit in input',
}
HEADING_OBSERVATION_SHA256 = sha256_json(HEADING_OBSERVATION_CONTRACT)


# 功能：
#   将世界水平位移投影为机体方位和距离；近重合位置不伪造有意义的朝向。
# 输入：
#   position、target：当前机体位置与观测目标的世界 ENU 米坐标。
#   orientation：世界 ENU 相对机体 FLU 的姿态。
#   radius：距离归一化尺度，单位米。
# 输出：
#   geometry：归一化方位、水平体轴距离与有效朝向标志三元组。
def _horizontal_geometry(position: Vector3, target: Vector3, orientation: QuaternionWxyz, radius: float):
    displacement = Vector3(x=target.x-position.x, y=target.y-position.y, z=0.)
    body = world_enu_to_body(orientation, displacement)
    distance = math.hypot(body.x, body.y)
    defined = distance >= .1
    geometry = (math.atan2(body.y, body.x)/math.pi if defined else 0.,
                min(distance/radius, 4.), float(defined))
    return geometry


# 功能：
#   编码导航目标和最近已测目标的精确方位，保留两种参照而不替模型选择控制动作。
# 输入：
#   position、goal：同一快照的机体位置和经批准的语义目标。
#   orientation：该快照的实测姿态。
#   obstacles：同一快照最多 256 个原始感知目标，不接受重复身份或失真数值。
#   local_radius_m：正且有限的距离归一化尺度。
# 输出：
#   features：八个有限观测特征；过期标志不表示可以继续飞行。
def encode_heading_observation(*, position: Vector3, goal: Vector3, orientation: QuaternionWxyz, obstacles: Sequence[DynamicObstacleObservation], local_radius_m: float) -> tuple[float, ...]:
    if type(local_radius_m) not in (float, int) or not .1 <= local_radius_m <= 10000.:
        raise ValueError('HEADING_OBSERVATION_RADIUS_INVALID')
    if type(obstacles) not in (list, tuple) or len(obstacles) > 256:
        raise ValueError('HEADING_OBSERVATION_TARGETS_INVALID')
    if (not isinstance(position, Vector3) or not isinstance(goal, Vector3)
            or not isinstance(orientation, QuaternionWxyz)
            or any(not isinstance(item, DynamicObstacleObservation) for item in obstacles)):
        raise ValueError('HEADING_OBSERVATION_INPUT_TYPE_INVALID')
    position = Vector3.model_validate(position.model_dump(), strict=True)
    goal = Vector3.model_validate(goal.model_dump(), strict=True)
    orientation = QuaternionWxyz.model_validate(orientation.model_dump(), strict=True)
    checked = [DynamicObstacleObservation.model_validate(item.model_dump(), strict=True) for item in obstacles]
    if len({item.obstacle_id for item in checked}) != len(checked):
        raise ValueError('HEADING_OBSERVATION_DUPLICATE_TARGET')
    goal_geometry = _horizontal_geometry(position, goal, orientation, local_radius_m)
    track_geometry = (0., 0., 0.)
    if checked:
        nearest = min(checked, key=lambda item: (
            math.dist((position.x, position.y, position.z), (item.position_m.x, item.position_m.y, item.position_m.z))
            - max(item.radius_m, item.height_m/2), item.obstacle_id))
        track_geometry = _horizontal_geometry(position, nearest.position_m, orientation, local_radius_m)
    features = (*goal_geometry, *track_geometry, float(bool(checked)),
                float(any(item.age_seconds > .25 or item.confidence <= 0 for item in checked)))
    return features


# 功能：
#   从与训练同结构的实时快照提取精确方位，拒绝缺几何、来源被修改或已过期的输入。
# 输入：
#   snapshot：带原始感知目标、同步状态、任务目标和完整内容摘要的导航快照。
#   now_unix_ms：调用方的当前 Unix 毫秒时间，不以旧快照时间代替当前时钟。
# 输出：
#   features：八维当前观测；不包含教师命令，不授权任何飞行动作。
def heading_observation_from_snapshot(snapshot: dict, *, now_unix_ms: int) -> tuple[float, ...]:
    if type(snapshot) is not dict or snapshot.get('snapshot_sha256') != sha256_json({
            key: value for key, value in snapshot.items() if key != 'snapshot_sha256'}):
        raise ValueError('HEADING_OBSERVATION_SNAPSHOT_UNBOUND')
    reference = snapshot.get('control_reference_observed_at_unix_ms')
    if (type(now_unix_ms) is not int or type(reference) is not int
            or not 0 <= reference <= now_unix_ms < 2**63 or now_unix_ms > reference + 250):
        raise ValueError('HEADING_OBSERVATION_CLOCK_INVALID')
    realtime = RealtimeFeatureSnapshot.model_validate(snapshot.get('realtime_feature_snapshot'))
    if realtime.control_deadline_unix_ms(now_unix_ms=now_unix_ms) <= now_unix_ms:
        raise ValueError('HEADING_OBSERVATION_SOURCE_EXPIRED')
    flight = next((row for row in realtime.encodings if row.encoder_role == 'flight-state-encoder'), None)
    if flight is None or flight.valid_mask[:4] != [1., 1., 1., 1.]:
        raise ValueError('HEADING_OBSERVATION_ATTITUDE_MISSING')
    rows = snapshot.get('dynamic_obstacles')
    if type(rows) is not list or len(rows) > 256:
        raise ValueError('HEADING_OBSERVATION_TARGETS_INVALID')
    keys = ('obstacle_id', 'position_m', 'velocity_mps', 'radius_m', 'height_m', 'confidence', 'observation_age_seconds')
    if any(type(row) is not dict or any(key not in row for key in keys) for row in rows):
        raise ValueError('HEADING_OBSERVATION_PRECISE_GEOMETRY_MISSING')
    obstacles = [DynamicObstacleObservation.model_validate({
        **{key: row[key] for key in keys if key != 'observation_age_seconds'},
        'age_seconds': row['observation_age_seconds']}, strict=True) for row in rows]
    orientation = QuaternionWxyz(**dict(zip(('w', 'x', 'y', 'z'), flight.features[:4], strict=True)))
    features = encode_heading_observation(position=Vector3.model_validate(snapshot.get('current_position_m'), strict=True),
        goal=Vector3.model_validate(snapshot.get('goal_position_m'), strict=True), orientation=orientation,
        obstacles=obstacles, local_radius_m=snapshot.get('local_radius_m'))
    return features
