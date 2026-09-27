"""Source-timed heading context, shared by offline preparation and shadow inference."""

import math

from .hashing import sha256_json
from .heading_observation import HEADING_OBSERVATION_SHA256

HEADING_CONTEXT_WIDTH = 23
HEADING_CONTEXT_CONTRACT = {
    'heading': HEADING_OBSERVATION_SHA256,
    'width': HEADING_CONTEXT_WIDTH,
    'order': ['current-heading8', 'body-clockwise-z-rate/20dps', 'gyro-valid',
              'previous-accepted-yaw/20dps', 'acceptance-age/250ms', 'accepted-valid',
              'previous-heading8', 'observation-gap/250ms', 'history-valid'],
    'history': 'strictly earlier source and receipt times; <=250ms; same stream and goal/track identity',
    'gyro': 'negative FLU body z angular velocity, not Euler heading derivative; clip [-4,4]',
    'command': 'transport acknowledgement is prior input, not measured response; no current/future label',
    'missing': 'zero values with zero validity; old source is never renewed',
    'yaw_limit_dps': 20.,
    'authority': 'research/shadow perception; no flight authority',
}
HEADING_CONTEXT_SHA256 = sha256_json(HEADING_CONTEXT_CONTRACT)


# 功能：
#   校验有限实数与八维几何边界，拒绝无效方位和缺测目标的非零编码。
# 输入：
#   geometry：当前或历史的八维方位编码。
# 输出：
#   values：经校验的不可变浮点几何。
def validate_geometry(geometry):
    if type(geometry) not in (list, tuple) or len(geometry) != 8:
        raise ValueError('HEADING_CONTEXT_GEOMETRY_INVALID')
    if any(type(v) not in (int, float) or not -4 <= v <= 4 for v in geometry):
        raise ValueError('HEADING_CONTEXT_GEOMETRY_INVALID')
    values = tuple(float(v) for v in geometry)
    if (any(abs(values[i]) > 1 for i in (0, 3)) or any(not 0 <= values[i] <= 4 for i in (1, 4))
            or any(values[i] not in (0., 1.) for i in (2, 5, 6, 7))
            or any(values[flag] == 0. and values[angle] != 0. for angle, flag in ((0, 2), (3, 5)))
            or (values[6] == 0. and any(values[i] != 0. for i in (3, 4, 5, 7)))):
        raise ValueError('HEADING_CONTEXT_GEOMETRY_INVALID')
    return values


# 功能：
#   组装当前几何、实际体轴转速、过去已接受动作和历史观测，阻止未来数据泄漏。
# 输入：
#   geometry、source_ms：当前几何及原始观测时间；gyro_flu_z_rad_s：同源体轴陀螺仪值或缺测。
#   previous_geometry、previous_source_ms：同任务且同目标身份的上一观测及时间。
#   previous_yaw_dps、accepted_ms：严格早于当前观测的实际传输回执及接受时间。
# 输出：
#   features：带缺测掩码的二十三维研究输入。
def encode_heading_context(geometry, source_ms, *, gyro_flu_z_rad_s=None, previous_geometry=None, previous_source_ms=None, previous_yaw_dps=None, accepted_ms=None):
    current = validate_geometry(geometry)
    if type(source_ms) is not int or not 0 <= source_ms < 2**63:
        raise ValueError('HEADING_CONTEXT_CLOCK_INVALID')
    gyro = (0., 0.)
    if gyro_flu_z_rad_s is not None:
        if type(gyro_flu_z_rad_s) not in (int, float) or not -1e6 <= gyro_flu_z_rad_s <= 1e6:
            raise ValueError('HEADING_CONTEXT_GYRO_INVALID')
        gyro = (max(-4., min(4., -math.degrees(gyro_flu_z_rad_s)/20.)), 1.)
    command = (0., 0., 0.)
    if (previous_yaw_dps is None) != (accepted_ms is None):
        raise ValueError('HEADING_CONTEXT_COMMAND_INCOMPLETE')
    if accepted_ms is not None:
        if (type(accepted_ms) is not int or not 0 <= accepted_ms < source_ms
                or type(previous_yaw_dps) not in (int, float) or not -20. <= previous_yaw_dps <= 20.):
            raise ValueError('HEADING_CONTEXT_COMMAND_INVALID')
        if source_ms-accepted_ms <= 250:
            command = (previous_yaw_dps/20., (source_ms-accepted_ms)/250., 1.)
    history = (0.,)*10
    if (previous_geometry is None) != (previous_source_ms is None):
        raise ValueError('HEADING_CONTEXT_HISTORY_INCOMPLETE')
    if previous_source_ms is not None:
        previous = validate_geometry(previous_geometry)
        if type(previous_source_ms) is not int or not 0 <= previous_source_ms < source_ms:
            raise ValueError('HEADING_CONTEXT_HISTORY_CLOCK_INVALID')
        if source_ms-previous_source_ms <= 250:
            history = (*previous, (source_ms-previous_source_ms)/250., 1.)
    features = (*current, *gyro, *command, *history)
    return features
