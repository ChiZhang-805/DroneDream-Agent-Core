"""Explicit current-observation input for a learned precision yaw branch.

The production navigation snapshot has no verified actuator receipt. Until one
is supplied through a bound interface, its command/history masks remain absent;
we never substitute the last prediction for an executed command.
"""

from .heading_context import encode_heading_context, validate_geometry
from .heading_observation import heading_observation_from_snapshot

PRECISION_HEADING_ARCHITECTURE = 'causal-gru-with-learned-heading-v1'


# 功能：
#   从同一新鲜源快照构造偏航分支输入；没有已验证执行回执时明确使用学习型单帧分支。
# 输入：
#   snapshot：带精确目标几何和内容摘要的导航快照。
#   now_unix_ms：当前真实时钟，不用原观测时间代替。
# 输出：
#   features：二十三维输入，过去命令和历史观测的有效位均为零。
def current_precision_heading_input(snapshot: dict, *, now_unix_ms: int) -> tuple[float, ...]:
    geometry = heading_observation_from_snapshot(snapshot, now_unix_ms=now_unix_ms)
    features = encode_heading_context(geometry, snapshot['control_reference_observed_at_unix_ms'])
    return features


# 功能：
#   验证当前生产分支的输入模式和物理尺度，拒绝未绑定的历史、虚构回执或改变偏航单位。
# 输入：
#   features：调用方持有的二十三维偏航输入。
#   yaw_limit_dps：本次模型输出实际采用的角速度尺度。
# 输出：
#   values：验证后的不可变特征。
def validate_current_precision_heading_input(features, *, yaw_limit_dps: float) -> tuple[float, ...]:
    if (type(features) not in (tuple, list) or len(features) != 23
            or type(yaw_limit_dps) not in (float, int) or yaw_limit_dps != 20.
            or any(type(value) not in (float, int) or value != 0. for value in features[8:])):
        raise ValueError('PRECISION_HEADING_CURRENT_INPUT_INVALID')
    values = (*validate_geometry(features[:8]), *(0.,) * 15)
    return values
