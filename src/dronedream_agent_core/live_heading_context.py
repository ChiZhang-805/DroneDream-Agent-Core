"""Source-bound live heading memory; no simulator truth, future receipts or label inputs."""

import math

from .control_execution_evidence import ControlApplicationRecord
from .hashing import sha256_json
from .heading_context import encode_heading_context
from .heading_observation import heading_observation_from_snapshot


class LiveHeadingContext:
    """Keep only a previously validated sensor encoding, not an unexecuted action."""

    # 功能：
    #   初始化只保留一个已验证观测的时序编码器，不自动恢复旧回合状态。
    # 输入：
    #   无。
    # 输出：
    #   None：创建空历史。
    def __init__(self):
        self._previous = None

    # 功能：
    #   从真实当前快照和已绑定过去回执形成训练同构输入；跨流或目标变化重置历史。
    # 输入：
    #   snapshot、stream_id：当前源快照与回合观测流；application：调用方已核对命令摘要的过去接受回执。
    #   now_unix_ms：实际当前时钟；mask_command：仅用于仿真压力测试的显式回执遮蔽。
    # 输出：
    #   features：含缺测位的二十三维输入，不向飞控发送动作。
    def encode(self, snapshot: dict, stream_id: str, application: ControlApplicationRecord | None, *, now_unix_ms: int, mask_command: bool = False):
        if type(stream_id) is not str or not 0 < len(stream_id) <= 256 or type(mask_command) is not bool:
            raise ValueError('LIVE_HEADING_STREAM_INVALID')
        geometry = heading_observation_from_snapshot(snapshot, now_unix_ms=now_unix_ms)
        source = snapshot['control_reference_observed_at_unix_ms']
        position, tracks = snapshot['current_position_m'], snapshot['dynamic_obstacles']
        nearest = min(tracks, key=lambda row: (math.dist(tuple(position[a] for a in 'xyz'),
            tuple(row['position_m'][a] for a in 'xyz'))-max(row['radius_m'], row['height_m']/2), row['obstacle_id'])) if tracks else None
        context = snapshot.get('strategic_context', {})
        if type(context) is not dict or type(context.get('task', {})) is not dict:
            raise ValueError('LIVE_HEADING_GOAL_IDENTITY_INVALID')
        goal_id = context.get('task', {}).get('navigation_goal_id')
        if goal_id is not None and (type(goal_id) is not str or not 1 <= len(goal_id) <= 160):
            raise ValueError('LIVE_HEADING_GOAL_IDENTITY_INVALID')
        # 同一个取件点也可能属于不同任务阶段；坐标相同不等于语义目标仍是上一代。
        # 没有语义标识的历史输入仍按流和位置隔离，但不能冒充有标识的新阶段。
        identity = sha256_json(dict(stream=stream_id, goal_id=goal_id,
            goal=snapshot['goal_position_m'], track=nearest['obstacle_id'] if nearest else None))
        history, history_time = None, None
        if self._previous is not None:
            old_identity, old_source, old_geometry = self._previous
            if old_identity == identity:
                if source <= old_source:
                    raise ValueError('LIVE_HEADING_SOURCE_NOT_NEW')
                if source-old_source <= 250:
                    history, history_time = old_geometry, old_source
        flight = next(row for row in snapshot['realtime_feature_snapshot']['encodings'] if row['encoder_role'] == 'flight-state-encoder')
        gyro = 5*flight['features'][13] if flight['valid_mask'][13] == 1 and abs(flight['features'][13]) < 4 else None
        yaw, accepted = None, None
        if application is not None:
            if not isinstance(application, ControlApplicationRecord):
                raise ValueError('LIVE_HEADING_APPLICATION_INVALID')
            application = ControlApplicationRecord.model_validate(application.model_dump(mode='python'), strict=True)
            if application.accepted_at_unix_ms >= source:
                raise ValueError('LIVE_HEADING_FUTURE_APPLICATION')
            rate = application.yaw_rate_application
            if (not mask_command and application.transport == 'velocity-ned' and rate is not None
                    and abs(rate.clockwise_rate_dps) <= 20.
                    and application.command_generated_at_unix_ms <= application.accepted_at_unix_ms < application.command_valid_until_unix_ms):
                yaw, accepted = rate.clockwise_rate_dps, application.accepted_at_unix_ms
        features = encode_heading_context(geometry, source, gyro_flu_z_rad_s=gyro, previous_geometry=history,
            previous_source_ms=history_time, previous_yaw_dps=yaw, accepted_ms=accepted)
        self._previous = (identity, source, geometry)
        return features
