"""Bounded receive-time alignment of actual native telemetry; no interpolation."""

import copy
import math
from collections import deque

from .control_feature_contract import FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS, FLIGHT_STATE_MAXIMUM_GAP_MS
from .source_clock import source_received_at_unix_ms

_FIELDS = {
    'position': ('north_m', 'east_m', 'down_m', 'north_m_s', 'east_m_s', 'down_m_s'),
    'attitude': ('roll_deg', 'pitch_deg', 'yaw_deg'),
    'imu': ('acceleration_forward_m_s2', 'acceleration_right_m_s2', 'acceleration_down_m_s2',
            'angular_velocity_forward_rad_s', 'angular_velocity_right_rad_s', 'angular_velocity_down_rad_s'),
}


class NativeReceiveAligner:
    """Single publisher owns bounded original position, attitude and IMU samples."""

    # 功能：
    #   创建固定容量的三路原始样本缓存，避免异步到达时盲目拼接三个最新值。
    # 输入：
    #   无。
    # 输出：
    #   无：初始化缓存及已发出的各来源时间下界。
    def __init__(self):
        self._buffers = {name: deque(maxlen=16) for name in ('position', 'attitude', 'imu')}
        self._published = {name: -1 for name in self._buffers}
        self._last_collection = -1
        self._last_seen = {}
        self.summary = {'paired_count': 0, 'historical_pair_count': 0, 'unpaired_count': 0}

    # 功能：
    #   清除配对历史但保留已发布时刻，故障恢复不能倒退到旧状态。
    # 输入：
    #   无。
    # 输出：
    #   无：清空三路缓存。
    def clear(self):
        for buffer in self._buffers.values():
            buffer.clear()

    # 功能：
    #   核验当前三路样本本身有效；缓存只能消除到达相位差，不能替换损坏的当前测量。
    # 输入：
    #   items：原始位置、姿态和 IMU。
    #   now：原始缓存汇集时刻。
    # 输出：
    #   valid：字段和源年龄全部合格时为真。
    def _valid(self, items, now):
        for name, fields in _FIELDS.items():
            item = items[name]
            values = [getattr(item, field, None) if name == 'position' else item.get(field)
                      for field in fields]
            if any(type(value) not in (int, float) or not -1e100 < value < 1e100
                   or not math.isfinite(value) for value in values):
                return False
            if name != 'position':
                if type(item.get('timestamp_us')) is not int or not 0 <= item['timestamp_us'] < 2**63:
                    return False
                try:
                    source_received_at_unix_ms(item, collected_at_unix_ms=now, now_unix_ms=now,
                                              maximum_age_ms=FLIGHT_STATE_MAXIMUM_GAP_MS)
                except ValueError:
                    return False
        return True

    # 功能：
    #   1. 从真实接收样本中选取最新的时间一致组合，维持 50 ms 偏差和 250 ms 有效期。
    #   2. 不插值、不预测、不修改原接收时刻或设备时钟；缺样、错误和时钟倒退交给原消费者拒绝。
    # 输入：
    #   observed：独占的位置速度快照。
    #   dynamics：独占的动力学快照和原始接收时间。
    # 输出：
    #   aligned_observed、aligned_dynamics：时间配对的独立快照；无法配对时保留原输入。
    def align(self, observed, dynamics: dict):
        now = dynamics.get('collected_at_unix_ms')
        sources = dynamics.get('sources')
        if type(now) is int and now < self._last_collection:
            self.clear()
            raise ValueError('NATIVE_ALIGNMENT_COLLECTION_CLOCK_REGRESSED')
        issues = dynamics.get('blocking_issue_codes', [])
        # 电机等非配对来源的故障原样保留，由其安全门控处理；不妨碍其他三路做时间配对。
        # 未知故障不能被归为无关，只有这里明确的非配对来源允许继续。
        unrelated = ('actuator_output:', 'battery:', 'odometry:')
        pair_fault = (type(issues) is not list or any(type(issue) is not str
                      or not issue.startswith(unrelated) for issue in issues))
        if (type(now) is not int or not 0 <= now < 2**63
                or type(sources) is not dict or pair_fault
                or dynamics.get('schema_version') != 'dronedream.px4-dynamics-telemetry.v1'):
            self.clear()
            return observed, dynamics
        items = {'position': observed, 'attitude': sources.get('attitude'), 'imu': sources.get('imu')}
        times = {'position': getattr(observed, 'received_at_unix_ms', None)}
        for name in ('attitude', 'imu'):
            item = items[name]
            times[name] = item.get('received_at_unix_ms') if type(item) is dict else None
        if any(type(stamp) is not int or not 0 <= stamp <= now
               or now - stamp > FLIGHT_STATE_MAXIMUM_GAP_MS for stamp in times.values()):
            self.clear()
            return observed, dynamics
        if not self._valid(items, now):
            self.clear()
            return observed, dynamics
        self._last_collection = now
        signatures = {}
        for name, item in items.items():
            payload = item if name == 'position' else {
                key: value for key, value in item.items() if key != 'sample_age_seconds'}
            stamp = times[name]
            boot = stamp if name == 'position' else item['timestamp_us']
            previous = self._last_seen.get(name)
            if previous is not None:
                prev_stamp, prev_boot, prev_payload = previous
                if stamp < prev_stamp or boot < prev_boot or (boot == prev_boot and payload != prev_payload):
                    self.clear()
                    raise ValueError('NATIVE_ALIGNMENT_SOURCE_REGRESSED_OR_CONFLICTED')
            signatures[name] = (stamp, boot, copy.deepcopy(payload))
        self._last_seen.update(signatures)
        for name, stamp in times.items():
            buffer = self._buffers[name]
            if buffer and stamp < buffer[-1][0]:
                self.clear()
                return observed, dynamics
        for name, stamp in times.items():
            buffer = self._buffers[name]
            if not buffer or stamp != buffer[-1][0]:
                buffer.append((stamp, copy.deepcopy(items[name]), now))
            elif name != 'position' and items[name]['timestamp_us'] != buffer[-1][1]['timestamp_us']:
                buffer[-1] = (stamp, copy.deepcopy(items[name]), now)
            while buffer and now - buffer[0][0] > FLIGHT_STATE_MAXIMUM_GAP_MS:
                buffer.popleft()
        # 最多 48 个原始时间锚、每路最多 16 项；工作量与任务长度无关。
        anchors = sorted({row[0] for buffer in self._buffers.values() for row in buffer}, reverse=True)
        for anchor in anchors:
            chosen = {}
            for name, buffer in self._buffers.items():
                item = next(((stamp, value, collected) for stamp, value, collected in reversed(buffer)
                             if stamp >= self._published[name]
                             and anchor <= stamp <= anchor + FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS), None)
                if item is None:
                    break
                chosen[name] = item
            if len(chosen) != 3:
                continue
            aligned_dynamics = copy.deepcopy(dynamics)
            for name in ('attitude', 'imu'):
                stamp, item, collected = chosen[name]
                aligned_dynamics['sources'][name] = copy.deepcopy(item)
                # 年龄按保留的原接收时刻增长；不得把组装时刻写成源时刻。
                age = max((now - stamp) / 1000, item['sample_age_seconds'] + (now - collected) / 1000)
                if age > FLIGHT_STATE_MAXIMUM_GAP_MS / 1000:
                    break
                aligned_dynamics['sources'][name]['sample_age_seconds'] = age
            else:
                self._published = {name: item[0] for name, item in chosen.items()}
                self.summary['paired_count'] += 1
                self.summary['historical_pair_count'] += int(any(chosen[name][0] != times[name] for name in times))
                return copy.deepcopy(chosen['position'][1]), aligned_dynamics
        self.summary['unpaired_count'] += 1
        return observed, dynamics
