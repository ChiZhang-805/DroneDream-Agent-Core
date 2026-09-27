"""Asynchronous physical-source refresh tests; no simulator or flight evidence."""

import math

import pytest
from test_native_state_stream import packet

from dronedream_agent_core.native_flight_state import native_flight_state_sample
from dronedream_agent_core.native_state_stream import NativeStateBuffer
from dronedream_agent_core.realtime_feature_encoders import FlightStateEncoder, world_enu_to_body
from dronedream_agent_core.sensor_diagnostics import sensor_issue_code


# 功能：
#   复现姿态已推进、其他来源尚未推进的真实异步包形态，避免新姿态与旧编码混用。
# 输入：
#   stalled：保持旧来源身份的传感器或重复 IMU 设备时间。
# 输出：
#   None：当前姿态和体轴速度一致，历史槽数量和最旧期限不变，旧对象不被覆盖。
@pytest.mark.parametrize('stalled', ['odometry', 'imu', 'imu_boot_only'])
def test_partial_native_refresh_keeps_pose_encoding_consistent(stalled):
    buffer = NativeStateBuffer()
    for stamp in range(1000, 1320, 20):
        buffer.ingest(packet(stamp), now_unix_ms=stamp)
    previous = buffer.latest(now_unix_ms=1300)
    history_length = len(buffer._encoder._samples)
    updated = packet(1320)
    updated['dynamics']['sources']['attitude']['yaw_deg'] = 110.
    updated['observed_velocity_ned_mps']['east_m_s'] = 3.
    if stalled == 'imu_boot_only':
        updated['dynamics']['sources']['imu']['timestamp_us'] = 1300000
    else:
        updated['dynamics']['sources'][stalled] = packet(1300)['dynamics']['sources'][stalled]
    buffer.ingest(updated, now_unix_ms=1320)
    current = buffer.latest(now_unix_ms=1320)
    orientation = list(current.pose.orientation_world_from_body.model_dump().values())
    assert current.encoding.features[:4] == pytest.approx(orientation)
    assert current.encoding.features[:4] != previous.encoding.features[:4]
    velocity = world_enu_to_body(
        current.pose.orientation_world_from_body, current.pose.velocity_world_enu_mps)
    assert current.encoding.features[4:7] == pytest.approx(
        [velocity.x/20., velocity.y/20., velocity.z/20.])
    assert current.encoding.observed_at_unix_ms == previous.encoding.observed_at_unix_ms == 1300
    assert len(buffer._encoder._samples) == history_length
    assert current.encoding.source_sha256 != previous.encoding.source_sha256
    assert previous.encoding.history_slot_revision == 0
    assert current.encoding.history_slot_revision == 1
    assert not current.encoding.fresh_at(1600)
    # 相同局部包再次发布不产生新的编码或历史；下一个完整新样本则正常推进。
    buffer.ingest(updated, now_unix_ms=1330)
    assert buffer.latest(now_unix_ms=1330).encoding == current.encoding
    buffer.ingest(packet(1340), now_unix_ms=1340)
    assert buffer.latest(now_unix_ms=1340).encoding.observed_at_unix_ms == 1340
    assert buffer.latest(now_unix_ms=1340).encoding.history_slot_revision == 0


# 功能：
#   局部刷新不能被用来制造新历史、切换设备或接受损坏数值，失败不提交任何状态。
# 输入：
#   kind：独立新包、换源、时间回退或非有限数值。
# 输出：
#   None：刷新明确拒绝并保留原编码和历史。
@pytest.mark.parametrize('kind', ['new_sample', 'source', 'regression', 'nonfinite'])
def test_invalid_partial_refresh_does_not_commit(kind):
    encoder = FlightStateEncoder()
    original = native_flight_state_sample(packet(1000), now_unix_ms=1000)
    first = encoder.encode(original, encoded_at_unix_ms=1000)
    updates = ({'source_id': 'other'} if kind == 'source' else
               {'observed_at_unix_ms': 999} if kind == 'regression' else {})
    sample = original.model_copy(update=updates)
    if kind == 'new_sample':
        sample = native_flight_state_sample(packet(1020), now_unix_ms=1020)
    elif kind == 'nonfinite':
        sample = sample.model_copy(update={'quality': math.nan})
    with pytest.raises(ValueError):
        encoder.refresh_current(sample, encoded_at_unix_ms=1020)
    assert encoder._last_encoding == first
    assert len(encoder._samples) == 1
    assert encoder._samples[0] == original


# 功能：
#   新姿态不能为已经过期的原 IMU 延长寿命，避免局部更新无限续期。
# 输入：
#   无。
# 输出：
#   None：新编码保留旧来源时刻并继续被时效门禁拒绝。
def test_partial_refresh_never_revives_expired_source():
    encoder = FlightStateEncoder()
    original = native_flight_state_sample(packet(1000), now_unix_ms=1000)
    encoder.encode(original, encoded_at_unix_ms=1000)
    refreshed = encoder.refresh_current(original, encoded_at_unix_ms=1400)
    assert refreshed.observed_at_unix_ms == 1000
    assert not refreshed.fresh_at(1400)


# 功能：
#   复现里程计先推进、IMU 随后推进的异步交接；独立新样本不能被误判为旧槽刷新。
# 输入：
#   无。
# 输出：
#   None：新 IMU 到达后追加历史，保留真实来源时间，重复包不重复追加。
def test_imu_advance_after_partial_refresh_uses_encoder_slot():
    buffer = NativeStateBuffer()
    original = packet(1000)
    original['dynamics']['sources']['imu']['timestamp_us'] = 1010000
    buffer.ingest(original, now_unix_ms=1000)
    partial = packet(1005)
    partial['dynamics']['sources']['imu']['timestamp_us'] = 1010000
    buffer.ingest(partial, now_unix_ms=1005)
    refreshed = buffer.latest(now_unix_ms=1005)
    assert refreshed.sample.observed_at_unix_ms == 1005
    assert refreshed.encoding.observed_at_unix_ms == 1000
    assert len(buffer._encoder._samples) == 1
    complete = packet(1020)
    complete['dynamics']['sources']['odometry'] = partial['dynamics']['sources']['odometry']
    buffer.ingest(complete, now_unix_ms=1020)
    advanced = buffer.latest(now_unix_ms=1020)
    assert advanced.sample.observed_at_unix_ms == 1005
    assert advanced.encoding.observed_at_unix_ms == 1005
    assert advanced.encoding.history_slot_revision == 0
    assert len(buffer._encoder._samples) == 2
    buffer.ingest(complete, now_unix_ms=1021)
    assert len(buffer._encoder._samples) == 2
    assert buffer.latest(now_unix_ms=1021).encoding == advanced.encoding


# 功能：
#   保留异步交接拒收的固定原因，未知消息仍不得泄露到健康摘要。
# 输入：
#   无。
# 输出：
#   None：已知错误可定位、未知原文被去除。
def test_native_slot_rejection_has_bounded_diagnostic():
    assert sensor_issue_code(ValueError('FLIGHT_STATE_REFRESH_NOT_SAME_HISTORY_SLOT')) == (
        'FLIGHT_STATE_REFRESH_NOT_SAME_HISTORY_SLOT')
    assert sensor_issue_code(ValueError('flight state timestamps must increase strictly')) == (
        'FLIGHT_STATE_TIMESTAMP_NOT_INCREASING')
    assert sensor_issue_code(ValueError('unrecognized telemetry payload')) == 'ValueError'
