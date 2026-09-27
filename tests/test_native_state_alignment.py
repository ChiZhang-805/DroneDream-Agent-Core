import copy
from types import SimpleNamespace

import pytest

from dronedream_agent_core.native_state_alignment import NativeReceiveAligner


# 功能：
#   创建拥有独立原始时钟的三路遥测夹具，不模拟新的传感器采样。
# 输入：
#   now：汇集时刻。
#   stamps：位置、姿态、IMU 的原始接收时刻。
# 输出：
#   observed、dynamics：测试输入快照。
def packet(now, stamps):
    p, a, i = stamps
    observed = SimpleNamespace(north_m=0., east_m=0., down_m=0., north_m_s=0.,
                               east_m_s=0., down_m_s=0., received_at_unix_ms=p)
    dynamics = {'schema_version': 'dronedream.px4-dynamics-telemetry.v1',
                'collected_at_unix_ms': now, 'blocking_issue_codes': [], 'sources': {
        'attitude': {'roll_deg': 0., 'pitch_deg': 0., 'yaw_deg': 0., 'timestamp_us': a * 1000,
                     'received_at_unix_ms': a, 'sample_age_seconds': (now - a) / 1000},
        'imu': {'acceleration_forward_m_s2': 0., 'acceleration_right_m_s2': 0.,
                'acceleration_down_m_s2': -9.81, 'angular_velocity_forward_rad_s': 0.,
                'angular_velocity_right_rad_s': 0., 'angular_velocity_down_rad_s': 0.,
                'timestamp_us': i * 1000, 'received_at_unix_ms': i,
                'sample_age_seconds': (now - i) / 1000}}}
    return observed, dynamics


# 功能：
#   证明相位差使用真实历史配对，不改原始时间、不减少年龄，也不修改调用方对象。
# 输入：
#   无。
# 输出：
#   None：断言配对时差、时间和数据隔离。
def test_alignment_pairs_original_samples_without_freshening():
    aligner = NativeReceiveAligner()
    aligner.align(*packet(1000, (1000, 980, 990)))
    observed, dynamics = packet(1080, (1080, 1030, 990))
    original = copy.deepcopy(dynamics)
    aligned, paired = aligner.align(observed, dynamics)
    assert aligned.received_at_unix_ms == 1000
    assert paired['sources']['attitude']['received_at_unix_ms'] == 1030
    assert paired['sources']['imu']['timestamp_us'] == 990000
    assert paired['sources']['imu']['sample_age_seconds'] == pytest.approx(.09)
    assert dynamics == original
    assert observed.received_at_unix_ms == 1080
    aligned.north_m = 99
    paired['sources']['imu']['acceleration_forward_m_s2'] = 99
    aligned_again, paired_again = aligner.align(observed, dynamics)
    assert aligned_again.north_m == 0
    assert paired_again['sources']['imu']['acceleration_forward_m_s2'] == 0
    assert aligner.summary['historical_pair_count'] == 2


# 功能：
#   确认失效、坏字段及来源错误不能被上一帧的有效历史掩盖。
# 输入：
#   corruption：本轮注入的无效字段类别。
# 输出：
#   None：断言坏包原样交给下游安全检查，历史已清除。
@pytest.mark.parametrize('corruption', ['nan', 'bool', 'age', 'future', 'expired', 'missing', 'fault', 'schema'])
def test_invalid_current_measurement_cannot_be_masked(corruption):
    aligner = NativeReceiveAligner()
    aligner.align(*packet(1000, (1000, 980, 990)))
    observed, dynamics = packet(1080, (1080, 1030, 990))
    imu = dynamics['sources']['imu']
    if corruption == 'nan':
        imu['acceleration_forward_m_s2'] = float('nan')
    elif corruption == 'bool':
        observed.north_m = True
    elif corruption == 'age':
        imu['sample_age_seconds'] = -.01
    elif corruption == 'future':
        imu['received_at_unix_ms'] = 1081
    elif corruption == 'expired':
        imu['received_at_unix_ms'] = 829
    elif corruption == 'missing':
        del imu['timestamp_us']
    elif corruption == 'fault':
        dynamics['blocking_issue_codes'] = ['imu:stream_error']
    else:
        dynamics['schema_version'] = 'obsolete'
    actual = aligner.align(observed, dynamics)
    assert actual[0] is observed and actual[1] is dynamics
    assert all(not buffer for buffer in aligner._buffers.values())


# 功能：
#   拒绝同一来源时间下内容改变、接收时钟倒退和设备时钟倒退。
# 输入：
#   mutation：冲突类型。
# 输出：
#   None：断言发布失败且不沿用缓存。
@pytest.mark.parametrize('mutation', ['content', 'receive', 'boot'])
def test_conflicting_or_regressing_source_is_rejected(mutation):
    aligner = NativeReceiveAligner()
    aligner.align(*packet(1000, (1000, 980, 990)))
    observed, dynamics = packet(1010, (1010, 1000, 990))
    imu = dynamics['sources']['imu']
    if mutation == 'content':
        imu['angular_velocity_forward_rad_s'] = 1.
    elif mutation == 'receive':
        imu['received_at_unix_ms'] = 989
    else:
        imu['timestamp_us'] = 989000
    with pytest.raises(ValueError, match='REGRESSED_OR_CONFLICTED'):
        aligner.align(observed, dynamics)
    assert all(not buffer for buffer in aligner._buffers.values())


# 功能：
#   检查队列上限及有效期，超过期限时没有合成或重复新采样。
# 输入：
#   无。
# 输出：
#   None：断言原始时钟和固定队列上限。
def test_history_is_bounded_and_expired_samples_are_not_revived():
    aligner = NativeReceiveAligner()
    for now in range(1000, 1100):
        aligner.align(*packet(now, (now, now, now)))
    assert all(len(buffer) == 16 for buffer in aligner._buffers.values())
    observed, dynamics = packet(1400, (1400, 1400, 1099))
    result = aligner.align(observed, dynamics)
    assert result[0] is observed and result[1] is dynamics
    assert all(not buffer for buffer in aligner._buffers.values())


# 功能：
#   没有满足时间差的真实组合时保持失败证据，不插值出新的姿态或 IMU。
# 输入：
#   无。
# 输出：
#   None：断言不配对计数和原始对象。
def test_no_match_keeps_original_mismatch():
    aligner = NativeReceiveAligner()
    observed, dynamics = packet(1100, (1100, 1000, 990))
    actual = aligner.align(observed, dynamics)
    assert actual[0] is observed and actual[1] is dynamics
    assert aligner.summary['unpaired_count'] == 1


# 功能：
#   保存来源报告的较大年龄，避免墙钟取整或偏差使缓存年龄变小。
# 输入：
#   无。
# 输出：
#   None：断言源年龄随时间增长。
def test_original_age_cannot_be_reduced():
    aligner = NativeReceiveAligner()
    observed, dynamics = packet(1000, (1000, 980, 990))
    dynamics['sources']['imu']['sample_age_seconds'] = .08
    aligner.align(observed, dynamics)
    observed, dynamics = packet(1080, (1080, 1030, 990))
    _, actual = aligner.align(observed, dynamics)
    assert actual['sources']['imu']['sample_age_seconds'] == pytest.approx(.16)


# 功能：
#   系统时间倒退必须停止发布，不能将缓存清空当作安全恢复。
# 输入：
#   无。
# 输出：
#   None：断言明确的时钟错误。
def test_collection_clock_regression_is_rejected():
    aligner = NativeReceiveAligner()
    aligner.align(*packet(1100, (1100, 1100, 1100)))
    with pytest.raises(ValueError, match='COLLECTION_CLOCK_REGRESSED'):
        aligner.align(*packet(1099, (1099, 1099, 1099)))


# 功能：
#   电机过期不会禁用位置姿态配对，也不能通过配对抹掉电机故障。
# 输入：
#   无。
# 输出：
#   None：断言姿态时间已配对且原故障、原就绪状态保留。
def test_unrelated_source_fault_is_preserved_not_used_to_disable_pairing():
    aligner = NativeReceiveAligner()
    aligner.align(*packet(1000, (1000, 980, 990)))
    observed, dynamics = packet(1080, (1080, 1030, 990))
    dynamics['blocking_issue_codes'] = ['actuator_output:SAMPLE_STALE']
    dynamics['ready_for_payload_inference'] = False
    aligned, actual = aligner.align(observed, dynamics)
    assert aligned.received_at_unix_ms == 1000
    assert actual['blocking_issue_codes'] == ['actuator_output:SAMPLE_STALE']
    assert actual['ready_for_payload_inference'] is False


# 功能：
#   校验发布器实际使用配对结果，统计与写入的来源时间一致。
# 输入：
#   无。
# 输出：
#   None：断言两个发布包及关闭后的配对统计。
def test_publisher_integrates_alignment():
    import asyncio
    from dronedream_agent_core.native_telemetry_publisher import NativeTelemetryPublisher

    # 功能：
    #   以串行两次遥测驱动真实发布循环，然后等待写入排空。
    # 输入：
    #   无。
    # 输出：
    #   None：断言写入证据。
    async def scenario():
        sequence = [packet(1000, (1000, 980, 990)), packet(1080, (1080, 1030, 990))]
        emitted = []

        class Client:
            # 功能：
            #   返回本轮位置缓存。
            # 输入：
            #   timeout：接口等待预算。
            # 输出：
            #   observed：本轮位置。
            async def sample_position_velocity_ned(self, timeout):
                return sequence[min(len(emitted), 1)][0]

            # 功能：
            #   返回与本轮位置同时读取的动力学缓存。
            # 输入：
            #   max_age：接口年龄预算。
            # 输出：
            #   dynamics：本轮动力学。
            def latest_dynamics_telemetry(self, max_age):
                return sequence[min(len(emitted), 1)][1]

        publisher = NativeTelemetryPublisher(Client(), lambda p, d: emitted.append((p, d)))
        publisher.start()
        try:
            for _ in range(100):
                if len(emitted) >= 2:
                    break
                await asyncio.sleep(.005)
        finally:
            summary = await publisher.close()
        assert len(emitted) >= 2
        assert emitted[1][0].received_at_unix_ms == 1000
        assert emitted[1][1]['sources']['imu']['sample_age_seconds'] == pytest.approx(.09)
        assert summary['receive_alignment']['historical_pair_count'] >= 1
        assert summary['unique_source_count'] == 2

    asyncio.run(scenario())
