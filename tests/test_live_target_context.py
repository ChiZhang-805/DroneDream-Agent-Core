"""Actual loopback target delivery; no vehicle or model is substituted."""
import asyncio
import threading
import time
from unittest.mock import Mock

import pytest
from clock_fixtures import isolate_time

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core import runtime_phase_channel
from dronedream_agent_core.executor_snapshots import ExecutorSnapshots
from dronedream_agent_core.local_packet_channel import LatestPacketPublisher
from dronedream_agent_core.runtime_phase_channel import (
    PHASE_CONTRACT,
    RuntimePhaseBroadcaster,
    RuntimePhaseReceiver,
)


# 功能：
#   建立用于真实通信测试的完整目标，不携带动作授权或传感器期限。
# 输入：
#   stamp：目标原始更新时间。
# 输出：
#   target：有效的目标上下文。
def target(stamp):
    return {'schema_version': 'dronedream.local-safety-target.v1',
            'target_position_m': {'x': 1., 'y': 2., 'z': 3.},
            'navigation_goal_position_m': {'x': 2., 'y': 2., 'z': 3.},
            'navigation_goal_id': 'real-goal', 'control_profile': 'precision',
            'action_checkpoint_goal': False, 'tracking_recovery_active': False,
            'updated_at_unix_ms': stamp}


# 功能：
#   在有限时间内消费实际后台通信，不用一次固定睡眠假设包已送达。
# 输入：
#   receiver：真实接收器。
# 输出：
#   result：已接收的目标对象。
def receive(receiver):
    deadline = time.monotonic() + 1.
    while time.monotonic() < deadline:
        try:
            return receiver.read_target()
        except FileNotFoundError:
            time.sleep(.002)
    pytest.fail('target was not delivered')


# 功能：
#   验证目标原时刻、可变引用隔离和旧包期限，缓存与旧文件不能让过期上下文恢复。
# 输入：
#   tmp_path、monkeypatch：隔离目录及受控本机时钟。
# 输出：
#   None：不返回业务数据。
def test_target_channel_preserves_source_and_rejects_stale_context(tmp_path, monkeypatch):
    clock = Mock(return_value=10.1)
    isolate_time(monkeypatch, runtime_phase_channel, time=clock)
    receiver = RuntimePhaseReceiver(tmp_path / 'endpoint.json')
    publisher = LatestPacketPublisher(receiver.path, contract=PHASE_CONTRACT)
    path = tmp_path / 'local-safety-target.json'
    path.write_text('{}')
    try:
        with pytest.raises(FileNotFoundError):
            worker._read_control_target(path, now_unix_ms=10100, receiver=receiver)
        publisher.send({'schema_version': PHASE_CONTRACT.payload_schema,
                        'updated_at_unix_ms': 10000, 'state': {'phase': 'TRACK'},
                        'control_target': target(9000)})
        value = receive(receiver)
        value['target_position_m']['x'] = 999.
        actual = worker._read_control_target(path, now_unix_ms=10100, receiver=receiver)
        assert actual == target(9000)
        clock.return_value = 10.501
        with pytest.raises(ValueError, match='CONTEXT_EXPIRED'):
            worker._read_control_target(path, now_unix_ms=10501, receiver=receiver)
        clock.return_value = 9.999
        with pytest.raises(ValueError, match='CONTEXT_EXPIRED'):
            receiver.read_target()
    finally:
        publisher.close()
        receiver.close()
    with pytest.raises(RuntimeError, match='UNAVAILABLE'):
        receiver.read_target()


# 功能：
#   模拟诊断磁盘被阻塞，确认新目标仍经实时通道送达且字段校验不被绕过。
# 输入：
#   tmp_path：隔离的真实文件及套接字目录。
# 输出：
#   None：不返回业务数据。
def test_target_broadcast_does_not_wait_for_diagnostic_disk(tmp_path):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   阻塞诊断存储，保持它与实时通信隔离。
    # 输入：
    #   args、kwargs：原子文件写入器参数。
    # 输出：
    #   None：不返回业务数据。
    def writer(*args, **kwargs):
        entered.set()
        assert release.wait(2.)

    snapshots = ExecutorSnapshots(tmp_path, writer=writer)
    receiver = RuntimePhaseReceiver(tmp_path / 'endpoint.json')
    broadcaster = RuntimePhaseBroadcaster([receiver.path], snapshots)
    try:
        now = int(time.time() * 1000)
        snapshots.submit(snapshots.phase_path, {'phase': 'TRACK'})
        assert entered.wait(1.)
        snapshots.submit(snapshots.target_path, target(now))
        broadcaster.start()
        assert receive(receiver) == target(now)
        assert not snapshots.target_path.exists()
        invalid = target(now)
        invalid['action_checkpoint_goal'] = 'false'
        with pytest.raises(ValueError, match='FLAG_INVALID'):
            worker._read_control_target(snapshots.target_path, now_unix_ms=now,
                                        receiver=Mock(read_target=lambda: invalid))
        with pytest.raises(ValueError, match='SOURCE_AMBIGUOUS'):
            worker._read_control_target(snapshots.target_path, now_unix_ms=now,
                                        reader=Mock(), receiver=receiver)
    finally:
        asyncio.run(broadcaster.close())
        release.set()
        assert snapshots.close()['complete']
        receiver.close()


# 功能：
#   覆盖实际出现的 199 毫秒单调年龄、204 毫秒 UNIX 年龄分歧，以及边界、未来和坏类型。
# 输入：
#   mono、stamp、expected：图像单调时刻、UNIX 时刻和预期可记录性。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('mono,stamp,expected', [
    (9.801, 9796, False), (9.801, 9800, True), (9.801, 9799, False),
    (9.9, 9900, True), (9.9, 10001, False), (10.001, 9999, False),
    (float('nan'), 9900, False), (True, 9900, False), (9.9, True, False),
    (9.9, None, False), (9.9, -1, False),
])
def test_learning_rgb_uses_both_original_clocks(mono, stamp, expected):
    assert worker._learning_visual_current(source_monotonic=mono, source_unix_ms=stamp,
        reference_monotonic=10., reference_unix_ms=10000) is expected
