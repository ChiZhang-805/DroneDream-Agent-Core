"""Exercise actual spawned geometry workers and preserved source clocks."""

import time

import numpy as np
import pytest
from test_live_map_measurement import fixture

from dronedream_agent_core.isolated_map_measurement import IsolatedMapMeasurementSource
from dronedream_agent_core.live_map_measurement import MapMeasurementPending


# 功能：启动真实独立计算进程；输入：原有三面几何夹具；输出：进程及原始记录。
def process_fixture():
    source, record = fixture()
    isolated = IsolatedMapMeasurementSource(index=source.index, map_sha256=source.map_sha256,
        binding=source.binding, binding_sha256=source.binding_sha256,
        clock_domain=source.clock_domain, noise=source.noise)
    return isolated, record


# 功能：为新构造的测试观测设置真实单调接收时刻，不读取或重写任何已采集实验数据。
# 输入：当前测试记录；输出：同一测试记录；原始模拟源纳秒保持不变。
def receive_now(record):
    now = time.monotonic()
    record['scan']['observed_at_monotonic_seconds'] = now
    record['native_odometry_snapshot']['source_alignment']['received_monotonic_seconds'] = now


# 功能：独立进程真实求解与原算法一致，重复输入不续龄，关闭不遗留子进程。
# 输入：真实进程及几何夹具；输出：位姿、权限、生命周期断言。
def test_spawned_geometry_matches_serial_and_retains_history():
    worker, record = process_fixture()
    try:
        receive_now(record)
        result = worker.measure(record, source_now_ns=lambda: 1_050_000_000,
                                monotonic_now=time.monotonic)
        np.testing.assert_allclose(result['position_ned_m'], [.03, -.04, .02], atol=1e-6)
        assert result['source_timestamp_ns'] == 1_000_000_000
        assert not result['motion_permission_granted'] and not result['covariance_qualified']
        with pytest.raises(MapMeasurementPending, match='DID_NOT_ADVANCE'):
            worker.measure(record, source_now_ns=lambda: 1_050_000_000,
                           monotonic_now=time.monotonic)
    finally:
        worker.close()
    assert not worker._process.is_alive()
    assert worker.close()['closed']
    with pytest.raises(RuntimeError, match='CLOSED'):
        worker.measure(record, source_now_ns=lambda: 1_050_000_000, monotonic_now=time.monotonic)


# 功能：跨进程不吞身份错误，实际当前钟判定过期而非改写源钟；输入：故障类型；输出：明确拒绝。
@pytest.mark.parametrize('fault', ['identity', 'expired', 'missing-clock', 'future-clock'])
def test_spawned_validation_does_not_grant_authority(fault):
    worker, record = process_fixture()
    try:
        receive_now(record)
        stamp = 1_050_000_000
        if fault == 'identity':
            record['map_sha256'] = 'b' * 64
        elif fault == 'expired':
            stamp = 1_151_000_000
        elif fault == 'missing-clock':
            stamp = None
        else:
            stamp = 999_000_000
        with pytest.raises(ValueError) as caught:
            worker.measure(record, source_now_ns=lambda: stamp, monotonic_now=time.monotonic)
        assert isinstance(caught.value, MapMeasurementPending) == (
            fault in {'expired', 'missing-clock'})
    finally:
        worker.close()


# 功能：终止仅数学子进程后必须明确失败，不返回旧结果。
# 输入：已关闭的子计算进程；输出：失败及资源回收。
def test_dead_worker_fails_and_closes_channel():
    worker, record = process_fixture()
    worker._process.terminate()
    worker._process.join(2)
    try:
        receive_now(record)
        with pytest.raises(RuntimeError, match='CHANNEL_FAILED|COMPUTE_TIMEOUT'):
            worker.measure(record, source_now_ns=lambda: 1_050_000_000,
                           monotonic_now=time.monotonic)
    finally:
        worker.close()


# 功能：在途计算发生父钟错误后不能留下旧回复或发送线程；输入：中途断钟；输出：完整关闭。
def test_clock_exception_during_compute_closes_owned_resources():
    worker, record = process_fixture()
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError('TEST_CLOCK_SOURCE_FAILED')
        return 1_050_000_000
    try:
        receive_now(record)
        with pytest.raises(ValueError, match='TEST_CLOCK_SOURCE_FAILED'):
            worker.measure(record, source_now_ns=clock, monotonic_now=time.monotonic)
    finally:
        worker.close()
    assert not worker._process.is_alive()
    assert not worker._lock.locked()


# 功能：关闭失败不得声明已回收，后续调用仍可重试退出；输入：拒绝第一次退出的假进程。
# 输出：首次失败、第二次完成；仅测试资源状态，不启动仿真。
def test_close_failure_remains_retryable():
    from types import SimpleNamespace
    worker = IsolatedMapMeasurementSource.__new__(IsolatedMapMeasurementSource)
    alive = True
    worker._closed = worker._stopped = False
    worker._pipe = SimpleNamespace(close=lambda: None)
    worker._process = SimpleNamespace(pid=1, join=lambda timeout: None,
        is_alive=lambda: alive, terminate=lambda: None)
    with pytest.raises(RuntimeError, match='DID_NOT_STOP'):
        worker.close()
    assert worker._closed and not worker._stopped
    alive = False
    assert worker.close()['closed']


# 功能：损坏的跨进程回复立即销毁自有通道，不能被当作下一帧或有效测量。
# 输入：真实子进程返回后注入的协议故障；输出：固定错误、进程退出和锁释放。
@pytest.mark.parametrize('reply', [
    [], {'kind': 'unknown', 'value': {}}, {'kind': 'pending', 'value': ''},
    {'kind': 'error', 'value': {'message': 'bad'}},
    {'kind': 'result', 'value': {'source_timestamp_ns': True}},
    {'kind': 'result', 'value': {'source_timestamp_ns': 1_000_000_000,
                               'covariance_qualified': True, 'motion_permission_granted': False}},
    {'kind': 'result', 'value': {'source_timestamp_ns': 1_000_001_000,
                               'covariance_qualified': False, 'motion_permission_granted': False}},
])
def test_malformed_reply_cannot_reuse_channel(monkeypatch, reply):
    worker, record = process_fixture()
    monkeypatch.setattr('dronedream_agent_core.isolated_map_measurement.decode_json',
                        lambda *args, **kwargs: reply)
    try:
        receive_now(record)
        with pytest.raises(ValueError, match='ISOLATED_MAP_RESPONSE_INVALID'):
            worker.measure(record, source_now_ns=lambda: 1_050_000_000,
                           monotonic_now=time.monotonic)
        assert worker._closed and not worker._process.is_alive()
        assert not worker._lock.locked()
    finally:
        worker.close()
