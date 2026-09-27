"""Continuous geometry lifecycle: sparse visual corrections are not a route pose."""

import asyncio
import threading
from collections import deque

import pytest

from dronedream_agent_core.live_map_measurement import MapMeasurementPending
from dronedream_agent_core.live_map_measurement_stream import run_map_measurement_stream


class Source:
    # 功能：提供可控计算结果；输入：回调；输出：仅用于生命周期测试的源替身。
    def __init__(self, callback=None):
        self.callback = callback

    # 功能：模拟没有路线输入的单次测量；输入：来源记录与时钟；输出：原记录或指定故障。
    def measure(self, record, **kwargs):
        if self.callback:
            return self.callback(record)
        if isinstance(record, Exception):
            raise record
        return record


# 功能：地图纠偏暂缺时保持接收，依然只发布新且验证过的测量，不伪造运动许可。
# 输入：成功、超出地图间隔的缺失、真实恢复；输出：流恢复且不重放旧数据。
def test_optional_initialized_correction_recovers_without_replaying_pose():
    async def scenario():
        stop, evidence, sent = asyncio.Event(), {}, []
        async def publish(value):
            sent.append(value['source_timestamp_ns'])
            if len(sent) == 2:
                stop.set()
        def read():
            if not sent:
                return {'source_timestamp_ns': 1000}
            if evidence.get('status') == 'awaiting-correction':
                return {'source_timestamp_ns': 2000}
            return None
        await run_map_measurement_stream(source=Source(), read_latest=read,
            source_now_ns=lambda: 3000, publish=publish, stop=stop, evidence=evidence,
            maximum_gap_seconds=.01, optional_after_initialization=True)
        assert sent == [1000, 2000]
        assert evidence['correction_gap_seconds'] >= .01
        assert evidence['published'] == 2 and not evidence['motion_permission_granted']
    asyncio.run(scenario())


# 功能：可选纠偏只作用于已初始化的瞬态缺失；启动失败和错误身份仍必须拒绝。
# 输入：无初始观测或错误地图；输出：不放开启动准入。
@pytest.mark.parametrize('record', [None, ValueError('map-changed')])
def test_optional_correction_preserves_initialization_and_identity_failure(record):
    async def scenario():
        with pytest.raises((ValueError, RuntimeError), match='OBSERVATIONS_LOST|map-changed'):
            await run_map_measurement_stream(source=Source(), read_latest=lambda: record,
                source_now_ns=lambda: 1000, publish=None, stop=asyncio.Event(), evidence={},
                startup_timeout_seconds=.01, optional_after_initialization=True)
    asyncio.run(scenario())


# 功能：串联普通约束、缺失方向及恢复帧；输入：无；输出：缺失帧不重播，不中止短暂预测区间。
def test_partial_frames_wait_for_real_correction_without_replay():
    async def scenario():
        records = deque([{'source_timestamp_ns': 1_000_000_000},
            MapMeasurementPending('LIVE_MAP_PARTIAL_CONSTRAINT_ONLY'),
            {'source_timestamp_ns': 1_050_000_000}])
        stop, evidence, sent = asyncio.Event(), {}, []

        async def publish(value):
            sent.append(value['source_timestamp_ns'])
            if len(sent) == 2:
                stop.set()

        await run_map_measurement_stream(source=Source(), read_latest=records.popleft,
            source_now_ns=lambda: 1_060_000_000, publish=publish, stop=stop, evidence=evidence)
        assert sent == [1_000_000_000, 1_050_000_000]
        assert evidence['attempted'] == 3 and evidence['published'] == 2
        assert evidence['pending'] == {'LIVE_MAP_PARTIAL_CONSTRAINT_ONLY': 1}
        assert evidence['compute_drained'] and evidence['status'] == 'stopped'
        assert not evidence['motion_permission_granted']
        assert not evidence['covariance_qualified']

    asyncio.run(scenario())


# 功能：身份破坏和发送错误不得像普通遮挡一样重试；输入：故障阶段；输出：原错传播。
@pytest.mark.parametrize('stage', ['measure', 'publish'])
def test_nontransient_failure_is_reported_to_run_owner(stage):
    async def scenario():
        evidence = {}

        async def publish(value):
            raise RuntimeError('transport-failed')

        value = ValueError('map-changed') if stage == 'measure' else {'source_timestamp_ns': 1000}
        with pytest.raises((ValueError, RuntimeError), match='map-changed|transport-failed'):
            await run_map_measurement_stream(source=Source(), read_latest=lambda: value,
                source_now_ns=lambda: 2000, publish=publish, stop=asyncio.Event(),
                evidence=evidence)
        assert evidence['published'] == 0 and evidence['compute_drained']
        assert evidence['status'] == 'failed'

    asyncio.run(scenario())


# 功能：单次回读超时只丢弃当次确认，不更新成功时刻，不重发旧帧。
# 输入：第一帧发送后缺回读，第二帧独立成功；输出：只记录一次真实成功。
def test_missing_readback_requires_new_measurement_and_real_confirmation():
    async def scenario():
        records = deque([{'source_timestamp_ns': 1000}, {'source_timestamp_ns': 2000}])
        sent, evidence, stop = [], {}, asyncio.Event()
        async def publish(value):
            sent.append(value['source_timestamp_ns'])
            if len(sent) == 1:
                raise MapMeasurementPending('LIVE_MAP_READBACK_TIMEOUT')
            stop.set()
        await run_map_measurement_stream(source=Source(), read_latest=records.popleft,
            source_now_ns=lambda: 3000, publish=publish, stop=stop, evidence=evidence)
        assert sent == [1000, 2000]
        assert evidence['published'] == 1 and evidence['last_source_timestamp_ns'] == 2000
        assert evidence['pending'] == {'LIVE_MAP_READBACK_TIMEOUT': 1}
    asyncio.run(scenario())


# 功能：持续丢失回读仍触发总断流期限；重复同一帧则更早拒绝。
# 输入：不断超时的发送端，独立或重复源时刻；输出：零成功、不放宽期限。
@pytest.mark.parametrize('repeat', [False, True])
def test_missing_readbacks_do_not_renew_stream_deadline(repeat):
    async def scenario():
        evidence, count = {}, [0]
        def read():
            count[0] += 1
            return {'source_timestamp_ns': 1 if repeat else count[0]}
        async def publish(value):
            raise MapMeasurementPending('LIVE_MAP_READBACK_TIMEOUT')
        with pytest.raises((RuntimeError, ValueError), match='OBSERVATIONS_LOST|SOURCE_NOT_ADVANCING'):
            await run_map_measurement_stream(source=Source(), read_latest=read,
                source_now_ns=lambda: 100000000, publish=publish, stop=asyncio.Event(),
                evidence=evidence, startup_timeout_seconds=.03)
        assert evidence['published'] == 0 and evidence['last_source_timestamp_ns'] is None
    asyncio.run(scenario())


# 功能：持续无新观测必须终止，而非用历史位姿续期；输入：无；输出：有界失联故障。
def test_missing_stream_has_bounded_startup():
    async def scenario():
        evidence = {}
        with pytest.raises(RuntimeError, match='OBSERVATIONS_LOST'):
            await run_map_measurement_stream(source=Source(), read_latest=lambda: None,
                source_now_ns=lambda: 1000, publish=None, stop=asyncio.Event(), evidence=evidence,
                startup_timeout_seconds=.03)
        assert evidence['attempted'] == evidence['published'] == 0

    asyncio.run(scenario())


# 功能：持续收到退化几何不能误报为完全没有图像；输入：缺一维的观测；输出：准确启动根因。
def test_initial_partial_geometry_has_specific_failure():
    async def scenario():
        evidence = {}
        with pytest.raises(RuntimeError, match='STARTUP_GEOMETRY_INCOMPLETE'):
            await run_map_measurement_stream(source=Source(),
                read_latest=lambda: MapMeasurementPending('LIVE_MAP_PARTIAL_CONSTRAINT_ONLY'),
                source_now_ns=lambda: 1000, publish=None, stop=asyncio.Event(), evidence=evidence,
                startup_timeout_seconds=.03)
        assert evidence['published'] == 0 and evidence['attempted'] > 0
    asyncio.run(scenario())


# 功能：停止在途计算必须先回收计算再结束，不发送停止后完成的结果。
# 输入：可控线程计算和异步停止/取消。
# 输出：没有后台遗留，无停止后发送。
@pytest.mark.parametrize('cancel', [False, True])
def test_stop_and_cancel_drain_inflight_computation(cancel):
    async def scenario():
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        stop, evidence, sent = asyncio.Event(), {}, []

        def compute(record):
            entered.set()
            assert release.wait(2.)
            finished.set()
            return record

        async def publish(value):
            sent.append(value)

        task = asyncio.create_task(run_map_measurement_stream(source=Source(compute),
            read_latest=lambda: {'source_timestamp_ns': 1000}, source_now_ns=lambda: 2000,
            publish=publish, stop=stop, evidence=evidence))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.005)
            assert entered.is_set()
            if cancel:
                task.cancel()
            else:
                stop.set()
            await asyncio.sleep(.02)
            assert not task.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            assert finished.is_set() and evidence['compute_drained'] and not sent
            assert evidence['status'] == ('cancelled' if cancel else 'stopped')
            assert 'issue' not in evidence
            if cancel:
                assert evidence['stop_requested'] is False
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


# 功能：事件循环延迟也必须淘汰旧测量；输入：返回的旧源钟；输出：不伪造发送时间。
def test_expired_dispatch_waits_for_next_frame():
    async def scenario():
        stop, evidence = asyncio.Event(), {}

        def read_latest():
            if evidence['attempted']:
                stop.set()
                return None
            return {'source_timestamp_ns': 1000}

        await run_map_measurement_stream(source=Source(), read_latest=read_latest,
            source_now_ns=lambda: 151_000_001, publish=None, stop=stop, evidence=evidence)
        assert evidence['published'] == 0
        assert evidence['pending']['LIVE_MAP_SOURCE_EXPIRED'] == 1

    asyncio.run(scenario())
