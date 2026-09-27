"""Continuous measurement lifecycle, without route inputs or motion authority.

Transport callbacks belong to the run owner. A successful send is not firmware
fusion or localization qualification; those require independent readback/tests.
"""

import asyncio
import math
import time
from collections import Counter

from .live_map_measurement import (
    MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS,
    LiveMapMeasurementSource,
    MapMeasurementPending,
)


# 功能：在取消时等待唯一一次在途几何计算结束，避免旧运行后台线程继续修改估计器。
# 输入：executor future：有界点数的单次计算。
# 输出：实际计算结果；取消仍传播，但不遗留可跨运行写状态的线程。
async def _drainable_compute(future):
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Consume a computation exception without replacing the cancellation.
        if future.done() and not future.cancelled():
            future.exception()
        raise


# 功能：持续消费最新真实观测，允许单帧缺乏约束，超时或身份错误则将故障交给运行所有者。
# 输入：单运行测量器、认证来源读取、源钟、实际发送回调、停止事件及空统计对象。
#       启动等待和连续缺失期限只用于生命周期监督，不改变单帧有效期或飞控安全条件。
# 输出：有界统计，无原始图像或无界历史；不生成路线位置、不续发旧帧、不请求解锁。
async def run_map_measurement_stream(*, source, read_latest, source_now_ns, publish,
                                     stop, evidence, startup_timeout_seconds=30.,
                                     maximum_gap_seconds=2., optional_after_initialization=False):
    if (type(optional_after_initialization) is not bool or type(evidence) is not dict or evidence
            or not isinstance(stop, asyncio.Event)):
        raise ValueError('LIVE_MAP_STREAM_CONFIGURATION_INVALID')
    for value, maximum in ((startup_timeout_seconds, 60.), (maximum_gap_seconds, 2.)):
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('LIVE_MAP_STREAM_DEADLINE_INVALID')
    evidence.update(status='starting', attempted=0, published=0, discarded_on_stop=0,
        pending={}, last_source_timestamp_ns=None, maximum_publication_gap_seconds=0.,
        compute_drained=False, motion_permission_granted=False, covariance_qualified=False)
    started = last_published = time.monotonic()
    pending = Counter()
    loop = asyncio.get_running_loop()
    previous_source = None
    try:
        while not stop.is_set():
            now = time.monotonic()
            deadline = (startup_timeout_seconds if not evidence['published']
                        else maximum_gap_seconds)
            if now - last_published > deadline:
                if optional_after_initialization and evidence['published']:
                    # 地图纠偏暂时缺失不关闭接收器；运动许可仍由独立实时定位、
                    # 协方差和局部安全通道决定。绝不重发旧位姿或刷新成功时刻。
                    evidence['status'] = 'awaiting-correction'
                    evidence['correction_gap_seconds'] = now - last_published
                elif (not evidence['published'] and evidence['attempted'] > 0
                        and pending['LIVE_MAP_PARTIAL_CONSTRAINT_ONLY'] >= .8 * evidence['attempted']):
                    raise RuntimeError('LIVE_MAP_STARTUP_GEOMETRY_INCOMPLETE')
                else:
                    raise RuntimeError('LIVE_MAP_STREAM_OBSERVATIONS_LOST')
            record = read_latest()
            if record is None:
                await asyncio.sleep(.005)
                continue
            evidence['attempted'] += 1
            try:
                # Single outstanding calculation, no queue of stale images.
                measurement = await _drainable_compute(loop.run_in_executor(None,
                    lambda record=record: source.measure(record, source_now_ns=source_now_ns,
                                           monotonic_now=time.monotonic)))
                # Recheck after returning to the event loop: dispatch scheduling
                # can age an otherwise valid compute result.
                LiveMapMeasurementSource._require_source_age(
                    measurement['source_timestamp_ns'], source_now_ns(),
                    maximum_ns=MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS)
            except MapMeasurementPending as error:
                # Only explicitly typed absence is transient; malformed records,
                # clock changes and identity errors terminate this run instance.
                key = str(error)
                if key not in pending and len(pending) >= 16:
                    key = 'OTHER_PENDING'
                pending[key] += 1
                evidence['pending'] = dict(pending)
                continue
            if stop.is_set():
                evidence['discarded_on_stop'] += 1
                break
            stamp = measurement['source_timestamp_ns']
            if type(stamp) is not int or previous_source is not None and stamp <= previous_source:
                raise ValueError('LIVE_MAP_STREAM_SOURCE_NOT_ADVANCING')
            remaining = deadline - (time.monotonic() - last_published)
            if remaining <= 0:
                if optional_after_initialization and evidence['published']:
                    # 新测量已经独立通过源钟/几何校验，只限制本次真实发送回读。
                    remaining = maximum_gap_seconds
                else:
                    raise RuntimeError('LIVE_MAP_STREAM_OBSERVATIONS_LOST')
            previous_source = stamp  # 即使此次回读丢失，也不得再次发送同一来源帧。
            async with asyncio.timeout(remaining):
                try:
                    await publish(measurement)
                except MapMeasurementPending as error:
                    # 回读未证实的发送不增加成功计数、不刷新上次成功时刻或源钟。
                    # 只接受传输明确声明的短暂缺失，仍在原有总断流期限内消费下一份新观测。
                    key = str(error)
                    if key not in pending and len(pending) >= 16:
                        key = 'OTHER_PENDING'
                    pending[key] += 1
                    evidence['pending'] = dict(pending)
                    continue
            now = time.monotonic()
            if evidence['published']:
                evidence['maximum_publication_gap_seconds'] = max(
                    evidence['maximum_publication_gap_seconds'], now-last_published)
            last_published = now
            evidence['published'] += 1
            evidence['last_source_timestamp_ns'] = stamp
            evidence['status'] = 'streaming'
        evidence['status'] = 'stopped'
    except asyncio.CancelledError:
        # 取消不是观测或算法失败；仍向所有者传播，绝不能标为成功完成。
        evidence['status'] = 'cancelled'
        evidence['stop_requested'] = stop.is_set()
        raise
    except BaseException as error:
        evidence['status'] = 'failed'
        evidence['issue'] = type(error).__name__ + ':' + str(error)[:240]
        raise
    finally:
        evidence['compute_drained'] = True
        evidence['elapsed_seconds'] = time.monotonic()-started
