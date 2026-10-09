"""Run-owned PX4/Gazebo measurement transport; not a hardware or release gate.

Only a verified shared simulator clock may use this transport. The owner must
own/terminate the isolated PX4 process and configure estimator parameters before
calling it. This module never changes parameters, arms, or issues flight targets.
"""

import asyncio
import re
import socket
import time

from .external_vision_clock import (
    SharedSimulationClock,
    validate_external_vision_pose_readback,
    validate_external_vision_time_readback,
)
from .live_map_measurement import MapMeasurementPending
from .live_map_measurement_stream import run_map_measurement_stream

# Reply count is not convergence evidence.  A short handshake only proves that
# the dedicated link is bidirectional; every measurement is subsequently
# checked against PX4's actual timestamp, reset counter and pose readback.
MINIMUM_TIMESYNC_HANDSHAKES = 10


def drain_timesync(link, clock, evidence):
    """Answer a bounded batch of valid TIMESYNC requests on the owned MAVLink link."""
    for _ in range(64):
        message = link.recv_match(blocking=False)
        if message is None:
            break
        if message.get_type() != 'TIMESYNC' or message.tc1 != 0:
            continue
        response = clock.reply(tc1=message.tc1, ts1=message.ts1,
            system_id=message.get_srcSystem(), component_id=message.get_srcComponent(),
            now_monotonic=time.monotonic())
        if response is not None:
            link.mav.timesync_send(*response)
            evidence['timesync_replies'] += 1


def check_measurement_readback(readback, measurement, *, reset_counter):
    """Validate one firmware readback and return its source-time error."""
    observed = {}
    for name in ('timestamp', 'timestamp_sample', 'reset_counter'):
        matches = re.findall(r'^\s*' + name + r': (\d+)', readback, re.M)
        if len(matches) != 1:
            raise ValueError('SITL_MAP_VISION_READBACK_AMBIGUOUS')
        observed[name] = int(matches[0])
    difference = validate_external_vision_time_readback(
        source_timestamp_us=measurement['source_timestamp_ns']//1000,
        received_sample_us=observed['timestamp_sample'], received_publish_us=observed['timestamp'],
        expected_reset_counter=reset_counter, received_reset_counter=observed['reset_counter'],
        maximum_time_error_us=measurement.get('transport_clock_uncertainty_us', 2_000))
    validate_external_vision_pose_readback(readback=readback,
        position=measurement['position_ned_m'],
        quaternion=measurement['quaternion_ned_from_frd_wxyz'])
    return difference


async def wait_measurement_readback(command, measurement, *, reset_counter, timeout_seconds=.15):
    """Wait briefly for the current measurement while rejecting malformed readbacks."""
    source_us = measurement['source_timestamp_ns']//1000
    tolerance = measurement.get('transport_clock_uncertainty_us', 2_000)
    if type(tolerance) is not int or not 0 < tolerance <= 10_000:
        raise ValueError('EXTERNAL_VISION_TIME_UNCERTAINTY_INVALID')
    async with asyncio.timeout(timeout_seconds):
        while True:
            readback = await command('listener', 'vehicle_visual_odometry', '-n', '1')
            # 初次发送到固件与主题首次发布是异步的；仅原生明确的“未发布”允许短暂等待。
            # 空响应、重复字段、错误位姿仍立即拒绝，期限和来源时刻均不改变。
            if readback.strip() == 'never published':
                await asyncio.sleep(.005)
                continue
            stamps = re.findall(r'^\s*timestamp_sample: (\d+)', readback, re.M)
            if len(stamps) != 1:
                raise ValueError('SITL_MAP_VISION_READBACK_AMBIGUOUS')
            if int(stamps[0]) < source_us - tolerance:
                await asyncio.sleep(.005)
                continue
            error = check_measurement_readback(readback, measurement, reset_counter=reset_counter)
            return readback, error


async def run_sitl_map_vision_transport(*, source, read_latest, command, clock_domain,
                                       stop, evidence, on_verified=None,
                                       optional_after_initialization=False):
    """Stream map measurements to isolated PX4 and verify each clock and pose readback."""
    if type(evidence) is not dict or evidence:
        raise ValueError('SITL_MAP_VISION_EVIDENCE_INVALID')
    # Imported only in the Linux simulator, never during Windows UI startup.
    from gz.msgs10.clock_pb2 import Clock
    from gz.transport13 import Node
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common

    clock = SharedSimulationClock(clock_domain=clock_domain, system_id=1, component_id=1)
    node, link, subscribed = Node(), None, False
    sync_task = stream_task = None
    clock_issue = None
    evidence.update(timesync_replies=0, readbacks=0, maximum_time_error_us=0,
        motion_permission_granted=False, covariance_qualified=False, stream={})

    def receive_clock(message):
        """Observe the raw Gazebo clock without extrapolating from host time."""
        nonlocal clock_issue
        try:
            clock.observe(int(message.sim.sec)*1_000_000_000+int(message.sim.nsec),
                          received_monotonic=time.monotonic())
        except ValueError as error:
            clock_issue = str(error)

    try:
        subscribed = node.subscribe(Clock, '/clock', receive_clock)
        if not subscribed:
            raise RuntimeError('SITL_MAP_VISION_CLOCK_SUBSCRIBE_FAILED')
        link = mavutil.mavlink_connection('udpin:127.0.0.1:0', source_system=1,
                                         source_component=197)
        client_port = link.port.getsockname()[1]
        link.mav = common.MAVLink(link, srcSystem=1, srcComponent=197)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(('127.0.0.1', 0))
            px4_port = reservation.getsockname()[1]
        evidence['link_start'] = await command('mavlink', 'start', '-u', str(px4_port),
            '-o', str(client_port), '-t', '127.0.0.1', '-m', 'custom', '-r', '200000')
        await command('mavlink', 'stream', '-u', str(px4_port), '-s', 'TIMESYNC', '-r', '200')
        exchanged = asyncio.Event()

        async def synchronize():
            """Serve time synchronization until the owning run requests shutdown."""
            try:
                while not stop.is_set():
                    if clock_issue is not None:
                        raise RuntimeError(clock_issue)
                    drain_timesync(link, clock, evidence)
                    if evidence['timesync_replies'] >= MINIMUM_TIMESYNC_HANDSHAKES:
                        exchanged.set()
                    await asyncio.sleep(.001)
            finally:
                exchanged.set()

        sync_task = asyncio.create_task(synchronize())
        await asyncio.wait_for(exchanged.wait(), 10.)
        if sync_task.done():
            await sync_task
            if stop.is_set():
                return
            raise RuntimeError('SITL_MAP_VISION_CLOCK_ENDED')

        async def publish(measurement):
            """Publish one source-timestamped pose and record it only after readback."""
            unknown_velocity = [0.] * 21
            for slot in (0, 6, 11, 15, 18, 20):
                unknown_velocity[slot] = 100.
            link.mav.odometry_send(measurement['source_timestamp_ns']//1000, 1, 1,
                *measurement['position_ned_m'], list(measurement['quaternion_ned_from_frd_wxyz']),
                0., 0., 0., 0., 0., 0., list(measurement['pose_covariance_upper']),
                unknown_velocity, reset_counter=71, estimator_type=2, quality=0)
            try:
                readback, error = await wait_measurement_readback(
                    command, measurement, reset_counter=71)
            except TimeoutError as error:
                # 单次回读丢失不能算验证成功，也不重发旧观测；连续流监督器仍限制总缺失时间。
                # 后续必须取得新观测和新的固件回读；身份/位姿/时钟损坏不在此重试范围内。
                raise MapMeasurementPending('LIVE_MAP_READBACK_TIMEOUT') from error
            evidence['readbacks'] += 1
            evidence['maximum_time_error_us'] = max(evidence['maximum_time_error_us'], abs(error))
            if on_verified is not None:
                on_verified(measurement, readback)

        stream_task = asyncio.create_task(run_map_measurement_stream(source=source,
            read_latest=read_latest, source_now_ns=lambda: clock.current(
                now_monotonic=time.monotonic()), publish=publish, stop=stop,
            evidence=evidence['stream'],
            optional_after_initialization=optional_after_initialization))
        done, _ = await asyncio.wait((sync_task, stream_task), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
        if not stop.is_set():
            raise RuntimeError('SITL_MAP_VISION_STREAM_ENDED')
        await stream_task
    except BaseException as error:
        evidence['issue'] = type(error).__name__ + ':' + str(error)[:240]
        raise
    finally:
        for task in (sync_task, stream_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (sync_task, stream_task) if task is not None),
                             return_exceptions=True)
        if subscribed:
            node.unsubscribe('/clock')
        if link is not None:
            link.close()
        evidence['closed'] = True
