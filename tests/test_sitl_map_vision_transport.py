"""Source-clock readback and cleanup, without importing simulator libraries."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dronedream_agent_core import sitl_map_vision_transport as transport


# 功能：创建明确原始源时刻的传输夹具；输入：无；输出：未授权位置测量。
def measurement():
    return dict(source_timestamp_ns=1_000_000_000, position_ned_m=(1., 2., 3.),
        quaternion_ned_from_frd_wxyz=(1., 0., 0., 0.), pose_covariance_upper=(.01,)*21)


# 功能：提供固件文本格式的测试回读；输入：原始时刻及重置计数；输出：精确文本。
def readback(stamp=1_000_000, reset=71):
    return (f'timestamp: 1050000\ntimestamp_sample: {stamp}\nreset_counter: {reset}\n'
            'position: [1.0, 2.0, 3.0]\nq: [1.0, 0.0, 0.0, 0.0]\n')


# 功能：只接受测量器显式计入协方差的有限同步误差，其他身份/位姿检查保持不变。
@pytest.mark.parametrize('offset', [-8001, -8000, -2047, 2047, 8000, 8001])
def test_explicit_clock_error_budget(offset):
    value = {**measurement(), 'transport_clock_uncertainty_us': 8000}
    if abs(offset) > 8000:
        with pytest.raises(ValueError, match='NOT_PRESERVED'):
            transport.check_measurement_readback(readback(1_000_000+offset), value, reset_counter=71)
    else:
        assert transport.check_measurement_readback(readback(1_000_000+offset), value, reset_counter=71) == offset
    with pytest.raises(ValueError, match='COUNTER_LOST'):
        transport.check_measurement_readback(readback(reset=72), value, reset_counter=71)


# 功能：验证源钟、实体重置以及位姿回读不能互相替代；输入：篡改方式；输出：逐项拒绝。
@pytest.mark.parametrize('fault', ['none', 'retimed', 'reset', 'position', 'ambiguous'])
def test_firmware_readback_must_match_source(fault):
    text = readback()
    if fault == 'retimed':
        text = readback(stamp=1_010_000)
    elif fault == 'reset':
        text = readback(reset=72)
    elif fault == 'position':
        text = text.replace('[1.0, 2.0, 3.0]', '[2.0, 2.0, 3.0]')
    elif fault == 'ambiguous':
        text += 'timestamp_sample: 1000000\n'
    if fault == 'none':
        assert transport.check_measurement_readback(text, measurement(), reset_counter=71) == 0
    else:
        with pytest.raises(ValueError):
            transport.check_measurement_readback(text, measurement(), reset_counter=71)


# 功能：验证旧回读可短暂等待，但新回读损坏不能被重试吞掉。
# 输入：模拟返回顺序；输出：精确调用次数和原始错误，期限不改变来源年龄界。
@pytest.mark.parametrize('fault', ['none', 'position', 'future', 'stalled', 'not-published', 'empty', 'malformed'])
def test_readback_wait_is_bounded_and_does_not_hide_faults(fault):
    async def scenario():
        calls = []

        async def command(*args):
            calls.append(args)
            if len(calls) == 1 and fault in ('not-published', 'empty', 'malformed'):
                return {'not-published': 'never published\n', 'empty': '',
                        'malformed': 'never published\nunknown state'}[fault]
            if len(calls) == 1 or fault == 'stalled':
                return readback(stamp=900_000)
            if fault == 'position':
                return readback().replace('[1.0, 2.0, 3.0]', '[9.0, 2.0, 3.0]')
            return readback(stamp=1_010_000) if fault == 'future' else readback()

        if fault in ('none', 'not-published'):
            text, error = await transport.wait_measurement_readback(
                command, measurement(), reset_counter=71)
            assert text == readback() and error == 0 and len(calls) == 2
        else:
            with pytest.raises(TimeoutError if fault == 'stalled' else ValueError):
                await transport.wait_measurement_readback(
                        command, measurement(), reset_counter=71,
                        timeout_seconds=.025 if fault == 'stalled' else .5)
            if fault != 'stalled':
                assert len(calls) == (1 if fault in ('empty', 'malformed') else 2)

    asyncio.run(scenario())


# 功能：用最小仿真接口替身验证真实传输生命周期；输入：pytest 注入器与故障阶段。
# 输出：只请求时钟和视觉读回，无解锁或参数写入，任何失败都释放链路与订阅。
@pytest.mark.parametrize('fault', ['none', 'listener', 'stream', 'subscribe'])
def test_session_cleans_resources_without_motion_commands(monkeypatch, fault):
    calls = []

    class Node:
        def subscribe(self, *args):
            calls.append('subscribe')
            return fault != 'subscribe'

        def unsubscribe(self, topic):
            calls.append('unsubscribe')

    class Link:
        port = SimpleNamespace(getsockname=lambda: ('127.0.0.1', 43210))

        def close(self):
            calls.append('close-link')

    class Mav:
        def __init__(self, *args, **kwargs):
            pass

        def odometry_send(self, *args, **kwargs):
            calls.append(('odometry', args, kwargs))

    modules = {
        'gz.msgs10.clock_pb2': dict(Clock=object),
        'gz.transport13': dict(Node=Node),
        'pymavlink': dict(mavutil=SimpleNamespace(mavlink_connection=lambda *a, **kw: Link())),
        'pymavlink.dialects.v20': dict(common=SimpleNamespace(MAVLink=Mav)),
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)

    def drain(link, clock, evidence):
        evidence['timesync_replies'] += 650

    monkeypatch.setattr(transport, 'drain_timesync', drain)

    async def stream(**kwargs):
        if fault == 'stream':
            raise RuntimeError('stream-failed')
        await kwargs['publish'](measurement())
        kwargs['stop'].set()

    monkeypatch.setattr(transport, 'run_map_measurement_stream', stream)

    async def scenario():
        evidence, stop = {}, asyncio.Event()

        async def command(binary, *args):
            assert binary in ('mavlink', 'listener')
            calls.append((binary, args))
            if binary == 'listener':
                if fault == 'listener':
                    raise RuntimeError('listener-failed')
                return readback()
            return 'ok'

        kwargs = dict(source=None, read_latest=None, command=command,
            clock_domain='px4-gz-sitl:test', stop=stop, evidence=evidence)
        if fault != 'none':
            with pytest.raises(RuntimeError):
                await transport.run_sitl_map_vision_transport(**kwargs)
        else:
            await transport.run_sitl_map_vision_transport(**kwargs)
            assert evidence['readbacks'] == 1
            sent = next(item for item in calls if isinstance(item, tuple) and item[0] == 'odometry')
            assert sent[1][0] == 1_000_000  # Original source, not host time.
            assert sent[1][7:13] == (0.,)*6  # No reused native velocity.
            assert sent[2] == dict(reset_counter=71, estimator_type=2, quality=0)
        assert evidence['closed'] and not evidence['motion_permission_granted']
        if fault == 'subscribe':
            assert 'close-link' not in calls and 'unsubscribe' not in calls
        else:
            assert calls.count('close-link') == calls.count('unsubscribe') == 1

    asyncio.run(scenario())
