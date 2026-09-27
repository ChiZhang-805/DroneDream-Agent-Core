"""Bounded process isolation for native payload requests."""

import asyncio
import sys
from pathlib import Path

import pytest

from dronedream_agent_core.gazebo_payload_service import PayloadPoseService, validate_request

_POSE = [0., 0., 1., 0., 0., 0., 1.]
_FAKE_WORKER = Path(__file__).parent / 'fixtures' / 'payload_service_worker.py'


# 功能：
#   使用独立测试进程模拟延迟与损坏响应，保留真实管道、取消和进程回收行为。
# 输入：
#   monkeypatch：只替换待启动的程序路径。
# 输出：
#   processes：真实创建的子进程列表。
@pytest.fixture
def processes(monkeypatch):
    original = asyncio.create_subprocess_exec
    processes = []

    # 功能：
    #   将生产子进程入口替换成协议测试服务，不在单测中访问实际 Gazebo。
    # 输入：
    #   args、kwargs：原启动参数与管道选项。
    # 输出：
    #   process：拥有真实 PID 的测试进程。
    async def spawn(*args, **kwargs):
        process = await original(sys.executable, '-u', str(_FAKE_WORKER), **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    return processes


# 功能：
#   请求等待期间事件循环仍可推进，连续请求复用同一进程并绑定不同序号。
# 输入：
#   processes：隔离协议服务。
# 输出：
#   None：调度、序号、复用和关闭断言通过。
def test_waiting_request_does_not_block_control(processes):
    async def scenario():
        service = PayloadPoseService()
        task = asyncio.create_task(service.set_pose(world='delay', model='payload',
                                                    pose=_POSE, timeout_ms=500))
        ticks = 0
        while not task.done():
            await asyncio.sleep(.01)
            ticks += 1
        assert (await task)['sequence'] == 1
        assert ticks >= 10
        result = await service.set_pose(world='ok', model='payload', pose=_POSE, timeout_ms=500)
        assert result['sequence'] == 2
        assert len(processes) == 1
        await service.close()
        assert processes[0].returncode is not None
        with pytest.raises(RuntimeError, match='closed'):
            await service.set_pose(world='ok', model='payload', pose=_POSE, timeout_ms=500)
    asyncio.run(scenario())


# 功能：
#   错误序号、拒绝、输出超长、进程退出以及请求超时均使通道失效并回收进程。
# 输入：
#   mode：测试服务响应场景。
#   processes：隔离协议服务。
# 输出：
#   None：所有失败均不能留下可复用请求通道。
@pytest.mark.parametrize('mode', ['wrong', 'reject', 'oversized', 'exit', 'timeout'])
def test_failed_request_closes_process(mode, processes):
    async def scenario():
        service = PayloadPoseService()
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            await service.set_pose(world=mode, model='payload', pose=_POSE, timeout_ms=30)
        assert service._closed
        assert processes[0].returncode is not None
    asyncio.run(scenario())


# 功能：
#   取消尚在等待的设备服务时必须先杀掉并回收唯一拥有的进程，并拒绝并发请求。
# 输入：
#   processes：隔离协议服务。
# 输出：
#   None：取消、并发保护和资源回收断言通过。
def test_cancelled_request_is_reaped(processes):
    async def scenario():
        service = PayloadPoseService()
        task = asyncio.create_task(service.set_pose(world='timeout', model='payload',
                                                    pose=_POSE, timeout_ms=1000))
        while service._sequence == 0:
            await asyncio.sleep(.01)
        with pytest.raises(RuntimeError, match='already in use'):
            await service.set_pose(world='ok', model='payload', pose=_POSE, timeout_ms=1000)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None
        assert service._closed
    asyncio.run(scenario())


# 功能：
#   原生服务启动让出调度后，调用方修改输入不得改变已经检查并准备发送的位姿。
# 输入：
#   monkeypatch：启动边界的独立进程替身。
# 输出：
#   None：发送内容仍为调用时被冻结的位姿。
def test_pose_is_frozen_before_worker_start(monkeypatch):
    async def scenario():
        import json

        pose = list(_POSE)
        writes = []

        class Input:
            def write(self, content):
                writes.append(json.loads(content))

            async def drain(self):
                pass

            def close(self):
                pass

        class Output:
            def __init__(self):
                self.lines = iter((b'{"ready":true}\n', b'{"sequence":1,"accepted":true}\n'))

            async def readline(self):
                return next(self.lines)

        class Process:
            returncode = None

            def __init__(self):
                self.stdin, self.stdout = Input(), Output()

            def kill(self):
                self.returncode = -9

            async def wait(self):
                return self.returncode

        async def spawn(*args, **kwargs):
            pose[0] = 999.
            await asyncio.sleep(0)
            return Process()

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
        service = PayloadPoseService()
        await service.set_pose(world='world', model='payload', pose=pose, timeout_ms=500)
        await service.close()
        assert writes[0]['pose'] == _POSE
        assert pose[0] == 999.

    asyncio.run(scenario())


# 功能：
#   协议边界拒绝非有限位姿、非法名称、非单位旋转及伪装整数。
# 输入：
#   updates：对有效请求的非法修改。
# 输出：
#   None：非法请求在原生服务发送前被拒绝。
@pytest.mark.parametrize('updates', [dict(world='../other'), dict(pose=[float('nan')] * 7),
    dict(pose=[0.] * 7), dict(timeout_ms=True), dict(sequence=True), dict(extra='service')])
def test_invalid_native_request_rejected(updates):
    request = dict(world='world', model='model', pose=_POSE, timeout_ms=100, sequence=1)
    request.update(updates)
    with pytest.raises(ValueError):
        validate_request(request)
