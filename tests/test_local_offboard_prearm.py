"""Handshake fault injection without a vehicle or fabricated flight acceptance."""

import asyncio
from types import SimpleNamespace

import pytest

from dronedream_agent_core.local_offboard_prearm import prepare_local_offboard


# 功能：建立只有模式控制而没有解锁能力的测试接口，并保留真实调用顺序。
# 输入：解锁读取值、发送故障、模式故障、健康回执故障。
# 输出：可注入的异步接口与调用记录；不是物理验收证据。
def fixture(*, armed=False, publish_failure=False, mode_failure=False, health_failure=False):
    calls = []

    async def read_armed():
        calls.append('read-armed')
        return armed

    async def publish_hold():
        calls.append('hold')
        if publish_failure:
            raise RuntimeError('hold-failed')

    async def enter_mode():
        calls.append('enter')
        if mode_failure:
            raise TimeoutError('lost-ack')

    async def leave_mode():
        calls.append('leave')

    async def verify():
        before = calls.count('hold')
        await asyncio.sleep(.12)
        assert calls.count('hold') > before  # No gap during health subscription.
        return SimpleNamespace(connected=True, home_position_ok=True, local_position_ok=True,
                               armable=not health_failure, global_position_ok=False)

    return dict(read_armed=read_armed, publish_hold=publish_hold, enter_mode=enter_mode,
                leave_mode=leave_mode, verify_mode_and_health=verify), calls


# 功能：仅在固件明确允许本地控制且仍未解锁时交接，不要求伪造全球位置。
# 输入：有效健康回执。
# 输出：持续保持、模式进入和未解锁的证据；不自动退出成功模式。
def test_success_keeps_stream_through_readiness_and_never_arms():
    callbacks, calls = fixture()
    evidence = {}
    health = asyncio.run(prepare_local_offboard(**callbacks, evidence=evidence))
    assert health.armable and not health.global_position_ok
    # Host scheduling need not deliver exactly 24 callbacks in 1.2 seconds.
    # Actual span/gaps, plus the real firmware health callback, are the contract.
    assert calls.count('hold') >= 4
    assert evidence['hold_span_seconds'] >= 1.2
    assert evidence['maximum_hold_gap_seconds'] <= .4
    assert calls.count('read-armed') == 3
    assert calls.count('enter') == 1 and 'leave' not in calls
    assert calls[calls.index('enter') - 1] == 'hold'
    assert evidence['still_disarmed'] and not evidence['arm_requested']


# 功能：未知/已解锁状态在任何保持流和模式命令前拒绝，不能把假值隐式转换为 False。
# 输入：非明确 False 的状态。
# 输出：无命令发送，握手失败。
@pytest.mark.parametrize('armed', [True, None, 0, 'false'])
def test_not_explicitly_disarmed_never_sends(armed):
    callbacks, calls = fixture(armed=armed)
    with pytest.raises(RuntimeError, match='REQUIRES_DISARMED'):
        asyncio.run(prepare_local_offboard(**callbacks, evidence={}))
    assert calls == ['read-armed']


# 功能：丢失模式回执或健康拒绝时退出已尝试的模式；发送流失败不能被等待掩盖。
# 输入：三个独立故障位置。
# 输出：只对实际尝试的模式发出退出，保留失败状态且没有解锁请求。
@pytest.mark.parametrize('failure', ['publish_failure', 'mode_failure', 'health_failure'])
def test_failure_does_not_leak_mode_or_claim_readiness(failure):
    callbacks, calls = fixture(**{failure: True})
    evidence = {}
    with pytest.raises((RuntimeError, TimeoutError)):
        asyncio.run(prepare_local_offboard(**callbacks, evidence=evidence))
    assert evidence['status'] == 'failed'
    assert calls.count('leave') == (failure != 'publish_failure')
    assert not evidence['arm_requested']


# 功能：在模式进入回执等待时取消，仍尝试停控并清理保持刷新任务。
# 输入：模式命令发送后一直不返回的异步接口。
# 输出：取消被传播，退出只调用一次，没有遗留后台刷新。
def test_cancellation_after_dispatch_exits_owned_mode():
    async def scenario():
        callbacks, calls = fixture()
        entered = asyncio.Event()

        async def enter():
            calls.append('enter')
            entered.set()
            await asyncio.Event().wait()

        callbacks['enter_mode'] = enter
        evidence = {}
        task = asyncio.create_task(prepare_local_offboard(**callbacks, evidence=evidence))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        count = len(calls)
        await asyncio.sleep(.1)
        assert len(calls) == count and calls.count('leave') == 1
        assert evidence['failure_mode_cleanup'] == 'acknowledged'

    asyncio.run(scenario())


# 功能：模式或健康等待期间保持流故障必须立即终止，不能返回一次旧的健康成功。
# 输入：进入模式后失败的发布器，以及永不返回的模式/健康回调。
# 输出：传播发送错误并收回已请求的模式；无遗留刷新任务。
@pytest.mark.parametrize('blocked_stage', ['enter_mode', 'verify_mode_and_health'])
def test_stream_failure_interrupts_mode_and_health_waits(blocked_stage):
    async def scenario():
        callbacks, calls = fixture()
        blocked = asyncio.Event()

        async def block():
            blocked.set()
            await asyncio.Event().wait()

        async def publish():
            calls.append('hold')
            if blocked.is_set():
                raise RuntimeError('stream-broke-during-handshake')

        callbacks[blocked_stage] = block
        callbacks['publish_hold'] = publish
        evidence = {}
        with pytest.raises(RuntimeError, match='stream-broke-during-handshake'):
            await prepare_local_offboard(**callbacks, evidence=evidence)
        assert calls.count('leave') == 1 and evidence['status'] == 'failed'

    asyncio.run(scenario())


# 功能：超时和退出故障都要保留，退出故障不能覆盖导致握手失败的原始异常。
# 输入：健康流无限等待，退出接口报错。
# 输出：总超时被传播，退出故障另记；没有自动解锁。
def test_total_timeout_preserves_cleanup_failure():
    async def scenario():
        callbacks, calls = fixture()

        async def never_ready():
            await asyncio.Event().wait()

        async def leave_failure():
            calls.append('leave')
            raise RuntimeError('cleanup-failed')

        callbacks['verify_mode_and_health'] = never_ready
        callbacks['leave_mode'] = leave_failure
        evidence = {}
        with pytest.raises(TimeoutError):
            await prepare_local_offboard(**callbacks, evidence=evidence, timeout_seconds=3.)
        assert evidence['failure_mode_cleanup'] == 'RuntimeError:cleanup-failed'
        assert evidence['issue'].startswith('TimeoutError:')
        assert not evidence['arm_requested'] and calls.count('leave') == 1

    asyncio.run(scenario())
