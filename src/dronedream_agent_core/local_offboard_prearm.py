"""Bounded local-mode handshake; never arms and never suppresses firmware checks."""

import asyncio
import math
import time


# 功能：只接受原生明确未解锁状态，未知值及数值零都不能冒充 False。
# 输入：read_armed：读取实际飞控解锁状态的异步回调。
# 输出：无；当前飞控不是明确未解锁则拒绝模式准备。
async def _assert_disarmed(read_armed):
    if await read_armed() is not False:
        raise RuntimeError("LOCAL_PREARM_REQUIRES_DISARMED")


# 功能：在实测原地保持流持续有效时切换本地控制模式，再等飞控正式健康回执。
# 输入：原生解锁读取、保持值发送、模式进入/退出、模式及健康验证回调；有限准备总期限。
#       evidence：本次握手证据；publish_hold 只能发送已核验的实测保持目标，不能发送任务路线。
# 输出：飞控健康回执；成功后由调用方接管已激活模式，失败/取消必定尝试退出，永不请求解锁。
async def prepare_local_offboard(*, read_armed, publish_hold, enter_mode, leave_mode,
                                verify_mode_and_health, evidence, timeout_seconds=12.):
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 3 <= timeout_seconds <= 60 or type(evidence) is not dict or evidence):
        raise ValueError("LOCAL_PREARM_CONFIGURATION_INVALID")
    evidence.update({"arm_requested": False, "mode_requested": False,
                     "mode_acknowledged": False, "hold_publications": 0,
                     "status": "preparing"})
    refresh = None
    succeeded = False
    previous_publication = None

    # 功能：记录实际保持流间隔，不把名义 20Hz 或消息总数当作连续性证明。
    # 输入：实测保持发布回调；单调时钟测量实际完成间隔。
    # 输出：更新连续性证据；超过 400ms 即拒绝准备，给固件 2Hz 要求保留余量。
    async def publish_measured_hold():
        nonlocal previous_publication
        await publish_hold()
        current = time.monotonic()
        if previous_publication is not None:
            interval = current - previous_publication
            evidence['maximum_hold_gap_seconds'] = max(
                evidence.get('maximum_hold_gap_seconds', 0.), interval)
            if interval < 0 or interval > .4:
                raise RuntimeError('LOCAL_PREARM_HOLD_STREAM_GAP')
        else:
            evidence['first_hold_monotonic'] = current
        previous_publication = current
        evidence['hold_span_seconds'] = current - evidence['first_hold_monotonic']
        evidence['hold_publications'] += 1

    try:
        async with asyncio.timeout(timeout_seconds):
            await _assert_disarmed(read_armed)
            first = asyncio.Event()

            # 功能：整个握手阶段持续刷新原地保持目标，不让健康读取间隙终止 SDK 设定值流。
            # 输入：闭包中已验证的保持发布回调。
            # 输出：实际发送计数；发送失败立即通过守护等待传播，不虚报模式就绪。
            async def keep_hold():
                while True:
                    await publish_measured_hold()
                    first.set()
                    await asyncio.sleep(.05)

            # 功能：同时监视具体操作和保持流，保持流故障不能等到总超时才发现。
            # 输入：本次异步操作。
            # 输出：操作回执；任何分支退出都回收尚未完成的操作任务。
            async def guarded(operation):
                task = asyncio.create_task(operation)
                try:
                    done, _ = await asyncio.wait(
                        (task, refresh), return_when=asyncio.FIRST_COMPLETED,
                    )
                    if refresh in done:
                        await refresh
                        raise RuntimeError("LOCAL_PREARM_HOLD_STREAM_ENDED")
                    return await task
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            refresh = asyncio.create_task(keep_hold())
            await guarded(first.wait())
            # PX4 requires >1s of setpoint proof-of-life before Offboard entry.
            await guarded(asyncio.sleep(1.2))
            await guarded(_assert_disarmed(read_armed))

            # 功能：模式请求前立即重发实测保持值，避免等待遥测期间 SDK 旧模式心跳撤销设定值。
            # 输入：同一保持目标与模式接口；不追加任务目标或重新解锁。
            # 输出：本次模式回执；在发送模式命令前登记，丢失回执也必须退出清理。
            async def enter_with_hold():
                await publish_measured_hold()
                evidence["mode_requested"] = True
                await enter_mode()

            await guarded(enter_with_hold())
            evidence["mode_acknowledged"] = True
            health = await guarded(verify_mode_and_health())
            if any(getattr(health, name, None) is not True for name in
                   ("connected", "home_position_ok", "local_position_ok", "armable")):
                raise RuntimeError("LOCAL_PREARM_FIRMWARE_NOT_READY")
            await guarded(_assert_disarmed(read_armed))
            evidence["still_disarmed"] = True
            succeeded = True
            evidence["status"] = "ready"
            return health
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["issue"] = type(error).__name__ + ":" + str(error)[:300]
        raise
    finally:
        if refresh is not None:
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        if not succeeded and evidence["mode_requested"]:
            try:
                await asyncio.wait_for(leave_mode(), 3.)
                evidence["failure_mode_cleanup"] = "acknowledged"
            except (Exception, asyncio.CancelledError) as error:
                evidence["failure_mode_cleanup"] = type(error).__name__ + ":" + str(error)[:200]
