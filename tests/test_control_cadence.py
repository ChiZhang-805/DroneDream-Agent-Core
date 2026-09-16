import asyncio
import math
from types import SimpleNamespace

import pytest
from test_runtime_commands import _load_executor

from dronedream_agent_core.control_cadence import ControlTickPacer


class Clock:
    # 功能：
    #   创建可手动推进的单调测试时钟，记录等待量而不消耗真实时间。
    # 输入：
    #   self：测试时钟实例。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.now = 10.
        self.sleeps = []

    # 功能：
    #   记录一次等待并同步推进测试时刻。
    # 输入：
    #   self：测试时钟实例。
    #   seconds：模拟等待秒数。
    # 输出：
    #   None：不返回业务数据。
    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    # 功能：
    #   将测试时钟和等待函数绑定到整周期 20 Hz 调度器。
    # 输入：
    #   self：测试时钟实例。
    # 输出：
    #   pacer：不会访问真实时间的节奏控制器。
    def pacer(self):
        pacer = ControlTickPacer(20., clock=lambda: self.now, sleep=self.sleep)
        return pacer


# 功能：
#   验证传输函数内外的耗时共同计入一个周期，不各自额外等待。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cadence_accounts_for_dispatch_and_caller_work_together():
    clock = Clock()
    pacer = clock.pacer()

    # 功能：
    #   模拟每轮传感器传输及调用者记账，采集连续五轮开始时刻。
    # 输入：
    #   clock、pacer：闭包中的测试时钟与调度器。
    # 输出：
    #   starts：每轮开始的单调秒数列表。
    async def run():
        starts = [await pacer.wait()]
        for _ in range(4):
            clock.now += .015  # Telemetry and transport inside the helper.
            clock.now += .020  # Target publication/tracking outside helper.
            starts.append(await pacer.wait())
        return starts

    assert asyncio.run(run()) == pytest.approx([10., 10.05, 10.10, 10.15, 10.20])
    assert clock.sleeps == pytest.approx([.015] * 4)
    assert pacer.summary()["maximum_start_interval_ms"] == pytest.approx(50.)
    assert pacer.summary()["tick_count"] == 5


# 功能：
#   验证超时只重设下一周期基准，不补发已经错过的控制指令。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_overrun_does_not_sleep_again_or_replay_missed_ticks():
    clock = Clock()
    pacer = clock.pacer()

    # 功能：
    #   模拟跨过两个周期，验证恢复后仍按正常间隔推进。
    # 输入：
    #   clock、pacer：闭包中的测试时钟与调度器。
    # 输出：
    #   None：不返回业务数据。
    async def run():
        await pacer.wait()
        clock.now += .125
        assert await pacer.wait() == pytest.approx(10.125)
        assert not clock.sleeps
        assert await pacer.wait() == pytest.approx(10.175)

    asyncio.run(run())
    summary = pacer.summary()
    assert summary["maximum_start_lateness_ms"] == pytest.approx(75.)
    assert summary["minimum_start_interval_ms"] == pytest.approx(50.)
    assert summary["catch_up_bursts_allowed"] is False


# 功能：
#   验证调度频率及其倒数必须可表示为有限正数。
# 输入：
#   rate：非法频率，单位 Hz。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rate", [
    True, 0., -1., math.inf, math.nan, None, "20", [], 5e-324,
    pytest.param(10**1000, id="oversized-rate"),
])
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        ControlTickPacer(rate)


# 功能：
#   验证时钟回退或非法读数不能开始下一轮，也不能增加有效轮数。
# 输入：
#   value：第一轮之后的非法时钟读数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [
    9., math.inf, math.nan, None, "11", True,
    pytest.param(10**1000, id="oversized-clock"),
])
def test_clock_discontinuity_cannot_authorize_an_immediate_tick(value):
    clock = Clock()
    pacer = clock.pacer()

    # 功能：
    #   在已启动一轮后破坏时钟，检查调度器报告固定类别的错误。
    # 输入：
    #   clock、pacer、value：闭包中的时钟、调度器及非法读数。
    # 输出：
    #   None：不返回业务数据。
    async def run():
        await pacer.wait()
        clock.now = value
        with pytest.raises(RuntimeError, match="CLOCK_INVALID"):
            await pacer.wait()

    asyncio.run(run())
    assert pacer.summary()["tick_count"] == 1


# 功能：
#   验证执行器复用同一个调度器，并拒绝运行中偷偷改变控制频率。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_executor_preserves_cadence_between_separate_helper_calls():
    executor = _load_executor()
    clock = Clock()
    pacer = clock.pacer()
    args = SimpleNamespace(setpoint_rate_hz=20., _local_control_pacer=pacer)

    # 功能：
    #   调用两次真实执行器入口后改变配置，验证执行器拒绝继续。
    # 输入：
    #   executor、args、clock：闭包中的执行器、参数与测试时钟。
    # 输出：
    #   None：不返回业务数据。
    async def run():
        await executor._begin_local_control_tick(args)
        clock.now += .035
        await executor._begin_local_control_tick(args)
        assert clock.now == pytest.approx(10.05)
        args.setpoint_rate_hz = 30.
        with pytest.raises(executor.UserDirectedLanding, match="CADENCE_CHANGED"):
            await executor._begin_local_control_tick(args)

    asyncio.run(run())
    assert args._local_control_pacer is pacer
    assert pacer.summary()["tick_count"] == 2


# 功能：
#   验证初始时钟也严格校验，不能先把非法值计入一轮成功执行。
# 输入：
#   value：非法初始单调时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "10", math.inf,
                                   pytest.param(10**1000, id="oversized-clock")])
def test_invalid_initial_clock_is_not_counted(value):
    clock = Clock()
    clock.now = value
    pacer = clock.pacer()
    with pytest.raises(RuntimeError, match="CLOCK_INVALID"):
        asyncio.run(pacer.wait())
    assert pacer.summary()["tick_count"] == 0


# 功能：
#   验证时钟量级过大导致周期加法无效时，调度器不会立即重复发起一轮。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unrepresentable_next_tick_is_rejected():
    clock = Clock()
    clock.now = 1e308
    pacer = clock.pacer()
    asyncio.run(pacer.wait())
    with pytest.raises(RuntimeError, match="CLOCK_INVALID"):
        asyncio.run(pacer.wait())
    assert pacer.summary()["tick_count"] == 1


# 功能：
#   验证公开频率不能被单独修改，避免其与内部周期和验收摘要不一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_target_rate_cannot_be_relabelled_after_initialization():
    pacer = Clock().pacer()
    with pytest.raises(AttributeError):
        pacer.rate_hz = 30
    assert pacer.summary()["target_rate_hz"] == 20


# 功能：
#   验证提前唤醒后再次等待剩余时间，而不是提前开始控制轮次。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_early_wakeup_waits_for_remaining_interval():
    clock = Clock()
    waits = []

    # 功能：
    #   首次仅等待请求的一半，之后按请求完整推进测试时钟。
    # 输入：
    #   seconds：本次请求等待秒数。
    #   clock、waits：闭包中的时钟与等待记录。
    # 输出：
    #   None：不返回业务数据。
    async def early_sleep(seconds):
        waits.append(seconds)
        clock.now += seconds / 2 if len(waits) == 1 else seconds

    pacer = ControlTickPacer(20, clock=lambda: clock.now, sleep=early_sleep)
    asyncio.run(pacer.wait())
    assert asyncio.run(pacer.wait()) == pytest.approx(10.05)
    assert waits == pytest.approx([.05, .025])
    assert pacer.summary()["tick_count"] == 2


# 功能：
#   验证等待中的取消向调用者传播，未开始的一轮不进入成功统计。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancelled_wait_does_not_count_a_tick():
    clock = Clock()

    # 功能：
    #   模拟外部关闭控制循环时取消等待。
    # 输入：
    #   seconds：调度器请求等待的秒数。
    # 输出：
    #   None：不返回业务数据。
    async def cancelled_sleep(seconds):
        raise asyncio.CancelledError

    pacer = ControlTickPacer(20, clock=lambda: clock.now, sleep=cancelled_sleep)
    asyncio.run(pacer.wait())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pacer.wait())
    assert pacer.summary()["tick_count"] == 1


# 功能：
#   验证等待前读数合法但等待后时钟损坏时，仍拒绝发布本轮。
# 输入：
#   value：等待后的非法时钟读数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "10", 9., float("nan"), float("inf"),
                                   pytest.param(10**1000, id="oversized-clock")])
def test_invalid_clock_after_sleep_is_not_counted(value):
    clock = Clock()

    # 功能：
    #   在调度等待期间注入非法时钟值以检查唤醒后的验证分支。
    # 输入：
    #   seconds：请求等待秒数。
    #   clock、value：闭包中的时钟及替换值。
    # 输出：
    #   None：不返回业务数据。
    async def broken_sleep(seconds):
        clock.now = value

    pacer = ControlTickPacer(20, clock=lambda: clock.now, sleep=broken_sleep)
    asyncio.run(pacer.wait())
    with pytest.raises(RuntimeError, match="CLOCK_INVALID"):
        asyncio.run(pacer.wait())
    assert pacer.summary()["tick_count"] == 1
