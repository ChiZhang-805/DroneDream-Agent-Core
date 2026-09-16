"""Pace complete control iterations without scheduling catch-up bursts.

Sensor acquisition, dispatch, evidence and caller bookkeeping share one period.
Waiting at the end of only the dispatch helper hides caller work outside that
period and can make an advertised 20 Hz consumer run appreciably slower.
This scheduler does not grant authority or alter a sensor/command deadline.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from collections.abc import Awaitable, Callable


class ControlTickPacer:
    """Single-consumer monotonic cadence; never a sensor or authority clock."""

    # 功能：
    #   创建覆盖整轮工作的单消费者调度器，支持注入时钟用于可重复测试。
    # 输入：
    #   self：待初始化的调度器。
    #   rate_hz：目标完整循环频率，单位 Hz。
    #   clock：返回单调秒数的函数。
    #   sleep：按秒异步等待的函数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self, rate_hz: float, *, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if type(rate_hz) not in (int, float) or not 0 < rate_hz <= sys.float_info.max:
            raise ValueError("control cadence rate must be finite and positive")
        self._rate_hz = rate_hz
        self._period = 1.0 / rate_hz
        if not math.isfinite(self._period) or self._period <= 0:
            raise ValueError("control cadence period must be finite and positive")
        if not callable(clock) or not callable(sleep):
            raise ValueError("control cadence clock and sleep must be callable")
        self._clock, self._sleep = clock, sleep
        self._last_start: float | None = None
        self._ticks = 0
        self._minimum_interval: float | None = None
        self._maximum_interval = 0.0
        self._maximum_lateness = 0.0

    # 功能：
    #   提供只读目标频率，防止仅改标签而未改变实际周期。
    # 输入：
    #   self：调度器实例。
    # 输出：
    #   rate_hz：初始化时确定的 Hz 频率。
    @property
    def rate_hz(self) -> float:
        rate_hz = self._rate_hz
        return rate_hz

    # 功能：
    #   等待下一完整循环；早醒继续等待，迟到重新计时，不补发错过的动作。
    #   拒绝无效或回退的时钟，不增加成功轮数，也不改变传感器及指令期限。
    # 输入：
    #   self：每轮外部控制循环调用一次的单消费者调度器。
    # 输出：
    #   now：本轮开始时刻，使用单调秒数而非 UNIX 时间。
    async def wait(self) -> float:
        now = self._clock()
        if (type(now) not in (int, float) or not -sys.float_info.max <= now <= sys.float_info.max
                or (self._last_start is not None and now < self._last_start)):
            raise RuntimeError("CONTROL_CADENCE_CLOCK_INVALID")
        if self._last_start is not None:
            due = self._last_start + self._period
            # 绝对时刻太大可能吞掉一个周期；不能把舍入后的同一时刻当成下一轮。
            if not math.isfinite(due) or due <= self._last_start:
                raise RuntimeError("CONTROL_CADENCE_CLOCK_INVALID")
            while now < due:
                await self._sleep(due - now)
                next_now = self._clock()
                if (type(next_now) not in (int, float)
                        or not -sys.float_info.max <= next_now <= sys.float_info.max
                        or next_now < now):
                    raise RuntimeError("CONTROL_CADENCE_CLOCK_INVALID")
                now = next_now
            interval = now - self._last_start
            if not math.isfinite(interval * 1000):
                raise RuntimeError("CONTROL_CADENCE_CLOCK_INVALID")
            self._minimum_interval = (
                interval if self._minimum_interval is None
                else min(self._minimum_interval, interval)
            )
            self._maximum_interval = max(self._maximum_interval, interval)
            self._maximum_lateness = max(self._maximum_lateness, now - due)
        # A late iteration starts a new bounded period; never replay missed
        # ticks or issue a burst of stale actions to catch up with a schedule.
        self._last_start = now
        self._ticks += 1
        return now

    # 功能：
    #   返回实测循环间隔；这些统计不是推理延迟，也不代表有效动作频率。
    # 输入：
    #   self：具有累计调度统计的实例。
    # 输出：
    #   summary：目标频率、轮数、毫秒间隔及迟到统计。
    def summary(self) -> dict[str, object]:
        summary = {
            "target_rate_hz": self.rate_hz,
            "tick_count": self._ticks,
            "minimum_start_interval_ms": (
                None if self._minimum_interval is None else self._minimum_interval * 1000
            ),
            "maximum_start_interval_ms": self._maximum_interval * 1000,
            "maximum_start_lateness_ms": self._maximum_lateness * 1000,
            "catch_up_bursts_allowed": False,
        }
        return summary
