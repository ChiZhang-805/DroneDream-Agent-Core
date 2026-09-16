"""Independent native-state publication without competing MAVSDK subscriptions.

The client owns sensor collectors. This task reads those caches and performs
at most one file publication at a time outside the control event loop. Slow
disk I/O does not create an unbounded queue or refresh source timestamps.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

CLOSE_TIMEOUT_SECONDS = 2.0


class NativeTelemetryPublisher:
    """Publish cached native observations without competing sensor subscriptions."""

    # 功能：
    #   准备单路原生缓存发布器，不创建新的传感器订阅；目标频率不代表实测采样率。
    # 输入：
    #   client：拥有原生采集器、提供位置和动力学缓存的客户端。
    #   publish：在线程中执行的同步发布回调。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, client: Any, publish: Callable[[Any, dict], None]) -> None:
        if (not callable(publish) or inspect.iscoroutinefunction(publish)
                or inspect.iscoroutinefunction(inspect.getattr_static(
                    type(publish), "__call__", None))):
            raise TypeError("NATIVE_PUBLISHER_REQUIRES_SYNC_CALLBACK")
        self.client, self.publish = client, publish
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.summary: dict[str, object] = {
            "target_rate_hz": 50., "published_count": 0, "unique_source_count": 0,
            "error_count": 0, "last_issue": None, "drained": False,
        }
        self._last_sources: tuple | None = None

    # 功能：
    #   在调用者正在运行的事件循环中启动唯一发布任务；已关闭实例不能重启。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("native telemetry publisher already closed")
        if self._task is not None:
            raise RuntimeError("native telemetry publisher already started")
        # 先取循环再创建协程，避免没有活动循环时遗留从未等待的协程对象。
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run(), name="native-state-publication")

    # 功能：
    #   同步执行单次发布并拒绝未被执行的异步返回值，避免把创建协程算作写入成功。
    # 输入：
    #   observed：本次独占的位置和速度快照。
    #   dynamics：本次独占的动力学快照。
    # 输出：
    #   None：不返回业务数据。
    def _publish_once(self, observed: Any, dynamics: dict) -> None:
        result = self.publish(observed, dynamics)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("NATIVE_PUBLISHER_RETURNED_AWAITABLE")

    # 功能：
    #   1. 串行读取并复制原生缓存，以同一快照校验来源、计数和在线程中发布。
    #   2. 相邻来源标识变化与写入次数分别统计，重复缓存不虚增独立采样声明。
    #   3. 失败只记录错误，不伪造位姿或刷新旧文件期限；停止后等待当前写入完成。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                observed = copy.deepcopy(await self.client.sample_position_velocity_ned(.1))
                dynamics = copy.deepcopy(self.client.latest_dynamics_telemetry(.25))
                received = getattr(observed, "received_at_unix_ms", None)
                if type(received) is not int or received < 0:
                    raise ValueError("NATIVE_POSITION_RECEIVE_TIME_MISSING")
                if not isinstance(dynamics, dict):
                    raise ValueError("NATIVE_DYNAMICS_PACKET_INVALID")
                sources = dynamics.get("sources", {})
                if (not isinstance(sources, dict) or any(
                    not isinstance(name, str) or not isinstance(payload, dict)
                    for name, payload in sources.items()
                )):
                    raise ValueError("NATIVE_DYNAMICS_SOURCE_INVALID")
                identities = []
                for name, payload in sorted(sources.items()):
                    clocks = (payload.get("timestamp_us"), payload.get("received_at_unix_ms"))
                    if any(clock is not None and (type(clock) is not int or clock < 0)
                           for clock in clocks):
                        raise ValueError("NATIVE_DYNAMICS_SOURCE_TIME_INVALID")
                    identities.append((name, *clocks))
                source_ids = (received, *identities)
                # 拷贝在跨线程前完成：校验、统计和实际写入不能引用不同时间的缓存内容。
                # 相邻标识变化计数不是所有传感器各自的完整独立样本计数或质量评估。
                await asyncio.to_thread(self._publish_once, observed, dynamics)
                self.summary["published_count"] += 1
                if source_ids != self._last_sources:
                    self.summary["unique_source_count"] += 1
                    self._last_sources = source_ids
                self.summary["last_issue"] = None
            except Exception as error:
                self.summary["error_count"] += 1
                self.summary["last_issue"] = f"{type(error).__name__}:{error}"
                # 原文件及原始期限保持不变，消费者继续按期限拒绝过期状态。
            remaining = max(0., .02 - (time.monotonic() - started))
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=remaining)
        self.summary["drained"] = True

    # 功能：
    #   请求停止并有限等待排空；超时保留在途写入的任务归属，随后可再次等待完成。
    #   调用者取消也不取消后台写入；只有实际结束或从未启动时才报告 drained。
    # 输入：
    #   无。
    # 输出：
    #   summary：独立的发布统计及当前排空结果，未排空不能当作已关闭。
    async def close(self) -> dict[str, object]:
        self._stop.set()
        if self._task is not None:
            try:
                # 取消 to_thread 的等待不会杀死写线程；保留所有者才能等到真实结束。
                await asyncio.wait_for(asyncio.shield(self._task), timeout=CLOSE_TIMEOUT_SECONDS)
            except TimeoutError:
                self.summary["last_issue"] = "NATIVE_PUBLICATION_DID_NOT_DRAIN"
        else:
            self.summary["drained"] = True
        summary = dict(self.summary)
        return summary
