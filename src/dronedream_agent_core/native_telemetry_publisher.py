"""Independent native-state publication without competing MAVSDK subscriptions.

The client owns sensor collectors. The fast callback publishes every aligned
sample to the in-memory control channel. An optional, separately owned callback
coalesces durable evidence to the newest snapshot, so slow disk I/O cannot
throttle the control stream or create an unbounded queue.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .native_state_alignment import NativeReceiveAligner

CLOSE_TIMEOUT_SECONDS = 2.0
PUBLICATION_PERIOD_SECONDS = 0.05


class NativeTelemetryPublisher:
    """Publish cached native observations without competing sensor subscriptions."""

    # 功能：
    #   准备单路原生缓存发布器，不创建新的传感器订阅；目标频率不代表实测采样率。
    # 输入：
    #   client：拥有原生采集器、提供位置和动力学缓存的客户端。
    #   publish：在线程中执行的同步快速发布回调，不能包含持久化写入。
    #   persist：可选的同步证据回调；独立任务以最新值覆盖方式低频执行。
    #   persist_interval_seconds：两次常规证据写入之间的最短间隔。
    #   publication_rate_hz：缓存通道频率；相机配对可跟上 50 Hz 原生里程计，不增订阅。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        client: Any,
        publish: Callable[[Any, dict], None],
        *,
        persist: Callable[[Any, dict], None] | None = None,
        persist_interval_seconds: float = 0.5,
        publication_rate_hz: float = 1.0 / PUBLICATION_PERIOD_SECONDS,
    ) -> None:
        for name, callback in (("publish", publish), ("persist", persist)):
            if callback is None and name == "persist":
                continue
            if (not callable(callback) or inspect.iscoroutinefunction(callback)
                    or inspect.iscoroutinefunction(inspect.getattr_static(
                        type(callback), "__call__", None))):
                raise TypeError(f"NATIVE_PUBLISHER_REQUIRES_SYNC_{name.upper()}_CALLBACK")
        if (not isinstance(persist_interval_seconds, (int, float))
                or isinstance(persist_interval_seconds, bool)
                or not math.isfinite(persist_interval_seconds)
                or persist_interval_seconds <= 0):
            raise ValueError("NATIVE_PERSIST_INTERVAL_INVALID")
        if (type(publication_rate_hz) not in (int, float)
                or not 20 <= publication_rate_hz <= 100):
            raise ValueError("NATIVE_PUBLICATION_RATE_INVALID")
        self.client, self.publish, self.persist = client, publish, persist
        self.persist_interval_seconds = float(persist_interval_seconds)
        self._publication_period_seconds = 1.0 / publication_rate_hz
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._persist_task: asyncio.Task | None = None
        self._persist_event = asyncio.Event()
        self._publication_finished = asyncio.Event()
        self._latest_persist: tuple[int, Any, dict] | None = None
        self._persist_generation = 0
        self._persisted_generation = 0
        self.summary: dict[str, object] = {
            "target_rate_hz": float(publication_rate_hz),
            "published_count": 0,
            "unique_source_count": 0,
            "error_count": 0, "last_issue": None, "persisted_count": 0,
            "persist_error_count": 0, "last_persist_issue": None,
            "persist_interval_seconds": self.persist_interval_seconds,
            "drained": False,
        }
        self._last_sources: tuple | None = None
        self._aligner = NativeReceiveAligner()

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
        if self.persist is not None:
            self._persist_task = loop.create_task(
                self._run_persistence(), name="native-state-evidence-persistence"
            )

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
    #   同步执行单次持久化并拒绝异步返回值；该方法只由独立证据任务调用。
    # 输入：
    #   observed：已复制的位置和速度快照。
    #   dynamics：已复制的动力学快照。
    # 输出：
    #   None：不返回业务数据。
    def _persist_once(self, observed: Any, dynamics: dict) -> None:
        if self.persist is None:
            return
        result = self.persist(observed, dynamics)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("NATIVE_PERSIST_RETURNED_AWAITABLE")

    # 功能：
    #   保存最新待落盘快照并覆盖旧的未写快照，保证慢磁盘最多形成一个待处理项。
    # 输入：
    #   observed：本次位置和速度快照。
    #   dynamics：本次动力学快照。
    # 输出：
    #   None：不返回业务数据。
    def _stage_persistence(self, observed: Any, dynamics: dict) -> None:
        if self.persist is None:
            return
        self._persist_generation += 1
        self._latest_persist = (
            self._persist_generation,
            copy.deepcopy(observed),
            copy.deepcopy(dynamics),
        )
        self._persist_event.set()

    # 功能：
    #   独立于快速通道按最新值写证据；运行期合并中间快照，关闭时写完最后一份。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def _run_persistence(self) -> None:
        next_regular_write = 0.0
        while True:
            if self._latest_persist is None or (
                self._persisted_generation >= self._persist_generation
            ):
                if self._stop.is_set() and self._publication_finished.is_set():
                    break
                await self._persist_event.wait()
                self._persist_event.clear()
                continue

            if not self._stop.is_set():
                remaining = max(0.0, next_regular_write - time.monotonic())
                if remaining:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=remaining)

            pending = self._latest_persist
            if pending is None:
                continue
            generation, observed, dynamics = pending
            try:
                await asyncio.to_thread(self._persist_once, observed, dynamics)
                self.summary["persisted_count"] += 1
                self.summary["last_persist_issue"] = None
            except Exception as error:
                self.summary["persist_error_count"] += 1
                self.summary["last_persist_issue"] = f"{type(error).__name__}:{error}"
            finally:
                # Whether successful or failed, an attempted generation must not
                # spin forever. A newer sample schedules the next bounded attempt.
                self._persisted_generation = max(self._persisted_generation, generation)
                next_regular_write = time.monotonic() + self.persist_interval_seconds
            if self._persist_generation > generation:
                self._persist_event.set()
            if (self._stop.is_set() and self._publication_finished.is_set()
                    and self._persisted_generation >= self._persist_generation):
                break

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
                observed, dynamics = self._aligner.align(observed, dynamics)
                received = observed.received_at_unix_ms
                sources = dynamics.get("sources", {})
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
                self._stage_persistence(observed, dynamics)
                self.summary["published_count"] += 1
                if source_ids != self._last_sources:
                    self.summary["unique_source_count"] += 1
                    self._last_sources = source_ids
                self.summary["last_issue"] = None
            except Exception as error:
                self._aligner.clear()
                self.summary["error_count"] += 1
                self.summary["last_issue"] = f"{type(error).__name__}:{error}"
                # 原文件及原始期限保持不变，消费者继续按期限拒绝过期状态。
            remaining = max(0.0, self._publication_period_seconds - (time.monotonic() - started))
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=remaining)
        self._publication_finished.set()
        self._persist_event.set()
        self.summary["drained"] = self._persist_task is None or self._persist_task.done()

    # 功能：
    #   请求停止并有限等待排空；超时保留在途写入的任务归属，随后可再次等待完成。
    #   调用者取消也不取消后台写入；只有实际结束或从未启动时才报告 drained。
    # 输入：
    #   无。
    # 输出：
    #   summary：独立的发布统计及当前排空结果，未排空不能当作已关闭。
    async def close(self) -> dict[str, object]:
        self._stop.set()
        self._persist_event.set()
        tasks = tuple(task for task in (self._task, self._persist_task) if task is not None)
        deadline = time.monotonic() + CLOSE_TIMEOUT_SECONDS
        for task in tasks:
            try:
                # 取消 to_thread 的等待不会杀死写线程；保留所有者才能等到真实结束。
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=max(0.0, deadline - time.monotonic())
                )
            except TimeoutError:
                self.summary["last_issue"] = "NATIVE_PUBLICATION_DID_NOT_DRAIN"
                break
        self.summary["drained"] = all(task.done() for task in tasks)
        if not tasks:
            self.summary["drained"] = True
        summary = dict(self.summary)
        summary['receive_alignment'] = dict(self._aligner.summary)
        return summary
