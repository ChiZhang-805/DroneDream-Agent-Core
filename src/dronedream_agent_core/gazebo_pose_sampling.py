"""Fresh same-message pose sampling with operation-scoped subscription ownership."""

import asyncio
import math

from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions


class FreshGazeboPoseStream:
    # 功能：
    #   建立一次原生订阅；每次请求只接收请求之后到达的完整消息，不缓存旧位姿续期。
    # 输入：
    #   node、message_type、topic：原生传输连接。
    #   decoder：将同一条消息转换为完整位姿集合，缺少实体时返回 None。
    # 输出：
    #   None：初始化采样流。
    def __init__(self, node, message_type, topic, decoder):
        self._loop = asyncio.get_running_loop()
        self._decoder = decoder
        self._waiter = None
        self._closed = False
        self._subscriptions = GazeboSubscriptions(node)
        self._subscriptions.subscribe(message_type, topic, self._receive)

    # 功能：
    #   在原生回调线程捕获当前等待请求，阻止排队中的旧消息满足稍后才创建的请求。
    # 输入：
    #   message：原生同帧消息。
    # 输出：
    #   None：把解析结果投递给对应请求的事件循环。
    def _receive(self, message):
        waiter = self._waiter
        if waiter is None or self._closed or self._loop.is_closed():
            return
        try:
            value = self._decoder(message)
        except Exception as error:
            value = error
        if value is not None:
            try:
                self._loop.call_soon_threadsafe(self._resolve, waiter, value)
            except RuntimeError:
                if not self._loop.is_closed():
                    raise

    # 功能：
    #   只完成原消息绑定的有效等待者；取消、超时或已消费的请求不能被再次完成。
    # 输入：
    #   waiter：消息到达时的请求；value：完整位姿或解析错误。
    # 输出：
    #   None：完成或忽略该请求。
    def _resolve(self, waiter, value):
        if self._closed or waiter is not self._waiter or waiter.done():
            return
        if isinstance(value, Exception):
            waiter.set_exception(value)
        else:
            waiter.set_result(value)

    # 功能：
    #   在有限时间内等待下一条完整位姿，禁止并发采样竞争和关闭后的采样。
    # 输入：
    #   timeout_seconds：有限正数等待上限。
    # 输出：
    #   poses：新到达消息的位姿集合。
    async def sample(self, timeout_seconds):
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("Gazebo pose sample timeout is invalid")
        if self._closed or self._waiter is not None:
            raise RuntimeError("Gazebo pose sample lifecycle is invalid")
        waiter = self._loop.create_future()
        self._waiter = waiter
        try:
            poses = await asyncio.wait_for(waiter, timeout=timeout_seconds)
            return poses
        finally:
            self._waiter = None
            waiter.cancel()

    # 功能：
    #   在所属事件循环停止采样、取消等待并退订排空原生回调。
    # 输入：
    #   无。
    # 输出：
    #   summary：原生订阅关闭证据。
    def close(self):
        self._closed = True
        if self._waiter is not None:
            self._waiter.cancel()
        summary = self._subscriptions.close()
        return summary
