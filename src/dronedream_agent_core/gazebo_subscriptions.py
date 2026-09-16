"""Explicit ownership of native Gazebo callbacks through worker shutdown.

Stop delivering callbacks before downstream workers close; unsubscribe while
the Python interpreter is alive, not during pybind11/global destruction.
No sensor values, timestamps or control authority are manufactured here.
"""

import copy
import threading
import time


# 功能：
#   独立复核有效订阅数、退订数、排空数量及错误列表，不仅依赖关闭成功标记。
# 输入：
#   summary：订阅组保存的关闭回执。
# 输出：
#   complete：至少一个订阅被全部退订、回调排空且没有错误时为 True。
def subscription_shutdown_is_complete(summary) -> bool:
    if not isinstance(summary, dict):
        complete = False
        return complete
    count = summary.get("subscribed_count")
    complete = (summary.get("complete") is True and type(count) is int and 1 <= count <= 8
        and type(summary.get("unsubscribed_count")) is int
        and summary["unsubscribed_count"] == count
        and type(summary.get("active_callbacks_at_close")) is int
        and summary["active_callbacks_at_close"] == 0 and summary.get("errors") == [])
    return complete


class GazeboSubscriptions:
    """Own subscriptions and drain callbacks before their consumers shut down."""

    # 功能：
    #   验证原生订阅及退订接口，为调用方串行管理生命周期建立回调计数和关闭状态。
    # 输入：
    #   self：待初始化的订阅组。
    #   node：提供 subscribe 和 unsubscribe 的 Gazebo 节点。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, node):
        if not callable(getattr(node, "subscribe", None)) or not callable(
            getattr(node, "unsubscribe", None)
        ):
            raise ValueError("GAZEBO_SUBSCRIPTION_NODE_INVALID")
        self._node = node
        self._condition = threading.Condition()
        self._topics = []
        self._stopping = False
        self._active = 0
        self._summary = None

    # 功能：
    #   1. 校验主题、处理器和容量，仅在原生订阅确认成功后开放回调。
    #   2. 统计并发执行的回调，处理器抛出异常时仍归还计数，供关闭阶段等待排空。
    # 输入：
    #   self：订阅及关闭操作由调用方串行执行的订阅组。
    #   message_type：原生消息类型。
    #   topic：长度不超过 1024 字符、无 NUL 的非空主题。
    #   callback：消费消息的可调用处理器。
    # 输出：
    #   None：不返回业务数据。
    def subscribe(self, message_type, topic: str, callback) -> None:
        if (type(topic) is not str or not 1 <= len(topic) <= 1024 or not topic.strip()
                or "\0" in topic or not callable(callback)
                or self._stopping or topic in self._topics or len(self._topics) >= 8):
            raise ValueError("GAZEBO_SUBSCRIPTION_LIFECYCLE_INVALID")
        registered = False

        # 功能：
        #   在注册成功且尚未关闭时交付消息，无论处理成功或异常都减少执行中计数。
        # 输入：
        #   message：Gazebo 传递的原始消息。
        # 输出：
        #   None：不返回业务数据。
        def deliver(message):
            with self._condition:
                if self._stopping or not registered:
                    return
                self._active += 1
            try:
                callback(message)
            finally:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

        if self._node.subscribe(message_type, topic, deliver) is not True:
            raise RuntimeError("GAZEBO_SUBSCRIPTION_FAILED:" + topic)
        # 原生注册可能在返回前触发回调；确认前丢弃这些消息，失败注册始终没有交付资格。
        with self._condition:
            self._topics.append(topic)
            registered = True

    # 功能：
    #   1. 先禁止新回调，逆序退订，再有界等待已开始的回调排空并保存独立关闭回执。
    #   2. 某个退订失败仍继续其余清理；重复关闭保留首次失败结果，不把后来退出改算成功。
    #   3. 超时只限制回调排空等待，不能中断阻塞的原生 unsubscribe。
    # 输入：
    #   self：与订阅操作串行关闭的订阅组，不应由其正在执行的回调调用。
    #   timeout_seconds：原生退订返回后等待回调排空的最大秒数。
    # 输出：
    #   summary：独立的订阅数、退订数、残留回调数、错误列表及完成状态。
    def close(self, timeout_seconds: float = 2.) -> dict:
        if (type(timeout_seconds) not in (int, float)
                or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("GAZEBO_SUBSCRIPTION_DRAIN_TIMEOUT_INVALID")
        if self._summary is not None:
            summary = copy.deepcopy(self._summary)
            return summary
        with self._condition:
            self._stopping = True
        errors = []
        unsubscribed = 0
        for topic in reversed(self._topics):
            try:
                # 退订内部可能等待正在执行的回调，因此调用原生接口时不能持有回调锁。
                if self._node.unsubscribe(topic) is not True:
                    errors.append("unsubscribe-rejected:" + topic)
                else:
                    unsubscribed += 1
            except Exception as error:
                errors.append("unsubscribe-failed:" + topic + ":" + type(error).__name__[:80])
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors.append("callback-drain-timeout")
                    break
                self._condition.wait(remaining)
            active = self._active
        self._summary = {"complete": not errors and active == 0,
            "subscribed_count": len(self._topics), "unsubscribed_count": unsubscribed,
            "active_callbacks_at_close": active, "errors": errors}
        summary = copy.deepcopy(self._summary)
        return summary
