"""Complete, bounded transition persistence outside the live policy deadline."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .evidence_snapshot import detach_evidence


class TransitionWriter:
    """Persist every accepted training record or fail the collection explicitly."""

    # 功能：
    #   校验发布器与排队上限后启动单个写入线程，训练证据不采用丢帧队列。
    # 输入：
    #   self：待初始化的写入器。
    #   publish：接收绝对路径和独立记录、写入失败时抛出异常的回调。
    #   capacity：最多等待写入的记录数，不含正在写入的一条。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, publish: Callable[[Path, dict], None], *, capacity: int = 32):
        if type(capacity) is not int or not 1 <= capacity <= 256:
            raise ValueError("TRAINING_WRITER_CAPACITY_INVALID")
        if not callable(publish):
            raise ValueError("TRAINING_WRITER_PUBLISHER_INVALID")
        self._publish, self._capacity = publish, capacity
        self._condition = threading.Condition()
        self._pending: deque[tuple[Path, dict]] = deque()
        self._error: BaseException | None = None
        self._closed = False
        self._submitted = self._completed = 0
        self._thread = threading.Thread(target=self._run, name="training-transition-writer",
                                        daemon=True)
        self._thread.start()

    # 功能：
    #   把后台首次持久化错误传回采集线程，禁止把失败数据集作为成功结果。
    # 输入：
    #   self：持有锁与错误状态的写入器。
    # 输出：
    #   None：不返回业务数据。
    def check(self) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("TRAINING_TRANSITION_PERSISTENCE_FAILED") from self._error

    # 功能：
    #   固定目标路径并复制记录后入队；关闭、写入失败或队列已满时明确拒绝。
    # 输入：
    #   self：当前写入器。
    #   path：本条训练证据的文件路径。
    #   record：由采集器验证过的文本键记录对象。
    # 输出：
    #   None：不返回业务数据。
    def submit(self, path: Path, record: dict) -> None:
        if not isinstance(path, Path) or type(record) is not dict:
            raise ValueError("TRAINING_TRANSITION_RECORD_INVALID")
        # 只固定工作目录解释；是否允许该路径及无覆盖发布仍由发布器负责。
        destination = path.absolute()
        self.check()
        owned = detach_evidence(record)
        with self._condition:
            self.check()
            if self._closed:
                raise RuntimeError("TRAINING_TRANSITION_WRITER_CLOSED")
            if len(self._pending) >= self._capacity:
                self._error = RuntimeError("TRAINING_TRANSITION_QUEUE_FULL")
                self._condition.notify_all()
                self.check()
            self._pending.append((destination, owned))
            self._submitted += 1
            self._condition.notify()

    # 功能：
    #   依序在锁外发布记录，只有回调成功返回才增加完成数；保留首次失败原因。
    # 输入：
    #   self：具有队列、发布器与计数状态的写入器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closed)
                    if not self._pending:
                        return
                    path, record = self._pending.popleft()
                self._publish(path, record)
                with self._condition:
                    self._completed += 1
        except BaseException as error:
            with self._condition:
                if self._error is None:
                    self._error = error
                self._condition.notify_all()

    # 功能：
    #   1. 关闭提交入口并限时排空队列，核对全部已接纳记录的持久化计数。
    #   2. 超时只报告未完成，不强杀发布线程；可在写完后再次调用以取得最终结果。
    # 输入：
    #   self：需要关闭的写入器。
    #   timeout_seconds：正且不超过平台等待上限的最长等待秒数。
    # 输出：
    #   result：提交数、持久化数、完整性与仅仿真标记组成的回执。
    def close(self, *, timeout_seconds: float = 8.) -> dict:
        # 参数错误不得改变生命周期；None 会造成无限等待，巨大数可能溢出原生接口。
        if (type(timeout_seconds) not in (float, int)
                or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("TRAINING_WRITER_CLOSE_TIMEOUT_INVALID")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("TRAINING_TRANSITION_DRAIN_TIMEOUT")
        self.check()
        if self._completed != self._submitted or self._pending:
            raise RuntimeError("TRAINING_TRANSITION_DRAIN_INCOMPLETE")
        result = {"submitted": self._submitted, "persisted": self._completed,
                  "complete": True, "simulation_only": True}
        return result
