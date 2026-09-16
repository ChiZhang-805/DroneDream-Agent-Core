"""One running inference plus one replaceable input, never an unbounded queue.

Keys identify immutable computation inputs, not freshness or control authority.
Consumers must still check original source timestamps and action deadlines.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress


class LatestInferenceWorker:
    """Run immutable keyed work off-thread, coalescing superseded pending input."""

    # 功能：
    #   初始化一个运行槽和一个可替换等待槽，延迟创建线程，不复制调用方已冻结的输入。
    # 输入：
    #   encode：本地推理函数。
    #   name：有界工作线程名称。
    #   initializer：可选的线程专属初始化函数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, encode: Callable, *, name: str, initializer: Callable | None = None):
        if (not callable(encode) or initializer is not None and not callable(initializer)
                or type(name) is not str or not 1 <= len(name) <= 120 or not name.strip()):
            raise ValueError("LATEST_INFERENCE_CONFIGURATION_INVALID")
        self._encode, self._name, self._initializer = encode, name, initializer
        self._condition = threading.Condition()
        self._thread = None
        self._pending = self._active = self._latest = None
        self._closed, self._failure = False, None
        self._submitted = self._coalesced = self._completed = self._failed = 0

    # 功能：
    #   在持有条件锁时拒绝已经关闭或终止故障的工作器。
    # 输入：
    #   self：当前工作器。
    # 输出：
    #   None：不返回业务数据。
    def _require_open(self):
        if self._closed:
            raise RuntimeError("LATEST_INFERENCE_WORKER_CLOSED")
        if self._failure is not None:
            raise RuntimeError("LATEST_INFERENCE_WORKER_FAILED") from self._failure

    # 功能：
    #   返回最近同一计算身份的 Future；命中缓存不证明传感器仍然新鲜或动作已获授权。
    # 输入：
    #   self：当前工作器。
    #   key：调用方绑定像素、预处理与编码器身份的计算键。
    # 输出：
    #   future：未取消的同键 Future，未命中时为 None。
    def cached(self, key: str) -> Future | None:
        if type(key) is not str or not 1 <= len(key) <= 256:
            raise ValueError("LATEST_INFERENCE_KEY_INVALID")
        with self._condition:
            self._require_open()
            future = (self._latest[1] if self._latest is not None and self._latest[0] == key
                      and not self._latest[1].cancelled() else None)
            return future

    # 功能：
    #   接收冻结输入，合并同键请求并取消被更新输入替代的等待项；运行中的调用不能抢占。
    # 输入：
    #   self：当前工作器。
    #   key：本次输入的不可变计算身份。
    #   payload：调用方已经冻结的推理输入。
    # 输出：
    #   future：本次或已存在同键计算的 Future。
    def submit(self, key: str, payload) -> Future:
        if type(key) is not str or not 1 <= len(key) <= 256:
            raise ValueError("LATEST_INFERENCE_KEY_INVALID")
        displaced = None
        with self._condition:
            self._require_open()
            if (self._latest is not None and self._latest[0] == key
                    and not self._latest[1].cancelled()):
                return self._latest[1]
            if self._pending is not None:
                displaced = self._pending[2]
                self._pending = None
                self._coalesced += 1
            if self._active is not None and self._active[0] == key:
                future = self._active[1]
            else:
                future = Future()
                self._pending = key, payload, future
                self._submitted += 1
            self._latest = key, future
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                try:
                    self._thread.start()
                except BaseException as error:
                    # 常驻故障状态只存类别，不因启动失败保留输入与线程构造回溯。
                    self._failure = RuntimeError(type(error).__name__)
                    self._failed += 1
                    self._pending = self._latest = None
                    self._thread = None
                    raise
            self._condition.notify_all()
        # Future callbacks may call back into this worker; never invoke them
        # under the condition lock. Detached pending work cannot be started.
        if displaced is not None:
            displaced.cancel()
        return future

    # 功能：
    #   1. 在线程内初始化并串行推理，把普通失败交给 Future 后允许下一来源继续工作。
    #   2. 终止类异常或完成回调崩溃时关闭接收并结束等待项，避免无工作线程的永久挂起。
    # 输入：
    #   self：当前工作器及至多一个等待输入。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        try:
            if self._initializer is not None:
                self._initializer()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        return
                    key, payload, future = self._pending
                    self._pending = None
                    self._active = key, future
                result = None
                try:
                    if future.set_running_or_notify_cancel():
                        try:
                            result = self._encode(payload)
                        except Exception as error:
                            with self._condition:
                                self._failed += 1
                            future.set_exception(error)
                        except BaseException as error:
                            future.set_exception(error)
                            raise
                        else:
                            with self._condition:
                                self._completed += 1
                            future.set_result(result)
                finally:
                    # Future 完成回调运行在本线程；回调抛出终止异常也必须释放活动槽。
                    with self._condition:
                        self._active = None
                    del payload, future, result
        except BaseException as error:
            with self._condition:
                self._failure = RuntimeError(type(error).__name__)
                pending, self._pending = self._pending, None
                self._failed += 1
            if pending is not None and pending[2].set_running_or_notify_cancel():
                # 已设置失败状态；最后一个用户回调失败不能遗留尚未完成的内部任务。
                with suppress(BaseException):
                    pending[2].set_exception(error)

    # 功能：
    #   取消等待项并有限等待活动推理，明确报告超时，不声称能强行终止原生推理。
    # 输入：
    #   self：由外部所有者关闭的工作器。
    #   timeout_seconds：大于零、最多两秒的线程汇合预算。
    # 输出：
    #   receipt：线程确已停止后的提交、合并、完成和错误计数。
    def close(self, *, timeout_seconds: float = 2.) -> dict:
        if (type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 2.
                or not math.isfinite(timeout_seconds)):
            raise ValueError("LATEST_INFERENCE_CLOSE_TIMEOUT_INVALID")
        if threading.current_thread() is self._thread:
            raise RuntimeError("LATEST_INFERENCE_CLOSE_REQUIRES_EXTERNAL_OWNER")
        with self._condition:
            self._closed = True
            pending, self._pending = self._pending, None
            self._latest = None
            self._condition.notify_all()
        if pending is not None:
            pending[2].cancel()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
            if self._thread.is_alive():
                raise RuntimeError("LATEST_INFERENCE_WORKER_DID_NOT_STOP")
        with self._condition:
            self._encode = self._initializer = None
            receipt = {"submitted": self._submitted, "coalesced": self._coalesced,
                    "completed": self._completed, "failed": self._failed,
                    "thread_stopped": True, "maximum_pending_inputs": 1}
            return receipt
