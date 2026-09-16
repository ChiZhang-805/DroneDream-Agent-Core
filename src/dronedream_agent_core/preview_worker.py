"""A lossy, one-pending-frame UI preview; never a model or evidence stream."""

import threading
import time


class LatestPreviewWorker:
    """Decouple UI publishing from sensing with one replaceable pending frame."""

    # 功能：
    #   校验同步发布回调并启动单一预览线程；此通道允许丢帧，不承载模型或验收证据。
    # 输入：
    #   publish：必须最终返回的同步单帧发布函数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, publish):
        if not callable(publish):
            raise TypeError("PREVIEW_PUBLISHER_NOT_CALLABLE")
        self._publish = publish
        self._condition = threading.Condition()
        self._pending = None
        self._stopping = False
        self._received = self._replaced = self._published = self._failed = 0
        self._discarded = 0
        self._maximum_publish_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="simulation-ui-preview", daemon=True)
        self._thread.start()

    # 功能：
    #   不等待发布地替换待处理帧；关闭后拒绝接收，空帧不计入接收统计。
    # 输入：
    #   frame：非空的预览帧，按引用保留，生产者提交后不得再修改该对象。
    # 输出：
    #   accepted：本次是否进入预览队列，不代表已经发布。
    def submit(self, frame):
        with self._condition:
            if self._stopping:
                accepted = False
                return accepted
            if frame is None:
                raise ValueError("PREVIEW_FRAME_MISSING")
            self._received += 1
            self._replaced += self._pending is not None
            self._pending = frame
            self._condition.notify()
        accepted = True
        return accepted

    # 功能：
    #   取出最新帧后在队列锁外发布，统计单帧失败及处理时长，关闭时退出。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                if self._stopping:
                    return
                frame, self._pending = self._pending, None
            started = time.monotonic()
            try:
                self._publish(frame)
                self._published += 1
            except Exception:
                # 预览失败只记入此通道，不能污染原生观测或阻断位姿接收。
                self._failed += 1
            finally:
                self._maximum_publish_ms = max(
                    self._maximum_publish_ms, (time.monotonic() - started) * 1000
                )

    # 功能：
    #   停止接收、统计并丢弃待发布帧，等待在途发布结束；不能强制杀死阻塞回调。
    #   等待超时明确报错，调用方可在回调退出后再次关闭，不谎报线程已经终止。
    # 输入：
    #   无。
    # 输出：
    #   summary：线程退出后的接收、替换、关闭丢弃、成功、失败与最大耗时统计。
    def close(self):
        with self._condition:
            self._stopping = True
            self._discarded += self._pending is not None
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("SIMULATION_PREVIEW_WORKER_DID_NOT_STOP")
        summary = {
            "received": self._received,
            "replaced_pending": self._replaced,
            "discarded_pending": self._discarded,
            "published": self._published,
            "failed": self._failed,
            "maximum_publish_ms": self._maximum_publish_ms,
            "purpose": "ui-preview-only",
        }
        return summary
