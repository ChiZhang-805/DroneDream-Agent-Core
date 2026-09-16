"""Bounded RGB preparation off the sensor/control thread; never renew source time."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .model_image_cache import ModelImageCache, PreparedModelImage, model_image_dimensions
from .rgb_input_quality import PreparedRgbMeasurement, prepare_rgb_measurement
from .sensor_frame_clock import SensorFrameTime, require_model_frame_time


@dataclass(frozen=True)
class PreparedCameraSample:
    """One callback-owned source message paired with pixels prepared from that exact source."""
    # 回调移交不再修改的独立 protobuf 副本，让注册表的质量、标定和模型像素对应同一帧。
    message: Any
    image: PreparedModelImage
    measurement: PreparedRgbMeasurement | None = None


class LatestModelImageWorker:
    """One active conversion, one replaceable pending frame, one completed frame.

    Dropping unconsumed camera frames is intentional; this is not an evidence
    writer. Consumers still apply the original frame's age/quality gates. An
    encoding error hides the older result until a subsequent frame succeeds.
    """

    # 功能：
    #   固定输出尺寸与编码函数，建立一项执行中、一项待处理及一项已完成的有界工作线程。
    # 输入：
    #   self：待初始化的异步图像工作器。
    #   size：编码输出宽高。
    #   decoder：从独立来源消息生成 PNG、RGB 字节对的函数。
    #   prepare_sensor_quality：是否同时测量原始像素的注册表质量，不使用缩小图像替代。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, size: tuple[int, int], decoder: Callable, prepare_sensor_quality=False):
        self._size = model_image_dimensions(size)
        if not callable(decoder):
            raise ValueError("MODEL_RGB_DECODER_INVALID")
        if type(prepare_sensor_quality) is not bool:
            raise ValueError("MODEL_RGB_QUALITY_OPTION_INVALID")
        self._prepare_sensor_quality = prepare_sensor_quality
        self._decoder = decoder
        self._condition = threading.Condition()
        self._pending = None
        self._latest: PreparedCameraSample | None = None
        self._last_source: tuple[float, int, SensorFrameTime | None] | None = None
        self._error: Exception | None = None
        self._closed = False
        self._stopped = False
        self._submitted = self._coalesced = self._prepared = self._failed = 0
        self._thread = threading.Thread(target=self._run, name="model-rgb", daemon=True)
        self._thread.start()

    # 功能：
    #   1. 接收回调已复制且不再修改的来源消息，替换待处理槽，不等待像素编码。
    #   2. 拒绝来源回退、改时钟和终止线程的新任务；重复来源不增加提交统计。
    # 输入：
    #   self：维护有界队列的工作器。
    #   message：回调移交的独立来源消息。
    #   received_monotonic_seconds：实际来源单调钟秒数。
    #   received_at_unix_ms：实际来源 UNIX 毫秒。
    #   frame_time：入口保存的完整来源时钟，普通直连调用可为 None。
    # 输出：
    #   accepted：新来源成功提交时为 True，重复来源或已主动关闭时为 False。
    def submit(
        self, message, *, received_monotonic_seconds: float, received_at_unix_ms: int,
        frame_time: SensorFrameTime | None = None
    ) -> bool:
        mono = received_monotonic_seconds
        wall = received_at_unix_ms
        require_model_frame_time(message, frame_time, mono, wall)
        with self._condition:
            if self._closed:
                accepted = False
                return accepted
            if self._stopped:
                raise ValueError("MODEL_RGB_WORKER_STOPPED") from self._error
            if self._last_source is not None:
                if mono < self._last_source[0]:
                    raise ValueError("MODEL_RGB_SOURCE_TIME_REGRESSED")
                if mono == self._last_source[0]:
                    if wall != self._last_source[1] or frame_time != self._last_source[2]:
                        raise ValueError("MODEL_RGB_SOURCE_REDATED")
                    accepted = False
                    return accepted
            self._last_source = mono, wall, frame_time
            self._coalesced += int(self._pending is not None)
            self._submitted += 1
            self._pending = message, mono, wall, frame_time
            self._condition.notify()
        accepted = True
        return accepted

    # 功能：
    #   返回工作器仍开放且原始年龄有效的最近结果；编码失败或线程终止不能暴露旧图像。
    # 输入：
    #   self：当前图像工作器。
    #   now_monotonic_seconds：消费者的当前单调钟秒数。
    #   maximum_age_seconds：消费者允许的最大原始图像年龄。
    # 输出：
    #   sample：当前年龄合法的已完成来源与编码图像；已关闭、未完成或过期时为 None。
    def latest(
        self, *, now_monotonic_seconds: float, maximum_age_seconds: float
    ) -> PreparedCameraSample | None:
        if (type(now_monotonic_seconds) not in (float, int)
                or not 0 <= now_monotonic_seconds <= sys.float_info.max
                or type(maximum_age_seconds) not in (float, int)
                or not 0 < maximum_age_seconds <= sys.float_info.max):
            raise ValueError("MODEL_RGB_CONSUMER_TIME_INVALID")
        with self._condition:
            if self._closed:
                sample = None
                return sample
            if self._stopped:
                raise ValueError("MODEL_RGB_WORKER_STOPPED") from self._error
            if self._error is not None:
                raise ValueError("MODEL_RGB_ASYNC_PREPARATION_FAILED") from self._error
            sample = self._latest
        if sample is None:
            return sample
        age = now_monotonic_seconds - sample.image.received_monotonic_seconds
        if not 0 <= age <= maximum_age_seconds:
            sample = None
        return sample

    # 功能：
    #   1. 在锁外执行编码，成功后发布同一来源与图像，不更新原始采集时间。
    #   2. 普通异常允许下一帧恢复；线程终止类异常停止接收新任务，并使消费者明确失败。
    #   3. 持续诊断只保存错误类别，不保留原回溯及像素局部变量；关闭后的迟到结果直接撤回。
    # 输入：
    #   self：当前工作线程拥有的编码状态。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        cache = ModelImageCache()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                message, mono, wall, frame_time = self._pending
                self._pending = None
            try:
                measurement = (prepare_rgb_measurement(message)
                               if self._prepare_sensor_quality else None)
                image, _ = cache.prepare(message, received_monotonic_seconds=mono,
                    received_at_unix_ms=wall, size=self._size, decoder=self._decoder,
                    frame_time=frame_time)
            except BaseException as error:
                with self._condition:
                    if self._closed:
                        return
                    # 错误消息可能包含整张图像或凭据；仅保留有界类别，不跨循环持有原回溯。
                    self._error = RuntimeError(type(error).__name__[:80])
                    self._latest = None
                    self._failed += 1
                    if not isinstance(error, Exception):
                        self._stopped = True
                        self._pending = None
                        self._condition.notify_all()
                        return
            else:
                with self._condition:
                    if self._closed:
                        return
                    self._latest = PreparedCameraSample(message, image, measurement)
                    self._prepared += 1
                    self._error = None

    # 功能：
    #   立即撤回待处理及已完成图像，再按有界超时等待线程退出；超时不冒充正常关闭。
    # 输入：
    #   self：需要关闭的工作器，可重复关闭。
    #   timeout_seconds：等待当前编码退出的最大秒数。
    # 输出：
    #   summary：提交、合并、已完成与失败次数的独立统计字典。
    def close(self, *, timeout_seconds: float = 2.) -> dict:
        if (type(timeout_seconds) not in (float, int)
                or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("MODEL_RGB_WORKER_CLOSE_TIMEOUT_INVALID")
        with self._condition:
            self._closed = True
            self._pending = None
            self._latest = None
            self._error = None
            self._condition.notify_all()
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("MODEL_RGB_WORKER_DID_NOT_STOP")
        with self._condition:
            summary = {"submitted": self._submitted, "coalesced": self._coalesced,
                       "prepared": self._prepared, "failed": self._failed}
            return summary
