"""Bounded source-stamped buffers for the isolated camera calibration fixture.

Not a native state channel, observation authority, or flight training dataset.
Sampling discards are counted; a full queue is an error, never silent success.
"""

from __future__ import annotations

import math
import threading
from collections import deque

from dronedream_plugin_sdk.protocol import copy_json

from .simulation_sensor_frames import _rotation


class FixtureCapture:
    # 功能：
    #   建立分辨率和队列均有上限的校准采集器，采样时钟采用仿真源时间。
    # 输入：
    #   self：当前采集器。
    #   width：深度图像宽度，单位像素。
    #   height：深度图像高度，单位像素。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, width: int, height: int):
        if (type(width) is not int or type(height) is not int
                or min(width, height) < 2 or width * height > 640 * 480):
            raise ValueError("FIXTURE_CAPTURE_DIMENSIONS_INVALID")
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.active = True
        self.poses = []
        self.images = deque()
        self.errors = []
        self.sampled_images = 0
        self.skipped_images = 0
        self.last_image_ns = -1
        self.last_saved_image_ns = -1
        self.last_pose_ns = -1

    # 功能：
    #   在调用者持锁期间保存有界错误摘要，不让异常消息撑大采集内存。
    # 输入：
    #   self：当前采集器。
    #   code：错误文本。
    # 输出：
    #   None：不返回业务数据。
    def _error(self, code):
        if len(self.errors) < 16:
            self.errors.append(code[:512] if type(code) is str else "FIXTURE_CAPTURE_ERROR_INVALID")

    # 功能：
    #   校验来源时间及接收时间的类型和范围，不将墙钟当作仿真采样时钟。
    # 输入：
    #   record：已隔离的采集元数据对象。
    # 输出：
    #   stamp：仿真来源时间，单位纳秒。
    @staticmethod
    def _stamp(record):
        if not isinstance(record, dict):
            raise ValueError("FIXTURE_CAPTURE_RECORD_INVALID")
        stamp = record.get("simulation_time_ns")
        if type(stamp) is not int or not 0 <= stamp <= (1 << 63) - 1:
            raise ValueError("FIXTURE_CAPTURE_SOURCE_TIME_INVALID")
        for name in ("received_monotonic_seconds", "received_unix_ms"):
            value = record.get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("FIXTURE_CAPTURE_RECEIPT_TIME_INVALID")
        return stamp

    # 功能：
    #   接收回调外部错误，关闭之后不再改动采集结论。
    # 输入：
    #   self：当前采集器。
    #   code：订阅或消息转换错误文本。
    # 输出：
    #   None：不返回业务数据。
    def fail(self, code):
        with self.lock:
            if self.active:
                self._error(code)

    # 功能：
    #   1. 隔离并验证姿态记录，保留递增来源时间，忽略相同时间的重复姿态。
    #   2. 非法消息和超量记录转成诊断，不从订阅回调漏出输入校验异常。
    # 输入：
    #   self：当前采集器。
    #   record：包含时间、米制位置及 wxyz 姿态的原始记录。
    # 输出：
    #   None：不返回业务数据。
    def receive_pose(self, record):
        with self.lock:
            if not self.active:
                return
            try:
                # 先冻结嵌套列表及报头，再验证同一份快照，避免来源后续复用对象改写证据。
                record = copy_json(record, limit=16 * 1024)
                stamp = self._stamp(record)
                _rotation(record["orientation_wxyz"])
                p = record["position_m"]
                if len(p) != 3 or any(type(x) not in (int, float)
                        or not math.isfinite(x) or abs(x) > 1e5 for x in p):
                    raise ValueError("FIXTURE_CAPTURE_POSITION_INVALID")
                if stamp < self.last_pose_ns:
                    raise ValueError("FIXTURE_CAPTURE_POSE_TIME_REGRESSED")
                if stamp == self.last_pose_ns:
                    return
                if len(self.poses) >= 25000:
                    raise ValueError("FIXTURE_CAPTURE_POSE_LIMIT")
                self.last_pose_ns = stamp
                self.poses.append(record)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self._error(f"{type(exc).__name__}:{exc}")

    # 功能：
    #   1. 校验不可变深度字节及独立元数据，按源时间最多每 200 毫秒保存一帧。
    #   2. 分开记录采样跳帧与队列溢出；队列满不能被视为正常采样成功。
    # 输入：
    #   self：当前采集器。
    #   record：图像布局、来源时间和接收时间。
    #   data：逐行紧密排列的 R_FLOAT32 深度字节。
    # 输出：
    #   None：不返回业务数据。
    def receive_image(self, record, data: bytes):
        with self.lock:
            if not self.active:
                return
            try:
                record = copy_json(record, limit=16 * 1024)
                stamp = self._stamp(record)
                if (type(record.get("width")) is not int or type(record.get("height")) is not int
                        or record["width"] != self.width or record["height"] != self.height
                        or record.get("pixel_format") != "R_FLOAT32"
                        or type(record.get("step")) is not int
                        or record["step"] != self.width * 4
                        or not isinstance(data, bytes)
                        or len(data) != record["step"] * self.height):
                    raise ValueError("FIXTURE_CAPTURE_IMAGE_LAYOUT_INVALID")
                if stamp <= self.last_image_ns:
                    raise ValueError("FIXTURE_CAPTURE_IMAGE_TIME_NOT_PROGRESSING")
                self.last_image_ns = stamp
                if self.last_saved_image_ns >= 0 and stamp - self.last_saved_image_ns < 200_000_000:
                    self.skipped_images += 1
                    return
                if self.sampled_images >= 256 or len(self.images) >= 4:
                    raise ValueError("FIXTURE_CAPTURE_IMAGE_LIMIT_OR_QUEUE_FULL")
                self.last_saved_image_ns = stamp
                self.sampled_images += 1
                self.images.append((record, data))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self._error(f"{type(exc).__name__}:{exc}")

    # 功能：
    #   原子取走待处理图像，并读取姿态进度及错误摘要；不会清除历史姿态与错误。
    # 输入：
    #   self：当前采集器。
    # 输出：
    #   result：图像记录列表、最新姿态源时间及错误元组。
    def snapshot(self):
        with self.lock:
            queued = list(self.images)
            self.images.clear()
            result = queued, self.last_pose_ns, tuple(self.errors)
            return result

    # 功能：
    #   停止接收新记录，保留已经采集的资料供落盘与审计。
    # 输入：
    #   self：当前采集器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        with self.lock:
            self.active = False
