"""Bounded native observation stream independent of rendering and model calls.

The sampler requests no actuators and fabricates no samples. Its 50 Hz poll
target is not a claim that the producer delivers 50 independent observations.
Frames are aligned by original receive times, never by fitting simulator truth
or extrapolating physical velocities through a sub-real-time host interval.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from .native_flight_state import native_flight_state_sample
from .native_pose import NativeMapPose, native_map_pose
from .native_state_channel import NativeStateReceiver
from .plugin_files import read_plugin_file
from .realtime_feature_encoders import (
    FlightStateEncoder,
    FlightStateSample,
    RealtimeFeatureEncoding,
)
from .sensor_diagnostics import sensor_issue_code

_MAX_PACKET_BYTES = 262_144


# 功能：
#   在缓冲状态判断前验证消费时钟，不允许布尔、小数或越界整数参与对齐。
# 输入：
#   value：候选 UNIX 毫秒时刻。
# 输出：
#   None：不返回业务数据。
def _validate_consumer_time(value: int) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("IMAGE_NATIVE_ALIGNMENT_TIME_INVALID")


# 功能：
#   验证对齐计算的正数参数，避免隐式类型转换及巨大整数溢出。
# 输入：
#   values：量程、加速度或允许方差等正数参数。
# 输出：
#   None：不返回业务数据。
def _validate_alignment_limits(values: tuple[object, ...]) -> None:
    if any(type(value) not in (int, float) or not 0 < value <= sys.float_info.max
           for value in values):
        raise ValueError("IMAGE_NATIVE_ALIGNMENT_LIMIT_INVALID")


@dataclass(frozen=True)
class NativeStateObservation:
    pose: NativeMapPose
    sample: FlightStateSample
    encoding: RealtimeFeatureEncoding
    payload: dict[str, object]
    available_at_unix_ms: int


@dataclass(frozen=True)
class AlignedNativePose:
    pose: NativeMapPose
    receive_skew_ms: int
    conservative_variance_m2: float


# 功能：
#   根据接收时差、平移和旋转幅度计算保守对齐方差，不外推或移动原始位姿。
# 输入：
#   item：已校验的原生状态快照。
#   image_time：图像原始接收时刻。
#   maximum_range_m：图像所覆盖的最大度量距离。
#   maximum_acceleration_mps2：用于时间错位预算的加速度上限。
# 输出：
#   measurement：最大接收时差毫秒数与保守方差平方米数组成的二元组。
def _alignment_measurement(
    item: NativeStateObservation, *, image_time: int, maximum_range_m: float,
    maximum_acceleration_mps2: float,
) -> tuple[int, float]:
    offset = max(abs(image_time - item.pose.position_received_at_unix_ms),
                 abs(image_time - item.pose.attitude_received_at_unix_ms))
    gyro = item.sample.angular_velocity_body_rad_s
    if gyro is None:
        raise ValueError("IMAGE_NATIVE_ANGULAR_RATE_UNAVAILABLE")
    dt = offset / 1000
    speed = math.hypot(*item.pose.velocity_world_enu_mps.model_dump().values())
    omega = math.hypot(*gyro.model_dump().values())
    displacement = speed * dt + .5 * maximum_acceleration_mps2 * dt * dt
    # 先计算有界弦长系数；不要先将极大量程乘二而溢出，尤其在零时间差时。
    displacement += maximum_range_m * (2 * math.sin(min(math.pi, omega * dt) / 2))
    deviation = math.sqrt(item.sample.localization_covariance_m2) + displacement
    variance = deviation * deviation
    if not math.isfinite(variance):
        raise ValueError("IMAGE_NATIVE_ALIGNMENT_UNCERTAINTY_INVALID")
    measurement = (offset, variance)
    return measurement


# 功能：
#   将对齐测量与独立位姿副本组合，调用方不能修改缓冲中的历史状态。
# 输入：
#   item：被选中的原生状态快照。
#   image_time：图像接收时刻。
#   maximum_range_m：最大量程米数。
#   maximum_acceleration_mps2：加速度上限。
# 输出：
#   aligned：包含位姿副本、时差及保守方差的对齐结果。
def _pose_alignment(item, *, image_time, maximum_range_m, maximum_acceleration_mps2):
    offset, variance = _alignment_measurement(item, image_time=image_time,
        maximum_range_m=maximum_range_m, maximum_acceleration_mps2=maximum_acceleration_mps2)
    aligned = AlignedNativePose(copy.deepcopy(item.pose), offset, variance)
    return aligned


class NativeStateBuffer:
    """One producer owns the encoder; readers get detached immutable history."""

    # 功能：
    #   创建单生产者状态缓冲与独立特征编码器，限制历史容量。
    # 输入：
    #   self：正在初始化的缓冲对象。
    #   capacity：保留的历史观测数，范围为 16 至 256。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, capacity: int = 64) -> None:
        if type(capacity) is not int or not 16 <= capacity <= 256:
            raise ValueError("NATIVE_HISTORY_CAPACITY_INVALID")
        self._history: deque[NativeStateObservation] = deque(maxlen=capacity)
        self._encoder = FlightStateEncoder()
        self._binding: str | None = None
        self._latest: NativeStateObservation | None = None
        self._issue: str | None = None
        self._lock = threading.Lock()

    # 功能：
    #   标记当前来源不可用，后续消费不得借用此前好历史绕过故障。
    # 输入：
    #   self：状态缓冲对象。
    #   issue：有界诊断类别，不应包含原始遥测或凭据。
    # 输出：
    #   None：不返回业务数据。
    def invalidate(self, issue: str) -> None:
        if type(issue) is not str or not issue or len(issue) > 256:
            raise ValueError("NATIVE_STATE_ISSUE_INVALID")
        with self._lock:
            self._issue = issue

    # 功能：
    #   1. 隔离并校验原生遥测，拒绝运行中坐标绑定变化和来源时间回退。
    #   2. 只有真实来源推进才增加历史，异步重发保留最旧来源期限。
    # 输入：
    #   self：由单一生产者驱动的缓冲对象。
    #   payload：标准 JSON 遥测对象。
    #   now_unix_ms：本次接入的 UNIX 毫秒时刻。
    # 输出：
    #   None：不返回业务数据。
    def ingest(self, payload: dict[str, object], *, now_unix_ms: int) -> None:
        payload = copy_json(payload, limit=_MAX_PACKET_BYTES)
        pose = native_map_pose(payload, now_unix_ms=now_unix_ms)
        sample = native_flight_state_sample(payload, now_unix_ms=now_unix_ms)
        if sample.localization_covariance_m2 is None:
            raise ValueError("NATIVE_LOCALIZATION_COVARIANCE_UNAVAILABLE")
        with self._lock:
            if self._binding is not None and pose.binding_sha256 != self._binding:
                raise ValueError("NATIVE_MAP_BINDING_CHANGED_DURING_EXECUTION")
            previous = self._latest
            if previous is not None and (
                pose.position_received_at_unix_ms < previous.pose.position_received_at_unix_ms
                or pose.attitude_received_at_unix_ms < previous.pose.attitude_received_at_unix_ms
            ):
                raise ValueError("NATIVE_POSE_CLOCK_REGRESSED")
            if previous is not None and (
                sample.observed_at_unix_ms < previous.sample.observed_at_unix_ms
            ):
                raise ValueError("NATIVE_FLIGHT_STATE_TIMESTAMP_REGRESSED")
            if previous is None or sample.observed_at_unix_ms > previous.sample.observed_at_unix_ms:
                encoding = self._encoder.encode(sample, encoded_at_unix_ms=now_unix_ms)
                if "FLIGHT_STATE_IMU_CLOCK_RESET" in encoding.issue_codes:
                    self._history.clear()
            else:
                # 部分来源可能已变化，但最旧来源尚未推进，不能把轮询次数当成新历史。
                encoding = previous.encoding
            observation = NativeStateObservation(pose, sample, encoding, payload, now_unix_ms)
            if previous is None or (
                pose.position_received_at_unix_ms != previous.pose.position_received_at_unix_ms
                or pose.attitude_received_at_unix_ms != previous.pose.attitude_received_at_unix_ms
            ):
                self._history.append(observation)
            self._latest, self._binding, self._issue = observation, pose.binding_sha256, None

    # 功能：
    #   选取消费时刻已存在且仍有效的状态，锁外复制返回，避免阻塞生产者。
    # 输入：
    #   self：状态缓冲对象。
    #   now_unix_ms：本次控制周期捕获的消费时刻。
    # 输出：
    #   detached：归调用方所有的独立状态快照。
    def latest(self, *, now_unix_ms: int) -> NativeStateObservation:
        _validate_consumer_time(now_unix_ms)
        with self._lock:
            if self._issue is not None:
                raise ValueError(f"NATIVE_STATE_STREAM_UNAVAILABLE:{self._issue}")
            if self._latest is None:
                raise ValueError("NATIVE_STATE_STREAM_WARMING")
            observation = self._latest
            if observation.available_at_unix_ms > now_unix_ms:
                # 采样器可能在消费时刻捕获后推进，只能选用当时已存在的观测。
                observation = next((item for item in reversed(self._history)
                                    if item.available_at_unix_ms <= now_unix_ms), None)
            if observation is None:
                raise ValueError("NATIVE_STATE_STREAM_NOT_YET_AVAILABLE")
            if not observation.encoding.fresh_at(now_unix_ms):
                raise ValueError("NATIVE_STATE_STREAM_EXPIRED")
        # 锁外复制不会阻塞生产者；历史内容发布后只读，调用者获得独立对象。
        detached = copy.deepcopy(observation)
        return detached

    # 功能：
    #   1. 在感知计算后读取已经实际到达的同源状态，不把循环入口旧状态继续用于控制。
    #   2. 没有新样本时保留旧来源期限；拒绝绑定改变和任一位置、姿态、状态时钟回退。
    # 输入：
    #   previous：本轮感知计算开始前使用的状态。
    #   now_unix_ms：控制参考冻结时刻，未来收到的样本不得进入本次参考。
    # 输出：
    #   current：独立的新状态，或仍有效且未续龄的原状态。
    def latest_after(self, previous, *, now_unix_ms: int) -> NativeStateObservation:
        if not isinstance(previous, NativeStateObservation):
            raise ValueError("NATIVE_CONTROL_PREVIOUS_STATE_INVALID")
        current = self.latest(now_unix_ms=now_unix_ms)
        if current.pose.binding_sha256 != previous.pose.binding_sha256:
            raise ValueError("NATIVE_CONTROL_REFRESH_BINDING_CHANGED")
        if (current.sample.observed_at_unix_ms < previous.sample.observed_at_unix_ms
                or current.pose.position_received_at_unix_ms
                < previous.pose.position_received_at_unix_ms
                or current.pose.attitude_received_at_unix_ms
                < previous.pose.attitude_received_at_unix_ms):
            raise ValueError("NATIVE_CONTROL_REFRESH_CLOCK_REGRESSED")
        return current

    # 功能：
    #   1. 选取仍有效且已有配套原生状态的最新图像，避免最新帧暂未配对导致饥饿。
    #   2. 可按完整的不确定性预算筛选，不等待、不预测、不放宽时间边界。
    # 输入：
    #   self：状态缓冲对象。
    #   received_times：最多 32 个图像原始接收时刻。
    #   now_unix_ms：本次消费时刻。
    #   maximum_range_m：可选最大量程米数。
    #   maximum_acceleration_mps2：可选加速度上限。
    #   maximum_alignment_variance_m2：可选允许方差，必须与前两个参数一起提供。
    # 输出：
    #   selected：可配对的最新图像时刻，无合格候选时为 None。
    def select_image_time(
        self, received_times: tuple[int, ...], *, now_unix_ms: int,
        maximum_range_m: float | None = None,
        maximum_acceleration_mps2: float | None = None,
        maximum_alignment_variance_m2: float | None = None,
    ) -> int | None:
        _validate_consumer_time(now_unix_ms)
        if (not isinstance(received_times, (tuple, list)) or len(received_times) > 32
                or any(type(value) is not int or not 0 <= value < 2**63
                       for value in received_times)):
            raise ValueError("IMAGE_NATIVE_CANDIDATE_TIMES_INVALID")
        limits = (maximum_range_m, maximum_acceleration_mps2, maximum_alignment_variance_m2)
        check_uncertainty = any(value is not None for value in limits)
        if check_uncertainty:
            _validate_alignment_limits(limits)
        selected = None
        with self._lock:
            if self._issue is not None or not self._history:
                return selected
            history = tuple(self._history)
        for image_time in sorted(set(received_times), reverse=True):
            if not 0 <= now_unix_ms - image_time <= 250:
                continue
            candidates = [item for item in history if (
                item.available_at_unix_ms <= now_unix_ms
                and item.encoding.fresh_at(now_unix_ms)
                and abs(image_time - item.pose.position_received_at_unix_ms) <= 50
                and abs(image_time - item.pose.attitude_received_at_unix_ms) <= 50
            )]
            # 对齐还依赖 IMU 和协方差，位置较新不能覆盖其他来源已经过期的事实。
            # 这里只判断是否有可用配对，避免为每个候选复制整份状态。
            if candidates and (not check_uncertainty or any(
                item.sample.angular_velocity_body_rad_s is not None
                and _alignment_measurement(
                    item, image_time=image_time, maximum_range_m=maximum_range_m,
                    maximum_acceleration_mps2=maximum_acceleration_mps2,
                )[1] <= maximum_alignment_variance_m2 for item in candidates
            )):
                selected = image_time
                return selected
        return selected

    # 功能：
    #   在已接收且完整来源仍有效的候选中，选取不确定性最小的图像配对位姿。
    # 输入：
    #   self：状态缓冲对象。
    #   image_received_at_unix_ms：图像原始接收时刻。
    #   now_unix_ms：当前消费时刻。
    #   maximum_range_m：最大量程米数。
    #   maximum_acceleration_mps2：加速度上限。
    #   maximum_skew_ms：允许图像与位置、姿态的最大时差，不超过 50 毫秒。
    # 输出：
    #   aligned：独立位姿及其时差、不确定性。
    def align_image(
        self, *, image_received_at_unix_ms: int, now_unix_ms: int,
        maximum_range_m: float, maximum_acceleration_mps2: float,
        maximum_skew_ms: int = 50,
    ) -> AlignedNativePose:
        _validate_consumer_time(now_unix_ms)
        _validate_consumer_time(image_received_at_unix_ms)
        if (not 0 <= now_unix_ms - image_received_at_unix_ms <= 250
                or type(maximum_skew_ms) is not int or not 0 < maximum_skew_ms <= 50):
            raise ValueError("IMAGE_NATIVE_ALIGNMENT_TIME_INVALID")
        _validate_alignment_limits((maximum_range_m, maximum_acceleration_mps2))
        with self._lock:
            if self._issue is not None or not self._history:
                raise ValueError("IMAGE_NATIVE_STATE_UNAVAILABLE")
            history = tuple(self._history)
        candidates = [item for item in history if (
                item.available_at_unix_ms <= now_unix_ms
                and item.encoding.fresh_at(now_unix_ms)
                and abs(image_received_at_unix_ms - item.pose.position_received_at_unix_ms)
                <= maximum_skew_ms
                and abs(image_received_at_unix_ms - item.pose.attitude_received_at_unix_ms)
                <= maximum_skew_ms
        )]
        if not candidates:
            raise ValueError("IMAGE_NATIVE_POSE_NOT_SYNCHRONIZED")
        candidates = [item for item in candidates
                      if item.sample.angular_velocity_body_rad_s is not None]
        if not candidates:
            raise ValueError("IMAGE_NATIVE_ANGULAR_RATE_UNAVAILABLE")
        best = min(candidates, key=lambda item: _alignment_measurement(
            item, image_time=image_received_at_unix_ms, maximum_range_m=maximum_range_m,
            maximum_acceleration_mps2=maximum_acceleration_mps2)[1])
        aligned = _pose_alignment(best, image_time=image_received_at_unix_ms,
            maximum_range_m=maximum_range_m, maximum_acceleration_mps2=maximum_acceleration_mps2)
        return aligned


class NativeStateSampler:
    # 功能：
    #   创建独立原生状态采样器，轮询目标频率不代表真实传感器产出频率。
    # 输入：
    #   self：正在初始化的采样器。
    #   path：已有遥测文件路径。
    #   rate_hz：20 至 100 Hz 的轮询目标。
    #   channel_path：可选运行专属通道描述文件，提供时使用该通道。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path, *, rate_hz: float = 50.0,
                 channel_path: Path | None = None) -> None:
        if type(rate_hz) not in (int, float) or not 20 <= rate_hz <= 100:
            raise ValueError("NATIVE_SAMPLER_RATE_INVALID")
        self.path, self.rate_hz = path, rate_hz
        self.buffer = NativeStateBuffer()
        self._receiver = NativeStateReceiver(channel_path) if channel_path is not None else None
        self._stop = threading.Event()
        self._diagnostic_lock = threading.Lock()
        self._accepted = 0
        self._failures: dict[str, int] = {}
        self._thread = threading.Thread(target=self._run, name="native-state-observer", daemon=True)

    # 功能：
    #   启动独立采样线程；线程生命周期只允许启动一次。
    # 输入：
    #   self：已初始化的采样器。
    # 输出：
    #   None：不返回业务数据。
    def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("NATIVE_STATE_SAMPLER_CLOSED")
        self._thread.start()

    # 功能：
    #   有界读取已有原生数据；普通失败允许下轮恢复，循环终止后状态明确不可用。
    # 输入：
    #   self：包含停止事件、输入来源及状态缓冲的采样器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    if self._receiver is not None:
                        payload = self._receiver.read_latest()
                    else:
                        raw = read_plugin_file(self.path, limit=_MAX_PACKET_BYTES)
                        payload = decode_json(raw, limit=_MAX_PACKET_BYTES)
                        if type(payload) is not dict:
                            raise ValueError("NATIVE_PACKET_NOT_AN_OBJECT")
                    if payload is not None and not self._stop.is_set():
                        self.buffer.ingest(payload, now_unix_ms=int(time.time() * 1000))
                        with self._diagnostic_lock:
                            self._accepted += 1
                except (OSError, ValueError, TypeError) as error:
                    issue = sensor_issue_code(error)
                    self.buffer.invalidate(issue)
                    with self._diagnostic_lock:
                        self._failures[issue] = self._failures.get(issue, 0) + 1
                self._stop.wait(max(0, 1 / self.rate_hz - (time.monotonic() - started)))
        finally:
            # 非普通异常也不能让消费者继续看到“活着”的采样器；不吞掉终止异常。
            self.buffer.invalidate("SAMPLER_STOPPED")

    # 功能：
    #   复制接入和拒收分类计数，不把重复遥测接入次数当作独立物理样本数量。
    # 输入：
    #   无。
    # 输出：
    #   summary：当前采样线程与固定错误类别的统计。
    def summary(self) -> dict:
        with self._diagnostic_lock:
            summary = {"accepted_packets": self._accepted, "failures": dict(self._failures),
                       "thread_alive": self._thread.is_alive()}
        return summary

    # 功能：
    #   请求停止采样并等待线程结束，再关闭自有通道；超时必须如实报告。
    # 输入：
    #   self：准备关闭的采样器。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise RuntimeError("NATIVE_STATE_SAMPLER_DID_NOT_STOP")
        if self._receiver is not None:
            self._receiver.close()
        self.buffer.invalidate("SAMPLER_STOPPED")
