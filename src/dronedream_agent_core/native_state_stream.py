"""Bounded native observation stream independent of rendering and model calls.

The sampler requests no actuators and fabricates no samples. Its 50 Hz poll
target is not a claim that the producer delivers 50 independent observations.
Frames use original receive times by default, or bounded source timestamps
when the caller explicitly establishes a shared clock domain. Neither mode
fits simulator truth or extrapolates velocities through a host-time interval.
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
from .native_image_pose import NativeImagePoseBuffer
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
    native_odometry_snapshot: dict | None = None


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
    dynamics = item.payload.get("dynamics", {})
    odometry = dynamics.get("sources", {}).get("odometry")
    snapshot = None
    if isinstance(odometry, dict):
        # Preserve the packet's own clock and frame. Matching a pose to an image
        # does NOT synchronize this separate odometry covariance stream.
        snapshot = copy_json({
            "dynamics_collected_at_unix_ms": dynamics.get("collected_at_unix_ms"),
            "observation_available_at_unix_ms": item.available_at_unix_ms,
            "image_synchronized": False,
            "odometry": {key: odometry[key] for key in (
                "timestamp_us", "received_at_unix_ms", "sample_age_seconds",
                "frame_id", "child_frame_id", "pose_covariance_upper_m2",
                "twist_covariance_upper", "velocity_child_frame_m_s",
                "reset_counter", "estimator_type", "quality", "mavlink_source",
            ) if key in odometry},
        }, limit=8192)
    aligned = AlignedNativePose(copy.deepcopy(item.pose), offset, variance, snapshot)
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
    def __init__(self, *, capacity: int = 64, source_clock_domain: str | None = None) -> None:
        if type(capacity) is not int or not 16 <= capacity <= 256:
            raise ValueError("NATIVE_HISTORY_CAPACITY_INVALID")
        self._history: deque[NativeStateObservation] = deque(maxlen=capacity)
        self._encoder = FlightStateEncoder()
        self._binding: str | None = None
        self._latest: NativeStateObservation | None = None
        self._issue: str | None = None
        self._lock = threading.Lock()
        self._slot_transitions: deque[dict] = deque(maxlen=32)
        self._maximum_history_length = 0
        self._odometry_epoch = None
        self._odometry_reset_count = 0
        self._image_epoch_started_at_unix_ms = 0
        self._pending_odometry_reset_receive_ms = 0
        self._source_image = (NativeImagePoseBuffer(clock_domain=source_clock_domain)
                              if source_clock_domain is not None else None)

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
    #   2. 只有真实来源推进才增加编码历史，局部姿态更新刷新瞬时特征但保留最旧期限。
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
        odometry = payload.get("dynamics", {}).get("sources", {}).get("odometry", {})
        odometry_epoch = None
        if "reset_counter" in odometry:
            counter = odometry["reset_counter"]
            if type(counter) is not int or not 0 <= counter <= 255:
                raise ValueError("NATIVE_ODOMETRY_RESET_COUNTER_INVALID")
            odometry_epoch = (counter, odometry.get("frame_id"), odometry.get("child_frame_id"),
                              odometry.get("estimator_type"))
        with self._lock:
            if self._binding is not None and pose.binding_sha256 != self._binding:
                raise ValueError("NATIVE_MAP_BINDING_CHANGED_DURING_EXECUTION")
            if self._source_image is not None:
                self._source_image.ingest(payload, available_at_monotonic=time.monotonic())
            previous = self._latest
            if self._odometry_epoch is not None and odometry_epoch is None:
                raise ValueError("NATIVE_ODOMETRY_RESET_EVIDENCE_LOST")
            odometry_reset = (self._odometry_epoch is not None
                              and odometry_epoch != self._odometry_epoch)
            if previous is not None and (
                pose.position_received_at_unix_ms < previous.pose.position_received_at_unix_ms
                or pose.attitude_received_at_unix_ms < previous.pose.attitude_received_at_unix_ms
            ):
                raise ValueError("NATIVE_POSE_CLOCK_REGRESSED")
            if previous is not None and (
                sample.observed_at_unix_ms < previous.sample.observed_at_unix_ms
            ):
                raise ValueError("NATIVE_FLIGHT_STATE_TIMESTAMP_REGRESSED")
            if odometry_reset:
                # An estimator reset can jump position, velocity or attitude
                # without resetting the IMU clock. Neither image matching nor
                # temporal control features may retain the preceding segment.
                self._history.clear()
                self._encoder = FlightStateEncoder()
                self._odometry_reset_count += 1
                self._image_epoch_started_at_unix_ms = now_unix_ms
                self._pending_odometry_reset_receive_ms = odometry.get(
                    "received_at_unix_ms", now_unix_ms)
                _validate_consumer_time(self._pending_odometry_reset_receive_ms)
                # Mark the new epoch even while the other telemetry streams
                # catch up. Otherwise each retry counts the same reset again,
                # or a pre-reset attitude can enter a freshly emptied encoder.
                self._odometry_epoch = odometry_epoch
                self._latest = None
                previous = None
            if sample.observed_at_unix_ms < self._pending_odometry_reset_receive_ms:
                self._issue = "NATIVE_ODOMETRY_RESET_WAITING_FOR_COHERENT_STATE"
                raise ValueError(self._issue)
            same_slot = previous is not None and not odometry_reset and (
                # 局部更新的原始包可比编码槽更新；IMU 随后推进时必须比较编码槽，
                # 否则会把独立新样本送入 refresh_current 而被正确的槽校验拒绝。
                sample.observed_at_unix_ms == previous.encoding.observed_at_unix_ms
                or (sample.imu_timestamp_us is not None
                    and sample.imu_timestamp_us == previous.sample.imu_timestamp_us))
            if not same_slot:
                encoding = self._encoder.encode(sample, encoded_at_unix_ms=now_unix_ms)
                if "FLIGHT_STATE_IMU_CLOCK_RESET" in encoding.issue_codes:
                    self._history.clear()
            elif sample.source_sample_sha256 != previous.sample.source_sample_sha256:
                # 复合包的最旧来源不变，不代表姿态也没变；更新瞬时特征但不增加历史或期限。
                encoding = self._encoder.refresh_current(sample, encoded_at_unix_ms=now_unix_ms)
            else:
                # 所有物理来源完全相同的重发才复用原编码。
                encoding = previous.encoding
            observation = NativeStateObservation(pose, sample, encoding, payload, now_unix_ms)
            if previous is None or odometry_reset or (
                pose.position_received_at_unix_ms != previous.pose.position_received_at_unix_ms
                or pose.attitude_received_at_unix_ms != previous.pose.attitude_received_at_unix_ms
            ):
                self._history.append(observation)
            self._latest, self._binding, self._issue = observation, pose.binding_sha256, None
            self._odometry_epoch = odometry_epoch
            self._maximum_history_length = max(
                self._maximum_history_length, len(self._encoder._samples))
            if (previous is None
                    or encoding.observed_at_unix_ms != previous.encoding.observed_at_unix_ms):
                self._slot_transitions.append({"source_unix_ms": sample.observed_at_unix_ms,
                    "imu_timestamp_us": sample.imu_timestamp_us,
                    "history_length": len(self._encoder._samples),
                    "issue_codes": list(encoding.issue_codes)})

    # 功能：
    #   返回有限的历史槽推进诊断，保留原始时钟与预热原因，不记录画面或授权飞行。
    # 输入：
    #   无。
    # 输出：
    #   summary：最大历史长度和最多三十二条独立槽的只读副本。
    def history_summary(self) -> dict:
        with self._lock:
            return {"maximum_history_length": self._maximum_history_length,
                    "odometry_reset_count": self._odometry_reset_count,
                    "recent_slot_transitions": copy.deepcopy(list(self._slot_transitions))}

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
    #   只读检查是否实际收到更新且未过期的原生状态，不复制张量、不更新来源或授权动作。
    # 输入：
    #   previous_source_unix_ms：上一控制周期已经使用的最旧状态来源时间。
    #   now_unix_ms：检查时刻；未来、故障或尚未编码完成的状态不能触发交接。
    # 输出：
    #   available：存在可供下游重新读取和校验的独立新样本时为 True。
    def newer_sample_available(self, previous_source_unix_ms: int, *, now_unix_ms: int) -> bool:
        _validate_consumer_time(now_unix_ms)
        _validate_consumer_time(previous_source_unix_ms)
        with self._lock:
            item = self._latest
            available = bool(self._issue is None and item is not None
                and item.available_at_unix_ms <= now_unix_ms
                and previous_source_unix_ms < item.sample.observed_at_unix_ms <= now_unix_ms
                and item.encoding.fresh_at(now_unix_ms))
        return available

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
            image_epoch_start = self._image_epoch_started_at_unix_ms
        for image_time in sorted(set(received_times), reverse=True):
            if (not 0 <= now_unix_ms - image_time <= 250
                    or image_time < image_epoch_start):
                continue
            # 先排除时间上绝不可能匹配的历史，再完整重验剩余编码；
            # 不为每幅图重复复制全部 64 项历史，也不略过任何候选的完整性和年龄校验。
            candidates = [item for item in history if (
                item.available_at_unix_ms <= now_unix_ms
                and abs(image_time - item.pose.position_received_at_unix_ms) <= 50
                and abs(image_time - item.pose.attitude_received_at_unix_ms) <= 50
                and item.encoding.fresh_at(now_unix_ms)
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
            if image_received_at_unix_ms < self._image_epoch_started_at_unix_ms:
                raise ValueError("IMAGE_NATIVE_ESTIMATOR_RESET_BOUNDARY")
            history = tuple(self._history)
        candidates = [item for item in history if (
                item.available_at_unix_ms <= now_unix_ms
                and abs(image_received_at_unix_ms - item.pose.position_received_at_unix_ms)
                <= maximum_skew_ms
                and abs(image_received_at_unix_ms - item.pose.attitude_received_at_unix_ms)
                <= maximum_skew_ms
                and item.encoding.fresh_at(now_unix_ms)
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

    # 功能：在既有状态锁下读取源时钟配对；显式配置该模式后绝不回退到主机接收时刻配对。
    # 输入：图像源时刻及同域身份、接收/消费时刻、几何运动预算。
    # 输出：完整单包位姿及可追溯时间证据；保留原生协方差，不赋予地图定位资格。
    def align_source_image(self, *, image_timestamp_ns, clock_domain,
                           image_received_at_unix_ms, now_unix_ms, now_monotonic,
                           maximum_range_m, maximum_acceleration_mps2):
        with self._lock:
            if self._issue is not None or self._source_image is None:
                raise ValueError("IMAGE_NATIVE_SOURCE_CLOCK_UNAVAILABLE")
            result = self._source_image.align(image_timestamp_ns=image_timestamp_ns,
                clock_domain=clock_domain, image_received_unix_ms=image_received_at_unix_ms,
                now_unix_ms=now_unix_ms, now_monotonic=now_monotonic,
                maximum_range_m=maximum_range_m,
                maximum_acceleration_mps2=maximum_acceleration_mps2)
        return AlignedNativePose(result.pose,
            abs(result.pose.observed_at_unix_ms - image_received_at_unix_ms),
            result.conservative_variance_m2,
            {"image_synchronized": False, "pairing_mode": "bounded-source-clock-nearest",
             "source_alignment": result.source_evidence})

    # 功能：只选择同域源时间可配对的最新图像，不用接收时刻相近替代曝光时刻相近。
    # 输入：candidates：最多 32 对图像接收毫秒、源纳秒；其余参数为时间和不确定性预算。
    # 输出：符合现有方差预算的图像接收时刻；没有有效候选时为 None。
    def select_source_image_time(self, candidates, *, clock_domain, now_unix_ms,
                                 now_monotonic, maximum_range_m, maximum_acceleration_mps2,
                                 maximum_alignment_variance_m2):
        selected = self.select_source_image_pose(candidates, clock_domain=clock_domain,
            now_unix_ms=now_unix_ms, now_monotonic=now_monotonic,
            maximum_range_m=maximum_range_m, maximum_acceleration_mps2=maximum_acceleration_mps2,
            maximum_alignment_variance_m2=maximum_alignment_variance_m2)
        return selected[0] if selected is not None else None

    # 功能：一次配对即返回该图像和独立位姿快照，避免选帧后重读跨越估计器重置。
    # 输入：有界图像候选及原有时钟、运动和不确定性限制。
    # 输出：(接收毫秒、曝光纳秒、完整对齐位姿)，暂缺为 None；身份或结构损坏不能吞掉。
    def select_source_image_pose(self, candidates, *, clock_domain, now_unix_ms,
                                 now_monotonic, maximum_range_m, maximum_acceleration_mps2,
                                 maximum_alignment_variance_m2):
        _validate_consumer_time(now_unix_ms)
        _validate_alignment_limits((maximum_range_m, maximum_acceleration_mps2,
                                    maximum_alignment_variance_m2))
        if (type(candidates) is not tuple or len(candidates) > 32
                or any(type(row) is not tuple or len(row) != 2 for row in candidates)):
            raise ValueError("IMAGE_NATIVE_CANDIDATE_TIMES_INVALID")
        for received, stamp in candidates:
            _validate_consumer_time(received)
            if stamp is not None:
                _validate_consumer_time(stamp)
        for received, stamp in sorted(candidates, key=lambda row: row[0], reverse=True):
            if stamp is None:
                continue
            try:
                aligned = self.align_source_image(image_timestamp_ns=stamp,
                    clock_domain=clock_domain, image_received_at_unix_ms=received,
                    now_unix_ms=now_unix_ms, now_monotonic=now_monotonic,
                    maximum_range_m=maximum_range_m,
                    maximum_acceleration_mps2=maximum_acceleration_mps2)
            except ValueError as error:
                if str(error) not in {
                    "IMAGE_NATIVE_SOURCE_CLOCK_UNAVAILABLE", "IMAGE_SOURCE_HISTORY_UNAVAILABLE",
                    "IMAGE_SOURCE_RECEIPT_EXPIRED", "IMAGE_SOURCE_ODOMETRY_EXPIRED",
                    "NATIVE_ODOMETRY_SOURCE_TIME_NOT_PAIRED",
                }:
                    raise
                continue
            if aligned.conservative_variance_m2 <= maximum_alignment_variance_m2:
                return received, stamp, aligned
        return None


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
                 channel_path: Path | None = None, source_clock_domain: str | None = None) -> None:
        if type(rate_hz) not in (int, float) or not 20 <= rate_hz <= 100:
            raise ValueError("NATIVE_SAMPLER_RATE_INVALID")
        self.path, self.rate_hz = path, rate_hz
        self.buffer = NativeStateBuffer(source_clock_domain=source_clock_domain)
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
        summary["history"] = self.buffer.history_summary()
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
