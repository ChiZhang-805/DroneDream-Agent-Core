"""Fail-closed fusion boundary for onboard indoor-navigation perception.

Calibrated metric rays remain collision-authoritative.  Optional forward RGB
and fixed-width realtime encoders add semantic and vehicle-state context, but
can neither create free space nor bypass deterministic collision validation.
Local continuous policies may command bounded body-frame axes; they never own
motor mixing, unrestricted attitude, or raw actuator output.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import (
    DynamicObstacleObservation,
    IndoorNavigationCycleReceipt,
    NormalizedPilotControl,
    OnboardPerceptionFrame,
    PerceptionFusionHealth,
    QuaternionWxyz,
    RuntimeLocalSafetyObservation,
    TextNavigationDecision,
    Vector3,
)
from .control_timing import (
    CONTINUOUS_CONTROL_MODE,
    LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
    LOCAL_DISPATCH_RESERVE_MS,
)
from .control_uncertainty import finite_positive_number
from .dynamic_clearance import fresh_free_voxels, reachable_track_box_observed_free
from .hashing import sha256_json
from .local_world_model import MetricVoxelMap
from .navigation_context import bounded_navigation_context as _bounded_strategic_navigation_context
from .navigation_snapshot import NavigationSnapshotRequest, compile_navigation_snapshot
from .navigation_snapshot import (
    bind_navigation_control_output_contract as _bind_navigation_control_output_contract,
)
from .observation_validity import SourceDeadlineLatch, image_control_deadline
from .perception_evidence import (
    FrozenDynamicObstacle,
    FrozenPerceptionFrame,
    freeze_perception_frame,
)
from .prompts import TEXT_NAVIGATION_ADVISOR
from .realtime_feature_encoders import RealtimeFeatureSnapshot
from .runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorEnvelope,
    RuntimeSensorRegistry,
)

if TYPE_CHECKING:
    from .model_harness.model_port import StructuredCallResult, StructuredModelPort


# 功能：
#   仅在支持的平台绑定当前模型工作线程的 CPU，不改变感知线程或整个进程的亲和性。
# 输入：
#   cpu_ids：经构造器核验的 CPU 编号。
# 输出：
#   None：不返回业务数据。
def _set_current_thread_cpu_affinity(cpu_ids: tuple[int, ...]) -> None:
    setter = getattr(os, "sched_setaffinity", None)
    if callable(setter) and cpu_ids:
        setter(0, set(cpu_ids))


@dataclass(frozen=True)
class _DynamicTrackState:
    observed_at_unix_ms: int
    position_m: Vector3
    velocity_mps: Vector3
    observation: DynamicObstacleObservation


# 功能：
#   在创建状态或工作线程前拒绝布尔、非数字与非有限物理配置，避免 NaN 绕过阈值比较。
# 输入：
#   values：以配置名标识的待检查物理参数。
# 输出：
#   None：不返回业务数据。
def _finite_configuration(**values: object) -> None:
    for name, value in values.items():
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite physical number")


class RuntimePerceptionFusion:
    """Validate ordered sensor frames and update a conservative voxel map."""

    # 功能：
    #   建立单一所有者的保守感知融合流，校验门控参数后保存地图与传感器注册表引用。
    # 输入：
    #   world：融合观测所更新的米制局部地图。
    #   accepted_sensor_ids：允许接入且已标定的传感器身份集合。
    #   maximum_stream_age_seconds：帧的新鲜度上限，单位秒。
    #   maximum_localization_covariance_m2：允许的定位方差上限，单位平方米。
    #   maximum_sensor_offset_m：传感器原点相对机体定位的距离上限，单位米。
    #   minimum_rays_per_frame：单帧最少有效射线数量。
    #   minimum_dynamic_track_confidence：动态目标可信度门槛。
    #   maximum_dynamic_speed_mps：动态目标速度上限，单位米每秒。
    #   maximum_dynamic_acceleration_mps2：跨帧目标加速度上限，单位米每平方秒。
    #   maximum_dynamic_innovation_m：动态预测位置的基础残差容差，单位米。
    #   sensor_registry：可选的多模态传感器注册表。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        world: MetricVoxelMap,
        accepted_sensor_ids: set[str],
        maximum_stream_age_seconds: float = 0.25,
        maximum_localization_covariance_m2: float = 0.25,
        maximum_sensor_offset_m: float = 2.0,
        minimum_rays_per_frame: int = 8,
        minimum_dynamic_track_confidence: float = 0.35,
        maximum_dynamic_speed_mps: float = 15.0,
        maximum_dynamic_acceleration_mps2: float = 12.0,
        maximum_dynamic_innovation_m: float = 1.0,
        sensor_registry: RuntimeSensorRegistry | None = None,
    ) -> None:
        _finite_configuration(
            maximum_stream_age_seconds=maximum_stream_age_seconds,
            maximum_localization_covariance_m2=maximum_localization_covariance_m2,
            maximum_sensor_offset_m=maximum_sensor_offset_m,
            minimum_dynamic_track_confidence=minimum_dynamic_track_confidence,
            maximum_dynamic_speed_mps=maximum_dynamic_speed_mps,
            maximum_dynamic_acceleration_mps2=maximum_dynamic_acceleration_mps2,
            maximum_dynamic_innovation_m=maximum_dynamic_innovation_m,
        )
        if (
            type(accepted_sensor_ids) not in (set, frozenset)
            or not 1 <= len(accepted_sensor_ids) <= 64
            or any(
                type(sensor) is not str
                or len(sensor) > 160
                or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", sensor) is None
                for sensor in accepted_sensor_ids
            )
        ):
            raise ValueError("at least one calibrated perception sensor is required")
        if maximum_stream_age_seconds <= 0.0:
            raise ValueError("maximum stream age must be positive")
        if maximum_localization_covariance_m2 <= 0.0:
            raise ValueError("maximum localization covariance must be positive")
        if maximum_sensor_offset_m <= 0.0:
            raise ValueError("maximum sensor offset must be positive")
        if type(minimum_rays_per_frame) is not int or minimum_rays_per_frame < 1:
            raise ValueError("minimum rays per frame must be positive")
        if not 0.0 <= minimum_dynamic_track_confidence <= 1.0:
            raise ValueError("dynamic track confidence must be in [0, 1]")
        if maximum_dynamic_speed_mps <= 0.0:
            raise ValueError("maximum dynamic speed must be positive")
        if maximum_dynamic_acceleration_mps2 <= 0.0:
            raise ValueError("maximum dynamic acceleration must be positive")
        if maximum_dynamic_innovation_m <= 0.0:
            raise ValueError("maximum dynamic innovation must be positive")
        self.world = world
        self.accepted_sensor_ids = frozenset(accepted_sensor_ids)
        self.maximum_stream_age_seconds = maximum_stream_age_seconds
        self.maximum_localization_covariance_m2 = maximum_localization_covariance_m2
        self.maximum_sensor_offset_m = maximum_sensor_offset_m
        self.minimum_rays_per_frame = minimum_rays_per_frame
        self.minimum_dynamic_track_confidence = minimum_dynamic_track_confidence
        self.maximum_dynamic_speed_mps = maximum_dynamic_speed_mps
        self.maximum_dynamic_acceleration_mps2 = maximum_dynamic_acceleration_mps2
        self.maximum_dynamic_innovation_m = maximum_dynamic_innovation_m
        self.sensor_registry = sensor_registry
        self._latest_sequence_by_sensor: dict[str, int] = {}
        self._latest_frame: FrozenPerceptionFrame | None = None
        self._accepted_ray_count = 0
        self._dynamic_tracks: dict[str, _DynamicTrackState] = {}
        self._dynamic_track_issue_codes: list[str] = []
        self._clearance_confirmations: dict[str, tuple[float, str]] = {}
        self.last_dynamic_clearances: list[dict[str, object]] = []

    # 功能：
    #   判断是否已有接纳帧，不复制射线，也不声称该帧仍然新鲜。
    # 输入：
    #   self：当前融合流。
    # 输出：
    #   present：存在已接纳帧时为 True。
    @property
    def has_frame(self) -> bool:
        present = self._latest_frame is not None
        return present

    # 功能：
    #   将当前不可变帧还原成可独立使用的合同副本，保留源时间、射线及动态目标的年龄。
    # 输入：
    #   self：当前融合流。
    # 输出：
    #   frame：独立感知帧，没有观测时为 None。
    @property
    def latest_frame(self) -> OnboardPerceptionFrame | None:
        frame = (
            OnboardPerceptionFrame.model_validate(self._latest_frame.model_dump(mode="python"))
            if self._latest_frame is not None
            else None
        )
        return frame

    # 功能：
    #   返回已经严格验证的深度不可变观测图，供内部读者零复制使用，不续租源时间。
    # 输入：
    #   self：当前融合流。
    # 输出：
    #   frame：不可变感知帧，没有观测时为 None。
    @property
    def frozen_frame(self) -> FrozenPerceptionFrame | None:
        frame = self._latest_frame
        return frame

    # 功能：
    #   暂存动态目标更新，保留失踪目标原始检测时间，拒绝重复身份、同时间改几何及运动突变。
    # 输入：
    #   frame：已通过入口冻结验证的感知帧。
    # 输出：
    #   proposed：尚未提交的目标历史。
    #   issues：低置信度目标等健康问题列表。
    def _validated_dynamic_tracks(
        self, frame: OnboardPerceptionFrame
    ) -> tuple[dict[str, _DynamicTrackState], list[str]]:
        proposed = dict(self._dynamic_tracks)
        issues: list[str] = []
        ids = [obstacle.obstacle_id for obstacle in frame.dynamic_obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("DYNAMIC_TRACK_DUPLICATE_ID")
        for obstacle in frame.dynamic_obstacles:
            speed = math.dist((0.0, 0.0, 0.0), tuple(obstacle.velocity_mps.model_dump().values()))
            if speed > self.maximum_dynamic_speed_mps:
                raise ValueError("DYNAMIC_TRACK_SPEED_EXCEEDED")
            age_ms = obstacle.age_seconds * 1_000.0
            if not math.isfinite(age_ms):
                raise ValueError("DYNAMIC_TRACK_SOURCE_TIME_INVALID")
            observed_at = frame.observed_at_unix_ms - round(age_ms)
            if observed_at < 0:
                raise ValueError("DYNAMIC_TRACK_SOURCE_TIME_INVALID")
            if obstacle.confidence < self.minimum_dynamic_track_confidence:
                issues.append(f"DYNAMIC_TRACK_CONFIDENCE_LOW:{obstacle.obstacle_id}")
            previous = proposed.get(obstacle.obstacle_id)
            if previous is not None:
                elapsed_seconds = (observed_at - previous.observed_at_unix_ms) / 1_000.0
                if elapsed_seconds == 0.0:
                    # A tracker may explicitly retain a briefly occluded
                    # observation. Only its age can change; never rejuvenate
                    # it or accept altered geometry under the old timestamp.
                    if obstacle.model_dump(exclude={"age_seconds"}) != (
                        previous.observation.model_dump(exclude={"age_seconds"})
                    ):
                        raise ValueError("DYNAMIC_TRACK_REPLAY_CONFLICT")
                    continue
                if elapsed_seconds < 0.0:
                    raise ValueError("DYNAMIC_TRACK_TIMESTAMP_NOT_MONOTONIC")
                previous_position = tuple(previous.position_m.model_dump().values())
                previous_velocity = tuple(previous.velocity_mps.model_dump().values())
                predicted = tuple(
                    previous_position[axis] + previous_velocity[axis] * elapsed_seconds
                    for axis in range(3)
                )
                current_position = tuple(obstacle.position_m.model_dump().values())
                innovation = math.dist(current_position, predicted)
                allowable_innovation = (
                    self.maximum_dynamic_innovation_m
                    + 0.1 * self.maximum_dynamic_acceleration_mps2 * elapsed_seconds**2
                )
                if innovation > allowable_innovation:
                    raise ValueError("DYNAMIC_TRACK_POSITION_INNOVATION_EXCEEDED")
                acceleration = (
                    math.dist(tuple(obstacle.velocity_mps.model_dump().values()), previous_velocity)
                    / elapsed_seconds
                )
                if acceleration > self.maximum_dynamic_acceleration_mps2:
                    raise ValueError("DYNAMIC_TRACK_ACCELERATION_EXCEEDED")
            owned_obstacle = obstacle.model_copy(deep=True)
            proposed[obstacle.obstacle_id] = _DynamicTrackState(
                observed_at_unix_ms=observed_at,
                position_m=owned_obstacle.position_m,
                velocity_mps=owned_obstacle.velocity_mps,
                observation=owned_obstacle,
            )
        if len(proposed) > 256:
            raise ValueError("DYNAMIC_TRACK_CAPACITY_EXCEEDED")
        return proposed, issues

    # 功能：
    #   1. 严格冻结原始观测后检查来源、时钟和动态历史，不借用生产者正在修改的帧。
    #   2. 为地图和注册表准备同一帧的证据；只有被新扫描覆盖验证的区域才能清除旧目标。
    # 输入：
    #   frame：待接纳的标定感知帧。
    #   now_unix_ms：本次检查的 Unix 毫秒时钟。
    #   now_monotonic_seconds：可选本机单调秒数，省略时读取当前单调时钟。
    # 输出：
    #   health：提交后感知健康摘要，不代表定位精度或飞行许可。
    def ingest(
        self,
        frame: OnboardPerceptionFrame,
        *,
        now_unix_ms: int,
        now_monotonic_seconds: float | None = None,
    ) -> PerceptionFusionHealth:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("PERCEPTION_CLOCK_INVALID")
        if now_monotonic_seconds is not None and (
            type(now_monotonic_seconds) not in (int, float)
            or not (now_monotonic_seconds == 0 or finite_positive_number(now_monotonic_seconds))
        ):
            raise ValueError("PERCEPTION_CLOCK_INVALID")
        frame = freeze_perception_frame(frame)
        if frame.sensor_id not in self.accepted_sensor_ids:
            raise ValueError("PERCEPTION_SENSOR_NOT_CALIBRATED")
        previous_sequence = self._latest_sequence_by_sensor.get(frame.sensor_id, 0)
        if frame.sequence <= previous_sequence:
            raise ValueError("PERCEPTION_SEQUENCE_NOT_MONOTONIC")
        if self._latest_frame is not None and (
            frame.observed_at_unix_ms < self._latest_frame.observed_at_unix_ms
        ):
            raise ValueError("PERCEPTION_TIMESTAMP_REGRESSED")
        age_seconds = (now_unix_ms - frame.observed_at_unix_ms) / 1_000.0
        if age_seconds < -0.05:
            raise ValueError("PERCEPTION_TIMESTAMP_IN_FUTURE")
        if age_seconds > self.maximum_stream_age_seconds:
            raise ValueError("PERCEPTION_FRAME_STALE")
        if frame.localization_covariance_m2 > self.maximum_localization_covariance_m2:
            raise ValueError("LOCALIZATION_COVARIANCE_EXCEEDED")
        if len(frame.range_rays) < self.minimum_rays_per_frame:
            raise ValueError("PERCEPTION_RAY_DENSITY_INSUFFICIENT")
        localization = frame.localization_position_m
        for ray in frame.range_rays:
            sensor_offset = math.dist(
                (ray.origin_m.x, ray.origin_m.y, ray.origin_m.z),
                (localization.x, localization.y, localization.z),
            )
            if sensor_offset > self.maximum_sensor_offset_m:
                raise ValueError("PERCEPTION_RAY_ORIGIN_OUTSIDE_CALIBRATED_BODY")
        dynamic_tracks, dynamic_issues = self._validated_dynamic_tracks(frame)
        accepted_at_monotonic = (
            time.monotonic() if now_monotonic_seconds is None else now_monotonic_seconds
        )
        clearance_confirmations: dict[str, tuple[float, str]] = {}
        clearances: list[dict[str, object]] = []
        missing = {
            key: state
            for key, state in dynamic_tracks.items()
            if frame.observed_at_unix_ms > state.observed_at_unix_ms
        }
        if missing:
            free = fresh_free_voxels(
                frame.range_rays,
                resolution_m=self.world.resolution_m,
                now_monotonic_seconds=accepted_at_monotonic,
            )
            scan_time = max(ray.observed_at_monotonic_seconds for ray in frame.range_rays)
            scan_sha = sha256_json(frame.range_rays)
            for key, state in missing.items():
                track_age = (frame.observed_at_unix_ms - state.observed_at_unix_ms) / 1000
                clear = reachable_track_box_observed_free(
                    state.observation,
                    age_seconds=track_age,
                    free_voxels=free,
                    resolution_m=self.world.resolution_m,
                    localization_variance_m2=frame.localization_covariance_m2,
                    maximum_acceleration_mps2=self.maximum_dynamic_acceleration_mps2,
                )
                previous = self._clearance_confirmations.get(key)
                if not clear:
                    continue
                if (
                    previous is not None
                    and scan_sha != previous[1]
                    and 0 < scan_time - previous[0] <= 0.25
                ):
                    del dynamic_tracks[key]
                    clearances.append(
                        {
                            "obstacle_id": key,
                            "original_detection_unix_ms": state.observed_at_unix_ms,
                            "cleared_at_unix_ms": frame.observed_at_unix_ms,
                            "prior_scan_sha256": previous[1],
                            "scan_sha256": scan_sha,
                            "method": "two-fresh-scans-cover-inflated-reachable-box",
                        }
                    )
                else:
                    # A re-read scan cannot provide a second confirmation.
                    clearance_confirmations[key] = (
                        previous if previous and scan_time == previous[0] else (scan_time, scan_sha)
                    )
        # No hit, incomplete coverage, stale data or a missing ID alone never
        # clears a track. Remaining evidence retains its original deadline.
        retained = tuple(
            FrozenDynamicObstacle.model_validate(
                {
                    **state.observation.model_dump(mode="python"),
                    "age_seconds": (frame.observed_at_unix_ms - state.observed_at_unix_ms) / 1_000,
                },
                strict=True,
            )
            for _, state in sorted(dynamic_tracks.items())
        )
        # 上面仅更新已严格验证、不可变的目标元组；射线共享入口冻结图，不再整帧重复构建。
        frame = frame.model_copy(update={"dynamic_obstacles": retained})
        prepared_scan = self.world.prepare_scan(frame.range_rays)
        if self.sensor_registry is not None:
            contract = self.sensor_registry.contract(frame.sensor_id)
            observed_at_monotonic = max(
                ray.observed_at_monotonic_seconds for ray in frame.range_rays
            )
            mean_quality = sum(ray.confidence for ray in frame.range_rays) / len(frame.range_rays)
            self.sensor_registry.ingest(
                RuntimeSensorEnvelope(
                    sensor_id=contract.sensor_id,
                    vehicle_id=contract.vehicle_id,
                    modality=contract.modality,
                    sequence=frame.sequence,
                    sample_monotonic_seconds=observed_at_monotonic,
                    received_monotonic_seconds=accepted_at_monotonic,
                    coordinate_frame=contract.coordinate_frame,
                    unit=contract.unit,
                    calibration_sha256=contract.calibration_sha256,
                    intrinsics_sha256=contract.intrinsics_sha256,
                    extrinsics_sha256=contract.extrinsics_sha256,
                    covariance_diagonal=[frame.localization_covariance_m2] * 3,
                    quality=mean_quality,
                    coverage=frame.source_coverage if frame.source_coverage is not None else 0.0,
                    payload_sha256=sha256_json(frame),
                    source_kind="derived",
                ),
                now_monotonic_seconds=accepted_at_monotonic,
            )
        self.world.commit_scan(prepared_scan)
        self._latest_sequence_by_sensor[frame.sensor_id] = frame.sequence
        self._latest_frame = frame
        self._accepted_ray_count += len(frame.range_rays)
        self._dynamic_tracks = dynamic_tracks
        self._clearance_confirmations = clearance_confirmations
        self.last_dynamic_clearances = clearances
        self._dynamic_track_issue_codes = dynamic_issues
        health = self.health(
            now_unix_ms=now_unix_ms,
            now_monotonic_seconds=accepted_at_monotonic,
        )
        return health

    # 功能：
    #   汇总当前观测和注册表健康，不续租、删除或重新观测任何证据。
    # 输入：
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   now_monotonic_seconds：可选本机单调秒数。
    # 输出：
    #   health：感知时效、定位质量及缺失证据的状态摘要。
    def health(
        self,
        *,
        now_unix_ms: int,
        now_monotonic_seconds: float | None = None,
    ) -> PerceptionFusionHealth:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("PERCEPTION_CLOCK_INVALID")
        if now_monotonic_seconds is not None and (
            type(now_monotonic_seconds) not in (int, float)
            or not (now_monotonic_seconds == 0 or finite_positive_number(now_monotonic_seconds))
        ):
            raise ValueError("PERCEPTION_CLOCK_INVALID")
        frame = self._latest_frame
        if frame is None:
            health = PerceptionFusionHealth(
                sensor_id="none",
                latest_sequence=0,
                stream_age_seconds=0.0,
                localization_covariance_m2=0.0,
                accepted_ray_count=0,
                map_observation_count=self.world.observation_count,
                stream_healthy=False,
                issue_codes=["PERCEPTION_FRAME_NOT_RECEIVED"],
            )
            return health
        age_seconds = max(0.0, (now_unix_ms - frame.observed_at_unix_ms) / 1_000.0)
        issues: list[str] = []
        if frame.observed_at_unix_ms > now_unix_ms + 50:
            issues.append("PERCEPTION_TIMESTAMP_IN_FUTURE")
        if age_seconds > self.maximum_stream_age_seconds:
            issues.append("PERCEPTION_STREAM_STALE")
        if frame.localization_covariance_m2 > self.maximum_localization_covariance_m2:
            issues.append("LOCALIZATION_COVARIANCE_EXCEEDED")
        if self.world.observation_count < self.minimum_rays_per_frame:
            issues.append("METRIC_MAP_EVIDENCE_INSUFFICIENT")
        issues.extend(self._dynamic_track_issue_codes)
        for track_id, track in self._dynamic_tracks.items():
            if (now_unix_ms - track.observed_at_unix_ms) / 1_000 > 1.0:
                issues.append(f"DYNAMIC_TRACK_OBSERVATION_STALE:{track_id}")
            low_confidence = f"DYNAMIC_TRACK_CONFIDENCE_LOW:{track_id}"
            if track.observation.confidence < self.minimum_dynamic_track_confidence and (
                low_confidence not in issues
            ):
                issues.append(low_confidence)
        if self.sensor_registry is not None:
            multimodal = self.sensor_registry.snapshot(
                now_monotonic_seconds=(
                    time.monotonic() if now_monotonic_seconds is None else now_monotonic_seconds
                )
            )
            if not multimodal.ready_for_motion:
                issues.extend(f"MULTIMODAL_SENSOR:{issue}" for issue in multimodal.issue_codes[:24])
        health = PerceptionFusionHealth(
            sensor_id=frame.sensor_id,
            latest_sequence=frame.sequence,
            stream_age_seconds=age_seconds,
            localization_covariance_m2=frame.localization_covariance_m2,
            accepted_ray_count=self._accepted_ray_count,
            map_observation_count=self.world.observation_count,
            stream_healthy=not issues,
            issue_codes=(
                issues
                if len(issues) <= 31
                else [*issues[:31], "PERCEPTION_ADDITIONAL_ISSUES_OMITTED"]
            ),
        )
        return health

    # 功能：
    #   汇总注册表的多模态健康情况，未配置注册表时明确返回未知而非全部就绪。
    # 输入：
    #   now_monotonic_seconds：可选单调秒数，省略时使用当前本机时钟。
    # 输出：
    #   snapshot：多模态状态快照或 None。
    def multimodal_sensor_snapshot(
        self,
        *,
        now_monotonic_seconds: float | None = None,
    ) -> RuntimeMultimodalSensorSnapshot | None:
        if self.sensor_registry is None:
            return None
        snapshot = self.sensor_registry.snapshot(
            now_monotonic_seconds=(
                time.monotonic() if now_monotonic_seconds is None else now_monotonic_seconds
            )
        )
        return snapshot

    # 功能：
    #   从最近接纳帧生成安全层输入，保留真实采集时间与陈旧年龄，不凭调用刷新观测。
    # 输入：
    #   target_position_m：当前任务的宏观目标位置，单位米。
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   body_orientation_world_from_body：可选的已测机体姿态。
    #   current_acceleration_world_enu_mps2：可选的世界系实际加速度。
    # 输出：
    #   observation：带感知健康和动态障碍的安全观测。
    def local_safety_observation(
        self,
        *,
        target_position_m: Vector3,
        now_unix_ms: int,
        body_orientation_world_from_body: QuaternionWxyz | None = None,
        current_acceleration_world_enu_mps2: Vector3 | None = None,
    ) -> RuntimeLocalSafetyObservation:
        frame = self._latest_frame
        if frame is None:
            raise ValueError("PERCEPTION_FRAME_NOT_RECEIVED")
        health = self.health(now_unix_ms=now_unix_ms)
        observation = RuntimeLocalSafetyObservation(
            sequence=frame.sequence,
            observed_at_unix_ms=frame.observed_at_unix_ms,
            source="onboard",
            stream_healthy=health.stream_healthy,
            stream_age_seconds=health.stream_age_seconds,
            localization_covariance_m2=frame.localization_covariance_m2,
            current_position_m=frame.localization_position_m,
            current_velocity_mps=frame.localization_velocity_mps,
            current_acceleration_world_enu_mps2=(current_acceleration_world_enu_mps2),
            body_orientation_world_from_body=body_orientation_world_from_body,
            target_position_m=target_position_m,
            dynamic_obstacles=frame.dynamic_obstacles,
        )
        return observation

    # 功能：
    #   仅为健康感知流编译米制候选，连同健康判定一起绑定快照摘要。
    # 输入：
    #   goal_position_m：任务的宏观目标位置，单位米。
    #   required_clearance_m：所需净空，单位米。
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   candidate_speed_mps：候选运动的评估速度，单位米每秒。
    #   vehicle_radius_m：机体水平半径，单位米。
    #   vehicle_height_m：机体总高度，单位米。
    # 输出：
    #   snapshot：含健康和授权候选的导航输入字典。
    def text_navigation_snapshot(
        self,
        *,
        goal_position_m: Vector3,
        required_clearance_m: float,
        now_unix_ms: int,
        candidate_speed_mps: float = 0.5,
        vehicle_radius_m: float = 0.38,
        vehicle_height_m: float = 0.43,
    ) -> dict[str, object]:
        frame = self._latest_frame
        if frame is None:
            raise ValueError("PERCEPTION_FRAME_NOT_RECEIVED")
        health = self.health(now_unix_ms=now_unix_ms)
        if not health.stream_healthy:
            raise ValueError("PERCEPTION_STREAM_NOT_HEALTHY")
        snapshot = self.world.text_navigation_snapshot(
            current_position_m=frame.localization_position_m,
            current_velocity_mps=frame.localization_velocity_mps,
            goal_position_m=goal_position_m,
            dynamic_obstacles=frame.dynamic_obstacles,
            required_clearance_m=required_clearance_m,
            candidate_speed_mps=candidate_speed_mps,
            vehicle_radius_m=vehicle_radius_m,
            vehicle_height_m=vehicle_height_m,
        )
        snapshot["perception_health"] = health.model_dump(mode="json")
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        return snapshot


# 功能：
#   验证快照、决策身份和明确布尔门控，返回选中候选的独立副本；连续控制不伪造候选点。
# 输入：
#   snapshot：包含摘要的导航快照。
#   decision：模型返回的结构化导航决策。
# 输出：
#   selected：通过身份与米制门控的候选，非候选动作时为 None。
def validate_text_navigation_decision(
    snapshot: dict[str, object],
    decision: TextNavigationDecision,
) -> dict[str, object] | None:
    snapshot_payload = copy_json(snapshot, limit=8 * 1024 * 1024)
    if type(snapshot_payload) is not dict or not isinstance(decision, TextNavigationDecision):
        raise ValueError("TEXT_NAVIGATION_INPUT_INVALID")
    decision = TextNavigationDecision.model_validate(
        decision.model_dump(mode="python"), strict=True
    )
    snapshot_sha256 = snapshot_payload.pop("snapshot_sha256", "")
    if snapshot_sha256 != sha256_json(snapshot_payload):
        raise ValueError("TEXT_NAVIGATION_SNAPSHOT_HASH_INVALID")
    if decision.snapshot_sha256 != snapshot_sha256:
        raise ValueError("TEXT_NAVIGATION_DECISION_SNAPSHOT_MISMATCH")
    health = snapshot_payload.get("perception_health")
    if (
        type(health) is not dict or health.get("stream_healthy") is not True
    ) and decision.action not in {"hold", "abort"}:
        raise ValueError("TEXT_NAVIGATION_UNHEALTHY_STREAM_REQUIRES_HOLD")
    if decision.action != "select-candidate":
        return None
    candidates = snapshot_payload.get("authorized_candidate_paths")
    if not isinstance(candidates, list) or len(candidates) > 64:
        raise ValueError("TEXT_NAVIGATION_CANDIDATE_SET_MISSING")
    identifiers = [item.get("candidate_id") for item in candidates if type(item) is dict]
    if (
        len(identifiers) != len(candidates)
        or any(type(identifier) is not str or not identifier for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("TEXT_NAVIGATION_CANDIDATE_IDENTITY_INVALID")
    selected = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("candidate_id") == decision.selected_candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("TEXT_NAVIGATION_CANDIDATE_NOT_AUTHORIZED")
    if selected.get("deterministic_metric_path_validated") is not True:
        raise ValueError("TEXT_NAVIGATION_CANDIDATE_NOT_METRIC_VALIDATED")
    return selected


# 功能：
#   固定导航输入后调用模型，核对其返回决策；错误决策仍保留已经发生的物理调用回执。
# 输入：
#   port：供应商或本地策略的结构化端口。
#   snapshot：本次模型唯一可引用的导航快照。
#   context_id：可选模型上下文身份。
#   multimodal：可选的已绑定图像输入。
# 输出：
#   result：独立复制的模型返回结果。
#   selected：合法候选或 None。
def request_text_navigation_decision(
    *,
    port: StructuredModelPort,
    snapshot: dict[str, object],
    context_id: str | None = None,
    multimodal: list[dict[str, object]] | None = None,
) -> tuple[StructuredCallResult[TextNavigationDecision], dict[str, object] | None]:
    snapshot = copy_json(snapshot, limit=8 * 1024 * 1024)
    result = port.call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions=TEXT_NAVIGATION_ADVISOR,
        input_artifact={"text_navigation_snapshot": copy.deepcopy(snapshot)},
        context_id=context_id,
        multimodal=multimodal,
    )
    result = copy.deepcopy(result)
    try:
        selected = validate_text_navigation_decision(snapshot, result.artifact)
    except (ValueError, TypeError, AttributeError) as error:
        # 拒绝输出不能抹去已发生的推理；仅传回回执，不携带供应商正文。
        error.model_call_record = getattr(result, "record", None)
        raise
    return result, selected


class TextOnlyIndoorNavigationCoordinator:
    """Run event-driven model navigation over a fail-closed metric world model.

    The model selects a hash-bound semantic goal/frontier candidate.  This
    coordinator then independently regenerates the full-resolution 3-D path
    before exposing one adjacent waypoint to the high-rate safety controller.
    A down-sampled path shown to the model is never used as an actuator path.
    """

    # 功能：
    #   建立显式的同步候选路径兼容协调器，不将它当成传感器速率连续控制环。
    # 输入：
    #   fusion：持有当前感知与米制地图的融合器。
    #   port：负责选择候选的模型端口。
    #   required_clearance_m：路径净空下限，单位米。
    #   candidate_speed_mps：评估动态障碍的候选速度，单位米每秒。
    #   vehicle_radius_m：机体水平半径，单位米。
    #   vehicle_height_m：机体高度，单位米。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        fusion: RuntimePerceptionFusion,
        port: StructuredModelPort,
        required_clearance_m: float,
        candidate_speed_mps: float = 0.5,
        vehicle_radius_m: float = 0.38,
        vehicle_height_m: float = 0.43,
    ) -> None:
        _finite_configuration(
            required_clearance_m=required_clearance_m,
            candidate_speed_mps=candidate_speed_mps,
            vehicle_radius_m=vehicle_radius_m,
            vehicle_height_m=vehicle_height_m,
        )
        if required_clearance_m < 0.0:
            raise ValueError("required clearance must be non-negative")
        if min(candidate_speed_mps, vehicle_radius_m, vehicle_height_m) <= 0:
            raise ValueError("navigation speed and vehicle dimensions must be positive")
        self.fusion = fusion
        self.port = port
        self.required_clearance_m = required_clearance_m
        self.candidate_speed_mps = candidate_speed_mps
        self.vehicle_radius_m = vehicle_radius_m
        self.vehicle_height_m = vehicle_height_m
        self._sequence = 0

    # 功能：
    #   请求候选选择并在最新地图上重新求解、核验全分辨率路径；缺证据时返回保持回执。
    # 输入：
    #   goal_position_m：宏观目标位置，单位米。
    #   now_unix_ms：调用时的 Unix 毫秒时钟。
    #   trigger：本次导航评估的触发原因。
    #   context_id：可选模型上下文身份。
    # 输出：
    #   receipt：本次候选导航判定及其验证证据。
    def evaluate(
        self,
        *,
        goal_position_m: Vector3,
        now_unix_ms: int,
        trigger: str,
        context_id: str | None = None,
    ) -> IndoorNavigationCycleReceipt:
        self._sequence += 1
        health = self.fusion.health(now_unix_ms=now_unix_ms)
        receipt_fields: dict[str, object] = {
            "sequence": self._sequence,
            "trigger": trigger,
            "perception_health": health,
        }
        if not health.stream_healthy:
            return IndoorNavigationCycleReceipt(
                **receipt_fields,
                model_action="not-invoked",
                hold_reason="PERCEPTION_STREAM_NOT_HEALTHY",
            )

        snapshot = self.fusion.text_navigation_snapshot(
            goal_position_m=goal_position_m,
            required_clearance_m=self.required_clearance_m,
            now_unix_ms=now_unix_ms,
            candidate_speed_mps=self.candidate_speed_mps,
            vehicle_radius_m=self.vehicle_radius_m,
            vehicle_height_m=self.vehicle_height_m,
        )
        result, selected = request_text_navigation_decision(
            port=self.port,
            snapshot=snapshot,
            context_id=context_id,
        )
        decision = result.artifact
        call_record = getattr(result, "record", None)
        receipt_fields.update(
            {
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "model_action": decision.action,
                "model_call_id": getattr(call_record, "call_id", None),
            }
        )
        if selected is None:
            return IndoorNavigationCycleReceipt(
                **receipt_fields,
                hold_reason=(
                    "MODEL_REQUESTED_ABORT"
                    if decision.action == "abort"
                    else "MODEL_DID_NOT_SELECT_AUTHORIZED_PATH"
                ),
            )

        current = Vector3.model_validate(snapshot["current_position_m"])
        endpoint = Vector3.model_validate(selected["endpoint_m"])
        temporary_blocked_keys = frozenset()
        latest_frame = self.fusion.latest_frame
        if selected.get("kind") == "goal-dynamic-detour" and latest_frame is not None:
            temporary_blocked_keys = self.fusion.world.dynamic_obstacle_blocked_keys(
                dynamic_obstacles=latest_frame.dynamic_obstacles,
                required_clearance_m=self.required_clearance_m,
                vehicle_radius_m=self.vehicle_radius_m,
                vehicle_height_m=self.vehicle_height_m,
            )
        try:
            if selected.get("kind") == "goal-direct":
                regenerated_path = [current, endpoint]
                if not self.fusion.world.path_clearance_valid(
                    regenerated_path,
                    required_clearance_m=self.required_clearance_m,
                    temporary_blocked_keys=temporary_blocked_keys,
                    allow_first_position_as_start=True,
                ):
                    raise ValueError("direct goal path is no longer collision-free")
            else:
                regenerated_path = self.fusion.world.plan_path(
                    start_m=current,
                    goal_m=endpoint,
                    required_clearance_m=self.required_clearance_m,
                    temporary_blocked_keys=temporary_blocked_keys,
                    allow_current_position_as_start=True,
                )
        except ValueError:
            return IndoorNavigationCycleReceipt(
                **receipt_fields,
                selected_candidate_id=str(selected["candidate_id"]),
                hold_reason="AUTHORIZED_PATH_NO_LONGER_COLLISION_FREE",
            )
        regenerated_sha256 = sha256_json(
            [point.model_dump(mode="json") for point in regenerated_path]
        )
        if regenerated_sha256 != selected["metric_path_sha256"]:
            return IndoorNavigationCycleReceipt(
                **receipt_fields,
                selected_candidate_id=str(selected["candidate_id"]),
                hold_reason="METRIC_MAP_CHANGED_DURING_MODEL_DECISION",
            )
        if selected.get("dynamic_path_validated") is not True:
            return IndoorNavigationCycleReceipt(
                **receipt_fields,
                selected_candidate_id=str(selected["candidate_id"]),
                selected_metric_path_sha256=regenerated_sha256,
                deterministic_metric_path_revalidated=True,
                hold_reason="DYNAMIC_PATH_NOT_VALIDATED",
            )
        controller_target = next(
            (
                point
                for point in regenerated_path[1:]
                if math.dist(
                    (current.x, current.y, current.z),
                    (point.x, point.y, point.z),
                )
                >= self.fusion.world.resolution_m * 0.4
            ),
            endpoint,
        )
        return IndoorNavigationCycleReceipt(
            **receipt_fields,
            selected_candidate_id=str(selected["candidate_id"]),
            selected_metric_path_sha256=regenerated_sha256,
            controller_target_m=controller_target,
            deterministic_metric_path_revalidated=True,
            dynamic_path_revalidated=True,
        )


@dataclass(frozen=True)
class _NavigationWorkerResult:
    snapshot: dict[str, object] | None
    model_result: Any | None = None
    selected_candidate: dict[str, object] | None = None
    failure_reason: str | None = None
    failure_diagnostic: dict[str, object] | None = None
    discarded_call_record: Any | None = None


# 功能：
#   提取有界失败类型、原因码和数值诊断，仅按安全字段分组计算指纹，不调用异常的文本转换。
# 输入：
#   error：发生在快照、调用或后台结果阶段的异常。
#   stage：已知流水线阶段名称。
# 输出：
#   diagnostic：可安全放入导航失败回执的诊断字段。
def _failure_diagnostic(error: BaseException, *, stage: str) -> dict[str, object]:
    root = error
    chain = [error]
    seen: set[int] = set()
    while root.__cause__ is not None and id(root) not in seen and len(chain) < 16:
        seen.add(id(root))
        root = root.__cause__
        chain.append(root)
    exception_type = f"{type(error).__module__}.{type(error).__name__}"
    root_type = f"{type(root).__module__}.{type(root).__name__}"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,159}", exception_type) is None:
        exception_type = "builtins.Exception"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,159}", root_type) is None:
        root_type = "builtins.Exception"
    reason_code = next(
        (
            value
            for item in chain
            if isinstance((value := getattr(item, "reason_code", None)), str)
            and value
            and len(value) <= 160
            and value.isascii()
            and value[0].isalpha()
            and value.replace("_", "").isalnum()
            and value == value.upper()
        ),
        {
            "snapshot-compilation": "METRIC_NAVIGATION_SNAPSHOT_UNAVAILABLE",
            "model-invocation": "MODEL_INVOCATION_FAILED",
            "worker-result": "MODEL_WORKER_RESULT_FAILED",
        }.get(stage, "RUNTIME_FAILURE"),
    )
    # 异常正文可能很大、包含秘密或实现不可靠的 __str__；诊断不能再次破坏安全回执。
    fingerprint_source = "|".join(
        (
            stage,
            exception_type,
            root_type,
            reason_code,
        )
    )
    diagnostic: dict[str, object] = {
        "failure_stage": stage,
        "failure_exception_type": exception_type,
        "failure_root_exception_type": root_type,
        "failure_reason_code": reason_code,
        "failure_fingerprint": hashlib.sha256(
            fingerprint_source.encode("utf-8", errors="replace")
        ).hexdigest(),
    }
    diagnostic_metrics = next(
        (
            value
            for item in chain
            if isinstance(
                (value := getattr(item, "diagnostic_metrics", None)),
                dict,
            )
        ),
        None,
    )
    if diagnostic_metrics:
        safe_metrics = {}
        for index, (key, value) in enumerate(diagnostic_metrics.items()):
            if index >= 16:
                break
            if (
                type(key) is str
                and re.fullmatch(r"[a-z][a-z0-9-]{0,79}", key)
                and type(value) in (int, float)
                and 0 <= value <= 60_000
            ):
                safe_metrics[key] = value
        if safe_metrics:
            diagnostic["failure_diagnostic_metrics"] = safe_metrics
    return diagnostic


# 功能：
#   在后台编译输入并调用模型，原始观测期限贯穿准备与推理，过期结果只保留调用证据。
# 输入：
#   world：为本次请求冻结的局部米制地图。
#   frame：带原始采集时间的不可变感知帧。
#   health：提交时的感知健康判定。
#   port：本次选择的模型端口。
#   goal_position_m：宏观目标位置，单位米。
#   required_clearance_m：所需净空，单位米。
#   candidate_speed_mps：候选评估速度，单位米每秒。
#   vehicle_radius_m：机体水平半径，单位米。
#   vehicle_height_m：机体高度，单位米。
#   context_id：可选模型上下文身份。
#   multimodal：已冻结的图像传输输入。
#   visual_evidence：与图像字节对应的证据摘要。
#   multimodal_sensor_snapshot：可选多模态健康状态。
#   realtime_feature_snapshot：可选本地编码特征。
#   strategic_context：经过范围限制的宏观任务上下文。
#   maximum_snapshot_planning_seconds：快照规划计算预算，单位秒。
#   include_candidate_paths：是否编译兼容候选路径。
#   control_reference_observed_at_unix_ms：本次控制状态的原始毫秒时钟。
#   invocation_deadline_monotonic：连续控制请求的单调时钟截止时间。
# 输出：
#   worker_result：后台结果、失败原因或因过期保留的调用回执。
def _compile_and_request_navigation_decision(
    *,
    world: MetricVoxelMap,
    frame: OnboardPerceptionFrame,
    health: PerceptionFusionHealth,
    port: StructuredModelPort,
    goal_position_m: Vector3,
    required_clearance_m: float,
    candidate_speed_mps: float,
    vehicle_radius_m: float,
    vehicle_height_m: float,
    context_id: str | None,
    multimodal: list[dict[str, object]],
    visual_evidence: list[dict[str, object]],
    multimodal_sensor_snapshot: dict[str, object] | None,
    realtime_feature_snapshot: dict[str, object] | None,
    strategic_context: dict[str, object],
    maximum_snapshot_planning_seconds: float,
    include_candidate_paths: bool,
    control_reference_observed_at_unix_ms: int,
    invocation_deadline_monotonic: float | None = None,
) -> _NavigationWorkerResult:
    try:
        snapshot = compile_navigation_snapshot(
            NavigationSnapshotRequest(
                world=world,
                frame=frame,
                health=health,
                goal_position_m=goal_position_m,
                required_clearance_m=required_clearance_m,
                candidate_speed_mps=candidate_speed_mps,
                vehicle_radius_m=vehicle_radius_m,
                vehicle_height_m=vehicle_height_m,
                visual_evidence=visual_evidence,
                multimodal_sensor_snapshot=multimodal_sensor_snapshot,
                realtime_feature_snapshot=realtime_feature_snapshot,
                strategic_context=strategic_context,
                maximum_snapshot_planning_seconds=maximum_snapshot_planning_seconds,
                include_candidate_paths=include_candidate_paths,
                control_reference_observed_at_unix_ms=control_reference_observed_at_unix_ms,
            )
        )
    except ValueError as error:
        return _NavigationWorkerResult(
            snapshot=None,
            failure_reason="METRIC_NAVIGATION_SNAPSHOT_UNAVAILABLE",
            failure_diagnostic=_failure_diagnostic(error, stage="snapshot-compilation"),
        )
    if (
        invocation_deadline_monotonic is not None
        and time.monotonic() + LOCAL_DISPATCH_RESERVE_MS / 1000 >= invocation_deadline_monotonic
    ):
        return _NavigationWorkerResult(
            snapshot=snapshot, failure_reason="CONTROL_SOURCE_EXPIRED_DURING_PREPARATION"
        )
    try:
        result, selected = request_text_navigation_decision(
            port=port,
            snapshot=snapshot,
            context_id=context_id,
            multimodal=multimodal,
        )
    except Exception as error:
        return _NavigationWorkerResult(
            snapshot=snapshot,
            failure_reason="MODEL_NAVIGATION_INVOCATION_FAILED",
            failure_diagnostic=_failure_diagnostic(error, stage="model-invocation"),
            discarded_call_record=getattr(error, "model_call_record", None),
        )
    if (
        invocation_deadline_monotonic is not None
        and time.monotonic() + LOCAL_DISPATCH_RESERVE_MS / 1000 >= invocation_deadline_monotonic
    ):
        return _NavigationWorkerResult(
            snapshot=snapshot,
            failure_reason="CONTROL_SOURCE_EXPIRED_DURING_INFERENCE",
            discarded_call_record=getattr(result, "record", None),
        )
    return _NavigationWorkerResult(
        snapshot=snapshot, model_result=result, selected_candidate=selected
    )


@dataclass(frozen=True)
class _PendingNavigationCycle:
    sequence: int
    trigger: str
    perception_health: PerceptionFusionHealth
    goal_position_m: Vector3
    navigation_goal_id: str | None
    submitted_at_unix_ms: int
    maximum_decision_age_seconds: float
    future: Future[_NavigationWorkerResult]


@dataclass(frozen=True)
class _PathRevalidationResult:
    path: tuple[Vector3, ...] = ()
    path_sha256: str | None = None
    controller_target_m: Vector3 | None = None
    path_continuity_retained: bool = False
    failure_reason: str | None = None


# 功能：
#   从当前定位在局部授权折线上的最近投影处裁出剩余几何，不据此推进任务检查点。
# 输入：
#   path：同一局部目标的已授权米制折线。
#   current_position_m：本次实测世界系位置。
# 输出：
#   deduplicated：从当前位置连接至局部终点的去重折线。
def _remaining_authorized_polyline(
    *,
    path: list[Vector3],
    current_position_m: Vector3,
) -> list[Vector3]:
    if len(path) < 2:
        return [current_position_m, *path]
    current = (
        current_position_m.x,
        current_position_m.y,
        current_position_m.z,
    )
    best_distance_squared = math.inf
    best_progress_m = 0.0
    best_segment_index = 0
    best_projection = path[0]
    cumulative_m = 0.0
    for segment_index, (start, end) in enumerate(zip(path, path[1:], strict=False)):
        delta = (end.x - start.x, end.y - start.y, end.z - start.z)
        segment_length = math.sqrt(sum(component * component for component in delta))
        if segment_length <= 1e-12:
            continue
        ratio = sum(
            (current[index] - coordinate) * delta[index]
            for index, coordinate in enumerate((start.x, start.y, start.z))
        ) / (segment_length * segment_length)
        ratio = min(1.0, max(0.0, ratio))
        projection = Vector3(
            x=start.x + delta[0] * ratio,
            y=start.y + delta[1] * ratio,
            z=start.z + delta[2] * ratio,
        )
        distance_squared = sum(
            (current[index] - coordinate) ** 2
            for index, coordinate in enumerate((projection.x, projection.y, projection.z))
        )
        progress_m = cumulative_m + segment_length * ratio
        if distance_squared < best_distance_squared - 1e-12 or (
            abs(distance_squared - best_distance_squared) <= 1e-12 and progress_m > best_progress_m
        ):
            best_distance_squared = distance_squared
            best_progress_m = progress_m
            best_segment_index = segment_index
            best_projection = projection
        cumulative_m += segment_length

    trimmed = [current_position_m]
    if math.dist(current, tuple(best_projection.model_dump().values())) > 1e-9:
        trimmed.append(best_projection)
    trimmed.extend(path[best_segment_index + 1 :])
    deduplicated = [trimmed[0]]
    for point in trimmed[1:]:
        if (
            math.dist(
                tuple(deduplicated[-1].model_dump().values()),
                tuple(point.model_dump().values()),
            )
            > 1e-9
        ):
            deduplicated.append(point)
    return deduplicated


# 功能：
#   计算折线弧长，仅用于比较具有相同终点授权的候选，不赋予路径运动权限。
# 输入：
#   path：世界系米制路径点。
# 输出：
#   length_m：各相邻点距离之和，单位米。
def _polyline_length(path: list[Vector3]) -> float:
    length_m = sum(
        math.dist(
            tuple(start.model_dump().values()),
            tuple(end.model_dump().values()),
        )
        for start, end in zip(path, path[1:], strict=False)
    )
    return length_m


# 功能：
#   在局部已重验折线上选择有限前视目标，防止经过起点后被旧点拉回；实际运动仍由安全层复核。
# 输入：
#   path：同一局部语义目标对应的路径点。
#   current_position_m：当前实测位置，单位米。
#   fallback_target_m：没有路径点时的显式回退目标。
#   resolution_m：地图分辨率，单位米。
#   maximum_step_m：前视及输出距离上限，单位米。
# 输出：
#   target：距离受限的控制参考点。
def _bounded_path_controller_target(
    *,
    path: tuple[Vector3, ...] | list[Vector3],
    current_position_m: Vector3,
    fallback_target_m: Vector3,
    resolution_m: float,
    maximum_step_m: float,
) -> Vector3:
    _finite_configuration(resolution_m=resolution_m, maximum_step_m=maximum_step_m)
    if resolution_m <= 0.0 or maximum_step_m <= 0.0:
        raise ValueError("controller lookahead bounds must be positive")
    points = list(path)
    if not points:
        points = [fallback_target_m]

    current = (
        current_position_m.x,
        current_position_m.y,
        current_position_m.z,
    )
    next_path_point = points[-1]
    if len(points) >= 2:
        # Project the live vehicle position onto the authorized polyline and
        # advance from that projection.  Scanning from path[0] for the first
        # distant point is unsafe here: once the vehicle has moved far enough,
        # the now-behind start point becomes "distant" again and pulls the
        # controller backwards on every other update.
        best_distance_squared = math.inf
        best_progress_m = 0.0
        cumulative_m = 0.0
        segments: list[tuple[Vector3, Vector3, float, float]] = []
        for start, end in zip(points, points[1:], strict=False):
            delta = (end.x - start.x, end.y - start.y, end.z - start.z)
            segment_length = math.sqrt(sum(component * component for component in delta))
            if segment_length <= 1e-12:
                continue
            projection = sum(
                (current[index] - coordinate) * delta[index]
                for index, coordinate in enumerate((start.x, start.y, start.z))
            ) / (segment_length * segment_length)
            projection = min(1.0, max(0.0, projection))
            projected = (
                start.x + delta[0] * projection,
                start.y + delta[1] * projection,
                start.z + delta[2] * projection,
            )
            distance_squared = sum((current[index] - projected[index]) ** 2 for index in range(3))
            progress_m = cumulative_m + segment_length * projection
            if distance_squared < best_distance_squared - 1e-12 or (
                abs(distance_squared - best_distance_squared) <= 1e-12
                and progress_m > best_progress_m
            ):
                best_distance_squared = distance_squared
                best_progress_m = progress_m
            segments.append((start, end, cumulative_m, segment_length))
            cumulative_m += segment_length

        # The complete polyline was already revalidated against the latest
        # metric map and dynamic-obstacle blocks. Constraining the controller
        # target to one voxel therefore adds no collision proof; it only caps
        # the local planner's desired speed to roughly one voxel per second.
        # Permit the coordinator's separately bounded multi-voxel lookahead.
        # Sensor-rate predictive safety still evaluates the actual velocity
        # over its three-second horizon before any PX4 command is published.
        target_progress_m = min(cumulative_m, best_progress_m + maximum_step_m)
        for start, end, segment_start_m, segment_length in segments:
            segment_end_m = segment_start_m + segment_length
            if target_progress_m <= segment_end_m + 1e-12:
                ratio = min(
                    1.0,
                    max(0.0, (target_progress_m - segment_start_m) / segment_length),
                )
                next_path_point = Vector3(
                    x=start.x + (end.x - start.x) * ratio,
                    y=start.y + (end.y - start.y) * ratio,
                    z=start.z + (end.z - start.z) * ratio,
                )
                break

    distance = math.dist(
        current,
        (next_path_point.x, next_path_point.y, next_path_point.z),
    )
    if distance <= maximum_step_m or distance <= 1e-9:
        return next_path_point
    ratio = maximum_step_m / distance
    return Vector3(
        x=current_position_m.x + (next_path_point.x - current_position_m.x) * ratio,
        y=current_position_m.y + (next_path_point.y - current_position_m.y) * ratio,
        z=current_position_m.z + (next_path_point.z - current_position_m.z) * ratio,
    )


# 功能：
#   保留模型选择的局部终点，依据最新米制观测和动态占用重验或重建路径，不能执行旧几何。
# 输入：
#   world：本次重验独立的地图快照。
#   frame：本次冻结的定位和动态障碍帧。
#   selected：原模型从授权集合中选择的候选。
#   required_clearance_m：路径净空下限，单位米。
#   vehicle_radius_m：机体水平半径，单位米。
#   vehicle_height_m：机体高度，单位米。
#   maximum_controller_step_m：控制前视距离上限，单位米。
#   preferred_path：用于减少路径抖动的已有同终点路径。
#   planning_deadline_monotonic_seconds：重新寻路的单调时钟截止时间。
# 输出：
#   result：当前可用路径及其摘要，或明确失败原因。
def _revalidate_selected_navigation_path(
    *,
    world: MetricVoxelMap,
    frame: OnboardPerceptionFrame,
    selected: dict[str, object],
    required_clearance_m: float,
    vehicle_radius_m: float,
    vehicle_height_m: float,
    maximum_controller_step_m: float,
    preferred_path: tuple[Vector3, ...] = (),
    planning_deadline_monotonic_seconds: float | None = None,
) -> _PathRevalidationResult:
    endpoint = Vector3.model_validate(selected["endpoint_m"])
    try:
        submitted_path = [Vector3.model_validate(point) for point in selected["path_points_m"]]
    except (KeyError, TypeError, ValueError):
        return _PathRevalidationResult(
            failure_reason="AUTHORIZED_ENDPOINT_NO_LONGER_COLLISION_FREE"
        )
    if (
        len(submitted_path) < 2
        or math.dist(
            tuple(submitted_path[-1].model_dump().values()),
            tuple(endpoint.model_dump().values()),
        )
        > world.resolution_m
    ):
        return _PathRevalidationResult(
            failure_reason="AUTHORIZED_ENDPOINT_NO_LONGER_COLLISION_FREE"
        )
    temporary_blocked_keys = world.dynamic_obstacle_blocked_keys(
        dynamic_obstacles=frame.dynamic_obstacles,
        required_clearance_m=required_clearance_m,
        vehicle_radius_m=vehicle_radius_m,
        vehicle_height_m=vehicle_height_m,
    )
    continuity_path_retained = False
    submitted_remaining = _remaining_authorized_polyline(
        path=submitted_path,
        current_position_m=frame.localization_position_m,
    )
    regenerated_path: list[Vector3]
    if (
        len(preferred_path) >= 2
        and math.dist(
            tuple(preferred_path[-1].model_dump().values()),
            tuple(endpoint.model_dump().values()),
        )
        <= 1e-6
    ):
        preferred_remaining = _remaining_authorized_polyline(
            path=list(preferred_path),
            current_position_m=frame.localization_position_m,
        )
        preferred_path_valid = world.path_clearance_valid(
            preferred_remaining,
            required_clearance_m=required_clearance_m,
            temporary_blocked_keys=temporary_blocked_keys,
            allow_first_position_as_start=True,
        )
        # Continuity prevents controller chatter, but it must not pin a
        # vehicle near a waypoint to an old voxel detour after the model has
        # selected a newly revalidated, materially shorter path to the exact
        # same endpoint.  Half a voxel absorbs harmless discretization noise;
        # a larger excess is real detour length and can otherwise make a
        # strict waypoint settle gate unreachable before its deadline.
        preferred_not_materially_longer = (
            _polyline_length(preferred_remaining)
            <= _polyline_length(submitted_remaining) + world.resolution_m * 0.5
        )
        if preferred_path_valid and preferred_not_materially_longer:
            regenerated_path = preferred_remaining
            continuity_path_retained = True
        else:
            regenerated_path = submitted_remaining
    else:
        regenerated_path = submitted_remaining
    if not world.path_clearance_valid(
        regenerated_path,
        required_clearance_m=required_clearance_m,
        temporary_blocked_keys=temporary_blocked_keys,
        allow_first_position_as_start=True,
    ):
        # The model call is intentionally slower than depth fusion. A path that
        # was safe in the hash-bound snapshot can therefore be stale by the
        # time its selected endpoint reaches the controller. Preserve the
        # model's endpoint authority, but regenerate the executable geometry
        # from the newest metric map. This is the same deterministic planner
        # that authored the candidate; if no current path exists we still fail
        # closed instead of executing stale geometry.
        try:
            regenerated_path = world.plan_path(
                start_m=frame.localization_position_m,
                goal_m=endpoint,
                required_clearance_m=required_clearance_m,
                temporary_blocked_keys=temporary_blocked_keys,
                allow_current_position_as_start=True,
                deadline_monotonic_seconds=planning_deadline_monotonic_seconds,
            )
        except ValueError as error:
            return _PathRevalidationResult(
                failure_reason=(
                    "AUTHORIZED_ENDPOINT_REVALIDATION_DEADLINE"
                    if "deadline exceeded" in str(error)
                    else "AUTHORIZED_ENDPOINT_NO_LONGER_COLLISION_FREE"
                )
            )
        continuity_path_retained = False
    path_sha256 = sha256_json([point.model_dump(mode="json") for point in regenerated_path])
    current = frame.localization_position_m
    controller_target = _bounded_path_controller_target(
        path=regenerated_path[1:],
        current_position_m=current,
        fallback_target_m=endpoint,
        resolution_m=world.resolution_m,
        maximum_step_m=maximum_controller_step_m,
    )
    return _PathRevalidationResult(
        path=tuple(regenerated_path),
        path_sha256=path_sha256,
        controller_target_m=controller_target,
        path_continuity_retained=continuity_path_retained,
    )


@dataclass(frozen=True)
class _PendingPathRevalidation:
    sequence: int
    trigger: str
    snapshot_sha256: str
    goal_position_m: Vector3
    navigation_goal_id: str | None
    submitted_at_unix_ms: int
    maximum_decision_age_seconds: float
    model_action: str
    model_call_id: str | None
    source_expert: str
    selected_candidate_id: str
    controller_step_scale: float
    pilot_control: NormalizedPilotControl | None
    future: Future[_PathRevalidationResult]


@dataclass(frozen=True)
class NavigationControlDirective:
    """One bounded controller target and its causal model-lease evidence."""

    target_m: Vector3
    model_navigation_authorized: bool
    reason: str
    model_call_id: str | None = None
    selected_candidate_id: str | None = None
    path_sha256: str | None = None
    navigation_snapshot_sha256: str | None = None
    valid_until_unix_ms: int = 0
    navigation_goal_id: str | None = None
    controller_step_scale: float = 1.0
    source_expert: str | None = None
    pilot_control: NormalizedPilotControl | None = None


class EventDrivenIndoorNavigationCoordinator:
    """Keep model navigation off the sensor-rate safety thread.

    A single model request may be outstanding.  The submitted snapshot is
    immutable and hash-bound.  A legacy package's selected endpoint is
    replanned against the latest metric map; a continuous package's four axes
    are converted to vehicle-bounded physical velocities and revalidated on
    the sensor-rate path.  Provider latency or failure can never block depth
    ingestion, collision checks, or command lease refreshes.
    """

    _NAVIGATION_LIVE_EVIDENCE_RADIUS_M = 10.0
    _MAXIMUM_CONTROLLER_LOOKAHEAD_VOXELS = 4.0

    # 功能：
    #   建立单所有者异步协调器，将候选兼容模式与短租期连续四轴模式分开，禁止延迟扩大控制权。
    # 输入：
    #   fusion：持有感知帧和地图的融合器。
    #   port：当前选择的本地策略或模型端口。
    #   required_clearance_m：所需净空，单位米。
    #   candidate_speed_mps：候选评估速度，单位米每秒。
    #   vehicle_radius_m：机体水平半径，单位米。
    #   vehicle_height_m：机体高度，单位米。
    #   maximum_decision_age_seconds：决策最老允许年龄，连续模式另有更短硬上限。
    #   control_lease_seconds：控制租期，不能延长连续模式的原始源期限。
    #   maximum_controller_step_m：候选参考点最大前视距离，单位米。
    #   maximum_revalidation_seconds：路径重验计算预算，单位秒。
    #   maximum_snapshot_planning_seconds：输入快照规划预算，单位秒。
    #   model_worker_cpu_ids：可选工作线程 CPU 编号。
    #   control_output_mode：候选选择或连续机体系控制模式。
    #   include_candidate_paths：是否附带候选路径，缺省按模式选择。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        fusion: RuntimePerceptionFusion,
        port: StructuredModelPort,
        required_clearance_m: float,
        candidate_speed_mps: float = 0.5,
        vehicle_radius_m: float = 0.38,
        vehicle_height_m: float = 0.43,
        maximum_decision_age_seconds: float = 12.0,
        control_lease_seconds: float = 6.0,
        maximum_controller_step_m: float | None = None,
        maximum_revalidation_seconds: float = 0.5,
        maximum_snapshot_planning_seconds: float = 2.0,
        model_worker_cpu_ids: tuple[int, ...] | None = None,
        control_output_mode: str = "legacy-candidate-selection",
        include_candidate_paths: bool | None = None,
    ) -> None:
        _finite_configuration(
            required_clearance_m=required_clearance_m,
            candidate_speed_mps=candidate_speed_mps,
            vehicle_radius_m=vehicle_radius_m,
            vehicle_height_m=vehicle_height_m,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            control_lease_seconds=control_lease_seconds,
            maximum_revalidation_seconds=maximum_revalidation_seconds,
            maximum_snapshot_planning_seconds=maximum_snapshot_planning_seconds,
        )
        if required_clearance_m < 0.0:
            raise ValueError("required clearance must be non-negative")
        if min(candidate_speed_mps, vehicle_radius_m, vehicle_height_m) <= 0:
            raise ValueError("navigation speed and vehicle dimensions must be positive")
        if maximum_decision_age_seconds <= 0.0 or control_lease_seconds <= 0.0:
            raise ValueError("navigation decision ages must be positive")
        if maximum_revalidation_seconds <= 0.0:
            raise ValueError("navigation revalidation deadline must be positive")
        if maximum_snapshot_planning_seconds <= 0.0:
            raise ValueError("navigation snapshot planning deadline must be positive")
        if control_output_mode not in {
            "legacy-candidate-selection",
            "normalized-body-velocity",
        }:
            raise ValueError("unsupported local navigation output mode")
        if include_candidate_paths is not None and type(include_candidate_paths) is not bool:
            raise ValueError("include_candidate_paths must be boolean")
        if maximum_controller_step_m is not None:
            _finite_configuration(maximum_controller_step_m=maximum_controller_step_m)
        if model_worker_cpu_ids is None:
            model_worker_cpu_ids = ()
        if (
            type(model_worker_cpu_ids) not in (tuple, list)
            or len(model_worker_cpu_ids) > 256
            or any(type(cpu_id) is not int or cpu_id < 0 for cpu_id in model_worker_cpu_ids)
        ):
            raise ValueError("model worker CPU IDs must be non-negative")
        normalized_cpu_ids = tuple(sorted(set(model_worker_cpu_ids)))
        if len(normalized_cpu_ids) != len(model_worker_cpu_ids):
            raise ValueError("model worker CPU IDs must be unique")
        self.fusion = fusion
        self.port = port
        self.required_clearance_m = required_clearance_m
        self.candidate_speed_mps = candidate_speed_mps
        self.vehicle_radius_m = vehicle_radius_m
        self.vehicle_height_m = vehicle_height_m
        if control_output_mode == CONTINUOUS_CONTROL_MODE:
            maximum_decision_age_seconds = min(
                maximum_decision_age_seconds,
                LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
            )
            control_lease_seconds = min(control_lease_seconds, LOCAL_CONTROL_MAXIMUM_AGE_SECONDS)
        self.maximum_decision_age_seconds = maximum_decision_age_seconds
        self.control_lease_seconds = control_lease_seconds
        self.maximum_revalidation_seconds = maximum_revalidation_seconds
        self.maximum_snapshot_planning_seconds = maximum_snapshot_planning_seconds
        self.control_output_mode = control_output_mode
        self.include_candidate_paths = (
            control_output_mode == "legacy-candidate-selection"
            if include_candidate_paths is None
            else include_candidate_paths
        )
        self.model_worker_cpu_ids = normalized_cpu_ids
        self.maximum_controller_step_m = (
            min(0.12, self.fusion.world.resolution_m * 0.45)
            if maximum_controller_step_m is None
            else maximum_controller_step_m
        )
        maximum_lookahead_m = (
            self.fusion.world.resolution_m * self._MAXIMUM_CONTROLLER_LOOKAHEAD_VOXELS
        )
        if not 0.0 < self.maximum_controller_step_m <= maximum_lookahead_m:
            raise ValueError(
                "model navigation controller step is outside the revalidated lookahead bound"
            )
        self._executor_generation = 0
        self._closed = False
        self._executor = self._new_executor()
        self._sequence = 0
        self._feature_deadline_latch = SourceDeadlineLatch()
        self._image_deadline_latch = SourceDeadlineLatch()
        self._pending: _PendingNavigationCycle | None = None
        self._expired_local_future: Future[_NavigationWorkerResult] | None = None
        self._pending_revalidation: _PendingPathRevalidation | None = None
        self._active_path: tuple[Vector3, ...] = ()
        self._active_hold_until_unix_ms = 0
        self._active_path_until_unix_ms = 0
        self._active_issued_at_unix_ms = 0
        self._active_model_call_id: str | None = None
        self._active_source_expert: str | None = None
        self._active_selected_candidate_id: str | None = None
        self._active_path_sha256: str | None = None
        self._active_navigation_goal_id: str | None = None
        self._active_controller_step_scale = 1.0
        self._active_pilot_control: NormalizedPilotControl | None = None
        self._active_navigation_snapshot_sha256: str | None = None
        self._model_call_records: deque[Any] = deque()
        self._last_submitted_snapshot: dict[str, object] | None = None

    # 功能：
    #   判断请求、路径重验或隔离中的过期推理是否占用工作槽，避免有状态推理并发。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   pending：工作槽仍被占用时为 True。
    @property
    def request_pending(self) -> bool:
        return (
            self._pending is not None
            or self._pending_revalidation is not None
            or self._expired_local_future is not None
            and not self._expired_local_future.done()
        )

    # 功能：
    #   指示仍在运行的连续控制请求，不把旧隔离工作或兼容寻路算作新控制请求。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   pending：存在未完成连续控制请求时为 True。
    @property
    def continuous_request_pending(self) -> bool:
        return (
            self.control_output_mode == CONTINUOUS_CONTROL_MODE
            and self._pending is not None
            and not self._pending.future.done()
        )

    # 功能：
    #   仅为可能有效的连续决策提示优先消费，不让已知失败结果抢占新感知处理时间。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   ready：已有待核验连续结果时为 True。
    @property
    def continuous_result_ready(self) -> bool:
        pending = self._pending
        if (
            self.control_output_mode != CONTINUOUS_CONTROL_MODE
            or pending is None
            or not pending.future.done()
        ):
            return False
        try:
            result = pending.future.result()
        except Exception:
            return False
        return (
            isinstance(result, _NavigationWorkerResult)
            and result.failure_reason is None
            and result.model_result is not None
        )

    # 功能：
    #   计算在途连续请求的剩余原始预算，只供调度参考，不随新观测续租。
    # 输入：
    #   now_unix_ms：当前 Unix 毫秒时钟。
    # 输出：
    #   remaining_ms：剩余毫秒数，没有适用请求时为零。
    def continuous_request_remaining_ms(self, *, now_unix_ms: int) -> int:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("CONTROL_DEADLINE_CLOCK_INVALID")
        pending = self._pending
        if (
            self.control_output_mode != CONTINUOUS_CONTROL_MODE
            or pending is None
            or now_unix_ms < pending.submitted_at_unix_ms
        ):
            return 0
        deadline = pending.submitted_at_unix_ms + round(pending.maximum_decision_age_seconds * 1000)
        return max(0, deadline - now_unix_ms)

    # 功能：
    #   创建单工作线程的执行器，亲和性仅在该线程初始化时设置。
    # 输入：
    #   self：包含模式与 CPU 配置的协调器。
    # 输出：
    #   executor：本次工作代次的执行器。
    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(f"dronedream-local-navigation-model-{self._executor_generation}"),
            initializer=(_set_current_thread_cpu_affinity if self.model_worker_cpu_ids else None),
            initargs=(self.model_worker_cpu_ids,) if self.model_worker_cpu_ids else (),
        )

    # 功能：
    #   隔离超时工作并按需复位候选传输；不能杀死已运行线程，旧任务结束前不并发启动新工作。
    # 输入：
    #   future：超时的推理或路径重验工作。
    #   reset_transport：是否要求候选模型端口更换失效连接。
    # 输出：
    #   None：不返回业务数据。
    def _quarantine_timed_out_work(self, future: Future, *, reset_transport: bool) -> None:
        future.cancel()
        self._expired_local_future = future
        if reset_transport:
            reset = getattr(self.port, "reset_transport", None)
            if callable(reset):
                reset()

    # 功能：
    #   连续模式保持源数据硬期限；候选规划模式另计寻路与供应商响应预算，后者仍需重验路径。
    # 输入：
    #   self：当前控制模式及端口配置。
    # 输出：
    #   limit_seconds：下一请求的最大允许等待秒数。
    def _next_invocation_age_limit_seconds(self) -> float:
        if self.control_output_mode == CONTINUOUS_CONTROL_MODE:
            return self.maximum_decision_age_seconds
        provider_timeout = getattr(self.port, "invocation_timeout_seconds", None)
        if type(provider_timeout) in (int, float) and 0.0 < provider_timeout < math.inf:
            # The immutable navigation snapshot may spend its separately
            # bounded planning budget generating a safe local candidate before
            # the provider is invoked. Account for that work explicitly rather
            # than misclassifying millisecond local inference as stale after a
            # legitimate two-second 3-D search. The selected endpoint still
            # undergoes a second, latest-map revalidation before any lease is
            # established, so this allowance never bypasses live safety.
            return max(
                self.maximum_decision_age_seconds,
                self.maximum_snapshot_planning_seconds + float(provider_timeout) + 1.0,
            )
        return self.maximum_decision_age_seconds

    # 功能：
    #   为没有调用模型的健康失败保留当前证据，不为不健康帧执行昂贵候选寻路。
    # 输入：
    #   health：当前失败健康判定。
    #   goal_position_m：本次目标位置，单位米。
    #   strategic_context：待限制范围的任务上下文。
    # 输出：
    #   snapshot_sha256：已保存失败快照的摘要，无法构造时为 None。
    def _record_non_invoked_snapshot(
        self,
        *,
        health: PerceptionFusionHealth,
        goal_position_m: Vector3,
        strategic_context: Mapping[str, object] | None,
    ) -> str | None:
        frame = self.fusion.latest_frame
        if frame is None:
            return None
        try:
            bounded_strategic_context = _bounded_strategic_navigation_context(strategic_context)
            snapshot = self.fusion.world.text_navigation_snapshot(
                current_position_m=frame.localization_position_m,
                current_velocity_mps=frame.localization_velocity_mps,
                goal_position_m=goal_position_m,
                dynamic_obstacles=frame.dynamic_obstacles,
                required_clearance_m=self.required_clearance_m,
                candidate_speed_mps=self.candidate_speed_mps,
                vehicle_radius_m=self.vehicle_radius_m,
                vehicle_height_m=self.vehicle_height_m,
                include_candidate_paths=False,
            )
        except ValueError:
            return None
        snapshot["perception_health"] = health.model_dump(mode="json")
        multimodal_sensor_snapshot = self.fusion.multimodal_sensor_snapshot()
        if multimodal_sensor_snapshot is not None:
            snapshot["multimodal_sensor_snapshot"] = multimodal_sensor_snapshot.model_dump(
                mode="json"
            )
        if bounded_strategic_context:
            snapshot["strategic_context"] = bounded_strategic_context
            _bind_navigation_control_output_contract(snapshot, bounded_strategic_context)
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        self._last_submitted_snapshot = dict(snapshot)
        return str(snapshot["snapshot_sha256"])

    # 功能：
    #   冻结本次输入并申请唯一后台工作槽；原始图像/编码时效不足或感知不健康时只生成拒绝回执。
    # 输入：
    #   goal_position_m：当前宏观目标位置，单位米。
    #   navigation_goal_id：语义目标代次，区别同坐标的不同往返阶段。
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   trigger：调度触发原因。
    #   context_id：可选模型上下文身份。
    #   multimodal：最多四张待绑定的摄像头图像。
    #   strategic_context：云端宏观任务上下文。
    #   realtime_feature_snapshot：本地编码器的实时特征快照。
    # 输出：
    #   receipt：立即拒绝时的回执；已经提交或暂忙时为 None。
    def schedule(
        self,
        *,
        goal_position_m: Vector3,
        navigation_goal_id: str | None = None,
        now_unix_ms: int,
        trigger: str,
        context_id: str | None = None,
        multimodal: list[dict[str, object]] | None = None,
        strategic_context: Mapping[str, object] | None = None,
        realtime_feature_snapshot: Mapping[str, object] | None = None,
    ) -> IndoorNavigationCycleReceipt | None:
        preparation_started_monotonic = time.monotonic()
        if self._closed:
            raise ValueError("NAVIGATION_COORDINATOR_CLOSED")
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("CONTROL_DEADLINE_CLOCK_INVALID")
        if trigger not in {
            "initial",
            "periodic",
            "goal-amended",
            "progress-stalled",
            "dynamic-obstacle",
            "new-map-evidence",
        }:
            raise ValueError("NAVIGATION_TRIGGER_INVALID")
        if navigation_goal_id is not None and (
            type(navigation_goal_id) is not str or not 1 <= len(navigation_goal_id) <= 160
        ):
            raise ValueError("NAVIGATION_GOAL_ID_INVALID")
        if self.request_pending:
            return None
        # Freeze caller-owned inputs before either the deadline check or the
        # asynchronous compilation. A shallow mapping copy still shares its
        # encoding lists; a borrowed goal could change both a pending request
        # and its later goal-revision check without creating a new request.
        goal_position_m = Vector3.model_validate(
            goal_position_m.model_dump(mode="python"), strict=True
        )
        realtime_feature_snapshot = (
            copy.deepcopy(dict(realtime_feature_snapshot))
            if realtime_feature_snapshot is not None
            else None
        )
        # A timed-out local job is quarantined until it finishes. Its result
        # is never consumed, and it must not race a new call on the same ONNX
        # backend's temporal buffers. No unbounded thread replacement.
        self._collect_expired_call()
        self._sequence += 1
        health = self.fusion.health(now_unix_ms=now_unix_ms)
        if len(self._model_call_records) >= 64:
            # 消费方未取走证据时暂停新推理，不覆盖旧回执，也不让证据队列无界增长。
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="MODEL_EVIDENCE_BACKPRESSURE",
            )
        age_limit = self._next_invocation_age_limit_seconds()
        if self.control_output_mode == CONTINUOUS_CONTROL_MODE:
            try:
                features = RealtimeFeatureSnapshot.model_validate(realtime_feature_snapshot)
                if not features.fresh_at(now_unix_ms):
                    raise ValueError("continuous features are stale or not ready")
                if features.policy_feature_contract_sha256() is None:
                    raise ValueError("continuous feature contract is incomplete")
                source_deadline = features.control_deadline_unix_ms(now_unix_ms=now_unix_ms)
                source_deadline = self._feature_deadline_latch.restrict(
                    source_unix_ms=min(
                        e.observed_at_unix_ms
                        for e in features.encodings
                        if e.encoder_role in features.required_roles
                    ),
                    deadline_unix_ms=source_deadline,
                )
                age_limit = min(age_limit, (source_deadline - now_unix_ms) / 1000)
                if age_limit <= LOCAL_DISPATCH_RESERVE_MS / 1000:
                    raise ValueError("continuous source deadline too close for publication")
            except (ValueError, TypeError):
                return IndoorNavigationCycleReceipt(
                    sequence=self._sequence,
                    trigger=trigger,
                    perception_health=health,
                    model_action="not-invoked",
                    hold_reason="CONTINUOUS_CONTROL_SOURCE_NOT_READY",
                )
        if not health.stream_healthy:
            snapshot_sha256 = self._record_non_invoked_snapshot(
                health=health,
                goal_position_m=goal_position_m,
                strategic_context=strategic_context,
            )
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                snapshot_sha256=snapshot_sha256,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="PERCEPTION_STREAM_NOT_HEALTHY",
            )
        frame = self.fusion.frozen_frame
        if frame is None:
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="PERCEPTION_FRAME_NOT_RECEIVED",
            )
        try:
            bounded_strategic_context = _bounded_strategic_navigation_context(strategic_context)
        except ValueError:
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="STRATEGIC_NAVIGATION_CONTEXT_INVALID",
            )
        # The coordinator's configured mode is authoritative. Context may
        # describe it, never silently override it or leave a continuous model
        # with the candidate-only instruction set inherited from an old caller.
        task_context = bounded_strategic_context.get("task", {})
        if (
            not isinstance(task_context, dict)
            or task_context.get(
                "local_navigation_output_mode",
                self.control_output_mode,
            )
            != self.control_output_mode
        ):
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="MODEL_NAVIGATION_CONTEXT_MODE_MISMATCH",
            )
        bounded_strategic_context["task"] = {
            **task_context,
            "local_navigation_output_mode": self.control_output_mode,
        }
        if multimodal is not None and (
            type(multimodal) is not list
            or len(multimodal) > 4
            or any(type(item) is not dict for item in multimodal)
        ):
            raise ValueError("VISUAL_NAVIGATION_MEDIA_INVALID")
        media = copy.deepcopy(multimodal) if multimodal is not None else []
        if self.control_output_mode == CONTINUOUS_CONTROL_MODE and media:
            image_deadline = min(
                image_control_deadline(item, now_unix_ms=now_unix_ms) for item in media
            )
            if image_deadline:
                image_deadline = self._image_deadline_latch.restrict(
                    source_unix_ms=min(item["scene_source_unix_ns"] // 1_000_000 for item in media),
                    deadline_unix_ms=image_deadline,
                )
            age_limit = min(age_limit, (image_deadline - now_unix_ms) / 1000)
            if age_limit <= LOCAL_DISPATCH_RESERVE_MS / 1000:
                return IndoorNavigationCycleReceipt(
                    sequence=self._sequence,
                    trigger=trigger,
                    perception_health=health,
                    model_action="not-invoked",
                    hold_reason="VISUAL_CONTROL_SOURCE_NOT_READY",
                )
        multimodal_sensor_snapshot = self.fusion.multimodal_sensor_snapshot()
        visual_evidence: list[dict[str, object]] = []
        if media:
            for item in media:
                embedded = item.get("content_bytes")
                embedded_available = isinstance(embedded, bytes) and bool(embedded)
                # Embedded, hash-bound camera evidence need not touch a mounted
                # filesystem. Only file-backed transports resolve/stat a path.
                path = Path(str(item.get("path", "")))
                if not embedded_available:
                    path = path.resolve()
                if item.get("kind") != "image-file" or (
                    not embedded_available and not path.is_file()
                ):
                    return IndoorNavigationCycleReceipt(
                        sequence=self._sequence,
                        trigger=trigger,
                        perception_health=health,
                        model_action="not-invoked",
                        hold_reason="VISUAL_NAVIGATION_EVIDENCE_UNAVAILABLE",
                    )
                visual_evidence.append(
                    {
                        "kind": "forward-rgb-camera",
                        "sha256": self._validated_visual_sha256(item, path=path),
                        "content_type": str(item.get("content_type", "image/png")),
                        **(
                            {
                                "source_observed_at_unix_ms": item["scene_source_unix_ns"]
                                // 1_000_000,
                                "control_deadline_unix_ms": image_deadline,
                                "timestamp_basis": item["timestamp_basis"],
                            }
                            if self.control_output_mode == CONTINUOUS_CONTROL_MODE
                            else {}
                        ),
                        **(
                            {"model_rgb_sha256": item["model_rgb_sha256"]}
                            if isinstance(item.get("model_rgb_sha256"), str)
                            else {}
                        ),
                    }
                )
        invocation_deadline = (
            preparation_started_monotonic + age_limit
            if self.control_output_mode == CONTINUOUS_CONTROL_MODE
            else None
        )
        if (
            invocation_deadline is not None
            and time.monotonic() + LOCAL_DISPATCH_RESERVE_MS / 1000 >= invocation_deadline
        ):
            return IndoorNavigationCycleReceipt(
                sequence=self._sequence,
                trigger=trigger,
                perception_health=health,
                model_action="not-invoked",
                hold_reason="CONTROL_SOURCE_EXPIRED_DURING_PREPARATION",
            )
        prime = getattr(self.port, "prime_multimodal", None)
        if callable(prime):
            try:
                prime(copy.deepcopy(media))
            except Exception as error:
                return IndoorNavigationCycleReceipt(
                    sequence=self._sequence,
                    trigger=trigger,
                    perception_health=health,
                    model_action="not-invoked",
                    hold_reason="VISUAL_NAVIGATION_PREPARATION_FAILED",
                    **_failure_diagnostic(error, stage="model-invocation"),
                )
        future = self._executor.submit(
            _compile_and_request_navigation_decision,
            # Static route/map priors remain complete and immutable.  Only
            # mutable depth evidence is frozen to the local reasoning horizon;
            # copying the entire mission history here previously delayed the
            # fail-closed command publisher after long flights.
            world=self.fusion.world.navigation_clone(
                center_m=frame.localization_position_m,
                radius_m=self._NAVIGATION_LIVE_EVIDENCE_RADIUS_M,
            ),
            # Every nested vector and sequence is immutable. The asynchronous
            # compiler may retain this exact validated source across new scans.
            frame=frame,
            health=health,
            port=self.port,
            goal_position_m=goal_position_m,
            required_clearance_m=self.required_clearance_m,
            candidate_speed_mps=self.candidate_speed_mps,
            vehicle_radius_m=self.vehicle_radius_m,
            vehicle_height_m=self.vehicle_height_m,
            context_id=context_id,
            multimodal=media,
            visual_evidence=visual_evidence,
            multimodal_sensor_snapshot=(
                multimodal_sensor_snapshot.model_dump(mode="json")
                if multimodal_sensor_snapshot is not None
                else None
            ),
            realtime_feature_snapshot=(
                dict(realtime_feature_snapshot) if realtime_feature_snapshot is not None else None
            ),
            strategic_context=bounded_strategic_context,
            maximum_snapshot_planning_seconds=(self.maximum_snapshot_planning_seconds),
            include_candidate_paths=(self.include_candidate_paths),
            control_reference_observed_at_unix_ms=now_unix_ms,
            invocation_deadline_monotonic=invocation_deadline,
        )
        self._pending = _PendingNavigationCycle(
            sequence=self._sequence,
            trigger=trigger,
            perception_health=health,
            goal_position_m=goal_position_m,
            navigation_goal_id=navigation_goal_id,
            submitted_at_unix_ms=now_unix_ms,
            maximum_decision_age_seconds=age_limit,
            future=future,
        )
        return None

    # 功能：
    #   将摄像头输入固定为同一份不可变字节，再核对摘要和大小，避免算摘要与推理间重开文件。
    # 输入：
    #   item：本次独立复制的图像描述；校验后补入字节及摘要。
    #   path：没有内嵌字节时的本地图像路径。
    # 输出：
    #   observed_sha256：实际交给模型的图像字节摘要。
    @staticmethod
    def _validated_visual_sha256(
        item: dict[str, object],
        *,
        path: Path,
    ) -> str:
        embedded = item.get("content_bytes")
        if embedded is None:
            with path.open("rb") as stream:
                embedded = stream.read(12 * 1024 * 1024 + 1)
        elif not isinstance(item.get("content_sha256"), str):
            raise ValueError("visual evidence bytes are invalid")
        if type(embedded) is not bytes or not 0 < len(embedded) <= 12 * 1024 * 1024:
            raise ValueError("visual evidence exceeds bounded size")
        observed_sha256 = hashlib.sha256(embedded).hexdigest()
        if any(item[key] != observed_sha256 for key in ("content_sha256", "sha256") if key in item):
            raise ValueError("visual evidence byte hash does not match")
        item["content_bytes"] = embedded
        item["content_sha256"] = observed_sha256
        return observed_sha256

    # 功能：
    #   回收已结束的隔离工作，只提取真实调用回执，绝不重新采用超时动作。
    # 输入：
    #   self：持有唯一隔离工作槽的协调器。
    # 输出：
    #   None：不返回业务数据。
    def _collect_expired_call(self) -> None:
        future = self._expired_local_future
        if future is None or not future.done():
            return
        self._expired_local_future = None
        try:
            result = future.result()
        except Exception:
            return
        if isinstance(result, _NavigationWorkerResult):
            record = result.discarded_call_record or getattr(result.model_result, "record", None)
            if record is not None:
                self._model_call_records.append(copy.deepcopy(record))

    # 功能：
    #   消费已完成模型结果或路径重验，核对当前目标、时效与感知健康后才建立短时控制租约。
    # 输入：
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   current_goal_position_m：当前宏观目标位置，用于识别规划期间的目标修改。
    #   current_navigation_goal_id：当前语义目标代次。
    #   current_perception_health：可选本次传感器循环已经采样的健康状态。
    #   current_frame：可选本次循环的当前定位帧。
    # 输出：
    #   receipt：本次消费的判定回执；还没有可消费结果时为 None。
    def poll(
        self,
        *,
        now_unix_ms: int,
        current_goal_position_m: Vector3 | None = None,
        current_navigation_goal_id: str | None = None,
        current_perception_health: PerceptionFusionHealth | None = None,
        current_frame: OnboardPerceptionFrame | None = None,
    ) -> IndoorNavigationCycleReceipt | None:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("CONTROL_DEADLINE_CLOCK_INVALID")
        self._collect_expired_call()
        if self._closed:
            return None

        # 功能：
        #   使用调用方本循环已采样的健康判定或重新获取，不信任被原地破坏的模型字段。
        # 输入：
        #   current_perception_health：外层提供的可选健康判定。
        #   now_unix_ms：外层本循环的毫秒时钟。
        # 输出：
        #   health：独立且严格验证的当前健康判定。
        def current_health() -> PerceptionFusionHealth:
            if current_perception_health is not None:
                health = PerceptionFusionHealth.model_validate(
                    current_perception_health.model_dump(mode="python"), strict=True
                )
                return health
            return self.fusion.health(now_unix_ms=now_unix_ms)

        revalidation = self._pending_revalidation
        if revalidation is not None:
            if not revalidation.future.done():
                age_seconds = (now_unix_ms - revalidation.submitted_at_unix_ms) / 1_000.0
                if age_seconds < 0 or age_seconds > revalidation.maximum_decision_age_seconds:
                    self._pending_revalidation = None
                    self._quarantine_timed_out_work(revalidation.future, reset_transport=False)
                    return IndoorNavigationCycleReceipt(
                        sequence=revalidation.sequence,
                        trigger=revalidation.trigger,
                        snapshot_sha256=revalidation.snapshot_sha256,
                        perception_health=current_health(),
                        model_action=revalidation.model_action,
                        model_call_id=revalidation.model_call_id,
                        selected_candidate_id=revalidation.selected_candidate_id,
                        hold_reason="MODEL_NAVIGATION_REVALIDATION_TIMEOUT",
                    )
                return None
            self._pending_revalidation = None
            health = current_health()
            base = {
                "sequence": revalidation.sequence,
                "trigger": revalidation.trigger,
                "snapshot_sha256": revalidation.snapshot_sha256,
                "perception_health": health,
                "model_action": revalidation.model_action,
                "model_call_id": revalidation.model_call_id,
                "selected_candidate_id": revalidation.selected_candidate_id,
            }
            if (
                current_goal_position_m is not None
                and current_goal_position_m != revalidation.goal_position_m
                or current_navigation_goal_id is not None
                and current_navigation_goal_id != revalidation.navigation_goal_id
            ):
                return IndoorNavigationCycleReceipt(
                    **base,
                    hold_reason="MODEL_NAVIGATION_GOAL_CHANGED_DURING_CALL",
                )
            age_seconds = (now_unix_ms - revalidation.submitted_at_unix_ms) / 1_000.0
            if age_seconds < 0 or age_seconds > revalidation.maximum_decision_age_seconds:
                return IndoorNavigationCycleReceipt(
                    **base,
                    hold_reason="MODEL_NAVIGATION_DECISION_STALE",
                )
            if not health.stream_healthy:
                return IndoorNavigationCycleReceipt(
                    **base,
                    hold_reason="PERCEPTION_CHANGED_TO_UNHEALTHY_DURING_MODEL_CALL",
                )
            try:
                path_result = revalidation.future.result()
            except Exception:
                path_result = _PathRevalidationResult(
                    failure_reason="AUTHORIZED_ENDPOINT_NO_LONGER_COLLISION_FREE"
                )
            if path_result.failure_reason is not None:
                return IndoorNavigationCycleReceipt(
                    **base,
                    hold_reason=path_result.failure_reason,
                )
            self._active_path = path_result.path
            self._active_issued_at_unix_ms = now_unix_ms
            self._active_path_until_unix_ms = now_unix_ms + round(
                self.control_lease_seconds * 1_000.0
            )
            self._active_model_call_id = revalidation.model_call_id
            self._active_source_expert = revalidation.source_expert
            self._active_selected_candidate_id = revalidation.selected_candidate_id
            self._active_path_sha256 = path_result.path_sha256
            self._active_navigation_goal_id = revalidation.navigation_goal_id
            self._active_controller_step_scale = revalidation.controller_step_scale
            self._active_pilot_control = revalidation.pilot_control
            self._active_navigation_snapshot_sha256 = revalidation.snapshot_sha256
            self._active_hold_until_unix_ms = 0
            return IndoorNavigationCycleReceipt(
                **base,
                selected_metric_path_sha256=path_result.path_sha256,
                controller_target_m=path_result.controller_target_m,
                controller_step_scale=revalidation.controller_step_scale,
                deterministic_metric_path_revalidated=True,
                dynamic_path_revalidated=True,
                path_continuity_retained=path_result.path_continuity_retained,
            )

        pending = self._pending
        if pending is None:
            return None
        if not pending.future.done():
            age_seconds = (now_unix_ms - pending.submitted_at_unix_ms) / 1_000.0
            if 0 <= age_seconds <= pending.maximum_decision_age_seconds:
                return None
            self._pending = None
            self._quarantine_timed_out_work(
                pending.future, reset_transport=self.control_output_mode != CONTINUOUS_CONTROL_MODE
            )
            if self.control_output_mode == CONTINUOUS_CONTROL_MODE:
                self._active_path_until_unix_ms = 0
            return IndoorNavigationCycleReceipt(
                sequence=pending.sequence,
                trigger=pending.trigger,
                perception_health=current_health(),
                model_action="not-invoked",
                hold_reason="MODEL_NAVIGATION_INVOCATION_TIMEOUT",
            )
        self._pending = None
        try:
            worker_result = pending.future.result()
        except Exception as error:
            worker_result = _NavigationWorkerResult(
                snapshot=None,
                failure_reason="MODEL_NAVIGATION_INVOCATION_FAILED",
                failure_diagnostic=_failure_diagnostic(error, stage="worker-result"),
            )
        snapshot = worker_result.snapshot
        if snapshot is not None:
            self._last_submitted_snapshot = dict(snapshot)
        base = {
            "sequence": pending.sequence,
            "trigger": pending.trigger,
            "snapshot_sha256": (str(snapshot["snapshot_sha256"]) if snapshot is not None else None),
            "perception_health": current_health(),
        }
        if worker_result.failure_reason is not None:
            if worker_result.discarded_call_record is not None:
                # A result can lose control authority without erasing the fact
                # that its inference actually ran (and its latency/usage trace).
                self._model_call_records.append(copy.deepcopy(worker_result.discarded_call_record))
            if worker_result.failure_reason == "MODEL_NAVIGATION_INVOCATION_FAILED":
                advance = getattr(self.port, "advance_after_failure", None)
                if callable(advance):
                    advance()
            return IndoorNavigationCycleReceipt(
                **base,
                **(worker_result.failure_diagnostic or {}),
                model_action="not-invoked",
                hold_reason=worker_result.failure_reason,
            )
        result = worker_result.model_result
        selected = worker_result.selected_candidate
        if result is None:
            return IndoorNavigationCycleReceipt(
                **base,
                model_action="not-invoked",
                hold_reason="MODEL_NAVIGATION_INVOCATION_FAILED",
            )
        decision = result.artifact
        call_record = getattr(result, "record", None)
        if call_record is not None:
            self._model_call_records.append(copy.deepcopy(call_record))
        call_id = getattr(call_record, "call_id", None)
        local_expert_trace = getattr(call_record, "local_expert_trace", None)
        training_trace = getattr(call_record, "simulation_training_trace", None)
        source_expert = getattr(
            local_expert_trace or training_trace,
            "selected_navigation_role",
            "local-navigation-policy",
        )
        base.update({"model_action": decision.action, "model_call_id": call_id})
        if (
            decision.action == "pilot-control"
            and self.control_output_mode != CONTINUOUS_CONTROL_MODE
        ) or (
            decision.action == "select-candidate"
            and self.control_output_mode == CONTINUOUS_CONTROL_MODE
        ):
            # A provider/package mismatch must not borrow the other mode's
            # lease or path semantics. Retain the actual call record above.
            self._active_path = ()
            self._active_path_until_unix_ms = 0
            self._active_hold_until_unix_ms = 0
            self._active_pilot_control = None
            return IndoorNavigationCycleReceipt(
                **base,
                hold_reason="MODEL_NAVIGATION_OUTPUT_MODE_MISMATCH",
            )
        if (
            current_goal_position_m is not None
            and current_goal_position_m != pending.goal_position_m
            or current_navigation_goal_id is not None
            and current_navigation_goal_id != pending.navigation_goal_id
        ):
            return IndoorNavigationCycleReceipt(
                **base,
                hold_reason="MODEL_NAVIGATION_GOAL_CHANGED_DURING_CALL",
            )
        age_seconds = (now_unix_ms - pending.submitted_at_unix_ms) / 1_000.0
        if age_seconds < 0 or age_seconds > pending.maximum_decision_age_seconds:
            return IndoorNavigationCycleReceipt(
                **base,
                hold_reason="MODEL_NAVIGATION_DECISION_STALE",
            )
        health = base["perception_health"]
        if not isinstance(health, PerceptionFusionHealth) or not health.stream_healthy:
            return IndoorNavigationCycleReceipt(
                **base,
                hold_reason="PERCEPTION_CHANGED_TO_UNHEALTHY_DURING_MODEL_CALL",
            )
        if selected is None:
            if decision.action == "pilot-control":
                latest_frame = current_frame or self.fusion.latest_frame
                if latest_frame is None:
                    return IndoorNavigationCycleReceipt(
                        **base,
                        hold_reason="PERCEPTION_FRAME_NOT_RECEIVED",
                    )
                # The cloud/global planner owns this coarse route goal.  A
                # continuous local policy needs its direction as a lease and
                # progress reference, not as a sequence of commanded points.
                # Every actual velocity remains subject to the per-tick
                # predictive collision, acceleration and jerk envelope.
                direct_path = (
                    latest_frame.localization_position_m,
                    pending.goal_position_m,
                )
                path_sha256 = sha256_json([point.model_dump(mode="json") for point in direct_path])
                self._active_path = direct_path
                self._active_issued_at_unix_ms = now_unix_ms
                self._active_path_until_unix_ms = min(
                    now_unix_ms + round(self.control_lease_seconds * 1_000.0),
                    pending.submitted_at_unix_ms
                    + round(pending.maximum_decision_age_seconds * 1_000.0),
                )
                self._active_model_call_id = call_id
                self._active_source_expert = source_expert
                self._active_selected_candidate_id = None
                self._active_path_sha256 = path_sha256
                self._active_navigation_goal_id = pending.navigation_goal_id
                self._active_controller_step_scale = decision.controller_step_scale
                self._active_pilot_control = decision.pilot_control
                self._active_navigation_snapshot_sha256 = decision.snapshot_sha256
                self._active_hold_until_unix_ms = 0
                return IndoorNavigationCycleReceipt(
                    **base,
                    selected_metric_path_sha256=path_sha256,
                    controller_target_m=pending.goal_position_m,
                    controller_step_scale=decision.controller_step_scale,
                )
            self._active_path = ()
            self._active_path_until_unix_ms = 0
            self._active_issued_at_unix_ms = now_unix_ms
            # Preserve the causal non-motion proposal without granting a path
            # or axis lease. Hold receipts need identity just as motion does.
            self._active_model_call_id = call_id
            self._active_source_expert = None
            self._active_selected_candidate_id = None
            self._active_path_sha256 = None
            self._active_navigation_goal_id = pending.navigation_goal_id
            self._active_controller_step_scale = 1.0
            self._active_pilot_control = None
            self._active_navigation_snapshot_sha256 = decision.snapshot_sha256
            self._active_hold_until_unix_ms = min(
                now_unix_ms + round(self.control_lease_seconds * 1_000.0),
                pending.submitted_at_unix_ms + round(pending.maximum_decision_age_seconds * 1000),
            )
            return IndoorNavigationCycleReceipt(
                **base,
                hold_reason=(
                    "MODEL_REQUESTED_ABORT"
                    if decision.action == "abort"
                    else "MODEL_DID_NOT_SELECT_AUTHORIZED_PATH"
                ),
            )

        # A renderer may deliver depth more slowly than the worker control rate.
        # Revalidate against the latest admitted occupancy map but bind the path
        # start to the caller's current measured localization when supplied.
        latest_frame = current_frame or self.fusion.latest_frame
        if latest_frame is None:
            return IndoorNavigationCycleReceipt(
                **base,
                selected_candidate_id=str(selected["candidate_id"]),
                hold_reason="PERCEPTION_FRAME_NOT_RECEIVED",
            )
        self._pending_revalidation = _PendingPathRevalidation(
            sequence=pending.sequence,
            trigger=pending.trigger,
            snapshot_sha256=str(snapshot["snapshot_sha256"]),
            goal_position_m=pending.goal_position_m,
            navigation_goal_id=pending.navigation_goal_id,
            submitted_at_unix_ms=pending.submitted_at_unix_ms,
            maximum_decision_age_seconds=pending.maximum_decision_age_seconds,
            model_action=decision.action,
            model_call_id=call_id,
            source_expert=source_expert,
            selected_candidate_id=str(selected["candidate_id"]),
            controller_step_scale=decision.controller_step_scale,
            pilot_control=decision.pilot_control,
            future=self._executor.submit(
                _revalidate_selected_navigation_path,
                world=self.fusion.world.navigation_clone(
                    center_m=latest_frame.localization_position_m,
                    radius_m=self._NAVIGATION_LIVE_EVIDENCE_RADIUS_M,
                ),
                frame=latest_frame.model_copy(deep=True),
                selected=dict(selected),
                required_clearance_m=self.required_clearance_m,
                vehicle_radius_m=self.vehicle_radius_m,
                vehicle_height_m=self.vehicle_height_m,
                maximum_controller_step_m=self.maximum_controller_step_m,
                preferred_path=self._active_path,
                # Candidate generation already performed the expensive search.
                # Revalidation may regenerate geometry from the newest depth
                # map, but that work must remain shorter than the model-path
                # lease.  Without a planner deadline, one pathological 3-D A*
                # search can outlive its future, and repeated executor rotation
                # then leaves multiple CPU-bound searches running concurrently.
                planning_deadline_monotonic_seconds=(
                    time.monotonic()
                    + min(
                        self.maximum_revalidation_seconds,
                        max(
                            0.001,
                            pending.maximum_decision_age_seconds - age_seconds,
                        ),
                    )
                ),
            ),
        )
        return None

    # 功能：
    #   为仍调用候选参考点接口的消费者转发指令目标；此便捷接口不独立授予运动权限。
    # 输入：
    #   current_position_m：当前实际位置，单位米。
    #   fallback_target_m：无模型租约时的显式参考目标。
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   navigation_goal_id：当前语义目标代次。
    #   maximum_step_m：可选的更小前视距离上限。
    # 输出：
    #   target：完整指令所包含的目标位置。
    def controller_target(
        self,
        *,
        current_position_m: Vector3,
        fallback_target_m: Vector3,
        now_unix_ms: int,
        navigation_goal_id: str | None = None,
        maximum_step_m: float | None = None,
    ) -> Vector3:
        target = self.controller_directive(
            current_position_m=current_position_m,
            fallback_target_m=fallback_target_m,
            now_unix_ms=now_unix_ms,
            navigation_goal_id=navigation_goal_id,
            maximum_step_m=maximum_step_m,
        ).target_m
        return target

    # 功能：
    #   按当前时间和语义目标查询模型租约，提供有界参考及四轴意图，关闭或时钟回退时撤销授权。
    # 输入：
    #   current_position_m：当前实际世界系位置，单位米。
    #   fallback_target_m：未授权时保留的宏观参考点。
    #   now_unix_ms：当前 Unix 毫秒时钟。
    #   navigation_goal_id：当前语义目标代次。
    #   maximum_step_m：可选的更小前视距离，单位米。
    # 输出：
    #   directive：包含独立位置/四轴副本及租约证据的控制参考。
    def controller_directive(
        self,
        *,
        current_position_m: Vector3,
        fallback_target_m: Vector3,
        now_unix_ms: int,
        navigation_goal_id: str | None = None,
        maximum_step_m: float | None = None,
    ) -> NavigationControlDirective:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("CONTROL_DEADLINE_CLOCK_INVALID")

        current_position_m = Vector3.model_validate(
            current_position_m.model_dump(mode="python"), strict=True
        )
        fallback_target_m = Vector3.model_validate(
            fallback_target_m.model_dump(mode="python"), strict=True
        )
        requested_step_m = (
            self.maximum_controller_step_m if maximum_step_m is None else maximum_step_m
        )
        _finite_configuration(maximum_step_m=requested_step_m)
        if not 0.0 < requested_step_m <= self.maximum_controller_step_m:
            raise ValueError(
                "controller directive step must be positive and no larger than the "
                "revalidated model-navigation lookahead"
            )

        if (
            self._closed
            or now_unix_ms < self._active_issued_at_unix_ms
            or navigation_goal_id != self._active_navigation_goal_id
        ):
            # A hash-valid path is still unsafe once the semantic goal epoch
            # changes.  Holding at the measured position also covers repeated
            # coordinates on a round trip, where position equality alone
            # cannot distinguish the outbound and return intentions.
            self._active_path = ()
            self._active_hold_until_unix_ms = 0
            self._active_path_until_unix_ms = 0
            self._active_model_call_id = None
            self._active_source_expert = None
            self._active_selected_candidate_id = None
            self._active_path_sha256 = None
            self._active_navigation_goal_id = None
            self._active_controller_step_scale = 1.0
            self._active_pilot_control = None
            self._active_navigation_snapshot_sha256 = None
            return NavigationControlDirective(
                target_m=current_position_m,
                model_navigation_authorized=False,
                reason=(
                    "model-lease-closed"
                    if self._closed
                    else "model-lease-clock-regressed"
                    if now_unix_ms < self._active_issued_at_unix_ms
                    else "model-lease-goal-changed"
                ),
            )
        if now_unix_ms < self._active_hold_until_unix_ms:
            return NavigationControlDirective(
                target_m=current_position_m,
                model_navigation_authorized=False,
                reason="model-requested-hold",
                model_call_id=self._active_model_call_id,
                navigation_snapshot_sha256=self._active_navigation_snapshot_sha256,
                valid_until_unix_ms=self._active_hold_until_unix_ms,
            )
        if now_unix_ms >= self._active_path_until_unix_ms or not self._active_path:
            return NavigationControlDirective(
                target_m=fallback_target_m,
                model_navigation_authorized=False,
                reason=(
                    "model-lease-expired"
                    if self._active_path_until_unix_ms > 0
                    else "model-lease-not-established"
                ),
            )
        effective_step_m = requested_step_m * self._active_controller_step_scale
        target = _bounded_path_controller_target(
            path=self._active_path,
            current_position_m=current_position_m,
            fallback_target_m=fallback_target_m,
            resolution_m=self.fusion.world.resolution_m,
            maximum_step_m=effective_step_m,
        )
        return NavigationControlDirective(
            target_m=target.model_copy(deep=True),
            model_navigation_authorized=True,
            reason="model-path-lease-active",
            model_call_id=self._active_model_call_id,
            selected_candidate_id=self._active_selected_candidate_id,
            path_sha256=self._active_path_sha256,
            navigation_snapshot_sha256=self._active_navigation_snapshot_sha256,
            valid_until_unix_ms=self._active_path_until_unix_ms,
            navigation_goal_id=self._active_navigation_goal_id,
            controller_step_scale=self._active_controller_step_scale,
            source_expert=self._active_source_expert,
            pilot_control=(
                self._active_pilot_control.model_copy(deep=True)
                if self._active_pilot_control is not None
                else None
            ),
        )

    # 功能：
    #   回收已结束的隔离调用并按队列顺序交出回执，关闭后仍允许记录迟到结果，不恢复控制权。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   record：待记录的调用回执或 None。
    def pop_model_call_record(self) -> Any | None:
        self._collect_expired_call()
        record = self._model_call_records.popleft() if self._model_call_records else None
        return record

    # 功能：
    #   交出本次真实模型输入或未调用失败快照，取走后清空，不把健康状态重新标新。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   snapshot：待保存的原始输入快照或 None。
    def pop_submitted_snapshot(self) -> dict[str, object] | None:
        snapshot = self._last_submitted_snapshot
        self._last_submitted_snapshot = None
        return snapshot

    # 功能：
    #   幂等撤销控制租约并取消排队工作，然后关闭端口；已在运行的线程只能等其自身退出。
    # 输入：
    #   self：当前协调器。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active_path = ()
        self._active_path_until_unix_ms = 0
        self._active_hold_until_unix_ms = 0
        self._active_pilot_control = None
        if self._pending is not None:
            self._pending.future.cancel()
            self._expired_local_future = self._pending.future
            self._pending = None
        if self._pending_revalidation is not None:
            self._pending_revalidation.future.cancel()
            self._pending_revalidation = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        close_port = getattr(self.port, "close", None)
        if callable(close_port):
            close_port()
