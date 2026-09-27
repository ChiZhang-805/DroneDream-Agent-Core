"""Causal map registration initialized by measured odometry increments.

The route is deliberately not an estimator input. A previous correction is
carried across measured motion, not across a reset, a clock gap or a new map.
This module estimates means only: repeated frames and retained null directions
do not constitute calibrated covariance or permission to move.
"""

import math
import re
from dataclasses import dataclass

import numpy as np

from .local_map_alignment import _origins, _points
from .local_pose_alignment import MapPoseAlignmentLimits, MapPoseFit, fit_map_pose


class TemporalMapSourceExpiredError(ValueError):
    """An authentic but late observation must not permanently poison a stream."""


@dataclass(frozen=True)
class TemporalMapPoseResult:
    fit: MapPoseFit
    source_timestamp_ns: int
    previous_correction_timestamp_ns: int | None
    history_used: bool
    history_retired_reason: str | None
    failed_temporal_fit: MapPoseFit | None = None
    reinitialization_attempt: MapPoseFit | None = None
    covariance_qualified: bool = False
    motion_permission_granted: bool = False


# 功能：将既有小角度旋转矩阵转换为旋转向量，保留左乘地图修正的方向。
# 输入：rotation：求解器给出的世界坐标修正旋转，受既有十度上限约束。
# 输出：弧度旋转向量；不把欧拉角当作旋转向量。
def _rotation_vector(rotation):
    angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    factor = 0.5 if angle < 1e-8 else angle / (2 * math.sin(angle))
    return skew * factor


class TemporalMapPoseTracker:
    """Single-owner bounded state; input rays must share the native map frame."""

    # 功能：绑定本次地图、坐标变换和源时钟，限制历史使用时长；不加载路线或真值。
    # 输入：地图和坐标绑定摘要、时钟域、最大源时间间隔及原有配准限制。
    # 输出：空的连续配准器，尚无任何已校正位姿。
    def __init__(self, *, map_sha256, binding_sha256, clock_domain,
                 maximum_history_gap_ns=250_000_000, limits=None):
        if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in (map_sha256, binding_sha256)):
            raise ValueError("TEMPORAL_MAP_IDENTITY_INVALID")
        if type(clock_domain) is not str or not 1 <= len(clock_domain) <= 160:
            raise ValueError("TEMPORAL_MAP_CLOCK_INVALID")
        if (type(maximum_history_gap_ns) is not int
                or not 0 < maximum_history_gap_ns <= 250_000_000):
            raise ValueError("TEMPORAL_MAP_GAP_LIMIT_INVALID")
        if limits is not None and not isinstance(limits, MapPoseAlignmentLimits):
            raise ValueError("TEMPORAL_MAP_LIMITS_INVALID")
        self._identity = (map_sha256, binding_sha256, clock_domain)
        self._maximum_gap = maximum_history_gap_ns
        self._limits = limits or MapPoseAlignmentLimits()
        self._last_source = None
        self._reset_counter = None
        self._accepted = None
        self._invalidated = False
        self._native_identity = None
        self._pending_retirement_reason = None

    # 功能：把真实原生消息与同帧深度扫描核对后接入连续配准，拒绝坐标、来源和年龄错配。
    # 输入：原始里程计、深度扫描、标定外参、部署绑定、图像源时间和当前单调时刻。
    # 输出：连续配准候选；原始协方差保持不变，既不输入路线也不读取仿真真值。
    def update_native(self, *, packet, scan, mount, index, binding, binding_sha256,
                      image_timestamp_ns, now_monotonic, map_sha256, clock_domain):
        from .contracts import CalibratedRangeSensorMount, RawMetricRangeScan
        from .hashing import sha256_json
        from .native_odometry_source import SourceOdometry, _integer, _real, source_pose_in_map
        from .sensor_bridge import MetricRangeSensorBridge

        if self._invalidated:
            raise ValueError("TEMPORAL_MAP_TRACKER_INVALIDATED")
        try:
            if (type(packet) is not SourceOdometry or type(scan) is not RawMetricRangeScan
                    or type(mount) is not CalibratedRangeSensorMount):
                raise ValueError("TEMPORAL_MAP_NATIVE_INPUT_INVALID")
            if not 1 <= len(scan.samples) <= 512:
                raise ValueError("TEMPORAL_MAP_NATIVE_SAMPLE_BUDGET")
            scan = RawMetricRangeScan.model_validate(scan.model_dump(), strict=True)
            mount = CalibratedRangeSensorMount.model_validate(mount.model_dump(), strict=True)
            stamp = _integer(image_timestamp_ns, 1, 2**63 - 1)
            now = _real(now_monotonic)
            if stamp % 1000 or abs(packet.timestamp_us - stamp // 1000) > 20_000:
                raise ValueError("TEMPORAL_MAP_NATIVE_SOURCE_SKEW")
            ages = [now - received for received in
                    (packet.received_monotonic_seconds, scan.observed_at_monotonic_seconds)]
            if any(not math.isfinite(age) or age < 0 for age in ages):
                raise ValueError("TEMPORAL_MAP_NATIVE_SOURCE_CLOCK_INVALID")
            if (map_sha256, binding_sha256, clock_domain) != self._identity:
                raise ValueError("TEMPORAL_MAP_IDENTITY_CHANGED")
            identity = (packet.system_id, packet.component_id, packet.frame_id,
                        packet.child_frame_id, sha256_json(mount.model_dump(mode="json")),
                        scan.sensor_id)
            if self._native_identity is not None and identity != self._native_identity:
                raise ValueError("TEMPORAL_MAP_NATIVE_IDENTITY_CHANGED")
            position, orientation, velocity = source_pose_in_map(
                packet, binding=binding, binding_sha256=binding_sha256)
            for actual, expected in ((scan.body_position_world_enu_m, position),
                                     (scan.body_velocity_world_enu_mps, velocity)):
                if not np.allclose([getattr(actual, a) for a in "xyz"],
                                   [getattr(expected, a) for a in "xyz"], atol=1e-8, rtol=0):
                    raise ValueError("TEMPORAL_MAP_NATIVE_SCAN_MISMATCH")
            q = np.array([getattr(scan.body_orientation_world_from_body, a) for a in "wxyz"])
            expected_q = np.array([getattr(orientation, a) for a in "wxyz"])
            if min(np.linalg.norm(q - expected_q), np.linalg.norm(q + expected_q)) > 1e-8:
                raise ValueError("TEMPORAL_MAP_NATIVE_SCAN_MISMATCH")
            # 已验证的迟到帧不是身份损坏：丢弃本帧与旧修正，下一张真实新帧
            # 必须冷启动。未来时间、身份错配等仍永久停用本实例。
            if any(age > .25 for age in ages):
                raise TemporalMapSourceExpiredError("TEMPORAL_MAP_NATIVE_SOURCE_EXPIRED")
            frame = MetricRangeSensorBridge(mount).assemble(scan)
            hits = [ray for ray in frame.range_rays if ray.hit]
            result = self.update(
                [[getattr(ray.endpoint_m, a) for a in "xyz"] for ray in hits], index,
                sensor_origins_world_m=[[getattr(ray.origin_m, a) for a in "xyz"] for ray in hits],
                reference_position_world_m=[getattr(position, a) for a in "xyz"],
                source_timestamp_ns=stamp, reset_counter=packet.reset_counter,
                map_sha256=map_sha256, binding_sha256=binding_sha256, clock_domain=clock_domain)
            self._native_identity = identity
            return result
        except TemporalMapSourceExpiredError:
            self._accepted = None
            self._pending_retirement_reason = "NATIVE_SOURCE_EXPIRED"
            raise
        except (ValueError, TypeError):
            self._accepted = None
            self._invalidated = True
            raise

    # 功能：校验新观测身份与递增时间，发生损坏后停止复用历史，不用旧状态掩盖故障。
    # 输入：原始地图系点、射线原点、原生机体位置及来源身份；没有规划路线参数。
    # 输出：包含完整配准和历史来源的候选结果；不修改原生遥测或飞控状态。
    def update(self, points_world_m, index, *, sensor_origins_world_m,
               reference_position_world_m, source_timestamp_ns, reset_counter,
               map_sha256, binding_sha256, clock_domain):
        if self._invalidated:
            raise ValueError("TEMPORAL_MAP_TRACKER_INVALIDATED")
        try:
            if (map_sha256, binding_sha256, clock_domain) != self._identity:
                raise ValueError("TEMPORAL_MAP_IDENTITY_CHANGED")
            if (type(source_timestamp_ns) is not int
                    or not 0 <= source_timestamp_ns < 2**63
                    or type(reset_counter) is not int or not 0 <= reset_counter <= 255):
                raise ValueError("TEMPORAL_MAP_SOURCE_INVALID")
            if self._last_source is not None and source_timestamp_ns <= self._last_source:
                raise ValueError("TEMPORAL_MAP_SOURCE_NOT_ADVANCING")
            points = _points(points_world_m)
            origins = _origins(sensor_origins_world_m, points)
            reference = _points([reference_position_world_m])[0]
            return self._update(points, origins, reference, index,
                                source_timestamp_ns, reset_counter)
        except (ValueError, TypeError):
            self._accepted = None
            self._invalidated = True
            raise

    # 功能：用上一校正位姿和真实原生位移预测初值，再让本帧几何修正可观测分量。
    # 输入：已校验观测、配准索引、源时间和估计器重置计数。
    # 输出：单帧候选及历史使用记录；失败帧保留，历史过期或重置后重新初始化。
    def _update(self, points, origins, reference, index, stamp, reset_counter):
        retired = self._pending_retirement_reason
        self._pending_retirement_reason = None
        if self._reset_counter is not None and reset_counter != self._reset_counter:
            self._accepted = None
            retired = "ESTIMATOR_RESET"
        elif self._accepted is not None and stamp - self._accepted[0] > self._maximum_gap:
            self._accepted = None
            retired = "SOURCE_GAP"
        self._last_source, self._reset_counter = stamp, reset_counter
        previous = self._accepted
        initial = {}
        if previous is not None:
            _, old_native, old_corrected, rotation = previous
            # p_map(k) = p_map(k-1) + R_map_native * measured native displacement.
            # Rotation is essential: adding the raw displacement unrotated
            # would drift after a heading correction. Never snap to a route.
            predicted = old_corrected + rotation @ (reference - old_native)
            initial = {
                "initial_correction_world_m": (predicted - reference).tolist(),
                "initial_rotation_vector_world_rad": _rotation_vector(rotation).tolist(),
            }
        fit = fit_map_pose(points, index, sensor_origins_world_m=origins,
                           reference_position_world_m=reference.tolist(),
                           limits=self._limits, **initial)
        failed_temporal = None
        cold = None
        if previous is not None and not fit.usable_candidate:
            # Changed visible associations can trap a warm start. A bounded
            # cold retry must independently observe ALL pose directions and
            # agree with the carried pose. Partial cold fits cannot erase the
            # missing component previously supplied by motion history.
            failed_temporal = fit
            cold = fit_map_pose(points, index, sensor_origins_world_m=origins,
                                reference_position_world_m=reference.tolist(), limits=self._limits)
            if cold.usable_candidate and cold.observed_pose_rank == 6:
                position_gap = np.linalg.norm(np.asarray(cold.correction_world_m)
                                              - initial["initial_correction_world_m"])
                relative_rotation = np.asarray(cold.rotation_world_from_input) @ previous[3].T
                rotation_gap = np.linalg.norm(_rotation_vector(relative_rotation))
                if position_gap <= .02 and rotation_gap <= math.radians(.5):
                    fit = cold
                    retired = "FULL_POSE_REINITIALIZATION"
        if fit.usable_candidate:
            self._accepted = (stamp, reference.copy(),
                              reference + np.asarray(fit.correction_world_m),
                              np.asarray(fit.rotation_world_from_input))
        # A failed fit never renews the accepted history's source timestamp.
        return TemporalMapPoseResult(fit, stamp, previous[0] if previous else None,
                                     previous is not None, retired, failed_temporal, cold)
