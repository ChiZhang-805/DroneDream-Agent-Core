"""Run-bound causal geometry measurements, independent of route and simulator truth.

This front end does not confer calibration or flight approval. It preserves
source clocks and only measures observable geometry; firmware propagation owns
the interval between accepted measurements. No route projection is a position.
"""

import math
from dataclasses import asdict, dataclass

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import CalibratedRangeSensorMount, RawMetricRangeScan
from .hashing import sha256_json
from .map_measurement_noise import (
    map_attitude_variance_candidate,
    map_position_noise_candidate,
    map_position_noise_to_ned,
)
from .map_pose_transport import map_pose_transport_candidate
from .native_odometry_source import decode_source_odometry
from .temporal_map_pose import TemporalMapPoseTracker, TemporalMapSourceExpiredError

# PX4 readback still requires <=200ms actual source-to-receipt age. Reserve
# one complete 20Hz sensor period (50ms) for dispatch; this is not a new source
# time or an extension of the firmware receipt/sensor alignment safety limits.
MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS = 150_000_000


class MapMeasurementPending(ValueError):
    """No new admissible visual constraint; never replace it with route coordinates."""


@dataclass(frozen=True)
class MapNoiseBounds:
    point_to_plane_sigma_m: float
    position_floor_m: float
    attitude_floor_rad: float
    speed_bound_mps: float
    acceleration_bound_mps2: float
    angular_speed_bound_radps: float
    angular_acceleration_bound_radps2: float
    transport_clock_uncertainty_seconds: float = .002

    # 功能：校验显式标定与动力学界，不从路线或像素数量推断精度。
    # 输入：数值噪声配置；这些数值的适用性由外层独立验收负责。
    # 输出：不可变且数值合法的配置，不授予标定或飞行资格。
    def __post_init__(self):
        for name, value in asdict(self).items():
            maximum = 1. if name.endswith(('_sigma_m', '_floor_m', '_floor_rad')) else 100.
            if name == 'transport_clock_uncertainty_seconds':
                maximum = .01
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or not 0 < value <= maximum):
                raise ValueError('LIVE_MAP_NOISE_BOUND_INVALID:' + name)


class LiveMapMeasurementSource:
    """One serial worker per immutable map/frame/clock binding; no device paths."""

    # 功能：绑定地图、坐标及噪声输入，复制可变绑定，不让外部修改改变后续位置解释。
    # 输入：预编译光学地图、地图摘要、原生坐标绑定、运行钟与显式噪声界。
    # 输出：空连续测量器；不读取真值、路线、历史任务或用户目录。
    def __init__(self, *, index, map_sha256, binding, binding_sha256, clock_domain, noise):
        if type(noise) is not MapNoiseBounds or sha256_json(binding) != binding_sha256:
            raise ValueError('LIVE_MAP_SOURCE_CONFIGURATION_INVALID')
        self.tracker = TemporalMapPoseTracker(map_sha256=map_sha256,
            binding_sha256=binding_sha256, clock_domain=clock_domain)
        self.index, self.map_sha256, self.clock_domain = index, map_sha256, clock_domain
        self.binding, self.binding_sha256 = copy_json(binding), binding_sha256
        self.noise = noise
        self.last_attempted_source = None

    # 功能：将同帧原始观测配准为源时间不变的视觉约束，重复帧/过期帧不能刷新定位有效期。
    # 输入：record：认证通道解码后的几何记录；两个函数分别读取原始源钟和单调接收钟。
    # 输出：NED/FRD 位姿、保守协方差及审计证据；暂时无约束抛 Pending，身份损坏直接失败。
    def measure(self, record, *, source_now_ns, monotonic_now):
        if (type(record) is not dict or record.get('map_sha256') != self.map_sha256
                or record.get('truth_correction_applied') is not False
                or record.get('motion_permission_granted') is not False
                or record.get('native_pose_binding_sha256') != self.binding_sha256):
            raise ValueError('LIVE_MAP_CAPTURE_BINDING_MISMATCH')
        alignment = record['native_odometry_snapshot']['source_alignment']
        if alignment['clock_domain'] != self.clock_domain:
            raise ValueError('LIVE_MAP_SOURCE_CLOCK_MISMATCH')
        stamp = alignment['image_timestamp_ns']
        if type(stamp) is not int or not 0 < stamp < 2**63 or stamp % 1000:
            raise ValueError('LIVE_MAP_SOURCE_STAMP_INVALID')
        scan = RawMetricRangeScan.model_validate(record['scan'], strict=True)
        mount = CalibratedRangeSensorMount.model_validate(record['mount'], strict=True)
        packet = decode_source_odometry(alignment['fields_json'],
            system_id=alignment['system_id'], component_id=alignment['component_id'],
            received_monotonic=alignment['received_monotonic_seconds'])
        now = source_now_ns()
        self._require_source_age(stamp, now, maximum_ns=150_000_000)
        if self.last_attempted_source is not None and stamp <= self.last_attempted_source:
            raise MapMeasurementPending('LIVE_MAP_SOURCE_DID_NOT_ADVANCE')
        self.last_attempted_source = stamp
        started = monotonic_now()
        try:
            result = self.tracker.update_native(packet=packet, scan=scan, mount=mount,
                index=self.index, binding=self.binding, binding_sha256=self.binding_sha256,
                image_timestamp_ns=stamp, now_monotonic=started, map_sha256=self.map_sha256,
                clock_domain=self.clock_domain)
        except TemporalMapSourceExpiredError as error:
            raise MapMeasurementPending(str(error)) from error
        fit = result.fit
        if not fit.usable_candidate or fit.issue is not None:
            # 保留求解器的具体拒绝原因，避免所有失败都被汇总成“无几何”而无法复现。
            # 仅允许固定格式代码穿过隔离进程；不把原始点云或路径塞入异常文本。
            issue = fit.issue or 'UNSPECIFIED'
            if (type(issue) is not str or len(issue) > 96
                    or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in issue)):
                raise ValueError('LIVE_MAP_GEOMETRY_DIAGNOSTIC_INVALID')
            raise MapMeasurementPending('LIVE_MAP_NO_USABLE_GEOMETRY_' + issue)
        if fit.observed_pose_rank != 6 or fit.observed_translation_rank != 3:
            # Keep measured motion history, but do not manufacture independent
            # information along the null space of this image's geometry.
            raise MapMeasurementPending('LIVE_MAP_PARTIAL_CONSTRAINT_ONLY')
        candidate = map_pose_transport_candidate(packet=packet, fit=fit, binding=self.binding,
            binding_sha256=self.binding_sha256, image_timestamp_ns=stamp)
        noise = self.noise
        skew = abs(candidate.source_skew_us) / 1e6
        translation = map_position_noise_candidate(fit=fit,
            point_to_plane_sigma_m=noise.point_to_plane_sigma_m,
            position_floor_m=noise.position_floor_m, source_skew_seconds=skew,
            speed_bound_mps=noise.speed_bound_mps,
            acceleration_bound_mps2=noise.acceleration_bound_mps2)
        matrix = map_position_noise_to_ned(translation)
        w, x, y, z = candidate.quaternion_ned_from_frd_wxyz
        angular = map_attitude_variance_candidate(fit=fit,
            point_to_plane_sigma_m=noise.point_to_plane_sigma_m,
            attitude_floor_rad=noise.attitude_floor_rad, source_skew_seconds=skew,
            angular_speed_bound_radps=noise.angular_speed_bound_radps,
            angular_acceleration_bound_radps2=noise.angular_acceleration_bound_radps2,
            pitch_rad=math.asin(max(-1., min(1., 2*(w*y-z*x)))))
        covariance = [0.] * 21
        # The receiver's time synchronizer introduces a bounded sample-time
        # uncertainty even on a shared simulation clock. Pay for that uncertainty
        # in position AND attitude, rather than treating a tiny offset as either
        # perfect synchronization or a complete loss of localization.
        dt = noise.transport_clock_uncertainty_seconds
        displacement = noise.speed_bound_mps*dt + .5*noise.acceleration_bound_mps2*dt*dt
        pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
        angle = (noise.angular_speed_bound_radps*dt
                 + .5*noise.angular_acceleration_bound_radps2*dt*dt)
        for row, slot in enumerate((0, 6, 11)):
            covariance[slot] = (math.sqrt(sum(abs(value) for value in matrix[row])) + displacement)**2
        for slot in (15, 18, 20):
            covariance[slot] = (math.sqrt(angular) + math.sqrt(3.)/abs(math.cos(pitch))*angle)**2
        now = source_now_ns()
        self._require_source_age(stamp, now, maximum_ns=MAXIMUM_MEASUREMENT_DISPATCH_AGE_NS)
        return {'source_timestamp_ns': stamp, 'position_ned_m': candidate.position_ned_m,
            'transport_clock_uncertainty_us': round(dt*1e6),
            'quaternion_ned_from_frd_wxyz': candidate.quaternion_ned_from_frd_wxyz,
            'pose_covariance_upper': tuple(covariance), 'candidate': asdict(candidate),
            'temporal_result': asdict(result), 'noise_candidate': asdict(translation),
            'source_age_at_return_ms': (now-stamp)/1e6,
            'compute_wall_ms': (monotonic_now()-started)*1000,
            'covariance_qualified': False, 'motion_permission_granted': False}

    # 功能：核对原始源钟而非接收时刻；迟到可等下一帧，非法或倒退钟不能静默重试。
    # 输入：图像纳秒、当前真实源纳秒和固定年龄上限。
    # 输出：无；没有新钟或过期则 Pending，未来图像及畸形钟直接拒绝。
    @staticmethod
    def _require_source_age(stamp, now, *, maximum_ns):
        if now is None:
            raise MapMeasurementPending('LIVE_MAP_SOURCE_CLOCK_NOT_READY')
        if type(now) is not int or now < stamp or now >= 2**63:
            raise ValueError('LIVE_MAP_SOURCE_CLOCK_INVALID')
        if now - stamp > maximum_ns:
            raise MapMeasurementPending('LIVE_MAP_SOURCE_EXPIRED')
