"""Geometry-conditioned external-position noise candidates, never release approval.

A bounded calibration error is not a proof for new sensors or maps. Unobservable
directions are not encoded as precise measurements, and pixel count never
multiplies information. Source alignment uncertainty is added conservatively.
"""

from dataclasses import dataclass

import numpy as np

from .local_map_alignment import _finite
from .local_pose_alignment import MapPoseFit, rotation_exp_and_left_jacobian
from .map_localization_uncertainty import _array


@dataclass(frozen=True)
class MapPositionNoiseCandidate:
    covariance_world_m2: tuple[tuple[float, float, float], ...]
    variance_bound_m2: float
    alignment_displacement_bound_m: float
    covariance_qualified: bool = False
    motion_permission_granted: bool = False


# 功能：只对满秩地图约束构造待校准噪声，不把历史带来的缺失方向伪装成本帧测量。
# 输入：fit：图像配准结果；point_to_plane_sigma_m：独立标定值；position_floor_m：系统误差底限；
#       source_skew_seconds、speed_bound_mps、acceleration_bound_mps2：时间对齐的运动误差上界。
# 输出：保守三维协方差候选及其方向无关上界；不授予飞行或定位验收资格。
def map_position_noise_candidate(*, fit: MapPoseFit, point_to_plane_sigma_m,
                                 position_floor_m, source_skew_seconds,
                                 speed_bound_mps, acceleration_bound_mps2):
    if (type(fit) is not MapPoseFit or not fit.usable_candidate or fit.issue is not None
            or fit.observed_translation_rank != 3 or fit.observed_pose_rank != 6
            or fit.translation_information_shape is None):
        raise ValueError("MAP_MEASUREMENT_FULL_OBSERVABILITY_REQUIRED")
    for name, value, maximum, positive in (
        ("sigma", point_to_plane_sigma_m, 1., True),
        ("floor", position_floor_m, 1., True),
        ("skew", source_skew_seconds, .02, False),
        ("speed", speed_bound_mps, 100., False),
        ("acceleration", acceleration_bound_mps2, 100., False),
    ):
        if not _finite(value) or not 0 <= value <= maximum or positive and value == 0:
            raise ValueError("MAP_MEASUREMENT_NOISE_INVALID:" + name)
    information = _array(fit.translation_information_shape, matrix=True)
    if not np.allclose(information, information.T, atol=1e-12, rtol=1e-10):
        raise ValueError("MAP_MEASUREMENT_INFORMATION_INVALID")
    eigenvalues, basis = np.linalg.eigh(information)
    if eigenvalues.min() <= 1e-10 or eigenvalues.max() > 1. + 1e-10:
        raise ValueError("MAP_MEASUREMENT_INFORMATION_INVALID")
    # Equal map/sensor bias is shared by all pixels: never divide by matched_count.
    covariance = (basis * (point_to_plane_sigma_m**2 / eigenvalues)) @ basis.T
    eigenvalues, basis = np.linalg.eigh(covariance)
    covariance = (basis * np.maximum(eigenvalues, position_floor_m**2)) @ basis.T
    displacement = (speed_bound_mps * source_skew_seconds
                    + .5 * acceleration_bound_mps2 * source_skew_seconds**2)
    # Unknown correlation: Var(a+b) <= (1+alpha) Var(a) + (1+1/alpha) Var(b).
    # This upper bound is conservative without pretending alignment error is independent.
    if displacement > 0:
        alpha = displacement / float(np.sqrt(np.trace(covariance) / 3))
        covariance = (1 + alpha) * covariance + (1 + 1 / alpha) * displacement**2 * np.eye(3)
    covariance = (covariance + covariance.T) * .5
    bound = float(np.max(np.sum(np.abs(covariance), axis=1)))
    if not np.isfinite(covariance).all() or bound > 100.:
        raise ValueError("MAP_MEASUREMENT_COVARIANCE_OUT_OF_RANGE")
    return MapPositionNoiseCandidate(tuple(tuple(float(x) for x in row) for row in covariance),
                                     bound, float(displacement))


# 功能：将地图 ENU 协方差旋转为飞控 NED 位置协方差，保留交叉项和严格正定性。
# 输入：candidate：未验收的地图噪声候选。
# 输出：NED 三乘三矩阵；不复用原生速度作为独立视觉速度测量。
def map_position_noise_to_ned(candidate: MapPositionNoiseCandidate):
    if type(candidate) is not MapPositionNoiseCandidate:
        raise ValueError("MAP_MEASUREMENT_NOISE_CANDIDATE_INVALID")
    covariance = _array(candidate.covariance_world_m2, matrix=True)
    if (not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10)
            or np.linalg.eigvalsh(covariance).min() <= 0):
        raise ValueError("MAP_MEASUREMENT_COVARIANCE_INVALID")
    rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    transformed = rotation @ covariance @ rotation.T
    return tuple(tuple(float(value) for value in row) for row in transformed)


# 功能：从联合信息矩阵边缘化平移，给出保守姿态方差；不把原生姿态重复当作独立观测。
# 输入：fit：满秩配准；sigma、floor：待独立验证的标定值；源钟偏差、角速率及角加速度上界；
#       pitch_rad：拟发送姿态的俯仰角，用于覆盖旋转扰动到欧拉角的放大。
# 输出：三个欧拉角可共用的方差上界；只供测量构造，不授予融合或起飞资格。
def map_attitude_variance_candidate(*, fit, point_to_plane_sigma_m, attitude_floor_rad,
                                   source_skew_seconds, angular_speed_bound_radps,
                                   angular_acceleration_bound_radps2, pitch_rad):
    if (type(fit) is not MapPoseFit or not fit.usable_candidate or fit.issue is not None
            or fit.observed_pose_rank != 6 or fit.observed_translation_rank != 3
            or fit.pose_information_shape is None):
        raise ValueError("MAP_MEASUREMENT_FULL_OBSERVABILITY_REQUIRED")
    for name, value, maximum, positive in (
        ("sigma", point_to_plane_sigma_m, 1., True),
        ("attitude_floor", attitude_floor_rad, 1., True),
        ("skew", source_skew_seconds, .02, False),
        ("angular_speed", angular_speed_bound_radps, 100., False),
        ("angular_acceleration", angular_acceleration_bound_radps2, 100., False),
    ):
        if not _finite(value) or not 0 <= value <= maximum or positive and value == 0:
            raise ValueError("MAP_MEASUREMENT_NOISE_INVALID:" + name)
    if not _finite(pitch_rad) or abs(pitch_rad) > np.arccos(.25):
        raise ValueError("MAP_MEASUREMENT_ATTITUDE_NEAR_SINGULARITY")
    try:
        information = np.asarray(fit.pose_information_shape, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("MAP_MEASUREMENT_INFORMATION_INVALID") from error
    if (information.shape != (6, 6) or not np.isfinite(information).all()
            or not np.allclose(information, information.T, atol=1e-12, rtol=1e-10)):
        raise ValueError("MAP_MEASUREMENT_INFORMATION_INVALID")
    spectrum = np.linalg.eigvalsh(information)
    if spectrum.min() <= 1e-10 or spectrum.max() > 1e8:
        raise ValueError("MAP_MEASUREMENT_INFORMATION_INVALID")
    covariance = point_to_plane_sigma_m**2 * np.linalg.inv(information)[3:, 3:]
    # Inverting the complete system marginalizes uncertain translation. Inverting
    # just its rotation block would assume exact position and overstate precision.
    _, left = rotation_exp_and_left_jacobian(fit.rotation_vector_world_rad)
    covariance = left @ covariance @ left.T
    sigma = np.sqrt(max(attitude_floor_rad**2, float(np.linalg.eigvalsh(covariance).max())))
    angular_displacement = (angular_speed_bound_radps * source_skew_seconds
                            + .5 * angular_acceleration_bound_radps2 * source_skew_seconds**2)
    # Unknown correlation: sum standard-deviation bounds, not their variances.
    # Euler-rate Jacobian Frobenius norm squared = 1 + 2/cos(pitch)^2;
    # 3/cos(pitch)^2 is an upper bound including every angle/cross covariance.
    variance = (sigma + angular_displacement)**2 * 3. / np.cos(pitch_rad)**2
    if not np.isfinite(variance) or variance > 100.:
        raise ValueError("MAP_MEASUREMENT_COVARIANCE_OUT_OF_RANGE")
    return float(variance)
