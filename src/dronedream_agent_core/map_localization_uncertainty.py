"""Offline covariance-intersection candidate; never an execution authority.

Map matching and native/previous estimates can share observations. Their
cross-correlation must not be assumed zero. Unit-noise map information is
normalized by total robust weight, not multiplied by the number of pixels.
The caller must independently calibrate the input uncertainty model; passing a
small residual or a good covariance-intersection result cannot do that.
"""

from dataclasses import dataclass

import numpy as np

from .local_map_alignment import _finite
from .local_pose_alignment import MapPoseFit


@dataclass(frozen=True)
class MapLocalizationCandidate:
    position_world_m: tuple[float, float, float]
    covariance_world_m2: tuple[tuple[float, float, float], ...]
    variance_bound_m2: float
    native_information_weight: float
    observed_translation_rank: int
    covariance_qualified: bool = False
    motion_permission_granted: bool = False


# 功能：严格校验米制三维向量和矩阵，不允许 NumPy 把布尔或字符串转成数值证据。
# 输入：value：三维向量或三乘三矩阵；matrix：要求矩阵时为真。
# 输出：独立 float64 数组。
def _array(value, *, matrix=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("MAP_LOCALIZATION_ARRAY_INVALID")
    if matrix:
        if any(
            not isinstance(row, (list, tuple)) or len(row) != 3 or any(not _finite(v) for v in row)
            for row in value
        ):
            raise ValueError("MAP_LOCALIZATION_ARRAY_INVALID")
    elif any(not _finite(v) for v in value):
        raise ValueError("MAP_LOCALIZATION_ARRAY_INVALID")
    result = np.array(value, dtype=np.float64)
    if np.max(np.abs(result)) > (10_000.0 if matrix else 1e6):
        raise ValueError("MAP_LOCALIZATION_ARRAY_RANGE_INVALID")
    return result


# 功能：以未知互相关的协方差交叉法融合先验和地图约束；缺失方向信息为零而不是零方差。
# 输入：position_world_m、covariance_world_m2：同一地图系先验；fit：该位姿的联合拟合；
#       point_to_plane_sigma_m：调用方明确提供的误差模型，必须另行校准，不能用残差冒充。
# 输出：待验证候选；不修改飞控、不授予定位资格或运动权限。
def fuse_map_localization_candidate(
    *, position_world_m, covariance_world_m2, fit: MapPoseFit, point_to_plane_sigma_m
):
    position = _array(position_world_m)
    covariance = _array(covariance_world_m2, matrix=True)
    if (
        not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10)
        or np.linalg.eigvalsh(covariance).min() <= 1e-12
    ):
        raise ValueError("MAP_LOCALIZATION_PRIOR_COVARIANCE_INVALID")
    if not _finite(point_to_plane_sigma_m) or not 1e-5 <= point_to_plane_sigma_m <= 1.0:
        raise ValueError("MAP_LOCALIZATION_NOISE_MODEL_REQUIRED")
    if (
        not isinstance(fit, MapPoseFit)
        or fit.usable_candidate is not True
        or fit.issue is not None
        or fit.translation_information_shape is None
    ):
        raise ValueError("MAP_LOCALIZATION_USABLE_FIT_REQUIRED")
    if not np.allclose(_array(fit.reference_position_world_m), position, atol=1e-9, rtol=0):
        raise ValueError("MAP_LOCALIZATION_FIT_REFERENCE_MISMATCH")
    information = _array(fit.translation_information_shape, matrix=True)
    if not np.allclose(information, information.T, atol=1e-12, rtol=1e-10):
        raise ValueError("MAP_LOCALIZATION_INFORMATION_INVALID")
    eigenvalues, basis = np.linalg.eigh(information)
    if eigenvalues.min() < -1e-10 or eigenvalues.max() > 1.0 + 1e-10:
        raise ValueError("MAP_LOCALIZATION_INFORMATION_INVALID")
    observed = eigenvalues > 1e-10
    rank = int(np.count_nonzero(observed))
    if type(fit.observed_translation_rank) is not int or rank != fit.observed_translation_rank:
        raise ValueError("MAP_LOCALIZATION_INFORMATION_RANK_MISMATCH")
    # Remove round-off in null directions, never assign them a made-up minimum
    # eigenvalue; a two-plane image is still missing one positional constraint.
    map_precision = (
        (basis[:, observed] * eigenvalues[observed])
        @ basis[:, observed].T
        / point_to_plane_sigma_m**2
    )
    native_precision = np.linalg.inv(covariance)
    correction = _array(fit.correction_world_m)
    # Work in correction coordinates to avoid large-world-coordinate cancellation.
    map_rhs = map_precision @ correction
    best = None
    for index in range(21):
        weight = index / 20.0
        if index == 0 and rank != 3:
            continue
        precision = weight * native_precision + (1.0 - weight) * map_precision
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        # Include the native-only endpoint: fusion need not improve an already
        # stronger prior. A tie prefers the prior to avoid gratuitous correction.
        score = (-float(logdet), -weight)
        if best is None or score < best[0]:
            fused = np.linalg.inv(precision)
            estimate = position + fused @ ((1.0 - weight) * map_rhs)
            best = score, weight, fused, estimate
    if best is None:
        raise ValueError("MAP_LOCALIZATION_FUSION_NOT_FINITE")
    _, weight, fused, estimate = best
    fused = (fused + fused.T) * 0.5
    if not np.isfinite(fused).all() or not np.isfinite(estimate).all():
        raise ValueError("MAP_LOCALIZATION_FUSION_NOT_FINITE")
    bound = float(np.max(np.sum(np.abs(fused), axis=1)))
    return MapLocalizationCandidate(
        tuple(float(v) for v in estimate),
        tuple(tuple(float(v) for v in row) for row in fused),
        bound,
        float(weight),
        rank,
    )
