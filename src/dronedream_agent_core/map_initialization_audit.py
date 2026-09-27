"""Bounded registration sensitivity to an explicitly supplied pose prior.

This is an offline diagnostic, NOT calibrated measurement covariance. It varies
the pose used to project the SAME camera observations, preserving ray origins.
Sensor noise, map bias and untested convergence basins are not estimated here.
Inspired by the initialization analysis in Brossard et al., arXiv:1909.05722;
this implements a positive-weight cubature probe, not that paper's full method.
"""

from dataclasses import asdict, replace

import numpy as np

from .local_map_alignment import _finite, _origins, _points
from .local_pose_alignment import fit_map_pose, rotation_exp_and_left_jacobian
from .map_pose_search import search_map_pose


# 功能：校验地图系六维初始误差协方差，保留交叉项与零方向，不补造未知量。
# 输入：value：前三维米、后三维世界轴小旋转弧度的六乘六协方差。
# 输出：对称半正定矩阵；布尔、未知项、非对称及负方差明确拒绝。
def _prior(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 6
        or any(
            not isinstance(row, (list, tuple))
            or len(row) != 6
            or any(not _finite(v) or abs(v) > 100 for v in row)
            for row in value
        )
    ):
        raise ValueError("MAP_INITIALIZATION_PRIOR_INVALID")
    covariance = np.array(value, dtype=float)
    if not np.allclose(covariance, covariance.T, atol=1e-14, rtol=1e-12):
        raise ValueError("MAP_INITIALIZATION_PRIOR_INVALID")
    if np.linalg.eigvalsh(covariance).min() < -1e-14:
        raise ValueError("MAP_INITIALIZATION_PRIOR_INVALID")
    return (covariance + covariance.T) * 0.5


# 功能：固定地图及相机观测，分别改变初始位置和姿态，暴露局部配准对初始化的依赖。
# 输入：points、sensor_origins_world_m：同帧命中点与射线起点；reference_position_world_m：
#       当前机体地图位置；prior_covariance_world：已明确来源的地图系六维先验；index：地图索引。
# 输出：中心与十二个等权立方采样结果、位置传播协方差及输入输出互相关；任何失败均保留，
#       且不发布部分成功样本统计，不授予飞行或定位资格。此函数不读取真值。
def audit_map_initialization(
    points,
    index,
    *,
    sensor_origins_world_m,
    reference_position_world_m,
    prior_covariance_world,
    limits=None,
    registration_mode="local",
):
    points = _points(points)
    reference = _points([reference_position_world_m])[0]
    origins = _origins(sensor_origins_world_m, points)
    covariance = _prior(prior_covariance_world)
    if registration_mode not in ("local", "multistart"):
        raise ValueError("MAP_INITIALIZATION_REGISTRATION_MODE_INVALID")
    eigenvalues, basis = np.linalg.eigh(covariance)
    roots = basis * np.sqrt(6 * np.maximum(eigenvalues, 0))
    offsets = np.vstack((np.zeros((1, 6)), roots.T, -roots.T))
    # Invalid diagnostic envelopes are rejected rather than clipped into the
    # solver's basin. Broad priors may legitimately fail registration below.
    if np.max(np.linalg.norm(offsets[:, 3:], axis=1)) > np.pi:
        raise ValueError("MAP_INITIALIZATION_ROTATION_ENVELOPE_INVALID")
    rows = []
    for offset in offsets:
        rotation = rotation_exp_and_left_jacobian(offset[3:])[0]
        shifted_reference = reference + offset[:3]
        transformed = shifted_reference + (points - reference) @ rotation.T
        shifted_origins = shifted_reference + (origins - reference) @ rotation.T
        solver = fit_map_pose if registration_mode == "local" else search_map_pose
        solved = solver(
            transformed,
            index,
            sensor_origins_world_m=shifted_origins,
            reference_position_world_m=shifted_reference,
            limits=limits,
        )
        search = None
        if registration_mode == "multistart":
            search = asdict(solved)
            fit = solved.candidate
            if fit is None:
                fit = replace(
                    solved.attempts[0],
                    usable_candidate=False,
                    issue=solved.issue,
                    correction_world_m=(0.0, 0.0, 0.0),
                    rotation_vector_world_rad=(0.0, 0.0, 0.0),
                    rotation_world_from_input=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    translation_information_shape=None,
                )
        else:
            fit = solved
        row = {
            "initial_offset_world": offset.tolist(),
            "fit": asdict(fit),
            "output_delta_world_m": None,
            "output_rotation_world": None,
            "search": search,
        }
        if fit.usable_candidate:
            row["output_delta_world_m"] = (offset[:3] + fit.correction_world_m).tolist()
            row["output_rotation_world"] = (
                np.array(fit.rotation_world_from_input) @ rotation
            ).tolist()
        rows.append(row)
    result = {
        "schema": "dronedream.map-initialization-audit.v1",
        "registration_mode": registration_mode,
        "samples": rows,
        "failed_samples": sum(not row["fit"]["usable_candidate"] for row in rows),
        "prior_covariance_world": covariance.tolist(),
        "propagation": None,
        "covariance_qualified": False,
        "motion_permission_granted": False,
    }
    if result["failed_samples"]:
        return result
    outputs = np.array([row["output_delta_world_m"] for row in rows[1:]])
    mean = outputs.mean(axis=0)
    deviations = outputs - mean
    result["propagation"] = {
        "mean_delta_world_m": mean.tolist(),
        "position_covariance_from_initialization_m2": (deviations.T @ deviations / 12).tolist(),
        "input_pose_output_position_cross_covariance": (offsets[1:].T @ deviations / 12).tolist(),
        "minimum_translation_rank": min(row["fit"]["observed_translation_rank"] for row in rows),
        "maximum_position_deviation_from_center_m": float(
            np.max(np.linalg.norm(outputs - np.array(rows[0]["output_delta_world_m"]), axis=1))
        ),
    }
    return result
