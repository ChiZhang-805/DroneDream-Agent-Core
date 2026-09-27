"""Bounded joint position/attitude registration against visible map surfaces.

An offline candidate estimator, not an authority to replace native state. The
rotation acts about the INPUT body reference, not the world origin. Unknown
directions remain explicit; neither a small residual nor inverse curvature is
a qualified covariance. Uses the same visibility geometry as translation-only
calibration, with no nearest-surface fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .local_map_alignment import (
    MapAlignmentLimits,
    MapSurfaceIndex,
    _finite,
    _huber_cost,
    _origins,
    _points,
)


# 功能：
#   计算 SO(3) 指数映射及左雅可比；零附近用级数避免消减误差，输入不是欧拉角。
# 输入：
#   vector：长度三的有限旋转向量，弧度模长不超过 π。
# 输出：
#   matrices：旋转矩阵与左雅可比矩阵二元组。
def rotation_exp_and_left_jacobian(vector):
    try:
        w = _points([vector])[0]
    except ValueError as error:
        raise ValueError("MAP_POSE_ROTATION_VECTOR_INVALID") from error
    if w.shape != (3,) or not np.isfinite(w).all() or np.linalg.norm(w) > math.pi:
        raise ValueError("MAP_POSE_ROTATION_VECTOR_INVALID")
    x, y, z = w
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta2 = float(w @ w)
    if theta2 < 1e-8:
        a = 1 - theta2 / 6 + theta2**2 / 120
        b = 0.5 - theta2 / 24 + theta2**2 / 720
        c = 1 / 6 - theta2 / 120 + theta2**2 / 5040
    else:
        theta = math.sqrt(theta2)
        a, b = math.sin(theta) / theta, (1 - math.cos(theta)) / theta2
        c = (1 - a) / theta2
    square = skew @ skew
    matrices = np.eye(3) + a * skew + b * square, np.eye(3) + b * skew + c * square
    return matrices


@dataclass(frozen=True)
class MapPoseAlignmentLimits:
    geometry: MapAlignmentLimits = field(default_factory=MapAlignmentLimits)
    maximum_rotation_rad: float = math.radians(5.0)
    rotation_length_scale_m: float = 2.0
    minimum_scaled_eigenvalue: float = 1e-4
    relative_eigenvalue_threshold: float = 1e-4
    maximum_iterations: int = 12

    # 功能：
    #   约束平移/姿态联合估计的旋转、尺度、可观测阈值及迭代成本。
    # 输入：
    #   self：拟合限制配置。
    # 输出：
    #   None：配置有效时返回，否则抛出异常。
    def __post_init__(self):
        if not isinstance(self.geometry, MapAlignmentLimits):
            raise ValueError("MAP_POSE_GEOMETRY_LIMITS_INVALID")
        for name, low, high in (
            ("maximum_rotation_rad", 0.0, math.radians(10.0)),
            ("rotation_length_scale_m", 0.1, 20.0),
            ("minimum_scaled_eigenvalue", 0.0, 0.1),
            ("relative_eigenvalue_threshold", 0.0, 0.1),
        ):
            value = getattr(self, name)
            if not _finite(value) or not low < value <= high:
                raise ValueError("MAP_POSE_LIMIT_INVALID:" + name)
        if type(self.maximum_iterations) is not int or not 1 <= self.maximum_iterations <= 24:
            raise ValueError("MAP_POSE_ITERATION_LIMIT_INVALID")


@dataclass(frozen=True)
class MapPoseFit:
    usable_candidate: bool
    correction_world_m: tuple[float, float, float]
    rotation_vector_world_rad: tuple[float, float, float]
    rotation_world_from_input: tuple[tuple[float, float, float], ...]
    reference_position_world_m: tuple[float, float, float]
    observed_pose_rank: int
    unobserved_scaled_pose_directions: tuple[tuple[float, ...], ...]
    observed_translation_rank: int
    unobserved_directions_world: tuple[tuple[float, float, float], ...]
    rotation_length_scale_m: float
    matched_count: int
    residual_p95_m: float | None
    iterations: int
    issue: str | None
    retired_correspondence_count: int
    attitude_held_fixed: bool = False
    covariance_qualified: bool = False
    motion_permission_granted: bool = False
    # Unit-noise, normalized point-to-plane information after eliminating
    # unknown attitude. This is NOT a calibrated covariance or authority.
    translation_information_shape: tuple[tuple[float, float, float], ...] | None = None
    # Normalized physical Jacobian information: translation metres, rotation
    # exponential coordinates radians. Only populated for a full-rank fit.
    pose_information_shape: tuple[tuple[float, ...], ...] | None = None


# 功能：
#   绕输入机体参考点变换点云并计算点到面雅可比，不能绕远处世界原点旋转。
# 输入：
#   points_relative：相对机体参考点的观测点。
#   reference：机体在世界坐标系的输入参考点。
#   state：前三项米、后三项 length 倍弧度的修正状态。
#   normals：当前匹配表面法向。
#   length：旋转弧度与平移米之间的条件数缩放长度。
# 输出：
#   transformed：变换后点、六维雅可比及旋转矩阵三元组。
def _transform_and_jacobian(points_relative, reference, state, normals, length):
    rotation, left = rotation_exp_and_left_jacobian(state[3:] / length)
    rotated = points_relative @ rotation.T
    points = reference + state[:3] + rotated
    angular = (np.cross(rotated, normals) @ left) / length
    transformed = points, np.column_stack((normals, angular)), rotation
    return transformed


# 功能：
#   按缩放信息矩阵谱判断当前六维可观测子空间；小残差不自动带来满秩证据。
# 输入：
#   jacobian：当前点到面雅可比。
#   weights：归一化鲁棒权重。
#   limits：绝对与相对特征值阈值。
# 输出：
#   subspaces：有效特征值、可观测基和未观测方向三元组。
def _observable(jacobian, weights, limits):
    gram = jacobian.T @ (weights[:, None] * jacobian)
    eigenvalues, vectors = np.linalg.eigh(gram)
    threshold = max(
        limits.minimum_scaled_eigenvalue,
        float(eigenvalues[-1]) * limits.relative_eigenvalue_threshold,
    )
    observed = eigenvalues >= threshold
    subspaces = eigenvalues[observed], vectors[:, observed], vectors[:, ~observed].T
    return subspaces


# 功能：
#   先消去未知旋转可解释的方向再评估平移可观测性，避免把姿态歧义误当定位证据。
# 输入：
#   jacobian：当前六维雅可比。
#   weights：当前点权重。
#   limits：有效方向阈值。
# 输出：
#   observability：平移秩和未约束方向，不是融合后的协方差。
def _translation_observability(jacobian, weights, limits):
    # Use EXACTLY the spectrum retained by the actual six-dimensional solver.
    # Reusing the raw Jacobian here used to resurrect a weak mixed position/
    # attitude mode that the solver had deliberately removed. That could label
    # position as fully observable while leaving centimetres of correction in
    # the discarded pose direction.
    eigenvalues, observed_basis, _ = _observable(jacobian, weights, limits)
    if not len(eigenvalues):
        return 0, np.eye(3), np.zeros((3, 3))
    weighted = np.sqrt(eigenvalues[:, None]) * observed_basis.T
    translation, angular = weighted[:, :3], weighted[:, 3:]
    u, singular, _ = np.linalg.svd(angular, full_matrices=False)
    basis = u[:, singular > 1e-10]
    remaining = translation - basis @ (basis.T @ translation)
    eigenvalues, vectors = np.linalg.eigh(remaining.T @ remaining)
    observed = eigenvalues >= limits.minimum_scaled_eigenvalue
    information = (vectors[:, observed] * eigenvalues[observed]) @ vectors[:, observed].T
    observability = int(np.count_nonzero(observed)), vectors[:, ~observed].T, information
    return observability


# 功能：只在本轮可观测方向求解增量，不因零空间随迭代旋转而删除已有状态、造成上升步。
# 输入：state：当前六维缩放状态；gradient：同一鲁棒目标梯度；
#       eigenvalues、basis：该目标保留的正特征值和正交可观测基。
# 输出：下一候选状态；未观测方向保持当前猜测而非被当作准确值，资格仍由独立信息矩阵决定。
def _observable_proposal(state, gradient, eigenvalues, basis):
    # g.T @ delta = -sum((B.T @ g)**2 / eigenvalues) <= 0.
    # Projecting the TOTAL state instead adds -(I-BB.T)@state, which can
    # point uphill when a weak, position/attitude-coupled mode changes.
    return state - basis @ ((basis.T @ gradient) / eigenvalues)


# 功能：
#   联合拟合平移和姿态，逐轮重新核对可见性；仅更新可观测增量，不授予运动权限。
#   不可观测方向仍是未验证的初始/迭代猜测，调用方不能把部分约束候选当作完整精确位姿。
#   候选超出包络时拒绝而非裁剪后宣称成功，失败不回退为虚假的零误差定位。
# 输入：
#   value：世界坐标系观测点阵。
#   index：当前地图表面索引。
#   sensor_origins_world_m：每条射线的真实采样起点。
#   reference_position_world_m：旋转所围绕的输入机体参考点。
#   limits：可选的几何、旋转和迭代上限。
#   initial_correction_world_m、initial_rotation_vector_world_rad：可选成对初始迭代值；
#       不更改原始参考系，后续仍校验总修正包络，不能分阶段绕过上限。
# 输出：
#   fit：六维修正候选、可观测子空间、支持度、残差和原因。
def fit_map_pose(
    value,
    index: MapSurfaceIndex,
    *,
    sensor_origins_world_m,
    reference_position_world_m,
    limits: MapPoseAlignmentLimits | None = None,
    initial_correction_world_m=None,
    initial_rotation_vector_world_rad=None,
):
    points = _points(value)
    reference = _points([reference_position_world_m])[0]
    origins = _origins(sensor_origins_world_m, points, "MAP_POSE_ORIGINS_INVALID")
    limits = MapPoseAlignmentLimits() if limits is None else limits
    if not isinstance(limits, MapPoseAlignmentLimits) or not isinstance(index, MapSurfaceIndex):
        raise ValueError("MAP_POSE_INPUT_CONTRACT_INVALID")
    if not isinstance(limits.geometry, MapAlignmentLimits):
        raise ValueError("MAP_POSE_GEOMETRY_LIMITS_INVALID")
    limits = replace(limits, geometry=replace(limits.geometry))
    geometry, length = limits.geometry, limits.rotation_length_scale_m
    relative, origin_relative = points - reference, origins - reference
    state, rank, null, translation_rank, translation_null = np.zeros(6), 0, np.eye(6), 0, np.eye(3)
    # A coarse map search may supply a starting iterate, not a new origin.
    # Every reported correction and envelope remains relative to the ORIGINAL
    # input pose, so splitting a correction into stages cannot bypass limits.
    if (initial_correction_world_m is None) != (initial_rotation_vector_world_rad is None):
        raise ValueError("MAP_POSE_INITIAL_STATE_INCOMPLETE")
    if initial_correction_world_m is not None:
        initial_position = _points([initial_correction_world_m])[0]
        initial_rotation = _points([initial_rotation_vector_world_rad])[0]
        if (np.linalg.norm(initial_position) > geometry.maximum_translation_m
                or np.linalg.norm(initial_rotation) > limits.maximum_rotation_rad):
            raise ValueError("MAP_POSE_INITIAL_STATE_OUTSIDE_ENVELOPE")
        state = np.concatenate((initial_position, initial_rotation * length))
    count, iteration, retired, residual = 0, 0, 0, None
    translation_information = None
    pose_information = None
    eligible = None

    # 功能：
    #   汇总本次联合拟合，失败时清零修正但保留实际支持度和退役点诊断。
    # 输入：
    #   issue：失败原因或成功时的 None。
    # 输出：
    #   fit：不含运动权限或伪造协方差的结果。
    def result(issue):
        accepted = issue is None
        rotation = rotation_exp_and_left_jacobian(state[3:] / length)[0] if accepted else np.eye(3)
        fit = MapPoseFit(
            accepted,
            tuple(float(v) for v in (state[:3] if accepted else np.zeros(3))),
            tuple(float(v) for v in (state[3:] / length if accepted else np.zeros(3))),
            tuple(tuple(float(v) for v in row) for row in rotation),
            tuple(float(v) for v in reference),
            rank,
            tuple(tuple(float(v) for v in row) for row in null),
            translation_rank,
            tuple(tuple(float(v) for v in row) for row in translation_null),
            length,
            count,
            residual,
            iteration,
            issue,
            retired,
            translation_information_shape=(
                tuple(tuple(float(v) for v in row) for row in translation_information)
                if accepted and translation_information is not None else None
            ),
            pose_information_shape=(
                tuple(tuple(float(v) for v in row) for row in pose_information)
                if accepted and rank == 6 and pose_information is not None else None
            ),
        )
        return fit

    for step in range(limits.maximum_iterations):
        iteration = step + 1
        rotation = rotation_exp_and_left_jacobian(state[3:] / length)[0]
        transformed = reference + state[:3] + relative @ rotation.T
        shifted_origins = reference + state[:3] + origin_relative @ rotation.T
        matches = index.match(transformed, geometry, sensor_origins_world_m=shifted_origins)
        if eligible is None:
            eligible = matches.valid.copy()
        else:
            retired += int(np.count_nonzero(eligible & ~matches.valid))
            eligible &= matches.valid
        matches = replace(matches, valid=matches.valid & eligible)
        mask = matches.valid
        count = int(np.count_nonzero(mask))
        if count < max(
            geometry.minimum_correspondences, len(points) * geometry.minimum_matched_fraction
        ):
            return result("MAP_POSE_CORRESPONDENCES_INSUFFICIENT")
        normals, offsets = matches.normals[mask], matches.offsets[mask]
        positioned, jacobian, _ = _transform_and_jacobian(
            relative[mask], reference, state, normals, length
        )
        errors = np.einsum("ni,ni->n", normals, positioned) - offsets
        weights = np.minimum(1.0, geometry.huber_scale_m / np.maximum(np.abs(errors), 1e-12))
        weights /= np.sum(weights)
        eigenvalues, basis, null = _observable(jacobian, weights, limits)
        rank = len(eigenvalues)
        physical_jacobian = jacobian * np.array([1., 1., 1., length, length, length])
        pose_information = physical_jacobian.T @ (weights[:, None] * physical_jacobian)
        translation_rank, translation_null, translation_information = _translation_observability(
            jacobian, weights, limits)
        if not rank:
            return result("MAP_POSE_NO_OBSERVABLE_DIRECTION")
        gradient = jacobian.T @ (weights * errors)
        proposal = _observable_proposal(state, gradient, eigenvalues, basis)
        # Do not turn an out-of-envelope minimum into a clipped 'success'.
        if (
            np.linalg.norm(proposal[:3]) > geometry.maximum_translation_m
            or np.linalg.norm(proposal[3:] / length) > limits.maximum_rotation_rad
        ):
            return result("MAP_POSE_CORRECTION_OUTSIDE_ENVELOPE")
        delta = proposal - state
        if np.linalg.norm(delta) <= 1e-6:
            residual = float(np.quantile(np.abs(errors), 0.95))
            if residual > geometry.maximum_residual_p95_m:
                return result("MAP_POSE_RESIDUAL_EXCEEDS_ENVELOPE")
            return result(None)
        cost = _huber_cost(errors, geometry.huber_scale_m)
        accepted = False
        # Same correspondence set and objective for all eight trial steps.
        # Visibility/rank are recomputed next iteration, even after tiny steps.
        for exponent in range(8):
            candidate = state + delta * (0.5**exponent)
            positioned, _, _ = _transform_and_jacobian(
                relative[mask], reference, candidate, normals, length
            )
            changed_errors = np.einsum("ni,ni->n", normals, positioned) - offsets
            if _huber_cost(changed_errors, geometry.huber_scale_m) < cost:
                state, accepted = candidate, True
                break
        if not accepted:
            return result("MAP_POSE_NO_DESCENT")
    return result("MAP_POSE_DID_NOT_CONVERGE")
