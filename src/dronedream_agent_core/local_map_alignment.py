"""Bounded point-to-plane translation fitting, conditional on input attitude.

This computes a real geometric correction, not a navigation command. It leaves
unobserved translation directions unchanged and NEVER invents a covariance.
Registration convergence is not proof of map association, calibrated attitude,
clock validity or motion authority. Qualification/fusion belongs downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .map_surface_geometry import BOX, CYLINDER, SPHERE, first_surface_hits


# 功能：
#   核验配置中的有限物理标量，布尔值与超大整数不能进入几何计算。
# 输入：
#   value：待核验配置值。
# 输出：
#   finite：值是可表示的有限数。
def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


# 功能：
#   为正值配置统一提供明确错误码。
# 输入：
#   value：候选配置值。
#   name：错误码中的配置名称。
# 输出：
#   None：合法时返回，非法时抛出异常。
def _positive(value, name):
    if not _finite(value) or value <= 0:
        raise ValueError(f"MAP_ALIGNMENT_{name}_INVALID")


# 功能：
#   在分配浮点副本前检查点数、形状和原始类型，禁止把文本、布尔或复数强转成观测位置。
# 输入：
#   value：最多 512 个三维点的数值数组或嵌套序列。
# 输出：
#   points：独立、有限且坐标绝对值不超过一百万米的 float64 点阵。
def _points(value):
    try:
        if isinstance(value, np.ndarray):
            raw = value
        elif isinstance(value, (list, tuple)) and 1 <= len(value) <= 512:
            for row in value:
                if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 3:
                    raise ValueError("MAP_ALIGNMENT_POINTS_INVALID")
                if any(isinstance(item, (bool, np.bool_)) for item in row):
                    raise ValueError("MAP_ALIGNMENT_POINTS_INVALID")
            raw = np.asarray(value)
        else:
            raise ValueError("MAP_ALIGNMENT_POINTS_INVALID")
        if (
            raw.ndim != 2
            or raw.shape[1] != 3
            or not 1 <= len(raw) <= 512
            or raw.dtype.kind not in "fiu"
        ):
            raise ValueError("MAP_ALIGNMENT_POINTS_INVALID")
        points = np.array(raw, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("MAP_ALIGNMENT_POINTS_INVALID") from exc
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not 1 <= len(points) <= 512
        or not np.isfinite(points).all()
        or np.max(np.abs(points)) > 1e6
    ):
        raise ValueError("MAP_ALIGNMENT_POINTS_INVALID")
    return points


# 功能：
#   校验真实射线起点，单个三维起点可广播，逐点起点必须与观测点数一致。
# 输入：
#   value：一个三维起点或逐点起点阵列。
#   points：已验证的观测点阵。
#   error_code：当前调用方的起点错误码。
# 输出：
#   origins：独立的逐点起点阵列。
def _origins(value, points, error_code="MAP_ALIGNMENT_ORIGINS_INVALID"):
    try:
        raw = np.asarray(value)
        if raw.shape == (3,):
            single = _points([value])
            origins = np.broadcast_to(single, points.shape).copy()
        else:
            origins = _points(value)
        if origins.shape != points.shape:
            raise ValueError(error_code)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(error_code) from error
    return origins


# 功能：
#   计算 Huber 目标，限制离群点对局部几何配准的牵引，不能据此宣称噪声已标定。
# 输入：
#   errors：当前点到面的残差。
#   scale：正值 Huber 阈值。
# 输出：
#   cost：当前残差集合的鲁棒目标值。
def _huber_cost(errors, scale):
    absolute = np.abs(errors)
    clipped = np.minimum(absolute, scale)
    cost = float(np.sum(0.5 * clipped**2 + scale * (absolute - clipped)))
    return cost


# 功能：
#   在修正包络内作有界 Huber 一维导数搜索，改善饱和残差收敛；不能把裁剪到边界当收敛。
# 输入：
#   correction、proposal：当前平移修正与候选修正。
#   normals、rhs：本次固定的平面法向与方程右端。
#   limits：搜索包络和鲁棒阈值。
# 输出：
#   candidate：降低同一目标值的候选；不成立时返回原 proposal。
def _line_refinement(correction, proposal, normals, rhs, limits):
    direction = proposal - correction
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return proposal
    direction /= length
    derivative_axis = normals @ direction
    errors = normals @ correction - rhs
    start_derivative = derivative_axis @ np.clip(
        errors, -limits.huber_scale_m, limits.huber_scale_m
    )
    projected = correction @ direction
    radius = limits.maximum_translation_m
    high = -projected + np.sqrt(max(0.0, projected**2 + radius**2 - correction @ correction))
    end_derivative = derivative_axis @ np.clip(
        errors + high * derivative_axis, -limits.huber_scale_m, limits.huber_scale_m
    )
    if start_derivative >= 0 or end_derivative <= 0:
        return proposal
    low = 0.0
    for _ in range(22):
        middle = (low + high) / 2
        derivative = derivative_axis @ np.clip(
            errors + middle * derivative_axis, -limits.huber_scale_m, limits.huber_scale_m
        )
        if derivative < 0:
            low = middle
        else:
            high = middle
    candidate = correction + ((low + high) / 2) * direction
    if _huber_cost(normals @ candidate - rhs, limits.huber_scale_m) < _huber_cost(
        normals @ proposal - rhs, limits.huber_scale_m
    ):
        return candidate
    return proposal


@dataclass(frozen=True)
class MapAlignmentLimits:
    # Experimental operating envelope, not calibrated sensor noise/confidence.
    maximum_association_distance_m: float = 0.25
    maximum_translation_m: float = 0.20
    maximum_residual_p95_m: float = 0.08
    huber_scale_m: float = 0.02
    minimum_face_interior_m: float = 0.02
    ambiguity_distance_m: float = 0.015
    minimum_correspondences: int = 24
    minimum_matched_fraction: float = 0.60
    minimum_normal_eigenvalue: float = 0.01
    maximum_iterations: int = 12

    # 功能：
    #   校验误差包络、对应点支持度及迭代预算，配置上界不能被布尔或非有限值绕过。
    # 输入：
    #   self：当前几何拟合限制。
    # 输出：
    #   None：约束成立时返回，非法配置抛出异常。
    def __post_init__(self):
        for name in (
            "maximum_association_distance_m",
            "maximum_translation_m",
            "maximum_residual_p95_m",
            "huber_scale_m",
            "minimum_face_interior_m",
            "ambiguity_distance_m",
            "minimum_matched_fraction",
            "minimum_normal_eigenvalue",
        ):
            _positive(getattr(self, name), name.upper())
        if (
            self.maximum_association_distance_m > 2
            or self.maximum_translation_m > 1
            or self.maximum_residual_p95_m > self.maximum_association_distance_m
            or self.huber_scale_m > self.maximum_residual_p95_m
            or self.minimum_matched_fraction > 1
            or self.minimum_normal_eigenvalue >= 1
            or type(self.minimum_correspondences) is not int
            or not 6 <= self.minimum_correspondences <= 512
            or type(self.maximum_iterations) is not int
            or not 1 <= self.maximum_iterations <= 24
        ):
            raise ValueError("MAP_ALIGNMENT_LIMITS_INVALID")


@dataclass(frozen=True)
class PlaneCorrespondences:
    normals: np.ndarray
    offsets: np.ndarray
    distances: np.ndarray
    valid: np.ndarray
    surface_ids: np.ndarray


class MapSurfaceIndex:
    """Compile immutable boxes, capped cylinders and spheres; 256 per query.

    Match only the FIRST surface along each measured ray. Nearest
    endpoint matching can jump to the back of thin walls/floors. Edges and
    competing first-hit faces are rejected instead of chosen arbitrarily.
    Unknown/ambiguous shapes are rejected, never silently removed from visibility.
    Primitive coverage alone does not establish map completeness or accuracy.
    """

    # 功能：
    #   将明确的盒体、封口圆柱及球体编译成只读数值索引，未知形状不能悄悄从可见性中消失。
    # 输入：
    #   primitives：当前地图最多两万个几何原语，坐标及尺寸必须显式给出。
    # 输出：
    #   self：拥有独立几何数组的索引；原语存在不证明地图完整或定位准确。
    def __init__(self, primitives):
        if not isinstance(primitives, list) or not 1 <= len(primitives) <= 20_000:
            raise ValueError("MAP_ALIGNMENT_MAP_INVALID")
        centers, halves, rotations, kinds = [], [], [], []
        for primitive in primitives:
            if not isinstance(primitive, dict):
                raise ValueError("MAP_ALIGNMENT_PRIMITIVE_INVALID")
            size_keys = [f"size_{a}" for a in "xyz"]
            lengths = [k for k in ("height_m", "length_m") if k in primitive]
            if any(k in primitive for k in size_keys):
                if not all(k in primitive for k in size_keys) or lengths or "radius_m" in primitive:
                    raise ValueError("MAP_ALIGNMENT_SHAPE_AMBIGUOUS")
                sizes, kind = [primitive[k] for k in size_keys], BOX
            elif "radius_m" in primitive and len(lengths) <= 1:
                radius = primitive["radius_m"]
                if not _finite(radius) or not 0 < radius <= 5e5:
                    raise ValueError("MAP_ALIGNMENT_RADIUS_INVALID")
                sizes = [radius * 2, radius * 2, primitive[lengths[0]] if lengths else radius * 2]
                kind = CYLINDER if lengths else SPHERE
            else:
                raise ValueError("MAP_ALIGNMENT_SHAPE_UNSUPPORTED")
            values = [primitive.get(f"center_{a}") for a in "xyz"]
            angles = [primitive.get(f"{a}_rad", 0.0) for a in ("roll", "pitch", "yaw")]
            for value in [*values, *sizes, *angles]:
                if not _finite(value):
                    raise ValueError("MAP_ALIGNMENT_PRIMITIVE_INVALID")
            if min(sizes) <= 0 or max(abs(v) for v in [*values, *sizes]) > 1e6:
                raise ValueError("MAP_ALIGNMENT_PRIMITIVE_INVALID")
            cr, cp, cy = np.cos(angles)
            sr, sp, sy = np.sin(angles)
            rotation = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            centers.append(values)
            halves.append(np.array(sizes) / 2)
            rotations.append(rotation)
            kinds.append(kind)
        self.centers, self.halves, self.rotations = map(np.asarray, (centers, halves, rotations))
        self.kinds = np.asarray(kinds)
        self.extents = np.einsum("bij,bj->bi", np.abs(self.rotations), self.halves)
        sphere, cylinder = self.kinds == SPHERE, self.kinds == CYLINDER
        self.extents[sphere] = self.halves[sphere]
        self.extents[cylinder] = (
            np.linalg.norm(self.rotations[cylinder, :, :2], axis=2) * self.halves[cylinder, 0, None]
            + np.abs(self.rotations[cylinder, :, 2]) * self.halves[cylinder, 2, None]
        )
        self.primitive_counts = {
            name: int(np.count_nonzero(self.kinds == kind))
            for name, kind in (("box", BOX), ("cylinder", CYLINDER), ("sphere", SPHERE))
        }
        for value in (self.centers, self.halves, self.rotations, self.extents, self.kinds):
            value.setflags(write=False)

    # 功能：
    #   沿测量射线选择首个可见表面，拒绝遮挡、内部起点、边缘和竞争面，不回退最近点猜测。
    # 输入：
    #   value：世界坐标系观测点。
    #   limits：已验证的关联距离与几何边界。
    #   sensor_origins_world_m：同帧各射线真实起点或单个共享起点。
    # 输出：
    #   matches：逐点法向、偏移、残差、有效性和表面身份；无邻近原语时全部无效。
    def match(
        self, value, limits: MapAlignmentLimits, *, sensor_origins_world_m
    ) -> PlaneCorrespondences:
        points = _points(value)
        if not isinstance(limits, MapAlignmentLimits):
            raise ValueError("MAP_ALIGNMENT_LIMITS_INVALID")
        limits = replace(limits)
        origins = _origins(sensor_origins_world_m, points)
        if np.any(np.linalg.norm(points - origins, axis=1) < 1e-6):
            raise ValueError("MAP_ALIGNMENT_ZERO_LENGTH_RAY")
        padding = limits.maximum_association_distance_m
        low = np.minimum(points.min(axis=0), origins.min(axis=0)) - padding
        high = np.maximum(points.max(axis=0), origins.max(axis=0)) + padding
        nearby = np.all(
            (self.centers + self.extents >= low) & (self.centers - self.extents <= high), axis=1
        )
        indices = np.flatnonzero(nearby)
        if len(indices) > 256:
            raise ValueError("MAP_ALIGNMENT_LOCAL_MAP_BUDGET_EXCEEDED")
        if not len(indices):
            return PlaneCorrespondences(
                np.zeros_like(points),
                np.zeros(len(points)),
                np.full(len(points), np.inf),
                np.zeros(len(points), bool),
                np.full(len(points), -1, int),
            )
        centers, halves, rotations = (
            value[indices] for value in (self.centers, self.halves, self.rotations)
        )
        local_origin = np.einsum("nbi,bij->nbj", origins[:, None] - centers, rotations)
        world_direction = points - origins
        world_direction /= np.linalg.norm(world_direction, axis=1, keepdims=True)
        direction = np.einsum("ni,bij->nbj", world_direction, rotations)
        enter, local_normals, interiors, face_ids, inside, planar = first_surface_hits(
            local_origin, direction, halves, self.kinds[indices], limits.minimum_face_interior_m
        )
        inside = np.any(inside, axis=1)
        normals = np.einsum("nbj,bij->nbi", local_normals, rotations)
        intersections = (
            local_origin + direction * np.where(np.isfinite(enter), enter, 0.0)[..., None]
        )
        offsets = np.einsum("nbi,bi->nb", normals, centers) + np.einsum(
            "nbi,nbi->nb", local_normals, intersections
        )
        row = np.arange(len(points))
        best = np.argmin(enter, axis=1)
        best_normal, best_offset = normals[row, best], offsets[row, best]
        best_enter = enter[row, best]
        interior = interiors[row, best]
        best_distance = np.abs(np.einsum("ni,ni->n", best_normal, points) - best_offset)
        dot = np.einsum("nbi,ni->nb", normals, best_normal)
        same_plane = (
            planar
            & planar[row, best, None]
            & (np.abs(dot) > 1.0 - 1e-9)
            & (np.abs(offsets - np.sign(dot) * best_offset[:, None]) < 1e-6)
        )
        # The selected curved surface is not its own occluding competitor.
        same_plane[row, best] = True
        competing = np.min(np.where(same_plane, np.inf, enter), axis=1)
        separation = competing - np.where(np.isfinite(best_enter), best_enter, 0.0)
        valid = (
            interior
            & ~inside
            & np.isfinite(best_enter)
            & (best_distance <= padding)
            & (separation > limits.ambiguity_distance_m)
        )
        surfaces = indices[best] * 6 + face_ids[row, best]
        matches = PlaneCorrespondences(best_normal, best_offset, best_distance, valid, surfaces)
        return matches


@dataclass(frozen=True)
class MapTranslationFit:
    usable_candidate: bool
    correction_world_m: tuple[float, float, float]
    observed_translation_rank: int
    unobserved_directions_world: tuple[tuple[float, float, float], ...]
    matched_count: int
    residual_p95_m: float | None
    iterations: int
    issue: str | None
    retired_correspondence_count: int = 0
    attitude_held_fixed: bool = True
    covariance_qualified: bool = False


# 功能：
#   1. 在姿态固定前提下用鲁棒迭代计算真实点到面平移修正，不生成导航指令。
#   2. 每轮求当前可观测子空间的总修正，不能累积已失去约束的方向。
#   3. 收敛后重查可见表面及对应关系，失效对应点只退役不复活；不虚构协方差。
# 输入：
#   value：当前世界坐标系观测点阵。
#   index：所选地图的真实表面索引。
#   sensor_origins_world_m：观测时射线起点。
#   limits：可选的严格迭代及修正包络。
# 输出：
#   fit：平移候选、可观测方向、残差及拒绝原因，失败时不输出有效修正。
def fit_map_translation(
    value,
    index: MapSurfaceIndex,
    *,
    sensor_origins_world_m,
    limits: MapAlignmentLimits | None = None,
) -> MapTranslationFit:
    points = _points(value)
    if limits is None:
        limits = MapAlignmentLimits()
    if not isinstance(limits, MapAlignmentLimits):
        raise ValueError("MAP_ALIGNMENT_LIMITS_INVALID")
    limits = replace(limits)
    if not isinstance(index, MapSurfaceIndex):
        raise ValueError("MAP_ALIGNMENT_MAP_INVALID")
    correction = np.zeros(3)
    rank, null, count, residual, iteration, retired = 0, np.eye(3), 0, None, 0, 0

    # 功能：
    #   输出当前实际可观测方向，失败修正清零，避免调用方误用最后一轮未获准候选。
    # 输入：
    #   issue：拒绝原因或成功时的 None。
    # 输出：
    #   fit：包含支持度和退役对应点数的拟合记录。
    def result(issue):
        fit = MapTranslationFit(
            issue is None,
            tuple(correction if issue is None else np.zeros(3)),
            rank,
            tuple(tuple(v) for v in null),
            count,
            residual,
            iteration,
            issue,
            retired,
        )
        return fit

    # For a fixed correspondence set translation fitting is linear apart
    # from robust weights. Do not repeat spatial lookup in every IRLS step.
    # Re-associate at convergence and solve again if the face set changed.
    origins = _origins(sensor_origins_world_m, points)
    matches = index.match(points, limits, sensor_origins_world_m=origins)
    # A boundary correspondence that becomes geometrically invalid may not
    # alternate back into the same solve. Retire it for this solve only; every
    # frame starts a new active set. Minimum support/rank/residual checks remain.
    eligible = matches.valid.copy()
    for step in range(limits.maximum_iterations):
        iteration = step + 1
        count = int(np.count_nonzero(matches.valid))
        if count < max(
            limits.minimum_correspondences, len(points) * limits.minimum_matched_fraction
        ):
            return result("MAP_ALIGNMENT_CORRESPONDENCES_INSUFFICIENT")
        normals = matches.normals[matches.valid]
        rhs = matches.offsets[matches.valid] - np.einsum("ni,ni->n", normals, points[matches.valid])
        errors = normals @ correction - rhs
        weights = np.minimum(1.0, limits.huber_scale_m / np.maximum(np.abs(errors), 1e-12))
        # Normalizing weights makes the directional rank independent of a
        # uniformly duplicated point set. It is NOT a covariance normalization.
        weights /= np.sum(weights)
        gram = normals.T @ (normals * weights[:, None])
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        observable = eigenvalues >= limits.minimum_normal_eigenvalue
        rank, null = int(np.count_nonzero(observable)), eigenvectors[:, ~observable].T
        if not rank:
            return result("MAP_ALIGNMENT_NO_OBSERVABLE_TRANSLATION")
        basis = eigenvectors[:, observable]
        proposal = basis @ ((basis.T @ (normals.T @ (weights * rhs))) / eigenvalues[observable])
        # Once enough residuals lie in Huber's quadratic region, a Newton
        # step can finish cases where IRLS creeps near the clipping boundary.
        # Use it only in the observed subspace and only when it lowers the
        # SAME robust objective more than IRLS; otherwise retain IRLS.
        quadratic = np.abs(errors) < limits.huber_scale_m
        jacobian = normals[quadratic] @ basis
        curvature = jacobian.T @ jacobian
        if len(jacobian) >= rank and np.linalg.eigvalsh(curvature).min() > 1e-8:
            gradient = normals.T @ np.clip(errors, -limits.huber_scale_m, limits.huber_scale_m)
            newton = basis @ (basis.T @ correction - np.linalg.solve(curvature, basis.T @ gradient))

            if np.linalg.norm(newton) <= limits.maximum_translation_m and _huber_cost(
                normals @ newton - rhs, limits.huber_scale_m
            ) <= _huber_cost(normals @ proposal - rhs, limits.huber_scale_m):
                proposal = newton
        if np.linalg.norm(proposal) > limits.maximum_translation_m:
            return result("MAP_ALIGNMENT_CORRECTION_OUTSIDE_ENVELOPE")
        # A full-gradient line can zig-zag between a stiff wall-normal axis
        # and a clipped, nearly linear floor axis. A bounded coordinate sweep
        # in the observed basis avoids that without loosening convergence.
        for axis in basis.T:
            errors_at_proposal = normals @ proposal - rhs
            derivative = (normals @ axis) @ np.clip(
                errors_at_proposal, -limits.huber_scale_m, limits.huber_scale_m
            )
            probe = proposal - np.sign(derivative) * 1e-4 * axis
            candidate = _line_refinement(proposal, probe, normals, rhs, limits)
            if np.linalg.norm(candidate) <= limits.maximum_translation_m and _huber_cost(
                normals @ candidate - rhs, limits.huber_scale_m
            ) < _huber_cost(errors_at_proposal, limits.huber_scale_m):
                proposal = candidate
        converged = np.linalg.norm(proposal - correction) <= 1e-5
        correction = proposal
        if converged:
            # Re-evaluate after the last update. Old correspondences must not
            # claim a match after a point crosses a face boundary.
            final = index.match(
                points + correction, limits, sensor_origins_world_m=origins + correction
            )
            retired += int(np.count_nonzero(eligible & ~final.valid))
            eligible &= final.valid
            final = replace(final, valid=final.valid & eligible)
            if not np.array_equal(final.valid, matches.valid):
                matches = final
                continue
            if not np.array_equal(
                final.surface_ids[final.valid], matches.surface_ids[matches.valid]
            ):
                matches = final
                continue
            # A sphere/cylinder side can keep its primitive ID while its local
            # tangent plane changes. Re-solve until the geometry stabilizes too.
            if not np.allclose(
                final.normals[final.valid], normals, rtol=0.0, atol=1e-7
            ) or not np.allclose(
                final.offsets[final.valid], matches.offsets[matches.valid], rtol=0.0, atol=1e-7
            ):
                matches = final
                continue
            residual = float(np.quantile(final.distances[final.valid], 0.95))
            if residual > limits.maximum_residual_p95_m:
                return result("MAP_ALIGNMENT_RESIDUAL_EXCEEDS_ENVELOPE")
            return result(None)
    return result("MAP_ALIGNMENT_DID_NOT_CONVERGE")
