"""Analytic first-visible surfaces for bounded local map registration.

Rays and geometry are in primitive-local coordinates, in metres. All surfaces,
including unusable rims/tangencies, participate in occlusion. No mesh sampling,
learned confidence, pose fitting, or flight authority is implemented here.
"""

from __future__ import annotations

import numpy as np

BOX, CYLINDER, SPHERE, MESH = 0, 1, 2, 3


# 功能：以有限批次求三角网格首个射线交点；背面、折边及内部起点不作为定位支持，但仍遮挡。
# 输入：origins、directions：同一网格局部坐标中的射线；triangles：显式三角形；margin：边距。
# 输出：首交距离、面法向、有效面标记、三角形身份与背面/内部标记。
def mesh_surface_hits(
    origins, directions, triangles, margin, *, double_sided=False, winding_sign=1.0
):
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.cross(edge1, edge2)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    count = len(origins)
    result = (
        np.full(count, np.inf),
        np.zeros((count, 3)),
        np.zeros(count, bool),
        np.zeros(count, int),
        np.zeros(count, bool),
    )
    for start in range(0, count, 32):
        o, d = origins[start : start + 32], directions[start : start + 32]
        h = np.cross(d[:, None], edge2)
        det = np.einsum("tj,ntj->nt", edge1, h)
        divisor = np.where(np.abs(det) > 1e-12, det, 1.0)
        s = o[:, None] - triangles[:, 0]
        u = np.einsum("ntj,ntj->nt", s, h) / divisor
        q = np.cross(s, edge1)
        v = np.einsum("nj,ntj->nt", d, q) / divisor
        t = np.einsum("tj,ntj->nt", edge2, q) / divisor
        hit = (
            (np.abs(det) > 1e-12)
            & (u >= -1e-10)
            & (v >= -1e-10)
            & (u + v <= 1 + 1e-10)
            & (t > 1e-9)
        )
        physical_times = np.where(hit, t, np.inf)
        physical_best = np.argmin(physical_times, axis=1)
        # Physical inside testing is separate from the optical culling rule.
        # A reversed single-sided shell can show its far wall to an outside
        # camera; that does not put the camera physically inside the shell.
        physical_inside = np.einsum("nj,nj->n", normal[physical_best] * winding_sign, d) > 1e-12
        physical_inside &= np.isfinite(np.min(physical_times, axis=1))
        visible = hit if double_sided else hit & (det > 1e-12)
        t = np.where(visible, t, np.inf)
        best = np.argmin(t, axis=1)
        row = np.arange(len(o))
        depth = t[row, best]
        normals = normal[best]
        if double_sided:
            normals = (
                normals * np.where(np.einsum("nj,nj->n", normals, d) > 0.0, -1.0, 1.0)[:, None]
            )
        front = -np.einsum("nj,nj->n", normals, d) > 0.02
        # All facets of the actual mesh remain occluders. Reject a first hit
        # if a nearby competing facet has a materially different normal.
        dot = np.einsum("tj,nj->nt", normal, normals)
        crease = np.any(
            visible
            & (np.abs(t - np.where(np.isfinite(depth), depth, 0.0)[:, None]) < margin)
            & (np.abs(dot) < np.cos(np.deg2rad(20))),
            axis=1,
        )
        for destination, value in zip(
            result,
            (depth, normals, front & ~crease & np.isfinite(depth), best, physical_inside),
            strict=True,
        ):
            destination[start : start + len(o)] = value
    return result


# 功能：
#   稳定求解球面或圆柱侧面的正射线交点，避免相近大数相减丢失较近根；无侧交点返回无穷远。
# 输入：
#   origin：图元局部坐标中的射线原点批次。
#   direction：对应的二维或三维方向批次。
#   radius：可广播到射线批次的图元半径。
# 输出：
#   result：较近、较远两个正交点参数数组。
def _quadratic_roots(origin, direction, radius):
    a = np.sum(direction * direction, axis=-1)
    b = np.sum(origin * direction, axis=-1)  # Half of the linear coefficient.
    c = np.sum(origin * origin, axis=-1) - radius * radius
    divisor = np.where(a > 1e-24, a, 1.0)
    closest = origin - (b / divisor)[..., None] * direction
    discriminant = a * (radius * radius - np.sum(closest * closest, axis=-1))
    real = (a > 1e-24) & (discriminant >= 0.0)
    q = -b - np.copysign(np.sqrt(np.maximum(discriminant, 0.0)), b)
    safe_q = np.where(np.abs(q) > 1e-24, q, 1.0)
    first, second = q / divisor, c / safe_q
    repeated = np.abs(q) <= 1e-24
    first = np.where(repeated, -b / divisor, first)
    second = np.where(repeated, first, second)
    result = tuple(
        np.where(real & (t > 0.0), t, np.inf)
        for t in (np.minimum(first, second), np.maximum(first, second))
    )
    return result


# 功能：
#   用三轴平板求交计算盒表面，保留边缘遮挡，同时标记不宜用于配准的边缘或内部起点。
# 输入：
#   origin：盒局部坐标中的射线原点，最后一维为 xyz。
#   direction：同坐标系的射线方向。
#   half：盒的三个半尺寸。
#   margin：配准对应点须离边缘留出的距离。
# 输出：
#   result：进入参数、法向、面内部标记、面标识和起点位于盒内的标记。
def _box_hits(origin, direction, half, margin):
    # 分轴批量计算避免反复建立 N×M×3 中间张量；仍逐轴求交，不能跳过
    # 遮挡面。严格大于比较保留原 argmax 的 x→y→z 同值优先顺序。
    entries, exits, outside, contained = [], [], [], []
    for axis in range(3):
        o, d, h = origin[..., axis], direction[..., axis], half[..., axis]
        parallel = np.abs(d) < 1e-12
        divisor = np.where(parallel, 1., d)
        first, second = (-h - o) / divisor, (h - o) / divisor
        entries.append(np.where(parallel, -np.inf, np.minimum(first, second)))
        exits.append(np.where(parallel, np.inf, np.maximum(first, second)))
        outside.append(parallel & (np.abs(o) > h))
        contained.append(np.abs(o) <= h)
    enter = np.maximum(np.maximum(entries[0], entries[1]), entries[2])
    leave = np.minimum(np.minimum(exits[0], exits[1]), exits[2])
    hit = ((leave >= np.maximum(enter, 0.)) & (enter > 0.)
           & ~(outside[0] | outside[1] | outside[2]))
    axis = np.where(entries[1] > entries[0], 1, 0)
    axis = np.where(entries[2] > np.maximum(entries[0], entries[1]), 2, axis)
    component = np.where(axis == 0, direction[..., 0],
                         np.where(axis == 1, direction[..., 1], direction[..., 2]))
    sign = -np.sign(component)
    normals = np.zeros_like(origin)
    interior = np.ones(enter.shape, dtype=bool)
    distance = np.where(hit, enter, 0.)
    for dim in range(3):
        normals[..., dim] = np.where(axis == dim, sign, 0.)
        point = origin[..., dim] + direction[..., dim] * distance
        interior &= (axis == dim) | (half[..., dim] - np.abs(point) > margin)
    enter = np.where(hit, enter, np.inf)
    inside = contained[0] & contained[1] & contained[2]
    result = enter, normals, interior, axis * 2 + (sign > 0), inside
    return result


# 功能：
#   计算球面首个正交点与法向；切线接触仍遮挡，但不能作为稳定配准对应。
# 输入：
#   origin：球局部坐标中的射线原点批次。
#   direction：归一化的射线方向批次。
#   half：第一个分量保存球半径的半尺寸数组。
#   margin：统一求交接口参数，球面无多面体边缘，不使用该边距。
# 输出：
#   result：交点参数、法向、稳定对应标记、曲面标识和内部起点标记。
def _sphere_hits(origin, direction, half, margin):
    radius = half[..., 0]
    first, second = _quadratic_roots(origin, direction, radius)
    enter = np.minimum(first, second)
    point = origin + direction * np.where(np.isfinite(enter), enter, 0.0)[..., None]
    normals = point / np.maximum(np.linalg.norm(point, axis=-1, keepdims=True), 1e-15)
    # Tangencies still occlude, but their vanishing normal sensitivity cannot
    # be treated as a well-supported correspondence.
    interior = -np.sum(normals * direction, axis=-1) > 0.02
    inside = np.sum(origin * origin, axis=-1) <= radius * radius
    result = enter, normals, interior, np.zeros(enter.shape, dtype=int), inside
    return result


# 功能：
#   比较圆柱侧面与上下端盖的交点，保留最先发生的遮挡，并区分曲面、端盖和边缘。
# 输入：
#   origin：以圆柱中心为原点、z 为轴线的射线原点。
#   direction：相同坐标系的归一化射线方向。
#   half：第一个分量为半径，第三个分量为半高。
#   margin：用于配准的交点与圆柱边缘之间的最小距离。
# 输出：
#   result：交点参数、法向、稳定对应标记、曲面标识和内部起点标记。
def _cylinder_hits(origin, direction, half, margin):
    radius, height = half[..., 0], half[..., 2]
    first, second = _quadratic_roots(origin[..., :2], direction[..., :2], radius)
    side_times = []
    for t in (first, second):
        z = origin[..., 2] + np.where(np.isfinite(t), t, 0.0) * direction[..., 2]
        side_times.append(np.where(np.abs(z) <= height, t, np.inf))
    dz = direction[..., 2]
    caps = []
    for sign in (-1.0, 1.0):
        t = (sign * height - origin[..., 2]) / np.where(np.abs(dz) > 1e-12, dz, 1.0)
        xy = origin[..., :2] + t[..., None] * direction[..., :2]
        hit = (np.abs(dz) > 1e-12) & (t > 0.0) & (np.sum(xy * xy, axis=-1) <= radius * radius)
        caps.append(np.where(hit, t, np.inf))
    candidates = np.stack((*side_times, *caps), axis=-1)
    selected = np.argmin(candidates, axis=-1)
    enter = np.min(candidates, axis=-1)
    point = origin + direction * np.where(np.isfinite(enter), enter, 0.0)[..., None]
    radial = np.linalg.norm(point[..., :2], axis=-1)
    normals = np.zeros_like(point)
    normals[..., :2] = point[..., :2] / np.maximum(radial[..., None], 1e-15)
    cap = selected >= 2
    normals[cap] = 0.0
    normals[..., 2] = np.where(cap, np.where(selected == 2, -1.0, 1.0), 0.0)
    interior = np.where(cap, radius - radial, height - np.abs(point[..., 2])) > margin
    interior &= -np.sum(normals * direction, axis=-1) > 0.02
    inside = (np.sum(origin[..., :2] ** 2, axis=-1) <= radius * radius) & (
        np.abs(origin[..., 2]) <= height
    )
    # One side ID, separate lower/upper caps; curved normals are checked again
    # by the solver rather than pretending a constant face ID implies a plane.
    surface = np.where(cap, selected - 1, 0)
    result = enter, normals, interior, surface, inside
    return result


# 功能：
#   1. 分派盒、圆柱和球的解析求交，并汇总曲面类型，不做姿态拟合或运动授权。
#   2. 保留不适合配准的遮挡面与内部起点标记，交由调用方筛选，而非把未匹配当作自由空间。
# 输入：
#   origins：调用方已校验并限量的原点张量，形状为射线数×图元数×3。
#   directions：同形状、图元局部坐标中的归一化方向张量。
#   halves：图元数×3 的半尺寸数组。
#   kinds：盒、圆柱或球的图元类别数组。
#   margin：配准对应点的表面边距，单位米。
# 输出：
#   result：交点参数、法向、内部面标记、曲面标识、内部起点和平面标记六个数组。
def first_surface_hits(origins, directions, halves, kinds, margin):
    shape = origins.shape[:2]
    enter, normals = np.full(shape, np.inf), np.zeros_like(origins)
    interior, inside = np.zeros(shape, bool), np.zeros(shape, bool)
    surfaces = np.zeros(shape, int)
    for kind, kernel in ((BOX, _box_hits), (CYLINDER, _cylinder_hits), (SPHERE, _sphere_hits)):
        selected = kinds == kind
        if np.any(selected):
            values = kernel(origins[:, selected], directions[:, selected], halves[selected], margin)
            for destination, value in zip(
                (enter, normals, interior, surfaces, inside), values, strict=True
            ):
                destination[:, selected] = value
    planar = (kinds == BOX)[None, :] | ((kinds == CYLINDER)[None, :] & (surfaces > 0))
    result = enter, normals, interior, surfaces, inside, planar
    return result
