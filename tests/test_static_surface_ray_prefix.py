"""Observed ray prefixes retain every front occluder without tracing beyond them."""

import numpy as np
import pytest

import dronedream_agent_core.map_surface_geometry as surface_geometry
from dronedream_agent_core.local_map_alignment import MapAlignmentLimits, MapSurfaceIndex
from dronedream_agent_core.map_surface_geometry import (
    MeshQueryBudget,
    MeshRayIndex,
    mesh_surface_hits,
)


# 功能：生成一串完整闭合小盒，模拟同一树冠块中沿射线方向的许多远处表面。
def _boxes(centers):
    vertices = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
    faces = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                      (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                      (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    return np.concatenate([(vertices + [x, 0., 0.])[faces] for x in centers])


def test_batched_slabs_keep_all_surface_and_inside_results(monkeypatch):
    triangles = _boxes(range(3, 403, 4))
    index = MeshRayIndex(triangles)
    rng = np.random.default_rng(614)
    origins = rng.uniform([-2., -2., -2.], [5., 2., 2.], size=(128, 3))
    directions = rng.normal(size=(128, 3))
    directions[:, 0] = np.abs(directions[:, 0])
    directions[::5] = [1., 0., 0.]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    results = []
    for batch in (1, 128):
        monkeypatch.setattr(surface_geometry, "_MESH_RAY_BATCH", batch)
        results.append(mesh_surface_hits(origins, directions, triangles, .02, acceleration=index))
    for scalar, batched in zip(*results, strict=True):
        np.testing.assert_array_equal(scalar, batched)


# 功能：首交与所有遮挡完全不变，同时避免遍历观测端点之后的大量真实面。
def test_prefix_keeps_nearest_occluder_and_reduces_real_triangle_work():
    triangles = _boxes(range(3, 403, 4))
    index = MeshRayIndex(triangles)
    origins, directions = np.array([[0., .2, .1]]), np.array([[1., 0., 0.]])
    full_budget, prefix_budget = MeshQueryBudget(), MeshQueryBudget()
    full = mesh_surface_hits(origins, directions, triangles, .02,
                             acceleration=index, work_budget=full_budget)
    prefix = mesh_surface_hits(origins, directions, triangles, .02, acceleration=index,
                               work_budget=prefix_budget, maximum_ranges=np.array([3.15]))
    for left, right in zip(full, prefix, strict=True):
        np.testing.assert_array_equal(left, right)
    assert prefix[0][0] == pytest.approx(2.)
    # The unlimited query also stops at its already-proved nearest surface.
    assert prefix_budget.triangle_pairs <= full_budget.triangle_pairs < len(triangles) / 4
    # A window around the endpoint would have omitted this nearer occluder.
    point = np.array([[100., .2, .1]])
    surface = MapSurfaceIndex([dict(center_x=0., center_y=0., center_z=0.,
                                    triangles=triangles.tolist())])
    for bounded in (False, True):
        match = surface.match(point, MapAlignmentLimits(), sensor_origins_world_m=origins,
                              observed_range_prefix=bounded)
        assert match.surface_ranges_m[0] == pytest.approx(2.)
        assert not match.valid[0]


# 功能：端点附近竞争表面和内部起点保持原有拒绝语义，不能用裁剪取得静态身份。
@pytest.mark.parametrize("point,origin", [([2., .2, .1], [0., .2, .1]),
                                        ([4., .2, .1], [3., .2, .1])])
def test_prefix_preserves_ambiguity_and_inside_rejection(point, origin):
    meshes = [dict(center_x=0., center_y=0., center_z=0., triangles=_boxes([x]).tolist())
              for x in (3., 3.01)]
    index = MapSurfaceIndex(meshes)
    full = index.match([point], MapAlignmentLimits(), sensor_origins_world_m=origin)
    bounded = index.match([point], MapAlignmentLimits(), sensor_origins_world_m=origin,
                          observed_range_prefix=True)
    assert not full.valid[0]
    np.testing.assert_array_equal(full.valid, bounded.valid)
    np.testing.assert_array_equal(full.rejection_reasons, bounded.rejection_reasons)


# 功能：外层壳出口远在端点之后时，内部源不能因近处另一个面获得虚假静态身份。
def test_prefix_does_not_hide_enclosing_shell_exit_after_endpoint():
    index = MapSurfaceIndex([
        dict(center_x=0., center_y=0., center_z=0., triangles=_boxes([3.]).tolist()),
        dict(center_x=0., center_y=0., center_z=0., triangles=(_boxes([0.]) * 10).tolist()),
    ])
    for bounded in (False, True):
        result = index.match([[2., .2, .1]], MapAlignmentLimits(),
                             sensor_origins_world_m=[[0., .2, .1]], observed_range_prefix=bounded)
        assert not result.valid[0]
        assert result.rejection_reasons[0] == 'origin_inside_or_backface'


# 功能：前缀参数拒绝布尔伪数、负数、非有限数及错误维度，不悄悄省略几何。
@pytest.mark.parametrize("ranges", [[True], [-1.], [float('nan')], [1., 2.]])
def test_invalid_prefix_ranges_are_rejected(ranges):
    with pytest.raises(ValueError, match="MAP_ALIGNMENT_RAY_RANGE_INVALID"):
        mesh_surface_hits(np.array([[0., .2, .1]]), np.array([[1., 0., 0.]]), _boxes([3.]),
                          .02, maximum_ranges=ranges)


# 功能：在新批次边界与不整批上逐项对比旧 32 射线实现，保留首交、背面、内部与折边语义。
@pytest.mark.parametrize("count", [127, 128, 129, 301])
@pytest.mark.parametrize("reversed_shell,double_sided", [(False, False), (True, False),
                                                       (False, True)])
def test_larger_ray_batches_preserve_exact_results_and_work_budget(
    monkeypatch, count, reversed_shell, double_sided,
):
    triangles = _boxes(range(3, 83, 4))
    if reversed_shell:
        triangles = triangles[:, ::-1].copy()
    index = MeshRayIndex(triangles)
    # Front face, inside source, two-front-face crease, miss, and reverse-direction hit.
    origins = np.array([[0., .2, .1], [3., .2, .1], [0., 3., .1],
                        [0., 5., .1], [84., .2, .1]])[np.arange(count) % 5]
    directions = np.array([[1., 0., 0.], [1., 0., 0.], [1., -1., 0.],
                           [1., 0., 0.], [-1., 0., 0.]])[np.arange(count) % 5]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    configured_batch = surface_geometry._MESH_RAY_BATCH
    assert configured_batch == 128
    for ranges in (None, np.full(count, 12.)):
        results, budgets = [], []
        for batch in (32, configured_batch):
            monkeypatch.setattr(surface_geometry, "_MESH_RAY_BATCH", batch)
            budget = MeshQueryBudget()
            results.append(mesh_surface_hits(
                origins, directions, triangles, .02, acceleration=index,
                double_sided=double_sided, winding_sign=-1. if reversed_shell else 1.,
                work_budget=budget, maximum_ranges=ranges,
            ))
            budgets.append(budget)
        for old, new in zip(*results, strict=True):
            assert len(new) == count  # Includes every ray after a partial final batch.
            np.testing.assert_array_equal(old, new)
        assert budgets[0] == budgets[1]
        assert budgets[1].maximum_triangle_pairs == 8_388_608
        assert budgets[1].maximum_bound_pairs == 2_097_152
        if not reversed_shell:
            assert results[1][4][1]  # An inside source remains inside.
            assert not results[1][2][2]  # A crease remains unsuitable as a map support.


# 功能：保留优化前逐射线展开的求交作为差分参照，不共用单起点计算分支。
def _expanded_leaf_hits(origins, directions, index, faces, double_sided, budget):
    budget.charge(len(origins) * len(faces))
    edge1, edge2 = index.edge1[faces], index.edge2[faces]
    h = np.cross(directions[:, None], edge2)
    det = np.einsum("tj,ntj->nt", edge1, h)
    divisor = np.where(np.abs(det) > 1e-12, det, 1.0)
    s = origins[:, None] - index.triangles[faces, 0]
    u = np.einsum("ntj,ntj->nt", s, h) / divisor
    q = np.cross(s, edge1)
    v = np.einsum("nj,ntj->nt", directions, q) / divisor
    t = np.einsum("tj,ntj->nt", edge2, q) / divisor
    hit = ((np.abs(det) > 1e-12) & (u >= -1e-10) & (v >= -1e-10)
           & (u + v <= 1 + 1e-10) & (t > 1e-9))
    return np.where(hit, t, np.inf), hit if double_sided else hit & (det > 1e-12)


# 功能：共享与仅相差一 ULP 的起点都须保留完整首交、折边、内部和背面结果。
@pytest.mark.parametrize("count", [127, 128, 129, 301])
@pytest.mark.parametrize("origin_x,mixed_origin", [(0., False), (3., False), (0., True)])
def test_shared_camera_origin_matches_expanded_intersections(monkeypatch, count,
                                                            origin_x, mixed_origin):
    triangles = _boxes(range(3, 83, 4))
    origins = np.tile([origin_x, .2, .1], (count, 1))
    if mixed_origin:
        origins[::2, 1] = np.nextafter(origins[::2, 1], np.inf)
    # Includes exact edges, near-tangent rays, misses and reversed rays.
    directions = np.array([[1., 0., 0.], [2., .8, .9], [1., 1e-13, 0.],
                           [1., 5., .1], [-1., 0., 0.]])[np.arange(count) % 5]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    optimized = surface_geometry._mesh_leaf_hits
    for winding_sign, double_sided in ((1., False), (-1., False), (-1., True)):
        geometry = triangles if winding_sign == 1. else triangles[:, ::-1].copy()
        index = MeshRayIndex(geometry)
        results, budgets = [], []
        for implementation in (_expanded_leaf_hits, optimized):
            monkeypatch.setattr(surface_geometry, "_mesh_leaf_hits", implementation)
            budget = MeshQueryBudget()
            results.append(mesh_surface_hits(
                origins, directions, geometry, .02, acceleration=index,
                double_sided=double_sided, winding_sign=winding_sign, work_budget=budget,
            ))
            budgets.append(budget)
        for expanded, shared in zip(*results, strict=True):
            np.testing.assert_array_equal(expanded, shared)
        assert budgets[0] == budgets[1]
