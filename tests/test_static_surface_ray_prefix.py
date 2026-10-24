"""Observed ray prefixes retain every front occluder without tracing beyond them."""

import numpy as np
import pytest

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
