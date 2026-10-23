"""Whole-map storage must not become whole-map work for a local camera frame."""

import pickle

import numpy as np
import pytest

from dronedream_agent_core.local_map_alignment import MapAlignmentLimits, MapSurfaceIndex
from dronedream_agent_core.map_surface_geometry import (
    MAX_INDEX_MESH_TRIANGLES,
    MAX_MESH_TRIANGLES,
    MeshQueryBudget,
    MeshRayIndex,
    mesh_surface_hits,
)


# 功能：构造绕序一致的完整立方体；输入：无；输出：12 个真实三角面。
def cube():
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return vertices[np.asarray(faces)]


# 功能：验证超过旧全局预算的精确分块索引只查询附近完整块，且 spawn 序列化不复制成 JSON。
# 输入：pytest 替身；输出：局部首交和 pickle 往返保持一致，远处网格不参与求交。
def test_large_map_queries_only_local_chunks_and_survives_spawn_pickle(monkeypatch):
    import dronedream_agent_core.local_map_alignment as alignment

    raw = np.tile(cube(), (540, 1, 1)).tolist()
    primitives = [dict(center_x=1000.0 + n * 20, center_y=0.0, center_z=0.0,
                       triangles=raw) for n in range(11)]
    primitives.append(dict(center_x=4.0, center_y=0.0, center_z=0.0,
                           triangles=cube().tolist()))
    index = MapSurfaceIndex(primitives)
    assert sum(len(mesh) for mesh in index.meshes.values()) > 65_536
    sizes = []
    original = alignment.mesh_surface_hits

    # 功能：记录实际到达精确求交的面数；输入：生产参数；输出：未改变的生产求交结果。
    def measured(*args, **kwargs):
        sizes.append(len(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment, "mesh_surface_hits", measured)
    result = index.match([[3.0, 0.2, 0.1]], MapAlignmentLimits(),
                         sensor_origins_world_m=[0, 0.2, 0.1])
    assert result.valid[0]
    assert sizes == [12]
    wire = pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(wire) < 20 * 1024 * 1024
    restored = pickle.loads(wire)
    assert all(restored.meshes[key] is value.triangles
               for key, value in restored.mesh_acceleration.items())
    again = restored.match([[3.0, 0.2, 0.1]], MapAlignmentLimits(),
                           sensor_origins_world_m=[0, 0.2, 0.1])
    np.testing.assert_array_equal(again.valid, result.valid)
    np.testing.assert_allclose(again.normals, result.normals)
    np.testing.assert_allclose(again.surface_ranges_m, result.surface_ranges_m)


# 功能：证明 BVH 对稀疏局部射线裁掉远面，实际射线×面计算低于全量广播。
# 输入：无；输出：8192 面中局部射线保留正确最近表面，工作量不随全图广播。
def test_local_triangle_query_is_spatially_bounded():
    triangles = np.array([[[5., n * 2 - .5, -.5], [5., n * 2 - .5, .5],
                           [5., n * 2 + .5, -.5]] for n in range(8192)])
    triangles.setflags(write=False)
    index = MeshRayIndex(triangles)
    origins = np.tile([0., -.2, -.2], (32, 1))
    directions = np.tile([1., 0., 0.], (32, 1))
    budget = MeshQueryBudget()
    hits = mesh_surface_hits(origins, directions, triangles, .01,
                             acceleration=index, work_budget=budget)
    np.testing.assert_allclose(hits[0], 5.)
    assert hits[2].all()
    assert budget.triangle_pairs <= 32 * 64 * 2
    assert budget.triangle_pairs < len(triangles) * len(origins) // 8


# 功能：共享预算跨多个网格累计，超额在昂贵三角广播前明确拒绝。
# 输入：无；输出：两个分别合法的查询不能重置同一帧的工作上限。
def test_query_budget_is_shared_and_still_rejects_excess_work():
    triangles = cube()
    index = MeshRayIndex(triangles)
    budget = MeshQueryBudget(maximum_triangle_pairs=36)
    origin, direction = np.array([[-3., .2, .1]]), np.array([[1., 0., 0.]])
    mesh_surface_hits(origin, direction, triangles, .02, acceleration=index, work_budget=budget)
    with pytest.raises(ValueError, match="LOCAL_MESH_QUERY_BUDGET_EXCEEDED"):
        mesh_surface_hits(origin, direction, triangles, .02,
                          acceleration=index, work_budget=budget)
    with pytest.raises(ValueError, match="LOCAL_MESH_QUERY_BUDGET_EXCEEDED"):
        mesh_surface_hits(origin, direction, triangles, .02,
                          acceleration=index, work_budget=MeshQueryBudget(maximum_bound_pairs=0))


# 功能：全图存储仍有限，且超额检查先于任何数值数组和空间树分配。
# 输入：pytest 替身；输出：旧预算提升后仍拒绝大实例集和单块过大。
def test_global_storage_budget_rejects_before_mesh_allocation(monkeypatch):
    import dronedream_agent_core.local_map_alignment as alignment

    # 功能：断言禁止分配；输入：任意参数；输出：若触发即测试失败。
    def forbidden(*args, **kwargs):
        raise AssertionError("budget must precede BVH allocation")

    monkeypatch.setattr(alignment, "MeshRayIndex", forbidden)
    raw = [None] * MAX_MESH_TRIANGLES
    primitive = dict(center_x=0., center_y=0., center_z=0., triangles=raw)
    with pytest.raises(ValueError, match="MESH_BUDGET_INVALID"):
        MapSurfaceIndex([primitive] * (MAX_INDEX_MESH_TRIANGLES // MAX_MESH_TRIANGLES + 1))
    with pytest.raises(ValueError, match="MESH_BUDGET_INVALID"):
        MapSurfaceIndex([dict(primitive, triangles=[None] * (MAX_MESH_TRIANGLES + 1))])


# 功能：核验正反射线、壳内起点和共面射线，旋转后的局部索引不能改变遮挡关系。
# 输入：无；输出：两外部首交保留、内部不授予定位、平行离面射线不伪造交点。
def test_negative_inside_and_parallel_rays_in_rotated_local_frame():
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    center = np.array([17., -23., 4.])
    local_origins = np.array([[-3., .2, .1], [3., .2, .1], [0., .2, .1], [0., 2., 0.]])
    local_points = np.array([[-1., .2, .1], [1., .2, .1], [1., .2, .1], [0., 3., 0.]])
    index = MapSurfaceIndex([dict(center_x=center[0].item(), center_y=center[1].item(),
                                 center_z=center[2].item(), yaw_rad=angle,
                                 triangles=cube().tolist(), mesh_double_sided=True)])
    result = index.match(local_points @ rotation.T + center, MapAlignmentLimits(),
                         sensor_origins_world_m=local_origins @ rotation.T + center)
    np.testing.assert_array_equal(result.valid, [True, True, False, False])
    np.testing.assert_allclose(result.normals[:2], np.array([[-1., 0., 0.], [1., 0., 0.]])
                               @ rotation.T, atol=1e-12)
    assert result.rejection_reasons[2] == "origin_inside_or_backface"

    triangle = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    hit = mesh_surface_hits(np.array([[.2, .2, 0.]]), np.array([[1., 0., 0.]]),
                            triangle, .02)
    assert not np.isfinite(hit[0][0])
    assert not hit[2][0]
