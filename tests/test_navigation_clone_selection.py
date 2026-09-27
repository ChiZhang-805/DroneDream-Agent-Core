"""Independent scalar checks for local evidence selection; synthetic, not training data."""

import math
import random

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.navigation_sectors import select_navigation_keys


# 功能：
#   使用独立全证据扫描核对局部空间索引，覆盖非整数分辨率、负坐标及稀疏大范围查询。
# 输入：
#   resolution：各组地图分辨率。
# 输出：
#   None：所有随机查询与标量全扫描相同。
@pytest.mark.parametrize('resolution', [.03, .25, .37, 1.])
def test_local_selection_matches_full_scalar_scan(resolution):
    world = MetricVoxelMap(resolution_m=resolution, minimum_bound_m=Vector3(x=-10., y=-10., z=-10.),
                           maximum_bound_m=Vector3(x=10., y=10., z=10.))
    rng = random.Random(683)
    edge = int(20. / resolution)
    for _ in range(2500):
        key = tuple(rng.randrange(edge) for _ in range(3))
        world._update_log_odds(key, measurement=rng.choice([-4., 0., 4.]), observed_at=1.)
    for _ in range(25):
        point = tuple(rng.uniform(-9.5, 9.5) for _ in range(3))
        radius = rng.choice([.01, .2, 2., 8., 40.])
        expected = {key for key in world._evidence
                    if math.dist(point, world._center_point_for_key(key))
                    <= radius + math.sqrt(3.) * resolution / 2.}
        assert world._local_navigation_keys(center_m=Vector3(x=point[0], y=point[1], z=point[2]),
                                            radius_m=radius) == expected


# 功能：
#   确认体素外接球扩展范围跨越空间桶时仍保留证据，避免旧包围盒提前漏筛。
# 输入：
#   无：球心位于第十四格末端，相邻桶第十六格与查询球相交。
# 输出：
#   None：跨桶证据包含在独立克隆中。
def test_voxel_overlap_across_chunk_boundary_is_not_dropped():
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=0., y=0., z=0.),
                           maximum_bound_m=Vector3(x=8., y=8., z=8.))
    world._update_log_odds((16, 1, 1), measurement=4., observed_at=1.)
    clone = world.navigation_clone(center_m=Vector3(x=3.749, y=.375, z=.375), radius_m=.2)
    assert set(clone._evidence) == {(16, 1, 1)}


# 功能：
#   对精确球面及相邻浮点半径核对数值筛选，包含多批输入、空输入及大坐标。
# 输入：
#   offset：地图原点；size：体素键数量。
# 输出：
#   None：逐项标量距离判定与批处理结果相同。
@pytest.mark.parametrize('offset', [0., -1e8, 1e12])
@pytest.mark.parametrize('size', [0, 1, 5000])
def test_batched_sphere_boundary_parity(offset, size):
    keys = [(index % 31, index // 31 % 31, index // 961) for index in range(size)]
    minimum, resolution, current = (offset, offset, offset), .37, (offset + 2., offset + 3., offset + 1.)
    radius = math.dist(current, tuple(offset + (value + .5) * resolution for value in (9, 9, 2)))
    for threshold in [math.nextafter(radius, 0.), radius, math.nextafter(radius, math.inf)]:
        expected = {key for key in keys if math.dist(current,
                    tuple(offset + (value + .5) * resolution for value in key)) <= threshold}
        assert select_navigation_keys(iter(keys), minimum, resolution, current, threshold) == expected


# 功能：
#   验证损坏的空间索引不会静默丢弃体素生成地图，失败前不改变共享证据的所有权。
# 输入：
#   monkeypatch：注入缺失证据键的夹具。
# 输出：
#   None：不返回业务数据。
def test_missing_index_evidence_rejects_clone_without_changing_owner(monkeypatch):
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=0., y=0., z=0.),
                           maximum_bound_m=Vector3(x=8., y=8., z=8.))
    world._update_log_odds((1, 1, 1), measurement=4., observed_at=1.)
    owner = world._evidence_owner
    monkeypatch.setattr(world, '_local_navigation_keys', lambda **_: {(1, 1, 1), (2, 2, 2)})
    with pytest.raises(ValueError, match='NAVIGATION_EVIDENCE_INDEX_INCONSISTENT'):
        world.navigation_clone(center_m=Vector3(x=1., y=1., z=1.), radius_m=3.)
    assert world._evidence_owner is owner
    assert world._evidence[(1, 1, 1)].owner is owner


# 功能：
#   核对一次跨越多桶的筛选只调用统一分块入口，且集合、证据数值和后续写入隔离仍正确。
# 输入：
#   monkeypatch：观察数值筛选批次入口的夹具。
# 输出：
#   None：不返回业务数据。
def test_boundary_chunks_share_one_bounded_selection_without_changing_cells(monkeypatch):
    from unittest.mock import Mock
    import dronedream_agent_core.local_world_model as module

    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-12., y=-12., z=-12.),
                           maximum_bound_m=Vector3(x=12., y=12., z=12.))
    for x in range(0, 96, 3):
        for y in range(0, 96, 3):
            for z in range(0, 96, 7):
                world._update_log_odds((x, y, z), measurement=4. if x % 2 else -4., observed_at=1.)
    expected = {key for key in world._evidence
                if math.dist((0., 0., 0.), world._center_point_for_key(key))
                <= 7. + math.sqrt(3.) * .25 / 2.}
    observed = Mock(wraps=module.select_navigation_keys)
    monkeypatch.setattr(module, 'select_navigation_keys', observed)
    clone = world.navigation_clone(center_m=Vector3(x=0., y=0., z=0.), radius_m=7.)
    assert observed.call_count == 1
    assert set(clone._evidence) == expected
    key = next(iter(expected))
    before = clone._evidence[key].log_odds
    world._update_log_odds(key, measurement=-2., observed_at=2.)
    assert clone._evidence[key].log_odds == before
    assert clone._evidence[key] is not world._evidence[key]
