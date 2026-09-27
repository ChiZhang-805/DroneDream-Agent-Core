"""Cache invalidation/ownership fixtures, not training observations."""

import math
import random
from unittest.mock import patch

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.navigation_sectors import (
    pack_navigation_sector_block,
    reduce_navigation_sectors,
)


# 功能：
#   创建含多个空间桶和不确定分类的有界地图，仅供数值回归。
# 输入：
#   无：固定随机种子与地图边界。
# 输出：
#   world：可独立修改的测试地图。
def world_fixture():
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-8, y=-8, z=-8),
                           maximum_bound_m=Vector3(x=8, y=8, z=8))
    rng = random.Random(551)
    for _ in range(3000):
        world._update_log_odds(tuple(rng.randrange(64) for _ in range(3)),
                              measurement=rng.choice([-4., 0., 4.]), observed_at=1.)
    return world


# 功能：
#   用完全不读取缓存的全证据归约对照生产快照，核对每个方向和地图缓存容量。
# 输入：
#   world：待验地图；position、radius：查询位置和范围。
# 输出：
#   snapshot：已经通过独立全量归约对照的生产快照。
def assert_uncached_parity(world, position=(0., 0., 0.), radius=8.):
    goal = (2., 1., 0.)
    heading = math.atan2(goal[1] - position[1], goal[0] - position[0])
    expected = reduce_navigation_sectors(world._classified_navigation_evidence(),
        world._minimum, world.resolution_m, position, heading, radius)
    snapshot = world.text_navigation_snapshot(current_position_m=Vector3(x=position[0], y=position[1], z=position[2]),
        current_velocity_mps=Vector3(x=0., y=0., z=0.), goal_position_m=Vector3(x=2., y=1., z=0.),
        local_radius_m=radius, required_clearance_m=.2, include_candidate_paths=False)
    actual = [[row['observed_free_voxels'], row['occupied_voxels'], row['nearest_occupied_clearance_m']]
              for row in snapshot['egocentric_sectors']]
    assert actual == expected
    assert world._navigation_blocks.keys() <= world._evidence_keys_by_chunk.keys()
    return snapshot


# 功能：
#   已为空地的证据刷新不重建数字桶，但位置变化仍重新计算距离与方向。
# 输入：
#   无：跨桶地图和显式负观测。
# 输出：
#   None：复用与逐值对照通过后返回。
def test_warm_blocks_reuse_only_unchanged_classification():
    world = world_fixture()
    assert_uncached_parity(world, radius=100.)
    cached = dict(world._navigation_blocks)
    free_key = next(iter(world._observed_free_keys))
    with patch('dronedream_agent_core.local_world_model.pack_navigation_sector_block',
               side_effect=AssertionError('unchanged block rebuilt')):
        world._commit_free_scan(((free_key, 2.),), observed_at=2.)
        assert_uncached_parity(world, position=(-1., 2., 1.), radius=5.)
    assert all(world._navigation_blocks[key] is block for key, block in cached.items())


# 功能：
#   逐次穿过空地、不确定和占用边界后重新核对，包含新增不确定体素。
# 输入：
#   无：固定随机更新顺序。
# 输出：
#   None：所有分类转换后的摘要与全扫描一致。
def test_cache_invalidates_on_insert_and_all_probability_transitions():
    world = world_fixture()
    assert_uncached_parity(world)
    rng = random.Random(552)
    keys = list(world._evidence)[:20] + [(0, 0, 0), (63, 63, 63)]
    for index in range(60):
        key = rng.choice(keys)
        world._update_log_odds(key, measurement=rng.choice([-12., -.7, 0., .7, 12.]), observed_at=2. + index)
        assert_uncached_parity(world, position=(.125, -.125, .125))


# 功能：
#   静态占用覆盖、裁剪和年龄淘汰后缓存立即失效，原克隆保留自己的旧视图。
# 输入：
#   local_clone：是否只克隆局部半径。
# 输出：
#   None：原图和独立克隆均与各自全量证据一致。
@pytest.mark.parametrize('local_clone', [False, True])
def test_cache_static_override_pruning_and_clone_isolation(local_clone):
    world = world_fixture()
    assert_uncached_parity(world, radius=100.)
    clone = world.navigation_clone(**({'center_m': Vector3(x=0., y=0., z=0.), 'radius_m': 3.}
                                     if local_clone else {}))
    old_clone = assert_uncached_parity(clone)
    key = next(iter(world._observed_free_keys))
    world.seed_known_static_region([(world.center_for(key), True)], source_sha256='a' * 64)
    assert_uncached_parity(world)
    assert assert_uncached_parity(clone) == old_clone
    world.prune_live_evidence(center_m=Vector3(x=0., y=0., z=0.), radius_m=4.,
                             now_monotonic_seconds=3., maximum_age_seconds=10.)
    assert_uncached_parity(world)
    assert assert_uncached_parity(clone) == old_clone
    clone._update_log_odds(key, measurement=12., observed_at=5.)
    assert_uncached_parity(clone)
    assert_uncached_parity(world)
    world.prune_live_evidence(center_m=Vector3(x=0., y=0., z=0.), radius_m=100.,
                             now_monotonic_seconds=100., maximum_age_seconds=10.)
    assert not world._navigation_blocks
    assert_uncached_parity(world)


# 功能：
#   缓存数字块不可重新打开写权限，不能通过共享数组改写另一个地图视图。
# 输入：
#   无：一个包含空地与占用的独立数字块。
# 输出：
#   None：只读与容量检查通过后返回。
def test_packed_blocks_are_bounded_and_irreversibly_read_only():
    block = pack_navigation_sector_block([((0, 0, 0), 1), ((1, 2, 3), 2)], (0., 0., 0.), .25)
    for array in block:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match='NAVIGATION_SECTOR_BLOCK_CAPACITY'):
        pack_navigation_sector_block([((0, 0, 0), 1)] * 4097, (0., 0., 0.), .25)


# 功能：
#   地图数值几何发生变化时禁止复用旧坐标数组，保持与当前公开属性一致。
# 输入：
#   无：先建立缓存再调整测试地图分辨率。
# 输出：
#   None：当前几何的缓存与全量计算一致。
def test_geometry_change_rebuilds_numeric_blocks():
    world = world_fixture()
    assert_uncached_parity(world)
    world.resolution_m = .3
    assert_uncached_parity(world)
