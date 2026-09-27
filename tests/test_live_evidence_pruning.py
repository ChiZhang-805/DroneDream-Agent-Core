"""Exact scalar-oracle tests for live-map pruning; no simulated flight evidence."""

import math
import random

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap


# 功能：
#   与逐体素旧公式核对完整保留集合、年龄边界、阈值索引和快照隔离。
# 输入：
#   resolution、radius：覆盖不同桶尺度及球面交界的地图参数。
# 输出：
#   None：任一逐值差异通过断言报错。
@pytest.mark.parametrize('resolution', [.125, .25, 1.])
@pytest.mark.parametrize('radius', [.1, 4., 12., 40.])
def test_pruning_matches_scalar_geometry_and_age(resolution, radius):
    rng = random.Random(803)
    world = MetricVoxelMap(resolution_m=resolution,
        minimum_bound_m=Vector3(x=-30, y=-30, z=-30),
        maximum_bound_m=Vector3(x=30, y=30, z=30))
    center = Vector3(x=.23, y=-.57, z=1.23)
    for index in range(12_000):
        key = world.key_for(tuple(rng.uniform(-25, 25) for _ in range(3)))
        world._update_log_odds(key, measurement=1. if index % 2 else -1.,
            observed_at=[9., math.nextafter(10., -math.inf), 10., 11.][index % 4])
    snapshot = world.navigation_clone()
    before = set(world._evidence)
    radius_squared = (radius + math.sqrt(3.) * resolution / 2.)**2
    expected = {key for key, value in world._evidence.items()
        if value.latest_monotonic_seconds >= 10.
        and sum((world._center_point_for_key(key)[i] - (center.x, center.y, center.z)[i])**2
                for i in range(3)) <= radius_squared}
    removed = world.prune_live_evidence(center_m=center, radius_m=radius,
        now_monotonic_seconds=40., maximum_age_seconds=30.)
    assert set(world._evidence) == expected
    assert removed == len(before - expected)
    assert world._non_static_evidence_count == len(expected)
    assert set().union(set(), *world._evidence_keys_by_chunk.values()) == expected
    assert world._occupied_keys == snapshot._occupied_keys & expected
    assert world._observed_free_keys == snapshot._observed_free_keys & expected
    assert set(snapshot._evidence) == before


# 功能：
#   保留球面浮点边界的原平方距离判定，证明整桶捷径不改变边界归属。
# 输入：
#   direction：在目标距离的前、正好和后一个可表示浮点数设置半径。
# 输出：
#   None：保留集合必须与原标量公式完全一致。
@pytest.mark.parametrize('direction', [-math.inf, None, math.inf])
def test_pruning_spherical_boundary(direction):
    world = MetricVoxelMap(resolution_m=.25,
        minimum_bound_m=Vector3(x=0, y=0, z=0),
        maximum_bound_m=Vector3(x=30, y=30, z=30))
    center = Vector3(x=.125, y=.125, z=.125)
    keys = [(48, 0, 0), (32, 32, 0), (48, 1, 0), (47, 0, 0)]
    for key in keys:
        world._update_log_odds(key, measurement=1., observed_at=10.)
    radius = 12. - math.sqrt(3.) * .25 / 2.
    if direction is not None:
        radius = math.nextafter(radius, direction)
    square = (radius + math.sqrt(3.) * .25 / 2.)**2
    expected = {key for key in keys if sum((key[i] * .25)**2 for i in range(3)) <= square}
    world.prune_live_evidence(center_m=center, radius_m=radius,
        now_monotonic_seconds=20., maximum_age_seconds=10.)
    assert set(world._evidence) == expected
