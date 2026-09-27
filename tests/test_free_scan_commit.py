"""Compare the free-evidence fast path with the original scalar updater."""

import math
import random

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap, _FREE_LOG_ODDS_THRESHOLD


# 功能：
#   构造有限测试地图，不提供真实地图或飞行权限。
# 输入：
#   无。
# 输出：
#   world：独立测试地图。
def _world():
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=0, y=0, z=0),
                           maximum_bound_m=Vector3(x=20, y=20, z=20))
    return world


# 功能：
#   提取全部可变证据及索引值，忽略仅用于写入隔离的对象身份。
# 输入：
#   world：待比较地图。
# 输出：
#   state：完整数值证据、空间索引及静态撤销状态。
def _state(world):
    state = ({key: (value.log_odds, value.observations, value.latest_monotonic_seconds)
              for key, value in world._evidence.items()}, world._occupied_keys.copy(),
             world._observed_free_keys.copy(),
             {key: value.copy() for key, value in world._evidence_keys_by_chunk.items()},
             world._non_static_evidence_count, world._invalidated_static_free_keys.copy())
    return state


# 功能：
#   在阈值、饱和、零证据、静态空地撤销及双向快照写入中逐值核对原更新算法。
# 输入：
#   seed：可复现的更新顺序。
# 输出：
#   None：数值、索引或任何旧快照发生不一致时测试失败。
@pytest.mark.parametrize('seed', range(8))
def test_free_scan_matches_scalar_and_preserves_all_snapshots(seed):
    rng = random.Random(seed)
    actual, expected = _world(), _world()
    keys = [(index, 2, 3) for index in range(60)]
    initial = [-6., -5.9, _FREE_LOG_ODDS_THRESHOLD,
               math.nextafter(_FREE_LOG_ODDS_THRESHOLD, math.inf), -.1, 0., .8, 6.]
    for world in (actual, expected):
        world._known_static_free_keys = set(keys[:12])
        world._known_static_occupied_keys = set(keys[12:24])
        for index, key in enumerate(keys[:48]):
            world._update_log_odds(key, measurement=initial[index % len(initial)], observed_at=1.)
    snapshots = []
    for tick in range(20):
        if tick % 3 == 0:
            clone = actual.navigation_clone()
            snapshots.append((clone, _state(clone)))
            expected.navigation_clone()
        strengths = tuple((key, rng.choice([0., .000001, .05, .8, 4., 9.])) for key in keys)
        # 内部更新也须保持最大真实时间，不因调用者重放旧时间而回退。
        stamp = rng.choice([.5, 1., 1. + tick / 10])
        actual._commit_free_scan(strengths, stamp)
        for key, strength in strengths:
            expected._update_log_odds(key, measurement=-strength, observed_at=stamp)
        assert _state(actual) == _state(expected)
        for clone, frozen in snapshots:
            assert _state(clone) == frozen
    parent_before = _state(actual)
    child = actual.navigation_clone()
    child._commit_free_scan(tuple((key, 1.) for key in keys), 20.)
    assert _state(actual) == parent_before
    # 后续真实命中必须照常把空地转为占用，不能被捷径遗留的索引掩盖。
    for world in (actual, expected):
        for key in keys:
            world._update_log_odds(key, measurement=15., observed_at=21.)
    assert _state(actual) == _state(expected)
