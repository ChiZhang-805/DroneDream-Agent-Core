"""Probability/index agreement fixtures; not sensor or flight evidence."""

import math

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import (
    _FREE_LOG_ODDS_THRESHOLD,
    _OCCUPIED_LOG_ODDS_THRESHOLD,
    MetricVoxelMap,
)


# 功能：
#   核对阈值两侧可表示浮点数的概率、实时索引与导航分类，覆盖已有占用降到边界的转换。
# 输入：
#   threshold、direction：概率阈值和逐位移动方向。
# 输出：
#   None：所有索引必须与原概率公式一致。
@pytest.mark.parametrize('threshold', [_FREE_LOG_ODDS_THRESHOLD, _OCCUPIED_LOG_ODDS_THRESHOLD])
@pytest.mark.parametrize('direction', [-math.inf, math.inf])
def test_occupancy_indexes_match_probability_at_adjacent_floats(threshold, direction):
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=0,y=0,z=0),
                           maximum_bound_m=Vector3(x=10,y=10,z=10))
    value = threshold
    for index in range(12):
        key = (index, 2, 2)
        world._update_log_odds(key, measurement=value, observed_at=1.)
        probability = world.occupancy_probability(key)
        assert (key in world._occupied_keys) == (probability >= .65)
        assert (key in world._observed_free_keys) == (probability <= .35)
        value = math.nextafter(value, direction)
    clone = world.navigation_clone()
    for key in list(world._evidence):
        world._update_log_odds(key, measurement=8., observed_at=2.)
        world._update_log_odds(key, measurement=threshold - 6., observed_at=3.)
        probability = world.occupancy_probability(key)
        assert (key in world._occupied_keys) == (probability >= .65)
        assert (key in world._observed_free_keys) == (probability <= .35)
    for selected in (world, clone):
        classified = dict(selected._classified_navigation_evidence())
        assert {key for key, kind in classified.items() if kind == 2} == selected._occupied_keys
        assert {key for key, kind in classified.items() if kind == 1} == selected._observed_free_keys
