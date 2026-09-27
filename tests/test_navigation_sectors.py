"""Numerical parity fixtures, not flight observations or training data."""

import math
import random
from unittest.mock import patch

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.navigation_sectors import reduce_navigation_sectors


# 功能：
#   原样执行旧标量归约，作为数值优化的独立结果参照，不进入产品代码。
# 输入：
#   items、minimum、resolution、current、heading、radius：体素分类和查询条件。
# 输出：
#   result：按旧顺序计算的八方向摘要。
def scalar_sectors(items, minimum, resolution, current, heading, radius):
    result = [[0, 0, None] for _ in range(8)]
    for key, kind in items:
        center = tuple(minimum[i] + (key[i] + .5) * resolution for i in range(3))
        distance = math.dist(current, center)
        if distance > radius:
            continue
        relative = (math.atan2(center[1] - current[1], center[0] - current[0])
                    - heading + math.pi) % (2 * math.pi) - math.pi
        sector = result[int(round(relative / (math.pi / 4))) % 8]
        sector[kind - 1] += 1
        if kind == 2:
            clearance = max(0., distance - math.sqrt(3.) * resolution / 2.)
            if sector[2] is None or clearance < sector[2]:
                sector[2] = round(clearance, 3)
    return result


# 功能：
#   核对跨块、多种分辨率、负坐标和方向时摘要逐值等于旧实现。
# 输入：
#   resolution、heading：参数化地图分辨率和目标方位。
# 输出：
#   None：通过断言返回。
@pytest.mark.parametrize('resolution', [.1, .25, 1.])
@pytest.mark.parametrize('heading', [-math.pi, -.2, 0., math.pi / 8, math.pi])
def test_batched_sectors_equal_scalar(resolution, heading):
    rng = random.Random(491)
    items = [(tuple(rng.randrange(-40, 40) for _ in range(3)), rng.choice([1, 2]))
             for _ in range(9000)]
    arguments = ((-1.2, -3.4, -2.1), resolution, (.3, -.1, .7), heading, 8.)
    assert reduce_navigation_sectors(iter(items), *arguments) == scalar_sectors(items, *arguments)


# 功能：
#   对半径、扇区半格、原点和毫米舍入附近逐个做标量对照，排除数值边界漂移。
# 输入：
#   无：固定构造正负轴、对角与相邻浮点数。
# 输出：
#   None：通过断言返回。
def test_sector_boundaries_keep_scalar_rounding():
    items = [((x, y, z), 2) for x, y, z in
             [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0), (0, 0, 1)]]
    for direction in [-math.pi, -math.pi / 8, math.pi / 8, 7 * math.pi / 8]:
        for heading in [math.nextafter(direction, -math.inf), direction, math.nextafter(direction, math.inf)]:
            for radius in [math.nextafter(1., 0.), 1., math.nextafter(1., 2.), math.sqrt(2)]:
                arguments = ((-.5, -.5, -.5), 1., (0., 0., 0.), heading, radius)
                assert reduce_navigation_sectors(items, *arguments) == scalar_sectors(items, *arguments)
    assert reduce_navigation_sectors([], (0., 0., 0.), 1., (0., 0., 0.), 0., 1.) == [[0, 0, None] for _ in range(8)]


# 功能：
#   核对球外整块、全空地、全占用及跨分块输入，筛选和一次计数不能改变任何方向摘要。
# 输入：
#   kind：输入体素分类；radius：全部排除、部分保留或全部保留的查询半径。
# 输出：
#   None：逐值对照通过后返回。
@pytest.mark.parametrize('kind', [1, 2])
@pytest.mark.parametrize('radius', [.01, 2., 100.])
def test_filtered_sector_batches_equal_scalar(kind, radius):
    items = [((x, y, z), kind) for x in range(-18, 19)
             for y in range(-18, 19) for z in [-1, 0, 1, 2]]
    arguments = ((-.5, -.5, -.5), .25, (.17, -.23, .81), -.71, radius)
    expected = scalar_sectors(items, *arguments)
    assert reduce_navigation_sectors(iter(items), *arguments) == expected


# 功能：
#   确认产品入口仅汇总实际证据，并保持静态占用覆盖、模糊概率和快照隔离规则。
# 输入：
#   无：小地图中的已知空地、占用及不确定证据。
# 输出：
#   None：通过断言返回。
def test_product_snapshot_classification_and_clone_isolation():
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-4,y=-4,z=-4),
                           maximum_bound_m=Vector3(x=4,y=4,z=4))
    for index, value in enumerate([-6., -.6190392084062235, -.1, 0., .1, .6190392084062235, 6.]):
        world._update_log_odds((10 + index, 16, 16), measurement=value, observed_at=1.)
    world.seed_known_static_region([(Vector3(x=-1.375,y=.125,z=.125), True),
                                   (Vector3(x=2,y=2,z=2), False)], source_sha256='a' * 64)
    clone = world.navigation_clone()
    classified = [(key, 1 if clone.is_observed_free(key) else 2) for key in clone._evidence
                  if clone.is_observed_free(key) or clone.is_occupied(key)]
    expected = scalar_sectors(classified, clone._minimum, .25, (0.,0.,0.), 0., 8.)
    world._update_log_odds((10,16,16), measurement=12., observed_at=2.)
    snapshot = clone.text_navigation_snapshot(current_position_m=Vector3(x=0,y=0,z=0),
        current_velocity_mps=Vector3(x=0,y=0,z=0), goal_position_m=Vector3(x=1,y=0,z=0),
        required_clearance_m=.2, include_candidate_paths=False)
    actual = [[sector['observed_free_voxels'], sector['occupied_voxels'],
               sector['nearest_occupied_clearance_m']] for sector in snapshot['egocentric_sectors']]
    assert actual == expected


# 功能：
#   逐值对照局部桶和全图扫描的完整快照，覆盖小/大查询、负坐标与桶/球面边界。
# 输入：
#   radius：查询半径。
#   position：查询位置三元组。
# 输出：
#   None：无业务返回值。
@pytest.mark.parametrize('radius', [.2500001, 1., 4., 8., 1000.])
@pytest.mark.parametrize('position', [(0., 0., 0.), (-15.875, -7.875, .125),
                                      (math.nextafter(4., 0.), -4., 3.)])
def test_scoped_navigation_matches_full_map(radius, position):
    rng = random.Random(805)
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-32,y=-32,z=-32),
                           maximum_bound_m=Vector3(x=32,y=32,z=32))
    for _ in range(6000):
        key = tuple(rng.randrange(256) for _ in range(3))
        world._update_log_odds(key, measurement=rng.choice([-4., -.62, 0., .62, 4.]), observed_at=1.)
    # 球面和桶边界附近的有证据体素必须通过最终原始距离判定，不能在粗筛阶段丢掉。
    for offset in [-8.125, -8., -7.875, -.125, 0., .125, 7.875, 8., 8.125]:
        point = Vector3(x=position[0] + offset, y=position[1], z=position[2])
        world._update_log_odds(world.key_for(point), measurement=4., observed_at=1.)
    world.seed_known_static_region([(Vector3(x=.125,y=.125,z=.125), True)], source_sha256='a' * 64)
    arguments = dict(current_position_m=Vector3(x=position[0],y=position[1],z=position[2]),
        current_velocity_mps=Vector3(x=0,y=0,z=0), goal_position_m=Vector3(x=1,y=1,z=0),
        required_clearance_m=.15, local_radius_m=radius, include_candidate_paths=False)
    snapshot = world.text_navigation_snapshot(**arguments)
    with patch.object(world, '_navigation_sector_chunks', lambda *_: iter(world._evidence_keys_by_chunk)):
        reference = world.text_navigation_snapshot(**arguments)
    assert snapshot == reference
    clone = world.navigation_clone()
    world.prune_live_evidence(center_m=Vector3(x=0,y=0,z=0), radius_m=2.,
                              now_monotonic_seconds=40., maximum_age_seconds=30.)
    for selected in (world, clone):
        actual = selected.text_navigation_snapshot(**arguments)
        with patch.object(selected, '_navigation_sector_chunks', lambda *_, selected=selected: iter(selected._evidence_keys_by_chunk)):
            assert actual == selected.text_navigation_snapshot(**arguments)


# 功能：
#   检查全图及局部查询只提供已有键，不再随机查找证据对象。
# 输入：
#   radius：小查询或覆盖全部证据的查询半径。
# 输出：
#   None：断言访问方式与证据集合一致。
@pytest.mark.parametrize('radius', [1., 1000.])
def test_full_coverage_avoids_random_evidence_lookups(radius):
    class LookupCountingDict(dict):
        lookups = 0

        # 功能：
        #   计数测试中的逐键随机访问，不改变返回对象。
        # 输入：
        #   key：已有体素键。
        # 输出：
        #   evidence：原字典中的证据对象。
        def __getitem__(self, key):
            self.lookups += 1
            evidence = super().__getitem__(key)
            return evidence

    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-32,y=-32,z=-32),
                           maximum_bound_m=Vector3(x=32,y=32,z=32))
    for point in [(0., 0., 0.), (20., 20., 20.), (-20., -20., -20.)]:
        world._update_log_odds(world.key_for(point), measurement=-4., observed_at=1.)
    world._evidence = LookupCountingDict(world._evidence)
    records = set(world._navigation_sector_keys((0., 0., 0.), radius))
    expected_keys = set(world._evidence) if radius == 1000. else {world.key_for((0., 0., 0.))}
    assert records == expected_keys
    assert world._evidence.lookups == 0
    assert dict(world._classified_navigation_evidence(records)) == {key: 1 for key in expected_keys}
    assert world._evidence.lookups == 0
