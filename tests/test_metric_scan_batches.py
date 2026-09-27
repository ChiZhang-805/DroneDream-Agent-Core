import math
import random

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.metric_ray_sampling import sample_metric_rays


# 功能：
#   按原始标量公式计算体素键，作为与分组数组内核独立的比较基准。
# 输入：
#   world：提供固定边界及分辨率的合成地图。
#   ray：当前合成米制射线。
# 输出：
#   keys：原顺序去重的标量体素键。
def _scalar_keys(world, ray):
    origin = ray.origin_m.x, ray.origin_m.y, ray.origin_m.z
    endpoint = ray.endpoint_m.x, ray.endpoint_m.y, ray.endpoint_m.z
    steps = max(1, math.ceil(math.dist(origin, endpoint) / (world.resolution_m * .45)))
    keys = []
    for step in range(steps + 1):
        ratio = step / steps
        key = tuple(math.floor((origin[i] + (endpoint[i] - origin[i]) * ratio
                                - world._minimum[i]) * (1. / world.resolution_m))
                    for i in range(3))
        if not keys or key != keys[-1]:
            keys.append(key)
    return keys


# 功能：
#   构造有明确有限边界的空白测试地图，不提供通行先验。
# 输入：
#   extent：三轴对称边界的米数。
# 输出：
#   world：独立的测试地图。
def _world(extent=10.):
    world = MetricVoxelMap(resolution_m=.25,
        minimum_bound_m=Vector3(x=-extent, y=-extent, z=-extent),
        maximum_bound_m=Vector3(x=extent, y=extent, z=extent))
    return world


# 功能：
#   在跨组的混合射线上对比每条体素顺序及整帧最大强度、命中优先和首次插入顺序。
# 输入：
#   seed：固定随机种子。
# 输出：
#   None：分组结果与独立标量算法不一致时测试失败。
@pytest.mark.parametrize("seed", range(5))
def test_grouped_rays_and_complete_updates_match_scalar(seed):
    rng, world = random.Random(seed), _world()
    rays = []
    for n in range(137):
        origin = Vector3(**{axis: rng.uniform(-10, 10) for axis in "xyz"})
        endpoint = (origin.model_copy() if n % 11 == 0 else
                    Vector3(**{axis: rng.uniform(-10, 10) for axis in "xyz"}))
        rays.append(RangeRayObservation(origin_m=origin, endpoint_m=endpoint,
            hit=bool(n % 2), confidence=rng.choice((0., .1, .5, .7, 1.)),
            observed_at_monotonic_seconds=1.))
    expected_hits, expected_free = {}, {}
    sampled = list(world._sample_scan(rays))
    for ray, (snapshot, actual) in zip(rays, sampled, strict=True):
        expected = _scalar_keys(world, ray)
        assert actual == expected
        assert all(type(key) is tuple and len(key) == 3
                   and all(type(axis) is int for axis in key) for key in actual)
        assert snapshot.confidence == ray.confidence
        bounded = max(.5, min(.99, ray.confidence))
        strength = math.log(bounded / (1. - bounded))
        for key in expected[:-1] if ray.hit else expected:
            expected_free[key] = max(expected_free.get(key, 0.), strength)
        if ray.hit:
            expected_hits[expected[-1]] = max(expected_hits.get(expected[-1], 0.), strength)
    prepared = world.prepare_scan(rays)
    assert prepared.occupied_strengths == tuple(expected_hits.items())
    assert prepared.free_strengths == tuple((k, v) for k, v in expected_free.items()
                                           if k not in expected_hits)
    assert prepared.ray_count == len(rays)
    assert not world._evidence


# 功能：
#   检查正负体素边界两侧相邻浮点数、零长度射线和短射线填充均保持标量结果。
# 输入：
#   axis：贴近体素边界的坐标轴；direction：相邻浮点数所在方向。
# 输出：
#   无：所有键必须是独立的普通整数三元组。
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('direction', [-math.inf, math.inf])
def test_record_conversion_preserves_boundary_and_zero_length(axis, direction):
    world = _world()
    rays = []
    for value in (-.25, 0., .25):
        coordinates = [0., 0., 0.]
        coordinates[axis] = math.nextafter(value, direction)
        endpoint = Vector3(**dict(zip('xyz', coordinates, strict=True)))
        for origin in (endpoint, Vector3(x=-1., y=-1., z=-1.)):
            rays.append(RangeRayObservation(origin_m=origin, endpoint_m=endpoint,
                hit=True, confidence=.9, observed_at_monotonic_seconds=1.))
    actual = [keys for _, keys in world._sample_scan(rays)]
    assert actual == [_scalar_keys(world, ray) for ray in rays]
    saved = [list(keys) for keys in actual]
    rays[0].endpoint_m.x = 2.
    assert actual == saved


# 功能：
#   同置信度内核与降低再升高的混合置信度均逐体素保留最大证据和首次插入顺序。
# 输入：
#   confidences：覆盖等值、降低、升高、零强度和命中冲突的序列。
#   block_limit：批次键引用上限；monkeypatch：临时替换测试上限。
# 输出：
#   无。
@pytest.mark.parametrize('block_limit', [4, 17, 65_536])
@pytest.mark.parametrize("confidences", [(.92,) * 80, (.92, .7, .95, .5, .8, 1., 0.) * 12])
def test_uniform_strength_fast_path_matches_ordered_scalar(confidences, block_limit, monkeypatch):
    from dronedream_agent_core import local_world_model as module

    monkeypatch.setattr(module, 'MAX_BATCH_POINTS', block_limit)
    world = _world()
    rays, frees, hits = [], {}, {}
    for index, confidence in enumerate(confidences):
        ray = RangeRayObservation(origin_m=Vector3(x=0, y=0, z=0),
            endpoint_m=Vector3(x=1 + (index % 3) / 4, y=0, z=0),
            hit=bool(index % 2), confidence=confidence, observed_at_monotonic_seconds=1.)
        rays.append(ray)
        keys = _scalar_keys(world, ray)
        bounded = max(.5, min(.99, confidence))
        strength = math.log(bounded / (1. - bounded))
        for key in keys[:-1] if ray.hit else keys:
            frees[key] = max(frees.get(key, 0.), strength)
        if ray.hit:
            hits[keys[-1]] = max(hits.get(keys[-1], 0.), strength)
    prepared = world.prepare_scan(rays)
    assert prepared.free_strengths == tuple((k, v) for k, v in frees.items() if k not in hits)
    assert prepared.occupied_strengths == tuple(hits.items())


# 功能：
#   验证长射线跨越 65536 点分块，且与短射线同帧时不因补齐长度放大计算量。
# 输入：
#   monkeypatch：记录实际送入内核的分组大小。
# 输出：
#   None：端点、顺序或分组预算不满足原始算法时测试失败。
def test_long_ray_chunks_preserve_scalar_order_and_bounded_padding(monkeypatch):
    from dronedream_agent_core import local_world_model as module

    world = _world(10_000.)
    short = RangeRayObservation(origin_m=Vector3(x=0, y=0, z=0),
        endpoint_m=Vector3(x=1, y=1, z=1), hit=True, confidence=.9,
        observed_at_monotonic_seconds=1.)
    long = short.model_copy(update={"endpoint_m": Vector3(x=10_000, y=0, z=0)})
    batches = []

    # 功能：
    #   记录分组形状并调用真正的数值内核，不替代体素计算。
    # 输入：
    #   rays、minimum、resolution_m：原始数值内核参数。
    # 输出：
    #   result：真实内核返回的按射线分组的体素列表。
    def record_batch(rays, minimum, resolution_m):
        batches.append((len(rays), max(ray.steps for ray in rays)))
        result = sample_metric_rays(rays, minimum, resolution_m)
        return result

    monkeypatch.setattr(module, "sample_metric_rays", record_batch)
    records = list(world._sample_scan([short, long, short]))
    assert [keys for _, keys in records] == [_scalar_keys(world, ray)
                                            for ray in (short, long, short)]
    assert batches == [(1, records[0][0].steps), (1, records[1][0].steps),
                       (1, records[2][0].steps)]
    assert records[1][0].steps > 65_536


# 功能：
#   验证外部迭代器重复使用一个可变消息时，每次读取的坐标、命中与置信度独立保存。
# 输入：
#   无。
# 输出：
#   None：后续修改污染已读取样本时测试失败。
def test_reused_mutable_input_message_is_snapshotted_before_next_item():
    world = _world()
    ray = RangeRayObservation(origin_m=Vector3(x=0, y=0, z=0),
        endpoint_m=Vector3(x=1, y=0, z=0), hit=False, confidence=.5,
        observed_at_monotonic_seconds=1.)
    expected = []

    # 功能：
    #   在每次产生样本前修改同一消息，模拟重用接收缓冲的调用者。
    # 输入：
    #   无。
    # 输出：
    #   ray：当前一次读取对应的可变消息对象。
    def observations():
        for n in range(70):
            ray.endpoint_m.x = n / 10.
            ray.hit = bool(n % 2)
            ray.confidence = .5 + n / 200.
            expected.append((_scalar_keys(world, ray), ray.hit, ray.confidence))
            yield ray

    actual = list(world._sample_scan(observations()))
    assert [(keys, snapshot.hit, snapshot.confidence) for snapshot, keys in actual] == expected


# 功能：
#   验证批次后段非法数据仍整体拒绝，已经计算出的前批不能部分写入地图。
# 输入：
#   invalid：超遍历预算、异步来源或非法样本的构造种类。
# 输出：
#   None：拒绝失败或出现地图写入时测试失败。
@pytest.mark.parametrize("invalid", ["work", "clock", "sample"])
def test_late_invalid_sample_cannot_commit_an_earlier_batch(invalid):
    world = _world()
    ray = RangeRayObservation(origin_m=Vector3(x=0, y=0, z=0),
        endpoint_m=Vector3(x=1, y=0, z=0), hit=False, confidence=.9,
        observed_at_monotonic_seconds=1.)
    rays = [ray] * 70
    if invalid == "clock":
        rays.append(ray.model_copy(update={"observed_at_monotonic_seconds": 2.}))
    elif invalid == "sample":
        rays.append(None)
    else:
        # 单条合法而全帧遍历超过两百万点；不请求超大数组。
        rays = [ray] * 200_000
    with pytest.raises(ValueError):
        world.prepare_scan(rays)
    assert world.observation_count == 0
    assert world.latest_observation_monotonic_seconds is None
    assert not world._evidence
