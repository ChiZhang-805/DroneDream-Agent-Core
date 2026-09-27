"""Native-kernel exactness, malformed inputs and map transaction isolation."""

import gc
import math
import random
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.metric_scan_native import backend

pytestmark = pytest.mark.skipif(
    backend is None, reason="native extension not built on this platform"
)


# 功能：
#   以两种分辨率及不同正负边界覆盖大量混合置信度、短射线和往返射线，逐值核对实现。
# 输入：
#   seed：固定随机种子；resolution：地图分辨率。
# 输出：
#   None：原实现与编译内核的完整待提交证据完全一致。
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("resolution", [0.25, 0.37])
def test_native_matches_python_order_values_and_transaction(seed, resolution):
    world = MetricVoxelMap(
        resolution_m=resolution,
        minimum_bound_m=Vector3(x=-20, y=-20, z=-20),
        maximum_bound_m=Vector3(x=20, y=20, z=20),
    )
    rng = random.Random(seed)
    rays = []
    for index in range(240):
        origin = Vector3(**{axis: rng.uniform(-19, 19) for axis in "xyz"})
        endpoint = (
            origin.model_copy()
            if index % 7 == 0
            else Vector3(**{axis: rng.uniform(-19, 19) for axis in "xyz"})
        )
        rays.append(
            RangeRayObservation(
                origin_m=origin,
                endpoint_m=endpoint,
                hit=bool(index % 2),
                confidence=rng.choice([0.0, 0.5, 0.7, 0.92, 1.0]),
                observed_at_monotonic_seconds=1.0,
            )
        )
    with patch("dronedream_agent_core.local_world_model.NATIVE_SCAN_AVAILABLE", False):
        reference = world.prepare_scan(rays)
    with patch("dronedream_agent_core.local_world_model.NATIVE_SCAN_AVAILABLE", True):
        actual = world.prepare_scan(rays)
    assert actual == reference
    assert world.observation_count == 0
    world.commit_scan(actual)
    with pytest.raises(ValueError, match="PREPARED_STATE_CHANGED"):
        world.commit_scan(actual)


# 功能：
#   独立核对精确体素边界、相邻浮点数以及长射线跨批时的最终键与强度。
# 输入：
#   direction：浮点相邻方向。
# 输出：
#   None：精确证据与原实现完全一致。
@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_native_boundary_and_long_ray(direction):
    world = MetricVoxelMap(
        resolution_m=0.25,
        minimum_bound_m=Vector3(x=-10000, y=-1, z=-1),
        maximum_bound_m=Vector3(x=10000, y=1, z=1),
    )
    rays = [
        RangeRayObservation(
            origin_m=Vector3(x=-9999, y=0, z=0),
            endpoint_m=Vector3(x=math.nextafter(value, direction), y=0.0, z=0.0),
            hit=hit,
            confidence=0.92,
            observed_at_monotonic_seconds=1.0,
        )
        for value in [0.0, 0.25]
        for hit in [False, True]
    ]
    with patch("dronedream_agent_core.local_world_model.NATIVE_SCAN_AVAILABLE", False):
        reference = world.prepare_scan(rays)
    assert world._prepare_native_scan(rays) == reference


# 功能：
#   验证直接调用编译入口也拒绝非法容器、非有限数据、错误布尔位和超量遍历。
# 输入：
#   kind：非法输入种类。
# 输出：
#   None：输入在返回任何结果前被拒绝。
@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "list-row",
        "nan",
        "bool-steps",
        "negative-steps",
        "huge-steps",
        "bad-hit",
        "strength",
        "total-work",
        "coordinate",
    ],
)
def test_native_direct_boundary_is_checked(kind):
    row = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 10, 2.4, True)
    rays = [row]
    if kind == "empty":
        rays = []
    elif kind == "list-row":
        rays = [list(row)]
    elif kind == "nan":
        rays = [((math.nan, 0.0, 0.0), *row[1:])]
    elif kind == "bool-steps":
        rays = [(*row[:2], True, *row[3:])]
    elif kind == "negative-steps":
        rays = [(*row[:2], -1, *row[3:])]
    elif kind == "huge-steps":
        rays = [(*row[:2], 2**100, *row[3:])]
    elif kind == "bad-hit":
        rays = [(*row[:4], 1)]
    elif kind == "strength":
        rays = [(*row[:3], float("inf"), row[4])]
    elif kind == "total-work":
        rays = [(*row[:2], 1000000, *row[3:])] * 2
    elif kind == "coordinate":
        rays = [((1e100, 0.0, 0.0), *row[1:])]
    with pytest.raises((ValueError, OverflowError)):
        backend.integrate(rays, (0.0, 0.0, 0.0), 0.25)


# 功能：
#   拒绝无效分辨率，避免 bool、非有限值或隐式转换回调改变输入合同。
# 输入：
#   resolution：待拒绝的分辨率。
# 输出：
#   None。
@pytest.mark.parametrize(
    "resolution", [True, False, 0.0, -1.0, math.nan, math.inf, "0.25", 2**10000]
)
def test_native_resolution_boundary(resolution):
    with pytest.raises((ValueError, OverflowError)):
        backend.integrate(
            [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 10, 2.4, True)], (0.0, 0.0, 0.0), resolution
        )


# 功能：
#   并发调用使用独立容器，不能跨调用泄漏占用、空闲或置信度证据。
# 输入：
#   无。
# 输出：
#   None。
def test_native_concurrent_calls_are_isolated():
    inputs = [
        [((0.0, 0.0, 0.0), (float(index), 0.25, -0.25), 15000, 0.7, bool(index % 2))]
        for index in range(1, 9)
    ]
    expected = [backend.integrate(rays, (-10.0, -10.0, -10.0), 0.25) for rays in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(backend.integrate, rays, (-10.0, -10.0, -10.0), 0.25) for rays in inputs
        ]
        assert [future.result() for future in futures] == expected


# 功能：
#   核对原生返回图全为不可变纯值且无需循环扫描，反复回收不改变结果或全局回收策略。
# 输入：
#   hit：分别覆盖有命中、无命中及空占用子元组。
# 输出：
#   None：数值图、引用生命周期与外部 GC 开关和阈值均保持正确。
@pytest.mark.parametrize("hit", [True, False])
def test_native_value_graph_does_not_require_cyclic_collection(hit):
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    rays = [((0.0, 0.0, 0.0), (8.0, 1.0, -1.0), 800, 0.7, hit)]
    expected = backend.integrate(rays, (-10.0, -10.0, -10.0), 0.25)
    for _ in range(12):
        result = backend.integrate(rays, (-10.0, -10.0, -10.0), 0.25)
        assert type(result) is tuple and not gc.is_tracked(result)
        for group in result:
            assert type(group) is tuple and not gc.is_tracked(group)
            for row in group:
                assert type(row) is tuple and not gc.is_tracked(row)
                key, strength = row
                assert type(key) is tuple and not gc.is_tracked(key)
                assert all(type(axis) is int for axis in key)
                assert type(strength) is float
        gc.collect()
        assert result == expected
    assert (gc.isenabled(), gc.get_threshold()) == (enabled, thresholds)
