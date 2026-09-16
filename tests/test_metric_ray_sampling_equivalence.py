"""Sampling optimization must preserve every ordered voxel, including boundaries."""
import math
import random

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap


# 功能：
#   用原始逐点公式提供独立参考，不调用优化路径。
# 输入：
#   origin、endpoint：已验证米制端点；minimum、resolution：地图网格参数。
# 输出：
#   keys：保留采样顺序、去除相邻重复后的体素键。
def reference(origin, endpoint, minimum, resolution):
    count = max(1, math.ceil(math.dist(origin, endpoint) / (resolution * .45)))
    delta = tuple(endpoint[a] - origin[a] for a in range(3))
    keys = []
    for index in range(count + 1):
        ratio = index / count
        key = tuple(math.floor((origin[a] + delta[a] * ratio - minimum[a]) * (1. / resolution))
                    for a in range(3))
        if not keys or keys[-1] != key:
            keys.append(key)
    return keys


# 功能：
#   对随机、反向、零长、网格边界及跨数组分块射线逐键比对，不只比较数量。
# 输入：
#   resolution：本轮栅格边长。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('resolution', [.025, .1, .25, .37, 1.])
def test_sampling_preserves_reference_voxel_sequence(resolution):
    minimum = (-30., -30., -30.)
    world = MetricVoxelMap(resolution_m=resolution,
        minimum_bound_m=Vector3(x=-30,y=-30,z=-30), maximum_bound_m=Vector3(x=30,y=30,z=30))
    rng = random.Random(805)
    pairs = [(tuple(rng.uniform(-29.,29.) for _ in range(3)),
              tuple(rng.uniform(-29.,29.) for _ in range(3))) for _ in range(90)]
    pairs += [((-30.,-30.,-30.), (30.,30.,30.)), ((1.,1.,1.),(1.,1.,1.)),
              ((0.,0.,0.), (math.nextafter(10., -math.inf),10.,0.)),
              ((0.,0.,0.), (math.nextafter(10., math.inf),10.,0.))]
    for origin, endpoint in pairs:
        ray = RangeRayObservation(origin_m=Vector3(x=origin[0],y=origin[1],z=origin[2]),
            endpoint_m=Vector3(x=endpoint[0],y=endpoint[1],z=endpoint[2]), hit=True,
            confidence=.95, observed_at_monotonic_seconds=1.)
        actual = world._ray_keys(ray)
        assert actual == reference(origin, endpoint, minimum, resolution)
        assert all(type(value) is int for key in actual for value in key)


# 功能：
#   核对最大合法地图偏移、分块边界和反向遍历，不允许整数溢出或块间重复体素。
# 输入：
#   offset：大地图坐标偏移。
# 输出：
#   None：逐键断言与独立标量公式完全相同。
@pytest.mark.parametrize("offset", [-1e9, 1e9-100.])
def test_chunked_sampling_preserves_large_map_offsets(offset):
    minimum = (offset, offset, offset)
    world = MetricVoxelMap(resolution_m=.025,
        minimum_bound_m=Vector3(x=offset, y=offset, z=offset),
        maximum_bound_m=Vector3(x=offset+100., y=offset+100., z=offset+100.))
    origin, endpoint = minimum, (offset+70., offset+50., offset+1.)
    for start, end in ((origin, endpoint), (endpoint, origin)):
        ray = RangeRayObservation(origin_m=Vector3(x=start[0], y=start[1], z=start[2]),
            endpoint_m=Vector3(x=end[0], y=end[1], z=end[2]), hit=True,
            confidence=.95, observed_at_monotonic_seconds=1.)
        actual = world._ray_keys(ray)
        assert actual == reference(start, end, minimum, .025)
        assert all(previous != current for previous, current in
                   zip(actual, actual[1:], strict=False))
