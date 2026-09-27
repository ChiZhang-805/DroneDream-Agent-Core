"""Independent sequential reference for the occupancy hot path, not a runtime fallback."""

import math
import random
from types import MethodType

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap, _VoxelEvidence


# 功能：
#   用逐次标量更新和原概率公式作独立基准，不复用生产分类近似或缓存实现。
# 输入：
#   self、key：独立测试地图和体素键；measurement、observed_at：证据增量与原始时刻。
# 输出：
#   无。
def reference_update(self, key, *, measurement, observed_at):
    evidence = self._evidence.get(key)
    if evidence is None:
        evidence = _VoxelEvidence(owner=self._evidence_owner)
        self._evidence[key] = evidence
        if key not in self._known_static_occupied_keys and key not in self._known_static_free_keys:
            self._non_static_evidence_count += 1
        self._evidence_keys_by_chunk.setdefault(self._chunk_for_key(key), set()).add(key)
    elif evidence.owner is not self._evidence_owner:
        evidence = _VoxelEvidence(evidence.log_odds, evidence.observations,
                                  evidence.latest_monotonic_seconds, self._evidence_owner)
        self._evidence[key] = evidence
    evidence.log_odds = max(-6.0, min(6.0, evidence.log_odds + measurement))
    evidence.observations += 1
    evidence.latest_monotonic_seconds = max(evidence.latest_monotonic_seconds, observed_at)
    # 阈值必须按用户可见概率定义。对数阈值的浮点近似不是独立概率基准。
    probability = 1. / (1. + math.exp(-evidence.log_odds))
    if probability >= .65:
        self._occupied_keys.add(key)
        self._observed_free_keys.discard(key)
    elif probability <= .35:
        self._observed_free_keys.add(key)
        self._occupied_keys.discard(key)
    else:
        self._occupied_keys.discard(key)
        self._observed_free_keys.discard(key)


# 功能：
#   创建独立地图，可显式改用仅供测试的逐标量更新器。
# 输入：
#   reference：是否注入基准更新器。
# 输出：
#   result：不与生产地图共享证据的测试对象。
def world(*, reference=False):
    result = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-4, y=-4, z=-4),
                            maximum_bound_m=Vector3(x=4, y=4, z=4))
    if reference:
        result._update_log_odds = MethodType(reference_update, result)
    return result


# 功能：
#   对比原始浮点位、索引、计数及时间，避免只看最终分类而漏掉证据变化。
# 输入：
#   first、second：生产实现与独立基准地图。
# 输出：
#   无。
def assert_equivalent(first, second):
    # 功能：
    #   把内部证据转换为可逐位对照的表示。
    # 输入：
    #   item：待比较的地图。
    # 输出：
    #   evidence_values：体素键到证据值的独立字典。
    def values(item):
        evidence_values = {key: (e.log_odds.hex(), e.observations, e.latest_monotonic_seconds.hex())
                           for key, e in item._evidence.items()}
        return evidence_values
    assert values(first) == values(second)
    assert first._occupied_keys == second._occupied_keys
    assert first._observed_free_keys == second._observed_free_keys
    assert first._evidence_keys_by_chunk == second._evidence_keys_by_chunk
    assert first._non_static_evidence_count == second._non_static_evidence_count
    assert first.observation_count == second.observation_count
    assert (first.latest_observation_monotonic_seconds
            == second.latest_observation_monotonic_seconds)


# 功能：
#   覆盖阈值相邻浮点数、饱和、反转及时间倒退，逐次核对而不只比较最终结果。
# 输入：
#   measurement：本次反复施加的证据增量。
# 输出：
#   无。
@pytest.mark.parametrize("measurement", [
    -12., -6., -.7, math.log(.35/.65), math.nextafter(math.log(.35/.65), 0),
    -0., 0., math.nextafter(math.log(.65/.35), 0), math.log(.65/.35), .7, 6., 12.,
])
def test_every_transition_matches_sequential_reference(measurement):
    actual, expected = world(), world(reference=True)
    key = (1, 1, 1)
    # Saturation, reversal, threshold crossing, unchanged and regressing times.
    for index, delta in enumerate([measurement]*4 + [-measurement]*12 + [measurement]*8):
        stamp = 7. if index % 3 else 8.
        actual._update_log_odds(key, measurement=delta, observed_at=stamp)
        expected._update_log_odds(key, measurement=delta, observed_at=stamp)
        assert_equivalent(actual, expected)


# 功能：
#   检查随机更新及克隆后写入保持逐位一致，不修改原地图或另一份克隆。
# 输入：
#   无：使用固定种子生成合成证据。
# 输出：
#   无。
def test_random_updates_and_snapshot_detachment_are_bitwise_equivalent():
    rng = random.Random(805)
    actual, expected = world(), world(reference=True)
    snapshots = []
    for index in range(1500):
        key = tuple(rng.randrange(8) for _ in range(3))
        measurement = rng.uniform(-6, 6)
        stamp = rng.uniform(0, 100)
        for item in (actual, expected):
            item._update_log_odds(key, measurement=measurement, observed_at=stamp)
        assert_equivalent(actual, expected)
        if index in {100, 500, 1000}:
            snapshots.append((actual.navigation_clone(), expected.navigation_clone()))
    for first, second in snapshots:
        assert_equivalent(first, second)
        # Snapshot writes must not affect the live map or another snapshot.
        second._update_log_odds = MethodType(reference_update, second)
        for key in list(first._evidence):
            first._update_log_odds(key, measurement=-12., observed_at=200.)
            second._update_log_odds(key, measurement=-12., observed_at=200.)
        assert_equivalent(first, second)
        assert_equivalent(actual, expected)


# 功能：
#   从实际射线集成入口对照自由空间、命中证据及来源时间。
# 输入：
#   无：使用固定种子的合成射线。
# 输出：
#   无。
def test_ray_hits_and_free_space_keep_exact_indexes_and_observation_times():
    actual, expected = world(), world(reference=True)
    rng = random.Random(902)
    for index in range(120):
        ray = RangeRayObservation(
            origin_m=Vector3(x=0, y=0, z=0),
            endpoint_m=Vector3(**dict(zip("xyz", (rng.uniform(-4, 4) for _ in range(3)),
                                        strict=True))),
            hit=index % 3 != 0, confidence=rng.uniform(.05, .99),
            observed_at_monotonic_seconds=float(index % 10),
        )
        actual.integrate_ray(ray)
        expected.integrate_ray(ray)
        assert_equivalent(actual, expected)
