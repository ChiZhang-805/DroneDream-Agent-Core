"""Independent sequential reference for the occupancy hot path, not a runtime fallback."""

import math
import random
from types import MethodType

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap, _VoxelEvidence


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
    if evidence.log_odds >= math.log(.65 / .35):
        self._occupied_keys.add(key)
        self._observed_free_keys.discard(key)
    elif evidence.log_odds <= math.log(.35 / .65):
        self._observed_free_keys.add(key)
        self._occupied_keys.discard(key)
    else:
        self._occupied_keys.discard(key)
        self._observed_free_keys.discard(key)


def world(*, reference=False):
    result = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-4, y=-4, z=-4),
                            maximum_bound_m=Vector3(x=4, y=4, z=4))
    if reference:
        result._update_log_odds = MethodType(reference_update, result)
    return result


def assert_equivalent(first, second):
    def values(item):
        return {key: (e.log_odds.hex(), e.observations, e.latest_monotonic_seconds.hex())
                for key, e in item._evidence.items()}
    assert values(first) == values(second)
    assert first._occupied_keys == second._occupied_keys
    assert first._observed_free_keys == second._observed_free_keys
    assert first._evidence_keys_by_chunk == second._evidence_keys_by_chunk
    assert first._non_static_evidence_count == second._non_static_evidence_count
    assert first.observation_count == second.observation_count
    assert (first.latest_observation_monotonic_seconds
            == second.latest_observation_monotonic_seconds)


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
