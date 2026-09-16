import itertools

import pytest

from dronedream_agent_core.contracts import RangeRayObservation, Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap


def world():
    return MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=0, y=0, z=0),
                          maximum_bound_m=Vector3(x=5, y=5, z=5))


def ray(*, hit, stamp=1., confidence=.92):
    return RangeRayObservation(origin_m=Vector3(x=.125, y=.125, z=.125),
        endpoint_m=Vector3(x=1.125 if hit else 3.125, y=.125, z=.125),
        hit=hit, confidence=confidence, observed_at_monotonic_seconds=stamp)


def test_measured_obstacle_cannot_be_cleared_by_other_pixels_in_the_same_scan():
    results = []
    for observations in itertools.permutations([ray(hit=True), ray(hit=False), ray(hit=False)]):
        target = world()
        target.integrate_scan(observations)
        key = target.key_for(ray(hit=True).endpoint_m)
        assert target.is_occupied(key)
        assert not target.is_observed_free(key)
        results.append({k: v.log_odds for k, v in target._evidence.items()})
    assert all(value == results[0] for value in results)


def test_correlated_duplicate_pixels_do_not_manufacture_confidence():
    once, repeated = world(), world()
    once.integrate_scan([ray(hit=True)])
    repeated.integrate_scan([ray(hit=True)]*100)
    assert {k: v.log_odds for k, v in once._evidence.items()} == {
        k: v.log_odds for k, v in repeated._evidence.items()}


def test_new_hit_immediately_revokes_old_free_space():
    target = world()
    key = target.key_for(ray(hit=True).endpoint_m)
    for stamp in range(1, 6):
        target.integrate_scan([ray(hit=False, stamp=stamp)])
    assert target.is_observed_free(key)
    target.integrate_scan([ray(hit=True, stamp=6), ray(hit=False, stamp=6)])
    assert target.is_occupied(key)
    target.integrate_scan([ray(hit=False, stamp=7)])
    assert not target.is_observed_free(key)  # One contradictory scan is insufficient.
    target.integrate_scan([ray(hit=False, stamp=8)])
    assert target.is_observed_free(key)


def test_unknown_hit_confidence_never_means_free_space():
    target = world()
    target.integrate_scan([ray(hit=True, confidence=.2), ray(hit=False)])
    key = target.key_for(ray(hit=True).endpoint_m)
    assert not target.is_observed_free(key)


def test_live_unknown_overrides_static_free_in_exported_snapshot():
    target = world()
    endpoint = ray(hit=True).endpoint_m
    target.seed_known_static_region([(endpoint, False)], source_sha256="a"*64)
    target.integrate_scan([ray(hit=True, confidence=.2)])
    key = target.key_for(endpoint)
    assert not target.is_observed_free(key)
    assert key not in target.snapshot().free_voxels


def test_single_independent_low_quality_hit_never_clears_its_endpoint():
    target = world()
    target.integrate_ray(ray(hit=True, confidence=.2))
    assert not target.is_observed_free(target.key_for(ray(hit=True).endpoint_m))


def test_frozen_map_does_not_receive_later_static_prior_edits():
    target = world()
    endpoint = ray(hit=True).endpoint_m
    target.seed_known_static_region([(endpoint, False)], source_sha256="a"*64)
    clone = target.navigation_clone()
    target.seed_known_static_region([(endpoint, True)], source_sha256="a"*64)
    assert clone.is_observed_free(clone.key_for(endpoint))
    assert not target.is_observed_free(target.key_for(endpoint))


def test_prepared_scan_is_detached_and_cannot_cross_map_or_be_replayed():
    target, other = world(), world()
    source = ray(hit=True)
    prepared = target.prepare_scan([source])
    assert target.observation_count == 0
    source.endpoint_m.x = 4.
    with pytest.raises(ValueError, match="PREPARED_STATE_CHANGED"):
        other.commit_scan(prepared)
    target.commit_scan(prepared)
    assert target.is_occupied(target.key_for(ray(hit=True).endpoint_m))
    with pytest.raises(ValueError, match="PREPARED_STATE_CHANGED"):
        target.commit_scan(prepared)


@pytest.mark.parametrize("invalid", ["time", "replay", "bounds"])
def test_invalid_scan_has_no_partial_map_updates(invalid):
    target = world()
    target.integrate_scan([ray(hit=False)])
    previous = {k: v.log_odds for k, v in target._evidence.items()}
    observations = [ray(hit=True, stamp=2.)]
    if invalid == "time":
        observations.append(ray(hit=False, stamp=3.))
    elif invalid == "replay":
        observations = [ray(hit=True)]
    else:
        observations.append(RangeRayObservation(origin_m=Vector3(x=0, y=0, z=0),
            endpoint_m=Vector3(x=6, y=0, z=0), hit=False, confidence=.9,
            observed_at_monotonic_seconds=2.))
    with pytest.raises(ValueError):
        target.integrate_scan(observations)
    assert {k: v.log_odds for k, v in target._evidence.items()} == previous
