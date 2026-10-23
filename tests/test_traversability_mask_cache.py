"""Static mask reuse must never reuse live occupancy or revive expired evidence."""
import math
from itertools import product

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.traversability import TraversableAirspace


def mask(*, blocked=False):
    layer = {"schema_version": "dronedream.traversability.v1", "required": True,
             "minimum_agl_m": 0., "maximum_agl_m": 12., "corridors": [],
             "areas": [{"id": "ground", "kind": "lawn", "ground_z_m": 0.,
                        "polygon_xy_m": [[-4., -4.], [40., -4.], [40., 40.], [-4., 40.]]}],
             "exclusions": ([{"id": "tree", "kind": "tree-canopy",
                              "polygon_xy_m": [[0., 0.], [4., 0.], [4., 4.], [0., 4.]]}]
                            if blocked else [])}
    return TraversableAirspace({"traversability": layer}, [], radius_m=0., half_height_m=0.)


def world():
    result = MetricVoxelMap(resolution_m=.5, minimum_bound_m=Vector3(x=0., y=0., z=0.),
                            maximum_bound_m=Vector3(x=40., y=40., z=12.))
    result.bind_traversable_airspace(mask(), source_sha256="a" * 64)
    return result


@pytest.mark.parametrize("blocked", [False, True])
def test_repeated_cells_reuse_only_exact_static_mask(monkeypatch, blocked):
    grid = world()
    domain = mask(blocked=blocked)
    grid.bind_traversable_airspace(domain, source_sha256="a" * 64)
    point = Vector3(x=2.25, y=2.25, z=2.25)
    key = grid.key_for(point)
    grid.seed_known_static_region([(point, False)], source_sha256="a" * 64)
    calls = []
    original = domain.clearance
    def clearance(p):
        calls.append(p)
        return original(p)
    monkeypatch.setattr(domain, "clearance", clearance)
    for _ in range(100):
        assert grid.is_observed_free(key) is not blocked
        assert grid.is_occupied(key) is blocked
    assert calls == [grid._center_point_for_key(key)]


def test_live_hit_and_expiry_still_revoke_static_free_cell():
    grid = world()
    point = Vector3(x=2.25, y=2.25, z=2.25)
    key = grid.key_for(point)
    grid.seed_known_static_region([(point, False)], source_sha256="a" * 64)
    assert grid.is_observed_free(key)
    clone = grid.navigation_clone()
    assert not clone._traversability_mask_cache
    grid.mark_box(minimum_m=point, maximum_m=point, occupied=True, confidence=.9,
                  observed_at_monotonic_seconds=1.)
    assert grid.is_occupied(key) and not grid.is_observed_free(key)
    assert clone.is_observed_free(key) and not clone.is_occupied(key)
    grid.mark_box(minimum_m=point, maximum_m=point, occupied=False, confidence=.99,
                  observed_at_monotonic_seconds=2.)
    assert grid.is_observed_free(key) and not grid.is_occupied(key)
    grid.prune_live_evidence(center_m=point, radius_m=5., now_monotonic_seconds=10.,
                             maximum_age_seconds=2.)
    assert grid.is_occupied(key) and not grid.is_observed_free(key)
    assert clone.is_observed_free(key)


def test_expired_measured_free_cell_is_not_cached_as_free():
    grid = world()
    point = Vector3(x=2.25, y=2.25, z=2.25)
    key = grid.key_for(point)
    grid.mark_box(minimum_m=point, maximum_m=point, occupied=False,
                  observed_at_monotonic_seconds=1.)
    assert grid.is_observed_free(key)
    grid.prune_live_evidence(center_m=point, radius_m=5., now_monotonic_seconds=10.,
                             maximum_age_seconds=2.)
    assert not grid.is_observed_free(key) and grid.is_occupied(key)


def test_binding_refresh_discards_cached_mask_and_other_asset_is_rejected():
    grid = world()
    key = grid.key_for((2.25, 2.25, 2.25))
    assert not grid._traversability_blocks(key)
    grid.bind_traversable_airspace(mask(blocked=True), source_sha256="a" * 64)
    assert grid._traversability_blocks(key)
    with pytest.raises(ValueError, match="METRIC_TRAVERSABILITY_SOURCE_CHANGED"):
        grid.bind_traversable_airspace(mask(), source_sha256="b" * 64)
    assert grid._traversability_blocks(key)


def test_cache_is_bounded_and_eviction_preserves_exact_boundary_classification():
    grid = world()
    domain = grid._operating_airspace
    for key in product(range(22), range(22), range(18)):
        expected = domain.clearance(grid._center_point_for_key(key))[0] < math.sqrt(3.) * .25
        assert grid._traversability_blocks(key) is expected
    assert len(grid._traversability_mask_cache) == 8192
    assert (0, 0, 0) not in grid._traversability_mask_cache
    assert grid._traversability_blocks((0, 0, 0))
    assert len(grid._traversability_mask_cache) == 8192
