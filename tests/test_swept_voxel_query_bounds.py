"""Exact segment/voxel oracle for narrowed, per-query neighborhood enumeration."""
import random
from itertools import product

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap, _segment_hits_box


def _world():
    world = MetricVoxelMap(resolution_m=.5, minimum_bound_m=Vector3(x=0., y=0., z=0.),
                           maximum_bound_m=Vector3(x=4., y=4., z=4.))
    cells = [(world._center_point_for_key(key), False) for key in product(range(8), repeat=3)]
    world.seed_known_static_region(cells, source_sha256="a" * 64)
    world.bind_qualified_route([Vector3(x=1., y=1., z=1.), Vector3(x=3., y=3., z=3.)],
                               route_sha256="b" * 64)
    return world


def _exhaustive(world, points, margin, *, qualified=False, temporary=frozenset()):
    # Enumerate every voxel in this small test domain, including a complete
    # exterior border, rather than reproducing the production sampling loop.
    blocked = []
    for key in product(range(-3, 11), repeat=3):
        free = (key not in temporary and not world.is_occupied(key)
                and (not world.unknown_is_occupied or world.is_observed_free(key)))
        if qualified:
            free = (key in world._known_static_free_keys
                    and key not in world._invalidated_static_free_keys
                    and world.is_observed_free(key) and not world.is_occupied(key))
        if not free:
            blocked.append(world._center_point_for_key(key))
    for a, b in list(zip(points, points[1:], strict=False)) or [(points[0], points[0])]:
        if any(_segment_hits_box(tuple(a), tuple(b), center, .25 + margin) for center in blocked):
            return False
    return True


@pytest.mark.parametrize("qualified", [False, True])
def test_sweeps_match_exhaustive_boxes_with_turns_grazing_and_revisited_cells(qualified):
    world = _world()
    obstacle = Vector3(x=2.25, y=2.25, z=2.25)
    world.mark_box(minimum_m=obstacle, maximum_m=obstacle, occupied=True,
                   observed_at_monotonic_seconds=1.)
    rng = random.Random(772)
    paths = [[tuple(rng.uniform(.6, 3.4) for _ in range(3)) for _ in range(3)]
             for _ in range(60)]
    paths += [[(1., 1., 2.25), (1., 3., 2.25), (2.25, 2.25, 2.25)]]
    paths += [[(1., 1.75 + epsilon, 2.25), (3., 1.75 + epsilon, 2.25)]
              for epsilon in (-1e-10, 0., 1e-10)]
    for points in paths:
        margin = rng.choice((0., .1, .25))
        vectors = [Vector3(x=p[0], y=p[1], z=p[2]) for p in points]
        expected = _exhaustive(world, points, margin, qualified=qualified)
        actual = (world.qualified_static_path_covered(vectors, extra_clearance_m=margin)
                  if qualified else world.path_clearance_valid(
                      vectors, required_clearance_m=margin))
        assert actual == expected, (points, margin)


def test_free_cell_reuse_ends_at_query_boundary(monkeypatch):
    world = _world()
    points = [Vector3(x=x, y=1.25, z=1.25) for x in (1.1, 1.2, 1.3, 1.4, 1.5)]
    calls = []
    original = world.is_occupied

    def occupied(key):
        calls.append(key)
        return original(key)

    monkeypatch.setattr(world, "is_occupied", occupied)
    assert world.path_clearance_valid(points, required_clearance_m=.1)
    assert len(calls) == len(set(calls))
    obstacle = Vector3(x=1.25, y=1.25, z=1.25)
    world.mark_box(minimum_m=obstacle, maximum_m=obstacle, occupied=True,
                   observed_at_monotonic_seconds=1.)
    assert not world.path_clearance_valid(points, required_clearance_m=.1)
    assert not world.qualified_static_path_covered(points, extra_clearance_m=.1)
    world.prune_live_evidence(center_m=obstacle, radius_m=5., now_monotonic_seconds=10.,
                              maximum_age_seconds=2.)
    assert not world.path_clearance_valid(points, required_clearance_m=.1)


def test_aabb_clipping_retains_corner_touch_and_original_work_budget(monkeypatch):
    world = _world()
    low, high = world._swept_key_bounds((1., 1., 1.), (2., 2., 2.), .25)
    assert low == [1, 1, 1] and high == [4, 4, 4]
    # Narrowing actual enumeration must not bypass the original bounded-work contract.
    monkeypatch.setattr("dronedream_agent_core.local_world_model._MAX_GRID_WORK", 1000)
    points = [Vector3(x=1., y=1., z=1.), Vector3(x=3., y=3., z=3.)] * 1000
    with pytest.raises(ValueError, match="WORK_BUDGET"):
        world.path_clearance_valid(points, required_clearance_m=.1)
