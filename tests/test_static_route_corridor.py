"""Exact static-prior coverage and bounded work, without a simulator."""

import itertools
import math

import pytest

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap


def voxel_world(low=-3., high=12., resolution=.5):
    return MetricVoxelMap(
        resolution_m=resolution,
        minimum_bound_m=Vector3(x=low, y=low, z=low),
        maximum_bound_m=Vector3(x=high, y=high, z=high),
        unknown_is_occupied=True,
    )


def exhaustive_capsules(world, segments, padding):
    """Previous exact enumeration, used only on small fixtures as an oracle."""
    result = set()
    minimum = tuple(getattr(world.minimum_bound_m, a) for a in ("x", "y", "z"))
    maximum = tuple(getattr(world.maximum_bound_m, a) for a in ("x", "y", "z"))
    for start, end in segments:
        first = world.key_for(tuple(max(minimum[i], min(start[i], end[i]) - padding)
                                    for i in range(3)))
        last = world.key_for(tuple(min(maximum[i] - world.resolution_m * .01,
                                      max(start[i], end[i]) + padding) for i in range(3)))
        for key in itertools.product(*(range(first[i], last[i] + 1) for i in range(3))):
            center = world.center_for(key)
            position = (center.x, center.y, center.z)
            if worker._point_segment_distance(position, start, end) <= padding:
                result.add(key)
    return result


@pytest.mark.parametrize("end", list(itertools.permutations((8., 3.25, -.75))))
def test_slab_capsule_equals_exhaustive_voxel_set_in_all_axes(end):
    world = voxel_world()
    segments = [((-.75, -.5, .25), end), (end, (1.5, 4., 5.5))]
    keys, inspected = worker._qualified_route_corridor_keys(world, segments, 1.25)
    assert keys == exhaustive_capsules(world, segments, 1.25)
    assert inspected >= len(keys)


def test_capsule_preserves_clipped_bounds_endpoint_caps_and_zero_length():
    world = voxel_world()
    segments = [((-2.75, -2.75, -2.75), (11.75, 11.75, 11.75)),
                ((2., 2., 2.), (2., 2., 2.)),
                ((4., 3., 2.), (-1., 0., 1.))]
    keys, _ = worker._qualified_route_corridor_keys(world, segments, 1.)
    assert keys == exhaustive_capsules(world, segments, 1.)


def test_long_diagonal_round_trip_avoids_empty_box_and_repeated_return_work():
    world = voxel_world(high=260., resolution=.25)
    start, goal = (0., 0., 2.), (250., 250., 2.)
    one_way, inspected = worker._qualified_route_corridor_keys(world, [(start, goal)], 2.)
    round_trip, round_trip_work = worker._qualified_route_corridor_keys(
        world, [(start, goal), (goal, start)] * 10, 2.)
    old_box_work = math.prod((1017, 1017, 17))
    assert old_box_work > worker._STATIC_ROUTE_TRAVERSAL_BUDGET
    assert inspected < 600_000
    assert round_trip == one_way
    assert round_trip_work == inspected
    for position in (start, (125., 125., 2.), goal):
        assert world.key_for(position) in one_way
    assert world.key_for((125., 130., 2.)) not in one_way


def test_true_corridor_work_still_hits_unchanged_budget(monkeypatch):
    world = voxel_world()
    monkeypatch.setattr(worker, "_STATIC_ROUTE_TRAVERSAL_BUDGET", 100)
    with pytest.raises(ValueError, match="traversal safety budget"):
        worker._qualified_route_corridor_keys(world, [((0., 0., 0.), (8., 8., 8.))], 2.)


def test_unique_voxel_budget_remains_independent(monkeypatch):
    world = voxel_world()
    monkeypatch.setattr(worker, "_STATIC_ROUTE_VOXEL_BUDGET", 10)
    with pytest.raises(ValueError, match="voxel budget"):
        worker._qualified_route_corridor_keys(world, [((0., 0., 0.), (8., 8., 8.))], 2.)
