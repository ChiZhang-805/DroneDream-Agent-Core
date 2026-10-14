from __future__ import annotations

import math

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap
from scripts.runtime_depth_safety_worker import (
    _point_segment_distance,
    _qualified_route_corridor_keys,
)


def _world(*, extent: float = 10.0, resolution: float = 0.5) -> MetricVoxelMap:
    return MetricVoxelMap(
        resolution_m=resolution,
        minimum_bound_m=Vector3(x=-extent, y=-extent, z=-extent),
        maximum_bound_m=Vector3(x=extent, y=extent, z=extent),
        unknown_is_occupied=True,
    )


def _brute_force_keys(
    world: MetricVoxelMap,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    padding_m: float,
) -> set[tuple[int, int, int]]:
    low = Vector3(
        **{
            field: max(
                getattr(world.minimum_bound_m, field),
                min(start[axis], end[axis]) - padding_m,
            )
            for axis, field in enumerate(("x", "y", "z"))
        }
    )
    high = Vector3(
        **{
            field: min(
                getattr(world.maximum_bound_m, field) - world.resolution_m * 0.01,
                max(start[axis], end[axis]) + padding_m,
            )
            for axis, field in enumerate(("x", "y", "z"))
        }
    )
    first, last = world.key_for(low), world.key_for(high)
    result: set[tuple[int, int, int]] = set()
    for x in range(first[0], last[0] + 1):
        for y in range(first[1], last[1] + 1):
            for z in range(first[2], last[2] + 1):
                key = (x, y, z)
                center = world.center_for(key)
                if _point_segment_distance(
                    (center.x, center.y, center.z), start, end
                ) <= padding_m:
                    result.add(key)
    return result


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((-4.0, 1.0, 0.0), (4.0, 1.0, 0.0)),
        ((-4.0, -3.0, -1.0), (4.0, 3.0, 2.0)),
        ((1.0, -2.0, -4.0), (1.0, -2.0, 4.0)),
        ((-3.7, 2.2, 1.1), (-3.7, 2.2, 1.1)),
        ((-9.8, -8.5, -1.0), (8.9, 7.4, 2.0)),
    ],
)
def test_slab_corridor_enumeration_matches_brute_force_capsule(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> None:
    world = _world()

    actual, inspected = _qualified_route_corridor_keys(
        world=world,
        start=start,
        end=end,
        padding_m=1.25,
    )

    assert actual == _brute_force_keys(world, start, end, 1.25)
    assert inspected >= len(actual)


def test_long_diagonal_corridor_avoids_full_three_dimensional_box() -> None:
    world = _world(extent=80.0, resolution=0.25)
    start = (-60.0, -45.0, 0.0)
    end = (60.0, 45.0, 5.0)
    padding_m = 2.0

    keys, inspected = _qualified_route_corridor_keys(
        world=world,
        start=start,
        end=end,
        padding_m=padding_m,
    )
    first = world.key_for(Vector3(x=-62.0, y=-47.0, z=-2.0))
    last = world.key_for(Vector3(x=62.0, y=47.0, z=7.0))
    full_box = math.prod(last[index] - first[index] + 1 for index in range(3))

    assert keys
    assert inspected < full_box // 5


def test_corridor_enumeration_keeps_the_traversal_budget() -> None:
    with pytest.raises(ValueError, match="traversal safety budget"):
        _qualified_route_corridor_keys(
            world=_world(),
            start=(-8.0, -8.0, 0.0),
            end=(8.0, 8.0, 0.0),
            padding_m=2.0,
            maximum_candidates=1,
        )
