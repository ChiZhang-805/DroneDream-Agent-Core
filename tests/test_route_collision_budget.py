"""Detailed ground maps retain full geometry without charging skipped queries."""

import json

import pytest

from dronedream_agent_core import collision
from dronedream_agent_core.contracts import GraphRoute, Vector3


def _route(xs):
    return GraphRoute(
        start_node="start", goal_node="goal",
        node_ids=["start", *[f"p{i}" for i in range(len(xs) - 2)], "goal"],
        edge_ids=[f"e{i}" for i in range(len(xs) - 1)],
        positions_m=[Vector3(x=float(x), y=0., z=2.) for x in xs],
        route_length_m=sum(abs(b - a) for a, b in zip(xs, xs[1:], strict=False)),
        all_edges_flight_verified=False,
    )


def _box(name, x, y):
    return dict(name=name, center_x=x, center_y=y, center_z=2.,
                size_x=.1, size_y=.5, size_z=2.)


def _semantic(path, primitives, *, traversable=True):
    value = {"collision_primitives": primitives}
    if traversable:
        value["traversability"] = {
            "schema_version": "dronedream.traversability.v1", "required": True,
            "minimum_agl_m": 0., "maximum_agl_m": 12.,
            "corridors": [{"id": "long-road", "kind": "road", "ground_z_m": 0.,
                           "centerline_xy_m": [[-2., 0.], [852., 0.]], "width_m": 8.}],
            "areas": [], "exclusions": [],
        }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("obstructed", [False, True])
def test_long_route_with_distant_cells_preserves_exact_clearance(tmp_path, obstructed):
    near = _box("near-wall", 425., 0. if obstructed else 2.)
    distant = [_box(f"distant-{i}", float(i % 64), 100. + i // 64)
               for i in range(3200)]
    route = _route([0., 850.])
    options = dict(vehicle_diameter_m=.76, vehicle_height_m=.43, sample_interval_m=.05)
    baseline = collision.validate_route_clearance(
        route, _semantic(tmp_path / "baseline.json", [near]), **options)
    actual = collision.validate_route_clearance(
        route, _semantic(tmp_path / "detailed.json", [near, *distant]), **options)
    assert actual.sample_count * actual.primitive_count > 50_000_000
    assert actual.accepted is (not obstructed)
    assert actual.accepted == baseline.accepted
    assert actual.minimum_clearance_m == baseline.minimum_clearance_m
    assert actual.minimum_clearance_point == baseline.minimum_clearance_point
    assert actual.minimum_clearance_primitive == baseline.minimum_clearance_primitive
    assert actual.segment_minimum_clearances_m == baseline.segment_minimum_clearances_m
    assert actual.collision_count == baseline.collision_count
    assert actual.collisions == baseline.collisions


@pytest.mark.parametrize("traversable", [False, True])
def test_relevant_query_budget_accumulates_across_segments(tmp_path, monkeypatch, traversable):
    path = _semantic(tmp_path / "map.json", [_box("a", 0., 2.), _box("b", 1., 2.)],
                     traversable=traversable)
    calls = []
    original = collision._clearance

    def observe(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    # Each segment alone fits. Two do not; neither the route nor the spatial
    # filter may reset the ceiling after completing the first segment.
    monkeypatch.setattr(collision, "_MAX_ROUTE_COLLISION_WORK", 10)
    monkeypatch.setattr(collision, "_clearance", observe)
    with pytest.raises(ValueError, match="route collision query budget"):
        collision.validate_route_clearance(
            _route([0., 1., 2.]), path, vehicle_diameter_m=.76,
            vehicle_height_m=.43, sample_interval_m=.5)
    assert len(calls) == 6


def test_distant_bounds_scans_still_have_a_total_budget(tmp_path, monkeypatch):
    path = _semantic(tmp_path / "map.json", [_box(f"far-{i}", 0., 100.) for i in range(6)])
    monkeypatch.setattr(collision, "_MAX_ROUTE_COLLISION_WORK", 10)
    with pytest.raises(ValueError, match="route collision query budget"):
        collision.validate_route_clearance(
            _route([0., 1., 2.]), path, vehicle_diameter_m=.76,
            vehicle_height_m=.43, sample_interval_m=.5)


def test_actual_full_query_ceiling_is_not_increased(tmp_path, monkeypatch):
    path = _semantic(tmp_path / "dense.json", [
        _box(f"near-{i}", 425., 2.) for i in range(3200)])

    def unexpected_query(*args, **kwargs):
        pytest.fail("over-budget exact queries must be rejected before evaluation")

    monkeypatch.setattr(collision, "_clearance", unexpected_query)
    with pytest.raises(ValueError, match="route collision query budget"):
        collision.validate_route_clearance(
            _route([0., 850.]), path, vehicle_diameter_m=.76,
            vehicle_height_m=.43, sample_interval_m=.05)


def test_total_sample_budget_remains_independent_of_spatial_pruning(tmp_path, monkeypatch):
    path = _semantic(tmp_path / "map.json", [_box("far", 0., 100.)])
    monkeypatch.setattr(collision, "_MAX_ROUTE_SAMPLES", 5)
    with pytest.raises(ValueError, match="route collision query budget"):
        collision.validate_route_clearance(
            _route([0., 1., 2.]), path, vehicle_diameter_m=.76,
            vehicle_height_m=.43, sample_interval_m=.5)
