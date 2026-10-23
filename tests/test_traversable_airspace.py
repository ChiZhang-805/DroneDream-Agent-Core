import json
import math

import pytest

from dronedream_agent_core.collision import validate_route_clearance
from dronedream_agent_core.contracts import (
    GraphRoute,
    MapAsset,
    MapEdge,
    MapNode,
    RouteQuery,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety
from dronedream_agent_core.traversability import TraversableAirspace


def _ground():
    return {"name": "ground", "semantic": "terrain", "center_x": 3., "center_y": 3.,
            "center_z": -.1, "size_x": 40., "size_y": 40., "size_z": .2}


def _layer():
    return {"schema_version": "dronedream.traversability.v1", "required": True,
            "minimum_agl_m": 0., "maximum_agl_m": 12., "areas": [{
                "id": "open-lawn", "kind": "lawn", "ground_z_m": 0.,
                "polygon_xy_m": [[-2., -3.], [10., -3.], [10., 11.], [-2., 11.]]}],
            "corridors": [], "exclusions": []}


def _write(tmp_path, layer=None, obstacles=(), runtime=()):
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps({"coordinate_frame": "ENU", "collision_primitives": [
        _ground(), *obstacles], "runtime_collision_primitives": list(runtime),
        "traversability": _layer() if layer is None else layer}))
    return path


def _graph():
    return MapAsset(asset_id="traversable-map", name="traversable-map", nodes=[
        MapNode(node_id="start", label="start", semantic="launch",
                position_m=Vector3(x=0., y=0., z=1.)),
        MapNode(node_id="goal", label="goal", semantic="pickup",
                position_m=Vector3(x=6., y=0., z=1.)),
        MapNode(node_id="other", label="other", semantic="corridor",
                position_m=Vector3(x=0., y=1., z=1.)),
    ], edges=[MapEdge(edge_id="unrelated", from_node="start", to_node="other",
                      distance_m=1., minimum_clearance_m=.5, speed_limit_mps=1.)],
        named_entities={"start": "start", "goal": "goal"})


def _route(points):
    return GraphRoute(start_node="start", goal_node="goal",
                      node_ids=["start", *[f"p{i}" for i in range(len(points) - 2)], "goal"],
                      edge_ids=[f"e{i}" for i in range(len(points) - 1)],
                      positions_m=[Vector3(x=p[0], y=p[1], z=p[2]) for p in points],
                      route_length_m=sum(math.dist(a, b) for a, b
                                         in zip(points, points[1:], strict=False)),
                      all_edges_flight_verified=False)


def _report(path, route):
    return validate_route_clearance(route, path, vehicle_diameter_m=.4, vehicle_height_m=.3)


def _planner(path):
    return KnownMapMetricPlanner(graph=_graph(), semantic_path=path,
                                vehicle_diameter_m=.4, vehicle_height_m=.3,
                                policy=MetricPlannerPolicy(resolution_m=.5,
                                    search_padding_m=1., required_clearance_m=.15,
                                    preferred_airspace_weight=0., maximum_expansions=60000))


@pytest.mark.parametrize("altitude", [1., 8., 35.])
def test_building_footprint_is_forbidden_below_and_above_roof(tmp_path, altitude):
    building = {"name": "chemistry", "semantic": "building", "center_x": 3.,
                "center_y": 0., "center_z": 2., "size_x": 2., "size_y": 3., "size_z": 3.}
    path = _write(tmp_path, obstacles=[building])
    planner = _planner(path)
    assert planner.clearance((3., 0., altitude)) < 0
    report = _report(path, _route([(0., 0., altitude), (6., 0., altitude)]))
    assert not report.accepted
    assert any(c.primitive_name.startswith("traversability:") for c in report.collisions)


def test_clear_lawn_and_parking_allow_geometry_routing_without_road_edges(tmp_path):
    layer = _layer()
    layer["areas"][0]["kind"] = "parking"
    path = _write(tmp_path, layer)
    planner = _planner(path)
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert route.route_length_m == pytest.approx(6.)
    assert _report(path, route).accepted
    assert not route.all_edges_flight_verified


def test_tree_canopy_exclusion_also_blocks_flying_under_or_over_it(tmp_path):
    layer = _layer()
    layer["exclusions"] = [{"id": "canopy", "kind": "tree-canopy",
                            "polygon_xy_m": [[2., -1.], [4., -1.], [4., 1.], [2., 1.]]}]
    path = _write(tmp_path, layer)
    for altitude in (1., 10.):
        assert not _report(path, _route([(0., 0., altitude), (6., 0., altitude)])).accepted
    route = _planner(path).plan(RouteQuery(start_node="start", goal_node="goal"))
    assert any(abs(p.y) > 1.3 for p in route.positions_m)
    assert _report(path, route).accepted


def test_runtime_building_footprints_cannot_be_omitted_from_vector_layer(tmp_path):
    building = {"name": "runtime-building", "semantic": "building", "center_x": 3.,
                "center_y": 0., "center_z": 1., "size_x": 1., "size_y": 1.,
                "size_z": 2., "yaw_rad": .4}
    path = _write(tmp_path, runtime=[building])
    assert not _report(path, _route([(0., 0., 8.), (6., 0., 8.)])).accepted


def test_detour_expands_beyond_endpoint_box_and_simplification_preserves_corridor(tmp_path,
                                                                             monkeypatch):
    layer = _layer()
    layer["areas"] = []
    layer["maximum_agl_m"] = 3.
    layer["corridors"] = [{"id": "forest-road", "kind": "path", "width_m": 2.,
                           "ground_z_m": 0.,
                           "centerline_xy_m": [[0., 0.], [0., 8.], [6., 8.], [6., 0.]]}]
    path = _write(tmp_path, layer)
    planner = _planner(path)
    searched = []
    actual = planner._plan_in_bounds

    def record(query, start, goal, low, high, budget, prior):
        searched.append((low, high))
        return actual(query, start, goal, low, high, budget, prior)

    monkeypatch.setattr(planner, "_plan_in_bounds", record)
    # Exercise expanding metric search independently of the centreline roadmap.
    monkeypatch.setattr(planner, "_ground_roadmap_route", lambda *_: None)
    assert not _report(path, _route([(0., 0., 1.), (6., 0., 1.)])).accepted
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert len(searched) > 1
    assert max(p.y for p in route.positions_m) > 7.
    assert route.route_length_m > 18.
    assert _report(path, route).accepted
    with pytest.raises(ValueError, match="endpoint connection"):
        planner._simplify([(0., 0., 1.), (6., 0., 1.)])


def test_aircraft_footprint_and_altitude_are_hard_constraints(tmp_path):
    layer = _layer()
    layer["areas"] = []
    layer["corridors"] = [{"id": "road", "kind": "road", "width_m": 1.,
                           "ground_z_m": 2., "centerline_xy_m": [[0., 0.], [6., 0.]]}]
    path = _write(tmp_path, layer)
    planner = _planner(path)
    assert planner.clearance((3., .4, 3.)) < 0  # Body, not just its centre, must fit.
    assert planner.clearance((3., 0., 1.)) < 0  # Local ground elevation matters.
    assert planner.clearance((3., 0., 14.)) < 0  # Complete body below upper boundary.
    assert _report(path, _route([(0., 0., 3.), (6., 0., 3.)])).accepted


def test_small_exclusion_between_waypoints_is_checked_continuously(tmp_path):
    layer = _layer()
    layer["exclusions"] = [{"id": "small-tree", "kind": "tree-canopy",
                            "polygon_xy_m": [[2.99, -.01], [3.01, -.01],
                                             [3.01, .01], [2.99, .01]]}]
    path = _write(tmp_path, layer)
    planner = _planner(path)
    assert planner.clearance((0., 0., 5.)) > 0
    assert planner.clearance((6., 0., 5.)) > 0
    assert not planner._segment_is_free((0., 0., 5.), (6., 0., 5.))
    assert not _report(path, _route([(0., 0., 5.), (6., 0., 5.)])).accepted


def test_distant_collision_broad_phase_keeps_exact_continuous_clearance_report(tmp_path,
                                                                            monkeypatch):
    import dronedream_agent_core.collision as collision
    nearby = {"name": "tilted-fixture", "semantic": "fixture", "center_x": 3.,
              "center_y": .5, "center_z": 2., "size_x": .3, "size_y": .3,
              "size_z": 3., "roll_rad": .7, "pitch_rad": .4, "yaw_rad": .3}
    distant = [dict(nearby, name=f"distant-{i}", center_x=100. + i * 2.) for i in range(80)]
    path = _write(tmp_path, obstacles=[nearby, *distant])
    route = _route([(0., 0., 1.), (6., 0., 1.)])
    calls = []
    actual = collision._clearance

    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(collision, "_clearance", counted)
    bounded = _report(path, route)
    bounded_count = len(calls)
    calls.clear()
    # Exhaustive oracle: force every real primitive through the narrow phase.
    monkeypatch.setattr(collision, "primitive_bounds", lambda *_: (
        (-1e6, -1e6, -1e6), (1e6, 1e6, 1e6)))
    exhaustive = _report(path, route)
    assert not bounded.accepted
    assert bounded == exhaustive
    assert bounded_count < len(calls) / 10


@pytest.mark.parametrize("bad", [None, {}, {"required": False},
                               {"schema_version": "dronedream.traversability.v1",
                                "required": True, "minimum_agl_m": 0.,
                                "maximum_agl_m": 12., "areas": []}])
def test_malformed_present_layer_never_silently_becomes_free_space(tmp_path, bad):
    path = _write(tmp_path)
    data = json.loads(path.read_text())
    data["traversability"] = bad
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="traversability"):
        _planner(path)
    with pytest.raises(ValueError, match="traversability"):
        _report(path, _route([(0., 0., 1.), (6., 0., 1.)]))


def test_shared_clearance_is_one_lipschitz_across_index_bins():
    import random
    layer = _layer()
    layer["exclusions"] = [{"id": "roof", "kind": "building",
                            "polygon_xy_m": [[2., -1.], [4., -1.], [4., 1.], [2., 1.]]}]
    airspace = TraversableAirspace({"traversability": layer}, [],
                                  radius_m=.2, half_height_m=.15, clearance_cap_m=.65)
    rng = random.Random(81)
    for _ in range(500):
        a = (rng.uniform(-3, 12), rng.uniform(-5, 13), rng.uniform(-1, 14))
        b = tuple(value + rng.uniform(-.5, .5) for value in a)
        assert abs(airspace.clearance(a)[0] - airspace.clearance(b)[0]) <= math.dist(a, b) + 1e-9


def test_subgrid_endpoint_route_preserves_both_authoritative_positions(tmp_path):
    planner = _planner(_write(tmp_path))
    start, goal = Vector3(x=.08, y=.03, z=1.), Vector3(x=.13, y=.07, z=1.)
    route = planner.plan_positions(RouteQuery(start_node="start", goal_node="goal"),
                                   start_position_m=start, goal_position_m=goal)
    assert route.positions_m == [start, goal]
    assert route.route_length_m == pytest.approx(math.hypot(.05, .04))


def test_narrow_corridor_does_not_disappear_between_grid_samples(tmp_path, monkeypatch):
    layer = _layer()
    layer["areas"] = []
    layer["corridors"] = [{"id": "narrow-road", "kind": "path", "width_m": 1.,
                           "ground_z_m": 0., "centerline_xy_m": [
                               [0., 0.], [.13, 4.17], [6.11, 4.17], [6., 0.]]}]
    path = _write(tmp_path, layer)
    planner = _planner(path)
    monkeypatch.setattr(
        planner, "_plan_in_bounds",
        lambda *_: pytest.fail("continuous centreline search should find this road"))
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    assert route.route_length_m > 12.
    assert _report(path, route).minimum_clearance_m >= .15


def test_live_free_space_evidence_and_cloning_cannot_override_canopy_policy():
    layer = _layer()
    layer["exclusions"] = [{"id": "canopy", "kind": "tree-canopy",
                            "polygon_xy_m": [[2., -1.], [4., -1.], [4., 1.], [2., 1.]]}]
    airspace = TraversableAirspace({"traversability": layer}, [],
                                  radius_m=0., half_height_m=0.)
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-3, y=-4, z=0),
                           maximum_bound_m=Vector3(x=12, y=12, z=15), unknown_is_occupied=False)
    world.bind_traversable_airspace(airspace, source_sha256="a" * 64)
    point = Vector3(x=3., y=0., z=8.)
    world.seed_known_static_region([(point, False)], source_sha256="a" * 64)
    for model in (world, world.navigation_clone()):
        key = model.key_for(point)
        assert model.is_occupied(key)
        assert not model.is_observed_free(key)
        assert list(model._classified_navigation_evidence([key])) == [(key, 2)]
        assert model.motion_coverage_identity()["traversability_source_sha256"] == "a" * 64
        assert not model.path_clearance_valid(
            [Vector3(x=0., y=0., z=8.), Vector3(x=6., y=0., z=8.)],
            required_clearance_m=.3, allow_first_position_as_start=True)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        world.seed_known_static_region([(point, False)], source_sha256="b" * 64)


def test_runtime_control_cannot_shortcut_over_canopy_even_with_empty_collision_list():
    layer = _layer()
    layer["exclusions"] = [{"id": "canopy", "kind": "tree-canopy",
                            "polygon_xy_m": [[.7, -3.], [4., -3.], [4., 3.], [.7, 3.]]}]
    airspace = TraversableAirspace({"traversability": layer}, [],
                                  radius_m=.2, half_height_m=.15)
    vehicle = VehicleAsset(asset_id="test", name="Test", dry_mass_kg=2.,
        max_takeoff_mass_kg=3., body_radius_m=.2, body_height_m=.3,
        max_speed_mps=2., max_acceleration_mps2=4., qualified_range_m=100.,
        reserve_battery_percent=30., max_pickup_payload_kg=.2, sensors=["depth-camera"])
    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=10000,
        source="simulation-ground-truth", stream_healthy=True, stream_age_seconds=.01,
        localization_covariance_m2=0., current_position_m=Vector3(x=0., y=0., z=8.),
        current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=6., y=0., z=8.), dynamic_obstacles=[])
    result = evaluate_runtime_local_safety(observation=observation, vehicle=vehicle,
        static_primitives=[], required_clearance_m=.15, generated_at_unix_ms=10010,
        operating_airspace=airspace)
    assert result.decision.action != "continue"
    if result.decision.action != "hold":
        path = [(0., 0., 8.), *[(p.x, p.y, p.z) for p in result.decision.predicted_path_m]]
        assert airspace.path_clearance(path)[0] >= .15
