"""Exercise the actual planner -> qualification -> PX4 schedule conversion.

These are geometric/contract checks, not native flight qualification receipts.
"""

import json

import pytest
from test_px4_executor_integrity import executor_module
from test_traversable_airspace import _graph, _layer, _write

from dronedream_agent_core.contracts import RouteQuery, VehicleAsset
from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy
from dronedream_agent_core.orchestrator import (
    MissionPreparationBlocked,
    _validate_plugin_track_tightening,
)
from dronedream_agent_core.px4_track import route_to_px4_track
from scripts.run_school_map_depth_qualification import _qualification_route


def _planned_ground_route(tmp_path):
    layer = _layer()
    layer["areas"] = []
    layer["maximum_agl_m"] = 3.
    layer["corridors"] = [{"id": "between-buildings", "kind": "path", "width_m": 2.,
                           "ground_z_m": 0.,
                           "centerline_xy_m": [[0., 0.], [0., 8.], [6., 8.], [6., 0.]]}]
    path = _write(tmp_path, layer)
    semantic = json.loads(path.read_text())
    semantic["schema_version"] = "dronedream.map-semantic.v1"
    semantic.pop("runtime_collision_primitives")
    semantic["runtime_bindings"] = {
        "schema_version": "dronedream.map-runtime-bindings.v1",
        "simulator": "gazebo-harmonic", "coordinate_frame": "ENU",
        "vehicle_spawn": {"x": 0., "y": 0., "z": .15},
        "mission_launch_waypoint": {"x": 0., "y": 0., "z": 1.},
    }
    path.write_text(json.dumps(semantic))
    vehicle = VehicleAsset(asset_id="ground-test-drone", name="Ground test drone",
        dry_mass_kg=1., max_takeoff_mass_kg=2., body_radius_m=.2, body_height_m=.3,
        collision_center_offset_model_m={"x": 0., "y": 0., "z": .15},
        max_speed_mps=2., max_acceleration_mps2=1., qualified_range_m=1000.,
        reserve_battery_percent=20., max_pickup_payload_kg=.1, sensors=["camera"])
    graph = _graph()
    graph.nodes[1].position_m.z = 2.5
    planner = KnownMapMetricPlanner(graph=graph, semantic_path=path,
        vehicle_diameter_m=.4, vehicle_height_m=.3,
        policy=MetricPlannerPolicy(resolution_m=.5, required_clearance_m=.15))
    route = planner.plan(RouteQuery(start_node="start", goal_node="goal"))
    return route, graph, path, vehicle, planner


@pytest.mark.parametrize("qualification", [False, True])
def test_low_ground_route_survives_complete_px4_setpoint_chain(tmp_path, qualification):
    route, graph, semantic, vehicle, planner = _planned_ground_route(tmp_path)
    original = list(route.positions_m)
    if qualification:
        route, graph = _qualification_route(route, point_count=None, speed_limit_mps=1.)
        assert route.positions_m == [*original, *reversed(original[:-1])]
    track = route_to_px4_track(route, graph, semantic, vehicle=vehicle)
    _validate_plugin_track_tightening(track, route, vehicle)
    track_path = tmp_path / "track.json"
    track_path.write_text(track.model_dump_json())
    executor = executor_module()
    loaded = executor.load_reference_track_plan(track_path)
    schedule = executor.build_setpoint_schedule_plan(
        loaded.points, executor.ControllerParams(vel_limit=1., accel_limit=.5), 20.,
        stop_at_waypoints=loaded.stop_at_waypoints,
        waypoint_hold_seconds=loaded.waypoint_hold_seconds)
    root = track.coordinate_contract.model_root_world_enu_m
    offset = track.coordinate_contract.collision_center_offset_model_m

    def world(point):
        return (point.east_m + root[0] + offset[0],
                point.north_m + root[1] + offset[1],
                -point.down_m + root[2] + offset[2])

    # Every stop survives; neither an endpoint chord nor a fixed-height route
    # can pass this check. Only takeoff's initial vertical ingress is excluded.
    assert len(schedule.waypoint_arrival_indices) == len(route.positions_m) - 1
    for index, expected in zip(schedule.waypoint_arrival_indices,
                               route.positions_m[1:], strict=True):
        assert world(schedule.schedule[index]) == pytest.approx(
            (expected.x, expected.y, expected.z))
    actual = [world(point) for point in schedule.schedule[schedule.track_start_index:]]
    assert max(point[1] for point in actual) > 7.
    assert max(point[2] for point in actual) <= 2.5 + 1e-9
    assert min(planner.clearance(point) for point in actual) >= .15


@pytest.mark.parametrize("mutation", ["fixed-high-altitude", "endpoint-shortcut"])
def test_plugin_cannot_replace_low_ground_route_before_execution(tmp_path, mutation):
    route, graph, semantic, vehicle, _ = _planned_ground_route(tmp_path)
    track = route_to_px4_track(route, graph, semantic, vehicle=vehicle)
    if mutation == "fixed-high-altitude":
        for point in track.points:
            point.z = 35.
        expected = "PLUGIN_TRACK_SAFETY_ENVELOPE_RELAXED"
    else:
        track = track.model_copy(update={
            "points": [track.points[0], track.points[-1]],
            "source_world_points": [track.source_world_points[0], track.source_world_points[-1]],
        })
        expected = "PLUGIN_TRACK_GEOMETRY_CHANGED"
    with pytest.raises(MissionPreparationBlocked, match=expected):
        _validate_plugin_track_tightening(track, route, vehicle)
