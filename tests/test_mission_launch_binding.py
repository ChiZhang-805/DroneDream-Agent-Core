"""Real launch/track/prepare contracts on offline fixtures, never flight evidence."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from test_map_semantic_boundaries import _graph, _semantic
from test_mission_map_refinement import FixtureVision
from test_planning_energy_gates import _vehicle
from test_preparation_pipeline_integrity import _PlanningPort
from test_traversable_airspace import _ground, _layer

from dronedream_agent_core.assets import load_map_catalog
from dronedream_agent_core.cli import _bind_development_harness_request
from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.contracts import (
    GraphRoute,
    IntentArtifact,
    MapNode,
    MissionRequest,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.map_gap_refinement import MapImageSource
from dronedream_agent_core.mission_launch_binding import (
    default_launch_node,
    prepare_mission_launch,
    resolve_mission_execution_semantic,
)
from dronedream_agent_core.mission_map_refinement import prepare_ground_routes
from dronedream_agent_core.orchestrator import MissionOrchestrator, PreparationConfig
from dronedream_agent_core.plugin_api import ToolEnvironment, build_snapshot_tool_registry
from dronedream_agent_core.px4_track import route_to_px4_track
from dronedream_agent_core.runtime_bindings import resolve_grounded_vehicle_model_root


@pytest.fixture
def inputs(tmp_path):
    graph, semantic, vehicle = _graph(), _semantic(), _vehicle()
    vehicle.collision_center_offset_model_m = Vector3(x=.05, y=0, z=.1)
    graph.nodes.append(MapNode(node_id="default", label="Default", semantic="launch",
                              position_m=Vector3(x=-5, y=0, z=2)))
    graph.named_entities["default-launch"] = "default"
    semantic["runtime_bindings"] = {
        "schema_version": "dronedream.map-runtime-bindings.v1",
        "simulator": "gazebo-harmonic", "coordinate_frame": "ENU",
        "vehicle_spawn": {"x": -5, "y": 0, "z": 0},
        "mission_launch_waypoint": {"x": -5, "y": 0, "z": 2}}
    semantic["collision_primitives"] = [_ground()]
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic), encoding="utf-8")
    intent = IntentArtifact(goal="From start visit target and return", start_entity="start",
                            target_entity="target", return_entity="start", payload_action="none")
    return dict(graph=graph, semantic_path=path, vehicle=vehicle, start_node="a",
                intent_sha256=sha256_json(intent), output_dir=tmp_path / "plan"), intent


def _prepared(inputs, intent):
    path, receipt = prepare_mission_launch(**inputs)
    prepared = SimpleNamespace(intent=intent,
        contract=SimpleNamespace(start_node="a", map_semantic_sha256=hashlib.sha256(
            path.read_bytes()).hexdigest()), model_calls=[],
        evidence=[SimpleNamespace(event_type="mission.launch-binding", payload=receipt)])
    return path, prepared


def _resolve(inputs, prepared):
    return resolve_mission_execution_semantic(prepared=prepared,
        output_dir=inputs["output_dir"], base_semantic_path=inputs["semantic_path"],
        graph=inputs["graph"], vehicle=inputs["vehicle"],
        source_loader=lambda: pytest.fail("No visual refinement was performed"))


def test_nondefault_start_binds_grounded_spawn_and_local_track_not_default_position(inputs):
    args, intent = inputs
    before = args["semantic_path"].read_bytes()
    path, prepared = _prepared(args, intent)
    assert args["semantic_path"].read_bytes() == before
    assert _resolve(args, prepared) == path
    base, derived = json.loads(before), json.loads(path.read_bytes())
    assert {k: v for k, v in derived.items() if k != "runtime_bindings"} == {
        k: v for k, v in base.items() if k != "runtime_bindings"}
    root = resolve_grounded_vehicle_model_root(derived, args["vehicle"])
    assert root.x == -.05  # Horizontal collision-center offset is preserved too.
    assert root.y == 0
    assert root.z == pytest.approx(args["vehicle"].body_height_m / 2 - .1)
    assert derived["runtime_bindings"]["mission_launch_waypoint"]["z"] == 2
    assert derived["runtime_bindings"]["vehicle_spawn"]["z"] == 0
    graph = args["graph"]
    route = GraphRoute(start_node="a", goal_node="a", node_ids=["a", "b", "a"],
        edge_ids=["ab", "ab"], positions_m=[graph.nodes[i].position_m for i in (0, 1, 0)],
        route_length_m=6, all_edges_flight_verified=False)
    with pytest.raises(ValueError, match="qualified PX4 launch"):
        route_to_px4_track(route, graph, args["semantic_path"], vehicle=args["vehicle"])
    track = route_to_px4_track(route, graph, path, vehicle=args["vehicle"])
    assert track.points[0].x == track.points[0].y == 0
    assert track.points[-1].x == track.points[-1].y == 0
    assert track.source_world_points[0].up_m == 2
    assert track.coordinate_contract.model_root_world_enu_m == [root.x, root.y, root.z]


def test_omitted_start_uses_runtime_bound_default_without_deriving_or_sorting_names(inputs):
    args, _ = inputs
    assert default_launch_node(args["graph"], args["semantic_path"]) == "default"
    path, receipt = prepare_mission_launch(**{**args, "start_node": "default"})
    assert path == args["semantic_path"] and receipt is None
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("problem", ["no-support", "wall", "tree", "outside", "below-body"])
def test_named_start_must_have_real_support_and_a_clear_whole_vertical_climb(inputs, problem):
    args, _ = inputs
    semantic = json.loads(args["semantic_path"].read_bytes())
    if problem == "no-support":
        semantic["collision_primitives"][0]["center_x"] = 100
    elif problem == "wall":
        # Both center endpoints are clear, but the middle of takeoff intersects.
        semantic["collision_primitives"].append(dict(name="overhang", semantic="obstacle",
            center_x=0, center_y=0, center_z=1.1, size_x=2, size_y=2, size_z=.1))
    elif problem in {"tree", "outside"}:
        semantic["traversability"] = _layer()
        if problem == "tree":
            semantic["traversability"]["exclusions"] = [dict(id="canopy", kind="tree-canopy",
                polygon_xy_m=[[-1, -1], [1, -1], [1, 1], [-1, 1]])]
        else:
            semantic["traversability"]["areas"][0]["polygon_xy_m"] = [
                [10, 10], [12, 10], [12, 12], [10, 12]]
    else:
        args["graph"].nodes[0].position_m.z = .01
    args["semantic_path"].write_text(json.dumps(semantic), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_mission_launch(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("mutation", ["intent", "node", "vehicle", "receipt", "snapshot"])
def test_executor_reconstructs_exact_binding_and_rejects_drift(inputs, mutation):
    args, intent = inputs
    path, prepared = _prepared(args, intent)
    if mutation == "intent":
        prepared.intent.goal = "Different accepted intent"
    elif mutation == "node":
        prepared.contract.start_node = "default"
    elif mutation == "vehicle":
        args["vehicle"].body_height_m += .02
    elif mutation == "receipt":
        prepared.evidence[0].payload["semantic_relative_path"] = "../semantic.json"
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="MISSION_LAUNCH"):
        _resolve(args, prepared)


def test_runtime_support_does_not_replace_original_launch_pad(inputs):
    args, _ = inputs
    semantic = json.loads(args["semantic_path"].read_bytes())
    semantic["collision_primitives"] = [dict(name="launch", semantic="launch-pad",
        center_x=-5, center_y=0, center_z=.04, size_x=2, size_y=2, size_z=.08)]
    semantic["runtime_bindings"]["vehicle_spawn"]["z"] = .067
    original = resolve_grounded_vehicle_model_root(semantic, args["vehicle"])
    semantic["runtime_collision_primitives"] = [_ground()]
    assert resolve_grounded_vehicle_model_root(semantic, args["vehicle"]) == original


@pytest.mark.parametrize("with_refinement", [False, True])
def test_complete_preparation_rebinds_real_tools_then_executor_resolves_same_bytes(
    inputs, with_refinement,
):
    args, _ = inputs
    graph, vehicle, path, output = (args[k] for k in (
        "graph", "vehicle", "semantic_path", "output_dir"))
    vehicle.collision_center_offset_model_m.x = 0
    semantic = json.loads(path.read_bytes())
    if with_refinement:
        semantic["traversability"] = _layer()
        semantic["traversability"]["areas"][0]["polygon_xy_m"] = [
            [-2, -2], [1, -2], [1, 2], [-2, 2]]
        path.write_text(json.dumps(semantic), encoding="utf-8")
    original = path.read_bytes()
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(image, format="PNG")
    raw = image.getvalue()
    source = MapImageSource(graph.asset_id, hashlib.sha256(original).hexdigest(), raw,
        hashlib.sha256(raw).hexdigest(), (1, 0, -16, 0, 1, -16), "offline-pipeline-fixture")
    sdf = path.parent / "vehicle.sdf"
    sdf.write_text('<sdf version="1.9"><model name="fixture"/></sdf>')
    port, vision = _PlanningPort(), FixtureVision()
    context = ContextStore(path.parent / "context.sqlite")
    config = PreparationConfig(provider="kimi", critic_provider="kimi",
                               persisted_task_context=False)
    rebound = []

    def tools_for(semantic_path):
        rebound.append(semantic_path)
        return build_snapshot_tool_registry(ToolEnvironment(map_graph=graph,
            semantic_path=semantic_path, vehicle_diameter_m=vehicle.body_radius_m * 2,
            vehicle_height_m=vehicle.body_height_m, vehicle=vehicle,
            waypoint_hold_seconds=.4), instance.plugin_snapshot)

    instance = MissionOrchestrator(config=config,
        map_catalog=load_map_catalog(path, qualified_graph=graph), map_graph=graph,
        semantic_path=path, vehicle_sdf=sdf, vehicle_asset_id=vehicle.asset_id,
        vehicle=vehicle, context_store=context, primary_port=port, critic_port=port,
        map_refinement=(lambda focus, directory, budget: prepare_ground_routes(
            graph=graph, vehicle=vehicle, semantic_path=path, source_loader=lambda: source,
            model_ports=[vision], focus_nodes=focus, output_dir=directory,
            maximum_model_calls=budget, maximum_search_seconds=2)) if with_refinement else None,
        tool_registry_factory=tools_for)
    try:
        request = _bind_development_harness_request(MissionRequest(conversation_id="launch-test",
            message="From start, visit target and return without a payload",
            start_entity="default-launch"), config=config,
            plugin_snapshot=instance.plugin_snapshot)
        prepared = instance.prepare(request, output)
        assert path.read_bytes() == original
        assert prepared.contract.start_node == prepared.contract.return_node == "a"
        assert prepared.contract.map_sha256 == sha256_json(graph)
        assert prepared.contract.map_semantic_sha256 == prepared.route_clearance.semantic_sha256
        assert prepared.route_clearance.accepted
        assert prepared.px4_track.coordinate_contract.model_root_world_enu_m[:2] == [0, 0]
        assert prepared.px4_track.points[0].x == prepared.px4_track.points[0].y == 0
        actual = resolve_mission_execution_semantic(prepared=prepared, output_dir=output,
            base_semantic_path=path, graph=graph, vehicle=vehicle, source_loader=lambda: source)
        assert actual == rebound[-1]
        assert len(rebound) == (2 if with_refinement else 1)
        assert actual.parent.name == "mission-launch"
    finally:
        instance.close()
        context.close()
