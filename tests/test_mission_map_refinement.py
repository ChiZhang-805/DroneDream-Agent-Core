"""Real geometry/refinement integration on synthetic fixtures, never flight evidence."""

import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image
from test_traversable_airspace import _graph, _ground

from dronedream_agent_core.contracts import MapEdge, ModelCallRecord, VehicleAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.map_gap_refinement import MapImageSource, MapViewInterpretation
from dronedream_agent_core.mission_map_refinement import (
    MapObservationPending,
    prepare_ground_routes,
    resolve_execution_semantic,
    validate_ground_supplement,
)


@pytest.fixture
def inputs(tmp_path):
    layer = {"schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0., "maximum_agl_m": 12., "areas": [
            {"id": "start-pad", "kind": "parking", "ground_z_m": 0.,
             "polygon_xy_m": [[-2., -2.], [1., -2.], [1., 2.], [-2., 2.]]},
            {"id": "goal-pad", "kind": "parking", "ground_z_m": 0.,
             "polygon_xy_m": [[5., -2.], [8., -2.], [8., 2.], [5., 2.]]}],
        "corridors": [], "exclusions": [{"id": "tree", "kind": "tree-canopy",
            "polygon_xy_m": [[2., -1.], [4., -1.], [4., 1.], [2., 1.]]}]}
    semantic = {"coordinate_frame": "ENU", "collision_primitives": [_ground()],
                "traversability": layer}
    path = tmp_path / "base-semantic.json"
    path.write_text(json.dumps(semantic), encoding="utf-8")
    image = Image.new("RGB", (64, 64), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    source = MapImageSource("traversable-map", hashlib.sha256(path.read_bytes()).hexdigest(),
                            raw, hashlib.sha256(raw).hexdigest(),
                            (1, 0, -16, 0, 1, -16), "synthetic-test-image")
    vehicle = VehicleAsset(asset_id="test", name="Test", dry_mass_kg=2.,
        max_takeoff_mass_kg=3., body_radius_m=.2, body_height_m=.3,
        max_speed_mps=2., max_acceleration_mps2=4., qualified_range_m=100.,
        reserve_battery_percent=30., max_pickup_payload_kg=.2, sensors=["depth-camera"])
    return dict(graph=_graph(), vehicle=vehicle, semantic_path=path,
        source_loader=lambda: source, focus_nodes=["start", "goal", "start"],
        output_dir=tmp_path / "plan-0001", maximum_model_calls=4,
        maximum_search_seconds=2.)


class FixtureVision:
    supports_image_input = True

    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        context = kwargs["input_artifact"]
        w, h = context["image_size_px"]
        output = MapViewInterpretation.model_validate({
            "view_id": context["view_id"], "image_sha256": context["image_sha256"],
            "areas": [{"kind": "road", "confidence": .95,
                "observation": "Synthetic test-only visible pavement",
                "polygon_px": [{"x": 2, "y": 2}, {"x": w-2, "y": 2},
                               {"x": w-2, "y": h-2}, {"x": 2, "y": h-2}]}],
            "corridors": [], "unresolved_observations": []})
        record = ModelCallRecord(call_id=f"model-{uuid4().hex[:24]}", role="context_summarizer",
            attempt=1, input_sha256=sha256_json(context), output_sha256=sha256_json(output),
            output_schema="MapViewInterpretation", provider="test-only", model="test-only",
            latency_ms=0, created_at=datetime.now(UTC))
        return SimpleNamespace(artifact=output, record=record)


def _prepared(result):
    return SimpleNamespace(
        contract=SimpleNamespace(map_semantic_sha256=result.receipt["derived_semantic_sha256"]),
        evidence=[SimpleNamespace(event_type="mission.map-refinement", payload=result.receipt)],
        model_calls=[ModelCallRecord.model_validate(call) for call in result.model_calls])


def _resolve(inputs, prepared):
    return resolve_execution_semantic(prepared=prepared, output_dir=inputs["output_dir"],
        base_semantic_path=inputs["semantic_path"], graph=inputs["graph"],
        source_loader=inputs["source_loader"])


def test_actual_replanning_routes_around_canopy_and_executor_receives_same_snapshot(inputs):
    raw = inputs["semantic_path"].read_bytes()
    port = FixtureVision()
    result = prepare_ground_routes(**inputs, model_ports=[port])
    assert result.semantic_path != inputs["semantic_path"]
    assert inputs["semantic_path"].read_bytes() == raw
    assert result.receipt["qualification"].endswith("not-flight-qualified")
    assert result.physical_calls == len(port.calls) == 1
    assert len(result.receipt["route_checks"]) == 2
    for check in result.receipt["route_checks"]:
        assert check["clearance"]["accepted"]
        assert check["clearance"]["semantic_sha256"] == result.receipt["derived_semantic_sha256"]
        assert any(abs(point["y"]) > 1.2 for point in check["route"]["positions_m"])
    assert _resolve(inputs, _prepared(result)) == result.semantic_path


def test_connected_round_trip_refines_target_before_freezing_execution(inputs):
    # The old graph is fully connected, but its edge crosses the known tree.
    connected = inputs["graph"].model_copy(update={"edges": [MapEdge(
        edge_id="old-over-canopy", from_node="start", to_node="goal", distance_m=6,
        minimum_clearance_m=2, speed_limit_mps=2, bidirectional=True)]})
    configured = {**inputs, "graph": connected}
    vision = FixtureVision()
    result = prepare_ground_routes(**configured, model_ports=[vision])
    assert vision.calls[0]["input_artifact"]["focus_node_ids"] == ["goal"]
    assert result.physical_calls == 1
    assert len(result.receipt["route_checks"]) == 2
    for check in result.receipt["route_checks"]:
        assert check["clearance"]["accepted"]
        assert any(abs(point["y"]) > 1.2 for point in check["route"]["positions_m"])
        assert all(point["z"] < 12 for point in check["route"]["positions_m"])
    assert _resolve(configured, _prepared(result)) == result.semantic_path


def test_text_only_provider_retains_pending_observations_without_a_fake_plan(inputs):
    class TextPort:
        supports_image_input = False

        def call(self, **kwargs):
            pytest.fail("Text-only provider must not be sent imaginary pixels")

    with pytest.raises(MapObservationPending) as caught:
        prepare_ground_routes(**inputs, model_ports=[TextPort()])
    assert caught.value.details["state"] == "awaiting-observation"
    assert caught.value.details["physical_calls"] == 0
    assert len(caught.value.details["pending_requests"]) == 4
    assert (inputs["output_dir"] / "map-refinement/pending.json").is_file()
    assert not (inputs["output_dir"] / "prepared-mission.json").exists()


@pytest.mark.parametrize("tamper", ["snapshot", "path", "base", "image", "model", "pixels"])
def test_execution_rejects_tampered_refinement(inputs, tamper):
    result = prepare_ground_routes(**inputs, model_ports=[FixtureVision()])
    prepared = _prepared(result)
    if tamper == "snapshot":
        result.semantic_path.write_bytes(result.semantic_path.read_bytes() + b" ")
    elif tamper == "path":
        result.receipt["semantic_relative_path"] = "../base-semantic.json"
    elif tamper == "base":
        result.receipt["base_semantic_sha256"] = "b" * 64
    elif tamper == "image":
        result.receipt["source_image_sha256"] = "b" * 64
    elif tamper == "model":
        prepared.model_calls = []
    else:
        # Even a coherent forged new file/digest cannot detach geometry from the
        # actual model output and its supplied image registration.
        candidate = json.loads(result.semantic_path.read_bytes())
        patch = result.receipt["patches"][0]
        patch["areas"][0]["polygon_xy_m"][0][0] += 1
        changed_id = patch["areas"][0]["id"]
        for area in candidate["traversability"]["areas"]:
            if area["id"] == changed_id:
                area["polygon_xy_m"][0][0] += 1
        raw = json.dumps(candidate).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = result.semantic_path.parent / f"semantic-{digest}.json"
        path.write_bytes(raw)
        prepared.contract.map_semantic_sha256 = digest
        result.receipt["derived_semantic_sha256"] = digest
        result.receipt["semantic_relative_path"] = path.relative_to(inputs["output_dir"]).as_posix()
    with pytest.raises(ValueError, match="MAP_REFINEMENT_"):
        _resolve(inputs, prepared)


def test_supplement_cannot_rewrite_collision_world_or_existing_exclusions(inputs):
    base = json.loads(inputs["semantic_path"].read_bytes())
    candidate = deepcopy(base)
    candidate["collision_primitives"] = []
    with pytest.raises(ValueError, match="UNRELATED_MUTATION"):
        validate_ground_supplement(base, candidate, [])
    candidate = deepcopy(base)
    candidate["traversability"]["exclusions"] = []
    with pytest.raises(ValueError, match="UNRELATED_MUTATION"):
        validate_ground_supplement(base, candidate, [])


def test_pending_attempt_can_resume_with_newly_available_vision_provider(inputs):
    with pytest.raises(MapObservationPending):
        prepare_ground_routes(**inputs, model_ports=[])
    resumed = {**inputs, "output_dir": inputs["output_dir"].parent / "plan-0002"}
    result = prepare_ground_routes(**resumed, model_ports=[FixtureVision()])
    assert _resolve(resumed, _prepared(result)) == result.semantic_path


def test_continuation_has_fresh_call_budget_without_recharging_history(inputs):
    class UncertainVision(FixtureVision):
        def call(self, **kwargs):
            response = super().call(**kwargs)
            response.artifact = response.artifact.model_copy(update={"areas": []})
            response.record = response.record.model_copy(update={
                "output_sha256": sha256_json(response.artifact)})
            return response

    limited = {**inputs, "maximum_model_calls": 1}
    with pytest.raises(MapObservationPending) as pending:
        prepare_ground_routes(**limited, model_ports=[UncertainVision()])
    assert pending.value.details["physical_calls"] == 1
    resumed = {**limited, "output_dir": inputs["output_dir"].parent / "plan-0002"}
    vision = FixtureVision()
    result = prepare_ground_routes(**resumed, model_ports=[vision])
    assert len(vision.calls) == result.physical_calls == 1
    assert result.receipt["physical_calls"] == 2
    assert len(result.model_calls) == 2
    assert _resolve(resumed, _prepared(result)) == result.semantic_path


def test_old_alternate_only_pending_resumes_original_image_and_binds_executor(inputs):
    class UncertainVision(FixtureVision):
        def call(self, **kwargs):
            response = super().call(**kwargs)
            response.artifact = response.artifact.model_copy(update={"areas": []})
            response.record = response.record.model_copy(update={
                "output_sha256": sha256_json(response.artifact)})
            return response

    with pytest.raises(MapObservationPending) as caught:
        prepare_ground_routes(**inputs, model_ports=[UncertainVision()])
    details = caught.value.details
    attempted = {record["view_id"] for record in details["observation_records"]
                 if "view_id" in record}
    # A persisted receipt from the previous implementation had no primary views left.
    pending_path = inputs["output_dir"] / "map-refinement/pending.json"
    pending = json.loads(pending_path.read_bytes())
    pending["pending_requests"] = [{"reason": "inspect-remaining-occlusion",
        "focus_node_ids": inputs["focus_nodes"], "bounds_xy_m": [-30, -30, 36, 30],
        "scale": "alternate"}]
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    resumed = {**inputs, "output_dir": inputs["output_dir"].parent / "plan-0002",
               "maximum_model_calls": 1}
    vision = FixtureVision()
    result = prepare_ground_routes(**resumed, model_ports=[vision])
    context = vision.calls[0]["input_artifact"]
    assert context["view_id"] not in attempted
    assert context["observation_scale"] != "alternate"
    assert context["observation_reason"] == "inspect-unseen-registered-image-tile"
    assert context["source_image_sha256"] == inputs["source_loader"]().image_sha256
    assert result.physical_calls == 1
    assert result.receipt["physical_calls"] == details["physical_calls"] + 1
    assert len(result.model_calls) == result.receipt["physical_calls"]
    assert _resolve(resumed, _prepared(result)) == result.semantic_path
    candidate = json.loads(result.semantic_path.read_bytes())
    original = json.loads(inputs["semantic_path"].read_bytes())
    assert candidate["traversability"]["exclusions"] == original["traversability"]["exclusions"]
    assert candidate["traversability"]["maximum_agl_m"] == 12


@pytest.mark.usefixtures("isolated_server_credentials")
def test_pending_observation_api_retains_task_without_plan_or_failure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore

    identity = VerifiedIdentity(owner_account_id="observation-owner", tenant_id=None,
        organization_id=None, issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000)
    store = AppStore(tmp_path / "app")
    token = "e" * 64
    client = TestClient(create_app(store=store, token=token,
        account_memory_path=tmp_path / "memory.sqlite3", identity_verifier=lambda _: identity))
    headers = {"Authorization": f"Bearer {token}", "X-DroneDream-Identity-Token": "signed"}
    thread = store.create_thread("Inspect ground", "gpt-5.4", "en-US")

    def pending(_self, **_kwargs):
        raise MapObservationPending({"state": "awaiting-observation",
            "next_action": "inspect-map-gaps", "pending_requests": [],
            "candidate_semantic": {"private": "geometry"},
            "pending_artifact": "private-path", "actuator_authority": False})

    monkeypatch.setattr("dronedream_agent_app.mission_service.MissionService.prepare", pending)
    response = client.post(f"/v1/threads/{thread['thread_id']}/prepare", headers=headers, json={
        "expected_owner_account_id": identity.owner_account_id, "source_edition": "autonomy",
        "message": "Collect the chemistry sample", "map_id": "map", "vehicle_id": "vehicle",
        "model_id": "gpt-5.4", "model_grant": "ddg_abcdefghijklmnopqrstuvwxyz",
        "locale": "en-US",
        "gateway_base_url": "https://example.supabase.co/functions/v1/model-gateway"})
    assert response.status_code == 202
    assert response.json()["status"] == "map_refinement_pending"
    assert response.json()["actuator_authority"] is False
    assert "candidate_semantic" not in response.json()
    assert "pending_artifact" not in response.json()
    assert store.get_thread(thread["thread_id"])["state"] == "planning"


def test_progress_mailbox_distinguishes_pending_observation_from_failure():
    from dronedream_agent_app.preparation_progress import PreparationProgress

    progress = PreparationProgress()
    key = ("owner", None, None, "thread", "request")

    def pending():
        raise MapObservationPending({"state": "awaiting-observation"})

    with pytest.raises(MapObservationPending):
        progress.run(key, pending)
    snapshot = progress.read(key, 0)
    assert snapshot["state"] == "awaiting_observation"
    assert snapshot["events"][-1]["stage"] == "map_refinement_pending"
    assert all(event["stage"] != "failed" for event in snapshot["events"])


def test_full_preparation_freezes_refined_hash_and_rebinds_real_tools(tmp_path):
    from test_map_semantic_boundaries import _graph as pipeline_graph
    from test_map_semantic_boundaries import _semantic
    from test_planning_energy_gates import _vehicle
    from test_preparation_pipeline_integrity import _PlanningPort

    from dronedream_agent_core.assets import load_map_catalog
    from dronedream_agent_core.cli import _bind_development_harness_request
    from dronedream_agent_core.context import ContextStore
    from dronedream_agent_core.contracts import MissionRequest, Vector3
    from dronedream_agent_core.orchestrator import MissionOrchestrator, PreparationConfig
    from dronedream_agent_core.plugin_api import ToolEnvironment, build_snapshot_tool_registry

    graph, vehicle = pipeline_graph(), _vehicle()
    vehicle.collision_center_offset_model_m = Vector3(x=0, y=0, z=.1)
    semantic = _semantic()
    semantic["runtime_bindings"] = {"schema_version": "dronedream.map-runtime-bindings.v1",
        "simulator": "gazebo-harmonic", "coordinate_frame": "ENU",
        "vehicle_spawn": {"x": 0, "y": 0, "z": 0},
        "mission_launch_waypoint": {"x": 0, "y": 0, "z": 2}}
    semantic["collision_primitives"] = [_ground()]
    semantic["traversability"] = {
        "schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0, "maximum_agl_m": 12, "corridors": [], "exclusions": [],
        "areas": [{"id": "launch-only", "kind": "parking", "ground_z_m": 0,
            "polygon_xy_m": [[-2, -2], [1, -2], [1, 2], [-2, 2]]}]}
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic), encoding="utf-8")
    original = path.read_bytes()
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(image, format="PNG")
    raw = image.getvalue()
    source = MapImageSource(graph.asset_id, hashlib.sha256(original).hexdigest(), raw,
        hashlib.sha256(raw).hexdigest(), (1, 0, -16, 0, 1, -16), "pipeline-fixture")
    sdf = tmp_path / "vehicle.sdf"
    sdf.write_text('<sdf version="1.9"><model name="fixture"/></sdf>')
    port, vision = _PlanningPort(), FixtureVision()
    context = ContextStore(tmp_path / "context.sqlite")
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
        map_refinement=lambda focus, directory, budget: prepare_ground_routes(
            graph=graph, vehicle=vehicle, semantic_path=path, source_loader=lambda: source,
            model_ports=[vision], focus_nodes=focus, output_dir=directory,
            maximum_model_calls=budget, maximum_search_seconds=2),
        tool_registry_factory=tools_for)
    output = tmp_path / "plan-0001"
    try:
        request = _bind_development_harness_request(MissionRequest(conversation_id="refined-test",
            message="Visit target and return without a payload", start_entity="start"),
            config=config, plugin_snapshot=instance.plugin_snapshot)
        prepared = instance.prepare(request, output)
        assert path.read_bytes() == original
        assert rebound and rebound[0] != path
        assert prepared.contract.map_semantic_sha256 != source.semantic_sha256
        assert prepared.contract.map_semantic_sha256 == prepared.route_clearance.semantic_sha256
        assert prepared.route_clearance.accepted
        assert resolve_execution_semantic(prepared=prepared, output_dir=output,
            base_semantic_path=path, graph=graph, source_loader=lambda: source) == rebound[0]
    finally:
        instance.close()
        context.close()
