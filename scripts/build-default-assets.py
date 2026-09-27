from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from dronedream_agent_app.asset_runtime_resolver import (
    ResolvedDevelopmentMissionInput,
    resolve_development_mission_input,
)
from dronedream_agent_core.asset_packages import (
    AssetFile,
    AssetIR,
    AssetSource,
    DDPkgManifest,
    GeometrySummary,
    InterfaceSummary,
    PhysicsSummary,
    ReadinessSummary,
    RuntimeExtension,
    SemanticSummary,
    SensorSummary,
    SimulationTarget,
    VehicleSummary,
    inspect_ddpkg,
    package_content_sha256,
)
from dronedream_agent_core.asset_pair_qualification import (
    bind_qualification_receipt,
    build_pair_qualification_receipt,
    prepare_asset_pair_qualification,
)

MAP_ASSET_ID = "dronedream.school-map.v1"
VEHICLE_ASSET_ID = "dronedream.my-drone.v1"
ARCHIVE_TIMESTAMP = (2026, 8, 19, 0, 0, 0)
RUNTIME_QUALIFICATION_FIELDS = (
    "gazebo_runtime_verified",
    "px4_mission_smoke_verified",
    "simulation_execution_ready",
)
CURRENT_RUNTIME_GATES = (
    "controlled_vehicle_entity_selected",
    "controlled_vehicle_identity_confirmed",
    "development_fault_injection_absent",
    "development_payload_collection_absent",
    "executor_completed",
    "goal_observed",
    "landing_confirmed",
    "live_depth_metric_map_present",
    "live_depth_perception_healthy",
    "live_depth_safety_history_present",
    "local_policy_simulation_admission_recorded",
    "local_safety_command_present",
    "local_safety_observation_present",
    "local_safety_observation_sequence_advanced",
    "model_navigation_authorized_control_applied",
    "model_navigation_authorized_schedule_advance_recorded",
    "model_navigation_control_authority_required",
    "model_navigation_cycle_recorded",
    "model_navigation_fresh_revalidation_stable",
    "model_navigation_invocation_failure_absent",
    "model_navigation_primary_provider_call_recorded",
    "model_navigation_provider_cadence_bounded",
    "model_navigation_provider_call_recorded",
    "model_navigation_route_fallback_absent",
    "model_navigation_snapshot_recorded",
    "model_navigation_visual_frame_recorded",
    "native_terminal_lifecycle_published",
    "no_live_abort",
    "offboard_timing_complete",
    "px4_ulog_present",
    "ros_observations_present",
    "runtime_evidence_writer_complete",
    "runtime_pose_samples_present",
    "runtime_snapshot_writer_complete",
    "static_route_clearance_bound",
)
CURRENT_RUNTIME_ACTIONS = (
    "delivery.precontact-hold",
    "pickup",
    "delivery.confirm-custody",
    "delivery.verify-loaded-stability",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)


def _extract_current_map_package(package: Path, destination: Path) -> tuple[Path, Path]:
    """Extract only the validated current normalized map payload."""

    inspected = inspect_ddpkg(package)
    qualification = inspected.manifest.qualification
    if (
        inspected.manifest.asset_kind not in {"map", "world"}
        or inspected.manifest.asset_id != MAP_ASSET_ID
        or qualification is None
        or qualification.maturity != "qualified"
        or qualification.content_sha256 != inspected.manifest.content_sha256
    ):
        raise ValueError("DEFAULT_MAP_PACKAGE_NOT_CURRENT_QUALIFIED")
    prefix = "normalized/map/"
    selected = [entry for entry in inspected.manifest.files if entry.path.startswith(prefix)]
    if not selected:
        raise ValueError("DEFAULT_MAP_PACKAGE_PAYLOAD_MISSING")
    with zipfile.ZipFile(package) as bundle:
        for entry in selected:
            relative = entry.path.removeprefix(prefix)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(entry.path))
    gazebo = destination / "gazebo"
    graph = destination / "navigation-graph.json"
    _write_current_map_contract(gazebo / "semantic.json")
    if not graph.is_file():
        raise ValueError("DEFAULT_MAP_PACKAGE_GRAPH_MISSING")
    return gazebo, graph


def _stage_development_vehicle(
    source_manifest: Path,
    destination: Path,
) -> tuple[ResolvedDevelopmentMissionInput, Path]:
    """Stage exact validated development vehicle inputs for qualification promotion."""

    source = resolve_development_mission_input(source_manifest)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source.vehicle_sdf, destination / "model.sdf")
    shutil.copyfile(source.vehicle_summary, destination / "summary.json")
    shutil.copyfile(source.payload_sdf, destination / "takeout-payload.sdf")
    (destination / "model.config").write_text(
        """<?xml version="1.0"?>
<model>
  <name>My Drone</name>
  <version>1.0.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>DroneDream</name></author>
  <description>Qualified PX4 x500 depth vehicle with detachable payload.</description>
</model>
""",
        encoding="utf-8",
        newline="\n",
    )
    return source, source.controller_params


def _assert_promotion_input_binding(
    *,
    run_dir: Path,
    map_source: Path,
    vehicle_source: Path,
    controller_params: Path,
) -> None:
    """Require qualification evidence to hash-bind every promoted runtime input."""

    evidence = _json(run_dir / "mission_evidence.json")
    artifacts = evidence.get("artifacts")
    expected = {
        "world_sha256": _sha256(map_source / "world.sdf"),
        "semantic_sha256": _sha256(map_source / "semantic.json"),
        "vehicle_sha256": _sha256(vehicle_source / "model.sdf"),
        "controller_params_sha256": _sha256(controller_params),
    }
    if not isinstance(artifacts, dict) or any(
        artifacts.get(name) != digest for name, digest in expected.items()
    ):
        raise ValueError("DEFAULT_ASSET_PROMOTION_INPUT_MISMATCH")


def _write_zip_payload(bundle: zipfile.ZipFile, path: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(path, date_time=ARCHIVE_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, payload, compresslevel=9)


def _current_asset_role(*, kind: str, relative: str) -> str:
    lowered = relative.casefold()
    suffix = Path(relative).suffix.casefold()
    if kind == "map" and lowered == "gazebo/semantic.json":
        return "semantic"
    if kind == "vehicle" and lowered == "controller_params.json":
        return "controller"
    if suffix in {".sdf", ".world"}:
        return "sdf"
    if suffix in {".obj", ".dae", ".stl", ".gltf", ".glb", ".ppm", ".png"}:
        return "visual"
    return "metadata"


def _runtime_extensions(payloads: dict[str, bytes]) -> list[RuntimeExtension]:
    extensions: list[RuntimeExtension] = []
    for path, payload in payloads.items():
        if Path(path).suffix.casefold() not in {".sdf", ".world"}:
            continue
        root = ElementTree.fromstring(payload)
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "plugin":
                continue
            extensions.append(
                RuntimeExtension(
                    name=element.attrib.get("name") or "unnamed-plugin",
                    filename=element.attrib.get("filename"),
                    source_path=path,
                )
            )
    return extensions


def _write_current_source_package(
    *,
    source_root: Path,
    destination: Path,
    kind: str,
    asset_id: str,
    name: str,
    created_at: datetime,
) -> Path:
    """Build a native first-party DDPKG without embedding a retired ZIP tree."""

    if kind not in {"map", "vehicle"}:
        raise ValueError("DEFAULT_ASSET_KIND_INVALID")
    prefix = f"normalized/{kind}"
    payloads: dict[str, bytes] = {}
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root).as_posix()
        if relative == "manifest.json" or relative.startswith("qualification/"):
            continue
        payloads[f"{prefix}/{relative}"] = source.read_bytes()
    if not payloads:
        raise ValueError("DEFAULT_ASSET_SOURCE_EMPTY")

    files = [
        AssetFile(
            path=path,
            role=_current_asset_role(
                kind=kind,
                relative=path.removeprefix(f"{prefix}/"),
            ),
            media_type=(
                "application/json"
                if path.casefold().endswith(".json")
                else "application/xml"
                if Path(path).suffix.casefold() in {".sdf", ".world"}
                else "application/octet-stream"
            ),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        for path, payload in sorted(payloads.items())
    ]
    provenance_sha256 = package_content_sha256(files)
    target_path = f"{prefix}/gazebo/world.sdf" if kind == "map" else f"{prefix}/gazebo/model.sdf"
    if target_path not in payloads:
        raise ValueError("DEFAULT_ASSET_GAZEBO_TARGET_MISSING")

    if kind == "map":
        graph = json.loads(payloads[f"{prefix}/navigation-graph.json"])
        summary = json.loads(payloads[f"{prefix}/gazebo/summary.json"])
        vehicle = None
        sensors: list[SensorSummary] = []
        semantics = SemanticSummary(
            named_place_count=len(graph.get("named_entities", {})),
            navigation_graph_path=f"{prefix}/navigation-graph.json",
        )
        geometry = GeometrySummary(
            visual_paths=[entry.path for entry in files if entry.role == "visual"],
            coordinate_frames=["map_enu"],
            up_axis="z",
            visual_count=int(summary.get("visual_primitive_count", 0)),
            collision_count=int(summary.get("collision_primitive_count", 0)),
        )
        physics = PhysicsSummary(collision_complete=True)
        capabilities = ["gazebo", "semantic-map", "content-addressed"]
    else:
        metadata_path = f"{prefix}/vehicle.json"
        vehicle_metadata = json.loads(payloads[metadata_path])
        summary = json.loads(payloads[f"{prefix}/gazebo/summary.json"])
        vehicle = VehicleSummary(
            vehicle_class="multirotor",
            actuator_count=4,
            rotor_count=4,
            autopilot_profile="x500",
            payload_limit_kg=float(vehicle_metadata["max_pickup_payload_kg"]),
        )
        sensors = [
            SensorSummary(kind=str(sensor), source_path=metadata_path)
            for sensor in vehicle_metadata.get("sensors", [])
        ]
        semantics = SemanticSummary()
        geometry = GeometrySummary(
            coordinate_frames=["base_link_frd"],
            up_axis="z",
            visual_count=1,
            collision_count=1,
        )
        physics = PhysicsSummary(
            link_count=1,
            mass_entry_count=1,
            inertia_entry_count=1,
            collision_complete=True,
            mass_complete=True,
            inertia_complete=True,
        )
        if summary.get("source_model") != "model://x500_depth":
            raise ValueError("DEFAULT_VEHICLE_PX4_PROFILE_INVALID")
        capabilities = [
            "gazebo",
            "content-addressed",
            "px4.sitl-model=x500",
            "perception.forward-depth",
        ]

    asset_ir = AssetIR(
        asset_id=asset_id,
        name=name,
        version="1.0.0",
        kind=kind,
        source=AssetSource(
            adapter_id=f"dronedream.first-party-{kind}",
            adapter_version="1.0.0",
            application="DroneDream",
            application_version="1.0.0",
            source_format="dronedream-native",
            source_sha256=provenance_sha256,
        ),
        coordinate_frame="map_enu" if kind == "map" else "base_link_frd",
        files=files,
        simulation_targets=[
            SimulationTarget(
                target_id="gazebo-harmonic",
                simulator="gazebo-harmonic",
                simulator_version="runtime-detected",
                autopilot="none" if kind == "map" else "px4",
                entrypoint=target_path,
            )
        ],
        runtime_extensions=_runtime_extensions(payloads),
        geometry=geometry,
        physics=physics,
        vehicle=vehicle,
        sensors=sensors,
        semantics=semantics,
        interfaces=InterfaceSummary(
            gazebo_entrypoints=[target_path],
            px4_profiles=[] if kind == "map" else ["x500"],
        ),
        readiness=ReadinessSummary(
            maturity_ceiling="physics_ready",
            qualification_required=True,
            runtime_validation_performed=False,
        ),
        semantic_layers=[] if kind == "vehicle" else ["navigation", "collision"],
        capabilities=capabilities,
        license_expression="NOASSERTION",
    )
    ir_payload = (
        json.dumps(
            asset_ir.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    ir_file = AssetFile(
        path="normalized/asset-ir.json",
        role="asset_ir",
        media_type="application/json",
        sha256=hashlib.sha256(ir_payload).hexdigest(),
        size_bytes=len(ir_payload),
    )
    manifest = DDPkgManifest(
        package_id=f"{asset_id}.1.0.0",
        asset_id=asset_id,
        asset_kind=kind,
        content_sha256=package_content_sha256([*files, ir_file]),
        files=[*files, ir_file],
        created_at=created_at,
    )
    manifest_payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, payload in sorted(payloads.items()):
            _write_zip_payload(bundle, path, payload)
        _write_zip_payload(bundle, ir_file.path, ir_payload)
        _write_zip_payload(bundle, "manifest.json", manifest_payload)
    inspect_ddpkg(destination)
    return destination


def _current_runtime_action_receipts(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    receipt_dir = run_dir / "runtime-actions" / "receipts"
    receipt_paths = sorted(receipt_dir.glob("*.receipt.json"))
    receipts = [(path, _json(path)) for path in receipt_paths]
    by_action: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, receipt in receipts:
        action = receipt.get("action")
        if not isinstance(action, str) or action in by_action:
            raise ValueError("DEFAULT_ASSET_RUNTIME_ACTION_RECEIPTS_INVALID")
        by_action[action] = (path, receipt)
    if any(action not in by_action for action in CURRENT_RUNTIME_ACTIONS):
        raise ValueError("DEFAULT_ASSET_RUNTIME_ACTION_RECEIPTS_INCOMPLETE")
    for action in CURRENT_RUNTIME_ACTIONS:
        _path, receipt = by_action[action]
        gates = receipt.get("deterministic_gates")
        output = receipt.get("output")
        if (
            receipt.get("status") != "accepted"
            or not isinstance(receipt.get("attempts"), int)
            or isinstance(receipt.get("attempts"), bool)
            or receipt["attempts"] < 1
            or receipt.get("issue_codes") != []
            or not isinstance(gates, dict)
            or any(value is not True for value in gates.values())
            or not gates
            or not isinstance(output, dict)
            or output.get("confirmed") is not True
        ):
            raise ValueError("DEFAULT_ASSET_RUNTIME_ACTION_NOT_ACCEPTED")
    for action in (
        "pickup",
        "delivery.confirm-custody",
        "delivery.verify-loaded-stability",
    ):
        if by_action[action][1]["output"].get("detached") is not False:
            raise ValueError("DEFAULT_ASSET_PAYLOAD_NOT_RETAINED")
    return receipts


def _qualification_receipt(run_dir: Path) -> dict[str, Any]:
    evidence_path = run_dir / "mission_evidence.json"
    workflow_path = run_dir / "workflow-result.json"
    evidence = _json(evidence_path)
    workflow = _json(workflow_path)
    if (
        evidence.get("schema_version") != "dronedream.generic-px4-gazebo-run.v1"
        or evidence.get("status") != "verified"
        or workflow.get("schema_version") != "dronedream.simulation-workflow-result.v1"
        or workflow.get("status") != "verified"
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_NOT_VERIFIED")
    gates = evidence.get("gates")
    measurements = evidence.get("measurements")
    if (
        not isinstance(gates, dict)
        or any(gates.get(gate) is not True for gate in CURRENT_RUNTIME_GATES)
        or not isinstance(measurements, dict)
        or measurements.get("executor_return_code") != 0
        or measurements.get("landing_state") != "ON_GROUND"
        or measurements.get("abort_reason") is not None
        or measurements.get("external_abort_request") is not None
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_GATES_INCOMPLETE")
    completion = workflow.get("completion_assessment")
    checkpoint_decisions = workflow.get("checkpoint_decisions")
    if (
        not isinstance(completion, dict)
        or completion.get("accepted") is not True
        or completion.get("issue_codes") != []
        or not isinstance(checkpoint_decisions, list)
        or not checkpoint_decisions
        or any(
            not isinstance(decision, dict)
            or decision.get("continue_authorized") is not True
            or not isinstance(decision.get("assessment"), dict)
            or decision["assessment"].get("action") != "accept"
            or decision["assessment"].get("issue_codes") != []
            for decision in checkpoint_decisions
        )
    ):
        raise ValueError("DEFAULT_ASSET_WORKFLOW_NOT_ACCEPTED")
    action_receipts = _current_runtime_action_receipts(run_dir)
    embedded_action_receipts = workflow.get("runtime_action_receipts")
    if not isinstance(embedded_action_receipts, list):
        raise ValueError("DEFAULT_ASSET_WORKFLOW_ACTION_RECEIPTS_INVALID")
    embedded_by_step = {
        receipt.get("step_id"): receipt
        for receipt in embedded_action_receipts
        if isinstance(receipt, dict) and isinstance(receipt.get("step_id"), str)
    }
    if len(embedded_by_step) != len(embedded_action_receipts) or any(
        embedded_by_step.get(receipt.get("step_id")) != receipt
        for _path, receipt in action_receipts
    ):
        raise ValueError("DEFAULT_ASSET_WORKFLOW_ACTION_RECEIPTS_MISMATCH")
    model_navigation = measurements.get("model_navigation")
    if (
        not isinstance(model_navigation, dict)
        or model_navigation.get("provider") != "local-policy"
        or model_navigation.get("provider_call_count", 0) <= 0
        or model_navigation.get("fallback_provider_call_count") != 0
        or model_navigation.get("invocation_failure_count") != 0
        or model_navigation.get("invocation_timeout_count") != 0
    ):
        raise ValueError("DEFAULT_ASSET_MODEL_NAVIGATION_NOT_QUALIFIED")
    action_evidence = {
        receipt["action"]: {
            "file": f"qualification/runtime-actions/receipts/{path.name}",
            "sha256": _sha256(path),
            "status": "accepted",
        }
        for path, receipt in action_receipts
        if receipt["action"] in CURRENT_RUNTIME_ACTIONS
    }
    return {
        "schema_version": "dronedream.asset-qualification-receipt.v1",
        "source": "real-gazebo-harmonic-px4-sitl",
        "mission_evidence_file": "qualification/mission_evidence.json",
        "mission_evidence_sha256": _sha256(evidence_path),
        "workflow_result_file": "qualification/workflow-result.json",
        "workflow_result_sha256": _sha256(workflow_path),
        "required_gates": {gate: True for gate in CURRENT_RUNTIME_GATES},
        "required_runtime_actions": action_evidence,
        "checkpoint_decision_count": len(checkpoint_decisions),
        "completion_assessment_sha256": _canonical_json_sha256(completion),
        "gazebo_runtime_verified": True,
        "px4_mission_smoke_verified": True,
        "simulation_execution_ready": True,
        "measurements": measurements,
    }


def _copy_qualification_evidence(run_dir: Path, destination: Path) -> None:
    qualification = destination / "qualification"
    qualification.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "mission_evidence.json", qualification / "mission_evidence.json")
    shutil.copyfile(run_dir / "workflow-result.json", qualification / "workflow-result.json")
    receipt_destination = qualification / "runtime-actions" / "receipts"
    receipt_destination.mkdir(parents=True, exist_ok=True)
    for path, _receipt in _current_runtime_action_receipts(run_dir):
        shutil.copyfile(path, receipt_destination / path.name)


def _verified_track_distance_m(run_dir: Path) -> float:
    """Return the exact spatial distance supported by the verified run.

    A promoted vehicle must never inherit a range number from an earlier map or
    route.  The reference track is already hash-bound by the qualification
    evidence, so derive the advertised range from those accepted setpoints.
    ``math.fsum`` avoids adding an artificial rounding margin beyond what the
    vehicle physically completed.
    """

    track = _json(run_dir / "reference_track.json")
    points = track.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("DEFAULT_ASSET_REFERENCE_TRACK_INVALID")
    coordinates: list[tuple[float, float, float]] = []
    for point in points:
        if not isinstance(point, dict) or not all(
            isinstance(point.get(axis), (int, float)) for axis in ("x", "y", "z")
        ):
            raise ValueError("DEFAULT_ASSET_REFERENCE_POINT_INVALID")
        coordinate = tuple(float(point[axis]) for axis in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in coordinate):
            raise ValueError("DEFAULT_ASSET_REFERENCE_POINT_INVALID")
        coordinates.append(coordinate)
    distance_m = math.fsum(
        math.dist(start, end)
        for start, end in zip(coordinates, coordinates[1:], strict=False)
    )
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("DEFAULT_ASSET_REFERENCE_TRACK_DISTANCE_INVALID")
    return distance_m


def _artifact_identity(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep export identity separate from post-export runtime qualification.

    The source exporter deliberately writes the three runtime flags as false
    before Gazebo/PX4 acceptance runs. Carrying those provisional flags into a
    subsequently qualified bundle makes one manifest contradict itself. The
    immutable hashes and physical/export metadata remain the artifact identity;
    runtime truth lives only in the signed qualification receipt copied below.
    """
    identity = dict(summary)
    for field in RUNTIME_QUALIFICATION_FIELDS:
        identity.pop(field, None)
    identity["identity_scope"] = "export-integrity"
    return identity


def _verified_pickup_track_point_index(*, run_dir: Path, point_count: int) -> int:
    """Resolve the physical pickup from current runtime evidence.

    Pickup is a domain action anchored to a runtime checkpoint; it is not a
    navigation phase.  Requiring a ``phase == \"pickup\"`` marker on the flight
    track mixes the retired route encoding with the current action contract and
    can silently bind a promoted map to the wrong mission point.  Require the
    independently produced checkpoint contract and Gazebo payload-spawn receipt
    to agree instead.
    """

    checkpoints = _json(run_dir / "runtime-checkpoints.json")
    payload_spawn = _json(run_dir / "payload_spawn.json")
    if checkpoints.get("schema_version") != "dronedream.runtime-checkpoints.v1":
        raise ValueError("DEFAULT_ASSET_RUNTIME_CHECKPOINTS_INVALID")
    checkpoint_entries = checkpoints.get("checkpoints")
    if not isinstance(checkpoint_entries, list):
        raise ValueError("DEFAULT_ASSET_RUNTIME_CHECKPOINTS_INVALID")
    if payload_spawn.get("accepted") is not True:
        raise ValueError("DEFAULT_ASSET_PAYLOAD_SPAWN_NOT_ACCEPTED")
    checkpoint_id = payload_spawn.get("checkpoint_id")
    track_point_index = payload_spawn.get("track_point_index")
    if (
        not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or not isinstance(track_point_index, int)
        or isinstance(track_point_index, bool)
        or track_point_index <= 0
        or track_point_index >= point_count - 1
    ):
        raise ValueError("DEFAULT_ASSET_PAYLOAD_CHECKPOINT_BINDING_INVALID")
    matching = [
        checkpoint
        for checkpoint in checkpoint_entries
        if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id") == checkpoint_id
    ]
    if len(matching) != 1 or matching[0].get("track_point_index") != track_point_index:
        raise ValueError("DEFAULT_ASSET_PAYLOAD_CHECKPOINT_BINDING_INVALID")
    return track_point_index


def _synchronize_verified_route_graph(*, run_dir: Path, graph: dict[str, Any]) -> None:
    """Bind verified graph nodes and limits to the exact accepted flight track."""
    track = _json(run_dir / "reference_track.json")
    points = track.get("points")
    contract = track.get("coordinate_contract")
    if not isinstance(points, list) or not isinstance(contract, dict):
        raise ValueError("DEFAULT_ASSET_REFERENCE_TRACK_INVALID")
    pickup_index = _verified_pickup_track_point_index(
        run_dir=run_dir,
        point_count=len(points),
    )
    outbound = points[: pickup_index + 1]
    model_root = contract.get("model_root_world_enu_m")
    collision_offset = contract.get("collision_center_offset_model_m")
    if (
        not isinstance(model_root, list)
        or len(model_root) != 3
        or not all(isinstance(value, (int, float)) for value in model_root)
        or not isinstance(collision_offset, list)
        or len(collision_offset) != 3
        or not all(isinstance(value, (int, float)) for value in collision_offset)
    ):
        raise ValueError("DEFAULT_ASSET_REFERENCE_FRAME_INVALID")

    graph_nodes = graph.get("nodes")
    graph_edges = graph.get("edges")
    if not isinstance(graph_nodes, list) or not isinstance(graph_edges, list):
        raise ValueError("DEFAULT_ASSET_GRAPH_INVALID")
    verified_nodes = {
        node.get("node_id"): node
        for node in graph_nodes
        if isinstance(node, dict)
        and isinstance(node.get("node_id"), str)
        and node["node_id"].startswith("verified-")
    }
    if len(verified_nodes) != len(outbound):
        raise ValueError("DEFAULT_ASSET_GRAPH_VERIFIED_NODE_COUNT_MISMATCH")

    world_positions: list[dict[str, float]] = []
    for index, point in enumerate(outbound):
        if not isinstance(point, dict) or not all(
            isinstance(point.get(axis), (int, float)) for axis in ("x", "y", "z")
        ):
            raise ValueError("DEFAULT_ASSET_REFERENCE_POINT_INVALID")
        position = {
            "x": float(model_root[0]) + float(collision_offset[0]) + float(point["y"]),
            "y": float(model_root[1]) + float(collision_offset[1]) + float(point["x"]),
            "z": float(model_root[2]) + float(collision_offset[2]) + float(point["z"]),
        }
        node_id = f"verified-{index:03d}"
        node = verified_nodes.get(node_id)
        if node is None:
            raise ValueError(f"DEFAULT_ASSET_GRAPH_NODE_MISSING:{node_id}")
        node["position_m"] = position
        world_positions.append(position)

    verified_edges = [
        edge
        for edge in graph_edges
        if isinstance(edge, dict) and edge.get("qualification") == "flight-verified"
    ]
    if len(verified_edges) != len(outbound) - 1:
        raise ValueError("DEFAULT_ASSET_GRAPH_VERIFIED_EDGE_COUNT_MISMATCH")
    edges_by_id = {edge.get("edge_id"): edge for edge in verified_edges}
    for index in range(len(outbound) - 1):
        edge_id = f"verified-edge-{index:03d}"
        edge = edges_by_id.get(edge_id)
        if edge is None:
            raise ValueError(f"DEFAULT_ASSET_GRAPH_EDGE_MISSING:{edge_id}")
        start = world_positions[index]
        end = world_positions[index + 1]
        edge["distance_m"] = math.dist(
            (start["x"], start["y"], start["z"]),
            (end["x"], end["y"], end["z"]),
        )
        destination = outbound[index + 1]
        speed_limit = destination.get("speed_limit_mps")
        if not isinstance(speed_limit, (int, float)) or float(speed_limit) <= 0:
            raise ValueError("DEFAULT_ASSET_REFERENCE_SPEED_INVALID")
        edge["speed_limit_mps"] = float(speed_limit)


def _write_current_map_contract(semantic_path: Path) -> None:
    """Validate and canonically rewrite only the current first-party map contract."""

    semantic = _json(semantic_path)
    if semantic.get("schema_version") != "dronedream.map-semantic.v1" or (
        "simulation_bindings" in semantic
    ):
        raise ValueError("DEFAULT_MAP_CONTRACT_OBSOLETE")
    if semantic.get("coordinate_frame") != "ENU":
        raise ValueError("DEFAULT_MAP_COORDINATE_FRAME_INVALID")
    entities = semantic.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("DEFAULT_MAP_ENTITY_CATALOG_MISSING")
    entity_ids = {entry.get("entity_id") for entry in entities if isinstance(entry, dict)}
    if not {"office-launch-pad", "takeout-pickup-pad"}.issubset(entity_ids):
        raise ValueError("DEFAULT_MAP_MISSION_ENTITIES_MISSING")
    runtime_bindings = semantic.get("runtime_bindings")
    if (
        not isinstance(runtime_bindings, dict)
        or runtime_bindings.get("schema_version") != "dronedream.map-runtime-bindings.v1"
        or runtime_bindings.get("simulator") != "gazebo-harmonic"
        or runtime_bindings.get("coordinate_frame") != "ENU"
        or not isinstance(runtime_bindings.get("vehicle_spawn"), dict)
        or not isinstance(runtime_bindings.get("mission_launch_waypoint"), dict)
    ):
        raise ValueError("DEFAULT_MAP_RUNTIME_BINDINGS_INVALID")
    _write_json(semantic_path, semantic)


def _build_map_bundle(
    run_dir: Path,
    graph_source: Path,
    destination: Path,
    map_source: Path,
) -> None:
    _copy_tree(map_source, destination / "gazebo")
    _write_current_map_contract(destination / "gazebo" / "semantic.json")
    graph = _json(graph_source)
    evidence_sha256 = _sha256(run_dir / "mission_evidence.json")
    graph["asset_id"] = MAP_ASSET_ID
    graph["name"] = "School Map"
    _synchronize_verified_route_graph(run_dir=run_dir, graph=graph)
    for edge in graph["edges"]:
        if edge.get("qualification") == "flight-verified":
            edge["evidence_sha256"] = evidence_sha256
    _write_json(destination / "navigation-graph.json", graph)
    qualification = _qualification_receipt(run_dir)
    _write_json(destination / "qualification" / "receipt.json", qualification)
    _copy_qualification_evidence(run_dir, destination)
    summary = _artifact_identity(_json(map_source / "summary.json"))
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": "dronedream.asset-bundle.v1",
            "kind": "map",
            "asset_id": MAP_ASSET_ID,
            "name": "School Map",
            "qualification_status": "qualified",
            "source": "dronedream-bundled",
            "coordinate_frame": "map_enu",
            "files": {
                "graph": "navigation-graph.json",
                "semantic": "gazebo/semantic.json",
                "world_sdf": "gazebo/world.sdf",
                "physics_world_sdf": "gazebo/world.physics.sdf",
                "qualification_receipt": "qualification/receipt.json",
            },
            "artifact_identity": summary,
            "qualification": qualification,
        },
    )


def _build_vehicle_bundle(
    run_dir: Path,
    destination: Path,
    vehicle_source: Path,
    controller_params: Path,
) -> None:
    _copy_tree(vehicle_source, destination / "gazebo")
    shutil.copyfile(controller_params, destination / "controller_params.json")
    vehicle_summary = _artifact_identity(_json(vehicle_source / "summary.json"))
    vehicle_sdf = (vehicle_source / "model.sdf").read_text(encoding="utf-8")
    perception = vehicle_summary.get("perception")
    if (
        vehicle_summary.get("source_model") != "model://x500_depth"
        or "model://x500_depth" not in vehicle_sdf
    ):
        raise ValueError("DEFAULT_VEHICLE_DEPTH_RUNTIME_MISSING")
    if perception is None:
        perception = {
            "sensor_id": "oakd-lite-depth",
            "source_model": "model://x500_depth",
            "streams": ["forward-depth", "forward-rgb"],
        }
    if (
        not isinstance(perception, dict)
        or perception.get("sensor_id") != "oakd-lite-depth"
        or perception.get("source_model") != "model://x500_depth"
    ):
        raise ValueError("DEFAULT_VEHICLE_PERCEPTION_CONTRACT_INVALID")
    vehicle_summary["perception"] = perception
    mission_payload = vehicle_summary.get("mission_payload")
    if not isinstance(mission_payload, dict):
        raise ValueError("DEFAULT_VEHICLE_PAYLOAD_CONTRACT_MISSING")
    dry_mass_kg = float(vehicle_summary["dry_mass_kg"])
    qualified_payload_kg = float(mission_payload["mass_kg"])
    design_payload_limit_kg = float(vehicle_summary["maximum_qualified_payload_kg"])
    loaded_thrust_to_weight = float(mission_payload["loaded_thrust_to_weight"])
    if qualified_payload_kg <= 0.0 or qualified_payload_kg > design_payload_limit_kg:
        raise ValueError("DEFAULT_VEHICLE_PAYLOAD_QUALIFICATION_INVALID")
    vehicle_summary["maximum_design_payload_kg"] = design_payload_limit_kg
    vehicle_summary["maximum_qualified_payload_kg"] = qualified_payload_kg
    _write_json(destination / "gazebo" / "summary.json", vehicle_summary)
    qualified_range_m = _verified_track_distance_m(run_dir)
    vehicle = {
        "schema_version": "dronedream.vehicle.v1",
        "asset_id": VEHICLE_ASSET_ID,
        "name": "My Drone",
        "coordinate_frame": "base_link_frd",
        "dry_mass_kg": dry_mass_kg,
        "max_takeoff_mass_kg": dry_mass_kg + qualified_payload_kg,
        "body_radius_m": 0.38,
        "body_height_m": 0.43,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.228},
        "max_speed_mps": 1.2,
        "max_acceleration_mps2": 0.8,
        # Bound to the exact successfully completed Gazebo/PX4 pickup-return
        # reference track; no value is inherited from an older route.
        "qualified_range_m": qualified_range_m,
        "reserve_battery_percent": 30.0,
        "max_pickup_payload_kg": qualified_payload_kg,
        "sensors": [
            "imu",
            "magnetometer",
            "barometer",
            "gps",
            "odometry",
            "oakd-lite-depth",
        ],
    }
    _write_json(destination / "vehicle.json", vehicle)
    qualification = _qualification_receipt(run_dir)
    _write_json(destination / "qualification" / "receipt.json", qualification)
    _copy_qualification_evidence(run_dir, destination)
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": "dronedream.asset-bundle.v1",
            "kind": "vehicle",
            "asset_id": VEHICLE_ASSET_ID,
            "name": "My Drone",
            "qualification_status": "qualified",
            "source": "dronedream-bundled",
            "files": {
                "vehicle_sdf": "gazebo/model.sdf",
                "controller_params": "controller_params.json",
                "vehicle_metadata": "vehicle.json",
                "payload_sdf": "gazebo/takeout-payload.sdf",
                "qualification_receipt": "qualification/receipt.json",
            },
            "artifact_identity": vehicle_summary,
            "physical_material_contract": {
                "source": "PX4 x500 pinned SDF",
                "mass_and_inertia_from_source_model": True,
                "rotor_dynamics_from_source_model": True,
                "qualified_range_m": qualified_range_m,
                "qualified_range_basis": "completed Gazebo/PX4 pickup-return route",
                "payload_mass_kg": qualified_payload_kg,
                "loaded_thrust_to_weight": loaded_thrust_to_weight,
                "perception_sensor": perception,
            },
            "qualification": qualification,
        },
    )


def _qualified_runtime_evidence(
    *,
    plan: Any,
    qualification_root: Path,
    mission_evidence_path: Path,
) -> dict[str, Any]:
    source = _json(mission_evidence_path)
    source_gates = source.get("gates")
    measurements = source.get("measurements")
    if (
        source.get("schema_version") != "dronedream.generic-px4-gazebo-run.v1"
        or source.get("status") != "verified"
        or not isinstance(source_gates, dict)
        or not isinstance(measurements, dict)
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_NOT_VERIFIED")
    gates = {gate: source_gates.get(gate) is True for gate in plan.required_runtime_gates}
    inputs = plan.inputs
    artifacts = {
        "world_sha256": _sha256(qualification_root / inputs.world_sdf),
        "semantic_sha256": _sha256(qualification_root / inputs.semantic),
        "vehicle_sha256": _sha256(qualification_root / inputs.vehicle_sdf),
        "controller_params_sha256": _sha256(qualification_root / inputs.controller_params),
        "route_sha256": plan.route_sha256,
        "track_sha256": plan.track_sha256,
        "clearance_sha256": plan.clearance_sha256,
        "source_mission_evidence_sha256": _sha256(mission_evidence_path),
    }
    return {
        "schema_version": "dronedream.default-assets-real-gazebo-px4.v1",
        "status": "verified",
        "world": inputs.world_name,
        "vehicle": inputs.vehicle_name,
        "gates": gates,
        "measurements": measurements,
        "source_mission_evidence": source,
        "artifacts": artifacts,
    }


def _build_qualified_pair(
    *,
    output: Path,
    map_root: Path,
    vehicle_root: Path,
    temporary: Path,
    run_dir: Path,
    environment_versions: dict[str, str],
    supersedes_index: Path | None,
) -> dict[str, Any]:
    qualified_at = datetime.fromtimestamp(
        (run_dir / "mission_evidence.json").stat().st_mtime,
        tz=UTC,
    )
    source_map = _write_current_source_package(
        source_root=map_root,
        destination=temporary / "school-map-source.ddpkg",
        kind="map",
        asset_id=MAP_ASSET_ID,
        name="School Map",
        created_at=qualified_at,
    )
    source_vehicle = _write_current_source_package(
        source_root=vehicle_root,
        destination=temporary / "my-drone-source.ddpkg",
        kind="vehicle",
        asset_id=VEHICLE_ASSET_ID,
        name="My Drone",
        created_at=qualified_at,
    )
    qualification_root = temporary / "qualification"
    plan = prepare_asset_pair_qualification(
        map_archive=source_map,
        vehicle_archive=source_vehicle,
        work_root=qualification_root,
    )
    mission_evidence_path = run_dir / "mission_evidence.json"
    runtime_evidence = _qualified_runtime_evidence(
        plan=plan,
        qualification_root=qualification_root,
        mission_evidence_path=mission_evidence_path,
    )
    receipt = build_pair_qualification_receipt(
        plan=plan,
        work_root=qualification_root,
        runtime_evidence=runtime_evidence,
        environment_versions=environment_versions,
        qualified_at=qualified_at,
    )
    map_package = bind_qualification_receipt(
        source_archive=source_map,
        destination_archive=output / "school-map.ddpkg",
        receipt=receipt,
    )
    vehicle_package = bind_qualification_receipt(
        source_archive=source_vehicle,
        destination_archive=output / "my-drone.ddpkg",
        receipt=receipt,
    )
    source_by_kind = {
        "map": inspect_ddpkg(source_map),
        "vehicle": inspect_ddpkg(source_vehicle),
    }
    qualified_by_kind = {
        "map": inspect_ddpkg(map_package),
        "vehicle": inspect_ddpkg(vehicle_package),
    }
    packages = []
    for kind, filename in (("map", "school-map.ddpkg"), ("vehicle", "my-drone.ddpkg")):
        source = source_by_kind[kind]
        qualified = qualified_by_kind[kind]
        packages.append(
            {
                "kind": kind,
                "asset_id": qualified.manifest.asset_id,
                "file": filename,
                "sha256": _sha256(output / filename),
                "source_content_sha256": source.manifest.content_sha256,
                "content_sha256": qualified.manifest.content_sha256,
            }
        )

    superseded = _superseded_pair_records(supersedes_index)
    receipt_payload = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    return {
        "schema_version": "dronedream.bundled-qualified-pair.v1",
        "qualification_id": receipt.qualification_id,
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "packages": packages,
        "superseded_qualified_pairs": superseded,
    }


def _superseded_pair_records(index_path: Path | None) -> list[dict[str, str]]:
    """Carry the complete retired-pair chain into the next bundled index."""

    if index_path is None:
        return []
    old = _json(index_path)
    old_pair = old.get("qualified_pair")
    old_packages = old_pair.get("packages") if isinstance(old_pair, dict) else None
    inherited = old.get("superseded_qualified_pairs", [])
    if not isinstance(old_packages, list) or not isinstance(inherited, list):
        raise ValueError("SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID")
    old_by_kind = {
        str(item.get("kind")): item for item in old_packages if isinstance(item, dict)
    }
    if not isinstance(old_by_kind.get("map"), dict) or not isinstance(
        old_by_kind.get("vehicle"), dict
    ):
        raise ValueError("SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID")
    records: list[object] = [
        *inherited,
        {
            "qualification_id": old_pair.get("qualification_id"),
            "map_content_sha256": old_by_kind["map"].get("content_sha256"),
            "map_source_content_sha256": old_by_kind["map"].get(
                "source_content_sha256"
            ),
            "vehicle_content_sha256": old_by_kind["vehicle"].get("content_sha256"),
            "vehicle_source_content_sha256": old_by_kind["vehicle"].get(
                "source_content_sha256"
            ),
        },
    ]
    keys = (
        "qualification_id",
        "map_content_sha256",
        "map_source_content_sha256",
        "vehicle_content_sha256",
        "vehicle_source_content_sha256",
    )
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID")
        values = tuple(record.get(key) for key in keys)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID")
        identity = tuple(str(value) for value in values)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(dict(zip(keys, identity, strict=True)))
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--map-package", type=Path, required=True)
    parser.add_argument("--vehicle-development-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes-index", type=Path)
    parser.add_argument("--gazebo-version", required=True)
    parser.add_argument("--ros-distribution", required=True)
    parser.add_argument("--px4-commit", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"default asset output is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="dronedream-default-assets-", dir=output.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        map_root = temporary / "map"
        vehicle_root = temporary / "vehicle"
        map_source, graph = _extract_current_map_package(
            args.map_package.resolve(), temporary / "current-map"
        )
        _development_input, controller_params = _stage_development_vehicle(
            args.vehicle_development_input.resolve(), temporary / "current-vehicle"
        )
        _assert_promotion_input_binding(
            run_dir=run_dir,
            map_source=map_source,
            vehicle_source=temporary / "current-vehicle",
            controller_params=controller_params,
        )
        _build_map_bundle(run_dir, graph, map_root, map_source)
        _build_vehicle_bundle(
            run_dir,
            vehicle_root,
            temporary / "current-vehicle",
            controller_params,
        )
        output.mkdir(parents=True, exist_ok=True)
        qualified_pair = _build_qualified_pair(
            output=output,
            map_root=map_root,
            vehicle_root=vehicle_root,
            temporary=temporary,
            run_dir=run_dir,
            environment_versions={
                "gazebo_sim": args.gazebo_version,
                "ros_distribution": args.ros_distribution,
                "px4_commit": args.px4_commit,
                "runtime_manifest_sha256": args.runtime_manifest_sha256,
            },
            supersedes_index=(
                args.supersedes_index.resolve() if args.supersedes_index is not None else None
            ),
        )
        _write_json(
            output / "index.json",
            {
                "schema_version": "dronedream.bundled-assets.v2",
                "qualified_pair": {
                    key: value
                    for key, value in qualified_pair.items()
                    if key != "superseded_qualified_pairs"
                },
                "superseded_qualified_pairs": qualified_pair["superseded_qualified_pairs"],
            },
        )


if __name__ == "__main__":
    main()
