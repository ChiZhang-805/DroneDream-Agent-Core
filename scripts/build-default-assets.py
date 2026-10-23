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
from dronedream_agent_core.asset_payload_qualification import verify_asset_payload_run

MAP_ASSET_ID = "dronedream.school-map.v1"
VEHICLE_ASSET_ID = "dronedream.my-drone.v1"
ARCHIVE_TIMESTAMP = (2026, 8, 19, 0, 0, 0)
RUNTIME_QUALIFICATION_FIELDS = (
    "gazebo_runtime_verified",
    "px4_mission_smoke_verified",
    "simulation_execution_ready",
)
CURRENT_RUNTIME_GATES = (
    "executor_completed",
    "landed_on_office_pad",
    "loaded_thrust_to_weight_qualified",
    "offboard_timing_complete",
    "office_return_reached",
    "physical_payload_attached",
    "physical_payload_retained",
    "physical_payload_spawned",
    "pickup_reached",
    "px4_landing_confirmed",
    "takeoff_gate_achieved",
    "zero_unsafe_dynamic_penetrations",
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


def _stage_current_vehicle(
    source_root: Path,
    controller_params: Path,
    destination: Path,
) -> Path:
    """Stage the exact vehicle bytes captured by the verified Kumpula run."""

    required = ("model.sdf", "summary.json", "takeout-payload.sdf")
    if not source_root.is_dir() or any(not (source_root / name).is_file() for name in required):
        raise FileNotFoundError("verified vehicle source is incomplete")
    if not controller_params.is_file():
        raise FileNotFoundError("verified controller parameters are unavailable")
    destination.mkdir(parents=True, exist_ok=False)
    _copy_tree(source_root, destination)
    if (destination / "model.config").is_file():
        return controller_params
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
    return controller_params


def _assert_promotion_input_binding(
    *,
    run_dir: Path,
    map_source: Path,
    vehicle_source: Path,
    controller_params: Path,
) -> None:
    """Require qualification evidence to hash-bind every promoted runtime input."""

    if (run_dir / 'asset-qualification-inputs.json').is_file():
        verify_asset_payload_run(run_dir)
        inputs = _json(run_dir / 'asset-qualification-inputs.json')
        for root, key in ((map_source, 'map_files'), (vehicle_source, 'vehicle_files')):
            for relative, digest in inputs[key].items():
                candidate = (root / relative).resolve()
                if not candidate.is_relative_to(root.resolve()) or _sha256(candidate) != digest:
                    raise ValueError('DEFAULT_ASSET_PROMOTION_INPUT_MISMATCH')
        if _sha256(controller_params) != inputs['runtime_inputs']['controller_params_sha256']:
            raise ValueError('DEFAULT_ASSET_PROMOTION_INPUT_MISMATCH')
        return

    preflight = _json(run_dir / "preflight_contract.json")
    map_hashes = preflight.get("school_map_package_sha256")
    vehicle_hashes = preflight.get("my_drone_package_sha256")
    run_map = run_dir / "school-map"
    run_vehicle = run_dir / "my-drone"
    source_semantic = _json(map_source / "semantic.json")
    source_semantic.pop("collision_primitives", None)
    run_semantic = _json(run_map / "semantic.json")
    for semantic in (source_semantic, run_semantic):
        # Human-facing labels may be corrected after a physical qualification run
        # without changing the qualified geometry, entity identities, or positions.
        # Keep every operational field hash-bound while excluding only those labels.
        semantic.pop("scene_id", None)
        semantic.pop("display_name", None)
        semantic.pop("aliases", None)
        for entity in semantic.get("entities", []):
            if isinstance(entity, dict):
                entity.pop("name", None)
                entity.pop("name_zh", None)
                entity.pop("aliases", None)
                # `semantic` is the current parser-facing spelling of the retained
                # historical `kind` value.  Treat the duplicate as a schema repair,
                # not as a change to the qualified geometry or mission behavior.
                if entity.get("semantic") == entity.get("kind"):
                    entity.pop("semantic", None)
        bindings = semantic.get("runtime_bindings")
        if isinstance(bindings, dict):
            spawn = bindings.get("vehicle_spawn")
            launch = bindings.get("mission_launch_waypoint")
            bindings.clear()
            bindings.update(
                {
                    "schema_version": "dronedream.map-runtime-bindings.v1",
                    "simulator": "gazebo-harmonic",
                    "coordinate_frame": "ENU",
                    "vehicle_spawn": {axis: spawn[axis] for axis in ("x", "y", "z")}
                    if isinstance(spawn, dict)
                    else spawn,
                    "mission_launch_waypoint": {axis: launch[axis] for axis in ("x", "y", "z")}
                    if isinstance(launch, dict)
                    else launch,
                }
            )
    if (
        not isinstance(map_hashes, dict)
        or not isinstance(vehicle_hashes, dict)
        or _sha256(map_source / "world.sdf") != map_hashes.get("world.sdf")
        or _sha256(run_map / "world.sdf") != map_hashes.get("world.sdf")
        or _sha256(map_source / "collision-primitives.json")
        != map_hashes.get("collision-primitives.json")
        or source_semantic != run_semantic
        or _sha256(vehicle_source / "model.sdf") != vehicle_hashes.get("model.sdf")
        or _sha256(run_vehicle / "model.sdf") != vehicle_hashes.get("model.sdf")
        or controller_params.read_bytes() != (run_dir / "controller_params.json").read_bytes()
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


def _qualification_receipt(run_dir: Path) -> dict[str, Any]:
    evidence_path = run_dir / "mission_evidence.json"
    if (run_dir / 'asset-qualification-inputs.json').is_file():
        verified = verify_asset_payload_run(run_dir)
        return {
            'schema_version': 'dronedream.asset-qualification-receipt.v1',
            'source': 'real-gazebo-harmonic-px4-sitl',
            'mission_evidence_file': 'qualification/mission_evidence.json',
            'mission_evidence_sha256': _sha256(evidence_path),
            'required_gates': verified['gates'],
            'pose_sample_count': verified['measurements']['pose_sample_count'],
            'payload_spawn_file': 'qualification/payload_spawn.json',
            'payload_spawn_sha256': _sha256(run_dir / 'payload_spawn.json'),
            'gazebo_runtime_verified': True, 'px4_mission_smoke_verified': True,
            'simulation_execution_ready': True,
            'measurements': verified['measurements'],
            'verified_source_files_sha256': verified['verified_source_files_sha256'],
            'verified_native_runtime_files_sha256': (
                verified['verified_native_runtime_files_sha256']
            ),
        }
    evidence = _json(evidence_path)
    if (
        evidence.get("schema_version") != "dronedream.school-map-px4-mission-evidence.v1"
        or evidence.get("status") != "verified"
        or evidence.get("gazebo_runtime_verified") is not True
        or evidence.get("px4_mission_smoke_verified") is not True
        or evidence.get("simulation_execution_ready") is not True
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_NOT_VERIFIED")
    gates = evidence.get("gates")
    measurements = evidence.get("measurements")
    if (
        not isinstance(gates, dict)
        or any(gates.get(gate) is not True for gate in CURRENT_RUNTIME_GATES)
        or not isinstance(measurements, dict)
        or evidence.get("executor_return_code") != 0
        or evidence.get("process_failure") is not None
        or not isinstance(evidence.get("pose_sample_count"), int)
        or evidence["pose_sample_count"] < 10
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_GATES_INCOMPLETE")
    return {
        "schema_version": "dronedream.asset-qualification-receipt.v1",
        "source": "real-gazebo-harmonic-px4-sitl",
        "mission_evidence_file": "qualification/mission_evidence.json",
        "mission_evidence_sha256": _sha256(evidence_path),
        "required_gates": {gate: True for gate in CURRENT_RUNTIME_GATES},
        "pose_sample_count": evidence["pose_sample_count"],
        "payload_spawn_file": "qualification/payload_spawn.json",
        "payload_spawn_sha256": _sha256(run_dir / "payload_spawn.json"),
        "gazebo_runtime_verified": True,
        "px4_mission_smoke_verified": True,
        "simulation_execution_ready": True,
        "measurements": measurements,
    }


def _copy_qualification_evidence(run_dir: Path, destination: Path) -> None:
    qualification = destination / "qualification"
    qualification.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'asset-qualification-inputs.json').is_file():
        verified = verify_asset_payload_run(run_dir)
        for name in verified['verified_source_files_sha256']:
            target = qualification / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / name, target)
        _write_json(qualification / 'asset-qualification-evidence.json', verified)
        return
    for name in (
        "mission_evidence.json",
        "preflight_contract.json",
        "reference_track.json",
        "payload_spawn.json",
        "offboard_timing.json",
        "live_safety_gate.json",
        "px4_state_policy.json",
    ):
        shutil.copyfile(run_dir / name, qualification / name)


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
        math.dist(start, end) for start, end in zip(coordinates, coordinates[1:], strict=False)
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

    payload_spawn = _json(run_dir / "payload_spawn.json")
    if payload_spawn.get("accepted") is not True:
        raise ValueError("DEFAULT_ASSET_PAYLOAD_SPAWN_NOT_ACCEPTED")
    track = _json(run_dir / "reference_track.json")
    points = track.get("points")
    pickup_indices = [
        index
        for index, point in enumerate(points if isinstance(points, list) else [])
        if isinstance(point, dict) and point.get("phase") == "pickup"
    ]
    if len(pickup_indices) != 1:
        raise ValueError("DEFAULT_ASSET_PAYLOAD_CHECKPOINT_BINDING_INVALID")
    track_point_index = pickup_indices[0]
    if track_point_index <= 0 or track_point_index >= point_count - 1:
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
    if collision_offset is None and isinstance(
        contract.get("collision_center_above_model_root_m"), (int, float)
    ):
        collision_offset = [0.0, 0.0, float(contract["collision_center_above_model_root_m"])]
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
        if isinstance(edge, dict)
        and isinstance(edge.get("edge_id"), str)
        and edge["edge_id"].startswith("verified-edge-")
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
        edge["qualification"] = "flight-verified"
        edge["bidirectional"] = True


def _write_current_map_contract(semantic_path: Path) -> None:
    """Validate and canonically rewrite only the current first-party map contract."""

    semantic = _json(semantic_path)
    if semantic.get("schema_version") != "dronedream.map-semantic.v1" or (
        "simulation_bindings" in semantic
    ):
        raise ValueError("DEFAULT_MAP_CONTRACT_OBSOLETE")
    if semantic.get("coordinate_frame") != "ENU":
        raise ValueError("DEFAULT_MAP_COORDINATE_FRAME_INVALID")
    if semantic.get("scene_id") != "kumpula-campus":
        raise ValueError("DEFAULT_MAP_SCENE_ID_INVALID")
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
    semantic["runtime_bindings"] = {
        "schema_version": "dronedream.map-runtime-bindings.v1",
        "simulator": "gazebo-harmonic",
        "coordinate_frame": "ENU",
        "vehicle_spawn": {
            axis: runtime_bindings["vehicle_spawn"][axis] for axis in ("x", "y", "z")
        },
        "mission_launch_waypoint": {
            axis: runtime_bindings["mission_launch_waypoint"][axis] for axis in ("x", "y", "z")
        },
    }
    _write_json(semantic_path, semantic)


def _build_map_bundle(
    run_dir: Path,
    graph_source: Path,
    destination: Path,
    map_source: Path,
) -> None:
    _copy_tree(map_source, destination / "gazebo")
    if not (run_dir / 'asset-qualification-inputs.json').is_file():
        _write_current_map_contract(destination / "gazebo" / "semantic.json")
    graph = _json(graph_source)
    evidence_sha256 = _sha256(run_dir / "mission_evidence.json")
    graph["asset_id"] = MAP_ASSET_ID
    graph["name"] = "Kumpula Campus"
    # An acceptance flight is evidence about that run, never the source of a
    # campus road network. Keep every authored junction and surface constraint.
    _validate_ground_route_graph(
        graph=graph, semantic_path=destination / "gazebo" / "semantic.json"
    )
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
            "name": "Kumpula Campus",
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


def _validate_ground_route_graph(*, graph: dict[str, Any], semantic_path: Path) -> None:
    """Reject a stale elevated graph without rewriting any of its geometry."""
    from dronedream_agent_core.contracts import MapAsset
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner

    semantic = _json(semantic_path)
    if not isinstance(semantic.get("traversability"), dict):
        raise ValueError("DEFAULT_MAP_GROUND_TRAVERSABILITY_REQUIRED")
    model = MapAsset.model_validate(graph)
    # Full bundled X500 envelope, not a zero-radius point. The separate pair
    # qualifier still validates the actual selected vehicle and physical run.
    planner = KnownMapMetricPlanner(
        graph=model, semantic_path=semantic_path,
        vehicle_diameter_m=0.76, vehicle_height_m=0.6,
    )
    positions = {
        node.node_id: (node.position_m.x, node.position_m.y, node.position_m.z)
        for node in model.nodes
    }
    for edge in model.edges:
        if not planner._segment_is_free(positions[edge.from_node], positions[edge.to_node]):
            raise ValueError(f"DEFAULT_MAP_EDGE_OUTSIDE_TRAVERSABLE_GROUND:{edge.edge_id}")


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
    elif isinstance(perception, dict):
        perception = dict(perception)
        perception.setdefault("source_model", vehicle_summary.get("source_model"))
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
    if not (run_dir / "asset-qualification-inputs.json").is_file():
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
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 1.5,
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
    if (mission_evidence_path.parent / 'asset-qualification-inputs.json').is_file():
        # Preserve all seven native input hashes. The pair importer consumes
        # these exact flight artifacts instead of preparing a different route.
        verify_asset_payload_run(mission_evidence_path.parent)
        return source
    source_gates = source.get("gates")
    measurements = source.get("measurements")
    if (
        source.get("schema_version") != "dronedream.school-map-px4-mission-evidence.v1"
        or source.get("status") != "verified"
        or not isinstance(source_gates, dict)
        or any(source_gates.get(gate) is not True for gate in CURRENT_RUNTIME_GATES)
        or not isinstance(measurements, dict)
    ):
        raise ValueError("DEFAULT_ASSET_SOURCE_RUN_NOT_VERIFIED")
    # The pair API retains historical generic gate names.  Bind each of them
    # to a concrete fact in the current Kumpula run rather than copying a stale
    # boolean from an older test harness.
    compatibility_gates = {
        "executor_completed": source_gates["executor_completed"],
        "offboard_timing_complete": source_gates["offboard_timing_complete"],
        "runtime_pose_samples_present": source.get("pose_sample_count", 0) >= 10,
        "ros_observations_present": source.get("pose_sample_count", 0) >= 10,
        "goal_observed": (source_gates["pickup_reached"] and source_gates["office_return_reached"]),
        "landing_confirmed": source_gates["px4_landing_confirmed"],
        "native_terminal_lifecycle_published": source.get("process_failure") is None,
        "no_live_abort": source.get("process_failure") is None,
        "px4_ulog_present": bool(source.get("artifacts", {}).get("px4_ulogs")),
        "static_route_clearance_bound": source_gates["zero_unsafe_dynamic_penetrations"],
    }
    gates = {gate: compatibility_gates.get(gate) is True for gate in plan.required_runtime_gates}
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
        name="Kumpula Campus",
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
    recorded = (run_dir / 'asset-qualification-inputs.json').is_file()
    plan = prepare_asset_pair_qualification(
        map_archive=source_map,
        vehicle_archive=source_vehicle,
        work_root=qualification_root,
        maximum_translation_m=300.0,
        **({'recorded_route_path': run_dir / 'mission-route.json',
            'recorded_track_path': run_dir / 'reference_track.json',
            'recorded_clearance_path': run_dir / 'clearance.json'} if recorded else {}),
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
        "resource_id": "dronedream-school-map",
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
    old_by_kind = {str(item.get("kind")): item for item in old_packages if isinstance(item, dict)}
    if not isinstance(old_by_kind.get("map"), dict) or not isinstance(
        old_by_kind.get("vehicle"), dict
    ):
        raise ValueError("SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID")
    records: list[object] = [
        *inherited,
        {
            "qualification_id": old_pair.get("qualification_id"),
            "map_content_sha256": old_by_kind["map"].get("content_sha256"),
            "map_source_content_sha256": old_by_kind["map"].get("source_content_sha256"),
            "vehicle_content_sha256": old_by_kind["vehicle"].get("content_sha256"),
            "vehicle_source_content_sha256": old_by_kind["vehicle"].get("source_content_sha256"),
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
    parser.add_argument("--map-source", type=Path, required=True)
    parser.add_argument("--map-graph", type=Path, required=True)
    parser.add_argument("--vehicle-source", type=Path, required=True)
    parser.add_argument("--controller-params", type=Path, required=True)
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
        map_source = args.map_source.resolve()
        graph = args.map_graph.resolve()
        if not map_source.is_dir() or not graph.is_file():
            raise FileNotFoundError("current Kumpula map source or graph is unavailable")
        recorded = (run_dir / 'asset-qualification-inputs.json').is_file()
        if not recorded:
            _write_current_map_contract(map_source / "semantic.json")
        controller_params = _stage_current_vehicle(
            args.vehicle_source.resolve(),
            args.controller_params.resolve(),
            temporary / "current-vehicle",
        )
        _assert_promotion_input_binding(
            run_dir=run_dir,
            map_source=map_source,
            vehicle_source=(args.vehicle_source.resolve() if recorded
                            else temporary / "current-vehicle"),
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
        current_pair = {
            key: value
            for key, value in qualified_pair.items()
            if key != "superseded_qualified_pairs"
        }
        superseded = [
            record
            for record in qualified_pair["superseded_qualified_pairs"]
            if record["qualification_id"] != current_pair["qualification_id"]
        ]
        _write_json(
            output / "index.json",
            {
                "schema_version": "dronedream.bundled-assets.v3",
                "default_qualification_id": current_pair["qualification_id"],
                "qualified_pair": current_pair,
                "qualified_pairs": [current_pair],
                "superseded_qualified_pairs": superseded,
            },
        )


if __name__ == "__main__":
    main()
