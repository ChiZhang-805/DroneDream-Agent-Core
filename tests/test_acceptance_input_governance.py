from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.asset_runtime_resolver import (
    AssetRuntimeResolutionError,
    ResolvedMapAsset,
    ResolvedVehicleAsset,
    assert_development_map_binding,
    rebind_development_mission_input_to_map,
    rebind_development_mission_input_to_qualified_vehicle,
    resolve_development_mission_input,
)
from dronedream_agent_app.storage import AppStore


def _load_planning_acceptance_module():
    path = Path(__file__).parents[1] / "scripts" / "run_pluginized_planning_acceptance.py"
    spec = importlib.util.spec_from_file_location("planning_acceptance_governance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_current_mission_manifest(
    root: Path, *, map_binding: dict[str, str] | None = None
) -> Path:
    files = {
        "vehicle_sdf": (
            "model.sdf",
            "<sdf><model name='my_drone'><include><uri>model://x500_depth</uri>"
            "</include><plugin name='DetachableJoint'>"
            "<parent_link>base_link</parent_link>"
            "<child_model>takeout_payload</child_model>"
            "<child_link>payload_link</child_link>"
            "<attach_topic>/model/my_drone/takeout_payload/attach</attach_topic>"
            "<detach_topic>/model/my_drone/takeout_payload/detach</detach_topic>"
            "<output_topic>/model/my_drone/takeout_payload/state</output_topic>"
            "</plugin></model></sdf>",
        ),
        "payload_sdf": (
            "payload.sdf",
            "<sdf><model name='takeout_payload'><link name='payload_link'><inertial>"
            "<mass>0.04</mass></inertial></link></model></sdf>",
        ),
        "vehicle_metadata": (
            "vehicle.json",
            json.dumps(
                {
                    "schema_version": "dronedream.vehicle.v1",
                    "asset_id": "vehicle-current",
                    "name": "Current test vehicle",
                    "coordinate_frame": "base_link_frd",
                    "dry_mass_kg": 2.0,
                    "max_takeoff_mass_kg": 2.1,
                    "body_radius_m": 0.3,
                    "body_height_m": 0.4,
                    "max_speed_mps": 1.0,
                    "max_acceleration_mps2": 0.8,
                    "qualified_range_m": 100.0,
                    "reserve_battery_percent": 30.0,
                    "max_pickup_payload_kg": 0.05,
                    "sensors": ["depth"],
                }
            ),
        ),
        "controller_params": (
            "controller.json",
            json.dumps({"vel_limit": 1.0, "accel_limit": 0.8}),
        ),
    }
    files["vehicle_summary"] = (
        "summary.json",
        json.dumps(
            {
                "model_name": "my_drone",
                "source_model": "model://x500_depth",
                "dry_mass_kg": 2.0,
                "maximum_qualified_payload_kg": 0.05,
                "mission_payload": {
                    "model_name": "takeout_payload",
                    "link_name": "payload_link",
                    "mass_kg": 0.04,
                    "loaded_mass_kg": 2.04,
                    "center_above_model_root_m": 0.12,
                    "maximum_attachment_error_m": 0.02,
                    "attach_topic": "/model/my_drone/takeout_payload/attach",
                    "detach_topic": "/model/my_drone/takeout_payload/detach",
                    "state_topic": "/model/my_drone/takeout_payload/state",
                },
                "payload_files_sha256": {},
            }
        ),
    )
    declarations = {}
    for name, (relative, contents) in files.items():
        path = root / relative
        path.write_text(contents, encoding="utf-8")
        declarations[name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["payload_files_sha256"] = {
        "model.sdf": declarations["vehicle_sdf"]["sha256"],
        "takeout-payload.sdf": declarations["payload_sdf"]["sha256"],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    declarations["vehicle_summary"]["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest = root / "current-input.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.current-development-mission-input.v1",
                "status": "current-development",
                "asset_id": "vehicle-current",
                "capabilities": ["forward-depth", "payload-attachment"],
                "map_binding": map_binding
                or {
                    "asset_id": "map-current",
                    "content_sha256": "a" * 64,
                    "world_sdf_sha256": "b" * 64,
                    "semantic_sha256": "c" * 64,
                    "graph_sha256": "d" * 64,
                },
                "files": declarations,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_development_mission_is_one_atomic_hash_pinned_input(tmp_path: Path) -> None:
    manifest = _write_current_mission_manifest(tmp_path)

    resolved = resolve_development_mission_input(manifest)

    assert resolved.asset_id == "vehicle-current"
    assert resolved.map_asset_id == "map-current"
    assert resolved.vehicle_sdf == (tmp_path / "model.sdf").resolve()
    assert resolved.payload_mass_kg == pytest.approx(0.04)
    assert resolved.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()

    (tmp_path / "model.sdf").write_text("stale", encoding="utf-8")
    with pytest.raises(AssetRuntimeResolutionError, match="FILE_HASH_MISMATCH"):
        resolve_development_mission_input(manifest)


def test_retired_development_manifest_cannot_be_reused(tmp_path: Path) -> None:
    manifest = _write_current_mission_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "historical-evidence-only"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssetRuntimeResolutionError, match="MANIFEST_NOT_CURRENT"):
        resolve_development_mission_input(manifest)


def test_development_payload_must_fit_the_same_vehicle_manifest(tmp_path: Path) -> None:
    manifest = _write_current_mission_manifest(tmp_path)
    payload_path = tmp_path / "payload.sdf"
    payload_path.write_text(
        "<sdf><model name='payload'><link name='payload-link'><inertial>"
        "<mass>0.2</mass></inertial></link></model></sdf>",
        encoding="utf-8",
    )
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    contents["files"]["payload_sdf"]["sha256"] = hashlib.sha256(
        payload_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(contents), encoding="utf-8")

    with pytest.raises(
        AssetRuntimeResolutionError,
        match="DEVELOPMENT_PAYLOAD_EXCEEDS_VEHICLE_CAPACITY",
    ):
        resolve_development_mission_input(manifest)


def test_development_mission_rejects_a_different_qualified_map(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    semantic = tmp_path / "semantic.json"
    graph = tmp_path / "graph.json"
    world.write_text("<sdf/>", encoding="utf-8")
    semantic.write_text("{}", encoding="utf-8")
    graph.write_text("{}", encoding="utf-8")
    content_sha256 = "e" * 64
    map_binding = {
        "asset_id": "map-current",
        "content_sha256": content_sha256,
        "world_sdf_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
        "semantic_sha256": hashlib.sha256(semantic.read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
    }
    manifest = _write_current_mission_manifest(tmp_path, map_binding=map_binding)
    development_input = resolve_development_mission_input(manifest)
    selected_map = ResolvedMapAsset(
        asset_id="map-current",
        content_sha256=content_sha256,
        root=tmp_path,
        world_sdf=world,
        semantic=semantic,
        graph=graph,
        versioned=True,
    )

    assert_development_map_binding(development_input, selected_map)
    semantic.write_text('{"stale":true}', encoding="utf-8")
    with pytest.raises(AssetRuntimeResolutionError, match="MAP_BINDING_MISMATCH"):
        assert_development_map_binding(development_input, selected_map)


def test_development_input_rebinding_copies_vehicle_and_uses_current_map(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = _write_current_mission_manifest(source_root)
    map_root = tmp_path / "map"
    map_root.mkdir()
    world = map_root / "world.sdf"
    semantic = map_root / "semantic.json"
    graph = map_root / "graph.json"
    world.write_text("<sdf/>", encoding="utf-8")
    semantic.write_text('{"schema_version":"dronedream.map-semantic.v1"}', encoding="utf-8")
    graph.write_text('{"asset_id":"map-current"}', encoding="utf-8")
    selected_map = ResolvedMapAsset(
        asset_id="map-current",
        content_sha256="f" * 64,
        root=map_root,
        world_sdf=world,
        semantic=semantic,
        graph=graph,
        versioned=True,
    )

    result = rebind_development_mission_input_to_map(
        source_manifest,
        selected_map,
        tmp_path / "rebound",
    )
    rebound = resolve_development_mission_input(result)

    assert_development_map_binding(rebound, selected_map)
    assert rebound.asset_id == "vehicle-current"
    assert rebound.vehicle_sdf.read_bytes() == (source_root / "model.sdf").read_bytes()
    assert rebound.vehicle_metadata.read_bytes() == (source_root / "vehicle.json").read_bytes()
    with pytest.raises(FileExistsError):
        rebind_development_mission_input_to_map(
            source_manifest,
            selected_map,
            tmp_path / "rebound",
        )


def test_development_input_can_adopt_only_a_byte_identical_qualified_vehicle(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = _write_current_mission_manifest(source_root)
    qualified_root = tmp_path / "qualified"
    qualified_root.mkdir()
    qualified_sdf = qualified_root / "model.sdf"
    qualified_controller = qualified_root / "controller.json"
    qualified_metadata = qualified_root / "vehicle.json"
    qualified_summary = qualified_root / "summary.json"
    qualified_payload = qualified_root / "takeout-payload.sdf"
    qualified_sdf.write_bytes((source_root / "model.sdf").read_bytes())
    qualified_controller.write_bytes((source_root / "controller.json").read_bytes())
    qualified_summary.write_bytes((source_root / "summary.json").read_bytes())
    qualified_payload.write_bytes((source_root / "payload.sdf").read_bytes())
    metadata = json.loads((source_root / "vehicle.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "asset_id": "vehicle-promoted",
            "name": "Promoted vehicle",
            "sensors": [
                "imu",
                "magnetometer",
                "barometer",
                "gps",
                "odometry",
                "oakd-lite-depth",
            ],
        }
    )
    qualified_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    selected_vehicle = ResolvedVehicleAsset(
        asset_id="vehicle-promoted",
        content_sha256="e" * 64,
        root=qualified_root,
        vehicle_sdf=qualified_sdf,
        vehicle_metadata=qualified_metadata,
        vehicle_summary=qualified_summary,
        controller_params=qualified_controller,
        payload_sdf=qualified_payload,
        versioned=True,
    )

    result = rebind_development_mission_input_to_qualified_vehicle(
        source_manifest,
        selected_vehicle,
        tmp_path / "promoted-input",
    )
    rebound = resolve_development_mission_input(result)

    assert rebound.asset_id == "vehicle-promoted"
    assert rebound.map_asset_id == "map-current"
    assert rebound.vehicle_sdf.read_bytes() == qualified_sdf.read_bytes()
    assert rebound.vehicle_metadata.read_bytes() == qualified_metadata.read_bytes()
    assert rebound.vehicle_summary.read_bytes() == qualified_summary.read_bytes()
    assert rebound.payload_sdf.read_bytes() == qualified_payload.read_bytes()

    mismatched_sdf = qualified_root / "different-model.sdf"
    mismatched_sdf.write_text("<sdf/>", encoding="utf-8")
    incompatible = ResolvedVehicleAsset(
        **{
            **selected_vehicle.__dict__,
            "vehicle_sdf": mismatched_sdf,
        }
    )
    with pytest.raises(
        AssetRuntimeResolutionError,
        match="DEVELOPMENT_INPUT_VEHICLE_RUNTIME_MISMATCH",
    ):
        rebind_development_mission_input_to_qualified_vehicle(
            source_manifest,
            incompatible,
            tmp_path / "incompatible-input",
        )

    payload_contract_missing = ResolvedVehicleAsset(
        **{
            **selected_vehicle.__dict__,
            "payload_sdf": None,
        }
    )
    with pytest.raises(
        AssetRuntimeResolutionError,
        match="DEVELOPMENT_INPUT_VEHICLE_PAYLOAD_CONTRACT_MISSING",
    ):
        rebind_development_mission_input_to_qualified_vehicle(
            source_manifest,
            payload_contract_missing,
            tmp_path / "payload-contract-missing-input",
        )


def test_planning_acceptance_selects_exact_current_qualified_pair(tmp_path: Path) -> None:
    module = _load_planning_acceptance_module()
    resources = (
        Path(__file__).parents[1] / "app" / "desktop" / "src-tauri" / "resources" / "default-assets"
    )
    index = json.loads((resources / "index.json").read_text(encoding="utf-8"))
    expected = {entry["kind"]: entry for entry in index["qualified_pair"]["packages"]}
    store = AppStore(tmp_path / "store")
    try:
        AssetImportService(store).seed_bundled_sources(resources)
        map_record, vehicle_record, binding = module._current_bundled_pair(store, resources)
    finally:
        store.close()

    assert map_record["content_sha256"] == expected["map"]["content_sha256"]
    assert vehicle_record["content_sha256"] == expected["vehicle"]["content_sha256"]
    assert "asset-versions" in str(map_record["bundle_root"])
    assert "asset-versions" in str(vehicle_record["bundle_root"])
    assert binding["map_qualified_content_sha256"] == expected["map"]["content_sha256"]


def test_retired_development_asset_flags_are_removed() -> None:
    repo = Path(__file__).parents[1]
    sources = (
        repo / "scripts" / "run_pluginized_planning_acceptance.py",
        repo / "scripts" / "run_pluginized_runtime_acceptance.py",
    )
    retired_flags = (
        "--vehicle-sdf-override",
        "--vehicle-metadata-override",
        "--controller-params-override",
        "--development-vehicle-input",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert all(flag not in text for flag in retired_flags)
