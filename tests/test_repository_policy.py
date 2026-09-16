from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from dronedream_agent_core.contracts import PreparedMission
from scripts.audit_public_experiment_labels import find_public_experiment_labels


def test_public_repository_has_no_experiment_sequence_labels() -> None:
    repository = Path(__file__).resolve().parents[1]

    assert find_public_experiment_labels(repository) == []


def test_native_watchdog_and_ros_observer_share_every_runtime_phase() -> None:
    repository = Path(__file__).resolve().parents[1]
    native_source = (
        repository / "ros_ws/src/dronedream_agent_plugin_api/src/safe_hold_capability.cpp"
    ).read_text(encoding="utf-8")
    observer_source = (
        repository / "ros_ws/src/dronedream_agent_ros/dronedream_agent_ros/gazebo_pose_observer.py"
    ).read_text(encoding="utf-8")
    native_body = native_source.split("valid_runtime_phase", 1)[1].split(
        "CapabilityConfiguration", 1
    )[0]
    observer_body = observer_source.split("allowed = {", 1)[1].split("}", 1)[0]

    native_phases = set(re.findall(r'phase == "([A-Z_]+)"', native_body))
    observer_phases = set(re.findall(r'"([A-Z_]+)"', observer_body))

    assert native_phases == observer_phases
    assert {
        "PREFLIGHT",
        "TAKEOFF",
        "PERCEPTION_REFRESH_HOLD",
        "TRACKING_RECOVERY",
        "WAYPOINT_SETTLE",
        "CHECKPOINT",
        "LANDING",
        "LANDED",
    } <= native_phases


def test_native_watchdog_stops_after_physical_landing_even_when_mission_failed() -> None:
    repository = Path(__file__).resolve().parents[1]
    native_source = (
        repository / "ros_ws/src/dronedream_agent_plugin_api/src/capability_host.cpp"
    ).read_text(encoding="utf-8")
    lifecycle_callback = native_source.split("lifecycle_subscription_ =", 1)[1].split(
        "return plugins_.empty()", 1
    )[0]

    assert 'message->terminal_state != "ON_GROUND"' in lifecycle_callback
    assert "!message->landing_confirmed" in lifecycle_callback
    assert "!message->safe_to_stop_watchdog" in lifecycle_callback
    assert "message->executor_return_code != 0" not in lifecycle_callback


def test_current_asset_boundaries_cannot_restore_retired_implicit_inputs() -> None:
    """Keep obsolete asset-selection routes out of every active product boundary.

    The low-level flight process is allowed to receive already-resolved file paths.
    Selection itself must remain content-addressed (or use the one atomic,
    hash-pinned development manifest), so a later refactor cannot silently revive
    an old map, vehicle, controller, payload, or named destination.
    """

    repository = Path(__file__).resolve().parents[1]
    product_boundaries = (
        repository / "src/dronedream_agent_app/asset_runtime_resolver.py",
        repository / "src/dronedream_agent_app/mission_service.py",
        repository / "src/dronedream_agent_app/runtime_manager.py",
        repository / "scripts/run_pluginized_planning_acceptance.py",
        repository / "scripts/run_pluginized_runtime_acceptance.py",
    )
    retired_selection_tokens = (
        "resolve_legacy_map",
        "resolve_legacy_vehicle",
        ".get_asset(",
        "--vehicle-sdf-override",
        "--vehicle-metadata-override",
        "--controller-params-override",
        "--development-vehicle-input",
        "resolve_development_vehicle_input",
        "current-development-vehicle-input",
        "vehicle_input_binding",
        "app-store/assets",
    )
    for path in product_boundaries:
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in retired_selection_tokens), path

    active_entity_sources = (
        repository / "src/dronedream_agent_core/assets.py",
        repository / "src/dronedream_agent_core/navigation.py",
    )
    retired_entity_tokens = (
        "office-handoff-point",
        "load_school_map_catalog",
    )
    for path in active_entity_sources:
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in retired_entity_tokens), path

    map_contract_sources = (
        repository / "src/dronedream_agent_core/assets.py",
        repository / "src/dronedream_agent_core/runtime_bindings.py",
    )
    for path in map_contract_sources:
        source = path.read_text(encoding="utf-8")
        assert "dronedream.autonomy.school-map-semantic.v1" not in source, path
        assert 'semantic.get("simulation_bindings")' not in source, path


def test_default_assets_have_one_current_content_addressed_boundary() -> None:
    """Prevent the retired mutable ZIP repository from returning beside DDPKG."""

    repository = Path(__file__).resolve().parents[1]
    asset_root = repository / "app" / "desktop" / "src-tauri" / "resources" / "default-assets"
    index = json.loads((asset_root / "index.json").read_text(encoding="utf-8"))
    packages = index["qualified_pair"]["packages"]

    assert index["schema_version"] == "dronedream.bundled-assets.v2"
    assert "bundles" not in index
    assert {(entry["kind"], entry["file"]) for entry in packages} == {
        ("map", "school-map.ddpkg"),
        ("vehicle", "my-drone.ddpkg"),
    }
    assert sorted(path.name for path in asset_root.iterdir() if path.is_file()) == [
        "index.json",
        "my-drone.ddpkg",
        "school-map.ddpkg",
    ]

    for entry in packages:
        archive = asset_root / entry["file"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry["sha256"]
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            assert not any("legacy" in name.casefold() for name in names)
            assert not any(name.casefold().endswith(".zip") for name in names)
            assert not any(name.casefold().startswith("source/") for name in names)
            manifest = json.loads(bundle.read("manifest.json"))
            asset_ir = json.loads(bundle.read(manifest["asset_ir_path"]))
            assert asset_ir["source"]["source_format"] == "dronedream-native"
            assert manifest["content_sha256"] == entry["content_sha256"]

            if entry["kind"] == "map":
                semantic = json.loads(bundle.read("normalized/map/gazebo/semantic.json"))
                assert semantic["schema_version"] == "dronedream.map-semantic.v1"
                assert "simulation_bindings" not in semantic
                assert semantic["runtime_bindings"]
                assert semantic["entities"]
            else:
                vehicle = json.loads(bundle.read("normalized/vehicle/vehicle.json"))
                assert vehicle["collision_center_offset_model_m"] == {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.228,
                }
                vehicle_sdf = bundle.read("normalized/vehicle/gazebo/model.sdf").decode("utf-8")
                if "oakd-lite-depth" in vehicle["sensors"]:
                    assert "model://x500_depth" in vehicle_sdf

    active_sources = (
        repository / "src/dronedream_agent_app/server.py",
        repository / "src/dronedream_agent_app/storage.py",
        repository / "src/dronedream_agent_app/plugin_manager.py",
    )
    retired_tokens = (
        "seed_bundled_assets",
        "import_asset_bundle",
        '"maps": store.list_assets',
        '"vehicles": store.list_assets',
        "/v1/assets/import",
    )
    for path in active_sources:
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in retired_tokens), path

    assert not (repository / "src/dronedream_agent_plugins/asset_map_canonical.py").exists()
    assert not (repository / "src/dronedream_agent_plugins/asset_vehicle_canonical.py").exists()


def test_payload_runtime_cannot_restore_retired_map_side_vehicle_geometry() -> None:
    """Vehicle and payload geometry cannot be sourced from an older map."""

    repository = Path(__file__).resolve().parents[1]
    source = (repository / "src/dronedream_agent_core/gazebo_adapter.py").read_text(
        encoding="utf-8"
    )
    payload_resolver = source.split("def _payload_spawn_spec(", 1)[1].split(
        "def _prepare_rootfs(", 1
    )[0]

    assert 'semantic.get("simulation_bindings")' not in payload_resolver
    assert "center_above_px4_model_root_m" not in payload_resolver
    assert 'step.parameters.get("payload_mount_offset_model_m")' in payload_resolver

    runtime_body = source.split("def run_px4_gazebo_track(", 1)[1]
    assert 'semantic["vehicle_clearance"]' not in runtime_body
    assert "selected_vehicle.body_radius_m * 2.0" in runtime_body
    assert "selected_vehicle.body_height_m" in runtime_body

    track_source = (repository / "src/dronedream_agent_core/px4_track.py").read_text(
        encoding="utf-8"
    )
    assert "vehicle: VehicleAsset | None" not in track_source
    binding_source = (repository / "src/dronedream_agent_core/runtime_bindings.py").read_text(
        encoding="utf-8"
    )
    contract_source = (repository / "src/dronedream_agent_core/contracts.py").read_text(
        encoding="utf-8"
    )
    assert "vehicle_collision_center_offset" not in contract_source
    assert 'legacy.get("vehicle_collision_center_offset")' not in binding_source

    default_asset_builder = (repository / "scripts/build-default-assets.py").read_text(
        encoding="utf-8"
    )
    assert 'semantic.pop("simulation_bindings"' not in default_asset_builder
    assert "DEFAULT_MAP_CONTRACT_OBSOLETE" in default_asset_builder
    assert "DEFAULT_VEHICLE_DEPTH_RUNTIME_MISSING" in default_asset_builder


def test_runtime_cli_has_no_developer_machine_workspace_fallback() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "src/dronedream_agent_core/cli.py").read_text(encoding="utf-8")

    assert "/mnt/c/Users/" not in source
    assert "Documents/Codex" not in source
    assert "/opt/dronedream/source" not in source
    assert 'add_argument("--vehicle-diameter-m"' not in source
    assert 'add_argument("--vehicle-height-m"' not in source

    staging_source = (repository / "scripts/stage-development-runtime-resources.ps1").read_text(
        encoding="utf-8"
    )
    assert "Q:\\DroneDream-Workspace" not in staging_source


def test_acceptance_entrypoints_use_current_map_entity_identity() -> None:
    """Acceptance defaults cannot depend on a retired map-specific display alias."""

    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/run_pluginized_planning_acceptance.py",
        "scripts/run_plan_revision_acceptance.py",
        "examples/mission_request.office_takeout.json",
        "examples/mission_request.office_to_campus_gate.json",
    ):
        source = (repository / relative).read_text(encoding="utf-8")
        assert "办公室无人机起降坪" not in source, relative
        assert "office-launch-pad" in source, relative


def test_active_controller_contract_cannot_reintroduce_inert_gain_fields() -> None:
    """Only limits used by the PX4-facing runtime belong in the active contract."""

    repository = Path(__file__).resolve().parents[1]
    active_sources = (
        repository / "runtime/px4_offboard_track_executor.py",
        repository / "src/dronedream_agent_core/gazebo_adapter.py",
    )
    retired_fields = ("kp_xy", "ki_xy", "kd_xy", "disturbance_rejection")

    for path in active_sources:
        source = path.read_text(encoding="utf-8")
        assert all(field not in source for field in retired_fields), path


def test_active_local_control_cannot_restore_moving_goal_identity_fallback() -> None:
    repository = Path(__file__).resolve().parents[1]
    active_sources = (
        repository / "scripts/px4_checkpoint_executor.py",
        repository / "scripts/runtime_depth_safety_worker.py",
    )

    for path in active_sources:
        assert "current-control-setpoint" not in path.read_text(encoding="utf-8"), path


def test_current_prepared_mission_is_the_only_active_execution_default() -> None:
    repository = Path(__file__).resolve().parents[1]
    active_entrypoints = (
        repository / "src/dronedream_agent_core/execution.py",
        repository / "src/dronedream_agent_app/runtime_manager.py",
        repository / "scripts/run_pluginized_runtime_acceptance.py",
    )

    assert PreparedMission.model_fields["schema_version"].default == (
        "dronedream.prepared-mission.v4"
    )
    for path in active_entrypoints:
        source = path.read_text(encoding="utf-8")
        assert 'schema_version != "dronedream.prepared-mission.v4"' in source, path
        assert "PREPARED_MISSION_SCHEMA_OBSOLETE" in source, path


def test_shared_runtime_submodule_cannot_supply_agent_flight_code() -> None:
    """The shared repository is an installer base, not a second flight runtime."""

    repository = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (repository / "shared/runtime-base.contract.json").read_text(encoding="utf-8")
    )
    selected_paths = {item["path"] for item in contract["source_files"]}

    assert selected_paths
    assert all(
        path.startswith("desktop/src-tauri/src/") or path == "runtime/release-public-keys.json"
        for path in selected_paths
    )
    assert all("executor" not in path and "simulator" not in path for path in selected_paths)

    desktop_source = (repository / "app/desktop/src-tauri/src/lib.rs").read_text(encoding="utf-8")
    assert "px4_offboard_track_executor" not in desktop_source
    assert not (repository / "runtime/provenance.json").exists()

    for build_script in (
        repository / "scripts/build-autonomy-windows.ps1",
        repository / "scripts/stage-development-runtime-resources.ps1",
    ):
        source = build_script.read_text(encoding="utf-8")
        assert "write_runtime_provenance.py" in source
        assert 'Join-Path $runtimeSource "provenance.json"' not in source


def test_release_and_development_runtime_use_one_local_policy_stager() -> None:
    repository = Path(__file__).resolve().parents[1]
    release = (repository / "scripts/build-autonomy-windows.ps1").read_text(encoding="utf-8")
    development = (repository / "scripts/stage-development-runtime-resources.ps1").read_text(
        encoding="utf-8"
    )

    assert "stage_local_policy_runtime.py" in release
    assert "stage_local_policy_runtime.py" in development
    assert "LocalPolicyPackage" in release
    assert "LocalPolicySimulationAdmission" in release
    assert "Copy-Item -LiteralPath $packageSource" not in development
    assert "dronedream.local-policy-runtime-catalog.v1" not in release
    assert "dronedream.local-policy-runtime-catalog.v1" not in development


def test_forward_visual_runtime_requires_route_tangent_heading() -> None:
    """A body-fixed forward camera cannot authorize sideways or backward flight."""

    repository = Path(__file__).resolve().parents[1]
    adapter = (repository / "src/dronedream_agent_core/gazebo_adapter.py").read_text(
        encoding="utf-8"
    )
    runtime_body = adapter.split("def run_px4_gazebo_track(", 1)[1]
    assert (
        'local_navigation_visual_enabled and heading_policy != "route-tangent-relative"'
        in " ".join(runtime_body.split())
    )

    acceptance = (repository / "scripts/run_pluginized_runtime_acceptance.py").read_text(
        encoding="utf-8"
    )
    qualification = (repository / "scripts/run_school_map_depth_qualification.py").read_text(
        encoding="utf-8"
    )
    assert 'default="route-tangent-relative"' in acceptance
    assert 'default="route-tangent-relative"' in qualification


def test_active_mavsdk_entrypoints_use_explicit_udpin_connection() -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "runtime/px4_offboard_track_executor.py",
        "scripts/px4_checkpoint_executor.py",
    ):
        source = (repository / relative).read_text(encoding="utf-8")
        assert '"udpin://0.0.0.0:14540"' in source, relative
        assert '"udp://:14540"' not in source, relative


def test_default_asset_promotion_never_uses_pinned_environment_defaults() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "scripts/build-default-assets.py").read_text(encoding="utf-8")
    for argument in (
        "--gazebo-version",
        "--ros-distribution",
        "--px4-commit",
        "--runtime-manifest-sha256",
    ):
        assert f'parser.add_argument("{argument}", required=True)' in source


def test_default_vehicle_range_is_derived_from_the_verified_track() -> None:
    """A new map qualification cannot inherit the prior route's distance."""

    repository = Path(__file__).resolve().parents[1]
    source = (repository / "scripts/build-default-assets.py").read_text(encoding="utf-8")

    assert "def _verified_track_distance_m(" in source
    assert 'qualified_range_m = _verified_track_distance_m(run_dir)' in source
    assert '"qualified_range_m": 413.213' not in source

    depth_qualification = (
        repository / "scripts/run_school_map_depth_qualification.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("vehicle_metadata", type=Path)' in depth_qualification
    assert "vehicle = _load_vehicle(" in depth_qualification
    assert "qualified_range_m=413.213" not in depth_qualification
