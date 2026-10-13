from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_bundled_named_buildings_resolve_to_distinct_outdoor_handoffs(tmp_path: Path) -> None:
    from dronedream_agent_core.assets import load_map_catalog, resolve_map_entity
    from dronedream_agent_core.contracts import MapAsset

    package = Path(__file__).parents[1] / "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg"
    with zipfile.ZipFile(package) as bundle:
        graph = MapAsset.model_validate_json(bundle.read("normalized/map/navigation-graph.json"))
        semantic = tmp_path / "semantic.json"
        semantic.write_bytes(bundle.read("normalized/map/gazebo/semantic.json"))
    catalog = load_map_catalog(semantic, qualified_graph=graph)
    resolved = [resolve_map_entity(name, catalog, graph) for name in ("Chemicum", "Dynamicum", "Exactum", "Physicum")]
    assert len(set(resolved)) == 4
    assert resolve_map_entity("physicum", catalog, graph) == "physicum-delivery-zone"
    assert resolve_map_entity("物理楼", catalog, graph) == "physicum-delivery-zone"
    assert resolve_map_entity("计算机科学楼", catalog, graph) == "exactum-delivery-zone"
    # Added endpoints do not pretend to be flight-verified route edges.
    assert all(edge.from_node not in resolved[1:] and edge.to_node not in resolved[1:] for edge in graph.edges)


def _load_builder() -> ModuleType:
    source = Path(__file__).resolve().parents[1] / "scripts/build-default-assets.py"
    spec = importlib.util.spec_from_file_location("default_asset_builder", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("default asset builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qualified_range_comes_from_the_exact_verified_track(tmp_path: Path) -> None:
    builder = _load_builder()
    (tmp_path / "reference_track.json").write_text(
        json.dumps(
            {
                "points": [
                    {"x": 0.0, "y": 0.0, "z": 1.0},
                    {"x": 3.0, "y": 4.0, "z": 1.0},
                    {"x": 3.0, "y": 4.0, "z": 13.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert builder._verified_track_distance_m(tmp_path) == pytest.approx(17.0)


def test_qualified_range_rejects_non_finite_track_points(tmp_path: Path) -> None:
    builder = _load_builder()
    (tmp_path / "reference_track.json").write_text(
        '{"points":[{"x":0,"y":0,"z":0},{"x":NaN,"y":1,"z":1}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="REFERENCE_POINT_INVALID"):
        builder._verified_track_distance_m(tmp_path)


def _write_range_qualification_fixture(tmp_path: Path, builder: ModuleType) -> SimpleNamespace:
    source = tmp_path / "source"
    map_source = source / "map"
    vehicle_source = source / "vehicle"
    controller_params = source / "controller.json"
    for path, content in (
        (map_source / "world.physics.sdf", "world"),
        (map_source / "semantic.json", "semantic"),
        (vehicle_source / "model.sdf", "vehicle"),
        (controller_params, "controller"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    qualification_root = tmp_path / "range-run"
    simulation = qualification_root / "simulation"
    plan_root = qualification_root / "plan"
    simulation.mkdir(parents=True)
    plan_root.mkdir(parents=True)
    route = simulation / "mission-route.json"
    track = simulation / "reference_track.json"
    clearance = plan_root / "clearance.json"
    route.write_text('{"route":"corridor"}', encoding="utf-8")
    track.write_text(
        json.dumps({"points": [{"x": 0, "y": 0, "z": 1}, {"x": 3, "y": 4, "z": 1}]}),
        encoding="utf-8",
    )
    clearance.write_text(
        json.dumps({"accepted": True, "collision_count": 0, "minimum_clearance_m": 0.9}),
        encoding="utf-8",
    )
    gates = {
        "executor_completed": True,
        "landing_confirmed": True,
        "static_route_clearance_bound": True,
    }
    artifacts = {
        "world_sha256": builder._sha256(map_source / "world.physics.sdf"),
        "semantic_sha256": builder._sha256(map_source / "semantic.json"),
        "vehicle_sha256": builder._sha256(vehicle_source / "model.sdf"),
        "controller_params_sha256": builder._sha256(controller_params),
        "route_sha256": builder._sha256(route),
        "track_sha256": builder._sha256(track),
        "clearance_sha256": builder._sha256(clearance),
    }
    (simulation / "mission_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.generic-px4-gazebo-run.v1",
                "status": "verified",
                "gates": gates,
                "artifacts": artifacts,
                "measurements": {
                    "executor_return_code": 0,
                    "abort_reason": None,
                    "external_abort_request": None,
                    "landing_state": "ON_GROUND",
                    "pose_sample_count": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    (qualification_root / "qualification-summary.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.school-map-depth-qualification-summary.v1",
                "status": "verified",
                "gates": gates,
                "training_data_collection": False,
                "route_completion_mode": "return-to-launch",
                "operational_qualification_granted": True,
            }
        ),
        encoding="utf-8",
    )
    (plan_root / "qualification-plan.json").write_text(
        json.dumps(
            {
                "mode": "full-round-trip",
                "operational_qualification_requested": True,
                "route_length_m": 5.0,
                "required_clearance_m": 0.8,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        qualification_root=qualification_root,
        map_source=map_source,
        vehicle_source=vehicle_source,
        controller_params=controller_params,
    )


def test_range_qualification_extends_only_from_bound_operational_round_trip(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    fixture = _write_range_qualification_fixture(tmp_path, builder)

    receipt, distance = builder._range_qualification_receipt(
        fixture.qualification_root,
        map_source=fixture.map_source,
        vehicle_source=fixture.vehicle_source,
        controller_params=fixture.controller_params,
    )

    assert distance == pytest.approx(5.0)
    assert receipt["qualified_range_m"] == pytest.approx(5.0)
    assert receipt["minimum_clearance_m"] == pytest.approx(0.9)


def test_range_qualification_rejects_a_vehicle_hash_mismatch(tmp_path: Path) -> None:
    builder = _load_builder()
    fixture = _write_range_qualification_fixture(tmp_path, builder)
    (fixture.vehicle_source / "model.sdf").write_text("different vehicle", encoding="utf-8")

    with pytest.raises(ValueError, match="INPUT_BINDING_INVALID"):
        builder._range_qualification_receipt(
            fixture.qualification_root,
            map_source=fixture.map_source,
            vehicle_source=fixture.vehicle_source,
            controller_params=fixture.controller_params,
        )


def test_pickup_index_comes_from_matching_checkpoint_and_payload_evidence(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    (tmp_path / "reference_track.json").write_text(
        json.dumps(
            {
                "points": [
                    {"x": 0.0, "y": 0.0, "z": 1.0, "phase": "launch"},
                    {"x": 1.0, "y": 0.0, "z": 5.0, "phase": "transit"},
                    {"x": 2.0, "y": 0.0, "z": 2.0, "phase": "pickup"},
                    {"x": 1.0, "y": 0.0, "z": 5.0, "phase": "return"},
                    {"x": 0.0, "y": 0.0, "z": 1.0, "phase": "land"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "payload_spawn.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "track_point_index": 2,
            }
        ),
        encoding="utf-8",
    )

    assert (
        builder._verified_pickup_track_point_index(
            run_dir=tmp_path,
            point_count=5,
        )
        == 2
    )


def test_pickup_index_rejects_ambiguous_runtime_evidence(tmp_path: Path) -> None:
    builder = _load_builder()
    (tmp_path / "reference_track.json").write_text(
        json.dumps(
            {
                "points": [
                    {"x": 0.0, "y": 0.0, "z": 1.0, "phase": "launch"},
                    {"x": 1.0, "y": 0.0, "z": 5.0, "phase": "transit"},
                    {"x": 2.0, "y": 0.0, "z": 2.0, "phase": "pickup"},
                    {"x": 0.0, "y": 0.0, "z": 1.0, "phase": "pickup"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "payload_spawn.json").write_text(
        json.dumps(
            {
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PAYLOAD_CHECKPOINT_BINDING_INVALID"):
        builder._verified_pickup_track_point_index(
            run_dir=tmp_path,
            point_count=5,
        )


def test_qualified_runtime_evidence_uses_current_gate_contract(tmp_path: Path) -> None:
    builder = _load_builder()
    gates = {gate: True for gate in builder.CURRENT_RUNTIME_GATES}
    mission_evidence = tmp_path / "mission_evidence.json"
    mission_evidence.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.school-map-px4-mission-evidence.v1",
                "status": "verified",
                "gates": gates,
                "pose_sample_count": 10,
                "measurements": {"pose_sample_count": 10, "landing_state": "ON_GROUND"},
                "process_failure": None,
                "artifacts": {"px4_ulogs": ["flight.ulg"]},
            }
        ),
        encoding="utf-8",
    )
    qualification_root = tmp_path / "qualification"
    inputs = SimpleNamespace(
        world_sdf=Path("map/world.sdf"),
        semantic=Path("map/semantic.json"),
        vehicle_sdf=Path("vehicle/model.sdf"),
        controller_params=Path("vehicle/controller.json"),
        world_name="world",
        vehicle_name="vehicle",
    )
    for relative in (
        inputs.world_sdf,
        inputs.semantic,
        inputs.vehicle_sdf,
        inputs.controller_params,
    ):
        target = qualification_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(relative), encoding="utf-8")
    generic_gates = [
        "executor_completed",
        "offboard_timing_complete",
        "runtime_pose_samples_present",
        "ros_observations_present",
        "goal_observed",
        "landing_confirmed",
        "native_terminal_lifecycle_published",
        "no_live_abort",
        "px4_ulog_present",
        "static_route_clearance_bound",
    ]
    plan = SimpleNamespace(
        required_runtime_gates=generic_gates,
        inputs=inputs,
        route_sha256="route",
        track_sha256="track",
        clearance_sha256="clearance",
    )

    result = builder._qualified_runtime_evidence(
        plan=plan,
        qualification_root=qualification_root,
        mission_evidence_path=mission_evidence,
    )

    assert all(result["gates"].values())
    assert result["measurements"]["pose_sample_count"] == 10


def test_superseded_pair_records_preserve_the_complete_retirement_chain(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    historical = {
        "qualification_id": "asset-qualification-historical",
        "map_content_sha256": "historical-map",
        "map_source_content_sha256": "historical-map-source",
        "vehicle_content_sha256": "historical-vehicle",
        "vehicle_source_content_sha256": "historical-vehicle-source",
    }
    current = {
        "qualification_id": "asset-qualification-current",
        "map_content_sha256": "current-map",
        "map_source_content_sha256": "current-map-source",
        "vehicle_content_sha256": "current-vehicle",
        "vehicle_source_content_sha256": "current-vehicle-source",
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "qualified_pair": {
                    "qualification_id": current["qualification_id"],
                    "packages": [
                        {
                            "kind": "map",
                            "content_sha256": current["map_content_sha256"],
                            "source_content_sha256": current["map_source_content_sha256"],
                        },
                        {
                            "kind": "vehicle",
                            "content_sha256": current["vehicle_content_sha256"],
                            "source_content_sha256": current["vehicle_source_content_sha256"],
                        },
                    ],
                },
                "superseded_qualified_pairs": [historical, historical],
            }
        ),
        encoding="utf-8",
    )

    records = builder._superseded_pair_records(index_path)

    assert records == [historical, current]


def test_superseded_pair_records_reject_malformed_history(tmp_path: Path) -> None:
    builder = _load_builder()
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "qualified_pair": {
                    "qualification_id": "asset-qualification-current",
                    "packages": [
                        {
                            "kind": "map",
                            "content_sha256": "current-map",
                            "source_content_sha256": "current-map-source",
                        },
                        {
                            "kind": "vehicle",
                            "content_sha256": "current-vehicle",
                            "source_content_sha256": "current-vehicle-source",
                        },
                    ],
                },
                "superseded_qualified_pairs": [{"qualification_id": "incomplete"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SUPERSEDED_DEFAULT_ASSET_INDEX_INVALID"):
        builder._superseded_pair_records(index_path)
