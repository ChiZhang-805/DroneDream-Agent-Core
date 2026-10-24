from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


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


def test_independent_range_evidence_is_bound_to_airframe_and_controller(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    vehicle = tmp_path / "vehicle"
    vehicle.mkdir()
    (vehicle / "model.sdf").write_text("exact-airframe", encoding="utf-8")
    controller = tmp_path / "controller.json"
    controller.write_text('{"vel_limit":6}', encoding="utf-8")
    receipt = tmp_path / "capability.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.vehicle-capability-evidence.v1",
                "status": "verified",
                "vehicle_sdf_sha256": builder._sha256(vehicle / "model.sdf"),
                "controller_params_sha256": builder._sha256(controller),
                "qualified_range_m": 594.8,
            }
        ),
        encoding="utf-8",
    )

    assert builder._independently_qualified_range_m(
        receipt, vehicle_source=vehicle, controller_params=controller
    ) == pytest.approx(594.8)
    controller.write_text('{"vel_limit":1.2}', encoding="utf-8")
    with pytest.raises(ValueError, match="CONTROLLER_PARAMS_SHA256_MISMATCH"):
        builder._independently_qualified_range_m(
            receipt, vehicle_source=vehicle, controller_params=controller
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
