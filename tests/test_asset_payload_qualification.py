"""Qualification consumes production contracts and actual device facts, not labels."""

import asyncio
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core.asset_payload_qualification import (
    NATIVE_GATES,
    PAYLOAD_ACTIONS,
    build_asset_payload_course,
    decide_asset_checkpoint,
    file_sha256,
    separate_release_from_landing,
    verify_asset_payload_run,
)
from dronedream_agent_core.contracts import (
    GraphRoute,
    RuntimeActionExecutionReceipt,
    RuntimeCheckpointRequest,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.gazebo_adapter import _validate_execution_contract_binding
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.simulation_payload_runtime import LIBRARY
from scripts import px4_checkpoint_executor as executor
from scripts.run_school_map_depth_qualification import _qualification_route, _write_json


@pytest.fixture
def course(tmp_path):
    outbound = GraphRoute(
        start_node="school",
        goal_node="a-different-building",
        node_ids=["school", "corner", "a-different-building"],
        edge_ids=["one", "two"],
        positions_m=[
            Vector3(x=0.0, y=0.0, z=1.5),
            Vector3(x=20.0, y=0.0, z=2.0),
            Vector3(x=20.0, y=20.0, z=2.0),
        ],
        route_length_m=40.00624902374255,
        all_edges_flight_verified=False,
    )
    route, graph = _qualification_route(outbound, point_count=None, speed_limit_mps=0.8)
    vehicle = VehicleAsset(
        asset_id="fixture",
        name="fixture",
        dry_mass_kg=2.0,
        max_takeoff_mass_kg=2.2,
        body_radius_m=0.38,
        body_height_m=0.43,
        max_speed_mps=1.2,
        max_acceleration_mps2=0.8,
        qualified_range_m=40.0,
        reserve_battery_percent=30.0,
        max_pickup_payload_kg=0.1,
        sensors=["imu", "odometry"],
    )
    sdf = tmp_path / "vehicle.sdf"
    sdf.write_text(
        """<sdf version="1.10"><model name="drone"><plugin name="detachable_joint">
        <parent_link>base_link</parent_link><child_model>payload</child_model>
        <child_link>payload_link</child_link><attach_topic>/payload/attach</attach_topic>
        <detach_topic>/payload/detach</detach_topic><output_topic>/payload/state</output_topic>
        </plugin></model></sdf>""",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "summary.json",
        {
            "model_name": "drone",
            "mission_payload": {
                "model_name": "payload",
                "center_above_model_root_m": 0.12,
                "maximum_attachment_error_m": 0.02,
            },
        },
    )
    (tmp_path / "takeout-payload.sdf").write_text(
        """<sdf version="1.10"><model name="payload">
        <link name="payload_link"><inertial><mass>0.1</mass><inertia><ixx>0.01</ixx>
        <iyy>0.02</iyy><izz>0.03</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia></inertial></link></model></sdf>""",
        encoding="utf-8",
    )
    route, graph = separate_release_from_landing(route, graph, vehicle)
    return route, graph, vehicle, sdf, "a" * 64


def test_course_uses_actual_turn_and_return_with_production_drivers(course):
    route, _, vehicle, _, _ = course
    result = build_asset_payload_course(*course)
    actions, checkpoints = result["actions"], result["checkpoints"]
    assert tuple(step.action for step in actions.steps) == PAYLOAD_ACTIONS
    assert [cp.track_point_index for cp in checkpoints.checkpoints] == [2, 4, 5]
    assert route.positions_m[-2].x > 1.99
    assert route.positions_m[-2].y == 0.0
    assert route.route_length_m == pytest.approx(80.0124980474851)
    assert [step.checkpoint_id for step in actions.steps] == ["checkpoint-001"] * 4 + [
        "checkpoint-002"
    ] * 4
    assert actions.steps[7].depends_on == ["release-area"]
    assert all(not step.action.startswith("training.") for step in actions.steps)
    assert vehicle.qualified_range_m == 40.0  # compiling a longer trial grants no range
    _validate_execution_contract_binding(
        checkpoints, actions, route, result["mission"].contract_id, False
    )


@pytest.fixture
def run_evidence(course, tmp_path):
    """Unit fixtures exercise verifier decisions; they are not flight evidence."""
    result = build_asset_payload_course(*course)
    route, _, _, sdf, _ = course
    root = tmp_path / "unit-run"
    actions, checkpoints = result["actions"], result["checkpoints"]
    for name, value in {
        "mission-route.json": route.model_dump(mode="json"),
        "reference_track.json": {"points": [{"x": 0, "y": 0, "z": 0}, {"x": 3, "y": 4, "z": 0}]},
        "clearance.json": {"accepted": True},
        "runtime-actions.json": actions.model_dump(mode="json"),
        "runtime-checkpoints.json": checkpoints.model_dump(mode="json"),
    }.items():
        _write_json(root / name, value)
    run_files = {
        name: file_sha256(root / name)
        for name in (
            "mission-route.json",
            "reference_track.json",
            "clearance.json",
            "runtime-actions.json",
            "runtime-checkpoints.json",
        )
    }
    artifact_hashes = dict(
        world_sha256="b" * 64,
        semantic_sha256="a" * 64,
        vehicle_sha256="c" * 64,
        controller_params_sha256="d" * 64,
    )
    _write_json(
        root / "asset-qualification-inputs.json",
        {
            "schema_version": "dronedream.asset-qualification-inputs.v1",
            "run_files": run_files,
            "runtime_inputs": artifact_hashes,
            "loaded_thrust_to_weight": 1.7,
            "minimum_thrust_to_weight": 1.6,
            "reserve_battery_percent": 30.0,
        },
    )
    for checkpoint in checkpoints.checkpoints:
        request = RuntimeCheckpointRequest(
            contract_id=checkpoints.contract_id,
            checkpoint=checkpoint,
            observed_position_ned_m=Vector3(x=1.0, y=2.0, z=-1.0),
            commanded_position_ned_m=Vector3(x=1.0, y=2.0, z=-1.0),
            observed_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
            position_error_m=0.0,
            speed_mps=0.0,
            battery_percent=80.0,
            deterministic_gates={
                "position_error_within_0_75_m": True,
                "speed_within_0_50_mps": True,
                "battery_above_10_percent": True,
                "telemetry_finite": True,
            },
        )
        _write_json(
            root / f"checkpoints/{checkpoint.checkpoint_id}.request.json",
            request.model_dump(mode="json"),
        )
        _write_json(
            root / f"checkpoints/{checkpoint.checkpoint_id}.asset-decision.json",
            decide_asset_checkpoint(request, checkpoints).model_dump(mode="json"),
        )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for i, step in enumerate(actions.steps):
        output = {
            "confirmed": True,
            "detached": False,
            "transport": "gazebo-transport",
            "operation": step.parameters["operation"],
            "position_error_m": 0.01,
            "speed_mps": 0.01,
            "state_readback_source": "gazebo-detachable-joint-event",
            "custody_state_accepted": True,
            "payload_physics_binding_confirmed": True,
            "loaded_hover_stable": True,
            "return_authorized": True,
            "minimum_predicted_clearance_m": 1.0,
            **{
                key: step.parameters[key]
                for key in ("payload_mass_kg", "payload_inertia_kg_m2", "payload_sdf_sha256")
                if key in step.parameters
            },
        }
        if i == 1:
            output["payload_mount_alignment"] = {
                "binding_sha256": step.parameters["payload_mount_binding_sha256"],
                "attachment_pose_readback": {
                    "payload_position_world_enu_m": [1.0, 2.0, 3.0],
                    "expected_position_world_enu_m": [1.0, 2.0, 3.0],
                    "alignment_error_m": 0.0,
                    "accepted": True,
                },
            }
        if i == 7:
            output["detached"] = True
        receipt = RuntimeActionExecutionReceipt(
            execution_contract_sha256=sha256_json(actions),
            step_sha256=sha256_json(step),
            **{
                key: getattr(step, key)
                for key in ("step_id", "task_id", "action", "adapter_id", "runtime_executor")
            },
            status="accepted",
            attempts=1,
            started_at=start + timedelta(seconds=i * 2),
            completed_at=start + timedelta(seconds=i * 2 + 1),
            output=output,
            observed_success_evidence=step.required_success_evidence,
            deterministic_gates={"driver_confirmed": True},
        )
        _write_json(
            root / f"runtime-actions/receipts/{step.step_id}.receipt.json",
            receipt.model_dump(mode="json"),
        )
    placement = root / "payload-placement"
    placement.mkdir()
    (placement / "payload.sdf").write_bytes((sdf.parent / "takeout-payload.sdf").read_bytes())
    (placement / LIBRARY).write_bytes(b"unit-test-library")
    deployment = {
        "original_sha256": actions.steps[2].parameters["payload_sdf_sha256"],
        "staged_sha256": file_sha256(placement / "payload.sdf"),
        "library_sha256": file_sha256(placement / LIBRARY),
    }
    _write_json(placement / "deployment.json", deployment)
    _write_json(
        root / "payload_spawn.json",
        {
            "accepted": True,
            "checkpoint_id": "checkpoint-001",
            "track_point_index": 2,
            "placement_runtime": deployment,
            "sdf_sha256": deployment["staged_sha256"],
        },
    )
    _write_json(root / "offboard_timing.json", {"status": "complete"})
    _write_json(
        root / "mission_evidence.json",
        {
            "schema_version": "dronedream.generic-px4-gazebo-run.v1",
            "status": "verified",
            "gates": dict.fromkeys(NATIVE_GATES, True),
            "measurements": {"pose_sample_count": 500},
            "artifacts": {
                **artifact_hashes,
                "route_sha256": run_files["mission-route.json"],
                "track_sha256": run_files["reference_track.json"],
                "clearance_sha256": run_files["clearance.json"],
            },
        },
    )
    return root


def test_builder_consumes_exact_native_evidence_and_copies_real_files(run_evidence, tmp_path):
    verified = verify_asset_payload_run(run_evidence)
    assert verified["status"] == "verified"
    assert verified["model_qualification_granted"] is False
    spec = importlib.util.spec_from_file_location(
        "builder", Path(__file__).parents[1] / "scripts/build-default-assets.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder._qualification_receipt(run_evidence)["simulation_execution_ready"] is True
    builder._copy_qualification_evidence(run_evidence, tmp_path / "bundle")
    copied = tmp_path / "bundle/qualification"
    assert json.loads((copied / "asset-qualification-evidence.json").read_text()) == verified
    for name, digest in verified["verified_source_files_sha256"].items():
        assert file_sha256(copied / name) == digest
    for name, digest in verified["verified_native_runtime_files_sha256"].items():
        assert file_sha256(run_evidence / name) == digest
        assert not (copied / name).exists()
    source = json.loads((run_evidence / "mission_evidence.json").read_text())
    assert (
        builder._qualified_runtime_evidence(
            plan=None,
            qualification_root=tmp_path,
            mission_evidence_path=run_evidence / "mission_evidence.json",
        )["artifacts"]
        == source["artifacts"]
    )


def test_promoted_vehicle_retains_the_flown_source_bytes(run_evidence, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "builder", Path(__file__).parents[1] / "scripts/build-default-assets.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = tmp_path / "vehicle-source"
    source.mkdir()
    (source / "model.sdf").write_text("<sdf>model://x500_depth</sdf>")
    (source / "model.config").write_text("<model><name>Original model</name></model>")
    (source / "takeout-payload.sdf").write_text("<sdf>original payload</sdf>")
    (source / "summary.json").write_text(
        json.dumps(
            {
                "source_model": "model://x500_depth",
                "dry_mass_kg": 2.1,
                "maximum_qualified_payload_kg": 0.1,
                "mission_payload": {"mass_kg": 0.04, "loaded_thrust_to_weight": 1.7},
                "gazebo_runtime_verified": False,
            }
        )
    )
    controller = tmp_path / "controller.json"
    controller.write_text("{}")
    staged, bundle = tmp_path / "staged", tmp_path / "vehicle-bundle"
    builder._stage_current_vehicle(source, controller, staged)
    builder._build_vehicle_bundle(run_evidence, bundle, staged, controller)
    for original in source.iterdir():
        assert (staged / original.name).read_bytes() == original.read_bytes()
        assert (bundle / "gazebo" / original.name).read_bytes() == original.read_bytes()
    metadata = json.loads((bundle / "vehicle.json").read_text())
    assert metadata["qualified_range_m"] == 5.0
    assert metadata["max_pickup_payload_kg"] == 0.04


@pytest.mark.parametrize(
    "damage",
    [
        "missing-action",
        "old-route",
        "landing",
        "teacher",
        "native-depth",
        "detached-return",
        "false-release",
        "unstable-return",
        "false-checkpoint",
    ],
)
def test_qualification_rejects_incomplete_or_cross_run_evidence(run_evidence, damage):
    if damage == "missing-action":
        (run_evidence / "runtime-actions/receipts/action-008.receipt.json").unlink()
    elif damage == "old-route":
        (run_evidence / "mission-route.json").write_text("{}")
    else:
        name, field, value = {
            "landing": ("mission_evidence.json", ("gates", "landing_confirmed"), False),
            "teacher": (
                "mission_evidence.json",
                ("gates", "offline_training_authority_absent"),
                False,
            ),
            "native-depth": (
                "mission_evidence.json",
                ("gates", "live_depth_perception_healthy"),
                False,
            ),
            "detached-return": (
                "runtime-actions/receipts/action-005.receipt.json",
                ("output", "detached"),
                True,
            ),
            "false-release": (
                "runtime-actions/receipts/action-008.receipt.json",
                ("output", "detached"),
                False,
            ),
            "unstable-return": (
                "runtime-actions/receipts/action-006.receipt.json",
                ("output", "loaded_hover_stable"),
                False,
            ),
            "false-checkpoint": (
                "checkpoints/checkpoint-002.request.json",
                ("observed_position_ned_m", "x"),
                5.0,
            ),
        }[damage]
        path = run_evidence / name
        value_dict = json.loads(path.read_text())
        value_dict[field[0]][field[1]] = value
        _write_json(path, value_dict)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_asset_payload_run(run_evidence)


def test_asset_responder_requires_independent_safety_before_touching_flight():
    args = SimpleNamespace(
        asset_qualification_checkpoints=True,
        independent_route_control=False,
        require_model_control_authority=False,
        local_safety_required=True,
        checkpoint_contract="a",
        runtime_action_contract="b",
    )
    with pytest.raises(ValueError, match="asset qualification requires"):
        asyncio.run(executor._run(args, None))
