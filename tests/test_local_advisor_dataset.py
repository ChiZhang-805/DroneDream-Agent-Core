from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_advisor_training import LocalAdvisorTrainingSample
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from scripts.build_local_advisor_dataset import (
    _payload_targets,
    _payload_transition_samples_from_multimodal_dataset,
    _validate_payload_collection_source_diversity,
    _verified_payload_collection_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_payload_collection_source_is_hash_bound_and_non_qualifying(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    simulation = run_root / "simulation"
    simulation.mkdir(parents=True)
    workflow_path = simulation / "workflow-result.json"
    workflow_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "runtime_evidence": {
                    "gates": {"development_payload_collection_absent": False}
                },
            }
        ),
        encoding="utf-8",
    )
    snapshots_path = simulation / "model-navigation-snapshots.jsonl"
    cycles_path = simulation / "model-navigation-cycles.jsonl"
    calls_path = simulation / "model-navigation-model-calls.jsonl"
    for path, record in (
        (snapshots_path, {"snapshot": "bound"}),
        (cycles_path, {"cycle": "bound"}),
        (calls_path, {"call": "bound"}),
    ):
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    multimodal_records_path = simulation / "multimodal-dataset" / "records.jsonl"
    multimodal_records_path.parent.mkdir()
    multimodal_records_path.write_text(json.dumps({"sensor": "bound"}) + "\n", encoding="utf-8")
    summary_path = run_root / "runtime-acceptance-summary.json"
    summary = {
        "schema_version": "dronedream.pluginized-runtime-acceptance.v1",
        "status": "verified-development-payload-collection",
        "flight_qualification_granted": False,
        "workflow_sha256": _sha256(workflow_path),
        "multimodal_dataset": {
            "summary": {
                "record_count": 1,
                "records_sha256": _sha256(multimodal_records_path),
            }
        },
        "development_payload_collection": {
            "status": "verified-development-data",
            "development_only": True,
            "flight_qualification_granted": False,
            "model_navigation_snapshots_sha256": _sha256(snapshots_path),
            "model_navigation_cycles_sha256": _sha256(cycles_path),
            "model_navigation_model_calls_sha256": _sha256(calls_path),
            "bounded_motion_cycle_count": 50,
            "stale_dynamics_motion_cycle_count": 0,
            "oversized_motion_cycle_count": 0,
            "maximum_controller_step_scale": 0.2,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    receipt = _verified_payload_collection_source(simulation)

    assert receipt["verification"] == "verified-development-payload-collection"
    assert receipt["flight_qualification_granted"] is False
    assert receipt["bounded_payload_motion_cycle_count"] == 50
    assert receipt["multimodal_dataset_records_sha256"] == _sha256(
        multimodal_records_path
    )

    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["runtime_evidence"] = {
        "gates": {
            "development_payload_collection_absent": False,
            "model_navigation_visual_frame_recorded": False,
            "semantic_supervision_recorded_for_every_sample": False,
        },
        "measurements": {
            "model_navigation": {"visual_enabled": False},
            "multimodal_dataset": {"enabled": True, "semantic_topic": None},
        },
    }
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    summary["workflow_sha256"] = _sha256(workflow_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt = _verified_payload_collection_source(simulation)
    assert receipt["verification"] == "verified-development-payload-collection"

    summary["flight_qualification_granted"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="PAYLOAD_COLLECTION_EVIDENCE_INVALID"):
        _verified_payload_collection_source(simulation)

    summary["flight_qualification_granted"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    calls_path.write_text(json.dumps({"call": "tampered"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PAYLOAD_COLLECTION_EVIDENCE_INVALID"):
        _verified_payload_collection_source(simulation)


def test_payload_collection_requires_two_substantial_missions_per_split() -> None:
    first = {"mission_evidence_sha256": "a" * 64, "sample_count": 50}
    second = {"mission_evidence_sha256": "b" * 64, "sample_count": 50}

    with pytest.raises(RuntimeError, match="FEWER_THAN_TWO_MISSIONS"):
        _validate_payload_collection_source_diversity(
            [first],
            split="training",
        )

    _validate_payload_collection_source_diversity(
        [first, second],
        split="training",
    )

    second["sample_count"] = 49
    with pytest.raises(RuntimeError, match="FEWER_THAN_FIFTY_SAMPLES"):
        _validate_payload_collection_source_diversity(
            [first, second],
            split="validation",
        )


def _snapshot(
    index: int,
    *,
    payload_evidence: bool,
    multimodal_evidence: bool = False,
) -> dict[str, object]:
    payload = (
        {
            "state": "attached" if index == 0 else "loaded-stable",
            "observed_payload_mass_kg": 0.02,
            "maximum_payload_kg": 0.05,
            "within_declared_payload_limit": True,
            "accepted_transition_steps": ["attach", "confirm"],
            "dynamics": {
                "available": True,
                "ready": True,
                "telemetry_schema_version": "dronedream.px4-dynamics-telemetry.v1",
                "telemetry_payload_sha256": "a" * 64,
                "maximum_sample_age_seconds": 3.0,
                "source_sample_age_seconds": {
                    "imu": 0.01,
                    "attitude": 0.01,
                    "battery": 0.01,
                    "actuator_output": 0.01,
                },
                "issue_codes": [],
                "restart_counts": {},
                "acceleration_forward_m_s2": 0.0,
                "acceleration_right_m_s2": 0.0,
                "acceleration_down_m_s2": -9.80665,
                "angular_velocity_forward_rad_s": 0.0,
                "angular_velocity_right_rad_s": 0.0,
                "angular_velocity_down_rad_s": 0.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "current_battery_a": 4.0,
                "voltage_v": 16.0,
                "actuator_normalization_ready": True,
                "actuator_normalization_kind": "configured-absolute-maximum",
                "actuator_normalization_absolute_maximum": 1000.0,
                "actuator_mean": 0.5,
                "actuator_max_abs": 1.0 if index == 2 else 0.6,
            },
        }
        if payload_evidence
        else {
            "state": "no-runtime-payload-evidence",
            "observed_payload_mass_kg": None,
            "maximum_payload_kg": 0.05,
            "within_declared_payload_limit": None,
            "accepted_transition_steps": [],
        }
    )
    snapshot: dict[str, object] = {
        "current_position_m": {"x": index * 0.1, "y": 0.0, "z": 1.0},
        "current_velocity_mps": {
            "x": 0.05 if index % 2 == 0 else 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "goal_position_m": {"x": 2.0, "y": 0.0, "z": 1.0},
        "goal_distance_m": max(0.0, 2.0 - index * 0.1),
        "local_radius_m": 8.0,
        "vehicle_envelope_m": {
            "candidate_speed": 0.5,
            "body_radius": 0.2,
            "body_height": 0.1,
            "required_local_clearance": 0.3,
        },
        "perception_health": {
            "stream_healthy": index % 5 != 0,
            "stream_age_seconds": 0.0 if index % 5 else 0.4,
            "localization_covariance_m2": 0.01,
            "issue_codes": [] if index % 5 else ["PERCEPTION_STREAM_STALE"],
        },
        "egocentric_sectors": [],
        "authorized_candidate_paths": [],
        "strategic_context": {
            "task": {
                "phase": "HOVER",
                "control_profile": "precision",
                "decision_trigger": "periodic",
            },
            "payload": payload,
        },
    }
    if multimodal_evidence:
        ready = index % 3 != 0
        snapshot["multimodal_sensor_snapshot"] = {
            "ready_for_motion": ready,
            "issue_codes": [] if ready else ["forward-rgb:SENSOR_SAMPLE_STALE"],
            "statuses": [
                {
                    "sensor_id": "forward-rgb",
                    "modality": "rgb-camera",
                    "required_for_motion": True,
                    "latest_sequence": index + 1,
                    "sample_age_seconds": 0.02 if ready else 0.8,
                    "quality": 0.9,
                    "coverage": 1.0,
                    "health": "healthy" if ready else "degraded",
                    "issue_codes": [] if ready else ["SENSOR_SAMPLE_STALE"],
                }
            ],
        }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    return snapshot


def test_payload_transition_samples_use_hash_bound_sensor_rate_records(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        0,
        payload_evidence=True,
        multimodal_evidence=True,
    )
    state = {
        "current_position_m": snapshot["current_position_m"],
        "current_velocity_mps": snapshot["current_velocity_mps"],
        "navigation_goal_m": snapshot["goal_position_m"],
        "control_profile": "precision",
        "payload": snapshot["strategic_context"]["payload"],
    }
    record = {
        "recorded_at_unix_ms": 1_000,
        "state": state,
        "sensor_snapshot": snapshot["multimodal_sensor_snapshot"],
    }
    record["record_sha256"] = sha256_json(record)
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    samples = _payload_transition_samples_from_multimodal_dataset(
        snapshot_timeline=[(1_000, snapshot)],
        records_path=records_path,
        source_identity="a" * 64,
    )

    assert len(samples) == 1
    sample = next(iter(samples.values()))
    assert sample.role == "payload-dynamics-adapter"
    assert sample.risk_target == 0.8
    assert sample.controller_step_scale_target == 0.4

    record["state"]["current_position_m"]["x"] = 99.0
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MULTIMODAL_RECORD_HASH_MISMATCH"):
        _payload_transition_samples_from_multimodal_dataset(
            snapshot_timeline=[(1_000, snapshot)],
            records_path=records_path,
            source_identity="a" * 64,
        )


def test_payload_target_rejects_unproven_or_stale_dynamics() -> None:
    snapshot = _snapshot(1, payload_evidence=True)
    assert _payload_targets(snapshot, motion_unstable=False) is not None

    dynamics = snapshot["strategic_context"]["payload"]["dynamics"]
    dynamics.pop("telemetry_payload_sha256")
    assert _payload_targets(snapshot, motion_unstable=False) is None

    dynamics["telemetry_payload_sha256"] = "a" * 64
    dynamics["source_sample_age_seconds"]["imu"] = 3.1
    assert _payload_targets(snapshot, motion_unstable=False) is None

    dynamics["source_sample_age_seconds"]["imu"] = 0.01
    dynamics["source_sample_age_seconds"].pop("actuator_output")
    assert _payload_targets(snapshot, motion_unstable=False) is None

    for invalid_mass in (True, 0.0, -0.1, float("nan"), float("inf")):
        invalid = _snapshot(1, payload_evidence=True)
        invalid["strategic_context"]["payload"][
            "observed_payload_mass_kg"
        ] = invalid_mass
        assert _payload_targets(invalid, motion_unstable=False) is None


def _write_run(
    root: Path,
    *,
    payload_evidence: bool = True,
    offset: int = 0,
    sensor_rate_history: bool = False,
    multimodal_evidence: bool = False,
) -> None:
    root.mkdir()
    snapshots = []
    cycles = []
    for sequence in range(12):
        snapshot = _snapshot(
            sequence + offset,
            payload_evidence=payload_evidence,
            multimodal_evidence=multimodal_evidence,
        )
        snapshots.append(
            {
                "recorded_at_unix_ms": 1_000 + sequence,
                "snapshot": snapshot,
            }
        )
        cycles.append(
            {
                "sequence": sequence,
                "recorded_at_unix_ms": 1_000 + sequence,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "model_action": "hold" if sequence % 4 == 0 else "select-candidate",
                "hold_reason": "PRECISION_SETTLE_REQUIRED" if sequence % 4 == 0 else None,
            }
        )
    snapshot_path = root / "model-navigation-snapshots.jsonl"
    cycle_path = root / "model-navigation-cycles.jsonl"
    snapshot_path.write_text(
        "".join(json.dumps(item) + "\n" for item in snapshots),
        encoding="utf-8",
    )
    cycle_path.write_text(
        "".join(json.dumps(item) + "\n" for item in cycles),
        encoding="utf-8",
    )
    depth_history_path = root / "depth-local-safety-history.jsonl"
    if sensor_rate_history:
        depth_history_path.write_text(
            "".join(
                json.dumps(
                    {
                        "recorded_at_unix_ms": 1_000 + sequence,
                        "observation": {
                            "sequence": sequence,
                            "stream_healthy": sequence % 3 != 0,
                            "stream_age_seconds": 0.5 if sequence % 3 == 0 else 0.01,
                            "localization_covariance_m2": 0.01,
                            "current_position_m": {
                                "x": (sequence + offset) * 0.1,
                                "y": 0.0,
                                "z": 1.0,
                            },
                            "current_velocity_mps": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                            },
                        },
                    }
                )
                + "\n"
                for sequence in range(12)
            ),
            encoding="utf-8",
        )
    route_path = root / "mission-route.json"
    route_path.write_text(json.dumps({"positions_m": [
        {"x": 0., "y": float(offset), "z": 1.},
        {"x": 2., "y": float(offset), "z": 1.}]}), encoding="utf-8")
    evidence = {
        "status": "verified",
        "artifacts": {
            "model_navigation_snapshots_sha256": _sha256(snapshot_path),
            "model_navigation_cycles_sha256": _sha256(cycle_path),
            "depth_safety_history_sha256": (
                _sha256(depth_history_path) if sensor_rate_history else None
            ),
            "semantic_sha256": ("a" if offset == 0 else "b") * 64,
            "route_sha256": _sha256(route_path),
        },
    }
    (root / "mission_evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )


def _write_verified_fault_recovery(run_root: Path) -> Path:
    run_root.mkdir()
    simulation = run_root / "simulation"
    _write_run(simulation, offset=100, sensor_rate_history=True, multimodal_evidence=True)
    evidence_path = simulation / "mission_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["status"] = "failed"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    summary_path = run_root / "qualification-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "gates": {"development_fault_injection_absent": False},
            }
        ),
        encoding="utf-8",
    )
    fault_path = simulation / "development-depth-fault.json"
    fault_path.write_text(json.dumps({"development_only": True}), encoding="utf-8")
    executor_history = (
        simulation / "runtime-state" / "local-safety-executor-history.jsonl"
    )
    executor_history.parent.mkdir()
    executor_history.write_text(
        json.dumps({"status": "accepted", "action": "hold"}) + "\n",
        encoding="utf-8",
    )
    depth_history = simulation / "depth-local-safety-history.jsonl"
    receipt = {
        "schema_version": "dronedream.depth-fault-recovery-verification.v1",
        "status": "verified",
        "development_only": True,
        "flight_qualification_granted": False,
        "source": {
            "run_root": str(run_root.resolve()),
            "qualification_summary_sha256": _sha256(summary_path),
            "fault_evidence_sha256": _sha256(fault_path),
            "executor_history_sha256": _sha256(executor_history),
            "depth_history_sha256": _sha256(depth_history),
        },
        "gates": {
            "motion_not_authorized_during_depth_outage": True,
            "mission_completed_after_recovery": True,
        },
    }
    receipt_path = run_root / "depth-fault-recovery-verification.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return simulation


def _command(
    repository: Path,
    training_run: Path,
    validation_run: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repository / "scripts" / "build_local_advisor_dataset.py"),
        "--training-run-dir",
        str(training_run),
        "--validation-run-dir",
        str(validation_run),
        "--training-output",
        str(output_root / "training.jsonl"),
        "--validation-output",
        str(output_root / "validation.jsonl"),
        "--receipt-output",
        str(output_root / "receipt.json"),
        "--require-verified-source",
    ]


def test_builder_keeps_complete_missions_in_independent_splits(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    training_run = tmp_path / "training-run"
    validation_run = tmp_path / "validation-run"
    _write_run(training_run)
    _write_run(validation_run, offset=100)
    output_root = tmp_path / "dataset"

    result = subprocess.run(
        _command(repository, training_run, validation_run, output_root),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads((output_root / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["split_method"] == SPATIAL_SPLIT_CONTRACT
    assert receipt["qualification_granted"] is False
    assert set(receipt["training_sample_counts"]) == {
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
    }
    training_samples = [
        LocalAdvisorTrainingSample.model_validate_json(line)
        for line in (output_root / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(sample.risk_target == 1.0 for sample in training_samples)
    assert any(
        sample.role == "payload-dynamics-adapter"
        and sample.controller_step_scale_target < 1.0
        for sample in training_samples
    )
    assert any(
        sample.role == "payload-dynamics-adapter"
        and sample.risk_target == 0.8
        and sample.payload_features[-1] == 1.0
        for sample in training_samples
    )


def test_builder_refuses_to_invent_missing_payload_evidence(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    training_run = tmp_path / "training-run"
    validation_run = tmp_path / "validation-run"
    _write_run(training_run, payload_evidence=False)
    _write_run(validation_run, payload_evidence=False, offset=100)

    result = subprocess.run(
        _command(repository, training_run, validation_run, tmp_path / "dataset"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "LOCAL_ADVISOR_TRAINING_SAMPLES_MISSING_PAYLOAD-DYNAMICS-ADAPTER" in (
        result.stderr
    )


def test_builder_uses_sensor_rate_health_history_for_perception_advisor(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    training_run = tmp_path / "training-run"
    validation_run = tmp_path / "validation-run"
    _write_run(training_run, sensor_rate_history=True)
    _write_run(validation_run, offset=100, sensor_rate_history=True)
    output_root = tmp_path / "dataset"

    result = subprocess.run(
        _command(repository, training_run, validation_run, output_root),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    samples = [
        LocalAdvisorTrainingSample.model_validate_json(line)
        for line in (output_root / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    perception = [sample for sample in samples if sample.role == "perception-health-critic"]
    assert len(perception) == 12
    assert sum(sample.risk_target == 1.0 for sample in perception) == 4
    assert all(
        sample.state_features[7] == (0.0 if sample.risk_target == 1.0 else 1.0)
        for sample in perception
    )


def test_builder_emits_cross_modal_samples_only_from_bound_envelopes(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    training_run = tmp_path / "training-run"
    validation_run = tmp_path / "validation-run"
    _write_run(training_run, multimodal_evidence=True)
    _write_run(validation_run, offset=100, multimodal_evidence=True)
    output_root = tmp_path / "dataset"
    command = _command(repository, training_run, validation_run, output_root)
    command.extend(["--role", "cross-modal-consistency-critic"])

    result = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    samples = [
        LocalAdvisorTrainingSample.model_validate_json(line)
        for line in (output_root / "training.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(samples) == 12
    assert all(sample.role == "cross-modal-consistency-critic" for sample in samples)
    assert sum(sample.risk_target == 1.0 for sample in samples) == 4
    assert all(len(sample.sensor_features) == 64 for sample in samples)


def test_builder_accepts_only_hash_bound_verified_fault_recovery_source(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    training_run = tmp_path / "training-run"
    validation_root = tmp_path / "validation-run"
    _write_run(training_run, multimodal_evidence=True)
    validation_run = _write_verified_fault_recovery(validation_root)
    output_root = tmp_path / "dataset"
    command = _command(repository, training_run, validation_run, output_root)
    command.extend(
        [
            "--role",
            "cross-modal-consistency-critic",
            "--allow-verified-fault-recovery-source",
        ]
    )

    result = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads((output_root / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["verified_fault_recovery_sources_allowed"] is True
    source = receipt["validation_sources"][0]
    assert source["verification"] == "verified-fail-closed-depth-fault-recovery"
    assert source["flight_qualification_granted"] is False

    # Tampering with any bound source invalidates the recovery receipt.
    (validation_root / "qualification-summary.json").write_text(
        json.dumps({"status": "tampered"}), encoding="utf-8"
    )
    tampered_output = tmp_path / "tampered-dataset"
    tampered_command = _command(
        repository,
        training_run,
        validation_run,
        tampered_output,
    )
    tampered_command.extend(
        [
            "--role",
            "cross-modal-consistency-critic",
            "--allow-verified-fault-recovery-source",
        ]
    )
    tampered = subprocess.run(
        tampered_command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode == 1
    assert "LOCAL_ADVISOR_FAULT_RECOVERY_EVIDENCE_INVALID" in tampered.stderr
