#!/usr/bin/env python3
"""Build leak-resistant datasets for independent local flight advisors.

The builder accepts complete runtime directories instead of unrelated rows.  A
whole mission is assigned to either training or validation, and every sample is
bound to the recorded navigation snapshot, cycle, and mission evidence hashes.
Failure runs may be useful negative evidence, but callers must opt in by
omitting ``--require-verified-source``; their status is always retained in the
receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingSample,
    TrainableAdvisorRole,
)
from dronedream_agent_core.local_expert_harness import requested_navigation_expert
from dronedream_agent_core.local_policy_packages import (
    LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
    LOCAL_POLICY_STATE_FEATURE_COUNT,
    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
)
from dronedream_agent_core.local_policy_port import compile_local_policy_features
from dronedream_agent_core.training.advisor_sources import (
    RecordedSource,
    validate_advisor_spatial_splits,
)
from dronedream_agent_core.training.mission_groups import (
    SPATIAL_SPLIT_CONTRACT,
    recorded_mission_group,
)

ALL_ADVISOR_ROLES: tuple[TrainableAdvisorRole, ...] = (
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
)
DEFAULT_ADVISOR_ROLES: tuple[TrainableAdvisorRole, ...] = ALL_ADVISOR_ROLES[:3]
_PAYLOAD_RUNTIME_STATES = {"attached", "custody-confirmed", "loaded-stable"}
_MINIMUM_VERIFIED_PAYLOAD_MOTION_SAMPLES = 50
_MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT = 2


def _sha256(path: Path | RecordedSource) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path | RecordedSource) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _jsonl(path: Path | RecordedSource) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _verified_snapshot(snapshot: dict[str, object]) -> str:
    payload = dict(snapshot)
    observed = str(payload.pop("snapshot_sha256", ""))
    if not observed or observed != sha256_json(payload):
        raise RuntimeError("LOCAL_ADVISOR_SNAPSHOT_HASH_MISMATCH")
    return observed


def _verified_fault_recovery_source(
    run_dir: Path,
    *,
    depth_history_path: Path | None,
) -> dict[str, object]:
    """Validate a separate fail-closed recovery receipt for a fault campaign.

    A development fault is intentionally excluded from normal mission
    qualification.  It can still provide the most important negative examples
    for health and consistency critics, but only when the independent recovery
    verifier proved that motion stopped during the outage, fresh sensing
    returned, model-authorized motion resumed, and the mission completed.  This
    receipt never grants flight qualification; it only makes the recorded
    samples eligible for a development training or held-out split.
    """

    run_root = run_dir.parent
    receipt_path = run_root / "depth-fault-recovery-verification.json"
    summary_path = run_root / "qualification-summary.json"
    fault_path = run_dir / "development-depth-fault.json"
    executor_history_path = (
        run_dir / "runtime-state" / "local-safety-executor-history.jsonl"
    )
    for path in (
        receipt_path,
        summary_path,
        fault_path,
        executor_history_path,
        depth_history_path,
    ):
        if path is None or not path.is_file():
            raise RuntimeError("LOCAL_ADVISOR_FAULT_RECOVERY_EVIDENCE_MISSING")

    receipt = _json(receipt_path)
    source = _mapping(receipt.get("source"))
    gates = _mapping(receipt.get("gates"))
    source_run_root = source.get("run_root")
    if (
        receipt.get("schema_version")
        != "dronedream.depth-fault-recovery-verification.v1"
        or receipt.get("status") != "verified"
        or receipt.get("development_only") is not True
        or receipt.get("flight_qualification_granted") is not False
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(source_run_root, str)
        or Path(source_run_root).resolve() != run_root.resolve()
        or source.get("qualification_summary_sha256") != _sha256(summary_path)
        or source.get("fault_evidence_sha256") != _sha256(fault_path)
        or source.get("executor_history_sha256") != _sha256(executor_history_path)
        or source.get("depth_history_sha256") != _sha256(depth_history_path)
    ):
        raise RuntimeError("LOCAL_ADVISOR_FAULT_RECOVERY_EVIDENCE_INVALID")
    return {
        "verification": "verified-fail-closed-depth-fault-recovery",
        "fault_recovery_receipt": str(receipt_path.resolve()),
        "fault_recovery_receipt_sha256": _sha256(receipt_path),
        "flight_qualification_granted": False,
    }


def _verified_payload_collection_source(run_dir: Path) -> dict[str, object]:
    """Validate one deliberately non-qualifying payload data campaign."""

    run_root = run_dir.parent
    summary_path = run_root / "runtime-acceptance-summary.json"
    workflow_path = run_dir / "workflow-result.json"
    snapshots_path = run_dir / "model-navigation-snapshots.jsonl"
    cycles_path = run_dir / "model-navigation-cycles.jsonl"
    calls_path = run_dir / "model-navigation-model-calls.jsonl"
    multimodal_records_path = run_dir / "multimodal-dataset" / "records.jsonl"
    for path in (
        summary_path,
        workflow_path,
        snapshots_path,
        cycles_path,
        calls_path,
        multimodal_records_path,
    ):
        if not path.is_file():
            raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_COLLECTION_EVIDENCE_MISSING")
    summary = _json(summary_path)
    workflow = _json(workflow_path)
    development = _mapping(summary.get("development_payload_collection"))
    multimodal_summary = _mapping(_mapping(summary.get("multimodal_dataset")).get("summary"))
    runtime = _mapping(workflow.get("runtime_evidence"))
    gates = _mapping(runtime.get("gates"))
    measurements = _mapping(runtime.get("measurements"))
    allowed_false_gates = {"development_payload_collection_absent"}
    model_navigation = _mapping(measurements.get("model_navigation"))
    if model_navigation.get("visual_enabled") is False:
        allowed_false_gates.add("model_navigation_visual_frame_recorded")
    multimodal_dataset = _mapping(measurements.get("multimodal_dataset"))
    if (
        multimodal_dataset.get("enabled") is True
        and multimodal_dataset.get("semantic_topic") is None
    ):
        allowed_false_gates.add("semantic_supervision_recorded_for_every_sample")
    false_gates = {key for key, value in gates.items() if value is not True}
    bounded_motion_count = development.get("bounded_motion_cycle_count")
    if (
        summary.get("schema_version")
        != "dronedream.pluginized-runtime-acceptance.v1"
        or summary.get("status") != "verified-development-payload-collection"
        or summary.get("flight_qualification_granted") is not False
        or summary.get("workflow_sha256") != _sha256(workflow_path)
        or workflow.get("status") == "verified"
        or false_gates != allowed_false_gates
        or development.get("status") != "verified-development-data"
        or development.get("development_only") is not True
        or development.get("flight_qualification_granted") is not False
        or development.get("model_navigation_snapshots_sha256")
        != _sha256(snapshots_path)
        or development.get("model_navigation_cycles_sha256") != _sha256(cycles_path)
        or development.get("model_navigation_model_calls_sha256")
        != _sha256(calls_path)
        or multimodal_summary.get("records_sha256")
        != _sha256(multimodal_records_path)
        or not isinstance(multimodal_summary.get("record_count"), int)
        or isinstance(multimodal_summary.get("record_count"), bool)
        or int(multimodal_summary["record_count"]) <= 0
        or not isinstance(bounded_motion_count, int)
        or isinstance(bounded_motion_count, bool)
        or bounded_motion_count < _MINIMUM_VERIFIED_PAYLOAD_MOTION_SAMPLES
        or development.get("stale_dynamics_motion_cycle_count") != 0
        or development.get("oversized_motion_cycle_count") != 0
        or development.get("maximum_controller_step_scale") != 0.2
    ):
        raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_COLLECTION_EVIDENCE_INVALID")
    return {
        "verification": "verified-development-payload-collection",
        "payload_collection_summary": str(summary_path.resolve()),
        "payload_collection_summary_sha256": _sha256(summary_path),
        "model_navigation_model_calls_sha256": _sha256(calls_path),
        "multimodal_dataset_records_sha256": _sha256(multimodal_records_path),
        "bounded_payload_motion_cycle_count": bounded_motion_count,
        "flight_qualification_granted": False,
    }


def _verified_payload_recovery_source(run_dir: Path) -> dict[str, object]:
    """Validate a failed mission's separate payload-segment recovery receipt."""

    run_root = run_dir.parent
    receipt_path = run_root / "payload-recovery-verification.json"
    if not receipt_path.is_file():
        raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_RECOVERY_EVIDENCE_MISSING")
    receipt = _json(receipt_path)
    source = _mapping(receipt.get("source"))
    gates = _mapping(receipt.get("gates"))
    planning_root_value = source.get("planning_root")
    source_run_root = source.get("run_root")
    if not isinstance(planning_root_value, str):
        raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_RECOVERY_EVIDENCE_INVALID")
    planning_root = Path(planning_root_value)
    bound_files = {
        "prepared_mission_sha256": planning_root / "plan" / "prepared-mission.json",
        "execution_authority_sha256": run_root
        / "model-harness-execution-authority.json",
        "workflow_result_sha256": run_dir / "workflow-result.json",
        "mission_evidence_sha256": run_dir / "mission_evidence.json",
        "offboard_timing_sha256": run_dir / "offboard_timing.json",
        "model_navigation_snapshots_sha256": run_dir
        / "model-navigation-snapshots.jsonl",
        "model_navigation_cycles_sha256": run_dir / "model-navigation-cycles.jsonl",
        "model_navigation_model_calls_sha256": run_dir
        / "model-navigation-model-calls.jsonl",
        "px4_identity_telemetry_sha256": run_dir
        / "runtime-state"
        / "px4-identity-telemetry.json",
        "multimodal_summary_sha256": run_dir
        / "multimodal-dataset"
        / "summary.json",
        "multimodal_records_sha256": run_dir
        / "multimodal-dataset"
        / "records.jsonl",
    }
    if (
        receipt.get("schema_version")
        != "dronedream.payload-recovery-verification.v1"
        or receipt.get("status") != "verified"
        or receipt.get("development_only") is not True
        or receipt.get("flight_qualification_granted") is not False
        or receipt.get("whole_mission_status") != "failed"
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(source_run_root, str)
        or Path(source_run_root).resolve() != run_root.resolve()
        or any(
            not path.is_file() or source.get(key) != _sha256(path)
            for key, path in bound_files.items()
        )
    ):
        raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_RECOVERY_EVIDENCE_INVALID")
    measurements = _mapping(receipt.get("measurements"))
    bounded_motion_count = measurements.get("bounded_loaded_motion_cycle_count")
    if (
        not isinstance(bounded_motion_count, int)
        or isinstance(bounded_motion_count, bool)
        or bounded_motion_count < _MINIMUM_VERIFIED_PAYLOAD_MOTION_SAMPLES
    ):
        raise RuntimeError("LOCAL_ADVISOR_PAYLOAD_RECOVERY_EVIDENCE_INVALID")
    return {
        "verification": "verified-development-payload-recovery",
        "payload_recovery_receipt": str(receipt_path.resolve()),
        "payload_recovery_receipt_sha256": _sha256(receipt_path),
        "bounded_payload_motion_cycle_count": bounded_motion_count,
        "whole_mission_status": "failed",
        "flight_qualification_granted": False,
    }


def _source_receipt(
    run_dir: Path,
    *,
    require_verified: bool,
    allow_verified_fault_recovery: bool,
    allow_verified_payload_collection: bool,
    allow_verified_payload_recovery: bool,
) -> tuple[dict[str, object], RecordedSource, RecordedSource,
           RecordedSource | None, RecordedSource | None]:
    resolved = run_dir.resolve()
    evidence_path = resolved / "mission_evidence.json"
    snapshot_path = resolved / "model-navigation-snapshots.jsonl"
    cycle_path = resolved / "model-navigation-cycles.jsonl"
    for path in (evidence_path, snapshot_path, cycle_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete local advisor source: {path}")
    evidence_source = RecordedSource.read(evidence_path)
    evidence = _json(evidence_source)
    status = str(evidence.get("status", "unknown"))
    artifacts = _mapping(evidence.get("artifacts"))
    snapshot_source, cycle_source = (RecordedSource.read(p) for p in (snapshot_path, cycle_path))
    snapshot_sha256 = _sha256(snapshot_source)
    cycle_sha256 = _sha256(cycle_source)
    if artifacts.get("model_navigation_snapshots_sha256") != snapshot_sha256:
        raise RuntimeError("LOCAL_ADVISOR_SOURCE_SNAPSHOT_FILE_HASH_MISMATCH")
    if artifacts.get("model_navigation_cycles_sha256") != cycle_sha256:
        raise RuntimeError("LOCAL_ADVISOR_SOURCE_CYCLE_FILE_HASH_MISMATCH")
    depth_history_path = resolved / "depth-local-safety-history.jsonl"
    depth_history_sha256 = None
    depth_history_source = None
    if depth_history_path.is_file():
        depth_history_source = RecordedSource.read(depth_history_path)
        depth_history_sha256 = _sha256(depth_history_source)
        if artifacts.get("depth_safety_history_sha256") != depth_history_sha256:
            raise RuntimeError("LOCAL_ADVISOR_SOURCE_DEPTH_HISTORY_FILE_HASH_MISMATCH")
    multimodal_records_path = resolved / "multimodal-dataset" / "records.jsonl"
    multimodal_records_sha256 = None
    multimodal_records_source = None
    if multimodal_records_path.is_file():
        multimodal_records_source = RecordedSource.read(multimodal_records_path)
        multimodal_records_sha256 = _sha256(multimodal_records_source)
        if artifacts.get("multimodal_dataset_records_sha256") != multimodal_records_sha256:
            raise RuntimeError("LOCAL_ADVISOR_SOURCE_MULTIMODAL_RECORDS_HASH_MISMATCH")
    source_verification: dict[str, object] = {"verification": "verified-mission"}
    if require_verified and status != "verified":
        payload_source_allowed = bool(
            allow_verified_payload_collection or allow_verified_payload_recovery
        )
        if allow_verified_fault_recovery and payload_source_allowed:
            raise RuntimeError("LOCAL_ADVISOR_MULTIPLE_DEVELOPMENT_SOURCE_MODES")
        recovery_receipt = resolved.parent / "payload-recovery-verification.json"
        if allow_verified_payload_recovery and recovery_receipt.is_file():
            source_verification = _verified_payload_recovery_source(resolved)
        elif allow_verified_payload_collection:
            source_verification = _verified_payload_collection_source(resolved)
        elif allow_verified_payload_recovery:
            source_verification = _verified_payload_recovery_source(resolved)
        elif allow_verified_fault_recovery:
            source_verification = _verified_fault_recovery_source(
                resolved,
                depth_history_path=(
                    depth_history_path if depth_history_path.is_file() else None
                ),
            )
        else:
            raise RuntimeError("LOCAL_ADVISOR_SOURCE_MISSION_NOT_VERIFIED")
    return (
        {
            "run_directory": str(resolved),
            "mission_status": status,
            "mission_evidence_sha256": _sha256(evidence_source),
            "snapshot_file_sha256": snapshot_sha256,
            "cycle_file_sha256": cycle_sha256,
            "depth_safety_history_sha256": depth_history_sha256,
            "multimodal_dataset_records_sha256": multimodal_records_sha256,
            "semantic_sha256": artifacts.get("semantic_sha256"),
            "mission_split": recorded_mission_group(resolved, artifacts).model_dump(mode="json"),
            **source_verification,
        },
        snapshot_source,
        cycle_source,
        depth_history_source,
        multimodal_records_source,
    )


def _history_tensor(
    history: deque[tuple[float, ...]],
) -> tuple[list[list[float]], list[float]]:
    padding = LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH - len(history)
    rows = [
        *(
            [0.0] * LOCAL_POLICY_STATE_FEATURE_COUNT
            for _ in range(padding)
        ),
        *(list(row) for row in history),
    ]
    return rows, [0.0] * padding + [1.0] * len(history)


def _payload_history_tensor(
    history: deque[tuple[float, ...]],
) -> list[list[float]]:
    padding = LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH - len(history)
    return [
        *(
            [0.0] * LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
            for _ in range(padding)
        ),
        *(list(row) for row in history),
    ]


def _velocity(snapshot: dict[str, object]) -> tuple[float, float, float]:
    velocity = _mapping(snapshot.get("current_velocity_mps"))
    return (
        _number(velocity.get("x")),
        _number(velocity.get("y")),
        _number(velocity.get("z")),
    )


def _settle_unstable(
    snapshot: dict[str, object],
    velocity_history: deque[tuple[float, float, float]],
) -> bool:
    strategic = _mapping(snapshot.get("strategic_context"))
    task = _mapping(strategic.get("task"))
    phase = str(task.get("phase", "UNKNOWN")).strip().upper()
    envelope = _mapping(snapshot.get("vehicle_envelope_m"))
    candidate_speed = max(0.05, _number(envelope.get("candidate_speed"), 0.5))
    current = velocity_history[-1]
    speed = math.sqrt(sum(value * value for value in current))
    stationary_phase = phase in {
        "CHECKPOINT",
        "ACTION",
        "PICKUP",
        "HOVER",
        "WAYPOINT_SETTLE",
        "LAND",
        "LANDING",
        "COMPLETE",
    }
    speed_limit = (
        max(0.12, min(0.25, candidate_speed * 0.35))
        if stationary_phase
        else max(0.25, candidate_speed * 0.9)
    )
    if speed > speed_limit or (stationary_phase and abs(current[2]) > 0.12):
        return True
    if len(velocity_history) >= 2:
        previous = velocity_history[-2]
        previous_speed = math.sqrt(sum(value * value for value in previous))
        reversal = sum(a * b for a, b in zip(previous, current, strict=True)) < -0.01
        if reversal and min(previous_speed, speed) > 0.1:
            return True
    return False


def _state_anomaly_recorded(snapshot: dict[str, object]) -> bool:
    """Use recorded health verdicts, never an invented clean-state label."""

    health = _mapping(snapshot.get("perception_health"))
    issue_codes = health.get("issue_codes")
    health_issue = isinstance(issue_codes, list) and bool(issue_codes)
    multimodal = _mapping(snapshot.get("multimodal_sensor_snapshot"))
    multimodal_issue = bool(multimodal) and not bool(
        multimodal.get("ready_for_motion", False)
    )
    strategic = _mapping(snapshot.get("strategic_context"))
    task = _mapping(strategic.get("task"))
    trigger = str(task.get("decision_trigger", "")).strip().lower()
    return bool(
        not health.get("stream_healthy", False)
        or health_issue
        or multimodal_issue
        or trigger in {"state-anomaly", "identity-divergence", "sensor-conflict"}
    )


def _payload_targets(
    snapshot: dict[str, object],
    *,
    motion_unstable: bool,
) -> tuple[float, float] | None:
    strategic = _mapping(snapshot.get("strategic_context"))
    payload = _mapping(strategic.get("payload"))
    dynamics = _mapping(payload.get("dynamics"))
    state = str(payload.get("state", "")).strip().lower()
    observed_mass = payload.get("observed_payload_mass_kg")
    maximum_mass = _number(payload.get("maximum_payload_kg"))
    within_limit = payload.get("within_declared_payload_limit")
    maximum_sample_age_seconds = dynamics.get("maximum_sample_age_seconds")
    source_sample_age_seconds = _mapping(
        dynamics.get("source_sample_age_seconds")
    )
    telemetry_payload_sha256 = dynamics.get("telemetry_payload_sha256")
    issue_codes = dynamics.get("issue_codes")
    freshness_contract_valid = bool(
        dynamics.get("telemetry_schema_version")
        == "dronedream.px4-dynamics-telemetry.v1"
        and isinstance(telemetry_payload_sha256, str)
        and len(telemetry_payload_sha256) == 64
        and all(character in "0123456789abcdef" for character in telemetry_payload_sha256)
        and isinstance(maximum_sample_age_seconds, int | float)
        and not isinstance(maximum_sample_age_seconds, bool)
        and math.isfinite(float(maximum_sample_age_seconds))
        and 0.0 < float(maximum_sample_age_seconds) <= 3.0
        and all(
            isinstance(source_sample_age_seconds.get(source_name), int | float)
            and not isinstance(source_sample_age_seconds.get(source_name), bool)
            and math.isfinite(float(source_sample_age_seconds[source_name]))
            and 0.0
            <= float(source_sample_age_seconds[source_name])
            <= float(maximum_sample_age_seconds)
            for source_name in ("imu", "attitude", "battery", "actuator_output")
        )
        and isinstance(issue_codes, list)
        and not issue_codes
        and dynamics.get("actuator_normalization_ready") is True
    )
    if (
        state not in _PAYLOAD_RUNTIME_STATES
        or isinstance(observed_mass, bool)
        or not isinstance(observed_mass, int | float)
        or not math.isfinite(float(observed_mass))
        or float(observed_mass) <= 0.0
        or maximum_mass <= 0.0
        or not isinstance(within_limit, bool)
        or dynamics.get("available") is not True
        or dynamics.get("ready") is not True
        or not freshness_contract_valid
    ):
        return None
    required_dynamics = (
        "acceleration_forward_m_s2",
        "acceleration_right_m_s2",
        "acceleration_down_m_s2",
        "angular_velocity_forward_rad_s",
        "angular_velocity_right_rad_s",
        "angular_velocity_down_rad_s",
        "roll_deg",
        "pitch_deg",
        "current_battery_a",
        "voltage_v",
        "actuator_mean",
        "actuator_max_abs",
    )
    if any(
        isinstance(dynamics.get(name), bool)
        or not isinstance(dynamics.get(name), int | float)
        or not math.isfinite(float(dynamics[name]))
        for name in required_dynamics
    ):
        return None
    mass_ratio = max(0.0, float(observed_mass) / maximum_mass)
    transition_incomplete = state != "loaded-stable"
    overloaded = not within_limit or mass_ratio > 1.0 + 1e-9
    acceleration_magnitude = math.sqrt(
        sum(
            float(dynamics[name]) ** 2
            for name in (
                "acceleration_forward_m_s2",
                "acceleration_right_m_s2",
                "acceleration_down_m_s2",
            )
        )
    )
    angular_rate_magnitude = math.sqrt(
        sum(
            float(dynamics[name]) ** 2
            for name in (
                "angular_velocity_forward_rad_s",
                "angular_velocity_right_rad_s",
                "angular_velocity_down_rad_s",
            )
        )
    )
    dynamics_unstable = bool(
        abs(acceleration_magnitude - 9.80665) > 3.0
        or angular_rate_magnitude > 0.8
        or abs(float(dynamics["roll_deg"])) > 20.0
        or abs(float(dynamics["pitch_deg"])) > 20.0
        or _number(dynamics.get("actuator_max_abs")) > 0.95
    )
    risk_target = (
        1.0
        if overloaded
        else 0.8
        if transition_incomplete or motion_unstable or dynamics_unstable
        else 0.0
    )
    if overloaded:
        scale_target = 0.1
    else:
        # Weak supervision from the declared payload envelope.  This is a
        # conservative training target, never a flight-qualification claim.
        scale_target = max(0.3, min(1.0, 1.0 - 0.55 * mass_ratio))
        if transition_incomplete or dynamics_unstable:
            scale_target = min(scale_target, 0.4)
    return risk_target, scale_target


def _perception_samples_from_depth_history(
    *,
    snapshot_timeline: list[tuple[int, dict[str, object]]],
    depth_history_path: Path | RecordedSource,
    source_identity: str,
) -> dict[str, LocalAdvisorTrainingSample]:
    """Compile real sensor-rate health verdicts against the nearest prior map snapshot."""

    if not snapshot_timeline:
        return {}
    history: deque[tuple[float, ...]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    samples: dict[str, LocalAdvisorTrainingSample] = {}
    depth_history_sha256 = _sha256(depth_history_path)
    snapshot_index = 0
    active_snapshot: dict[str, object] | None = None
    for record in sorted(
        _jsonl(depth_history_path),
        key=lambda item: int(_number(item.get("recorded_at_unix_ms"))),
    ):
        recorded_at = int(_number(record.get("recorded_at_unix_ms")))
        while (
            snapshot_index < len(snapshot_timeline)
            and snapshot_timeline[snapshot_index][0] <= recorded_at
        ):
            active_snapshot = snapshot_timeline[snapshot_index][1]
            snapshot_index += 1
        if active_snapshot is None:
            continue
        observation = _mapping(record.get("observation"))
        current = _mapping(observation.get("current_position_m"))
        velocity = _mapping(observation.get("current_velocity_mps"))
        if not current or not velocity or "stream_healthy" not in observation:
            continue
        snapshot = copy.deepcopy(active_snapshot)
        snapshot["current_position_m"] = current
        snapshot["current_velocity_mps"] = velocity
        goal = _mapping(snapshot.get("goal_position_m"))
        if goal:
            snapshot["goal_distance_m"] = math.dist(
                (
                    _number(current.get("x")),
                    _number(current.get("y")),
                    _number(current.get("z")),
                ),
                (
                    _number(goal.get("x")),
                    _number(goal.get("y")),
                    _number(goal.get("z")),
                ),
            )
        base_health = _mapping(snapshot.get("perception_health"))
        unsafe = not bool(observation.get("stream_healthy", False))
        snapshot["perception_health"] = {
            **base_health,
            "stream_healthy": not unsafe,
            "stream_age_seconds": max(
                0.0, _number(observation.get("stream_age_seconds"))
            ),
            "localization_covariance_m2": max(
                0.0, _number(observation.get("localization_covariance_m2"))
            ),
            "issue_codes": (
                ["RECORDED_RUNTIME_PERCEPTION_OR_IDENTITY_UNHEALTHY"]
                if unsafe
                else []
            ),
        }
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        batch = compile_local_policy_features(snapshot)
        history.append(batch.state_features)
        state_history, history_mask = _history_tensor(history)
        sample = LocalAdvisorTrainingSample(
            role="perception-health-critic",
            state_features=list(batch.state_features),
            state_history=state_history,
            history_mask=history_mask,
            maneuver_features=list(batch.maneuver_features),
            payload_features=list(batch.payload_features),
            risk_target=1.0 if unsafe else 0.0,
            controller_step_scale_target=1.0,
            sample_weight=3.0 if unsafe else 1.0,
        )
        identity = sha256_json(
            {
                "source": source_identity,
                "depth_history_sha256": depth_history_sha256,
                "recorded_at_unix_ms": recorded_at,
                "observation_sha256": sha256_json(observation),
                "role": "perception-health-critic",
            }
        )
        samples.setdefault(identity, sample)
    return samples


def _payload_transition_samples_from_multimodal_dataset(
    *,
    snapshot_timeline: list[tuple[int, dict[str, object]]],
    records_path: Path | RecordedSource,
    source_identity: str,
) -> dict[str, LocalAdvisorTrainingSample]:
    """Compile real sensor-rate post-attachment transitions against model context.

    Full local-policy snapshots arrive at the bounded decision cadence, which can
    legitimately be slower than the short physical transition between payload
    attachment and the loaded-stability gate. The multimodal recorder observes
    that same transition at sensor rate. Align each content-hashed transition
    record to the latest model snapshot so the payload adapter sees real IMU,
    attitude, actuator, and vehicle-state evidence without inventing labels.
    """

    if not snapshot_timeline:
        return {}
    snapshot_index = 0
    active_snapshot: dict[str, object] | None = None
    history: deque[tuple[float, ...]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    payload_history: deque[tuple[float, ...]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    velocity_history: deque[tuple[float, float, float]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    samples: dict[str, LocalAdvisorTrainingSample] = {}
    for record in sorted(
        _jsonl(records_path),
        key=lambda item: int(_number(item.get("recorded_at_unix_ms"))),
    ):
        recorded_at = int(_number(record.get("recorded_at_unix_ms")))
        while (
            snapshot_index < len(snapshot_timeline)
            and snapshot_timeline[snapshot_index][0] <= recorded_at
        ):
            active_snapshot = snapshot_timeline[snapshot_index][1]
            snapshot_index += 1
        if active_snapshot is None:
            continue
        observed_record_sha256 = str(record.get("record_sha256", ""))
        record_payload = dict(record)
        record_payload.pop("record_sha256", None)
        if (
            len(observed_record_sha256) != 64
            or sha256_json(record_payload) != observed_record_sha256
        ):
            raise RuntimeError("LOCAL_ADVISOR_MULTIMODAL_RECORD_HASH_MISMATCH")
        state = _mapping(record.get("state"))
        payload = _mapping(state.get("payload"))
        if str(payload.get("state", "")).strip().lower() not in {
            "attached",
            "custody-confirmed",
        }:
            continue
        current = _mapping(state.get("current_position_m"))
        velocity = _mapping(state.get("current_velocity_mps"))
        goal = _mapping(state.get("navigation_goal_m"))
        sensor_snapshot = _mapping(record.get("sensor_snapshot"))
        if not current or not velocity or not goal or not sensor_snapshot:
            continue
        snapshot = copy.deepcopy(active_snapshot)
        snapshot["current_position_m"] = current
        snapshot["current_velocity_mps"] = velocity
        snapshot["goal_position_m"] = goal
        snapshot["goal_distance_m"] = math.dist(
            tuple(_number(current.get(axis)) for axis in ("x", "y", "z")),
            tuple(_number(goal.get(axis)) for axis in ("x", "y", "z")),
        )
        snapshot["multimodal_sensor_snapshot"] = sensor_snapshot
        strategic = _mapping(snapshot.get("strategic_context"))
        strategic["payload"] = payload
        task = _mapping(strategic.get("task"))
        task["control_profile"] = str(
            state.get("control_profile", task.get("control_profile", "precision"))
        )
        strategic["task"] = task
        snapshot["strategic_context"] = strategic
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        batch = compile_local_policy_features(snapshot)
        history.append(batch.state_features)
        payload_history.append(batch.payload_features)
        velocity_history.append(_velocity(snapshot))
        state_history, history_mask = _history_tensor(history)
        targets = _payload_targets(
            snapshot,
            motion_unstable=_settle_unstable(snapshot, velocity_history),
        )
        if targets is None or targets[0] < 0.5:
            continue
        risk_target, scale_target = targets
        sample = LocalAdvisorTrainingSample(
            role="payload-dynamics-adapter",
            state_features=list(batch.state_features),
            state_history=state_history,
            history_mask=history_mask,
            maneuver_features=list(batch.maneuver_features),
            payload_features=list(batch.payload_features),
            payload_history=_payload_history_tensor(payload_history),
            sensor_features=list(batch.sensor_features),
            risk_target=risk_target,
            controller_step_scale_target=scale_target,
            sample_weight=3.0,
        )
        identity = sha256_json(
            {
                "source": source_identity,
                "multimodal_records_sha256": _sha256(records_path),
                "record_sha256": observed_record_sha256,
                "role": "payload-dynamics-adapter",
                "risk_target": risk_target,
                "controller_step_scale_target": scale_target,
            }
        )
        samples.setdefault(identity, sample)
    return samples


def _collect_run_samples(
    snapshot_path: Path | RecordedSource,
    cycle_path: Path | RecordedSource,
    depth_history_path: Path | RecordedSource | None,
    multimodal_records_path: Path | RecordedSource | None,
    *,
    roles: tuple[TrainableAdvisorRole, ...],
    source_identity: str,
) -> dict[str, LocalAdvisorTrainingSample]:
    snapshots: dict[str, dict[str, object]] = {}
    snapshot_timeline: list[tuple[int, dict[str, object]]] = []
    for record in _jsonl(snapshot_path):
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        snapshot_sha256 = _verified_snapshot(snapshot)
        snapshots[snapshot_sha256] = snapshot
        snapshot_timeline.append(
            (int(_number(record.get("recorded_at_unix_ms"))), snapshot)
        )
    snapshot_timeline.sort(key=lambda item: item[0])
    history: deque[tuple[float, ...]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    payload_history: deque[tuple[float, ...]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    velocity_history: deque[tuple[float, float, float]] = deque(
        maxlen=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
    )
    samples: dict[str, LocalAdvisorTrainingSample] = {}
    cycles = sorted(
        _jsonl(cycle_path),
        key=lambda cycle: (
            int(_number(cycle.get("sequence"))),
            int(_number(cycle.get("recorded_at_unix_ms"))),
        ),
    )
    for cycle in cycles:
        snapshot_sha256 = str(cycle.get("snapshot_sha256", ""))
        snapshot = snapshots.get(snapshot_sha256)
        if snapshot is None:
            raise RuntimeError("LOCAL_ADVISOR_CYCLE_SNAPSHOT_MISSING")
        batch = compile_local_policy_features(snapshot)
        history.append(batch.state_features)
        payload_history.append(batch.payload_features)
        velocity_history.append(_velocity(snapshot))
        state_history, history_mask = _history_tensor(history)
        settle_unstable = _settle_unstable(snapshot, velocity_history)
        role_targets: list[tuple[TrainableAdvisorRole, float, float, float]] = []
        if "perception-health-critic" in roles and depth_history_path is None:
            health = _mapping(snapshot.get("perception_health"))
            unsafe = not bool(health.get("stream_healthy", False))
            role_targets.append(
                (
                    "perception-health-critic",
                    1.0 if unsafe else 0.0,
                    1.0,
                    3.0 if unsafe else 1.0,
                )
            )
        if (
            "settle-stability-critic" in roles
            and requested_navigation_expert(snapshot) == "precision-maneuver-policy"
        ):
            role_targets.append(
                (
                    "settle-stability-critic",
                    1.0 if settle_unstable else 0.0,
                    1.0,
                    2.0 if settle_unstable else 1.0,
                )
            )
        if "payload-dynamics-adapter" in roles:
            payload_targets = _payload_targets(
                snapshot,
                motion_unstable=settle_unstable,
            )
            if payload_targets is not None:
                payload_risk, payload_scale = payload_targets
                role_targets.append(
                    (
                        "payload-dynamics-adapter",
                        payload_risk,
                        payload_scale,
                        3.0 if payload_risk >= 0.5 else 1.0,
                    )
                )
        if "state-anomaly-detector" in roles:
            state_anomaly = _state_anomaly_recorded(snapshot)
            role_targets.append(
                (
                    "state-anomaly-detector",
                    1.0 if state_anomaly else 0.0,
                    1.0,
                    3.0 if state_anomaly else 1.0,
                )
            )
        if "cross-modal-consistency-critic" in roles:
            multimodal = _mapping(snapshot.get("multimodal_sensor_snapshot"))
            statuses = multimodal.get("statuses")
            if isinstance(statuses, list) and statuses:
                cross_modal_unsafe = not bool(
                    multimodal.get("ready_for_motion", False)
                )
                role_targets.append(
                    (
                        "cross-modal-consistency-critic",
                        1.0 if cross_modal_unsafe else 0.0,
                        1.0,
                        3.0 if cross_modal_unsafe else 1.0,
                    )
                )
        for role, risk_target, scale_target, weight in role_targets:
            sample = LocalAdvisorTrainingSample(
                role=role,
                state_features=list(batch.state_features),
                state_history=state_history,
                history_mask=history_mask,
                maneuver_features=list(batch.maneuver_features),
                payload_features=list(batch.payload_features),
                payload_history=(
                    _payload_history_tensor(payload_history)
                    if role == "payload-dynamics-adapter"
                    else None
                ),
                sensor_features=list(batch.sensor_features),
                risk_target=risk_target,
                controller_step_scale_target=scale_target,
                sample_weight=weight,
            )
            identity = sha256_json(
                {
                    "source": source_identity,
                    "snapshot_sha256": snapshot_sha256,
                    "role": role,
                    "risk_target": risk_target,
                    "controller_step_scale_target": scale_target,
                }
            )
            samples.setdefault(identity, sample)
    if "perception-health-critic" in roles and depth_history_path is not None:
        samples.update(
            _perception_samples_from_depth_history(
                snapshot_timeline=snapshot_timeline,
                depth_history_path=depth_history_path,
                source_identity=source_identity,
            )
        )
    if "payload-dynamics-adapter" in roles and multimodal_records_path is not None:
        samples.update(
            _payload_transition_samples_from_multimodal_dataset(
                snapshot_timeline=snapshot_timeline,
                records_path=multimodal_records_path,
                source_identity=source_identity,
            )
        )
    return samples


def _collect_sources(
    run_dirs: list[Path],
    *,
    roles: tuple[TrainableAdvisorRole, ...],
    require_verified: bool,
    allow_verified_fault_recovery: bool,
    allow_verified_payload_collection: bool,
    allow_verified_payload_recovery: bool,
) -> tuple[dict[str, LocalAdvisorTrainingSample], list[dict[str, object]]]:
    samples: dict[str, LocalAdvisorTrainingSample] = {}
    receipts = []
    for run_dir in run_dirs:
        (
            receipt,
            snapshot_path,
            cycle_path,
            depth_history_path,
            multimodal_records_path,
        ) = _source_receipt(
            run_dir,
            require_verified=require_verified,
            allow_verified_fault_recovery=allow_verified_fault_recovery,
            allow_verified_payload_collection=allow_verified_payload_collection,
            allow_verified_payload_recovery=allow_verified_payload_recovery,
        )
        source_identity = str(receipt["mission_evidence_sha256"])
        run_samples = _collect_run_samples(
            snapshot_path,
            cycle_path,
            depth_history_path,
            multimodal_records_path,
            roles=roles,
            source_identity=source_identity,
        )
        overlap = samples.keys() & run_samples.keys()
        if overlap:
            raise RuntimeError("LOCAL_ADVISOR_DUPLICATE_SAMPLE_WITHIN_SPLIT")
        samples.update(run_samples)
        receipt["sample_count"] = len(run_samples)
        receipts.append(receipt)
    return samples, receipts


def _counts(samples: dict[str, LocalAdvisorTrainingSample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.role for sample in samples.values()).items()))


def _class_counts(
    samples: dict[str, LocalAdvisorTrainingSample],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for sample in samples.values():
        role_counts = counts.setdefault(sample.role, Counter())
        role_counts["risky" if sample.risk_target >= 0.5 else "safe"] += 1
    return {
        role: {"risky": role_counts["risky"], "safe": role_counts["safe"]}
        for role, role_counts in sorted(counts.items())
    }


def _validate_payload_collection_source_diversity(
    receipts: list[dict[str, object]],
    *,
    split: str,
) -> None:
    """Prevent correlated frames from one payload flight masquerading as diversity."""

    mission_hashes = {
        value
        for receipt in receipts
        if isinstance((value := receipt.get("mission_evidence_sha256")), str)
        and len(value) == 64
    }
    split_code = split.upper()
    if len(mission_hashes) < _MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT:
        raise RuntimeError(
            f"PAYLOAD_COLLECTION_{split_code}_HAS_FEWER_THAN_TWO_MISSIONS"
        )
    for receipt in receipts:
        sample_count = receipt.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < _MINIMUM_VERIFIED_PAYLOAD_MOTION_SAMPLES
        ):
            raise RuntimeError(
                f"PAYLOAD_COLLECTION_{split_code}_MISSION_HAS_FEWER_THAN_FIFTY_SAMPLES"
            )


def _write_jsonl(
    path: Path,
    samples: dict[str, LocalAdvisorTrainingSample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(sample.model_dump_json() + "\n" for _, sample in sorted(samples.items())),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--validation-run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=ALL_ADVISOR_ROLES,
        help=(
            "Advisor to include; repeat as needed. Cross-modal training must be "
            "requested because legacy evidence lacks multimodal envelopes."
        ),
    )
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--require-verified-source", action="store_true")
    parser.add_argument(
        "--allow-verified-fault-recovery-source",
        action="store_true",
        help=(
            "Accept a non-qualifying development depth-fault mission only when "
            "its sibling fail-closed recovery receipt is verified and hash-bound."
        ),
    )
    parser.add_argument(
        "--allow-verified-payload-collection-source",
        action="store_true",
        help=(
            "Accept an explicitly non-qualifying payload collection mission only "
            "when its complete runtime summary proves fresh, reduced-step motion."
        ),
    )
    parser.add_argument(
        "--allow-verified-payload-recovery-source",
        action="store_true",
        help=(
            "Accept only the hash-bound payload segments of a failed development "
            "mission whose sibling payload recovery receipt is verified."
        ),
    )
    args = parser.parse_args()
    outputs = (args.training_output, args.validation_output, args.receipt_output)
    if any(path.exists() for path in outputs):
        raise FileExistsError("local advisor dataset output already exists")
    training_dirs = {path.resolve() for path in args.training_run_dir}
    validation_dirs = {path.resolve() for path in args.validation_run_dir}
    if len(training_dirs) != len(args.training_run_dir) or len(validation_dirs) != len(
        args.validation_run_dir
    ):
        parser.error("run directories must be unique within each split")
    if training_dirs & validation_dirs:
        parser.error("a complete mission cannot appear in both dataset splits")
    roles = tuple(dict.fromkeys(args.role or DEFAULT_ADVISOR_ROLES))
    if args.allow_verified_fault_recovery_source and not args.require_verified_source:
        parser.error(
            "verified fault-recovery sources require --require-verified-source"
        )
    payload_source_allowed = bool(
        args.allow_verified_payload_collection_source
        or args.allow_verified_payload_recovery_source
    )
    if payload_source_allowed:
        if not args.require_verified_source:
            parser.error(
                "verified payload sources require --require-verified-source"
            )
        if roles != ("payload-dynamics-adapter",):
            parser.error(
                "verified payload sources may train only the payload adapter"
            )
    if (
        args.allow_verified_fault_recovery_source
        and payload_source_allowed
    ):
        parser.error("development source modes are mutually exclusive")
    training, training_receipts = _collect_sources(
        args.training_run_dir,
        roles=roles,
        require_verified=args.require_verified_source,
        allow_verified_fault_recovery=args.allow_verified_fault_recovery_source,
        allow_verified_payload_collection=(
            args.allow_verified_payload_collection_source
        ),
        allow_verified_payload_recovery=(
            args.allow_verified_payload_recovery_source
        ),
    )
    validation, validation_receipts = _collect_sources(
        args.validation_run_dir,
        roles=roles,
        require_verified=args.require_verified_source,
        allow_verified_fault_recovery=args.allow_verified_fault_recovery_source,
        allow_verified_payload_collection=(
            args.allow_verified_payload_collection_source
        ),
        allow_verified_payload_recovery=(
            args.allow_verified_payload_recovery_source
        ),
    )
    if payload_source_allowed:
        _validate_payload_collection_source_diversity(
            training_receipts,
            split="training",
        )
        _validate_payload_collection_source_diversity(
            validation_receipts,
            split="validation",
        )
    if training.keys() & validation.keys():
        raise RuntimeError("LOCAL_ADVISOR_TRAINING_VALIDATION_SAMPLE_OVERLAP")
    training_groups, validation_groups = validate_advisor_spatial_splits(
        training_receipts, validation_receipts)
    training_counts = _counts(training)
    validation_counts = _counts(validation)
    for role in roles:
        if training_counts.get(role, 0) == 0:
            raise RuntimeError(f"LOCAL_ADVISOR_TRAINING_SAMPLES_MISSING_{role.upper()}")
        if validation_counts.get(role, 0) == 0:
            raise RuntimeError(f"LOCAL_ADVISOR_VALIDATION_SAMPLES_MISSING_{role.upper()}")
    _write_jsonl(args.training_output, training)
    _write_jsonl(args.validation_output, validation)
    receipt = {
        "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
        "split_method": SPATIAL_SPLIT_CONTRACT,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "training_groups": sorted(training_groups),
        "validation_groups": sorted(validation_groups),
        "roles": list(roles),
        "label_policy": {
            "perception-health-critic": (
                "recorded sensor-rate fused perception and identity health verdict; "
                "map context comes from the nearest prior hash-bound navigation snapshot"
            ),
            "settle-stability-critic": (
                "recorded velocity, vertical-motion, and reversal stability in precision context"
            ),
            "payload-dynamics-adapter": (
                "recorded payload state and declared mass envelope; weak supervision only"
            ),
            "state-anomaly-detector": (
                "temporal state history labeled only from recorded perception, identity, "
                "multimodal readiness, and explicit anomaly trigger verdicts"
            ),
            "cross-modal-consistency-critic": (
                "hash-bound per-modality presence, freshness, quality, coverage, and "
                "required-motion readiness verdict"
            ),
        },
        "verified_sources_required": args.require_verified_source,
        "verified_fault_recovery_sources_allowed": (
            args.allow_verified_fault_recovery_source
        ),
        "verified_payload_collection_sources_allowed": (
            args.allow_verified_payload_collection_source
        ),
        "verified_payload_recovery_sources_allowed": (
            args.allow_verified_payload_recovery_source
        ),
        "minimum_payload_collection_missions_per_split": (
            _MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT
            if payload_source_allowed
            else None
        ),
        "training_sources": training_receipts,
        "validation_sources": validation_receipts,
        "training_sample_counts": training_counts,
        "validation_sample_counts": validation_counts,
        "training_class_counts": _class_counts(training),
        "validation_class_counts": _class_counts(validation),
        "training_output_sha256": _sha256(args.training_output),
        "validation_output_sha256": _sha256(args.validation_output),
        "qualification_granted": False,
    }
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
