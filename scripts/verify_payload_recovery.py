#!/usr/bin/env python3
"""Verify payload flight segments without misclassifying a failed mission.

A payload development flight may prove attachment, custody, loaded stability,
fresh dynamics sensing, and model-authorized loaded motion even when the whole
mission later fails closed.  This verifier emits a hash-bound, development-only
receipt for those segments.  It deliberately never grants flight qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dronedream_agent_core.hashing import sha256_json

_PAYLOAD_STATES = {"attached", "custody-confirmed", "loaded-stable"}
_REQUIRED_ACTIONS = {
    "delivery.precontact-hold",
    "pickup",
    "delivery.confirm-custody",
    "delivery.verify-loaded-stability",
}
_ALLOWED_FALSE_GATES = {
    "development_payload_collection_absent",
    "executor_completed",
    "goal_observed",
    "model_navigation_visual_frame_recorded",
    "native_terminal_lifecycle_published",
    "no_live_abort",
    "offboard_timing_complete",
    "semantic_supervision_recorded_for_every_sample",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        values.append(value)
    return values


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _is_applied_model_motion(
    cycle: dict[str, Any],
    *,
    call: dict[str, Any] | None,
) -> bool:
    if not isinstance(cycle.get("controller_target_m"), dict):
        return False
    action = cycle.get("model_action")
    if action == "select-candidate":
        return cycle.get("selected_candidate_id") is not None
    if action != "pilot-control" or not isinstance(call, dict):
        return False
    trace = _mapping(call.get("local_expert_trace"))
    return bool(
        trace.get("navigation_action") == "pilot-control"
        and isinstance(trace.get("pilot_control"), dict)
    )


def _safe_terminal_outcome(measurements: dict[str, Any]) -> bool:
    if measurements.get("landing_state") != "ON_GROUND":
        return False
    abort_reason = measurements.get("abort_reason")
    if abort_reason is None:
        return True
    event = _mapping(measurements.get("live_safety_event"))
    primitive = str(event.get("collision_primitive", "")).casefold()
    return bool(
        abort_reason == "LIVE_STATIC_COLLISION"
        and event.get("reason") == abort_reason
        and event.get("runtime_phase") in {"LANDING", "LANDED", "COMPLETE"}
        and any(token in primitive for token in ("floor", "ground", "road", "pad"))
        and -0.03 <= _number(event.get("clearance_m"), -math.inf) < -0.001
    )


def _action_evidence(
    simulation_root: Path,
    *,
    runtime_actions: dict[str, Any],
    timing: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    steps = runtime_actions.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("prepared mission has no runtime action steps")
    action_contract_sha256 = sha256_json(runtime_actions)
    timing_actions = {
        str(record.get("step_id")): record
        for record in timing.get("runtime_actions", [])
        if isinstance(record, dict)
    }
    receipts: dict[str, dict[str, Any]] = {}
    receipt_file_hashes: dict[str, str] = {}
    receipt_payload_hashes: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id", ""))
        receipt_path = (
            simulation_root / "runtime-actions" / "receipts" / f"{step_id}.receipt.json"
        )
        if receipt_path.is_file():
            receipts[step_id] = _json(receipt_path)
            receipt_file_hashes[step_id] = _sha256(receipt_path)
            receipt_payload_hashes[step_id] = sha256_json(receipts[step_id])

    receipt_contracts_bound = bool(receipts) and all(
        receipt.get("execution_contract_sha256") == action_contract_sha256
        for receipt in receipts.values()
    )
    receipt_steps_bound = len(receipts) == len(steps) and all(
        (receipt := receipts.get(str(step.get("step_id", "")))) is not None
        and receipt.get("step_sha256") == sha256_json(step)
        for step in steps
        if isinstance(step, dict)
    )
    receipts_accepted = len(receipts) == len(steps) and all(
        receipt.get("status") == "accepted"
        and not receipt.get("issue_codes")
        and all(
            accepted is True
            for accepted in _mapping(receipt.get("deterministic_gates")).values()
        )
        for receipt in receipts.values()
    )
    timing_hashes_bound = len(timing_actions) == len(steps) and all(
        timing_actions.get(step_id, {}).get("receipt_sha256") == receipt_sha256
        for step_id, receipt_sha256 in receipt_payload_hashes.items()
    )
    actions = {str(step.get("action", "")) for step in steps if isinstance(step, dict)}
    pickup = next(
        (receipt for receipt in receipts.values() if receipt.get("action") == "pickup"),
        {},
    )
    loaded = next(
        (
            receipt
            for receipt in receipts.values()
            if receipt.get("action") == "delivery.verify-loaded-stability"
        ),
        {},
    )
    pickup_output = _mapping(pickup.get("output"))
    alignment = _mapping(pickup_output.get("payload_mount_alignment"))
    readback = _mapping(alignment.get("attachment_pose_readback"))
    loaded_output = _mapping(loaded.get("output"))
    gates = {
        "required_payload_action_sequence_present": actions == _REQUIRED_ACTIONS,
        "runtime_action_contract_bound": receipt_contracts_bound,
        "runtime_action_steps_bound": receipt_steps_bound,
        "all_payload_actions_accepted": receipts_accepted,
        "timing_receipt_hashes_bound": timing_hashes_bound,
        "physical_attachment_confirmed": (
            pickup_output.get("confirmed") is True
            and pickup_output.get("detached") is False
            and readback.get("accepted") is True
            and 0.0 <= _number(readback.get("alignment_error_m"), math.inf)
            <= _number(readback.get("maximum_alignment_error_m"), -math.inf)
        ),
        "loaded_stability_confirmed": (
            loaded_output.get("confirmed") is True
            and loaded_output.get("detached") is False
            and loaded_output.get("loaded_hover_stable") is True
            and loaded_output.get("return_authorized") is True
        ),
    }
    return gates, {
        "runtime_action_contract_sha256": action_contract_sha256,
        "receipt_file_sha256_by_step": receipt_file_hashes,
        "receipt_payload_sha256_by_step": receipt_payload_hashes,
    }


def verify_payload_recovery(run_root: Path, planning_root: Path) -> dict[str, Any]:
    simulation_root = run_root / "simulation"
    prepared_path = planning_root / "plan" / "prepared-mission.json"
    authority_path = run_root / "model-harness-execution-authority.json"
    workflow_path = simulation_root / "workflow-result.json"
    mission_path = simulation_root / "mission_evidence.json"
    timing_path = simulation_root / "offboard_timing.json"
    snapshots_path = simulation_root / "model-navigation-snapshots.jsonl"
    cycles_path = simulation_root / "model-navigation-cycles.jsonl"
    calls_path = simulation_root / "model-navigation-model-calls.jsonl"
    telemetry_path = simulation_root / "runtime-state" / "px4-identity-telemetry.json"
    multimodal_summary_path = simulation_root / "multimodal-dataset" / "summary.json"
    multimodal_records_path = simulation_root / "multimodal-dataset" / "records.jsonl"
    for path in (
        prepared_path,
        authority_path,
        workflow_path,
        mission_path,
        timing_path,
        snapshots_path,
        cycles_path,
        calls_path,
        telemetry_path,
        multimodal_summary_path,
        multimodal_records_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    prepared = _json(prepared_path)
    authority = _json(authority_path)
    workflow = _json(workflow_path)
    mission = _json(mission_path)
    timing = _json(timing_path)
    telemetry = _json(telemetry_path)
    multimodal_summary = _json(multimodal_summary_path)
    runtime_actions = _mapping(prepared.get("runtime_actions"))
    runtime_evidence = _mapping(workflow.get("runtime_evidence"))
    runtime_gates = _mapping(runtime_evidence.get("gates"))
    measurements = _mapping(runtime_evidence.get("measurements"))
    model_navigation = _mapping(measurements.get("model_navigation"))
    development = _mapping(measurements.get("development_payload_collection"))
    dynamics = _mapping(telemetry.get("dynamics"))
    action_gates, action_measurements = _action_evidence(
        simulation_root,
        runtime_actions=runtime_actions,
        timing=timing,
    )

    snapshot_rows = _jsonl(snapshots_path)
    calls = {
        str(row.get("call_id")): row
        for row in _jsonl(calls_path)
        if isinstance(row.get("call_id"), str)
    }
    snapshots: dict[str, dict[str, Any]] = {}
    snapshot_hashes_valid = True
    for row in snapshot_rows:
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, dict):
            snapshot_hashes_valid = False
            continue
        payload = dict(snapshot)
        observed = str(payload.pop("snapshot_sha256", ""))
        if len(observed) != 64 or sha256_json(payload) != observed:
            snapshot_hashes_valid = False
        snapshots[observed] = snapshot

    loaded_cycle_count = 0
    bounded_loaded_motion_count = 0
    stale_loaded_cycle_count = 0
    stale_loaded_motion_count = 0
    oversized_loaded_motion_count = 0
    for cycle in _jsonl(cycles_path):
        snapshot = snapshots.get(str(cycle.get("snapshot_sha256", "")))
        strategic = _mapping(snapshot.get("strategic_context")) if snapshot else {}
        payload = _mapping(strategic.get("payload"))
        if str(payload.get("state", "")).casefold() not in _PAYLOAD_STATES:
            continue
        loaded_cycle_count += 1
        payload_dynamics = _mapping(payload.get("dynamics"))
        if payload_dynamics.get("ready") is not True:
            stale_loaded_cycle_count += 1
        is_applied_motion = _is_applied_model_motion(
            cycle,
            call=calls.get(str(cycle.get("model_call_id", ""))),
        )
        if is_applied_motion:
            if payload_dynamics.get("ready") is not True:
                stale_loaded_motion_count += 1
            scale = _number(cycle.get("controller_step_scale"), math.inf)
            if scale <= 0.2 + 1e-9:
                bounded_loaded_motion_count += 1
            else:
                oversized_loaded_motion_count += 1

    false_runtime_gates = {
        str(name) for name, accepted in runtime_gates.items() if accepted is not True
    }
    mission_artifacts = _mapping(mission.get("artifacts"))
    multimodal_measurement = _mapping(measurements.get("multimodal_dataset"))
    multimodal_embedded_summary = _mapping(multimodal_measurement.get("summary"))
    dynamics_sources = _mapping(dynamics.get("sources"))
    gates = {
        "whole_mission_remains_failed": (
            workflow.get("status") == "failed"
            and mission.get("status") == "failed"
            and _mapping(workflow.get("completion_assessment")).get("accepted") is False
        ),
        "flight_qualification_not_granted": (
            development.get("development_only") is True
            and development.get("flight_qualification_granted") is False
        ),
        "prepared_mission_authority_bound": (
            authority.get("prepared_mission_sha256") == sha256_json(prepared)
            and authority.get("contract_id") == runtime_actions.get("contract_id")
        ),
        "only_payload_recovery_runtime_gates_failed": (
            bool(false_runtime_gates)
            and false_runtime_gates <= _ALLOWED_FALSE_GATES
            and "development_payload_collection_absent" in false_runtime_gates
        ),
        "safe_terminal_landing_observed": _safe_terminal_outcome(measurements),
        "local_experts_controlled_loaded_motion": (
            model_navigation.get("provider") == "local-policy"
            and model_navigation.get("model_control_authority_required") is True
            and int(model_navigation.get("provider_call_count", 0)) > 0
            and int(model_navigation.get("authorized_control_applied_count", 0)) > 0
        ),
        "development_payload_motion_bound": (
            development.get("enabled") is True
            and development.get("fresh_px4_dynamics_required") is True
            and development.get("maximum_controller_step_scale") == 0.2
            and bounded_loaded_motion_count >= 50
            and stale_loaded_motion_count == 0
            and oversized_loaded_motion_count == 0
        ),
        "model_snapshot_hashes_valid": snapshot_hashes_valid and bool(snapshots),
        "model_artifacts_hash_bound": (
            mission_artifacts.get("model_navigation_snapshots_sha256")
            == _sha256(snapshots_path)
            and mission_artifacts.get("model_navigation_cycles_sha256")
            == _sha256(cycles_path)
            and mission_artifacts.get("model_navigation_calls_sha256")
            == _sha256(calls_path)
        ),
        "fresh_px4_dynamics_recorded": (
            dynamics.get("schema_version") == "dronedream.px4-dynamics-telemetry.v1"
            and dynamics.get("ready_for_payload_inference") is True
            and {"imu", "attitude", "battery"} <= set(dynamics_sources)
            and all(
                0.0
                <= _number(_mapping(dynamics_sources[name]).get("sample_age_seconds"), math.inf)
                <= _number(dynamics.get("maximum_sample_age_seconds"), -math.inf)
                for name in ("imu", "attitude", "battery")
            )
        ),
        "multimodal_dataset_complete_and_bound": (
            multimodal_measurement.get("enabled") is True
            and int(multimodal_summary.get("record_count", 0)) > 0
            and multimodal_summary.get("records_sha256")
            == _sha256(multimodal_records_path)
            and multimodal_embedded_summary.get("records_sha256")
            == multimodal_summary.get("records_sha256")
            and mission_artifacts.get("multimodal_dataset_records_sha256")
            == multimodal_summary.get("records_sha256")
            and multimodal_summary.get("writer_issue") is None
            and multimodal_summary.get("writer_pending") is False
            and multimodal_summary.get("writer_submitted_count")
            == multimodal_summary.get("writer_completed_count")
        ),
        **action_gates,
    }
    return {
        "schema_version": "dronedream.payload-recovery-verification.v1",
        "status": "verified" if all(gates.values()) else "failed",
        "development_only": True,
        "flight_qualification_granted": False,
        "whole_mission_status": "failed",
        "source": {
            "run_root": str(run_root.resolve()),
            "planning_root": str(planning_root.resolve()),
            "prepared_mission_sha256": _sha256(prepared_path),
            "execution_authority_sha256": _sha256(authority_path),
            "workflow_result_sha256": _sha256(workflow_path),
            "mission_evidence_sha256": _sha256(mission_path),
            "offboard_timing_sha256": _sha256(timing_path),
            "model_navigation_snapshots_sha256": _sha256(snapshots_path),
            "model_navigation_cycles_sha256": _sha256(cycles_path),
            "model_navigation_model_calls_sha256": _sha256(calls_path),
            "px4_identity_telemetry_sha256": _sha256(telemetry_path),
            "multimodal_summary_sha256": _sha256(multimodal_summary_path),
            "multimodal_records_sha256": _sha256(multimodal_records_path),
        },
        "gates": gates,
        "measurements": {
            "false_runtime_gates": sorted(false_runtime_gates),
            "loaded_cycle_count": loaded_cycle_count,
            "bounded_loaded_motion_cycle_count": bounded_loaded_motion_count,
            "stale_loaded_cycle_count": stale_loaded_cycle_count,
            "stale_loaded_motion_cycle_count": stale_loaded_motion_count,
            "oversized_loaded_motion_cycle_count": oversized_loaded_motion_count,
            "model_provider_call_count": int(model_navigation.get("provider_call_count", 0)),
            "model_authorized_control_applied_count": int(
                model_navigation.get("authorized_control_applied_count", 0)
            ),
            "multimodal_record_count": int(multimodal_summary.get("record_count", 0)),
            "terminal_abort_reason": measurements.get("abort_reason"),
            **action_measurements,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("planning_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = verify_payload_recovery(args.run_root, args.planning_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
