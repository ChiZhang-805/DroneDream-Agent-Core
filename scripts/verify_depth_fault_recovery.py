#!/usr/bin/env python3
"""Verify that a development depth outage failed closed and then recovered.

Fault-injected flights are deliberately ineligible for normal qualification.
This verifier produces a separate, hash-bound development receipt only when
the underlying mission completed and the executor stopped accepting motion
during the configured depth gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        values.append(value)
    return values


def verify_depth_fault_recovery(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "qualification-summary.json"
    simulation_root = run_root / "simulation"
    fault_path = simulation_root / "development-depth-fault.json"
    executor_history_path = (
        simulation_root / "runtime-state" / "local-safety-executor-history.jsonl"
    )
    depth_history_path = simulation_root / "depth-local-safety-history.jsonl"
    for path in (
        summary_path,
        fault_path,
        executor_history_path,
        depth_history_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = _json(summary_path)
    fault = _json(fault_path)
    executor_history = _jsonl(executor_history_path)
    depth_history = _jsonl(depth_history_path)
    activation_ms = int(fault.get("activated_at_unix_ms", 0))
    recovery_ms = int(fault.get("recovered_at_unix_ms", 0))
    configured_duration_ms = round(float(fault.get("drop_duration_seconds", 0.0)) * 1_000)

    before = [
        record
        for record in executor_history
        if int(record.get("recorded_at_unix_ms", 0)) < activation_ms
        and record.get("status") == "accepted"
        and record.get("action") == "continue"
        and record.get("model_navigation_authorized") is True
    ]
    during = [
        record
        for record in executor_history
        if activation_ms <= int(record.get("recorded_at_unix_ms", 0)) <= recovery_ms
    ]
    after = [
        record
        for record in executor_history
        if int(record.get("recorded_at_unix_ms", 0)) > recovery_ms
        and record.get("status") == "accepted"
        and record.get("action") == "continue"
        and record.get("model_navigation_authorized") is True
    ]
    unsafe_during = [
        record
        for record in during
        if record.get("status") == "accepted" and record.get("action") == "continue"
    ]
    fail_closed_during = [
        record
        for record in during
        if (
            record.get("status") == "accepted"
            and record.get("action") == "hold"
        )
        or record.get("status") == "command-unavailable"
    ]

    depth_timestamps = sorted(
        int(record["recorded_at_unix_ms"])
        for record in depth_history
        if isinstance(record.get("recorded_at_unix_ms"), int)
    )
    depth_before = [value for value in depth_timestamps if value < activation_ms]
    depth_after = [value for value in depth_timestamps if value > recovery_ms]
    observed_gap_ms = (
        depth_after[0] - depth_before[-1]
        if depth_before and depth_after
        else 0
    )

    summary_gates = summary.get("gates")
    if not isinstance(summary_gates, dict):
        raise ValueError("qualification summary has no gates object")
    false_summary_gates = sorted(
        str(name) for name, accepted in summary_gates.items() if accepted is not True
    )
    measurements = summary.get("measurements")
    if not isinstance(measurements, dict):
        measurements = {}
    model_navigation = measurements.get("model_navigation")
    if not isinstance(model_navigation, dict):
        model_navigation = {}

    gates = {
        "development_fault_declared": fault.get("development_only") is True,
        "depth_fault_activated": fault.get("activated") is True and activation_ms > 0,
        "depth_fault_recovered": (
            fault.get("recovered") is True and recovery_ms > activation_ms
        ),
        "fault_duration_matches_contract": (
            configured_duration_ms > 0
            and abs((recovery_ms - activation_ms) - configured_duration_ms) <= 1_000
        ),
        "fault_began_during_model_authorized_motion": bool(before),
        "motion_not_authorized_during_depth_outage": not unsafe_during,
        "executor_held_during_depth_outage": bool(fail_closed_during),
        "fresh_depth_gap_observed": (
            observed_gap_ms >= max(1, configured_duration_ms - 500)
        ),
        "model_authorized_motion_resumed_after_recovery": bool(after),
        "only_normal_qualification_exclusion_is_fault_injection": (
            false_summary_gates == ["development_fault_injection_absent"]
        ),
        "mission_completed_after_recovery": (
            summary_gates.get("executor_completed") is True
            and summary_gates.get("goal_observed") is True
            and summary_gates.get("landing_confirmed") is True
            and measurements.get("abort_reason") is None
            and measurements.get("landing_state") == "ON_GROUND"
        ),
        "local_experts_participated_after_fault": (
            int(model_navigation.get("provider_call_count", 0)) > 0
            and int(model_navigation.get("authorized_control_applied_count", 0)) > 0
        ),
    }
    return {
        "schema_version": "dronedream.depth-fault-recovery-verification.v1",
        "status": "verified" if all(gates.values()) else "failed",
        "development_only": True,
        "flight_qualification_granted": False,
        "source": {
            "run_root": str(run_root.resolve()),
            "qualification_summary_sha256": _sha256(summary_path),
            "fault_evidence_sha256": _sha256(fault_path),
            "executor_history_sha256": _sha256(executor_history_path),
            "depth_history_sha256": _sha256(depth_history_path),
        },
        "gates": gates,
        "measurements": {
            "activation_at_unix_ms": activation_ms,
            "recovery_at_unix_ms": recovery_ms,
            "configured_duration_ms": configured_duration_ms,
            "observed_fault_window_ms": recovery_ms - activation_ms,
            "observed_fresh_depth_gap_ms": observed_gap_ms,
            "pre_fault_model_authorized_continue_count": len(before),
            "fault_window_hold_or_unavailable_count": len(fail_closed_during),
            "fault_window_unsafe_continue_count": len(unsafe_during),
            "post_recovery_model_authorized_continue_count": len(after),
            "model_provider_call_count": int(
                model_navigation.get("provider_call_count", 0)
            ),
            "model_authorized_control_applied_count": int(
                model_navigation.get("authorized_control_applied_count", 0)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = verify_depth_fault_recovery(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
