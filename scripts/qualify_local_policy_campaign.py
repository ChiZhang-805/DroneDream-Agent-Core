#!/usr/bin/env python3
"""Aggregate independent Gazebo/PX4 trials into a policy qualification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.contracts import LocalExpertInferenceTrace
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import route_local_experts
from dronedream_agent_core.local_policy_packages import (
    LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE,
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    PolicyArtifactRole,
    latency_percentile,
    load_local_policy_package,
    local_policy_receipt_supports_control_contract,
)
from dronedream_agent_core.realtime_feature_encoders import (
    POLICY_REALTIME_FEATURE_COUNT,
    REALTIME_CONTROL_FEATURE_COUNT,
    RealtimeFeatureSnapshot,
)

MINIMUM_QUALIFICATION_TRIALS = 20
REQUIRED_QUALIFICATION_GATES = frozenset(
    {
        "controlled_vehicle_entity_selected",
        "controlled_vehicle_identity_confirmed",
        "development_fault_injection_absent",
        "executor_completed",
        "goal_observed",
        "landing_confirmed",
        "live_depth_metric_map_present",
        "live_depth_perception_healthy",
        "live_depth_safety_history_present",
        "local_policy_simulation_admission_recorded",
        "local_safety_command_present",
        "local_safety_observation_present",
        "local_safety_observation_sequence_advanced",
        "model_navigation_authorized_control_applied",
        "model_navigation_authorized_schedule_advance_recorded",
        "model_navigation_control_authority_required",
        "model_navigation_cycle_recorded",
        "model_navigation_primary_provider_call_recorded",
        "model_navigation_provider_cadence_bounded",
        "model_navigation_invocation_failure_absent",
        "model_navigation_fresh_revalidation_stable",
        "model_navigation_provider_call_recorded",
        "model_navigation_route_fallback_absent",
        "model_navigation_snapshot_recorded",
        "native_terminal_lifecycle_published",
        "no_live_abort",
        "offboard_timing_complete",
        "px4_ulog_present",
        "ros_observations_present",
        "runtime_pose_samples_present",
        "static_route_clearance_bound",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _snapshot_identity(snapshot: dict[str, object]) -> str:
    payload = dict(snapshot)
    observed = str(payload.pop("snapshot_sha256", ""))
    if not observed or observed != sha256_json(payload):
        raise ValueError("qualification snapshot hash mismatch")
    return observed


def _model_stack(model: object, package_id: str) -> tuple[str, ...]:
    value = str(model or "")
    prefix = package_id + "/"
    if not value.startswith(prefix):
        return ()
    return tuple(item for item in value[len(prefix) :].split("+") if item)


def _expert_invocations(
    run_dir: Path,
    *,
    package_id: str,
    package_roles: set[PolicyArtifactRole],
    realtime_feature_count: int | None,
) -> tuple[
    Counter[PolicyArtifactRole],
    Counter[PolicyArtifactRole],
    list[str],
    int,
    int,
    int,
    int,
]:
    snapshot_path = run_dir / "model-navigation-snapshots.jsonl"
    cycle_path = run_dir / "model-navigation-cycles.jsonl"
    calls_path = run_dir / "model-navigation-model-calls.jsonl"
    for path in (snapshot_path, cycle_path, calls_path):
        if not path.is_file():
            raise FileNotFoundError(f"expert audit evidence is missing: {path}")
    snapshots: dict[str, dict[str, object]] = {}
    for record in _jsonl(snapshot_path):
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        snapshots[_snapshot_identity(snapshot)] = snapshot
    local_calls = {
        str(call.get("call_id")): call
        for call in _jsonl(calls_path)
        if call.get("provider") == "local-policy" and call.get("call_id")
    }
    expected: Counter[PolicyArtifactRole] = Counter()
    observed: Counter[PolicyArtifactRole] = Counter()
    issues: list[str] = []
    linked_call_ids: set[str] = set()
    local_policy_call_count = 0
    realtime_feature_ready_call_count = 0
    model_motion_decision_count = 0
    pilot_control_command_count = 0
    for cycle in _jsonl(cycle_path):
        call_id = str(cycle.get("model_call_id") or "")
        if not call_id and cycle.get("model_action") == "not-invoked":
            # A fail-closed sensor or freshness gate is a legitimate cycle
            # outcome and intentionally has no provider call to link.  It must
            # remain visible in the flight evidence without invalidating the
            # audit of calls that actually crossed the Harness boundary.
            continue
        call = local_calls.get(call_id)
        if call is None:
            issues.append("TRIAL_EXPERT_CALL_LINK_MISSING")
            continue
        if call_id in linked_call_ids:
            issues.append("TRIAL_EXPERT_CALL_LINK_DUPLICATED")
            continue
        linked_call_ids.add(call_id)
        local_policy_call_count += 1
        snapshot = snapshots.get(str(cycle.get("snapshot_sha256") or ""))
        if snapshot is None:
            issues.append("TRIAL_EXPERT_SNAPSHOT_LINK_MISSING")
            continue
        if realtime_feature_count is not None:
            try:
                realtime_snapshot = RealtimeFeatureSnapshot.model_validate(
                    snapshot.get("realtime_feature_snapshot")
                )
            except (TypeError, ValueError):
                issues.append("TRIAL_REALTIME_FEATURE_SNAPSHOT_INVALID")
            else:
                if (
                    realtime_snapshot.ready_for_control
                    and not realtime_snapshot.issue_codes
                    # The 10 mission-reference features are appended by the
                    # policy adapter, not produced by the sensor encoders.
                    and realtime_feature_count in {
                        REALTIME_CONTROL_FEATURE_COUNT, POLICY_REALTIME_FEATURE_COUNT,
                    }
                    and len(realtime_snapshot.required_roles) == 3
                    and (
                        realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
                        or realtime_snapshot.policy_feature_contract_sha256()
                        == CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                    )
                    and len(realtime_snapshot.fused_features) == REALTIME_CONTROL_FEATURE_COUNT
                    and len(realtime_snapshot.fused_valid_mask) == REALTIME_CONTROL_FEATURE_COUNT
                ):
                    realtime_feature_ready_call_count += 1
                else:
                    issues.append("TRIAL_REALTIME_FEATURE_SNAPSHOT_NOT_READY")
        routing = route_local_experts(snapshot, available_roles=package_roles)
        expected_stack: tuple[str, ...] = (
            routing.selected_navigation_role,
            *(("perception-encoder",) if "perception-encoder" in package_roles else ()),
            *routing.advisory_roles,
        )
        invoked_stack = _model_stack(call.get("model"), package_id)
        if invoked_stack != expected_stack:
            issues.append("TRIAL_EXPERT_INVOCATION_STACK_MISMATCH")
        trace_payload = call.get("local_expert_trace")
        if (
            realtime_feature_count is not None
            and isinstance(trace_payload, dict)
            and trace_payload.get("navigation_action") == "pilot-control"
        ):
            model_motion_decision_count += 1
            if not isinstance(trace_payload.get("pilot_control"), dict):
                issues.append("TRIAL_PILOT_CONTROL_COMMAND_MISSING")
            else:
                pilot_control_command_count += 1
        try:
            trace = LocalExpertInferenceTrace.model_validate(trace_payload)
        except (TypeError, ValueError):
            issues.append("TRIAL_EXPERT_INFERENCE_TRACE_INVALID")
        else:
            if (
                trace.requested_navigation_role != routing.requested_navigation_role
                or trace.selected_navigation_role != routing.selected_navigation_role
                or trace.fallback_used != routing.fallback_used
            ):
                issues.append("TRIAL_EXPERT_INFERENCE_TRACE_ROUTING_MISMATCH")
            if set(trace.advisory_risk_scores) != set(routing.advisory_roles):
                issues.append("TRIAL_EXPERT_INFERENCE_TRACE_ADVISOR_MISMATCH")
        for role in expected_stack:
            if role in package_roles:
                expected[role] += 1  # type: ignore[index]
        for role in invoked_stack:
            if role not in package_roles:
                issues.append("TRIAL_UNKNOWN_EXPERT_IN_INVOCATION_STACK")
                continue
            observed[role] += 1  # type: ignore[index]
    if set(local_calls) != linked_call_ids:
        issues.append("TRIAL_UNLINKED_LOCAL_POLICY_CALL")
    return (
        expected,
        observed,
        issues,
        local_policy_call_count,
        realtime_feature_ready_call_count,
        model_motion_decision_count,
        pilot_control_command_count,
    )


def _safety_intervention_episodes(path: Path) -> int:
    if not path.is_file():
        return 0
    episodes = 0
    intervening = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        command = record.get("command") if isinstance(record, dict) else None
        decision = command.get("decision") if isinstance(command, dict) else None
        action = decision.get("action") if isinstance(decision, dict) else None
        active = action in {"slow", "hold", "replan"}
        if active and not intervening:
            episodes += 1
        intervening = active
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--simulation-admission", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-trials", type=int, default=MINIMUM_QUALIFICATION_TRIALS)
    parser.add_argument("--minimum-distinct-maps", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.minimum_trials < MINIMUM_QUALIFICATION_TRIALS:
        parser.error(f"--minimum-trials cannot be below {MINIMUM_QUALIFICATION_TRIALS}")
    if len({path.resolve() for path in args.run_dir}) != len(args.run_dir):
        parser.error("qualification run directories must be unique")

    package = load_local_policy_package(args.package)
    admission = LocalPolicySimulationAdmissionReceipt.model_validate_json(
        args.simulation_admission.read_text(encoding="utf-8")
    )
    if not admission.admitted_to_simulation:
        raise ValueError("policy was not admitted to simulation")
    if admission.policy_package_sha256 != package.package_sha256:
        raise ValueError("simulation admission does not bind the selected package")
    required_distinct_maps = args.minimum_distinct_maps
    scope_minimum_maps = 2 if package.manifest.scope == "general" else 1
    if required_distinct_maps is None:
        required_distinct_maps = scope_minimum_maps
    if required_distinct_maps < scope_minimum_maps:
        parser.error(
            f"--minimum-distinct-maps cannot be below {scope_minimum_maps} "
            f"for {package.manifest.scope} policies"
        )

    issues: list[str] = []
    if admission.admission_scope != "standard-simulation":
        issues.append("BOOTSTRAP_SIMULATION_ADMISSION_NOT_QUALIFICATION_ELIGIBLE")
    if "risk-critic" not in package.artifact_paths:
        issues.append("INDEPENDENT_RISK_CRITIC_REQUIRED")
    continuous_control = package.manifest.pilot_control_mode is not None
    if not local_policy_receipt_supports_control_contract(package, admission):
        issues.append("SIMULATION_ADMISSION_EXPERT_CONTROL_EVIDENCE_INVALID")
    if continuous_control and (
        admission.realtime_feature_ready_sample_count != admission.sample_count
        or admission.pilot_control_target_sample_count != admission.sample_count
        or admission.pilot_control_mean_absolute_error is None
        or admission.pilot_control_mean_absolute_error > LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE
    ):
        issues.append("SIMULATION_ADMISSION_CONTINUOUS_CONTROL_EVIDENCE_INVALID")
    successful = 0
    collisions = 0
    interventions = 0
    latencies: list[float] = []
    expert_expected: Counter[PolicyArtifactRole] = Counter()
    expert_observed: Counter[PolicyArtifactRole] = Counter()
    local_policy_call_count = 0
    realtime_feature_ready_call_count = 0
    model_motion_decision_count = 0
    pilot_control_command_count = 0
    trial_evidence_hashes: list[str] = []
    semantic_hashes: set[str] = set()
    for run_dir in args.run_dir:
        evidence_path = run_dir / "mission_evidence.json"
        selection_path = run_dir / "local-policy-selection.json"
        calls_path = run_dir / "model-navigation-model-calls.jsonl"
        if not evidence_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError(f"incomplete local-policy trial: {run_dir}")
        evidence = _json(evidence_path)
        selection = _json(selection_path)
        (
            trial_expected,
            trial_observed,
            trial_expert_issues,
            trial_local_policy_calls,
            trial_realtime_ready_calls,
            trial_motion_decisions,
            trial_pilot_commands,
        ) = _expert_invocations(
            run_dir,
            package_id=package.manifest.package_id,
            package_roles=set(package.artifact_paths),
            realtime_feature_count=package.manifest.realtime_feature_count,
        )
        expert_expected.update(trial_expected)
        expert_observed.update(trial_observed)
        issues.extend(trial_expert_issues)
        local_policy_call_count += trial_local_policy_calls
        realtime_feature_ready_call_count += trial_realtime_ready_calls
        model_motion_decision_count += trial_motion_decisions
        pilot_control_command_count += trial_pilot_commands
        safety_path = run_dir / "depth-local-safety-history.jsonl"
        trial_artifact_hashes = {
            "mission_evidence_sha256": _sha256(evidence_path),
            "selection_sha256": _sha256(selection_path),
            "model_calls_sha256": _sha256(calls_path) if calls_path.is_file() else None,
            "safety_history_sha256": (_sha256(safety_path) if safety_path.is_file() else None),
        }
        trial_evidence_hashes.append(sha256_json(trial_artifact_hashes))
        if (
            selection.get("package_sha256") != package.package_sha256
            or selection.get("simulation_only") is not True
            or selection.get("qualification_receipt_id") != admission.receipt_id
        ):
            raise ValueError(f"trial selected a different or qualified policy: {run_dir}")
        gates = evidence.get("gates")
        measurements = evidence.get("measurements")
        artifacts = evidence.get("artifacts")
        if not isinstance(gates, dict) or not isinstance(measurements, dict):
            raise ValueError(f"trial evidence is missing gates or measurements: {run_dir}")
        if isinstance(artifacts, dict) and isinstance(artifacts.get("semantic_sha256"), str):
            semantic_hashes.add(str(artifacts["semantic_sha256"]))
        trial_collision = measurements.get("live_safety_event") is not None
        collisions += int(trial_collision)
        missing_or_failed_gates = sorted(
            gate for gate in REQUIRED_QUALIFICATION_GATES if gates.get(gate) is not True
        )
        if missing_or_failed_gates:
            issues.append("TRIAL_REQUIRED_GATE_FAILED")
        navigation = measurements.get("model_navigation")
        navigation_valid = bool(
            isinstance(navigation, dict)
            and navigation.get("provider") == "local-policy"
            and isinstance(navigation.get("primary_provider_call_count"), int)
            and navigation["primary_provider_call_count"] > 0
            and navigation.get("fallback_provider_call_count") == 0
            and isinstance(navigation.get("authorized_control_applied_count"), int)
            and navigation["authorized_control_applied_count"] > 0
        )
        if not navigation_valid:
            issues.append("TRIAL_MODEL_AUTHORITY_EVIDENCE_INVALID")
        trial_passed = bool(
            evidence.get("status") == "verified"
            and not missing_or_failed_gates
            and navigation_valid
            and not trial_collision
        )
        successful += int(trial_passed)
        interventions += _safety_intervention_episodes(safety_path)
        trial_call_count = 0
        if calls_path.is_file():
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                call = json.loads(line)
                if call.get("provider") != "local-policy":
                    continue
                latency = call.get("latency_ms")
                if isinstance(latency, int | float) and latency >= 0:
                    latencies.append(float(latency))
                    trial_call_count += 1
        if trial_call_count == 0:
            issues.append("TRIAL_WITHOUT_LOCAL_POLICY_CALLS")

    trial_count = len(args.run_dir)
    if len(set(trial_evidence_hashes)) != trial_count:
        issues.append("DUPLICATE_TRIAL_EVIDENCE")
    if trial_count < args.minimum_trials:
        issues.append("INSUFFICIENT_TRIALS")
    if len(semantic_hashes) < required_distinct_maps:
        issues.append("INSUFFICIENT_MAP_DIVERSITY")
    for role in package.artifact_paths:
        if expert_expected[role] == 0:
            issues.append(f"EXPERT_ROLE_NOT_EXERCISED_{role.upper()}")
        elif expert_observed[role] != expert_expected[role]:
            issues.append(f"EXPERT_ROLE_INVOCATION_COUNT_MISMATCH_{role.upper()}")
    if collisions:
        issues.append("COLLISION_OBSERVED")
    if successful / trial_count < 0.95:
        issues.append("SUCCESS_RATE_TOO_LOW")
    if not latencies:
        issues.append("MISSING_INFERENCE_LATENCY_EVIDENCE")
        latencies = [0.0]
    p50 = latency_percentile(latencies, 50.0)
    p95 = latency_percentile(latencies, 95.0)
    p99 = latency_percentile(latencies, 99.0)
    if p99 > package.manifest.maximum_inference_latency_ms:
        issues.append("INFERENCE_LATENCY_TOO_HIGH")
    if continuous_control:
        if realtime_feature_ready_call_count != local_policy_call_count:
            issues.append("REALTIME_FEATURE_READY_RATE_BELOW_ONE")
        if model_motion_decision_count == 0:
            issues.append("PILOT_CONTROL_MOTION_DECISION_MISSING")
        if pilot_control_command_count != model_motion_decision_count:
            issues.append("PILOT_CONTROL_COMMAND_COVERAGE_INCOMPLETE")
    issues = sorted(set(issues))
    suite_payload = {
        "script_sha256": _sha256(Path(__file__)),
        "simulation_admission_sha256": _sha256(args.simulation_admission),
        "trial_evidence_sha256": sorted(trial_evidence_hashes),
        "required_distinct_maps": required_distinct_maps,
        "minimum_trials": args.minimum_trials,
    }
    receipt_payload = {
        "policy_package_sha256": package.package_sha256,
        "evaluation_suite_sha256": sha256_json(suite_payload),
        "map_sha256": package.manifest.map_sha256,
        "vehicle_sha256": package.manifest.vehicle_sha256,
        "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
        "trial_count": trial_count,
        "successful_trial_count": successful,
        "collision_count": collisions,
        "deterministic_safety_intervention_count": interventions,
        "p50_inference_latency_ms": p50,
        "p95_inference_latency_ms": p95,
        "p99_inference_latency_ms": p99,
        "expert_expected_invocation_counts": dict(sorted(expert_expected.items())),
        "expert_observed_invocation_counts": dict(sorted(expert_observed.items())),
        "local_policy_call_count": (local_policy_call_count if continuous_control else None),
        "realtime_feature_ready_call_count": (
            realtime_feature_ready_call_count if continuous_control else None
        ),
        "model_motion_decision_count": (
            model_motion_decision_count if continuous_control else None
        ),
        "pilot_control_command_count": (
            pilot_control_command_count if continuous_control else None
        ),
        "pilot_control_mean_absolute_error": (
            admission.pilot_control_mean_absolute_error if continuous_control else None
        ),
        "navigation_expert_metrics": {
            role: metrics.model_dump(mode="json")
            for role, metrics in admission.navigation_expert_metrics.items()
        },
        "qualified": not issues,
        "issue_codes": issues,
    }
    receipt = LocalPolicyQualificationReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id="policy-qualification-" + sha256_json(receipt_payload)[:32],
        **receipt_payload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(receipt.model_dump_json())
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
