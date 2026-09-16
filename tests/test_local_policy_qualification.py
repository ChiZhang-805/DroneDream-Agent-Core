from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from control_fixtures import complete_feature_snapshot, qualified_pilot_metrics

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import (
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
)
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT

VEHICLE_SHA256 = "a" * 64
SENSOR_SHA256 = "b" * 64


def _write_package(root: Path, *, continuous_control: bool = False) -> object:
    root.mkdir()
    artifacts = []
    for role, name in (
        ("local-navigation-policy", "navigation.onnx"),
        ("risk-critic", "risk.onnx"),
    ):
        content = f"{role}-fixture".encode()
        (root / name).write_bytes(content)
        artifacts.append(
            {
                "role": role,
                "relative_path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "format": "onnx",
                "input_names": [
                    "state_features",
                    *(["proposed_control"] if continuous_control and role == "risk-critic"
                      else ["candidate_features", "candidate_mask"]),
                    *(
                        ["realtime_features", "realtime_valid_mask"]
                        if continuous_control
                        else []
                    ),
                ],
                "output_names": (
                    [
                        "candidate_scores",
                        "action_scores",
                        "risk_score",
                        *(["pilot_control"] if continuous_control else []),
                    ]
                    if role == "local-navigation-policy"
                    else ["risk_score"]
                ),
            }
        )
    manifest = {
        "package_id": "local.qualification-fixture",
        "display_name": "Qualification Fixture",
        "scope": "general",
        "vehicle_sha256": VEHICLE_SHA256,
        "sensor_contract_sha256": SENSOR_SHA256,
        "maximum_inference_latency_ms": 100,
        "realtime_feature_count": (
            POLICY_REALTIME_FEATURE_COUNT if continuous_control else None
        ),
        "control_feature_contract_sha256": (
            CURRENT_POLICY_FEATURE_CONTRACT_SHA256 if continuous_control else None
        ),
        "pilot_control_mode": (
            "normalized-body-velocity" if continuous_control else None
        ),
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_local_policy_package(root)


def _required_gates() -> dict[str, bool]:
    # Keep this explicit so a new production gate fails the fixture until the
    # qualification test is intentionally updated.
    return {
        "controlled_vehicle_entity_selected": True,
        "controlled_vehicle_identity_confirmed": True,
        "development_fault_injection_absent": True,
        "executor_completed": True,
        "goal_observed": True,
        "landing_confirmed": True,
        "live_depth_metric_map_present": True,
        "live_depth_perception_healthy": True,
        "live_depth_safety_history_present": True,
        "local_policy_simulation_admission_recorded": True,
        "local_safety_command_present": True,
        "local_safety_observation_present": True,
        "local_safety_observation_sequence_advanced": True,
        "model_navigation_authorized_control_applied": True,
        "model_navigation_authorized_schedule_advance_recorded": True,
        "model_navigation_control_authority_required": True,
        "model_navigation_cycle_recorded": True,
        "model_navigation_primary_provider_call_recorded": True,
        "model_navigation_provider_cadence_bounded": True,
        "model_navigation_invocation_failure_absent": True,
        "model_navigation_fresh_revalidation_stable": True,
        "model_navigation_provider_call_recorded": True,
        "model_navigation_route_fallback_absent": True,
        "model_navigation_snapshot_recorded": True,
        "native_terminal_lifecycle_published": True,
        "no_live_abort": True,
        "offboard_timing_complete": True,
        "px4_ulog_present": True,
        "ros_observations_present": True,
        "runtime_pose_samples_present": True,
        "static_route_clearance_bound": True,
    }


def _write_trial(
    root: Path,
    *,
    package: object,
    admission: LocalPolicySimulationAdmissionReceipt,
    index: int,
    duplicate_content: bool = False,
    omit_risk_from_stack: bool = False,
    omit_trace: bool = False,
    continuous_control: bool = False,
    omit_pilot_control: bool = False,
) -> None:
    root.mkdir()
    content_index = 0 if duplicate_content else index
    evidence = {
        "schema_version": "dronedream.generic-px4-gazebo-run.v1",
        "status": "verified",
        "trial_id": f"trial-{content_index}",
        "gates": _required_gates(),
        "measurements": {
            "live_safety_event": None,
            "model_navigation": {
                "provider": "local-policy",
                "primary_provider_call_count": 100,
                "fallback_provider_call_count": 0,
                "authorized_control_applied_count": 95,
            },
        },
        "artifacts": {"semantic_sha256": ("c" if content_index % 2 == 0 else "d") * 64},
    }
    (root / "mission_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    selection = {
        "package_sha256": package.package_sha256,  # type: ignore[attr-defined]
        "simulation_only": True,
        "qualification_receipt_id": admission.receipt_id,
    }
    (root / "local-policy-selection.json").write_text(json.dumps(selection), encoding="utf-8")
    call_id = f"call-{content_index}"
    model_stack = f"{package.manifest.package_id}/local-navigation-policy"  # type: ignore[attr-defined]
    if not omit_risk_from_stack:
        model_stack += "+risk-critic"
    call = {
        "call_id": call_id,
        "role": "local_navigation_advisor",
        "provider": "local-policy",
        "model": model_stack,
        "latency_ms": 1 + content_index % 3,
    }
    if not omit_trace:
        call["local_expert_trace"] = {
            "requested_navigation_role": "local-navigation-policy",
            "selected_navigation_role": "local-navigation-policy",
            "fallback_used": False,
            "navigation_action": (
                "pilot-control" if continuous_control else "hold"
            ),
            "navigation_selected_candidate_id": None,
            "candidate_scores": [0.0] * 8,
            "action_scores": (
                [-1.0, -2.0, -3.0, 2.0]
                if continuous_control
                else [1.0, 0.0, -1.0]
            ),
            "navigation_risk_score": 0.1,
            "advisory_risk_scores": {"risk-critic": 0.1},
            "aggregate_risk_score": 0.1,
            "controller_step_scale": 1.0,
            **(
                {
                    "pilot_control": {
                        "forward_axis": 0.4,
                        "right_axis": 0.0,
                        "up_axis": 0.0,
                        "yaw_axis": 0.1,
                    }
                }
                if continuous_control and not omit_pilot_control
                else {}
            ),
        }
    (root / "model-navigation-model-calls.jsonl").write_text(
        json.dumps(call)
        + "\n",
        encoding="utf-8",
    )
    snapshot: dict[str, object] = {
        "strategic_context": {
            "task": {
                "phase": "TRACK",
                "control_profile": "transit",
                "decision_trigger": "periodic",
            }
        },
        "trial_marker": content_index,
    }
    if continuous_control:
        snapshot["realtime_feature_snapshot"] = complete_feature_snapshot(1).model_dump(mode="json")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    (root / "model-navigation-snapshots.jsonl").write_text(
        json.dumps({"snapshot": snapshot}) + "\n",
        encoding="utf-8",
    )
    (root / "model-navigation-cycles.jsonl").write_text(
        json.dumps(
            {
                "model_call_id": call_id,
                "snapshot_sha256": snapshot["snapshot_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "model-navigation-cycles.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "model_action": "not-invoked",
                    "model_call_id": None,
                    "hold_reason": "PERCEPTION_STREAM_NOT_HEALTHY",
                }
            )
            + "\n"
        )
    (root / "depth-local-safety-history.jsonl").write_text(
        json.dumps({"command": {"decision": {"action": "continue"}}}) + "\n",
        encoding="utf-8",
    )


def _run_qualification(
    tmp_path: Path,
    *,
    duplicate_content: bool,
    omit_risk_on_last: bool = False,
    inject_development_fault_on_last: bool = False,
    inject_model_invocation_failure_on_last: bool = False,
    inject_unstable_fresh_revalidation_on_last: bool = False,
    omit_trace_on_last: bool = False,
    bootstrap_admission: bool = False,
    continuous_control: bool = False,
    omit_pilot_control_on_last: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    package = _write_package(
        tmp_path / "package", continuous_control=continuous_control
    )
    admission = LocalPolicySimulationAdmissionReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id="policy-simulation-admission-" + "1" * 32,
        policy_package_sha256=package.package_sha256,  # type: ignore[attr-defined]
        dataset_receipt_sha256="e" * 64,
        evaluation_suite_sha256="f" * 64,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        sample_count=100,
        motion_sample_count=50,
        non_motion_sample_count=50,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=1.0,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.01,
        p50_inference_latency_ms=1.0,
        p95_inference_latency_ms=2.0,
        p99_inference_latency_ms=3.0,
        realtime_feature_ready_sample_count=(100 if continuous_control else None),
        pilot_control_target_sample_count=(100 if continuous_control else None),
        pilot_control_mean_absolute_error=(0.05 if continuous_control else None),
        navigation_expert_metrics=(
            {"local-navigation-policy": qualified_pilot_metrics()} if continuous_control else {}
        ),
        navigation_training_dataset_receipt_sha256=(
            "9" * 64 if bootstrap_admission else None
        ),
        navigation_admission_split=("validation" if bootstrap_admission else None),
        navigation_admission_source_evidence_sha256=(
            ["8" * 64] if bootstrap_admission else []
        ),
        admission_scope=(
            "recovery-bootstrap-simulation"
            if bootstrap_admission
            else "standard-simulation"
        ),
        minimum_non_motion_sample_count=(3 if bootstrap_admission else 20),
        admitted_to_simulation=True,
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(admission.model_dump_json(), encoding="utf-8")
    runs = []
    for index in range(20):
        run = tmp_path / f"trial-{index}"
        _write_trial(
            run,
            package=package,
            admission=admission,
            index=index,
            duplicate_content=duplicate_content,
            omit_risk_from_stack=omit_risk_on_last and index == 19,
            omit_trace=omit_trace_on_last and index == 19,
            continuous_control=continuous_control,
            omit_pilot_control=omit_pilot_control_on_last and index == 19,
        )
        if inject_development_fault_on_last and index == 19:
            evidence_path = run / "mission_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["gates"]["development_fault_injection_absent"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        if inject_model_invocation_failure_on_last and index == 19:
            evidence_path = run / "mission_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["gates"]["model_navigation_invocation_failure_absent"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        if inject_unstable_fresh_revalidation_on_last and index == 19:
            evidence_path = run / "mission_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["gates"]["model_navigation_fresh_revalidation_stable"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        runs.append(run)
    output = tmp_path / "qualification.json"
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repository / "scripts" / "qualify_local_policy_campaign.py"),
        "--package",
        str(package.root),  # type: ignore[attr-defined]
        "--simulation-admission",
        str(admission_path),
        "--output",
        str(output),
    ]
    for run in runs:
        command.extend(("--run-dir", str(run)))
    result = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_qualification_accepts_twenty_distinct_closed_loop_trials(
    tmp_path: Path,
) -> None:
    result, receipt = _run_qualification(tmp_path, duplicate_content=False)

    assert result.returncode == 0, result.stderr
    assert receipt["qualified"] is True
    assert receipt["trial_count"] == 20
    assert receipt["successful_trial_count"] == 20


def test_continuous_control_qualification_requires_live_features_and_pilot_commands(
    tmp_path: Path,
) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        continuous_control=True,
    )

    assert result.returncode == 0, result.stderr
    assert receipt["qualified"] is True
    assert receipt["local_policy_call_count"] == 20
    assert receipt["realtime_feature_ready_call_count"] == 20
    assert receipt["model_motion_decision_count"] == 20
    assert receipt["pilot_control_command_count"] == 20
    assert receipt["pilot_control_mean_absolute_error"] == 0.05


def test_continuous_control_qualification_rejects_a_missing_pilot_command(
    tmp_path: Path,
) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        continuous_control=True,
        omit_pilot_control_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_PILOT_CONTROL_COMMAND_MISSING" in receipt["issue_codes"]
    assert "PILOT_CONTROL_COMMAND_COVERAGE_INCOMPLETE" in receipt["issue_codes"]


def test_qualification_rejects_recovery_bootstrap_admission(
    tmp_path: Path,
) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        bootstrap_admission=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert (
        "BOOTSTRAP_SIMULATION_ADMISSION_NOT_QUALIFICATION_ELIGIBLE"
        in receipt["issue_codes"]
    )


def test_qualification_rejects_copied_trial_evidence(tmp_path: Path) -> None:
    result, receipt = _run_qualification(tmp_path, duplicate_content=True)

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "DUPLICATE_TRIAL_EVIDENCE" in receipt["issue_codes"]
    assert "INSUFFICIENT_MAP_DIVERSITY" in receipt["issue_codes"]


def test_qualification_rejects_a_missing_required_expert_invocation(
    tmp_path: Path,
) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        omit_risk_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_EXPERT_INVOCATION_STACK_MISMATCH" in receipt["issue_codes"]
    assert (
        "EXPERT_ROLE_INVOCATION_COUNT_MISMATCH_RISK-CRITIC"
        in receipt["issue_codes"]
    )
    assert receipt["expert_expected_invocation_counts"]["risk-critic"] == 20
    assert receipt["expert_observed_invocation_counts"]["risk-critic"] == 19


def test_qualification_rejects_development_fault_injection(tmp_path: Path) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        inject_development_fault_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_REQUIRED_GATE_FAILED" in receipt["issue_codes"]


def test_qualification_rejects_model_invocation_failure(tmp_path: Path) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        inject_model_invocation_failure_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_REQUIRED_GATE_FAILED" in receipt["issue_codes"]


def test_qualification_rejects_unstable_fresh_revalidation(tmp_path: Path) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        inject_unstable_fresh_revalidation_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_REQUIRED_GATE_FAILED" in receipt["issue_codes"]


def test_qualification_rejects_missing_expert_vote_trace(tmp_path: Path) -> None:
    result, receipt = _run_qualification(
        tmp_path,
        duplicate_content=False,
        omit_trace_on_last=True,
    )

    assert result.returncode == 1
    assert receipt["qualified"] is False
    assert "TRIAL_EXPERT_INFERENCE_TRACE_INVALID" in receipt["issue_codes"]
