from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample
from scripts.build_local_policy_dataset import (
    _collect_samples,
    _exclude_cross_split_visual_duplicates,
    _filter_expert_samples,
    _pilot_control_from_trace,
    _recovery_episode_outcomes,
    _teacher_control_label,
    _verified_source_receipt,
)


def test_retired_teacher_proposal_without_actual_execution_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="EXECUTION_RECEIPT_REQUIRED"):
        _teacher_control_label(
            {},
            {
                "command": {
                    "decision": {
                        "selected_velocity_mps": {"x": 0.4, "y": 0.0, "z": 0.0},
                    }
                }
            },
            route_speed_limit_mps=0.8,
        )


def test_candidate_trace_cannot_be_relabelled_as_direct_pilot_control() -> None:
    assert (
        _pilot_control_from_trace(
            {
                "navigation_action": "select-candidate",
                "navigation_selected_candidate_id": "candidate-old",
            },
            target_action_index=0,
        )
        is None
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_policy_training_sample_rejects_non_finite_features(invalid: float) -> None:
    state = [0.0] * 46
    state[0] = invalid

    with pytest.raises(ValueError, match="finite number"):
        LocalPolicyTrainingSample(
            state_features=state,
            candidate_features=[[0.0] * 15 for _ in range(8)],
            candidate_mask=[1.0] + [0.0] * 7,
            target_action_index=0,
            risk_target=0.0,
        )


def _write_campaign(root: Path, *, status: str) -> tuple[Path, Path]:
    root.mkdir()
    snapshots = root / "model-navigation-snapshots.jsonl"
    cycles = root / "model-navigation-cycles.jsonl"
    snapshots.write_text('{"snapshot": {}}\n', encoding="utf-8")
    cycles.write_text('{"model_action": "hold"}\n', encoding="utf-8")
    (root / "mission_evidence.json").write_text(
        json.dumps(
            {
                "status": status,
                "artifacts": {
                    "model_navigation_snapshots_sha256": _sha256(snapshots),
                    "model_navigation_cycles_sha256": _sha256(cycles),
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshots, cycles


def test_verified_dataset_source_is_content_bound(tmp_path: Path) -> None:
    snapshots, cycles = _write_campaign(tmp_path / "accepted", status="verified")

    receipt = _verified_source_receipt(snapshots, cycles)

    assert receipt["mission_status"] == "verified"
    assert len(str(receipt["mission_evidence_sha256"])) == 64


def test_failed_campaign_cannot_train_a_local_expert(tmp_path: Path) -> None:
    snapshots, cycles = _write_campaign(tmp_path / "failed", status="failed")

    with pytest.raises(RuntimeError, match="MISSION_NOT_VERIFIED"):
        _verified_source_receipt(snapshots, cycles)


def _write_traced_sample(
    root: Path,
    *,
    fallback: bool,
    phase: str = "HOVER",
    control_profile: str = "precision",
    decision_trigger: str = "periodic",
    requested_role: str = "precision-maneuver-policy",
    navigation_risk_score: float = 0.2,
    recovery_episode_id: str | None = None,
    recovery_follow_up_distance_m: float | None = None,
) -> tuple[Path, Path]:
    root.mkdir()
    snapshot = {
        "snapshot_sha256": "a" * 64,
        "current_position_m": {"x": 0.0, "y": 0.0, "z": 1.0},
        "current_velocity_mps": {"x": 0.0, "y": 0.0, "z": 0.0},
        "goal_position_m": {"x": 1.0, "y": 0.0, "z": 1.0},
        "goal_distance_m": 1.0,
        "local_radius_m": 8.0,
        "vehicle_envelope_m": {
            "body_radius": 0.4,
            "body_height": 0.5,
            "candidate_speed": 0.5,
            "required_local_clearance": 0.5,
        },
        "perception_health": {
            "stream_healthy": True,
            "stream_age_seconds": 0.0,
            "localization_covariance_m2": 0.01,
        },
        "egocentric_sectors": [],
        "authorized_candidate_paths": [
            {
                "candidate_id": "candidate-0123456789abcdef",
                "kind": "goal-direct",
                "endpoint_m": {"x": 1.0, "y": 0.0, "z": 1.0},
                "path_length_m": 1.0,
                "minimum_predicted_dynamic_clearance_m": 2.0,
                "time_to_minimum_dynamic_clearance_seconds": 2.0,
                "required_clearance_m": 0.5,
                "deterministic_metric_path_validated": True,
                "dynamic_path_validated": True,
            }
        ],
        "strategic_context": {
            "task": {
                "phase": phase,
                "control_profile": control_profile,
                "decision_trigger": decision_trigger,
                "navigation_goal_id": "goal-001",
                "recovery_episode_id": recovery_episode_id,
            }
        },
    }
    snapshots = root / "model-navigation-snapshots.jsonl"
    cycles = root / "model-navigation-cycles.jsonl"
    calls = root / "model-navigation-model-calls.jsonl"
    snapshot_records = [json.dumps({"snapshot": snapshot})]
    if recovery_follow_up_distance_m is not None:
        follow_up = json.loads(json.dumps(snapshot))
        follow_up["snapshot_sha256"] = "b" * 64
        follow_up["goal_distance_m"] = recovery_follow_up_distance_m
        follow_up["strategic_context"]["task"]["decision_trigger"] = "periodic"
        follow_up["strategic_context"]["task"]["recovery_episode_id"] = None
        snapshot_records.append(json.dumps({"snapshot": follow_up}))
    snapshots.write_text("\n".join(snapshot_records) + "\n", encoding="utf-8")
    cycles.write_text(
        json.dumps(
            {
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "model_call_id": "model-" + "1" * 24,
                "model_action": "hold",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls.write_text(
        json.dumps(
            {
                "call_id": "model-" + "1" * 24,
                "local_expert_trace": {
                    "requested_navigation_role": requested_role,
                    "selected_navigation_role": (
                        "local-navigation-policy" if fallback else requested_role
                    ),
                    "fallback_used": fallback,
                    "navigation_action": "select-candidate",
                    "navigation_selected_candidate_id": ("candidate-0123456789abcdef"),
                    "navigation_risk_score": navigation_risk_score,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshots, cycles


def test_expert_trace_labels_navigation_before_advisor_veto(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(tmp_path / "traced", fallback=False)

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
    )

    sample = next(iter(samples.values()))
    assert sample.navigation_expert_role == "precision-maneuver-policy"
    assert sample.target_action_index == 0
    assert sample.risk_target == 0.2


def test_expert_trace_does_not_mislabel_general_fallback_as_specialist(
    tmp_path: Path,
) -> None:
    snapshots, cycles = _write_traced_sample(tmp_path / "fallback", fallback=True)

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
    )

    assert samples == {}


def test_failed_invocation_is_excluded_without_inventing_expert_trace(
    tmp_path: Path,
) -> None:
    snapshots, cycles = _write_traced_sample(tmp_path / "failed-invocation", fallback=False)
    cycle = json.loads(cycles.read_text(encoding="utf-8"))
    cycle["model_action"] = "not-invoked"
    cycle["model_call_id"] = None
    cycles.write_text(json.dumps(cycle) + "\n", encoding="utf-8")
    cycles.with_name("model-navigation-model-calls.jsonl").write_text("", encoding="utf-8")

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
    )

    assert samples == {}


def test_expert_trace_fallback_can_only_be_used_as_explicit_bootstrap(
    tmp_path: Path,
) -> None:
    snapshots, cycles = _write_traced_sample(tmp_path / "fallback", fallback=True)

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
        allow_fallback_bootstrap=True,
    )

    sample = next(iter(samples.values()))
    assert sample.navigation_expert_role == "precision-maneuver-policy"
    assert sample.target_action_index == 0
    assert sample.risk_target == 0.2


def test_fallback_bootstrap_promotes_teacher_risk_veto_to_strong_target(
    tmp_path: Path,
) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "risky-fallback",
        fallback=True,
        navigation_risk_score=0.56,
    )

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
        allow_fallback_bootstrap=True,
    )

    assert next(iter(samples.values())).risk_target == 1.0


def test_fallback_bootstrap_requires_typed_trace(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(tmp_path / "fallback", fallback=True)

    with pytest.raises(ValueError, match="requires typed expert traces"):
        _collect_samples(
            [(snapshots, cycles)],
            allow_fallback_bootstrap=True,
        )


def test_normal_local_replan_cannot_bootstrap_recovery_expert(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "normal-local-replan",
        fallback=True,
        phase="LOCAL_REPLAN",
        decision_trigger="periodic",
        requested_role="recovery-policy",
    )

    with pytest.raises(RuntimeError, match="EXPERT_ROUTING_MISMATCH"):
        _collect_samples(
            [(snapshots, cycles)],
            require_expert_trace=True,
            allow_fallback_bootstrap=True,
        )


def test_recovery_bootstrap_requires_explicit_episode_identity(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "missing-episode",
        fallback=True,
        phase="TRACK",
        control_profile="transit",
        decision_trigger="progress-stalled",
        requested_role="recovery-policy",
    )

    with pytest.raises(RuntimeError, match="RECOVERY_EPISODE_ID_MISSING"):
        _collect_samples(
            [(snapshots, cycles)],
            require_expert_trace=True,
            allow_fallback_bootstrap=True,
            require_recovery_episode_identity=True,
        )


def test_recovery_bootstrap_accepts_hash_bound_episode_identity(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "episode-bound",
        fallback=True,
        phase="TRACK",
        control_profile="transit",
        decision_trigger="progress-stalled",
        requested_role="recovery-policy",
        recovery_episode_id="recovery-0123456789abcdef01234567",
        recovery_follow_up_distance_m=0.8,
    )

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
        allow_fallback_bootstrap=True,
        require_recovery_episode_identity=True,
    )

    assert len(samples) == 1


def test_recovery_bootstrap_rejects_episode_without_restored_progress(
    tmp_path: Path,
) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "no-restored-progress",
        fallback=True,
        phase="TRACK",
        control_profile="transit",
        decision_trigger="progress-stalled",
        requested_role="recovery-policy",
        recovery_episode_id="recovery-0123456789abcdef01234567",
        recovery_follow_up_distance_m=0.98,
    )

    with pytest.raises(RuntimeError, match="RECOVERY_OUTCOME_NOT_VERIFIED"):
        _collect_samples(
            [(snapshots, cycles)],
            require_expert_trace=True,
            allow_fallback_bootstrap=True,
            require_recovery_episode_identity=True,
        )


def test_recovery_bootstrap_admits_only_one_sample_per_episode(tmp_path: Path) -> None:
    snapshots, cycles = _write_traced_sample(
        tmp_path / "multi-cycle-episode",
        fallback=True,
        phase="TRACK",
        control_profile="transit",
        decision_trigger="progress-stalled",
        requested_role="recovery-policy",
        recovery_episode_id="recovery-0123456789abcdef01234567",
        recovery_follow_up_distance_m=0.8,
    )
    snapshot_records = [json.loads(line) for line in snapshots.read_text().splitlines()]
    second = json.loads(json.dumps(snapshot_records[0]))
    second["snapshot"]["snapshot_sha256"] = "c" * 64
    second["snapshot"]["goal_distance_m"] = 0.99
    snapshot_records.insert(1, second)
    snapshots.write_text(
        "".join(json.dumps(record) + "\n" for record in snapshot_records),
        encoding="utf-8",
    )
    second_call_id = "model-" + "2" * 24
    with cycles.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "snapshot_sha256": "c" * 64,
                    "model_call_id": second_call_id,
                    "model_action": "hold",
                }
            )
            + "\n"
        )
    calls = snapshots.parent / "model-navigation-model-calls.jsonl"
    second_call = json.loads(calls.read_text(encoding="utf-8").splitlines()[0])
    second_call["call_id"] = second_call_id
    second_call["local_expert_trace"]["navigation_action"] = "hold"
    second_call["local_expert_trace"]["navigation_selected_candidate_id"] = None
    second_call["local_expert_trace"]["navigation_risk_score"] = 0.56
    with calls.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second_call) + "\n")

    samples = _collect_samples(
        [(snapshots, cycles)],
        require_expert_trace=True,
        allow_fallback_bootstrap=True,
        require_recovery_episode_identity=True,
    )

    assert len(samples) == 1
    selected = next(iter(samples.values()))
    assert selected.target_action_index == 8
    assert selected.risk_target == 1.0


def test_far_away_goal_amendment_does_not_claim_recovery_success() -> None:
    episode_id = "recovery-0123456789abcdef01234567"

    outcomes = _recovery_episode_outcomes(
        [
            {
                "goal_distance_m": 4.0,
                "strategic_context": {
                    "task": {
                        "navigation_goal_id": "goal-before-amendment",
                        "recovery_episode_id": episode_id,
                    }
                },
            },
            {
                "goal_distance_m": 2.0,
                "strategic_context": {
                    "task": {
                        "navigation_goal_id": "goal-after-amendment",
                        "recovery_episode_id": None,
                    }
                },
            },
        ]
    )

    assert outcomes[episode_id] is False


def _visual_sample(*, snapshot: str, visual: str | None) -> LocalPolicyTrainingSample:
    return LocalPolicyTrainingSample(
        source_snapshot_sha256=snapshot * 64,
        source_visual_sha256=visual * 64 if visual is not None else None,
        state_features=[0.0] * 46,
        candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[1.0] + [0.0] * 7,
        target_action_index=0,
        risk_target=0.0,
    )


def test_specialist_filter_excludes_unrelated_navigation_roles() -> None:
    recovery = _visual_sample(snapshot="a", visual="1").model_copy(
        update={"navigation_expert_role": "recovery-policy"}
    )
    precision = _visual_sample(snapshot="b", visual="2").model_copy(
        update={"navigation_expert_role": "precision-maneuver-policy"}
    )

    filtered = _filter_expert_samples(
        {"recovery": recovery, "precision": precision},
        "recovery-policy",
    )

    assert filtered == {"recovery": recovery}
    assert _filter_expert_samples(filtered, None) == filtered


def test_cross_split_visual_duplicates_are_retained_only_for_training() -> None:
    training = [
        _visual_sample(snapshot="a", visual="1"),
        _visual_sample(snapshot="b", visual=None),
    ]
    validation = [
        _visual_sample(snapshot="c", visual="1"),
        _visual_sample(snapshot="d", visual="2"),
        _visual_sample(snapshot="e", visual=None),
    ]

    filtered, duplicate_hash_count, rejected_sample_count = _exclude_cross_split_visual_duplicates(
        training, validation
    )

    assert [sample.source_snapshot_sha256 for sample in filtered] == ["d" * 64, "e" * 64]
    assert duplicate_hash_count == 1
    assert rejected_sample_count == 1
