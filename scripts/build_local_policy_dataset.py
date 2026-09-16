#!/usr/bin/env python3
"""Distill hash-bound model navigation evidence into local policy samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from dronedream_agent_core.contracts import NormalizedPilotControl, RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import (
    NAVIGATION_EXPERT_ROLES,
    NavigationExpertRole,
    requested_navigation_expert,
)
from dronedream_agent_core.local_policy_port import compile_local_policy_features
from dronedream_agent_core.local_policy_training import (
    LOCAL_POLICY_LEGACY_ACTION_COUNT,
    LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX,
    LocalPolicyTrainingSample,
)
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.realtime_feature_encoders import (
    pilot_control_limits_for_profile,
)
from dronedream_agent_core.training.executed_control import executed_pilot_control


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _teacher_route_speed_limits(source_directory: Path) -> dict[str, float]:
    """Read the executed track's per-goal control envelope.

    The continuous target is normalized against the route speed that the
    runtime used, not against the much shorter legacy candidate-search speed.
    ``reference_track.json`` is hash-bound by the verified mission evidence,
    so this conversion cannot be changed after the teacher flight.
    """

    payload = json.loads((source_directory / "reference_track.json").read_text(encoding="utf-8"))
    points = payload.get("points")
    if payload.get("schema_version") != "dronedream.px4-track.v2" or not isinstance(points, list):
        raise RuntimeError("LOCAL_POLICY_TEACHER_REFERENCE_TRACK_INVALID")
    limits: dict[str, float] = {}
    for index, point in enumerate(points):
        speed = point.get("speed_limit_mps") if isinstance(point, dict) else None
        if not isinstance(speed, (int, float)) or not math.isfinite(float(speed)) or speed <= 0.0:
            raise RuntimeError("LOCAL_POLICY_TEACHER_REFERENCE_SPEED_INVALID")
        limits[f"source-waypoint-{index:04d}"] = float(speed)
    return limits


def _snapshot_navigation_goal_id(snapshot: dict[str, object]) -> str:
    strategic = snapshot.get("strategic_context")
    task = strategic.get("task") if isinstance(strategic, dict) else None
    goal_id = task.get("navigation_goal_id") if isinstance(task, dict) else None
    if not isinstance(goal_id, str) or not goal_id:
        raise RuntimeError("LOCAL_POLICY_TEACHER_NAVIGATION_GOAL_ID_MISSING")
    return goal_id


def _verified_source_receipt(
    snapshot_path: Path,
    cycle_path: Path,
    *,
    require_expert_trace: bool = False,
    teacher_control_history_path: Path | None = None,
) -> dict[str, object]:
    if snapshot_path.parent != cycle_path.parent:
        raise RuntimeError("LOCAL_POLICY_SOURCE_EVIDENCE_DIRECTORY_MISMATCH")
    evidence_path = snapshot_path.parent / "mission_evidence.json"
    if not evidence_path.is_file():
        raise RuntimeError("LOCAL_POLICY_SOURCE_MISSION_EVIDENCE_MISSING")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "verified":
        raise RuntimeError("LOCAL_POLICY_SOURCE_MISSION_NOT_VERIFIED")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("LOCAL_POLICY_SOURCE_ARTIFACT_RECEIPTS_MISSING")
    if artifacts.get("model_navigation_snapshots_sha256") != _sha256(snapshot_path):
        raise RuntimeError("LOCAL_POLICY_SOURCE_SNAPSHOT_HASH_MISMATCH")
    if artifacts.get("model_navigation_cycles_sha256") != _sha256(cycle_path):
        raise RuntimeError("LOCAL_POLICY_SOURCE_CYCLE_HASH_MISMATCH")
    receipt = {
        "mission_evidence_file": str(evidence_path.resolve()),
        "mission_evidence_sha256": _sha256(evidence_path),
        "mission_status": "verified",
    }
    if require_expert_trace:
        call_path = snapshot_path.parent / "model-navigation-model-calls.jsonl"
        if not call_path.is_file():
            raise RuntimeError("LOCAL_POLICY_SOURCE_MODEL_CALL_EVIDENCE_MISSING")
        if artifacts.get("model_navigation_calls_sha256") != _sha256(call_path):
            raise RuntimeError("LOCAL_POLICY_SOURCE_MODEL_CALL_HASH_MISMATCH")
        receipt.update(
            {
                "model_call_file": str(call_path.resolve()),
                "model_call_sha256": _sha256(call_path),
            }
        )
    if teacher_control_history_path is not None:
        applications_path = snapshot_path.parent / "runtime-state" / "control-applications.jsonl"
        if not applications_path.is_file():
            raise RuntimeError("LOCAL_POLICY_TEACHER_EXECUTION_RECEIPT_REQUIRED")
        if artifacts.get("control_applications_sha256") != _sha256(applications_path):
            raise RuntimeError("LOCAL_POLICY_TEACHER_EXECUTION_HASH_MISMATCH")
        receipt["teacher_control_applications_sha256"] = _sha256(applications_path)
        if teacher_control_history_path.parent != snapshot_path.parent:
            raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_DIRECTORY_MISMATCH")
        if not teacher_control_history_path.is_file():
            raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_HISTORY_MISSING")
        if artifacts.get("depth_safety_history_sha256") != _sha256(teacher_control_history_path):
            raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_HASH_MISMATCH")
        reference_track_path = snapshot_path.parent / "reference_track.json"
        if not reference_track_path.is_file():
            raise RuntimeError("LOCAL_POLICY_TEACHER_REFERENCE_TRACK_MISSING")
        if artifacts.get("track_sha256") != _sha256(reference_track_path):
            raise RuntimeError("LOCAL_POLICY_TEACHER_REFERENCE_TRACK_HASH_MISMATCH")
        receipt.update(
            {
                "teacher_control_history_file": str(teacher_control_history_path.resolve()),
                "teacher_control_history_sha256": _sha256(teacher_control_history_path),
                "teacher_reference_track_file": str(reference_track_path.resolve()),
                "teacher_reference_track_sha256": _sha256(reference_track_path),
            }
        )
    return receipt


def _target(cycle: dict[str, object], candidate_ids: tuple[str, ...]) -> int | None:
    action = cycle.get("model_action")
    if action == "pilot-control":
        return LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    if action == "select-candidate":
        selected = str(cycle.get("selected_candidate_id", ""))
        return candidate_ids.index(selected) if selected in candidate_ids else None
    return {"hold": 8, "request-new-scan": 9, "abort": 10}.get(str(action))


def _trace_target(trace: dict[str, object], candidate_ids: tuple[str, ...]) -> int | None:
    action = trace.get("navigation_action")
    if action == "pilot-control":
        return LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    if action == "select-candidate":
        selected = str(trace.get("navigation_selected_candidate_id", ""))
        return candidate_ids.index(selected) if selected in candidate_ids else None
    return {"hold": 8, "request-new-scan": 9, "abort": 10}.get(str(action))


def _pilot_control_from_trace(
    trace: dict[str, object] | None,
    *,
    target_action_index: int,
) -> list[float] | None:
    if target_action_index in range(8, LOCAL_POLICY_LEGACY_ACTION_COUNT):
        return [0.0, 0.0, 0.0, 0.0]
    payload = trace.get("pilot_control") if trace is not None else None
    if not isinstance(payload, dict):
        return None
    control = NormalizedPilotControl.model_validate(payload)
    return [
        control.forward_axis,
        control.right_axis,
        control.up_axis,
        control.yaw_axis,
    ]


def _teacher_control_label(
    snapshot: dict[str, object],
    control_record: dict[str, object],
    *,
    route_speed_limit_mps: float,
    application: ControlApplicationRecord | None = None,
) -> tuple[int, float, list[float], str, str] | None:
    """Use actual acknowledged controls, never the safety worker's proposal."""
    if application is None:
        raise RuntimeError("LOCAL_POLICY_TEACHER_EXECUTION_RECEIPT_REQUIRED")
    command = RuntimeLocalSafetyCommand.model_validate(control_record.get("command"))
    if (
        command.navigation_control_authority != "route-fallback"
        or command.requested_control_intent is not None
    ):
        raise RuntimeError("LOCAL_POLICY_TEACHER_COMMAND_NOT_DETERMINISTIC")
    if command.decision.action not in {"continue", "slow"}:
        return None
    strategic = snapshot.get("strategic_context")
    task = strategic.get("task") if isinstance(strategic, dict) else None
    if not isinstance(task, dict) or task.get("local_navigation_output_mode") != (
        "normalized-body-velocity"
    ):
        raise RuntimeError("LOCAL_POLICY_TEACHER_CONTINUOUS_OBSERVATION_REQUIRED")
    profile = task.get("control_profile")
    if profile not in {"cruise", "precision"}:
        raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_PROFILE_MISSING")
    limits = PilotControlLimits(
        *pilot_control_limits_for_profile(
            profile=profile,
            route_speed_limit_mps=route_speed_limit_mps,
        )
    )
    pilot = executed_pilot_control(snapshot, command, application, limits=limits)
    axes = [pilot.forward_axis, pilot.right_axis, pilot.up_axis, pilot.yaw_axis]
    if not any(abs(value) > 1e-9 for value in axes):
        return 8, 0.0, axes, "hold", "teacher-zero-velocity"
    # A verified successful demonstration is not counterfactual risk evidence.
    # Independent risk datasets retain their own action-conditioned assessor.
    return LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX, 0.0, axes, "pilot-control", ""


def _recovery_episode_id(snapshot: dict[str, object]) -> str | None:
    strategic = snapshot.get("strategic_context")
    task = strategic.get("task") if isinstance(strategic, dict) else None
    episode_id = task.get("recovery_episode_id") if isinstance(task, dict) else None
    if (
        not isinstance(episode_id, str)
        or not episode_id.startswith("recovery-")
        or len(episode_id) != 33
    ):
        return None
    return episode_id


def _recovery_episode_outcomes(
    ordered_snapshots: list[dict[str, object]],
) -> dict[str, bool]:
    """Prove that each recovery episode was followed by material progress.

    One recovery request can span several local-policy cycles.  Admission must
    therefore reason about the episode as a whole rather than treating every
    cycle as an independent success.  Progress is proven by either advancing to
    a different semantic goal or reducing the same goal's range by a meaningful
    amount after the first recovery decision.
    """

    first_indices: dict[str, int] = {}
    for index, snapshot in enumerate(ordered_snapshots):
        episode_id = _recovery_episode_id(snapshot)
        if episode_id is not None:
            first_indices.setdefault(episode_id, index)

    outcomes: dict[str, bool] = {}
    for episode_id, first_index in first_indices.items():
        first = ordered_snapshots[first_index]
        strategic = first.get("strategic_context")
        task = strategic.get("task") if isinstance(strategic, dict) else None
        goal_id = task.get("navigation_goal_id") if isinstance(task, dict) else None
        initial_distance = first.get("goal_distance_m")
        if not isinstance(goal_id, str) or not goal_id:
            outcomes[episode_id] = False
            continue
        if not isinstance(initial_distance, (int, float)) or not math.isfinite(
            float(initial_distance)
        ):
            outcomes[episode_id] = False
            continue
        initial_distance = float(initial_distance)
        required_reduction = max(0.05, min(0.25, initial_distance * 0.1))
        outcomes[episode_id] = False
        for later in ordered_snapshots[first_index + 1 :]:
            later_strategic = later.get("strategic_context")
            later_task = later_strategic.get("task") if isinstance(later_strategic, dict) else None
            later_goal_id = (
                later_task.get("navigation_goal_id") if isinstance(later_task, dict) else None
            )
            if isinstance(later_goal_id, str) and later_goal_id and later_goal_id != goal_id:
                # A new goal proves completion only when recovery began inside
                # the bounded arrival neighborhood.  A far-away user amendment
                # must not relabel an ineffective recovery as success.
                outcomes[episode_id] = initial_distance <= 0.5
                break
            if later_goal_id != goal_id:
                continue
            later_distance = later.get("goal_distance_m")
            if (
                isinstance(later_distance, (int, float))
                and math.isfinite(float(later_distance))
                and initial_distance - float(later_distance) >= required_reduction
            ):
                outcomes[episode_id] = True
                break
    return outcomes


def _collect_samples(
    pairs: list[tuple[Path, Path]],
    *,
    require_expert_trace: bool = False,
    allow_fallback_bootstrap: bool = False,
    require_recovery_episode_identity: bool = False,
    include_pilot_control_targets: bool = False,
    teacher_control_history_paths: list[Path] | None = None,
) -> dict[str, LocalPolicyTrainingSample]:
    if allow_fallback_bootstrap and not require_expert_trace:
        raise ValueError("fallback bootstrap requires typed expert traces")
    if require_recovery_episode_identity and not require_expert_trace:
        raise ValueError("recovery episode identity requires typed expert traces")
    samples: dict[str, LocalPolicyTrainingSample] = {}
    recovery_episode_samples: dict[
        str,
        tuple[tuple[int, int, float, float, str], str, LocalPolicyTrainingSample],
    ] = {}
    if teacher_control_history_paths is not None and len(teacher_control_history_paths) != len(
        pairs
    ):
        raise ValueError("each source pair requires its matching teacher control history")
    for pair_index, (snapshot_path, cycle_path) in enumerate(pairs):
        model_calls: dict[str, dict[str, object]] = {}
        if require_expert_trace:
            call_path = snapshot_path.parent / "model-navigation-model-calls.jsonl"
            if not call_path.is_file():
                raise RuntimeError("LOCAL_POLICY_SOURCE_MODEL_CALL_EVIDENCE_MISSING")
            model_calls = {str(record.get("call_id", "")): record for record in _jsonl(call_path)}
        snapshot_records = _jsonl(snapshot_path)
        ordered_snapshots = [
            record["snapshot"]
            for record in snapshot_records
            if isinstance(record.get("snapshot"), dict)
        ]
        snapshots = {
            str(snapshot.get("snapshot_sha256")): snapshot for snapshot in ordered_snapshots
        }
        snapshot_times = {
            str(record["snapshot"].get("snapshot_sha256")): int(
                record["snapshot"].get(
                    "control_reference_observed_at_unix_ms",
                    record.get("recorded_at_unix_ms", -1),
                )
            )
            for record in snapshot_records
            if isinstance(record.get("snapshot"), dict)
        }
        teacher_controls: dict[int, dict[str, object]] = {}
        applications: dict[str, ControlApplicationRecord] = {}
        teacher_route_speed_limits: dict[str, float] = {}
        if teacher_control_history_paths is not None:
            applications_path = (
                snapshot_path.parent / "runtime-state" / "control-applications.jsonl"
            )
            if not applications_path.is_file():
                raise RuntimeError("LOCAL_POLICY_TEACHER_EXECUTION_RECEIPT_REQUIRED")
            for payload in _jsonl(applications_path):
                application = ControlApplicationRecord.model_validate(payload)
                applications.setdefault(application.command_sha256, application)
            teacher_controls = {
                int(record.get("recorded_at_unix_ms", -1)): record
                for record in _jsonl(teacher_control_history_paths[pair_index])
            }
            teacher_route_speed_limits = _teacher_route_speed_limits(snapshot_path.parent)
        recovery_outcomes = (
            _recovery_episode_outcomes(ordered_snapshots)
            if require_recovery_episode_identity
            else {}
        )
        for cycle in _jsonl(cycle_path):
            snapshot_sha256 = str(cycle.get("snapshot_sha256", ""))
            snapshot = snapshots.get(snapshot_sha256)
            if snapshot is None:
                continue
            # A failed or timed-out provider invocation deliberately has no
            # expert vote and therefore no supervised target.  Exclude that
            # safety receipt before enforcing typed-trace integrity for
            # decisions that actually came from an expert.  Inventing a trace
            # here would turn a fail-closed hold into a false training label.
            if cycle.get("model_action") == "not-invoked":
                continue
            teacher_route_speed_limit_mps: float | None = None
            continuous_control_speed_mps: float | None = None
            control_record: dict[str, object] | None = None
            feature_snapshot = snapshot
            if teacher_controls:
                snapshot_time = snapshot_times.get(snapshot_sha256, -1)
                control_record = teacher_controls.get(snapshot_time)
                if control_record is None:
                    raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_ALIGNMENT_MISSING")
                # The student must see the ORIGINAL deployment-time goal.
                # Replacing it with a teacher lookahead creates train/runtime
                # skew and smuggles the teacher's private path into the input.
                navigation_goal_id = _snapshot_navigation_goal_id(snapshot)
                teacher_route_speed_limit_mps = teacher_route_speed_limits.get(navigation_goal_id)
                if teacher_route_speed_limit_mps is None:
                    raise RuntimeError("LOCAL_POLICY_TEACHER_REFERENCE_GOAL_MISSING")
                strategic = snapshot.get("strategic_context")
                task = strategic.get("task") if isinstance(strategic, dict) else None
                control_profile = task.get("control_profile") if isinstance(task, dict) else None
                if control_profile not in {"cruise", "precision"}:
                    raise RuntimeError("LOCAL_POLICY_TEACHER_CONTROL_PROFILE_MISSING")
                continuous_control_speed_mps = pilot_control_limits_for_profile(
                    profile=control_profile,
                    route_speed_limit_mps=teacher_route_speed_limit_mps,
                )[0]
            batch = compile_local_policy_features(
                feature_snapshot,
                continuous_control_speed_mps=continuous_control_speed_mps,
                include_candidate_features=not include_pilot_control_targets,
            )
            requested_role = requested_navigation_expert(feature_snapshot)
            recovery_episode_id: str | None = None
            if requested_role == "recovery-policy" and require_recovery_episode_identity:
                recovery_episode_id = _recovery_episode_id(snapshot)
                if recovery_episode_id is None:
                    raise RuntimeError("LOCAL_POLICY_SOURCE_RECOVERY_EPISODE_ID_MISSING")
                if recovery_outcomes.get(recovery_episode_id) is not True:
                    raise RuntimeError("LOCAL_POLICY_SOURCE_RECOVERY_OUTCOME_NOT_VERIFIED")
            trace: dict[str, object] | None = None
            if require_expert_trace:
                call_record = model_calls.get(str(cycle.get("model_call_id", "")))
                if not isinstance(call_record, dict) or not isinstance(
                    call_record.get("local_expert_trace"), dict
                ):
                    raise RuntimeError("LOCAL_POLICY_SOURCE_EXPERT_TRACE_MISSING")
                trace = dict(call_record["local_expert_trace"])
                if trace.get("requested_navigation_role") != requested_role:
                    raise RuntimeError("LOCAL_POLICY_SOURCE_EXPERT_ROUTING_MISMATCH")
                selected_role = str(trace.get("selected_navigation_role", ""))
                fallback_used = trace.get("fallback_used") is True
                if allow_fallback_bootstrap:
                    # Bootstrap data deliberately distills the admitted general
                    # navigation expert into a missing specialist role.  Keep it
                    # separate from specialist-produced qualification evidence.
                    if selected_role == requested_role:
                        continue
                    if (
                        requested_role == "local-navigation-policy"
                        or selected_role != "local-navigation-policy"
                        or not fallback_used
                    ):
                        raise RuntimeError("LOCAL_POLICY_SOURCE_FALLBACK_TRACE_INVALID")
                elif selected_role != requested_role:
                    # A fallback is useful bootstrap evidence for the selected
                    # general policy, but it is not evidence produced by the
                    # specialist that the snapshot requested.
                    continue
                target = _trace_target(trace, batch.candidate_ids)
                risk_target = float(trace.get("navigation_risk_score", -1.0))
                if not 0.0 <= risk_target <= 1.0:
                    raise RuntimeError("LOCAL_POLICY_SOURCE_EXPERT_RISK_INVALID")
                if allow_fallback_bootstrap and risk_target >= 0.5:
                    # Bootstrap specialists learn the teacher's conservative
                    # safety class, not a brittle value only a few hundredths
                    # above the runtime hold threshold. Keep safe-side
                    # continuous scores unchanged, but promote every teacher
                    # risk veto to an unambiguous risky target.
                    risk_target = 1.0
                action = str(trace.get("navigation_action", ""))
                hold_reason = ""
            else:
                target = _target(cycle, batch.candidate_ids)
                action = str(cycle.get("model_action"))
                hold_reason = str(cycle.get("hold_reason") or "")
                risk_target = (
                    0.0
                    if action == "select-candidate"
                    else 1.0
                    if action in {"hold", "abort"}
                    else 0.7
                )
            teacher_pilot_control: list[float] | None = None
            if teacher_controls:
                assert control_record is not None
                if teacher_route_speed_limit_mps is None:
                    raise AssertionError("teacher route speed was not resolved")
                teacher_label = _teacher_control_label(
                    feature_snapshot,
                    control_record,
                    route_speed_limit_mps=teacher_route_speed_limit_mps,
                    application=applications.get(
                        sha256_json(
                            RuntimeLocalSafetyCommand.model_validate(control_record.get("command"))
                        )
                    ),
                )
                if teacher_label is None:
                    continue
                (
                    target,
                    risk_target,
                    teacher_pilot_control,
                    action,
                    hold_reason,
                ) = teacher_label
            if target is None:
                continue
            pilot_control_target: list[float] = []
            if include_pilot_control_targets:
                direct_target = teacher_pilot_control or _pilot_control_from_trace(
                    trace,
                    target_action_index=target,
                )
                if direct_target is None:
                    raise RuntimeError("LOCAL_POLICY_PILOT_TARGET_DIRECT_CONTROL_REQUIRED")
                pilot_control_target = direct_target
            sample_weight = (
                2.0 if target in range(8, LOCAL_POLICY_LEGACY_ACTION_COUNT) or hold_reason else 1.0
            )
            sample = LocalPolicyTrainingSample(
                temporal_evidence=batch.temporal_evidence,
                pilot_control_limits=batch.pilot_control_limits,
                navigation_expert_role=requested_role,
                source_snapshot_sha256=snapshot_sha256,
                source_visual_sha256=(
                    str(snapshot["visual_evidence"][0].get("sha256"))
                    if isinstance(snapshot.get("visual_evidence"), list)
                    and snapshot["visual_evidence"]
                    and isinstance(snapshot["visual_evidence"][0], dict)
                    and snapshot["visual_evidence"][0].get("sha256")
                    else None
                ),
                state_features=list(batch.state_features),
                candidate_features=[list(item) for item in batch.candidate_features],
                candidate_mask=list(batch.candidate_mask),
                realtime_features=(
                    list(batch.realtime_features) if batch.realtime_features_ready else []
                ),
                realtime_valid_mask=(
                    list(batch.realtime_valid_mask) if batch.realtime_features_ready else []
                ),
                target_pilot_control=pilot_control_target,
                control_feature_contract_sha256=(
                    batch.control_feature_contract_sha256 if batch.realtime_features_ready else None
                ),
                target_action_index=target,
                risk_target=risk_target,
                sample_weight=sample_weight,
            )
            identity = hashlib.sha256(
                json.dumps(
                    {
                        "snapshot_sha256": snapshot_sha256,
                        "target": target,
                        "risk": risk_target,
                        "label_source": (
                            "verified-executed-control"
                            if teacher_controls
                            else (
                                "fallback-navigation-expert-trace"
                                if allow_fallback_bootstrap
                                else (
                                    "navigation-expert-trace"
                                    if require_expert_trace
                                    else "final-ensemble-decision"
                                )
                            )
                        ),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if recovery_episode_id is None:
                samples.setdefault(identity, sample)
                continue

            # A recovery incident often begins with a benign vote and becomes
            # safety-critical only after another observation confirms that the
            # corridor is still blocked.  Counting every frame would leak one
            # incident into many samples, while always keeping the first frame
            # discards the very hold/rescan decision the specialist must
            # learn.  Retain exactly one deterministic, most-conservative
            # representative for the whole outcome-proven episode.
            safety_action_rank = (
                sample.target_action_index - 7
                if sample.target_action_index in range(8, LOCAL_POLICY_LEGACY_ACTION_COUNT)
                else 0
            )
            risk_class = int(
                sample.risk_target >= 0.5
                or sample.target_action_index in range(8, LOCAL_POLICY_LEGACY_ACTION_COUNT)
            )
            priority = (
                risk_class,
                safety_action_rank,
                sample.risk_target,
                sample.sample_weight,
                identity,
            )
            selected = recovery_episode_samples.get(recovery_episode_id)
            if selected is None or priority > selected[0]:
                recovery_episode_samples[recovery_episode_id] = (
                    priority,
                    identity,
                    sample,
                )
    for _, identity, sample in recovery_episode_samples.values():
        samples.setdefault(identity, sample)
    return samples


def _source_receipts(
    pairs: list[tuple[Path, Path]],
    *,
    require_verified: bool,
    require_expert_trace: bool,
    teacher_control_history_paths: list[Path] | None = None,
) -> list[dict[str, object]]:
    receipts = []
    if teacher_control_history_paths is not None and len(teacher_control_history_paths) != len(
        pairs
    ):
        raise ValueError("source receipt teacher history count differs from source pairs")
    for index, (snapshot_path, cycle_path) in enumerate(pairs):
        teacher_history = (
            teacher_control_history_paths[index]
            if teacher_control_history_paths is not None
            else None
        )
        receipt: dict[str, object] = {
            "snapshot_file": str(snapshot_path.resolve()),
            "snapshot_sha256": _sha256(snapshot_path),
            "cycle_file": str(cycle_path.resolve()),
            "cycle_sha256": _sha256(cycle_path),
        }
        if require_verified:
            receipt.update(
                _verified_source_receipt(
                    snapshot_path,
                    cycle_path,
                    require_expert_trace=require_expert_trace,
                    teacher_control_history_path=teacher_history,
                )
            )
        elif require_expert_trace:
            call_path = snapshot_path.parent / "model-navigation-model-calls.jsonl"
            if not call_path.is_file():
                raise RuntimeError("LOCAL_POLICY_SOURCE_MODEL_CALL_EVIDENCE_MISSING")
            receipt.update(
                {
                    "model_call_file": str(call_path.resolve()),
                    "model_call_sha256": _sha256(call_path),
                }
            )
        receipts.append(receipt)
    return receipts


def _expert_counts(
    records: list[LocalPolicyTrainingSample],
) -> dict[str, int]:
    counts = Counter(record.navigation_expert_role for record in records)
    return dict(sorted(counts.items()))


def _filter_expert_samples(
    samples: dict[str, LocalPolicyTrainingSample],
    role: NavigationExpertRole | None,
) -> dict[str, LocalPolicyTrainingSample]:
    """Limit a specialist corpus before split checks and visual encoding.

    A recovery augmentation must not make thousands of unrelated precision
    rows part of its content-bound dataset or visual-encoding receipt.  The
    unfiltered default remains available for general multi-role corpora.
    """

    if role is None:
        return samples
    return {
        identity: sample
        for identity, sample in samples.items()
        if sample.navigation_expert_role == role
    }


def _exclude_cross_split_visual_duplicates(
    training: list[LocalPolicyTrainingSample],
    validation: list[LocalPolicyTrainingSample],
) -> tuple[list[LocalPolicyTrainingSample], int, int]:
    """Keep an exact RGB observation in only one evaluation split.

    One forward frame may back several distinct navigation snapshots.  Those
    snapshots are legitimate training examples because the non-visual state
    can differ, but an identical content hash in both splits leaks visual
    evidence into held-out evaluation.  Training owns the duplicate and every
    validation sample bound to that RGB hash is excluded.  Missing visual
    evidence is deliberately ignored here; a visual-policy encoder applies a
    separate fail-closed requirement before it can train.
    """

    training_visual_hashes = {
        sample.source_visual_sha256
        for sample in training
        if sample.source_visual_sha256 is not None
    }
    duplicate_hashes = {
        sample.source_visual_sha256
        for sample in validation
        if sample.source_visual_sha256 in training_visual_hashes
    }
    filtered = [
        sample for sample in validation if sample.source_visual_sha256 not in duplicate_hashes
    ]
    return filtered, len(duplicate_hashes), len(validation) - len(filtered)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, action="append", required=True)
    parser.add_argument("--cycles", type=Path, action="append", required=True)
    parser.add_argument("--validation-snapshots", type=Path, action="append")
    parser.add_argument("--validation-cycles", type=Path, action="append")
    parser.add_argument(
        "--teacher-control-history",
        type=Path,
        action="append",
        help=(
            "Hash-bound deterministic local-safety history that actually controlled "
            "each verified training flight. Repeat once per --snapshots source."
        ),
    )
    parser.add_argument(
        "--validation-teacher-control-history",
        type=Path,
        action="append",
        help=("Independent teacher control history for each explicit validation source."),
    )
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument(
        "--require-verified-source",
        action="store_true",
        help="Reject any source campaign that did not pass its complete runtime gates.",
    )
    parser.add_argument(
        "--require-expert-trace",
        action="store_true",
        help=(
            "Train only from the selected navigation expert's pre-advisor vote and "
            "reject legacy ensemble-only labels."
        ),
    )
    parser.add_argument(
        "--allow-fallback-bootstrap",
        action="store_true",
        help=(
            "Create simulation-only specialist bootstrap labels from an admitted "
            "general navigation expert's typed fallback vote. Requires verified "
            "sources and --require-expert-trace."
        ),
    )
    parser.add_argument(
        "--require-recovery-episode-identity",
        action="store_true",
        help=(
            "Reject recovery samples that are not bound to one explicit semantic "
            "stall episode, admit at most one sample per episode, and require "
            "material post-decision progress. Required for recovery specialist "
            "admission datasets."
        ),
    )
    parser.add_argument(
        "--navigation-expert-role",
        choices=NAVIGATION_EXPERT_ROLES,
        help=(
            "Write only the selected navigation expert's rows. Specialist "
            "training should use this to keep unrelated roles out of the "
            "content-bound visual corpus."
        ),
    )
    parser.add_argument(
        "--include-pilot-control-targets",
        action="store_true",
        help=(
            "Train the continuous body-velocity head only from typed direct-control "
            "expert output or hash-bound control that a verified teacher executed."
        ),
    )
    args = parser.parse_args()
    if args.allow_fallback_bootstrap and not (
        args.require_expert_trace and args.require_verified_source
    ):
        parser.error(
            "fallback bootstrap requires --require-expert-trace and --require-verified-source"
        )
    if args.require_recovery_episode_identity and not (
        args.require_expert_trace and args.require_verified_source
    ):
        parser.error(
            "recovery episode identity requires --require-expert-trace and "
            "--require-verified-source"
        )
    if args.include_pilot_control_targets and not (
        args.require_expert_trace and args.require_verified_source
    ):
        parser.error(
            "pilot-control targets require --require-expert-trace and --require-verified-source"
        )
    if (args.teacher_control_history or args.validation_teacher_control_history) and not (
        args.include_pilot_control_targets
    ):
        parser.error("teacher control histories require --include-pilot-control-targets")
    if len(args.snapshots) != len(args.cycles):
        parser.error("each snapshot evidence file requires its matching cycle evidence file")
    if args.teacher_control_history and len(args.teacher_control_history) != len(args.snapshots):
        parser.error("each training snapshot source requires one teacher control history")
    if bool(args.validation_snapshots) != bool(args.validation_cycles):
        parser.error("explicit validation snapshots and cycles must be supplied together")
    if args.validation_snapshots and len(args.validation_snapshots) != len(args.validation_cycles):
        parser.error("each validation snapshot file requires its matching cycle file")
    if args.validation_teacher_control_history and len(
        args.validation_teacher_control_history
    ) != len(args.validation_snapshots or []):
        parser.error("each validation snapshot source requires one teacher control history")
    if (
        bool(args.teacher_control_history) != bool(args.validation_teacher_control_history)
        and args.validation_snapshots
    ):
        parser.error("teacher control labels must be used consistently across explicit splits")
    if not 5 <= args.validation_percent <= 50:
        parser.error("validation percent must be between 5 and 50")
    if (
        args.training_output.exists()
        or args.validation_output.exists()
        or (args.receipt_output is not None and args.receipt_output.exists())
    ):
        raise FileExistsError("local policy dataset output already exists")

    training_pairs = list(zip(args.snapshots, args.cycles, strict=True))
    explicit_validation_pairs = list(
        zip(
            args.validation_snapshots or [],
            args.validation_cycles or [],
            strict=True,
        )
    )
    if args.require_verified_source:
        all_pairs = [*training_pairs, *explicit_validation_pairs]
        all_histories = [
            *(args.teacher_control_history or [None] * len(training_pairs)),
            *(args.validation_teacher_control_history or [None] * len(explicit_validation_pairs)),
        ]
        for (snapshot_path, cycle_path), teacher_history in zip(
            all_pairs, all_histories, strict=True
        ):
            _verified_source_receipt(
                snapshot_path,
                cycle_path,
                require_expert_trace=args.require_expert_trace,
                teacher_control_history_path=teacher_history,
            )
    samples = _collect_samples(
        training_pairs,
        require_expert_trace=args.require_expert_trace,
        allow_fallback_bootstrap=args.allow_fallback_bootstrap,
        require_recovery_episode_identity=args.require_recovery_episode_identity,
        include_pilot_control_targets=args.include_pilot_control_targets,
        teacher_control_history_paths=args.teacher_control_history,
    )
    samples = _filter_expert_samples(samples, args.navigation_expert_role)
    if len(samples) < 10:
        raise RuntimeError("LOCAL_POLICY_DATASET_HAS_FEWER_THAN_TEN_UNIQUE_SAMPLES")

    if explicit_validation_pairs:
        validation_samples = _collect_samples(
            explicit_validation_pairs,
            require_expert_trace=args.require_expert_trace,
            allow_fallback_bootstrap=args.allow_fallback_bootstrap,
            require_recovery_episode_identity=(args.require_recovery_episode_identity),
            include_pilot_control_targets=args.include_pilot_control_targets,
            teacher_control_history_paths=args.validation_teacher_control_history,
        )
        validation_samples = _filter_expert_samples(
            validation_samples,
            args.navigation_expert_role,
        )
        overlap = samples.keys() & validation_samples.keys()
        if overlap:
            raise RuntimeError("LOCAL_POLICY_TRAINING_VALIDATION_OVERLAP")
        training = [sample for _, sample in sorted(samples.items())]
        validation = [sample for _, sample in sorted(validation_samples.items())]
    else:
        training = []
        validation = []
        for identity, sample in sorted(samples.items()):
            bucket = int(identity[:8], 16) % 100
            target = validation if bucket < args.validation_percent else training
            target.append(sample)
    validation_sample_count_before_visual_deduplication = len(validation)
    (
        validation,
        cross_split_duplicate_visual_hash_count,
        rejected_validation_visual_sample_count,
    ) = _exclude_cross_split_visual_duplicates(training, validation)
    if not training or not validation:
        raise RuntimeError("LOCAL_POLICY_DATASET_SPLIT_IS_EMPTY")
    for path, records in (
        (args.training_output, training),
        (args.validation_output, validation),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )
    if args.receipt_output is not None:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(
            json.dumps(
                {
                    "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                    "split_method": (
                        "held-out-source-campaign"
                        if explicit_validation_pairs
                        else "deterministic-sample-hash"
                    ),
                    "navigation_label_source": (
                        "verified-executed-control"
                        if args.teacher_control_history
                        else (
                            "fallback-navigation-expert-trace"
                            if args.allow_fallback_bootstrap
                            else (
                                "navigation-expert-trace"
                                if args.require_expert_trace
                                else "final-ensemble-decision"
                            )
                        )
                    ),
                    "navigation_expert_role_filter": args.navigation_expert_role,
                    "pilot_control_targets_included": (args.include_pilot_control_targets),
                    "pilot_control_target_semantics": (
                        "normalized-body-velocity" if args.include_pilot_control_targets else None
                    ),
                    "teacher_control_reference_alignment": (
                        "original-observation-and-acknowledged-velocity-command"
                        if args.teacher_control_history
                        else None
                    ),
                    "bootstrap_only": args.allow_fallback_bootstrap,
                    "fallback_teacher_navigation_role": (
                        "local-navigation-policy" if args.allow_fallback_bootstrap else None
                    ),
                    "fallback_risky_target_policy": (
                        "teacher-risk-at-least-0.5-promoted-to-1.0"
                        if args.allow_fallback_bootstrap
                        else None
                    ),
                    "recovery_episode_identity_required": (args.require_recovery_episode_identity),
                    "recovery_outcome_progress_required": (args.require_recovery_episode_identity),
                    "recovery_samples_capped_per_episode": (
                        1 if args.require_recovery_episode_identity else None
                    ),
                    "training_sources": _source_receipts(
                        training_pairs,
                        require_verified=args.require_verified_source,
                        require_expert_trace=args.require_expert_trace,
                        teacher_control_history_paths=args.teacher_control_history,
                    ),
                    "validation_sources": _source_receipts(
                        explicit_validation_pairs or training_pairs,
                        require_verified=args.require_verified_source,
                        require_expert_trace=args.require_expert_trace,
                        teacher_control_history_paths=(
                            args.validation_teacher_control_history
                            if explicit_validation_pairs
                            else args.teacher_control_history
                        ),
                    ),
                    "training_sample_count": len(training),
                    "validation_sample_count": len(validation),
                    "validation_sample_count_before_visual_deduplication": (
                        validation_sample_count_before_visual_deduplication
                    ),
                    "cross_split_duplicate_visual_hash_count": (
                        cross_split_duplicate_visual_hash_count
                    ),
                    "rejected_validation_visual_sample_count": (
                        rejected_validation_visual_sample_count
                    ),
                    "cross_split_visual_duplicate_policy": ("retain-training-exclude-validation"),
                    "training_expert_sample_counts": _expert_counts(training),
                    "validation_expert_sample_counts": _expert_counts(validation),
                    "verified_sources_required": args.require_verified_source,
                    "expert_trace_required": args.require_expert_trace,
                    "training_output_sha256": _sha256(args.training_output),
                    "validation_output_sha256": _sha256(args.validation_output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "source_training_unique_sample_count": len(samples),
                "source_validation_unique_sample_count": (
                    len(validation_samples)
                    if explicit_validation_pairs
                    else validation_sample_count_before_visual_deduplication
                ),
                "admitted_unique_sample_count": len(training) + len(validation),
                "training_sample_count": len(training),
                "validation_sample_count": len(validation),
                "cross_split_duplicate_visual_hash_count": (
                    cross_split_duplicate_visual_hash_count
                ),
                "rejected_validation_visual_sample_count": (
                    rejected_validation_visual_sample_count
                ),
                "training_expert_sample_counts": _expert_counts(training),
                "validation_expert_sample_counts": _expert_counts(validation),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
