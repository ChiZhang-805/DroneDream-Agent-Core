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

from dronedream_agent_core.control_feature_contract import (
    CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    FLIGHT_STATE_MAXIMUM_GAP_MS,
)
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
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.temporal_evidence import ObservationHistory
from dronedream_agent_core.training.advisor_sources import (
    RecordedSource,
    validate_advisor_spatial_splits,
)
from dronedream_agent_core.training.dagger_artifacts import write_rows
from dronedream_agent_core.training.evidence_files import decode_evidence_rows
from dronedream_agent_core.training.mission_groups import (
    SPATIAL_SPLIT_CONTRACT,
    recorded_mission_group,
)
from dronedream_plugin_sdk.protocol import decode_json

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


# 功能：
#   获取已冻结来源或普通文件的有界字节，拒绝链接与读取期间的身份改变。
# 输入：
#   path：来源快照或明确文件路径。
# 输出：
#   content：最多 256 MiB 的不可变原始字节。
def _source_bytes(path: Path | RecordedSource) -> bytes:
    content = (path.read_bytes() if isinstance(path, RecordedSource)
               else read_plugin_file(path, limit=256 * 1024**2))
    return content


# 功能：
#   计算本次实际读取来源的内容身份。
# 输入：
#   path：冻结来源或有界文件路径。
# 输出：
#   digest：原始字节的 SHA-256。
def _sha256(path: Path | RecordedSource) -> str:
    digest = hashlib.sha256(_source_bytes(path)).hexdigest()
    return digest


# 功能：
#   严格读取回执对象，拒绝重复字段、非有限值和非对象结构。
# 输入：
#   path：冻结来源或回执文件路径。
# 输出：
#   value：无歧义的回执对象。
def _json(path: Path | RecordedSource) -> dict[str, object]:
    value = decode_json(_source_bytes(path), limit=4 * 1024**2, node_limit=1_000_000)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


# 功能：
#   逐行检查完整记录，拒绝截断、重复字段和超出预算的数据。
# 输入：
#   path：已固定来源或明确数据文件路径。
# 输出：
#   records：通过统一训练证据边界的对象列表。
def _jsonl(path: Path | RecordedSource) -> list[dict[str, object]]:
    records = decode_evidence_rows(_source_bytes(path))
    return records


# 功能：
#   将额外恢复证明的解析值和摘要绑定到同一次读取，不能拿替换后的文件给旧判断背书。
# 输入：
#   path：开发采集或恢复证明文件。
# 输出：
#   value、digest：同一冻结字节产生的严格对象与 SHA-256。
def _frozen_source_object(path: Path) -> tuple[dict, str]:
    source = RecordedSource.read(path)
    value, digest = _json(source), _sha256(source)
    return value, digest


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

    receipt, receipt_digest = _frozen_source_object(receipt_path)
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
        "fault_recovery_receipt_sha256": receipt_digest,
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
    summary, summary_digest = _frozen_source_object(summary_path)
    workflow, workflow_digest = _frozen_source_object(workflow_path)
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
        or summary.get("workflow_sha256") != workflow_digest
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
        "payload_collection_summary_sha256": summary_digest,
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
    receipt, receipt_digest = _frozen_source_object(receipt_path)
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
        "payload_recovery_receipt_sha256": receipt_digest,
        "bounded_payload_motion_cycle_count": bounded_motion_count,
        "whole_mission_status": "failed",
        "flight_qualification_granted": False,
    }


# 功能：
#   按真实记录类型冻结辅助专家来源；当前记录校验失败不回退到旧格式或开发豁免。
# 输入：
#   run_dir：仿真运行目录。
#   require_verified：旧记录是否要求验证通过；当前记录始终要求通过。
#   allow_verified_fault_recovery：旧故障恢复证据的显式接纳范围。
#   allow_verified_payload_collection、allow_verified_payload_recovery：旧载荷证据的显式接纳范围。
#   retained_route：原运行路线缺失时显式指定的原始文件，仍须匹配运行记录的完整摘要。
# 输出：
#   source：来源回执及冻结的观测、可选旧周期、深度历史和多模态记录。
def _source_receipt(
    run_dir: Path,
    *,
    require_verified: bool,
    allow_verified_fault_recovery: bool,
    allow_verified_payload_collection: bool,
    allow_verified_payload_recovery: bool,
    retained_route: Path | None = None,
) -> tuple[dict[str, object], RecordedSource, RecordedSource | None,
           RecordedSource | None, RecordedSource | None]:
    check_plain_plugin_path(run_dir.absolute())
    resolved = run_dir.resolve()
    # 当前采集没有旧模型调用周期；按自身协议校验，失败时不退回旧来源。
    if (resolved / "learning-observations.jsonl").exists():
        if retained_route is not None:
            raise ValueError("ADVISOR_CURRENT_SOURCE_CANNOT_REPLACE_ROUTE")
        if (allow_verified_payload_collection and not allow_verified_fault_recovery
                and not allow_verified_payload_recovery):
            from dronedream_agent_core.training.payload_teacher_sources import read_payload_teacher_source

            receipt, observations = read_payload_teacher_source(resolved)
            return receipt, observations, None, None, None
        if (allow_verified_fault_recovery and not allow_verified_payload_collection
                and not allow_verified_payload_recovery
                and (resolved / "development-depth-fault.json").is_file()):
            from dronedream_agent_core.training.teacher_fault_sources import (
                read_teacher_fault_advisor_source,
            )

            receipt, observations, depth = read_teacher_fault_advisor_source(resolved)
            return receipt, observations, None, depth, None
        if any((allow_verified_fault_recovery, allow_verified_payload_collection,
                allow_verified_payload_recovery)):
            raise ValueError("ADVISOR_CURRENT_SOURCE_CANNOT_USE_LEGACY_RECOVERY_OVERRIDE")
        from dronedream_agent_core.training.current_advisor_sources import (
            read_current_advisor_source,
        )

        receipt, observations = read_current_advisor_source(resolved)
        return receipt, observations, None, None, None
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
            "mission_split": recorded_mission_group(
                resolved, artifacts, retained_route=retained_route).model_dump(mode="json"),
            "retained_route_evidence": str(retained_route.absolute()) if retained_route else None,
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


# 功能：
#   1. 从冻结观测生成辅助专家监督，当前来源不要求或伪造旧模型调用周期。
#   2. 当前观测在断流时清空时间历史；标签不是行为模仿动作或飞行准入。
# 输入：
#   snapshot_path：冻结的快照或当前学习观测。
#   cycle_path：可选旧调用周期；None 表示按当前学习观测时间处理。
#   depth_history_path、multimodal_records_path：可选独立绑定的旧补充来源。
#   roles：本次选择的辅助专家；source_identity：来源证据摘要。
# 输出：
#   samples：按来源、快照、角色和监督内容去重的样本。
def _collect_run_samples(
    snapshot_path: Path | RecordedSource,
    cycle_path: Path | RecordedSource | None,
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
    # 当前来源按记录顺序检查传感器时钟，不排序掩盖真实回退。
    if cycle_path is not None:
        snapshot_timeline.sort(key=lambda item: item[0])
    elif depth_history_path is not None:
        from dronedream_agent_core.training.teacher_sensor_replay import (
            replay_teacher_sensor_diagnostics,
        )

        # 当前故障任务只产生有原始日志绑定的感知标签，不能成为动作或载荷监督。
        return replay_teacher_sensor_diagnostics(snapshot_timeline, _jsonl(depth_history_path),
            roles=roles, source_identity=source_identity)
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
        _jsonl(cycle_path) if cycle_path is not None else [],
        key=lambda cycle: (
            int(_number(cycle.get("sequence"))),
            int(_number(cycle.get("recorded_at_unix_ms"))),
        ),
    )
    if cycle_path is None:
        ordered_snapshots = snapshot_timeline
    else:
        ordered_snapshots = []
        for cycle in cycles:
            snapshot = snapshots.get(str(cycle.get("snapshot_sha256", "")))
            if snapshot is None:
                raise RuntimeError("LOCAL_ADVISOR_CYCLE_SNAPSHOT_MISSING")
            ordered_snapshots.append((int(_number(cycle.get("recorded_at_unix_ms"))), snapshot))
    current_history = ObservationHistory(LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                                         maximum_gap_ms=FLIGHT_STATE_MAXIMUM_GAP_MS)
    current_velocity_history = ObservationHistory(LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                                                  maximum_gap_ms=FLIGHT_STATE_MAXIMUM_GAP_MS)
    for _recorded_at, snapshot in ordered_snapshots:
        snapshot_sha256 = _verified_snapshot(snapshot)
        batch = compile_local_policy_features(snapshot)
        if cycle_path is None:
            # 与在线后端共用传感器来源去重、任务切换、重启及间隔规则；
            # 日志写盘延迟不能改变有效历史，重复调用也不能填满窗口。
            if batch.temporal_evidence is None:
                current_history.clear()
                current_velocity_history.clear()
            else:
                current_history.append(batch.temporal_evidence, batch.state_features,
                                       batch.payload_features)
                current_velocity_history.append(batch.temporal_evidence, _velocity(snapshot), ())
            history = deque(row[0] for row in current_history.rows)
            payload_history = deque(row[1] for row in current_history.rows)
            velocity_history = deque(row[0] for row in current_velocity_history.rows)
            if not velocity_history:
                # 无来源身份时仍可判断当前速度，但不能伪造时序变化或有效掩码。
                velocity_history.append(_velocity(snapshot))
        else:
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


# 功能：
#   逐份核验并冻结辅助训练来源，拒绝重复样本和越权角色，不改写原始证据。
# 输入：
#   run_dirs、roles：明确的来源与专家；require_verified：来源准入要求。
#   allow_verified_fault_recovery：显式故障恢复用途。
#   allow_verified_payload_collection、allow_verified_payload_recovery：显式负载采集和恢复用途。
#   retained_routes：运行绝对路径到原始路线文件的明确映射。
# 输出：
#   samples：去重样本；receipts：每份来源及原始路线身份。
def _collect_sources(
    run_dirs: list[Path],
    *,
    roles: tuple[TrainableAdvisorRole, ...],
    require_verified: bool,
    allow_verified_fault_recovery: bool,
    allow_verified_payload_collection: bool,
    allow_verified_payload_recovery: bool,
    retained_routes: dict[Path, Path] | None = None,
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
            retained_route=(retained_routes or {}).get(run_dir.resolve()),
        )
        source_identity = str(receipt["mission_evidence_sha256"])
        if "allowed_roles" in receipt and not set(roles).issubset(receipt["allowed_roles"]):
            raise ValueError("LOCAL_ADVISOR_SOURCE_ROLE_NOT_ALLOWED")
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


# 功能：
#   独占发布确定排序的完整样本文件，不覆盖其他作业已写入的结果。
# 输入：
#   path、samples：新输出路径及按内容身份索引的样本。
# 输出：
#   None：文件写入成功后不返回业务数据。
def _write_jsonl(
    path: Path,
    samples: dict[str, LocalAdvisorTrainingSample],
) -> None:
    check_plain_plugin_path(path.absolute())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(path, [sample for _, sample in sorted(samples.items())])


# 功能：
#   检查完整任务的空间隔离和各专家覆盖，先发布样本再提交最终回执，不授予飞行资格。
# 输入：
#   命令行参数：训练与验证运行目录、专家类别及三个互不重复的输出路径。
# 输出：
#   exit_code：所有样本及完整回执发布成功时为零。
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
    parser.add_argument("--retained-route", type=Path, nargs=2, action="append", default=[],
                        metavar=("RUN_DIRECTORY", "ORIGINAL_ROUTE"),
                        help="Bind a missing legacy route to an explicitly retained file; "
                             "the original mission SHA-256 must match.")
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
    for path in outputs:
        check_plain_plugin_path(path.absolute())
    if len({str(path.absolute()).casefold() for path in outputs}) != len(outputs):
        parser.error("dataset outputs must be distinct paths")
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
    retained_routes = {}
    for run, route in args.retained_route:
        key = run.resolve()
        if key not in training_dirs | validation_dirs or key in retained_routes:
            parser.error("retained route must name one unique requested run directory")
        check_plain_plugin_path(route.absolute())
        retained_routes[key] = route.absolute()
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
        retained_routes=retained_routes,
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
        retained_routes=retained_routes,
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
    publish_runtime_json(args.receipt_output, receipt, replace_existing=False)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
