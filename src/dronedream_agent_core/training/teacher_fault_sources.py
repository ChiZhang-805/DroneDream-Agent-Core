"""Fault-injected teacher sensor evidence, never behavior or flight qualification."""

import hashlib
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
from ..hashing import sha256_json
from ..simulation_teacher_contract import SIMULATION_TEACHER_CONTRACT_SHA256
from .advisor_sources import RecordedSource
from .evidence_files import decode_evidence_rows, read_evidence_dataset, read_evidence_object
from .mission_groups import recorded_mission_group


# 功能：
#   1. 核验故障接管及恢复，不把缺日志当作安全停车。
#   2. 仅将摘要绑定、制动生成前且原租期内的故障前命令单独记为在途命令，不授予飞行资格。
# 输入：
#   fault：原生故障注入记录；rows：按原始顺序读取的执行器日志。
#   depth_rows：已绑定摘要的安全决策日志，缺失时不接纳任何故障中运动。
# 输出：
#   counts：运动、接管、恢复及单独列出的在途命令数量。
def verify_teacher_fault_window(fault: dict, rows: list[dict], depth_rows=None) -> dict:
    start, end = fault.get("activated_at_unix_ms"), fault.get("recovered_at_unix_ms")
    duration = fault.get("drop_duration_seconds")
    if (fault.get("schema_version") != "dronedream.development-depth-fault.v1"
            or fault.get("development_only") is not True
            or fault.get("fault_kind") != "depth-frame-drop"
            or fault.get("activated") is not True or fault.get("recovered") is not True
            or type(start) is not int or type(end) is not int or not 0 < start < end
            or type(duration) not in (int, float) or not 0 < duration <= 30
            or abs(end - start - duration * 1000) > 1000):
        raise ValueError("TEACHER_ADVISOR_FAULT_WINDOW_INVALID")
    counts = {"before": 0, "held_or_unavailable": 0, "after": 0}
    prior, braking_at = _fault_command_boundaries(depth_rows or [], start, end)
    inflight = 0
    previous = -1
    for row in rows:
        stamp = row.get("recorded_at_unix_ms")
        if type(stamp) is not int or stamp < previous:
            raise ValueError("TEACHER_ADVISOR_EXECUTOR_TIME_INVALID")
        previous = stamp
        motion = row.get("status") == "accepted" and row.get("action") in {
            "continue", "slow", "replan"}
        if start <= stamp <= end:
            if motion:
                command = prior.get(row.get("command_sha256"))
                if (command is None or braking_at is None or stamp >= braking_at
                        or stamp >= command.valid_until_unix_ms
                        or row.get("command_generated_at_unix_ms") != command.generated_at_unix_ms
                        or row.get("command_observation_sha256") != command.observation_sha256
                        or row.get("command_sequence") != command.observation_sequence
                        or row.get("command_valid_until_unix_ms") != command.valid_until_unix_ms
                        or row.get("action") != command.decision.action):
                    raise ValueError("TEACHER_ADVISOR_MOTION_DURING_FAULT")
                inflight += 1
            if (row.get("status") == "command-unavailable" or
                    (row.get("status") == "accepted" and row.get("action") == "hold")):
                counts["held_or_unavailable"] += 1
        elif motion:
            counts["before" if stamp < start else "after"] += 1
    if min(counts.values()) < 1:
        raise ValueError("TEACHER_ADVISOR_FAULT_RECOVERY_NOT_OBSERVED")
    if inflight:
        counts["inflight_pre_fault_commands"] = inflight
    return counts


# 功能：
#   验证观测和命令完整配对，找出仍有原始租期的故障前命令及第一条故障制动生成时刻。
# 输入：
#   rows：冻结的安全日志；start、end：故障区间，单位毫秒。
# 输出：
#   prior：故障前命令摘要索引；braking_at：首个制动生成时刻或 None。
def _fault_command_boundaries(rows: list[dict], start: int, end: int):
    prior, braking_at, previous = {}, None, -1
    maximum_age_ms = round(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS * 1000)
    for row in rows:
        stamp = row.get("recorded_at_unix_ms")
        if type(stamp) is not int or stamp < previous:
            raise ValueError("TEACHER_ADVISOR_DEPTH_TIME_INVALID")
        previous = stamp
        raw = row.get("command")
        if raw is None:
            continue
        command = RuntimeLocalSafetyCommand.model_validate(raw)
        observation = RuntimeLocalSafetyObservation.model_validate(row.get("observation"))
        if (command.observation_sha256 != sha256_json(observation)
                or command.observation_sequence != observation.sequence
                or command.source != "onboard" or observation.source != "onboard"):
            raise ValueError("TEACHER_ADVISOR_COMMAND_BINDING_INVALID")
        if (stamp < start and observation.stream_healthy
                and observation.observed_at_unix_ms <= command.generated_at_unix_ms < start
                and start < command.valid_until_unix_ms
                <= observation.observed_at_unix_ms + maximum_age_ms
                and command.decision.action in {"continue", "slow", "replan"}):
            prior[sha256_json(command)] = command
        if (start <= stamp <= end and not observation.stream_healthy
                and command.decision.action == "hold"
                and start <= command.generated_at_unix_ms <= end):
            braking_at = (command.generated_at_unix_ms if braking_at is None else
                          min(braking_at, command.generated_at_unix_ms))
    return prior, braking_at


# 功能：
#   1. 仅接纳正常落地、仅因故障注入被排除的当前教师感知记录，保留原失败状态。
#   2. 冻结任务、故障、写盘、观测及执行日志摘要；不创建动作标签或飞行资格。
# 输入：
#   root：完整教师故障运行的 simulation 目录。
# 输出：
#   receipt：明确限定用途的来源回执；observations、depth：原始冻结数据。
def read_teacher_fault_advisor_source(root: Path):
    root = root.absolute()
    evidence, evidence_sha = read_evidence_object(root / "mission_evidence.json")
    gates = evidence.get("gates")
    if (evidence.get("schema_version") != "dronedream.generic-px4-gazebo-run.v1"
            or evidence.get("status") != "failed" or type(gates) is not dict or not gates
            or {k for k, v in gates.items() if v is not True} != {
                "development_fault_injection_absent"}
            or gates.get("landing_confirmed") is not True
            or gates.get("executor_completed") is not True):
        raise ValueError("TEACHER_ADVISOR_PHYSICAL_RUN_INVALID")
    measurements = evidence.get("measurements")
    learning = measurements.get("simulation_learning") if type(measurements) is dict else None
    if (type(learning) is not dict or learning.get("observations_recorded") is not True
            or learning.get("deterministic_teacher_control") is not True
            or learning.get("model_control_qualification_granted") is not False
            or learning.get("teacher_contract_sha256") != SIMULATION_TEACHER_CONTRACT_SHA256):
        raise ValueError("TEACHER_ADVISOR_CURRENT_CONTRACT_REQUIRED")
    artifacts = evidence.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("TEACHER_ADVISOR_ARTIFACTS_MISSING")
    files = {
        "observations": ("learning-observations.jsonl", "learning_observations_sha256"),
        "summary": ("learning-observation-summary.json", "learning_observation_summary_sha256"),
        "depth": ("depth-local-safety-history.jsonl", "depth_safety_history_sha256"),
        "executor": ("runtime-state/local-safety-executor-history.jsonl",
                     "local_safety_executor_history_sha256"),
        "fault": ("development-depth-fault.json", "development_fault_injection_sha256"),
        "writer": ("runtime-evidence-writer-summary.json",
                   "runtime_evidence_writer_summary_sha256"),
    }
    contents = read_evidence_dataset(root, {k: p for k, (p, _) in files.items()},
        {k: artifacts.get(d) for k, (_, d) in files.items()},
        error_prefix="TEACHER_ADVISOR_SOURCE_HASH_MISMATCH")
    objects = {k: decode_json(contents[k], limit=2 * 1024**2)
               for k in ("fault", "summary", "writer")}
    if any(type(v) is not dict for v in objects.values()):
        raise ValueError("TEACHER_ADVISOR_RECORDING_INVALID")
    summary = objects["summary"]
    if (any(objects[k].get("complete") is not True or objects[k].get("issue_code") is not None
            for k in ("summary", "writer"))
            or type(summary.get("completed")) is not int or summary["completed"] < 1
            or type(summary.get("submitted")) is not int
            or summary["submitted"] != summary["completed"]):
        raise ValueError("TEACHER_ADVISOR_RECORDING_INCOMPLETE")
    counts = verify_teacher_fault_window(
        objects["fault"], decode_evidence_rows(contents["executor"]),
        decode_evidence_rows(contents["depth"]))
    rows = decode_evidence_rows(contents["observations"])
    if len(rows) != summary["completed"]:
        raise ValueError("TEACHER_ADVISOR_RECORD_COUNT_MISMATCH")
    for row in rows:
        snapshot = row.get("snapshot")
        if (row.get("evidence_kind") != "simulation-observation-only"
                or row.get("model_invoked") is not False
                or row.get("control_authority_granted") is not False
                or row.get("policy_feature_contract_sha256")
                != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                or type(snapshot) is not dict):
            raise ValueError("TEACHER_ADVISOR_OBSERVATION_INVALID")
        content = dict(snapshot)
        digest = content.pop("snapshot_sha256", None)
        if digest != sha256_json(content):
            raise ValueError("TEACHER_ADVISOR_SNAPSHOT_HASH_MISMATCH")
        context = snapshot.get("strategic_context")
        task = context.get("task") if type(context) is dict else None
        if (type(task) is not dict or task.get("simulation_teacher_contract_sha256")
                != SIMULATION_TEACHER_CONTRACT_SHA256
                or type(row.get("recorded_at_unix_ms")) is not int
                or row["recorded_at_unix_ms"]
                != snapshot.get("control_reference_observed_at_unix_ms")):
            raise ValueError("TEACHER_ADVISOR_SNAPSHOT_CONTRACT_INVALID")
    split = recorded_mission_group(root, artifacts)
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in contents.items()}
    receipt = {
        "run_directory": str(root), "source_kind": "verified-teacher-depth-fault-observations",
        "mission_status": "failed", "mission_evidence_sha256": evidence_sha,
        "snapshot_file_sha256": hashes["observations"], "cycle_file_sha256": None,
        "depth_safety_history_sha256": hashes["depth"], "multimodal_dataset_records_sha256": None,
        "semantic_sha256": split.semantic_sha256, "mission_split": split.model_dump(mode="json"),
        "verification": "verified-teacher-fault-observations-only",
        "teacher_contract_sha256": SIMULATION_TEACHER_CONTRACT_SHA256,
        "verified_source_files_sha256": hashes, "fault_window_counts": counts,
        "allowed_roles": ["perception-health-critic", "state-anomaly-detector",
                          "cross-modal-consistency-critic"],
        "flight_qualification_granted": False, "behavior_action_labels_granted": False,
    }
    observations = RecordedSource(root / files["observations"][0], contents["observations"])
    depth = RecordedSource(root / files["depth"][0], contents["depth"])
    return receipt, observations, depth
