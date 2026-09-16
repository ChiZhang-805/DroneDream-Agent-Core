"""Shared, exclusive-create DAgger datasets consumed by the causal trainer."""

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import encode_json

from ..contracts import NormalizedPilotControl
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyObservation, LocalPolicyTrainingSample
from ..pilot_control_mapping import action_risk_features
from ..plugin_files import check_plain_plugin_path
from .dagger import ProposedActionRisk, TeacherCorrection
from .evidence_files import (
    MAX_TRAINING_DATASET_BYTES,
    MAX_TRAINING_ROW_BYTES,
    MAX_TRAINING_ROWS,
    decode_evidence_rows,
    read_evidence_dataset,
    read_evidence_object,
    training_json_value,
)
from .flight_environment import MODES, PilotAction
from .mission_groups import MissionGroupEvidence, MissionGroupManifest
from .stream_capture import StreamControlVisit


# 功能：
#   独占创建有界严格 JSONL 文件，检查短写及文件替换；失败保留现场而不返回成功摘要。
# 输入：
#   path：必须尚不存在的目标证据路径。
#   rows：字典或契约模型记录的迭代器。
# 输出：
#   checksum：实际完整写入字节的 SHA-256 摘要。
def write_rows(path, rows):
    path = Path(path).absolute()
    check_plain_plugin_path(path)
    digest = hashlib.sha256()
    total = 0
    with path.open("xb") as handle:
        initial = os.fstat(handle.fileno())
        for count, row in enumerate(rows, start=1):
            if count > MAX_TRAINING_ROWS:
                raise ValueError("TRAINING_DATASET_ROW_LIMIT_EXCEEDED")
            if isinstance(row, BaseModel):
                row = row.model_dump(mode="json")
            if type(row) is not dict:
                raise ValueError("TRAINING_DATASET_ROW_NOT_OBJECT")
            data = encode_json(training_json_value(row), limit=MAX_TRAINING_ROW_BYTES - 1,
                               node_limit=1_000_000).encode("utf-8") + b"\n"
            total += len(data)
            if total > MAX_TRAINING_DATASET_BYTES:
                raise ValueError("TRAINING_DATASET_SIZE_LIMIT_EXCEEDED")
            if handle.write(data) != len(data):
                raise OSError("TRAINING_DATASET_SHORT_WRITE")
            digest.update(data)
        handle.flush()
        final = os.fstat(handle.fileno())
        if not os.path.samestat(initial, final) or final.st_size != total:
            raise ValueError("TRAINING_DATASET_FILE_CHANGED_DURING_WRITE")
    check_plain_plugin_path(path)
    if not os.path.samestat(final, path.stat()):
        raise ValueError("TRAINING_DATASET_FILE_REPLACED")
    checksum = digest.hexdigest()
    return checksum


# 功能：
#   按物理观测时刻合并真实历史，剥离监督标签并保持整条路线分组，不虚构连续流下一状态。
# 输入：
#   visits：已验证的学生访问，可为因果步骤或连续流动作。
#   behavior_samples：以当前带标签样本补全的原始观测。
#   episode_groups：回合身份到地图路线分组证据的映射。
# 输出：
#   history：仅含观测信息的有序历史。
#   groups：传感器流到空间任务组的映射。
def training_history(visits, behavior_samples, *, episode_groups):
    observations, groups = {}, {}
    for visit in visits:
        split = episode_groups.get(visit.observation.episode_id)
        if not isinstance(split, MissionGroupEvidence):
            raise ValueError("DAGGER_NATIVE_ROUTE_GROUP_REQUIRED")
        split = MissionGroupEvidence.model_validate(split.model_dump())
        if split.semantic_sha256 != visit.observation.map_sha256:
            raise ValueError("DAGGER_NATIVE_ROUTE_MAP_MISMATCH")
        group = split.group_sha256
        samples = [*visit.observation.prior_observations, visit.observation.sample]
        if type(visit) is not StreamControlVisit:
            samples.extend([
                *visit.step.observation.prior_observations, visit.step.observation.sample])
        # Streaming imitation has no counterfactual next-state/reward field.
        for sample in samples:
            evidence = sample.temporal_evidence
            if evidence.stream_id in groups and groups[evidence.stream_id] != group:
                raise ValueError("DAGGER_HISTORY_MISSION_GROUP_CONFLICT")
            groups[evidence.stream_id] = group
            key = evidence.stream_id, evidence.observed_at_unix_ms
            old = observations.get(key)
            if old is not None and old.temporal_evidence != evidence:
                raise ValueError("DAGGER_HISTORY_SOURCE_CONFLICT")
            observations.setdefault(key, sample)
    # Current labelled inputs take precedence over earlier same-state snapshots
    # which may have different geometry age or no attached visual features.
    seen = set()
    for label in behavior_samples:
        evidence = label.temporal_evidence
        key = evidence.stream_id, evidence.observed_at_unix_ms
        if key in seen or key not in observations:
            raise ValueError("DAGGER_LABEL_SOURCE_DUPLICATE_OR_MISSING")
        seen.add(key)
        observations[key] = LocalPolicyObservation.model_validate(
            {name: getattr(label, name) for name in LocalPolicyObservation.model_fields}
        )
    # The actor history is an observation-only schema, even if a collector
    # supplied a training-sample subclass. Never serialize its supervision.
    history = [LocalPolicyObservation.model_validate({
        name: getattr(observations[key], name) for name in LocalPolicyObservation.model_fields
    }) for key in sorted(observations)]
    return history, groups


# 功能：
#   分别导出行为纠正、原提案风险、纯观测历史和空间分组，禁止覆盖既有证据。
# 输入：
#   output：输出目录。
#   visits：已接受的学生访问。
#   result：教师纠正与风险标注结果。
#   episode_groups：每个回合的独立路线分组证据。
# 输出：
#   hashes：各逻辑文件实际写入字节的摘要。
def write_training_artifacts(output, visits, result, *, episode_groups):
    history, groups = training_history(
        visits, result.behavior_samples, episode_groups=episode_groups)
    hashes = {}
    for name, rows in (
        ("behavior-corrections", result.behavior_samples),
        ("proposed-action-risk", result.action_risk_samples),
        ("correction-records", result.records),
        ("training-observations", history),
    ):
        hashes[name] = write_rows(output / (name + ".jsonl"), rows)
    manifest = MissionGroupManifest(groups=groups, evidence=list(episode_groups.values()))
    hashes["stream-groups"] = write_rows(output / "stream-groups.json", [manifest])
    return hashes


# 功能：
#   读取唯一完成回执及有界数据文件，核对视觉契约、原始摘要、标签语义和历史分组。
# 输入：
#   root：已完成离线标注的数据集目录。
#   expected_visual_input：可选的期望视觉输入契约。
# 输出：
#   labels：验证后的行为训练标签。
#   history：不含教师监督字段的观测历史。
#   groups：语义地图与路线分组清单。
#   receipt_digest：实际完成回执的原始字节摘要。
def load_dagger_training_artifacts(root, *, expected_visual_input=None):
    check_plain_plugin_path(root)
    receipts = [
        p
        for p in (root / "collection-receipt.jsonl", root / "annotation-receipt.jsonl")
        if p.is_file()
    ]
    if len(receipts) != 1:
        raise ValueError("DAGGER_DATASET_RECEIPT_MISSING_OR_AMBIGUOUS")
    receipt, receipt_digest = read_evidence_object(receipts[0])
    if not isinstance(receipt, dict):
        raise ValueError("DAGGER_DATASET_NOT_NATIVE_ANNOTATION")
    if expected_visual_input is not None:
        from .visual_lineage import require_matching_visual_input

        require_matching_visual_input(receipt.get("visual_input_contract"), expected_visual_input)
    if receipt.get("purpose") not in {
        "native-student-visited-dagger",
        "grounded-native-transition-annotation",
        "grounded-native-stream-annotation",
    }:
        raise ValueError("DAGGER_DATASET_NOT_NATIVE_ANNOTATION")
    files = {
        "behavior-corrections": "behavior-corrections.jsonl",
        "training-observations": "training-observations.jsonl",
        "stream-groups": "stream-groups.json",
        "correction-records": "correction-records.jsonl",
        "proposed-action-risk": "proposed-action-risk.jsonl",
    }
    expected = receipt.get("file_sha256")
    # 原生采集入口同时保存原始访问；独立离线标注入口只有五项训练产物。
    # 只接纳这一项明确的附属来源并一起核验，不能简单过滤掉所有未知清单项。
    if (receipt.get("purpose") == "grounded-native-stream-annotation"
            and type(expected) is dict and "student-visits" in expected):
        files["student-visits"] = "student-visits.jsonl"
    contents = read_evidence_dataset(root, files, expected,
                                     error_prefix="DAGGER_DATASET_CONTENT_CHANGED")
    if "student-visits" in contents:
        # 此文件仅保留采集原貌，不作为教师标签或奖励转移输入。
        visits = decode_evidence_rows(contents["student-visits"])
        if (not visits or type(receipt.get("student_steps")) is not int
                or len(visits) != receipt["student_steps"]
                or any(row.get("not_a_reward_transition") is not True for row in visits)):
            raise ValueError("DAGGER_DATASET_STUDENT_VISIT_INVENTORY_INVALID")
    labels = [LocalPolicyTrainingSample.model_validate(row)
              for row in decode_evidence_rows(contents["behavior-corrections"])]
    history = [LocalPolicyObservation.model_validate(row)
               for row in decode_evidence_rows(contents["training-observations"])]
    risks = [LocalPolicyTrainingSample.model_validate(row)
             for row in decode_evidence_rows(contents["proposed-action-risk"])]
    group_rows = decode_evidence_rows(contents["stream-groups"])
    if len(group_rows) != 1:
        raise ValueError("DAGGER_DATASET_GROUP_MANIFEST_AMBIGUOUS")
    groups = MissionGroupManifest.model_validate(group_rows[0])
    records = decode_evidence_rows(contents["correction-records"])
    if not labels or not history or len(records) != len(labels) or any(
        row.get("evidence_kind") != "px4-gazebo" or row.get("annotation_mode") != "after-quiescence"
        for row in records
    ):
        raise ValueError("DAGGER_DATASET_CORRECTION_RECORDS_INVALID")
    _validate_record_labels(records, labels, risks)
    streams = {sample.temporal_evidence.stream_id for sample in [*labels, *history]}
    if streams != set(groups.groups):
        raise ValueError("DAGGER_HISTORY_GROUP_STREAMS_DIFFER")
    return labels, history, groups, receipt_digest


# 功能：
#   逐条核对原提案、教师纠正及风险张量，防止摘要自洽但标签对应关系错误的数据进入训练。
# 输入：
#   records：离线纠正记录。
#   labels：行为训练标签。
#   risks：原提案的动作条件风险标签。
# 输出：
#   None：不返回业务数据。
def _validate_record_labels(records, labels, risks):
    expected_risks = []
    identities = set()
    for row, label in zip(records, labels, strict=True):
        identity = row["episode_id"], row["sequence"]
        proposal = PilotAction.model_validate(row["student_proposal"])
        correction = TeacherCorrection.model_validate(row["teacher_correction"])
        if (identity in identities or row.get("teacher_selected") is not False
                or correction.observation_sha256 != row.get("observation_sha256")
                or label.target_action_index != MODES.index(correction.action.mode) + 8
                or label.target_pilot_control != correction.action.axes
                or label.risk_target != correction.verified_action_risk
                or label.risk_proposed_control):
            raise ValueError("DAGGER_DATASET_CORRECTION_LABEL_MISMATCH")
        identities.add(identity)
        if row.get("proposed_action_risk") is None:
            continue
        risk = ProposedActionRisk.model_validate(row["proposed_action_risk"])
        if (proposal.mode != "pilot-control" or label.pilot_control_limits is None
                or risk.observation_sha256 != correction.observation_sha256
                or risk.proposed_action_sha256 != sha256_json(proposal)):
            raise ValueError("DAGGER_DATASET_ACTION_RISK_BINDING_MISMATCH")
        features = action_risk_features(NormalizedPilotControl(
            **dict(zip(("forward_axis", "right_axis", "up_axis", "yaw_axis"),
                       proposal.axes, strict=True))), label.pilot_control_limits, harness_scale=1.)
        expected_risks.append(label.model_copy(update={
            "risk_target": risk.risk, "risk_proposed_control": list(features)}))
    if risks != expected_risks:
        raise ValueError("DAGGER_DATASET_ACTION_RISK_LABEL_MISMATCH")
