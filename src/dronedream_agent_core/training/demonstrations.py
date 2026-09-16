"""Compile verified, executed simulation demonstrations without an old policy.

The observation recorder has no control authority. Labels come exclusively from
accepted velocity transport receipts, not proposals or measured vehicle speed.
File hashes attest consistency, not authentication of an untrusted external run.
Only independently verified PX4/Gazebo runs produced by our executor are inputs.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import RuntimeLocalSafetyCommand
from ..control_execution_evidence import ControlApplicationRecord
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import requested_navigation_expert
from ..local_policy_port import compile_local_policy_features
from ..local_policy_training import LocalPolicyObservation, LocalPolicyTrainingSample
from ..pilot_control_mapping import PilotControlLimits
from ..plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from .evidence_files import (
    MAX_TRAINING_DATASET_BYTES,
    MAX_TRAINING_ROWS,
    decode_evidence_rows,
    read_evidence_dataset,
    read_evidence_object,
)
from .executed_control import executed_pilot_control
from .mission_groups import recorded_mission_group


# 功能：
#   对有界普通证据文件计算摘要，复核读取期间身份，不跟随链接或无限增长文件。
# 输入：
#   path：待检查的本地文件，最大 256 MiB。
# 输出：
#   digest：同一次读取字节的 SHA-256。
def file_sha256(path: Path) -> str:
    digest = hash_plugin_file(Path(path).absolute(), limit=MAX_TRAINING_DATASET_BYTES)
    return digest


# 功能：
#   从已经核对摘要的元数据字节严格解析对象，不重新读取可能变化的文件。
# 输入：
#   content：已固定的原始 JSON 字节，最大 4 MiB。
# 输出：
#   payload：无歧义字段及非有限数值的元数据对象。
def _json(content: bytes) -> dict:
    payload = decode_json(content, limit=4 * 1024 * 1024, node_limit=1_000_000)
    if type(payload) is not dict:
        raise ValueError("DEMONSTRATION_EXPECTED_JSON_OBJECT")
    return payload


# 功能：
#   解析已冻结证据中的完整行，拒绝空白行、歧义键和未结束的末行。
# 输入：
#   content：已核对摘要的原始 JSONL 字节。
# 输出：
#   rows：有界且严格解析的记录列表。
def _rows(content: bytes):
    rows = decode_evidence_rows(content)
    return rows


# 功能：
#   在连接输入与执行回执前验证规范的资产摘要身份。
# 输入：
#   value：候选小写 SHA-256 字符串。
# 输出：
#   value：校验通过的原摘要。
def _digest(value) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("DEMONSTRATION_ASSET_IDENTITY_MISSING")
    return value


# 功能：
#   验证运行目录内的规范路径并读取同一份图像字节，避免先验摘要后重开文件。
# 输入：
#   root：已固定且不包含链接的运行根目录。
#   relative：规范的运行内相对图像路径。
#   digest：记录中图像的文件摘要。
# 输出：
#   content：通过路径、身份、16 MiB 大小及摘要检查的图像字节。
def _bound_content(root: Path, relative: str, digest) -> bytes:
    portable_plugin_path(relative)
    content = read_plugin_file(root / relative, limit=16 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != _digest(digest):
        raise ValueError("DEMONSTRATION_FILE_HASH_MISMATCH:" + relative)
    return content


@dataclass(frozen=True)
class DemonstrationCorpus:
    """Executed labels plus unlabelled sensory history and independent split evidence."""

    samples: list[LocalPolicyTrainingSample]
    stream_groups: dict[str, str]
    receipts: list[dict]
    counts: dict[str, int]
    observations: list[LocalPolicyObservation] = field(default_factory=list)


# 功能：
#   校验实际前向 PNG 的文件、像素、尺寸和源时刻，显式无视觉模式允许没有图像。
# 输入：
#   root：当前运行根目录。
#   snapshot：控制输入的原始快照。
#   required：是否必须有当前相机证据。
# 输出：
#   digest：校验图像文件的摘要；明确无视觉且无图像时为 None。
def _visual(root: Path, snapshot: dict, *, required: bool) -> str | None:
    evidence = snapshot.get("visual_evidence", [])
    if not evidence:
        if required:
            raise ValueError("DEMONSTRATION_FORWARD_CAMERA_REQUIRED")
        return None
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError("DEMONSTRATION_VISUAL_IDENTITY_AMBIGUOUS")
    item = evidence[0]
    if not isinstance(item, dict) or item.get("kind") != "forward-rgb-camera":
        raise ValueError("DEMONSTRATION_FORWARD_CAMERA_REQUIRED")
    relative = item.get("relative_path")
    if not isinstance(relative, str):
        raise ValueError("DEMONSTRATION_VISUAL_PATH_MISSING")
    parts = PurePosixPath(relative.replace("\\", "/")).parts
    if len(parts) != 2 or parts[0] != "learning-observation-frames" or ".." in parts:
        raise ValueError("DEMONSTRATION_VISUAL_PATH_INVALID")
    content = _bound_content(root, "/".join(parts), item.get("sha256"))
    observed = item.get("observed_at_unix_ms")
    reference = snapshot.get("control_reference_observed_at_unix_ms")
    if (
        type(observed) is not int
        or type(reference) is not int
        or not 0 <= observed <= reference <= 2**63 - 1
        or not 0 <= reference - observed <= 200
    ):
        raise ValueError("DEMONSTRATION_VISUAL_TIME_INVALID")
    from PIL import Image

    with Image.open(BytesIO(content)) as image:
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (item.get("width"), item.get("height"))
            or any(not 64 <= size <= 1024 for size in image.size)
        ):
            raise ValueError("DEMONSTRATION_VISUAL_ENCODING_INVALID")
        if hashlib.sha256(image.tobytes()).hexdigest() != item.get("model_rgb_sha256"):
            raise ValueError("DEMONSTRATION_VISUAL_RGB_HASH_MISMATCH")
    digest = item["sha256"]
    return digest


# 功能：
#   1. 读取明确通过独立验证的教师运行，固定原始字节、路线划分及当前控制特征契约。
#   2. 仅由已接受的速度执行回执生成四轴标签，未执行观测只保留为感知历史。
#   3. 安全覆盖不作为模仿标签；文件摘要证明内容一致，不证明外部伪造运行的真实性。
# 输入：
#   roots：1 至 128 个本机运行目录。
#   require_visual：是否要求逐帧核对原相机图像。
# 输出：
#   corpus：已执行样本、无标签历史、空间组、来源回执和计数。
def collect_demonstrations(
    roots: list[Path], *, require_visual: bool = True
) -> DemonstrationCorpus:
    if type(roots) not in (list, tuple) or not 1 <= len(roots) <= 128:
        raise ValueError("DEMONSTRATION_ROOTS_INVALID")
    if type(require_visual) is not bool:
        raise ValueError("DEMONSTRATION_VISUAL_MODE_INVALID")
    roots = tuple(Path(directory).absolute() for directory in roots)
    samples, receipts, groups, observations = [], [], {}, []
    total_bytes = 0
    counts = Counter()
    seen_roots, seen_sources, seen_snapshots = set(), set(), set()
    for directory in roots:
        check_plain_plugin_path(directory)
        root = directory.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("DEMONSTRATION_ROOT_INVALID")
        if root in seen_roots:
            raise ValueError("DEMONSTRATION_DUPLICATE_RUN")
        seen_roots.add(root)
        evidence, evidence_digest = read_evidence_object(root / "mission_evidence.json")
        gates = evidence.get("gates")
        if (
            evidence.get("schema_version") != "dronedream.generic-px4-gazebo-run.v1"
            or evidence.get("status") != "verified"
            or not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("DEMONSTRATION_PHYSICAL_RUN_NOT_VERIFIED")
        measurements = evidence.get("measurements")
        learning = measurements.get("simulation_learning") if type(measurements) is dict else None
        if (
            type(learning) is not dict
            or learning.get("observations_recorded") is not True
            or learning.get("deterministic_teacher_control") is not True
            or learning.get("model_control_qualification_granted") is not False
        ):
            raise ValueError("DEMONSTRATION_EXPLICIT_TEACHER_REQUIRED")
        assets = evidence.get("artifacts", {})
        if type(assets) is not dict:
            raise ValueError("DEMONSTRATION_ASSET_IDENTITY_MISSING")
        for name in ("world_sha256", "semantic_sha256", "vehicle_sha256", "route_sha256"):
            _digest(assets.get(name))
        files = {
            "observations": ("learning-observations.jsonl", "learning_observations_sha256"),
            "summary": ("learning-observation-summary.json", "learning_observation_summary_sha256"),
            "commands": ("depth-local-safety-history.jsonl", "depth_safety_history_sha256"),
            "applications": (
                "runtime-state/control-applications.jsonl",
                "control_applications_sha256",
            ),
            "writer": (
                "runtime-evidence-writer-summary.json",
                "runtime_evidence_writer_summary_sha256",
            ),
        }
        contents = read_evidence_dataset(
            root,
            {key: relative for key, (relative, _) in files.items()},
            {key: _digest(assets.get(digest)) for key, (_, digest) in files.items()},
            error_prefix="DEMONSTRATION_FILE_HASH_MISMATCH",
        )
        total_bytes += sum(len(value) for value in contents.values())
        if total_bytes > MAX_TRAINING_DATASET_BYTES:
            raise ValueError("DEMONSTRATION_DATASET_CAPACITY")
        file_hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
        }
        summary, writer = _json(contents["summary"]), _json(contents["writer"])
        if (
            summary.get("complete") is not True
            or summary.get("issue_code") is not None
            or type(summary.get("submitted")) is not int
            or type(summary.get("completed")) is not int
            or not 1 <= summary.get("completed") <= MAX_TRAINING_ROWS
            or summary.get("submitted") != summary.get("completed")
            or writer.get("complete") is not True
            or writer.get("issue_code") is not None
        ):
            raise ValueError("DEMONSTRATION_RECORDING_INCOMPLETE")
        split = recorded_mission_group(root, assets)
        group = split.group_sha256
        commands = {}
        for row in _rows(contents["commands"]):
            if row.get("command") is not None:
                command = RuntimeLocalSafetyCommand.model_validate(row["command"])
                commands[sha256_json(command)] = command
        applications = {}
        previous_sequence, previous_time = 0, -1
        for row in _rows(contents["applications"]):
            application = ControlApplicationRecord.model_validate(row)
            if (
                application.sequence != previous_sequence + 1
                or application.accepted_at_unix_ms < previous_time
            ):
                raise ValueError("DEMONSTRATION_EXECUTION_ORDER_INVALID")
            previous_sequence, previous_time = application.sequence, application.accepted_at_unix_ms
            applications.setdefault(application.command_sha256, application)
        row_count, previous_source_time = 0, {}
        for row in _rows(contents["observations"]):
            row_count += 1
            if len(observations) >= MAX_TRAINING_ROWS:
                raise ValueError("DEMONSTRATION_OBSERVATION_CAPACITY")
            if (
                row.get("evidence_kind") != "simulation-observation-only"
                or row.get("policy_feature_contract_sha256")
                != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                or row.get("model_invoked") is not False
                or row.get("control_authority_granted") is not False
            ):
                raise ValueError("DEMONSTRATION_OBSERVATION_AUTHORITY_INVALID")
            snapshot = row.get("snapshot")
            if not isinstance(snapshot, dict):
                raise ValueError("DEMONSTRATION_OBSERVATION_MISSING")
            content = dict(snapshot)
            digest = content.pop("snapshot_sha256", None)
            if digest != sha256_json(content):
                raise ValueError("DEMONSTRATION_SNAPSHOT_HASH_MISMATCH")
            if row.get("recorded_at_unix_ms") != snapshot.get(
                "control_reference_observed_at_unix_ms"
            ):
                raise ValueError("DEMONSTRATION_OBSERVATION_TIME_MISMATCH")
            if digest in seen_snapshots:
                raise ValueError("DEMONSTRATION_REPEATED_OBSERVATION")
            seen_snapshots.add(digest)
            command_hash = _digest(row.get("evaluated_command_sha256"))
            command = commands.get(command_hash)
            if command is None:
                raise ValueError("DEMONSTRATION_EVALUATED_COMMAND_MISSING")
            if (
                command.requested_control_intent is not None
                or command.model_navigation_authorized
                or command.navigation_control_authority != "route-fallback"
            ):
                raise ValueError("DEMONSTRATION_MIXED_MODEL_AUTHORITY")
            task = snapshot.get("strategic_context", {}).get("task", {})
            if (
                task.get("local_navigation_output_mode") != "normalized-body-velocity"
                or not task.get("control_session_id")
                or task.get("navigation_goal_id") != command.navigation_goal_id
            ):
                raise ValueError("DEMONSTRATION_TASK_REFERENCE_MISMATCH")
            limits = PilotControlLimits(**task["normalized_pilot_control_limits"])
            batch = compile_local_policy_features(snapshot, include_candidate_features=False)
            if (
                not batch.realtime_features_ready
                or batch.temporal_evidence is None
                or batch.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            ):
                raise ValueError("DEMONSTRATION_CURRENT_DEPLOYMENT_INPUT_REQUIRED")
            temporal = batch.temporal_evidence
            if (
                temporal.sample_sha256 in seen_sources
                or temporal.observed_at_unix_ms <= previous_source_time.get(temporal.stream_id, -1)
            ):
                raise ValueError("DEMONSTRATION_SOURCE_REPLAY_OR_CLOCK_CONFLICT")
            seen_sources.add(temporal.sample_sha256)
            previous_source_time[temporal.stream_id] = temporal.observed_at_unix_ms
            if temporal.stream_id in groups and groups[temporal.stream_id] != group:
                raise ValueError("DEMONSTRATION_STREAM_MISSION_CONFLICT")
            groups[temporal.stream_id] = group
            visual = _visual(root, snapshot, required=require_visual)
            observation = LocalPolicyObservation(
                temporal_evidence=temporal,
                pilot_control_limits=limits,
                navigation_expert_role=requested_navigation_expert(snapshot),
                source_snapshot_sha256=digest,
                source_visual_sha256=visual,
                state_features=list(batch.state_features),
                candidate_features=[list(item) for item in batch.candidate_features],
                candidate_mask=list(batch.candidate_mask),
                realtime_features=list(batch.realtime_features),
                realtime_valid_mask=list(batch.realtime_valid_mask),
                control_feature_contract_sha256=batch.control_feature_contract_sha256,
            )
            # Unexecuted proposals have no action label, but their authentic
            # observations still belong in the causal sensory history.
            observations.append(observation)
            application = applications.get(command_hash)
            if application is None:
                counts["not_executed"] += 1
                continue
            if command.decision.action not in {"continue", "slow"}:
                counts["safety_override"] += 1
                continue
            if application.transport != "velocity-ned":
                raise ValueError("DEMONSTRATION_POSITION_CONTROL_IS_NOT_A_VELOCITY_LABEL")
            # Expired execution is an integrity failure, not a row to hide.
            target = executed_pilot_control(snapshot, command, application, limits=limits)
            axes = [target.forward_axis, target.right_axis, target.up_axis, target.yaw_axis]
            samples.append(
                LocalPolicyTrainingSample(
                    **observation.model_dump(),
                    target_pilot_control=axes,
                    target_action_index=11 if any(abs(value) > 1e-9 for value in axes) else 8,
                    # Successful behavior only, NOT a counterfactual action-risk label.
                    risk_target=0.0,
                    risk_proposed_control=[],
                )
            )
            counts["accepted"] += 1
        if row_count != summary.get("completed") or row_count == 0:
            raise ValueError("DEMONSTRATION_RECORD_COUNT_MISMATCH")
        counts["recorded"] += row_count
        receipts.append(
            {
                "root": str(root),
                "mission_group": group,
                "mission_split": split.model_dump(mode="json"),
                "mission_evidence_sha256": evidence_digest,
                "files": file_hashes,
                "observation_count": row_count,
                "observation_summary": summary,
            }
        )
    if not samples:
        raise ValueError("DEMONSTRATION_NO_EXECUTED_SAMPLES")
    corpus = DemonstrationCorpus(samples, groups, receipts, dict(counts), observations)
    return corpus


# 功能：
#   核对两份非空有界数据集的路线、流、快照、图像与物理来源不交叉，防止留出泄漏。
#   冲突时抛出对应来源错误，不授予飞行资格。
# 输入：
#   train：训练语料。
#   validation：独立验证语料。
# 输出：
#   None：不返回业务数据。
def validate_demonstration_splits(train: DemonstrationCorpus, validation: DemonstrationCorpus):
    for corpus in (train, validation):
        if (
            not isinstance(corpus, DemonstrationCorpus)
            or type(corpus.samples) is not list
            or not 1 <= len(corpus.samples) <= MAX_TRAINING_ROWS
            or type(corpus.observations) is not list
            or len(corpus.observations) > MAX_TRAINING_ROWS
            or type(corpus.stream_groups) is not dict
            or not corpus.stream_groups
        ):
            raise ValueError("DEMONSTRATION_CORPUS_INVALID")
    if set(train.stream_groups.values()) & set(validation.stream_groups.values()):
        raise ValueError("DEMONSTRATION_MISSION_GROUP_LEAKAGE")
    if set(train.stream_groups) & set(validation.stream_groups):
        raise ValueError("DEMONSTRATION_STREAM_LEAKAGE")
    for name in ("source_snapshot_sha256", "source_visual_sha256"):
        left = {getattr(sample, name) for sample in (*train.samples, *train.observations)} - {None}
        right = {
            getattr(sample, name) for sample in (*validation.samples, *validation.observations)
        } - {None}
        if left & right:
            raise ValueError("DEMONSTRATION_OBSERVATION_LEAKAGE:" + name)
    if {s.temporal_evidence.sample_sha256 for s in (*train.samples, *train.observations)} & {
        s.temporal_evidence.sample_sha256 for s in (*validation.samples, *validation.observations)
    }:
        raise ValueError("DEMONSTRATION_PHYSICAL_SOURCE_LEAKAGE")
