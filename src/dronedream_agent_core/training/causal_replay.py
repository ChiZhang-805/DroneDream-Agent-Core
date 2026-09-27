"""Content-bound BC/DAgger replay and frozen holdouts for offline refinement.

The producer writes actual exported labels/history, not hashes of the earlier
mixed-role input. Consumers parse the same bytes they verified. Old receipts
cannot acquire current provenance by reusing similarly named files.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from ..local_policy_training import LocalPolicyObservation, LocalPolicyTrainingSample
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from ..plugin_values import plugin_json_value
from .mission_groups import SPATIAL_SPLIT_CONTRACT, MissionGroupManifest, mission_group_evidence

CAUSAL_SPLIT_CONTRACT = SPATIAL_SPLIT_CONTRACT
MAX_REPLAY_FILE_BYTES = 256 * 1024 * 1024
MAX_REPLAY_TOTAL_BYTES = 512 * 1024 * 1024
MAX_REPLAY_RECORD_BYTES = 4 * 1024 * 1024
MAX_REPLAY_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_REPLAY_RECORDS = 250000
REPLAY_FILES = {
    "training_replay": "training-replay.jsonl",
    "validation_replay": "validation-replay.jsonl",
    "training_observations": "training-observations.jsonl",
    "validation_observations": "validation-observations.jsonl",
    "stream_groups": "stream-groups.json",
}


# 功能：
#   检查训练与留出的任务组和物理来源完全分离；不足以训练的无标签历史也参与隔离。
# 输入：
#   training：训练侧标签与观测记录。
#   validation：留出侧标签与观测记录。
#   manifest：具有地图路线证据的来源分组清单。
# 输出：
#   sets：训练和留出分别涉及的任务组集合列表。
def validate_split_sources(training, validation, manifest: MissionGroupManifest):
    if not isinstance(manifest, MissionGroupManifest):
        raise ValueError("CAUSAL_SPLIT_MANIFEST_INVALID")
    manifest = MissionGroupManifest.model_validate(manifest.model_dump())
    sets, identities = [], []
    for rows in (training, validation):
        if not rows or any(not isinstance(row, LocalPolicyObservation)
                           or row.temporal_evidence is None for row in rows):
            raise ValueError("CAUSAL_SPLIT_SOURCE_PROVENANCE_REQUIRED")
        streams = {row.temporal_evidence.stream_id for row in rows}
        if streams - manifest.groups.keys():
            raise ValueError("CAUSAL_SPLIT_STREAM_GROUP_MISSING")
        sets.append({manifest.groups[stream] for stream in streams})
        identities.append({row.temporal_evidence.sample_sha256 for row in rows})
    if sets[0] & sets[1]:
        raise ValueError("CAUSAL_TRAINING_MISSION_GROUP_LEAKAGE")
    if identities[0] & identities[1]:
        raise ValueError("CAUSAL_TRAINING_SOURCE_LEAKAGE")
    return sets


# 功能：
#   独立保存原统计并补充完整来源组；窗口统计不冒充全部历史路线的保护范围。
# 输入：
#   metrics：原始训练和评价统计字典。
#   groups：训练与留出的完整来源组集合。
# 输出：
#   result：独立拥有嵌套 JSON 值并保留原窗口分组的统计字典。
def bind_source_group_metrics(metrics: dict, groups) -> dict:
    result = copy_json(metrics, limit=MAX_REPLAY_RECORD_BYTES)
    if not isinstance(result, dict):
        raise ValueError("CAUSAL_SPLIT_METRICS_INVALID")
    for split, values in zip(("training", "validation"), groups, strict=True):
        result[split + "_window_groups"] = result.get(split + "_groups", [])
        result[split + "_groups"] = sorted(values)
    return result


# 功能：
#   1. 检查所有固定目标后独占导出回放及分组清单，散列与写入使用同一批字节。
#   2. 限制记录、文件和总字节；异常保留已写现场，不返回完整成功，也不删除既有产物。
# 输入：
#   output：已经存在的目标目录。
#   training：训练标签记录。
#   validation：留出标签记录。
#   training_history：训练侧的无标签历史。
#   validation_history：留出侧的无标签历史。
#   manifest：带实际路线证据的来源分组清单。
# 输出：
#   hashes：五个成功完整写入文件的字节摘要映射。
def export_replay_bundle(output: Path, *, training, validation, training_history,
                         validation_history, manifest: MissionGroupManifest) -> dict[str, str]:
    for filename in REPLAY_FILES.values():
        target = output / filename
        check_plain_plugin_path(target)
        if target.exists():
            raise FileExistsError(target)
    manifest = MissionGroupManifest.model_validate(manifest.model_dump())
    validate_split_sources([*training, *training_history], [*validation, *validation_history],
                           manifest)
    rows_by_name = dict(training_replay=training, validation_replay=validation,
        training_observations=training_history, validation_observations=validation_history)
    hashes = {}
    total_bytes = 0
    for name, filename in REPLAY_FILES.items():
        rows = [manifest] if name == "stream_groups" else rows_by_name[name]
        digest = hashlib.sha256()
        file_bytes = 0
        with (output / filename).open("xb") as stream:
            for index, row in enumerate(rows):
                if index >= MAX_REPLAY_RECORDS:
                    raise ValueError("CAUSAL_REPLAY_RECORD_LIMIT")
                content = (row.model_dump_json() + "\n").encode("utf-8")
                limit = MAX_REPLAY_MANIFEST_BYTES if name == "stream_groups" else (
                    MAX_REPLAY_RECORD_BYTES)
                decode_json(content, limit=limit)
                contract = (MissionGroupManifest if name == "stream_groups" else
                            LocalPolicyObservation if name.endswith("observations") else
                            LocalPolicyTrainingSample)
                contract.model_validate_json(content, strict=True)
                file_bytes += len(content)
                total_bytes += len(content)
                if file_bytes > MAX_REPLAY_FILE_BYTES or total_bytes > MAX_REPLAY_TOTAL_BYTES:
                    raise ValueError("CAUSAL_REPLAY_BYTE_LIMIT")
                if stream.write(content) != len(content):
                    raise OSError("CAUSAL_REPLAY_SHORT_WRITE")
                digest.update(content)
        hashes[name] = digest.hexdigest()
    return hashes


# 功能：
#   从固定文件清单有界捕获并核对回放字节；拒绝链接、读取漂移及未绑定内容。
# 输入：
#   paths：恰好五个回放文件的路径映射。
#   receipt：当前分组契约和五个文件预期摘要。
# 输出：
#   contents：已核对的独立文件字节映射，下游不得重新读路径替换这些内容。
def read_bound_replay(paths: dict[str, Path], receipt: dict) -> dict[str, bytes]:
    if not isinstance(paths, dict) or not isinstance(receipt, dict):
        raise ValueError("CAUSAL_REFINEMENT_REQUIRES_BOUND_REPLAY_AND_HOLDOUT")
    expected = receipt.get("replay_artifact_sha256")
    if (receipt.get("split_contract") != CAUSAL_SPLIT_CONTRACT
            or not isinstance(expected, dict) or set(expected) != set(REPLAY_FILES)
            or set(paths) != set(REPLAY_FILES)):
        raise ValueError("CAUSAL_REFINEMENT_REQUIRES_BOUND_REPLAY_AND_HOLDOUT")
    contents = {}
    remaining = MAX_REPLAY_TOTAL_BYTES
    for name, path in paths.items():
        content = read_plugin_file(path, limit=min(MAX_REPLAY_FILE_BYTES, remaining))
        if hashlib.sha256(content).hexdigest() != expected[name]:
            raise ValueError("CAUSAL_REFINEMENT_REPLAY_CONTENT_CHANGED:" + name)
        contents[name] = content
        remaining -= len(content)
    return contents


# 功能：
#   严格解码已捕获的固定回放集合并复核来源分组；空白记录、重复键和超预算内容均拒绝。
# 输入：
#   contents：读取绑定阶段返回的五个文件字节映射。
# 输出：
#   decoded：训练／留出标签及无标签观测列表。
#   manifest：重新验证的路线分组清单。
#   groups：训练及留出实际涉及的独立路线组。
def decode_replay(contents: dict[str, bytes]):
    if (not isinstance(contents, dict) or set(contents) != set(REPLAY_FILES)
            or any(not isinstance(value, bytes) or len(value) > MAX_REPLAY_FILE_BYTES
                   for value in contents.values())
            or sum(len(value) for value in contents.values()) > MAX_REPLAY_TOTAL_BYTES):
        raise ValueError("CAUSAL_REPLAY_BUNDLE_INVALID")
    decode_json(contents["stream_groups"], limit=MAX_REPLAY_MANIFEST_BYTES)
    manifest = MissionGroupManifest.model_validate_json(contents["stream_groups"])
    decoded = {}
    for name in REPLAY_FILES.keys() - {"stream_groups"}:
        cls = LocalPolicyObservation if name.endswith("observations") else LocalPolicyTrainingSample
        rows = []
        with io.BytesIO(contents[name]) as source:
            while line := source.readline(MAX_REPLAY_RECORD_BYTES + 1):
                if len(rows) >= MAX_REPLAY_RECORDS:
                    raise ValueError("CAUSAL_REPLAY_RECORD_LIMIT")
                value = decode_json(line, limit=MAX_REPLAY_RECORD_BYTES)
                if not isinstance(value, dict):
                    raise ValueError("CAUSAL_REPLAY_RECORD_INVALID")
                rows.append(cls.model_validate_json(line, strict=True))
        decoded[name] = rows
    groups = validate_split_sources(
        [*decoded["training_replay"], *decoded["training_observations"]],
        [*decoded["validation_replay"], *decoded["validation_observations"]], manifest)
    return decoded, manifest, groups


# 功能：
#   拒绝把历史训练已接触的路线重新标为留出验证路线。
# 输入：
#   receipt：旧基座的当前契约训练回执。
#   validation_groups：本次准备留出的路线组摘要集合。
# 输出：
#   None：不返回业务数据。
def require_unseen_validation(receipt: dict, validation_groups: set[str]) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("CAUSAL_WARMSTART_CURRENT_SPLIT_PROVENANCE_REQUIRED")
    metrics = receipt.get("metrics")
    training = metrics.get("training_groups") if isinstance(metrics, dict) else None
    if (receipt.get("split_contract") != CAUSAL_SPLIT_CONTRACT
            or not isinstance(training, list) or not training
            or any(not isinstance(group, str) or len(group) != 64
                   or set(group) - set("0123456789abcdef") for group in training)):
        raise ValueError("CAUSAL_WARMSTART_CURRENT_SPLIT_PROVENANCE_REQUIRED")
    if set(training) & validation_groups:
        raise ValueError("CAUSAL_WARMSTART_ALREADY_TRAINED_ON_VALIDATION")


# 功能：
#   从当前训练回执恢复当前及祖先调参留出组，迁移历史缺失时不能猜测补齐。
# 输入：
#   receipt：上一阶段保存的训练回执。
# 输出：
#   protected：仍不得用于新训练采集的路线组摘要集合。
def protected_validation_groups(receipt: dict) -> set[str]:
    if not isinstance(receipt, dict):
        raise ValueError("TRAINING_COLLECTION_REQUIRES_CURRENT_HELD_OUT_GROUPS")
    metrics = receipt.get("metrics")
    groups = metrics.get("validation_groups") if isinstance(metrics, dict) else None
    if (receipt.get("split_contract") != CAUSAL_SPLIT_CONTRACT
            or not isinstance(groups, list) or not groups
            or any(not isinstance(g, str) or len(g) != 64
                   or set(g) - set("0123456789abcdef") for g in groups)):
        raise ValueError("TRAINING_COLLECTION_REQUIRES_CURRENT_HELD_OUT_GROUPS")
    ancestors = metrics.get("historical_validation_groups", [])
    has_parent = bool(receipt.get("initial_policy_sha256") or receipt.get("encoder_initialization"))
    if (has_parent and "historical_validation_groups" not in metrics
            or not isinstance(ancestors, list) or len(ancestors) > 10_000
            or any(type(group) is not str or len(group) != 64
                   or set(group) - set("0123456789abcdef") for group in ancestors)):
        raise ValueError("TRAINING_COLLECTION_REQUIRES_ANCESTRAL_HELD_OUT_GROUPS")
    protected = set(groups) | set(ancestors)
    return protected


# 功能：
#   在构造 PPO／DAgger 仿真器之前绑定实际路线与语义文件，拒绝改名后混入的留出路线。
# 输入：
#   config：包含地图、路线路径及预期摘要的采集配置。
#   held_out_groups：额外必须保护的留出路线组。
# 输出：
#   actual：独立持有嵌套配置并合并留出保护的配置字典。
#   split：从实际已核对路线字节派生的空间分组证据。
def prepare_rollout_config(config, held_out_groups):
    actual = plugin_json_value(config, limit=MAX_REPLAY_RECORD_BYTES)
    if not isinstance(actual, dict):
        raise ValueError("TRAINING_ROLLOUT_CONFIG_INVALID")
    hashes = actual.get("asset_sha256", {})
    for name in ("route", "semantic"):
        content = read_plugin_file(Path(actual[name]), limit=64 * 1024 * 1024)
        if hashlib.sha256(content).hexdigest() != hashes.get(name):
            raise ValueError("TRAINING_ROLLOUT_ASSET_CHANGED:" + name)
        if name == "route":
            route = content
    split = mission_group_evidence(route, hashes["semantic"],
                                   expected_route_sha256=hashes["route"])
    blocked = set(actual.get("held_out_route_groups", [])) | set(held_out_groups)
    if split.group_sha256 in blocked:
        raise ValueError("TRAINING_ROLLOUT_USES_HELD_OUT_SPATIAL_ROUTE")
    actual["held_out_route_groups"] = sorted(blocked)
    return actual, split
