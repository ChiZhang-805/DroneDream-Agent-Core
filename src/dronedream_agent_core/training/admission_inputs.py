"""Freeze explicit admission inputs before any benchmark; never discover substitute assets."""

import argparse
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath

from dronedream_plugin_sdk.protocol import decode_json

from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..local_policy_packages import load_local_policy_package
from ..local_policy_training import LocalPolicyObservation
from ..plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .evidence_publication import publish_evidence_bytes
from .mission_groups import MissionGroupEvidence, MissionGroupManifest

MAX_ADMISSION_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ADMISSION_DATA_BYTES = 256 * 1024 * 1024
MAX_ADMISSION_RECEIPT_BYTES = 4 * 1024 * 1024


# 功能：
#   1. 在调用方独占目录复制模型和全部显式数据、回执，再核对模型包身份。
#   2. 所有后续评估只消费私有副本；总量有界，不推理、不发布资格，也不修改原参数对象。
# 输入：
#   arguments：解析完成且输出位置已经预验的评估命令行参数。
#   root：本次临时输入目录，结束时由调用方清理。
# 输出：
#   frozen：输入路径全部指向固定副本、输出位置不变的独立参数对象。
def freeze_admission_inputs(arguments: argparse.Namespace, root: Path) -> argparse.Namespace:
    check_plain_plugin_path(root)
    if not root.is_dir():
        raise ValueError("admission input root must be an existing private directory")
    frozen = argparse.Namespace(**vars(arguments))
    remaining = MAX_ADMISSION_SOURCE_BYTES
    indexed = {}
    inputs = root / "inputs"
    inputs.mkdir()

    # 功能：
    #   按实际复制字节消耗总预算；相同路径和预算复用同一私有副本，不二次打开来源消费。
    # 输入：
    #   path：明确指定的原始数据或回执路径。
    #   limit：该类文件的单文件预算。
    # 输出：
    #   destination：可供后续解析、计算和取摘要的固定副本。
    def freeze_file(path, limit):
        nonlocal remaining
        check_plain_plugin_path(path)
        source = path.resolve(strict=True)
        key = (source, limit)
        if key in indexed:
            destination = indexed[key]
            return destination
        destination = inputs / f"input-{len(indexed):04d}{source.suffix}"
        hash_plugin_file(source, limit=min(limit, remaining), destination=destination)
        remaining -= destination.stat().st_size
        indexed[key] = destination
        return destination

    package = load_local_policy_package(arguments.package)
    causal = package.manifest.navigation_architecture == "causal-gru-control"
    heading_path = getattr(arguments, 'heading_observations', None)
    if package.manifest.requires_heading_evidence() != (
            heading_path is not None):
        raise ValueError('HEADING_ADMISSION_EXPLICIT_SOURCE_REQUIRED')
    history_paths = [
        getattr(arguments, name, None) for name in ("validation_observations", "stream_groups")
    ]
    if causal and any(path is None for path in history_paths):
        raise ValueError("causal admission requires explicit observations and stream groups")
    if not causal and any(path is not None for path in history_paths):
        raise ValueError("noncausal admission cannot consume causal observation sources")
    manifest_content = read_plugin_file(package.manifest_path, limit=2 * 1024 * 1024)
    remaining -= len(manifest_content)
    package_root = root / "package"
    package_root.mkdir()
    for artifact in package.manifest.artifacts:
        destination = package_root.joinpath(*PurePosixPath(artifact.relative_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hash_plugin_file(
            package.artifact_paths[artifact.role], limit=remaining, destination=destination
        )
        remaining -= destination.stat().st_size
        if digest != artifact.sha256:
            raise ValueError("ADMISSION_MODEL_CHANGED_WHILE_FREEZING")
    publish_evidence_bytes(package_root / "manifest.json", manifest_content, limit=2 * 1024 * 1024)
    copied = load_local_policy_package(package_root)
    if copied.package_sha256 != package.package_sha256:
        raise ValueError("ADMISSION_MANIFEST_CHANGED_WHILE_FREEZING")
    frozen.package = package_root
    frozen.expert_training_groups = set()
    frozen.risk_training_evidence = None
    if causal:
        # 因果包必须把完整训练来源一起冻结，不能只复制权重后失去独立留出的依据。
        from .artifact_assembly import expert_spatial_groups, validate_expert_training_receipt
        from .ensemble_lineage import load_base_lineage

        lineage, lineage_digest = load_base_lineage(package)
        assembly = freeze_file(package.root / "assembly-receipt.json", MAX_ADMISSION_RECEIPT_BYTES)
        if hash_plugin_file(assembly, limit=MAX_ADMISSION_RECEIPT_BYTES) != lineage_digest:
            raise ValueError("ADMISSION_LINEAGE_CHANGED_WHILE_FREEZING")
        training_groups, validation_groups = set(), set()
        for artifact in package.manifest.artifacts:
            name = f"training-evidence/{artifact.role}.json"
            entry = lineage["expert_evidence"].get(artifact.role)
            if (
                not isinstance(entry, dict)
                or entry.get("training_receipt_path") != name
                or (entry.get("artifact_sha256") != artifact.sha256)
            ):
                raise ValueError("ADMISSION_EXPERT_LINEAGE_INVALID")
            evidence_path = freeze_file(package.root / name, MAX_ADMISSION_RECEIPT_BYTES)
            content = read_plugin_file(evidence_path, limit=MAX_ADMISSION_RECEIPT_BYTES)
            if hashlib.sha256(content).hexdigest() != entry.get("training_receipt_sha256"):
                raise ValueError("ADMISSION_EXPERT_TRAINING_RECEIPT_CHANGED")
            evidence = decode_json(content, limit=MAX_ADMISSION_RECEIPT_BYTES)
            validate_expert_training_receipt(
                artifact.role, artifact.sha256, evidence, package.manifest
            )
            if artifact.role == 'risk-critic':
                frozen.risk_training_evidence = evidence_path
            if artifact.role != "perception-encoder":
                trained, held_out = expert_spatial_groups(artifact.role, evidence)
                training_groups.update(trained)
                validation_groups.update(held_out)
        if training_groups & validation_groups:
            raise ValueError("ADMISSION_CROSS_EXPERT_TRAINING_HOLDOUT_LEAKAGE")
        frozen.expert_training_groups = training_groups | validation_groups
    frozen.validation_data = freeze_file(arguments.validation_data, MAX_ADMISSION_DATA_BYTES)
    for name, limit in (
        ("validation_observations", MAX_ADMISSION_DATA_BYTES),
        ("stream_groups", 64 * 1024 * 1024),
        ("heading_observations", MAX_ADMISSION_DATA_BYTES),
    ):
        source = getattr(arguments, name, None)
        setattr(frozen, name, freeze_file(source, limit) if source is not None else None)
    for name in (
        "dataset_receipt",
        "visual_encoding_receipt",
        "composition_receipt",
        "augmentation_receipt",
        "rebinding_receipt",
        "training_receipt",
        "navigation_training_dataset_receipt",
        "inherited_admission_receipt",
    ):
        source = getattr(arguments, name)
        setattr(
            frozen,
            name,
            freeze_file(source, MAX_ADMISSION_RECEIPT_BYTES) if source is not None else None,
        )
    for name in ("advisor_validation_data", "advisor_training_receipt", "advisor_dataset_receipt"):
        values = getattr(arguments, name)
        if not isinstance(values, list) or len(values) > 32:
            raise ValueError("ADMISSION_ADVISOR_SOURCE_COUNT_INVALID")
        limit = (
            MAX_ADMISSION_DATA_BYTES
            if name == "advisor_validation_data"
            else MAX_ADMISSION_RECEIPT_BYTES
        )
        setattr(frozen, name, [freeze_file(path, limit) for path in values])
    risk_roots = getattr(arguments, 'risk_validation_data', [])
    if type(risk_roots) is not list or len(risk_roots) > 16:
        raise ValueError('ADMISSION_RISK_SOURCE_COUNT_INVALID')
    frozen.risk_validation_data = []
    if risk_roots:
        from .action_risk_artifacts import FILES
        for index, source_root in enumerate(risk_roots):
            check_plain_plugin_path(source_root)
            destination_root = root / f'action-risk-{index:02d}'
            destination_root.mkdir()
            for name in ('dataset-receipt.jsonl', *FILES.values()):
                destination = destination_root / name
                hash_plugin_file(source_root / name, limit=min(remaining, MAX_ADMISSION_DATA_BYTES),
                                 destination=destination)
                remaining -= destination.stat().st_size
            frozen.risk_validation_data.append(destination_root)
    return frozen


# 功能：
#   1. 按已确认的模型架构解析对应数据生产者的回执，不把历史任务名划分伪装成当前空间留出。
#   2. 当前因果数据重算路线分组，并拒绝已用于任一专家训练或调参的验证路线。
# 输入：
#   receipt：严格解析的导航数据集原回执。
#   causal：当前包是否为因果连续控制架构。
#   historical_groups：已从完整专家训练来源验证的历史路线组集合。
# 输出：
#   normalized：供准入统一读取的分区摘要与来源，不替换或改写磁盘原回执。
def navigation_dataset_contract(
    receipt: dict, *, causal: bool, historical_groups: set[str]
) -> dict:
    if not isinstance(receipt, dict):
        raise ValueError("navigation dataset receipt must be an object")
    if not causal:
        if (
            receipt.get("schema_version") != "dronedream.local-policy-dataset-receipt.v1"
            or receipt.get("split_method") != "held-out-source-campaign"
            or receipt.get("verified_sources_required") is not True
        ):
            raise ValueError("legacy navigation dataset receipt contract is invalid")
        normalized = dict(receipt)
    else:
        if (
            receipt.get("split_method") != "map-spatial-route"
            or receipt.get("label_source") != "verified-executed-simulation-teacher-control"
            or receipt.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or receipt.get("qualified_for_flight") is not False
        ):
            raise ValueError("causal admission requires current executed spatial dataset evidence")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {"training", "validation"}:
            raise ValueError("causal admission dataset partitions are invalid")
        normalized = {}
        groups = {}
        for split, values in outputs.items():
            if not isinstance(values, dict):
                raise ValueError("causal admission dataset partition is invalid")
            sources = values.get("sources")
            if not isinstance(sources, list) or not 1 <= len(sources) <= 10_000:
                raise ValueError("causal admission dataset source evidence is missing")
            groups[split] = set()
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("causal admission source is invalid")
                evidence = MissionGroupEvidence.model_validate(source.get("mission_split"))
                if evidence.group_sha256 != source.get("mission_group"):
                    raise ValueError("causal admission spatial source identity differs")
                groups[split].add(evidence.group_sha256)
            normalized[f"{split}_output_sha256"] = values.get("sha256")
            normalized[f"{split}_sources"] = sources
        if groups["training"] & groups["validation"]:
            raise ValueError("causal admission spatial partitions overlap")
        if not historical_groups or groups["validation"] & historical_groups:
            raise ValueError("causal admission reuses an expert training or tuning spatial group")
    digests = [normalized.get(f"{split}_output_sha256") for split in ("training", "validation")]
    if (
        any(
            type(value) is not str or len(value) != 64 or set(value) - set("0123456789abcdef")
            for value in digests
        )
        or digests[0] == digests[1]
    ):
        raise ValueError("navigation dataset partition hashes are invalid or identical")
    return normalized


# 功能：
#   1. 从显式历史和分组文件读取已绑定字节，核对全部流的空间来源及真实观测数量。
#   2. 历史仅补充已发生的传感器信息，不补造标签，不接受训练路线或未声明路线混入。
# 输入：
#   observation_path：当前验证集完整观测历史路径。
#   group_path：当前数据生产者输出的流与路线映射路径。
#   receipt：已验证空间分区的原始数据集回执。
# 输出：
#   observations：重新验证、来源属于本次验证空间组的完整观测列表。
def read_admission_observations(observation_path: Path, group_path: Path, receipt: dict):
    partition = receipt["outputs"]["validation"]
    content = read_plugin_file(observation_path, limit=MAX_ADMISSION_DATA_BYTES)
    groups_content = read_plugin_file(group_path, limit=64 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != partition.get(
        "observation_history_sha256"
    ) or hashlib.sha256(groups_content).hexdigest() != receipt.get("stream_groups_sha256"):
        raise ValueError("causal admission history or stream groups differ from dataset receipt")
    manifest = MissionGroupManifest.model_validate(
        decode_json(groups_content, limit=64 * 1024 * 1024)
    )
    expected = {source["mission_group"] for source in partition["sources"]}
    observations, actual_groups = [], set()
    with BytesIO(content) as stream:
        while line := stream.readline(MAX_ADMISSION_RECEIPT_BYTES + 1):
            if len(observations) >= 250_000:
                raise ValueError("causal admission observation count exceeds budget")
            decode_json(line, limit=MAX_ADMISSION_RECEIPT_BYTES)
            row = LocalPolicyObservation.model_validate_json(line, strict=True)
            if row.temporal_evidence is None:
                raise ValueError("causal admission observation lacks temporal provenance")
            group = manifest.groups.get(row.temporal_evidence.stream_id)
            if group not in expected:
                raise ValueError("causal admission observation uses an undeclared spatial group")
            actual_groups.add(group)
            observations.append(row)
    count = partition.get("observation_count")
    if type(count) is not int or count != len(observations) or not observations:
        raise ValueError("causal admission observation count differs from dataset receipt")
    if actual_groups != expected:
        raise ValueError("causal admission observed spatial groups differ from declared sources")
    return observations


# 功能：
#   1. 严格匹配动作标签与完整原观测，按流内时间顺序回放，拒绝重复、缺失或被替换的来源。
#   2. 历史不需要离线视觉向量；当前计分帧使用该标签绑定的视觉向量，不挪用其他帧。
# 输入：
#   samples：调用方已重新验证的行为监督样本。
#   observations：可选独立完整观测；省略仅用于标签本身就是完整流的直接评估。
# 输出：
#   replay：按时间排序的观测与可选对应标签组成的元组列表。
def bind_admission_history(samples, observations=None):
    labelled = {}
    for sample in samples:
        if sample.temporal_evidence is None:
            raise ValueError("CAUSAL_ADMISSION_SOURCE_PROVENANCE_REQUIRED")
        identity = (sample.temporal_evidence.stream_id, sample.temporal_evidence.sample_sha256)
        if identity in labelled:
            raise ValueError("CAUSAL_ADMISSION_DUPLICATE_LABEL")
        labelled[identity] = sample
    observations = samples if observations is None else observations
    if not isinstance(observations, list) or not 1 <= len(observations) <= 250_000:
        raise ValueError("CAUSAL_ADMISSION_OBSERVATION_COUNT_INVALID")
    replay, seen, timestamps = [], set(), set()
    for row in observations:
        row = LocalPolicyObservation.model_validate(
            {name: getattr(row, name) for name in LocalPolicyObservation.model_fields}
        )
        evidence = row.temporal_evidence
        if evidence is None:
            raise ValueError("CAUSAL_ADMISSION_SOURCE_PROVENANCE_REQUIRED")
        identity = evidence.stream_id, evidence.sample_sha256
        instant = evidence.stream_id, evidence.observed_at_unix_ms
        if identity in seen or instant in timestamps:
            raise ValueError("CAUSAL_ADMISSION_DUPLICATE_OBSERVATION")
        if (
            row.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or any(row.candidate_mask)
            or any(value for candidate in row.candidate_features for value in candidate)
        ):
            raise ValueError("CAUSAL_ADMISSION_CURRENT_CONTROL_INPUT_REQUIRED")
        seen.add(identity)
        timestamps.add(instant)
        label = labelled.get(identity)
        if label is not None:
            if any(
                getattr(row, name) != getattr(label, name)
                for name in LocalPolicyObservation.model_fields
                if name != "visual_features"
            ):
                raise ValueError("CAUSAL_ADMISSION_LABEL_OBSERVATION_MISMATCH")
            if (
                label.target_action_index not in range(8, 12)
                or len(label.target_pilot_control) != 4
            ):
                raise ValueError("CAUSAL_ADMISSION_CONTINUOUS_LABEL_REQUIRED")
        replay.append((label if label is not None else row, label))
    if set(labelled) - seen:
        raise ValueError("CAUSAL_ADMISSION_LABEL_WITHOUT_SOURCE_OBSERVATION")
    replay.sort(
        key=lambda pair: (
            pair[0].temporal_evidence.stream_id,
            pair[0].temporal_evidence.observed_at_unix_ms,
        )
    )
    return replay
