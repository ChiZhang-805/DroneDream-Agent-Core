#!/usr/bin/env python3
"""Maintain explicit candidate-era baselines; never rewrite current causal controllers.

Current continuous-control roles use the causal role/ensemble training entrypoints.
This retained compatibility command does not grant runtime or flight authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dronedream_agent_core.asset_package_storage import publish_asset_directory
from dronedream_agent_core.local_expert_harness import NavigationExpertRole
from dronedream_agent_core.local_policy_packages import (
    LoadedLocalPolicyPackage,
    LocalPolicyPackageManifest,
    load_local_policy_package,
)
from dronedream_agent_core.local_policy_port import OnnxLocalPolicyBackend
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingMetrics,
    LocalPolicyTrainingSample,
    LocalRiskCriticMetrics,
    evaluate_local_policy,
    evaluate_local_risk_critic,
    expand_local_risk_critic_visual_inputs,
    load_local_policy_onnx,
    load_local_risk_critic_onnx,
    load_training_samples,
    parse_training_samples,
    train_local_policy,
    train_local_risk_critic,
    write_local_policy_package,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.evidence_files import read_evidence_object
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)

MINIMUM_TRAINING_SAMPLES = 50
MINIMUM_VALIDATION_SAMPLES = 50
MINIMUM_TRAINING_SAFE_SAMPLES = 20
MINIMUM_TRAINING_RISKY_SAMPLES = 10
MINIMUM_VALIDATION_SAFE_SAMPLES = 20
MINIMUM_VALIDATION_RISKY_SAMPLES = 10
MINIMUM_RISK_TRAINING_SAFE_SAMPLES = 20
MINIMUM_RISK_TRAINING_RISKY_SAMPLES = 20
MINIMUM_RISK_VALIDATION_SAFE_SAMPLES = 20
MINIMUM_RISK_VALIDATION_RISKY_SAMPLES = 20

_AUGMENTATION_PRESERVED_ROLES = frozenset(
    {
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
        "risk-critic",
        "perception-encoder",
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
)


# 功能：
#   对普通输入文件进行有界散列，不把可替换路径或无界读取当作来源证明。
# 输入：
#   path：样本、模型或回执路径。
# 输出：
#   digest：实际读取至多 256 MiB 字节的 SHA-256。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=256 * 1024 * 1024)
    return digest


# 功能：
#   为完整监督行生成稳定标识，供精确行重用检查；不替代整任务来源独立性。
# 输入：
#   sample：已校验的训练样本。
# 输出：
#   digest：包括空字段在内的序列化样本摘要。
def _sample_identity(sample: LocalPolicyTrainingSample) -> str:
    digest = hashlib.sha256(sample.model_dump_json(exclude_none=False).encode("utf-8")).hexdigest()
    return digest


# 功能：
#   按风险监督值统计安全与危险样本，不把低风险扫描或停留动作冒充危险样本。
# 输入：
#   sample：带独立风险标签的监督样本。
# 输出：
#   category：安全或危险类别名称。
def _risk_class(sample: LocalPolicyTrainingSample) -> str:
    category = _risk_target_class(sample)
    return category


# 功能：
#   找出此增补器无法保留的角色，防止重建包时静默丢失未知组件。
# 输入：
#   roles：原包包含的角色集合。
# 输出：
#   unsupported：不在可保留白名单中的角色。
def _unsupported_artifact_roles(roles: set[str]) -> set[str]:
    unsupported = roles - _AUGMENTATION_PRESERVED_ROLES
    return unsupported


# 功能：
#   首次视觉校准只扩展一次特征尾段，后续兼容模型保留宽度，拒绝不同视觉契约。
# 输入：
#   model：无视觉或已具兼容视觉尾段的风险网络。
#   visual_feature_count：需要的正整数视觉维数。
#   random_seed：首次扩展所用随机种子。
# 输出：
#   prepared：已具备目标视觉输入宽度的模型。
def _prepare_visual_risk_model_for_calibration(
    model: Any,
    *,
    visual_feature_count: int | None,
    random_seed: int,
) -> Any:
    if type(visual_feature_count) is not int or not 1 <= visual_feature_count <= 65_536:
        raise ValueError("visual risk calibration requires a positive feature count")
    if model.visual_feature_count == 0:
        prepared = expand_local_risk_critic_visual_inputs(
            model,
            visual_feature_count=visual_feature_count,
            random_seed=random_seed,
        )
        return prepared
    if model.visual_feature_count != visual_feature_count:
        raise ValueError("visual risk critic feature width differs from the package")
    prepared = model
    return prepared


# 功能：
#   选择指定专家的样本，并分别检查训练和验证分区的总量及风险双类支持。
# 输入：
#   samples：当前分区的完整样本。
#   role：精细操纵或恢复专家。
#   split：training 或 validation 分区名称。
# 输出：
#   selected：属于该专家且满足最小支持量的样本。
#   class_counts：安全和危险样本数量。
def _select_and_validate_samples(
    samples: list[LocalPolicyTrainingSample],
    *,
    role: NavigationExpertRole,
    split: str,
) -> tuple[list[LocalPolicyTrainingSample], dict[str, int]]:
    if role not in {"precision-maneuver-policy", "recovery-policy"} or split not in {
        "training",
        "validation",
    }:
        raise ValueError("invalid specialist role or split")
    samples = [LocalPolicyTrainingSample.model_validate(sample.model_dump()) for sample in samples]
    selected = [sample for sample in samples if sample.navigation_expert_role == role]
    counts = Counter(_risk_class(sample) for sample in selected)
    limits = {
        "training": (
            MINIMUM_TRAINING_SAMPLES,
            MINIMUM_TRAINING_SAFE_SAMPLES,
            MINIMUM_TRAINING_RISKY_SAMPLES,
        ),
        "validation": (
            MINIMUM_VALIDATION_SAMPLES,
            MINIMUM_VALIDATION_SAFE_SAMPLES,
            MINIMUM_VALIDATION_RISKY_SAMPLES,
        ),
    }
    minimum_total, minimum_safe, minimum_risky = limits[split]
    if len(selected) < minimum_total:
        raise RuntimeError(f"LOCAL_EXPERT_{split.upper()}_DATASET_TOO_SMALL")
    if counts["safe"] < minimum_safe:
        raise RuntimeError(f"LOCAL_EXPERT_{split.upper()}_SAFE_CLASS_TOO_SMALL")
    if counts["risky"] < minimum_risky:
        raise RuntimeError(f"LOCAL_EXPERT_{split.upper()}_RISKY_CLASS_TOO_SMALL")
    class_counts = {"safe": counts["safe"], "risky": counts["risky"]}
    return selected, class_counts


# 功能：
#   检查候选策略的运动授权、选择、风险召回及误差，不接受无双类支持的统计。
# 输入：
#   metrics：当前独立验证分区的实际评价结果。
# 输出：
#   None：不返回业务数据。
def _validate_metrics(metrics: LocalPolicyTrainingMetrics) -> None:
    metrics = LocalPolicyTrainingMetrics.model_validate(metrics.model_dump())
    if min(metrics.risky_sample_count, metrics.safe_sample_count) <= 0:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_CLASS_SUPPORT_MISSING")
    if metrics.motion_authorization_accuracy < 0.95:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_MOTION_AUTHORIZATION_TOO_LOW")
    if metrics.candidate_selection_accuracy < 0.9:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_CANDIDATE_ACCURACY_TOO_LOW")
    if metrics.risk_hold_recall < 0.95:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_RISK_HOLD_RECALL_TOO_LOW")
    if metrics.safe_motion_recall < 0.95:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_SAFE_MOTION_RECALL_TOO_LOW")
    if metrics.risk_mean_absolute_error > 0.2:
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_RISK_ERROR_TOO_HIGH")


# 功能：
#   拒绝训练与验证之间重复的完整监督行，来源任务独立性仍由回执另外验证。
# 输入：
#   training：训练分区样本。
#   validation：留出分区样本。
# 输出：
#   None：不返回业务数据。
def _assert_disjoint(
    training: list[LocalPolicyTrainingSample],
    validation: list[LocalPolicyTrainingSample],
) -> None:
    overlap = {_sample_identity(sample) for sample in training} & {
        _sample_identity(sample) for sample in validation
    }
    if overlap:
        raise RuntimeError("LOCAL_EXPERT_TRAINING_VALIDATION_OVERLAP")


# 功能：
#   核验标签来源、专家过滤、恢复证据与数据摘要；后备教师数据只允许显式仿真引导用途。
# 输入：
#   receipt_path：数据集来源回执。
#   training_path：训练数据路径。
#   validation_path：验证数据路径。
#   role：被训练的专家角色。
#   bootstrap_from_fallback：是否明确允许后备教师引导数据。
#   training_source_sha256：视觉编码之前的可选训练来源摘要。
#   validation_source_sha256：视觉编码之前的可选验证来源摘要。
# 输出：
#   receipt：通过来源和用途检查的回执对象。
def _validate_dataset_receipt(
    receipt_path: Path,
    *,
    training_path: Path,
    validation_path: Path,
    role: NavigationExpertRole,
    bootstrap_from_fallback: bool = False,
    training_source_sha256: str | None = None,
    validation_source_sha256: str | None = None,
) -> dict[str, object]:
    receipt, _ = read_evidence_object(receipt_path, limit=4 * 1024 * 1024)
    if receipt.get("schema_version") != "dronedream.local-policy-dataset-receipt.v1":
        raise RuntimeError("LOCAL_EXPERT_DATASET_RECEIPT_SCHEMA_INVALID")
    if receipt.get("split_method") != "held-out-source-campaign":
        raise RuntimeError("LOCAL_EXPERT_DATASET_NOT_MISSION_HELD_OUT")
    expected_label_source = (
        "fallback-navigation-expert-trace" if bootstrap_from_fallback else "navigation-expert-trace"
    )
    if receipt.get("navigation_label_source") != expected_label_source:
        raise RuntimeError("LOCAL_EXPERT_DATASET_HAS_ENTANGLED_ADVISOR_LABELS")
    expert_filter = receipt.get("navigation_expert_role_filter")
    if expert_filter is not None and expert_filter != role:
        raise RuntimeError("LOCAL_EXPERT_DATASET_ROLE_FILTER_MISMATCH")
    if bootstrap_from_fallback:
        if receipt.get("bootstrap_only") is not True:
            raise RuntimeError("LOCAL_EXPERT_BOOTSTRAP_MARKER_MISSING")
        if receipt.get("fallback_teacher_navigation_role") != ("local-navigation-policy"):
            raise RuntimeError("LOCAL_EXPERT_BOOTSTRAP_TEACHER_INVALID")
        if receipt.get("fallback_risky_target_policy") != (
            "teacher-risk-at-least-0.5-promoted-to-1.0"
        ):
            raise RuntimeError("LOCAL_EXPERT_BOOTSTRAP_RISK_TARGET_POLICY_INVALID")
    elif receipt.get("bootstrap_only") is True:
        raise RuntimeError("LOCAL_EXPERT_BOOTSTRAP_REQUIRES_EXPLICIT_OPT_IN")
    if receipt.get("verified_sources_required") is not True:
        raise RuntimeError("LOCAL_EXPERT_DATASET_SOURCE_NOT_VERIFIED")
    if receipt.get("expert_trace_required") is not True:
        raise RuntimeError("LOCAL_EXPERT_DATASET_TRACE_NOT_REQUIRED")
    if role == "recovery-policy" and receipt.get("recovery_episode_identity_required") is not True:
        raise RuntimeError("LOCAL_EXPERT_RECOVERY_EPISODE_IDENTITY_NOT_REQUIRED")
    if role == "recovery-policy" and receipt.get("recovery_outcome_progress_required") is not True:
        raise RuntimeError("LOCAL_EXPERT_RECOVERY_OUTCOME_NOT_REQUIRED")
    if role == "recovery-policy" and (
        type(receipt.get("recovery_samples_capped_per_episode")) is not int
        or receipt.get("recovery_samples_capped_per_episode") != 1
    ):
        raise RuntimeError("LOCAL_EXPERT_RECOVERY_EPISODE_SAMPLE_CAP_INVALID")
    if receipt.get("training_output_sha256") != (training_source_sha256 or _sha256(training_path)):
        raise RuntimeError("LOCAL_EXPERT_TRAINING_DATA_HASH_MISMATCH")
    if receipt.get("validation_output_sha256") != (
        validation_source_sha256 or _sha256(validation_path)
    ):
        raise RuntimeError("LOCAL_EXPERT_VALIDATION_DATA_HASH_MISMATCH")
    return receipt


# 功能：
#   提取两个分区的完整任务摘要并拒绝交叉重用，摘要必须为有效小写十六进制。
# 输入：
#   receipt：含训练和验证任务来源列表的回执。
# 输出：
#   training_sources：训练任务摘要集合。
#   validation_sources：与训练任务互不相交的验证任务摘要集合。
def _dataset_source_evidence(
    receipt: dict[str, object],
) -> tuple[set[str], set[str]]:
    split_hashes: list[set[str]] = []
    for split in ("training", "validation"):
        raw_sources = receipt.get(f"{split}_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise RuntimeError("LOCAL_EXPERT_DATASET_SOURCE_EVIDENCE_MISSING")
        hashes: set[str] = set()
        for source in raw_sources:
            if not isinstance(source, dict):
                raise RuntimeError("LOCAL_EXPERT_DATASET_SOURCE_EVIDENCE_INVALID")
            digest = source.get("mission_evidence_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not set(digest) <= set("0123456789abcdef")
            ):
                raise RuntimeError("LOCAL_EXPERT_DATASET_SOURCE_EVIDENCE_INVALID")
            hashes.add(digest)
        split_hashes.append(hashes)
    training_sources, validation_sources = split_hashes
    if training_sources & validation_sources:
        raise RuntimeError("LOCAL_EXPERT_DATASET_SOURCE_MISSION_OVERLAP")
    return training_sources, validation_sources


# 功能：
#   按独立风险网络真正消费的风险目标分类，0.5 及以上属于危险。
# 输入：
#   sample：已验证的监督样本。
# 输出：
#   category：安全或危险类别名称。
def _risk_target_class(sample: LocalPolicyTrainingSample) -> str:
    category = "risky" if sample.risk_target >= 0.5 else "safe"
    return category


# 功能：
#   检查独立风险专家的双类支持、召回与绝对误差，不使用行为动作类别替代风险标签。
# 输入：
#   metrics：独立留出数据上的风险网络指标。
# 输出：
#   None：不返回业务数据。
def _validate_risk_metrics(metrics: LocalRiskCriticMetrics) -> None:
    metrics = LocalRiskCriticMetrics.model_validate(metrics.model_dump())
    if min(metrics.risky_sample_count, metrics.safe_sample_count) <= 0:
        raise RuntimeError("LOCAL_RISK_CRITIC_VALIDATION_CLASS_SUPPORT_MISSING")
    if metrics.risk_hold_recall < 0.95:
        raise RuntimeError("LOCAL_RISK_CRITIC_VALIDATION_HOLD_RECALL_TOO_LOW")
    if metrics.safe_motion_recall < 0.95:
        raise RuntimeError("LOCAL_RISK_CRITIC_VALIDATION_SAFE_RECALL_TOO_LOW")
    if metrics.risk_mean_absolute_error > 0.2:
        raise RuntimeError("LOCAL_RISK_CRITIC_VALIDATION_ERROR_TOO_HIGH")


# 功能：
#   读取至多 128 组风险校准数据，绑定同次字节并验证完整任务与监督行的分区独立性。
# 输入：
#   specifications：训练、验证、来源回执三元组，视觉组另含两个编码回执。
#   perception_encoder_sha256：视觉编码器摘要。
#   visual_contract：视觉宽高、特征数与归一化契约。
# 输出：
#   training：合并的训练样本。
#   validation：合并的验证样本。
#   training_counts：训练风险双类数量。
#   validation_counts：验证风险双类数量。
#   evidence：每组实际消费内容及任务来源证据。
#   all_sources：本次风险校准涉及的完整任务摘要。
def _load_risk_calibration_datasets(
    specifications: list[list[Path]],
    *,
    perception_encoder_sha256: str | None = None,
    visual_contract: dict[str, object] | None = None,
) -> tuple[
    list[LocalPolicyTrainingSample],
    list[LocalPolicyTrainingSample],
    dict[str, int],
    dict[str, int],
    list[dict[str, object]],
    set[str],
]:
    if not 1 <= len(specifications) <= 128:
        raise ValueError("risk calibration requires one to 128 datasets")
    training: list[LocalPolicyTrainingSample] = []
    validation: list[LocalPolicyTrainingSample] = []
    training_sources: set[str] = set()
    validation_sources: set[str] = set()
    evidence: list[dict[str, object]] = []
    total_bytes = 0
    for specification in specifications:
        if len(specification) not in {3, 5}:
            raise RuntimeError("LOCAL_RISK_CRITIC_DATASET_SPECIFICATION_INVALID")
        training_path, validation_path, receipt_path = specification[:3]
        receipt, receipt_sha256 = read_evidence_object(receipt_path, limit=4 * 1024 * 1024)
        training_content = read_plugin_file(training_path, limit=256 * 1024 * 1024)
        validation_content = read_plugin_file(validation_path, limit=256 * 1024 * 1024)
        total_bytes += len(training_content) + len(validation_content)
        if total_bytes > 1024 * 1024 * 1024:
            raise ValueError("risk calibration data exceed the aggregate byte budget")
        training_sha256 = hashlib.sha256(training_content).hexdigest()
        validation_sha256 = hashlib.sha256(validation_content).hexdigest()
        if receipt.get("schema_version") != "dronedream.local-policy-dataset-receipt.v1":
            raise RuntimeError("LOCAL_RISK_CRITIC_DATASET_RECEIPT_SCHEMA_INVALID")
        if receipt.get("split_method") != "held-out-source-campaign":
            raise RuntimeError("LOCAL_RISK_CRITIC_DATASET_NOT_MISSION_HELD_OUT")
        if receipt.get("verified_sources_required") is not True:
            raise RuntimeError("LOCAL_RISK_CRITIC_DATASET_SOURCE_NOT_VERIFIED")
        if receipt.get("bootstrap_only") is True:
            raise RuntimeError("LOCAL_RISK_CRITIC_BOOTSTRAP_DATA_FORBIDDEN")
        if receipt.get("navigation_label_source") not in {
            "final-ensemble-decision",
            "navigation-expert-trace",
        }:
            raise RuntimeError("LOCAL_RISK_CRITIC_LABEL_SOURCE_INVALID")
        if len(specification) == 3:
            if receipt.get("training_output_sha256") != training_sha256:
                raise RuntimeError("LOCAL_RISK_CRITIC_TRAINING_DATA_HASH_MISMATCH")
            if receipt.get("validation_output_sha256") != validation_sha256:
                raise RuntimeError("LOCAL_RISK_CRITIC_VALIDATION_DATA_HASH_MISMATCH")
            visual_receipt_hashes: dict[str, str] | None = None
        else:
            if perception_encoder_sha256 is None or visual_contract is None:
                raise RuntimeError("LOCAL_RISK_CRITIC_VISUAL_CONTRACT_MISSING")
            training_visual_receipt_path, validation_visual_receipt_path = specification[3:]
            _validate_visual_encoding_receipt(
                training_visual_receipt_path,
                encoded_path=training_path,
                source_policy_sha256=str(receipt["training_output_sha256"]),
                perception_encoder_sha256=perception_encoder_sha256,
                expected_sample_count=receipt.get("training_sample_count"),
                expected_contract=visual_contract,
                encoded_sha256=training_sha256,
            )
            _validate_visual_encoding_receipt(
                validation_visual_receipt_path,
                encoded_path=validation_path,
                source_policy_sha256=str(receipt["validation_output_sha256"]),
                perception_encoder_sha256=perception_encoder_sha256,
                expected_sample_count=receipt.get("validation_sample_count"),
                expected_contract=visual_contract,
                encoded_sha256=validation_sha256,
            )
            visual_receipt_hashes = {
                "training": _sha256(training_visual_receipt_path),
                "validation": _sha256(validation_visual_receipt_path),
            }
        source_training, source_validation = _dataset_source_evidence(receipt)
        if training_sources & source_training or validation_sources & source_validation:
            raise RuntimeError("LOCAL_RISK_CRITIC_SOURCE_MISSION_DUPLICATED")
        training_sources.update(source_training)
        validation_sources.update(source_validation)
        training.extend(parse_training_samples(training_content))
        validation.extend(parse_training_samples(validation_content))
        del training_content, validation_content
        if len(training) + len(validation) > 250_000:
            raise ValueError("risk calibration exceeds the aggregate sample budget")
        evidence.append(
            {
                "training_data_sha256": training_sha256,
                "validation_data_sha256": validation_sha256,
                "dataset_receipt_sha256": receipt_sha256,
                "navigation_label_source": receipt["navigation_label_source"],
                "training_source_evidence_sha256": sorted(source_training),
                "validation_source_evidence_sha256": sorted(source_validation),
                "visual_encoding_receipts_sha256": visual_receipt_hashes,
            }
        )
    if training_sources & validation_sources:
        raise RuntimeError("LOCAL_RISK_CRITIC_SOURCE_MISSION_OVERLAP")
    _assert_disjoint(training, validation)
    training_counts = Counter(_risk_target_class(sample) for sample in training)
    validation_counts = Counter(_risk_target_class(sample) for sample in validation)
    minimums = (
        (
            "TRAINING",
            training_counts,
            MINIMUM_RISK_TRAINING_SAFE_SAMPLES,
            MINIMUM_RISK_TRAINING_RISKY_SAMPLES,
        ),
        (
            "VALIDATION",
            validation_counts,
            MINIMUM_RISK_VALIDATION_SAFE_SAMPLES,
            MINIMUM_RISK_VALIDATION_RISKY_SAMPLES,
        ),
    )
    for split, counts, minimum_safe, minimum_risky in minimums:
        if counts["safe"] < minimum_safe:
            raise RuntimeError(f"LOCAL_RISK_CRITIC_{split}_SAFE_CLASS_TOO_SMALL")
        if counts["risky"] < minimum_risky:
            raise RuntimeError(f"LOCAL_RISK_CRITIC_{split}_RISKY_CLASS_TOO_SMALL")
    all_sources = training_sources | validation_sources
    training_counts = {"safe": training_counts["safe"], "risky": training_counts["risky"]}
    validation_counts = {"safe": validation_counts["safe"], "risky": validation_counts["risky"]}
    return training, validation, training_counts, validation_counts, evidence, all_sources


# 功能：
#   将重训的专家绑定到原包、原数据回执及所有历史任务，不允许截断来源链。
# 输入：
#   receipt_path：原专家增补回执。
#   dataset_receipt_path：原训练数据集回执。
#   base_package_sha256：当前基座包摘要。
#   role：本次要重训的专家。
# 输出：
#   historical_set：包括原训练与验证任务在内的全部历史来源。
def _validate_source_augmentation_lineage(
    *,
    receipt_path: Path,
    dataset_receipt_path: Path,
    base_package_sha256: str,
    role: NavigationExpertRole,
) -> set[str]:
    receipt, _ = read_evidence_object(receipt_path, limit=4 * 1024 * 1024)
    source_dataset, source_dataset_sha256 = read_evidence_object(
        dataset_receipt_path, limit=4 * 1024 * 1024
    )
    if receipt.get("schema_version") != "dronedream.local-policy-augmentation-receipt.v1":
        raise RuntimeError("LOCAL_EXPERT_SOURCE_AUGMENTATION_SCHEMA_INVALID")
    if receipt.get("package_sha256") != base_package_sha256:
        raise RuntimeError("LOCAL_EXPERT_SOURCE_AUGMENTATION_PACKAGE_MISMATCH")
    if receipt.get("expert_role") != role:
        raise RuntimeError("LOCAL_EXPERT_SOURCE_AUGMENTATION_ROLE_MISMATCH")
    if receipt.get("dataset_receipt_sha256") != source_dataset_sha256:
        raise RuntimeError("LOCAL_EXPERT_SOURCE_DATASET_RECEIPT_HASH_MISMATCH")
    if source_dataset.get("schema_version") != "dronedream.local-policy-dataset-receipt.v1":
        raise RuntimeError("LOCAL_EXPERT_SOURCE_DATASET_RECEIPT_SCHEMA_INVALID")
    source_training, source_validation = _dataset_source_evidence(source_dataset)
    historical = receipt.get("historical_source_evidence_sha256")
    if historical is None:
        historical_set = source_training | source_validation
        return historical_set
    if (
        not isinstance(historical, list)
        or not historical
        or not all(
            isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in historical
        )
    ):
        raise RuntimeError("LOCAL_EXPERT_SOURCE_LINEAGE_INVALID")
    historical_set = set(historical)
    if not (source_training | source_validation).issubset(historical_set):
        raise RuntimeError("LOCAL_EXPERT_SOURCE_LINEAGE_INCOMPLETE")
    return historical_set


# 功能：
#   核对编码前后数据、编码器身份、严格整数形状和有限延迟，禁止编码回执授予资格。
# 输入：
#   receipt_path：编码证据回执。
#   encoded_path：编码后的样本文件。
#   source_policy_sha256：编码前样本摘要。
#   perception_encoder_sha256：要求使用的编码器摘要。
#   expected_sample_count：原数据的正整数行数。
#   expected_contract：原包要求的视觉契约。
#   encoded_sha256：调用者已经读取并保留的同次编码字节摘要。
# 输出：
#   receipt：通过绑定和范围检查的编码回执。
def _validate_visual_encoding_receipt(
    receipt_path: Path,
    *,
    encoded_path: Path,
    source_policy_sha256: str,
    perception_encoder_sha256: str,
    expected_sample_count: int,
    expected_contract: dict[str, object],
    encoded_sha256: str | None = None,
) -> dict[str, object]:
    receipt, _ = read_evidence_object(receipt_path, limit=4 * 1024 * 1024)
    if not all(
        isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in (source_policy_sha256, perception_encoder_sha256)
    ):
        raise ValueError("visual source hashes must be lowercase SHA-256 values")
    if receipt.get("source_policy_data_sha256") != source_policy_sha256:
        raise RuntimeError("LOCAL_EXPERT_VISUAL_SOURCE_DATA_HASH_MISMATCH")
    if receipt.get("output_sha256") != (encoded_sha256 or _sha256(encoded_path)):
        raise RuntimeError("LOCAL_EXPERT_VISUAL_ENCODED_DATA_HASH_MISMATCH")
    if receipt.get("perception_encoder_sha256") != perception_encoder_sha256:
        raise RuntimeError("LOCAL_EXPERT_VISUAL_ENCODER_HASH_MISMATCH")
    if (
        type(expected_sample_count) is not int
        or expected_sample_count <= 0
        or type(receipt.get("sample_count")) is not int
        or receipt.get("sample_count") != expected_sample_count
    ):
        raise RuntimeError("LOCAL_EXPERT_VISUAL_SAMPLE_COUNT_MISMATCH")
    if any(
        type(receipt.get(key)) is not type(value) or receipt.get(key) != value
        for key, value in expected_contract.items()
    ):
        raise RuntimeError("LOCAL_EXPERT_VISUAL_CONTRACT_MISMATCH")
    for field in ("p99_preprocess_latency_ms", "p99_encoder_latency_ms"):
        value = receipt.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError("LOCAL_EXPERT_VISUAL_LATENCY_EVIDENCE_INVALID")
    if receipt.get("qualification_granted") is not False:
        raise RuntimeError("LOCAL_EXPERT_VISUAL_RECEIPT_SCOPE_INVALID")
    return receipt


# 功能：
#   在本次独占目录冻结基座和全部输入，后续训练与回执只消费这些副本，不重新读取外部路径。
# 输入：
#   args：包含普通来源路径和最终输出路径的参数。
#   directory：本次独占的临时目录；总输入最多 1 GiB，不修改原数据。
# 输出：
#   frozen：输入路径替换为固定副本的参数，输出路径保持不变。
#   base：由冻结清单及模型重新校验的基座。
def _freeze_augmentation_sources(args: argparse.Namespace, directory: Path):
    frozen = argparse.Namespace(**vars(args))
    byte_count = 0
    copied: dict[tuple[Path, int], Path] = {}

    # 功能：
    #   有界读取同一来源一次并独占发布副本，重复参数复用同一身份，不扩张总字节预算。
    # 输入：
    #   path：需要冻结的普通文件。
    #   limit：此来源的最大字节数。
    #   target：基座模型需要保留的相对布局目标，其余输入自动命名。
    # 输出：
    #   snapshot：冻结副本的普通文件路径。
    def freeze_file(path: Path, limit: int, target: Path | None = None) -> Path:
        nonlocal byte_count
        check_plain_plugin_path(path)
        path = path.resolve(strict=True)
        key = (path, limit)
        if target is None and key in copied:
            return copied[key]
        content = read_plugin_file(path, limit=limit)
        byte_count += len(content)
        if byte_count > 1024 * 1024 * 1024:
            raise ValueError("augmentation inputs exceed the aggregate byte budget")
        snapshot = target or directory / f"input-{len(copied)}{path.suffix}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        publish_evidence_bytes(snapshot, content, limit=limit)
        copied[key] = snapshot
        return snapshot

    frozen.base_package = directory / "base"
    manifest_path = freeze_file(
        args.base_package / "manifest.json", 4 * 1024 * 1024, frozen.base_package / "manifest.json"
    )
    payload, _ = read_evidence_object(manifest_path, limit=4 * 1024 * 1024)
    manifest = LocalPolicyPackageManifest.model_validate(payload)
    if manifest.pilot_control_mode is not None:
        raise ValueError(
            "candidate-era augmentation does not support continuous control; "
            "use train_causal_control_role.py or train_causal_control_ensemble.py"
        )
    for artifact in manifest.artifacts:
        snapshot = freeze_file(
            args.base_package / artifact.relative_path,
            256 * 1024 * 1024,
            frozen.base_package / artifact.relative_path,
        )
        if _sha256(snapshot) != artifact.sha256:
            raise ValueError("augmentation source artifact changed or has invalid digest")
    base = load_local_policy_package(frozen.base_package)
    for name in (
        "training_data",
        "validation_data",
        "dataset_receipt",
        "training_visual_receipt",
        "validation_visual_receipt",
        "source_augmentation_receipt",
        "source_training_dataset_receipt",
    ):
        path = getattr(args, name)
        if path is not None:
            limit = (256 if name in {"training_data", "validation_data"} else 4) * 1024 * 1024
            setattr(frozen, name, freeze_file(path, limit))
    for name, width in (("risk_dataset", 3), ("risk_visual_dataset", 5)):
        specifications = getattr(args, name) or []
        if len(specifications) > 128:
            raise ValueError("risk calibration exceeds the dataset count budget")
        frozen_specifications = []
        for specification in specifications:
            if len(specification) != width:
                raise ValueError("invalid risk dataset specification")
            frozen_specifications.append(
                [
                    freeze_file(path, (256 if index < 2 else 4) * 1024 * 1024)
                    for index, path in enumerate(specification)
                ]
            )
        setattr(frozen, name, frozen_specifications)
    return frozen, base


# 功能：
#   解析候选时代增补命令、检查输出隔离并固定所有来源，拒绝用此前馈入口重写因果控制器。
# 输入：
#   无：命令行提供基座、独立来源数据、专家角色、训练配置及新输出路径。
# 输出：
#   status：训练、验证、关闭后端及候选和回执发布成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--training-visual-receipt", type=Path)
    parser.add_argument("--validation-visual-receipt", type=Path)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--source-augmentation-receipt", type=Path)
    parser.add_argument("--source-training-dataset-receipt", type=Path)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--expert-role",
        choices=("precision-maneuver-policy", "recovery-policy"),
        required=True,
    )
    parser.add_argument(
        "--bootstrap-from-fallback",
        action="store_true",
        help=(
            "Train a simulation-only specialist candidate from verified general-policy "
            "fallback traces. The resulting package receives no qualification."
        ),
    )
    parser.add_argument("--hidden-feature-count", type=int, default=96)
    parser.add_argument("--epoch-count", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--risk-loss-weight", type=float, default=0.75)
    parser.add_argument(
        "--risk-class-balance-power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to bounded inverse-frequency risk-class weights. "
            "Use zero to preserve the recorded sample weights without class rebalancing."
        ),
    )
    parser.add_argument("--risk-class-weight-cap", type=float, default=16.0)
    parser.add_argument("--random-seed", type=int, default=906)
    parser.add_argument(
        "--risk-dataset",
        type=Path,
        nargs=3,
        action="append",
        metavar=("TRAINING_DATA", "VALIDATION_DATA", "DATASET_RECEIPT"),
        help=(
            "Retrain the independent risk critic from one or more content-bound "
            "training/validation/receipt triples. General and specialist triples "
            "may be combined, but their complete source missions must stay disjoint."
        ),
    )
    parser.add_argument(
        "--risk-visual-dataset",
        type=Path,
        nargs=5,
        action="append",
        metavar=(
            "TRAINING_DATA",
            "VALIDATION_DATA",
            "DATASET_RECEIPT",
            "TRAINING_VISUAL_RECEIPT",
            "VALIDATION_VISUAL_RECEIPT",
        ),
        help=(
            "Retrain a visual independent risk critic from content-bound encoded "
            "data and both encoder receipts. Do not mix with --risk-dataset."
        ),
    )
    parser.add_argument("--risk-epoch-count", type=int, default=240)
    parser.add_argument("--risk-learning-rate", type=float, default=0.001)
    parser.add_argument("--risk-random-seed", type=int, default=1814)
    parser.add_argument("--risk-critic-class-balance-power", type=float, default=1.0)
    parser.add_argument("--risk-critic-class-weight-cap", type=float, default=16.0)
    parser.add_argument(
        "--risk-critic-classification-margin",
        type=float,
        default=0.0,
        help=(
            "Move risk-only training targets away from the 0.5 runtime veto "
            "boundary while retaining the original continuous labels for metrics."
        ),
    )
    parser.add_argument(
        "--validation-used-for-configuration-selection",
        action="store_true",
        help=(
            "Record that the held-out mission influenced specialist hyperparameter "
            "selection. A different mission is then mandatory for admission."
        ),
    )
    args = parser.parse_args()
    for name in ("base_package", "output_package", "training_receipt"):
        check_plain_plugin_path(getattr(args, name))
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.output_package.is_relative_to(args.base_package)
        or args.training_receipt.is_relative_to(args.base_package)
        or args.training_receipt.is_relative_to(args.output_package)
    ):
        raise ValueError("augmentation outputs must stay outside source and candidate packages")
    if args.output_package.exists():
        raise FileExistsError(args.output_package)
    if args.training_receipt.exists():
        raise FileExistsError(args.training_receipt)
    if bool(args.training_visual_receipt) != bool(args.validation_visual_receipt):
        parser.error("visual specialist training requires both split encoding receipts")
    if bool(args.source_augmentation_receipt) != bool(args.source_training_dataset_receipt):
        parser.error(
            "source augmentation and source training dataset receipts must be supplied together"
        )
    if args.risk_dataset and args.risk_visual_dataset:
        parser.error("visual and nonvisual risk calibration datasets cannot be mixed")
    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output_package.parent, prefix=".augmentation-inputs-"
    ) as directory:
        frozen, base = _freeze_augmentation_sources(args, Path(directory))
        status = _augment_candidate(frozen, parser, base)
    return status


# 功能：
#   1. 在优化前核验全部数据、来源链和配置，训练指定候选专家及可选风险头。
#   2. 检查实际后端和未授权角色变化，无覆盖发布候选与回执；两者不是跨文件事务，不授予资格。
# 输入：
#   args：来源路径均已冻结、输出路径已隔离的命令行参数。
#   parser：用于报告相互依赖的参数错误。
#   base：冻结且通过摘要校验的候选时代基座。
# 输出：
#   status：所有阶段及证据发布成功时为零。
def _augment_candidate(
    args: argparse.Namespace, parser: argparse.ArgumentParser, base: LoadedLocalPolicyPackage
) -> int:
    role = args.expert_role
    if role in base.artifact_paths and args.source_augmentation_receipt is None:
        parser.error("retraining an existing specialist requires its augmentation lineage")
    perception_encoder = base.artifact_paths.get("perception-encoder")
    if perception_encoder is None and args.training_visual_receipt is not None:
        parser.error("nonvisual base package cannot use visual encoding receipts")
    if perception_encoder is not None and args.training_visual_receipt is None:
        parser.error("visual base package requires both split encoding receipts")
    training_visual_receipt = (
        read_evidence_object(args.training_visual_receipt, limit=4 * 1024 * 1024)[0]
        if args.training_visual_receipt is not None
        else None
    )
    validation_visual_receipt = (
        read_evidence_object(args.validation_visual_receipt, limit=4 * 1024 * 1024)[0]
        if args.validation_visual_receipt is not None
        else None
    )
    dataset_receipt = _validate_dataset_receipt(
        args.dataset_receipt,
        training_path=args.training_data,
        validation_path=args.validation_data,
        role=role,
        bootstrap_from_fallback=args.bootstrap_from_fallback,
        training_source_sha256=(
            str(training_visual_receipt.get("source_policy_data_sha256"))
            if training_visual_receipt is not None
            else None
        ),
        validation_source_sha256=(
            str(validation_visual_receipt.get("source_policy_data_sha256"))
            if validation_visual_receipt is not None
            else None
        ),
    )
    training_sources, validation_sources = _dataset_source_evidence(dataset_receipt)
    inherited_source_evidence: set[str] = set()
    if args.source_augmentation_receipt is not None:
        inherited_source_evidence = _validate_source_augmentation_lineage(
            receipt_path=args.source_augmentation_receipt,
            dataset_receipt_path=args.source_training_dataset_receipt,
            base_package_sha256=base.package_sha256,
            role=role,
        )
    historical_source_evidence = sorted(
        inherited_source_evidence | training_sources | validation_sources
    )
    risk_training: list[LocalPolicyTrainingSample] = []
    risk_validation: list[LocalPolicyTrainingSample] = []
    risk_training_counts: dict[str, int] | None = None
    risk_validation_counts: dict[str, int] | None = None
    risk_dataset_evidence: list[dict[str, object]] = []
    risk_dataset_specifications = args.risk_visual_dataset or args.risk_dataset or []
    if risk_dataset_specifications:
        risk_visual_contract = (
            {
                "visual_width": base.manifest.visual_width,
                "visual_height": base.manifest.visual_height,
                "visual_feature_count": base.manifest.visual_feature_count,
                "visual_normalization": base.manifest.visual_normalization,
            }
            if args.risk_visual_dataset
            else None
        )
        (
            risk_training,
            risk_validation,
            risk_training_counts,
            risk_validation_counts,
            risk_dataset_evidence,
            risk_source_evidence,
        ) = _load_risk_calibration_datasets(
            risk_dataset_specifications,
            perception_encoder_sha256=(
                _sha256(perception_encoder)
                if args.risk_visual_dataset and perception_encoder is not None
                else None
            ),
            visual_contract=risk_visual_contract,
        )
        historical_source_evidence = sorted(set(historical_source_evidence) | risk_source_evidence)
    if perception_encoder is not None:
        expected_visual_contract = {
            "visual_width": base.manifest.visual_width,
            "visual_height": base.manifest.visual_height,
            "visual_feature_count": base.manifest.visual_feature_count,
            "visual_normalization": base.manifest.visual_normalization,
        }
        encoder_sha256 = _sha256(perception_encoder)
        assert args.training_visual_receipt is not None
        assert args.validation_visual_receipt is not None
        _validate_visual_encoding_receipt(
            args.training_visual_receipt,
            encoded_path=args.training_data,
            source_policy_sha256=str(dataset_receipt["training_output_sha256"]),
            perception_encoder_sha256=encoder_sha256,
            expected_sample_count=dataset_receipt.get("training_sample_count"),
            expected_contract=expected_visual_contract,
        )
        _validate_visual_encoding_receipt(
            args.validation_visual_receipt,
            encoded_path=args.validation_data,
            source_policy_sha256=str(dataset_receipt["validation_output_sha256"]),
            perception_encoder_sha256=encoder_sha256,
            expected_sample_count=dataset_receipt.get("validation_sample_count"),
            expected_contract=expected_visual_contract,
        )
    unsupported_roles = _unsupported_artifact_roles(set(base.artifact_paths))
    if unsupported_roles:
        raise RuntimeError("LOCAL_POLICY_AUGMENTATION_HAS_UNSUPPORTED_ARTIFACTS")
    if "risk-critic" not in base.artifact_paths:
        raise RuntimeError("LOCAL_POLICY_AUGMENTATION_REQUIRES_RISK_CRITIC")

    training, training_counts = _select_and_validate_samples(
        load_training_samples(args.training_data),
        role=role,
        split="training",
    )
    validation, validation_counts = _select_and_validate_samples(
        load_training_samples(args.validation_data),
        role=role,
        split="validation",
    )
    _assert_disjoint(training, validation)
    initial_path = base.artifact_paths.get(role)
    initial_model = load_local_policy_onnx(initial_path) if initial_path is not None else None
    config = LocalPolicyTrainingConfig(
        hidden_feature_count=(
            int(initial_model.input_bias.size)
            if initial_model is not None
            else args.hidden_feature_count
        ),
        epoch_count=args.epoch_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        risk_loss_weight=args.risk_loss_weight,
        risk_class_balance_power=args.risk_class_balance_power,
        risk_class_weight_cap=args.risk_class_weight_cap,
        random_seed=args.random_seed,
    )
    navigation_model = load_local_policy_onnx(base.artifact_paths["local-navigation-policy"])
    risk_model = load_local_risk_critic_onnx(base.artifact_paths["risk-critic"])
    preserved_experts = {
        other_role: load_local_policy_onnx(path)
        for other_role, path in base.artifact_paths.items()
        if other_role in {"precision-maneuver-policy", "recovery-policy"} and other_role != role
    }
    risk_config: LocalPolicyTrainingConfig | None = None
    risk_training_metrics: LocalRiskCriticMetrics | None = None
    risk_validation_metrics: LocalRiskCriticMetrics | None = None
    if risk_dataset_specifications:
        if not 0.0 <= args.risk_critic_classification_margin < 0.5:
            raise ValueError("risk classification margin must be in [0, 0.5)")
        risk_config = LocalPolicyTrainingConfig(
            hidden_feature_count=int(risk_model.input_bias.shape[0]),
            epoch_count=args.risk_epoch_count,
            batch_size=args.batch_size,
            learning_rate=args.risk_learning_rate,
            risk_loss_weight=1.0,
            risk_class_balance_power=args.risk_critic_class_balance_power,
            risk_class_weight_cap=args.risk_critic_class_weight_cap,
            random_seed=args.risk_random_seed,
        )
        if args.risk_visual_dataset:
            risk_model = _prepare_visual_risk_model_for_calibration(
                risk_model,
                visual_feature_count=base.manifest.visual_feature_count,
                random_seed=args.risk_random_seed,
            )
    # 数据与全部启用配置均已核验，避免专家优化完才发现风险训练参数无效。
    expert_model, training_metrics = train_local_policy(
        training, config, initial_model=initial_model
    )
    validation_metrics = evaluate_local_policy(expert_model, validation)
    _validate_metrics(validation_metrics)
    if risk_config is not None:
        risk_model, risk_training_metrics = train_local_risk_critic(
            risk_training,
            risk_config,
            initial_model=risk_model,
            classification_margin=args.risk_critic_classification_margin,
        )
        risk_validation_metrics = evaluate_local_risk_critic(
            risk_model,
            risk_validation,
        )
        _validate_risk_metrics(risk_validation_metrics)
    precision_model = (
        expert_model
        if role == "precision-maneuver-policy"
        else (preserved_experts.get("precision-maneuver-policy"))
    )
    recovery_model = (
        expert_model if role == "recovery-policy" else (preserved_experts.get("recovery-policy"))
    )
    advisor_arguments = {
        "state_anomaly_detector_path": base.artifact_paths.get("state-anomaly-detector"),
        "perception_health_critic_path": base.artifact_paths.get("perception-health-critic"),
        "settle_stability_critic_path": base.artifact_paths.get("settle-stability-critic"),
        "payload_dynamics_adapter_path": base.artifact_paths.get("payload-dynamics-adapter"),
        "cross_modal_consistency_critic_path": base.artifact_paths.get(
            "cross-modal-consistency-critic"
        ),
    }

    args.output_package.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output_package.parent,
        prefix=f".{args.output_package.name}.augmentation-",
    ) as temporary_directory:
        staged_package = Path(temporary_directory) / "package"
        write_local_policy_package(
            control_feature_contract_sha256=base.manifest.control_feature_contract_sha256,
            output_root=staged_package,
            model=navigation_model,
            package_id=args.package_id,
            display_name=args.display_name,
            scope=base.manifest.scope,
            vehicle_sha256=base.manifest.vehicle_sha256,
            sensor_contract_sha256=base.manifest.sensor_contract_sha256,
            maximum_inference_latency_ms=base.manifest.maximum_inference_latency_ms,
            perception_encoder_path=base.artifact_paths.get("perception-encoder"),
            visual_width=base.manifest.visual_width,
            visual_height=base.manifest.visual_height,
            visual_feature_count=base.manifest.visual_feature_count,
            visual_normalization=base.manifest.visual_normalization,
            risk_model=risk_model,
            precision_model=precision_model,
            recovery_model=recovery_model,
            base_package_sha256=base.manifest.base_package_sha256,
            map_sha256=base.manifest.map_sha256,
            **advisor_arguments,
        )
        staged = load_local_policy_package(staged_package)
        backend = OnnxLocalPolicyBackend(staged, execution_providers=["CPUExecutionProvider"])
        backend.close()
        base_hashes = {artifact.role: artifact.sha256 for artifact in base.manifest.artifacts}
        staged_hashes = {artifact.role: artifact.sha256 for artifact in staged.manifest.artifacts}
        changed_roles = {
            existing_role
            for existing_role in base_hashes.keys() | staged_hashes.keys()
            if staged_hashes.get(existing_role) != base_hashes.get(existing_role)
        }
        expected_changed_roles = {role}
        if risk_dataset_specifications:
            expected_changed_roles.add("risk-critic")
        if changed_roles - expected_changed_roles:
            raise RuntimeError("LOCAL_POLICY_AUGMENTATION_CHANGED_BASE_ARTIFACT")
        if role not in changed_roles:
            raise RuntimeError("LOCAL_POLICY_AUGMENTATION_SPECIALIST_UNCHANGED")
        if risk_dataset_specifications and "risk-critic" not in changed_roles:
            raise RuntimeError("LOCAL_POLICY_AUGMENTATION_RISK_CRITIC_UNCHANGED")
        publish_asset_directory(staged_package, args.output_package)

    package = load_local_policy_package(args.output_package)
    receipt = {
        "schema_version": "dronedream.local-policy-augmentation-receipt.v1",
        "package_id": package.manifest.package_id,
        "package_sha256": package.package_sha256,
        "source_package_id": base.manifest.package_id,
        "source_package_sha256": base.package_sha256,
        "expert_role": role,
        "vehicle_sha256": package.manifest.vehicle_sha256,
        "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
        "training_data_sha256": _sha256(args.training_data),
        "validation_data_sha256": _sha256(args.validation_data),
        "dataset_receipt_sha256": _sha256(args.dataset_receipt),
        "source_augmentation_receipt_sha256": (
            _sha256(args.source_augmentation_receipt)
            if args.source_augmentation_receipt is not None
            else None
        ),
        "source_training_dataset_receipt_sha256": (
            _sha256(args.source_training_dataset_receipt)
            if args.source_training_dataset_receipt is not None
            else None
        ),
        "historical_source_evidence_sha256": historical_source_evidence,
        "training_visual_encoding_receipt_sha256": (
            _sha256(args.training_visual_receipt)
            if args.training_visual_receipt is not None
            else None
        ),
        "validation_visual_encoding_receipt_sha256": (
            _sha256(args.validation_visual_receipt)
            if args.validation_visual_receipt is not None
            else None
        ),
        "dataset_split_method": dataset_receipt["split_method"],
        "navigation_label_source": dataset_receipt["navigation_label_source"],
        "bootstrap_only": args.bootstrap_from_fallback,
        "deployment_scope": ("simulation-only" if args.bootstrap_from_fallback else "unqualified"),
        "training_class_counts": training_counts,
        "validation_class_counts": validation_counts,
        "training_config": config.model_dump(mode="json"),
        "training_metrics": training_metrics.model_dump(mode="json"),
        "validation_metrics": validation_metrics.model_dump(mode="json"),
        "risk_dataset_evidence": risk_dataset_evidence,
        "risk_training_class_counts": risk_training_counts,
        "risk_validation_class_counts": risk_validation_counts,
        "risk_training_config": (
            {
                **risk_config.model_dump(mode="json"),
                "classification_margin": args.risk_critic_classification_margin,
            }
            if risk_config is not None
            else None
        ),
        "risk_training_metrics": (
            risk_training_metrics.model_dump(mode="json")
            if risk_training_metrics is not None
            else None
        ),
        "risk_validation_metrics": (
            risk_validation_metrics.model_dump(mode="json")
            if risk_validation_metrics is not None
            else None
        ),
        "validation_used_for_configuration_selection": (
            args.validation_used_for_configuration_selection
        ),
        "fresh_admission_validation_required": True,
        "preserved_artifact_sha256": {
            preserved_role: digest
            for preserved_role, digest in sorted(base_hashes.items())
            if preserved_role
            not in ({role, "risk-critic"} if risk_dataset_specifications else {role})
        },
        "qualification_granted": False,
        "qualification_requirement": (
            "A bootstrap candidate must first produce specialist-selected traces; all "
            "augmented packages must then pass fresh combined offline admission and a "
            "separate multi-map Gazebo/PX4 qualification campaign."
            if args.bootstrap_from_fallback
            else "The augmented package must pass fresh combined offline admission and "
            "a separate multi-map Gazebo/PX4 qualification campaign."
        ),
    }
    args.training_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(args.training_receipt, receipt)
    print(
        json.dumps(
            {
                "package": str(args.output_package),
                "package_sha256": package.package_sha256,
                "expert_role": role,
                "training_class_counts": training_counts,
                "validation_class_counts": validation_counts,
                "validation_motion_authorization_accuracy": (
                    validation_metrics.motion_authorization_accuracy
                ),
                "validation_candidate_selection_accuracy": (
                    validation_metrics.candidate_selection_accuracy
                ),
                "qualification_granted": False,
            },
            sort_keys=True,
        )
    )
    status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
