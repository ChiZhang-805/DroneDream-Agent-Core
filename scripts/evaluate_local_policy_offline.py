#!/usr/bin/env python3
"""Benchmark one immutable local policy and issue simulation-only admission."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingMetrics,
    LocalAdvisorTrainingSample,
)
from dronedream_agent_core.local_expert_harness import (
    NAVIGATION_EXPERT_ROLES,
    LocalExpertRoutingDecision,
)
from dronedream_agent_core.local_policy_packages import (
    LOCAL_POLICY_MAXIMUM_CANDIDATES,
    LocalPolicyAdvisorAdmissionMetrics,
    LocalPolicySimulationAdmissionReceipt,
    PolicyArtifactRole,
    latency_percentile,
    load_local_policy_package,
)
from dronedream_agent_core.local_policy_port import (
    LocalPolicyFeatureBatch,
    LocalPolicyRawInference,
    OnnxLocalPolicyBackend,
)
from dronedream_agent_core.local_policy_quality import (
    navigation_quality_issues,
    summarize_pilot_axes,
)
from dronedream_agent_core.local_policy_training import (
    LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX,
    LocalPolicyTrainingMetrics,
    LocalPolicyTrainingSample,
    load_training_samples,
    require_behavior_supervision,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.action_risk_evaluation import evaluate_action_risk_artifact
from dronedream_agent_core.training.admission_inputs import (
    bind_admission_history,
    freeze_admission_inputs,
    navigation_dataset_contract,
    read_admission_observations,
)
from dronedream_agent_core.training.advisor_sources import validate_advisor_spatial_splits
from dronedream_agent_core.training.artifact_assembly import validate_embedded_graph
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.heading_admission import read_heading_admission_inputs
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from dronedream_agent_core.training.risk_admission_evidence import (
    ActionRiskAdmissionEvidence,
    action_risk_evidence_from_reports,
    action_risk_evidence_issues,
    action_risk_summary,
)
from dronedream_agent_core.training.visual_lineage import VisualInputContract
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   通过普通文件边界流式核对输入身份，不无界读取，也不允许来源链接绕过检查。
# 输入：
#   path：本次评估的模型、数据、回执或代码路径。
# 输出：
#   digest：最多 256 MiB 实际内容的 SHA-256。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=256 * 1024 * 1024)
    return digest


# 功能：
#   一次严格读取有界回执对象，重复键、非有限值和错误容器不能进入准入判断。
# 输入：
#   path：最多 4 MiB 的普通 JSON 回执文件。
# 输出：
#   receipt：无歧义的回执对象。
#   digest：同次消费字节的 SHA-256。
def _read_receipt(path: Path) -> tuple[dict, str]:
    content = read_plugin_file(path, limit=4 * 1024 * 1024)
    receipt = decode_json(content, limit=4 * 1024 * 1024, node_limit=1_000_000)
    if not isinstance(receipt, dict):
        raise ValueError("admission receipt must be an object")
    digest = hashlib.sha256(content).hexdigest()
    return receipt, digest


# 功能：
#   区分真实小写十六进制内容身份与仅长度相同的任务标签。
# 输入：
#   value：待检查的来源值。
# 输出：
#   valid：是否满足 SHA-256 格式。
def _valid_hash(value) -> bool:
    valid = type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")
    return valid


# 功能：
#   提取有界、严格的任务来源摘要集合，供训练、调参与新准入集合做交叉检查。
# 输入：
#   receipt：已严格解析的数据集回执。
#   split：training 或 validation 分区。
# 输出：
#   sources：排序去重后的来源摘要。
def _source_evidence_hashes(receipt: dict[str, object], split: str) -> list[str]:
    if split not in {"training", "validation"} or not isinstance(receipt, dict):
        raise ValueError("admission source split is invalid")
    raw_sources = receipt.get(f"{split}_sources")
    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= 10_000:
        raise ValueError(f"advisor admission receipt has no {split} sources")
    hashes: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("advisor admission source record is invalid")
        value = source.get("mission_evidence_sha256")
        if not _valid_hash(value):
            raise ValueError("advisor admission source evidence hash is invalid")
        hashes.add(value)
    sources = sorted(hashes)
    return sources


# 功能：
#   验证顾问准入数据由新任务组成，不能重复使用训练或调参任务及数据摘要。
# 输入：
#   data_path：本次冻结的顾问数据。
#   dataset_receipt_path：绑定数据划分与任务来源的回执。
#   training_receipt：原训练回执及历史任务摘要。
# 输出：
#   split：当前数据对应的分区名。
#   admission_sources：与历史任务不重叠的新来源摘要。
def _validate_fresh_advisor_admission_data(
    *,
    data_path: Path,
    dataset_receipt_path: Path,
    training_receipt: dict[str, object],
) -> tuple[str, list[str]]:
    if training_receipt.get("schema_version") != "dronedream.local-advisor-training-receipt.v1":
        raise ValueError("advisor training receipt schema is invalid")
    if training_receipt.get("fresh_admission_validation_required") is not True:
        raise ValueError("advisor training receipt does not require fresh admission data")
    receipt, _ = _read_receipt(dataset_receipt_path)
    if receipt.get("schema_version") != "dronedream.local-advisor-dataset-receipt.v1":
        raise ValueError("advisor admission dataset receipt schema is invalid")
    current_spatial = receipt.get("split_method") == SPATIAL_SPLIT_CONTRACT
    if not current_spatial and receipt.get("split_method") != "held-out-complete-mission":
        raise ValueError("advisor admission data is not split by complete mission")
    if not current_spatial and training_receipt.get("feature_contract_sha256") is not None:
        raise ValueError("current advisor training cannot use legacy admission partitions")
    if receipt.get("verified_sources_required") is not True:
        raise ValueError("advisor admission data does not require verified sources")
    data_sha256 = _sha256(data_path)
    if data_sha256 in {
        training_receipt.get("training_data_sha256"),
        training_receipt.get("validation_data_sha256"),
    }:
        raise ValueError("advisor admission data reuses training or tuning rows")
    if receipt.get("validation_output_sha256") == data_sha256:
        split = "validation"
    elif receipt.get("training_output_sha256") == data_sha256:
        split = "training"
    else:
        raise ValueError("advisor admission data differs from its dataset receipt")
    admission_sources = _source_evidence_hashes(receipt, split)
    historical_sources: set[str] = set()
    for key in (
        "training_source_evidence_sha256",
        "validation_source_evidence_sha256",
    ):
        values = training_receipt.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError("advisor training receipt source evidence is incomplete")
        if len(values) > 10_000 or not all(_valid_hash(value) for value in values):
            raise ValueError("advisor training receipt source evidence is invalid")
        historical_sources.update(values)
    if historical_sources & set(admission_sources):
        raise ValueError("advisor admission reuses a training or tuning source mission")
    if current_spatial:
        if receipt.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256 or (
            training_receipt.get("feature_contract_sha256")
            != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        ):
            raise ValueError("advisor admission feature contract differs from current training")
        groups = validate_advisor_spatial_splits(
            receipt.get("training_sources"), receipt.get("validation_sources")
        )
        for name, actual in zip(("training", "validation"), groups, strict=True):
            if receipt.get(name + "_groups") != sorted(actual):
                raise ValueError("advisor admission source spatial groups differ")
        historical_groups = set()
        for name in ("training_groups", "validation_groups"):
            values = training_receipt.get(name)
            if (
                not isinstance(values, list)
                or not values
                or not all(_valid_hash(v) for v in values)
            ):
                raise ValueError("advisor training spatial lineage is missing")
            historical_groups.update(values)
        selected_groups = groups[0 if split == "training" else 1]
        if historical_groups & selected_groups:
            raise ValueError("advisor admission reuses a training or tuning spatial route")
    return split, admission_sources


# 功能：
#   只继承字节及输入语义均保留的顾问证据，排除已替换角色；继承不授予新包飞行资格。
# 输入：
#   package_path：本次冻结并实际评估的模型包。
#   preservation_receipt_path：组合、增补、训练或重绑定产生的保留记录。
#   inherited_admission_receipt_path：绑定原包的已接受准入回执。
#   trainable_advisor_roles：允许继承证据的顾问角色集合。
# 输出：
#   metrics：允许保留的角色指标。
#   evidence：同次读取的来源回执摘要及指标记录。
def _load_inherited_advisor_evidence(
    *,
    package_path: Path,
    preservation_receipt_path: Path,
    inherited_admission_receipt_path: Path,
    trainable_advisor_roles: set[PolicyArtifactRole],
) -> tuple[
    dict[PolicyArtifactRole, LocalPolicyAdvisorAdmissionMetrics],
    list[dict[str, object]],
]:
    package = load_local_policy_package(package_path)
    preservation, preservation_digest = _read_receipt(preservation_receipt_path)
    schema_version = preservation.get("schema_version")
    if schema_version not in {
        "dronedream.local-policy-composition-receipt.v1",
        "dronedream.local-policy-augmentation-receipt.v1",
        "dronedream.local-policy-rebinding-receipt.v1",
        "dronedream.local-policy-training-receipt.v1",
    }:
        raise ValueError("local policy preservation receipt schema is invalid")
    if preservation.get("package_sha256") != package.package_sha256:
        raise ValueError("preservation receipt is not bound to the evaluated package")
    if preservation.get("qualification_granted") is not False:
        raise ValueError("preservation receipt cannot grant qualification")
    preserved = preservation.get("preserved_artifact_sha256")
    if not isinstance(preserved, dict):
        raise ValueError("preservation receipt has no preserved artifact hashes")
    package_hashes = {artifact.role: artifact.sha256 for artifact in package.manifest.artifacts}
    for raw_role, digest in preserved.items():
        role = str(raw_role)
        if not isinstance(digest, str) or package_hashes.get(role) != digest:
            raise ValueError("preserved artifact evidence is invalid")

    replaced_advisors: set[PolicyArtifactRole] = set()
    if schema_version == "dronedream.local-policy-composition-receipt.v1":
        replacements = preservation.get("advisor_evidence")
        if not isinstance(replacements, list) or not replacements:
            raise ValueError("composition receipt has no replacement advisor evidence")
        for value in replacements:
            if not isinstance(value, dict):
                raise ValueError("composition replacement advisor evidence is invalid")
            raw_role = value.get("role")
            if raw_role not in trainable_advisor_roles:
                raise ValueError(
                    f"composition has unsupported replacement advisor role: {raw_role}"
                )
            role: PolicyArtifactRole = raw_role
            if role in replaced_advisors:
                raise ValueError(f"composition repeats replacement advisor: {role}")
            artifact_sha256 = value.get("artifact_sha256")
            if (
                not isinstance(artifact_sha256, str)
                or package_hashes.get(role) != artifact_sha256
                or role in preserved
            ):
                raise ValueError(f"composition replacement advisor evidence is invalid: {role}")
            replaced_advisors.add(role)

    inherited_payload, inherited_digest = _read_receipt(inherited_admission_receipt_path)
    inherited = LocalPolicySimulationAdmissionReceipt.model_validate(inherited_payload)
    if package.manifest.pilot_control_mode is not None and (
        inherited.control_feature_contract_sha256 is None
        or inherited.control_feature_contract_sha256
        != package.manifest.control_feature_contract_sha256
    ):
        raise ValueError("preserved feature semantics differ; fresh advisor evidence required")
    if (
        not inherited.admitted_to_simulation
        or inherited.issue_codes
        or inherited.admission_scope != "standard-simulation"
    ):
        raise ValueError("inherited advisor admission was not accepted")
    if preservation.get("source_package_sha256") != inherited.policy_package_sha256:
        raise ValueError("inherited admission is not bound to the preserved source")
    vehicle_changed = inherited.vehicle_sha256 != package.manifest.vehicle_sha256
    if vehicle_changed and schema_version != "dronedream.local-policy-rebinding-receipt.v1":
        raise ValueError("changed vehicle requires fresh advisor evidence, not inherited metrics")
    if schema_version == "dronedream.local-policy-rebinding-receipt.v1":
        rebinding_valid = (
            preservation.get("source_vehicle_sha256") == inherited.vehicle_sha256
            and preservation.get("source_simulation_admission_sha256") == inherited_digest
            and preservation.get("vehicle_sha256") == package.manifest.vehicle_sha256
            and preservation.get("sensor_contract_sha256")
            == package.manifest.sensor_contract_sha256
            and preservation.get("map_sha256") == package.manifest.map_sha256
            and preservation.get("runtime_input_contract_unchanged") is True
            and preservation.get("fresh_simulation_admission_required") is True
            and preservation.get("current_asset_closed_loop_required") is True
            and preservation.get("qualification_granted") is False
        )
        if not rebinding_valid:
            raise ValueError("local policy rebinding receipt is invalid")
    elif schema_version == "dronedream.local-policy-training-receipt.v1":
        training_preservation_valid = (
            preservation.get("source_vehicle_sha256") == inherited.vehicle_sha256
            and preservation.get("vehicle_sha256") == package.manifest.vehicle_sha256
            and preservation.get("sensor_contract_sha256")
            == inherited.sensor_contract_sha256
            == package.manifest.sensor_contract_sha256
            and preservation.get("map_sha256") == package.manifest.map_sha256
            and preservation.get("preserved_advisor_input_contract_unchanged") is True
            and preservation.get("fresh_simulation_admission_required") is True
            and preservation.get("qualification_granted") is False
        )
        if not training_preservation_valid:
            raise ValueError("local policy training preservation receipt is invalid")
    elif (
        inherited.vehicle_sha256 != package.manifest.vehicle_sha256
        or inherited.sensor_contract_sha256 != package.manifest.sensor_contract_sha256
        or inherited.map_sha256 != package.manifest.map_sha256
    ):
        raise ValueError("inherited admission uses a different package contract")
    if inherited.sensor_contract_sha256 != package.manifest.sensor_contract_sha256 or (
        schema_version != "dronedream.local-policy-training-receipt.v1"
        and inherited.map_sha256 != package.manifest.map_sha256
    ):
        raise ValueError("inherited admission changes the model input contract")

    metrics: dict[PolicyArtifactRole, LocalPolicyAdvisorAdmissionMetrics] = {}
    evidence: list[dict[str, object]] = []
    if vehicle_changed:
        # 原回执仅证明重绑定来源。新载具必须继续提供顾问新评估，不能继承旧机型指标。
        return metrics, evidence
    for role, value in inherited.advisor_metrics.items():
        if role not in trainable_advisor_roles:
            raise ValueError(f"inherited admission has unsupported advisor role: {role}")
        artifact_sha256 = preserved.get(role)
        if not isinstance(artifact_sha256, str):
            if role in replaced_advisors:
                continue
            raise ValueError(f"inherited advisor was not preserved: {role}")
        metrics[role] = value
        evidence.append(
            {
                "role": role,
                "evidence_kind": "byte-identical-inherited-admission",
                "artifact_sha256": artifact_sha256,
                "preservation_receipt_schema": schema_version,
                "preservation_receipt_sha256": preservation_digest,
                "inherited_admission_receipt_sha256": inherited_digest,
                "inherited_policy_package_sha256": inherited.policy_package_sha256,
                "inherited_advisor_evaluation_sha256": (inherited.advisor_evaluation_sha256),
                "metrics": value.model_dump(mode="json"),
            }
        )
    return metrics, evidence


# 功能：
#   核对增补导航专家的训练来源链和本次新任务，禁止重用训练或调参任务作为准入。
# 输入：
#   source_data_sha256：实际消费的原始验证数据摘要。
#   admission_dataset_receipt_path：本次新任务数据回执。
#   augmentation_receipt_path：专家增补回执。
#   training_dataset_receipt_path：增补使用的训练划分回执。
# 输出：
#   admission_split：匹配的数据分区。
#   admission_sources：不与历史来源重叠的任务摘要。
def _validate_fresh_navigation_admission_data(
    *,
    source_data_sha256: str,
    admission_dataset_receipt_path: Path,
    augmentation_receipt_path: Path,
    training_dataset_receipt_path: Path,
) -> tuple[str, list[str]]:
    if not _valid_hash(source_data_sha256):
        raise ValueError("navigation admission data hash is invalid")
    augmentation, _ = _read_receipt(augmentation_receipt_path)
    training_dataset, training_digest = _read_receipt(training_dataset_receipt_path)
    admission_dataset, admission_digest = _read_receipt(admission_dataset_receipt_path)
    if augmentation.get("schema_version") != "dronedream.local-policy-augmentation-receipt.v1":
        raise ValueError("navigation training receipt schema is invalid")
    if augmentation.get("fresh_admission_validation_required") is not True:
        raise ValueError("navigation training receipt does not require fresh admission")
    if augmentation.get("qualification_granted") is not False:
        raise ValueError("navigation training receipt cannot grant qualification")
    if augmentation.get("dataset_receipt_sha256") != training_digest:
        raise ValueError("navigation training dataset receipt hash is invalid")

    for label, receipt in (
        ("training", training_dataset),
        ("admission", admission_dataset),
    ):
        if receipt.get("schema_version") != "dronedream.local-policy-dataset-receipt.v1":
            raise ValueError(f"navigation {label} dataset receipt schema is invalid")
        if receipt.get("split_method") != "held-out-source-campaign":
            raise ValueError(f"navigation {label} data is not mission-held-out")
        if receipt.get("verified_sources_required") is not True:
            raise ValueError(f"navigation {label} data permits unverified sources")

    if admission_digest == training_digest:
        raise ValueError("navigation admission reuses the training dataset receipt")
    if admission_dataset.get("validation_output_sha256") == source_data_sha256:
        admission_split = "validation"
    elif admission_dataset.get("training_output_sha256") == source_data_sha256:
        admission_split = "training"
    else:
        raise ValueError("navigation admission data differs from its dataset receipt")

    admission_sources = _source_evidence_hashes(admission_dataset, admission_split)
    current_training_sources = {
        *_source_evidence_hashes(training_dataset, "training"),
        *_source_evidence_hashes(training_dataset, "validation"),
    }
    historical_lineage = augmentation.get("historical_source_evidence_sha256")
    if historical_lineage is None:
        historical_sources = current_training_sources
    else:
        if (
            not isinstance(historical_lineage, list)
            or not historical_lineage
            or len(historical_lineage) > 10_000
            or not all(_valid_hash(value) for value in historical_lineage)
        ):
            raise ValueError("navigation training source lineage is invalid")
        historical_sources = set(historical_lineage)
        if not current_training_sources.issubset(historical_sources):
            raise ValueError("navigation training source lineage is incomplete")
    if historical_sources & set(admission_sources):
        raise ValueError("navigation admission reuses a training or tuning source mission")
    return admission_split, admission_sources


# 功能：
#   区分普通准入与恢复冷启动采集；后者仍需强留出指标且不能用于最终闭环资格。
# 输入：
#   augmentation_receipt_path：可选专家增补回执，未提供时采用普通仿真支持数量。
# 输出：
#   scope：普通或恢复引导仿真范围。
#   motion_minimum：最低运动样本数。
#   non_motion_minimum：最低非运动样本数。
def _navigation_admission_scope(
    augmentation_receipt_path: Path | None,
) -> tuple[str, int, int]:
    if augmentation_receipt_path is None:
        scope, motion_minimum, non_motion_minimum = "standard-simulation", 20, 20
        return scope, motion_minimum, non_motion_minimum
    receipt, _ = _read_receipt(augmentation_receipt_path)
    if "bootstrap_only" in receipt and type(receipt["bootstrap_only"]) is not bool:
        raise ValueError("recovery bootstrap flag is invalid")
    if receipt.get("bootstrap_only") is not True:
        scope, motion_minimum, non_motion_minimum = "standard-simulation", 20, 20
        return scope, motion_minimum, non_motion_minimum
    if (
        receipt.get("schema_version") != "dronedream.local-policy-augmentation-receipt.v1"
        or receipt.get("expert_role") != "recovery-policy"
        or receipt.get("deployment_scope") != "simulation-only"
        or receipt.get("fresh_admission_validation_required") is not True
        or receipt.get("qualification_granted") is not False
    ):
        raise ValueError("recovery bootstrap augmentation receipt is invalid")
    validation_counts = receipt.get("validation_class_counts")
    validation_metrics = receipt.get("validation_metrics")
    if not isinstance(validation_counts, dict) or not isinstance(validation_metrics, dict):
        raise ValueError("recovery bootstrap validation evidence is missing")
    metric_names = (
        "motion_authorization_accuracy",
        "candidate_selection_accuracy",
        "risk_hold_recall",
        "safe_motion_recall",
        "risk_mean_absolute_error",
    )
    if any(
        type(validation_counts.get(key)) is not int or validation_counts[key] < 0
        for key in ("safe", "risky")
    ) or any(
        type(validation_metrics.get(key)) not in (int, float)
        or not 0 <= validation_metrics[key] <= 1
        for key in metric_names
    ):
        raise ValueError("recovery bootstrap validation values are invalid")
    if (
        int(validation_counts.get("safe", 0)) < 20
        or int(validation_counts.get("risky", 0)) < 10
        or float(validation_metrics.get("motion_authorization_accuracy", 0.0)) < 0.95
        or float(validation_metrics.get("candidate_selection_accuracy", 0.0)) < 0.9
        or float(validation_metrics.get("risk_hold_recall", 0.0)) < 0.95
        or float(validation_metrics.get("safe_motion_recall", 0.0)) < 0.95
        or float(validation_metrics.get("risk_mean_absolute_error", 1.0)) > 0.2
    ):
        raise ValueError("recovery bootstrap held-out validation is insufficient")
    scope, motion_minimum, non_motion_minimum = "recovery-bootstrap-simulation", 20, 3
    return scope, motion_minimum, non_motion_minimum


# 功能：
#   加载实际包和已绑定样本，按因果时间推进全部专家历史，评估后关闭后端并保留原异常。
# 输入：
#   package_path：本次冻结模型包路径。
#   dataset_path：完整验证数据路径。
#   navigation_role：可选目标角色；其他角色样本仍参与历史准备而不计入该角色指标。
#   observations：可选、与原始分区绑定的完整观测列表。
#   heading_inputs：可选、已从同一原始快照核验的偏航输入索引。
# 输出：
#   metrics：实际计分样本的导航质量指标。
#   latencies：包含特征和时序准备的有效毫秒观测。
def _evaluate_runtime_package(
    package_path: Path,
    dataset_path: Path,
    *,
    navigation_role: str | None = None,
    observations=None,
    heading_inputs=None,
    action_risk_evidence=None,
) -> tuple[LocalPolicyTrainingMetrics, list[float]]:

    package = load_local_policy_package(package_path)
    samples = load_training_samples(dataset_path)
    if navigation_role is not None:
        if navigation_role not in NAVIGATION_EXPERT_ROLES:
            raise ValueError("unsupported navigation expert evaluation role")
        if not any(sample.navigation_expert_role == navigation_role for sample in samples):
            raise ValueError(f"navigation expert has no held-out samples: {navigation_role}")
    if package.manifest.navigation_architecture == "causal-gru-control":
        if any(s.temporal_evidence is None for s in samples):
            raise ValueError("CAUSAL_ADMISSION_SOURCE_PROVENANCE_REQUIRED")
        identities = [
            (s.temporal_evidence.stream_id, s.temporal_evidence.sample_sha256) for s in samples
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("CAUSAL_ADMISSION_DUPLICATE_OBSERVATION")
        samples.sort(
            key=lambda s: (s.temporal_evidence.stream_id, s.temporal_evidence.observed_at_unix_ms)
        )
    backend = OnnxLocalPolicyBackend(
        package,
        execution_providers=["CPUExecutionProvider"],
        allow_precomputed_visual_features=True,
    )
    try:
        metrics, latencies = _evaluate_runtime_samples(
            package, samples, backend, navigation_role=navigation_role, observations=observations,
            heading_inputs=heading_inputs,
            action_risk_evidence=action_risk_evidence,
        )
    except BaseException as error:
        try:
            backend.close()
        except BaseException as cleanup_error:
            error.add_note(f"local policy backend cleanup also failed: {cleanup_error}")
        raise
    else:
        backend.close()
    return metrics, latencies


# 功能：
#   逐条重验行为样本和后端数值，按部署控制模式计算动作、风险及四轴误差，不为风险提案造标签。
# 输入：
#   package：本次已验证的模型包。
#   samples：包含完整时间顺序的验证样本。
#   backend：实际本地专家后端；测试可显式替换。
#   navigation_role：可选限定计分角色。
#   observations：可选完整历史，只有匹配的标签行才参与指标计算。
#   heading_inputs：只为显式声明偏航扩展的当前控制分支提供同源额外输入。
#   action_risk_evidence：同包风险图的独立动作标签评估；不能用行为零占位替代。
# 输出：
#   metrics：只针对实际评估行的质量和类别支持数量。
#   latencies：包含本地特征构造及历史准备的有效调用耗时。
def _evaluate_runtime_samples(
    package, samples, backend, *, navigation_role=None, observations=None, heading_inputs=None,
    action_risk_evidence=None,
):
    import numpy as np

    if not isinstance(samples, list) or not 1 <= len(samples) <= 250_000:
        raise ValueError("local policy validation sample count is invalid")
    samples = [LocalPolicyTrainingSample.model_validate(sample.model_dump()) for sample in samples]
    require_behavior_supervision(samples)
    if package.manifest.requires_heading_evidence() != (
            heading_inputs is not None):
        raise ValueError('HEADING_ADMISSION_EXPLICIT_SOURCE_REQUIRED')
    causal = package.manifest.navigation_architecture == "causal-gru-control"
    if causal and action_risk_evidence is None:
        raise ValueError('CAUSAL_ADMISSION_ACTION_RISK_EVIDENCE_REQUIRED')
    if action_risk_evidence is not None:
        action_risk_evidence = ActionRiskAdmissionEvidence.model_validate(
            action_risk_evidence.model_dump(), strict=True)
        expected_risk = next((a.sha256 for a in package.manifest.artifacts
                              if a.role == 'risk-critic'), None)
        if expected_risk != action_risk_evidence.model_sha256:
            raise ValueError('ADMISSION_ACTION_RISK_MODEL_MISMATCH')
    if not causal and observations is not None:
        raise ValueError("noncausal runtime evaluation cannot consume causal histories")
    replay = bind_admission_history(samples, observations) if causal else [(s, s) for s in samples]
    latencies: list[float] = []
    raw_predictions: list[int] = []
    effective_predictions: list[int] = []
    risk_holds: list[bool] = []
    risk_scores: list[float] = []
    pilot_controls: list[list[float]] = []
    cross_entropies: list[float] = []
    evaluated_samples = []
    for index, (sample, label) in enumerate(replay):
        started = time.perf_counter()
        if (
            package.manifest.pilot_control_mode is not None
            and sample.navigation_expert_role not in package.artifact_paths
        ):
            raise ValueError("CONTINUOUS_ADMISSION_REQUIRED_EXPERT_MISSING")
        batch = LocalPolicyFeatureBatch(
            temporal_evidence=sample.temporal_evidence,
            pilot_control_limits=sample.pilot_control_limits,
            control_feature_contract_sha256=sample.control_feature_contract_sha256,
            state_features=tuple(sample.state_features),
            candidate_features=tuple(tuple(candidate) for candidate in sample.candidate_features),
            candidate_mask=tuple(sample.candidate_mask),
            candidate_ids=tuple(
                f"candidate-{candidate_index}"
                for candidate_index, mask in enumerate(sample.candidate_mask)
                if mask == 1.0
            ),
            realtime_features=tuple(sample.realtime_features),
            realtime_valid_mask=tuple(sample.realtime_valid_mask),
            realtime_features_ready=bool(sample.realtime_features),
            realtime_snapshot_sha256=(
                sha256_json(
                    {
                        "source_snapshot_sha256": sample.source_snapshot_sha256,
                        "realtime_features": sample.realtime_features,
                        "realtime_valid_mask": sample.realtime_valid_mask,
                    }
                )
                if sample.realtime_features
                else None
            ),
            precomputed_visual_features=(
                tuple(sample.visual_features) if sample.visual_features else None
            ),
            navigation_expert_role=(
                sample.navigation_expert_role
                if sample.navigation_expert_role in package.artifact_paths
                else "local-navigation-policy"
            ),
            expert_routing=LocalExpertRoutingDecision(
                requested_navigation_role=sample.navigation_expert_role,
                selected_navigation_role=(
                    sample.navigation_expert_role
                    if sample.navigation_expert_role in package.artifact_paths
                    else "local-navigation-policy"
                ),
                advisory_roles=(["risk-critic"] if "risk-critic" in package.artifact_paths else []),
                runtime_phase="OFFLINE_NAVIGATION_EVALUATION",
                decision_trigger="held-out-evaluation",
                fallback_used=sample.navigation_expert_role not in package.artifact_paths,
                reason_codes=["OFFLINE_NAVIGATION_ADMISSION_EVALUATION"],
            ),
        )
        # Advance all real observations, including those controlled by another
        # specialist. Role filtering must not erase the history seen onboard.
        backend.prepare_temporal_context(batch)
        if not backend.motion_history_ready():
            continue
        if label is None:
            continue
        if navigation_role is not None and sample.navigation_expert_role != navigation_role:
            continue
        if (heading_inputs is not None
                and package.manifest.heading_context_for_role(batch.navigation_expert_role) is not None):
            batch = replace(batch, precision_heading_context=heading_inputs.features_for(sample))
        output = backend.infer(batch, multimodal=[])
        output = LocalPolicyRawInference.model_validate(output.model_dump())
        evaluated_samples.append(sample)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if index >= min(20, len(replay) // 4):
            latencies.append(elapsed_ms)
        if not (
            all(math.isfinite(value) for value in output.candidate_scores)
            and all(math.isfinite(value) for value in output.action_scores)
            and math.isfinite(output.risk_score)
        ):
            raise RuntimeError("LOCAL_POLICY_ONNX_OUTPUT_NOT_FINITE")
        continuous_control = package.manifest.pilot_control_mode is not None
        if continuous_control:
            if len(output.action_scores) != 4:
                raise RuntimeError("LOCAL_POLICY_PILOT_ACTION_OUTPUT_MISSING")
            if sample.target_action_index < LOCAL_POLICY_MAXIMUM_CANDIDATES:
                raise RuntimeError("LOCAL_POLICY_LEGACY_TARGET_IN_CONTINUOUS_DATASET")
            scores = np.asarray(output.action_scores, dtype=np.float64)
            scored_target = sample.target_action_index - LOCAL_POLICY_MAXIMUM_CANDIDATES
            prediction_offset = LOCAL_POLICY_MAXIMUM_CANDIDATES
        else:
            scores = np.asarray([*output.candidate_scores, *output.action_scores], dtype=np.float64)
            scores[:LOCAL_POLICY_MAXIMUM_CANDIDATES] = np.where(
                np.asarray(sample.candidate_mask) == 1.0,
                scores[:LOCAL_POLICY_MAXIMUM_CANDIDATES],
                -np.inf,
            )
            scored_target = sample.target_action_index
            prediction_offset = 0
        shifted = scores - scores.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        cross_entropies.append(-math.log(max(1e-12, float(probabilities[scored_target]))))
        raw_prediction = int(scores.argmax()) + prediction_offset
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2])
        risk_hold = output.risk_score >= package.manifest.risk_hold_threshold
        effective_prediction = raw_prediction
        if risk_hold or margin < package.manifest.minimum_selection_margin:
            effective_prediction = LOCAL_POLICY_MAXIMUM_CANDIDATES
        raw_predictions.append(raw_prediction)
        effective_predictions.append(effective_prediction)
        risk_holds.append(risk_hold)
        risk_scores.append(output.risk_score)
        if output.pilot_control is not None:
            pilot_controls.append(
                [
                    output.pilot_control.forward_axis,
                    output.pilot_control.right_axis,
                    output.pilot_control.up_axis,
                    output.pilot_control.yaw_axis,
                ]
            )
    if not latencies:
        raise RuntimeError("LOCAL_POLICY_LATENCY_SAMPLE_EMPTY")
    samples = evaluated_samples
    targets = np.asarray([sample.target_action_index for sample in samples], dtype=np.int64)
    risks = np.asarray([sample.risk_target for sample in samples], dtype=np.float64)
    raw = np.asarray(raw_predictions, dtype=np.int64)
    effective = np.asarray(effective_predictions, dtype=np.int64)
    holds = np.asarray(risk_holds, dtype=np.bool_)
    runtime_risk = np.asarray(risk_scores, dtype=np.float64)
    target_motion = (targets < LOCAL_POLICY_MAXIMUM_CANDIDATES) | (
        targets == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    )
    predicted_motion = (effective < LOCAL_POLICY_MAXIMUM_CANDIDATES) | (
        effective == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    )
    candidate_targets = targets < LOCAL_POLICY_MAXIMUM_CANDIDATES
    candidate_count = int(candidate_targets.sum())
    target_risky = risks >= 0.5
    risky_count = int(target_risky.sum())
    safe_count = len(samples) - risky_count
    pilot_control_mean_absolute_error = None
    if pilot_controls:
        if len(pilot_controls) != len(samples) or any(
            not sample.target_pilot_control for sample in samples
        ):
            raise ValueError("continuous-control admission requires a target for every sample")
        target_pilot_control = np.asarray(
            [sample.target_pilot_control for sample in samples], dtype=np.float64
        )
        pilot_control_mean_absolute_error = float(
            np.abs(np.asarray(pilot_controls, dtype=np.float64) - target_pilot_control).mean()
        )
    metrics = LocalPolicyTrainingMetrics(
        sample_count=len(samples),
        motion_sample_count=int(target_motion.sum()),
        non_motion_sample_count=int((~target_motion).sum()),
        authorized_motion_recall=(
            float(predicted_motion[target_motion].mean()) if target_motion.any() else 0.0
        ),
        non_motion_recall=(
            float((~predicted_motion[~target_motion]).mean()) if (~target_motion).any() else 0.0
        ),
        risky_sample_count=risky_count,
        safe_sample_count=safe_count,
        candidate_sample_count=candidate_count,
        action_accuracy=float((raw == targets).mean()),
        motion_authorization_accuracy=float((predicted_motion == target_motion).mean()),
        candidate_selection_accuracy=(
            float((effective[candidate_targets] == targets[candidate_targets]).mean())
            if candidate_count
            else 1.0
        ),
        risk_hold_recall=(float(holds[target_risky].mean()) if risky_count else 0.0),
        safe_motion_recall=(float((~holds[~target_risky]).mean()) if safe_count else 0.0),
        risk_mean_absolute_error=float(np.abs(runtime_risk - risks).mean()),
        mean_cross_entropy=float(np.mean(cross_entropies)),
        pilot_control_mean_absolute_error=pilot_control_mean_absolute_error,
        pilot_axis_evidence=(
            summarize_pilot_axes(
                [
                    p
                    for p, s in zip(pilot_controls, samples, strict=True)
                    if s.target_action_index == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
                ],
                [
                    s.target_pilot_control
                    for s in samples
                    if s.target_action_index == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
                ],
            )
            if pilot_controls
            else {}
        ),
    )
    if action_risk_evidence is not None:
        # 上面的 risk_target 属于示范占位，不能给模型的新动作评分。
        # 行为授权仍使用实际运行时接管结果；风险准确率来自同动作的独立专用测试。
        metrics = LocalPolicyTrainingMetrics.model_validate({
            **metrics.model_dump(), **action_risk_summary(action_risk_evidence),
            'action_risk_evidence': action_risk_evidence.model_dump(),
        })
    return metrics, latencies


# 功能：
#   严格解析有界顾问 JSONL，每行必须有效，不通过跳过坏行制造更好的评估集合。
# 输入：
#   path：本次冻结且至多 256 MiB 的顾问样本文件。
# 输出：
#   samples：最多二十五万条、单行至多 4 MiB 的已验证样本。
def _load_advisor_samples(path: Path) -> list[LocalAdvisorTrainingSample]:
    content = read_plugin_file(path, limit=256 * 1024 * 1024)
    samples = []
    with BytesIO(content) as source:
        while line := source.readline(4 * 1024 * 1024 + 1):
            if len(samples) >= 250_000 or len(line) > 4 * 1024 * 1024 or not line.strip():
                raise ValueError("local advisor validation row is invalid")
            samples.append(
                LocalAdvisorTrainingSample.model_validate(decode_json(line, limit=4 * 1024 * 1024))
            )
    if not samples:
        raise ValueError("local advisor validation dataset is empty")
    return samples


# 功能：
#   根据实际打包输入名构造顾问张量；保留旧负载 state_history 的明确接口，不按角色猜代际。
# 输入：
#   session：拥有实际 ONNX 输入声明的会话。
#   sample：已经重新校验的对应角色样本。
#   role：当前顾问角色。
# 输出：
#   feeds：名称完整匹配声明的 float32 输入字典。
def _advisor_feeds(
    session: Any,
    sample: LocalAdvisorTrainingSample,
    *,
    role: PolicyArtifactRole,
) -> dict[str, Any]:
    import numpy as np

    input_names = {value.name for value in session.get_inputs()}
    if role == "perception-health-critic":
        feeds = {"state_features": np.asarray([sample.state_features], dtype=np.float32)}
    elif role == "settle-stability-critic":
        feeds = {
            "maneuver_features": np.asarray([sample.maneuver_features], dtype=np.float32),
            "state_history": np.asarray([sample.state_history], dtype=np.float32),
            "history_mask": np.asarray([sample.history_mask], dtype=np.float32),
        }
    elif role == "payload-dynamics-adapter":
        history_name = "payload_history" if "payload_history" in input_names else "state_history"
        history = (
            sample.payload_history if history_name == "payload_history" else sample.state_history
        )
        if history is None:
            raise ValueError("payload advisor validation sample has no payload history")
        feeds = {
            "payload_features": np.asarray([sample.payload_features], dtype=np.float32),
            history_name: np.asarray([history], dtype=np.float32),
            "history_mask": np.asarray([sample.history_mask], dtype=np.float32),
        }
        # 当前负载图必须同时看到瞬时运动和历史速度；不能仅完成旧三输入接口。
        # 只识别完整的五输入合同，缺项或多项仍由下面的精确集合检查拒绝。
        if input_names == {"payload_features", "maneuver_features", "payload_history",
                           "state_history", "history_mask"}:
            feeds["maneuver_features"] = np.asarray([sample.maneuver_features], dtype=np.float32)
            feeds["state_history"] = np.asarray([sample.state_history], dtype=np.float32)
    elif role == "cross-modal-consistency-critic":
        feeds = {"sensor_features": np.asarray([sample.sensor_features], dtype=np.float32)}
    elif role == "state-anomaly-detector":
        feeds = {
            "state_history": np.asarray([sample.state_history], dtype=np.float32),
            "history_mask": np.asarray([sample.history_mask], dtype=np.float32),
        }
    else:
        raise ValueError(f"unsupported trainable advisor role: {role}")
    if set(feeds) != input_names:
        raise ValueError(
            f"advisor artifact input contract mismatch for {role}: "
            f"expected={sorted(input_names)} resolved={sorted(feeds)}"
        )
    return feeds


# 功能：
#   以单线程 CPU 执行固定内嵌顾问图，验证标量类型及范围，计算风险和负载调节指标。
# 输入：
#   artifact_path：本次冻结的模型文件。
#   samples：包含本角色的已验证顾问数据。
#   role：本次实际执行的顾问角色。
# 输出：
#   metrics：带真实类别支持数量的质量指标；缺失类别不计为满分。
#   latencies：含输入构造的实际毫秒耗时，剔除预热段。
def _evaluate_advisor_artifact(
    artifact_path: Path,
    samples: list[LocalAdvisorTrainingSample],
    *,
    role: PolicyArtifactRole,
) -> tuple[LocalAdvisorTrainingMetrics, list[float]]:
    import numpy as np
    import onnxruntime as ort

    selected = [
        LocalAdvisorTrainingSample.model_validate(sample.model_dump())
        for sample in samples
        if sample.role == role
    ]
    if not selected:
        raise ValueError(f"advisor validation data has no {role} samples")
    import onnx

    content = read_plugin_file(artifact_path, limit=256 * 1024 * 1024)
    document = onnx.load_model_from_string(content)
    output_names = (
        ["risk_score", "controller_step_scale"]
        if role == "payload-dynamics-adapter"
        else ["anomaly_score"]
        if role == "state-anomaly-detector"
        else ["risk_score"]
    )
    validate_embedded_graph(
        content, input_names=[item.name for item in document.graph.input], output_names=output_names
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(content, options, providers=["CPUExecutionProvider"])
    risks: list[float] = []
    scales: list[float] = []
    latencies: list[float] = []
    for index, sample in enumerate(selected):
        started = time.perf_counter()
        feeds = _advisor_feeds(session, sample, role=role)
        outputs = session.run(output_names, feeds)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if index >= min(20, len(selected) // 4):
            latencies.append(elapsed_ms)
        if (
            not isinstance(outputs, (list, tuple))
            or len(outputs) != len(output_names)
            or any(
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.float32)
                or value.shape not in {(), (1,), (1, 1)}
                or not np.isfinite(value).all()
                for value in outputs
            )
        ):
            raise RuntimeError("LOCAL_ADVISOR_ONNX_OUTPUT_INVALID")
        risk = float(outputs[0].item())
        if not 0.0 <= risk <= 1.0:
            raise RuntimeError("LOCAL_ADVISOR_ONNX_RISK_INVALID")
        risks.append(risk)
        if role == "payload-dynamics-adapter":
            scale = float(outputs[1].item())
            if not math.isfinite(scale) or not 0.1 <= scale <= 1.0:
                raise RuntimeError("LOCAL_ADVISOR_ONNX_SCALE_INVALID")
            scales.append(scale)
    if not latencies:
        raise RuntimeError("LOCAL_ADVISOR_LATENCY_SAMPLE_EMPTY")
    target_risk = np.asarray([sample.risk_target for sample in selected], dtype=np.float64)
    predicted_risk = np.asarray(risks, dtype=np.float64)
    risky = target_risk >= 0.5
    safe = ~risky
    predicted_risky = predicted_risk >= 0.5
    scale_error = None
    if role == "payload-dynamics-adapter":
        target_scale = np.asarray(
            [sample.controller_step_scale_target for sample in selected],
            dtype=np.float64,
        )
        scale_error = float(np.abs(np.asarray(scales, dtype=np.float64) - target_scale).mean())
    metrics = LocalAdvisorTrainingMetrics(
        sample_count=len(selected),
        risky_sample_count=int(risky.sum()),
        safe_sample_count=int(safe.sum()),
        risk_hold_recall=(float(predicted_risky[risky].mean()) if risky.any() else 0.0),
        safe_motion_recall=(float((~predicted_risky[safe]).mean()) if safe.any() else 0.0),
        risk_mean_absolute_error=float(np.abs(predicted_risk - target_risk).mean()),
        controller_step_scale_mean_absolute_error=scale_error,
    )
    return metrics, latencies


# 功能：
#   预验互斥来源和输出，固定本次所有模型、数据及回执后执行评估，结束时清理私有副本。
# 输入：
#   无：命令行提供模型包、验证集、来源及视觉回执、可选顾问证据和新输出文件。
# 输出：
#   status：准入成功为零，指标未达标为一；解析、执行和发布异常继续抛出。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--validation-observations", type=Path)
    parser.add_argument("--heading-observations", type=Path)
    parser.add_argument("--stream-groups", type=Path)
    parser.add_argument("--visual-encoding-receipt", type=Path)
    parser.add_argument("--composition-receipt", type=Path)
    parser.add_argument("--augmentation-receipt", type=Path)
    parser.add_argument("--rebinding-receipt", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--navigation-training-dataset-receipt", type=Path)
    parser.add_argument("--inherited-admission-receipt", type=Path)
    parser.add_argument("--advisor-validation-data", type=Path, action="append", default=[])
    parser.add_argument("--advisor-training-receipt", type=Path, action="append", default=[])
    parser.add_argument("--advisor-dataset-receipt", type=Path, action="append", default=[])
    parser.add_argument('--risk-validation-data', type=Path, action='append', default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_plain_plugin_path(args.output)
    check_plain_plugin_path(args.package)
    args.output = args.output.resolve()
    args.package = args.package.resolve()
    if args.output.is_relative_to(args.package):
        raise ValueError("admission receipt must stay outside the evaluated package")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (
        len(args.advisor_validation_data)
        == len(args.advisor_training_receipt)
        == len(args.advisor_dataset_receipt)
    ):
        parser.error(
            "each advisor admission dataset requires matching training and dataset receipts"
        )
    preservation_receipts = [
        value
        for value in (
            args.composition_receipt,
            args.augmentation_receipt,
            args.rebinding_receipt,
            args.training_receipt,
        )
        if value is not None
    ]
    if len(preservation_receipts) > 1:
        parser.error(
            "composition, augmentation, rebinding, and training receipts are mutually exclusive"
        )
    preservation_receipt = preservation_receipts[0] if preservation_receipts else None
    if (preservation_receipt is None) != (args.inherited_admission_receipt is None):
        parser.error("preservation and inherited admission receipts must be supplied together")
    if (args.augmentation_receipt is None) != (args.navigation_training_dataset_receipt is None):
        parser.error(
            "augmentation and navigation training dataset receipts must be supplied together"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_suite_sha256 = _sha256(Path(__file__))
    with TemporaryDirectory(dir=args.output.parent, prefix=".admission-inputs-") as directory:
        frozen = freeze_admission_inputs(args, Path(directory))
        status = _evaluate_frozen_inputs(frozen)
    return status


# 功能：
#   1. 对固定输入评估导航、视觉耗时和顾问指标，按来源契约和质量阈值生成准入或拒绝回执。
#   2. 回执无覆盖发布，不把离线准入当成闭环成功或飞行资格。
# 输入：
#   args：全部来源都位于本次独占目录的参数对象。
# 输出：
#   status：普通或引导仿真准入成功为零，指标拒绝为一。
def _evaluate_frozen_inputs(args) -> int:
    preservation_receipt = next(
        (
            getattr(args, name)
            for name in (
                "composition_receipt",
                "augmentation_receipt",
                "rebinding_receipt",
                "training_receipt",
            )
            if getattr(args, name) is not None
        ),
        None,
    )
    package = load_local_policy_package(args.package)
    original_dataset_receipt, dataset_digest = _read_receipt(args.dataset_receipt)
    dataset_receipt = navigation_dataset_contract(
        original_dataset_receipt,
        causal=package.manifest.navigation_architecture == "causal-gru-control",
        historical_groups=args.expert_training_groups,
    )
    observations = None
    if package.manifest.navigation_architecture == "causal-gru-control":
        observations = read_admission_observations(
            args.validation_observations, args.stream_groups, original_dataset_receipt
        )
    training_sources = set(_source_evidence_hashes(dataset_receipt, "training"))
    validation_sources = set(_source_evidence_hashes(dataset_receipt, "validation"))
    if training_sources & validation_sources:
        raise ValueError("navigation admission source missions overlap across partitions")
    samples = load_training_samples(args.validation_data)
    perception_encoder = package.artifact_paths.get("perception-encoder")
    visual_receipt_sha256 = None
    perception_encoder_sha256 = None
    visual_preprocess_p99 = None
    visual_encoder_p99 = None
    source_validation_sha256: str
    if perception_encoder is None:
        if args.visual_encoding_receipt is not None:
            raise ValueError("nonvisual package cannot use a visual encoding receipt")
        if any(sample.visual_features for sample in samples):
            raise ValueError("nonvisual package cannot evaluate visual policy samples")
        if dataset_receipt.get("validation_output_sha256") != _sha256(args.validation_data):
            raise ValueError("validation data does not match its dataset receipt")
        source_validation_sha256 = _sha256(args.validation_data)
    else:
        if args.visual_encoding_receipt is None:
            raise ValueError("visual package requires a visual encoding receipt")
        visual_receipt, visual_receipt_sha256 = _read_receipt(args.visual_encoding_receipt)
        if visual_receipt.get(
            "schema_version"
        ) != "dronedream.local-policy-visual-encoding-receipt.v1" or (
            visual_receipt.get("qualification_granted") is not False
        ):
            raise ValueError("visual encoding receipt contract is invalid")
        source_validation_sha256 = dataset_receipt.get("validation_output_sha256")
        if not isinstance(source_validation_sha256, str):
            raise ValueError("visual source validation hash is invalid")
        if visual_receipt.get("source_policy_data_sha256") != source_validation_sha256:
            raise ValueError("visual encoding receipt is not bound to the held-out split")
        if visual_receipt.get("output_sha256") != _sha256(args.validation_data):
            raise ValueError("visual validation data does not match its encoding receipt")
        perception_encoder_sha256 = _sha256(perception_encoder)
        if visual_receipt.get("perception_encoder_sha256") != perception_encoder_sha256:
            raise ValueError("visual validation features use a different encoder")
        declared_visual = VisualInputContract.model_validate(
            {name: visual_receipt.get(name) for name in VisualInputContract.model_fields}
        )
        if declared_visual != VisualInputContract.from_manifest(
            package.manifest, perception_encoder_sha256
        ):
            raise ValueError("visual validation features use a different input contract")
        if type(visual_receipt.get("sample_count")) is not int or visual_receipt[
            "sample_count"
        ] != len(samples):
            raise ValueError("visual encoding sample count differs from validation data")
        feature_count = package.manifest.visual_feature_count
        if feature_count is None or any(
            len(sample.visual_features) != feature_count for sample in samples
        ):
            raise ValueError("visual validation sample feature width is invalid")
        visual_preprocess_p99 = latency_percentile(
            [visual_receipt.get("p99_preprocess_latency_ms")], 99
        )
        visual_encoder_p99 = latency_percentile([visual_receipt.get("p99_encoder_latency_ms")], 99)
    heading_path = getattr(args, 'heading_observations', None)
    heading_inputs = (read_heading_admission_inputs(heading_path)
                      if heading_path is not None else None)
    action_risk_evidence = _evaluate_frozen_action_risk(package, args)
    metrics, latencies = _evaluate_runtime_package(
        args.package, args.validation_data, observations=observations,
        heading_inputs=heading_inputs,
        action_risk_evidence=action_risk_evidence)
    # 因果暖机行不是计分样本。回执类别数量取实际评估指标，不能用原输入总行数充数。
    evaluated_count = metrics.sample_count
    motion_sample_count, non_motion_sample_count = (
        metrics.motion_sample_count,
        metrics.non_motion_sample_count,
    )
    realtime_complete = all(
        sample.realtime_features and sample.realtime_valid_mask for sample in samples
    )
    pilot_complete = all(sample.target_pilot_control for sample in samples)
    realtime_feature_ready_sample_count = evaluated_count if realtime_complete else 0
    pilot_control_target_sample_count = evaluated_count if pilot_complete else 0
    percentiles = {
        "p50": latency_percentile(latencies, 50.0),
        "p95": latency_percentile(latencies, 95.0),
        "p99": latency_percentile(latencies, 99.0),
    }
    issues: list[str] = []
    if action_risk_evidence is not None:
        issues.extend(action_risk_evidence_issues(action_risk_evidence,
            maximum_latency_ms=package.manifest.maximum_inference_latency_ms))
    navigation_expert_metrics: dict[str, LocalPolicyTrainingMetrics] = {}
    for role in NAVIGATION_EXPERT_ROLES:
        if role not in package.artifact_paths:
            continue
        if not any(sample.navigation_expert_role == role for sample in samples):
            issues.append(f"NAVIGATION_EXPERT_EVIDENCE_MISSING_{role.upper()}")
            continue
        role_metrics, _role_latencies = _evaluate_runtime_package(
            args.package, args.validation_data, navigation_role=role, observations=observations,
            heading_inputs=heading_inputs,
            action_risk_evidence=action_risk_evidence,
        )
        navigation_expert_metrics[role] = role_metrics
        issues.extend(
            f"{role.upper()}_{issue}"
            for issue in navigation_quality_issues(
                role_metrics, continuous_control=package.manifest.pilot_control_mode is not None
            )
        )
    advisor_metrics: dict[PolicyArtifactRole, LocalPolicyAdvisorAdmissionMetrics] = {}
    advisor_evidence: list[dict[str, object]] = []
    navigation_admission_split = None
    navigation_admission_source_hashes: list[str] = []
    (
        admission_scope,
        minimum_motion_sample_count,
        minimum_non_motion_sample_count,
    ) = _navigation_admission_scope(args.augmentation_receipt)
    trainable_advisor_roles: set[PolicyArtifactRole] = {
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
    if preservation_receipt is not None:
        advisor_metrics, advisor_evidence = _load_inherited_advisor_evidence(
            package_path=args.package,
            preservation_receipt_path=preservation_receipt,
            inherited_admission_receipt_path=args.inherited_admission_receipt,
            trainable_advisor_roles=trainable_advisor_roles,
        )
    if args.augmentation_receipt is not None:
        navigation_admission_split, navigation_admission_source_hashes = (
            _validate_fresh_navigation_admission_data(
                source_data_sha256=source_validation_sha256,
                admission_dataset_receipt_path=args.dataset_receipt,
                augmentation_receipt_path=args.augmentation_receipt,
                training_dataset_receipt_path=(args.navigation_training_dataset_receipt),
            )
        )
    for advisor_data_path, advisor_receipt_path, advisor_dataset_receipt_path in zip(
        args.advisor_validation_data,
        args.advisor_training_receipt,
        args.advisor_dataset_receipt,
        strict=True,
    ):
        advisor_receipt, _ = _read_receipt(advisor_receipt_path)
        if advisor_receipt.get("training_accepted") is not True:
            raise ValueError("advisor training receipt was not accepted")
        admission_split, admission_source_hashes = _validate_fresh_advisor_admission_data(
            data_path=advisor_data_path,
            dataset_receipt_path=advisor_dataset_receipt_path,
            training_receipt=advisor_receipt,
        )
        receipt_advisors = advisor_receipt.get("advisors")
        if not isinstance(receipt_advisors, dict):
            raise ValueError("advisor training receipt has no advisor records")
        advisor_samples = _load_advisor_samples(advisor_data_path)
        for raw_role, record in receipt_advisors.items():
            role = str(raw_role)
            if role not in trainable_advisor_roles or not isinstance(record, dict):
                raise ValueError(f"unsupported advisor training receipt role: {role}")
            if record.get("accepted") is not True:
                raise ValueError(f"advisor training record was not accepted: {role}")
            if role in advisor_metrics:
                raise ValueError(f"duplicate advisor admission evidence: {role}")
            artifact_path = package.artifact_paths.get(role)  # type: ignore[arg-type]
            if artifact_path is None:
                raise ValueError(f"advisor evidence is not present in package: {role}")
            if record.get("artifact_sha256") != _sha256(artifact_path):
                raise ValueError(f"packaged advisor differs from training receipt: {role}")
            advisor_result, advisor_latencies = _evaluate_advisor_artifact(
                artifact_path,
                advisor_samples,
                role=role,  # type: ignore[arg-type]
            )
            advisor_p99 = latency_percentile(advisor_latencies, 99.0)
            advisor_metrics[role] = LocalPolicyAdvisorAdmissionMetrics(  # type: ignore[index]
                **advisor_result.model_dump(mode="json"),
                p99_inference_latency_ms=advisor_p99,
            )
            if advisor_result.risky_sample_count < 20:
                issues.append(f"ADVISOR_RISKY_SAMPLES_TOO_LOW_{role.upper()}")
            if advisor_result.safe_sample_count < 20:
                issues.append(f"ADVISOR_SAFE_SAMPLES_TOO_LOW_{role.upper()}")
            if advisor_result.risk_hold_recall < 0.95:
                issues.append(f"ADVISOR_RISK_HOLD_RECALL_TOO_LOW_{role.upper()}")
            if advisor_result.safe_motion_recall < 0.95:
                issues.append(f"ADVISOR_SAFE_RECALL_TOO_LOW_{role.upper()}")
            if advisor_result.risk_mean_absolute_error > 0.2:
                issues.append(f"ADVISOR_RISK_ERROR_TOO_HIGH_{role.upper()}")
            scale_error = advisor_result.controller_step_scale_mean_absolute_error
            if role == "payload-dynamics-adapter" and (scale_error is None or scale_error > 0.15):
                issues.append("ADVISOR_PAYLOAD_SCALE_ERROR_TOO_HIGH")
            if advisor_p99 > package.manifest.maximum_inference_latency_ms:
                issues.append(f"ADVISOR_INFERENCE_LATENCY_TOO_HIGH_{role.upper()}")
            advisor_evidence.append(
                {
                    "role": role,
                    "training_receipt_sha256": _sha256(advisor_receipt_path),
                    "admission_dataset_receipt_sha256": _sha256(advisor_dataset_receipt_path),
                    "admission_data_sha256": _sha256(advisor_data_path),
                    "admission_split": admission_split,
                    "admission_source_evidence_sha256": admission_source_hashes,
                    "artifact_sha256": _sha256(artifact_path),
                    "metrics": advisor_metrics[role].model_dump(mode="json"),
                }
            )
    advertised_trainable_advisors = trainable_advisor_roles & set(package.artifact_paths)
    missing_advisor_evidence = advertised_trainable_advisors - set(advisor_metrics)
    for role in sorted(missing_advisor_evidence):
        issues.append(f"ADVISOR_OFFLINE_EVIDENCE_MISSING_{role.upper()}")
    if motion_sample_count < minimum_motion_sample_count:
        issues.append("INSUFFICIENT_MOTION_SAMPLES")
    if non_motion_sample_count < minimum_non_motion_sample_count:
        issues.append("INSUFFICIENT_NON_MOTION_SAMPLES")
    issues.extend(
        navigation_quality_issues(
            metrics, continuous_control=package.manifest.pilot_control_mode is not None
        )
    )
    if package.manifest.pilot_control_mode is not None:
        if not realtime_complete:
            issues.append("REALTIME_FEATURE_EVIDENCE_INCOMPLETE")
        if not pilot_complete:
            issues.append("PILOT_CONTROL_TARGET_EVIDENCE_INCOMPLETE")
        if metrics.pilot_control_mean_absolute_error is None:
            issues.append("PILOT_CONTROL_ERROR_EVIDENCE_MISSING")
        elif metrics.pilot_control_mean_absolute_error > 0.2:
            issues.append("PILOT_CONTROL_ERROR_TOO_HIGH")
    if package.manifest.pilot_control_mode is None and "risk-critic" not in package.artifact_paths:
        issues.append("INDEPENDENT_RISK_CRITIC_REQUIRED")
    if percentiles["p99"] > package.manifest.maximum_inference_latency_ms:
        issues.append("INFERENCE_LATENCY_TOO_HIGH")
    combined_p99 = (
        percentiles["p99"]
        + (visual_preprocess_p99 or 0.0)
        + (visual_encoder_p99 or 0.0)
        + sum(item.p99_inference_latency_ms for item in advisor_metrics.values())
    )
    if combined_p99 > package.manifest.maximum_inference_latency_ms:
        issues.append("COMBINED_INFERENCE_LATENCY_TOO_HIGH")
    advisor_evaluation_sha256 = (
        sha256_json(sorted(advisor_evidence, key=lambda item: str(item["role"])))
        if advisor_evidence
        else None
    )
    receipt_payload = {
        "policy_package_sha256": package.package_sha256,
        "navigation_expert_metrics": {
            role: value.model_dump(mode="json")
            for role, value in sorted(navigation_expert_metrics.items())
        },
        "dataset_receipt_sha256": dataset_digest,
        "evaluation_suite_sha256": args.evaluation_suite_sha256,
        "map_sha256": package.manifest.map_sha256,
        "vehicle_sha256": package.manifest.vehicle_sha256,
        "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
        "sample_count": evaluated_count,
        "motion_sample_count": motion_sample_count,
        "non_motion_sample_count": non_motion_sample_count,
        "motion_authorization_accuracy": metrics.motion_authorization_accuracy,
        "candidate_selection_accuracy": metrics.candidate_selection_accuracy,
        "risk_hold_recall": metrics.risk_hold_recall,
        "safe_motion_recall": metrics.safe_motion_recall,
        "risk_mean_absolute_error": metrics.risk_mean_absolute_error,
        "realtime_feature_ready_sample_count": (
            realtime_feature_ready_sample_count
            if package.manifest.pilot_control_mode is not None
            else None
        ),
        "pilot_control_target_sample_count": (
            pilot_control_target_sample_count
            if package.manifest.pilot_control_mode is not None
            else None
        ),
        "pilot_control_mean_absolute_error": (
            metrics.pilot_control_mean_absolute_error
            if package.manifest.pilot_control_mode is not None
            else None
        ),
        "p50_inference_latency_ms": percentiles["p50"],
        "p95_inference_latency_ms": percentiles["p95"],
        "p99_inference_latency_ms": percentiles["p99"],
        "visual_encoding_receipt_sha256": visual_receipt_sha256,
        "perception_encoder_sha256": perception_encoder_sha256,
        "visual_preprocess_p99_latency_ms": visual_preprocess_p99,
        "visual_encoder_p99_inference_latency_ms": visual_encoder_p99,
        "advisor_evaluation_sha256": advisor_evaluation_sha256,
        "advisor_metrics": {
            role: value.model_dump(mode="json") for role, value in sorted(advisor_metrics.items())
        },
        "combined_p99_inference_latency_ms": combined_p99,
        "artifact_preservation_receipt_sha256": (
            _sha256(preservation_receipt) if preservation_receipt is not None else None
        ),
        "inherited_admission_receipt_sha256": (
            _sha256(args.inherited_admission_receipt)
            if args.inherited_admission_receipt is not None
            else None
        ),
        "navigation_training_dataset_receipt_sha256": (
            _sha256(args.navigation_training_dataset_receipt)
            if args.navigation_training_dataset_receipt is not None
            else None
        ),
        "navigation_admission_split": navigation_admission_split,
        "navigation_admission_source_evidence_sha256": (navigation_admission_source_hashes),
        "admission_scope": admission_scope,
        "minimum_motion_sample_count": minimum_motion_sample_count,
        "minimum_non_motion_sample_count": minimum_non_motion_sample_count,
        "admitted_to_simulation": not issues,
        "issue_codes": list(dict.fromkeys(issues)),
    }
    if heading_path is not None:
        receipt_payload['heading_observations_sha256'] = _sha256(heading_path)
    receipt = LocalPolicySimulationAdmissionReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id=("policy-simulation-admission-" + sha256_json(receipt_payload)[:32]),
        **receipt_payload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(args.output, receipt.model_dump(mode="json"))
    print(receipt.model_dump_json())
    status = 0 if receipt.admitted_to_simulation else 1
    return status


# 功能：
#   使用已冻结的同包风险图与独立风险数据重新推理，拒绝训练/开发分组重用和缺失类别。
# 输入：
#   package：本次固定模型包；args：包含固定风险数据与原训练回执的评估参数。
# 输出：
#   evidence：当前因果控制所需的同动作风险证据；旧非因果接口无此证据时为空。
def _evaluate_frozen_action_risk(package, args):
    roots = getattr(args, 'risk_validation_data', [])
    if not roots:
        if package.manifest.navigation_architecture == 'causal-gru-control':
            raise ValueError('CAUSAL_ADMISSION_ACTION_RISK_EVIDENCE_REQUIRED')
        return None
    evidence_path = getattr(args, 'risk_training_evidence', None)
    if evidence_path is None or 'risk-critic' not in package.artifact_paths:
        raise ValueError('ADMISSION_RISK_TRAINING_SOURCE_REQUIRED')
    training, _ = _read_receipt(evidence_path)
    artifact = next(a for a in package.manifest.artifacts if a.role == 'risk-critic')
    if training.get('model_sha256') != artifact.sha256:
        raise ValueError('ADMISSION_ACTION_RISK_MODEL_MISMATCH')
    excluded = set(args.expert_training_groups)
    reports = []
    for root in roots:
        result = evaluate_action_risk_artifact(package.artifact_paths['risk-critic'], [root],
            model_sha256=artifact.sha256, teacher_sha256=training['teacher_config_sha256'],
            excluded_groups=excluded,
            maximum_latency_ms=package.manifest.maximum_inference_latency_ms)
        if not result['inference_performed']:
            raise ValueError('ADMISSION_ACTION_RISK_COVERAGE_INCOMPLETE:'
                             + ','.join(result['issue_codes']))
        reports.append(result)
        excluded.update(result['test_groups'])
    evidence = action_risk_evidence_from_reports(reports)
    return evidence


if __name__ == "__main__":
    raise SystemExit(main())
