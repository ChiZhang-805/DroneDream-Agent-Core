#!/usr/bin/env python3
"""Train independently qualified local perception and payload advisors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from dronedream_agent_core.asset_package_storage import publish_asset_directory
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    LocalAdvisorTrainingMetrics,
    LocalAdvisorTrainingSample,
    TrainableAdvisorRole,
    evaluate_local_advisor,
    export_local_advisor_onnx,
    train_local_advisor,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.advisor_sources import (
    RecordedSource,
    validate_advisor_spatial_splits,
)
from dronedream_agent_core.training.dagger_artifacts import write_rows
from dronedream_agent_core.training.evidence_files import decode_evidence_rows
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from dronedream_plugin_sdk.protocol import decode_json

MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT = 2
ADVISOR_ROLES = get_args(TrainableAdvisorRole)


# 功能：
#   从已冻结的字节严格读取完整 JSONL 样本，拒绝歧义记录、空集和隐式数值转换。
# 输入：
#   path：读取期间已检查身份与大小的不可变数据来源。
# 输出：
#   samples：通过当前专家输入契约的训练样本。
def _load_samples(path: RecordedSource) -> list[LocalAdvisorTrainingSample]:
    samples = [
        LocalAdvisorTrainingSample.model_validate(row, strict=True)
        for row in decode_evidence_rows(path.read_bytes())
    ]
    if not samples:
        raise ValueError("local advisor dataset is empty")
    return samples


# 功能：
#   计算冻结来源的原始字节摘要，不在训练完成后重新读取可能已变化的路径。
# 输入：
#   path：不可变数据来源。
# 输出：
#   checksum：实际读取字节的 SHA-256。
def _sha256(path: RecordedSource) -> str:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksum


# 功能：
#   将完整样本规范序列化为内容摘要，用于发现两个分区中的重复样本。
# 输入：
#   sample：已验证的专家训练样本。
# 输出：
#   checksum：包含标签及显式空字段的样本摘要。
def _sample_sha256(sample: LocalAdvisorTrainingSample) -> str:
    checksum = hashlib.sha256(
        sample.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()
    return checksum


# 功能：
#   拒绝训练与验证的完整样本重复；任务和空间隔离由数据集回执另行校验。
# 输入：
#   training：训练分区样本。
#   validation：验证分区样本。
# 输出：
#   None：不返回业务数据。
def _assert_disjoint_samples(
    training: list[LocalAdvisorTrainingSample],
    validation: list[LocalAdvisorTrainingSample],
) -> None:
    overlap = {_sample_sha256(sample) for sample in training} & {
        _sample_sha256(sample) for sample in validation
    }
    if overlap:
        raise ValueError("local advisor training and validation samples overlap")


# 功能：
#   提取有界来源列表中的规范摘要，去重后用于源任务隔离与任务数量检查。
# 输入：
#   receipt：严格解析的数据集回执。
#   split：training 或 validation 分区名。
# 输出：
#   source_hashes：按字典序排列的不重复原始任务摘要。
def _source_evidence_hashes(receipt: dict[str, object], split: str) -> list[str]:
    if split not in {"training", "validation"}:
        raise ValueError("local advisor dataset split is invalid")
    raw_sources = receipt.get(f"{split}_sources")
    if type(raw_sources) is not list or not 1 <= len(raw_sources) <= 10_000:
        raise ValueError(f"local advisor dataset receipt has no {split} sources")
    hashes: set[str] = set()
    for source in raw_sources:
        if type(source) is not dict:
            raise ValueError("local advisor dataset source record is invalid")
        value = source.get("mission_evidence_sha256")
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("local advisor dataset source evidence hash is invalid")
        hashes.add(value)
    source_hashes = sorted(hashes)
    return source_hashes


# 功能：
#   1. 核对回执协议、特征契约、实际数据字节、专家角色及空间任务隔离。
#   2. 负载采集来源额外要求每个分区的独立任务数量，不把训练通过当作飞行资格。
# 输入：
#   path：冻结的数据集回执来源。
#   training_data：冻结的训练字节。
#   validation_data：冻结的验证字节。
#   roles：本次明确训练的专家角色。
# 输出：
#   receipt：通过绑定检查的回执。
#   training_sources：训练任务摘要列表。
#   validation_sources：验证任务摘要列表。
def _validate_dataset_receipt(
    path: RecordedSource,
    *,
    training_data: RecordedSource,
    validation_data: RecordedSource,
    roles: tuple[TrainableAdvisorRole, ...],
) -> tuple[dict[str, object], list[str], list[str]]:
    receipt = decode_json(path.read_bytes(), limit=4 * 1024 * 1024, node_limit=1_000_000)
    if type(receipt) is not dict:
        raise ValueError("local advisor dataset receipt is not an object")
    if receipt.get("schema_version") != "dronedream.local-advisor-dataset-receipt.v1":
        raise ValueError("local advisor dataset receipt schema is invalid")
    if receipt.get("split_method") != SPATIAL_SPLIT_CONTRACT:
        raise ValueError("local advisor dataset requires content-bound spatial splits")
    if receipt.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
        raise ValueError("local advisor dataset uses an incompatible feature contract")
    if receipt.get("verified_sources_required") is not True:
        raise ValueError("local advisor dataset does not require verified sources")
    if receipt.get("training_output_sha256") != _sha256(training_data):
        raise ValueError("local advisor training data hash differs from its receipt")
    if receipt.get("validation_output_sha256") != _sha256(validation_data):
        raise ValueError("local advisor validation data hash differs from its receipt")
    receipt_roles = receipt.get("roles")
    if (
        type(receipt_roles) is not list
        or not receipt_roles
        or len(receipt_roles) > len(ADVISOR_ROLES)
        or any(type(role) is not str or role not in ADVISOR_ROLES for role in receipt_roles)
        or len(set(receipt_roles)) != len(receipt_roles)
        or not roles
        or any(role not in ADVISOR_ROLES for role in roles)
        or not set(roles).issubset(receipt_roles)
    ):
        raise ValueError("local advisor dataset receipt does not cover requested roles")
    training_sources = _source_evidence_hashes(receipt, "training")
    validation_sources = _source_evidence_hashes(receipt, "validation")
    if set(training_sources) & set(validation_sources):
        raise ValueError("local advisor dataset source missions overlap across splits")
    groups = validate_advisor_spatial_splits(
        receipt["training_sources"], receipt["validation_sources"]
    )
    for name, actual in zip(("training", "validation"), groups, strict=True):
        if receipt.get(name + "_groups") != sorted(actual):
            raise ValueError("local advisor dataset spatial groups differ from bound sources")
    if (
        receipt.get("verified_payload_collection_sources_allowed") is True
        or receipt.get("verified_payload_recovery_sources_allowed") is True
    ):
        if "payload-dynamics-adapter" not in roles:
            raise ValueError("payload collection receipt does not train the payload adapter")
        if (
            receipt.get("minimum_payload_collection_missions_per_split")
            != MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT
        ):
            raise ValueError("payload collection source-diversity policy is invalid")
        if len(training_sources) < MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT:
            raise ValueError("payload collection training split has too few missions")
        if len(validation_sources) < MINIMUM_PAYLOAD_COLLECTION_MISSIONS_PER_SPLIT:
            raise ValueError("payload collection validation split has too few missions")
    return receipt, training_sources, validation_sources


# 功能：
#   重新验证指标数值后执行风险召回、安全运动召回、误差及负载尺度的验收阈值。
# 输入：
#   role：当前专家角色。
#   metrics：独立验证分区的指标。
# 输出：
#   None：不返回业务数据。
def _validate_metrics(role: TrainableAdvisorRole, metrics: LocalAdvisorTrainingMetrics) -> None:
    if role not in ADVISOR_ROLES:
        raise ValueError("local advisor role is invalid")
    metrics = LocalAdvisorTrainingMetrics.model_validate(metrics.model_dump(), strict=True)
    role_code = role.upper().replace("-", "_")
    if metrics.risk_hold_recall < 0.95:
        raise RuntimeError(f"{role_code}_RISK_HOLD_RECALL_TOO_LOW")
    if metrics.safe_motion_recall < 0.95:
        raise RuntimeError(f"{role_code}_SAFE_MOTION_RECALL_TOO_LOW")
    if metrics.risk_mean_absolute_error > 0.2:
        raise RuntimeError(f"{role_code}_RISK_ERROR_TOO_HIGH")
    scale_error = metrics.controller_step_scale_mean_absolute_error
    if role == "payload-dynamics-adapter" and (scale_error is None or scale_error > 0.15):
        raise RuntimeError("PAYLOAD_DYNAMICS_ADAPTER_STEP_SCALE_ERROR_TOO_HIGH")


# 功能：
#   复核指标数量守恒，确保训练和验证各自至少包含 20 条危险及 20 条安全样本。
# 输入：
#   role：当前专家角色。
#   metrics：指定分区的指标。
#   split：training 或 validation 分区名。
# 输出：
#   None：不返回业务数据。
def _validate_class_coverage(
    role: TrainableAdvisorRole,
    metrics: LocalAdvisorTrainingMetrics,
    *,
    split: str,
) -> None:
    if role not in ADVISOR_ROLES or split not in {"training", "validation"}:
        raise ValueError("local advisor role or split is invalid")
    metrics = LocalAdvisorTrainingMetrics.model_validate(metrics.model_dump(), strict=True)
    role_code = role.upper().replace("-", "_")
    split_code = split.upper()
    if metrics.risky_sample_count < 20:
        raise RuntimeError(f"{role_code}_{split_code}_RISKY_CLASS_HAS_FEWER_THAN_TWENTY_SAMPLES")
    if metrics.safe_sample_count < 20:
        raise RuntimeError(f"{role_code}_{split_code}_SAFE_CLASS_HAS_FEWER_THAN_TWENTY_SAMPLES")


# 功能：
#   1. 固定命令行路径、数据和配置后逐专家训练，独立验证并保留每个专家的验收记录。
#   2. 全部通过时无覆盖发布模型目录，独占创建回执；不授予仿真或真机飞行资格。
#   3. 目录和回执不是跨文件事务，回执写入失败时已发布模型不能被当作完整合格包。
# 输入：
#   无；运行参数由命令行提供。
# 输出：
#   exit_code：全部训练验收通过为 0，存在阈值失败为 1。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--hidden-feature-count", type=int, default=32)
    parser.add_argument("--epoch-count", type=int, default=100)
    parser.add_argument("--payload-history-dropout", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--random-seed", type=int, default=2905)
    parser.add_argument("--settle-motion-only", action="store_true",
                        help="Restrict settle history to velocity and physical scale; "
                             "other roles reject this option.")
    parser.add_argument("--sensor-state-only", action="store_true",
                        help="Restrict health/anomaly critics to physical state and health; "
                             "exclude mission and map context.")
    parser.add_argument(
        "--validation-used-for-configuration-selection",
        action="store_true",
        help=(
            "Record that the supplied validation split influenced architecture or "
            "hyperparameter selection. A different complete mission is always required "
            "for simulation admission, but this marker makes the tuning boundary explicit."
        ),
    )
    parser.add_argument(
        "--role",
        action="append",
        choices=(
            "perception-health-critic",
            "settle-stability-critic",
            "payload-dynamics-adapter",
            "state-anomaly-detector",
            "cross-modal-consistency-critic",
        ),
        help=(
            "Advisor to train; repeat as needed. Defaults to perception health, "
            "settle stability and payload dynamics; other roles require explicit selection."
        ),
    )
    args = parser.parse_args()
    # 在优化及第三方导出回调前固定目标，不依赖过程内可能变化的工作目录。
    args.output_directory = args.output_directory.absolute()
    args.training_receipt = args.training_receipt.absolute()
    check_plain_plugin_path(args.output_directory)
    check_plain_plugin_path(args.training_receipt)
    if args.training_receipt.is_relative_to(args.output_directory):
        raise ValueError("local advisor training receipt must be outside the model directory")
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    if args.training_receipt.exists():
        raise FileExistsError(args.training_receipt)

    roles: tuple[TrainableAdvisorRole, ...] = tuple(
        dict.fromkeys(
            args.role
            or (
                "perception-health-critic",
                "settle-stability-critic",
                "payload-dynamics-adapter",
            )
        )
    )
    if args.settle_motion_only and roles != ("settle-stability-critic",):
        parser.error("--settle-motion-only requires only --role settle-stability-critic")
    if args.sensor_state_only and (args.settle_motion_only or not set(roles) <= {
            "perception-health-critic", "state-anomaly-detector"}):
        parser.error("--sensor-state-only requires only health/anomaly critics")
    # 一次验证全部配置，避免后一个种子越界时才中断已经完成的前序训练。
    configs = [
        LocalAdvisorTrainingConfig(
            hidden_feature_count=args.hidden_feature_count,
            epoch_count=args.epoch_count,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed + index,
            settle_motion_only=args.settle_motion_only,
            sensor_state_only=args.sensor_state_only,
            payload_history_dropout=args.payload_history_dropout,
        )
        for index in range(len(roles))
    ]
    # Hash, validate, optimize and report these exact immutable inputs. A path
    # modified during optimization cannot replace the recorded training data.
    receipt_source, training_source, validation_source = (
        RecordedSource.read(path)
        for path in (args.dataset_receipt, args.training_data, args.validation_data)
    )
    dataset_receipt, training_sources, validation_sources = _validate_dataset_receipt(
        receipt_source,
        training_data=training_source,
        validation_data=validation_source,
        roles=roles,
    )
    training = _load_samples(training_source)
    validation = _load_samples(validation_source)
    _assert_disjoint_samples(training, validation)
    receipts: dict[str, object] = {}
    issue_codes: list[str] = []
    args.output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output_directory.parent,
        prefix=f".{args.output_directory.name}.staging-",
    ) as temporary_root:
        staging = Path(temporary_root) / "advisors"
        staging.mkdir(parents=True)
        for role, config in zip(roles, configs, strict=True):
            model, training_metrics = train_local_advisor(
                training,
                config,
                role=role,
            )
            validation_metrics = evaluate_local_advisor(model, validation)
            try:
                _validate_class_coverage(
                    role,
                    training_metrics,
                    split="training",
                )
                _validate_class_coverage(
                    role,
                    validation_metrics,
                    split="validation",
                )
                _validate_metrics(role, validation_metrics)
            except RuntimeError as error:
                issue_code = str(error)
                issue_codes.append(issue_code)
                receipts[role] = {
                    "artifact": None,
                    "artifact_sha256": None,
                    "training_config": config.model_dump(mode="json"),
                    "training_metrics": training_metrics.model_dump(mode="json"),
                    "validation_metrics": validation_metrics.model_dump(mode="json"),
                    "accepted": False,
                    "issue_code": issue_code,
                }
                continue
            path = staging / f"{role}.onnx"
            artifact_sha256 = export_local_advisor_onnx(model, path)
            receipts[role] = {
                "artifact": path.name,
                "artifact_sha256": artifact_sha256,
                "training_config": config.model_dump(mode="json"),
                "training_metrics": training_metrics.model_dump(mode="json"),
                "validation_metrics": validation_metrics.model_dump(mode="json"),
                "accepted": True,
            }
        if not issue_codes:
            # 检查已知冲突，最终由无覆盖发布和独占文件创建拒绝后续竞争。
            check_plain_plugin_path(args.training_receipt)
            if args.training_receipt.exists():
                raise FileExistsError(args.training_receipt)
            publish_asset_directory(staging, args.output_directory)

    receipt = {
        "schema_version": "dronedream.local-advisor-training-receipt.v1",
        "dataset_receipt_sha256": _sha256(receipt_source),
        "dataset_split_method": dataset_receipt["split_method"],
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "training_groups": dataset_receipt["training_groups"],
        "validation_groups": dataset_receipt["validation_groups"],
        "verified_sources_required": True,
        "training_data_sha256": _sha256(training_source),
        "validation_data_sha256": _sha256(validation_source),
        "training_source_evidence_sha256": training_sources,
        "validation_source_evidence_sha256": validation_sources,
        "validation_used_for_configuration_selection": (
            args.validation_used_for_configuration_selection
        ),
        "fresh_admission_validation_required": True,
        "advisors": receipts,
        "training_accepted": not issue_codes,
        "issue_codes": issue_codes,
        "qualification_granted": False,
        "qualification_requirement": (
            "Run an independent held-out PX4/Gazebo campaign before flight selection."
        ),
    }
    check_plain_plugin_path(args.training_receipt)
    args.training_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.training_receipt, [receipt])
    print(json.dumps(receipt, ensure_ascii=False))
    exit_code = 0 if not issue_codes else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
