"""Risk-only optimization and independent route-level validation.

No optimizer receives hypothetical probe actions as behavior-cloning targets.
Offline agreement with a nominal teacher is distinct from physical safety.
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..local_policy_quality import LocalRiskCriticMetrics
from ..local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingSample,
    NumpyLocalPolicyModel,
    evaluate_local_risk_critic,
    export_local_risk_critic_onnx,
    policy_input_arrays,
    train_local_risk_critic,
)
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from ..realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from .action_risk_artifacts import validate_action_risk_splits
from .dagger_artifacts import write_rows
from .evidence_snapshot import detach_evidence
from .mission_groups import SPATIAL_SPLIT_CONTRACT


# 功能：
#   核对统计用的样本、风险标签和观测摘要，拒绝歧义计数，不代替数据集来源接纳。
# 输入：
#   datasets：已经接纳或明确用于纯指标测试的数据集列表。
# 输出：
#   records：按输入次序排列的样本和观测记录对。
def _metric_records(datasets):
    if type(datasets) not in (list, tuple) or not 1 <= len(datasets) <= 128:
        raise ValueError("ACTION_RISK_DATASET_COUNT_INVALID")
    total = sum(len(dataset.samples) for dataset in datasets)
    if not 1 <= total <= 250_000:
        raise ValueError("ACTION_RISK_SAMPLE_COUNT_INVALID")
    records = []
    for dataset in datasets:
        for sample, record in zip(dataset.samples, dataset.records, strict=True):
            if (
                not isinstance(sample, LocalPolicyTrainingSample)
                or type(sample.risk_target) not in (int, float)
                or not math.isfinite(sample.risk_target)
                or not 0 <= sample.risk_target <= 1
            ):
                raise ValueError("ACTION_RISK_TARGET_INVALID")
            key = record.get("observation_sha256") if type(record) is dict else None
            if (
                type(key) is not str
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
            ):
                raise ValueError("ACTION_RISK_OBSERVATION_IDENTITY_INVALID")
            records.append((sample, record))
    return records


# 功能：
#   按独立观测统计安全、危险和动作风险差异，不把同一画面的多个控制探针当作新场景。
# 输入：
#   datasets：待计算覆盖的数据集。
# 输出：
#   coverage：样本、独立观测、类别、动作差异和地图路线组的数量。
def risk_class_coverage(datasets):
    safe, unsafe, contrast = set(), set(), set()
    grouped = defaultdict(list)
    for sample, record in _metric_records(datasets):
        key = record["observation_sha256"]
        (unsafe if sample.risk_target >= 0.5 else safe).add(key)
        grouped[key].append(sample.risk_target)
    for key, targets in grouped.items():
        if max(targets) - min(targets) >= 0.1:
            contrast.add(key)
    coverage = {
        "sample_count": sum(len(dataset.samples) for dataset in datasets),
        "independent_observation_count": len(grouped),
        "safe_observation_count": len(safe),
        "unsafe_observation_count": len(unsafe),
        "action_contrast_observation_count": len(contrast),
        "route_map_group_count": len(set().union(*(dataset.groups for dataset in datasets))),
    }
    return coverage


# 功能：
#   在优化前固定检查两个分区的安全、危险和动作差异观测数量，不能按训练结果降低门槛。
# 输入：
#   training：训练分区的数据集。
#   validation：验证分区的数据集。
# 输出：
#   issues：未达到各类二十条独立观测要求的问题码。
def coverage_issues(training, validation):
    issues = []
    # Twenty probes of one image are still one observed state, not twenty
    # demonstrations of independent safety behavior. Never tune this after a run.
    for split, datasets in (("training", training), ("validation", validation)):
        coverage = risk_class_coverage(datasets)
        for field in (
            "safe_observation_count",
            "unsafe_observation_count",
            "action_contrast_observation_count",
        ):
            if coverage[field] < 20:
                issues.append(f"ACTION_RISK_{split.upper()}_{field.upper()}_BELOW_TWENTY")
    return issues


# 功能：
#   校验并复制当前动作条件网络，按部署输入顺序计算有限参考风险，拒绝旧的无动作网络。
# 输入：
#   model：不含视觉旁路的当前动作风险网络。
#   samples：带真实物理归一化候选控制的样本。
# 输出：
#   predictions：一维风险评分。
#   feeds：同一批计算对应的 ONNX 输入张量。
def _predict(model, samples):
    if (
        not isinstance(model, NumpyLocalPolicyModel)
        or model.action_conditioned is not True
        or model.visual_feature_count != 0
        or model.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
    ):
        raise ValueError("ACTION_RISK_MODEL_CONTRACT_INVALID")
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= 250_000:
        raise ValueError("ACTION_RISK_SAMPLE_COUNT_INVALID")
    samples = [LocalPolicyTrainingSample.model_validate(sample.model_dump()) for sample in samples]
    inputs = policy_input_arrays(
        samples, include_visual=False, include_realtime=True, include_proposed_control=True
    )
    source = model.input_weight
    if type(source) is not np.ndarray or source.ndim != 2 or not 8 <= source.shape[1] <= 1024:
        raise ValueError("ACTION_RISK_MODEL_SHAPE_INVALID")
    hidden_width = source.shape[1]
    parameters = {}
    for name, shape in {
        "input_weight": (inputs.shape[1], hidden_width),
        "input_bias": (hidden_width,),
        "risk_weight": (hidden_width, 1),
        "risk_bias": (1,),
    }.items():
        value = getattr(model, name)
        if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != shape:
            raise ValueError("ACTION_RISK_MODEL_SHAPE_INVALID")
        parameters[name] = value.copy()
        if not np.isfinite(parameters[name]).all():
            raise ValueError("ACTION_RISK_MODEL_NONFINITE")
    feeds = {
        "state_features": np.asarray([row.state_features for row in samples], dtype=np.float32),
        "realtime_features": np.asarray(
            [row.realtime_features for row in samples], dtype=np.float32
        ),
        "realtime_valid_mask": np.asarray(
            [row.realtime_valid_mask for row in samples], dtype=np.float32
        ),
        "proposed_control": np.asarray(
            [row.risk_proposed_control for row in samples], dtype=np.float32
        ),
    }
    try:
        with np.errstate(over="raise", invalid="raise"):
            hidden = np.maximum(inputs @ parameters["input_weight"] + parameters["input_bias"], 0.0)
            logits = hidden @ parameters["risk_weight"] + parameters["risk_bias"]
    except FloatingPointError as error:
        raise ValueError("ACTION_RISK_REFERENCE_OVERFLOW") from error
    if not np.isfinite(logits).all():
        raise ValueError("ACTION_RISK_REFERENCE_NONFINITE")
    predictions = (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).reshape(-1)
    return predictions, feeds


# 功能：
#   比较同一观测中最危险与最安全动作的评分，使用并列极值的最差组合避免挑选有利案例。
# 输入：
#   datasets：样本与观测绑定的数据集。
#   predictions：一维、有限且位于零至一范围的数值评分。
# 输出：
#   discrimination：可比较观测数、正确数及比例；没有对照时比例为 None。
def action_discrimination(datasets, predictions):
    groups = defaultdict(list)
    records = _metric_records(datasets)
    predictions = np.asarray(predictions)
    if (
        predictions.dtype.kind not in "fiu"
        or predictions.shape != (len(records),)
        or not np.isfinite(predictions).all()
        or np.any(predictions < 0)
        or np.any(predictions > 1)
    ):
        raise ValueError("ACTION_RISK_PREDICTIONS_INVALID")
    for (sample, record), prediction in zip(records, predictions, strict=True):
        groups[record["observation_sha256"]].append((sample.risk_target, float(prediction)))
    pairs = []
    for rows in groups.values():
        # Compare all tied extreme actions conservatively; don't cherry-pick the
        # one safe/unsafe action pair the model happens to rank correctly.
        minimum, maximum = min(v[0] for v in rows), max(v[0] for v in rows)
        if maximum - minimum < 0.1:
            continue
        predicted_low = max(v[1] for v in rows if v[0] == minimum)
        predicted_high = min(v[1] for v in rows if v[0] == maximum)
        pairs.append(predicted_high > predicted_low)
    discrimination = {
        "observation_count": len(pairs),
        "correct_count": sum(pairs),
        "correct_fraction": sum(pairs) / len(pairs) if pairs else None,
    }
    return discrimination


# 功能：
#   1. 固定已接纳证据和配置，先检查空间隔离与覆盖，再只优化风险网络而非控制动作。
#   2. 核对独立验证和导出等价性，保存完整离线回执，永不授予飞行资格。
# 输入：
#   training：已由数据集加载器接纳的训练分区。
#   validation：已接纳且空间独立的验证分区。
#   config：风险网络训练参数。
#   output：尚不存在的本次离线训练输出目录。
# 输出：
#   receipt：明确包含优化、离线验收及后续物理验证要求的回执。
def train_action_risk_expert(training, validation, config: LocalPolicyTrainingConfig, output: Path):
    config = LocalPolicyTrainingConfig.model_validate(config.model_dump(), strict=True)
    output = Path(output).absolute()
    check_plain_plugin_path(output)
    _metric_records(training)
    _metric_records(validation)
    # 数据集冻结类内部仍有可变字典和模型；优化器只拿副本，回执不共享调用方容器。
    training, validation = (
        [
            replace(
                dataset,
                samples=tuple(
                    LocalPolicyTrainingSample.model_validate(row.model_dump())
                    for row in dataset.samples
                ),
                records=tuple(detach_evidence(row) for row in dataset.records),
                observations=tuple(detach_evidence(row) for row in dataset.observations),
                receipt=detach_evidence(dataset.receipt),
                groups=frozenset(dataset.groups),
            )
            for dataset in split
        ]
        for split in (training, validation)
    )
    validate_action_risk_splits(training, validation)
    output.mkdir(parents=True, exist_ok=False)
    issues = coverage_issues(training, validation)
    receipt = {
        "purpose": "native-action-risk-offline-training",
        "control_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "training_config": config.model_dump(),
        "teacher_config": training[0].receipt["teacher_config"],
        "teacher_config_sha256": training[0].teacher_config_sha256,
        "split_method": "whole-route-map-holdout",
        "split_contract": SPATIAL_SPLIT_CONTRACT,
        "training_groups": sorted(set().union(*(d.groups for d in training))),
        "validation_groups": sorted(set().union(*(d.groups for d in validation))),
        "dataset_receipts_sha256": {
            "training": [d.receipt_sha256 for d in training],
            "validation": [d.receipt_sha256 for d in validation],
        },
        "coverage": {
            "training": risk_class_coverage(training),
            "validation": risk_class_coverage(validation),
        },
        "optimized": False,
        "offline_validation_passed": False,
        "qualified_for_flight": False,
        "not_a_calibrated_collision_probability": True,
        "independent_physical_validation_required": True,
        "issue_codes": issues,
    }
    if not issues:
        train_samples = [sample for d in training for sample in d.samples]
        validation_samples = [sample for d in validation for sample in d.samples]
        model, train_metrics = train_local_risk_critic(
            [detach_evidence(row) for row in train_samples], config.model_copy(deep=True)
        )
        validation_metrics = evaluate_local_risk_critic(
            model, [detach_evidence(row) for row in validation_samples]
        )
        train_metrics = LocalRiskCriticMetrics.model_validate(
            train_metrics.model_dump(), strict=True
        )
        validation_metrics = LocalRiskCriticMetrics.model_validate(
            validation_metrics.model_dump(), strict=True
        )
        for metrics, rows in (
            (train_metrics, train_samples),
            (validation_metrics, validation_samples),
        ):
            risky_count = sum(row.risk_target >= 0.5 for row in rows)
            if (
                metrics.sample_count != len(rows)
                or metrics.risky_sample_count != risky_count
                or metrics.safe_sample_count != len(rows) - risky_count
            ):
                raise ValueError("ACTION_RISK_METRICS_SAMPLE_COUNTS_DIFFER")
        predictions, _ = _predict(model, validation_samples)
        discrimination = action_discrimination(validation, predictions)
        receipt.update(
            optimized=True,
            training_metrics=train_metrics.model_dump(),
            validation_metrics=validation_metrics.model_dump(),
            action_discrimination=discrimination,
        )
        for split, metrics in (("training", train_metrics), ("validation", validation_metrics)):
            if metrics.risk_hold_recall < 0.95:
                issues.append(f"ACTION_RISK_{split.upper()}_UNSAFE_RECALL_BELOW_THRESHOLD")
            if metrics.safe_motion_recall < 0.95:
                issues.append(f"ACTION_RISK_{split.upper()}_SAFE_RECALL_BELOW_THRESHOLD")
            if metrics.risk_mean_absolute_error > 0.2:
                issues.append(f"ACTION_RISK_{split.upper()}_ERROR_ABOVE_THRESHOLD")
        if discrimination["correct_fraction"] is None or discrimination["correct_fraction"] < 0.95:
            issues.append("ACTION_RISK_VALIDATION_ACTION_DISCRIMINATION_BELOW_THRESHOLD")
        path = output / "risk-critic.onnx"
        export_local_risk_critic_onnx(model, path)
        content = read_plugin_file(path, limit=16 * 1024 * 1024)
        equivalence = verify_risk_export(model, content, train_samples + validation_samples)
        receipt.update(
            model_sha256=hashlib.sha256(content).hexdigest(),
            model_bytes=len(content),
            onnx_maximum_absolute_error=equivalence,
            learned_parameter_count=sum(
                v.size
                for v in (model.input_weight, model.input_bias, model.risk_weight, model.risk_bias)
            ),
            offline_validation_passed=not issues,
        )
    write_rows(output / "training-receipt.jsonl", [receipt])
    return receipt


# 功能：
#   使用实际 CPU ONNX 会话分批核对参考输出，拒绝非有限值、形状错误和超过容差的结果。
# 输入：
#   model：待核对的动作条件风险网络。
#   content：已经从同一文件有界读取的 ONNX 原始字节。
#   samples：非空且有界的保留样本。
# 输出：
#   maximum：所有批次中的最大绝对误差。
def verify_risk_export(model, content: bytes, samples):
    import onnxruntime as ort

    if type(content) is not bytes or not 0 < len(content) <= 16 * 1024 * 1024:
        raise ValueError("ACTION_RISK_EXPORT_BYTES_INVALID")
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= 250_000:
        raise ValueError("ACTION_RISK_EXPORT_EQUIVALENCE_FAILED")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        content, sess_options=options, providers=["CPUExecutionProvider"]
    )
    maximum = 0.0
    for start in range(0, len(samples), 256):
        expected, feeds = _predict(model, samples[start : start + 256])
        raw = session.run(["risk_score"], feeds)[0]
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (len(expected), 1)
            or raw.dtype.kind != "f"
            or np.any(raw < 0)
            or np.any(raw > 1)
        ):
            raise ValueError("ACTION_RISK_EXPORT_OUTPUT_INVALID")
        actual = raw.reshape(-1)
        if (
            actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
        ):
            raise ValueError("ACTION_RISK_EXPORT_OUTPUT_INVALID")
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    if not samples or maximum > 1e-5:
        raise ValueError("ACTION_RISK_EXPORT_EQUIVALENCE_FAILED")
    return maximum
