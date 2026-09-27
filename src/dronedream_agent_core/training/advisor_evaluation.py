"""Frozen ONNX advisor evaluation on a new spatial holdout, without optimization."""

import hashlib
from pathlib import Path
from typing import get_args

from dronedream_plugin_sdk.protocol import decode_json

from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..local_advisor_training import LocalAdvisorTrainingSample, TrainableAdvisorRole
from ..plugin_files import read_plugin_file
from .advisor_sources import RecordedSource, advisor_spatial_groups
from .evidence_files import decode_evidence_rows
from .mission_groups import SPATIAL_SPLIT_CONTRACT


# 功能：
#   从明确来源冻结小型 JSON 对象及内容身份，不在解析后重新读取文件。
# 输入：
#   path：原始训练或数据集回执路径。
# 输出：
#   value：严格解析的对象；digest：相同原始字节的摘要。
def _receipt(path: Path):
    content = read_plugin_file(path.absolute(), limit=4 * 1024**2)
    value = decode_json(content, limit=4 * 1024**2)
    if type(value) is not dict:
        raise ValueError("ADVISOR_FROZEN_RECEIPT_NOT_OBJECT")
    return value, hashlib.sha256(content).hexdigest()


# 功能：
#   为明确的专家角色建立部署一致的具名张量，只按原始历史掩码传入样本。
# 输入：
#   role：待评估专家；samples：最多 128 条已验证的对应样本。
# 输出：
#   feeds：名称到有限 float32 数组的映射。
def advisor_feeds(role: str, samples: list[LocalAdvisorTrainingSample]):
    import numpy as np

    names = {
        "perception-health-critic": ("state_features",),
        "cross-modal-consistency-critic": ("sensor_features",),
        "settle-stability-critic": ("maneuver_features", "state_history", "history_mask"),
        "state-anomaly-detector": ("state_history", "history_mask"),
        "payload-dynamics-adapter": ("payload_features", "maneuver_features", "payload_history", "state_history", "history_mask"),
    }
    if role not in names or not 1 <= len(samples) <= 128:
        raise ValueError("ADVISOR_FROZEN_BATCH_INVALID")
    rows = [LocalAdvisorTrainingSample.model_validate(s.model_dump(), strict=True) for s in samples]
    if any(row.role != role for row in rows):
        raise ValueError("ADVISOR_FROZEN_ROLE_MISMATCH")
    with np.errstate(over="raise", invalid="raise"):
        feeds = {name: np.asarray([getattr(s, name) for s in rows], dtype=np.float32)
                 for name in names[role]}
    if any(not np.isfinite(value).all() for value in feeds.values()):
        raise ValueError("ADVISOR_FROZEN_NONFINITE_INPUT")
    return feeds


# 功能：
#   1. 核验冻结权重及当前训练来源，拒绝与训练或调参分区重叠的新测试集。
#   2. 通过真实 CPU ONNX 运行逐批评估，不运行优化器，不授予飞行资格。
# 输入：
#   role：专家角色；artifact：冻结 ONNX；training_receipt：训练与权重绑定回执。
#   dataset_receipt：含 validation_sources 的新测试数据回执；test_data：其验证侧样本。
# 输出：
#   result：绑定全部输入摘要、测试组、实测预测指标及失败原因的独立报告。
def evaluate_frozen_advisor(*, role: str, artifact: Path, training_receipt: Path,
                            dataset_receipt: Path, test_data: Path):
    import numpy as np
    import onnxruntime as ort

    from .artifact_assembly import validate_embedded_graph

    if role not in get_args(TrainableAdvisorRole):
        raise ValueError("ADVISOR_FROZEN_ROLE_INVALID")
    training, training_sha = _receipt(training_receipt)
    dataset, dataset_sha = _receipt(dataset_receipt)
    content = read_plugin_file(artifact.absolute(), limit=4 * 1024**2)
    artifact_sha = hashlib.sha256(content).hexdigest()
    advisors = training.get("advisors")
    advisor = advisors.get(role) if type(advisors) is dict else None
    if type(advisor) is not dict:
        raise ValueError("ADVISOR_FROZEN_TRAINING_BINDING_INVALID")
    if (training.get("schema_version") != "dronedream.local-advisor-training-receipt.v1"
            or training.get("training_accepted") is not True
            or training.get("qualification_granted") is not False
            or training.get("issue_codes") != []
            or training.get("verified_sources_required") is not True
            or advisor.get("accepted") is not True
            or advisor.get("artifact_sha256") != artifact_sha
            or training.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or training.get("dataset_split_method") != SPATIAL_SPLIT_CONTRACT):
        raise ValueError("ADVISOR_FROZEN_TRAINING_BINDING_INVALID")
    source = RecordedSource.read(test_data)
    source_sha = hashlib.sha256(source.content).hexdigest()
    if (dataset.get("schema_version") != "dronedream.local-advisor-dataset-receipt.v1"
            or dataset.get("verified_sources_required") is not True
            or dataset.get("qualification_granted") is not False
            or dataset.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or dataset.get("split_method") != SPATIAL_SPLIT_CONTRACT
            or dataset.get("validation_output_sha256") != source_sha
            or source_sha in (training.get("training_data_sha256"),
                              training.get("validation_data_sha256"))):
        raise ValueError("ADVISOR_FROZEN_TEST_BINDING_INVALID")
    groups = advisor_spatial_groups(dataset.get("validation_sources"))
    declared_groups = dataset.get("validation_groups")
    if (type(declared_groups) is not list
            or any(type(g) is not str for g in declared_groups)
            or len(declared_groups) != len(set(declared_groups))
            or groups != set(declared_groups)):
        raise ValueError("ADVISOR_FROZEN_TEST_GROUP_BINDING_INVALID")
    seen_groups = set()
    for split in ("training", "validation"):
        declared = training.get(split + "_groups")
        if (type(declared) is not list or not declared
                or any(type(g) is not str or len(g) != 64
                       or set(g) - set("0123456789abcdef") for g in declared)):
            raise ValueError("ADVISOR_FROZEN_TRAINING_GROUPS_INVALID")
        if len(set(declared)) != len(declared) or seen_groups & set(declared):
            raise ValueError("ADVISOR_FROZEN_TRAINING_GROUPS_INVALID")
        seen_groups.update(declared)
    if groups & seen_groups:
        raise ValueError("ADVISOR_FROZEN_TEST_OVERLAPS_TRAINING_OR_TUNING")
    rows = [LocalAdvisorTrainingSample.model_validate(r, strict=True)
            for r in decode_evidence_rows(source.content)]
    samples = [r for r in rows if r.role == role]
    roles = dataset.get("roles")
    sample_counts = dataset.get("validation_sample_counts")
    if (not samples or type(roles) is not list or role not in roles
            or type(sample_counts) is not dict
            or type(sample_counts.get(role)) is not int or len(samples) != sample_counts[role]):
        raise ValueError("ADVISOR_FROZEN_TEST_COUNTS_INVALID")
    primary_output = "anomaly_score" if role == "state-anomaly-detector" else "risk_score"
    outputs = [primary_output] + (["controller_step_scale"]
                                if role == "payload-dynamics-adapter" else [])
    feeds = advisor_feeds(role, samples[:128])
    validate_embedded_graph(content, input_names=list(feeds), output_names=outputs)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(content, sess_options=options,
                                  providers=["CPUExecutionProvider"])
    predictions = []
    for start in range(0, len(samples), 128):
        batch = samples[start:start + 128]
        values = session.run(outputs, advisor_feeds(role, batch))
        if any(v.shape != (len(batch), 1) or not np.isfinite(v).all()
               or (v < (0.1 if name == "controller_step_scale" else 0.0)).any()
               or (v > 1.0).any() for name, v in zip(outputs, values, strict=True)):
            raise ValueError("ADVISOR_FROZEN_OUTPUT_CONTRACT_INVALID")
        predictions.append(np.concatenate(values, axis=1))
    predictions = np.concatenate(predictions, axis=0)
    targets = np.asarray([s.risk_target for s in samples])
    risky = targets >= .5
    decision = predictions[:, 0] >= .5
    counts = {"risky": int(risky.sum()), "safe": int((~risky).sum())}
    class_counts = dataset.get("validation_class_counts")
    declared_counts = class_counts.get(role) if type(class_counts) is dict else None
    if (type(declared_counts) is not dict
            or any(type(v) is not int for v in declared_counts.values())
            or counts != declared_counts):
        raise ValueError("ADVISOR_FROZEN_TEST_CLASS_COUNTS_INVALID")
    metrics = {
        "sample_count": len(samples), **counts,
        "risk_hold_recall": float(decision[risky].mean()) if risky.any() else None,
        "safe_motion_recall": float((~decision[~risky]).mean()) if (~risky).any() else None,
        "risk_mean_absolute_error": float(np.abs(predictions[:, 0] - targets).mean()),
    }
    issues = []
    if min(counts.values()) < 20:
        issues.append("ADVISOR_FROZEN_TEST_CLASS_COVERAGE_TOO_LOW")
    if metrics["risk_hold_recall"] is None or metrics["risk_hold_recall"] < .95:
        issues.append("ADVISOR_FROZEN_TEST_RISK_RECALL_TOO_LOW")
    if metrics["safe_motion_recall"] is None or metrics["safe_motion_recall"] < .95:
        issues.append("ADVISOR_FROZEN_TEST_SAFE_RECALL_TOO_LOW")
    if metrics["risk_mean_absolute_error"] > .2:
        issues.append("ADVISOR_FROZEN_TEST_RISK_ERROR_TOO_HIGH")
    if role == "payload-dynamics-adapter":
        error = float(np.abs(predictions[:, 1] - np.asarray(
            [s.controller_step_scale_target for s in samples])).mean())
        metrics["controller_step_scale_mean_absolute_error"] = error
        if error > .15:
            issues.append("ADVISOR_FROZEN_TEST_SCALE_ERROR_TOO_HIGH")
    result = {
        "purpose": "frozen-local-advisor-independent-spatial-test", "role": role,
        "artifact_sha256": artifact_sha, "training_receipt_sha256": training_sha,
        "dataset_receipt_sha256": dataset_sha, "test_data_sha256": source_sha,
        "test_groups": sorted(groups), "metrics": metrics, "issue_codes": issues,
        "accepted": not issues, "optimized": False, "qualified_for_flight": False,
    }
    return result
