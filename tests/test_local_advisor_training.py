from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from test_mission_groups import split_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    LocalAdvisorTrainingMetrics,
    LocalAdvisorTrainingSample,
    evaluate_local_advisor,
    export_local_advisor_onnx,
    train_local_advisor,
)
from dronedream_agent_core.training.advisor_sources import RecordedSource
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from scripts.train_local_advisors import (
    _validate_class_coverage,
    _validate_dataset_receipt,
)


# 功能：
#   对小型测试夹具文件计算原始摘要，绑定下面构造的数据集回执。
# 输入：
#   path：测试文件路径。
# 输出：
#   checksum：文件字节的 SHA-256。
def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksum


# 功能：
#   验证训练入口重新检查负载来源的独立任务数量，单个任务不能充当多任务验证。
# 输入：
#   tmp_path：隔离的源数据和回执目录。
# 输出：
#   None：不返回业务数据。
def test_payload_training_rechecks_independent_mission_diversity(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("{}\n", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
        "split_method": SPATIAL_SPLIT_CONTRACT,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "training_groups": [split_fixture().group_sha256],
        "validation_groups": [split_fixture(y=2.0).group_sha256],
        "verified_sources_required": True,
        "verified_payload_collection_sources_allowed": True,
        "minimum_payload_collection_missions_per_split": 2,
        "roles": ["payload-dynamics-adapter"],
        "training_output_sha256": _file_sha256(training),
        "validation_output_sha256": _file_sha256(validation),
        "training_sources": [
            {
                "mission_evidence_sha256": "a" * 64,
                "semantic_sha256": "a" * 64,
                "mission_split": split_fixture().model_dump(),
            }
        ],
        "validation_sources": [
            {
                "mission_evidence_sha256": "c" * 64,
                "semantic_sha256": "a" * 64,
                "mission_split": split_fixture(y=2.0).model_dump(),
            }
        ],
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="training split has too few missions"):
        _validate_dataset_receipt(
            RecordedSource.read(receipt_path),
            training_data=RecordedSource.read(training),
            validation_data=RecordedSource.read(validation),
            roles=("payload-dynamics-adapter",),
        )

    receipt["training_sources"].append(
        {**receipt["training_sources"][0], "mission_evidence_sha256": "b" * 64}
    )
    receipt["validation_sources"].append(
        {**receipt["validation_sources"][0], "mission_evidence_sha256": "d" * 64}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _, training_sources, validation_sources = _validate_dataset_receipt(
        RecordedSource.read(receipt_path),
        training_data=RecordedSource.read(training),
        validation_data=RecordedSource.read(validation),
        roles=("payload-dynamics-adapter",),
    )
    assert len(training_sources) == 2
    assert len(validation_sources) == 2


# 功能：
#   构造各专家维度匹配的合成样本，以可分离的特征验证离线优化和导出链路。
# 输入：
#   role：待训练的专家角色。
#   risky：是否生成危险类别。
# 输出：
#   sample：带对应风险标签及负载尺度标签的样本。
def _sample(*, role: str, risky: bool) -> LocalAdvisorTrainingSample:
    state = [0.0] * 46
    state[6] = 0.0 if risky else 1.0
    history = [list(state) for _ in range(8)]
    payload = [0.0] * 24
    payload[5] = 1.0
    payload[6] = 1.6 if risky else 0.2
    payload_history = [list(payload) for _ in range(8)]
    sensor = [0.0] * 64
    sensor[0] = 0.0 if risky else 1.0
    sensor[60] = 0.0 if risky else 1.0
    sample = LocalAdvisorTrainingSample(
        role=role,
        state_features=state,
        state_history=history,
        history_mask=[1.0] * 8,
        maneuver_features=[1.0 if risky else 0.0] + [0.0] * 12,
        payload_features=payload,
        payload_history=payload_history if role == "payload-dynamics-adapter" else None,
        sensor_features=sensor,
        risk_target=1.0 if risky else 0.0,
        controller_step_scale_target=(0.3 if role == "payload-dynamics-adapter" and risky else 1.0),
    )
    return sample


# 功能：
#   保持当前负载输入相同，仅改变历史，验证真实训练和 ONNX 推理使用了时间信息。
# 输入：
#   tmp_path：仅保存本测试模型的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_payload_advisor_uses_history_not_only_the_current_sample(tmp_path: Path) -> None:
    current_payload = [0.0] * 24
    current_payload[5] = 1.0
    current_payload[6] = 0.4

    # 功能：
    #   构造稳定与增长振荡两种负载历史，并保留完全相同的当前特征。
    # 输入：
    #   risky：选择增长风险历史或稳定历史。
    # 输出：
    #   sample：供训练和推理共用的时间样本。
    def temporal_sample(*, risky: bool) -> LocalAdvisorTrainingSample:
        payload_history = []
        for index in range(8):
            row = list(current_payload)
            row[6] = 0.3 + 0.3 * index if risky else 0.4
            row[7] = 0.2 if risky else 0.0
            payload_history.append(row)
        sample = LocalAdvisorTrainingSample(
            role="payload-dynamics-adapter",
            state_features=[0.0] * 46,
            state_history=[[0.0] * 46 for _ in range(8)],
            history_mask=[1.0] * 8,
            maneuver_features=[0.0] * 13,
            payload_features=list(current_payload),
            payload_history=payload_history,
            sensor_features=[0.0] * 64,
            risk_target=1.0 if risky else 0.0,
            controller_step_scale_target=0.25 if risky else 0.9,
        )
        return sample

    samples = [temporal_sample(risky=index % 2 == 1) for index in range(128)]
    model, metrics = train_local_advisor(
        samples,
        LocalAdvisorTrainingConfig(
            hidden_feature_count=16,
            epoch_count=150,
            batch_size=16,
            learning_rate=0.01,
            random_seed=919,
        ),
        role="payload-dynamics-adapter",
    )
    assert metrics.risk_hold_recall == 1.0
    assert metrics.safe_motion_recall == 1.0
    model_path = tmp_path / "payload-dynamics-adapter.onnx"
    export_local_advisor_onnx(model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    # 功能：
    #   使用实际 CPU ONNX 会话推理一个负载样本，取出风险和控制尺度标量。
    # 输入：
    #   sample：当前负载、历史及掩码组成的样本。
    # 输出：
    #   risk_score：预测风险。
    #   step_scale：预测控制尺度。
    def infer(sample: LocalAdvisorTrainingSample) -> tuple[float, float]:
        risk, scale = session.run(
            ["risk_score", "controller_step_scale"],
            {
                "payload_features": np.asarray([sample.payload_features], dtype=np.float32),
                "maneuver_features": np.asarray([sample.maneuver_features], dtype=np.float32),
                "state_history": np.asarray([sample.state_history], dtype=np.float32),
                "payload_history": np.asarray([sample.payload_history], dtype=np.float32),
                "history_mask": np.asarray([sample.history_mask], dtype=np.float32),
            },
        )
        risk_score, step_scale = float(risk[0, 0]), float(scale[0, 0])
        return risk_score, step_scale

    safe_risk, safe_scale = infer(temporal_sample(risky=False))
    risky_risk, risky_scale = infer(temporal_sample(risky=True))

    # Current payload_features are byte-for-byte identical; only history differs.
    assert risky_risk - safe_risk > 0.5
    assert safe_scale - risky_scale > 0.3


# 功能：
#   验证高召回率不能掩盖类别样本不足，训练与验证分别执行相同覆盖要求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_training_and_validation_each_require_both_payload_classes() -> None:
    insufficient = LocalAdvisorTrainingMetrics(
        sample_count=40,
        risky_sample_count=19,
        safe_sample_count=21,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.0,
        controller_step_scale_mean_absolute_error=0.0,
    )

    with pytest.raises(RuntimeError, match="TRAINING_RISKY_CLASS"):
        _validate_class_coverage(
            "payload-dynamics-adapter",
            insufficient,
            split="training",
        )
    with pytest.raises(RuntimeError, match="VALIDATION_RISKY_CLASS"):
        _validate_class_coverage(
            "payload-dynamics-adapter",
            insufficient,
            split="validation",
        )


# 功能：
#   确认非有限负载特征在样本契约入口被拒绝，不能进入模型优化。
# 输入：
#   invalid：NaN 或正负无穷大。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_advisor_training_sample_rejects_non_finite_features(invalid: float) -> None:
    payload = [0.0] * 24
    payload[0] = invalid

    with pytest.raises(ValueError, match="finite number"):
        LocalAdvisorTrainingSample(
            role="payload-dynamics-adapter",
            state_features=[0.0] * 46,
            state_history=[[0.0] * 46 for _ in range(8)],
            history_mask=[1.0] * 8,
            maneuver_features=[0.0] * 13,
            payload_features=payload,
            payload_history=[[0.0] * 24 for _ in range(8)],
            sensor_features=[0.0] * 64,
            risk_target=0.0,
        )


# 功能：
#   训练五种独立专家并逐一验证真实 CPU ONNX 输入输出及风险区分能力。
# 输入：
#   tmp_path：测试模型导出目录。
# 输出：
#   None：不返回业务数据。
def test_trains_and_exports_independent_local_advisors(tmp_path: Path) -> None:
    samples = [
        _sample(role=role, risky=index % 2 == 1)
        for role in (
            "perception-health-critic",
            "settle-stability-critic",
            "payload-dynamics-adapter",
            "state-anomaly-detector",
            "cross-modal-consistency-critic",
        )
        for index in range(64)
    ]
    exported: dict[str, Path] = {}
    for index, role in enumerate(
        (
            "perception-health-critic",
            "settle-stability-critic",
            "payload-dynamics-adapter",
            "state-anomaly-detector",
            "cross-modal-consistency-critic",
        )
    ):
        model, training_metrics = train_local_advisor(
            samples,
            LocalAdvisorTrainingConfig(
                hidden_feature_count=16,
                epoch_count=80,
                batch_size=16,
                learning_rate=0.01,
                random_seed=2905 + index,
            ),
            role=role,
        )
        validation_metrics = evaluate_local_advisor(model, samples)
        assert training_metrics.risk_hold_recall == 1.0
        assert training_metrics.risky_sample_count == 32
        assert training_metrics.safe_sample_count == 32
        assert validation_metrics.safe_motion_recall == 1.0
        assert validation_metrics.risk_mean_absolute_error < 0.05
        if role == "payload-dynamics-adapter":
            assert validation_metrics.controller_step_scale_mean_absolute_error is not None
            assert validation_metrics.controller_step_scale_mean_absolute_error < 0.05
        output_path = tmp_path / f"{role}.onnx"
        assert len(export_local_advisor_onnx(model, output_path)) == 64
        exported[role] = output_path

    perception = ort.InferenceSession(
        str(exported["perception-health-critic"]),
        providers=["CPUExecutionProvider"],
    )
    safe_perception = _sample(role="perception-health-critic", risky=False)
    risky_perception = _sample(role="perception-health-critic", risky=True)
    safe_risk = perception.run(
        ["risk_score"],
        {"state_features": np.asarray([safe_perception.state_features], dtype=np.float32)},
    )[0]
    risky_risk = perception.run(
        ["risk_score"],
        {"state_features": np.asarray([risky_perception.state_features], dtype=np.float32)},
    )[0]
    assert float(safe_risk[0, 0]) < 0.5
    assert float(risky_risk[0, 0]) >= 0.5

    settle = ort.InferenceSession(
        str(exported["settle-stability-critic"]),
        providers=["CPUExecutionProvider"],
    )
    risky_settle = _sample(role="settle-stability-critic", risky=True)
    settle_risk = settle.run(
        ["risk_score"],
        {
            "maneuver_features": np.asarray([risky_settle.maneuver_features], dtype=np.float32),
            "state_history": np.asarray([risky_settle.state_history], dtype=np.float32),
            "history_mask": np.asarray([risky_settle.history_mask], dtype=np.float32),
        },
    )[0]
    assert float(settle_risk[0, 0]) >= 0.5

    payload = ort.InferenceSession(
        str(exported["payload-dynamics-adapter"]),
        providers=["CPUExecutionProvider"],
    )
    risky_payload = _sample(role="payload-dynamics-adapter", risky=True)
    payload_risk, payload_scale = payload.run(
        ["risk_score", "controller_step_scale"],
        {
            "payload_features": np.asarray([risky_payload.payload_features], dtype=np.float32),
            "maneuver_features": np.asarray([risky_payload.maneuver_features], dtype=np.float32),
            "state_history": np.asarray([risky_payload.state_history], dtype=np.float32),
            "payload_history": np.asarray([risky_payload.payload_history], dtype=np.float32),
            "history_mask": np.asarray([risky_payload.history_mask], dtype=np.float32),
        },
    )
    assert float(payload_risk[0, 0]) >= 0.5
    assert 0.1 <= float(payload_scale[0, 0]) < 0.5

    anomaly = ort.InferenceSession(
        str(exported["state-anomaly-detector"]),
        providers=["CPUExecutionProvider"],
    )
    risky_anomaly = _sample(role="state-anomaly-detector", risky=True)
    anomaly_score = anomaly.run(
        ["anomaly_score"],
        {
            "state_history": np.asarray([risky_anomaly.state_history], dtype=np.float32),
            "history_mask": np.asarray([risky_anomaly.history_mask], dtype=np.float32),
        },
    )[0]
    assert float(anomaly_score[0, 0]) >= 0.5

    cross_modal = ort.InferenceSession(
        str(exported["cross-modal-consistency-critic"]),
        providers=["CPUExecutionProvider"],
    )
    risky_cross_modal = _sample(
        role="cross-modal-consistency-critic",
        risky=True,
    )
    cross_modal_risk = cross_modal.run(
        ["risk_score"],
        {"sensor_features": np.asarray([risky_cross_modal.sensor_features], dtype=np.float32)},
    )[0]
    assert float(cross_modal_risk[0, 0]) >= 0.5
