"""Admission metric and lineage boundaries, with explicitly synthetic inference."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_advisor_numeric_boundaries import model as advisor_model
from test_local_advisor_training import _sample
from test_local_policy_admission_dataset import _training_receipt
from test_local_policy_runtime_staging import _package
from test_mission_groups import split_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import export_local_advisor_onnx
from dronedream_agent_core.local_policy_port import LocalPolicyRawInference
from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from scripts import evaluate_local_policy_offline as evaluation


# 功能：
#   建立候选导航文件测试包、单条合法监督样本及显式固定输出的后端。
# 输入：
#   tmp_path：仅用于本测试的包目录。
# 输出：
#   case：结构真实但不执行产品模型的评估上下文。
@pytest.fixture
def runtime_case(tmp_path):
    package = _package(tmp_path / "package")
    sample = LocalPolicyTrainingSample(
        state_features=[0.0] * 46,
        candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[1.0] + [0.0] * 7,
        target_action_index=0,
        risk_target=0.0,
        visual_features=[0.0] * 139,
    )
    output = LocalPolicyRawInference(
        candidate_scores=[10.0] + [0.0] * 7, action_scores=[0.0] * 3, risk_score=0.0
    )
    backend = Mock(motion_history_ready=Mock(return_value=True), infer=Mock(return_value=output))
    case = SimpleNamespace(package=package, sample=sample, output=output, backend=backend)
    return case


# 功能：
#   准入评估重新验证被外部改写的样本，不能把非法风险标签或风险提案当作行为监督。
# 输入：
#   runtime_case：合法基础样本和固定测试后端。
#   change：绕过字段赋值验证后的非法修改。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change", [{"candidate_mask": [0.0] * 8}, {"risk_proposed_control": [0.0] * 4}]
)
def test_runtime_evaluation_rechecks_samples(runtime_case, change):
    case = runtime_case
    sample = case.sample.model_copy(update=change)
    with pytest.raises(ValueError):
        evaluation._evaluate_runtime_samples(case.package, [sample], case.backend)
    case.backend.infer.assert_not_called()


# 功能：
#   指标计算不能信任构造后被修改的后端风险值，非法输出应在计分前拒绝。
# 输入：
#   runtime_case：固定推理上下文。
# 输出：
#   None：不返回业务数据。
def test_runtime_evaluation_rechecks_raw_output(runtime_case):
    case = runtime_case
    case.backend.infer.return_value = case.output.model_copy(update={"risk_score": -0.1})
    with pytest.raises(ValueError):
        evaluation._evaluate_runtime_samples(case.package, [case.sample], case.backend)


# 功能：
#   记录整个本地准备加推理耗时，不能把时序准备延迟排除在准入预算之外。
# 输入：
#   runtime_case：始终就绪的固定后端。
#   monkeypatch：提供可重复单调时钟的工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_latency_includes_temporal_preparation(runtime_case, monkeypatch):
    case = runtime_case
    clock = [0.0]

    # 功能：
    #   模拟准备历史消耗五毫秒，推理自身不额外消耗本测试时钟。
    # 输入：
    #   batch：本次真实构建的特征批次。
    # 输出：
    #   None：不返回业务数据。
    def prepare(batch):
        clock[0] += 0.005

    case.backend.prepare_temporal_context.side_effect = prepare
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: clock[0])
    _, latencies = evaluation._evaluate_runtime_samples(case.package, [case.sample], case.backend)
    assert latencies == pytest.approx([5.0])


# 功能：
#   原始调用失败与关闭失败同时发生时，必须保留推理原异常作为主要失败原因。
# 输入：
#   runtime_case：包加载和样本读取的测试替身来源。
#   monkeypatch：分别注入推理和关闭异常的工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_evaluation_preserves_primary_failure(runtime_case, monkeypatch):
    case = runtime_case
    monkeypatch.setattr(evaluation, "load_local_policy_package", Mock(return_value=case.package))
    monkeypatch.setattr(evaluation, "load_training_samples", Mock(return_value=[case.sample]))
    case.backend.infer.side_effect = RuntimeError("primary inference failure")
    case.backend.close.side_effect = ValueError("secondary close failure")
    monkeypatch.setattr(evaluation, "OnnxLocalPolicyBackend", Mock(return_value=case.backend))
    with pytest.raises(RuntimeError, match="primary inference failure"):
        evaluation._evaluate_runtime_package(case.package.root, case.package.root / "unused")
    case.backend.close.assert_called_once()


# 功能：
#   顾问样本读取必须拒绝重复字段和空白行，不能悄悄减少或改写验证集合。
# 输入：
#   tmp_path：合成顾问数据目录。
#   kind：非法空白行或重复风险字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["blank", "duplicate"])
def test_advisor_reader_is_strict(tmp_path, kind):
    sample = _sample(role="perception-health-critic", risky=False)
    original = (sample.model_dump_json() + "\n").encode()
    content = original + b"\n" if kind == "blank" else b'{"risk_target":1.0,' + original[1:]
    path = tmp_path / "advisors.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        evaluation._load_advisor_samples(path)


# 功能：
#   顾问标量输出不接受多元素数组、布尔值或超范围风险，不通过截取第一项掩盖错误。
# 输入：
#   tmp_path：实际导出的微型测试模型；本例显式替换推理输出，不执行产品模型。
#   monkeypatch：注入错误会话输出的工具。
#   output：后端非法风险张量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "output",
    [np.array([0.0, 1.0], dtype=np.float32), np.array([False]), np.array([-0.1], dtype=np.float32)],
)
def test_advisor_evaluation_rejects_invalid_output(tmp_path, monkeypatch, output):
    import onnxruntime as ort

    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="state_features")]
    session.run.return_value = [output]
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))
    sample = _sample(role="perception-health-critic", risky=False)
    artifact = tmp_path / "advisor.onnx"
    export_local_advisor_onnx(advisor_model(), artifact)
    with pytest.raises((ValueError, RuntimeError)):
        evaluation._evaluate_advisor_artifact(artifact, [sample], role="perception-health-critic")


# 功能：
#   六十四字符但并非十六进制的任务身份不能参与来源隔离判断。
# 输入：
#   无：构造包含非摘要任务身份的来源列表。
# 输出：
#   None：不返回业务数据。
def test_admission_sources_require_hex_identity():
    with pytest.raises(ValueError):
        evaluation._source_evidence_hashes(
            {"validation_sources": [{"mission_evidence_sha256": "z" * 64}]}, "validation"
        )


# 功能：
#   恢复引导阈值不接受 NaN 或数值字符串，防止比较失败被误判为验证通过。
# 输入：
#   tmp_path：合成引导回执目录。
#   invalid：将要替换正常精度的非法数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [float("nan"), "1.0"])
def test_bootstrap_metrics_reject_nonfinite_and_coercion(tmp_path, invalid):
    receipt = {
        "schema_version": "dronedream.local-policy-augmentation-receipt.v1",
        "bootstrap_only": True,
        "expert_role": "recovery-policy",
        "deployment_scope": "simulation-only",
        "fresh_admission_validation_required": True,
        "qualification_granted": False,
        "validation_class_counts": {"safe": 40, "risky": 20},
        "validation_metrics": {
            "motion_authorization_accuracy": invalid,
            "candidate_selection_accuracy": 1.0,
            "risk_hold_recall": 1.0,
            "safe_motion_recall": 1.0,
            "risk_mean_absolute_error": 0.0,
        },
    }
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation._navigation_admission_scope(path)


# 功能：
#   当前顾问必须使用新空间路线，不能仅用新任务名或新数据摘要绕过训练／调参路线隔离。
# 输入：
#   tmp_path：独立顾问数据与空间回执目录。
#   seen：是否把实际验证路线列入历史训练组。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seen", [False, True])
def test_current_advisor_admission_checks_real_spatial_groups(tmp_path, seen):
    data, receipt_path = tmp_path / "data.jsonl", tmp_path / "data-receipt.json"
    row = _sample(role="perception-health-critic", risky=False)
    data.write_bytes((row.model_dump_json() + "\n").encode())
    split_train, split_validation = split_fixture(y=3.0), split_fixture(y=4.0)
    receipt = {
        "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
        "split_method": SPATIAL_SPLIT_CONTRACT,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "verified_sources_required": True,
        "training_output_sha256": "e" * 64,
        "validation_output_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    for name, split, digest in (
        ("training", split_train, "8"),
        ("validation", split_validation, "9"),
    ):
        receipt[name + "_sources"] = [
            {
                "mission_evidence_sha256": digest * 64,
                "mission_split": split.model_dump(),
                "semantic_sha256": split.semantic_sha256,
            }
        ]
        receipt[name + "_groups"] = [split.group_sha256]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    training = _training_receipt()
    training.update(
        feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        training_groups=[
            split_validation.group_sha256 if seen else split_fixture(y=5.0).group_sha256
        ],
        validation_groups=[split_fixture(y=6.0).group_sha256],
    )
    if seen:
        with pytest.raises(ValueError, match="training or tuning spatial route"):
            evaluation._validate_fresh_advisor_admission_data(
                data_path=data, dataset_receipt_path=receipt_path, training_receipt=training
            )
    else:
        split, sources = evaluation._validate_fresh_advisor_admission_data(
            data_path=data, dataset_receipt_path=receipt_path, training_receipt=training
        )
        assert split == "validation" and sources == ["9" * 64]
