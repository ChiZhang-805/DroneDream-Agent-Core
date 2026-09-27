"""Synthetic fixtures test frozen evaluation boundaries, not aircraft capability."""

import hashlib
import json

import pytest
from test_local_advisor_training import _sample
from test_mission_groups import split_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    export_local_advisor_onnx,
    train_local_advisor,
)
from dronedream_agent_core.training.advisor_evaluation import advisor_feeds, evaluate_frozen_advisor
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT

ROLE = "settle-stability-critic"


# 功能：
#   构造隔离合成模型及三个空间分组，供实际 ONNX 测试和损坏回执测试共用。
# 输入：
#   tmp_path：pytest 管理的临时目录。
# 输出：
#   value：评估参数、可变回执和原始样本，均不进入正式训练集。
@pytest.fixture
def frozen_case(tmp_path):
    samples = [_sample(role=ROLE, risky=bool(i % 2)) for i in range(40)]
    model, _ = train_local_advisor(samples, LocalAdvisorTrainingConfig(
        epoch_count=300, settle_motion_only=True), role=ROLE)
    args = {k: tmp_path / v for k, v in {
        "artifact": "frozen.onnx", "training_receipt": "training.json",
        "dataset_receipt": "dataset.json", "test_data": "test.jsonl"}.items()}
    export_local_advisor_onnx(model, args["artifact"])
    args["test_data"].write_text(
        "".join(s.model_dump_json() + "\n" for s in samples), encoding="utf-8")
    split = split_fixture(y=10.0)
    training = {
        "schema_version": "dronedream.local-advisor-training-receipt.v1",
        "training_accepted": True, "qualification_granted": False, "issue_codes": [],
        "verified_sources_required": True,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "dataset_split_method": SPATIAL_SPLIT_CONTRACT,
        "training_groups": [split_fixture().group_sha256],
        "validation_groups": [split_fixture(y=2.0).group_sha256],
        "training_data_sha256": "a" * 64, "validation_data_sha256": "b" * 64,
        "advisors": {ROLE: {"accepted": True, "artifact_sha256": hashlib.sha256(
            args["artifact"].read_bytes()).hexdigest()}},
    }
    dataset = {
        "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
        "verified_sources_required": True, "qualification_granted": False,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "split_method": SPATIAL_SPLIT_CONTRACT,
        "validation_output_sha256": hashlib.sha256(args["test_data"].read_bytes()).hexdigest(),
        "validation_groups": [split.group_sha256], "validation_sources": [{
            "mission_split": split.model_dump(), "semantic_sha256": split.semantic_sha256}],
        "roles": [ROLE], "validation_sample_counts": {ROLE: 40},
        "validation_class_counts": {ROLE: {"safe": 20, "risky": 20}},
    }
    value = (args, training, dataset, samples)
    return value


# 功能：
#   保存合成回执并执行真实冻结模型评估，不训练或重写模型。
# 输入：
#   case：测试夹具的路径、回执和样本。
# 输出：
#   result：独立评估函数返回的报告。
def evaluate_case(case):
    args, training, dataset, _ = case
    args["training_receipt"].write_text(json.dumps(training), encoding="utf-8")
    args["dataset_receipt"].write_text(json.dumps(dataset), encoding="utf-8")
    result = evaluate_frozen_advisor(role=ROLE, **args)
    return result


# 功能：
#   检验真实 ONNX 分批输出及独立报告，不把合成测试通过升级为飞行授权。
# 输入：
#   frozen_case：隔离的完整合成夹具。
# 输出：
#   None：两类识别、冻结权重和无飞行授权通过断言。
def test_frozen_onnx_evaluation(frozen_case):
    original = frozen_case[0]["artifact"].read_bytes()
    result = evaluate_case(frozen_case)
    assert result["accepted"] is True, result
    assert result["optimized"] is result["qualified_for_flight"] is False
    assert result["metrics"]["risk_hold_recall"] == 1.0
    assert result["metrics"]["safe_motion_recall"] == 1.0
    assert original == frozen_case[0]["artifact"].read_bytes()


# 功能：
#   对另外四种专家实际导出和执行 ONNX，核验专用输出名及负载双输出，不改名迁就评估器。
# 输入：
#   frozen_case：隔离回执模板；role：待核验的真实导出角色。
# 输出：
#   None：每种接口都生成冻结评估报告，原模型字节保持不变。
@pytest.mark.parametrize("role", ["perception-health-critic", "state-anomaly-detector",
    "cross-modal-consistency-critic", "payload-dynamics-adapter"])
def test_frozen_evaluation_all_advisor_interfaces(frozen_case, role):
    args, training, dataset, _ = frozen_case
    samples = [_sample(role=role, risky=bool(i % 2)) for i in range(40)]
    model, _ = train_local_advisor(samples, LocalAdvisorTrainingConfig(epoch_count=100), role=role)
    args["artifact"] = args["artifact"].with_name(role + ".onnx")
    export_local_advisor_onnx(model, args["artifact"])
    original = args["artifact"].read_bytes()
    args["test_data"].write_text("".join(s.model_dump_json() + "\n" for s in samples),
                                  encoding="utf-8")
    training["advisors"] = {role: {"accepted": True,
        "artifact_sha256": hashlib.sha256(original).hexdigest()}}
    dataset.update(roles=[role], validation_sample_counts={role: 40},
        validation_class_counts={role: {"safe": 20, "risky": 20}},
        validation_output_sha256=hashlib.sha256(args["test_data"].read_bytes()).hexdigest())
    args["training_receipt"].write_text(json.dumps(training), encoding="utf-8")
    args["dataset_receipt"].write_text(json.dumps(dataset), encoding="utf-8")
    result = evaluate_frozen_advisor(role=role, **args)
    assert result["metrics"]["sample_count"] == 40
    assert result["role"] == role
    assert result["optimized"] is result["qualified_for_flight"] is False
    assert ("controller_step_scale_mean_absolute_error" in result["metrics"]) == (
        role == "payload-dynamics-adapter")
    assert args["artifact"].read_bytes() == original


# 功能：
#   拒绝测试复用调参分组、模型身份错绑、未验证来源和损坏计数等证据。
# 输入：
#   frozen_case：有效合成夹具；damage：要破坏的边界。
# 输出：
#   None：每个错误必须显式拒绝，不能出具通过报告。
@pytest.mark.parametrize("damage", ["overlap", "artifact", "unverified", "advisors",
                                    "groups", "counts", "classes", "training_overlap"])
def test_frozen_evidence_rejected(frozen_case, damage):
    _, training, dataset, _ = frozen_case
    if damage == "overlap":
        training["validation_groups"] = dataset["validation_groups"]
    elif damage == "artifact":
        training["advisors"][ROLE]["artifact_sha256"] = "0" * 64
    elif damage == "unverified":
        training["verified_sources_required"] = False
    elif damage == "advisors":
        training["advisors"] = None
    elif damage == "groups":
        dataset["validation_groups"] = [None]
    elif damage == "counts":
        dataset["validation_sample_counts"] = None
    elif damage == "classes":
        dataset["validation_class_counts"] = None
    else:
        training["validation_groups"] = training["training_groups"]
    with pytest.raises(ValueError, match="ADVISOR_FROZEN_"):
        evaluate_case(frozen_case)


# 功能：
#   拒绝空批次、过大批次及角色与张量不相符的输入。
# 输入：
#   无。
# 输出：
#   None：无效请求均被阻断。
def test_frozen_feeds_boundaries():
    sample = _sample(role=ROLE, risky=False)
    for samples in ([], [sample] * 129):
        with pytest.raises(ValueError, match="BATCH_INVALID"):
            advisor_feeds(ROLE, samples)
    with pytest.raises(ValueError, match="ROLE_MISMATCH"):
        advisor_feeds("perception-health-critic", [sample])
