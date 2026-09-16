from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    LocalAdvisorTrainingSample,
    export_local_advisor_onnx,
    train_local_advisor,
)
from dronedream_agent_core.local_policy_composition import (
    compose_local_policy_advisors,
    load_local_advisor_artifact_evidence,
)
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingSample,
    train_local_policy,
    write_local_policy_package,
)
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT


# 功能：
#   构造固定候选选择的合成监督，仅用于检查模型组合而非飞行能力。
# 输入：
#   index：改变候选特征位置的样本序号。
# 输出：
#   sample：有效候选及安全风险标签组成的样本。
def _navigation_sample(index: int) -> LocalPolicyTrainingSample:
    candidates = [[0.0] * 15 for _ in range(8)]
    candidates[0][index % 15] = 1.0
    sample = LocalPolicyTrainingSample(
        state_features=[0.0] * 46,
        candidate_features=candidates,
        candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_action_index=0,
        risk_target=0.0,
    )
    return sample


# 功能：
#   实际训练两轮小型前馈网络并导出可读基座，测试指标不授予资格。
# 输入：
#   path：尚不存在的基座测试目录。
# 输出：
#   package：实际读回的合成测试模型包。
def _write_base_package(path: Path) -> object:
    model, _metrics = train_local_policy(
        [_navigation_sample(index) for index in range(8)],
        LocalPolicyTrainingConfig(
            hidden_feature_count=16,
            epoch_count=2,
            batch_size=8,
            random_seed=41,
        ),
    )
    write_local_policy_package(
        output_root=path,
        model=model,
        package_id="test.composition.base",
        display_name="Composition Base",
        scope="general",
        vehicle_sha256="a" * 64,
        sensor_contract_sha256="b" * 64,
        maximum_inference_latency_ms=250,
    )
    package = load_local_policy_package(path)
    return package


# 功能：
#   生成风险与状态特征关联的异常检测训练样本，用于测试导出和组合链路。
# 输入：
#   risky：当前合成样本是否属于危险类别。
# 输出：
#   sample：带八帧状态历史的顾问监督样本。
def _advisor_sample(*, risky: bool) -> LocalAdvisorTrainingSample:
    state = [0.0] * 46
    state[0] = 1.0 if risky else 0.0
    sample = LocalAdvisorTrainingSample(
        role="state-anomaly-detector",
        state_features=state,
        state_history=[list(state) for _ in range(8)],
        history_mask=[1.0] * 8,
        maneuver_features=[0.0] * 13,
        payload_features=[0.0] * 24,
        sensor_features=[0.0] * 64,
        risk_target=1.0 if risky else 0.0,
    )
    return sample


# 功能：
#   实际训练并导出小型顾问，同时建立明确合成的来源回执，不伪装成真实任务验收。
# 输入：
#   root：隔离的测试模型目录。
# 输出：
#   artifact：实际导出的 ONNX 文件路径。
#   receipt：绑定该模型摘要的测试训练回执路径。
def _write_advisor_and_receipt(root: Path) -> tuple[Path, Path]:
    model, _metrics = train_local_advisor(
        [_advisor_sample(risky=index % 2 == 0) for index in range(40)],
        LocalAdvisorTrainingConfig(
            hidden_feature_count=8,
            epoch_count=10,
            batch_size=8,
            learning_rate=0.01,
            random_seed=42,
        ),
        role="state-anomaly-detector",
    )
    artifact = root / "state-anomaly-detector.onnx"
    digest = export_local_advisor_onnx(model, artifact)
    receipt = root / "training-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-advisor-training-receipt.v1",
                "dataset_split_method": SPATIAL_SPLIT_CONTRACT,
                "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
                "training_groups": ["a" * 64],
                "validation_groups": ["b" * 64],
                "training_data_sha256": "c" * 64,
                "validation_data_sha256": "d" * 64,
                "validation_used_for_configuration_selection": False,
                "fresh_admission_validation_required": True,
                "advisors": {
                    "state-anomaly-detector": {
                        "artifact": artifact.name,
                        "artifact_sha256": digest,
                        "accepted": True,
                    }
                },
                "training_accepted": True,
                "issue_codes": [],
                "qualification_granted": False,
            }
        ),
        encoding="utf-8",
    )
    return artifact, receipt


# 功能：
#   实际组装、加载并核对新顾问与全部未替换权重，回执不得继承飞行资格。
# 输入：
#   tmp_path：基座、顾问与候选输出的测试目录。
# 输出：
#   None：不返回业务数据。
def test_composition_preserves_base_and_binds_advisor_training(tmp_path: Path) -> None:
    base = _write_base_package(tmp_path / "base")
    artifact, training_receipt = _write_advisor_and_receipt(tmp_path)
    evidence = load_local_advisor_artifact_evidence(
        role="state-anomaly-detector",
        artifact_path=artifact,
        training_receipt_path=training_receipt,
    )
    receipt_path = tmp_path / "composition-receipt.json"
    composed = compose_local_policy_advisors(
        base_package_path=base.root,
        output_package_path=tmp_path / "composed",
        composition_receipt_path=receipt_path,
        package_id="test.composition.with-anomaly",
        display_name="Composition with Anomaly",
        additions=(evidence,),
    )

    assert set(composed.artifact_paths) == {
        "local-navigation-policy",
        "risk-critic",
        "state-anomaly-detector",
    }
    base_hashes = {item.role: item.sha256 for item in base.manifest.artifacts}
    composed_hashes = {item.role: item.sha256 for item in composed.manifest.artifacts}
    assert composed_hashes["local-navigation-policy"] == base_hashes["local-navigation-policy"]
    assert composed_hashes["risk-critic"] == base_hashes["risk-critic"]
    assert (
        composed_hashes["state-anomaly-detector"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_package_sha256"] == base.package_sha256
    assert receipt["package_sha256"] == composed.package_sha256
    assert receipt["preserved_artifact_sha256"] == {
        "local-navigation-policy": base_hashes["local-navigation-policy"],
        "risk-critic": base_hashes["risk-critic"],
    }
    assert receipt["fresh_admission_validation_required"] is True
    assert receipt["qualification_granted"] is False


# 功能：
#   顾问模型字节与训练摘要不符时必须拒绝，不能靠相同文件名通过。
# 输入：
#   tmp_path：测试输入目录。
# 输出：
#   None：不返回业务数据。
def test_composition_rejects_tampered_advisor_artifact(tmp_path: Path) -> None:
    artifact, training_receipt = _write_advisor_and_receipt(tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash differs"):
        load_local_advisor_artifact_evidence(
            role="state-anomaly-detector",
            artifact_path=artifact,
            training_receipt_path=training_receipt,
        )


# 功能：
#   缺少受支持回执契约的历史文件不能作为新顾问组合来源。
# 输入：
#   tmp_path：模型和错误回执的测试目录。
# 输出：
#   None：不返回业务数据。
def test_composition_rejects_legacy_unbound_training_receipt(tmp_path: Path) -> None:
    artifact, training_receipt = _write_advisor_and_receipt(tmp_path)
    payload = json.loads(training_receipt.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    training_receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is invalid"):
        load_local_advisor_artifact_evidence(
            role="state-anomaly-detector",
            artifact_path=artifact,
            training_receipt_path=training_receipt,
        )
