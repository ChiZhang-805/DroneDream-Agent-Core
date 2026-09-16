from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample


# 功能：
#   按真实脚本路径加载候选增补入口，便于直接检查命令行内部校验，不触发入口运行。
# 输入：
#   无：使用测试文件所属仓库的脚本路径。
# 输出：
#   module：已完成导入的独立脚本模块。
def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "augment_local_policy_package.py"
    spec = importlib.util.spec_from_file_location("test_local_policy_augmentation_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 功能：
#   构造可区分的候选时代监督行，危险时选择停留；此合成数据不代表实际飞行证据。
# 输入：
#   index：区分状态和候选特征的索引。
#   risky：是否赋予危险标签。
# 输出：
#   sample：精细操纵专家的合法监督样本。
def _sample(index: int, *, risky: bool) -> LocalPolicyTrainingSample:
    state = [0.0] * 46
    state[index % len(state)] = float(index + 1)
    candidates = [[0.0] * 15 for _ in range(8)]
    candidates[0][index % 15] = float(index + 1)
    sample = LocalPolicyTrainingSample(
        navigation_expert_role="precision-maneuver-policy",
        state_features=state,
        candidate_features=candidates,
        candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_action_index=8 if risky else 0,
        risk_target=1.0 if risky else 0.0,
    )
    return sample


# 功能：
#   验证最小总量达标仍须同时具有安全和危险监督，单一类别不能通过数据准入。
# 输入：
#   无：构造双类以及仅安全的监督样本。
# 输出：
#   None：不返回业务数据。
def test_specialist_dataset_requires_both_safety_classes() -> None:
    module = _module()
    samples = [_sample(index, risky=index < 10) for index in range(50)]

    selected, counts = module._select_and_validate_samples(
        samples,
        role="precision-maneuver-policy",
        split="training",
    )

    assert len(selected) == 50
    assert counts == {"safe": 40, "risky": 10}

    with pytest.raises(RuntimeError, match="RISKY_CLASS_TOO_SMALL"):
        module._select_and_validate_samples(
            [_sample(index, risky=False) for index in range(50)],
            role="precision-maneuver-policy",
            split="validation",
        )


# 功能：
#   检查视觉及已知顾问属于可保留角色，未知角色必须阻止重建而非静默丢弃。
# 输入：
#   无：显式已知角色及未知角色集合。
# 输出：
#   None：不返回业务数据。
def test_specialist_augmentation_preserves_visual_encoder_role() -> None:
    module = _module()

    assert (
        module._unsupported_artifact_roles(
            {
                "local-navigation-policy",
                "risk-critic",
                "perception-encoder",
                "state-anomaly-detector",
            }
        )
        == set()
    )
    assert module._unsupported_artifact_roles({"unknown-expert"}) == {"unknown-expert"}


# 功能：
#   检查已有兼容视觉尾段不被重复扩展，宽度不兼容时拒绝继续。
# 输入：
#   无：仅模拟视觉宽度属性的模型替身，不执行网络推理。
# 输出：
#   None：不返回业务数据。
def test_visual_risk_recalibration_reuses_an_already_visual_model() -> None:
    module = _module()
    model = type("VisualRiskModel", (), {"visual_feature_count": 139})()

    prepared = module._prepare_visual_risk_model_for_calibration(
        model,
        visual_feature_count=139,
        random_seed=41,
    )

    assert prepared is model
    with pytest.raises(ValueError, match="feature width differs"):
        module._prepare_visual_risk_model_for_calibration(
            model,
            visual_feature_count=128,
            random_seed=41,
        )


# 功能：
#   验证完整监督行跨训练与验证分区重用会被拒绝，互异行可以通过此项检查。
# 输入：
#   无：独立索引的合成样本与重复行。
# 输出：
#   None：不返回业务数据。
def test_specialist_dataset_rejects_cross_split_sample_overlap() -> None:
    module = _module()
    training = [_sample(index, risky=index < 10) for index in range(50)]
    validation = [_sample(index + 100, risky=index < 10) for index in range(50)]
    module._assert_disjoint(training, validation)

    with pytest.raises(RuntimeError, match="TRAINING_VALIDATION_OVERLAP"):
        module._assert_disjoint(training, [*validation, training[0]])


# 功能：
#   拒绝把经过多顾问干预的最终集合动作当作单专家原始监督标签。
# 输入：
#   tmp_path：数据占位文件和实际回执的测试目录。
# 输出：
#   None：不返回业务数据。
def test_specialist_training_rejects_entangled_ensemble_labels(tmp_path: Path) -> None:
    module = _module()
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("training\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "navigation_label_source": "final-ensemble-decision",
                "verified_sources_required": True,
                "expert_trace_required": False,
                "training_output_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                "validation_output_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ENTANGLED_ADVISOR_LABELS"):
        module._validate_dataset_receipt(
            receipt,
            training_path=training,
            validation_path=validation,
            role="precision-maneuver-policy",
        )


# 功能：
#   拒绝用已筛选为恢复专家的数据训练精细操纵专家，即使摘要及验证标记一致。
# 输入：
#   tmp_path：测试来源及回执目录。
# 输出：
#   None：不返回业务数据。
def test_specialist_training_rejects_different_role_filtered_dataset(tmp_path: Path) -> None:
    module = _module()
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("training\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "navigation_label_source": "navigation-expert-trace",
                "navigation_expert_role_filter": "recovery-policy",
                "verified_sources_required": True,
                "expert_trace_required": True,
                "training_output_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                "validation_output_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ROLE_FILTER_MISMATCH"):
        module._validate_dataset_receipt(
            receipt,
            training_path=training,
            validation_path=validation,
            role="precision-maneuver-policy",
        )


# 功能：
#   检查命令行保留两个训练器的类别平衡和风险间隔参数；不以源码存在替代执行测试。
# 输入：
#   无：读取实际增补入口源码。
# 输出：
#   None：不返回业务数据。
def test_augmentation_cli_exposes_risk_class_weight_configuration() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "augment_local_policy_package.py").read_text(
        encoding="utf-8"
    )

    assert '"--risk-class-balance-power"' in source
    assert '"--risk-class-weight-cap"' in source
    assert "risk_class_balance_power=args.risk_class_balance_power" in source
    assert "risk_class_weight_cap=args.risk_class_weight_cap" in source
    assert '"--risk-critic-class-balance-power"' in source
    assert "risk_class_balance_power=args.risk_critic_class_balance_power" in source
    assert '"--risk-critic-classification-margin"' in source


# 功能：
#   写出双类均衡且内容摘要绑定的合成风险数据，分区任务标识可指定用于交叉重用测试。
# 输入：
#   root：尚不存在的测试数据目录。
#   training_source：训练分区的完整任务摘要。
#   validation_source：验证分区的完整任务摘要。
# 输出：
#   training：四十行训练数据文件。
#   validation：四十行验证数据文件。
#   receipt：绑定上述内容和任务身份的测试回执。
def _write_risk_dataset(
    root: Path,
    *,
    training_source: str = "a" * 64,
    validation_source: str = "b" * 64,
) -> tuple[Path, Path, Path]:
    root.mkdir()
    training = root / "training.jsonl"
    validation = root / "validation.jsonl"
    training_samples = [_sample(index, risky=index < 20) for index in range(40)]
    validation_samples = [_sample(index + 100, risky=index < 20) for index in range(40)]
    training.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in training_samples),
        encoding="utf-8",
    )
    validation.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in validation_samples),
        encoding="utf-8",
    )
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "navigation_label_source": "navigation-expert-trace",
                "verified_sources_required": True,
                "bootstrap_only": False,
                "training_output_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                "validation_output_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
                "training_sources": [{"mission_evidence_sha256": training_source}],
                "validation_sources": [{"mission_evidence_sha256": validation_source}],
            }
        ),
        encoding="utf-8",
    )
    return training, validation, receipt


# 功能：
#   实际读回双类均衡数据并核对样本数、来源集合和回执摘要，不进行模型训练。
# 输入：
#   tmp_path：风险数据目录。
# 输出：
#   None：不返回业务数据。
def test_risk_critic_calibration_accepts_content_bound_balanced_data(tmp_path: Path) -> None:
    module = _module()
    training, validation, receipt = _write_risk_dataset(tmp_path / "risk")

    (
        loaded_training,
        loaded_validation,
        training_counts,
        validation_counts,
        evidence,
        source_hashes,
    ) = module._load_risk_calibration_datasets([[training, validation, receipt]])

    assert len(loaded_training) == 40
    assert len(loaded_validation) == 40
    assert training_counts == {"safe": 20, "risky": 20}
    assert validation_counts == {"safe": 20, "risky": 20}
    assert evidence[0]["dataset_receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert source_hashes == {"a" * 64, "b" * 64}


# 功能：
#   检查两组各自独立的数据合并后，训练任务被另一组当作验证任务时仍须拒绝。
# 输入：
#   tmp_path：两组内容绑定数据目录。
# 输出：
#   None：不返回业务数据。
def test_risk_critic_calibration_rejects_cross_split_source_reuse(tmp_path: Path) -> None:
    module = _module()
    first = _write_risk_dataset(tmp_path / "first")
    second = _write_risk_dataset(
        tmp_path / "second",
        training_source="c" * 64,
        validation_source="a" * 64,
    )

    with pytest.raises(RuntimeError, match="SOURCE_MISSION_OVERLAP"):
        module._load_risk_calibration_datasets([list(first), list(second)])


# 功能：
#   后备导航教师的引导标签必须明确选择引导用途，默认专家训练不能偷偷消费此数据。
# 输入：
#   tmp_path：带引导来源声明的数据回执目录。
# 输出：
#   None：不返回业务数据。
def test_specialist_bootstrap_requires_explicit_opt_in(tmp_path: Path) -> None:
    module = _module()
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("training\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "navigation_label_source": "fallback-navigation-expert-trace",
                "bootstrap_only": True,
                "fallback_teacher_navigation_role": "local-navigation-policy",
                "fallback_risky_target_policy": ("teacher-risk-at-least-0.5-promoted-to-1.0"),
                "verified_sources_required": True,
                "expert_trace_required": True,
                "training_output_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                "validation_output_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ENTANGLED_ADVISOR_LABELS"):
        module._validate_dataset_receipt(
            receipt,
            training_path=training,
            validation_path=validation,
            role="precision-maneuver-policy",
        )

    accepted = module._validate_dataset_receipt(
        receipt,
        training_path=training,
        validation_path=validation,
        role="precision-maneuver-policy",
        bootstrap_from_fallback=True,
    )
    assert accepted["bootstrap_only"] is True


# 功能：
#   检查编码回执绑定原始数据、编码结果、视觉契约与编码器，替换编码器摘要必须失败。
# 输入：
#   tmp_path：占位图数据和实际回执目录；占位模型不用于推理。
# 输出：
#   None：不返回业务数据。
def test_visual_encoding_receipt_binds_source_encoder_contract_and_output(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.jsonl"
    source.write_text("unencoded\n", encoding="utf-8")
    encoded = tmp_path / "encoded.jsonl"
    encoded.write_text("encoded\n", encoding="utf-8")
    encoder = tmp_path / "encoder.onnx"
    encoder.write_bytes(b"encoder")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    encoder_sha256 = hashlib.sha256(encoder.read_bytes()).hexdigest()
    contract = {
        "visual_width": 224,
        "visual_height": 128,
        "visual_feature_count": 139,
        "visual_normalization": "imagenet",
    }
    receipt = tmp_path / "visual-receipt.json"
    payload = {
        "source_policy_data_sha256": source_sha256,
        "output_sha256": hashlib.sha256(encoded.read_bytes()).hexdigest(),
        "perception_encoder_sha256": encoder_sha256,
        "sample_count": 50,
        **contract,
        "p99_preprocess_latency_ms": 1.5,
        "p99_encoder_latency_ms": 4.0,
        "qualification_granted": False,
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    accepted = module._validate_visual_encoding_receipt(
        receipt,
        encoded_path=encoded,
        source_policy_sha256=source_sha256,
        perception_encoder_sha256=encoder_sha256,
        expected_sample_count=50,
        expected_contract=contract,
    )
    assert accepted["visual_feature_count"] == 139

    payload["perception_encoder_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="VISUAL_ENCODER_HASH_MISMATCH"):
        module._validate_visual_encoding_receipt(
            receipt,
            encoded_path=encoded,
            source_policy_sha256=source_sha256,
            perception_encoder_sha256=encoder_sha256,
            expected_sample_count=50,
            expected_contract=contract,
        )


# 功能：
#   逐步验证恢复专家必需回合身份、结果进展及每回合上限，不能靠普通来源标签放行。
# 输入：
#   tmp_path：恢复引导回执目录。
# 输出：
#   None：不返回业务数据。
def test_recovery_specialist_requires_episode_bound_dataset(tmp_path: Path) -> None:
    module = _module()
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("training\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    payload = {
        "schema_version": "dronedream.local-policy-dataset-receipt.v1",
        "split_method": "held-out-source-campaign",
        "navigation_label_source": "fallback-navigation-expert-trace",
        "bootstrap_only": True,
        "fallback_teacher_navigation_role": "local-navigation-policy",
        "fallback_risky_target_policy": ("teacher-risk-at-least-0.5-promoted-to-1.0"),
        "verified_sources_required": True,
        "expert_trace_required": True,
        "training_output_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
        "validation_output_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="RECOVERY_EPISODE_IDENTITY_NOT_REQUIRED"):
        module._validate_dataset_receipt(
            receipt,
            training_path=training,
            validation_path=validation,
            role="recovery-policy",
            bootstrap_from_fallback=True,
        )

    payload["recovery_episode_identity_required"] = True
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RECOVERY_OUTCOME_NOT_REQUIRED"):
        module._validate_dataset_receipt(
            receipt,
            training_path=training,
            validation_path=validation,
            role="recovery-policy",
            bootstrap_from_fallback=True,
        )

    payload["recovery_outcome_progress_required"] = True
    payload["recovery_samples_capped_per_episode"] = 1
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    accepted = module._validate_dataset_receipt(
        receipt,
        training_path=training,
        validation_path=validation,
        role="recovery-policy",
        bootstrap_from_fallback=True,
    )
    assert accepted["recovery_episode_identity_required"] is True
    assert accepted["recovery_outcome_progress_required"] is True


# 功能：
#   重训读取原包及数据回执时保留更早的历史任务，不能只记录最近一次分区。
# 输入：
#   tmp_path：内容绑定的原数据与增补回执目录。
# 输出：
#   None：不返回业务数据。
def test_specialist_retraining_carries_transitive_source_lineage(tmp_path: Path) -> None:
    module = _module()
    dataset_receipt = tmp_path / "source-dataset-receipt.json"
    dataset_receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "training_sources": [{"mission_evidence_sha256": "a" * 64}],
                "validation_sources": [{"mission_evidence_sha256": "b" * 64}],
            }
        ),
        encoding="utf-8",
    )
    augmentation_receipt = tmp_path / "source-augmentation-receipt.json"
    augmentation_receipt.write_text(
        json.dumps(
            {
                "schema_version": ("dronedream.local-policy-augmentation-receipt.v1"),
                "package_sha256": "c" * 64,
                "expert_role": "precision-maneuver-policy",
                "dataset_receipt_sha256": hashlib.sha256(dataset_receipt.read_bytes()).hexdigest(),
                "historical_source_evidence_sha256": [
                    "a" * 64,
                    "b" * 64,
                    "d" * 64,
                ],
            }
        ),
        encoding="utf-8",
    )

    lineage = module._validate_source_augmentation_lineage(
        receipt_path=augmentation_receipt,
        dataset_receipt_path=dataset_receipt,
        base_package_sha256="c" * 64,
        role="precision-maneuver-policy",
    )

    assert lineage == {"a" * 64, "b" * 64, "d" * 64}


# 功能：
#   原训练回执的历史集合漏掉原验证任务时，拒绝继续传递不完整来源链。
# 输入：
#   tmp_path：故意遗漏历史来源的回执目录。
# 输出：
#   None：不返回业务数据。
def test_specialist_retraining_rejects_incomplete_source_lineage(tmp_path: Path) -> None:
    module = _module()
    dataset_receipt = tmp_path / "source-dataset-receipt.json"
    dataset_receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "training_sources": [{"mission_evidence_sha256": "a" * 64}],
                "validation_sources": [{"mission_evidence_sha256": "b" * 64}],
            }
        ),
        encoding="utf-8",
    )
    augmentation_receipt = tmp_path / "source-augmentation-receipt.json"
    augmentation_receipt.write_text(
        json.dumps(
            {
                "schema_version": ("dronedream.local-policy-augmentation-receipt.v1"),
                "package_sha256": "c" * 64,
                "expert_role": "precision-maneuver-policy",
                "dataset_receipt_sha256": hashlib.sha256(dataset_receipt.read_bytes()).hexdigest(),
                "historical_source_evidence_sha256": ["a" * 64],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SOURCE_LINEAGE_INCOMPLETE"):
        module._validate_source_augmentation_lineage(
            receipt_path=augmentation_receipt,
            dataset_receipt_path=dataset_receipt,
            base_package_sha256="c" * 64,
            role="precision-maneuver-policy",
        )
