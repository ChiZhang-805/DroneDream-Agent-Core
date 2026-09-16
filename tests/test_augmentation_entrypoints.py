"""Candidate-era augmentation boundaries; optimizer metrics are explicit test doubles."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from control_fixtures import qualified_pilot_metrics
from test_local_policy_augmentation import _sample, _write_risk_dataset
from test_local_policy_composition import _write_base_package

from dronedream_agent_core import local_policy_training as training
from scripts import augment_local_policy_package as cli


# 功能：
#   准备可真实读写和导出的前馈基座及独立来源样本，仅隔离专家优化与质量数值。
# 输入：
#   tmp_path：测试输入输出目录。
#   monkeypatch：设置命令行及明确训练替身的工具。
# 输出：
#   case：原始参数、来源、输出及优化器替身。
@pytest.fixture
def augmentation_case(tmp_path, monkeypatch):
    base = _write_base_package(tmp_path / "base")
    paths = [tmp_path / "training.jsonl", tmp_path / "validation.jsonl"]
    for path, offset in zip(paths, [0, 100], strict=True):
        rows = [_sample(offset + index, risky=index < 15) for index in range(50)]
        path.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "navigation_label_source": "navigation-expert-trace",
                "verified_sources_required": True,
                "expert_trace_required": True,
                "training_sources": [{"mission_evidence_sha256": "a" * 64}],
                "validation_sources": [{"mission_evidence_sha256": "b" * 64}],
                "training_output_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "validation_output_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output, receipt = tmp_path / "candidate", tmp_path / "training-receipt.json"
    options = {
        "base-package": base.root,
        "training-data": paths[0],
        "validation-data": paths[1],
        "dataset-receipt": dataset,
        "output-package": output,
        "training-receipt": receipt,
        "package-id": "test.augmented",
        "display-name": "Test",
        "epoch-count": 1,
        "expert-role": "precision-maneuver-policy",
    }
    argv = ["augmentation"]
    for name, value in options.items():
        argv.extend(["--" + name, str(value)])
    metric = qualified_pilot_metrics()
    model = training._new_model(training.LocalPolicyTrainingConfig(hidden_feature_count=8))
    optimizer = Mock(return_value=(model, metric))
    monkeypatch.setattr(cli, "train_local_policy", optimizer)
    monkeypatch.setattr(cli, "evaluate_local_policy", Mock(return_value=metric))
    monkeypatch.setattr(sys, "argv", argv)
    case = SimpleNamespace(
        argv=argv,
        base=base,
        training=paths[0],
        dataset=dataset,
        receipt=receipt,
        output=output,
        optimizer=optimizer,
    )
    return case


# 功能：
#   首次增加专家必须计为新增角色，实际包与回执读回应保持基座其他权重不变。
# 输入：
#   augmentation_case：实际文件上下文及明确优化器替身。
# 输出：
#   None：不返回业务数据。
def test_augmentation_can_add_previously_absent_expert(augmentation_case):
    case = augmentation_case
    assert cli.main() == 0
    package = cli.load_local_policy_package(case.output)
    receipt = json.loads(case.receipt.read_bytes())
    assert "precision-maneuver-policy" in package.artifact_paths
    assert receipt["package_sha256"] == package.package_sha256
    assert receipt["qualification_granted"] is False


# 功能：
#   风险校准的非法随机种子必须在任何专家优化开始之前被发现。
# 输入：
#   augmentation_case：有效增补上下文。
#   tmp_path：额外风险数据目录。
#   monkeypatch：追加风险训练参数的工具。
# 输出：
#   None：不返回业务数据。
def test_augmentation_preflights_risk_config(augmentation_case, tmp_path, monkeypatch):
    case = augmentation_case
    paths = _write_risk_dataset(tmp_path / "risk")
    monkeypatch.setattr(
        sys, "argv", [*case.argv, "--risk-dataset", *map(str, paths), "--risk-random-seed", "-1"]
    )
    with pytest.raises(ValueError):
        cli.main()
    case.optimizer.assert_not_called()


# 功能：
#   来源任务摘要必须是十六进制，不得仅长度正确就作为独立证据。
# 输入：
#   无：直接构造非法来源摘要。
# 输出：
#   None：不返回业务数据。
def test_augmentation_rejects_nonhex_mission_identity():
    receipt = {
        "training_sources": [{"mission_evidence_sha256": "z" * 64}],
        "validation_sources": [{"mission_evidence_sha256": "a" * 64}],
    }
    with pytest.raises(RuntimeError):
        cli._dataset_source_evidence(receipt)


# 功能：
#   校准回执的重复键不能静默覆盖，避免失败来源变为已验证。
# 输入：
#   tmp_path：风险数据及回执目录。
# 输出：
#   None：不返回业务数据。
def test_augmentation_rejects_duplicate_calibration_receipt(tmp_path):
    paths = _write_risk_dataset(tmp_path / "risk")
    receipt = paths[2]
    receipt.write_bytes(b'{"verified_sources_required":false,' + receipt.read_bytes()[1:])
    with pytest.raises(ValueError):
        cli._load_risk_calibration_datasets([list(paths)])


# 功能：
#   训练期间原始数据被替换时，实际训练来源及发布回执仍绑定开始时的固定副本。
# 输入：
#   augmentation_case：有效基座、来源和优化器替身。
# 输出：
#   None：不返回业务数据。
def test_augmentation_freezes_sources_before_optimizer(augmentation_case):
    case = augmentation_case
    original_hash = hashlib.sha256(case.training.read_bytes()).hexdigest()
    original_dataset_hash = hashlib.sha256(case.dataset.read_bytes()).hexdigest()

    # 功能：
    #   模拟训练运行期间外部输入发生变化，不改动优化器已经收到的样本。
    # 输入：
    #   samples：由冻结来源读出的监督行。
    #   config：已验证配置。
    #   initial_model：可选热启动网络。
    # 输出：
    #   result：预先声明的测试网络和质量替身。
    def replace_sources(samples, config, *, initial_model):
        assert len(samples) == 50
        case.training.write_bytes(b"external replacement")
        case.dataset.write_bytes(b"external receipt replacement")
        result = case.optimizer.return_value
        return result

    case.optimizer.side_effect = replace_sources
    assert cli.main() == 0
    receipt = json.loads(case.receipt.read_bytes())
    assert receipt["training_data_sha256"] == original_hash
    assert receipt["dataset_receipt_sha256"] == original_dataset_hash
    assert receipt["source_package_sha256"] == case.base.package_sha256
    assert not list(case.output.parent.glob(".augmentation-inputs-*"))


# 功能：
#   优化期间出现同名回执时保留对方文件并失败，不用成功回执覆盖它。
# 输入：
#   augmentation_case：候选输出及训练替身。
# 输出：
#   None：不返回业务数据。
def test_augmentation_preserves_competing_receipt(augmentation_case):
    case = augmentation_case

    # 功能：
    #   在输出预检之后制造另一个写入者，验证最终发布仍采用无覆盖操作。
    # 输入：
    #   samples：已固定的训练样本。
    #   config：训练配置。
    #   initial_model：热启动网络。
    # 输出：
    #   result：隔离训练的固定网络及指标。
    def compete(samples, config, *, initial_model):
        case.receipt.write_bytes(b"other writer")
        result = case.optimizer.return_value
        return result

    case.optimizer.side_effect = compete
    with pytest.raises(FileExistsError):
        cli.main()
    assert case.receipt.read_bytes() == b"other writer"
    # 候选和外部回执不是同一事务；候选可留下，但绝不据此取得飞行资格。
    assert case.output.is_dir()


# 功能：
#   实际后端关闭失败时不得发布候选或成功回执，且正常实例在注入异常之前确实释放。
# 输入：
#   augmentation_case：合法候选训练上下文。
#   monkeypatch：在真实关闭之后注入故障的工具。
# 输出：
#   None：不返回业务数据。
def test_augmentation_closes_backend_before_publication(augmentation_case, monkeypatch):
    close = cli.OnnxLocalPolicyBackend.close
    closed = []

    # 功能：
    #   先关闭实际 ONNX 后端，再报告关闭失败，避免测试自身保留推理资源。
    # 输入：
    #   backend：本次实际构造的后端实例。
    # 输出：
    #   None：不返回业务数据。
    def fail_after_close(backend):
        close(backend)
        closed.append(True)
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(cli.OnnxLocalPolicyBackend, "close", fail_after_close)
    with pytest.raises(RuntimeError, match="injected close failure"):
        cli.main()
    assert closed == [True]
    assert not augmentation_case.output.exists()
    assert not augmentation_case.receipt.exists()


# 功能：
#   风险支持数量按风险标签而非扫描或停留动作统计，低风险停留不能充当危险样本。
# 输入：
#   无：构造低风险、非移动的样本。
# 输出：
#   None：不返回业务数据。
def test_augmentation_hold_is_not_automatically_risky():
    sample = _sample(1, risky=False).model_copy(update={"target_action_index": 8})
    assert cli._risk_class(sample) == "safe"


# 功能：
#   输出位于原基座或回执位于候选包内时，在优化和来源副本创建之前拒绝。
# 输入：
#   augmentation_case：有效输入与参数。
#   monkeypatch：替换输出选项的工具。
#   destination：需要测试的嵌套输出位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("destination", ["base", "candidate"])
def test_augmentation_rejects_nested_outputs(augmentation_case, monkeypatch, destination):
    case = augmentation_case
    argv = list(case.argv)
    if destination == "base":
        argv[argv.index("--output-package") + 1] = str(case.base.root / "nested")
    else:
        argv[argv.index("--training-receipt") + 1] = str(case.output / "receipt.json")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="outside source and candidate"):
        cli.main()
    case.optimizer.assert_not_called()


# 功能：
#   风险分类间隔非法时在专家优化之前拒绝，而非训练结束后才报错。
# 输入：
#   augmentation_case：有效专家上下文。
#   tmp_path：额外风险数据目录。
#   monkeypatch：追加间隔参数的工具。
#   margin：负数、上界及非有限数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("margin", ["-0.1", "0.5", "nan", "inf"])
def test_augmentation_preflights_risk_margin(augmentation_case, tmp_path, monkeypatch, margin):
    case = augmentation_case
    paths = _write_risk_dataset(tmp_path / "risk")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *case.argv,
            "--risk-dataset",
            *map(str, paths),
            "--risk-critic-classification-margin",
            margin,
        ],
    )
    with pytest.raises(ValueError):
        cli.main()
    case.optimizer.assert_not_called()


# 功能：
#   基座清单重复字段必须在模型读取与优化前拒绝，不保留最后一项掩盖前项。
# 输入：
#   augmentation_case：实际可加载基座与训练替身。
# 输出：
#   None：不返回业务数据。
def test_augmentation_rejects_duplicate_base_manifest(augmentation_case):
    case = augmentation_case
    path = case.base.manifest_path
    path.write_bytes(b'{"package_id":"shadow",' + path.read_bytes()[1:])
    with pytest.raises(ValueError):
        cli.main()
    case.optimizer.assert_not_called()
