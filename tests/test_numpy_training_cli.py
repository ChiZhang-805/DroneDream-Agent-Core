"""Feed-forward trainer input/publication checks; optimizer is explicitly isolated."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from control_fixtures import qualified_pilot_metrics
from test_causal_policy import samples
from test_stream_collection import stream_fixture

from dronedream_agent_core import local_policy_training as training
from scripts import train_local_policy as cli


# 功能：
#   建立有效传感器、车辆和两组不同来源样本，用明确替身隔离训练与整包导出。
# 输入：
#   tmp_path：隔离源码输入及候选输出目录。
#   monkeypatch：替换优化器、指标和整包导出的测试工具。
# 输出：
#   case：实际命令行参数、样本路径和替身对象。
@pytest.fixture
def training_case(tmp_path, monkeypatch):
    episode, _, _, _ = stream_fixture(tmp_path)
    vehicle = json.loads((episode / "reset.json").read_bytes())["config"]["vehicle"]
    sensor = tmp_path / "sensor.json"
    sensor.write_text(
        json.dumps(
            {
                "sensor_id": "depth",
                "translation_body_m": {"x": 0, "y": 0, "z": 0},
                "orientation_body_from_sensor": {"w": 1, "x": 0, "y": 0, "z": 0},
                "minimum_range_m": 0.1,
                "maximum_range_m": 20,
            }
        ),
        encoding="utf-8",
    )
    train_path, valid_path = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    for path, rows in ((train_path, samples()), (valid_path, samples(100, "val"))):
        path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")
    output, receipt = tmp_path / "package", tmp_path / "receipt.json"
    options = {
        "training-data": train_path,
        "validation-data": valid_path,
        "output-package": output,
        "training-receipt": receipt,
        "package-id": "synthetic",
        "display-name": "Synthetic",
        "scope": "general",
        "vehicle-metadata": vehicle,
        "sensor-contract": sensor,
        "epoch-count": 1,
    }
    argv = ["baseline-training"]
    for name, value in options.items():
        argv.extend(["--" + name, str(value)])
    metric = qualified_pilot_metrics()
    optimizer = Mock(return_value=(SimpleNamespace(), metric))
    monkeypatch.setattr(cli, "train_local_policy", optimizer)
    monkeypatch.setattr(cli, "evaluate_local_policy", Mock(return_value=metric))
    # 这是文件事务测试，不让固定测试指标被误认成一次学习效果测量。
    monkeypatch.setattr(cli, "_validate_navigation_metrics", Mock())
    publisher = Mock()
    monkeypatch.setattr(cli, "write_local_policy_package", publisher)
    manifest = SimpleNamespace(
        package_id="synthetic",
        control_feature_contract_sha256="a" * 64,
        scope="general",
        base_package_sha256=None,
        map_sha256=None,
        vehicle_sha256="b" * 64,
        sensor_contract_sha256="c" * 64,
    )
    monkeypatch.setattr(
        cli,
        "load_local_policy_package",
        Mock(return_value=SimpleNamespace(manifest=manifest, package_sha256="d" * 64)),
    )
    monkeypatch.setattr(sys, "argv", argv)
    case = SimpleNamespace(
        argv=argv,
        train=train_path,
        valid=valid_path,
        receipt=receipt,
        optimizer=optimizer,
        publisher=publisher,
    )
    return case


# 功能：
#   通用训练不能悄悄接受只适用于地图热启动的仿真准入凭据。
# 输入：
#   training_case：有效通用训练上下文。
#   monkeypatch：追加错误命令行选项的工具。
# 输出：
#   None：不返回业务数据。
def test_general_training_rejects_orphan_simulation_admission(training_case, monkeypatch):
    case = training_case
    monkeypatch.setattr(sys, "argv", [*case.argv, "--base-simulation-admission", "missing.json"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    case.optimizer.assert_not_called()


# 功能：
#   后续恢复专家的非法随机种子必须在通用模型训练之前被发现。
# 输入：
#   training_case：有效基础输入。
#   monkeypatch：追加恢复专家配置的工具。
# 输出：
#   None：不返回业务数据。
def test_specialist_config_is_validated_before_any_optimizer(training_case, monkeypatch):
    case = training_case
    case.optimizer.side_effect = AssertionError("optimizer-must-not-run")
    monkeypatch.setattr(
        sys, "argv", [*case.argv, "--train-recovery-expert", "--random-seed", str(2**32 - 1)]
    )
    with pytest.raises(ValueError):
        cli.main()
    case.optimizer.assert_not_called()


# 功能：
#   训练期间外部原始文件变化不能让回执摘要指向未被优化器实际读取的数据。
# 输入：
#   training_case：可在发布时改写源文件的测试上下文。
# 输出：
#   None：不返回业务数据。
def test_receipt_hash_binds_consumed_training_bytes(training_case):
    case = training_case
    original = case.train.read_bytes()
    case.publisher.side_effect = lambda **_: case.train.write_bytes(original + b"\n")
    assert cli.main() == 0
    receipt = json.loads(case.receipt.read_bytes())
    assert receipt["training_data_sha256"] == hashlib.sha256(original).hexdigest()


# 功能：
#   输出过程中出现的既有回执不能在训练结束时被覆盖。
# 输入：
#   training_case：可模拟回执路径竞争的上下文。
# 输出：
#   None：不返回业务数据。
def test_competing_receipt_is_preserved(training_case):
    case = training_case
    case.publisher.side_effect = lambda **_: case.receipt.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        cli.main()
    assert case.receipt.read_bytes() == b"existing"


# 功能：
#   样本读取器拒绝重复字段和空白行，不能使用后值覆盖或静默过滤损坏输入。
# 输入：
#   tmp_path：独立样本文件目录。
#   corruption：重复字段或空白行故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["duplicate", "blank"])
def test_sample_loader_rejects_ambiguous_rows(tmp_path, corruption):
    raw = samples()[0].model_dump_json().encode()
    content = b'{"risk_target":0.8,' + raw[1:] if corruption == "duplicate" else raw + b"\n\n"
    path = tmp_path / "samples.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        training.load_training_samples(path)


# 功能：
#   训练与验证直接复用相同字节必须在优化前失败，不把内容不同误称为完整空间独立。
# 输入：
#   training_case：独立的命令行上下文。
# 输出：
#   None：不返回业务数据。
def test_cli_rejects_identical_split_content(training_case):
    case = training_case
    case.valid.write_bytes(case.train.read_bytes())
    with pytest.raises(ValueError, match="identical source"):
        cli.main()
    case.optimizer.assert_not_called()


# 功能：
#   风险提案不能混入前馈行为训练，也不能在优化后才发现监督类型错误。
# 输入：
#   training_case：具备有效行为样本的上下文。
# 输出：
#   None：不返回业务数据。
def test_cli_rejects_action_risk_supervision_before_training(training_case):
    case = training_case
    rows = [row.model_copy(update={"risk_proposed_control": [0.1, 0, 0, 0]}) for row in samples()]
    case.train.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="CANNOT_TRAIN_OR_EVALUATE_BEHAVIOR"):
        cli.main()
    case.optimizer.assert_not_called()
