"""Offline advisor entry-point boundaries; no product model or flight qualification."""

import hashlib
import json
import sys

import pytest
from test_local_advisor_training import _sample
from test_mission_groups import split_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_advisor_training import LocalAdvisorTrainingMetrics
from dronedream_agent_core.training.advisor_sources import RecordedSource
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from scripts import train_local_advisors as cli

ROLE = "perception-health-critic"


# 功能：
#   构造数量守恒且全部通过阈值的离线验收指标，供边界测试单独破坏字段。
# 输入：
#   无。
# 输出：
#   metrics：40 条样本、两类各 20 条的指标。
def metrics():
    result = LocalAdvisorTrainingMetrics(
        sample_count=40,
        risky_sample_count=20,
        safe_sample_count=20,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.0,
    )
    return result


# 功能：
#   确认样本读取拒绝重复键、空白记录、未终止末行和布尔数值，不静默改变训练输入。
# 输入：
#   tmp_path：隔离测试目录。
#   mutation：破坏 JSONL 字节的方法。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["duplicate", "blank", "unterminated", "boolean"])
def test_sample_reader_rejects_ambiguous_rows(tmp_path, mutation):
    row = _sample(role=ROLE, risky=False).model_dump_json()
    if mutation == "duplicate":
        row = '{"risk_target":0.9,' + row[1:]
    elif mutation == "boolean":
        row = row.replace('"risk_target":0.0', '"risk_target":false')
    data = (row + ("" if mutation == "unterminated" else "\n")).encode()
    if mutation == "blank":
        data += b"\n"
    with pytest.raises(ValueError):
        cli._load_samples(RecordedSource(tmp_path / "data.jsonl", data))


# 功能：
#   验证不能用非十六进制字符串冒充源任务摘要。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_source_digest_requires_lowercase_hex():
    with pytest.raises(ValueError):
        cli._source_evidence_hashes(
            {"training_sources": [{"mission_evidence_sha256": "z" * 64}]}, "training"
        )


# 功能：
#   验证验收入口重新校验指标，防止绕过模型赋值验证后的 NaN 或布尔值通过比较。
# 输入：
#   field：被破坏的指标名。
#   value：不合法数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("risk_hold_recall", float("nan")),
        ("risk_mean_absolute_error", float("nan")),
        ("safe_motion_recall", True),
    ],
)
def test_acceptance_revalidates_metrics(field, value):
    broken = metrics().model_copy(update={field: value})
    with pytest.raises(ValueError):
        cli._validate_metrics(ROLE, broken)


# 功能：
#   验证类别覆盖同时检查总数守恒，不接受独立篡改后的安全样本数量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_class_gate_revalidates_counts():
    broken = metrics().model_copy(update={"safe_sample_count": 100})
    with pytest.raises(ValueError):
        cli._validate_class_coverage(ROLE, broken, split="training")


# 功能：
#   生成两个内容与空间分离的小型 JSONL 数据集及内容绑定回执，不执行真实训练。
# 输入：
#   root：隔离文件目录。
# 输出：
#   paths：训练数据、验证数据和数据集回执路径。
def dataset(root):
    paths = [root / name for name in ("training.jsonl", "validation.jsonl", "dataset.json")]
    for index, path in enumerate(paths[:2]):
        row = _sample(role=ROLE, risky=False)
        row.state_features[0] = float(index)
        path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    groups = [split_fixture(), split_fixture(y=2.0)]
    receipt = {
        "schema_version": "dronedream.local-advisor-dataset-receipt.v1",
        "split_method": SPATIAL_SPLIT_CONTRACT,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "verified_sources_required": True,
        "roles": [ROLE],
    }
    for index, name in enumerate(("training", "validation")):
        receipt[name + "_output_sha256"] = hashlib.sha256(paths[index].read_bytes()).hexdigest()
        receipt[name + "_groups"] = [groups[index].group_sha256]
        receipt[name + "_sources"] = [
            {
                "mission_evidence_sha256": ("b" if index else "c") * 64,
                "semantic_sha256": "a" * 64,
                "mission_split": groups[index].model_dump(),
            }
        ]
    paths[2].write_text(json.dumps(receipt), encoding="utf-8")
    return paths


# 功能：
#   验证数据集回执不能通过重复字段覆盖原有失败声明。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_rejects_duplicate_keys(tmp_path):
    training, validation, receipt = dataset(tmp_path)
    data = '{"verified_sources_required":false,' + receipt.read_text(encoding="utf-8")[1:]
    with pytest.raises(ValueError):
        cli._validate_dataset_receipt(
            RecordedSource(receipt, data.encode()),
            training_data=RecordedSource.read(training),
            validation_data=RecordedSource.read(validation),
            roles=(ROLE,),
        )


# 功能：
#   模拟耗时训练期间目标回执被其他流程创建，验证最终发布不会覆盖该文件。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：仅替换计算和导出，不绕过文件发布。
# 输出：
#   None：不返回业务数据。
def test_training_does_not_overwrite_late_receipt(tmp_path, monkeypatch):
    training, validation, source = dataset(tmp_path)
    destination, receipt = tmp_path / "models", tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--training-data",
            str(training),
            "--validation-data",
            str(validation),
            "--dataset-receipt",
            str(source),
            "--output-directory",
            str(destination),
            "--training-receipt",
            str(receipt),
            "--role",
            ROLE,
        ],
    )
    monkeypatch.setattr(cli, "train_local_advisor", lambda *a, **k: (object(), metrics()))
    monkeypatch.setattr(cli, "evaluate_local_advisor", lambda *a: metrics())

    # 功能：
    #   在导出期间制造非本次流程所有的回执，保留真实发布路径用于检测覆盖。
    # 输入：
    #   model：未实际使用的测试模型。
    #   path：训练入口分配的暂存模型路径。
    # 输出：
    #   checksum：导出测试字节的摘要。
    def export(model, path):
        path.write_bytes(b"test-model")
        receipt.write_text("do not overwrite", encoding="utf-8")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        return checksum

    monkeypatch.setattr(cli, "export_local_advisor_onnx", export)
    with pytest.raises(FileExistsError):
        cli.main()
    assert receipt.read_text(encoding="utf-8") == "do not overwrite"
