"""Composition evidence boundaries with real small ONNX files and isolated paths."""

import json
import sys

import pytest
from test_local_policy_composition import _write_advisor_and_receipt, _write_base_package

from dronedream_agent_core import local_policy_composition as composition
from scripts import compose_local_policy_advisors as cli


# 功能：
#   回执的摘要、配置选择布尔值及问题列表必须符合契约，不能靠宽松转换接受。
# 输入：
#   tmp_path：独立模型与回执目录。
#   field：注入故障的字段。
#   value：应拒绝的字段内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("training_data_sha256", "z" * 64),
        ("training_data_sha256", "d" * 64),
        ("validation_used_for_configuration_selection", "false"),
        ("issue_codes", ["training_failed"]),
    ],
)
def test_advisor_evidence_rejects_ambiguous_receipt_fields(tmp_path, field, value):
    artifact, path = _write_advisor_and_receipt(tmp_path)
    receipt = json.loads(path.read_bytes())
    receipt[field] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        composition.load_local_advisor_artifact_evidence(
            role="state-anomaly-detector", artifact_path=artifact, training_receipt_path=path
        )


# 功能：
#   重复 JSON 键不能用后值覆盖前值，将失败回执伪装成训练已接受。
# 输入：
#   tmp_path：测试输入目录。
# 输出：
#   None：不返回业务数据。
def test_advisor_evidence_rejects_duplicate_receipt_keys(tmp_path):
    artifact, path = _write_advisor_and_receipt(tmp_path)
    original = path.read_bytes()
    path.write_bytes(b'{"training_accepted":false,' + original[1:])
    with pytest.raises(ValueError):
        composition.load_local_advisor_artifact_evidence(
            role="state-anomaly-detector", artifact_path=artifact, training_receipt_path=path
        )


# 功能：
#   实际组装期间出现同名回执时必须保留既有内容，不覆盖后宣称成功。
# 输入：
#   tmp_path：基座、增补专家与候选输出目录。
#   monkeypatch：在后端正常关闭时注入回执竞争。
# 输出：
#   None：不返回业务数据。
def test_composition_preserves_competing_receipt(tmp_path, monkeypatch):
    base = _write_base_package(tmp_path / "base")
    artifact, path = _write_advisor_and_receipt(tmp_path)
    evidence = composition.load_local_advisor_artifact_evidence(
        role="state-anomaly-detector", artifact_path=artifact, training_receipt_path=path
    )
    receipt = tmp_path / "composition.json"
    original_close = composition.OnnxLocalPolicyBackend.close

    # 功能：
    #   真正关闭 ONNX 后端后制造目标回执冲突，不替代实际模型加载。
    # 输入：
    #   self：已成功创建的 ONNX 后端。
    # 输出：
    #   None：不返回业务数据。
    def racing_close(self):
        original_close(self)
        receipt.write_bytes(b"preserved")

    monkeypatch.setattr(composition.OnnxLocalPolicyBackend, "close", racing_close)
    with pytest.raises(FileExistsError):
        composition.compose_local_policy_advisors(
            base_package_path=base.root,
            output_package_path=tmp_path / "candidate",
            composition_receipt_path=receipt,
            package_id="test.new",
            display_name="Test",
            additions=(evidence,),
        )
    assert receipt.read_bytes() == b"preserved"


# 功能：
#   输出回执不能写入原基座或候选包内部，防止独立证据操作改变模型包内容。
# 输入：
#   tmp_path：独立测试目录。
#   inside：原基座或新候选目标。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("inside", ["base", "candidate"])
def test_composition_rejects_receipt_inside_model_package(tmp_path, inside):
    base = _write_base_package(tmp_path / "base")
    artifact, path = _write_advisor_and_receipt(tmp_path)
    evidence = composition.load_local_advisor_artifact_evidence(
        role="state-anomaly-detector", artifact_path=artifact, training_receipt_path=path
    )
    with pytest.raises(ValueError):
        composition.compose_local_policy_advisors(
            base_package_path=base.root,
            output_package_path=tmp_path / "candidate",
            composition_receipt_path=tmp_path / inside / "composition.json",
            package_id="test.new",
            display_name="Test",
            additions=(evidence,),
        )


# 功能：
#   通过实际命令行入口组装并重新加载候选，确认回执摘要与输出一致且不授予资格。
# 输入：
#   tmp_path：独立输入输出目录。
#   monkeypatch：设置命令行参数的工具。
#   capsys：捕获结构化标准输出。
# 输出：
#   None：不返回业务数据。
def test_composition_cli_publishes_bound_candidate(tmp_path, monkeypatch, capsys):
    base = _write_base_package(tmp_path / "base")
    artifact, training_receipt = _write_advisor_and_receipt(tmp_path)
    candidate, receipt = tmp_path / "candidate", tmp_path / "composition.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "composition",
            "--base-package",
            str(base.root),
            "--output-package",
            str(candidate),
            "--composition-receipt",
            str(receipt),
            "--package-id",
            "test.cli",
            "--display-name",
            "Test",
            "--advisor",
            "state-anomaly-detector",
            str(artifact),
            str(training_receipt),
        ],
    )
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    stored = json.loads(receipt.read_bytes())
    package = composition.load_local_policy_package(candidate)
    assert output["package_sha256"] == stored["package_sha256"] == package.package_sha256
    assert output["qualification_granted"] is stored["qualification_granted"] is False
