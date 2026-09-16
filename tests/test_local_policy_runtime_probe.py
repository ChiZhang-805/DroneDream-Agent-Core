"""Actual ONNX interface execution tests, never real model or flight evidence."""

import hashlib
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from test_causal_packaging import base_package  # noqa: F401
from test_complete_artifact_assembly import recipe  # noqa: F401
from test_local_policy_runtime_staging import _admission

from dronedream_agent_core.local_policy_port import OnnxLocalPolicyBackend
from dronedream_agent_core.local_policy_runtime_probe import _probe_session
from dronedream_agent_core.training.artifact_assembly import assemble_complete_ensemble


# 功能：
#   构造仅供边界测试的真实 ONNX 常量会话，输入与输出类型均由用例明确指定。
# 输入：
#   value：合成输出数组，不代表训练所得权重。
#   input_shape：图声明的状态输入尺寸。
# 输出：
#   session：CPU 单线程真实 ONNX 会话。
def _constant_session(value, input_shape=(1, 46)):
    output_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["risk_score"],
                               value=onnx.numpy_helper.from_array(value))],
        "synthetic-interface-test",
        [onnx.helper.make_tensor_value_info("state_features", onnx.TensorProto.FLOAT, input_shape)],
        [onnx.helper.make_tensor_value_info("risk_score", output_type, value.shape)],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, ["CPUExecutionProvider"])
    return session


# 功能：
#   会话能加载但实际输出错误时必须在接口预检阶段拒绝，而不是交给用户飞行时发现。
# 输入：
#   value：非有限、越界、错误秩或错误类型的合成输出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [
    np.asarray([np.nan], dtype=np.float32), np.asarray([np.inf], dtype=np.float32),
    np.asarray([-0.1], dtype=np.float32), np.asarray([1.1], dtype=np.float32),
    np.asarray([0.5], dtype=np.float64), np.asarray([[[0.5]]], dtype=np.float32),
])
def test_loadable_graph_with_bad_output_is_rejected(value):
    session = _constant_session(value)
    with pytest.raises(RuntimeError, match="LOCAL_POLICY_PROBE"):
        _probe_session("risk-critic", session, {"state_features": (1, 46)}, {"risk_score": 1})


# 功能：
#   同名但宽度不符或批次不符的图不能通过检查，动态批次按当前单批次输入实际计算。
# 输入：
#   shape：ONNX 声明形状。
#   accepted：该声明是否与当前运行输入相容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("shape,accepted", [
    ((1, 45), False), ((2, 46), False), ((1, "unknown"), False),
    (("batch", 46), True), ((1, 46), True),
])
def test_input_shape_must_match_runtime_not_only_name(shape, accepted):
    session = _constant_session(np.asarray([0.5], dtype=np.float32), shape)
    if accepted:
        result = _probe_session(
            "risk-critic", session, {"state_features": (1, 46)}, {"risk_score": 1},
        )
        assert result["computed"] is True
    else:
        with pytest.raises(RuntimeError, match="PROBE_INPUT_MISMATCH"):
            _probe_session("risk-critic", session, {"state_features": (1, 46)}, {"risk_score": 1})


# 功能：
#   完整合成包实际计算全部十专家，且检查不会填充真实控制历史或授予飞行资格。
# 输入：
#   recipe：仅用于接口测试的十角色配方。
#   tmp_path：独立发布目录。
# 输出：
#   None：不返回业务数据。
def test_all_ten_experts_compute_without_control_history_mutation(recipe, tmp_path):  # noqa: F811
    package = assemble_complete_ensemble(
        recipe=recipe, source_root=tmp_path, output_root=tmp_path / "complete",
    )
    backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    try:
        assert backend.motion_history_ready() is False
        result = backend.verify_runtime_io()
        assert result["expert_count"] == 10
        assert result["qualification_granted"] is False
        assert all(item["computed"] for item in result["experts"].values())
        assert backend.motion_history_ready() is False
        assert not backend._observation_history.rows
    finally:
        backend.close()


# 功能：
#   将真实图的负载缩放输出改成越界常量并重新绑定测试来源，验证组装入口拒绝且不发布半成品。
# 输入：
#   recipe：仅供测试的合成配方。
#   tmp_path：独立来源及输出目录。
# 输出：
#   None：不返回业务数据。
def test_assembly_rejects_computational_failure_before_publication(recipe, tmp_path):  # noqa: F811
    role = "payload-dynamics-adapter"
    source = next(item for item in recipe.sources if item.role == role)
    graph = onnx.load(str(source.artifact_path))
    for node in graph.graph.node:
        for index, name in enumerate(node.output):
            if name == "controller_step_scale":
                node.output[index] = "unused_original_scale"
    graph.graph.node.append(onnx.helper.make_node(
        "Constant", [], ["controller_step_scale"],
        value=onnx.numpy_helper.from_array(np.asarray([[2.0]], dtype=np.float32)),
    ))
    content = graph.SerializeToString()
    source.artifact_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    artifact = next(item for item in recipe.manifest.artifacts if item.role == role)
    artifact.sha256 = digest
    receipt = json.loads(source.training_receipt_path.read_bytes())
    receipt["advisors"][role]["artifact_sha256"] = digest
    receipt_content = json.dumps(receipt).encode()
    source.training_receipt_path.write_bytes(receipt_content)
    source.training_receipt_sha256 = hashlib.sha256(receipt_content).hexdigest()
    output = tmp_path / "must-not-publish"
    with pytest.raises(RuntimeError, match="PROBE:payload-dynamics-adapter"):
        assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".complete-ensemble-*"))


# 功能：
#   完整合成包经正式只读预检入口计算十专家；测试许可和准入仅留在临时目录，不是产品授权。
# 输入：
#   recipe：仅供测试的十专家配方。
#   tmp_path：隔离模型及回执目录。
#   monkeypatch：限定本次测试命令行参数。
#   capsys：捕获接口预检输出。
# 输出：
#   None：不返回业务数据。
def test_staging_check_executes_all_experts(recipe, tmp_path, monkeypatch, capsys):  # noqa: F811
    from scripts.stage_local_policy_runtime import main

    package = assemble_complete_ensemble(
        recipe=recipe, source_root=tmp_path, output_root=tmp_path / "staging-input",
    )
    admission = tmp_path / "admission.json"
    _admission(admission, package, continuous_evidence=True, expert_evidence=True)
    licenses = tmp_path / "licenses.json"
    licenses.write_text(json.dumps({
        "schema_version": "dronedream.model-distribution-licenses.v1",
        "package_sha256": package.package_sha256,
        "artifacts": [
            {"role": a.role, "sha256": a.sha256, "redistribution_approved": True,
             "license_id": "LicenseRef-Test-Only", "source": "synthetic test graph",
             "license_text": "Test fixture only; not permission for real model weights."}
            for a in package.manifest.artifacts
        ],
    }))
    monkeypatch.setattr(sys, "argv", [
        "stage", "--package", str(package.root), "--simulation-admission", str(admission),
        "--distribution-licenses", str(licenses), "--check-only", "--verify-onnx",
    ])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["computed_expert_count"] == result["expert_count"] == 10
    assert result["onnx_verified"] is True
    assert result["package_sha256"] == package.package_sha256
    assert result["deployment_scope"] == "simulation-only"
