"""Actual ONNX parity and provenance failures with isolated synthetic fixtures."""

import hashlib
import json

import onnx
import pytest
import torch
from test_causal_policy import samples
from test_mission_groups import group_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.training import causal_export_validation as validation
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    causal_examples,
    export_causal_policy,
    save_causal_checkpoint,
)
from dronedream_agent_core.training.causal_replay import CAUSAL_SPLIT_CONTRACT, export_replay_bundle
from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads


# 功能：
#   为独立验证构造真实可运行的小网络及合成窗口，避免把测试夹具记入正式数据。
# 输入：
#   tmp_path：隔离导出目录。
#   visual_width：合成视觉向量长度。
# 输出：
#   model：随机初始化的测试网络。
#   content：实际导出的 ONNX 字节。
#   examples：有真实时序契约的合成验证窗口。
def fixture_export(tmp_path, visual_width=0):
    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                               head_width=32, visual_feature_count=visual_width, epochs=1)
    model = CausalPilotPolicy(config)
    rows = samples(100, "val")
    for row in rows:
        row.visual_features = [0.25] * visual_width
    examples = causal_examples(rows, stream_groups={"val": "route-b"}, history_length=4)
    path = tmp_path / "local-navigation-policy.onnx"
    export_causal_policy(model, path)
    content = path.read_bytes()
    return model, content, examples


# 功能：
#   有视觉和无视觉端口都必须运行实际单窗口推理，保持调用方训练模式及无飞行权限。
# 输入：
#   tmp_path：隔离模型目录。
#   visual_width：本次测试的可选视觉宽度。
# 输出：
#   None：断言不成立时测试失败。
@pytest.mark.parametrize("visual_width", [0, 7])
def test_actual_export_parity_and_latency(tmp_path, visual_width):
    with training_cpu_threads(1):
        model, content, examples = fixture_export(tmp_path, visual_width)
        assert model.training
        report = validation.evaluate_causal_export(model, content, examples)
        assert model.training
    assert report["parity_passed"] is True
    assert report["window_count"] == len(examples)
    assert report["latency_sample_count"] == len(examples)
    assert report["qualified_for_flight"] is False
    assert report["maximum_absolute_error"]["pilot_control"] < 1e-5
    assert 0 < report["latency_ms"]["p50"] <= report["latency_ms"]["maximum"]
    assert set(report["onnx_moving_mae_by_axis"]) == {"forward", "right", "up", "yaw"}
    assert report["validation_groups"]["route-b"]["moving_windows"] > 0
    assert report["moving_absolute_error_by_axis"]["forward"]["maximum"] >= 0


# 功能：
#   按每条记录的真实控制尺度报告物理请求误差，不把归一化误差直接当米每秒。
# 输入：
#   tmp_path：隔离模型目录。
# 输出：
#   None：单位缩放或来源覆盖错误时测试失败。
def test_error_uses_per_sample_physical_scales(tmp_path):
    with training_cpu_threads(1):
        model, content, examples = fixture_export(tmp_path)
        for example in examples:
            example.sample.pilot_control_limits = PilotControlLimits(0.4, 0.2, 45.0)
        report = validation.evaluate_causal_export(model, content, examples)
    physical = report["physical_request_error"]
    assert physical["window_count"] == len(examples)
    assert physical["missing_limits_windows"] == 0
    for axis, scale, unit in (("forward", 0.4, "mps"), ("right", 0.4, "mps"),
                             ("up", 0.2, "mps"), ("yaw", 45.0, "dps")):
        assert physical["mae"][axis + "_" + unit] == pytest.approx(
            report["onnx_moving_mae_by_axis"][axis] * scale)


# 功能：
#   元数据、输出端口或权重被替换时拒绝导出；比较失败仍恢复 Torch 模式。
# 输入：
#   tmp_path：隔离模型目录。
#   fault：本次注入的部署不一致类别。
# 输出：
#   None：没有拒绝不一致时测试失败。
@pytest.mark.parametrize("fault", ["metadata", "port", "weights", "nonfinite"])
def test_export_rejects_mismatches(tmp_path, fault):
    with training_cpu_threads(1):
        model, content, examples = fixture_export(tmp_path)
        graph = onnx.load_model_from_string(content)
        if fault == "metadata":
            graph.metadata_props[0].value = "old-policy"
        elif fault == "port":
            graph.graph.output[0].name = "invalid_output"
            # 重命名图内生产者，以保持 ONNX 可载入而真实触发端口校验。
            for node in graph.graph.node:
                for index, name in enumerate(node.output):
                    if name == "candidate_scores":
                        node.output[index] = "invalid_output"
        else:
            with torch.no_grad():
                model.axes_head.bias.fill_(float("nan") if fault == "nonfinite" else 9.0)
        with pytest.raises(ValueError, match="CAUSAL_EXPORT_"):
            validation.evaluate_causal_export(model, graph.SerializeToString(), examples)
        assert model.training


# 功能：
#   生成带检查点、五项回放及完整摘要的测试候选，验证入口必须实际检查绑定。
# 输入：
#   tmp_path：隔离候选目录。
# 输出：
#   receipt：与测试产物绑定的可变回执夹具。
def fixture_candidate(tmp_path):
    model, content, _ = fixture_export(tmp_path)
    checkpoint = tmp_path / "local-navigation-policy.pt"
    save_causal_checkpoint(model, checkpoint)
    train, held = samples(), samples(100, "val")
    histories = [[LocalPolicyObservation(**{name: getattr(row, name)
                  for name in LocalPolicyObservation.model_fields}) for row in rows]
                 for rows in (train, held)]
    hashes = export_replay_bundle(tmp_path, training=train, validation=held,
        training_history=histories[0], validation_history=histories[1], manifest=group_fixture())
    receipt = {
        "architecture": "causal-gru-control", "expert_role": "local-navigation-policy",
        "config": model.config.model_dump(), "visual_input_contract": None,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "split_contract": CAUSAL_SPLIT_CONTRACT, "replay_artifact_sha256": hashes,
    }
    (tmp_path / "training-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


# 功能：
#   完整验证入口读取实际绑定的验证回放，不接触测试集，也不改动任何候选字节。
# 输入：
#   tmp_path：隔离候选目录。
# 输出：
#   None：内容或权限断言失败时测试失败。
def test_complete_candidate_preserves_files(tmp_path):
    with training_cpu_threads(1):
        fixture_candidate(tmp_path)
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        report = validation.validate_causal_candidate(tmp_path)
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert report["parity_passed"]
    assert report["test_set_read"] is False
    assert report["weights_modified"] is False
    assert report["training_replay_evaluation"]["evaluation_split"] == "training-replay"
    assert report["validation_evaluation_split"] == "validation"
    assert report["training_risk_target_range"] == [0.0, 1.0]


# 功能：
#   权重、配置和历史任一被改动都必须失败，不能拿旧回执为新产物背书。
# 输入：
#   tmp_path：隔离候选目录。
#   fault：需要破坏的绑定部位。
# 输出：
#   None：错误产物未被拒绝时测试失败。
@pytest.mark.parametrize("fault", ["onnx", "checkpoint", "replay", "config", "role"])
def test_candidate_rejects_broken_bindings(tmp_path, fault):
    with training_cpu_threads(1):
        receipt = fixture_candidate(tmp_path)
        if fault in {"config", "role"}:
            if fault == "config":
                receipt["config"]["epochs"] = 2
            else:
                receipt["expert_role"] = "../../escape"
            (tmp_path / "training-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        else:
            filename = {"onnx": "local-navigation-policy.onnx",
                        "checkpoint": "local-navigation-policy.pt",
                        "replay": "validation-observations.jsonl"}[fault]
            with (tmp_path / filename).open("ab") as stream:
                stream.write(b" ")
        with pytest.raises(ValueError, match="CAUSAL_"):
            validation.validate_causal_candidate(tmp_path)
