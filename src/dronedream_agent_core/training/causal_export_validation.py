"""Offline, source-bound validation of actual causal control exports; no flight authority."""

import hashlib
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
import torch

from dronedream_plugin_sdk.protocol import decode_json

from ..causal_control import CONTROL_HISTORY_CONTRACT_SHA256
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..plugin_files import read_plugin_file
from .causal_policy import (
    FrozenCausalEvaluation,
    causal_examples,
    load_causal_checkpoint,
    validate_learning_examples,
)
from .causal_replay import REPLAY_FILES, decode_replay, read_bound_replay
from .causal_training_inputs import training_cpu_threads
from .visual_lineage import verify_causal_checkpoint_receipt

INPUT_NAMES = ("state_features", "realtime_features", "realtime_valid_mask",
               "control_history", "control_history_mask")
OUTPUT_NAMES = ("candidate_scores", "action_scores", "risk_score", "pilot_control")


# 功能：
#   1. 逐个真实窗口比较 Torch 与单线程 CPU ONNX 的全部输出，测量预热后的纯推理延迟。
#   2. 统计不代替整条实时链路延迟或飞行验收，不授予飞行资格。
# 输入：
#   model：已载入的当前因果控制检查点。
#   content：与训练回执绑定的 ONNX 字节。
#   examples：独立验证任务的完整窗口，不参与梯度或检查点选择。
# 输出：
#   report：数值一致性、验证误差及纯 ONNX 延迟统计。
def evaluate_causal_export(model, content: bytes, examples: list) -> dict:
    if type(content) is not bytes or not content or len(content) > 256 * 1024 * 1024:
        raise ValueError("CAUSAL_EXPORT_CONTENT_INVALID")
    validate_learning_examples(examples, model.config.history_length)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        content, sess_options=options, providers=["CPUExecutionProvider"])
    names = (*INPUT_NAMES, "visual_features") if model.config.visual_feature_count else INPUT_NAMES
    if (tuple(item.name for item in session.get_inputs()) != names
            or tuple(item.name for item in session.get_outputs()) != OUTPUT_NAMES):
        raise ValueError("CAUSAL_EXPORT_PORT_MISMATCH")
    metadata = session.get_modelmeta().custom_metadata_map
    expected_metadata = {
        "architecture": "causal-gru-control",
        "history_length": str(model.config.history_length),
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
        "config_sha256": sha256_json(model.config),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("CAUSAL_EXPORT_METADATA_MISMATCH")
    evaluation = FrozenCausalEvaluation.compile(examples, model.config)
    arrays = tuple(tensor.numpy() for tensor in evaluation.inputs)
    first = {name: array[:1] for name, array in zip(names, arrays, strict=True)}
    # 预热只使用真实验证输入，不制造训练记录；计时不包含解析与 Torch 参考计算。
    for _ in range(8):
        session.run(list(OUTPUT_NAMES), first)
    elapsed, errors, axes, modes = [], np.zeros(4), [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for index in range(len(examples)):
                reference = model(*(tensor[index:index + 1] for tensor in evaluation.inputs))
                feed = {name: array[index:index + 1]
                        for name, array in zip(names, arrays, strict=True)}
                start = perf_counter_ns()
                actual = session.run(list(OUTPUT_NAMES), feed)
                elapsed.append((perf_counter_ns() - start) / 1e6)
                for slot, (expected, result) in enumerate(zip(reference, actual, strict=True)):
                    expected = expected.numpy()
                    if (result.shape != expected.shape or result.dtype != np.float32
                            or not np.isfinite(result).all() or not np.isfinite(expected).all()):
                        raise ValueError("CAUSAL_EXPORT_OUTPUT_INVALID")
                    # 禁用坐标候选必须精确保持；其他浮点输出允许不同 CPU 内核的舍入差异。
                    matches = (np.array_equal(expected, result) if slot == 0 else
                               np.allclose(expected, result, rtol=1e-5, atol=1e-5))
                    if not matches:
                        raise ValueError("CAUSAL_EXPORT_NUMERICAL_MISMATCH:" + OUTPUT_NAMES[slot])
                    errors[slot] = max(errors[slot], float(np.abs(expected - result).max()))
                if (np.any(np.abs(actual[3]) > 1) or np.any(actual[2] < 0)
                        or np.any(actual[2] > 1)):
                    raise ValueError("CAUSAL_EXPORT_CONTROL_RANGE_INVALID")
                axes.append(actual[3][0])
                modes.append(int(actual[1][0].argmax()))
    finally:
        model.train(was_training)
    targets = evaluation.target_axes.numpy()
    moving = evaluation.target_modes.numpy() == 3
    error = np.abs(np.asarray(axes) - targets)
    report = {
        "window_count": len(examples), "parity_passed": True,
        "maximum_absolute_error": dict(zip(OUTPUT_NAMES, errors.tolist(), strict=True)),
        "onnx_mode_accuracy": float(np.mean(np.asarray(modes) == evaluation.target_modes.numpy())),
        "onnx_moving_axis_mae": float(error[moving].mean()) if moving.any() else None,
        "zero_control_moving_axis_mae": (
            float(np.abs(targets[moving]).mean()) if moving.any() else None),
        "onnx_moving_mae_by_axis": dict(zip(("forward", "right", "up", "yaw"),
            error[moving].mean(axis=0).tolist() if moving.any() else [None] * 4, strict=True)),
        "latency_ms": {name: float(np.percentile(elapsed, percentile)) for name, percentile
                       in (("p50", 50), ("p95", 95), ("p99", 99), ("maximum", 100))},
        "latency_scope": "warm-single-window-onnx-cpu-only",
        "latency_sample_count": len(elapsed), "warmup_count": 8, "onnx_threads": 1,
        "torch_validation": evaluation.evaluate(model), "qualified_for_flight": False,
    }
    # 分组和尾部误差防止大量相似直行窗口掩盖少数路线或控制轴的失配。
    report["moving_absolute_error_by_axis"] = {
        name: {stat: float(np.percentile(error[moving, index], percentile)) if moving.any()
               else None for stat, percentile in (("p50", 50), ("p95", 95), ("maximum", 100))}
        for index, name in enumerate(("forward", "right", "up", "yaw"))
    }
    report["validation_groups"] = {}
    for group in sorted({example.group_id for example in examples}):
        selected = np.asarray([example.group_id == group for example in examples]) & moving
        report["validation_groups"][group] = {
            "moving_windows": int(selected.sum()),
            "moving_axis_mae": float(error[selected].mean()) if selected.any() else None,
        }
    physical = []
    for index, example in enumerate(examples):
        limits = example.sample.pilot_control_limits
        if moving[index] and limits is not None:
            scales = (limits.horizontal_speed_mps, limits.horizontal_speed_mps,
                      limits.vertical_speed_mps, limits.yaw_rate_dps)
            physical.append(error[index] * np.asarray(scales))
    report["physical_request_error"] = {
        "scope": "before-harness-scaling-and-actuator-envelope",
        "window_count": len(physical),
        "missing_limits_windows": int(moving.sum()) - len(physical),
        "mae": dict(zip(("forward_mps", "right_mps", "up_mps", "yaw_dps"),
            np.asarray(physical).mean(axis=0).tolist() if physical else [None] * 4, strict=True)),
    }
    return report


# 功能：
#   绑定权重、回放、配置和任务隔离后评价单专家候选，不读取测试集或修改任何模型。
# 输入：
#   directory：训练命令独占生成的专家目录。
# 输出：
#   report：绑定全部实际输入摘要的导出验证报告。
def validate_causal_candidate(directory: Path) -> dict:
    directory = directory.absolute()
    raw_receipt = read_plugin_file(directory / "training-receipt.json", limit=4 * 1024 * 1024)
    receipt = decode_json(raw_receipt, limit=4 * 1024 * 1024)
    if not isinstance(receipt, dict) or receipt.get("expert_role") not in NAVIGATION_EXPERT_ROLES:
        raise ValueError("CAUSAL_CANDIDATE_ROLE_INVALID")
    role = receipt["expert_role"]
    checkpoint = read_plugin_file(directory / (role + ".pt"), limit=256 * 1024 * 1024)
    verify_causal_checkpoint_receipt(checkpoint, raw_receipt, expert_role=role)
    content = read_plugin_file(directory / (role + ".onnx"), limit=256 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != receipt.get("artifact_sha256"):
        raise ValueError("CAUSAL_CANDIDATE_ONNX_CHANGED")
    with training_cpu_threads(1):
        model = load_causal_checkpoint(checkpoint)
        if receipt.get("config") != model.config.model_dump():
            raise ValueError("CAUSAL_CANDIDATE_CONFIG_MISMATCH")
        paths = {name: directory / filename for name, filename in REPLAY_FILES.items()}
        contents = read_bound_replay(paths, receipt)
        decoded, manifest, _ = decode_replay(contents)
        examples = causal_examples(decoded["validation_replay"], stream_groups=manifest.groups,
            history_length=model.config.history_length, navigation_role=role,
            history_observations=decoded["validation_observations"])
        report = evaluate_causal_export(model, content, examples)
        train_examples = causal_examples(decoded["training_replay"], stream_groups=manifest.groups,
            history_length=model.config.history_length, navigation_role=role,
            history_observations=decoded["training_observations"])
        # 只重新评价固定权重，训练集误差与独立验证误差分开保存，不做任何优化。
        report["training_replay_evaluation"] = FrozenCausalEvaluation.compile(
            train_examples, model.config).evaluate(model)
        report["training_replay_evaluation"]["evaluation_split"] = "training-replay"
        report["validation_evaluation_split"] = "validation"
        report["training_risk_target_range"] = [
            min(example.sample.risk_target for example in train_examples),
            max(example.sample.risk_target for example in train_examples),
        ]
    report.update({
        "schema_version": "dronedream.causal-export-validation.v1", "expert_role": role,
        "training_receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "onnx_sha256": hashlib.sha256(content).hexdigest(),
        "replay_artifact_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        "test_set_read": False, "weights_modified": False,
    })
    return report
