"""Evaluate a frozen exported model on an untouched split; never train or promote it."""

import argparse
import hashlib
import json
from pathlib import Path

from dronedream_agent_core.local_vision_training import (
    LOCAL_VISION_ARCHITECTURE,
    LocalVisionTrainingConfig,
    evaluate_local_vision_model,
    resolve_local_vision_samples,
)
from dronedream_agent_core.plugin_files import hash_plugin_file, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json, read_runtime_object
from dronedream_agent_core.training.artifact_assembly import validate_embedded_graph
from dronedream_agent_core.training.vision_manifest import (
    inspect_vision_split,
    read_vision_manifest,
    require_disjoint_vision_splits,
)


# 功能：
#   在 CPU 用实际已导出的 ONNX 五输出执行共享评估口径，禁止外部权重和扩展算子。
# 输入：
#   raw、config、samples：已核对摘要的模型、CPU 配置和独立测试样本。
# 输出：
#   metrics：实际 ONNX 的留出指标，不修改任何权重。
def evaluate_bytes(raw, config, samples):
    if config.device != "cpu":
        raise ValueError("VISION_FROZEN_EVALUATION_REQUIRES_CPU")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    graph = onnx.load_model_from_string(raw)
    outputs = ["visual_features", "semantic_logits", "traversability_logits", "scene_logits",
               "quality_logits"]
    validate_embedded_graph(raw, input_names=["forward_rgb"], output_names=outputs)
    if graph.functions or any(op.domain not in ("", "ai.onnx") for op in graph.opset_import):
        raise ValueError("VISION_FROZEN_ONNX_EXTENSION_NOT_ALLOWED")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if (len(inputs) != 1 or inputs[0].name != "forward_rgb"
            or inputs[0].shape != [1, 3, config.height, config.width]
            or inputs[0].type != "tensor(float)"
            or {output.name for output in session.get_outputs()} != set(outputs)
            or any(output.type != "tensor(float)" for output in session.get_outputs())):
        raise ValueError("VISION_FROZEN_ONNX_CONTRACT_MISMATCH")

    class FrozenOnnx(torch.nn.Module):
        # 功能：
        #   将共享评估器的 CPU 图像送入固定 ONNX 会话，逐输出检查尺寸和有限数值。
        # 输入：
        #   image：已经按训练契约归一化的单张前向 RGB 张量。
        # 输出：
        #   tensors：五个实际模型输出，不能参与梯度更新。
        def forward(self, image):
            values = session.run(outputs, {"forward_rgb": image.detach().cpu().numpy()})
            shapes = [(1, 139), (1, 8, config.height, config.width), (1, 1), (1, 6), (1, 4)]
            shapes[0] = (1, config.embedding_feature_count + 11)
            if any(value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all()
                   for value, shape in zip(values, shapes, strict=True)):
                raise ValueError("VISION_FROZEN_ONNX_OUTPUT_INVALID")
            tensors = tuple(torch.from_numpy(value) for value in values)
            return tensors

    metrics = evaluate_local_vision_model(FrozenOnnx(), samples, config)
    return metrics


# 功能：
#   1. 固定已通过训练门控的模型及三个清单，确认测试集未混入训练和早停验证。
#   2. 执行冻结评估并复查所有输入摘要；结果只记录、不自动改阈值或重新训练。
# 输入：
#   命令行参数：训练回执、实际 ONNX、三个数据根和新评估回执。
# 输出：
#   exit_code：评估完成为零；不等同测试通过或飞行资格。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-receipt", "model", "training-root", "validation-root", "test-root",
                 "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source = read_runtime_object(args.training_receipt)
    if (source.get("architecture") != LOCAL_VISION_ARCHITECTURE
            or source.get("training_accepted") is not True):
        raise ValueError("VISION_FROZEN_TRAINING_NOT_ACCEPTED")
    datasets, digests, roots = {}, {}, {}
    for split in ("training", "validation", "test"):
        root = getattr(args, split + "_root")
        datasets[split], digests[split] = read_vision_manifest(root / "samples.jsonl")
        roots[split] = root
    require_disjoint_vision_splits(datasets)
    for split in ("training", "validation"):
        if digests[split] != source.get(split + "_data_sha256"):
            raise ValueError("VISION_FROZEN_SOURCE_BINDING_MISMATCH")
    config = LocalVisionTrainingConfig.model_validate({**source["config"], "device": "cpu"})
    raw = read_plugin_file(args.model, limit=256 * 1024**2)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source.get("artifact_sha256"):
        raise ValueError("VISION_FROZEN_MODEL_HASH_MISMATCH")
    coverage = inspect_vision_split(args.test_root, datasets["test"])
    samples = resolve_local_vision_samples(args.test_root, datasets["test"])
    metrics = evaluate_bytes(raw, config, samples)
    for split, root in roots.items():
        if hash_plugin_file(root / "samples.jsonl", limit=256 * 1024**2) != digests[split]:
            raise ValueError("VISION_FROZEN_MANIFEST_CHANGED")
    inspect_vision_split(args.test_root, datasets["test"])
    report = {"schema": "dronedream.frozen-vision-evaluation.v1", "artifact_sha256": digest,
              "manifest_sha256": digests, "metrics": metrics.model_dump(mode="json"),
              "test_coverage": coverage, "weights_updated": False,
              "evaluation_completed": True, "qualification_granted": False}
    publish_runtime_json(args.receipt, report, replace_existing=False)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
