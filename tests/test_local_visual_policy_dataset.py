from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample


# 功能：
#   对本测试生成的小文件计算摘要，用于独立核对入口的内容绑定。
# 输入：
#   path：测试生成的图像或输出数据文件。
# 输出：
#   digest：实际文件字节的 SHA-256。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# 功能：
#   写出可由 ONNX Runtime 执行的像素均值图，仅验证编码链路，不模拟产品视觉语义能力。
# 输入：
#   path：本测试新建的 ONNX 文件。
# 输出：
#   None：不返回业务数据。
def _encoder(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["forward_rgb"],
                ["visual_features"],
                axes=[1, 2, 3],
                keepdims=0,
            )
        ],
        "visual-dataset-test-encoder",
        [helper.make_tensor_value_info("forward_rgb", TensorProto.FLOAT, [None, 3, 32, 32])],
        [helper.make_tensor_value_info("visual_features", TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = min(model.ir_version, 11)
    onnx.checker.check_model(model)
    onnx.save(model, path)


# 功能：
#   从真实 PNG 经共享预处理和实际 ONNX 求均值，读回样本及回执核对特征、数量和摘要。
# 输入：
#   tmp_path：测试原图、模型、输入数据及输出证据目录。
# 输出：
#   None：不返回业务数据。
def test_encodes_only_hash_bound_forward_frames(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    frame = frame_root / "forward.png"
    with Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)) as picture:
        picture.save(frame)
    encoder = tmp_path / "encoder.onnx"
    _encoder(encoder)
    source = tmp_path / "policy.jsonl"
    sample = LocalPolicyTrainingSample(
        source_snapshot_sha256="a" * 64,
        source_visual_sha256=_sha256(frame),
        state_features=[0.0] * 46,
        candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_action_index=0,
        risk_target=0.0,
    )
    source.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "visual-policy.jsonl"
    receipt_path = tmp_path / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "encode_local_policy_visual_features.py"),
            "--policy-data",
            str(source),
            "--frame-root",
            str(frame_root),
            "--perception-encoder",
            str(encoder),
            "--width",
            "32",
            "--height",
            "32",
            "--visual-feature-count",
            "1",
            "--normalization",
            "zero-to-one",
            "--output",
            str(output),
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    encoded = LocalPolicyTrainingSample.model_validate_json(
        output.read_text(encoding="utf-8").strip()
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert encoded.visual_features == [1.0]
    assert receipt["schema_version"] == ("dronedream.local-policy-visual-encoding-receipt.v1")
    assert receipt["sample_count"] == 1
    assert receipt["unique_visual_frame_count"] == 1
    assert receipt["p99_preprocess_latency_ms"] >= 0.0
    assert receipt["output_sha256"] == _sha256(output)
