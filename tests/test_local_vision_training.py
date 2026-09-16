from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from dronedream_agent_core.local_vision_training import (
    LocalVisionTrainingConfig,
    LocalVisionTrainingSample,
    benchmark_local_vision_onnx,
    build_local_vision_model,
    count_trainable_parameters,
    export_local_vision_onnx,
    load_local_vision_label_map,
    local_vision_feature_count,
    local_vision_semantic_class_weights,
    resolve_local_vision_samples,
    train_local_vision_model,
)


# 功能：
#   为隔离测试生成的小图像计算实际字节摘要。
# 输入：
#   path：测试制品路径。
# 输出：
#   digest：实际内容摘要。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


# 功能：
#   生成四张真实编码的合成 RGB 和掩码，建立可重复的小规模训练数据。
# 输入：
#   root：隔离测试目录。
# 输出：
#   samples：带图像、掩码摘要及监督目标的测试样本列表。
def _dataset(root: Path) -> list[LocalVisionTrainingSample]:
    samples = []
    for index in range(4):
        image_path = root / f"image-{index}.png"
        mask_path = root / f"mask-{index}.png"
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :, 1] = 180 if index % 2 == 0 else 30
        pixels[:, 32:, 0] = 220 if index % 2 else 20
        mask = np.full((64, 64), 1 if index % 2 == 0 else 2, dtype=np.uint8)
        with Image.fromarray(pixels) as image:
            image.save(image_path)
        with Image.fromarray(mask) as labels:
            labels.save(mask_path)
        samples.append(
            LocalVisionTrainingSample(
                flight_id="flight-safe" if index < 2 else "flight-risky",
                map_sha256=("a" if index < 2 else "b") * 64,
                image_relative_path=image_path.name,
                image_sha256=_sha256(image_path),
                semantic_mask_relative_path=mask_path.name,
                semantic_mask_sha256=_sha256(mask_path),
                traversability_target=1.0 if index % 2 == 0 else 0.0,
                scene_targets=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                quality_targets=[0.0, 0.0, 0.0, 0.0],
            )
        )
    return samples


# 功能：
#   在四张合成图上完成一轮本地训练、实际 ONNX 导出和 CPU 推理，验证接口链路。
# 输入：
#   tmp_path：独立测试目录，不写产品模型包或触发预训练下载。
# 输出：
#   None：不返回业务数据。
def test_trains_exports_and_benchmarks_forward_vision_expert(tmp_path: Path) -> None:
    config = LocalVisionTrainingConfig(
        width=64,
        height=64,
        embedding_feature_count=32,
        epoch_count=1,
        batch_size=2,
        freeze_backbone_epochs=1,
    )
    resolved = resolve_local_vision_samples(tmp_path, _dataset(tmp_path))
    class_weights = local_vision_semantic_class_weights(resolved, config)
    model = build_local_vision_model(config, pretrained_backbone=False)

    model, metrics = train_local_vision_model(model, resolved, config)
    output_path = tmp_path / "forward-vision.onnx"
    artifact_sha256 = export_local_vision_onnx(model, output_path, config)
    benchmark = benchmark_local_vision_onnx(output_path, config, iterations=5)

    assert metrics.sample_count == 4
    assert metrics.semantic_sample_count == 4
    assert metrics.mean_loss >= 0.0
    assert metrics.semantic_class_target_pixels[1] > 0
    assert metrics.semantic_class_target_pixels[2] > 0
    assert len(metrics.semantic_class_iou) == 8
    assert metrics.semantic_mean_iou is not None
    assert class_weights[1] == pytest.approx(1.0)
    assert class_weights[2] == pytest.approx(1.0)
    assert class_weights[0] == 0.0
    assert 3_000_000 <= count_trainable_parameters(model) <= 15_000_000
    assert len(artifact_sha256) == 64
    assert benchmark["provider"] == ["CPUExecutionProvider"]
    assert benchmark["visual_feature_count"] == local_vision_feature_count(32)
    assert benchmark["p99_inference_latency_ms"] > 0.0
    assert benchmark["measured_outputs"] == ["visual_features"]
    assert benchmark["includes_preprocessing"] is False
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    assert {item.name for item in session.get_inputs()} == {"forward_rgb"}
    assert "visual_features" in {item.name for item in session.get_outputs()}


# 功能：
#   验证样本实际图像被修改后不能通过摘要准入。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_vision_dataset_rejects_content_drift(tmp_path: Path) -> None:
    samples = _dataset(tmp_path)
    (tmp_path / samples[0].image_relative_path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="image hash mismatch"):
        resolve_local_vision_samples(tmp_path, samples)


# 功能：
#   校验固定类别顺序、编号和实际文件摘要的加载结果。
# 输入：
#   tmp_path：隔离标签定义目录。
# 输出：
#   None：不返回业务数据。
def test_loads_exact_simulator_label_map_contract(tmp_path: Path) -> None:
    path = tmp_path / "vision-label-map.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.vision-label-map.v1",
                "background_class_id": 0,
                "classes": [
                    {"class_id": index, "name": name}
                    for index, name in enumerate(
                        (
                            "background",
                            "traversable",
                            "obstacle",
                            "doorway",
                            "stairs",
                            "person",
                            "glass",
                            "pickup-marker",
                        )
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    digest, class_ids = load_local_vision_label_map(path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert class_ids == frozenset(range(8))


# 功能：
#   拒绝类别名称与编号关系被调整的监督契约，防止训练标签含义漂移。
# 输入：
#   tmp_path：隔离标签定义目录。
# 输出：
#   None：不返回业务数据。
def test_rejects_reordered_simulator_label_contract(tmp_path: Path) -> None:
    path = tmp_path / "vision-label-map.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.vision-label-map.v1",
                "background_class_id": 0,
                "classes": [
                    {"class_id": 0, "name": "obstacle"},
                    {"class_id": 1, "name": "background"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="classes"):
        load_local_vision_label_map(path)
