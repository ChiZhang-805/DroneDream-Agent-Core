#!/usr/bin/env python3
"""Train, export, and gate the local forward RGB semantic expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp

from dronedream_agent_core.asset_package_storage import publish_asset_file
from dronedream_agent_core.local_vision_training import (
    LocalVisionTrainingConfig,
    LocalVisionTrainingSample,
    benchmark_local_vision_onnx,
    build_local_vision_model,
    count_trainable_parameters,
    evaluate_local_vision_model,
    export_local_vision_onnx,
    local_vision_feature_count,
    local_vision_semantic_class_weights,
    resolve_local_vision_samples,
    train_local_vision_model,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path, hash_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json, encode_json

MAX_DATASET_BYTES = 256 * 1024**2
MAX_SAMPLES = 100_000
MAX_MODEL_BYTES = 256 * 1024**2


# 功能：
#   以有界普通文件流同时解析样本并计算清单摘要，拒绝截断、重复键和读取期间替换。
# 输入：
#   path：换行分隔的训练或验证清单。
# 输出：
#   samples、digest：实际读取的类型化样本及同一字节流的摘要。
def _samples(path: Path) -> tuple[list[LocalVisionTrainingSample], str]:
    check_plain_plugin_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_DATASET_BYTES:
        raise ValueError("LOCAL_VISION_DATASET_SIZE_OR_TYPE_INVALID")
    samples = []
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not os.path.samestat(before, opened) or before.st_size != opened.st_size:
            raise ValueError("LOCAL_VISION_DATASET_CHANGED")
        # 逐行读取本身带上限，不先为异常长行分配任意大小的内存。
        while line := stream.readline(MAX_MESSAGE_BYTES + 1):
            total += len(line)
            if total > min(opened.st_size, MAX_DATASET_BYTES):
                raise ValueError("LOCAL_VISION_DATASET_CHANGED_OR_OVERSIZED")
            if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                raise ValueError("LOCAL_VISION_DATASET_LINE_INVALID")
            hasher.update(line)
            if not line.strip():
                continue
            if len(samples) >= MAX_SAMPLES:
                raise ValueError("LOCAL_VISION_DATASET_SAMPLE_BUDGET_EXCEEDED")
            samples.append(LocalVisionTrainingSample.model_validate(decode_json(line)))
        after = os.fstat(stream.fileno())
    check_plain_plugin_path(path)
    current = path.stat()
    if (
        total != opened.st_size
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or not os.path.samestat(after, current)
        or current.st_size != after.st_size
        or current.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("LOCAL_VISION_DATASET_CHANGED")
    if not samples:
        raise ValueError("local vision dataset is empty")
    digest = hasher.hexdigest()
    return samples, digest


# 功能：
#   发布前复核清单与所有已解析图像／掩码的实际摘要，防止训练过程中的文件漂移。
# 输入：
#   manifests：解析时已固定路径和摘要的清单集合。
#   datasets：训练和验证的已解析样本集合。
# 输出：
#   None：不返回业务数据。
def _verify_sources(manifests: tuple, datasets: tuple) -> None:
    for path, expected in manifests:
        if hash_plugin_file(path, limit=MAX_DATASET_BYTES) != expected:
            raise ValueError("LOCAL_VISION_DATASET_CHANGED")
    # 相同路径只读一次；若标签为同一路径声明冲突摘要，仍需明确拒绝。
    verified = {}
    for dataset in datasets:
        for sample, image, mask in dataset:
            for path, expected, limit in (
                (image, sample.image_sha256, 16 * 1024**2),
                (mask, sample.semantic_mask_sha256, 4 * 1024**2),
            ):
                if path is None:
                    continue
                if path not in verified:
                    verified[path] = hash_plugin_file(path, limit=limit)
                if verified[path] != expected:
                    raise ValueError("LOCAL_VISION_SOURCE_ARTIFACT_CHANGED")


# 功能：
#   1. 按完整飞行划分执行训练、留出验证和 CPU 基准测试，不授予飞行资格。
#   2. 无覆盖发布来源绑定的模型与回执；只有成功回执代表本次发布完成。
#   3. 异常时不删除已发布模型或未知暂存内容，保留可恢复现场。
# 输入：
#   命令行参数：清单与制品路径、训练超参数、评测轮数和开发初始化开关。
# 输出：
#   exit_code：门控通过为 0，保留拒绝回执时为 1。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--embedding-feature-count", type=int, default=128)
    parser.add_argument("--epoch-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--random-seed", type=int, default=805)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument(
        "--allow-uninitialized-backbone",
        action="store_true",
        help="Development smoke test only; an uninitialized backbone cannot be admitted.",
    )
    args = parser.parse_args()
    # 固定绝对路径但不解析掉符号链接，所有检查先于依赖加载和可能的权重下载。
    for name in (
        "training_root",
        "training_data",
        "validation_root",
        "validation_data",
        "output_model",
        "training_receipt",
    ):
        path = getattr(args, name).absolute()
        check_plain_plugin_path(path)
        setattr(args, name, path)
    if args.output_model == args.training_receipt:
        raise ValueError("LOCAL_VISION_OUTPUT_PATHS_OVERLAP")
    if args.output_model.exists() or args.training_receipt.exists():
        raise FileExistsError("local vision output already exists")
    if not 5 <= args.benchmark_iterations <= 10_000:
        raise ValueError("LOCAL_VISION_BENCHMARK_ITERATIONS_INVALID")

    training_samples, training_digest = _samples(args.training_data)
    validation_samples, validation_digest = _samples(args.validation_data)
    if len(training_samples) + len(validation_samples) > MAX_SAMPLES:
        raise ValueError("LOCAL_VISION_COMBINED_SAMPLE_BUDGET_EXCEEDED")
    manifests = ((args.training_data, training_digest), (args.validation_data, validation_digest))
    training_flights = {sample.flight_id for sample in training_samples}
    validation_flights = {sample.flight_id for sample in validation_samples}
    if training_flights & validation_flights:
        raise RuntimeError("LOCAL_VISION_COMPLETE_FLIGHT_SPLIT_OVERLAP")
    training_images = {sample.image_sha256 for sample in training_samples}
    validation_images = {sample.image_sha256 for sample in validation_samples}
    if training_images & validation_images:
        raise RuntimeError("LOCAL_VISION_IMAGE_CONTENT_SPLIT_OVERLAP")

    config = LocalVisionTrainingConfig(
        width=args.width,
        height=args.height,
        embedding_feature_count=args.embedding_feature_count,
        epoch_count=args.epoch_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    training = resolve_local_vision_samples(args.training_root, training_samples)
    validation = resolve_local_vision_samples(args.validation_root, validation_samples)
    semantic_class_weights = local_vision_semantic_class_weights(training, config)
    model = build_local_vision_model(
        config,
        pretrained_backbone=not args.allow_uninitialized_backbone,
    )
    model, training_metrics = train_local_vision_model(model, training, config)
    validation_metrics = evaluate_local_vision_model(model, validation, config)
    check_plain_plugin_path(args.output_model)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        mkdtemp(dir=args.output_model.parent, prefix=f".{args.output_model.name}.staging-")
    )
    directory_identity = temporary.stat()
    staged = temporary / "forward-vision.onnx"
    staged_identity = None
    try:
        artifact_sha256 = export_local_vision_onnx(model, staged, config)
        check_plain_plugin_path(staged)
        staged_identity = staged.stat()
        benchmark = benchmark_local_vision_onnx(
            staged,
            config,
            iterations=args.benchmark_iterations,
        )
        latency = benchmark.get("p99_inference_latency_ms")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("LOCAL_VISION_BENCHMARK_LATENCY_INVALID")
        issues = []
        if args.allow_uninitialized_backbone:
            issues.append("LOCAL_VISION_BACKBONE_UNINITIALIZED")
        if validation_metrics.sample_count < 20:
            issues.append("LOCAL_VISION_VALIDATION_SAMPLE_COUNT_TOO_LOW")
        if validation_metrics.semantic_sample_count < 20:
            issues.append("LOCAL_VISION_SEMANTIC_SAMPLE_COUNT_TOO_LOW")
        semantic_accuracy = validation_metrics.semantic_pixel_accuracy
        if semantic_accuracy is None or semantic_accuracy < 0.75:
            issues.append("LOCAL_VISION_SEMANTIC_PIXEL_ACCURACY_TOO_LOW")
        minimum_class_pixels = 100
        if any(
            count < minimum_class_pixels for count in training_metrics.semantic_class_target_pixels
        ):
            issues.append("LOCAL_VISION_TRAINING_SEMANTIC_CLASS_COVERAGE_INCOMPLETE")
        if any(
            count < minimum_class_pixels
            for count in validation_metrics.semantic_class_target_pixels
        ):
            issues.append("LOCAL_VISION_VALIDATION_SEMANTIC_CLASS_COVERAGE_INCOMPLETE")
        if (
            validation_metrics.semantic_mean_iou is None
            or validation_metrics.semantic_mean_iou < 0.5
        ):
            issues.append("LOCAL_VISION_SEMANTIC_MEAN_IOU_TOO_LOW")
        critical_class_ious = validation_metrics.semantic_class_iou[3:]
        if any(value is None or value < 0.3 for value in critical_class_ious):
            issues.append("LOCAL_VISION_CRITICAL_SEMANTIC_CLASS_IOU_TOO_LOW")
        if validation_metrics.traversability_mean_absolute_error > 0.2:
            issues.append("LOCAL_VISION_TRAVERSABILITY_ERROR_TOO_HIGH")
        if validation_metrics.scene_binary_accuracy < 0.85:
            issues.append("LOCAL_VISION_SCENE_ACCURACY_TOO_LOW")
        if validation_metrics.quality_binary_accuracy < 0.85:
            issues.append("LOCAL_VISION_QUALITY_ACCURACY_TOO_LOW")
        if latency > 100.0:
            issues.append("LOCAL_VISION_CPU_P99_EXCEEDS_HARD_BOUND")
        receipt = {
            "backbone_initialization": (
                "uninitialized-development-only"
                if args.allow_uninitialized_backbone
                else "mobilenet-v3-large-imagenet1k-v2"
            ),
            "training_data_sha256": training_digest,
            "validation_data_sha256": validation_digest,
            "training_flight_count": len(training_flights),
            "validation_flight_count": len(validation_flights),
            "split_method": "held-out-complete-flight",
            "config": config.model_dump(mode="json"),
            "parameter_count": count_trainable_parameters(model),
            "visual_feature_count": local_vision_feature_count(config.embedding_feature_count),
            "semantic_class_weights": semantic_class_weights,
            "artifact_sha256": artifact_sha256 if not issues else None,
            "training_metrics": training_metrics.model_dump(mode="json"),
            "validation_metrics": validation_metrics.model_dump(mode="json"),
            "benchmark": benchmark,
            "training_accepted": not issues,
            "issue_codes": issues,
            "flight_qualification_granted": False,
        }
        # 回执无法严格序列化时不先发布模型；后续发布仍自行验证实际字节及身份。
        encode_json(receipt)
        _verify_sources(manifests, (training, validation))
        check_plain_plugin_path(args.training_receipt)
        if args.training_receipt.exists():
            raise FileExistsError("local vision training receipt already exists")
        if not issues:
            publish_asset_file(
                staged, args.output_model, expected_sha256=artifact_sha256, limit=MAX_MODEL_BYTES
            )
        # 两个独立目标无法组成文件系统原子事务。回执最后写入；失败留下的模型不算交付。
        publish_runtime_json(args.training_receipt, receipt, replace_existing=False)
    finally:
        if staged_identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(staged)
                if os.path.samestat(staged_identity, staged.stat()):
                    staged.unlink()
        with suppress(OSError, ValueError):
            check_plain_plugin_path(temporary)
            if os.path.samestat(directory_identity, temporary.stat()):
                # 只删除空且仍归属本次操作的目录，不递归清理意外出现的资料。
                temporary.rmdir()
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    exit_code = 0 if not issues else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
