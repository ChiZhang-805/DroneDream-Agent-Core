#!/usr/bin/env python3
"""Train, export, and gate the local forward RGB semantic expert."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext, suppress
from pathlib import Path
from tempfile import mkdtemp

from dronedream_agent_core.asset_package_storage import publish_asset_file
from dronedream_agent_core.local_vision_training import (
    LOCAL_VISION_ARCHITECTURE,
    LOCAL_VISION_EMBEDDING_SUPERVISION,
    LOCAL_VISION_SPLIT_METHOD,
    LocalVisionTrainingConfig,
    LocalVisionTrainingSample,
    benchmark_local_vision_onnx,
    build_local_vision_model,
    count_trainable_parameters,
    evaluate_local_vision_model,
    export_local_vision_onnx,
    local_vision_feature_count,
    local_vision_semantic_class_weights,
    local_vision_training_device,
    resolve_local_vision_samples,
    train_local_vision_model,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path, hash_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_initialization import VISION_INITIALIZATIONS
from dronedream_agent_core.training.vision_manifest import read_vision_manifest
from dronedream_agent_core.training.vision_session import (
    TrainingBudgetReached,
    VisionTrainingSession,
)
from dronedream_plugin_sdk.protocol import encode_json

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
    samples, digest = read_vision_manifest(path)
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
#   exit_code：门控通过为 0，保留拒绝回执为 1，预算停止并保存检查点为 2。
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pretrained-source", choices=VISION_INITIALIZATIONS,
                        default="coco-voc-segmentation")
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--pretrained-sha256")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-batches", type=int, default=100)
    parser.add_argument("--max-training-seconds", type=float, default=3600.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
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
    if args.resume and args.run_directory is None:
        raise ValueError("LOCAL_VISION_RESUME_DIRECTORY_REQUIRED")
    if (args.pretrained_weights is None) != (args.pretrained_sha256 is None):
        raise ValueError("VISION_PRETRAINED_FILE_BINDING_REQUIRED")
    if args.pretrained_weights is not None:
        if args.allow_uninitialized_backbone or args.pretrained_source != "coco-voc-segmentation":
            raise ValueError("VISION_PRETRAINED_FILE_SOURCE_MISMATCH")
        file_hash = hash_plugin_file(args.pretrained_weights, limit=MAX_MODEL_BYTES)
        if file_hash != args.pretrained_sha256:
            raise ValueError("VISION_PRETRAINED_FILE_HASH_MISMATCH")
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
    training_groups = {sample.scene_group_id for sample in training_samples
                       if sample.scene_group_id is not None}
    validation_groups = {sample.scene_group_id for sample in validation_samples
                         if sample.scene_group_id is not None}
    if training_groups & validation_groups:
        raise RuntimeError("LOCAL_VISION_SPATIAL_GROUP_SPLIT_OVERLAP")

    config = LocalVisionTrainingConfig(
        device=args.device,
        pretrained_source=args.pretrained_source,
        width=args.width,
        height=args.height,
        embedding_feature_count=args.embedding_feature_count,
        epoch_count=args.epoch_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    # 在可能下载预训练权重之前确认训练设备；不能把 CUDA 请求静默降为 CPU。
    local_vision_training_device(config)
    training = resolve_local_vision_samples(args.training_root, training_samples)
    validation = resolve_local_vision_samples(args.validation_root, validation_samples)
    semantic_class_weights = local_vision_semantic_class_weights(training, config)
    session = None
    if args.run_directory is not None:
        # 使用真实训练实现摘要绑定恢复点；修改实现之后不复用旧优化器状态。
        import dronedream_agent_core.local_vision_training as training_module
        import dronedream_agent_core.training.vision_initialization as initialization_module
        import dronedream_agent_core.training.vision_manifest as manifest_module
        import dronedream_agent_core.training.vision_session as session_module

        binding = {
            "architecture": LOCAL_VISION_ARCHITECTURE, "config": config.model_dump(mode="json"),
            "training_sha256": training_digest, "validation_sha256": validation_digest,
            "allow_uninitialized": args.allow_uninitialized_backbone,
            "pretrained_file_sha256": args.pretrained_sha256,
            "patience": args.early_stop_patience,
            "implementation": {
                **{module.__name__: hash_plugin_file(Path(module.__file__), limit=1024**2)
                   for module in (training_module, session_module, initialization_module,
                                  manifest_module)},
                "entrypoint": hash_plugin_file(Path(__file__), limit=1024**2),
            },
        }
        session = VisionTrainingSession(args.run_directory, binding, resume=args.resume,
            checkpoint_batches=args.checkpoint_batches, max_seconds=args.max_training_seconds,
            patience=args.early_stop_patience)
    model = build_local_vision_model(config,
        pretrained_backbone=not args.allow_uninitialized_backbone and not args.resume,
        weights_path=args.pretrained_weights if not args.resume else None,
        weights_sha256=args.pretrained_sha256 if not args.resume else None)
    try:
        with session.writer() if session else nullcontext():
            if session is None:
                model, training_metrics = train_local_vision_model(model, training, config)
            else:
                model, training_metrics = train_local_vision_model(model, training, config,
                    session=session, validation_samples=validation)
            # 验证、导出与回执发布仍属于本次作业，不能提前释放锁让恢复进程抢跑。
            return _publish_training_result(args, model, config, (training, validation),
                                            manifests, training_metrics, semantic_class_weights,
                                            session)
    except TrainingBudgetReached:
        print(json.dumps({"status": "paused-at-completed-batch", "checkpoint": session.last_record,
                          "training_accepted": False, "flight_qualification_granted": False}))
        return 2


# 功能：
#   在调用者持有作业锁期间评估和导出完成的网络，复查来源后发布模型及最后回执。
# 输入：
#   args、model、config：已校验的输出参数、完成的网络及训练配置。
#   datasets、manifests：训练和验证样本路径及原始清单摘要。
#   training_metrics、semantic_class_weights、session：训练指标、类别权重及可选会话。
# 输出：
#   exit_code：通过全部模型门控为零；拒绝回执为一，不授予飞行资格。
def _publish_training_result(args, model, config, datasets, manifests, training_metrics,
                             semantic_class_weights, session):
    training, validation = datasets
    training_samples = [entry[0] for entry in training]
    validation_samples = [entry[0] for entry in validation]
    training_digest, validation_digest = (entry[1] for entry in manifests)
    training_flights = {sample.flight_id for sample in training_samples}
    validation_flights = {sample.flight_id for sample in validation_samples}
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
        for split_name, samples in (("TRAINING", training_samples),
                                     ("VALIDATION", validation_samples)):
            coverage = [sum([*sample.scene_target_weights, *sample.quality_target_weights][i] > 0
                            for sample in samples) for i in range(10)]
            if any(count == 0 for count in coverage):
                issues.append(f"LOCAL_VISION_{split_name}_AUXILIARY_SUPERVISION_MISSING")
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
        # 背景以外的所有类别都直接影响导航；平均 IoU 不能掩盖障碍或可通行面的失效。
        critical_class_ious = validation_metrics.semantic_class_iou[1:]
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
            "architecture": LOCAL_VISION_ARCHITECTURE,
            "training_device": config.device,
            "embedding_supervision": LOCAL_VISION_EMBEDDING_SUPERVISION,
            "backbone_initialization": model.initialization_record["source"],
            "initialization": model.initialization_record,
            "session_checkpoint": session.last_record if session else None,
            "selected_validation_loss": session.best_score if session else None,
            "training_data_sha256": training_digest,
            "validation_data_sha256": validation_digest,
            "training_flight_count": len(training_flights),
            "validation_flight_count": len(validation_flights),
            "split_method": LOCAL_VISION_SPLIT_METHOD,
            "source_kinds": sorted({sample.source_kind for sample in training_samples}),
            "spatial_group_disjoint": True,
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
