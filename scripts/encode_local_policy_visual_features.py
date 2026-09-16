#!/usr/bin/env python3
"""Bind recorded forward RGB frames into local-policy training tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from dronedream_agent_core.local_policy_packages import latency_percentile
from dronedream_agent_core.local_policy_port import select_onnx_execution_providers
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingSample,
    parse_training_samples,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.training.artifact_assembly import validate_embedded_graph
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)
from dronedream_agent_core.visual_control_input import forward_rgb_tensor

MAXIMUM_IMAGE_BYTES = 16 * 1024 * 1024
MAXIMUM_DATASET_BYTES = 256 * 1024 * 1024
MAXIMUM_FRAME_SCAN_BYTES = 4 * 1024 * 1024 * 1024
MAXIMUM_FRAME_ENTRIES = 250_000


# 功能：
#   复用公共最近秩统计，零百分位返回最小值，非法或空延迟证据直接拒绝。
# 输入：
#   values：非负有限的毫秒观测。
#   percent：零至一百的百分位。
# 输出：
#   latency：对应秩的实际观测值。
def _percentile(values: list[float], percent: float) -> float:
    latency = latency_percentile(values, percent)
    return latency


# 功能：
#   在明确指定的图像根内建立摘要索引，拒绝链接及无界扫描；实际编码时仍复核同一摘要。
# 输入：
#   roots：至多三十二个普通目录，遍历条目和读取总字节均受预算约束。
# 输出：
#   index：PNG 内容摘要到来源路径的映射，不代表视觉质量或观测时效。
def _frame_index(roots: list[Path]) -> dict[str, Path]:
    if not isinstance(roots, list) or not 1 <= len(roots) <= 32:
        raise ValueError("LOCAL_POLICY_VISUAL_FRAME_ROOTS_INVALID")
    index: dict[str, Path] = {}
    scanned = entries = 0
    visited = set()

    # 功能：
    #   扫描不可读目录时立即保留原错误，不静默略过可能存放已绑定图像的目录。
    # 输入：
    #   error：文件系统遍历失败。
    # 输出：
    #   None：函数直接抛出原异常。
    def fail_scan(error):
        raise error

    for root in roots:
        check_plain_plugin_path(root)
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("visual frame root must be a directory")
        for directory, subdirectories, filenames in os.walk(
            resolved, followlinks=False, onerror=fail_scan
        ):
            parent = Path(directory)
            check_plain_plugin_path(parent)
            entries += len(subdirectories) + len(filenames)
            if entries > MAXIMUM_FRAME_ENTRIES:
                raise ValueError("LOCAL_POLICY_VISUAL_FRAME_SCAN_LIMIT")
            # 先检查再进入子目录，不通过目录链接悄悄扩大来源范围。
            for name in subdirectories:
                check_plain_plugin_path(parent / name)
            for name in filenames:
                path = parent / name
                if path.suffix.casefold() != ".png" or path in visited:
                    continue
                visited.add(path)
                content = read_plugin_file(path, limit=MAXIMUM_IMAGE_BYTES)
                scanned += len(content)
                if scanned > MAXIMUM_FRAME_SCAN_BYTES:
                    raise ValueError("LOCAL_POLICY_VISUAL_FRAME_SCAN_LIMIT")
                digest = hashlib.sha256(content).hexdigest()
                existing = index.get(digest)
                if existing is not None:
                    original = read_plugin_file(existing, limit=MAXIMUM_IMAGE_BYTES)
                    scanned += len(original)
                    if scanned > MAXIMUM_FRAME_SCAN_BYTES:
                        raise ValueError("LOCAL_POLICY_VISUAL_FRAME_SCAN_LIMIT")
                    if original != content:
                        raise RuntimeError("LOCAL_POLICY_VISUAL_FRAME_IDENTITY_CHANGED")
                index.setdefault(digest, path)
    if not index:
        raise ValueError("visual frame roots contain no admissible PNG frames")
    return index


# 功能：
#   有界读取单张编码图像并核对来源摘要，调用与部署链共用的 RGB 归一化算法。
# 输入：
#   path：索引选定的 PNG 路径。
#   width：目标宽度。
#   height：目标高度。
#   normalization：训练和部署约定的归一化方式。
#   expected_sha256：样本记录的来源摘要；直接独立调用时可不提供。
# 输出：
#   tensor：单批次 1×3×height×width 的 float32 图像张量。
def _tensor(
    path: Path,
    *,
    width: int,
    height: int,
    normalization: str,
    expected_sha256: str | None = None,
):
    content = read_plugin_file(path, limit=MAXIMUM_IMAGE_BYTES)
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("LOCAL_POLICY_VISUAL_FRAME_CHANGED_DURING_ENCODING")
    tensor = forward_rgb_tensor(
        {"content_bytes": content}, width=width, height=height, normalization=normalization
    )
    return tensor


# 功能：
#   在创建推理会话前检查内嵌权重及 RGB 输入／特征输出的类型、秩和已知维数。
# 输入：
#   content：固定且至多 256 MiB 的编码器字节。
#   width：约定图像宽度。
#   height：约定图像高度。
#   feature_count：约定输出向量长度。
# 输出：
#   None：不返回业务数据。
def _validate_encoder(content: bytes, *, width: int, height: int, feature_count: int) -> None:
    import onnx

    document = onnx.load_model_from_string(content)
    inputs, outputs = document.graph.input, document.graph.output
    validate_embedded_graph(
        content, input_names=["forward_rgb"], output_names=[item.name for item in outputs]
    )
    matches = [item for item in outputs if item.name == "visual_features"]
    if len(matches) != 1:
        raise ValueError("perception encoder tensor contract is incompatible")

    # 功能：
    #   核验 float32 张量声明；动态轴可由实际执行确定，但静态轴不能与契约冲突。
    # 输入：
    #   value：ONNX 输入或输出信息。
    #   expected：允许的单一布局维数。
    # 输出：
    #   valid：已声明类型及已知维数是否匹配。
    def compatible(value, expected):
        tensor = value.type.tensor_type
        valid = (
            value.type.HasField("tensor_type")
            and tensor.elem_type == onnx.TensorProto.FLOAT
            and tensor.HasField("shape")
            and len(tensor.shape.dim) == len(expected)
            and all(
                not axis.HasField("dim_value") or axis.dim_value == size
                for axis, size in zip(tensor.shape.dim, expected, strict=True)
            )
        )
        return valid

    if not compatible(inputs[0], (1, 3, height, width)) or not (
        compatible(matches[0], (feature_count,)) or compatible(matches[0], (1, feature_count))
    ):
        raise ValueError("perception encoder tensor shape or type is incompatible")


# 功能：
#   1. 固定严格样本和编码器字节，按样本摘要寻找原图并生成实际视觉特征及分项耗时。
#   2. 数据与回执分别无覆盖发布；回执竞争可能留下无回执数据，不授予训练或飞行资格。
# 输入：
#   无：命令行提供来源样本、图像根、编码器、预处理契约及新输出路径。
# 输出：
#   status：实际编码数据和对应回执均发布成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-data", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, action="append", required=True)
    parser.add_argument("--perception-encoder", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--visual-feature-count", type=int, required=True)
    parser.add_argument(
        "--normalization",
        choices=("zero-to-one", "minus-one-to-one", "imagenet"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for name in ("policy_data", "perception_encoder", "output", "receipt"):
        check_plain_plugin_path(getattr(args, name))
        setattr(args, name, getattr(args, name).resolve())
    if args.output.is_relative_to(args.receipt) or args.receipt.is_relative_to(args.output):
        raise ValueError("visual dataset and receipt must use distinct non-nested paths")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("visual policy dataset output already exists")
    if not 32 <= args.width <= 1_024 or not 32 <= args.height <= 1_024:
        parser.error("visual dimensions are outside the package contract")
    if not 1 <= args.visual_feature_count <= 65_536:
        parser.error("visual feature count is outside the package contract")

    import numpy as np
    import onnxruntime as ort

    source_content = read_plugin_file(args.policy_data, limit=MAXIMUM_DATASET_BYTES)
    samples = parse_training_samples(source_content)
    if any(sample.source_visual_sha256 is None for sample in samples):
        raise RuntimeError("LOCAL_POLICY_SAMPLE_HAS_NO_VISUAL_EVIDENCE")
    if len(samples) * args.visual_feature_count * 4 > MAXIMUM_DATASET_BYTES:
        raise ValueError("LOCAL_POLICY_VISUAL_DATASET_SIZE_INVALID")
    frames = _frame_index(args.frame_root)
    encoder_content = read_plugin_file(args.perception_encoder, limit=256 * 1024 * 1024)
    _validate_encoder(
        encoder_content,
        width=args.width,
        height=args.height,
        feature_count=args.visual_feature_count,
    )
    available = ort.get_available_providers()
    providers = select_onnx_execution_providers(available)
    session_options = ort.SessionOptions()
    # Dataset preparation may run beside a closed-loop simulator.  A default
    # ONNX Runtime thread pool can briefly occupy every host core and distort
    # the flight package's tail-latency evidence, so use the same bounded
    # single-thread execution policy as the production local expert backend.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        encoder_content,
        sess_options=session_options,
        providers=providers,
    )
    encoded = bytearray()
    preprocess_latencies = []
    inference_latencies = []
    source_hashes = []
    for sample in samples:
        digest = sample.source_visual_sha256
        if digest is None:
            raise RuntimeError("LOCAL_POLICY_SAMPLE_HAS_NO_VISUAL_EVIDENCE")
        path = frames.get(digest)
        if path is None:
            raise RuntimeError("LOCAL_POLICY_BOUND_VISUAL_FRAME_MISSING")
        started = time.perf_counter()
        tensor = _tensor(
            path,
            width=args.width,
            height=args.height,
            normalization=args.normalization,
            expected_sha256=digest,
        )
        preprocess_latencies.append((time.perf_counter() - started) * 1_000.0)
        started = time.perf_counter()
        outputs = session.run(["visual_features"], {"forward_rgb": tensor})
        inference_latencies.append((time.perf_counter() - started) * 1_000.0)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("LOCAL_POLICY_VISUAL_FEATURE_OUTPUT_INVALID")
        output = outputs[0]
        if (
            not isinstance(output, np.ndarray)
            or output.dtype != np.dtype(np.float32)
            or output.shape not in {(args.visual_feature_count,), (1, args.visual_feature_count)}
            or not np.isfinite(output).all()
        ):
            raise RuntimeError("LOCAL_POLICY_VISUAL_FEATURE_OUTPUT_INVALID")
        # 不通过类型转换或任意展平掩盖错误批次；加入特征后重新校验整条训练样本。
        payload = sample.model_dump(mode="json")
        payload["visual_features"] = output.reshape(-1).tolist()
        observed = LocalPolicyTrainingSample.model_validate(payload)
        row = (observed.model_dump_json() + "\n").encode("utf-8")
        if len(row) > 4 * 1024 * 1024 or len(encoded) + len(row) > MAXIMUM_DATASET_BYTES:
            raise ValueError("LOCAL_POLICY_VISUAL_DATASET_SIZE_INVALID")
        encoded.extend(row)
        source_hashes.append(digest)

    output_content = bytes(encoded)
    receipt = {
        "schema_version": "dronedream.local-policy-visual-encoding-receipt.v1",
        "source_policy_data_sha256": hashlib.sha256(source_content).hexdigest(),
        "perception_encoder_sha256": hashlib.sha256(encoder_content).hexdigest(),
        "provider_chain": session.get_providers(),
        "sample_count": len(samples),
        "unique_visual_frame_count": len(set(source_hashes)),
        "visual_width": args.width,
        "visual_height": args.height,
        "visual_feature_count": args.visual_feature_count,
        "visual_normalization": args.normalization,
        "p95_preprocess_latency_ms": _percentile(preprocess_latencies, 95.0),
        "p99_preprocess_latency_ms": _percentile(preprocess_latencies, 99.0),
        "p95_encoder_latency_ms": _percentile(inference_latencies, 95.0),
        "p99_encoder_latency_ms": _percentile(inference_latencies, 99.0),
        "output_sha256": hashlib.sha256(output_content).hexdigest(),
        "qualification_granted": False,
    }
    # 所有张量和耗时验证完再发布，不把完成前的半份数据暴露为可消费文件。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    publish_evidence_bytes(args.output, output_content, limit=MAXIMUM_DATASET_BYTES)
    write_evidence_object(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
