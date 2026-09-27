"""Bounded vision manifest and split inspection before any paid compute is started."""

from __future__ import annotations

import hashlib
import os
import stat
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from dronedream_agent_core.local_vision_training import (
    LOCAL_VISION_SEMANTIC_CLASSES,
    LocalVisionTrainingSample,
    resolve_local_vision_samples,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json

MAX_DATASET_BYTES = 256 * 1024**2
MAX_SAMPLES = 100_000


# 功能：
#   从同一有界字节流读取完整清单并计算摘要，拒绝重复键、截断和读取时替换。
# 输入：
#   path：普通 JSONL 清单文件。
# 输出：
#   samples、digest：类型化视觉样本及实际输入摘要。
def read_vision_manifest(path: Path):
    check_plain_plugin_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_DATASET_BYTES:
        raise ValueError("LOCAL_VISION_DATASET_SIZE_OR_TYPE_INVALID")
    samples, hasher, total = [], hashlib.sha256(), 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not os.path.samestat(before, opened) or before.st_size != opened.st_size:
            raise ValueError("LOCAL_VISION_DATASET_CHANGED")
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
    if (total != opened.st_size or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns or not os.path.samestat(after, current)
            or current.st_size != after.st_size or current.st_mtime_ns != after.st_mtime_ns):
        raise ValueError("LOCAL_VISION_DATASET_CHANGED")
    if not samples:
        raise ValueError("local vision dataset is empty")
    digest = hasher.hexdigest()
    return samples, digest


# 功能：
#   解码同一摘要绑定的 RGB 和标签图，检查尺寸、类别及有监督覆盖，不导入 Torch。
# 输入：
#   root、samples：指定划分及其已解析样本。
#   allow_duplicates：仅在原始采集核查阶段统计重复；正式训练预检必须保持 False。
# 输出：
#   report：实际像素、标签和唯一图片数量，用于租卡前发现数据缺口。
def inspect_vision_split(root: Path, samples: list, *, allow_duplicates=False) -> dict:
    if type(allow_duplicates) is not bool:
        raise ValueError("VISION_AUDIT_DUPLICATE_POLICY_INVALID")
    resolved = resolve_local_vision_samples(root, samples)
    counts = np.zeros(len(LOCAL_VISION_SEMANTIC_CLASSES), dtype=np.int64)
    seen, labelled, duplicates, supervised_masks = set(), 0, 0, 0
    supervision = np.zeros(10, dtype=np.int64)
    for sample, image_path, mask_path in resolved:
        raw = read_plugin_file(image_path, limit=16 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != sample.image_sha256:
            raise ValueError("VISION_AUDIT_IMAGE_CHANGED")
        with Image.open(BytesIO(raw), formats=["PNG"]) as image:
            if (image.mode != "RGB" or getattr(image, "n_frames", 1) != 1
                    or not 1 <= image.width <= 4096 or not 1 <= image.height <= 2160):
                raise ValueError("VISION_AUDIT_RGB_INVALID")
            dimensions = image.size
            image.load()
        if sample.image_sha256 in seen:
            duplicates += 1
            if not allow_duplicates:
                raise ValueError("VISION_AUDIT_DUPLICATE_IMAGE")
        seen.add(sample.image_sha256)
        scene_weights = [v * sample.perception_supervision_enabled
                         for v in sample.scene_target_weights]
        supervision += np.array([*scene_weights, *sample.quality_target_weights]) > 0
        if mask_path is None:
            continue
        raw = read_plugin_file(mask_path, limit=4 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != sample.semantic_mask_sha256:
            raise ValueError("VISION_AUDIT_MASK_CHANGED")
        with Image.open(BytesIO(raw), formats=["PNG"]) as mask:
            if (mask.mode not in ("L", "P") or getattr(mask, "n_frames", 1) != 1
                    or not 1 <= mask.width <= 4096 or not 1 <= mask.height <= 2160
                    or dimensions[0] * mask.height != dimensions[1] * mask.width):
                raise ValueError("VISION_AUDIT_MASK_INVALID")
            pixels = np.asarray(mask, dtype=np.uint8)
            if int(pixels.max()) >= len(counts):
                raise ValueError("VISION_AUDIT_CLASS_UNSUPPORTED")
            if sample.perception_supervision_enabled:
                counts += np.bincount(pixels.reshape(-1), minlength=len(counts))
                supervised_masks += 1
        labelled += 1
    report = {"samples": len(samples), "unique_images": len(seen), "labelled_samples": labelled,
              "duplicate_images": duplicates, "perception_supervised_samples": supervised_masks,
              "semantic_pixels": counts.tolist(), "auxiliary_label_counts": supervision.tolist(),
              "source_kinds": sorted({sample.source_kind for sample in samples})}
    return report


# 功能：
#   检查训练、验证、最终测试的整段及空间分组隔离，图像相同即拒绝跨集使用。
# 输入：
#   splits：按名称索引的样本集合。
# 输出：
#   None：检查通过不返回业务数据；来源泄漏时抛出异常。
def require_disjoint_vision_splits(splits: dict) -> None:
    flights, groups, images = {}, {}, {}
    for split, samples in splits.items():
        for sample in samples:
            keys = [(flights, sample.flight_id), (images, sample.image_sha256)]
            if sample.scene_group_id is not None:
                # 改光照或加人会改变地图摘要，不能因此把同一区域拆进另一个集合。
                keys.append((groups, sample.scene_group_id))
            for registry, key in keys:
                if key in registry and registry[key] != split:
                    raise ValueError("VISION_DATASET_SPLIT_LEAKAGE")
                registry[key] = split
