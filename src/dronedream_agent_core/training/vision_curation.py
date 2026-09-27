"""Predeclared image-only curation; never inspect model predictions or test scores."""

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from dronedream_agent_core.plugin_files import read_plugin_file

CURATION_POLICY = {
    "schema": "dronedream.vision-curation-policy.v1",
    "exact_duplicate_images": "keep-first-in-explicit-source-order",
    "uninformative_perception_std_threshold": 2.0,
    "uninformative_single_class_fraction": 0.995,
    "quality_only_exposure_samples_retained": True,
    "predictions_or_evaluation_scores_used": False,
}


# 功能：
#   使用预先固定的规则剔除重复图和没有空间信息的单色单类别图，不查看模型成绩。
# 输入：
#   root、sample、seen：当前原生采集划分、来源样本及已接纳的图像摘要集合。
# 输出：
#   decision：保留决定、客观统计和拒绝原因；原始图片不删除。
def curate_render_sample(root, sample, seen):
    decision = {"flight_id": sample.flight_id, "image_sha256": sample.image_sha256,
                "retained": False, "reason": None}
    if sample.image_sha256 in seen:
        decision["reason"] = "EXACT_DUPLICATE_RGB"
        return decision
    raw = read_plugin_file(root / sample.image_relative_path, limit=16 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != sample.image_sha256:
        raise ValueError("VISION_CURATION_RGB_CHANGED")
    with Image.open(BytesIO(raw)) as image:
        rgb = np.asarray(image, dtype=np.float32)
        deviation = float(rgb.std(axis=(0, 1)).mean())
    decision["rgb_spatial_std"] = round(deviation, 6)
    if sample.perception_supervision_enabled and deviation < 2.0:
        mask_raw = read_plugin_file(root / sample.semantic_mask_relative_path, limit=4 * 1024**2)
        if hashlib.sha256(mask_raw).hexdigest() != sample.semantic_mask_sha256:
            raise ValueError("VISION_CURATION_MASK_CHANGED")
        with Image.open(BytesIO(mask_raw)) as mask:
            pixels = np.asarray(mask, dtype=np.uint8)
            largest = int(np.bincount(pixels.reshape(-1), minlength=8).max())
            dominant = largest / pixels.size
        decision["dominant_class_fraction"] = round(dominant, 6)
        if dominant >= 0.995:
            decision["reason"] = "NO_SPATIAL_INFORMATION_SINGLE_CLASS"
            return decision
    seen.add(sample.image_sha256)
    decision["retained"] = True
    decision["reason"] = "RETAINED"
    return decision
