"""Model-input visibility coverage, distinct from native resolution pixel totals."""

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES
from dronedream_agent_core.plugin_files import read_plugin_file


# 功能：
#   用训练时完全相同的最近邻缩放检查目标是否仍可见，原图非法类别不能被缩放隐藏。
# 输入：
#   raw、width、height：已绑定摘要的标签 PNG 和模型实际输入尺寸。
# 输出：
#   original_counts、resized_counts：各类别在原图和模型输入中的像素数。
def mask_resolution_counts(raw, width, height):
    if (type(width) is not int or type(height) is not int
            or not 64 <= width <= 1024 or not 64 <= height <= 1024):
        raise ValueError("VISION_VISIBILITY_DIMENSIONS_INVALID")
    with Image.open(BytesIO(raw), formats=["PNG"]) as mask:
        if (mask.mode not in ("L", "P") or getattr(mask, "n_frames", 1) != 1
                or not 1 <= mask.width <= 4096 or not 1 <= mask.height <= 2160):
            raise ValueError("VISION_VISIBILITY_MASK_INVALID")
        original = np.asarray(mask, dtype=np.uint8)
        if int(original.max()) >= len(LOCAL_VISION_SEMANTIC_CLASSES):
            raise ValueError("VISION_VISIBILITY_CLASS_INVALID")
        with mask.resize((width, height), Image.Resampling.NEAREST) as resized:
            smaller = np.asarray(resized, dtype=np.uint8)
            resized_counts = np.bincount(smaller.reshape(-1), minlength=8)
        original_counts = np.bincount(original.reshape(-1), minlength=8)
    return original_counts, resized_counts


# 功能：
#   统计每类在模型尺寸下的可见图片数、空间分组数及缩小后消失情况，不使用模型成绩。
# 输入：
#   root、samples：已经通过原始格式检查的明确划分和样本。
#   width、height、minimum_pixels：模型输入尺寸及单图可见像素的固定诊断下限。
# 输出：
#   report：各类有效覆盖，不把多张相邻图片当多个独立场景或能力提升证明。
def inspect_model_resolution(root, samples, width=224, height=128, minimum_pixels=16):
    if (type(minimum_pixels) is not int or not 1 <= minimum_pixels <= 4096
            or type(width) is not int or type(height) is not int
            or not 64 <= width <= 1024 or not 64 <= height <= 1024):
        raise ValueError("VISION_VISIBILITY_POLICY_INVALID")
    totals, visible, disappeared = (np.zeros(8, dtype=np.int64) for _ in range(3))
    groups = [set() for _ in range(8)]
    supervised, excluded = 0, 0
    for sample in samples:
        if not sample.perception_supervision_enabled or sample.semantic_mask_relative_path is None:
            excluded += 1
            continue
        raw = read_plugin_file(root / sample.semantic_mask_relative_path, limit=4 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != sample.semantic_mask_sha256:
            raise ValueError("VISION_VISIBILITY_MASK_CHANGED")
        original, resized = mask_resolution_counts(raw, width, height)
        totals += resized
        visible += resized >= minimum_pixels
        disappeared += (original > 0) & (resized == 0)
        for index, count in enumerate(resized):
            if count >= minimum_pixels:
                groups[index].add(sample.scene_group_id or sample.flight_id)
        supervised += 1
    report = {"input_width": width, "input_height": height,
              "minimum_visible_pixels": minimum_pixels,
              "supervised_samples": supervised, "excluded_samples": excluded,
              "class_pixels": totals.tolist(), "class_visible_images": visible.tolist(),
              "class_visible_groups": [len(values) for values in groups],
              "class_visible_group_ids": [sorted(values) for values in groups],
              "class_disappeared_images": disappeared.tolist(),
              "class_order": list(LOCAL_VISION_SEMANTIC_CLASSES),
              "predictions_used": False, "model_capability_proven": False}
    return report
