"""Offline vision weight transfer; never imported by the flight execution loop."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_plugin_sdk.protocol import encode_json

MAX_WEIGHT_BYTES = 256 * 1024**2
VISION_INITIALIZATIONS = ("coco-voc-segmentation", "imagenet-backbone")


# 功能：
#   给实际加载的张量内容、名称、形状及类型生成摘要，不依赖可变的文件名。
# 输入：
#   state：仅包含有限稠密 CPU 或 GPU 张量的模型参数字典。
# 输出：
#   digest：与设备无关的实际参数内容 SHA-256。
def tensor_state_sha256(state: dict) -> str:
    import torch

    if not isinstance(state, dict) or not state or len(state) > 10_000:
        raise ValueError("VISION_WEIGHT_STATE_INVALID")
    hasher = hashlib.sha256()
    size = 0
    for name in sorted(state):
        value = state[name]
        if (type(name) is not str or not isinstance(value, torch.Tensor)
                or value.layout != torch.strided or not torch.isfinite(value).all()):
            raise ValueError("VISION_WEIGHT_TENSOR_INVALID")
        size += value.numel() * value.element_size()
        if size > MAX_WEIGHT_BYTES:
            raise ValueError("VISION_WEIGHT_BUDGET_EXCEEDED")
        hasher.update(encode_json([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        hasher.update(raw.tobytes())
    digest = hasher.hexdigest()
    return digest


# 功能：
#   先验证普通文件摘要与 ZIP 展开预算，再用受限加载器读取可信来源的权重字典。
# 输入：
#   path：显式选择的预训练权重文件。
#   expected_sha256：上传清单固定的完整文件摘要。
# 输出：
#   state：从同一份已校验字节载入的权重，不加载任意 Python 模型对象。
def read_vision_weights(path: Path, expected_sha256: str) -> dict:
    import torch

    raw = read_plugin_file(path, limit=MAX_WEIGHT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("VISION_PRETRAINED_FILE_HASH_MISMATCH")
    with ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 10_000 or sum(item.file_size for item in members) > MAX_WEIGHT_BYTES:
            raise ValueError("VISION_PRETRAINED_ARCHIVE_BUDGET_EXCEEDED")
    state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    tensor_state_sha256(state)
    return state


# 功能：
#   保留预训练分割的通用层，仅替换低层和高层分类输出；只复制语义相同的类别。
# 输入：
#   base：已经加载完整分割预训练权重的 LRASPP 网络。
#   source_classes：权重发布者定义的有序类别。
#   target_classes：DroneDream 八类有序定义。
# 输出：
#   copied：实际复制分类权重的同义类别列表；背景定义不同，不迁移背景行。
def adapt_segmentation_classes(base, source_classes: list[str], target_classes: tuple[str, ...]):
    import torch
    from torch import nn

    if len(source_classes) != len(set(source_classes)):
        raise ValueError("VISION_PRETRAINED_CLASS_NAMES_AMBIGUOUS")
    copied = [name for name in target_classes if name != "background" and name in source_classes]
    for attribute in ("low_classifier", "high_classifier"):
        original = getattr(base.classifier, attribute)
        if (not isinstance(original, nn.Conv2d) or original.kernel_size != (1, 1)
                or original.out_channels != len(source_classes) or original.bias is None):
            raise ValueError("VISION_PRETRAINED_CLASSIFIER_INCOMPATIBLE")
        replacement = nn.Conv2d(original.in_channels, len(target_classes), 1)
        with torch.no_grad():
            for name in copied:
                source_index = source_classes.index(name)
                target_index = target_classes.index(name)
                replacement.weight[target_index].copy_(original.weight[source_index])
                replacement.bias[target_index].copy_(original.bias[source_index])
        setattr(base.classifier, attribute, replacement)
    return copied


# 功能：
#   构造明确来源的视觉基座；优先完整分割迁移，显式兼容选项才使用分类骨干。
# 输入：
#   classes：目标分割类别。
#   pretrained、source：是否使用预训练及所选来源；禁用时不触发网络下载。
#   weights_path、weights_sha256：可选离线完整分割权重及摘要，必须成对提供。
# 输出：
#   base、record：已适配网络及实际初始化来源，不把新头声称为预训练头。
def initialize_vision_base(classes, *, pretrained, source, weights_path=None, weights_sha256=None):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )

    if type(pretrained) is not bool or source not in VISION_INITIALIZATIONS:
        raise ValueError("VISION_INITIALIZATION_INVALID")
    if (weights_path is None) != (weights_sha256 is None):
        raise ValueError("VISION_PRETRAINED_FILE_BINDING_REQUIRED")
    if weights_path is not None and (not pretrained or source != "coco-voc-segmentation"):
        raise ValueError("VISION_PRETRAINED_FILE_SOURCE_MISMATCH")
    if not pretrained:
        base = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None,
                                        num_classes=len(classes))
        record = {"source": "uninitialized-development-only", "tensor_sha256": None,
                  "copied_semantic_classes": []}
        return base, record
    if source == "imagenet-backbone":
        base = lraspp_mobilenet_v3_large(weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2, num_classes=len(classes))
        record = {"source": "mobilenet-v3-large-imagenet1k-v2",
                  "tensor_sha256": tensor_state_sha256(base.backbone.state_dict()),
                  "copied_semantic_classes": []}
        return base, record
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    state = (read_vision_weights(weights_path, weights_sha256) if weights_path is not None
             else weights.get_state_dict(progress=False, check_hash=True))
    digest = tensor_state_sha256(state)
    source_classes = weights.meta["categories"]
    base = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None,
                                    num_classes=len(source_classes))
    base.load_state_dict(state, strict=True)
    copied = adapt_segmentation_classes(base, source_classes, classes)
    record = {"source": "lraspp-mobilenet-v3-large-coco-voc-v1", "tensor_sha256": digest,
              "file_sha256": weights_sha256, "upstream_url": weights.url,
              "copied_semantic_classes": copied}
    return base, record
