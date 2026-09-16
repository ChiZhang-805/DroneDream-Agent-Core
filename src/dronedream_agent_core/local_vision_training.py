"""Training and ONNX export for the forward RGB semantic expert.

PyTorch and torchvision are development-only dependencies.  Product Runtime
loads only the exported ONNX graph through :mod:`local_policy_port`; importing
the ordinary Agent Core therefore does not import either training framework.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from .asset_package_storage import publish_asset_file
from .contracts import StrictModel
from .plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from .plugin_values import plugin_json_value

_UnitTarget = Annotated[float, Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False)]
_MAX_IMAGE_BYTES = 16 * 1024**2
_MAX_MODEL_BYTES = 256 * 1024**2

LOCAL_VISION_SEMANTIC_CLASSES = (
    "background",
    "traversable",
    "obstacle",
    "doorway",
    "stairs",
    "person",
    "glass",
    "pickup-marker",
)
LOCAL_VISION_SCENE_LABELS = (
    "indoor",
    "outdoor",
    "doorway",
    "stairs",
    "person-present",
    "glass-present",
)
LOCAL_VISION_QUALITY_LABELS = (
    "underexposed",
    "overexposed",
    "blurred",
    "occluded",
)


# 功能：
#   验证固定八类别的模拟器监督契约，以同一批有界普通文件字节解析并计算摘要。
# 输入：
#   path：标签定义文件路径。
# 输出：
#   digest、class_ids：实际定义文件摘要及完整类别编号集合。
def load_local_vision_label_map(path: Path) -> tuple[str, frozenset[int]]:
    raw = read_plugin_file(path, limit=64 * 1024)
    payload = decode_json(raw, limit=64 * 1024)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "dronedream.vision-label-map.v1"
    ):
        raise ValueError("vision label map schema is unsupported")
    classes = payload.get("classes")
    expected = [
        {"class_id": class_id, "name": name}
        for class_id, name in enumerate(LOCAL_VISION_SEMANTIC_CLASSES)
    ]
    if (classes != expected or type(payload.get("background_class_id")) is not int
            or payload["background_class_id"] != 0
            or any(type(item.get("class_id")) is not int for item in classes)):
        raise ValueError("vision label map classes do not match the model contract")
    digest = hashlib.sha256(raw).hexdigest()
    class_ids = frozenset(range(len(expected)))
    return digest, class_ids


# 功能：
#   计算嵌入、可通行性、场景和质量概率拼接后的固定特征宽度。
# 输入：
#   embedding_feature_count：允许范围内的整数嵌入维数。
# 输出：
#   count：运行端视觉特征向量的总维数。
def local_vision_feature_count(embedding_feature_count: int) -> int:
    if type(embedding_feature_count) is not int or not 16 <= embedding_feature_count <= 1024:
        raise ValueError("vision embedding feature count is invalid")
    count = (
        embedding_feature_count
        + 1
        + len(LOCAL_VISION_SCENE_LABELS)
        + len(LOCAL_VISION_QUALITY_LABELS)
    )
    return count


class LocalVisionTrainingSample(StrictModel):
    """One hash-bound aligned RGB supervision sample."""
    model_config = {"strict": True}

    flight_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=120)
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_relative_path: str = Field(min_length=1, max_length=240)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_mask_relative_path: str | None = Field(default=None, max_length=240)
    semantic_mask_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_record_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_sensor_snapshot_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    rgb_semantic_time_offset_seconds: float | None = Field(
        default=None,
        ge=0.0,
        le=0.1,
    )
    traversability_target: _UnitTarget
    scene_targets: list[_UnitTarget] = Field(
        min_length=len(LOCAL_VISION_SCENE_LABELS),
        max_length=len(LOCAL_VISION_SCENE_LABELS),
    )
    scene_target_weights: list[_UnitTarget] = Field(
        default_factory=lambda: [1.0] * len(LOCAL_VISION_SCENE_LABELS),
        min_length=len(LOCAL_VISION_SCENE_LABELS),
        max_length=len(LOCAL_VISION_SCENE_LABELS),
    )
    quality_targets: list[_UnitTarget] = Field(
        min_length=len(LOCAL_VISION_QUALITY_LABELS),
        max_length=len(LOCAL_VISION_QUALITY_LABELS),
    )
    quality_target_weights: list[_UnitTarget] = Field(
        default_factory=lambda: [1.0] * len(LOCAL_VISION_QUALITY_LABELS),
        min_length=len(LOCAL_VISION_QUALITY_LABELS),
        max_length=len(LOCAL_VISION_QUALITY_LABELS),
    )
    sample_weight: float = Field(default=1.0, gt=0.0, le=100.0)

    # 功能：
    #   规范历史反斜杠并校验跨平台相对路径，拒绝路径逃逸、设备名及含糊拼写。
    # 输入：
    #   cls：训练样本类型。
    #   value：图像或可选标签路径。
    # 输出：
    #   normalized：通过校验的相对路径；可选缺失值保持 None。
    @field_validator("image_relative_path", "semantic_mask_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/")
        candidate = PurePosixPath(normalized)
        if (
            candidate.is_absolute()
            or normalized != candidate.as_posix()
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or not candidate.parts
            or ":" in candidate.parts[0]
        ):
            raise ValueError("vision sample paths must be normalized and relative")
        return portable_plugin_path(normalized)

    # 功能：
    #   要求掩码路径与摘要成对，辅助监督权重至少启用一个有效标签。
    # 输入：
    #   self：标量范围已验证的训练样本。
    # 输出：
    #   self：监督绑定一致的样本。
    @model_validator(mode="after")
    def validate_mask_binding(self) -> LocalVisionTrainingSample:
        if (self.semantic_mask_relative_path is None) != (
            self.semantic_mask_sha256 is None
        ):
            raise ValueError("semantic mask path and hash must be supplied together")
        for name, weights in (
            ("scene", self.scene_target_weights),
            ("quality", self.quality_target_weights),
        ):
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in weights):
                raise ValueError(f"{name} target weights must be within zero and one")
            if sum(weights) <= 0.0:
                raise ValueError(f"{name} target weights must enable at least one label")
        return self


class LocalVisionTrainingConfig(StrictModel):
    model_config = {"strict": True}
    width: int = Field(default=224, ge=64, le=1_024)
    height: int = Field(default=128, ge=64, le=1_024)
    embedding_feature_count: int = Field(default=128, ge=16, le=1_024)
    epoch_count: int = Field(default=20, ge=1, le=10_000)
    batch_size: int = Field(default=16, ge=1, le=1_024)
    learning_rate: float = Field(default=0.0003, gt=0.0, le=0.1)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=1.0)
    semantic_loss_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    traversability_loss_weight: float = Field(default=0.5, ge=0.0, le=100.0)
    scene_loss_weight: float = Field(default=0.5, ge=0.0, le=100.0)
    quality_loss_weight: float = Field(default=0.25, ge=0.0, le=100.0)
    semantic_class_balance_power: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_class_weight_cap: float = Field(default=8.0, ge=1.0, le=100.0)
    random_seed: int = Field(default=805, ge=0, le=2**32 - 1)
    freeze_backbone_epochs: int = Field(default=2, ge=0, le=10_000)

    # 功能：
    #   拒绝没有任何训练目标或输入张量规模明显超过本地开发预算的配置。
    # 输入：
    #   self：字段范围通过检查的训练配置。
    # 输出：
    #   self：损失目标和批量输入预算有效的配置。
    @model_validator(mode="after")
    def validate_training_budget(self) -> LocalVisionTrainingConfig:
        if not any(value > 0 for value in (
            self.semantic_loss_weight, self.traversability_loss_weight,
            self.scene_loss_weight, self.quality_loss_weight,
        )):
            raise ValueError("vision training must enable at least one loss")
        if self.width * self.height * self.batch_size > 16 * 1024**2:
            raise ValueError("vision training batch exceeds the input pixel budget")
        return self


class LocalVisionTrainingMetrics(StrictModel):
    model_config = {"strict": True, "allow_inf_nan": False}
    sample_count: int = Field(ge=1)
    semantic_sample_count: int = Field(ge=0)
    mean_loss: float = Field(ge=0.0)
    semantic_pixel_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_class_target_pixels: list[int] = Field(
        min_length=len(LOCAL_VISION_SEMANTIC_CLASSES),
        max_length=len(LOCAL_VISION_SEMANTIC_CLASSES),
    )
    semantic_class_iou: list[float | None] = Field(
        min_length=len(LOCAL_VISION_SEMANTIC_CLASSES),
        max_length=len(LOCAL_VISION_SEMANTIC_CLASSES),
    )
    semantic_mean_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    traversability_mean_absolute_error: float = Field(ge=0.0, le=1.0)
    scene_binary_accuracy: float = Field(ge=0.0, le=1.0)
    quality_binary_accuracy: float = Field(ge=0.0, le=1.0)


# 功能：
#   只在开发训练入口加载 PyTorch，运行时导入本模块不加载训练框架。
# 输入：
#   无。
# 输出：
#   torch、nn、functional：训练计算、网络层和函数接口。
def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("LOCAL_VISION_TRAINING_DEPENDENCIES_NOT_INSTALLED") from error
    return torch, nn, functional


# 功能：
#   冻结并重新验证可能被调用方修改的训练配置，不用旧的实例校验状态跳过检查。
# 输入：
#   config：当前训练配置。
# 输出：
#   checked：独立且重新通过类型和预算验证的配置。
def _checked_config(config: LocalVisionTrainingConfig) -> LocalVisionTrainingConfig:
    checked = LocalVisionTrainingConfig.model_validate(plugin_json_value(config))
    return checked


# 功能：
#   构造 MobileNetV3-Large、Lite R-ASPP 及共享辅助头，只有显式许可才加载预训练骨干。
# 输入：
#   config：嵌入维数、图像尺寸和初始化种子。
#   pretrained_backbone：是否允许使用可触发下载的 ImageNet 骨干权重。
# 输出：
#   model：开发训练使用的多输出视觉网络。
def build_local_vision_model(config: LocalVisionTrainingConfig,
                             *, pretrained_backbone: bool) -> Any:
    if type(pretrained_backbone) is not bool:
        raise ValueError("pretrained backbone requires an explicit boolean")
    config = _checked_config(config)
    torch, nn, functional = _require_torch()
    torch.manual_seed(config.random_seed)
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    base = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=(
            MobileNet_V3_Large_Weights.IMAGENET1K_V2
            if pretrained_backbone
            else None
        ),
        num_classes=len(LOCAL_VISION_SEMANTIC_CLASSES),
    )

    class DroneDreamForwardVision(nn.Module):
        # 功能：
        #   复用骨干与分割头，建立嵌入、可通行性、场景和图像质量的输出投影。
        # 输入：
        #   self：新网络；闭包持有已构造骨干和已验证配置。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self) -> None:
            super().__init__()
            self.backbone = base.backbone
            self.semantic_head = base.classifier
            self.embedding_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(960, config.embedding_feature_count),
                nn.Tanh(),
            )
            # 嵌入投影当前没有独立监督损失；表征会随共享骨干变化，不宣称该头单独训练收敛。
            self.traversability_head = nn.Linear(960, 1)
            self.scene_head = nn.Linear(960, len(LOCAL_VISION_SCENE_LABELS))
            self.quality_head = nn.Linear(960, len(LOCAL_VISION_QUALITY_LABELS))

        # 功能：
        #   提取共享视觉特征，将分割 logits 上采样到输入尺寸并拼接运行端特征向量。
        # 输入：
        #   self：当前视觉网络。
        #   forward_rgb：按 ImageNet 均值方差规范化的 N×3×H×W 张量。
        # 输出：
        #   visual_features、semantic_logits、traversability_logits、scene_logits、quality_logits：
        #     拼接特征及分别供分割、可通行性、场景和质量损失使用的未激活输出。
        def forward(self, forward_rgb: Any) -> tuple[Any, Any, Any, Any, Any]:
            features = self.backbone(forward_rgb)
            semantic_logits = self.semantic_head(features)
            semantic_logits = functional.interpolate(
                semantic_logits,
                size=forward_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pooled = functional.adaptive_avg_pool2d(features["high"], 1).flatten(1)
            embedding = self.embedding_head(features["high"])
            traversability_logits = self.traversability_head(pooled)
            scene_logits = self.scene_head(pooled)
            quality_logits = self.quality_head(pooled)
            visual_features = torch.cat(
                (
                    embedding,
                    torch.sigmoid(traversability_logits),
                    torch.sigmoid(scene_logits),
                    torch.sigmoid(quality_logits),
                ),
                dim=1,
            )
            return (
                visual_features,
                semantic_logits,
                traversability_logits,
                scene_logits,
                quality_logits,
            )

    model = DroneDreamForwardVision()
    return model


# 功能：
#   统计模型全部参数，含当前轮临时冻结的骨干，供既有产品模型大小报告使用。
# 输入：
#   model：需要统计的网络；保留既有函数名，但口径不是当前可求梯度参数的数量。
# 输出：
#   count：全部参数元素总数。
def count_trainable_parameters(model: Any) -> int:
    count = sum(parameter.numel() for parameter in model.parameters())
    return count


# 功能：
#   流式散列普通文件并检查读取身份，避免一次读入任意大的模型文件。
# 输入：
#   path：制品文件路径。
#   limit：该类制品的最大字节数。
# 输出：
#   digest：实际读取内容的摘要。
def _sha256(path: Path, *, limit: int = _MAX_MODEL_BYTES) -> str:
    digest = hash_plugin_file(path, limit=limit)
    return digest


# 功能：
#   逐样本冻结监督并校验相对路径和摘要，限制样本总量，不先解析掉目录中的链接。
# 输入：
#   dataset_root：当前训练或验证数据集根。
#   samples：已划分好的样本序列。
# 输出：
#   resolved：独立样本、RGB 路径及可选掩码路径组成的列表。
def resolve_local_vision_samples(dataset_root: Path, samples: Iterable[LocalVisionTrainingSample],
                                ) -> list[tuple[LocalVisionTrainingSample, Path, Path | None]]:
    root = dataset_root.absolute()
    check_plain_plugin_path(root)
    resolved = []
    for sample in samples:
        if len(resolved) >= 100_000:
            raise ValueError("vision sample count exceeds the development budget")
        sample = LocalVisionTrainingSample.model_validate(plugin_json_value(sample))
        image = root / sample.image_relative_path
        check_plain_plugin_path(image)
        if root not in image.parents or not image.is_file():
            raise ValueError("vision image escapes the dataset root")
        if _sha256(image, limit=_MAX_IMAGE_BYTES) != sample.image_sha256:
            raise ValueError("vision image hash mismatch")
        mask = None
        if sample.semantic_mask_relative_path is not None:
            mask = root / sample.semantic_mask_relative_path
            check_plain_plugin_path(mask)
            if root not in mask.parents or not mask.is_file():
                raise ValueError("vision mask escapes the dataset root")
            if _sha256(mask, limit=4 * 1024**2) != sample.semantic_mask_sha256:
                raise ValueError("vision mask hash mismatch")
        resolved.append((sample, image, mask))
    if not resolved:
        raise ValueError("vision dataset is empty")
    return resolved


# 功能：
#   从同一份有界、摘要匹配的 RGB PNG 字节解码并规范化，防止训练时读到替换图像。
# 输入：
#   path：RGB 图像路径。
#   config：训练输入尺寸。
#   expected_sha256：来源样本绑定摘要；生产训练提供，独立算法测试可省略。
# 输出：
#   tensor：ImageNet 规范化的 3×H×W CPU float32 张量。
def _image_tensor(path: Path, config: LocalVisionTrainingConfig,
                  *, expected_sha256: str | None = None) -> Any:
    import numpy as np
    from PIL import Image

    raw = read_plugin_file(path, limit=_MAX_IMAGE_BYTES)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("vision image hash mismatch")
    with BytesIO(raw) as stream, Image.open(stream, formats=["PNG"]) as image:
        if (image.mode != "RGB" or not 1 <= image.width <= 4096 or not 1 <= image.height <= 2160
                or getattr(image, "n_frames", 1) != 1):
            raise ValueError("vision RGB image mode or dimensions are invalid")
        with image.resize((config.width, config.height)) as resized:
            pixels = np.array(resized, dtype=np.float32) / 255.0
    pixels = (
        pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    torch, _nn, _functional = _require_torch()
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1))
    return tensor


# 功能：
#   读取摘要绑定的类别索引 PNG，用最近邻缩放而不混合标签或改变调色板索引含义。
# 输入：
#   path：单通道或索引调色板语义掩码路径。
#   config：模型目标尺寸。
#   expected_sha256：来源样本的掩码摘要；生产训练提供。
# 输出：
#   tensor：H×W CPU int64 类别索引张量。
def _mask_tensor(path: Path, config: LocalVisionTrainingConfig,
                 *, expected_sha256: str | None = None) -> Any:
    import numpy as np
    from PIL import Image

    raw = read_plugin_file(path, limit=4 * 1024**2)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("vision mask hash mismatch")
    with BytesIO(raw) as stream, Image.open(stream, formats=["PNG"]) as image:
        if (image.mode not in {"L", "P"} or not 1 <= image.width <= 4096
                or not 1 <= image.height <= 2160 or getattr(image, "n_frames", 1) != 1):
            raise ValueError("vision mask mode or dimensions are invalid")
        # 缩小可能恰好丢掉非法像素，原图类别范围必须在最近邻采样之前验证。
        original = np.asarray(image)
        if int(original.min()) < 0 or int(original.max()) >= len(LOCAL_VISION_SEMANTIC_CLASSES):
            raise ValueError("vision semantic mask class is outside the contract")
        with image.resize((config.width, config.height), Image.Resampling.NEAREST) as resized:
            mask = np.array(resized, dtype=np.int64)
    if mask.size == 0 or int(mask.min()) < 0 or int(mask.max()) >= len(
        LOCAL_VISION_SEMANTIC_CLASSES
    ):
        raise ValueError("vision semantic mask class is outside the contract")
    torch, _nn, _functional = _require_torch()
    tensor = torch.from_numpy(mask)
    return tensor


# 功能：
#   保持给定洗牌顺序按固定上限分批，最后不足一批时仍完整保留。
# 输入：
#   values：本轮有序样本索引。
#   batch_size：严格正整数批大小。
# 输出：
#   batch：逐次产出的索引子列表。
def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("vision batch size is invalid")
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        yield batch


# 功能：
#   冻结准备好的样本并复核监督类型与路径配对，拒绝空集和超预算列表。
# 输入：
#   resolved_samples：已完成数据集路径解析的样本及图像／掩码路径。
# 输出：
#   checked：本次训练或评估独立持有的样本列表。
def _checked_samples(resolved_samples) -> list[tuple[LocalVisionTrainingSample, Path, Path | None]]:
    if type(resolved_samples) is not list or not 1 <= len(resolved_samples) <= 100_000:
        raise ValueError("vision resolved sample count is invalid")
    checked = []
    for entry in resolved_samples:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError("vision resolved sample entry is invalid")
        sample, image, mask = entry
        sample = LocalVisionTrainingSample.model_validate(plugin_json_value(sample))
        if not isinstance(image, Path) or (mask is not None and not isinstance(mask, Path)):
            raise ValueError("vision resolved sample path is invalid")
        if (mask is None) != (sample.semantic_mask_sha256 is None):
            raise ValueError("vision resolved mask binding is incomplete")
        checked.append((sample, image, mask))
    return checked


# 功能：
#   按训练集缩放后的类别像素频率计算有界逆频率权重，缺失类别不给出伪造覆盖证据。
# 输入：
#   resolved_samples：只来自训练划分的已绑定样本。
#   config：尺寸、平衡指数和权重上限。
# 输出：
#   weights：按固定类别顺序排列的分割损失权重。
def local_vision_semantic_class_weights(
    resolved_samples: list[tuple[LocalVisionTrainingSample, Path, Path | None]],
    config: LocalVisionTrainingConfig,
) -> list[float]:
    config = _checked_config(config)
    resolved_samples = _checked_samples(resolved_samples)
    torch, _nn, _functional = _require_torch()
    class_count = len(LOCAL_VISION_SEMANTIC_CLASSES)
    counts = torch.zeros(class_count, dtype=torch.float64)
    for sample, _image_path, mask_path in resolved_samples:
        if mask_path is None:
            continue
        mask = _mask_tensor(mask_path, config,
                             expected_sha256=sample.semantic_mask_sha256).reshape(-1)
        counts += torch.bincount(mask, minlength=class_count).to(torch.float64)
    observed = counts > 0
    if not bool(observed.any()):
        weights = [1.0] * class_count
        return weights
    total = counts.sum()
    raw = torch.zeros_like(counts)
    # 默认平方根逆频率提高稀少类别的梯度贡献；先限幅再规范化，不让少量像素支配更新。
    raw[observed] = (total / counts[observed]).pow(
        config.semantic_class_balance_power
    )
    raw[observed] = raw[observed].clamp(max=config.semantic_class_weight_cap)
    raw[observed] /= raw[observed].mean()
    weights = [float(value) for value in raw]
    return weights


# 功能：
#   用有来源摘要的整次飞行样本微调分割和辅助任务，冻结轮次禁止骨干更新统计量。
# 输入：
#   model：当前 CPU 视觉网络。
#   resolved_samples：训练划分的样本与制品路径。
#   config：优化器、轮数、损失权重和确定性洗牌配置。
# 输出：
#   model、metrics：完成训练的网络及该训练集上的评估指标，不代表独立验证资格。
def train_local_vision_model(
    model: Any,
    resolved_samples: list[tuple[LocalVisionTrainingSample, Path, Path | None]],
    config: LocalVisionTrainingConfig,
) -> tuple[Any, LocalVisionTrainingMetrics]:
    config = _checked_config(config)
    resolved_samples = _checked_samples(resolved_samples)
    if (not any(mask is not None for _, _, mask in resolved_samples)
            and not any(value > 0 for value in (config.traversability_loss_weight,
                       config.scene_loss_weight, config.quality_loss_weight))):
        raise ValueError("vision training has no supervised active loss")
    torch, _nn, functional = _require_torch()
    torch.manual_seed(config.random_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    semantic_class_weights = torch.tensor(
        local_vision_semantic_class_weights(resolved_samples, config),
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(config.random_seed)
    for epoch in range(config.epoch_count):
        freeze_backbone = epoch < config.freeze_backbone_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not freeze_backbone)
        order = torch.randperm(len(resolved_samples), generator=generator).tolist()
        model.train()
        if freeze_backbone:
            # requires_grad 不会冻结 BatchNorm 的运行统计；冻结轮次必须同时进入 eval。
            model.backbone.eval()
        for indices in _batches(order, config.batch_size):
            batch = [resolved_samples[index] for index in indices]
            images = torch.stack([_image_tensor(item[1], config,
                expected_sha256=item[0].image_sha256) for item in batch])
            outputs = model(images)
            _visual, semantic, traversability, scene, quality = outputs
            weights = torch.tensor(
                [item[0].sample_weight for item in batch], dtype=torch.float32
            )
            traversability_target = torch.tensor(
                [[item[0].traversability_target] for item in batch],
                dtype=torch.float32,
            )
            scene_target = torch.tensor(
                [item[0].scene_targets for item in batch], dtype=torch.float32
            )
            quality_target = torch.tensor(
                [item[0].quality_targets for item in batch], dtype=torch.float32
            )
            scene_target_weights = torch.tensor(
                [item[0].scene_target_weights for item in batch], dtype=torch.float32
            )
            quality_target_weights = torch.tensor(
                [item[0].quality_target_weights for item in batch], dtype=torch.float32
            )
            loss = (
                functional.binary_cross_entropy_with_logits(
                    traversability,
                    traversability_target,
                    reduction="none",
                ).mean(1)
                * weights
            ).mean() * config.traversability_loss_weight
            loss += (
                (
                    functional.binary_cross_entropy_with_logits(
                        scene, scene_target, reduction="none"
                    )
                    * scene_target_weights
                ).sum(1)
                / scene_target_weights.sum(1).clamp_min(1.0)
                * weights
            ).mean() * config.scene_loss_weight
            loss += (
                (
                    functional.binary_cross_entropy_with_logits(
                        quality, quality_target, reduction="none"
                    )
                    * quality_target_weights
                ).sum(1)
                / quality_target_weights.sum(1).clamp_min(1.0)
                * weights
            ).mean() * config.quality_loss_weight
            semantic_losses = []
            for batch_index, (sample, _image, mask_path) in enumerate(batch):
                if mask_path is not None:
                    semantic_losses.append(
                        functional.cross_entropy(
                            semantic[batch_index : batch_index + 1],
                            _mask_tensor(mask_path, config,
                                expected_sha256=sample.semantic_mask_sha256).unsqueeze(0),
                            weight=semantic_class_weights,
                        ) * weights[batch_index]
                    )
            if semantic_losses:
                loss += torch.stack(semantic_losses).mean() * config.semantic_loss_weight
            if not torch.isfinite(loss):
                raise RuntimeError("LOCAL_VISION_TRAINING_LOSS_NOT_FINITE")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise RuntimeError("LOCAL_VISION_TRAINING_GRADIENT_NOT_FINITE")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("LOCAL_VISION_TRAINING_PARAMETER_NOT_FINITE")
    metrics = evaluate_local_vision_model(model, resolved_samples, config)
    return model, metrics


# 功能：
#   计算无梯度逐样本误差、有效标签准确率与类别 IoU，均值损失保留未加训练权重的口径。
# 输入：
#   model：当前 CPU 视觉网络。
#   resolved_samples：明确选定的训练集或独立验证集，来源由调用方管理。
#   config：图像与标签输入尺寸。
# 输出：
#   metrics：所提供数据集上的诊断指标，不自动授予模型发布或飞行资格。
def evaluate_local_vision_model(
    model: Any,
    resolved_samples: list[tuple[LocalVisionTrainingSample, Path, Path | None]],
    config: LocalVisionTrainingConfig,
) -> LocalVisionTrainingMetrics:
    config = _checked_config(config)
    resolved_samples = _checked_samples(resolved_samples)
    torch, _nn, functional = _require_torch()
    model.eval()
    losses = []
    traversability_errors = []
    scene_correct = 0
    quality_correct = 0
    scene_total = 0
    quality_total = 0
    semantic_correct = 0
    semantic_total = 0
    semantic_class_target_pixels = [0] * len(LOCAL_VISION_SEMANTIC_CLASSES)
    semantic_class_intersections = [0] * len(LOCAL_VISION_SEMANTIC_CLASSES)
    semantic_class_unions = [0] * len(LOCAL_VISION_SEMANTIC_CLASSES)
    with torch.inference_mode():
        for sample, image_path, mask_path in resolved_samples:
            image = _image_tensor(image_path, config,
                                   expected_sha256=sample.image_sha256).unsqueeze(0)
            outputs = model(image)
            if any(not torch.isfinite(output).all() for output in outputs):
                raise RuntimeError("LOCAL_VISION_EVALUATION_OUTPUT_NOT_FINITE")
            _visual, semantic, traversability, scene, quality = outputs
            traversability_probability = torch.sigmoid(traversability)[0, 0]
            traversability_errors.append(
                abs(float(traversability_probability) - sample.traversability_target)
            )
            scene_target = torch.tensor(sample.scene_targets, dtype=torch.float32)
            quality_target = torch.tensor(sample.quality_targets, dtype=torch.float32)
            scene_weights = torch.tensor(
                sample.scene_target_weights, dtype=torch.float32
            )
            quality_weights = torch.tensor(
                sample.quality_target_weights, dtype=torch.float32
            )
            scene_correct += int(
                (
                    ((torch.sigmoid(scene[0]) >= 0.5) == (scene_target >= 0.5))
                    * (scene_weights > 0.0)
                ).sum()
            )
            quality_correct += int(
                (
                    ((torch.sigmoid(quality[0]) >= 0.5) == (quality_target >= 0.5))
                    * (quality_weights > 0.0)
                ).sum()
            )
            scene_total += int((scene_weights > 0.0).sum())
            quality_total += int((quality_weights > 0.0).sum())
            sample_loss = functional.binary_cross_entropy_with_logits(
                traversability,
                torch.tensor([[sample.traversability_target]], dtype=torch.float32),
            )
            sample_loss += (
                functional.binary_cross_entropy_with_logits(
                    scene[0], scene_target, reduction="none"
                )
                * scene_weights
            ).sum() / scene_weights.sum().clamp_min(1.0)
            sample_loss += (
                functional.binary_cross_entropy_with_logits(
                    quality[0], quality_target, reduction="none"
                )
                * quality_weights
            ).sum() / quality_weights.sum().clamp_min(1.0)
            if mask_path is not None:
                mask = _mask_tensor(mask_path, config, expected_sha256=sample.semantic_mask_sha256)
                sample_loss += functional.cross_entropy(semantic, mask.unsqueeze(0))
                prediction = semantic.argmax(1)[0]
                semantic_correct += int((prediction == mask).sum())
                semantic_total += int(mask.numel())
                for class_id in range(len(LOCAL_VISION_SEMANTIC_CLASSES)):
                    target_class = mask == class_id
                    predicted_class = prediction == class_id
                    semantic_class_target_pixels[class_id] += int(target_class.sum())
                    semantic_class_intersections[class_id] += int(
                        (target_class & predicted_class).sum()
                    )
                    semantic_class_unions[class_id] += int(
                        (target_class | predicted_class).sum()
                    )
            if not torch.isfinite(sample_loss):
                raise RuntimeError("LOCAL_VISION_EVALUATION_LOSS_NOT_FINITE")
            losses.append(float(sample_loss))
    semantic_class_iou = [
        (
            semantic_class_intersections[class_id]
            / semantic_class_unions[class_id]
            if semantic_class_unions[class_id]
            else None
        )
        for class_id in range(len(LOCAL_VISION_SEMANTIC_CLASSES))
    ]
    observed_class_iou = [
        semantic_class_iou[class_id]
        for class_id, target_pixels in enumerate(semantic_class_target_pixels)
        if target_pixels > 0 and semantic_class_iou[class_id] is not None
    ]
    metrics = LocalVisionTrainingMetrics(
        sample_count=len(resolved_samples),
        semantic_sample_count=sum(item[2] is not None for item in resolved_samples),
        mean_loss=sum(losses) / len(losses),
        semantic_pixel_accuracy=(
            semantic_correct / semantic_total if semantic_total else None
        ),
        semantic_class_target_pixels=semantic_class_target_pixels,
        semantic_class_iou=semantic_class_iou,
        semantic_mean_iou=(
            sum(observed_class_iou) / len(observed_class_iou)
            if observed_class_iou
            else None
        ),
        traversability_mean_absolute_error=(
            sum(traversability_errors) / len(traversability_errors)
        ),
        scene_binary_accuracy=scene_correct / scene_total if scene_total else 0.0,
        quality_binary_accuracy=(
            quality_correct / quality_total if quality_total else 0.0
        ),
    )
    return metrics


# 功能：
#   将固定输入形状的五输出网络导出到自有暂存流，校验图结构后无覆盖发布新 ONNX。
# 输入：
#   model：准备冻结的 CPU 视觉网络。
#   output_path：必须尚不存在的最终模型文件。
#   config：模型固定输入尺寸与特征维数。
# 输出：
#   digest：已发布 ONNX 实际字节的 SHA-256。
def export_local_vision_onnx(model: Any, output_path: Path,
                            config: LocalVisionTrainingConfig) -> str:
    check_plain_plugin_path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    config = _checked_config(config)
    torch, _nn, _functional = _require_torch()
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, config.height, config.width), dtype=torch.float32)
    if sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()
           ) > _MAX_MODEL_BYTES:
        raise ValueError("vision parameters exceed the export byte budget")
    temporary = None
    identity = None
    try:
        # 保留独占描述符交给传统导出器；不开启外部权重旁文件，也不覆盖已有模型。
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".vision-export-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            identity = os.fstat(stream.fileno())
            torch.onnx.export(model, (example,), stream, input_names=["forward_rgb"],
                output_names=["visual_features", "semantic_logits", "traversability_logits",
                              "scene_logits", "quality_logits"],
                opset_version=17, do_constant_folding=True, dynamo=False, external_data=False)
            stream.flush()
            os.fsync(stream.fileno())
        check_plain_plugin_path(temporary)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("LOCAL_VISION_EXPORT_TEMPORARY_CHANGED")
        raw = read_plugin_file(temporary, limit=_MAX_MODEL_BYTES)
        if not raw:
            raise RuntimeError("LOCAL_VISION_ONNX_SIZE_OUTSIDE_DEVELOPMENT_BOUND")
        import onnx

        document = onnx.load_model_from_string(raw)
        onnx.checker.check_model(document)
        observed_inputs = {item.name for item in document.graph.input}
        observed_outputs = {item.name for item in document.graph.output}
        if observed_inputs != {"forward_rgb"} or observed_outputs != {
            "visual_features", "semantic_logits", "traversability_logits", "scene_logits",
            "quality_logits",
        }:
            raise RuntimeError("LOCAL_VISION_ONNX_TENSOR_CONTRACT_MISMATCH")
        tensor = document.graph.input[0].type.tensor_type
        shape = [dim.dim_value for dim in tensor.shape.dim]
        if (tensor.elem_type != onnx.TensorProto.FLOAT
                or shape != [1, 3, config.height, config.width]):
            raise RuntimeError("LOCAL_VISION_ONNX_INPUT_SHAPE_MISMATCH")
        digest = hashlib.sha256(raw).hexdigest()
        publish_asset_file(temporary, output_path, expected_sha256=digest, limit=_MAX_MODEL_BYTES)
        return digest
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


# 功能：
#   按运行端只请求 visual_features 的调用方式测量 CPU 推理时延，逐轮核对输出。
# 输入：
#   path：待测本地 ONNX 路径。
#   config：输入尺寸与特征宽度。
#   iterations：5 至 10000 次测量，不包含加载、预处理和最初三次预热。
# 输出：
#   benchmark：分位推理耗时、提供程序、模型字节数与特征宽度。
def benchmark_local_vision_onnx(path: Path, config: LocalVisionTrainingConfig,
                                *, iterations: int = 50) -> dict[str, object]:
    if type(iterations) is not int or not 5 <= iterations <= 10_000:
        raise ValueError("vision benchmark iteration count is outside the safe range")
    config = _checked_config(config)
    import time

    import numpy as np
    import onnxruntime as ort

    raw = read_plugin_file(path, limit=_MAX_MODEL_BYTES)
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 3, config.height, config.width), dtype=np.float32)
    latencies = []
    for iteration in range(iterations + 3):
        started = time.perf_counter()
        output = session.run(["visual_features"], {"forward_rgb": sample})[0]
        elapsed = (time.perf_counter() - started) * 1_000.0
        if (output.shape != (1, local_vision_feature_count(config.embedding_feature_count))
                or not np.isfinite(output).all()):
            raise RuntimeError("LOCAL_VISION_FEATURE_SHAPE_OR_VALUE_INVALID")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("LOCAL_VISION_BENCHMARK_CLOCK_INVALID")
        if iteration >= 3:
            latencies.append(elapsed)
    ordered = sorted(latencies)

    # 功能：
    #   在已排序的测量样本中使用 nearest-rank 规则选择分位值。
    # 输入：
    #   percent：本函数内部固定使用的百分位。
    # 输出：
    #   latency：对应的毫秒耗时。
    def percentile(percent: float) -> float:
        index = min(len(ordered) - 1, math.ceil(percent / 100.0 * len(ordered)) - 1)
        latency = ordered[index]
        return latency

    benchmark = {
        "provider": session.get_providers(),
        "iteration_count": iterations,
        "p50_inference_latency_ms": percentile(50.0),
        "p95_inference_latency_ms": percentile(95.0),
        "p99_inference_latency_ms": percentile(99.0),
        "artifact_size_bytes": len(raw),
        "visual_feature_count": int(output.shape[1]),
        "measured_outputs": ["visual_features"],
        "includes_preprocessing": False,
    }
    return benchmark
