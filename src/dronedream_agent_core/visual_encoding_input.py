"""Owned pixel-only computation input; carries no sensor age or permissions."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .hashing import sha256_json
from .plugin_files import read_plugin_file

MAXIMUM_IMAGE_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class VisualEncodingInput:
    """Immutable bytes and preprocessing identity for one visual computation."""
    content: bytes
    raw_rgb: bool
    width: int
    height: int
    normalization: str
    source_sha256: str
    encoder_sha256: str

    # 功能：
    #   按冻结像素、布局、归一化和编码器生成计算缓存键，不包含或续期传感器权限。
    # 输入：
    #   self：已冻结的视觉输入。
    # 输出：
    #   key：代表同一次像素计算身份的摘要。
    @property
    def key(self) -> str:
        key = sha256_json({"source_sha256": self.source_sha256, "raw_rgb": self.raw_rgb,
                            "width": self.width, "height": self.height,
                            "normalization": self.normalization,
                            "encoder_sha256": self.encoder_sha256})
        return key

    # 功能：
    #   从冻结字节建立训练与部署共用的 float32 NCHW 张量，不重新打开原图路径。
    # 输入：
    #   self：包含原始 RGB 或编码图片字节的冻结输入。
    # 输出：
    #   tensor：形状为 1×3×height×width 的预处理张量。
    def tensor(self):
        from .visual_control_input import forward_rgb_tensor

        media = ({"model_rgb_bytes": self.content, "model_rgb_width": self.width,
                  "model_rgb_height": self.height} if self.raw_rgb
                 else {"content_bytes": self.content})
        tensor = forward_rgb_tensor(media, width=self.width, height=self.height,
                                    normalization=self.normalization)
        return tensor


# 功能：
#   1. 核对单张图像的类型、尺寸、摘要和预处理配置，将字节冻结给异步编码器。
#   2. 文件仅有界读取一次；冻结后的像素不会随文件变化，观测新鲜度仍须独立检查。
# 输入：
#   multimodal：包含唯一 image-file 记录的列表或元组。
#   width：编码器所需的像素宽度。
#   height：编码器所需的像素高度。
#   normalization：编码器使用的归一化方法。
#   encoder_sha256：已校验的编码器权重摘要。
# 输出：
#   source：持有不可变图像字节与计算身份的视觉输入。
def freeze_visual_input(multimodal: list[dict], *, width: int, height: int,
                        normalization: str, encoder_sha256: str) -> VisualEncodingInput:
    if (not isinstance(encoder_sha256, str) or len(encoder_sha256) != 64
            or any(c not in "0123456789abcdef" for c in encoder_sha256)):
        raise ValueError("VISUAL_ENCODER_DIGEST_INVALID")
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 4096
            or not 1 <= height <= 2160 or width * height * 3 > MAXIMUM_IMAGE_BYTES):
        raise ValueError("VISUAL_ENCODING_DIMENSIONS_INVALID")
    if not isinstance(normalization, str) or normalization not in {
        "zero-to-one", "minus-one-to-one", "imagenet"
    }:
        raise ValueError("FORWARD_RGB_NORMALIZATION_UNSUPPORTED")
    if (not isinstance(multimodal, (list, tuple)) or len(multimodal) != 1
            or not isinstance(multimodal[0], dict) or multimodal[0].get("kind") != "image-file"):
        raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_REQUIRED")
    item = multimodal[0]
    raw = item.get("model_rgb_bytes")
    if raw is not None:
        if (type(item.get("model_rgb_width")) is not int
                or type(item.get("model_rgb_height")) is not int
                or item["model_rgb_width"] != width or item["model_rgb_height"] != height):
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB8_DIMENSIONS_MISMATCH")
        if not isinstance(raw, bytes) or len(raw) != width * height * 3:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB8_SIZE_MISMATCH")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item.get("model_rgb_sha256"):
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB8_HASH_MISMATCH")
        content = raw
    else:
        content = item.get("content_bytes")
        supplied = item.get("content_sha256")
        if content is None:
            # Freeze file bytes once, not a path reopened later by a worker.
            content = read_plugin_file(Path(str(item.get("path", ""))), limit=MAXIMUM_IMAGE_BYTES)
        elif not isinstance(content, bytes) or not isinstance(supplied, str):
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_BYTES_INVALID")
        if not content or len(content) > MAXIMUM_IMAGE_BYTES:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_UNAVAILABLE")
        digest = hashlib.sha256(content).hexdigest()
        if supplied is not None and digest != supplied:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_BYTES_HASH_MISMATCH")
    source = VisualEncodingInput(content, raw is not None, width, height, normalization,
                                 digest, encoder_sha256)
    return source
