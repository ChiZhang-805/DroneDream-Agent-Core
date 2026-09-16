"""Identical RGB tensor preparation for deployed experts and offline rollouts."""

import io
from pathlib import Path

from .plugin_files import read_plugin_file
from .visual_encoding_input import MAXIMUM_IMAGE_BYTES


# 功能：
#   1. 将调用方已验证并绑定身份的 RGB 或编码图像转换为训练和部署共用的张量。
#   2. 应用指定归一化并转为 NCHW；尺寸转换不改变或续期原观测时间。
#   3. 同时限制压缩文件和解码后像素预算，避免小文件展开为过量内存。
# 输入：
#   media：已通过调用方字节预算和来源检查的图像记录。
#   width：目标像素宽度。
#   height：目标像素高度。
#   normalization：编码器要求的归一化方式。
# 输出：
#   tensor：形状为 1×3×height×width 的 float32 张量。
def forward_rgb_tensor(media: dict, *, width: int, height: int, normalization: str):
    import numpy as np
    from PIL import Image

    if not isinstance(media, dict):
        raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_REQUIRED")
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 4096
            or not 1 <= height <= 2160 or width * height * 3 > MAXIMUM_IMAGE_BYTES):
        raise ValueError("VISUAL_ENCODING_DIMENSIONS_INVALID")
    if not isinstance(normalization, str) or normalization not in {
        "zero-to-one", "minus-one-to-one", "imagenet"
    }:
        raise ValueError("FORWARD_RGB_NORMALIZATION_UNSUPPORTED")
    raw = media.get("model_rgb_bytes")
    if raw is not None:
        if media.get("model_rgb_width") != width or media.get("model_rgb_height") != height:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB8_DIMENSIONS_MISMATCH")
        if not isinstance(raw, bytes) or len(raw) != width * height * 3:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB8_SIZE_MISMATCH")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        pixels = pixels.astype(np.float32) / 255.0
    else:
        embedded = media.get("content_bytes")
        if embedded is None:
            embedded = read_plugin_file(
                Path(str(media.get("path", ""))), limit=MAXIMUM_IMAGE_BYTES
            )
        if not isinstance(embedded, bytes) or not embedded or len(embedded) > MAXIMUM_IMAGE_BYTES:
            raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_BYTES_INVALID")
        with io.BytesIO(embedded) as source, Image.open(source) as image:
            # 检查图像头给出的原始像素量，再解码和转换；压缩字节数不代表展开内存。
            if image.width * image.height * 3 > MAXIMUM_IMAGE_BYTES:
                raise RuntimeError("LOCAL_POLICY_FORWARD_RGB_DECODED_PIXEL_BUDGET")
            with image.convert("RGB") as rgb, rgb.resize((width, height)) as resized:
                pixels = np.asarray(resized, dtype=np.float32) / 255.
    if normalization == "minus-one-to-one":
        pixels = pixels * 2.0 - 1.0
    elif normalization == "imagenet":
        pixels = (pixels - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray(
            [.229, .224, .225], dtype=np.float32,
        )
    tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
    return tensor
