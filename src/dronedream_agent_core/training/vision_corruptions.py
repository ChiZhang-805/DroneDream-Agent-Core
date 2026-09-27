"""Controlled rendered-image corruptions; never inferred labels for real camera photos."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
from pydantic import Field, model_validator

from dronedream_agent_core.contracts import StrictModel


class RenderCorruption(StrictModel):
    exposure_multiplier: float = Field(default=1.0, ge=0.02, le=4.0)
    blur_sigma_px: float = Field(default=0.0, ge=0.0, le=4.0)
    occlusion_rect: list[float] | None = Field(default=None, min_length=4, max_length=4)

    # 功能：
    #   验证明确的合成遮挡矩形，不把真实未知遮挡伪装成标注。
    # 输入：
    #   self：曝光、模糊及归一化左上右下坐标。
    # 输出：
    #   self：位于图像内部且具有正面积的变换配置。
    @model_validator(mode="after")
    def validate_rect(self):
        if self.occlusion_rect is not None:
            left, top, right, bottom = self.occlusion_rect
            if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
                raise ValueError("VISION_CORRUPTION_RECT_INVALID")
        if 0 < self.blur_sigma_px < 1.0:
            raise ValueError("VISION_CORRUPTION_AMBIGUOUS_BLUR_NOT_LABELLED")
        return self


# 功能：
#   1. 对明确无镜头遮挡的静态渲染帧应用受控曝光、模糊或遮挡，保留可重现参数。
#   2. 遮挡区域标为未知背景，不能继续以看不见的物体作为可见语义监督。
# 输入：
#   rgb_bytes、mask_bytes、config：原始渲染 PNG、同刻类别 PNG 和明确变换配置。
# 输出：
#   rgb_result、mask_result、quality：派生 PNG 和已知模糊/遮挡标注；原字节不变。
def corrupt_rendered_pair(rgb_bytes, mask_bytes, config):
    config = RenderCorruption.model_validate(config.model_dump(mode="python"))
    with Image.open(BytesIO(rgb_bytes)) as image, Image.open(BytesIO(mask_bytes)) as mask:
        if image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
            raise ValueError("VISION_CORRUPTION_LAYOUT_INVALID")
        if image.width > 1280 or image.height > 720:
            raise ValueError("VISION_CORRUPTION_SIZE_BUDGET")
        rgb = image.copy()
        labels = np.array(mask, copy=True)
    try:
        if config.blur_sigma_px:
            replacement = rgb.filter(ImageFilter.GaussianBlur(config.blur_sigma_px))
            rgb.close()
            rgb = replacement
        pixels = np.asarray(rgb, dtype=np.float32) * config.exposure_multiplier
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        if config.occlusion_rect is not None:
            left, top, right, bottom = config.occlusion_rect
            width, height = rgb.size
            x0, y0 = int(left * width), int(top * height)
            x1, y1 = max(x0 + 1, int(right * width)), max(y0 + 1, int(bottom * height))
            pixels[y0:y1, x0:x1] = 0
            labels[y0:y1, x0:x1] = 0
        with Image.fromarray(pixels) as changed, Image.fromarray(labels) as changed_mask:
            rgb_buffer, mask_buffer = BytesIO(), BytesIO()
            changed.save(rgb_buffer, format="PNG")
            changed_mask.save(mask_buffer, format="PNG")
            rgb_result, mask_result = rgb_buffer.getvalue(), mask_buffer.getvalue()
    finally:
        rgb.close()
    quality = {"blurred": float(config.blur_sigma_px >= 1.0),
               "occluded": float(config.occlusion_rect is not None)}
    return rgb_result, mask_result, quality
