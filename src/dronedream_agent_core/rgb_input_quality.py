"""Image layout and observability, not learned accuracy or a free-space decision."""

import hashlib
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageStat

RGB_QUALITY_SEMANTICS = "luminance-p05-p95; spatial-neighbours; exposure; no-padding-or-alpha"
_PIXEL_FORMATS = {3: ("RGB", "RGB", 3), 4: ("RGBA", "RGBA", 4),
                  8: ("RGB", "BGR", 3), 5: ("RGBA", "BGRA", 4)}


@dataclass(frozen=True, slots=True)
class RgbFramePixels:
    """Owned pixel bytes and layout; source clock remains separately bound at ingress."""

    width: int
    height: int
    step: int
    pixel_format_type: int
    data: bytes


# 功能：
#   1. 验证 RGB 布局与字节预算后复制可变缓冲，拒绝把整数解释为 bytes 分配长度。
#   2. 将布局和像素固定为同一份不可变记录，供解码、质量与摘要共同使用。
# 输入：
#   message：包含图像尺寸、行跨度、像素格式及真实字节缓冲的消息。
# 输出：
#   snapshot：完成布局校验的不可变像素快照。
def freeze_gazebo_rgb(message) -> RgbFramePixels:
    width, height, step, pixel_format = (
        getattr(message, name, None) for name in ("width", "height", "step", "pixel_format_type")
    )
    if (any(type(v) is not int for v in (width, height, step, pixel_format))
            or not 1 <= width <= 4096 or not 1 <= height <= 2160):
        raise ValueError("FORWARD_RGB_DIMENSIONS_INVALID")
    if pixel_format not in _PIXEL_FORMATS:
        raise ValueError("FORWARD_RGB_FORMAT_UNSUPPORTED")
    channels = _PIXEL_FORMATS[pixel_format][2]
    if not width * channels <= step <= width * channels + 65_536:
        raise ValueError("FORWARD_RGB_ROW_STEP_INVALID")
    data = getattr(message, "data", None)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("FORWARD_RGB_PAYLOAD_TYPE_INVALID")
    try:
        size = data.nbytes if isinstance(data, memoryview) else len(data)
    except (TypeError, ValueError, BufferError) as error:
        raise ValueError("FORWARD_RGB_PAYLOAD_TYPE_INVALID") from error
    # 先检查实际字节预算，再分配快照；memoryview 的元素数不一定等于字节数。
    if not step * height <= size <= 64 * 1024 * 1024:
        raise ValueError("FORWARD_RGB_FRAME_INCOMPLETE")
    try:
        payload = bytes(data)
    except (TypeError, ValueError, BufferError) as error:
        raise ValueError("FORWARD_RGB_PAYLOAD_TYPE_INVALID") from error
    if len(payload) != size:
        raise ValueError("FORWARD_RGB_FRAME_CHANGED")
    snapshot = RgbFramePixels(width, height, step, pixel_format, payload)
    return snapshot


# 功能：
#   按共享像素格式和真实行跨度解码快照，忽略行填充与透明通道，不静默修补损坏布局。
# 输入：
#   message：原始图像消息或已固定的像素快照。
# 输出：
#   rgb：由调用方负责关闭的三通道 Pillow 图像。
def decode_gazebo_rgb(message) -> Image.Image:
    snapshot = freeze_gazebo_rgb(message)
    mode, raw_mode, _ = _PIXEL_FORMATS[snapshot.pixel_format_type]
    with Image.frombytes(mode, (snapshot.width, snapshot.height), snapshot.data,
                         "raw", raw_mode, snapshot.step, 1) as source:
        rgb = source.convert("RGB")
    return rgb


@dataclass(frozen=True)
class RgbInputQuality:
    score: float
    robust_luminance_span: float
    mean_spatial_difference: float
    exposure_clipped_fraction: float
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class PreparedRgbMeasurement:
    snapshot: RgbFramePixels
    sha256: str
    quality: RgbInputQuality


# 功能：
#   在图像工作线程中完成原始像素的质量计算，固定同一份像素、摘要和测量结果。
# 输入：
#   message：工作器独占的原始相机消息。
# 输出：
#   measurement：不携新时钟的不可变质量结果；消费端仍校验原始帧与年龄。
def prepare_rgb_measurement(message) -> PreparedRgbMeasurement:
    snapshot = freeze_gazebo_rgb(message)
    with decode_gazebo_rgb(snapshot) as rgb:
        quality = measure_rgb_input_quality(rgb)
    measurement = PreparedRgbMeasurement(
        snapshot, hashlib.sha256(snapshot.data).hexdigest(), quality)
    return measurement


# 功能：
#   1. 用亮度分位差、真实相邻像素差和曝光剪切比例评估图像可观测性。
#   2. 孤立亮点不代表纹理；这些统计不证明语义、标定、模糊程度或飞行安全。
#   3. 关闭内部临时图像，保持调用方输入图像可用。
# 输入：
#   rgb：非空且在解码尺寸预算内的三通道 Pillow 图像。
# 输出：
#   quality：包含质量分数、亮度跨度、邻域差、曝光比例及问题代码的不可变记录。
def measure_rgb_input_quality(rgb: Image.Image) -> RgbInputQuality:
    if (not isinstance(rgb, Image.Image) or rgb.mode != "RGB"
            or not 1 <= rgb.width <= 4096 or not 1 <= rgb.height <= 2160):
        raise ValueError("FORWARD_RGB_QUALITY_IMAGE_INVALID")
    with rgb.convert("L") as gray:
        histogram = gray.histogram()
        count = gray.width * gray.height
        width, height = gray.size
        differences = []
        if width > 1:
            with (gray.crop((0, 0, width-1, height)) as left,
                  gray.crop((1, 0, width, height)) as right,
                  ImageChops.difference(left, right) as horizontal):
                differences.append(ImageStat.Stat(horizontal).mean[0])
        if height > 1:
            with (gray.crop((0, 0, width, height-1)) as top,
                  gray.crop((0, 1, width, height)) as bottom,
                  ImageChops.difference(top, bottom) as vertical):
                differences.append(ImageStat.Stat(vertical).mean[0])
    cumulative, low, high = 0, None, 255
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        if low is None and cumulative >= max(1, count * .05):
            low = value
        if cumulative >= count * .95:
            high = value
            break
    spatial = sum(differences) / len(differences) if differences else 0.
    clipped = (sum(histogram[:4]) + sum(histogram[252:])) / count
    span = float(high - low)
    score = min(1., span / 32., spatial / 2.) * (1. - clipped)
    issues = []
    if span < 8 or spatial < .25:
        issues.append("RGB_LOW_SPATIAL_INFORMATION")
    if clipped > .5:
        issues.append("RGB_EXPOSURE_CLIPPED")
    quality = RgbInputQuality(score, span, spatial, clipped, tuple(issues))
    return quality
