"""Depth-one RGB conversion cache with immutable original capture-receive identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .sensor_frame_clock import SensorFrameTime, require_model_frame_time

MAX_MODEL_PNG_BYTES = 64 * 1024 * 1024


# 功能：
#   在启动飞行相关会话前检查真实 Pillow 扩展能否创建、缩放和读取 RGB 像素。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def require_model_image_runtime() -> None:
    try:
        from PIL import Image

        with (
            Image.new("RGB", (2, 2), (17, 31, 47)) as source,
            source.resize((1, 1)) as resized,
        ):
            if resized.tobytes() != bytes((17, 31, 47)):
                raise RuntimeError("MODEL_RGB_CODEC_SELF_CHECK_FAILED")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "MODEL_RGB_RUNTIME_UNAVAILABLE:install declared Pillow dependency"
        ) from error


# 功能：
#   校验并固定模型图像的二维尺寸，拒绝非法容器、布尔分量及超出像素预算的尺寸。
# 输入：
#   size：由宽、高两个整数构成的列表或元组。
# 输出：
#   dimensions：不再引用调用方可变列表的宽高元组。
def model_image_dimensions(size) -> tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("MODEL_RGB_SIZE_INVALID")
    dimensions = tuple(size)
    if (len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions)
            or dimensions[0] > 4096 or dimensions[1] > 2160):
        raise ValueError("MODEL_RGB_SIZE_INVALID")
    return dimensions


# 功能：
#   先按实际字节数检查编码输出预算，再复制成不可变字节，不执行整数分配或列表转换。
# 输入：
#   value：编码器返回的 bytes、bytearray 或 memoryview。
#   minimum_bytes：允许的最小字节数。
#   maximum_bytes：允许的最大字节数；RGB 的上下限相同以要求完整像素。
# 输出：
#   owned：通过实际长度检查且不引用可变缓冲的字节内容。
def _owned_image_bytes(value, minimum_bytes: int, maximum_bytes: int) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID")
    try:
        count = value.nbytes if isinstance(value, memoryview) else len(value)
    except (TypeError, ValueError, BufferError) as error:
        raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID") from error
    if not minimum_bytes <= count <= maximum_bytes:
        raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID")
    try:
        owned = bytes(value)
    except (TypeError, ValueError, BufferError) as error:
        raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID") from error
    if len(owned) != count:
        raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID")
    return owned


@dataclass(frozen=True)
class PreparedModelImage:
    """Own converted image bytes while retaining their original source-time identity."""

    received_monotonic_seconds: float
    received_at_unix_ms: int
    size: tuple[int, int]
    png: bytes
    rgb: bytes
    png_sha256: str
    rgb_sha256: str
    frame_time: SensorFrameTime | None = None

    # 功能：
    #   生成含内联图像、摘要及原始时钟的多模态描述，只构造目标路径而不写文件。
    # 输入：
    #   self：已准备好的独立图像与来源记录。
    #   directory：调用方将来保存图片时使用的目录。
    # 输出：
    #   payload：含 PNG、RGB、尺寸及来源时间依据的描述字典。
    def multimodal(self, directory: Path) -> dict:
        path = directory / f"forward-rgb-{self.received_at_unix_ms}-{self.png_sha256[:12]}.png"
        payload = {
            "kind": "image-file",
            "path": str(path),
            "content_type": "image/png", "content_bytes": self.png,
            "content_sha256": self.png_sha256, "model_rgb_bytes": self.rgb,
            "model_rgb_sha256": self.rgb_sha256,
            "model_rgb_width": self.size[0], "model_rgb_height": self.size[1],
            "observed_at_unix_ms": self.received_at_unix_ms,
            "timestamp_basis": "host-receive-not-hardware-exposure",
        }
        if self.frame_time is not None and self.frame_time.clock_kind == "scene-source":
            payload.update({"timestamp_basis": "simulation-scene-capture",
                "scene_epoch": self.frame_time.epoch, "scene_sha256": self.frame_time.scene_sha256,
                "scene_sequence": self.frame_time.sequence,
                "scene_simulation_ns": self.frame_time.simulation_ns,
                "scene_source_unix_ns": self.frame_time.source_unix_ns,
                "host_received_unix_ns": self.frame_time.received_unix_ns})
        return payload


class ModelImageCache:
    """Reuse one source frame's conversion, never renew its control-validity deadline."""

    # 功能：
    #   建立只保留最近一份编码结果的缓存，实例由单一图像流的调用线程独占。
    # 输入：
    #   self：待初始化的图像缓存。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self._latest: PreparedModelImage | None = None

    # 功能：
    #   1. 校验来源时钟与尺寸，相同来源和输出尺寸复用编码，拒绝时间回退或重新标注来源。
    #   2. 校验并持有编码器输出的独立字节，全部成功后才替换缓存；不据此授予控制权限。
    # 输入：
    #   self：单一图像流独占的缓存。
    #   message：调用方持有且不会继续修改的来源消息。
    #   received_monotonic_seconds：历史参数名，实际表示已校验的来源单调钟秒数。
    #   received_at_unix_ms：历史参数名，实际表示原始来源 UNIX 毫秒。
    #   size：需要的模型图像宽高。
    #   decoder：将来源消息转换为 PNG 与 RGB 字节对的编码函数。
    #   frame_time：入口保留的完整来源时钟，普通直连调用可为 None。
    # 输出：
    #   result：图像记录和是否进行了新编码组成的元组。
    def prepare(
        self, message, *, received_monotonic_seconds: float, received_at_unix_ms: int,
        size: tuple[int, int], decoder, frame_time: SensorFrameTime | None = None
    ):
        require_model_frame_time(message, frame_time,
                                 received_monotonic_seconds, received_at_unix_ms)
        dimensions = model_image_dimensions(size)
        if not callable(decoder):
            raise ValueError("MODEL_RGB_DECODER_INVALID")
        previous = self._latest
        if previous is not None:
            if received_monotonic_seconds < previous.received_monotonic_seconds:
                raise ValueError("MODEL_RGB_SOURCE_TIME_REGRESSED")
            if received_monotonic_seconds == previous.received_monotonic_seconds:
                if (received_at_unix_ms != previous.received_at_unix_ms
                        or frame_time != previous.frame_time):
                    raise ValueError("MODEL_RGB_SOURCE_REDATED")
                if dimensions == previous.size:
                    result = previous, False
                    return result
        encoded = decoder(message, output_size=dimensions)
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise ValueError("MODEL_RGB_DECODED_PAYLOAD_INVALID")
        png = _owned_image_bytes(encoded[0], 1, MAX_MODEL_PNG_BYTES)
        rgb_size = dimensions[0] * dimensions[1] * 3
        rgb = _owned_image_bytes(encoded[1], rgb_size, rgb_size)
        # 摘要和描述使用同一份独立字节；失败的输出不得替换已验证缓存或推进来源时间。
        prepared = PreparedModelImage(
            received_monotonic_seconds, received_at_unix_ms, dimensions, png, rgb,
            hashlib.sha256(png).hexdigest(), hashlib.sha256(rgb).hexdigest(),
            frame_time,
        )
        self._latest = prepared
        result = prepared, True
        return result
