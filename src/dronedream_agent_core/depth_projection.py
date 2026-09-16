"""Project metric depth-camera frames into calibrated sensor-frame rays."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import RawRangeSample, Vector3
from .hashing import sha256_json

DEPTH_REDUCTION_SEMANTICS = (
    "all-source-pixels; nearest-radial-hit-per-tile at original pixel; "
    "otherwise one observed no-hit ray; invalid remains unknown; valid-pixel coverage"
)


# 功能：
#   按实际校准尺寸维持约二十列、十五行的角向区块预算，不上采样或虚构源像素信息。
# 输入：
#   width、height：深度图像宽高像素数。
# 输出：
#   stride：每个区块的像素边长。
def metric_depth_sample_stride(*, width: int, height: int) -> int:
    if (type(width) is not int or type(height) is not int
            or not 2 <= width <= 8192 or not 2 <= height <= 8192):
        raise ValueError("metric depth sampling dimensions are invalid")
    stride = max(1, min(width // 20, height // 15))
    return stride


@dataclass(frozen=True)
class DepthProjectionCalibration:
    """Pinhole intrinsics and explicit no-return meaning, bounded before pixel allocation."""
    width: int
    height: int
    horizontal_fov_rad: float
    minimum_depth_m: float
    maximum_depth_m: float
    sample_stride_pixels: int = 16
    confidence: float = 0.92
    # Missing hardware depth is NOT evidence of an unobstructed ray. Only a
    # source explicitly declaring Gazebo's clipping semantics may enable it.
    no_return_mode: Literal["unknown", "gazebo-far-clip"] = "unknown"
    fx_pixels: float | None = None
    fy_pixels: float | None = None
    cx_pixels: float | None = None
    cy_pixels: float | None = None

    # 功能：
    #   在像素分配前验证光学、量程、置信度、可选显式内参及像素／射线预算。
    # 输入：
    #   self：本次不可变校准配置。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        metric_depth_sample_stride(width=self.width, height=self.height)
        if any(isinstance(v, bool) or not isinstance(v, int | float)
               or not -sys.float_info.max <= v <= sys.float_info.max
               for v in (self.horizontal_fov_rad, self.minimum_depth_m,
                         self.maximum_depth_m, self.confidence)):
            raise ValueError("depth calibration requires finite numeric values")
        if self.width * self.height > 4_194_304:
            raise ValueError("depth image pixel budget exceeded")
        if not 0.1 <= self.horizontal_fov_rad < math.pi:
            raise ValueError("horizontal field of view is invalid")
        if not math.isfinite(self.minimum_depth_m) or self.minimum_depth_m < 0.0:
            raise ValueError("minimum depth must not be negative")
        if not self.minimum_depth_m < self.maximum_depth_m <= 1_000.0:
            raise ValueError("maximum depth must exceed minimum depth")
        if type(self.sample_stride_pixels) is not int or not 1 <= self.sample_stride_pixels <= 8192:
            raise ValueError("sample stride must be positive")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("projection confidence must be in (0, 1]")
        if type(self.no_return_mode) is not str or self.no_return_mode not in {
            "unknown", "gazebo-far-clip"
        }:
            raise ValueError("depth no-return semantics are invalid")
        explicit = (self.fx_pixels, self.fy_pixels, self.cx_pixels, self.cy_pixels)
        if any(value is not None for value in explicit):
            if any(isinstance(v, bool) or not isinstance(v, int | float)
                   or not -sys.float_info.max <= v <= sys.float_info.max for v in explicit):
                raise ValueError("depth intrinsics must be complete and finite")
            if (min(self.fx_pixels, self.fy_pixels) <= 0
                    or not 0 <= self.cx_pixels < self.width
                    or not 0 <= self.cy_pixels < self.height):
                raise ValueError("depth intrinsics are outside the calibrated image")
        if self.tile_count > 250_000:
            raise ValueError("depth output ray budget exceeded")
        fx, fy, cx, cy = self.intrinsics
        horizontal = max(cx, self.width - 1 - cx) / fx
        vertical = max(cy, self.height - 1 - cy) / fy
        if not math.isfinite(self.maximum_depth_m**2 * (
                1. + horizontal * horizontal + vertical * vertical)):
            raise ValueError("depth intrinsics exceed finite projection geometry")

    # 功能：
    #   计算包含右侧和底部不完整区块在内的最坏射线输出数量。
    # 输入：
    #   self：当前校准及区块边长。
    # 输出：
    #   count：完整覆盖源图像所需的区块数。
    @property
    def tile_count(self) -> int:
        stride = self.sample_stride_pixels
        count = ((self.width + stride - 1) // stride) * ((self.height + stride - 1) // stride)
        return count

    # 功能：
    #   优先返回实测显式内参；仅在未提供时按视场角和方形像素假设计算针孔内参。
    # 输入：
    #   self：当前图像和光学校准。
    # 输出：
    #   intrinsics：fx、fy、cx、cy 四项像素坐标参数。
    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        if self.fx_pixels is not None:
            intrinsics = self.fx_pixels, self.fy_pixels, self.cx_pixels, self.cy_pixels
            return intrinsics
        focal = self.width / (2.0 * math.tan(self.horizontal_fov_rad / 2.0))
        intrinsics = focal, focal, (self.width - 1) / 2.0, (self.height - 1) / 2.0
        return intrinsics

    # 功能：
    #   将校准数值、像素编码及归约语义一起绑定，避免同参数但不同预处理被误认同一输入。
    # 输入：
    #   self：当前校准。
    # 输出：
    #   payload：校准内容身份字典。
    @property
    def identity_payload(self) -> dict:
        payload = {**asdict(self), "intrinsics_pixels": self.intrinsics,
                "pixel_format": "little-endian-axial-float32",
                "reduction": DEPTH_REDUCTION_SEMANTICS}
        return payload

    # 功能：
    #   计算校准内容摘要，不将摘要自洽视为来源可信或飞行授权。
    # 输入：
    #   self：当前校准。
    # 输出：
    #   digest：完整校准身份的 SHA-256。
    @property
    def sha256(self) -> str:
        digest = sha256_json(self.identity_payload)
        return digest


@dataclass(frozen=True)
class ProjectedDepthFrame:
    """Reduced metric rays with measured source coverage; no full-frustum free-space claim."""
    samples: tuple[RawRangeSample, ...]
    source_pixel_count: int
    valid_pixel_count: int
    calibration_sha256: str

    # 功能：
    #   计算实测有效源像素比例，不据此宣称整个视锥或每个区块全部无障碍。
    # 输入：
    #   self：当前投影结果及像素计数。
    # 输出：
    #   coverage：有效像素占全部源像素的比例。
    @property
    def source_coverage(self) -> float:
        coverage = self.valid_pixel_count / self.source_pixel_count
        return coverage


# 功能：
#   为仍在调用的检查脚本提供列表接口，复用唯一正式投影实现，不维护旧的平行算法。
# 输入：
#   data：小端轴向浮点像素缓冲。
#   row_step_bytes：每行实际字节跨度，包含行填充。
#   calibration：已验证的光学及量程配置。
# 输出：
#   samples：校准米制射线列表。
def project_float32_depth_image(*, data: bytes, row_step_bytes: int,
                               calibration: DepthProjectionCalibration) -> list[RawRangeSample]:
    samples = list(project_metric_depth_frame(
        data=data, row_step_bytes=row_step_bytes, calibration=calibration,
    ).samples)
    return samples


# 功能：
#   1. 检查所有源像素，每个区块保留最近径向障碍及该像素的真实方向，避免漏掉细障碍。
#   2. 无命中区块只输出真实观测到的无回波射线；不平滑补洞，不把缺失区域当作自由空间。
#   3. 先按实际字节数和行布局验证缓冲，再冻结像素以避免可变输入在计算中漂移。
# 输入：
#   data：bytes、bytearray 或连续 memoryview 形式的小端轴向 float32 图像。
#   row_step_bytes：包含行填充的实际字节跨度。
#   calibration：不可变针孔内参、量程及无回波语义。
# 输出：
#   result：校准射线、全图像素计数、有效像素数和校准摘要。
def project_metric_depth_frame(
    *, data: bytes, row_step_bytes: int, calibration: DepthProjectionCalibration,
) -> ProjectedDepthFrame:
    if not isinstance(calibration, DepthProjectionCalibration):
        raise ValueError("depth projection requires calibrated optics")
    minimum_step = calibration.width * 4
    if type(row_step_bytes) is not int or row_step_bytes < minimum_step:
        raise ValueError("depth image row step is smaller than width * sizeof(float)")
    required_bytes = row_step_bytes * calibration.height
    if type(data) not in (bytes, bytearray, memoryview):
        raise ValueError("depth image requires a pixel byte buffer")
    if isinstance(data, memoryview) and not data.c_contiguous:
        raise ValueError("depth image buffer must be contiguous")
    size = data.nbytes if isinstance(data, memoryview) else len(data)
    if row_step_bytes > minimum_step + 65_536 or size > 32 * 1024 * 1024:
        raise ValueError("depth image buffer budget exceeded")
    if size < required_bytes:
        raise ValueError("depth image payload is truncated")
    if size != required_bytes:
        raise ValueError("depth image payload contains unexpected trailing bytes")
    data = data if type(data) is bytes else bytes(data)
    if len(data) != required_bytes:
        raise ValueError("depth image buffer changed while freezing")

    # One vectorized implementation, not an accelerated/legacy dual path.
    # The input view respects byte strides and excludes row padding. Float64
    # geometry prevents overflow when squaring corrupted finite float32 data.
    import numpy as np

    depth = np.ndarray((calibration.height, calibration.width), dtype="<f4",
                       buffer=data, strides=(row_step_bytes, 4)).astype(np.float64)
    fx, fy, center_x, center_y = calibration.intrinsics
    horizontal = (center_x - np.arange(calibration.width)) / fx
    vertical = (center_y - np.arange(calibration.height)) / fy
    far = calibration.maximum_depth_m
    finite = np.isfinite(depth)
    valid = finite & (depth >= calibration.minimum_depth_m) & (depth <= far + 1e-5)
    if calibration.no_return_mode == "gazebo-far-clip":
        valid |= np.isposinf(depth)
    else:
        valid &= depth < far - 1e-5
    # Invalid or no-return depth never participates in geometry arithmetic.
    # This also avoids overflow/NaN warnings from corrupted finite float32 data.
    measured_depth = np.where(valid & finite, np.minimum(depth, far), 0.)
    radial_squared = measured_depth**2 * (1. + horizontal[None, :]**2 + vertical[:, None]**2)
    hit = valid & finite & (radial_squared < far * far) & (depth < far - 1e-5)
    hit_cost = np.where(hit, radial_squared, np.inf)
    samples = []
    stride = calibration.sample_stride_pixels
    # Distance to a real tile-center pixel is only a tie-break for no-hit rays.
    col_cost = np.abs(np.arange(calibration.width) % stride - stride // 2)
    row_cost = np.abs(np.arange(calibration.height) % stride - stride // 2)
    for top in range(0, calibration.height, stride):
        for left in range(0, calibration.width, stride):
            bottom, right = min(top+stride, calibration.height), min(left+stride, calibration.width)
            costs = hit_cost[top:bottom, left:right]
            flat = int(costs.argmin())
            y, x = divmod(flat, right-left)
            value = float(costs[y, x])
            is_hit = math.isfinite(value)
            if not is_hit:
                tile_valid = valid[top:bottom, left:right]
                if not tile_valid.any():
                    continue
                free_cost = np.where(tile_valid,
                    row_cost[top:bottom, None] + col_cost[None, left:right], np.inf)
                y, x = divmod(int(free_cost.argmin()), right-left)
            row, col = top+y, left+x
            samples.append(RawRangeSample(
                direction_sensor=Vector3(x=1.0, y=float(horizontal[col]), z=float(vertical[row])),
                range_m=math.sqrt(value) if is_hit else far,
                hit=is_hit, confidence=calibration.confidence,
            ))
    valid_pixels = int(np.count_nonzero(valid))
    if not samples:
        raise ValueError("depth image contains no calibrated metric samples")
    result = ProjectedDepthFrame(tuple(samples), calibration.width * calibration.height,
                                 valid_pixels, calibration.sha256)
    return result
