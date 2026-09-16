"""Bounded numerical sampling of map-validated rays; no occupancy or control authority."""

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float, float]
VoxelKey = tuple[int, int, int]
MAX_BATCH_RAYS = 64
MAX_BATCH_POINTS = 65_536


@dataclass(frozen=True, slots=True)
class MetricRaySample:
    """Immutable scalar snapshot taken before advancing an external ray iterator."""

    origin: Point
    endpoint: Point
    steps: int
    traversal_work: float
    hit: bool
    confidence: float
    observed_at_monotonic_seconds: float


# 功能：
#   1. 分组采样已经由地图入口验证的射线，保持原除法、乘法、加减及取整顺序。
#   2. 每块最多 65536 个采样点，保留逐射线有序去重和跨块端点，不合并占用证据。
# 输入：
#   rays：最多 64 条不可变射线标量快照。
#   minimum：地图三轴最小边界。
#   resolution_m：地图已经验证的正分辨率。
# 输出：
#   traversed：与输入顺序一一对应的体素键列表。
def sample_metric_rays(rays: list[MetricRaySample], minimum: Point, resolution_m: float):
    if not 1 <= len(rays) <= MAX_BATCH_RAYS or any(
        type(ray.steps) is not int or not 1 <= ray.steps <= 2_000_000 for ray in rays
    ):
        raise ValueError("METRIC_RAY_BATCH_INVALID")
    maximum_steps = max(ray.steps for ray in rays)
    if len(rays) > 1 and len(rays) * (maximum_steps + 1) > MAX_BATCH_POINTS:
        raise ValueError("METRIC_RAY_BATCH_PADDING_BUDGET_EXCEEDED")
    steps = np.asarray([ray.steps for ray in rays], dtype=np.int64)
    origins = np.asarray([ray.origin for ray in rays], dtype=np.float64)
    deltas = np.asarray([tuple(ray.endpoint[i] - ray.origin[i] for i in range(3))
                         for ray in rays], dtype=np.float64)
    inverse_resolution = 1. / resolution_m
    traversed: list[list[VoxelKey]] = [[] for _ in rays]
    columns = MAX_BATCH_POINTS // len(rays)
    for start in range(0, maximum_steps + 1, columns):
        indices = np.arange(start, min(maximum_steps + 1, start + columns), dtype=np.float64)
        # 短射线的填充位置固定在自身终点且不输出，不外推成额外已知空间。
        ratio = np.minimum(indices[None, :], steps[:, None]) / steps[:, None]
        coordinates = ratio[:, :, None] * deltas[:, None, :]
        coordinates += origins[:, None, :]
        coordinates -= minimum
        coordinates *= inverse_resolution
        keys = np.floor(coordinates).astype(np.int64)
        for row, ray in enumerate(rays):
            count = min(len(indices), max(0, ray.steps + 1 - start))
            if not count:
                continue
            current = keys[row, :count]
            keep = np.empty(count, dtype=np.bool_)
            previous = traversed[row]
            keep[0] = not previous or tuple(current[0]) != previous[-1]
            keep[1:] = np.any(current[1:] != current[:-1], axis=1)
            previous.extend(map(tuple, current[keep].tolist()))
    return traversed
