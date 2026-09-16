"""Conservative lost-track clearance from fresh metric free-space observations.

Track age alone never clears an obstacle. Every voxel of its acceleration- and
localization-inflated reachable box must have been traversed by the current
scan, beyond the hit exclusion margin. Qualified-map priors do not count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

from .contracts import DynamicObstacleObservation, RangeRayObservation


# 功能：
#   校验边界物理数值，避免布尔值及不可表示的大整数进入距离或时间计算。
# 输入：
#   value：候选物理标量。
# 输出：
#   valid：是否为有限且可表示的数值。
def _finite_number(value: object) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   收集新鲜高置信度射线穿过的空闲体素，命中优先；扫描预算耗尽则舍弃全部局部证据。
#   这是给定分辨率下的离散证据，不证明两条射线之间的每一点都为空闲。
# 输入：
#   rays：已经对齐世界坐标与单调时钟的距离射线。
#   resolution_m：体素边长，单位米。
#   now_monotonic_seconds：当前单调时钟秒数。
#   maximum_age_seconds：允许的观测最大年龄，单位秒。
#   maximum_samples：射线检查及投影采样合计的工作预算。
# 输出：
#   free_voxels：本次完整扫描得到、且排除了命中位置的体素键集合。
def fresh_free_voxels(
    rays: Sequence[RangeRayObservation],
    *,
    resolution_m: float,
    now_monotonic_seconds: float,
    maximum_age_seconds: float = 0.25,
    maximum_samples: int = 100_000,
) -> set[tuple[int, int, int]]:
    if (
        type(resolution_m) not in (int, float)
        or not 0.05 <= resolution_m <= 1.0
        or not _finite_number(now_monotonic_seconds)
        or now_monotonic_seconds < 0
        or not _finite_number(maximum_age_seconds)
        or maximum_age_seconds < 0
        or type(maximum_samples) is not int
        or not 1 <= maximum_samples <= 100_000
    ):
        raise ValueError("DYNAMIC_CLEARANCE_GRID_INVALID")
    free, hit_keys = set(), set()
    samples = 0
    for ray in rays:
        # 旧帧、低置信度及零长度射线也会占用 CPU，不能只计算最终生成的投影样本。
        samples += 1
        if samples > maximum_samples:
            return set()
        if (
            not _finite_number(ray.observed_at_monotonic_seconds)
            or type(ray.hit) is not bool
            or not _finite_number(ray.confidence)
        ):
            return set()
        age = now_monotonic_seconds - ray.observed_at_monotonic_seconds
        if not 0 <= age <= maximum_age_seconds:
            continue
        start, end = (
            (ray.origin_m.x, ray.origin_m.y, ray.origin_m.z),
            (ray.endpoint_m.x, ray.endpoint_m.y, ray.endpoint_m.z),
        )
        if not all(_finite_number(value) for value in (*start, *end)):
            return set()
        distance = math.dist(start, end)
        if not math.isfinite(distance) or not all(
            math.isfinite(v / resolution_m) for v in (*start, *end)
        ):
            return set()
        if ray.hit:
            hit_keys.add(tuple(math.floor(v / resolution_m) for v in end))
        # 起点上的命中仍是障碍，必须先记录，再跳过没有长度的空闲射线。
        if distance == 0:
            continue
        if not 0.8 <= ray.confidence <= 1.0:
            continue
        # Exclude the terminal voxel and its diagonal footprint from free evidence.
        free_distance = max(0.0, distance - math.sqrt(3) * resolution_m)
        proposed_count = free_distance / (resolution_m * 0.5)
        if not math.isfinite(proposed_count) or proposed_count > maximum_samples - samples:
            return set()
        count = math.ceil(proposed_count)
        samples += count
        if samples > maximum_samples:
            return set()  # bounded realtime work; no partial evidence is a clearance
        sx, sy, sz = start
        dx, dy, dz = (end[0] - sx, end[1] - sy, end[2] - sz)
        floor = math.floor
        for index in range(count):
            fraction = (index * resolution_m * 0.5) / distance
            # 与原逐轴表达式保持算术顺序，不在每个采样点创建生成器和重复计算方向。
            free.add((floor((sx + fraction * dx) / resolution_m),
                      floor((sy + fraction * dy) / resolution_m),
                      floor((sz + fraction * dz) / resolution_m)))
    free_voxels = free - hit_keys
    return free_voxels


# 功能：
#   要求动态目标可达包络的每个体素都有新鲜空闲证据，包络包含加速和三倍定位标准差。
#   失联超过一秒、体积无效、溢出或预算不足均保留未解决障碍，不以时间流逝清除目标。
# 输入：
#   track：上次观测到的目标位置、速度与几何尺寸。
#   age_seconds：从原观测到本次判断的时间，单位秒。
#   free_voxels：完整新扫描提供的空闲体素。
#   resolution_m：体素边长，单位米。
#   localization_variance_m2：定位方差，单位平方米。
#   maximum_acceleration_mps2：目标可能的加速度上限，单位米每平方秒。
#   maximum_voxels：允许完整检查的目标包络体素数上限。
# 输出：
#   observed_free：本次扫描是否完整覆盖该目标的保守可达包络。
def reachable_track_box_observed_free(
    track: DynamicObstacleObservation,
    *,
    age_seconds: float,
    free_voxels: set[tuple[int, int, int]],
    resolution_m: float,
    localization_variance_m2: float,
    maximum_acceleration_mps2: float,
    maximum_voxels: int = 4096,
) -> bool:
    # Past one second the identity/acceleration prediction is no longer accepted
    # as a compact clearance volume. Reacquisition or a new verified scan is required.
    if (
        type(age_seconds) not in (int, float)
        or not 0.1 <= age_seconds <= 1.0
        or type(localization_variance_m2) not in (int, float)
        or not 0 < localization_variance_m2 <= 0.25
    ):
        return False
    if (
        type(resolution_m) not in (int, float)
        or not 0.05 <= resolution_m <= 1.0
        or type(maximum_acceleration_mps2) not in (int, float)
        or not 0 < maximum_acceleration_mps2 <= 50
        or type(maximum_voxels) is not int
        or not 1 <= maximum_voxels <= 4096
    ):
        raise ValueError("DYNAMIC_CLEARANCE_LIMIT_INVALID")
    if (
        type(track.radius_m) not in (int, float)
        or not 0 < track.radius_m <= 20.0
        or type(track.height_m) not in (int, float)
        or not 0 < track.height_m <= 50.0
    ):
        return False
    if not all(
        _finite_number(value)
        for value in (
            track.position_m.x,
            track.position_m.y,
            track.position_m.z,
            track.velocity_mps.x,
            track.velocity_mps.y,
            track.velocity_mps.z,
        )
    ):
        return False
    inflation = 0.5 * maximum_acceleration_mps2 * age_seconds**2 + 3 * math.sqrt(
        localization_variance_m2
    )
    center = tuple(
        p + v * age_seconds
        for p, v in zip(
            (track.position_m.x, track.position_m.y, track.position_m.z),
            (track.velocity_mps.x, track.velocity_mps.y, track.velocity_mps.z),
            strict=True,
        )
    )
    half = (track.radius_m + inflation, track.radius_m + inflation, track.height_m / 2 + inflation)
    coordinates = [
        ((center[i] - half[i]) / resolution_m, (center[i] + half[i]) / resolution_m)
        for i in range(3)
    ]
    if not all(math.isfinite(value) for pair in coordinates for value in pair):
        return False
    bounds = [(math.floor(low), math.floor(high) + 1) for low, high in coordinates]
    # Compare integer widths before constructing ranges: len(range(...)) can
    # itself overflow for malformed finite coordinates beyond machine indices.
    if math.prod(high - low for low, high in bounds) > maximum_voxels:
        return False
    spans = [range(low, high) for low, high in bounds]
    observed_free = all(key in free_voxels for key in product(*spans))
    return observed_free
