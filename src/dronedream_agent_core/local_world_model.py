"""Metric occupancy model for depth/Lidar/VIO-driven local autonomy.

Unknown cells are blocked until calibrated range evidence marks them free.  The
language model may select a semantic goal or frontier, but never authors metric
free space or bypasses this planner.
"""

from __future__ import annotations

import heapq
import math
import re
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from itertools import islice, product

import numpy as np

from .contracts import (
    DynamicObstacleObservation,
    MetricVoxelMapSnapshot,
    RangeRayObservation,
    Vector3,
)
from .hashing import sha256_json
from .occupancy_collision import bounded_occupied_cell_boxes
from .perception_evidence import FrozenVector3

VoxelKey = tuple[int, int, int]
Point = tuple[float, float, float]
SECTOR_LABELS = (
    "front",
    "front-left",
    "left",
    "rear-left",
    "rear",
    "rear-right",
    "right",
    "front-right",
)
_OCCUPIED_LOG_ODDS_THRESHOLD = math.log(0.65 / 0.35)
_FREE_LOG_ODDS_THRESHOLD = math.log(0.35 / 0.65)
_MAX_GRID_WORK = 2_000_000


# 功能：
#   在公开数值边界拒绝非有限值和布尔伪数值，避免无界遍历或方向反转。
# 输入：
#   value：待核对数值；name：错误字段名；minimum、maximum：允许的闭区间。
# 输出：
#   result：有限浮点数。
def _number(value: float, name: str, minimum: float = 0.0, maximum: float = 1e9) -> float:
    if (
        type(value) not in (int, float)
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    result = float(value)
    return result


# 功能：
#   有界消费外部集合，超限时整体拒绝而非静默截断环境或路线。
# 输入：
#   values：待消费序列；limit：最大项数；name：对应错误字段。
# 输出：
#   result：独立元组容器。
def _bounded_items(values: Iterable, limit: int, name: str) -> tuple:
    result = tuple(islice(values, limit + 1))
    if len(result) > limit:
        raise ValueError(f"{name} exceeds item budget")
    return result


# 功能：
#   严格复制三维向量，使地图边界和绑定路线不可被调用者后续改写。
# 输入：
#   value：米制位置或速度向量。
# 输出：
#   result：重新验证且不可变的三维向量。
def _frozen_vector(value: Vector3) -> FrozenVector3:
    if not isinstance(value, Vector3):
        raise ValueError("METRIC_VECTOR_TYPE_INVALID")
    result = FrozenVector3.model_validate(value.model_dump(mode="python"), strict=True)
    if any(abs(component) > 1e9 for component in (result.x, result.y, result.z)):
        raise ValueError("METRIC_VECTOR_MAGNITUDE_EXCEEDED")
    return result


@dataclass(frozen=True)
class PreparedMetricScan:
    """Owned immutable update; preparation cannot change map or registry state."""

    owner: object
    previous_observation_count: int
    previous_source_time: float | None
    observed_at: float
    ray_count: int
    occupied_strengths: tuple[tuple[VoxelKey, float], ...]
    free_strengths: tuple[tuple[VoxelKey, float], ...]


# 功能：
#   将已验证三维向量取为几何计算使用的坐标元组，不改变坐标系。
# 输入：
#   value：三维向量。
# 输出：
#   result：按 x、y、z 排列的三元组。
def _point(value: Vector3) -> Point:
    result = value.x, value.y, value.z
    return result


# 功能：
#   计算向量长度，避免先逐项平方造成可避免的中间溢出。
# 输入：
#   value：三维分量。
# 输出：
#   result：欧氏长度。
def _magnitude(value: Point) -> float:
    result = math.hypot(*value)
    return result


# 功能：
#   计算点到有限线段的最近距离，用于重复经过同一任务节点时匹配当前路段。
# 输入：
#   point：当前位置；start、end：路段两端。
# 输出：
#   distance：截断投影后的最近距离。
def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    delta = tuple(end[index] - start[index] for index in range(3))
    length_squared = sum(component * component for component in delta)
    if length_squared <= 1e-12:
        return math.dist(point, start)
    progress = max(
        0.0,
        min(
            1.0,
            sum((point[index] - start[index]) * delta[index] for index in range(3))
            / length_squared,
        ),
    )
    projection = tuple(start[index] + delta[index] * progress for index in range(3))
    distance = math.dist(point, projection)
    return distance


# 功能：
#   累加折线各段的实际米制长度，不以摘要下采样点代替完整路径。
# 输入：
#   path：按飞行顺序排列的坐标。
# 输出：
#   length：折线总长。
def _path_length(path: list[Vector3]) -> float:
    length = sum(
        math.dist(_point(first), _point(second))
        for first, second in zip(path, path[1:], strict=False)
    )
    return length


# 功能：
#   为模型摘要均匀选取路径点并保留两端，完整几何另由摘要绑定。
# 输入：
#   path：完整路径；limit：至少两个的展示点上限。
# 输出：
#   payload：展示用坐标字典列表。
def _bounded_path_points(path: list[Vector3], *, limit: int = 24) -> list[dict[str, float]]:
    if type(limit) is not int or not 2 <= limit <= 4096:
        raise ValueError("path summary limit must be in [2, 4096]")
    if len(path) <= limit:
        selected = path
    else:
        indices = sorted({round(index * (len(path) - 1) / (limit - 1)) for index in range(limit)})
        selected = [path[index] for index in indices]
    payload = [point.model_dump(mode="json") for point in selected]
    return payload


# 功能：
#   用三个轴的参数区间相交检查完整线段与轴对齐盒是否接触，不依赖采样点命中。
# 输入：
#   first、second：线段端点；center：盒中心；half_extent：各轴半长。
# 输出：
#   intersects：包含边界接触的相交标志。
def _segment_hits_box(first: Point, second: Point, center: Point, half_extent: float) -> bool:
    enter, leave = 0.0, 1.0
    for axis in range(3):
        delta = second[axis] - first[axis]
        low, high = center[axis] - half_extent, center[axis] + half_extent
        if delta == 0.0:
            if first[axis] < low or first[axis] > high:
                return False
            continue
        near, far = sorted(((low - first[axis]) / delta, (high - first[axis]) / delta))
        enter, leave = max(enter, near), min(leave, far)
        if enter > leave:
            return False
    intersects = True
    return intersects


# 功能：
#   按路径时间预测动态目标，扣除观测年龄、预测及采样间隙余量后评估净空。
# 输入：
#   path：完整折线；dynamic_obstacles：已验证目标；path_speed_mps：候选速度。
#   vehicle_radius_m、vehicle_height_m：机身包络；sample_spacing_m：路径采样间距。
# 输出：
#   result：最小预测净空、对应目标标识及抵达该状态的时间三元组。
def _dynamic_path_clearance(
    path: list[Vector3],
    *,
    dynamic_obstacles: tuple[DynamicObstacleObservation, ...],
    path_speed_mps: float,
    vehicle_radius_m: float,
    vehicle_height_m: float,
    sample_spacing_m: float,
) -> tuple[float, str | None, float]:
    path_speed_mps = _number(path_speed_mps, "dynamic path speed", minimum=1e-6)
    sample_spacing_m = _number(sample_spacing_m, "dynamic sample spacing", minimum=1e-6)
    if not path:
        raise ValueError("DYNAMIC_PATH_EMPTY")
    if not dynamic_obstacles:
        return 999.0, None, 0.0
    elapsed = 0.0
    minimum = 999.0
    threat: str | None = None
    time_to_minimum = 0.0
    work = 0
    for start, end in list(zip(path, path[1:], strict=False)) or [(path[0], path[0])]:
        first = _point(start)
        second = _point(end)
        distance = math.dist(first, second)
        sample_count = max(1, math.ceil(distance / sample_spacing_m))
        work += (sample_count + 1) * len(dynamic_obstacles)
        if work > _MAX_GRID_WORK:
            raise ValueError("DYNAMIC_PATH_WORK_BUDGET_EXCEEDED")
        for sample_index in range(sample_count + 1):
            ratio = sample_index / sample_count
            position = tuple(
                first[axis] + (second[axis] - first[axis]) * ratio for axis in range(3)
            )
            sample_elapsed = elapsed + distance * ratio / path_speed_mps
            for obstacle in dynamic_obstacles:
                obstacle_speed = _magnitude(_point(obstacle.velocity_mps))
                prediction_seconds = obstacle.age_seconds + sample_elapsed
                predicted = tuple(
                    _point(obstacle.position_m)[axis]
                    + _point(obstacle.velocity_mps)[axis] * prediction_seconds
                    for axis in range(3)
                )
                uncertainty = (1.0 - obstacle.confidence) * 0.5
                uncertainty += obstacle.age_seconds * obstacle_speed * 0.25
                # Prediction uncertainty grows with horizon; a path that is
                # only safe under an exact constant-velocity forecast is not
                # authorized for a language-model selection.
                uncertainty += sample_elapsed * 0.08
                # 两次采样之间相对运动仍连续存在，扣除最大半步相对位移，
                # 防止快速目标只在采样间隙穿过航线而被判为安全。
                uncertainty += (
                    (path_speed_mps + obstacle_speed + 0.08)
                    * distance
                    / (sample_count * path_speed_mps * 2.0)
                )
                horizontal = math.hypot(position[0] - predicted[0], position[1] - predicted[1])
                horizontal -= vehicle_radius_m + obstacle.radius_m + uncertainty
                vertical = abs(position[2] - predicted[2])
                vertical -= (vehicle_height_m + obstacle.height_m) / 2 + uncertainty
                clearance = (
                    math.hypot(max(0.0, horizontal), max(0.0, vertical))
                    if horizontal > 0.0 and vertical > 0.0
                    else max(horizontal, vertical)
                )
                if not math.isfinite(clearance):
                    raise ValueError("DYNAMIC_PATH_PREDICTION_NONFINITE")
                if clearance < minimum:
                    minimum = clearance
                    threat = obstacle.obstacle_id
                    time_to_minimum = sample_elapsed
        elapsed += distance / path_speed_mps
    result = minimum, threat, time_to_minimum
    return result


@dataclass(slots=True)
class _VoxelEvidence:
    log_odds: float = 0.0
    observations: int = 0
    latest_monotonic_seconds: float = 0.0
    owner: object | None = None


class MetricVoxelMap:
    """A bounded conservative 3-D occupancy map."""

    _NAVIGATION_CHUNK_EDGE_VOXELS = 16

    # 功能：
    #   建立独立局部体素地图与写时复制索引，默认阻止进入尚未观测的空间。
    # 输入：
    #   resolution_m：分辨率；minimum_bound_m、maximum_bound_m：边界。
    #   unknown_is_occupied：未知空间策略。
    # 输出：
    #   无：初始化 self 的地图、先验、路线与观测索引。
    def __init__(
        self,
        *,
        resolution_m: float,
        minimum_bound_m: Vector3,
        maximum_bound_m: Vector3,
        unknown_is_occupied: bool = True,
    ) -> None:
        resolution_m = _number(resolution_m, "resolution_m", maximum=5.0)
        if resolution_m <= 0.02:
            raise ValueError("resolution_m must be in (0.02, 5.0]")
        if type(unknown_is_occupied) is not bool:
            raise ValueError("unknown_is_occupied must be boolean")
        minimum_bound_m, maximum_bound_m = map(_frozen_vector, (minimum_bound_m, maximum_bound_m))
        minimum = _point(minimum_bound_m)
        maximum = _point(maximum_bound_m)
        if any(minimum[index] >= maximum[index] for index in range(3)):
            raise ValueError("metric map bounds must have positive volume")
        self.resolution_m = resolution_m
        self.minimum_bound_m = minimum_bound_m
        self.maximum_bound_m = maximum_bound_m
        self.unknown_is_occupied = unknown_is_occupied
        self._minimum = minimum
        self._maximum = maximum
        self._evidence: dict[VoxelKey, _VoxelEvidence] = {}
        self._evidence_owner = object()
        # Model snapshots are deliberately local.  Keep a coarse spatial index
        # so freezing the current navigation neighborhood is proportional to
        # nearby evidence instead of every voxel accumulated over a long
        # mission.  The safety-rate map itself remains complete.
        self._evidence_keys_by_chunk: dict[VoxelKey, set[VoxelKey]] = {}
        # Keep exact threshold indexes alongside the evidence dictionary.  The
        # live depth safety loop needs recent occupied endpoints at sensor rate;
        # rescanning every observed free-space voxel makes that loop slower as a
        # mission progresses even though the local obstacle set stays small.
        self._occupied_keys: set[VoxelKey] = set()
        self._observed_free_keys: set[VoxelKey] = set()
        self._known_static_occupied_keys: set[VoxelKey] = set()
        self._known_static_free_keys: set[VoxelKey] = set()
        # 被实测命中否定的旧空地不能在观测淘汰后自动复活。此集合仅增长，
        # 后来的新空地观测仍可暂时通行；观测再次过期后必须重新确认。
        self._invalidated_static_free_keys: set[VoxelKey] = set()
        self._static_invalidation_owner = self._evidence_owner
        self._non_static_evidence_count = 0
        self._known_static_source_sha256: str | None = None
        self._qualified_route_points: tuple[Vector3, ...] = ()
        self._qualified_route_sha256: str | None = None
        self._qualified_route_minimum_clearance_m: float | None = None
        self.observation_count = 0
        self.latest_observation_monotonic_seconds: float | None = None
        self.pruned_live_evidence_count = 0

    # 功能：
    #   检查三维有限坐标是否位于已绑定的闭区间地图边界。
    # 输入：
    #   point：坐标元组。
    # 输出：
    #   inside：边界包含关系。
    def _inside(self, point: Point) -> bool:
        inside = len(point) == 3 and all(
            type(point[index]) in (int, float)
            and abs(point[index]) <= 1e9
            and math.isfinite(point[index])
            and self._minimum[index] <= point[index] <= self._maximum[index]
            for index in range(3)
        )
        return inside

    # 功能：
    #   将界内坐标映射到当前分辨率的体素索引。
    # 输入：
    #   point：位置向量或三元组。
    # 输出：
    #   key：整型体素键。
    def key_for(self, point: Vector3 | Point) -> VoxelKey:
        raw = _point(point) if isinstance(point, Vector3) else point
        if not self._inside(raw):
            raise ValueError("point is outside metric map bounds")
        key = tuple(
            int(math.floor((raw[index] - self._minimum[index]) / self.resolution_m))
            for index in range(3)
        )
        return key

    # 功能：
    #   在内部几何热路径计算体素中心，避免为每次邻居查询分配模型对象。
    # 输入：
    #   key：内部整型体素索引。
    # 输出：
    #   center：地图坐标系的中心三元组。
    def _center_point_for_key(self, key: VoxelKey) -> Point:
        minimum, resolution = self._minimum, self.resolution_m
        center = (
            minimum[0] + (key[0] + 0.5) * resolution,
            minimum[1] + (key[1] + 0.5) * resolution,
            minimum[2] + (key[2] + 0.5) * resolution,
        )
        return center

    # 功能：
    #   为公开调用返回带合同类型的体素中心，允许计算边界邻居供保守碰撞检查。
    # 输入：
    #   key：三个整数组成的体素键。
    # 输出：
    #   center：三维中心位置。
    def center_for(self, key: VoxelKey) -> Vector3:
        if len(key) != 3 or any(type(value) is not int for value in key):
            raise ValueError("METRIC_VOXEL_KEY_INVALID")
        values = self._center_point_for_key(key)
        center = Vector3(x=values[0], y=values[1], z=values[2])
        return center

    # 功能：
    #   将已验证的占用置信度转为有界对数几率增量。
    # 输入：
    #   key：体素键；occupied：命中标志；confidence：置信度；observed_at：原始观测时间。
    # 输出：
    #   无：更新该体素证据。
    def _update(
        self,
        key: VoxelKey,
        *,
        occupied: bool,
        confidence: float,
        observed_at: float,
    ) -> None:
        bounded = max(0.5, min(0.99, confidence))
        measurement = math.log(bounded / (1.0 - bounded))
        self._update_log_odds(
            key,
            measurement=measurement if occupied else -abs(measurement),
            observed_at=observed_at,
        )

    # 功能：
    #   在单写线程累加一项规范化证据，同时维护占用索引和克隆隔离。
    # 输入：
    #   key：体素键；measurement：有符号对数几率；observed_at：原始采集时间。
    # 输出：
    #   无：更新地图内部证据和阈值索引。
    def _update_log_odds(
        self,
        key: VoxelKey,
        *,
        measurement: float,
        observed_at: float,
    ) -> None:
        if (
            measurement >= 0.0
            and key in self._known_static_free_keys
            and key not in self._invalidated_static_free_keys
        ):
            if self._static_invalidation_owner is not self._evidence_owner:
                self._invalidated_static_free_keys = self._invalidated_static_free_keys.copy()
                self._static_invalidation_owner = self._evidence_owner
            self._invalidated_static_free_keys.add(key)
        evidence = self._evidence.get(key)
        if evidence is None:
            evidence = _VoxelEvidence(owner=self._evidence_owner)
            self._evidence[key] = evidence
            if (
                key not in self._known_static_occupied_keys
                and key not in self._known_static_free_keys
            ):
                self._non_static_evidence_count += 1
            chunk = self._chunk_for_key(key)
            self._evidence_keys_by_chunk.setdefault(chunk, set()).add(key)
        elif evidence.owner is not self._evidence_owner:
            # A navigation snapshot shares this value. Detach only the first
            # time it is changed, not every voxel on every model request.
            evidence = _VoxelEvidence(
                evidence.log_odds,
                evidence.observations,
                evidence.latest_monotonic_seconds,
                self._evidence_owner,
            )
            self._evidence[key] = evidence
        previous = evidence.log_odds
        updated = previous + measurement
        # Preserve each addition and clamp in observation order. In particular,
        # summing a frame's measurements first changes saturated hit/free
        # transitions and is not an equivalent occupancy update.
        evidence.log_odds = -6.0 if updated < -6.0 else (6.0 if updated > 6.0 else updated)
        evidence.observations += 1
        if observed_at > evidence.latest_monotonic_seconds:
            evidence.latest_monotonic_seconds = observed_at
        # Logistic probability is monotonic in log odds. Comparing in log-odds
        # space preserves the occupancy thresholds without evaluating an
        # exponential for every traversed voxel (tens of thousands per frame).
        if evidence.log_odds >= _OCCUPIED_LOG_ODDS_THRESHOLD:
            if previous < _OCCUPIED_LOG_ODDS_THRESHOLD:
                self._occupied_keys.add(key)
                self._observed_free_keys.discard(key)
        elif evidence.log_odds <= _FREE_LOG_ODDS_THRESHOLD:
            if previous > _FREE_LOG_ODDS_THRESHOLD:
                self._observed_free_keys.add(key)
                self._occupied_keys.discard(key)
        elif previous >= _OCCUPIED_LOG_ODDS_THRESHOLD or previous <= _FREE_LOG_ODDS_THRESHOLD:
            self._occupied_keys.discard(key)
            self._observed_free_keys.discard(key)

    # 功能：
    #   1. 校验米制射线，按原采样公式与浮点运算顺序记录体素；遗漏单元仍保持未知。
    #   2. 分块向量化取整并保留有序去重，限制临时数组大小，不改变端点或采样密度。
    # 输入：
    #   observation：标定后的原始时间射线。
    # 输出：
    #   traversed：按射线顺序去重的体素键。
    def _ray_keys(self, observation: RangeRayObservation) -> list[VoxelKey]:
        if not isinstance(observation, RangeRayObservation) or type(observation.hit) is not bool:
            raise ValueError("METRIC_SCAN_SAMPLE_INVALID")
        origin = _point(observation.origin_m)
        endpoint = _point(observation.endpoint_m)
        if not self._inside(origin) or not self._inside(endpoint):
            raise ValueError("range ray must remain inside metric map bounds")
        if (
            type(observation.observed_at_monotonic_seconds) not in (int, float)
            or type(observation.confidence) not in (int, float)
            or not math.isfinite(observation.observed_at_monotonic_seconds)
            or observation.observed_at_monotonic_seconds < 0
            or not 0 <= observation.confidence <= 1
        ):
            raise ValueError("METRIC_SCAN_SAMPLE_INVALID")
        distance = math.dist(origin, endpoint)
        if not math.isfinite(distance) or distance / (self.resolution_m * 0.45) > 2_000_000:
            raise ValueError("METRIC_SCAN_TRAVERSAL_BUDGET_EXCEEDED")
        count = max(1, math.ceil(distance / (self.resolution_m * 0.45)))
        traversed: list[VoxelKey] = []
        delta = np.asarray(tuple(endpoint[axis] - origin[axis] for axis in range(3)))
        inverse_resolution = 1.0 / self.resolution_m
        for start in range(0, count + 1, 4096):
            ratio = np.arange(start, min(count + 1, start + 4096), dtype=np.float64) / count
            # 各步独立执行，不合并乘加、不改用 linspace，避免体素边界上出现舍入差异。
            coordinates = ratio[:, None] * delta
            coordinates += origin
            coordinates -= self._minimum
            coordinates *= inverse_resolution
            keys = np.floor(coordinates).astype(np.int64)
            # 地图坐标已限 ±1e9、分辨率 > .02，网格索引落在 int64 可精确表示范围。
            keep = np.empty(len(keys), dtype=np.bool_)
            keep[0] = not traversed or tuple(keys[0]) != traversed[-1]
            keep[1:] = np.any(keys[1:] != keys[:-1], axis=1)
            traversed.extend(map(tuple, keys[keep].tolist()))
        return traversed

    # 功能：
    #   顺序融合独立射线；同一幅图的相关像素必须使用 integrate_scan。
    # 输入：
    #   observation：独立测距观测。
    # 输出：
    #   无：更新经过的空地、命中终点及计数。
    def integrate_ray(self, observation: RangeRayObservation) -> None:
        traversed = self._ray_keys(observation)
        bounded = max(0.5, min(0.99, observation.confidence))
        occupied_measurement = math.log(bounded / (1.0 - bounded))
        free_measurement = -abs(occupied_measurement)
        free = traversed[:-1] if observation.hit else traversed
        for key in free:
            self._update_log_odds(
                key,
                measurement=free_measurement,
                observed_at=observation.observed_at_monotonic_seconds,
            )
        if observation.hit:
            self._update_log_odds(
                traversed[-1],
                measurement=occupied_measurement,
                observed_at=observation.observed_at_monotonic_seconds,
            )
        self.observation_count += 1
        self.latest_observation_monotonic_seconds = max(
            self.latest_observation_monotonic_seconds or 0.0,
            observation.observed_at_monotonic_seconds,
        )

    # 功能：
    #   有界消费相互独立的射线序列，逐项保留已成功融合的真实观测。
    # 输入：
    #   observations：独立射线序列，不是同帧像素。
    # 输出：
    #   无：顺序更新地图，单项失败时此前已完成项仍保留。
    def integrate_rays(self, observations: Iterable[RangeRayObservation]) -> None:
        for observation in _bounded_items(observations, 250_000, "independent rays"):
            self.integrate_ray(observation)

    # 功能：
    #   在完整扫描校验通过后一次提交该帧的去相关占用证据。
    # 输入：
    #   observations：同一原始时间的射线帧。
    # 输出：
    #   无：提交地图更新，验证失败不更新任何体素。
    def integrate_scan(self, observations: Iterable[RangeRayObservation]) -> None:
        self.commit_scan(self.prepare_scan(observations))

    # 功能：
    #   1. 先验证整帧时间、边界和遍历预算，不提前改变地图。
    #   2. 每体素每帧仅取最大证据，命中优先，重复像素不能制造置信度。
    # 输入：
    #   observations：同一传感器帧的米制射线。
    # 输出：
    #   prepared：绑定当前地图代次的不可变待提交更新。
    def prepare_scan(self, observations: Iterable[RangeRayObservation]) -> PreparedMetricScan:
        hits: dict[VoxelKey, float] = {}
        frees: dict[VoxelKey, float] = {}
        stamp, count, work = None, 0, 0
        for ray in observations:
            if stamp is None:
                stamp = ray.observed_at_monotonic_seconds
            if ray.observed_at_monotonic_seconds != stamp:
                raise ValueError("METRIC_SCAN_SOURCE_TIMES_DIFFER")
            if (
                self.latest_observation_monotonic_seconds is not None
                and stamp <= self.latest_observation_monotonic_seconds
            ):
                raise ValueError("METRIC_SCAN_SOURCE_NOT_NEW")
            distance = math.dist(_point(ray.origin_m), _point(ray.endpoint_m))
            work += distance / (self.resolution_m * 0.45) + 2
            if not math.isfinite(work) or work > 2_000_000:
                raise ValueError("METRIC_SCAN_TRAVERSAL_BUDGET_EXCEEDED")
            keys = self._ray_keys(ray)
            bounded = max(0.5, min(0.99, ray.confidence))
            strength = math.log(bounded / (1.0 - bounded))
            # Low-confidence hits are not negative occupancy measurements.
            # The upstream quality gate controls whether motion is permitted.
            for key in keys[:-1] if ray.hit else keys:
                previous = frees.get(key)
                # 相邻像素多次经过同一体素时保留最强证据，不重复写入相等值。
                if previous is None or strength > previous:
                    frees[key] = strength
            if ray.hit:
                endpoint = keys[-1]
                previous = hits.get(endpoint)
                if previous is None or strength > previous:
                    hits[endpoint] = strength
            count += 1
            if count > 250_000 or len(frees) + len(hits) > 2_000_000:
                raise ValueError("METRIC_SCAN_EVIDENCE_BUDGET_EXCEEDED")
        if not count:
            raise ValueError("METRIC_SCAN_EMPTY")
        prepared = PreparedMetricScan(
            self._evidence_owner,
            self.observation_count,
            self.latest_observation_monotonic_seconds,
            stamp,
            count,
            tuple(hits.items()),
            tuple((key, strength) for key, strength in frees.items() if key not in hits),
        )
        return prepared

    # 功能：
    #   在拥有地图的线程提交已通过外部时效门禁的更新，禁止跨地图和重复提交。
    # 输入：
    #   prepared：本地图 prepare_scan 产生的待提交记录。
    # 输出：
    #   无：更新证据与原始观测时间。
    def commit_scan(self, prepared: PreparedMetricScan) -> None:
        if (
            prepared.owner is not self._evidence_owner
            or prepared.previous_observation_count != self.observation_count
            or prepared.previous_source_time != self.latest_observation_monotonic_seconds
        ):
            raise ValueError("METRIC_SCAN_PREPARED_STATE_CHANGED")
        for key, strength in prepared.free_strengths:
            self._update_log_odds(key, measurement=-strength, observed_at=prepared.observed_at)
        for key, strength in prepared.occupied_strengths:
            previous = self._evidence.get(key)
            old = previous.log_odds if previous is not None else 0.0
            # Never leave a new measured hit classified as free due to history.
            self._update_log_odds(
                key, measurement=strength - min(0.0, old), observed_at=prepared.observed_at
            )
        self.observation_count += prepared.ray_count
        self.latest_observation_monotonic_seconds = prepared.observed_at

    # 功能：
    #   淘汰超龄或远离局部范围的实时证据，已被否定的旧空地不会因此恢复通行。
    # 输入：
    #   center_m、radius_m：保留范围；now_monotonic_seconds：当前时刻。
    #   maximum_age_seconds：最大年龄。
    # 输出：
    #   removed：本次移除的实时体素数量。
    def prune_live_evidence(
        self,
        *,
        center_m: Vector3,
        radius_m: float,
        now_monotonic_seconds: float,
        maximum_age_seconds: float,
    ) -> int:
        radius_m = _number(radius_m, "live evidence retention radius")
        maximum_age_seconds = _number(maximum_age_seconds, "live evidence maximum age")
        now_monotonic_seconds = _number(
            now_monotonic_seconds, "live evidence pruning time", maximum=1e15
        )
        if radius_m <= 0.0:
            raise ValueError("live evidence retention radius must be positive")
        if maximum_age_seconds <= 0.0:
            raise ValueError("live evidence maximum age must be positive")
        if not math.isfinite(now_monotonic_seconds):
            raise ValueError("live evidence pruning time must be finite")
        center = _point(center_m)
        cutoff = now_monotonic_seconds - maximum_age_seconds
        voxel_radius = math.sqrt(3.0) * self.resolution_m / 2.0
        retained_radius = radius_m + voxel_radius
        maximum_distance_squared = retained_radius**2
        center_key = self.key_for(center_m)
        voxel_extent = math.ceil(retained_radius / self.resolution_m)
        minimum_chunk = self._chunk_for_key(tuple(value - voxel_extent for value in center_key))
        maximum_chunk = self._chunk_for_key(tuple(value + voxel_extent for value in center_key))
        removed = 0
        for chunk, chunk_keys in list(self._evidence_keys_by_chunk.items()):
            inside_local_cube = all(
                minimum_chunk[axis] <= chunk[axis] <= maximum_chunk[axis] for axis in range(3)
            )
            if not inside_local_cube:
                removable = tuple(chunk_keys)
            else:
                local_removable: list[VoxelKey] = []
                for key in chunk_keys:
                    evidence = self._evidence[key]
                    voxel_center = self._center_point_for_key(key)
                    if (
                        evidence.latest_monotonic_seconds < cutoff
                        or sum((voxel_center[axis] - center[axis]) ** 2 for axis in range(3))
                        > maximum_distance_squared
                    ):
                        local_removable.append(key)
                removable = tuple(local_removable)
            for key in removable:
                self._evidence.pop(key, None)
                if (
                    key not in self._known_static_occupied_keys
                    and key not in self._known_static_free_keys
                ):
                    self._non_static_evidence_count -= 1
                self._occupied_keys.discard(key)
                self._observed_free_keys.discard(key)
                chunk_keys.discard(key)
            if not chunk_keys:
                self._evidence_keys_by_chunk.pop(chunk, None)
            removed += len(removable)
        self.pruned_live_evidence_count += removed
        return removed

    # 功能：
    #   导入显式矩形分类证据；先核对完整体积预算，不能作为真实传感器回执。
    # 输入：
    #   minimum_m、maximum_m：矩形两角；occupied、confidence：分类。
    #   observed_at_monotonic_seconds：来源时间。
    # 输出：
    #   无：逐体素更新矩形证据及计数。
    def mark_box(
        self,
        *,
        minimum_m: Vector3,
        maximum_m: Vector3,
        occupied: bool,
        confidence: float = 0.99,
        observed_at_monotonic_seconds: float = 0.0,
    ) -> None:
        minimum = self.key_for(minimum_m)
        maximum = self.key_for(maximum_m)
        if type(occupied) is not bool or any(minimum[i] > maximum[i] for i in range(3)):
            raise ValueError("METRIC_BOX_INVALID")
        confidence = _number(confidence, "box confidence", maximum=1.0)
        observed_at_monotonic_seconds = _number(
            observed_at_monotonic_seconds, "box timestamp", maximum=1e15
        )
        if math.prod(maximum[i] - minimum[i] + 1 for i in range(3)) > _MAX_GRID_WORK:
            raise ValueError("METRIC_BOX_WORK_BUDGET_EXCEEDED")
        for x in range(minimum[0], maximum[0] + 1):
            for y in range(minimum[1], maximum[1] + 1):
                for z in range(minimum[2], maximum[2] + 1):
                    self._update(
                        (x, y, z),
                        occupied=occupied,
                        confidence=confidence,
                        observed_at=observed_at_monotonic_seconds,
                    )
        self.observation_count += 1
        self.latest_observation_monotonic_seconds = max(
            self.latest_observation_monotonic_seconds or 0.0,
            observed_at_monotonic_seconds,
        )

    # 功能：
    #   导入同一摘要的合格静态先验，禁止给旧地图重新贴上另一份地图的身份。
    # 输入：
    #   classifications：位置和占用布尔序列；source_sha256：经过上游核验的当前地图摘要。
    # 输出：
    #   counts：此次分类中的空地数、占用数二元组。
    def seed_known_static_region(
        self,
        classifications: Iterable[tuple[Vector3, bool]],
        *,
        source_sha256: str,
    ) -> tuple[int, int]:
        if type(source_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
            raise ValueError("known static map source must be a lowercase SHA-256 digest")
        if self._known_static_source_sha256 not in (None, source_sha256):
            raise ValueError("METRIC_STATIC_SOURCE_CHANGED: construct a new map for new assets")
        free: set[VoxelKey] = set()
        occupied: set[VoxelKey] = set()
        for point, is_occupied in _bounded_items(
            classifications, _MAX_GRID_WORK, "static classifications"
        ):
            if type(is_occupied) is not bool:
                raise ValueError("METRIC_STATIC_CLASSIFICATION_INVALID")
            key = self.key_for(point)
            if is_occupied:
                occupied.add(key)
                free.discard(key)
            elif key not in occupied:
                free.add(key)
        existing_static = self._known_static_occupied_keys | self._known_static_free_keys
        newly_static = (occupied | free) - existing_static
        self._non_static_evidence_count -= sum(1 for key in newly_static if key in self._evidence)
        # Navigation clones share these priors. Replace sets rather than mutate
        # an already published model's frozen view.
        self._known_static_occupied_keys = self._known_static_occupied_keys | occupied
        self._known_static_free_keys = (
            self._known_static_free_keys | free
        ) - self._known_static_occupied_keys
        self._known_static_source_sha256 = source_sha256
        self._invalidated_static_free_keys = self._invalidated_static_free_keys | {
            key for key in free if key in self._evidence and key not in self._observed_free_keys
        }
        self._static_invalidation_owner = self._evidence_owner
        counts = len(free - occupied), len(occupied)
        return counts

    # 功能：
    #   冻结上游已完成净空核验的当前路线，不借用可变坐标对象。
    # 输入：
    #   points：完整有序路径；route_sha256：上游路线证据摘要；minimum_clearance_m：已验证净空。
    # 输出：
    #   无：更新本地图的路线绑定，不自行产生飞行批准。
    def bind_qualified_route(
        self,
        points: Iterable[Vector3],
        *,
        route_sha256: str,
        minimum_clearance_m: float | None = None,
    ) -> None:
        route_points = tuple(
            _frozen_vector(point) for point in _bounded_items(points, 100_000, "qualified route")
        )
        if len(route_points) < 2:
            raise ValueError("qualified route requires at least two points")
        if type(route_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", route_sha256) is None:
            raise ValueError("qualified route must have a lowercase SHA-256 digest")
        if minimum_clearance_m is not None:
            minimum_clearance_m = _number(minimum_clearance_m, "qualified route minimum clearance")
            if minimum_clearance_m <= 0.0:
                raise ValueError("qualified route minimum clearance must be positive")
        for point in route_points:
            self.key_for(point)
        self._qualified_route_points = route_points
        self._qualified_route_sha256 = route_sha256
        self._qualified_route_minimum_clearance_m = minimum_clearance_m

    # 功能：
    #   将体素键映射到粗粒度空间索引桶。
    # 输入：
    #   key：体素键；cls：包含桶边长的地图类。
    # 输出：
    #   chunk：桶的三维整数索引。
    @classmethod
    def _chunk_for_key(cls, key: VoxelKey) -> VoxelKey:
        edge = cls._NAVIGATION_CHUNK_EDGE_VOXELS
        chunk = key[0] // edge, key[1] // edge, key[2] // edge
        return chunk

    # 功能：
    #   查找局部实时证据；在稀疏大范围查询时遍历已有桶而非空空间。
    # 输入：
    #   center_m：查询中心；radius_m：米制半径。
    # 输出：
    #   selected：与局部球体相交的证据键集合。
    def _local_navigation_keys(
        self,
        *,
        center_m: Vector3,
        radius_m: float,
    ) -> set[VoxelKey]:
        radius_m = _number(radius_m, "navigation clone radius")
        if radius_m <= 0.0:
            raise ValueError("navigation clone radius must be positive")
        center = self.key_for(center_m)
        voxel_radius = math.ceil(radius_m / self.resolution_m)
        minimum = tuple(value - voxel_radius for value in center)
        maximum = tuple(value + voxel_radius for value in center)
        minimum_chunk = self._chunk_for_key(minimum)
        maximum_chunk = self._chunk_for_key(maximum)
        radius_with_voxel_m = radius_m + math.sqrt(3.0) * self.resolution_m / 2.0
        center_point = _point(center_m)
        selected: set[VoxelKey] = set()
        volume = math.prod(maximum_chunk[i] - minimum_chunk[i] + 1 for i in range(3))
        chunks = (
            (
                chunk
                for chunk in self._evidence_keys_by_chunk
                if all(minimum_chunk[i] <= chunk[i] <= maximum_chunk[i] for i in range(3))
            )
            if volume > len(self._evidence_keys_by_chunk)
            else product(*(range(minimum_chunk[i], maximum_chunk[i] + 1) for i in range(3)))
        )
        for chunk in chunks:
            for key in self._evidence_keys_by_chunk.get(chunk, ()):
                distance = math.dist(center_point, self._center_point_for_key(key))
                if distance <= radius_with_voxel_m:
                    selected.add(key)
        return selected

    # 功能：
    #   冻结模型线程使用的地图视图，实时证据可局部裁剪，静态先验及撤销状态保持完整。
    # 输入：
    #   center_m、radius_m：同时提供则只复制局部实时索引，否则复制全部实时索引。
    # 输出：
    #   clone：与后续写入隔离的单读者/单写者地图副本。
    def navigation_clone(
        self,
        *,
        center_m: Vector3 | None = None,
        radius_m: float | None = None,
    ) -> MetricVoxelMap:
        if (center_m is None) != (radius_m is None):
            raise ValueError("navigation clone center and radius must be supplied together")
        keys = (
            set(self._evidence)
            if center_m is None or radius_m is None
            else self._local_navigation_keys(center_m=center_m, radius_m=radius_m)
        )

        clone = MetricVoxelMap(
            resolution_m=self.resolution_m,
            minimum_bound_m=self.minimum_bound_m,
            maximum_bound_m=self.maximum_bound_m,
            unknown_is_occupied=self.unknown_is_occupied,
        )
        clone._evidence = {
            key: evidence for key in keys if (evidence := self._evidence.get(key)) is not None
        }
        # Both dictionaries now reference read-only shared values. Neither
        # map owns their tokens, so either side's next update must detach.
        # A map is still single-writer; the frozen clone can be read elsewhere.
        self._evidence_owner = object()
        for key in clone._evidence:
            chunk = clone._chunk_for_key(key)
            clone._evidence_keys_by_chunk.setdefault(chunk, set()).add(key)
        clone._occupied_keys = self._occupied_keys & keys
        clone._observed_free_keys = self._observed_free_keys & keys
        # Static priors and route bindings are immutable after worker startup,
        # so sharing them avoids copying hundreds of thousands of voxels every
        # model cycle while the mutable depth evidence remains isolated.
        clone._known_static_occupied_keys = self._known_static_occupied_keys
        clone._known_static_free_keys = self._known_static_free_keys
        clone._invalidated_static_free_keys = self._invalidated_static_free_keys
        clone._static_invalidation_owner = self._static_invalidation_owner
        clone._non_static_evidence_count = sum(
            1
            for key in clone._evidence
            if key not in clone._known_static_occupied_keys
            and key not in clone._known_static_free_keys
        )
        clone._known_static_source_sha256 = self._known_static_source_sha256
        clone._qualified_route_points = self._qualified_route_points
        clone._qualified_route_sha256 = self._qualified_route_sha256
        clone._qualified_route_minimum_clearance_m = self._qualified_route_minimum_clearance_m
        clone.observation_count = self.observation_count
        clone.pruned_live_evidence_count = self.pruned_live_evidence_count
        clone.latest_observation_monotonic_seconds = self.latest_observation_monotonic_seconds
        return clone

    # 功能：
    #   读取实时对数几率对应概率；没有证据时保留未知，不伪造零占用。
    # 输入：
    #   key：体素键。
    # 输出：
    #   probability：实时占用概率或 None。
    def occupancy_probability(self, key: VoxelKey) -> float | None:
        evidence = self._evidence.get(key)
        if evidence is None:
            return None
        probability = 1.0 / (1.0 + math.exp(-evidence.log_odds))
        return probability

    # 功能：
    #   综合静态障碍、实时占用及未知策略判断体素是否阻塞。
    # 输入：
    #   key：体素键。
    # 输出：
    #   occupied：按当前策略得到的阻塞标志。
    def is_occupied(self, key: VoxelKey) -> bool:
        if key in self._known_static_occupied_keys:
            return True
        probability = self.occupancy_probability(key)
        if probability is None:
            return self.unknown_is_occupied and (
                key not in self._known_static_free_keys or key in self._invalidated_static_free_keys
            )
        occupied = probability >= 0.65
        return occupied

    # 功能：
    #   判断是否存在实时空地证据或仍有效的静态空地先验。
    # 输入：
    #   key：体素键。
    # 输出：
    #   free：具有空地依据的标志。
    def is_observed_free(self, key: VoxelKey) -> bool:
        if key in self._known_static_occupied_keys:
            return False
        probability = self.occupancy_probability(key)
        if probability is not None:
            return probability <= 0.35
        free = key in self._known_static_free_keys and key not in self._invalidated_static_free_keys
        return free

    # 功能：
    #   快速检查体素中心及其净空邻域；连续线段另由 path_clearance_valid 复验。
    # 输入：
    #   key：中心键；required_clearance_m：净空；temporary_blocked_keys：本周期临时阻塞键。
    # 输出：
    #   clear：中心级净空检查结果。
    def _clearance_ok(
        self,
        key: VoxelKey,
        required_clearance_m: float,
        temporary_blocked_keys: frozenset[VoxelKey] = frozenset(),
    ) -> bool:
        if key in temporary_blocked_keys:
            return False
        if self.is_occupied(key):
            return False
        radius = math.ceil(required_clearance_m / self.resolution_m) + 1
        if (2 * radius + 1) ** 3 > _MAX_GRID_WORK:
            raise ValueError("METRIC_CLEARANCE_WORK_BUDGET_EXCEEDED")
        voxel_radius = math.sqrt(3.0) * self.resolution_m / 2.0
        for x in range(key[0] - radius, key[0] + radius + 1):
            for y in range(key[1] - radius, key[1] + radius + 1):
                for z in range(key[2] - radius, key[2] + radius + 1):
                    candidate = (x, y, z)
                    if not self.is_occupied(candidate) and (
                        not self.unknown_is_occupied or self.is_observed_free(candidate)
                    ):
                        continue
                    if (
                        self.resolution_m
                        * math.sqrt(
                            (candidate[0] - key[0]) ** 2
                            + (candidate[1] - key[1]) ** 2
                            + (candidate[2] - key[2]) ** 2
                        )
                        - voxel_radius
                        < required_clearance_m
                    ):
                        return False
        clear = True
        return clear

    # 功能：
    #   1. 在当前可通行体素中进行有界三维 A*，限制垂直动作代价。
    #   2. 拒绝斜向切角，并复验返回折线的实际端点与连续净空。
    # 输入：
    #   start_m、goal_m：实际端点；required_clearance_m：净空；maximum_expansions：搜索上限。
    #   temporary_blocked_keys：动态阻塞；vertical_cost_multiplier：垂直代价。
    #   allow_current_position_as_start：起点豁免。
    #   deadline_monotonic_seconds：软搜索截止时刻。
    # 输出：
    #   path：经连续净空复验的米制折线。
    def plan_path(
        self,
        *,
        start_m: Vector3,
        goal_m: Vector3,
        required_clearance_m: float,
        maximum_expansions: int = 200_000,
        temporary_blocked_keys: frozenset[VoxelKey] = frozenset(),
        vertical_cost_multiplier: float = 2.0,
        allow_current_position_as_start: bool = False,
        deadline_monotonic_seconds: float | None = None,
    ) -> list[Vector3]:
        required_clearance_m = _number(required_clearance_m, "required clearance")
        vertical_cost_multiplier = _number(
            vertical_cost_multiplier, "vertical cost multiplier", minimum=1.0, maximum=10.0
        )
        if type(maximum_expansions) is not int or not 1 <= maximum_expansions <= 200_000:
            raise ValueError("maximum expansions must be in [1, 200000]")
        if type(allow_current_position_as_start) is not bool:
            raise ValueError("start exemption must be boolean")
        if deadline_monotonic_seconds is not None:
            deadline_monotonic_seconds = _number(
                deadline_monotonic_seconds, "planning deadline", maximum=1e15
            )
        start_m, goal_m = map(_frozen_vector, (start_m, goal_m))
        start = self.key_for(start_m)
        goal = self.key_for(goal_m)
        if not allow_current_position_as_start and not self._clearance_ok(
            start, required_clearance_m, temporary_blocked_keys
        ):
            raise ValueError("start is not in observed collision-free space")
        if not self._clearance_ok(goal, required_clearance_m, temporary_blocked_keys):
            raise ValueError("goal is not in observed collision-free space")
        frontier: list[tuple[float, float, VoxelKey]] = [(0.0, 0.0, start)]
        came_from: dict[VoxelKey, VoxelKey] = {}
        costs = {start: 0.0}
        visited: set[VoxelKey] = set()
        offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        while frontier and len(visited) < maximum_expansions:
            if (
                deadline_monotonic_seconds is not None
                and len(visited) % 8 == 0
                and time.monotonic() >= deadline_monotonic_seconds
            ):
                raise ValueError("metric path planning deadline exceeded")
            _estimate, current_cost, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                keys = [goal]
                while keys[-1] != start:
                    keys.append(came_from[keys[-1]])
                keys.reverse()
                path = [
                    Vector3(**start_m.model_dump()),
                    *[self.center_for(key) for key in keys[1:-1]],
                    Vector3(**goal_m.model_dump()),
                ]
                if not self.path_clearance_valid(
                    path,
                    required_clearance_m=required_clearance_m,
                    temporary_blocked_keys=temporary_blocked_keys,
                    allow_first_position_as_start=allow_current_position_as_start,
                ):
                    raise ValueError("metric path endpoint or segment clearance failed")
                return path
            for offset in offsets:
                neighbor = tuple(current[index] + offset[index] for index in range(3))
                center = self._center_point_for_key(neighbor)
                if not self._inside(center) or not self._clearance_ok(
                    neighbor, required_clearance_m, temporary_blocked_keys
                ):
                    continue
                # 斜移必须连同各轴的中间邻格检查；仅测试终点会允许从两堵墙的交角穿过。
                axes = [axis for axis, delta in enumerate(offset) if delta]
                if len(axes) > 1 and any(
                    not self._clearance_ok(
                        tuple(current[i] + (offset[i] if i in selected else 0) for i in range(3)),
                        required_clearance_m,
                        temporary_blocked_keys,
                    )
                    for mask in range(1, (1 << len(axes)) - 1)
                    for selected in ({axes[bit] for bit in range(len(axes)) if mask & (1 << bit)},)
                ):
                    continue
                step = self.resolution_m * math.sqrt(
                    offset[0] ** 2 + offset[1] ** 2 + (offset[2] * vertical_cost_multiplier) ** 2
                )
                candidate_cost = current_cost + step
                if candidate_cost >= costs.get(neighbor, math.inf):
                    continue
                if (current == start or neighbor == goal) and not self.path_clearance_valid(
                    [
                        start_m if current == start else self.center_for(current),
                        goal_m if neighbor == goal else self.center_for(neighbor),
                    ],
                    required_clearance_m=required_clearance_m,
                    temporary_blocked_keys=temporary_blocked_keys,
                    allow_first_position_as_start=allow_current_position_as_start
                    and current == start,
                ):
                    # 真实端点通常不在格子中心；应在搜索时排除坏连接，继续找其它入口，
                    # 而不是找到一条中心路径后才发现最后一小段穿墙并放弃可行替代路。
                    continue
                costs[neighbor] = candidate_cost
                came_from[neighbor] = current
                heapq.heappush(
                    frontier,
                    (
                        candidate_cost
                        + math.sqrt(
                            (center[0] - goal_m.x) ** 2
                            + (center[1] - goal_m.y) ** 2
                            + ((center[2] - goal_m.z) * vertical_cost_multiplier) ** 2
                        ),
                        candidate_cost,
                        neighbor,
                    ),
                )
        raise ValueError("no observed collision-free path exists inside the search bound")

    # 功能：
    #   以有界采样查找相邻体素，再对完整线段做连续盒相交检查，避免漏掉短暂穿角。
    # 输入：
    #   path_m：米制折线；required_clearance_m：净空；temporary_blocked_keys：临时障碍。
    #   allow_first_position_as_start：首点豁免。
    # 输出：
    #   valid：整条折线是否满足保守连续碰撞检查。
    def path_clearance_valid(
        self,
        path_m: Iterable[Vector3],
        *,
        required_clearance_m: float,
        temporary_blocked_keys: frozenset[VoxelKey] = frozenset(),
        allow_first_position_as_start: bool = False,
    ) -> bool:
        required_clearance_m = _number(required_clearance_m, "required clearance")
        if type(allow_first_position_as_start) is not bool:
            raise ValueError("start exemption must be boolean")
        points = [_frozen_vector(point) for point in _bounded_items(path_m, 200_000, "metric path")]
        if not points:
            return False
        sample_spacing_m = self.resolution_m / 2.0
        radius = math.ceil(required_clearance_m / self.resolution_m) + 1
        neighborhood = (2 * radius + 1) ** 3
        work = 0
        segments = list(zip(points, points[1:], strict=False)) or [(points[0], points[0])]
        for segment_index, (start, end) in enumerate(segments):
            first = _point(start)
            second = _point(end)
            if not self._inside(first) or not self._inside(second):
                return False
            distance = math.dist(first, second)
            if not math.isfinite(distance):
                raise ValueError("METRIC_PATH_GEOMETRY_INVALID")
            count = max(1, math.ceil(distance / sample_spacing_m))
            work += (count + 1) * neighborhood
            if work > _MAX_GRID_WORK:
                raise ValueError("METRIC_PATH_WORK_BUDGET_EXCEEDED")
            checked: set[VoxelKey] = set()
            for index in range(count + 1):
                if allow_first_position_as_start and segment_index == 0 and index == 0:
                    continue
                ratio = index / count
                position = tuple(
                    first[axis] + (second[axis] - first[axis]) * ratio for axis in range(3)
                )
                key = self.key_for(position)
                # 半体素采样只用于枚举邻近盒；最终检查完整线段而不是采样点。
                # 净空按各轴扩张，盒角处略保守，绝不把拐角当成可穿过的缝隙。
                for candidate in product(
                    *(range(key[i] - radius, key[i] + radius + 1) for i in range(3))
                ):
                    if candidate in checked:
                        continue
                    checked.add(candidate)
                    if (
                        candidate not in temporary_blocked_keys
                        and not self.is_occupied(candidate)
                        and (not self.unknown_is_occupied or self.is_observed_free(candidate))
                    ):
                        continue
                    if _segment_hits_box(
                        first,
                        second,
                        self._center_point_for_key(candidate),
                        self.resolution_m / 2.0 + required_clearance_m,
                    ):
                        return False
        valid = True
        return valid

    # 功能：
    #   从确实已知为空的单元提取邻接未知区域的探索前沿，不枚举整片空立方体。
    # 输入：
    #   limit：返回数量；center_m、radius_m：可选局部筛选范围。
    # 输出：
    #   frontiers：有限数量的前沿中心。
    def frontier_centers(
        self,
        *,
        limit: int = 256,
        center_m: Vector3 | None = None,
        radius_m: float | None = None,
    ) -> list[Vector3]:
        if (center_m is None) != (radius_m is None):
            raise ValueError("frontier center and radius must be supplied together")
        if type(limit) is not int or not 0 <= limit <= 4096:
            raise ValueError("frontier limit must be in [0, 4096]")
        if not limit:
            return []
        if radius_m is not None:
            radius_m = _number(radius_m, "frontier radius", minimum=1e-6)
            self.key_for(center_m)
        keys = sorted(
            self._observed_free_keys
            | (
                self._known_static_free_keys
                - self._invalidated_static_free_keys
                - self._evidence.keys()
            )
        )
        frontiers: list[Vector3] = []
        for key in keys:
            if (
                center_m is not None
                and math.dist(self._center_point_for_key(key), _point(center_m)) > radius_m
            ):
                continue
            if not self.is_observed_free(key):
                continue
            if any(
                (key[0] + dx, key[1] + dy, key[2] + dz) not in self._evidence
                and not self.is_observed_free((key[0] + dx, key[1] + dy, key[2] + dz))
                and (key[0] + dx, key[1] + dy, key[2] + dz) not in self._known_static_occupied_keys
                for dx, dy, dz in (
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                )
            ):
                frontiers.append(self.center_for(key))
                if len(frontiers) >= limit:
                    break
        return frontiers

    # 功能：
    #   导出独立地图合同，空地列表排除实时未知及已撤销静态权限。
    # 输入：
    #   无：读取当前地图。
    # 输出：
    #   snapshot：包含真实计数、边界和分类键的快照。
    def snapshot(self) -> MetricVoxelMapSnapshot:
        snapshot = MetricVoxelMapSnapshot(
            resolution_m=self.resolution_m,
            minimum_bound_m=self.minimum_bound_m,
            maximum_bound_m=self.maximum_bound_m,
            occupied_voxels=sorted(self._occupied_keys | self._known_static_occupied_keys),
            free_voxels=sorted(
                (
                    self._observed_free_keys
                    | (
                        self._known_static_free_keys
                        - self._evidence.keys()
                        - self._invalidated_static_free_keys
                    )
                )
                - self._known_static_occupied_keys
                - self._occupied_keys
            ),
            observation_count=self.observation_count,
            latest_observation_monotonic_seconds=self.latest_observation_monotonic_seconds,
        )
        return snapshot

    # 功能：
    #   将新鲜局部实测占用转成控制器碰撞盒；超量保守包围，不静默删除障碍。
    # 输入：
    #   center_m、radius_m：局部范围；now_monotonic_seconds、maximum_age_seconds：时效。
    #   limit：盒数量上限。
    # 输出：
    #   result：合并后的碰撞盒，粗化盒带独立名称标志。
    def local_occupied_box_primitives(
        self,
        *,
        center_m: Vector3,
        radius_m: float,
        now_monotonic_seconds: float,
        maximum_age_seconds: float = 0.75,
        limit: int = 96,
    ) -> list[dict[str, float | str]]:
        radius_m = _number(radius_m, "local occupancy radius", minimum=1e-6)
        maximum_age_seconds = _number(
            maximum_age_seconds, "local occupancy maximum age", minimum=1e-6
        )
        now_monotonic_seconds = _number(now_monotonic_seconds, "local occupancy time", maximum=1e15)
        center_m = _frozen_vector(center_m)
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("local occupancy radius must be positive")
        if not math.isfinite(maximum_age_seconds) or maximum_age_seconds <= 0.0:
            raise ValueError("local occupancy maximum age must be positive")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("local occupancy primitive limit must be a positive integer")
        if not math.isfinite(now_monotonic_seconds) or now_monotonic_seconds < 0.0:
            raise ValueError("local occupancy time must be finite and nonnegative")
        center = _point(center_m)
        candidates: list[VoxelKey] = []
        for key in self._occupied_keys:
            evidence = self._evidence[key]
            age = now_monotonic_seconds - evidence.latest_monotonic_seconds
            if age < -0.05 or age > maximum_age_seconds:
                continue
            voxel_center = _point(self.center_for(key))
            # Include cells touching the query sphere even if their center is
            # outside it. Omitting the near face would erase measured volume.
            distance_squared = sum(
                max(abs(a - b) - self.resolution_m / 2.0, 0.0) ** 2
                for a, b in zip(center, voxel_center, strict=True)
            )
            if distance_squared <= radius_m**2:
                candidates.append(key)
        result: list[dict[str, float | str]] = []
        origin = _point(self.minimum_bound_m)
        center_in_cells = tuple(
            (center[axis] - origin[axis]) / self.resolution_m for axis in range(3)
        )
        for box, coarsened in bounded_occupied_cell_boxes(
            candidates,
            center_in_cells=center_in_cells,
            limit=limit,
        ):
            box_center = tuple(
                origin[axis] + (box[axis] + box[axis + 3]) * self.resolution_m / 2.0
                for axis in range(3)
            )
            size = tuple((box[axis + 3] - box[axis]) * self.resolution_m for axis in range(3))
            prefix = "perception-coarsened-overflow" if coarsened else "perception-voxel"
            result.append(
                {
                    "name": f"{prefix}-{'-'.join(str(value) for value in box)}",
                    "shape": "box",
                    "center_x": box_center[0],
                    "center_y": box_center[1],
                    "center_z": box_center[2],
                    "size_x": size[0],
                    "size_y": size[1],
                    "size_z": size[2],
                    "yaw_rad": 0.0,
                }
            )
        return result

    # 功能：
    #   为传感器频率遥测提供常数项元数据，不执行全图序列化或寻路。
    # 输入：
    #   无：读取已维护计数与绑定信息。
    # 输出：
    #   summary：证据统计、来源摘要和权限边界。
    def runtime_evidence_summary(self) -> dict[str, object]:
        summary = {
            "source_of_truth": "metric-range-observations-not-rendered-image",
            "resolution_m": self.resolution_m,
            "bounds_m": {
                "minimum": self.minimum_bound_m.model_dump(mode="json"),
                "maximum": self.maximum_bound_m.model_dump(mode="json"),
            },
            "observed_free_voxel_count": len(self._observed_free_keys),
            "occupied_voxel_count": len(self._occupied_keys),
            "known_static_free_voxel_count": len(self._known_static_free_keys),
            "known_static_occupied_voxel_count": len(self._known_static_occupied_keys),
            "known_static_source_sha256": self._known_static_source_sha256,
            "invalidated_static_free_voxel_count": len(self._invalidated_static_free_keys),
            "qualified_route_sha256": self._qualified_route_sha256,
            "evidence_voxel_count": (
                len(self._known_static_free_keys)
                + len(self._known_static_occupied_keys)
                + self._non_static_evidence_count
            ),
            "range_ray_observation_count": self.observation_count,
            "pruned_live_evidence_count": self.pruned_live_evidence_count,
            "latest_observation_monotonic_seconds": (self.latest_observation_monotonic_seconds),
            "unknown_space_policy": "blocked-until-observed"
            if self.unknown_is_occupied
            else "unobserved-space-permitted",
            "known_static_map": {
                "source_sha256": self._known_static_source_sha256,
                "qualified_route_sha256": self._qualified_route_sha256,
                "qualified_route_minimum_clearance_m": (self._qualified_route_minimum_clearance_m),
                "free_voxel_count": len(self._known_static_free_keys),
                "occupied_voxel_count": len(self._known_static_occupied_keys),
            },
            "model_authority": (
                "select semantic goal/frontier only; metric path and actuator commands "
                "require deterministic safety approval"
            ),
        }
        return summary

    # 功能：
    #   汇总模型规划使用的地图分类及探索前沿，与高频遥测路径分开。
    # 输入：
    #   无：读取地图状态。
    # 输出：
    #   summary：有数量上限的文本地图摘要。
    def text_map_summary(self) -> dict[str, object]:
        snapshot = self.snapshot()
        summary = {
            "source_of_truth": "metric-range-observations-not-rendered-image",
            "resolution_m": snapshot.resolution_m,
            "bounds_m": {
                "minimum": snapshot.minimum_bound_m.model_dump(mode="json"),
                "maximum": snapshot.maximum_bound_m.model_dump(mode="json"),
            },
            "observed_free_voxel_count": len(snapshot.free_voxels),
            "occupied_voxel_count": len(snapshot.occupied_voxels),
            "unknown_space_policy": "blocked-until-observed"
            if self.unknown_is_occupied
            else "unobserved-space-permitted",
            "frontier_centers_m": [
                point.model_dump(mode="json") for point in self.frontier_centers(limit=32)
            ],
            "model_authority": (
                "select semantic goal/frontier only; metric path and actuator commands "
                "require deterministic safety approval"
            ),
        }
        return summary

    # 功能：
    #   将动态目标的预测扫掠包络栅格化，包含采样间位移余量并限制总计算量。
    # 输入：
    #   dynamic_obstacles：动态目标；required_clearance_m：净空。
    #   vehicle_radius_m、vehicle_height_m：机体包络。
    #   prediction_horizon_seconds：预测时长。
    # 输出：
    #   blocked：本次规划的临时禁入体素集合。
    def dynamic_obstacle_blocked_keys(
        self,
        *,
        dynamic_obstacles: Iterable[DynamicObstacleObservation],
        required_clearance_m: float,
        vehicle_radius_m: float,
        vehicle_height_m: float,
        prediction_horizon_seconds: float = 5.0,
    ) -> frozenset[VoxelKey]:
        prediction_horizon_seconds = _number(
            prediction_horizon_seconds, "dynamic prediction horizon", minimum=1e-6, maximum=30.0
        )
        required_clearance_m = _number(required_clearance_m, "dynamic required clearance")
        vehicle_radius_m = _number(vehicle_radius_m, "vehicle radius", minimum=1e-6, maximum=100.0)
        vehicle_height_m = _number(vehicle_height_m, "vehicle height", minimum=1e-6, maximum=100.0)
        obstacles = tuple(
            DynamicObstacleObservation.model_validate(
                obstacle.model_dump(mode="python"), strict=True
            )
            for obstacle in _bounded_items(dynamic_obstacles, 512, "dynamic obstacles")
        )
        blocked: set[VoxelKey] = set()
        work = 0
        voxel_radius = math.sqrt(3.0) * self.resolution_m / 2.0
        for obstacle in obstacles:
            speed = _magnitude(_point(obstacle.velocity_mps))
            sample_step_seconds = max(
                0.1,
                min(0.5, self.resolution_m / max(speed, self.resolution_m)),
            )
            sample_count = math.ceil(prediction_horizon_seconds / sample_step_seconds)
            for sample_index in range(sample_count + 1):
                seconds = obstacle.age_seconds + min(
                    prediction_horizon_seconds, sample_index * sample_step_seconds
                )
                predicted = tuple(
                    _point(obstacle.position_m)[axis]
                    + _point(obstacle.velocity_mps)[axis] * seconds
                    for axis in range(3)
                )
                uncertainty = (1.0 - obstacle.confidence) * 0.5 + seconds * 0.08
                uncertainty += (speed + 0.08) * sample_step_seconds / 2.0
                if not all(math.isfinite(value) for value in (*predicted, uncertainty)):
                    raise ValueError("DYNAMIC_OBSTACLE_PREDICTION_NONFINITE")
                horizontal_radius = (
                    vehicle_radius_m
                    + obstacle.radius_m
                    + required_clearance_m
                    + voxel_radius
                    + uncertainty
                )
                vertical_radius = (
                    (vehicle_height_m + obstacle.height_m) / 2.0
                    + required_clearance_m
                    + voxel_radius
                    + uncertainty
                )
                minimum_key = tuple(
                    max(
                        0,
                        int(
                            math.floor(
                                (
                                    predicted[axis]
                                    - (horizontal_radius if axis < 2 else vertical_radius)
                                    - self._minimum[axis]
                                )
                                / self.resolution_m
                            )
                        ),
                    )
                    for axis in range(3)
                )
                maximum_key = tuple(
                    int(
                        math.floor(
                            (
                                min(
                                    self._maximum[axis],
                                    predicted[axis]
                                    + (horizontal_radius if axis < 2 else vertical_radius),
                                )
                                - self._minimum[axis]
                            )
                            / self.resolution_m
                        )
                    )
                    for axis in range(3)
                )
                work += math.prod(max(0, maximum_key[i] - minimum_key[i] + 1) for i in range(3))
                if work > _MAX_GRID_WORK:
                    raise ValueError("DYNAMIC_OBSTACLE_WORK_BUDGET_EXCEEDED")
                for x in range(minimum_key[0], maximum_key[0] + 1):
                    for y in range(minimum_key[1], maximum_key[1] + 1):
                        for z in range(minimum_key[2], maximum_key[2] + 1):
                            key = (x, y, z)
                            center = _point(self.center_for(key))
                            if (
                                math.hypot(center[0] - predicted[0], center[1] - predicted[1])
                                <= horizontal_radius
                                and abs(center[2] - predicted[2]) <= vertical_radius
                            ):
                                blocked.add(key)
        result = frozenset(blocked)
        return result

    # 功能：
    #   1. 将米制感知和目标压缩为带来源摘要的文本状态，不从渲染图片捏造距离。
    #   2. 候选模式先核验几何与动态净空；连续控制模式只提供环境，不生成旧式路径候选。
    # 输入：
    #   current_position_m、current_velocity_mps、goal_position_m：当前状态与语义目标位置。
    #   dynamic_obstacles：目标观测；required_clearance_m、local_radius_m：净空与局部范围。
    #   maximum_frontier_candidates、candidate_speed_mps：候选数量与速度。
    #   vehicle_radius_m、vehicle_height_m：机体尺寸。
    #   maximum_planning_seconds：规划软期限；include_candidate_paths：是否启用候选规划。
    # 输出：
    #   payload：绑定摘要的环境、动态预测和可选候选集合。
    def text_navigation_snapshot(
        self,
        *,
        current_position_m: Vector3,
        current_velocity_mps: Vector3,
        goal_position_m: Vector3,
        dynamic_obstacles: Iterable[DynamicObstacleObservation] = (),
        required_clearance_m: float,
        local_radius_m: float = 8.0,
        maximum_frontier_candidates: int = 8,
        candidate_speed_mps: float = 0.5,
        vehicle_radius_m: float = 0.38,
        vehicle_height_m: float = 0.43,
        maximum_planning_seconds: float = 2.0,
        include_candidate_paths: bool = True,
    ) -> dict[str, object]:
        if type(include_candidate_paths) is not bool:
            raise ValueError("include_candidate_paths must be boolean")
        if (
            type(maximum_frontier_candidates) is not int
            or not 0 <= maximum_frontier_candidates <= 32
        ):
            raise ValueError("maximum frontier candidates must be in [0, 32]")
        required_clearance_m = _number(required_clearance_m, "required clearance")
        local_radius_m = _number(local_radius_m, "local reasoning radius")
        candidate_speed_mps = _number(
            candidate_speed_mps, "candidate speed", minimum=1e-6, maximum=1000.0
        )
        vehicle_radius_m = _number(vehicle_radius_m, "vehicle radius", minimum=1e-6, maximum=100.0)
        vehicle_height_m = _number(vehicle_height_m, "vehicle height", minimum=1e-6, maximum=100.0)
        maximum_planning_seconds = _number(
            maximum_planning_seconds, "maximum planning time", minimum=1e-6, maximum=60.0
        )
        current_position_m, current_velocity_mps, goal_position_m = map(
            _frozen_vector, (current_position_m, current_velocity_mps, goal_position_m)
        )
        self.key_for(current_position_m)
        self.key_for(goal_position_m)
        if required_clearance_m < 0.0:
            raise ValueError("required clearance must be non-negative")
        if local_radius_m <= self.resolution_m:
            raise ValueError("local reasoning radius must exceed map resolution")
        if maximum_frontier_candidates < 0:
            raise ValueError("maximum frontier candidates must be non-negative")
        if candidate_speed_mps <= 0.0:
            raise ValueError("candidate speed must be positive")
        if vehicle_radius_m <= 0.0 or vehicle_height_m <= 0.0:
            raise ValueError("vehicle collision envelope must be positive")
        if maximum_planning_seconds <= 0.0:
            raise ValueError("maximum planning time must be positive")
        planning_deadline = time.monotonic() + maximum_planning_seconds
        tracked_obstacles = tuple(
            DynamicObstacleObservation.model_validate(
                obstacle.model_dump(mode="python"), strict=True
            )
            for obstacle in _bounded_items(dynamic_obstacles, 512, "dynamic obstacles")
        )
        current = _point(current_position_m)
        goal = _point(goal_position_m)
        current_goal_distance = math.dist(current, goal)
        goal_heading = math.atan2(goal[1] - current[1], goal[0] - current[0])
        voxel_radius = math.sqrt(3.0) * self.resolution_m / 2.0
        sectors: dict[str, dict[str, float | int | None | str]] = {
            label: {
                "observed_free_voxels": 0,
                "occupied_voxels": 0,
                "nearest_occupied_clearance_m": None,
                "status": "unknown-blocked",
            }
            for label in SECTOR_LABELS
        }
        for key in self._evidence:
            center = self._center_point_for_key(key)
            distance = math.dist(current, center)
            if distance > local_radius_m:
                continue
            relative = (
                math.atan2(center[1] - current[1], center[0] - current[0]) - goal_heading + math.pi
            ) % (2 * math.pi) - math.pi
            sector_index = int(round(relative / (math.pi / 4))) % 8
            sector = sectors[SECTOR_LABELS[sector_index]]
            if self.is_observed_free(key):
                sector["observed_free_voxels"] = int(sector["observed_free_voxels"]) + 1
            elif self.is_occupied(key):
                sector["occupied_voxels"] = int(sector["occupied_voxels"]) + 1
                clearance = max(0.0, distance - voxel_radius)
                prior = sector["nearest_occupied_clearance_m"]
                if prior is None or clearance < float(prior):
                    sector["nearest_occupied_clearance_m"] = round(clearance, 3)
        for sector in sectors.values():
            free_count = int(sector["observed_free_voxels"])
            occupied_count = int(sector["occupied_voxels"])
            nearest = sector["nearest_occupied_clearance_m"]
            if nearest is not None and float(nearest) < required_clearance_m:
                sector["status"] = "blocked"
            elif free_count > 0 and free_count >= occupied_count:
                sector["status"] = "observed-partial"

        candidate_paths: list[dict[str, object]] = []
        candidate_generation_issues: list[str] = []

        temporary_dynamic_blocks = (
            self.dynamic_obstacle_blocked_keys(
                dynamic_obstacles=tracked_obstacles,
                required_clearance_m=required_clearance_m,
                vehicle_radius_m=vehicle_radius_m,
                vehicle_height_m=vehicle_height_m,
            )
            if include_candidate_paths
            else frozenset()
        )

        # 功能：
        #   对完整几何候选补动态预测门禁，保存完整路径摘要和有限展示点。
        # 输入：
        #   kind：候选来源；endpoint：真实终点；path：完整已验证路径。
        # 输出：
        #   无：追加通过的候选或记录拒绝原因。
        def add_candidate(kind: str, endpoint: Vector3, path: list[Vector3]) -> None:
            dynamic_clearance, dynamic_threat, dynamic_time = _dynamic_path_clearance(
                path,
                dynamic_obstacles=tracked_obstacles,
                path_speed_mps=candidate_speed_mps,
                vehicle_radius_m=vehicle_radius_m,
                vehicle_height_m=vehicle_height_m,
                sample_spacing_m=max(0.05, self.resolution_m / 2.0),
            )
            if dynamic_clearance < required_clearance_m:
                candidate_generation_issues.append(
                    f"DYNAMIC_PATH_BLOCKED:{dynamic_threat or 'unknown'}"
                )
                return
            path_payload = _bounded_path_points(path)
            identity = {
                "kind": kind,
                "endpoint_m": endpoint.model_dump(mode="json"),
                "path_points_m": path_payload,
            }
            candidate_paths.append(
                {
                    "candidate_id": f"candidate-{sha256_json(identity)[:16]}",
                    **identity,
                    "metric_path_sha256": sha256_json(
                        [point.model_dump(mode="json") for point in path]
                    ),
                    "path_length_m": round(_path_length(path), 3),
                    "required_clearance_m": required_clearance_m,
                    "deterministic_metric_path_validated": True,
                    "dynamic_path_validated": True,
                    "minimum_predicted_dynamic_clearance_m": round(dynamic_clearance, 3),
                    "time_to_minimum_dynamic_clearance_seconds": round(dynamic_time, 3),
                }
            )

        # Generate a short, hash-bound rejoin to the already qualified route
        # before attempting a more expensive search to the semantic goal.
        # A goal can be geometrically inside the local radius while an unknown
        # voxel or wall makes its A* search consume the entire cycle deadline.
        # If that search runs first, the simpler route rejoin inherits an
        # already-expired deadline and the model receives no legal action even
        # in observed free space.
        if current_goal_distance > local_radius_m:
            candidate_generation_issues.append("GOAL_OUTSIDE_LOCAL_PLANNING_HORIZON")

        matching_segments = (
            [
                (start, end)
                for start, end in zip(
                    self._qualified_route_points,
                    self._qualified_route_points[1:],
                    strict=False,
                )
                if math.dist(_point(end), goal) <= self.resolution_m
            ]
            if include_candidate_paths
            else []
        )
        if matching_segments:
            segment_start, segment_end = min(
                matching_segments,
                # A round trip can contain the same semantic waypoint more
                # than once. Choosing by distance to the segment *start*
                # selects a later return segment whose clamped projection is
                # already complete, leaving the model with no candidate. The
                # aircraft's distance to the whole segment identifies the
                # active outbound/return leg without adding mutable route
                # indices to the model's authority boundary.
                key=lambda segment: (
                    _point_segment_distance(
                        current,
                        _point(segment[0]),
                        _point(segment[1]),
                    ),
                    math.dist(_point(segment[0]), current),
                ),
            )
            start_tuple = _point(segment_start)
            end_tuple = _point(segment_end)
            segment_delta = tuple(end_tuple[axis] - start_tuple[axis] for axis in range(3))
            segment_length_squared = sum(value * value for value in segment_delta)
            if segment_length_squared > 1e-9:
                progress = max(
                    0.0,
                    min(
                        1.0,
                        sum(
                            (current[axis] - start_tuple[axis]) * segment_delta[axis]
                            for axis in range(3)
                        )
                        / segment_length_squared,
                    ),
                )
                segment_length = math.sqrt(segment_length_squared)
                remaining = segment_length * (1.0 - progress)
                lookahead_distance = min(local_radius_m * 0.5, remaining)
                if lookahead_distance >= self.resolution_m * 0.8:
                    ratio = min(1.0, progress + lookahead_distance / segment_length)
                    route_point = tuple(
                        start_tuple[axis] + segment_delta[axis] * ratio for axis in range(3)
                    )
                    key = self.key_for(route_point)
                    endpoint = self.center_for(key)
                    path = [current_position_m, endpoint]
                    straight_rejoin_valid = self.path_clearance_valid(
                        path,
                        required_clearance_m=required_clearance_m,
                        temporary_blocked_keys=temporary_dynamic_blocks,
                        allow_first_position_as_start=True,
                    )
                    if not straight_rejoin_valid:
                        # Model-directed local motion may intentionally leave
                        # the exact route centerline to preserve clearance. A
                        # later straight chord back to the long qualified
                        # segment can cross a wall/unknown voxel even though a
                        # short observed-free rejoin exists. Replan only to the
                        # bounded local lookahead; never plan the far semantic
                        # goal or let the model author replacement geometry.
                        try:
                            path = self.plan_path(
                                start_m=current_position_m,
                                goal_m=endpoint,
                                required_clearance_m=required_clearance_m,
                                temporary_blocked_keys=temporary_dynamic_blocks,
                                allow_current_position_as_start=True,
                                deadline_monotonic_seconds=planning_deadline,
                            )
                        except ValueError:
                            candidate_generation_issues.append(
                                "QUALIFIED_ROUTE_LOCAL_REJOIN_UNAVAILABLE"
                            )
                            path = []
                    if path:
                        add_candidate(
                            "qualified-route-local-lookahead",
                            endpoint,
                            path,
                        )
            seen_route_keys: set[VoxelKey] = set()
            for ratio in (0.35, 0.65, 0.9):
                route_point = tuple(
                    start_tuple[axis] + (end_tuple[axis] - start_tuple[axis]) * ratio
                    for axis in range(3)
                )
                key = self.key_for(route_point)
                if key in seen_route_keys or not self._clearance_ok(
                    key, required_clearance_m, temporary_dynamic_blocks
                ):
                    continue
                seen_route_keys.add(key)
                endpoint = self.center_for(key)
                if (
                    math.dist(current, _point(endpoint)) < self.resolution_m * 0.8
                    or math.dist(current, _point(endpoint)) > local_radius_m
                    or math.dist(_point(endpoint), goal)
                    >= current_goal_distance - self.resolution_m * 0.4
                ):
                    continue
                path = [current_position_m, endpoint]
                if self.path_clearance_valid(
                    path,
                    required_clearance_m=required_clearance_m,
                    temporary_blocked_keys=temporary_dynamic_blocks,
                    allow_first_position_as_start=True,
                ):
                    add_candidate(
                        "qualified-route-lookahead",
                        endpoint,
                        path,
                    )
        # Route lookahead endpoints are voxel centers. They are useful while
        # making progress along a long segment, but they must not become a
        # substitute for the exact signed waypoint during final approach. If
        # the direct segment to the metric goal is currently observed and
        # collision-free, expose that exact endpoint without paying for a full
        # A* search. Inside two voxels it is the only useful completion
        # candidate; retaining a neighboring route-center candidate can make a
        # valid model decision hover forever just outside the waypoint gate.
        if include_candidate_paths and current_goal_distance <= local_radius_m:
            direct_goal_path = [current_position_m, goal_position_m]
            if self.path_clearance_valid(
                direct_goal_path,
                required_clearance_m=required_clearance_m,
                temporary_blocked_keys=temporary_dynamic_blocks,
                allow_first_position_as_start=True,
            ):
                previous_candidate_count = len(candidate_paths)
                add_candidate("goal-direct", goal_position_m, direct_goal_path)
                if (
                    len(candidate_paths) > previous_candidate_count
                    and current_goal_distance <= self.resolution_m * 2.0
                ):
                    candidate_paths[:] = [candidate_paths[-1]]
        # Only spend the remaining cycle budget on a full goal search when the
        # qualified-route pass did not already produce a safe progress action.
        # Close to the end of a segment, where no meaningful lookahead remains,
        # this still offers the exact goal (and a dynamic detour when needed).
        if (
            include_candidate_paths
            and current_goal_distance <= local_radius_m
            and not candidate_paths
        ):
            with suppress(ValueError):
                add_candidate(
                    "goal",
                    goal_position_m,
                    self.plan_path(
                        start_m=current_position_m,
                        goal_m=goal_position_m,
                        required_clearance_m=required_clearance_m,
                        allow_current_position_as_start=True,
                        deadline_monotonic_seconds=planning_deadline,
                    ),
                )
            if temporary_dynamic_blocks and not candidate_paths:
                with suppress(ValueError):
                    add_candidate(
                        "goal-dynamic-detour",
                        goal_position_m,
                        self.plan_path(
                            start_m=current_position_m,
                            goal_m=goal_position_m,
                            required_clearance_m=required_clearance_m,
                            temporary_blocked_keys=temporary_dynamic_blocks,
                            allow_current_position_as_start=True,
                            deadline_monotonic_seconds=planning_deadline,
                        ),
                    )
        frontiers = (
            []
            if candidate_paths or not include_candidate_paths
            else sorted(
                self.frontier_centers(
                    limit=256,
                    center_m=current_position_m,
                    radius_m=local_radius_m,
                ),
                key=lambda point: (
                    math.dist(_point(point), goal),
                    math.dist(_point(point), current),
                ),
            )
        )
        frontier_candidate_count = 0
        for frontier in frontiers:
            if frontier_candidate_count >= maximum_frontier_candidates:
                break
            try:
                path = self.plan_path(
                    start_m=current_position_m,
                    goal_m=frontier,
                    required_clearance_m=required_clearance_m,
                    temporary_blocked_keys=temporary_dynamic_blocks,
                    allow_current_position_as_start=True,
                    deadline_monotonic_seconds=planning_deadline,
                )
            except ValueError:
                continue
            add_candidate("observed-frontier", frontier, path)
            frontier_candidate_count += 1

        dynamic_summaries: list[dict[str, object]] = []
        own_velocity = _point(current_velocity_mps)
        for obstacle in tracked_obstacles:
            obstacle_position = _point(obstacle.position_m)
            relative_position = tuple(
                obstacle_position[index] - current[index] for index in range(3)
            )
            relative_velocity = tuple(
                _point(obstacle.velocity_mps)[index] - own_velocity[index] for index in range(3)
            )
            speed_squared = sum(value * value for value in relative_velocity)
            closest_time = (
                max(
                    0.0,
                    min(
                        5.0,
                        -sum(
                            relative_position[index] * relative_velocity[index]
                            for index in range(3)
                        )
                        / speed_squared,
                    ),
                )
                if speed_squared > 1e-9
                else 0.0
            )
            closest = tuple(
                relative_position[index] + relative_velocity[index] * closest_time
                for index in range(3)
            )
            dynamic_summaries.append(
                {
                    "obstacle_id": obstacle.obstacle_id,
                    "distance_m": round(math.dist(current, obstacle_position), 3),
                    "time_to_closest_approach_seconds": round(closest_time, 3),
                    "closest_center_separation_m": round(_magnitude(closest), 3),
                    "confidence": obstacle.confidence,
                    "observation_age_seconds": obstacle.age_seconds,
                }
            )

        payload: dict[str, object] = {
            "schema_version": "dronedream.text-navigation-snapshot.v1",
            "source_of_truth": "metric-range-and-localization-evidence-not-rendered-image",
            "visual_input_required": False,
            "current_position_m": current_position_m.model_dump(mode="json"),
            "current_velocity_mps": current_velocity_mps.model_dump(mode="json"),
            "goal_position_m": goal_position_m.model_dump(mode="json"),
            "goal_distance_m": round(math.dist(current, goal), 3),
            "vehicle_envelope_m": {
                "body_radius": vehicle_radius_m,
                "body_height": vehicle_height_m,
                "candidate_speed": candidate_speed_mps,
                "required_local_clearance": required_clearance_m,
            },
            "known_static_map": {
                "source_sha256": self._known_static_source_sha256,
                "qualified_route_sha256": self._qualified_route_sha256,
                "qualified_route_minimum_clearance_m": (self._qualified_route_minimum_clearance_m),
                "free_voxel_count": len(self._known_static_free_keys),
                "occupied_voxel_count": len(self._known_static_occupied_keys),
            },
            "unknown_space_policy": "blocked-until-observed"
            if self.unknown_is_occupied
            else "unobserved-space-permitted",
            "local_radius_m": local_radius_m,
            "egocentric_sector_reference": "front points toward current semantic goal",
            "egocentric_sectors": [{"sector": label, **sectors[label]} for label in SECTOR_LABELS],
            "dynamic_obstacles": sorted(
                dynamic_summaries,
                key=lambda item: (
                    float(item["time_to_closest_approach_seconds"]),
                    float(item["closest_center_separation_m"]),
                ),
            ),
            "authorized_candidate_paths": candidate_paths,
            "candidate_generation_issues": sorted(set(candidate_generation_issues)),
            "model_authority": {
                "may_select": (
                    "candidate_id or safe hold"
                    if include_candidate_paths
                    else "bounded body-frame control intent or safe hold"
                ),
                "may_not_author": [
                    "free-space cells",
                    "metric coordinates",
                    "motor or attitude commands",
                ],
            },
        }
        payload["snapshot_sha256"] = sha256_json(payload)
        return payload
