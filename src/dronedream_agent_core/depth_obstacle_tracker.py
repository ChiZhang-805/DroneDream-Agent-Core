"""Conservative world-frame motion tracks derived from metric depth endpoints."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .collision import _clearance, _finite, _validated_primitive, primitive_bounds
from .contracts import DynamicObstacleObservation, OnboardPerceptionFrame, Vector3

Point = tuple[float, float, float]
VoxelKey = tuple[int, int, int]
QueryKey = tuple[VoxelKey, VoxelKey]


# 功能：
#   提取已验证的世界 ENU 坐标，不改变坐标系或单位。
# 输入：
#   value：三轴米坐标对象。
# 输出：
#   point：依次为东、北、上坐标的不可变三元组。
def _point(value: Vector3) -> Point:
    point = value.x, value.y, value.z
    return point


class _StaticPrimitiveIndex:
    _BIN_M = 2.0
    _PROBE_M = 0.001
    _MAX_BINS_PER_PRIMITIVE = 4096
    _MAX_BIN_ENTRIES = 1_000_000
    _MAX_QUERY_BINS = 4096
    _MAX_CACHED_QUERIES = 128
    _MAX_CACHED_CANDIDATES = 512

    # 功能：
    #   1. 验证并只读保存静态几何，建立有总条目上限的空间索引；超预算基元保留为直接查询。
    #   2. 有界缓存格范围对应的候选编号，不缓存点的距离或是否落在障碍内。
    # 输入：
    #   primitives：最多十万个地图基元。
    #   exclusion_m：静态结构附近的非负排除余量，单位米。
    # 输出：
    #   None：初始化索引，不返回业务数据。
    def __init__(self, primitives: list[dict[str, Any]], *, exclusion_m: float) -> None:
        if (
            not isinstance(primitives, list)
            or len(primitives) > 100_000
            or not _finite(exclusion_m)
            or exclusion_m < 0
        ):
            raise ValueError("DEPTH_TRACK_STATIC_GEOMETRY_INVALID")
        # 物理字段已规范为独立浮点值；只读快照禁止后续调用修改尺寸而沿用旧索引。
        self._primitives = tuple(MappingProxyType(_validated_primitive(p)) for p in primitives)
        self._exclusion_m = float(exclusion_m)
        self._bins: dict[VoxelKey, list[int]] = defaultdict(list)
        self._large_primitives: list[int] = []
        self._candidate_cache: OrderedDict[QueryKey, tuple[int, ...]] = OrderedDict()
        entries = 0
        for index, primitive in enumerate(self.primitives):
            # 宽阶段索引也要包含最终净空查询的一毫米探针，否则格边界会漏选基元。
            low, high = primitive_bounds(dict(primitive), exclusion_m + self._PROBE_M)
            if any(not math.isfinite(value) for value in (*low, *high)) or any(
                low[i] > high[i] for i in range(3)
            ):
                raise ValueError("DEPTH_TRACK_STATIC_GEOMETRY_INVALID")
            first = self._key(low)
            last = self._key(high)
            needed = math.prod(last[i] - first[i] + 1 for i in range(3))
            if needed > self._MAX_BINS_PER_PRIMITIVE or entries + needed > self._MAX_BIN_ENTRIES:
                # 超预算仅改变索引策略，不删除建筑物或把未知区域判为空闲。
                self._large_primitives.append(index)
                continue
            entries += needed
            for x in range(first[0], last[0] + 1):
                for y in range(first[1], last[1] + 1):
                    for z in range(first[2], last[2] + 1):
                        self._bins[(x, y, z)].append(index)

    # 功能：
    #   暴露不可替换的只读几何序列，避免外部修改物理字段破坏初始化时的索引。
    # 输入：
    #   self：已建立空间索引的对象。
    # 输出：
    #   primitives：参与净空计算的只读几何快照元组。
    @property
    def primitives(self):
        primitives = self._primitives
        return primitives

    # 功能：
    #   保持净空查询与建索引时使用同一排除余量，禁止只改查询余量而不重建索引。
    # 输入：
    #   self：已建立空间索引的对象。
    # 输出：
    #   exclusion_m：初始化时验证并固定的非负排除距离。
    @property
    def exclusion_m(self) -> float:
        exclusion_m = self._exclusion_m
        return exclusion_m

    # 功能：
    #   将有限世界坐标映射到两米宽索引格，负坐标向下取整保持边界一致。
    # 输入：
    #   point：世界三轴米坐标。
    # 输出：
    #   key：三轴整数格索引。
    def _key(self, point: Point) -> VoxelKey:
        key = tuple(math.floor(value / self._BIN_M) for value in point)
        return key

    # 功能：
    #   1. 按完整膨胀格范围复用候选编号，始终纳入未建立格索引的大基元。
    #   2. 缓存同时限制查询数和单项编号数；大查询仍全量计算，不因缓存预算遗漏障碍。
    # 输入：
    #   self：当前不可变地图的索引。
    #   low、high：包含定位余量的三轴起止格编号。
    # 输出：
    #   candidates：需要逐点计算距离的基元编号序列。
    def _candidates(self, low: VoxelKey, high: VoxelKey) -> tuple[int, ...] | range:
        key = low, high
        cached = self._candidate_cache.get(key)
        if cached is not None:
            self._candidate_cache.move_to_end(key)
            candidates = cached
            return candidates
        if math.prod(high[i] - low[i] + 1 for i in range(3)) > self._MAX_QUERY_BINS:
            # range 不为极大查询额外分配十万个整数，仍保留完整地图。
            candidates = range(len(self.primitives))
            return candidates
        indices = {
            index
            for x in range(low[0], high[0] + 1)
            for y in range(low[1], high[1] + 1)
            for z in range(low[2], high[2] + 1)
            for index in self._bins.get((x, y, z), ())
        }
        indices.update(self._large_primitives)
        candidates = tuple(sorted(indices))
        if len(candidates) <= self._MAX_CACHED_CANDIDATES:
            self._candidate_cache[key] = candidates
            if len(self._candidate_cache) > self._MAX_CACHED_QUERIES:
                self._candidate_cache.popitem(last=False)
        return candidates

    # 功能：
    #   判断端点是否可由静态几何及定位余量解释；大范围查询退为有界全扫描而不漏检。
    # 输入：
    #   point：世界三轴米坐标。
    #   uncertainty_m：定位不确定性对应的非负额外膨胀距离。
    # 输出：
    #   contained：端点是否处于静态排除区。
    def contains_known_static(self, point: Point, *, uncertainty_m: float = 0.0) -> bool:
        if (
            not _finite(uncertainty_m)
            or uncertainty_m < 0
            or not isinstance(point, (tuple, list))
            or len(point) != 3
            or not all(_finite(value) for value in point)
            or not all(
                math.isfinite(value + sign * uncertainty_m) for value in point for sign in (-1, 1)
            )
            or not math.isfinite(self.exclusion_m + uncertainty_m)
        ):
            raise ValueError("DEPTH_TRACK_STATIC_QUERY_INVALID")
        # 定位不确定度可能跨格；不能只查端点中心所在的单格。
        low = self._key(tuple(value - uncertainty_m for value in point))
        high = self._key(tuple(value + uncertainty_m for value in point))
        contained = False
        for index in self._candidates(low, high):
            # 地图只在初始化验证，当前点和不确定度在本次入口验证；不跳过数值溢出检查。
            clearance = _clearance(
                point,
                self.primitives[index],
                radius_m=self._PROBE_M,
                half_height_m=self._PROBE_M,
            )
            if not math.isfinite(clearance):
                raise ValueError("collision clearance overflow")
            if clearance <= self.exclusion_m + uncertainty_m:
                contained = True
                break
        return contained


@dataclass(frozen=True)
class _Cluster:
    center: Point
    minimum: Point
    maximum: Point
    point_count: int
    confidence: float


@dataclass(frozen=True)
class _Track:
    track_id: int
    center: Point
    velocity: Point
    observed_at_seconds: float
    observed_at_unix_ms: int
    seen_count: int
    cluster: _Cluster
    motion_anchor: Point
    motion_anchor_variance_m2: float
    motion_confirmed: bool = False


@dataclass(frozen=True)
class PreparedDepthTracks:
    """Candidate history, committed only after the corresponding scan is accepted."""

    owner: object
    revision: int
    tracks: tuple[_Track, ...]
    next_track_id: int
    sensor_id: str
    sequence: int
    observed_at_seconds: float
    observed_at_unix_ms: int
    observations: tuple[DynamicObstacleObservation, ...]


class DepthMotionTracker:
    """Track compact non-map depth clusters without granting actuator authority.

    World-frame projection removes camera ego-motion.  Endpoints already
    explained by the qualified static map are discarded, then compact voxel
    clusters are associated over time.  Only a repeatedly observed cluster
    with meaningful motion is emitted as a predictive dynamic obstacle; all
    depth hits remain available to the immediate occupancy safety layer.
    """

    # 功能：
    #   建立单一深度来源的有界运动跟踪器；输出预测障碍假设，不提供执行器授权。
    # 输入：
    #   known_static_primitives：可选的当前地图静态基元。
    #   voxel_size_m：聚类体素边长，单位米。
    #   minimum_points：形成一簇所需的不重复端点数。
    #   association_distance_m：预测中心到新簇中心的最大关联距离。
    #   minimum_dynamic_speed_mps：确认运动的最低估计速度。
    #   maximum_dynamic_speed_mps：关联允许的最大实测速度。
    #   maximum_track_age_seconds：未再次观测时保留历史的秒数上限。
    #   static_exclusion_m：静态结构附近的排除余量，单位米。
    #   maximum_dynamic_acceleration_mps2：拒绝不合理关联跳变的加速度上限。
    # 输出：
    #   None：初始化配置、历史及候选提交状态。
    def __init__(
        self,
        *,
        known_static_primitives: list[dict[str, Any]] | None = None,
        voxel_size_m: float = 0.35,
        minimum_points: int = 3,
        association_distance_m: float = 1.2,
        minimum_dynamic_speed_mps: float = 0.12,
        maximum_dynamic_speed_mps: float = 5.0,
        maximum_track_age_seconds: float = 0.8,
        static_exclusion_m: float = 0.12,
        maximum_dynamic_acceleration_mps2: float = 12.0,
    ) -> None:
        if type(minimum_points) is not int or not 1 <= minimum_points <= 250_000:
            raise ValueError("depth tracker minimum points must be a bounded positive integer")
        if (
            not all(
                _finite(value) and value > 0.0
                for value in (
                    voxel_size_m,
                    association_distance_m,
                    maximum_track_age_seconds,
                    minimum_dynamic_speed_mps,
                    maximum_dynamic_speed_mps,
                    maximum_dynamic_acceleration_mps2,
                )
            )
            or not _finite(static_exclusion_m)
            or static_exclusion_m < 0
        ):
            raise ValueError("depth tracker bounds must be finite and valid")
        if minimum_dynamic_speed_mps >= maximum_dynamic_speed_mps:
            raise ValueError("depth tracker speed bounds are invalid")
        self.voxel_size_m = voxel_size_m
        self.minimum_points = minimum_points
        self.association_distance_m = association_distance_m
        self.minimum_dynamic_speed_mps = minimum_dynamic_speed_mps
        self.maximum_dynamic_speed_mps = maximum_dynamic_speed_mps
        self.maximum_track_age_seconds = maximum_track_age_seconds
        self.maximum_dynamic_acceleration_mps2 = maximum_dynamic_acceleration_mps2
        primitives = [] if known_static_primitives is None else known_static_primitives
        if not isinstance(primitives, list):
            raise ValueError("DEPTH_TRACK_STATIC_GEOMETRY_INVALID")
        self._static = (
            _StaticPrimitiveIndex(primitives, exclusion_m=static_exclusion_m)
            if primitives
            else None
        )
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_frame_time: float | None = None
        self._last_frame_unix_ms: int | None = None
        self._sensor_id: str | None = None
        self._sequence = 0
        self._revision = 0
        self._owner = object()
        self._pending: PreparedDepthTracks | None = None
        self._pending_observations: tuple[str, ...] = ()

    # 功能：
    #   将端点映射为聚类格索引，拒绝极小体素导致的除法溢出。
    # 输入：
    #   point：已验证的世界三轴米坐标。
    # 输出：
    #   key：体素三轴整数索引。
    def _key(self, point: Point) -> VoxelKey:
        scaled = tuple(value / self.voxel_size_m for value in point)
        if not all(math.isfinite(value) for value in scaled):
            raise ValueError("DEPTH_TRACK_VOXEL_SCALE_INVALID")
        key = tuple(math.floor(value) for value in scaled)
        return key

    # 功能：
    #   将可靠且非地图表面的不重复端点按二十六邻域聚类，不修改原始占据射线。
    # 输入：
    #   frame：已校验、固定的当前深度帧。
    # 输出：
    #   clusters：按支持点数量和中心坐标排序的簇，包含边界与置信度。
    def _clusters(self, frame: OnboardPerceptionFrame) -> list[_Cluster]:
        buckets: dict[VoxelKey, list[Point]] = defaultdict(list)
        # A projected endpoint indistinguishable from mapped structure cannot
        # establish independent object motion as the camera pans along it.
        # Use measured projection uncertainty, not a map-specific fitted offset.
        # This excludes only a predictive-track hypothesis: the original ray
        # stays occupied in fusion and collision safety, even beside a wall.
        uncertainty_m = 3.0 * math.sqrt(frame.localization_covariance_m2)
        quality_by_point: dict[Point, float] = {}
        for ray in frame.range_rays:
            if not ray.hit or ray.confidence < 0.35:
                continue
            point = _point(ray.endpoint_m)
            if self._static is not None and self._static.contains_known_static(
                point,
                uncertainty_m=uncertainty_m,
            ):
                continue
            if point not in quality_by_point:
                buckets[self._key(point)].append(point)
                quality_by_point[point] = ray.confidence
            else:
                # Repeating one point cannot create independent support.
                quality_by_point[point] = min(quality_by_point[point], ray.confidence)
        pending = set(buckets)
        clusters: list[_Cluster] = []
        neighbor_offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        while pending:
            root = pending.pop()
            component = [root]
            queue: deque[VoxelKey] = deque([root])
            while queue:
                current = queue.popleft()
                for offset in neighbor_offsets:
                    neighbor = tuple(current[index] + offset[index] for index in range(3))
                    if neighbor in pending:
                        pending.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            points = [point for key in component for point in buckets[key]]
            if len(points) < self.minimum_points:
                continue
            minimum = tuple(min(point[index] for point in points) for index in range(3))
            maximum = tuple(max(point[index] for point in points) for index in range(3))
            center = tuple(
                math.fsum(point[index] / len(points) for point in points) for index in range(3)
            )
            clusters.append(
                _Cluster(
                    center=center,
                    minimum=minimum,
                    maximum=maximum,
                    point_count=len(points),
                    confidence=min(
                        sum(quality_by_point[point] for point in points) / len(points),
                        frame.source_coverage if frame.source_coverage is not None else 1.0,
                    ),
                )
            )
        clusters.sort(key=lambda cluster: (-cluster.point_count, cluster.center))
        return clusters

    # 功能：
    #   为无需外部融合验收的调用一次完成准备和提交，失败不推进历史。
    # 输入：
    #   frame：当前世界坐标深度帧。
    #   observed_at_monotonic_seconds：同一扫描的本机单调时钟。
    # 输出：
    #   observations：本帧预测障碍列表。
    def update(
        self,
        frame: OnboardPerceptionFrame,
        *,
        observed_at_monotonic_seconds: float,
    ) -> list[DynamicObstacleObservation]:
        prepared = self.prepare(frame, observed_at_monotonic_seconds=observed_at_monotonic_seconds)
        self.commit(prepared)
        observations = list(prepared.observations)
        return observations

    # 功能：
    #   1. 严格重验帧、来源和时序，估计运动并保留遮挡目标的原始年龄。
    #   2. 生成等待融合接纳的候选，不修改已提交历史；新候选替代旧候选。
    # 输入：
    #   frame：当前深度帧及定位不确定度。
    #   observed_at_monotonic_seconds：与所有射线一致且递增的本机时钟。
    # 输出：
    #   prepared：绑定本跟踪器及当前历史的待提交状态与障碍假设。
    def prepare(
        self,
        frame: OnboardPerceptionFrame,
        *,
        observed_at_monotonic_seconds: float,
    ) -> PreparedDepthTracks:
        frame = OnboardPerceptionFrame.model_validate(frame.model_dump(mode="python"), strict=True)
        if (
            not _finite(observed_at_monotonic_seconds)
            or (observed_at_monotonic_seconds < 0)
            or (
                self._last_frame_time is not None
                and observed_at_monotonic_seconds <= self._last_frame_time
            )
            or (
                self._last_frame_unix_ms is not None
                and frame.observed_at_unix_ms <= self._last_frame_unix_ms
            )
        ):
            raise ValueError("DEPTH_TRACK_FRAME_TIMESTAMP_NOT_MONOTONIC")
        if self._sensor_id is not None and frame.sensor_id != self._sensor_id:
            raise ValueError("DEPTH_TRACK_SOURCE_CHANGED")
        if frame.sequence <= self._sequence:
            raise ValueError("DEPTH_TRACK_SEQUENCE_NOT_MONOTONIC")
        if any(
            not math.isclose(
                ray.observed_at_monotonic_seconds,
                observed_at_monotonic_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for ray in frame.range_rays
        ):
            raise ValueError("DEPTH_TRACK_SCAN_CLOCK_MISMATCH")
        if frame.localization_covariance_m2 > 0.25:
            raise ValueError("DEPTH_TRACK_LOCALIZATION_UNCERTAIN")
        previous = {
            track_id: track
            for track_id, track in self._tracks.items()
            if 0.0
            < observed_at_monotonic_seconds - track.observed_at_seconds
            <= self.maximum_track_age_seconds
        }
        available = set(previous)
        updated: dict[int, _Track] = {}
        next_track_id = self._next_track_id
        clusters = self._clusters(frame)
        if len(clusters) + len(previous) > 1024:
            raise ValueError("DEPTH_TRACK_CAPACITY_EXCEEDED")
        for cluster in clusters:
            match_id: int | None = None
            match_distance = math.inf
            for track_id in sorted(available):
                track = previous[track_id]
                elapsed = observed_at_monotonic_seconds - track.observed_at_seconds
                predicted = tuple(track.center[i] + track.velocity[i] * elapsed for i in range(3))
                distance = math.dist(cluster.center, predicted)
                if distance <= self.association_distance_m and distance < match_distance:
                    match_id = track_id
                    match_distance = distance
            if match_id is None:
                track_id = next_track_id
                next_track_id += 1
                updated[track_id] = _Track(
                    track_id=track_id,
                    center=cluster.center,
                    velocity=(0.0, 0.0, 0.0),
                    observed_at_seconds=observed_at_monotonic_seconds,
                    observed_at_unix_ms=frame.observed_at_unix_ms,
                    seen_count=1,
                    cluster=cluster,
                    motion_anchor=cluster.center,
                    motion_anchor_variance_m2=frame.localization_covariance_m2,
                )
                continue
            prior = previous[match_id]
            elapsed = observed_at_monotonic_seconds - prior.observed_at_seconds
            raw_velocity = tuple(
                (cluster.center[index] - prior.center[index]) / elapsed for index in range(3)
            )
            raw_speed = math.dist((0.0, 0.0, 0.0), raw_velocity)
            if raw_speed > self.maximum_dynamic_speed_mps:
                continue
            velocity = tuple(
                0.65 * prior.velocity[index] + 0.35 * raw_velocity[index] for index in range(3)
            )
            if (
                math.dist(velocity, prior.velocity) / elapsed
                > self.maximum_dynamic_acceleration_mps2
            ):
                # A centroid/association jump is not a measured acceleration.
                # Retain the original detection and its age instead of feeding
                # an impossible estimate into fusion, or clamping it to a
                # fabricated new measurement. All depth hits remain occupied.
                continue
            available.remove(match_id)
            # Absolute localization uncertainty must not be differentiated into
            # a confident moving object. Accumulate displacement relative to
            # the original cluster, without assuming independent pose errors.
            # Until motion is distinguishable, raw occupied rays still guard
            # the vehicle; no predictive velocity is declared established.
            motion_noise_bound = 3.0 * (
                math.sqrt(prior.motion_anchor_variance_m2)
                + math.sqrt(frame.localization_covariance_m2)
            )
            measurable_motion = math.dist(cluster.center, prior.motion_anchor) > motion_noise_bound
            updated[match_id] = _Track(
                track_id=match_id,
                center=cluster.center,
                velocity=velocity,
                observed_at_seconds=observed_at_monotonic_seconds,
                observed_at_unix_ms=frame.observed_at_unix_ms,
                seen_count=prior.seen_count + 1,
                cluster=cluster,
                motion_anchor=prior.motion_anchor,
                motion_anchor_variance_m2=prior.motion_anchor_variance_m2,
                motion_confirmed=(
                    prior.motion_confirmed
                    or (
                        measurable_motion
                        and math.dist((0.0, 0.0, 0.0), velocity) >= self.minimum_dynamic_speed_mps
                    )
                ),
            )
        # Occlusion is not evidence of free space. Retain the last measured
        # center and timestamp, not a prediction disguised as a new detection.
        updated.update((track_id, previous[track_id]) for track_id in available)
        if len(updated) > 512:
            raise ValueError("DEPTH_TRACK_CAPACITY_EXCEEDED")
        observations: list[DynamicObstacleObservation] = []
        for track in updated.values():
            size = tuple(
                track.cluster.maximum[index] - track.cluster.minimum[index] for index in range(3)
            )
            if track.seen_count < 2 or not track.motion_confirmed or max(size) > 3.0:
                continue
            extent = tuple(
                max(
                    abs(track.cluster.maximum[i] - track.center[i]),
                    abs(track.cluster.minimum[i] - track.center[i]),
                )
                for i in range(3)
            )
            observations.append(
                DynamicObstacleObservation(
                    obstacle_id=f"depth-track-{track.track_id}",
                    position_m=Vector3(x=track.center[0], y=track.center[1], z=track.center[2]),
                    velocity_mps=Vector3(
                        x=track.velocity[0], y=track.velocity[1], z=track.velocity[2]
                    ),
                    # The density-weighted tracking centroid need not be the
                    # box midpoint. Enclose every measured endpoint without
                    # clipping dimensions; unseen surfaces remain unknown.
                    radius_m=max(0.2, math.hypot(extent[0], extent[1]) + 0.15),
                    height_m=max(0.3, 2 * extent[2] + 0.3),
                    confidence=min(
                        track.cluster.confidence, 0.95, 0.55 + 0.08 * (track.seen_count - 2)
                    ),
                    age_seconds=(frame.observed_at_unix_ms - track.observed_at_unix_ms) / 1_000,
                )
            )
        prepared = PreparedDepthTracks(
            self._owner,
            self._revision,
            tuple(updated.values()),
            next_track_id,
            frame.sensor_id,
            frame.sequence,
            observed_at_monotonic_seconds,
            frame.observed_at_unix_ms,
            tuple(sorted(observations, key=lambda value: value.obstacle_id)),
        )
        # dataclass 的 frozen 不会冻结内部 Pydantic 对象；提交前须核对实际交付内容。
        self._pending_observations = tuple(item.model_dump_json() for item in prepared.observations)
        self._pending = prepared
        return prepared

    # 功能：
    #   融合接纳后提交本实例最新且未改写的候选；拒绝克隆、重放及跨实例提交。
    # 输入：
    #   prepared：由本跟踪器最近一次成功准备的候选。
    # 输出：
    #   None：更新历史及序号，不返回业务数据。
    def commit(self, prepared: PreparedDepthTracks) -> None:
        if (
            not isinstance(prepared, PreparedDepthTracks)
            or prepared is not self._pending
            or prepared.owner is not self._owner
            or prepared.revision != self._revision
        ):
            raise ValueError("DEPTH_TRACK_PREPARED_STATE_SUPERSEDED")
        if (
            tuple(item.model_dump_json() for item in prepared.observations)
            != self._pending_observations
        ):
            raise ValueError("DEPTH_TRACK_PREPARED_STATE_MODIFIED")
        self._tracks = {track.track_id: track for track in prepared.tracks}
        self._next_track_id = prepared.next_track_id
        self._sensor_id = prepared.sensor_id
        self._sequence = prepared.sequence
        self._last_frame_time = prepared.observed_at_seconds
        self._last_frame_unix_ms = prepared.observed_at_unix_ms
        self._revision += 1
        self._pending = None
        self._pending_observations = ()
