"""Metric 3-D global planning over qualified static collision semantics.

The topology graph remains useful for named places, but it is deliberately not
treated as the flight path. This module derives search bounds and route geometry
from qualified collision geometry, using the complete vehicle envelope plus an
explicit operational margin. Authored graph edges are never required.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dronedream_plugin_sdk.protocol import decode_json

from .assets import MAX_SEMANTIC_BYTES
from .collision import (
    _box_half_sizes,
    _clearance,
    planning_collision_primitives,
    primitive_bounds,
)
from .contracts import GraphRoute, MapAsset, RouteQuery, Vector3
from .plugin_files import read_plugin_file
from .preferred_airspace import PreferredAirspace, checked_point

GridKey = tuple[int, int, int]
Point = tuple[float, float, float]


@dataclass(frozen=True)
class MetricPlannerPolicy:
    resolution_m: float = 0.45
    required_clearance_m: float = 0.36
    search_padding_m: float = 3.0
    vertical_cost_multiplier: float = 2.4
    clearance_cost_weight: float = 0.18
    corridor_penalty_weight: float = 0.0
    corridor_penalty_radius_cells: int = 2
    maximum_expansions: int = 450_000
    maximum_search_seconds: float | None = None
    preferred_airspace_weight: float = 1.5
    allow_semantic_endpoint_projection: bool = False
    maximum_semantic_endpoint_projection_m: float = 2.5
    avoid_building_overflight: bool = True

    # 功能：
    #   校验规划精度、非负代价和计算预算，禁止 NaN、布尔或负惩罚破坏有界 A* 搜索。
    # 输入：
    #   self：不可变的规划策略配置。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        for name in ("resolution_m", "required_clearance_m", "search_padding_m",
                     "vertical_cost_multiplier", "clearance_cost_weight", "corridor_penalty_weight",
                     "maximum_search_seconds", "preferred_airspace_weight"):
            value = getattr(self, name)
            if name == "maximum_search_seconds" and value is None:
                continue
            if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
                raise ValueError(f"metric planner {name} must be finite and non-negative")
        if (type(self.corridor_penalty_radius_cells) is not int
                or not 0 <= self.corridor_penalty_radius_cells <= 64):
            raise ValueError("metric planner corridor penalty radius is outside its budget")
        if type(self.maximum_expansions) is not int:
            raise ValueError("maximum expansions must be an integer")
        if not 0.15 <= self.resolution_m <= 2.0:
            raise ValueError("metric planner resolution is outside the supported range")
        if self.required_clearance_m < 0:
            raise ValueError("required clearance must not be negative")
        if self.search_padding_m <= 0:
            raise ValueError("search padding must be positive")
        if self.vertical_cost_multiplier < 1:
            raise ValueError("vertical motion must not be cheaper than horizontal motion")
        if self.maximum_expansions < 1:
            raise ValueError("maximum expansions must be positive")
        if self.maximum_search_seconds is not None and self.maximum_search_seconds <= 0:
            raise ValueError("maximum search duration must be positive")
        if type(self.allow_semantic_endpoint_projection) is not bool:
            raise ValueError("semantic endpoint projection policy must be boolean")
        if type(self.avoid_building_overflight) is not bool:
            raise ValueError("building overflight policy must be boolean")
        if (
            type(self.maximum_semantic_endpoint_projection_m) not in (int, float)
            or not 0 < self.maximum_semantic_endpoint_projection_m <= 5.0
        ):
            raise ValueError("semantic endpoint projection distance is outside its budget")


# 功能：
#   复用碰撞模块的完整旋转包围盒并膨胀操作余量，避免路径索引漏掉倾斜楼板。
# 输入：
#   primitive：已加载的米制碰撞图元。
#   expansion_m：需要在各轴保守扩张的距离。
# 输出：
#   bounds：扩张后的三维最小和最大坐标。
def _primitive_bounds(primitive: dict[str, Any], expansion_m: float) -> tuple[Point, Point]:
    bounds = primitive_bounds(primitive, expansion_m)
    return bounds


class KnownMapMetricPlanner:
    """A* planner whose occupancy source is qualified metric collision geometry."""

    _SPATIAL_BIN_M = 2.0

    # 功能：
    #   1. 从同一有界文件快照解析静态几何并计算摘要，可与独立净空报告绑定。
    #   2. 依据机体包络建立有界粗筛索引和路线缓存，图节点仅用于命名及搜索范围。
    # 输入：
    #   self：新规划器。
    #   graph：可选拓扑图；仅查询净空时允许为空。
    #   semantic_path：静态碰撞语义文件。
    #   vehicle_diameter_m、vehicle_height_m：完整机体尺寸。
    #   policy：分辨率、操作净空及搜索预算。
    #   expected_semantic_sha256：可选的上游已批准地图摘要。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        graph: MapAsset | None,
        semantic_path: Path,
        vehicle_diameter_m: float,
        vehicle_height_m: float,
        policy: MetricPlannerPolicy | None = None,
        expected_semantic_sha256: str | None = None,
    ) -> None:
        if any(type(value) not in (int, float) or not 0 < value <= sys.float_info.max
               for value in (vehicle_diameter_m, vehicle_height_m)):
            raise ValueError("vehicle envelope dimensions must be positive")
        raw = read_plugin_file(semantic_path, limit=MAX_SEMANTIC_BYTES)
        semantic_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_semantic_sha256 is not None and semantic_sha256 != expected_semantic_sha256:
            raise ValueError("qualified static map semantic hash mismatch")
        semantic = decode_json(raw, limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000)
        if not isinstance(semantic, dict):
            raise ValueError("semantic artifact has no collision primitives")
        primitives = planning_collision_primitives(semantic)
        self.graph = graph
        self.semantic_path = semantic_path
        self.semantic_sha256 = semantic_sha256
        self.vehicle_radius_m = vehicle_diameter_m / 2
        self.vehicle_half_height_m = vehicle_height_m / 2
        self.policy = policy or MetricPlannerPolicy()
        # 数值已经由统一入口验证；冻结独立副本，避免每个采样点重复解析数千个图元。
        self.primitives = tuple(MappingProxyType(primitive) for primitive in primitives)
        self.preferred_airspace = PreferredAirspace(
            semantic, {"semantic_file_sha256": semantic_sha256},
            self.vehicle_radius_m, vehicle_height_m)
        self.nodes = {node.node_id: node for node in graph.nodes} if graph is not None else {}
        self._route_cache: dict[tuple[str, str, Point, Point], GraphRoute] = {}
        self._spatial: dict[GridKey, list[int]] = defaultdict(list)
        self._overflight_spatial: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._building_footprints: list[tuple[float, float, float, float, float]] = []
        self._clearance_cap_m = self.policy.required_clearance_m + self.policy.resolution_m
        # 圆形障碍检查采用机体外包球，索引必须使用同样的外包半径，不能只取最大半轴。
        expansion = math.hypot(self.vehicle_radius_m, self.vehicle_half_height_m)
        expansion += self._clearance_cap_m
        indexed_cells = 0
        for index, primitive in enumerate(primitives):
            low, high = _primitive_bounds(primitive, expansion)
            if any(not math.isfinite(value) for value in (*low, *high)) or any(
                low[axis] > high[axis] for axis in range(3)
            ):
                raise ValueError("metric planner primitive bounds invalid")
            bins = self._bin_range(low, high)
            indexed_cells += len(bins)
            if indexed_cells > 2_000_000:
                raise ValueError("metric planner spatial index budget exceeded")
            for key in bins:
                self._spatial[key].append(index)
            if self.policy.avoid_building_overflight and primitive.get("semantic") == "building":
                if "size_x" not in primitive:
                    raise ValueError("building overflight policy requires box geometry")
                half_x, half_y, _ = _box_half_sizes(primitive)
                footprint_low, footprint_high = _primitive_bounds(
                    primitive,
                    self.vehicle_radius_m,
                )
                footprint_index = len(self._building_footprints)
                self._building_footprints.append((
                    float(primitive["center_x"]),
                    float(primitive["center_y"]),
                    half_x,
                    half_y,
                    float(primitive.get("yaw_rad", 0.0)),
                ))
                outer_low = (
                    footprint_low[0] - self._clearance_cap_m,
                    footprint_low[1] - self._clearance_cap_m,
                )
                outer_high = (
                    footprint_high[0] + self._clearance_cap_m,
                    footprint_high[1] + self._clearance_cap_m,
                )
                first = tuple(math.floor(value / self._SPATIAL_BIN_M) for value in outer_low)
                last = tuple(math.floor(value / self._SPATIAL_BIN_M) for value in outer_high)
                for x in range(first[0], last[0] + 1):
                    for y in range(first[1], last[1] + 1):
                        self._overflight_spatial[(x, y)].append(footprint_index)

    # 功能：
    #   将世界坐标映射到粗筛网格，负坐标使用向下取整，不能截断到零。
    # 输入：
    #   self：当前规划器。
    #   point：世界米制三维坐标。
    # 输出：
    #   key：空间索引的整数键。
    def _bin_key(self, point: Point) -> GridKey:
        return tuple(math.floor(value / self._SPATIAL_BIN_M) for value in point)  # type: ignore[return-value]

    # 功能：
    #   枚举包围盒覆盖的空间桶，在分配列表前检查单图元索引预算。
    # 输入：
    #   self：当前规划器。
    #   low、high：保守包围盒下界与上界。
    # 输出：
    #   keys：需登记图元的空间桶键列表。
    def _bin_range(self, low: Point, high: Point) -> list[GridKey]:
        first = self._bin_key(low)
        last = self._bin_key(high)
        if math.prod(max(0, last[i] - first[i] + 1) for i in range(3)) > 1_000_000:
            raise ValueError("metric planner primitive index budget exceeded")
        return [
            (x, y, z)
            for x in range(first[0], last[0] + 1)
            for y in range(first[1], last[1] + 1)
            for z in range(first[2], last[2] + 1)
        ]

    # 功能：
    #   对冻结几何计算饱和净空下界；索引外图元距机体必大于饱和值，不能把漏选当作无限净空。
    # 输入：
    #   self：当前规划器。
    #   point：机体碰撞包络中心坐标。
    # 输出：
    #   clearance_m：不超过真实保守净空及饱和值的有限距离，不代表实时障碍感知。
    def clearance(self, point: Point) -> float:
        point = checked_point(point)
        key = self._bin_key(point)
        candidates = self._spatial.get(key, ())
        clearance_m = self._clearance_cap_m
        for index in candidates:
            value = _clearance(
                point,
                self.primitives[index],
                radius_m=self.vehicle_radius_m,
                half_height_m=self.vehicle_half_height_m,
            )
            if not math.isfinite(value):
                raise ValueError("metric planner clearance overflow")
            clearance_m = min(clearance_m, value)
        # Building roofs are not ordinary open air for the default campus
        # mission policy.  Keep the complete vehicle footprint outside the
        # building's horizontal bounds at every altitude, so a collision-free
        # climb cannot turn into a socially unsafe roof shortcut.
        for index in self._overflight_spatial.get((key[0], key[1]), ()):
            center_x, center_y, half_x, half_y, yaw = self._building_footprints[index]
            delta_x, delta_y = point[0] - center_x, point[1] - center_y
            cosine, sine = math.cos(yaw), math.sin(yaw)
            local_x = cosine * delta_x + sine * delta_y
            local_y = -sine * delta_x + cosine * delta_y
            outside_x = max(abs(local_x) - half_x, 0.0)
            outside_y = max(abs(local_y) - half_y, 0.0)
            horizontal_clearance = math.hypot(outside_x, outside_y) - self.vehicle_radius_m
            clearance_m = min(clearance_m, horizontal_clearance)
        return clearance_m

    # 功能：
    #   以净空的 1-Lipschitz 下界证明整段安全；近障段自适应细分，无法证明时保守拒绝。
    # 输入：
    #   self：当前规划器。
    #   start、end：线段端点。
    # 输出：
    #   free：包括采样点之间的整条线段均满足操作净空时为真。
    def _segment_is_free(self, start: Point, end: Point) -> bool:
        start, end = checked_point(start), checked_point(end)
        pending = [(start, end, self.clearance(start), self.clearance(end))]
        checks = 0
        while pending:
            first, last, first_clearance, last_clearance = pending.pop()
            distance = math.dist(first, last)
            minimum = min(first_clearance, last_clearance)
            if minimum < self.policy.required_clearance_m:
                return False
            # 到最近端点至多半段长；这是真正的连续净空证明，不是跳过中间障碍。
            if minimum - distance / 2 >= self.policy.required_clearance_m:
                continue
            # 端点余量可能只有几毫米，不能在厘米级停止并误判整条窄通道不可达。
            if distance <= 0.0001:
                return False
            checks += 1
            if checks > 250_000:
                raise ValueError("metric segment proof budget exceeded")
            middle = tuple((first[axis] + last[axis]) / 2 for axis in range(3))
            middle_clearance = self.clearance(middle)
            pending.append((middle, last, middle_clearance, last_clearance))
            pending.append((first, middle, first_clearance, middle_clearance))
        return True

    # 功能：
    #   仅依据当前实测起终点和地图碰撞几何建立搜索范围，不要求地图作者预画航线。
    # 输入：
    #   self：当前规划器。
    #   exact_start、exact_goal：当前真实任务起终点。
    # 输出：
    #   bounds：本次有界搜索空间的下界和上界。
    def _search_bounds(
        self,
        *,
        exact_start: Point,
        exact_goal: Point,
    ) -> tuple[Point, Point]:
        points = [exact_start, exact_goal]
        # Obstacles that cross the direct search prism enlarge the prism far
        # enough for A* to get around their complete vehicle-expanded bounds.
        # Horizontal terrain/floor slabs deliberately do not expand X/Y: doing
        # so would turn every local request into a full-map voxel search.
        padding = max(
            self.policy.search_padding_m,
            self.vehicle_radius_m + self.policy.required_clearance_m + self.policy.resolution_m,
        )
        low = [min(point[axis] for point in points) - padding for axis in range(3)]
        high = [max(point[axis] for point in points) + padding for axis in range(3)]
        lateral_semantics = {"terrain", "floor", "ground", "ground-plane"}
        for primitive in self.primitives:
            primitive_low, primitive_high = _primitive_bounds(
                dict(primitive),
                self.vehicle_radius_m + self.policy.required_clearance_m,
            )
            if primitive.get("semantic") in lateral_semantics:
                covers_both_endpoints = all(
                    primitive_low[axis] <= exact_start[axis] <= primitive_high[axis]
                    and primitive_low[axis] <= exact_goal[axis] <= primitive_high[axis]
                    for axis in (0, 1)
                )
                if covers_both_endpoints and primitive_high[2] <= min(
                    exact_start[2], exact_goal[2]
                ):
                    # A finite ground mesh must behave as a floor for this local
                    # search. Otherwise A* can leave its underside and tunnel
                    # below terrain because the mesh itself has finite thickness.
                    low[2] = max(low[2], primitive_high[2])
                continue
            overlaps_direct_prism = all(
                primitive_high[axis] >= low[axis] and primitive_low[axis] <= high[axis]
                for axis in range(3)
            )
            if not overlaps_direct_prism:
                continue
            for axis in (0, 1):
                low[axis] = min(low[axis], primitive_low[axis] - padding)
                high[axis] = max(high[axis], primitive_high[axis] + padding)
        preferred_upper = max(
            (band.preferred_upper_z_m for point in points
             if (band := self.preferred_airspace.band_at(point)) is not None),
            default=max(point[2] for point in points))
        return (
            (
                low[0],
                low[1],
                low[2],
            ),
            (
                high[0],
                high[1],
                max(high[2], preferred_upper),
            ),
        )

    # 功能：
    #   将命名节点解析为实际端点后求解碰撞几何路线，不伪装为经过真机飞行验证的边。
    # 输入：
    #   self：当前规划器。
    #   query：起终点及路线要求。
    #   prior_corridors：应施加额外代价以探索其他路线的历史走廊。
    # 输出：
    #   route：新的米制几何路线。
    def plan(
        self,
        query: RouteQuery,
        *,
        prior_corridors: list[list[Point]] | None = None,
    ) -> GraphRoute:
        if query.start_node not in self.nodes or query.goal_node not in self.nodes:
            raise ValueError("route endpoint is not present in graph")
        if query.require_flight_verified_edges:
            raise ValueError("metric geometry routes are not flight-verified edges")
        start_node = self.nodes[query.start_node]
        goal_node = self.nodes[query.goal_node]
        # A qualified launch node is an exact vehicle pose, not a semantic area
        # label.  Projecting it can produce a collision-clear route that no longer
        # begins or returns at the PX4 launch binding.  Non-launch semantic targets
        # still use the same deterministic projection in both directions so adjacent
        # route segments remain continuous.
        start_point = (
            start_node.position_m
            if start_node.semantic == "launch"
            else self._semantic_action_pose(start_node.position_m)
        )
        goal_point = (
            goal_node.position_m
            if goal_node.semantic == "launch"
            else self._semantic_action_pose(goal_node.position_m)
        )
        return self.plan_positions(
            query,
            start_position_m=start_point,
            goal_position_m=goal_point,
            prior_corridors=prior_corridors,
        )

    def _semantic_action_pose(self, anchor: Vector3) -> Vector3:
        """Resolve a map label anchor to the nearest clearance-valid operation pose.

        Imported maps commonly place pantry, doorway and launch labels on a wall or
        room boundary.  Those labels identify an area; they are not authoritative live
        vehicle poses.  Initial planning may therefore use a nearby collision-valid pose,
        while ``plan_positions`` deliberately remains fail-closed for measured positions.
        """

        origin = (anchor.x, anchor.y, anchor.z)
        if self.clearance(origin) >= self.policy.required_clearance_m:
            return anchor
        if not self.policy.allow_semantic_endpoint_projection:
            return anchor
        spacing = max(0.15, min(self.policy.resolution_m / 2, 0.3))
        radius = self.policy.maximum_semantic_endpoint_projection_m
        extent = math.ceil(radius / spacing)
        candidates: list[tuple[float, float, float, Point]] = []
        for x_index in range(-extent, extent + 1):
            for y_index in range(-extent, extent + 1):
                for z_index in range(-extent, extent + 1):
                    offset = (x_index * spacing, y_index * spacing, z_index * spacing)
                    distance = math.sqrt(sum(value * value for value in offset))
                    if distance == 0 or distance > radius:
                        continue
                    point = tuple(origin[axis] + offset[axis] for axis in range(3))
                    vertical = abs(offset[2])
                    preference = self.preferred_airspace.cost(point)
                    candidates.append((distance + vertical * 1.4 + preference * 0.1,
                                       distance, vertical, point))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _, _, _, point in candidates:
            if self.clearance(point) >= self.policy.required_clearance_m:
                return Vector3(x=point[0], y=point[1], z=point[2])
        raise ValueError("semantic route endpoint has no nearby operational pose")

    # 功能：
    #   1. 在有界三维网格执行 A*，代价包括水平、垂直运动、净空和可选历史走廊惩罚。
    #   2. 保留精确端点，验证连接并简化路线；仅无历史惩罚查询复用对称往返缓存。
    # 输入：
    #   self：当前规划器。
    #   query：命名起终点及验收要求。
    #   start_position_m、goal_position_m：本次权威起终点，不使用旧节点坐标替代。
    #   prior_corridors：可选历史路线点集。
    # 输出：
    #   route：独立持有的三维路线，飞行验证标记为假。
    def plan_positions(
        self,
        query: RouteQuery,
        *,
        start_position_m: Vector3,
        goal_position_m: Vector3,
        prior_corridors: list[list[Point]] | None = None,
    ) -> GraphRoute:
        if query.start_node not in self.nodes or query.goal_node not in self.nodes:
            raise ValueError("route endpoint is not present in graph")
        if query.require_flight_verified_edges:
            raise ValueError("metric geometry routes are not flight-verified edges")
        exact_start = (
            start_position_m.x,
            start_position_m.y,
            start_position_m.z,
        )
        exact_goal = (
            goal_position_m.x,
            goal_position_m.y,
            goal_position_m.z,
        )
        cache_key = (query.start_node, query.goal_node, exact_start, exact_goal)
        # 带历史走廊惩罚的求解不是同一查询，不能误用或覆盖无惩罚的往返路线缓存。
        cached = self._route_cache.get(cache_key) if not prior_corridors else None
        if cached is not None:
            return cached.model_copy(deep=True)

        # 功能：
        #   将经连续碰撞证明的点列封装并缓存为本次语义航段；反向复用依据静态几何对称性。
        # 输入：
        #   points：包含精确起终点的路线点。
        # 输出：
        #   route：绑定命名端点且不冒充飞行验证边的路线。
        def finish(points: list[Point]) -> GraphRoute:
            simplified = self._round_horizontal_corners(self._simplify(points))
            digest = hashlib.sha256(
                json.dumps(simplified, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            node_ids = [
                query.start_node,
                *(f"metric-{digest}-{index:04d}" for index in range(1, len(simplified) - 1)),
                query.goal_node,
            ]
            edge_ids = [
                f"metric-edge-{digest}-{index:04d}" for index in range(len(simplified) - 1)
            ]
            route = GraphRoute(
                start_node=query.start_node,
                goal_node=query.goal_node,
                node_ids=node_ids,
                edge_ids=edge_ids,
                positions_m=[Vector3(x=item[0], y=item[1], z=item[2]) for item in simplified],
                route_length_m=sum(
                    math.dist(first, second)
                    for first, second in zip(simplified, simplified[1:], strict=False)
                ),
                all_edges_flight_verified=False,
            )
            if prior_corridors:
                return route.model_copy(deep=True)
            self._route_cache[cache_key] = route
            reverse_key = (query.goal_node, query.start_node, exact_goal, exact_start)
            self._route_cache[reverse_key] = GraphRoute(
                start_node=query.goal_node,
                goal_node=query.start_node,
                node_ids=list(reversed(route.node_ids)),
                edge_ids=list(reversed(route.edge_ids)),
                positions_m=list(reversed(route.positions_m)),
                route_length_m=route.route_length_m,
                all_edges_flight_verified=False,
            )
            return route.model_copy(deep=True)

        if self.clearance(exact_start) < self.policy.required_clearance_m:
            raise ValueError("metric route start violates operational clearance")
        if self.clearance(exact_goal) < self.policy.required_clearance_m:
            raise ValueError("metric route goal violates operational clearance")

        # Common outdoor legs do not need a full voxel search. First prove the
        # direct segment; then derive one cruise height from actual obstacle
        # bounds intersecting the start-goal prism. Both paths still undergo the
        # same continuous swept-volume proof as A* output.
        if (
            not prior_corridors
            and (
                self.preferred_airspace.domain is None
                or self.policy.preferred_airspace_weight == 0
            )
            and self._segment_is_free(exact_start, exact_goal)
        ):
            return finish([exact_start, exact_goal])
        if not prior_corridors:
            preferred_upper = max(
                (
                    band.preferred_upper_z_m
                    for point in (exact_start, exact_goal)
                    if (band := self.preferred_airspace.band_at(point)) is not None
                ),
                default=max(exact_start[2], exact_goal[2]),
            )
            cruise_z = max(exact_start[2], exact_goal[2], preferred_upper)
            x_low, x_high = sorted((exact_start[0], exact_goal[0]))
            y_low, y_high = sorted((exact_start[1], exact_goal[1]))
            for primitive in self.primitives:
                primitive_low, primitive_high = _primitive_bounds(
                    dict(primitive),
                    self.vehicle_radius_m + self.policy.required_clearance_m,
                )
                if (
                    primitive_high[0] >= x_low
                    and primitive_low[0] <= x_high
                    and primitive_high[1] >= y_low
                    and primitive_low[1] <= y_high
                ):
                    cruise_z = max(cruise_z, primitive_high[2] + self.policy.resolution_m)
            cruise = [
                exact_start,
                (exact_start[0], exact_start[1], cruise_z),
                (exact_goal[0], exact_goal[1], cruise_z),
                exact_goal,
            ]
            if all(
                self._segment_is_free(first, second)
                for first, second in zip(cruise, cruise[1:], strict=False)
            ):
                return finish(cruise)
        search_start, search_goal = exact_start, exact_goal
        horizontal_corridor = False
        if self.policy.avoid_building_overflight and not prior_corridors:
            # Outdoor campus missions use a low, constant-height corridor between
            # buildings.  Searching every altitude after roof overflight has been
            # disabled wastes most of the voxel budget on equivalent XY cells and
            # can still prefer an unnecessarily high path.  The endpoint climbs
            # remain explicit and receive the same continuous clearance proof.
            corridor_z = max(exact_start[2], exact_goal[2]) + max(
                1.0,
                self.vehicle_half_height_m
                + self.policy.required_clearance_m
                + self.policy.resolution_m,
            )
            candidate_start = (exact_start[0], exact_start[1], corridor_z)
            candidate_goal = (exact_goal[0], exact_goal[1], corridor_z)
            if (
                self._segment_is_free(exact_start, candidate_start)
                and self._segment_is_free(candidate_goal, exact_goal)
            ):
                search_start, search_goal = candidate_start, candidate_goal
                horizontal_corridor = True
        low, high = self._search_bounds(
            exact_start=search_start,
            exact_goal=search_goal,
        )
        # A one-metre class lateral lattice is sufficient for an outdoor
        # vehicle-plus-clearance corridor and keeps a campus detour responsive.
        # Every resulting diagonal is still checked continuously against the
        # original geometry, so this changes search density rather than safety.
        resolution = (
            max(self.policy.resolution_m, min(1.0, self.policy.resolution_m * 2))
            if horizontal_corridor
            else self.policy.resolution_m
        )
        axis_counts = [math.floor((high[axis] - low[axis]) / resolution) + 1
                       for axis in range(3)]
        if any(count < 1 or count > 100_000 for count in axis_counts):
            raise ValueError("metric planner axis budget exceeded")
        # 保留起终点每个真实坐标分量，使窄门/楼梯内仍可先竖直调整，而不是强迫斜着接上网格。
        axes = [sorted({*(low[axis] + index * resolution for index in range(axis_counts[axis])),
                        search_start[axis], search_goal[axis]}) for axis in range(3)]
        if horizontal_corridor:
            axes[2] = [search_start[2]]
        if math.prod(len(axis) for axis in axes) > 20_000_000:
            raise ValueError("metric planner grid budget exceeded")

        # 功能：
        #   将包含精确端点坐标的非均匀网格键还原为世界米制坐标。
        # 输入：
        #   key：以本次搜索下界为原点的网格键。
        # 输出：
        #   position：世界三维位置。
        def key_to_point(key: GridKey) -> Point:
            position = tuple(axes[axis][key[axis]] for axis in range(3))
            return position

        # 功能：
        #   将世界位置量化到本次搜索网格；原始精确端点另行保留。
        # 输入：
        #   point：世界三维位置。
        # 输出：
        #   key：最近网格键。
        def point_to_key(point: Point) -> GridKey:
            indices = []
            for axis in range(3):
                right = min(bisect_left(axes[axis], point[axis]), len(axes[axis]) - 1)
                left = max(0, right - 1)
                indices.append(min((left, right),
                                   key=lambda item: abs(axes[axis][item] - point[axis])))
            key = tuple(indices)
            return key

        maximum_key = tuple(len(axis) - 1 for axis in axes)
        start_key = point_to_key(search_start)
        goal_key = point_to_key(search_goal)
        point_cache: dict[GridKey, Point] = {}
        clearance_cache: dict[GridKey, float] = {}

        # 功能：
        #   缓存网格坐标换算，并允许精确端点覆盖无法满足净空的量化点。
        # 输入：
        #   key：待查询网格键。
        # 输出：
        #   value：该节点实际使用的世界坐标。
        def point(key: GridKey) -> Point:
            value = point_cache.get(key)
            if value is None:
                value = key_to_point(key)
                point_cache[key] = value
            return value

        # 功能：
        #   检查节点是否仍位于本次有限搜索空间。
        # 输入：
        #   key：待扩展邻居键。
        # 输出：
        #   valid：三个坐标均在边界内时为真。
        def valid_key(key: GridKey) -> bool:
            return all(0 <= key[axis] <= maximum_key[axis] for axis in range(3))

        # 功能：
        #   缓存同一网格节点的几何净空，减少重复碰撞计算。
        # 输入：
        #   key：待查询节点。
        # 输出：
        #   value：完整机体相对静态障碍的净空。
        def key_clearance(key: GridKey) -> float:
            value = clearance_cache.get(key)
            if value is None:
                value = self.clearance(point(key))
                clearance_cache[key] = value
            return value

        if key_clearance(start_key) < self.policy.required_clearance_m:
            # The exact endpoint is authoritative; retain it as a special grid node.
            point_cache[start_key] = search_start
            clearance_cache[start_key] = self.clearance(search_start)
        if key_clearance(goal_key) < self.policy.required_clearance_m:
            point_cache[goal_key] = search_goal
            clearance_cache[goal_key] = self.clearance(search_goal)

        penalized: set[GridKey] = set()
        for corridor in prior_corridors or []:
            for corridor_point in corridor:
                center = point_to_key(corridor_point)
                radius = self.policy.corridor_penalty_radius_cells
                penalized.update(
                    (center[0] + dx, center[1] + dy, center[2] + dz)
                    for dx in range(-radius, radius + 1)
                    for dy in range(-radius, radius + 1)
                    for dz in range(-1, 2)
                    if dx * dx + dy * dy <= radius * radius
                )

        moves = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]

        # 功能：
        #   估计剩余水平距离与加权垂直距离，不把非负障碍惩罚加入乐观下界。
        # 输入：
        #   key：当前搜索节点。
        # 输出：
        #   estimate：到目标的基础运动代价下界。
        def heuristic(key: GridKey) -> float:
            position = point(key)
            dx, dy, dz = (position[i] - search_goal[i] for i in range(3))
            estimate = math.hypot(dx, dy) + abs(dz) * self.policy.vertical_cost_multiplier
            return estimate

        queue: list[tuple[float, float, GridKey]] = [(heuristic(start_key), 0.0, start_key)]
        costs = {start_key: 0.0}
        previous: dict[GridKey, GridKey] = {}
        expansions = 0
        search_started = time.monotonic()
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost != costs.get(current):
                continue
            if current == goal_key:
                break
            expansions += 1
            if expansions > self.policy.maximum_expansions:
                raise ValueError("metric route search expansion budget exhausted")
            if (
                self.policy.maximum_search_seconds is not None
                and expansions % 64 == 0
                and time.monotonic() - search_started
                > self.policy.maximum_search_seconds
            ):
                raise ValueError("metric route search deadline exceeded")
            current_point = point(current)
            for dx, dy, dz in moves:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not valid_key(neighbor):
                    continue
                clearance = key_clearance(neighbor)
                if clearance < self.policy.required_clearance_m:
                    continue
                neighbor_point = point(neighbor)
                horizontal = math.hypot(neighbor_point[0] - current_point[0],
                                        neighbor_point[1] - current_point[1])
                vertical = (abs(neighbor_point[2] - current_point[2])
                            * self.policy.vertical_cost_multiplier)
                step_cost = horizontal + vertical
                if self.preferred_airspace.domain is not None:
                    step_cost += (math.hypot(horizontal, vertical)
                                  * self.policy.preferred_airspace_weight
                                  * self.preferred_airspace.cost(neighbor_point))
                if math.isfinite(clearance):
                    step_cost += self.policy.clearance_cost_weight / max(
                        clearance - self.policy.required_clearance_m + 0.05,
                        0.05,
                    )
                if neighbor in penalized:
                    step_cost += self.policy.corridor_penalty_weight
                candidate = cost + step_cost
                if candidate < costs.get(neighbor, math.inf):
                    # 没有更低代价的边不会进入路线，先剪枝再做昂贵的连续碰撞证明。
                    if not self._segment_is_free(current_point, neighbor_point):
                        continue
                    costs[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
        if goal_key not in costs:
            raise ValueError("no metric route satisfies operational clearance")

        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(previous[keys[-1]])
        keys.reverse()
        searched_points = [point(key) for key in keys]
        raw_points = [exact_start, *searched_points, exact_goal]
        raw_points = [
            item for index, item in enumerate(raw_points)
            if index == 0 or math.dist(item, raw_points[index - 1]) > 1e-9
        ]
        return finish(raw_points)

    def _round_horizontal_corners(self, points: list[Point]) -> list[Point]:
        """Replace safe horizontal sharp turns with short collision-checked arcs."""

        if len(points) < 3:
            return points
        output = [points[0]]
        for previous, corner, following in zip(
            points, points[1:-1], points[2:], strict=False
        ):
            incoming = tuple(corner[axis] - previous[axis] for axis in range(3))
            outgoing = tuple(following[axis] - corner[axis] for axis in range(3))
            incoming_horizontal = math.hypot(incoming[0], incoming[1])
            outgoing_horizontal = math.hypot(outgoing[0], outgoing[1])
            if (
                incoming_horizontal < 0.5
                or outgoing_horizontal < 0.5
                or abs(incoming[2]) > 0.05
                or abs(outgoing[2]) > 0.05
            ):
                output.append(corner)
                continue
            trim = min(2.0, incoming_horizontal * 0.2, outgoing_horizontal * 0.2)
            entry = (
                corner[0] - incoming[0] / incoming_horizontal * trim,
                corner[1] - incoming[1] / incoming_horizontal * trim,
                corner[2],
            )
            exit_point = (
                corner[0] + outgoing[0] / outgoing_horizontal * trim,
                corner[1] + outgoing[1] / outgoing_horizontal * trim,
                corner[2],
            )
            curve = [entry]
            for index in range(1, 5):
                ratio = index / 4
                inverse = 1 - ratio
                curve.append(
                    tuple(
                        inverse * inverse * entry[axis]
                        + 2 * inverse * ratio * corner[axis]
                        + ratio * ratio * exit_point[axis]
                        for axis in range(3)
                    )
                )
            candidate = [output[-1], *curve]
            if all(
                self._segment_is_free(first, second)
                for first, second in zip(candidate, candidate[1:], strict=False)
            ):
                output.extend(curve)
            else:
                output.append(corner)
        output.append(points[-1])
        if not all(
            self._segment_is_free(first, second)
            for first, second in zip(output, output[1:], strict=False)
        ):
            raise ValueError("rounded metric route violates operational clearance")
        return output

    # 功能：
    #   贪心选择满足净空的最远可直达点，连相邻点和两点路线也必须复验，不能漏检端点接驳。
    # 输入：
    #   self：当前规划器。
    #   points：网格搜索得到且保留精确端点的原始路线。
    # 输出：
    #   output：通过线段净空检查的简化路线。
    def _simplify(self, points: list[Point]) -> list[Point]:
        if len(points) <= 2:
            if len(points) == 2 and not self._segment_is_free(points[0], points[1]):
                raise ValueError("metric route endpoint connection violates operational clearance")
            return points
        output = [points[0]]
        preference_prefix = [0.0]
        for first, second in zip(points, points[1:], strict=False):
            preference_prefix.append(
                preference_prefix[-1] + self._preference_integral(first, second))
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if (self._segment_is_free(points[anchor], points[candidate])
                        and self._preference_integral(points[anchor], points[candidate])
                        <= preference_prefix[candidate] - preference_prefix[anchor] + 0.05):
                    break
                candidate -= 1
            if not self._segment_is_free(points[anchor], points[candidate]):
                raise ValueError("metric route endpoint connection violates operational clearance")
            output.append(points[candidate])
            anchor = candidate
        return output

    # 功能：
    #   对捷径积分软偏离代价，防止几何简化把已规划的爬升抹平成低空直线。
    # 输入：
    #   start、end：待检查线段。
    # 输出：
    #   cost：按米累积的软偏离代价。
    def _preference_integral(self, start: Point, end: Point) -> float:
        if self.preferred_airspace.domain is None or self.policy.preferred_airspace_weight == 0:
            return 0.0
        distance = math.dist(start, end)
        count = max(1, math.ceil(distance / 0.25))
        cost = distance / count * sum(self.preferred_airspace.cost(tuple(
            start[axis] + (end[axis] - start[axis]) * (index + 0.5) / count
            for axis in range(3))) for index in range(count))
        return cost


# 功能：
#   使用指定机体和操作净空创建一次性规划器，求解命名端点之间的静态几何路线。
# 输入：
#   graph、semantic_path：拓扑命名图和碰撞语义文件。
#   query：路线查询。
#   vehicle_diameter_m、vehicle_height_m、required_clearance_m：机体尺寸及净空要求。
# 输出：
#   route：经局部几何采样检查的路线，不授予真实飞行资格。
def metric_clearance_route(
    graph: MapAsset,
    semantic_path: Path,
    query: RouteQuery,
    *,
    vehicle_diameter_m: float,
    vehicle_height_m: float,
    required_clearance_m: float = 0.36,
) -> GraphRoute:
    planner = KnownMapMetricPlanner(
        graph=graph,
        semantic_path=semantic_path,
        vehicle_diameter_m=vehicle_diameter_m,
        vehicle_height_m=vehicle_height_m,
        policy=MetricPlannerPolicy(required_clearance_m=required_clearance_m),
    )
    return planner.plan(query)
