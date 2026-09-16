"""Metric 3-D global planning over qualified static collision semantics.

The topology graph remains useful for named places and for bounding the search, but
it is deliberately not treated as the flight path.  This module searches the
qualified collision geometry with the complete vehicle envelope plus an explicit
operational margin.  It therefore produces genuinely different paths when free
space permits instead of applying different scores to the same graph chain.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dronedream_plugin_sdk.protocol import decode_json

from .assets import MAX_SEMANTIC_BYTES
from .collision import vehicle_clearance
from .contracts import GraphRoute, MapAsset, RouteQuery, Vector3
from .navigation import shortest_route
from .plugin_files import read_plugin_file

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

    # 功能：
    #   校验规划精度、非负代价和计算预算，禁止 NaN、布尔或负惩罚破坏有界 A* 搜索。
    # 输入：
    #   self：不可变的规划策略配置。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        for name in ("resolution_m", "required_clearance_m", "search_padding_m",
                     "vertical_cost_multiplier", "clearance_cost_weight", "corridor_penalty_weight",
                     "maximum_search_seconds"):
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


# 功能：
#   按机体及操作余量扩张图元包围盒，空间索引只筛候选，精确净空由碰撞几何计算。
# 输入：
#   primitive：已加载的米制碰撞图元。
#   expansion_m：需要在各轴保守扩张的距离。
# 输出：
#   bounds：扩张后的三维最小和最大坐标。
def _primitive_bounds(primitive: dict[str, Any], expansion_m: float) -> tuple[Point, Point]:
    dimensions = ("size_x", "size_y", "size_z") if "size_x" in primitive else ("radius_m",)
    for name in ("center_x", "center_y", "center_z", *dimensions):
        value = primitive.get(name)
        if (type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max):
            raise ValueError("metric planner primitive coordinate or dimension invalid")
        if name in dimensions and value <= 0:
            raise ValueError("metric planner primitive dimension must be positive")
    for name in ("length_m", "height_m", "roll_rad", "pitch_rad", "yaw_rad"):
        value = primitive.get(name, 0.)
        if (type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max
                or (name in {"length_m", "height_m"} and value < 0.)):
            raise ValueError("metric planner primitive extent or rotation invalid")
    center = (
        float(primitive["center_x"]),
        float(primitive["center_y"]),
        float(primitive["center_z"]),
    )
    if "size_x" in primitive:
        half_x = float(primitive["size_x"]) / 2
        half_y = float(primitive["size_y"]) / 2
        yaw = float(primitive.get("yaw_rad", 0.0))
        extent_x = abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y
        extent_y = abs(math.sin(yaw)) * half_x + abs(math.cos(yaw)) * half_y
        extent_z = float(primitive["size_z"]) / 2
    else:
        radius = float(primitive["radius_m"])
        axial = float(primitive.get("length_m", primitive.get("height_m", 0.0))) / 2
        # A conservative sphere bounds arbitrarily oriented capsules/cylinders.
        extent_x = extent_y = extent_z = radius + axial
    extents = (extent_x + expansion_m, extent_y + expansion_m, extent_z + expansion_m)
    return (
        tuple(center[index] - extents[index] for index in range(3)),
        tuple(center[index] + extents[index] for index in range(3)),
    )


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
        primitives = semantic.get("collision_primitives") if isinstance(semantic, dict) else None
        if not isinstance(primitives, list) or not primitives:
            raise ValueError("semantic artifact has no collision primitives")
        if any(not isinstance(item, dict) for item in primitives):
            raise ValueError("collision primitive is not an object")
        self.graph = graph
        self.semantic_path = semantic_path
        self.semantic_sha256 = semantic_sha256
        self.vehicle_radius_m = vehicle_diameter_m / 2
        self.vehicle_half_height_m = vehicle_height_m / 2
        self.policy = policy or MetricPlannerPolicy()
        self.primitives: list[dict[str, Any]] = primitives
        self.nodes = {node.node_id: node for node in graph.nodes} if graph is not None else {}
        self._route_cache: dict[tuple[str, str, Point, Point], GraphRoute] = {}
        self._spatial: dict[GridKey, list[int]] = defaultdict(list)
        expansion = max(self.vehicle_radius_m, self.vehicle_half_height_m)
        expansion += self.policy.required_clearance_m + self.policy.resolution_m
        indexed_cells = 0
        for index, primitive in enumerate(self.primitives):
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
    #   对覆盖当前位置的粗筛图元计算完整机体净空；此静态模型不代表实时障碍感知。
    # 输入：
    #   self：当前规划器。
    #   point：机体碰撞包络中心坐标。
    # 输出：
    #   clearance_m：最小净空；局部没有静态候选时为正无穷。
    def clearance(self, point: Point) -> float:
        candidates = self._spatial.get(self._bin_key(point), ())
        if not candidates:
            return math.inf
        return min(
            vehicle_clearance(
                point,
                self.primitives[index],
                radius_m=self.vehicle_radius_m,
                half_height_m=self.vehicle_half_height_m,
            )
            for index in candidates
        )

    # 功能：
    #   以比导出验收更细的间隔检查整条连接线段，包含两端点且使用同一操作净空。
    # 输入：
    #   self：当前规划器。
    #   start、end：线段端点。
    # 输出：
    #   free：全部采样均满足净空时为真。
    def _segment_is_free(self, start: Point, end: Point) -> bool:
        distance = math.dist(start, end)
        # The exported route is independently revalidated at a 0.1 m interval.
        # Search and line-of-sight simplification must sample more densely than
        # that final gate; otherwise a narrow clearance minimum can fall between
        # planner samples and produce a route that the authoritative validator
        # immediately rejects.
        count = max(1, math.ceil(distance / min(0.05, self.policy.resolution_m / 4)))
        for index in range(count + 1):
            ratio = index / count
            point = tuple(start[axis] + (end[axis] - start[axis]) * ratio for axis in range(3))
            if self.clearance(point) < self.policy.required_clearance_m:
                return False
        return True

    # 功能：
    #   用命名拓扑路线限定搜索范围，同时包含当前实测起终点，不把无人机吸回旧图节点。
    # 输入：
    #   self：带拓扑图的规划器。
    #   query：命名起终点查询。
    #   exact_start、exact_goal：当前真实任务起终点。
    # 输出：
    #   bounds：本次有界搜索空间的下界和上界。
    def _search_bounds(
        self,
        query: RouteQuery,
        *,
        exact_start: Point,
        exact_goal: Point,
    ) -> tuple[Point, Point]:
        if self.graph is None:
            raise ValueError("topology graph is required for bounded global planning")
        baseline = shortest_route(
            self.graph,
            query.model_copy(update={"require_flight_verified_edges": False}),
        )
        points = [(point.x, point.y, point.z) for point in baseline.positions_m]
        # The graph supplies a broad search corridor, never an instruction to
        # snap a live vehicle or a revised semantic action pose back onto an
        # old route node.
        points.extend((exact_start, exact_goal))
        padding = self.policy.search_padding_m
        return (
            (
                min(point[0] for point in points) - padding,
                min(point[1] for point in points) - padding,
                max(self.vehicle_half_height_m + 0.05, min(point[2] for point in points) - padding),
            ),
            (
                max(point[0] for point in points) + padding,
                max(point[1] for point in points) + padding,
                max(point[2] for point in points) + padding,
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
        start_point = self.nodes[query.start_node].position_m
        goal_point = self.nodes[query.goal_node].position_m
        return self.plan_positions(
            query,
            start_position_m=start_point,
            goal_position_m=goal_point,
            prior_corridors=prior_corridors,
        )

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
        low, high = self._search_bounds(
            query,
            exact_start=exact_start,
            exact_goal=exact_goal,
        )
        resolution = self.policy.resolution_m

        # 功能：
        #   将本次局部网格键还原为世界米制坐标。
        # 输入：
        #   key：以本次搜索下界为原点的网格键。
        # 输出：
        #   position：世界三维位置。
        def key_to_point(key: GridKey) -> Point:
            return tuple(low[axis] + key[axis] * resolution for axis in range(3))  # type: ignore[return-value]

        # 功能：
        #   将世界位置量化到本次搜索网格；原始精确端点另行保留。
        # 输入：
        #   point：世界三维位置。
        # 输出：
        #   key：最近网格键。
        def point_to_key(point: Point) -> GridKey:
            return tuple(round((point[axis] - low[axis]) / resolution) for axis in range(3))  # type: ignore[return-value]

        maximum_key = tuple(math.floor((high[axis] - low[axis]) / resolution) for axis in range(3))
        start_key = point_to_key(exact_start)
        goal_key = point_to_key(exact_goal)
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

        if self.clearance(exact_start) < self.policy.required_clearance_m:
            raise ValueError("metric route start violates operational clearance")
        if self.clearance(exact_goal) < self.policy.required_clearance_m:
            raise ValueError("metric route goal violates operational clearance")
        if key_clearance(start_key) < self.policy.required_clearance_m:
            # The exact endpoint is authoritative; retain it as a special grid node.
            point_cache[start_key] = exact_start
            clearance_cache[start_key] = self.clearance(exact_start)
        if key_clearance(goal_key) < self.policy.required_clearance_m:
            point_cache[goal_key] = exact_goal
            clearance_cache[goal_key] = self.clearance(exact_goal)

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
            dx = (key[0] - goal_key[0]) * resolution
            dy = (key[1] - goal_key[1]) * resolution
            dz = (key[2] - goal_key[2]) * resolution
            return math.sqrt(dx * dx + dy * dy) + abs(dz) * self.policy.vertical_cost_multiplier

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
                if not self._segment_is_free(current_point, neighbor_point):
                    continue
                horizontal = resolution * math.hypot(dx, dy)
                vertical = resolution * abs(dz) * self.policy.vertical_cost_multiplier
                step_cost = horizontal + vertical
                if math.isfinite(clearance):
                    step_cost += self.policy.clearance_cost_weight / max(
                        clearance - self.policy.required_clearance_m + 0.05,
                        0.05,
                    )
                if neighbor in penalized:
                    step_cost += self.policy.corridor_penalty_weight
                candidate = cost + step_cost
                if candidate < costs.get(neighbor, math.inf):
                    costs[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
        if goal_key not in costs:
            raise ValueError("no metric route satisfies operational clearance")

        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(previous[keys[-1]])
        keys.reverse()
        raw_points = [exact_start, *(point(key) for key in keys[1:-1]), exact_goal]
        simplified = self._simplify(raw_points)
        digest = hashlib.sha256(
            json.dumps(simplified, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        node_ids = [
            query.start_node,
            *(f"metric-{digest}-{index:04d}" for index in range(1, len(simplified) - 1)),
            query.goal_node,
        ]
        edge_ids = [f"metric-edge-{digest}-{index:04d}" for index in range(len(simplified) - 1)]
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
            all_edges_flight_verified=route.all_edges_flight_verified,
        )
        return route.model_copy(deep=True)

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
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1:
                if self._segment_is_free(points[anchor], points[candidate]):
                    break
                candidate -= 1
            if not self._segment_is_free(points[anchor], points[candidate]):
                raise ValueError("metric route endpoint connection violates operational clearance")
            output.append(points[candidate])
            anchor = candidate
        return output


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
