"""Generic routing over the selected map; never generate a built-in flight demonstration."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable

from dronedream_plugin_sdk.protocol import encode_json

from .contracts import GraphRoute, MapAsset, MapEdge, RouteQuery


# 功能：
#   冻结并严格核验当前图和查询，构造后改写及字符串安全开关不能绕过规划边界。
# 输入：
#   graph：当前所选地图图结构。
#   query：起终点和资格约束。
# 输出：
#   inputs：独立的图和查询二元组。
def _validated_inputs(graph: MapAsset, query: RouteQuery) -> tuple[MapAsset, RouteQuery]:
    if not isinstance(graph, MapAsset) or not isinstance(query, RouteQuery):
        raise ValueError("ROUTE_INPUT_CONTRACT_INVALID")
    graph = MapAsset.model_validate_json(
        encode_json(graph.model_dump(mode="json"), limit=16 * 1024 * 1024, node_limit=1_000_000),
        strict=True,
    )
    query = RouteQuery.model_validate_json(encode_json(query.model_dump(mode="json")), strict=True)
    inputs = graph, query
    return inputs


# 功能：
#   在当前地图和资格过滤下求最短图路线，不生成实时杆量或证明路线无碰撞。
# 输入：
#   graph：当前地图拓扑。
#   query：起终点和边资格要求。
# 输出：
#   route：距离权重最小的图路线。
def shortest_route(graph: MapAsset, query: RouteQuery) -> GraphRoute:
    route = _weighted_route(graph, query, edge_cost=lambda edge: edge.distance_m)
    return route


# 功能：
#   对窄通道和未验证边增加代价，提供仍需碰撞检查的候选。
# 输入：
#   graph：当前地图拓扑。
#   query：起终点和边资格要求。
# 输出：
#   route：优先宽裕通道的路线。
def clearance_first_route(graph: MapAsset, query: RouteQuery) -> GraphRoute:
    # 功能：
    #   计算距离、窄通道和缺少飞行证据的综合代价。
    # 输入：
    #   edge：本次独立图快照中的边。
    # 输出：
    #   cost：非负规划代价。
    def edge_cost(edge: MapEdge) -> float:
        clearance_penalty = edge.distance_m * 0.65 / max(edge.minimum_clearance_m, 0.15)
        qualification_penalty = 0.0 if edge.qualification == "flight-verified" else 2.5
        cost = edge.distance_m + clearance_penalty + qualification_penalty
        return cost

    route = _weighted_route(graph, query, edge_cost=edge_cost)
    return route


# 功能：
#   用距离、高度变化及加减速代理代价筛选路线；这是规划偏好，不是已标定的电池能耗模型。
# 输入：
#   graph：当前地图拓扑。
#   query：起终点和边资格要求。
# 输出：
#   route：能耗代理代价较低的路线。
def energy_efficient_route(graph: MapAsset, query: RouteQuery) -> GraphRoute:
    graph, query = _validated_inputs(graph, query)
    nodes = {node.node_id: node for node in graph.nodes}

    # 功能：
    #   从同一图快照计算边的距离、高度变化与低速切换代价。
    # 输入：
    #   edge：本次路线候选边。
    # 输出：
    #   cost：有限的能耗代理代价。
    def edge_cost(edge: MapEdge) -> float:
        first = nodes[edge.from_node].position_m
        second = nodes[edge.to_node].position_m
        climb = abs(second.z - first.z)
        acceleration_proxy = 0.3 / max(edge.speed_limit_mps, 0.1)
        cost = edge.distance_m + climb * 2.5 + acceleration_proxy
        return cost

    route = _weighted_route(graph, query, edge_cost=edge_cost)
    return route


# 功能：
#   优先已验证、较少切换且限速适中的边，不把较低代价视为已经稳定飞行。
# 输入：
#   graph：当前地图拓扑。
#   query：起终点和边资格要求。
# 输出：
#   route：稳定性偏好路线。
def stability_first_route(graph: MapAsset, query: RouteQuery) -> GraphRoute:
    # 功能：
    #   汇总证据、切换和高速惩罚，保持 Dijkstra 所要求的非负代价。
    # 输入：
    #   edge：当前候选边。
    # 输出：
    #   cost：该边的稳定性代理代价。
    def edge_cost(edge: MapEdge) -> float:
        qualification_penalty = 0.0 if edge.qualification == "flight-verified" else 4.0
        transition_penalty = 0.75
        speed_penalty = max(0.0, edge.speed_limit_mps - 1.2) * 2.0
        cost = edge.distance_m + qualification_penalty + transition_penalty + speed_penalty
        return cost

    route = _weighted_route(graph, query, edge_cost=edge_cost)
    return route


# 功能：
#   用非负有限边权执行 Dijkstra，严格检查图结构并隔离返回坐标，不允许循环负权或无穷代价。
#   路线是宏观参考，实际运动必须经过实时控制、安全约束和飞控回执。
# 输入：
#   graph：本次所选图。
#   query：明确起终点与是否只用已验证边。
#   edge_cost：只根据当前边计算非负有限代价的回调。
# 输出：
#   route：有序节点、边、独立坐标和真实图路径长度；不可达时抛出异常。
def _weighted_route(
    graph: MapAsset, query: RouteQuery, *, edge_cost: Callable[[MapEdge], float]
) -> GraphRoute:
    graph, query = _validated_inputs(graph, query)
    known = {node.node_id: node for node in graph.nodes}
    if query.start_node not in known or query.goal_node not in known:
        raise ValueError("route endpoint is not present in graph")
    adjacency: dict[str, list[tuple[str, MapEdge, float]]] = {node_id: [] for node_id in known}
    for edge in graph.edges:
        if query.require_flight_verified_edges and (
            edge.qualification != "flight-verified" or edge.evidence_sha256 is None
        ):
            continue
        # 只计算一次代价；回调副本不能改写寻路使用的端点、距离或资格。
        raw_cost = edge_cost(edge.model_copy(deep=True))
        try:
            if type(raw_cost) not in (int, float) or not math.isfinite(raw_cost) or raw_cost < 0:
                raise ValueError("ROUTE_EDGE_COST_INVALID")
            weight = float(raw_cost)
        except OverflowError as error:
            raise ValueError("ROUTE_EDGE_COST_INVALID") from error
        adjacency[edge.from_node].append((edge.to_node, edge, weight))
        if edge.bidirectional:
            adjacency[edge.to_node].append((edge.from_node, edge, weight))

    queue: list[tuple[float, str]] = [(0.0, query.start_node)]
    costs = {query.start_node: 0.0}
    previous: dict[str, tuple[str, MapEdge]] = {}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == query.goal_node:
            break
        for neighbor, edge, weight in adjacency[current]:
            candidate = cost + weight
            if not math.isfinite(candidate):
                raise ValueError("ROUTE_PATH_COST_OVERFLOW")
            if candidate < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate
                previous[neighbor] = (current, edge)
                heapq.heappush(queue, (candidate, neighbor))
    if query.goal_node not in costs:
        raise ValueError("no route satisfies graph qualification policy")

    node_ids = [query.goal_node]
    used_edges: list[MapEdge] = []
    while node_ids[-1] != query.start_node:
        parent, edge = previous[node_ids[-1]]
        used_edges.append(edge)
        node_ids.append(parent)
    node_ids.reverse()
    used_edges.reverse()
    route = GraphRoute(
        start_node=query.start_node,
        goal_node=query.goal_node,
        node_ids=node_ids,
        edge_ids=[edge.edge_id for edge in used_edges],
        positions_m=[known[node_id].position_m for node_id in node_ids],
        route_length_m=sum(edge.distance_m for edge in used_edges),
        all_edges_flight_verified=bool(used_edges)
        and all(
            edge.qualification == "flight-verified" and edge.evidence_sha256 is not None
            for edge in used_edges
        ),
    )
    return route
