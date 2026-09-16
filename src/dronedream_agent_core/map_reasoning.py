"""Bounded structured map context for text-only model roles.

The model receives topology and semantic facts, never pixels or raw meshes as the
source of truth.  Geometry, clearance, and actuator authority remain deterministic.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from dronedream_plugin_sdk.protocol import encode_json

from .assets import MAX_SEMANTIC_BYTES, _load_object, _text, _text_list
from .contracts import GraphRoute, MapAsset, MapCatalog, RouteQuery
from .hashing import sha256_json
from .navigation import shortest_route

MAX_INCLUDED_NODES = 96
MAX_INCLUDED_EDGES = 192
MAX_FOCUS_NODES = 9
MAX_ROUTE_ATTEMPTS = 128


# 功能：
#   仅摘要与当前目录散列相同的语义字节，缺失字段保持未知，不把字符串 false 当成真。
# 输入：
#   path：当前语义文件路径。
#   expected_sha256：目录绑定的原始文件散列。
# 输出：
#   summary：已验证的环境摘要，读取或字段异常时只返回不可用状态与原因。
def _semantic_summary(path: Path, expected_sha256: str) -> dict[str, object]:
    try:
        value, raw = _load_object(path)
    except (OSError, ValueError):
        return {"available": False, "issue_codes": ["MAP_SEMANTIC_CONTEXT_UNAVAILABLE"]}
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return {"available": False, "issue_codes": ["MAP_SEMANTIC_CONTEXT_HASH_MISMATCH"]}
    try:
        if (
            value.get("schema_version") != "dronedream.map-semantic.v1"
            or value.get("coordinate_frame") != "ENU"
        ):
            raise ValueError("schema")
        limits = _text_list(
            value.get("known_export_limits", value.get("known_limits", [])),
            field="LIMITS",
            maximum=64,
        )
        dynamic_people = value.get("dynamic_people", {})
        navigation_layers = value.get("navigation_layers", {})
        if not isinstance(dynamic_people, dict) or not isinstance(navigation_layers, dict):
            raise ValueError("layers")
        for fields, keys in (
            (dynamic_people, ("runtime_spawn_required", "static_collision_present")),
            (navigation_layers, ("occupancy_ready", "esdf_ready")),
        ):
            if any(key in fields and type(fields[key]) is not bool for key in keys):
                raise ValueError("boolean")
        static_primitives = value.get("collision_primitives", [])
        runtime_primitives = value.get("runtime_collision_primitives", [])
        for primitives in (static_primitives, runtime_primitives):
            if (
                not isinstance(primitives, list)
                or len(primitives) > 100_000
                or any(not isinstance(item, dict) for item in primitives)
            ):
                raise ValueError("primitives")
        # 显式空运行碰撞列表表示空，只有字段不存在时才采用静态库存，不能混淆两种情况。
        active_primitives = (
            runtime_primitives if "runtime_collision_primitives" in value else static_primitives
        )
        obstacle_semantics = Counter(
            _text(item.get("semantic", "unspecified"), field="KIND", maximum=96)
            for item in active_primitives
        )
        geometry_scope = value.get("geometry_scope")
        if geometry_scope is not None:
            geometry_scope = _text(geometry_scope, field="GEOMETRY_SCOPE", maximum=240)
    except ValueError:
        return {"available": False, "issue_codes": ["MAP_SEMANTIC_CONTEXT_INVALID"]}
    summary = {
        "available": True,
        "semantic_sha256": expected_sha256,
        "coordinate_frame": value.get("coordinate_frame"),
        "geometry_scope": geometry_scope,
        "collision_primitive_count": len(static_primitives),
        "runtime_collision_primitive_count": len(runtime_primitives),
        "occupancy_ready": navigation_layers.get("occupancy_ready"),
        "esdf_ready": navigation_layers.get("esdf_ready"),
        "dynamic_obstacles_runtime_required": dynamic_people.get("runtime_spawn_required"),
        "dynamic_obstacles_in_static_collision": dynamic_people.get("static_collision_present"),
        "collision_semantic_inventory": [
            {"semantic": semantic, "primitive_count": count}
            for semantic, count in sorted(obstacle_semantics.items())[:32]
        ],
        "known_limits": limits[:16],
        "inventory_truncated": len(obstacle_semantics) > 32,
        "known_limits_truncated": len(limits) > 16,
    }
    return summary


# 功能：
#   摘要整条候选路线的瓶颈与资格，长路线只展示前缀但散列和统计覆盖完整路线。
# 输入：
#   graph：产生该候选的本次图快照。
#   route：寻路器返回的完整候选。
# 输出：
#   summary：模型参考摘要；零移动没有通道宽度，不伪造安全余量。
def _route_summary(graph: MapAsset, route: GraphRoute) -> dict[str, object]:
    edges = {edge.edge_id: edge for edge in graph.edges}
    nodes = {node.node_id: node for node in graph.nodes}
    route_edges = [edges[edge_id] for edge_id in route.edge_ids]
    bottleneck = min(route_edges, key=lambda edge: edge.minimum_clearance_m, default=None)
    unverified = [
        edge.edge_id
        for edge in route_edges
        if edge.qualification != "flight-verified" or edge.evidence_sha256 is None
    ]
    summary = {
        "node_ids": route.node_ids[:MAX_INCLUDED_NODES],
        "edge_ids": route.edge_ids[: MAX_INCLUDED_NODES - 1],
        "complete_route_node_count": len(route.node_ids),
        "complete_route_edge_count": len(route.edge_ids),
        "route_details_truncated": len(route.node_ids) > MAX_INCLUDED_NODES,
        "route_length_m": round(route.route_length_m, 3),
        "minimum_clearance_m": round(bottleneck.minimum_clearance_m, 3) if bottleneck else None,
        "bottleneck_edge_id": bottleneck.edge_id if bottleneck else None,
        "minimum_speed_limit_mps": round(min(edge.speed_limit_mps for edge in route_edges), 3)
        if route_edges
        else None,
        "semantic_sequence": [
            nodes[node_id].semantic for node_id in route.node_ids[:MAX_INCLUDED_NODES]
        ],
        "all_edges_flight_verified": route.all_edges_flight_verified,
        "unverified_edge_ids": unverified[:MAX_INCLUDED_EDGES],
        "unverified_edge_count": len(unverified),
        "route_sha256": sha256_json(route),
    }
    return summary


# 功能：
#   从最短路逐边移除生成有界备选；候选可以共享其他边，不声称完整枚举或边不相交。
# 输入：
#   graph：当前已校验图。
#   start_node：起点标识。
#   goal_node：终点标识。
#   limit：最多一至五条候选。
# 输出：
#   alternatives：按资格、长度、净空排序的候选摘要。
def _route_alternatives(
    graph: MapAsset, *, start_node: str, goal_node: str, limit: int = 5
) -> list[dict[str, object]]:
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("MAP_ROUTE_ALTERNATIVE_LIMIT_INVALID")
    first = shortest_route(graph, RouteQuery(start_node=start_node, goal_node=goal_node))
    discovered: dict[tuple[str, ...], GraphRoute] = {tuple(first.edge_ids): first}
    frontier = [first]
    attempts = 0
    while frontier and len(discovered) < limit and attempts < MAX_ROUTE_ATTEMPTS:
        basis = frontier.pop(0)
        for blocked_edge_id in basis.edge_ids:
            if attempts >= MAX_ROUTE_ATTEMPTS:
                break
            attempts += 1
            reduced = graph.model_copy(
                update={"edges": [edge for edge in graph.edges if edge.edge_id != blocked_edge_id]}
            )
            try:
                candidate = shortest_route(
                    reduced,
                    RouteQuery(start_node=start_node, goal_node=goal_node),
                )
            except ValueError:
                continue
            key = tuple(candidate.edge_ids)
            if key in discovered:
                continue
            discovered[key] = candidate
            frontier.append(candidate)
            if len(discovered) >= limit:
                break
    clearances = {edge.edge_id: edge.minimum_clearance_m for edge in graph.edges}
    ordered = sorted(
        discovered.values(),
        key=lambda route: (
            not route.all_edges_flight_verified,
            route.route_length_m,
            -min((clearances[edge_id] for edge_id in route.edge_ids), default=0.0),
        ),
    )
    alternatives = [_route_summary(graph, route) for route in ordered[:limit]]
    return alternatives


# 功能：
#   为云端规划角色构造有界、散列绑定的当前地图视图；不授予局部控制或碰撞认证权限。
#   保留往返访问顺序，优先展示任务节点，超出上下文预算明确披露截断。
# 输入：
#   graph：所选地图拓扑。
#   catalog：同一资产选择的语义目录。
#   semantic_path：目录散列对应的语义文件。
#   focus_nodes：按任务顺序访问的节点，允许返程重复。
# 输出：
#   context：云端模型可读的路线和环境摘要；它不是实时传感器观测。
def build_map_reasoning_context(
    graph: MapAsset,
    catalog: MapCatalog,
    semantic_path: Path,
    *,
    focus_nodes: list[str] | None = None,
) -> dict[str, object]:
    graph = MapAsset.model_validate_json(
        encode_json(graph.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000),
        strict=True,
    )
    catalog = MapCatalog.model_validate_json(
        encode_json(
            catalog.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000
        ),
        strict=True,
    )
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    if focus_nodes is not None and (
        not isinstance(focus_nodes, list)
        or len(focus_nodes) > 1024
        or any(not isinstance(node, str) or node not in nodes_by_id for node in focus_nodes)
    ):
        raise ValueError("MAP_FOCUS_NODES_INVALID")
    ordered_focus = [
        node
        for index, node in enumerate(focus_nodes or [])
        if index == 0 or node != focus_nodes[index - 1]
    ]
    focus = ordered_focus[:MAX_FOCUS_NODES]
    selected = set(focus)
    selected.update(node for node in graph.named_entities.values() if node in nodes_by_id)
    focus_routes: list[dict[str, object]] = []
    for start, goal in zip(focus, focus[1:], strict=False):
        if start == goal:
            continue
        try:
            alternatives = _route_alternatives(
                graph,
                start_node=start,
                goal_node=goal,
            )
        except ValueError:
            focus_routes.append(
                {
                    "start_node": start,
                    "goal_node": goal,
                    "issue_codes": ["MAP_TOPOLOGY_DISCONNECTED"],
                }
            )
            continue
        selected.update(
            node_id for alternative in alternatives for node_id in alternative["node_ids"]
        )
        preferred = alternatives[0]
        focus_routes.append(
            {
                "start_node": start,
                "goal_node": goal,
                **preferred,
                "candidate_route_count": len(alternatives),
                "candidate_routes": alternatives,
                "selection_policy": (
                    "model may rank candidates by task risk; deterministic clearance "
                    "validation remains mandatory"
                ),
            }
        )

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.from_node].add(edge.to_node)
        if edge.bidirectional:
            adjacency[edge.to_node].add(edge.from_node)
    for node_id in list(selected):
        selected.update(adjacency[node_id])
    if not selected:
        selected.update(node.node_id for node in graph.nodes[:MAX_INCLUDED_NODES])
    selected_order = list(
        dict.fromkeys([*focus, *(node.node_id for node in graph.nodes if node.node_id in selected)])
    )
    selected = set(selected_order[:MAX_INCLUDED_NODES])
    selected_edges = [
        edge for edge in graph.edges if edge.from_node in selected and edge.to_node in selected
    ][:MAX_INCLUDED_EDGES]
    x = [node.position_m.x for node in graph.nodes]
    y = [node.position_m.y for node in graph.nodes]
    z = [node.position_m.z for node in graph.nodes]

    named = sorted(graph.named_entities.items(), key=lambda item: (item[1] not in focus, item[0]))[
        :MAX_INCLUDED_NODES
    ]
    context = {
        "schema_version": "dronedream.model-map-context.v1",
        "source_of_truth": "qualified-structured-map-not-rendered-image",
        "visual_input_required": False,
        "graph_sha256": sha256_json(graph),
        "semantic_sha256": catalog.semantic_sha256,
        "coordinate_frame": graph.coordinate_frame,
        "bounds_m": {
            "minimum": {"x": min(x), "y": min(y), "z": min(z)},
            "maximum": {"x": max(x), "y": max(y), "z": max(z)},
        },
        "topology": {
            "complete_graph_node_count": len(graph.nodes),
            "complete_graph_edge_count": len(graph.edges),
            "included_node_count": len(selected),
            "included_edge_count": len(selected_edges),
            "complete_named_entity_count": len(graph.named_entities),
            "included_named_entity_count": len(named),
            "complete_focus_node_count": len(ordered_focus),
            "focus_nodes_truncated": len(ordered_focus) > len(focus),
            "truncated_for_model_context": (
                len(selected) < len(graph.nodes)
                or len(selected_edges) < len(graph.edges)
                or len(named) < len(graph.named_entities)
                or len(ordered_focus) > len(focus)
                or any(route.get("route_details_truncated", False) for route in focus_routes)
            ),
            "focus_nodes": focus,
            "focus_routes": focus_routes,
        },
        "named_entities": [
            {
                "entity_id": entity_id,
                "node_id": node_id,
                "position_m": nodes_by_id[node_id].position_m.model_dump(mode="json"),
                "semantic": nodes_by_id[node_id].semantic,
            }
            for entity_id, node_id in named
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "position_m": node.position_m.model_dump(mode="json"),
                "semantic": node.semantic,
            }
            for node in graph.nodes
            if node.node_id in selected
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "distance_m": round(edge.distance_m, 3),
                "minimum_clearance_m": round(edge.minimum_clearance_m, 3),
                "speed_limit_mps": round(edge.speed_limit_mps, 3),
                "bidirectional": edge.bidirectional,
                "qualification": edge.qualification,
            }
            for edge in selected_edges
        ],
        "semantic_environment": _semantic_summary(semantic_path, catalog.semantic_sha256),
        "catalog_known_limits": catalog.known_limits[:16],
        "model_authority": {
            "may_reason_about": [
                "task-order",
                "route-objective-priority",
                "risk-flags",
                "replan-request",
            ],
            "may_not_author": [
                "collision-clearance-facts",
                "unlisted-coordinates",
                "actuator-commands",
            ],
        },
    }
    # 图契约中的标识字符串可能很长，条目数限制之外再设最终字节上限，失败不截断标识。
    encode_json(context, limit=2 * 1024 * 1024, node_limit=100_000)
    return context
