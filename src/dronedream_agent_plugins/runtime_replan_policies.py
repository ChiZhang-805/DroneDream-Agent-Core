"""Select graph rejoin candidates; core collision/hold gates authorize the actual join."""

from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.contracts import MapAsset, RouteQuery, Vector3
from dronedream_agent_core.navigation import shortest_route
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin, policy_boolean, policy_number


# 功能：
#   重新校验可变位置与地图，并限制接入半径为正且不超过二十米。
# 输入：
#   current_world：当前世界坐标系位置。
#   graph：本次任务地图。
#   configuration：在线换路策略配置。
#   default：未设置接入半径时的策略默认值，单位米。
# 输出：
#   inputs：校验后的位置、地图和接入距离上限组成的三元组。
def _anchor_inputs(
    current_world: Vector3, graph: MapAsset, configuration: dict[str, Any], default: float
) -> tuple[Vector3, MapAsset, float]:
    current_world = Vector3.model_validate(current_world.model_dump(mode="python"))
    graph = MapAsset.model_validate(graph.model_dump(mode="python"))
    maximum = policy_number(configuration, "maximum_join_distance_m", default, 0.0, 20.0)
    if maximum <= 0.0:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    inputs = (current_world, graph, maximum)
    return inputs


# 功能：
#   计算本地 ENU 坐标下的三维直线距离，不把直线距离当作无碰撞接入路径证明。
# 输入：
#   node：候选图节点。
#   current_world：当前世界坐标位置。
# 输出：
#   distance_m：当前位置至节点的直线距离，单位米。
def _join_distance(node: Any, current_world: Vector3) -> float:
    distance_m = math.dist(
        (node.position_m.x, node.position_m.y, node.position_m.z),
        (current_world.x, current_world.y, current_world.z),
    )
    return distance_m


# 功能：
#   在接入半径内推荐最近节点，距离并列按节点标识选择，实际接入仍需核心安全门控。
# 输入：
#   current_world：当前世界坐标位置。
#   graph：任务地图。
#   configuration：接入半径配置。
#   _：该策略不使用的任务上下文。
# 输出：
#   selection：候选锚点、接入半径和历史边验证要求。
def _nearest_anchor(
    *, current_world: Vector3, graph: MapAsset, configuration: dict[str, Any], **_: Any
) -> dict[str, object]:
    current_world, graph, maximum = _anchor_inputs(current_world, graph, configuration, 6.0)
    anchor = min(
        graph.nodes,
        key=lambda node: (_join_distance(node, current_world), node.node_id),
    )
    if _join_distance(anchor, current_world) > maximum:
        raise ValueError("RUNTIME_REPLAN_ANCHOR_OUTSIDE_JOIN_RADIUS")
    selection = {
        "anchor_node": anchor.node_id,
        "maximum_join_distance_m": maximum,
        "requires_flight_verified_anchor": False,
    }
    return selection


# 功能：
#   仅在连接过历史验证边的节点中选择近邻；该条件不代表接入段或后续整条路线已验证。
# 输入：
#   current_world：当前世界坐标位置。
#   graph：含边验收标记的任务地图。
#   configuration：最大接入半径配置。
#   _：该策略不使用的任务上下文。
# 输出：
#   selection：满足半径及关联验证边条件的候选锚点配置。
def _verified_anchor(
    *, current_world: Vector3, graph: MapAsset, configuration: dict[str, Any], **_: Any
) -> dict[str, object]:
    current_world, graph, maximum = _anchor_inputs(current_world, graph, configuration, 4.0)
    verified_nodes = {
        node_id
        for edge in graph.edges
        if edge.qualification == "flight-verified"
        for node_id in (edge.from_node, edge.to_node)
    }
    candidates = [node for node in graph.nodes if node.node_id in verified_nodes]
    if not candidates:
        raise ValueError("RUNTIME_REPLAN_VERIFIED_ANCHOR_UNAVAILABLE")
    anchor = min(
        candidates,
        key=lambda node: (_join_distance(node, current_world), node.node_id),
    )
    if _join_distance(anchor, current_world) > maximum:
        raise ValueError("RUNTIME_REPLAN_ANCHOR_OUTSIDE_JOIN_RADIUS")
    selection = {
        "anchor_node": anchor.node_id,
        "maximum_join_distance_m": maximum,
        "requires_flight_verified_anchor": True,
    }
    return selection


# 功能：
#   1. 先验证目标到返程节点的有向路径，再排除无法接续目标的近邻死端。
#   2. 联合接入距离及剩余路线长度评分，推荐可继续完成任务的锚点，不直接命令飞行。
# 输入：
#   current_world：当前世界坐标位置。
#   graph：本任务拓扑地图。
#   target_node：当前尚需到达的任务目标。
#   return_node：合同指定的返程节点。
#   configuration：接入半径、目标距离权重及历史验证边要求。
#   _：该策略不使用的其他运行上下文。
# 输出：
#   selection：候选锚点、距离、评分和拓扑可达性结果。
def _mission_continuity_anchor(
    *,
    current_world: Vector3,
    graph: MapAsset,
    target_node: str,
    return_node: str,
    configuration: dict[str, Any],
    **_: Any,
) -> dict[str, object]:
    current_world, graph, maximum = _anchor_inputs(current_world, graph, configuration, 8.0)
    route_weight = policy_number(configuration, "target_route_weight", 0.35, 0.0, 2.0)
    require_verified = policy_boolean(configuration, "require_flight_verified_edges", False)
    # The return leg is invariant across candidates. Check it once, preserving
    # directed-edge semantics instead of repeating an identical graph search.
    try:
        shortest_route(
            graph,
            RouteQuery(
                start_node=target_node,
                goal_node=return_node,
                require_flight_verified_edges=require_verified,
            ),
        )
    except ValueError as error:
        raise ValueError("RUNTIME_REPLAN_CONTINUITY_ANCHOR_UNAVAILABLE") from error
    ranked: list[tuple[float, float, str, float]] = []
    for node in graph.nodes:
        join_distance = _join_distance(node, current_world)
        if join_distance > maximum:
            continue
        try:
            outbound = shortest_route(
                graph,
                RouteQuery(
                    start_node=node.node_id,
                    goal_node=target_node,
                    require_flight_verified_edges=require_verified,
                ),
            )
        except ValueError:
            continue
        score = join_distance + outbound.route_length_m * route_weight
        ranked.append((score, join_distance, node.node_id, outbound.route_length_m))
    if not ranked:
        raise ValueError("RUNTIME_REPLAN_CONTINUITY_ANCHOR_UNAVAILABLE")
    score, join_distance, anchor_node, target_distance = min(ranked)
    selection = {
        "anchor_node": anchor_node,
        "maximum_join_distance_m": maximum,
        "requires_flight_verified_anchor": require_verified,
        "join_distance_m": join_distance,
        "target_route_distance_m": target_distance,
        "continuity_score": score,
        "target_and_return_reachable": True,
    }
    return selection


# 功能：
#   注册仅在安全悬停时允许切换的锚点策略，声明最小读取权限及失败关闭规则。
# 输入：
#   无。
# 输出：
#   definitions：近邻、历史验证及任务连续性锚点策略定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "runtime.replan-nearest-anchor",
            "最近安全锚点换路",
            "从稳定悬停位置接入最近图节点，并对接入距离设置硬上限。",
            _nearest_anchor,
            6.0,
            True,
            {},
        ),
        (
            "runtime.replan-verified-anchor",
            "已验证锚点换路",
            "只允许接入至少连接一条飞行验证边的图节点，适合高风险任务。",
            _verified_anchor,
            4.0,
            False,
            {},
        ),
        (
            "runtime.replan-mission-continuity",
            "任务连续性锚点换路",
            "排除无法继续到目标并返航的死端节点，再联合接入距离与剩余目标路径选择锚点。",
            _mission_continuity_anchor,
            8.0,
            False,
            {
                "target_route_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.35,
                },
                "require_flight_verified_edges": {
                    "type": "boolean",
                    "default": False,
                },
            },
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.select",
            capability_kind="runtime-replanner",
            capability_name=name,
            capability_description=description,
            category_id="runtime",
            category_label="运行期与在线换路",
            slot_id="runtime.replan-policy",
            slot_label="在线换路锚点策略",
            activation_mode="single",
            category_order=70,
            slot_order=25,
            plugin_order=index * 10,
            hooks={"select_anchor": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
            swap_policy="safe-hold",
            configuration_schema={
                "type": "object",
                "properties": {
                    "maximum_join_distance_m": {
                        "type": "number",
                        "exclusiveMinimum": 0.0,
                        "maximum": 20.0,
                        "default": default_maximum,
                    },
                    **extra_properties,
                },
                "additionalProperties": False,
            },
            permissions=["mission.read", "telemetry.read", "configuration.read"],
            metadata={
                "differentiation": (
                    "mission-reachability-and-join-cost"
                    if plugin_id == "runtime.replan-mission-continuity"
                    else "join-proximity"
                )
            },
        )
        for index, (
            plugin_id,
            name,
            description,
            handler,
            default_maximum,
            enabled,
            extra_properties,
        ) in enumerate(values, start=1)
    ]
    return definitions
