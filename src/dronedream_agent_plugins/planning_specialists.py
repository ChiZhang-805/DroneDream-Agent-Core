"""Layer-specific planning checks, not learned experts or live flight-control authorization."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from dronedream_agent_core.contracts import (
    GraphRoute,
    MapAsset,
    MissionContract,
    PlannerContribution,
    PlannerValidation,
    Px4Track,
    RouteClearanceReport,
    TaskGraph,
    VehicleAsset,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import (
    clearance_evidence_matches_route,
    hook_plugin,
    policy_number,
    within_qualified_range,
)

_LAYERS = (
    "semantic",
    "temporal",
    "global",
    "local",
    "indoor",
    "outdoor",
    "dynamic-obstacle",
    "energy",
    "link",
    "payload",
    "regulatory",
)


# 功能：
#   复用共享的距离边界检查，只容忍微小序列化误差，不扩大已验收航程。
# 输入：
#   distance_m：计划需要的距离，单位米。
#   qualified_range_m：验收允许的上限，单位米。
# 输出：
#   accepted：距离满足已验收范围时为 True。
def _within_qualified_range(distance_m: float, qualified_range_m: float) -> bool:
    accepted = within_qualified_range(distance_m, qualified_range_m)
    return accepted


# 功能：
#   按边方向进行广度优先可达检查，未知端点及悬空边不得因为名称相同而被接受。
# 输入：
#   graph：本任务的拓扑地图。
#   start：出发节点标识。
#   goal：目的节点标识。
# 输出：
#   reachable：有效拓扑中存在符合边方向的连接时为 True。
def _reachable(graph: MapAsset, start: str, goal: str) -> bool:
    reachable = False
    neighbours: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    if start not in neighbours or goal not in neighbours:
        return reachable
    for edge in graph.edges:
        if edge.from_node not in neighbours or edge.to_node not in neighbours:
            return reachable
        neighbours[edge.from_node].add(edge.to_node)
        if edge.bidirectional:
            neighbours[edge.to_node].add(edge.from_node)
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == goal:
            reachable = True
            return reachable
        for neighbour in neighbours[node] - seen:
            seen.add(neighbour)
            queue.append(neighbour)
    return reachable


# 功能：
#   根据任务与资产判定当前规划层是否适用，列出约束和指标，不冒充物理验证已经完成。
# 输入：
#   layer：已声明的规划层标识。
#   contract：冻结的任务合同。
#   map_graph：任务地图。
#   vehicle：飞行器资产。
#   _：当前贡献计算不使用的扩展上下文。
# 输出：
#   contribution：当前层要求、输入依赖和可直接检查的基础门控。
def _contribution(
    layer: str,
    *,
    contract: MissionContract,
    map_graph: MapAsset,
    vehicle: VehicleAsset,
    **_: Any,
) -> PlannerContribution:
    contract = MissionContract.model_validate(contract.model_dump(mode="python"), strict=True)
    map_graph = MapAsset.model_validate(map_graph.model_dump(mode="python"), strict=True)
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    semantics = {node.semantic for node in map_graph.nodes}
    normalized_constraints = " ".join(contract.constraints).casefold()
    applicable = {
        "semantic": True,
        "temporal": any(
            token in normalized_constraints for token in ("deadline", "time", "时间", "之前")
        ),
        "global": True,
        "local": True,
        "indoor": bool(semantics & {"corridor", "stairs", "door", "office"}),
        "outdoor": "outdoor" in semantics,
        "dynamic-obstacle": True,
        "energy": True,
        "link": (
            "outdoor" in semantics and bool(semantics & {"corridor", "stairs", "door", "office"})
        ),
        "payload": contract.payload_action == "pickup",
        "regulatory": any(
            token in normalized_constraints
            for token in ("regulatory", "airspace", "法规", "空域", "禁飞")
        ),
    }[layer]
    constraints = {
        "semantic": ["all targets resolve to immutable map node identifiers"],
        "temporal": [
            "route duration must be finite and holds bounded; deadlines need separate evidence"
        ],
        "global": ["start, target, and return remain connected in the selected map graph"],
        "local": ["every continuous trajectory segment passes collision-envelope clearance"],
        "indoor": ["door, stair, and corridor transitions respect vehicle envelope and speed"],
        "outdoor": ["outdoor legs retain a reachable return or safe-landing path"],
        "dynamic-obstacle": ["online obstacles can only tighten motion or trigger hold/replan"],
        "energy": ["distance, climb, payload, and reserve remain inside qualified envelope"],
        "link": ["indoor/outdoor transitions retain heartbeat and lost-link behavior"],
        "payload": ["pickup identity, attachment, mass update, and custody evidence are required"],
        "regulatory": ["real-flight airspace constraints require explicit regulatory evidence"],
    }[layer]
    metrics = {
        "semantic": ["grounded-target-ratio"],
        "temporal": ["estimated-duration-seconds", "deadline-slack-seconds"],
        "global": ["route-length-m", "graph-edge-count"],
        "local": ["minimum-clearance-m", "turn-speed-mps"],
        "indoor": ["door-transition-count", "stair-transition-count"],
        "outdoor": ["outdoor-distance-m", "landing-option-count"],
        "dynamic-obstacle": ["replan-latency-ms", "minimum-time-to-collision-s"],
        "energy": ["energy-proxy", "reserve-fraction"],
        "link": ["link-margin-db", "heartbeat-gap-ms"],
        "payload": ["payload-mass-kg", "custody-evidence-count"],
        "regulatory": ["regulated-zone-intersection-count"],
    }[layer]
    gates = {
        "contract_nodes_exist": {
            contract.start_node,
            contract.target_node,
            contract.return_node,
        }.issubset({node.node_id for node in map_graph.nodes}),
        "vehicle_envelope_positive": vehicle.body_radius_m > 0 and vehicle.body_height_m > 0,
    }
    contribution = PlannerContribution(
        planner_id=f"planning.{layer}",
        layer=layer,  # type: ignore[arg-type]
        applicable=applicable,
        hard_constraints=constraints if applicable else [],
        objective_metrics=metrics if applicable else [],
        required_inputs=["mission-contract", "map-graph", "vehicle-asset"],
        deterministic_gates=gates,
    )
    return contribution


# 功能：
#   1. 重新校验所有输入模型，按指定规划层检查已产生的任务制品。
#   2. 净空接受必须与实际路线及报告内容一致；能耗配置只可收紧已验收范围。
#   3. 有限估计时长、文本规则及仿真范围声明均不代表实时能力或真实空域授权。
# 输入：
#   layer：当前校验的规划层标识。
#   contract：冻结任务目标、约束和安全规则的合同。
#   map_graph：选定地图。
#   vehicle：机体性能与验收包络。
#   task_graph：任务动作依赖图。
#   route：待检查的实际路线。
#   clearance：该路线的净空报告。
#   px4_track：准备输出的飞行轨迹。
#   configuration：可选的规划层限制配置。
#   _：该层不使用的其他任务制品。
# 输出：
#   validation：本层各项门控、接受状态及失败原因。
def _validation(
    layer: str,
    *,
    contract: MissionContract,
    map_graph: MapAsset,
    vehicle: VehicleAsset,
    task_graph: TaskGraph,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    px4_track: Px4Track,
    configuration: dict[str, object] | None = None,
    **_: Any,
) -> PlannerValidation:
    contract = MissionContract.model_validate(contract.model_dump(mode="python"), strict=True)
    map_graph = MapAsset.model_validate(map_graph.model_dump(mode="python"), strict=True)
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    task_graph = TaskGraph.model_validate(task_graph.model_dump(mode="python"), strict=True)
    route = GraphRoute.model_validate(route.model_dump(mode="python"), strict=True)
    clearance = RouteClearanceReport.model_validate(
        clearance.model_dump(mode="python"), strict=True
    )
    px4_track = Px4Track.model_validate(px4_track.model_dump(mode="python"), strict=True)
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict) or set(configured) - {"qualified_range_m"}:
        raise ValueError("PLANNER_CONFIGURATION_INVALID")
    route_node_ids = set(route.node_ids)
    route_semantics = {node.semantic for node in map_graph.nodes if node.node_id in route_node_ids}
    requested_range_m = (
        policy_number(configured, "qualified_range_m", 10, 10, 100000)
        if "qualified_range_m" in configured
        else vehicle.qualified_range_m
    )
    # A plugin may tighten the certified envelope, never expand the vehicle asset.
    maximum_range_m = min(requested_range_m, vehicle.qualified_range_m)
    normalized_constraints = " ".join(contract.constraints).casefold()
    # 多个层复用同一报告，必须复用同一内容检查，不能各自只信 accepted 标记。
    clearance_valid = clearance_evidence_matches_route(clearance, route)
    gates: dict[str, bool]
    if layer == "semantic":
        gates = {
            "start_bound": route.start_node == contract.start_node,
            "return_bound": route.goal_node == contract.return_node,
            "target_present": contract.target_node in route.node_ids,
        }
    elif layer == "temporal":
        # A finite lower-bound travel time is not a guarantee of a user deadline:
        # acceleration, queued actions, obstacles and model latency add time.
        gates = {
            "route_duration_finite": math.isfinite(
                route.route_length_m / max(vehicle.max_speed_mps, 0.1)
            ),
            "bounded_waypoint_holds": px4_track.waypoint_hold_seconds <= 30.0,
        }
    elif layer == "global":
        gates = {
            "start_to_target_reachable": _reachable(
                map_graph, contract.start_node, contract.target_node
            ),
            "target_to_return_reachable": _reachable(
                map_graph, contract.target_node, contract.return_node
            ),
            "route_nonempty": bool(route.edge_ids),
        }
    elif layer == "local":
        gates = {
            "clearance_accepted": clearance.accepted,
            "clearance_bound_to_route": clearance.route_sha256 == sha256_json(route),
            "clearance_report_consistent": clearance_valid,
        }
    elif layer == "indoor":
        gates = {
            "indoor_transition_clear": (
                not route_semantics.intersection({"door", "stairs", "corridor"}) or clearance_valid
            ),
            "indoor_speed_bounded": all(
                point.speed_limit_mps <= min(vehicle.max_speed_mps, 2.0)
                for point in px4_track.points
            ),
        }
    elif layer == "outdoor":
        gates = {
            "return_path_present": contract.return_node in route.node_ids,
            "outdoor_clearance": "outdoor" not in route_semantics or clearance_valid,
        }
    elif layer == "dynamic-obstacle":
        gates = {
            "static_envelope_clear": clearance_valid,
            "runtime_hold_rule_frozen": any(
                "hold" in rule.casefold() or "悬停" in rule
                for rule in contract.immutable_safety_rules
            ),
        }
    elif layer == "energy":
        gates = {
            "qualified_range": _within_qualified_range(
                route.route_length_m,
                maximum_range_m,
            ),
            "return_reserve_declared": vehicle.reserve_battery_percent >= 10.0,
        }
    elif layer == "link":
        gates = {
            "lost_link_rule_frozen": any(
                token in " ".join(contract.immutable_safety_rules).casefold()
                for token in ("link", "abort", "hold", "返航", "悬停")
            )
        }
    elif layer == "payload":
        pickups = [node for node in task_graph.nodes if node.action == "pickup"]
        pickup_required = contract.payload_action == "pickup"
        gates = {
            "pickup_action_present": len(pickups) == (1 if pickup_required else 0),
            "pickup_target_bound": all(
                node.target_node == contract.target_node for node in pickups
            ),
            "payload_capacity_positive": (not pickup_required or vehicle.max_pickup_payload_kg > 0),
        }
    elif layer == "regulatory":
        # This declaration check cannot certify airspace authorization. Real
        # flight additionally requires the core's independent admission evidence.
        regulated = any(
            token in normalized_constraints
            for token in ("regulatory", "airspace", "法规", "空域", "禁飞")
        )
        simulation_scope = any(
            token in normalized_constraints for token in ("simulation", "sim-only", "仿真")
        )
        gates = {"regulatory_evidence_or_simulation_scope": not regulated or simulation_scope}
    else:
        raise ValueError("PLANNER_LAYER_UNKNOWN")
    failed = [name for name, accepted in gates.items() if not accepted]
    validation = PlannerValidation(
        planner_id=f"planning.{layer}",
        accepted=not failed,
        deterministic_gates=gates,
        issue_codes=[
            f"PLANNER_{layer.upper().replace('-', '_')}_{name.upper()}" for name in failed
        ],
    )
    return validation


# 功能：
#   为单个规划层绑定贡献与校验回调，校验失败关闭，不授予执行器控制权限。
# 输入：
#   layer：本定义所绑定的规划层标识。
#   order：在规划插槽中的装配顺序。
# 输出：
#   definition：当前规划层插件定义。
def _definition(layer: str, order: int) -> PluginDefinition:
    display = layer.replace("-", " ").title()

    # 功能：
    #   将当前任务及资产传入已冻结层级的约束贡献计算。
    # 输入：
    #   kwargs：合同、地图、机体及扩展上下文。
    # 输出：
    #   contribution：当前层贡献的规划要求。
    def contribute(**kwargs: Any) -> PlannerContribution:
        contribution = _contribution(layer, **kwargs)
        return contribution

    # 功能：
    #   对准备阶段提供的制品执行已绑定层级的门控检查。
    # 输入：
    #   kwargs：本次合同、资产、路线、任务图、净空及轨迹等制品。
    # 输出：
    #   validation：当前层的校验结果。
    def validate(**kwargs: Any) -> PlannerValidation:
        validation = _validation(layer, **kwargs)
        return validation

    definition = hook_plugin(
        module_name=__name__,
        plugin_id=f"planning.specialist-{layer}",
        name=f"{display} Planner",
        description=f"Provides typed {display.lower()} constraints and deterministic validation.",
        capability_id=f"planning.specialist-{layer}.plan",
        capability_kind="planner",
        capability_name=f"{display} Planner",
        capability_description=(
            f"Contributes and validates the {display.lower()} layer without actuator authority."
        ),
        category_id="planning",
        category_label="任务规划",
        slot_id="planning.specialists",
        slot_label="分层规划器",
        activation_mode="multiple",
        category_order=40,
        slot_order=25,
        plugin_order=order,
        hooks={"contribute_planning": contribute, "validate_planning": validate},
        default_enabled=True,
        failure_mode="fail-closed",
        configuration_schema=(
            {
                "type": "object",
                "properties": {
                    "qualified_range_m": {
                        "type": "number",
                        "minimum": 10,
                        "maximum": 100000,
                    }
                },
                "additionalProperties": False,
            }
            if layer == "energy"
            else {}
        ),
    )
    return definition


# 功能：
#   注册十一类互补的规则式规划检查，数量不代表训练模型或实时控制器的数量。
# 输入：
#   无。
# 输出：
#   definitions：本产品内置的分层规划插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [_definition(layer, index * 10) for index, layer in enumerate(_LAYERS, start=1)]
    return definitions
