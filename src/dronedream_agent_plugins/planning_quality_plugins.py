"""Advisory plan scores and independent veto gates; neither grants flight authority."""

from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.contracts import (
    FlightPlan,
    GraphRoute,
    MissionContract,
    Px4Track,
    RouteClearanceReport,
    SemanticPlan,
    TaskGraph,
    VehicleAsset,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import (
    bounded_number,
    clearance_evidence_matches_route,
    hook_plugin,
    policy_number,
    within_qualified_range,
)


# 功能：
#   使用共享航程边界检查，保留浮点序列化容差，不允许真实航程超限。
# 输入：
#   distance_m：所需航程，单位米。
#   qualified_range_m：已验收航程，单位米。
# 输出：
#   accepted：航程满足限制时为 True。
def _within_qualified_range(distance_m: float, qualified_range_m: float) -> bool:
    accepted = within_qualified_range(distance_m, qualified_range_m)
    return accepted


# 功能：
#   导出路线长度作为越低越好的排序指标，不把该声明当作实时续航测量。
# 输入：
#   route：待评分路线。
#   _：本评分不使用的其他制品。
# 输出：
#   score：距离指标的数值、单位和偏好方向。
def _distance_score(*, route: GraphRoute, **_: Any) -> dict[str, object]:
    score = {
        "metric": "distance",
        "value": route.route_length_m,
        "unit": "m",
        "preference": "lower",
    }
    return score


# 功能：
#   导出带正负号的净空指标，负值可用于说明拒绝路线，不能因输出评分就视为安全。
# 输入：
#   clearance：路线净空报告。
#   _：本评分不使用的其他制品。
# 输出：
#   score：最小净空的数值、单位及越高越好的偏好。
def _clearance_score(*, clearance: RouteClearanceReport, **_: Any) -> dict[str, object]:
    score = {
        "metric": "minimum-clearance",
        "value": clearance.minimum_clearance_m,
        "unit": "m",
        "preference": "higher",
    }
    return score


# 功能：
#   计算相邻速度上限变化的均方根，不冒称实际加速度、姿态或闭环稳定性测量。
# 输入：
#   px4_track：含速度上限的飞行轨迹。
#   _：本评分不使用的其他制品。
# 输出：
#   score：速度变化指标及单位与偏好。
def _stability_score(*, px4_track: Px4Track, **_: Any) -> dict[str, object]:
    speeds = [point.speed_limit_mps for point in px4_track.points]
    if not speeds or any(not bounded_number(speed, 0, math.inf) for speed in speeds):
        raise ValueError("TRACK_SPEED_VALUES_INVALID")
    changes = [abs(second - first) for first, second in zip(speeds, speeds[1:], strict=False)]
    score = {
        "metric": "speed-transition-rms",
        "value": math.hypot(*changes) / math.sqrt(max(len(changes), 1)),
        "unit": "mps",
        "preference": "lower",
    }
    return score


# 功能：
#   用距离、正向爬升及航点数量估算相对代价，单位为加权米而非焦耳或实际电量。
# 输入：
#   route：待比较路线。
#   px4_track：提供高度变化和航点数量的轨迹。
#   _：本估计不使用的其他制品。
# 输出：
#   score：能耗代理分值、爬升量及越低越好的偏好。
def _energy_proxy(*, route: GraphRoute, px4_track: Px4Track, **_: Any) -> dict[str, object]:
    climb_m = sum(
        max(0.0, second.up_m - first.up_m)
        for first, second in zip(
            px4_track.source_world_points,
            px4_track.source_world_points[1:],
            strict=False,
        )
    )
    proxy = route.route_length_m + climb_m * 2.5 + len(px4_track.points) * 0.05
    score = {
        "metric": "energy-proxy",
        "value": proxy,
        "unit": "weighted-m",
        "preference": "lower",
        "climb_m": climb_m,
    }
    return score


# 功能：
#   核对合同身份、路线两端、语义计划摘要及净空内容一致，拒绝借用其他任务的制品。
# 输入：
#   contract：当前任务合同。
#   semantic_plan：本次语义计划。
#   flight_plan：应绑定当前合同及语义摘要的飞行计划。
#   route：待执行路线。
#   clearance：实际路线的净空报告。
#   _：本门控不使用的其他制品。
# 输出：
#   validation：身份及证据门控结果和失败原因。
def _route_binding_gate(
    *,
    contract: MissionContract,
    semantic_plan: SemanticPlan,
    flight_plan: FlightPlan,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    **_: Any,
) -> dict[str, object]:
    gates = {
        "flight_plan_contract_bound": flight_plan.contract_id == contract.contract_id,
        "route_start_bound": route.start_node == contract.start_node,
        "route_return_bound": route.goal_node == contract.return_node,
        "semantic_hash_bound": flight_plan.semantic_plan_sha256 == sha256_json(semantic_plan),
        "clearance_route_bound": clearance.route_sha256 == sha256_json(route),
        "continuous_clearance_accepted": clearance_evidence_matches_route(clearance, route),
    }
    failed = [name for name, accepted in gates.items() if not accepted]
    validation = {
        "validator": "route-binding",
        "accepted": not failed,
        "gates": gates,
        "issue_codes": [f"ROUTE_BINDING_{name.upper()}" for name in failed],
    }
    return validation


# 功能：
#   取件合同必须恰有一个绑定目标的取件动作，非取件合同不能额外加入取件副作用。
# 输入：
#   contract：当前合同的目标与载荷动作。
#   task_graph：待执行的任务动作图。
#   _：本门控不使用的其他制品。
# 输出：
#   validation：取件流程检查结果、数量及修复要求。
def _payload_gate(
    *, contract: MissionContract, task_graph: TaskGraph, **_: Any
) -> dict[str, object]:
    pickup_tasks = [task for task in task_graph.nodes if task.action == "pickup"]
    accepted = (
        len(pickup_tasks) == 1 and pickup_tasks[0].target_node == contract.target_node
        if contract.payload_action == "pickup"
        else not pickup_tasks
    )
    validation = {
        "validator": "payload-workflow",
        "accepted": accepted,
        "pickup_task_count": len(pickup_tasks),
        "issue_codes": [] if accepted else ["PAYLOAD_PICKUP_TASK_INVALID"],
        "repair_instructions": (
            [] if accepted else ["Create exactly one pickup task at the contract target node."]
        ),
    }
    return validation


# 功能：
#   独立检查非空、有限低速和航点停止策略，不让较高的评分覆盖准备阶段的否决。
# 输入：
#   px4_track：待检查的轨迹。
#   _：本门控不使用的其他制品。
# 输出：
#   validation：速度与停止策略的接受状态及拒绝原因。
def _stability_gate(*, px4_track: Px4Track, **_: Any) -> dict[str, object]:
    speeds = [point.speed_limit_mps for point in px4_track.points]
    valid = bool(speeds) and all(bounded_number(speed, 0, 3) for speed in speeds)
    max_speed = max(speeds) if valid else None
    accepted = valid and px4_track.stop_at_waypoints is True
    validation = {
        "validator": "track-stability",
        "accepted": accepted,
        "maximum_speed_mps": max_speed,
        "stop_at_waypoints": px4_track.stop_at_waypoints,
        "issue_codes": [] if accepted else ["TRACK_STABILITY_POLICY_REJECTED"],
    }
    return validation


# 功能：
#   校验已证明净空与速度的组合，在狭窄段收紧速度；配置错误不得静默回退默认。
# 输入：
#   flight_plan：包含各段净空和速度限制的飞行计划。
#   configuration：狭窄净空阈值及相应速度上限。
#   _：本门控不使用的其他制品。
# 输出：
#   validation：各段值有效性、违规段列表及修复要求。
def _clearance_speed_gate(
    *,
    flight_plan: FlightPlan,
    configuration: dict[str, object] | None = None,
    **_: Any,
) -> dict[str, object]:
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict) or set(configured) - {
        "tight_clearance_m",
        "maximum_tight_speed_mps",
    }:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    tight_clearance_m = policy_number(configured, "tight_clearance_m", 1.25, 0.3, 10)
    maximum_tight_speed_mps = policy_number(configured, "maximum_tight_speed_mps", 1, 0.1, 3)
    valid = bool(flight_plan.segments) and all(
        bounded_number(segment.minimum_clearance_m, 0, math.inf)
        and bounded_number(segment.speed_limit_mps, 0, 3)
        for segment in flight_plan.segments
    )
    violations = [
        {
            "segment_id": segment.segment_id,
            "minimum_clearance_m": segment.minimum_clearance_m,
            "speed_limit_mps": segment.speed_limit_mps,
        }
        for segment in flight_plan.segments
        if bounded_number(segment.minimum_clearance_m, 0, math.inf)
        and bounded_number(segment.speed_limit_mps, 0, 3)
        and segment.minimum_clearance_m < tight_clearance_m
        and segment.speed_limit_mps > maximum_tight_speed_mps
    ]
    accepted = valid and not violations
    validation = {
        "validator": "clearance-speed-coupling",
        "accepted": accepted,
        "tight_clearance_m": tight_clearance_m,
        "maximum_tight_speed_mps": maximum_tight_speed_mps,
        "violations": violations,
        "segment_values_valid": valid,
        "issue_codes": [] if accepted else ["CLEARANCE_SPEED_COUPLING_REJECTED"],
        "repair_instructions": (
            []
            if accepted
            else ["Reduce speed on every segment whose proven clearance is below the threshold."]
        ),
    }
    return validation


# 功能：
#   1. 使用已含资产余量的验收航程，避免再次重复扣除同一份保留电量。
#   2. 额外配置只能收紧航程或提高余量，拒绝非法配置和不合法资产包络。
# 输入：
#   route：待评估的路线。
#   vehicle：提供已验收航程及保留电量的机体资产。
#   configuration：可选的更严格航程上限与保留比例。
#   _：本门控不使用的其他制品。
# 输出：
#   validation：可用范围、采用的余量、接受状态及修复要求。
def _energy_reserve_gate(
    *,
    route: GraphRoute,
    vehicle: VehicleAsset,
    configuration: dict[str, object] | None = None,
    **_: Any,
) -> dict[str, object]:
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict) or set(configured) - {
        "qualified_range_m",
        "reserve_fraction",
    }:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    if (
        not bounded_number(vehicle.qualified_range_m, 0, 1_000_000)
        or vehicle.qualified_range_m == 0
        or not bounded_number(vehicle.reserve_battery_percent, 10, 90)
    ):
        raise ValueError("ENERGY_ASSET_ENVELOPE_INVALID")
    # Defaults come from the broader asset contract; an explicit override must
    # satisfy this plugin's own narrower settings schema.
    requested_range_m = (
        policy_number(configured, "qualified_range_m", 10, 10, 100_000)
        if "qualified_range_m" in configured
        else vehicle.qualified_range_m
    )
    range_m = min(requested_range_m, vehicle.qualified_range_m)
    asset_reserve = vehicle.reserve_battery_percent / 100.0
    reserve = max(
        policy_number(configured, "reserve_fraction", 0.05, 0.05, 0.8)
        if "reserve_fraction" in configured
        else asset_reserve,
        asset_reserve,
    )
    # qualified_range_m already includes the asset's declared reserve. A stricter
    # plugin reserve scales that envelope down; it can never increase it.
    usable = range_m * (1.0 - reserve) / (1.0 - asset_reserve)
    accepted = _within_qualified_range(route.route_length_m, usable)
    validation = {
        "validator": "energy-reserve",
        "accepted": accepted,
        "qualified_range_m": range_m,
        "asset_qualified_range_m": vehicle.qualified_range_m,
        "asset_reserve_fraction": asset_reserve,
        "reserve_fraction": reserve,
        "usable_range_m": usable,
        "route_length_m": route.route_length_m,
        "issue_codes": [] if accepted else ["ENERGY_RESERVE_INSUFFICIENT"],
        "repair_instructions": (
            [] if accepted else ["Shorten the route or select a qualified higher-range vehicle."]
        ),
    }
    return validation


# 功能：
#   汇总绑定本合同的路线、净空和非空检查点是否就绪，仅供准备结果展示，不授予起飞权。
# 输入：
#   contract：当前任务合同。
#   route：本次路线。
#   clearance：应匹配本路线的净空报告。
#   runtime_checkpoints：应匹配本合同的检查点契约。
#   _：本评估不使用的其他制品。
# 输出：
#   evaluation：准备就绪状态及距离、净空和检查点数量。
def _readiness_evaluation(
    *,
    contract: MissionContract,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    runtime_checkpoints: Any,
    **_: Any,
) -> dict[str, object]:
    evaluation = {
        "evaluation": "preflight-readiness",
        "contract_id": contract.contract_id,
        "route_length_m": route.route_length_m,
        "minimum_clearance_m": clearance.minimum_clearance_m,
        "checkpoint_count": len(runtime_checkpoints.checkpoints),
        "ready": (
            clearance_evidence_matches_route(clearance, route)
            and runtime_checkpoints.contract_id == contract.contract_id
            and route.start_node == contract.start_node
            and route.goal_node == contract.return_node
            and bool(runtime_checkpoints.checkpoints)
        ),
    }
    return evaluation


# 功能：
#   用任务、图边和轨迹点数量描述相对工作量，不当作校准后的耗时或成功概率。
# 输入：
#   task_graph：任务动作图。
#   route：本次路线。
#   px4_track：本次飞行轨迹。
#   _：本启发式评估不使用的其他制品。
# 输出：
#   evaluation：复杂度分数及各项构成数量。
def _complexity_evaluation(
    *, task_graph: TaskGraph, route: GraphRoute, px4_track: Px4Track, **_: Any
) -> dict[str, object]:
    complexity = len(task_graph.nodes) + len(route.edge_ids) + len(px4_track.points) / 10.0
    evaluation = {
        "evaluation": "mission-complexity",
        "score": round(complexity, 3),
        "task_count": len(task_graph.nodes),
        "edge_count": len(route.edge_ids),
        "track_point_count": len(px4_track.points),
    }
    return evaluation


# 功能：
#   分别注册建议式评分、失败关闭的计划否决门及建议式就绪评估，三者不能互相替代。
# 输入：
#   无。
# 输出：
#   definitions：评分器、计划门控和准备评估插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    scorers = [
        ("planning.score-distance", "距离评分", "计算任务路线总长度。", _distance_score),
        (
            "planning.score-clearance",
            "净空评分",
            "记录连续碰撞检查得到的最小净空。",
            _clearance_score,
        ),
        (
            "planning.score-stability",
            "稳定性评分",
            "计算航迹速度变化的均方根指标。",
            _stability_score,
        ),
        (
            "planning.score-energy",
            "能耗代理评分",
            "根据距离、爬升和航点数量估算相对能耗。",
            _energy_proxy,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(scorers, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.score",
                capability_kind="plan-scorer",
                capability_name=name,
                capability_description=description,
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.plan-scorers",
                slot_label="计划评分器",
                activation_mode="multiple",
                category_order=40,
                slot_order=30,
                plugin_order=index * 10,
                hooks={"score_plan": handler},
                default_enabled=True,
                failure_mode="advisory",
            )
        )
    validators = [
        (
            "validation.route-binding",
            "路线绑定验证",
            "验证合同、语义计划、飞行计划、路线和净空哈希的一致性。",
            _route_binding_gate,
            True,
            {},
        ),
        (
            "validation.payload-workflow",
            "载荷流程验证",
            "确保取件任务只有一个且绑定合同目标。",
            _payload_gate,
            True,
            {},
        ),
        (
            "validation.track-stability",
            "航迹稳定性验证",
            "限制过高速度并要求航点停止策略。",
            _stability_gate,
            True,
            {},
        ),
        (
            "validation.energy-reserve",
            "能源余量验证",
            "依据合格航程和预留比例否决能源不足的计划。",
            _energy_reserve_gate,
            False,
            {
                "type": "object",
                "properties": {
                    "qualified_range_m": {"type": "number", "minimum": 10, "maximum": 100000},
                    "reserve_fraction": {"type": "number", "minimum": 0.05, "maximum": 0.8},
                },
                "additionalProperties": False,
            },
        ),
        (
            "validation.clearance-speed-coupling",
            "净空速度耦合验证",
            "只在狭窄净空航段收紧速度，避免全局限速掩盖局部碰撞风险。",
            _clearance_speed_gate,
            False,
            {
                "type": "object",
                "properties": {
                    "tight_clearance_m": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 10.0,
                        "default": 1.25,
                    },
                    "maximum_tight_speed_mps": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 3.0,
                        "default": 1.0,
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]
    for index, (
        plugin_id,
        name,
        description,
        handler,
        enabled,
        schema,
    ) in enumerate(validators, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.validate",
                capability_kind="plan-validator",
                capability_name=name,
                capability_description=description,
                category_id="validation",
                category_label="安全与验证",
                slot_id="validation.plan-gates",
                slot_label="计划否决门",
                activation_mode="multiple",
                category_order=60,
                slot_order=30,
                plugin_order=index * 10,
                hooks={"validate_plan": handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
                configuration_schema=schema,
            )
        )
    evaluations = [
        (
            "evaluation.preflight-readiness",
            "起飞前就绪度",
            "汇总路线、净空和检查点是否达到起飞前闭环要求。",
            _readiness_evaluation,
        ),
        (
            "evaluation.mission-complexity",
            "任务复杂度",
            "根据任务、图边和航迹点数量形成可比较的复杂度指标。",
            _complexity_evaluation,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(evaluations, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.evaluate",
                capability_kind="evaluator",
                capability_name=name,
                capability_description=description,
                category_id="evaluation",
                category_label="证据与评测",
                slot_id="evaluation.preflight",
                slot_label="起飞前评测",
                activation_mode="multiple",
                category_order=90,
                slot_order=10,
                plugin_order=index * 10,
                hooks={"evaluate_preflight": handler},
                default_enabled=True,
                failure_mode="advisory",
            )
        )
    return definitions
