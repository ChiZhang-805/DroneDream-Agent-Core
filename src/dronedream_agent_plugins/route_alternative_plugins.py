"""Composable route-candidate generators and deterministic multi-objective ranking."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from dronedream_agent_core.collision import (
    CONSERVATIVE_VEHICLE_ENVELOPE_SCALE, localization_required_clearance_m,
)
from dronedream_agent_core.contracts import (
    GraphRoute,
    RouteAlternativeDecision,
    RouteAlternativeSet,
    RouteQuery,
)
from dronedream_agent_core.known_map_planner import (
    KnownMapMetricPlanner,
    MetricPlannerPolicy,
)
from dronedream_agent_core.navigation import (
    clearance_first_route,
    energy_efficient_route,
    shortest_route,
    stability_first_route,
)
from dronedream_agent_core.plugin_api import PluginDefinition, ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)
from dronedream_agent_core.tools import ToolPlugin
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import bounded_number, clearance_evidence_matches_route


# 功能：
#   校验机体尺寸，并按独立运行门控相同的机体比例与最低余量计算规划净空。
# 输入：
#   vehicle_diameter_m：未扩张的机体直径，单位米。
# 输出：
#   clearance_m：规划搜索必须保留的净空，单位米。
def _planning_clearance_requirement_m(vehicle_diameter_m: float) -> float:
    if not bounded_number(vehicle_diameter_m, 0, math.inf) or vehicle_diameter_m <= 0:
        raise ValueError("ROUTE_ALTERNATIVE_VEHICLE_DIAMETER_INVALID")
    vehicle_radius_m = vehicle_diameter_m / 2.0
    clearance_m = max(0.35, vehicle_radius_m * 0.75)
    return clearance_m


# 功能：
#   注册基于任务地图的候选路线生成器；候选结果仍须通过独立净空与任务合同检查。
# 输入：
#   plugin_id：候选策略插件标识。
#   name：显示名称。
#   description：策略用途描述。
#   order：候选策略的装配顺序。
#   planner：接收地图及路线查询的规划函数。
# 输出：
#   definition：包含候选路线工具工厂的插件定义。
def _candidate_definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    order: int,
    planner: Callable,
) -> PluginDefinition:
    # 功能：
    #   将规划函数接到本次任务选择的地图，不使用固定示例地图或启动飞行执行器。
    # 输入：
    #   environment：包含本次地图的工具环境。
    # 输出：
    #   plugins：当前策略的规划工具列表。
    def tools(environment: ToolEnvironment) -> list[ToolPlugin]:
        plugins = [
            ToolPlugin(
                tool_id=f"{plugin_id}.candidate",
                version="1.0.0",
                authority="plan",
                input_type=RouteQuery,
                output_type=GraphRoute,
                handler=lambda query: planner(environment.map_graph, query),
            )
        ]
        return plugins

    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id=plugin_id,
            name=name,
            version="1.0.0",
            description=description,
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definitions"
            ),
            capabilities=[
                PluginCapability(
                    capability_id=f"{plugin_id}.candidate",
                    kind="plan-optimizer",
                    name=name,
                    description=description,
                    authority="plan",
                    input_schema=RouteQuery.model_json_schema(),
                    output_schema=GraphRoute.model_json_schema(),
                    metadata={"produces_alternative": True},
                )
            ],
            permissions=["asset.read", "mission.read"],
            default_enabled=True,
            removable=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.route-candidates",
                slot_label="候选路线生成器",
                activation_mode="multiple",
                scope="mission",
                failure_mode="isolate",
                category_order=40,
                slot_order=14,
                plugin_order=order,
            ),
        ),
        tool_factory=tools,
    )
    return definition


# 功能：
#   声明在合格三维几何中搜索自由空间路线的可选候选插件，不把拓扑骨架直接当作答案。
# 输入：
#   无。
# 输出：
#   definition：绑定度量规划工具工厂的候选插件定义。
def _metric_candidate_definition() -> PluginDefinition:
    # 功能：
    #   1. 使用保守机体包络及运行净空创建有界搜索器。
    #   2. 几何不可用时仅停用此可选策略，不伪造路线或阻止其他合法地图工具装配。
    # 输入：
    #   environment：本任务的地图、语义制品路径和机体尺寸。
    # 输出：
    #   plugins：度量路线工具列表；缺少可用几何时为空列表。
    def tools(environment: ToolEnvironment) -> list[ToolPlugin]:
        planning_clearance_m = _planning_clearance_requirement_m(environment.vehicle_diameter_m)
        variance = getattr(environment, "planning_localization_variance_m2", None)
        if variance is not None:
            planning_clearance_m = max(planning_clearance_m, localization_required_clearance_m(
                variance))
        try:
            planning_phase = getattr(environment, "planning_phase", "runtime")
            if planning_phase not in {"initial", "runtime"}:
                raise ValueError("METRIC_PLANNING_PHASE_INVALID")
            planner = KnownMapMetricPlanner(
                graph=environment.map_graph,
                semantic_path=environment.semantic_path,
                # The selected route is later checked by the conservative
                # clearance plug-in.  Search with that same expanded envelope
                # so a path cannot pass planning and then lose clearance only
                # because validation silently made the aircraft larger.
                vehicle_diameter_m=(
                    environment.vehicle_diameter_m * CONSERVATIVE_VEHICLE_ENVELOPE_SCALE
                ),
                vehicle_height_m=(
                    environment.vehicle_height_m * CONSERVATIVE_VEHICLE_ENVELOPE_SCALE
                ),
                policy=MetricPlannerPolicy(
                    resolution_m=0.45,
                    required_clearance_m=planning_clearance_m,
                    vertical_cost_multiplier=2.4,
                    clearance_cost_weight=0.22,
                    # 起飞前长路线允许充分求解；飞行中改令仍只有短预算，本地安全控制独立运行。
                    maximum_search_seconds=60.0 if planning_phase == "initial" else 8.0,
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # This optional candidate generator is meaningful only for a
            # qualified metric semantic artifact.  Plugin discovery must still
            # expose unrelated tools for graph-only maps and migration tests;
            # the manifest's isolate failure mode therefore makes the metric
            # tool unavailable instead of crashing the entire registry.
            plugins = []
            return plugins
        plugins = [
            ToolPlugin(
                tool_id="planning.candidate-metric-geometry.candidate",
                version="1.0.0",
                authority="plan",
                input_type=RouteQuery,
                output_type=GraphRoute,
                handler=planner.plan,
            )
        ]
        return plugins

    description = (
        "直接在合格碰撞几何中执行三维度量搜索，把机体包络和运行净空作为硬约束，"
        "不把预录制导航链路当作飞行路线。"
    )
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="planning.candidate-metric-geometry",
            name="三维度量自由空间路线",
            version="1.0.0",
            description=description,
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definitions"
            ),
            capabilities=[
                PluginCapability(
                    capability_id="planning.candidate-metric-geometry.candidate",
                    kind="plan-optimizer",
                    name="三维度量自由空间路线",
                    description=description,
                    authority="plan",
                    input_schema=RouteQuery.model_json_schema(),
                    output_schema=GraphRoute.model_json_schema(),
                    metadata={
                        "produces_alternative": True,
                        "occupancy_source": "qualified-collision-semantics",
                        "vehicle_scaled_operational_clearance": True,
                    },
                )
            ],
            permissions=["asset.read", "mission.read"],
            default_enabled=True,
            removable=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.route-candidates",
                slot_label="候选路线生成器",
                activation_mode="multiple",
                scope="mission",
                failure_mode="isolate",
                category_order=40,
                slot_order=14,
                plugin_order=8,
            ),
        ),
        tool_factory=tools,
    )
    return definition


# 功能：
#   冻结本任务排序配置并创建纯评分工具，评分只能选择通过硬门控的候选，不能推翻否决。
# 输入：
#   environment：提供本任务可选权重配置的工具环境。
# 输出：
#   plugins：多目标路线排序工具列表。
def _ranker_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    configuration = copy_json(
        {} if environment.plugin_configuration is None else environment.plugin_configuration
    )
    metric_policy = {
        "distance_m": ("distance_weight", 0.20),
        "minimum_clearance_m": ("clearance_weight", 0.46),
        "energy_proxy": ("energy_weight", 0.14),
        "transition_count": ("stability_weight", 0.10),
        "qualification_penalty": ("qualification_weight", 0.10),
    }
    if not isinstance(configuration, dict) or set(configuration) - {
        entry[0] for entry in metric_policy.values()
    }:
        raise ValueError("ROUTE_ALTERNATIVE_WEIGHTS_INVALID")

    # 功能：
    #   1. 重新校验候选类型、唯一身份、可行性证据及指标与路线的一致性。
    #   2. 只在可行候选间归一化权重与指标，得分相同时按标识稳定排序。
    #   3. 输出选择理由与被拒候选问题；此检查不替代对实际地图几何的独立验证。
    # 输入：
    #   value：带路线、净空报告、门控与目标权重的候选集合。
    # 输出：
    #   decision：选中标识、排名、归一化分数及拒绝原因。
    def rank(value: RouteAlternativeSet) -> RouteAlternativeDecision:
        value = RouteAlternativeSet.model_validate(value.model_dump(mode="python"), strict=True)
        candidates = value.candidates
        if len({item.alternative_id for item in candidates}) != len(candidates):
            raise ValueError("ROUTE_ALTERNATIVE_IDENTIFIERS_DUPLICATE")
        for candidate in candidates:
            # A bare feasible flag must not promote a failed hard gate or a
            # report borrowed from another route. This checks consistency; core
            # still reruns clearance against the selected map's actual geometry.
            if candidate.feasible and (
                not candidate.hard_gates
                or not all(candidate.hard_gates.values())
                or not clearance_evidence_matches_route(candidate.clearance, candidate.route)
            ):
                raise ValueError("ROUTE_ALTERNATIVE_FEASIBILITY_EVIDENCE_INVALID")
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            raise ValueError("ROUTE_ALTERNATIVE_NO_FEASIBLE_CANDIDATE")
        if set(value.objective_weights) - set(metric_policy):
            raise ValueError("ROUTE_ALTERNATIVE_WEIGHTS_INVALID")
        weights = {
            metric: configuration.get(key, value.objective_weights.get(metric, default))
            for metric, (key, default) in metric_policy.items()
        }
        if any(not bounded_number(weight, 0, 1) for weight in weights.values()):
            raise ValueError("ROUTE_ALTERNATIVE_WEIGHTS_INVALID")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("ROUTE_ALTERNATIVE_WEIGHTS_INVALID")
        weights = {key: weight / total_weight for key, weight in weights.items()}
        ranges: dict[str, tuple[float, float]] = {}
        for metric in weights:
            values = [candidate.objectives.get(metric) for candidate in feasible]
            if any(not bounded_number(number, 0, math.inf) for number in values):
                raise ValueError("ROUTE_ALTERNATIVE_METRICS_INVALID")
            # Each range is computed once, not once per candidate. All values
            # are finite/nonnegative so subtraction cannot overflow to infinity.
            ranges[metric] = (min(values), max(values))

        for candidate in feasible:
            expected_metrics = {
                "distance_m": candidate.route.route_length_m,
                "minimum_clearance_m": candidate.clearance.minimum_clearance_m,
                "transition_count": float(len(candidate.route.edge_ids)),
                "qualification_penalty": 0.0 if candidate.route.all_edges_flight_verified else 1.0,
            }
            # 哈希只绑定路线内容，不自动绑定旁边的评分字段；不能让两套数字各说各话。
            # 能耗代理由估计器提供，不在此强行规定为某一种物理模型。
            if any(
                not math.isclose(candidate.objectives[metric], expected, rel_tol=1e-9, abs_tol=1e-9)
                for metric, expected in expected_metrics.items()
            ):
                raise ValueError("ROUTE_ALTERNATIVE_METRIC_EVIDENCE_MISMATCH")

        # 功能：
        #   将净空按越大越好、其余代价按越小越好映射至同一尺度，相同指标给相同分数。
        # 输入：
        #   candidate：已通过指标和证据检查的候选。
        #   metric：当前评分指标名称。
        # 输出：
        #   score：该指标在当前可行候选集合中的归一化分数。
        def normalized(candidate, metric: str) -> float:
            low, high = ranges[metric]
            if high == low:
                score = 1.0
            elif metric == "minimum_clearance_m":
                score = (candidate.objectives[metric] - low) / (high - low)
            else:
                score = (high - candidate.objectives[metric]) / (high - low)
            return score

        scores = {
            candidate.alternative_id: round(
                sum(weights[metric] * normalized(candidate, metric) for metric in weights),
                8,
            )
            for candidate in feasible
        }
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        selected = next(item for item in feasible if item.alternative_id == ranked[0])
        rejected = {
            candidate.alternative_id: candidate.issue_codes
            for candidate in candidates
            if not candidate.feasible
        }
        decision = RouteAlternativeDecision(
            selected_alternative_id=selected.alternative_id,
            ranked_alternative_ids=ranked,
            normalized_scores=scores,
            selection_reasons=[
                "所有硬约束均已通过",
                (
                    "按距离、连续净空、能耗代理与转向/航段稳定性进行归一化加权，"
                    f"综合得分 {scores[selected.alternative_id]:.4f}"
                ),
                f"选中策略 {selected.strategy_tool_id}",
            ],
            rejected_alternatives=rejected,
        )
        return decision

    plugins = [
        ToolPlugin(
            tool_id="planning.multi-objective-ranker.rank",
            version="1.0.0",
            authority="plan",
            input_type=RouteAlternativeSet,
            output_type=RouteAlternativeDecision,
            handler=rank,
        )
    ]
    return plugins


# 功能：
#   暴露零至一的权重配置，不允许通过禁用排序器或调权重绕过硬约束。
# 输入：
#   无。
# 输出：
#   definition：单选、失败关闭的路线排序插件定义。
def _ranker_definition() -> PluginDefinition:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            name: {"type": "number", "minimum": 0, "maximum": 1}
            for name in (
                "distance_weight",
                "clearance_weight",
                "energy_weight",
                "stability_weight",
                "qualification_weight",
            )
        },
    }
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="planning.multi-objective-ranker",
            name="多目标路线排序",
            version="1.0.0",
            description="在硬约束通过后对距离、净空、能耗与稳定性进行归一化排序。",
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definitions"
            ),
            capabilities=[
                PluginCapability(
                    capability_id="planning.multi-objective-ranker.rank",
                    kind="plan-optimizer",
                    name="多目标路线排序",
                    description="从结构化候选路线中选择具有证据的可行解。",
                    authority="plan",
                    input_schema=RouteAlternativeSet.model_json_schema(),
                    output_schema=RouteAlternativeDecision.model_json_schema(),
                    metadata={"deterministic": True, "normalization": "min-max"},
                )
            ],
            permissions=["mission.read"],
            default_enabled=True,
            removable=False,
            disable_allowed=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.alternative-ranker",
                slot_label="候选路线排序器",
                activation_mode="single",
                scope="mission",
                failure_mode="fail-closed",
                category_order=40,
                slot_order=16,
                plugin_order=10,
            ),
            configuration_schema=schema,
        ),
        tool_factory=_ranker_tools,
    )
    return definition


# 功能：
#   组合度量自由空间、距离、净空、能耗和稳定性候选，以及独立的证据约束排序器。
# 输入：
#   无。
# 输出：
#   definitions：候选生成与排序插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _metric_candidate_definition(),
        _candidate_definition(
            plugin_id="planning.candidate-distance",
            name="最短距离候选路线",
            description=(
                "生成纯距离最短的基线候选，让多目标排序器能够量化安全、能耗与稳定性策略的实际绕行代价。"
            ),
            order=5,
            planner=shortest_route,
        ),
        _candidate_definition(
            plugin_id="planning.candidate-clearance",
            name="净空候选路线",
            description="生成偏好宽裕净空与飞行验证边的候选路线。",
            order=10,
            planner=clearance_first_route,
        ),
        _candidate_definition(
            plugin_id="planning.candidate-energy",
            name="能耗候选路线",
            description="生成降低距离、爬升与加速度代理的候选路线。",
            order=20,
            planner=energy_efficient_route,
        ),
        _candidate_definition(
            plugin_id="planning.candidate-stability",
            name="稳定候选路线",
            description="生成偏好验证边、较少过渡与温和速度的候选路线。",
            order=30,
            planner=stability_first_route,
        ),
        _ranker_definition(),
    ]
    return definitions
