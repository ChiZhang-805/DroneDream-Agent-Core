from __future__ import annotations

from collections.abc import Callable

from dronedream_agent_core.contracts import GraphRoute, RouteQuery
from dronedream_agent_core.navigation import energy_efficient_route, stability_first_route
from dronedream_agent_core.plugin_api import PluginDefinition, ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)
from dronedream_agent_core.tools import ToolPlugin


def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    order: int,
    route: Callable,
    objective_profile: str,
    best_for: list[str],
    tradeoffs: list[str],
) -> PluginDefinition:
    """Build one graph-cost variant without changing input/output or safety contracts."""

    def tools(environment: ToolEnvironment) -> list[ToolPlugin]:
        """Close over the selected mission map for this strategy instance."""
        return [
            ToolPlugin(
                tool_id=plugin_id,
                version="1.0.0",
                authority="plan",
                input_type=RouteQuery,
                output_type=GraphRoute,
                handler=lambda query: route(environment.map_graph, query),
                routing_metadata={
                    "algorithm": "dijkstra",
                    "objective_profile": objective_profile,
                    "best_for": list(best_for),
                    "tradeoffs": list(tradeoffs),
                    "selection_authority": "model",
                    "requires_deterministic_validation": True,
                },
            )
        ]

    return PluginDefinition(
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
                    capability_id=plugin_id,
                    kind="planner",
                    name=name,
                    description=description,
                    authority="plan",
                    input_schema=RouteQuery.model_json_schema(),
                    output_schema=GraphRoute.model_json_schema(),
                    metadata={
                        "algorithm": "dijkstra",
                        "objective_profile": objective_profile,
                        "selection_authority": "model",
                    },
                )
            ],
            permissions=["asset.read", "mission.read"],
            default_enabled=False,
            removable=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.route-strategy",
                slot_label="路径策略",
                activation_mode="single",
                scope="mission",
                failure_mode="fail-closed",
                category_order=40,
                slot_order=10,
                plugin_order=order,
            ),
        ),
        tool_factory=tools,
    )


def plugin_definitions() -> list[PluginDefinition]:
    """Offer energy/stability proxies, not claims of measured battery consumption."""
    return [
        _definition(
            plugin_id="navigation.energy-efficient-route",
            name="能耗优先路径",
            description="联合距离、爬升和速度变化代理选择相对低能耗路线。",
            order=30,
            route=energy_efficient_route,
            objective_profile="energy-proxy",
            best_for=["long routes", "climb-sensitive missions"],
            tradeoffs=["proxy is not measured battery use", "topology-bound"],
        ),
        _definition(
            plugin_id="navigation.stability-first-route",
            name="稳定优先路径",
            description="优先飞行验证边、较少航段和温和速度包络。",
            order=40,
            route=stability_first_route,
            objective_profile="stability-first",
            best_for=["payload-sensitive flight", "few transitions"],
            tradeoffs=["longer route possible", "topology-bound"],
        ),
    ]
