from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   将替代规划预算注册到同一互斥插槽，保留只读任务权限与失败关闭策略。
# 输入：
#   plugin_id：工作流策略标识。
#   name：展示名称。
#   description：策略用途说明。
#   order：同插槽内的排序值。
#   metadata：各类最大轮数、调用次数及每轮审查次数。
# 输出：
#   definition：带具体预算元数据的规划工作流定义。
def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    order: int,
    metadata: dict[str, int],
) -> PluginDefinition:
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
                    capability_id=f"{plugin_id}.policy",
                    kind="workflow-policy",
                    name=name,
                    description=description,
                    authority="plan",
                    metadata=metadata,
                )
            ],
            permissions=["mission.read"],
            default_enabled=False,
            removable=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.workflow-policy",
                slot_label="规划工作流",
                activation_mode="single",
                scope="mission",
                failure_mode="fail-closed",
                category_order=40,
                slot_order=20,
                plugin_order=order,
            ),
        )
    )
    return definition


# 功能：
#   提供低调用量预览与高审查密度委员会两种方案，均不移除必需安全检查。
# 输入：
#   无。
# 输出：
#   definitions：默认未启用的两个替代工作流定义。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _definition(
            plugin_id="workflow.fast-preview",
            name="快速计划预览",
            description="以较少模型调用快速形成可供用户修改的初版计划，安全门保持不变。",
            order=30,
            metadata={
                "max_intent_rounds": 2,
                "max_planning_rounds": 2,
                "plugin_router_rounds": 1,
                "maximum_plugin_calls": 4,
                "intent_reviews_per_round": 1,
                "plan_reviews_per_round": 1,
            },
        ),
        _definition(
            plugin_id="workflow.committee-review",
            name="委员会审查流程",
            description="每轮执行三次独立意图与计划审查，全部接受后才生成可确认计划。",
            order=40,
            metadata={
                "max_intent_rounds": 5,
                "max_planning_rounds": 5,
                "plugin_router_rounds": 3,
                "maximum_plugin_calls": 16,
                "intent_reviews_per_round": 3,
                "plan_reviews_per_round": 3,
            },
        ),
    ]
    return definitions
