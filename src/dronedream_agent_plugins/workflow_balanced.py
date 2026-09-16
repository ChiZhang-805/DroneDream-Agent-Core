from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   声明均衡规划的意图、修复、工具路由及审查预算，轮数上限不代表必须耗尽的调用数。
# 输入：
#   无。
# 输出：
#   definition：默认启用且只含规划权限的工作流策略定义。
def plugin_definition() -> PluginDefinition:
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="workflow.balanced",
            name="均衡规划流程",
            version="1.0.0",
            description="在规划质量、模型调用次数和响应时间之间保持均衡。",
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definition"
            ),
            capabilities=[
                PluginCapability(
                    capability_id="workflow.balanced.policy",
                    kind="workflow-policy",
                    name="均衡规划",
                    description="最多三轮意图审查、五轮规划修复和两轮工具路由。",
                    authority="plan",
                    metadata={
                        "max_intent_rounds": 3,
                        "max_planning_rounds": 5,
                        "plugin_router_rounds": 2,
                        "maximum_plugin_calls": 8,
                        "intent_reviews_per_round": 1,
                        "plan_reviews_per_round": 1,
                    },
                )
            ],
            permissions=["mission.read"],
            default_enabled=True,
            removable=False,
            placement=PluginPlacement(
                category_id="planning",
                category_label="任务规划",
                slot_id="planning.workflow-policy",
                slot_label="规划工作流",
                activation_mode="single",
                scope="mission",
                category_order=20,
                slot_order=20,
                plugin_order=10,
            ),
        )
    )
    return definition
