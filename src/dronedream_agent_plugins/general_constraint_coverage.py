"""Keyword coverage is an advisory writing aid, never proof that a mission is safe."""

from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition, ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)
from dronedream_agent_core.tools import ToolPlugin

from ._helpers import constraint_text

INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["constraints"],
    "properties": {
        "constraints": {
            "type": "array",
            "maxItems": 32,
            "items": {"type": "string", "maxLength": 4000},
        }
    },
}
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["covered", "missing", "coverage_ratio"],
    "properties": {
        "covered": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "coverage_ratio": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


# 功能：
#   检查约束文字是否提及四类执行维度；否定句也可能命中，不能代替语义理解或安全验证。
# 输入：
#   value：包含数量和长度受限的 constraints 文本列表。
# 输出：
#   coverage：提及维度、未提及维度及提及比例。
def _coverage(value: dict[str, object]) -> dict[str, object]:
    rendered = constraint_text(value.get("constraints"))
    dimensions = {
        "safety priority": ("safe", "安全"),
        "return behavior": ("return", "返回", "返程"),
        "speed policy": ("speed", "速度", "快", "慢"),
        "execution confirmation": ("confirm", "确认", "执行前"),
    }
    covered = [
        name for name, tokens in dimensions.items() if any(token in rendered for token in tokens)
    ]
    missing = [name for name in dimensions if name not in covered]
    coverage = {
        "covered": covered,
        "missing": missing,
        "coverage_ratio": len(covered) / len(dimensions),
    }
    return coverage


# 功能：
#   声明只读的约束文字检查工具，将模型参数来源绑定到任务合同。
# 输入：
#   _environment：统一工具工厂提供的宿主环境，本工具不访问其中服务。
# 输出：
#   tools：带有输入输出 Schema 与只读权限的工具列表。
def _tools(_environment: ToolEnvironment) -> list[ToolPlugin]:
    tools = [
        ToolPlugin(
            tool_id="general.constraint-coverage",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            handler=_coverage,
            routing_metadata={
                "recommended_when": {"always": True},
                "domains": ["safety", "planning"],
                "purpose": "Check whether mission constraints cover core execution boundaries.",
                "required_argument_sources": {"constraints": "mission_contract.constraints"},
            },
        )
    ]
    return tools


# 功能：
#   注册约束覆盖顾问的能力和展示位置，不授予安全放行权或执行器控制权。
# 输入：
#   无。
# 输出：
#   definition：任务顾问插件定义及其工具工厂。
def plugin_definition() -> PluginDefinition:
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="general.constraint-coverage",
            name="约束覆盖检查",
            version="1.0.0",
            description="提示约束文本提及的安全、返程、速度和确认维度，不替代实际安全验证。",
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definition"
            ),
            capabilities=[
                PluginCapability(
                    capability_id="general.constraint-coverage",
                    kind="evidence",
                    name="约束覆盖检查",
                    description="指出任务约束中已覆盖和仍缺少的维度。",
                    input_schema=INPUT_SCHEMA,
                    output_schema=OUTPUT_SCHEMA,
                    metadata={
                        "recommended_when": {"always": True},
                        "domains": ["safety", "planning"],
                        "purpose": (
                            "Check whether mission constraints cover core execution boundaries."
                        ),
                        "required_argument_sources": {
                            "constraints": "mission_contract.constraints"
                        },
                    },
                )
            ],
            permissions=["mission.read"],
            default_enabled=True,
            removable=False,
            placement=PluginPlacement(
                category_id="general",
                category_label="通用增强",
                slot_id="general.mission-advisors",
                slot_label="任务顾问",
                activation_mode="multiple",
                scope="general",
                category_order=0,
                slot_order=10,
                plugin_order=20,
            ),
        ),
        tool_factory=_tools,
    )
    return definition
