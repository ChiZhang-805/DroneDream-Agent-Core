from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   为读取、规划和仿真建议有限重试，只缓存读取或规划结果，最终权限由工具注册表检查。
# 输入：
#   tool：含工具权限类别的目录信息。
#   _：本策略不使用的扩展参数。
# 输出：
#   policy：重试、缓存、并行度和来源回执要求。
def _resilient(*, tool: dict[str, object], **_: Any) -> dict[str, object]:
    authority = tool.get("authority")
    policy = {
        "maximum_attempts": 2 if authority in {"read", "plan", "simulate"} else 1,
        "cache": authority in {"read", "plan"},
        "parallelism": 4,
        "provenance_required": True,
    }
    return policy


# 功能：
#   禁用缓存与自动重试，将批量工具并行度限制为一，便于复现实验。
# 输入：
#   _：固定串行策略不使用的调度参数。
# 输出：
#   policy：单次串行执行并要求来源回执的策略对象。
def _strict_serial(**_: Any) -> dict[str, object]:
    policy = {
        "maximum_attempts": 1,
        "cache": False,
        "parallelism": 1,
        "provenance_required": True,
    }
    return policy


# 功能：
#   注册弹性并行与严格串行两种互斥工具策略，均不能赋予执行器权限。
# 输入：
#   无。
# 输出：
#   definitions：包含实际策略钩子、默认状态和失败模式的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.resolve",
            capability_kind="tool-execution-policy",
            capability_name=name,
            capability_description=description,
            category_id="tools",
            category_label="工具与集成",
            slot_id="tools.execution-policy",
            slot_label="工具执行策略",
            activation_mode="single",
            category_order=50,
            slot_order=20,
            plugin_order=order,
            hooks={"resolve_tool_execution": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
        )
        for plugin_id, name, description, handler, enabled, order in [
            (
                "tools.execution-resilient",
                "弹性并行执行",
                "为只读和规划工具提供有界重试、缓存、并行和来源收据。",
                _resilient,
                True,
                10,
            ),
            (
                "tools.execution-strict-serial",
                "严格串行执行",
                "关闭缓存与重试，逐个执行工具以便复现实验。",
                _strict_serial,
                False,
                20,
            ),
        ]
    ]
    return definitions
