from __future__ import annotations

from typing import Any

from dronedream_agent_core.contracts import MissionContract
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   保留规则推荐并允许模型追加可选工具，不代替最终工具权限检查。
# 输入：
#   recommended_tool_ids：宿主规则层提供的工具建议列表。
#   _：本路由器不使用的扩展参数。
# 输出：
#   recommendation：混合路由模式及原顺序的推荐标识列表。
def _hybrid(*, recommended_tool_ids: list[str], **_: Any) -> dict[str, object]:
    recommendation = {
        "strategy": "rules-plus-model",
        "recommended_tool_ids": recommended_tool_ids,
    }
    return recommendation


# 功能：
#   把可选工具选择交给模型，不删除核心强制执行的飞行安全检查。
# 输入：
#   _：本策略不使用的调度参数。
# 输出：
#   recommendation：模型选择模式及空的规则推荐列表。
def _model_only(**_: Any) -> dict[str, object]:
    recommendation = {"strategy": "model", "recommended_tool_ids": []}
    return recommendation


# 功能：
#   在原推荐基础上追加目录中涉及安全、能源、通信和降落的工具，保留合同身份。
# 输入：
#   contract：当前任务合同。
#   catalog：宿主提供的已准入工具目录。
#   recommended_tool_ids：原有规则推荐列表。
#   _：本路由器不使用的扩展参数。
# 输出：
#   recommendation：安全优先模式、去重排序后的建议与合同标识。
def _safety_first(
    *,
    contract: MissionContract,
    catalog: list[dict[str, object]],
    recommended_tool_ids: list[str],
    **_: Any,
) -> dict[str, object]:
    values = set(recommended_tool_ids)
    for item in catalog:
        metadata = item.get("routing_metadata")
        if not isinstance(metadata, dict):
            continue
        domains = metadata.get("domains", [])
        if isinstance(domains, list) and set(domains).intersection(
            {"safety", "energy", "communications", "landing"}
        ):
            values.add(str(item["tool_id"]))
    recommendation = {
        "strategy": "safety-first-hybrid",
        "recommended_tool_ids": sorted(values),
        "contract_id": contract.contract_id,
    }
    return recommendation


# 功能：
#   注册混合、模型选择和安全优先三种互斥的可选工具推荐策略。
# 输入：
#   无。
# 输出：
#   definitions：包含推荐钩子、默认状态与失败关闭策略的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "tools.router-hybrid",
            "混合工具路由",
            "由确定性推荐条件提供底线，再由模型补充任务相关工具。",
            _hybrid,
            True,
        ),
        (
            "tools.router-model",
            "模型工具路由",
            "让模型完全根据结构化工具目录选择可选工具。",
            _model_only,
            False,
        ),
        (
            "tools.router-safety-first",
            "安全优先工具路由",
            "自动推荐安全、能源、通信和紧急降落相关工具，再交给模型补充。",
            _safety_first,
            False,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.recommend",
            capability_kind="tool-router",
            capability_name=name,
            capability_description=description,
            category_id="tools",
            category_label="工具与集成",
            slot_id="tools.router-policy",
            slot_label="工具路由策略",
            activation_mode="single",
            category_order=50,
            slot_order=10,
            plugin_order=index * 10,
            hooks={"recommend_tools": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
        )
        for index, (plugin_id, name, description, handler, enabled) in enumerate(values, start=1)
    ]
    return definitions
