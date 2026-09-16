from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin

CRITIC_ROLES = frozenset(
    {
        "intent_critic",
        "plan_critic",
        "execution_monitor",
        "completion_verifier",
    }
)

SAFETY_ROLES = frozenset(
    {
        "execution_monitor",
        "completion_verifier",
        "runtime_message_classifier",
        "runtime_amendment_validator",
    }
)

PERCEPTION_ROLES = frozenset(
    {
        "attachment_interpreter",
        "scene_interpreter",
        "target_verifier",
    }
)


# 功能：
#   按角色选择安全、感知、审查或主端口；重叠角色先匹配安全职责。
# 输入：
#   role：本次调用的任务角色。
#   requested_port：调用方原请求端口，本策略按职责重新选择。
#   _：调度器传入但本策略不使用的扩展参数。
# 输出：
#   selection：所选端口名称。
def _specialist(*, role: str, requested_port: str, **_: Any) -> dict[str, str]:
    if role in SAFETY_ROLES:
        port = "safety"
    elif role in PERCEPTION_ROLES:
        port = "perception"
    else:
        port = "critic" if role in CRITIC_ROLES else "primary"
    selection = {"port": port}
    return selection


# 功能：
#   将各角色统一交给主端口，保留 Harness 原有的独立审查步骤。
# 输入：
#   _：统一选择不使用的调度参数。
# 输出：
#   selection：主端口选择结果。
def _unified(**_: Any) -> dict[str, str]:
    selection = {"port": "primary"}
    return selection


# 功能：
#   把审查、监控、验收和运行消息分类交给 critic，其余角色保留原端口。
# 输入：
#   role：本次调用的任务角色。
#   requested_port：非审查角色原请求的端口。
#   _：本策略不使用的扩展参数。
# 输出：
#   selection：所选端口名称。
def _adversarial(*, role: str, requested_port: str, **_: Any) -> dict[str, str]:
    review_role = role.endswith("critic") or role in {
        "execution_monitor",
        "completion_verifier",
        "runtime_message_classifier",
    }
    selection = {"port": "critic" if review_role else requested_port}
    return selection


# 功能：
#   注册三种互斥的模型角色分配策略；这些钩子只选端口，不输出飞控指令。
# 输入：
#   无。
# 输出：
#   definitions：角色专用、统一和对抗式分配的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "models.role-specialist",
            "角色专用模型分配",
            "规划使用主端口，审查使用 critic，安全与感知角色使用对应专用端口。",
            _specialist,
            True,
        ),
        (
            "models.role-unified",
            "统一模型分配",
            "所有 Harness 角色使用同一个主模型，适合单一私有模型部署。",
            _unified,
            False,
        ),
        (
            "models.role-adversarial",
            "对抗式审查分配",
            "扩大 critic 端口覆盖范围，让运行消息和最终验收也由审查模型处理。",
            _adversarial,
            False,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.select",
            capability_kind="model-policy",
            capability_name=name,
            capability_description=description,
            category_id="models",
            category_label="模型与推理",
            slot_id="models.role-policy",
            slot_label="模型角色分配",
            activation_mode="single",
            category_order=20,
            slot_order=20,
            plugin_order=index * 10,
            hooks={"select_port": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
        )
        for index, (plugin_id, name, description, handler, enabled) in enumerate(values, start=1)
    ]
    return definitions
