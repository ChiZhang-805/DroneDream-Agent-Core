"""Role-scoped prompt reminders; code-enforced schemas and safety gates remain authoritative."""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   冻结适用角色集合，构造仅向目标角色补充提示的钩子，不赋予任何工具权限。
# 输入：
#   fragment：需要追加的提示片段。
#   roles：允许追加的角色集合，None 表示所有角色。
# 输出：
#   augment：保存提示片段与角色快照的调用钩子。
def _append(fragment: str, *, roles: set[str] | None = None):
    selected_roles = frozenset(roles) if roles is not None else None

    # 功能：
    #   对匹配角色追加提示片段，其他角色保持输入文本原样。
    # 输入：
    #   value：当前提示文本。
    #   role：本次模型调用的角色。
    #   _：提示钩子不使用的扩展参数。
    # 输出：
    #   prompt：保留原文或已追加片段的提示文本。
    def augment(*, value: str, role: str, **_: Any) -> str:
        if selected_roles is not None and role not in selected_roles:
            prompt = value
        else:
            prompt = f"{value.rstrip()}\n\nPLUGIN PROMPT PACK:\n{fragment.strip()}\n"
        return prompt

    return augment


# 功能：
#   构建结构、当前地图、审查、载荷、简洁表达和运行稳定性的有序角色提示扩展。
# 输入：
#   无。
# 输出：
#   definitions：包含角色范围、默认启用状态和顺序的提示插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    # 提示约束用于指导模型；真正的权限、Schema 和执行安全仍由代码门禁判断。
    values = [
        (
            "prompt.structured-discipline",
            "结构化输出纪律",
            "强调只填写目标 Schema、保留未知值并避免把推测写成事实。",
            "Return only the requested structured artifact. Preserve unknown values as unknown; "
            "never invent telemetry, map entities, tool results, permissions, or evidence.",
            None,
            True,
        ),
        (
            "prompt.campus-grounding",
            "当前地图落地",
            "要求所有位置、路线和任务动作绑定地图实体及节点。",
            "Ground every location in the supplied map catalog. Never create an entity, "
            "node, doorway, floor, road, pickup point, or landing site absent from that catalog.",
            {"intent_parser", "intent_critic", "task_decomposer", "global_planner", "plan_critic"},
            True,
        ),
        (
            "prompt.adversarial-review",
            "对抗式审查",
            "让 critic 主动寻找遗漏约束、错误假设和证据不足。",
            "Act as an independent adversarial reviewer. Search for omitted user constraints, "
            "identity confusion, stale-plan assumptions, unsafe side effects, and evidence gaps.",
            {"intent_critic", "plan_critic", "completion_verifier"},
            True,
        ),
        (
            "prompt.payload-custody",
            "载荷交接与保管",
            "强化取件身份、抓取确认、质量变化和返程载荷状态。",
            "For payload missions, keep pickup identity, pre-contact hold, attachment evidence, "
            "mass/inertia update, custody state, and return authorization as explicit steps.",
            {"intent_parser", "task_decomposer", "global_planner", "plan_critic"},
            False,
        ),
        (
            "prompt.operator-concise",
            "操作员简洁输出",
            "在不减少结构化字段的前提下压缩面向用户的解释。",
            "Keep human-facing summaries short and operational. Do not omit any required "
            "structured field, gate, issue code, repair instruction, or evidence reference.",
            None,
            False,
        ),
        (
            "prompt.runtime-stability",
            "运行期稳定优先",
            "在检查点和中途指令处理中优先稳定悬停、冻结副作用与证据绑定。",
            "During execution, treat stable hold, inhibited side effects, execution identity, "
            "telemetry freshness, deterministic gates, and replacement-track adoption evidence "
            "as mandatory. Never infer that motion may resume from model confidence alone.",
            {
                "execution_monitor",
                "runtime_message_classifier",
                "completion_verifier",
            },
            True,
        ),
    ]
    definitions: list[PluginDefinition] = []
    for index, (plugin_id, name, description, fragment, roles, enabled) in enumerate(
        values, start=1
    ):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.augment",
                capability_kind="prompt-pack",
                capability_name=name,
                capability_description=description,
                category_id="models",
                category_label="模型与推理",
                slot_id="models.prompt-packs",
                slot_label="Prompt 扩展管线",
                activation_mode="pipeline",
                category_order=20,
                slot_order=30,
                plugin_order=index * 10,
                pipeline_order=index * 10,
                hooks={"augment_prompt": _append(fragment, roles=roles)},
                default_enabled=enabled,
                failure_mode="isolate",
            )
        )
    return definitions
