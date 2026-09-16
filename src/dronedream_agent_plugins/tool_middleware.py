from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin

SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "password", "secret")


# 功能：
#   递归拒绝名称含密钥、密码或认证标记的字段，不把此检查宣称为自由文本秘密识别。
# 输入：
#   value：宿主已完成类型与复杂度预算检查的工具输入。
#   _：本过滤器不使用的工具身份等扩展参数。
# 输出：
#   value：通过字段名称检查的原输入。
def _secret_guard(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    # 功能：
    #   遍历对象和数组并累积字段路径，只在错误中报告字段位置，不输出对应秘密值。
    # 输入：
    #   current：当前待检查的子值。
    #   path：从根对象到当前值的字段或索引路径。
    # 输出：
    #   None：不返回业务数据。
    def walk(current: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(current, dict):
            for key, item in current.items():
                normalized = str(key).casefold().replace("-", "_")
                if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                    raise ValueError("TOOL_ARGUMENT_CONTAINS_SECRET:" + ".".join((*path, str(key))))
                walk(item, (*path, str(key)))
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                walk(item, (*path, str(index)))

    walk(value)
    return value


# 功能：
#   在具体工具校验前拒绝 NaN 和无穷浮点数；有限并不代表满足物理范围或动作安全。
# 输入：
#   value：宿主已限制复杂度的工具输入。
#   _：本过滤器不使用的扩展参数。
# 输出：
#   value：所有浮点叶子均有限的原输入。
def _finite_number_guard(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    # 功能：
    #   逐层检查字典值与数组元素，不改变数值或用零替换未知测量。
    # 输入：
    #   current：当前子树或标量。
    # 输出：
    #   None：不返回业务数据。
    def walk(current: object) -> None:
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError("TOOL_ARGUMENT_NON_FINITE")
        if isinstance(current, dict):
            for item in current.values():
                walk(item)
        elif isinstance(current, (list, tuple)):
            for item in current:
                walk(item)

    walk(value)
    return value


# 功能：
#   拒绝顶层以下划线命名的内部字段进入工具公开结果与缓存，不对任意嵌套字段脱敏。
# 输入：
#   value：准备交付的工具输出对象。
#   _：本过滤器不使用的扩展参数。
# 输出：
#   value：没有顶层私有字段的原输出。
def _output_guard(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    if any(str(key).startswith("_") for key in value):
        raise ValueError("TOOL_OUTPUT_PRIVATE_FIELD")
    return value


# 功能：
#   声明按顺序运行的输入密钥、有限数值与输出边界过滤器，失败必须阻止当前调用。
# 输入：
#   无。
# 输出：
#   definitions：默认启用且只在下一任务切换的管线插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "tools.middleware-secret-guard",
            "工具密钥隔离",
            "拒绝把 API Key、密码、Authorization 或 secret 字段传给任务工具。",
            "before_tool_call",
            _secret_guard,
            10,
        ),
        (
            "tools.middleware-finite-numbers",
            "有限数值检查",
            "拒绝 NaN 与无穷数进入导航、仿真和评测工具。",
            "before_tool_call",
            _finite_number_guard,
            20,
        ),
        (
            "tools.middleware-output-boundary",
            "工具输出边界",
            "拒绝工具输出以下划线开头的内部字段。",
            "after_tool_call",
            _output_guard,
            30,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.filter",
            capability_kind="tool-middleware",
            capability_name=name,
            capability_description=description,
            category_id="tools",
            category_label="工具与集成",
            slot_id="tools.middleware",
            slot_label="工具调用管线",
            activation_mode="pipeline",
            category_order=50,
            slot_order=30,
            plugin_order=order,
            pipeline_order=order,
            hooks={hook: handler},
            default_enabled=True,
            failure_mode="fail-closed",
            swap_policy="next-mission",
        )
        for plugin_id, name, description, hook, handler, order in values
    ]
    return definitions
