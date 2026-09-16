from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   比对模型输出的角色、Schema 名称和内容摘要绑定，结构字段语义由模型端口校验。
# 输入：
#   value：包含 artifact 与 record 的输出信封。
#   role：宿主期望的调用角色。
#   expected_schema：宿主期望的结构类型名称。
#   _：本检查器不使用的扩展参数。
# 输出：
#   value：通过绑定校验的原始输出信封。
def _provenance_guard(
    *, value: dict[str, object], role: str, expected_schema: str, **_: Any
) -> dict[str, object]:
    artifact = value.get("artifact")
    record = value.get("record")
    if not isinstance(artifact, dict) or not isinstance(record, dict):
        raise ValueError("MODEL_OUTPUT_ENVELOPE_INVALID")
    if record.get("role") != role:
        raise ValueError("MODEL_OUTPUT_ROLE_MISMATCH")
    if record.get("output_schema") != expected_schema:
        raise ValueError("MODEL_OUTPUT_SCHEMA_BINDING_MISMATCH")
    if record.get("output_sha256") != sha256_json(artifact):
        raise ValueError("MODEL_OUTPUT_HASH_MISMATCH")
    return value


# 功能：
#   检查输出材料的嵌套数值，拒绝 NaN 和无穷数进入规划工具链。
# 输入：
#   value：包含待检查 artifact 的输出信封。
#   _：本检查器不使用的扩展参数。
# 输出：
#   value：数值检查通过的原始输出信封。
def _finite_value_guard(*, value: dict[str, object], **_: Any) -> dict[str, object]:
    # 功能：
    #   遍历字典及序列，在非有限浮点节点处报告对应字段路径。
    # 输入：
    #   item：当前检查的节点。
    #   path：用于定位非法值的字段与索引路径。
    # 输出：
    #   None：不返回业务数据。
    def validate(item: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("MODEL_OUTPUT_NON_FINITE:" + ".".join(path))
        if isinstance(item, dict):
            for key, child in item.items():
                validate(child, (*path, str(key)))
        # 本地钩子可能返回会被编码成 JSON 数组的元组，不能仅检查外部解码后的列表。
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                validate(child, (*path, str(index)))

    validate(value.get("artifact"), ("artifact",))
    return value


# 功能：
#   注册先绑定、后数值检查的有序输出门，任一检查失败都阻断后续处理。
# 输入：
#   无。
# 输出：
#   definitions：溯源与有限数值输出检查的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "models.output-provenance-guard",
            "结构化输出溯源门",
            "校验模型角色、目标 Schema 和输出哈希绑定，拒绝错位或被修改的结构化结果。",
            _provenance_guard,
            10,
        ),
        (
            "models.output-finite-value-guard",
            "有限数值输出门",
            "递归拒绝 NaN 和无穷数，防止异常数值进入规划与飞控转换链路。",
            _finite_value_guard,
            20,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.validate",
            capability_kind="structured-decoder",
            capability_name=name,
            capability_description=description,
            category_id="models",
            category_label="模型与推理",
            slot_id="models.structured-output-guards",
            slot_label="结构化输出门",
            activation_mode="pipeline",
            category_order=20,
            slot_order=30,
            plugin_order=order,
            pipeline_order=order,
            hooks={"validate_output": handler},
            default_enabled=True,
            failure_mode="fail-closed",
            permissions=["mission.read"],
        )
        for plugin_id, name, description, handler, order in values
    ]
    return definitions
