from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from dronedream_agent_core.contracts import GraphRoute, RouteClearanceReport
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   判断物理标量是否为区间内有限数值，拒绝布尔值、文本和不能安全转换的大整数。
# 输入：
#   value：待检查的数值。
#   minimum：调用方定义的下界。
#   maximum：调用方定义的上界。
# 输出：
#   accepted：数值有限且位于闭区间内时为 True。
def bounded_number(value: object, minimum: float, maximum: float) -> bool:
    accepted = False
    if type(value) not in (int, float):
        return accepted
    try:
        accepted = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        return accepted
    return accepted


# 功能：
#   从策略配置读取有限数值并验证物理范围，直接调用钩子时也不能绕过界面校验。
# 输入：
#   configured：本次钩子的配置字典。
#   key：需要读取的配置键。
#   default：键缺失时采用的值，仍需通过相同校验。
#   minimum：允许下界。
#   maximum：允许上界。
# 输出：
#   selected：转换成浮点数的有效配置值。
def policy_number(
    configured: dict[str, object],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if not isinstance(configured, dict):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    value = configured.get(key, default)
    if not bounded_number(value, minimum, maximum):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    selected = float(value)
    return selected


# 功能：
#   比较距离与已验收范围，允许微小序列化误差，不接受非有限值、负距离或明显超界。
# 输入：
#   distance_m：需要检查的距离，单位米。
#   qualified_range_m：已验收距离上界，单位米。
# 输出：
#   accepted：满足上界或固定微小浮点容差时为 True。
def within_qualified_range(distance_m: float, qualified_range_m: float) -> bool:
    accepted = (
        bounded_number(distance_m, 0, math.inf)
        and bounded_number(qualified_range_m, 0, math.inf)
        and (
            distance_m <= qualified_range_m
            or math.isclose(distance_m, qualified_range_m, rel_tol=1e-12, abs_tol=1e-6)
        )
    )
    return accepted


# 功能：
#   核对净空报告的接受标记、碰撞内容和路线摘要一致，不替代对选定地图的实际几何检测。
# 输入：
#   clearance：已完成类型重校验的净空报告。
#   route：已完成类型重校验的实际候选路线。
# 输出：
#   accepted：报告无碰撞、净空非负且精确绑定当前路线时为 True。
def clearance_evidence_matches_route(clearance: RouteClearanceReport, route: GraphRoute) -> bool:
    accepted = (
        clearance.accepted is True
        and type(clearance.collision_count) is int
        and clearance.collision_count == 0
        and not clearance.collisions
        and bounded_number(clearance.minimum_clearance_m, 0, math.inf)
        and all(
            bounded_number(value, 0, math.inf) for value in clearance.segment_minimum_clearances_m
        )
        and clearance.route_sha256 == sha256_json(route)
    )
    return accepted


# 功能：
#   读取闭区间内的整数预算，不将布尔值当作计数，也不截断浮点数。
# 输入：
#   configured：策略配置字典。
#   key：预算键。
#   default：键缺失时的默认预算。
#   minimum：允许下界。
#   maximum：允许上界。
# 输出：
#   value：验证通过的原整数预算。
def policy_integer(
    configured: dict[str, object], key: str, default: int, minimum: int, maximum: int
) -> int:
    if not isinstance(configured, dict):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    value = configured.get(key, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    return value


# 功能：
#   严格读取布尔开关，避免非空文本 false 被当成允许或同意。
# 输入：
#   configured：策略配置字典。
#   key：开关键。
#   default：键缺失时的默认布尔值。
# 输出：
#   value：验证通过的布尔开关。
def policy_boolean(configured: dict[str, object], key: str, default: bool) -> bool:
    if not isinstance(configured, dict):
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    value = configured.get(key, default)
    if type(value) is not bool:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    return value


# 功能：
#   合并数量和单项长度受限的建议约束文本，不将任意对象强制转换后放入提示词。
# 输入：
#   value：应为最多 32 条、每条最多 4000 字符的文本列表。
# 输出：
#   text：以空格连接并转为小写的约束文本。
def constraint_text(value: object) -> str:
    if (
        not isinstance(value, list)
        or len(value) > 32
        or any(not isinstance(item, str) or len(item) > 4000 for item in value)
    ):
        raise ValueError("MISSION_ADVISOR_CONSTRAINTS_INVALID")
    text = " ".join(value).lower()
    return text


# 功能：
#   1. 创建第一方规划钩子的完整清单，明确插槽、失败、替换、顺序和最小权限规则。
#   2. 通用对象 Schema 只约束封装，具体业务语义仍由 Harness 阶段校验，不授予执行器权限。
# 输入：
#   module_name：内置定义所在模块。
#   plugin_id：稳定插件标识。
#   name：插件展示名称。
#   description：插件功能描述。
#   capability_id：能力标识。
#   capability_kind：能力类别。
#   capability_name：能力展示名称。
#   capability_description：能力描述。
#   category_id：目录类别标识。
#   category_label：目录类别名称。
#   slot_id：运行插槽标识。
#   slot_label：插槽展示名称。
#   activation_mode：单选、多选或流水线激活方式。
#   category_order：目录类别排序。
#   slot_order：插槽展示排序。
#   plugin_order：同插槽插件展示排序。
#   hooks：钩子名称到实际处理函数的映射。
#   default_enabled：是否作为产品默认选择。
#   failure_mode：隔离失败或失败关闭策略。
#   swap_policy：运行时切换策略。
#   pipeline_order：流水线并列节点排序权重。
#   runs_after：应先于本插件执行的标识列表。
#   runs_before：应后于本插件执行的标识列表。
#   metadata：能力附加元数据。
#   configuration_schema：本插件配置的结构约束。
#   permissions：明确权限列表；None 使用 mission.read 默认，空列表保持无权限。
#   scope：能力生效范围。
#   disable_allowed：是否允许用户停用。
# 输出：
#   definition：包含完整清单和钩子函数的第一方插件定义。
def hook_plugin(
    *,
    module_name: str,
    plugin_id: str,
    name: str,
    description: str,
    capability_id: str,
    capability_kind: str,
    capability_name: str,
    capability_description: str,
    category_id: str,
    category_label: str,
    slot_id: str,
    slot_label: str,
    activation_mode: str,
    category_order: int,
    slot_order: int,
    plugin_order: int,
    hooks: dict[str, Callable[..., Any]],
    default_enabled: bool = False,
    failure_mode: str = "isolate",
    swap_policy: str = "next-mission",
    pipeline_order: int = 500,
    runs_after: list[str] | None = None,
    runs_before: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    configuration_schema: dict[str, Any] | None = None,
    permissions: list[str] | None = None,
    scope: str = "mission",
    disable_allowed: bool = True,
) -> PluginDefinition:
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id=plugin_id,
            name=name,
            version="1.0.0",
            description=description,
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{module_name}:plugin_definitions"
            ),
            capabilities=[
                PluginCapability(
                    capability_id=capability_id,
                    kind=capability_kind,  # type: ignore[arg-type]
                    name=capability_name,
                    description=capability_description,
                    authority="plan",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object", "additionalProperties": True},
                    metadata=metadata or {},
                )
            ],
            # 只有未指定权限时使用默认读取许可；显式空列表不能获得任何默认权限。
            permissions=(["mission.read"] if permissions is None else permissions),  # type: ignore[arg-type]
            default_enabled=default_enabled,
            removable=False,
            disable_allowed=disable_allowed,
            placement=PluginPlacement(
                category_id=category_id,
                category_label=category_label,
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode=activation_mode,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
                failure_mode=failure_mode,  # type: ignore[arg-type]
                swap_policy=swap_policy,  # type: ignore[arg-type]
                category_order=category_order,
                slot_order=slot_order,
                plugin_order=plugin_order,
                pipeline_order=pipeline_order,
                runs_after=runs_after or [],
                runs_before=runs_before or [],
            ),
            configuration_schema=configuration_schema or {},
        ),
        hooks=hooks,
    )
    return definition
