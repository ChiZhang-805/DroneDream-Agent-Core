"""Additive qualification checks for externally sourced map and vehicle pairs.

These plugins make the qualification suite extensible while the immutable core
still owns runtime gates, promotion, safety authorization, and evidence hashes.
"""

from __future__ import annotations

import re
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import hook_plugin


# 功能：
#   去重问题码并复制证据明细；附加检查通过不代表核心已经批准资产提升。
# 输入：
#   check_id：本次检查的稳定标识。
#   plugin_id：执行该检查的插件标识。
#   capability_id：对应能力标识。
#   issues：按发现顺序记录的问题码。
#   details：需要保存的有界 JSON 证据明细。
# 输出：
#   verdict：检查身份、通过标记、问题码和独立证据明细。
def _verdict(
    *,
    check_id: str,
    plugin_id: str,
    capability_id: str,
    issues: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    unique = list(dict.fromkeys(issues))
    verdict = {
        "check_id": check_id,
        "plugin_id": plugin_id,
        "capability_id": capability_id,
        "accepted": not unique,
        "issue_codes": unique,
        "details": copy_json(details),
    }
    return verdict


# 功能：
#   检查接口或门控名称是非空且数量受限的字符串列表，避免空集合的 all() 被当作通过。
# 输入：
#   value：待检查的名称列表。
#   maximum：当前接口允许的最多名称数量。
# 输出：
#   accepted：列表形状、数量及所有名称是否有效。
def _string_list(value: object, maximum: int) -> bool:
    accepted = (
        isinstance(value, list)
        and 1 <= len(value) <= maximum
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )
    return accepted


# 功能：
#   核对碰撞几何数量符合资产 IR 的整数范围，不把布尔值或数字字符串当作几何证据。
# 输入：
#   value：资产几何声明。
# 输出：
#   present：是否声明了合法的正整数碰撞几何数量。
def _collision_geometry_present(value: object) -> bool:
    count = value.get("collision_count") if isinstance(value, dict) else None
    present = type(count) is int and 1 <= count <= 1_000_000
    return present


# 功能：
#   提取完整合法的仿真目标集合；任一损坏项使整个集合无效，不能靠其余项掩盖错误。
# 输入：
#   value：最多三十二项的仿真目标声明。
# 输出：
#   result：有效仿真器名称集合，输入无效时为空集合。
def _simulators(value: object) -> set[str]:
    result: set[str] = set()
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        return result
    for item in value:
        name = item.get("simulator") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip():
            result.clear()
            return result
        result.add(name)
    return result


# 功能：
#   检查地图与无人机的来源声明及哈希格式；源文件字节与摘要绑定仍由核心独立核验。
# 输入：
#   map_asset_ir：地图资产的中间表示。
#   vehicle_asset_ir：无人机资产的中间表示。
#   _：本检查不使用的其余资格参数。
# 输出：
#   verdict：来源声明检查结果及提取出的来源字段。
def check_source_provenance(
    *,
    map_asset_ir: dict[str, Any],
    vehicle_asset_ir: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    issues: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    for role, value in (("map", map_asset_ir), ("vehicle", vehicle_asset_ir)):
        source = value.get("source")
        if not isinstance(source, dict):
            issues.append(f"ASSET_{role.upper()}_SOURCE_PROVENANCE_MISSING")
            continue
        sources[role] = {
            "adapter_id": source.get("adapter_id"),
            "source_format": source.get("source_format"),
            "source_sha256": source.get("source_sha256"),
        }
        if not all(
            isinstance(sources[role].get(key), str) and sources[role][key].strip()
            for key in sources[role]
        ):
            issues.append(f"ASSET_{role.upper()}_SOURCE_PROVENANCE_INCOMPLETE")
        digest = source.get("source_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            issues.append(f"ASSET_{role.upper()}_SOURCE_HASH_INVALID")
    verdict = _verdict(
        check_id="source-provenance",
        plugin_id="dronedream.asset-source-provenance",
        capability_id="asset-source-provenance-check",
        issues=issues,
        details={"sources": sources},
    )
    return verdict


# 功能：
#   拒绝缺少碰撞几何、质量或惯量的物理声明；声明完整不代替实际运行验收。
# 输入：
#   map_asset_ir：地图资产的中间表示。
#   vehicle_asset_ir：无人机资产的中间表示。
#   _：本检查不使用的其余资格参数。
# 输出：
#   verdict：物理声明检查结果、几何计数及物理明细副本。
def check_physics_completeness(
    *,
    map_asset_ir: dict[str, Any],
    vehicle_asset_ir: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    issues: list[str] = []
    map_geometry = map_asset_ir.get("geometry")
    vehicle_geometry = vehicle_asset_ir.get("geometry")
    vehicle_physics = vehicle_asset_ir.get("physics")
    if not _collision_geometry_present(map_geometry):
        issues.append("ASSET_MAP_COLLISION_GEOMETRY_MISSING")
    if not _collision_geometry_present(vehicle_geometry):
        issues.append("ASSET_VEHICLE_COLLISION_GEOMETRY_MISSING")
    if not isinstance(vehicle_physics, dict):
        issues.append("ASSET_VEHICLE_PHYSICS_MISSING")
        vehicle_physics = {}
    for field, code in (
        ("collision_complete", "ASSET_VEHICLE_COLLISION_INCOMPLETE"),
        ("mass_complete", "ASSET_VEHICLE_MASS_INCOMPLETE"),
        ("inertia_complete", "ASSET_VEHICLE_INERTIA_INCOMPLETE"),
    ):
        if vehicle_physics.get(field) is not True:
            issues.append(code)
    verdict = _verdict(
        check_id="physics-completeness",
        plugin_id="dronedream.asset-physics-completeness",
        capability_id="asset-physics-completeness-check",
        issues=issues,
        details={
            "map_collision_count": map_geometry.get("collision_count", 0)
            if isinstance(map_geometry, dict)
            else 0,
            "vehicle_collision_count": vehicle_geometry.get("collision_count", 0)
            if isinstance(vehicle_geometry, dict)
            else 0,
            "vehicle_physics": vehicle_physics,
        },
    )
    return verdict


# 功能：
#   要求仿真器交集、导航与飞控接口、环境身份，以及逐项明确为真的必要运行门控。
# 输入：
#   plan：包含必要运行门控名称的资格计划。
#   map_asset_ir：地图资产的中间表示。
#   vehicle_asset_ir：无人机资产的中间表示。
#   runtime_evidence：上游绑定的运行证据。
#   environment_versions：非空的运行组件与版本映射。
#   _：接口检查不使用的扩展参数。
# 输出：
#   verdict：兼容性检查结果及共同仿真器、环境和门控名称。
def check_runtime_interfaces(
    *,
    plan: dict[str, Any],
    map_asset_ir: dict[str, Any],
    vehicle_asset_ir: dict[str, Any],
    runtime_evidence: dict[str, Any],
    environment_versions: dict[str, str],
    **_: Any,
) -> dict[str, Any]:
    issues: list[str] = []
    map_targets = _simulators(map_asset_ir.get("simulation_targets"))
    vehicle_targets = _simulators(vehicle_asset_ir.get("simulation_targets"))
    common_targets = sorted(map_targets & vehicle_targets)
    if not common_targets:
        issues.append("ASSET_PAIR_SIMULATOR_INTERFACE_MISMATCH")
    map_semantics = map_asset_ir.get("semantics")
    navigation_path = (
        map_semantics.get("navigation_graph_path") if isinstance(map_semantics, dict) else None
    )
    if not isinstance(navigation_path, str) or not navigation_path.strip():
        issues.append("ASSET_MAP_NAVIGATION_GRAPH_MISSING")
    vehicle_interfaces = vehicle_asset_ir.get("interfaces")
    if not isinstance(vehicle_interfaces, dict) or not _string_list(
        vehicle_interfaces.get("px4_profiles"), 64
    ):
        issues.append("ASSET_VEHICLE_FLIGHT_PROFILE_MISSING")
    gates = runtime_evidence.get("gates")
    required_gates = plan.get("required_runtime_gates", [])
    if (
        not _string_list(required_gates, 64)
        or not isinstance(gates, dict)
        or any(gates.get(gate) is not True for gate in required_gates)
    ):
        issues.append("ASSET_RUNTIME_GATE_SET_INCOMPLETE")
    if (
        not isinstance(environment_versions, dict)
        or not environment_versions
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in environment_versions.items()
        )
    ):
        issues.append("ASSET_RUNTIME_ENVIRONMENT_UNBOUND")
    verdict = _verdict(
        check_id="runtime-interface-compatibility",
        plugin_id="dronedream.asset-runtime-interfaces",
        capability_id="asset-runtime-interface-check",
        issues=issues,
        details={
            "common_simulators": common_targets,
            "environment_versions": environment_versions,
            "required_runtime_gates": required_gates,
        },
    )
    return verdict


# 功能：
#   创建只读附加资格插件；失败必须阻止本检查通过，不能放宽不可替换的核心门控。
# 输入：
#   plugin_id：插件唯一标识。
#   name：展示名称。
#   description：检查职责说明。
#   capability_id：能力唯一标识。
#   order：同类检查的展示顺序。
#   hook：执行实际附加检查的函数。
# 输出：
#   definition：带权限、失败策略及认证更新要求的插件定义。
def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    capability_id: str,
    order: int,
    hook: Any,
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=capability_id,
        capability_kind="qualification-check",
        capability_name=name,
        capability_description=description,
        category_id="assets",
        category_label="资产与资格认证",
        slot_id="assets.qualification-checks",
        slot_label="附加资格认证检查",
        activation_mode="multiple",
        category_order=760,
        slot_order=740,
        plugin_order=order,
        hooks={"qualify_asset": hook},
        default_enabled=True,
        failure_mode="fail-closed",
        swap_policy="certified-update",
        scope="general",
        permissions=["asset.read"],
    )
    return definition


# 功能：
#   注册来源、物理及接口三项附加检查，不另建可绕过核心资格链的批准入口。
# 输入：
#   无。
# 输出：
#   definitions：三个资格检查插件的定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _definition(
            plugin_id="dronedream.asset-source-provenance",
            name="资产来源链检查",
            description="核对地图与无人机资产的适配器、源格式和内容哈希来源。",
            capability_id="asset-source-provenance-check",
            order=100,
            hook=check_source_provenance,
        ),
        _definition(
            plugin_id="dronedream.asset-physics-completeness",
            name="物理完整性检查",
            description="检查碰撞几何、质量和惯量等仿真所需的物理声明。",
            capability_id="asset-physics-completeness-check",
            order=200,
            hook=check_physics_completeness,
        ),
        _definition(
            plugin_id="dronedream.asset-runtime-interfaces",
            name="运行时接口兼容性检查",
            description="核对地图、无人机、飞控和仿真器接口与运行证据。",
            capability_id="asset-runtime-interface-check",
            order=300,
            hook=check_runtime_interfaces,
        ),
    ]
    return definitions
