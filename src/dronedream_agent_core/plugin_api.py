"""Runtime-neutral plugin definitions discovered without core tool-name coupling."""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

import jsonschema
from pydantic import ValidationError

from .capability_broker import CapabilityBrokerHostServices
from .contracts import MapAsset, VehicleAsset
from .extensions import ExtensionPlugin, ExtensionRegistry
from .hashing import sha256_json
from .plugin_contracts import (
    PLUGIN_EXTENSION_HOOKS,
    PLUGIN_MCP_CAPABILITY_KINDS,
    PluginCapability,
    PluginManifest,
    PluginSnapshot,
    PluginSnapshotEntry,
)
from .plugin_files import check_plain_plugin_path, verify_plugin_files
from .plugin_process import McpStdioClient, PluginProcessError, resolve_plugin_command
from .plugin_values import plugin_json_value
from .tools import ToolPlugin, ToolRegistry

if TYPE_CHECKING:
    from .capability_broker import CoreCapabilityBroker, ScopedCapabilityBroker


@dataclass(frozen=True)
class ToolEnvironment:
    """Per-mission assets and per-plugin configuration/broker, not global grants."""

    map_graph: MapAsset
    semantic_path: Path
    vehicle_diameter_m: float
    vehicle_height_m: float
    waypoint_hold_seconds: float
    vehicle: VehicleAsset | None = None
    plugin_configuration: dict[str, object] | None = None
    capability_broker: ScopedCapabilityBroker | None = None
    broker_factory: CoreCapabilityBroker | None = None
    # 起飞前允许充分求解；默认仍是飞行中短预算，缺少声明不能自动扩大等待时间。
    planning_phase: Literal["initial", "runtime"] = "runtime"
    # 已在宿主验证来源/时效的定位方差；None 仅表示尚无测量的条件规划。
    planning_localization_variance_m2: float | None = None


ToolFactory = Callable[[ToolEnvironment], list[ToolPlugin]]


@dataclass(frozen=True)
class PluginDefinition:
    """Describe discoverable capabilities; manifest presence alone executes nothing."""

    manifest: PluginManifest
    tool_factory: ToolFactory | None = None
    hooks: dict[str, Callable[..., Any]] | None = None


class PluginDefinitionProvider(Protocol):
    # 功能：
    #   约定内置插件模块向发现器提供一个定义，不在协议中实现执行或授权逻辑。
    # 输入：
    #   无。
    # 输出：
    #   definition：实现方返回的插件定义。
    def __call__(self) -> PluginDefinition: ...


# 功能：
#   复用桌面端的有限、有界值转换，保证独立 Runtime 和桌面钩子的 JSON 输入语义相同。
# 输入：
#   value：准备传给 MCP 的宿主值。
# 输出：
#   converted：隔离可变引用后的 JSON 兼容值。
def _jsonable_mcp_value(value: Any) -> Any:
    converted = plugin_json_value(value)
    return converted


# 功能：
#   在非 Windows 桥接端将 Windows 盘符安装路径映射到 WSL 挂载点，其他路径按本机解释。
# 输入：
#   value：冻结安装目录的路径文本。
# 输出：
#   path：尚未完成归属、存在性和负载校验的本机路径。
def _snapshot_bundle_path(value: str) -> Path:
    if os.name != "nt" and (PureWindowsPath(value).drive or value.startswith("\\\\")):
        from .windows_paths import windows_drive_path_to_wsl

        return Path(windows_drive_path_to_wsl(value))
    path = Path(value)
    return path


# 功能：
#   识别仅用于已完成资产导入的只读历史条目，执行阶段不加载其代码，也不修改冻结快照。
# 输入：
#   entry：已经通过完整快照摘要及清单校验的条目。
# 输出：
#   preparation_only：是否仅包含无运行钩子的只读地图或机型导入能力。
def _preparation_only_asset_entry(entry: PluginSnapshotEntry) -> bool:
    manifest = entry.manifest
    return bool(
        manifest is not None
        and manifest.runtime.kind == "builtin-python"
        and entry.bundle_root is None
        and set(manifest.permissions) <= {"asset.read"}
        and manifest.capabilities
        and all(cap.kind in {"map-importer", "vehicle-importer"}
                and cap.authority == "read" and "extension_hook" not in cap.metadata
                for cap in manifest.capabilities)
    )


# 功能：
#   1. 核对外部快照身份、完整清单、能力集合、可执行入口和当前磁盘负载摘要。
#   2. 本函数只验证内容绑定；安装签名和用户执行许可由上游负责，摘要匹配不代表信任。
# 输入：
#   entry：已经通过完整快照校验的外部插件条目。
# 输出：
#   verified：有效清单与本机安装目录组成的二元组。
def _verified_external_manifest(entry: PluginSnapshotEntry) -> tuple[PluginManifest, Path]:
    manifest = entry.manifest
    if manifest is None or entry.bundle_root is None:
        raise ValueError(f"PLUGIN_SNAPSHOT_EXTERNAL_METADATA_MISSING:{entry.plugin_id}")
    if (
        manifest.plugin_id != entry.plugin_id
        or manifest.version != entry.version
        or sha256_json(manifest) != entry.manifest_sha256
        or [item.capability_id for item in manifest.capabilities] != entry.capability_ids
    ):
        raise ValueError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
    if manifest.runtime.kind != "mcp-stdio":
        raise ValueError(f"PLUGIN_RUNTIME_NOT_REBUILDABLE:{entry.plugin_id}")
    installed_root = _snapshot_bundle_path(entry.bundle_root)
    check_plain_plugin_path(installed_root)
    root = installed_root.resolve()
    if not root.is_dir():
        raise ValueError(f"PLUGIN_SNAPSHOT_BUNDLE_MISSING:{entry.plugin_id}")
    resolve_plugin_command(root, manifest.runtime.command)
    try:
        verify_plugin_files(root, manifest.file_sha256)
    except (OSError, ValueError) as error:
        raise ValueError(f"PLUGIN_SNAPSHOT_FILE_HASH_DRIFT:{entry.plugin_id}") from error
    verified = (manifest, root)
    return verified


# 功能：
#   重验冻结负载，绑定能力级最小权限及宿主代理，经隔离客户端执行并验证输入输出 Schema。
# 输入：
#   root：构建注册表时绑定的安装目录。
#   manifest：构建时绑定的完整清单。
#   entry：当前调用拥有的冻结条目。
#   capability：实际调用能力的权限与结构约束。
#   arguments：已转换的 JSON 输入对象。
#   broker_factory：本任务可用的宿主代理工厂，None 表示不提供宿主服务。
# 输出：
#   output：通过输出 Schema 校验的插件结果。
def _call_external_mcp(
    *,
    root: Path,
    manifest: PluginManifest,
    entry: PluginSnapshotEntry,
    capability: PluginCapability,
    arguments: dict[str, Any],
    broker_factory: CoreCapabilityBroker | None = None,
) -> dict[str, Any]:
    verified_manifest, verified_root = _verified_external_manifest(entry)
    if verified_manifest != manifest or verified_root != root:
        raise PluginProcessError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
    if capability.input_schema:
        jsonschema.validate(arguments, capability.input_schema)
    runtime = manifest.runtime
    permissions = list(
        capability.required_permissions
        if capability.required_permissions is not None
        else manifest.permissions
    )
    host_services = None
    if broker_factory is not None:
        scoped_manifest = manifest.model_copy(update={"permissions": permissions})
        host_services = CapabilityBrokerHostServices(broker_factory.scope(scoped_manifest))
    with McpStdioClient(
        plugin_root=root,
        command=runtime.command,
        protocol_version=runtime.protocol_version,
        startup_timeout_seconds=runtime.startup_timeout_seconds,
        call_timeout_seconds=runtime.call_timeout_seconds,
        configuration=dict(entry.configuration),
        permissions=permissions,
        resource_policy=manifest.resource_policy,
        host_services=host_services,
        require_os_isolation=True,
    ) as client:
        output = client.call_tool(capability.capability_id, arguments)
    if capability.output_schema:
        jsonschema.validate(output, capability.output_schema)
    return output


# 功能：
#   从已经获准加载的 module:attribute 引用取得对象；模块导入会执行代码，不接收用户任意文本。
# 输入：
#   reference：调用方已审核的模块与属性引用。
# 输出：
#   loaded：从已导入模块取得的属性对象。
def load_object(reference: str) -> object:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("PLUGIN_ENTRYPOINT_INVALID")
    module = importlib.import_module(module_name)
    loaded = getattr(module, attribute)
    return loaded


# 功能：
#   从第一方包逐模块发现定义，兼容冻结可执行程序；拒绝错误返回类型及重复插件标识。
# 输入：
#   无。
# 输出：
#   definitions：按稳定插件标识索引的内置定义集合。
def discover_builtin_plugins() -> dict[str, PluginDefinition]:
    package = importlib.import_module("dronedream_agent_plugins")
    definitions: dict[str, PluginDefinition] = {}
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        provider = getattr(module, "plugin_definition", None)
        providers = getattr(module, "plugin_definitions", None)
        if provider is None and providers is None:
            continue
        discovered = [provider()] if provider is not None else list(providers())
        for definition in discovered:
            if not isinstance(definition, PluginDefinition):
                raise TypeError(f"invalid plugin definition from {module_info.name}")
            plugin_id = definition.manifest.plugin_id
            if plugin_id in definitions:
                raise ValueError(f"duplicate discovered plugin id: {plugin_id}")
            definitions[plugin_id] = definition
    return definitions


# 功能：
#   为独立命令行选择默认启用的第一方工具，建立只读、规划和仿真权限注册表及其快照。
# 输入：
#   environment：调用方提供的任务环境。
# 输出：
#   discovered：工具注册表和本次默认选择快照组成的二元组。
def build_discovered_tool_registry(
    environment: ToolEnvironment,
) -> tuple[ToolRegistry, PluginSnapshot]:
    registry = ToolRegistry(allowed_authorities={"read", "plan", "simulate"})
    entries: list[PluginSnapshotEntry] = []
    for plugin_id, definition in sorted(discover_builtin_plugins().items()):
        manifest = definition.manifest
        if not manifest.default_enabled:
            continue
        package_sha256 = sha256_json(manifest)
        entries.append(
            PluginSnapshotEntry(
                plugin_id=plugin_id,
                version=manifest.version,
                package_sha256=package_sha256,
                manifest_sha256=package_sha256,
                configuration_sha256=sha256_json({}),
                configuration={},
                capability_ids=[item.capability_id for item in manifest.capabilities],
                manifest=manifest,
                selection_source="product_managed_default",
                selected_by="product_managed",
            )
        )
        if definition.tool_factory is None:
            continue
        for tool in definition.tool_factory(environment):
            registry.register(
                replace(
                    tool,
                    plugin_id=plugin_id,
                    plugin_package_sha256=package_sha256,
                    slot_id=manifest.placement.slot_id,
                )
            )
    snapshot = PluginSnapshot(
        snapshot_id=f"plugin-snapshot-{uuid4().hex[:24]}",
        catalog_sha256=sha256_json([item.model_dump(mode="json") for item in entries]),
        plugins=entries,
        created_at=datetime.now(UTC),
    )
    discovered = (registry, snapshot)
    return discovered


# 功能：
#   按精确冻结选择重建内外部工具，绑定每个插件自己的配置和宿主权限，并接入对应钩子。
# 输入：
#   environment：本次任务的地图、机体与代理环境。
#   snapshot：上游已准入且需要重新核对内容一致性的插件快照。
# 输出：
#   registry：与冻结身份和配置绑定的工具注册表。
def build_snapshot_tool_registry(
    environment: ToolEnvironment, snapshot: PluginSnapshot
) -> ToolRegistry:
    snapshot = verify_plugin_snapshot(snapshot)
    registry = ToolRegistry(allowed_authorities={"read", "plan", "simulate"})
    definitions = discover_builtin_plugins()
    for entry in snapshot.plugins:
        definition = definitions.get(entry.plugin_id)
        if definition is None:
            if _preparation_only_asset_entry(entry):
                continue
            manifest, root = _verified_external_manifest(entry)
            for capability in manifest.capabilities:
                if (
                    capability.kind not in PLUGIN_MCP_CAPABILITY_KINDS
                    or "extension_hook" in capability.metadata
                ):
                    continue

                # 功能：
                #   为当前循环的外部能力固定身份，在实际调用时转换输入并重新核验负载。
                # 输入：
                #   value：工具输入对象。
                #   _root：闭包绑定的安装目录。
                #   _manifest：闭包绑定的清单。
                #   _entry：闭包绑定的快照条目。
                #   _capability：闭包绑定的能力契约。
                # 输出：
                #   output：实际协议调用的校验后输出。
                def call_external(
                    value: dict[str, object],
                    *,
                    _root: Path = root,
                    _manifest: PluginManifest = manifest,
                    _entry: PluginSnapshotEntry = entry,
                    _capability: PluginCapability = capability,
                ) -> dict[str, Any]:
                    output = _call_external_mcp(
                        root=_root,
                        manifest=_manifest,
                        entry=_entry,
                        capability=_capability,
                        arguments=_jsonable_mcp_value(value),
                        broker_factory=environment.broker_factory,
                    )
                    return output

                registry.register(
                    ToolPlugin(
                        tool_id=capability.capability_id,
                        version=manifest.version,
                        authority=capability.authority,  # type: ignore[arg-type]
                        input_type=None,
                        output_type=None,
                        input_schema=capability.input_schema,
                        output_schema=capability.output_schema,
                        handler=call_external,
                        plugin_id=entry.plugin_id,
                        plugin_package_sha256=entry.package_sha256,
                        routing_metadata=capability.metadata,
                        slot_id=manifest.placement.slot_id,
                    )
                )
            continue
        if definition.tool_factory is None:
            continue
        manifest = definition.manifest
        manifest_sha256 = sha256_json(manifest)
        if (
            manifest.version != entry.version
            or manifest_sha256 != entry.manifest_sha256
            or manifest_sha256 != entry.package_sha256
        ):
            raise ValueError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
        # 恢复本条目的冻结配置，不继承可变界面默认值或其他插件的宿主权限域。
        bound_environment = replace(
            environment,
            plugin_configuration=copy.deepcopy(entry.configuration),
            capability_broker=(
                environment.broker_factory.scope(manifest)
                if environment.broker_factory is not None
                else None
            ),
        )
        for tool in definition.tool_factory(bound_environment):
            registry.register(
                replace(
                    tool,
                    plugin_id=entry.plugin_id,
                    plugin_package_sha256=entry.package_sha256,
                    slot_id=manifest.placement.slot_id,
                )
            )
    registry.configure_extensions(
        build_discovered_extension_registry(
            snapshot,
            broker_factory=environment.broker_factory,
        )
    )
    return registry


# 功能：
#   1. 重新验证当前快照的完整数据契约，并隔离所有可变条目，拒绝重复插件和能力。
#   2. 核对配置、清单与目录摘要；历史快照可作历史证据，但不能构建新的执行注册表。
# 输入：
#   snapshot：准备进入重建链路的快照模型。
# 输出：
#   owned：结构与摘要均有效、与调用方原对象隔离的快照。
def verify_plugin_snapshot(snapshot: PluginSnapshot) -> PluginSnapshot:
    if snapshot.schema_version != "dronedream.plugin-snapshot.v2":
        raise ValueError("PLUGIN_SNAPSHOT_OBSOLETE")
    try:
        # 复制不能代替重新校验：篡改字段后重算摘要仍不能使非法来源或权限合法化。
        owned = PluginSnapshot.model_validate(snapshot.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise ValueError("PLUGIN_SNAPSHOT_INVALID") from error
    identities = [entry.plugin_id for entry in owned.plugins]
    if len(identities) != len(set(identities)):
        raise ValueError("PLUGIN_SNAPSHOT_DUPLICATE_SELECTION")
    for entry in owned.plugins:
        if sha256_json(entry.configuration) != entry.configuration_sha256:
            raise ValueError(f"PLUGIN_SNAPSHOT_CONFIGURATION_DRIFT:{entry.plugin_id}")
        if entry.manifest is not None and (
            entry.manifest.plugin_id != entry.plugin_id
            or entry.manifest.version != entry.version
            or sha256_json(entry.manifest) != entry.manifest_sha256
            or [item.capability_id for item in entry.manifest.capabilities] != entry.capability_ids
        ):
            raise ValueError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
    capabilities = [capability for entry in owned.plugins for capability in entry.capability_ids]
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("PLUGIN_SNAPSHOT_DUPLICATE_CAPABILITY")
    if sha256_json(owned.plugins) != owned.catalog_sha256:
        raise ValueError("PLUGIN_SNAPSHOT_CATALOG_DRIFT")
    return owned


# 功能：
#   1. 重建内置及外部 Harness 非工具能力，有快照时使用精确选择，没有快照时仅取内置默认。
#   2. 各钩子固定其处理器与配置副本，流水线外部结果必须明确包含下一步使用的 value。
# 输入：
#   snapshot：可选的已准入冻结选择。
#   broker_factory：任务允许提供给外部插件的宿主代理工厂。
# 输出：
#   registry：按插槽、顺序、失败策略和切换策略配置的扩展注册表。
def build_discovered_extension_registry(
    snapshot: PluginSnapshot | None = None,
    *,
    broker_factory: CoreCapabilityBroker | None = None,
) -> ExtensionRegistry:
    if snapshot is not None:
        snapshot = verify_plugin_snapshot(snapshot)
    definitions = discover_builtin_plugins()
    selected = None if snapshot is None else {item.plugin_id: item for item in snapshot.plugins}
    registry = ExtensionRegistry()
    for plugin_id, definition in sorted(definitions.items()):
        if not definition.hooks:
            continue
        manifest = definition.manifest
        if selected is None and not manifest.default_enabled:
            continue
        package_sha256 = sha256_json(manifest)
        entry = selected.get(plugin_id) if selected is not None else None
        if selected is not None and entry is None:
            continue
        if entry is not None and (
            entry.version != manifest.version
            or entry.manifest_sha256 != package_sha256
            or entry.package_sha256 != package_sha256
        ):
            raise ValueError(f"PLUGIN_SNAPSHOT_DRIFT:{plugin_id}")
        capability = manifest.capabilities[0]
        placement = manifest.placement
        hooks: dict[str, Callable[..., Any]] = {}
        for hook_name, handler in definition.hooks.items():
            signature = inspect.signature(handler)
            accepts_configuration = "configuration" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )

            # 功能：
            #   调用绑定的内置处理器，每次提供独立配置副本，不允许临时参数覆盖冻结配置。
            # 输入：
            #   _handler：当前循环绑定的实际处理器。
            #   _configuration：处理器拥有的冻结配置。
            #   _accepts_configuration：签名是否接收配置参数。
            #   kwargs：本次钩子的业务输入。
            # 输出：
            #   output：处理器实际返回的业务结果。
            def call_hook(
                *,
                _handler: Callable[..., Any] = handler,
                _configuration: dict[str, Any] = dict(entry.configuration) if entry else {},
                _accepts_configuration: bool = accepts_configuration,
                **kwargs: Any,
            ) -> Any:
                if _accepts_configuration:
                    # 钩子可修改自己的一次性副本，不能影响原快照或下次调用。
                    kwargs["configuration"] = copy.deepcopy(_configuration)
                output = _handler(**kwargs)
                return output

            hooks[hook_name] = call_hook
        registry.register(
            ExtensionPlugin(
                plugin_id=plugin_id,
                version=manifest.version,
                package_sha256=package_sha256,
                capability_id=capability.capability_id,
                slot_id=placement.slot_id,
                activation_mode=placement.activation_mode,
                failure_mode=placement.failure_mode,
                swap_policy=placement.swap_policy,
                pipeline_order=placement.pipeline_order,
                runs_after=tuple(placement.runs_after),
                runs_before=tuple(placement.runs_before),
                hooks=hooks,
            )
        )
    if snapshot is not None:
        for entry in snapshot.plugins:
            if entry.plugin_id in definitions:
                continue
            if _preparation_only_asset_entry(entry):
                continue
            manifest, root = _verified_external_manifest(entry)
            capabilities = [
                item
                for item in manifest.capabilities
                if item.metadata.get("extension_hook") in PLUGIN_EXTENSION_HOOKS
            ]
            if not capabilities:
                continue
            placement = manifest.placement
            for capability in capabilities:
                hook_name = str(capability.metadata["extension_hook"])

                # 功能：
                #   调用冻结外部钩子；流水线只传递明确返回的 value，不把缺失字段猜成成功结果。
                # 输入：
                #   _root：冻结安装目录。
                #   _manifest：冻结完整清单。
                #   _entry：冻结选择条目。
                #   _capability：当前钩子的能力与权限声明。
                #   kwargs：本次钩子的结构化业务输入。
                # 输出：
                #   result：流水线 value 或普通钩子的完整结果对象。
                def call_external_extension(
                    *,
                    _root: Path = root,
                    _manifest: PluginManifest = manifest,
                    _entry: PluginSnapshotEntry = entry,
                    _capability: PluginCapability = capability,
                    **kwargs: Any,
                ) -> Any:
                    output = _call_external_mcp(
                        root=_root,
                        manifest=_manifest,
                        entry=_entry,
                        capability=_capability,
                        arguments=_jsonable_mcp_value(kwargs),
                        broker_factory=broker_factory,
                    )
                    if _manifest.placement.activation_mode == "pipeline":
                        if "value" not in output:
                            raise PluginProcessError("PLUGIN_EXTENSION_PIPELINE_VALUE_MISSING")
                        result = output["value"]
                        return result
                    result = output
                    return result

                registry.register(
                    ExtensionPlugin(
                        plugin_id=entry.plugin_id,
                        version=entry.version,
                        package_sha256=entry.package_sha256,
                        capability_id=capability.capability_id,
                        slot_id=placement.slot_id,
                        activation_mode=placement.activation_mode,
                        failure_mode=placement.failure_mode,
                        swap_policy=placement.swap_policy,
                        pipeline_order=placement.pipeline_order,
                        runs_after=tuple(placement.runs_after),
                        runs_before=tuple(placement.runs_before),
                        hooks={hook_name: call_external_extension},
                    )
                )
    return registry
