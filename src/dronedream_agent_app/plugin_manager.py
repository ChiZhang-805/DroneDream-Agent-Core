"""Transactional, reversible plugin management for the desktop application."""

from __future__ import annotations

import copy
import hashlib
import inspect
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import jsonschema
from pydantic import ValidationError

from dronedream_agent_core.asset_package_storage import copy_asset_source, publish_asset_file
from dronedream_agent_core.asset_packages import (
    MAX_PACKAGE_UNCOMPRESSED_BYTES,
    AssetKind,
    inspect_ddpkg,
)
from dronedream_agent_core.asset_pair_qualification import AssetQualificationPluginCheck
from dronedream_agent_core.asset_source_adapters import (
    AssetSourceAdapterDescriptor,
    AssetSourceAdapterError,
    AssetSourceDetection,
)
from dronedream_agent_core.asset_source_adapters import (
    detect_asset_source as detect_builtin_asset_source,
)
from dronedream_agent_core.asset_source_adapters import (
    source_adapter_catalog as builtin_source_adapter_catalog,
)
from dronedream_agent_core.capability_broker import (
    CapabilityBrokerHostServices,
    CoreCapabilityBroker,
)
from dronedream_agent_core.extensions import ExtensionPlugin, ExtensionRegistry
from dronedream_agent_core.hashing import canonical_json, sha256_json
from dronedream_agent_core.plugin_api import (
    PluginDefinition,
    ToolEnvironment,
    discover_builtin_plugins,
    verify_plugin_snapshot,
)
from dronedream_agent_core.plugin_contracts import (
    PLUGIN_EXTENSION_HOOKS,
    PLUGIN_MCP_CAPABILITY_KINDS,
    PluginCapability,
    PluginGovernanceOperation,
    PluginGovernancePolicy,
    PluginLifecycleReceipt,
    PluginManifest,
    PluginSnapshot,
    PluginSnapshotEntry,
    PluginUsageEvent,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
    verify_plugin_files,
)
from dronedream_agent_core.plugin_governance import evaluate_plugin_governance
from dronedream_agent_core.plugin_panels import materialize_panel, validate_panel_document
from dronedream_agent_core.plugin_process import (
    McpSessionFailure,
    McpSessionPool,
    McpStdioClient,
    PluginProcessError,
    resolve_plugin_command,
)
from dronedream_agent_core.plugin_trust import PluginTrustStore, TrustStoreError
from dronedream_agent_core.plugin_values import plugin_json_value
from dronedream_agent_core.plugin_versions import version_matches, version_precedence
from dronedream_agent_core.tools import ToolPlugin, ToolRegistry
from dronedream_agent_core.zip_index import validate_zip_index
from dronedream_plugin_sdk.protocol import copy_json, decode_json

from .storage import AppStore


class PluginManagerError(RuntimeError):
    """A plugin lifecycle transition was rejected without weakening the core."""


# 功能：
#   使用共享有界读取器计算普通输出文件摘要，拒绝链接、过大文件或读取期间的变化。
# 输入：
#   path：转换器输出文件路径。
# 输出：
#   checksum：实际读取字节的 SHA-256 摘要。
def _file_sha256(path: Path) -> str:
    checksum = hash_plugin_file(path, limit=MAX_PACKAGE_UNCOMPRESSED_BYTES)
    return checksum


# 功能：
#   复用清单校验的版本匹配规则，不另设宽松的旧版本解析入口。
# 输入：
#   version：已安装版本。
#   requirement：依赖要求。
# 输出：
#   matched：该版本是否满足依赖约束。
def _version_matches(version: str, requirement: str) -> bool:
    matched = version_matches(version, requirement)
    return matched


# 功能：
#   用共享编码器把钩子值转成有限 JSON 数据，将转换失败映射为管理器错误。
# 输入：
#   value：准备传给外部插件或核算字节数的值。
# 输出：
#   converted：与输入引用隔离的 JSON 兼容值。
def _mcp_json_value(value: Any) -> Any:
    try:
        converted = plugin_json_value(value)
        return converted
    except ValueError as error:
        raise PluginManagerError(str(error)) from error


# 功能：
#   提取安全错误标识，不把 Schema 实例内容、本地路径或凭据写入插件故障记录。
# 输入：
#   error：调用或校验产生的异常。
# 输出：
#   issue：稳定错误代码或不包含异常正文的类型名。
def _failure_issue(error: BaseException) -> str:
    if isinstance(error, jsonschema.ValidationError):
        issue = "PLUGIN_SCHEMA_VALIDATION_FAILED"
        return issue
    candidate = str(error).split(":", 1)[0]
    if 1 <= len(candidate) <= 96 and all(
        char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for char in candidate
    ):
        issue = candidate
        return issue
    issue = type(error).__name__
    return issue


class PluginManager:
    required_tool_slots = frozenset(
        {
            "planning.route-strategy",
            "planning.alternative-ranker",
            "safety.route-clearance",
            "flight-control.track-export",
            "runtime.track-export",
        }
    )
    required_single_slots = frozenset(
        {
            *required_tool_slots,
            "safety.interruption-policy",
            "simulation.runtime-adapter",
            "simulation.simulator-descriptor",
            "simulation.clock-policy",
            "simulation.monte-carlo-policy",
            "runtime.track-export",
            "planning.workflow-policy",
            "harness.profile",
            "harness.workflow-topology",
            "harness.scheduler",
            "harness.retry-policy",
            "harness.timeout-policy",
            "harness.budget-policy",
            "harness.fallback-policy",
            "harness.cache-policy",
            "harness.event-bus",
            "models.role-policy",
            "models.runtime-router",
            "models.consensus-policy",
            "context.compaction-strategy",
            "context.store",
            "context.retrieval-policy",
            "context.summarization-policy",
            "context.retention-policy",
            "tools.router-policy",
            "tools.execution-policy",
            "runtime.checkpoint-policy",
            "runtime.replan-policy",
            "runtime.amendment-classifier",
            "runtime.amendment-policy",
            "native.transport",
            "native.state-estimator",
            "native.localization",
            "native.controller",
            "native.watchdog",
        }
    )
    protected_slots = frozenset(
        {
            *required_tool_slots,
            "safety.interruption-policy",
            "simulation.runtime-adapter",
        }
    )

    # 功能：
    #   初始化内置目录、信任库和会话池，保留用户选择；初始化失败时关闭已创建会话池。
    # 输入：
    #   store：调用方拥有的应用数据库和插件目录。
    #   app_version：用于兼容性校验的软件版本。
    #   official_plugins_root：可选的安装器提供的官方插件目录。
    #   plugin_isolator_path：可选的受信任进程隔离器路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        store: AppStore,
        *,
        app_version: str = "1.0.0",
        official_plugins_root: Path | None = None,
        plugin_isolator_path: Path | None = None,
    ) -> None:
        self.store = store
        self.app_version = app_version
        self._definitions = discover_builtin_plugins()
        self._plugin_isolator_path = plugin_isolator_path
        self._disable_guard: Callable[[str, str], list[str]] | None = None
        self._mutation_lock = threading.RLock()
        self._call_condition = threading.Condition()
        self._inflight_calls: dict[str, int] = {}
        self._official_packages: dict[tuple[str, str], str] = {}
        self.trust_store = PluginTrustStore(store.root / "trust" / "plugin-publishers.json")
        self._mcp_sessions = McpSessionPool(on_unhealthy=self._quarantine_unhealthy_session)
        try:
            self.seed_builtin_plugins()
            if official_plugins_root is not None:
                self.seed_official_plugins(official_plugins_root)
        except BaseException:
            # 初始化失败后不能让后台探测继续持有半初始化管理器和调用方数据库。
            self._mcp_sessions.close()
            raise

    # 功能：
    #   安装任务协调器的退出或替换检查，不改变插件权限与安全限制。
    # 输入：
    #   guard：接收插件标识和替换策略、返回受影响任务标识的回调。
    # 输出：
    #   None：不返回业务数据。
    def set_disable_guard(self, guard: Callable[[str, str], list[str]]) -> None:
        self._disable_guard = guard

    # 功能：
    #   回收插件会话；调用方必须在关闭共享数据库前完成这一步。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._mcp_sessions.close()

    # 功能：
    #   只隔离仍匹配失败包和配置的当前启用实例，忽略旧会话迟到报告并原子保存回执。
    # 输入：
    #   failure：包含包摘要、配置摘要与错误标识的会话故障。
    # 输出：
    #   None：不返回业务数据。
    def _quarantine_unhealthy_session(self, failure: McpSessionFailure) -> None:
        try:
            with self._mutation_lock, self.store.atomic_plugin_changes():
                plugin = self.store.get_plugin(failure.plugin_id)
                configuration = self.store.get_plugin_configuration(failure.plugin_id)[
                    "configuration"
                ]
                if (
                    not bool(plugin["enabled"])
                    or plugin["status"] != "healthy"
                    or plugin["package_sha256"] != failure.package_sha256
                    or sha256_json(configuration) != failure.configuration_sha256
                ):
                    return
                failed = self.store.set_plugin_lifecycle(
                    failure.plugin_id,
                    enabled=False,
                    status="quarantined",
                    health="failed",
                    last_error=failure.issue_code[:800],
                )
                self._receipt(
                    plugin=failed,
                    operation="quarantine",
                    previous_state="healthy",
                    current_state="quarantined",
                    accepted=True,
                    issue_codes=[failure.issue_code[:96]],
                )
        except (KeyError, RuntimeError, ValueError):
            return

    # 功能：
    #   获取精确运行身份对应的协议客户端，为外部插件强制隔离并绑定最小宿主权限域。
    # 输入：
    #   plugin：安装目录和包摘要记录。
    #   manifest：当前使用的完整清单。
    #   configuration：可选的任务冻结配置；省略时读取当前配置。
    #   capability：可选的能力级权限约束。
    #   broker_factory：创建宿主文件与网络访问代理的工厂。
    # 输出：
    #   client：与包、配置和宿主授权作用域绑定的客户端。
    def _mcp_client(
        self,
        plugin: dict[str, object],
        manifest: PluginManifest,
        *,
        configuration: dict[str, Any] | None = None,
        capability: PluginCapability | None = None,
        broker_factory: CoreCapabilityBroker | None = None,
    ) -> McpStdioClient:
        runtime = manifest.runtime
        permissions = list(
            capability.required_permissions
            if capability is not None and capability.required_permissions is not None
            else manifest.permissions
        )
        host_services = None
        host_scope_id = "none"
        if broker_factory is not None:
            scoped_manifest = manifest.model_copy(update={"permissions": permissions})
            host_services = CapabilityBrokerHostServices(broker_factory.scope(scoped_manifest))
            host_scope_id = (
                f"{id(broker_factory)}:{capability.capability_id if capability else '*'}"
            )
        client = self._mcp_sessions.get(
            plugin_id=str(plugin["plugin_id"]),
            package_sha256=str(plugin["package_sha256"]),
            plugin_root=Path(str(plugin["bundle_root"])),
            command=runtime.command,
            protocol_version=runtime.protocol_version,
            startup_timeout_seconds=runtime.startup_timeout_seconds,
            call_timeout_seconds=runtime.call_timeout_seconds,
            configuration=configuration
            if configuration is not None
            else dict(
                self.store.get_plugin_configuration(str(plugin["plugin_id"]))["configuration"]
            ),
            permissions=permissions,
            resource_policy=manifest.resource_policy,
            host_scope_id=host_scope_id,
            host_services=host_services,
            require_os_isolation=not bool(plugin["builtin"]),
            isolator_path=self._plugin_isolator_path,
            client_factory=McpStdioClient,
        )
        return client

    # 功能：
    #   将调用故障绑定回原任务包和配置，仅在当前活动身份仍匹配时执行隔离。
    # 输入：
    #   entry：本次调用的冻结插件条目。
    #   error：调用或输出验证异常。
    # 输出：
    #   None：不返回业务数据。
    def _quarantine_failed_snapshot(self, entry: PluginSnapshotEntry, error: BaseException) -> None:
        self._quarantine_unhealthy_session(
            McpSessionFailure(
                entry.plugin_id,
                entry.package_sha256,
                entry.configuration_sha256,
                _failure_issue(error),
            )
        )

    # 功能：
    #   仅为启用且健康的当前目录调用增加在途计数，供替换和停用等待使用。
    # 输入：
    #   plugin_id：准备调用的插件标识。
    # 输出：
    #   None：不返回业务数据。
    def _acquire_call(self, plugin_id: str) -> None:
        with self._call_condition:
            plugin = self.store.get_plugin(plugin_id)
            if not bool(plugin["enabled"]) or plugin["status"] != "healthy":
                raise PluginManagerError("PLUGIN_NOT_ACCEPTING_CALLS")
            self._inflight_calls[plugin_id] = self._inflight_calls.get(plugin_id, 0) + 1

    # 功能：
    #   1. 为已有任务取得冻结调用计数，普通的“下个任务生效”停用不打断当前任务。
    #   2. 隔离、卸载和撤销立即拒绝调用；撤销按冻结包校验，不套用同名新包的状态。
    # 输入：
    #   entry：任务已冻结的清单与包身份。
    # 输出：
    #   None：不返回业务数据。
    def _acquire_snapshot_call(self, entry: PluginSnapshotEntry) -> None:
        plugin_id = entry.plugin_id
        with self._call_condition:
            plugin = self.store.get_plugin(plugin_id)
            if plugin["status"] in {"quarantined", "uninstalled", "failed"}:
                raise PluginManagerError("PLUGIN_NOT_ACCEPTING_CALLS")
            if not bool(plugin["builtin"]):
                if entry.manifest is None:
                    raise PluginManagerError("PLUGIN_SNAPSHOT_MANIFEST_MISSING")
                decision = self.trust_store.verify(entry.manifest, entry.package_sha256)
                if decision.status == "revoked":
                    raise PluginManagerError("PLUGIN_PACKAGE_REVOKED")
            self._inflight_calls[plugin_id] = self._inflight_calls.get(plugin_id, 0) + 1

    # 功能：
    #   归还一次在途调用计数，并唤醒等待插件空闲的生命周期操作。
    # 输入：
    #   plugin_id：已结束调用所属插件标识。
    # 输出：
    #   None：不返回业务数据。
    def _release_call(self, plugin_id: str) -> None:
        with self._call_condition:
            remaining = self._inflight_calls.get(plugin_id, 1) - 1
            if remaining <= 0:
                self._inflight_calls.pop(plugin_id, None)
            else:
                self._inflight_calls[plugin_id] = remaining
            self._call_condition.notify_all()

    # 功能：
    #   使用单调时钟等待插件在途调用清零，超过预算即拒绝继续替换或停用。
    # 输入：
    #   plugin_id：等待空闲的插件标识。
    #   timeout_seconds：等待秒数预算。
    # 输出：
    #   None：不返回业务数据。
    def _drain_calls(self, plugin_id: str, timeout_seconds: float = 20.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._call_condition:
            while self._inflight_calls.get(plugin_id, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PluginManagerError("PLUGIN_INFLIGHT_DRAIN_TIMEOUT")
                self._call_condition.wait(remaining)

    # 功能：
    #   计算执行清单的规范 JSON 摘要，不受字典插入顺序影响。
    # 输入：
    #   manifest：包含完整执行元数据的插件清单。
    # 输出：
    #   checksum：任务绑定所使用的清单摘要。
    @staticmethod
    def _manifest_hash(manifest: PluginManifest) -> str:
        checksum = sha256_json(manifest)
        return checksum

    # 功能：
    #   刷新产品内置定义并保留已有用户选择；不允许停用的安全能力始终启用。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def seed_builtin_plugins(self) -> None:
        for plugin_id, definition in sorted(self._definitions.items()):
            manifest = definition.manifest
            package_sha256 = self._manifest_hash(manifest)
            try:
                existing = self.store.get_plugin(plugin_id)
            except KeyError:
                enabled = manifest.default_enabled
            else:
                enabled = bool(existing["enabled"])
                if not manifest.disable_allowed:
                    enabled = True
            self.store.upsert_plugin(
                manifest=manifest,
                package_sha256=package_sha256,
                bundle_root=None,
                builtin=True,
                enabled=enabled,
                status="healthy" if enabled else "disabled",
                health="healthy" if enabled else "unknown",
                trust_status="verified",
                trust_decision={"status": "verified", "source": "builtin"},
            )

    # 功能：
    #   1. 根据安装器提供的有界索引逐包核对摘要、版本和安装内容，拒绝同版本内容漂移。
    #   2. 每个包独立安装；后续包失败不删除此前有效包，索引本身的信任由安装器负责。
    # 输入：
    #   root：安装器提供的官方索引与归档根目录。
    # 输出：
    #   None：不返回业务数据。
    def seed_official_plugins(self, root: Path) -> None:
        index_path = root / "index.json"
        if not index_path.is_file():
            raise PluginManagerError("OFFICIAL_PLUGIN_INDEX_MISSING")
        try:
            index_bytes = read_plugin_file(index_path, limit=2 * 1024 * 1024)
            index = decode_json(index_bytes)
            if (
                not isinstance(index, dict)
                or index.get("schema_version") != "dronedream.official-plugin-index.v1"
            ):
                raise PluginManagerError("OFFICIAL_PLUGIN_INDEX_INVALID")
            entries = index["plugins"]
            if not isinstance(entries, list) or len(entries) > 256:
                raise TypeError("plugins must be a list")
            identities: set[tuple[str, str]] = set()
            filenames: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict) or any(
                    not isinstance(entry.get(key), str) or not entry[key]
                    for key in ("file", "plugin_id", "version", "sha256")
                ):
                    raise TypeError("plugin entry must be an object")
                portable_plugin_path(entry["file"])
                version_precedence(entry["version"])
                identity = (entry["plugin_id"], entry["version"])
                if identity in identities or entry["file"].casefold() in filenames:
                    raise PluginManagerError("OFFICIAL_PLUGIN_INDEX_DUPLICATE")
                identities.add(identity)
                filenames.add(entry["file"].casefold())
                if len(entry["sha256"]) != 64 or any(
                    char not in "0123456789abcdef" for char in entry["sha256"]
                ):
                    raise PluginManagerError("OFFICIAL_PLUGIN_ARCHIVE_HASH_INVALID")
            for entry in entries:
                original_archive = root / entry["file"]
                check_plain_plugin_path(original_archive)
                archive = original_archive.resolve()
                if root.resolve() not in archive.parents or not archive.is_file():
                    raise PluginManagerError("OFFICIAL_PLUGIN_ARCHIVE_MISSING")
                digest = entry["sha256"]
                plugin_id = entry["plugin_id"]
                version = entry["version"]
                try:
                    installed = self.store.get_plugin_version(plugin_id, version)
                except KeyError:
                    installed = None
                if installed is not None:
                    if installed["package_sha256"] != digest:
                        raise PluginManagerError("OFFICIAL_PLUGIN_VERSION_DRIFT")
                    if hash_plugin_file(archive, limit=576 * 1024 * 1024) != digest:
                        raise PluginManagerError("OFFICIAL_PLUGIN_ARCHIVE_HASH_MISMATCH")
                    self._verify_integrity(
                        Path(installed["bundle_root"]), self._manifest(installed)
                    )
                else:
                    imported = self.import_bundle(
                        archive, expected_sha256=digest, expected_identity=(plugin_id, version)
                    )
                    imported_version = imported.get("staged_version", imported.get("version"))
                    if imported["plugin_id"] != plugin_id or imported_version != version:
                        raise PluginManagerError("OFFICIAL_PLUGIN_IDENTITY_MISMATCH")
                trust_decision = {
                    "schema_version": "dronedream.plugin-trust-decision.v1",
                    "status": "verified",
                    "source": "bundled-official-index",
                    "plugin_id": plugin_id,
                    "version": version,
                    "package_sha256": digest,
                    "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                    "issue_codes": [],
                }
                self.store.set_plugin_version_trust(
                    plugin_id,
                    version=version,
                    trust_status="verified",
                    trust_decision=trust_decision,
                )
                self._official_packages[(plugin_id, version)] = digest
                current = self.store.get_plugin(plugin_id)
                if current["version"] != version and not bool(current["enabled"]):
                    self.activate_version(plugin_id, version, operation="activate")
                elif current["version"] == version:
                    self.store.set_plugin_trust(
                        plugin_id,
                        version=version,
                        trust_status="verified",
                        trust_decision=trust_decision,
                    )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise PluginManagerError("OFFICIAL_PLUGIN_INDEX_INVALID") from error

    # 功能：
    #   为独立的目录记录补充展示字段，仅列出插件而不自动激活。
    # 输入：
    #   无。
    # 输出：
    #   values：可用于界面展示的插件记录列表。
    def list_plugins(self) -> list[dict[str, object]]:
        values = self.store.list_plugins()
        for value in values:
            self._decorate(value)
        return values

    # 功能：
    #   从清单派生能力、权限、依赖和插槽展示字段，不创建另一套配置来源。
    # 输入：
    #   value：需要原位补充展示字段的独立插件记录。
    # 输出：
    #   None：不返回业务数据。
    def _decorate(self, value: dict[str, object]) -> None:
        manifest = value.get("manifest")
        if not isinstance(manifest, dict):
            return
        parsed = PluginManifest.model_validate(manifest)
        value["capabilities"] = [item.model_dump(mode="json") for item in parsed.capabilities]
        value["permissions"] = list(parsed.permissions)
        value["dependencies"] = [item.model_dump(mode="json") for item in parsed.dependencies]
        placement = parsed.placement.model_dump(mode="json")
        value["placement"] = placement
        value["disable_allowed"] = parsed.disable_allowed
        value["slot_required"] = (
            isinstance(placement, dict) and placement.get("slot_id") in self.required_single_slots
        )

    # 功能：
    #   汇集指定插件的当前配置、已安装版本、治理决策、生命周期和调用用量记录。
    # 输入：
    #   plugin_id：要查询的插件标识。
    # 输出：
    #   value：插件详情对象。
    def get_plugin(self, plugin_id: str) -> dict[str, object]:
        value = self.store.get_plugin(plugin_id)
        self._decorate(value)
        value["versions"] = self.store.list_plugin_versions(plugin_id)
        value["events"] = self.store.list_plugin_events(plugin_id)
        value["governance_decisions"] = self.store.list_plugin_governance_decisions(plugin_id)
        value["usage"] = self.store.list_plugin_usage(plugin_id)
        value["usage_summary"] = self.store.summarize_plugin_usage(plugin_id)
        value["configuration"] = self.store.get_plugin_configuration(plugin_id)["configuration"]
        return value

    # 功能：
    #   按治理数据契约校验持久设置，缺省值由正式契约提供。
    # 输入：
    #   无。
    # 输出：
    #   policy：有效的插件治理策略。
    def governance_policy(self) -> PluginGovernancePolicy:
        value = self.store.get_settings().get("plugin_governance", {})
        policy = PluginGovernancePolicy.model_validate(value)
        return policy

    # 功能：
    #   重新校验策略，在生命周期锁内验证整个外部目录及数量上限，再保存一致的治理设置。
    # 输入：
    #   policy：调用方提交的新治理策略。
    # 输出：
    #   settings：保存后的应用设置。
    def set_governance_policy(self, policy: PluginGovernancePolicy) -> dict[str, object]:
        policy = PluginGovernancePolicy.model_validate(
            policy.model_dump(mode="python"), strict=True
        )
        with self._mutation_lock:
            # 同一次目录快照用于数量和成员检查，避免反复遍历及检查后并发插入违规包。
            plugins = [item for item in self.store.list_plugins() if not bool(item["builtin"])]
            if len(plugins) > policy.maximum_external_plugins:
                raise PluginManagerError(
                    "PLUGIN_GOVERNANCE_CATALOG_REJECTED:GOVERNANCE_EXTERNAL_PLUGIN_LIMIT"
                )
            for plugin in plugins:
                manifest = self._manifest(plugin)
                decision = evaluate_plugin_governance(
                    policy=policy,
                    manifest=manifest,
                    operation="enable" if bool(plugin["enabled"]) else "import",
                    trust_status=str(plugin["trust_status"]),
                    installed_external_plugins=len(plugins) - 1,
                )
                if not decision.accepted:
                    raise PluginManagerError(
                        "PLUGIN_GOVERNANCE_CATALOG_REJECTED:"
                        + manifest.plugin_id
                        + ":"
                        + ",".join(decision.issue_codes)
                    )
            settings = self.store.patch_settings(
                {"plugin_governance": policy.model_dump(mode="json")}
            )
            return settings

    # 功能：
    #   在允许生命周期变更前记录精确治理决策，更新已有插件不重复占用安装配额。
    # 输入：
    #   manifest：待操作插件清单。
    #   operation：计划执行的治理操作。
    #   trust_status：当前已验证的信任状态。
    # 输出：
    #   payload：获准操作的治理决策记录。
    def _govern(
        self,
        *,
        manifest: PluginManifest,
        operation: PluginGovernanceOperation,
        trust_status: str,
    ) -> dict[str, object]:
        policy = self.governance_policy()
        # 同一插件的升级不应再次消耗外部插件数量配额。
        external_count = sum(
            not bool(item["builtin"]) and item["plugin_id"] != manifest.plugin_id
            for item in self.store.list_plugins()
        )
        decision = evaluate_plugin_governance(
            policy=policy,
            manifest=manifest,
            operation=operation,
            trust_status=trust_status,
            installed_external_plugins=external_count,
        )
        payload = decision.model_dump(mode="json")
        self.store.record_plugin_governance_decision(payload)
        if not decision.accepted:
            raise PluginManagerError("PLUGIN_GOVERNANCE_DENIED:" + ",".join(decision.issue_codes))
        return payload

    # 功能：
    #   测量可编码钩子值的 UTF-8 字节数；不支持的值记零，不作为模型 token 或收费估算。
    # 输入：
    #   value：输入或输出业务值。
    # 输出：
    #   size：成功编码的字节数或不可编码时的零。
    @staticmethod
    def _encoded_size(value: Any) -> int:
        try:
            size = len(canonical_json(_mcp_json_value(value)).encode("utf-8"))
        except (PluginManagerError, TypeError, ValueError):
            size = 0
        return size

    # 功能：
    #   保存本地插件调用的时长、结果和字节量；模型供应商 token 与账户额度由独立链路核算。
    # 输入：
    #   plugin_id：实际被调用的插件标识。
    #   plugin_version：实际插件版本。
    #   capability_id：实际能力标识。
    #   slot_id：实际运行插槽。
    #   invocation_kind：钩子或工具调用类别。
    #   started：调用开始的高精度单调时间。
    #   input_value：被记录字节量的输入。
    #   output_value：被记录字节量的可选输出。
    #   issue_code：失败标识；没有失败时为空。
    # 输出：
    #   None：不返回业务数据。
    def _record_usage(
        self,
        *,
        plugin_id: str,
        plugin_version: str,
        capability_id: str,
        slot_id: str,
        invocation_kind: str,
        started: float,
        input_value: Any,
        output_value: Any = None,
        issue_code: str | None = None,
    ) -> None:
        event = PluginUsageEvent(
            invocation_id=f"plugin-call-{uuid4().hex[:24]}",
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            capability_id=capability_id,
            slot_id=slot_id,
            invocation_kind=invocation_kind,  # type: ignore[arg-type]
            outcome="error" if issue_code else "success",
            duration_ms=max(0.0, (time.perf_counter() - started) * 1_000),
            input_bytes=self._encoded_size(input_value),
            output_bytes=self._encoded_size(output_value),
            issue_code=issue_code[:160] if issue_code else None,
        )
        self.store.record_plugin_usage(event.model_dump(mode="json"))

    # 功能：
    #   解析唯一启用的健康内置钩子，按实际选中插件记录调用并始终归还在途计数。
    # 输入：
    #   slot_id：必须只有一个活动实现的插槽标识。
    #   hook：所需钩子名称。
    #   kwargs：传给钩子的业务参数。
    # 输出：
    #   output：选中钩子的实际返回值。
    def invoke_single_slot(self, slot_id: str, hook: str, **kwargs: Any) -> Any:
        matches: list[tuple[dict[str, object], PluginDefinition]] = []
        for plugin in self.store.list_plugins():
            if not bool(plugin["enabled"]) or plugin["status"] != "healthy":
                continue
            manifest = self._manifest(plugin)
            if manifest.placement.slot_id != slot_id:
                continue
            definition = self._definitions.get(manifest.plugin_id)
            if definition is not None:
                matches.append((plugin, definition))
        if len(matches) != 1:
            raise PluginManagerError(f"PLUGIN_SLOT_RESOLUTION_FAILED:{slot_id}")
        plugin, definition = matches[0]
        # 遍历结束时的 manifest 可能属于无关插件，计量前必须重取选中实现的清单。
        manifest = self._manifest(plugin)
        handler = (definition.hooks or {}).get(hook)
        if handler is None:
            raise PluginManagerError(f"PLUGIN_SLOT_HOOK_MISSING:{slot_id}:{hook}")
        plugin_id = str(plugin["plugin_id"])
        capability_id = manifest.capabilities[0].capability_id
        started = time.perf_counter()
        self._acquire_call(plugin_id)
        try:
            output = handler(**kwargs)
            self._record_usage(
                plugin_id=plugin_id,
                plugin_version=manifest.version,
                capability_id=capability_id,
                slot_id=slot_id,
                invocation_kind="hook",
                started=started,
                input_value=kwargs,
                output_value=output,
            )
            return output
        except BaseException as error:
            self._record_usage(
                plugin_id=plugin_id,
                plugin_version=manifest.version,
                capability_id=capability_id,
                slot_id=slot_id,
                invocation_kind="hook",
                started=started,
                input_value=kwargs,
                issue_code=type(error).__name__,
            )
            raise
        finally:
            self._release_call(plugin_id)

    # 功能：
    #   1. 执行已启用的附加资产检查并绑定回执，不允许插件覆盖已失败的核心资格门槛。
    #   2. 导出证据保留全部身份和配置摘要，但剔除本机安装路径。
    # 输入：
    #   plan：准备验证的资产运行计划。
    #   map_asset_ir：地图规范表示。
    #   vehicle_asset_ir：无人机规范表示。
    #   runtime_evidence：当前运行验证证据。
    #   environment_versions：验证环境版本集合。
    # 输出：
    #   result：可移植插件快照、摘要、检查结果和钩子回执。
    def run_asset_qualification_checks(
        self,
        *,
        plan: dict[str, object],
        map_asset_ir: dict[str, object],
        vehicle_asset_ir: dict[str, object],
        runtime_evidence: dict[str, object],
        environment_versions: dict[str, str],
    ) -> dict[str, object]:
        snapshot = self.snapshot()
        registry = self.build_extension_registry(snapshot=snapshot)
        outputs, receipts = registry.invoke_multiple(
            "assets.qualification-checks",
            "qualify_asset",
            plan=plan,
            map_asset_ir=map_asset_ir,
            vehicle_asset_ir=vehicle_asset_ir,
            runtime_evidence=runtime_evidence,
            environment_versions=environment_versions,
        )
        failed_receipts = [receipt for receipt in receipts if receipt.outcome != "accepted"]
        if failed_receipts:
            codes = [
                code
                for receipt in failed_receipts
                for code in (receipt.issue_codes or ["PLUGIN_HOOK_NOT_ACCEPTED"])
            ]
            raise PluginManagerError(
                "ASSET_QUALIFICATION_PLUGIN_EXECUTION_FAILED:" + ",".join(dict.fromkeys(codes))
            )
        accepted_receipts = [receipt for receipt in receipts if receipt.outcome == "accepted"]
        if len(outputs) != len(accepted_receipts):
            raise PluginManagerError("ASSET_QUALIFICATION_PLUGIN_RECEIPT_MISMATCH")
        checks: list[AssetQualificationPluginCheck] = []
        for output, receipt in zip(outputs, accepted_receipts, strict=True):
            if not isinstance(output, dict):
                raise PluginManagerError("ASSET_QUALIFICATION_PLUGIN_OUTPUT_INVALID")
            try:
                check = AssetQualificationPluginCheck.model_validate(
                    {
                        **output,
                        "plugin_id": receipt.plugin_id,
                        "capability_id": receipt.capability_id,
                    }
                )
            except ValidationError as error:
                raise PluginManagerError("ASSET_QUALIFICATION_PLUGIN_OUTPUT_INVALID") from error
            checks.append(check)
        rejected = [code for check in checks if not check.accepted for code in check.issue_codes]
        if rejected:
            raise PluginManagerError(
                "ASSET_QUALIFICATION_PLUGIN_REJECTED:" + ",".join(dict.fromkeys(rejected))
            )
        # 运行注册表继续持有完整快照，导出的可移植证据副本只移除本机路径。
        evidence_snapshot = snapshot.model_copy(
            update={
                "plugins": [
                    entry.model_copy(update={"bundle_root": None}) for entry in snapshot.plugins
                ]
            }
        )
        result = {
            "plugin_snapshot": evidence_snapshot.model_dump(mode="json"),
            "plugin_snapshot_sha256": sha256_json(evidence_snapshot),
            "plugin_checks": [check.model_dump(mode="json") for check in checks],
            "plugin_hook_receipts": [receipt.model_dump(mode="json") for receipt in receipts],
        }
        return result

    # 功能：
    #   优先使用核心格式识别；仅在无法识别时尝试活动插件声明的后缀，歧义时要求明确声明。
    # 输入：
    #   source：原始资产文件。
    #   declared_format：用户声明格式或 auto。
    # 输出：
    #   detection：核心识别结果或绑定唯一外部适配器的识别结果。
    def detect_asset_source_with_plugins(
        self, source: Path, declared_format: str = "auto"
    ) -> AssetSourceDetection:
        try:
            detection = detect_builtin_asset_source(source, declared_format)
            return detection
        except AssetSourceAdapterError as error:
            if str(error) != "ASSET_SOURCE_FORMAT_UNRECOGNIZED":
                raise
        suffix = source.suffix.casefold()
        matches: list[AssetSourceAdapterDescriptor] = []
        for value in self.source_adapter_catalog():
            if not bool(value.get("enabled")):
                continue
            descriptor = AssetSourceAdapterDescriptor.model_validate(
                {
                    key: value[key]
                    for key in AssetSourceAdapterDescriptor.model_fields
                    if key in value
                }
            )
            if suffix not in descriptor.file_extensions:
                continue
            if declared_format not in {"", "auto"} and declared_format not in (
                descriptor.source_formats
            ):
                continue
            matches.append(descriptor)
        unique = {entry.adapter_id: entry for entry in matches}
        if not unique:
            raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_UNRECOGNIZED")
        if len(unique) != 1:
            raise AssetSourceAdapterError("ASSET_SOURCE_ADAPTER_AMBIGUOUS")
        descriptor = next(iter(unique.values()))
        if declared_format not in {"", "auto"}:
            source_format = declared_format
        elif len(descriptor.source_formats) == 1:
            source_format = descriptor.source_formats[0]
        else:
            raise AssetSourceAdapterError("ASSET_SOURCE_FORMAT_DECLARATION_REQUIRED")
        asset_kind: AssetKind | None = (
            descriptor.asset_kinds[0] if len(descriptor.asset_kinds) == 1 else None
        )
        detection = AssetSourceDetection(
            source_format=source_format,
            adapter_id=descriptor.adapter_id,
            asset_kind=asset_kind,
            confidence="extension",
            can_normalize_locally=False,
            required_inputs=[f"plugin_adapter:{descriptor.adapter_id}"],
        )
        return detection

    # 功能：
    #   1. 按识别结果选择唯一启用的隔离转换器，仅授予输入快照读取和暂存输出权限。
    #   2. 核对转换包摘要、来源、格式和资产类型，不接受插件自行授予飞行资格。
    #   3. 完整验证后无覆盖发布；输入准备和目标发布错误不归因为插件自身故障。
    # 输入：
    #   self：管理插件目录、运行会话及权限的当前实例。
    #   source：调用方选定的原始源文件。
    #   detection：需要插件转换的格式识别结果。
    #   destination：用于接收规范化结果的新文件路径。
    #   expected_kind：调用方要求的资产类型，可为 None。
    # 输出：
    #   destination：通过验证的新资产包路径；没有对应的外部适配器时返回 None。
    def normalize_asset_source_with_plugin(
        self,
        *,
        source: Path,
        detection: AssetSourceDetection,
        destination: Path,
        expected_kind: AssetKind | None,
    ) -> Path | None:
        detected_kind = detection.asset_kind or expected_kind
        if detected_kind is None:
            raise PluginManagerError("ASSET_SOURCE_KIND_REQUIRED")
        if (
            expected_kind is not None
            and expected_kind != detected_kind
            and {
                expected_kind,
                detected_kind,
            }
            != {"map", "world"}
        ):
            raise PluginManagerError("ASSET_SOURCE_KIND_MISMATCH")
        capability_kind = "vehicle-importer" if detected_kind == "vehicle" else "map-importer"
        slot_id = "assets.vehicle-importer" if detected_kind == "vehicle" else "assets.map-importer"
        matches: list[tuple[dict[str, object], PluginManifest, PluginCapability]] = []
        for plugin in self.store.list_plugins():
            if not bool(plugin["enabled"]) or plugin["status"] != "healthy":
                continue
            manifest = self._manifest(plugin)
            if manifest.placement.slot_id != slot_id:
                continue
            for capability in manifest.capabilities:
                if capability.kind != capability_kind:
                    continue
                raw_descriptors = capability.metadata.get("asset_source_adapters", [])
                if not isinstance(raw_descriptors, list):
                    raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_METADATA_INVALID")
                try:
                    descriptors = [
                        AssetSourceAdapterDescriptor.model_validate(value)
                        for value in raw_descriptors
                    ]
                except (TypeError, ValidationError) as error:
                    raise PluginManagerError(
                        "PLUGIN_ASSET_SOURCE_ADAPTER_METADATA_INVALID"
                    ) from error
                if any(
                    descriptor.adapter_id == detection.adapter_id
                    and detection.source_format in descriptor.source_formats
                    and detected_kind in descriptor.asset_kinds
                    for descriptor in descriptors
                ):
                    matches.append((plugin, manifest, capability))
        if not matches:
            return None
        if len(matches) != 1:
            raise PluginManagerError(
                f"PLUGIN_SOURCE_ADAPTER_RESOLUTION_FAILED:{detection.adapter_id}"
            )
        plugin, manifest, capability = matches[0]
        plugin_id = str(plugin["plugin_id"])
        if bool(plugin["builtin"]):
            # Core declarative parsers are invoked directly by the import service;
            # an in-process plugin must never impersonate the isolated adapter path.
            return None
        if manifest.runtime.kind != "mcp-stdio":
            raise PluginManagerError("ASSET_SOURCE_ADAPTER_RUNTIME_UNSUPPORTED")
        if "asset.read" not in manifest.permissions:
            raise PluginManagerError("ASSET_IMPORTER_READ_PERMISSION_REQUIRED")
        if "asset.write-staging" not in manifest.permissions:
            raise PluginManagerError("ASSET_IMPORTER_STAGING_PERMISSION_REQUIRED")
        check_plain_plugin_path(destination)
        if destination.exists():
            raise PluginManagerError("ASSET_SOURCE_ADAPTER_DESTINATION_EXISTS")

        source_sha256: str | None = None
        quarantine_on_failure = True
        started = time.perf_counter()
        self._acquire_call(plugin_id)
        try:
            plugin_root = Path(str(plugin["bundle_root"]))
            self._verify_integrity(plugin_root, manifest)
            # 输入文件、空间不足等宿主准备错误不能成为停用正常插件的依据。
            quarantine_on_failure = False
            with tempfile.TemporaryDirectory(
                prefix="dd-source-adapter-", dir=self.store.plugin_staging_root
            ) as temporary:
                exchange_root = Path(temporary)
                input_root = exchange_root / "input"
                output_root = exchange_root / "output"
                input_root.mkdir()
                output_root.mkdir()
                source_suffix = source.suffix.casefold()
                if (
                    not source_suffix
                    or len(source_suffix) > 16
                    or any(
                        character not in ".abcdefghijklmnopqrstuvwxyz0123456789"
                        for character in source_suffix
                    )
                ):
                    source_suffix = ".source"
                input_source = input_root / f"source{source_suffix}"
                source_sha256 = hash_plugin_file(
                    source, destination=input_source, limit=MAX_PACKAGE_UNCOMPRESSED_BYTES
                )
                output_package = output_root / "normalized.ddpkg"
                broker = CoreCapabilityBroker(
                    read_roots={"imports": input_root},
                    write_roots={"staging": output_root},
                )
                quarantine_on_failure = True
                client = self._mcp_client(
                    plugin,
                    manifest,
                    capability=capability,
                    broker_factory=broker,
                )
                arguments = {
                    "asset_kind": detected_kind,
                    "source_format": detection.source_format,
                    "adapter_id": detection.adapter_id,
                    "input_root": "imports",
                    "input_path": input_source.name,
                    "input_sha256": source_sha256,
                    "output_root": "staging",
                    "output_path": output_package.name,
                }
                try:
                    jsonschema.validate(arguments, capability.input_schema)
                    result = client.call_tool(capability.capability_id, arguments)
                    jsonschema.validate(result, capability.output_schema)
                finally:
                    # Retire the exact temporary permission scope before reading
                    # output evidence or deleting its exchange directories.
                    self._mcp_sessions.release(client)
                if (
                    not output_package.is_file()
                    or output_package.stat().st_size <= 0
                    or output_package.stat().st_size > MAX_PACKAGE_UNCOMPRESSED_BYTES
                ):
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_OUTPUT_INVALID")
                output_sha256 = _file_sha256(output_package)
                if result.get("output_sha256") != output_sha256:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_OUTPUT_HASH_MISMATCH")
                inspected = inspect_ddpkg(output_package)
                if inspected.manifest.qualification is not None:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_QUALIFICATION_FORBIDDEN")
                if inspected.asset_ir.source.adapter_id != detection.adapter_id:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_PROVENANCE_ID_MISMATCH")
                if inspected.asset_ir.source.source_format != detection.source_format:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_PROVENANCE_FORMAT_MISMATCH")
                if inspected.asset_ir.source.source_sha256 != source_sha256:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_PROVENANCE_HASH_MISMATCH")
                if inspected.manifest.asset_kind != detected_kind and {
                    inspected.manifest.asset_kind,
                    detected_kind,
                } != {"map", "world"}:
                    raise PluginManagerError("ASSET_SOURCE_ADAPTER_KIND_MISMATCH")
                # 插件已退出且输出已验证，随后发布冲突属于宿主文件操作而非插件失信。
                quarantine_on_failure = False
                publish_asset_file(
                    output_package,
                    destination,
                    expected_sha256=output_sha256,
                    limit=MAX_PACKAGE_UNCOMPRESSED_BYTES,
                )
                self._record_usage(
                    plugin_id=plugin_id,
                    plugin_version=manifest.version,
                    capability_id=capability.capability_id,
                    slot_id=slot_id,
                    invocation_kind="hook",
                    started=started,
                    input_value=arguments,
                    output_value={
                        "output_sha256": output_sha256,
                        "asset_id": inspected.manifest.asset_id,
                        "asset_kind": inspected.manifest.asset_kind,
                    },
                )
                return destination
        except (
            PluginProcessError,
            PluginManagerError,
            jsonschema.ValidationError,
            ValueError,
            OSError,
        ) as error:
            self._record_usage(
                plugin_id=plugin_id,
                plugin_version=manifest.version,
                capability_id=capability.capability_id,
                slot_id=slot_id,
                invocation_kind="hook",
                started=started,
                input_value={
                    "asset_kind": detected_kind,
                    "source_format": detection.source_format,
                    "adapter_id": detection.adapter_id,
                    "input_sha256": source_sha256,
                },
                issue_code=type(error).__name__,
            )
            if quarantine_on_failure:
                failed = self.store.set_plugin_lifecycle(
                    plugin_id,
                    enabled=False,
                    status="quarantined",
                    health="failed",
                    last_error=_failure_issue(error),
                )
                self._receipt(
                    plugin=failed,
                    operation="quarantine",
                    previous_state="healthy",
                    current_state="quarantined",
                    accepted=True,
                    issue_codes=[_failure_issue(error)],
                )
            if isinstance(error, PluginManagerError):
                raise
            raise PluginManagerError(str(error)) from error
        finally:
            self._release_call(plugin_id)

    # 功能：
    #   根据已校验的任务快照解析唯一插槽能力，不把冻结选择重新绑定到最新目录。
    # 输入：
    #   snapshot：任务冻结快照。
    #   slot_id：待解析的插槽标识。
    # 输出：
    #   selected：快照条目、对应清单和唯一能力的元组。
    def capability_for_slot(
        self, snapshot: PluginSnapshot, slot_id: str
    ) -> tuple[PluginSnapshotEntry, PluginManifest, PluginCapability]:
        snapshot = verify_plugin_snapshot(snapshot)
        matches: list[tuple[PluginSnapshotEntry, PluginManifest, PluginCapability]] = []
        for entry in snapshot.plugins:
            manifest = entry.manifest
            if manifest is None or (
                manifest.plugin_id != entry.plugin_id
                or manifest.version != entry.version
                or self._manifest_hash(manifest) != entry.manifest_sha256
                or [item.capability_id for item in manifest.capabilities] != entry.capability_ids
            ):
                raise PluginManagerError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
            if manifest.placement.slot_id != slot_id:
                continue
            for capability in manifest.capabilities:
                matches.append((entry, manifest, capability))
        if len(matches) != 1:
            raise PluginManagerError(f"PLUGIN_SLOT_RESOLUTION_FAILED:{slot_id}")
        selected = matches[0]
        return selected

    # 功能：
    #   核对活动声明式插件的安装字节后生成面板数据，不允许调用方冒充宿主状态来源。
    # 输入：
    #   plugin_id：要展示面板的插件标识。
    #   data_sources：可选的业务数据源，不能覆盖 plugin、configuration 或 events。
    # 输出：
    #   panel_document：完成绑定与格式验证的声明式面板。
    def ui_panel_document(
        self, plugin_id: str, *, data_sources: dict[str, object] | None = None
    ) -> dict[str, object]:
        plugin = self.store.get_plugin(plugin_id)
        manifest = self._manifest(plugin)
        if (
            not bool(plugin["enabled"])
            or plugin["status"] != "healthy"
            or manifest.runtime.kind != "ui-declarative"
        ):
            raise PluginManagerError("UI_PLUGIN_NOT_ACTIVE")
        capability = next((item for item in manifest.capabilities if item.kind == "ui-panel"), None)
        if capability is None:
            raise PluginManagerError("UI_PLUGIN_PANEL_MISSING")
        entrypoint = capability.metadata.get("entrypoint")
        if not isinstance(entrypoint, str):
            raise PluginManagerError("UI_PLUGIN_ENTRYPOINT_MISSING")
        installed_root = Path(str(plugin["bundle_root"]))
        self._verify_integrity(installed_root, manifest)
        root = installed_root.resolve()
        panel = (root / entrypoint).resolve()
        if root not in panel.parents or not panel.is_file() or panel.stat().st_size > 256_000:
            raise PluginManagerError("UI_PLUGIN_ENTRYPOINT_INVALID")
        try:
            value = decode_json(read_plugin_file(panel, limit=256_000), limit=256_000)
            document = validate_panel_document(value)
        except (OSError, ValueError, ValidationError) as error:
            raise PluginManagerError("UI_PLUGIN_DOCUMENT_INVALID") from error
        sources = {
            **(data_sources or {}),
            "plugin": plugin,
            "configuration": self.store.get_plugin_configuration(plugin_id)["configuration"],
            "events": {"items": self.store.list_plugin_events(plugin_id)},
        }
        panel_document = materialize_panel(
            document,
            sources=sources,
            configuration_schema=manifest.configuration_schema,
        )
        return panel_document

    # 功能：
    #   查找对指定插件存在必选依赖的已安装插件，按需仅保留启用者。
    # 输入：
    #   plugin_id：被依赖插件标识。
    #   enabled_only：是否只检查当前活动依赖。
    # 输出：
    #   dependents：按标识排序的依赖者列表。
    def _dependents(self, plugin_id: str, *, enabled_only: bool) -> list[str]:
        dependents: list[str] = []
        for candidate in self.store.list_plugins():
            if enabled_only and not bool(candidate["enabled"]):
                continue
            manifest = self._manifest(candidate)
            if any(
                dependency.plugin_id == plugin_id and not dependency.optional
                for dependency in manifest.dependencies
            ):
                dependents.append(manifest.plugin_id)
        dependents.sort()
        return dependents

    # 功能：
    #   校验并保存生命周期回执，由调用方将其与关联状态变更放入同一事务。
    # 输入：
    #   plugin：本次操作的插件身份与包摘要。
    #   operation：生命周期操作名称。
    #   previous_state：操作前状态。
    #   current_state：操作后状态。
    #   accepted：操作是否获准。
    #   issue_codes：附带问题或决策标识。
    # 输出：
    #   payload：已保存的回执对象。
    def _receipt(
        self,
        *,
        plugin: dict[str, object],
        operation: str,
        previous_state: str | None,
        current_state: str,
        accepted: bool,
        issue_codes: list[str] | None = None,
    ) -> dict[str, object]:
        receipt = PluginLifecycleReceipt.model_validate(
            {
                "receipt_id": f"plugin-event-{uuid4().hex[:24]}",
                "plugin_id": plugin["plugin_id"],
                "version": plugin["version"],
                "operation": operation,
                "previous_state": previous_state,
                "current_state": current_state,
                "accepted": accepted,
                "issue_codes": issue_codes or [],
                "package_sha256": plugin["package_sha256"],
                "created_at": datetime.now(UTC),
            }
        )
        payload = receipt.model_dump(mode="json")
        self.store.record_plugin_event(payload)
        return payload

    # 功能：
    #   在目录预算预检之后、解压之前检查所有成员的路径冲突、类型、加密和大小限制。
    # 输入：
    #   bundle：已完成目录分配预算检查的 ZIP 对象。
    # 输出：
    #   members：通过完整命名空间检查的成员列表。
    @staticmethod
    def _validated_archive_members(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        members = bundle.infolist()
        total_size = sum(item.file_size for item in members)
        if not members or len(members) > 2_000 or total_size > 512 * 1024 * 1024:
            raise PluginManagerError("PLUGIN_BUNDLE_LIMIT_EXCEEDED")
        names: dict[str, bool] = {}
        for item in members:
            try:
                name = portable_plugin_path(item.filename[:-1] if item.is_dir() else item.filename)
            except ValueError as error:
                raise PluginManagerError("PLUGIN_BUNDLE_PATH_TRAVERSAL") from error
            key = name.casefold()
            if key in names:
                raise PluginManagerError("PLUGIN_BUNDLE_DUPLICATE_PATH")
            names[key] = item.is_dir()
            if item.file_size > 256 * 1024 * 1024:
                raise PluginManagerError("PLUGIN_BUNDLE_FILE_TOO_LARGE")
            if name == "plugin.json" and item.file_size > 2 * 1024 * 1024:
                raise PluginManagerError("PLUGIN_MANIFEST_TOO_LARGE")
            unix_type = (item.external_attr >> 16) & 0o170000
            if unix_type == stat.S_IFLNK:
                raise PluginManagerError("PLUGIN_BUNDLE_SYMLINK_FORBIDDEN")
            if unix_type not in {0, stat.S_IFREG, stat.S_IFDIR} or item.flag_bits & 1:
                raise PluginManagerError("PLUGIN_BUNDLE_UNSUPPORTED_MEMBER")
            if unix_type == stat.S_IFDIR and not item.is_dir():
                raise PluginManagerError("PLUGIN_BUNDLE_DIRECTORY_TYPE_MISMATCH")
        for name in names:
            parts = name.split("/")
            if any(names.get("/".join(parts[:index])) is False for index in range(1, len(parts))):
                raise PluginManagerError("PLUGIN_BUNDLE_FILE_DIRECTORY_COLLISION")
        return members

    # 功能：
    #   使用安装和运行共用的负载验证器，拒绝额外文件、链接、身份变化或摘要不符。
    # 输入：
    #   staging：待核验的安装或暂存目录。
    #   manifest：提供全部负载摘要的清单。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _verify_integrity(staging: Path, manifest: PluginManifest) -> None:
        try:
            verify_plugin_files(staging, manifest.file_sha256)
        except (OSError, ValueError) as error:
            raise PluginManagerError(str(error)) from error

    # 功能：
    #   检查依赖的安装版本，激活时额外要求依赖已启用；缺失的可选依赖可以跳过。
    # 输入：
    #   manifest：待安装或激活的插件清单。
    #   require_enabled：是否检查依赖启用状态。
    # 输出：
    #   None：不返回业务数据。
    def _validate_dependencies(self, manifest: PluginManifest, *, require_enabled: bool) -> None:
        for dependency in manifest.dependencies:
            try:
                installed = self.store.get_plugin(dependency.plugin_id)
            except KeyError:
                if dependency.optional:
                    continue
                raise PluginManagerError(
                    f"PLUGIN_DEPENDENCY_MISSING:{dependency.plugin_id}"
                ) from None
            if not _version_matches(str(installed["version"]), dependency.version):
                raise PluginManagerError(f"PLUGIN_DEPENDENCY_VERSION:{dependency.plugin_id}")
            if require_enabled and not bool(installed["enabled"]):
                raise PluginManagerError(f"PLUGIN_DEPENDENCY_DISABLED:{dependency.plugin_id}")

    # 功能：
    #   同时检查候选声明的冲突和当前活动插件反向声明的冲突。
    # 输入：
    #   manifest：待激活插件清单。
    # 输出：
    #   None：不返回业务数据。
    def _validate_conflicts(self, manifest: PluginManifest) -> None:
        enabled = {
            str(item["plugin_id"]): self._manifest(item)
            for item in self.store.list_plugins()
            if bool(item["enabled"]) and item["plugin_id"] != manifest.plugin_id
        }
        direct = set(manifest.conflicts).intersection(enabled)
        reverse = {
            plugin_id
            for plugin_id, candidate in enabled.items()
            if manifest.plugin_id in candidate.conflicts
        }
        conflicts = sorted(direct | reverse)
        if conflicts:
            raise PluginManagerError("PLUGIN_CONFLICT:" + ",".join(conflicts))

    # 功能：
    #   核对同插槽激活模式一致性，管线模式用拓扑排序拒绝循环依赖。
    # 输入：
    #   manifest：准备加入插槽的候选清单。
    # 输出：
    #   None：不返回业务数据。
    def _validate_slot_contract(self, manifest: PluginManifest) -> None:
        peers = []
        for record in self.store.list_plugins():
            if record["plugin_id"] == manifest.plugin_id:
                continue
            candidate = self._manifest(record)
            if candidate.placement.slot_id == manifest.placement.slot_id:
                peers.append(candidate)
        if any(
            peer.placement.activation_mode != manifest.placement.activation_mode for peer in peers
        ):
            raise PluginManagerError("PLUGIN_SLOT_ACTIVATION_MODE_MISMATCH")
        if manifest.placement.activation_mode != "pipeline":
            return
        active = [
            peer for peer in peers if bool(self.store.get_plugin(peer.plugin_id)["enabled"])
        ] + [manifest]
        by_id = {item.plugin_id: item for item in active}
        edges: dict[str, set[str]] = {plugin_id: set() for plugin_id in by_id}
        indegree: dict[str, int] = {plugin_id: 0 for plugin_id in by_id}
        for item in active:
            for predecessor in item.placement.runs_after:
                if predecessor in by_id and item.plugin_id not in edges[predecessor]:
                    edges[predecessor].add(item.plugin_id)
                    indegree[item.plugin_id] += 1
            for successor in item.placement.runs_before:
                if successor in by_id and successor not in edges[item.plugin_id]:
                    edges[item.plugin_id].add(successor)
                    indegree[successor] += 1
        ready = [plugin_id for plugin_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for successor in edges[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(by_id):
            raise PluginManagerError("PLUGIN_PIPELINE_ORDERING_CYCLE")

    # 功能：
    #   静态检查程序位置、面板文档或 ROS 包结构，不运行用户代码；不赋予原生插件认证。
    # 输入：
    #   staging：待检查的包目录。
    #   manifest：已校验插件清单。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _validate_static_bundle(staging: Path, manifest: PluginManifest) -> None:
        if manifest.runtime.kind == "mcp-stdio":
            if not any(item.kind in PLUGIN_MCP_CAPABILITY_KINDS for item in manifest.capabilities):
                raise PluginManagerError("MCP_PLUGIN_HAS_NO_TOOL_CAPABILITY")
            try:
                resolve_plugin_command(staging, manifest.runtime.command)
            except PluginProcessError as error:
                raise PluginManagerError(str(error)) from error
            return
        if manifest.runtime.kind == "ui-declarative":
            for capability in manifest.capabilities:
                if capability.kind != "ui-panel":
                    continue
                entrypoint = capability.metadata.get("entrypoint")
                if not isinstance(entrypoint, str) or not entrypoint:
                    raise PluginManagerError("UI_PLUGIN_ENTRYPOINT_MISSING")
                panel = (staging / entrypoint).resolve()
                if staging.resolve() not in panel.parents or not panel.is_file():
                    raise PluginManagerError("UI_PLUGIN_ENTRYPOINT_INVALID")
                try:
                    value = decode_json(read_plugin_file(panel, limit=256_000), limit=256_000)
                except (OSError, ValueError) as error:
                    raise PluginManagerError("UI_PLUGIN_DOCUMENT_INVALID") from error
                try:
                    validate_panel_document(value)
                except (ValueError, ValidationError) as error:
                    raise PluginManagerError("UI_PLUGIN_DOCUMENT_INVALID") from error
            return
        if manifest.runtime.kind == "ros2-node":
            package_name = manifest.runtime.command[0]
            package_manifest = staging / "ros_ws" / "src" / package_name / "package.xml"
            if not package_manifest.is_file():
                raise PluginManagerError("ROS2_PLUGIN_PACKAGE_MISSING")

    # 功能：
    #   1. 将普通输入文件有界冻结到当前操作持有的临时流，复核源身份及读取期间变化。
    #   2. 验证可选预期摘要后，仅将同一快照交给安装器，避免校验与安装重读不同文件。
    # 输入：
    #   self：当前插件管理器。
    #   archive：待导入的本地归档路径。
    #   expected_sha256：市场或官方目录所要求的归档摘要，可为 None。
    #   expected_identity：要求的插件标识与版本元组，可为 None。
    # 输出：
    #   installed：已安装或暂存的插件信息及其生命周期回执。
    def import_bundle(
        self,
        archive: Path,
        *,
        expected_sha256: str | None = None,
        expected_identity: tuple[str, str] | None = None,
    ) -> dict[str, object]:
        check_plain_plugin_path(archive)
        if not archive.is_file():
            raise PluginManagerError("PLUGIN_BUNDLE_MUST_BE_ZIP")
        with self._mutation_lock, tempfile.TemporaryFile(dir=self.store.plugins_root) as frozen:
            try:
                package_sha256 = copy_asset_source(archive, frozen, 576 * 1024 * 1024)
            except (OSError, ValueError) as error:
                raise PluginManagerError("PLUGIN_BUNDLE_SOURCE_INVALID_OR_CHANGED") from error
            frozen.seek(0)
            if not zipfile.is_zipfile(frozen):
                raise PluginManagerError("PLUGIN_BUNDLE_MUST_BE_ZIP")
            frozen.seek(0)
            if expected_sha256 is not None and package_sha256 != expected_sha256:
                raise PluginManagerError("OFFICIAL_PLUGIN_ARCHIVE_HASH_MISMATCH")
            installed = self._import_frozen_bundle(
                frozen, package_sha256, expected_identity=expected_identity
            )
            return installed

    # 功能：
    #   1. 先检查目录分配预算，再解压和核对清单、内容、信任及权限限制。
    #   2. 保持已安装版本身份不可变；安装目录与数据库变更失败时仅回收本次材料。
    # 输入：
    #   self：当前插件管理器。
    #   archive：当前操作独占持有的归档快照流。
    #   package_sha256：该快照的实际摘要。
    #   expected_identity：要求的插件标识与版本元组，可为 None。
    # 输出：
    #   result：插件安装或暂存结果以及生命周期回执。
    def _import_frozen_bundle(
        self,
        archive: BinaryIO,
        package_sha256: str,
        *,
        expected_identity: tuple[str, str] | None = None,
    ) -> dict[str, object]:
        try:
            validate_zip_index(archive, maximum_members=2_000)
        except ValueError as error:
            raise PluginManagerError("PLUGIN_BUNDLE_INDEX_INVALID") from error
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.store.plugins_root))
        moved = False
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = self._validated_archive_members(bundle)
                bundle.extractall(staging, members)
            manifest_path = staging / "plugin.json"
            if not manifest_path.is_file():
                raise PluginManagerError("PLUGIN_MANIFEST_MISSING")
            try:
                manifest = PluginManifest.model_validate(
                    decode_json(read_plugin_file(manifest_path, limit=2 * 1024 * 1024))
                )
            except (OSError, ValueError) as error:
                raise PluginManagerError("PLUGIN_MANIFEST_INVALID") from error
            if (
                expected_identity is not None
                and (manifest.plugin_id, manifest.version) != expected_identity
            ):
                raise PluginManagerError("OFFICIAL_PLUGIN_IDENTITY_MISMATCH")
            if manifest.runtime.kind not in {"mcp-stdio", "ros2-node", "ui-declarative"}:
                raise PluginManagerError("PLUGIN_RUNTIME_NOT_IMPORTABLE")
            if manifest.runtime.kind == "ros2-node":
                raise PluginManagerError("NATIVE_PLUGIN_REQUIRES_CERTIFIED_INSTALLER")
            if not manifest.removable:
                raise PluginManagerError("THIRD_PARTY_PLUGIN_MUST_BE_REMOVABLE")
            if "vehicle.actuate" in manifest.permissions:
                raise PluginManagerError("THIRD_PARTY_ACTUATION_NOT_CERTIFIED")
            if manifest.placement.slot_id in self.protected_slots:
                raise PluginManagerError("THIRD_PARTY_PROTECTED_SLOT_NOT_CERTIFIED")
            extension_capabilities = [
                capability
                for capability in manifest.capabilities
                if "extension_hook" in capability.metadata
            ]
            if extension_capabilities:
                if manifest.runtime.kind != "mcp-stdio":
                    raise PluginManagerError("EXTENSION_HOOK_REQUIRES_MCP_STDIO")
                if len(extension_capabilities) > 32:
                    raise PluginManagerError("EXTENSION_PLUGIN_CAPABILITY_LIMIT")
                if (
                    manifest.placement.activation_mode == "single"
                    and len(extension_capabilities) != 1
                ):
                    raise PluginManagerError("SINGLE_SLOT_REQUIRES_ONE_HOOK_CAPABILITY")
                for extension_capability in extension_capabilities:
                    extension_hook = extension_capability.metadata.get("extension_hook")
                    if extension_hook not in PLUGIN_EXTENSION_HOOKS:
                        raise PluginManagerError("EXTENSION_HOOK_NOT_SUPPORTED")
                    if (
                        not extension_capability.input_schema
                        or not extension_capability.output_schema
                    ):
                        raise PluginManagerError("EXTENSION_HOOK_SCHEMA_REQUIRED")
                    if extension_hook == "import_asset" and (
                        len(extension_capabilities) != 1
                        or extension_capability.kind not in {"map-importer", "vehicle-importer"}
                        or manifest.placement.slot_id
                        not in {"assets.map-importer", "assets.vehicle-importer"}
                        or "asset.write-staging" not in manifest.permissions
                    ):
                        raise PluginManagerError("ASSET_IMPORTER_CONTRACT_INVALID")
            self._asset_source_adapter_descriptors(manifest, builtin=False)
            if version_precedence(manifest.minimum_app_version) > version_precedence(
                self.app_version
            ):
                raise PluginManagerError("PLUGIN_APP_VERSION_INCOMPATIBLE")
            self._verify_integrity(staging, manifest)
            trust_decision = self.trust_store.verify(manifest, package_sha256)
            self._govern(
                manifest=manifest,
                operation="import",
                trust_status=trust_decision.status,
            )
            self._validate_static_bundle(staging, manifest)
            self._validate_dependencies(manifest, require_enabled=False)
            destination = self.store.plugins_root / manifest.plugin_id / manifest.version
            check_plain_plugin_path(destination)
            destination = destination.resolve()
            root = self.store.plugins_root.resolve()
            if root not in destination.parents:
                raise PluginManagerError("PLUGIN_DESTINATION_INVALID")
            try:
                current = self.store.get_plugin(manifest.plugin_id)
            except KeyError:
                current = None
            try:
                existing_version = self.store.get_plugin_version(
                    manifest.plugin_id, manifest.version
                )
            except KeyError:
                existing_version = None
            if existing_version is not None and (
                existing_version["package_sha256"] != package_sha256
            ):
                raise PluginManagerError("PLUGIN_VERSION_IMMUTABLE")
            if destination.exists():
                if existing_version is None:
                    raise PluginManagerError("PLUGIN_VERSION_DIRECTORY_UNTRACKED")
                self._verify_integrity(destination, manifest)
                shutil.rmtree(staging)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                check_plain_plugin_path(destination.parent)
                # Exclusive directory ownership prevents shutil.move from nesting
                # staging inside a version created by another installer.
                destination.mkdir()
                moved = True
                for child in staging.iterdir():
                    shutil.move(str(child), destination / child.name)
                self._verify_integrity(destination, manifest)
            with self.store.atomic_plugin_changes():
                if current is None:
                    installed = self.store.upsert_plugin(
                        manifest=manifest,
                        package_sha256=package_sha256,
                        bundle_root=destination,
                        builtin=False,
                        enabled=False,
                        status="disabled",
                        health="unknown",
                        trust_status=trust_decision.status,
                        trust_decision=trust_decision.model_dump(mode="json"),
                    )
                    operation = "install"
                else:
                    if existing_version is None:
                        self.store.install_plugin_version(
                            manifest=manifest,
                            package_sha256=package_sha256,
                            bundle_root=destination,
                            trust_status=trust_decision.status,
                            trust_decision=trust_decision.model_dump(mode="json"),
                        )
                    installed = current
                    operation = "update"
                    installed = {
                        **installed,
                        "staged_version": manifest.version,
                        "staged_package_sha256": package_sha256,
                    }
                receipt = self._receipt(
                    plugin={
                        **installed,
                        "version": manifest.version,
                        "package_sha256": package_sha256,
                    },
                    operation=operation,
                    previous_state=str(current["status"]) if current is not None else None,
                    current_state=str(installed["status"]),
                    accepted=True,
                )
                result = {**installed, "receipt": receipt}
                return result
        except BaseException:
            if moved and "destination" in locals() and destination.exists():
                # Only this import's exclusively created version is disposable;
                # re-resolve its containment before recursive rollback cleanup.
                check_plain_plugin_path(destination)
                if self.store.plugins_root.resolve() not in destination.resolve().parents:
                    raise PluginManagerError("PLUGIN_ROLLBACK_DESTINATION_INVALID") from None
                shutil.rmtree(destination, ignore_errors=True)
            raise
        finally:
            if staging.exists():
                check_plain_plugin_path(staging)
                if self.store.plugins_root.resolve() not in staging.resolve().parents:
                    raise PluginManagerError("PLUGIN_STAGING_DESTINATION_INVALID")
                shutil.rmtree(staging, ignore_errors=True)

    # 功能：
    #   重新校验存储的清单字典，不直接信任可变记录中的元数据。
    # 输入：
    #   plugin：包含 manifest 字段的插件记录。
    # 输出：
    #   manifest：通过正式契约校验的清单对象。
    @staticmethod
    def _manifest(plugin: dict[str, object]) -> PluginManifest:
        value = plugin.get("manifest")
        if not isinstance(value, dict):
            raise PluginManagerError("PLUGIN_MANIFEST_RECORD_INVALID")
        manifest = PluginManifest.model_validate(value)
        return manifest

    # 功能：
    #   启动隔离客户端枚举工具前重新核对安装字节，避免执行被替换的负载。
    # 输入：
    #   plugin：安装记录。
    #   manifest：对应清单与摘要列表。
    # 输出：
    #   tools：插件实际报告的工具目录。
    def _mcp_tools(
        self, plugin: dict[str, object], manifest: PluginManifest
    ) -> list[dict[str, Any]]:
        root = Path(str(plugin["bundle_root"]))
        self._verify_integrity(root, manifest)
        try:
            tools = self._mcp_client(plugin, manifest).list_tools()
            return tools
        except PluginProcessError as error:
            raise PluginManagerError(str(error)) from error

    # 功能：
    #   在生命周期锁内检查插件健康，避免检查旧版本的结果覆盖同时发生的新选择。
    # 输入：
    #   plugin_id：待检查插件标识。
    # 输出：
    #   result：检查后的插件记录与生命周期回执。
    def healthcheck(self, plugin_id: str) -> dict[str, object]:
        with self._mutation_lock:
            result = self._healthcheck(plugin_id)
            return result

    # 功能：
    #   1. 检查内置定义或声明式文档，外部工具枚举前重新验证信任和治理许可。
    #   2. 检查失败时隔离并记录回执；健康检查不代表飞行资格，也不自动启用插件。
    # 输入：
    #   plugin_id：调用方已持有生命周期锁的目标插件标识。
    # 输出：
    #   result：检查后的插件记录与生命周期回执。
    def _healthcheck(self, plugin_id: str) -> dict[str, object]:
        plugin = self.store.get_plugin(plugin_id)
        manifest = self._manifest(plugin)
        previous = str(plugin["status"])
        try:
            if bool(plugin["builtin"]):
                definition = self._definitions.get(plugin_id)
                if definition is None or definition.manifest != manifest:
                    raise PluginManagerError("BUILTIN_PLUGIN_DEFINITION_MISMATCH")
            elif manifest.runtime.kind == "mcp-stdio":
                # 枚举工具也会启动外部代码，不能把“只检查健康”当成绕过执行许可的入口。
                plugin = self._check_activation_trust(plugin)
                catalog = self._mcp_tools(plugin, manifest)
                expected = {
                    item.capability_id: item
                    for item in manifest.capabilities
                    if item.kind in PLUGIN_MCP_CAPABILITY_KINDS
                }
                if any(
                    not isinstance(item.get("name"), str) or not item["name"] for item in catalog
                ):
                    raise PluginManagerError("PLUGIN_TOOL_CATALOG_INVALID")
                actual = {item["name"]: item for item in catalog}
                if len(actual) != len(catalog):
                    raise PluginManagerError("PLUGIN_TOOL_CATALOG_DUPLICATE")
                if set(expected) != set(actual):
                    raise PluginManagerError("PLUGIN_TOOL_CATALOG_MISMATCH")
                for tool_id, capability in expected.items():
                    tool = actual[tool_id]
                    if tool.get("inputSchema") != capability.input_schema:
                        raise PluginManagerError("PLUGIN_TOOL_INPUT_SCHEMA_MISMATCH")
                    if tool.get("outputSchema") != capability.output_schema:
                        raise PluginManagerError("PLUGIN_TOOL_OUTPUT_SCHEMA_MISMATCH")
            elif manifest.runtime.kind in {"ros2-node", "ui-declarative"}:
                root = Path(str(plugin["bundle_root"]))
                self._verify_integrity(root, manifest)
                self._validate_static_bundle(root, manifest)
            self.store.set_plugin_lifecycle(
                plugin_id,
                status="healthy" if bool(plugin["enabled"]) else "disabled",
                health="healthy",
            )
        except (PluginManagerError, ValidationError) as error:
            failed = self.store.set_plugin_lifecycle(
                plugin_id,
                enabled=False,
                status="quarantined",
                health="failed",
                last_error=_failure_issue(error),
            )
            self._receipt(
                plugin=failed,
                operation="quarantine",
                previous_state=previous,
                current_state="quarantined",
                accepted=True,
                issue_codes=[_failure_issue(error)],
            )
            raise
        checked = self.store.get_plugin(plugin_id)
        receipt = self._receipt(
            plugin=checked,
            operation="healthcheck",
            previous_state=previous,
            current_state=str(checked["status"]),
            accepted=True,
        )
        result = {**checked, "receipt": receipt}
        return result

    # 功能：
    #   在生命周期锁内启用插件，并绑定用户或 Harness 设计者明确选择的来源。
    # 输入：
    #   plugin_id：目标插件标识。
    #   selected_by：选择来源。
    #   harness_revision_sha256：设计者选择所绑定的配置摘要。
    # 输出：
    #   result：启用记录、替换信息和回执。
    def enable(
        self,
        plugin_id: str,
        *,
        selected_by: str = "account_configurable",
        harness_revision_sha256: str | None = None,
    ) -> dict[str, object]:
        with self._mutation_lock:
            result = self._enable(
                plugin_id,
                selected_by=selected_by,
                harness_revision_sha256=harness_revision_sha256,
            )
            return result

    # 功能：
    #   重新核对激活信任、撤销与治理限制，单独激活和整组切换使用同一规则。
    # 输入：
    #   plugin：候选插件记录。
    # 输出：
    #   plugin：完成信任同步且获准激活的记录。
    def _check_activation_trust(self, plugin: dict[str, object]) -> dict[str, object]:
        manifest = self._manifest(plugin)
        if not bool(plugin["builtin"]):
            try:
                decision = self.trust_store.verify(manifest, str(plugin["package_sha256"]))
            except TrustStoreError as error:
                raise PluginManagerError(str(error)) from error
            official = self._official_packages.get((manifest.plugin_id, manifest.version))
            # 安装器核验的官方包可以没有独立发布者签名，但明确撤销仍优先于安装器信任。
            if decision.status == "revoked" or official != plugin["package_sha256"]:
                self.store.set_plugin_trust(
                    manifest.plugin_id,
                    version=manifest.version,
                    trust_status=decision.status,
                    trust_decision=decision.model_dump(mode="json"),
                )
                plugin = self.store.get_plugin(manifest.plugin_id)
                if decision.status not in {"verified", "local-approved"}:
                    raise PluginManagerError(
                        "PLUGIN_TRUST_REQUIRED:" + ",".join(decision.issue_codes)
                    )
        self._govern(
            manifest=manifest, operation="enable", trust_status=str(plugin["trust_status"])
        )
        return plugin

    # 功能：
    #   1. 核对信任、依赖和插槽约束，检查健康并等待被替换插件的调用退出。
    #   2. 原子发布新选择及来源回执；旧会话关闭或发布失败时恢复临时退出状态。
    # 输入：
    #   plugin_id：目标插件标识。
    #   selected_by：可选的明确选择来源。
    #   harness_revision_sha256：设计者选择所绑定的配置摘要。
    # 输出：
    #   result：新启用记录、被替换插件、受影响任务及回执。
    def _enable(
        self,
        plugin_id: str,
        *,
        selected_by: str | None = None,
        harness_revision_sha256: str | None = None,
    ) -> dict[str, object]:
        if selected_by not in {None, "account_configurable", "agent_harness_designer"}:
            raise PluginManagerError("PLUGIN_SELECTION_SURFACE_INVALID")
        if selected_by == "agent_harness_designer" and harness_revision_sha256 is None:
            raise PluginManagerError("PLUGIN_HARNESS_REVISION_BINDING_REQUIRED")
        if selected_by != "agent_harness_designer" and harness_revision_sha256 is not None:
            raise PluginManagerError("PLUGIN_HARNESS_REVISION_BINDING_INVALID")
        plugin = self.store.get_plugin(plugin_id)
        manifest = self._manifest(plugin)
        plugin = self._check_activation_trust(plugin)
        previous = str(plugin["status"])
        self._validate_dependencies(manifest, require_enabled=True)
        self._validate_conflicts(manifest)
        self._validate_slot_contract(manifest)
        self.healthcheck(plugin_id)
        peers: list[tuple[dict[str, object], PluginManifest]] = []
        if manifest.placement.activation_mode == "single":
            for candidate in self.store.list_plugins():
                if candidate["plugin_id"] == plugin_id or not bool(candidate["enabled"]):
                    continue
                candidate_manifest = self._manifest(candidate)
                if candidate_manifest.placement.slot_id == manifest.placement.slot_id:
                    peers.append((candidate, candidate_manifest))
        affected_threads: list[str] = []
        for _candidate, candidate_manifest in peers:
            dependents = self._dependents(candidate_manifest.plugin_id, enabled_only=True)
            if dependents:
                raise PluginManagerError("PLUGIN_REQUIRED_BY_ENABLED:" + ",".join(dependents))
        draining = {
            str(candidate["plugin_id"]): {
                "enabled": True,
                "status": "draining",
                "health": str(candidate["health"]),
            }
            for candidate, _candidate_manifest in peers
        }
        if draining:
            self.store.set_plugin_lifecycles(draining)
        try:
            for _candidate, candidate_manifest in peers:
                self._drain_calls(candidate_manifest.plugin_id)
                if self._disable_guard:
                    affected_threads.extend(
                        self._disable_guard(
                            candidate_manifest.plugin_id,
                            candidate_manifest.placement.swap_policy,
                        )
                    )
        except (RuntimeError, PluginManagerError) as error:
            if peers:
                self.store.set_plugin_lifecycles(
                    {
                        str(candidate["plugin_id"]): {
                            "enabled": bool(candidate["enabled"]),
                            "status": str(candidate["status"]),
                            "health": str(candidate["health"]),
                            "last_error": candidate.get("last_error"),
                        }
                        for candidate, _candidate_manifest in peers
                    }
                )
            raise PluginManagerError(f"PLUGIN_SWAP_FAILED:{error}") from error
        updates: dict[str, dict[str, object]] = {
            plugin_id: {
                "enabled": True,
                "status": "healthy",
                "health": "healthy",
                "last_error": None,
            }
        }
        for candidate, _candidate_manifest in peers:
            updates[str(candidate["plugin_id"])] = {
                "enabled": False,
                "status": "disabled",
                "health": "unknown",
                "last_error": None,
            }
        try:
            # 关闭会话也可能失败，必须与数据库发布共享恢复边界，不能残留 draining。
            for candidate, _candidate_manifest in peers:
                self._mcp_sessions.invalidate(str(candidate["plugin_id"]))
            with self.store.atomic_plugin_changes():
                self.store.set_plugin_lifecycles(updates)
                enabled = self.store.get_plugin(plugin_id)
                replaced = [str(candidate["plugin_id"]) for candidate, _manifest in peers]
                replacement_receipts = [
                    self._receipt(
                        plugin=self.store.get_plugin(replaced_id),
                        operation="disable",
                        previous_state="healthy",
                        current_state="disabled",
                        accepted=True,
                    )
                    for replaced_id in replaced
                ]
                receipt = self._receipt(
                    plugin=enabled,
                    operation="enable",
                    previous_state=previous,
                    current_state="healthy",
                    accepted=True,
                )
                if selected_by is not None:
                    try:
                        enabled = self.store.set_plugin_selection_provenance(
                            plugin_id,
                            selection_source="explicit",
                            selected_by=selected_by,  # type: ignore[arg-type]
                            selection_receipt_sha256=sha256_json(receipt),
                            harness_revision_sha256=harness_revision_sha256,
                        )
                    except ValueError as error:
                        raise PluginManagerError(str(error)) from error
                result = {
                    **enabled,
                    "receipt": receipt,
                    "replacement_receipts": replacement_receipts,
                    "replaced_plugins": replaced,
                    "affected_threads": sorted(set(affected_threads)),
                }
                return result
        except BaseException:
            self._restore_lifecycles([candidate for candidate, _manifest in peers])
            raise

    # 功能：
    #   操作失败后按原记录撤回临时退出状态，保留原先启用、健康及错误字段。
    # 输入：
    #   records：操作前的插件状态记录。
    # 输出：
    #   None：不返回业务数据。
    def _restore_lifecycles(self, records: list[dict[str, object]]) -> None:
        if records:
            self.store.set_plugin_lifecycles(
                {
                    str(record["plugin_id"]): {
                        "enabled": bool(record["enabled"]),
                        "status": str(record["status"]),
                        "health": str(record["health"]),
                        "last_error": record.get("last_error"),
                    }
                    for record in records
                }
            )

    # 功能：
    #   串行执行插件停用，新目录准入与既有任务的冻结绑定分别处理。
    # 输入：
    #   plugin_id：待停用插件标识。
    # 输出：
    #   result：停用状态及受影响任务信息。
    def disable(self, plugin_id: str) -> dict[str, object]:
        with self._mutation_lock:
            result = self._disable(plugin_id)
            return result

    # 功能：
    #   保护必需插槽和依赖，等待调用退出后关闭会话并停用；关闭或发布失败时恢复原状态。
    # 输入：
    #   plugin_id：待停用插件标识。
    #   replacement_plugin_id：内部替换操作可提供的后继插件标识。
    # 输出：
    #   result：停用记录、回执和受影响任务列表。
    def _disable(
        self, plugin_id: str, *, replacement_plugin_id: str | None = None
    ) -> dict[str, object]:
        plugin = self.store.get_plugin(plugin_id)
        manifest = self._manifest(plugin)
        if not manifest.disable_allowed:
            raise PluginManagerError("PLUGIN_REQUIRED_BY_SAFETY_KERNEL")
        if (
            manifest.placement.slot_id in self.required_single_slots
            and replacement_plugin_id is None
            and not any(
                bool(candidate["enabled"])
                and candidate["plugin_id"] != plugin_id
                and self._manifest(candidate).placement.slot_id == manifest.placement.slot_id
                for candidate in self.store.list_plugins()
            )
        ):
            raise PluginManagerError("PLUGIN_SLOT_REQUIRES_ONE_ENABLED")
        dependents = self._dependents(plugin_id, enabled_only=True)
        if dependents:
            raise PluginManagerError("PLUGIN_REQUIRED_BY_ENABLED:" + ",".join(dependents))
        previous = str(plugin["status"])
        self.store.set_plugin_lifecycle(
            plugin_id,
            enabled=bool(plugin["enabled"]),
            status="draining",
            health=str(plugin["health"]),
        )
        try:
            self._drain_calls(plugin_id)
            affected_threads = (
                self._disable_guard(plugin_id, manifest.placement.swap_policy)
                if self._disable_guard
                else []
            )
        except (RuntimeError, PluginManagerError) as error:
            self.store.set_plugin_lifecycle(
                plugin_id,
                enabled=bool(plugin["enabled"]),
                status=previous,
                health=str(plugin["health"]),
                last_error=plugin.get("last_error"),
            )
            raise PluginManagerError(f"PLUGIN_DRAIN_FAILED:{error}") from error
        try:
            self._mcp_sessions.invalidate(plugin_id)
            with self.store.atomic_plugin_changes():
                disabled = self.store.set_plugin_lifecycle(
                    plugin_id, enabled=False, status="disabled", health="unknown"
                )
                receipt = self._receipt(
                    plugin=disabled,
                    operation="disable",
                    previous_state=previous,
                    current_state="disabled",
                    accepted=True,
                )
                result = {**disabled, "receipt": receipt, "affected_threads": affected_threads}
                return result
        except BaseException:
            self._restore_lifecycles([plugin])
            raise

    # 功能：
    #   选择已安装的历史版本并保持禁用，不在回滚选择时悄悄执行历史代码。
    # 输入：
    #   plugin_id：需要恢复的插件标识。
    #   version：已安装目标版本。
    # 输出：
    #   result：恢复选择的记录与回执。
    def rollback(self, plugin_id: str, version: str) -> dict[str, object]:
        result = self.activate_version(plugin_id, version, operation="rollback")
        return result

    # 功能：
    #   1. 按更新通道策略切换版本并检查健康，选择、校验和回执失败均进入恢复流程。
    #   2. 先回收失败版本再恢复旧版本；回收或恢复失败时不谎报回滚成功，也不强行重新启用。
    # 输入：
    #   plugin_id：目标插件标识。
    #   version：已安装且准备提升的版本。
    # 输出：
    #   result：当前版本、是否发生提升及验证和生命周期证据。
    def promote_version(self, plugin_id: str, version: str) -> dict[str, object]:
        with self._mutation_lock:
            current = self.store.get_plugin(plugin_id)
            previous_version = str(current["version"])
            previous_enabled = bool(current["enabled"])
            if previous_version == version:
                result = {**current, "promoted": False, "rollback": False}
                return result
            target = self.store.get_plugin_version(plugin_id, version)
            target_manifest = PluginManifest.model_validate(target["manifest"])
            self._govern(
                manifest=target_manifest,
                operation="promote",
                trust_status=str(target["trust_status"]),
            )
            selected_ring = str(self.store.get_settings().get("plugin_update_ring", "stable"))
            allowed_rings = {
                "stable": {"stable"},
                "preview": {"stable", "preview"},
                "canary": {"stable", "preview", "canary"},
                "pinned": set(),
            }
            if target_manifest.provenance.update_ring not in allowed_rings.get(
                selected_ring, {"stable"}
            ):
                raise PluginManagerError("PLUGIN_UPDATE_RING_DENIED")
            if previous_enabled:
                self._disable(plugin_id)
            try:
                self._mcp_sessions.invalidate(plugin_id)
                self.store.activate_plugin_version(plugin_id, version)
                promoted = (
                    self._enable(plugin_id) if previous_enabled else self.healthcheck(plugin_id)
                )
                selected = self.store.get_plugin(plugin_id)
                promotion_receipt = self._receipt(
                    plugin=selected,
                    operation="update",
                    previous_state=str(current["status"]),
                    current_state=str(selected["status"]),
                    accepted=True,
                    issue_codes=[
                        "PLUGIN_PROMOTION_HEALTH_GATED",
                        f"PREVIOUS_VERSION_{previous_version}",
                        f"ACTIVE_VERSION_{version}",
                    ],
                )
                result = {
                    **selected,
                    "promoted": True,
                    "rollback": False,
                    "receipt": promotion_receipt,
                    "healthcheck": promoted,
                }
                return result
            except BaseException as error:
                # 覆盖普通 I/O、数据库和中断异常，不能只恢复被包装为领域错误的失败。
                recovery_errors: list[Exception] = []
                try:
                    self._mcp_sessions.invalidate(plugin_id)
                except Exception as cleanup_error:
                    recovery_errors.append(cleanup_error)
                try:
                    self.store.activate_plugin_version(plugin_id, previous_version)
                    # 无法确认新会话退出时，只恢复版本选择，绝不再启动另一个版本。
                    if not recovery_errors:
                        if previous_enabled:
                            self._enable(plugin_id)
                        else:
                            self.healthcheck(plugin_id)
                except Exception as rollback_error:
                    recovery_errors.append(rollback_error)
                restored = self.store.get_plugin(plugin_id)
                rollback_receipt: dict[str, object] | None = None
                try:
                    rollback_receipt = self._receipt(
                        plugin=restored,
                        operation="rollback",
                        previous_state="failed",
                        current_state=str(restored["status"]),
                        accepted=not recovery_errors,
                        issue_codes=[
                            "PLUGIN_PROMOTION_AUTO_ROLLBACK",
                            f"FAILED_VERSION_{version}",
                            f"ACTIVE_VERSION_{restored['version']}",
                            _failure_issue(error),
                        ]
                        + [f"ROLLBACK_FAILED:{type(item).__name__}" for item in recovery_errors],
                    )
                except Exception as receipt_error:
                    recovery_errors.append(receipt_error)
                # 清理不能将用户中断替换成普通业务错误；恢复失败的细节仍保留为异常原因。
                if not isinstance(error, Exception):
                    if recovery_errors:
                        raise error from ExceptionGroup(
                            "Plugin promotion recovery", recovery_errors
                        )
                    raise
                if recovery_errors:
                    raise PluginManagerError(
                        f"PLUGIN_PROMOTION_AND_ROLLBACK_FAILED:{error}"
                    ) from ExceptionGroup("Plugin promotion recovery", [error, *recovery_errors])
                assert rollback_receipt is not None
                raise PluginManagerError(
                    f"PLUGIN_PROMOTION_ROLLED_BACK:{error}:{rollback_receipt['receipt_id']}"
                ) from error

    # 功能：
    #   在生命周期锁内选择已安装版本，先确认目标存在以免错误请求停用正常版本。
    # 输入：
    #   plugin_id：插件标识。
    #   version：已安装目标版本。
    #   operation：版本选择或回滚操作名称。
    # 输出：
    #   result：选择后的禁用记录和回执。
    def activate_version(
        self, plugin_id: str, version: str, *, operation: str = "activate"
    ) -> dict[str, object]:
        if operation not in {"activate", "rollback"}:
            raise PluginManagerError("PLUGIN_VERSION_OPERATION_INVALID")
        with self._mutation_lock:
            # 不存在的目标不能先停用当前正常版本。
            self.store.get_plugin_version(plugin_id, version)
            result = self._activate_version(plugin_id, version, operation=operation)
            return result

    # 功能：
    #   在调用方持有生命周期锁时提交已验证版本选择，失败时恢复原生命周期状态。
    # 输入：
    #   plugin_id：插件标识。
    #   version：已验证存在的目标版本。
    #   operation：版本选择或回滚操作名称。
    # 输出：
    #   result：选择后的禁用记录和回执。
    def _activate_version(
        self, plugin_id: str, version: str, *, operation: str
    ) -> dict[str, object]:
        current = self.store.get_plugin(plugin_id)
        manifest = self._manifest(current)
        if not manifest.removable:
            raise PluginManagerError("PLUGIN_REQUIRED_BY_SAFETY_KERNEL")
        if bool(current["enabled"]):
            self.disable(plugin_id)
        try:
            self._mcp_sessions.invalidate(plugin_id)
            with self.store.atomic_plugin_changes():
                rolled_back = self.store.activate_plugin_version(plugin_id, version)
                receipt = self._receipt(
                    plugin=rolled_back,
                    operation=operation,
                    previous_state=str(current["status"]),
                    current_state="disabled",
                    accepted=True,
                )
                result = {**rolled_back, "receipt": receipt}
                return result
        except BaseException:
            self._restore_lifecycles([current])
            raise

    # 功能：
    #   仅卸载可移除且无依赖、无在途使用的插件，保留生命周期证据，不卸载内置安全能力。
    # 输入：
    #   plugin_id：用户要求卸载的插件标识。
    # 输出：
    #   result：卸载状态与回执。
    def uninstall(self, plugin_id: str) -> dict[str, object]:
        with self._mutation_lock:
            plugin = self.store.get_plugin(plugin_id)
            manifest = self._manifest(plugin)
            if bool(plugin["builtin"]) or not manifest.removable:
                raise PluginManagerError("BUILTIN_PLUGIN_NOT_UNINSTALLABLE")
            dependents = self._dependents(plugin_id, enabled_only=False)
            if dependents:
                raise PluginManagerError("PLUGIN_REQUIRED_BY:" + ",".join(dependents))
            if self._disable_guard is not None:
                # 删除比下次任务生效的停用更强，活动任务仍拥有其冻结快照绑定的目录。
                try:
                    self._disable_guard(plugin_id, "restart")
                except RuntimeError as error:
                    raise PluginManagerError(f"PLUGIN_UNINSTALL_REQUIRES_IDLE:{error}") from error
            if bool(plugin["enabled"]):
                self._disable(plugin_id)
            self._drain_calls(plugin_id)
            self._mcp_sessions.invalidate(plugin_id)
            root = (self.store.plugins_root / plugin_id).resolve()
            plugins_root = self.store.plugins_root.resolve()
            if root.parent != plugins_root:
                raise PluginManagerError("PLUGIN_UNINSTALL_PATH_INVALID")
            if root.exists():
                shutil.rmtree(root)
            removed = self.store.mark_plugin_uninstalled(plugin_id)
            receipt = self._receipt(
                plugin=removed,
                operation="uninstall",
                previous_state=str(plugin["status"]),
                current_state="uninstalled",
                accepted=True,
            )
            result = {**removed, "receipt": receipt}
            return result

    # 功能：
    #   在生命周期锁内校验并替换配置，使旧配置的客户端失效。
    # 输入：
    #   plugin_id：目标插件标识。
    #   configuration：准备保存的完整配置对象。
    # 输出：
    #   result：持久化配置记录。
    def configure(self, plugin_id: str, configuration: dict[str, object]) -> dict[str, object]:
        with self._mutation_lock:
            result = self._configure(plugin_id, configuration)
            return result

    # 功能：
    #   按治理规则为当前选定的精确包与清单记录本地批准，不自动启用插件。
    # 输入：
    #   plugin_id：用户明确批准的插件标识。
    # 输出：
    #   result：原内置记录或外部插件信任记录及回执。
    def approve_local_package(self, plugin_id: str) -> dict[str, object]:
        with self._mutation_lock:
            plugin = self.store.get_plugin(plugin_id)
            if bool(plugin["builtin"]):
                result = plugin
                return result
            manifest = self._manifest(plugin)
            self._govern(
                manifest=manifest,
                operation="trust-local-package",
                trust_status=str(plugin["trust_status"]),
            )
            package_sha256 = str(plugin["package_sha256"])
            try:
                self.trust_store.approve_local(manifest, package_sha256)
                decision = self.trust_store.verify(manifest, package_sha256)
            except TrustStoreError as error:
                raise PluginManagerError(str(error)) from error
            trusted = self.store.set_plugin_trust(
                plugin_id,
                version=manifest.version,
                trust_status=decision.status,
                trust_decision=decision.model_dump(mode="json"),
            )
            receipt = self._receipt(
                plugin=trusted,
                operation="trust-local-package",
                previous_state=str(plugin["status"]),
                current_state=str(trusted["status"]),
                accepted=True,
                issue_codes=[f"TRUST_STATUS:{decision.status}"],
            )
            result = {**trusted, "receipt": receipt}
            return result

    # 功能：
    #   为已安装或暂存的指定版本记录精确包批准，不把批准自动转移到当前其他版本。
    # 输入：
    #   plugin_id：插件标识。
    #   version：用户批准的已安装版本。
    # 输出：
    #   result：该版本的信任记录与回执，内置版本直接返回原记录。
    def approve_local_version(self, plugin_id: str, version: str) -> dict[str, object]:
        with self._mutation_lock:
            current = self.store.get_plugin(plugin_id)
            if bool(current["builtin"]):
                result = self.store.get_plugin_version(plugin_id, version)
                return result
            target = self.store.get_plugin_version(plugin_id, version)
            manifest = PluginManifest.model_validate(target["manifest"])
            self._govern(
                manifest=manifest,
                operation="trust-local-package",
                trust_status=str(target["trust_status"]),
            )
            package_sha256 = str(target["package_sha256"])
            try:
                self.trust_store.approve_local(manifest, package_sha256)
                decision = self.trust_store.verify(manifest, package_sha256)
            except TrustStoreError as error:
                raise PluginManagerError(str(error)) from error
            trusted = self.store.set_plugin_version_trust(
                plugin_id,
                version=version,
                trust_status=decision.status,
                trust_decision=decision.model_dump(mode="json"),
            )
            receipt_plugin = {
                "plugin_id": plugin_id,
                "version": version,
                "package_sha256": package_sha256,
                "status": current["status"],
            }
            receipt = self._receipt(
                plugin=receipt_plugin,
                operation="trust-local-package",
                previous_state=str(current["status"]),
                current_state=str(current["status"]),
                accepted=True,
                issue_codes=[f"TRUST_STATUS:{decision.status}", "STAGED_VERSION_APPROVAL"],
            )
            result = {**trusted, "receipt": receipt}
            return result

    # 功能：
    #   停用当前外部插件后持久撤销其精确包摘要，不允许本地撤销产品内置根信任。
    # 输入：
    #   plugin_id：当前选定的外部插件标识。
    # 输出：
    #   result：撤销后的信任记录和回执。
    def revoke_package(self, plugin_id: str) -> dict[str, object]:
        with self._mutation_lock:
            plugin = self.store.get_plugin(plugin_id)
            if bool(plugin["builtin"]):
                raise PluginManagerError("BUILTIN_TRUST_CANNOT_BE_REVOKED_LOCALLY")
            if bool(plugin["enabled"]):
                self._disable(plugin_id)
                plugin = self.store.get_plugin(plugin_id)
            manifest = self._manifest(plugin)
            package_sha256 = str(plugin["package_sha256"])
            try:
                self.trust_store.revoke_package(package_sha256)
                decision = self.trust_store.verify(manifest, package_sha256)
            except TrustStoreError as error:
                raise PluginManagerError(str(error)) from error
            revoked = self.store.set_plugin_trust(
                plugin_id,
                version=manifest.version,
                trust_status=decision.status,
                trust_decision=decision.model_dump(mode="json"),
            )
            receipt = self._receipt(
                plugin=revoked,
                operation="revoke-package",
                previous_state=str(plugin["status"]),
                current_state=str(revoked["status"]),
                accepted=True,
                issue_codes=[f"TRUST_STATUS:{decision.status}"],
            )
            result = {**revoked, "receipt": receipt}
            return result

    # 功能：
    #   校验并登记发布者公钥绑定，不执行导入或激活插件。
    # 输入：
    #   key_id：发布者密钥稳定标识。
    #   publisher：发布者名称。
    #   public_key_base64：Ed25519 公钥 Base64 文本。
    # 输出：
    #   result：登记成功的公钥标识与发布者信息。
    def add_trusted_publisher(
        self, *, key_id: str, publisher: str, public_key_base64: str
    ) -> dict[str, object]:
        try:
            self.trust_store.add_publisher(
                key_id=key_id,
                publisher=publisher,
                public_key_base64=public_key_base64,
            )
        except TrustStoreError as error:
            raise PluginManagerError(str(error)) from error
        result = {"key_id": key_id, "publisher": publisher, "trusted": True}
        return result

    # 功能：
    #   先限制 JSON 大小与深度，再拒绝内联凭据键并验证 Schema，关闭旧会话后保存配置。
    # 输入：
    #   plugin_id：目标插件标识。
    #   configuration：完整配置对象。
    # 输出：
    #   result：持久配置记录。
    def _configure(self, plugin_id: str, configuration: dict[str, object]) -> dict[str, object]:
        try:
            configuration = copy_json(configuration, limit=256_000)
        except ValueError as error:
            raise PluginManagerError("PLUGIN_CONFIGURATION_INVALID_JSON") from error
        if not isinstance(configuration, dict):
            raise PluginManagerError("PLUGIN_CONFIGURATION_OBJECT_REQUIRED")
        plugin = self.store.get_plugin(plugin_id)
        manifest = self._manifest(plugin)
        sensitive_fragments = {"api_key", "apikey", "authorization", "password", "secret"}

        # 功能：
        #   在已通过深度与大小预算的 JSON 中拒绝凭据特征键，不把值或完整路径写入错误。
        # 输入：
        #   value：当前待遍历的 JSON 子值。
        #   path：递归访问路径，不输出到日志。
        # 输出：
        #   None：不返回业务数据。
        def reject_inline_secrets(value: object, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized = str(key).casefold().replace("-", "_")
                    if any(fragment in normalized for fragment in sensitive_fragments):
                        raise PluginManagerError("PLUGIN_CONFIGURATION_INLINE_SECRET_FORBIDDEN")
                    reject_inline_secrets(item, (*path, str(key)))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    reject_inline_secrets(item, (*path, str(index)))

        reject_inline_secrets(configuration)
        schema = manifest.configuration_schema
        if schema:
            jsonschema.validate(configuration, schema)
        elif configuration:
            raise PluginManagerError("PLUGIN_HAS_NO_CONFIGURATION")
        self._mcp_sessions.invalidate(plugin_id)
        result = self.store.save_plugin_configuration(plugin_id, configuration)
        return result

    # 功能：
    #   校验拟启用集合的必需能力、插槽数量、依赖、冲突与管线拓扑，不提前改变实际目录。
    # 输入：
    #   enabled_ids：拟启用插件标识集合。
    # 输出：
    #   None：不返回业务数据。
    def _validate_enabled_catalog(self, enabled_ids: set[str]) -> None:
        records = {str(item["plugin_id"]): item for item in self.store.list_plugins()}
        missing = sorted(enabled_ids - records.keys())
        if missing:
            raise PluginManagerError("PLUGIN_PROFILE_MEMBER_MISSING:" + ",".join(missing))
        manifests = {plugin_id: self._manifest(records[plugin_id]) for plugin_id in enabled_ids}
        for plugin_id, record in records.items():
            manifest = self._manifest(record)
            if not manifest.disable_allowed and plugin_id not in enabled_ids:
                raise PluginManagerError("PLUGIN_REQUIRED_BY_SAFETY_KERNEL")
        slot_members: dict[str, list[PluginManifest]] = {}
        for manifest in manifests.values():
            slot_members.setdefault(manifest.placement.slot_id, []).append(manifest)
        for slot_id, members in slot_members.items():
            if len({item.placement.activation_mode for item in members}) != 1:
                raise PluginManagerError(f"PLUGIN_SLOT_ACTIVATION_MODE_CONFLICT:{slot_id}")
            if members[0].placement.activation_mode == "single" and len(members) != 1:
                raise PluginManagerError(f"PLUGIN_SINGLE_SLOT_CARDINALITY:{slot_id}")
        missing_required = sorted(
            slot_id
            for slot_id in self.required_single_slots
            if len(slot_members.get(slot_id, [])) != 1
        )
        if missing_required:
            raise PluginManagerError(
                "PLUGIN_REQUIRED_SLOT_CARDINALITY:" + ",".join(missing_required)
            )
        for plugin_id, manifest in manifests.items():
            for dependency in manifest.dependencies:
                if dependency.optional:
                    continue
                dependency_manifest = manifests.get(dependency.plugin_id)
                if dependency_manifest is None or not _version_matches(
                    dependency_manifest.version, dependency.version
                ):
                    raise PluginManagerError(
                        f"PLUGIN_DEPENDENCY_UNAVAILABLE:{plugin_id}:{dependency.plugin_id}"
                    )
            conflicts = sorted(set(manifest.conflicts) & enabled_ids)
            if conflicts:
                raise PluginManagerError(f"PLUGIN_CONFLICT:{plugin_id}:" + ",".join(conflicts))
        for slot_id, members in slot_members.items():
            if members[0].placement.activation_mode != "pipeline":
                continue
            by_id = {item.plugin_id: item for item in members}
            edges: dict[str, set[str]] = {plugin_id: set() for plugin_id in by_id}
            indegree: dict[str, int] = {plugin_id: 0 for plugin_id in by_id}
            for item in members:
                for predecessor in item.placement.runs_after:
                    if predecessor in by_id and item.plugin_id not in edges[predecessor]:
                        edges[predecessor].add(item.plugin_id)
                        indegree[item.plugin_id] += 1
                for successor in item.placement.runs_before:
                    if successor in by_id and successor not in edges[item.plugin_id]:
                        edges[item.plugin_id].add(successor)
                        indegree[successor] += 1
            ready = [plugin_id for plugin_id, degree in indegree.items() if degree == 0]
            visited = 0
            while ready:
                current = ready.pop()
                visited += 1
                for successor in edges[current]:
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        ready.append(successor)
            if visited != len(by_id):
                raise PluginManagerError(f"PLUGIN_PIPELINE_CYCLE:{slot_id}")

    # 功能：
    #   原子选择 Harness 组合并绑定选择来源，保留既有任务快照，失败时恢复临时目录状态。
    # 输入：
    #   profile_plugin_id：提供推荐和管理插件集合的配置插件标识。
    #   selected_by：用户或设计者选择来源。
    #   harness_revision_sha256：设计者选择绑定的配置摘要。
    # 输出：
    #   result：启停插件列表、受影响任务、新目录摘要和生命周期回执。
    def apply_profile(
        self,
        profile_plugin_id: str,
        *,
        selected_by: str = "account_configurable",
        harness_revision_sha256: str | None = None,
    ) -> dict[str, object]:
        if selected_by not in {"account_configurable", "agent_harness_designer"}:
            raise PluginManagerError("PLUGIN_SELECTION_SURFACE_INVALID")
        if selected_by == "agent_harness_designer" and harness_revision_sha256 is None:
            raise PluginManagerError("PLUGIN_HARNESS_REVISION_BINDING_REQUIRED")
        if selected_by != "agent_harness_designer" and harness_revision_sha256 is not None:
            raise PluginManagerError("PLUGIN_HARNESS_REVISION_BINDING_INVALID")
        with self._mutation_lock:
            profile_record = self.store.get_plugin(profile_plugin_id)
            profile_manifest = self._manifest(profile_record)
            if profile_manifest.placement.slot_id != "harness.profile":
                raise PluginManagerError("PLUGIN_NOT_HARNESS_PROFILE")
            capability = next(
                (item for item in profile_manifest.capabilities if item.kind == "harness-profile"),
                None,
            )
            if capability is None:
                raise PluginManagerError("PLUGIN_PROFILE_CAPABILITY_MISSING")
            recommended_value = capability.metadata.get("recommended_plugins", [])
            managed_value = capability.metadata.get("managed_plugins", [])
            if (
                not isinstance(recommended_value, list)
                or not all(isinstance(item, str) for item in recommended_value)
                or not isinstance(managed_value, list)
                or not all(isinstance(item, str) for item in managed_value)
            ):
                raise PluginManagerError("PLUGIN_PROFILE_METADATA_INVALID")
            recommended = set(recommended_value)
            recommended.add(profile_plugin_id)
            managed = set(managed_value)
            records = {str(item["plugin_id"]): item for item in self.store.list_plugins()}
            missing = sorted((recommended | managed) - records.keys())
            if missing:
                raise PluginManagerError("PLUGIN_PROFILE_MEMBER_MISSING:" + ",".join(missing))
            desired_enabled = {
                plugin_id for plugin_id, record in records.items() if bool(record["enabled"])
            }
            for plugin_id in recommended:
                manifest = self._manifest(records[plugin_id])
                if manifest.placement.activation_mode == "single":
                    desired_enabled -= {
                        candidate_id
                        for candidate_id, candidate in records.items()
                        if self._manifest(candidate).placement.slot_id == manifest.placement.slot_id
                    }
                desired_enabled.add(plugin_id)
            desired_enabled -= managed - recommended
            self._validate_enabled_catalog(desired_enabled)
            for plugin_id in sorted(recommended):
                self._check_activation_trust(records[plugin_id])
                self.healthcheck(plugin_id)
            disabled_ids = sorted(
                plugin_id
                for plugin_id, record in records.items()
                if bool(record["enabled"]) and plugin_id not in desired_enabled
            )
            enabled_ids = sorted(
                plugin_id
                for plugin_id, record in records.items()
                if not bool(record["enabled"]) and plugin_id in desired_enabled
            )
            affected_threads: list[str] = []
            if disabled_ids:
                self.store.set_plugin_lifecycles(
                    {
                        plugin_id: {
                            "enabled": True,
                            "status": "draining",
                            "health": str(records[plugin_id]["health"]),
                        }
                        for plugin_id in disabled_ids
                    }
                )
            try:
                for plugin_id in disabled_ids:
                    manifest = self._manifest(records[plugin_id])
                    self._drain_calls(plugin_id)
                    if self._disable_guard:
                        affected_threads.extend(
                            self._disable_guard(plugin_id, manifest.placement.swap_policy)
                        )
            except (RuntimeError, PluginManagerError) as error:
                self.store.set_plugin_lifecycles(
                    {
                        plugin_id: {
                            "enabled": bool(records[plugin_id]["enabled"]),
                            "status": str(records[plugin_id]["status"]),
                            "health": str(records[plugin_id]["health"]),
                            "last_error": records[plugin_id].get("last_error"),
                        }
                        for plugin_id in disabled_ids
                    }
                )
                raise PluginManagerError(f"PLUGIN_PROFILE_DRAIN_FAILED:{error}") from error
            updates: dict[str, dict[str, object]] = {}
            for plugin_id in enabled_ids:
                updates[plugin_id] = {
                    "enabled": True,
                    "status": "healthy",
                    "health": "healthy",
                    "last_error": None,
                }
            for plugin_id in disabled_ids:
                updates[plugin_id] = {
                    "enabled": False,
                    "status": "disabled",
                    "health": "unknown",
                    "last_error": None,
                }
            before = {
                plugin_id: {
                    "enabled": bool(records[plugin_id]["enabled"]),
                    "status": str(records[plugin_id]["status"]),
                    "health": str(records[plugin_id]["health"]),
                    "last_error": records[plugin_id].get("last_error"),
                }
                for plugin_id in updates
            }
            selected_plugin_ids = sorted(recommended & desired_enabled)
            try:
                for plugin_id in disabled_ids:
                    self._mcp_sessions.invalidate(plugin_id)
                with self.store.atomic_plugin_changes():
                    self.store.set_plugin_lifecycles(updates)
                    for plugin_id in selected_plugin_ids:
                        self.store.set_plugin_selection_provenance(
                            plugin_id,
                            selection_source="explicit",
                            selected_by=selected_by,  # type: ignore[arg-type]
                            harness_revision_sha256=harness_revision_sha256,
                        )
                    snapshot = self.snapshot()
                    receipts = []
                    for plugin_id in enabled_ids:
                        receipts.append(
                            self._receipt(
                                plugin=self.store.get_plugin(plugin_id),
                                operation="enable",
                                previous_state=str(records[plugin_id]["status"]),
                                current_state="healthy",
                                accepted=True,
                            )
                        )
                    for plugin_id in disabled_ids:
                        receipts.append(
                            self._receipt(
                                plugin=self.store.get_plugin(plugin_id),
                                operation="disable",
                                previous_state=str(records[plugin_id]["status"]),
                                current_state="disabled",
                                accepted=True,
                            )
                        )
                    result = {
                        "profile_plugin_id": profile_plugin_id,
                        "enabled_plugins": enabled_ids,
                        "disabled_plugins": disabled_ids,
                        "affected_threads": sorted(set(affected_threads)),
                        "snapshot_id": snapshot.snapshot_id,
                        "catalog_sha256": snapshot.catalog_sha256,
                        "receipts": receipts,
                    }
                    return result
            except BaseException:
                self.store.set_plugin_lifecycles(before)
                raise

    # 功能：
    #   从健康活动插件建立唯一模型归属，发现、路由与视觉能力查询共用这一绑定。
    # 输入：
    #   无。
    # 输出：
    #   bindings：模型标识到所属清单、能力和模型元数据的映射。
    def _active_model_bindings(self) -> dict[str, tuple[PluginManifest, PluginCapability, dict]]:
        bindings: dict[str, tuple[PluginManifest, PluginCapability, dict]] = {}
        for plugin in self.store.list_plugins():
            if (
                not plugin["enabled"]
                or plugin["status"] != "healthy"
                or plugin["health"] != "healthy"
            ):
                continue
            manifest = self._manifest(plugin)
            for capability in manifest.capabilities:
                if capability.kind != "model-provider":
                    continue
                models = capability.metadata.get("models", [])
                if not isinstance(models, list):
                    continue
                for model in models:
                    if not isinstance(model, dict) or not all(
                        isinstance(model.get(key), str) and model[key]
                        for key in ("id", "label", "provider")
                    ):
                        continue
                    model_id = model["id"]
                    if model_id in bindings:
                        raise PluginManagerError(f"MODEL_PROVIDER_BINDING_AMBIGUOUS:{model_id}")
                    bindings[model_id] = manifest, capability, model
        return bindings

    # 功能：
    #   发布健康且归属唯一的模型目录，只有明确声明 true 的模型才标记支持图像。
    # 输入：
    #   无。
    # 输出：
    #   catalog：供界面和任务选择使用的模型列表。
    def model_catalog(self) -> list[dict[str, object]]:
        catalog = [
            {
                "id": model_id,
                "model": model_id,
                "label": value["label"],
                "provider": value["provider"],
                "icon": value["provider"],
                "source": "default",
                "supports_image_input": value.get("supports_image_input") is True,
            }
            for model_id, (_manifest, _capability, value) in sorted(
                self._active_model_bindings().items()
            )
        ]
        return catalog

    # 功能：
    #   检查指定供应商是否由启用且健康的模型供应商插件提供，不把正在退出的插件算可用。
    # 输入：
    #   provider：供应商标识。
    # 输出：
    #   enabled：是否存在匹配的活动供应商。
    def model_provider_enabled(self, provider: str) -> bool:
        for plugin in self.store.list_plugins():
            if (
                not plugin["enabled"]
                or plugin["health"] != "healthy"
                or plugin["status"] != "healthy"
            ):
                continue
            manifest = self._manifest(plugin)
            if manifest.placement.slot_id != "models.providers":
                continue
            if any(
                capability.kind == "model-provider"
                and capability.metadata.get("provider") == provider
                for capability in manifest.capabilities
            ):
                enabled = True
                return enabled
        enabled = False
        return enabled

    # 功能：
    #   合并核心声明式解析器与健康活动插件的适配器，损坏的可选描述不阻塞应用启动。
    # 输入：
    #   无。
    # 输出：
    #   values：按目录顺序排列的适配器、启用状态和提供方列表。
    def source_adapter_catalog(self) -> list[dict[str, object]]:
        catalog: dict[str, dict[str, object]] = {}
        order: list[str] = []
        for value in builtin_source_adapter_catalog():
            descriptor = AssetSourceAdapterDescriptor.model_validate(value)
            order.append(descriptor.adapter_id)
            catalog[descriptor.adapter_id] = {
                **descriptor.model_dump(mode="json"),
                "enabled": descriptor.availability == "builtin",
                "provider_plugin_ids": [],
            }
        for plugin in self.store.list_plugins():
            if (
                not plugin["enabled"]
                or plugin["health"] != "healthy"
                or plugin["status"] != "healthy"
            ):
                continue
            manifest = self._manifest(plugin)
            try:
                descriptors = self._asset_source_adapter_descriptors(
                    manifest, builtin=bool(plugin["builtin"])
                )
            except PluginManagerError:
                # 可选插件元数据损坏只使它退出发现结果，不应拖垮整个应用启动。
                continue
            for descriptor in descriptors:
                existing = catalog.get(descriptor.adapter_id)
                if existing is None:
                    order.append(descriptor.adapter_id)
                    existing = {
                        **descriptor.model_dump(mode="json"),
                        "enabled": False,
                        "provider_plugin_ids": [],
                    }
                    catalog[descriptor.adapter_id] = existing
                providers = existing["provider_plugin_ids"]
                if not isinstance(providers, list):
                    raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_CATALOG_INVALID")
                plugin_id = str(plugin["plugin_id"])
                if plugin_id not in providers:
                    providers.append(plugin_id)
                existing["enabled"] = True
        values = [catalog[adapter_id] for adapter_id in order]
        return values

    # 功能：
    #   校验导入器支持的资产类型、适配器标识唯一性及执行边界，禁止外部插件冒充核心解析器。
    # 输入：
    #   manifest：提供适配器元数据的清单。
    #   builtin：是否来自产品内置定义。
    # 输出：
    #   descriptors：经验证的资产适配器描述列表。
    @staticmethod
    def _asset_source_adapter_descriptors(
        manifest: PluginManifest,
        *,
        builtin: bool,
    ) -> list[AssetSourceAdapterDescriptor]:
        importer_kinds = {
            kind
            for capability in manifest.capabilities
            for kind in (
                ({"map", "world"} if capability.kind == "map-importer" else set())
                | ({"vehicle"} if capability.kind == "vehicle-importer" else set())
            )
        }
        descriptors: list[AssetSourceAdapterDescriptor] = []
        for capability in manifest.capabilities:
            if "asset_source_adapters" not in capability.metadata:
                continue
            values = capability.metadata["asset_source_adapters"]
            if not isinstance(values, list) or not values:
                raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_METADATA_INVALID")
            for value in values:
                try:
                    descriptor = AssetSourceAdapterDescriptor.model_validate(value)
                except (TypeError, ValidationError) as error:
                    raise PluginManagerError(
                        "PLUGIN_ASSET_SOURCE_ADAPTER_METADATA_INVALID"
                    ) from error
                if not set(descriptor.asset_kinds).issubset(importer_kinds):
                    raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_KIND_INVALID")
                if not builtin and descriptor.execution_boundary == "declarative-parser":
                    raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_BOUNDARY_INVALID")
                descriptors.append(descriptor)
        ids = [descriptor.adapter_id for descriptor in descriptors]
        if len(ids) != len(set(ids)):
            raise PluginManagerError("PLUGIN_ASSET_SOURCE_ADAPTER_DUPLICATE")
        return descriptors

    # 功能：
    #   解析模型唯一的活动归属，不在重复提供者之间任意挑选。
    # 输入：
    #   model_id：准备调用的模型标识。
    # 输出：
    #   binding：供应商、插件和能力标识的元组。
    def model_binding_for_model(self, model_id: str) -> tuple[str, str, str]:
        try:
            manifest, capability, model = self._active_model_bindings()[model_id]
        except KeyError:
            raise PluginManagerError("MODEL_PROVIDER_PLUGIN_UNAVAILABLE") from None
        binding = model["provider"], manifest.plugin_id, capability.capability_id
        return binding

    # 功能：
    #   按模型实际路由使用的同一活动绑定判断图像支持，不把缺省或字符串当作 true。
    # 输入：
    #   model_id：待查询的模型标识。
    # 输出：
    #   supported：是否明确支持图像输入。
    def model_supports_image_input(self, model_id: str) -> bool:
        try:
            _manifest, _capability, model = self._active_model_bindings()[model_id]
        except KeyError:
            raise PluginManagerError("MODEL_PROVIDER_PLUGIN_UNAVAILABLE") from None
        supported = model.get("supports_image_input") is True
        return supported

    # 功能：
    #   从唯一活动模型绑定中取供应商，避免单独维护过期供应商映射。
    # 输入：
    #   model_id：模型标识。
    # 输出：
    #   provider：实际绑定的供应商标识。
    def provider_for_model(self, model_id: str) -> str:
        provider, _plugin_id, _capability_id = self.model_binding_for_model(model_id)
        return provider

    # 功能：
    #   对必须保持活动的能力同时检查冻结快照和当前包、配置、启用状态，拒绝漂移。
    # 输入：
    #   snapshot：任务冻结快照。
    #   required_capability_ids：本次必须仍活动的能力标识集合。
    # 输出：
    #   None：不返回业务数据。
    def assert_snapshot_active(
        self, snapshot: PluginSnapshot, required_capability_ids: set[str]
    ) -> None:
        snapshot = verify_plugin_snapshot(snapshot)
        by_capability = {
            capability_id: entry
            for entry in snapshot.plugins
            for capability_id in entry.capability_ids
        }
        missing = required_capability_ids - set(by_capability)
        if missing:
            raise PluginManagerError(
                "PLUGIN_SNAPSHOT_CAPABILITY_MISSING:" + ",".join(sorted(missing))
            )
        for capability_id in required_capability_ids:
            entry = by_capability[capability_id]
            plugin = self.store.get_plugin(entry.plugin_id)
            manifest = self._manifest(plugin)
            if (
                not bool(plugin["enabled"])
                or plugin["status"] != "healthy"
                or plugin["version"] != entry.version
                or plugin["package_sha256"] != entry.package_sha256
                or self._manifest_hash(manifest) != entry.manifest_sha256
                or sha256_json(
                    self.store.get_plugin_configuration(entry.plugin_id)["configuration"]
                )
                != entry.configuration_sha256
            ):
                raise PluginManagerError(f"PLUGIN_SNAPSHOT_CAPABILITY_DRIFT:{capability_id}")

    # 功能：
    #   冻结活动清单、包和配置摘要及选择来源，验证必需插槽数量并按需保存到任务。
    # 输入：
    #   thread_id：可选的所属任务标识。
    # 输出：
    #   snapshot：本次任务可使用的完整插件快照。
    def snapshot(self, *, thread_id: str | None = None) -> PluginSnapshot:
        entries: list[PluginSnapshotEntry] = []
        active_slot_counts: dict[str, int] = {}
        for plugin in self.store.list_plugins():
            if not plugin["enabled"] or plugin["status"] != "healthy":
                continue
            # 升级后保留旧安装记录供历史查询，但已不随产品提供的内置代码不能进入新计划。
            if plugin["builtin"] and plugin["plugin_id"] not in self._definitions:
                continue
            manifest = self._manifest(plugin)
            slot_id = manifest.placement.slot_id
            if manifest.placement.activation_mode == "single":
                active_slot_counts[slot_id] = active_slot_counts.get(slot_id, 0) + 1
            configuration = self.store.get_plugin_configuration(manifest.plugin_id)["configuration"]
            entries.append(
                PluginSnapshotEntry(
                    plugin_id=manifest.plugin_id,
                    version=manifest.version,
                    package_sha256=str(plugin["package_sha256"]),
                    manifest_sha256=self._manifest_hash(manifest),
                    configuration_sha256=sha256_json(configuration),
                    configuration=configuration,
                    capability_ids=[item.capability_id for item in manifest.capabilities],
                    manifest=manifest,
                    bundle_root=(str(plugin["bundle_root"]) if plugin["bundle_root"] else None),
                    selection_source=str(plugin["selection_source"]),  # type: ignore[arg-type]
                    selected_by=str(plugin["selected_by"]),  # type: ignore[arg-type]
                    selection_receipt_sha256=(
                        str(plugin["selection_receipt_sha256"])
                        if plugin.get("selection_receipt_sha256")
                        else None
                    ),
                    harness_revision_sha256=(
                        str(plugin["harness_revision_sha256"])
                        if plugin.get("harness_revision_sha256")
                        else None
                    ),
                )
            )
        invalid_slots = sorted(
            slot_id
            for slot_id in self.required_single_slots
            if active_slot_counts.get(slot_id, 0) != 1
        )
        if invalid_slots:
            raise PluginManagerError("PLUGIN_REQUIRED_SLOT_CARDINALITY:" + ",".join(invalid_slots))
        entries.sort(key=lambda value: value.plugin_id)
        catalog_payload = [item.model_dump(mode="json") for item in entries]
        snapshot = PluginSnapshot(
            snapshot_id=f"plugin-snapshot-{uuid4().hex[:24]}",
            catalog_sha256=sha256_json(catalog_payload),
            plugins=entries,
            created_at=datetime.now(UTC),
        )
        if thread_id is not None:
            self.store.save_plugin_snapshot(thread_id, snapshot)
        return snapshot

    # 功能：
    #   将工具绑定到已核验的任务身份与最小宿主权限域，仅允许读、规划和仿真权限。
    # 输入：
    #   environment：本次任务的工具环境与宿主代理工厂。
    #   snapshot：任务冻结插件集合。
    # 输出：
    #   registry：绑定调用计量和生命周期检查的工具注册表。
    def build_tool_registry(
        self,
        *,
        environment: ToolEnvironment,
        snapshot: PluginSnapshot,
    ) -> ToolRegistry:
        snapshot = verify_plugin_snapshot(snapshot)
        registry = ToolRegistry(allowed_authorities={"read", "plan", "simulate"})
        for entry in snapshot.plugins:
            plugin = self.store.get_plugin(entry.plugin_id)
            if (
                str(plugin["version"]) != entry.version
                or str(plugin["package_sha256"]) != entry.package_sha256
            ):
                raise PluginManagerError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
            manifest = self._manifest(plugin)
            if self._manifest_hash(manifest) != entry.manifest_sha256:
                raise PluginManagerError(f"PLUGIN_MANIFEST_DRIFT:{entry.plugin_id}")
            configuration = self.store.get_plugin_configuration(entry.plugin_id)["configuration"]
            if sha256_json(configuration) != entry.configuration_sha256:
                raise PluginManagerError(f"PLUGIN_CONFIGURATION_DRIFT:{entry.plugin_id}")
            if bool(plugin["builtin"]):
                definition: PluginDefinition | None = self._definitions.get(entry.plugin_id)
                if definition is None or definition.tool_factory is None:
                    continue
                configured_environment = replace(
                    environment,
                    plugin_configuration=copy.deepcopy(configuration),
                    capability_broker=(
                        environment.broker_factory.scope(manifest)
                        if environment.broker_factory is not None
                        else None
                    ),
                )
                for tool in definition.tool_factory(configured_environment):
                    original_handler = tool.handler

                    # 功能：
                    #   为内置工具绑定当前循环的精确身份，记录结果并始终释放在途计数。
                    # 输入：
                    #   value：工具输入。
                    #   handler：当前工具实际处理函数。
                    #   plugin_id：冻结插件标识。
                    #   plugin_version：冻结插件版本。
                    #   capability_id：实际工具能力标识。
                    #   slot_id：运行插槽。
                    #   snapshot_entry：执行时复查所需的冻结条目。
                    # 输出：
                    #   output：处理函数实际输出。
                    def call_builtin(
                        value: Any,
                        *,
                        handler: Callable[[Any], Any] = original_handler,
                        plugin_id: str = entry.plugin_id,
                        plugin_version: str = entry.version,
                        capability_id: str = tool.tool_id,
                        slot_id: str = manifest.placement.slot_id,
                        snapshot_entry: PluginSnapshotEntry = entry,
                    ) -> Any:
                        started = time.perf_counter()
                        self._acquire_snapshot_call(snapshot_entry)
                        try:
                            output = handler(value)
                            self._record_usage(
                                plugin_id=plugin_id,
                                plugin_version=plugin_version,
                                capability_id=capability_id,
                                slot_id=slot_id,
                                invocation_kind="tool",
                                started=started,
                                input_value=value,
                                output_value=output,
                            )
                            return output
                        except BaseException as error:
                            self._record_usage(
                                plugin_id=plugin_id,
                                plugin_version=plugin_version,
                                capability_id=capability_id,
                                slot_id=slot_id,
                                invocation_kind="tool",
                                started=started,
                                input_value=value,
                                issue_code=type(error).__name__,
                            )
                            raise
                        finally:
                            self._release_call(plugin_id)

                    registry.register(
                        replace(
                            tool,
                            handler=call_builtin,
                            plugin_id=entry.plugin_id,
                            plugin_package_sha256=entry.package_sha256,
                            slot_id=manifest.placement.slot_id,
                        )
                    )
                continue
            if manifest.runtime.kind != "mcp-stdio":
                continue
            for capability in manifest.capabilities:
                if capability.kind not in PLUGIN_MCP_CAPABILITY_KINDS:
                    continue
                if "extension_hook" in capability.metadata:
                    continue
                tool_id = capability.capability_id

                # 功能：
                #   调用外部工具前核对安装字节和输入，调用后验证输出并记录用量，异常时按快照隔离。
                # 输入：
                #   value：工具输入对象。
                #   plugin_record：闭包绑定的安装记录。
                #   plugin_manifest：闭包绑定的完整清单。
                #   capability_id：实际调用的能力标识。
                #   snapshot_entry：任务冻结条目。
                #   plugin_capability：输入输出 Schema 与最小权限定义。
                # 输出：
                #   output：通过输出 Schema 检查的结果对象。
                def call_mcp(
                    value: dict[str, object],
                    *,
                    plugin_record: dict[str, object] = plugin,
                    plugin_manifest: PluginManifest = manifest,
                    capability_id: str = tool_id,
                    snapshot_entry: PluginSnapshotEntry = entry,
                    plugin_capability: PluginCapability = capability,
                ) -> dict[str, Any]:
                    current_plugin_id = str(plugin_record["plugin_id"])
                    started = time.perf_counter()
                    self._acquire_snapshot_call(snapshot_entry)
                    try:
                        plugin_root = Path(str(plugin_record["bundle_root"]))
                        self._verify_integrity(plugin_root, plugin_manifest)
                        client = self._mcp_client(
                            plugin_record,
                            plugin_manifest,
                            configuration=dict(snapshot_entry.configuration),
                            capability=plugin_capability,
                            broker_factory=environment.broker_factory,
                        )
                        jsonschema.validate(value, plugin_capability.input_schema)
                        output = client.call_tool(capability_id, value)
                        jsonschema.validate(output, plugin_capability.output_schema)
                        self._record_usage(
                            plugin_id=current_plugin_id,
                            plugin_version=snapshot_entry.version,
                            capability_id=capability_id,
                            slot_id=plugin_manifest.placement.slot_id,
                            invocation_kind="tool",
                            started=started,
                            input_value=value,
                            output_value=output,
                        )
                        return output
                    except (PluginProcessError, jsonschema.ValidationError) as error:
                        self._record_usage(
                            plugin_id=current_plugin_id,
                            plugin_version=snapshot_entry.version,
                            capability_id=capability_id,
                            slot_id=plugin_manifest.placement.slot_id,
                            invocation_kind="tool",
                            started=started,
                            input_value=value,
                            issue_code=type(error).__name__,
                        )
                        self._quarantine_failed_snapshot(snapshot_entry, error)
                        raise
                    finally:
                        self._release_call(current_plugin_id)

                registry.register(
                    ToolPlugin(
                        tool_id=tool_id,
                        version=manifest.version,
                        authority=capability.authority,  # type: ignore[arg-type]
                        input_type=None,
                        output_type=None,
                        input_schema=capability.input_schema,
                        output_schema=capability.output_schema,
                        handler=call_mcp,
                        plugin_id=entry.plugin_id,
                        plugin_package_sha256=entry.package_sha256,
                        routing_metadata=capability.metadata,
                        slot_id=manifest.placement.slot_id,
                    )
                )
        available_slots = {str(item["slot_id"]) for item in registry.catalog() if item["slot_id"]}
        missing = self.required_tool_slots - available_slots
        if missing:
            missing_list = ",".join(sorted(missing))
            raise PluginManagerError(f"REQUIRED_PLUGIN_SLOT_MISSING:{missing_list}")
        return registry

    # 功能：
    #   构建冻结身份的内置和隔离外部扩展，绑定配置、在途计数及显式管线输出规则。
    # 输入：
    #   snapshot：本次任务的插件快照。
    #   broker_factory：外部扩展可用的宿主代理工厂。
    # 输出：
    #   registry：按插槽、钩子和执行顺序组织的扩展注册表。
    def build_extension_registry(
        self,
        *,
        snapshot: PluginSnapshot,
        broker_factory: CoreCapabilityBroker | None = None,
    ) -> ExtensionRegistry:
        snapshot = verify_plugin_snapshot(snapshot)
        registry = ExtensionRegistry()
        for entry in snapshot.plugins:
            plugin = self.store.get_plugin(entry.plugin_id)
            if (
                str(plugin["version"]) != entry.version
                or str(plugin["package_sha256"]) != entry.package_sha256
            ):
                raise PluginManagerError(f"PLUGIN_SNAPSHOT_DRIFT:{entry.plugin_id}")
            manifest = self._manifest(plugin)
            if self._manifest_hash(manifest) != entry.manifest_sha256:
                raise PluginManagerError(f"PLUGIN_MANIFEST_DRIFT:{entry.plugin_id}")
            definition = self._definitions.get(entry.plugin_id)
            wrapped: dict[str, Callable[..., Any]] = {}
            capability: PluginCapability | None = None

            # 功能：
            #   为一个钩子捕获冻结配置与插件身份，避免循环闭包迟绑定到最后一个插件。
            # 输入：
            #   handler：实际钩子函数。
            #   plugin_id：插件标识。
            #   plugin_version：插件版本。
            #   capability_id：能力标识。
            #   slot_id：运行插槽。
            #   configuration：本次任务冻结配置。
            #   snapshot_entry：执行准入所需快照条目。
            # 输出：
            #   call_hook：注入独立配置并负责计量的钩子包装函数。
            def wrap_hook(
                handler: Callable[..., Any],
                plugin_id: str,
                plugin_version: str,
                capability_id: str,
                slot_id: str,
                configuration: dict[str, object],
                snapshot_entry: PluginSnapshotEntry,
            ) -> Callable[..., Any]:
                parameters = inspect.signature(handler).parameters
                accepts_configuration = "configuration" in parameters or any(
                    item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
                )

                # 功能：
                #   每次调用注入新的冻结配置副本，计量结果，并在成功或失败后归还在途计数。
                # 输入：
                #   kwargs：钩子业务参数，其中 configuration 由任务冻结值接管。
                # 输出：
                #   output：实际钩子输出。
                def call_hook(**kwargs: Any) -> Any:
                    started = time.perf_counter()
                    self._acquire_snapshot_call(snapshot_entry)
                    try:
                        if accepts_configuration:
                            # 配置属于已准备任务，既不接受本次覆盖，也不继承上次调用的原位修改。
                            kwargs["configuration"] = copy.deepcopy(configuration)
                        output = handler(**kwargs)
                        self._record_usage(
                            plugin_id=plugin_id,
                            plugin_version=plugin_version,
                            capability_id=capability_id,
                            slot_id=slot_id,
                            invocation_kind="hook",
                            started=started,
                            input_value=kwargs,
                            output_value=output,
                        )
                        return output
                    except BaseException as error:
                        self._record_usage(
                            plugin_id=plugin_id,
                            plugin_version=plugin_version,
                            capability_id=capability_id,
                            slot_id=slot_id,
                            invocation_kind="hook",
                            started=started,
                            input_value=kwargs,
                            issue_code=type(error).__name__,
                        )
                        raise
                    finally:
                        self._release_call(plugin_id)

                return call_hook

            if definition is not None and definition.hooks:
                capability = manifest.capabilities[0]
                for hook_name, handler in definition.hooks.items():
                    wrapped[hook_name] = wrap_hook(
                        handler,
                        entry.plugin_id,
                        entry.version,
                        capability.capability_id,
                        manifest.placement.slot_id,
                        dict(entry.configuration),
                        entry,
                    )
            elif manifest.runtime.kind == "mcp-stdio":
                extension_capabilities = [
                    item
                    for item in manifest.capabilities
                    if item.metadata.get("extension_hook") in PLUGIN_EXTENSION_HOOKS
                ]
                if not extension_capabilities:
                    continue
                placement = manifest.placement
                for extension_capability in extension_capabilities:
                    hook_name = str(extension_capability.metadata["extension_hook"])

                    # 功能：
                    #   调用冻结外部钩子并验证输入输出，管线模式只传递显式 value，错误归属原快照。
                    # 输入：
                    #   _plugin：闭包绑定的安装记录。
                    #   _manifest：闭包绑定的清单。
                    #   _capability：当前钩子的能力契约。
                    #   _entry：本次任务冻结条目。
                    #   kwargs：钩子业务输入。
                    # 输出：
                    #   result：普通模式的输出对象或管线模式的 value 值。
                    def call_mcp_extension(
                        *,
                        _plugin: dict[str, object] = plugin,
                        _manifest: PluginManifest = manifest,
                        _capability: PluginCapability = extension_capability,
                        _entry: PluginSnapshotEntry = entry,
                        **kwargs: Any,
                    ) -> Any:
                        current_plugin_id = _entry.plugin_id
                        started = time.perf_counter()
                        arguments: Any = kwargs
                        self._acquire_snapshot_call(_entry)
                        try:
                            plugin_root = Path(str(_plugin["bundle_root"]))
                            self._verify_integrity(plugin_root, _manifest)
                            client = self._mcp_client(
                                _plugin,
                                _manifest,
                                configuration=dict(_entry.configuration),
                                capability=_capability,
                                broker_factory=broker_factory,
                            )
                            arguments = _mcp_json_value(kwargs)
                            jsonschema.validate(arguments, _capability.input_schema)
                            output = client.call_tool(_capability.capability_id, arguments)
                            jsonschema.validate(output, _capability.output_schema)
                            if (
                                _manifest.placement.activation_mode == "pipeline"
                                and "value" not in output
                            ):
                                raise PluginProcessError("PLUGIN_EXTENSION_PIPELINE_VALUE_MISSING")
                            self._record_usage(
                                plugin_id=current_plugin_id,
                                plugin_version=_entry.version,
                                capability_id=_capability.capability_id,
                                slot_id=_manifest.placement.slot_id,
                                invocation_kind="hook",
                                started=started,
                                input_value=arguments,
                                output_value=output,
                            )
                            if _manifest.placement.activation_mode == "pipeline":
                                result = output["value"]
                            else:
                                result = output
                            return result
                        except (PluginProcessError, jsonschema.ValidationError) as error:
                            self._record_usage(
                                plugin_id=current_plugin_id,
                                plugin_version=_entry.version,
                                capability_id=_capability.capability_id,
                                slot_id=_manifest.placement.slot_id,
                                invocation_kind="hook",
                                started=started,
                                input_value=arguments,
                                issue_code=type(error).__name__,
                            )
                            self._quarantine_failed_snapshot(_entry, error)
                            raise
                        finally:
                            self._release_call(current_plugin_id)

                    registry.register(
                        ExtensionPlugin(
                            plugin_id=entry.plugin_id,
                            version=entry.version,
                            package_sha256=entry.package_sha256,
                            capability_id=extension_capability.capability_id,
                            slot_id=placement.slot_id,
                            activation_mode=placement.activation_mode,
                            failure_mode=placement.failure_mode,
                            swap_policy=placement.swap_policy,
                            pipeline_order=placement.pipeline_order,
                            runs_after=tuple(placement.runs_after),
                            runs_before=tuple(placement.runs_before),
                            hooks={hook_name: call_mcp_extension},
                        )
                    )
                continue
            else:
                continue
            if capability is None:
                continue
            placement = manifest.placement
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
                    hooks=wrapped,
                )
            )
        return registry

    # 功能：
    #   序列化展示目录，不将展示数据当作可执行任务快照。
    # 输入：
    #   无。
    # 输出：
    #   rendered：规范 JSON 目录文本。
    def catalog_json(self) -> str:
        rendered = canonical_json(self.list_plugins())
        return rendered
