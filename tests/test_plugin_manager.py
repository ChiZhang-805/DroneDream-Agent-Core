from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dronedream_agent_app.plugin_manager import PluginManager, PluginManagerError
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.asset_packages import (
    AssetFile,
    AssetIR,
    AssetSource,
    DDPkgManifest,
    package_content_sha256,
)
from dronedream_agent_core.contracts import MapAsset, RouteQuery
from dronedream_agent_core.extensions import ExtensionExecutionError
from dronedream_agent_core.plugin_api import (
    ToolEnvironment,
    build_discovered_extension_registry,
    build_snapshot_tool_registry,
)
from dronedream_agent_core.plugin_contracts import PluginGovernancePolicy, PluginManifest


# 功能：
#   模拟升级后遗留内置插件记录，确保新快照不引用已退役实现且历史安装数据仍保留。
# 输入：
#   tmp_path：隔离数据库目录。
# 输出：
#   None：断言新快照可被执行端重建，未删除旧记录。
def test_upgrade_omits_retired_builtin_from_new_snapshot(tmp_path):
    store = AppStore(tmp_path)
    manager = PluginManager(store)
    plugin = next(item for item in manager._definitions.values()
                  if item.manifest.plugin_id == "input.channel-text")
    original = plugin.manifest
    old = original.model_copy(update={"plugin_id": "legacy.retired-importer"}, deep=True)
    old.capabilities[0].capability_id = "legacy.retired-importer.read"
    old.placement.slot_id = "legacy.retired-importer"
    store.upsert_plugin(manifest=old, package_sha256=manager._manifest_hash(old),
                        bundle_root=None, builtin=True, enabled=True, status="healthy",
                        health="healthy", trust_status="verified",
                        trust_decision={"status": "verified", "source": "builtin"})
    snapshot = manager.snapshot()
    assert "legacy.retired-importer" not in {entry.plugin_id for entry in snapshot.plugins}
    assert store.get_plugin("legacy.retired-importer")["enabled"]
    build_discovered_extension_registry(snapshot)


# 功能：
#   跟踪本文件每例创建的管理器和存储，测试结束先关闭所有会话，再关闭数据库。
# 输入：
#   monkeypatch：在本例期间包装构造函数，保留原类的静态方法和行为。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def _release_managers(monkeypatch):
    original_init = PluginManager.__init__
    stores: set[int] = set()
    with ExitStack() as store_cleanup, ExitStack() as manager_cleanup:
        # 功能：
        #   记录真实构造出来的资源，同一数据库只登记一次，兼容一次测试模拟多次应用启动。
        # 输入：
        #   self：正在初始化的管理器。
        #   store：本测试拥有的独立存储。
        #   options：传给原构造函数的安装和隔离选项。
        # 输出：
        #   None：不返回业务数据。
        def initialize(self, store, **options):
            if id(store) not in stores:
                stores.add(id(store))
                store_cleanup.callback(store.close)
            original_init(self, store, **options)
            manager_cleanup.callback(self.close)

        monkeypatch.setattr(PluginManager, "__init__", initialize)
        yield


# 功能：
#   构造不具备飞行控制权限的声明式面板清单，用真实摘要绑定固定测试负载。
# 输入：
#   version：本例使用的插件版本。
# 输出：
#   manifest：尚未实例化的清单字典。
def _manifest(*, version: str = "1.0.0") -> dict[str, object]:
    panel = b'{"title":"Route audit"}\n'
    manifest = {
        "schema_version": "dronedream.plugin-manifest.v1",
        "plugin_id": "example.route-audit",
        "name": "Route Audit",
        "version": version,
        "description": "Inspect a prepared route without actuator access.",
        "publisher": "Example",
        "api_version": "1.0",
        "minimum_app_version": "0.1.0",
        "runtime": {
            "kind": "ui-declarative",
            "protocol_version": "dronedream.plugin.v1",
            "startup_timeout_seconds": 15,
            "call_timeout_seconds": 60,
        },
        "capabilities": [
            {
                "capability_id": "example.route-audit.panel",
                "kind": "ui-panel",
                "name": "Route Audit",
                "description": "Show route evidence.",
                "authority": "read",
                "input_schema": {},
                "output_schema": {},
                "metadata": {"entrypoint": "ui/panel.json"},
            }
        ],
        "permissions": ["mission.read", "ui.panel"],
        "dependencies": [],
        "file_sha256": {"ui/panel.json": hashlib.sha256(panel).hexdigest()},
        "default_enabled": False,
        "removable": True,
        "configuration_schema": {},
    }
    return manifest


# 功能：
#   将面板清单与负载写成供导入测试使用的 ZIP，不生成或运行可执行代码。
# 输入：
#   path：隔离目录中的归档路径。
#   version：测试包版本。
# 输出：
#   path：已写入测试归档的路径。
def _bundle(path: Path, *, version: str = "1.0.0") -> Path:
    panel = b'{"title":"Route audit"}\n'
    manifest = _manifest(version=version)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr("ui/panel.json", panel)
    return path


# 功能：
#   创建带归档摘要的安装器索引，模拟应用升级交付的新插件包。
# 输入：
#   root：本测试拥有的官方包夹具目录。
#   version：本次提供的包版本。
# 输出：
#   root：包含索引和包的夹具目录。
def _official_index(root: Path, *, version: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = _bundle(root / f"route-audit-{version}.zip", version=version)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.official-plugin-index.v1",
                "plugins": [
                    {
                        "plugin_id": "example.route-audit",
                        "version": version,
                        "file": archive.name,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


# 功能：
#   创建两节点路线求解夹具；边的资格标记和摘要仅供单元测试，不是真实飞行证明。
# 输入：
#   无。
# 输出：
#   graph：经契约校验的最小地图。
def _graph() -> MapAsset:
    graph = MapAsset.model_validate(
        {
            "asset_id": "test.map",
            "name": "Test",
            "nodes": [
                {
                    "node_id": "start",
                    "label": "Start",
                    "position_m": {"x": 0, "y": 0, "z": 1},
                    "semantic": "launch",
                },
                {
                    "node_id": "goal",
                    "label": "Goal",
                    "position_m": {"x": 2, "y": 0, "z": 1},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "start-goal",
                    "from_node": "start",
                    "to_node": "goal",
                    "distance_m": 2,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                    "qualification": "flight-verified",
                    "evidence_sha256": "a" * 64,
                }
            ],
            "named_entities": {"start": "start", "goal": "goal"},
        }
    )
    return graph


# 功能：
#   定义提示词、输出校验和工具三种外部能力，使用惰性文件供假客户端检验冻结身份与权限。
# 输入：
#   root：夹具文件目录。
# 输出：
#   manifest：带三种能力及各自最小权限声明的测试清单。
def _external_harness_manifest(root: Path) -> PluginManifest:
    executable = b"snapshot-bound MCP executable"
    (root / "plugin.exe").write_bytes(executable)
    manifest = PluginManifest.model_validate(
        {
            "plugin_id": "example.external-harness",
            "name": "External Harness",
            "version": "1.0.0",
            "description": "Extends prompt preparation and provides a task tool.",
            "publisher": "Example",
            "runtime": {
                "kind": "mcp-stdio",
                "command": ["plugin.exe"],
                "protocol_version": "dronedream.plugin.v1",
                "startup_timeout_seconds": 15,
                "call_timeout_seconds": 60,
            },
            "capabilities": [
                {
                    "capability_id": "example.external-harness.prompt",
                    "kind": "prompt-pack",
                    "name": "External prompt",
                    "description": "Adds a tested external prompt fragment.",
                    "authority": "plan",
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "required": ["value"],
                        "properties": {"value": {"type": "object"}},
                    },
                    "required_permissions": ["mission.read"],
                    "metadata": {"extension_hook": "augment_prompt"},
                },
                {
                    "capability_id": "example.external-harness.echo",
                    "kind": "tool",
                    "name": "External echo",
                    "description": "Returns structured task data through MCP.",
                    "authority": "plan",
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "required": ["echo", "configuration"],
                        "properties": {
                            "echo": {"type": "object"},
                            "configuration": {"type": "object"},
                        },
                    },
                    "required_permissions": [],
                    "metadata": {},
                },
                {
                    "capability_id": "example.external-harness.output-guard",
                    "kind": "plan-validator",
                    "name": "External output guard",
                    "description": "Validates a structured output with its own permission scope.",
                    "authority": "plan",
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "required": ["value"],
                        "properties": {"value": {"type": "object"}},
                    },
                    "required_permissions": ["mission.read", "evidence.write"],
                    "metadata": {"extension_hook": "validate_output"},
                },
            ],
            "permissions": ["process.spawn", "mission.read", "evidence.write"],
            "file_sha256": {"plugin.exe": hashlib.sha256(executable).hexdigest()},
            "placement": {
                "category_id": "models",
                "category_label": "模型与提示词",
                "slot_id": "models.prompt-packs",
                "slot_label": "提示词包",
                "activation_mode": "pipeline",
                "scope": "mission",
                "failure_mode": "fail-closed",
                "swap_policy": "next-mission",
            },
            "configuration_schema": {"type": "object"},
        }
    )
    return manifest


# 功能：
#   声明只通过隔离插件与宿主受限文件代理转换地图的导入器，负载不会作为真实程序执行。
# 输入：
#   root：夹具目录。
# 输出：
#   manifest：附带 CityJSON 格式描述和暂存写权限的导入器清单。
def _external_map_importer_manifest(root: Path) -> PluginManifest:
    executable = b"asset converter"
    (root / "plugin.exe").write_bytes(executable)
    manifest = PluginManifest.model_validate(
        {
            "plugin_id": "example.external-map-importer",
            "name": "External map importer",
            "version": "1.0.0",
            "description": "Converts an external map into a canonical staged bundle.",
            "publisher": "Example",
            "runtime": {"kind": "mcp-stdio", "command": ["plugin.exe"]},
            "capabilities": [
                {
                    "capability_id": "example.external-map-importer.convert",
                    "kind": "map-importer",
                    "name": "Map converter",
                    "description": "Writes a hash-declared canonical map bundle.",
                    "authority": "plan",
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "required": ["output_sha256"],
                        "properties": {"output_sha256": {"type": "string", "minLength": 64}},
                    },
                    "metadata": {
                        "extension_hook": "import_asset",
                        "asset_source_adapters": [
                            {
                                "adapter_id": "example.cityjson",
                                "name": "Example CityJSON",
                                "version": "1.0.0",
                                "availability": "plugin_required",
                                "source_formats": ["cityjson"],
                                "file_extensions": [".cityjson"],
                                "asset_kinds": ["map", "world"],
                                "output_format": "ddpkg",
                                "execution_boundary": "isolated-plugin",
                            }
                        ],
                    },
                }
            ],
            "permissions": ["process.spawn", "asset.read", "asset.write-staging"],
            "file_sha256": {"plugin.exe": hashlib.sha256(executable).hexdigest()},
            "placement": {
                "category_id": "assets",
                "category_label": "地图与无人机",
                "slot_id": "assets.map-importer",
                "slot_label": "地图导入器",
                "activation_mode": "single",
                "scope": "general",
                "failure_mode": "fail-closed",
                "swap_policy": "next-mission",
            },
        }
    )
    return manifest


# 功能：
#   创建格式转换回执夹具，将源摘要、AssetIR 和文件索引绑定到同一份规范资产包。
# 输入：
#   path：测试规范包的输出路径。
#   source_sha256：被转换源文件的摘要。
# 输出：
#   payload：已封装的规范包字节。
def _source_adapter_package(path: Path, source_sha256: str) -> bytes:
    source_payload = b'{"type":"CityJSON","version":"2.0"}'
    source_file = AssetFile(
        path="normalized/source.city.json",
        role="source",
        media_type="application/json",
        sha256=hashlib.sha256(source_payload).hexdigest(),
        size_bytes=len(source_payload),
    )
    asset_ir = AssetIR(
        asset_id="example-city",
        name="Example City",
        version="0.1.0",
        kind="map",
        source=AssetSource(
            adapter_id="example.cityjson",
            adapter_version="1.0.0",
            application="Example CityJSON Adapter",
            source_format="cityjson",
            source_sha256=source_sha256,
        ),
        coordinate_frame="map_enu",
        files=[source_file],
        license_expression="NOASSERTION",
    )
    ir_payload = (asset_ir.model_dump_json() + "\n").encode()
    ir_file = AssetFile(
        path="normalized/asset-ir.json",
        role="asset_ir",
        media_type="application/json",
        sha256=hashlib.sha256(ir_payload).hexdigest(),
        size_bytes=len(ir_payload),
    )
    manifest_files = [source_file, ir_file]
    manifest = DDPkgManifest(
        package_id="example-city.0.1.0",
        asset_id="example-city",
        asset_kind="map",
        content_sha256=package_content_sha256(manifest_files),
        files=manifest_files,
        created_at=datetime.now(UTC),
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(source_file.path, source_payload)
        archive.writestr(ir_file.path, ir_payload)
        archive.writestr("manifest.json", manifest.model_dump_json())
    payload = path.read_bytes()
    return payload


class _FakeMcpClient:
    instances: list[_FakeMcpClient] = []
    source_adapter_payload: bytes | None = None

    # 功能：
    #   捕获宿主传入的配置、权限和文件服务以供断言，不建立进程或网络连接。
    # 输入：
    #   configuration：本次调用应绑定的配置。
    #   _kwargs：宿主构造客户端时传入的权限、服务及其他参数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, configuration=None, **_kwargs):
        self.configuration = configuration or {}
        self.permissions = list(_kwargs.get("permissions", []))
        self.host_services = _kwargs.get("host_services")
        self.instances.append(self)

    # 功能：
    #   将当前假客户端作为上下文资源，不创建额外实例。
    # 输入：
    #   无。
    # 输出：
    #   self：当前假客户端。
    def __enter__(self):
        return self

    # 功能：
    #   结束无外部资源的假客户端上下文，不抑制测试中的异常。
    # 输入：
    #   _type：异常类型或 None。
    #   _value：异常对象或 None。
    #   _traceback：异常堆栈或 None。
    # 输出：
    #   None：不返回业务数据。
    def __exit__(self, _type, _value, _traceback):
        return None

    # 功能：
    #   模拟协议工具返回；资产转换走真实宿主读写代理，其余能力返回可核对的合成结构。
    # 输入：
    #   name：能力标识。
    #   arguments：宿主已准备的结构化输入。
    # 输出：
    #   result：对应能力的结构化假结果。
    def call_tool(self, name, arguments):
        if name == "example.external-map-importer.convert":
            assert self.host_services is not None
            read_result = self.host_services(
                "dronedream/filesystem/read",
                {"root": arguments["input_root"], "path": arguments["input_path"]},
            )
            assert base64.b64decode(read_result["body_base64"]) == b"source"
            rendered = self.source_adapter_payload or b"canonical map bundle"
            self.host_services(
                "dronedream/filesystem/write",
                {
                    "root": arguments["output_root"],
                    "path": arguments["output_path"],
                    "body_base64": base64.b64encode(rendered).decode("ascii"),
                },
            )
            result = {"output_sha256": hashlib.sha256(rendered).hexdigest()}
            return result
        if name == "example.external-harness.prompt":
            result = {
                "value": {
                    **arguments["value"],
                    "external_prompt": self.configuration.get("tone"),
                }
            }
            return result
        if name == "example.external-harness.output-guard":
            result = {"value": {**arguments["value"], "external_guard": True}}
            return result
        result = {"echo": arguments, "configuration": self.configuration}
        return result


class _InvalidOutputMcpClient(_FakeMcpClient):
    # 功能：
    #   仅对提示词钩子返回违反输出 Schema 的结构，以验证失败隔离而非正常拼接。
    # 输入：
    #   name：能力标识。
    #   arguments：结构化调用输入。
    # 输出：
    #   result：故意非法或沿用正常假客户端的返回对象。
    def call_tool(self, name, arguments):
        if name == "example.external-harness.prompt":
            result = {"unexpected": arguments}
            return result
        result = super().call_tool(name, arguments)
        return result


# 功能：
#   验证内置插件发现结果实际生成模型与资产适配器目录，并携带模型视觉能力和包摘要。
# 输入：
#   tmp_path：独立应用存储目录。
# 输出：
#   None：不返回业务数据。
def test_builtin_plugins_are_discovered_and_model_catalog_is_live(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    plugins = manager.list_plugins()

    assert len(plugins) >= 8
    assert {item["id"] for item in manager.model_catalog()} == {
        "gpt-4.1",
        "gpt-5.1",
        "gpt-5.4",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "kimi-k2.6",
        "kimi-k3",
    }
    catalog = {str(item["id"]): item for item in manager.model_catalog()}
    assert catalog["gpt-5.4"]["supports_image_input"] is True
    assert catalog["deepseek-v4-flash"]["supports_image_input"] is False
    assert catalog["kimi-k2.6"]["supports_image_input"] is False
    assert all(item["package_sha256"] for item in plugins)
    adapters = {item["adapter_id"]: item for item in manager.source_adapter_catalog()}
    assert adapters["gazebo.sdf"]["enabled"] is True
    assert adapters["solidworks.urdf"]["enabled"] is False


# 功能：
#   验证路线工具来自冻结插件快照，调用回执绑定实际包身份且成功调用写入用量。
# 输入：
#   tmp_path：独立应用存储与语义文件目录。
# 输出：
#   None：不返回业务数据。
def test_tool_registry_is_built_from_plugin_snapshot_and_receipts_bind_package(tmp_path):
    store = AppStore(tmp_path)
    manager = PluginManager(store)
    semantic = tmp_path / "semantic.json"
    semantic.write_text("{}", encoding="utf-8")
    snapshot = manager.snapshot()
    registry = manager.build_tool_registry(
        environment=ToolEnvironment(
            map_graph=_graph(),
            semantic_path=semantic,
            vehicle_diameter_m=0.6,
            vehicle_height_m=0.4,
            waypoint_hold_seconds=0.4,
        ),
        snapshot=snapshot,
    )

    route, receipt = registry.call(
        "navigation.shortest-route", RouteQuery(start_node="start", goal_node="goal")
    )

    assert route.node_ids == ["start", "goal"]
    assert receipt.plugin_id == "navigation.shortest-route"
    assert receipt.plugin_package_sha256
    assert receipt.plugin_package_sha256 == next(
        item.package_sha256
        for item in snapshot.plugins
        if item.plugin_id == "navigation.shortest-route"
    )
    usage = store.summarize_plugin_usage("navigation.shortest-route")
    assert usage["calls"] == 1
    assert usage["successes"] == 1
    assert usage["input_bytes"] > 0
    assert usage["output_bytes"] > 0


# 功能：
#   验证受管发布者名单与禁止本地批准的策略分别生效，拒绝结果保留治理记录。
# 输入：
#   tmp_path：独立策略存储和归档目录。
# 输出：
#   None：不返回业务数据。
def test_enterprise_governance_rejects_unapproved_publisher_and_local_trust(tmp_path):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    policy = PluginGovernancePolicy(
        policy_id="managed-flight-lab",
        mode="managed",
        allowed_publishers=["DroneDream"],
        require_verified_signatures=True,
        allow_local_approval=False,
    )
    manager.set_governance_policy(policy)

    with pytest.raises(PluginManagerError, match="GOVERNANCE_PUBLISHER_NOT_ALLOWED"):
        manager.import_bundle(_bundle(tmp_path / "route-audit.zip"))

    relaxed = policy.model_copy(update={"allowed_publishers": ["Example"]})
    manager.set_governance_policy(relaxed)
    imported = manager.import_bundle(_bundle(tmp_path / "route-audit-allowed.zip"))
    assert imported["plugin_id"] == "example.route-audit"
    with pytest.raises(PluginManagerError, match="GOVERNANCE_LOCAL_APPROVAL_DISABLED"):
        manager.approve_local_package("example.route-audit")
    decisions = store.list_plugin_governance_decisions("example.route-audit")
    assert any(not bool(item["accepted"]) for item in decisions)


# 功能：
#   验证必需的单选插槽可以替换实现，但不能单独禁用其最后一个活动成员。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_single_slot_switches_implementation_without_leaving_required_slot_empty(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    switched = manager.enable("navigation.clearance-first-route")

    assert switched["replaced_plugins"] == ["navigation.shortest-route"]
    by_id = {item["plugin_id"]: item for item in manager.list_plugins()}
    assert by_id["navigation.clearance-first-route"]["enabled"] is True
    assert by_id["navigation.shortest-route"]["enabled"] is False
    with pytest.raises(PluginManagerError, match="SLOT_REQUIRES_ONE_ENABLED"):
        manager.disable("navigation.clearance-first-route")


# 功能：
#   验证默认快照为路线和工作流单选插槽各保留一个实现，不同时启用互斥策略。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_required_single_slots_have_exactly_one_active_implementation(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    snapshot = manager.snapshot()
    active_ids = {entry.plugin_id for entry in snapshot.plugins}

    assert "navigation.shortest-route" in active_ids
    assert "navigation.clearance-first-route" not in active_ids
    assert "workflow.balanced" in active_ids
    assert "workflow.deliberate" not in active_ids


# 功能：
#   验证多选任务顾问插槽允许独立能力并存，启用后者不替换前者。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_multiple_slot_keeps_independent_plugins_enabled(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    first = manager.enable("general.mission-risk-profile")
    second = manager.enable("general.constraint-coverage")

    assert first["replaced_plugins"] == []
    assert second["replaced_plugins"] == []
    enabled = {
        item["plugin_id"]
        for item in manager.list_plugins()
        if item["enabled"] and item["placement"]["slot_id"] == "general.mission-advisors"
    }
    assert enabled == {"general.mission-risk-profile", "general.constraint-coverage"}


# 功能：
#   验证配置变化会改变新快照中的配置摘要，已经创建的快照仍保留原配置。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_plugin_configuration_is_bound_into_task_snapshot(tmp_path):
    store = AppStore(tmp_path)
    manager = PluginManager(store)
    first = manager.snapshot()

    store.save_plugin_configuration("model.openai", {"region": "us"})
    second = manager.snapshot()

    first_entry = next(item for item in first.plugins if item.plugin_id == "model.openai")
    second_entry = next(item for item in second.plugins if item.plugin_id == "model.openai")
    assert first_entry.configuration_sha256 != second_entry.configuration_sha256
    assert first_entry.configuration == {}
    assert second_entry.configuration == {"region": "us"}


# 功能：
#   验证正常电池阈值可持久化，但内联密钥不能混入普通插件配置。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_plugin_configuration_rejects_inline_secrets_and_allows_safe_runtime_settings(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    saved = manager.configure(
        "runtime.anomaly-battery",
        {"minimum_battery_percent": 25},
    )

    assert saved["configuration"] == {"minimum_battery_percent": 25}
    with pytest.raises(PluginManagerError, match="INLINE_SECRET_FORBIDDEN"):
        manager.configure("runtime.anomaly-battery", {"api_key": "must-not-be-stored"})


# 功能：
#   1. 验证外部钩子和工具由冻结快照恢复，绑定原配置与各能力最小权限。
#   2. 验证下次任务生效的普通停用不改写已有快照；假客户端不执行外部进程。
# 输入：
#   tmp_path：独立存储、插件及语义文件目录。
#   monkeypatch：替换实际协议客户端。
# 输出：
#   None：不返回业务数据。
def test_external_mcp_hooks_and_tools_rebuild_from_frozen_snapshot(tmp_path, monkeypatch):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    bundle_root = tmp_path / "external"
    bundle_root.mkdir()
    manifest = _external_harness_manifest(bundle_root)
    store.upsert_plugin(
        manifest=manifest,
        package_sha256="c" * 64,
        bundle_root=bundle_root,
        builtin=False,
        enabled=True,
        status="healthy",
        health="healthy",
    )
    store.save_plugin_configuration(manifest.plugin_id, {"tone": "concise"})
    snapshot = manager.snapshot()

    _FakeMcpClient.instances.clear()
    monkeypatch.setattr("dronedream_agent_app.plugin_manager.McpStdioClient", _FakeMcpClient)
    manager_extensions = manager.build_extension_registry(snapshot=snapshot)
    prompt, receipts = manager_extensions.invoke_pipeline(
        "models.prompt-packs", "augment_prompt", {"base": True}
    )
    assert prompt == {"base": True, "external_prompt": "concise"}
    assert receipts[-1].plugin_id == manifest.plugin_id
    guarded, guard_receipts = manager_extensions.invoke_pipeline(
        "models.prompt-packs", "validate_output", {"structured": True}
    )
    assert guarded == {"structured": True, "external_guard": True}
    assert guard_receipts[-1].capability_id == "example.external-harness.output-guard"

    manager.disable(manifest.plugin_id)
    prompt_after_disable, _ = manager_extensions.invoke_pipeline(
        "models.prompt-packs", "augment_prompt", {"base": True}
    )
    assert prompt_after_disable["external_prompt"] == "concise"

    monkeypatch.setattr("dronedream_agent_core.plugin_api.McpStdioClient", _FakeMcpClient)
    restored_extensions = build_discovered_extension_registry(snapshot)
    restored_prompt, _ = restored_extensions.invoke_pipeline(
        "models.prompt-packs", "augment_prompt", {"base": True}
    )
    assert restored_prompt["external_prompt"] == "concise"

    semantic = tmp_path / "semantic.json"
    semantic.write_text("{}", encoding="utf-8")
    registry = build_snapshot_tool_registry(
        ToolEnvironment(
            map_graph=_graph(),
            semantic_path=semantic,
            vehicle_diameter_m=0.6,
            vehicle_height_m=0.4,
            waypoint_hold_seconds=0.4,
        ),
        snapshot,
    )
    output, receipt = registry.call("example.external-harness.echo", {"mission": "inspect"})
    assert output == {
        "echo": {"mission": "inspect"},
        "configuration": {"tone": "concise"},
    }
    assert receipt.plugin_id == manifest.plugin_id
    assert {tuple(client.permissions) for client in _FakeMcpClient.instances} == {
        (),
        ("mission.read",),
        ("mission.read", "evidence.write"),
    }


# 功能：
#   验证外部钩子输出不符合声明时中止调用并隔离插件，不继续使用非法输出。
# 输入：
#   tmp_path：独立插件存储和负载目录。
#   monkeypatch：注入返回非法结构的假客户端。
# 输出：
#   None：不返回业务数据。
def test_external_mcp_hook_invalid_output_is_fail_closed_and_quarantined(tmp_path, monkeypatch):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    bundle_root = tmp_path / "external"
    bundle_root.mkdir()
    manifest = _external_harness_manifest(bundle_root)
    store.upsert_plugin(
        manifest=manifest,
        package_sha256="e" * 64,
        bundle_root=bundle_root,
        builtin=False,
        enabled=True,
        status="healthy",
        health="healthy",
    )
    snapshot = manager.snapshot()
    monkeypatch.setattr(
        "dronedream_agent_app.plugin_manager.McpStdioClient",
        _InvalidOutputMcpClient,
    )

    extensions = manager.build_extension_registry(snapshot=snapshot)
    with pytest.raises(ExtensionExecutionError, match="PLUGIN_HOOK_FAILED"):
        extensions.invoke_pipeline("models.prompt-packs", "augment_prompt", {"base": True})

    quarantined = store.get_plugin(manifest.plugin_id)
    assert quarantined["enabled"] is False
    assert quarantined["status"] == "quarantined"
    assert quarantined["health"] == "failed"


# 功能：
#   验证任务仍持有插件时卸载被拒绝，已启用状态和冻结任务依赖的目录都保留。
# 输入：
#   tmp_path：独立安装目录。
# 输出：
#   None：不返回业务数据。
def test_uninstall_keeps_active_task_bundle_and_catalog_unchanged(tmp_path):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    manager.import_bundle(_bundle(tmp_path / "plugin.zip"))
    manager.approve_local_package("example.route-audit")
    manager.enable("example.route-audit")
    manager.set_disable_guard(
        lambda _plugin_id, policy: (_ for _ in ()).throw(RuntimeError(policy))
    )

    with pytest.raises(PluginManagerError, match="UNINSTALL_REQUIRES_IDLE:restart"):
        manager.uninstall("example.route-audit")

    plugin = store.get_plugin("example.route-audit")
    assert plugin["enabled"] is True
    assert plugin["status"] == "healthy"
    assert (store.plugins_root / "example.route-audit").is_dir()


# 功能：
#   验证启用的格式适配器参与探测，经过宿主受限读写生成绑定源摘要的规范资产包。
# 输入：
#   tmp_path：独立源文件、插件和转换输出目录。
#   monkeypatch：将外部执行替换为调用真实宿主代理的假客户端。
# 输出：
#   None：不返回业务数据。
def test_enabled_source_adapter_automatically_produces_provenance_bound_ddpkg(
    tmp_path, monkeypatch
):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    root = tmp_path / "map-importer"
    root.mkdir()
    manifest = _external_map_importer_manifest(root)
    store.upsert_plugin(
        manifest=manifest,
        package_sha256="d" * 64,
        bundle_root=root,
        builtin=False,
        enabled=True,
        status="healthy",
        health="healthy",
    )
    store.set_plugin_lifecycles(
        {
            manifest.plugin_id: {
                "enabled": True,
                "status": "healthy",
                "health": "healthy",
            },
        }
    )
    source = tmp_path / "source.cityjson"
    source.write_bytes(b"source")
    source_sha256 = hashlib.sha256(b"source").hexdigest()
    package_path = tmp_path / "adapter.ddpkg"
    monkeypatch.setattr(
        _FakeMcpClient,
        "source_adapter_payload",
        _source_adapter_package(package_path, source_sha256),
    )
    monkeypatch.setattr("dronedream_agent_app.plugin_manager.McpStdioClient", _FakeMcpClient)
    destination = tmp_path / "normalized.ddpkg"
    detection = manager.detect_asset_source_with_plugins(source)
    assert detection.adapter_id == "example.cityjson"
    assert detection.source_format == "cityjson"
    result = manager.normalize_asset_source_with_plugin(
        source=source,
        detection=detection,
        destination=destination,
        expected_kind="map",
    )

    assert result == destination
    assert destination.read_bytes() == package_path.read_bytes()
    assert store.get_plugin(manifest.plugin_id)["status"] == "healthy"


# 功能：
#   验证外部资产插件不能修改描述符以冒充宿主进程内的声明式解析器。
# 输入：
#   tmp_path：惰性导入器夹具目录。
# 输出：
#   None：不返回业务数据。
def test_external_asset_adapter_cannot_claim_in_process_declarative_parser(tmp_path):
    root = tmp_path / "map-importer"
    root.mkdir()
    manifest = _external_map_importer_manifest(root)
    capability = manifest.capabilities[0]
    metadata = dict(capability.metadata)
    descriptors = [dict(value) for value in metadata["asset_source_adapters"]]
    descriptors[0]["execution_boundary"] = "declarative-parser"
    metadata["asset_source_adapters"] = descriptors
    invalid = manifest.model_copy(
        update={
            "capabilities": [capability.model_copy(update={"metadata": metadata})],
        }
    )

    with pytest.raises(PluginManagerError, match="PLUGIN_ASSET_SOURCE_ADAPTER_BOUNDARY_INVALID"):
        PluginManager._asset_source_adapter_descriptors(invalid, builtin=False)


# 功能：
#   验证流水线插件可并存，配置调整只影响新快照而不污染已冻结配置。
# 输入：
#   tmp_path：独立应用存储目录。
# 输出：
#   None：不返回业务数据。
def test_pipeline_plugins_coexist_and_snapshot_configuration_is_immutable(tmp_path):
    manager = PluginManager(AppStore(tmp_path))
    snapshot = manager.snapshot()

    enabled = manager.enable("prompt.payload-custody")
    manager.configure("runtime.anomaly-battery", {"minimum_battery_percent": 30})

    assert enabled["replaced_plugins"] == []
    prompt_plugins = {
        item.plugin_id
        for item in manager.snapshot().plugins
        if item.plugin_id.startswith("prompt.")
    }
    assert {"prompt.structured-discipline", "prompt.payload-custody"} <= prompt_plugins
    frozen_battery = next(
        item for item in snapshot.plugins if item.plugin_id == "runtime.anomaly-battery"
    )
    assert frozen_battery.configuration == {}


# 功能：
#   验证产品默认、账户主动选择与 Harness 设计者选择分别冻结真实来源及必要摘要。
# 输入：
#   tmp_path：独立选择记录存储目录。
# 输出：
#   None：不返回业务数据。
def test_snapshot_freezes_actual_account_and_harness_selection_provenance(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    managed = next(
        item for item in manager.snapshot().plugins if item.plugin_id == "navigation.shortest-route"
    )
    assert managed.selection_source == "product_managed_default"
    assert managed.selected_by == "product_managed"

    manager.enable("navigation.clearance-first-route")
    account_selected = next(
        item
        for item in manager.snapshot().plugins
        if item.plugin_id == "navigation.clearance-first-route"
    )
    assert account_selected.selection_source == "explicit"
    assert account_selected.selected_by == "account_configurable"
    assert account_selected.selection_receipt_sha256 is not None
    assert account_selected.harness_revision_sha256 is None

    revision_sha256 = "a" * 64
    manager.apply_profile(
        "harness.profile-payload-delivery",
        selected_by="agent_harness_designer",
        harness_revision_sha256=revision_sha256,
    )
    profile_snapshot = manager.snapshot()
    profile_selected = next(
        item
        for item in profile_snapshot.plugins
        if item.plugin_id == "harness.profile-payload-delivery"
    )
    assert profile_selected.selection_source == "explicit"
    assert profile_selected.selected_by == "agent_harness_designer"
    assert profile_selected.harness_revision_sha256 == revision_sha256


# 功能：
#   验证递送配置一次选择相互配套的规划、校验和运行策略，并撤下互斥默认实现。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_persona_profile_atomically_selects_single_slots_and_optional_plugins(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    applied = manager.apply_profile("harness.profile-payload-delivery")
    active = {item.plugin_id for item in manager.snapshot().plugins}

    assert applied["profile_plugin_id"] == "harness.profile-payload-delivery"
    assert {
        "harness.profile-payload-delivery",
        "workflow.deliberate",
        "models.role-adversarial",
        "context.structured-window",
        "tools.router-safety-first",
        "navigation.stability-first-route",
        "safety.conservative-route-clearance",
        "px4.stability-track",
        "prompt.payload-custody",
        "validation.energy-reserve",
        "validation.clearance-speed-coupling",
        "runtime.checkpoint-risk-adaptive",
        "runtime.replan-mission-continuity",
    } <= active
    assert {
        "harness.profile-balanced",
        "workflow.balanced",
        "navigation.shortest-route",
        "safety.route-clearance",
        "px4.export-track",
    }.isdisjoint(active)


# 功能：
#   验证各产品配置选择其声明的检查点、重规划和间距速度耦合策略，而非全部套用默认。
# 输入：
#   tmp_path：独立插件存储目录。
#   profile_id：待应用配置。
#   checkpoint_plugin：期望的检查点策略。
#   replan_plugin：期望的重规划策略。
#   clearance_speed_gate：是否应启用间距速度耦合校验。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("profile_id", "checkpoint_plugin", "replan_plugin", "clearance_speed_gate"),
    [
        (
            "harness.profile-balanced",
            "runtime.checkpoint-every-segment",
            "runtime.replan-nearest-anchor",
            False,
        ),
        (
            "harness.profile-indoor-guardian",
            "runtime.checkpoint-risk-adaptive",
            "runtime.replan-verified-anchor",
            False,
        ),
        (
            "harness.profile-payload-delivery",
            "runtime.checkpoint-risk-adaptive",
            "runtime.replan-mission-continuity",
            True,
        ),
        (
            "harness.profile-field-readiness",
            "runtime.checkpoint-risk-adaptive",
            "runtime.replan-mission-continuity",
            True,
        ),
        (
            "harness.profile-infrastructure-inspection",
            "runtime.checkpoint-risk-adaptive",
            "runtime.replan-mission-continuity",
            True,
        ),
        (
            "harness.profile-area-survey",
            "runtime.checkpoint-mission-boundaries",
            "runtime.replan-nearest-anchor",
            False,
        ),
        (
            "harness.profile-emergency-response",
            "runtime.checkpoint-risk-adaptive",
            "runtime.replan-mission-continuity",
            True,
        ),
    ],
)
def test_persona_profiles_choose_distinct_runtime_policy_combinations(
    tmp_path,
    profile_id,
    checkpoint_plugin,
    replan_plugin,
    clearance_speed_gate,
):
    manager = PluginManager(AppStore(tmp_path))

    manager.apply_profile(profile_id)
    active = {item.plugin_id for item in manager.snapshot().plugins}

    assert checkpoint_plugin in active
    assert replan_plugin in active
    assert ("validation.clearance-speed-coupling" in active) is clearance_speed_gate


# 功能：
#   逐个应用当前内置配置并创建有效快照，确保任意时刻只选择一个配置档。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_all_first_party_persona_profiles_produce_a_valid_snapshot(tmp_path):
    manager = PluginManager(AppStore(tmp_path))
    profile_ids = [
        item["plugin_id"]
        for item in manager.list_plugins()
        if item["placement"]["slot_id"] == "harness.profile"
    ]

    for profile_id in profile_ids:
        manager.apply_profile(str(profile_id))
        snapshot = manager.snapshot()
        assert (
            sum(entry.plugin_id.startswith("harness.profile-") for entry in snapshot.plugins) == 1
        )


# 功能：
#   验证整组选择最终生成快照失败时恢复此前所有插件的启用状态。
# 输入：
#   tmp_path：独立存储目录。
#   monkeypatch：注入快照生成异常。
# 输出：
#   None：不返回业务数据。
def test_profile_failure_restores_the_previous_enabled_catalog(tmp_path, monkeypatch):
    manager = PluginManager(AppStore(tmp_path))
    before = {str(item["plugin_id"]): bool(item["enabled"]) for item in manager.list_plugins()}

    # 功能：
    #   在配置提交的快照阶段抛出合成异常，检验事务回滚覆盖末尾步骤。
    # 输入：
    #   thread_id：可选任务标识。
    # 输出：
    #   None：不返回业务数据。
    def fail_snapshot(*, thread_id=None):
        raise PluginManagerError("SYNTHETIC_PROFILE_COMMIT_FAILURE")

    monkeypatch.setattr(manager, "snapshot", fail_snapshot)
    with pytest.raises(PluginManagerError, match="SYNTHETIC_PROFILE_COMMIT_FAILURE"):
        manager.apply_profile("harness.profile-indoor-guardian")

    after = {str(item["plugin_id"]): bool(item["enabled"]) for item in manager.list_plugins()}
    assert after == before


# 功能：
#   验证插件声明的安全悬停或下次任务生效策略原样交给运行时退出检查。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_manifest_swap_policy_is_passed_to_runtime_guard(tmp_path):
    manager = PluginManager(AppStore(tmp_path))
    calls: list[tuple[str, str]] = []
    manager.set_disable_guard(
        lambda plugin_id, swap_policy: calls.append((plugin_id, swap_policy)) or []
    )

    manager.disable("runtime.anomaly-tracking")
    manager.enable("navigation.clearance-first-route")

    assert ("runtime.anomaly-tracking", "safe-hold") in calls
    assert ("navigation.shortest-route", "next-mission") in calls


# 功能：
#   验证单选替换遭运行时拒绝后，原实现仍启用且健康，候选实现仍禁用。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_failed_single_slot_guard_restores_the_previous_live_plugin(tmp_path):
    store = AppStore(tmp_path)
    manager = PluginManager(store)
    manager.set_disable_guard(
        lambda _plugin_id, _policy: (_ for _ in ()).throw(RuntimeError("busy"))
    )

    with pytest.raises(PluginManagerError, match="PLUGIN_SWAP_FAILED:busy"):
        manager.enable("navigation.clearance-first-route")

    original = store.get_plugin("navigation.shortest-route")
    replacement = store.get_plugin("navigation.clearance-first-route")
    assert original["enabled"] is True
    assert original["status"] == "healthy"
    assert replacement["enabled"] is False


# 功能：
#   验证整组替换的退出检查失败时恢复每个被临时标为退出中的插件。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_failed_profile_guard_restores_all_draining_plugins(tmp_path):
    store = AppStore(tmp_path)
    manager = PluginManager(store)
    before = {
        str(item["plugin_id"]): (bool(item["enabled"]), str(item["status"]))
        for item in store.list_plugins()
    }
    manager.set_disable_guard(
        lambda _plugin_id, _policy: (_ for _ in ()).throw(RuntimeError("busy"))
    )

    with pytest.raises(PluginManagerError, match="PLUGIN_PROFILE_DRAIN_FAILED:busy"):
        manager.apply_profile("harness.profile-indoor-guardian")

    after = {
        str(item["plugin_id"]): (bool(item["enabled"]), str(item["status"]))
        for item in store.list_plugins()
    }
    assert after == before


# 功能：
#   验证批量生命周期更新遇到不存在的成员时，已经处理的成员也整体回滚。
# 输入：
#   tmp_path：独立数据库目录。
# 输出：
#   None：不返回业务数据。
def test_batch_plugin_lifecycle_update_rolls_back_on_missing_member(tmp_path):
    store = AppStore(tmp_path)
    PluginManager(store)
    before = store.get_plugin("prompt.payload-custody")

    with pytest.raises(KeyError):
        store.set_plugin_lifecycles(
            {
                "prompt.payload-custody": {
                    "enabled": True,
                    "status": "healthy",
                    "health": "healthy",
                },
                "missing.plugin": {"enabled": True},
            }
        )

    after = store.get_plugin("prompt.payload-custody")
    assert after["enabled"] == before["enabled"]
    assert after["status"] == before["status"]


# 功能：
#   验证升级提供的新官方包可替换禁用插件的版本选择，但不会顺带将其启用。
# 输入：
#   tmp_path：独立官方索引及安装存储目录。
# 输出：
#   None：不返回业务数据。
def test_disabled_official_plugin_activates_new_staged_version_on_app_upgrade(tmp_path):
    store = AppStore(tmp_path / "store")
    official_root = _official_index(tmp_path / "official", version="1.0.0")
    PluginManager(store, official_plugins_root=official_root)

    _official_index(official_root, version="1.1.0")
    PluginManager(store, official_plugins_root=official_root)

    upgraded = store.get_plugin("example.route-audit")
    assert upgraded["version"] == "1.1.0"
    assert upgraded["enabled"] is False


# 功能：
#   检查包的安装、精确批准、启停、暂存更新、版本选择、回滚及卸载的完整本地生命周期。
# 输入：
#   tmp_path：只包含测试归档和安装记录的目录。
# 输出：
#   None：不返回业务数据。
def test_plugin_bundle_install_enable_disable_update_rollback_and_uninstall(tmp_path):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)

    installed = manager.import_bundle(_bundle(tmp_path / "plugin-1.zip"))
    assert installed["status"] == "disabled"
    assert installed["trust_status"] == "unverified"
    manager.approve_local_package("example.route-audit")
    assert manager.enable("example.route-audit")["health"] == "healthy"
    assert manager.disable("example.route-audit")["status"] == "disabled"

    updated = manager.import_bundle(_bundle(tmp_path / "plugin-2.zip", version="1.1.0"))
    assert updated["version"] == "1.0.0"
    assert updated["staged_version"] == "1.1.0"
    assert len(store.list_plugin_versions("example.route-audit")) == 2
    assert manager.activate_version("example.route-audit", "1.1.0")["version"] == "1.1.0"
    assert manager.rollback("example.route-audit", "1.0.0")["version"] == "1.0.0"

    removed = manager.uninstall("example.route-audit")
    assert removed["status"] == "uninstalled"
    assert not (store.plugins_root / "example.route-audit").exists()


# 功能：
#   验证新包健康检查失败时自动恢复原启用版本，并保存明确的回滚回执。
# 输入：
#   tmp_path：独立包及数据库目录。
#   monkeypatch：仅让新版本的健康检查失败。
# 输出：
#   None：不返回业务数据。
def test_health_gated_promotion_automatically_restores_enabled_previous_version(
    tmp_path, monkeypatch
):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    manager.import_bundle(_bundle(tmp_path / "plugin-1.zip"))
    manager.approve_local_package("example.route-audit")
    manager.enable("example.route-audit")
    manager.import_bundle(_bundle(tmp_path / "plugin-2.zip", version="1.1.0"))
    approved = manager.approve_local_version("example.route-audit", "1.1.0")
    assert approved["trust_status"] == "local-approved"
    assert store.get_plugin("example.route-audit")["version"] == "1.0.0"
    original_healthcheck = manager.healthcheck

    # 功能：
    #   为新版本注入健康失败，旧版本继续走真实检查，以验证恢复而不只是切回版本文字。
    # 输入：
    #   plugin_id：当前检查的插件标识。
    # 输出：
    #   result：原版本真实健康检查结果。
    def version_healthcheck(plugin_id: str):
        if store.get_plugin(plugin_id)["version"] == "1.1.0":
            raise PluginManagerError("NEW_VERSION_HEALTH_FAILED")
        result = original_healthcheck(plugin_id)
        return result

    monkeypatch.setattr(manager, "healthcheck", version_healthcheck)
    with pytest.raises(PluginManagerError, match="PROMOTION_ROLLED_BACK"):
        manager.promote_version("example.route-audit", "1.1.0")

    restored = store.get_plugin("example.route-audit")
    assert restored["version"] == "1.0.0"
    assert restored["enabled"] is True
    assert any(
        event["operation"] == "rollback"
        and bool(event["accepted"])
        and "PLUGIN_PROMOTION_AUTO_ROLLBACK" in event["receipt"]["issue_codes"]
        for event in store.list_plugin_events("example.route-audit")
    )


# 功能：
#   验证导入拒绝目录穿越归档及未获认证的执行器权限声明，不产生实际控制动作。
# 输入：
#   tmp_path：恶意归档夹具和独立安装目录。
# 输出：
#   None：不返回业务数据。
def test_plugin_import_rejects_path_traversal_and_uncertified_actuation(tmp_path):
    manager = PluginManager(AppStore(tmp_path / "store"))
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../plugin.json", "{}")
    with pytest.raises(PluginManagerError, match="PATH_TRAVERSAL"):
        manager.import_bundle(traversal)

    manifest = _manifest()
    manifest["permissions"] = ["process.spawn", "vehicle.actuate"]
    manifest["runtime"] = {
        "kind": "mcp-stdio",
        "command": ["plugin.exe"],
        "protocol_version": "2025-06-18",
        "startup_timeout_seconds": 15,
        "call_timeout_seconds": 60,
    }
    capability = manifest["capabilities"][0]  # type: ignore[index]
    capability["authority"] = "actuate"  # type: ignore[index]
    capability["kind"] = "tool"  # type: ignore[index]
    capability["input_schema"] = {"type": "object"}  # type: ignore[index]
    capability["output_schema"] = {"type": "object"}  # type: ignore[index]
    archive_path = tmp_path / "actuate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr("ui/panel.json", b'{"title":"Route audit"}\n')
    with pytest.raises(PluginManagerError, match="ACTUATION_NOT_CERTIFIED"):
        manager.import_bundle(archive_path)


# 功能：
#   验证安全悬停及内置仿真等受保护能力不能通过普通插件管理入口撤下。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_safety_kernel_plugins_cannot_be_disabled_or_uninstalled(tmp_path):
    manager = PluginManager(AppStore(tmp_path))

    with pytest.raises(PluginManagerError, match="REQUIRED_BY_SAFETY_KERNEL"):
        manager.disable("runtime.safe-hold")
    with pytest.raises(PluginManagerError, match="NOT_UNINSTALLABLE"):
        manager.uninstall("simulation.gazebo-px4")


# 功能：
#   验证停用提供者后模型目录、路由和快照检查一致拒绝其能力，其他提供者仍可查询。
# 输入：
#   tmp_path：独立插件存储目录。
# 输出：
#   None：不返回业务数据。
def test_disabling_model_provider_removes_its_models_without_affecting_others(tmp_path):
    manager = PluginManager(AppStore(tmp_path))
    snapshot = manager.snapshot()

    manager.disable("model.kimi")

    providers = {item["provider"] for item in manager.model_catalog()}
    assert providers == {"openai", "deepseek"}
    assert manager.provider_for_model("gpt-5.4") == "openai"
    assert manager.model_supports_image_input("gpt-5.4") is True
    assert manager.model_supports_image_input("deepseek-v4-flash") is False
    with pytest.raises(PluginManagerError, match="MODEL_PROVIDER_PLUGIN_UNAVAILABLE"):
        manager.provider_for_model("kimi-k3")
    with pytest.raises(PluginManagerError, match="PLUGIN_SNAPSHOT_CAPABILITY_DRIFT"):
        manager.assert_snapshot_active(snapshot, {"model.kimi.structured"})


# 功能：
#   验证被启用插件依赖的包不能被单独停用或卸载，避免留下断裂的运行依赖。
# 输入：
#   tmp_path：独立安装与数据库目录。
# 输出：
#   None：不返回业务数据。
def test_enabled_dependency_prevents_unsafe_plugin_withdrawal(tmp_path):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    manager.import_bundle(_bundle(tmp_path / "plugin.zip"))
    manager.approve_local_package("example.route-audit")
    manager.enable("example.route-audit")
    dependent_value = _manifest()
    dependent_value.update(
        {
            "plugin_id": "example.dependent-panel",
            "name": "Dependent Panel",
            "dependencies": [
                {
                    "plugin_id": "example.route-audit",
                    "version": ">=1.0.0",
                    "optional": False,
                }
            ],
        }
    )
    dependent = PluginManifest.model_validate(dependent_value)
    store.upsert_plugin(
        manifest=dependent,
        package_sha256="b" * 64,
        bundle_root=tmp_path,
        builtin=False,
        enabled=True,
        status="healthy",
        health="healthy",
    )

    with pytest.raises(PluginManagerError, match="PLUGIN_REQUIRED_BY_ENABLED"):
        manager.disable("example.route-audit")
    with pytest.raises(PluginManagerError, match="PLUGIN_REQUIRED_BY"):
        manager.uninstall("example.route-audit")
