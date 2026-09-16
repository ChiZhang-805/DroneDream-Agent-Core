"""Offline regressions for frozen plugin configuration and authority boundaries."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dronedream_agent_core import plugin_api
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_contracts import PluginSnapshot, PluginSnapshotEntry
from dronedream_agent_plugins._helpers import hook_plugin


# 功能：
#   创建不连接模型的单选钩子定义，允许测试替换处理函数或明确指定空权限。
# 输入：
#   handler：可选的测试钩子。
#   permissions：权限列表；None 使用夹具默认，空列表保持无权限。
# 输出：
#   plugin：供冻结与重建测试使用的插件定义。
def definition(*, handler=None, permissions=None):
    plugin = hook_plugin(
        module_name=__name__,
        plugin_id="test.snapshot-hook",
        name="Snapshot hook",
        description="Offline configuration boundary fixture.",
        capability_id="test.snapshot-hook.apply",
        capability_kind="model-policy",
        capability_name="Snapshot hook",
        capability_description="Frozen configuration fixture.",
        category_id="models",
        category_label="Models",
        slot_id="models.snapshot-test",
        slot_label="Snapshot test",
        activation_mode="single",
        category_order=1,
        slot_order=1,
        plugin_order=1,
        default_enabled=True,
        failure_mode="fail-closed",
        hooks={"select_port": handler or (lambda **_: {"port": "primary"})},
        permissions=permissions,
    )
    return plugin


# 功能：
#   将测试插件和配置冻结成完整摘要一致的快照，不代表外部发布者签名或飞行授权。
# 输入：
#   plugin：测试插件定义。
#   configuration：需要冻结的配置，None 表示空配置。
# 输出：
#   snapshot：当前格式的测试快照。
def snapshot_for(plugin, configuration=None):
    configuration = {} if configuration is None else configuration
    manifest = plugin.manifest
    digest = sha256_json(manifest)
    entry = PluginSnapshotEntry(
        plugin_id=manifest.plugin_id,
        version=manifest.version,
        package_sha256=digest,
        manifest_sha256=digest,
        configuration=configuration,
        configuration_sha256=sha256_json(configuration),
        capability_ids=[item.capability_id for item in manifest.capabilities],
        manifest=manifest,
    )
    snapshot = PluginSnapshot(
        snapshot_id="plugin-snapshot-" + "a" * 24,
        catalog_sha256=sha256_json([entry.model_dump(mode="json")]),
        plugins=[entry],
        created_at=datetime.now(UTC),
    )
    return snapshot


# 功能：
#   创建只用于观察工厂配置绑定的环境，故意设置不同的全局配置以暴露错误继承。
# 输入：
#   tmp_path：不会实际读取的语义文件位置。
# 输出：
#   context：没有真实地图、仿真器或宿主授权的测试环境。
def environment(tmp_path):
    # 下方工厂只检查配置，不求路线；空地图不能用于实际执行。
    context = plugin_api.ToolEnvironment(
        map_graph=None,
        semantic_path=tmp_path / "unused.json",
        vehicle_diameter_m=0.6,
        vehicle_height_m=0.4,
        waypoint_hold_seconds=0.4,
        plugin_configuration={"weight": "wrong-global-configuration"},
    )
    return context


# 功能：
#   验证显式空权限不会被默认 mission.read 权限替换。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_empty_permissions_remain_empty():
    assert definition(permissions=[]).manifest.permissions == []
    assert definition().manifest.permissions == ["mission.read"]


# 功能：
#   验证工具和钩子重建都先拒绝配置漂移、目录摘要漂移、重复选择和废弃快照。
# 输入：
#   tmp_path：测试环境目录。
#   monkeypatch：仅提供合成内置定义。
#   corruption：本例施加的快照破坏。
#   builder：工具或钩子重建入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["configuration", "catalog", "duplicate", "obsolete"])
@pytest.mark.parametrize("builder", ["tools", "hooks"])
def test_snapshot_corruption_rejected_before_build(tmp_path, monkeypatch, corruption, builder):
    plugin = definition()
    snapshot = snapshot_for(plugin, {"weight": 2})
    monkeypatch.setattr(
        plugin_api,
        "discover_builtin_plugins",
        lambda: {
            plugin.manifest.plugin_id: plugin,
        },
    )
    if corruption == "configuration":
        snapshot.plugins[0].configuration["weight"] = 999
    elif corruption == "catalog":
        snapshot.catalog_sha256 = "0" * 64
    elif corruption == "duplicate":
        snapshot.plugins.append(snapshot.plugins[0].model_copy(deep=True))
        snapshot.catalog_sha256 = sha256_json(snapshot.plugins)
    else:
        snapshot.schema_version = "dronedream.plugin-snapshot.v1"
    with pytest.raises(ValueError, match="PLUGIN_SNAPSHOT"):
        if builder == "tools":
            plugin_api.build_snapshot_tool_registry(environment(tmp_path), snapshot)
        else:
            plugin_api.build_discovered_extension_registry(snapshot)


# 功能：
#   验证工具工厂接收本插件的冻结配置，后续修改原快照不会改变工厂已收到的对象。
# 输入：
#   tmp_path：测试环境目录。
#   monkeypatch：注入只记录配置的工厂。
# 输出：
#   None：不返回业务数据。
def test_tool_factory_receives_its_frozen_configuration(tmp_path, monkeypatch):
    received = []

    # 功能：
    #   记录工厂实际接收的配置，返回空工具集以隔离本测试与实际执行。
    # 输入：
    #   bound_environment：重建器传入的插件环境。
    # 输出：
    #   registered：空工具列表。
    def tools(bound_environment):
        received.append(bound_environment.plugin_configuration)
        registered = []
        return registered

    plugin = replace(definition(), hooks=None, tool_factory=tools)
    snapshot = snapshot_for(plugin, {"weight": 2})
    monkeypatch.setattr(
        plugin_api,
        "discover_builtin_plugins",
        lambda: {
            plugin.manifest.plugin_id: plugin,
        },
    )
    plugin_api.build_snapshot_tool_registry(environment(tmp_path), snapshot)
    assert received == [{"weight": 2}]
    snapshot.plugins[0].configuration["weight"] = 999
    assert received == [{"weight": 2}]


# 功能：
#   验证调用参数不能覆盖已冻结配置，钩子内部修改也不能污染下一次调用或原快照。
# 输入：
#   monkeypatch：注入修改自身配置副本的钩子。
# 输出：
#   None：不返回业务数据。
def test_hook_configuration_cannot_be_overridden_or_mutated_between_calls(monkeypatch):
    # 功能：
    #   读取初值后故意修改嵌套配置，检验宿主是否为每次调用创建独立副本。
    # 输入：
    #   configuration：本次调用获得的配置。
    # 输出：
    #   result：修改前读到的权重。
    def handler(*, configuration):
        value = configuration["nested"]["weight"]
        configuration["nested"]["weight"] = 999
        result = {"weight": value}
        return result

    plugin = definition(handler=handler)
    snapshot = snapshot_for(plugin, {"nested": {"weight": 2}})
    monkeypatch.setattr(
        plugin_api,
        "discover_builtin_plugins",
        lambda: {
            plugin.manifest.plugin_id: plugin,
        },
    )
    registry = plugin_api.build_discovered_extension_registry(snapshot)
    hook = registry.plugins_for_slot("models.snapshot-test")[0].hooks["select_port"]
    assert hook(configuration={"nested": {"weight": 111}}) == {"weight": 2}
    assert hook() == {"weight": 2}
    assert hook() == {"weight": 2}
    assert snapshot.plugins[0].configuration == {"nested": {"weight": 2}}


# 功能：
#   验证重算目录摘要不能使非法选择来源、嵌套清单或缺少必需能力的快照变得有效。
# 输入：
#   corruption：在快照通过初次构造后施加的非法修改。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("corruption", ["selection", "manifest", "capabilities"])
def test_matching_hash_does_not_replace_snapshot_contract_validation(corruption):
    snapshot = snapshot_for(definition())
    entry = snapshot.plugins[0]
    if corruption == "selection":
        entry.selection_source = "unrecognized-origin"
    elif corruption == "manifest":
        entry.manifest.placement.swap_policy = "skip-safety"
        entry.manifest_sha256 = sha256_json(entry.manifest)
        entry.package_sha256 = entry.manifest_sha256
    else:
        entry.manifest = None
        entry.capability_ids = []
    snapshot.catalog_sha256 = sha256_json(snapshot.plugins)
    with pytest.raises(ValueError, match="PLUGIN_SNAPSHOT_INVALID"):
        plugin_api.verify_plugin_snapshot(snapshot)
