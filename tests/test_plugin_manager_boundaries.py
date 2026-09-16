"""Lifecycle failure injection uses disposable local stores, never installed user plugins."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from dronedream_agent_app.plugin_manager import PluginManager, PluginManagerError
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import PluginManifest
from dronedream_agent_core.plugin_files import portable_plugin_path, verify_plugin_files
from dronedream_agent_core.plugin_process import McpSessionFailure
from dronedream_agent_core.plugin_values import plugin_json_value
from dronedream_agent_core.plugin_versions import version_matches, version_precedence


# 功能：
#   为每个测试创建独立插件管理器，先关闭会话池再关闭数据库，避免后台回调访问已销毁存储。
# 输入：
#   tmp_path：独立存储目录。
# 输出：
#   service：供当前测试使用的插件管理器。
@pytest.fixture
def manager(tmp_path):
    store = AppStore(tmp_path / "store")
    service = PluginManager(store)
    try:
        yield service
    finally:
        try:
            service.close()
        finally:
            store.close()


# 功能：
#   创建不执行代码的声明式面板包，可包含需要独立校验的嵌套清单负载。
# 输入：
#   path：测试归档输出路径。
#   version：夹具插件版本。
#   nested：是否添加嵌套的普通清单负载。
# 输出：
#   path：已创建的测试包路径。
def panel_bundle(path: Path, *, version="1.0.0", nested=False) -> Path:
    files = {"ui/panel.json": b'{"title":"Fixture"}'}
    if nested:
        files["nested/plugin.json"] = b'{"purpose":"payload, not root manifest"}'
    manifest = {
        "plugin_id": "example.boundary",
        "name": "Boundary",
        "publisher": "Fixture",
        "version": version,
        "description": "Local fixture",
        "runtime": {"kind": "ui-declarative"},
        "capabilities": [
            {
                "capability_id": "example.boundary.panel",
                "kind": "ui-panel",
                "name": "Panel",
                "description": "Local fixture",
                "authority": "read",
                "metadata": {"entrypoint": "ui/panel.json"},
            }
        ],
        "permissions": ["ui.panel"],
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
    return path


# 功能：
#   验证便携路径拒绝跨目录、设备名、点段、控制字符等含义不一致或危险的名称。
# 输入：
#   name：故意构造的非法包内路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "/x",
        "a\\x",
        "a//x",
        "a/./x",
        "nul.txt",
        "x/COM1",
        "a:x",
        "a?x",
        "a. ",
        "a/",
        "x\x00y",
    ],
)
def test_portable_paths_reject_aliases_and_device_names(name):
    with pytest.raises(ValueError):
        portable_plugin_path(name)


# 功能：
#   验证解压之前拒绝大小写重名或文件与目录冲突，不靠解压后的路径覆盖来消除歧义。
# 输入：
#   names：相互冲突的归档成员名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("names", [["a", "A"], ["a", "a/b"], ["dir/x", "DIR/X"]])
def test_zip_namespace_is_checked_before_extraction(names):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in names:
            archive.writestr(name, b"x")
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive, pytest.raises(PluginManagerError):
        PluginManager._validated_archive_members(archive)


# 功能：
#   验证嵌套清单属于受摘要保护的负载，重导入不能掩盖已经被修改的安装内容。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_nested_manifest_is_a_real_payload_and_reimport_checks_installed_bytes(manager, tmp_path):
    archive = panel_bundle(tmp_path / "bundle.zip", nested=True)
    installed = manager.import_bundle(archive)
    root = Path(installed["bundle_root"])
    manifest = PluginManifest.model_validate(installed["manifest"])
    verify_plugin_files(root, manifest.file_sha256)
    (root / "nested/plugin.json").write_bytes(b"changed")
    with pytest.raises(PluginManagerError, match="HASH_MISMATCH"):
        manager.import_bundle(archive)
    assert (root / "nested/plugin.json").read_bytes() == b"changed"


# 功能：
#   验证安装回执保存失败时撤回数据库和本次创建目录，保留用户提供的归档。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：归档目录。
#   monkeypatch：注入回执写入失败。
# 输出：
#   None：不返回业务数据。
def test_install_receipt_failure_rolls_back_database_and_only_owned_directory(
    manager, tmp_path, monkeypatch
):
    archive = panel_bundle(tmp_path / "bundle.zip")

    # 功能：
    #   在生命周期回执保存入口制造事务失败。
    # 输入：
    #   _receipt：不实际保存的回执。
    # 输出：
    #   None：不返回业务数据。
    def fail_receipt(_receipt):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(manager.store, "record_plugin_event", fail_receipt)
    with pytest.raises(RuntimeError, match="injected"):
        manager.import_bundle(archive)
    with pytest.raises(KeyError):
        manager.store.get_plugin("example.boundary")
    with pytest.raises(KeyError):
        manager.store.get_plugin_version("example.boundary", "1.0.0")
    assert not (manager.store.plugins_root / "example.boundary/1.0.0").exists()
    assert archive.is_file()


# 功能：
#   验证选择不存在版本时直接拒绝，当前正常启用版本不被提前停用。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：面板包目录。
# 输出：
#   None：不返回业务数据。
def test_missing_version_does_not_disable_current_plugin(manager, tmp_path):
    manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
    manager.approve_local_package("example.boundary")
    manager.enable("example.boundary")
    with pytest.raises(KeyError):
        manager.activate_version("example.boundary", "9.9.9")
    assert manager.store.get_plugin("example.boundary")["enabled"] is True


# 功能：
#   验证退出检查失败时保留原来禁用的状态，恢复逻辑不能擅自启用插件。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_failed_drain_does_not_enable_an_already_disabled_plugin(manager, tmp_path):
    manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))

    # 功能：
    #   模拟任务协调器因忙碌而拒绝切换。
    # 输入：
    #   _id：待退出插件标识。
    #   _policy：请求使用的替换策略。
    # 输出：
    #   None：不返回业务数据。
    def deny(_id, _policy):
        raise RuntimeError("busy")

    manager.set_disable_guard(deny)
    with pytest.raises(PluginManagerError, match="DRAIN_FAILED"):
        manager.disable("example.boundary")
    assert manager.store.get_plugin("example.boundary")["enabled"] is False


# 功能：
#   验证钩子调用用量归属真正选中的实现，而非目录循环最后访问到的无关清单。
# 输入：
#   manager：独立插件管理器。
#   monkeypatch：用内存列表接收用量证据。
# 输出：
#   None：不返回业务数据。
def test_usage_is_attributed_to_selected_hook_not_last_catalog_manifest(manager, monkeypatch):
    definition = next(
        value
        for value in manager._definitions.values()
        if value.hooks
        and value.manifest.placement.activation_mode == "single"
        and manager.store.get_plugin(value.manifest.plugin_id)["enabled"]
    )
    manifest = definition.manifest
    manager._definitions[manifest.plugin_id] = replace(
        definition, hooks={"fixture": lambda: {"ok": True}}
    )
    usage = []
    monkeypatch.setattr(manager.store, "record_plugin_usage", usage.append)
    assert manager.invoke_single_slot(manifest.placement.slot_id, "fixture") == {"ok": True}
    assert usage[0]["capability_id"] == manifest.capabilities[0].capability_id
    assert usage[0]["plugin_version"] == manifest.version


# 功能：
#   核对插槽解析使用冻结清单，并拒绝配置被原位修改而未同步摘要的快照。
# 输入：
#   manager：独立插件管理器。
# 输出：
#   None：不返回业务数据。
def test_snapshot_slot_resolution_preserves_frozen_manifest_and_rejects_hash_drift(manager):
    snapshot = manager.snapshot()
    entry = snapshot.plugins[0]
    assert entry.manifest is not None
    slot = entry.manifest.placement.slot_id
    if entry.manifest.placement.activation_mode == "single" and len(entry.capability_ids) == 1:
        _selected, resolved, _capability = manager.capability_for_slot(snapshot, slot)
        assert resolved == entry.manifest
    entry.configuration["tampered"] = True
    with pytest.raises(ValueError, match="CONFIGURATION_DRIFT"):
        manager.build_extension_registry(snapshot=snapshot)


# 功能：
#   验证组合配置包含未批准的外部插件时仍要求本地同意，不通过内置组合绕过信任检查。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：未签名测试包目录。
# 输出：
#   None：不返回业务数据。
def test_profile_does_not_bypass_unsigned_external_plugin_consent(manager, tmp_path):
    manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
    definition = next(
        value
        for value in manager._definitions.values()
        if value.manifest.placement.slot_id == "harness.profile"
    )
    modified = definition.manifest.model_copy(deep=True)
    modified.capabilities[0].metadata["recommended_plugins"].append("example.boundary")
    manager._definitions[modified.plugin_id] = PluginDefinition(
        manifest=modified, hooks=definition.hooks
    )
    manager.store.upsert_plugin(
        manifest=modified,
        package_sha256=sha256_json(modified),
        bundle_root=None,
        builtin=True,
        enabled=False,
        status="disabled",
        health="healthy",
    )
    with pytest.raises(PluginManagerError, match="TRUST_REQUIRED"):
        manager.apply_profile(modified.plugin_id)
    assert manager.store.get_plugin("example.boundary")["enabled"] is False


# 功能：
#   验证明确的插件批事务可以整体撤回，但普通数据库事务仍拒绝意外嵌套。
# 输入：
#   manager：独立插件管理器及数据库。
# 输出：
#   None：不返回业务数据。
def test_explicit_store_batch_rolls_back_all_members_and_default_nesting_stays_forbidden(manager):
    store = manager.store
    with pytest.raises(RuntimeError, match="injected"), store.atomic_plugin_changes():
        store.patch_settings({"fixture": "temporary"})
        raise RuntimeError("injected")
    assert "fixture" not in store.get_settings()
    with (
        store.transaction(),
        pytest.raises(RuntimeError, match="NESTED_TRANSACTION"),
        store.transaction(),
    ):
        pass


# 功能：
#   核对零主版本、预发布和构建元数据的 SemVer 匹配边界，不以字符串大小比较替代。
# 输入：
#   actual：候选版本。
#   requirement：版本要求。
#   expected：应当得到的匹配结果。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "actual, requirement, expected",
    [
        ("0.2.0", "^0.1.0", False),
        ("0.1.9", "^0.1.0", True),
        ("0.0.2", "^0.0.1", False),
        ("1.0.0-rc.1", ">=1.0.0", False),
        ("1.0.0-rc.1", "1.0.0", False),
        ("1.0.0+a", "1.0.0+b", True),
        ("1.0.0-alpha.2", ">=1.0.0-alpha.1", True),
        ("2.0.0-alpha", "*", False),
    ],
)
def test_dependency_matching_uses_semver_prerelease_and_zero_major_bounds(
    actual, requirement, expected
):
    assert version_matches(actual, requirement) is expected


# 功能：
#   验证非法版本文本直接拒绝，不自动修补前导零或空版本段。
# 输入：
#   value：非法版本文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["1.0.0-01", "1.0.0-a..b", "1.0.0+", "01.0.0"])
def test_invalid_semver_is_not_coerced(value):
    with pytest.raises(ValueError):
        version_precedence(value)


# 功能：
#   验证钩子编码拒绝非字符串键、非有限数和循环引用，并保持输出副本及集合排序确定性。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_hook_json_conversion_is_finite_detached_and_deterministic():
    class Measurement(BaseModel):
        value: float

    for value in ({1: "numeric", "1": "text"}, Measurement(value=float("nan"))):
        with pytest.raises(ValueError):
            plugin_json_value(value)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        plugin_json_value(cycle)
    source = {"values": [1], "set": {"b", "a"}, "time": datetime(2026, 1, 1, tzinfo=UTC)}
    encoded = plugin_json_value(source)
    encoded["values"].append(2)
    assert source["values"] == [1]
    assert encoded["set"] == ["a", "b"]
    assert encoded["time"] == "2026-01-01T00:00:00+00:00"


# 功能：
#   验证普通停用不打断已有快照，但精确包撤销会立即阻止同一快照的新调用。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_disabled_snapshot_can_finish_but_revoked_package_cannot(manager, tmp_path):
    installed = manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
    manager.approve_local_package("example.boundary")
    manager.enable("example.boundary")
    entry = next(
        item for item in manager.snapshot().plugins if item.plugin_id == "example.boundary"
    )
    manager.disable(entry.plugin_id)
    manager._acquire_snapshot_call(entry)
    manager._release_call(entry.plugin_id)
    manager.trust_store.revoke_package(installed["package_sha256"])
    with pytest.raises(PluginManagerError, match="PACKAGE_REVOKED"):
        manager._acquire_snapshot_call(entry)
    assert entry.plugin_id not in manager._inflight_calls


# 功能：
#   验证旧配置会话的迟到故障不隔离新配置，身份一致的真实故障才隔离当前插件。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_late_session_failure_cannot_quarantine_replacement_configuration(manager, tmp_path):
    installed = manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
    manager.approve_local_package("example.boundary")
    manager.enable("example.boundary")
    failure = McpSessionFailure(
        "example.boundary",
        installed["package_sha256"],
        sha256_json({"old": True}),
        "PLUGIN_HEARTBEAT_INVALID",
    )
    manager._quarantine_unhealthy_session(failure)
    assert manager.store.get_plugin("example.boundary")["enabled"] is True
    manager._quarantine_unhealthy_session(replace(failure, configuration_sha256=sha256_json({})))
    assert manager.store.get_plugin("example.boundary")["status"] == "quarantined"


# 功能：
#   验证调用方不能覆盖任务冻结配置，前一次钩子修改也不会污染下一次调用。
# 输入：
#   manager：独立插件管理器。
# 输出：
#   None：不返回业务数据。
def test_hook_configuration_is_not_overridden_or_reused_after_mutation(manager):
    definition = next(
        value
        for value in manager._definitions.values()
        if value.hooks
        and value.manifest.placement.activation_mode == "single"
        and manager.store.get_plugin(value.manifest.plugin_id)["enabled"]
    )
    plugin_id = definition.manifest.plugin_id
    manager.store.save_plugin_configuration(plugin_id, {"nested": [1]})

    # 功能：
    #   返回收到的原配置并修改本次副本，用来检验跨调用引用是否隔离。
    # 输入：
    #   configuration：框架注入的配置副本。
    # 输出：
    #   result：修改前的配置列表。
    def handler(*, configuration):
        previous = list(configuration["nested"])
        configuration["nested"].append(2)
        result = {"previous": previous}
        return result

    manager._definitions[plugin_id] = replace(definition, hooks={"fixture": handler})
    registry = manager.build_extension_registry(snapshot=manager.snapshot())
    for _ in range(2):
        output, receipts = registry.invoke_single(
            definition.manifest.placement.slot_id, "fixture", configuration={"nested": [999]}
        )
        assert output == {"previous": [1]}
        assert receipts[0].outcome == "accepted"


# 功能：
#   验证停用回执保存失败时撤回临时状态，原健康启用插件不留在退出中。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
#   monkeypatch：注入回执写入失败。
# 输出：
#   None：不返回业务数据。
def test_disable_receipt_failure_restores_original_enabled_state(manager, tmp_path, monkeypatch):
    manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
    manager.approve_local_package("example.boundary")
    manager.enable("example.boundary")

    # 功能：
    #   模拟回执存储暂不可用，保持错误正文不包含用户数据。
    # 输入：
    #   _receipt：本次待保存回执。
    # 输出：
    #   None：不返回业务数据。
    def fail(_receipt):
        raise RuntimeError("receipt unavailable")

    monkeypatch.setattr(manager.store, "record_plugin_event", fail)
    with pytest.raises(RuntimeError, match="receipt unavailable"):
        manager.disable("example.boundary")
    record = manager.store.get_plugin("example.boundary")
    assert record["enabled"] is True and record["status"] == "healthy"


# 功能：
#   验证归档清单与官方索引要求的身份不符时，在创建安装记录之前拒绝。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_official_expected_identity_rejects_before_installing_other_plugin(manager, tmp_path):
    archive = panel_bundle(tmp_path / "bundle.zip")
    with pytest.raises(PluginManagerError, match="IDENTITY_MISMATCH"):
        manager.import_bundle(archive, expected_identity=("different.plugin", "1.0.0"))
    with pytest.raises(KeyError):
        manager.store.get_plugin("example.boundary")


# 功能：
#   验证正在退出的模型提供者不再出现在可用目录，也不能通过独立视觉能力查询绕过。
# 输入：
#   manager：独立插件管理器。
# 输出：
#   None：不返回业务数据。
def test_draining_provider_disappears_from_routing_and_visual_capability(manager):
    manager.store.set_plugin_lifecycle(
        "model.openai", enabled=True, status="draining", health="healthy"
    )
    assert all(item["provider"] != "openai" for item in manager.model_catalog())
    with pytest.raises(PluginManagerError, match="UNAVAILABLE"):
        manager.model_supports_image_input("gpt-5.4")


# 功能：
#   验证关闭旧会话失败时恢复原目录状态，不把插件留在无法接收新调用的 draining 状态。
# 输入：
#   manager：使用独立数据库的插件管理器。
#   tmp_path：惰性面板插件归档目录。
#   monkeypatch：注入会话关闭失败。
#   operation：需要检查的禁用、替换或整组切换操作。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("operation", ["disable", "replace", "profile"])
def test_failed_session_close_restores_draining_catalog(manager, tmp_path, monkeypatch, operation):
    if operation == "disable":
        manager.import_bundle(panel_bundle(tmp_path / "bundle.zip"))
        manager.approve_local_package("example.boundary")
        manager.enable("example.boundary")
    before = {
        item["plugin_id"]: (item["enabled"], item["status"])
        for item in manager.store.list_plugins()
    }

    # 功能：
    #   模拟池已尝试回收但报告退出异常，检查上层事务是否完整撤回临时状态。
    # 输入：
    #   plugin_id：需要关闭会话的插件标识。
    # 输出：
    #   None：不返回业务数据。
    def reject_invalidation(plugin_id):
        raise RuntimeError("fixture session cleanup failed")

    monkeypatch.setattr(manager._mcp_sessions, "invalidate", reject_invalidation)
    with pytest.raises(RuntimeError, match="fixture session cleanup failed"):
        if operation == "disable":
            manager.disable("example.boundary")
        elif operation == "replace":
            manager.enable("navigation.clearance-first-route")
        else:
            manager.apply_profile("harness.profile-payload-delivery")
    after = {
        item["plugin_id"]: (item["enabled"], item["status"])
        for item in manager.store.list_plugins()
    }
    assert after == before


# 功能：
#   验证未批准或已撤销的外部可执行插件不能借健康检查启动进程，已批准包才可枚举工具。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：只含惰性测试字节的插件目录。
#   monkeypatch：拦截工具枚举，保证不会执行外部进程。
#   approval：未批准、本地批准或已撤销状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("approval", ["unapproved", "approved", "revoked"])
def test_healthcheck_enforces_execution_trust(manager, tmp_path, monkeypatch, approval):
    root = tmp_path / "executable-fixture"
    root.mkdir()
    payload = b"inert fixture, not an executable"
    (root / "plugin.exe").write_bytes(payload)
    manifest = PluginManifest.model_validate(
        {
            "plugin_id": "example.health-permission",
            "name": "Health permission fixture",
            "version": "1.0.0",
            "publisher": "Fixture",
            "description": "Exercise health-check approval without launching a process.",
            "runtime": {"kind": "mcp-stdio", "command": ["plugin.exe"]},
            "permissions": ["process.spawn"],
            "capabilities": [
                {
                    "capability_id": "example.health-permission.echo",
                    "kind": "tool",
                    "name": "Echo",
                    "authority": "read",
                    "description": "Inert test tool.",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                }
            ],
            "file_sha256": {"plugin.exe": hashlib.sha256(payload).hexdigest()},
        }
    )
    package_sha256 = hashlib.sha256(b"fixture package").hexdigest()
    manager.store.upsert_plugin(
        manifest=manifest,
        package_sha256=package_sha256,
        bundle_root=root,
        builtin=False,
        enabled=False,
        status="disabled",
        health="unknown",
    )
    if approval != "unapproved":
        manager.approve_local_package(manifest.plugin_id)
    if approval == "revoked":
        manager.trust_store.revoke_package(package_sha256)
    calls = []

    # 功能：
    #   记录是否到达可执行工具枚举边界，返回与清单一致的合成目录而不启动进程。
    # 输入：
    #   plugin：当前插件记录。
    #   candidate：当前清单。
    # 输出：
    #   catalog：与声明匹配的工具目录。
    def enumerate_tools(plugin, candidate):
        calls.append(plugin["plugin_id"])
        catalog = [
            {
                "name": candidate.capabilities[0].capability_id,
                "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
            }
        ]
        return catalog

    monkeypatch.setattr(manager, "_mcp_tools", enumerate_tools)
    if approval == "approved":
        checked = manager.healthcheck(manifest.plugin_id)
        assert checked["health"] == "healthy"
        assert checked["enabled"] is False
        assert calls == [manifest.plugin_id]
    else:
        with pytest.raises(PluginManagerError, match="PLUGIN_TRUST_REQUIRED"):
            manager.healthcheck(manifest.plugin_id)
        assert calls == []


# 功能：
#   验证版本提升的选择、健康检查或最终回执出现普通异常时，原启用版本会恢复。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：两个惰性版本包的输出目录。
#   monkeypatch：仅让目标步骤首次失败。
#   step：失败发生在版本选择、健康检查还是更新回执。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("step", ["activate", "health", "receipt"])
def test_promotion_recovers_from_late_operational_error(manager, tmp_path, monkeypatch, step):
    plugin_id = "example.boundary"
    manager.import_bundle(panel_bundle(tmp_path / "old.zip"))
    manager.approve_local_package(plugin_id)
    manager.enable(plugin_id)
    manager.import_bundle(panel_bundle(tmp_path / "new.zip", version="1.1.0"))
    manager.approve_local_version(plugin_id, "1.1.0")
    original_activate = manager.store.activate_plugin_version
    original_health = manager.healthcheck
    original_event = manager.store.record_plugin_event
    failed = False

    # 功能：
    #   在目标版本写入后注入一次异常，检验部分成功的数据库选择是否会被撤回。
    # 输入：
    #   selected_id：待切换的插件标识。
    #   version：目标版本。
    # 输出：
    #   record：正常情况下的版本选择结果。
    def activate(selected_id, version):
        nonlocal failed
        record = original_activate(selected_id, version)
        if step == "activate" and version == "1.1.0" and not failed:
            failed = True
            raise RuntimeError("fixture promotion failure")
        return record

    # 功能：
    #   只让新版本首次健康检查出现普通运行错误，旧版本仍使用实际检查流程。
    # 输入：
    #   selected_id：待检查的插件标识。
    # 输出：
    #   record：正常情况下的健康检查结果。
    def health(selected_id):
        nonlocal failed
        if (
            step == "health"
            and not failed
            and manager.store.get_plugin(selected_id)["version"] == "1.1.0"
        ):
            failed = True
            raise RuntimeError("fixture promotion failure")
        record = original_health(selected_id)
        return record

    # 功能：
    #   只让提升成功回执首次保存失败，保留后续回滚回执的正常写入能力。
    # 输入：
    #   event：待写入的生命周期回执。
    # 输出：
    #   None：不返回业务数据。
    def record_event(event):
        nonlocal failed
        if step == "receipt" and event["operation"] == "update" and not failed:
            failed = True
            raise RuntimeError("fixture promotion failure")
        original_event(event)

    monkeypatch.setattr(manager.store, "activate_plugin_version", activate)
    monkeypatch.setattr(manager, "healthcheck", health)
    monkeypatch.setattr(manager.store, "record_plugin_event", record_event)
    with pytest.raises(RuntimeError, match="fixture promotion failure"):
        manager.promote_version(plugin_id, "1.1.0")
    restored = manager.store.get_plugin(plugin_id)
    assert restored["version"] == "1.0.0"
    assert restored["enabled"] is True
    assert restored["health"] == "healthy"
    assert any(
        item["operation"] == "rollback" and item["accepted"]
        for item in manager.store.list_plugin_events(plugin_id)
    )


# 功能：
#   验证提升中断后保留中断类型；失败版本清理不成功时只恢复旧选择，不重新启用或伪报成功。
# 输入：
#   manager：独立插件管理器。
#   tmp_path：惰性版本包目录。
#   monkeypatch：注入健康检查中断或恢复清理失败。
#   cleanup_fails：是否模拟无法确认失败版本退出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_promotion_recovery_preserves_interrupt_and_cleanup_state(
    manager, tmp_path, monkeypatch, cleanup_fails
):
    plugin_id = "example.boundary"
    manager.import_bundle(panel_bundle(tmp_path / "old.zip"))
    manager.approve_local_package(plugin_id)
    manager.enable(plugin_id)
    manager.import_bundle(panel_bundle(tmp_path / "new.zip", version="1.1.0"))
    manager.approve_local_version(plugin_id, "1.1.0")
    original_health = manager.healthcheck
    original_invalidate = manager._mcp_sessions.invalidate

    # 功能：
    #   模拟新版本健康检查被用户中断，旧版本仍可接受真实检查。
    # 输入：
    #   selected_id：当前检查的插件标识。
    # 输出：
    #   record：旧版本真实检查结果。
    def interrupt_health(selected_id):
        if manager.store.get_plugin(selected_id)["version"] == "1.1.0":
            raise KeyboardInterrupt("fixture interrupted promotion")
        record = original_health(selected_id)
        return record

    # 功能：
    #   可选地让新版本恢复阶段的会话关闭失败，不影响提升前原版本的正常关闭。
    # 输入：
    #   selected_id：待关闭会话的插件标识。
    # 输出：
    #   None：不返回业务数据。
    def invalidate(selected_id):
        if cleanup_fails and manager.store.get_plugin(selected_id)["version"] == "1.1.0":
            raise RuntimeError("fixture recovery cleanup failed")
        original_invalidate(selected_id)

    monkeypatch.setattr(manager, "healthcheck", interrupt_health)
    monkeypatch.setattr(manager._mcp_sessions, "invalidate", invalidate)
    with pytest.raises(KeyboardInterrupt, match="fixture interrupted promotion") as raised:
        manager.promote_version(plugin_id, "1.1.0")
    restored = manager.store.get_plugin(plugin_id)
    assert restored["version"] == "1.0.0"
    assert restored["enabled"] is not cleanup_fails
    receipts = [
        item
        for item in manager.store.list_plugin_events(plugin_id)
        if item["operation"] == "rollback"
    ]
    assert len(receipts) == 1
    assert bool(receipts[0]["accepted"]) is not cleanup_fails
    if cleanup_fails:
        assert isinstance(raised.value.__cause__, ExceptionGroup)
