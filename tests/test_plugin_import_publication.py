"""Inert plugin/package fixtures; no real process, account or installed asset is used."""

import hashlib
import zipfile

import pytest
from test_plugin_manager import (
    _external_map_importer_manifest,
    _FakeMcpClient,
    _source_adapter_package,
)

import dronedream_agent_app.plugin_manager as manager_module
import dronedream_agent_core.asset_package_storage as asset_storage
from dronedream_agent_app.plugin_manager import PluginManager, PluginManagerError
from dronedream_agent_app.storage import AppStore


# 功能：
#   建立只读写测试目录的转换器替身，并在测试结束时先释放会话池、再关闭数据库。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：自动恢复客户端替身及其类级状态的测试替换器。
# 输出：
#   fixture：管理器、转换器清单、源文件及识别结果组成的元组。
@pytest.fixture
def adapter_fixture(tmp_path, monkeypatch):
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    try:
        root = tmp_path / "adapter"
        root.mkdir()
        manifest = _external_map_importer_manifest(root)
        store.upsert_plugin(
            manifest=manifest, package_sha256="d" * 64, bundle_root=root,
            builtin=False, enabled=True, status="healthy", health="healthy",
        )
        source = tmp_path / "source.cityjson"
        source.write_bytes(b"source")
        payload = _source_adapter_package(
            tmp_path / "adapter.ddpkg", hashlib.sha256(b"source").hexdigest()
        )
        monkeypatch.setattr(_FakeMcpClient, "source_adapter_payload", payload)
        monkeypatch.setattr(_FakeMcpClient, "instances", [])
        monkeypatch.setattr(manager_module, "McpStdioClient", _FakeMcpClient)
        detection = manager.detect_asset_source_with_plugins(source)
        fixture = manager, manifest, source, detection
        yield fixture
    finally:
        manager.close()
        store.close()


# 功能：
#   验证超量 ZIP 目录在成员对象构造之前被拒绝，不以解压前检查冒充分配前检查。
# 输入：
#   adapter_fixture：独占测试管理器及转换器信息。
#   tmp_path：测试独占目录。
#   monkeypatch：拦截完整 ZIP 对象构造的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_plugin_budget_precedes_zipfile_allocation(adapter_fixture, tmp_path, monkeypatch):
    manager, _, _, _ = adapter_fixture
    path = tmp_path / "too-many.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(2001):
            archive.writestr(f"empty-{index}", b"")

    # 功能：
    #   将进入完整成员表分配阶段视为失败，检验预检的执行顺序。
    # 输入：
    #   args：原构造函数的位置参数。
    #   kwargs：原构造函数的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("ZIP allocation preceded directory budget")

    monkeypatch.setattr(manager_module.zipfile, "ZipFile", forbidden)
    with pytest.raises(PluginManagerError, match="INDEX_INVALID"):
        manager.import_bundle(path)
    assert not list(manager.store.plugins_root.glob(".staging-*"))


# 功能：
#   验证既有目标不会被转换结果覆盖，调用方路径冲突也不会将正常转换器错误隔离。
# 输入：
#   adapter_fixture：独占测试管理器、转换器清单、源文件与识别结果。
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_adapter_preserves_existing_output_and_plugin_health(adapter_fixture, tmp_path):
    manager, manifest, source, detection = adapter_fixture
    destination = tmp_path / "existing.ddpkg"
    destination.write_bytes(b"existing owner")
    with pytest.raises(PluginManagerError, match="DESTINATION_EXISTS"):
        manager.normalize_asset_source_with_plugin(
            source=source, detection=detection, destination=destination, expected_kind="map"
        )
    assert destination.read_bytes() == b"existing owner"
    assert manager.store.get_plugin(manifest.plugin_id)["status"] == "healthy"
    assert _FakeMcpClient.instances == []


# 功能：
#   验证转换已经成功但发布时目标被抢先创建，仍保留竞争方内容和正常插件状态。
# 输入：
#   adapter_fixture：独占测试管理器、转换器清单、源文件与识别结果。
#   tmp_path：测试独占目录。
#   monkeypatch：在硬链接发布前插入竞争写入的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_adapter_publication_race_keeps_other_output(adapter_fixture, tmp_path, monkeypatch):
    manager, manifest, source, detection = adapter_fixture
    destination = tmp_path / "result.ddpkg"
    original_link = asset_storage.os.link

    # 功能：
    #   仅在目标资产发布时创建竞争输出，其他硬链接调用维持原行为。
    # 输入：
    #   temporary：本次待发布的暂存文件。
    #   target：硬链接发布的目标路径。
    # 输出：
    #   result：底层硬链接调用的返回值。
    def competing_writer(temporary, target):
        if target == destination:
            target.write_bytes(b"other publisher")
        result = original_link(temporary, target)
        return result

    monkeypatch.setattr(asset_storage.os, "link", competing_writer)
    with pytest.raises(PluginManagerError):
        manager.normalize_asset_source_with_plugin(
            source=source, detection=detection, destination=destination, expected_kind="map"
        )
    assert destination.read_bytes() == b"other publisher"
    assert manager.store.get_plugin(manifest.plugin_id)["status"] == "healthy"
    assert not list(tmp_path.glob(".result.ddpkg-*.tmp"))
    assert manager.store.summarize_plugin_usage(manifest.plugin_id)["errors"] == 1


# 功能：
#   验证宿主错误不误隔离的改动没有放宽插件输出检查：真实的无效转换结果仍隔离。
# 输入：
#   adapter_fixture：独占测试管理器、转换器清单、源文件与识别结果。
#   tmp_path：测试独占目录。
#   monkeypatch：仅将转换器输出替换为无效包的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_invalid_adapter_output_still_quarantines_plugin(adapter_fixture, tmp_path, monkeypatch):
    manager, manifest, source, detection = adapter_fixture
    monkeypatch.setattr(_FakeMcpClient, "source_adapter_payload", b"not an asset package")
    destination = tmp_path / "invalid.ddpkg"
    with pytest.raises(PluginManagerError):
        manager.normalize_asset_source_with_plugin(
            source=source, detection=detection, destination=destination, expected_kind="map"
        )
    assert not destination.exists()
    assert manager.store.get_plugin(manifest.plugin_id)["status"] == "quarantined"


# 功能：
#   验证复制转换器输入时发生本地读取错误，不会被归因成插件失败并永久停用插件。
# 输入：
#   adapter_fixture：独占测试管理器、转换器清单、源文件与识别结果。
#   tmp_path：测试独占目录。
#   monkeypatch：仅令输入快照读取失败的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_local_input_failure_does_not_quarantine_adapter(adapter_fixture, tmp_path, monkeypatch):
    manager, manifest, source, detection = adapter_fixture
    original = manager_module.hash_plugin_file

    # 功能：
    #   对当前输入模拟读取失败，其他摘要计算保持原实现。
    # 输入：
    #   path：本次需要读取的文件。
    #   kwargs：原摘要函数的预算和目标参数。
    # 输出：
    #   digest：未被故障注入时原函数返回的摘要。
    def fail_input(path, **kwargs):
        if path == source:
            raise OSError("fixture input is unavailable")
        digest = original(path, **kwargs)
        return digest

    monkeypatch.setattr(manager_module, "hash_plugin_file", fail_input)
    with pytest.raises(PluginManagerError):
        manager.normalize_asset_source_with_plugin(
            source=source, detection=detection,
            destination=tmp_path / "result.ddpkg", expected_kind="map",
        )
    assert manager.store.get_plugin(manifest.plugin_id)["status"] == "healthy"
    assert _FakeMcpClient.instances == []
