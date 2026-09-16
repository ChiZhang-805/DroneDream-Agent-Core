"""Keep remote catalog, staged bytes and the real installer bound to one selection."""

import os

import pytest
from test_plugin_marketplace import _marketplace

from dronedream_agent_app import plugin_marketplace as module
from dronedream_agent_app.plugin_manager import PluginManagerError


# 功能：
#   验证目录展示和安装两条路径都拒绝重复 JSON 字段，不默默选择最后一次值。
# 输入：
#   tmp_path：构造本地来源和无害归档的测试目录。
#   monkeypatch：替换目录下载的测试工具。
# 输出：
#   None：不返回业务数据。
def test_catalog_and_install_reject_duplicate_json(tmp_path, monkeypatch):
    service, index, _archive = _marketplace(tmp_path)
    duplicate = index.replace(b'"entries":', b'"entries":[],"entries":', 1)
    monkeypatch.setattr(service, "_download", lambda *args, **kwargs: duplicate)
    catalog = service.catalog()
    assert catalog["entries"] == []
    assert catalog["errors"] == [
        {"source_id": "official-lab", "issue_code": "PLUGIN_MARKETPLACE_INDEX_INVALID"}
    ]
    with pytest.raises(module.PluginMarketplaceError, match="INDEX_INVALID"):
        service.install(source_id="official-lab", plugin_id="example.route-audit", version="1.0.0")


# 功能：
#   验证安装边界继续携带目录摘要和包身份，且暂存路径被另一写入者替换后不误删。
# 输入：
#   tmp_path：构造本地数据和替换文件的测试目录。
#   monkeypatch：替换下载和安装入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_install_binds_final_bytes_and_preserves_replacement(tmp_path, monkeypatch):
    service, index, archive = _marketplace(tmp_path)
    monkeypatch.setattr(
        service, "_download", lambda url, **kwargs: index if url.endswith("index.json") else archive
    )
    replacement = tmp_path / "other-writer.zip"
    replacement.write_bytes(archive + b"different archive bytes")
    original_import = service.manager.import_bundle
    paths = []

    # 功能：
    #   在最终导入前替换暂存材料，再交给真实管理器校验，预期因摘要变化而失败。
    # 输入：
    #   path：市场服务创建的暂存归档。
    #   kwargs：必须传递到管理器的预期摘要和身份。
    # 输出：
    #   installed：真实管理器返回的安装记录。
    def replace_before_import(path, **kwargs):
        paths.append(path)
        assert kwargs.get("expected_identity") == ("example.route-audit", "1.0.0")
        assert kwargs.get("expected_sha256") is not None
        os.replace(replacement, path)
        installed = original_import(path, **kwargs)
        return installed

    monkeypatch.setattr(service.manager, "import_bundle", replace_before_import)
    with pytest.raises(PluginManagerError, match="ARCHIVE_HASH_MISMATCH"):
        service.install(source_id="official-lab", plugin_id="example.route-audit", version="1.0.0")
    assert paths[0].read_bytes() == archive + b"different archive bytes"


# 功能：
#   验证 ZIP 索引预算在普通 ZipFile 分配成员对象之前执行。
# 输入：
#   tmp_path：无害目录与归档夹具所属目录。
#   monkeypatch：注入索引超限并阻止普通解包器启动的测试工具。
# 输出：
#   None：不返回业务数据。
def test_install_checks_zip_index_before_member_allocation(tmp_path, monkeypatch):
    service, index, archive = _marketplace(tmp_path)
    monkeypatch.setattr(
        service, "_download", lambda url, **kwargs: index if url.endswith("index.json") else archive
    )

    # 功能：
    #   模拟 ZIP 索引超过允许预算，不分配大量真实成员。
    # 输入：
    #   source：与普通解包器共用的归档字节流。
    #   maximum_members：允许的成员数上限。
    # 输出：
    #   None：不返回业务数据。
    def rejected_index(source, *, maximum_members):
        assert maximum_members == 20_000
        raise ValueError("ZIP_INDEX_LIMIT_EXCEEDED")

    monkeypatch.setattr(module, "validate_zip_index", rejected_index, raising=False)
    monkeypatch.setattr(
        module.zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("late guard")
    )
    with pytest.raises(module.PluginMarketplaceError, match="ARCHIVE_INVALID"):
        service.install(source_id="official-lab", plugin_id="example.route-audit", version="1.0.0")
