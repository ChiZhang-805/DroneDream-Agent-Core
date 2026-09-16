from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.plugin_marketplace import (
    PluginMarketplaceError,
    PluginMarketplaceService,
)
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.plugin_contracts import (
    PluginMarketplaceIndex,
    PluginMarketplaceSource,
)


# 功能：
#   创建仅含声明式界面的无害 ZIP 夹具，使用真实成员字节及对应摘要。
# 输入：
#   path：本次测试拥有的目标归档路径。
# 输出：
#   path：已写入测试插件的归档路径。
def _bundle(path: Path) -> Path:
    panel = b'{"title":"Route audit"}\n'
    manifest = {
        "schema_version": "dronedream.plugin-manifest.v1",
        "plugin_id": "example.route-audit",
        "name": "Route Audit",
        "version": "1.0.0",
        "description": "Inspect a prepared route without actuator access.",
        "publisher": "Example",
        "runtime": {"kind": "ui-declarative"},
        "capabilities": [
            {
                "capability_id": "example.route-audit.panel",
                "kind": "ui-panel",
                "name": "Route Audit",
                "description": "Show route evidence.",
                "authority": "read",
                "metadata": {"entrypoint": "ui/panel.json"},
            }
        ],
        "permissions": ["mission.read", "ui.panel"],
        "file_sha256": {"ui/panel.json": hashlib.sha256(panel).hexdigest()},
    }
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("plugin.json", json.dumps(manifest))
        bundle.writestr("ui/panel.json", panel)
    return path


# 功能：
#   构造真实本地存储和安装器，并使测试目录引用同一份归档的摘要，不进行网络请求。
# 输入：
#   tmp_path：本次测试拥有的数据根目录。
# 输出：
#   result：市场服务、目录 JSON 字节和该目录指定的归档字节。
def _marketplace(tmp_path: Path) -> tuple[PluginMarketplaceService, bytes, bytes]:
    store = AppStore(tmp_path / "store")
    manager = PluginManager(store)
    service = PluginMarketplaceService(store, manager)
    source = PluginMarketplaceSource(
        source_id="official-lab",
        name="Official lab",
        index_url="https://plugins.example.test/index.json",
    )
    service.replace_sources([source])
    archive_path = _bundle(tmp_path / "route-audit.zip")
    archive = archive_path.read_bytes()
    index = (
        PluginMarketplaceIndex.model_validate(
            {
                "generated_at": datetime.now(UTC),
                "entries": [
                    {
                        "plugin_id": "example.route-audit",
                        "version": "1.0.0",
                        "name": "Route Audit",
                        "description": "Inspect a route.",
                        "publisher": "Example",
                        "archive_url": "https://plugins.example.test/route-audit.zip",
                        "archive_sha256": hashlib.sha256(archive).hexdigest(),
                        "category_id": "planning",
                    }
                ],
            }
        )
        .model_dump_json()
        .encode()
    )
    result = service, index, archive
    return result


# 功能：
#   验证离线目录与下载替身可通过真实安装管理器完成一致身份的安装。
# 输入：
#   tmp_path：隔离插件存储目录。
#   monkeypatch：替换网络下载的测试工具。
# 输出：
#   None：不返回业务数据。
def test_marketplace_catalog_and_install_are_digest_pinned(tmp_path, monkeypatch):
    service, index, archive = _marketplace(tmp_path)

    # 功能：
    #   依据请求地址返回目录或归档，不访问网络。
    # 输入：
    #   url：目录或归档地址。
    #   kwargs：下载字节及时间预算。
    # 输出：
    #   payload：固定的测试响应字节。
    def download(url: str, **kwargs) -> bytes:
        payload = index if url.endswith("index.json") else archive
        return payload

    monkeypatch.setattr(service, "_download", download)
    catalog = service.catalog()
    assert catalog["entries"][0]["source_id"] == "official-lab"

    installed = service.install(
        source_id="official-lab",
        plugin_id="example.route-audit",
        version="1.0.0",
    )
    assert installed["plugin_id"] == "example.route-audit"


# 功能：
#   验证包名称和版本匹配不能弥补归档内容与目录摘要不一致。
# 输入：
#   tmp_path：隔离插件存储目录。
#   monkeypatch：注入变化归档的测试工具。
# 输出：
#   None：不返回业务数据。
def test_marketplace_rejects_archive_hash_drift(tmp_path, monkeypatch):
    service, index, _archive = _marketplace(tmp_path)

    # 功能：
    #   保持目录不变，只将下载的归档替换成不同字节。
    # 输入：
    #   url：请求的目录或归档地址。
    #   kwargs：调用方给定的下载预算。
    # 输出：
    #   payload：真实目录或被篡改的归档字节。
    def download(url: str, **kwargs) -> bytes:
        payload = index if url.endswith("index.json") else b"tampered"
        return payload

    monkeypatch.setattr(service, "_download", download)
    with pytest.raises(PluginMarketplaceError, match="ARCHIVE_HASH_MISMATCH"):
        service.install(
            source_id="official-lab",
            plugin_id="example.route-audit",
            version="1.0.0",
        )


# 功能：
#   验证 HTTP 地址无法成为可保存的插件目录来源。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_marketplace_sources_refuse_insecure_transport():
    with pytest.raises(ValueError, match="HTTPS"):
        PluginMarketplaceSource(
            source_id="insecure-source",
            name="Insecure",
            index_url="http://plugins.example.test/index.json",
        )
