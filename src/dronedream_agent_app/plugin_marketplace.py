"""HTTPS-only plugin catalog discovery and digest-pinned installation."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dronedream_agent_core.plugin_contracts import (
    PluginManifest,
    PluginMarketplaceIndex,
    PluginMarketplaceSource,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.zip_index import validate_zip_index
from dronedream_plugin_sdk.protocol import decode_json

from .plugin_manager import PluginManager, PluginManagerError
from .storage import AppStore


class PluginMarketplaceError(RuntimeError):
    """A catalog or archive failed a bounded validation step."""


# 功能：
#   在发送请求或执行跳转前校验 HTTPS 地址；地址合法不代表插件发布者可信。
# 输入：
#   url：准备使用的插件目录或归档地址。
# 输出：
#   None：不返回业务数据。
def _https_url(url: str) -> None:
    try:
        if (
            not isinstance(url, str)
            or len(url) > 4_096
            or any(ord(ch) <= 32 or ch.isspace() for ch in url)
        ):
            raise ValueError("invalid URL")
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port == 0
            or "\\" in parsed.netloc
        ):
            raise ValueError("HTTPS URL required")
    except ValueError as error:
        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_HTTPS_REQUIRED") from error


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    """Permit HTTPS CDN redirects, never an HTTP downgrade followed by a late check."""

    # 功能：
    #   只接受 HTTPS 跳转，拒绝时关闭原响应体，不等待 urllib 的成功跳转清理。
    # 输入：
    #   self：当前跳转处理器。
    #   req：原始请求。
    #   fp：原响应体。
    #   code：重定向状态码。
    #   msg：状态说明。
    #   headers：响应头。
    #   newurl：跳转后的目标地址。
    # 输出：
    #   redirected：新请求或不支持跳转时的 None。
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _https_url(newurl)
        except PluginMarketplaceError as error:
            # urllib normally closes the redirect body after this hook returns.
            # A rejected destination raises here instead, so we own that cleanup.
            if fp is not None:
                fp.close()
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_REDIRECT_DENIED") from error
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        return redirected


class PluginMarketplaceService:
    # 功能：
    #   绑定已有应用存储与安装管理器，不创建另一套插件状态或信任判断。
    # 输入：
    #   self：插件市场服务。
    #   store：应用配置与暂存目录所属存储。
    #   manager：负责安装及信任决策的管理器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, store: AppStore, manager: PluginManager) -> None:
        self.store = store
        self.manager = manager

    # 功能：
    #   重新验证持久化来源的数量、类型、唯一标识和 HTTPS 地址，不依赖旧的 UI 校验。
    # 输入：
    #   self：持有应用存储的市场服务。
    # 输出：
    #   sources：通过当前规则验证的来源快照。
    def sources(self) -> list[PluginMarketplaceSource]:
        raw = self.store.get_settings().get("plugin_marketplace_sources", [])
        if not isinstance(raw, list) or len(raw) > 32:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCES_INVALID")
        try:
            sources = [PluginMarketplaceSource.model_validate(item, strict=True) for item in raw]
            for source in sources:
                _https_url(source.index_url)
        except ValidationError as error:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCES_INVALID") from error
        if len({item.source_id for item in sources}) != len(sources):
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCE_DUPLICATE")
        return sources

    # 功能：
    #   分离并验证来源数据，在全部校验通过后一次替换保存的来源列表。
    # 输入：
    #   self：持有应用存储的市场服务。
    #   sources：准备保存的来源对象列表。
    # 输出：
    #   payload：实际提交给存储的 JSON 兼容来源列表。
    def replace_sources(self, sources: list[PluginMarketplaceSource]) -> list[dict[str, Any]]:
        if not isinstance(sources, list) or len(sources) > 32:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCES_INVALID")
        try:
            sources = [
                PluginMarketplaceSource.model_validate(item.model_dump(), strict=True)
                for item in sources
            ]
            for source in sources:
                _https_url(source.index_url)
        except (AttributeError, TypeError, ValidationError) as error:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCES_INVALID") from error
        if len(sources) > 32 or len({item.source_id for item in sources}) != len(sources):
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCES_INVALID")
        payload = [item.model_dump(mode="json") for item in sources]
        self.store.patch_settings({"plugin_marketplace_sources": payload})
        return payload

    # 功能：
    #   按实际字节限制下载内容，并拒绝整体预算之后才读到的响应。
    #   套接字超时和读取后的时限检查不等同于 DNS 或阻塞系统调用的硬取消。
    # 输入：
    #   url：来源配置或索引提供的 HTTPS 地址。
    #   limit：允许接收的最大字节数。
    #   timeout_seconds：下载的总秒数预算。
    # 输出：
    #   result：已完整读取且符合长度声明的响应字节。
    @staticmethod
    def _download(url: str, *, limit: int, timeout_seconds: float) -> bytes:
        _https_url(url)
        if type(limit) is not int or not 0 < limit <= 512 * 1024 * 1024:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_LIMIT_INVALID")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_TIMEOUT_INVALID")
        deadline = time.monotonic() + timeout_seconds
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json, application/zip",
                "User-Agent": "DroneDream-AGENT/1.0.0",
            },
        )
        try:
            opener = urllib.request.build_opener(_HttpsRedirect())
            with opener.open(request, timeout=timeout_seconds) as response:
                _https_url(response.geturl())
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    if not declared.isascii() or not declared.isdecimal():
                        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_LENGTH_INVALID")
                    if len(declared) > 16 or int(declared) > limit:
                        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_RESPONSE_TOO_LARGE")
                payload = bytearray()
                while True:
                    if time.monotonic() >= deadline:
                        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_TIMEOUT")
                    chunk = response.read(min(64 * 1024, limit + 1 - len(payload)))
                    if time.monotonic() >= deadline:
                        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_TIMEOUT")
                    if not chunk:
                        break
                    payload.extend(chunk)
                    if len(payload) > limit:
                        raise PluginMarketplaceError("PLUGIN_MARKETPLACE_RESPONSE_TOO_LARGE")
                if declared is not None and len(payload) != int(declared):
                    raise PluginMarketplaceError("PLUGIN_MARKETPLACE_LENGTH_MISMATCH")
        except urllib.error.HTTPError as error:
            error.close()
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_FETCH_FAILED") from error
        except (OSError, ValueError, urllib.error.URLError) as error:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_FETCH_FAILED") from error
        if len(payload) > limit:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_RESPONSE_TOO_LARGE")
        result = bytes(payload)
        return result

    # 功能：
    #   聚合一次来源快照中的启用目录，单源失败不阻断其他来源，也不暴露原始错误正文。
    # 输入：
    #   self：包含来源配置与下载能力的市场服务。
    # 输出：
    #   result：来源、排序后的插件条目和分源错误代码。
    def catalog(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        sources = self.sources()
        for source in sources:
            if not source.enabled:
                continue
            try:
                raw = self._download(source.index_url, limit=2 * 1024 * 1024, timeout_seconds=15)
                index = PluginMarketplaceIndex.model_validate(
                    decode_json(raw, limit=2 * 1024 * 1024)
                )
                entries.extend(
                    {
                        **entry.model_dump(mode="json"),
                        "source_id": source.source_id,
                        "source_name": source.name,
                    }
                    for entry in index.entries
                )
            except (PluginMarketplaceError, ValidationError, ValueError) as error:
                issue = (
                    str(error)
                    if isinstance(error, PluginMarketplaceError)
                    else ("PLUGIN_MARKETPLACE_INDEX_INVALID")
                )
                errors.append({"source_id": source.source_id, "issue_code": issue})
        entries.sort(
            key=lambda item: (
                str(item["category_id"]),
                str(item["name"]),
                str(item["version"]),
            )
        )
        result = {
            "sources": [item.model_dump(mode="json") for item in sources],
            "entries": entries,
            "errors": errors,
        }
        return result

    # 功能：
    #   1. 重新读取选定来源并核对归档摘要、包身份及有界 ZIP 索引和声明。
    #   2. 将预期摘要和身份继续传给实际安装器，防止暂存文件变化绕过下载验证。
    #   3. 只清理本次仍拥有的暂存文件；摘要一致不能代替安装器的信任与权限判断。
    # 输入：
    #   self：持有来源、暂存空间和安装管理器的市场服务。
    #   source_id：用户选择的启用来源标识。
    #   plugin_id：准备安装的插件标识。
    #   version：该插件的精确包版本。
    # 输出：
    #   installed：实际安装管理器返回的安装记录。
    def install(self, *, source_id: str, plugin_id: str, version: str) -> dict[str, Any]:
        source = next(
            (item for item in self.sources() if item.source_id == source_id and item.enabled),
            None,
        )
        if source is None:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_SOURCE_NOT_FOUND")
        raw_index = self._download(source.index_url, limit=2 * 1024 * 1024, timeout_seconds=15)
        try:
            index = PluginMarketplaceIndex.model_validate(
                decode_json(raw_index, limit=2 * 1024 * 1024)
            )
        except ValueError as error:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_INDEX_INVALID") from error
        entry = next(
            (
                item
                for item in index.entries
                if item.plugin_id == plugin_id and item.version == version
            ),
            None,
        )
        if entry is None:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_ENTRY_NOT_FOUND")
        archive = self._download(entry.archive_url, limit=512 * 1024 * 1024, timeout_seconds=60)
        if hashlib.sha256(archive).hexdigest() != entry.archive_sha256:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_ARCHIVE_HASH_MISMATCH")
        try:
            frozen = io.BytesIO(archive)
            validate_zip_index(frozen, maximum_members=20_000)
            with zipfile.ZipFile(frozen) as bundle:
                if sum(info.filename == "plugin.json" for info in bundle.infolist()) != 1:
                    raise PluginMarketplaceError("PLUGIN_MARKETPLACE_MANIFEST_AMBIGUOUS")
                manifest_info = bundle.getinfo("plugin.json")
                if manifest_info.file_size > 2 * 1024 * 1024:
                    raise PluginMarketplaceError("PLUGIN_MANIFEST_TOO_LARGE")
                with bundle.open(manifest_info) as stream:
                    manifest_bytes = stream.read(2 * 1024 * 1024 + 1)
                if len(manifest_bytes) > 2 * 1024 * 1024:
                    raise PluginMarketplaceError("PLUGIN_MANIFEST_TOO_LARGE")
                manifest = PluginManifest.model_validate(
                    decode_json(manifest_bytes, limit=2 * 1024 * 1024)
                )
        except (KeyError, OSError, ValidationError, ValueError, zipfile.BadZipFile) as error:
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_ARCHIVE_INVALID") from error
        if (manifest.plugin_id, manifest.version) != (plugin_id, version):
            raise PluginMarketplaceError("PLUGIN_MARKETPLACE_IDENTITY_MISMATCH")
        check_plain_plugin_path(self.store.plugin_staging_root)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="dd-marketplace-", suffix=".zip", dir=self.store.plugin_staging_root
        )
        temporary = Path(temporary_name)
        owned = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(archive)
                output.flush()
                os.fsync(output.fileno())
            installed = self.manager.import_bundle(
                temporary,
                expected_sha256=entry.archive_sha256,
                expected_identity=(plugin_id, version),
            )
        except PluginManagerError:
            raise
        finally:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(owned, temporary.stat()):
                    temporary.unlink()
        return installed
