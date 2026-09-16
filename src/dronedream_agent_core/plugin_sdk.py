"""Scaffolding, bounded packaging, signing, and a local plugin lifecycle check.

ZIP ordering/timestamps are stable; provenance/signing times intentionally make
separate builds distinct. Only the host's OS policy provides execution isolation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .asset_package_storage import publish_asset_file
from .plugin_contracts import PluginManifest
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .plugin_panels import validate_panel_document
from .plugin_trust import unsigned_manifest_sha256

_IGNORED_PARTS = frozenset({".git", ".venv", "build", "dist", "__pycache__", ".pytest_cache"})
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_SOURCE_FILES = 1_998  # Reserve two ZIP entries for the manifest and generated SBOM.
_MAX_SOURCE_ENTRIES = 10_000


# 功能：
#   复用安装端的路径检查，拒绝链接及重解析祖先，并保留源码工具原有的错误标识。
# 输入：
#   path：即将访问的源码、密钥或输出路径。
# 输出：
#   None：不返回业务数据。
def _check_plain_path(path: Path) -> None:
    try:
        check_plain_plugin_path(path)
    except ValueError as error:
        raise ValueError("PLUGIN_SOURCE_LINK_NOT_ALLOWED") from error


# 功能：
#   复用共享有界读取与身份复核，小文件按实际大小读取，源文件变化时不继续签名或打包。
# 输入：
#   path：待读取的普通源码或密钥描述文件。
#   limit：非负整数形式的最大允许字节数。
# 输出：
#   value：读取及身份校验通过的原始字节。
def _read_source_file(path: Path, *, limit: int) -> bytes:
    if type(limit) is not int or limit < 0:
        raise ValueError("PLUGIN_SOURCE_LIMIT_INVALID")
    try:
        value = read_plugin_file(path, limit=limit)
    except ValueError as error:
        issue = {
            "PLUGIN_FILE_TOO_LARGE_OR_INVALID": "PLUGIN_SOURCE_FILE_INVALID",
            "PLUGIN_FILE_LINK_FORBIDDEN": "PLUGIN_SOURCE_LINK_NOT_ALLOWED",
        }.get(str(error), "PLUGIN_SOURCE_FILE_CHANGED_OR_TOO_LARGE")
        raise ValueError(issue) from error
    return value


# 功能：
#   将受限 JSON 值写为新脚手架文档，使用独占文本创建而不截断既有文件。
# 输入：
#   path：待创建的文档路径。
#   value：可序列化为严格 JSON 的文档内容。
# 输出：
#   None：不返回业务数据。
def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, encode_json(value) + "\n")


# 功能：
#   独占创建 UTF-8、LF 换行的脚手架文本，拒绝链接路径和已存在的目标。
# 输入：
#   path：新文本文件的路径。
#   rendered：调用方准备好的文本，包括源码或构建模板。
# 输出：
#   None：不返回业务数据。
def _write_text(path: Path, rendered: str) -> None:
    _check_plain_path(path)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(rendered)


# 功能：
#   1. 校验插件身份后创建声明式 UI 或 MCP 开发脚手架，不覆盖非空源码目录。
#   2. MCP 示例仅回显参数，需开发者替换实现并构建程序，不提供生产飞行控制能力。
# 输入：
#   root：新脚手架目录或已有空目录。
#   plugin_id：符合插件命名规则的标识。
#   name：插件显示名称。
#   publisher：发行者名称。
#   kind：mcp 或 ui 类型选择。
# 输出：
#   root：已创建脚手架的目录路径。
def scaffold_plugin(
    root: Path,
    *,
    plugin_id: str,
    name: str,
    publisher: str,
    kind: Literal["mcp", "ui"] = "mcp",
) -> Path:
    if kind not in {"mcp", "ui"}:
        raise ValueError("PLUGIN_SCAFFOLD_KIND_INVALID")
    if (
        not isinstance(plugin_id, str)
        or re.fullmatch(r"[a-z][a-z0-9._-]{2,111}", plugin_id) is None
        or not isinstance(name, str)
        or not 1 <= len(name.strip()) <= 120
        or not isinstance(publisher, str)
        or not 1 <= len(publisher.strip()) <= 120
    ):
        raise ValueError("PLUGIN_SCAFFOLD_IDENTITY_INVALID")
    _check_plain_path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("PLUGIN_SCAFFOLD_DIRECTORY_NOT_EMPTY")
    root.mkdir(parents=True, exist_ok=True)
    capability_id = f"{plugin_id}.inspect"
    if kind == "ui":
        panel = {
            "schema_version": "dronedream.ui-panel.v1",
            "title": name,
            "sections": [
                {
                    "section_id": "status",
                    "title": "状态",
                    "widgets": [
                        {
                            "widget_id": "plugin-health",
                            "kind": "status",
                            "label": "健康",
                            "source": "plugin",
                            "path": "health",
                        }
                    ],
                    "actions": [{"action_id": "plugin.healthcheck", "label": "健康检查"}],
                }
            ],
        }
        _write_json(root / "ui" / "panel.json", panel)
        manifest: dict[str, Any] = {
            "plugin_id": plugin_id,
            "name": name,
            "version": "0.1.0",
            "description": f"{name} declarative panel.",
            "publisher": publisher,
            "runtime": {"kind": "ui-declarative"},
            "capabilities": [
                {
                    "capability_id": capability_id,
                    "kind": "ui-panel",
                    "name": name,
                    "description": f"Render {name} without executable UI code.",
                    "authority": "read",
                    "metadata": {"entrypoint": "ui/panel.json"},
                }
            ],
            "permissions": ["ui.panel"],
            "placement": {
                "category_id": "general",
                "category_label": "通用",
                "slot_id": "general.panels",
                "slot_label": "面板",
                "activation_mode": "multiple",
                "scope": "interface",
                "failure_mode": "isolate",
                "swap_policy": "anytime",
            },
        }
    else:
        server_source = (
            "from dronedream_plugin_sdk import McpPluginServer, ToolContext, ToolSpec\n\n"
            'INPUT = {"type": "object", "additionalProperties": False, '
            '"properties": {"text": {"type": "string"}}}\n'
            'OUTPUT = {"type": "object", "additionalProperties": False, '
            '"required": ["accepted"], "properties": {"accepted": {"type": "boolean"}, '
            '"echo": {"type": "string"}}}\n\n'
            "# 功能：\n#   仅回显输入作为开发示例，不承担生产控制能力。\n"
            "# 输入：\n#   value：通过 Schema 校验的参数。\n"
            "#   context：包含取消与进度接口的工具上下文。\n"
            "# 输出：\n#   result：参数回显及示例接收标记。\n"
            "def inspect(value: dict, context: ToolContext) -> dict:\n"
            '    context.progress(0.5, "inspecting")\n'
            '    result = {"accepted": True, "echo": str(value.get("text", ""))}\n'
            '    return result\n\n'
            f'McpPluginServer(name={name!r}, version="0.1.0", tools=[\n'
            f'    ToolSpec(name={capability_id!r}, description="Inspect a bounded input.",\n'
            "             input_schema=INPUT, output_schema=OUTPUT, handler=inspect)\n"
            "]).run()\n"
        )
        _write_text(root / "plugin_server.py", server_source)
        _write_text(
            root / "build.ps1",
            "$ErrorActionPreference='Stop'\n"
            "python -m PyInstaller --onefile --name plugin plugin_server.py "
            "--distpath bin --workpath build --specpath build\n",
        )
        manifest = {
            "plugin_id": plugin_id,
            "name": name,
            "version": "0.1.0",
            "description": f"{name} MCP tool.",
            "publisher": publisher,
            "runtime": {"kind": "mcp-stdio", "command": ["bin/plugin.exe"]},
            "capabilities": [
                {
                    "capability_id": capability_id,
                    "kind": "tool",
                    "name": name,
                    "description": "Inspect a bounded input.",
                    "authority": "read",
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"text": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["accepted"],
                        "properties": {
                            "accepted": {"type": "boolean"},
                            "echo": {"type": "string"},
                        },
                    },
                }
            ],
            "permissions": ["process.spawn", "mission.read"],
            "placement": {
                "category_id": "tools",
                "category_label": "工具",
                "slot_id": "tools.external",
                "slot_label": "外部工具",
                "activation_mode": "multiple",
                "scope": "mission",
                "failure_mode": "isolate",
                "swap_policy": "next-mission",
            },
        }
    _write_json(root / "plugin.json", manifest)
    _write_text(
        root / "README.md",
        f"# {name}\n\nValidate, package, and sandbox with `dronedream-plugin`.\n",
    )
    return root


# 功能：
#   1. 在条目数、文件数、单文件和总字节预算内收集普通源码文件，排除生成目录与链接。
#   2. 排除已知凭据文件名及调用方指定的签名密钥，但不将文件名过滤冒充通用秘密检测。
#   3. 返回稳定排序的内容快照，防止后续摘要与归档分别读取出不同版本的源文件。
# 输入：
#   root：插件源码根目录。
#   excluded：需要额外排除的已解析绝对文件路径集合。
# 输出：
#   files：按包内路径排序的原始字节字典。
def _package_files(root: Path, *, excluded: set[Path] | None = None) -> dict[str, bytes]:
    _check_plain_path(root)
    root = root.resolve()
    excluded = excluded or set()
    values: dict[str, bytes] = {}
    portable_names: set[str] = {"plugin.json", "sbom.cdx.json"}
    pending = [root]
    entries = total = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > _MAX_SOURCE_ENTRIES:
                    raise ValueError("PLUGIN_SOURCE_ENTRY_LIMIT")
                name = child.name.lower()
                if name in _IGNORED_PARTS:
                    continue
                path = Path(child.path)
                relative = path.relative_to(root)
                if (
                    len(relative.parts) > 32
                    or len(relative.as_posix().encode("utf-8")) > 512
                    or any(
                        part.endswith((" ", ".")) or ":" in part or "\\" in part
                        for part in relative.parts
                    )
                ):
                    raise ValueError("PLUGIN_SOURCE_PATH_INVALID")
                _check_plain_path(path)
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise ValueError("PLUGIN_SOURCE_FILE_INVALID")
                if path.resolve() in excluded or relative.as_posix() in {
                    "plugin.json",
                    "sbom.cdx.json",
                }:
                    continue
                if (
                    name == ".env"
                    or name.startswith(".env.")
                    or name == "publisher-key.json"
                    or name.endswith((".key", ".pem", ".pfx", ".p12", ".keystore"))
                ):
                    continue
                portable = relative.as_posix().casefold()
                if portable in portable_names:
                    raise ValueError("PLUGIN_SOURCE_CASE_COLLISION")
                portable_names.add(portable)
                if len(values) >= _MAX_SOURCE_FILES:
                    raise ValueError("PLUGIN_SOURCE_FILE_LIMIT")
                value = _read_source_file(
                    path, limit=min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total)
                )
                total += len(value)
                values[relative.as_posix()] = value
    files = dict(sorted(values.items()))
    return files


# 功能：
#   1. 独占生成 Ed25519 开发签名密钥文件，返回值仅包含公钥描述。
#   2. POSIX 文件权限限制为所有者读写；Windows 继承目录 ACL，此文件不是凭据库加密存储。
# 输入：
#   path：尚不存在的密钥文件路径。
#   key_id：发行者公钥标识。
#   publisher：密钥绑定的发行者名称。
# 输出：
#   public_descriptor：不包含私钥的公钥描述字典。
def generate_publisher_key(path: Path, *, key_id: str, publisher: str) -> dict[str, str]:
    if (
        not isinstance(key_id, str)
        or re.fullmatch(r"[a-z][a-z0-9._-]{2,119}", key_id) is None
        or not isinstance(publisher, str)
        or not 1 <= len(publisher.strip()) <= 120
    ):
        raise ValueError("PLUGIN_PUBLISHER_IDENTITY_INVALID")
    _check_plain_path(path)
    if path.exists():
        raise ValueError("PLUGIN_PUBLISHER_KEY_EXISTS")
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    public_bytes = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    payload = {
        "schema_version": "dronedream.publisher-key.v1",
        "key_id": key_id,
        "publisher": publisher,
        "private_key_base64": base64.b64encode(private_bytes).decode("ascii"),
        "public_key_base64": base64.b64encode(public_bytes).decode("ascii"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 最终创建仍用 O_EXCL，不能只依赖前面的 exists 检查来保护已存在密钥。
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("PLUGIN_PUBLISHER_KEY_EXISTS") from error
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
        target.write(encode_json(payload) + "\n")
        target.flush()
        os.fsync(target.fileno())
    public_descriptor = {
        key: value for key, value in payload.items() if key != "private_key_base64"
    }
    return public_descriptor


# 功能：
#   有界读取当前密钥描述，严格检查字段、Base64 和私公钥配对，不接受任意类型转换。
# 输入：
#   path：本地签名密钥描述文件。
# 输出：
#   signing_identity：公钥标识、发行者名称及私钥对象组成的元组。
def _load_signing_key(path: Path) -> tuple[str, str, Ed25519PrivateKey]:
    value = decode_json(_read_source_file(path, limit=16_384), limit=16_384)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "dronedream.publisher-key.v1"
        or not isinstance(value.get("key_id"), str)
        or re.fullmatch(r"[a-z][a-z0-9._-]{2,119}", value["key_id"]) is None
        or not isinstance(value.get("publisher"), str)
        or not 1 <= len(value["publisher"].strip()) <= 120
        or not isinstance(value.get("private_key_base64"), str)
    ):
        raise ValueError("PLUGIN_SIGNING_KEY_INVALID")
    private = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(value["private_key_base64"], validate=True)
    )
    public = base64.b64encode(
        private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    if value.get("public_key_base64") != public:
        raise ValueError("PLUGIN_SIGNING_KEY_PAIR_MISMATCH")
    signing_identity = value["key_id"], value["publisher"], private
    return signing_identity


# 功能：
#   1. 冻结文件内容，生成清单和 SBOM，校验面板并可选签署清单摘要。
#   2. 在独占描述符中完成 ZIP，复核暂存身份后通过共享无覆盖发布器交付输出。
#   3. 清理仅针对本次仍持有身份的暂存文件，不删除其他写入者后来替换的内容。
# 输入：
#   root：插件源码目录。
#   output：源码目录之外、尚不存在的目标 ZIP 路径。
#   signing_key：可选的本地签名密钥文件。
# 输出：
#   report：插件身份、实际输出路径、包摘要、签名状态和负载文件数量组成的报告。
def build_plugin_bundle(
    root: Path, output: Path, *, signing_key: Path | None = None
) -> dict[str, object]:
    _check_plain_path(root)
    _check_plain_path(output)
    root = root.resolve()
    output = output.absolute()
    if output.resolve().is_relative_to(root):
        raise ValueError("PLUGIN_OUTPUT_MUST_BE_OUTSIDE_SOURCE")
    if output.exists():
        raise ValueError("PLUGIN_OUTPUT_EXISTS")
    if signing_key is not None and output.resolve() == signing_key.resolve():
        raise ValueError("PLUGIN_OUTPUT_IS_SIGNING_KEY")
    raw = decode_json(_read_source_file(root / "plugin.json", limit=2 * 1024 * 1024))
    if not isinstance(raw, dict):
        raise ValueError("PLUGIN_MANIFEST_INVALID")
    files = _package_files(root, excluded={signing_key.resolve()} if signing_key else None)
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": raw.get("name"),
                "version": raw.get("version"),
            }
        },
        "components": [],
    }
    files["sbom.cdx.json"] = (
        json.dumps(sbom, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    raw["file_sha256"] = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    provenance = dict(raw.get("provenance", {}))
    provenance.update(
        {
            "build_system": "dronedream-plugin-sdk",
            "build_timestamp": datetime.now(UTC).isoformat(),
            "sbom_sha256": raw["file_sha256"]["sbom.cdx.json"],
        }
    )
    raw["provenance"] = provenance
    raw.pop("signature", None)
    manifest = PluginManifest.model_validate(raw)
    if manifest.runtime.kind == "ui-declarative":
        panel_entrypoints = [
            item.metadata.get("entrypoint")
            for item in manifest.capabilities
            if item.kind == "ui-panel"
        ]
        for entrypoint in panel_entrypoints:
            if isinstance(entrypoint, str):
                validate_panel_document(decode_json(files[entrypoint], limit=256_000))
    if signing_key is not None:
        key_id, publisher, private = _load_signing_key(signing_key)
        if publisher != manifest.publisher:
            raise ValueError("PLUGIN_SIGNING_PUBLISHER_MISMATCH")
        digest = unsigned_manifest_sha256(manifest)
        raw["signature"] = {
            "algorithm": "ed25519",
            "publisher_key_id": key_id,
            "signed_manifest_sha256": digest,
            "signature_base64": base64.b64encode(private.sign(bytes.fromhex(digest))).decode(),
            "signed_at": datetime.now(UTC).isoformat(),
        }
        manifest = PluginManifest.model_validate(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {"plugin.json": (manifest.model_dump_json(indent=2) + "\n").encode(), **files}
    if sum(map(len, members.values())) > _MAX_TOTAL_BYTES:
        raise ValueError("PLUGIN_PACKAGE_SIZE_LIMIT")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".plugin-build-", suffix=".zip", dir=output.parent
    )
    temporary = Path(temporary_name)
    identity = None
    try:
        try:
            identity = os.fstat(descriptor)
            target_context = os.fdopen(descriptor, "w+b")
        except BaseException:
            # fdopen 未成功返回时，描述符尚未完成交接；不能仅依赖流上下文负责关闭。
            with suppress(OSError):
                if identity is None or os.path.samestat(identity, os.fstat(descriptor)):
                    os.close(descriptor)
            raise
        with target_context as target:
            with zipfile.ZipFile(
                target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as bundle:
                for name, value in members.items():
                    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    bundle.writestr(info, value)
            target.flush()
            os.fsync(target.fileno())
            target.seek(0)
            digest = hashlib.file_digest(target, "sha256").hexdigest()
        _check_plain_path(temporary)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("PLUGIN_PACKAGE_STAGING_CHANGED")
        # 复用身份与摘要校验发布器，不保留另一条仅按暂存名称链接的旧发布路径。
        publish_asset_file(
            temporary, output, expected_sha256=digest, limit=576 * 1024 * 1024
        )
    finally:
        if identity is not None:
            with suppress(OSError, ValueError):
                _check_plain_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()
    report = {
        "plugin_id": manifest.plugin_id,
        "version": manifest.version,
        "output": str(output.resolve()),
        "package_sha256": digest,
        "signed": manifest.signature is not None,
        "files": len(files),
    }
    return report


# 功能：
#   校验源码清单与面板语义并报告缺失的可执行入口，不执行源码或自动构建程序。
# 输入：
#   root：待检查的源码目录。
# 输出：
#   report：插件身份、能力与文件数量、打包准备状态及缺失入口列表。
def validate_plugin_source(root: Path) -> dict[str, object]:
    manifest = PluginManifest.model_validate(
        decode_json(_read_source_file(root / "plugin.json", limit=2 * 1024 * 1024))
    )
    files = _package_files(root)
    for capability in manifest.capabilities:
        if capability.kind == "ui-panel":
            entrypoint = capability.metadata.get("entrypoint")
            if not isinstance(entrypoint, str) or entrypoint not in files:
                raise ValueError("PLUGIN_PANEL_ENTRYPOINT_MISSING")
            validate_panel_document(decode_json(files[entrypoint], limit=256_000))
    missing_runtime: list[str] = []
    if (
        manifest.runtime.kind == "mcp-stdio"
        and manifest.runtime.command
        and manifest.runtime.command[0] not in files
    ):
        missing_runtime.append(manifest.runtime.command[0])
    report = {
        "plugin_id": manifest.plugin_id,
        "version": manifest.version,
        "runtime_kind": manifest.runtime.kind,
        "capabilities": len(manifest.capabilities),
        "files": len(files),
        "ready_to_package": not missing_runtime,
        "missing_runtime": missing_runtime,
    }
    return report


# 功能：
#   1. 在一次性状态库中验证导入、批准、启用、健康检查和停用的生命周期，并有序回收资源。
#   2. 此操作可能执行包内代码，受管理器的操作系统策略约束，不等同于静态检查或飞行认证。
# 输入：
#   bundle：调用方明确选择执行生命周期检查的本地插件包。
# 输出：
#   report：插件身份、健康和隔离状态及生命周期事件数量。
def sandbox_plugin_bundle(bundle: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="dronedream-plugin-sandbox-") as temporary:
        store = AppStore(Path(temporary) / "store")
        try:
            manager = PluginManager(store)
            try:
                imported = manager.import_bundle(bundle)
                plugin_id = str(imported["plugin_id"])
                manager.approve_local_package(plugin_id)
                manager.enable(plugin_id)
                checked = manager.healthcheck(plugin_id)
                manager.disable(plugin_id)
                report = {
                    "plugin_id": plugin_id,
                    "version": imported["version"],
                    "health": checked["health"],
                    "quarantined": checked["status"] == "quarantined",
                    "lifecycle_events": len(store.list_plugin_events(plugin_id)),
                }
                return report
            finally:
                manager.close()
        finally:
            store.close()
