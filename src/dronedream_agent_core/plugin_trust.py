"""Ed25519 publisher trust, exact-package local approval, and revocation checks."""

from __future__ import annotations

import base64
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .hashing import sha256_json
from .plugin_contracts import PluginManifest
from .plugin_files import check_plain_plugin_path, read_plugin_file


class TrustStoreError(RuntimeError):
    """The trust store or a requested trust transition is invalid."""


class PluginTrustDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "local-approved", "unverified", "revoked"]
    issue_codes: list[str] = Field(default_factory=list)
    publisher_key_id: str | None = None
    manifest_sha256: str
    package_sha256: str


# 功能：
#   对完整清单计算稳定摘要，但排除签名信封，避免签名内容反过来改变待签摘要。
# 输入：
#   manifest：已通过结构校验的插件清单。
# 输出：
#   checksum：未含签名信封的清单 SHA-256 摘要。
def unsigned_manifest_sha256(manifest: PluginManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"signature"})
    checksum = sha256_json(payload)
    return checksum


class PluginTrustStore:
    """Small durable trust store; approvals are bound to immutable package hashes."""

    # 功能：
    #   在跨进程锁内创建或校验信任库，避免第二个打开者覆盖已经保存的授权与撤销记录。
    # 输入：
    #   path：本地信任库文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path) -> None:
        self.path = path.absolute()
        self._thread_lock = threading.RLock()
        self._check_paths()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.path.exists():
                self._write(self._empty())
            else:
                self._load()

    # 功能：
    #   在创建目录或访问权限文件前拒绝链接、重解析点和非普通的信任库或锁文件。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _check_paths(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".lock")
        for path in (self.path, lock_path, *self.path.parents):
            if path.exists() or path.is_symlink():
                metadata = path.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & 0x400
                ):
                    raise TrustStoreError("PLUGIN_TRUST_STORE_LINK_INVALID")
                if path in (self.path, lock_path) and not stat.S_ISREG(metadata.st_mode):
                    raise TrustStoreError("PLUGIN_TRUST_STORE_INVALID")

    # 功能：
    #   1. 用线程锁与持久侧车文件锁串行化完整的读取、修改、写入事务。
    #   2. 获锁后复核路径身份，最迟等待五秒；退出时解锁但不删除锁文件。
    # 输入：
    #   无。
    # 输出：
    #   None：上下文进入时不提供业务数据。
    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            lock_path = self.path.with_name(self.path.name + ".lock")
            self._check_paths()
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                lock_context = os.fdopen(descriptor, "r+b", buffering=0)
            except BaseException:
                # fdopen 成功才接管描述符；交接失败时必须由创建方回收。
                with suppress(OSError):
                    os.close(descriptor)
                raise
            with lock_context as lock:
                if os.fstat(lock.fileno()).st_size == 0:
                    lock.write(b"\0")
                deadline = time.monotonic() + 5.0
                while True:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            lock.seek(0)
                            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as error:
                        if time.monotonic() >= deadline:
                            raise TrustStoreError("PLUGIN_TRUST_STORE_BUSY") from error
                        time.sleep(0.01)
                try:
                    self._check_paths()
                    if not os.path.samestat(os.fstat(lock.fileno()), lock_path.stat()):
                        raise TrustStoreError("PLUGIN_TRUST_LOCK_CHANGED")
                    yield
                finally:
                    # 侧车必须持久保留，否则并发写入者可能分别锁住不同的 inode。
                    if os.name == "nt":
                        lock.seek(0)
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # 功能：
    #   构建没有任何发布者信任或本地授权的新库，不自动赋予插件执行权限。
    # 输入：
    #   无。
    # 输出：
    #   value：空信任库的完整结构。
    @staticmethod
    def _empty() -> dict[str, Any]:
        value = {
            "schema_version": "dronedream.plugin-trust-store.v1",
            "publishers": {},
            "revoked_publisher_keys": [],
            "revoked_packages": [],
            "revoked_plugins": [],
            "local_approvals": {},
        }
        return value

    # 功能：
    #   读取至多两 MiB 的普通文件并检查读取身份与权限容器结构，损坏时拒绝加载。
    # 输入：
    #   无。
    # 输出：
    #   value：通过校验的单次信任快照。
    def _load(self) -> dict[str, Any]:
        try:
            value = decode_json(read_plugin_file(self.path, limit=2 * 1024 * 1024))
        except (OSError, ValueError) as error:
            raise TrustStoreError("PLUGIN_TRUST_STORE_INVALID") from error
        if not isinstance(value, dict) or value.get("schema_version") != (
            "dronedream.plugin-trust-store.v1"
        ):
            raise TrustStoreError("PLUGIN_TRUST_STORE_INVALID")
        for field in ("publishers", "local_approvals"):
            if not isinstance(value.get(field), dict) or any(
                not isinstance(entry, dict) for entry in value[field].values()
            ):
                raise TrustStoreError("PLUGIN_TRUST_STORE_INVALID")
        for field in ("revoked_publisher_keys", "revoked_packages", "revoked_plugins"):
            if not isinstance(value.get(field), list) or any(
                not isinstance(entry, str) for entry in value[field]
            ):
                raise TrustStoreError("PLUGIN_TRUST_STORE_INVALID")
        return value

    # 功能：
    #   1. 在调用方已持有的信任库锁内刷盘并原子替换完整快照。
    #   2. 拒绝写入期间可检测的目标变化或暂存替换，失败时只清理本次拥有的文件。
    # 输入：
    #   value：完整的新信任库数据。
    # 输出：
    #   None：不返回业务数据。
    def _write(self, value: dict[str, Any]) -> None:
        rendered = encode_json(value, limit=2 * 1024 * 1024 - 1)
        self._check_paths()
        before = self.path.stat() if self.path.exists() else None
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".plugin-trust-", suffix=".json", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        identity = None
        try:
            try:
                identity = os.fstat(descriptor)
                output_context = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
            except BaseException:
                with suppress(OSError):
                    os.close(descriptor)
                raise
            with output_context as output:
                output.write(rendered + "\n")
                output.flush()
                os.fsync(output.fileno())
                written = os.fstat(output.fileno())
            check_plain_plugin_path(temporary)
            current = temporary.stat()
            if (
                not os.path.samestat(identity, current)
                or current.st_size != written.st_size
                or current.st_mtime_ns != written.st_mtime_ns
            ):
                raise TrustStoreError("PLUGIN_TRUST_STAGING_CHANGED")
            self._check_paths()
            current_store = self.path.stat() if self.path.exists() else None
            if (before is None) != (current_store is None) or (
                before is not None
                and current_store is not None
                and (
                    not os.path.samestat(before, current_store)
                    or before.st_size != current_store.st_size
                    or before.st_mtime_ns != current_store.st_mtime_ns
                )
            ):
                raise TrustStoreError("PLUGIN_TRUST_STORE_CHANGED")
            # 合作写入者受侧车锁保护；这些复核不替代针对恶意并发目录改写的系统隔离。
            os.replace(temporary, self.path)
        finally:
            if identity is not None:
                with suppress(OSError, ValueError):
                    check_plain_plugin_path(temporary)
                    if os.path.samestat(identity, temporary.stat()):
                        temporary.unlink()

    # 功能：
    #   校验 Ed25519 公钥并将密钥标识绑定到发布者；更换密钥必须使用新标识。
    # 输入：
    #   key_id：用于查找发布者密钥的稳定标识。
    #   publisher：与清单完全匹配的发布者名称。
    #   public_key_base64：三十二字节 Ed25519 公钥的 Base64 文本。
    # 输出：
    #   None：不返回业务数据。
    def add_publisher(self, *, key_id: str, publisher: str, public_key_base64: str) -> None:
        if (
            not isinstance(key_id, str)
            or re.fullmatch(r"[a-z][a-z0-9._-]{2,119}", key_id) is None
            or not isinstance(publisher, str)
            or not 1 <= len(publisher.strip()) <= 120
        ):
            raise TrustStoreError("PLUGIN_PUBLISHER_IDENTITY_INVALID")
        try:
            public_key = base64.b64decode(public_key_base64, validate=True)
            Ed25519PublicKey.from_public_bytes(public_key)
        except (ValueError, TypeError) as error:
            raise TrustStoreError("PLUGIN_PUBLISHER_KEY_INVALID") from error
        if len(public_key) != 32:
            raise TrustStoreError("PLUGIN_PUBLISHER_KEY_INVALID")
        with self._locked():
            value = self._load()
            publishers = value["publishers"]
            previous = publishers.get(key_id)
            if previous is not None and (
                previous.get("publisher") != publisher
                or previous.get("public_key_base64") != public_key_base64
            ):
                raise TrustStoreError("PLUGIN_PUBLISHER_KEY_ID_ALREADY_BOUND")
            publishers[key_id] = {
                "publisher": publisher,
                "public_key_base64": public_key_base64,
                "added_at": datetime.now(UTC).isoformat(),
            }
            self._write(value)

    # 功能：
    #   将本地批准同时绑定到包字节摘要、清单摘要和插件坐标，不扩大为发布者级授权。
    # 输入：
    #   manifest：用户批准的插件清单。
    #   package_sha256：调用方根据实际包字节计算的摘要。
    # 输出：
    #   None：不返回业务数据。
    def approve_local(self, manifest: PluginManifest, package_sha256: str) -> None:
        self._validate_package_hash(package_sha256)
        with self._locked():
            value = self._load()
            if package_sha256 in value["revoked_packages"]:
                raise TrustStoreError("PLUGIN_PACKAGE_REVOKED")
            value["local_approvals"][package_sha256] = {
                "plugin_id": manifest.plugin_id,
                "version": manifest.version,
                "manifest_sha256": unsigned_manifest_sha256(manifest),
                "approved_at": datetime.now(UTC).isoformat(),
            }
            self._write(value)

    # 功能：
    #   在同一事务内撤销指定包并清除其本地批准，重复撤销不重复追加记录。
    # 输入：
    #   package_sha256：需要撤销的包字节摘要。
    # 输出：
    #   None：不返回业务数据。
    def revoke_package(self, package_sha256: str) -> None:
        self._validate_package_hash(package_sha256)
        with self._locked():
            value = self._load()
            if package_sha256 not in value["revoked_packages"]:
                value["revoked_packages"].append(package_sha256)
            value["local_approvals"].pop(package_sha256, None)
            self._write(value)

    # 功能：
    #   撤销发布者密钥并保留历史记录，不清除其他包或密钥的撤销状态。
    # 输入：
    #   key_id：需要撤销的发布者密钥标识。
    # 输出：
    #   None：不返回业务数据。
    def revoke_publisher_key(self, key_id: str) -> None:
        if not isinstance(key_id, str) or re.fullmatch(r"[a-z][a-z0-9._-]{2,119}", key_id) is None:
            raise TrustStoreError("PLUGIN_PUBLISHER_IDENTITY_INVALID")
        with self._locked():
            value = self._load()
            if key_id not in value["revoked_publisher_keys"]:
                value["revoked_publisher_keys"].append(key_id)
            self._write(value)

    # 功能：
    #   拒绝非小写十六进制 SHA-256 摘要，避免模糊包坐标进入授权或撤销逻辑。
    # 输入：
    #   value：待校验的包摘要。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _validate_package_hash(value: str) -> None:
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise TrustStoreError("PLUGIN_PACKAGE_HASH_INVALID")

    # 功能：
    #   1. 基于同一快照先检查包与发布者撤销，再检查精确本地批准或 Ed25519 签名。
    #   2. 仅判断本次读取时的信任状态，不代替包字节核验或之后的撤销复查。
    # 输入：
    #   manifest：待判断的插件清单。
    #   package_sha256：调用方从实际安装包字节计算的摘要。
    # 输出：
    #   decision：信任状态、问题代码及绑定的清单和包摘要。
    def verify(self, manifest: PluginManifest, package_sha256: str) -> PluginTrustDecision:
        self._validate_package_hash(package_sha256)
        with self._locked():
            value = self._load()
        manifest_sha256 = unsigned_manifest_sha256(manifest)
        plugin_coordinate = f"{manifest.plugin_id}@{manifest.version}"
        if package_sha256 in value.get("revoked_packages", []) or plugin_coordinate in value.get(
            "revoked_plugins", []
        ):
            decision = PluginTrustDecision(
                status="revoked",
                issue_codes=["PLUGIN_PACKAGE_REVOKED"],
                manifest_sha256=manifest_sha256,
                package_sha256=package_sha256,
            )
            return decision

        signature = manifest.signature
        if signature is not None:
            key_id = signature.publisher_key_id
            if key_id in value.get("revoked_publisher_keys", []):
                decision = PluginTrustDecision(
                    status="revoked",
                    issue_codes=["PLUGIN_PUBLISHER_KEY_REVOKED"],
                    publisher_key_id=key_id,
                    manifest_sha256=manifest_sha256,
                    package_sha256=package_sha256,
                )
                return decision
        approval = value.get("local_approvals", {}).get(package_sha256)
        if isinstance(approval, dict) and (approval.get("plugin_id"), approval.get("version")) == (
            manifest.plugin_id,
            manifest.version,
        ):
            # 历史批准只绑定精确包摘要；保留其恢复入口，新批准同时绑定清单摘要。
            # 这不是按名称放行旧包，调用方仍必须先核对实际安装字节。
            if approval.get("manifest_sha256", manifest_sha256) != manifest_sha256:
                decision = PluginTrustDecision(
                    status="unverified",
                    issue_codes=["PLUGIN_APPROVED_MANIFEST_MISMATCH"],
                    manifest_sha256=manifest_sha256,
                    package_sha256=package_sha256,
                )
                return decision
            decision = PluginTrustDecision(
                status="local-approved",
                manifest_sha256=manifest_sha256,
                package_sha256=package_sha256,
            )
            return decision

        if signature is not None:
            key_id = signature.publisher_key_id
            publisher = value.get("publishers", {}).get(key_id)
            if not isinstance(publisher, dict):
                decision = PluginTrustDecision(
                    status="unverified",
                    issue_codes=["PLUGIN_PUBLISHER_UNKNOWN"],
                    publisher_key_id=key_id,
                    manifest_sha256=manifest_sha256,
                    package_sha256=package_sha256,
                )
                return decision
            if publisher.get("publisher") != manifest.publisher:
                issue = "PLUGIN_PUBLISHER_NAME_MISMATCH"
            elif signature.signed_manifest_sha256 != manifest_sha256:
                issue = "PLUGIN_SIGNED_MANIFEST_HASH_MISMATCH"
            else:
                try:
                    public_bytes = base64.b64decode(
                        str(publisher["public_key_base64"]), validate=True
                    )
                    signed = base64.b64decode(signature.signature_base64, validate=True)
                    Ed25519PublicKey.from_public_bytes(public_bytes).verify(
                        signed, bytes.fromhex(manifest_sha256)
                    )
                except (InvalidSignature, ValueError, TypeError, KeyError):
                    issue = "PLUGIN_SIGNATURE_INVALID"
                else:
                    decision = PluginTrustDecision(
                        status="verified",
                        publisher_key_id=key_id,
                        manifest_sha256=manifest_sha256,
                        package_sha256=package_sha256,
                    )
                    return decision
            decision = PluginTrustDecision(
                status="unverified",
                issue_codes=[issue],
                publisher_key_id=key_id,
                manifest_sha256=manifest_sha256,
                package_sha256=package_sha256,
            )
            return decision

        decision = PluginTrustDecision(
            status="unverified",
            issue_codes=["PLUGIN_SIGNATURE_REQUIRED"],
            manifest_sha256=manifest_sha256,
            package_sha256=package_sha256,
        )
        return decision
