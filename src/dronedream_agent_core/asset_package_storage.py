"""Bounded I/O for immutable asset directories; never grants flight qualification.

Filesystem checks detect ordinary corruption and replacement, not hostile-process
isolation. Windows uses no-replace rename; Linux requires RENAME_NOREPLACE support.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
import tempfile
import zipfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from dronedream_plugin_sdk.protocol import decode_json

from .asset_packages import (
    MAX_MANIFEST_BYTES,
    MAX_MEMBER_UNCOMPRESSED_BYTES,
    AssetFile,
    DDPkgManifest,
    InspectedDDPkg,
    inspect_ddpkg,
    open_verified_ddpkg,
)
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file


# 功能：
#   有界读取并验证已安装资产的清单，保留实际读取字节以供重新归档。
# 输入：
#   root：资产内容目录。
# 输出：
#   manifest_data：类型化清单与其原始字节组成的元组。
def read_stored_manifest(root: Path) -> tuple[DDPkgManifest, bytes]:
    payload = read_plugin_file(root / "manifest.json", limit=MAX_MANIFEST_BYTES)
    manifest = DDPkgManifest.model_validate(
        decode_json(payload, limit=MAX_MANIFEST_BYTES, node_limit=2_000_000)
    )
    manifest_data = manifest, payload
    return manifest_data


# 功能：
#   逐块复制索引中的成员，按实际字节检查大小和摘要，不信任声明长度本身。
# 输入：
#   source：当前成员的输入流。
#   target：调用方独占创建的输出流。
#   entry：已校验的成员索引。
# 输出：
#   copied：实际复制的字节数。
def copy_indexed_asset_stream(source: BinaryIO, target: BinaryIO, entry: AssetFile) -> int:
    entry = AssetFile.model_validate(entry.model_dump(mode="python"))
    copied = 0
    digest = hashlib.sha256()
    while chunk := source.read(min(1024 * 1024, entry.size_bytes - copied + 1)):
        copied += len(chunk)
        if copied > entry.size_bytes or copied > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(f"ASSET_VERSION_SIZE_MISMATCH:{entry.path}")
        digest.update(chunk)
        target.write(chunk)
    if copied != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise ValueError(f"ASSET_VERSION_HASH_MISMATCH:{entry.path}")
    return copied


# 功能：
#   将普通源文件有界复制到调用方已独占打开的输出流，复核文件身份与读取期间变化。
# 输入：
#   source_path：需要冻结的源文件。
#   target：调用方持有的输出流。
#   limit：允许复制的最大字节数。
# 输出：
#   digest：实际复制字节的 SHA-256。
def copy_asset_source(source_path: Path, target: BinaryIO, limit: int) -> str:
    if type(limit) is not int or limit < 0:
        raise ValueError("ASSET_VERSION_COPY_LIMIT_INVALID")
    check_plain_plugin_path(source_path)
    before = source_path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("ASSET_VERSION_SOURCE_SIZE_OR_TYPE_INVALID")
    with source_path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(before, opened)
            or opened.st_size != before.st_size
        ):
            raise ValueError("ASSET_VERSION_FILE_CHANGED")
        hasher = hashlib.sha256()
        copied = 0
        while chunk := source.read(min(1024 * 1024, limit - copied + 1)):
            copied += len(chunk)
            if copied > limit:
                raise ValueError("ASSET_VERSION_SOURCE_SIZE_OR_TYPE_INVALID")
            hasher.update(chunk)
            target.write(chunk)
        after = os.fstat(source.fileno())
    check_plain_plugin_path(source_path)
    if (
        copied != opened.st_size
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or not os.path.samestat(after, source_path.stat())
    ):
        raise ValueError("ASSET_VERSION_FILE_CHANGED")
    digest = hasher.hexdigest()
    return digest


# 功能：
#   1. 将源文件复制到独占持有的临时流，摘要匹配后无覆盖发布为新文件。
#   2. 失败时只清理身份仍属于本次操作的暂存文件，不误删后来占用该名称的文件。
#   3. 路径复核不替代恶意进程竞争下的系统隔离；目标冲突时不回退到覆盖操作。
# 输入：
#   source：已完成语义检查的普通源文件。
#   destination：尚不存在的新目标路径。
#   expected_sha256：语义检查对应的原始文件摘要。
#   limit：允许复制的最大字节数。
# 输出：
#   destination：完整复制并发布的目标文件路径。
def publish_asset_file(
    source: Path, destination: Path, *, expected_sha256: str, limit: int
) -> Path:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("ASSET_FILE_EXPECTED_HASH_INVALID")
    if type(limit) is not int or limit < 0:
        raise ValueError("ASSET_VERSION_COPY_LIMIT_INVALID")
    check_plain_plugin_path(destination)
    if destination.exists():
        raise FileExistsError("ASSET_FILE_DESTINATION_EXISTS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    identity = None
    try:
        # 保留独占创建返回的原始描述符，不按临时名称关闭后重开写入。
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent, delete=False
        ) as target:
            temporary = Path(target.name)
            identity = os.fstat(target.fileno())
            digest = copy_asset_source(source, target, limit)
            if digest != expected_sha256:
                raise ValueError("ASSET_FILE_COPY_HASH_MISMATCH")
        check_plain_plugin_path(temporary)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("ASSET_FILE_STAGING_CHANGED")
        check_plain_plugin_path(destination)
        # 硬链接创建要么成功发布全部字节，要么因目标已存在而失败，不能 replace。
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


# 功能：
#   1. 对照清单检查目录中的全部文件和必要父目录，拒绝额外文件及链接。
#   2. 复核清单与成员实际字节，不把内容摘要形式的目录名当作验证结果。
# 输入：
#   root：待检查的已安装或暂存目录。
#   expected：从源包读取的预期清单。
# 输出：
#   None：不返回业务数据。
def verify_stored_asset(root: Path, expected: DDPkgManifest) -> None:
    expected = DDPkgManifest.model_validate(expected.model_dump(mode="python"))
    manifest, _ = read_stored_manifest(root)
    if manifest.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("ASSET_VERSION_MANIFEST_MISMATCH")
    files = {"manifest.json", *(entry.path for entry in expected.files)}
    directories = {
        parent.as_posix()
        for name in files
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    remaining = set(files)
    pending = [root]
    while pending:
        directory = pending.pop()
        check_plain_plugin_path(directory)
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(root).as_posix()
                check_plain_plugin_path(path)
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode) and relative in directories:
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode) and relative in remaining:
                    remaining.remove(relative)
                else:
                    # 只遍历索引允许的父目录，额外目录立即拒绝，不递归扫描任意树。
                    raise ValueError("ASSET_VERSION_FILE_INDEX_MISMATCH")
    if remaining:
        raise ValueError("ASSET_VERSION_FILE_MISSING")
    for entry in expected.files:
        path = root / entry.path
        try:
            digest = hash_plugin_file(path, limit=entry.size_bytes)
        except ValueError as error:
            raise ValueError(f"ASSET_VERSION_HASH_MISMATCH:{entry.path}") from error
        if path.stat().st_size != entry.size_bytes or digest != entry.sha256:
            raise ValueError(f"ASSET_VERSION_HASH_MISMATCH:{entry.path}")


# 功能：
#   在调用方独占的空目录中提取资产，同一个归档描述符负责校验和复制。
# 输入：
#   archive：待提取的隔离资产包。
#   destination：调用方已独占创建的空目录。
#   inspected：先前检查的完整元数据。
# 输出：
#   None：不返回业务数据。
def extract_verified_asset(archive: Path, destination: Path, inspected: InspectedDDPkg) -> None:
    check_plain_plugin_path(destination)
    if not destination.is_dir() or next(destination.iterdir(), None) is not None:
        raise ValueError("ASSET_VERSION_STAGING_NOT_EMPTY")
    with open_verified_ddpkg(archive) as (bundle, current):
        if current.model_dump(mode="json") != inspected.model_dump(mode="json"):
            raise ValueError("ASSET_VERSION_SOURCE_CHANGED")
        with bundle.open("manifest.json") as source:
            payload = source.read(MAX_MANIFEST_BYTES + 1)
        manifest = DDPkgManifest.model_validate(
            decode_json(payload, limit=MAX_MANIFEST_BYTES, node_limit=2_000_000)
        )
        if manifest.model_dump(mode="json") != current.manifest.model_dump(mode="json"):
            raise ValueError("ASSET_VERSION_SOURCE_CHANGED")
        with (destination / "manifest.json").open("xb") as output:
            output.write(payload)
        for entry in current.manifest.files:
            target = destination / entry.path
            check_plain_plugin_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry.path) as source, target.open("xb") as output:
                copy_indexed_asset_stream(source, output, entry)
    verify_stored_asset(destination, current.manifest)


# 功能：
#   原子发布独占暂存目录；目标已存在时失败，不回退到会覆盖目录的操作。
# 输入：
#   source：已验证且属于调用方的暂存目录。
#   destination：尚不存在的最终目录。
# 输出：
#   None：不返回业务数据。
def publish_asset_directory(source: Path, destination: Path) -> None:
    check_plain_plugin_path(source)
    check_plain_plugin_path(destination)
    if os.name == "nt":
        os.rename(source, destination)
    elif sys.platform.startswith("linux"):
        try:
            rename = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError as error:
            raise OSError(errno.ENOTSUP, "ASSET_DIRECTORY_PUBLICATION_UNSUPPORTED") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        # 使用绝对路径；1 为 RENAME_NOREPLACE，底层不支持时明确失败。
        result = rename(
            -100, os.fsencode(source.absolute()), -100, os.fsencode(destination.absolute()), 1
        )
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, "ASSET_DIRECTORY_PUBLICATION_FAILED", str(destination))
    else:
        raise OSError(errno.ENOTSUP, "ASSET_DIRECTORY_PUBLICATION_UNSUPPORTED")


# 功能：
#   1. 从冻结清单重建资产包，复制期间复核大小、摘要和文件身份。
#   2. 校验完整新包后无覆盖发布；失败时只清理本次独占创建的临时文件。
# 输入：
#   root：已安装的内容目录。
#   destination：尚不存在的导出路径。
#   expected_asset_id：调用方选定的资产标识。
#   expected_content_sha256：调用方选定的内容摘要。
# 输出：
#   destination：完成验证的新资产包路径。
def export_stored_asset(
    root: Path,
    destination: Path,
    *,
    expected_asset_id: str,
    expected_content_sha256: str,
) -> Path:
    check_plain_plugin_path(root)
    check_plain_plugin_path(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("ASSET_VERSION_EXPORT_INSIDE_SOURCE")
    if destination.exists():
        raise FileExistsError("ASSET_VERSION_EXPORT_DESTINATION_EXISTS")
    manifest, manifest_payload = read_stored_manifest(root)
    if manifest.asset_id != expected_asset_id or manifest.content_sha256 != expected_content_sha256:
        raise ValueError("ASSET_VERSION_MANIFEST_INVALID")
    verify_stored_asset(root, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid4().hex}.tmp")
    owns_temporary = False
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            owns_temporary = True
            archive.writestr("manifest.json", manifest_payload)
            for entry in manifest.files:
                source_path = root / entry.path
                check_plain_plugin_path(source_path)
                before = source_path.stat()
                with source_path.open("rb") as source:
                    opened = os.fstat(source.fileno())
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or not os.path.samestat(before, opened)
                        or opened.st_size != entry.size_bytes
                    ):
                        raise ValueError("ASSET_VERSION_FILE_CHANGED")
                    with archive.open(entry.path, "w", force_zip64=True) as target:
                        copy_indexed_asset_stream(source, target, entry)
                    after = os.fstat(source.fileno())
                check_plain_plugin_path(source_path)
                if (
                    after.st_size != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or not os.path.samestat(after, source_path.stat())
                ):
                    raise ValueError("ASSET_VERSION_FILE_CHANGED")
        exported = inspect_ddpkg(temporary)
        if exported.manifest.model_dump(mode="json") != manifest.model_dump(mode="json"):
            raise ValueError("ASSET_VERSION_EXPORT_MISMATCH")
        check_plain_plugin_path(destination)
        os.link(temporary, destination)
        return destination
    finally:
        if owns_temporary:
            with suppress(OSError):
                temporary.unlink()
