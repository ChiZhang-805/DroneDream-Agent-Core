"""Bound packaged Runtime contents to their release inventory, without installing them.

Content hashes detect stale or changed files; publisher authenticity remains the
installer's responsibility. Filesystem checks do not replace process isolation.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_plugin_sdk.protocol import decode_json

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_RESOURCE_FILES = 100_000
MAX_RESOURCE_ENTRIES = 200_000
MAX_RESOURCE_BYTES = 50_000_000_000


# 功能：
#   1. 校验当前发布清单的格式、唯一规范路径、明确大小和摘要。
#   2. 有界枚举全部资源并逐文件复核，拒绝未索引的旧 wheel、源码、模型或其他文件。
#   3. 此检查只验证内容一致性，不执行资源、不安装软件，也不验证发布者签名。
# 输入：
#   root：产品分发的 runtime 资源目录，不是 WSL 安装目录。
# 输出：
#   manifest_sha256：通过完整内容检查的清单实际字节摘要。
def verify_runtime_resource_manifest(root: Path) -> str:
    check_plain_plugin_path(root)
    if not root.is_dir():
        raise ValueError("RUNTIME_RESOURCE_ROOT_INVALID")
    raw = read_plugin_file(root / "runtime-manifest.json", limit=MAX_MANIFEST_BYTES)
    manifest = decode_json(raw, limit=MAX_MANIFEST_BYTES, node_limit=2_000_000)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise ValueError("RUNTIME_RESOURCE_MANIFEST_INVALID")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_RESOURCE_FILES:
        raise ValueError("RUNTIME_RESOURCE_INDEX_INVALID")
    expected: dict[str, tuple[int, str]] = {}
    folded: set[str] = {"runtime-manifest.json"}
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("RUNTIME_RESOURCE_ENTRY_INVALID")
        name = portable_plugin_path(entry.get("path"))
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            name.casefold() in folded
            or type(size) is not int
            or not 0 <= size <= MAX_RESOURCE_BYTES
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("RUNTIME_RESOURCE_ENTRY_INVALID")
        total_bytes += size
        if total_bytes > MAX_RESOURCE_BYTES:
            raise ValueError("RUNTIME_RESOURCE_BYTES_EXCEEDED")
        expected[name] = size, digest
        folded.add(name.casefold())

    pending = [root]
    actual: set[str] = set()
    seen: set[str] = set()
    count = 0
    while pending:
        directory = pending.pop()
        check_plain_plugin_path(directory)
        with os.scandir(directory) as children:
            for child in children:
                count += 1
                if count > MAX_RESOURCE_ENTRIES:
                    raise ValueError("RUNTIME_RESOURCE_ENTRIES_EXCEEDED")
                path = Path(child.path)
                name = portable_plugin_path(path.relative_to(root).as_posix())
                if name.casefold() in seen:
                    raise ValueError("RUNTIME_RESOURCE_PATH_COLLISION")
                seen.add(name.casefold())
                check_plain_plugin_path(path)
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    # 空 ROS 源目录可合法存在，但空目录也消耗枚举预算。
                    pending.append(path)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("RUNTIME_RESOURCE_FILE_TYPE_INVALID")
                elif name != "runtime-manifest.json":
                    if name not in expected or metadata.st_size != expected[name][0]:
                        raise ValueError("RUNTIME_RESOURCE_FILE_INDEX_MISMATCH")
                    actual.add(name)
    if actual != expected.keys():
        raise ValueError("RUNTIME_RESOURCE_FILE_MISSING")
    for name, (size, digest) in expected.items():
        observed = hash_plugin_file(root / name, limit=size)
        if observed != digest:
            raise ValueError("RUNTIME_RESOURCE_HASH_MISMATCH")
    # 检查结束时清单也必须仍是同一内容，不能将旧索引的校验结果绑定到新清单。
    if read_plugin_file(root / "runtime-manifest.json", limit=MAX_MANIFEST_BYTES) != raw:
        raise ValueError("RUNTIME_RESOURCE_MANIFEST_CHANGED")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    return manifest_sha256
