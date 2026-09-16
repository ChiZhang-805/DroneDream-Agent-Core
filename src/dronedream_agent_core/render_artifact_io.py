"""Bounded render artifact identity checks; not a native-test or flight attestation."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file

MAX_RENDER_FILE_BYTES = 4 * 1024**3
MAX_NATIVE_SOURCE_BYTES = 64 * 1024**2
MAX_NATIVE_SOURCE_ENTRIES = 4096
MAX_DEPENDENCY_BYTES = 4 * 1024**3


# 功能：
#   有界计算普通制品文件的摘要，复用读取期间的身份、大小、修改时间及链接检查。
# 输入：
#   path：需要绑定实际字节的本地文件。
# 输出：
#   digest：当前文件内容的 SHA-256 摘要。
def file_sha(path: Path) -> str:
    digest = hash_plugin_file(path, limit=MAX_RENDER_FILE_BYTES)
    return digest


# 功能：
#   读取一份有界 JSON 快照并拒绝重复键、非有限数字、错误根类型与超深结构。
# 输入：
#   path：需要读取的普通回执或清单文件。
#   limit：允许的原始字节上限。
#   error_code：调用模块使用的错误前缀。
# 输出：
#   snapshot：解析字典与同一次读取的原始字节组成的二元组。
def read_object_snapshot(path: Path, *, limit: int, error_code: str) -> tuple[dict, bytes]:
    try:
        content = read_plugin_file(path, limit=limit)
        data = decode_json(content, limit=limit, node_limit=2_000_000)
        if not isinstance(data, dict):
            raise ValueError("RENDER_OBJECT_EXPECTED")
    except (OSError, ValueError, RecursionError) as error:
        raise ValueError(error_code) from error
    snapshot = data, content
    return snapshot


# 功能：
#   在条目和字节预算内递归绑定普通目录，子目录也计数，链接与特殊文件均拒绝。
# 输入：
#   source：实际读取的源码或资源目录。
#   maximum_entries：本次剩余目录及文件条目预算。
#   maximum_bytes：本次剩余文件内容字节预算。
# 输出：
#   result：相对路径摘要表、实际条目数和实际文件总字节数。
def inventory_tree(
    source: Path, *, maximum_entries: int, maximum_bytes: int
) -> tuple[dict, int, int]:
    if any(type(value) is not int or value < 0 for value in (maximum_entries, maximum_bytes)):
        raise ValueError("RENDER_INVENTORY_BUDGET_INVALID")
    check_plain_plugin_path(source)
    if not source.is_dir():
        raise ValueError("RENDER_NATIVE_SOURCE_ROOT_INVALID")
    inventory: dict[str, str] = {}
    pending, count, total = [source], 0, 0
    while pending:
        directory = pending.pop()
        check_plain_plugin_path(directory)
        with os.scandir(directory) as children:
            for child in children:
                count += 1
                if count > maximum_entries:
                    raise ValueError("RENDER_NATIVE_SOURCE_LIMIT")
                path = Path(child.path)
                check_plain_plugin_path(path)
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise ValueError("RENDER_NATIVE_SOURCE_NOT_REGULAR")
                size = child.stat(follow_symlinks=False).st_size
                total += size
                if total > maximum_bytes:
                    raise ValueError("RENDER_NATIVE_SOURCE_LIMIT")
                # 摘要读取上限绑定刚计入预算的大小，增长不能突破总预算后才被发现。
                inventory[path.relative_to(source).as_posix()] = hash_plugin_file(path, limit=size)
    inventory = dict(sorted(inventory.items()))
    result = inventory, count, total
    return result


# 功能：
#   以原生源码预算完整枚举当前构建输入，空目录不能形成有效源码绑定。
# 输入：
#   source：原生构建实际使用的源码目录。
# 输出：
#   inventory：以规范相对路径为键的源码摘要表。
def source_inventory(source: Path) -> dict[str, str]:
    inventory, _, _ = inventory_tree(source, maximum_entries=MAX_NATIVE_SOURCE_ENTRIES,
                                    maximum_bytes=MAX_NATIVE_SOURCE_BYTES)
    if not inventory:
        raise ValueError("RENDER_NATIVE_SOURCE_EMPTY")
    return inventory


# 功能：
#   在文件数及总字节预算内逐个校验构建依赖，只接纳绝对路径和合法摘要。
# 输入：
#   inventory：由已解析回执提供的路径到 SHA-256 映射。
#   error_code：依赖不完整、被替换或超限时的错误前缀。
# 输出：
#   None：不返回业务数据。
def validate_dependencies(inventory: object, *, error_code: str) -> None:
    if not isinstance(inventory, dict) or not 1 <= len(inventory) <= 2048:
        raise ValueError(error_code)
    total = 0
    for name, digest in inventory.items():
        if (not isinstance(name, str) or not isinstance(digest, str)
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            raise ValueError(error_code)
        path = Path(name)
        try:
            if not path.is_absolute():
                raise ValueError("RENDER_DEPENDENCY_PATH_NOT_ABSOLUTE")
            size = path.stat().st_size
            total += size
            if total > MAX_DEPENDENCY_BYTES or hash_plugin_file(path, limit=size) != digest:
                raise ValueError("RENDER_DEPENDENCY_CHANGED_OR_TOO_LARGE")
        except (OSError, ValueError) as error:
            raise ValueError(error_code + ":" + name) from error
