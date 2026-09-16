"""Portable paths and bounded content checks shared by install and task dispatch.

These checks reject static links and detect file replacement during a read. They
do not replace OS isolation against a hostile process racing directory changes.
"""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack
from pathlib import Path

MAX_PLUGIN_FILE_BYTES = 256 * 1024 * 1024
MAX_PLUGIN_TOTAL_BYTES = 512 * 1024 * 1024
MAX_PLUGIN_ENTRIES = 2_000
_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in "123456789¹²³"
}


# 功能：
#   检查 Windows 与 Linux 含义一致的相对路径，拒绝设备名、点段及非法字符。
# 输入：
#   value：以正斜杠分隔的包内路径。
# 输出：
#   value：通过校验、无需修改拼写的原路径。
def portable_plugin_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
        raise ValueError("PLUGIN_FILE_PATH_INVALID")
    parts = value.split("/")
    if len(parts) > 32 or any(
        not part
        or part in {".", ".."}
        or part.endswith((".", " "))
        or part.split(".", 1)[0].casefold() in _DEVICES
        or any(ord(char) < 32 or char in '\\:*?"<>|' for char in part)
        for part in parts
    ):
        raise ValueError("PLUGIN_FILE_PATH_INVALID")
    return value


# 功能：
#   检查目标及全部父目录，拒绝符号链接和 Windows 重解析点，不替代系统沙箱。
# 输入：
#   path：准备访问或创建的本地路径。
# 输出：
#   None：不返回业务数据。
def check_plain_plugin_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError("PLUGIN_FILE_LINK_FORBIDDEN")


# 功能：
#   有界读取普通文件，复核读取前后的身份、大小与修改时间，拒绝并发替换或增长。
# 输入：
#   path：需要读取的文件路径。
#   limit：允许读取的最大字节数，必须为非负整数。
# 输出：
#   value：通过实际读取和身份检查的字节内容。
def read_plugin_file(path: Path, *, limit: int) -> bytes:
    if type(limit) is not int or limit < 0:
        raise ValueError("PLUGIN_FILE_LIMIT_INVALID")
    check_plain_plugin_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("PLUGIN_FILE_TOO_LARGE_OR_INVALID")
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ValueError("PLUGIN_FILE_CHANGED")
        # 预算只是上限；小文件按已知大小加一读取，既发现增长又避免大额预分配。
        value = source.read(min(limit, opened.st_size) + 1)
        after = os.fstat(source.fileno())
    if (
        len(value) > limit
        or len(value) != opened.st_size
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ValueError("PLUGIN_FILE_CHANGED_OR_TOO_LARGE")
    check_plain_plugin_path(path)
    if not os.path.samestat(after, path.stat()):
        raise ValueError("PLUGIN_FILE_CHANGED")
    return value


# 功能：
#   1. 流式计算普通文件的摘要，可同时将同一批字节冻结到独占创建的新文件。
#   2. 拒绝读取期间的身份或大小变化；失败现场由调用方处理，不覆盖或删除目标。
# 输入：
#   path：源文件路径。
#   limit：允许读取的最大字节数，必须为非负整数。
#   destination：可选的新目标文件路径，必须尚不存在。
# 输出：
#   checksum：实际读取及复制字节的 SHA-256 十六进制摘要。
def hash_plugin_file(path: Path, *, limit: int, destination: Path | None = None) -> str:
    if type(limit) is not int or limit < 0:
        raise ValueError("PLUGIN_FILE_LIMIT_INVALID")
    check_plain_plugin_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("PLUGIN_FILE_TOO_LARGE_OR_INVALID")
    with ExitStack() as stack:
        source = stack.enter_context(path.open("rb"))
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ValueError("PLUGIN_FILE_CHANGED")
        target = None
        if destination is not None:
            check_plain_plugin_path(destination)
            target = stack.enter_context(destination.open("xb"))
        digest = hashlib.sha256()
        total = 0
        maximum = min(limit, opened.st_size)
        while chunk := source.read(min(1024 * 1024, maximum - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ValueError("PLUGIN_FILE_TOO_LARGE")
            if total > opened.st_size:
                raise ValueError("PLUGIN_FILE_CHANGED")
            digest.update(chunk)
            if target is not None:
                target.write(chunk)
        after = os.fstat(source.fileno())
        if (
            total != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ValueError("PLUGIN_FILE_CHANGED")
    check_plain_plugin_path(path)
    if not os.path.samestat(after, path.stat()):
        raise ValueError("PLUGIN_FILE_CHANGED")
    checksum = digest.hexdigest()
    return checksum


# 功能：
#   1. 对照预期索引检查全部负载文件，拒绝额外文件、大小写冲突、链接和特殊文件。
#   2. 根目录 plugin.json 因包含索引而不自索引，嵌套清单仍需校验；目录也计入容量限制。
# 输入：
#   root：插件内容根目录。
#   expected：包内规范路径与预期 SHA-256 摘要的映射。
# 输出：
#   None：不返回业务数据。
def verify_plugin_files(root: Path, expected: dict[str, str]) -> None:
    check_plain_plugin_path(root)
    if not root.is_dir():
        raise ValueError("PLUGIN_BUNDLE_ROOT_MISSING")
    normalized = {portable_plugin_path(name): digest for name, digest in expected.items()}
    if len({name.casefold() for name in normalized}) != len(normalized):
        raise ValueError("PLUGIN_FILE_PATH_COLLISION")
    pending = [root]
    files: dict[str, Path] = {}
    names: set[str] = set()
    entries = 0
    while pending:
        directory = pending.pop()
        check_plain_plugin_path(directory)
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > MAX_PLUGIN_ENTRIES:
                    raise ValueError("PLUGIN_BUNDLE_LIMIT_EXCEEDED")
                path = Path(child.path)
                name = portable_plugin_path(path.relative_to(root).as_posix())
                if name.casefold() in names:
                    raise ValueError("PLUGIN_FILE_PATH_COLLISION")
                names.add(name.casefold())
                metadata = child.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & 0x400
                ):
                    raise ValueError("PLUGIN_FILE_LINK_FORBIDDEN")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("PLUGIN_FILE_TYPE_FORBIDDEN")
                elif name != "plugin.json":
                    files[name] = path
    if files.keys() != normalized.keys():
        raise ValueError("PLUGIN_FILE_MANIFEST_MISMATCH")
    remaining = MAX_PLUGIN_TOTAL_BYTES
    for name, path in sorted(files.items()):
        content = read_plugin_file(path, limit=min(MAX_PLUGIN_FILE_BYTES, remaining))
        remaining -= len(content)
        if hashlib.sha256(content).hexdigest() != normalized[name]:
            raise ValueError(f"PLUGIN_FILE_HASH_MISMATCH:{name}")
