"""Owned read handles for previews; path checks are not an OS race sandbox."""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from dronedream_agent_core.plugin_files import check_plain_plugin_path

MAX_ATTACHMENT_BYTES = 512 * 1024 * 1024


# 功能：
#   1. 打开已校验的普通文件并核对实际句柄身份，成功读取后复核身份、大小和时间。
#   2. 由上下文管理器释放句柄；调用者仍须限制读取量，不代替恶意目录竞争的 OS 隔离。
# 输入：
#   source：不超过 512 MiB 的本地附件路径。
# 输出：
#   stream：本次上下文内可用的二进制读取流。
@contextmanager
def open_preview_source(source: Path) -> Iterator[BinaryIO]:
    check_plain_plugin_path(source)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ATTACHMENT_BYTES:
        raise ValueError("ATTACHMENT_SOURCE_INVALID")
    with source.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ValueError("ATTACHMENT_FILE_CHANGED")
        yield stream
        # 只有成功解析的内容会被发布；解析失败时仍由 with 关闭句柄，不覆盖解析异常。
        after = os.fstat(stream.fileno())
        check_plain_plugin_path(source)
        current = source.stat()
        if (
            not os.path.samestat(after, current)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or current.st_size != after.st_size
            or current.st_mtime_ns != after.st_mtime_ns
        ):
            raise ValueError("ATTACHMENT_FILE_CHANGED")
