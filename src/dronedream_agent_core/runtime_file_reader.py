"""Pinned run-directory reads for live local evidence, not arbitrary plugin imports.

POSIX consumers keep a verified directory handle instead of repeating every
ancestor metadata lookup across WSL/Windows on every control observation.
The current directory name, opened file and final name must still agree.
This detects replacement; it is not isolation from a hostile same-user process.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .plugin_files import check_plain_plugin_path
from .runtime_control_io import read_runtime_object


class PinnedRuntimeObjectReader:
    """One file in one immutable run-directory binding; caller owns close()."""

    # 功能：
    #   核对完整路径，在支持目录句柄的平台固定运行目录身份；其他平台保留严格普通读取。
    # 输入：
    #   path：本次运行固定的状态文件路径，文件可尚未发布。
    #   maximum_bytes：单份 JSON 的字节预算。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path, *, maximum_bytes: int):
        encode_json(0, limit=maximum_bytes)
        self.path = path.absolute()
        self.maximum_bytes = maximum_bytes
        self._directory = None
        self._closed = False
        check_plain_plugin_path(self.path)
        supported = (os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
                     and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"))
        if supported:
            before = self.path.parent.stat()
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                opened = os.fstat(directory)
                check_plain_plugin_path(self.path)
                if not os.path.samestat(before, opened) or not os.path.samestat(
                        opened, self.path.parent.stat()):
                    raise ValueError("RUNTIME_READER_DIRECTORY_CHANGED")
            except BaseException:
                os.close(directory)
                raise
            self._directory = directory
            self._directory_identity = opened

    # 功能：
    #   有界读取固定文件，拒绝链接、非普通文件、读中替换、增长或所属目录变化。
    # 输入：
    #   self：持有运行目录身份的读取器。
    # 输出：
    #   value：通过有限 JSON 与对象检查的当前快照。
    def read(self) -> dict:
        if self._closed:
            raise ValueError("RUNTIME_READER_CLOSED")
        if self._directory is None:
            return read_runtime_object(self.path, maximum_bytes=self.maximum_bytes)
        self._check_directory()
        before = os.stat(self.path.name, dir_fd=self._directory, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > self.maximum_bytes:
            raise ValueError("RUNTIME_READER_FILE_INVALID")
        # NONBLOCK 避免在 stat 与 open 之间被换成 FIFO 时卡住读取线程。
        descriptor = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=self._directory)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                raise ValueError("RUNTIME_READER_FILE_CHANGED")
            content = stream.read(min(opened.st_size, self.maximum_bytes) + 1)
            after = os.fstat(stream.fileno())
        current = os.stat(self.path.name, dir_fd=self._directory, follow_symlinks=False)
        self._check_directory()
        if (len(content) != opened.st_size or len(content) > self.maximum_bytes
                or after.st_size != opened.st_size or current.st_size != opened.st_size
                or after.st_mtime_ns != opened.st_mtime_ns
                or current.st_mtime_ns != opened.st_mtime_ns
                or not os.path.samestat(after, current)):
            raise ValueError("RUNTIME_READER_FILE_CHANGED")
        value = decode_json(content, limit=self.maximum_bytes)
        if not isinstance(value, dict):
            raise ValueError("RUNTIME_CONTROL_EVIDENCE_NOT_OBJECT")
        return value

    # 功能：
    #   核对当前目录项仍是最初固定的普通目录，不沿替换后的链接或另一个运行目录读取。
    # 输入：
    #   self：当前固定目录读取器。
    # 输出：
    #   None：不返回业务数据。
    def _check_directory(self) -> None:
        current = self.path.parent.lstat()
        if (not stat.S_ISDIR(current.st_mode)
                or not os.path.samestat(self._directory_identity, current)):
            raise ValueError("RUNTIME_READER_DIRECTORY_CHANGED")

    # 功能：
    #   幂等关闭目录句柄，关闭后的实例不能再读取或自动绑定到另一个运行。
    # 输入：
    #   self：当前读取器。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._directory is not None:
                os.close(self._directory)
                self._directory = None
