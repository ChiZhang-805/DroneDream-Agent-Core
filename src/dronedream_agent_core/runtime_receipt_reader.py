"""Run-scoped receipt reads without repeated ancestor walks on mounted disks."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES

from .plugin_files import check_plain_plugin_path
from .runtime_file_reader import PinnedRuntimeObjectReader


class RuntimeReceiptReader:
    # 功能：
    #   固定当前运行的回执入口，仅缓存读取句柄，不缓存回执内容或延长状态有效期。
    # 输入：
    #   directory：本次运行的动作回执目录，允许尚未创建。
    # 输出：
    #   None：建立有界的读取器集合。
    def __init__(self, directory: Path):
        self.directory = directory.absolute()
        check_plain_plugin_path(self.directory)
        self._identity = None
        self._reader: PinnedRuntimeObjectReader | None = None
        self._names: set[str] = set()
        self._closed = False

    # 功能：
    #   1. 每次读取全部当前回执字节，减少重复祖先路径检查，保留文件替换及损坏检查。
    #   2. 目录或既有回执消失、目录替换、条目超限时拒绝，不退回无载荷状态。
    # 输入：
    #   self：本次运行独占的读取器。
    # 输出：
    #   receipts：按原有文件名顺序排列的路径与回执二元组列表。
    def read(self) -> list[tuple[Path, dict]]:
        if self._closed:
            raise ValueError("RUNTIME_RECEIPT_READER_CLOSED")
        try:
            current = self.directory.lstat()
        except FileNotFoundError:
            if self._identity is not None:
                raise ValueError("RUNTIME_RECEIPT_DIRECTORY_REMOVED") from None
            return []
        if not stat.S_ISDIR(current.st_mode):
            raise ValueError("RUNTIME_RECEIPT_DIRECTORY_INVALID")
        if self._identity is None:
            check_plain_plugin_path(self.directory)
            self._identity = current
        if not os.path.samestat(self._identity, current):
            raise ValueError("RUNTIME_RECEIPT_DIRECTORY_CHANGED")
        names = []
        with os.scandir(self.directory) as entries:
            for index, entry in enumerate(entries):
                if index >= 4096:
                    raise ValueError("RUNTIME_RECEIPT_DIRECTORY_LIMIT")
                if entry.name.endswith(".receipt.json"):
                    names.append(entry.name)
                    if len(names) > 256:
                        raise ValueError("RUNTIME_RECEIPT_COUNT_LIMIT")
        if not self._names <= set(names):
            raise ValueError("RUNTIME_RECEIPT_REMOVED")
        receipts = []
        if names:
            names.sort()
            if self._reader is None:
                self._reader = PinnedRuntimeObjectReader(
                    self.directory / names[0], maximum_bytes=MAX_MESSAGE_BYTES)
            values = self._reader.read_siblings(names)
            receipts = [(self.directory / name, value) for name, value in zip(names, values, strict=True)]
        after = self.directory.lstat()
        if not stat.S_ISDIR(after.st_mode) or not os.path.samestat(self._identity, after):
            raise ValueError("RUNTIME_RECEIPT_DIRECTORY_CHANGED")
        self._names = set(names)
        return receipts

    # 功能：
    #   关闭全部句柄，退出后不可重新绑定到任何目录。
    # 输入：
    #   self：需要回收的读取器。
    # 输出：
    #   None：完成幂等资源回收。
    def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.close()
