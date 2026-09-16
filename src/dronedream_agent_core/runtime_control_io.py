"""Bounded local control evidence I/O, not a hardware or filesystem sandbox."""

from __future__ import annotations

import json
import math
import os
import stat
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json, encode_json

from .plugin_files import check_plain_plugin_path, read_plugin_file
from .plugin_values import plugin_json_value

MAX_RUNTIME_REPLACEMENT_BYTES = 16 * 1024 * 1024


# 功能：
#   1. 通过独占硬链接转移控制文件，不覆盖已领取或已处理的同名消息。
#   2. 删除源目录项前核对文件身份与内容元数据；异常时保留现场，不宣称恶意并发隔离。
# 输入：
#   source：当前明确拥有的普通控制文件。
#   destination：同一文件系统中必须尚不存在的接收路径。
# 输出：
#   None：不返回业务数据。
def transfer_runtime_file(source: Path, destination: Path) -> None:
    check_plain_plugin_path(source)
    check_plain_plugin_path(destination)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("RUNTIME_CONTROL_TRANSFER_SOURCE_INVALID")
    # link 以不覆盖方式领取文件；若进程此刻退出，两份目录项保留同一证据供恢复。
    os.link(source, destination)
    check_plain_plugin_path(source)
    check_plain_plugin_path(destination)
    after = source.stat()
    claimed = destination.stat()
    if (
        not os.path.samestat(before, after)
        or not os.path.samestat(before, claimed)
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError("RUNTIME_CONTROL_TRANSFER_SOURCE_CHANGED")
    source.unlink()


# 功能：
#   有界读取普通文件并核对读取身份，拒绝重复键、非有限值、过深结构和非对象证据。
# 输入：
#   path：运行记录的明确本地路径。
#   maximum_bytes：该种记录的字节上限；大型替换轨迹由宿主单独设置。
# 输出：
#   value：通过普通文件和有限 JSON 检查的证据字典。
def read_runtime_object(path: Path, *, maximum_bytes: int = MAX_MESSAGE_BYTES) -> dict:
    encode_json(0, limit=maximum_bytes)
    payload = read_plugin_file(path, limit=maximum_bytes)
    value = decode_json(payload, limit=maximum_bytes)
    if not isinstance(value, dict):
        raise ValueError("RUNTIME_CONTROL_EVIDENCE_NOT_OBJECT")
    return value


# 功能：
#   1. 写盘前校验原始有限数据及字节预算，通过独占暂存文件发布完整 JSON。
#   2. 发布与清理均核对暂存身份，不删除未归属文件；只读打开者不会看到半份内容。
#   3. 拒绝静态链接，但不声称能够隔离恶意进程对目录的并发改写。
# 输入：
#   path：证据目标路径。
#   payload：类型化模型或标准 JSON 数据。
#   replace_existing：为假时保留先到达的操作员请求或已发布决定。
#   maximum_bytes：本次制品的完整 UTF-8 字节上限。
#   replace_timeout_seconds：覆盖遇到读锁后最多等待的秒数，零表示不重试。
#   replace_retry_seconds：读锁重试间隔；实际等待不超过剩余预算。
# 输出：
#   None：不返回业务数据。
def publish_runtime_json(
    path: Path,
    payload: object,
    *,
    replace_existing: bool = True,
    maximum_bytes: int = MAX_MESSAGE_BYTES,
    replace_timeout_seconds: float = 0.0,
    replace_retry_seconds: float = 0.01,
) -> None:
    if type(replace_existing) is not bool:
        raise ValueError("RUNTIME_CONTROL_REPLACE_MODE_INVALID")
    if any(
        type(value) not in (int, float) or not 0 <= value <= 3600
        for value in (replace_timeout_seconds, replace_retry_seconds)
    ):
        raise ValueError("RUNTIME_CONTROL_RETRY_BUDGET_INVALID")
    # 不先做 JSON 模型序列化，以免 NaN 先变成 null 而掩盖损坏原值。
    plugin_json_value(payload, limit=maximum_bytes)
    rendered = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    ) + "\n"
    if len(rendered.encode("utf-8")) > maximum_bytes:
        raise ValueError("RUNTIME_CONTROL_EVIDENCE_TOO_LARGE")
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    owned = None
    try:
        check_plain_plugin_path(temporary)
        # 固定换行编码，使 Windows 实际写入字节数与上面的 UTF-8 预算一致。
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            owned = os.fstat(stream.fileno())
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + replace_timeout_seconds
        if not math.isfinite(deadline):
            raise ValueError("RUNTIME_CONTROL_RETRY_CLOCK_INVALID")
        while True:
            # 每次重试均复核身份；等待读锁不能成为接纳被替换暂存文件的窗口。
            check_plain_plugin_path(path)
            check_plain_plugin_path(temporary)
            if not os.path.samestat(owned, temporary.stat()):
                raise ValueError("RUNTIME_CONTROL_TEMPORARY_REPLACED")
            try:
                if replace_existing:
                    temporary.replace(path)
                else:
                    os.link(temporary, path)
                break
            except PermissionError:
                remaining = deadline - time.monotonic()
                if not math.isfinite(remaining) or remaining <= 0:
                    raise
                time.sleep(min(remaining, max(0.001, replace_retry_seconds)))
                if time.monotonic() >= deadline:
                    raise
    finally:
        if owned is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(owned, temporary.stat()):
                    temporary.unlink()
