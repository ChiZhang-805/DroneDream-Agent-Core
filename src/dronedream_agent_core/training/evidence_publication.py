"""Complete, no-replace training evidence publication shared by runtime and CLI."""

import os
import tempfile
from contextlib import suppress
from pathlib import Path

from dronedream_plugin_sdk.protocol import encode_json

from ..plugin_files import check_plain_plugin_path
from .evidence_files import MAX_TRAINING_ROW_BYTES, training_json_value


# 功能：
#   1. 通过独占暂存描述符写入并同步全部字节，再无覆盖发布为新证据文件。
#   2. 失败只清理仍属于本次操作的暂存身份，路径检查不替代恶意进程隔离。
# 输入：
#   path：父目录已存在、目标尚不存在的证据路径。
#   content：已由调用方验证语义的不可变字节。
#   limit：最多 512 MiB 的本文件字节预算。
# 输出：
#   None：不返回业务数据。
def publish_evidence_bytes(path: Path, content: bytes, *, limit: int) -> None:
    if (
        not isinstance(path, Path)
        or type(content) is not bytes
        or type(limit) is not int
        or not 0 < limit <= 512 * 1024 * 1024
        or len(content) > limit
    ):
        raise ValueError("TRAINING_EVIDENCE_BYTES_INVALID")
    path = path.absolute()
    check_plain_plugin_path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = identity = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + path.name + ".", suffix=".pending", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            identity = os.fstat(handle.fileno())
            if handle.write(content) != len(content):
                raise OSError("TRAINING_EVIDENCE_SHORT_WRITE")
            handle.flush()
            os.fsync(handle.fileno())
        check_plain_plugin_path(temporary)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("TRAINING_EVIDENCE_STAGING_CHANGED")
        check_plain_plugin_path(path)
        os.link(temporary, path)
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


# 功能：
#   严格编码有界训练对象并调用共享无覆盖发布器，拒绝歧义类型或非有限数值。
# 输入：
#   path：父目录已存在的新文件路径。
#   value：最多 4 MiB 的标准训练记录对象，元组会明确转为数组。
# 输出：
#   None：不返回业务数据。
def write_evidence_object(path: Path, value: dict) -> None:
    if type(value) is not dict:
        raise ValueError("TRAINING_EVIDENCE_OBJECT_INVALID")
    content = encode_json(
        training_json_value(value), limit=MAX_TRAINING_ROW_BYTES, node_limit=1_000_000
    ).encode("utf-8")
    publish_evidence_bytes(path, content, limit=MAX_TRAINING_ROW_BYTES)
