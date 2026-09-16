"""ROS 运行包独立部署时使用的有界 JSON 读取与独占中止发布。"""

import json
import math
import os
import tempfile
from pathlib import Path

MAX_JSON_BYTES = 4 * 1024 * 1024


# 功能：
#   拒绝重复 JSON 字段，避免同一控制记录在不同读取器中得到不同解释。
# 输入：
#   pairs：解析器按原顺序返回的字段与值。
# 输出：
#   result：字段唯一的字典。
def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ROS_JSON_DUPLICATE_FIELD")
        result[key] = value
    return result


# 功能：
#   限制输入字节、嵌套层数与节点数，拒绝非有限数字及非对象根。
# 输入：
#   raw：来自文件或服务请求的 UTF-8 JSON 字节或文本。
# 输出：
#   payload：经过边界校验的 JSON 对象。
def decode_object(raw: bytes | str) -> dict:
    if not isinstance(raw, (bytes, str)) or len(raw) > MAX_JSON_BYTES:
        raise ValueError("ROS_JSON_SIZE_INVALID")
    if isinstance(raw, str) and len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("ROS_JSON_SIZE_INVALID")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_fields)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("ROS_JSON_ENCODING_OR_DEPTH_INVALID") from error
    if not isinstance(payload, dict):
        raise ValueError("ROS_JSON_OBJECT_REQUIRED")
    pending = [(payload, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > 64 or count > 100_000:
            raise ValueError("ROS_JSON_COMPLEXITY_EXCEEDED")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("ROS_JSON_NONFINITE")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
    return payload


# 功能：
#   有界读取一次运行记录，不允许无限文件消耗内存；内容交给统一 JSON 边界校验。
# 输入：
#   path：当前运行的具体记录路径。
# 输出：
#   payload：校验后的 JSON 对象。
def read_object(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    payload = decode_object(raw)
    return payload


# 功能：
#   原子、独占地发布中止记录，保留已有操作员或其他安全组件的中止原因。
# 输入：
#   path：本次执行器的中止文件路径。
#   payload：标准 JSON 中止记录。
# 输出：
#   published：本调用首次发布时为真，已有中止记录时为假。
def publish_abort(path: Path, payload: dict) -> bool:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
    decode_object(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        current = temporary.lstat()
        if temporary.is_symlink() or (current.st_dev, current.st_ino) != (
            identity.st_dev,
            identity.st_ino,
        ):
            raise ValueError("ROS_ABORT_TEMPORARY_REPLACED")
        try:
            # 同目录硬链接是不可覆盖的完整文件发布；目标存在时绝不改写原原因。
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return False
        published = True
        return published
    finally:
        try:
            current = temporary.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                temporary.unlink()
        except FileNotFoundError:
            pass
