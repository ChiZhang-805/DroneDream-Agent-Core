"""Single-read identities for bounded offline training evidence objects."""

import hashlib
from io import BytesIO
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..plugin_files import portable_plugin_path, read_plugin_file

MAX_TRAINING_DATASET_BYTES = 256 * 1024 * 1024
MAX_TRAINING_ROW_BYTES = 4 * 1024 * 1024
MAX_TRAINING_ROWS = 250_000


# 功能：
#   有界复制训练记录，将物理计算使用的元组转为 JSON 数组，不调用自定义转换或迭代器。
# 输入：
#   value：由标准 Python 字典、列表、元组和 JSON 标量组成的记录。
# 输出：
#   payload：容器独立且仅含 JSON 类型的记录，数值及实际编码大小由后续编码器继续检查。
def training_json_value(value):
    remaining = 1_000_000

    # 功能：
    #   在分配子容器前约束节点与嵌套深度，拒绝错误键和不能确定序列化语义的对象。
    # 输入：
    #   item：当前节点。
    #   depth：从记录根开始的嵌套层数。
    # 输出：
    #   copied：当前节点对应的独立 JSON 值。
    def copy_value(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise ValueError("TRAINING_DATASET_COMPLEXITY_LIMIT")
        kind = type(item)
        if kind is dict:
            if len(item) * 2 > remaining or any(type(key) is not str for key in item):
                raise ValueError("TRAINING_DATASET_KEYS_OR_COMPLEXITY_INVALID")
            copied = {copy_value(key, depth + 1): copy_value(part, depth + 1)
                      for key, part in item.items()}
        elif kind in (list, tuple):
            if len(item) > remaining:
                raise ValueError("TRAINING_DATASET_COMPLEXITY_LIMIT")
            copied = [copy_value(part, depth + 1) for part in item]
        elif kind in (str, int, float, bool) or item is None:
            if kind is str and len(item) > MAX_TRAINING_ROW_BYTES:
                raise ValueError("TRAINING_DATASET_ROW_OVERSIZED")
            copied = item
        else:
            raise ValueError("TRAINING_DATASET_VALUE_TYPE_INVALID")
        return copied

    payload = copy_value(value, 0)
    return payload


# 功能：
#   有界读取同一普通文件并严格解析对象，拒绝歧义字段、符号链接及读取期间替换。
# 输入：
#   path：离线读取的证据路径。
#   limit：允许的最大文件字节数。
# 输出：
#   payload：解析且验证 JSON 结构后的对象。
#   digest：实际读取原始字节的摘要。
def read_evidence_object(path: Path, *, limit: int = 2 * 1024 * 1024):
    content = read_plugin_file(path, limit=limit)
    payload = decode_json(content, limit=limit, node_limit=1_000_000)
    if type(payload) is not dict:
        raise ValueError("TRAINING_EVIDENCE_NOT_OBJECT")
    digest = hashlib.sha256(content).hexdigest()
    return payload, digest


# 功能：
#   从已固定的有界字节解析完整 JSONL 对象，拒绝空白行、缺少换行、重复键和非有限值。
# 输入：
#   content：整个证据文件的不可变字节，允许空文件表示没有此类可选记录。
# 输出：
#   rows：至多二十五万条、每条至多 4 MiB 的严格对象列表。
def decode_evidence_rows(content: bytes) -> list[dict]:
    if type(content) is not bytes or len(content) > MAX_TRAINING_DATASET_BYTES:
        raise ValueError("TRAINING_DATASET_BYTES_INVALID")
    rows = []
    with BytesIO(content) as stream:
        while line := stream.readline(MAX_TRAINING_ROW_BYTES + 1):
            if (len(rows) >= MAX_TRAINING_ROWS or len(line) > MAX_TRAINING_ROW_BYTES
                    or not line.endswith(b"\n") or not line.strip()):
                raise ValueError("TRAINING_DATASET_ROW_INCOMPLETE_OR_OVERSIZED")
            row = decode_json(line, limit=MAX_TRAINING_ROW_BYTES, node_limit=1_000_000)
            if type(row) is not dict:
                raise ValueError("TRAINING_DATASET_ROW_NOT_OBJECT")
            rows.append(row)
    return rows


# 功能：
#   对照完整文件清单读取同一批原始字节，核对摘要并限制所有文件合计占用。
# 输入：
#   root：数据集根目录。
#   files：固定逻辑类别到规范相对文件名的映射。
#   expected：同一完整类别集合的原始字节摘要。
#   error_prefix：调用方用于区分数据集的错误前缀。
# 输出：
#   contents：合计至多 256 MiB、与清单摘要一致的不可变文件字节映射。
def read_evidence_dataset(root: Path, files: dict[str, str], expected, *, error_prefix: str):
    if type(expected) is not dict or set(expected) != set(files):
        raise ValueError(error_prefix + ":INVENTORY")
    remaining = MAX_TRAINING_DATASET_BYTES
    contents = {}
    for name, relative in files.items():
        portable_plugin_path(relative)
        digest = expected[name]
        if (type(digest) is not str or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(error_prefix + ":" + name)
        content = read_plugin_file(root / relative, limit=remaining)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(error_prefix + ":" + name)
        contents[name] = content
        remaining -= len(content)
    return contents
