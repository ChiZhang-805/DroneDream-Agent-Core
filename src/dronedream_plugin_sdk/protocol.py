"""Bounded JSON shared by the SDK and its host; not an execution sandbox.

JSON Schema describes a tool's values, but does not bound a stdio frame, stop
non-finite numbers, or detach mutable dictionaries. These checks precede schema
validation on both sides of the process boundary.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, TextIO

MAX_MESSAGE_BYTES = 2 * 1024 * 1024
# 宿主资产索引可申请大于消息帧的预算，但不得把该入口变成无限制 JSON 读取器。
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
# 只缓存小型声明的成功检查；不保留可变 Schema，也不让 2 MiB 消息占满缓存。
MAX_CACHED_SCHEMA_BYTES = 16 * 1024
MAX_CACHED_SCHEMAS = 128


# 功能：
#   在分配缓冲或读取输入前验证字节预算，禁止布尔值、非整数、非正数及超大上限。
# 输入：
#   limit：调用方申请的 UTF-8 字节数上限。
# 输出：
#   None：不返回业务数据。
def _validate_byte_limit(limit: int) -> None:
    if type(limit) is not int or not 0 < limit <= MAX_JSON_BYTES:
        raise ValueError("JSON_BYTE_BUDGET_INVALID")


# 功能：
#   1. 迭代检查 JSON 类型、有限数值、嵌套深度和节点预算，不调用对象自定义序列化。
#   2. 先计算字节下界以尽早拒绝过大内容，再按完整 UTF-8 编码校验实际大小。
#   3. 重复引用按每次传输计数；循环引用在深度上限处拒绝，不保留对象身份。
# 输入：
#   value：待编码的标准 JSON 值。
#   limit：允许的 UTF-8 字节数；默认使用插件消息预算。
#   node_limit：宿主设置的节点上限，外部 RPC 不能自行扩大该值。
# 输出：
#   rendered：验证通过的紧凑 JSON 字符串。
def encode_json(
    value: object, *, limit: int = MAX_MESSAGE_BYTES, node_limit: int = MAX_JSON_NODES
) -> str:
    _validate_byte_limit(limit)
    if type(node_limit) is not int or not 0 < node_limit <= 2_000_000:
        raise ValueError("JSON_NODE_BUDGET_INVALID")
    pending = [(value, 0)]
    nodes = 0
    minimum_bytes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > node_limit or depth > MAX_JSON_DEPTH:
            raise ValueError("PLUGIN_JSON_COMPLEXITY_LIMIT")
        item_type = type(item)
        if item_type is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("PLUGIN_JSON_KEY_INVALID")
            if len(item) * 2 + len(pending) + nodes > node_limit:
                raise ValueError("PLUGIN_JSON_COMPLEXITY_LIMIT")
            pending.extend((part, depth + 1) for pair in item.items() for part in pair)
            # 花括号、每对键值间的冒号，以及相邻键值对间的逗号分别只计一次。
            minimum_bytes += 2 + len(item) + max(0, len(item) - 1)
        elif item_type is list:
            if len(item) + len(pending) + nodes > node_limit:
                raise ValueError("PLUGIN_JSON_COMPLEXITY_LIMIT")
            pending.extend((part, depth + 1) for part in item)
            # n 个元素只有 n-1 个逗号；多算一个会拒绝恰好占满预算的合法数组。
            minimum_bytes += 2 + max(0, len(item) - 1)
        elif item_type is str:
            # 先用字符数拒绝明显超限的字符串，避免为它再分配完整 UTF-8 副本。
            if len(item) > limit:
                raise ValueError("PLUGIN_JSON_SIZE_LIMIT")
            minimum_bytes += len(item.encode("utf-8")) + 2
        elif item_type is float:
            if not math.isfinite(item):
                raise ValueError("PLUGIN_JSON_NUMBER_INVALID")
            minimum_bytes += 1
        elif item_type is int:
            if item.bit_length() > 13_000:
                raise ValueError("PLUGIN_JSON_NUMBER_INVALID")
            minimum_bytes += 1
        elif item is None or item_type is bool:
            minimum_bytes += 1
        else:
            raise ValueError("PLUGIN_JSON_TYPE_INVALID")
        if minimum_bytes > limit:
            raise ValueError("PLUGIN_JSON_SIZE_LIMIT")
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(rendered.encode("utf-8")) > limit:
        raise ValueError("PLUGIN_JSON_SIZE_LIMIT")
    return rendered


# 功能：
#   在解析对象时拒绝重复键，避免同一文档被不同校验器或执行器理解成不同内容。
# 输入：
#   pairs：解析器按原文顺序提供的键值对。
# 输出：
#   result：没有重复键的字典。
def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("PLUGIN_JSON_DUPLICATE_KEY")
        result[key] = value
    return result


# 功能：
#   拒绝 Python JSON 解析器默认允许的 NaN、Infinity 等非标准常量。
# 输入：
#   _value：解析器识别出的非标准常量文本。
# 输出：
#   None：不返回业务数据。
def _invalid_constant(_value: str) -> None:
    raise ValueError("PLUGIN_JSON_NUMBER_INVALID")


# 功能：
#   1. 在解析前检查输入字节预算，严格解码 UTF-8 并拒绝重复键与非有限数值。
#   2. 复用出站校验约束解析后的类型、节点和深度，避免进出站规则不一致。
# 输入：
#   value：包含一个完整 JSON 值的字符串或字节串。
#   limit：允许的 UTF-8 字节数上限。
#   node_limit：允许的 JSON 节点数上限。
# 输出：
#   parsed：校验通过的 Python JSON 值。
def decode_json(
    value: str | bytes, *, limit: int = MAX_MESSAGE_BYTES, node_limit: int = MAX_JSON_NODES
) -> Any:
    _validate_byte_limit(limit)
    if not isinstance(value, (str, bytes)):
        raise ValueError("PLUGIN_JSON_INPUT_TYPE_INVALID")
    if type(node_limit) is not int or not 0 < node_limit <= 2_000_000:
        raise ValueError("JSON_NODE_BUDGET_INVALID")
    if len(value) > limit or (isinstance(value, str) and len(value.encode("utf-8")) > limit):
        raise ValueError("PLUGIN_JSON_SIZE_LIMIT")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    try:
        parsed = json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
        encode_json(parsed, limit=limit, node_limit=node_limit)
    except RecursionError as error:
        raise ValueError("PLUGIN_JSON_COMPLEXITY_LIMIT") from error
    return parsed


# 功能：
#   校验并复制 JSON 值，使跨调用边界的数据不再共享可变字典或列表。
# 输入：
#   value：待隔离的标准 JSON 值。
#   limit：允许的 UTF-8 字节数上限。
# 输出：
#   detached：不再引用原始可变容器的 JSON 值。
def copy_json(value: Any, *, limit: int = MAX_MESSAGE_BYTES) -> Any:
    detached = json.loads(encode_json(value, limit=limit))
    return detached


# 功能：
#   1. 限量读取单帧，管道直接按 UTF-8 字节读取，不依赖 Windows 当前代码页。
#   2. 文本测试流先按字符限量，再核对字节数；超限后由调用方关闭通道，不能执行剩余碎片。
# 输入：
#   stream：插件标准输入或具备 readline 的文本流。
#   limit：允许的帧字节数，包含行尾换行符。
# 输出：
#   line：读取并解码的帧字符串；流结束时为 None。
def read_frame(stream: TextIO, *, limit: int = MAX_MESSAGE_BYTES) -> str | None:
    _validate_byte_limit(limit)
    binary = getattr(stream, "buffer", None)
    line = binary.readline(limit + 1) if binary is not None else stream.readline(limit + 1)
    if line in ("", b""):
        line = None
        return line
    if len(line) > limit:
        raise ValueError("PLUGIN_PROTOCOL_MESSAGE_TOO_LARGE")
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="strict")
    if len(line.encode("utf-8")) > limit:
        raise ValueError("PLUGIN_PROTOCOL_MESSAGE_TOO_LARGE")
    return line


# 功能：
#   校验请求标识，保留字符串与整数的区别，拒绝布尔值、小数及超出精确表示范围的整数。
# 输入：
#   value：JSON-RPC 请求或响应中的候选标识。
# 输出：
#   valid：标识满足本协议约束时为 True，否则为 False。
def valid_request_id(value: object) -> bool:
    valid = (type(value) is str and 0 < len(value) <= 160) or (
        type(value) is int and abs(value) <= 2**53 - 1
    )
    return valid


# 功能：
#   1. 复制并验证 JSON Schema，只允许文档内部引用，禁止验证时远程取回外部定义。
#   2. 对内容和校验器均未变化的小型声明复用成功检查，每次仍检查预算并返回独立副本。
#   3. 此检查不验证具体工具参数，也不约束正则或递归的执行成本，调用方仍需相关检查。
# 输入：
#   schema：插件声明的工具输入或输出 Schema。
# 输出：
#   detached：独立复制且引用范围检查通过的 Schema。
def validate_local_schema(schema: dict[str, Any]) -> dict[str, Any]:
    import jsonschema

    rendered = encode_json(schema)
    detached = json.loads(rendered)
    validator = jsonschema.validators.validator_for(detached)
    if len(rendered.encode("utf-8")) <= MAX_CACHED_SCHEMA_BYTES:
        _check_cached_schema(rendered, validator)
    else:
        _check_schema(detached, validator)
    return detached


# 功能：
#   1. 按完整 JSON 内容和当前选中的校验器缓存成功检查，失败结果不会进入缓存。
#   2. 只缓存不可变文本与空返回值，不把某次调用修改过的字典传给后续调用。
# 输入：
#   rendered：通过 JSON 预算检查且符合缓存长度上限的 Schema 文本。
#   validator：按 Schema 草案选中的当前校验器类。
# 输出：
#   None：不返回业务数据。
@lru_cache(maxsize=MAX_CACHED_SCHEMAS)
def _check_cached_schema(rendered: str, validator: type) -> None:
    _check_schema(json.loads(rendered), validator)


# 功能：
#   1. 检查 Schema 语法以及所有承载 Schema 的关键字中的引用范围。
#   2. 允许示例数据和名为 $ref 的业务属性，但禁止真正引用外部定义。
# 输入：
#   detached：与调用方可变对象隔离的 Schema。
#   validator：按 Schema 草案选中的校验器类。
# 输出：
#   None：不返回业务数据。
def _check_schema(detached: dict[str, Any], validator: type) -> None:
    validator.check_schema(detached)
    pending: list[object] = [detached]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                value = item.get(key)
                if key in item and (
                    not isinstance(value, str) or (value and not value.startswith("#"))
                ):
                    raise ValueError("PLUGIN_SCHEMA_EXTERNAL_REFERENCE")
            # 只遍历每个属性的 Schema，不把属性名本身当成 Schema 关键字。
            for key in (
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
                "dependencies",
            ):
                mapping = item.get(key)
                if isinstance(mapping, dict):
                    pending.extend(value for value in mapping.values() if isinstance(value, dict))
            for key in (
                "items",
                "additionalItems",
                "additionalProperties",
                "contains",
                "not",
                "if",
                "then",
                "else",
                "propertyNames",
                "unevaluatedItems",
                "unevaluatedProperties",
                "contentSchema",
                "allOf",
                "anyOf",
                "oneOf",
                "prefixItems",
            ):
                value = item.get(key)
                if isinstance(value, (dict, list)):
                    pending.append(value)
        elif isinstance(item, list):
            pending.extend(item)
