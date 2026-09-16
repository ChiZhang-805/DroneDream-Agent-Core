"""One bounded, loss-aware conversion boundary for local and external hooks."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import (
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_MESSAGE_BYTES,
    copy_json,
    encode_json,
)


# 功能：
#   1. 将插件钩子数据转换为独立 JSON 值，显式处理路径、时间、枚举和不可变集合。
#   2. 拒绝未知对象和非字符串键，不使用可能泄露敏感信息的 repr 兜底。
#   3. 限制递归与节点规模，统一拒绝非有限数值，并使集合的传输顺序可重复。
# 输入：
#   value：宿主或插件传入的待转换对象。
#   limit：宿主明确选择的字节上限；默认消息预算不因大型本机制品而扩大。
# 输出：
#   detached：转换、预算校验并隔离可变引用后的 JSON 值。
def plugin_json_value(value: Any, *, limit: int = MAX_MESSAGE_BYTES) -> Any:
    # 先复用协议预算检查，再遍历对象，错误预算不得触发模型序列化。
    encode_json(0, limit=limit)
    remaining = MAX_JSON_NODES

    # 功能：
    #   递归转换受支持的对象，每次访问扣减共享预算，阻止循环或过深输入无限展开。
    # 输入：
    #   item：当前待转换的节点。
    #   depth：当前节点的递归深度。
    # 输出：
    #   converted：当前节点转换后的 JSON 兼容值。
    def convert(item: Any, depth: int) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            raise ValueError("PLUGIN_EXTENSION_INPUT_TOO_COMPLEX")
        if isinstance(item, BaseModel):
            # Python 模式保留 NaN 等非法值，不能先变成 null 后绕过严格数值检查。
            try:
                dumped = item.model_dump(mode="python")
            except (TypeError, ValueError) as error:
                raise ValueError("PLUGIN_EXTENSION_INPUT_SERIALIZATION_FAILED") from error
            converted = convert(dumped, depth + 1)
        elif isinstance(item, Enum):
            converted = convert(item.value, depth + 1)
        elif isinstance(item, Path):
            converted = str(item)
        elif isinstance(item, (datetime, date)):
            converted = item.isoformat()
        elif type(item) is dict:
            if len(item) * 2 > remaining:
                raise ValueError("PLUGIN_EXTENSION_INPUT_TOO_COMPLEX")
            if any(type(key) is not str for key in item):
                raise ValueError("PLUGIN_EXTENSION_INPUT_NON_STRING_KEY")
            converted = {key: convert(child, depth + 1) for key, child in item.items()}
        elif type(item) in (list, tuple, set, frozenset):
            if len(item) > remaining:
                raise ValueError("PLUGIN_EXTENSION_INPUT_TOO_COMPLEX")
            children = [convert(child, depth + 1) for child in item]
            converted = (
                sorted(children, key=lambda child: encode_json(child, limit=limit))
                if isinstance(item, (set, frozenset))
                else children
            )
        elif type(item) in (str, int, float, bool) or item is None:
            converted = item
        else:
            raise ValueError(f"PLUGIN_EXTENSION_INPUT_NOT_JSON:{type(item).__name__}")
        return converted

    try:
        detached = copy_json(convert(value, 0), limit=limit)
    except RecursionError as error:
        raise ValueError("PLUGIN_EXTENSION_INPUT_TOO_COMPLEX") from error
    return detached
