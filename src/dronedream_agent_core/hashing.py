"""Canonical hashes used to bind artifacts and evidence without hidden state."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel


# 功能：
#   按既有制品字节契约规范模型、时间和容器，保留字典键字符串化的历史行为。
# 输入：
#   value：准备生成规范表示的业务值。
# 输出：
#   normalized：规范化值；未支持的对象保持原值，由后续 JSON 编码拒绝。
def json_value(value: Any) -> Any:
    # 原生标量是传感器数据的高频情况；精确类型判断不改变自定义子类的既有转换路径。
    kind = type(value)
    if value is None or kind is float or kind is int or kind is str or kind is bool:
        normalized = value
        return normalized
    if isinstance(value, BaseModel):
        # 标准 Pydantic JSON 模式已递归转换且返回新容器，无需再次遍历整个张量结构。
        dumped = value.model_dump(mode="json")
        if type(value).model_dump is BaseModel.model_dump:
            normalized = dumped
        else:
            # 重写的 model_dump 可能忽略 JSON 模式，需要保留原有递归处理。
            normalized = json_value(dumped)
    elif isinstance(value, datetime):
        normalized = value.isoformat()
    elif isinstance(value, dict):
        # 此处为历史摘要兼容而转字符串；安全边界必须在之前拒绝非字符串键和转换冲突。
        normalized = {str(key): json_value(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        normalized = [json_value(item) for item in value]
    else:
        normalized = value
    return normalized


# 功能：
#   对规范化值使用固定键顺序与紧凑 JSON 格式，保留 Unicode 并拒绝非有限浮点数。
# 输入：
#   value：准备编码为规范 JSON 的业务值。
# 输出：
#   encoded：可按 UTF-8 生成稳定字节的 JSON 文本。
def canonical_json(value: Any) -> str:
    encoded = json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded


# 功能：
#   计算规范 JSON 的 SHA-256 内容摘要，不据此判断作者可信、授权成立或执行成功。
# 输入：
#   value：准备绑定内容身份的业务值。
# 输出：
#   digest：规范 UTF-8 字节对应的十六进制 SHA-256 摘要。
def sha256_json(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest
