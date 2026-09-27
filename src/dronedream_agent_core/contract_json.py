"""Bounded JSON contract decoding for persisted execution artifacts."""

from typing import TypeVar

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import decode_json

Contract = TypeVar("Contract", bound=BaseModel)


# 功能：
#   1. 检查 JSON 字节、节点、深度、重复键及非有限数，再按契约读取同一份原始内容。
#   2. 保留 JSON 中时间字符串等类型的合法表示，不放宽严格字段，不改写已绑定的计划。
# 输入：
#   raw：已经从有界文件读取器取得的 JSON 文本或字节。
#   contract：期望的契约模型类。
#   limit：最大 UTF-8 字节数。
#   node_limit：最大 JSON 节点数。
# 输出：
#   artifact：通过 JSON 边界与模型严格校验的契约对象。
def decode_contract_json(raw: str | bytes, contract: type[Contract], *, limit: int, node_limit: int) -> Contract:
    decode_json(raw, limit=limit, node_limit=node_limit)
    # 不能把 decode_json 的字典交给 model_validate：严格 datetime 字段在 Python 模式
    # 只接受 datetime 对象，而持久化 JSON 只能用时间字符串表示。
    artifact = contract.model_validate_json(raw, strict=True)
    return artifact
