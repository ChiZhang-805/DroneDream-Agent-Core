"""Immutable, bounded native captures; decode only after confirmed landing.

Keeping complete Pydantic/dict histories for a whole flight grows the cyclic
collector's object graph even when no garbage is found. JSON bytes preserve
the original numeric values and timestamps without retaining those containers.
Packing performs no reward computation, disk I/O, compression or control.
"""

import hashlib
from dataclasses import dataclass

from pydantic import TypeAdapter

from dronedream_plugin_sdk.protocol import decode_json

from .native_transition import CapturedNativeTransition

MAXIMUM_CAPTURE_BYTES = 4 * 1024 * 1024
MAXIMUM_ARCHIVE_BYTES = 256 * 1024 * 1024
MAXIMUM_CAPTURE_JSON_NODES = 1_000_000
_CAPTURE = TypeAdapter(CapturedNativeTransition)


@dataclass(frozen=True)
class PackedNativeTransition:
    """Immutable encoded capture plus content identity, not a qualification receipt."""
    content: bytes
    sha256: str

    # 功能：
    #   核对捕获大小与摘要，再严格解析和验证完整原生转移；不授予落地或飞行资格。
    # 输入：
    #   self：由不可变字节与对应摘要组成的捕获包。
    # 输出：
    #   capture：类型与数值契约验证后的原生转移。
    def unpack(self) -> CapturedNativeTransition:
        if type(self.content) is not bytes or not 0 < len(self.content) <= MAXIMUM_CAPTURE_BYTES:
            raise ValueError("NATIVE_CAPTURE_SIZE_INVALID")
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise ValueError("NATIVE_CAPTURE_CONTENT_CHANGED")
        # 深度／节点和重复键检查在离线解包执行，不加到每次实时捕获的序列化路径。
        decode_json(self.content, limit=MAXIMUM_CAPTURE_BYTES,
                    node_limit=MAXIMUM_CAPTURE_JSON_NODES)
        # 继续使用 JSON 类型验证，以保留冻结 dataclass 在 JSON 与 Python 下的解析区别。
        capture = _CAPTURE.validate_json(self.content)
        return capture


# 功能：
#   将调用方已验证的捕获序列化为有界不可变字节，不计算奖励、不写磁盘、不控制飞机。
# 输入：
#   capture：原生观测、提案、执行回执与独立见证组成的转移。
# 输出：
#   packed：序列化内容与实际 SHA-256 摘要组成的捕获包。
def pack_native_transition(capture: CapturedNativeTransition) -> PackedNativeTransition:
    if type(capture) is not CapturedNativeTransition:
        raise ValueError("NATIVE_CAPTURE_TYPE_INVALID")
    content = _CAPTURE.dump_json(capture, warnings="error")
    if not 0 < len(content) <= MAXIMUM_CAPTURE_BYTES:
        raise ValueError("NATIVE_CAPTURE_SIZE_INVALID")
    packed = PackedNativeTransition(content, hashlib.sha256(content).hexdigest())
    return packed
