"""JSON 编解码与逐帧读取的预算边界；不启动插件或连接网络。"""

import io
import json

import pytest

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json, read_frame


# 功能：
#   验证紧凑 JSON 恰好占满字节预算时仍可编码、解析和复制，防止容器标点重复计数。
# 输入：
#   value：包含空容器、单元素容器、嵌套对象或多字节字符的原始值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [[0], [[], []], {"x": [0]}, ["中"], [], {}, True, None])
def test_exact_wire_budget_accepts_complete_value(value):
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    limit = len(rendered.encode("utf-8"))
    assert encode_json(value, limit=limit) == rendered
    assert decode_json(rendered, limit=limit) == value
    assert decode_json(rendered.encode("utf-8"), limit=limit) == value
    assert copy_json(value, limit=limit) == value


# 功能：
#   验证容量不足一字节时三个 JSON 入口都拒绝值，避免修正预估后放松最终字节校验。
# 输入：
#   value：序列化后至少占两字节的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [[0], [[], []], {"x": [0]}, ["中"], [], {}, True, None])
def test_wire_budget_rejects_one_byte_overflow(value):
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    limit = len(rendered.encode("utf-8")) - 1
    for codec, argument in ((encode_json, value), (decode_json, rendered), (copy_json, value)):
        with pytest.raises(ValueError, match="PLUGIN_JSON_SIZE_LIMIT"):
            codec(argument, limit=limit)


# 功能：
#   验证所有 JSON 入口拒绝无效预算，避免布尔值、浮点数或无限值被当作字节上限。
# 输入：
#   limit：类型错误、非正数或超出宿主硬上限的预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "limit", [None, True, False, -2, -1, 0, 1.5, float("inf"), "64", 64 * 1024 * 1024 + 1]
)
def test_json_budget_rejects_invalid_limits(limit):
    for codec, argument in ((encode_json, 0), (decode_json, "0"), (copy_json, 0)):
        with pytest.raises(ValueError, match="JSON_BYTE_BUDGET_INVALID"):
            codec(argument, limit=limit)


# 功能：
#   验证逐帧入口在触碰输入流之前拒绝无效预算，尤其禁止负数变成 readline 的无限读取。
# 输入：
#   limit：不允许交给输入流的预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "limit", [None, True, False, -2, -1, 0, 1.5, float("inf"), "64", 64 * 1024 * 1024 + 1]
)
def test_frame_budget_is_checked_before_reading(limit):
    class UntouchedStream:
        # 功能：
        #   若预算校验前发生读取就直接使测试失败，不实际分配或读取超大内容。
        # 输入：
        #   size：被测代码传递的读取上限。
        # 输出：
        #   None：不返回业务数据。
        def readline(self, size):
            pytest.fail(f"invalid budget reached stream: {size!r}")

    with pytest.raises(ValueError, match="JSON_BYTE_BUDGET_INVALID"):
        read_frame(UntouchedStream(), limit=limit)


# 功能：
#   验证文本流与二进制管道均按 UTF-8 字节计算完整帧，且不吞掉下一帧。
# 输入：
#   binary：是否使用带二进制缓冲的文本包装流。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("binary", [False, True])
def test_frame_budget_preserves_utf8_and_next_frame(binary):
    payload = '"中"\n0\n'
    stream = (
        io.TextIOWrapper(io.BytesIO(payload.encode("utf-8")), encoding="ascii")
        if binary
        else io.StringIO(payload)
    )
    with stream:
        assert read_frame(stream, limit=6) == '"中"\n'
        assert read_frame(stream, limit=2) == "0\n"
        assert read_frame(stream, limit=2) is None
