"""Host receipt clock boundaries; these tests do not synchronize hardware clocks."""

import pytest

from dronedream_agent_core.source_clock import source_received_at_unix_ms


# 功能：
#   拒绝非法采样年龄，不能因巨大整数溢出而绕过统一的来源错误处理。
# 输入：
#   age：错误类型、非有限值或超过浮点范围的年龄。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("age", [10**400, True, None, "0", float("nan"), float("inf")])
def test_source_age_has_a_stable_invalid_value_error(age):
    with pytest.raises(ValueError, match="SOURCE_SAMPLE_AGE_INVALID"):
        source_received_at_unix_ms(
            {"sample_age_seconds": age}, collected_at_unix_ms=1000,
            now_unix_ms=1020, maximum_age_ms=250,
        )


# 功能：
#   在时间运算前拒绝无效调用参数，布尔值和浮点数不能充当毫秒整数。
# 输入：
#   field：被修改的时间参数。
#   value：违反参数契约的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("collected_at_unix_ms", True), ("collected_at_unix_ms", 1000.0),
    ("collected_at_unix_ms", None), ("collected_at_unix_ms", -1),
    ("now_unix_ms", 1020.0), ("now_unix_ms", None), ("now_unix_ms", 2**63),
    ("maximum_age_ms", True), ("maximum_age_ms", 250.0),
    ("maximum_age_ms", None), ("maximum_age_ms", 0), ("maximum_age_ms", 2**63),
])
def test_clock_arguments_are_strict_bounded_integers(field, value):
    arguments = {"collected_at_unix_ms": 1000, "now_unix_ms": 1020, "maximum_age_ms": 250}
    arguments[field] = value
    with pytest.raises(ValueError, match="SOURCE_.*INVALID"):
        source_received_at_unix_ms({"sample_age_seconds": 0}, **arguments)


# 功能：
#   拒绝不是对象的观测包和显式为空的接收时刻，不能将损坏字段当成旧格式缺省。
# 输入：
#   packet：损坏的观测包。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("packet", [None, [], {"sample_age_seconds": 0,
                                            "received_at_unix_ms": None}])
def test_invalid_packet_does_not_gain_a_fallback_timestamp(packet):
    with pytest.raises(ValueError, match="SOURCE_.*INVALID"):
        source_received_at_unix_ms(
            packet, collected_at_unix_ms=1000, now_unix_ms=1020, maximum_age_ms=250,
        )


# 功能：
#   保留确实没有接收字段的历史观测，按向过去取整计算年龄，不刷新缓存时间。
# 输入：
#   age：已记录的源年龄秒数。
#   expected：保守恢复的主机接收毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("age,expected", [(0, 1000), (.0001, 999), (.0101, 989), (.25, 750)])
def test_age_only_record_is_conservatively_rounded(age, expected):
    received = source_received_at_unix_ms(
        {"sample_age_seconds": age}, collected_at_unix_ms=1000,
        now_unix_ms=1000, maximum_age_ms=250,
    )
    assert received == expected


# 功能：
#   同时核对显式接收时刻和报告年龄，任一已过期都不能通过；未来时刻也拒绝。
# 输入：
#   packet：带显式接收时刻的观测。
#   collected：缓存汇集时刻。
#   now：消费时刻。
#   code：预期拒绝类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("packet,collected,now,code", [
    ({"sample_age_seconds": 0, "received_at_unix_ms": 1000}, 1400, 1400, "EXPIRED"),
    ({"sample_age_seconds": .4, "received_at_unix_ms": 1400}, 1400, 1400, "EXPIRED"),
    ({"sample_age_seconds": 0, "received_at_unix_ms": 1000}, 1000, 999, "FUTURE"),
    ({"sample_age_seconds": 0, "received_at_unix_ms": 1001}, 1000, 1020, "INVALID"),
    ({"sample_age_seconds": 0, "received_at_unix_ms": True}, 1000, 1020, "INVALID"),
    ({"sample_age_seconds": 0, "received_at_unix_ms": 2**63}, 1000, 1020, "INVALID"),
])
def test_explicit_receipt_keeps_its_own_deadline(packet, collected, now, code):
    with pytest.raises(ValueError, match=code):
        source_received_at_unix_ms(
            packet, collected_at_unix_ms=collected, now_unix_ms=now, maximum_age_ms=250,
        )


# 功能：
#   验证边界年龄可以通过，并原样保留显式接收时刻，不从发布时刻重建。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_explicit_receipt_is_not_recomputed():
    received = source_received_at_unix_ms(
        {"sample_age_seconds": .01, "received_at_unix_ms": 770},
        collected_at_unix_ms=1000, now_unix_ms=1020, maximum_age_ms=250,
    )
    assert received == 770
