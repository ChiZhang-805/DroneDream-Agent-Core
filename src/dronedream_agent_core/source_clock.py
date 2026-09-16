"""Source receive times survive cache publication; boot clocks are not UTC."""

import math
import sys
from collections.abc import Mapping


# 功能：
#   1. 校验主机接收时刻和消费时刻，缓存重新发布不能刷新源观测的有效期。
#   2. 对缺少接收字段的历史记录保守恢复主机时间，不把它解释为设备曝光时间。
# 输入：
#   packet：含源年龄以及可选主机接收时刻的观测对象。
#   collected_at_unix_ms：组装缓存包时的 UNIX 毫秒时刻。
#   now_unix_ms：当前消费 UNIX 毫秒时刻。
#   maximum_age_ms：允许的最大源年龄毫秒数。
# 输出：
#   received：通过有效期校验的原始或保守恢复的主机接收时刻。
def source_received_at_unix_ms(
    packet: Mapping[str, object], *, collected_at_unix_ms: int, now_unix_ms: int,
    maximum_age_ms: int,
) -> int:
    if not isinstance(packet, Mapping):
        raise ValueError("SOURCE_PACKET_INVALID")
    if any(type(value) is not int or not 0 <= value < 2**63
           for value in (collected_at_unix_ms, now_unix_ms)):
        raise ValueError("SOURCE_CLOCK_ARGUMENT_INVALID")
    if type(maximum_age_ms) is not int or not 0 < maximum_age_ms < 2**63:
        raise ValueError("SOURCE_AGE_LIMIT_INVALID")
    age = packet.get("sample_age_seconds")
    # 先比较范围，避免 math.isfinite 将巨大整数转换为浮点数时溢出。
    if type(age) not in (int, float) or not -sys.float_info.max <= age <= sys.float_info.max:
        raise ValueError("SOURCE_SAMPLE_AGE_INVALID")
    if not 0 <= age <= maximum_age_ms / 1000:
        raise ValueError("SOURCE_SAMPLE_EXPIRED")
    received = packet.get("received_at_unix_ms")
    if "received_at_unix_ms" not in packet:
        # 仅旧记录缺字段时恢复；显式 null 表示损坏，不得当成刚收到的观测。
        # 向过去取整避免把不足一毫秒的等待算成零；不改写设备启动时钟。
        received = collected_at_unix_ms - math.ceil(age * 1000)
    if type(received) is not int or not 0 <= received < 2**63:
        raise ValueError("SOURCE_RECEIVE_TIME_INVALID")
    if collected_at_unix_ms > now_unix_ms:
        raise ValueError("SOURCE_TIME_IN_FUTURE")
    if received > collected_at_unix_ms:
        raise ValueError("SOURCE_RECEIVE_TIME_INVALID")
    if now_unix_ms - received > maximum_age_ms:
        raise ValueError("SOURCE_SAMPLE_EXPIRED")
    return received
