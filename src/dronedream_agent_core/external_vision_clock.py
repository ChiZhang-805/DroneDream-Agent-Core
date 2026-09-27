"""Run-bound source-clock transport checks, not localization qualification.

This supports only an explicitly verified PX4/Gazebo shared clock. A hardware
boot clock or a host monotonic clock is not interchangeable with that source.
Never infer synchronization from a reply count or restamp a delayed observation.
"""

from __future__ import annotations

import math
import re
import sys
import threading


# 功能：检查源时间戳及实体编号的精确整数类型，拒绝布尔值和隐式数值转换。
# 输入：value：协议字段；maximum：协议上界。
# 输出：合法正整数，否则抛出协议错误。
def _positive_integer(value, maximum=2**63 - 1):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError("EXTERNAL_VISION_CLOCK_INTEGER_INVALID")
    return value


# 功能：校验接收时刻，不将非有限值或文本转换成时钟证据。
# 输入：value：主机单调时钟秒数。
# 输出：有限、非负的原始数值。
def _monotonic(value):
    if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
        raise ValueError("EXTERNAL_VISION_RECEIPT_CLOCK_INVALID")
    return value


class SharedSimulationClock:
    """Thread-safe latest-source mailbox for a dedicated MAVLink connection."""

    # 功能：将时钟实例绑定到单次已核对的仿真与飞控实体，不允许自动推断跨时钟关系。
    # 输入：clock_domain：运行专属共享时钟标识；system_id、component_id：原生飞控实体。
    # 输出：未收到源时钟、尚不可回应同步请求的实例。
    def __init__(self, *, clock_domain, system_id, component_id):
        if type(clock_domain) is not str or not re.fullmatch(
            r"px4-gz-sitl:[A-Za-z0-9_-]{1,100}", clock_domain
        ):
            raise ValueError("EXTERNAL_VISION_SHARED_CLOCK_BINDING_REQUIRED")
        self.clock_domain = clock_domain
        self.system_id = _positive_integer(system_id, 255)
        self.component_id = _positive_integer(component_id, 255)
        self._lock = threading.Lock()
        self._sample = None
        self._last_receipt = None
        self._last_request = None
        self._invalid = False

    # 功能：只保留真实前进的源时钟；重复钟不续期，回退永久失效直到创建新运行实例。
    # 输入：source_ns：Gazebo 原始仿真纳秒；received_monotonic：实际接收时刻。
    # 输出：无；无时间外推、无来源时间平移。
    def observe(self, source_ns, *, received_monotonic):
        if type(source_ns) is not int or not 0 <= source_ns <= 2**63 - 1:
            raise ValueError("EXTERNAL_VISION_CLOCK_INTEGER_INVALID")
        _monotonic(received_monotonic)
        with self._lock:
            if self._invalid:
                raise ValueError("EXTERNAL_VISION_CLOCK_RESET_REQUIRES_NEW_BINDING")
            if self._last_receipt is not None and received_monotonic < self._last_receipt:
                self._invalid = True
                raise ValueError("EXTERNAL_VISION_RECEIPT_CLOCK_REGRESSED")
            if self._sample is not None and source_ns < self._sample[0]:
                self._invalid = True
                raise ValueError("EXTERNAL_VISION_SOURCE_CLOCK_REGRESSED")
            self._last_receipt = received_monotonic
            if self._sample is None or source_ns > self._sample[0]:
                self._sample = (source_ns, received_monotonic)

    # 功能：读取不超过 30 毫秒接收延迟的实际源时钟，暂停或过期时不伪造新时刻。
    # 输入：now_monotonic：调用时刻。
    # 输出：源纳秒；未就绪、失效、未来接收或过期返回 None。
    def current(self, *, now_monotonic):
        _monotonic(now_monotonic)
        with self._lock:
            if self._invalid or self._sample is None:
                return None
            source, received = self._sample
            return source if source > 0 and 0 <= now_monotonic - received <= 0.030 else None

    # 功能：只回应绑定实体的新同步请求，使用真实源钟而非回显请求钟或主机钟。
    # 输入：tc1、ts1：TIMESYNC 字段；实体编号；now_monotonic：接收处理时刻。
    # 输出：(源钟纳秒, 原请求纳秒)，不合格请求返回 None；回复数量不是收敛证据。
    def reply(self, *, tc1, ts1, system_id, component_id, now_monotonic):
        if type(tc1) is not int or tc1 != 0:
            return None
        _positive_integer(ts1)
        _positive_integer(system_id, 255)
        _positive_integer(component_id, 255)
        _monotonic(now_monotonic)
        with self._lock:
            if (
                self._invalid
                or self._sample is None
                or (system_id, component_id) != (self.system_id, self.component_id)
            ):
                return None
            source, received = self._sample
            if (
                source == 0
                or not 0 <= now_monotonic - received <= 0.030
                or not -10_000_000 <= source - ts1 <= 100_000_000
                or (self._last_request is not None and ts1 <= self._last_request)
            ):
                return None
            self._last_request = ts1
            return source, ts1


# 功能：以真实飞控回读检查源采样时间及重置标记；拒绝接收时刻替代、未来或过旧观测。
# 输入：源采样微秒、飞控回读采样和发布微秒、发送和回读的重置标记。
# 输出：有符号时间传输误差（微秒）；不授予定位精度、噪声模型或飞行资格。
def validate_external_vision_time_readback(
    *,
    source_timestamp_us,
    received_sample_us,
    received_publish_us,
    expected_reset_counter,
    received_reset_counter,
    maximum_time_error_us=2_000,
):
    if type(maximum_time_error_us) is not int or not 0 < maximum_time_error_us <= 10_000:
        raise ValueError('EXTERNAL_VISION_TIME_UNCERTAINTY_INVALID')
    for value in (source_timestamp_us, received_sample_us, received_publish_us):
        _positive_integer(value)
    for value in (expected_reset_counter, received_reset_counter):
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError("EXTERNAL_VISION_RESET_COUNTER_INVALID")
    if expected_reset_counter != received_reset_counter:
        raise ValueError("EXTERNAL_VISION_RESET_COUNTER_LOST")
    error_us = received_sample_us - source_timestamp_us
    if abs(error_us) > maximum_time_error_us:
        raise ValueError("EXTERNAL_VISION_SAMPLE_TIMESTAMP_NOT_PRESERVED")
    if not 0 <= received_publish_us - received_sample_us <= 200_000:
        raise ValueError("EXTERNAL_VISION_SAMPLE_AGE_INVALID")
    return error_us


# 功能：严格验证 PX4 文本回读的实际位置与四元数，避免 NaN 比较为假而误报成功。
# 输入：readback：有界 listener 输出；position、quaternion：实际发送的 NED/FRD 均值。
# 输出：无；缺失、重复、非有限或超出打印精度的回读立即拒绝，不授予融合资格。
def validate_external_vision_pose_readback(*, readback, position, quaternion):
    if type(readback) is not str or len(readback) > 32768:
        raise ValueError("EXTERNAL_VISION_POSE_READBACK_INVALID")
    for name, expected, size, tolerance in (
        ("position", position, 3, 1e-5),
        ("q", quaternion, 4, 2e-5),
    ):
        if (not isinstance(expected, (tuple, list)) or len(expected) != size
                or any(type(v) not in (int, float) or not -1e6 <= v <= 1e6 for v in expected)):
            raise ValueError("EXTERNAL_VISION_EXPECTED_POSE_INVALID")
        if name == "q" and abs(math.hypot(*expected)-1.) > 1e-4:
            raise ValueError("EXTERNAL_VISION_EXPECTED_POSE_INVALID")
        matches = re.findall(r"^\s*"+name+r": \[([^\]]+)\]", readback, re.M)
        if len(matches) != 1:
            raise ValueError("EXTERNAL_VISION_POSE_READBACK_AMBIGUOUS")
        try:
            actual = [float(value.strip()) for value in matches[0].split(",")]
        except (ValueError, OverflowError) as error:
            raise ValueError("EXTERNAL_VISION_POSE_READBACK_INVALID") from error
        if (len(actual) != size or not all(math.isfinite(value) for value in actual)
                or any(abs(a-b) > tolerance for a,b in zip(actual, expected, strict=True))):
            raise ValueError("EXTERNAL_VISION_POSE_READBACK_MISMATCH")
