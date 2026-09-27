"""Source-clock transport boundaries; no flight or estimator qualification."""

import pytest

from dronedream_agent_core.external_vision_clock import (
    SharedSimulationClock,
    validate_external_vision_time_readback,
)


# 功能：建立无观测的独立测试时钟，避免沿用旧运行的时钟状态。
# 输入：无。
# 输出：绑定到测试飞控的时钟实例。
def clock():
    return SharedSimulationClock(clock_domain="px4-gz-sitl:test", system_id=1, component_id=1)


# 功能：构造真实请求形状并允许各边界测试改写单个字段。
# 输入：value：时钟实例；changes：明确测试变体。
# 输出：被测方法实际回应，不预置同步成功状态。
def reply(value, **changes):
    return value.reply(
        **(
            dict(tc1=0, ts1=1_000_000_000, system_id=1, component_id=1, now_monotonic=10.0)
            | changes
        )
    )


# 功能：没有新源观测时不回复；重复源钟不能让暂停的仿真看起来仍在前进。
# 输入：无。
# 输出：无；检查首次回复、重复请求、过期和重新收到新源观测的行为。
def test_new_source_clock_is_required_and_duplicates_do_not_refresh_it():
    value = clock()
    assert reply(value) is None
    value.observe(1_002_000_000, received_monotonic=10.0)
    assert reply(value) == (1_002_000_000, 1_000_000_000)
    assert reply(value) is None
    value.observe(1_002_000_000, received_monotonic=10.1)
    assert value.current(now_monotonic=10.1) is None
    value.observe(1_102_000_000, received_monotonic=10.2)
    assert reply(value, ts1=1_100_000_000, now_monotonic=10.2) == (1_102_000_000, 1_100_000_000)


# 功能：拒绝错误实体、过旧/过早请求、未来接收、过期源钟与非请求消息。
# 输入：changes：无效请求变体。
# 输出：无；拒绝不会消耗下一次合法请求。
@pytest.mark.parametrize(
    "changes",
    [
        {"system_id": 2},
        {"component_id": 197},
        {"tc1": 1},
        {"tc1": False},
        {"ts1": 800_000_000},
        {"ts1": 1_020_000_000},
        {"now_monotonic": 9.99},
        {"now_monotonic": 10.1},
    ],
)
def test_invalid_request_cannot_use_the_shared_clock(changes):
    value = clock()
    value.observe(1_000_000_000, received_monotonic=10.0)
    assert reply(value, **changes) is None
    assert reply(value) is not None


# 功能：任一时钟回退后必须重建运行绑定，不能在旧历史中继续回复。
# 输入：source、received：回退类型。
# 输出：无；之后的新数据也不能自动复活失效绑定。
@pytest.mark.parametrize("source,received", [(0, 11.0), (900_000_000, 11.0), (1_100_000_000, 9.0)])
def test_clock_regression_is_latched(source, received):
    value = clock()
    value.observe(1_000_000_000, received_monotonic=10.0)
    with pytest.raises(ValueError, match="REGRESSED"):
        value.observe(source, received_monotonic=received)
    assert reply(value) is None
    with pytest.raises(ValueError, match="NEW_BINDING"):
        value.observe(1_200_000_000, received_monotonic=12.0)


# 功能：校验明确运行绑定和时间类型，拒绝不可靠的时钟猜测。
# 输入：domain：错误标识。
# 输出：无；构造立即拒绝。
@pytest.mark.parametrize("domain", [None, "host-monotonic", "px4-gz-sitl:", "px4-gz-sitl:a/b"])
def test_clock_binding_is_explicit(domain):
    with pytest.raises(ValueError, match="BINDING"):
        SharedSimulationClock(clock_domain=domain, system_id=1, component_id=1)


# 功能：严格拒绝畸形源钟，不用默认零或类型转换隐藏错误。
# 输入：source、received：非法时钟变体。
# 输出：无；没有可用回复。
@pytest.mark.parametrize(
    "source,received",
    [
        (True, 10.0),
        (-1, 10.0),
        (1.0, 10.0),
        (2**63, 10.0),
        (1, True),
        (1, "10"),
        (1, float("nan")),
        (1, float("inf")),
        (1, -1.0),
        (1, 2**4096),
    ],
)
def test_malformed_clock_values(source, received):
    value = clock()
    with pytest.raises(ValueError):
        value.observe(source, received_monotonic=received)
    assert reply(value) is None


# 功能：仿真初始零时刻合法，但不能将零响应发送为另一条 TIMESYNC 请求。
# 输入：无。
# 输出：无；真实源钟前进后才具备回应能力。
def test_initial_zero_clock_waits_for_actual_simulation_progress():
    value = clock()
    value.observe(0, received_monotonic=10.0)
    assert value.current(now_monotonic=10.0) is None
    assert reply(value, ts1=1) is None
    value.observe(1_000_000_000, received_monotonic=10.1)
    assert reply(value, now_monotonic=10.1) == (1_000_000_000, 1_000_000_000)


# 功能：验证飞控保留原始采样时刻，而不是仅检查已经发送若干次同步回应。
# 输入：changes：回读误差或重置标记变体；expected：应出现的明确错误。
# 输出：无；正常回读保留真实 34 微秒误差，其余指定情况拒绝。
@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, None),
        ({"received_sample_us": 1_100_000}, "NOT_PRESERVED"),
        ({"received_sample_us": 1_002_001}, "NOT_PRESERVED"),
        ({"received_sample_us": 997_999}, "NOT_PRESERVED"),
        ({"received_publish_us": 1_000_000}, "AGE_INVALID"),
        ({"received_publish_us": 1_300_000}, "AGE_INVALID"),
        ({"received_reset_counter": 0}, "COUNTER_LOST"),
        ({"received_reset_counter": True}, "COUNTER_INVALID"),
        ({"source_timestamp_us": True}, "INTEGER_INVALID"),
    ],
)
def test_source_timestamp_and_reset_readback(changes, expected):
    values = (
        dict(
            source_timestamp_us=1_000_000,
            received_sample_us=1_000_034,
            received_publish_us=1_100_000,
            expected_reset_counter=71,
            received_reset_counter=71,
        )
        | changes
    )
    if expected:
        with pytest.raises(ValueError, match=expected):
            validate_external_vision_time_readback(**values)
    else:
        assert validate_external_vision_time_readback(**values) == 34
