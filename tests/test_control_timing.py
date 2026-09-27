import math

import pytest

from dronedream_agent_core.control_timing import (
    LOCAL_SETPOINT_PERIOD_SECONDS,
    LOCAL_DISPATCH_RESERVE_MS,
    TRAINING_DISPATCH_RESERVE_MS,
    LOCAL_TRANSPORT_BUDGET_MS,
    validate_model_dispatch_rate,
    continuous_control_cadence_bounded,
    continuous_control_evidence_required,
    continuous_timing_is_bounded,
    resolve_control_timing,
)


# 功能：分离决策频率与实际发令轮询，训练和部署共用同一执行轮询，来源寿命不变。
# 输入：生产共享常量；输出：交接余量来自真实轮询周期而非随意调小的阈值。
def test_executor_reserve_matches_faster_dispatch_cadence():
    assert LOCAL_SETPOINT_PERIOD_SECONDS == .02
    assert LOCAL_DISPATCH_RESERVE_MS == 40
    assert LOCAL_DISPATCH_RESERVE_MS == round(LOCAL_SETPOINT_PERIOD_SECONDS * 1000) + LOCAL_TRANSPORT_BUDGET_MS
    assert TRAINING_DISPATCH_RESERVE_MS == LOCAL_DISPATCH_RESERVE_MS


# 功能：验证 CLI 不能用慢轮询冒充较短发令等待；输入：非法频率；输出：启动前拒绝。
@pytest.mark.parametrize('rate', [20, 49.9, 101, True, None, float('nan'), float('inf')])
def test_dispatch_rejects_rate_that_cannot_honor_reserve(rate):
    with pytest.raises(ValueError, match='MODEL_DISPATCH_RATE_INSUFFICIENT'):
        validate_model_dispatch_rate(rate)
    validate_model_dispatch_rate(50)
    validate_model_dispatch_rate(100)


# 功能：
#   验证训练与部署采用同一连续控制证据门槛，云端规划不冒充本地执行。
# 输入：
#   provider：提供者标识。
#   expected：拥有控制授权时是否需要连续证据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("provider,expected", [
    ("local-policy", True), ("simulation-training", True), ("kimi", False), (None, False),
    ([], False), ({}, False),
])
def test_training_and_deployment_require_same_continuous_evidence(provider, expected):
    assert continuous_control_evidence_required(provider, True) is expected
    assert not continuous_control_evidence_required(provider, False)


# 功能：
#   验证单条指令通过检查，不能掩盖连续控制间隔超过时限。
# 输入：
#   gap：成功控制之间的最大间隔，单位秒。
#   expected：间隔是否在设计上限以内。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("gap,expected", [
    (0., True), (.05, True), (.25, True), (.250001, False), (8.09, False),
    (None, False), (True, False), (-.1, False), (float("nan"), False),
    (float("inf"), False), ("0.05", False),
    pytest.param(10**1000, False, id="oversized-gap"),
])
def test_valid_individual_commands_do_not_hide_continuous_model_gaps(gap, expected):
    assert continuous_control_cadence_bounded({"accepted": True}, gap) is expected
    assert not continuous_control_cadence_bounded({"accepted": False}, gap)
    assert not continuous_control_cadence_bounded({}, gap)

# 功能：
#   验证本地连续控制周期与授权期限不会随云端请求超时延长。
# 输入：
#   period：请求的决策周期，单位秒。
#   timeout：提供者超时时间，单位秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("period,timeout", [(3., 60.), (.2, 1.), (.05, .25)])
def test_continuous_mode_never_inherits_cloud_cadence_or_lease(period, timeout):
    timing = resolve_control_timing(
        output_mode="normalized-body-velocity", sensor_tick_hz=8,
        decision_period_seconds=period, provider_timeout_seconds=timeout,
    )
    assert timing.sensor_tick_hz == 20
    assert timing.decision_period_seconds == .05
    assert timing.maximum_decision_age_seconds == .25
    assert timing.control_lease_seconds == .25


# 功能：
#   验证显式粗粒度建议模式保持原有调用间隔，避免增加云端请求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cloud_advisory_period_does_not_create_more_api_calls():
    timing = resolve_control_timing(
        output_mode="legacy-candidate-selection", sensor_tick_hz=8,
        decision_period_seconds=3, provider_timeout_seconds=10,
    )
    assert timing.decision_period_seconds == 3
    assert timing.sensor_tick_hz == 8


# 功能：
#   验证验收读取实际生效配置，并拒绝低频、过期授权或云端运动接管。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_qualification_requires_effective_cadence_and_source_deadline():
    timing = resolve_control_timing(
        output_mode="normalized-body-velocity", sensor_tick_hz=8,
        decision_period_seconds=3, provider_timeout_seconds=60,
    )
    receipt = {"control_output_mode": "normalized-body-velocity", "effective": timing.as_dict(),
               "cloud_fallback_controls_motion": False}
    assert continuous_timing_is_bounded(receipt)
    for key, value in (("control_lease_seconds", 6), ("sensor_tick_hz", 8),
                       ("decision_period_seconds", 3), ("maximum_decision_age_seconds", 12)):
        assert not continuous_timing_is_bounded({
            **receipt, "effective": {**receipt["effective"], key: value},
        })
    assert not continuous_timing_is_bounded({**receipt, "cloud_fallback_controls_motion": True})
    assert not continuous_timing_is_bounded(None)


# 功能：
#   验证三项外部时序配置均严格拒绝非正、非有限或非数字输入。
# 输入：
#   invalid：非法配置值。
#   field：待检查的配置字段名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", [
    "sensor_tick_hz", "decision_period_seconds", "provider_timeout_seconds",
])
@pytest.mark.parametrize("invalid", [
    math.nan, math.inf, 0, -1, True, "0.05", None, [],
    pytest.param(10**1000, id="oversized-integer"),
])
def test_invalid_timing_rejected(invalid, field):
    values = {"sensor_tick_hz": 8, "decision_period_seconds": .05,
              "provider_timeout_seconds": 1}
    values[field] = invalid
    with pytest.raises(ValueError):
        resolve_control_timing(output_mode="normalized-body-velocity", **values)


# 功能：
#   验证外部验收回执中的非法数值只导致拒绝，不导致验收程序崩溃。
# 输入：
#   invalid：伪造的生效配置字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [
    True, "20", None, [], {}, pytest.param(10**1000, id="oversized-integer"),
])
def test_qualification_rejects_invalid_effective_values(invalid):
    effective = resolve_control_timing(
        output_mode="normalized-body-velocity", sensor_tick_hz=20,
        decision_period_seconds=.05, provider_timeout_seconds=1,
    ).as_dict()
    for field in effective:
        assert not continuous_timing_is_bounded({
            "control_output_mode": "normalized-body-velocity",
            "cloud_fallback_controls_motion": False,
            "effective": {**effective, field: invalid},
        })


# 功能：
#   验证有限输入在倒数或倍乘溢出后仍会被拒绝，不发布无限时序预算。
# 输入：
#   mode：连续控制或显式粗粒度建议模式。
#   period：触发不可表示结果的周期，单位秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode,period", [
    ("normalized-body-velocity", 5e-324), ("legacy-candidate-selection", 1e308),
])
def test_unrepresentable_derived_timing_is_rejected(mode, period):
    with pytest.raises(ValueError):
        resolve_control_timing(output_mode=mode, sensor_tick_hz=20,
                               decision_period_seconds=period, provider_timeout_seconds=1)
