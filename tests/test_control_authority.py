import math

import pytest

from dronedream_agent_core.control_authority import (
    control_application_category,
    integrate_model_yaw,
    measured_hold_heading,
    remaining_control_validity_ms,
)


# 功能：
#   新鲜观测可生成保持航向，越过正负 180 度时保持同一物理方向。
# 输入：
#   heading：原生航向角度。
#   age：观测年龄秒数。
#   expected：规范化后的预期航向。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("heading,age,expected", [(114., .02, 114.), (181., .25, -179.)])
def test_hold_uses_fresh_native_heading(heading, age, expected):
    assert measured_hold_heading(yaw_deg=heading, sample_age_seconds=age) == expected


# 功能：
#   缺失、非有限、未来或过期姿态均不能被用作保持航向。
# 输入：
#   heading：候选原生航向。
#   age：候选源年龄。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("heading,age", [
    (math.nan, .02), (True, .02), ("90", .02), (90., None), (90., True),
    (90., math.inf), (90., -.01), (90., .25001), (10**400, .01), (90., 10**400),
])
def test_hold_rejects_missing_nonfinite_future_or_stale_attitude(heading, age):
    with pytest.raises(ValueError, match="HOLD_HEADING"):
        measured_hold_heading(yaw_deg=heading, sample_age_seconds=age)


# 功能：
#   积分保留转向符号，并按允许速率裁剪，覆盖跨越正负 180 度的情况。
# 输入：
#   heading：初始航向。
#   rate：请求角速度。
#   expected：裁剪并积分后的预期航向。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "heading,rate,expected",
    [
        (77.0, 0.0, 77.0),
        (0.0, 40.0, 1.0),
        (0.0, -40.0, -1.0),
        (179.5, 20.0, -179.5),
        (-179.5, -20.0, 179.5),
    ],
)
def test_yaw_integrates_signed_bounded_command(heading, rate, expected) -> None:
    assert integrate_model_yaw(
        previous_heading_deg=heading,
        requested_rate_dps=rate,
        maximum_rate_dps=20.0,
        step_seconds=0.05,
    ) == pytest.approx(expected)


# 功能：
#   拒绝非法时间步和超长追赶式积分，不让调度停顿变成突然大幅转向。
# 输入：
#   step：候选积分时间秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("step", [0.0, -0.05, 0.5, math.inf, math.nan])
def test_yaw_rejects_invalid_or_catchup_interval(step) -> None:
    with pytest.raises(ValueError):
        integrate_model_yaw(
            previous_heading_deg=0.0,
            requested_rate_dps=20.0,
            maximum_rate_dps=20.0,
            step_seconds=step,
        )


# 功能：
#   时间推进只会减少授权有效期，不足最小发送窗口时返回无授权。
# 输入：
#   now：当前 UNIX 毫秒时刻。
#   expected：预期剩余窗口或 None。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "now,expected", [(1000, 500), (1400, 100), (1450, 50), (1451, None), (1500, None), (1600, None)]
)
def test_safety_recheck_never_renews_model_decision(now, expected) -> None:
    assert (
        remaining_control_validity_ms(
            now_unix_ms=now,
            authority_deadline_unix_ms=1500,
            requested_validity_ms=600,
        )
        == expected
    )


# 功能：
#   安全替代、航向辅助和路径控制分别归类，不能计为未经修改的模型控制。
# 输入：
#   source：实际控制来源。
#   action：实际应用动作。
#   assisted：是否使用航向辅助。
#   expected：预期归因类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "source,action,assisted,expected",
    [
        ("local-model-body-control", "continue", False, "model-enveloped"),
        ("local-model-body-control", "slow", True, "model-enveloped-heading-assisted"),
        ("deterministic-safety-override", "slow", False, "safety-direction-override"),
        ("local-model-body-control", "hold", False, "safety-brake-or-hold"),
        ("route-target", "continue", False, "route-derived"),
        ("deterministic-brake", "slow", False, "safety-brake-or-hold"),
        ("local-model-body-control", "replan", False, "safety-brake-or-hold"),
    ],
)
def test_safety_replacement_is_not_counted_as_original_model_control(
    source, action, assisted, expected
) -> None:
    assert (
        control_application_category(
            model_authorized=True,
            action=action,
            control_source=source,
            heading_assisted=assisted,
        )
        == expected
    )


# 功能：
#   航向积分拒绝错误类型与不可表示数值，不能由布尔值产生真实转向。
# 输入：
#   field：被修改的积分参数。
#   value：非法输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["previous_heading_deg", "requested_rate_dps",
                                    "maximum_rate_dps", "step_seconds"])
@pytest.mark.parametrize("value", [True, None, "1", 10**400])
def test_yaw_inputs_are_strict_finite_numbers(field, value):
    arguments = dict(previous_heading_deg=0, requested_rate_dps=1,
                     maximum_rate_dps=20, step_seconds=.05)
    arguments[field] = value
    with pytest.raises(ValueError):
        integrate_model_yaw(**arguments)


# 功能：
#   大幅累计航向先化到一个周期再积分，避免小转角被大数精度吞掉。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_large_cumulative_heading_preserves_the_next_small_turn():
    # 10**20 可由 float 精确表示，整数模运算为独立角度参照。
    expected = ((10**20 % 360) + 180) % 360 - 180
    assert measured_hold_heading(yaw_deg=1e20, sample_age_seconds=.01) == expected
    assert integrate_model_yaw(previous_heading_deg=1e20, requested_rate_dps=20,
                               maximum_rate_dps=20, step_seconds=.05) == expected + 1


# 功能：
#   控制期限必须为有界整数，拒绝布尔、浮点及超范围时刻。
# 输入：
#   field：被修改的期限字段。
#   value：非法期限值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("now_unix_ms", True), ("now_unix_ms", None), ("now_unix_ms", 1000.0),
    ("authority_deadline_unix_ms", 2**63), ("authority_deadline_unix_ms", 1500.0),
    ("requested_validity_ms", 600.0), ("minimum_publication_window_ms", 50.0),
])
def test_control_deadline_arguments_do_not_coerce_types(field, value):
    arguments = dict(now_unix_ms=1000, authority_deadline_unix_ms=1500,
                     requested_validity_ms=600, minimum_publication_window_ms=50)
    arguments[field] = value
    with pytest.raises(ValueError):
        remaining_control_validity_ms(**arguments)


# 功能：
#   统计归因必须使用真实布尔授权标记，不把非空字符串当成模型获得控制权。
# 输入：
#   field：被修改的归因字段。
#   value：错误类型或空内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("model_authorized", "false"),
    ("model_authorized", 1), ("heading_assisted", "false"), ("action", []),
    ("action", ""), ("control_source", None)])
def test_control_attribution_requires_explicit_typed_evidence(field, value):
    arguments = dict(model_authorized=True, action="continue",
                     control_source="local-model-body-control", heading_assisted=False)
    arguments[field] = value
    with pytest.raises(ValueError):
        control_application_category(**arguments)


# 功能：
#   无授权动作与未知控制来源分别处理，未知来源不能被补成模型控制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_absent_authority_and_unknown_source_are_not_model_control():
    assert control_application_category(model_authorized=False, action="continue",
        control_source="route-target", heading_assisted=False) == "without-model-authority"
    with pytest.raises(ValueError, match="unrecognized"):
        control_application_category(model_authorized=True, action="continue",
                                     control_source="unknown", heading_assisted=False)
