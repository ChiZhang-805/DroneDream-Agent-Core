"""Live localization cannot inherit a smaller, unmeasured planning allowance."""

import json
from types import SimpleNamespace

import pytest
from test_runtime_commands import _load_executor, _two_segment_track

from dronedream_agent_core.collision import build_tracking_corridor_budget
from dronedream_agent_core.control_uncertainty import (
    finite_positive_number,
    localization_uncertainty_margin_m,
)
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   验证定位方差拒绝非法类型、负数和不可表示数值，不发生隐式转换。
# 输入：
#   value：非法方差样本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, False, "0.01", None, -1., float("nan"), float("inf"),
                                  pytest.param(10**1000, id="oversized-variance")])
def test_uncertainty_rejects_invalid_variance(value):
    with pytest.raises(ValueError, match="VARIANCE_BOUND_INVALID"):
        localization_uncertainty_margin_m(value)


# 功能：
#   验证实测定位方差压缩可用跟踪走廊，不能沿用更乐观的规划假设。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_planning_and_runtime_use_same_measured_uncertainty_scale():
    prior = build_tracking_corridor_budget(.435)
    live = build_tracking_corridor_budget(.435, localization_covariance_m2=.001)
    assert prior["reserved_uncertainty_m"] == .03  # Assumed, not measured.
    assert live["reserved_uncertainty_m"] == localization_uncertainty_margin_m(.001)
    assert live["tracking_lag_limit_m"] < prior["tracking_lag_limit_m"]
    assert (live["reserved_uncertainty_m"] + live["required_local_clearance_m"]
            + live["tracking_lag_limit_m"]) < .435
    with pytest.raises(ValueError, match="cannot fund"):
        build_tracking_corridor_budget(.435, localization_covariance_m2=.0535)


# 功能：
#   验证原生预检将实测余量传入真实执行器，且仅收紧各路段跟踪限制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_actual_executor_spends_live_budget_without_increasing_any_tracking_limit():
    executor = _load_executor()
    args = SimpleNamespace(tracking_lag_limit_m=.20, tracking_rejoin_tolerance_m=.18,
        _tracking_segment_policies=({"minimum_route_clearance_m": .435,
            "tracking_lag_limit_m": .19, "tracking_rejoin_tolerance_m": .17},
            {"minimum_route_clearance_m": .80,
             "tracking_lag_limit_m": .05, "tracking_rejoin_tolerance_m": .04}))
    assert executor._native_route_preflight_clearance(args) == .435
    budget = build_tracking_corridor_budget(.435, localization_covariance_m2=.0025)
    result = executor._apply_native_preflight_tracking_budget(args, {
        "uncertainty_basis": "live-native-localization", "tracking_budget": budget})
    assert result["effective_lag_limit_m"] == pytest.approx(.108)
    assert args.tracking_rejoin_tolerance_m == pytest.approx(.0972)
    assert args._tracking_segment_policies[0]["tracking_lag_limit_m"] == pytest.approx(.108)
    assert args._tracking_segment_policies[1]["tracking_lag_limit_m"] == .05
    assert args._tracking_segment_policies[1]["tracking_rejoin_tolerance_m"] == .04


# 功能：
#   验证缺少可用实测净空的路段不能通过原生飞行预检。
# 输入：
#   policies：缺少有效净空值的路段策略。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("policies", [(), ({},), ({"minimum_route_clearance_m": True},)])
def test_native_motion_cannot_use_an_unmeasured_corridor(policies):
    executor = _load_executor()
    with pytest.raises(ValueError, match="MEASURED_CLEARANCE_REQUIRED"):
        executor._native_route_preflight_clearance(
            SimpleNamespace(_tracking_segment_policies=policies))


# 功能：
#   验证净空预算入口拒绝非物理值并保持稳定的业务错误类别。
# 输入：
#   value：非法净空值，合法值应以米表示。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "0.435", float("nan"), float("inf"), -1,
                                  pytest.param(10**1000, id="oversized-clearance")])
def test_route_budget_rejects_nonphysical_clearance_without_overflow(value):
    with pytest.raises(ValueError, match="no positive tracking corridor"):
        build_tracking_corridor_budget(value)


# 功能：
#   验证轨迹摘要绑定的路段策略保留实测净空，并拒绝损坏的文件字段。
# 输入：
#   tmp_path：pytest 独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_actual_track_policy_parser_preserves_measured_clearance_for_preflight(tmp_path):
    executor = _load_executor()
    track = _two_segment_track()
    payload = {"schema_version": "dronedream.tracking-corridor-policy.v1",
        "track_sha256": sha256_json(track), "segment_policies": [
            {"segment_index": index, "tracking_lag_limit_m": .2,
             "tracking_rejoin_tolerance_m": .18, "control_profile": "precision",
             "minimum_route_clearance_m": clearance}
            for index, clearance in enumerate((.435, .8))]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    policies = executor._load_tracking_segment_policies(path, track=track)
    assert executor._native_route_preflight_clearance(
        SimpleNamespace(_tracking_segment_policies=policies)) == .435
    for value in (True, "0.435", float("nan"), -1, 10**1000):
        payload["segment_policies"][0]["minimum_route_clearance_m"] = value
        path.write_text(json.dumps(payload), encoding="utf-8")
        # 非有限数值现在在共享 JSON 入口即被拒绝，其余错误仍由净空字段门控拒绝。
        with pytest.raises(
            ValueError, match="measured clearance is invalid|PLUGIN_JSON_NUMBER_INVALID"
        ):
            executor._load_tracking_segment_policies(path, track=track)


# 功能：
#   验证共享正数判定保留最小正浮点数，但拒绝零和非法外部值。
# 输入：
#   value：待检查的数值。
#   expected：预期的有效性。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value,expected", [
    (1, True), (5e-324, True), (1e308, True), (0, False), (-1, False),
    (True, False), (None, False), ("1", False), ({}, False),
    (float("nan"), False), (float("inf"), False),
    pytest.param(10**1000, False, id="oversized-budget"),
])
def test_positive_budget_validator_is_strict(value, expected):
    assert finite_positive_number(value) is expected


# 功能：
#   验证方差换算的数值边界及单位，零不额外引入未声明的最小余量。
# 输入：
#   variance：以平方米表示的有效方差上界。
#   expected：预期三倍标准差余量，单位米。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("variance,expected", [(0, 0), (.01, .3), (1e308, 3e154)])
def test_uncertainty_margin_numerical_boundaries(variance, expected):
    assert localization_uncertainty_margin_m(variance) == pytest.approx(expected)
