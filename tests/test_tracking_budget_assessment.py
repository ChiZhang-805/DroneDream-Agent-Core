"""Shared planning/runtime budget arithmetic, including failed native admission."""
import pytest

from dronedream_agent_core.collision import assess_tracking_corridor_budget, build_tracking_corridor_budget
from dronedream_agent_core.native_preflight import NativePerceptionReadiness


# 功能：
#   复现窄路线的规划假设与实际定位不匹配，确保诊断保留缺口而执行预算仍拒绝。
# 输入：
#   无；数字是回归用例，不替代新运行的原生观测。
# 输出：
#   无。
def test_narrow_route_does_not_turn_assumption_into_native_qualification():
    clearance = 0.24257163658958567
    planned = assess_tracking_corridor_budget(clearance)
    assert planned['funded'] and not planned['live_localization_checked']
    actual = assess_tracking_corridor_budget(clearance, 0.0537)
    assert not actual['funded'] and actual['clearance_deficit_m'] > 0.5
    assert actual['maximum_localization_variance_m2'] < 0.002
    with pytest.raises(ValueError):
        build_tracking_corridor_budget(clearance, localization_covariance_m2=0.0537)


# 功能：
#   检查宽窄路线的执行公式与诊断一致，非法方差不能成为已知误差或成功资格。
# 输入：
#   variance：可能的无效协方差输入。
# 输出：
#   无。
@pytest.mark.parametrize('variance', [float('nan'), float('inf'), -1, True, '0.01'])
def test_invalid_variance_remains_rejected(variance):
    with pytest.raises(ValueError):
        assess_tracking_corridor_budget(1.0, variance)


# 功能：
#   起飞检查失败时留下本次实际预算，但不得保留已失效的连续合格帧。
# 输入：
#   无。
# 输出：
#   无。
def test_failed_preflight_retains_budget_diagnostic_not_authority():
    gate = NativePerceptionReadiness(minimum_route_clearance_m=0.2426)
    sample = {'schema_version': 'dronedream.perception-fusion-health.v1',
              'stream_healthy': True, 'identity_accepted': True, 'truth_correction_applied': False,
              'pose_source': 'native-estimator-fixed-deployment-binding', 'realtime_features_ready': True,
              'latest_sequence': 1, 'updated_at_unix_ms': 1000,
              'localization_observed_at_unix_ms': 1000, 'localization_covariance_m2': 0.0537}
    assert not gate.observe(sample, now_unix_ms=1000)
    assert gate.independent_frames == 0 and gate.tracking_budget is None
    assert gate.budget_assessment['funded'] is False
    assert not gate.observe({}, now_unix_ms=1001)
    assert gate.budget_assessment is None


# 功能：拒绝无法序列化的预算溢出及无效净空，避免规划诊断产生非标准 JSON。
# 输入：clearance：外部地图或计算所得的净空。
# 输出：无。
@pytest.mark.parametrize('clearance', [float('nan'), float('inf'), -1, 0, True, '1', 1e308, 10**1000])
def test_invalid_clearance_remains_rejected(clearance):
    with pytest.raises(ValueError):
        assess_tracking_corridor_budget(clearance)


# 功能：核对宽窄路线在相同实测方差下使用同一套门槛，不因诊断入口而改变资格。
# 输入：clearance、variance：不同路线及定位条件。
# 输出：无。
@pytest.mark.parametrize('clearance', [.08, .1, .2426, .35, .89, 1, 3])
@pytest.mark.parametrize('variance', [0, .001, .0537, .5])
def test_assessment_and_execution_budget_agree(clearance, variance):
    assessment = assess_tracking_corridor_budget(clearance, variance)
    if assessment['funded']:
        budget = build_tracking_corridor_budget(clearance, localization_covariance_m2=variance)
        assert budget['reserved_uncertainty_m'] == assessment['reserved_uncertainty_m']
        assert budget['tracking_lag_limit_m'] > 0
    else:
        with pytest.raises(ValueError):
            build_tracking_corridor_budget(clearance, localization_covariance_m2=variance)
