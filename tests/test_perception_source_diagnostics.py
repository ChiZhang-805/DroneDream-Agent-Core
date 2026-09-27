import pytest

from dronedream_agent_core.perception_source_diagnostics import PerceptionSourceDiagnostics


# 功能：
#   验证过期、未来来源分别统计，取得副本不能回写原统计或获得运动权限。
# 输入：
#   无。
# 输出：
#   无。
def test_source_age_statistics_preserve_expiry_and_future_clocks():
    stats = PerceptionSourceDiagnostics()
    stats.record("selected", now_ms=1000, depth_ms=900, state_ms=950)
    stats.record("health", now_ms=1200, depth_ms=900, state_ms=1201)
    record = stats.snapshot()
    assert record["source_ages"]["health:depth"] == {
        "count": 1, "total_ms": 300, "maximum_ms": 300,
        "future_count": 0, "over_250_ms_count": 1}
    assert record["source_ages"]["health:state"]["future_count"] == 1
    assert record["source_ages"]["health:state"]["count"] == 0
    assert record["qualification_granted"] is False
    record["source_ages"]["health:depth"]["count"] = 99
    assert stats.snapshot()["source_ages"]["health:depth"]["count"] == 1


# 功能：
#   验证统计容量、同轮去重及拒绝坏输入，避免诊断自身产生不受控开销。
# 输入：
#   无。
# 输出：
#   无。
def test_feature_issue_counts_are_bounded_and_invalid_input_is_atomic():
    stats = PerceptionSourceDiagnostics()
    for index in range(100):
        stats.record_feature_issues([str(index), str(index)])
    assert len(stats.snapshot()["feature_issue_cycle_counts"]) == 33
    assert stats.snapshot()["feature_issue_cycle_counts"]["OTHER"] == 68
    before = stats.snapshot()
    for issues in (["a", []], [None], ["x" * 193], "code"):
        with pytest.raises(ValueError):
            stats.record_feature_issues(issues)
    with pytest.raises(ValueError):
        stats.record("unknown", now_ms=1000, depth_ms=900, state_ms=950)
    assert stats.snapshot() == before
