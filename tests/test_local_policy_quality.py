import pytest

from dronedream_agent_core.local_policy_quality import (
    LocalPolicyTrainingMetrics,
    navigation_quality_issues,
    summarize_pilot_axes,
)


# 功能：
#   构造具有双向四轴、风险和非运动样本支持的合格连续控制统计夹具。
# 输入：
#   无。
# 输出：
#   metrics：除候选排序外满足默认门限的合成指标。
def _metrics() -> LocalPolicyTrainingMetrics:
    targets = [[0.5] * 4, [-0.5] * 4] * 20
    metrics = LocalPolicyTrainingMetrics(
        sample_count=80,
        motion_sample_count=40,
        non_motion_sample_count=40,
        authorized_motion_recall=1.0,
        non_motion_recall=1.0,
        risky_sample_count=40,
        safe_sample_count=40,
        action_accuracy=1.0,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=0.0,
        candidate_sample_count=0,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.0,
        mean_cross_entropy=0.0,
        pilot_control_mean_absolute_error=0.0,
        pilot_axis_evidence=summarize_pilot_axes(targets, targets),
    )
    return metrics


# 功能：
#   检查连续模式仍执行运动授权质量门限，不因免检候选排序而免去其它验收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_continuous_quality_skips_only_candidate_ranking() -> None:
    metrics = _metrics()
    assert navigation_quality_issues(metrics, continuous_control=True) == []
    metrics.motion_authorization_accuracy = 0.84
    assert "MOTION_AUTHORIZATION_ACCURACY_TOO_LOW" in navigation_quality_issues(
        metrics, continuous_control=True
    )


# 功能：
#   检查没有危险样本时，即使写入完美危险召回率也不能通过支持数量门限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_hazard_examples_cannot_qualify_a_perfect_recall() -> None:
    metrics = _metrics()
    metrics.risky_sample_count = 0
    metrics.safe_sample_count = 80
    assert "INSUFFICIENT_RISKY_SAMPLES" in navigation_quality_issues(
        metrics, continuous_control=True
    )


# 功能：
#   检查训练中始终保持零偏航不能证明模型具备偏航双向操纵能力。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_constant_yaw_cannot_qualify_four_axis_control() -> None:
    metrics = _metrics()
    targets = [[0.5, 0.5, 0.5, 0.0], [-0.5, -0.5, -0.5, 0.0]] * 20
    metrics.pilot_axis_evidence = summarize_pilot_axes(targets, targets)
    assert metrics.pilot_axis_evidence["yaw"].active_mean_absolute_error is None
    assert navigation_quality_issues(metrics, continuous_control=True) == [
        "PILOT_AXIS_DIRECTION_COVERAGE_INSUFFICIENT_YAW"
    ]


# 功能：
#   用大量零动作和少量错误活动样本核对活动误差不会被全体平均值稀释。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_active_axis_errors_cannot_hide_in_many_zero_targets() -> None:
    targets = [[0.0] * 4] * 1000 + [[0.5] * 4, [-0.5] * 4] * 20
    evidence = summarize_pilot_axes([[0.0] * 4] * len(targets), targets)
    assert evidence["yaw"].mean_absolute_error < 0.02
    assert evidence["yaw"].active_mean_absolute_error == 0.5
    metrics = _metrics()
    metrics.pilot_axis_evidence = evidence
    metrics.motion_sample_count = len(targets)
    metrics.non_motion_sample_count = 40
    metrics.sample_count = len(targets) + 40
    metrics.safe_sample_count = len(targets)
    assert "PILOT_AXIS_ERROR_TOO_HIGH_YAW" in navigation_quality_issues(
        metrics, continuous_control=True
    )


# 功能：
#   检查轴数不足、非有限及超出归一化范围的控制值不能进入证据统计。
# 输入：
#   bad：非法预测向量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [[0.0] * 3, [float("nan")] * 4, [1.01] * 4])
def test_axis_evidence_rejects_invalid_control_values(bad) -> None:
    with pytest.raises(ValueError):
        summarize_pilot_axes([bad], [[0.0] * 4])


# 功能：
#   检查缺失轴证据必须明确报告，不能将空字典视为全部轴已经通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_role_or_axis_metrics_are_not_success_evidence() -> None:
    metrics = _metrics()
    metrics.pilot_axis_evidence = {}
    assert "PILOT_AXIS_EVIDENCE_INCOMPLETE" in navigation_quality_issues(
        metrics, continuous_control=True
    )


# 功能：
#   检查总是悬停的模型即使整体准确率很高，也必须因允许运动召回不足而拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_always_hold_cannot_hide_behind_overall_accuracy_or_safe_risk_head() -> None:
    metrics = _metrics()
    metrics.motion_authorization_accuracy = 0.99
    metrics.authorized_motion_recall = 0.0
    assert "AUTHORIZED_MOTION_RECALL_TOO_LOW" in navigation_quality_issues(
        metrics, continuous_control=True
    )


# 功能：
#   检查质量门限不接受布尔、分数或非有限的样本数，避免比较失效放过证据不足的模型。
# 输入：
#   field：需要替换的证据数量门限。
#   value：非法门限值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["minimum_risk_class_samples", "minimum_axis_direction_samples"])
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf")])
def test_quality_requires_integer_support_thresholds(field, value):
    with pytest.raises(ValueError, match="threshold"):
        navigation_quality_issues(_metrics(), continuous_control=True, **{field: value})


# 功能：
#   检查连续控制模式不能由普通真值转换决定，防止字符串 false 跳过候选排序验收。
# 输入：
#   mode：非法模式参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["false", 1, None])
def test_quality_requires_explicit_boolean_mode(mode):
    with pytest.raises(ValueError, match="boolean"):
        navigation_quality_issues(_metrics(), continuous_control=mode)


# 功能：
#   检查验收入口重新验证指标对象，不能因 NaN 大小比较均为假而放过被篡改的模型指标。
# 输入：
#   field：被修改为 NaN 的关键指标名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["risk_hold_recall", "risk_mean_absolute_error",
    "motion_authorization_accuracy", "pilot_control_mean_absolute_error"])
def test_quality_revalidates_mutated_metrics(field):
    metrics = _metrics().model_copy(update={field: float("nan")})
    with pytest.raises(ValueError):
        navigation_quality_issues(metrics, continuous_control=True)


# 功能：
#   检查布尔值不能作为控制误差门限，避免 True 被解释为容忍一整单位的误差。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_quality_rejects_boolean_error_threshold():
    with pytest.raises(ValueError, match="threshold"):
        navigation_quality_issues(_metrics(), continuous_control=True,
                                  maximum_pilot_control_mae=True)


# 功能：
#   检查重新校验时仍严格区分布尔值和数值，不能将被篡改的 True 转成完美召回率。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_quality_revalidation_cannot_coerce_boolean_metrics():
    metrics = _metrics().model_copy(update={"risk_hold_recall": True})
    with pytest.raises(ValueError):
        navigation_quality_issues(metrics, continuous_control=True)
