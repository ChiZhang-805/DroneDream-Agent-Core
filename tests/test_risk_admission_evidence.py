"""Risk/behavior denominator separation with explicitly synthetic admission evidence."""

import pytest
from test_local_policy_quality import _metrics

from dronedream_agent_core.local_policy_quality import navigation_quality_issues
from dronedream_agent_core.training.risk_admission_evidence import (
    ActionRiskAdmissionEvidence,
    ActionRiskRegionEvidence,
    action_risk_evidence_from_reports,
    action_risk_evidence_issues,
    action_risk_summary,
)


# 功能：
#   建立专用于接口测试的独立风险支持，不能写入真实模型包或替代真实推理验收。
# 输入：
#   model_sha：测试风险图的摘要。
# 输出：
#   evidence：一百二十探针、六十观测的显式合成证据。
def synthetic_risk_evidence(model_sha='a'*64):
    evidence = ActionRiskAdmissionEvidence(model_sha256=model_sha, teacher_config_sha256='b'*64,
        regions=[ActionRiskRegionEvidence(dataset_receipt_sha256=['c'*64], test_groups=['d'*64],
            sample_count=120, risky_sample_count=60, safe_sample_count=60,
            independent_observation_count=60, safe_observation_count=60,
            unsafe_observation_count=60, action_contrast_observation_count=60,
            risk_hold_recall=1., safe_motion_recall=1., risk_mean_absolute_error=0.,
            action_discrimination_fraction=1., p99_input_and_inference_ms=.5)])
    return evidence


# 功能：
#   不同分母不混入行为计数，原有行为门槛与逐区域风险门槛同时生效。
# 输入：
#   无：明确合成的行为与风险证据。
# 输出：
#   None：通过断言核对。
def test_separate_denominators_preserve_all_quality_gates():
    metrics = _metrics()
    evidence = synthetic_risk_evidence()
    metrics = type(metrics).model_validate({**metrics.model_dump(), **action_risk_summary(evidence),
                                           'action_risk_evidence': evidence.model_dump()})
    assert metrics.sample_count == 80
    assert metrics.risky_sample_count + metrics.safe_sample_count == 120
    assert navigation_quality_issues(metrics, continuous_control=True) == []
    metrics.non_motion_recall = .94
    assert 'NON_MOTION_RECALL_TOO_LOW' in navigation_quality_issues(
        metrics, continuous_control=True)
    metrics.action_risk_evidence.regions[0].risk_hold_recall = .90
    issues = navigation_quality_issues(metrics, continuous_control=True)
    assert 'ACTION_RISK_SUMMARY_MISMATCH' in issues
    assert 'ACTION_RISK_REGION_0_RECALL_TOO_LOW' in issues


# 功能：
#   风险每个独立区域都必须满足支持量、召回、误差、对照和预算，不被平均数掩盖。
# 输入：
#   field、value、code：单项未达标的合成指标及拒绝代码。
# 输出：
#   None：通过断言核对。
@pytest.mark.parametrize(('field', 'value', 'code'), [
    ('unsafe_observation_count', 19, 'OBSERVATION_SUPPORT_INSUFFICIENT'),
    ('safe_motion_recall', .94, 'RECALL_TOO_LOW'),
    ('risk_hold_recall', .94, 'RECALL_TOO_LOW'),
    ('risk_mean_absolute_error', .21, 'ERROR_TOO_HIGH'),
    ('action_discrimination_fraction', .94, 'DISCRIMINATION_TOO_LOW'),
    ('p99_input_and_inference_ms', 21., 'LATENCY_TOO_HIGH')])
def test_each_risk_gate_remains_required(field, value, code):
    evidence = synthetic_risk_evidence()
    setattr(evidence.regions[0], field, value)
    assert 'ACTION_RISK_REGION_0_'+code in action_risk_evidence_issues(
        evidence, maximum_latency_ms=20.)


# 功能：
#   拒绝重复分组、伪造独立观测量或不完整类别计数。
# 输入：
#   change：绕过赋值验证后仍必须被重新核对的改变。
# 输出：
#   None：通过异常断言核对。
@pytest.mark.parametrize('change', [dict(safe_sample_count=61), dict(unsafe_observation_count=61),
                                  dict(test_groups=['d'*64]*2)])
def test_risk_evidence_rechecks_mutated_support(change):
    evidence = synthetic_risk_evidence()
    evidence.regions[0] = evidence.regions[0].model_copy(update=change)
    with pytest.raises(ValueError):
        action_risk_evidence_issues(evidence)


# 功能：
#   旧指标序列化不被默认新字段改变，跨区域重复来源不能扩大证据数量。
# 输入：
#   无：合成旧指标及风险来源。
# 输出：
#   None：通过断言核对。
def test_legacy_identity_and_repeated_region():
    assert 'action_risk_evidence' not in _metrics().model_dump()
    evidence = synthetic_risk_evidence()
    evidence.regions.append(evidence.regions[0])
    with pytest.raises(ValueError, match='DUPLICATE_REGION'):
        action_risk_evidence_issues(evidence)


# 功能：
#   报告转换时重新核对动作对照的分母与正确数，不接受好看的比例配上另一组计数。
# 输入：
#   mutation：对照计数或比例的单项破坏；None 表示完整有效的合成报告。
# 输出：
#   None：错误来源被拒绝，合法夹具转换后保持原模型身份。
@pytest.mark.parametrize('mutation', [None, {'observation_count': 59},
    {'correct_count': 61}, {'correct_count': True}, {'correct_fraction': .99}])
def test_report_discrimination_counts_are_bound(mutation):
    evidence = synthetic_risk_evidence()
    region = evidence.regions[0]
    report = dict(inference_performed=True, qualified_for_flight=False,
        model_sha256=evidence.model_sha256, teacher_config_sha256=evidence.teacher_config_sha256,
        metrics={**action_risk_summary(evidence), 'sample_count': 120},
        coverage={key: getattr(region, key) for key in ('sample_count',
            'independent_observation_count', 'safe_observation_count', 'unsafe_observation_count',
            'action_contrast_observation_count')},
        action_discrimination=dict(observation_count=60, correct_count=60, correct_fraction=1.),
        dataset_receipt_sha256=region.dataset_receipt_sha256, test_groups=region.test_groups,
        p99_input_and_inference_ms=.5)
    if mutation is not None:
        report['action_discrimination'].update(mutation)
        with pytest.raises(ValueError, match='SUPPORT_INVALID'):
            action_risk_evidence_from_reports([report])
    else:
        assert action_risk_evidence_from_reports([report]) == evidence


# 功能：
#   将已用于失败诊断的旧终测来源排除于未来准入，不将其伪装成新测试。
# 输入：
#   无；使用来源摘要而非产品数据的合成回执。
# 输出：
#   None：诊断组未排除或重复来源被接纳时失败。
def test_selection_diagnostic_groups_are_not_unseen():
    from dronedream_agent_core.training.artifact_assembly import expert_spatial_groups
    from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT

    receipt = dict(split_contract=SPATIAL_SPLIT_CONTRACT, training_groups=['a'*64],
        validation_groups=['b'*64], selection_diagnostic_sources=[
            dict(dataset_receipt_sha256='c'*64, groups=['d'*64, 'a'*64])])
    assert expert_spatial_groups('risk-critic', receipt) == [{'a'*64}, {'b'*64, 'd'*64}]
    receipt['selection_diagnostic_sources'] *= 2
    with pytest.raises(ValueError, match='DIAGNOSTICS_INVALID'):
        expert_spatial_groups('risk-critic', receipt)


# 功能：
#   保留控制模型祖先的调参路线，避免重训换目录后把这些路线当成新留出。
# 输入：
#   无；具有显式祖先标记的合成模型来源。
# 输出：
#   None：历史验证来源丢失、重叠或遗漏时失败。
def test_actor_ancestral_validation_groups_remain_excluded():
    from dronedream_agent_core.training.artifact_assembly import expert_spatial_groups
    from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT

    receipt = dict(split_contract=SPATIAL_SPLIT_CONTRACT, initial_policy_sha256='e'*64,
        metrics=dict(training_groups=['a'*64], validation_groups=['b'*64],
                     historical_validation_groups=['c'*64]))
    assert expert_spatial_groups('local-navigation-policy', receipt) == [
        {'a'*64}, {'b'*64, 'c'*64}]
    receipt['metrics']['historical_validation_groups'] = ['a'*64]
    with pytest.raises(ValueError, match='SPATIAL_GROUPS_OVERLAP'):
        expert_spatial_groups('local-navigation-policy', receipt)
    del receipt['metrics']['historical_validation_groups']
    with pytest.raises(ValueError, match='ANCESTRAL_VALIDATION'):
        expert_spatial_groups('local-navigation-policy', receipt)
