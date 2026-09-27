"""Boundaries of offline action-risk metrics, distinct from flight qualification."""

from dataclasses import replace

import numpy as np
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    _new_model,
    export_local_risk_critic_onnx,
    train_local_risk_critic,
)
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from dronedream_agent_core.training import action_risk_training as training
from dronedream_agent_core.training.action_risk_artifacts import ActionRiskDataset


# 功能：
#   在读取数据或创建输出目录前拒绝无效的训练分类间隔，不改变几何标签和验收门槛。
# 输入：
#   margin：待验证的间隔；tmp_path：隔离输出目录。
# 输出：
#   None：非法间隔不产生训练制品。
@pytest.mark.parametrize('margin', [True, '0.1', -.01, .5, float('nan'), float('inf')])
def test_invalid_training_margin_has_no_artifact(margin, tmp_path):
    output = tmp_path / 'candidate'
    with pytest.raises(ValueError, match='CLASSIFICATION_MARGIN_INVALID'):
        training.train_action_risk_expert([], [], LocalPolicyTrainingConfig(), output,
                                         classification_margin=margin)
    assert not output.exists()


# 功能：
#   实际训练量纲差异明显的合成样本，确认折叠后的原始输入与真实 ONNX 输出一致。
# 输入：
#   tmp_path：导出模型隔离目录。
# 输出：
#   None：原输入被改变、双类不能区分或导出不等价时测试失败。
def test_normalization_is_folded_without_changing_input_contract(tmp_path):
    samples = _samples()
    for sample in samples:
        sample.state_features[0] = 3.
        sample.risk_proposed_control[0] *= .025
    frozen = [sample.model_dump() for sample in samples]
    model, metrics = train_local_risk_critic(samples, LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=200, learning_rate=.01, batch_size=16),
        normalize_inputs=True)
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.
    assert [sample.model_dump() for sample in samples] == frozen
    path = tmp_path / 'risk.onnx'
    export_local_risk_critic_onnx(model, path)
    assert training.verify_risk_export(model, path.read_bytes(), samples) <= 1e-5
    assert np.count_nonzero(model.input_weight[0]) == 0


# 功能：
#   验证物理风险投影不改变源记录，部署原始张量中被屏蔽的目标和采样密度不能影响输出。
# 输入：
#   无。
# 输出：
#   None：投影与实际折叠权重不一致则测试失败。
def test_physical_projection_remains_invariant_on_raw_inputs():
    samples = _samples()
    frozen = [sample.model_dump() for sample in samples]
    projected = training.project_physical_risk_inputs(samples)
    model, _ = train_local_risk_critic(projected, LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=100, learning_rate=.01, batch_size=16),
        normalize_inputs=True)
    expected, _ = training._predict(model, samples)
    changed = [sample.model_copy(deep=True) for sample in samples]
    for sample in changed:
        sample.state_features[:7] = [1.] * 7
        sample.state_features[12] = .06
        offset = training.SENSOR_FEATURE_COUNT - training.FLIGHT_STATE_FEATURE_COUNT
        sample.realtime_features[offset:offset + 4] = [0., 0., 0., 1.]
        sample.state_features[14:] = [1.] * (len(sample.state_features) - 14)
        sample.realtime_features[2:5] = [1.] * 3
        sample.realtime_features[-10:] = [1.] * 10
    actual, _ = training._predict(model, changed)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0.)
    assert [sample.model_dump() for sample in samples] == frozen


# 功能：
#   同一物理观测只改变策略杆量尺度时投影完全相同，仍保留机体尺寸、净空、协方差和实测速度。
# 输入：
#   无。
# 输出：
#   None：策略单位与物理风险输入混淆时测试失败。
def test_physical_projection_keeps_metric_state_not_policy_scale():
    sample = _samples()[0]
    sample.state_features[10:14] = [.08, .04, .02, .05]
    sample.realtime_features[420:424] = [.01, -.02, .03, .2]
    changed = sample.model_copy(deep=True)
    changed.state_features[12] = .1
    changed.realtime_features[-10:-7] = [.2, -.4, .6]
    first, second = training.project_physical_risk_inputs([sample, changed])
    assert first.model_dump() == second.model_dump()
    assert first.state_features[10:14] == [.08, .04, 0., .05]
    assert first.realtime_features[420:424] == [.01, -.02, .03, .2]


# 功能：
#   为纯指标检查构造按观测配对的安全和危险动作，不作为已接纳训练证据。
# 输入：
#   无。
# 输出：
#   dataset：只供指标测试的合成风险数据。
def metric_dataset():
    samples = tuple(_samples())
    records = tuple({"observation_sha256": f"{i // 2:064x}"} for i in range(len(samples)))
    dataset = ActionRiskDataset(samples, records, (), frozenset({"a" * 64}), "b" * 64, "c" * 64, {})
    return dataset


# 功能：
#   将已用于诊断的来源登记为非独立测试，且这些记录不进入训练或开发的样本计数。
# 输入：
#   tmp_path：独占候选目录；本例合成数据仅测试来源登记，不满足正式训练覆盖。
# 输出：
#   None：诊断来源被混作训练样本或遗漏登记时失败。
def test_diagnostic_sources_are_recorded_without_training(tmp_path):
    base = metric_dataset()
    base = replace(base, receipt={'teacher_config': {'unit': True}})
    validation = replace(base, groups=frozenset({'d'*64}), receipt_sha256='e'*64)
    diagnostic = dict(dataset_receipt_sha256='1'*64,
                      teacher_config_sha256=base.teacher_config_sha256, groups=['f'*64])
    receipt = training.train_action_risk_expert([base], [validation],
        LocalPolicyTrainingConfig(), tmp_path/'candidate', diagnostic_sources=[diagnostic])
    assert receipt['optimized'] is False
    assert receipt['coverage']['training']['sample_count'] == len(base.samples)
    assert receipt['training_groups'] == ['a'*64]
    assert receipt['validation_groups'] == ['d'*64]
    assert receipt['selection_diagnostic_sources'] == [diagnostic]
    assert receipt['qualified_for_flight'] is False


# 功能：
#   验证动作区分指标拒绝越界、布尔和非一维预测，而非只看能否正确排序。
# 输入：
#   kind：预测数组的破坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["range", "boolean", "matrix"])
def test_discrimination_rejects_invalid_probabilities(kind):
    dataset = metric_dataset()
    predictions = np.asarray([s.risk_target for s in dataset.samples])
    if kind == "range":
        predictions = predictions * 20 - 10
    elif kind == "boolean":
        predictions = predictions.astype(bool)
    else:
        predictions = predictions.reshape(-1, 1)
    with pytest.raises(ValueError):
        training.action_discrimination([dataset], predictions)


# 功能：
#   验证覆盖统计不能把非法风险标签或伪造观测标识计入独立证据数量。
# 输入：
#   kind：标签或观测摘要的破坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["nan", "digest"])
def test_coverage_rejects_invalid_records(kind):
    dataset = metric_dataset()
    if kind == "nan":
        dataset.samples[0].__dict__["risk_target"] = float("nan")
    else:
        dataset.records[0]["observation_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        training.risk_class_coverage([dataset])


# 功能：
#   验证参考网络拒绝错误广播形状与关闭动作条件标记，防止用旧网络计算新控制风险。
# 输入：
#   kind：模型契约的破坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["shape", "not_action_conditioned", "infinity"])
def test_reference_rejects_bad_network(kind):
    model = _new_model(
        LocalPolicyTrainingConfig(hidden_feature_count=8),
        realtime_feature_count=POLICY_REALTIME_FEATURE_COUNT,
        action_conditioned=True,
    )
    if kind == "shape":
        model = replace(model, input_bias=np.zeros(1, dtype=np.float32))
    elif kind == "not_action_conditioned":
        model = replace(model, action_conditioned=False)
    else:
        model.risk_bias[:] = float("inf")
    with pytest.raises(ValueError):
        training._predict(model, _samples())
