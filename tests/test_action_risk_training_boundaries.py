"""Boundaries of offline action-risk metrics, distinct from flight qualification."""

from dataclasses import replace

import numpy as np
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingConfig, _new_model
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from dronedream_agent_core.training import action_risk_training as training
from dronedream_agent_core.training.action_risk_artifacts import ActionRiskDataset


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
