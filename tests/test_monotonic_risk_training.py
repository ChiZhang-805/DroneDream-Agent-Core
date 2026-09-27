"""Exact partial monotonicity tests; synthetic labels never qualify a vehicle."""

import numpy as np
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    export_local_risk_critic_onnx,
    train_local_risk_critic,
)
from dronedream_agent_core.training.action_risk_training import _predict, verify_risk_export


# 功能：
#   训练显式距离约束的合成网络，核验全部距离偏导符号及真实 ONNX 导出的一致性。
# 输入：
#   normalize：是否折叠训练归一化；tmp_path：独立测试目录。
# 输出：
#   None：输入被改写、距离增加导致风险升高或导出不等价时失败。
@pytest.mark.parametrize('normalize', [False, True])
def test_distance_monotonicity_survives_compaction_and_export(normalize, tmp_path):
    samples = _samples()
    for row in samples:
        row.realtime_features[0:2] = [.1 if row.risk_target else .9] * 2
    frozen = [row.model_dump() for row in samples]
    model, metrics = train_local_risk_critic(samples, LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=120, batch_size=16, learning_rate=.01),
        normalize_inputs=normalize, monotonic_geometry=True)
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.
    assert [row.model_dump() for row in samples] == frozen
    distance_columns = [46+i for i in range(130) if i % 5 < 2]
    assert (model.input_weight[distance_columns] <= 0.).all()
    assert (model.risk_weight >= 0.).all()
    near, far = [], []
    generator = np.random.default_rng(221)
    for index in range(48):
        first = samples[index % len(samples)].model_copy(deep=True)
        second = first.model_copy(deep=True)
        for column in range(130):
            if column % 5 < 2:
                first.realtime_features[column] = float(generator.uniform(0., .5))
                second.realtime_features[column] = first.realtime_features[column] + .4
        near.append(first)
        far.append(second)
    first, _ = _predict(model, near)
    second, _ = _predict(model, far)
    assert (second <= first + 1e-7).all()
    path = tmp_path / 'risk.onnx'
    export_local_risk_critic_onnx(model, path)
    assert verify_risk_export(model, path.read_bytes(), near+far) <= 1e-5


# 功能：
#   拒绝真假值以外的约束选项，避免字符串 false 被隐式当作启用。
# 输入：
#   flag：不符合严格布尔约定的参数。
# 输出：
#   None：非法选项进入优化则失败。
@pytest.mark.parametrize('flag', [0, 1, 'false', None])
def test_monotonic_flag_requires_boolean(flag):
    with pytest.raises(ValueError, match='explicit boolean'):
        train_local_risk_critic(_samples(), LocalPolicyTrainingConfig(epoch_count=1),
                               monotonic_geometry=flag)
