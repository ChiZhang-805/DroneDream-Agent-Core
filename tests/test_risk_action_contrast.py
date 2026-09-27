"""Numerical and provenance checks; synthetic examples do not qualify flight."""

import numpy as np
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core import local_policy_training
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    train_local_risk_critic,
)
from dronedream_agent_core.training.risk_action_contrast import (
    action_contrast_loss,
    action_group_batches,
    prepare_action_groups,
)


# 功能：
#   用有限差分核验排序损失的 logits 梯度，包含并列标签下最差动作对。
# 输入：
#   无；标签和预测均为显式合成数据。
# 输出：
#   None：梯度方向或幅度错误时断言失败。
def test_contrast_gradient_matches_finite_difference():
    logits = np.array([[-.2], [.1], [-.1], [.2]], dtype=np.float64)
    targets = np.array([[.8], [.8], [.1], [.1]])
    probabilities = 1 / (1 + np.exp(-logits))
    loss, delta = action_contrast_loss(probabilities, targets, [(0, 4)], .7)
    assert loss > 0 and delta[0] < 0 and delta[3] > 0
    assert delta[1] == delta[2] == 0
    for index in range(4):
        left, right = logits.copy(), logits.copy()
        left[index] -= 1e-6
        right[index] += 1e-6
        lower = action_contrast_loss(1 / (1 + np.exp(-left)), targets, [(0, 4)], .7)[0]
        upper = action_contrast_loss(1 / (1 + np.exp(-right)), targets, [(0, 4)], .7)[0]
        assert delta[index, 0] == pytest.approx((upper - lower) / 2e-6, abs=1e-8)


# 功能：
#   保持完整观测组且覆盖每个样本一次，超过期望批大小的单组也不能拆散。
# 输入：
#   无；两组不同动作、相同观测的合成矩阵。
# 输出：
#   None：组身份、完整性或批边界不一致时失败。
def test_group_batches_preserve_every_action():
    inputs = np.zeros((8, 9))
    inputs[:, -1] = np.arange(8)
    groups = prepare_action_groups(inputs, ['a' * 64] * 5 + ['b' * 64] * 3, .5)
    batches = list(action_group_batches(groups, 4, np.random.default_rng(1)))
    assert sorted(len(indices) for indices, _ in batches) == [3, 5]
    assert sorted(np.concatenate([indices for indices, _ in batches]).tolist()) == list(range(8))
    assert all(bounds == [(0, len(indices))] for indices, bounds in batches)
    inputs[1, 0] = .01
    with pytest.raises(ValueError, match='OBSERVATION_CHANGED'):
        prepare_action_groups(inputs, ['a' * 64] * 5 + ['b' * 64] * 3, .5)


# 功能：
#   拒绝错误权重和缺失观测身份，避免把启用失败默默变为普通训练。
# 输入：
#   weight：无效或缺少对应组的训练权重。
# 输出：
#   None：未拒绝非法配置时失败。
@pytest.mark.parametrize('weight', [True, -.1, 4.1, float('nan'), float('inf'), .5])
def test_invalid_contrast_configuration_rejected(weight):
    with pytest.raises(ValueError):
        prepare_action_groups(np.zeros((2, 9)), None, weight)


# 功能：
#   排序已满足或没有可区别风险时不施加额外梯度，不强行修改等价动作。
# 输入：
#   无；两个手工可计算的观测组。
# 输出：
#   None：无必要的梯度出现时失败。
def test_no_contrast_for_equal_labels_or_satisfied_order():
    probabilities = np.array([[.9], [.1], [.2], [.8]])
    targets = np.array([[.9], [.1], [.5], [.5]])
    loss, delta = action_contrast_loss(probabilities, targets, [(0, 2), (2, 4)], 1.)
    assert loss == 0 and not delta.any()


# 功能：
#   真正训练合成动作条件网络，检查分组排序和默认关闭路径均能保持动作识别。
# 输入：
#   无；只用本文件的合成动作，不使用产品留出数据。
# 输出：
#   None：训练数值或默认兼容性不符合预期时失败。
def test_training_uses_groups_and_default_is_unchanged():
    samples = _samples()
    config = LocalPolicyTrainingConfig(hidden_feature_count=16, epoch_count=100,
                                       learning_rate=.01, batch_size=16)
    first, _ = train_local_risk_critic(samples, config)
    second, _ = train_local_risk_critic(samples, config, contrast_loss_weight=0.)
    for field in ('input_weight', 'input_bias', 'risk_weight', 'risk_bias'):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    _, metrics = train_local_risk_critic(samples, config, normalize_inputs=True,
        contrast_loss_weight=.5, contrast_group_ids=['a' * 64] * len(samples))
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.


# 功能：
#   核验纯安全和纯危险观测批仍保留全局类别成本，不能被各批自己的权重和抵消。
# 输入：
#   monkeypatch：记录真实优化器收到的梯度，不更新权重以消除批次顺序影响。
# 输出：
#   None：两种类别梯度比例没有保留指定成本时失败。
def test_uniform_observation_batches_keep_global_class_cost(monkeypatch):
    samples = _samples()
    groups = [('b' if sample.risk_target else 'a') * 64 for sample in samples]
    config = LocalPolicyTrainingConfig(hidden_feature_count=16, epoch_count=1, batch_size=16)
    captured = []

    # 功能：
    #   保存每批输出偏置梯度，不修改初始模型。
    # 输入：
    #   args：真实 Adam 更新参数，其中第二项是各层梯度。
    # 输出：
    #   None：梯度副本加入 captured。
    def capture(*args):
        captured.append(float(args[1][-1][0]))

    monkeypatch.setattr(local_policy_training, '_adam_update', capture)
    for cost in (1., 2.):
        train_local_risk_critic(samples, config, contrast_group_ids=groups,
                               contrast_loss_weight=.5, unsafe_sample_cost=cost)
    baseline = sorted(captured[:2])
    adjusted = sorted(captured[2:])
    assert baseline[0] < 0 < baseline[1]
    # 两类等量时全局均值从 1 变为 1.5；危险成本加倍、安全成本不变。
    assert adjusted[0] / baseline[0] == pytest.approx(4 / 3, rel=1e-6)
    assert adjusted[1] / baseline[1] == pytest.approx(2 / 3, rel=1e-6)
