"""Exact-zero optimizer compaction changes neither model inputs nor risk objectives."""

import numpy as np
from test_action_conditioned_risk import _samples

from dronedream_agent_core import local_policy_training as training


# 功能：
#   验证仅删除严格双零列，小幅物理信号和待正则化的非零旧权重必须保留。
# 输入：
#   无；使用显式的小型矩阵。
# 输出：
#   None：投影列或矩阵乘积不一致时测试失败。
def test_projection_preserves_small_signals_and_regularized_weights():
    inputs = np.asarray([[0., 0., 1e-12, 1.], [0., 0., 0., 2.]], dtype=np.float32)
    weight = np.asarray([[0., 0.], [1., 0.], [0., 0.], [.2, .3]], dtype=np.float32)
    projected, parameters, columns = training._risk_optimizer_projection(inputs, weight)
    assert columns.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(projected @ parameters, inputs @ weight)
    assert inputs.shape == (2, 4) and weight.shape == (4, 2)


# 功能：
#   对比同种子、同批次和同目标的完整优化与压缩优化，恢复权重宽度后逐项检查数值等价。
# 输入：
#   monkeypatch：仅在第二次对照运行中关闭计算压缩。
# 输出：
#   None：参数或预测超出浮点运算容差时测试失败。
def test_compact_training_matches_dense_reference(monkeypatch):
    samples = _samples()
    config = training.LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=30, batch_size=16,
        random_seed=17, learning_rate=.002, l2_weight=.001,
    )
    compact, compact_metrics = training.train_local_risk_critic(
        samples, config, normalize_inputs=True,
    )

    # 功能：
    #   保留所有矩阵列作为未优化的对照，不改变网络初始化、损失或随机顺序。
    # 输入：
    #   inputs、input_weight：原训练矩阵与权重。
    # 输出：
    #   result：原对象与表示无需恢复映射的 None。
    def dense_reference(inputs, input_weight):
        result = inputs, input_weight, None
        return result

    monkeypatch.setattr(training, '_risk_optimizer_projection', dense_reference)
    dense, dense_metrics = training.train_local_risk_critic(samples, config, normalize_inputs=True)
    for field in ('input_weight', 'input_bias', 'risk_weight', 'risk_bias'):
        np.testing.assert_allclose(
            getattr(compact, field), getattr(dense, field), atol=5e-6, rtol=5e-6)
    assert compact_metrics.model_dump() == dense_metrics.model_dump()


# 功能：
#   全零或完全有效矩阵都保持可计算，避免零宽特例以及不必要的全量复制。
# 输入：
#   无；使用全零和全一的独立矩阵。
# 输出：
#   None：输出形状或原对象复用不正确时测试失败。
def test_empty_projection_and_dense_identity():
    inputs = np.zeros((3, 4), dtype=np.float32)
    weight = np.zeros((4, 8), dtype=np.float32)
    projected, parameters, columns = training._risk_optimizer_projection(inputs, weight)
    assert columns.size == 0 and (projected @ parameters).shape == (3, 8)
    inputs[:] = 1.
    projected, parameters, columns = training._risk_optimizer_projection(inputs, weight)
    assert projected is inputs and parameters is weight and columns is None
