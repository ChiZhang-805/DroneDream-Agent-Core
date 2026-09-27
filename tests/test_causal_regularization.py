"""Offline regularization cannot modify recorded demonstrations or deployed feature identity."""

from types import SimpleNamespace

import pytest
import torch
from test_causal_policy import samples

from dronedream_agent_core.training.causal_policy import (
    CausalPolicyConfig,
    causal_examples,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_regularization import (
    CausalRegularization,
    drop_visual_block,
    effective_sample_weights,
    validate_regularization,
)
from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads


# 功能：
#   路线均衡只改变优化权重，保留各组内部质量比例及原始记录，并拒绝不可表示的权重。
# 输入：
#   无。
# 输出：
#   None：约束失败时测试失败。
def test_group_balancing_preserves_quality_ratios():
    examples = [SimpleNamespace(group_id=group, sample=SimpleNamespace(sample_weight=weight))
                for group, weight in [('a', 1.), ('a', 3.), ('b', 2.)]]
    before = [(e.group_id, e.sample.sample_weight) for e in examples]
    weights = effective_sample_weights(examples, balance_groups=True)
    assert weights.tolist() == pytest.approx([0.75, 2.25, 3.0])
    assert weights[:2].sum() == weights[2]
    assert before == [(e.group_id, e.sample.sample_weight) for e in examples]
    assert effective_sample_weights(examples, balance_groups=False).tolist() == [1., 3., 2.]
    examples[0].sample.sample_weight = 1e-300
    with pytest.raises(ValueError, match='NOT_REPRESENTABLE'):
        effective_sample_weights(examples, balance_groups=False)


# 功能：
#   视觉屏蔽必须可复现且以整块为单位，不修改原特征、不放大保留值，零概率不消耗随机流。
# 输入：
#   无。
# 输出：
#   None：数据或随机性约束失败时测试失败。
def test_visual_dropout_is_reproducible_and_nonmutating():
    inputs = tuple(torch.ones(64, 3) for _ in range(6))
    first = torch.Generator().manual_seed(805)
    second = torch.Generator().manual_seed(805)
    result = drop_visual_block(inputs, .35, first)
    repeated = drop_visual_block(inputs, .35, second)
    assert torch.equal(result[-1], repeated[-1])
    assert torch.all(inputs[-1] == 1)
    assert ((result[-1] == 0).all(dim=1) | (result[-1] == 1).all(dim=1)).all()
    assert (result[-1] == 0).any() and (result[-1] == 1).any()
    assert all(a is b for a, b in zip(inputs[:-1], result[:-1], strict=True))
    state = first.get_state().clone()
    assert drop_visual_block(inputs, 0., first) is inputs
    assert torch.equal(state, first.get_state())


# 功能：
#   无视觉网络、非法概率或被篡改配置不能进入视觉正则化，默认关闭策略保持原行为。
# 输入：
#   probability：非法或不能用于无视觉网络的非零屏蔽概率。
# 输出：
#   None：错误配置未被拒绝时测试失败。
@pytest.mark.parametrize('probability', [-1., float('nan'), 1., True, .35])
def test_regularization_rejects_invalid_or_nonvisual_configuration(probability):
    config = CausalRegularization().model_copy(update={'visual_block_dropout': probability})
    with pytest.raises(ValueError):
        validate_regularization(config, 0)
    assert validate_regularization(None, 0).model_dump() == {
        'balance_mission_groups': False, 'visual_block_dropout': 0.0}


# 功能：
#   默认训练与显式关闭正则化产生相同权重；非零视觉正则化能完成训练且不改监督记录。
# 输入：
#   无。
# 输出：
#   None：训练结果或输入保护失败时测试失败。
def test_training_defaults_preserved_and_regularized_path_runs():
    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                               head_width=32, visual_feature_count=7, epochs=2)
    rows = [samples(), samples(100, 'val')]
    for split in rows:
        for row in split:
            row.visual_features = [.25] * 7
    train = causal_examples(rows[0], stream_groups={'train': 'route-a'}, history_length=4)
    held = causal_examples(rows[1], stream_groups={'val': 'route-b'}, history_length=4)
    original = [[row.model_dump_json() for row in split] for split in rows]
    with training_cpu_threads(1):
        base, _ = train_causal_policy(train, held, config)
        disabled = CausalRegularization()
        explicit, _ = train_causal_policy(train, held, config, regularization=disabled)
        assert all(torch.equal(value, explicit.state_dict()[name])
                   for name, value in base.state_dict().items())
        settings = CausalRegularization(balance_mission_groups=True, visual_block_dropout=.35)
        changed, metrics = train_causal_policy(train, held, config, regularization=settings)
        assert metrics['regularization']['visual_block_dropout'] == .35
        assert any(not torch.equal(value, changed.state_dict()[name])
                   for name, value in base.state_dict().items())
        assert metrics['qualified_for_flight'] is False
    assert original == [[row.model_dump_json() for row in split] for split in rows]
