"""Synthetic refinement contracts, not evidence of improved flight performance."""

import io

import pytest
import torch

from dronedream_agent_core.training.heading_refinement import (
    ConditionedHeadingPolicy, heading_training_weights, reflect_heading_training,
    validate_heading_training_tensors,
    load_conditioned_heading_policy, ARCHITECTURE, BoundedHeadingExport,
)
from dronedream_agent_core.heading_observation import HEADING_OBSERVATION_SHA256


# 功能：
#   验证镜像只改变两个方位和偏航方向，不改原张量、距离、状态位或样本来源。
# 输入：
#   无。
# 输出：
#   None：反射两次恢复原观测，零转向保持为零。
def test_reflection_is_an_owned_involution():
    x = torch.tensor([[.1, .5, 1., -.2, .3, 1., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]])
    y = torch.tensor([[.7], [0.]])
    original = x.clone()
    reflected, labels = reflect_heading_training(x, y)
    assert torch.equal(x, original)
    assert torch.equal(reflected[:, [0, 3]], -x[:, [0, 3]])
    twice, original_labels = reflect_heading_training(reflected, labels)
    assert torch.equal(twice, x) and torch.equal(original_labels, y)
    assert reflected.data_ptr() != x.data_ptr()


# 功能：
#   验证路线均衡和少见转向加权，顺序变化不改变各样本权重。
# 输入：
#   balanced：是否平衡组内转向类别。
# 输出：
#   None：所有权重有限且为正，两组总质量相同。
@pytest.mark.parametrize('balanced', [False, True])
def test_heading_weights_preserve_group_mass_and_order(balanced):
    x, y = torch.zeros(12, 8), torch.tensor([[0.]]*8 + [[.1], [-.4], [.9], [-.9]])
    groups = ['a']*10 + ['b']*2
    weights = heading_training_weights(x, y, groups, balance_turns=balanced)
    assert weights[:10].sum() == pytest.approx(6.)
    assert weights[10:].sum() == pytest.approx(6.)
    if balanced:
        assert weights[8] > weights[0]
    else:
        assert torch.equal(weights[:10], torch.full((10,), .6))
    reverse = heading_training_weights(x.flip(0), y.flip(0), groups[::-1], balance_turns=balanced)
    assert torch.allclose(reverse.flip(0), weights)


# 功能：
#   坏输入不能在增强时被默默修正或进入训练。
# 输入：
#   case：损坏数据类别。
# 输出：
#   None：训练边界拒绝对应损坏。
@pytest.mark.parametrize('case', ['nan', 'mask', 'angle', 'distance', 'axis', 'shape', 'dtype'])
def test_heading_training_rejects_invalid_tensors(case):
    x, y = torch.zeros(2, 8), torch.zeros(2, 1)
    if case == 'nan':
        x[0, 0] = float('nan')
    elif case == 'mask':
        x[0, 2] = .5
    elif case == 'angle':
        x[0, 3] = 1.1
    elif case == 'distance':
        x[0, 4] = -1.
    elif case == 'axis':
        y[0, 0] = 2.
    elif case == 'shape':
        y = torch.zeros(2)
    else:
        x = x.double()
    with pytest.raises(ValueError, match='HEADING_REFINEMENT'):
        validate_heading_training_tensors(x, y)


# 功能：
#   确认数值条件改进仍由网络学习输出，输入尺度不作为直接控制公式。
# 输入：
#   无。
# 输出：
#   None：控制幅度有界且所有可训练层有有限梯度。
def test_conditioned_policy_has_learned_bounded_output():
    model = ConditionedHeadingPolicy()
    result = model(torch.randn(4, 8))
    assert result.shape == (4, 1) and (result.abs() <= 1).all()
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert torch.equal(model(torch.ones(4, 8)), torch.zeros(4, 1))


# 功能：
#   检查候选加载的来源绑定与参数边界，防止变更尺度、旧协议或错误权重进入评价。
# 输入：
#   damage：单项损坏方式，空值表示合法检查点。
# 输出：
#   None：合法权重逐项恢复，损坏候选明确拒绝。
@pytest.mark.parametrize('damage', [None, 'scale', 'nonfinite', 'shape', 'dtype', 'extra', 'plan', 'schema', 'authority', 'limit'])
def test_conditioned_checkpoint_boundaries(damage):
    model = ConditionedHeadingPolicy()
    payload = dict(architecture=ARCHITECTURE, state_dict=dict(model.state_dict()), qualified_for_flight=False,
        heading_observation_sha256=HEADING_OBSERVATION_SHA256, yaw_limit_dps=20., plan_sha256='a'*64)
    if damage == 'scale':
        payload['state_dict']['input_scale'][0] = 1.
    elif damage == 'nonfinite':
        payload['state_dict']['network.0.weight'][0, 0] = float('nan')
    elif damage == 'shape':
        payload['state_dict']['network.0.weight'] = torch.zeros(1)
    elif damage == 'dtype':
        payload['state_dict']['input_scale'] = payload['state_dict']['input_scale'].double()
    elif damage == 'extra':
        payload['unbound'] = True
    elif damage == 'plan':
        payload['plan_sha256'] = 'b'*64
    elif damage == 'schema':
        payload['architecture'] = 'heading-observation-mlp-v1'
    elif damage == 'authority':
        payload['qualified_for_flight'] = True
    elif damage == 'limit':
        payload['yaw_limit_dps'] = 30.
    content = io.BytesIO()
    torch.save(payload, content)
    if damage:
        with pytest.raises(ValueError, match='CONDITIONED_HEADING'):
            load_conditioned_heading_policy(content.getvalue(), expected_plan_sha256='a'*64)
    else:
        restored = load_conditioned_heading_policy(content.getvalue(), expected_plan_sha256='a'*64)
        assert not restored.training
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())


# 功能：
#   检查导出边界保持权重不变，并在浮点微小越界时真实裁剪而非放宽验收阈值。
# 输入：
#   monkeypatch：测试框架的临时替换器。
# 输出：
#   None：越界轴被严格限制，正常轴保持原值。
def test_export_bound_is_explicit_and_preserves_normal_values(monkeypatch):
    model = ConditionedHeadingPolicy()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    values = torch.tensor([[1.0000001], [-1.0000001], [.25]])
    monkeypatch.setattr(model, 'forward', lambda features: values)
    result = BoundedHeadingExport(model)(torch.zeros(3, 8))
    assert torch.equal(result, torch.tensor([[1.], [-1.], [.25]]))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
