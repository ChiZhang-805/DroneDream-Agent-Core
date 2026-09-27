"""Causality, periodic geometry and regularization boundaries; not flight evidence."""

import math
import io

import pytest
import torch

from dronedream_agent_core.heading_context import encode_heading_context
from dronedream_agent_core.training.heading_stability import (
    ARCHITECTURE, StableHeadingPolicy, circular_heading, perturb_context, reflect_context, temporal_delta_loss,
    load_stable_heading_policy,
)
from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256, validate_geometry

GEOMETRY = (.1, .5, 1., -.2, .2, 1., 1., 0.)


# 功能：
#   核验时序输入的物理方向、过去回执及有效位，不用当前标签作为输入。
# 输入：
#   无。
# 输出：
#   None：陀螺仪 FLU 逆时针为正与手柄顺时针为正的符号转换正确。
def test_context_units_and_strict_past():
    result = encode_heading_context(GEOMETRY, 1000, gyro_flu_z_rad_s=math.radians(10),
        previous_geometry=GEOMETRY, previous_source_ms=900, previous_yaw_dps=-5., accepted_ms=950)
    assert len(result) == 23
    assert result[8:13] == (-.5, 1., -.25, .2, 1.)
    assert result[13:21] == GEOMETRY and result[21:] == (.4, 1.)


# 功能：
#   过期历史不更新时钟、不伪装为有效零动作。
# 输入：
#   无。
# 输出：
#   None：过期动作和几何都降为明确缺测。
def test_stale_context_resets_values_and_masks():
    result = encode_heading_context(GEOMETRY, 1000, previous_geometry=GEOMETRY,
        previous_source_ms=700, previous_yaw_dps=5., accepted_ms=749)
    assert result[8:] == (0.,)*15


# 功能：
#   拒绝未来、同毫秒、损坏或不完整的过去信息。
# 输入：
#   kwargs：单项非法输入。
# 输出：
#   None：编码器在训练和实时共用边界拒绝非法输入。
@pytest.mark.parametrize('kwargs', [dict(previous_yaw_dps=2., accepted_ms=1000),
    dict(previous_yaw_dps=2., accepted_ms=1001), dict(previous_yaw_dps=21., accepted_ms=999),
    dict(previous_yaw_dps=2.), dict(previous_geometry=GEOMETRY, previous_source_ms=1000),
    dict(previous_geometry=GEOMETRY), dict(gyro_flu_z_rad_s=float('nan'))])
def test_context_rejects_invalid_causal_input(kwargs):
    with pytest.raises(ValueError, match='HEADING_CONTEXT'):
        encode_heading_context(GEOMETRY, 1000, **kwargs)


# 功能：
#   周期角度跨越正负半圈不产生数值大跳，缺测方向不产生虚假余弦常数。
# 输入：
#   无。
# 输出：
#   None：接缝连续且缺测输出为零。
def test_circular_features_are_continuous_and_masked():
    x = torch.zeros(2, 8)
    x[:, 2] = 1.
    x[:, 0] = torch.tensor([1.-1e-6, -1.+1e-6])
    features = circular_heading(x)
    assert torch.max((features[0]-features[1]).abs()) < 3e-5
    assert torch.equal(circular_heading(torch.zeros(1, 8)), torch.zeros(1, 10))


# 功能：
#   镜像两次恢复原记录，有界扰动不改变距离、时钟或原张量。
# 输入：
#   无。
# 输出：
#   None：增强不污染原始观测和标签。
def test_context_augmentation_preserves_inputs_and_metadata():
    x = torch.tensor([encode_heading_context(GEOMETRY, 1000, previous_geometry=GEOMETRY, previous_source_ms=900)])
    y = torch.tensor([[.5]])
    original = x.clone()
    reflected, labels = reflect_context(x, y)
    twice, original_labels = reflect_context(reflected, labels)
    assert torch.equal(twice, x) and torch.equal(original_labels, y)
    noisy = perturb_context(x, torch.tensor([[.25/180]]))
    unchanged = [i for i in range(23) if i not in (0, 3, 13, 16)]
    assert torch.equal(noisy[:, unchanged], x[:, unchanged]) and torch.equal(x, original)


# 功能：
#   时间约束只惩罚动作变化的预测误差，不惩罚正确的紧急转向。
# 输入：
#   无。
# 输出：
#   None：真实变化匹配时无损失，缺测配对不参与约束。
def test_temporal_loss_does_not_suppress_true_manoeuvres():
    current, previous = torch.tensor([[1.], [-1.]], requires_grad=True), torch.tensor([[-1.], [1.]])
    valid = torch.tensor([True, False])
    loss = temporal_delta_loss(current, previous, current.detach(), previous, valid)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(current.grad).all()


# 功能：
#   核对两种输入分支的输出边界与梯度，避免缺测上下文影响纯几何基线。
# 输入：
#   context：是否启用时序信息。
# 输出：
#   None：模型输出有界，纯几何分支不读取历史。
@pytest.mark.parametrize('context', [False, True])
def test_stable_policy_contract(context):
    model = StableHeadingPolicy(use_context=context)
    x = torch.zeros(4, 23)
    result = model(x)
    assert result.shape == (4, 1) and (result.abs() <= 1).all()
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if not context:
        x[:, 8:] = 1.
        assert torch.equal(result, model(x))


# 功能：
#   检查新研究加载器拒绝旧契约、权限伪装、错误结构以及不有限权重。
# 输入：
#   damage：正常或单项损坏的检查点类型。
# 输出：
#   None：合法权重恢复一致，非法权重不获得模型实例。
@pytest.mark.parametrize('damage', ['none', 'contract', 'data', 'authority', 'flag', 'shape', 'nan', 'dtype', 'extra'])
def test_stability_checkpoint_contract(damage):
    model = StableHeadingPolicy(use_context=True)
    payload = dict(architecture=ARCHITECTURE, feature_contract_sha256=HEADING_CONTEXT_SHA256,
        plan_sha256='a'*64, data_sha256='b'*64, use_context=True, yaw_limit_dps=20.,
        state_dict=dict(model.state_dict()), qualified_for_flight=False)
    if damage in ('contract', 'data', 'authority', 'flag', 'extra'):
        key, value = {'contract': ('feature_contract_sha256', 'c'*64), 'data': ('data_sha256', 'c'*64),
            'authority': ('qualified_for_flight', True), 'flag': ('use_context', 1), 'extra': ('unexpected', 1)}[damage]
        payload[key] = value
    elif damage in ('shape', 'nan', 'dtype'):
        key = next(iter(payload['state_dict']))
        weight = payload['state_dict'][key]
        payload['state_dict'][key] = {'shape': torch.zeros(1), 'nan': torch.full_like(weight, float('nan')),
            'dtype': weight.double()}[damage]
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    if damage == 'none':
        loaded = load_stable_heading_policy(buffer.getvalue(), plan_sha256='a'*64, data_sha256='b'*64)
        assert torch.equal(loaded(torch.zeros(2, 23)), model(torch.zeros(2, 23)))
    else:
        with pytest.raises(ValueError, match='STABILITY_CHECKPOINT'):
            load_stable_heading_policy(buffer.getvalue(), plan_sha256='a'*64, data_sha256='b'*64)


# 功能：
#   拒绝数值溢出、无效方位及缺测目标伪装，保留近重合目标的有效距离。
# 输入：
#   无。
# 输出：
#   None：非法原始输入明确失败，而不是溢出或静默截断。
def test_geometry_missing_flags_and_numeric_overflow():
    for index, value in ((0, 10**400), (0, float('nan')), (2, 0.), (6, 0.)):
        geometry = list(GEOMETRY)
        geometry[index] = value
        with pytest.raises(ValueError, match='HEADING_CONTEXT'):
            validate_geometry(geometry)
    with pytest.raises(ValueError, match='HEADING_CONTEXT'):
        encode_heading_context(GEOMETRY, 1000, gyro_flu_z_rad_s=10**400)
    assert validate_geometry((0., .01, 0., 0., 0., 0., 0., 0.))[1] == .01
