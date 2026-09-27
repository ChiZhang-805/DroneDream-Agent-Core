"""Synthetic learned-component serialization checks; not flight evidence."""

import io

import pytest
import torch

from dronedream_agent_core.training.heading_policy import HeadingPolicy, load_heading_policy, save_heading_policy


# 功能：
#   验证小网络输出幅度、可训练梯度、检查点往返与防覆盖边界。
# 输入：
#   tmp_path：隔离的临时目录。
# 输出：
#   None：预测和参数保持一致且不会覆盖原权重。
def test_heading_model_output_and_checkpoint(tmp_path):
    model = HeadingPolicy()
    inputs = torch.randn(32, 8)
    prediction = model(inputs)
    assert prediction.shape == (32, 1)
    assert torch.isfinite(prediction).all() and prediction.abs().max() <= 1
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / 'weights.pt'
    save_heading_policy(model, path)
    loaded = load_heading_policy(path.read_bytes())
    assert torch.equal(loaded(inputs), model(inputs))
    with pytest.raises(FileExistsError):
        save_heading_policy(model, path)


# 功能：
#   拒绝感知契约被改、参数非有限、参数缺失及尺寸错误的检查点。
# 输入：
#   tmp_path：隔离目录；kind：篡改类别。
# 输出：
#   None：不兼容权重不能被静默采用。
@pytest.mark.parametrize('kind', ['contract', 'nonfinite', 'missing', 'shape', 'authority', 'extra'])
def test_heading_checkpoint_rejects_mutation(tmp_path, kind):
    path = tmp_path / 'candidate.pt'
    save_heading_policy(HeadingPolicy(), path)
    payload = torch.load(path, weights_only=True)
    key = next(iter(payload['state_dict']))
    if kind == 'contract':
        payload['heading_observation_sha256'] = '0' * 64
    elif kind == 'nonfinite':
        payload['state_dict'][key].flatten()[0] = float('nan')
    elif kind == 'missing':
        del payload['state_dict'][key]
    elif kind == 'authority':
        payload['qualified_for_flight'] = True
    elif kind == 'extra':
        payload['unknown_contract'] = 'unverified'
    else:
        payload['state_dict'][key] = torch.zeros(1)
    output = io.BytesIO()
    torch.save(payload, output)
    with pytest.raises(ValueError, match='HEADING_CHECKPOINT'):
        load_heading_policy(output.getvalue())
