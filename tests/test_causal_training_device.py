"""Offline training device boundaries; CUDA execution is never simulated as real GPU evidence."""

import json
import sys
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import pytest
import torch
from test_causal_policy import samples
from test_causal_training_cli import arguments

from dronedream_agent_core.training.causal_device import causal_training_device
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    causal_examples,
    example_tensors,
    export_causal_policy,
    load_causal_checkpoint,
    save_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_regularization import CausalRegularization
from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads
from scripts import train_causal_control_ensemble as ensemble
from scripts import train_causal_control_role as role


# 功能：
#   构造带视觉块的独立合成训练和验证窗口，用于设备路径测试而非产品数据。
# 输入：
#   无。
# 输出：
#   config、train、validation：固定小网络配置和彼此来源隔离的窗口。
def device_examples():
    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                               head_width=32, visual_feature_count=7, epochs=2)
    rows = [samples(), samples(100, 'val')]
    for split in rows:
        for row in split:
            row.visual_features = [.25] * 7
    train = causal_examples(rows[0], stream_groups={'train': 'route-a'}, history_length=4)
    validation = causal_examples(rows[1], stream_groups={'val': 'route-b'}, history_length=4)
    return config, train, validation


# 功能：
#   拒绝未知设备、隐式自动选择及非字符串值，CPU 不初始化 CUDA。
# 输入：
#   monkeypatch：禁止 CUDA 探测的测试隔离器。
#   requested：非法设备值。
# 输出：
#   None：非法值被拒绝且 CPU 路径不触发 GPU。
@pytest.mark.parametrize('requested', ['auto', 'cuda:1', 'mps', None, True, 0])
def test_device_is_explicit_and_cpu_does_not_probe_cuda(monkeypatch, requested):
    probe = Mock(side_effect=AssertionError('unexpected CUDA access'))
    monkeypatch.setattr(torch.cuda, 'is_available', probe)
    assert causal_training_device('cpu').type == 'cpu'
    with pytest.raises(ValueError, match='DEVICE_INVALID'):
        causal_training_device(requested)
    probe.assert_not_called()


# 功能：
#   CUDA 枚举成功但内核无法执行时必须明确失败，不伪装为成功或改用 CPU。
# 输入：
#   monkeypatch：注入设备初始化失败。
# 输出：
#   None：原始初始化异常被保留为原因。
def test_cuda_kernel_probe_failure_is_not_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    failure = RuntimeError('driver initialization failed')
    monkeypatch.setattr(torch, 'ones', Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match='CUDA_INITIALIZATION_FAILED') as caught:
        causal_training_device('cuda')
    assert caught.value.__cause__ is failure


# 功能：
#   两个训练入口都必须在读取大型数据或创建输出前拒绝不可用 CUDA。
# 输入：
#   tmp_path：合成 CLI 夹具目录。
#   monkeypatch：隔离命令行和设备状态。
#   implementation：单专家或集合入口。
# 输出：
#   None：没有读取训练数据、没有输出目录且线程设置恢复。
@pytest.mark.parametrize('implementation', [role, ensemble])
def test_unavailable_cuda_fails_before_data_and_output(tmp_path, monkeypatch, implementation):
    argv, _, _ = arguments(tmp_path, implementation)
    monkeypatch.setattr(sys, 'argv', argv + ['--device', 'cuda'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    reader = Mock(side_effect=AssertionError('must not read training data'))
    monkeypatch.setattr(implementation, 'read_causal_training_inputs', reader)
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError, match='CUDA_UNAVAILABLE'):
        implementation.main()
    reader.assert_not_called()
    assert not (tmp_path / 'output').exists()
    assert torch.get_num_threads() == before


# 功能：
#   默认设备和显式 CPU 的优化结果必须逐值一致，并在回执记录实际设备与版本。
# 输入：
#   无：使用合成数据和固定视觉屏蔽随机流。
# 输出：
#   None：数据未被设备处理修改，模型保持 CPU 可导出状态。
def test_explicit_cpu_preserves_default_weights_and_input():
    config, train, validation = device_examples()
    original = [example.sample.model_dump_json() for example in train]
    regularization = CausalRegularization(balance_mission_groups=True, visual_block_dropout=.35)
    with training_cpu_threads(1):
        default, _ = train_causal_policy(train, validation, config, regularization=regularization)
        explicit, metrics = train_causal_policy(train, validation, config,
            regularization=regularization, device='cpu')
    assert all(torch.equal(value, explicit.state_dict()[name]) for name, value in default.state_dict().items())
    assert original == [example.sample.model_dump_json() for example in train]
    assert {parameter.device.type for parameter in explicit.parameters()} == {'cpu'}
    assert metrics['training_device'] == metrics['evaluation_device'] == metrics['export_device'] == 'cpu'
    assert metrics['torch_version'] == str(torch.__version__)
    assert metrics['torch_cuda_version'] is None


# 功能：
#   运行真实 CPU CLI 训练和导出，确认设备信息进入最终来源回执。
# 输入：
#   tmp_path：合成回放和独占输出目录。
#   monkeypatch：隔离 CLI 参数。
# 输出：
#   None：实际入口完成且回执报告 CPU，不赋予模型飞行资格。
def test_cpu_cli_records_device_and_exports(tmp_path, monkeypatch):
    argv, _, _ = arguments(tmp_path, role)
    monkeypatch.setattr(sys, 'argv', argv + ['--device', 'cpu'])
    assert role.main() == 0
    receipt = json.loads((tmp_path / 'output/training-receipt.json').read_bytes())
    assert receipt['metrics']['training_device'] == 'cpu'
    assert receipt['metrics']['evaluation_device'] == 'cpu'
    assert receipt['qualified_for_flight'] is False


# 功能：
#   在真实 CUDA 设备执行编码迁移、正则化与梯度更新，再在 CPU 加载和运行实际 ONNX。
# 输入：
#   tmp_path：独占导出目录。
# 输出：
#   None：没有 CUDA 时明确跳过，不能把本用例当作 GPU 已验证。
@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA device required')
def test_actual_cuda_training_returns_portable_cpu_weights_and_onnx(tmp_path):
    config, train, validation = device_examples()
    source = CausalPilotPolicy(config)
    before = {key: value.clone() for key, value in source.state_dict().items()}
    with training_cpu_threads(1):
        model, metrics = train_causal_policy(train, validation, config, initial_encoder=source,
            regularization=CausalRegularization(balance_mission_groups=True, visual_block_dropout=.35),
            device='cuda')
    assert metrics['training_device'] == 'cuda'
    assert metrics['evaluation_device'] == 'cpu'
    assert metrics['training_device_name'] == torch.cuda.get_device_name()
    assert all(torch.equal(value, before[key]) for key, value in source.state_dict().items())
    assert {parameter.device.type for parameter in model.parameters()} == {'cpu'}
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
    checkpoint = tmp_path / 'control.pt'
    save_causal_checkpoint(model, checkpoint)
    loaded = load_causal_checkpoint(checkpoint.read_bytes())
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    path = tmp_path / 'control.onnx'
    export_causal_policy(model, path)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    tensors = tuple(value[:1] for value in example_tensors(validation, 7))
    with torch.no_grad():
        expected = model(*tensors)
    actual = session.run(None, {node.name: tensor.numpy()
        for node, tensor in zip(session.get_inputs(), tensors, strict=True)})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference.numpy(), atol=1e-5, rtol=1e-5)
