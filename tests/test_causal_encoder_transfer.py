"""Explicit encoder-only initialization and inherited holdout boundaries; synthetic fixtures only."""

import hashlib
import json
import sys

import pytest
import torch
from test_causal_training_cli import arguments
from test_ppo_cli_memory import replay_fixture

from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    initialize_sensor_encoder,
    load_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    require_unseen_validation,
)
from scripts import train_causal_control_role as cli


# 功能：
#   验证只复制编码器／循环层，各输出头保持本次初始化，后续更新不改源权重。
# 输入：
#   无：使用两个不同随机初始化的离线测试网络。
# 输出：
#   None：全部隔离与键集合断言通过后返回。
def test_transfer_copies_sensor_layers_only_without_parameter_aliases():
    config = CausalPolicyConfig(encoder_width=32, recurrent_width=32, head_width=32, epochs=1)
    source, target = CausalPilotPolicy(config), CausalPilotPolicy(config)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    original = {key: value.clone() for key, value in source.state_dict().items()}
    keys = initialize_sensor_encoder(target, source)
    assert keys and all(key.startswith(('observation_encoder.', 'recurrent.')) for key in keys)
    for key, value in target.state_dict().items():
        assert torch.equal(value, original[key] if key in keys else before[key])
        assert value.data_ptr() != source.state_dict()[key].data_ptr()
    with torch.no_grad():
        next(target.parameters()).add_(1.)
    assert all(torch.equal(value, original[key]) for key, value in source.state_dict().items())


# 功能：
#   结构、同对象、非有限或精度错误必须在复制任何参数之前拒绝。
# 输入：
#   damage：合成源网络的破坏方式。
# 输出：
#   None：源无效时目标保持逐值不变。
@pytest.mark.parametrize('damage', ['same', 'width', 'history', 'visual', 'nan', 'float64'])
def test_invalid_transfer_is_atomic(damage):
    config = CausalPolicyConfig(encoder_width=32, recurrent_width=32, head_width=32)
    target = CausalPilotPolicy(config)
    source = CausalPilotPolicy(config)
    if damage == 'same':
        source = target
    elif damage in ('width', 'history', 'visual'):
        field, value = {'width': ('encoder_width', 64), 'history': ('history_length', 8),
                        'visual': ('visual_feature_count', 1)}[damage]
        source = CausalPilotPolicy(CausalPolicyConfig(**{**config.model_dump(), field: value}))
    elif damage == 'nan':
        with torch.no_grad():
            next(source.recurrent.parameters()).fill_(float('nan'))
    else:
        source.double()
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match='CAUSAL_ENCODER_TRANSFER'):
        initialize_sensor_encoder(target, source)
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


# 功能：
#   验证完整热启动和部分编码迁移互斥，在构造任何训练张量前失败。
# 输入：
#   无：只构造离线小网络，不需要真实采集数据。
# 输出：
#   None：冲突被明确拒绝。
def test_initialization_modes_are_exclusive():
    config = CausalPolicyConfig(epochs=1)
    source = CausalPilotPolicy(config)
    with pytest.raises(ValueError, match='INITIALIZATION_MODES_CONFLICT'):
        train_causal_policy([], [], config, initial_policy=source, initial_encoder=source)


# 功能：
#   验证旧基座训练组在新回执继续保留，下一次精调不能将祖先路线伪装为未见留出。
# 输入：
#   无：使用明确的合成路线摘要。
# 输出：
#   None：指标原对象不变，祖先组被后续留出校验拒绝。
def test_transfer_and_warmstart_keep_ancestral_training_groups():
    original = {'training_groups': ['b' * 64], 'validation_groups': ['c' * 64]}
    ancestor = dict(split_contract=CAUSAL_SPLIT_CONTRACT,
        metrics={'training_groups': ['a' * 64], 'validation_groups': ['d' * 64],
                 'historical_validation_groups': ['e' * 64]})
    merged = cli.inherit_training_provenance(original, [ancestor], {'c' * 64})
    assert original['training_groups'] == ['b' * 64]
    assert merged['training_groups'] == ['a' * 64, 'b' * 64]
    assert merged['historical_validation_groups'] == ['d' * 64, 'e' * 64]
    with pytest.raises(ValueError, match='ALREADY_TRAINED_ON_VALIDATION'):
        require_unseen_validation(dict(split_contract=CAUSAL_SPLIT_CONTRACT, metrics=merged), {'a' * 64})


# 功能：
#   使用真实检查点复核跨角色迁移的来源、架构、角色和留出拒绝边界。
# 输入：
#   tmp_path：合成回放专属目录。
#   damage：待验证的来源破坏方式，None 为合法导航到恢复迁移。
# 输出：
#   None：非法来源被拒绝，合法结果准确记录源角色与权重摘要。
@pytest.mark.parametrize('damage', [None, 'hash', 'role', 'config', 'validation', 'missing', 'architecture'])
def test_transfer_receipt_and_holdout_boundaries(tmp_path, damage):
    _, content, raw = replay_fixture(tmp_path)
    receipt = json.loads(raw)
    config = CausalPolicyConfig.model_validate(receipt['config'])
    validation = set(receipt['metrics']['validation_groups'])
    if damage == 'hash':
        receipt['checkpoint_sha256'] = '0' * 64
    elif damage == 'role':
        receipt['expert_role'] = 'recovery-policy'
    elif damage == 'config':
        receipt['config']['seed'] += 1
    elif damage == 'validation':
        validation = set(receipt['metrics']['training_groups'])
    elif damage == 'architecture':
        config = config.model_copy(update={'encoder_width': 64})
    path = tmp_path / 'source-receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    options = dict(expert_role='recovery-policy', config=config,
                   visual_input_contract=None, validation_groups=validation)
    if damage is not None:
        with pytest.raises(ValueError):
            cli.load_encoder_transfer(tmp_path / 'base.pt', None if damage == 'missing' else path, **options)
    else:
        _, provenance, bound = cli.load_encoder_transfer(tmp_path / 'base.pt', path, **options)
        assert provenance['source_expert_role'] == 'local-navigation-policy'
        assert provenance['checkpoint_sha256'] == hashlib.sha256(content).hexdigest()
        assert provenance['output_heads_copied'] is False
        assert bound == receipt


# 功能：
#   经真实 CLI 完成编码迁移或同角色热启动训练与 ONNX 导出，保留源检查点并累积来源。
# 输入：
#   tmp_path：隔离的合成数据与输出目录。
#   monkeypatch：仅替换命令行参数，不替换训练器。
#   mode：明确选择编码器迁移或完整同角色热启动。
# 输出：
#   None：实际训练产物与迁移回执通过检查。
@pytest.mark.parametrize('mode', ['encoder', 'warmstart'])
def test_actual_cli_encoder_transfer_preserves_source_and_provenance(tmp_path, monkeypatch, mode):
    argv, config_path, _ = arguments(tmp_path, cli)
    content = (tmp_path / 'base.pt').read_bytes()
    config = json.loads(config_path.read_bytes())
    # 合成源标为精细专家；标签及目标仍为导航，验证入口允许明确的不同专家编码迁移。
    from test_mission_groups import group_fixture

    from dronedream_agent_core.control_feature_contract import (
        CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    )

    manifest = group_fixture()
    source_role = 'precision-maneuver-policy' if mode == 'encoder' else 'local-navigation-policy'
    receipt = dict(architecture='causal-gru-control', expert_role=source_role,
        checkpoint_sha256=hashlib.sha256(content).hexdigest(), config=config,
        visual_input_contract=None, feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        split_contract=CAUSAL_SPLIT_CONTRACT,
        metrics=dict(training_groups=[manifest.groups['train'], 'a' * 64],
                     validation_groups=[manifest.groups['val']]))
    path = tmp_path / 'source-receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    prefix = 'encoder' if mode == 'encoder' else 'base'
    monkeypatch.setattr(sys, 'argv', argv + [f'--{prefix}-policy', str(tmp_path / 'base.pt'),
                                          f'--{prefix}-training-receipt', str(path)])
    assert cli.main() == 0
    actual = json.loads((tmp_path / 'output/training-receipt.json').read_bytes())
    expected = 'transferred-sensor-encoder' if mode == 'encoder' else 'existing-causal-policy'
    assert actual['metrics']['initialization'] == expected
    assert len(actual['metrics']['transferred_encoder_state_keys']) == (6 if mode == 'encoder' else 0)
    assert 'a' * 64 in actual['metrics']['training_groups']
    if mode == 'encoder':
        assert actual['initial_policy_sha256'] is None
        assert actual['encoder_initialization']['checkpoint_sha256'] == hashlib.sha256(content).hexdigest()
    else:
        assert actual['encoder_initialization'] is None
        assert actual['initial_policy_sha256'] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / 'base.pt').read_bytes() == content
    model = load_causal_checkpoint(tmp_path / 'output/local-navigation-policy.pt')
    assert model.config.model_dump() == config
    assert actual['qualified_for_flight'] is False


# 功能：
#   缺失配对回执或混用两种初始化时，在读取不存在的来源和创建输出之前拒绝 CLI 请求。
# 输入：
#   tmp_path、monkeypatch：隔离的合成数据与命令行。
#   flags：非法初始化选项集合。
# 输出：
#   None：初始化错误明确失败，输出目录未创建。
@pytest.mark.parametrize('flags', [
    ['--encoder-policy'], ['--encoder-training-receipt'],
    ['--encoder-policy', '--encoder-training-receipt', '--base-policy'],
    ['--encoder-policy', '--encoder-training-receipt', '--base-training-receipt'],
])
def test_cli_rejects_incomplete_or_conflicting_initialization(tmp_path, monkeypatch, flags):
    argv, _, _ = arguments(tmp_path, cli)
    for flag in flags:
        argv += [flag, str(tmp_path / 'missing')]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='REQUIRES_POLICY_AND_RECEIPT|INITIALIZATION_MODES_CONFLICT'):
        cli.main()
    assert not (tmp_path / 'output').exists()
