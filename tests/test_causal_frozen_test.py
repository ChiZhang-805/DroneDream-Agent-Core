"""Frozen test evaluation uses synthetic sources here; formal test predictions remain untouched."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_control_shard_assembly import fixture

from dronedream_agent_core.training.causal_frozen_test import evaluate_frozen_causal_candidate
from dronedream_agent_core.training.causal_policy import CausalPolicyConfig
from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads
from dronedream_agent_core.training.control_shards import assemble_control_shards
from scripts import evaluate_frozen_causal_role as cli
from scripts import train_causal_control_role as trainer


# 功能：
#   在三个独立合成路线组实际组装、训练并导出小网络，避免替身掩盖入口契约错误。
# 输入：
#   tmp_path：本用例隔离目录。
# 输出：
#   candidate、data、receipt_sha、assembly_sha：实际产物与事先冻结的两个摘要。
def frozen_fixture(tmp_path):
    source, _ = fixture(tmp_path)
    data = tmp_path / 'data'
    assemble_control_shards(source, data, feature_count=1)
    config = tmp_path / 'config.json'
    config.write_text(CausalPolicyConfig(history_length=4, visual_feature_count=1,
        encoder_width=32, recurrent_width=32, head_width=32, epochs=1).model_dump_json())
    candidate = tmp_path / 'candidate'
    args = SimpleNamespace(train=data / 'training-replay.jsonl', validation=data / 'validation-replay.jsonl',
        training_observations=data / 'training-observations.jsonl', validation_observations=data / 'validation-observations.jsonl',
        stream_groups=data / 'stream-groups.json', training_visual_receipt=data / 'training-visual-receipt.json',
        validation_visual_receipt=data / 'validation-visual-receipt.json', config=config, output=candidate,
        expert_role='local-navigation-policy', base_policy=None, base_training_receipt=None, dagger_dataset=[])
    with training_cpu_threads(1):
        assert trainer.train_role(args) == 0
    receipt_sha = hashlib.sha256((candidate / 'training-receipt.json').read_bytes()).hexdigest()
    assembly_sha = hashlib.sha256((data / 'assembly-receipt.json').read_bytes()).hexdigest()
    return candidate, data, receipt_sha, assembly_sha


# 功能：
#   真正运行冻结测试的 Torch/ONNX 推理，结果只标为 test，所有输入文件保持不变。
# 输入：
#   tmp_path：独立合成数据和候选目录。
# 输出：
#   None：全部数值一致、身份和不可写边界成立。
def test_frozen_test_runs_actual_onnx_without_mutation(tmp_path):
    candidate, data, receipt_sha, assembly_sha = frozen_fixture(tmp_path)
    before = {path: path.read_bytes() for root in (candidate, data) for path in root.iterdir()}
    report = evaluate_frozen_causal_candidate(candidate, data, receipt_sha, assembly_sha)
    assert report['evaluation_split'] == 'test' and report['test_set_read']
    assert report['parity_passed'] and report['window_count'] > 0
    assert 'torch_validation' not in report and 'validation_groups' not in report
    assert report['torch_test']['test_window_count'] == report['window_count']
    assert not report['weights_modified'] and not report['qualified_for_flight']
    assert not report['candidate_selection_performed']
    assert before == {path: path.read_bytes() for path in before}


# 功能：
#   拒绝未冻结身份、旧训练集合以及迁移祖先的训练/调参泄漏，不能只看当前回放分组。
# 输入：
#   tmp_path：合成文件目录。
#   fault：需要破坏的绑定或祖先分组。
# 输出：
#   None：故障必须在任何最终测试模型推理前被拒绝。
@pytest.mark.parametrize('fault', ['receipt', 'assembly', 'graph', 'input', 'ancestor', 'tuning', 'ancestral-tuning'])
def test_frozen_test_rejects_identity_and_ancestry_faults(tmp_path, monkeypatch, fault):
    candidate, data, receipt_sha, assembly_sha = frozen_fixture(tmp_path)
    if fault == 'receipt':
        receipt_sha = 'a' * 64
    elif fault == 'assembly':
        assembly_sha = 'b' * 64
    elif fault == 'graph':
        with (candidate / 'local-navigation-policy.onnx').open('ab') as stream:
            stream.write(b'changed')
    else:
        path = candidate / 'training-receipt.json'
        receipt = json.loads(path.read_bytes())
        if fault == 'input':
            receipt['input_sha256']['train'] = 'c' * 64
        else:
            group = json.loads((data / 'stream-groups.json').read_bytes())['groups']['test']
            key = {'ancestor': 'training_groups', 'tuning': 'validation_groups',
                   'ancestral-tuning': 'historical_validation_groups'}[fault]
            receipt['metrics'][key].append(group)
        path.write_text(json.dumps(receipt))
        receipt_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = Mock(side_effect=AssertionError('must not predict from rejected inputs'))
    monkeypatch.setattr('dronedream_agent_core.training.causal_frozen_test.evaluate_causal_export', predictor)
    expected = dict(receipt='INPUT_CHANGED:training-receipt', assembly='CONTROL_ASSEMBLY_SOURCE_CHANGED',
        graph='INPUT_CHANGED:local-navigation-policy', input='WRONG_TRAINING_ASSEMBLY',
        ancestor='ALREADY_TRAINED_ON_VALIDATION', tuning='PREVIOUSLY_USED_FOR_TUNING',
        **{'ancestral-tuning': 'PREVIOUSLY_USED_FOR_TUNING'})
    with pytest.raises(ValueError, match=expected[fault]):
        evaluate_frozen_causal_candidate(candidate, data, receipt_sha, assembly_sha)
    predictor.assert_not_called()


# 功能：
#   即便外部重新冻结了数据摘要，也不能用训练时不同的视觉编码身份或当前调参记录冒充最终测试。
# 输入：
#   tmp_path、monkeypatch：隔离候选、数据和禁止预测的探针。
#   fault：视觉编码身份或当前验证来源冲突。
# 输出：
#   None：明确的契约/来源错误在实际预测前触发。
@pytest.mark.parametrize('fault', ['visual', 'validation-rows'])
def test_frozen_test_rejects_rebound_incompatible_sources(tmp_path, monkeypatch, fault):
    candidate, data, receipt_sha, _ = frozen_fixture(tmp_path)
    assembly_path = data / 'assembly-receipt.json'
    assembly = json.loads(assembly_path.read_bytes())
    if fault == 'visual':
        name = 'test-visual-receipt.json'
        payload = json.loads((data / name).read_bytes())
        for segment in payload['segments']:
            segment['encoding_receipt']['perception_encoder_sha256'] = 'b' * 64
        (data / name).write_text(json.dumps(payload))
        changed = [name]
        expected = 'VISUAL_INPUT_IDENTITY_MISMATCH'
    else:
        changed = ['test-replay.jsonl', 'test-observations.jsonl']
        for name in changed:
            (data / name).write_bytes((data / name.replace('test-', 'validation-', 1)).read_bytes())
        expected = 'MISSION_GROUP_LEAKAGE'
    for name in changed:
        assembly['file_sha256'][name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    assembly_path.write_text(json.dumps(assembly))
    assembly_sha = hashlib.sha256(assembly_path.read_bytes()).hexdigest()
    predictor = Mock(side_effect=AssertionError('must not predict'))
    monkeypatch.setattr('dronedream_agent_core.training.causal_frozen_test.evaluate_causal_export', predictor)
    with pytest.raises(ValueError, match=expected):
        evaluate_frozen_causal_candidate(candidate, data, receipt_sha, assembly_sha)
    predictor.assert_not_called()


# 功能：
#   已有报告不能被 CLI 覆盖，也不能因为再次调用而悄悄重复读取最终测试预测。
# 输入：
#   tmp_path、monkeypatch：隔离输出与命令行。
# 输出：
#   None：既有报告原样保留，评价器未被调用。
def test_frozen_test_cli_preserves_existing_report(tmp_path, monkeypatch):
    output = tmp_path / 'keep.json'
    output.write_text('preserve')
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--candidate', str(tmp_path), '--dataset', str(tmp_path),
        '--training-receipt-sha256', 'a' * 64, '--assembly-sha256', 'b' * 64, '--output', str(output)])
    predictor = Mock(side_effect=AssertionError('must not evaluate'))
    monkeypatch.setattr(cli, 'evaluate_frozen_causal_candidate', predictor)
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_text() == 'preserve'
    predictor.assert_not_called()
