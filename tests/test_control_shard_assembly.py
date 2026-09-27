"""Synthetic byte-lineage and split-isolation tests, never training evidence."""

import hashlib
import json

import pytest
from test_causal_policy import samples
from test_visual_training_lineage import encoding

import dronedream_agent_core.training.control_shards as shard_module
from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.training.causal_policy import CausalPolicyConfig
from dronedream_agent_core.training.causal_training_inputs import read_causal_training_inputs
from dronedream_agent_core.training.control_shards import (
    assemble_control_shards,
    compose_visual_shards,
)
from dronedream_agent_core.training.mission_groups import (
    MissionGroupManifest,
    mission_group_evidence,
)
from dronedream_agent_core.training.visual_lineage import (
    verify_visual_split,
    verify_visual_training_inputs,
)


# 功能：
#   构造三组明确标为合成的测试材料，各自独立路线和真实字节摘要用于验证合并器。
# 输入：
#   root：测试框架提供的临时目录。
# 输出：
#   path、inventory：测试索引的路径和对象。
def fixture(root):
    inventory = dict(schema_version='dronedream.control-preparation-inventory.v1',
                     visual_encoder_sha256='a' * 64, shards=[])
    for index, split in enumerate(('training', 'validation', 'test')):
        directory = root / split
        directory.mkdir()
        rows = samples(start=index * 1000, stream=split)
        rows = [row.model_copy(update=dict(visual_features=[.1],
            source_visual_sha256=None, source_snapshot_sha256=None)) for row in rows]
        history = [LocalPolicyObservation(**{key: getattr(row, key)
                   for key in LocalPolicyObservation.model_fields}) for row in rows]
        route = json.dumps({'positions_m': [{'x': index * 100., 'y': 0., 'z': 2.},
            {'x': index * 100. + 2., 'y': 0., 'z': 2.}]}).encode()
        evidence = mission_group_evidence(route, 'e' * 64)
        manifest = MissionGroupManifest(groups={split: evidence.group_sha256}, evidence=[evidence])
        parts = dict(encoded=('\n'.join(row.model_dump_json() for row in rows) + '\n').encode(),
                     history=('\n'.join(row.model_dump_json() for row in history) + '\n').encode(),
                     groups=manifest.model_dump_json().encode())
        parts['raw'] = ('\n'.join(row.model_copy(update={'visual_features': []}).model_dump_json()
                                 for row in rows) + '\n').encode()
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in parts.items()}
        receipt = encoding(parts['encoded'])
        receipt['source_policy_data_sha256'] = hashes['raw']
        receipt_bytes = json.dumps(receipt).encode()
        for name, filename in [('raw', f'{split}.jsonl'), ('encoded', f'{split}-visual.jsonl'),
                               ('history', f'{split}-observations.jsonl'), ('groups', 'stream-groups.json')]:
            (directory / filename).write_bytes(parts[name])
        (directory / f'{split}-visual-receipt.json').write_bytes(receipt_bytes)
        inventory['shards'].append(dict(split=split, directory=str(directory), file_sha256=hashes,
            visual_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest()))
    path = root / 'index.json'
    path.write_text(json.dumps(inventory), encoding='utf-8')
    return path, inventory


# 功能：
#   证明组合回执保留每段原样字节，可被现有训练入口核验，未产生新的视觉推理声明。
# 输入：
#   无。
# 输出：
#   None：组合身份或原样内容不符即失败。
def test_visual_composition_preserves_original_segments():
    parts = [b'first\n', b'second\n']
    data, receipt = compose_visual_shards(parts, [encoding(part) for part in parts], 1)
    assert data == b''.join(parts)
    assert receipt['schema_version'] == 'dronedream.visual-shard-composition.v1'
    other = b'validation\n'
    contract = verify_visual_training_inputs(training_content=data, validation_content=other,
        receipt_contents=[json.dumps(receipt).encode(), json.dumps(encoding(other)).encode()], feature_count=1)
    assert contract['perception_encoder_sha256'] == 'a' * 64


# 功能：
#   重绑整体摘要不能掩盖分片修改，长度、计数、嵌套或重复分片均须拒绝。
# 输入：
#   change：需要注入的损坏方式。
# 输出：
#   None：任何损坏均抛出 ValueError。
@pytest.mark.parametrize('change', ['bytes', 'length', 'count', 'nested', 'duplicate', 'encoder', 'trailing'])
def test_composition_rejects_tampering(change):
    data, receipt = compose_visual_shards([b'first\n', b'second\n'],
        [encoding(b'first\n'), encoding(b'second\n')], 1)
    if change == 'bytes':
        data = b'other\nsecond\n'
    elif change == 'length':
        receipt['segments'][0]['byte_length'] = True
    elif change == 'count':
        receipt['sample_count'] = True
    elif change == 'nested':
        receipt['segments'][0]['encoding_receipt'] = dict(schema_version='dronedream.visual-shard-composition.v1')
    elif change == 'duplicate':
        data = b'first\nfirst\n'
        receipt['segments'][1] = dict(receipt['segments'][0])
    elif change == 'encoder':
        receipt['segments'][1]['encoding_receipt']['perception_encoder_sha256'] = 'b' * 64
    elif change == 'trailing':
        data += b'extra\n'
    receipt['output_sha256'] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        verify_visual_split(data, json.dumps(receipt).encode(), feature_count=1)


# 功能：
#   完整三划分材料原样导出，重算窗口但不授予训练充分性或飞行资格，已有产物禁止覆盖。
# 输入：
#   tmp_path：测试临时目录。
# 输出：
#   None：核对原字节、三划分、最终回执及无覆盖行为。
def test_complete_assembly_preserves_bytes_and_has_no_authority(tmp_path):
    source, _ = fixture(tmp_path)
    target = tmp_path / 'assembled'
    report = assemble_control_shards(source, target, feature_count=1)
    for split in ('training', 'validation', 'test'):
        assert (target / f'{split}-replay.jsonl').read_bytes() == (tmp_path / split / f'{split}-visual.jsonl').read_bytes()
    assert report['training_ready'] is False
    for split in ('training', 'validation', 'test'):
        for role, count in report['counts'][split]['windows'].items():
            coverage = report['counts'][split]['window_coverage'][role]
            assert coverage['complete_windows'] == count
            assert coverage['independent_event_count'] is None
    assert report['qualified_for_flight'] is False
    assert (target / 'assembly-receipt.json').is_file()
    with pytest.raises(FileExistsError):
        assemble_control_shards(source, target, feature_count=1)


# 功能：
#   将三集合组合产物交给真正的训练读取边界，确保测试来源不会作为训练或验证样本读入。
# 输入：
#   tmp_path：合成材料临时目录。
# 输出：
#   None：实际读取器、视觉契约或集合隔离不一致时失败，不执行参数训练。
def test_assembled_files_load_through_actual_training_boundary(tmp_path):
    source, _ = fixture(tmp_path)
    target = tmp_path / 'assembled'
    assemble_control_shards(source, target, feature_count=1)
    config = tmp_path / 'config.json'
    config.write_text(CausalPolicyConfig(visual_feature_count=1).model_dump_json())
    content, parsed_config, decoded, manifest, groups = read_causal_training_inputs({
        'train': target / 'training-replay.jsonl', 'validation': target / 'validation-replay.jsonl',
        'training_observations': target / 'training-observations.jsonl',
        'validation_observations': target / 'validation-observations.jsonl',
        'stream_groups': target / 'stream-groups.json', 'config': config})
    assert parsed_config.visual_feature_count == 1
    assert {row.temporal_evidence.stream_id for row in decoded['training_replay']} == {'training'}
    assert {row.temporal_evidence.stream_id for row in decoded['validation_replay']} == {'validation'}
    verify_visual_training_inputs(training_content=content['train'], validation_content=content['validation'],
        receipt_contents=[(target / f'{split}-visual-receipt.json').read_bytes()
                          for split in ('training', 'validation')], feature_count=1)


# 功能：
#   损坏或缺失划分在创建输出前拒绝，重复采集来源不能提升计数。
# 输入：
#   tmp_path：临时目录；change：输入损坏方式。
# 输出：
#   None：拒绝并保持输出目录不存在。
@pytest.mark.parametrize('change', ['hash', 'duplicate', 'missing-test', 'bad-encoder'])
def test_assembly_failure_does_not_publish_partial_success(tmp_path, change):
    source, inventory = fixture(tmp_path)
    if change == 'hash':
        inventory['shards'][0]['file_sha256']['history'] = 'f' * 64
    elif change == 'duplicate':
        inventory['shards'].append(inventory['shards'][0])
    elif change == 'missing-test':
        inventory['shards'] = inventory['shards'][:-1]
    else:
        inventory['visual_encoder_sha256'] = 'b' * 64
    source.write_text(json.dumps(inventory), encoding='utf-8')
    target = tmp_path / 'failed'
    with pytest.raises(ValueError):
        assemble_control_shards(source, target, feature_count=1)
    assert not target.exists()


# 功能：
#   即使重绑视觉输出摘要，编码过程也不能改变教师监督；原标签字节变化同样拒绝。
# 输入：
#   tmp_path：合成数据临时目录；change：标签或原始字节损坏方式。
# 输出：
#   None：损坏在发布输出前被拒绝。
@pytest.mark.parametrize('change', ['raw-bytes', 'encoded-label'])
def test_assembly_rechecks_original_labels(tmp_path, change):
    source, inventory = fixture(tmp_path)
    shard = inventory['shards'][0]
    directory = tmp_path / 'training'
    if change == 'raw-bytes':
        with (directory / 'training.jsonl').open('ab') as stream:
            stream.write(b'\n')
    else:
        path = directory / 'training-visual.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]['risk_target'] = .85 if rows[0]['risk_target'] != .85 else .15
        content = b''.join(json.dumps(row).encode() + b'\n' for row in rows)
        path.write_bytes(content)
        shard['file_sha256']['encoded'] = hashlib.sha256(content).hexdigest()
        receipt_path = directory / 'training-visual-receipt.json'
        receipt = json.loads(receipt_path.read_bytes())
        receipt['output_sha256'] = shard['file_sha256']['encoded']
        receipt_bytes = json.dumps(receipt).encode()
        receipt_path.write_bytes(receipt_bytes)
        shard['visual_receipt_sha256'] = hashlib.sha256(receipt_bytes).hexdigest()
        source.write_text(json.dumps(inventory), encoding='utf-8')
    with pytest.raises(ValueError, match='CONTROL_ASSEMBLY_(SOURCE_CHANGED|ENCODING_CHANGED_SUPERVISION)'):
        assemble_control_shards(source, tmp_path / 'rejected', feature_count=1)
    assert not (tmp_path / 'rejected').exists()


# 功能：
#   原标签审计与最终训练数据分开计量，避免同一监督的两份表示重复消耗输出预算。
# 输入：
#   tmp_path：合成输入目录；monkeypatch：仅限本测试的容量限制替换器。
# 输出：
#   None：原标签加输出超预算、但二者各自合规时应完成组装。
def test_original_label_audit_has_separate_bounded_budget(tmp_path, monkeypatch):
    source, inventory = fixture(tmp_path)
    report = assemble_control_shards(source, tmp_path / 'reference', feature_count=1)
    output_bytes = sum((tmp_path / 'reference' / name).stat().st_size for name in report['file_sha256'])
    source_bytes = sum(path.stat().st_size for split in ('training', 'validation', 'test')
                       for path in (tmp_path / split).iterdir() if path.suffix == '.jsonl')
    limit = output_bytes + 4096
    assert source_bytes > limit
    monkeypatch.setattr(shard_module, 'MAX_REPLAY_TOTAL_BYTES', limit)
    result = assemble_control_shards(source, tmp_path / 'assembled', feature_count=1)
    assert result['file_sha256'] == report['file_sha256']


# 功能：
#   分离原标签预算不能解除最终合并文件的总容量限制。
# 输入：
#   tmp_path：合成输入目录；monkeypatch：容量限制替换器。
# 输出：
#   None：超过预算时必须拒绝且不发布成功目录。
def test_separate_audit_budget_does_not_allow_oversized_output(tmp_path, monkeypatch):
    source, _ = fixture(tmp_path)
    monkeypatch.setattr(shard_module, 'MAX_REPLAY_TOTAL_BYTES', 1024)
    with pytest.raises(ValueError, match='TOO_LARGE'):
        assemble_control_shards(source, tmp_path / 'rejected', feature_count=1)
    assert not (tmp_path / 'rejected').exists()
