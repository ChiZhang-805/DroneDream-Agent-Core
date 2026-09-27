"""Preserve measured heading source geometry without creating teacher-derived inputs."""

import json

import pytest
from test_executed_demonstration_dataset import no_command_history_fixture, rebind, source_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.demonstrations import collect_demonstrations
from dronedream_agent_core.training.heading_admission import read_heading_admission_inputs
from scripts.build_executed_policy_dataset import _write_dataset


# 功能：
#   创建精细控制的合成已执行记录，显式记录空动态物体列表而不是从缺失推断没有物体。
# 输入：
#   root：独立的合成运行目录。
# 输出：
#   row：真实生产编码器可读取的测试观测信封。
def precision_source(root, profile='precision'):
    row = source_fixture(root)
    snapshot = row['snapshot']
    snapshot['strategic_context']['task']['control_profile'] = profile
    snapshot['local_radius_m'] = 8.
    snapshot['dynamic_obstacles'] = []
    snapshot['snapshot_sha256'] = sha256_json({
        key: value for key, value in snapshot.items() if key != 'snapshot_sha256'
    })
    (root / 'learning-observations.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    rebind(root)
    return row


# 功能：
#   验证偏航来源按显式开关保留，并能重现同一行为样本输入，不改变原标签和历史数量。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：原始内容、标签或部署输入不一致时测试失败。
@pytest.mark.parametrize('profile',['precision','transit'])
def test_heading_source_round_trip_preserves_original(tmp_path, profile):
    root = tmp_path / 'run'
    row = precision_source(root, profile)
    original = collect_demonstrations([root], require_visual=False)
    exported = collect_demonstrations([root], require_visual=False, include_heading_sources=True)
    assert not original.heading_sources
    assert exported.samples == original.samples
    assert exported.observations == original.observations
    assert exported.heading_sources == [{
        'snapshot': row['snapshot'], 'recorded_at_unix_ms': row['recorded_at_unix_ms'],
    }]
    # 仅检验发布器和读取器的字节链路；合成同一语料不作训练/验证独立性证明。
    staging = tmp_path / 'staging'
    staging.mkdir()
    receipt = _write_dataset(staging, exported, exported, visual_required=False)
    path = staging / 'validation-heading-sources.jsonl'
    inputs = read_heading_admission_inputs(path)
    assert len(inputs.features_for(exported.samples[0])) == 23
    output = receipt['outputs']['validation']
    assert output['heading_observation_count'] == 1
    assert len(output['heading_observations_sha256']) == 64


# 功能：
#   拒绝用同宽度旧记录补造偏航几何；未显式启用新分支时保留原采集读取能力。
# 输入：
#   tmp_path：独立目录；missing：缺失的原始字段名。
# 输出：
#   None：新分支导出拒绝缺信息的来源，原文件保持原样。
@pytest.mark.parametrize('missing', ['dynamic_obstacles', 'local_radius_m'])
def test_heading_export_does_not_invent_missing_geometry(tmp_path, missing):
    root = tmp_path / 'run'
    row = precision_source(root)
    snapshot = row['snapshot']
    snapshot.pop(missing)
    snapshot['snapshot_sha256'] = sha256_json({
        key: value for key, value in snapshot.items() if key != 'snapshot_sha256'
    })
    path = root / 'learning-observations.jsonl'
    content = json.dumps(row) + '\n'
    path.write_text(content, encoding='utf-8')
    rebind(root)
    assert collect_demonstrations([root], require_visual=False).samples
    with pytest.raises(ValueError, match='HEADING_OBSERVATION'):
        collect_demonstrations([root], require_visual=False, include_heading_sources=True)
    assert path.read_text(encoding='utf-8') == content


# 功能：
#   保存无执行标签的精细控制历史，额外偏航输入只绑定实际计分帧，不凭缺少侧输入删除历史。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：完整历史保留两条、动作标签及偏航来源各一条，否则失败。
def test_heading_export_keeps_unlabelled_history_without_inventing_geometry(tmp_path):
    root = tmp_path / 'run'
    original, later, witness = no_command_history_fixture(root)
    for row in (original, later):
        snapshot = row['snapshot']
        snapshot['strategic_context']['task']['control_profile'] = 'precision'
        if row is original:
            snapshot.update(local_radius_m=8., dynamic_obstacles=[])
        snapshot['snapshot_sha256'] = sha256_json({
            key: value for key, value in snapshot.items() if key != 'snapshot_sha256'
        })
    commands = root / 'depth-local-safety-history.jsonl'
    commands.write_text(commands.read_text() + json.dumps(witness) + '\n')
    (root / 'learning-observations.jsonl').write_text(
        json.dumps(original) + '\n' + json.dumps(later) + '\n', encoding='utf-8',
    )
    (root / 'learning-observation-summary.json').write_text(
        json.dumps(dict(complete=True, submitted=2, completed=2)), encoding='utf-8',
    )
    rebind(root)
    corpus = collect_demonstrations([root], require_visual=True,
                                   allow_nonvisual_history=True, include_heading_sources=True)
    assert len(corpus.observations) == 2
    assert len(corpus.samples) == len(corpus.heading_sources) == 1
    assert corpus.heading_sources[0]['snapshot'] == original['snapshot']
