"""Explicitly synthetic coverage and integrity checks, never training material."""

from types import SimpleNamespace

import pytest
from test_executed_demonstration_dataset import source_fixture

from dronedream_agent_core.local_expert_harness import NAVIGATION_EXPERT_ROLES
from dronedream_agent_core.temporal_evidence import TemporalEvidence
from dronedream_agent_core.training.demonstration_readiness import (
    demonstration_readiness,
    label_history_coverage,
    temporal_coverage,
)
from dronedream_agent_core.training.demonstrations import collect_demonstrations


# 功能：
#   验证孤立的合法动作不能被报告成完整的时序训练窗口。
# 输入：
#   tmp_path：隔离的合成运行目录。
# 输出：
#   None：计数保留动作但明确显示窗口缺口。
def test_isolated_action_has_zero_windows(tmp_path):
    source_fixture(tmp_path / 'run')
    corpus = collect_demonstrations([tmp_path / 'run'], require_visual=False)
    report = demonstration_readiness(corpus)
    assert sum(row['executed_labels'] for row in report['roles'].values()) == 1
    assert all(row['complete_windows'] == 0 for row in report['roles'].values())
    assert not report['all_roles_have_windows'] and not report['qualified_for_flight']


# 功能：
#   验证完整性错误不被统计层吞掉并伪装成普通样本不足。
# 输入：
#   tmp_path：隔离的合成运行目录。
# 输出：
#   None：缺少任务来源映射时仍拒绝训练材料。
def test_readiness_does_not_hide_invalid_sources(tmp_path):
    source_fixture(tmp_path / 'run')
    corpus = collect_demonstrations([tmp_path / 'run'], require_visual=False)
    corpus.stream_groups.clear()
    with pytest.raises(ValueError, match='MISSION_GROUP_MISSING'):
        demonstration_readiness(corpus)


# 功能：
#   复核二百五十毫秒的边界和显式重置，诊断必须与部署端窗口接纳一致。
# 输入：
#   gap：相邻真实来源间隔；reset：是否在最后一个观测显式重置。
# 输出：
#   None：诊断不把断档历史算成完整窗口。
@pytest.mark.parametrize('gap,reset,ready,longest', [(250, False, 1, 16), (251, False, 0, 1), (200, True, 0, 15)])
def test_temporal_coverage_uses_actual_history_boundary(gap, reset, ready, longest):
    observations = [SimpleNamespace(temporal_evidence=TemporalEvidence(stream_id='synthetic',
        sample_sha256=f'{index:064x}', observed_at_unix_ms=1000 + index * gap,
        reset_history=reset and index == 15)) for index in range(16)]
    report = temporal_coverage(observations)
    assert report['complete_history_observations'] == ready
    assert report['longest_contiguous_observations'] == longest
    assert report['gap_reset_count'] == (15 if gap == 251 else 0)
    assert report['explicit_reset_count'] == int(reset)
    assert report['source_gap_max_ms'] == gap


# 功能：
#   拒绝重复来源，不通过重复同一帧填满历史；空语料不虚构间隔统计。
# 输入：
#   无。
# 输出：
#   None：重复来源报错，空统计保持明确缺失。
def test_temporal_coverage_rejects_replay_and_handles_empty():
    evidence = TemporalEvidence(stream_id='synthetic', sample_sha256='a' * 64, observed_at_unix_ms=1000)
    row = SimpleNamespace(temporal_evidence=evidence)
    with pytest.raises(ValueError, match='REPEATED_SOURCE'):
        temporal_coverage([row, row])
    assert temporal_coverage([])['source_gap_max_ms'] is None


# 功能：
#   构造只供历史诊断的合成来源，不生成正式训练图像或执行回执。
# 输入：
#   index：独立摘要序号。
#   timestamp：来源毫秒时刻。
#   stream：合成来源流。
#   reset：显式重置标志。
# 输出：
#   row：带身份和合法角色的合成诊断记录。
def history_row(index, timestamp, stream='synthetic', reset=False):
    row = SimpleNamespace(temporal_evidence=TemporalEvidence(stream_id=stream,
        sample_sha256=f'{index:064x}', observed_at_unix_ms=timestamp, reset_history=reset),
        navigation_expert_role=NAVIGATION_EXPERT_ROLES[0])
    return row


# 功能：
#   证明其他角色和无标签观测同样补充真实历史，角色切换不应凭空清零。
# 输入：
#   无。
# 输出：
#   None：末帧恢复标签可直接使用前十五帧历史。
def test_label_history_preserves_unlabelled_and_other_role_context():
    observations = [history_row(index, 1000 + 200 * index) for index in range(16)]
    observations[-1].navigation_expert_role = 'recovery-policy'
    report = label_history_coverage([observations[0], observations[-1]], observations)
    assert report['roles']['recovery-policy']['ready_labels'] == 1
    assert report['roles']['recovery-policy']['incomplete_labels'] == 0
    assert sum(row['ready_labels'] for row in report['roles'].values()) == 1


# 功能：
#   检查临界间隔、重置与来源切换的原因归属，不能把旧历史带入新连续段。
# 输入：
#   kind：制造的历史边界类型。
# 输出：
#   None：诊断给出的可用历史和部署规则一致。
@pytest.mark.parametrize('kind', ['source-gap', 'explicit-reset', 'source-switch'])
def test_label_history_attributes_incomplete_window_to_actual_reset(kind):
    observations = [history_row(index, 1000 + 250 * index) for index in range(16)]
    observations.append(history_row(16, 5001 if kind == 'source-gap' else 5000,
        stream='synthetic-z' if kind == 'source-switch' else 'synthetic', reset=kind == 'explicit-reset'))
    observations.append(history_row(17, 5200,
        stream='synthetic-z' if kind == 'source-switch' else 'synthetic'))
    report = label_history_coverage(observations[-2:], observations)
    role = report['roles'][NAVIGATION_EXPERT_ROLES[0]]
    assert role['ready_labels'] == 0 and role['incomplete_labels'] == 2
    assert role['reset_causes'] == {kind: 2}
    assert role['available_history_histogram'] == {'1': 1, '2': 1}
    assert role['examples'][0]['reset_gap_ms'] == (251 if kind == 'source-gap' else 250 if kind == 'explicit-reset' else None)


# 功能：
#   拒绝重复标签、重复观测、缺失来源和同摘要改时，不能用异常材料伪装窗口覆盖。
# 输入：
#   mutation：待验证的来源损坏类型。
# 输出：
#   None：每种损坏均抛出明确错误。
@pytest.mark.parametrize('mutation,error', [('label', 'REPEATED_LABEL'), ('source', 'REPEATED_SOURCE'),
    ('missing', 'LABEL_WITHOUT_SOURCE'), ('timestamp', 'LABEL_OBSERVATION_MISMATCH')])
def test_label_history_rejects_corrupt_lineage(mutation, error):
    observations = [history_row(0, 1000)]
    labels = [history_row(0, 1001 if mutation == 'timestamp' else 1000)]
    if mutation == 'label':
        labels *= 2
    elif mutation == 'source':
        observations *= 2
    elif mutation == 'missing':
        observations = []
    with pytest.raises(ValueError, match=error):
        label_history_coverage(labels, observations)


# 功能：
#   确保覆盖报告能通过产品严格回执发布边界，不只通过宽松 json.dumps。
# 输入：
#   无。
# 输出：
#   None：整数直方图键及 Counter 等非原生类型不会泄漏到发布接口。
def test_label_history_report_is_strict_evidence_json():
    from dronedream_agent_core.training.evidence_files import training_json_value

    rows = [history_row(0, 1000)]
    report = label_history_coverage(rows, rows)
    assert training_json_value(report) == report
