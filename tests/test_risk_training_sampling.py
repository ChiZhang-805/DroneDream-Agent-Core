"""Synthetic selection tests; these fixtures are never qualification evidence."""

from dataclasses import replace

import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingConfig
from dronedream_agent_core.training.action_risk_artifacts import ActionRiskDataset
from dronedream_agent_core.training.action_risk_training import train_action_risk_expert
from dronedream_agent_core.training.risk_training_sampling import (
    select_training_observations,
    training_selection_receipt,
)


# 功能：
#   生成四十组两动作合成记录，保留每组风险相反的标签以检查完整组采样。
# 输入：
#   无。
# 输出：
#   data：仅用于采样逻辑测试的数据对象。
def dataset():
    pair = _samples()[:2]
    records = tuple({'observation_sha256': f'{i:064x}'} for i in range(40) for _ in pair)
    data = ActionRiskDataset(tuple(pair*40), records, ('all-original-sources',),
                             frozenset({'1'*64}), '2'*64, '3'*64, {'original': True})
    return data


# 功能：
#   验证采样均匀覆盖首尾、保留组内所有动作、保留全来源隔离范围且不修改原始对象。
# 输入：
#   无。
# 输出：
#   None：遗漏一个动作、修改标签或丢掉原来源时失败。
def test_selection_keeps_whole_groups_and_source_scope():
    source = dataset()
    selected = select_training_observations(source, 20)
    ids = [f'{i*39//19:064x}' for i in range(20)]
    assert [row['observation_sha256'] for row in selected.records] == [
        i for i in ids for _ in range(2)]
    assert len(selected.samples) == 40
    assert [row.risk_target for row in selected.samples] == [0., 1.]*20
    assert selected.observations == source.observations
    assert selected.groups == source.groups
    assert selected.receipt == source.receipt
    assert len(source.samples) == 80 and source.training_observation_selection is None
    receipt = selected.training_observation_selection
    assert receipt['omitted_probe_count'] == 40
    assert receipt['source_observation_count'] == 40
    assert receipt['selected_observation_count'] == 20
    assert training_selection_receipt(selected) == receipt
    assert training_selection_receipt(source) is None


# 功能：
#   风险标签变化不能改变采样结果，不允许根据困难样本的成绩挑选训练子集。
# 输入：
#   无。
# 输出：
#   None：原始顺序相同则选择身份完全一致。
def test_selection_does_not_use_risk_labels():
    source = dataset()
    changed = replace(source, samples=tuple(s.model_copy(update={'risk_target': .7})
                                            for s in source.samples))
    assert select_training_observations(source, 20).training_observation_selection == (
        select_training_observations(changed, 20).training_observation_selection)
    full = select_training_observations(source, 48)
    assert full.samples == source.samples
    assert full.training_observation_selection['omitted_probe_count'] == 0


# 功能：
#   拒绝非法采样上限，防止零样本、布尔值或无界配置进入训练。
# 输入：
#   maximum：待检验的非法上限。
# 输出：
#   None：非法参数被明确拒绝。
@pytest.mark.parametrize('maximum', [True, 0, 19, 10001, 20., None])
def test_invalid_training_selection_limit(maximum):
    with pytest.raises(ValueError, match='LIMIT_INVALID'):
        select_training_observations(dataset(), maximum)


# 功能：
#   拒绝对已裁选视图再次采样，防止二次采样丢失全来源和未采用数量。
# 输入：
#   无。
# 输出：
#   None：重复选择被拒绝。
def test_selection_cannot_be_nested():
    selected = select_training_observations(dataset(), 20)
    with pytest.raises(ValueError, match='SOURCE_INVALID'):
        select_training_observations(selected, 20)


# 功能：
#   检出选样回执中的数量、身份和方法篡改，不能用错误说明美化实际覆盖。
# 输入：
#   field、value：篡改字段和值。
# 输出：
#   None：与训练视图不一致的回执被拒绝。
@pytest.mark.parametrize('field,value', [
    ('selected_probe_count', 80), ('source_probe_count', 0),
    ('omitted_probe_count', 0), ('selected_observation_count', 40),
    ('selected_observation_sha256', []), ('maximum_observations', True),
    ('outcomes_used_for_selection', True), ('dataset_receipt_sha256', '4'*64),
    ('extra', 'unrecognized'), ('source_order_sha256', 'bad'),
])
def test_selection_receipt_tampering_is_rejected(field, value):
    selected = select_training_observations(dataset(), 20)
    selected.training_observation_selection[field] = value
    with pytest.raises(ValueError, match='RISK_TRAINING_SELECTION_'):
        training_selection_receipt(selected)


# 功能：
#   验证训练入口拒绝裁选验证集，避免通过减少难例来改变独立验收结果。
# 输入：
#   tmp_path：独占输出目录。
# 输出：
#   None：验证集经过裁选时必须在创建输出前拒绝。
def test_validation_subsampling_is_rejected_before_artifact(tmp_path):
    source = dataset()
    validation = select_training_observations(source, 20)
    with pytest.raises(ValueError, match='VALIDATION_SUBSAMPLING_FORBIDDEN'):
        train_action_risk_expert([source], [validation], LocalPolicyTrainingConfig(),
                                tmp_path/'model')
    assert not (tmp_path/'model').exists()
