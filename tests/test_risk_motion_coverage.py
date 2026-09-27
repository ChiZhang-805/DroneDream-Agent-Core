"""Direction coverage diagnostics; synthetic examples are not flight evidence."""
from dataclasses import replace

import pytest
from test_action_risk_training_boundaries import metric_dataset

from dronedream_agent_core.training.action_risk_training import risk_motion_coverage


# 功能：验证水平、升降、斜向和原地转向各归入唯一类别，且同一源的重复探针不虚增观测数。
# 输入：无；四类动作的合成样本。
# 输出：None；断言类别计数、风险边界和原数据未修改。
def test_mutually_exclusive_direction_coverage():
    dataset = metric_dataset()
    actions = ([0, 0, 0, 1], [.01, 0, 0, 0], [.01, 0, -.01, 0], [0, .01, .01, 0])
    for index, sample in enumerate(dataset.samples):
        sample.risk_proposed_control = list(actions[(index // 2) % 4])
        sample.risk_target = .5 if index % 2 else .49
    original = [sample.model_dump() for sample in dataset.samples]
    result = risk_motion_coverage([dataset])
    for row in result.values():
        assert row == dict(sample_count=8, safe_sample_count=4, unsafe_sample_count=4,
                           independent_observation_count=4, safe_observation_count=4,
                           unsafe_observation_count=4)
    assert [sample.model_dump() for sample in dataset.samples] == original
    repeated = replace(dataset, samples=dataset.samples * 2, records=dataset.records * 2)
    for row in risk_motion_coverage([repeated]).values():
        assert row['sample_count'] == 16
        assert row['independent_observation_count'] == 4


# 功能：覆盖缺失不能靠省略该类或把近零升降偷偷算成平飞隐藏。
# 输入：无；仅含水平动作的合成数据以及微小上升提案。
# 输出：None；空类保留零计数，微小上升仍可见。
def test_empty_directions_and_small_vertical_motion_are_visible():
    dataset = metric_dataset()
    result = risk_motion_coverage([dataset])
    assert result['descending']['sample_count'] == 0
    assert result['ascending']['sample_count'] == 0
    assert result['level-translation']['sample_count'] == 32
    dataset.samples[0].risk_proposed_control[2] = 1e-9
    assert risk_motion_coverage([dataset])['ascending']['sample_count'] == 1


# 功能：拒绝缺失、损坏或越界提案，不能将坏输入默认计入悬停类别。
# 输入：action：非法合成提案。
# 输出：None；全部输入在统计前被拒绝。
@pytest.mark.parametrize('action', [[], [0, 0, 0], [True, 0, 0, 0],
                                   [float('nan'), 0, 0, 0], [0, 0, 1.01, 0]])
def test_malformed_actions_are_not_coverage(action):
    dataset = metric_dataset()
    dataset.samples[0].__dict__['risk_proposed_control'] = action
    with pytest.raises(ValueError, match='MOTION_CONTROL_INVALID'):
        risk_motion_coverage([dataset])
