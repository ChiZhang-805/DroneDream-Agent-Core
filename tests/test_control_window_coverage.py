"""Synthetic overlap and axis diagnostics, never flight or training material."""

from dataclasses import replace

import pytest
from test_causal_policy import samples

from dronedream_agent_core.training.causal_policy import causal_examples
from dronedream_agent_core.training.control_coverage import control_window_coverage


# 功能：
#   从合成连续来源构造指定任务组的四帧窗口，供覆盖统计边界测试使用。
# 输入：
#   stream、group、start：互斥来源及任务组标识。
# 输出：
#   examples：八帧连续合成来源产生的五个重叠窗口。
def windows(stream='training', group='route-a', start=0):
    rows = samples(start=start, stream=stream)[:8]
    examples = causal_examples(rows, stream_groups={stream: group}, history_length=4)
    return examples


# 功能：
#   验证重复飞同一路线增加来源流但不增加独立任务组，重叠帧不重复算成不相交窗口。
# 输入：
#   无。
# 输出：
#   None：统计不符时断言失败。
def test_overlap_and_mission_counts_are_distinct():
    examples = windows() + windows('repeat', start=100)
    report = control_window_coverage(list(reversed(examples)), 4)
    assert report['complete_windows'] == 10
    assert report['greedy_source_disjoint_windows'] == 4
    assert report['source_stream_count'] == 2
    assert report['mission_group_count'] == 1
    assert report['independent_event_count'] is None
    assert not report['qualification_granted']
    assert report == control_window_coverage(examples, 4)


# 功能：
#   逐轴检查正负号、零附近阈值和移动模式筛选，不把保持模式的零轴当作运动多样性。
# 输入：
#   无。
# 输出：
#   None：边界、模式或极值不符时断言失败。
def test_axis_boundaries_and_nonmoving_exclusion():
    examples = windows()[:3]
    examples = [replace(example, sample=example.sample.model_copy(update={
        'target_action_index': mode, 'target_pilot_control': axes}))
        for example, mode, axes in zip(examples, [11, 11, 8],
            [[.01, -.01, .02, -.5], [.3, -.3, -.1, .2], [0., 0., 0., 0.]], strict=True)]
    report = control_window_coverage(examples, 4)
    assert report['moving_windows'] == 2
    assert report['moving_axes']['forward'] == dict(minimum=.01, maximum=.3, positive=1, negative=0, neutral=1)
    assert report['moving_axes']['right']['negative'] == 1
    assert report['moving_axes']['right']['neutral'] == 1
    assert report['moving_axes']['up']['negative'] == 1
    assert report['moving_axes']['up']['positive'] == 1
    assert report['moving_axes']['yaw']['minimum'] == -.5


# 功能：
#   拒绝重复当前来源、同流同刻冲突、同流换任务组以及被破坏的完整历史。
# 输入：
#   change：合成数据损坏方式。
# 输出：
#   None：错误必须抛出，不能作为普通缺口吞掉。
@pytest.mark.parametrize('change', ['duplicate', 'time', 'group', 'history'])
def test_invalid_window_evidence_is_rejected(change):
    examples = windows()
    if change == 'duplicate':
        examples.append(examples[0])
    elif change == 'time':
        sample = examples[1].sample
        altered = sample.temporal_evidence.model_copy(update={
            'observed_at_unix_ms': examples[0].sample.temporal_evidence.observed_at_unix_ms})
        examples[1] = replace(examples[1], sample=sample.model_copy(update={'temporal_evidence': altered}))
    elif change == 'group':
        examples[1] = replace(examples[1], group_id='other-route')
    else:
        examples[1] = replace(examples[1], source_sha256=())
    with pytest.raises(ValueError):
        control_window_coverage(examples, 4)


# 功能：
#   空专家保留明确零覆盖及未知极值；非法历史配置不能因没有样本而绕过校验。
# 输入：
#   无。
# 输出：
#   None：空统计或配置拒绝不符合约定时失败。
def test_empty_role_has_no_invented_coverage():
    report = control_window_coverage([])
    assert report['complete_windows'] == report['mission_group_count'] == 0
    assert report['greedy_source_disjoint_windows'] == 0
    assert report['moving_axes']['forward']['minimum'] is None
    with pytest.raises(ValueError):
        control_window_coverage([], True)
