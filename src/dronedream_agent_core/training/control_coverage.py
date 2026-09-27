"""Causal demonstration diversity diagnostics; never grant training or flight authority."""

from collections import Counter

from ..causal_control import CONTROL_HISTORY_LENGTH
from .causal_policy import validate_learning_examples
from .flight_environment import MODES


# 功能：
#   1. 核对完整窗口后统计任务、来源流、控制模式和移动四轴覆盖，不把重叠窗口称为独立事件。
#   2. 按来源时间贪心选取不共享任何历史帧的窗口，仅报告数量，不删除或重采样训练材料。
# 输入：
#   examples：同一专家的因果窗口；空列表表示该专家没有完整窗口。
#   history_length：与部署一致的历史长度。
# 输出：
#   report：来源、非重叠窗口和归一化四轴统计。
def control_window_coverage(examples, history_length=CONTROL_HISTORY_LENGTH):
    if type(history_length) is not int or not 4 <= history_length <= 32:
        raise ValueError('CAUSAL_CONTROL_HISTORY_LENGTH_INVALID')
    if examples:
        validate_learning_examples(examples, history_length)
    ordered = sorted(examples, key=lambda item: (
        item.sample.temporal_evidence.stream_id,
        item.sample.temporal_evidence.observed_at_unix_ms,
        item.sample.temporal_evidence.sample_sha256))
    identities, times, used, groups, streams = set(), set(), set(), set(), {}
    modes = Counter()
    nonoverlapping = 0
    moving = []
    for example in ordered:
        sample, group = example.sample, example.group_id
        evidence = sample.temporal_evidence
        identity = evidence.sample_sha256
        time_key = evidence.stream_id, evidence.observed_at_unix_ms
        if identity in identities or time_key in times:
            raise ValueError('CONTROL_COVERAGE_DUPLICATE_WINDOW')
        identities.add(identity)
        times.add(time_key)
        previous = streams.setdefault(evidence.stream_id, group)
        if previous != group:
            raise ValueError('CONTROL_COVERAGE_STREAM_GROUP_CHANGED')
        groups.add(group)
        sources = set(example.source_sha256)
        if not sources & used:
            used.update(sources)
            nonoverlapping += 1
        modes[MODES[sample.target_action_index - 8]] += 1
        # 与留出评价一致：保持／请求新扫描／终止模式不充作移动轴监督覆盖。
        if sample.target_action_index == 11:
            moving.append(sample.target_pilot_control)
    axes = {}
    for index, name in enumerate(('forward', 'right', 'up', 'yaw')):
        values = [row[index] for row in moving]
        axes[name] = dict(minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            positive=sum(value > .01 for value in values),
            negative=sum(value < -.01 for value in values),
            neutral=sum(abs(value) <= .01 for value in values))
    report = dict(complete_windows=len(ordered), mission_group_count=len(groups),
        source_stream_count=len(streams), greedy_source_disjoint_windows=nonoverlapping,
        history_length=history_length, moving_windows=len(moving),
        mode_windows={name: modes[name] for name in MODES},
        normalized_axis_neutral_threshold=.01, moving_axes=axes,
        independent_event_count=None, qualification_granted=False)
    return report
