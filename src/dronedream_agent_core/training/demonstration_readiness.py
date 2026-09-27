"""Pre-training causal coverage checks; counts never confer flight qualification."""

from collections import Counter

from ..causal_control import CONTROL_HISTORY_LENGTH
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..temporal_evidence import ObservationHistory
from .causal_policy import causal_examples


# 功能：
#   逐标签追溯部署端历史为何未就绪，保留无标签及其他专家观测，不把角色切换当成重置。
# 输入：
#   samples：已校验的执行标签。
#   observations：完整独立观测流。
#   history_length：部署端使用的历史长度。
# 输出：
#   report：按专家归类的就绪数、未就绪原因、可用历史长度及最多八个断点示例。
def label_history_coverage(samples, observations, history_length=CONTROL_HISTORY_LENGTH):
    if type(history_length) is not int or not 4 <= history_length <= 32:
        raise ValueError('CAUSAL_CONTROL_HISTORY_LENGTH_INVALID')
    labelled = {}
    for sample in samples:
        evidence = sample.temporal_evidence
        identity = evidence.stream_id, evidence.sample_sha256
        if identity in labelled:
            raise ValueError('DEMONSTRATION_COVERAGE_REPEATED_LABEL')
        if sample.navigation_expert_role not in NAVIGATION_EXPERT_ROLES:
            raise ValueError('CAUSAL_TRAINING_MANOEUVRE_ROLE_INVALID')
        labelled[identity] = sample
    roles = {role: {'ready_labels': 0, 'incomplete_labels': 0, 'reset_causes': Counter(),
        'available_history_histogram': Counter(), 'examples': []} for role in NAVIGATION_EXPERT_ROLES}
    history, seen = ObservationHistory(history_length), set()
    reset_reason, reset_gap, reset_time = 'initial-warmup', None, None
    for observation in sorted(observations, key=lambda row: (
            row.temporal_evidence.stream_id, row.temporal_evidence.observed_at_unix_ms)):
        evidence = observation.temporal_evidence
        identity = evidence.stream_id, evidence.sample_sha256
        if identity in seen:
            raise ValueError('DEMONSTRATION_COVERAGE_REPEATED_SOURCE')
        seen.add(identity)
        previous = history.latest
        same_stream = previous is not None and previous.stream_id == evidence.stream_id
        gap = evidence.observed_at_unix_ms - previous.observed_at_unix_ms if same_stream else None
        # 重置条件仅用于解释；是否接受及是否完整仍交给部署共用的 ObservationHistory。
        if previous is None or not same_stream or evidence.reset_history or gap > history.maximum_gap_ms:
            reset_reason = ('explicit-reset' if evidence.reset_history else 'source-gap' if same_stream
                            else 'source-switch' if previous is not None else 'initial-warmup')
            reset_gap, reset_time = gap, evidence.observed_at_unix_ms
        history.append(evidence, (), ())
        label = labelled.get(identity)
        if label is None:
            continue
        if label.temporal_evidence != evidence:
            raise ValueError('CAUSAL_TRAINING_LABEL_OBSERVATION_MISMATCH')
        role = roles[label.navigation_expert_role]
        available = len(history.rows)
        role['available_history_histogram'][available] += 1
        role['ready_labels' if history.ready else 'incomplete_labels'] += 1
        if not history.ready:
            role['reset_causes'][reset_reason] += 1
            if len(role['examples']) < 8:
                role['examples'].append({'stream_id': evidence.stream_id,
                    'sample_sha256': evidence.sample_sha256, 'observed_at_unix_ms': evidence.observed_at_unix_ms,
                    'available_history': available, 'reset_reason': reset_reason,
                    'reset_at_unix_ms': reset_time, 'reset_gap_ms': reset_gap})
    if set(labelled) - seen:
        raise ValueError('CAUSAL_TRAINING_LABEL_WITHOUT_SOURCE_OBSERVATION')
    # 发布回执要求原生字典和字符串键；不能把 Counter 或整数键交给严格 JSON 边界。
    for role in roles.values():
        role['reset_causes'] = dict(role['reset_causes'])
        role['available_history_histogram'] = {
            str(length): count for length, count in sorted(role['available_history_histogram'].items())}
    report = {'history_length': history_length, 'maximum_gap_ms': history.maximum_gap_ms,
              'roles': roles, 'qualification_granted': False}
    return report


# 功能：
#   使用部署端同一历史接纳规则统计采样断档和连续长度，不靠插值、重放或放宽间隔补足窗口。
# 输入：
#   observations：已验证语料的真实观测历史。
#   history_length：部署和训练共用的窗口长度。
# 输出：
#   report：来源间隔、显式重置、断档和可形成完整历史的观测数量。
def temporal_coverage(observations, history_length=CONTROL_HISTORY_LENGTH):
    if type(history_length) is not int or not 4 <= history_length <= 32:
        raise ValueError('CAUSAL_CONTROL_HISTORY_LENGTH_INVALID')
    history = ObservationHistory(history_length)
    gaps, streams, identities = [], set(), set()
    gap_resets, explicit_resets, ready_count, longest, streak = 0, 0, 0, 0, 0
    for observation in sorted(observations, key=lambda item: (
            item.temporal_evidence.stream_id, item.temporal_evidence.observed_at_unix_ms)):
        evidence = observation.temporal_evidence
        identity = evidence.stream_id, evidence.sample_sha256
        if identity in identities:
            raise ValueError('DEMONSTRATION_COVERAGE_REPEATED_SOURCE')
        identities.add(identity)
        previous = history.latest
        if previous is not None and previous.stream_id == evidence.stream_id:
            gap = evidence.observed_at_unix_ms - previous.observed_at_unix_ms
            gaps.append(gap)
            gap_resets += int(gap > history.maximum_gap_ms)
        explicit_resets += int(evidence.reset_history)
        history.append(evidence, (), ())
        streak = 1 if len(history.rows) == 1 else streak + 1
        longest = max(longest, streak)
        ready_count += int(history.ready)
        streams.add(evidence.stream_id)
    gaps.sort()
    report = {'observation_count': len(identities), 'stream_count': len(streams),
        'maximum_gap_ms': history.maximum_gap_ms, 'gap_reset_count': gap_resets,
        'explicit_reset_count': explicit_resets, 'longest_contiguous_observations': longest,
        'complete_history_observations': ready_count,
        'source_gap_median_ms': gaps[len(gaps) // 2] if gaps else None,
        'source_gap_p95_ms': gaps[max(0, (95 * len(gaps) + 99) // 100 - 1)] if gaps else None,
        'source_gap_max_ms': gaps[-1] if gaps else None,
        'qualification_granted': False}
    return report


# 功能：
#   逐专家用实际训练窗口构造器核对历史覆盖，不用照片数量替代连续控制样本数量。
#   仅将合法但缺少完整窗口的情况记为零；损坏、泄漏或契约错误继续抛出。
# 输入：
#   corpus：已通过执行证据校验的动作标签、观测历史及任务分区。
#   history_length：与后续控制模型一致的历史帧数。
# 输出：
#   report：每个专家的动作与有效窗口数量；不代表模型效果或飞行资格。
def demonstration_readiness(corpus, history_length=CONTROL_HISTORY_LENGTH):
    labels = Counter(sample.navigation_expert_role for sample in corpus.samples)
    roles = {}
    for role in NAVIGATION_EXPERT_ROLES:
        try:
            windows = causal_examples(corpus.samples, stream_groups=corpus.stream_groups,
                history_length=history_length, navigation_role=role,
                history_observations=corpus.observations)
            count = len(windows)
        except ValueError as exc:
            if str(exc) != 'CAUSAL_TRAINING_HAS_NO_COMPLETE_WINDOWS':
                raise
            count = 0
        roles[role] = {'executed_labels': labels[role], 'complete_windows': count}
    history_report = label_history_coverage(corpus.samples, corpus.observations, history_length)
    if any(history_report['roles'][role]['ready_labels'] != data['complete_windows']
           for role, data in roles.items()):
        raise ValueError('DEMONSTRATION_COVERAGE_HISTORY_PARITY_FAILED')
    report = {'history_length': history_length, 'roles': roles,
        'label_history_coverage': history_report,
        'temporal_coverage': temporal_coverage(corpus.observations, history_length),
        'all_roles_have_windows': all(row['complete_windows'] > 0 for row in roles.values()),
        'qualified_for_flight': False}
    return report
