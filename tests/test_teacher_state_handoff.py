"""Independent native feedback may interleave with depth, never renew its lease."""

import pytest
from test_control_source_deadline import retained_features
from test_native_state_stream import packet

from dronedream_agent_core.native_state_stream import NativeStateBuffer
from dronedream_agent_core.runtime_scheduling import ReadyControlScheduler, teacher_state_handoff_ready


# 功能：
#   用真实原生状态编码器及来源期限构造交接条件，不用重复图片产生独立状态。
# 输入：
#   changes：本用例覆盖的条件字段。
# 输出：
#   values：传给生产调度检查的独立参数。
def inputs(**changes):
    buffer = NativeStateBuffer()
    for stamp in range(1000, 1171, 10):
        buffer.ingest(packet(stamp), now_unix_ms=stamp)
    values = dict(enabled=True, scheduler=ReadyControlScheduler(), state_buffer=buffer,
        features=retained_features(captured=1150), previous_state_ms=1150,
        depth_sequence=1, now_unix_ms=1180, perception_healthy=True,
        depth_processing_failed=False, fault_active=False)
    values.update(changes)
    return values


# 功能：
#   验证每个深度序列最多一次真实状态交接，消费后必须让出下一深度的处理机会。
# 输入：
#   无。
# 输出：
#   None：重复、回退或伪新状态触发断言失败。
def test_feedback_requires_new_state_and_yields_to_next_depth():
    values = inputs()
    assert teacher_state_handoff_ready(**values)
    values['scheduler'].consume(1)
    assert not teacher_state_handoff_ready(**values)
    values['depth_sequence'] = 2
    assert teacher_state_handoff_ready(**values)
    values['previous_state_ms'] = 1170
    assert not teacher_state_handoff_ready(**values)
    # 重发相同来源只有到达钟变化，不能获得新反馈资格。
    values['state_buffer'].ingest(packet(1170), now_unix_ms=1180)
    assert not teacher_state_handoff_ready(**values)


# 功能：
#   边界逐项拒绝未启用、故障、期限不足和没有新状态，不延长深度观测授权。
# 输入：
#   change：单一拒绝条件。
# 输出：
#   None：调度检查必须拒绝。
@pytest.mark.parametrize('change', [dict(enabled=False), dict(enabled=1), dict(features=None),
    dict(perception_healthy=False), dict(depth_processing_failed=True), dict(fault_active=True),
    dict(now_unix_ms=1211), dict(depth_sequence=0), dict(previous_state_ms=1170)])
def test_feedback_rejects_unsafe_or_replayed_conditions(change):
    assert not teacher_state_handoff_ready(**inputs(**change))


# 功能：
#   故障、未来到达、过期和损坏编码不能在元数据捷径中被误当作可用状态。
# 输入：
#   无。
# 输出：
#   None：原生缓冲的时序和有效性检查保持严格。
def test_native_feedback_metadata_keeps_original_validity():
    buffer = inputs()['state_buffer']
    assert buffer.newer_sample_available(1150, now_unix_ms=1180)
    assert not buffer.newer_sample_available(1150, now_unix_ms=1160)
    assert not buffer.newer_sample_available(1150, now_unix_ms=2000)
    buffer.invalidate('receiver-fault')
    assert not buffer.newer_sample_available(1150, now_unix_ms=1180)
    buffer.ingest(packet(1180), now_unix_ms=1180)
    buffer._latest.encoding.features[0] = float('nan')
    assert not buffer.newer_sample_available(1150, now_unix_ms=1180)


# 功能：
#   发现非法消费钟和上一来源钟时明确报错，不将布尔值或浮点当成来源时刻。
# 输入：
#   clock：非法时间；field：被篡改的参数名。
# 输出：
#   None：生产边界必须抛出 ValueError。
@pytest.mark.parametrize('clock', [-1, True, 1.5, None, 2**63])
@pytest.mark.parametrize('field', ['previous_source_unix_ms', 'now_unix_ms'])
def test_feedback_rejects_invalid_clocks(clock, field):
    values = dict(previous_source_unix_ms=1150, now_unix_ms=1180)
    values[field] = clock
    with pytest.raises(ValueError):
        inputs()['state_buffer'].newer_sample_available(**values)
