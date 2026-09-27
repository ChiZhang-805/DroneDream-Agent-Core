"""Independent observations survive skipped inference without future-frame leakage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from control_fixtures import complete_feature_snapshot
from test_adaptive_sensor_admission import coordinator, schedule

from dronedream_agent_core.causal_control import (
    CausalControlHistory, CONTROL_STATE_WIDTH, CONTROL_REALTIME_WIDTH)
from dronedream_agent_core.policy_observation_buffer import PolicyObservationBuffer
from dronedream_agent_core.temporal_evidence import TemporalEvidence


# 功能：构造具不同真实来源身份的数值夹具；输入：序号、时钟与来源；输出：冻结特征。
def batch(number, *, stamp=None, stream='vehicle-a', revision=0, reset=False):
    return SimpleNamespace(temporal_evidence=TemporalEvidence(stream_id=stream,
        sample_sha256=f'{number:064x}', observed_at_unix_ms=stamp if stamp is not None else number*50,
        history_slot_revision=revision, reset_history=reset),
        realtime_features_ready=True, state_features=(float(number),)*CONTROL_STATE_WIDTH,
        payload_features=(), realtime_features=(float(number),)*CONTROL_REALTIME_WIDTH,
        realtime_valid_mask=(1.,)*CONTROL_REALTIME_WIDTH)


# 功能：模拟模型隔很久才推理，但连续真实观测仍完整；输入：无；输出：16帧因果窗口。
def test_skipped_inference_preserves_real_contiguous_window_without_future_frames():
    queue, history = PolicyObservationBuffer(), CausalControlHistory()
    for i in range(1, 21):
        queue.stage(batch(i))
    for row in queue.take_through(batch(16).temporal_evidence):
        history.append(row.temporal_evidence, row.state_features,
                       row.realtime_features, row.realtime_valid_mask)
    assert history.ready
    values, mask = history.values()
    assert [row[0] for row in values] == list(range(1, 17))
    assert mask == [1.]*16
    assert [r.state_features[0] for r in queue.take_through(batch(20).temporal_evidence)] == [17,18,19,20]


# 功能：核对队列身份私有、同槽修订不增帧；输入：无；输出：旧调用看不到未来修订。
def test_revision_is_owned_and_never_leaks_backwards():
    queue = PolicyObservationBuffer()
    original = batch(1)
    queue.stage(original)
    original.temporal_evidence.observed_at_unix_ms = 99
    assert not queue.stage(batch(1))
    revised = batch(2, stamp=50, revision=1)
    queue.stage(revised)
    assert queue.take_through(batch(1).temporal_evidence) == ()
    rows = queue.take_through(revised.temporal_evidence)
    assert len(rows) == 1 and rows[0].state_features[0] == 2


# 功能：拒绝来源改时、倒序和无授权修订；输入：异常来源；输出：不覆盖已保存观测。
@pytest.mark.parametrize('bad', [batch(1,stamp=101),batch(2,stamp=50),batch(2,stamp=100)])
def test_invalid_source_order_cannot_replace_history(bad):
    queue = PolicyObservationBuffer()
    queue.stage(batch(1,stamp=100))
    with pytest.raises(ValueError, match='SOURCE_ORDER_INVALID'):
        queue.stage(bad)
    assert len(queue.take_through(batch(1,stamp=100).temporal_evidence)) == 1


# 功能：核对换来源及溢出容量不能跨任务拼历史；输入：无；输出：只剩当前流的有界记录。
def test_capacity_stream_switch_and_missing_source():
    queue = PolicyObservationBuffer(capacity=32)
    for i in range(1,100):
        queue.stage(batch(i))
    assert len(queue.take_through(batch(99).temporal_evidence)) == 32
    queue.stage(batch(1,stream='vehicle-b'))
    assert queue.take_through(batch(100).temporal_evidence) == ()
    assert len(queue.take_through(batch(1,stream='vehicle-b').temporal_evidence)) == 1
    assert queue.take_through(None) == ()


# 功能：飞行发布余量不足时仍可记真实观测，但绝不提交推理或刷新来源；输入：夹具；输出：断言。
def test_observation_without_dispatch_reserve_has_no_motion_authority(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    port.supports_observation_history = True
    port.observe_navigation_snapshot = Mock()
    try:
        result = schedule(instance, now_unix_ms=1211,
                          realtime_feature_snapshot=complete_feature_snapshot().model_dump())
        assert result.hold_reason == 'CONTROL_SOURCE_EXPIRED_DURING_PREPARATION'
        port.observe_navigation_snapshot.assert_called_once()
        executor.submit.assert_not_called()
        port.prime_multimodal.assert_not_called()
        stored = port.observe_navigation_snapshot.call_args.args[0]
        assert stored['realtime_feature_snapshot']['encodings'][0]['observed_at_unix_ms'] == 1000
    finally:
        instance.close()


# 功能：模型忙时仅排队观测，不产生并发调用或覆盖在途请求；输入：夹具；输出：一次推理提交。
def test_pending_model_observes_without_concurrent_inference(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    port.supports_observation_history = True
    port.observe_navigation_snapshot = Mock()
    try:
        assert schedule(instance) is None
        pending = instance._pending
        assert schedule(instance, now_unix_ms=1050) is None
        assert instance._pending is pending
        assert port.observe_navigation_snapshot.call_count == 2
        executor.submit.assert_called_once()
    finally:
        instance.close()


# 功能：RGB 短缺时保留当前数值历史，后续恢复图像才允许推理，不授权无图像驾驶。
# 输入：真实协调器与隔离模型执行器；输出：历史存在、动作调用不存在的断言。
def test_missing_current_rgb_preserves_observation_without_model_call(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    port.supports_observation_history = True
    port.observe_navigation_snapshot = Mock()
    try:
        assert schedule(instance, observation_only=True) is None
        port.observe_navigation_snapshot.assert_called_once()
        port.prime_multimodal.assert_not_called()
        executor.submit.assert_not_called()
        assert not instance.request_pending
        assert schedule(instance, now_unix_ms=1050) is None
        assert port.observe_navigation_snapshot.call_count == 2
        executor.submit.assert_called_once()
    finally:
        instance.close()


# 功能：缺失关键编码不能以仅观察模式越过验证；输入：错误融合快照；输出：拒绝入队。
def test_observation_only_still_requires_valid_source_evidence(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    port.supports_observation_history = True
    port.observe_navigation_snapshot = Mock()
    try:
        result = schedule(instance, observation_only=True, realtime_feature_snapshot=None)
        assert result.failure_reason_code == "CONTINUOUS_FEATURE_SCHEMA_INVALID"
        port.observe_navigation_snapshot.assert_not_called()
        executor.submit.assert_not_called()
    finally:
        instance.close()
