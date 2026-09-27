from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_control_source_deadline import pending_coordinator, retained_features

from dronedream_agent_core.runtime_scheduling import pending_dispatch_budget_ready
from dronedream_agent_core.runtime_scheduling import model_handoff_budget_ready


# 功能：
#   无模型、无请求或已过期请求不解析无用张量，也不会获得调度资格。
# 输入：
#   无。
# 输出：
#   无。
def test_absent_or_expired_request_does_not_revalidate_unused_features():
    # 功能：
    #   把无意义的深层快照校验转成明确失败，以防高频空闲循环重复消耗处理预算。
    # 输入：
    #   now_unix_ms：测试时钟。
    # 输出：
    #   无。
    def forbidden(*, now_unix_ms):
        raise AssertionError("idle loop must not rebuild a feature snapshot")

    features = SimpleNamespace(control_deadline_unix_ms=forbidden)
    assert not pending_dispatch_budget_ready(None, features, now_unix_ms=1200)
    coordinator = pending_coordinator(age=.2)
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1200)
    coordinator._pending = None
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1000)


# 功能：
#   实际请求仍须逐项检查观测期限，预算边界及输入篡改不能因调度优化而通过。
# 输入：
#   now：预算边界附近的毫秒时刻；expected：原公式的预期结果。
# 输出：
#   无。
@pytest.mark.parametrize("now,expected", [(1210, True), (1211, False), (1250, False)])
def test_pending_request_retains_both_original_deadlines(now, expected):
    features = retained_features(captured=1100)
    coordinator = pending_coordinator()
    assert pending_dispatch_budget_ready(coordinator, features, now_unix_ms=now) is expected
    features.encodings[0].features[0] = float("nan")
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=now)


# 功能：
#   无任务时也拒绝非法消费钟，不让快速路径静默接受错误类型。
# 输入：
#   clock：非法时钟。
# 输出：
#   无。
@pytest.mark.parametrize("clock", [-1, True, 1.5, None, 2**63])
def test_idle_dispatch_rejects_invalid_clock(clock):
    with pytest.raises(ValueError, match="CONTROL_DEADLINE_CLOCK_INVALID"):
        pending_dispatch_budget_ready(None, None, now_unix_ms=clock)


# 功能：
#   验证新原生状态可以解除旧状态造成的调度阻塞，但不能延长几何或原模型请求期限。
# 输入：
#   无。
# 输出：
#   None：只更新真实状态来源，输入原对象及几何时钟保持不变。
def test_current_native_state_can_enable_handoff_without_renewing_geometry():
    from control_fixtures import complete_feature_snapshot
    from dronedream_agent_core.realtime_feature_encoders import fuse_realtime_features
    base = complete_feature_snapshot(1000)
    older = base.encodings[-1].model_copy(update={'observed_at_unix_ms': 870, 'encoded_at_unix_ms': 1000})
    features = fuse_realtime_features([*base.encodings[:-1], older], captured_at_unix_ms=1000)
    before = features.model_dump()
    newer = base.encodings[-1].model_copy(update={'observed_at_unix_ms': 1100, 'encoded_at_unix_ms': 1100})
    buffer = SimpleNamespace(latest=Mock(return_value=SimpleNamespace(encoding=newer)))
    coordinator = pending_coordinator()
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1100)
    assert pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1100, state_buffer=buffer)
    assert features.model_dump() == before
    # 请求仍有余量时，独立的旧几何期限仍必须挡住交接。
    coordinator._pending.maximum_decision_age_seconds = .5
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1220, state_buffer=buffer)
    buffer.latest.side_effect = ValueError('NATIVE_STATE_STREAM_EXPIRED')
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1100, state_buffer=buffer)
    buffer.latest.side_effect = None
    buffer.latest.return_value = SimpleNamespace(encoding=newer.model_copy(update={'source_ids': ['other-vehicle']}))
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1100, state_buffer=buffer)
    buffer.latest.reset_mock()
    coordinator._pending = None
    assert not pending_dispatch_budget_ready(coordinator, features, now_unix_ms=1100, state_buffer=buffer)
    buffer.latest.assert_not_called()


# 功能：
#   等待中的调度只读原期限，完成时必须恢复完整张量检查，不能因优化接受损坏输入。
# 输入：
#   无。
# 输出：
#   None：过期请求、无任务和篡改张量均不产生可消费的结果。
def test_waiting_uses_original_lease_but_completed_result_revalidates_features():
    coordinator = pending_coordinator()
    features = retained_features(captured=1100)
    validate = Mock(wraps=features.control_deadline_unix_ms)
    proxy = SimpleNamespace(control_deadline_unix_ms=validate)
    for now in (1100, 1150, 1210):
        assert model_handoff_budget_ready(coordinator, proxy, handoff_state=(True, False), now_unix_ms=now)
    validate.assert_not_called()
    assert not model_handoff_budget_ready(coordinator, proxy, handoff_state=(True, False), now_unix_ms=1211)
    assert not model_handoff_budget_ready(coordinator, proxy, handoff_state=(False, False), now_unix_ms=1100)
    assert model_handoff_budget_ready(coordinator, proxy, handoff_state=(False, True), now_unix_ms=1100)
    validate.assert_called_once_with(now_unix_ms=1100)
    features.encodings[0].features[0] = float('nan')
    assert not model_handoff_budget_ready(coordinator, proxy, handoff_state=(False, True), now_unix_ms=1100)


# 功能：
#   调度状态要求一个精确的二元布尔快照，非法组合不能进入快捷路径。
# 输入：
#   state：非法状态表示。
# 输出：
#   None：异常在任何期限查询之前抛出。
@pytest.mark.parametrize('state', [None, (True, True), (1, False), [True, False], (True,)])
def test_handoff_rejects_inconsistent_state(state):
    with pytest.raises(ValueError, match='CONTROL_HANDOFF_STATE_INVALID'):
        model_handoff_budget_ready(None, None, handoff_state=state, now_unix_ms=1000)
