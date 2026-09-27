"""Prepared tensors save work, never cache authorization or mutable caller input."""

import copy
from unittest.mock import patch

import pytest
from control_fixtures import complete_feature_snapshot
from test_local_policy_packages import _snapshot, _write_package, _PilotControlBackend
from test_local_control_harness_integrity import call
import dronedream_agent_core.local_policy_port as policy
from dronedream_agent_core.hashing import sha256_json


# 功能：生成内容绑定的连续输入；输入：用于区分内容的目标距离；输出：独立测试快照。
def snapshot(distance=1.):
    value = _snapshot()
    value['authorized_candidate_paths'] = []
    value['goal_distance_m'] = distance
    value['realtime_feature_snapshot'] = complete_feature_snapshot().model_dump(mode='json')
    value['snapshot_sha256'] = sha256_json({k:v for k,v in value.items() if k != 'snapshot_sha256'})
    return value


# 功能：来源超期仍拒绝控制，同时保留数值成本证据；不把观测期限改写为现在。
def test_expired_budget_keeps_numeric_diagnostics(port):
    with patch.object(policy.time, 'monotonic', return_value=1.1):
        with pytest.raises(TimeoutError) as caught:
            port._check_call_budget(1., control_deadline_monotonic=1.08)
    metrics = caught.value.diagnostic_metrics
    assert metrics['total-latency-ms'] == pytest.approx(100.)
    assert metrics['source-budget-at-entry-ms'] == pytest.approx(80.)
    assert metrics['inference-completed'] == 0


# 功能：建立不连接飞控的真实端口；输入：隔离目录；输出：连续模型端口。
@pytest.fixture
def port(tmp_path):
    instance = policy.LocalPolicyPort(_write_package(tmp_path / 'package',
        package_id='test.prepared', continuous_control=True), backend=_PilotControlBackend())
    yield instance
    instance.close()


# 功能：证明已编译输入仅移交一次，第二次仍重新编译；输入：端口；输出：调用计数与决策断言。
def test_single_use_owned_tensor_handoff(port):
    value = snapshot()
    with patch.object(policy, 'compile_local_policy_features', wraps=policy.compile_local_policy_features) as compile:
        port.observe_navigation_snapshot(value)
        assert compile.call_count == 1
        first = call(port, value)
        assert compile.call_count == 1
        assert call(port, value).artifact == first.artifact
        assert compile.call_count == 2
    assert first.artifact.action == 'pilot-control'


# 功能：拒绝内容被改但摘要沿用的输入，允许独立旧副本；输入：端口；输出：缓存不能掩盖篡改。
def test_cache_does_not_mask_input_mutation(port):
    value = snapshot()
    owned = copy.deepcopy(value)
    port.observe_navigation_snapshot(value)
    value['realtime_feature_snapshot']['fused_features'][0] = 999.
    with pytest.raises(ValueError, match='SNAPSHOT_HASH_INVALID'):
        call(port, value)
    assert call(port, owned).artifact.action == 'pilot-control'


# 功能：缓存只保留四份不同内容，不能用相同来源覆盖不同任务输入；输入：端口；输出：最旧条目被淘汰。
def test_cache_is_bounded_and_content_specific(port):
    for i in range(8):
        port.observe_navigation_snapshot(snapshot(float(i)))
    assert len(port._prepared_inputs) == 4
    assert snapshot(0.)['snapshot_sha256'] not in port._prepared_inputs
    with patch.object(policy, 'compile_local_policy_features', wraps=policy.compile_local_policy_features) as compile:
        call(port, snapshot(0.))
        assert compile.call_count == 1
        call(port, snapshot(7.))
        assert compile.call_count == 1


# 功能：缓存命中也必须拒绝已经耗尽的实际调用期限；输入：端口；输出：不能从缓存续期。
def test_cached_input_cannot_extend_deadline(port):
    value = snapshot()
    port.observe_navigation_snapshot(value)
    with pytest.raises(ValueError, match='CONTROL_DEADLINE_INVALID'):
        port.call(role='local_navigation_advisor', output_type=policy.TextNavigationDecision,
            instructions='', input_artifact={'text_navigation_snapshot': value},
            control_deadline_monotonic=policy.time.monotonic()-1)
