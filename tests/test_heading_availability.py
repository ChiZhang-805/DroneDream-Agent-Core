"""Availability routing and simulation observation access; no flight side effects."""

from unittest.mock import Mock

import pytest
import torch

from dronedream_agent_core.training.heading_availability import AvailabilityHeadingPolicy
from dronedream_agent_core.training.heading_stability import StableHeadingPolicy
from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment
from types import SimpleNamespace
from control_fixtures import complete_feature_snapshot
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.live_heading_context import LiveHeadingContext
from dronedream_agent_core.training.runtime_evidence import ControlReceiptMonitor
from collections import OrderedDict
from threading import Lock


# 功能：
#   验证缺失回执或陀螺仪时不再调用时序结果作为控制输出，临界年龄下连续退回单帧分支。
# 输入：
#   age_weighted：是否使用年龄权重。
# 输出：
#   None：组合输出与约定的专家选择或权重一致且有界。
@pytest.mark.parametrize('age_weighted', [False, True])
def test_missing_context_uses_current_expert(age_weighted):
    temporal, current = StableHeadingPolicy(use_context=True), StableHeadingPolicy(use_context=False)
    model = AvailabilityHeadingPolicy(temporal, current, age_weighted=age_weighted)
    x = torch.zeros(4, 23)
    x[:, 9] = 1.
    x[1:, 12] = 1.
    x[2, 9] = 0.
    x[3, 11] = 1.
    result = model(x)
    assert torch.equal(result[0], current(x)[0])
    assert torch.equal(result[2], current(x)[2])
    assert torch.allclose(result[1], temporal(x)[1])
    if age_weighted:
        assert torch.equal(result[3], current(x)[3])
    assert (result.abs() <= 1).all()


# 功能：
#   验证公开仿真快照返回副本，关闭、空请求或身份不匹配时拒绝调用。
# 输入：
#   damage：正常或损坏的环境状态。
# 输出：
#   None：外部策略无法修改内部请求或读取错配观测。
@pytest.mark.parametrize('damage', ['none', 'closed', 'pending', 'identity'])
def test_environment_context_is_detached(damage):
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._pending = {'snapshot': {'snapshot_sha256': 'a'*64, 'control_reference_observed_at_unix_ms': 1000, 'nested': {'x': 1}}}
    env._observation = SimpleNamespace(sample=SimpleNamespace(source_snapshot_sha256='a'*64))
    env._quiesced, env._closed = False, damage == 'closed'
    env._receipt_monitor = Mock()
    env._receipt_monitor.latest_before.return_value = None
    if damage == 'pending':
        env._pending = None
    elif damage == 'identity':
        env._observation.sample.source_snapshot_sha256 = 'b'*64
    if damage == 'none':
        snapshot, application = env.current_control_context()
        snapshot['nested']['x'] = 2
        assert env._pending['snapshot']['nested']['x'] == 1 and application is None
        env._receipt_monitor.latest_before.assert_called_once_with(1000)
    else:
        with pytest.raises((ValueError, RuntimeError), match='PX4_TRAINING_CONTEXT'):
            env.current_control_context()


# 功能：
#   创建时间一致的合成源快照，供实时编码的重置和失效边界测试。
# 输入：
#   stamp：Unix 毫秒时间；goal_x：目标位置。
# 输出：
#   snapshot：具有有效内容摘要和完整几何的测试快照。
def live_snapshot(stamp=1000, goal_x=3.):
    snapshot = dict(current_position_m=dict(x=0., y=0., z=1.), goal_position_m=dict(x=goal_x, y=1., z=1.),
        dynamic_obstacles=[], local_radius_m=8., control_reference_observed_at_unix_ms=stamp,
        realtime_feature_snapshot=complete_feature_snapshot(stamp).model_dump(mode='json'))
    snapshot['snapshot_sha256'] = sha256_json(snapshot)
    return snapshot


# 功能：
#   核验当前原生几何按真实时间形成历史，目标或流变化重置，过期及重复源拒绝。
# 输入：
#   change：第二帧的来源或目标变化。
# 输出：
#   None：历史有效位符合真实连续性，不续期、不读取真值。
@pytest.mark.parametrize('change', ['continuous', 'goal', 'stream', 'stale', 'duplicate'])
def test_live_heading_memory_obeys_source_boundaries(change):
    state = LiveHeadingContext()
    first = state.encode(live_snapshot(), 'stream-a', None, now_unix_ms=1000)
    assert first[12] == 0. and first[22] == 0.
    source = 1000 if change == 'duplicate' else 1100
    snapshot = live_snapshot(source, 4. if change == 'goal' else 3.)
    now = 1500 if change == 'stale' else source
    if change in ('stale', 'duplicate'):
        with pytest.raises(ValueError):
            state.encode(snapshot, 'stream-a', None, now_unix_ms=now)
    else:
        result = state.encode(snapshot, 'stream-b' if change == 'stream' else 'stream-a', None, now_unix_ms=now)
        assert result[22] == (1. if change == 'continuous' else 0.)


# 功能：
#   同坐标的新任务阶段不能继承旧目标的几何差分，损坏的阶段标识也不能作为合法身份。
# 输入：
#   next_goal_id：下一观测的语义目标身份。
# 输出：
#   None：历史有效位只在目标代次相同时保持有效。
@pytest.mark.parametrize('next_goal_id', ['pickup', 'return', None, '', 7, True])
def test_same_coordinate_goal_revision_resets_geometry_history(next_goal_id):
    state = LiveHeadingContext()
    first = live_snapshot()
    first['strategic_context'] = {'task': {'navigation_goal_id': 'pickup'}}
    first['snapshot_sha256'] = sha256_json({k: v for k, v in first.items() if k != 'snapshot_sha256'})
    state.encode(first, 'stream-a', None, now_unix_ms=1000)
    second = live_snapshot(1100)
    second['strategic_context'] = {'task': {'navigation_goal_id': next_goal_id}}
    second['snapshot_sha256'] = sha256_json({k: v for k, v in second.items() if k != 'snapshot_sha256'})
    if next_goal_id in ('pickup', 'return', None):
        result = state.encode(second, 'stream-a', None, now_unix_ms=1100)
        assert result[22] == float(next_goal_id == 'pickup')
    else:
        with pytest.raises(ValueError, match='LIVE_HEADING_GOAL_IDENTITY_INVALID'):
            state.encode(second, 'stream-a', None, now_unix_ms=1100)


# 功能：
#   验证当前输入只接收已匹配命令的严格过去回执；较新的不完整回执不能被更旧记录掩盖。
# 输入：
#   无。
# 输出：
#   None：时间、绑定与副本隔离均按实时输入要求生效。
def test_latest_receipt_does_not_reach_back_past_incomplete_record():
    from test_control_execution_evidence import evidence
    application, data = evidence()
    monitor = object.__new__(ControlReceiptMonitor)
    monitor._lock, monitor._error = Lock(), None
    monitor._known_commands, monitor._known_applications = OrderedDict(), OrderedDict()
    monitor._last_sequence, monitor._last_time = 0, -1
    monitor._ingest(data['command_records'], [application])
    stamp = application['accepted_at_unix_ms']
    assert monitor.latest_before(stamp) is None
    result = monitor.latest_before(stamp+1)
    assert result.command_sha256 == application['command_sha256']
    result.control_source = 'changed-locally'
    assert monitor.latest_before(stamp+1).control_source != 'changed-locally'
    following = dict(application, sequence=2, accepted_at_unix_ms=stamp+2, command_sha256='f'*64)
    monitor._ingest([], [following])
    assert monitor.latest_before(stamp+3) is None
    assert monitor.latest_before(stamp+2) is not None
    monitor._error = RuntimeError('injected')
    with pytest.raises(RuntimeError, match='MONITOR_FAILED'):
        monitor.latest_before(stamp+3)
