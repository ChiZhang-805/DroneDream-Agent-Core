"""Synthetic deadline faults, not physical-flight qualification evidence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from clock_fixtures import isolate_time
from test_px4_training_lifecycle import environment
from test_stream_collection import StreamLifecycle, collect, stream_fixture

from dronedream_agent_core.training import px4_environment as native
from dronedream_agent_core.training import stream_collection as collection
from dronedream_agent_core.training.flight_environment import ControlPreparationExpired, PilotAction
from dronedream_agent_core.training.observations import TrainingObservationError


# 功能：
#   验证只有流式模式使用实际准备耗时，其他模式保留原有预测余量且不直接发动作。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定时钟的测试工具。
#   mode：当前采集模式。
#   budget：该模式应使用的输入余量毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mode,budget", [(None, 90), ("reward-step", 90), ("stream-imitation", 40)]
)
def test_only_stream_preparation_uses_actual_instead_of_predicted_cost(
    tmp_path, monkeypatch, mode, budget
):
    env = environment(tmp_path)
    env._collection_mode = mode
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock(), reply=Mock())
    rejected = {"valid_until_unix_ms": 1000 + budget - 1}
    accepted = {"valid_until_unix_ms": 1000 + budget}
    env._receive_source_request = Mock(side_effect=[rejected, accepted])
    monkeypatch.setattr(native, "time", SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 1.0))
    assert env._next_request(deadline=2.0) is accepted
    assert env._input_admission_budget_ms() == budget
    assert env._input_rejection_counts == {"insufficient-input-budget": 1}
    env._exchange.reply.assert_not_called()  # Input admission is not authority.
    assert accepted["valid_until_unix_ms"] == 1000 + budget


# 功能：
#   验证准备耗尽派发余量后，不发送提案、不保存捕获、不推进序号或续租源请求。
# 输入：
#   tmp_path：合成执行证据目录。
#   monkeypatch：把当前时间推至期限边缘的测试工具。
# 输出：
#   None：不返回业务数据。
def test_stream_preparation_expiry_has_no_reply_capture_or_sequence_advance(tmp_path, monkeypatch):
    _, _, capture, _ = stream_fixture(tmp_path)
    env = environment(tmp_path)
    env._quiesced, env._sequence, env._retained_capture_bytes = False, 0, 0
    env.config.expert_role, env.config.episode_steps = "local-navigation-policy", 3
    env._pending, env._observation = capture.request, capture.observation
    env._prepared_input = object()
    env._stream_captures = []
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock(), discard_pending=Mock())
    env._monitor = SimpleNamespace(initial_witness=Mock(return_value=capture.initial_witness))
    env._application = Mock()
    original_deadline = capture.request["valid_until_unix_ms"]
    # An expensive computation has already spent the admitted reserve.
    isolate_time(monkeypatch, native, time=lambda: (original_deadline - 39) / 1000.0)
    with pytest.raises(ControlPreparationExpired):
        env.submit_stream_action(PilotAction(mode="pilot-control", axes=[0.2, 0.0, 0.0, 0.0]))
    env._exchange.reply.assert_not_called()
    env._application.assert_not_called()
    env._exchange.discard_pending.assert_called_once()
    assert env._sequence == 0 and not env._stream_captures
    assert env._retained_capture_bytes == 0
    assert env._pending is env._prepared_input is None
    assert capture.request["valid_until_unix_ms"] == original_deadline


# 功能：
#   验证过期准备只对新源重新计算，成功提交前不延长原始进度截止时刻。
# 输入：
#   monkeypatch：提供可推进时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_expired_preparations_retry_only_new_sources_and_keep_progress_deadline(monkeypatch):
    env = StreamLifecycle()
    clock, deadlines, attempted, sent = [1.0], [], [], []
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    next_source, submit = env.next_stream_observation, env.submit_stream_action

    # 功能：
    #   记录每次传递的截止时刻，再取得替身环境的下一条源观测。
    # 输入：
    #   deadline：采集器已有的绝对单调截止秒数。
    # 输出：
    #   observation：合成的新源观测。
    def next_observation(*, deadline):
        deadlines.append(deadline)
        observation = next_source(deadline=deadline)
        return observation

    # 功能：
    #   记录实际采样源并模拟一次推理耗时，生成可区分各次尝试的前向轴幅度。
    # 输入：
    #   observation：当前新源观测。
    # 输出：
    #   action：本次尝试对应的合成控制提案。
    def actor(observation):
        attempted.append(observation.sample.temporal_evidence.observed_at_unix_ms)
        clock[0] += 0.1
        action = PilotAction(mode="pilot-control", axes=[len(attempted) / 10.0, 0.0, 0.0, 0.0])
        return action

    # 功能：
    #   拒绝前两次已过期准备，之后记录实际交给替身环境的轴幅度。
    # 输入：
    #   action：采集器尝试提交的控制提案。
    # 输出：
    #   None：不返回业务数据。
    def dispatch(action):
        if len(attempted) <= 2:
            raise ControlPreparationExpired("not sent")
        sent.append(action.axes[0])
        submit(action)

    env.next_stream_observation, env.submit_stream_action = next_observation, dispatch
    collect(env, actor, steps=2)
    assert len(attempted) == len(set(attempted)) == 4
    assert sent == [0.3, 0.4]
    assert deadlines[:2] == [3.0, 3.0]
    assert deadlines[2] == pytest.approx(3.3)
    assert env.events[-2:] == ["ground", "join-actual-receipts"]


# 功能：
#   验证终止提案过期后结束采集，不向模型索要新的运动决策。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_expired_abort_is_not_replaced_by_a_new_motion_decision():
    env = StreamLifecycle()
    env.submit_stream_action = Mock(side_effect=ControlPreparationExpired("not sent"))
    actor = Mock(return_value=PilotAction(mode="abort", axes=[0.0] * 4))
    with pytest.raises(ControlPreparationExpired):
        collect(env, actor)
    actor.assert_called_once()
    assert env.events == ["reset", "ground"]


# 功能：
#   验证传输或身份故障不能伪装成可重试的准备过期，仍必须执行停止流程。
# 输入：
#   fault：待注入的非准备过期异常。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "fault", [TimeoutError("transport"), ConnectionError("transport"), ValueError("identity")]
)
def test_other_submission_faults_are_never_retried(fault):
    env = StreamLifecycle()
    env.submit_stream_action = Mock(side_effect=fault)
    with pytest.raises(type(fault), match=str(fault)):
        collect(env)
    assert env.events == ["reset", "ground"]


# 功能：
#   验证准备失败后再次收到同一源观测也不能重复决策或发送动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rejected_preparation_does_not_permit_replayed_input():
    env = StreamLifecycle()
    env.submit_stream_action = Mock(side_effect=ControlPreparationExpired("not sent"))
    env.next_stream_observation = lambda **_: env.inner.observation()
    with pytest.raises(ValueError, match="REPLAYED_ENVIRONMENT_STATE"):
        collect(env)
    assert env.events == ["reset", "ground"]


# 功能：
#   验证推理耗时已用尽进度期限时，不能通过重新计时继续控制。
# 输入：
#   monkeypatch：固定并推进单调时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_inference_cannot_extend_progress_deadline(monkeypatch):
    env, clock = StreamLifecycle(), [1.0]
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    # 功能：
    #   模拟耗费两秒的策略计算，再返回中性提案。
    # 输入：
    #   _：未使用的合成观测。
    # 输出：
    #   action：在进度时限处应被拒绝的保持提案。
    def actor(_):
        clock[0] += 2.0
        action = PilotAction(mode="hold", axes=[0.0] * 4)
        return action

    with pytest.raises(TimeoutError, match="PROGRESS_TIMEOUT"):
        collect(env, actor)
    assert env.events == ["reset", "ground"]


# 功能：
#   验证时钟不推进且准备持续失败时也有有限尝试次数，不形成无限循环。
# 输入：
#   monkeypatch：固定不前进时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_nonprogressing_preparation_has_a_finite_attempt_bound(monkeypatch):
    env = StreamLifecycle()
    monkeypatch.setattr(collection, "time", SimpleNamespace(monotonic=lambda: 1.0))
    env.submit_stream_action = Mock(side_effect=ControlPreparationExpired("not sent"))
    with pytest.raises(TimeoutError, match="PREPARATION_ATTEMPT_LIMIT"):
        collect(env, steps=1)
    assert env.submit_stream_action.call_count == 16
    assert env.events[-1] == "ground" and "join-actual-receipts" not in env.events


# 功能：
#   建立只测试流式等待语义的替身环境，不生成控制或物理状态。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   env：持有前次时间戳和可检查拒绝回调的环境。
def stream_wait_env(tmp_path):
    env = environment(tmp_path)
    env._observation = SimpleNamespace(
        sample=SimpleNamespace(temporal_evidence=SimpleNamespace(observed_at_unix_ms=1000))
    )
    env._exchange = SimpleNamespace(discard_pending=Mock())
    env._record_rejected_input = Mock()
    return env


# 功能：
#   验证布尔、非有限、超预算或错误类型的等待期限在接收新输入前被拒绝。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定当前单调时间的测试工具。
#   deadline：待拒绝的非法期限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "deadline", [False, True, 0.0, -1.0, float("nan"), float("inf"), 12.01, "12", 10**400]
)
def test_invalid_stream_wait_bound_does_not_consume_input(tmp_path, monkeypatch, deadline):
    env = stream_wait_env(tmp_path)
    env._next_request = Mock()
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: 10.0))
    with pytest.raises(ValueError, match="WAIT_DEADLINE_INVALID"):
        env.next_stream_observation(deadline=deadline)
    env._next_request.assert_not_called()


# 功能：
#   验证已过期的合法进度期限原样触发超时，不能重新取得输入。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定当前时间的测试工具。
# 输出：
#   None：不返回业务数据。
def test_expired_progress_deadline_stays_expired(tmp_path, monkeypatch):
    env = stream_wait_env(tmp_path)
    env._receive_source_request = Mock()
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: 10.0))
    with pytest.raises(TimeoutError, match="NEXT_OBSERVATION_TIMEOUT"):
        env.next_stream_observation(deadline=9.0)
    env._receive_source_request.assert_not_called()
    assert env._rollout_stop_reason["required_remaining_input_ms"] == 40


# 功能：
#   验证采用时过期的源推进去重水位，后续只采用更晚源且不延长等待期限。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定当前单调时间的测试工具。
# 输出：
#   None：不返回业务数据。
def test_adoption_expiry_advances_source_watermark_without_extending_wait(tmp_path, monkeypatch):
    env = stream_wait_env(tmp_path)

    # 功能：
    #   构造只携带原生时间戳的等待器夹具请求。
    # 输入：
    #   stamp：原生观测的 Unix 毫秒时刻。
    # 输出：
    #   payload：供等待器去重的合成请求。
    def request(stamp):
        payload = {"observation": {"temporal_evidence": {"observed_at_unix_ms": stamp}}}
        return payload

    expired, repeated, fresh = request(1010), request(1010), request(1020)
    env._next_request = Mock(side_effect=[expired, repeated, fresh])
    env._adopt = Mock(
        side_effect=[TrainingObservationError("TRAINING_INPUT_SENSOR_EVIDENCE_EXPIRED"), "fresh"]
    )
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: 10.0))
    assert env.next_stream_observation(deadline=11.0) == "fresh"
    assert [call.kwargs["deadline"] for call in env._next_request.call_args_list] == [11.0] * 3
    assert [call.args[0] for call in env._adopt.call_args_list] == [expired, fresh]
    assert env._exchange.discard_pending.call_count == 2


# 功能：
#   验证来源摘要错误直接终止采用，不按过期重试掩盖证据故障。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_adoption_provenance_fault_is_not_reclassified_as_expiry(tmp_path):
    env = stream_wait_env(tmp_path)
    env._next_request = Mock(
        return_value={"observation": {"temporal_evidence": {"observed_at_unix_ms": 1010}}}
    )
    env._adopt = Mock(side_effect=TrainingObservationError("TRAINING_INPUT_SNAPSHOT_HASH_MISMATCH"))
    with pytest.raises(TrainingObservationError, match="HASH_MISMATCH"):
        env.next_stream_observation()
    env._next_request.assert_called_once()
