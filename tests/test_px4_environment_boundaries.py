"""Environment ownership and cleanup faults; never launches a simulation or aircraft."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_causal_policy import samples
from test_native_action_risk_artifacts import native_episode
from test_px4_training_lifecycle import environment
from test_training_observation_boundary import current_request, observation

from dronedream_agent_core.training import evidence_publication
from dronedream_agent_core.training import px4_environment as module
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   构造真实准备对象与合成观测连接的环境，不启动运行器或伪造执行回执。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定当前观测时钟的测试工具。
# 输出：
#   env：可以接纳该请求的离线环境。
#   request：用于检查输入所有权的原始请求。
def adopting_environment(tmp_path, monkeypatch):
    env = environment(tmp_path)
    request = current_request()
    env._sequence, env.visual = 0, None
    env.config.mission_id = "unit"
    env._prepared_input = module.PreparedTrainingInput.from_request(request)
    env.config.expert_role = env._prepared_input.sample.navigation_expert_role
    stamp = request["snapshot"]["control_reference_observed_at_unix_ms"]
    monkeypatch.setattr(module.time, "time", lambda: stamp / 1000.0)
    return env, request


# 功能：
#   验证无覆盖发布失败时不误删后来占用暂存名称的其他文件。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：注入暂存名称被替换及链接失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_publish_failure_preserves_replaced_staging_file(tmp_path, monkeypatch):
    replacement = []

    # 功能：
    #   保存原始暂存文件后占用其旧名称，模拟发布失败时的身份竞争。
    # 输入：
    #   source：发布器准备链接的暂存路径。
    #   destination：最终目标路径，本回调不会创建它。
    # 输出：
    #   None：不返回业务数据。
    def fail_link(source, destination):
        source = Path(source)
        source.rename(tmp_path / "original-retained.json")
        source.write_text("unrelated", encoding="utf-8")
        replacement.append(source)
        raise OSError("injected-publication-failure")

    monkeypatch.setattr(evidence_publication.os, "link", fail_link)
    with pytest.raises(OSError, match="injected-publication-failure"):
        module._write_new(tmp_path / "final.json", {"complete": True})
    assert replacement[0].read_text() == "unrelated"
    assert not (tmp_path / "final.json").exists()


# 功能：
#   验证视觉关闭失败不掩盖原始落地失败，且故障关闭后不能重新接纳动作。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_close_preserves_landing_failure_and_seals_environment(tmp_path):
    env = environment(tmp_path)
    env._quiesced = False
    primary = RuntimeError("native-ground-proof-missing")
    env.quiesce = Mock(side_effect=primary)
    env.visual = SimpleNamespace(close=Mock(side_effect=OSError("visual-close-failed")))
    with pytest.raises(RuntimeError, match="native-ground-proof-missing") as caught:
        env.close()
    assert caught.value is primary
    assert env._closed and not env._quiesced
    assert "visual-close-failed" in repr(caught.value.__notes__)
    env.visual.close.assert_called_once()


# 功能：
#   验证源历史持有独立数值容器，调用方后续更改样本不会改写过去证据。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_source_history_is_owned(tmp_path):
    env = environment(tmp_path)
    sample = observation(samples()[0])
    original = sample.state_features[0]
    env._retain_source_observation(sample)
    sample.state_features[0] += 1.0
    assert env._source_history[0].state_features[0] == original


# 功能：
#   验证返回学习器的观测和原始请求都不能改写环境内部待执行证据。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定源时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_adopt_detaches_request_and_returned_observation(tmp_path, monkeypatch):
    env, request = adopting_environment(tmp_path, monkeypatch)
    result = env._adopt(request)
    original = result.sample.state_features[0]
    result.sample.state_features[0] += 1.0
    request["snapshot"]["snapshot_sha256"] = "f" * 64
    assert env._observation.sample.state_features[0] == original
    assert env._pending["snapshot"]["snapshot_sha256"] != "f" * 64


# 功能：
#   验证模型实例内部列表被绕过验证改坏后，在任何提案发送前拒绝。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定源时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_mutated_action_is_revalidated_before_submission(tmp_path, monkeypatch):
    env, request = adopting_environment(tmp_path, monkeypatch)
    env._adopt(request)
    env._quiesced = False
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock())
    action = PilotAction(mode="pilot-control", axes=[0.0] * 4)
    action.axes.pop()
    with pytest.raises(ValueError):
        env._submit_action(action)
    env._exchange.reply.assert_not_called()


# 功能：
#   验证非法计时不污染稍后写入的诊断 JSON，也不改变已保存的时间记录。
# 输入：
#   tmp_path：独立测试目录。
#   field：待破坏的诊断字段。
#   value：非法的字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", True),
        ("sequence", -1),
        ("input_preparation_ms", float("nan")),
        ("actor_sampling_ms", -0.1),
        ("actor_sampling_ms", float("inf")),
    ],
)
def test_invalid_actor_timing_is_rejected_before_append(tmp_path, field, value):
    env = environment(tmp_path)
    timing = dict(sequence=0, input_preparation_ms=1.0, actor_sampling_ms=2.0)
    timing[field] = value
    with pytest.raises(ValueError, match="ACTOR_TIMING_INVALID"):
        env.record_actor_timing(**timing)
    assert not env._interface_timings


# 功能：
#   验证未知采集模式不能写入回合状态。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_unknown_collection_mode_does_not_latch(tmp_path):
    env = environment(tmp_path)
    with pytest.raises(ValueError, match="COLLECTION_MODE_INVALID"):
        env._select_collection_mode("reward-steps")
    assert getattr(env, "_collection_mode", None) is None


# 功能：
#   验证同步奖励入口达到步数上限后，不会额外发送一个动作。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_reward_step_limit_precedes_submission(tmp_path):
    env = environment(tmp_path)
    env._sequence = env.config.episode_steps = 3
    env._submit_action = Mock(side_effect=AssertionError("must-not-submit"))
    with pytest.raises(ValueError, match="REWARD_ACTION_LIMIT_REACHED"):
        env.capture_step(PilotAction(mode="hold", axes=[0.0] * 4))
    env._submit_action.assert_not_called()


# 功能：
#   验证构造器复制已验证配置，调用方不能改写正在使用的限速或资产身份。
# 输入：
#   tmp_path：合成资产目录。
#   monkeypatch：只替换平台检查，不执行原生运行器的测试工具。
# 输出：
#   None：不返回业务数据。
def test_constructor_owns_configuration(tmp_path, monkeypatch):
    episode, _ = native_episode(tmp_path)
    config = module.Px4TrainingConfig.model_validate_json(
        json.dumps(json.loads((episode / "reset.json").read_text())["config"])
    )
    config.runner = (
        Path(module.__file__).resolve().parents[3]
        / "scripts"
        / ("run_school_map_depth_qualification.py")
    )
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    env = module.Px4GazeboTrainingEnvironment(config)
    original = env.config.asset_sha256["semantic"]
    config.asset_sha256["semantic"] = "f" * 64
    config.speed_limit_mps = 1.5
    assert env.config.asset_sha256["semantic"] == original
    assert env.config.speed_limit_mps == 0.4
    env.close()


# 功能：
#   验证原生阶段在推理期间进入结束状态后，已接纳的旧观测也不能发送新控制。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定源时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_ending_during_inference_blocks_submission(tmp_path, monkeypatch):
    env, request = adopting_environment(tmp_path, monkeypatch)
    env._adopt(request)
    env._quiesced = False
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock(), discard_pending=Mock())
    env._phase_observer = SimpleNamespace(latest=lambda: {"executor_phase": "LANDING"})
    with pytest.raises(RuntimeError, match="EXECUTOR_ENDING:LANDING"):
        env._submit_action(PilotAction(mode="pilot-control", axes=[0.1, 0.0, 0.0, 0.0]))
    env._exchange.reply.assert_not_called()
    assert env._pending is None and not env._quiesced


# 功能：
#   验证动作通道拿到的提案副本不与返回并归档的提案共享轴列表。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定源时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_reply_callback_cannot_rewrite_retained_proposal(tmp_path, monkeypatch):
    env, request = adopting_environment(tmp_path, monkeypatch)
    env._adopt(request)
    env._quiesced = False
    env._transition_writer = SimpleNamespace(check=Mock())

    # 功能：
    #   模拟会原地修改传入对象的通道替身，不进行真实控制发送。
    # 输入：
    #   proposal：环境交给通道的独立提案。
    # 输出：
    #   None：不返回业务数据。
    def mutate(proposal):
        proposal.action.axes[0] = 0.9

    env._exchange = SimpleNamespace(reply=mutate)
    action = PilotAction(mode="pilot-control", axes=[0.1, 0.0, 0.0, 0.0])
    _, _, proposal = env._submit_action(action)
    assert proposal.action.axes[0] == action.axes[0] == 0.1


# 功能：
#   验证重置失败后再次清理失败时保留最早的启动异常及附加清理信息。
# 输入：
#   tmp_path：独立资产目录。
# 输出：
#   None：不返回业务数据。
def test_failed_reset_preserves_original_error(tmp_path):
    env = environment(tmp_path)
    primary = RuntimeError("runner-start-failed")
    env._start_episode = Mock(side_effect=primary)
    env.quiesce = Mock(side_effect=[None, OSError("cleanup-failed")])
    with pytest.raises(RuntimeError, match="runner-start-failed") as caught:
        env.reset(seed=5)
    assert caught.value is primary
    assert "cleanup-failed" in repr(primary.__notes__)
    assert env.quiesce.call_count == 2


# 功能：
#   验证两种阻塞等待入口先拒绝非法期限，不访问接收端或回执端。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定单调时钟的测试工具。
#   deadline：非法等待截止值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), 10**400, 611.0])
def test_wait_deadline_validation_precedes_io(tmp_path, monkeypatch, deadline):
    env = environment(tmp_path)
    env._exchange = Mock()
    env._receipt_monitor = Mock()
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 10.0))
    with pytest.raises(ValueError, match="WAIT_DEADLINE_INVALID"):
        env._next_request(deadline=deadline)
    with pytest.raises(ValueError, match="WAIT_DEADLINE_INVALID"):
        env._application("unit", deadline=deadline)
    assert not env._exchange.mock_calls and not env._receipt_monitor.mock_calls


# 功能：
#   验证有歧义的停止诊断只能标记不可用，不能选择其中一个成功字段作为结论。
# 输入：
#   tmp_path：合成停止证据目录。
# 输出：
#   None：不返回业务数据。
def test_ambiguous_stop_diagnostics_do_not_claim_success(tmp_path):
    env = environment(tmp_path)
    env._quiesced = False
    env.simulation = tmp_path
    env._rollout_stop_reason = {"reason": "unit", "qualified_for_flight": False}
    (tmp_path / "offboard_timing.json").write_text(
        '{"status":"failed","status":"passed"}', encoding="utf-8"
    )
    env.quiesce()
    record = json.loads((tmp_path / "rollout-stop-reason.json").read_text())
    assert record["executor_outcome"] == {"status": "unavailable"}


# 功能：
#   检查同步入口的终止和截断都会请求安全停止，未结束步骤继续采集。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：隔离奖励验证器的测试工具。
#   terminated：任务是否终止。
#   truncated：采集是否达到步数上限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True)])
def test_reward_step_quiesces_for_either_terminal_condition(
    tmp_path, monkeypatch, terminated, truncated
):
    env = environment(tmp_path)
    env._sequence = 1
    env._pending = {"valid_until_unix_ms": 1000}
    env.verifier = object()
    env._transition_writer = SimpleNamespace(submit=Mock())
    env.capture_step = Mock(return_value=object())
    env.quiesce = Mock()
    expected = SimpleNamespace(terminated=terminated, truncated=truncated)
    monkeypatch.setattr(module, "evaluate_native_transition", lambda *_: (expected, {}))
    result = env.step(PilotAction(mode="hold", axes=[0.0] * 4))
    assert result is expected
    assert env.quiesce.call_count == int(terminated or truncated)
    env._transition_writer.submit.assert_called_once()


# 功能：
#   验证空、错误类型或超量的延期捕获集合不能写出完整结算回执。
# 输入：
#   tmp_path：独立测试目录。
#   captures：错误的集合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("captures", [None, [], {}, [None] * 257])
def test_finalize_rejects_invalid_collection_before_writes(tmp_path, captures):
    env = environment(tmp_path)
    with pytest.raises(ValueError, match="CAPTURE_COUNT_INVALID"):
        env.finalize_captures(captures)
    assert not (tmp_path / "deferred-outcome-receipt.json").exists()
