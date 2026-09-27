"""Fault-injection lifecycle tests; no aircraft or simulated flight is launched."""

import hashlib
import json
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dronedream_agent_core.runtime_phase import runtime_phase_context
from dronedream_agent_core.training import px4_environment as module


# 功能：
#   创建无真实进程的生命周期夹具，保留真实路径、摘要与阶段读取逻辑。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   env：使用确定性替身的训练环境，不具备真实飞行资格。
def environment(tmp_path):
    env = object.__new__(module.Px4GazeboTrainingEnvironment)
    env._closed, env._quiesced = False, True
    env._policy_sha256 = "a" * 64
    env._process = env._exchange = env._monitor = env._log = env._pending = None
    env._interface_timings = deque(maxlen=512)
    env.episode_path = tmp_path
    env._source_history = deque(maxlen=32)
    # Deterministic lifecycle source for these consumer tests. The production
    # threaded reader and its nonblocking behavior are covered separately.
    env._phase_observer = SimpleNamespace(
        latest=lambda: runtime_phase_context(tmp_path / "runtime-phase.json"),
        close=lambda: {"test_double": True, "confirms_physical_landing": False},
    )
    asset = tmp_path / "asset.json"
    asset.write_text("{}", encoding="utf-8")
    env.config = SimpleNamespace(
        initial_collection_mode=None,
        output_root=tmp_path,
        quiesce_timeout_seconds=30,
        episode_steps=256,
        asset_sha256={
            name: hashlib.sha256(asset.read_bytes()).hexdigest() for name in module.ASSET_FIELDS
        },
        **{name: asset for name in module.ASSET_FIELDS},
    )
    from test_mission_groups import route

    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps(route([(0, 0, 1), (1, 0, 1)])))
    env.config.route = route_path
    env.config.asset_sha256["route"] = hashlib.sha256(route_path.read_bytes()).hexdigest()
    env.mission_split = module.mission_group_evidence(
        route_path.read_bytes(), env.config.asset_sha256["semantic"]
    )
    return env


# 功能：
#   验证同时改写路线和配置摘要也不能改掉构造时绑定的任务分区。
# 输入：
#   tmp_path：合成路线目录。
# 输出：
#   None：不返回业务数据。
def test_mutating_route_and_config_hash_does_not_change_bound_mission(tmp_path):
    from test_mission_groups import route

    env = environment(tmp_path)
    env.config.route.write_text(json.dumps(route([(0, 0, 1), (2, 0, 1)])))
    env.config.asset_sha256["route"] = hashlib.sha256(env.config.route.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="MISSION_CHANGED_BEFORE_RESET"):
        env.reset(seed=1)
    assert not list(tmp_path.glob("episode-*"))


# 功能：
#   验证证据发布不覆盖操作员终止请求，非有限 JSON 失败不留下暂存文件。
# 输入：
#   tmp_path：独立文件目录。
# 输出：
#   None：不返回业务数据。
def test_atomic_evidence_preserves_existing_operator_abort(tmp_path):
    path = tmp_path / "abort.json"
    module._write_new(path, {"reason": "operator"})
    with pytest.raises(FileExistsError):
        module._write_new(path, {"reason": "learner"})
    assert json.loads(path.read_text()) == {"reason": "operator"}
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    other = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        module._write_new(other, {"value": float("nan")})
    assert not other.exists() and not list(tmp_path.glob("*.pending"))


# 功能：
#   在观察器启动失败时检查端点关闭、描述文件清理与静止状态恢复。
# 输入：
#   tmp_path：独立回合目录。
#   monkeypatch：注入观察器构造失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_observer_start_cleans_training_descriptor(tmp_path, monkeypatch):
    env = environment(tmp_path)
    monkeypatch.setattr(module, "IndependentPoseMonitor", Mock(side_effect=OSError("observer")))
    with pytest.raises(OSError, match="observer"):
        env.reset(seed=1)
    assert env._quiesced and env._exchange is None
    assert not list(tmp_path.glob("episode-*/policy-channel.json"))


# 功能：
#   验证退出码不能证明落地，补齐原生地面证据后才能完成停止流程。
# 输入：
#   tmp_path：合成生命周期证据目录。
# 输出：
#   None：不返回业务数据。
def test_exit_code_alone_never_authorizes_reset_as_grounded(tmp_path):
    env = environment(tmp_path)
    env._quiesced = False
    env.simulation = tmp_path
    env._process = SimpleNamespace(poll=lambda: 0)
    env._exchange = SimpleNamespace(close=Mock())
    with pytest.raises(FileNotFoundError):
        env.quiesce()
    assert not env._quiesced
    env._exchange.close.assert_called_once()
    module._write_new(
        tmp_path / "native-terminal-lifecycle.json",
        {
            "terminal_state": "ON_GROUND",
            "landing_confirmed": True,
            "safe_to_stop_watchdog": True,
        },
    )
    env.quiesce()
    assert env._quiesced and env._process is None


# 功能：
#   验证连接取消可以等待新源，而鉴权失败直接上抛，不形成虚假动作转移。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：提供已验证观测替身的测试工具。
# 输出：
#   None：不返回业务数据。
def test_cancelled_request_without_packet_is_not_a_flight_transition(tmp_path, monkeypatch):
    from test_causal_policy import samples
    from test_training_observation_boundary import observation

    env = environment(tmp_path)
    env._process = SimpleNamespace(poll=lambda: None)
    request = {
        "valid_until_unix_ms": int(time.time() * 1000) + 200,
        "snapshot": {"control_reference_observed_at_unix_ms": 1000},
    }
    monkeypatch.setattr(
        module.PreparedTrainingInput,
        "from_request",
        lambda *args: SimpleNamespace(sample=observation(samples()[0])),
    )
    env._exchange = SimpleNamespace(wait_request=Mock(side_effect=[ConnectionError(), request]))
    assert env._next_request(deadline=time.monotonic() + 1) is request
    assert env._exchange.wait_request.call_count == 2
    env._exchange.wait_request.side_effect = ValueError("TRAINING_POLICY_AUTHENTICATION_FAILED")
    with pytest.raises(ValueError, match="AUTHENTICATION_FAILED"):
        env._next_request(deadline=time.monotonic() + 1)


# 功能：
#   检查明确未启动证明的每个必要字段，错误路径或启动状态不得替代真实落地。
# 输入：
#   tmp_path：独立证据目录。
#   fault：待破坏字段类别，None 表示合法未启动证明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", [None, "path", "spawn-attempt", "alive-process", "state"])
def test_preflight_stop_is_not_a_fake_landing_or_success(tmp_path, fault):
    env = environment(tmp_path)
    env._quiesced = False
    env.simulation = tmp_path
    env._process = SimpleNamespace(poll=lambda: 1)
    receipt = {
        "run_directory": str(tmp_path.resolve()),
        "flight_vehicle_spawn_attempted": False,
        "all_started_processes_exited": True,
        "terminal_state": "NOT_STARTED",
        "qualification_granted": False,
    }
    changes = {
        "path": {"run_directory": str(tmp_path.parent)},
        "spawn-attempt": {"flight_vehicle_spawn_attempted": True},
        "alive-process": {"all_started_processes_exited": False},
        "state": {"terminal_state": "ON_GROUND"},
    }
    receipt.update(changes.get(fault, {}))
    module._write_new(tmp_path / "simulation-preflight-stop.json", receipt)
    if fault:
        with pytest.raises(FileNotFoundError):
            env.quiesce()
        assert not env._quiesced
    else:
        env.quiesce()
        assert env._quiesced
        assert not (tmp_path / "native-terminal-lifecycle.json").exists()


# 功能：
#   验证未发送动作也可保留真实历史，重复原生状态不增行，大间隔重新开始历史。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_source_history_retains_unacted_observations_but_never_renews_their_time(tmp_path):
    from test_causal_policy import samples
    from test_training_observation_boundary import observation

    env = environment(tmp_path)
    rows = [observation(s) for s in samples()[:4]]
    for row in rows:
        env._retain_source_observation(row)
    assert list(env._source_history) == rows
    env._retain_source_observation(rows[-1])
    assert len(env._source_history) == 4
    repeated_native_state = rows[-1].model_copy(
        update={
            "source_snapshot_sha256": "e" * 64,
            "state_features": [v + 0.001 for v in rows[-1].state_features],
        }
    )
    env._retain_source_observation(repeated_native_state)
    assert list(env._source_history) == rows
    later = rows[-1].model_copy(
        update={
            "temporal_evidence": rows[-1].temporal_evidence.model_copy(
                update={
                    "observed_at_unix_ms": rows[-1].temporal_evidence.observed_at_unix_ms + 251,
                    "sample_sha256": "c" * 64,
                }
            )
        }
    )
    env._retain_source_observation(later)
    assert list(env._source_history) == [later]


# 功能：复现原生同刻状态的合法修订；训练与部署保持同槽替换，不伪造独立时间步。
# 输入：tmp_path：隔离环境。输出：数量不增加，真实回退、旧修订和重标时间仍拒绝。
def test_source_history_revisions_match_deployed_history(tmp_path):
    from test_causal_policy import samples
    from test_training_observation_boundary import observation
    from dronedream_agent_core.causal_control import CausalControlHistory

    env = environment(tmp_path)
    runtime = CausalControlHistory(4)
    rows = [observation(s) for s in samples()[:4]]
    revised = rows[-1].model_copy(deep=True)
    revised.temporal_evidence = revised.temporal_evidence.model_copy(update={
        'sample_sha256':'e'*64, 'history_slot_revision':1})
    revised.state_features[0] += .001
    for row in [*rows, revised, revised]:
        env._retain_source_observation(row)
        runtime.append(row.temporal_evidence, row.state_features,
                       row.realtime_features, row.realtime_valid_mask)
    assert len(env._source_history) == len(runtime.history.rows) == 4
    assert env._source_history[-1] == revised
    assert runtime.history.latest == revised.temporal_evidence
    assert env._source_history[-1] is not revised
    with pytest.raises(ValueError, match='SAME_TIME_CONFLICT'):
        env._retain_source_observation(rows[-1])
    bad = revised.model_copy(deep=True)
    bad.temporal_evidence = revised.temporal_evidence.model_copy(update={
        'observed_at_unix_ms': revised.temporal_evidence.observed_at_unix_ms+1})
    with pytest.raises(ValueError, match='SAMPLE_REDATED'):
        env._retain_source_observation(bad)
    bad.temporal_evidence = revised.temporal_evidence.model_copy(update={
        'observed_at_unix_ms':revised.temporal_evidence.observed_at_unix_ms-1,
        'sample_sha256':'f'*64})
    with pytest.raises(ValueError, match='CLOCK_REGRESSED'):
        env._retain_source_observation(bad)
    assert not env._source_history


# 功能：
#   等待执行回执时连续接收新观测，验证只保存历史、不发送替代控制。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：注入真实形状观测的测试工具。
# 输出：
#   None：不返回业务数据。
def test_execution_wait_keeps_real_observations_without_actions(tmp_path, monkeypatch):
    from test_causal_policy import samples
    from test_training_observation_boundary import observation

    env = environment(tmp_path)
    env._process = SimpleNamespace(poll=lambda: None)
    rows = [observation(s) for s in samples()[:3]]
    env._retain_source_observation(rows[0])
    requests = [
        {
            "valid_until_unix_ms": int(time.time() * 1000) + 200,
            "snapshot": {"control_reference_observed_at_unix_ms": 1000 + i * 50},
        }
        for i in range(2)
    ]
    env._exchange = SimpleNamespace(
        wait_request=Mock(side_effect=requests), discard_pending=Mock(), reply=Mock()
    )
    env._receipt_monitor = SimpleNamespace(
        poll=Mock(side_effect=[None, None, None, None, ("command", "ack")])
    )
    monkeypatch.setattr(
        module.PreparedTrainingInput,
        "from_request",
        Mock(side_effect=[SimpleNamespace(sample=row) for row in rows[1:]]),
    )
    assert env._application("model-x", deadline=time.monotonic() + 1) == ("command", "ack")
    assert list(env._source_history) == rows
    assert env._exchange.discard_pending.call_count == 2
    env._exchange.reply.assert_not_called()


# 功能：
#   检查同步输入接纳边界包含准备及派发余量，少一毫秒就拒绝。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定单调和 Unix 时钟的测试工具。
# 输出：
#   None：不返回业务数据。
def test_input_admission_reserves_complete_learner_and_dispatch_budget(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock())
    budget = module.LOCAL_DISPATCH_RESERVE_MS + module.TRAINING_REPLY_PREPARATION_RESERVE_MS
    rejected, accepted = (
        {"valid_until_unix_ms": 1000 + budget - 1},
        {"valid_until_unix_ms": 1000 + budget},
    )
    env._receive_source_request = Mock(side_effect=[rejected, accepted])
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 1.0))
    assert env._next_request(deadline=2.0) is accepted
    env._exchange.discard_pending.assert_called_once()
    assert env._input_rejection_counts == {"insufficient-input-budget": 1}
    assert env._interface_timings[-1]["remaining_input_lease_ms"] == budget - 1


# 功能：接收阶段丢弃过期输入，不改变新帧的期限；输入：过期异常；输出：下一真实请求。
@pytest.mark.parametrize('reason', sorted(module.EXPIRED_POLICY_REQUESTS))
def test_expired_transport_waits_for_new_input(tmp_path, monkeypatch, reason):
    env = environment(tmp_path)
    env._process = SimpleNamespace(poll=lambda: None)
    accepted = {'valid_until_unix_ms':1240}
    env._receive_source_request = Mock(side_effect=[ValueError(reason),accepted])
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda:1.,monotonic=lambda:1.))
    assert env._next_request(deadline=2.) is accepted
    assert accepted['valid_until_unix_ms'] == 1240


# 功能：超长租约不是普通过期，不得静默重试；输入：非法期限；输出：原错误。
def test_future_lease_is_not_ignored(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env._process = SimpleNamespace(poll=lambda: None)
    env._receive_source_request = Mock(side_effect=ValueError('TRAINING_POLICY_INPUT_LEASE_EXCEEDS_BOUND'))
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda:1.,monotonic=lambda:1.))
    with pytest.raises(ValueError, match='LEASE_EXCEEDS_BOUND'):
        env._next_request(deadline=2.)


# 功能：
#   验证诊断明细滚动淘汰后累计拒绝次数仍完整，且不改写原始请求期限。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：固定当前时间的测试工具。
# 输出：
#   None：不返回业务数据。
def test_input_rejection_totals_survive_bounded_diagnostic_tail(tmp_path, monkeypatch):
    env = environment(tmp_path)
    request = {"valid_until_unix_ms": 1120, "snapshot": {"snapshot_sha256": "a" * 64}}
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 1.0))
    for _ in range(520):
        env._record_rejected_input(request, "independent-start-baseline-missing")
    env._record_rejected_input(request, "input-expired-at-adoption")
    assert len(env._interface_timings) == 512
    assert env._input_rejection_counts == {
        "independent-start-baseline-missing": 520,
        "input-expired-at-adoption": 1,
    }
    assert env._interface_timings[-1]["snapshot_sha256"] == "a" * 64
    assert request["valid_until_unix_ms"] == 1120


# 功能：
#   验证保持状态仍可解释底层任务阶段，但阶段上下文不传递调度授权。
# 输入：
#   tmp_path：合成执行阶段目录。
# 输出：
#   None：不返回业务数据。
def test_local_hold_preserves_underlying_mission_phase_without_granting_authority(tmp_path):
    from test_runtime_commands import _load_depth_worker

    worker = _load_depth_worker()
    path = tmp_path / "runtime-phase.json"
    module._write_new(
        path,
        {
            "phase": "MODEL_AUTHORITY_HOLD",
            "schedule_advancement_authorized": False,
            "enclosing_executor_state": {"phase": "TRACK", "checkpoint_id": None},
        },
    )
    context = worker._runtime_phase_context(path)
    assert context["phase"] == "MODEL_AUTHORITY_HOLD"
    assert context["executor_phase"] == "TRACK"
    assert "schedule_advancement_authorized" not in context


# 功能：
#   验证结束阶段停止新模型请求，仍需独立的原生落地证据才允许结算。
# 输入：
#   tmp_path：独立阶段及落地证据目录。
#   phase：四种结束阶段之一。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", ["LANDING", "LANDED", "COMPLETE", "FAILED"])
def test_shutdown_stops_new_model_requests_but_never_confirms_physical_landing(tmp_path, phase):
    from test_runtime_commands import _load_depth_worker

    worker = _load_depth_worker()
    assert (
        worker._model_cycle_trigger(
            model_cycle_started=False,
            now_monotonic=5.0,
            next_model_cycle_monotonic=0.0,
            target_changed=True,
            dynamic_obstacle_changed=True,
            progress_recovery_requested=True,
            executor_phase=phase,
        )
        is None
    )
    env = environment(tmp_path)
    env._quiesced = False
    env._process = SimpleNamespace(poll=lambda: 1)
    env._exchange = SimpleNamespace(discard_pending=Mock(), close=Mock())
    env.simulation = env.episode_path = tmp_path
    env._prepared_input = SimpleNamespace(
        admit=lambda *args, **kw: SimpleNamespace(
            source_snapshot_sha256="a" * 64, navigation_expert_role="precision-maneuver-policy"
        )
    )
    env.config.expert_role = "local-navigation-policy"
    request = {"snapshot": {"strategic_context": {"task": {"executor_phase": phase}}}}
    with pytest.raises(RuntimeError, match="PX4_TRAINING_EXECUTOR_ENDING:" + phase):
        env._adopt(request)
    env._exchange.discard_pending.assert_called_once()
    assert not (tmp_path / "rollout-stop-reason.json").exists()
    with pytest.raises(FileNotFoundError):
        env.quiesce()
    assert not env._quiesced  # Even a COMPLETE label is not landing telemetry.
    module._write_new(
        tmp_path / "native-terminal-lifecycle.json",
        {
            "terminal_state": "ON_GROUND",
            "landing_confirmed": True,
            "safe_to_stop_watchdog": True,
        },
    )
    env.quiesce()
    reason = json.loads((tmp_path / "rollout-stop-reason.json").read_text())
    assert reason["reason"] == "executor-ending" and reason["executor_phase"] == phase
    assert reason["qualified_for_flight"] is False


# 功能：
#   验证缺失或损坏阶段为未知，不误归结束；正常运行阶段仍可触发周期计算。
# 输入：
#   tmp_path：阶段读取夹具目录。
# 输出：
#   None：不返回业务数据。
def test_unknown_and_tracking_phase_remain_distinct_from_shutdown(tmp_path):
    from test_runtime_commands import _load_depth_worker

    worker = _load_depth_worker()
    for content in (None, "[]", "invalid"):
        phase_path = tmp_path / "phase.json"
        if content is not None:
            phase_path.write_text(content)
        assert worker._runtime_phase_context(phase_path)["executor_phase"] == "UNKNOWN"
    for phase in ("TRACK", "TAKEOFF", "UNKNOWN"):
        assert (
            worker._model_cycle_trigger(
                model_cycle_started=True,
                now_monotonic=5.0,
                next_model_cycle_monotonic=0.0,
                target_changed=False,
                dynamic_obstacle_changed=False,
                progress_recovery_requested=False,
                executor_phase=phase,
            )
            == "periodic"
        )


# 功能：
#   验证等待器在有无剩余时间时都识别结束阶段，并在落地后记录真实终止诊断摘要。
# 输入：
#   tmp_path：独立运行阶段与诊断目录。
#   phase：需要终止接收的执行阶段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", ["LANDING", "LANDED", "COMPLETE", "FAILED"])
def test_waiter_recognizes_executor_stop_without_a_new_actor_request(tmp_path, phase):
    env = environment(tmp_path)
    env._quiesced = False
    env.simulation = env.episode_path = tmp_path
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock(), wait_request=Mock(), close=Mock())
    module._write_new(tmp_path / "runtime-phase.json", {"phase": phase})
    # This also checks the exhausted wait path, not only the top of the loop.
    for deadline in [time.monotonic() + 2, time.monotonic() - 1]:
        with pytest.raises(RuntimeError, match="PX4_TRAINING_EXECUTOR_ENDING:" + phase):
            env._next_request(deadline=deadline)
    env._exchange.wait_request.assert_not_called()
    assert env._rollout_stop_reason["executor_phase"] == phase
    assert not env._quiesced and not (tmp_path / "rollout-stop-reason.json").exists()
    env._process = SimpleNamespace(poll=lambda: 2)
    with pytest.raises(FileNotFoundError):
        env.quiesce()
    module._write_new(
        tmp_path / "native-terminal-lifecycle.json",
        {"terminal_state": "ON_GROUND", "landing_confirmed": True, "safe_to_stop_watchdog": True},
    )
    failure = "UserDirectedLanding: MODEL_HOLD_REQUIRES_FRESH_NATIVE_HEADING"
    timing = {"status": "failed", "failure": failure}
    module._write_new(tmp_path / "offboard_timing.json", timing)
    env.quiesce()
    reason = json.loads((tmp_path / "rollout-stop-reason.json").read_text())
    assert reason["executor_outcome"]["failure"] == failure
    assert (
        reason["executor_outcome"]["timing_sha256"]
        == hashlib.sha256((tmp_path / "offboard_timing.json").read_bytes()).hexdigest()
    )
    assert reason["qualified_for_flight"] is False


# 功能：
#   验证正常等待超时仍为失败，并记录未放宽的原有接纳预算。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_true_input_timeout_is_still_failure_with_unchanged_admission_budget(tmp_path):
    env = environment(tmp_path)
    env.simulation = tmp_path
    env._exchange = SimpleNamespace(discard_pending=Mock())
    with pytest.raises(TimeoutError, match="PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT"):
        env._next_request(deadline=time.monotonic() - 1)
    assert env._rollout_stop_reason["reason"] == "next-observation-timeout"
    assert env._rollout_stop_reason["required_remaining_input_ms"] == (
        module.LOCAL_DISPATCH_RESERVE_MS
        + module.TRAINING_REPLY_PREPARATION_RESERVE_MS)


# 功能：
#   在接收过程中切换到降落阶段，验证刚到达的请求也不得继续控制。
# 输入：
#   tmp_path：独立阶段文件目录。
# 输出：
#   None：不返回业务数据。
def test_ending_during_receive_discards_previously_live_request(tmp_path):
    env = environment(tmp_path)
    env.simulation = tmp_path
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock())

    # 功能：
    #   接收期间写入降落阶段，再返回时间未过期的合成请求。
    # 输入：
    #   kwargs：等待器传入的接收参数，本夹具不等待网络。
    # 输出：
    #   request：用于验证结束阶段优先拒绝的请求。
    def receive(**kwargs):
        module._write_new(tmp_path / "runtime-phase.json", {"phase": "LANDING"})
        request = {"valid_until_unix_ms": int(time.time() * 1000) + 250}
        return request

    env._receive_source_request = receive
    with pytest.raises(RuntimeError, match="PX4_TRAINING_EXECUTOR_ENDING:LANDING"):
        env._next_request(deadline=time.monotonic() + 2)
    env._exchange.discard_pending.assert_called_once()


# 功能：
#   对比同一 RGB 字节的磁盘与实时张量，检查三种归一化完全一致。
# 输入：
#   tmp_path：独立 PNG 目录。
#   normalization：待检查的归一化方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("normalization", ["zero-to-one", "minus-one-to-one", "imagenet"])
def test_rgb_tensor_is_identical_between_recording_and_realtime(tmp_path, normalization):
    import numpy as np
    from PIL import Image

    from dronedream_agent_core.visual_control_input import forward_rgb_tensor

    values = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    path: Path = tmp_path / "image.png"
    Image.fromarray(values).save(path)
    kwargs = dict(width=64, height=64, normalization=normalization)
    disk = forward_rgb_tensor({"path": str(path)}, **kwargs)
    live = forward_rgb_tensor(
        {"model_rgb_bytes": values.tobytes(), "model_rgb_width": 64, "model_rgb_height": 64},
        **kwargs,
    )
    np.testing.assert_array_equal(disk, live)


# 功能：
#   验证非法权重摘要不能覆盖上一有效绑定。
# 输入：
#   tmp_path：独立测试目录。
#   identity：缺失、错误类型或非法十六进制摘要。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("identity", [None, 123, [], "", "G" * 64])
def test_invalid_weight_identity_keeps_previous_binding(tmp_path, identity):
    env = environment(tmp_path)
    with pytest.raises(ValueError, match="POLICY_IDENTITY_REQUIRES_QUIESCED_VALID_WEIGHTS"):
        env.bind_policy_identity(identity)
    assert env._policy_sha256 == "a" * 64
