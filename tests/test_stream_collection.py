"""Synthetic scheduling/ledger fixtures; no aircraft or API calls."""

import json
from collections import deque
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from clock_fixtures import isolate_time
from test_native_action_risk_artifacts import native_episode
from test_offline_flight_learning import UnitEnvironment
from test_px4_training_lifecycle import environment
from test_runtime_evidence_inventory import receipt
from test_training_observation_boundary import current_request
from test_training_runtime_evidence import pose

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_evidence import (
    control_evidence_inventory,
    navigation_evidence_inventory,
)
from dronedream_agent_core.training import px4_environment
from dronedream_agent_core.training.dagger_artifacts import (
    load_dagger_training_artifacts,
    write_rows,
    write_training_artifacts,
)
from dronedream_agent_core.training.flight_environment import FlightObservation, PilotAction
from dronedream_agent_core.training.policy_exchange import TrainingProposal
from dronedream_agent_core.training.px4_environment import _write_new
from dronedream_agent_core.training.runtime_evidence import IndependentPoseMonitor
from dronedream_agent_core.training.stream_capture import (
    StreamActionCapture,
    finalize_stream_captures,
    pack_stream_capture,
    stream_capture_from_record,
)
from dronedream_agent_core.training.stream_collection import (
    collect_native_control_stream,
    label_stream_visits,
)
from dronedream_agent_core.training.stream_episode import load_grounded_stream_episode


class StreamLifecycle:
    simulation_only, evidence_kind = True, "px4-gazebo"

    # 功能：
    #   创建无物理引擎的控制流环境，记录事件顺序与模拟落地状态。
    # 输入：
    #   self：调度测试环境。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.inner, self.events, self.grounded = UnitEnvironment(), [], True

    # 功能：
    #   断言绑定策略发生在模拟地面状态且提供完整摘要。
    # 输入：
    #   self：测试环境。
    #   digest：待绑定策略摘要。
    # 输出：
    #   None：不返回业务数据。
    def bind_policy_identity(self, digest):
        assert self.grounded and len(digest) == 64

    # 功能：
    #   记录启动事件并返回带回合身份的初始合成观测。
    # 输入：
    #   self：测试环境。
    #   seed：本轮随机种子。
    # 输出：
    #   observation：初始观测。
    def reset(self, *, seed):
        self.grounded = False
        self.events.append("reset")
        observation = self.inner.reset(seed=seed)
        return observation

    # 功能：
    #   记录不等待前次结果的提交事件，不实际发送任何飞控命令。
    # 输入：
    #   self：测试环境。
    #   action：学生提案。
    # 输出：
    #   None：不返回业务数据。
    def submit_stream_action(self, action):
        assert not self.grounded
        self.events.append("submit-without-wait")

    # 功能：
    #   推进合成传感器来源，模拟独立于动作回执的新观测到达。
    # 输入：
    #   self：测试环境。
    #   deadline：接口传入的等待期限，本简单夹具不模拟耗时。
    # 输出：
    #   observation：序号递增的合成观测。
    def next_stream_observation(self, *, deadline=None):
        self.events.append("new-source")
        self.inner.sequence += 1
        observation = self.inner.observation()
        return observation

    # 功能：
    #   记录合成落地事件，为离线结算提供顺序断言。
    # 输入：
    #   self：测试环境。
    # 输出：
    #   None：不返回业务数据。
    def quiesce(self):
        self.events.append("ground")
        self.grounded = True

    # 功能：
    #   只在模拟落地后记录结算事件；无实际控制器时不制造执行证据。
    # 输入：
    #   self：测试环境。
    # 输出：
    #   visits：空访问列表。
    def finalize_stream_captures(self):
        if not self.grounded:
            raise ValueError("CONFIRMED_QUIESCENCE_REQUIRED")
        self.events.append("join-actual-receipts")
        visits = []  # 调度夹具没有实际控制器，不制造已执行标签。
        return visits


# 功能：
#   用固定身份参数调用连续采集器，默认策略只输出合成前向幅度。
# 输入：
#   env：调度夹具。
#   student：可选自定义策略回调。
#   steps：最大提交数量。
# 输出：
#   visits：采集器返回的落地后访问记录。
def collect(env, student=None, steps=3):
    visits = collect_native_control_stream(
        env,
        student or (lambda _: PilotAction(mode="pilot-control", axes=[0.2, 0.0, 0.0, 0.0])),
        seed=805,
        steps=steps,
        held_out_missions=set(),
        student_policy_sha256="a" * 64,
    )
    return visits


# 功能：
#   验证新输入与提案交替推进，最终才落地并关联实际回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_new_sources_are_served_without_waiting_for_action_outcome():
    env = StreamLifecycle()
    assert collect(env) == []
    assert env.events == [
        "reset",
        "submit-without-wait",
        "new-source",
        "submit-without-wait",
        "new-source",
        "submit-without-wait",
        "ground",
        "join-actual-receipts",
    ]


# 功能：
#   验证后续输入超时仍先落地再结算先前提案，原超时向外传播。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_later_input_timeout_preserves_proposals_only_after_landing():
    env = StreamLifecycle()
    env.next_stream_observation = Mock(side_effect=TimeoutError("input"))
    with pytest.raises(TimeoutError, match="input"):
        collect(env)
    assert env.events == ["reset", "submit-without-wait", "ground", "join-actual-receipts"]


# 功能：
#   验证未确认落地时结算拒绝生成训练标签。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_landing_blocks_all_training_labels():
    env = StreamLifecycle()
    env.quiesce = Mock(side_effect=TimeoutError("not landed"))
    with pytest.raises(ValueError, match="CONFIRMED_QUIESCENCE"):
        collect(env)
    assert "join-actual-receipts" not in env.events


# 功能：
#   验证源观测被改动时禁止提交，终止提案只提交一次且不重启回合。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_input_is_not_submitted_and_abort_never_restarts():
    # 功能：
    #   制造策略篡改源特征的错误，再返回表面合法的保持动作。
    # 输入：
    #   observation：策略收到的原始观测。
    # 输出：
    #   proposal：应被采集边界拒绝的零轴提案。
    def corrupt(observation):
        observation.sample.state_features[0] += 1
        proposal = PilotAction(mode="hold", axes=[0.0] * 4)
        return proposal

    env = StreamLifecycle()
    with pytest.raises(ValueError, match="MUTATED_SOURCE"):
        collect(env, corrupt)
    assert env.events == ["reset", "ground"]
    aborted = StreamLifecycle()
    collect(aborted, lambda _: PilotAction(mode="abort", axes=[0.0] * 4))
    assert aborted.events == ["reset", "submit-without-wait", "ground", "join-actual-receipts"]


# 功能：
#   验证独立见证只选取健康、未过期的过去观测，并返回独立对象。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_witness_is_past_healthy_owned_label_only():
    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = Lock(), deque([pose(1, 0), pose(3, 1)]), None
    witness = monitor.initial_witness(1100)
    assert witness.observed_at_unix_ms == 1050
    witness.current_position_m.x = 88
    assert monitor._rows[0].current_position_m.x == 0
    with pytest.raises(ValueError, match="INITIAL_WITNESS"):
        monitor.initial_witness(1049)
    with pytest.raises(ValueError, match="INITIAL_WITNESS"):
        monitor.initial_witness(1251)


# 功能：
#   创建关联提案、命令与执行回执的离线夹具，所有数据均为合成测试证据。
# 输入：
#   root：独立测试目录。
#   accepted：是否写入实际接纳格式的回执行。
# 输出：
#   result：回合路径、教师配置、原始捕获及不可变捕获的四元组。
def stream_fixture(root, *, accepted=True):
    episode, teacher_config = native_episode(root)
    row = json.loads((episode / "transition-000001.json").read_bytes())
    (episode / "transition-000001.json").rename(episode / "fixture-source.json")
    reset_path = episode / "reset.json"
    reset = json.loads(reset_path.read_bytes())
    reset.update(simulation_only=True, qualification_granted=False)
    reset_path.write_text(json.dumps(reset))
    request = current_request()
    proposal = TrainingProposal.model_validate(row["proposal"])
    capture = StreamActionCapture(
        FlightObservation.model_validate(row["source_observation"]),
        request,
        proposal,
        RuntimeLocalSafetyObservation.model_validate(row["outcome_receipt"]["observations"][0]),
    )
    packed = pack_stream_capture(capture)
    command_payload = row["command"]
    command_payload["model_call_id"] = "model-" + sha256_json(proposal)[:24]
    command_payload["requested_control_intent"]["model_call_id"] = command_payload["model_call_id"]
    command_payload["requested_control_intent"].update(
        forward_velocity_mps=0.1, right_velocity_mps=-0.2, up_velocity_mps=0.05
    )
    command = RuntimeLocalSafetyCommand.model_validate(command_payload)
    application = ControlApplicationRecord.model_validate(row["application"]).model_copy(
        update={"command_sha256": sha256_json(command), "intent": command.requested_control_intent}
    )
    simulation = episode / "flight/simulation"
    (simulation / "runtime-state").mkdir()
    write_rows(
        simulation / "depth-local-safety-history.jsonl",
        [{"command": command.model_dump(mode="json")}],
    )
    write_rows(
        simulation / "runtime-state/control-applications.jsonl",
        [application.model_dump(mode="json")] if accepted else [],
    )
    write_rows(
        simulation / "runtime-state/local-safety-executor-history.jsonl", [{"source": "unit"}]
    )
    _write_new(
        simulation / "runtime-state/control-application-writer.json",
        receipt(control_evidence_inventory(simulation)),
    )
    _write_new(
        simulation / "runtime-evidence-writer-summary.json",
        receipt(navigation_evidence_inventory(simulation)),
    )
    result = episode, teacher_config, capture, packed
    return result


# 功能：
#   验证捕获独立性、完整账本读回、离线标签导出及导出后源文件篡改检测。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_immutable_capture_full_ledger_readback_and_offline_training_roundtrip(tmp_path):
    episode, config, capture, packed = stream_fixture(tmp_path)
    assert packed.unpack() == capture
    capture.request["multimodal"].append({"tamper": True})
    assert not packed.unpack().request["multimodal"]
    visits = finalize_stream_captures(episode, [packed], write_new=_write_new)
    assert len(visits) == 1 and not hasattr(visits[0], "step")
    raw = json.loads((episode / "stream-action-000000.json").read_bytes())
    assert stream_capture_from_record(raw)[0] == packed
    loaded = load_grounded_stream_episode(episode, config)
    assert list(loaded.visits) == visits
    labels = label_stream_visits(
        visits, loaded.oracle.correction, loaded.oracle.risk, held_out_missions=set()
    )
    assert labels.records[0]["not_a_reward_transition"]
    output = tmp_path / "annotated"
    output.mkdir()
    hashes = write_training_artifacts(
        output, visits, labels, episode_groups={episode.name: loaded.mission_split}
    )
    write_rows(
        output / "annotation-receipt.jsonl",
        [{"purpose": "grounded-native-stream-annotation", "file_sha256": hashes}],
    )
    assert load_dagger_training_artifacts(output)[0] == labels.behavior_samples
    loaded.verify_sources(episode)
    (episode / "stream-action-000000.json").write_text("{}")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        loaded.verify_sources(episode)


# 功能：
#   验证没有执行接纳的提案保留在缺失列表中，不生成已执行动作文件。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_unaccepted_proposals_never_become_applied_labels(tmp_path):
    episode, _, _, packed = stream_fixture(tmp_path, accepted=False)
    assert finalize_stream_captures(episode, [packed], write_new=_write_new) == []
    summary = json.loads((episode / "stream-capture-receipt.json").read_bytes())
    assert summary["actually_accepted_decisions"] == 0 and len(summary["unaccepted_proposals"]) == 1
    assert not list(episode.glob("stream-action-*.json"))


# 功能：
#   验证截断账本、未排空写入器、未确认落地或捕获摘要错误均阻止成功回执。
# 输入：
#   tmp_path：本例独立目录。
#   fault：注入的证据故障类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["truncated-ledger", "not-drained", "landing", "bad-capture"])
def test_faults_cannot_grant_stream_control_evidence(tmp_path, fault):
    episode, _, _, packed = stream_fixture(tmp_path)
    simulation = episode / "flight/simulation"
    if fault == "truncated-ledger":
        with (simulation / "runtime-state/control-applications.jsonl").open("ab") as handle:
            handle.write(b"{")
    elif fault == "not-drained":
        (simulation / "runtime-state/control-application-writer.json").write_text("{}")
    elif fault == "landing":
        (simulation / "native-terminal-lifecycle.json").write_text("{}")
    else:
        packed = replace(packed, sha256="0" * 64)
    with pytest.raises(ValueError):
        finalize_stream_captures(episode, [packed], write_new=_write_new)
    assert not (episode / "stream-capture-receipt.json").exists()


# 功能：
#   验证适配器提交提案时不等待执行回执，且不可中途切换为奖励步进模式。
# 输入：
#   tmp_path：本例独立目录。
#   monkeypatch：替换当前时刻以保持提案有效的工具。
# 输出：
#   None：不返回业务数据。
def test_stream_submission_never_waits_for_previous_application(tmp_path, monkeypatch):
    _, _, capture, _ = stream_fixture(tmp_path)
    env = environment(tmp_path)
    env._quiesced, env._sequence, env._retained_capture_bytes = False, 0, 0
    env.config.expert_role = "local-navigation-policy"
    env.config.episode_steps = 3
    env._policy_sha256 = capture.proposal.policy_sha256
    env._stream_captures = []
    env._pending, env._observation = capture.request, capture.observation
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock(), discard_pending=Mock())
    env._monitor = SimpleNamespace(initial_witness=Mock(return_value=capture.initial_witness))
    env._application = Mock(side_effect=AssertionError("must not wait for application"))
    isolate_time(monkeypatch, px4_environment, time=lambda: 1.05)
    env.submit_stream_action(capture.proposal.action)
    assert env._sequence == 1 and len(env._stream_captures) == 1
    env._exchange.reply.assert_called_once()
    env._application.assert_not_called()
    assert env._pending is None
    with pytest.raises(ValueError, match="MODE_CHANGED"):
        env.capture_step(capture.proposal.action)


# 功能：
#   验证几何教师拒绝错误的实际动作延迟，不用无效时间推算制动空间。
# 输入：
#   latency：待拒绝的延迟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("latency", [-0.1, 0.251, True, float("nan"), float("inf")])
def test_nominal_teacher_rejects_invalid_observed_latency(latency):
    from test_counterfactual_teacher import action, state, teacher

    from dronedream_agent_core.pilot_control_mapping import PilotControlLimits

    with pytest.raises(ValueError, match="OBSERVED_LATENCY"):
        teacher().evaluate(
            replace(state(), observed_action_latency_seconds=latency),
            action(),
            PilotControlLimits(1.0, 1.0, 20.0),
        )


# 功能：
#   验证实测延迟比配置更慢时扩大预测时间窗并收紧障碍余量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_measured_latency_cannot_be_replaced_by_faster_configured_delay():
    from test_counterfactual_teacher import action, state, teacher

    from dronedream_agent_core.pilot_control_mapping import PilotControlLimits

    evaluator, limits = teacher(wall_x=1.05), PilotControlLimits(1.0, 1.0, 20.0)
    original = evaluator.evaluate(state(), action(-0.5), limits)
    delayed = evaluator.evaluate(
        replace(state(), observed_action_latency_seconds=0.2), action(-0.5), limits
    )
    assert original["effective_latency_seconds"] == 0.07
    assert delayed["effective_latency_seconds"] == 0.2
    assert delayed["clearance_lower_bound_m"] < original["clearance_lower_bound_m"]
    assert delayed["times_seconds"][-1] > original["times_seconds"][-1]


# 功能：
#   验证独立安全保持不会冒充模型运动，而在训练记录中保留干预标记。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_safety_hold_without_model_motion_authority_is_retained_as_intervention(tmp_path):
    from dronedream_agent_core.contracts import Vector3
    from dronedream_agent_core.control_execution_evidence import control_application_record
    from dronedream_agent_core.training.stream_capture import stream_visit

    episode, _, capture, packed = stream_fixture(tmp_path)
    row = json.loads((episode / "flight/simulation/depth-local-safety-history.jsonl").read_bytes())
    command = RuntimeLocalSafetyCommand.model_validate(row["command"])
    command = command.model_copy(
        update={
            "model_navigation_authorized": False,
            "requested_control_intent": None,
            "model_navigation_snapshot_sha256": None,
            "model_path_sha256": None,
            "decision": command.decision.model_copy(
                update={
                    "action": "hold",
                    "control_source": "deterministic-brake",
                    "selected_velocity_mps": Vector3(x=0, y=0, z=0),
                    "selected_yaw_rate_dps": 0.0,
                }
            ),
        }
    )
    command = RuntimeLocalSafetyCommand.model_validate(command.model_dump())
    application = control_application_record(
        command,
        sequence=1,
        accepted_at_unix_ms=1100,
        transport="position-velocity-ned",
        velocity_ned_mps=(0.0, 0.0, 0.0),
        position_ned_m=(1.0, 2.0, 3.0),
        yaw_heading_deg=20.0,
    )
    visit = stream_visit(capture, packed, command, application)
    assert visit.applied_action.mode == "hold" and visit.safety_intervened


# 功能：
#   验证命令不能沿用同一提案身份却改成另一组物理轴值。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_intent_cannot_borrow_proposal_identity_for_other_axis_values(tmp_path):
    from dronedream_agent_core.training.stream_capture import grounded_control_index, stream_visit

    episode, _, capture, packed = stream_fixture(tmp_path)
    call_id = "model-" + sha256_json(capture.proposal)[:24]
    command, application = grounded_control_index(episode / "flight/simulation", {call_id})[call_id]
    command.requested_control_intent.forward_velocity_mps += 0.01
    with pytest.raises(ValueError, match="PHYSICAL_REQUEST_MISMATCH"):
        stream_visit(capture, packed, command, application)


# 功能：
#   验证连续流跳过同时间戳输入后即可采用新来源，无须先看到上一动作回执。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_new_input_before_previous_action_acceptance_is_available_to_stream_actor(tmp_path):
    env = environment(tmp_path)
    env._collection_mode = "stream-imitation"
    env._observation = SimpleNamespace(
        sample=SimpleNamespace(temporal_evidence=SimpleNamespace(observed_at_unix_ms=1000))
    )
    same = {"observation": {"temporal_evidence": {"observed_at_unix_ms": 1000}}}
    new = {"observation": {"temporal_evidence": {"observed_at_unix_ms": 1050}}}
    env._next_request, env._adopt = Mock(side_effect=[same, new]), Mock(return_value="new-input")
    env._application = Mock(side_effect=AssertionError("receipt not yet available"))
    env._record_rejected_input = Mock()
    env._exchange = SimpleNamespace(discard_pending=Mock())
    assert env.next_stream_observation() == "new-input"
    env._adopt.assert_called_once_with(new)
    env._application.assert_not_called()
    env._exchange.discard_pending.assert_called_once()


# 功能：
#   验证包含视觉载荷的流记录必须有实际绑定的视觉编码器，不能只相信已存特征。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_stream_artifacts_cannot_bypass_visual_encoder_validation(tmp_path):
    from dronedream_agent_core.training.stream_episode import validate_stream_visual

    _, _, capture, _ = stream_fixture(tmp_path)
    validate_stream_visual(capture, None)
    capture.request["multimodal"] = [{"not": "real-rgb"}]
    with pytest.raises(ValueError, match="VISUAL_ENCODER_MISSING"):
        validate_stream_visual(capture, None)


# 功能：
#   验证离线视觉编码器在成功、来源绑定失败和业务异常三条路径都关闭。
# 输入：
#   monkeypatch：编码器工厂替换器。
#   failure：需要注入的失败位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["none", "binding", "body"])
def test_offline_visual_encoder_is_closed_on_success_and_failure(monkeypatch, failure):
    from dronedream_agent_core.training import stream_episode as module

    encoder = SimpleNamespace(sha256="a" * 64, input_contract={"test": "fixture"}, close=Mock())
    monkeypatch.setattr(module, "FrozenVisualEncoder", lambda _: encoder)
    reset = {"visual_encoder_sha256": "b" * 64 if failure == "binding" else encoder.sha256,
             "visual_input_contract": encoder.input_contract}

    # 功能：
    #   进入绑定编码器上下文，并按配置模拟离线处理错误。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def validate():
        with module._bound_visual_encoder(SimpleNamespace(visual_package="fixture", visual_encoder=None), reset):
            if failure == "body":
                raise ValueError("fixture read failure")

    if failure == "none":
        validate()
    else:
        with pytest.raises(ValueError):
            validate()
    encoder.close.assert_called_once()


# 功能：
#   验证教师上下文使用来源到实际接纳的延迟，并保留执行回执摘要。
# 输入：
#   tmp_path：本例独立目录。
# 输出：
#   None：不返回业务数据。
def test_native_oracle_uses_source_to_actual_acceptance_latency(tmp_path):
    from test_native_corrections import fixture

    oracle, observation, _, _ = fixture(tmp_path)
    state = oracle._context(observation)
    assert state.observed_action_latency_seconds == 0.1
    assert oracle.receipts[-1]["context"]["application_sha256"]


# 功能：
#   通过实际命令行入口验证行为标注与动作风险数据集共同使用落地验证后的来源。
# 输入：
#   tmp_path：合成回合、配置及导出目录。
#   monkeypatch：命令行参数替换器。
# 输出：
#   None：不返回业务数据。
def test_stream_cli_annotation_and_action_conditioned_risk_share_grounded_sources(
    tmp_path, monkeypatch
):
    import runpy
    import sys
    from pathlib import Path

    from dronedream_agent_core.training.action_risk_artifacts import load_action_risk_dataset

    episode, config, _, packed = stream_fixture(tmp_path)
    finalize_stream_captures(episode, [packed], write_new=_write_new)
    teacher = tmp_path / "teacher.json"
    teacher.write_text(config.model_copy(update={
        'risk_label_semantics': 'observation-clearance-v2'}).model_dump_json())
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for filename, output in (
        ("annotate_local_policy_dagger.py", "behavior"),
        ("build_native_action_risk_dataset.py", "risk"),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                filename,
                "--episode",
                str(episode),
                "--teacher-config",
                str(teacher),
                "--output",
                str(tmp_path / output),
            ],
        )
        assert runpy.run_path(str(scripts / filename))["main"]() == 0
    assert len(load_dagger_training_artifacts(tmp_path / "behavior")[0]) == 1
    assert len(load_action_risk_dataset(tmp_path / "risk").samples) == 161
