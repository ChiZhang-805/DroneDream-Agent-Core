"""Scheduling and evidence ownership fixtures; no physical simulator launched."""

import json
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_native_corrections import fixture
from test_offline_flight_learning import UnitEnvironment
from test_px4_training_lifecycle import environment

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.capture_archive import pack_native_transition
from dronedream_agent_core.training.flight_environment import (
    ControlPreparationExpired,
    FlightObservation,
    FlightStep,
    PilotAction,
)
from dronedream_agent_core.training.native_transition import (
    CapturedNativeTransition,
    evaluate_native_transition,
)
from dronedream_agent_core.training.outcome_verifier import SimulationOutcomeVerifier
from dronedream_agent_core.training.policy_exchange import TrainingProposal
from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment
from dronedream_agent_core.training.student_collection import (
    collect_native_observation_first_visits,
)


class CaptureLifecycle:
    simulation_only, evidence_kind = True, "px4-gazebo"

    # 功能：
    #   建立只记录调用顺序的合成环境，不启动原生仿真进程。
    # 输入：
    #   self：生命周期替身。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.inner, self.grounded, self.events = UnitEnvironment(), True, []
        self.count = 0

    # 功能：
    #   验证采集开始前在停止状态绑定策略身份。
    # 输入：
    #   self：生命周期替身。
    #   digest：策略摘要。
    # 输出：
    #   None：不返回业务数据。
    def bind_policy_identity(self, digest):
        assert self.grounded and len(digest) == 64

    # 功能：
    #   开始合成回合并记录起始观测，用状态标记模拟尚未停止的运行。
    # 输入：
    #   self：生命周期替身。
    #   seed：测试随机种子。
    # 输出：
    #   observation：本回合初始观测。
    def reset(self, *, seed):
        self.events.append("reset")
        self.grounded = False
        self.current = self.inner.reset(seed=seed)
        observation = self.current
        return observation

    # 功能：
    #   产生控制前后的合成观测，仅用于验证调度而不冒充完整原生证据。
    # 输入：
    #   self：处于运行状态的生命周期替身。
    #   action：合成环境接受的动作。
    # 输出：
    #   capture：控制前后观测和截断标记组成的测试对象。
    def capture_step(self, action):
        assert not self.grounded
        self.events.append("control")
        previous = self.current
        self.current = self.inner.step(action).observation
        self.count += 1
        capture = SimpleNamespace(
            source_observation=previous, observation=self.current, truncated=False
        )
        return capture

    # 功能：
    #   记录停止事件并切换替身状态，让测试约束先停止后评估的次序。
    # 输入：
    #   self：生命周期替身。
    # 输出：
    #   None：不返回业务数据。
    def quiesce(self):
        self.events.append("ground")
        self.grounded = True

    # 功能：
    #   原样保留调度替身的捕获；完整原生字节归档由独立用例验证。
    # 输入：
    #   self：尚处于运行状态的替身。
    #   capture：本步合成捕获。
    # 输出：
    #   capture：同一份合成捕获。
    def retain_capture(self, capture):
        # This fixture has no native actuator/witness contracts. Actual byte
        # retention and lossless validation are exercised separately below.
        assert not self.grounded
        return capture

    # 功能：
    #   仅在停止状态记录评估和持久化事件，保存已接受的合成捕获。
    # 输入：
    #   self：已停止的生命周期替身。
    #   captures：待最终化的捕获序列。
    # 输出：
    #   visits：保存的捕获列表。
    def finalize_captures(self, captures):
        assert self.grounded
        self.events.append("evaluate-and-persist")
        self.persisted = list(captures)
        visits = self.persisted
        return visits


# 功能：
#   以固定种子和策略身份驱动原生采集调度，默认学生只返回小幅前向动作。
# 输入：
#   env：采集环境或生命周期替身。
#   student：自定义学生回调，None 使用默认回调。
#   steps：采集步数。
#   kwargs：向生产采集入口传递的额外参数。
# 输出：
#   visits：停止后最终化的访问记录。
def collect(env, student=None, steps=3, **kwargs):
    visits = collect_native_observation_first_visits(
        env,
        student or (lambda _: PilotAction(mode="pilot-control", axes=[0.1, 0.0, 0.0, 0.0])),
        seed=805,
        steps=steps,
        held_out_missions=set(),
        student_policy_sha256="a" * 64,
        **kwargs,
    )
    return visits


# 功能：
#   验证所有学生控制先完成，确认停止后才计算独立奖励并持久化标签。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_decisions_precede_independent_rewards_and_grounding_precedes_all_labels():
    env = CaptureLifecycle()
    visits = collect(env)
    assert len(visits) == 3
    assert env.events == [
        "reset",
        "control",
        "control",
        "control",
        "ground",
        "evaluate-and-persist",
    ]


# 功能：
#   验证后续采集超时仍先停止环境，保留此前已完成的捕获并向上报告原超时。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_later_timeout_preserves_earlier_captures_after_native_quiescence():
    env = CaptureLifecycle()
    original = env.capture_step

    # 功能：
    #   首步执行真实测试替身，后续调用注入下一观测超时。
    # 输入：
    #   action：本次提交动作。
    # 输出：
    #   result：首次成功捕获。
    def capture(action):
        if env.count:
            raise TimeoutError("next source")
        result = original(action)
        return result

    env.capture_step = capture
    with pytest.raises(TimeoutError, match="next source"):
        collect(env)
    assert len(env.persisted) == 1
    assert env.events[-2:] == ["ground", "evaluate-and-persist"]


# 功能：
#   验证计时记录绑定当前观测序号，拒绝把上一轮模型计时计入新动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_collection_records_only_current_actor_timing():
    env = CaptureLifecycle()
    env.record_actor_timing = Mock()

    # 功能：
    #   返回悬停动作并附上与本次观测一致的合成准备及采样耗时。
    # 输入：
    #   observation：当前观测。
    # 输出：
    #   action：零轴悬停动作。
    def actor(observation):
        actor.last_timing = {"sequence": observation.sequence,
                             "input_preparation_ms": 1., "actor_sampling_ms": .5}
        action = PilotAction(mode="hold", axes=[0.] * 4)
        return action

    collect(env, actor)
    assert [c.kwargs["sequence"] for c in env.record_actor_timing.call_args_list] == [0, 1, 2]

    # 功能：
    #   返回动作但不更新旧计时元数据，用于复现跨观测的错误计时归属。
    # 输入：
    #   _：本次观测，在这个故障替身中不使用。
    # 输出：
    #   action：零轴悬停动作。
    def stale(_):
        action = PilotAction(mode="hold", axes=[0.] * 4)
        return action

    stale.last_timing = {"sequence": -1, "input_preparation_ms": 1., "actor_sampling_ms": .5}
    invalid = CaptureLifecycle()
    invalid.record_actor_timing = Mock()
    with pytest.raises(ValueError, match="TIMING_IDENTITY_MISMATCH"):
        collect(invalid, stale)


# 功能：
#   验证发送储备不足时丢弃提案并清理准备输入，不发送也不伪造已接受回执。
# 输入：
#   tmp_path：测试环境目录。
#   monkeypatch：时钟替换工具。
#   remaining_ms：模拟的剩余输入租期，单位毫秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("remaining_ms", [69, 5, 0, -100])
def test_native_actor_losing_dispatch_budget_is_not_sent_or_called_acknowledged(
    tmp_path, monkeypatch, remaining_ms
):
    env = environment(tmp_path)
    env._quiesced = False
    env._sequence = 0
    env.config.expert_role = "local-navigation-policy"
    env._observation = UnitEnvironment().reset(seed=1)
    env._pending = {"valid_until_unix_ms": 1000 + remaining_ms,
                    "snapshot": {"snapshot_sha256": "b" * 64}}
    env._prepared_input = object()
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock(), discard_pending=Mock())
    env._application = Mock()
    monkeypatch.setattr("dronedream_agent_core.training.px4_environment.time.time", lambda: 1.)
    with pytest.raises(ControlPreparationExpired, match="BUDGET_EXHAUSTED_BEFORE_SEND"):
        env.capture_step(PilotAction(mode="pilot-control", axes=[.2, 0., 0., 0.]))
    env._exchange.reply.assert_not_called()
    env._application.assert_not_called()
    env._exchange.discard_pending.assert_called_once()
    assert env._pending is env._prepared_input is None
    assert env._interface_timings[-1]["remaining_input_lease_ms"] == remaining_ms
    assert env._interface_timings[-1]["source_snapshot_sha256"] == "b" * 64
    assert env._interface_timings[-1]["not_submitted"]
    assert env._proposal_preparation_expirations == 1


# 功能：
#   验证学生回调改写观测会导致拒绝，环境停止且不发送该动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutating_actor_cannot_replace_original_environment_observation():
    env = CaptureLifecycle()

    # 功能：
    #   故意修改输入特征以验证采集边界对观测完整性的保护。
    # 输入：
    #   observation：被故障回调改写的观测。
    # 输出：
    #   action：不应被执行的悬停动作。
    def student(observation):
        observation.sample.state_features[0] += 1.0
        action = PilotAction(mode="hold", axes=[0.0] * 4)
        return action

    with pytest.raises(ValueError, match="MUTATED_SOURCE_OBSERVATION"):
        collect(env, student)
    assert env.events == ["reset", "ground"]


# 功能：
#   验证显式终止后只保留已有一步，不为填满采集配额再次开始回合。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_abort_does_not_take_off_again_to_fill_the_collection():
    env = CaptureLifecycle()
    assert len(collect(env, lambda _: PilotAction(mode="abort", axes=[0.0] * 4))) == 1
    assert env.events.count("reset") == 1


# 功能：
#   验证停止确认失败时最终化门槛仍阻止独立奖励和标签生成。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reward_oracle_never_runs_when_ground_confirmation_fails():
    env = CaptureLifecycle()
    env.quiesce = Mock(side_effect=TimeoutError("grounding failed"))
    with pytest.raises(AssertionError):  # fake finalizer enforces native quiescence
        collect(env)
    assert "evaluate-and-persist" not in env.events


# 功能：
#   验证即时和落地后评估产生同一结果，终止优先于截断，ENU 顺序不依赖 JSON 键顺序。
# 输入：
#   tmp_path：捕获与持久化测试目录。
#   terminal：注入的独立结果终止分类，None 为普通步骤。
#   coordinate_order：输入目标对象的键排列顺序。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("terminal", [None, "collision", "geofence_violation", "loss_of_control",
                                     "premature_landing", "verified_mission_complete"])
@pytest.mark.parametrize("coordinate_order", [("x", "y", "z"), ("z", "x", "y")])
def test_immediate_and_grounded_evaluation_share_exact_native_outcome(
    tmp_path, terminal, coordinate_order,
):
    oracle, _, path, _ = fixture(tmp_path)
    row = json.loads(path.read_bytes())
    original_step = FlightStep.model_validate(row["step"])
    # The source-binding fixture omits the runner's task stage. This pure
    # evaluator fixture supplies it explicitly; it does not launch/admit a run.
    row["source_snapshot"]["strategic_context"]["task"]["navigation_goal_id"] = "office"
    goal = {"x": 1.25, "y": -.75, "z": 2.25}
    row["source_snapshot"]["goal_position_m"] = {axis: goal[axis] for axis in coordinate_order}
    # 此测试有意改变任务目标，因此连同相关身份重新绑定，而不是绕过生产摘要门槛。
    row["source_snapshot"].pop("snapshot_sha256")
    digest = row["source_snapshot"]["snapshot_sha256"] = sha256_json(row["source_snapshot"])
    row["source_observation"]["sample"]["source_snapshot_sha256"] = digest
    row["command"]["model_navigation_snapshot_sha256"] = digest
    row["command"]["requested_control_intent"]["navigation_snapshot_sha256"] = digest
    command = RuntimeLocalSafetyCommand.model_validate(row["command"])
    row["application"]["command_sha256"] = sha256_json(command)
    row["application"]["intent"]["navigation_snapshot_sha256"] = digest
    capture = CapturedNativeTransition(
        FlightObservation.model_validate(row["source_observation"]),
        original_step.observation,
        row["source_snapshot"],
        TrainingProposal.model_validate(row["proposal"]),
        command,
        ControlApplicationRecord.model_validate(row["application"]),
        None,
        original_step.applied_action,
        False,
        tuple(
            RuntimeLocalSafetyObservation.model_validate(r)
            for r in row["outcome_receipt"]["observations"]
        ),
        False,
    )
    verifier = SimulationOutcomeVerifier(
        oracle.teacher.geometry.primitives, oracle.teacher.envelope
    )
    if terminal is not None:
        # Unit-injected outcomes exercise every terminal classification; these
        # flags are not physical-flight evidence or altered persisted receipts.
        original_evaluate = verifier.evaluate

        # 功能：
        #   在真实几何评估结果上注入单一终止标记，仅检验终止分支而不改变磁盘见证。
        # 输入：
        #   kwargs：传给原评估器的参数。
        # 输出：
        #   evidence：带指定终止分类的测试结果。
        #   receipt：原评估器产生的独立回执。
        def terminal_evaluate(**kwargs):
            evidence, receipt = original_evaluate(**kwargs)
            evidence = evidence.model_copy(update={terminal: True})
            return evidence, receipt

        verifier.evaluate = terminal_evaluate
        capture = replace(capture, truncated=True)
    step, record = evaluate_native_transition(capture, verifier)
    assert record["outcome_receipt"]["goal_position_m"] == [goal[axis] for axis in ("x", "y", "z")]
    assert step.terminated == (terminal is not None)
    assert not step.truncated
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._quiesced, env._process = True, None
    env.episode_path = tmp_path / "deferred" / "episode-unit"
    env.episode_path.mkdir(parents=True)
    env.verifier = verifier
    env._interface_timings = deque(maxlen=512)
    packed = env.retain_capture(capture)
    assert packed == pack_native_transition(capture)
    assert env._interface_timings[-1]["stage"] == "capture-retained"
    visits = env.finalize_captures([packed])
    assert visits[0].step == step
    persisted = json.loads((env.episode_path / "transition-000001.json").read_bytes())
    assert persisted.pop("outcome_evaluation_phase") == "after-confirmed-native-landing"
    assert persisted == record
    env._quiesced = False
    with pytest.raises(ValueError, match="CONFIRMED_QUIESCENCE"):
        env.finalize_captures([packed])


# 功能：
#   验证接收期间到达回执时，仅保留晚于实际动作且仍有准备预算的下一观测。
# 输入：
#   tmp_path：测试环境目录。
#   monkeypatch：固定墙钟的替换工具。
#   observed：下一观测采集时刻。
#   remaining：下一观测剩余有效毫秒。
#   kept：是否应保留该观测。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "observed,remaining,kept", [(1100, 200, True), (1099, 200, False), (1100, 119, False)]
)
def test_ack_arriving_during_receive_keeps_only_fresh_post_action_request(
    tmp_path,
    monkeypatch,
    observed,
    remaining,
    kept,
):
    env = environment(tmp_path)
    request = {
        "observation": {"temporal_evidence": {"observed_at_unix_ms": observed}},
        "valid_until_unix_ms": 1000 + remaining,
    }
    receipt = (object(), SimpleNamespace(accepted_at_unix_ms=1099))
    env._receipt_monitor = SimpleNamespace(poll=Mock(side_effect=[None, receipt]))
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock())
    env._receive_source_request = Mock(return_value=request)
    monkeypatch.setattr("dronedream_agent_core.training.px4_environment.time.time", lambda: 1.0)
    assert env._application("call", deadline=time.monotonic() + 1) == receipt
    assert env._exchange.discard_pending.call_count == int(not kept)
    if kept:
        assert env._next_request(deadline=time.monotonic() + 1) is request
        assert env._receive_source_request.call_count == 1
        assert env._following_request is None


# 功能：
#   验证曾保留的观测到模型接纳时仍要重查预算，过期后丢弃并读取新观测。
# 输入：
#   tmp_path：测试环境目录。
#   monkeypatch：固定墙钟的替换工具。
# 输出：
#   None：不返回业务数据。
def test_kept_request_is_rechecked_if_actor_admission_later_loses_budget(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env._following_request = {"valid_until_unix_ms": 1001}
    replacement = {"valid_until_unix_ms": 1200}
    env._process = SimpleNamespace(poll=lambda: None)
    env._exchange = SimpleNamespace(discard_pending=Mock())
    env._receive_source_request = Mock(return_value=replacement)
    monkeypatch.setattr("dronedream_agent_core.training.px4_environment.time.time", lambda: 1.0)
    assert env._next_request(deadline=time.monotonic() + 1) is replacement
    assert env._exchange.discard_pending.call_count == 1
