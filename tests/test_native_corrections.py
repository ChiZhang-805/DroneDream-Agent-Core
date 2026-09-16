"""Synthetic native-receipt binding tests; no simulation is launched."""

import json
from dataclasses import replace

import pytest
from test_control_execution_evidence import evidence
from test_counterfactual_teacher import teacher
from test_training_observation_boundary import current_request
from test_training_runtime_evidence import pose

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.training.executed_control import executed_training_action
from dronedream_agent_core.training.flight_environment import (
    FlightObservation,
    FlightStep,
    PilotAction,
)
from dronedream_agent_core.training.native_corrections import GroundedNativeCorrections
from dronedream_agent_core.training.outcome_verifier import SimulationOutcomeVerifier
from dronedream_agent_core.training.policy_exchange import TrainingProposal


# 功能：
#   建立绑定当前特征、四轴回执、独立几何见证和停止标记的合成原生训练步骤。
# 输入：
#   tmp_path：测试独占回合根目录。
# 输出：
#   oracle：使用合成证据的离线纠正器。
#   observation：控制前的原始学生观测。
#   path：本步转移文件路径。
#   terminal：独立停止状态文件路径。
def fixture(tmp_path):
    request = current_request()
    observation = FlightObservation(
        mission_id="unit",
        episode_id="episode-unit",
        map_sha256="a" * 64,
        sequence=0,
        sample=LocalPolicyObservation.model_validate(request["observation"]),
    )
    following = observation.model_copy(deep=True)
    following.sequence = 1
    following.sample.temporal_evidence.observed_at_unix_ms = 1200
    following.sample.temporal_evidence.sample_sha256 = "b" * 64
    _, inputs = evidence()
    command = RuntimeLocalSafetyCommand.model_validate(inputs["command_records"][0]["command"])
    digest = observation.sample.source_snapshot_sha256
    command = command.model_copy(
        update={
            "model_navigation_snapshot_sha256": digest,
            "requested_control_intent": command.requested_control_intent.model_copy(
                update={"navigation_snapshot_sha256": digest}
            ),
        }
    )
    application = control_application_record(
        command,
        sequence=1,
        accepted_at_unix_ms=1100,
        transport="velocity-ned",
        velocity_ned_mps=(0.2, 0.1, -0.05),
        yaw_heading_deg=20,
        yaw_rate_application={
            "previous_heading_deg": 19.75,
            "clockwise_rate_dps": 5.0,
            "integration_seconds": 0.05,
        },
    )
    action = PilotAction(mode="pilot-control", axes=[0.25, -0.5, 0.125, 0.25])
    applied, _ = executed_training_action(
        request["snapshot"], command, application, limits=observation.sample.pilot_control_limits
    )
    evaluator = teacher(wall_x=8.0)
    verifier = SimulationOutcomeVerifier(evaluator.geometry.primitives, evaluator.envelope)
    goal = request["snapshot"]["goal_position_m"]
    reward, receipt = verifier.evaluate(
        # 合成两帧具有连续发布序号；独立设置窗口时刻，不用跳号来借用 pose 夹具的时钟。
        observations=[pose(1, 0).model_copy(update={"observed_at_unix_ms": 1000}),
                      pose(2, 0.04).model_copy(update={"observed_at_unix_ms": 1200})],
        start_ms=1000,
        end_ms=1200,
        stage_id="office",
        goal_revision=sha256_json(goal),
        goal_position_m=tuple(goal.values()),
        safety_intervened=True,
        commanded_change_squared=0.0,
        power_joules=None,
    )
    step = FlightStep(
        observation=following,
        proposed_action_sha256=sha256_json(action),
        applied_action=applied,
        applied_command_sha256=sha256_json(command),
        evidence=reward,
        terminated=False,
        truncated=False,
    )
    row = {
        "source_observation": observation.model_dump(mode="json"),
        "source_snapshot": request["snapshot"],
        "command": command.model_dump(mode="json"),
        "application": application.model_dump(mode="json"),
        "proposal": TrainingProposal(
            request_sha256=sha256_json(request),
            policy_sha256="c" * 64,
            expert_role="local-navigation-policy",
            action=action,
        ).model_dump(),
        "step": step.model_dump(mode="json"),
        "outcome_receipt": receipt,
    }
    episode = tmp_path / observation.episode_id
    terminal = episode / "flight/simulation/native-terminal-lifecycle.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_text(
        json.dumps(
            {
                "terminal_state": "ON_GROUND",
                "landing_confirmed": True,
                "safe_to_stop_watchdog": True,
            }
        )
    )
    (episode / "reset.json").write_text(
        json.dumps(
            {
                "policy_sha256": "c" * 64,
                "mission_id": "unit",
                "config": {"asset_sha256": {"semantic": "a" * 64}},
            }
        )
    )
    path = episode / "transition-000001.json"
    path.write_text(json.dumps(row))
    oracle = GroundedNativeCorrections(
        tmp_path, evaluator, map_sha256="a" * 64, student_policy_sha256="c" * 64
    )
    return oracle, observation, path, terminal


# 功能：
#   验证教师使用独立真值而不改写学生输入；缓存不能掩盖停止状态后来发生的变化。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_grounded_witness_is_label_only_and_cache_cannot_hide_landing_change(tmp_path):
    oracle, observation, _, terminal = fixture(tmp_path)
    before = sha256_json(observation)
    state = oracle._context(observation)
    assert state.position.x == 0.0  # privileged label pose, not the actor's map position 1
    assert sha256_json(observation) == before
    assert state.context_sha256 == oracle._context(observation).context_sha256
    correction = oracle.correction(observation)
    assert correction.observation_sha256 == before
    terminal.write_text(
        json.dumps(
            {"terminal_state": "IN_AIR", "landing_confirmed": False, "safe_to_stop_watchdog": False}
        )
    )
    with pytest.raises(ValueError, match="LANDING_NOT_CONFIRMED"):
        oracle.correction(observation)


# 功能：
#   验证同回合序号的文件改变会使缓存失效，错绑策略、输入、动作或见证一律拒绝。
# 输入：
#   tmp_path：合成回合根目录。
#   mutation：需要故意损坏的证据字段。
#   error：预期出现的拒绝代码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation,error",
    [
        ("weights", "STUDENT_WEIGHTS"),
        ("snapshot", "SNAPSHOT_HASH"),
        ("applied", "PROPOSAL_MISMATCH"),
        ("executed", "APPLIED_ACTION_MISMATCH"),
        ("witness", "OUTCOME_BINDING"),
    ],
)
def test_wrong_snapshot_policy_action_or_witness_is_not_a_training_label(tmp_path, mutation, error):
    oracle, observation, path, _ = fixture(tmp_path)
    oracle._context(observation)  # prove file contents, not just sequence, invalidate caching
    row = json.loads(path.read_text())
    if mutation == "weights":
        row["proposal"]["policy_sha256"] = "d" * 64
    elif mutation == "snapshot":
        row["source_snapshot"]["goal_position_m"]["x"] += 1.0
    elif mutation == "applied":
        row["step"]["proposed_action_sha256"] = "d" * 64
    elif mutation == "executed":
        row["step"]["applied_action"]["axes"][0] += .1
    else:
        row["outcome_receipt"]["observations"][0]["current_position_m"]["x"] += 1.0
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=error):
        oracle.correction(observation)


# 功能：
#   验证增加位置不确定度会保守减小净空下界，而非把外推真值当成精确现状。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_counterfactual_does_not_claim_old_unmodelled_pose_is_exact(tmp_path):
    oracle, observation, _, _ = fixture(tmp_path)
    state = oracle._context(observation)
    ordinary = oracle.teacher.evaluate(
        state,
        PilotAction(mode="pilot-control", axes=[0.1, 0, 0, 0]),
        observation.sample.pilot_control_limits,
    )
    uncertain = oracle.teacher.evaluate(
        replace(state, additional_position_uncertainty_m=0.2),
        PilotAction(mode="pilot-control", axes=[0.1, 0, 0, 0]),
        observation.sample.pilot_control_limits,
    )
    assert uncertain["clearance_lower_bound_m"] == pytest.approx(
        ordinary["clearance_lower_bound_m"] - 0.2
    )
