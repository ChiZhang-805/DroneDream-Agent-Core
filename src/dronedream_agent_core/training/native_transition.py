"""Independent native outcome evaluation, separable from the next actor deadline.

Captures contain observations and actual actuator receipts, not rewards guessed
from a policy proposal. The same evaluator serves immediate PPO steps and
post-landing DAgger collection. It has no control or optimizer access.
"""

from dataclasses import dataclass, replace

from dronedream_plugin_sdk.protocol import copy_json

from ..contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from ..control_execution_evidence import ControlApplicationRecord
from ..hashing import sha256_json
from .executed_control import (
    NATIVE_ACTION_COORDINATE_FRAME,
    executed_training_action,
    transport_command_change_squared,
)
from .flight_environment import FlightObservation, FlightStep, PilotAction, SafetyPositionControl
from .outcome_verifier import SimulationOutcomeVerifier
from .policy_exchange import TrainingProposal


@dataclass(frozen=True)
class CapturedNativeTransition:
    """One source/action/receipt/outcome join; no inferred execution or reward."""
    source_observation: FlightObservation
    observation: FlightObservation
    snapshot: dict
    proposal: TrainingProposal
    command: RuntimeLocalSafetyCommand
    application: ControlApplicationRecord
    previous_application: ControlApplicationRecord | None
    applied: PilotAction | SafetyPositionControl
    deadline_intervened: bool
    witnesses: tuple[RuntimeLocalSafetyObservation, ...]
    truncated: bool


# 功能：
#   1. 重新验证并冻结单步证据，从实际回执恢复动作，与捕获声明交叉核对。
#   2. 使用独立仿真见证形成结果、终止或截断及完整记录，不发送控制或训练权重。
# 输入：
#   capture：原始观测、提案、回执、实际动作与独立见证组成的捕获。
#   verifier：独立几何和任务结果评估器。
# 输出：
#   step：与原观测及提案绑定的飞行步骤。
#   record：独立持有内容的训练记录。
def evaluate_native_transition(capture: CapturedNativeTransition,
                               verifier: SimulationOutcomeVerifier):
    if type(capture) is not CapturedNativeTransition:
        raise ValueError("NATIVE_CAPTURE_TYPE_INVALID")
    if type(capture.deadline_intervened) is not bool or type(capture.truncated) is not bool:
        raise ValueError("NATIVE_CAPTURE_FLAGS_INVALID")
    # 冻结 dataclass 并不能保护其中的模型和字典；评估和返回记录使用同一份独立副本。
    capture = replace(
        capture,
        source_observation=FlightObservation.model_validate(capture.source_observation.model_dump()),
        observation=FlightObservation.model_validate(capture.observation.model_dump()),
        snapshot=copy_json(capture.snapshot),
        proposal=TrainingProposal.model_validate(capture.proposal.model_dump()),
        command=RuntimeLocalSafetyCommand.model_validate(capture.command.model_dump()),
        application=ControlApplicationRecord.model_validate(capture.application.model_dump()),
        previous_application=(None if capture.previous_application is None else
                              ControlApplicationRecord.model_validate(
                                  capture.previous_application.model_dump())),
        applied=(SafetyPositionControl if isinstance(capture.applied, SafetyPositionControl)
                 else PilotAction).model_validate(capture.applied.model_dump()),
        witnesses=tuple(RuntimeLocalSafetyObservation.model_validate(row.model_dump())
                        for row in capture.witnesses),
    )
    previous, observation = capture.source_observation, capture.observation
    action, applied = capture.proposal.action, capture.applied
    content = dict(capture.snapshot)
    digest = content.pop("snapshot_sha256", None)
    if digest != sha256_json(content) or digest != previous.sample.source_snapshot_sha256:
        raise ValueError("NATIVE_CAPTURE_SOURCE_SNAPSHOT_MISMATCH")
    recovered, expired = executed_training_action(capture.snapshot, capture.command,
                                                 capture.application,
                                                 limits=previous.sample.pilot_control_limits)
    if recovered != applied:
        raise ValueError("NATIVE_CAPTURE_APPLIED_ACTION_MISMATCH")
    if expired != capture.deadline_intervened:
        raise ValueError("NATIVE_CAPTURE_DEADLINE_INTERVENTION_MISMATCH")
    start = previous.sample.temporal_evidence.observed_at_unix_ms
    end = observation.sample.temporal_evidence.observed_at_unix_ms
    task = capture.snapshot["strategic_context"]["task"]
    changed = (
        capture.deadline_intervened
        or isinstance(applied, SafetyPositionControl)
        or action.mode != applied.mode
        or any(abs(a - b) > 1e-6 for a, b in zip(action.axes, applied.axes, strict=True))
    )
    evidence, receipt = verifier.evaluate(
        observations=list(capture.witnesses),
        start_ms=start,
        end_ms=end,
        stage_id=task["navigation_goal_id"],
        goal_revision=sha256_json(capture.snapshot["goal_position_m"]),
        # JSON object key order is not a coordinate convention. Always extract
        # explicit ENU axes; reordered recordings must produce identical labels.
        goal_position_m=tuple(
            capture.snapshot["goal_position_m"][axis] for axis in ("x", "y", "z")
        ),
        safety_intervened=changed,
        commanded_change_squared=transport_command_change_squared(
            capture.previous_application,
            capture.application,
            limits=previous.sample.pilot_control_limits,
        ),
        power_joules=None,  # Unknown simulated current remains unknown, never free energy.
    )
    terminated = any((evidence.collision, evidence.geofence_violation, evidence.loss_of_control,
                      evidence.premature_landing, evidence.verified_mission_complete,
                      action.mode == "abort"))
    step = FlightStep(
        observation=observation,
        proposed_action_sha256=sha256_json(action),
        applied_action=applied,
        applied_command_sha256=sha256_json(capture.command),
        evidence=evidence,
        terminated=terminated,
        truncated=not terminated and capture.truncated,
    )
    step.validate_transition(previous, action)
    record = {
        "source_observation": previous.model_dump(mode="json"),
        "source_snapshot": capture.snapshot,
        "proposal": capture.proposal.model_dump(mode="json"),
        "application": capture.application.model_dump(mode="json"),
        "command": capture.command.model_dump(mode="json"),
        "outcome_receipt": receipt,
        "expired_model_proposal_replaced_by_safety_hold": capture.deadline_intervened,
        "command_change_metric": "normalized-transport-velocity-and-heading",
        "applied_action_coordinate_frame": NATIVE_ACTION_COORDINATE_FRAME,
        "step": step.model_dump(mode="json"),
    }
    return step, record
