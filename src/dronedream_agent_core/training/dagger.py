"""Audited DAgger collection on student-visited simulation states."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import Field

from ..contracts import NormalizedPilotControl, StrictModel
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyTrainingSample
from ..pilot_control_mapping import action_risk_features
from .collection_contracts import (
    frozen_held_out_missions,
    validate_collection_arguments,
    validate_collection_size,
)
from .flight_environment import (
    MODES,
    FlightObservation,
    FlightStep,
    FlightTrainingEnvironment,
    PilotAction,
    quiesced_simulation_rollout,
)


class TeacherCorrection(StrictModel):
    """An observation-bound teacher label, not proof that its action was executed."""
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: PilotAction
    verified_action_risk: float = Field(ge=0, le=1)
    verifier_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProposedActionRisk(StrictModel):
    """Assess the student's own proposed action separately from the teacher replacement."""
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposed_action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk: float = Field(ge=0, le=1)
    source: str = Field(pattern=r"^(swept-geometry|physical-counterfactual)$")
    verifier_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass
class DaggerResult:
    """Keep behavior and counterfactual-risk datasets distinct with joined collection records."""
    behavior_samples: list[LocalPolicyTrainingSample]
    action_risk_samples: list[LocalPolicyTrainingSample]
    records: list[dict]


# 功能：
#   1. 形成共用的行为纠正和学生动作风险标签，重新验证可变模型及其来源摘要。
#   2. 只有连续速度提案可训练四轴风险，保持与位置控制不能伪装成速度标签。
# 输入：
#   observation：部署侧可获得的观测，不包含教师专用真值。
#   proposed：学生原始四轴提案。
#   correction：绑定同一观测的教师动作与验证回执。
#   risk：可选的学生原始动作独立风险。
# 输出：
#   result：行为样本和可选动作风险样本的二元组。
def corrected_samples(observation: FlightObservation, proposed: PilotAction,
                      correction: TeacherCorrection, risk: ProposedActionRisk | None
                      ) -> tuple[LocalPolicyTrainingSample, LocalPolicyTrainingSample | None]:
    if (not isinstance(observation, FlightObservation) or not isinstance(proposed, PilotAction)
            or not isinstance(correction, TeacherCorrection)
            or (risk is not None and not isinstance(risk, ProposedActionRisk))):
        raise ValueError("DAGGER_LABEL_CONTRACT_INVALID")
    observation = FlightObservation.model_validate(observation.model_dump())
    proposed = PilotAction.model_validate(proposed.model_dump())
    correction = TeacherCorrection.model_validate(correction.model_dump())
    risk = None if risk is None else ProposedActionRisk.model_validate(risk.model_dump())
    digest = sha256_json(observation)
    if correction.observation_sha256 != digest:
        raise ValueError("DAGGER_TEACHER_WRONG_OBSERVATION")
    label = LocalPolicyTrainingSample.model_validate({
        # 保持传感器快照身份；回合观测摘要属于教师回执，替换特征身份会破坏历史匹配。
        **observation.sample.model_dump(),
        "target_action_index": MODES.index(correction.action.mode) + 8,
        "target_pilot_control": correction.action.axes,
        "risk_target": correction.verified_action_risk, "risk_proposed_control": [],
    })
    risk_label = None
    if risk is not None:
        if proposed.mode != "pilot-control":
            raise ValueError("DAGGER_ACTION_RISK_REQUIRES_VELOCITY_CONTROL")
        if (risk.observation_sha256 != digest
                or risk.proposed_action_sha256 != sha256_json(proposed)):
            raise ValueError("DAGGER_COUNTERFACTUAL_BINDING_MISMATCH")
        if observation.sample.pilot_control_limits is None:
            raise ValueError("DAGGER_PHYSICAL_ACTION_LIMITS_MISSING")
        physical = action_risk_features(NormalizedPilotControl(
            forward_axis=proposed.axes[0], right_axis=proposed.axes[1],
            up_axis=proposed.axes[2], yaw_axis=proposed.axes[3]),
            observation.sample.pilot_control_limits, harness_scale=1.)
        risk_label = LocalPolicyTrainingSample.model_validate({
            **label.model_dump(), "risk_target": risk.risk,
            "risk_proposed_control": list(physical),
        })
    result = label, risk_label
    return result


# 功能：
#   1. 显式执行同步教师／学生混合采集，统一输入预算并在退出时安全停止仿真。
#   2. 此入口不替代实时连续采集；同步教师耗时仍受具体仿真适配器的动作期限约束。
# 输入：
#   env：仅仿真的步进环境。
#   student：产生学生提案的策略。
#   teacher：同步产生纠正动作的教师。
#   risk_assessor：独立评价学生提案风险的回调。
#   seed：采集与混合抽样的非负整数种子。
#   steps：单次最多二百五十六步。
#   teacher_probability：零至一的教师执行概率。
#   held_out_missions：禁止参与训练的任务集合。
#   student_policy_sha256：原生仿真必须提供的冻结策略摘要。
# 输出：
#   result：分开的行为标签、风险标签及实际执行来源记录。
def collect_dagger(
    env: FlightTrainingEnvironment,
    student: Callable[[FlightObservation], PilotAction],
    teacher: Callable[[FlightObservation], TeacherCorrection],
    risk_assessor: Callable[[FlightObservation, PilotAction], ProposedActionRisk | None],
    *,
    seed: int,
    steps: int,
    teacher_probability: float,
    held_out_missions: set[str],
    student_policy_sha256: str | None = None,
) -> DaggerResult:
    validate_collection_size(steps=steps, seed=seed)
    if type(teacher_probability) not in (float, int) or not 0 <= teacher_probability <= 1:
        raise ValueError("DAGGER_COLLECTION_CONFIGURATION_INVALID")
    held_out_missions = frozen_held_out_missions(held_out_missions)
    if student_policy_sha256 is not None:
        validate_collection_arguments(steps=steps, seed=seed, policy_sha256=student_policy_sha256)
    with quiesced_simulation_rollout(env):
        if env.evidence_kind == "px4-gazebo":
            bind = getattr(env, "bind_policy_identity", None)
            if not callable(bind) or student_policy_sha256 is None:
                raise ValueError("DAGGER_PHYSICAL_ROLLOUT_REQUIRES_POLICY_IDENTITY")
            bind(student_policy_sha256)
        result = _collect_running(
            env, student, teacher, risk_assessor, seed=seed, steps=steps,
            teacher_probability=teacher_probability, held_out_missions=held_out_missions,
        )
    return result


# 功能：
#   采集学生到达的状态，分别记录教师纠正和实际动作；选中终止动作后立即结束本批。
# 输入：
#   env：已进入停止保护上下文的仿真环境。
#   student：学生提案回调。
#   teacher：教师纠正回调。
#   risk_assessor：学生动作风险回调。
#   seed：回合与混合选择的种子。
#   steps：本批采集上限。
#   teacher_probability：选择教师动作执行的概率。
#   held_out_missions：固定的留出任务集合。
# 输出：
#   result：行为、风险和执行记录的集合。
def _collect_running(
    env: FlightTrainingEnvironment,
    student: Callable[[FlightObservation], PilotAction],
    teacher: Callable[[FlightObservation], TeacherCorrection],
    risk_assessor: Callable[[FlightObservation, PilotAction], ProposedActionRisk | None],
    *,
    seed: int,
    steps: int,
    teacher_probability: float,
    held_out_missions: set[str],
) -> DaggerResult:
    if env.simulation_only is not True or env.evidence_kind not in {
        "px4-gazebo",
        "unit-test",
        "pretraining",
    }:
        raise ValueError("DAGGER_PHYSICAL_TRAINING_FORBIDDEN")
    generator = random.Random(seed)
    observation = env.reset(seed=seed)
    episode = 0
    seen: set[tuple[str, int]] = set()
    behavior, risks, records = [], [], []
    for step_index in range(steps):
        if observation.mission_id in held_out_missions:
            raise ValueError("DAGGER_HELD_OUT_MISSION_LEAKAGE")
        identity = (observation.episode_id, observation.sequence)
        if identity in seen:
            raise ValueError("DAGGER_REPLAYED_ENVIRONMENT_STATE")
        seen.add(identity)
        before = observation.model_copy(deep=True)
        # 每次回调返回后立即固定结果，下一回调不能通过共享引用改写先前提案。
        proposed = PilotAction.model_validate(
            student(observation.model_copy(deep=True)).model_dump(),
        )
        correction = TeacherCorrection.model_validate(
            teacher(observation.model_copy(deep=True)).model_dump(),
        )
        risk = risk_assessor(observation.model_copy(deep=True), proposed.model_copy(deep=True))
        risk = None if risk is None else ProposedActionRisk.model_validate(risk.model_dump())
        if observation != before:
            raise ValueError("DAGGER_CALLBACK_MUTATED_SOURCE_OBSERVATION")
        label, risk_label = corrected_samples(observation, proposed, correction, risk)
        behavior.append(label)
        if risk_label is not None:
            risks.append(risk_label)
        teacher_selected = generator.random() < teacher_probability
        selected = correction.action if teacher_selected else proposed
        step = FlightStep.model_validate(env.step(selected.model_copy(deep=True)).model_dump())
        step.validate_transition(before, selected)
        records.append(
            {
                "mission_id": before.mission_id,
                "episode_id": before.episode_id,
                "sequence": before.sequence,
                "evidence_kind": env.evidence_kind,
                "student_proposal": proposed.model_dump(),
                "teacher_correction": correction.model_dump(),
                "teacher_selected": teacher_selected,
                "applied_action": step.applied_action.model_dump(),
                "safety_intervened": step.evidence.safety_intervened,
                "applied_command_sha256": step.applied_command_sha256,
                "proposed_action_risk": risk.model_dump() if risk is not None else None,
            }
        )
        if selected.mode == "abort":
            break
        if (step.terminated or step.truncated) and step_index + 1 < steps:
            episode += 1
            observation = env.reset(seed=seed + episode)
        else:
            observation = step.observation
    result = DaggerResult(behavior, risks, records)
    return result
