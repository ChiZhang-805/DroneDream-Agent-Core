"""Collect student-visited physical states, then ask teachers only when grounded.

Teacher search and counterfactual geometry must not consume a live action's
deadline. This is the zero-teacher-mixture DAgger path; synchronous mixtures
remain available through collect_dagger for adapters with sufficient budget.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..hashing import sha256_json
from .collection_contracts import frozen_held_out_missions, validate_collection_arguments
from .dagger import DaggerResult, ProposedActionRisk, TeacherCorrection, corrected_samples
from .evidence_snapshot import detach_evidence
from .flight_environment import (
    FlightObservation,
    FlightStep,
    FlightTrainingEnvironment,
    PilotAction,
    quiesced_simulation_rollout,
)


@dataclass(frozen=True)
class StudentVisit:
    """Retain a detached source, proposal and verified next step for later supervision."""
    observation: FlightObservation
    proposal: PilotAction
    step: FlightStep


# 功能：
#   先完成学生控制并确认仿真停止，再用离线教师生成行为与动作风险标签。
# 输入：
#   env：仅仿真的训练环境。
#   student：由当前观测产生控制提案的学生策略。
#   teacher：停止飞行后调用的动作纠正器。
#   risk_assessor：独立评估学生原始动作风险的回调。
#   seed：首回合随机种子。
#   steps：最多采集步数。
#   held_out_missions：禁止参与训练的任务集合。
#   student_policy_sha256：当前冻结学生策略的内容摘要。
# 输出：
#   result：分开保存行为标签、风险标签和来源记录的结果。
def collect_deferred_dagger(
    env, student, teacher, risk_assessor, *, seed, steps, held_out_missions, student_policy_sha256
):
    held_out_missions = frozen_held_out_missions(held_out_missions)
    evidence_kind = env.evidence_kind
    visits = collect_student_visits(
        env,
        student,
        seed=seed,
        steps=steps,
        held_out_missions=held_out_missions,
        student_policy_sha256=student_policy_sha256,
    )
    result = label_student_visits(
        visits,
        teacher,
        risk_assessor,
        held_out_missions=held_out_missions,
        evidence_kind=evidence_kind,
    )
    return result


# 功能：
#   1. 采集绑定策略与来源的仿真状态，不在动作期限内调用教师或计算反事实。
#   2. 拒绝重复／留出状态和学生修改输入，显式终止后不再控制或重置。
# 输入：
#   env：提供重置、执行与安全停止的仿真环境。
#   student：当前观测到四轴动作的策略回调。
#   seed：首回合种子，普通回合结束后递增。
#   steps：一至二百五十六的采集上限。
#   held_out_missions：不允许采集为训练数据的任务集合。
#   student_policy_sha256：必须固定的策略摘要。
# 输出：
#   visits：独立保存原观测、学生提案与验证后下一步的访问列表。
def collect_student_visits(env: FlightTrainingEnvironment,
                          student: Callable[[FlightObservation], PilotAction], *, seed: int,
                          steps: int, held_out_missions: set[str] | frozenset[str],
                          student_policy_sha256: str) -> list[StudentVisit]:
    validate_collection_arguments(steps=steps, seed=seed, policy_sha256=student_policy_sha256)
    held_out_missions = frozen_held_out_missions(held_out_missions)
    visits, seen = [], set()
    with quiesced_simulation_rollout(env):
        bind = getattr(env, "bind_policy_identity", None)
        if env.evidence_kind == "px4-gazebo" and not callable(bind):
            raise ValueError("STUDENT_COLLECTION_POLICY_BINDING_REQUIRED")
        if callable(bind):
            bind(student_policy_sha256)
        observation = env.reset(seed=seed)
        episode = 0
        for index in range(steps):
            if observation.mission_id in held_out_missions:
                raise ValueError("DAGGER_HELD_OUT_MISSION_LEAKAGE")
            identity = observation.episode_id, observation.sequence
            if identity in seen:
                raise ValueError("DAGGER_REPLAYED_ENVIRONMENT_STATE")
            seen.add(identity)
            # 在回调前分离原始证据；教师几何求解与标注摘要放在确认停止之后。
            previous = detach_evidence(observation)
            proposal = student(observation)
            proposal = PilotAction.model_validate(proposal.model_dump())
            if observation != previous:
                raise ValueError("DAGGER_STUDENT_MUTATED_SOURCE_OBSERVATION")
            step = env.step(proposal.model_copy(deep=True))
            step.validate_transition(previous, proposal)
            visits.append(StudentVisit(previous, proposal, detach_evidence(step)))
            if proposal.mode == "abort":
                break  # 即使适配器尚未返回终止状态，也不为凑够步数再次起飞。
            if (step.terminated or step.truncated) and index + 1 < steps:
                episode += 1
                observation = env.reset(seed=seed + episode)
            else:
                observation = step.observation
    return visits


# 功能：
#   1. 在真实仿真控制期限内只采集原始输入、提案与独立执行见证，不计算训练奖励。
#   2. 确认仿真停止后再完成奖励计算与持久化，终止或截断后不补飞。
# 输入：
#   env：实现原生捕获、保留、停止和离线结算接口的仿真适配器。
#   student：冻结参数、只产生提案的本地策略。
#   seed：本次回合种子。
#   steps：最多采集步数。
#   held_out_missions：不能用于训练的任务集合。
#   student_policy_sha256：当前学生策略摘要。
# 输出：
#   visits：停止后完成独立验证的访问记录。
def collect_native_observation_first_visits(
    env, student, *, seed, steps, held_out_missions, student_policy_sha256
):
    validate_collection_arguments(steps=steps, seed=seed, policy_sha256=student_policy_sha256)
    held_out_missions = frozen_held_out_missions(held_out_missions)
    if env.evidence_kind != "px4-gazebo" or not all(
        callable(getattr(env, name, None))
        for name in (
            "capture_step",
            "retain_capture",
            "finalize_captures",
            "bind_policy_identity",
            "quiesce",
        )
    ):
        raise ValueError("OBSERVATION_FIRST_NATIVE_CAPTURE_INTERFACE_REQUIRED")
    captures, seen, visits = [], set(), []
    try:
        with quiesced_simulation_rollout(env):
            env.bind_policy_identity(student_policy_sha256)
            observation = env.reset(seed=seed)
            for _ in range(steps):
                if observation.mission_id in held_out_missions:
                    raise ValueError("DAGGER_HELD_OUT_MISSION_LEAKAGE")
                identity = observation.episode_id, observation.sequence
                if identity in seen:
                    raise ValueError("DAGGER_REPLAYED_ENVIRONMENT_STATE")
                seen.add(identity)
                previous = detach_evidence(observation)
                proposal = PilotAction.model_validate(student(observation).model_dump())
                # 学生不能改写适配器和独立证据共同引用的原始观测。
                if observation != previous:
                    raise ValueError("DAGGER_STUDENT_MUTATED_SOURCE_OBSERVATION")
                timing = getattr(student, "last_timing", None)
                record_timing = getattr(env, "record_actor_timing", None)
                if timing is not None and callable(record_timing):
                    if timing["sequence"] != observation.sequence:
                        raise ValueError("DAGGER_ACTOR_TIMING_IDENTITY_MISMATCH")
                    record_timing(**timing)
                captured = env.capture_step(proposal)
                if captured.source_observation != previous:
                    raise ValueError("DAGGER_CAPTURE_SOURCE_OBSERVATION_MISMATCH")
                captures.append(env.retain_capture(captured))
                observation = captured.observation
                ended = proposal.mode == "abort" or captured.truncated
                # 旧捕获保留为不可变字节，仅当前观测与历史窗口保持解码状态。
                del captured, previous
                if ended:
                    break  # 不为凑足条数再次起飞。
    finally:
        # 后续请求超时也保留先前捕获；结算器仍须独立拒绝未确认停止的运行。
        if captures:
            visits = env.finalize_captures(captures)
    return visits


# 功能：
#   对停止后采集的数据离线标注，固定整个批次，分别绑定教师动作和学生动作风险。
# 输入：
#   visits：经过环境来源验证的访问记录。
#   teacher：对独立观测给出纠正动作和验证回执的回调。
#   risk_assessor：对学生提案给出独立风险或不提供风险的回调。
#   held_out_missions：不能参与标注的留出任务集合。
#   evidence_kind：原生仿真、单元测试或预训练来源标记。
# 输出：
#   result：行为样本、动作风险样本及逐步来源记录。
def label_student_visits(visits: list[StudentVisit],
                        teacher: Callable[[FlightObservation], TeacherCorrection],
                        risk_assessor: Callable[[FlightObservation, PilotAction],
                                                ProposedActionRisk | None], *,
                        held_out_missions: set[str] | frozenset[str],
                        evidence_kind: str) -> DaggerResult:
    if (type(visits) not in (list, tuple) or not 1 <= len(visits) <= 256
            or any(type(visit) is not StudentVisit for visit in visits)
            or evidence_kind not in {"px4-gazebo", "unit-test", "pretraining"}):
        raise ValueError("DAGGER_STUDENT_VISITS_REQUIRED")
    held_out_missions = frozen_held_out_missions(held_out_missions)
    # 回调开始前复制整个批次，避免外部清空列表或改动后续记录造成静默遗漏。
    visits = tuple(StudentVisit(detach_evidence(v.observation), detach_evidence(v.proposal),
                                detach_evidence(v.step)) for v in visits)
    behavior, risks, records, seen = [], [], [], set()
    for visit in visits:
        observation, proposed, step = visit.observation, visit.proposal, visit.step
        identity = observation.episode_id, observation.sequence
        if identity in seen or observation.mission_id in held_out_missions:
            raise ValueError("DAGGER_DUPLICATE_OR_HELD_OUT_STATE")
        seen.add(identity)
        step.validate_transition(observation, proposed)
        digest = sha256_json(observation)
        # 教师拿到自己的副本；若基于改动后的观测签名，后续绑定检查会失败。
        correction = teacher(detach_evidence(observation))
        risk = risk_assessor(detach_evidence(observation), proposed.model_copy(deep=True))
        label, risk_label = corrected_samples(observation, proposed, correction, risk)
        behavior.append(label)
        if risk_label is not None:
            risks.append(risk_label)
        records.append(
            {
                "mission_id": observation.mission_id,
                "episode_id": observation.episode_id,
                "sequence": observation.sequence,
                "observation_sha256": digest,
                "evidence_kind": evidence_kind,
                "annotation_mode": "after-quiescence",
                "student_proposal": proposed.model_dump(),
                "teacher_selected": False,
                "teacher_correction": correction.model_dump(),
                "applied_action": step.applied_action.model_dump(),
                "safety_intervened": step.evidence.safety_intervened,
                "applied_command_sha256": step.applied_command_sha256,
                "outcome_receipt_sha256": step.evidence.verifier_receipt_sha256,
                "proposed_action_risk": risk.model_dump() if risk is not None else None,
            }
        )
    result = DaggerResult(behavior, risks, records)
    return result
