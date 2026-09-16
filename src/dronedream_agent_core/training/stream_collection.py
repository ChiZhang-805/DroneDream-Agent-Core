"""Continuous source-driven DAgger collection, with offline-only supervision."""

import time
from dataclasses import replace

from ..hashing import sha256_json
from .collection_contracts import frozen_held_out_missions, validate_collection_arguments
from .dagger import DaggerResult, corrected_samples
from .evidence_snapshot import detach_evidence
from .flight_environment import ControlPreparationExpired, PilotAction, quiesced_simulation_rollout
from .stream_capture import StreamControlVisit


# 功能：
#   检查单调时钟的数值与前后顺序，防止无效时差绕过采集期限。
# 输入：
#   previous：上次有效秒数，初次读取为零。
# 输出：
#   now：有限、不回退且可用于短期限比较的当前秒数。
def collection_clock(previous=0.):
    now = time.monotonic()
    if type(now) not in (int, float) or not previous <= now <= 2**42:
        raise ValueError("STREAM_COLLECTION_CLOCK_INVALID")
    return now


# 功能：
#   1. 按新传感器来源连续产生提案，不等待上一动作结果或在飞行中调用教师。
#   2. 过期提案只允许换新来源重算；无进展、终止或错误时进入环境停止流程。
# 输入：
#   env：提供原生连续控制与落地后结算的仿真环境。
#   student：当前观测到四轴控制的冻结策略。
#   seed：本回合随机种子。
#   steps：最多接纳的提案数量。
#   held_out_missions：禁止参与训练的任务集合。
#   student_policy_sha256：必须绑定的学生策略摘要。
# 输出：
#   visits：落地后与实际执行回执关联的访问记录，不含虚构的奖励转移。
def collect_native_control_stream(
    env, student, *, seed, steps, held_out_missions, student_policy_sha256
):
    validate_collection_arguments(steps=steps, seed=seed, policy_sha256=student_policy_sha256)
    held_out_missions = frozen_held_out_missions(held_out_missions)
    if env.evidence_kind != "px4-gazebo" or not all(
        callable(getattr(env, method, None))
        for method in (
            "submit_stream_action",
            "next_stream_observation",
            "finalize_stream_captures",
            "bind_policy_identity",
            "quiesce",
        )
    ):
        raise ValueError("NATIVE_CONTINUOUS_COLLECTION_INTERFACE_REQUIRED")
    episode_steps = getattr(getattr(env, "config", None), "episode_steps", steps)
    if type(episode_steps) is not int or not 1 <= episode_steps <= 256:
        raise ValueError("STREAM_COLLECTION_EPISODE_STEPS_INVALID")
    limit = min(steps, episode_steps)
    submitted, seen, visits = 0, set(), []
    try:
        with quiesced_simulation_rollout(env):
            env.bind_policy_identity(student_policy_sha256)
            observation = env.reset(seed=seed)
            # 预处理失败不能续期；只尝试新来源，不重放先前动作。
            now = collection_clock()
            progress_deadline = now + 2.0
            attempts = 0
            while submitted < limit:
                now = collection_clock(now)
                if now >= progress_deadline:
                    raise TimeoutError("STREAM_COLLECTION_PROGRESS_TIMEOUT")
                attempts += 1
                if attempts > limit * 16:
                    raise TimeoutError("STREAM_COLLECTION_PREPARATION_ATTEMPT_LIMIT")
                if observation.mission_id in held_out_missions:
                    raise ValueError("DAGGER_HELD_OUT_MISSION_LEAKAGE")
                temporal = observation.sample.temporal_evidence
                identity = observation.episode_id, temporal.stream_id, temporal.observed_at_unix_ms
                if identity in seen:
                    raise ValueError("DAGGER_REPLAYED_ENVIRONMENT_STATE")
                seen.add(identity)
                before = detach_evidence(observation)
                proposal = PilotAction.model_validate(student(observation).model_dump())
                if observation != before:
                    raise ValueError("DAGGER_STUDENT_MUTATED_SOURCE_OBSERVATION")
                timing = getattr(student, "last_timing", None)
                record = getattr(env, "record_actor_timing", None)
                if timing is not None and callable(record):
                    if timing["sequence"] != observation.sequence:
                        raise ValueError("DAGGER_ACTOR_TIMING_IDENTITY_MISMATCH")
                    record(**timing)
                now = collection_clock(now)
                if now >= progress_deadline:
                    raise TimeoutError("STREAM_COLLECTION_PROGRESS_TIMEOUT")
                try:
                    env.submit_stream_action(proposal)
                except ControlPreparationExpired:
                    if proposal.mode == "abort":
                        # 终止意图即使回复过期也仍有效，由上下文退出触发原生安全停止。
                        raise
                    observation = env.next_stream_observation(deadline=progress_deadline)
                    continue
                submitted += 1
                now = collection_clock(now)
                progress_deadline = now + 2.0
                if proposal.mode == "abort" or submitted == limit:
                    break
                # 此处不等待回执、奖励或教师；下一输入必须新鲜，但不声称由上一动作导致。
                observation = env.next_stream_observation(deadline=progress_deadline)
    finally:
        if submitted:
            # 结算器独立核查落地；后续来源超时不能丢掉已接纳提案。
            visits = env.finalize_stream_captures()
    return visits


# 功能：
#   离线标注已关联实际执行的控制流记录，不把相邻输入伪装成单动作造成的下一状态。
# 输入：
#   visits：落地验证后的连续控制访问列表。
#   teacher：产生观测绑定纠正标签的教师回调。
#   risk_assessor：独立评价学生原始动作的风险回调。
#   held_out_missions：禁止用于训练的任务集合。
# 输出：
#   result：行为样本、风险样本及明确无奖励转移的来源记录。
def label_stream_visits(visits, teacher, risk_assessor, *, held_out_missions):
    if (type(visits) not in (list, tuple) or not 1 <= len(visits) <= 256
            or any(type(visit) is not StreamControlVisit for visit in visits)):
        raise ValueError("GROUNDED_STREAM_VISITS_REQUIRED")
    held_out_missions = frozen_held_out_missions(held_out_missions)
    visits = tuple(replace(v, observation=detach_evidence(v.observation),
                           proposal=detach_evidence(v.proposal),
                           applied_action=detach_evidence(v.applied_action)) for v in visits)
    behavior, risks, records, seen = [], [], [], set()
    for visit in visits:
        observation, proposal = visit.observation, visit.proposal
        identity = observation.episode_id, observation.sequence
        if identity in seen or observation.mission_id in held_out_missions:
            raise ValueError("DAGGER_DUPLICATE_OR_HELD_OUT_STATE")
        seen.add(identity)
        # 两个教师分别拿到独立副本，不能改写录下的输入和学生动作。
        correction = teacher(detach_evidence(observation))
        risk = risk_assessor(detach_evidence(observation), proposal.model_copy(deep=True))
        label, risk_label = corrected_samples(observation, proposal, correction, risk)
        behavior.append(label)
        if risk_label is not None:
            risks.append(risk_label)
        records.append(
            {
                "mission_id": observation.mission_id,
                "episode_id": observation.episode_id,
                "sequence": observation.sequence,
                "observation_sha256": sha256_json(observation),
                "evidence_kind": "px4-gazebo",
                "annotation_mode": "after-quiescence",
                "collection_mode": "continuous-source-driven-imitation",
                "not_a_reward_transition": True,
                "student_proposal": proposal.model_dump(),
                "teacher_selected": False,
                "teacher_correction": correction.model_dump(),
                "applied_action": visit.applied_action.model_dump(),
                "safety_intervened": visit.safety_intervened,
                "applied_command_sha256": visit.command_sha256,
                "application_sha256": visit.application_sha256,
                "capture_sha256": visit.capture_sha256,
                "proposed_action_risk": risk.model_dump() if risk is not None else None,
            }
        )
    result = DaggerResult(behavior, risks, records)
    return result
