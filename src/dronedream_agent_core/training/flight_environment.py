"""Simulation step boundary and independently evidenced, non-farmable rewards.

This is not a replacement physics engine. A runtime adapter supplies actual
post-control observations and verifier evidence. Unit environments are labelled
as such and cannot produce flight qualification receipts.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Literal, Protocol

from pydantic import Field, model_serializer, model_validator

from ..contracts import StrictModel
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyObservation, policy_input_arrays

MODES = ("hold", "request-new-scan", "abort", "pilot-control")


class ControlPreparationExpired(TimeoutError):
    """A computed proposal missed dispatch reserve and was never submitted."""


class PilotAction(StrictModel):
    """Four normalized body velocity/yaw axes, not coordinates or motor thrust."""
    mode: Literal["hold", "request-new-scan", "abort", "pilot-control"]
    axes: list[float] = Field(min_length=4, max_length=4)

    # 功能：
    #   检查四轴幅度范围，并禁止保持、扫描与终止模式夹带非零运动轴。
    # 输入：
    #   self：经过字段解析的动作实例。
    # 输出：
    #   self：满足模式与轴值约束的动作。
    @model_validator(mode="after")
    def validate_axes(self):
        if any(not -1 <= value <= 1 for value in self.axes):
            raise ValueError("PILOT_ACTION_AXES_INVALID")
        if self.mode != "pilot-control" and any(self.axes):
            raise ValueError("NON_MOTION_ACTION_HAS_AXES")
        return self


class SafetyPositionControl(StrictModel):
    """Actual safety-controller output, never a normalized joystick label.

    Position control with velocity feed-forward is a different control law
    from pure velocity control. Preserve both setpoints instead of pretending
    a nonzero recovery/braking command was a zero-axis hold.
    """

    mode: Literal["safety-position-control"] = "safety-position-control"
    position_ned_m: tuple[float, float, float]
    velocity_feedforward_ned_mps: tuple[float, float, float]
    yaw_heading_deg: float
    safety_action: Literal["hold", "replan"]


class FlightObservation(StrictModel):
    """An episode-bound deployable sample, with explicit independent prior rows."""
    mission_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=0, strict=True)
    sample: LocalPolicyObservation
    prior_observations: list[LocalPolicyObservation] = Field(default_factory=list, max_length=32)

    # 功能：
    #   空历史不写入序列化结果，保持同一单帧观测的既有摘要语义。
    # 输入：
    #   self：待序列化观测。
    #   handler：Pydantic 标准字段序列化处理器。
    # 输出：
    #   result：包含真实历史或省略空历史的观测对象。
    @model_serializer(mode="wrap")
    def preserve_single_observation_identity(self, handler):
        result = handler(self)
        if not self.prior_observations:
            result.pop("prior_observations", None)
        return result

    # 功能：
    #   强制当前特征契约和连续控制语义，检查历史时间顺序、同流来源及去重。
    # 输入：
    #   self：当前样本与显式历史组成的观测。
    # 输出：
    #   self：通过契约和来源关系检查的观测。
    @model_validator(mode="after")
    def current_contract(self):
        if self.sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
            raise ValueError("TRAINING_OBSERVATION_CONTRACT_MISMATCH")
        if self.sample.temporal_evidence is None:
            raise ValueError("TRAINING_OBSERVATION_PROVENANCE_MISSING")
        previous_time = -1
        seen = {self.sample.temporal_evidence.sample_sha256}
        for prior in self.prior_observations:
            evidence = prior.temporal_evidence
            if (evidence is None
                    or evidence.stream_id != self.sample.temporal_evidence.stream_id
                    or not previous_time < evidence.observed_at_unix_ms
                    < self.sample.temporal_evidence.observed_at_unix_ms
                    or evidence.sample_sha256 in seen
                    or prior.control_feature_contract_sha256
                    != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                    or any(prior.candidate_mask)
                    or any(v for row in prior.candidate_features for v in row)):
                raise ValueError("TRAINING_PRIOR_OBSERVATION_IDENTITY_INVALID")
            seen.add(evidence.sample_sha256)
            previous_time = evidence.observed_at_unix_ms
        if any(self.sample.candidate_mask) or any(
            v for row in self.sample.candidate_features for v in row
        ):
            raise ValueError("CONTINUOUS_TRAINING_CANNOT_USE_ROUTE_CANDIDATES")
        return self

    # 功能：
    #   按部署侧完全一致的特征与掩码顺序形成当前观测的单行数值输入。
    # 输入：
    #   self：通过校验的训练观测。
    # 输出：
    #   features：共享特征编译器产生的展平数组。
    def policy_input(self):
        features = policy_input_arrays([self.sample])[0]
        return features


class RewardEvidence(StrictModel):
    """Verifier facts for reward calculation; an actor cannot self-declare success."""
    stage_id: str = Field(min_length=1)
    goal_revision: str = Field(min_length=1)
    elapsed_seconds: float = Field(gt=0, le=60)
    goal_distance_m: float = Field(ge=0)
    # PX4 may explicitly report unavailable current (e.g. -1 A). Unknown is
    # not zero consumption and cannot silently train an energy objective.
    power_joules: float | None = Field(ge=0)
    commanded_change_squared: float = Field(ge=0)
    safety_intervened: bool
    collision: bool = False
    geofence_violation: bool = False
    loss_of_control: bool = False
    premature_landing: bool = False
    required_wait: bool = False
    verified_stage_event_id: str | None = None
    verified_mission_complete: bool = False
    verifier_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FlightStep(StrictModel):
    observation: FlightObservation
    proposed_action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    applied_action: PilotAction | SafetyPositionControl = Field(discriminator="mode")
    applied_command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: RewardEvidence
    terminated: bool
    truncated: bool

    # 功能：
    #   检查安全替换、失败、成功与终止标志的一致性，截断不能同时标为终止。
    # 输入：
    #   self：动作执行后观测和独立见证组成的步骤。
    # 输出：
    #   self：结果标志互不矛盾的步骤。
    @model_validator(mode="after")
    def outcome_consistency(self):
        if isinstance(self.applied_action, SafetyPositionControl) and not (
            self.evidence.safety_intervened
        ):
            raise ValueError("UNREPORTED_CONTROL_INTERVENTION")
        fatal = any(
            (
                self.evidence.collision,
                self.evidence.geofence_violation,
                self.evidence.loss_of_control,
                self.evidence.premature_landing,
            )
        )
        if fatal and self.evidence.verified_mission_complete:
            raise ValueError("FAILED_FLIGHT_CANNOT_BE_SUCCESS")
        if (fatal or self.evidence.verified_mission_complete) and not self.terminated:
            raise ValueError("VERIFIED_TERMINAL_EVENT_NOT_TERMINATED")
        if self.terminated and self.truncated:
            raise ValueError("TERMINATION_AND_TRUNCATION_CONFLICT")
        return self

    # 功能：
    #   将本步与同回合连续序号、更新传感器来源和确切原始提案关联，检查未报告干预。
    # 输入：
    #   self：需要验证的已执行步骤。
    #   previous：本步控制前的观测。
    #   proposed：当时实际提交的提案。
    # 输出：
    #   None：不返回业务数据。
    def validate_transition(self, previous: FlightObservation, proposed: PilotAction) -> None:
        if (
            self.observation.episode_id != previous.episode_id
            or self.observation.mission_id != previous.mission_id
            or self.observation.map_sha256 != previous.map_sha256
            or self.observation.sequence != previous.sequence + 1
        ):
            raise ValueError("SIMULATION_STEP_IDENTITY_MISMATCH")
        before, after = previous.sample.temporal_evidence, self.observation.sample.temporal_evidence
        if (
            after.observed_at_unix_ms <= before.observed_at_unix_ms
            or after.sample_sha256 == before.sample_sha256
            or after.stream_id != before.stream_id
            or after.reset_history
        ):
            raise ValueError("SIMULATION_STEP_REUSED_OBSERVATION")
        if self.proposed_action_sha256 != sha256_json(proposed):
            raise ValueError("SIMULATION_STEP_PROPOSAL_MISMATCH")
        differs = isinstance(self.applied_action, SafetyPositionControl) or (
            self.applied_action.mode != proposed.mode or any(
            abs(a-b) > 1e-6 for a, b in zip(self.applied_action.axes, proposed.axes, strict=True)
        ))
        if differs and not self.evidence.safety_intervened:
            raise ValueError("UNREPORTED_CONTROL_INTERVENTION")


class FlightTrainingEnvironment(Protocol):
    """Adapters must guard physical control; learning is simulation-only."""

    simulation_only: bool
    evidence_kind: Literal["px4-gazebo", "unit-test", "pretraining"]

    # 功能：
    #   定义创建新仿真回合并取得初始观测的适配器接口。
    # 输入：
    #   self：仿真适配器。
    #   seed：本回合随机种子。
    # 输出：
    #   observation：实现方返回的初始观测。
    def reset(self, *, seed: int) -> FlightObservation: ...

    # 功能：
    #   定义经安全控制路径执行提案并取得独立结果证据的接口。
    # 输入：
    #   self：仿真适配器。
    #   action：四轴提案或显式非运动动作。
    # 输出：
    #   step：真实执行关联后的结果，不能由提案自行推断。
    def step(self, action: PilotAction) -> FlightStep: ...

    # 功能：
    #   定义撤销训练控制并确认停止的幂等接口；失败必须抛出异常。
    # 输入：
    #   self：仿真适配器。
    # 输出：
    #   None：不返回业务数据。
    def quiesce(self) -> None: ...

    # 功能：
    #   定义确认控制停止并释放适配器持有资源的接口。
    # 输入：
    #   self：仿真适配器。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None: ...


# 功能：
#   仅允许仿真训练，并在正常结束或异常退出时调用安全停止；停止失败继续向外传播。
# 输入：
#   env：带明确来源类型、原生仿真时必须具备停止方法的环境。
# 输出：
#   None：上下文不提供业务对象。
@contextmanager
def quiesced_simulation_rollout(env: FlightTrainingEnvironment):
    if env.simulation_only is not True or env.evidence_kind not in {
        "px4-gazebo", "unit-test", "pretraining",
    }:
        raise ValueError("PHYSICAL_FLIGHT_TRAINING_FORBIDDEN")
    stop = getattr(env, "quiesce", None)
    if env.evidence_kind == "px4-gazebo" and not callable(stop):
        raise ValueError("PHYSICAL_SIMULATION_QUIESCE_REQUIRED")
    try:
        yield
    finally:
        if callable(stop):
            # 此处不吞掉落地失败，否则调用方会继续进入优化器更新权重。
            stop()


class RewardConfig(StrictModel):
    progress_per_m: float = Field(default=1.0, ge=0)
    stage_completion: float = Field(default=5.0, ge=0)
    mission_completion: float = Field(default=25.0, ge=0)
    failure: float = Field(default=50.0, gt=0)
    time_per_second: float = Field(default=0.05, ge=0)
    energy_per_joule: float = Field(default=0.001, ge=0)
    action_change: float = Field(default=0.02, ge=0)
    intervention: float = Field(default=1.0, ge=0)


class FlightReward:
    """Stateful per-episode reward ledger; repeated progress/events cannot be farmed."""
    # 功能：
    #   重新验证并独立持有奖励配置，初始化只属于当前回合的累计账本。
    # 输入：
    #   self：待初始化奖励器。
    #   config：可选奖励权重，None 时采用默认配置。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, config: RewardConfig | None = None) -> None:
        if config is not None and not isinstance(config, RewardConfig):
            raise ValueError("REWARD_CONFIGURATION_INVALID")
        self.config = (RewardConfig() if config is None
                       else RewardConfig.model_validate(config.model_dump()))
        self.reset()

    # 功能：
    #   清空回合最佳距离、已奖励阶段及终止状态；只在新回合调用，不能因改目标重置。
    # 输入：
    #   self：奖励器。
    # 输出：
    #   None：不返回业务数据。
    def reset(self) -> None:
        self.stage: tuple[str, str] | None = None
        self.best_remaining: float | None = None
        self.best_by_goal: dict[tuple[str, str], float] = {}
        self.completed: set[str] = set()
        self.mission_awarded = False
        self.ended = False

    # 功能：
    #   1. 根据独立见证计算各项奖惩，只奖励同目标历史最佳进展且阶段事件仅奖励一次。
    #   2. 致命结果取消正向奖励；能耗未知时不能假设为零，异常或溢出不得更新账本。
    # 输入：
    #   self：当前回合奖励器。
    #   step：来自已验证执行链路的仿真步骤。
    # 输出：
    #   result：进展、阶段、任务、失败、耗时、能耗、动作变化与干预的有限有符号分量。
    def score(self, step: FlightStep) -> dict[str, float]:
        if self.ended:
            raise ValueError("REWARD_AFTER_EPISODE_END")
        if not isinstance(step, FlightStep):
            raise ValueError("REWARD_STEP_INVALID")
        # 不能依靠先前构造验证：嵌套列表和 model_copy(update=...) 仍可被调用方改写。
        step = FlightStep.model_validate(step.model_dump())
        e, c = step.evidence, RewardConfig.model_validate(self.config.model_dump())
        if e.power_joules is None and c.energy_per_joule > 0:
            raise ValueError("ENERGY_REWARD_REQUIRES_MEASURED_CONSUMPTION")
        stage = (e.stage_id, e.goal_revision)
        progress = 0.0
        prior_best = self.best_by_goal.get(stage, e.goal_distance_m)
        if stage == self.stage:
            progress = max(0.0, prior_best - e.goal_distance_m)
        stage_award = 0.0
        if e.verified_stage_event_id is not None and e.stage_id not in self.completed:
            stage_award = c.stage_completion
        fatal = any((e.collision, e.geofence_violation, e.loss_of_control, e.premature_landing))
        mission_award = (
            c.mission_completion
            if e.verified_mission_complete and not self.mission_awarded
            else 0.0
        )
        # 致命故障不能用进展或完成事件冲抵；先完成数值检查再提交状态。
        result = {
            "progress": 0.0 if fatal else progress * c.progress_per_m,
            "stage": 0.0 if fatal else stage_award,
            "mission": 0.0 if fatal else mission_award,
            "failure": -c.failure
            if fatal or (step.terminated and not e.verified_mission_complete)
            else 0.0,
            "time": 0.0 if e.required_wait else -e.elapsed_seconds * c.time_per_second,
            "energy": -(e.power_joules or 0.0) * c.energy_per_joule,
            "change": -e.commanded_change_squared * c.action_change,
            "intervention": -c.intervention if e.safety_intervened else 0.0,
        }
        if not all(math.isfinite(value) for value in result.values()) or not math.isfinite(
            sum(result.values())
        ):
            raise ValueError("REWARD_NONFINITE")
        self.stage = stage
        self.best_remaining = min(prior_best, e.goal_distance_m)
        self.best_by_goal[stage] = self.best_remaining
        if e.verified_stage_event_id is not None:
            self.completed.add(e.stage_id)
        self.mission_awarded |= e.verified_mission_complete
        self.ended = step.terminated or step.truncated
        return result
