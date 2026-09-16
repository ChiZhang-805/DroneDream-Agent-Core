"""Offline PPO for the very same recurrent policy exported to the local runtime.

BC trains perception/history and control together. This conservative refinement
freezes the entire shared representation (including the nonlinear trunk), so
the auxiliary risk output cannot drift while the four-axis/mode heads learn.
The action-conditioned safety critic is separately trained and never bypassed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from ..causal_control import (
    CONTROL_HISTORY_CONTRACT_SHA256,
    CONTROL_HISTORY_WIDTH,
    CONTROL_REALTIME_WIDTH,
    CONTROL_STATE_WIDTH,
    CausalControlHistory,
)
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from .causal_policy import (
    CausalExample,
    CausalPilotPolicy,
    _validated_config,
    example_tensors,
    validate_learning_examples,
)
from .causal_student import append_flight_history
from .evidence_snapshot import detach_evidence
from .flight_environment import FlightObservation, FlightStep, PilotAction, RewardConfig
from .ppo import HybridPilotDistribution, PPOConfig, PPOTrainer


class CausalPilotActorCritic(HybridPilotDistribution):
    """Train mode/axis heads and a training-only value head on a frozen representation."""

    # 功能：
    #   复制当前循环策略，只解冻模式和四轴头；新增训练专用价值头而不改变共享风险表示。
    # 输入：
    #   self：新建的 actor-critic 包装器。
    #   base：已经训练的循环控制策略。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, base: CausalPilotPolicy):
        super().__init__()
        if not isinstance(base, CausalPilotPolicy):
            raise ValueError("CAUSAL_PPO_BASE_POLICY_INVALID")
        self.policy = copy.deepcopy(base).cpu().eval()
        self.policy.config = _validated_config(self.policy.config)
        if any(
            value.dtype != torch.float32 or not torch.isfinite(value).all()
            for value in self.policy.state_dict().values()
        ):
            raise ValueError("CAUSAL_PPO_BASE_WEIGHTS_INVALID")
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        for head in (self.policy.mode_head, self.policy.axes_head):
            for parameter in head.parameters():
                parameter.requires_grad_(True)
        # 价值头随后清零，不让它临时的随机初始化推进调用方的全局随机数状态。
        with torch.random.fork_rng(devices=[]):
            self.value_head = nn.Linear(self.policy.config.encoder_width, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        length = self.policy.config.history_length
        self.widths = (
            CONTROL_STATE_WIDTH,
            CONTROL_REALTIME_WIDTH,
            CONTROL_REALTIME_WIDTH,
            length * CONTROL_HISTORY_WIDTH,
            length,
        )
        if self.policy.config.visual_feature_count:
            self.widths += (self.policy.config.visual_feature_count,)

    # 功能：
    #   恢复扁平输入的共享时间布局，计算模式、未压缩四轴均值和训练价值。
    # 输入：
    #   self：循环 actor-critic 网络。
    #   inputs：按已固定宽度排列的二维特征张量。
    # 输出：
    #   mode_logits：模式分布的未归一化值。
    #   axes_mean：四轴高斯分布的均值。
    #   values：每个样本的标量价值。
    def forward(self, inputs):
        columns = list(torch.split(inputs, self.widths, dim=-1))
        columns[3] = columns[3].reshape(
            -1, self.policy.config.history_length, CONTROL_HISTORY_WIDTH
        )
        hidden = self.policy.latent(*columns)
        mode_logits, axes_mean = self.policy.mode_head(hidden), self.policy.axes_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return mode_logits, axes_mean, values

    # 功能：
    #   复制部署用策略，不携带价值头、采样噪声或共享训练器参数引用。
    # 输入：
    #   self：已微调的 actor-critic。
    # 输出：
    #   model：可继续离线训练或进入独立导出验收的循环策略副本。
    def deployment_model(self) -> CausalPilotPolicy:
        model = copy.deepcopy(self.policy).cpu().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model


@dataclass(frozen=True)
class CausalReplay:
    inputs: np.ndarray
    targets: np.ndarray
    pilot_controls: np.ndarray


# 功能：
#   复用公共窗口张量构造器后展平，保持 PPO、行为克隆和导出模型的输入顺序一致。
# 输入：
#   examples：完整的因果窗口列表。
#   config：循环网络配置。
# 输出：
#   inputs：有限 float32 二维输入数组。
def flatten_examples(examples, config):
    config = _validated_config(config)
    columns = example_tensors(examples, config.visual_feature_count)
    inputs = np.concatenate(
        [column.numpy().reshape(len(examples), -1) for column in columns], axis=1
    )
    return inputs


class CausalPPOTrainer(PPOTrainer):
    """Refine one navigation expert offline, bound to current replay and feature hashes."""

    # 功能：
    #   固定单专家策略、完整来源窗口、奖励和优化配置，建立具有精确恢复身份的离线训练器。
    # 输入：
    #   self：新建的训练器。
    #   base：当前循环策略。
    #   replay：含真实来源与完整历史的行为回放。
    #   config：可选 PPO 参数。
    #   reward_config：可选奖励配置。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        base: CausalPilotPolicy,
        replay: list[CausalExample],
        config: PPOConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.config = PPOConfig.model_validate(
            (config if config is not None else PPOConfig()).model_dump(), strict=True
        )
        self.reward_config = RewardConfig.model_validate(
            (reward_config if reward_config is not None else RewardConfig()).model_dump(),
            strict=True,
        )
        if self.config.seed > 2**63 - 1:
            raise ValueError("PPO_SEED_INVALID")
        if type(replay) is not list or not 1 <= len(replay) <= 250_000:
            raise ValueError("CAUSAL_PPO_REPLAY_SIZE_INVALID")
        # 冻结后再校验、构造数组和摘要，不能把优化用的副本与外部原件混在一份回执里。
        replay = copy.deepcopy(replay)
        self.actor = CausalPilotActorCritic(base)
        base = self.actor.policy
        validate_learning_examples(replay, base.config.history_length)
        if not replay or any(
            e.sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or e.sample.temporal_evidence is None
            or not e.group_id.strip()
            or e.sample.target_action_index not in range(8, 12)
            or len(e.sample.target_pilot_control) != 4
            or len(e.history) != base.config.history_length
            or e.history_mask != [1.0] * base.config.history_length
            or any(e.sample.candidate_mask)
            or any(v for row in e.sample.candidate_features for v in row)
            for e in replay
        ):
            raise ValueError("CAUSAL_PPO_REQUIRES_COMPLETE_CURRENT_REPLAY")
        if len({e.sample.navigation_expert_role for e in replay}) != 1:
            raise ValueError("PPO_UPDATES_ONE_NAVIGATION_EXPERT_AT_A_TIME")
        self.expert_role = replay[0].sample.navigation_expert_role
        self.replay = CausalReplay(
            flatten_examples(replay, base.config),
            np.asarray([e.sample.target_action_index for e in replay], dtype=np.int64),
            np.asarray([e.sample.target_pilot_control for e in replay], dtype=np.float32),
        )
        self._initialize_optimizer()
        self.base_hash = sha256_json(
            {
                "architecture": "causal-gru-control",
                "config": base.config.model_dump(),
                "history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
                "weights": {
                    name: value.detach().cpu().tolist() for name, value in base.state_dict().items()
                },
            }
        )
        self.replay_hash = sha256_json(
            [
                {
                    "sample": e.sample.model_dump(mode="json"),
                    "history": e.history,
                    "history_mask": e.history_mask,
                    "group_id": e.group_id,
                    "source_sha256": e.source_sha256,
                }
                for e in replay
            ]
        )
        self.history = CausalControlHistory(base.config.history_length)
        self.warmup_records: list[dict] = []

    # 功能：
    #   清空本批预热证据，并通过父类的仿真停稳保护执行连续采集。
    # 输入：
    #   self：循环 PPO 训练器。
    #   env：明确允许离线采集的仿真或合成单元环境。
    # 输出：
    #   rollout：带控制提案、实际动作和独立结果证据的采集批次。
    def collect(self, env):
        self.warmup_records = []
        rollout = super().collect(env)
        return rollout

    # 功能：
    #   停稳后计算采集和预热记录的摘要，全部成功才提交，失败不丢掉待哈希的原始证据。
    # 输入：
    #   self：持有预热记录的训练器。
    #   rollout：父类完成物理采集后的批次。
    # 输出：
    #   None：不返回业务数据。
    def _finalize_evidence(self, rollout):
        staged = replace(rollout, records=[detach_evidence(row) for row in rollout.records])
        warmup = [detach_evidence(row) for row in self.warmup_records]
        super()._finalize_evidence(staged)
        for record in warmup:
            record["observation_sha256"] = sha256_json(record.pop("_observation_for_hash"))
        rollout.records, self.warmup_records = staged.records, warmup

    # 功能：
    #   重置后用有界真实悬停步骤积累独立历史，禁止以填充帧或被环境改写的提案替代。
    # 输入：
    #   self：拥有历史缓存和证据列表的训练器。
    #   env：当前仿真环境。
    # 输出：
    #   observation：历史已完整的最新观测。
    def _reset_environment(self, env):
        self.history.clear()
        observation = super()._reset_environment(env)
        observation = FlightObservation.model_validate(observation.model_dump())
        self._append(observation)
        hold = PilotAction(mode="hold", axes=[0.0] * 4)
        # Physical hold steps, never manufactured padding, establish history.
        # Bounded attempts avoid a broken sensor keeping training alive forever.
        for _ in range(2 * self.actor.policy.config.history_length):
            if self.history.ready:
                return observation
            step = env.step(hold.model_copy(deep=True))
            if not isinstance(step, FlightStep):
                raise ValueError("CAUSAL_PPO_WARMUP_STEP_INVALID")
            step = FlightStep.model_validate(step.model_dump())
            step.validate_transition(observation, hold)
            self.warmup_records.append(
                {
                    "purpose": "history-warmup",
                    "episode_id": observation.episode_id,
                    "sequence": observation.sequence,
                    "_observation_for_hash": observation.model_dump(mode="json"),
                    "proposed_action": hold.model_dump(),
                    "applied_action": step.applied_action.model_dump(),
                    "applied_command_sha256": step.applied_command_sha256,
                    "verifier_receipt_sha256": step.evidence.verifier_receipt_sha256,
                    "evidence_kind": env.evidence_kind,
                }
            )
            if step.terminated or step.truncated:
                raise ValueError("CAUSAL_PPO_EPISODE_ENDED_DURING_HISTORY_WARMUP")
            observation = step.observation
            self._append(observation)
        raise ValueError("CAUSAL_PPO_INDEPENDENT_HISTORY_UNAVAILABLE")

    # 功能：
    #   只将当前专家的真实独立观测追加到历史，阶段路由切换时停止该专家的采集。
    # 输入：
    #   self：当前专家训练器。
    #   observation：包含传感器来源的当前观测。
    # 输出：
    #   None：不返回业务数据。
    def _append(self, observation):
        sample = observation.sample
        if sample.navigation_expert_role != self.expert_role:
            raise ValueError("PPO_ENVIRONMENT_ROUTED_TO_DIFFERENT_EXPERT")
        append_flight_history(self.history, observation)

    # 功能：
    #   追加当前观测并要求完整历史，生成与离线回放相同的 actor 输入而不附带奖励真值。
    # 输入：
    #   self：历史窗口维护者。
    #   observation：当前实际观测。
    # 输出：
    #   inputs：当前单帧的扁平控制输入。
    def _observation_input(self, observation):
        self._append(observation)
        if not self.history.ready:
            raise ValueError("CAUSAL_PPO_HISTORY_INTERRUPTED")
        rows, mask = self.history.values()
        inputs = flatten_examples(
            [CausalExample(observation.sample, rows, mask, "live-rollout")],
            self.actor.policy.config,
        )[0]
        return inputs
