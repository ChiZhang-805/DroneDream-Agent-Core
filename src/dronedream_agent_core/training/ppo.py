"""Offline PPO refinement of the deployed four-axis policy, not an onboard learner.

The shared encoder and risk head remain frozen. Trainable categorical/continuous
heads are initialized from BC, with a separate training-only value head. Export
returns the original ONNX-compatible architecture; neither the value head nor
exploration noise ships on the aircraft.
"""

from __future__ import annotations

import copy
import io
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from pydantic import Field, model_validator
from torch import nn

from ..contracts import StrictModel
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyTrainingSample, NumpyLocalPolicyModel, _arrays
from ..pilot_control_mapping import PilotControlLimits
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from ..realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from .evidence_snapshot import detach_evidence
from .flight_environment import (
    MODES,
    FlightReward,
    FlightStep,
    FlightTrainingEnvironment,
    PilotAction,
    RewardConfig,
    quiesced_simulation_rollout,
)


class PPOConfig(StrictModel):
    """Offline update and per-axis exploration bounds, bound into resume receipts."""

    seed: int = Field(default=17, ge=0)
    rollout_steps: int = Field(default=256, ge=2, le=65536)
    update_epochs: int = Field(default=4, ge=1, le=64)
    minibatch_size: int = Field(default=64, ge=2, le=4096)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.01)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_ratio: float = Field(default=0.2, gt=0, le=0.5)
    value_weight: float = Field(default=0.5, gt=0, le=10)
    entropy_weight: float = Field(default=0.005, ge=0, le=1)
    retention_weight: float = Field(default=1.0, gt=0, le=100)
    gradient_norm: float = Field(default=0.5, gt=0, le=10)
    target_kl: float = Field(default=0.025, gt=0, le=1)
    initial_axis_standard_deviation: list[float] = Field(
        default_factory=lambda: [0.08, 0.08, 0.04, 0.06], min_length=4, max_length=4
    )
    minimum_axis_standard_deviation: float = Field(default=0.01, gt=0, le=1)
    maximum_axis_standard_deviation: float = Field(default=0.25, gt=0, le=1)

    # 功能：
    #   确保每根控制轴的初始探索噪声处于同一个声明范围内。
    # 输入：
    #   self：字段范围已初步验证的 PPO 配置。
    # 输出：
    #   self：满足探索边界的配置。
    @model_validator(mode="after")
    def exploration_envelope(self):
        if any(
            not self.minimum_axis_standard_deviation
            <= value
            <= self.maximum_axis_standard_deviation
            for value in self.initial_axis_standard_deviation
        ):
            raise ValueError("PPO_EXPLORATION_STANDARD_DEVIATION_OUTSIDE_ENVELOPE")
        return self


# 功能：
#   逆序计算广义优势；时间截断只用该回合最后状态引导，不跨重置回合传播回报。
# 输入：
#   rewards：逐步奖励。
#   values：各步骤起点价值。
#   next_values：各步骤实际终点价值。
#   terminated：真实终止掩码。
#   truncated：时间截断掩码。
#   gamma：奖励折扣。
#   gae_lambda：优势递推权重。
# 输出：
#   advantages：有限 float32 优势。
#   returns：有限 float32 价值目标。
def generalized_advantages(
    rewards, values, next_values, terminated, truncated, *, gamma, gae_lambda
):
    arrays = [np.asarray(a) for a in (rewards, values, next_values, terminated, truncated)]
    if not arrays[0].size or any(a.shape != arrays[0].shape or a.ndim != 1 for a in arrays):
        raise ValueError("GAE_SHAPE_MISMATCH")
    if any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in arrays[:3]):
        raise ValueError("GAE_NONFINITE_VALUE")
    if any(not np.isin(a, (0, 1)).all() for a in arrays[3:]):
        raise ValueError("GAE_TERMINAL_MASK_INVALID")
    if (
        np.any(arrays[3].astype(bool) & arrays[4].astype(bool))
        or type(gamma) not in (int, float)
        or type(gae_lambda) not in (int, float)
        or not 0 < gamma <= 1
        or not 0 <= gae_lambda <= 1
    ):
        raise ValueError("GAE_CONFIGURATION_INVALID")
    r, v, nv = (array.astype(np.float64) for array in arrays[:3])
    term, trunc = arrays[3:]
    advantages = np.zeros(r.shape, dtype=np.float64)
    carry = 0.0
    for i in reversed(range(len(r))):
        delta = r[i] + gamma * nv[i] * (not term[i]) - v[i]
        carry = delta + gamma * gae_lambda * (not (term[i] or trunc[i])) * carry
        advantages[i] = carry
    returns = advantages + v
    limit = np.finfo(np.float32).max
    if any(not np.isfinite(a).all() or np.any(np.abs(a) > limit) for a in (advantages, returns)):
        raise ValueError("GAE_FLOAT32_OVERFLOW")
    advantages, returns = advantages.astype(np.float32), returns.astype(np.float32)
    return advantages, returns


class HybridPilotDistribution(nn.Module):
    """Shared mixed-mode/tanh-Gaussian distribution for either policy architecture."""

    # 功能：
    #   创建训练专用的逐轴对数噪声和固定上下界，不将探索噪声写入部署策略。
    # 输入：
    #   self：混合离散模式与连续四轴的分布对象。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        super().__init__()
        defaults = PPOConfig()
        self.log_std = nn.Parameter(
            torch.tensor(defaults.initial_axis_standard_deviation, dtype=torch.float32).log()
        )
        self.register_buffer(
            "axis_log_std_bounds",
            torch.tensor(
                [
                    defaults.minimum_axis_standard_deviation,
                    defaults.maximum_axis_standard_deviation,
                ],
                dtype=torch.float32,
            ).log(),
        )

    # 功能：
    #   在对数空间限制噪声，保证优化不能越过配置中的探索幅度。
    # 输入：
    #   self：持有噪声参数和范围的分布。
    # 输出：
    #   deviation：四根控制轴的标准差。
    def axis_standard_deviation(self):
        deviation = self.log_std.clamp(
            self.axis_log_std_bounds[0], self.axis_log_std_bounds[1]
        ).exp()
        return deviation

    # 功能：
    #   在当前策略下重新计算已采样动作的概率，用于 PPO 新旧策略概率比。
    # 输入：
    #   self：当前策略分布。
    #   inputs：动作对应的输入批次。
    #   modes：已采样的模式索引。
    #   latent：tanh 前的连续轴采样。
    # 输出：
    #   outputs：动作对数概率、价值和模式熵。
    def log_probability(self, inputs, modes, latent):
        logits, mean, value = self(inputs)
        outputs = self._distribution_outputs(logits, mean, value, modes, latent)
        return outputs

    # 功能：
    #   修正 tanh 变量变换的雅可比，只有连续控制模式计入四轴密度，复用已有前向计算。
    # 输入：
    #   self：当前探索分布。
    #   logits：模式的未归一化分数。
    #   mean：连续轴均值。
    #   value：状态价值。
    #   modes：选择的模式。
    #   latent：变换前采样。
    # 输出：
    #   probability：混合动作对数概率。
    #   value：原状态价值。
    #   entropy：离散模式熵。
    def _distribution_outputs(self, logits, mean, value, modes, latent):
        # Sampling already evaluated the recurrent encoder. Reuse that exact
        # forward pass for its probability/value, not a second model invocation.
        categorical = torch.distributions.Categorical(logits=logits)
        normal = torch.distributions.Normal(mean, self.axis_standard_deviation())
        log_jacobian = 2.0 * (np.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
        axis_log_prob = (normal.log_prob(latent) - log_jacobian).sum(-1)
        probability = categorical.log_prob(modes) + (modes == 3) * axis_log_prob
        entropy = categorical.entropy()
        return probability, value, entropy

    # 功能：
    #   用独立随机数发生器采样一次提案，非运动模式返回中立四轴且不重复调用网络。
    # 输入：
    #   self：当前策略分布。
    #   inputs：单条状态输入。
    #   generator：本训练器的随机数发生器。
    # 输出：
    #   action：归一化控制提案。
    #   mode_index：选择的模式索引。
    #   latent_axes：tanh 前的轴采样副本。
    #   probability：该提案的对数概率。
    #   state_value：当前状态价值。
    @torch.no_grad()
    def sample(self, inputs, generator):
        logits, mean, value = self(inputs)
        mode = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)
        latent = mean + self.axis_standard_deviation() * torch.randn(
            mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
        )
        log_probability, value, _ = self._distribution_outputs(logits, mean, value, mode, latent)
        action = PilotAction(
            mode=MODES[int(mode[0])],
            axes=torch.tanh(latent[0]).tolist() if mode[0] == 3 else [0.0] * 4,
        )
        mode_index, latent_axes = int(mode[0]), latent[0].cpu().numpy().copy()
        probability, state_value = float(log_probability[0]), float(value[0])
        return action, mode_index, latent_axes, probability, state_value


class PilotActorCritic(HybridPilotDistribution):
    """Refine BC control heads while keeping shared encoding and risk weights frozen."""

    # 功能：
    #   复制当前四轴基座，只开放模式、控制轴和价值参数，冻结编码器与风险部分。
    # 输入：
    #   self：普通前馈 actor-critic。
    #   base：带当前实时特征和四轴头的基座。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, base: NumpyLocalPolicyModel):
        super().__init__()
        if not isinstance(base, NumpyLocalPolicyModel):
            raise ValueError("PPO_BASE_MODEL_INVALID")
        base = copy.deepcopy(base)
        if (
            base.action_conditioned
            or base.pilot_control_weight is None
            or base.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
        ):
            raise ValueError("PPO_REQUIRES_CURRENT_CONTINUOUS_BC_POLICY")
        if base.action_bias.shape != (12,):
            raise ValueError("PPO_HYBRID_ACTION_HEAD_INVALID")
        if type(base.input_weight) is not np.ndarray or base.input_weight.ndim != 2:
            raise ValueError("PPO_BASE_MODEL_SHAPE_INVALID")
        hidden = base.input_weight.shape[1]
        if not 8 <= hidden <= 1024:
            raise ValueError("PPO_BASE_MODEL_SHAPE_INVALID")
        shapes = {
            "input_weight": base.input_weight.shape,
            "input_bias": (hidden,),
            "action_weight": (hidden, 12),
            "action_bias": (12,),
            "risk_weight": (hidden, 1),
            "risk_bias": (1,),
            "pilot_control_weight": (hidden, 4),
            "pilot_control_bias": (4,),
        }
        for name, shape in shapes.items():
            value = getattr(base, name)
            if (
                type(value) is not np.ndarray
                or value.shape != shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError("PPO_BASE_WEIGHTS_INVALID:" + name)
        self.base = copy.deepcopy(base)
        self.register_buffer("encoder_weight", torch.tensor(base.input_weight.copy()))
        self.register_buffer("encoder_bias", torch.tensor(base.input_bias.copy()))
        self.mode_weight = nn.Parameter(torch.tensor(base.action_weight[:, 8:].copy()))
        self.mode_bias = nn.Parameter(torch.tensor(base.action_bias[8:].copy()))
        self.pilot_weight = nn.Parameter(torch.tensor(base.pilot_control_weight.copy()))
        self.pilot_bias = nn.Parameter(torch.tensor(base.pilot_control_bias.copy()))
        self.value_weight = nn.Parameter(torch.zeros((len(base.input_bias), 1)))
        self.value_bias = nn.Parameter(torch.zeros(1))

    # 功能：
    #   用冻结编码器计算模式、未压缩控制均值与训练价值。
    # 输入：
    #   self：前馈 actor-critic。
    #   inputs：与基座宽度一致的特征批次。
    # 输出：
    #   modes：模式分数。
    #   axes：四轴均值。
    #   values：标量状态价值。
    def forward(self, inputs):
        hidden = torch.relu(inputs @ self.encoder_weight + self.encoder_bias)
        modes = hidden @ self.mode_weight + self.mode_bias
        axes = hidden @ self.pilot_weight + self.pilot_bias
        values = (hidden @ self.value_weight + self.value_bias).squeeze(-1)
        return modes, axes, values

    # 功能：
    #   只将已微调模式与四轴头复制回基座，不导出训练价值或探索噪声。
    # 输入：
    #   self：当前前馈 actor-critic。
    # 输出：
    #   model：不共享可变参数的部署候选模型。
    def deployment_model(self) -> NumpyLocalPolicyModel:
        weights, bias = self.base.action_weight.copy(), self.base.action_bias.copy()
        weights[:, 8:] = self.mode_weight.detach().numpy()
        bias[8:] = self.mode_bias.detach().numpy()
        model = replace(
            copy.deepcopy(self.base),
            action_weight=weights,
            action_bias=bias,
            pilot_control_weight=self.pilot_weight.detach().numpy().copy(),
            pilot_control_bias=self.pilot_bias.detach().numpy().copy(),
        )
        return model


@dataclass
class Rollout:
    """Frozen-policy proposals, rewards and receipts from simulation collection."""

    inputs: np.ndarray
    modes: np.ndarray
    latent: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    intervention_count: int
    records: list[dict]


class PPOTrainer:
    """Separate physical collection from offline optimization and explicit checkpoint resume."""

    # 功能：
    #   固定单个专家的当前回放、基座、奖励和优化配置，再初始化带独立随机状态的训练器。
    # 输入：
    #   self：新训练器。
    #   base：前馈四轴基座。
    #   replay：有真实控制标签的行为回放。
    #   config：可选优化配置。
    #   reward_config：可选奖励配置。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        base: NumpyLocalPolicyModel,
        replay: list[LocalPolicyTrainingSample],
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
            raise ValueError("PPO_REPLAY_COUNT_INVALID")
        validated = []
        for sample in replay:
            if not isinstance(sample, LocalPolicyTrainingSample):
                raise ValueError("PPO_REPLAY_SAMPLE_INVALID")
            payload = sample.model_dump()
            if sample.pilot_control_limits is not None:
                payload["pilot_control_limits"] = PilotControlLimits(
                    **payload["pilot_control_limits"]
                )
            validated.append(LocalPolicyTrainingSample.model_validate(payload, strict=True))
        replay = validated
        self.actor = PilotActorCritic(base)
        base = self.actor.base
        if not replay or any(
            s.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            for s in replay
        ):
            raise ValueError("PPO_REQUIRES_CURRENT_TRAINING_REPLAY")
        if len({s.navigation_expert_role for s in replay}) != 1:
            raise ValueError("PPO_UPDATES_ONE_NAVIGATION_EXPERT_AT_A_TIME")
        self.expert_role = replay[0].navigation_expert_role
        if any(
            any(s.candidate_mask) or any(v for row in s.candidate_features for v in row)
            for s in replay
        ):
            raise ValueError("PPO_REPLAY_HAS_LEGACY_CANDIDATES")
        self.replay = _arrays(replay)
        if self.replay.pilot_controls is None or np.any(self.replay.targets < 8):
            raise ValueError("PPO_REPLAY_CONTINUOUS_LABELS_REQUIRED")
        if self.replay.inputs.shape[1] != base.input_weight.shape[0]:
            raise ValueError("PPO_REPLAY_INPUT_WIDTH_MISMATCH")
        self._initialize_optimizer()
        self.base_hash = sha256_json(
            {
                name: getattr(base, name).tolist()
                for name in (
                    "input_weight",
                    "input_bias",
                    "action_weight",
                    "action_bias",
                    "risk_weight",
                    "risk_bias",
                    "pilot_control_weight",
                    "pilot_control_bias",
                )
            }
        )
        self.replay_hash = sha256_json([s.model_dump(mode="json") for s in replay])

    # 功能：
    #   设置逐轴探索范围、独立随机数与 Adam，在仿真启动前用另一个随机发生器预热。
    # 输入：
    #   self：已经固定策略与回放的训练器。
    # 输出：
    #   None：不返回业务数据。
    def _initialize_optimizer(self):
        # BC fine-tuning explores locally around the learned controls. A fixed
        # latent sigma near .5 made an almost-neutral vertical command become
        # a large descent. Noise is explicit, per-axis, bounded and checkpoint-
        # bound; it never ships in the deployment model or bypasses the shield.
        with torch.no_grad():
            self.actor.log_std.copy_(
                torch.tensor(self.config.initial_axis_standard_deviation, dtype=torch.float32).log()
            )
            self.actor.axis_log_std_bounds.copy_(
                torch.tensor(
                    [
                        np.log(self.config.minimum_axis_standard_deviation),
                        np.log(self.config.maximum_axis_standard_deviation),
                    ],
                    dtype=torch.float32,
                )
            )
        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=self.config.learning_rate
        )
        self.updates = 0
        self.episode_counter = 0
        # The first distribution/tensor execution can initialize lazy libraries.
        # Warm with actual replay input while no simulator is running, with a
        # separate RNG so the reproducible rollout sequence is untouched.
        self.actor.sample(
            torch.tensor(self.replay.inputs[:1]), torch.Generator().manual_seed(self.config.seed)
        )

    # 功能：
    #   每次重置使用可复现且递增的回合种子，不将截断后的新回合混入上一回合。
    # 输入：
    #   self：持有回合计数的训练器。
    #   env：仿真环境。
    # 输出：
    #   observation：新回合首个观测的独立副本。
    def _reset_environment(self, env):
        if self.config.seed + self.episode_counter > 2**63 - 1:
            raise ValueError("PPO_EPISODE_SEED_EXHAUSTED")
        observation = env.reset(seed=self.config.seed + self.episode_counter)
        self.episode_counter += 1
        observation = detach_evidence(observation)
        return observation

    # 功能：
    #   提取部署一致的 actor 输入，不把独立奖励真值注入策略特征。
    # 输入：
    #   self：当前训练器。
    #   observation：当前传感器观测。
    # 输出：
    #   inputs：控制策略输入向量。
    def _observation_input(self, observation):
        inputs = observation.policy_input()
        return inputs

    # 功能：
    #   绑定当前策略身份进行受保护仿真采集，停稳后再做历史证据哈希。
    # 输入：
    #   self：离线训练器。
    #   env：有停稳能力的仿真或明确合成环境。
    # 输出：
    #   rollout：控制、奖励及来源绑定完整的采集批次。
    def collect(self, env: FlightTrainingEnvironment) -> Rollout:
        with quiesced_simulation_rollout(env):
            bind = getattr(env, "bind_policy_identity", None)
            if callable(bind):
                bind(
                    sha256_json(
                        {
                            "base_sha256": self.base_hash,
                            "updates": self.updates,
                            "actor_state": {
                                name: value.detach().cpu().tolist()
                                for name, value in self.actor.state_dict().items()
                            },
                        }
                    )
                )
            rollout = self._collect_running(env)
        # Canonicalizing all historical input rows is evidence work, not actor
        # inference. It must not consume the next observation's motion lease.
        self._finalize_evidence(rollout)
        return rollout

    # 功能：
    #   在停止仿真后完成原始观测摘要，全部成功才替换记录，失败保留原始证据。
    # 输入：
    #   self：当前训练器。
    #   rollout：尚含原始观测快照的批次。
    # 输出：
    #   None：不返回业务数据。
    def _finalize_evidence(self, rollout):
        records = [detach_evidence(record) for record in rollout.records]
        for record in records:
            record["observation_sha256"] = sha256_json(record.pop("_observation_for_hash"))
        rollout.records = records

    # 功能：
    #   只在仿真中采样和执行提案，分别保存原始提案与实际干预，不伪造监督动作或跨回合引导。
    # 输入：
    #   self：采集期间保持策略冻结的训练器。
    #   env：允许训练的仿真或合成环境。
    # 输出：
    #   rollout：固定策略概率、优势、价值目标和待最终哈希的证据。
    def _collect_running(self, env: FlightTrainingEnvironment) -> Rollout:
        if env.simulation_only is not True or env.evidence_kind not in {
            "px4-gazebo",
            "unit-test",
            "pretraining",
        }:
            raise ValueError("PPO_PHYSICAL_FLIGHT_TRAINING_FORBIDDEN")
        observation = self._reset_environment(env)
        reward = FlightReward(self.reward_config)
        inputs, modes, latent, logs, values, next_values, rewards, terms, truncs, records = (
            [] for _ in range(10)
        )
        interventions = 0
        for rollout_index in range(self.config.rollout_steps):
            if observation.sample.navigation_expert_role != self.expert_role:
                raise ValueError("PPO_ENVIRONMENT_ROUTED_TO_DIFFERENT_EXPERT")
            input_started = time.perf_counter()
            x = self._observation_input(observation)
            input_prepared = time.perf_counter()
            action, mode, z, log_prob, value = self.actor.sample(
                torch.tensor(x[None]), self.generator
            )
            record_timing = getattr(env, "record_actor_timing", None)
            if callable(record_timing):
                record_timing(
                    sequence=observation.sequence,
                    input_preparation_ms=(input_prepared - input_started) * 1000,
                    actor_sampling_ms=(time.perf_counter() - input_prepared) * 1000,
                )
            if rollout_index and not (terms[-1] or truncs[-1]):
                # Consecutive steps share this very observation. The actor is
                # frozen throughout collection, so its current value is exactly
                # the previous step's bootstrap, with no repeated GRU inference.
                next_values.append(value)
            step = env.step(action.model_copy(deep=True))
            if not isinstance(step, FlightStep):
                raise ValueError("PPO_ENVIRONMENT_STEP_INVALID")
            step = FlightStep.model_validate(step.model_dump())
            step.validate_transition(observation, action)
            components = reward.score(step)
            if step.terminated:
                next_values.append(0.0)
            elif step.truncated or rollout_index + 1 == self.config.rollout_steps:
                # A time limit bootstraps its FINAL observation before reset;
                # it cannot borrow the next episode's first state/value.
                with torch.no_grad():
                    next_values.append(
                        float(
                            self.actor(
                                torch.tensor(self._observation_input(step.observation)[None])
                            )[2][0]
                        )
                    )
            inputs.append(x)
            modes.append(mode)
            latent.append(z)
            logs.append(log_prob)
            values.append(value)
            rewards.append(sum(components.values()))
            terms.append(step.terminated)
            truncs.append(step.truncated)
            interventions += int(step.evidence.safety_intervened)
            records.append(
                {
                    "episode_id": observation.episode_id,
                    "sequence": observation.sequence,
                    # Pydantic's dump detaches nested mutable lists now; only
                    # canonical serialization/hashing is deferred until grounded.
                    "_observation_for_hash": observation.model_dump(mode="json"),
                    "proposed_action": action.model_dump(),
                    "applied_action": step.applied_action.model_dump(),
                    "reward_components": components,
                    "applied_command_sha256": step.applied_command_sha256,
                    "verifier_receipt_sha256": step.evidence.verifier_receipt_sha256,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "safety_intervened": step.evidence.safety_intervened,
                    "evidence_kind": env.evidence_kind,
                    "policy_log_probability": log_prob,
                }
            )
            ended = step.terminated or step.truncated
            if ended and rollout_index + 1 < self.config.rollout_steps:
                observation = self._reset_environment(env)
                reward.reset()
            else:
                observation = step.observation
        advantages, returns = generalized_advantages(
            rewards,
            values,
            next_values,
            terms,
            truncs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        rollout = Rollout(
            np.asarray(inputs, dtype=np.float32),
            np.asarray(modes),
            np.asarray(latent),
            np.asarray(logs, dtype=np.float32),
            advantages,
            returns.astype(np.float32),
            interventions,
            records,
        )
        return rollout

    # 功能：
    #   严格检查采集数组后执行 PPO 裁剪更新，保留行为克隆约束，并在 KL 超限时停止更新。
    # 输入：
    #   self：带优化器和固定回放的训练器。
    #   rollout：尚未用于此轮更新的采集数据。
    # 输出：
    #   metrics：更新次数、损失、最大 KL、提前停止及安全干预比例。
    def update(self, rollout: Rollout) -> dict[str, float]:
        c = self.config
        count = len(rollout.inputs)
        arrays = (
            rollout.inputs,
            rollout.modes,
            rollout.latent,
            rollout.old_log_probs,
            rollout.advantages,
            rollout.returns,
        )
        shapes = (
            (count, self.replay.inputs.shape[1]),
            (count,),
            (count, 4),
            (count,),
            (count,),
            (count,),
        )
        if (
            not 2 <= count <= 65536
            or type(rollout.intervention_count) is not int
            or not 0 <= rollout.intervention_count <= count
            or any(
                type(a) is not np.ndarray or a.shape != shape
                for a, shape in zip(arrays, shapes, strict=True)
            )
        ):
            raise ValueError("PPO_ROLLOUT_SHAPE_INVALID")
        if rollout.modes.dtype.kind not in "iu" or not np.isin(rollout.modes, range(4)).all():
            raise ValueError("PPO_ROLLOUT_MODES_INVALID")
        if any(a.dtype.kind != "f" for i, a in enumerate(arrays) if i != 1):
            raise ValueError("PPO_ROLLOUT_DTYPE_INVALID")
        x, modes, z, old, adv, target = (
            torch.tensor(a)
            for a in (
                rollout.inputs,
                rollout.modes,
                rollout.latent,
                rollout.old_log_probs,
                rollout.advantages,
                rollout.returns,
            )
        )
        if len(x) < 2 or any(not torch.isfinite(t).all() for t in (x, z, old, adv, target)):
            raise ValueError("PPO_ROLLOUT_NONFINITE_OR_EMPTY")
        # float64 归一化避免有限 float32 的求和／方差中间溢出，之后恢复模型 dtype。
        adv = adv.double()
        adv = ((adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)).float()
        x, z, old, target, modes = x.float(), z.float(), old.float(), target.float(), modes.long()
        if any(not torch.isfinite(t).all() for t in (x, z, old, adv, target)):
            raise ValueError("PPO_ROLLOUT_FLOAT32_OVERFLOW")
        replay_x = torch.tensor(self.replay.inputs)
        replay_modes = torch.tensor(self.replay.targets - 8)
        replay_axes = torch.tensor(self.replay.pilot_controls)
        stopped, losses, kls = False, [], []
        for _ in range(c.update_epochs):
            order = torch.randperm(len(x), generator=self.generator)
            for indices in order.split(c.minibatch_size):
                log_prob, value, entropy = self.actor.log_probability(
                    x[indices], modes[indices], z[indices]
                )
                log_ratio = log_prob - old[indices]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(kl):
                    raise ValueError("PPO_NONFINITE_POLICY_DIVERGENCE")
                kls.append(float(kl))
                if float(kl) > c.target_kl:
                    stopped = True
                    break
                surrogate = torch.minimum(
                    ratio * adv[indices],
                    ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio) * adv[indices],
                )
                ri = torch.randint(len(replay_x), (len(indices),), generator=self.generator)
                rmode, raxes, _ = self.actor(replay_x[ri])
                retention = torch.nn.functional.cross_entropy(rmode, replay_modes[ri])
                moving = replay_modes[ri] == 3
                if moving.any():
                    retention = retention + torch.nn.functional.mse_loss(
                        torch.tanh(raxes[moving]), replay_axes[ri][moving]
                    )
                loss = (
                    -surrogate.mean()
                    + c.value_weight * (value - target[indices]).square().mean()
                    - c.entropy_weight * entropy.mean()
                    + c.retention_weight * retention
                )
                if not torch.isfinite(loss):
                    raise ValueError("PPO_NONFINITE_LOSS")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), c.gradient_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                if any(
                    not torch.isfinite(parameter).all() for parameter in self.actor.parameters()
                ):
                    raise ValueError("PPO_NONFINITE_UPDATED_WEIGHTS")
                losses.append(float(loss.detach()))
            if stopped:
                break
        self.updates += 1
        metrics = {
            "updates": float(self.updates),
            "minibatch_updates": float(len(losses)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "max_approximate_kl": max(kls, default=0.0),
            "kl_early_stop": float(stopped),
            "safety_intervention_fraction": rollout.intervention_count / len(x),
        }
        return metrics

    # 功能：
    #   独占写入带基座、回放和配置绑定的优化器、随机状态与计数，不覆盖已有模型或检查点。
    # 输入：
    #   self：需要保存的离线训练器。
    #   path：尚不存在的检查点路径。
    # 输出：
    #   None：不返回业务数据。
    def save_checkpoint(self, path: Path) -> None:
        path = Path(path).absolute()
        check_plain_plugin_path(path)
        if path.exists():
            raise FileExistsError(path)
        # Caller chooses a new build-local path; never overwrite base/candidate assets.
        # 先完成有界序列化，避免生成本程序无法恢复的过大检查点或在序列化失败时留下空文件。
        with io.BytesIO() as handle:
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "generator": self.generator.get_state(),
                    "updates": self.updates,
                    "episode_counter": self.episode_counter,
                    "base_hash": self.base_hash,
                    "replay_hash": self.replay_hash,
                    "config": self.config.model_dump(),
                    "reward_config": self.reward_config.model_dump(),
                    "feature_contract": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
                    "refinement_context_sha256": getattr(self, "refinement_context_sha256", None),
                },
                handle,
            )
            if handle.tell() > 64 * 1024 * 1024:
                raise ValueError("PPO_CHECKPOINT_BYTE_BUDGET_EXCEEDED")
            content = handle.getvalue()
        check_plain_plugin_path(path)
        with path.open("xb") as handle:
            if handle.write(content) != len(content):
                raise OSError("PPO_CHECKPOINT_SHORT_WRITE")

    # 功能：
    #   有界读取并核对绑定、冻结权重、优化器和随机状态，全部成功才原子替换活动训练状态。
    # 输入：
    #   self：保持不变直到验证全部通过的训练器。
    #   path：需要恢复的普通检查点文件。
    # 输出：
    #   None：不返回业务数据。
    def restore_checkpoint(self, path: Path) -> None:
        content = read_plugin_file(Path(path).absolute(), limit=64 * 1024 * 1024)
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or (
            payload.get("base_hash") != self.base_hash
            or payload.get("replay_hash") != self.replay_hash
            or payload.get("config") != self.config.model_dump()
            or payload.get("reward_config") != self.reward_config.model_dump()
            or payload.get("feature_contract") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or payload.get("refinement_context_sha256")
            != getattr(self, "refinement_context_sha256", None)
        ):
            raise ValueError("PPO_CHECKPOINT_BINDING_MISMATCH")
        if any(
            type(payload.get(name)) is not int or payload[name] < 0
            for name in ("updates", "episode_counter")
        ):
            raise ValueError("PPO_CHECKPOINT_COUNTER_INVALID")
        # Copy as a pair: optimizer parameters must refer to the staged actor,
        # not to separately deep-copied tensors or the still-active learner.
        actor, optimizer = copy.deepcopy((self.actor, self.optimizer))
        generator = torch.Generator()
        try:
            expected = self.actor.state_dict()
            supplied = payload["actor"]
            trainable = {
                name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
            }
            if (
                not isinstance(supplied, dict)
                or set(supplied) != set(expected)
                or any(
                    not isinstance(supplied[name], torch.Tensor)
                    or supplied[name].shape != value.shape
                    or supplied[name].dtype != value.dtype
                    or (name not in trainable and not torch.equal(supplied[name], value))
                    for name, value in expected.items()
                )
            ):
                raise ValueError("checkpoint altered frozen or incompatible actor state")
            actor.load_state_dict(payload["actor"], strict=True)
            if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
                raise ValueError("nonfinite actor weights")
            optimizer.load_state_dict(payload["optimizer"])
            for current, staged in zip(
                self.optimizer.param_groups, optimizer.param_groups, strict=True
            ):
                if {k: v for k, v in current.items() if k != "params"} != {
                    k: v for k, v in staged.items() if k != "params"
                }:
                    raise ValueError("optimizer configuration changed")
            for parameter, state in optimizer.state.items():
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid Adam state fields")
                for name, value in state.items():
                    if not isinstance(value, torch.Tensor) or (
                        not torch.isfinite(value).all()
                        or value.dtype != torch.float32
                        or (name != "step" and value.shape != parameter.shape)
                        or (
                            name == "step"
                            and (
                                value.numel() != 1
                                or value.item() < 0
                                or value.item() != int(value.item())
                            )
                        )
                        or (name == "exp_avg_sq" and (value < 0).any())
                    ):
                        raise ValueError("invalid optimizer state")
            generator.set_state(payload["generator"])
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            raise ValueError("PPO_CHECKPOINT_STATE_INVALID") from error
        self.actor, self.optimizer, self.generator = actor, optimizer, generator
        self.updates, self.episode_counter = payload["updates"], payload["episode_counter"]
