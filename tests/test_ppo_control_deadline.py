"""Physics-free equivalence and scheduling checks, not flight acceptance."""

import numpy as np
import pytest
import torch
from test_offline_flight_learning import UnitEnvironment, base, replay

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training import ppo
from dronedream_agent_core.training.ppo import PPOConfig, PPOTrainer, generalized_advantages


# 功能：
#   核对一次采样只运行一次网络，同时与显式概率重算保持一致。
# 输入：
#   monkeypatch：包装前向调用以记录调用次数。
# 输出：
#   None：不返回业务数据。
def test_sampling_evaluates_model_once_and_preserves_exact_policy_probability(monkeypatch):
    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    original, calls = trainer.actor.forward, []

    # 功能：
    #   记录前向调用后委托真实网络，不替换模型的计算结果。
    # 输入：
    #   inputs：真实输入张量。
    # 输出：
    #   outputs：模式、轴均值和价值。
    def forward(inputs):
        calls.append(inputs)
        outputs = original(inputs)
        return outputs

    monkeypatch.setattr(trainer.actor, "forward", forward)
    inputs = torch.tensor(trainer.replay.inputs[:1])
    action, mode, latent, probability, value = trainer.actor.sample(inputs, trainer.generator)
    assert len(calls) == 1
    expected, expected_value, _ = trainer.actor.log_probability(
        inputs, torch.tensor([mode]), torch.tensor(latent[None])
    )
    assert probability == pytest.approx(float(expected[0].detach()), abs=1e-7)
    assert value == pytest.approx(float(expected_value[0].detach()), abs=1e-7)
    assert action.mode in {"hold", "request-new-scan", "abort", "pilot-control"}


# 功能：
#   验证逐轴探索上下界生效，且调整采样噪声不会直接改动部署基座的控制均值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_local_exploration_is_axis_specific_bounded_and_does_not_change_bc_mean():
    model = base()
    config = PPOConfig(rollout_steps=2, initial_axis_standard_deviation=[0.08, 0.06, 0.03, 0.04])
    trainer = PPOTrainer(model, replay(), config)
    np.testing.assert_allclose(
        trainer.actor.axis_standard_deviation().detach(), [0.08, 0.06, 0.03, 0.04], rtol=1e-6
    )
    np.testing.assert_array_equal(
        trainer.actor.deployment_model().pilot_control_weight, model.pilot_control_weight
    )
    np.testing.assert_array_equal(
        trainer.actor.deployment_model().pilot_control_bias, model.pilot_control_bias
    )
    with torch.no_grad():
        trainer.actor.log_std.fill_(50.0)
    np.testing.assert_allclose(trainer.actor.axis_standard_deviation().detach(), [0.25] * 4)
    with torch.no_grad():
        trainer.actor.log_std.fill_(-50.0)
    np.testing.assert_allclose(trainer.actor.axis_standard_deviation().detach(), [0.01] * 4)
    with pytest.raises(ValueError, match="EXPLORATION_STANDARD_DEVIATION_OUTSIDE_ENVELOPE"):
        PPOConfig(initial_axis_standard_deviation=[0.1, 0.1, 0.5, 0.1])
    with pytest.raises(ValueError):
        PPOConfig(initial_axis_standard_deviation=[0.1, 0.1, float("nan"), 0.1])


# 功能：
#   核对连续步骤复用下一状态价值，而截断处必须额外计算真实终点，不能借用重置状态。
# 输入：
#   monkeypatch：注入可追踪状态序号的合成价值网络。
# 输出：
#   None：不返回业务数据。
def test_bootstrap_reuses_next_value_but_time_limit_uses_final_state(monkeypatch):
    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=6))
    calls = []

    # 功能：
    #   记录调用序号并将输入首列作为价值，便于逐步检查引导来源。
    # 输入：
    #   inputs：只含合成序号的批次。
    # 输出：
    #   modes：中性模式分数。
    #   axes：中性四轴均值。
    #   values：当前序号对应的价值。
    def forward(inputs):
        calls.extend(inputs[:, 0].tolist())
        modes, axes = torch.zeros((len(inputs), 4)), torch.zeros((len(inputs), 4))
        values = inputs[:, 0]
        return modes, axes, values

    monkeypatch.setattr(trainer.actor, "forward", forward)
    monkeypatch.setattr(
        trainer,
        "_observation_input",
        lambda observation: np.array([observation.sequence], np.float32),
    )
    rollout = trainer.collect(UnitEnvironment())
    assert calls == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0]
    expected, returns = generalized_advantages(
        [sum(r["reward_components"].values()) for r in rollout.records],
        [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        [False] * 6,
        [False, False, False, True, False, False],
        gamma=trainer.config.gamma,
        gae_lambda=trainer.config.gae_lambda,
    )
    np.testing.assert_array_equal(rollout.advantages, expected)
    np.testing.assert_array_equal(rollout.returns, returns)


# 功能：
#   验证大观测哈希在停稳后执行，且停稳时修改原对象不会改变此前采集的证据。
# 输入：
#   monkeypatch：检查哈希调用时机的替换工具。
# 输出：
#   None：不返回业务数据。
def test_large_observation_hashing_waits_for_quiescence_and_snapshot_is_detached(monkeypatch):
    class Environment(UnitEnvironment):
        # 功能：
        #   初始化停止标志、观测引用和预期摘要列表。
        # 输入：
        #   self：合成环境实例。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self):
            self.stopped = False
            self.observations, self.expected_hashes = [], []

        # 功能：
        #   记录本次原始观测及其当时的摘要，作为后续别名隔离断言的对照。
        # 输入：
        #   self：当前测试环境。
        # 输出：
        #   observation：父类产生的真实形状合成观测。
        def observation(self):
            observation = super().observation()
            self.observations.append(observation)
            self.expected_hashes.append(sha256_json(observation))
            return observation

        # 功能：
        #   停止后故意修改环境仍持有的原始观测，用于验证采集器已经独立持有证据。
        # 输入：
        #   self：持有历史观测引用的环境。
        # 输出：
        #   None：不返回业务数据。
        def quiesce(self):
            self.stopped = True
            for observation in self.observations:
                observation.sample.state_features[0] = 99.0

    env = Environment()

    # 功能：
    #   只对大型观测检查停止标志，其余策略身份哈希仍允许在采集前计算。
    # 输入：
    #   value：待哈希的数据。
    # 输出：
    #   digest：真实共享哈希函数返回的摘要。
    def guarded_hash(value):
        if isinstance(value, dict) and "mission_id" in value and "sample" in value:
            assert env.stopped
        digest = sha256_json(value)
        return digest

    monkeypatch.setattr(ppo, "sha256_json", guarded_hash)
    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    rollout = trainer.collect(env)
    assert [r["observation_sha256"] for r in rollout.records] == env.expected_hashes[:2]
    assert all("_observation_for_hash" not in r for r in rollout.records)
