"""Recurrent PPO configuration ownership and simulation warmup boundaries."""

import pytest
from test_causal_policy import samples
from test_causal_ppo import trainer
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    causal_examples,
)
from dronedream_agent_core.training.causal_ppo import CausalPPOTrainer
from dronedream_agent_core.training.flight_environment import RewardConfig
from dronedream_agent_core.training.ppo import PPOConfig


# 功能：
#   验证调用方后改配置不会改变已初始化训练器的采集或奖励语义。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_trainer_owns_configuration():
    config, reward = PPOConfig(rollout_steps=2), RewardConfig()
    original_reward = reward.model_dump()
    base = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1))
    replay = causal_examples(samples(), stream_groups={"train": "route"}, history_length=4)
    learner = CausalPPOTrainer(base, replay, config, reward)
    config.rollout_steps = 4
    reward.__dict__.clear()
    assert learner.config.rollout_steps == 2
    assert learner.reward_config.model_dump() == original_reward


# 功能：
#   验证被绕过赋值验证修改的零步配置不会构造出表面成功的训练器。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_trainer_revalidates_configuration():
    base = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1))
    replay = causal_examples(samples(), stream_groups={"train": "route"}, history_length=4)
    config = PPOConfig().model_copy(update={"rollout_steps": 0})
    with pytest.raises(ValueError):
        CausalPPOTrainer(base, replay, config)


# 功能：
#   验证环境不能原地改写预热悬停提案后再用同一对象通过动作归属校验。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_warmup_control_cannot_be_rewritten_by_environment():
    class RewritesProposal(UnitEnvironment):
        # 功能：
        #   将收到的悬停改为运动，再返回相应步骤，模拟环境持有了提案别名。
        # 输入：
        #   self：合成单元环境。
        #   action：采集器传入的提案对象。
        # 输出：
        #   step：对应被篡改提案的合成步骤。
        def step(self, action):
            action.mode = "pilot-control"
            action.axes[0] = 0.5
            step = super().step(action)
            return step

    with pytest.raises(ValueError):
        trainer().collect(RewritesProposal())
