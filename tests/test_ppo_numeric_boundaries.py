"""Offline PPO numerical, action-ownership and checkpoint trust boundaries."""

import copy

import numpy as np
import pytest
import torch
from test_offline_flight_learning import UnitEnvironment, base, replay

from dronedream_agent_core.training.ppo import PPOConfig, PPOTrainer, generalized_advantages


# 功能：
#   验证优势估计拒绝 float32 溢出和布尔折扣，不能输出不可训练的数值。
# 输入：
#   fault：损坏回报或折扣的方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["overflow", "boolean"])
def test_gae_rejects_overflow_and_boolean_discount(fault):
    with pytest.raises(ValueError):
        generalized_advantages(
            [1e39 if fault == "overflow" else 1.0],
            [0.0],
            [0.0],
            [True],
            [False],
            gamma=True if fault == "boolean" else 0.99,
            gae_lambda=0.95,
        )


# 功能：
#   验证普通 PPO 与循环 PPO 一样独立持有配置，不让外部写入改变采集语义。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ppo_owns_configuration():
    config = PPOConfig(rollout_steps=2)
    learner = PPOTrainer(base(), replay(), config)
    config.rollout_steps = 8
    assert learner.config.rollout_steps == 2


# 功能：
#   验证检查点不能改变冻结编码器或恢复负的 Adam 二阶矩，拒绝时原训练器保持不变。
# 输入：
#   tmp_path：隔离检查点目录。
#   fault：冻结参数或优化器状态的破坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["frozen", "negative_variance"])
def test_restore_rejects_invalid_frozen_or_optimizer_state(tmp_path, fault):
    learner = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2, update_epochs=1))
    learner.update(learner.collect(UnitEnvironment()))
    source, broken = tmp_path / "source.pt", tmp_path / "broken.pt"
    learner.save_checkpoint(source)
    payload = torch.load(source, weights_only=True)
    if fault == "frozen":
        payload["actor"]["encoder_weight"][0, 0] += 1
    else:
        for state in payload["optimizer"]["state"].values():
            state["exp_avg_sq"].fill_(-1)
    torch.save(payload, broken)
    actor, optimizer = learner.actor, learner.optimizer
    weights = copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError):
        learner.restore_checkpoint(broken)
    assert learner.actor is actor and learner.optimizer is optimizer
    assert all(
        torch.equal(value, learner.actor.state_dict()[name]) for name, value in weights.items()
    )


# 功能：
#   验证采集时环境不能改写原提案，再将相同对象作为合法动作返回。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_collect_rejects_rewritten_proposal():
    class RewritingEnvironment(UnitEnvironment):
        # 功能：
        #   修改传入提案的轴幅度，制造与原始采样控制不一致的执行回执。
        # 输入：
        #   self：合成环境。
        #   action：收到的控制提案。
        # 输出：
        #   step：对应篡改提案的合成步骤。
        def step(self, action):
            action.mode = "pilot-control"
            action.axes[0] = -0.9
            step = super().step(action)
            return step

    learner = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    with pytest.raises(ValueError):
        learner.collect(RewritingEnvironment())


# 功能：
#   验证二维价值目标不能通过广播改变损失含义，错误批次不得更新模型。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_update_rejects_broadcast_value_targets():
    learner = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    rollout = learner.collect(UnitEnvironment())
    rollout.returns = np.asarray(rollout.returns).reshape(-1, 1)
    with pytest.raises(ValueError):
        learner.update(rollout)
    assert learner.updates == 0
