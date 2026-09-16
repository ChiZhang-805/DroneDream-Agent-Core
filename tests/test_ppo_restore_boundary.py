"""A rejected offline resume cannot partially replace the current learner."""

import copy

import pytest
import torch
from test_offline_flight_learning import base, replay

from dronedream_agent_core.training.ppo import PPOConfig, PPOTrainer


# 功能：
#   分别破坏随机数、优化器、权重和计数，验证恢复失败不会部分替换现有训练状态。
# 输入：
#   tmp_path：隔离检查点目录。
#   fault：要触发的恢复故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["rng", "optimizer", "nan", "counter"])
def test_resume_failure_is_atomic_and_rejects_nonfinite_state(tmp_path, fault):
    learner = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    path = tmp_path / "original.pt"
    learner.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["actor"]["pilot_bias"] += 1
    if fault == "rng":
        payload["generator"] = torch.tensor([1, 2], dtype=torch.uint8)
    elif fault == "optimizer":
        payload["optimizer"]["param_groups"] = []
    elif fault == "nan":
        payload["actor"]["pilot_bias"][0] = float("nan")
    else:
        payload["updates"] = True
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    actor = copy.deepcopy(learner.actor.state_dict())
    generator = learner.generator.get_state().clone()
    optimizer = learner.optimizer
    with pytest.raises(ValueError, match="PPO_CHECKPOINT"):
        learner.restore_checkpoint(broken)
    assert all(torch.equal(value, learner.actor.state_dict()[key]) for key, value in actor.items())
    assert torch.equal(generator, learner.generator.get_state())
    assert learner.optimizer is optimizer and learner.updates == 0
