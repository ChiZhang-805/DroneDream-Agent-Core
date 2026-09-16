"""Physics-free tests of recurrent PPO; no flight qualification is implied."""

from dataclasses import replace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from test_causal_policy import samples
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.causal_control import CONTROL_HISTORY_WIDTH
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    causal_examples,
    export_causal_policy,
    load_causal_checkpoint,
    save_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_ppo import CausalPPOTrainer
from dronedream_agent_core.training.ppo import PPOConfig


# 功能：
#   将本模块测试的 Torch 线程数限制为一，并在成功或失败后恢复原值。
# 输入：
#   无。
# 输出：
#   None：向测试交出执行权，不返回业务数据。
@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# 功能：
#   构造固定随机种子、四帧真实历史和八步采集的小型循环 PPO 测试训练器。
# 输入：
#   无。
# 输出：
#   learner：只用于合成环境验证的训练器。
def trainer():
    torch.manual_seed(805)
    base = CausalPilotPolicy(
        CausalPolicyConfig(
            history_length=4, encoder_width=32, recurrent_width=32, head_width=32, epochs=1
        )
    )
    with torch.no_grad():
        base.mode_head.bias[3] = 4.0
    examples = causal_examples(samples(), stream_groups={"train": "train-route"}, history_length=4)
    learner = CausalPPOTrainer(
        base, examples, PPOConfig(rollout_steps=8, update_epochs=2, minibatch_size=4)
    )
    return learner


# 功能：
#   验证 PPO 改变部署四轴头但冻结共享风险表示，并核对实际 ONNX 和 actor 四轴输出。
# 输入：
#   tmp_path：测试模型导出目录。
# 输出：
#   None：不返回业务数据。
def test_causal_ppo_updates_deployed_axes_without_changing_shared_risk(tmp_path):
    learner = trainer()
    before = {k: v.clone() for k, v in learner.actor.policy.state_dict().items()}
    rollout = learner.collect(UnitEnvironment())
    assert len(rollout.records) == 8
    assert len(learner.warmup_records) == 24  # three physical holds for each short episode
    assert all(r["proposed_action"]["mode"] == "hold" for r in learner.warmup_records)
    assert all(r["sequence"] == 3 for r in rollout.records)
    metrics = learner.update(rollout)
    assert metrics["minibatch_updates"] > 0
    deployed = learner.actor.deployment_model()
    assert not torch.equal(deployed.axes_head.weight, before["axes_head.weight"])
    for name, weight in deployed.state_dict().items():
        if not name.startswith(("mode_head.", "axes_head.")):
            assert torch.equal(weight, before[name]), name
    path = tmp_path / "recurrent.onnx"
    export_causal_policy(deployed, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    columns = list(torch.split(torch.tensor(rollout.inputs[:1]), learner.actor.widths, dim=-1))
    columns[3] = columns[3].reshape(1, 4, CONTROL_HISTORY_WIDTH)
    feeds = {i.name: v.numpy() for i, v in zip(session.get_inputs(), columns, strict=True)}
    actual = session.run(["pilot_control"], feeds)[0]
    with torch.no_grad():
        expected = learner.actor(torch.tensor(rollout.inputs[:1]))[1].tanh().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


# 功能：
#   验证恢复后采样、证据和优化一致，且纯策略检查点完整保留部署参数而不覆盖已有文件。
# 输入：
#   tmp_path：隔离检查点目录。
# 输出：
#   None：不返回业务数据。
def test_causal_checkpoint_preserves_optimizer_rng_and_history_reset(tmp_path):
    first = trainer()
    first.update(first.collect(UnitEnvironment()))
    checkpoint = tmp_path / "ppo.pt"
    first.save_checkpoint(checkpoint)
    second = trainer()
    second.restore_checkpoint(checkpoint)
    a, b = first.collect(UnitEnvironment()), second.collect(UnitEnvironment())
    np.testing.assert_array_equal(a.latent, b.latent)
    assert a.records == b.records
    assert first.update(a) == second.update(b)
    bc = tmp_path / "policy.pt"
    save_causal_checkpoint(first.actor.deployment_model(), bc)
    loaded = load_causal_checkpoint(bc)
    assert all(
        torch.equal(v, loaded.state_dict()[k]) for k, v in first.actor.policy.state_dict().items()
    )
    with pytest.raises(FileExistsError):
        save_causal_checkpoint(loaded, bc)


# 功能：
#   验证回合在预热期间结束时明确失败，不能通过虚构历史继续训练。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_fabricated_history_if_episode_ends_during_warmup():
    class ShortEpisode(UnitEnvironment):
        # 功能：
        #   将合成环境的第一步标记为时间截断，触发预热期间回合结束边界。
        # 输入：
        #   self：短回合环境。
        #   action：待执行的控制。
        # 输出：
        #   step：带截断标记的步骤副本。
        def step(self, action):
            step = super().step(action).model_copy(update={"truncated": True})
            return step

    with pytest.raises(ValueError, match="ENDED_DURING_HISTORY_WARMUP"):
        trainer().collect(ShortEpisode())


# 功能：
#   验证训练与留出的历史早期帧来源重叠也被拒绝，不能只检查当前帧。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_validation_leak_in_earlier_history_frame_is_rejected():
    train = causal_examples(samples(), stream_groups={"train": "a"}, history_length=4)
    val = causal_examples(samples(100, "val"), stream_groups={"val": "b"}, history_length=4)
    val[0] = replace(val[0], source_sha256=(train[0].source_sha256[0], *val[0].source_sha256[1:]))
    with pytest.raises(ValueError, match="SOURCE_LEAKAGE"):
        train_causal_policy(train, val, CausalPolicyConfig(history_length=4, epochs=1))


# 功能：
#   验证策略检查点中的非有限权重被拒绝，不可进入后续微调或导出。
# 输入：
#   tmp_path：隔离的合法和损坏检查点目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_rejects_nonfinite_or_wrong_identity(tmp_path):
    model = trainer().actor.deployment_model()
    valid = tmp_path / "valid.pt"
    save_causal_checkpoint(model, valid)
    payload = torch.load(valid, weights_only=True)
    payload["state_dict"]["axes_head.bias"][0] = float("nan")
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="WEIGHTS_INVALID"):
        load_causal_checkpoint(bad)
