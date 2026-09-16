"""Synthetic physics-free algorithm tests; never flight acceptance evidence."""

import numpy as np
import pytest
import torch
from test_action_conditioned_risk import _samples
from test_temporal_evidence import evidence

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    _new_model,
    export_local_policy_onnx,
    load_local_policy_onnx,
)
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from dronedream_agent_core.training.dagger import (
    ProposedActionRisk,
    TeacherCorrection,
    collect_dagger,
)
from dronedream_agent_core.training.flight_environment import (
    FlightObservation,
    FlightReward,
    FlightStep,
    PilotAction,
    RewardEvidence,
    SafetyPositionControl,
)
from dronedream_agent_core.training.ppo import PPOConfig, PPOTrainer, generalized_advantages


# 功能：
#   生成当前四轴契约、真实形状特征和显式时间来源的离线合成样本。
# 输入：
#   无。
# 输出：
#   samples：带控制目标和物理限额的训练样本列表。
def replay():
    samples = [
        sample.model_copy(
            update={
                "target_action_index": 11,
                "target_pilot_control": [0.3, 0.0, 0.0, -0.2],
                "risk_proposed_control": [],
                "temporal_evidence": evidence(i),
                "pilot_control_limits": PilotControlLimits(1.0, 1.0, 20.0),
            }
        )
        for i, sample in enumerate(_samples())
    ]
    return samples


# 功能：
#   创建小型合成基座并偏置为连续控制模式，供实际训练与导出测试使用。
# 输入：
#   无。
# 输出：
#   model：保留实时特征与四轴头的测试基座。
def base():
    model = _new_model(
        LocalPolicyTrainingConfig(hidden_feature_count=8),
        realtime_feature_count=POLICY_REALTIME_FEATURE_COUNT,
        include_pilot_control=True,
    )
    model.action_bias[11] = 4.0
    return model


class UnitEnvironment:
    simulation_only = True
    evidence_kind = "unit-test"

    # 功能：
    #   用种子区分合成回合，清空步序号后取得初始观测。
    # 输入：
    #   self：无物理引擎的单元环境。
    #   seed：回合种子。
    # 输出：
    #   observation：序号为零的观测。
    def reset(self, *, seed):
        self.seed, self.sequence = seed, 0
        observation = self.observation()
        return observation

    # 功能：
    #   按当前回合与序号构造更新的合成来源，不把夹具数据标成物理证据。
    # 输入：
    #   self：持有当前种子与序号的单元环境。
    # 输出：
    #   observation：当前契约下的合成训练观测。
    def observation(self):
        sample = replay()[0].model_copy(
            update={"temporal_evidence": evidence(self.sequence, stream=f"episode-{self.seed}")}
        )
        observation = FlightObservation(
            mission_id="unit-mission",
            episode_id=f"episode-{self.seed}",
            map_sha256="a" * 64,
            sequence=self.sequence,
            sample=sample,
        )
        return observation

    # 功能：
    #   模拟提案接纳与序号推进，生成固定格式奖励见证，并在第四步模拟时间截断。
    # 输入：
    #   self：单元环境。
    #   action：需要关联到结果摘要的测试动作。
    # 输出：
    #   step：纯合成的步骤结果。
    def step(self, action):
        self.sequence += 1
        step = FlightStep(
            observation=self.observation(),
            proposed_action_sha256=sha256_json(action),
            applied_action=action,
            applied_command_sha256="c" * 64,
            evidence=RewardEvidence(
                stage_id="office",
                goal_revision="original",
                elapsed_seconds=0.05,
                goal_distance_m=10 - self.sequence * 0.05,
                power_joules=0.1,
                commanded_change_squared=0.0,
                safety_intervened=False,
                verifier_receipt_sha256="b" * 64,
            ),
            terminated=False,
            truncated=self.sequence == 4,
        )
        return step

    # 功能：
    #   提供单元环境关闭接口；该夹具没有需要释放的原生资源。
    # 输入：
    #   self：单元环境。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        pass


# 功能：
#   验证优势估计允许时间截断处价值引导，但不能跨重置回合传播下一步回报。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_gae_bootstraps_final_time_limit_but_never_next_reset_episode():
    advantage, returns = generalized_advantages(
        [1.0, 2.0, 100.0],
        [3.0, 4.0, 8.0],
        [4.0, 5.0, 9.0],
        [False, False, True],
        [False, True, False],
        gamma=0.9,
        gae_lambda=0.8,
    )
    assert advantage[1] == pytest.approx(2 + 0.9 * 5 - 4)
    assert advantage[0] == pytest.approx(1 + 0.9 * 4 - 3 + 0.9 * 0.8 * advantage[1])
    assert returns[2] == pytest.approx(100)


# 功能：
#   执行实际 PPO 更新与 ONNX 推理，验证控制头改变而编码器和风险头保持不变。
# 输入：
#   tmp_path：测试模型导出目录。
# 输出：
#   None：不返回业务数据。
def test_ppo_changes_real_exported_control_weights_and_preserves_risk_encoder(tmp_path):
    initial = base()
    trainer = PPOTrainer(
        initial, replay(), PPOConfig(rollout_steps=8, update_epochs=2, minibatch_size=4)
    )
    rollout = trainer.collect(UnitEnvironment())
    assert len(rollout.records) == 8
    metrics = trainer.update(rollout)
    changed = trainer.actor.deployment_model()
    assert metrics["minibatch_updates"] > 0
    assert not np.array_equal(changed.pilot_control_weight, initial.pilot_control_weight)
    assert np.array_equal(changed.input_weight, initial.input_weight)
    assert np.array_equal(changed.risk_weight, initial.risk_weight)
    path = tmp_path / "refined.onnx"
    export_local_policy_onnx(changed, path)
    loaded = load_local_policy_onnx(path)
    assert np.array_equal(loaded.pilot_control_weight, changed.pilot_control_weight)
    # Compare all deployed heads against the actual ORT graph.
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample = replay()[0]
    inputs = {
        "state_features": np.asarray([sample.state_features], np.float32),
        "candidate_features": np.asarray([sample.candidate_features], np.float32),
        "candidate_mask": np.asarray([sample.candidate_mask], np.float32),
        "realtime_features": np.asarray([sample.realtime_features], np.float32),
        "realtime_valid_mask": np.asarray([sample.realtime_valid_mask], np.float32),
    }
    actual = session.run(["pilot_control"], inputs)[0]
    with torch.no_grad():
        expected = trainer.actor(torch.tensor(rollout.inputs[:1]))[1].tanh().numpy()
    assert actual == pytest.approx(expected, abs=1e-6)


# 功能：
#   验证检查点恢复优化器与随机状态，重复训练可复现且拒绝覆盖或错绑定配置。
# 输入：
#   tmp_path：检查点目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_restores_optimizer_rng_and_training_binding(tmp_path):
    model, config = base(), PPOConfig(rollout_steps=4, update_epochs=1)
    first = PPOTrainer(model, replay(), config)
    first.update(first.collect(UnitEnvironment()))
    checkpoint = tmp_path / "checkpoint.pt"
    first.save_checkpoint(checkpoint)
    second = PPOTrainer(model, replay(), config)
    second.restore_checkpoint(checkpoint)
    x, y = first.collect(UnitEnvironment()), second.collect(UnitEnvironment())
    assert np.array_equal(x.latent, y.latent) and x.records == y.records
    assert first.update(x) == second.update(y)
    with pytest.raises(FileExistsError):
        first.save_checkpoint(checkpoint)
    wrong = PPOTrainer(model, replay(), PPOConfig(rollout_steps=4, seed=18))
    with pytest.raises(ValueError, match="BINDING"):
        wrong.restore_checkpoint(checkpoint)


# 功能：
#   验证过大策略散度停止后续批次，而非有限散度明确失败。
# 输入：
#   monkeypatch：策略概率替换器。
#   shift：注入到后续概率计算的偏移。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("shift", [1., float("inf")])
def test_ppo_reports_rejected_minibatch_divergence_or_fails_on_nonfinite(monkeypatch, shift):
    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=4, minibatch_size=2))
    rollout = trainer.collect(UnitEnvironment())
    original = trainer.actor.log_probability
    calls = 0

    # 功能：
    #   保持首轮旧策略概率，随后注入偏移以构造散度故障。
    # 输入：
    #   args：概率计算所需的张量参数。
    # 输出：
    #   result：偏移后的概率、价值与熵三元组。
    def shifted(*args):
        nonlocal calls
        calls += 1
        probability, value, entropy = original(*args)
        result = probability + (shift if calls > 1 else 0.), value, entropy
        return result

    monkeypatch.setattr(trainer.actor, "log_probability", shifted)
    if not np.isfinite(shift):
        with pytest.raises(ValueError, match="NONFINITE_POLICY_DIVERGENCE"):
            trainer.update(rollout)
        return
    metrics = trainer.update(rollout)
    assert metrics["minibatch_updates"] == 1
    assert metrics["kl_early_stop"] == 1
    assert metrics["max_approximate_kl"] > trainer.config.target_kl


# 功能：
#   验证安全保持替换被记录为干预，训练仍保留原始提案而不篡改其概率对应动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ppo_never_relabels_applied_action_as_proposal():
    class Shielded(UnitEnvironment):
        # 功能：
        #   将合成实际动作替换为安全保持，并显式报告干预。
        # 输入：
        #   self：带安全替换的单元环境。
        #   action：原始学生提案。
        # 输出：
        #   result：保留提案身份但实际执行保持的步骤。
        def step(self, action):
            step = super().step(action)
            result = FlightStep.model_validate(
                {
                    **step.model_dump(),
                    "applied_action": PilotAction(mode="hold", axes=[0.0] * 4).model_dump(),
                    "evidence": {**step.evidence.model_dump(), "safety_intervened": True},
                }
            )
            return result

    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=8))
    rollout = trainer.collect(Shielded())
    assert rollout.intervention_count == 8
    assert any(r["proposed_action"] != r["applied_action"] for r in rollout.records)
    for r in rollout.records:
        assert r["reward_components"]["intervention"] < 0


# 功能：
#   验证位置与速度前馈式安全控制保留独立类型，不能被错误反推成学生四轴标签。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ppo_records_position_safety_intervention_but_trains_original_proposal():
    class Shielded(UnitEnvironment):
        # 功能：
        #   构造位置与速度前馈的安全替换，先证明未报告干预会失败再补齐标记。
        # 输入：
        #   self：安全替换夹具。
        #   action：原始提案。
        # 输出：
        #   result：明确标记安全替换后的有效步骤。
        def step(self, action):
            step = super().step(action)
            applied = SafetyPositionControl(position_ned_m=(1., 2., 3.),
                velocity_feedforward_ned_mps=(.1, -.1, 0.), yaw_heading_deg=22.,
                safety_action="replan")
            payload = {**step.model_dump(), "applied_action": applied.model_dump()}
            with pytest.raises(ValueError, match="UNREPORTED_CONTROL_INTERVENTION"):
                FlightStep.model_validate(payload)
            payload["evidence"]["safety_intervened"] = True
            result = FlightStep.model_validate(payload)
            return result

    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=4))
    rollout = trainer.collect(Shielded())
    assert rollout.intervention_count == 4
    for record, mode in zip(rollout.records, rollout.modes, strict=True):
        assert record["applied_action"]["mode"] == "safety-position-control"
        assert "axes" not in record["applied_action"]
        assert record["proposed_action"]["mode"] != "safety-position-control"
        assert mode in range(4)
        assert record["reward_components"]["intervention"] < 0
    assert trainer.update(rollout)["minibatch_updates"] > 0


# 功能：
#   验证反复远离靠近、目标切换与提前落地不能反复获得正奖励。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rewards_cannot_farm_oscillations_goal_changes_or_early_landing():
    env, reward = UnitEnvironment(), FlightReward()
    env.reset(seed=1)
    action = PilotAction(mode="hold", axes=[0.0] * 4)
    original = env.step(action)

    # 功能：
    #   在固定合成步骤上改变几何或终止见证，隔离测试奖励账本行为。
    # 输入：
    #   remaining：当前目标剩余距离。
    #   updates：步骤标志或见证字段的覆盖值。
    # 输出：
    #   result：本次步骤的分项奖励。
    def scored(remaining, **updates):
        result = reward.score(
            FlightStep.model_validate(
                {
                    **original.model_dump(),
                    **updates,
                    "evidence": {
                        **original.evidence.model_dump(),
                        "goal_distance_m": remaining,
                        **updates.pop("evidence", {}),
                    },
                }
            )
        )
        return result

    assert scored(10)["progress"] == 0
    assert scored(9)["progress"] == 1
    assert scored(10)["progress"] == 0
    assert scored(9)["progress"] == 0
    assert scored(0, evidence={"goal_revision": "amended"})["progress"] == 0
    failed = scored(
        0, terminated=True, evidence={"premature_landing": True, "verified_stage_event_id": "event"}
    )
    assert failed["progress"] == failed["stage"] == failed["mission"] == 0
    assert failed["failure"] < 0
    with pytest.raises(ValueError, match="AFTER_EPISODE_END"):
        scored(0)


# 功能：
#   验证 DAgger 区分教师纠正、学生提案风险和实际执行，并禁止留出任务参与采集。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dagger_records_student_states_teacher_correction_and_separate_risk():
    # 功能：
    #   给出与教师相反方向的固定学生提案，便于核对标签归属。
    # 输入：
    #   observation：当前合成观测。
    # 输出：
    #   proposal：前向幅度零点八的动作。
    def student(observation):
        proposal = PilotAction(mode="pilot-control", axes=[0.8, 0, 0, 0])
        return proposal

    # 功能：
    #   为同一观测提供反向低风险纠正动作，不声称该动作已经执行。
    # 输入：
    #   observation：待纠正观测。
    # 输出：
    #   correction：具有观测和验证回执摘要的教师标签。
    def teacher(observation):
        correction = TeacherCorrection(
            observation_sha256=sha256_json(observation),
            action=PilotAction(mode="pilot-control", axes=[-0.2, 0, 0, 0]),
            verified_action_risk=0.05,
            verifier_receipt_sha256="f" * 64,
        )
        return correction

    # 功能：
    #   独立给学生原始提案绑定高风险评价，不能混用教师动作的风险。
    # 输入：
    #   observation：该提案的源观测。
    #   action：学生原始动作。
    # 输出：
    #   assessment：带双重来源摘要的风险评价。
    def assessor(observation, action):
        assessment = ProposedActionRisk(
            observation_sha256=sha256_json(observation),
            proposed_action_sha256=sha256_json(action),
            risk=0.9,
            source="swept-geometry",
            verifier_receipt_sha256="e" * 64,
        )
        return assessment

    result = collect_dagger(
        UnitEnvironment(),
        student,
        teacher,
        assessor,
        seed=17,
        steps=4,
        teacher_probability=0.0,
        held_out_missions=set(),
    )
    assert len(result.behavior_samples) == len(result.action_risk_samples) == 4
    assert result.behavior_samples[0].target_pilot_control[0] == -0.2
    assert result.action_risk_samples[0].risk_proposed_control[0] == pytest.approx(0.8 / 20)
    assert result.action_risk_samples[0].risk_target == 0.9
    assert result.behavior_samples[0].risk_target == 0.05
    assert result.records[0]["applied_action"]["axes"][0] == 0.8
    with pytest.raises(ValueError, match="HELD_OUT"):
        collect_dagger(
            UnitEnvironment(),
            student,
            teacher,
            assessor,
            seed=17,
            steps=1,
            teacher_probability=1.0,
            held_out_missions={"unit-mission"},
        )


# 功能：
#   验证真实飞行训练禁用，并拒绝使用与当前专家角色不匹配的回放样本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_real_flight_training_and_wrong_expert_replay_are_rejected():
    trainer = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    env = UnitEnvironment()
    env.simulation_only = False
    with pytest.raises(ValueError, match="PHYSICAL_FLIGHT"):
        trainer.collect(env)
    data = replay()
    data[0] = data[0].model_copy(update={"navigation_expert_role": "recovery-policy"})
    with pytest.raises(ValueError, match="ONE_NAVIGATION_EXPERT"):
        PPOTrainer(base(), data)
