"""Synthetic lifecycle/history tests, not physical flight acceptance."""

import pytest
import torch
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.causal_policy import CausalPilotPolicy, CausalPolicyConfig
from dronedream_agent_core.training.causal_student import CausalStudent
from dronedream_agent_core.training.dagger import ProposedActionRisk, TeacherCorrection
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.student_collection import collect_deferred_dagger


class Lifecycle(UnitEnvironment):
    # 功能：
    #   初始化可观测的合成落地状态与重置计数，用于验证教师调用顺序。
    # 输入：
    #   self：测试环境。
    #   broken_stop：是否在停止时模拟落地未确认。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, broken_stop=False):
        self.grounded = True
        self.broken_stop = broken_stop
        self.resets = 0

    # 功能：
    #   记录本次启动并把环境标记为未落地，然后生成合成初始观测。
    # 输入：
    #   self：测试环境。
    #   seed：回合种子。
    # 输出：
    #   observation：带回合身份的合成观测。
    def reset(self, *, seed):
        self.grounded = False
        self.resets += 1
        observation = super().reset(seed=seed)
        return observation

    # 功能：
    #   模拟确认落地或明确失败，供停止前后调用顺序断言使用。
    # 输入：
    #   self：测试环境。
    # 输出：
    #   None：不返回业务数据。
    def quiesce(self):
        if self.broken_stop:
            raise RuntimeError("landing not confirmed")
        self.grounded = True


# 功能：
#   以固定合成动作和独立教师执行一次延迟标注，验证两类策略运行的时机。
# 输入：
#   env：带落地状态的测试环境。
#   held_out：应被拒绝的留出任务。
#   wrong_risk：是否故意使风险回执绑定另一动作。
# 输出：
#   result：四步采集的行为、风险与来源结果。
def collect(env, *, held_out=frozenset(), wrong_risk=False):
    # 功能：
    #   断言学生仅在采集阶段被调用并给出固定前向提案。
    # 输入：
    #   _observation：当前合成观测。
    # 输出：
    #   proposal：前向幅度零点五的动作。
    def student(_observation):
        assert not env.grounded
        proposal = PilotAction(mode="pilot-control", axes=[.5, 0., 0., 0.])
        return proposal

    # 功能：
    #   断言纠正器只在落地后运行，并提供与学生提案不同的教师动作。
    # 输入：
    #   observation：被标注观测。
    # 输出：
    #   correction：绑定该观测的低风险后向动作标签。
    def teacher(observation):
        assert env.grounded
        correction = TeacherCorrection(observation_sha256=sha256_json(observation),
            action=PilotAction(mode="pilot-control", axes=[-.1, 0., 0., 0.]),
            verified_action_risk=.1, verifier_receipt_sha256="f" * 64)
        return correction

    # 功能：
    #   断言独立风险计算发生在落地后，按测试配置绑定正确或错误的学生动作。
    # 输入：
    #   observation：当前待标注观测。
    #   proposed：学生实际提出的控制。
    # 输出：
    #   assessment：带来源和动作摘要的高风险评价。
    def risk(observation, proposed):
        assert env.grounded
        assessment = ProposedActionRisk(observation_sha256=sha256_json(observation),
            proposed_action_sha256="a" * 64 if wrong_risk else sha256_json(proposed),
            risk=.9, source="swept-geometry", verifier_receipt_sha256="e" * 64)
        return assessment

    result = collect_deferred_dagger(env, student, teacher, risk, seed=805, steps=4,
        held_out_missions=set(held_out), student_policy_sha256="d" * 64)
    return result


# 功能：
#   验证教师纠正标签与实际动作独立，学生风险使用真实物理尺度，末步不额外起飞。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teachers_run_only_after_grounding_and_do_not_change_actual_actions():
    env = Lifecycle()
    result = collect(env)
    assert env.grounded and env.resets == 1  # no empty extra flight after final truncation
    assert len(result.behavior_samples) == len(result.records) == 4
    assert all(row.target_pilot_control[0] == -.1 for row in result.behavior_samples)
    assert all(row.risk_target == .9 for row in result.action_risk_samples)
    assert all(row.risk_proposed_control[0] == .5 / 20 for row in result.action_risk_samples)
    assert all(row["applied_action"]["axes"][0] == .5 for row in result.records)
    assert all(not row["teacher_selected"] for row in result.records)


# 功能：
#   验证停止失败向外传播，教师不能在未确认落地时开始标注。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stop_failure_blocks_teacher_annotations():
    with pytest.raises(RuntimeError, match="landing not confirmed"):
        collect(Lifecycle(broken_stop=True))


# 功能：
#   验证学生修改传感器特征时不执行动作，并仍完成环境停止。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_student_cannot_change_observation_before_actuation():
    from dronedream_agent_core.training.student_collection import collect_student_visits

    env = Lifecycle()

    # 功能：
    #   注入学生改写源特征的错误，返回表面合法动作供边界检查拒绝。
    # 输入：
    #   observation：原始观测。
    # 输出：
    #   proposal：零轴保持动作。
    def corrupt(observation):
        observation.sample.state_features[0] += 1
        proposal = PilotAction(mode="hold", axes=[0.] * 4)
        return proposal

    with pytest.raises(ValueError, match="STUDENT_MUTATED_SOURCE"):
        collect_student_visits(env, corrupt, seed=805, steps=1,
                              held_out_missions=set(), student_policy_sha256="d" * 64)
    assert env.sequence == 0 and env.grounded


# 功能：
#   验证教师依据篡改后的输入签名会被拒绝，原始访问记录不受影响。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_offline_teacher_cannot_rewrite_retained_visit():
    from dronedream_agent_core.training.student_collection import (
        collect_student_visits,
        label_student_visits,
    )

    visits = collect_student_visits(Lifecycle(),
        lambda _: PilotAction(mode="hold", axes=[0.] * 4), seed=805, steps=1,
        held_out_missions=set(), student_policy_sha256="d" * 64)
    original = sha256_json(visits[0].observation)

    # 功能：
    #   修改教师收到的副本并绑定其摘要，制造与保留观测不一致的回执。
    # 输入：
    #   observation：教师获得的独立副本。
    # 输出：
    #   correction：故意绑定错误观测的纠正标签。
    def corrupt(observation):
        observation.sample.state_features[0] += 1
        correction = TeacherCorrection(observation_sha256=sha256_json(observation),
            action=PilotAction(mode="hold", axes=[0.] * 4),
            verified_action_risk=0., verifier_receipt_sha256="f" * 64)
        return correction

    with pytest.raises(ValueError, match="TEACHER_WRONG_OBSERVATION"):
        label_student_visits(visits, corrupt, lambda *_: None,
                             held_out_missions=set(), evidence_kind="unit-test")
    assert sha256_json(visits[0].observation) == original


# 功能：
#   验证留出任务和未绑定学生动作的风险都不能形成训练数据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_held_out_state_and_mismatched_risk_never_become_training_data():
    env = Lifecycle()
    with pytest.raises(ValueError, match="HELD_OUT"):
        collect(env, held_out={"unit-mission"})
    assert env.grounded
    with pytest.raises(ValueError, match="COUNTERFACTUAL_BINDING"):
        collect(Lifecycle(), wrong_risk=True)


# 功能：
#   用实际因果网络验证历史不足时保持、历史完整后四轴输出及回合／专家隔离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_causal_student_requires_real_history_and_resets_between_flights():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, encoder_width=32,
                                                    recurrent_width=32, head_width=32))
        with torch.no_grad():
            model.mode_head.weight.zero_()
            model.mode_head.bias.zero_()
            model.mode_head.bias[3] = 5.
            model.axes_head.weight.zero_()
            model.axes_head.bias.copy_(torch.tensor([.2, -.3, .4, -.5]))
        student = CausalStudent(model, expert_role="local-navigation-policy")
        env = UnitEnvironment()
        observation = env.reset(seed=1)
        for _ in range(3):
            assert student(observation).mode == "hold"
            observation = env.step(PilotAction(mode="hold", axes=[0.] * 4)).observation
        action = student(observation)
        assert action.mode == "pilot-control"
        assert action.axes == pytest.approx(torch.tensor([.2, -.3, .4, -.5]).tanh().tolist())
        with pytest.raises(ValueError, match="REPLAYED"):
            student(observation)
        assert student(env.reset(seed=2)).mode == "hold"
        observation = env.step(action).observation
        observation.sample.navigation_expert_role = "recovery-policy"
        with pytest.raises(ValueError, match="EXPERT_ROUTING"):
            student(observation)
    finally:
        torch.set_num_threads(previous_threads)


# 功能：
#   验证保持、扫描与终止不能冒充连续速度提案来训练动作风险头。
# 输入：
#   mode：待检查的非速度控制模式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["hold", "request-new-scan", "abort"])
def test_position_or_stop_modes_cannot_label_velocity_risk(mode):
    from dronedream_agent_core.training.dagger import corrected_samples

    observation = UnitEnvironment().reset(seed=1)
    action = PilotAction(mode=mode, axes=[0.] * 4)
    correction = TeacherCorrection(observation_sha256=sha256_json(observation), action=action,
        verified_action_risk=.1, verifier_receipt_sha256="f" * 64)
    risk = ProposedActionRisk(observation_sha256=sha256_json(observation),
        proposed_action_sha256=sha256_json(action), risk=.1, source="swept-geometry",
        verifier_receipt_sha256="e" * 64)
    with pytest.raises(ValueError, match="REQUIRES_VELOCITY_CONTROL"):
        corrected_samples(observation, action, correction, risk)
