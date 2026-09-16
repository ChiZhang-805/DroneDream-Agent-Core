"""Mutable callback boundaries for nominal, unexecuted risk probes."""

import pytest
from test_native_corrections import fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.action_risk_dataset import counterfactual_risk_samples
from dronedream_agent_core.training.dagger import ProposedActionRisk
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   为测试观测与动作构造摘要一致的名义风险回执，不调用飞控。
# 输入：
#   observation：测试观测。
#   action：未实际执行的动作探针。
# 输出：
#   risk：测试专用的名义风险回执。
def assessment(observation, action):
    risk = ProposedActionRisk(observation_sha256=sha256_json(observation),
                             proposed_action_sha256=sha256_json(action), risk=0.2,
                             source="swept-geometry", verifier_receipt_sha256="a" * 64)
    return risk


# 功能：
#   验证评估回调原地修改其输入时，不会改写调用方观测、原动作或生成标签。
# 输入：
#   tmp_path：合成原生回合目录。
#   fault：故意修改观测身份或动作幅度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["observation", "action"])
def test_assessor_cannot_mutate_source_or_action(tmp_path, fault):
    _, observation, _, _ = fixture(tmp_path)
    action = PilotAction(mode="pilot-control", axes=[0.5, 0.0, 0.0, 0.0])
    before = sha256_json(observation)

    # 功能：
    #   返回原输入的合法评估，同时故意原地改写回调收到的对象。
    # 输入：
    #   received：评估器收到的观测。
    #   proposed：评估器收到的动作。
    # 输出：
    #   risk：改写前身份对应的风险记录。
    def mutating_assessor(received, proposed):
        risk = assessment(received, proposed)
        if fault == "observation":
            received.episode_id = "changed-episode"
        else:
            proposed.axes[0] = -0.5
        return risk

    labels = list(counterfactual_risk_samples(observation, [action], mutating_assessor))
    assert sha256_json(observation) == before
    assert labels[0][0].target_pilot_control == [0.5, 0.0, 0.0, 0.0]
    assert labels[0][1]["episode_id"] == observation.episode_id


# 功能：
#   验证第一条结果交给调用方后修改原列表，不会替换尚未评估的后续探针。
# 输入：
#   tmp_path：合成回合目录。
# 输出：
#   None：不返回业务数据。
def test_probe_batch_is_frozen_before_first_yield(tmp_path):
    _, observation, _, _ = fixture(tmp_path)
    actions = [PilotAction(mode="pilot-control", axes=[axis, 0.0, 0.0, 0.0])
               for axis in (0.0, 0.5)]
    samples = counterfactual_risk_samples(observation, actions, assessment)
    next(samples)
    actions[1].axes[0] = -0.5
    label, _ = next(samples)
    assert label.target_pilot_control == [0.5, 0.0, 0.0, 0.0]


# 功能：
#   验证重复探针在调用任何外部评估器之前就被整批拒绝。
# 输入：
#   tmp_path：合成回合目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_batch_has_no_partial_assessments(tmp_path):
    _, observation, _, _ = fixture(tmp_path)
    action = PilotAction(mode="pilot-control", axes=[0.5, 0.0, 0.0, 0.0])
    calls = []

    # 功能：
    #   记录是否被调用，并返回合法名义评估供完整批次测试使用。
    # 输入：
    #   received：评估观测。
    #   proposed：评估探针。
    # 输出：
    #   risk：测试评估回执。
    def tracked_assessor(received, proposed):
        calls.append(proposed)
        risk = assessment(received, proposed)
        return risk

    with pytest.raises(ValueError):
        list(counterfactual_risk_samples(observation, [action, action], tracked_assessor))
    assert calls == []


# 功能：
#   验证绕过模型赋值检查的回执仍需重验，名义计算不能声称是真实反事实执行。
# 输入：
#   tmp_path：合成回合目录。
#   change：非法验证摘要或错误回执来源。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [{"verifier_receipt_sha256": "invalid"},
                                    {"source": "physical-counterfactual"}])
def test_risk_receipt_is_revalidated_and_nominal(tmp_path, change):
    _, observation, _, _ = fixture(tmp_path)
    action = PilotAction(mode="pilot-control", axes=[0.5, 0.0, 0.0, 0.0])
    risk = assessment(observation, action).model_copy(update=change)
    with pytest.raises(ValueError):
        list(counterfactual_risk_samples(observation, [action], lambda *_: risk))
