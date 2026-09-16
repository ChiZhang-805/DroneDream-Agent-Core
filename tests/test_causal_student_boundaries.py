"""Student admission faults use synthetic observations and locally exported graphs."""

import json

import pytest
import torch
from test_offline_flight_learning import UnitEnvironment
from test_onnx_causal_student import exported  # noqa: F401

from dronedream_agent_core.causal_control import CausalControlHistory
from dronedream_agent_core.training.causal_policy import CausalPilotPolicy, CausalPolicyConfig
from dronedream_agent_core.training.causal_student import (
    CausalStudent,
    OnnxCausalStudent,
    append_flight_history,
)
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   带重复字段的回执即使最后一个值正确，也不能静默覆盖前值后启动模型。
# 输入：
#   exported：实际导出的合成 ONNX 及配套基座和回执。
# 输出：
#   None：不返回业务数据。
def test_duplicate_receipt_keys_are_rejected(exported):  # noqa: F811
    model, content, receipt = exported
    encoded = ('{"expert_role":"recovery-policy",' + json.dumps(receipt)[1:]).encode()
    with pytest.raises(ValueError):
        OnnxCausalStudent(model, expert_role="local-navigation-policy",
                          onnx_content=content, receipt_content=encoded)


# 功能：
#   绕过实例校验的旧语义、非法掩码、候选坐标或回合序号必须在改变学生历史前被拒绝。
# 输入：
#   fault：将被破坏的观测字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["contract", "candidate", "sequence", "mask", "missing-time"])
def test_student_revalidates_observation_before_advancing_history(fault):
    model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1))
    student = CausalStudent(model, expert_role="local-navigation-policy")
    observation = UnitEnvironment().reset(seed=1)
    if fault == "contract":
        observation.sample.control_feature_contract_sha256 = "a" * 64
    elif fault == "candidate":
        observation.sample.candidate_mask[0] = 1.
    elif fault == "sequence":
        observation = observation.model_copy(update={"sequence": True})
    elif fault == "mask":
        observation.sample.realtime_valid_mask[0] = .5
    else:
        observation.sample.temporal_evidence = None
    with pytest.raises(ValueError):
        student(observation)
    assert student.history.history.latest is None
    assert student._last_sequence == -1


# 功能：
#   PPO 共用的历史入口也必须重验观测，不能仅依赖确定性学生自己的检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_shared_history_entry_rejects_mutated_observation():
    observation = UnitEnvironment().reset(seed=1)
    observation.sample.control_feature_contract_sha256 = "a" * 64
    history = CausalControlHistory(4)
    with pytest.raises(ValueError):
        append_flight_history(history, observation)
    assert history.history.latest is None


# 功能：
#   形状错误或越界风险不能被作为普通模型输出忽略，即使模式和四轴本身看起来正常。
# 输入：
#   monkeypatch：替换学生的推理返回值。
#   fault：错误模式宽度或越界风险。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["mode-shape", "risk-range"])
def test_student_output_checks_include_shapes_and_risk(monkeypatch, fault):
    model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1))
    student = CausalStudent(model, expert_role="local-navigation-policy")
    env = UnitEnvironment()
    observation = env.reset(seed=1)
    for _ in range(3):
        student(observation)
        observation = env.step(PilotAction(mode="hold", axes=[0.] * 4)).observation
    modes = torch.zeros(1, 5 if fault == "mode-shape" else 4)
    risk = torch.tensor([[1.2 if fault == "risk-range" else .2]])
    outputs = (torch.zeros(1, 8), modes, risk, torch.zeros(1, 4))
    monkeypatch.setattr(student, "_predict", lambda columns: outputs)
    with pytest.raises(ValueError, match="OUTPUT"):
        student(observation)
