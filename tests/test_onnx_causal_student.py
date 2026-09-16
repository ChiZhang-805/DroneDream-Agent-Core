"""Actual CPU graphs with synthetic observations; no aircraft or qualification."""

import hashlib
import json

import pytest
import torch
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    export_causal_policy,
)
from dronedream_agent_core.training.causal_student import CausalStudent, OnnxCausalStudent
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   导出可实际运行的合成连续控制模型和回执，并在结束时恢复 Torch 全局线程数。
# 输入：
#   tmp_path：隔离模型输出目录。
# 输出：
#   model：用于数值对照的因果基座。
#   content：实际 ONNX 文件字节。
#   receipt：与模型摘要、角色及配置绑定的测试回执。
@pytest.fixture
def exported(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = CausalPolicyConfig(history_length=4, encoder_width=32,
                                   recurrent_width=32, head_width=32)
        model = CausalPilotPolicy(config).eval()
        with torch.no_grad():
            model.mode_head.weight.zero_()
            model.mode_head.bias.zero_()
            model.mode_head.bias[3] = 5.
        path = tmp_path / "student.onnx"
        digest = export_causal_policy(model, path)
        receipt = {"artifact_sha256": digest, "expert_role": "local-navigation-policy",
                   "config": config.model_dump(),
                   "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256}
        content = path.read_bytes()
        yield model, content, receipt
    finally:
        torch.set_num_threads(threads)


# 功能：
#   从真实测试导出物构建学生，允许覆盖回执字段以逐项注入绑定错误。
# 输入：
#   exported：测试基座、模型字节及回执。
#   updates：覆盖到回执的测试字段。
# 输出：
#   actor：经过启动校验的 ONNX 学生。
def student(exported, **updates):
    model, content, receipt = exported
    actor = OnnxCausalStudent(model, expert_role="local-navigation-policy", onnx_content=content,
                              receipt_content=json.dumps({**receipt, **updates}).encode())
    return actor


# 功能：
#   对照实际 ONNX 和 Torch 的历史、控制及重置语义，输入不能被原位修改，重放必须拒绝。
# 输入：
#   exported：真实导出的合成模型及回执。
# 输出：
#   None：不返回业务数据。
def test_cpu_deployment_graph_keeps_history_actions_and_reset_semantics(exported):
    optimized = student(exported)
    reference = CausalStudent(exported[0], expert_role="local-navigation-policy")
    assert optimized.startup_maximum_absolute_error < 1e-5
    assert optimized.artifact_sha256 == hashlib.sha256(exported[1]).hexdigest()
    env = UnitEnvironment()
    observation = env.reset(seed=1)
    for index in range(8):
        before = observation.model_dump_json()
        expected, actual = reference(observation), optimized(observation)
        assert expected.mode == actual.mode
        assert actual.axes == pytest.approx(expected.axes, abs=1e-5)
        assert optimized.last_timing["sequence"] == observation.sequence
        assert optimized.last_timing["input_preparation_ms"] >= 0
        if index < 3:
            assert optimized.last_timing["actor_sampling_ms"] == 0
        else:
            assert optimized.last_timing["actor_sampling_ms"] > 0
        assert before == observation.model_dump_json()
        assert actual.mode == ("hold" if index < 3 else "pilot-control")
        if index < 7:
            observation = env.step(PilotAction(mode="hold", axes=[0.] * 4)).observation
    with pytest.raises(ValueError, match="REPLAYED"):
        optimized(observation)
    assert optimized.last_timing is None
    assert optimized(env.reset(seed=2)).mode == "hold"


# 功能：
#   过期准备可在相同提交序号下用更新的物理来源重试，但不能复用旧样本凑满历史。
# 输入：
#   exported：真实测试模型及回执。
# 输出：
#   None：不返回业务数据。
def test_stream_retry_uses_new_sensor_time_at_unchanged_submission_index(exported):
    from test_stream_collection import StreamLifecycle, collect

    from dronedream_agent_core.training.flight_environment import ControlPreparationExpired

    optimized, env = student(exported), StreamLifecycle()
    original_next, original_submit = env.next_stream_observation, env.submit_stream_action
    attempts, submitted = [], []

    # 功能：
    #   保持环境真实样本时刻，只把动作序号绑定到已实际提交的数量。
    # 输入：
    #   deadline：采集器传入的观察期限。
    # 输出：
    #   observation：来源更新而提交序号可不变的测试观测。
    def next_input(*, deadline):
        observation = original_next(deadline=deadline)
        observation = observation.model_copy(update={"sequence": len(submitted)})
        return observation

    # 功能：
    #   记录每次推理尝试的提交序号和物理时刻，再调用实际 ONNX 学生。
    # 输入：
    #   observation：当前采集观测。
    # 输出：
    #   action：学生提出的动作。
    def actor(observation):
        attempts.append((observation.sequence,
                         observation.sample.temporal_evidence.observed_at_unix_ms))
        action = optimized(observation)
        return action

    # 功能：
    #   模拟前三次准备过期，只把最后一次有效提案送入环境。
    # 输入：
    #   action：待提交的学生动作。
    # 输出：
    #   None：不返回业务数据。
    def submit(action):
        if len(attempts) < 4:
            raise ControlPreparationExpired("synthetic preparation failure")
        submitted.append(action)
        original_submit(action)

    env.next_stream_observation, env.submit_stream_action = next_input, submit
    collect(env, actor, steps=1)
    assert [sequence for sequence, _ in attempts] == [0, 0, 0, 0]
    assert len({stamp for _, stamp in attempts}) == 4
    assert len(submitted) == 1 and submitted[0].mode == "pilot-control"
    assert env.events[-2:] == ["ground", "join-actual-receipts"]


# 功能：
#   提交序号递增不能掩盖旧传感器样本，新传感器样本也不能掩盖旧提交序号。
# 输入：
#   exported：测试部署模型及回执。
#   fault：将要注入的重放形式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["same-source-next-index", "older-index-new-source"])
def test_index_and_sensor_time_cannot_mask_each_others_replay(exported, fault):
    optimized, env = student(exported), UnitEnvironment()
    first = env.reset(seed=1).model_copy(update={"sequence": 1})
    optimized(first)
    if fault == "same-source-next-index":
        bad = first.model_copy(update={"sequence": 2})
    else:
        env.sequence += 1
        bad = env.observation().model_copy(update={"sequence": 0})
    with pytest.raises(ValueError, match="OBSERVATION_REPLAYED"):
        optimized(bad)


# 功能：
#   回执模型摘要、角色、配置或特征契约不匹配时，禁止启动学生。
# 输入：
#   exported：合法测试导出物。
#   updates：损坏的回执字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("updates", [{"artifact_sha256": "a" * 64},
    {"expert_role": "recovery-policy"}, {"feature_contract_sha256": "a" * 64}, {"config": {}}])
def test_mismatched_receipt_cannot_start_student(exported, updates):
    with pytest.raises(ValueError, match="ONNX_RECEIPT_MISMATCH"):
        student(exported, **updates)


# 功能：
#   即使配置相同，已改变基座参数与 ONNX 输出不一致也必须在启动探测时拒绝。
# 输入：
#   exported：原始对应的合成基座、部署图和回执。
# 输出：
#   None：不返回业务数据。
def test_graph_cannot_silently_serve_different_weights_with_matching_configuration(exported):
    model, _, _ = exported
    with torch.no_grad():
        model.axes_head.bias.add_(.5)
    with pytest.raises(ValueError, match="DIFFERS_FROM_CHECKPOINT"):
        student(exported)
