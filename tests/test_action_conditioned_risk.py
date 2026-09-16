"""Behavioral/inference tests only; these synthetic labels never qualify flight."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_policy_packages import _write_payload_adapter

from dronedream_agent_core.contracts import NormalizedPilotControl
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_port import (
    LocalPolicyFeatureBatch,
    LocalPolicyPort,
    OnnxLocalPolicyBackend,
    _bounded_scalar,
)
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingSample,
    _new_model,
    expand_local_risk_critic_visual_inputs,
    load_local_risk_critic_onnx,
    train_local_risk_critic,
    write_local_policy_package,
)
from dronedream_agent_core.pilot_control_mapping import (
    PilotControlLimits,
    action_risk_features,
    physical_pilot_request,
)
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT


# 功能：
#   建议模型的非法标量必须被拒绝，不能裁剪、取首项或默认零后伪装成有效建议。
# 输入：
#   output：非法数值或错误形状的模型输出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("output", [[float("nan")], [float("inf")], [0.1, .2], [], [-.01], [1.01]])
def test_advisors_reject_invalid_scalar_instead_of_clipping_or_taking_first(output):
    with pytest.raises(RuntimeError, match="LOCAL_POLICY_TEST_OUTPUT"):
        _bounded_scalar(output, "TEST")


# 功能：
#   构造环境相同而前后控制风险相反的合成样本，用于检查风险网络是否依赖动作。
# 输入：
#   无。
# 输出：
#   samples：三十二条显式绑定候选控制的风险训练记录。
def _samples():
    samples = [LocalPolicyTrainingSample(
        state_features=[0.0] * 46, candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[0.0] * 8, target_action_index=8,
        risk_target=float(index % 2),
        risk_proposed_control=[0.8 if index % 2 else -0.8, 0, 0, 0],
        realtime_features=[0.0] * POLICY_REALTIME_FEATURE_COUNT,
        realtime_valid_mask=[1.0] * POLICY_REALTIME_FEATURE_COUNT,
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    ) for index in range(32)]
    return samples


# 功能：
#   实际训练一个小型合成风险网络并检查双类召回，不据此授予飞行资格。
# 输入：
#   无。
# 输出：
#   model：能区分合成前后动作风险的测试网络。
@pytest.fixture(scope="module")
def critic():
    model, metrics = train_local_risk_critic(_samples(), LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=200, learning_rate=0.01, batch_size=16,
    ))
    assert metrics.risk_hold_recall == 1.0
    assert metrics.safe_motion_recall == 1.0
    return model


# 功能：
#   构造可实际载入的测试包，固定连续前向输出并按需加入负载缩放专家。
# 输入：
#   tmp_path：测试模型输出目录。
#   critic：已经训练的动作条件风险网络。
#   forward：固定的归一化前向幅度。
#   payload：是否加入负载适配器。
# 输出：
#   package：经实际包加载器验证的测试策略包。
def _package(tmp_path, critic, forward, payload=False):
    model = _new_model(LocalPolicyTrainingConfig(hidden_feature_count=8),
                       realtime_feature_count=POLICY_REALTIME_FEATURE_COUNT,
                       include_pilot_control=True)
    model.input_weight[:] = 0
    model.action_weight[:] = 0
    model.action_bias[11] = 10
    model.pilot_control_weight[:] = 0
    model.pilot_control_bias[0] = np.arctanh(forward)
    model.risk_weight[:] = 0
    model.risk_bias[:] = -12
    payload_path = tmp_path / "payload.onnx" if payload else None
    if payload_path:
        _write_payload_adapter(payload_path)
    root = tmp_path / "package"
    write_local_policy_package(
        output_root=root, model=model, risk_model=critic,
        package_id="local.action-risk-fixture", display_name="Action risk fixture", scope="general",
        vehicle_sha256="a" * 64, sensor_contract_sha256="b" * 64,
        maximum_inference_latency_ms=500, payload_dynamics_adapter_path=payload_path,
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    )
    package = load_local_policy_package(root)
    return package


# 功能：
#   构造完整且禁用坐标候选的实时输入，绑定显式物理控制限额。
# 输入：
#   无。
# 输出：
#   batch：传给实际 ONNX 后端的合成输入批次。
def _batch():
    batch = LocalPolicyFeatureBatch(
        state_features=(0.0,) * 46, candidate_features=tuple((0.0,) * 15 for _ in range(8)),
        candidate_mask=(0.0,) * 8, candidate_ids=(),
        realtime_features=(0.0,) * POLICY_REALTIME_FEATURE_COUNT,
        realtime_valid_mask=(1.0,) * POLICY_REALTIME_FEATURE_COUNT,
        realtime_features_ready=True, realtime_snapshot_sha256="a" * 64,
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        pilot_control_limits=PilotControlLimits(20, 1, 20),
    )
    return batch


# 功能：
#   在相同环境下改变实际控制方向，应改变 ONNX 风险否决和最终决策，而非只改变说明文字。
# 输入：
#   tmp_path：测试包目录。
#   critic：已训练的合成风险模型。
#   forward：提议前向幅度。
#   expected：期望的最终动作类别。
#   request：负责测试异常或结束时关闭后端的 pytest 生命周期。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("forward,expected", [(0.8, "hold"), (-0.8, "pilot-control")])
def test_same_environment_different_proposed_action_changes_real_onnx_veto(
    tmp_path, critic, forward, expected, request,
):
    package = _package(tmp_path, critic, forward)
    loaded = load_local_risk_critic_onnx(package.artifact_paths["risk-critic"])
    assert loaded.action_conditioned
    assert loaded.visual_feature_count == 0
    assert np.array_equal(loaded.input_weight, critic.input_weight)
    backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    request.addfinalizer(backend.close)
    result = backend.infer(_batch(), multimodal=[])
    assert result.navigation_risk_score < 0.01
    assert result.advisory_risk_scores["risk-critic"] > 0.9 if forward > 0 else (
        result.advisory_risk_scores["risk-critic"] < 0.1
    )
    decision = LocalPolicyPort(package, backend=backend)._decision(
        snapshot={"snapshot_sha256": "a" * 64}, batch=_batch(), raw=result,
    )
    assert decision.action == expected
    inputs = {item.name for item in backend._risk_session.get_inputs()}
    assert "proposed_control" in inputs
    assert not inputs.intersection({"candidate_features", "candidate_mask"})


# 功能：
#   风险模型接收负载适配后真正将要执行的幅度，缺少物理限额时必须拒绝判断。
# 输入：
#   tmp_path：测试包目录。
#   critic：合成动作条件风险模型。
#   request：登记后端关闭的 pytest 生命周期。
# 输出：
#   None：不返回业务数据。
def test_risk_consumes_post_payload_scale_and_rejects_missing_limits(tmp_path, critic, request):
    package = _package(tmp_path, critic, 0.8, payload=True)
    backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    request.addfinalizer(backend.close)
    session = backend._risk_session
    seen = {}

    class ObservedSession:
        # 功能：
        #   保留实际风险 ONNX 会话的输入声明，不使用虚构签名绕过后端检查。
        # 输入：
        #   self：观察会话包装器。
        # 输出：
        #   inputs：实际会话输入元数据。
        def get_inputs(self):
            inputs = session.get_inputs()
            return inputs

        # 功能：
        #   保存实际传入的控制特征，同时仍运行真实 ONNX 推理。
        # 输入：
        #   self：观察会话包装器。
        #   names：需要返回的输出名称。
        #   feeds：输入名称到张量的映射。
        # 输出：
        #   outputs：实际 ONNX 推理返回的张量列表。
        def run(self, names, feeds):
            seen.update(feeds)
            outputs = session.run(names, feeds)
            return outputs

    backend._risk_session = ObservedSession()
    result = backend.infer(_batch(), multimodal=[])
    assert result.controller_step_scale == pytest.approx(0.4)
    assert seen["proposed_control"][0].tolist() == pytest.approx([0.32, 0, 0, 0])
    assert result.invoked_advisory_roles[-1] == "risk-critic"
    with pytest.raises(RuntimeError, match="ACTION_RISK_CONTEXT_MISSING"):
        backend.infer(replace(_batch(), pilot_control_limits=None), multimodal=[])


# 功能：
#   风险训练必须有显式提议动作、当前动作条件架构及安全／危险两类样本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_action_risk_never_borrows_teacher_actions_or_legacy_weights():
    samples = _samples()
    samples[0] = samples[0].model_copy(update={"risk_proposed_control": []})
    with pytest.raises(ValueError, match="explicit proposed controls"):
        train_local_risk_critic(samples, LocalPolicyTrainingConfig(epoch_count=1))
    with pytest.raises(ValueError, match="cannot be mixed"):
        train_local_risk_critic(_samples(), LocalPolicyTrainingConfig(epoch_count=1),
                               initial_model=_new_model(LocalPolicyTrainingConfig()))
    with pytest.raises(ValueError, match="safe and unsafe"):
        train_local_risk_critic(_samples()[::2], LocalPolicyTrainingConfig(epoch_count=1))


# 功能：
#   错误形状、非有限或越界的风险输出不能被计为安全投票。
# 输入：
#   tmp_path：测试包目录。
#   critic：合成动作条件模型。
#   bad：故意注入的错误推理结果。
#   reason：对应形状/非有限错误或概率越界的精确错误码。
#   request：登记后端关闭的 pytest 生命周期。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad,reason", [
    (float("nan"), "RISK_CRITIC_OUTPUT_INVALID"),
    (float("inf"), "RISK_CRITIC_OUTPUT_INVALID"),
    (-0.1, "RISK_CRITIC_OUTPUT_OUT_OF_RANGE"),
    (1.1, "RISK_CRITIC_OUTPUT_OUT_OF_RANGE"),
    ([0.0, 1.0], "RISK_CRITIC_OUTPUT_INVALID"),
])
def test_malformed_risk_output_cannot_become_a_safe_vote(tmp_path, critic, bad, reason, request):
    package = _package(tmp_path, critic, 0.8)
    backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    request.addfinalizer(backend.close)
    session = backend._risk_session

    class InvalidSession:
        # 功能：
        #   保留真实输入声明，使本测试只破坏推理输出而不提前改变接口契约。
        # 输入：
        #   self：故障注入包装器。
        # 输出：
        #   inputs：实际模型输入声明。
        def get_inputs(self):
            inputs = session.get_inputs()
            return inputs

        # 功能：
        #   注入可控的坏风险张量，验证推理后数值和形状检查。
        # 输入：
        #   self：故障注入包装器。
        #   names：调用方请求的输出名称。
        #   feeds：本测试不修改的输入映射。
        # 输出：
        #   outputs：仅包含故障风险张量的列表。
        def run(self, names, feeds):
            outputs = [np.asarray(bad, dtype=np.float32)]
            return outputs

    backend._risk_session = InvalidSession()
    with pytest.raises(RuntimeError, match=reason):
        backend.infer(_batch(), multimodal=[])


# 功能：
#   扩展视觉输入时仍保持动作特征尾部及原传感器权重的对应位置。
# 输入：
#   critic：扩展前动作条件模型。
# 输出：
#   None：不返回业务数据。
def test_action_risk_visual_extension_preserves_action_tail(critic):
    expanded = expand_local_risk_critic_visual_inputs(
        critic, visual_feature_count=3, random_seed=805,
    )
    assert expanded.action_conditioned
    assert np.array_equal(expanded.input_weight[-4:], critic.input_weight[-4:])
    assert np.array_equal(expanded.input_weight[:-7], critic.input_weight[:-4])


# 功能：
#   风险编码与执行器使用相同的有符号物理速度和偏航幅度，并包含 Harness 缩放。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_critic_and_actuator_share_signed_physical_mapping():
    control = NormalizedPilotControl(forward_axis=0.5, right_axis=-0.25,
                                     up_axis=0.8, yaw_axis=-0.6)
    limits = PilotControlLimits(2, 0.5, 30)
    physical = physical_pilot_request(control, limits, harness_scale=0.4)
    assert physical == pytest.approx([0.4, -0.2, 0.16, -7.2])
    features = action_risk_features(control, limits, harness_scale=0.4)
    assert features == pytest.approx([physical[0]/20, physical[1]/20,
                                      physical[2]/20, physical[3]/180])
