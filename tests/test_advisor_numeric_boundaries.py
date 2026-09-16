"""Numeric ownership, history masking and immutable advisor export checks."""

from dataclasses import replace

import numpy as np
import onnxruntime as ort
import pytest
from test_local_advisor_training import _sample

from dronedream_agent_core import local_advisor_training as training


# 功能：
#   初始化确定性的小型专家权重，仅供数值与导出边界测试。
# 输入：
#   role：专家角色。
# 输出：
#   model：未经训练且不具备飞行资格的权重对象。
def model(role="perception-health-critic"):
    result = training._new_model(role, training.LocalAdvisorTrainingConfig())
    return result


# 功能：
#   验证未知专家角色不会静默套用负载模型的输入维数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unknown_role_has_no_payload_fallback():
    with pytest.raises(ValueError):
        training._input_count("not-an-advisor")


# 功能：
#   验证训练入口重验已经构造但随后被绕过赋值校验修改的样本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_training_rechecks_mutated_sample():
    sample = _sample(role="perception-health-critic", risky=False)
    sample.risk_target = 0.0
    sample.__dict__["risk_target"] = -5.0
    with pytest.raises(ValueError):
        training.train_local_advisor(
            [sample], training.LocalAdvisorTrainingConfig(epoch_count=1), role=sample.role
        )


# 功能：
#   验证样本转成 float32 时的溢出不能被饱和 Sigmoid 掩盖成合法风险。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_float32_overflow_rejected_before_training():
    sample = _sample(role="perception-health-critic", risky=False)
    sample.state_features[0] = 1e100
    with pytest.raises(ValueError):
        training._arrays([sample], sample.role)


# 功能：
#   验证错误配置不能通过可变对象使训练循环零次执行却返回模型。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_training_rechecks_config():
    sample = _sample(role="perception-health-critic", risky=False)
    config = training.LocalAdvisorTrainingConfig().model_copy(update={"epoch_count": 0})
    with pytest.raises(ValueError):
        training.train_local_advisor([sample], config, role=sample.role)


# 功能：
#   模拟优化开始后调用方改写标签，确认最终训练指标仍来自最初参与优化的数组。
# 输入：
#   monkeypatch：在初始化网络时触发外部样本修改。
# 输出：
#   None：不返回业务数据。
def test_training_metrics_use_frozen_arrays(monkeypatch):
    sample = _sample(role="perception-health-critic", risky=False)
    original = training._new_model

    # 功能：
    #   创建正常网络后改变原始样本的风险标签，模拟训练期间的调用方写入。
    # 输入：
    #   role：当前专家角色。
    #   config：已验证的训练配置。
    # 输出：
    #   candidate：未修改的初始网络。
    def change_source(role, config):
        candidate = original(role, config)
        sample.risk_target = 1.0
        return candidate

    monkeypatch.setattr(training, "_new_model", change_source)
    _, result = training.train_local_advisor(
        [sample], training.LocalAdvisorTrainingConfig(epoch_count=1), role=sample.role
    )
    assert sample.risk_target == 1.0
    assert result.risky_sample_count == 0 and result.safe_sample_count == 1


# 功能：
#   验证评估前拒绝坏权重和非法维度，不让饱和或广播隐藏不合法模型。
# 输入：
#   kind：权重破坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["infinite", "wrong_bias", "extra_scale"])
def test_evaluation_rechecks_model(kind):
    candidate = model()
    if kind == "infinite":
        candidate.input_weight.fill(float("inf"))
    elif kind == "wrong_bias":
        candidate = replace(candidate, input_bias=np.zeros(1, dtype=np.float32))
    else:
        candidate = replace(
            candidate,
            scale_weight=np.zeros((32, 1), dtype=np.float32),
            scale_bias=np.zeros(1, dtype=np.float32),
        )
    with pytest.raises(ValueError):
        training.evaluate_local_advisor(candidate, [_sample(role=candidate.role, risky=False)])


# 功能：
#   验证被掩码排除的历史不影响本机输入与实际 ONNX 结果，两个实现保持一致。
# 输入：
#   tmp_path：隔离模型目录。
#   role：使用时间历史的专家类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "role",
    [
        "settle-stability-critic",
        "state-anomaly-detector",
        "payload-dynamics-adapter",
    ],
)
def test_masked_history_is_ignored_in_training_and_onnx(tmp_path, role):
    clean = _sample(role=role, risky=False)
    clean.history_mask = [0.0] * 8
    noisy = clean.model_copy(deep=True)
    for row in noisy.state_history:
        row[:] = [30.0] * len(row)
    if noisy.payload_history is not None:
        for row in noisy.payload_history:
            row[:] = [30.0] * len(row)
    assert training._sample_input(clean) == training._sample_input(noisy)
    candidate = model(role)
    path = tmp_path / "masked.onnx"
    training.export_local_advisor_onnx(candidate, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    predictions = []
    for sample in (clean, noisy):
        feed = {
            port.name: np.asarray([getattr(sample, port.name)], dtype=np.float32)
            for port in session.get_inputs()
        }
        predictions.append(session.run(None, feed))
    for left, right in zip(*predictions, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6)
    inputs = training._arrays([clean], role)[0]
    local = training._forward(candidate, inputs)
    np.testing.assert_allclose(predictions[0][0], local[2], atol=1e-6)
    if role == "payload-dynamics-adapter":
        np.testing.assert_allclose(predictions[0][1], local[3], atol=1e-6)


# 功能：
#   验证导出不覆盖原有文件，错误模型也不能产生可选用的新 ONNX。
# 输入：
#   tmp_path：隔离导出目录。
# 输出：
#   None：不返回业务数据。
def test_export_preserves_existing_model_and_rejects_bad_weights(tmp_path):
    destination = tmp_path / "advisor.onnx"
    destination.write_bytes(b"existing-model")
    with pytest.raises(FileExistsError):
        training.export_local_advisor_onnx(model(), destination)
    assert destination.read_bytes() == b"existing-model"
    candidate = model()
    candidate.risk_bias[:] = float("nan")
    fresh = tmp_path / "bad.onnx"
    with pytest.raises(ValueError):
        training.export_local_advisor_onnx(candidate, fresh)
    assert not fresh.exists()
