"""Input-dependence tests, not physical flight or formal training samples."""

import numpy as np
import onnxruntime as ort
import pytest
from test_local_advisor_training import _sample

from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    _trainable_input_mask,
    export_local_advisor_onnx,
    train_local_advisor,
)


# 功能：
#   验证运动投影不能应用到职责不同的感知或载荷专家。
# 输入：
#   role：不允许使用该投影的角色。
# 输出：
#   None：配置必须被拒绝，不静默丢弃其他专家需要的信息。
@pytest.mark.parametrize("role", ["payload-dynamics-adapter", "perception-health-critic",
                                  "state-anomaly-detector", "cross-modal-consistency-critic"])
def test_projection_rejects_other_experts(role):
    with pytest.raises(ValueError, match="requires settle"):
        _trainable_input_mask(role, LocalAdvisorTrainingConfig(settle_motion_only=True))


# 功能：
#   1. 实际训练并导出，验证被排除的地图上下文在梯度更新后仍不影响部署输出。
#   2. 保持机动输入能够区分合成两类样本，避免用恒定输出假装不变性。
# 输入：
#   tmp_path：只存本次合成测试 ONNX 的隔离目录。
# 输出：
#   None：权重稀疏约束、上下文不变性和非恒定风险预测均通过断言。
def test_motion_projection_survives_training_and_onnx(tmp_path):
    role = "settle-stability-critic"
    config = LocalAdvisorTrainingConfig(epoch_count=100, settle_motion_only=True)
    samples = [_sample(role=role, risky=bool(i % 2)) for i in range(40)]
    model, metrics = train_local_advisor(samples, config, role=role)
    mask = _trainable_input_mask(role, config)[:, 0]
    assert np.all(model.input_weight[mask == 0] == 0)
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.0
    path = tmp_path / "settle.onnx"
    export_local_advisor_onnx(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    history = np.asarray([s.state_history for s in samples[:2]], dtype=np.float32)
    feeds = {
        "state_history": history,
        "history_mask": np.asarray([s.history_mask for s in samples[:2]], dtype=np.float32),
        "maneuver_features": np.asarray(
            [s.maneuver_features for s in samples[:2]], dtype=np.float32),
    }
    original = session.run(["risk_score"], feeds)[0]
    changed = history.copy()
    excluded = [i for i in range(changed.shape[-1]) if i not in (0, 1, 2, 12)]
    changed[:, :, excluded] = 2.0
    actual = session.run(["risk_score"], {**feeds, "state_history": changed})[0]
    np.testing.assert_array_equal(actual, original)
    assert float(actual[0, 0]) < .5 < float(actual[1, 0])
