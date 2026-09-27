"""Synthetic training/export invariance; not field reliability evidence."""

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
#   专用状态投影不能侵入跨模态、负载或稳定性专家，也不能同时启用另一种投影。
# 输入：
#   role：受检角色；settle：是否同时配置稳定性投影。
# 输出：
#   None：不符合职责的配置被拒绝。
@pytest.mark.parametrize("role,settle", [("cross-modal-consistency-critic", False),
    ("payload-dynamics-adapter", False), ("settle-stability-critic", False),
    ("perception-health-critic", True)])
def test_sensor_projection_rejects_incompatible_role(role, settle):
    config = LocalAdvisorTrainingConfig(sensor_state_only=True, settle_motion_only=settle)
    with pytest.raises(ValueError, match="sensor-state input projection"):
        _trainable_input_mask(role, config)


# 功能：
#   实际优化并导出状态专家，核验地图上下文无法改变 ONNX 输出，健康变化仍可被区分。
# 输入：
#   tmp_path：合成导出目录；role：健康或历史状态异常角色。
# 输出：
#   None：零权重约束、非恒定输出及上下文不变性均通过断言。
@pytest.mark.parametrize("role", ["perception-health-critic", "state-anomaly-detector"])
def test_sensor_projection_train_export_and_context_invariance(tmp_path, role):
    config = LocalAdvisorTrainingConfig(sensor_state_only=True, epoch_count=100)
    samples = []
    for index in range(40):
        risky = bool(index % 2)
        state = [0.] * 46
        state[7:10] = [float(not risky), float(risky), .04]
        samples.append(_sample(role=role, risky=risky).model_copy(update={
            "state_features": state, "state_history": [list(state) for _ in range(8)],
            "history_mask": [1.] * 8}))
    model, metrics = train_local_advisor(samples, config, role=role)
    mask = _trainable_input_mask(role, config)[:, 0]
    assert np.all(model.input_weight[mask == 0] == 0)
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.
    path = tmp_path / "sensor.onnx"
    export_local_advisor_onnx(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if role == "perception-health-critic":
        name = "state_features"
        original = np.asarray([s.state_features for s in samples[:2]], dtype=np.float32)
        feeds = {name: original}
        retained = (7, 8, 9)
    else:
        name = "state_history"
        original = np.asarray([s.state_history for s in samples[:2]], dtype=np.float32)
        feeds = {name: original, "history_mask": np.ones((2, 8), dtype=np.float32)}
        retained = (0, 1, 2, 7, 8, 9, 12)
    output = "anomaly_score" if role == "state-anomaly-detector" else "risk_score"
    expected = session.run([output], feeds)[0]
    changed = original.copy()
    changed[..., [i for i in range(46) if i not in retained]] = 3.
    if role == "state-anomaly-detector":
        changed[:, :-1, 7:10] = 3.
    actual = session.run([output], {**feeds, name: changed})[0]
    np.testing.assert_array_equal(actual, expected)
    assert float(actual[0, 0]) < .5 < float(actual[1, 0])
