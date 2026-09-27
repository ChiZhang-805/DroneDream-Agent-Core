"""Synthetic regressions for observable payload supervision, not flight qualification."""
from copy import deepcopy

import numpy as np
import onnxruntime as ort
import pytest
from test_local_advisor_training import _sample

from dronedream_agent_core.local_advisor_training import (
    LocalAdvisorTrainingConfig,
    _arrays,
    _forward,
    _new_model,
    _sample_input,
    drop_payload_history_prefix,
    export_local_advisor_onnx,
    train_local_advisor,
)
from dronedream_agent_core.training.advisor_evaluation import advisor_feeds
from scripts import evaluate_local_policy_offline as package_evaluation


# 功能：
#   只改变同源运动信息，确认真实优化和导出能区分相同负载下的静止与快速下降。
# 输入：
#   tmp_path：合成 ONNX 的独立目录。
# 输出：
#   None：由风险分离及 NumPy/ONNX 一致性断言给出结果。
def test_identical_payload_different_motion_is_observable(tmp_path):
    safe = _sample(role='payload-dynamics-adapter', risky=False)
    unsafe = deepcopy(safe)
    unsafe.maneuver_features[-4:] = [3., 3., .6, 4.]
    for row in unsafe.state_history:
        row[2], row[12] = -3., .2
    unsafe.risk_target = 1.
    rows = [safe, unsafe] * 30
    model, _ = train_local_advisor(rows, LocalAdvisorTrainingConfig(
        epoch_count=100, payload_history_dropout=True), role='payload-dynamics-adapter')
    inputs, _, _, _ = _arrays([safe, unsafe], 'payload-dynamics-adapter')
    expected = _forward(model, inputs)[2]
    assert expected[0, 0] < .2 and expected[1, 0] > .8
    path = tmp_path/'payload.onnx'
    export_local_advisor_onnx(model, path)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    feeds = advisor_feeds('payload-dynamics-adapter', [safe, unsafe])
    actual = session.run(['risk_score'], feeds)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    # 最终模型包准入工具也必须实际运行相同五输入图，不只检查独立训练评估器。
    for index, sample in enumerate((safe, unsafe)):
        package_feeds = package_evaluation._advisor_feeds(
            session, sample, role='payload-dynamics-adapter')
        np.testing.assert_allclose(session.run(['risk_score'], package_feeds)[0],
                                   expected[index:index + 1], atol=1e-6)
    # 缺帧有效位同时遮蔽物理与运动历史，不允许旧无效槽向部署图漏入数字。
    safe.history_mask[:6] = [0.] * 6
    safe.payload_history[:6] = [[9.] * 24 for _ in range(6)]
    safe.state_history[:6] = [[9.] * 46 for _ in range(6)]
    masked = np.asarray([_sample_input(safe)], dtype=np.float32)
    np.testing.assert_allclose(session.run(['risk_score'], advisor_feeds(
        'payload-dynamics-adapter', [safe]))[0], _forward(model, masked)[2], atol=1e-6)


# 功能：
#   检查缺帧增强不修改原样本、当前帧或最近两帧；非法角色与越界数量在训练前拒绝。
# 输入：
#   无。
# 输出：
#   None：由精确张量位置及错误断言给出结果。
def test_history_dropout_preserves_supervision_and_ownership():
    source = np.ones((2, 605), dtype=np.float32)
    # 24 当前载荷 + 13 当前机动 + 8×24 载荷历史 + 8×46 运动历史 + 8 掩码。
    assert len(_sample_input(_sample(role='payload-dynamics-adapter', risky=False))) == 605
    output = drop_payload_history_prefix(source, [0, 6])
    assert np.all(source == 1) and np.all(output[0] == 1)
    assert np.all(output[1, :37] == 1)
    assert np.all(output[1, 37:181] == 0) and np.all(output[1, 181:229] == 1)
    assert np.all(output[1, 229:505] == 0) and np.all(output[1, 505:597] == 1)
    assert output[1, 597:].tolist() == [0.] * 6 + [1.] * 2
    for invalid in ([7, 0], [-1, 0], [True, False], [1.5, 0.], [0]):
        with pytest.raises(ValueError):
            drop_payload_history_prefix(source, invalid)
    with pytest.raises(ValueError, match='requires payload'):
        _new_model('state-anomaly-detector', LocalAdvisorTrainingConfig(payload_history_dropout=True))
