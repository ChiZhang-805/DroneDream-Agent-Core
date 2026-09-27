"""Structural and actual ONNX covariance monotonicity; synthetic test evidence only."""
import numpy as np
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    export_local_risk_critic_onnx,
    train_local_risk_critic,
)
from dronedream_agent_core.training.action_risk_training import _predict, verify_risk_export
from dronedream_agent_core.training.risk_nearfield import embed_nearfield_transform
from dronedream_agent_core.training.risk_uncertainty_monotonicity import (
    UNCERTAINTY_COLUMNS,
    project_uncertainty_paths,
    validate_uncertainty_graph,
)


# 功能：检查正负读出路径投影，同时保留普通特征，避免强迫整个网络只能使用正权重。
# 输入：无；压缩列和符号相反的权重夹具。
# 输出：None；两类协方差路径均非负，普通列未被改写。
def test_projection_handles_both_readout_signs_and_compact_columns():
    columns = np.array([0, *UNCERTAINTY_COLUMNS, 800])
    weight = np.array([[1., -2.], [1., 2.], [-1., -2.], [-4., 3.]])
    readout = np.array([[2.], [-3.]])
    original = weight.copy()
    project_uncertainty_paths(weight, readout, columns)
    assert (weight[1:3]*readout[:, 0] >= 0).all()
    np.testing.assert_array_equal(weight[[0, 3]], original[[0, 3]])
    readout *= -1
    project_uncertainty_paths(weight, readout, columns)
    assert (weight[1:3]*readout[:, 0] >= 0).all()


# 功能：训练窄方差范围后跨范围检验单调性、归一化折叠和真实 ONNX 输出，不靠删方差特征。
# 输入：normalize：归一化开关；tmp_path：独立导出目录。
# 输出：None；定位更准确导致更高风险、源样本变化或导出不等价时失败。
@pytest.mark.parametrize('normalize', [False, True])
def test_monotonic_uncertainty_survives_export_and_out_of_domain(normalize, tmp_path):
    samples = _samples()
    for sample in samples:
        sample.risk_proposed_control = [0., 0., 0., 0.]
        sample.state_features[9] = .3 if sample.risk_target else .2
        sample.realtime_features[UNCERTAINTY_COLUMNS[1]-46] = sample.state_features[9]
        # 本夹具只提供方差，不能把另外445个恒零传感器通道全声称有效。
        sample.realtime_valid_mask = [0.] * len(sample.realtime_valid_mask)
        sample.realtime_valid_mask[UNCERTAINTY_COLUMNS[1]-46] = 1.
    frozen = [sample.model_dump() for sample in samples]
    model, metrics = train_local_risk_critic(samples, LocalPolicyTrainingConfig(
        hidden_feature_count=16, epoch_count=200, batch_size=16, learning_rate=.01),
        normalize_inputs=normalize, monotonic_uncertainty=True)
    assert metrics.risk_hold_recall == metrics.safe_motion_recall == 1.
    assert [s.model_dump() for s in samples] == frozen
    assert (model.input_weight[list(UNCERTAINTY_COLUMNS)]*model.risk_weight[:, 0] >= 0.).all()
    checks = []
    for column in UNCERTAINTY_COLUMNS:
        sweep = []
        for value in (0., .0001, .001, .05, .2, .3, .8, 1., 4.):
            sample = samples[0].model_copy(deep=True)
            if column == 9:
                sample.state_features[9] = value
            else:
                sample.realtime_features[column-46] = value
            sweep.append(sample)
        predictions, _ = _predict(model, sweep)
        assert (np.diff(predictions) >= -1e-7).all()
        checks.extend(sweep)
    path = tmp_path/'risk.onnx'
    export_local_risk_critic_onnx(model, path)
    graph = path.read_bytes()
    validate_uncertainty_graph(graph, True)
    validate_uncertainty_graph(embed_nearfield_transform(graph, 4.), True)
    assert verify_risk_export(model, graph, checks) <= 1e-5

    import onnx
    from onnx import numpy_helper
    tampered = onnx.load_model_from_string(graph)
    for item in tampered.graph.initializer:
        if item.name == 'input_weight':
            weights = numpy_helper.to_array(item).copy()
            slot = int(np.argmax(np.abs(model.risk_weight[:, 0])))
            weights[9, slot] = -np.sign(model.risk_weight[slot, 0])
            item.CopyFrom(numpy_helper.from_array(weights, 'input_weight'))
    with pytest.raises(ValueError, match='PATH_NEGATIVE'):
        validate_uncertainty_graph(tampered.SerializeToString(), True)
    tampered.graph.node[-1].op_type = 'Tanh'
    with pytest.raises(ValueError, match='GRAPH_INVALID'):
        validate_uncertainty_graph(tampered.SerializeToString(), True)


# 功能：严格拒绝非布尔约束，防止字符串 false 隐式启用或被回执伪造。
# 输入：flag：非法值；输出：None；优化和验收两端均拒绝。
@pytest.mark.parametrize('flag', [0, 1, 'false', None])
def test_flag_is_explicit(flag):
    with pytest.raises(ValueError, match='explicit boolean'):
        train_local_risk_critic(_samples(), LocalPolicyTrainingConfig(epoch_count=1),
                               monotonic_uncertainty=flag)
    with pytest.raises(ValueError, match='CONSTRAINT_INVALID'):
        validate_uncertainty_graph(b'not-a-model', flag)
