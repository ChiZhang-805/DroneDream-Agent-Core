"""Near-field transforms are graph operations, not implicit caller assumptions."""

import numpy as np
import onnx
import pytest
from test_action_conditioned_risk import _samples

from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    _new_model,
    export_local_risk_critic_onnx,
    load_local_risk_critic_onnx,
    train_local_risk_critic,
)
from dronedream_agent_core.realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from dronedream_agent_core.training.action_risk_training import (
    train_action_risk_expert,
    verify_risk_export,
)
from dronedream_agent_core.training.risk_nearfield import (
    embed_nearfield_transform,
    nearfield_descriptor,
    project_nearfield_inputs,
    validate_nearfield_graph,
)


# 功能：
#   拒绝错误距离配置且不生成输出目录。
# 输入：
#   cap：非法距离；tmp_path：隔离目录。
# 输出：
#   None：通过异常与目录不存在断言验证边界。
@pytest.mark.parametrize('cap', [True, '4', 0, -.1, .2, 20, float('nan'), float('inf')])
def test_invalid_cap_before_output(cap, tmp_path):
    output = tmp_path / 'candidate'
    with pytest.raises(ValueError, match='NEARFIELD_DISTANCE_INVALID'):
        train_action_risk_expert([], [], LocalPolicyTrainingConfig(), output,
                                normalize_inputs=True, physical_context_only=True,
                                geometry_distance_cap_m=cap)
    assert not output.exists()


# 功能：
#   检查只改变指定距离列，未命中掩码和实际动作始终保留。
# 输入：
#   无。
# 输出：
#   None：源记录被改变或其他列被裁剪时测试失败。
def test_nearfield_preserves_every_other_column_and_source():
    source = _samples()[0]
    source.realtime_features = [.9] * POLICY_REALTIME_FEATURE_COUNT
    source.realtime_features[0] = .03
    source.realtime_valid_mask[0] = 0.
    frozen = source.model_dump()
    projected = project_nearfield_inputs([source], 4.)[0]
    assert source.model_dump() == frozen
    cap = nearfield_descriptor(4.)['normalized_cap']
    for index, value in enumerate(projected.realtime_features):
        expected = (min(source.realtime_features[index], cap)
                    if index < 130 and index % 5 < 2 else .9)
        assert value == expected
    modified = projected.model_dump()
    modified['realtime_features'] = frozen['realtime_features']
    assert modified == frozen


# 功能：
#   构造确实响应距离输入的网络，避免全零网络让预处理缺失测试虚假通过。
# 输入：
#   tmp_path：模型输出目录。
# 输出：
#   model、raw、wrapped：参数、原图和带实际近场运算的图。
def _distance_network(tmp_path):
    model = _new_model(LocalPolicyTrainingConfig(hidden_feature_count=8),
                       realtime_feature_count=POLICY_REALTIME_FEATURE_COUNT,
                       action_conditioned=True)
    model.input_weight[:] = 0.
    model.input_bias[:] = 0.
    model.risk_weight[:] = 0.
    model.risk_bias[:] = 0.
    model.input_weight[46, 0] = 2.
    model.risk_weight[0, 0] = 1.
    path = tmp_path / 'raw.onnx'
    export_local_risk_critic_onnx(model, path)
    raw = path.read_bytes()
    wrapped = embed_nearfield_transform(raw, 4.)
    return model, raw, wrapped


# 功能：
#   用原始输入执行实际 ONNX，验证内嵌变换等价，故意去掉变换必须失败。
# 输入：
#   tmp_path：导出目录。
# 输出：
#   None：原图缺少变换而未被发现时测试失败。
def test_real_raw_input_parity_detects_missing_transform(tmp_path):
    model, raw, wrapped = _distance_network(tmp_path)
    samples = _samples()
    for row in samples:
        row.realtime_features[0] = .9
    assert verify_risk_export(model, wrapped, samples, geometry_distance_cap_m=4.) < 1e-6
    with pytest.raises(ValueError, match='EQUIVALENCE_FAILED'):
        verify_risk_export(model, raw, samples, geometry_distance_cap_m=4.)
    with pytest.raises(ValueError, match='ALREADY_EMBEDDED'):
        embed_nearfield_transform(wrapped, 4.)
    target = tmp_path / 'wrapped.onnx'
    target.write_bytes(wrapped)
    with pytest.raises(ValueError, match='complete ONNX graph'):
        load_local_risk_critic_onnx(target)


# 功能：
#   检查远处背景变化不改变风险，但近场净空变化仍改变输出。
# 输入：
#   tmp_path：导出目录。
# 输出：
#   None：实际执行结果失去所需近场敏感性或远场不变性时测试失败。
def test_far_invariance_and_near_sensitivity(tmp_path):
    import onnxruntime as ort

    from dronedream_agent_core.training.action_risk_training import _predict

    model, _, wrapped = _distance_network(tmp_path)
    rows = _samples()[:4]
    for row, distance in zip(rows, [.8, .5, .01, .1], strict=True):
        row.realtime_features[0] = distance
    _, feeds = _predict(model, rows)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(wrapped, sess_options=options,
                                   providers=['CPUExecutionProvider'])

    result = session.run(['risk_score'], feeds)[0].reshape(-1)
    assert result[0] == result[1]
    assert result[2] < result[3] < result[0]
    assert np.isfinite(result).all()


# 功能：
#   拒绝错误特征宽度和占用内部保留名的图，不覆盖已有运算。
# 输入：
#   tmp_path：导出目录。
# 输出：
#   None：错误接口或命名冲突被明确拒绝。
def test_wrapper_rejects_contract_and_name_collision(tmp_path):
    _, raw, _ = _distance_network(tmp_path)
    graph = onnx.load_model_from_string(raw)
    graph.graph.node[0].name = 'nearfield_metric_clip'
    with pytest.raises(ValueError, match='NAME_COLLISION'):
        embed_nearfield_transform(graph.SerializeToString(), 4.)
    graph = onnx.load_model_from_string(raw)
    item = next(item for item in graph.graph.input if item.name == 'realtime_features')
    item.type.tensor_type.shape.dim[1].dim_value = 445
    with pytest.raises(ValueError, match='REALTIME_WIDTH_INVALID'):
        embed_nearfield_transform(graph.SerializeToString(), 4.)


# 功能：
#   检查部署接纳能识别错误传感器、错误截断权重、遗漏声明及绕开图内变换。
# 输入：
#   tmp_path：导出目录。
# 输出：
#   None：图、回执与传感器三者必须完全一致。
def test_admission_binds_actual_graph_and_sensor(tmp_path):
    _, raw, wrapped = _distance_network(tmp_path)
    descriptor = nearfield_descriptor(4.)
    sensor = descriptor['sensor_contract_sha256']
    validate_nearfield_graph(wrapped, descriptor, sensor)
    validate_nearfield_graph(raw, None, sensor)
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        validate_nearfield_graph(wrapped, descriptor, '0' * 64)
    with pytest.raises(ValueError, match='UNRECORDED'):
        validate_nearfield_graph(wrapped, None, sensor)
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        validate_nearfield_graph(raw, descriptor, sensor)
    graph = onnx.load_model_from_string(wrapped)
    for node in graph.graph.node[1:]:
        for index, value in enumerate(node.input):
            if value == 'nearfield_geometry':
                node.input[index] = 'realtime_features'
    with pytest.raises(ValueError, match='GRAPH_MISMATCH'):
        validate_nearfield_graph(graph.SerializeToString(), descriptor, sensor)
    graph = onnx.load_model_from_string(wrapped)
    weights = next(w for w in graph.graph.initializer if w.name == 'nearfield_distance_caps')
    caps = onnx.numpy_helper.to_array(weights).copy()
    caps[0] = .9
    weights.CopyFrom(onnx.numpy_helper.from_array(caps, weights.name))
    with pytest.raises(ValueError, match='CAPS_INVALID'):
        validate_nearfield_graph(graph.SerializeToString(), descriptor, sensor)


# 功能：
#   拒绝非法危险成本，避免布尔值、无穷值或极大权重破坏优化与回执。
# 输入：
#   cost：待检查权重；tmp_path：隔离输出目录。
# 输出：
#   None：底层训练器和生产入口均拒绝且没有产物。
@pytest.mark.parametrize('cost', [True, '1.3', .9, 4.01, float('nan'), float('inf')])
def test_unsafe_cost_invalid_before_training(cost, tmp_path):
    with pytest.raises(ValueError, match='unsafe sample cost'):
        train_local_risk_critic(_samples(), LocalPolicyTrainingConfig(), unsafe_sample_cost=cost)
    with pytest.raises(ValueError, match='UNSAFE_SAMPLE_COST_INVALID'):
        train_action_risk_expert([], [], LocalPolicyTrainingConfig(), tmp_path/'output',
                                unsafe_sample_cost=cost)
    assert not (tmp_path/'output').exists()


# 功能：
#   证明默认成本保持原训练结果，显式成本只改变优化，不改变原样本、标签或计分类别。
# 输入：
#   无。
# 输出：
#   None：默认兼容性或原始标签受损时测试失败。
def test_unsafe_cost_is_optimization_only():
    rows = _samples()
    frozen = [row.model_dump() for row in rows]
    config = LocalPolicyTrainingConfig(hidden_feature_count=8, epoch_count=8, batch_size=16)
    first, baseline = train_local_risk_critic(rows, config)
    explicit, _ = train_local_risk_critic(rows, config, unsafe_sample_cost=1.)
    weighted, metrics = train_local_risk_critic(rows, config, unsafe_sample_cost=1.3)
    for name in ('input_weight', 'input_bias', 'risk_weight', 'risk_bias'):
        np.testing.assert_array_equal(getattr(first, name), getattr(explicit, name))
    assert not np.array_equal(first.input_weight, weighted.input_weight)
    assert [row.model_dump() for row in rows] == frozen
    assert metrics.risky_sample_count == baseline.risky_sample_count == 16
    assert metrics.safe_sample_count == baseline.safe_sample_count == 16
