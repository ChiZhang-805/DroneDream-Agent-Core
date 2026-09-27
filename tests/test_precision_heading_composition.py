"""Frozen graph composition and axis ownership; synthetic tests are not flight evidence."""

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper

from dronedream_agent_core.causal_control import CONTROL_HISTORY_CONTRACT_SHA256
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256
from dronedream_agent_core.training.precision_heading_composition import compose_precision_heading


# 功能：
#   创建具有相同内部常量名的两个最小测试图，检查组合后的作用域与四轴归属。
# 输入：
#   无。
# 输出：
#   translation、heading：声明当前接口的合成 ONNX 图，不是生产权重。
def graphs():
    shapes = dict(state_features=[1, 46], realtime_features=[1, 446],
        realtime_valid_mask=[1, 446], control_history=[1, 16, 939],
        control_history_mask=[1, 16], visual_features=[1, 139])
    values = dict(candidate_scores=np.zeros((1, 8), np.float32),
        action_scores=np.array([[1., 0., 0., 0.]], np.float32),
        risk_score=np.array([[.1]], np.float32), pilot_control=np.array([[.1, .2, .3, .4]], np.float32))
    nodes = [helper.make_node('Constant', [], [name], name='shared-name',
                              value=onnx.numpy_helper.from_array(value)) for name, value in values.items()]
    translation = helper.make_model(helper.make_graph(nodes, 'translation',
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in shapes.items()],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape) for name, value in values.items()]),
        opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    helper.set_model_props(translation, dict(architecture='causal-gru-control', history_length='16',
        feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        history_contract_sha256=CONTROL_HISTORY_CONTRACT_SHA256))
    heading = helper.make_model(helper.make_graph([
        helper.make_node('Gather', ['heading_context', 'shared-name'], ['selected'], axis=1),
        helper.make_node('Tanh', ['selected'], ['yaw_axis'])], 'heading',
        [helper.make_tensor_value_info('heading_context', onnx.TensorProto.FLOAT, ['batch', 23])],
        [helper.make_tensor_value_info('yaw_axis', onnx.TensorProto.FLOAT, ['batch', 1])],
        [helper.make_tensor('shared-name', onnx.TensorProto.INT64, [1], [0])]),
        opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    helper.set_model_props(heading, dict(architecture='heading-availability-v1',
        feature_contract_sha256=HEADING_CONTEXT_SHA256, qualified_for_flight='false'))
    return translation, heading


# 功能：
#   使用调用前冻结的摘要执行组合，不允许测试跳过同一条来源检查链。
# 输入：
#   translation、heading：测试图对象。
# 输出：
#   result：组合字节及独立回执。
def compose_pair(translation, heading):
    left, right = translation.SerializeToString(), heading.SerializeToString()
    result = compose_precision_heading(left, right, translation_sha256=hashlib.sha256(left).hexdigest(),
                                        heading_sha256=hashlib.sha256(right).hexdigest())
    return result


# 功能：
#   实际运行组合图，验证偏航变化只影响第四轴，不改变平移、其他输出或原始图。
# 输入：
#   yaw_input：提供给偏航网络的测试输入。
# 输出：
#   None：三轴来源、第四轴来源和未准入声明均满足断言。
@pytest.mark.parametrize('yaw_input', [-1., 0., 1.])
def test_actual_composed_inference_preserves_axis_ownership(yaw_input):
    translation, heading = graphs()
    original = translation.SerializeToString(), heading.SerializeToString()
    content, receipt = compose_pair(translation, heading)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(content, sess_options=options, providers=['CPUExecutionProvider'])
    feeds = {item.name: np.zeros(item.shape, np.float32) for item in session.get_inputs()}
    feeds['heading_context'][0, 0] = yaw_input
    pilot, risk = session.run(['pilot_control', 'risk_score'], feeds)
    np.testing.assert_allclose(pilot, [[.1, .2, .3, np.tanh(yaw_input)]], atol=1e-7)
    np.testing.assert_allclose(risk, [[.1]], atol=1e-7)
    assert original == (translation.SerializeToString(), heading.SerializeToString())
    assert receipt['artifact_sha256'] == hashlib.sha256(content).hexdigest()
    assert receipt['qualified_for_flight'] is False
    assert receipt['inherited_training_metrics'] is False


# 功能：
#   拒绝错误协议、错误张量布局和自行声称具有飞行资格的偏航来源。
# 输入：
#   damage：需要破坏的独立边界。
# 输出：
#   None：组合在输出任何新模型前失败。
@pytest.mark.parametrize('damage', ['heading-contract', 'translation-contract', 'qualified',
                                  'heading-shape', 'pilot-shape', 'opset', 'custom-op'])
def test_composition_rejects_incompatible_sources(damage):
    translation, heading = graphs()
    if damage == 'heading-contract':
        heading.metadata_props[1].value = '0' * 64
    elif damage == 'translation-contract':
        translation.metadata_props[2].value = '0' * 64
    elif damage == 'qualified':
        heading.metadata_props[2].value = 'true'
    elif damage == 'heading-shape':
        heading.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 24
    elif damage == 'pilot-shape':
        translation.graph.output[3].type.tensor_type.shape.dim[1].dim_value = 3
    elif damage == 'opset':
        heading.opset_import[0].version = 18
    else:
        heading.graph.node[0].domain = 'custom'
        heading.opset_import.append(helper.make_opsetid('custom', 1))
    with pytest.raises(ValueError, match='HEADING_COMPOSITION'):
        compose_pair(translation, heading)


# 功能：
#   阻止组合时把不同来源的图伪装成已冻结权重。
# 输入：
#   无。
# 输出：
#   None：摘要不匹配在 ONNX 解码前被拒绝。
def test_composition_rejects_unbound_bytes():
    translation, heading = graphs()
    left, right = translation.SerializeToString(), heading.SerializeToString()
    with pytest.raises(ValueError, match='SOURCE_MISMATCH'):
        compose_precision_heading(left, right, translation_sha256='0' * 64,
                                  heading_sha256=hashlib.sha256(right).hexdigest())
