"""Synthetic evaluator tests, not flight or product-model acceptance evidence."""

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from onnx import TensorProto, helper
from test_action_conditioned_risk import _samples

from dronedream_agent_core.training import action_risk_evaluation as evaluation


# 功能：
#   创建显式合成来源与按真实动作第一轴分类的微型图，验证标签必须对应同一提案。
# 输入：
#   tmp_path、monkeypatch：独占目录与加载器替身。
# 输出：
#   case：真实 ONNX 会话输入及明确合成的二十组动作对照。
@pytest.fixture
def case(tmp_path, monkeypatch):
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [None, width])
              for name, width in [('state_features', 46), ('realtime_features', 446),
                                  ('realtime_valid_mask', 446), ('proposed_control', 4)]]
    graph = helper.make_graph([
        helper.make_node('Gather', ['proposed_control', 'axis'], ['first'], axis=1),
        helper.make_node('Mul', ['first', 'gain'], ['logit']),
        helper.make_node('Sigmoid', ['logit'], ['risk_score'])], 'synthetic-action-risk', inputs,
        [helper.make_tensor_value_info('risk_score', TensorProto.FLOAT, [None, 1])],
        [helper.make_tensor('axis', TensorProto.INT64, [1], [0]),
         helper.make_tensor('gain', TensorProto.FLOAT, [1], [10.])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    content = model.SerializeToString()
    path = tmp_path/'risk.onnx'
    path.write_bytes(content)
    prototypes = _samples()
    samples = [prototypes[i % 2].model_copy(deep=True) for i in range(40)]
    dataset = SimpleNamespace(samples=samples,
        records=[dict(observation_sha256=f'{i // 2:064x}') for i in range(40)],
        groups=frozenset({'a'*64}), receipt_sha256='b'*64, teacher_config_sha256='c'*64)
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(evaluation, 'load_action_risk_dataset', loader)
    result = SimpleNamespace(path=path, dataset=dataset, loader=loader,
        arguments=dict(model_sha256=hashlib.sha256(content).hexdigest(), teacher_sha256='c'*64,
                       excluded_groups={'d'*64}, maximum_latency_ms=60000.))
    return result


# 功能：
#   实际图收到风险提案而非行为目标，独立对照的原门槛通过但不授予飞行资格。
# 输入：
#   case：合成风险图与动作绑定数据。
# 输出：
#   None：通过断言核对。
def test_actual_graph_uses_bound_proposed_action(case):
    for sample in case.dataset.samples:
        sample.target_pilot_control = [0., 0., 0., 0.]
    report = evaluation.evaluate_action_risk_artifact(
        case.path, [case.path.parent], **case.arguments)
    assert report['independent_test_passed'] is True
    assert report['qualified_for_flight'] is False
    assert report['metrics']['risk_hold_recall'] == 1.
    assert report['metrics']['safe_motion_recall'] == 1.
    assert report['coverage']['independent_observation_count'] == 20
    assert report['action_discrimination']['correct_fraction'] == 1.


# 功能：
#   覆盖不足时提前拒绝且不调用模型，不能以全安全数据计算出完美风险成绩。
# 输入：
#   case、monkeypatch：合成数据与禁止推理的会话替身。
# 输出：
#   None：通过断言核对。
def test_missing_risk_class_precedes_inference(case, monkeypatch):
    import onnxruntime
    for sample in case.dataset.samples:
        sample.risk_target = 0.
    session = Mock(side_effect=AssertionError('inference must not happen'))
    monkeypatch.setattr(onnxruntime, 'InferenceSession', session)
    report = evaluation.evaluate_action_risk_artifact(
        case.path, [case.path.parent], **case.arguments)
    assert not report['independent_test_passed'] and not report['inference_performed']
    assert report['metrics'] is None
    assert 'UNSAFE_OBSERVATION_COUNT_BELOW_TWENTY' in report['issue_codes']
    session.assert_not_called()


# 功能：
#   模型摘要、教师来源或训练留出分组不匹配时，在实际图调用前拒绝。
# 输入：
#   case：合成上下文；change：非法来源字段。
# 输出：
#   None：通过异常断言核对。
@pytest.mark.parametrize('change', [dict(model_sha256='e'*64), dict(teacher_sha256='e'*64),
                                  dict(excluded_groups={'a'*64}), dict(excluded_groups=set()),
                                  dict(maximum_latency_ms=True),
                                  dict(maximum_latency_ms=float('nan'))])
def test_identity_and_budget_rejected(case, change):
    with pytest.raises(ValueError):
        evaluation.evaluate_action_risk_artifact(case.path, [case.path.parent],
                                                **(case.arguments | change))


# 功能：
#   同一来源不能通过重复目录复制充足观测数量。
# 输入：
#   case：单一合成来源。
# 输出：
#   None：通过异常断言核对。
def test_repeated_dataset_cannot_inflate_coverage(case):
    with pytest.raises(ValueError, match='SOURCE_OVERLAP'):
        evaluation.evaluate_action_risk_artifact(case.path, [case.path.parent]*2, **case.arguments)


# 功能：
#   输出非法形状、类型或范围必须拒绝，不能裁剪后参与计分。
# 输入：
#   case、monkeypatch：合成输入及会话替身；value：非法图输出。
# 输出：
#   None：通过异常断言核对。
@pytest.mark.parametrize('value', [np.array([[np.nan]], np.float32), np.array([[1.1]], np.float32),
                                 np.array([.5], np.float32), np.array([[.5]], np.float64)])
def test_invalid_graph_output_is_not_normalized(case, monkeypatch, value):
    import onnxruntime
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name=name) for name in
        ('state_features', 'realtime_features', 'realtime_valid_mask', 'proposed_control')]
    session.get_outputs.return_value = [SimpleNamespace(name='risk_score')]
    session.run.return_value = [value]
    monkeypatch.setattr(onnxruntime, 'InferenceSession', Mock(return_value=session))
    with pytest.raises(ValueError, match='OUTPUT_INVALID'):
        evaluation.evaluate_action_risk_artifact(case.path, [case.path.parent], **case.arguments)
