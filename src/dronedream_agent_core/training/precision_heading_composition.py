"""Compose frozen translation and learned yaw without changing either source model.

The result is a new, unqualified artifact. Source receipts and independent
four-axis evaluation must be supplied separately; composition is not training.
"""

import hashlib

from ..causal_control import CONTROL_HISTORY_CONTRACT_SHA256, CONTROL_HISTORY_WIDTH
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..heading_context import HEADING_CONTEXT_SHA256, HEADING_CONTEXT_WIDTH
from ..precision_heading_input import PRECISION_HEADING_ARCHITECTURE
from .artifact_assembly import validate_embedded_graph

COMPOSITION_ARCHITECTURE = PRECISION_HEADING_ARCHITECTURE


# 功能：
#   核对冻结图的字节身份、嵌入式权重和标准算子，拒绝隐式版本转换或外部代码。
# 输入：
#   content：明确提供的 ONNX 字节。
#   expected_sha256：外部选定的来源摘要。
# 输出：
#   model：通过结构和来源验证的独立 ONNX 模型。
def _read_model(content: bytes, expected_sha256: str):
    import onnx
    from google.protobuf.message import DecodeError

    if (type(content) is not bytes or not 0 < len(content) <= 16 * 1024**2
            or type(expected_sha256) is not str
            or hashlib.sha256(content).hexdigest() != expected_sha256):
        raise ValueError('HEADING_COMPOSITION_SOURCE_MISMATCH')
    try:
        model = onnx.load_model_from_string(content)
    except DecodeError as error:
        raise ValueError('HEADING_COMPOSITION_SOURCE_PARSE_INVALID') from error
    validate_embedded_graph(content, input_names=[i.name for i in model.graph.input],
                            output_names=[i.name for i in model.graph.output])
    if (model.functions or model.graph.sparse_initializer or len(model.opset_import) != 1
            or model.opset_import[0].domain != '' or model.opset_import[0].version != 17
            or any(node.domain not in ('', 'ai.onnx') for node in model.graph.node)
            or any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for node in model.graph.node for attribute in node.attribute)):
        raise ValueError('HEADING_COMPOSITION_GRAPH_CONTRACT_INVALID')
    return model


# 功能：
#   检查具名张量的浮点类型与完整维度，动态批次仅在明确允许的第一维接受。
# 输入：
#   value：ONNX 张量声明。
#   shape：预期维度，None 表示允许动态的一批。
# 输出：
#   valid：张量是否符合预期接口。
def _shape_matches(value, shape) -> bool:
    import onnx

    tensor = value.type.tensor_type
    dimensions = tensor.shape.dim
    valid = (tensor.elem_type == onnx.TensorProto.FLOAT and len(dimensions) == len(shape)
             and all((dimension.dim_value == expected if expected is not None
                      else dimension.dim_value == 1 or bool(dimension.dim_param))
                     for dimension, expected in zip(dimensions, shape, strict=True)))
    return valid


# 功能：
#   1. 保留原模型的三轴平移和其他输出，将第四轴明确交给冻结的学习型偏航网络。
#   2. 为两分支隔离名称并绑定输入协议及原权重摘要；不继承任何飞行资格。
# 输入：
#   translation_bytes、heading_bytes：原因果控制图与偏航图的独立字节。
#   translation_sha256、heading_sha256：调用方明确冻结的两个来源摘要。
#   role：明确限定精细或巡航角色；默认值保留历史组合的字节及回执身份。
# 输出：
#   content：具有额外 heading_context 输入的新图字节。
#   receipt：记录分工与来源的新组合回执，不冒充原模型训练回执。
def compose_precision_heading(translation_bytes: bytes, heading_bytes: bytes, *, translation_sha256: str, heading_sha256: str,
                              role: str = 'precision-maneuver-policy'):
    import onnx
    from onnx import compose, helper

    if role not in ('precision-maneuver-policy', 'local-navigation-policy'):
        raise ValueError('HEADING_COMPOSITION_ROLE_INVALID')
    translation = _read_model(translation_bytes, translation_sha256)
    heading = _read_model(heading_bytes, heading_sha256)
    metadata = {entry.key: entry.value for entry in translation.metadata_props}
    heading_metadata = {entry.key: entry.value for entry in heading.metadata_props}
    inputs = {entry.name: entry for entry in translation.graph.input}
    outputs = {entry.name: entry for entry in translation.graph.output}
    expected_inputs = {'state_features', 'realtime_features', 'realtime_valid_mask',
                       'control_history', 'control_history_mask', 'visual_features'}
    if (metadata.get('architecture') != 'causal-gru-control'
            or metadata.get('feature_contract_sha256') != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or metadata.get('history_contract_sha256') != CONTROL_HISTORY_CONTRACT_SHA256
            or set(inputs) != expected_inputs
            or set(outputs) != {'candidate_scores', 'action_scores', 'risk_score', 'pilot_control'}
            or not _shape_matches(outputs['pilot_control'], (1, 4))):
        raise ValueError('HEADING_COMPOSITION_TRANSLATION_CONTRACT_INVALID')
    history_length = metadata.get('history_length', '')
    if not history_length.isdecimal() or not 4 <= int(history_length) <= 32:
        raise ValueError('HEADING_COMPOSITION_HISTORY_INVALID')
    shapes = {'state_features': (1, 46), 'realtime_features': (1, 446),
              'realtime_valid_mask': (1, 446),
              'control_history': (1, int(history_length), CONTROL_HISTORY_WIDTH),
              'control_history_mask': (1, int(history_length))}
    if any(not _shape_matches(inputs[name], shape) for name, shape in shapes.items()):
        raise ValueError('HEADING_COMPOSITION_TRANSLATION_SHAPE_INVALID')
    if (heading_metadata.get('architecture') != 'heading-availability-v1'
            or heading_metadata.get('feature_contract_sha256') != HEADING_CONTEXT_SHA256
            or heading_metadata.get('qualified_for_flight') != 'false'
            or [value.name for value in heading.graph.input] != ['heading_context']
            or [value.name for value in heading.graph.output] != ['yaw_axis']
            or not _shape_matches(heading.graph.input[0], (None, HEADING_CONTEXT_WIDTH))
            or not _shape_matches(heading.graph.output[0], (None, 1))):
        raise ValueError('HEADING_COMPOSITION_YAW_CONTRACT_INVALID')

    # 所有原始节点、常量及中间值都加前缀，外部输入通过 Identity 显式桥接；
    # 不能假定两个独立导出的网络没有使用相同的内部名字。
    translation_branch = compose.add_prefix(translation, 'translation/')
    heading_branch = compose.add_prefix(heading, 'heading/')
    nodes = [helper.make_node('Identity', [name], ['translation/' + name]) for name in inputs]
    nodes.append(helper.make_node('Identity', ['heading_context'], ['heading/heading_context']))
    nodes.extend(translation_branch.graph.node)
    nodes.extend(heading_branch.graph.node)
    nodes.append(helper.make_node('Gather', ['translation/pilot_control', 'composition/xyz_indices'],
                                  ['composition/translation_axes'], axis=1))
    nodes.append(helper.make_node('Concat', ['composition/translation_axes', 'heading/yaw_axis'],
                                  ['pilot_control'], axis=1))
    nodes.extend(helper.make_node('Identity', ['translation/' + name], [name])
                 for name in outputs if name != 'pilot_control')
    heading_input = helper.make_tensor_value_info('heading_context', onnx.TensorProto.FLOAT,
                                                   [1, HEADING_CONTEXT_WIDTH])
    graph = helper.make_graph(nodes, COMPOSITION_ARCHITECTURE,
                              [*translation.graph.input, heading_input], list(translation.graph.output),
                              [*translation_branch.graph.initializer, *heading_branch.graph.initializer,
                               helper.make_tensor('composition/xyz_indices', onnx.TensorProto.INT64,
                                                  [3], [0, 1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = max(translation.ir_version, heading.ir_version)
    helper.set_model_props(model, {**metadata, 'architecture': COMPOSITION_ARCHITECTURE,
        'heading_context_sha256': HEADING_CONTEXT_SHA256,
        'translation_source_sha256': translation_sha256, 'heading_source_sha256': heading_sha256,
        'yaw_limit_dps': '20.0', 'qualified_for_flight': 'false'})
    onnx.checker.check_model(model)
    content = model.SerializeToString()
    purpose = ('frozen-navigation-heading-composition' if role == 'local-navigation-policy'
               else 'frozen-precision-heading-composition')
    receipt = dict(purpose=purpose, architecture=COMPOSITION_ARCHITECTURE,
        artifact_sha256=hashlib.sha256(content).hexdigest(), translation_sha256=translation_sha256,
        heading_sha256=heading_sha256, heading_context_sha256=HEADING_CONTEXT_SHA256,
        axis_ownership=['translation', 'translation', 'translation', 'heading'],
        yaw_limit_dps=20., optimized=False, qualified_for_flight=False,
        inherited_training_metrics=False, independent_four_axis_validation_required=True)
    return content, receipt
