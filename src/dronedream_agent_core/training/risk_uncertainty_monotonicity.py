"""Exact partial monotonicity in measured positional variance, not flight proof."""
import numpy as np

from ..control_feature_contract import FLIGHT_STATE_FEATURE_COUNT, SENSOR_FEATURE_COUNT
from ..local_policy_packages import LOCAL_POLICY_STATE_FEATURE_COUNT
from ..realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT

UNCERTAINTY_COLUMNS = (
    9,
    LOCAL_POLICY_STATE_FEATURE_COUNT + SENSOR_FEATURE_COUNT - FLIGHT_STATE_FEATURE_COUNT + 7,
)


# 功能：约束每条协方差到风险输出的有效路径非负，允许其他特征和输出权重保留双符号。
# 输入：weight：压缩首层权重；readout：风险读出；columns：对应原始特征列索引。
# 输出：原位投影冲突路径为零；固定其他输入时，增加定位方差不能降低风险。
def project_uncertainty_paths(weight, readout, columns):
    indices = np.flatnonzero(np.isin(columns, UNCERTAINTY_COLUMNS))
    rows = weight[indices]
    rows[rows * readout[:, 0] < 0.] = 0.
    weight[indices] = rows


# 功能：检查实际部署图的单隐层运算和协方差路径符号，而非只相信回执文字。
# 输入：content：实际 ONNX 字节；enabled：严格布尔声明。
# 输出：None；声明启用但存在反向路径或未知运算时拒绝组合。
def validate_uncertainty_graph(content, enabled):
    if type(enabled) is not bool:
        raise ValueError('ACTION_RISK_UNCERTAINTY_CONSTRAINT_INVALID')
    if not enabled:
        return
    import onnx
    from onnx import numpy_helper

    graph = onnx.load_model_from_string(content)
    onnx.checker.check_model(graph)
    nodes = list(graph.graph.node)
    realtime = 'realtime_features'
    if nodes and nodes[0].op_type == 'Min':
        first = nodes.pop(0)
        if (list(first.input) != ['realtime_features', 'nearfield_distance_caps']
                or list(first.output) != ['nearfield_geometry'] or first.attribute or first.domain):
            raise ValueError('ACTION_RISK_UNCERTAINTY_GRAPH_INVALID')
        realtime = 'nearfield_geometry'
    expected = [
        ('Concat', ['state_features', realtime, 'realtime_valid_mask', 'proposed_control'],
         ['flat_features']),
        ('MatMul', ['flat_features', 'input_weight'], ['hidden_linear']),
        ('Add', ['hidden_linear', 'input_bias'], ['hidden_biased']),
        ('Relu', ['hidden_biased'], ['hidden']),
        ('MatMul', ['hidden', 'risk_weight'], ['risk_linear']),
        ('Add', ['risk_linear', 'risk_bias'], ['risk_logit']),
        ('Sigmoid', ['risk_logit'], ['risk_score']),
    ]
    if len(nodes) != len(expected) or [o.name for o in graph.graph.output] != ['risk_score']:
        raise ValueError('ACTION_RISK_UNCERTAINTY_GRAPH_INVALID')
    for index, (node, (kind, inputs, outputs)) in enumerate(zip(nodes, expected, strict=True)):
        if (node.domain or node.op_type != kind or list(node.input) != inputs
                or list(node.output) != outputs):
            raise ValueError('ACTION_RISK_UNCERTAINTY_GRAPH_INVALID')
        if index == 0:
            if (len(node.attribute) != 1 or node.attribute[0].name != 'axis'
                    or node.attribute[0].i != 1):
                raise ValueError('ACTION_RISK_UNCERTAINTY_GRAPH_INVALID')
        elif node.attribute:
            raise ValueError('ACTION_RISK_UNCERTAINTY_GRAPH_INVALID')
    values = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    weight, readout = values.get('input_weight'), values.get('risk_weight')
    width = LOCAL_POLICY_STATE_FEATURE_COUNT + 2 * POLICY_REALTIME_FEATURE_COUNT + 4
    if (weight is None or readout is None or weight.ndim != 2 or weight.shape[0] != width
            or readout.shape != (weight.shape[1], 1)
            or not np.isfinite(weight).all() or not np.isfinite(readout).all()):
        raise ValueError('ACTION_RISK_UNCERTAINTY_WEIGHTS_INVALID')
    if np.any(weight[list(UNCERTAINTY_COLUMNS)] * readout[:, 0] < 0.):
        raise ValueError('ACTION_RISK_UNCERTAINTY_PATH_NEGATIVE')
