"""Metric near-field representation embedded in the deployed risk graph."""

import json
import math

import numpy as np

from ..control_feature_contract import (
    FLIGHT_STATE_FEATURE_COUNT,
    GEOMETRY_FEATURE_COUNT,
    SENSOR_FEATURE_COUNT,
)
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyTrainingSample
from ..realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from ..runtime_sensor_contracts import oakd_lite_depth_sensor_contract

TRANSFORM_KEY = "dronedream.risk_input_transform"


# 功能：
#   将明确的米制近场范围绑定到当前传感器量程，拒绝非法或超过量程的配置。
# 输入：
#   distance_m：近场距离上限，单位米。
#   position_uncertainty_floor_m：可选的教师定位余量下限；启用时输出版本化径向特征契约。
# 输出：
#   descriptor：包含算法版本、归一化距离和传感器身份的变换契约。
def nearfield_descriptor(distance_m, position_uncertainty_floor_m=None):
    mount = oakd_lite_depth_sensor_contract()
    if (
        type(distance_m) not in (int, float)
        or not math.isfinite(distance_m)
        or not mount.minimum_range_m < distance_m <= mount.maximum_range_m
    ):
        raise ValueError("ACTION_RISK_NEARFIELD_DISTANCE_INVALID")
    descriptor = {
        "kind": "geometry-distance-upper-cap-v1",
        "distance_m": float(distance_m),
        "normalized_cap": float(np.float32(distance_m / mount.maximum_range_m)),
        "sensor_contract_sha256": sha256_json(mount),
    }
    if position_uncertainty_floor_m is not None:
        if (
            type(position_uncertainty_floor_m) not in (int, float)
            or not math.isfinite(position_uncertainty_floor_m)
            or not 0 <= position_uncertainty_floor_m <= 2
        ):
            raise ValueError("ACTION_RISK_POSITION_RESERVE_INVALID")
        descriptor.update(
            kind="uncertainty-reserved-geometry-upper-cap-v1",
            position_uncertainty_floor_m=float(position_uncertainty_floor_m),
            variance_feature_index=SENSOR_FEATURE_COUNT - FLIGHT_STATE_FEATURE_COUNT + 7,
            variance_scale_m2=0.25,
            sigma_multiplier=3.0,
            missing_variance_normalized=4.0,
            reserve_definition="max(floor,3*sqrt(variance)); valid radial distances only",
            scope="learned-input-representation-not-certified-geometric-clearance",
        )
    return descriptor


# 功能：
#   仅截断二十六格几何的距离与射线可达距离，保留掩码、动作、状态和原始标签。
# 输入：
#   samples：不原地修改的动作风险样本。
#   distance_m：已选择的米制近场范围。
#   position_uncertainty_floor_m：可选定位余量下限，不改变原始定位方差或监督标签。
# 输出：
#   projected：用于参考计算的独立样本副本。
def project_nearfield_inputs(samples, distance_m, position_uncertainty_floor_m=None):
    descriptor = nearfield_descriptor(distance_m, position_uncertainty_floor_m)
    cap = descriptor["normalized_cap"]
    projected = []
    for source in samples:
        sample = LocalPolicyTrainingSample.model_validate(source.model_dump())
        if len(sample.realtime_features) != POLICY_REALTIME_FEATURE_COUNT:
            raise ValueError("ACTION_RISK_REALTIME_WIDTH_INVALID")
        reserve = 0.0
        if position_uncertainty_floor_m is not None:
            # 只生成有物理单位的学习特征；径向余量不是整段扫掠净空的证明。
            column = descriptor["variance_feature_index"]
            variance = (
                sample.realtime_features[column]
                if sample.realtime_valid_mask[column] > 0.5
                else 4.0
            )
            reserve = max(
                float(position_uncertainty_floor_m),
                3.0 * math.sqrt(min(4.0, max(0.0, variance)) * 0.25),
            )
            reserve /= oakd_lite_depth_sensor_contract().maximum_range_m
        for offset in range(0, GEOMETRY_FEATURE_COUNT, 5):
            for column in (offset, offset + 1):
                value = sample.realtime_features[column]
                if position_uncertainty_floor_m is not None:
                    value = max(0.0, value - reserve * sample.realtime_valid_mask[column])
                sample.realtime_features[column] = min(value, cap)
        projected.append(sample)
    return projected


# 功能：
#   将近场变换写入实际 ONNX 运算图，保持生产输入不变，不依赖调用方预处理。
# 输入：
#   content：未包含近场变换且权重内嵌的风险网络字节。
#   distance_m：近场距离上限，单位米。
#   position_uncertainty_floor_m：可选定位余量下限；变换与裁剪都嵌入模型，不依赖调用方。
# 输出：
#   wrapped：经检查且带有传感器契约元数据的完整网络字节。
def embed_nearfield_transform(content, distance_m, position_uncertainty_floor_m=None):
    import onnx
    from onnx import helper, numpy_helper

    descriptor = nearfield_descriptor(distance_m, position_uncertainty_floor_m)
    if type(content) is not bytes or not 0 < len(content) <= 16 * 1024**2:
        raise ValueError("ACTION_RISK_EXPORT_BYTES_INVALID")
    graph = onnx.load_model_from_string(content)
    onnx.checker.check_model(graph)
    if any(item.key == TRANSFORM_KEY for item in graph.metadata_props):
        raise ValueError("ACTION_RISK_NEARFIELD_ALREADY_EMBEDDED")
    if any(onnx.external_data_helper.uses_external_data(item) for item in graph.graph.initializer):
        raise ValueError("ACTION_RISK_EXTERNAL_WEIGHTS_FORBIDDEN")
    inputs = {item.name: item for item in graph.graph.input}
    if set(inputs) != {
        "state_features",
        "realtime_features",
        "realtime_valid_mask",
        "proposed_control",
    }:
        raise ValueError("ACTION_RISK_MODEL_CONTRACT_INVALID")
    dimensions = inputs["realtime_features"].type.tensor_type.shape.dim
    if (
        len(dimensions) != 2
        or dimensions[1].dim_value != POLICY_REALTIME_FEATURE_COUNT
        or inputs["realtime_features"].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("ACTION_RISK_REALTIME_WIDTH_INVALID")
    reserved = {"nearfield_geometry", "nearfield_distance_caps", "nearfield_metric_clip"}
    prefix, initializers = _reserve_graph(descriptor)
    reserved.update(item.name for item in initializers)
    for node in prefix:
        reserved.add(node.name)
        reserved.update(node.output)
    names = set(inputs) | {item.name for item in graph.graph.initializer}
    for node in graph.graph.node:
        names.update(node.output)
        names.add(node.name)
        if any(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for attribute in node.attribute
        ):
            raise ValueError("ACTION_RISK_NEARFIELD_NESTED_GRAPH_FORBIDDEN")
    if names & reserved:
        raise ValueError("ACTION_RISK_NEARFIELD_NAME_COLLISION")
    cap = np.full(POLICY_REALTIME_FEATURE_COUNT, np.finfo(np.float32).max, dtype=np.float32)
    for offset in range(0, GEOMETRY_FEATURE_COUNT, 5):
        cap[offset : offset + 2] = descriptor["normalized_cap"]
    consumers = 0
    for node in graph.graph.node:
        for index, name in enumerate(node.input):
            if name == "realtime_features":
                node.input[index] = "nearfield_geometry"
                consumers += 1
    if not consumers:
        raise ValueError("ACTION_RISK_REALTIME_INPUT_UNUSED")
    graph.graph.initializer.append(numpy_helper.from_array(cap, "nearfield_distance_caps"))
    graph.graph.initializer.extend(initializers)
    graph.graph.node.insert(
        0,
        helper.make_node(
            "Min",
            ["reserved_geometry" if prefix else "realtime_features", "nearfield_distance_caps"],
            ["nearfield_geometry"],
            name="nearfield_metric_clip",
        ),
    )
    for node in reversed(prefix):
        graph.graph.node.insert(0, node)
    metadata = graph.metadata_props.add()
    metadata.key = TRANSFORM_KEY
    metadata.value = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    onnx.checker.check_model(graph)
    wrapped = graph.SerializeToString()
    return wrapped


# 功能：生成固定且可逐节点核对的定位余量前缀，缺失方差采用编码上界，不伪造精确定位。
# 输入：descriptor：已验证的学习特征契约。输出：节点和常量；不提供飞行放行判断。
def _reserve_graph(descriptor):
    if descriptor["kind"] == "geometry-distance-upper-cap-v1":
        return [], []
    from onnx import helper, numpy_helper

    constants = {
        "reserve_variance_index": np.asarray([descriptor["variance_feature_index"]], np.int64),
        "reserve_zero": np.asarray(0.0, np.float32),
        "reserve_four": np.asarray(4.0, np.float32),
        "reserve_half": np.asarray(0.5, np.float32),
        "reserve_variance_scale": np.asarray(0.25, np.float32),
        "reserve_three": np.asarray(3.0, np.float32),
        "reserve_floor_m": np.asarray(descriptor["position_uncertainty_floor_m"], np.float32),
        "reserve_range_m": np.asarray(
            oakd_lite_depth_sensor_contract().maximum_range_m, np.float32
        ),
    }
    mask = np.zeros(POLICY_REALTIME_FEATURE_COUNT, np.float32)
    floors = np.full(POLICY_REALTIME_FEATURE_COUNT, -np.finfo(np.float32).max, np.float32)
    for index in range(0, GEOMETRY_FEATURE_COUNT, 5):
        mask[index : index + 2] = 1.0
        floors[index : index + 2] = 0.0
    constants.update(reserve_distance_mask=mask, reserve_distance_floor=floors)
    definitions = [
        (
            "Gather",
            ["realtime_features", "reserve_variance_index"],
            "reserve_variance",
            {"axis": 1},
        ),
        (
            "Gather",
            ["realtime_valid_mask", "reserve_variance_index"],
            "reserve_variance_mask",
            {"axis": 1},
        ),
        ("Greater", ["reserve_variance_mask", "reserve_half"], "reserve_variance_valid", {}),
        (
            "Where",
            ["reserve_variance_valid", "reserve_variance", "reserve_four"],
            "reserve_known_variance",
            {},
        ),
        ("Max", ["reserve_known_variance", "reserve_zero"], "reserve_nonnegative_variance", {}),
        ("Min", ["reserve_nonnegative_variance", "reserve_four"], "reserve_bounded_variance", {}),
        ("Mul", ["reserve_bounded_variance", "reserve_variance_scale"], "reserve_variance_m2", {}),
        ("Sqrt", ["reserve_variance_m2"], "reserve_sigma_m", {}),
        ("Mul", ["reserve_sigma_m", "reserve_three"], "reserve_three_sigma_m", {}),
        ("Max", ["reserve_three_sigma_m", "reserve_floor_m"], "reserve_margin_m", {}),
        ("Div", ["reserve_margin_m", "reserve_range_m"], "reserve_normalized", {}),
        ("Mul", ["reserve_normalized", "reserve_distance_mask"], "reserve_distances", {}),
        ("Mul", ["reserve_distances", "realtime_valid_mask"], "reserve_valid_distances", {}),
        (
            "Sub",
            ["realtime_features", "reserve_valid_distances"],
            "reserve_subtracted_geometry",
            {},
        ),
        ("Max", ["reserve_subtracted_geometry", "reserve_distance_floor"], "reserved_geometry", {}),
    ]
    nodes = [
        helper.make_node(op, inputs, [output], name=output + "_op", **attributes)
        for op, inputs, output, attributes in definitions
    ]
    return nodes, [numpy_helper.from_array(value, name) for name, value in constants.items()]


# 功能：
#   核对图内的真实变换、训练回执及传感器身份，拒绝只声明元数据但绕过裁剪的图。
# 输入：
#   content：实际权重字节；descriptor：回执中的可选变换契约。
#   sensor_sha256：目标模型包使用的传感器身份。
# 输出：
#   None：变换声明或实际运算不一致时抛出异常。
def validate_nearfield_graph(content, descriptor, sensor_sha256):
    import onnx
    from onnx import numpy_helper

    from dronedream_plugin_sdk.protocol import decode_json

    graph = onnx.load_model_from_string(content)
    metadata = [item.value for item in graph.metadata_props if item.key == TRANSFORM_KEY]
    if descriptor is None:
        if metadata:
            raise ValueError("ACTION_RISK_UNRECORDED_INPUT_TRANSFORM")
        return
    if (
        type(descriptor) is not dict
        or descriptor
        != nearfield_descriptor(
            descriptor.get("distance_m"), descriptor.get("position_uncertainty_floor_m")
        )
        or descriptor["sensor_contract_sha256"] != sensor_sha256
        or len(metadata) != 1
        or decode_json(metadata[0], limit=4096) != descriptor
    ):
        raise ValueError("ACTION_RISK_NEARFIELD_CONTRACT_MISMATCH")
    nodes = list(graph.graph.node)
    prefix, constants = _reserve_graph(descriptor)
    if prefix:
        if [node.SerializeToString() for node in nodes[: len(prefix)]] != [
            node.SerializeToString() for node in prefix
        ]:
            raise ValueError("ACTION_RISK_POSITION_RESERVE_GRAPH_INVALID")
        for expected_constant in constants:
            matches = [
                item for item in graph.graph.initializer if item.name == expected_constant.name
            ]
            if (
                len(matches) != 1
                or matches[0].SerializeToString() != expected_constant.SerializeToString()
            ):
                raise ValueError("ACTION_RISK_POSITION_RESERVE_CONSTANT_INVALID")
        nodes = nodes[len(prefix) :]
        intermediates = {output for node in prefix for output in node.output}
        if any(name in intermediates for node in nodes[1:] for name in node.input):
            raise ValueError("ACTION_RISK_POSITION_RESERVE_BYPASSED")
    if (
        not nodes
        or nodes[0].op_type != "Min"
        or nodes[0].domain
        or nodes[0].name != "nearfield_metric_clip"
        or list(nodes[0].input)
        != ["reserved_geometry" if prefix else "realtime_features", "nearfield_distance_caps"]
        or list(nodes[0].output) != ["nearfield_geometry"]
        or any("realtime_features" in node.input for node in nodes[1:])
        or not any("nearfield_geometry" in node.input for node in nodes[1:])
    ):
        raise ValueError("ACTION_RISK_NEARFIELD_GRAPH_MISMATCH")
    weights = [item for item in graph.graph.initializer if item.name == "nearfield_distance_caps"]
    if len(weights) != 1 or onnx.external_data_helper.uses_external_data(weights[0]):
        raise ValueError("ACTION_RISK_NEARFIELD_CAPS_INVALID")
    caps = numpy_helper.to_array(weights[0])
    expected = np.full(POLICY_REALTIME_FEATURE_COUNT, np.finfo(np.float32).max, dtype=np.float32)
    for offset in range(0, GEOMETRY_FEATURE_COUNT, 5):
        expected[offset : offset + 2] = descriptor["normalized_cap"]
    if (
        caps.dtype != np.float32
        or caps.shape != expected.shape
        or not np.array_equal(caps, expected)
    ):
        raise ValueError("ACTION_RISK_NEARFIELD_CAPS_INVALID")
