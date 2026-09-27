"""Position reserve is an explicit learned representation, never flight authority."""

import copy

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from test_action_conditioned_risk import _samples
from test_risk_nearfield import _distance_network

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingConfig
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


# 功能：为真实 ONNX 数值测试构造不同协方差、有效性和有符号运动输入，非训练数据。
# 输入：无。输出：隔离样本，协方差缺失时不得假定精确定位。
def _reserve_samples():
    samples = []
    for covariance, mask in ((0.0, 1.0), (0.002, 1.0), (0.21, 1.0), (4.0, 1.0), (0.002, 0.0)):
        row = copy.deepcopy(_samples()[0])
        row.realtime_features = [0.2] * POLICY_REALTIME_FEATURE_COUNT
        row.realtime_valid_mask = [1.0] * POLICY_REALTIME_FEATURE_COUNT
        row.realtime_features[423] = covariance
        row.realtime_valid_mask[423] = mask
        row.realtime_features[0] = 0.015
        row.realtime_features[5] = -0.01
        row.realtime_valid_mask[5] = 0.0
        row.realtime_features[425] = -0.4
        samples.append(row)
    return samples


# 功能：核对米制余量、近场裁剪顺序和未观测方位掩码，保留原始样本及有符号其他特征。
# 输入：无。输出：任何标签、状态或非几何列被改写时测试失败。
def test_position_reserve_projection_and_ownership():
    rows = _reserve_samples()
    originals = [row.model_dump() for row in rows]
    projected = project_nearfield_inputs(rows, 4.0, 0.1)
    for source, actual in zip(rows, projected, strict=True):
        expected = (
            max(
                0.1,
                3
                * np.sqrt(
                    0.25
                    * (source.realtime_features[423] if source.realtime_valid_mask[423] else 4.0)
                ),
            )
            / 19.1
        )
        assert actual.realtime_features[0] == pytest.approx(max(0.0, 0.015 - expected))
        assert actual.realtime_features[1] == pytest.approx(0.2 - expected)
        assert actual.realtime_features[425] == -0.4
        assert actual.realtime_valid_mask == source.realtime_valid_mask
        document = actual.model_dump()
        document["realtime_features"] = source.model_dump()["realtime_features"]
        assert document == source.model_dump()
    assert [row.model_dump() for row in rows] == originals
    # 同一径向可用距离使用相同余量表达；不能先截断远距离再扣除余量。
    far = copy.deepcopy(rows[-1])
    far.realtime_features[0] = 1.0
    assert project_nearfield_inputs([far], 4.0, 0.1)[0].realtime_features[0] == pytest.approx(
        4.0 / 19.1
    )


# 功能：用非零真实网络检查 NumPy 和 ONNX，遗漏图变换或声明时必须失败。
# 输入：tmp_path：隔离目录。输出：经源张量调用的数值等价断言。
def test_position_reserve_real_export_and_receipt(tmp_path):
    model, raw, _ = _distance_network(tmp_path)
    content = embed_nearfield_transform(raw, 4.0, 0.1)
    descriptor = nearfield_descriptor(4.0, 0.1)
    validate_nearfield_graph(content, descriptor, descriptor["sensor_contract_sha256"])
    assert (
        verify_risk_export(
            model,
            content,
            _reserve_samples(),
            geometry_distance_cap_m=4.0,
            position_uncertainty_floor_m=0.1,
        )
        < 1e-5
    )
    with pytest.raises(ValueError, match="EQUIVALENCE_FAILED"):
        verify_risk_export(
            model,
            raw,
            _reserve_samples(),
            geometry_distance_cap_m=4.0,
            position_uncertainty_floor_m=0.1,
        )
    with pytest.raises(ValueError, match="NEARFIELD_CONTRACT_MISMATCH"):
        validate_nearfield_graph(
            content, nearfield_descriptor(4.0), descriptor["sensor_contract_sha256"]
        )


# 功能：核查图中实际常量、前缀和绕过引用，不信任只写在元数据中的余量声明。
# 输入：tmp_path：隔离目录；tamper：破坏种类。输出：被破坏的图无法通过模型组合校验。
@pytest.mark.parametrize("tamper", ["sigma", "subtract", "bypass", "duplicate"])
def test_position_reserve_graph_tamper_rejected(tmp_path, tamper):
    _, raw, _ = _distance_network(tmp_path)
    graph = onnx.load_model_from_string(embed_nearfield_transform(raw, 4.0, 0.1))
    if tamper == "sigma":
        for item in graph.graph.initializer:
            if item.name == "reserve_three":
                item.CopyFrom(numpy_helper.from_array(np.asarray(0.1, np.float32), item.name))
    elif tamper == "subtract":
        graph.graph.node[13].op_type = "Add"
    elif tamper == "bypass":
        graph.graph.node[16].input[1] = "reserved_geometry"
    else:
        item = next(item for item in graph.graph.initializer if item.name == "reserve_three")
        graph.graph.initializer.append(item)
    descriptor = nearfield_descriptor(4.0, 0.1)
    with pytest.raises(ValueError, match="POSITION_RESERVE"):
        validate_nearfield_graph(
            graph.SerializeToString(), descriptor, descriptor["sensor_contract_sha256"]
        )


# 功能：拒绝无效下限和未经证明的复合单调声明，防止在输出文件创建后才发现参数错误。
# 输入：tmp_path：隔离目录。输出：拒绝边界配置且不落地候选。
def test_position_reserve_invalid_options(tmp_path):
    for value in (True, "0.1", -0.1, 3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="POSITION_RESERVE_INVALID"):
            nearfield_descriptor(4.0, value)
    with pytest.raises(ValueError, match="COMPOSED_MONOTONICITY"):
        train_action_risk_expert(
            [],
            [],
            LocalPolicyTrainingConfig(),
            tmp_path / "candidate",
            geometry_distance_cap_m=4.0,
            position_uncertainty_floor_m=0.1,
            normalize_inputs=True,
            physical_context_only=True,
            monotonic_uncertainty=True,
        )
    assert not (tmp_path / "candidate").exists()


# 功能：验证余量下限来自原教师而不是为了放行临时缩小，覆盖采样接纳到训练入口。
# 输入：tmp_path：隔离目录；floor：配置下限。输出：错误组合拒绝，合法组合仍不授予资格。
@pytest.mark.parametrize("floor", [0.1, 0.2])
def test_position_reserve_binds_teacher_before_creating_output(tmp_path, floor):
    from dataclasses import replace

    from test_action_risk_training_boundaries import metric_dataset

    base = replace(metric_dataset(), receipt={"teacher_config": {"position_uncertainty_m": 0.1}})
    validation = replace(base, groups=frozenset({"d" * 64}), receipt_sha256="e" * 64)
    output = tmp_path / "candidate"
    if floor != 0.1:
        with pytest.raises(ValueError, match="POSITION_RESERVE_TEACHER_MISMATCH"):
            train_action_risk_expert(
                [base],
                [validation],
                LocalPolicyTrainingConfig(),
                output,
                normalize_inputs=True,
                physical_context_only=True,
                geometry_distance_cap_m=4.0,
                position_uncertainty_floor_m=floor,
            )
        assert not output.exists()
    else:
        receipt = train_action_risk_expert(
            [base],
            [validation],
            LocalPolicyTrainingConfig(),
            output,
            normalize_inputs=True,
            physical_context_only=True,
            geometry_distance_cap_m=4.0,
            position_uncertainty_floor_m=floor,
        )
        assert receipt["embedded_input_transform"] == nearfield_descriptor(4.0, 0.1)
        assert receipt["qualified_for_flight"] is False


# 功能：直接检查组合准入的教师绑定与复合单调声明，防止训练后人为修改元数据。
# 输入：无。输出：合法学习变换被识别，错误下限或未经验证的整体单调声明被拒绝。
def test_position_reserve_assembly_metadata():
    from types import SimpleNamespace

    from dronedream_agent_core.hashing import sha256_json
    from dronedream_agent_core.training.artifact_assembly import validate_expert_training_receipt
    from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT

    descriptor = nearfield_descriptor(4.0, 0.1)
    teacher = {"position_uncertainty_m": 0.1}
    manifest = SimpleNamespace(
        heading_context_for_role=lambda _: None,
        sensor_contract_sha256=descriptor["sensor_contract_sha256"],
        control_feature_contract_sha256="a" * 64,
    )
    receipt = dict(
        embedded_input_transform=descriptor,
        unwrapped_model_sha256="b" * 64,
        purpose="native-action-risk-offline-training",
        split_contract=SPATIAL_SPLIT_CONTRACT,
        training_groups=["f" * 64],
        validation_groups=["1" * 64],
        model_sha256="c" * 64,
        control_feature_contract_sha256="a" * 64,
        optimized=True,
        offline_validation_passed=True,
        qualified_for_flight=False,
        independent_physical_validation_required=True,
        teacher_config=teacher,
        teacher_config_sha256=sha256_json(teacher),
        dataset_receipts_sha256=dict(training=["d" * 64], validation=["e" * 64]),
    )
    validate_expert_training_receipt("risk-critic", "c" * 64, receipt, manifest)
    for changed in (
        dict(teacher_config={"position_uncertainty_m": 0.2}),
        dict(monotonic_localization_uncertainty=True),
    ):
        with pytest.raises(ValueError, match="POSITION_RESERVE_MISMATCH"):
            validate_expert_training_receipt(
                "risk-critic", "c" * 64, {**receipt, **changed}, manifest
            )
