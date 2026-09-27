from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from clock_fixtures import isolate_monotonic
from control_fixtures import complete_feature_snapshot, qualified_pilot_metrics
from pydantic import ValidationError

import dronedream_agent_core.local_policy_port as local_policy_port
from dronedream_agent_core.contracts import NormalizedPilotControl, TextNavigationDecision
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyPackageManifest,
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    latency_percentile,
    load_local_policy_package,
    local_policy_receipt_supports_control_contract,
    select_local_policy,
    select_local_policy_for_simulation,
)
from dronedream_agent_core.local_policy_port import (
    LocalPolicyFeatureBatch,
    LocalPolicyPort,
    LocalPolicyRawInference,
    OnnxLocalPolicyBackend,
    compile_local_policy_features,
    select_onnx_execution_providers,
)
from dronedream_agent_core.local_policy_training import (
    LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX,
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingSample,
    NumpyLocalPolicyModel,
    evaluate_local_policy,
    evaluate_local_risk_critic,
    expand_local_policy_realtime_control_inputs,
    expand_local_risk_critic_visual_inputs,
    risk_class_balanced_sample_weights,
    train_local_policy,
    train_local_risk_critic,
    write_local_policy_package,
)
from dronedream_agent_core.model_harness.model_port import ModelInvocationError
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.realtime_feature_encoders import (
    POLICY_REALTIME_FEATURE_COUNT,
    REALTIME_CONTROL_FEATURE_COUNT,
)
from dronedream_agent_core.temporal_evidence import TemporalEvidence


# 功能：
#   对测试自己生成的小文件计算摘要，为来源及输出断言提供内容身份。
# 输入：
#   path：测试临时文件。
# 输出：
#   digest：文件内容的 SHA-256。
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


MAP_SHA256 = "a" * 64
VEHICLE_SHA256 = "b" * 64
SENSOR_SHA256 = "c" * 64
SUITE_SHA256 = "d" * 64


# 功能：
#   验证推理提供程序优先选择可用加速器并保留 CPU 回退，不混入互斥加速方案。
# 输入：
#   无：显式构造 TensorRT/CUDA、DirectML 及仅 CPU 的可用集合。
# 输出：
#   None：不返回业务数据。
def test_onnx_provider_selection_prefers_one_accelerator_with_safe_fallbacks() -> None:
    assert select_onnx_execution_providers(
        {
            "CPUExecutionProvider",
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        }
    ) == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert select_onnx_execution_providers({"CPUExecutionProvider", "DmlExecutionProvider"}) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert select_onnx_execution_providers({"CPUExecutionProvider"}) == ["CPUExecutionProvider"]


# 功能：
#   明确指定的推理程序必须可用且非空，不能偷偷忽略无效配置。
# 输入：
#   无：可用 CPU/OpenVINO 及不可用 CUDA 的测试配置。
# 输出：
#   None：不返回业务数据。
def test_onnx_provider_selection_validates_operator_override() -> None:
    assert select_onnx_execution_providers(
        {"CPUExecutionProvider", "OpenVINOExecutionProvider"},
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    ) == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="PROVIDER_UNAVAILABLE"):
        select_onnx_execution_providers(
            {"CPUExecutionProvider"},
            ["CUDAExecutionProvider"],
        )
    with pytest.raises(RuntimeError, match="REQUEST_INVALID"):
        select_onnx_execution_providers(
            {"CPUExecutionProvider"},
            [],
        )


# 功能：
#   风险召回按风险监督分类，非运动动作不等于危险；缺失危险类时召回不能伪造为满分。
# 输入：
#   无：零权重网络、扫描及停留的两种风险标签。
# 输出：
#   None：不返回业务数据。
def test_risk_metrics_use_risk_targets_not_non_motion_actions() -> None:
    import numpy as np

    samples = []
    for target_action, risk_target in ((9, 0.4), (8, 0.8)):
        samples.append(
            LocalPolicyTrainingSample(
                state_features=[0.0] * 46,
                candidate_features=[[0.0] * 15 for _ in range(8)],
                candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                target_action_index=target_action,
                risk_target=risk_target,
            )
        )
    model = NumpyLocalPolicyModel(
        input_weight=np.zeros((174, 8), dtype=np.float32),
        input_bias=np.zeros(8, dtype=np.float32),
        action_weight=np.zeros((8, 11), dtype=np.float32),
        action_bias=np.zeros(11, dtype=np.float32),
        risk_weight=np.zeros((8, 1), dtype=np.float32),
        risk_bias=np.zeros(1, dtype=np.float32),
    )

    policy_metrics = evaluate_local_policy(model, samples)
    critic_metrics = evaluate_local_risk_critic(model, samples)

    assert policy_metrics.risk_hold_recall == 1.0
    assert policy_metrics.safe_motion_recall == 0.0
    assert critic_metrics.risk_hold_recall == 1.0
    assert critic_metrics.safe_motion_recall == 0.0
    all_safe = [sample.model_copy(update={"risk_target": 0.1}) for sample in samples]
    unsupported_policy = evaluate_local_policy(model, all_safe)
    unsupported_critic = evaluate_local_risk_critic(model, all_safe)
    assert unsupported_policy.risky_sample_count == 0
    assert unsupported_policy.risk_hold_recall == 0.0
    assert unsupported_critic.risky_sample_count == 0
    assert unsupported_critic.risk_hold_recall == 0.0


# 功能：
#   实时输入扩展保留原状态及候选权重，只增加新特征和四轴输出；这不授予新契约资格。
# 输入：
#   无：具有可追踪递增输入权重的合成网络。
# 输出：
#   None：不返回业务数据。
def test_realtime_control_expansion_preserves_existing_policy_rows() -> None:
    import numpy as np

    model = NumpyLocalPolicyModel(
        input_weight=np.arange(174 * 8, dtype=np.float32).reshape(174, 8),
        input_bias=np.zeros(8, dtype=np.float32),
        action_weight=np.ones((8, 11), dtype=np.float32),
        action_bias=np.zeros(11, dtype=np.float32),
        risk_weight=np.ones((8, 1), dtype=np.float32),
        risk_bias=np.zeros(1, dtype=np.float32),
    )

    expanded = expand_local_policy_realtime_control_inputs(
        model,
        realtime_feature_count=REALTIME_CONTROL_FEATURE_COUNT,
        include_pilot_control=True,
        random_seed=123,
    )

    assert expanded.input_weight.shape == (174 + 2 * REALTIME_CONTROL_FEATURE_COUNT, 8)
    assert np.array_equal(expanded.input_weight[:174], model.input_weight)
    assert expanded.realtime_feature_count == REALTIME_CONTROL_FEATURE_COUNT
    assert expanded.pilot_control_weight is not None
    assert expanded.pilot_control_weight.shape == (8, 4)


# 功能：
#   默认类别平衡将十八个安全样本和两个危险样本调整为相等的总权重贡献。
# 输入：
#   无：风险标签和等权初始样本。
# 输出：
#   None：不返回业务数据。
def test_risk_class_balance_equalizes_training_class_contribution() -> None:
    import numpy as np

    targets = np.asarray([0.0] * 18 + [1.0] * 2, dtype=np.float32)
    weights = risk_class_balanced_sample_weights(
        targets,
        np.ones(20, dtype=np.float32),
        LocalPolicyTrainingConfig(),
    )

    assert float(weights[targets < 0.5].sum()) == pytest.approx(10.0)
    assert float(weights[targets >= 0.5].sum()) == pytest.approx(10.0)


# 功能：
#   分类间隔达到 0.5 时必须在优化前失败，不能把目标推至非法范围。
# 输入：
#   无：单个合法风险样本及非法间隔。
# 输出：
#   None：不返回业务数据。
def test_risk_critic_rejects_invalid_classification_margin() -> None:
    sample = LocalPolicyTrainingSample(
        state_features=[0.0] * 46,
        candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_action_index=0,
        risk_target=0.4,
    )

    with pytest.raises(ValueError, match="classification margin"):
        train_local_risk_critic(
            [sample],
            LocalPolicyTrainingConfig(),
            classification_margin=0.5,
        )


# 功能：
#   写出绑定摘要的清单和占位模型，用于加载及选择规则测试；占位字节不是可执行 ONNX。
# 输入：
#   root：尚不存在的测试目录。
#   package_id：包标识，用于区分测试权重字节。
#   scope：通用或地图专用范围。
#   base_package_sha256：地图包的回退基座摘要。
#   map_sha256：地图包适用的地图摘要。
#   include_risk_critic：是否包含独立风险角色。
#   expert_roles：另外声明的导航专家。
#   continuous_control：是否声明实时特征及四轴控制输出。
# 输出：
#   package：实际加载并核对内容身份的测试包。
def _write_package(
    root: Path,
    *,
    package_id: str,
    scope: str = "general",
    base_package_sha256: str | None = None,
    map_sha256: str | None = None,
    include_risk_critic: bool = True,
    expert_roles: tuple[str, ...] = (),
    continuous_control: bool = False,
) -> object:
    root.mkdir()
    model = f"model:{package_id}".encode()
    artifact_path = root / "local-navigation.onnx"
    artifact_path.write_bytes(model)
    artifacts = [
        {
            "role": "local-navigation-policy",
            "relative_path": artifact_path.name,
            "sha256": hashlib.sha256(model).hexdigest(),
            "format": "onnx",
            "input_names": [
                "state_features",
                "candidate_features",
                "candidate_mask",
                *(["realtime_features", "realtime_valid_mask"] if continuous_control else []),
            ],
            "output_names": [
                "candidate_scores",
                "action_scores",
                "risk_score",
                *(["pilot_control"] if continuous_control else []),
            ],
        }
    ]
    if include_risk_critic:
        critic = f"critic:{package_id}".encode()
        critic_path = root / "risk-critic.onnx"
        critic_path.write_bytes(critic)
        artifacts.append(
            {
                "role": "risk-critic",
                "relative_path": critic_path.name,
                "sha256": hashlib.sha256(critic).hexdigest(),
                "format": "onnx",
                "input_names": [
                    "state_features",
                    *(
                        ["realtime_features", "realtime_valid_mask", "proposed_control"]
                        if continuous_control
                        else ["candidate_features", "candidate_mask"]
                    ),
                ],
                "output_names": ["risk_score"],
            }
        )
    for role in expert_roles:
        expert = f"expert:{role}:{package_id}".encode()
        expert_path = root / f"{role}.onnx"
        expert_path.write_bytes(expert)
        artifacts.append(
            {
                "role": role,
                "relative_path": expert_path.name,
                "sha256": hashlib.sha256(expert).hexdigest(),
                "format": "onnx",
                "input_names": [
                    "state_features",
                    "candidate_features",
                    "candidate_mask",
                    *(["realtime_features", "realtime_valid_mask"] if continuous_control else []),
                ],
                "output_names": [
                    "candidate_scores",
                    "action_scores",
                    "risk_score",
                    *(["pilot_control"] if continuous_control else []),
                ],
            }
        )
    manifest = {
        "package_id": package_id,
        "display_name": package_id,
        "scope": scope,
        "base_package_sha256": base_package_sha256,
        "map_sha256": map_sha256,
        "vehicle_sha256": VEHICLE_SHA256,
        "sensor_contract_sha256": SENSOR_SHA256,
        "state_feature_count": 46,
        "candidate_feature_count": 15,
        "maximum_candidates": 8,
        "maximum_inference_latency_ms": 500,
        "minimum_selection_margin": 0.05,
        "risk_hold_threshold": 0.5,
        "realtime_feature_count": POLICY_REALTIME_FEATURE_COUNT if continuous_control else None,
        "pilot_control_mode": ("normalized-body-velocity" if continuous_control else None),
        "control_feature_contract_sha256": (
            CURRENT_POLICY_FEATURE_CONTRACT_SHA256 if continuous_control else None
        ),
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    package = load_local_policy_package(root)
    return package


# 功能：
#   构造满足基本数值门槛的资格回执夹具，不声称这些固定指标来自飞行验收。
# 输入：
#   package：需要绑定的测试包。
#   suffix：生成回执标识的单个十六进制字符。
#   map_sha256：与包范围对应的地图摘要或空值。
# 输出：
#   receipt：基本资格回执，连续控制测试还须补足对应证据。
def _qualification(package: object, *, suffix: str, map_sha256: str | None) -> object:
    receipt = LocalPolicyQualificationReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id=f"policy-qualification-{suffix * 32}",
        policy_package_sha256=package.package_sha256,  # type: ignore[attr-defined]
        evaluation_suite_sha256=SUITE_SHA256,
        map_sha256=map_sha256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        trial_count=100,
        successful_trial_count=100,
        collision_count=0,
        deterministic_safety_intervention_count=2,
        p50_inference_latency_ms=5.0,
        p95_inference_latency_ms=8.0,
        p99_inference_latency_ms=10.0,
        qualified=True,
    )
    return receipt


# 功能：
#   构造仅供选择逻辑测试的仿真准入回执，不代表真实训练指标或正式飞行资格。
# 输入：
#   package：需要绑定的测试包。
#   suffix：回执标识用单个十六进制字符。
#   map_sha256：地图绑定或空值。
# 输出：
#   receipt：固定双类计数及指标的测试准入回执。
def _admission(package: object, *, suffix: str, map_sha256: str | None) -> object:
    receipt = LocalPolicySimulationAdmissionReceipt(
        control_feature_contract_sha256=package.manifest.control_feature_contract_sha256,
        receipt_id=f"policy-simulation-admission-{suffix * 32}",
        policy_package_sha256=package.package_sha256,  # type: ignore[attr-defined]
        dataset_receipt_sha256="e" * 64,
        evaluation_suite_sha256=SUITE_SHA256,
        map_sha256=map_sha256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        sample_count=100,
        motion_sample_count=50,
        non_motion_sample_count=50,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=0.98,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        p50_inference_latency_ms=1.0,
        p95_inference_latency_ms=2.0,
        p99_inference_latency_ms=3.0,
        admitted_to_simulation=True,
    )
    return receipt


# 功能：
#   恢复引导通道允许明确的低非运动样本门槛，但少于该固定下限仍必须拒绝。
# 输入：
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_recovery_bootstrap_admission_allows_only_the_bounded_collection_lane(
    tmp_path: Path,
) -> None:
    package = _write_package(
        tmp_path / "bootstrap-package",
        package_id="local.recovery-bootstrap-fixture",
    )

    receipt = LocalPolicySimulationAdmissionReceipt(
        receipt_id="policy-simulation-admission-" + "9" * 32,
        policy_package_sha256=package.package_sha256,  # type: ignore[attr-defined]
        dataset_receipt_sha256="d" * 64,
        evaluation_suite_sha256="e" * 64,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        sample_count=23,
        motion_sample_count=20,
        non_motion_sample_count=3,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=1.0,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.01,
        p50_inference_latency_ms=1.0,
        p95_inference_latency_ms=2.0,
        p99_inference_latency_ms=3.0,
        navigation_training_dataset_receipt_sha256="f" * 64,
        navigation_admission_split="validation",
        navigation_admission_source_evidence_sha256=["1" * 64],
        admission_scope="recovery-bootstrap-simulation",
        minimum_motion_sample_count=20,
        minimum_non_motion_sample_count=3,
        admitted_to_simulation=True,
    )

    assert receipt.admitted_to_simulation is True
    assert receipt.admission_scope == "recovery-bootstrap-simulation"

    with pytest.raises(ValidationError, match="hard gates"):
        LocalPolicySimulationAdmissionReceipt.model_validate(
            {
                **receipt.model_dump(mode="json"),
                "sample_count": 22,
                "motion_sample_count": 20,
                "non_motion_sample_count": 2,
            }
        )


# 功能：
#   标准仿真准入不能借用恢复引导的低样本门槛，范围和阈值必须一致。
# 输入：
#   tmp_path：标准测试包目录。
# 输出：
#   None：不返回业务数据。
def test_standard_admission_cannot_lower_its_sample_thresholds(tmp_path: Path) -> None:
    package = _write_package(
        tmp_path / "standard-package",
        package_id="local.standard-admission-fixture",
    )

    with pytest.raises(ValidationError, match="thresholds are immutable"):
        LocalPolicySimulationAdmissionReceipt(
            receipt_id="policy-simulation-admission-" + "8" * 32,
            policy_package_sha256=package.package_sha256,  # type: ignore[attr-defined]
            dataset_receipt_sha256="d" * 64,
            evaluation_suite_sha256="e" * 64,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
            sample_count=23,
            motion_sample_count=20,
            non_motion_sample_count=3,
            motion_authorization_accuracy=1.0,
            candidate_selection_accuracy=1.0,
            risk_hold_recall=1.0,
            safe_motion_recall=1.0,
            p50_inference_latency_ms=1.0,
            p95_inference_latency_ms=2.0,
            p99_inference_latency_ms=3.0,
            minimum_non_motion_sample_count=3,
            admitted_to_simulation=True,
        )


# 功能：
#   构造带来源摘要、八方向空间摘要及唯一已验证候选的测试观测，不采集真实传感器。
# 输入：
#   无：使用固定位置、速度、局部环境及载具包络。
# 输出：
#   payload：可供特征编译和候选绑定断言使用的完整观测。
def _snapshot() -> dict[str, object]:
    sectors = [
        {
            "sector": name,
            "observed_free_voxels": 4,
            "occupied_voxels": 1,
            "nearest_occupied_clearance_m": 3.0,
            "status": "observed-partial",
        }
        for name in (
            "front",
            "front-left",
            "left",
            "back-left",
            "back",
            "back-right",
            "right",
            "front-right",
        )
    ]
    payload: dict[str, object] = {
        "current_position_m": {"x": 1.0, "y": 2.0, "z": 3.0},
        "current_velocity_mps": {"x": 0.1, "y": 0.0, "z": 0.0},
        "goal_position_m": {"x": 5.0, "y": 2.0, "z": 3.0},
        "goal_distance_m": 4.0,
        "local_radius_m": 8.0,
        "vehicle_envelope_m": {
            "body_radius": 0.4,
            "body_height": 0.5,
            "candidate_speed": 0.5,
            "required_local_clearance": 0.5,
        },
        "perception_health": {
            "stream_healthy": True,
            "stream_age_seconds": 0.02,
            "localization_covariance_m2": 0.01,
        },
        "egocentric_sectors": sectors,
        "authorized_candidate_paths": [
            {
                "candidate_id": "candidate-0123456789abcdef",
                "kind": "goal-direct",
                "endpoint_m": {"x": 5.0, "y": 2.0, "z": 3.0},
                "path_length_m": 4.0,
                "minimum_predicted_dynamic_clearance_m": 5.0,
                "time_to_minimum_dynamic_clearance_seconds": 2.0,
                "required_clearance_m": 0.5,
                "deterministic_metric_path_validated": True,
                "dynamic_path_validated": True,
            }
        ],
    }
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload


# 功能：
#   导出读取历史及掩码但固定返回高异常的测试图，验证异常专家能否否决运动。
# 输入：
#   path：新 ONNX 测试文件路径。
# 输出：
#   None：不返回业务数据。
def _write_high_anomaly_detector(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Flatten", ["state_history"], ["flat_state"], axis=1),
        helper.make_node(
            "ReduceMean",
            ["flat_state"],
            ["state_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node(
            "ReduceMean",
            ["history_mask"],
            ["mask_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Mul", ["state_mean", "zero"], ["state_zero"]),
        helper.make_node("Mul", ["mask_mean", "zero"], ["mask_zero"]),
        helper.make_node("Add", ["state_zero", "mask_zero"], ["evidence_zero"]),
        helper.make_node("Add", ["evidence_zero", "high_logit"], ["anomaly_logit"]),
        helper.make_node("Sigmoid", ["anomaly_logit"], ["anomaly_score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test-state-anomaly-detector",
        [
            helper.make_tensor_value_info(
                "state_history",
                TensorProto.FLOAT,
                [None, 8, 46],
            ),
            helper.make_tensor_value_info(
                "history_mask",
                TensorProto.FLOAT,
                [None, 8],
            ),
        ],
        [helper.make_tensor_value_info("anomaly_score", TensorProto.FLOAT, [None, 1])],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
            numpy_helper.from_array(np.asarray([4.0], dtype=np.float32), "high_logit"),
        ],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test anomaly detector",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


# 功能：
#   导出固定低风险的感知健康图，隔离其他专家的效果；它不是训练好的感知模型。
# 输入：
#   path：测试 ONNX 输出路径。
# 输出：
#   None：不返回业务数据。
def _write_low_perception_health_critic(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node(
            "ReduceMean",
            ["state_features"],
            ["state_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Mul", ["state_mean", "zero"], ["evidence_zero"]),
        helper.make_node("Add", ["evidence_zero", "low_logit"], ["risk_logit"]),
        helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test-perception-health-critic",
        [helper.make_tensor_value_info("state_features", TensorProto.FLOAT, [None, 46])],
        [helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1])],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
            numpy_helper.from_array(np.asarray([-4.0], dtype=np.float32), "low_logit"),
        ],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test perception health critic",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


# 功能：
#   用固定低风险图验证跨模态专家的 64 维接口与调用记录，不评价真实跨模态准确率。
# 输入：
#   path：测试图输出路径。
# 输出：
#   None：不返回业务数据。
def _write_low_cross_modal_consistency_critic(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["sensor_features"],
                ["sensor_mean"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node("Mul", ["sensor_mean", "zero"], ["evidence_zero"]),
            helper.make_node("Add", ["evidence_zero", "low_logit"], ["risk_logit"]),
            helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
        ],
        "test-cross-modal-consistency-critic",
        [helper.make_tensor_value_info("sensor_features", TensorProto.FLOAT, [None, 64])],
        [helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1])],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
            numpy_helper.from_array(np.asarray([-4.0], dtype=np.float32), "low_logit"),
        ],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test cross-modal consistency critic",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


# 功能：
#   将 32×32 RGB 图像求平均为亮度特征，专用于证明视觉字节能改变下游输出的测试。
# 输入：
#   path：单特征测试编码器输出路径。
# 输出：
#   None：不返回业务数据。
def _write_brightness_perception_encoder(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["forward_rgb"],
                ["visual_features"],
                axes=[1, 2, 3],
                keepdims=0,
            )
        ],
        "test-brightness-perception-encoder",
        [helper.make_tensor_value_info("forward_rgb", TensorProto.FLOAT, [None, 3, 32, 32])],
        [helper.make_tensor_value_info("visual_features", TensorProto.FLOAT, [None])],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test visual encoder",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


# 功能：
#   导出固定低风险及 0.4 步长比例的载荷测试图，核对历史、载荷输入与控制缩放连接。
# 输入：
#   path：测试载荷适配器路径。
# 输出：
#   None：不返回业务数据。
def _write_payload_adapter(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Flatten", ["state_history"], ["flat_state"], axis=1),
        helper.make_node("ReduceMean", ["flat_state"], ["state_mean"], axes=[1], keepdims=1),
        helper.make_node(
            "ReduceMean",
            ["history_mask"],
            ["mask_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node(
            "ReduceMean",
            ["payload_features"],
            ["payload_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Add", ["state_mean", "mask_mean"], ["temporal_mean"]),
        helper.make_node("Add", ["temporal_mean", "payload_mean"], ["all_mean"]),
        helper.make_node("Mul", ["all_mean", "zero"], ["evidence_zero"]),
        helper.make_node("Add", ["evidence_zero", "low_logit"], ["risk_logit"]),
        helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
        helper.make_node("Add", ["evidence_zero", "step_scale"], ["controller_step_scale"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test-payload-dynamics-adapter",
        [
            helper.make_tensor_value_info("payload_features", TensorProto.FLOAT, [None, 24]),
            helper.make_tensor_value_info("state_history", TensorProto.FLOAT, [None, 8, 46]),
            helper.make_tensor_value_info("history_mask", TensorProto.FLOAT, [None, 8]),
        ],
        [
            helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("controller_step_scale", TensorProto.FLOAT, [None, 1]),
        ],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
            numpy_helper.from_array(np.asarray([-4.0], dtype=np.float32), "low_logit"),
            numpy_helper.from_array(np.asarray([0.4], dtype=np.float32), "step_scale"),
        ],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test payload dynamics adapter",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


# 功能：
#   导出固定低风险的稳定性测试图，检查机动特征与历史掩码输入的实际接线。
# 输入：
#   path：测试稳定性图路径。
# 输出：
#   None：不返回业务数据。
def _write_low_settle_stability_critic(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Flatten", ["state_history"], ["flat_state"], axis=1),
        helper.make_node("ReduceMean", ["flat_state"], ["state_mean"], axes=[1], keepdims=1),
        helper.make_node(
            "ReduceMean",
            ["history_mask"],
            ["mask_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node(
            "ReduceMean",
            ["maneuver_features"],
            ["maneuver_mean"],
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Add", ["state_mean", "mask_mean"], ["temporal_mean"]),
        helper.make_node("Add", ["temporal_mean", "maneuver_mean"], ["all_mean"]),
        helper.make_node("Mul", ["all_mean", "zero"], ["evidence_zero"]),
        helper.make_node("Add", ["evidence_zero", "low_logit"], ["risk_logit"]),
        helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test-settle-stability-critic",
        [
            helper.make_tensor_value_info("maneuver_features", TensorProto.FLOAT, [None, 13]),
            helper.make_tensor_value_info("state_history", TensorProto.FLOAT, [None, 8, 46]),
            helper.make_tensor_value_info("history_mask", TensorProto.FLOAT, [None, 8]),
        ],
        [helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1])],
        initializer=[
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "zero"),
            numpy_helper.from_array(np.asarray([-4.0], dtype=np.float32), "low_logit"),
        ],
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream test settle stability critic",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    onnx.save(document, path)


class _SelectingBackend:
    # 功能：
    #   初始化候选选择替身的最后调用角色，用于判断是否实际进入推理。
    # 输入：
    #   self：当前测试后端。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self.last_navigation_expert_role = None

    # 功能：
    #   固定偏好第一个已授权候选，并记录请求角色和风险贡献，用于测试端口的选择逻辑。
    # 输入：
    #   self：候选选择替身。
    #   batch：必须包含第一个有效候选的特征。
    #   multimodal：此替身要求为空的图像载荷。
    # 输出：
    #   result：带明确风险来源的固定推理结果。
    def infer(
        self,
        batch: LocalPolicyFeatureBatch,
        *,
        multimodal: list[dict[str, object]],
    ) -> LocalPolicyRawInference:
        assert not multimodal
        assert batch.candidate_mask[0] == 1.0
        self.last_navigation_expert_role = batch.navigation_expert_role
        result = LocalPolicyRawInference(
            candidate_scores=[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            action_scores=[0.0, -1.0, -2.0],
            risk_score=0.1,
            invoked_advisory_roles=["risk-critic"],
            advisory_risk_scores={"risk-critic": 0.1},
        )
        return result


class _PayloadSelectingBackend(_SelectingBackend):
    # 功能：
    #   在候选选择替身结果上增加载荷贡献及步长缩放，验证端口是否进一步施加采集上限。
    # 输入：
    #   self：载荷测试替身。
    #   batch：候选特征。
    #   multimodal：继承要求为空的视觉输入。
    # 输出：
    #   result：增加载荷来源及 0.59 步长比例的测试结果。
    def infer(
        self,
        batch: LocalPolicyFeatureBatch,
        *,
        multimodal: list[dict[str, object]],
    ) -> LocalPolicyRawInference:
        raw = super().infer(batch, multimodal=multimodal)
        result = raw.model_copy(
            update={
                "controller_step_scale": 0.59,
                "invoked_advisory_roles": [
                    "risk-critic",
                    "payload-dynamics-adapter",
                ],
                "advisory_risk_scores": {
                    "risk-critic": 0.1,
                    "payload-dynamics-adapter": 0.1,
                },
            }
        )
        return result


class _PilotControlBackend:
    # 功能：
    #   在实时特征已就绪时返回固定四轴指令，测试连续控制不依赖候选点选择。
    # 输入：
    #   self：连续控制测试替身。
    #   batch：已就绪的实时特征。
    #   multimodal：此替身不使用的视觉输入。
    # 输出：
    #   result：具有风险贡献及四轴操纵量的推理结果。
    def infer(
        self,
        batch: LocalPolicyFeatureBatch,
        *,
        multimodal: list[dict[str, object]],
    ) -> LocalPolicyRawInference:
        assert not multimodal
        assert batch.realtime_features_ready is True
        result = LocalPolicyRawInference(
            candidate_scores=[0.0] * 8,
            action_scores=[-2.0, -3.0, -4.0, 3.0],
            risk_score=0.1,
            invoked_advisory_roles=["risk-critic"],
            advisory_risk_scores={"risk-critic": 0.1},
            pilot_control=NormalizedPilotControl(
                forward_axis=0.7,
                right_axis=-0.2,
                up_axis=0.1,
                yaw_axis=0.25,
            ),
        )
        return result


# 功能：
#   验证普通通用包真实加载后保留范围、内容身份及导航和风险角色路径。
# 输入：
#   tmp_path：占位模型及清单目录。
# 输出：
#   None：不返回业务数据。
def test_loads_hash_bound_general_policy_package(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")

    assert package.manifest.scope == "general"  # type: ignore[attr-defined]
    assert len(package.package_sha256) == 64  # type: ignore[attr-defined]
    assert package.artifact_paths["local-navigation-policy"].is_file()  # type: ignore[attr-defined]
    assert package.artifact_paths["risk-critic"].is_file()  # type: ignore[attr-defined]


# 功能：
#   包身份只绑定实际发布字段，新 schema 的默认字段不能悄悄改变历史包摘要。
# 输入：
#   tmp_path：省略可选字段的测试清单目录。
# 输出：
#   None：不返回业务数据。
def test_package_identity_uses_published_manifest_without_schema_default_drift(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    published_manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    assert "realtime_feature_count" in published_manifest
    published_manifest.pop("realtime_feature_count")
    published_manifest.pop("pilot_control_mode")
    package.manifest_path.write_text(json.dumps(published_manifest), encoding="utf-8")  # type: ignore[attr-defined]

    reloaded = load_local_policy_package(package.root)  # type: ignore[attr-defined]
    expected_entries = [
        {
            "role": artifact.role,
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
        }
        for artifact in reloaded.manifest.artifacts
    ]
    expected_sha256 = sha256_json(
        {
            "manifest": published_manifest,
            "artifacts": sorted(expected_entries, key=lambda item: item["role"]),
        }
    )

    assert reloaded.package_sha256 == expected_sha256


# 功能：
#   模型字节在清单发布后变化时拒绝加载，不沿用先前已验证身份。
# 输入：
#   tmp_path：将被替换模型字节的测试包。
# 输出：
#   None：不返回业务数据。
def test_rejects_policy_artifact_hash_drift(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    package.artifact_paths["local-navigation-policy"].write_bytes(b"tampered")  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="hash mismatch"):
        load_local_policy_package(package.root)  # type: ignore[attr-defined]


# 功能：
#   地图、设备和资格均匹配时选择地图专家，并明确记录合格通用回退包。
# 输入：
#   tmp_path：通用与地图专家测试目录。
# 输出：
#   None：不返回业务数据。
def test_selects_exact_map_specialist_with_qualified_general_rollback(tmp_path: Path) -> None:
    general = _write_package(tmp_path / "general", package_id="local.general")
    specialist = _write_package(
        tmp_path / "specialist",
        package_id="local.school-map",
        scope="map-specialist",
        base_package_sha256=general.package_sha256,  # type: ignore[attr-defined]
        map_sha256=MAP_SHA256,
    )

    selection = select_local_policy(
        packages=[general, specialist],  # type: ignore[list-item]
        receipts=[
            _qualification(general, suffix="1", map_sha256=None),
            _qualification(specialist, suffix="2", map_sha256=MAP_SHA256),
        ],  # type: ignore[list-item]
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )

    assert selection.package_sha256 == specialist.package_sha256  # type: ignore[attr-defined]
    assert selection.rollback_package_sha256 == general.package_sha256  # type: ignore[attr-defined]


# 功能：
#   已有资格但适用另一地图的专家不能被选择，应使用匹配设备的通用包。
# 输入：
#   tmp_path：正确通用包和错误地图专家目录。
# 输出：
#   None：不返回业务数据。
def test_wrong_map_specialist_falls_back_to_general(tmp_path: Path) -> None:
    general = _write_package(tmp_path / "general", package_id="local.general")
    specialist = _write_package(
        tmp_path / "specialist",
        package_id="local.other-map",
        scope="map-specialist",
        base_package_sha256=general.package_sha256,  # type: ignore[attr-defined]
        map_sha256="e" * 64,
    )

    selection = select_local_policy(
        packages=[general, specialist],  # type: ignore[list-item]
        receipts=[
            _qualification(general, suffix="3", map_sha256=None),
            _qualification(specialist, suffix="4", map_sha256="e" * 64),
        ],  # type: ignore[list-item]
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )

    assert selection.scope == "general"
    assert selection.package_sha256 == general.package_sha256  # type: ignore[attr-defined]


# 功能：
#   仿真准入返回 simulation_only 标志，缺少正式资格时正式选择仍须失败。
# 输入：
#   tmp_path：只有仿真回执的通用测试包。
# 输出：
#   None：不返回业务数据。
def test_simulation_admission_never_looks_production_qualified(tmp_path: Path) -> None:
    general = _write_package(tmp_path / "general", package_id="local.general")

    selection = select_local_policy_for_simulation(
        packages=[general],  # type: ignore[list-item]
        admissions=[_admission(general, suffix="6", map_sha256=None)],  # type: ignore[list-item]
        qualification_receipts=[],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )

    assert selection.simulation_only is True
    assert selection.selection_reason == "general-simulation-admitted"
    with pytest.raises(ValueError, match="NO_QUALIFIED"):
        select_local_policy(
            packages=[general],  # type: ignore[list-item]
            receipts=[],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   仿真地图专家允许以仿真准入通用包作回退，但整条选择链仍仅限仿真。
# 输入：
#   tmp_path：两份仿真测试包目录。
# 输出：
#   None：不返回业务数据。
def test_simulation_specialist_accepts_simulation_admitted_general_lineage(
    tmp_path: Path,
) -> None:
    general = _write_package(tmp_path / "general", package_id="local.general")
    specialist = _write_package(
        tmp_path / "specialist",
        package_id="local.specialist",
        scope="map-specialist",
        base_package_sha256=general.package_sha256,  # type: ignore[attr-defined]
        map_sha256=MAP_SHA256,
    )

    selection = select_local_policy_for_simulation(
        packages=[general, specialist],  # type: ignore[list-item]
        admissions=[
            _admission(general, suffix="1", map_sha256=None),
            _admission(specialist, suffix="2", map_sha256=MAP_SHA256),
        ],  # type: ignore[list-item]
        qualification_receipts=[],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )

    assert selection.package_sha256 == specialist.package_sha256  # type: ignore[attr-defined]
    assert selection.rollback_package_sha256 == general.package_sha256  # type: ignore[attr-defined]
    assert selection.simulation_only is True


# 功能：
#   旧候选回执不能授权连续控制，必须补足实时输入、四轴误差及各导航专家质量证据。
# 输入：
#   tmp_path：声明连续控制的测试包目录。
# 输出：
#   None：不返回业务数据。
def test_continuous_control_package_requires_new_admission_and_qualification_evidence(
    tmp_path: Path,
) -> None:
    package = _write_package(
        tmp_path / "continuous",
        package_id="local.continuous",
        continuous_control=True,
    )
    old_admission = _admission(package, suffix="a", map_sha256=None)
    old_qualification = _qualification(package, suffix="b", map_sha256=None)

    with pytest.raises(ValueError, match="NO_SIMULATION_ADMITTED"):
        select_local_policy_for_simulation(
            packages=[package],  # type: ignore[list-item]
            admissions=[old_admission],  # type: ignore[list-item]
            qualification_receipts=[],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )
    with pytest.raises(ValueError, match="NO_QUALIFIED"):
        select_local_policy(
            packages=[package],  # type: ignore[list-item]
            receipts=[old_qualification],  # type: ignore[list-item]
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )

    expert_metrics = {"local-navigation-policy": qualified_pilot_metrics()}
    admission = old_admission.model_copy(  # type: ignore[attr-defined]
        update={
            "realtime_feature_ready_sample_count": 100,
            "pilot_control_target_sample_count": 100,
            "navigation_expert_metrics": expert_metrics,
            "pilot_control_mean_absolute_error": 0.05,
        }
    )
    qualification = old_qualification.model_copy(  # type: ignore[attr-defined]
        update={
            "local_policy_call_count": 200,
            "realtime_feature_ready_call_count": 200,
            "model_motion_decision_count": 160,
            "pilot_control_command_count": 160,
            "navigation_expert_metrics": expert_metrics,
            "pilot_control_mean_absolute_error": 0.05,
        }
    )

    simulation_selection = select_local_policy_for_simulation(
        packages=[package],  # type: ignore[list-item]
        admissions=[admission],
        qualification_receipts=[],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )
    production_selection = select_local_policy(
        packages=[package],  # type: ignore[list-item]
        receipts=[qualification],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )

    assert simulation_selection.simulation_only is True
    assert production_selection.simulation_only is False


# 功能：
#   候选时代导航包缺少独立风险专家时，不能仅凭仿真回执进入选择结果。
# 输入：
#   tmp_path：无风险专家的测试包目录。
# 输出：
#   None：不返回业务数据。
def test_navigation_only_package_is_not_selectable(tmp_path: Path) -> None:
    package = _write_package(
        tmp_path / "navigation-only",
        package_id="local.navigation-only",
        include_risk_critic=False,
    )

    with pytest.raises(ValueError, match="NO_SIMULATION_ADMITTED"):
        select_local_policy_for_simulation(
            packages=[package],  # type: ignore[list-item]
            admissions=[_admission(package, suffix="7", map_sha256=None)],  # type: ignore[list-item]
            qualification_receipts=[],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   只覆盖水平控制的旧特征宽度不能修改标签后冒充当前完整控制输入。
# 输入：
#   tmp_path：当前连续模型测试包目录。
#   old_width：已不兼容的历史特征宽度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("old_width", [100, 110])
def test_horizontal_only_weights_cannot_be_relabelled_as_current(tmp_path, old_width):
    package = _write_package(
        tmp_path / "continuous", package_id="local.current", continuous_control=True
    )
    data = package.manifest.model_dump(mode="json")
    data["realtime_feature_count"] = old_width
    with pytest.raises(ValidationError, match="feature width is incompatible"):
        LocalPolicyPackageManifest.model_validate(data)


# 功能：
#   连续控制的独立风险专家必须读取具体提议动作，不能使用只看候选的旧风险接口。
# 输入：
#   tmp_path：将风险输入改回旧接口的清单目录。
# 输出：
#   None：不返回业务数据。
def test_continuous_package_rejects_legacy_independent_risk_contract(tmp_path):
    package = _write_package(
        tmp_path / "continuous", package_id="local.current", continuous_control=True
    )
    data = package.manifest.model_dump(mode="json")
    critic = next(item for item in data["artifacts"] if item["role"] == "risk-critic")
    critic["input_names"] = [
        "state_features",
        "candidate_features",
        "candidate_mask",
        "realtime_features",
        "realtime_valid_mask",
    ]
    with pytest.raises(ValidationError, match="risk critic tensor contract"):
        LocalPolicyPackageManifest.model_validate(data)


# 功能：
#   连续控制开发包即使有良好四轴回执，缺少动作风险专家也不能取得正式资格。
# 输入：
#   tmp_path：明确不含独立风险图的测试包。
# 输出：
#   None：不返回业务数据。
def test_continuous_development_package_cannot_become_production_without_action_critic(tmp_path):
    package = _write_package(
        tmp_path / "without-risk",
        package_id="local.development",
        continuous_control=True,
        include_risk_critic=False,
    )
    qualification = _qualification(package, suffix="a", map_sha256=None).model_copy(
        update={
            "local_policy_call_count": 200,
            "realtime_feature_ready_call_count": 200,
            "model_motion_decision_count": 160,
            "pilot_control_command_count": 160,
            "navigation_expert_metrics": {"local-navigation-policy": qualified_pilot_metrics()},
            "pilot_control_mean_absolute_error": 0.05,
        }
    )
    with pytest.raises(ValueError, match="NO_QUALIFIED"):
        select_local_policy(
            packages=[package],
            receipts=[qualification],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   载荷角色不能直接复用导航图的张量声明，必须使用其专门的载荷与时序接口。
# 输入：
#   tmp_path：错误载荷接口测试包目录。
# 输出：
#   None：不返回业务数据。
def test_payload_adapter_requires_its_dedicated_tensor_contract(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="payload dynamics adapter tensor contract"):
        _write_package(
            tmp_path / "payload-not-wired",
            package_id="local.payload-not-wired",
            expert_roles=("payload-dynamics-adapter",),
        )


# 功能：
#   资格声明不能覆盖实际碰撞计数，即使其他成功率和延迟数值都达标。
# 输入：
#   无：构造仍有一次碰撞但声明合格的回执。
# 输出：
#   None：不返回业务数据。
def test_qualified_receipt_cannot_hide_a_collision() -> None:
    with pytest.raises(ValidationError, match="hard gates"):
        LocalPolicyQualificationReceipt(
            receipt_id="policy-qualification-" + "5" * 32,
            policy_package_sha256="f" * 64,
            evaluation_suite_sha256=SUITE_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
            trial_count=10,
            successful_trial_count=10,
            collision_count=1,
            deterministic_safety_intervention_count=0,
            p50_inference_latency_ms=1.0,
            p95_inference_latency_ms=2.0,
            p99_inference_latency_ms=3.0,
            qualified=True,
        )


# 功能：
#   核对状态、传感器、载荷及候选张量宽度，并检查测试后端选择映射为真实候选身份及轨迹。
# 输入：
#   tmp_path：候选接口测试包目录。
# 输出：
#   None：不返回业务数据。
def test_compiles_fixed_policy_features_and_selects_bound_candidate(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    snapshot = _snapshot()

    batch = compile_local_policy_features(snapshot)
    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert len(batch.state_features) == 46
    assert len(batch.sensor_features) == 64
    assert len(batch.payload_features) == 24
    assert batch.sensor_features[0] == 0.0
    assert len(batch.candidate_features) == 8
    assert all(len(candidate) == 15 for candidate in batch.candidate_features)
    assert result.artifact.action == "select-candidate"
    assert result.artifact.selected_candidate_id == "candidate-0123456789abcdef"
    assert result.record.provider == "local-policy"
    assert result.record.input_tokens == 0
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.requested_navigation_role == ("local-navigation-policy")
    assert result.record.local_expert_trace.navigation_action == "select-candidate"
    assert result.record.local_expert_trace.navigation_selected_candidate_id == (
        "candidate-0123456789abcdef"
    )


# 功能：
#   后端绕过校验生成非法风险、遗漏来源或重复角色时，端口必须重新检查完整推理结果。
# 输入：
#   tmp_path：测试包目录。
#   mutation：注入推理结果的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation",
    [
        {"risk_score": -1.0},
        {"risk_score": 0.01},
        {"advisory_risk_scores": {}},
        {"invoked_advisory_roles": ["risk-critic", "risk-critic"]},
    ],
)
def test_backend_cannot_return_mutated_or_unaccounted_expert_evidence(tmp_path, mutation):
    package = _write_package(tmp_path / "general", package_id="local.general")

    class InvalidBackend(_SelectingBackend):
        # 功能：
        #   绕过 Pydantic 赋值校验注入错误值，明确模拟不可信后端回传而非合法模型结果。
        # 输入：
        #   self：故障测试后端。
        #   batch：有效候选特征。
        #   multimodal：继承测试后端的视觉输入。
        # 输出：
        #   result：未经校验修改过的推理对象。
        def infer(self, batch, *, multimodal):
            # model_copy(update=...) skips Pydantic validation. The port must
            # validate the complete backend return before making any decision.
            result = super().infer(batch, multimodal=multimodal).model_copy(update=mutation)
            return result

    with pytest.raises(ModelInvocationError) as failure:
        LocalPolicyPort(package, backend=InvalidBackend()).call(
            role="local_navigation_advisor",
            output_type=TextNavigationDecision,
            instructions="bounded",
            input_artifact={"text_navigation_snapshot": _snapshot()},
        )
    assert failure.value.reason_code == "LOCAL_POLICY_INFERENCE_FAILED"


# 功能：
#   观测正文与摘要不一致时在推理之前拒绝，不能把伪造距离输入本地专家。
# 输入：
#   tmp_path：测试包目录。
# 输出：
#   None：不返回业务数据。
def test_port_rejects_tampered_snapshot_before_calling_backend(tmp_path):
    package = _write_package(tmp_path / "general", package_id="local.general")
    backend = _SelectingBackend()
    snapshot = _snapshot()
    snapshot["goal_distance_m"] = 999.0
    with pytest.raises(ValueError, match="SNAPSHOT_HASH_INVALID"):
        LocalPolicyPort(package, backend=backend).call(
            role="local_navigation_advisor",
            output_type=TextNavigationDecision,
            instructions="bounded",
            input_artifact={"text_navigation_snapshot": snapshot},
        )
    assert backend.last_navigation_expert_role is None


# 功能：
#   连续模式不编译候选点输入，速度按当前控制上限归一化而非沿用旧候选速度。
# 输入：
#   无：包含控制上限的合成观测。
# 输出：
#   None：不返回业务数据。
def test_continuous_policy_features_ignore_legacy_candidates_and_use_control_scale() -> None:
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "task": {
            "local_navigation_output_mode": "normalized-body-velocity",
            "normalized_pilot_control_limits": {
                "horizontal_speed_mps": 0.8,
            },
        }
    }

    batch = compile_local_policy_features(snapshot, include_candidate_features=False)

    assert batch.candidate_ids == ()
    assert batch.candidate_mask == (0.0,) * 8
    assert batch.state_features[0] == pytest.approx(0.125)


# 功能：
#   无候选点但实时特征完整时，端口可返回带四轴的直接控制并记录对应专家轨迹。
# 输入：
#   tmp_path：连续控制测试包。
# 输出：
#   None：不返回业务数据。
def test_continuous_policy_issues_direct_body_control_without_selecting_candidate(
    tmp_path: Path,
) -> None:
    package = _write_package(
        tmp_path / "continuous",
        package_id="local.continuous",
        continuous_control=True,
    )
    snapshot = _snapshot()
    snapshot["authorized_candidate_paths"] = []
    snapshot["realtime_feature_snapshot"] = complete_feature_snapshot().model_dump(mode="json")
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = sha256_json(snapshot)

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_PilotControlBackend(),
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded continuous control",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "pilot-control"
    assert result.artifact.selected_candidate_id is None
    assert result.artifact.pilot_control is not None
    assert result.artifact.pilot_control.forward_axis == pytest.approx(0.7)
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.navigation_action == "pilot-control"
    assert len(result.record.local_expert_trace.action_scores) == 4


# 功能：
#   实际耗时只超限 0.4 毫秒也必须拒绝，不能因整数记录精度吞掉超时。
# 输入：
#   tmp_path：具有 500 毫秒预算的测试包。
#   monkeypatch：提供确定时钟序列的工具。
# 输出：
#   None：不返回业务数据。
def test_port_rejects_submillisecond_latency_over_qualified_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    ticks = iter((100.0, 100.0, 100.5004))
    isolate_monotonic(monkeypatch, local_policy_port, lambda: next(ticks))

    with pytest.raises(ModelInvocationError) as failure:
        LocalPolicyPort(
            package,  # type: ignore[arg-type]
            backend=_SelectingBackend(),
        ).call(
            role="local_navigation_advisor",
            output_type=TextNavigationDecision,
            instructions="bounded",
            input_artifact={"text_navigation_snapshot": _snapshot()},
        )

    assert failure.value.reason_code == "LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED"
    assert failure.value.diagnostic_metrics["total-latency-ms"] == pytest.approx(500.4)
    assert failure.value.diagnostic_metrics["qualified-limit-ms"] == 500.0


# 功能：
#   仅明确配置的有界主机调度宽限可以容忍轻微超时，超过允许宽限本身的配置必须拒绝。
# 输入：
#   tmp_path：测试包目录。
#   monkeypatch：固定耗时序列的工具。
# 输出：
#   None：不返回业务数据。
def test_port_accepts_only_the_explicit_bounded_host_scheduling_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    ticks = iter((100.0, 100.0, 100.5204, 100.5204, 100.5204))
    isolate_monotonic(monkeypatch, local_policy_port, lambda: next(ticks))

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
        scheduling_jitter_grace_ms=50.0,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": _snapshot()},
    )

    assert result.record.latency_ms == 520
    with pytest.raises(ValueError, match="scheduling jitter grace"):
        LocalPolicyPort(
            package,  # type: ignore[arg-type]
            backend=_SelectingBackend(),
            scheduling_jitter_grace_ms=50.1,
        )


# 功能：
#   缺失或非法负载质量编码为未知，不伪造已经测量到零质量的证据。
# 输入：
#   mass：缺失、错误类型、越界或非有限的质量值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mass", [None, True, "0.1", -0.1, math.nan, math.inf, 10**400])
def test_invalid_payload_mass_is_unknown_not_measured_zero(mass) -> None:
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "observed_payload_mass_kg": mass,
            "maximum_payload_kg": 0.1,
        }
    }
    # Exercise the numeric encoder directly; a wire envelope rejects NaN earlier.
    features = compile_local_policy_features(snapshot).payload_features
    assert features[5:7] == (0.0, 0.0)


# 功能：
#   合法质量保留存在标记，负载比例有界且零质量与未知值明确区分。
# 输入：
#   mass：真实有效的质量数值。
#   ratio：依据测试载荷上限计算的截断比例。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mass,ratio", [(0, 0.0), (0.05, 0.5), (0.3, 2.0)])
def test_valid_payload_mass_retains_presence_and_bounded_ratio(mass, ratio) -> None:
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "observed_payload_mass_kg": mass,
            "maximum_payload_kg": 0.1,
        }
    }
    features = compile_local_policy_features(snapshot).payload_features
    assert features[5:7] == (1.0, ratio)


# 功能：
#   核对载荷状态、质量、加速度、角速度、姿态、电流电压及执行器信息的实际特征位置和比例。
# 输入：
#   无：带最新动力学信息的合成观测。
# 输出：
#   None：不返回业务数据。
def test_payload_features_include_fresh_px4_dynamics() -> None:
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "task": {"control_profile": "precision"},
        "payload": {
            "state": "loaded-stable",
            "observed_payload_mass_kg": 0.05,
            "maximum_payload_kg": 0.1,
            "within_declared_payload_limit": True,
            "accepted_transition_steps": ["attach", "confirm"],
            "dynamics": {
                "available": True,
                "ready": True,
                "acceleration_forward_m_s2": 0.980665,
                "acceleration_right_m_s2": -1.96133,
                "acceleration_down_m_s2": -9.80665,
                "angular_velocity_forward_rad_s": 0.5,
                "angular_velocity_right_rad_s": -1.0,
                "angular_velocity_down_rad_s": 1.5,
                "roll_deg": 4.5,
                "pitch_deg": -9.0,
                "current_battery_a": 5.0,
                "voltage_v": 15.0,
                "actuator_mean": 0.5,
                "actuator_max_abs": 0.8,
            },
        },
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    features = compile_local_policy_features(snapshot).payload_features

    assert len(features) == 24
    assert features[:10] == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.25, 1.0])
    assert features[10:] == pytest.approx(
        [1.0, 1.0, 0.1, -0.2, -1.0, 0.1, -0.2, 0.3, 0.1, -0.2, 0.25, 0.6, 0.5, 0.8]
    )


# 功能：
#   常规模式带载而缺少载荷适配器时必须停留，即使基础导航建议移动。
# 输入：
#   tmp_path：没有载荷专家的测试包目录。
# 输出：
#   None：不返回业务数据。
def test_port_holds_loaded_payload_when_dynamics_adapter_is_absent(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "loaded-stable",
            "dynamics": {"available": True, "ready": True},
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "hold"
    assert "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE" in result.artifact.risk_notes


# 功能：
#   明确开启开发采集且载荷合规、动力学新鲜时，后备动作仍被限制为 0.2 步长。
# 输入：
#   tmp_path：基础导航测试包目录。
# 输出：
#   None：不返回业务数据。
def test_development_payload_collection_uses_fresh_dynamics_with_bounded_step(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "loaded-stable",
            "within_declared_payload_limit": True,
            "dynamics": {"available": True, "ready": True},
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
        development_payload_collection=True,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "select-candidate"
    assert result.artifact.controller_step_scale == pytest.approx(0.2)
    assert "LOCAL_POLICY_CONTROLLER_STEP_RESTRICTED" in result.artifact.risk_notes
    assert "LOCAL_EXPERT_DEVELOPMENT_PAYLOAD_COLLECTION_FALLBACK" in result.artifact.risk_notes
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.controller_step_scale == pytest.approx(0.2)


# 功能：
#   开发采集不能越过动力学未就绪限制，过期信息必须保持停留并记录原因。
# 输入：
#   tmp_path：采集测试包目录。
# 输出：
#   None：不返回业务数据。
def test_development_payload_collection_still_holds_stale_dynamics(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "loaded-stable",
            "dynamics": {"available": True, "ready": False},
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
        development_payload_collection=True,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "hold"
    assert "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE" in result.artifact.risk_notes
    assert "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY" in result.artifact.risk_notes


# 功能：
#   开发采集上限必须继续约束已有载荷专家较大的动作比例，不被其 0.59 建议覆盖。
# 输入：
#   tmp_path：使用明确载荷推理替身的测试上下文。
# 输出：
#   None：不返回业务数据。
def test_development_payload_collection_caps_an_existing_payload_adapter(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    package.artifact_paths["payload-dynamics-adapter"] = tmp_path / "payload.onnx"  # type: ignore[index]
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "loaded-stable",
            "within_declared_payload_limit": True,
            "dynamics": {"available": True, "ready": True},
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_PayloadSelectingBackend(),
        development_payload_collection=True,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "select-candidate"
    assert result.artifact.controller_step_scale == pytest.approx(0.2)
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.controller_step_scale == pytest.approx(0.2)


# 功能：
#   空载去程不应受到带载采集的额外步长限制，控制轨迹应保留单位比例。
# 输入：
#   tmp_path：空载导航测试包目录。
# 输出：
#   None：不返回业务数据。
def test_development_payload_collection_does_not_slow_empty_outbound_flight(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "general", package_id="local.general")
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "payload": {
            "state": "empty",
            "dynamics": {"available": True, "ready": True},
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
        development_payload_collection=True,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.artifact.action == "select-candidate"
    assert result.artifact.controller_step_scale == pytest.approx(1.0)
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.controller_step_scale == pytest.approx(1.0)


# 功能：
#   核对 RGB、深度、雷达的存在掩码、年龄、质量、覆盖率及整体健康特征的固定位置。
# 输入：
#   无：三种模态均健康且带序号的合成传感器摘要。
# 输出：
#   None：不返回业务数据。
def test_compiles_mask_aware_multimodal_sensor_features() -> None:
    snapshot = _snapshot()
    snapshot["multimodal_sensor_snapshot"] = {
        "ready_for_motion": True,
        "issue_codes": [],
        "statuses": [
            {
                "sensor_id": "oakd-lite-forward-rgb",
                "modality": "rgb-camera",
                "required_for_motion": True,
                "latest_sequence": 4,
                "sample_age_seconds": 0.04,
                "quality": 0.8,
                "coverage": 1.0,
                "health": "healthy",
                "issue_codes": [],
            },
            {
                "sensor_id": "oakd-lite-depth",
                "modality": "depth-camera",
                "required_for_motion": True,
                "latest_sequence": 7,
                "sample_age_seconds": 0.02,
                "quality": 0.95,
                "coverage": 0.9,
                "health": "healthy",
                "issue_codes": [],
            },
            {
                "sensor_id": "mmwave-front",
                "modality": "radar",
                "required_for_motion": True,
                "latest_sequence": 11,
                "sample_age_seconds": 0.03,
                "quality": 0.9,
                "coverage": 0.75,
                "health": "healthy",
                "issue_codes": [],
            },
        ],
    }

    features = compile_local_policy_features(snapshot).sensor_features

    assert features[:5] == pytest.approx((1.0, 1.0, 0.02, 0.8, 1.0))
    assert features[5:10] == pytest.approx((1.0, 1.0, 0.01, 0.95, 0.9))
    assert features[10:15] == pytest.approx((1.0, 1.0, 0.015, 0.9, 0.75))
    assert features[60:] == pytest.approx((1.0, 0.1875, 0.0, 1.0))


# 功能：
#   精细动作阶段应请求包内精细专家，并在调用记录中标明所选角色且无回退。
# 输入：
#   tmp_path：包含精细导航角色的测试包。
# 输出：
#   None：不返回业务数据。
def test_port_routes_precision_phase_to_qualified_precision_expert(tmp_path: Path) -> None:
    package = _write_package(
        tmp_path / "expert-package",
        package_id="local.experts",
        expert_roles=("precision-maneuver-policy",),
    )
    snapshot = _snapshot()
    snapshot["strategic_context"] = {
        "task": {
            "phase": "ACTION",
            "control_profile": "precision",
            "decision_trigger": "periodic",
            "action_checkpoint_goal": True,
        }
    }
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    backend = _SelectingBackend()

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=backend,
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert backend.last_navigation_expert_role == "precision-maneuver-policy"
    assert result.record.model.endswith("/precision-maneuver-policy+risk-critic")
    assert result.artifact.risk_notes == []
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.selected_navigation_role == (
        "precision-maneuver-policy"
    )
    assert result.record.local_expert_trace.fallback_used is False


# 功能：
#   缺少精细专家时，候选时代通用回退必须记录请求角色、实际角色及回退原因。
# 输入：
#   tmp_path：仅通用导航与风险角色的测试包。
# 输出：
#   None：不返回业务数据。
def test_port_records_general_fallback_when_precision_expert_is_absent(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "general-only", package_id="local.general")
    snapshot = _snapshot()
    snapshot["strategic_context"] = {"task": {"phase": "LAND", "control_profile": "precision"}}
    snapshot["snapshot_sha256"] = sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    result = LocalPolicyPort(
        package,  # type: ignore[arg-type]
        backend=_SelectingBackend(),
    ).call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot},
    )

    assert result.record.model.endswith("/local-navigation-policy+risk-critic")
    assert "LOCAL_EXPERT_GENERAL_FALLBACK" in result.artifact.risk_notes
    assert result.record.local_expert_trace is not None
    assert result.record.local_expert_trace.requested_navigation_role == (
        "precision-maneuver-policy"
    )
    assert result.record.local_expert_trace.selected_navigation_role == ("local-navigation-policy")
    assert result.record.local_expert_trace.fallback_used is True


# 功能：
#   检查最近秩百分位在偶数样本及尾部位置的结果，避免不一致的插值口径。
# 输入：
#   无：顺序打乱的四个延迟样本。
# 输出：
#   None：不返回业务数据。
def test_latency_percentile_uses_nearest_rank() -> None:
    assert latency_percentile([4.0, 1.0, 3.0, 2.0], 50.0) == 2.0
    assert latency_percentile([4.0, 1.0, 3.0, 2.0], 99.0) == 4.0


# 功能：
#   通用包不能包含地图绑定，防止通用范围声明掩盖实际专用条件。
# 输入：
#   无：带错误地图绑定的通用清单。
# 输出：
#   None：不返回业务数据。
def test_general_manifest_rejects_map_binding() -> None:
    with pytest.raises(ValidationError, match="cannot bind"):
        LocalPolicyPackageManifest(
            package_id="local.invalid",
            display_name="Invalid",
            scope="general",
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
            maximum_inference_latency_ms=100,
            artifacts=[
                {
                    "role": "local-navigation-policy",
                    "relative_path": "policy.onnx",
                    "sha256": "f" * 64,
                    "input_names": [
                        "state_features",
                        "candidate_features",
                        "candidate_mask",
                    ],
                    "output_names": ["candidate_scores", "action_scores", "risk_score"],
                }
            ],
        )


# 功能：
#   读取实时特征及掩码的图必须声明兼容宽度，不能只增加输入名而省略契约。
# 输入：
#   无：相同输入张量在声明和未声明宽度时的两份清单。
# 输出：
#   None：不返回业务数据。
def test_manifest_binds_realtime_feature_inputs_to_declared_width() -> None:
    artifact = {
        "role": "local-navigation-policy",
        "relative_path": "policy.onnx",
        "sha256": "f" * 64,
        "input_names": [
            "state_features",
            "candidate_features",
            "candidate_mask",
            "realtime_features",
            "realtime_valid_mask",
        ],
        "output_names": ["candidate_scores", "action_scores", "risk_score"],
    }
    manifest = LocalPolicyPackageManifest(
        package_id="local.realtime",
        display_name="Realtime",
        scope="general",
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        maximum_inference_latency_ms=100,
        realtime_feature_count=REALTIME_CONTROL_FEATURE_COUNT,
        artifacts=[artifact],
    )

    assert manifest.realtime_feature_count == REALTIME_CONTROL_FEATURE_COUNT
    with pytest.raises(ValidationError, match="declared realtime feature count"):
        LocalPolicyPackageManifest(
            package_id="local.realtime-undeclared",
            display_name="Realtime undeclared",
            scope="general",
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
            maximum_inference_latency_ms=100,
            artifacts=[artifact],
        )


# 功能：
#   连续模型或回执缺少当前特征语义摘要时，即使指标良好也不能继承控制契约资格。
# 输入：
#   tmp_path：连续控制测试包目录。
# 输出：
#   None：不返回业务数据。
def test_unbound_continuous_package_cannot_inherit_otherwise_good_receipt(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "pilot", package_id="local.pilot", continuous_control=True)
    receipt = _admission(package, suffix="a", map_sha256=None).model_copy(
        update={
            "realtime_feature_ready_sample_count": 100,
            "pilot_control_target_sample_count": 100,
            "pilot_control_mean_absolute_error": 0.05,
            "navigation_expert_metrics": {"local-navigation-policy": qualified_pilot_metrics()},
        }
    )
    assert local_policy_receipt_supports_control_contract(package, receipt)
    assert not local_policy_receipt_supports_control_contract(
        package,
        receipt.model_copy(update={"control_feature_contract_sha256": None}),
    )
    for contract in (None, "f" * 64):
        manifest = package.manifest.model_copy(update={"control_feature_contract_sha256": contract})
        from dataclasses import replace

        assert not local_policy_receipt_supports_control_contract(
            replace(package, manifest=manifest),
            receipt,
        )


# 功能：
#   1. 用小型合成实时数据实际训练、导出并推理四轴网络，风险提案不得混入行为克隆。
#   2. 检查语义不匹配和实时信息缺失时失败，离线评价因方向覆盖不足不能准入。
# 输入：
#   tmp_path：小型测试权重、来源与评价回执目录。
#   request：登记后端清理，即使中途断言失败也关闭工作线程。
# 输出：
#   None：不返回业务数据。
def test_realtime_feature_policy_trains_exports_and_fails_closed_without_live_input(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    samples = []
    for index in range(48):
        positive = index % 2 == 0
        realtime = [0.0] * POLICY_REALTIME_FEATURE_COUNT
        realtime[0] = 1.0 if positive else -1.0
        samples.append(
            LocalPolicyTrainingSample(
                state_features=[0.0] * 46,
                candidate_features=[[0.0] * 15 for _ in range(8)],
                candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                realtime_features=realtime,
                control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
                realtime_valid_mask=[1.0] * POLICY_REALTIME_FEATURE_COUNT,
                target_pilot_control=([0.8, -0.25, 0.1, 0.4] if positive else [0.0, 0.0, 0.0, 0.0]),
                target_action_index=(LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX if positive else 8),
                risk_target=0.0 if positive else 1.0,
                pilot_control_limits=PilotControlLimits(1.0, 1.0, 10.0),
            )
        )
    config = LocalPolicyTrainingConfig(
        hidden_feature_count=12,
        epoch_count=60,
        batch_size=12,
        learning_rate=0.01,
    )
    model, metrics = train_local_policy(samples, config)
    # Risk probes are distinct supervision, never actions for BC to imitate.
    risk_samples = [
        sample.model_copy(
            update={
                "risk_proposed_control": [0.04 if index % 2 == 0 else 0.08, -0.0125, 0.005, 0.02]
            }
        )
        for index, sample in enumerate(samples)
    ]
    with pytest.raises(ValueError, match="ACTION_RISK_SUPERVISION"):
        train_local_policy(risk_samples, config)
    risk_model, _risk_metrics = train_local_risk_critic(risk_samples, config)
    package_root = tmp_path / "realtime-policy"
    unbound = [
        sample.model_copy(update={"control_feature_contract_sha256": None}) for sample in samples
    ]
    with pytest.raises(ValueError, match="feature semantics"):
        train_local_policy(unbound, config)
    write_local_policy_package(
        output_root=package_root,
        model=model,
        risk_model=risk_model,
        package_id="local.realtime-trained",
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        display_name="Realtime trained",
        scope="general",
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        maximum_inference_latency_ms=100,
    )
    package = load_local_policy_package(package_root)
    backend = OnnxLocalPolicyBackend(package)
    request.addfinalizer(backend.close)
    batch = LocalPolicyFeatureBatch(
        state_features=(0.0,) * 46,
        candidate_features=tuple((0.0,) * 15 for _ in range(8)),
        candidate_mask=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        candidate_ids=("candidate-1",),
        realtime_features=tuple(samples[0].realtime_features),
        control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        realtime_valid_mask=(1.0,) * POLICY_REALTIME_FEATURE_COUNT,
        realtime_features_ready=True,
        realtime_snapshot_sha256="a" * 64,
        pilot_control_limits=PilotControlLimits(1.0, 1.0, 10.0),
    )

    result = backend.infer(batch, multimodal=[])
    assert package.manifest.realtime_feature_count == POLICY_REALTIME_FEATURE_COUNT
    assert package.manifest.pilot_control_mode == "normalized-body-velocity"
    assert len(result.candidate_scores) == 8
    assert metrics.pilot_control_mean_absolute_error is not None
    assert metrics.pilot_control_mean_absolute_error < 0.15
    assert result.pilot_control is not None
    assert result.pilot_control.forward_axis > 0.5
    from dataclasses import replace

    for contract in (None, "f" * 64):
        with pytest.raises(ValueError, match="feature semantics"):
            backend.infer(replace(batch, control_feature_contract_sha256=contract), multimodal=[])
    with pytest.raises(RuntimeError, match="REALTIME_FEATURES_NOT_READY"):
        backend.infer(
            LocalPolicyFeatureBatch(
                state_features=batch.state_features,
                candidate_features=batch.candidate_features,
                candidate_mask=batch.candidate_mask,
                candidate_ids=batch.candidate_ids,
            ),
            multimodal=[],
        )
    backend.close()

    validation_path = tmp_path / "continuous-validation.jsonl"
    validation_path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples),
        encoding="utf-8",
    )
    dataset_receipt = tmp_path / "continuous-dataset-receipt.json"
    dataset_receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "verified_sources_required": True,
                "training_output_sha256": "a" * 64,
                "training_sources": [{"mission_evidence_sha256": "c" * 64}],
                "validation_sources": [{"mission_evidence_sha256": "d" * 64}],
                "validation_output_sha256": _file_sha256(validation_path),
            }
        ),
        encoding="utf-8",
    )
    admission_path = tmp_path / "continuous-simulation-admission.json"
    repository = Path(__file__).resolve().parents[1]
    evaluation = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "evaluate_local_policy_offline.py"),
            "--package",
            str(package_root),
            "--validation-data",
            str(validation_path),
            "--dataset-receipt",
            str(dataset_receipt),
            "--output",
            str(admission_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert evaluation.returncode == 1, evaluation.stderr
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["admitted_to_simulation"] is False
    assert "PILOT_AXIS_DIRECTION_COVERAGE_INSUFFICIENT_YAW" in admission["issue_codes"]
    assert admission["realtime_feature_ready_sample_count"] == len(samples)
    assert admission["pilot_control_target_sample_count"] == len(samples)
    assert admission["pilot_control_mean_absolute_error"] < 0.15


# 功能：
#   实际训练小型候选网络并连接各测试顾问，时序暖机后高异常专家须否决动作并完整记录贡献。
# 输入：
#   tmp_path：测试训练输出及固定顾问图目录；不运行真实飞行。
#   request：登记直接后端和端口的失败路径清理。
# 输出：
#   None：不返回业务数据。
def test_trains_exports_and_executes_compact_onnx_policy(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    samples = []
    for index in range(64):
        candidate_mask = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        state = [0.0] * 46
        candidates = [[0.0] * 15 for _ in range(8)]
        if index % 2 == 0:
            state[0] = 1.0
            candidates[0][3] = 0.2
            candidates[1][3] = 0.8
            target = 0
            risk = 0.0
        else:
            state[0] = -1.0
            candidates[0][4] = 0.01
            candidates[1][4] = 0.02
            target = 8
            risk = 1.0
        samples.append(
            LocalPolicyTrainingSample(
                state_features=state,
                candidate_features=candidates,
                candidate_mask=candidate_mask,
                target_action_index=target,
                risk_target=risk,
            )
        )
    model, metrics = train_local_policy(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=24,
            epoch_count=50,
            batch_size=16,
            learning_rate=0.01,
        ),
    )
    risk_model, risk_metrics = train_local_risk_critic(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=16,
            epoch_count=50,
            batch_size=16,
            learning_rate=0.01,
            random_seed=1814,
        ),
    )
    anomaly_path = tmp_path / "source-state-anomaly-detector.onnx"
    _write_high_anomaly_detector(anomaly_path)
    perception_health_path = tmp_path / "source-perception-health-critic.onnx"
    _write_low_perception_health_critic(perception_health_path)
    payload_adapter_path = tmp_path / "source-payload-dynamics-adapter.onnx"
    _write_payload_adapter(payload_adapter_path)
    settle_stability_path = tmp_path / "source-settle-stability-critic.onnx"
    _write_low_settle_stability_critic(settle_stability_path)
    cross_modal_path = tmp_path / "source-cross-modal-consistency-critic.onnx"
    _write_low_cross_modal_consistency_critic(cross_modal_path)
    package_root = tmp_path / "trained"
    write_local_policy_package(
        output_root=package_root,
        model=model,
        package_id="local.trained",
        display_name="Trained",
        scope="general",
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        maximum_inference_latency_ms=500,
        risk_model=risk_model,
        precision_model=model,
        recovery_model=model,
        state_anomaly_detector_path=anomaly_path,
        perception_health_critic_path=perception_health_path,
        payload_dynamics_adapter_path=payload_adapter_path,
        settle_stability_critic_path=settle_stability_path,
        cross_modal_consistency_critic_path=cross_modal_path,
    )
    package = load_local_policy_package(package_root)
    batch = LocalPolicyFeatureBatch(
        state_features=tuple(samples[0].state_features),
        candidate_features=tuple(tuple(item) for item in samples[0].candidate_features),
        candidate_mask=tuple(samples[0].candidate_mask),
        candidate_ids=("candidate-one", "candidate-two"),
        navigation_expert_role="precision-maneuver-policy",
    )

    direct_backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    request.addfinalizer(direct_backend.close)

    # 功能：
    #   为相同特征附加递增的独立观测身份，让时序测试真正按新样本推进历史。
    # 输入：
    #   sequence：测试观测序号。
    # 输出：
    #   observed：带新采样时间和摘要的特征批次。
    def observed_batch(sequence):
        observed = replace(
            batch,
            temporal_evidence=TemporalEvidence(
                stream_id="test-flight",
                sample_sha256=sha256_json({"sample": sequence}),
                observed_at_unix_ms=1000 + 50 * sequence,
            ),
        )
        return observed

    for sequence in range(7):
        warmup_inference = direct_backend.infer(observed_batch(sequence), multimodal=[])
        assert "state-anomaly-detector" not in warmup_inference.invoked_advisory_roles
    inference = direct_backend.infer(observed_batch(7), multimodal=[])
    runtime_snapshot = _snapshot()
    runtime_snapshot["strategic_context"] = {
        "task": {"phase": "HOVER", "control_profile": "precision"},
        "payload": {
            "state": "loaded-stable",
            "within_declared_payload_limit": True,
            "dynamics": {
                "available": True,
                "ready": True,
            },
        },
    }
    runtime_snapshot.pop("snapshot_sha256", None)
    runtime_snapshot["snapshot_sha256"] = sha256_json(runtime_snapshot)
    port = LocalPolicyPort(
        package,
        backend=OnnxLocalPolicyBackend(
            package,
            execution_providers=["CPUExecutionProvider"],
        ),
    )
    request.addfinalizer(port.close)

    # 功能：
    #   重建当前实时融合样本及外层观测摘要，避免只更新时间却沿用旧来源身份。
    # 输入：
    #   sequence：每 50 毫秒递增的测试序号。
    # 输出：
    #   current：具有一致内外来源的当前观测。
    def snapshot_at(sequence):
        current = dict(runtime_snapshot)
        rt = complete_feature_snapshot(1000 + sequence * 50).model_dump(mode="json")
        for encoding in rt["encodings"]:
            encoding["source_sha256"] = sha256_json({"sample": sequence})
        from dronedream_agent_core.realtime_feature_encoders import (
            RealtimeFeatureEncoding,
            fuse_realtime_features,
        )

        current["realtime_feature_snapshot"] = fuse_realtime_features(
            [RealtimeFeatureEncoding.model_validate(e) for e in rt["encodings"]],
            captured_at_unix_ms=1000 + sequence * 50,
        ).model_dump(mode="json")
        # A new physical sample changes both the inner tensor and outer input identity.
        current["snapshot_sha256"] = sha256_json(
            {key: value for key, value in current.items() if key != "snapshot_sha256"}
        )
        return current

    for sequence in range(7):
        warmup_result = port.call(
            role="local_navigation_advisor",
            output_type=TextNavigationDecision,
            instructions="bounded",
            input_artifact={"text_navigation_snapshot": snapshot_at(sequence)},
        )
        assert warmup_result.record.local_expert_trace is not None
        assert (
            "state-anomaly-detector"
            not in warmup_result.record.local_expert_trace.advisory_risk_scores
        )
    port_result = port.call(
        role="local_navigation_advisor",
        output_type=TextNavigationDecision,
        instructions="bounded",
        input_artifact={"text_navigation_snapshot": snapshot_at(7)},
    )

    assert metrics.action_accuracy >= 0.99
    assert metrics.risk_mean_absolute_error <= 0.05
    assert risk_metrics.risk_hold_recall == 1.0
    assert risk_metrics.safe_motion_recall == 1.0
    assert set(package.artifact_paths) == {
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
        "risk-critic",
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
    assert len(inference.candidate_scores) == 8
    assert inference.candidate_scores[0] > inference.action_scores[0]
    assert 0.0 <= inference.risk_score <= 1.0
    assert inference.invoked_advisory_roles == [
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
        "risk-critic",
    ]
    expected_latency_roles = {
        "precision-maneuver-policy",
        "risk-critic",
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
    assert set(inference.expert_latency_ms) == expected_latency_roles
    assert all(value >= 0.0 for value in inference.expert_latency_ms.values())
    assert inference.controller_step_scale == pytest.approx(0.4)
    assert port_result.record.model.endswith(
        "/precision-maneuver-policy+perception-health-critic+"
        "settle-stability-critic+payload-dynamics-adapter+state-anomaly-detector"
        "+cross-modal-consistency-critic+risk-critic"
    )
    assert port_result.artifact.action == "hold"
    assert "LOCAL_POLICY_RISK_THRESHOLD_REACHED" in port_result.artifact.risk_notes
    assert port_result.record.local_expert_trace is not None
    assert set(port_result.record.local_expert_trace.expert_latency_ms) == expected_latency_roles
    assert set(port_result.record.local_expert_trace.advisory_risk_scores) == {
        "risk-critic",
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
        "cross-modal-consistency-critic",
    }
    assert port_result.record.local_expert_trace.aggregate_risk_score >= max(
        port_result.record.local_expert_trace.advisory_risk_scores.values()
    )


# 功能：
#   实际编码黑白图片并改变导航及风险输出，检查文件、内存和预取路径一致；亮度图仅为接线测试。
# 输入：
#   tmp_path：合成图像及小型训练模型目录。
#   request：登记视觉预取线程的失败路径清理。
# 输出：
#   None：不返回业务数据。
def test_forward_rgb_encoder_materially_changes_navigation_vote(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    import numpy as np
    from PIL import Image

    samples = []
    for index in range(128):
        bright = index % 2 == 0
        samples.append(
            LocalPolicyTrainingSample(
                state_features=[0.0] * 46,
                candidate_features=[[0.0] * 15 for _ in range(8)],
                candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                visual_features=[1.0 if bright else 0.0],
                target_action_index=0 if bright else 8,
                risk_target=0.0 if bright else 1.0,
            )
        )
    model, metrics = train_local_policy(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=16,
            epoch_count=100,
            batch_size=16,
            learning_rate=0.01,
        ),
    )
    risk_model, _risk_metrics = train_local_risk_critic(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=8,
            epoch_count=40,
            batch_size=16,
            learning_rate=0.01,
        ),
    )
    risk_model = expand_local_risk_critic_visual_inputs(
        risk_model,
        visual_feature_count=1,
        random_seed=1814,
    )
    risk_model, _risk_metrics = train_local_risk_critic(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=8,
            epoch_count=40,
            batch_size=16,
            learning_rate=0.01,
        ),
        initial_model=risk_model,
    )
    encoder = tmp_path / "brightness-encoder.onnx"
    _write_brightness_perception_encoder(encoder)
    package_root = tmp_path / "visual-package"
    write_local_policy_package(
        output_root=package_root,
        model=model,
        package_id="local.visual",
        display_name="Visual policy",
        scope="general",
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        maximum_inference_latency_ms=500,
        risk_model=risk_model,
        perception_encoder_path=encoder,
        visual_width=32,
        visual_height=32,
        visual_feature_count=1,
        visual_normalization="zero-to-one",
    )
    package = load_local_policy_package(package_root)
    backend = OnnxLocalPolicyBackend(
        package,
        execution_providers=["CPUExecutionProvider"],
    )
    request.addfinalizer(backend.close)
    batch = LocalPolicyFeatureBatch(
        state_features=tuple(samples[0].state_features),
        candidate_features=tuple(tuple(row) for row in samples[0].candidate_features),
        candidate_mask=tuple(samples[0].candidate_mask),
        candidate_ids=("candidate",),
    )
    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(black_path)
    Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)).save(white_path)

    black = backend.infer(
        batch,
        multimodal=[{"kind": "image-file", "path": str(black_path)}],
    )
    white = backend.infer(
        batch,
        multimodal=[{"kind": "image-file", "path": str(white_path)}],
    )
    white_bytes = white_path.read_bytes()
    white_from_memory = backend.infer(
        batch,
        multimodal=[
            {
                "kind": "image-file",
                "path": str(white_path),
                "content_bytes": white_bytes,
                "content_sha256": hashlib.sha256(white_bytes).hexdigest(),
            }
        ],
    )
    white_rgb8 = bytes([255]) * (32 * 32 * 3)
    online_media = [
        {
            "kind": "image-file",
            "path": str(white_path),
            "content_bytes": white_bytes,
            "content_sha256": hashlib.sha256(white_bytes).hexdigest(),
            "model_rgb_bytes": white_rgb8,
            "model_rgb_sha256": hashlib.sha256(white_rgb8).hexdigest(),
            "model_rgb_width": 32,
            "model_rgb_height": 32,
        }
    ]
    backend.prime_visual(online_media)
    white_prefetched = backend.infer(batch, multimodal=online_media)

    assert metrics.action_accuracy >= 0.99
    assert int(np.argmax([*black.candidate_scores, *black.action_scores])) == 8
    assert int(np.argmax([*white.candidate_scores, *white.action_scores])) == 0
    assert np.allclose(white_from_memory.candidate_scores, white.candidate_scores)
    assert np.allclose(white_from_memory.action_scores, white.action_scores)
    assert white_from_memory.risk_score == pytest.approx(white.risk_score)
    assert np.allclose(white_prefetched.candidate_scores, white.candidate_scores)
    assert np.allclose(white_prefetched.action_scores, white.action_scores)
    assert black.advisory_risk_scores["risk-critic"] > 0.9
    assert white.advisory_risk_scores["risk-critic"] < 0.1
    assert "visual-input-preprocess" in white_prefetched.pipeline_latency_ms
    assert "visual-prefetch-wait" in white_prefetched.pipeline_latency_ms
    assert "backend-wall" in white_prefetched.pipeline_latency_ms
    assert "perception-encoder" in package.artifact_paths
    risk_artifact = next(
        artifact for artifact in package.manifest.artifacts if artifact.role == "risk-critic"
    )
    assert "visual_features" in risk_artifact.input_names

    with pytest.raises(RuntimeError, match="FORWARD_RGB_BYTES_HASH_MISMATCH"):
        backend.infer(
            batch,
            multimodal=[
                {
                    "kind": "image-file",
                    "path": str(white_path),
                    "content_bytes": white_bytes,
                    "content_sha256": "0" * 64,
                }
            ],
        )
    backend.close()


# 功能：
#   预编码视觉仅允许显式离线评价，常规推理拒绝；评价回执须绑定编码器和前处理延迟。
# 输入：
#   tmp_path：小型模型、预编码样本与回执目录。
#   request：登记生产和离线测试后端的清理。
# 输出：
#   None：不返回业务数据。
def test_precomputed_visual_features_are_offline_only(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    samples = [
        LocalPolicyTrainingSample(
            state_features=[0.0] * 46,
            candidate_features=[[0.0] * 15 for _ in range(8)],
            candidate_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            visual_features=[1.0],
            target_action_index=0,
            risk_target=0.0,
        )
        for _ in range(32)
    ]
    model, _metrics = train_local_policy(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=8,
            epoch_count=20,
            batch_size=8,
            learning_rate=0.01,
        ),
    )
    risk_model, _risk_metrics = train_local_risk_critic(
        samples,
        LocalPolicyTrainingConfig(
            hidden_feature_count=8,
            epoch_count=20,
            batch_size=8,
            learning_rate=0.01,
        ),
    )
    encoder = tmp_path / "encoder.onnx"
    _write_brightness_perception_encoder(encoder)
    package_root = tmp_path / "visual-package"
    write_local_policy_package(
        output_root=package_root,
        model=model,
        package_id="local.visual.precomputed",
        display_name="Visual policy",
        scope="general",
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        maximum_inference_latency_ms=500,
        risk_model=risk_model,
        perception_encoder_path=encoder,
        visual_width=32,
        visual_height=32,
        visual_feature_count=1,
        visual_normalization="zero-to-one",
    )
    package = load_local_policy_package(package_root)
    batch = LocalPolicyFeatureBatch(
        state_features=tuple(samples[0].state_features),
        candidate_features=tuple(tuple(row) for row in samples[0].candidate_features),
        candidate_mask=tuple(samples[0].candidate_mask),
        candidate_ids=("candidate",),
        precomputed_visual_features=(1.0,),
    )

    production = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    request.addfinalizer(production.close)
    with pytest.raises(RuntimeError, match="LOCAL_POLICY_PRECOMPUTED_VISUAL_DISABLED"):
        production.infer(batch, multimodal=[])

    offline = OnnxLocalPolicyBackend(
        package,
        execution_providers=["CPUExecutionProvider"],
        allow_precomputed_visual_features=True,
    )
    request.addfinalizer(offline.close)
    result = offline.infer(batch, multimodal=[])
    assert all(math.isfinite(value) for value in result.candidate_scores)

    source_validation = tmp_path / "validation-source.jsonl"
    source_validation.write_text(
        "".join(
            sample.model_copy(update={"visual_features": []}).model_dump_json() + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    encoded_validation = tmp_path / "validation-visual.jsonl"
    encoded_validation.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples),
        encoding="utf-8",
    )
    dataset_receipt = tmp_path / "dataset-receipt.json"
    dataset_receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-dataset-receipt.v1",
                "split_method": "held-out-source-campaign",
                "verified_sources_required": True,
                "training_output_sha256": "a" * 64,
                "training_sources": [{"mission_evidence_sha256": "c" * 64}],
                "validation_sources": [{"mission_evidence_sha256": "d" * 64}],
                "validation_output_sha256": _file_sha256(source_validation),
            }
        ),
        encoding="utf-8",
    )
    visual_receipt = tmp_path / "visual-receipt.json"
    visual_receipt.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-visual-encoding-receipt.v1",
                "qualification_granted": False,
                "source_policy_data_sha256": _file_sha256(source_validation),
                "output_sha256": _file_sha256(encoded_validation),
                "perception_encoder_sha256": _file_sha256(
                    package.artifact_paths["perception-encoder"]
                ),
                "sample_count": len(samples),
                "visual_width": 32,
                "visual_height": 32,
                "visual_feature_count": 1,
                "visual_normalization": "zero-to-one",
                "p99_preprocess_latency_ms": 1.5,
                "p99_encoder_latency_ms": 2.5,
            }
        ),
        encoding="utf-8",
    )
    admission_path = tmp_path / "simulation-admission.json"
    repository = Path(__file__).resolve().parents[1]
    evaluation = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "evaluate_local_policy_offline.py"),
            "--package",
            str(package_root),
            "--validation-data",
            str(encoded_validation),
            "--dataset-receipt",
            str(dataset_receipt),
            "--visual-encoding-receipt",
            str(visual_receipt),
            "--output",
            str(admission_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert evaluation.returncode == 1, evaluation.stderr
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["visual_encoding_receipt_sha256"] == _file_sha256(visual_receipt)
    assert admission["perception_encoder_sha256"] == _file_sha256(
        package.artifact_paths["perception-encoder"]
    )
    assert admission["visual_preprocess_p99_latency_ms"] == 1.5
    assert admission["visual_encoder_p99_inference_latency_ms"] == 2.5
    assert admission["combined_p99_inference_latency_ms"] >= (
        admission["p99_inference_latency_ms"] + 4.0
    )
