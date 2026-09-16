#!/usr/bin/env python3
"""Train a feed-forward development baseline, not the recurrent causal GRU pilot.

The current temporal controller uses train_causal_control_role/ensemble instead.
This baseline still serves explicit legacy/data-comparison callers and does not
automatically grant runtime or flight authority to its exported package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dronedream_agent_core.contracts import CalibratedRangeSensorMount, VehicleAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import NavigationExpertRole
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
    local_policy_receipt_supports_control_contract,
)
from dronedream_agent_core.local_policy_quality import navigation_quality_issues
from dronedream_agent_core.local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingMetrics,
    LocalPolicyTrainingSample,
    NumpyLocalPolicyModel,
    copy_numpy_policy_model,
    evaluate_local_policy,
    evaluate_local_risk_critic,
    expand_local_policy_realtime_control_inputs,
    load_local_policy_onnx,
    load_local_risk_critic_onnx,
    parse_training_samples,
    require_behavior_supervision,
    train_local_policy,
    train_local_risk_critic,
    write_local_policy_package,
)
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.evidence_files import read_evidence_object
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：
#   流式计算有界普通文件摘要，检查读取期间文件身份和内容变化。
# 输入：
#   path：至多 256 MiB 的模型或来源文件。
# 输出：
#   digest：实际读取原始字节的 SHA-256 摘要。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=256 * 1024 * 1024)
    return digest


# 功能：
#   按明确的专家类别选出监督样本，样本不足时阻止开始该专家训练。
# 输入：
#   samples：当前训练或留出数据，不合并两种分区。
#   role：要训练的操纵专家类别。
#   minimum_count：该分区所需的最少样本数。
# 输出：
#   selected：属于指定专家的样本列表。
def _expert_samples(
    samples: list[LocalPolicyTrainingSample],
    role: NavigationExpertRole,
    *,
    minimum_count: int,
) -> list[LocalPolicyTrainingSample]:
    selected = [
        sample for sample in samples if getattr(sample, "navigation_expert_role", None) == role
    ]
    if len(selected) < minimum_count:
        raise RuntimeError(f"LOCAL_EXPERT_{role.upper().replace('-', '_')}_DATASET_TOO_SMALL")
    return selected


# 功能：
#   复制全部参数数组及输入语义标识，专家精调不能原地改写通用策略的权重。
# 输入：
#   model：已训练的前馈通用策略。
# 输出：
#   cloned：不共享参数数组的策略副本。
def _clone_policy_model(model: NumpyLocalPolicyModel) -> NumpyLocalPolicyModel:
    cloned = copy_numpy_policy_model(model)
    return cloned


# 功能：
#   应用统一的操纵质量及类别支持检查，缺少支持的指标不能被当成成功。
# 输入：
#   metrics：真实训练或评价返回的统计。
#   prefix：用于定位当前专家和分区的错误前缀。
#   maximum_pilot_control_mae：允许的四轴平均绝对误差。
#   continuous_control：是否按连续四轴控制而非候选选择评价。
# 输出：
#   None：不返回业务数据。
def _validate_navigation_metrics(
    metrics: LocalPolicyTrainingMetrics,
    *,
    prefix: str,
    maximum_pilot_control_mae: float,
    continuous_control: bool,
) -> None:
    issues = navigation_quality_issues(
        metrics,
        continuous_control=continuous_control,
        maximum_pilot_control_mae=maximum_pilot_control_mae,
    )
    if issues:
        raise RuntimeError("; ".join(f"{prefix}_{issue}" for issue in issues))


# 功能：
#   解析前馈基线参数并检查通用／地图专用来源边界，再执行离线训练。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：候选训练及回执发布成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--scope", choices=("general", "map-specialist"), required=True)
    parser.add_argument("--vehicle-metadata", type=Path, required=True)
    parser.add_argument("--sensor-contract", type=Path, required=True)
    parser.add_argument("--map-semantic", type=Path)
    parser.add_argument("--base-package", type=Path)
    parser.add_argument("--base-qualification", type=Path)
    parser.add_argument(
        "--base-simulation-admission",
        type=Path,
        help=(
            "Simulation-only evidence for an admitted general package used as "
            "the warm start. This does not grant the trained package runtime authority."
        ),
    )
    parser.add_argument("--hidden-feature-count", type=int, default=64)
    parser.add_argument("--risk-hidden-feature-count", type=int, default=48)
    parser.add_argument(
        "--precision-hidden-feature-count",
        type=int,
        default=96,
        help="Fresh expert width; warm starts retain their existing backbone width.",
    )
    parser.add_argument(
        "--recovery-hidden-feature-count",
        type=int,
        default=96,
        help="Fresh expert width; warm starts retain their existing backbone width.",
    )
    parser.add_argument("--epoch-count", type=int, default=120)
    parser.add_argument("--risk-epoch-count", type=int, default=120)
    parser.add_argument(
        "--expert-epoch-count",
        type=int,
        help="Optional specialist fine-tuning epoch count; defaults to --epoch-count.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--expert-learning-rate",
        type=float,
        help="Optional specialist fine-tuning rate; defaults to --learning-rate.",
    )
    parser.add_argument("--risk-loss-weight", type=float, default=0.5)
    parser.add_argument("--pilot-control-loss-weight", type=float, default=1.0)
    parser.add_argument("--maximum-pilot-control-mae", type=float, default=0.2)
    parser.add_argument("--action-class-balance-power", type=float, default=1.0)
    parser.add_argument("--action-class-weight-cap", type=float, default=16.0)
    parser.add_argument("--risk-class-balance-power", type=float, default=1.0)
    parser.add_argument("--risk-class-weight-cap", type=float, default=16.0)
    parser.add_argument("--random-seed", type=int, default=805)
    parser.add_argument("--risk-random-seed", type=int, default=1814)
    parser.add_argument("--train-precision-expert", action="store_true")
    parser.add_argument("--train-recovery-expert", action="store_true")
    parser.add_argument("--state-anomaly-detector", type=Path)
    parser.add_argument("--perception-health-critic", type=Path)
    parser.add_argument("--settle-stability-critic", type=Path)
    parser.add_argument("--payload-dynamics-adapter", type=Path)
    parser.add_argument("--cross-modal-consistency-critic", type=Path)
    parser.add_argument("--perception-encoder", type=Path)
    parser.add_argument("--visual-width", type=int)
    parser.add_argument("--visual-height", type=int)
    parser.add_argument("--visual-feature-count", type=int)
    parser.add_argument(
        "--visual-normalization",
        choices=("zero-to-one", "minus-one-to-one", "imagenet"),
        default="zero-to-one",
    )
    parser.add_argument("--maximum-inference-latency-ms", type=int, default=100)
    args = parser.parse_args()
    args.training_receipt, args.output_package = (
        args.training_receipt.absolute(),
        args.output_package.absolute(),
    )
    check_plain_plugin_path(args.training_receipt)
    check_plain_plugin_path(args.output_package)
    if args.training_receipt.exists():
        raise FileExistsError(args.training_receipt)
    if args.output_package.exists():
        raise FileExistsError(args.output_package)
    base_evidence_count = sum(
        item is not None for item in (args.base_qualification, args.base_simulation_admission)
    )
    if args.scope == "map-specialist" and (
        args.map_semantic is None or args.base_package is None or base_evidence_count != 1
    ):
        parser.error(
            "map-specialist training requires a map, base package, and exactly "
            "one base qualification or simulation admission"
        )
    if args.scope == "general" and (
        args.map_semantic is not None
        or args.base_package is not None
        or args.base_qualification is not None
        or args.base_simulation_admission is not None
    ):
        parser.error("general training cannot bind map-specific base evidence")
    exit_code = train_baseline(args, parser)
    return exit_code


# 功能：
#   1. 固定训练来源并预验整批配置，训练前馈基线及显式启用的专家。
#   2. 发布候选包和独占回执，记录实际消费的来源，不把基线权重冒充因果 GRU。
# 输入：
#   args：已通过路径和作用域检查的命令行参数。
#   parser：用于报告跨参数约束的命令行解析器。
# 输出：
#   exit_code：全部候选产物及回执成功时为零。
def train_baseline(args, parser) -> int:
    vehicle_payload, vehicle_source_sha = read_evidence_object(
        args.vehicle_metadata, limit=4 * 1024 * 1024
    )
    sensor_payload, sensor_source_sha = read_evidence_object(
        args.sensor_contract, limit=4 * 1024 * 1024
    )
    vehicle = VehicleAsset.model_validate(vehicle_payload)
    sensor = CalibratedRangeSensorMount.model_validate(sensor_payload)
    training_content = read_plugin_file(args.training_data, limit=256 * 1024 * 1024)
    validation_content = read_plugin_file(args.validation_data, limit=256 * 1024 * 1024)
    training_sha = hashlib.sha256(training_content).hexdigest()
    validation_sha = hashlib.sha256(validation_content).hexdigest()
    training = parse_training_samples(training_content)
    validation = parse_training_samples(validation_content)
    del training_content, validation_content
    require_behavior_supervision(training)
    require_behavior_supervision(validation)
    if training_sha == validation_sha:
        raise ValueError("baseline training and validation cannot use identical source content")
    map_sha = _sha256(args.map_semantic) if args.map_semantic is not None else None
    config, risk_config, expert_configs = training_configurations(args)
    visual_arguments = (
        args.perception_encoder,
        args.visual_width,
        args.visual_height,
        args.visual_feature_count,
    )
    if any(value is not None for value in visual_arguments) and any(
        value is None for value in visual_arguments
    ):
        parser.error("visual policy training requires encoder, dimensions, and feature count")
    observed_visual_counts = {len(sample.visual_features) for sample in [*training, *validation]}
    if len(observed_visual_counts) != 1:
        raise ValueError("visual feature counts differ across policy datasets")
    observed_visual_count = next(iter(observed_visual_counts))
    if observed_visual_count != (args.visual_feature_count or 0):
        raise ValueError("policy dataset does not match the declared visual feature count")
    observed_realtime_counts = {
        len(sample.realtime_features) for sample in [*training, *validation]
    }
    observed_pilot_target_counts = {
        len(sample.target_pilot_control) for sample in [*training, *validation]
    }
    if len(observed_realtime_counts) != 1 or len(observed_pilot_target_counts) != 1:
        raise ValueError("realtime policy contracts differ across policy datasets")
    observed_realtime_count = next(iter(observed_realtime_counts))
    observed_pilot_target_count = next(iter(observed_pilot_target_counts))
    feature_contracts = {
        sample.control_feature_contract_sha256 for sample in [*training, *validation]
    }
    if observed_pilot_target_count and (len(feature_contracts) != 1 or None in feature_contracts):
        raise ValueError("continuous control datasets require one explicit feature contract")
    feature_contract = next(iter(feature_contracts)) if len(feature_contracts) == 1 else None
    if observed_pilot_target_count and not observed_realtime_count:
        raise ValueError("pilot-control training requires realtime policy features")
    if not 0.0 <= args.maximum_pilot_control_mae <= 2.0:
        parser.error("maximum pilot-control MAE must be between 0 and 2")
    expert_datasets = {
        role: (
            _expert_samples(training, role, minimum_count=10),
            _expert_samples(validation, role, minimum_count=5),
        )
        for role in expert_configs
    }
    base = None
    base_evidence_sha = None
    initial_model = None
    initial_risk_model = None
    initial_precision_model = None
    initial_recovery_model = None
    if args.base_package is not None:
        base = load_local_policy_package(args.base_package)
        if observed_pilot_target_count and base.manifest.control_feature_contract_sha256 != (
            feature_contract
        ):
            raise ValueError("base feature semantics differ; retrain instead of reusing weights")
        if base.manifest.scope != "general":
            raise ValueError("map specialization must start from a general policy")
        if base.manifest.vehicle_sha256 != sha256_json(vehicle):
            raise ValueError("base policy vehicle binding does not match")
        if base.manifest.sensor_contract_sha256 != sha256_json(sensor):
            raise ValueError("base policy sensor binding does not match")
        if args.base_qualification is not None:
            payload, base_evidence_sha = read_evidence_object(
                args.base_qualification, limit=4 * 1024 * 1024
            )
            base_evidence = LocalPolicyQualificationReceipt.model_validate(payload)
            evidence_granted = base_evidence.qualified
        else:
            assert args.base_simulation_admission is not None
            payload, base_evidence_sha = read_evidence_object(
                args.base_simulation_admission, limit=4 * 1024 * 1024
            )
            base_evidence = LocalPolicySimulationAdmissionReceipt.model_validate(payload)
            evidence_granted = base_evidence.admitted_to_simulation
        if (
            not evidence_granted
            or base_evidence.policy_package_sha256 != base.package_sha256
            or base_evidence.map_sha256 is not None
            or base_evidence.vehicle_sha256 != base.manifest.vehicle_sha256
            or base_evidence.sensor_contract_sha256 != base.manifest.sensor_contract_sha256
            or not local_policy_receipt_supports_control_contract(base, base_evidence)
        ):
            raise ValueError("base policy evidence does not bind the admitted general package")
        initial_model = load_local_policy_onnx(base.artifact_paths["local-navigation-policy"])
        risk_path = base.artifact_paths.get("risk-critic")
        if risk_path is not None:
            initial_risk_model = load_local_risk_critic_onnx(risk_path)
        precision_path = base.artifact_paths.get("precision-maneuver-policy")
        if precision_path is not None:
            initial_precision_model = load_local_policy_onnx(precision_path)
        recovery_path = base.artifact_paths.get("recovery-policy")
        if recovery_path is not None:
            initial_recovery_model = load_local_policy_onnx(recovery_path)
        # Candidate-selection and continuous pilot control are different
        # authority contracts. Reusing a legacy candidate network's hidden
        # representation made a new four-axis head appear "warm started" while
        # retaining coordinate-policy biases. A direct-control specialist gets
        # a fresh backbone; only byte-identical, separately admitted safety
        # experts are intentionally preserved below.
        fresh_direct_control_backbone = bool(
            observed_pilot_target_count and base.manifest.pilot_control_mode is None
        )
        if fresh_direct_control_backbone:
            initial_model = None
            initial_precision_model = None
            initial_recovery_model = None
            initial_risk_model = None
        else:
            initial_model = expand_local_policy_realtime_control_inputs(
                initial_model,
                realtime_feature_count=observed_realtime_count,
                include_pilot_control=bool(observed_pilot_target_count),
                random_seed=args.random_seed,
            )
        if initial_risk_model is not None and not observed_pilot_target_count:
            initial_risk_model = expand_local_policy_realtime_control_inputs(
                initial_risk_model,
                realtime_feature_count=observed_realtime_count,
                include_pilot_control=False,
                random_seed=args.risk_random_seed,
            )
        if initial_precision_model is not None:
            initial_precision_model = expand_local_policy_realtime_control_inputs(
                initial_precision_model,
                realtime_feature_count=observed_realtime_count,
                include_pilot_control=bool(observed_pilot_target_count),
                random_seed=args.random_seed + 101,
            )
        if initial_recovery_model is not None:
            initial_recovery_model = expand_local_policy_realtime_control_inputs(
                initial_recovery_model,
                realtime_feature_count=observed_realtime_count,
                include_pilot_control=bool(observed_pilot_target_count),
                random_seed=args.random_seed + 202,
            )
    if initial_model is not None:
        config = LocalPolicyTrainingConfig.model_validate(
            {**config.model_dump(), "hidden_feature_count": int(initial_model.input_bias.shape[0])}
        )
    if initial_risk_model is not None:
        risk_config = LocalPolicyTrainingConfig.model_validate(
            {
                **risk_config.model_dump(),
                "hidden_feature_count": int(initial_risk_model.input_bias.shape[0]),
            }
        )
    model, training_metrics = train_local_policy(
        training,
        config,
        initial_model=initial_model,
    )
    validation_metrics = evaluate_local_policy(model, validation)
    _validate_navigation_metrics(
        training_metrics,
        prefix="LOCAL_POLICY_TRAINING",
        maximum_pilot_control_mae=args.maximum_pilot_control_mae,
        continuous_control=bool(observed_pilot_target_count),
    )
    if observed_pilot_target_count:
        # Missing counterfactual/executed-action labels are an explicit missing
        # capability, never an invitation to reuse the old candidate critic.
        # Action-risk labels cannot train behavior, so the previous mixed-data
        # branch was unreachable. The independent action-risk training pipeline
        # owns those labels; this baseline output is explicitly development-only.
        risk_model = None
        risk_training_metrics = None
        risk_validation_metrics = None
    else:
        risk_model, risk_training_metrics = train_local_risk_critic(
            training,
            risk_config,
            initial_model=initial_risk_model,
        )
        risk_validation_metrics = evaluate_local_risk_critic(risk_model, validation)
    _validate_navigation_metrics(
        validation_metrics,
        prefix="LOCAL_POLICY_VALIDATION",
        maximum_pilot_control_mae=args.maximum_pilot_control_mae,
        continuous_control=bool(observed_pilot_target_count),
    )
    if risk_validation_metrics is not None and risk_validation_metrics.risk_hold_recall < 0.95:
        raise RuntimeError(
            "LOCAL_RISK_CRITIC_VALIDATION_HOLD_RECALL_TOO_LOW: "
            f"{risk_validation_metrics.risk_hold_recall:.6f}"
        )
    if risk_validation_metrics is not None and risk_validation_metrics.safe_motion_recall < 0.95:
        raise RuntimeError(
            "LOCAL_RISK_CRITIC_VALIDATION_SAFE_RECALL_TOO_LOW: "
            f"{risk_validation_metrics.safe_motion_recall:.6f}"
        )
    if (
        risk_validation_metrics is not None
        and risk_validation_metrics.risk_mean_absolute_error > 0.2
    ):
        raise RuntimeError(
            "LOCAL_RISK_CRITIC_VALIDATION_ERROR_TOO_HIGH: "
            f"{risk_validation_metrics.risk_mean_absolute_error:.6f}"
        )

    expert_models: dict[NavigationExpertRole, NumpyLocalPolicyModel] = {}
    expert_receipts: dict[str, object] = {}
    expert_requests = (
        (
            "precision-maneuver-policy",
            args.train_precision_expert,
            initial_precision_model,
        ),
        (
            "recovery-policy",
            args.train_recovery_expert,
            initial_recovery_model,
        ),
    )
    for role, enabled, initial_expert in expert_requests:
        if not enabled:
            continue
        expert_config = expert_configs[role]
        expert_training, expert_validation = expert_datasets[role]
        # The four continuous axes describe the same controller-neutral body
        # velocity contract in every role.  Specialists therefore start from
        # the all-sample control backbone instead of relearning that physical
        # mapping from a much smaller precision/recovery subset.  Fine-tuning
        # remains independent, so their action and risk heads can specialize.
        specialist_initial = (
            _clone_policy_model(model) if observed_pilot_target_count else initial_expert
        )
        if specialist_initial is not None:
            # 热启动实际保留原骨干宽度，回执不能仍写一个未用于网络的请求宽度。
            expert_config = LocalPolicyTrainingConfig.model_validate(
                {
                    **expert_config.model_dump(),
                    "hidden_feature_count": int(specialist_initial.input_bias.shape[0]),
                }
            )
        # Replay only the training split. Keep the specialist's own examples
        # dominant while retaining the shared physical and hazard response.
        replay = [sample for sample in training if sample.navigation_expert_role != role]
        replay_weight = min(1.0, len(expert_training) / max(1, len(replay)))
        rehearsal = (
            [
                sample.model_copy(
                    update={
                        "sample_weight": sample.sample_weight * replay_weight,
                    }
                )
                for sample in replay
            ]
            if observed_pilot_target_count
            else []
        )
        expert_model, expert_training_metrics = train_local_policy(
            [*expert_training, *rehearsal],
            expert_config,
            initial_model=specialist_initial,
        )
        expert_validation_metrics = evaluate_local_policy(
            expert_model,
            expert_validation,
        )
        _validate_navigation_metrics(
            expert_validation_metrics,
            prefix=f"LOCAL_EXPERT_{role.upper().replace('-', '_')}_VALIDATION",
            maximum_pilot_control_mae=args.maximum_pilot_control_mae,
            continuous_control=bool(observed_pilot_target_count),
        )
        retention_metrics = evaluate_local_policy(expert_model, validation)
        _validate_navigation_metrics(
            retention_metrics,
            prefix=f"LOCAL_EXPERT_{role.upper().replace('-', '_')}_RETENTION",
            maximum_pilot_control_mae=args.maximum_pilot_control_mae,
            continuous_control=bool(observed_pilot_target_count),
        )
        expert_models[role] = expert_model
        expert_receipts[role] = {
            "training_config": expert_config.model_dump(mode="json"),
            "training_metrics": expert_training_metrics.model_dump(mode="json"),
            "validation_metrics": expert_validation_metrics.model_dump(mode="json"),
            "retention_metrics": retention_metrics.model_dump(mode="json"),
            "specialist_training_sample_count": len(expert_training),
            "rehearsal_training_sample_count": len(rehearsal),
        }
    write_local_policy_package(
        control_feature_contract_sha256=feature_contract,
        output_root=args.output_package,
        model=model,
        package_id=args.package_id,
        display_name=args.display_name,
        scope=args.scope,
        vehicle_sha256=sha256_json(vehicle),
        sensor_contract_sha256=sha256_json(sensor),
        maximum_inference_latency_ms=args.maximum_inference_latency_ms,
        risk_model=risk_model,
        include_independent_risk_critic=risk_model is not None,
        precision_model=expert_models.get("precision-maneuver-policy"),
        recovery_model=expert_models.get("recovery-policy"),
        state_anomaly_detector_path=args.state_anomaly_detector,
        perception_health_critic_path=args.perception_health_critic,
        settle_stability_critic_path=args.settle_stability_critic,
        payload_dynamics_adapter_path=args.payload_dynamics_adapter,
        cross_modal_consistency_critic_path=args.cross_modal_consistency_critic,
        perception_encoder_path=args.perception_encoder,
        visual_width=args.visual_width,
        visual_height=args.visual_height,
        visual_feature_count=args.visual_feature_count,
        visual_normalization=args.visual_normalization,
        base_package_sha256=(base.package_sha256 if base is not None else None),
        map_sha256=map_sha,
    )
    package = load_local_policy_package(args.output_package)
    preserved_source_paths = {
        "state-anomaly-detector": args.state_anomaly_detector,
        "perception-health-critic": args.perception_health_critic,
        "settle-stability-critic": args.settle_stability_critic,
        "payload-dynamics-adapter": args.payload_dynamics_adapter,
        "cross-modal-consistency-critic": args.cross_modal_consistency_critic,
        "perception-encoder": args.perception_encoder,
    }
    preserved_artifact_sha256: dict[str, str] = {}
    if base is not None:
        for role, supplied_path in preserved_source_paths.items():
            if supplied_path is None:
                continue
            source_path = base.artifact_paths.get(role)  # type: ignore[arg-type]
            packaged_path = package.artifact_paths.get(role)  # type: ignore[arg-type]
            if (
                source_path is None
                or packaged_path is None
                or _sha256(supplied_path) != _sha256(source_path)
                or _sha256(packaged_path) != _sha256(source_path)
            ):
                raise ValueError(f"preserved base artifact differs: {role}")
            preserved_artifact_sha256[role] = _sha256(packaged_path)
    receipt = {
        "schema_version": "dronedream.local-policy-training-receipt.v1",
        "package_id": package.manifest.package_id,
        "package_sha256": package.package_sha256,
        "control_feature_contract_sha256": package.manifest.control_feature_contract_sha256,
        "scope": package.manifest.scope,
        "source_package_sha256": base.package_sha256 if base is not None else None,
        "source_vehicle_sha256": (base.manifest.vehicle_sha256 if base is not None else None),
        "base_package_sha256": package.manifest.base_package_sha256,
        "base_evidence_sha256": base_evidence_sha,
        "base_evidence_scope": (
            "flight-qualified"
            if args.base_qualification is not None
            else "simulation-admitted"
            if args.base_simulation_admission is not None
            else None
        ),
        "map_sha256": package.manifest.map_sha256,
        "vehicle_sha256": package.manifest.vehicle_sha256,
        "sensor_contract_sha256": package.manifest.sensor_contract_sha256,
        "preserved_artifact_sha256": preserved_artifact_sha256,
        "preserved_advisor_input_contract_unchanged": bool(
            base is not None
            and preserved_artifact_sha256
            and base.manifest.control_feature_contract_sha256
            == package.manifest.control_feature_contract_sha256
        ),
        "fresh_simulation_admission_required": True,
        "training_data_sha256": training_sha,
        "validation_data_sha256": validation_sha,
        "vehicle_source_sha256": vehicle_source_sha,
        "sensor_source_sha256": sensor_source_sha,
        "training_config": config.model_dump(mode="json"),
        "risk_training_config": risk_config.model_dump(mode="json"),
        "training_metrics": training_metrics.model_dump(mode="json"),
        "validation_metrics": validation_metrics.model_dump(mode="json"),
        "risk_training_metrics": (
            risk_training_metrics.model_dump(mode="json")
            if risk_training_metrics is not None
            else None
        ),
        "risk_validation_metrics": (
            risk_validation_metrics.model_dump(mode="json")
            if risk_validation_metrics is not None
            else None
        ),
        "risk_critic_source": (
            "missing-action-risk-labels-development-only"
            if risk_model is None
            else "independent-action-conditioned-training"
            if risk_model.action_conditioned
            else "fresh-independent-training-corpus"
        ),
        "realtime_feature_count": observed_realtime_count,
        "pilot_control_target_count": observed_pilot_target_count,
        "navigation_initialization_source": (
            "fresh-direct-control-contract"
            if observed_pilot_target_count
            and base is not None
            and base.manifest.pilot_control_mode is None
            else "compatible-base-or-fresh-general"
        ),
        "maximum_pilot_control_mae": args.maximum_pilot_control_mae,
        "expert_training": expert_receipts,
        "qualification_granted": False,
        "qualification_requirement": (
            "A separate held-out Gazebo/PX4 campaign must pass before selection."
        ),
    }
    args.training_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(args.training_receipt, receipt)
    print(
        json.dumps(
            {
                "package": str(args.output_package),
                "package_sha256": package.package_sha256,
                "validation_motion_authorization_accuracy": (
                    validation_metrics.motion_authorization_accuracy
                ),
                "validation_candidate_selection_accuracy": (
                    validation_metrics.candidate_selection_accuracy
                ),
                "qualification_granted": False,
            },
            ensure_ascii=False,
        )
    )
    exit_code = 0
    return exit_code


# 功能：
#   在任何优化开始前验证基线、风险及全部启用专家的超参数，防止后续种子越界才中断。
# 输入：
#   args：包含网络宽度、损失权重、训练规模及专家开关的命令行参数。
# 输出：
#   config：通用前馈策略训练配置。
#   risk_config：独立风险网络训练配置。
#   experts：启用的专家类别到其有效训练配置的映射。
def training_configurations(args):
    common = dict(
        epoch_count=args.epoch_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        risk_loss_weight=args.risk_loss_weight,
        pilot_control_loss_weight=args.pilot_control_loss_weight,
        action_class_balance_power=args.action_class_balance_power,
        action_class_weight_cap=args.action_class_weight_cap,
        risk_class_balance_power=args.risk_class_balance_power,
        risk_class_weight_cap=args.risk_class_weight_cap,
    )
    config = LocalPolicyTrainingConfig(
        **common, hidden_feature_count=args.hidden_feature_count, random_seed=args.random_seed
    )
    risk_config = LocalPolicyTrainingConfig(
        **{
            **common,
            "hidden_feature_count": args.risk_hidden_feature_count,
            "epoch_count": args.risk_epoch_count,
            "risk_loss_weight": 1.0,
            "pilot_control_loss_weight": 1.0,
            "random_seed": args.risk_random_seed,
        }
    )
    experts = {}
    for role, enabled, width, offset in (
        (
            "precision-maneuver-policy",
            args.train_precision_expert,
            args.precision_hidden_feature_count,
            101,
        ),
        ("recovery-policy", args.train_recovery_expert, args.recovery_hidden_feature_count, 202),
    ):
        if enabled:
            experts[role] = LocalPolicyTrainingConfig(
                **{
                    **common,
                    "hidden_feature_count": width,
                    "random_seed": args.random_seed + offset,
                    "epoch_count": args.expert_epoch_count
                    if args.expert_epoch_count is not None
                    else args.epoch_count,
                    "learning_rate": args.expert_learning_rate
                    if args.expert_learning_rate is not None
                    else args.learning_rate,
                }
            )
    return config, risk_config, experts


if __name__ == "__main__":
    raise SystemExit(main())
