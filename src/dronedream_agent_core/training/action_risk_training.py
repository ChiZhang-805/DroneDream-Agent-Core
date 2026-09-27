"""Risk-only optimization and independent route-level validation.

No optimizer receives hypothetical probe actions as behavior-cloning targets.
Offline agreement with a nominal teacher is distinct from physical safety.
"""

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..control_feature_contract import (
    CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    FLIGHT_STATE_FEATURE_COUNT,
    GEOMETRY_FEATURE_COUNT,
    SENSOR_FEATURE_COUNT,
)
from ..local_policy_quality import LocalRiskCriticMetrics
from ..local_policy_training import (
    LocalPolicyTrainingConfig,
    LocalPolicyTrainingSample,
    NumpyLocalPolicyModel,
    evaluate_local_risk_critic,
    export_local_risk_critic_onnx,
    policy_input_arrays,
    train_local_risk_critic,
)
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from ..realtime_feature_encoders import POLICY_REALTIME_FEATURE_COUNT
from .action_risk_artifacts import validate_action_risk_splits
from .dagger_artifacts import write_rows
from .evidence_publication import publish_evidence_bytes
from .evidence_snapshot import detach_evidence
from .mission_groups import SPATIAL_SPLIT_CONTRACT
from .risk_nearfield import (
    embed_nearfield_transform,
    nearfield_descriptor,
    project_nearfield_inputs,
)
from .risk_training_sampling import training_selection_receipt


# 功能：
#   为名义几何风险屏蔽任务方向、策略速度尺度和图像采样密度，保留物理运动、净空和动作。
# 输入：
#   samples：原始、已接纳的训练样本，不原地修改。
# 输出：
#   projected：同宽度副本；屏蔽列在归一化训练中保持零权重并折叠入部署网络。
def project_physical_risk_inputs(samples):
    projected = []
    for source in samples:
        sample = LocalPolicyTrainingSample.model_validate(source.model_dump())
        if len(sample.realtime_features) != POLICY_REALTIME_FEATURE_COUNT:
            raise ValueError('ACTION_RISK_REALTIME_WIDTH_INVALID')
        # 风险提案和26格几何都是机体系；世界速度已有机体系速度等价输入，
        # 全局目的地和世界朝向不是相对障碍碰撞的原因，避免地图朝向成为捷径。
        sample.state_features[:7] = [0.] * 7
        # 提案已按固定20m/s物理单位编码；同一个实际动作不能因为策略杆量
        # 的满刻度不同而改变风险。保留13列真实净空要求，不混同于12列杆量尺度。
        sample.state_features[12] = 0.
        flight_offset = SENSOR_FEATURE_COUNT - FLIGHT_STATE_FEATURE_COUNT
        sample.realtime_features[flight_offset:flight_offset + 4] = [0.] * 4
        # 这八个历史导航扇区相对于目标方向，并非机头方向。目标已屏蔽时，
        # 保留其距离会让同一个“前进”提案对应不同方位的障碍；风险网络只用
        # 下方明确的 FRU 26 格几何与掩码，不把目标朝向扇区误当作机体系。
        sample.state_features[14:] = [0.] * (len(sample.state_features) - 14)
        # 几何标签比较扫掠净空，不应把远处场景的像素占比学成障碍概率。
        for offset in range(0, GEOMETRY_FEATURE_COUNT, 5):
            sample.realtime_features[offset + 2:offset + 5] = [0.] * 3
        # 此末段前三列是速度/策略满刻度，前面的飞行状态已提供速度/20m/s。
        # 只保留后者，防止网络把训练课程的限速差异当成障碍或制动距离。
        sample.realtime_features[SENSOR_FEATURE_COUNT:] = [0.] * 10
        projected.append(sample)
    return projected


# 功能：
#   核对统计用的样本、风险标签和观测摘要，拒绝歧义计数，不代替数据集来源接纳。
# 输入：
#   datasets：已经接纳或明确用于纯指标测试的数据集列表。
# 输出：
#   records：按输入次序排列的样本和观测记录对。
def _metric_records(datasets):
    if type(datasets) not in (list, tuple) or not 1 <= len(datasets) <= 128:
        raise ValueError("ACTION_RISK_DATASET_COUNT_INVALID")
    total = sum(len(dataset.samples) for dataset in datasets)
    if not 1 <= total <= 250_000:
        raise ValueError("ACTION_RISK_SAMPLE_COUNT_INVALID")
    records = []
    for dataset in datasets:
        for sample, record in zip(dataset.samples, dataset.records, strict=True):
            if (
                not isinstance(sample, LocalPolicyTrainingSample)
                or type(sample.risk_target) not in (int, float)
                or not math.isfinite(sample.risk_target)
                or not 0 <= sample.risk_target <= 1
            ):
                raise ValueError("ACTION_RISK_TARGET_INVALID")
            key = record.get("observation_sha256") if type(record) is dict else None
            if (
                type(key) is not str
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
            ):
                raise ValueError("ACTION_RISK_OBSERVATION_IDENTITY_INVALID")
            records.append((sample, record))
    return records


# 功能：
#   按独立观测统计安全、危险和动作风险差异，不把同一画面的多个控制探针当作新场景。
# 输入：
#   datasets：待计算覆盖的数据集。
# 输出：
#   coverage：样本、独立观测、类别、动作差异和地图路线组的数量。
def risk_class_coverage(datasets):
    safe, unsafe, contrast = set(), set(), set()
    grouped = defaultdict(list)
    for sample, record in _metric_records(datasets):
        key = record["observation_sha256"]
        (unsafe if sample.risk_target >= 0.5 else safe).add(key)
        grouped[key].append(sample.risk_target)
    for key, targets in grouped.items():
        if max(targets) - min(targets) >= 0.1:
            contrast.add(key)
    coverage = {
        "sample_count": sum(len(dataset.samples) for dataset in datasets),
        "independent_observation_count": len(grouped),
        "safe_observation_count": len(safe),
        "unsafe_observation_count": len(unsafe),
        "action_contrast_observation_count": len(contrast),
        "route_map_group_count": len(set().union(*(dataset.groups for dataset in datasets))),
    }
    return coverage


# 功能：按互斥运动类型分别统计源观测与风险类别，暴露总样本数掩盖的方向覆盖缺口。
# 输入：datasets：已接纳的风险数据；动作使用部署中的物理归一化提案，不读取模型预测。
# 输出：coverage：各类探针数、独立观测数及双类支持量，不将重复探针当作新观测。
def risk_motion_coverage(datasets):
    kinds = ("descending", "ascending", "level-translation", "stationary-or-yaw")
    rows = {kind: {"sample_count": 0, "safe_sample_count": 0, "unsafe_sample_count": 0}
            for kind in kinds}
    identities = {kind: {"all": set(), "safe": set(), "unsafe": set()} for kind in kinds}
    for sample, record in _metric_records(datasets):
        action = sample.risk_proposed_control
        if (type(action) is not list or len(action) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not -1 <= v <= 1 for v in action)):
            raise ValueError("ACTION_RISK_MOTION_CONTROL_INVALID")
        # 组合动作优先按升降分组；仅严格零垂直速度的平移动作计入平飞，不丢掉斜向动作。
        kind = ("descending" if action[2] < 0 else "ascending" if action[2] > 0
                else "level-translation" if action[0] != 0 or action[1] != 0
                else "stationary-or-yaw")
        category = "unsafe" if sample.risk_target >= .5 else "safe"
        rows[kind]["sample_count"] += 1
        rows[kind][category + "_sample_count"] += 1
        identities[kind]["all"].add(record["observation_sha256"])
        identities[kind][category].add(record["observation_sha256"])
    for kind in kinds:
        rows[kind].update(independent_observation_count=len(identities[kind]["all"]),
                          safe_observation_count=len(identities[kind]["safe"]),
                          unsafe_observation_count=len(identities[kind]["unsafe"]))
    return rows


# 功能：
#   在优化前固定检查两个分区的安全、危险和动作差异观测数量，不能按训练结果降低门槛。
# 输入：
#   training：训练分区的数据集。
#   validation：验证分区的数据集。
# 输出：
#   issues：未达到各类二十条独立观测要求的问题码。
def coverage_issues(training, validation):
    issues = []
    # Twenty probes of one image are still one observed state, not twenty
    # demonstrations of independent safety behavior. Never tune this after a run.
    for split, datasets in (("training", training), ("validation", validation)):
        coverage = risk_class_coverage(datasets)
        for field in (
            "safe_observation_count",
            "unsafe_observation_count",
            "action_contrast_observation_count",
        ):
            if coverage[field] < 20:
                issues.append(f"ACTION_RISK_{split.upper()}_{field.upper()}_BELOW_TWENTY")
    return issues


# 功能：
#   校验并复制当前动作条件网络，按部署输入顺序计算有限参考风险，拒绝旧的无动作网络。
# 输入：
#   model：不含视觉旁路的当前动作风险网络。
#   samples：带真实物理归一化候选控制的样本。
# 输出：
#   predictions：一维风险评分。
#   feeds：同一批计算对应的 ONNX 输入张量。
def _predict(model, samples):
    if (
        not isinstance(model, NumpyLocalPolicyModel)
        or model.action_conditioned is not True
        or model.visual_feature_count != 0
        or model.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
    ):
        raise ValueError("ACTION_RISK_MODEL_CONTRACT_INVALID")
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= 250_000:
        raise ValueError("ACTION_RISK_SAMPLE_COUNT_INVALID")
    samples = [LocalPolicyTrainingSample.model_validate(sample.model_dump()) for sample in samples]
    inputs = policy_input_arrays(
        samples, include_visual=False, include_realtime=True, include_proposed_control=True
    )
    source = model.input_weight
    if type(source) is not np.ndarray or source.ndim != 2 or not 8 <= source.shape[1] <= 1024:
        raise ValueError("ACTION_RISK_MODEL_SHAPE_INVALID")
    hidden_width = source.shape[1]
    parameters = {}
    for name, shape in {
        "input_weight": (inputs.shape[1], hidden_width),
        "input_bias": (hidden_width,),
        "risk_weight": (hidden_width, 1),
        "risk_bias": (1,),
    }.items():
        value = getattr(model, name)
        if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != shape:
            raise ValueError("ACTION_RISK_MODEL_SHAPE_INVALID")
        parameters[name] = value.copy()
        if not np.isfinite(parameters[name]).all():
            raise ValueError("ACTION_RISK_MODEL_NONFINITE")
    feeds = {
        "state_features": np.asarray([row.state_features for row in samples], dtype=np.float32),
        "realtime_features": np.asarray(
            [row.realtime_features for row in samples], dtype=np.float32
        ),
        "realtime_valid_mask": np.asarray(
            [row.realtime_valid_mask for row in samples], dtype=np.float32
        ),
        "proposed_control": np.asarray(
            [row.risk_proposed_control for row in samples], dtype=np.float32
        ),
    }
    try:
        with np.errstate(over="raise", invalid="raise"):
            hidden = np.maximum(inputs @ parameters["input_weight"] + parameters["input_bias"], 0.0)
            logits = hidden @ parameters["risk_weight"] + parameters["risk_bias"]
    except FloatingPointError as error:
        raise ValueError("ACTION_RISK_REFERENCE_OVERFLOW") from error
    if not np.isfinite(logits).all():
        raise ValueError("ACTION_RISK_REFERENCE_NONFINITE")
    predictions = (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).reshape(-1)
    return predictions, feeds


# 功能：
#   比较同一观测中最危险与最安全动作的评分，使用并列极值的最差组合避免挑选有利案例。
# 输入：
#   datasets：样本与观测绑定的数据集。
#   predictions：一维、有限且位于零至一范围的数值评分。
# 输出：
#   discrimination：可比较观测数、正确数及比例；没有对照时比例为 None。
def action_discrimination(datasets, predictions):
    groups = defaultdict(list)
    records = _metric_records(datasets)
    predictions = np.asarray(predictions)
    if (
        predictions.dtype.kind not in "fiu"
        or predictions.shape != (len(records),)
        or not np.isfinite(predictions).all()
        or np.any(predictions < 0)
        or np.any(predictions > 1)
    ):
        raise ValueError("ACTION_RISK_PREDICTIONS_INVALID")
    for (sample, record), prediction in zip(records, predictions, strict=True):
        groups[record["observation_sha256"]].append((sample.risk_target, float(prediction)))
    pairs = []
    for rows in groups.values():
        # Compare all tied extreme actions conservatively; don't cherry-pick the
        # one safe/unsafe action pair the model happens to rank correctly.
        minimum, maximum = min(v[0] for v in rows), max(v[0] for v in rows)
        if maximum - minimum < 0.1:
            continue
        predicted_low = max(v[1] for v in rows if v[0] == minimum)
        predicted_high = min(v[1] for v in rows if v[0] == maximum)
        pairs.append(predicted_high > predicted_low)
    discrimination = {
        "observation_count": len(pairs),
        "correct_count": sum(pairs),
        "correct_fraction": sum(pairs) / len(pairs) if pairs else None,
    }
    return discrimination


# 功能：
#   1. 固定已接纳证据和配置，先检查空间隔离与覆盖，再只优化风险网络而非控制动作。
#   2. 核对独立验证和导出等价性，保存完整离线回执，永不授予飞行资格。
# 输入：
#   training：已由数据集加载器接纳的训练分区。
#   validation：已接纳且空间独立的验证分区。
#   config：风险网络训练参数。
#   output：尚不存在的本次离线训练输出目录。
#   classification_margin：仅用于优化的分类间隔；验证仍使用原始几何标签与既定门槛。
#   normalize_inputs：训练集内的量纲归一化开关，导出时折叠为原接口权重。
#   physical_context_only：屏蔽与名义碰撞无关的任务方向和采样密度，不屏蔽有效性掩码。
#   monotonic_geometry：启用并记录几何距离的偏单调实验约束，不更改验收标准。
#   monotonic_uncertainty：约束定位方差到风险输出的有效路径非负，防止低方差域反向外推。
#   geometry_distance_cap_m：可选米制近场范围，变换随模型导出，不裁剪标签。
#   position_uncertainty_floor_m：与教师一致的定位余量下限，生成径向学习特征而非放宽净空。
#   unsafe_sample_cost：仅优化时提高危险样本损失成本，仍独立检查正常动作放行率。
#   contrast_loss_weight：按同一观测的完整动作组训练排序，不读取验证观测进行优化。
#   diagnostic_sources：来源读取器核验过的旧失败分析元数据，不载入其标签参与优化。
# 输出：
#   receipt：明确包含优化、离线验收及后续物理验证要求的回执。
def train_action_risk_expert(training, validation, config: LocalPolicyTrainingConfig, output: Path,
                            *, classification_margin: float = 0.0, normalize_inputs: bool = False,
                            physical_context_only: bool = False, monotonic_geometry: bool = False,
                            geometry_distance_cap_m: float | None = None, unsafe_sample_cost=1.,
                            contrast_loss_weight=0., diagnostic_sources=(), monotonic_uncertainty=False,
                            position_uncertainty_floor_m=None):
    if (type(contrast_loss_weight) not in (int, float) or not math.isfinite(contrast_loss_weight)
            or not 0 <= contrast_loss_weight <= 4.):
        raise ValueError('ACTION_CONTRAST_WEIGHT_INVALID')
    if (type(unsafe_sample_cost) not in (int, float) or not math.isfinite(unsafe_sample_cost)
            or not 1. <= unsafe_sample_cost <= 4.):
        raise ValueError('ACTION_RISK_UNSAFE_SAMPLE_COST_INVALID')
    if type(normalize_inputs) is not bool:
        raise ValueError("ACTION_RISK_NORMALIZATION_INVALID")
    if type(monotonic_geometry) is not bool:
        raise ValueError("ACTION_RISK_MONOTONICITY_INVALID")
    if type(monotonic_uncertainty) is not bool:
        raise ValueError('ACTION_RISK_UNCERTAINTY_CONSTRAINT_INVALID')
    if type(physical_context_only) is not bool or (physical_context_only and not normalize_inputs):
        raise ValueError('ACTION_RISK_PHYSICAL_PROJECTION_REQUIRES_NORMALIZATION')
    transform = None
    if position_uncertainty_floor_m is not None:
        if geometry_distance_cap_m is None:
            raise ValueError('ACTION_RISK_POSITION_RESERVE_REQUIRES_NEARFIELD')
        # 原始方差路径单调不等于经过非线性几何变换后的整个网络单调，禁止错误声明。
        if monotonic_uncertainty:
            raise ValueError('ACTION_RISK_COMPOSED_MONOTONICITY_NOT_VERIFIED')
    if geometry_distance_cap_m is not None:
        transform = nearfield_descriptor(geometry_distance_cap_m, position_uncertainty_floor_m)
        if not physical_context_only:
            raise ValueError('ACTION_RISK_NEARFIELD_REQUIRES_PHYSICAL_PROJECTION')
    if (type(classification_margin) not in (int, float)
            or not math.isfinite(classification_margin) or not 0 <= classification_margin < .5):
        raise ValueError("ACTION_RISK_CLASSIFICATION_MARGIN_INVALID")
    config = LocalPolicyTrainingConfig.model_validate(config.model_dump(), strict=True)
    output = Path(output).absolute()
    check_plain_plugin_path(output)
    _metric_records(training)
    _metric_records(validation)
    selections = [training_selection_receipt(dataset) for dataset in training]
    if any(dataset.training_observation_selection is not None for dataset in validation):
        raise ValueError('ACTION_RISK_VALIDATION_SUBSAMPLING_FORBIDDEN')
    if type(diagnostic_sources) not in (list, tuple) or len(diagnostic_sources) > 16:
        raise ValueError('ACTION_RISK_DIAGNOSTIC_DATASET_COUNT_INVALID')
    diagnostic_sources = detach_evidence(list(diagnostic_sources))
    seen_diagnostics = set()
    for source in diagnostic_sources:
        if (type(source) is not dict or set(source) != {
                'dataset_receipt_sha256', 'teacher_config_sha256', 'groups'}
                or type(source['groups']) is not list or not 1 <= len(source['groups']) <= 10000
                or any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in [source['dataset_receipt_sha256'], *source['groups']])
                or len(set(source['groups'])) != len(source['groups'])
                or source['dataset_receipt_sha256'] in seen_diagnostics):
            raise ValueError('ACTION_RISK_DIAGNOSTIC_SOURCE_INVALID')
        if source['teacher_config_sha256'] != training[0].teacher_config_sha256:
            raise ValueError('ACTION_RISK_DIAGNOSTIC_TEACHER_MISMATCH')
        seen_diagnostics.add(source['dataset_receipt_sha256'])
    # 数据集冻结类内部仍有可变字典和模型；优化器只拿副本，回执不共享调用方容器。
    training, validation = (
        [
            replace(
                dataset,
                samples=tuple(
                    LocalPolicyTrainingSample.model_validate(row.model_dump())
                    for row in dataset.samples
                ),
                records=tuple(detach_evidence(row) for row in dataset.records),
                observations=tuple(detach_evidence(row) for row in dataset.observations),
                receipt=detach_evidence(dataset.receipt),
                groups=frozenset(dataset.groups),
            )
            for dataset in split
        ]
        for split in (training, validation)
    )
    validate_action_risk_splits(training, validation)
    if (position_uncertainty_floor_m is not None
            and any(d.receipt['teacher_config']['position_uncertainty_m']
                    != position_uncertainty_floor_m for d in [*training, *validation])):
        raise ValueError('ACTION_RISK_POSITION_RESERVE_TEACHER_MISMATCH')
    output.mkdir(parents=True, exist_ok=False)
    issues = coverage_issues(training, validation)
    receipt = {
        "purpose": "native-action-risk-offline-training",
        "physical_input_projection": (
            "body-geometry-fixed-physical-units-v4" if physical_context_only else None
        ),
        "control_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "training_config": config.model_dump(),
        "classification_margin": classification_margin,
        "training_only_input_normalization_folded": normalize_inputs,
        "physical_context_only": physical_context_only,
        "monotonic_geometry_distances": monotonic_geometry,
        "monotonic_localization_uncertainty": monotonic_uncertainty,
        "embedded_input_transform": transform,
        "unsafe_sample_cost": unsafe_sample_cost,
        "action_contrast_loss_weight": contrast_loss_weight,
        "action_contrast_margin": .1 if contrast_loss_weight else None,
        "action_contrast_batch_weighting": (
            'global-training-mean-v1' if contrast_loss_weight else None),
        "teacher_config": training[0].receipt["teacher_config"],
        "teacher_config_sha256": training[0].teacher_config_sha256,
        "split_method": "whole-route-map-holdout",
        "split_contract": SPATIAL_SPLIT_CONTRACT,
        "training_groups": sorted(set().union(*(d.groups for d in training))),
        "validation_groups": sorted(set().union(*(d.groups for d in validation))),
        "selection_diagnostic_sources": diagnostic_sources,
        "training_observation_selection": [
            detach_evidence(selection) for selection in selections if selection is not None],
        "dataset_receipts_sha256": {
            "training": [d.receipt_sha256 for d in training],
            "validation": [d.receipt_sha256 for d in validation],
        },
        "coverage": {
            "training": risk_class_coverage(training),
            "validation": risk_class_coverage(validation),
        },
        "motion_coverage": {
            "training": risk_motion_coverage(training),
            "validation": risk_motion_coverage(validation),
        },
        "optimized": False,
        "offline_validation_passed": False,
        "qualified_for_flight": False,
        "not_a_calibrated_collision_probability": True,
        "independent_physical_validation_required": True,
        "issue_codes": issues,
    }
    if not issues:
        train_samples = [sample for d in training for sample in d.samples]
        validation_samples = [sample for d in validation for sample in d.samples]
        optimizer_samples = (project_physical_risk_inputs(train_samples) if physical_context_only
                             else [detach_evidence(row) for row in train_samples])
        if transform is not None:
            optimizer_samples = project_nearfield_inputs(optimizer_samples, geometry_distance_cap_m,
                                                         position_uncertainty_floor_m)
        model, train_metrics = train_local_risk_critic(
            optimizer_samples, config.model_copy(deep=True),
            classification_margin=classification_margin,
            normalize_inputs=normalize_inputs,
            **({'monotonic_geometry': True} if monotonic_geometry else {}),
            **({'monotonic_uncertainty': True} if monotonic_uncertainty else {}),
            **({'unsafe_sample_cost': unsafe_sample_cost} if unsafe_sample_cost != 1. else {}),
            **({'contrast_loss_weight': contrast_loss_weight,
                'contrast_group_ids': [record['observation_sha256']
                                       for dataset in training for record in dataset.records]}
               if contrast_loss_weight else {}),
        )
        # 非线性近场变换不能折叠进矩阵；参考端显式应用，导出验收仍给实际图原始输入。
        train_reference = (project_nearfield_inputs(train_samples, geometry_distance_cap_m,
                                                    position_uncertainty_floor_m)
                           if transform is not None else train_samples)
        validation_reference = (
            project_nearfield_inputs(validation_samples, geometry_distance_cap_m,
                                    position_uncertainty_floor_m)
            if transform is not None else validation_samples)
        train_metrics = evaluate_local_risk_critic(model, train_reference)
        validation_metrics = evaluate_local_risk_critic(
            model, validation_reference
        )
        train_metrics = LocalRiskCriticMetrics.model_validate(
            train_metrics.model_dump(), strict=True
        )
        validation_metrics = LocalRiskCriticMetrics.model_validate(
            validation_metrics.model_dump(), strict=True
        )
        for metrics, rows in (
            (train_metrics, train_samples),
            (validation_metrics, validation_samples),
        ):
            risky_count = sum(row.risk_target >= 0.5 for row in rows)
            if (
                metrics.sample_count != len(rows)
                or metrics.risky_sample_count != risky_count
                or metrics.safe_sample_count != len(rows) - risky_count
            ):
                raise ValueError("ACTION_RISK_METRICS_SAMPLE_COUNTS_DIFFER")
        predictions, _ = _predict(model, validation_reference)
        discrimination = action_discrimination(validation, predictions)
        receipt.update(
            optimized=True,
            training_metrics=train_metrics.model_dump(),
            validation_metrics=validation_metrics.model_dump(),
            action_discrimination=discrimination,
        )
        for split, metrics in (("training", train_metrics), ("validation", validation_metrics)):
            if metrics.risk_hold_recall < 0.95:
                issues.append(f"ACTION_RISK_{split.upper()}_UNSAFE_RECALL_BELOW_THRESHOLD")
            if metrics.safe_motion_recall < 0.95:
                issues.append(f"ACTION_RISK_{split.upper()}_SAFE_RECALL_BELOW_THRESHOLD")
            if metrics.risk_mean_absolute_error > 0.2:
                issues.append(f"ACTION_RISK_{split.upper()}_ERROR_ABOVE_THRESHOLD")
        if discrimination["correct_fraction"] is None or discrimination["correct_fraction"] < 0.95:
            issues.append("ACTION_RISK_VALIDATION_ACTION_DISCRIMINATION_BELOW_THRESHOLD")
        path = output / "risk-critic.onnx"
        source = output / "unwrapped-risk-network.onnx" if transform is not None else path
        export_local_risk_critic_onnx(model, source)
        if transform is not None:
            source_content = read_plugin_file(source, limit=16 * 1024 * 1024)
            receipt["unwrapped_model_sha256"] = hashlib.sha256(source_content).hexdigest()
            publish_evidence_bytes(
                path, embed_nearfield_transform(source_content, geometry_distance_cap_m,
                                                position_uncertainty_floor_m),
                limit=16 * 1024 * 1024)
        content = read_plugin_file(path, limit=16 * 1024 * 1024)
        from .risk_uncertainty_monotonicity import validate_uncertainty_graph

        validate_uncertainty_graph(content, monotonic_uncertainty)
        equivalence = verify_risk_export(model, content, train_samples + validation_samples,
                                        **({'geometry_distance_cap_m': geometry_distance_cap_m,
                                            'position_uncertainty_floor_m': position_uncertainty_floor_m}
                                           if transform is not None else {}))
        receipt.update(
            model_sha256=hashlib.sha256(content).hexdigest(),
            model_bytes=len(content),
            onnx_maximum_absolute_error=equivalence,
            learned_parameter_count=sum(
                v.size
                for v in (model.input_weight, model.input_bias, model.risk_weight, model.risk_bias)
            ),
            offline_validation_passed=not issues,
        )
    write_rows(output / "training-receipt.jsonl", [receipt])
    return receipt


# 功能：
#   使用实际 CPU ONNX 会话分批核对参考输出，拒绝非有限值、形状错误和超过容差的结果。
# 输入：
#   model：待核对的动作条件风险网络。
#   content：已经从同一文件有界读取的 ONNX 原始字节。
#   samples：非空且有界的保留样本。
#   geometry_distance_cap_m：参考端使用的近场距离；ONNX 端始终接收未截断原始输入。
#   position_uncertainty_floor_m：参考端与图内一致的定位余量下限；None 保留旧模型语义。
# 输出：
#   maximum：所有批次中的最大绝对误差。
def verify_risk_export(model, content: bytes, samples, *, geometry_distance_cap_m=None,
                       position_uncertainty_floor_m=None):
    import onnxruntime as ort

    if type(content) is not bytes or not 0 < len(content) <= 16 * 1024 * 1024:
        raise ValueError("ACTION_RISK_EXPORT_BYTES_INVALID")
    if type(samples) not in (list, tuple) or not 1 <= len(samples) <= 250_000:
        raise ValueError("ACTION_RISK_EXPORT_EQUIVALENCE_FAILED")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        content, sess_options=options, providers=["CPUExecutionProvider"]
    )
    maximum = 0.0
    for start in range(0, len(samples), 256):
        expected, feeds = _predict(model, samples[start : start + 256])
        if geometry_distance_cap_m is not None:
            expected, _ = _predict(model, project_nearfield_inputs(
                samples[start : start + 256], geometry_distance_cap_m,
                position_uncertainty_floor_m))
        raw = session.run(["risk_score"], feeds)[0]
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (len(expected), 1)
            or raw.dtype.kind != "f"
            or np.any(raw < 0)
            or np.any(raw > 1)
        ):
            raise ValueError("ACTION_RISK_EXPORT_OUTPUT_INVALID")
        actual = raw.reshape(-1)
        if (
            actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
        ):
            raise ValueError("ACTION_RISK_EXPORT_OUTPUT_INVALID")
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    if not samples or maximum > 1e-5:
        raise ValueError("ACTION_RISK_EXPORT_EQUIVALENCE_FAILED")
    return maximum
