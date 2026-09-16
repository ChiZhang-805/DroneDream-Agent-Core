"""Training and ONNX export for independent local safety advisors.

These compact networks never author trajectories.  They consume the same
bounded tensors used by the local navigation Harness and may only increase the
aggregate risk score or reduce the controller lookahead.  That asymmetric
authority keeps training mistakes from bypassing deterministic path checks.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import Field, model_validator

from .asset_package_storage import publish_asset_file
from .contracts import StrictModel
from .local_policy_packages import (
    LOCAL_POLICY_MANEUVER_FEATURE_COUNT,
    LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
    LOCAL_POLICY_SENSOR_FEATURE_COUNT,
    LOCAL_POLICY_STATE_FEATURE_COUNT,
    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
)
from .plugin_files import check_plain_plugin_path

TrainableAdvisorRole = Literal[
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
]
_MAX_ADVISOR_ARRAY_BYTES = 256 * 1024 * 1024
_MAX_ADVISOR_MODEL_BYTES = 4 * 1024 * 1024


class LocalAdvisorTrainingSample(StrictModel):
    role: TrainableAdvisorRole
    state_features: list[float] = Field(
        min_length=LOCAL_POLICY_STATE_FEATURE_COUNT,
        max_length=LOCAL_POLICY_STATE_FEATURE_COUNT,
    )
    state_history: list[list[float]] = Field(
        min_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
        max_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
    )
    history_mask: list[float] = Field(
        min_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
        max_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
    )
    maneuver_features: list[float] = Field(
        min_length=LOCAL_POLICY_MANEUVER_FEATURE_COUNT,
        max_length=LOCAL_POLICY_MANEUVER_FEATURE_COUNT,
    )
    payload_features: list[float] = Field(
        min_length=LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
        max_length=LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
    )
    payload_history: list[list[float]] | None = Field(
        default=None,
        min_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
        max_length=LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
    )
    sensor_features: list[float] = Field(
        default_factory=lambda: [0.0] * LOCAL_POLICY_SENSOR_FEATURE_COUNT,
        min_length=LOCAL_POLICY_SENSOR_FEATURE_COUNT,
        max_length=LOCAL_POLICY_SENSOR_FEATURE_COUNT,
    )
    risk_target: float = Field(ge=0.0, le=1.0)
    controller_step_scale_target: float = Field(default=1.0, ge=0.1, le=1.0)
    sample_weight: float = Field(default=1.0, gt=0.0, le=100.0)

    # 功能：
    #   校验时间窗口宽度、二进制掩码及角色专属负载输出，拒绝混用其他专家的尺度标签。
    # 输入：
    #   self：字段类型与长度已经初步验证的训练样本。
    # 输出：
    #   self：满足时间与角色契约的样本。
    @model_validator(mode="after")
    def validate_temporal_contract(self) -> LocalAdvisorTrainingSample:
        if any(len(row) != LOCAL_POLICY_STATE_FEATURE_COUNT for row in self.state_history):
            raise ValueError("local advisor state history has an invalid width")
        if any(mask not in {0.0, 1.0} for mask in self.history_mask):
            raise ValueError("local advisor history mask must be binary")
        if self.payload_history is not None and any(
            len(row) != LOCAL_POLICY_PAYLOAD_FEATURE_COUNT for row in self.payload_history
        ):
            raise ValueError("local advisor payload history has an invalid width")
        if self.role == "payload-dynamics-adapter" and self.payload_history is None:
            raise ValueError("payload dynamics samples require dedicated payload history")
        if self.role != "payload-dynamics-adapter" and (self.controller_step_scale_target != 1.0):
            raise ValueError("risk-only advisor samples cannot train a step scale")
        return self


class LocalAdvisorTrainingConfig(StrictModel):
    hidden_feature_count: int = Field(default=32, ge=8, le=512)
    epoch_count: int = Field(default=100, ge=1, le=100_000)
    batch_size: int = Field(default=64, ge=1, le=65_536)
    learning_rate: float = Field(default=0.002, gt=0.0, le=1.0)
    controller_scale_loss_weight: float = Field(default=0.5, ge=0.0, le=100.0)
    l2_weight: float = Field(default=0.00001, ge=0.0, le=1.0)
    random_seed: int = Field(default=2905, ge=0, le=2**32 - 1)


@dataclass(frozen=True)
class NumpyLocalAdvisorModel:
    role: TrainableAdvisorRole
    input_weight: Any
    input_bias: Any
    risk_weight: Any
    risk_bias: Any
    scale_weight: Any | None = None
    scale_bias: Any | None = None


class LocalAdvisorTrainingMetrics(StrictModel):
    sample_count: int = Field(ge=1)
    risky_sample_count: int = Field(ge=0)
    safe_sample_count: int = Field(ge=0)
    risk_hold_recall: float = Field(ge=0.0, le=1.0)
    safe_motion_recall: float = Field(ge=0.0, le=1.0)
    risk_mean_absolute_error: float = Field(ge=0.0, le=1.0)
    controller_step_scale_mean_absolute_error: float | None = Field(default=None, ge=0.0, le=0.9)

    # 功能：
    #   校验危险与安全类别的数量之和等于总样本数，防止指标内部自相矛盾。
    # 输入：
    #   self：已经通过数值范围检查的指标。
    # 输出：
    #   self：数量守恒的指标。
    @model_validator(mode="after")
    def validate_class_counts(self) -> LocalAdvisorTrainingMetrics:
        if self.risky_sample_count + self.safe_sample_count != self.sample_count:
            raise ValueError("local advisor class counts do not sum to sample count")
        return self


# 功能：
#   按明确的专家角色计算扁平输入宽度，未知角色不能回退为负载网络。
# 输入：
#   role：五种可训练专家之一。
# 输出：
#   input_count：该角色输入特征的数量。
def _input_count(role: TrainableAdvisorRole) -> int:
    if role not in get_args(TrainableAdvisorRole):
        raise ValueError("local advisor role is invalid")
    if role == "perception-health-critic":
        input_count = LOCAL_POLICY_STATE_FEATURE_COUNT
    elif role == "cross-modal-consistency-critic":
        input_count = LOCAL_POLICY_SENSOR_FEATURE_COUNT
    else:
        temporal_feature_count = (
            LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
            if role == "payload-dynamics-adapter"
            else LOCAL_POLICY_STATE_FEATURE_COUNT
        )
        input_count = (
            LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH * temporal_feature_count
            + LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
        )
        if role == "settle-stability-critic":
            input_count += LOCAL_POLICY_MANEUVER_FEATURE_COUNT
        elif role == "payload-dynamics-adapter":
            input_count += LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
    return input_count


# 功能：
#   按专家输入顺序展开特征，将掩码为零的历史置零，并保留掩码作为缺测信息。
# 输入：
#   sample：经过完整契约检查的样本。
# 输出：
#   features：与导出图一致的扁平输入向量。
def _sample_input(sample: LocalAdvisorTrainingSample) -> list[float]:
    _input_count(sample.role)
    if sample.role == "perception-health-critic":
        features = list(sample.state_features)
    elif sample.role == "cross-modal-consistency-critic":
        features = list(sample.sensor_features)
    else:
        history = (
            sample.payload_history
            if sample.role == "payload-dynamics-adapter"
            else sample.state_history
        )
        if history is None:
            raise ValueError("payload dynamics samples require dedicated payload history")
        features = [
            value if mask else 0.0
            for row, mask in zip(history, sample.history_mask, strict=True)
            for value in row
        ]
        features.extend(sample.history_mask)
        if sample.role == "settle-stability-critic":
            features = [*sample.maneuver_features, *features]
        elif sample.role == "payload-dynamics-adapter":
            features = [*sample.payload_features, *features]
    return features


# 功能：
#   以固定随机种子初始化单隐层 ReLU 网络，只有负载专家拥有额外的连续尺度输出头。
# 输入：
#   role：明确的专家角色。
#   config：网络宽度与随机种子配置。
# 输出：
#   model：拥有独立 float32 参数的初始网络。
def _new_model(
    role: TrainableAdvisorRole,
    config: LocalAdvisorTrainingConfig,
) -> NumpyLocalAdvisorModel:
    import numpy as np

    config = LocalAdvisorTrainingConfig.model_validate(config.model_dump(), strict=True)
    generator = np.random.default_rng(config.random_seed)
    input_count = _input_count(role)
    input_scale = math.sqrt(2.0 / input_count)
    hidden_scale = math.sqrt(2.0 / config.hidden_feature_count)
    with_scale = role == "payload-dynamics-adapter"
    model = NumpyLocalAdvisorModel(
        role=role,
        input_weight=generator.normal(
            0.0,
            input_scale,
            (input_count, config.hidden_feature_count),
        ).astype(np.float32),
        input_bias=np.zeros(config.hidden_feature_count, dtype=np.float32),
        risk_weight=generator.normal(0.0, hidden_scale, (config.hidden_feature_count, 1)).astype(
            np.float32
        ),
        risk_bias=np.zeros(1, dtype=np.float32),
        scale_weight=(
            generator.normal(0.0, hidden_scale, (config.hidden_feature_count, 1)).astype(np.float32)
            if with_scale
            else None
        ),
        scale_bias=np.zeros(1, dtype=np.float32) if with_scale else None,
    )
    return model


# 功能：
#   重新验证所选角色样本，建立有界且有限的 float32 特征、标签和权重数组。
# 输入：
#   samples：包含一个或多个角色的样本列表。
#   role：本次选中的专家角色。
# 输出：
#   inputs：二维输入特征。
#   risks：风险标签列。
#   scales：控制尺度标签列。
#   weights：正样本权重列。
def _arrays(
    samples: list[LocalAdvisorTrainingSample],
    role: TrainableAdvisorRole,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    width = _input_count(role)
    if type(samples) is not list or not 1 <= len(samples) <= 250_000:
        raise ValueError("local advisor sample count is invalid")
    if any(not isinstance(sample, LocalAdvisorTrainingSample) for sample in samples):
        raise ValueError("local advisor sample type is invalid")
    selected = [sample for sample in samples if sample.role == role]
    if not selected:
        raise ValueError(f"local advisor training data has no {role} samples")
    if len(selected) * (width + 3) * 4 > _MAX_ADVISOR_ARRAY_BYTES:
        raise ValueError("local advisor dense array byte budget exceeded")
    # 复核已有模型实例并脱离调用方容器；不能只信任构造时曾通过验证。
    selected = [
        LocalAdvisorTrainingSample.model_validate(sample.model_dump(), strict=True)
        for sample in selected
    ]
    try:
        with np.errstate(over="raise", invalid="raise"):
            inputs = np.asarray([_sample_input(sample) for sample in selected], dtype=np.float32)
            risks = np.asarray([[sample.risk_target] for sample in selected], dtype=np.float32)
            scales = np.asarray(
                [[sample.controller_step_scale_target] for sample in selected], dtype=np.float32
            )
            weights = np.asarray([[sample.sample_weight] for sample in selected], dtype=np.float32)
    except FloatingPointError as error:
        raise ValueError("local advisor float32 conversion overflow") from error
    if any(not np.isfinite(array).all() for array in (inputs, risks, scales, weights)):
        raise ValueError("local advisor arrays must be finite")
    if not (weights > 0).all():
        raise ValueError("local advisor sample weight underflows float32")
    return inputs, risks, scales, weights


# 功能：
#   计算共享隐层、风险及可选尺度输出，拒绝溢出中间值而不把无穷大裁成合法风险。
# 输入：
#   model：参数形状已验证的网络。
#   inputs：有限的二维 float32 特征批次。
# 输出：
#   hidden_pre：ReLU 前的隐层值。
#   hidden：ReLU 后的隐层值。
#   risk：零到一之间的风险列。
#   scale：负载尺度列，风险专家为 None。
#   scale_sigmoid：尺度单元值，供反向传播计算梯度。
def _forward(model: NumpyLocalAdvisorModel, inputs: Any) -> tuple[Any, ...]:
    import numpy as np

    hidden_pre = inputs @ model.input_weight + model.input_bias
    if not np.isfinite(hidden_pre).all():
        raise ValueError("local advisor hidden values are nonfinite")
    hidden = np.maximum(hidden_pre, 0.0)
    risk_logit = hidden @ model.risk_weight + model.risk_bias
    if not np.isfinite(risk_logit).all():
        raise ValueError("local advisor risk logits are nonfinite")
    risk = 1.0 / (1.0 + np.exp(-np.clip(risk_logit, -40.0, 40.0)))
    scale = None
    scale_sigmoid = None
    if model.scale_weight is not None and model.scale_bias is not None:
        scale_logit = hidden @ model.scale_weight + model.scale_bias
        if not np.isfinite(scale_logit).all():
            raise ValueError("local advisor scale logits are nonfinite")
        scale_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(scale_logit, -40.0, 40.0)))
        scale = 0.1 + 0.9 * scale_sigmoid
    return hidden_pre, hidden, risk, scale, scale_sigmoid


# 功能：
#   1. 冻结输入与配置，以 Adam 优化类别平衡的二元风险损失和独立的负载尺度损失。
#   2. 检查梯度及更新数值，返回训练分区指标；独立验证及飞行接纳由上层另行执行。
# 输入：
#   samples：本次训练数据，不包含独立验证分区。
#   config：学习率、批量、轮数、损失权重及种子。
#   role：本次训练的专家角色。
# 输出：
#   model：优化后的独立专家参数。
#   metrics：本次冻结训练数组上的指标。
def train_local_advisor(
    samples: list[LocalAdvisorTrainingSample],
    config: LocalAdvisorTrainingConfig,
    *,
    role: TrainableAdvisorRole,
) -> tuple[NumpyLocalAdvisorModel, LocalAdvisorTrainingMetrics]:
    import numpy as np

    config = LocalAdvisorTrainingConfig.model_validate(config.model_dump(), strict=True)
    inputs, risks, scales, weights = _arrays(samples, role)
    risky_mask = risks[:, 0] >= 0.5
    risky_count = int(risky_mask.sum())
    safe_count = len(risks) - risky_count
    risk_class_weights = np.ones_like(weights)
    if risky_count and safe_count:
        # Real flight corpora are correctly dominated by nominal motion.  A
        # plain average would let a network ignore rare fail-closed examples
        # while still reporting deceptively low aggregate error.  Balance only
        # the risk head by observed class frequency; retain the original sample
        # weights for the continuous payload step-scale head.
        risk_class_weights[risky_mask, 0] = len(risks) / (2.0 * risky_count)
        risk_class_weights[~risky_mask, 0] = len(risks) / (2.0 * safe_count)
    risk_weights = weights * risk_class_weights
    model = _new_model(role, config)
    parameters = [
        model.input_weight,
        model.input_bias,
        model.risk_weight,
        model.risk_bias,
    ]
    if model.scale_weight is not None and model.scale_bias is not None:
        parameters.extend((model.scale_weight, model.scale_bias))
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    generator = np.random.default_rng(config.random_seed)
    step = 0
    for _epoch in range(config.epoch_count):
        order = generator.permutation(len(inputs))
        for start in range(0, len(inputs), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_inputs = inputs[indices]
            batch_risks = risks[indices]
            batch_scales = scales[indices]
            batch_weights = weights[indices]
            batch_risk_weights = risk_weights[indices]
            hidden_pre, hidden, risk, scale, scale_sigmoid = _forward(model, batch_inputs)
            risk_normalizer = max(1e-9, float(batch_risk_weights.sum()))
            risk_delta = (risk - batch_risks) * batch_risk_weights / risk_normalizer
            risk_weight_gradient = hidden.T @ risk_delta + config.l2_weight * model.risk_weight
            risk_bias_gradient = risk_delta.sum(axis=0)
            hidden_delta = risk_delta @ model.risk_weight.T
            gradients: list[Any] = []
            scale_weight_gradient = None
            scale_bias_gradient = None
            if scale is not None and scale_sigmoid is not None and model.scale_weight is not None:
                scale_delta = (
                    2.0
                    * (scale - batch_scales)
                    * 0.9
                    * scale_sigmoid
                    * (1.0 - scale_sigmoid)
                    * batch_weights
                    / max(1e-9, float(batch_weights.sum()))
                    * config.controller_scale_loss_weight
                )
                scale_weight_gradient = (
                    hidden.T @ scale_delta + config.l2_weight * model.scale_weight
                )
                scale_bias_gradient = scale_delta.sum(axis=0)
                hidden_delta += scale_delta @ model.scale_weight.T
            hidden_delta[hidden_pre <= 0.0] = 0.0
            input_weight_gradient = (
                batch_inputs.T @ hidden_delta + config.l2_weight * model.input_weight
            )
            input_bias_gradient = hidden_delta.sum(axis=0)
            gradients.extend(
                (
                    input_weight_gradient,
                    input_bias_gradient,
                    risk_weight_gradient,
                    risk_bias_gradient,
                )
            )
            if scale_weight_gradient is not None and scale_bias_gradient is not None:
                gradients.extend((scale_weight_gradient, scale_bias_gradient))
            if any(not np.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("local advisor training gradient is nonfinite")
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                first_moment[index] *= 0.9
                first_moment[index] += 0.1 * gradient
                second_moment[index] *= 0.999
                second_moment[index] += 0.001 * gradient * gradient
                corrected_first = first_moment[index] / (1.0 - 0.9**step)
                corrected_second = second_moment[index] / (1.0 - 0.999**step)
                parameter -= (
                    config.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
                )
                if any(
                    not np.isfinite(value).all()
                    for value in (first_moment[index], second_moment[index], parameter)
                ):
                    raise ValueError("local advisor optimizer state is nonfinite")
    # 指标必须基于真正参与优化的数组，不能在训练结束后重新读取调用方可变样本。
    metrics = _evaluate_arrays(model, inputs, risks, scales)
    return model, metrics


# 功能：
#   复制并验证网络权重和指定角色样本，独立计算风险召回与回归误差。
# 输入：
#   model：待评价的网络权重。
#   samples：评价分区的数据。
# 输出：
#   metrics：原始类别数量及未加权评价指标。
def evaluate_local_advisor(
    model: NumpyLocalAdvisorModel,
    samples: list[LocalAdvisorTrainingSample],
) -> LocalAdvisorTrainingMetrics:
    model = _validated_model(model)
    inputs, risks, scales, _weights = _arrays(samples, model.role)
    metrics = _evaluate_arrays(model, inputs, risks, scales)
    return metrics


# 功能：
#   对已经冻结的数组计算类别召回和绝对误差；缺少类别仍由上层覆盖门禁拒绝。
# 输入：
#   model：已验证网络。
#   inputs：冻结输入数组。
#   risks：冻结风险标签。
#   scales：冻结尺度标签。
# 输出：
#   metrics：数组对应的类别数量、召回和误差。
def _evaluate_arrays(model, inputs, risks, scales) -> LocalAdvisorTrainingMetrics:
    import numpy as np

    _hidden_pre, _hidden, predicted_risk, predicted_scale, _sigmoid = _forward(model, inputs)
    risky = risks[:, 0] >= 0.5
    safe = ~risky
    predicted_risky = predicted_risk[:, 0] >= 0.5
    hold_recall = float((predicted_risky[risky]).mean()) if risky.any() else 1.0
    safe_recall = float((~predicted_risky[safe]).mean()) if safe.any() else 1.0
    scale_mae = None
    if predicted_scale is not None:
        scale_mae = float(np.abs(predicted_scale - scales).mean())
    metrics = LocalAdvisorTrainingMetrics(
        sample_count=len(inputs),
        risky_sample_count=int(risky.sum()),
        safe_sample_count=int(safe.sum()),
        risk_hold_recall=hold_recall,
        safe_motion_recall=safe_recall,
        risk_mean_absolute_error=float(np.abs(predicted_risk - risks).mean()),
        controller_step_scale_mean_absolute_error=scale_mae,
    )
    return metrics


# 功能：
#   验证网络角色、精确参数形状、dtype 与有限值，再复制为独立参数快照。
# 输入：
#   model：待评估或导出的 NumPy 专家网络。
# 输出：
#   snapshot：不与调用方共享数组的合法网络。
def _validated_model(model: NumpyLocalAdvisorModel) -> NumpyLocalAdvisorModel:
    import numpy as np

    if not isinstance(model, NumpyLocalAdvisorModel):
        raise ValueError("local advisor model type is invalid")
    width = _input_count(model.role)
    source = model.input_weight
    if type(source) is not np.ndarray or source.ndim != 2 or not 8 <= source.shape[1] <= 512:
        raise ValueError("local advisor hidden shape is invalid")
    hidden = source.shape[1]
    shapes = {
        "input_weight": (width, hidden),
        "input_bias": (hidden,),
        "risk_weight": (hidden, 1),
        "risk_bias": (1,),
    }
    if model.role == "payload-dynamics-adapter":
        shapes.update(scale_weight=(hidden, 1), scale_bias=(1,))
    elif model.scale_weight is not None or model.scale_bias is not None:
        raise ValueError("risk-only advisor has unexpected scale weights")
    parameters = {}
    for name, shape in shapes.items():
        value = getattr(model, name)
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.float32:
            raise ValueError("local advisor parameter shape or dtype is invalid: " + name)
        copied = value.copy()
        if not np.isfinite(copied).all():
            raise ValueError("local advisor parameter is nonfinite: " + name)
        parameters[name] = copied
    snapshot = NumpyLocalAdvisorModel(role=model.role, **parameters)
    return snapshot


# 功能：
#   冻结权重并导出与本机历史掩码一致的 ONNX 图，通过结构校验后无覆盖发布。
# 输入：
#   model：待导出的合法网络。
#   output_path：必须尚不存在的目标文件路径。
# 输出：
#   checksum：已发布 ONNX 原始字节的 SHA-256。
def export_local_advisor_onnx(
    model: NumpyLocalAdvisorModel,
    output_path: Path,
) -> str:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    output_path = Path(output_path).absolute()
    check_plain_plugin_path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    model = _validated_model(model)
    initializers = [
        numpy_helper.from_array(model.input_weight, "input_weight"),
        numpy_helper.from_array(model.input_bias, "input_bias"),
        numpy_helper.from_array(model.risk_weight, "risk_weight"),
        numpy_helper.from_array(model.risk_bias, "risk_bias"),
    ]
    if model.role == "perception-health-critic":
        graph_inputs = [
            helper.make_tensor_value_info(
                "state_features",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_STATE_FEATURE_COUNT],
            )
        ]
        nodes = []
        flat_input = "state_features"
    elif model.role == "cross-modal-consistency-critic":
        graph_inputs = [
            helper.make_tensor_value_info(
                "sensor_features",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_SENSOR_FEATURE_COUNT],
            )
        ]
        nodes = []
        flat_input = "sensor_features"
    elif model.role == "settle-stability-critic":
        graph_inputs = [
            helper.make_tensor_value_info(
                "maneuver_features",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_MANEUVER_FEATURE_COUNT],
            ),
            helper.make_tensor_value_info(
                "state_history",
                TensorProto.FLOAT,
                [
                    None,
                    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                    LOCAL_POLICY_STATE_FEATURE_COUNT,
                ],
            ),
            helper.make_tensor_value_info(
                "history_mask",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH],
            ),
        ]
        nodes = [
            helper.make_node("Flatten", ["masked_history"], ["flat_history"], axis=1),
            helper.make_node(
                "Concat",
                ["maneuver_features", "flat_history", "history_mask"],
                ["advisor_features"],
                axis=1,
            ),
        ]
        flat_input = "advisor_features"
    elif model.role == "state-anomaly-detector":
        graph_inputs = [
            helper.make_tensor_value_info(
                "state_history",
                TensorProto.FLOAT,
                [
                    None,
                    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                    LOCAL_POLICY_STATE_FEATURE_COUNT,
                ],
            ),
            helper.make_tensor_value_info(
                "history_mask",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH],
            ),
        ]
        nodes = [
            helper.make_node("Flatten", ["masked_history"], ["flat_history"], axis=1),
            helper.make_node(
                "Concat",
                ["flat_history", "history_mask"],
                ["advisor_features"],
                axis=1,
            ),
        ]
        flat_input = "advisor_features"
    else:
        graph_inputs = [
            helper.make_tensor_value_info(
                "payload_features",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_PAYLOAD_FEATURE_COUNT],
            ),
            helper.make_tensor_value_info(
                "payload_history",
                TensorProto.FLOAT,
                [
                    None,
                    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                    LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
                ],
            ),
            helper.make_tensor_value_info(
                "history_mask",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH],
            ),
        ]
        nodes = [
            helper.make_node("Flatten", ["masked_history"], ["flat_history"], axis=1),
            helper.make_node(
                "Concat",
                ["payload_features", "flat_history", "history_mask"],
                ["advisor_features"],
                axis=1,
            ),
        ]
        flat_input = "advisor_features"
    if model.role in {
        "settle-stability-critic",
        "state-anomaly-detector",
        "payload-dynamics-adapter",
    }:
        history_name = (
            "payload_history" if model.role == "payload-dynamics-adapter" else "state_history"
        )
        initializers.append(numpy_helper.from_array(np.asarray([2], dtype=np.int64), "mask_axes"))
        # 同本机编码：先按时间掩码清零，再展开；缺测槽中的残余值不能成为学习特征。
        nodes[0:0] = [
            helper.make_node("Unsqueeze", ["history_mask", "mask_axes"], ["expanded_mask"]),
            helper.make_node("Mul", [history_name, "expanded_mask"], ["masked_history"]),
        ]
    nodes.extend(
        (
            helper.make_node("MatMul", [flat_input, "input_weight"], ["hidden_linear"]),
            helper.make_node("Add", ["hidden_linear", "input_bias"], ["hidden_biased"]),
            helper.make_node("Relu", ["hidden_biased"], ["hidden"]),
            helper.make_node("MatMul", ["hidden", "risk_weight"], ["risk_linear"]),
            helper.make_node("Add", ["risk_linear", "risk_bias"], ["risk_logit"]),
            helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
        )
    )
    primary_output_name = (
        "anomaly_score" if model.role == "state-anomaly-detector" else "risk_score"
    )
    if primary_output_name != "risk_score":
        nodes.append(helper.make_node("Identity", ["risk_score"], [primary_output_name]))
    graph_outputs = [
        helper.make_tensor_value_info(primary_output_name, TensorProto.FLOAT, [None, 1])
    ]
    if model.role == "payload-dynamics-adapter":
        if model.scale_weight is None or model.scale_bias is None:
            raise ValueError("payload advisor is missing controller scale weights")
        initializers.extend(
            (
                numpy_helper.from_array(model.scale_weight, "scale_weight"),
                numpy_helper.from_array(model.scale_bias, "scale_bias"),
                numpy_helper.from_array(np.asarray([0.9], dtype=np.float32), "scale_span"),
                numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), "scale_floor"),
            )
        )
        nodes.extend(
            (
                helper.make_node("MatMul", ["hidden", "scale_weight"], ["scale_linear"]),
                helper.make_node("Add", ["scale_linear", "scale_bias"], ["scale_logit"]),
                helper.make_node("Sigmoid", ["scale_logit"], ["scale_unit"]),
                helper.make_node("Mul", ["scale_unit", "scale_span"], ["scale_stretched"]),
                helper.make_node(
                    "Add",
                    ["scale_stretched", "scale_floor"],
                    ["controller_step_scale"],
                ),
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info("controller_step_scale", TensorProto.FLOAT, [None, 1])
        )
    graph = helper.make_graph(
        nodes,
        f"dronedream-{model.role}",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    document = helper.make_model(
        graph,
        producer_name=f"DroneDream {model.role} trainer",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document, full_check=True)
    content = document.SerializeToString()
    if not 0 < len(content) <= _MAX_ADVISOR_MODEL_BYTES:
        raise ValueError("local advisor ONNX byte budget exceeded")
    checksum = hashlib.sha256(content).hexdigest()
    check_plain_plugin_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 暂存目录只包含本次导出的文件；最终路径使用共享无覆盖发布器。
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".advisor-export-") as root:
        temporary = Path(root) / "advisor.onnx"
        with temporary.open("xb") as stream:
            if stream.write(content) != len(content):
                raise OSError("local advisor ONNX short write")
            stream.flush()
            os.fsync(stream.fileno())
        publish_asset_file(
            temporary, output_path, expected_sha256=checksum, limit=_MAX_ADVISOR_MODEL_BYTES
        )
    return checksum
