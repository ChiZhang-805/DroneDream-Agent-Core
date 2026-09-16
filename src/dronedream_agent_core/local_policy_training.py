"""Reproducible behavior-cloning baseline for small local navigation policies.

The trainer is intentionally NumPy-based: it produces a compact feed-forward
policy without requiring a large training framework in the product Runtime.
This feed-forward baseline is distinct from the current recurrent causal pilot;
its checkpoints cannot be substituted for that pilot's GRU checkpoints. Shared
observation types do not imply interchangeable weights. Runtime qualification
remains independent of training metrics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from .asset_package_storage import publish_asset_directory
from .contracts import StrictModel
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .local_expert_harness import NavigationExpertRole
from .local_policy_packages import (
    LOCAL_POLICY_CANDIDATE_FEATURE_COUNT,
    LOCAL_POLICY_MAXIMUM_CANDIDATES,
    LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,
    LOCAL_POLICY_STATE_FEATURE_COUNT,
    LocalPolicyArtifact,
    LocalPolicyPackageManifest,
    load_local_policy_package,
)
from .local_policy_quality import (
    LocalPolicyTrainingMetrics,
    LocalRiskCriticMetrics,
    summarize_pilot_axes,
)
from .pilot_control_mapping import ACTION_RISK_FEATURE_COUNT, PilotControlLimits
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .realtime_feature_encoders import (
    POLICY_REALTIME_FEATURE_COUNT,
    REALTIME_CONTROL_FEATURE_COUNT,
)
from .temporal_evidence import TemporalEvidence

LOCAL_POLICY_LEGACY_ACTION_COUNT = LOCAL_POLICY_MAXIMUM_CANDIDATES + 3
LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX = LOCAL_POLICY_LEGACY_ACTION_COUNT
LOCAL_POLICY_ACTION_COUNT = LOCAL_POLICY_LEGACY_ACTION_COUNT + 1
LOCAL_POLICY_FLAT_FEATURE_COUNT = (
    LOCAL_POLICY_STATE_FEATURE_COUNT
    + LOCAL_POLICY_MAXIMUM_CANDIDATES * LOCAL_POLICY_CANDIDATE_FEATURE_COUNT
    + LOCAL_POLICY_MAXIMUM_CANDIDATES
)


class LocalPolicyObservation(StrictModel):
    """Deployed policy inputs without invented teacher actions or risk labels."""

    temporal_evidence: TemporalEvidence | None = None
    pilot_control_limits: PilotControlLimits | None = None
    control_feature_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    navigation_expert_role: NavigationExpertRole = "local-navigation-policy"
    source_snapshot_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_visual_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    state_features: list[float] = Field(
        min_length=LOCAL_POLICY_STATE_FEATURE_COUNT,
        max_length=LOCAL_POLICY_STATE_FEATURE_COUNT,
    )
    candidate_features: list[list[float]] = Field(
        min_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
        max_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
    )
    candidate_mask: list[float] = Field(
        min_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
        max_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
    )
    visual_features: list[float] = Field(default_factory=list, max_length=65_536)
    realtime_features: list[float] = Field(
        default_factory=list,
        max_length=POLICY_REALTIME_FEATURE_COUNT,
    )
    realtime_valid_mask: list[float] = Field(
        default_factory=list,
        max_length=POLICY_REALTIME_FEATURE_COUNT,
    )

    # 功能：
    #   核验观测各段宽度、二值有效掩码和有限数值，不把缺失传感器当成有效观测。
    # 输入：
    #   self：含状态、候选、实时及视觉特征的观测对象。
    # 输出：
    #   self：通过观测结构校验的对象。
    @model_validator(mode="after")
    def validate_observation_shapes(self):
        if any(
            len(features) != LOCAL_POLICY_CANDIDATE_FEATURE_COUNT
            for features in self.candidate_features
        ):
            raise ValueError("local policy candidate training features have an invalid width")
        if any(mask not in {0.0, 1.0} for mask in self.candidate_mask):
            raise ValueError("local policy candidate mask must be binary")
        if len(self.realtime_features) != len(self.realtime_valid_mask):
            raise ValueError("realtime training features and mask must have equal width")
        if self.realtime_features and len(self.realtime_features) not in {
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("realtime training features have an invalid width")
        if any(mask not in {0.0, 1.0} for mask in self.realtime_valid_mask):
            raise ValueError("realtime training feature mask must be binary")
        values = (
            *self.state_features,
            *self.visual_features,
            *self.realtime_features,
            *(value for row in self.candidate_features for value in row),
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("POLICY_OBSERVATION_NONFINITE_FEATURE")
        return self


class LocalPolicyTrainingSample(LocalPolicyObservation):
    """An observation plus separately verified supervision, never a live input stub."""

    target_pilot_control: list[float] = Field(
        default_factory=list,
        max_length=LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,
    )
    target_action_index: int = Field(ge=0, lt=LOCAL_POLICY_ACTION_COUNT)
    risk_target: float = Field(ge=0.0, le=1.0)
    # Assessed proposal, not teacher replacement; physical FRU /20 and yaw /180.
    risk_proposed_control: list[float] = Field(
        default_factory=list, max_length=ACTION_RISK_FEATURE_COUNT
    )
    sample_weight: float = Field(default=1.0, gt=0.0, le=100.0)

    # 功能：
    #   核验监督动作的四轴幅度、停留中立值和候选可用性，风险提案与模仿动作分开存放。
    # 输入：
    #   self：观测及独立监督标签组成的样本。
    # 输出：
    #   self：通过动作与标签约束的样本。
    @model_validator(mode="after")
    def validate_shapes_and_target(self):
        if self.risk_proposed_control and (
            len(self.risk_proposed_control) != ACTION_RISK_FEATURE_COUNT
            or any(
                not math.isfinite(value) or not -1 <= value <= 1
                for value in self.risk_proposed_control
            )
        ):
            raise ValueError("risk proposed control must have four finite physical-normalized axes")
        if self.target_pilot_control and len(self.target_pilot_control) != (
            LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT
        ):
            raise ValueError("pilot-control training target has an invalid width")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.target_pilot_control
        ):
            raise ValueError("pilot-control training targets must be finite normalized axes")
        if (
            self.target_pilot_control
            and self.target_action_index
            in range(LOCAL_POLICY_MAXIMUM_CANDIDATES, LOCAL_POLICY_LEGACY_ACTION_COUNT)
            and any(abs(value) > 1e-9 for value in self.target_pilot_control)
        ):
            raise ValueError("non-motion policy targets require neutral pilot-control axes")
        if (
            self.target_action_index == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
            and len(self.target_pilot_control) != LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT
        ):
            raise ValueError("pilot-control action requires four explicit control axes")
        # Zero velocity/rate is a valid pilot request for braking or tracking
        # zero motion. It is not the safety layer's position-hold control law.
        # Excluding it here made the learning target disagree with deployment.
        if (
            self.target_action_index < LOCAL_POLICY_MAXIMUM_CANDIDATES
            and self.candidate_mask[self.target_action_index] != 1.0
        ):
            raise ValueError("local policy target selects a masked candidate")
        return self


class LocalPolicyTrainingConfig(StrictModel):
    hidden_feature_count: int = Field(default=64, ge=8, le=1_024)
    epoch_count: int = Field(default=120, ge=1, le=100_000)
    batch_size: int = Field(default=64, ge=1, le=65_536)
    learning_rate: float = Field(default=0.002, gt=0.0, le=1.0)
    risk_loss_weight: float = Field(default=0.5, ge=0.0, le=100.0)
    pilot_control_loss_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    action_class_balance_power: float = Field(default=1.0, ge=0.0, le=1.0)
    action_class_weight_cap: float = Field(default=16.0, ge=1.0, le=1_000.0)
    risk_class_balance_power: float = Field(default=1.0, ge=0.0, le=1.0)
    risk_class_weight_cap: float = Field(default=16.0, ge=1.0, le=1_000.0)
    l2_weight: float = Field(default=0.00001, ge=0.0, le=1.0)
    random_seed: int = Field(default=805, ge=0, le=2**32 - 1)


@dataclass(frozen=True)
class NumpyLocalPolicyModel:
    input_weight: Any
    input_bias: Any
    action_weight: Any
    action_bias: Any
    risk_weight: Any
    risk_bias: Any
    visual_feature_count: int = 0
    realtime_feature_count: int = 0
    pilot_control_weight: Any | None = None
    pilot_control_bias: Any | None = None
    action_conditioned: bool = False


# 功能：
#   按训练标签频数提高少数动作的梯度权重，并限制倍数，不使用验证集频数。
# 输入：
#   action_targets：零起始的整数动作类别。
#   sample_weights：与类别对齐的正且有限的原始样本权重。
#   config：类别平衡幂次与倍率上限。
# 输出：
#   balanced：保持原权重数组形状的平衡权重。
def action_class_balanced_sample_weights(
    action_targets: Any,
    sample_weights: Any,
    config: LocalPolicyTrainingConfig,
) -> Any:
    import numpy as np

    config = LocalPolicyTrainingConfig.model_validate(config.model_dump())
    raw_targets = np.asarray(action_targets)
    if (
        raw_targets.dtype.kind not in "iuf"
        or not np.isfinite(raw_targets).all()
        or np.any(raw_targets < 0)
        or np.any(raw_targets >= LOCAL_POLICY_ACTION_COUNT)
        or np.any(raw_targets != np.floor(raw_targets))
    ):
        raise ValueError("local policy action labels must be bounded integers")
    targets = raw_targets.astype(np.int64).reshape(-1)
    weight_array = np.asarray(sample_weights, dtype=np.float32)
    original_weight_shape = weight_array.shape
    weights = weight_array.reshape(-1)
    if (
        not targets.size
        or targets.shape != weights.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("action targets and sample weights must be finite and aligned")
    if config.action_class_balance_power == 0.0:
        balanced = weights.copy().reshape(original_weight_shape)
        return balanced
    counts = np.bincount(targets, minlength=LOCAL_POLICY_ACTION_COUNT)
    populated_class_count = int((counts > 0).sum())
    if populated_class_count <= 1:
        balanced = weights.copy().reshape(original_weight_shape)
        return balanced
    multipliers = np.ones(LOCAL_POLICY_ACTION_COUNT, dtype=np.float32)
    for class_index, count in enumerate(counts):
        if count > 0:
            multipliers[class_index] = min(
                config.action_class_weight_cap,
                (len(targets) / (populated_class_count * int(count)))
                ** config.action_class_balance_power,
            )
    balanced = (weights * multipliers[targets]).reshape(original_weight_shape)
    if not np.isfinite(balanced).all():
        raise ValueError("local policy balanced weights overflow")
    return balanced


# 功能：
#   在安全和危险样本间按训练集频数平衡梯度，保留原始权重并限制放大倍数。
# 输入：
#   risk_targets：零至一的有限监督风险，0.5 为类别分界。
#   sample_weights：与风险标签对齐的正且有限权重。
#   config：风险平衡幂次及倍率上限。
# 输出：
#   balanced：形状与原权重一致的风险损失权重。
def risk_class_balanced_sample_weights(
    risk_targets: Any,
    sample_weights: Any,
    config: LocalPolicyTrainingConfig,
) -> Any:
    import numpy as np

    config = LocalPolicyTrainingConfig.model_validate(config.model_dump())
    target_array = np.asarray(risk_targets, dtype=np.float32)
    weight_array = np.asarray(sample_weights, dtype=np.float32)
    original_weight_shape = weight_array.shape
    targets = target_array.reshape(-1)
    weights = weight_array.reshape(-1)
    if (
        not targets.size
        or targets.shape != weights.shape
        or not np.isfinite(weights).all()
        or not np.isfinite(targets).all()
        or np.any(weights <= 0)
        or np.any(targets < 0)
        or np.any(targets > 1)
    ):
        raise ValueError("risk targets and sample weights must be finite and aligned")
    risky = targets >= 0.5
    safe = ~risky
    risky_count = int(risky.sum())
    safe_count = int(safe.sum())
    if risky_count == 0 or safe_count == 0 or config.risk_class_balance_power == 0.0:
        balanced = weights.copy().reshape(original_weight_shape)
        return balanced
    total = len(targets)
    safe_multiplier = min(
        config.risk_class_weight_cap,
        (total / (2.0 * safe_count)) ** config.risk_class_balance_power,
    )
    risky_multiplier = min(
        config.risk_class_weight_cap,
        (total / (2.0 * risky_count)) ** config.risk_class_balance_power,
    )
    balanced = weights * np.where(risky, risky_multiplier, safe_multiplier).astype(np.float32)
    if not np.isfinite(balanced).all():
        raise ValueError("local risk balanced weights overflow")
    balanced = balanced.reshape(original_weight_shape)
    return balanced


@dataclass(frozen=True)
class _TrainingArrays:
    inputs: Any
    masks: Any
    targets: Any
    risks: Any
    weights: Any
    pilot_controls: Any | None


# 功能：
#   按前馈网络固定顺序拼接特征及有效掩码，动作风险模式用被评估提案替代候选表。
# 输入：
#   samples：非空且各特征段宽度一致的观测。
#   include_visual：是否包含视觉特征。
#   include_realtime：是否包含实时特征及对应掩码。
#   include_proposed_control：是否追加独立风险监督中的四轴提案。
# 输出：
#   result：批次大小乘展平特征数的有限 float32 矩阵。
def policy_input_arrays(
    samples: list[LocalPolicyObservation],
    *,
    include_visual: bool = True,
    include_realtime: bool = True,
    include_proposed_control: bool = False,
):
    import numpy as np

    if not samples:
        raise ValueError("POLICY_OBSERVATIONS_EMPTY")
    if len({len(s.visual_features) for s in samples}) != 1:
        raise ValueError("local policy visual feature count varies within the dataset")
    if len({len(s.realtime_features) for s in samples}) != 1:
        raise ValueError("local policy realtime feature count varies within the dataset")
    if include_proposed_control and any(
        not isinstance(s, LocalPolicyTrainingSample)
        or len(s.risk_proposed_control) != ACTION_RISK_FEATURE_COUNT
        for s in samples
    ):
        raise ValueError("ACTION_RISK_REQUIRES_ASSESSED_PROPOSAL")
    rows = [
        [
            *sample.state_features,
            *(
                value
                for row in sample.candidate_features
                for value in row
                if not include_proposed_control
            ),
            *(sample.candidate_mask if not include_proposed_control else ()),
            *(sample.realtime_features if include_realtime else ()),
            *(sample.realtime_valid_mask if include_realtime else ()),
            *(sample.visual_features if include_visual else ()),
            *(sample.risk_proposed_control if include_proposed_control else ()),
        ]
        for sample in samples
    ]
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("POLICY_OBSERVATION_NONFINITE_FEATURE")
    return result


# 功能：
#   拒绝把被评估的风险动作当成应该模仿的教师动作，防止两种监督混用。
# 输入：
#   samples：准备训练或评价行为策略的样本。
# 输出：
#   None：不返回业务数据。
def require_behavior_supervision(samples) -> None:
    if any(sample.risk_proposed_control for sample in samples):
        raise ValueError("ACTION_RISK_SUPERVISION_CANNOT_TRAIN_OR_EVALUATE_BEHAVIOR")


# 功能：
#   重验离线样本并按统一输入顺序建立监督数组，拒绝混合候选和连续控制标签。
# 输入：
#   samples：离线训练或评价样本，不直接信任可变实例的旧校验结果。
#   include_visual：是否编码视觉特征。
#   include_realtime：是否编码实时特征及掩码。
#   include_proposed_control：是否使用动作风险监督契约。
# 输出：
#   arrays：特征、候选掩码、动作、风险、权重及可选四轴目标组成的数组集。
def _arrays(
    samples: list[LocalPolicyTrainingSample],
    *,
    include_visual: bool = True,
    include_realtime: bool = True,
    include_proposed_control: bool = False,
) -> _TrainingArrays:
    if not samples:
        raise ValueError("local policy training data cannot be empty")
    import numpy as np

    samples = [LocalPolicyTrainingSample.model_validate(sample.model_dump()) for sample in samples]
    if not include_proposed_control:
        require_behavior_supervision(samples)
    if include_proposed_control and any(
        len(sample.risk_proposed_control) != ACTION_RISK_FEATURE_COUNT
        or sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        or len(sample.realtime_features) != POLICY_REALTIME_FEATURE_COUNT
        for sample in samples
    ):
        raise ValueError(
            "action risk requires explicit proposed controls and current feature semantics"
        )

    visual_counts = {len(sample.visual_features) for sample in samples}
    if len(visual_counts) != 1:
        raise ValueError("local policy visual feature count varies within the dataset")
    realtime_counts = {len(sample.realtime_features) for sample in samples}
    if len(realtime_counts) != 1:
        raise ValueError("local policy realtime feature count varies within the dataset")
    pilot_target_counts = {len(sample.target_pilot_control) for sample in samples}
    if len(pilot_target_counts) != 1:
        raise ValueError("local policy pilot-control labels vary within the dataset")
    pilot_target_count = next(iter(pilot_target_counts))
    if pilot_target_count:
        if next(iter(realtime_counts)) != POLICY_REALTIME_FEATURE_COUNT:
            raise ValueError(
                "pilot-control training requires the egocentric realtime control contract"
            )
        if any(sample.target_action_index < LOCAL_POLICY_MAXIMUM_CANDIDATES for sample in samples):
            raise ValueError("pilot-control training cannot mix legacy candidate-selection targets")
    arrays = _TrainingArrays(
        inputs=policy_input_arrays(
            samples,
            include_visual=include_visual,
            include_realtime=include_realtime,
            include_proposed_control=include_proposed_control,
        ),
        masks=np.asarray([sample.candidate_mask for sample in samples], dtype=np.float32),
        targets=np.asarray([sample.target_action_index for sample in samples], dtype=np.int64),
        risks=np.asarray([[sample.risk_target] for sample in samples], dtype=np.float32),
        weights=np.asarray([[sample.sample_weight] for sample in samples], dtype=np.float32),
        pilot_controls=(
            np.asarray(
                [sample.target_pilot_control for sample in samples],
                dtype=np.float32,
            )
            if pilot_target_count > 0
            else None
        ),
    )
    if not np.isfinite(arrays.weights).all() or np.any(arrays.weights <= 0):
        raise ValueError("local policy sample weights cannot underflow float32")
    return arrays


# 功能：
#   计算 ReLU 骨干、掩码后的动作分数、风险概率及可选 tanh 四轴控制，保留反传中间量。
# 输入：
#   model：形状已核验的前馈网络。
#   inputs：批次展平特征矩阵。
#   masks：每条观测的候选可用性掩码。
# 输出：
#   forward：激活前后骨干、动作分数与概率、风险概率和四轴控制组成的元组。
def _forward(model: NumpyLocalPolicyModel, inputs: Any, masks: Any) -> tuple[Any, ...]:
    import numpy as np

    hidden_pre = inputs @ model.input_weight + model.input_bias
    hidden = np.maximum(hidden_pre, 0.0)
    logits = hidden @ model.action_weight + model.action_bias
    logits = logits.copy()
    logits[:, :LOCAL_POLICY_MAXIMUM_CANDIDATES] += (1.0 - masks) * -10_000.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    risk_logits = hidden @ model.risk_weight + model.risk_bias
    risk = 1.0 / (1.0 + np.exp(-np.clip(risk_logits, -40.0, 40.0)))
    pilot_control = None
    if model.pilot_control_weight is not None and model.pilot_control_bias is not None:
        pilot_control = np.tanh(hidden @ model.pilot_control_weight + model.pilot_control_bias)
    forward = hidden_pre, hidden, logits, probabilities, risk, pilot_control
    return forward


# 功能：
#   按固定随机种子初始化前馈基线，不把随机参数视为训练结果或飞行资格。
# 输入：
#   config：骨干宽度及随机种子。
#   visual_feature_count：视觉特征尾段宽度。
#   realtime_feature_count：实时特征宽度，掩码占同等宽度。
#   include_pilot_control：是否创建四轴控制头及对应动作类别。
#   action_conditioned：是否用状态与被评估动作构建风险网络输入。
# 输出：
#   model：各参数独立持有的 float32 网络。
def _new_model(
    config: LocalPolicyTrainingConfig,
    *,
    visual_feature_count: int = 0,
    realtime_feature_count: int = 0,
    include_pilot_control: bool = False,
    action_conditioned: bool = False,
) -> NumpyLocalPolicyModel:
    import numpy as np

    config = LocalPolicyTrainingConfig.model_validate(config.model_dump())
    if (
        type(visual_feature_count) is not int
        or not 0 <= visual_feature_count <= 65_536
        or type(realtime_feature_count) is not int
        or realtime_feature_count
        not in {0, REALTIME_CONTROL_FEATURE_COUNT, POLICY_REALTIME_FEATURE_COUNT}
        or type(include_pilot_control) is not bool
        or type(action_conditioned) is not bool
        or (
            action_conditioned
            and (include_pilot_control or realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT)
        )
    ):
        raise ValueError("NUMPY_POLICY_INITIALIZATION_CONTRACT_INVALID")
    generator = np.random.default_rng(config.random_seed)
    input_feature_count = (
        (
            LOCAL_POLICY_STATE_FEATURE_COUNT + ACTION_RISK_FEATURE_COUNT
            if action_conditioned
            else LOCAL_POLICY_FLAT_FEATURE_COUNT
        )
        + 2 * realtime_feature_count
        + visual_feature_count
    )
    input_scale = math.sqrt(2.0 / input_feature_count)
    hidden_scale = math.sqrt(2.0 / config.hidden_feature_count)
    action_count = (
        LOCAL_POLICY_ACTION_COUNT if include_pilot_control else LOCAL_POLICY_LEGACY_ACTION_COUNT
    )
    model = NumpyLocalPolicyModel(
        input_weight=generator.normal(
            0.0,
            input_scale,
            (input_feature_count, config.hidden_feature_count),
        ).astype(np.float32),
        input_bias=np.zeros(config.hidden_feature_count, dtype=np.float32),
        action_weight=generator.normal(
            0.0,
            hidden_scale,
            (config.hidden_feature_count, action_count),
        ).astype(np.float32),
        action_bias=np.zeros(action_count, dtype=np.float32),
        risk_weight=generator.normal(0.0, hidden_scale, (config.hidden_feature_count, 1)).astype(
            np.float32
        ),
        risk_bias=np.zeros(1, dtype=np.float32),
        visual_feature_count=visual_feature_count,
        realtime_feature_count=realtime_feature_count,
        action_conditioned=action_conditioned,
        pilot_control_weight=(
            generator.normal(
                0.0,
                hidden_scale,
                (config.hidden_feature_count, LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT),
            ).astype(np.float32)
            if include_pilot_control
            else None
        ),
        pilot_control_bias=(
            np.zeros(LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT, dtype=np.float32)
            if include_pilot_control
            else None
        ),
    )
    return model


# 功能：
#   为无视觉风险网络加入小幅随机视觉权重，动作提案仍置于末尾，原参数独立复制。
# 输入：
#   model：原风险模型，不允许已存在视觉特征。
#   visual_feature_count：新增视觉特征数。
#   random_seed：新增权重的随机种子。
# 输出：
#   expanded：保持原输入语义且需要重新训练验证的模型副本。
def expand_local_risk_critic_visual_inputs(
    model: NumpyLocalPolicyModel,
    *,
    visual_feature_count: int,
    random_seed: int,
) -> NumpyLocalPolicyModel:
    import numpy as np

    validate_numpy_policy_model(model)
    if model.visual_feature_count != 0:
        raise ValueError("risk critic already consumes visual features")
    if type(visual_feature_count) is not int or not 1 <= visual_feature_count <= 65_536:
        raise ValueError("visual risk critic requires a positive feature count")
    hidden_count = int(model.input_bias.shape[0])
    generator = np.random.default_rng(random_seed)
    visual_weight = generator.normal(
        0.0,
        0.01 / math.sqrt(max(1, visual_feature_count)),
        (visual_feature_count, hidden_count),
    ).astype(np.float32)
    expanded = NumpyLocalPolicyModel(
        input_weight=(
            np.concatenate(
                [
                    model.input_weight[:-ACTION_RISK_FEATURE_COUNT].copy(),
                    visual_weight,
                    model.input_weight[-ACTION_RISK_FEATURE_COUNT:].copy(),
                ],
                axis=0,
            )
            if model.action_conditioned
            else np.concatenate([model.input_weight.copy(), visual_weight], axis=0)
        ),
        input_bias=model.input_bias.copy(),
        action_weight=model.action_weight.copy(),
        action_bias=model.action_bias.copy(),
        risk_weight=model.risk_weight.copy(),
        risk_bias=model.risk_bias.copy(),
        visual_feature_count=visual_feature_count,
        realtime_feature_count=model.realtime_feature_count,
        action_conditioned=model.action_conditioned,
        pilot_control_weight=(
            model.pilot_control_weight.copy() if model.pilot_control_weight is not None else None
        ),
        pilot_control_bias=(
            model.pilot_control_bias.copy() if model.pilot_control_bias is not None else None
        ),
    )
    return expanded


# 功能：
#   1. 在原候选和视觉权重之间扩展实时特征及掩码，保留既有权重对应的行语义。
#   2. 按需新增四轴控制头，拒绝移除已有控制能力或用该适配器迁移动作风险网络。
# 输入：
#   model：待扩展的前馈行为网络。
#   realtime_feature_count：目标实时特征宽度。
#   include_pilot_control：扩展后是否包含四轴控制。
#   random_seed：新增参数的随机种子。
# 输出：
#   expanded：独立持有权重、需要重新训练验收的前馈模型。
def expand_local_policy_realtime_control_inputs(
    model: NumpyLocalPolicyModel,
    *,
    realtime_feature_count: int,
    include_pilot_control: bool,
    random_seed: int,
) -> NumpyLocalPolicyModel:
    import numpy as np

    validate_numpy_policy_model(model)
    if (
        type(realtime_feature_count) is not int
        or type(include_pilot_control) is not bool
        or realtime_feature_count < model.realtime_feature_count
    ):
        raise ValueError("realtime policy expansion cannot shrink or relabel existing inputs")
    if model.action_conditioned:
        raise ValueError("an action critic cannot be upgraded through the navigation model adapter")

    if realtime_feature_count not in {
        0,
        REALTIME_CONTROL_FEATURE_COUNT,
        POLICY_REALTIME_FEATURE_COUNT,
    }:
        raise ValueError("realtime policy expansion has an invalid feature width")
    if model.realtime_feature_count not in {
        0,
        REALTIME_CONTROL_FEATURE_COUNT,
        realtime_feature_count,
    }:
        raise ValueError("policy already has a different realtime feature contract")
    model_has_pilot = (
        model.pilot_control_weight is not None and model.pilot_control_bias is not None
    )
    if (model.pilot_control_weight is None) != (model.pilot_control_bias is None):
        raise ValueError("policy has an incomplete pilot-control head")
    if model_has_pilot and not include_pilot_control:
        raise ValueError("cannot remove an existing pilot-control head")
    generator = np.random.default_rng(random_seed)
    hidden_count = int(model.input_bias.shape[0])
    input_weight = model.input_weight.copy()
    if model.realtime_feature_count == 0 and realtime_feature_count > 0:
        base_rows = input_weight[:LOCAL_POLICY_FLAT_FEATURE_COUNT]
        visual_rows = input_weight[LOCAL_POLICY_FLAT_FEATURE_COUNT:]
        realtime_rows = generator.normal(
            0.0,
            0.01 / math.sqrt(2 * realtime_feature_count),
            (2 * realtime_feature_count, hidden_count),
        ).astype(np.float32)
        input_weight = np.concatenate(
            (base_rows, realtime_rows, visual_rows),
            axis=0,
        )
    elif model.realtime_feature_count < realtime_feature_count:
        old_count = model.realtime_feature_count
        additional_count = realtime_feature_count - old_count
        base_rows = input_weight[:LOCAL_POLICY_FLAT_FEATURE_COUNT]
        feature_rows = input_weight[
            LOCAL_POLICY_FLAT_FEATURE_COUNT : LOCAL_POLICY_FLAT_FEATURE_COUNT + old_count
        ]
        mask_rows = input_weight[
            LOCAL_POLICY_FLAT_FEATURE_COUNT + old_count : LOCAL_POLICY_FLAT_FEATURE_COUNT
            + 2 * old_count
        ]
        visual_rows = input_weight[LOCAL_POLICY_FLAT_FEATURE_COUNT + 2 * old_count :]
        added_feature_rows = generator.normal(
            0.0,
            0.01 / math.sqrt(2 * realtime_feature_count),
            (additional_count, hidden_count),
        ).astype(np.float32)
        added_mask_rows = generator.normal(
            0.0,
            0.01 / math.sqrt(2 * realtime_feature_count),
            (additional_count, hidden_count),
        ).astype(np.float32)
        input_weight = np.concatenate(
            (
                base_rows,
                feature_rows,
                added_feature_rows,
                mask_rows,
                added_mask_rows,
                visual_rows,
            ),
            axis=0,
        )
    pilot_weight = (
        model.pilot_control_weight.copy() if model.pilot_control_weight is not None else None
    )
    pilot_bias = model.pilot_control_bias.copy() if model.pilot_control_bias is not None else None
    if include_pilot_control and not model_has_pilot:
        pilot_weight = generator.normal(
            0.0,
            0.01 / math.sqrt(LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT),
            (hidden_count, LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT),
        ).astype(np.float32)
        pilot_bias = np.zeros(LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT, dtype=np.float32)
    action_weight = model.action_weight.copy()
    action_bias = model.action_bias.copy()
    if include_pilot_control and action_weight.shape[1] == LOCAL_POLICY_LEGACY_ACTION_COUNT:
        action_weight = np.concatenate(
            (
                action_weight,
                generator.normal(
                    0.0,
                    0.01 / math.sqrt(hidden_count),
                    (hidden_count, 1),
                ).astype(np.float32),
            ),
            axis=1,
        )
        action_bias = np.concatenate((action_bias, np.zeros(1, dtype=np.float32)))
    expanded = NumpyLocalPolicyModel(
        input_weight=input_weight,
        input_bias=model.input_bias.copy(),
        action_weight=action_weight,
        action_bias=action_bias,
        risk_weight=model.risk_weight.copy(),
        risk_bias=model.risk_bias.copy(),
        visual_feature_count=model.visual_feature_count,
        realtime_feature_count=realtime_feature_count,
        pilot_control_weight=pilot_weight,
        pilot_control_bias=pilot_bias,
    )
    return expanded


# 功能：
#   核验前馈网络的数值、各头形状及输入宽度，阻止残缺控制头或广播偏置混入训练和导出。
# 输入：
#   model：含 NumPy 参数的前馈模型，不接受 GRU 检查点。
# 输出：
#   None：不返回业务数据。
def validate_numpy_policy_model(model: NumpyLocalPolicyModel) -> None:
    import numpy as np

    if (
        type(model.visual_feature_count) is not int
        or not 0 <= model.visual_feature_count <= 65_536
        or type(model.realtime_feature_count) is not int
        or model.realtime_feature_count
        not in {0, REALTIME_CONTROL_FEATURE_COUNT, POLICY_REALTIME_FEATURE_COUNT}
        or type(model.action_conditioned) is not bool
    ):
        raise ValueError("NUMPY_POLICY_FEATURE_CONTRACT_INVALID")
    pilot = model.pilot_control_weight is not None
    if pilot != (model.pilot_control_bias is not None):
        raise ValueError("NUMPY_POLICY_CONTROL_HEAD_INCOMPLETE")
    if model.action_conditioned and (
        pilot or model.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
    ):
        raise ValueError("NUMPY_POLICY_ACTION_RISK_CONTRACT_INVALID")
    if not isinstance(model.input_bias, np.ndarray) or model.input_bias.ndim != 1:
        raise ValueError("NUMPY_POLICY_HIDDEN_SHAPE_INVALID")
    hidden = model.input_bias.size
    if not 1 <= hidden <= 1024:
        raise ValueError("NUMPY_POLICY_HIDDEN_WIDTH_INVALID")
    width = (
        (
            LOCAL_POLICY_STATE_FEATURE_COUNT + ACTION_RISK_FEATURE_COUNT
            if model.action_conditioned
            else LOCAL_POLICY_FLAT_FEATURE_COUNT
        )
        + 2 * model.realtime_feature_count
        + model.visual_feature_count
    )
    actions = LOCAL_POLICY_ACTION_COUNT if pilot else LOCAL_POLICY_LEGACY_ACTION_COUNT
    expected = {
        "input_weight": (width, hidden),
        "input_bias": (hidden,),
        "action_weight": (hidden, actions),
        "action_bias": (actions,),
        "risk_weight": (hidden, 1),
        "risk_bias": (1,),
    }
    if pilot:
        expected.update(
            pilot_control_weight=(hidden, LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT),
            pilot_control_bias=(LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,),
        )
    for name, shape in expected.items():
        value = getattr(model, name)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != shape
            or not np.isfinite(value).all()
        ):
            raise ValueError("NUMPY_POLICY_PARAMETER_INVALID:" + name)


# 功能：
#   校验并复制全部训练参数，使热启动成功或异常均不原地改写调用方基座。
# 输入：
#   model：准备作为训练初始值的模型。
# 输出：
#   owned：不共享参数内存的可训练副本。
def copy_numpy_policy_model(model: NumpyLocalPolicyModel) -> NumpyLocalPolicyModel:
    validate_numpy_policy_model(model)
    names = (
        "input_weight",
        "input_bias",
        "action_weight",
        "action_bias",
        "risk_weight",
        "risk_bias",
        "pilot_control_weight",
        "pilot_control_bias",
    )
    owned = replace(
        model,
        **{name: getattr(model, name).copy() for name in names if getattr(model, name) is not None},
    )
    return owned


# 功能：
#   计算一次 Adam 更新并先检查全部候选参数和动量，数值溢出时不提交这一步。
# 输入：
#   parameters：当前训练器独立持有的参数数组。
#   gradients：与参数一一对应的梯度。
#   first_moment：各参数的一阶动量。
#   second_moment：各参数的二阶动量。
#   step：从一开始的更新计数。
#   learning_rate：当前有效学习率。
# 输出：
#   None：成功时原地提交本步参数及动量，不返回业务数据。
def _adam_update(parameters, gradients, first_moment, second_moment, step, learning_rate):
    import numpy as np

    pending = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for parameter, gradient, first, second in zip(
            parameters, gradients, first_moment, second_moment, strict=True
        ):
            if gradient.shape != parameter.shape or not np.isfinite(gradient).all():
                raise ValueError("NUMPY_POLICY_GRADIENT_INVALID")
            next_first = 0.9 * first + 0.1 * gradient
            next_second = 0.999 * second + 0.001 * gradient * gradient
            corrected_first = next_first / (1.0 - 0.9**step)
            corrected_second = next_second / (1.0 - 0.999**step)
            next_parameter = parameter - learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            if any(
                not np.isfinite(value).all() for value in (next_first, next_second, next_parameter)
            ):
                raise ValueError("NUMPY_POLICY_OPTIMIZER_NONFINITE")
            pending.append((next_parameter, next_first, next_second))
    for parameter, first, second, values in zip(
        parameters, first_moment, second_moment, pending, strict=True
    ):
        parameter[:], first[:], second[:] = values


# 功能：
#   1. 使用加权行为克隆训练前馈动作、风险及可选四轴头，连续模式不竞争坐标候选。
#   2. 无变化示范轴固定为示范常量，训练统计不代表部署或飞行准入。
# 输入：
#   samples：行为监督训练集，不能含动作风险提案标签。
#   config：网络初始化、损失及 Adam 超参数。
#   initial_model：可选热启动网络，训练只改变其副本。
# 输出：
#   model：训练后前馈候选模型。
#   metrics：在本训练集上计算的指标。
def train_local_policy(
    samples: list[LocalPolicyTrainingSample],
    config: LocalPolicyTrainingConfig,
    *,
    initial_model: NumpyLocalPolicyModel | None = None,
) -> tuple[NumpyLocalPolicyModel, LocalPolicyTrainingMetrics]:
    import numpy as np

    config = LocalPolicyTrainingConfig.model_validate(config.model_dump())
    if any(sample.target_pilot_control for sample in samples) and any(
        sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        for sample in samples
    ):
        raise ValueError("continuous control training requires matching feature semantics")
    arrays = _arrays(samples)
    balanced_action_weights = action_class_balanced_sample_weights(
        arrays.targets,
        arrays.weights,
        config,
    )
    balanced_risk_weights = risk_class_balanced_sample_weights(
        arrays.risks,
        arrays.weights,
        config,
    )
    visual_feature_count = len(samples[0].visual_features)
    realtime_feature_count = len(samples[0].realtime_features)
    model = (
        copy_numpy_policy_model(initial_model)
        if initial_model is not None
        else _new_model(
            config,
            visual_feature_count=visual_feature_count,
            realtime_feature_count=realtime_feature_count,
            include_pilot_control=arrays.pilot_controls is not None,
        )
    )
    validate_numpy_policy_model(model)
    if model.action_conditioned:
        raise ValueError("risk critic cannot be trained as a behavior policy")
    if model.visual_feature_count != visual_feature_count:
        raise ValueError("initial policy visual feature contract differs from dataset")
    if model.realtime_feature_count != realtime_feature_count:
        raise ValueError("initial policy realtime feature contract differs from dataset")
    model_has_pilot_control = (
        model.pilot_control_weight is not None and model.pilot_control_bias is not None
    )
    if model_has_pilot_control != (arrays.pilot_controls is not None):
        raise ValueError("initial policy pilot-control contract differs from dataset")
    parameters = [
        model.input_weight,
        model.input_bias,
        model.action_weight,
        model.action_bias,
        model.risk_weight,
        model.risk_bias,
    ]
    if model_has_pilot_control:
        parameters.extend((model.pilot_control_weight, model.pilot_control_bias))
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    generator = np.random.default_rng(config.random_seed)
    step = 0
    for _epoch in range(config.epoch_count):
        order = generator.permutation(len(samples))
        for start in range(0, len(samples), config.batch_size):
            indices = order[start : start + config.batch_size]
            inputs = arrays.inputs[indices]
            masks = arrays.masks[indices]
            targets = arrays.targets[indices]
            risk_targets = arrays.risks[indices]
            action_weights = balanced_action_weights[indices]
            risk_weights = balanced_risk_weights[indices]
            hidden_pre, hidden, logits, probabilities, risk, pilot_control = _forward(
                model, inputs, masks
            )
            action_normalizer = max(1e-9, float(action_weights.sum()))
            if model_has_pilot_control:
                # Continuous packages do not choose a pre-authored candidate.
                # Their mode head decides among hold/scan/abort/pilot-control;
                # the independent four-axis head then supplies the actual stick
                # magnitude.  Keeping legacy candidate logits out of this
                # softmax prevents a stale candidate classifier from silently
                # competing with direct control.
                action_logits = logits[:, LOCAL_POLICY_MAXIMUM_CANDIDATES:]
                shifted = action_logits - action_logits.max(axis=1, keepdims=True)
                action_probabilities = np.exp(shifted)
                action_probabilities /= action_probabilities.sum(axis=1, keepdims=True)
                action_delta = np.zeros_like(probabilities)
                action_delta[:, LOCAL_POLICY_MAXIMUM_CANDIDATES:] = action_probabilities
                action_delta[
                    np.arange(len(indices)),
                    targets,
                ] -= 1.0
            else:
                action_delta = probabilities
                action_delta[np.arange(len(indices)), targets] -= 1.0
            action_delta *= action_weights / action_normalizer
            risk_normalizer = max(1e-9, float(risk_weights.sum()))
            risk_delta = (
                (risk - risk_targets) * risk_weights / risk_normalizer * config.risk_loss_weight
            )
            action_weight_gradient = (
                hidden.T @ action_delta + config.l2_weight * model.action_weight
            )
            action_bias_gradient = action_delta.sum(axis=0)
            risk_weight_gradient = hidden.T @ risk_delta + config.l2_weight * model.risk_weight
            risk_bias_gradient = risk_delta.sum(axis=0)
            hidden_delta = action_delta @ model.action_weight.T + risk_delta @ model.risk_weight.T
            pilot_gradients: list[Any] = []
            if model_has_pilot_control:
                if pilot_control is None or arrays.pilot_controls is None:
                    raise AssertionError("pilot-control training tensors are unavailable")
                pilot_targets = arrays.pilot_controls[indices]
                # A long transit contains far more non-zero controls than
                # settle/hover commands.  Reuse the independently bounded
                # risk-class balance so a compact control head learns a real
                # neutral stick instead of averaging rare holds into motion.
                pilot_weights = risk_weights
                pilot_normalizer = max(1e-9, float(pilot_weights.sum()))
                pilot_delta = (
                    2.0
                    * (pilot_control - pilot_targets)
                    / LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT
                    * (pilot_weights / pilot_normalizer)
                    * config.pilot_control_loss_weight
                    * (1.0 - pilot_control * pilot_control)
                )
                if model.pilot_control_weight is None:
                    raise AssertionError("pilot-control output weight is unavailable")
                pilot_gradients = [
                    hidden.T @ pilot_delta + config.l2_weight * model.pilot_control_weight,
                    pilot_delta.sum(axis=0),
                ]
                hidden_delta += pilot_delta @ model.pilot_control_weight.T
            hidden_delta[hidden_pre <= 0.0] = 0.0
            input_weight_gradient = inputs.T @ hidden_delta + config.l2_weight * model.input_weight
            input_bias_gradient = hidden_delta.sum(axis=0)
            gradients = [
                input_weight_gradient,
                input_bias_gradient,
                action_weight_gradient,
                action_bias_gradient,
                risk_weight_gradient,
                risk_bias_gradient,
                *pilot_gradients,
            ]
            step += 1
            _adam_update(
                parameters, gradients, first_moment, second_moment, step, config.learning_rate
            )
    if model_has_pilot_control:
        if arrays.pilot_controls is None or model.pilot_control_weight is None:
            raise AssertionError("pilot-control training tensors are unavailable")
        # A control axis with no demonstrated variation has no evidence for
        # extrapolation.  Leaving an ordinary dense column fitted to a
        # constant target lets unseen sensor/visual states produce arbitrary
        # stick motion even though every teacher command kept that axis
        # centred.  Freeze such axes exactly at the demonstrated value.  This
        # is especially important for yaw when the qualified route-tangent
        # controller, rather than a learned yaw policy, generated the corpus.
        axis_ranges = np.ptp(arrays.pilot_controls, axis=0)
        for axis, axis_range in enumerate(axis_ranges):
            if float(axis_range) > 1e-7:
                continue
            constant = float(arrays.pilot_controls[0, axis])
            bounded = float(np.clip(constant, -0.999999, 0.999999))
            model.pilot_control_weight[:, axis] = 0.0
            model.pilot_control_bias[axis] = np.arctanh(bounded)
    metrics = evaluate_local_policy(model, samples)
    return model, metrics


# 功能：
#   离线训练不共享行为参数的风险网络，动作风险必须同时提供安全与危险提案。
# 输入：
#   samples：风险训练样本及独立风险标签。
#   config：网络初始化与优化超参数。
#   initial_model：可选风险基座，复制后再优化。
#   classification_margin：将标签推离 0.5 分界的最小间隔，范围零至小于 0.5。
# 输出：
#   model：训练后的独立风险网络。
#   metrics：本训练集风险指标，不授予飞行资格。
def train_local_risk_critic(
    samples: list[LocalPolicyTrainingSample],
    config: LocalPolicyTrainingConfig,
    *,
    initial_model: NumpyLocalPolicyModel | None = None,
    classification_margin: float = 0.0,
) -> tuple[NumpyLocalPolicyModel, LocalRiskCriticMetrics]:
    import numpy as np

    config = LocalPolicyTrainingConfig.model_validate(config.model_dump())
    if not 0.0 <= classification_margin < 0.5:
        raise ValueError("risk critic classification margin must be in [0, 0.5)")
    action_conditioned = any(
        sample.risk_proposed_control or sample.target_pilot_control for sample in samples
    )
    realtime_feature_count = len(samples[0].realtime_features) if samples else 0
    model = (
        copy_numpy_policy_model(initial_model)
        if initial_model is not None
        else _new_model(
            config,
            visual_feature_count=0,
            realtime_feature_count=realtime_feature_count,
            action_conditioned=action_conditioned,
        )
    )
    validate_numpy_policy_model(model)
    if model.action_conditioned != action_conditioned:
        raise ValueError("legacy and action-conditioned risk critics cannot be mixed")
    if model.realtime_feature_count != realtime_feature_count:
        raise ValueError("initial critic realtime feature contract differs from dataset")
    arrays = _arrays(
        samples,
        include_visual=model.visual_feature_count > 0,
        include_realtime=model.realtime_feature_count > 0,
        include_proposed_control=action_conditioned,
    )
    if action_conditioned and not (
        any(sample.risk_target < 0.5 for sample in samples)
        and any(sample.risk_target >= 0.5 for sample in samples)
    ):
        raise ValueError("action risk training requires safe and unsafe proposed actions")
    balanced_risk_weights = risk_class_balanced_sample_weights(
        arrays.risks,
        arrays.weights,
        config,
    )
    parameters = [
        model.input_weight,
        model.input_bias,
        model.risk_weight,
        model.risk_bias,
    ]
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    generator = np.random.default_rng(config.random_seed)
    step = 0
    for _epoch in range(config.epoch_count):
        order = generator.permutation(len(samples))
        for start in range(0, len(samples), config.batch_size):
            indices = order[start : start + config.batch_size]
            inputs = arrays.inputs[indices]
            targets = arrays.risks[indices]
            if classification_margin > 0.0:
                targets = np.where(
                    targets >= 0.5,
                    np.maximum(targets, 0.5 + classification_margin),
                    np.minimum(targets, 0.5 - classification_margin),
                )
            weights = balanced_risk_weights[indices]
            hidden_pre = inputs @ model.input_weight + model.input_bias
            hidden = np.maximum(hidden_pre, 0.0)
            logits = hidden @ model.risk_weight + model.risk_bias
            risk = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            normalizer = max(1e-9, float(weights.sum()))
            risk_delta = (risk - targets) * weights / normalizer
            risk_weight_gradient = hidden.T @ risk_delta + config.l2_weight * model.risk_weight
            risk_bias_gradient = risk_delta.sum(axis=0)
            hidden_delta = risk_delta @ model.risk_weight.T
            hidden_delta[hidden_pre <= 0.0] = 0.0
            input_weight_gradient = inputs.T @ hidden_delta + config.l2_weight * model.input_weight
            input_bias_gradient = hidden_delta.sum(axis=0)
            gradients = [
                input_weight_gradient,
                input_bias_gradient,
                risk_weight_gradient,
                risk_bias_gradient,
            ]
            step += 1
            _adam_update(
                parameters, gradients, first_moment, second_moment, step, config.learning_rate
            )
    metrics = evaluate_local_risk_critic(model, samples)
    return model, metrics


# 功能：
#   独立计算风险双类召回、绝对误差及交叉熵，缺少某类样本时该类召回记为零。
# 输入：
#   model：结构与数值可校验的风险网络。
#   samples：独立待评价数据，不更新权重。
# 输出：
#   metrics：包含双类支持数的风险评价指标。
def evaluate_local_risk_critic(
    model: NumpyLocalPolicyModel,
    samples: list[LocalPolicyTrainingSample],
) -> LocalRiskCriticMetrics:
    import numpy as np

    validate_numpy_policy_model(model)
    arrays = _arrays(
        samples,
        include_visual=model.visual_feature_count > 0,
        include_realtime=model.realtime_feature_count > 0,
        include_proposed_control=model.action_conditioned,
    )
    hidden = np.maximum(arrays.inputs @ model.input_weight + model.input_bias, 0.0)
    logits = hidden @ model.risk_weight + model.risk_bias
    risk = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    targets = arrays.risks.reshape(-1)
    risk_holds = risk.reshape(-1) >= 0.5
    target_risky = targets >= 0.5
    risky_count = int(target_risky.sum())
    safe_count = len(samples) - risky_count
    # float32 rounds 1 - 1e-9 back to exactly 1.0; use an epsilon that
    # remains representable so perfect classifications cannot produce log(0).
    probabilities = np.clip(risk.reshape(-1), 1e-7, 1.0 - 1e-7)
    binary_cross_entropy = -(
        targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    metrics = LocalRiskCriticMetrics(
        sample_count=len(samples),
        risky_sample_count=risky_count,
        safe_sample_count=safe_count,
        risk_hold_recall=(float(risk_holds[target_risky].mean()) if risky_count else 0.0),
        safe_motion_recall=(float((~risk_holds[~target_risky]).mean()) if safe_count else 0.0),
        risk_mean_absolute_error=float(np.abs(risk.reshape(-1) - targets).mean()),
        mean_binary_cross_entropy=float(binary_cross_entropy.mean()),
    )
    return metrics


# 功能：
#   按风险否决后的动作统计运动授权、候选选择与四轴误差，连续模式只在四种动作间判别。
# 输入：
#   model：待评价的行为策略，不允许使用动作风险网络替代。
#   samples：与模型输入输出契约一致的监督样本。
# 输出：
#   metrics：含类别支持数及控制轴证据的评价结果。
def evaluate_local_policy(
    model: NumpyLocalPolicyModel,
    samples: list[LocalPolicyTrainingSample],
) -> LocalPolicyTrainingMetrics:
    import numpy as np

    validate_numpy_policy_model(model)
    if model.action_conditioned:
        raise ValueError("risk critic cannot be evaluated as a behavior policy")
    arrays = _arrays(samples)
    if (
        (model.pilot_control_weight is not None) != (arrays.pilot_controls is not None)
        or model.visual_feature_count != len(samples[0].visual_features)
        or model.realtime_feature_count != len(samples[0].realtime_features)
    ):
        raise ValueError("evaluation policy contract differs from dataset")
    _hidden_pre, _hidden, logits, probabilities, risk, pilot_control = _forward(
        model, arrays.inputs, arrays.masks
    )
    model_has_pilot_control = (
        model.pilot_control_weight is not None and model.pilot_control_bias is not None
    )
    if model_has_pilot_control:
        action_logits = logits[:, LOCAL_POLICY_MAXIMUM_CANDIDATES:]
        shifted = action_logits - action_logits.max(axis=1, keepdims=True)
        scored_probabilities = np.exp(shifted)
        scored_probabilities /= scored_probabilities.sum(axis=1, keepdims=True)
        scored_targets = arrays.targets - LOCAL_POLICY_MAXIMUM_CANDIDATES
        selected_probabilities = scored_probabilities[np.arange(len(samples)), scored_targets]
        raw_predictions = scored_probabilities.argmax(axis=1) + LOCAL_POLICY_MAXIMUM_CANDIDATES
    else:
        selected_probabilities = probabilities[np.arange(len(samples)), arrays.targets]
        raw_predictions = probabilities.argmax(axis=1)
    cross_entropy = -np.log(np.clip(selected_probabilities, 1e-9, 1.0))
    risk_targets = arrays.risks.reshape(-1)
    risk_holds = risk.reshape(-1) >= 0.5
    target_risky = risk_targets >= 0.5
    target_motion = (arrays.targets < LOCAL_POLICY_MAXIMUM_CANDIDATES) | (
        arrays.targets == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    )
    effective_predictions = raw_predictions.copy()
    effective_predictions[risk_holds] = LOCAL_POLICY_MAXIMUM_CANDIDATES
    predicted_motion = (effective_predictions < LOCAL_POLICY_MAXIMUM_CANDIDATES) | (
        effective_predictions == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX
    )
    candidate_targets = arrays.targets < LOCAL_POLICY_MAXIMUM_CANDIDATES
    candidate_count = int(candidate_targets.sum())
    risky_count = int(target_risky.sum())
    safe_count = len(samples) - risky_count
    metrics = LocalPolicyTrainingMetrics(
        sample_count=len(samples),
        motion_sample_count=int(target_motion.sum()),
        non_motion_sample_count=int((~target_motion).sum()),
        authorized_motion_recall=(
            float(predicted_motion[target_motion].mean()) if target_motion.any() else 0.0
        ),
        non_motion_recall=(
            float((~predicted_motion[~target_motion]).mean()) if (~target_motion).any() else 0.0
        ),
        risky_sample_count=risky_count,
        safe_sample_count=safe_count,
        candidate_sample_count=candidate_count,
        action_accuracy=float((raw_predictions == arrays.targets).mean()),
        motion_authorization_accuracy=float((predicted_motion == target_motion).mean()),
        candidate_selection_accuracy=(
            float(
                (
                    effective_predictions[candidate_targets] == arrays.targets[candidate_targets]
                ).mean()
            )
            if candidate_count
            else 1.0
        ),
        # The action head owns hold/scan/abort intent.  The risk head is scored
        # against its independent continuous target so an uncertainty-driven
        # request-new-scan is not mislabeled as a high-risk observation.
        risk_hold_recall=(float(risk_holds[target_risky].mean()) if risky_count else 0.0),
        safe_motion_recall=(float((~risk_holds[~target_risky]).mean()) if safe_count else 0.0),
        risk_mean_absolute_error=float(np.abs(risk - arrays.risks).mean()),
        mean_cross_entropy=float(cross_entropy.mean()),
        pilot_control_mean_absolute_error=(
            float(np.abs(pilot_control - arrays.pilot_controls).mean())
            if pilot_control is not None and arrays.pilot_controls is not None
            else None
        ),
        pilot_axis_evidence=(
            summarize_pilot_axes(
                pilot_control[arrays.targets == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX],
                arrays.pilot_controls[arrays.targets == LOCAL_POLICY_PILOT_CONTROL_ACTION_INDEX],
            )
            if pilot_control is not None and arrays.pilot_controls is not None
            else {}
        ),
    )
    return metrics


# 功能：
#   将前馈策略及可选四轴头导出为内嵌权重 ONNX，完整校验后无覆盖发布。
# 输入：
#   model：输入、隐藏层和各输出头契约一致的有限 float32 模型。
#   output_path：尚不存在的目标模型文件。
# 输出：
#   digest：实际发布模型字节的 SHA-256。
def export_local_policy_onnx(model: NumpyLocalPolicyModel, output_path: Path) -> str:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from .training.evidence_publication import publish_evidence_bytes

    validate_numpy_policy_model(model)
    if model.action_conditioned:
        raise ValueError("action critic cannot be exported as a navigation policy")
    output_path = output_path.absolute()
    check_plain_plugin_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initializers = [
        numpy_helper.from_array(model.input_weight, "input_weight"),
        numpy_helper.from_array(model.input_bias, "input_bias"),
        numpy_helper.from_array(model.action_weight, "action_weight"),
        numpy_helper.from_array(model.action_bias, "action_bias"),
        numpy_helper.from_array(model.risk_weight, "risk_weight"),
        numpy_helper.from_array(model.risk_bias, "risk_bias"),
        numpy_helper.from_array(np.asarray([-1, 120], dtype=np.int64), "candidate_shape"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "slice_start_zero"),
        numpy_helper.from_array(np.asarray([8], dtype=np.int64), "slice_end_candidates"),
        numpy_helper.from_array(
            np.asarray([model.action_weight.shape[1]], dtype=np.int64),
            "slice_end_actions",
        ),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "slice_axis_one"),
        numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "mask_one"),
        numpy_helper.from_array(np.asarray([-10_000.0], dtype=np.float32), "mask_penalty"),
    ]
    pilot_control_enabled = (
        model.pilot_control_weight is not None and model.pilot_control_bias is not None
    )
    if (model.pilot_control_weight is None) != (model.pilot_control_bias is None):
        raise ValueError("local policy pilot-control head is incomplete")
    if pilot_control_enabled:
        initializers.extend(
            (
                numpy_helper.from_array(
                    model.pilot_control_weight,
                    "pilot_control_weight",
                ),
                numpy_helper.from_array(
                    model.pilot_control_bias,
                    "pilot_control_bias",
                ),
            )
        )
    flat_inputs = ["state_features", "candidate_flat", "candidate_mask"]
    graph_inputs = [
        helper.make_tensor_value_info(
            "state_features",
            TensorProto.FLOAT,
            [None, LOCAL_POLICY_STATE_FEATURE_COUNT],
        ),
        helper.make_tensor_value_info(
            "candidate_features",
            TensorProto.FLOAT,
            [
                None,
                LOCAL_POLICY_MAXIMUM_CANDIDATES,
                LOCAL_POLICY_CANDIDATE_FEATURE_COUNT,
            ],
        ),
        helper.make_tensor_value_info(
            "candidate_mask",
            TensorProto.FLOAT,
            [None, LOCAL_POLICY_MAXIMUM_CANDIDATES],
        ),
    ]
    if model.realtime_feature_count > 0:
        if model.realtime_feature_count not in {
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("local policy realtime feature count is incompatible")
        flat_inputs.extend(("realtime_features", "realtime_valid_mask"))
        graph_inputs.extend(
            (
                helper.make_tensor_value_info(
                    "realtime_features",
                    TensorProto.FLOAT,
                    [None, model.realtime_feature_count],
                ),
                helper.make_tensor_value_info(
                    "realtime_valid_mask",
                    TensorProto.FLOAT,
                    [None, model.realtime_feature_count],
                ),
            )
        )
    if model.visual_feature_count > 0:
        flat_inputs.append("visual_features")
        graph_inputs.append(
            helper.make_tensor_value_info(
                "visual_features",
                TensorProto.FLOAT,
                [None, model.visual_feature_count],
            )
        )
    nodes = [
        helper.make_node("Reshape", ["candidate_features", "candidate_shape"], ["candidate_flat"]),
        helper.make_node(
            "Concat",
            flat_inputs,
            ["flat_features"],
            axis=1,
        ),
        helper.make_node("MatMul", ["flat_features", "input_weight"], ["hidden_linear"]),
        helper.make_node("Add", ["hidden_linear", "input_bias"], ["hidden_biased"]),
        helper.make_node("Relu", ["hidden_biased"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "action_weight"], ["all_scores_linear"]),
        helper.make_node("Add", ["all_scores_linear", "action_bias"], ["all_scores"]),
        helper.make_node(
            "Slice",
            [
                "all_scores",
                "slice_start_zero",
                "slice_end_candidates",
                "slice_axis_one",
            ],
            ["candidate_scores_raw"],
        ),
        helper.make_node("Sub", ["mask_one", "candidate_mask"], ["inverse_mask"]),
        helper.make_node("Mul", ["inverse_mask", "mask_penalty"], ["masked_penalty"]),
        helper.make_node("Add", ["candidate_scores_raw", "masked_penalty"], ["candidate_scores"]),
        helper.make_node(
            "Slice",
            [
                "all_scores",
                "slice_end_candidates",
                "slice_end_actions",
                "slice_axis_one",
            ],
            ["action_scores"],
        ),
        helper.make_node("MatMul", ["hidden", "risk_weight"], ["risk_linear"]),
        helper.make_node("Add", ["risk_linear", "risk_bias"], ["risk_logit"]),
        helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info(
            "candidate_scores",
            TensorProto.FLOAT,
            [None, LOCAL_POLICY_MAXIMUM_CANDIDATES],
        ),
        helper.make_tensor_value_info(
            "action_scores",
            TensorProto.FLOAT,
            [None, model.action_weight.shape[1] - LOCAL_POLICY_MAXIMUM_CANDIDATES],
        ),
        helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1]),
    ]
    if pilot_control_enabled:
        nodes.extend(
            (
                helper.make_node(
                    "MatMul",
                    ["hidden", "pilot_control_weight"],
                    ["pilot_control_linear"],
                ),
                helper.make_node(
                    "Add",
                    ["pilot_control_linear", "pilot_control_bias"],
                    ["pilot_control_pre_activation"],
                ),
                helper.make_node(
                    "Tanh",
                    ["pilot_control_pre_activation"],
                    ["pilot_control"],
                ),
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                "pilot_control",
                TensorProto.FLOAT,
                [None, LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT],
            )
        )
    graph = helper.make_graph(
        nodes,
        "dronedream-local-navigation-policy",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream local policy trainer",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    content = document.SerializeToString()
    publish_evidence_bytes(output_path, content, limit=256 * 1024 * 1024)
    digest = hashlib.sha256(content).hexdigest()
    return digest


# 功能：
#   导出风险头独立执行图；动作条件模式使用被评估控制，不读取旧候选表。
# 输入：
#   model：契约完整的风险网络。
#   output_path：尚不存在的 ONNX 目标路径。
# 输出：
#   digest：无覆盖发布的实际模型字节摘要。
def export_local_risk_critic_onnx(model: NumpyLocalPolicyModel, output_path: Path) -> str:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from .training.evidence_publication import publish_evidence_bytes

    validate_numpy_policy_model(model)
    output_path = output_path.absolute()
    check_plain_plugin_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initializers = [
        numpy_helper.from_array(model.input_weight, "input_weight"),
        numpy_helper.from_array(model.input_bias, "input_bias"),
        numpy_helper.from_array(model.risk_weight, "risk_weight"),
        numpy_helper.from_array(model.risk_bias, "risk_bias"),
        numpy_helper.from_array(np.asarray([-1, 120], dtype=np.int64), "candidate_shape"),
    ]
    flat_inputs = ["state_features", "candidate_flat", "candidate_mask"]
    graph_inputs = [
        helper.make_tensor_value_info(
            "state_features",
            TensorProto.FLOAT,
            [None, LOCAL_POLICY_STATE_FEATURE_COUNT],
        ),
        helper.make_tensor_value_info(
            "candidate_features",
            TensorProto.FLOAT,
            [
                None,
                LOCAL_POLICY_MAXIMUM_CANDIDATES,
                LOCAL_POLICY_CANDIDATE_FEATURE_COUNT,
            ],
        ),
        helper.make_tensor_value_info(
            "candidate_mask",
            TensorProto.FLOAT,
            [None, LOCAL_POLICY_MAXIMUM_CANDIDATES],
        ),
    ]
    if model.action_conditioned:
        if model.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT:
            raise ValueError("action critic requires the current realtime control contract")
        flat_inputs = ["state_features"]
        graph_inputs = graph_inputs[:1]
        initializers = [item for item in initializers if item.name != "candidate_shape"]
    if model.realtime_feature_count > 0:
        if model.realtime_feature_count not in {
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("local risk critic realtime feature count is incompatible")
        flat_inputs.extend(("realtime_features", "realtime_valid_mask"))
        graph_inputs.extend(
            (
                helper.make_tensor_value_info(
                    "realtime_features",
                    TensorProto.FLOAT,
                    [None, model.realtime_feature_count],
                ),
                helper.make_tensor_value_info(
                    "realtime_valid_mask",
                    TensorProto.FLOAT,
                    [None, model.realtime_feature_count],
                ),
            )
        )
    if model.visual_feature_count > 0:
        flat_inputs.append("visual_features")
        graph_inputs.append(
            helper.make_tensor_value_info(
                "visual_features",
                TensorProto.FLOAT,
                [None, model.visual_feature_count],
            )
        )
    if model.action_conditioned:
        flat_inputs.append("proposed_control")
        graph_inputs.append(
            helper.make_tensor_value_info(
                "proposed_control",
                TensorProto.FLOAT,
                [None, ACTION_RISK_FEATURE_COUNT],
            )
        )
    nodes = [
        *(
            []
            if model.action_conditioned
            else [
                helper.make_node(
                    "Reshape",
                    ["candidate_features", "candidate_shape"],
                    ["candidate_flat"],
                )
            ]
        ),
        helper.make_node(
            "Concat",
            flat_inputs,
            ["flat_features"],
            axis=1,
        ),
        helper.make_node("MatMul", ["flat_features", "input_weight"], ["hidden_linear"]),
        helper.make_node("Add", ["hidden_linear", "input_bias"], ["hidden_biased"]),
        helper.make_node("Relu", ["hidden_biased"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "risk_weight"], ["risk_linear"]),
        helper.make_node("Add", ["risk_linear", "risk_bias"], ["risk_logit"]),
        helper.make_node("Sigmoid", ["risk_logit"], ["risk_score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dronedream-local-risk-critic",
        graph_inputs,
        [helper.make_tensor_value_info("risk_score", TensorProto.FLOAT, [None, 1])],
        initializer=initializers,
    )
    document = helper.make_model(
        graph,
        producer_name="DroneDream local risk critic trainer",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    document.ir_version = min(document.ir_version, 11)
    onnx.checker.check_model(document)
    content = document.SerializeToString()
    publish_evidence_bytes(output_path, content, limit=256 * 1024 * 1024)
    digest = hashlib.sha256(content).hexdigest()
    return digest


# 功能：
#   固定普通文件字节后检查内嵌 ONNX 图，拒绝外部权重及不唯一的接口名称。
# 输入：
#   path：至多 256 MiB 的 ONNX 文件。
# 输出：
#   content：同次读取并校验的原始字节。
#   document：通过图级检查的 ONNX 对象。
def _read_embedded_graph_file(path: Path):
    import onnx

    from .training.artifact_assembly import validate_embedded_graph

    if path.suffix.lower() != ".onnx":
        raise ValueError("model artifact must be an ONNX file")
    content = read_plugin_file(path, limit=256 * 1024 * 1024)
    document = onnx.load_model_from_string(content)
    validate_embedded_graph(
        content,
        input_names=[item.name for item in document.graph.input],
        output_names=[item.name for item in document.graph.output],
    )
    return content, document


# 功能：
#   将已检查的内嵌 ONNX 原始字节发布到新暂存文件，不重新打开来源复制不同内容。
# 输入：
#   source：明确指定的源模型文件。
#   destination：尚不存在的暂存文件。
# 输出：
#   document：源字节对应的图，供调用方检查专家接口。
#   digest：实际暂存字节的摘要。
def _copy_embedded_graph(source: Path, destination: Path):
    from .training.evidence_publication import publish_evidence_bytes

    content, document = _read_embedded_graph_file(source)
    publish_evidence_bytes(destination, content, limit=256 * 1024 * 1024)
    digest = hashlib.sha256(content).hexdigest()
    return document, digest


# 功能：
#   从已检查的内嵌图复制具名初始化参数，先验证骨干秩再推导网络宽度。
# 输入：
#   path：至多 256 MiB 的前馈网络文件。
# 输出：
#   document：内嵌 ONNX 图。
#   values：独立复制的具名参数数组。
def _read_numpy_graph(path: Path):
    from onnx import numpy_helper

    _, document = _read_embedded_graph_file(path)
    values = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in document.graph.initializer
    }
    if (
        "input_weight" not in values
        or values["input_weight"].ndim != 2
        or "input_bias" not in values
        or values["input_bias"].ndim != 1
    ):
        raise ValueError("NUMPY_POLICY_GRAPH_BACKBONE_INVALID")
    return document, values


# 功能：
#   对照实际权重核验全部具名输入输出的秩、float32 类型和静态特征宽度。
# 输入：
#   document：已检查且权重内嵌的 ONNX 图。
#   model：由图参数恢复并通过数值校验的前馈网络。
#   critic：是否只要求独立风险输出契约。
# 输出：
#   None：不返回业务数据。
def _validate_numpy_graph_io(document, model, *, critic):
    import onnx

    inputs = {"state_features": (LOCAL_POLICY_STATE_FEATURE_COUNT,)}
    if not model.action_conditioned:
        inputs.update(
            candidate_features=(
                LOCAL_POLICY_MAXIMUM_CANDIDATES,
                LOCAL_POLICY_CANDIDATE_FEATURE_COUNT,
            ),
            candidate_mask=(LOCAL_POLICY_MAXIMUM_CANDIDATES,),
        )
    if model.realtime_feature_count:
        inputs.update(
            realtime_features=(model.realtime_feature_count,),
            realtime_valid_mask=(model.realtime_feature_count,),
        )
    if model.visual_feature_count:
        inputs["visual_features"] = (model.visual_feature_count,)
    if model.action_conditioned:
        inputs["proposed_control"] = (ACTION_RISK_FEATURE_COUNT,)
    outputs = {"risk_score": (1,)}
    if not critic:
        outputs.update(
            candidate_scores=(LOCAL_POLICY_MAXIMUM_CANDIDATES,),
            action_scores=(model.action_bias.size - LOCAL_POLICY_MAXIMUM_CANDIDATES,),
        )
        if model.pilot_control_weight is not None:
            outputs["pilot_control"] = (LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,)
    for tensors, expected in ((document.graph.input, inputs), (document.graph.output, outputs)):
        if {item.name for item in tensors} != set(expected):
            raise ValueError("NUMPY_POLICY_GRAPH_TENSORS_INVALID")
        for item in tensors:
            tensor = item.type.tensor_type
            dims = tensor.shape.dim
            if (
                tensor.elem_type != onnx.TensorProto.FLOAT
                or len(dims) != len(expected[item.name]) + 1
                or (dims[0].HasField("dim_value") and dims[0].dim_value <= 0)
                or tuple(dim.dim_value for dim in dims[1:]) != expected[item.name]
            ):
                raise ValueError("NUMPY_POLICY_GRAPH_TENSOR_SHAPE_INVALID:" + item.name)
    onnx.checker.check_model(document, full_check=True)


# 功能：
#   从前馈行为 ONNX 恢复训练参数，核验全部头和输入输出形状，不把图加载成功当成准入。
# 输入：
#   path：明确指定的前馈行为模型路径。
# 输出：
#   model：数值、形状及图接口一致的模型副本。
def load_local_policy_onnx(path: Path) -> NumpyLocalPolicyModel:
    document, values = _read_numpy_graph(path)
    required = {
        "input_weight",
        "input_bias",
        "action_weight",
        "action_bias",
        "risk_weight",
        "risk_bias",
    }
    if not required.issubset(values):
        raise ValueError("local policy ONNX model is missing trainer weights")
    pilot_initializer_names = {"pilot_control_weight", "pilot_control_bias"}
    pilot_initializers_present = pilot_initializer_names.intersection(values)
    if pilot_initializers_present and pilot_initializers_present != pilot_initializer_names:
        raise ValueError("local policy pilot-control weights are incomplete")
    graph_outputs = {item.name for item in document.graph.output}
    pilot_control_enabled = pilot_initializers_present == pilot_initializer_names
    if pilot_control_enabled != ("pilot_control" in graph_outputs):
        raise ValueError("local policy pilot-control tensor contract is incompatible")
    graph_inputs = {item.name for item in document.graph.input}
    realtime_input_names = {"realtime_features", "realtime_valid_mask"}
    if bool(realtime_input_names.intersection(graph_inputs)) != realtime_input_names.issubset(
        graph_inputs
    ):
        raise ValueError("local policy realtime tensor contract is incomplete")
    realtime_feature_count = 0
    if realtime_input_names.issubset(graph_inputs):
        realtime_input = next(
            item for item in document.graph.input if item.name == "realtime_features"
        )
        dimensions = realtime_input.type.tensor_type.shape.dim
        if len(dimensions) != 2 or not dimensions[1].HasField("dim_value"):
            raise ValueError("local policy realtime feature width must be static")
        realtime_feature_count = int(dimensions[1].dim_value)
        if realtime_feature_count not in {
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("local policy realtime feature width is incompatible")
    visual_feature_count = max(
        0,
        int(values["input_weight"].shape[0])
        - LOCAL_POLICY_FLAT_FEATURE_COUNT
        - 2 * realtime_feature_count,
    )
    model = NumpyLocalPolicyModel(
        input_weight=values["input_weight"],
        input_bias=values["input_bias"],
        action_weight=values["action_weight"],
        action_bias=values["action_bias"],
        risk_weight=values["risk_weight"],
        risk_bias=values["risk_bias"],
        visual_feature_count=visual_feature_count,
        realtime_feature_count=realtime_feature_count,
        pilot_control_weight=(values["pilot_control_weight"] if pilot_control_enabled else None),
        pilot_control_bias=(values["pilot_control_bias"] if pilot_control_enabled else None),
    )
    if (model.visual_feature_count > 0) != ("visual_features" in graph_inputs):
        raise ValueError("local policy visual tensor contract is incompatible")
    if model.input_weight.shape[0] != (
        LOCAL_POLICY_FLAT_FEATURE_COUNT
        + 2 * model.realtime_feature_count
        + model.visual_feature_count
    ):
        raise ValueError("local policy ONNX input feature count is incompatible")
    validate_numpy_policy_model(model)
    expected_action_count = (
        LOCAL_POLICY_ACTION_COUNT if pilot_control_enabled else LOCAL_POLICY_LEGACY_ACTION_COUNT
    )
    if model.action_weight.shape[1] != expected_action_count:
        raise ValueError("local policy ONNX action count is incompatible")
    if pilot_control_enabled and (
        model.pilot_control_weight.shape
        != (int(model.input_bias.shape[0]), LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT)
        or model.pilot_control_bias.shape != (LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,)
    ):
        raise ValueError("local policy pilot-control output shape is incompatible")
    _validate_numpy_graph_io(document, model, critic=False)
    return model


# 功能：
#   恢复独立风险网络，保留动作条件标识并拒绝外部权重、非有限参数和不一致接口。
# 输入：
#   path：明确指定的前馈风险模型文件。
# 输出：
#   model：可供离线精调的独立参数副本。
def load_local_risk_critic_onnx(path: Path) -> NumpyLocalPolicyModel:
    import numpy as np

    document, values = _read_numpy_graph(path)
    required = {"input_weight", "input_bias", "risk_weight", "risk_bias"}
    if not required.issubset(values):
        raise ValueError("local risk critic ONNX model is missing trainer weights")
    hidden_count = int(values["input_bias"].shape[0])
    graph_inputs = {item.name for item in document.graph.input}
    action_conditioned = "proposed_control" in graph_inputs
    base_width = (
        LOCAL_POLICY_STATE_FEATURE_COUNT + ACTION_RISK_FEATURE_COUNT
        if action_conditioned
        else LOCAL_POLICY_FLAT_FEATURE_COUNT
    )
    if action_conditioned:
        proposed = next(item for item in document.graph.input if item.name == "proposed_control")
        shape = proposed.type.tensor_type.shape.dim
        if len(shape) != 2 or shape[1].dim_value != ACTION_RISK_FEATURE_COUNT:
            raise ValueError("action risk critic proposed-control shape is incompatible")
        if {"candidate_features", "candidate_mask"}.intersection(graph_inputs):
            raise ValueError("action risk critic cannot consume legacy candidates")
    realtime_input_names = {"realtime_features", "realtime_valid_mask"}
    if bool(realtime_input_names.intersection(graph_inputs)) != realtime_input_names.issubset(
        graph_inputs
    ):
        raise ValueError("local risk critic realtime tensor contract is incomplete")
    realtime_feature_count = 0
    if realtime_input_names.issubset(graph_inputs):
        realtime_input = next(
            item for item in document.graph.input if item.name == "realtime_features"
        )
        dimensions = realtime_input.type.tensor_type.shape.dim
        if len(dimensions) != 2 or not dimensions[1].HasField("dim_value"):
            raise ValueError("local risk critic realtime feature width must be static")
        realtime_feature_count = int(dimensions[1].dim_value)
        if realtime_feature_count not in {
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("local risk critic realtime feature width is incompatible")
    visual_feature_count = max(
        0,
        int(values["input_weight"].shape[0]) - base_width - 2 * realtime_feature_count,
    )
    model = NumpyLocalPolicyModel(
        input_weight=values["input_weight"],
        input_bias=values["input_bias"],
        action_weight=np.zeros((hidden_count, LOCAL_POLICY_LEGACY_ACTION_COUNT), dtype=np.float32),
        action_bias=np.zeros(LOCAL_POLICY_LEGACY_ACTION_COUNT, dtype=np.float32),
        risk_weight=values["risk_weight"],
        risk_bias=values["risk_bias"],
        visual_feature_count=visual_feature_count,
        realtime_feature_count=realtime_feature_count,
        action_conditioned=action_conditioned,
    )
    if (model.visual_feature_count > 0) != ("visual_features" in graph_inputs):
        raise ValueError("local risk critic visual tensor contract is incompatible")
    if model.input_weight.shape != (
        base_width + 2 * model.realtime_feature_count + model.visual_feature_count,
        hidden_count,
    ):
        raise ValueError("local risk critic input feature count is incompatible")
    if model.risk_weight.shape != (hidden_count, 1):
        raise ValueError("local risk critic output shape is incompatible")
    validate_numpy_policy_model(model)
    _validate_numpy_graph_io(document, model, critic=True)
    return model


# 功能：
#   1. 组装明确指定的前馈策略、专家和感知组件，各复制件使用实际校验的同一份字节。
#   2. 暂存包通过清单和摘要检查后无覆盖发布，不修改当前选择或自动授予飞行资格。
# 输入：
#   output_root：尚不存在的候选包目录。
#   model：前馈导航策略。
#   package_id：包标识。
#   display_name：显示名称。
#   scope：通用或指定地图作用域。
#   vehicle_sha256：车辆契约摘要。
#   sensor_contract_sha256：传感器契约摘要。
#   maximum_inference_latency_ms：声明的推理时限，仍需独立测量。
#   risk_model：独立风险网络及其输入语义。
#   risk_critic_path：可选保留的风险模型原始文件。
#   include_independent_risk_critic：是否包含独立风险专家。
#   precision_model：可选精细操纵策略。
#   recovery_model：可选恢复策略。
#   state_anomaly_detector_path：状态历史异常检测图。
#   perception_health_critic_path：感知健康检查图。
#   settle_stability_critic_path：局部稳定性检查图。
#   payload_dynamics_adapter_path：负载动力学适配图。
#   cross_modal_consistency_critic_path：跨模态一致性图。
#   perception_encoder_path：前向 RGB 编码器图。
#   visual_width：图像宽度。
#   visual_height：图像高度。
#   visual_feature_count：视觉编码维数。
#   visual_normalization：图像归一化方法。
#   base_package_sha256：地图专用包的基座摘要。
#   map_sha256：可选的绑定地图摘要。
#   control_feature_contract_sha256：连续控制特征的明确语义摘要。
# 输出：
#   manifest：已发布候选包的清单。
def write_local_policy_package(
    *,
    output_root: Path,
    model: NumpyLocalPolicyModel,
    package_id: str,
    display_name: str,
    scope: Literal["general", "map-specialist"],
    vehicle_sha256: str,
    sensor_contract_sha256: str,
    maximum_inference_latency_ms: int,
    risk_model: NumpyLocalPolicyModel | None = None,
    risk_critic_path: Path | None = None,
    include_independent_risk_critic: bool = True,
    precision_model: NumpyLocalPolicyModel | None = None,
    recovery_model: NumpyLocalPolicyModel | None = None,
    state_anomaly_detector_path: Path | None = None,
    perception_health_critic_path: Path | None = None,
    settle_stability_critic_path: Path | None = None,
    payload_dynamics_adapter_path: Path | None = None,
    cross_modal_consistency_critic_path: Path | None = None,
    perception_encoder_path: Path | None = None,
    visual_width: int | None = None,
    visual_height: int | None = None,
    visual_feature_count: int | None = None,
    visual_normalization: Literal[
        "zero-to-one",
        "minus-one-to-one",
        "imagenet",
    ] = "zero-to-one",
    base_package_sha256: str | None = None,
    map_sha256: str | None = None,
    control_feature_contract_sha256: str | None = None,
) -> LocalPolicyPackageManifest:
    output_root = output_root.absolute()
    check_plain_plugin_path(output_root)
    if model.pilot_control_weight is not None and (
        control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        raise ValueError("continuous control export requires explicit matching feature semantics")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=output_root.parent,
        prefix=f".{output_root.name}.staging-",
    ) as temporary_root:
        staging_root = Path(temporary_root) / "package"
        staging_root.mkdir()
        artifact_path = staging_root / "local-navigation.onnx"
        artifact_sha256 = export_local_policy_onnx(model, artifact_path)
        risk_path = staging_root / "risk-critic.onnx"
        if not include_independent_risk_critic and (
            risk_model is not None or risk_critic_path is not None
        ):
            raise ValueError("disabled independent risk critic cannot receive an artifact")
        if (
            include_independent_risk_critic
            and risk_model is None
            and model.visual_feature_count > 0
        ):
            raise ValueError("visual navigation requires an independent risk model")
        if include_independent_risk_critic and risk_critic_path is not None and risk_model is None:
            raise ValueError("a preserved risk critic requires its validated model contract")
        if risk_model is not None and risk_model.visual_feature_count not in {
            0,
            model.visual_feature_count,
        }:
            raise ValueError("risk critic and navigation visual contracts differ")
        if risk_model is not None and risk_model.realtime_feature_count not in {
            0,
            model.realtime_feature_count,
        }:
            raise ValueError("risk critic and navigation realtime contracts differ")
        pilot_control_enabled = (
            model.pilot_control_weight is not None and model.pilot_control_bias is not None
        )
        if (
            include_independent_risk_critic
            and pilot_control_enabled
            and (risk_model is None or not risk_model.action_conditioned)
        ):
            raise ValueError("continuous control requires an independent action-conditioned critic")
        if risk_model is not None and risk_model.action_conditioned != pilot_control_enabled:
            raise ValueError("navigation and risk action contracts differ")
        risk_sha256 = None
        if include_independent_risk_critic and risk_critic_path is None:
            risk_sha256 = export_local_risk_critic_onnx(risk_model or model, risk_path)
        elif include_independent_risk_critic:
            assert risk_critic_path is not None
            _, risk_sha256 = _copy_embedded_graph(risk_critic_path, risk_path)
            observed_risk = load_local_risk_critic_onnx(risk_path)
            expected_risk = risk_model or model
            if (
                observed_risk.visual_feature_count != expected_risk.visual_feature_count
                or observed_risk.action_conditioned != expected_risk.action_conditioned
                or observed_risk.realtime_feature_count != expected_risk.realtime_feature_count
                or observed_risk.input_weight.shape != expected_risk.input_weight.shape
            ):
                raise ValueError("preserved risk critic tensor contract differs")
        navigation_input_names = [
            "state_features",
            "candidate_features",
            "candidate_mask",
            *(
                ("realtime_features", "realtime_valid_mask")
                if model.realtime_feature_count > 0
                else ()
            ),
            *(("visual_features",) if model.visual_feature_count > 0 else ()),
        ]
        if model.visual_feature_count != (visual_feature_count or 0):
            raise ValueError("navigation model and visual feature contract differ")
        if (perception_encoder_path is None) != (model.visual_feature_count == 0):
            raise ValueError("visual navigation requires exactly one perception encoder")
        artifacts = [
            LocalPolicyArtifact(
                role="local-navigation-policy",
                relative_path=artifact_path.name,
                sha256=artifact_sha256,
                input_names=navigation_input_names,
                output_names=[
                    "candidate_scores",
                    "action_scores",
                    "risk_score",
                    *(("pilot_control",) if pilot_control_enabled else ()),
                ],
            )
        ]
        if include_independent_risk_critic:
            assert risk_sha256 is not None
            artifacts.append(
                LocalPolicyArtifact(
                    role="risk-critic",
                    relative_path=risk_path.name,
                    sha256=risk_sha256,
                    input_names=[
                        "state_features",
                        *(
                            []
                            if pilot_control_enabled
                            else ["candidate_features", "candidate_mask"]
                        ),
                        *(
                            ("realtime_features", "realtime_valid_mask")
                            if (risk_model or model).realtime_feature_count > 0
                            else ()
                        ),
                        *(
                            ("visual_features",)
                            if (risk_model or model).visual_feature_count > 0
                            else ()
                        ),
                        *(["proposed_control"] if pilot_control_enabled else []),
                    ],
                    output_names=["risk_score"],
                )
            )
        for expert_role, expert_model, filename in (
            ("precision-maneuver-policy", precision_model, "precision-maneuver.onnx"),
            ("recovery-policy", recovery_model, "recovery-policy.onnx"),
        ):
            if expert_model is None:
                continue
            if expert_model.visual_feature_count != model.visual_feature_count:
                raise ValueError("navigation experts have different visual contracts")
            if expert_model.realtime_feature_count != model.realtime_feature_count:
                raise ValueError("navigation experts have different realtime contracts")
            expert_has_pilot_control = (
                expert_model.pilot_control_weight is not None
                and expert_model.pilot_control_bias is not None
            )
            if expert_has_pilot_control != pilot_control_enabled:
                raise ValueError("navigation experts have different pilot-control contracts")
            expert_path = staging_root / filename
            expert_sha256 = export_local_policy_onnx(expert_model, expert_path)
            artifacts.append(
                LocalPolicyArtifact(
                    role=expert_role,
                    relative_path=expert_path.name,
                    sha256=expert_sha256,
                    input_names=navigation_input_names,
                    output_names=[
                        "candidate_scores",
                        "action_scores",
                        "risk_score",
                        *(("pilot_control",) if pilot_control_enabled else ()),
                    ],
                )
            )
        if perception_encoder_path is not None:
            perception_path = staging_root / "perception-encoder.onnx"
            document, perception_sha256 = _copy_embedded_graph(
                perception_encoder_path, perception_path
            )
            observed_inputs = {item.name for item in document.graph.input}
            observed_outputs = {item.name for item in document.graph.output}
            if observed_inputs != {"forward_rgb"} or "visual_features" not in observed_outputs:
                raise ValueError("perception encoder tensor contract is incompatible")
            artifacts.append(
                LocalPolicyArtifact(
                    role="perception-encoder",
                    relative_path=perception_path.name,
                    sha256=perception_sha256,
                    input_names=["forward_rgb"],
                    output_names=sorted(observed_outputs),
                )
            )
        if state_anomaly_detector_path is not None:
            anomaly_path = staging_root / "state-anomaly-detector.onnx"
            document, anomaly_sha256 = _copy_embedded_graph(
                state_anomaly_detector_path, anomaly_path
            )
            if {item.name for item in document.graph.input} != {
                "state_history",
                "history_mask",
            } or {item.name for item in document.graph.output} != {"anomaly_score"}:
                raise ValueError("state anomaly detector tensor contract is incompatible")
            artifacts.append(
                LocalPolicyArtifact(
                    role="state-anomaly-detector",
                    relative_path=anomaly_path.name,
                    sha256=anomaly_sha256,
                    input_names=["state_history", "history_mask"],
                    output_names=["anomaly_score"],
                )
            )
        for (
            advisor_role,
            source_path,
            filename,
            required_inputs,
            required_outputs,
        ) in (
            (
                "perception-health-critic",
                perception_health_critic_path,
                "perception-health-critic.onnx",
                {"state_features"},
                {"risk_score"},
            ),
            (
                "settle-stability-critic",
                settle_stability_critic_path,
                "settle-stability-critic.onnx",
                {"maneuver_features", "state_history", "history_mask"},
                {"risk_score"},
            ),
            (
                "payload-dynamics-adapter",
                payload_dynamics_adapter_path,
                "payload-dynamics-adapter.onnx",
                (
                    {"payload_features", "state_history", "history_mask"},
                    {"payload_features", "payload_history", "history_mask"},
                ),
                {"risk_score", "controller_step_scale"},
            ),
            (
                "cross-modal-consistency-critic",
                cross_modal_consistency_critic_path,
                "cross-modal-consistency-critic.onnx",
                {"sensor_features"},
                {"risk_score"},
            ),
        ):
            if source_path is None:
                continue
            advisor_path = staging_root / filename
            document, advisor_sha256 = _copy_embedded_graph(source_path, advisor_path)
            observed_inputs = {item.name for item in document.graph.input}
            accepted_inputs = (
                required_inputs if isinstance(required_inputs, tuple) else (required_inputs,)
            )
            if (
                observed_inputs not in accepted_inputs
                or {item.name for item in document.graph.output} != required_outputs
            ):
                raise ValueError(f"{advisor_role} tensor contract is incompatible")
            artifacts.append(
                LocalPolicyArtifact(
                    role=advisor_role,
                    relative_path=advisor_path.name,
                    sha256=advisor_sha256,
                    input_names=sorted(observed_inputs),
                    output_names=sorted(required_outputs),
                )
            )
        manifest = LocalPolicyPackageManifest(
            package_id=package_id,
            display_name=display_name,
            scope=scope,
            base_package_sha256=base_package_sha256,
            map_sha256=map_sha256,
            vehicle_sha256=vehicle_sha256,
            sensor_contract_sha256=sensor_contract_sha256,
            control_feature_contract_sha256=control_feature_contract_sha256,
            maximum_inference_latency_ms=maximum_inference_latency_ms,
            visual_width=visual_width,
            visual_height=visual_height,
            visual_feature_count=visual_feature_count,
            realtime_feature_count=(
                model.realtime_feature_count if model.realtime_feature_count > 0 else None
            ),
            pilot_control_mode=("normalized-body-velocity" if pilot_control_enabled else None),
            visual_normalization=visual_normalization,
            artifacts=artifacts,
        )
        (staging_root / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        # Re-open and hash-check every staged artifact before publishing the
        # directory name that selectors are allowed to consume.
        load_local_policy_package(staging_root)
        publish_asset_directory(staging_root, output_root)
    return manifest


# 功能：
#   解析有界样本字节，拒绝重复字段和空白行，保留合法末行没有换行符的输入兼容。
# 输入：
#   content：同次读取且至多 256 MiB 的不可变 JSONL 字节。
# 输出：
#   samples：最多二十五万条、单行至多 4 MiB 的已校验训练样本。
def parse_training_samples(content: bytes) -> list[LocalPolicyTrainingSample]:
    if type(content) is not bytes or len(content) > 256 * 1024 * 1024:
        raise ValueError("LOCAL_POLICY_DATASET_BYTES_INVALID")
    samples = []
    with BytesIO(content) as stream:
        while line := stream.readline(4 * 1024 * 1024 + 1):
            if len(samples) >= 250_000 or len(line) > 4 * 1024 * 1024 or not line.strip():
                raise ValueError("LOCAL_POLICY_DATASET_ROW_INVALID")
            row = decode_json(line, limit=4 * 1024 * 1024)
            samples.append(LocalPolicyTrainingSample.model_validate(row))
    if not samples:
        raise ValueError("local policy training dataset is empty")
    return samples


# 功能：
#   通过普通文件读取器固定一次有界字节，再交给共享严格解析器，不静默跳过损坏行。
# 输入：
#   path：训练或评价样本文件路径。
# 输出：
#   samples：实际读取内容对应的已验证样本列表。
def load_training_samples(path: Path) -> list[LocalPolicyTrainingSample]:
    content = read_plugin_file(path, limit=256 * 1024 * 1024)
    samples = parse_training_samples(content)
    return samples
