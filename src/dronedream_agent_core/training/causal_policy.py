"""Trainable recurrent local pilot; Torch is offline-only, ONNX is deployed.

The three manoeuvre roles use this common architecture, separate trained
weights, and the same axis contract. A GRU reads independent prior sensor
features; a nonlinear head combines its state with current observations and
optional visual features. It emits control magnitudes, not coordinates.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydantic import Field
from torch import nn

from ..causal_control import (
    CONTROL_HISTORY_CONTRACT_SHA256,
    CONTROL_HISTORY_LENGTH,
    CONTROL_HISTORY_WIDTH,
    CONTROL_REALTIME_WIDTH,
    CONTROL_STATE_WIDTH,
    CausalControlHistory,
    control_history_row,
)
from ..contracts import StrictModel
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from ..local_policy_training import (
    LocalPolicyObservation,
    LocalPolicyTrainingSample,
    require_behavior_supervision,
)
from ..pilot_control_mapping import PilotControlLimits
from .causal_device import causal_device_evidence, causal_training_device
from .causal_regularization import (
    CausalRegularization,
    drop_visual_block,
    effective_sample_weights,
    validate_regularization,
)
from .flight_environment import MODES


class CausalPolicyConfig(StrictModel):
    """Bound recurrent architecture and offline training sizes in exported model identity."""
    history_length: int = Field(default=CONTROL_HISTORY_LENGTH, ge=4, le=32)
    encoder_width: int = Field(default=64, ge=32, le=256)
    recurrent_width: int = Field(default=64, ge=32, le=256)
    head_width: int = Field(default=128, ge=32, le=512)
    visual_feature_count: int = Field(default=0, ge=0, le=65536)
    epochs: int = Field(default=120, ge=1, le=10000)
    batch_size: int = Field(default=64, ge=1, le=4096)
    learning_rate: float = Field(default=.001, gt=0, le=.01)
    gradient_norm: float = Field(default=1., gt=0, le=10)
    seed: int = Field(default=805, ge=0)


# 功能：
#   重新验证并独立持有离线网络配置，拒绝超出 Torch 随机种子范围的值。
# 输入：
#   config：可能经 model_copy 或直接修改过的配置实例。
# 输出：
#   validated：与调用方不共享可变字段的有效配置。
def _validated_config(config: CausalPolicyConfig) -> CausalPolicyConfig:
    if not isinstance(config, CausalPolicyConfig):
        raise ValueError("CAUSAL_CONFIG_INVALID")
    validated = CausalPolicyConfig.model_validate(config.model_dump(mode="python"), strict=True)
    if validated.seed > 2**63 - 1:
        raise ValueError("CAUSAL_CONFIG_SEED_INVALID")
    return validated


# 功能：
#   在离线边界重新验证并复制传感器记录及标签，阻止外部修改污染已构造窗口。
# 输入：
#   sample：观测记录或带监督标签的训练记录。
# 输出：
#   validated：包含独立容器及重新验证字段的对应契约实例。
def _validated_observation(sample):
    if not isinstance(sample, LocalPolicyObservation):
        raise ValueError("CAUSAL_TRAINING_OBSERVATION_INVALID")
    contract = (LocalPolicyTrainingSample if isinstance(sample, LocalPolicyTrainingSample)
                else LocalPolicyObservation)
    payload = sample.model_dump(mode="python")
    # 严格模式要求 dataclass 实例，不能把 model_dump 的字典直接当成合法实例。
    # 重新构造冻结限额既保持类型，也再次执行物理尺度校验。
    if sample.pilot_control_limits is not None:
        if not isinstance(sample.pilot_control_limits, PilotControlLimits):
            raise ValueError("CAUSAL_TRAINING_CONTROL_LIMITS_INVALID")
        payload["pilot_control_limits"] = PilotControlLimits(**payload["pilot_control_limits"])
    validated = contract.model_validate(payload, strict=True)
    return validated


class CausalPilotPolicy(nn.Module):
    """Encode causal sensor history into mode, four-axis control and risk proposals."""

    # 功能：
    #   建立编码器、循环状态和模式／控制轴／风险头；各专家共享结构而非共享训练权重。
    # 输入：
    #   self：新建的离线 Torch 策略对象。
    #   config：网络宽度、历史长度及可选视觉输入配置。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, config: CausalPolicyConfig) -> None:
        super().__init__()
        self.config = config = _validated_config(config)
        self.observation_encoder = nn.Sequential(
            nn.Linear(CONTROL_HISTORY_WIDTH, config.encoder_width), nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(config.encoder_width, config.recurrent_width)
        self.control_head = nn.Sequential(
            nn.Linear(config.encoder_width + config.recurrent_width + config.visual_feature_count,
                      config.head_width), nn.SiLU(),
            nn.Linear(config.head_width, config.encoder_width), nn.SiLU(),
        )
        self.mode_head = nn.Linear(config.encoder_width, 4)
        self.axes_head = nn.Linear(config.encoder_width, 4)
        self.risk_head = nn.Linear(config.encoder_width, 1)

    # 功能：
    #   融合当前特征、独立来源历史及可选视觉编码；零掩码填充行不能推进循环状态。
    # 输入：
    #   self：当前策略网络。
    #   state：批次状态张量。
    #   realtime：批次实时特征张量。
    #   realtime_mask：当前特征逐项有效位。
    #   history：包括当前行与来源间隔的历史张量。
    #   history_mask：真实观测与左填充行的有效位。
    #   visual：当前帧的可选视觉特征张量。
    # 输出：
    #   hidden：控制头使用的融合隐向量。
    def latent(self, state, realtime, realtime_mask, history, history_mask, visual=None):
        current = torch.cat((state, realtime * realtime_mask, realtime_mask,
                             history[:, -1, -1:]), dim=-1)
        encoded = self.observation_encoder(current)
        h = torch.zeros((state.shape[0], self.config.recurrent_width),
                        dtype=state.dtype, device=state.device)
        # 固定有界展开便于 ONNX 部署；历史构造层禁止未来帧和重复轮询伪样本。
        for index in range(self.config.history_length):
            candidate = self.recurrent(self.observation_encoder(history[:, index]), h)
            valid = history_mask[:, index:index + 1]
            h = valid * candidate + (1 - valid) * h
        pieces = [encoded, h]
        if self.config.visual_feature_count:
            pieces.append(visual)
        hidden = self.control_head(torch.cat(pieces, dim=-1))
        return hidden

    # 功能：
    #   输出模式、风险和四轴控制幅度，永久禁用坐标候选分支；控制仍须经过 Harness 裁决。
    # 输入：
    #   self：当前策略网络。
    #   state：当前状态张量。
    #   realtime：当前实时特征张量。
    #   realtime_mask：当前特征有效位。
    #   history：按来源排序的历史张量。
    #   history_mask：历史有效行标记。
    #   visual：可选当前视觉特征。
    # 输出：
    #   candidates：固定禁用的八个坐标候选分数。
    #   modes：四种控制模式的未归一化分数。
    #   risk：零到一的风险建议。
    #   axes：负一到一的前、右、上速度及顺时针偏航角速度归一化幅度。
    def forward(self, state, realtime, realtime_mask, history, history_mask, visual=None):
        hidden = self.latent(state, realtime, realtime_mask, history, history_mask, visual)
        modes = self.mode_head(hidden)
        axes = torch.tanh(self.axes_head(hidden))
        risk = torch.sigmoid(self.risk_head(hidden))
        # The outer port has a stable output contract. All coordinate-candidate
        # scores are disabled; no candidate tensor is an input to this graph.
        candidates = torch.full((state.shape[0], 8), -1e6, dtype=state.dtype, device=state.device)
        return candidates, modes, risk, axes


@dataclass(frozen=True)
class CausalExample:
    """One labelled current frame and its ordered, source-identified sensor history."""
    sample: LocalPolicyTrainingSample
    history: list[list[float]]
    history_mask: list[float]
    group_id: str
    source_sha256: tuple[str, ...] = ()


# 功能：
#   1. 只复制同契约的传感器编码器与历史循环层，不复制任何控制、模式或风险输出头。
#   2. 全部校验通过后独立复制参数，保留原模型与新模型之间的内存隔离。
# 输入：
#   target：已按本次随机种子初始化的新策略。
#   source：已在训练入口绑定来源、视觉身份和留出隔离的编码器基座。
# 输出：
#   copied_keys：实际复制的状态键列表。
def initialize_sensor_encoder(target: CausalPilotPolicy, source: CausalPilotPolicy) -> list[str]:
    if not isinstance(target, CausalPilotPolicy) or not isinstance(source, CausalPilotPolicy) or target is source:
        raise ValueError("CAUSAL_ENCODER_TRANSFER_MODEL_INVALID")
    target_config, source_config = _validated_config(target.config), _validated_config(source.config)
    fields = ("history_length", "encoder_width", "recurrent_width", "visual_feature_count")
    if any(getattr(target_config, field) != getattr(source_config, field) for field in fields):
        raise ValueError("CAUSAL_ENCODER_TRANSFER_ARCHITECTURE_MISMATCH")
    prefixes = ("observation_encoder.", "recurrent.")
    source_state, target_state = source.state_dict(), target.state_dict()
    copied_keys = sorted(key for key in target_state if key.startswith(prefixes))
    if not copied_keys or set(copied_keys) != {key for key in source_state if key.startswith(prefixes)}:
        raise ValueError("CAUSAL_ENCODER_TRANSFER_STATE_MISMATCH")
    for key in copied_keys:
        value, wanted = source_state[key], target_state[key]
        if value.shape != wanted.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("CAUSAL_ENCODER_TRANSFER_WEIGHTS_INVALID")
    # 保留全部新输出头；load_state_dict 将数据复制到目标，不让后续训练修改基座。
    target_state.update({key: source_state[key].detach().clone() for key in copied_keys})
    target.load_state_dict(target_state, strict=True)
    return copied_keys


# 功能：
#   1. 从独立来源构建只含过去及当前观测的完整窗口，保持任务分组与专家标签绑定。
#   2. 无标签观测只能填充历史，不能伪造动作；坐标候选及旧输入语义不可用于训练。
# 输入：
#   samples：具有已验证行为监督的连续控制标签记录。
#   stream_groups：来源标识到独立任务组的映射。
#   history_length：四到三十二行的历史长度。
#   navigation_role：可选的目标专家角色。
#   history_observations：可选完整观测流，省略时使用标签记录自身。
# 输出：
#   results：独立持有当前记录、完整历史和来源摘要的训练窗口列表。
def causal_examples(
    samples: list[LocalPolicyTrainingSample], *, stream_groups: dict[str, str],
    history_length: int = CONTROL_HISTORY_LENGTH,
    navigation_role: str | None = None,
    history_observations: list[LocalPolicyObservation] | None = None,
) -> list[CausalExample]:
    samples = [_validated_observation(sample) for sample in samples]
    if any(not isinstance(sample, LocalPolicyTrainingSample) for sample in samples):
        raise ValueError("CAUSAL_TRAINING_LABELS_REQUIRED")
    if not isinstance(stream_groups, dict):
        raise ValueError("CAUSAL_TRAINING_MISSION_GROUP_MISSING")
    require_behavior_supervision(samples)
    if not samples or any(sample.temporal_evidence is None for sample in samples):
        raise ValueError("CAUSAL_TRAINING_SOURCE_PROVENANCE_REQUIRED")
    if navigation_role is not None and navigation_role not in NAVIGATION_EXPERT_ROLES:
        raise ValueError("CAUSAL_TRAINING_MANOEUVRE_ROLE_INVALID")
    if navigation_role is None and len({sample.navigation_expert_role for sample in samples}) != 1:
        raise ValueError("CAUSAL_TRAINING_REQUIRES_ONE_MANOEUVRE_ROLE")
    history = CausalControlHistory(history_length)
    sources = deque(maxlen=history_length)
    seen = set()
    results = []
    labelled = {}
    for sample in samples:
        key = (sample.temporal_evidence.stream_id, sample.temporal_evidence.sample_sha256)
        if key in labelled:
            raise ValueError("CAUSAL_TRAINING_DUPLICATE_SOURCE")
        if sample.target_action_index not in range(8, 12) or len(sample.target_pilot_control) != 4:
            raise ValueError("CAUSAL_TRAINING_REQUIRES_CONTINUOUS_LABELS")
        labelled[key] = sample
    observations = samples if history_observations is None else [
        _validated_observation(sample) for sample in history_observations
    ]
    if not observations or any(s.temporal_evidence is None for s in observations):
        raise ValueError("CAUSAL_TRAINING_SOURCE_PROVENANCE_REQUIRED")
    for sample in sorted(observations, key=lambda s: (
        s.temporal_evidence.stream_id, s.temporal_evidence.observed_at_unix_ms,
    )):
        evidence = sample.temporal_evidence
        group = stream_groups.get(evidence.stream_id)
        if (not isinstance(group, str) or not group.strip() or len(group) > 256
                or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in group)):
            raise ValueError("CAUSAL_TRAINING_MISSION_GROUP_MISSING")
        group = group.strip()
        if sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
            raise ValueError("CAUSAL_TRAINING_FEATURE_CONTRACT_MISMATCH")
        if any(sample.candidate_mask) or any(v for row in sample.candidate_features for v in row):
            raise ValueError("CAUSAL_TRAINING_CANNOT_USE_COORDINATE_CANDIDATES")
        identity = (evidence.stream_id, evidence.sample_sha256)
        if identity in seen:
            raise ValueError("CAUSAL_TRAINING_DUPLICATE_SOURCE")
        seen.add(identity)
        label = labelled.get(identity)
        if label is not None and any(
            getattr(sample, name) != getattr(label, name)
            for name in LocalPolicyObservation.model_fields if name != "visual_features"
        ):
            raise ValueError("CAUSAL_TRAINING_LABEL_OBSERVATION_MISMATCH")
        # Historical rows use numerical state only. Current visual features
        # remain attached to the actual labelled frame, not to another row.
        independent = history.append(evidence, sample.state_features, sample.realtime_features,
                                     sample.realtime_valid_mask)
        if independent:
            if len(history.history.rows) == 1:
                sources.clear()
            sources.append(evidence.sample_sha256)
        else:
            # 源摘要随末槽修订替换，不能让来源列表比实际因果历史多出一行。
            sources[-1] = evidence.sample_sha256
        # Training warms history just like inference. Startup partial windows
        # cannot authorize flight and are not taught as ordinary motion.
        selected = label is not None and (
            navigation_role is None or label.navigation_expert_role == navigation_role)
        if history.ready and selected:
            # 所有观测仍先经过完整校验和历史推进；仅在确实输出训练窗口时
            # 复制二维数组，避免为暖机、断档及其他专家反复创建后立即丢弃的大列表。
            values, mask = history.values()
            results.append(CausalExample(label, values, mask, group, tuple(sources)))
    if set(labelled) - seen:
        raise ValueError("CAUSAL_TRAINING_LABEL_WITHOUT_SOURCE_OBSERVATION")
    if not results:
        raise ValueError("CAUSAL_TRAINING_HAS_NO_COMPLETE_WINDOWS")
    return results


# 功能：
#   校验所有历史行与当前帧的一致性，再生成有限 float32 张量；不能把错误掩码藏在旧行中。
# 输入：
#   examples：同一历史长度的训练或评价窗口。
#   visual_feature_count：当前模型需要的视觉特征宽度，零表示不用视觉输入。
# 输出：
#   tensors：状态、实时特征、有效位、历史、历史有效位及可选视觉张量的有序元组。
def example_tensors(examples: list[CausalExample], visual_feature_count: int):
    if not examples:
        raise ValueError("CAUSAL_TRAINING_EXAMPLES_EMPTY")
    if type(visual_feature_count) is not int or not 0 <= visual_feature_count <= 65536:
        raise ValueError("CAUSAL_TRAINING_VISUAL_WIDTH_MISMATCH")
    expected_length = len(examples[0].history)
    for example in examples:
        _validated_observation(example.sample)
        current = control_history_row(
            example.sample.state_features, example.sample.realtime_features,
            example.sample.realtime_valid_mask,
        )
        if (not 4 <= expected_length <= 32 or len(example.history) != expected_length
                or len(example.history_mask) != len(example.history)
                or any(type(v) not in (int, float) or v not in (0., 1.)
                       for v in example.history_mask)
                or example.history_mask[-1] != 1.
                or example.history_mask != sorted(example.history_mask)
                or any(len(row) != CONTROL_HISTORY_WIDTH for row in example.history)
                or tuple(example.history[-1][:-1]) != current
                or any(not 0 <= row[-1] <= 1 for row in example.history)):
            raise ValueError("CAUSAL_TRAINING_HISTORY_CONTENT_MISMATCH")
        for row, valid in zip(example.history, example.history_mask, strict=True):
            mask_start = CONTROL_STATE_WIDTH + CONTROL_REALTIME_WIDTH
            packed = control_history_row(row[:CONTROL_STATE_WIDTH],
                                         row[CONTROL_STATE_WIDTH:mask_start], row[mask_start:-1])
            if tuple(row[:-1]) != packed or (not valid and any(row)):
                raise ValueError("CAUSAL_TRAINING_HISTORY_CONTENT_MISMATCH")
    if any(len(e.sample.visual_features) != visual_feature_count for e in examples):
        raise ValueError("CAUSAL_TRAINING_VISUAL_WIDTH_MISMATCH")
    columns = (
        [e.sample.state_features for e in examples],
        [e.sample.realtime_features for e in examples],
        [e.sample.realtime_valid_mask for e in examples],
        [e.history for e in examples], [e.history_mask for e in examples],
    )
    if visual_feature_count:
        columns += ([e.sample.visual_features for e in examples],)
    arrays = [np.asarray(column, dtype=np.float32) for column in columns]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("CAUSAL_TRAINING_NONFINITE_INPUT")
    tensors = tuple(torch.from_numpy(array) for array in arrays)
    return tensors


# 功能：
#   在学习开始前重新检查可变标签、完整独立来源窗口、专家一致性和当前特征契约。
# 输入：
#   examples：准备学习或评价的因果窗口列表。
#   history_length：配置要求的完整历史长度。
# 输出：
#   None：不返回业务数据。
def validate_learning_examples(examples: list[CausalExample], history_length: int) -> None:
    if type(history_length) is not int or not 4 <= history_length <= 32:
        raise ValueError("CAUSAL_CONTROL_HISTORY_LENGTH_INVALID")
    require_behavior_supervision([example.sample for example in examples])
    if not examples:
        raise ValueError("CAUSAL_TRAINING_EXAMPLES_EMPTY")
    if len({e.sample.navigation_expert_role for e in examples}) != 1:
        raise ValueError("CAUSAL_TRAINING_REQUIRES_ONE_MANOEUVRE_ROLE")
    for example in examples:
        sample = _validated_observation(example.sample)
        if not isinstance(sample, LocalPolicyTrainingSample):
            raise ValueError("CAUSAL_TRAINING_LABELS_REQUIRED")
        if (sample.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                or sample.temporal_evidence is None or not isinstance(example.group_id, str)
                or not example.group_id or example.group_id != example.group_id.strip()
                or len(example.group_id) > 256
                or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in example.group_id)
                or len(example.history) != history_length
                or example.history_mask != [1.] * history_length
                or len(example.source_sha256) != history_length
                or any(not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{64}", source) is None
                       for source in example.source_sha256)
                or len(set(example.source_sha256)) != history_length
                or example.source_sha256[-1] != sample.temporal_evidence.sample_sha256
                or sample.target_action_index not in range(8, 12)
                or len(sample.target_pilot_control) != 4
                or any(sample.candidate_mask)
                or any(v for row in sample.candidate_features for v in row)):
            raise ValueError("CAUSAL_TRAINING_COMPLETE_CURRENT_WINDOWS_REQUIRED")


@dataclass(frozen=True)
class FrozenCausalEvaluation:
    """Held-out tensors without retaining the decoded observation graphs.

    Construct before a simulator starts; evaluate only after it is quiesced.
    Evaluation never backpropagates or selects a checkpoint on these labels.
    """

    inputs: tuple[torch.Tensor, ...]
    target_modes: torch.Tensor
    target_axes: torch.Tensor
    groups: tuple[str, ...]
    expert_role: str

    # 功能：
    #   在优化或仿真开始前冻结留出任务的评价张量，标签不参与梯度或检查点选择。
    # 输入：
    #   cls：冻结评价对象类型。
    #   examples：已经分离任务及来源的留出窗口。
    #   config：网络历史和视觉输入配置。
    # 输出：
    #   evaluation：独立张量、目标模式、四轴标签及任务组信息。
    @classmethod
    def compile(cls, examples, config):
        config = _validated_config(config)
        validate_learning_examples(examples, config.history_length)
        evaluation = cls(
            example_tensors(examples, config.visual_feature_count),
            torch.tensor([e.sample.target_action_index - 8 for e in examples]),
            torch.tensor([e.sample.target_pilot_control for e in examples], dtype=torch.float32),
            tuple(sorted({e.group_id for e in examples})),
            examples[0].sample.navigation_expert_role,
        )
        return evaluation

    # 功能：
    #   无梯度地报告留出模式和各轴误差；缺失类别保持空统计，不伪造准确率或飞行资格。
    # 输入：
    #   self：冻结的留出评价数据。
    #   model：待评价的因果策略，结束或异常后恢复其训练模式。
    # 输出：
    #   metrics：模式覆盖、轴误差、任务组和明确未授予飞行资格的统计字典。
    def evaluate(self, model: CausalPilotPolicy) -> dict:
        was_training = model.training
        model.eval()
        predicted_modes, controls = [], []
        try:
            with torch.no_grad():
                for start in range(0, len(self.target_modes), 256):
                    _, scores, risk, axes = model(*(v[start:start + 256] for v in self.inputs))
                    if any(not torch.isfinite(v).all() for v in (scores, risk, axes)):
                        raise ValueError("CAUSAL_EVALUATION_NONFINITE_OUTPUT")
                    predicted_modes.append(scores.argmax(dim=1))
                    controls.append(axes)
        finally:
            model.train(was_training)
        modes, axes = torch.cat(predicted_modes), torch.cat(controls)
        error = (axes - self.target_axes).abs()
        moving = self.target_modes == 3
        by_mode = {}
        for index, name in enumerate(MODES):
            selected = self.target_modes == index
            count = int(selected.sum())
            by_mode[name] = {"count": count, "accuracy":
                float((modes[selected] == index).float().mean()) if count else None}
        by_axis = {}
        for index, name in enumerate(("forward", "right", "up", "yaw")):
            target = self.target_axes[:, index]
            by_axis[name] = {}
            for sign, condition in (("positive", target > .01), ("negative", target < -.01),
                                    ("neutral", target.abs() <= .01)):
                selected = moving & condition
                count = int(selected.sum())
                by_axis[name][sign] = {"count": count, "mae":
                    float(error[selected, index].mean()) if count else None}
        metrics = {
            "validation_window_count": len(self.target_modes),
            "validation_groups": list(self.groups),
            "validation_mode_accuracy": float((modes == self.target_modes).float().mean()),
            "validation_moving_axis_mae": float(error[moving].mean()) if moving.any() else None,
            "validation_by_mode": by_mode, "validation_by_axis": by_axis,
            "qualified_for_flight": False,
        }
        return metrics


# 功能：
#   1. 检查任务组和来源独立性，只对训练窗口学习模式、四轴幅度及风险。
#   2. 精调时复制基座参数，保留原模型；最终独立评价不参与训练或选择检查点。
# 输入：
#   train：完整且带连续控制监督的训练窗口。
#   validation：任务及物理来源均独立的留出窗口。
#   config：离线学习配置。
#   initial_policy：可选同架构已训练基座，不原位修改。
#   initial_encoder：与完整热启动互斥，仅迁移已验证的编码器和历史层，其余头从零学习。
#   regularization：可选离线路线均衡与视觉屏蔽配置，不改变部署输入或已有网络结构。
#   device：显式 cpu 或 cuda；只将当前训练批次移至设备，冻结评价与导出返回 CPU。
# 输出：
#   model：完成优化并处于评价模式的独立策略。
#   metrics：训练损失、留出评价和来源统计，不授予飞行资格。
def train_causal_policy(train, validation, config: CausalPolicyConfig, *,
                       initial_policy: CausalPilotPolicy | None = None,
                       initial_encoder: CausalPilotPolicy | None = None,
                       regularization: CausalRegularization | None = None, device: str = 'cpu'):
    config = _validated_config(config)
    training_device = causal_training_device(device)
    regularization = validate_regularization(regularization, config.visual_feature_count)
    if initial_policy is not None and initial_encoder is not None:
        raise ValueError("CAUSAL_INITIALIZATION_MODES_CONFLICT")
    if not train or not validation:
        raise ValueError("CAUSAL_TRAINING_AND_VALIDATION_REQUIRED")
    if {e.group_id for e in train} & {e.group_id for e in validation}:
        raise ValueError("CAUSAL_TRAINING_MISSION_GROUP_LEAKAGE")
    validate_learning_examples(train, config.history_length)
    validate_learning_examples(validation, config.history_length)
    if train[0].sample.navigation_expert_role != validation[0].sample.navigation_expert_role:
        raise ValueError("CAUSAL_TRAINING_VALIDATION_ROLE_MISMATCH")
    if {source for e in train for source in e.source_sha256} & {
        source for e in validation for source in e.source_sha256
    }:
        raise ValueError("CAUSAL_TRAINING_SOURCE_LEAKAGE")
    # 留出输入先验证和冻结，避免优化完才发现评价数据损坏；绝不用于反向传播。
    evaluation = FrozenCausalEvaluation.compile(validation, config)
    inputs = example_tensors(train, config.visual_feature_count)
    sample_weights = effective_sample_weights(
        train, balance_groups=regularization.balance_mission_groups)
    torch.manual_seed(config.seed)
    model = CausalPilotPolicy(config)
    if initial_policy is not None:
        architecture_fields = ("history_length", "encoder_width", "recurrent_width", "head_width",
                               "visual_feature_count")
        if any(getattr(initial_policy.config, field) != getattr(config, field)
               for field in architecture_fields):
            raise ValueError("CAUSAL_REFINEMENT_ARCHITECTURE_MISMATCH")
        state = initial_policy.state_dict()
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("CAUSAL_REFINEMENT_INITIAL_WEIGHTS_NONFINITE")
        # load_state_dict copies into distinct tensors. Refinement must never
        # modify the checkpoint/model retained for baseline comparison.
        model.load_state_dict(state, strict=True)
    copied_encoder_keys = (initialize_sensor_encoder(model, initial_encoder)
                           if initial_encoder is not None else [])
    # 先复制独立基座，再移动目标模型；不移动或修改调用方保留的原基座。
    model.to(training_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    modes = torch.tensor([e.sample.target_action_index - 8 for e in train], dtype=torch.long)
    axes = torch.tensor([e.sample.target_pilot_control for e in train], dtype=torch.float32)
    risks = torch.tensor([[e.sample.risk_target] for e in train], dtype=torch.float32)
    counts = torch.bincount(modes, minlength=4).clamp_min(1)
    class_weights = (len(train) / (4 * counts)).clamp(.25, 16).to(training_device)
    generator = torch.Generator().manual_seed(config.seed)
    dropout_generator = torch.Generator().manual_seed(config.seed ^ 0x5A17)
    losses = []
    for _ in range(config.epochs):
        order = torch.randperm(len(train), generator=generator)
        for batch in order.split(config.batch_size):
            batch_inputs = tuple(value[batch] for value in inputs)
            batch_inputs = drop_visual_block(
                batch_inputs, regularization.visual_block_dropout, dropout_generator)
            # 顺序和视觉屏蔽始终使用 CPU 的固定随机流；仅本批数据上卡，
            # 不让整个历史语料随样本增长永久占满显存。
            batch_inputs = tuple(value.to(training_device) for value in batch_inputs)
            batch_modes = modes[batch].to(training_device)
            batch_axes = axes[batch].to(training_device)
            batch_risks = risks[batch].to(training_device)
            batch_weights = sample_weights[batch].to(training_device)
            _, logits, risk, control = model(*batch_inputs)
            mode_loss = nn.functional.cross_entropy(logits, batch_modes, weight=class_weights,
                                                    reduction="none")
            moving = batch_modes == 3
            axis_loss = (control - batch_axes).square().mean(dim=1)
            # Explicit neutral labels also teach stop magnitudes, but cannot
            # drown the active-axis control objective in long hold records.
            axis_weight = torch.where(moving, 1., .1)
            risk_loss = nn.functional.binary_cross_entropy(
                risk, batch_risks, reduction="none"
            )[:, 0]
            loss = ((mode_loss + axis_weight * axis_loss + .5 * risk_loss)
                    * batch_weights).sum() / batch_weights.sum()
            # 在反向传播前拒绝数值异常，不能把非有限控制损失交给优化器。
            if not torch.isfinite(loss):
                raise ValueError("CAUSAL_TRAINING_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            if any(not torch.isfinite(value).all() for value in model.parameters()):
                raise ValueError("CAUSAL_TRAINING_NONFINITE_PARAMETER")
            losses.append(float(loss.detach()))
    # CPU 评价/导出使用相同的已训练权重；检查点无需 CUDA 才能读取。
    device_evidence = causal_device_evidence(training_device)
    model.cpu()
    model.eval()
    metrics = {
        **device_evidence,
        **evaluation.evaluate(model),
        "training_window_count": len(train),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "last_training_loss": losses[-1], "qualified_for_flight": False,
        "initialization": ("existing-causal-policy" if initial_policy is not None else
                           "transferred-sensor-encoder" if initial_encoder is not None else "random"),
        "transferred_encoder_state_keys": copied_encoder_keys,
        "training_groups": sorted({e.group_id for e in train}),
        "validation_groups": sorted({e.group_id for e in validation}),
        "regularization": regularization.model_dump(),
    }
    return model, metrics


# 功能：
#   保存带当前特征及历史契约的有限权重检查点，独占创建而不覆盖基座或其他进程产物。
# 输入：
#   model：待保存的因果策略。
#   path：新的检查点文件路径。
# 输出：
#   None：不返回业务数据。
def save_causal_checkpoint(model: CausalPilotPolicy, path: Path) -> None:
    config = _validated_config(model.config)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("CAUSAL_CHECKPOINT_WEIGHTS_NONFINITE")
    with path.open("xb") as handle:
        torch.save({"architecture": "causal-gru-control", "config": config.model_dump(),
                    "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
                    "history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
                    "state_dict": model.state_dict()}, handle)


# 功能：
#   只载入当前语义契约、键集合及有限 float32 权重；支持用已散列字节避免二次读路径漂移。
# 输入：
#   path：本地检查点路径，或调用者已验证来源的完整字节。
# 输出：
#   model：严格载入参数并处于评价模式的因果策略。
def load_causal_checkpoint(path: Path | bytes) -> CausalPilotPolicy:
    # Callers that bind a receipt use the exact bytes they hash, not a second
    # pathname read that could load different weights during an update.
    payload = torch.load(io.BytesIO(path) if isinstance(path, bytes) else path,
                         map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("architecture") != "causal-gru-control"
            or payload.get("feature_contract_sha256") != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
            or payload.get("history_contract_sha256") != CONTROL_HISTORY_CONTRACT_SHA256):
        raise ValueError("CAUSAL_CHECKPOINT_IDENTITY_MISMATCH")
    config = CausalPolicyConfig.model_validate(payload.get("config"), strict=True)
    model = CausalPilotPolicy(config)
    state = payload.get("state_dict")
    if (not isinstance(state, dict) or set(state) != set(model.state_dict())
            or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
                   or not torch.isfinite(v).all() for v in state.values())):
        raise ValueError("CAUSAL_CHECKPOINT_WEIGHTS_INVALID")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# 功能：
#   1. 检查权重及零输入数值输出，在内存导出并验证完整 ONNX 结构和语义元数据。
#   2. 独占发布新文件，恢复调用者训练模式；结构和数值检查不等于飞行验收。
# 输入：
#   model：待导出的因果策略。
#   path：尚不存在的目标 ONNX 路径。
# 输出：
#   digest：实际发布文件字节的 SHA-256。
def export_causal_policy(model: CausalPilotPolicy, path: Path) -> str:
    import onnx

    if path.exists():
        raise FileExistsError(path)
    config = _validated_config(model.config)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("CAUSAL_EXPORT_WEIGHTS_NONFINITE")
    inputs = (
        torch.zeros(1, CONTROL_STATE_WIDTH), torch.zeros(1, CONTROL_REALTIME_WIDTH),
        torch.ones(1, CONTROL_REALTIME_WIDTH),
        torch.zeros(1, config.history_length, CONTROL_HISTORY_WIDTH),
        torch.ones(1, config.history_length),
    )
    names = ["state_features", "realtime_features", "realtime_valid_mask", "control_history",
             "control_history_mask"]
    if config.visual_feature_count:
        inputs += (torch.zeros(1, config.visual_feature_count),)
        names.append("visual_features")
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    buffer = io.BytesIO()
    try:
        model.eval()
        with torch.no_grad():
            if any(not torch.isfinite(value).all() for value in model(*inputs)):
                raise ValueError("CAUSAL_EXPORT_OUTPUT_NONFINITE")
        torch.onnx.export(
            model, inputs, buffer, input_names=names,
            output_names=["candidate_scores", "action_scores", "risk_score", "pilot_control"],
            opset_version=17, dynamo=False,
        )
    finally:
        model.train(was_training)
    graph = onnx.load_model_from_string(buffer.getvalue())
    for name, value in {
        "architecture": "causal-gru-control", "history_length": str(config.history_length),
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
        "config_sha256": sha256_json(config),
    }.items():
        item = graph.metadata_props.add()
        item.key, item.value = name, value
    onnx.checker.check_model(graph)
    content = graph.SerializeToString()
    # The preflight exists() check is only a fast error. Exclusive creation is
    # what protects an artifact created by another process during export.
    with path.open("xb") as stream:
        stream.write(content)
    digest = hashlib.sha256(content).hexdigest()
    return digest
