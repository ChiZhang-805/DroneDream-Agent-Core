"""Offline heading conditioning and declared reflection augmentation; no flight authority."""

import io
import re

import torch
from torch import nn

from ..heading_observation import HEADING_OBSERVATION_SHA256, HEADING_OBSERVATION_WIDTH

ARCHITECTURE = 'conditioned-heading-mlp-v1'
ANGLE_INPUT_SCALE = 9.


class ConditionedHeadingPolicy(nn.Module):
    """Learn yaw magnitude with better-conditioned bearing inputs, not a teacher formula."""

    # 功能：
    #   放大输入中的细小方位差供网络学习，控制输出全部来自可训练参数。
    # 输入：
    #   self：候选网络实例。
    # 输出：
    #   None：建立固定输入尺度及两个 32 单元隐层，不授予飞行权限。
    def __init__(self):
        super().__init__()
        self.register_buffer('input_scale', torch.tensor([ANGLE_INPUT_SCALE, 1., 1., ANGLE_INPUT_SCALE, 1., 1., 1., 1.]))
        self.network = nn.Sequential(nn.Linear(HEADING_OBSERVATION_WIDTH, 32), nn.SiLU(),
                                     nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    # 功能：
    #   将未经改变语义的八维观测映射为连续偏航幅度；缩放只改变网络输入的数值条件。
    # 输入：
    #   features：批次八维方位、距离、有效位及目标状态。
    # 输出：
    #   yaw_axis：负一至一的归一化偏航预测。
    def forward(self, features):
        yaw_axis = self.network(features * self.input_scale)
        return yaw_axis


# 功能：
#   校验训练计算边界，不接受无穷值、非法掩码或超出观测契约的几何值。
# 输入：
#   features、targets：CPU float32 的八维观测与单轴实际执行标签。
# 输出：
#   None：非法数据在增强和赋权前拒绝。
def validate_heading_training_tensors(features, targets):
    if (not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or features.device.type != 'cpu' or targets.device.type != 'cpu'
            or features.dtype != torch.float32 or targets.dtype != torch.float32
            or features.ndim != 2 or features.shape[1] != 8 or not len(features)
            or targets.shape != (len(features), 1)
            or not torch.isfinite(features).all() or not torch.isfinite(targets).all()
            or (targets.abs() > 1).any() or (features[:, [0, 3]].abs() > 1).any()
            or (features[:, [1, 4]] < 0).any() or (features[:, [1, 4]] > 4).any()
            or not ((features[:, [2, 5, 6, 7]] == 0) | (features[:, [2, 5, 6, 7]] == 1)).all()):
        raise ValueError('HEADING_REFINEMENT_TENSOR_INVALID')


# 功能：
#   为方位几何的左右反射创建独立训练副本；它不是新增实测轨迹或恢复事件。
# 输入：
#   features、targets：经验证的原观测与归一化偏航标签。
# 输出：
#   mirrored_features、mirrored_targets：两方位反号、对应偏航反号的增强张量。
def reflect_heading_training(features, targets):
    validate_heading_training_tensors(features, targets)
    mirrored_features, mirrored_targets = features.clone(), -targets
    mirrored_features[:, [0, 3]] *= -1
    return mirrored_features, mirrored_targets


# 功能：
#   保持路线总权重均衡，并可温和增加少见方向及幅度的权重，不根据验证误差选择样本。
# 输入：
#   features、targets：训练输入与原执行标签；groups：训练路线组标识。
#   balance_turns：是否启用组内转向类别平衡。
# 输出：
#   weights：均值为一的有限正权重，每条路线总权重相同。
def heading_training_weights(features, targets, groups, *, balance_turns: bool):
    validate_heading_training_tensors(features, targets)
    if (type(balance_turns) is not bool or type(groups) not in (list, tuple) or len(groups) != len(features)
            or any(type(group) is not str or not group for group in groups)):
        raise ValueError('HEADING_REFINEMENT_GROUPS_INVALID')
    axes = targets[:, 0]
    # 三个非零幅度区间分别保留正负方向；零附近是单独类别。边界与验证集无关。
    magnitudes = torch.bucketize(axes.abs(), torch.tensor([.01, .25, .75]))
    buckets = torch.where(magnitudes == 0, 0, magnitudes + (axes < 0).long() * 3)
    weights = torch.empty(len(features), dtype=torch.float32)
    indices = {}
    for index, group in enumerate(groups):
        indices.setdefault(group, []).append(index)
    for members in indices.values():
        values = torch.ones(len(members), dtype=torch.float32)
        if balance_turns:
            labels = buckets[members]
            unique, counts = labels.unique(return_counts=True)
            for label, count in zip(unique, counts, strict=True):
                values[labels == label] = min(4., (len(members) / (len(unique) * count.item()))**.5)
        weights[members] = values / values.sum() * (len(features) / len(indices))
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('HEADING_REFINEMENT_WEIGHTS_INVALID')
    return weights


# 功能：
#   加载隔离研究候选，核对感知协议、固定输入尺度和有限参数；不授予飞行权限。
# 输入：
#   content：已核对文件摘要的检查点字节；expected_plan_sha256：外部冻结计划摘要。
# 输出：
#   model：CPU 评价模式下的条件化偏航模型。
def load_conditioned_heading_policy(content: bytes, *, expected_plan_sha256: str) -> ConditionedHeadingPolicy:
    if (type(content) is not bytes or not 0 < len(content) <= 1024*1024
            or type(expected_plan_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', expected_plan_sha256) is None):
        raise ValueError('CONDITIONED_HEADING_INPUT_INVALID')
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
    if (type(payload) is not dict
            or set(payload) != {'architecture', 'heading_observation_sha256', 'state_dict',
                                'qualified_for_flight', 'yaw_limit_dps', 'plan_sha256'}
            or payload.get('architecture') != ARCHITECTURE
            or payload.get('heading_observation_sha256') != HEADING_OBSERVATION_SHA256
            or payload.get('qualified_for_flight') is not False
            or type(payload.get('yaw_limit_dps')) is not float or payload['yaw_limit_dps'] != 20.
            or payload.get('plan_sha256') != expected_plan_sha256):
        raise ValueError('CONDITIONED_HEADING_CONTRACT_INVALID')
    model = ConditionedHeadingPolicy()
    state, expected = payload.get('state_dict'), model.state_dict()
    if (type(state) is not dict or set(state) != set(expected)
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                   or value.layout != torch.strided or value.shape != expected[key].shape
                   or not torch.isfinite(value).all() for key, value in state.items())
            or not torch.equal(state['input_scale'], expected['input_scale'])):
        raise ValueError('CONDITIONED_HEADING_WEIGHTS_INVALID')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class BoundedHeadingExport(nn.Module):
    """Explicit floating-point boundary for research ONNX exports, not a safety controller."""

    # 功能：
    #   为已核验候选设置独立导出边界，保留原训练权重和原始导出证据。
    # 输入：
    #   policy：条件化偏航候选。
    # 输出：
    #   None：保存待导出候选，不改变其权重。
    def __init__(self, policy: ConditionedHeadingPolicy):
        super().__init__()
        if not isinstance(policy, ConditionedHeadingPolicy):
            raise ValueError('CONDITIONED_HEADING_EXPORT_TYPE_INVALID')
        self.policy = policy

    # 功能：
    #   显式限制归一化轴的数值边界，避免推理库的 Tanh 浮点近似产生微小越界。
    # 输入：
    #   features：八维感知观测。
    # 输出：
    #   yaw_axis：裁剪到负一至一的候选输出。
    def forward(self, features):
        yaw_axis = torch.clamp(self.policy(features), min=-1., max=1.)
        return yaw_axis
