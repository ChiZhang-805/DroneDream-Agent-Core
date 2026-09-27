"""Causal circular heading student and noise-aware research regularization."""

import io
import math
import re

import torch
from torch import nn

from ..heading_context import HEADING_CONTEXT_SHA256

ARCHITECTURE = 'circular-context-heading-v1'


# 功能：
#   用正弦余弦表示有定义的方位，消除正负半圈接缝；缺测保持零。
# 输入：
#   geometry：末维为八的原始方位几何。
# 输出：
#   circular：末维为十的周期连续表示。
def circular_heading(geometry):
    parts = []
    for start in (0, 3):
        angle, distance, valid = geometry[..., start:start+1], geometry[..., start+1:start+2], geometry[..., start+2:start+3]
        parts.extend((torch.sin(angle*math.pi)*valid*3., torch.cos(angle*math.pi)*valid, distance, valid))
    circular = torch.cat((*parts, geometry[..., 6:8]), dim=-1)
    return circular


class StableHeadingPolicy(nn.Module):
    """Small learned yaw component with an explicit non-authoritative context contract."""

    # 功能：
    #   构建循环角度表示和可选真实历史输入的小网络；不使用手写偏航控制公式。
    # 输入：
    #   use_context：是否使用转速、过去动作和历史几何。
    # 输出：
    #   None：创建当前研究网络。
    def __init__(self, *, use_context: bool):
        super().__init__()
        if type(use_context) is not bool:
            raise ValueError('HEADING_STABILITY_CONTEXT_FLAG_INVALID')
        self.use_context = use_context
        self.network = nn.Sequential(nn.Linear(27 if use_context else 10, 48), nn.SiLU(),
            nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    # 功能：
    #   从同源当前和严格过去的输入计算连续偏航，输出采用显式数值限幅。
    # 输入：
    #   features：批次二十三维时序方位输入。
    # 输出：
    #   yaw_axis：负一至一的候选偏航幅度。
    def forward(self, features):
        current = circular_heading(features[:, :8])
        if self.use_context:
            history = circular_heading(features[:, 13:21]) * features[:, 22:23]
            inputs = torch.cat((current, features[:, 8:13], history, features[:, 21:23]), dim=1)
        else:
            inputs = current
        yaw_axis = torch.clamp(self.network(inputs), -1., 1.)
        return yaw_axis


# 功能：
#   镜像当前和历史方位、实际转速及过去动作，保持距离、时钟和有效位不变。
# 输入：
#   features、targets：二十三维训练输入和原执行标签。
# 输出：
#   reflected、labels：独立的左右反射训练副本。
def reflect_context(features, targets):
    reflected = features.clone()
    reflected[:, [0, 3, 8, 10, 13, 16]] *= -1
    labels = -targets
    return reflected, labels


# 功能：
#   为感知而非真实运动施加有界相关角度扰动，保持周期性与缺测有效位。
# 输入：
#   features：二十三维训练输入；delta：逐样本归一化角度扰动。
# 输出：
#   noisy：当前共同旋转、历史半相关旋转的独立输入。
def perturb_context(features, delta):
    noisy = features.clone()
    for angle, valid, correlation in ((0, 2, 1.), (3, 5, 1.), (13, 15, .5), (16, 18, .5)):
        noisy[:, angle:angle+1] = (torch.remainder(features[:, angle:angle+1] + delta*correlation + 1., 2.) - 1.) * features[:, valid:valid+1]
    return noisy


# 功能：
#   对真实相邻窗口匹配动作变化而不是强迫不动，保留示范中的必要转向和制动。
# 输入：
#   current、previous：相邻窗口预测；target、previous_target：对应原执行标签；valid：同源连续窗口掩码。
# 输出：
#   loss：有效窗口的动作增量误差，无有效窗口时为可微零值。
def temporal_delta_loss(current, previous, target, previous_target, valid):
    errors = ((current-previous)-(target-previous_target)).square()[:, 0]
    weights = valid.to(errors.dtype)
    loss = (errors*weights).sum()/weights.sum().clamp_min(1.)
    return loss


# 功能：
#   严格核对研究权重的输入契约、训练身份和张量结构；加载不会授予飞行权限。
# 输入：
#   content：调用方已校验摘要的检查点字节；plan_sha256、data_sha256：预期训练计划和数据摘要。
# 输出：
#   model：CPU 评价模式的时序偏航候选。
def load_stable_heading_policy(content: bytes, *, plan_sha256: str, data_sha256: str) -> StableHeadingPolicy:
    if type(content) is not bytes or not 0 < len(content) <= 1024*1024:
        raise ValueError('STABILITY_CHECKPOINT_SIZE_INVALID')
    if any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None for value in (plan_sha256, data_sha256)):
        raise ValueError('STABILITY_CHECKPOINT_IDENTITY_INVALID')
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
    expected_keys = {'architecture', 'feature_contract_sha256', 'plan_sha256', 'data_sha256',
                     'use_context', 'yaw_limit_dps', 'state_dict', 'qualified_for_flight'}
    if (type(payload) is not dict or set(payload) != expected_keys
            or payload['architecture'] != ARCHITECTURE or payload['feature_contract_sha256'] != HEADING_CONTEXT_SHA256
            or payload['plan_sha256'] != plan_sha256 or payload['data_sha256'] != data_sha256
            or type(payload['use_context']) is not bool or type(payload['yaw_limit_dps']) is not float
            or payload['yaw_limit_dps'] != 20. or payload['qualified_for_flight'] is not False):
        raise ValueError('STABILITY_CHECKPOINT_CONTRACT_MISMATCH')
    model = StableHeadingPolicy(use_context=payload['use_context'])
    state, expected = payload['state_dict'], model.state_dict()
    if (type(state) is not dict or set(state) != set(expected)
            or any(type(value) is not torch.Tensor or value.layout != torch.strided or value.dtype != torch.float32
                   or value.shape != expected[key].shape or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError('STABILITY_CHECKPOINT_WEIGHTS_INVALID')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
