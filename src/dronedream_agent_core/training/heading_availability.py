"""Route incomplete temporal observations to an independently trained current-frame expert."""

import torch
from torch import nn

from .heading_stability import StableHeadingPolicy


class AvailabilityHeadingPolicy(nn.Module):
    """Research composition of two frozen learned experts, not a hand-coded yaw controller."""

    # 功能：
    #   用当前观测专家接管缺失的时序输入；两分支均为已训练模型，不伪造旧回执。
    # 输入：
    #   temporal、current：时序和单帧研究模型；age_weighted：是否随回执老化减少时序分支权重。
    # 输出：
    #   None：创建只读推理组合。
    def __init__(self, temporal: StableHeadingPolicy, current: StableHeadingPolicy, *, age_weighted: bool):
        super().__init__()
        if (not isinstance(temporal, StableHeadingPolicy) or temporal.use_context is not True
                or not isinstance(current, StableHeadingPolicy) or current.use_context is not False
                or type(age_weighted) is not bool):
            raise ValueError('HEADING_AVAILABILITY_BRANCH_CONTRACT_INVALID')
        self.temporal, self.current, self.age_weighted = temporal, current, age_weighted
        self.requires_grad_(False)
        self.eval()

    # 功能：
    #   根据有效位与实际年龄融合两个学习输出，回执过期时仅用当前观测，不续期、不修改标签。
    # 输入：
    #   features：上游已校验的二十三维当前输入。
    # 输出：
    #   yaw_axis：进入后续完整安全裁决的有界偏航提案。
    def forward(self, features):
        weight = features[:, 9:10]*features[:, 12:13]
        if self.age_weighted:
            weight = weight*torch.clamp(1.-features[:, 11:12], 0., 1.)
        current = self.current(features)
        yaw_axis = torch.clamp(current+weight*(self.temporal(features)-current), -1., 1.)
        return yaw_axis
