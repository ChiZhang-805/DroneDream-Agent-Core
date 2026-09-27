"""Offline-only balancing and modality dropout, independent of deployed feature contracts."""

import math

import torch
from pydantic import Field

from ..contracts import StrictModel


class CausalRegularization(StrictModel):
    """Training choices recorded separately from the unchanged network/export architecture."""

    balance_mission_groups: bool = Field(default=False, strict=True)
    visual_block_dropout: float = Field(default=0.0, ge=0.0, le=0.8, strict=True)


# 功能：
#   复制并重验离线正则化配置，无视觉网络不能要求视觉屏蔽；不改变部署输入契约。
# 输入：
#   value：显式训练配置，None 表示完全不启用新增训练策略。
#   visual_feature_count：网络的视觉输入维数。
# 输出：
#   config：与调用方不共享的已验证配置。
def validate_regularization(value, visual_feature_count: int) -> CausalRegularization:
    if value is not None and not isinstance(value, CausalRegularization):
        raise ValueError("CAUSAL_REGULARIZATION_CONFIG_INVALID")
    if type(visual_feature_count) is not int or not 0 <= visual_feature_count <= 65536:
        raise ValueError("CAUSAL_REGULARIZATION_VISUAL_WIDTH_INVALID")
    config = CausalRegularization() if value is None else CausalRegularization.model_validate(
        value.model_dump(mode="python"), strict=True)
    if config.visual_block_dropout and not visual_feature_count:
        raise ValueError("CAUSAL_REGULARIZATION_REQUIRES_VISUAL_INPUT")
    return config


# 功能：
#   1. 生成有效 float32 样本权重，可让每条训练路线拥有相同总权重且保留组内质量权重比例。
#   2. 不修改原始记录，不重标记动作，不借用验证或测试数据。
# 输入：
#   examples：已验证的训练窗口。
#   balance_groups：是否均衡路线总权重。
# 输出：
#   weights：与窗口一一对应的正有限权重张量。
def effective_sample_weights(examples, *, balance_groups: bool) -> torch.Tensor:
    if type(balance_groups) is not bool or not examples:
        raise ValueError("CAUSAL_TRAINING_WEIGHTS_INVALID")
    raw = [example.sample.sample_weight for example in examples]
    if any(type(value) not in (int, float) or not 0 < value <= 100 for value in raw):
        raise ValueError("CAUSAL_TRAINING_WEIGHTS_INVALID")
    masses = {}
    for example, weight in zip(examples, raw, strict=True):
        if not isinstance(example.group_id, str) or not example.group_id:
            raise ValueError("CAUSAL_TRAINING_WEIGHT_GROUP_INVALID")
        masses[example.group_id] = masses.get(example.group_id, 0.0) + weight
    if balance_groups:
        group_mass = math.fsum(raw) / len(masses)
        values = [weight / masses[example.group_id] * group_mass
                  for example, weight in zip(examples, raw, strict=True)]
    else:
        values = raw
    weights = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("CAUSAL_TRAINING_WEIGHTS_NOT_REPRESENTABLE")
    return weights


# 功能：
#   按训练随机流屏蔽部分样本的整块视觉输入，避免只记场景外观；不改原张量或运行期传感器门禁。
# 输入：
#   inputs：一个已选取训练批次的输入元组，启用屏蔽时末项必须为二维视觉张量。
#   probability：零到零点八的屏蔽概率；零时不消耗随机数。
#   generator：独立、可复现的 CPU 训练随机流。
# 输出：
#   augmented：仅改变训练计算输入的张量元组。
def drop_visual_block(inputs: tuple, probability: float, generator: torch.Generator) -> tuple:
    if type(probability) not in (int, float) or not 0 <= probability <= 0.8:
        raise ValueError("CAUSAL_VISUAL_DROPOUT_PROBABILITY_INVALID")
    if not probability:
        return inputs
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("CAUSAL_VISUAL_DROPOUT_GENERATOR_INVALID")
    if (len(inputs) != 6 or inputs[-1].ndim != 2 or inputs[-1].device.type != "cpu"
            or not inputs[-1].shape[1]):
        raise ValueError("CAUSAL_VISUAL_DROPOUT_INPUT_INVALID")
    keep = torch.rand((inputs[-1].shape[0], 1), generator=generator) >= probability
    # 不做反向概率放大，保留视觉特征本来的数值范围；部署时仍使用完整的真实视觉编码。
    augmented = (*inputs[:-1], inputs[-1] * keep.to(dtype=inputs[-1].dtype))
    return augmented
