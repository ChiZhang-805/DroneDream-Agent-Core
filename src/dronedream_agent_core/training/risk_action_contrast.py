"""Train action ordering within one observation, never across unrelated scenes."""

import math
import re

import numpy as np


# 功能：
#   按观测身份保留完整动作组，验证同组只有动作发生变化，不把不同场景当成动作对照。
# 输入：
#   inputs：未归一化的状态及末尾四轴动作矩阵。
#   group_ids：逐样本原始观测 SHA256；weight：显式对照损失权重。
# 输出：
#   groups：同观测的样本下标数组；关闭对照训练时为 None。
def prepare_action_groups(inputs, group_ids, weight):
    if (type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 4):
        raise ValueError('ACTION_CONTRAST_WEIGHT_INVALID')
    if weight == 0:
        if group_ids is not None:
            raise ValueError('ACTION_CONTRAST_DISABLED_WITH_GROUPS')
        return None
    if (type(group_ids) is not list or len(group_ids) != len(inputs)
            or inputs.ndim != 2 or inputs.shape[1] < 5 or not np.isfinite(inputs).all()):
        raise ValueError('ACTION_CONTRAST_GROUPS_INVALID')
    grouped = {}
    for index, identity in enumerate(group_ids):
        if type(identity) is not str or re.fullmatch(r'[0-9a-f]{64}', identity) is None:
            raise ValueError('ACTION_CONTRAST_OBSERVATION_ID_INVALID')
        grouped.setdefault(identity, []).append(index)
    groups = []
    for indices in grouped.values():
        if not 2 <= len(indices) <= 256:
            raise ValueError('ACTION_CONTRAST_GROUP_SIZE_INVALID')
        indices = np.asarray(indices, dtype=np.int64)
        if not np.all(inputs[indices, :-4] == inputs[indices[0], :-4]):
            raise ValueError('ACTION_CONTRAST_OBSERVATION_CHANGED')
        groups.append(indices)
    return groups


# 功能：
#   打乱整组而不拆开同观测的动作候选；单组大于批大小时只允许该完整组单独运行。
# 输入：
#   groups：已验证的动作组；batch_size：期望批大小；generator：训练随机数发生器。
# 输出：
#   batches：完整动作组组成的样本下标及各组在本批的起止下标。
def action_group_batches(groups, batch_size, generator):
    pending, bounds, size = [], [], 0
    for group_index in generator.permutation(len(groups)):
        indices = groups[group_index]
        if pending and size + len(indices) > batch_size:
            yield np.concatenate(pending), bounds
            pending, bounds, size = [], [], 0
        pending.append(indices)
        bounds.append((size, size + len(indices)))
        size += len(indices)
    if pending:
        yield np.concatenate(pending), bounds


# 功能：
#   对同观测中最危险与最安全动作的最差排序施加间隔损失，原始标签及分类阈值不变。
# 输入：
#   probabilities：sigmoid 风险输出；targets：原始教师风险标签。
#   bounds：本批完整组边界；weight：损失权重；margin：期望风险分数间隔。
# 输出：
#   loss：平均对照损失；delta：该损失对 sigmoid 前 logits 的梯度。
def action_contrast_loss(probabilities, targets, bounds, weight, margin=.1):
    delta = np.zeros_like(probabilities)
    loss, active = 0., 0
    for start, end in bounds:
        labels = targets[start:end, 0]
        if float(labels.max() - labels.min()) < .1:
            continue
        active += 1
        values = probabilities[start:end, 0]
        highest = np.flatnonzero(labels == labels.max())
        lowest = np.flatnonzero(labels == labels.min())
        # 全部极值并列动作都要满足排序；不挑最好的一对掩盖其他候选。
        high = start + highest[np.argmin(values[highest])]
        low = start + lowest[np.argmax(values[lowest])]
        deficit = max(0., margin - float(probabilities[high, 0] - probabilities[low, 0]))
        loss += weight * deficit * deficit
        for index, sign in ((high, -1.), (low, 1.)):
            probability = probabilities[index, 0]
            delta[index, 0] += sign * 2 * weight * deficit * probability * (1 - probability)
    divisor = max(1, active)
    return loss / divisor, delta / divisor
