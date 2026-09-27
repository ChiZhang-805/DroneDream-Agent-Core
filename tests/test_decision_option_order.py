"""Training option shuffles preserve action identity and deterministic resumes."""

import pytest
import torch

from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.laya_uav_training import (
    acceptable_mask,
    set_cross_entropy,
    training_option_order,
)


# 功能：选项换序同时移动标签，损失不因展示顺序发生改变；输入：多轮排列；输出：等价。
def test_permuted_targets_preserve_supervision():
    logits = torch.tensor([[.1, .5, -1., 1., .7]])
    labels = ["slow_down", "wait"]
    baseline = set_cross_entropy(logits, torch.tensor([acceptable_mask(labels, list(ACTIONS))]))
    orders = set()
    for epoch in range(8):
        order = training_option_order("sample-a", 924, epoch)
        assert order == training_option_order("sample-a", 924, epoch)
        indices = [ACTIONS.index(a) for a in order]
        loss = set_cross_entropy(logits[:, indices], torch.tensor([acceptable_mask(labels, order)]))
        assert torch.allclose(loss, baseline)
        orders.add(order)
    assert len(orders) > 1


# 功能：拒绝用重复/遗漏选项或重复标签伪装为有效映射；输入：异常排列；输出：抛错。
@pytest.mark.parametrize("labels,order", [(["wait", "wait"], ACTIONS),
                                        (["wait"], (*ACTIONS, "wait")),
                                        (["wait"], ACTIONS[:-1])])
def test_reject_ambiguous_mapping(labels, order):
    with pytest.raises(ValueError, match="LABEL_OR_ORDER_INVALID"):
        acceptable_mask(labels, order)
