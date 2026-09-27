"""A teacher can supplement labels, never mutate them or receive student gradients."""

import pytest
import torch

from dronedream_agent_core.laya_uav_training import set_cross_entropy
from scripts.train_decision_student import student_loss


# 功能：关闭教师时严格等价监督基线；有教师时梯度只流向学生；输出：原标签不变。
def test_distillation_is_auxiliary_and_detached():
    logits = torch.zeros((1, 5), requires_grad=True)
    accepted = torch.tensor([[True, False, False, False, False]])
    original = accepted.clone()
    assert torch.equal(student_loss(logits, accepted), set_cross_entropy(logits, accepted).mean())
    teacher = torch.tensor([[0., 0., 0., 0., 1.]], requires_grad=True)
    student_loss(logits, accepted, teacher).backward()
    assert teacher.grad is None and torch.equal(accepted, original)
    assert logits.grad[0, 0] < 0  # 即使教师错，真实标签仍提供增加正确选项的梯度。


# 功能：拒绝损坏教师概率，不能把 NaN/负值/未归一化结果送入训练；输出：明确异常。
@pytest.mark.parametrize("values", [[float("nan"), 0., 0., 0., 1.],
                                  [-.1, .1, 0., 0., 1.], [.1, .1, .1, .1, .1]])
def test_invalid_teacher_distribution(values):
    with pytest.raises(ValueError, match="TEACHER_PROBABILITIES_INVALID"):
        student_loss(torch.zeros((1, 5)), torch.ones((1, 5), dtype=torch.bool),
                     torch.tensor([values]))
