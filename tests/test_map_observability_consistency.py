"""Position information must not resurrect modes rejected by the pose solver."""

import numpy as np
import pytest

from dronedream_agent_core.local_pose_alignment import (
    MapPoseAlignmentLimits,
    _observable,
    _translation_observability,
)


# 功能：构造位置与姿态耦合、强弱方向已知的六维问题，不使用任何场景真值拟合阈值。
# 输入：weak：弱方向能量；mixed：是否让弱方向同时含平移与转动。
# 输出：具有指定谱的雅可比及归一化权重。
def _problem(weak, mixed=True):
    basis = np.eye(6)
    if mixed:
        angle = .15
        basis[0, 0], basis[0, 5] = np.cos(angle), -np.sin(angle)
        basis[5, 0], basis[5, 5] = np.sin(angle), np.cos(angle)
    eigenvalues = np.array([weak, .1, .2, .3, .4, 20.])
    return np.sqrt(6 * eigenvalues[:, None]) * basis.T, np.full(6, 1/6)


# 功能：回归门口弱约束被六维求解丢弃、却被位置协方差重新识别为有效的错误。
# 输入：无。
# 输出：无；位置未知方向必须与实际求解器的未知方向一致。
def test_weak_mixed_pose_mode_cannot_reappear_as_position_information():
    jacobian, weights = _problem(.0003)
    limits = MapPoseAlignmentLimits()
    eigenvalues, _, null = _observable(jacobian, weights, limits)
    assert len(eigenvalues) == 5
    rank, _, information = _translation_observability(jacobian, weights, limits)
    assert rank == 2
    assert np.linalg.norm(information @ null[0, :3]) < 1e-10


# 功能：覆盖完整可观测及只有转动方向缺失的情况，不能把所有五秩问题一律降为二维位置。
# 输入：无。
# 输出：无。
def test_full_pose_and_rotation_only_degeneracy_keep_valid_position_information():
    jacobian, weights = _problem(.01)
    rank, _, _ = _translation_observability(jacobian, weights, MapPoseAlignmentLimits())
    assert rank == 3
    jacobian = np.diag(np.sqrt(6 * np.array([.1, .2, .3, .4, .5, .000001])))
    rank, _, _ = _translation_observability(jacobian, weights, MapPoseAlignmentLimits())
    assert rank == 3


# 功能：检查多种旋转坐标基下，所有被丢弃的平移分量都不能携带定位信息。
# 输入：seed：固定的可复现基变换种子。
# 输出：无。
@pytest.mark.parametrize("seed", range(12))
def test_marginal_information_annihilates_every_rejected_pose_translation(seed):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    jacobian = np.sqrt(6*np.array([1e-7, .01, .03, .1, .3, 1.]))[:, None] * basis.T
    weights = np.full(6, 1/6)
    limits = MapPoseAlignmentLimits()
    _, _, null = _observable(jacobian, weights, limits)
    rank, _, information = _translation_observability(jacobian, weights, limits)
    assert rank <= 2
    assert np.linalg.norm(information @ null[:, :3].T) < 1e-10

