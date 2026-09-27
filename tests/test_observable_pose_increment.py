"""Changing nullspaces must not turn a truncated Newton update uphill."""
import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.local_pose_alignment import _observable_proposal, fit_map_pose


# 功能：在任意旋转弱约束基下验证增量是下降方向，不能通过重置未知分量改变已有解。
# 输入：seed：固定随机种子；rank：保留的可观测方向数。
# 输出：无；核对下降恒等式及零空间没有收到伪造的更新。
@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('rank', [1, 3, 5, 6])
def test_observable_increment_is_descent_and_preserves_unknown_components(seed, rank):
    rng = np.random.default_rng(seed)
    axes, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    basis = axes[:, :rank]
    null = axes[:, rank:]
    eigenvalues = np.linspace(.02, 10, rank)
    state, gradient = rng.normal(size=(2, 6))
    proposal = _observable_proposal(state, gradient, eigenvalues, basis)
    delta = proposal-state
    assert float(gradient @ delta) == pytest.approx(
        -float(np.sum((basis.T @ gradient)**2/eigenvalues)), abs=1e-10)
    np.testing.assert_allclose(null.T @ proposal, null.T @ state, atol=1e-13)


# 功能：复现旧总状态投影导致上升的最小例子，确保不是只改变报错文案。
# 输入：无；强方向已收敛，弱方向仍有梯度且当前状态非零。
# 输出：无；旧公式上升，新公式保持已收敛强方向和未知猜测。
def test_total_projection_would_increase_cost_but_increment_does_not():
    state = np.array([0., -.05, 0., 0., 0., 0.])
    gradient = np.array([0., .001, 0., 0., 0., 0.])
    basis = np.eye(6)[:, [0, 2, 3, 4, 5]]
    values = np.ones(5)
    old = basis @ (basis.T @ state - (basis.T @ gradient)/values)
    assert gradient @ (old-state) > 0
    np.testing.assert_array_equal(_observable_proposal(state, gradient, values, basis), state)


# 功能：两面场景无法识别沿墙位置，初始猜测不能变成虚假的三维定位资格。
# 输入：无；固定同场景和三个不同的不可观测方向初始猜测。
# 输出：无；可观测方向恢复，不可观测方向仍明确缺失且信息为零。
@pytest.mark.parametrize('unknown_guess', [-.06, 0., .06])
def test_partial_pose_keeps_unknown_initial_guess_explicit(unknown_guess):
    points, index = scene(planes=2)
    bias = np.array([.04, -.03, .02])
    fit = fit_map_pose(points+bias, index, sensor_origins_world_m=bias,
        reference_position_world_m=bias, initial_correction_world_m=(0., unknown_guess, 0.),
        initial_rotation_vector_world_rad=(0., 0., 0.))
    assert fit.usable_candidate
    assert fit.observed_translation_rank == 2
    assert fit.correction_world_m[1] == pytest.approx(unknown_guess, abs=1e-12)
    np.testing.assert_allclose(np.asarray(fit.translation_information_shape) @ [0., 1., 0.],
                               0., atol=1e-12)
    assert not fit.covariance_qualified and not fit.motion_permission_granted
