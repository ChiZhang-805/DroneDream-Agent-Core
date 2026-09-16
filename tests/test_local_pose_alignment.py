import dataclasses

import numpy as np
import pytest

from dronedream_agent_core.local_map_alignment import MapSurfaceIndex
from dronedream_agent_core.local_pose_alignment import (
    MapPoseAlignmentLimits,
    _transform_and_jacobian,
    fit_map_pose,
    rotation_exp_and_left_jacobian,
)


def scene(planes=3, offset=(0., 0., 0.)):
    """Analytic visible planes; numerical unit tests, not simulated flight."""
    y, z = np.meshgrid(np.linspace(-.7, .7, 9), np.linspace(-.7, .7, 9))
    wall = np.column_stack((np.ones(y.size), y.ravel(), z.ravel()))
    cloud = (wall, wall[:, [1, 2, 0]]*[1, 1, -1], wall[:, [1, 0, 2]])
    sizes = [(2., 20., 20.), (20., 20., 2.), (20., 2., 20.)]
    centers = [(2., 0., 0.), (0., 0., -2.), (0., 2., 0.)]
    index = MapSurfaceIndex([{**{f"center_{a}": float(v+offset[i])
                                for i, (a, v) in enumerate(zip("xyz", center, strict=True))},
                             **{f"size_{a}": v for a, v in zip("xyz", size, strict=True)}}
                            for center, size in zip(centers[:planes], sizes[:planes], strict=True)])
    return np.concatenate(cloud[:planes])+offset, index


@pytest.mark.parametrize("angle", [0., 1e-10, 1e-5, .03, .1])
def test_rotation_jacobian_matches_central_difference(angle):
    relative = np.array([[.4, -.3, 2.], [1., 2., -.5]])
    normals = np.array([[1., 0., 0.], [0., 0., 1.]])
    state = np.array([.02, -.03, .04, angle, -angle*.3, angle*.7])
    position, jacobian, rotation = _transform_and_jacobian(relative, np.array([20., -30., 40.]),
                                                         state, normals, 2.)
    assert position.shape == (2, 3)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    for axis in range(6):
        delta = np.eye(6)[axis]*1e-6
        high = _transform_and_jacobian(relative, np.array([20., -30., 40.]),
                                       state+delta, normals, 2.)[0]
        low = _transform_and_jacobian(relative, np.array([20., -30., 40.]),
                                      state-delta, normals, 2.)[0]
        numeric = np.einsum("ni,ni->n", (high-low)/2e-6, normals)
        np.testing.assert_allclose(jacobian[:, axis], numeric, atol=4e-9)


@pytest.mark.parametrize("planes,rank,translation_rank", [(1, 3, 1), (2, 5, 2), (3, 6, 3)])
def test_joint_fit_preserves_unobservable_directions(planes, rank, translation_rank):
    points, index = scene(planes)
    bias = np.array([.04, -.03, .02])
    rotation = rotation_exp_and_left_jacobian([.005, -.004, .009])[0]
    observed = points @ rotation.T + bias
    unchanged = observed.copy()
    fit = fit_map_pose(observed, index, sensor_origins_world_m=bias,
                       reference_position_world_m=bias)
    assert fit.usable_candidate, fit
    assert fit.observed_pose_rank == rank
    assert fit.observed_translation_rank == translation_rank
    assert fit.iterations <= 12
    correction = np.array(fit.correction_world_m)
    if planes > 1:
        np.testing.assert_allclose(np.array(fit.rotation_world_from_input) @ rotation,
                                   np.eye(3), atol=1e-6)
        np.testing.assert_allclose(correction[[0, 2]], -bias[[0, 2]], atol=1e-6)
    if planes == 3:
        np.testing.assert_allclose(correction, -bias, atol=1e-6)
    else:
        assert abs(correction[1]) < 1e-6
    assert not fit.covariance_qualified and not fit.motion_permission_granted
    np.testing.assert_array_equal(observed, unchanged)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.motion_permission_granted = True


def test_rotation_is_about_body_reference_not_distant_world_origin():
    anchor = np.array([-42.25, 15.3, 8.15])
    points, index = scene(offset=anchor)
    rotation = rotation_exp_and_left_jacobian([-.01, .012, -.015])[0]
    bias = np.array([.07, -.02, .04])
    reference = anchor+bias
    observed = reference+(points-anchor) @ rotation.T
    fit = fit_map_pose(observed, index, sensor_origins_world_m=reference+[.13, 0, .03],
                       reference_position_world_m=reference)
    assert fit.usable_candidate, fit
    np.testing.assert_allclose(fit.correction_world_m, -bias, atol=1e-6)
    np.testing.assert_allclose(np.array(fit.rotation_world_from_input) @ rotation,
                               np.eye(3), atol=1e-6)


def test_duplicate_points_do_not_create_information_or_change_rank():
    points, index = scene(1)
    first = fit_map_pose(points+[.05, 0, 0], index,
                         sensor_origins_world_m=[.05, 0, 0], reference_position_world_m=[.05, 0, 0])
    twice = fit_map_pose(np.repeat(points+[.05, 0, 0], 2, axis=0), index,
                         sensor_origins_world_m=[.05, 0, 0], reference_position_world_m=[.05, 0, 0])
    assert first.usable_candidate and twice.usable_candidate
    assert first.observed_pose_rank == twice.observed_pose_rank == 3
    np.testing.assert_allclose(first.correction_world_m, twice.correction_world_m, atol=1e-9)


@pytest.mark.parametrize("changes", [{"maximum_iterations": True}, {"maximum_iterations": 25},
    {"maximum_rotation_rad": float("nan")}, {"rotation_length_scale_m": 0},
    {"relative_eigenvalue_threshold": True}, {"geometry": {}},
    {"minimum_scaled_eigenvalue": 10**999}])
def test_invalid_limits_rejected(changes):
    with pytest.raises(ValueError):
        MapPoseAlignmentLimits(**changes)


def test_bounds_and_missing_visibility_reject_without_identity_success():
    points, index = scene(3)
    for bias, issue in (([.22, 0, 0], "OUTSIDE_ENVELOPE"), ([3., 0, 0], "CORRESPONDENCES")):
        fit = fit_map_pose(points+bias, index, sensor_origins_world_m=bias,
                           reference_position_world_m=bias)
        assert not fit.usable_candidate and issue in fit.issue
        np.testing.assert_array_equal(fit.correction_world_m, [0, 0, 0])
        np.testing.assert_array_equal(fit.rotation_world_from_input, np.eye(3))
    with pytest.raises(ValueError):
        fit_map_pose(points, index, sensor_origins_world_m=[0, 0, 0],
                     reference_position_world_m=[float("nan"), 0, 0])
    with pytest.raises(ValueError):
        fit_map_pose(points, index, sensor_origins_world_m=[0, 0],
                     reference_position_world_m=[0, 0, 0])


def test_iteration_exhaustion_is_not_a_successful_partial_update():
    points, index = scene(3)
    fit = fit_map_pose(points+[.05, -.03, .02], index, sensor_origins_world_m=[.05, -.03, .02],
        reference_position_world_m=[.05, -.03, .02],
        limits=MapPoseAlignmentLimits(maximum_iterations=1))
    assert fit.issue == "MAP_POSE_DID_NOT_CONVERGE"
    assert not fit.usable_candidate


@pytest.mark.parametrize("seed", range(8))
def test_noisy_three_plane_input_is_not_just_an_exact_arithmetic_demo(seed):
    points, index = scene(3)
    rng = np.random.default_rng(seed)
    bias = rng.uniform(-.06, .06, 3)
    angle = rng.uniform(-.015, .015, 3)
    rotation = rotation_exp_and_left_jacobian(angle)[0]
    # Synthetic radial noise: 10 mm standard deviation, clipped at 30 mm.
    # This is a stated robustness case, NOT calibrated camera noise evidence.
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    corrupted = points + rays*np.clip(rng.normal(0., .01, len(points)), -.03, .03)[:, None]
    fit = fit_map_pose(corrupted @ rotation.T + bias, index,
                       sensor_origins_world_m=bias, reference_position_world_m=bias)
    assert fit.usable_candidate, fit
    assert np.linalg.norm(np.asarray(fit.correction_world_m)+bias) < .006
    remainder = np.asarray(fit.rotation_world_from_input) @ rotation
    assert np.linalg.norm(remainder-np.eye(3)) < .012
    assert not fit.covariance_qualified


def test_systematic_sensor_bias_is_not_misrepresented_as_independent_noise():
    points, index = scene(3)
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    fit = fit_map_pose(points+rays*.02, index, sensor_origins_world_m=[0, 0, 0],
                       reference_position_world_m=[0, 0, 0])
    assert fit.usable_candidate, fit
    # A coherent sensor bias produces a coherent pose bias. Even many points
    # and a small residual cannot justify averaging this error away.
    assert np.linalg.norm(fit.correction_world_m) > .015
    assert not fit.covariance_qualified and not fit.motion_permission_granted


def test_sphere_couples_position_and_rotation_not_six_observable_axes():
    y, z = np.meshgrid(np.linspace(-.35, .35, 10), np.linspace(-.35, .35, 10))
    points = np.column_stack((2.-np.sqrt(1-y.ravel()**2-z.ravel()**2), y.ravel(), z.ravel()))
    index = MapSurfaceIndex([{"center_x": 2., "center_y": 0., "center_z": 0., "radius_m": 1.}])
    fit = fit_map_pose(points, index, sensor_origins_world_m=[0, 0, 0],
                       reference_position_world_m=[0, 0, 0])
    assert fit.usable_candidate
    assert fit.observed_pose_rank == 3
    # Camera bearing and orientation about its origin are correlated: only
    # range to the sphere centre is translation-observable without an attitude prior.
    assert fit.observed_translation_rank == 1


def test_sensor_occlusion_and_rotation_limit_do_not_fall_back_to_translation():
    points, index = scene(3)
    rotation = rotation_exp_and_left_jacobian([0., .05, 0.])[0]
    fit = fit_map_pose(points @ rotation.T, index, sensor_origins_world_m=[0, 0, 0],
        reference_position_world_m=[0, 0, 0],
        limits=MapPoseAlignmentLimits(maximum_rotation_rad=.02))
    assert not fit.usable_candidate and fit.issue == "MAP_POSE_CORRECTION_OUTSIDE_ENVELOPE"
    hidden = MapSurfaceIndex([{"center_x": .5, "center_y": 0., "center_z": 0.,
                              "size_x": .1, "size_y": 20., "size_z": 20.}])
    fit = fit_map_pose(points[:81], hidden, sensor_origins_world_m=[0, 0, 0],
                       reference_position_world_m=[0, 0, 0])
    assert not fit.usable_candidate
