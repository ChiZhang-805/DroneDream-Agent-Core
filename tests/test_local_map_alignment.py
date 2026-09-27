import dataclasses

import numpy as np
import pytest

from dronedream_agent_core.local_map_alignment import (
    MapAlignmentLimits,
    MapSurfaceIndex,
)
from dronedream_agent_core.local_map_alignment import (
    fit_map_translation as _fit,
)


def fit_map_translation(points, index, *, sensor_origins_world_m=(0., 0., 0.), **kwargs):
    return _fit(points, index, sensor_origins_world_m=sensor_origins_world_m, **kwargs)


def box(**changes):
    return {"center_x": 2., "center_y": 0., "center_z": 0.,
            "size_x": 2., "size_y": 20., "size_z": 20., **changes}


def rounded(**changes):
    return {"center_x": 0., "center_y": 0., "center_z": 0., "radius_m": 1., **changes}


def wall_points():
    y, z = np.meshgrid(np.linspace(-.8, .8, 12), np.linspace(-.8, .8, 10))
    return np.column_stack((np.ones(y.size), y.ravel(), z.ravel()))


def test_real_solver_recovers_wall_normal_but_does_not_invent_tangent_motion():
    points = wall_points() + [.07, .13, -.11]
    untouched = points.copy()
    result = fit_map_translation(points, MapSurfaceIndex([box()]))
    assert result.usable_candidate
    np.testing.assert_allclose(result.correction_world_m, [-.07, 0, 0], atol=1e-10)
    assert result.observed_translation_rank == 1
    assert len(result.unobserved_directions_world) == 2
    assert result.covariance_qualified is False and result.attitude_held_fixed
    np.testing.assert_array_equal(points, untouched)


def test_floor_and_wall_recovers_two_axes_but_preserves_corridor_direction():
    floor = wall_points()[:, [1, 2, 0]] * [1, 1, -1]
    points = np.concatenate((wall_points(), floor))
    result = fit_map_translation(points + [.06, -.09, .04], MapSurfaceIndex([
        box(), box(center_x=0., center_z=-2., size_x=20., size_z=2.)]))
    assert result.usable_candidate
    np.testing.assert_allclose(result.correction_world_m, [-.06, 0, -.04], atol=1e-9)
    assert result.observed_translation_rank == 2


def test_three_planes_recover_translation_and_duplication_does_not_change_rank():
    plane = wall_points()
    points = np.concatenate((plane, plane[:, [1, 0, 2]], plane[:, [1, 2, 0]]*[1, 1, -1]))
    index = MapSurfaceIndex([box(), box(center_x=0., center_y=2., size_x=20., size_y=2.),
                            box(center_x=0., center_z=-2., size_x=20., size_z=2.)])
    result = fit_map_translation(points + [.06, -.04, .03], index)
    assert result.usable_candidate and result.observed_translation_rank == 3
    np.testing.assert_allclose(result.correction_world_m, [-.06, .04, -.03], atol=1e-9)
    single = fit_map_translation(plane + [.06, 0, 0], MapSurfaceIndex([box()]))
    twice = fit_map_translation(np.repeat(plane + [.06, 0, 0], 2, axis=0),
                                MapSurfaceIndex([box()]))
    np.testing.assert_allclose(single.correction_world_m, twice.correction_world_m)
    assert single.observed_translation_rank == twice.observed_translation_rank


def test_rotation_of_map_and_points_rotates_correction_without_adding_rank():
    angle = .63
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    result = fit_map_translation((wall_points() + [.07, .1, .04]) @ rotation.T,
        MapSurfaceIndex([box(center_x=float(2*np.cos(angle)), center_z=float(-2*np.sin(angle)),
                             pitch_rad=angle)]))
    assert result.usable_candidate and result.observed_translation_rank == 1
    np.testing.assert_allclose(result.correction_world_m, rotation @ [-.07, 0, 0], atol=1e-9)


def test_competing_walls_edges_and_box_interior_do_not_fake_matches():
    limits = MapAlignmentLimits()
    competing = MapSurfaceIndex([box(), box(center_x=0., center_y=2., size_x=20., size_y=2.)])
    corner = wall_points()
    corner[:, :2] = [1.1, 1.1]
    assert not competing.match(corner, limits, sensor_origins_world_m=(0, 0, 0)).valid.any()
    cube = MapSurfaceIndex([box(center_x=0., size_y=2., size_z=2.)])
    match = cube.match([[1.01, 1.01, 0], [.9, .9, 0], [0, 0, 0]], limits,
                       sensor_origins_world_m=[[2, 2, 0], [2, 2, 0], [2, 0, 0]])
    assert not match.valid.any()
    inside = cube.match([[.94, 0, 0]], limits, sensor_origins_world_m=(2, 0, 0))
    assert inside.valid[0]
    assert inside.distances[0] == pytest.approx(.06)  # Not zero when inside solid.
    result = fit_map_translation(corner, competing)
    assert not result.usable_candidate
    np.testing.assert_array_equal(result.correction_world_m, [0, 0, 0])


def test_thin_wall_back_face_and_occluded_wall_are_not_selected():
    points = wall_points()
    index = MapSurfaceIndex([box(center_x=1.07, size_x=.14)])
    for shift in (.03, .11):
        result = fit_map_translation(points + [shift, 0, 0], index,
                                     sensor_origins_world_m=[shift, 0, 0])
        assert result.usable_candidate
        np.testing.assert_allclose(result.correction_world_m, [-shift, 0, 0], atol=1e-9)
    # A ray cannot see the distant wall through a solid foreground wall.
    blocked = MapSurfaceIndex([box(center_x=.5, size_x=.1), box()])
    result = fit_map_translation(points, blocked)
    assert not result.usable_candidate


def test_inside_sensor_zero_ray_and_invalid_origins_rejected():
    index = MapSurfaceIndex([box()])
    assert not fit_map_translation(wall_points(), index,
                                    sensor_origins_world_m=[2., 0., 0.]).usable_candidate
    with pytest.raises(ValueError, match="ZERO_LENGTH_RAY"):
        fit_map_translation([[1., 0., 0.]], index, sensor_origins_world_m=[1., 0., 0.])
    with pytest.raises(ValueError, match="ORIGINS_INVALID"):
        fit_map_translation(wall_points(), index, sensor_origins_world_m=[[0, 0, 0]])


def test_outliers_do_not_control_fit_and_excessive_correction_is_rejected():
    points = wall_points() + [.06, 0, 0]
    points[:6, 0] += .13
    result = fit_map_translation(points, MapSurfaceIndex([box()]))
    assert result.usable_candidate
    assert abs(result.correction_world_m[0] + .06) < .002
    rejected = fit_map_translation(wall_points() + [.22, 0, 0], MapSurfaceIndex([box()]))
    assert not rejected.usable_candidate
    assert rejected.issue == "MAP_ALIGNMENT_CORRECTION_OUTSIDE_ENVELOPE"


def test_clipped_residuals_converge_without_extending_iteration_limit():
    floor = wall_points()[:, [1, 2, 0]] * [1, 1, -1]
    floor[:59, 2] -= .03
    floor[59:, 2] += .05
    result = fit_map_translation(floor, MapSurfaceIndex([
        box(center_x=0., center_z=-2., size_x=20., size_z=2.)]))
    assert result.usable_candidate and result.iterations <= 6
    assert result.correction_world_m[2] == pytest.approx(-(.05-.02*59/61), abs=1e-6)
    coupled = np.concatenate((wall_points() + [.0007, 0, 0], floor))
    fit = fit_map_translation(coupled, MapSurfaceIndex([
        box(), box(center_x=0., center_z=-2., size_x=20., size_z=2.)]))
    assert fit.usable_candidate and fit.iterations <= 6
    np.testing.assert_allclose(fit.correction_world_m,
                               [-.0007, 0., -(.05-.02*59/61)], atol=1e-6)


def test_unmatched_or_oversized_inputs_do_not_produce_identity_success():
    index = MapSurfaceIndex([box()])
    missing = fit_map_translation(wall_points() + [10, 0, 0], index)
    assert not missing.usable_candidate and missing.residual_p95_m is None
    for points in ([[float("nan"), 0, 0]], np.zeros((513, 3)), np.zeros((5, 4)), []):
        with pytest.raises(ValueError, match="POINTS_INVALID"):
            fit_map_translation(points, index)
    dense = MapSurfaceIndex([box() for _ in range(257)])
    with pytest.raises(ValueError, match="LOCAL_MAP_BUDGET"):
        fit_map_translation(wall_points(), dense)


@pytest.mark.parametrize("change", [
    {"maximum_translation_m": True}, {"maximum_iterations": 0},
    {"minimum_normal_eigenvalue": 1.}, {"minimum_matched_fraction": float("nan")},
    {"maximum_translation_m": 10**999}, {"maximum_translation_m": -1},
])
def test_limits_are_validated(change):
    with pytest.raises(ValueError):
        MapAlignmentLimits(**change)


def test_index_owns_geometry_and_rejects_unsupported_shapes():
    primitive = box()
    index = MapSurfaceIndex([primitive])
    primitive["center_x"] = 100
    for unsupported in ({"radius_m": 1}, {"mesh": "unknown"}, box(radius_m=1)):
        with pytest.raises(ValueError, match="MAP_ALIGNMENT"):
            MapSurfaceIndex([box(), unsupported])
    assert index.primitive_counts == {"box": 1, "cylinder": 0, "sphere": 0, "mesh": 0}
    assert fit_map_translation(wall_points(), index).usable_candidate
    with pytest.raises(ValueError):
        index.centers[0, 0] = 2
    fit = fit_map_translation(wall_points(), index)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.covariance_qualified = True


@pytest.mark.parametrize("length_key", ["height_m", "length_m"])
def test_capped_cylinder_has_two_caps_and_a_side(length_key):
    index = MapSurfaceIndex([rounded(**{length_key: 2.})])
    match = index.match([[0, 0, 1.04], [0, 0, -1.04], [1.04, 0, 0], [-1.04, 0, 0]],
                        MapAlignmentLimits(), sensor_origins_world_m=[
                            [0, 0, 3], [0, 0, -3], [3, 0, 0], [-3, 0, 0]])
    assert match.valid.all()
    np.testing.assert_allclose(match.normals, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]])
    np.testing.assert_allclose(match.distances, .04, atol=1e-12)
    assert match.surface_ids[0] != match.surface_ids[1] != match.surface_ids[2]


def test_round_solids_occlude_walls_even_when_their_rim_is_not_a_usable_match():
    for primitive in (rounded(center_x=.5, radius_m=.2),
                      rounded(center_x=.5, radius_m=.2, height_m=2.)):
        index = MapSurfaceIndex([box(), primitive])
        match = index.match([[1, 0, 0]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0, 0])
        assert not match.valid[0]  # The wall is hidden; do not skip the blocker.
        assert match.surface_ids[0] // 6 == 1
        assert not index.match([[1, 0, 0]], MapAlignmentLimits(),
                               sensor_origins_world_m=[.5, 0, 0]).valid[0]
    cylinder = MapSurfaceIndex([rounded(height_m=2.)])
    assert not cylinder.match([[1.01, 0, 1.01]], MapAlignmentLimits(),
                              sensor_origins_world_m=[2, 0, 2]).valid[0]


def test_sphere_hits_and_tangent_rejection_are_finite_without_runtime_warnings():
    index = MapSurfaceIndex([rounded()])
    with np.errstate(all="raise"):
        match = index.match([[1.03, 0, 0], [0, 1, 0], [1.03, 0, 1.1]],
                            MapAlignmentLimits(), sensor_origins_world_m=[
                                [3, 0, 0], [-2, 1, 0], [3, 0, 1.1]])
    assert match.valid.tolist() == [True, False, False]
    np.testing.assert_allclose(match.normals[0], [1, 0, 0])
    assert match.distances[0] == pytest.approx(.03)


def test_rotated_cylinder_caps_are_not_world_axis_aligned():
    angle = .7
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    match = MapSurfaceIndex([rounded(height_m=2., pitch_rad=angle)]).match(
        np.array([[0, 0, 1.04], [1.04, 0, 0]]) @ rotation.T, MapAlignmentLimits(),
        sensor_origins_world_m=np.array([[0, 0, 3], [3, 0, 0]]) @ rotation.T)
    assert match.valid.all()
    np.testing.assert_allclose(match.normals, np.array([[0, 0, 1], [1, 0, 0]]) @ rotation.T,
                               atol=1e-12)
    np.testing.assert_allclose(match.distances, .04, atol=1e-12)


def test_launch_pad_does_not_bias_floor_height_estimate():
    # Reproduce the native capture's missing 8 cm circular launch pad, not an
    # arbitrary outlier tolerance change. Floor and pad provide distinct planes.
    x, y = np.meshgrid(np.linspace(-.4, .4, 8), np.linspace(-.4, .4, 8))
    pad = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.92)))
    floor = np.column_stack((x.ravel()+1.5, y.ravel(), np.full(x.size, -1.)))
    points = np.concatenate((pad, floor))
    shift = np.array([.0, .0, .04])
    index = MapSurfaceIndex([
        box(center_x=0., center_z=-1.5, size_x=20., size_z=1.),
        rounded(center_z=-.96, radius_m=.85, height_m=.08)])
    fit = fit_map_translation(points + shift, index, sensor_origins_world_m=shift)
    assert fit.usable_candidate and fit.observed_translation_rank == 1
    np.testing.assert_allclose(fit.correction_world_m, [0, 0, -.04], atol=1e-8)
    assert fit.residual_p95_m < 1e-8


def test_sphere_fit_reassociates_curved_normals_before_claiming_convergence():
    y, z = np.meshgrid(np.linspace(-.55, .55, 10), np.linspace(-.55, .55, 10))
    points = np.column_stack((2. - np.sqrt(1.-y.ravel()**2-z.ravel()**2),
                              y.ravel(), z.ravel()))
    shift = np.array([.04, .02, -.03])
    result = fit_map_translation(points + shift, MapSurfaceIndex([rounded(center_x=2.)]),
                                 sensor_origins_world_m=shift)
    assert result.usable_candidate and result.observed_translation_rank == 3
    np.testing.assert_allclose(result.correction_world_m, -shift, atol=1e-6)
    assert result.residual_p95_m < 1e-6


@pytest.mark.parametrize("changes", [
    {"height_m": 1., "length_m": 2.}, {"height_m": 0.}, {"length_m": float("nan")},
    {"radius_m": True}, {"radius_m": 10**999}, {"radius_m": -.1}, {"size_x": 2.},
])
def test_invalid_round_geometry_cannot_be_silently_ignored(changes):
    with pytest.raises(ValueError, match="MAP_ALIGNMENT"):
        MapSurfaceIndex([box(), rounded(**changes)])


def test_boundary_admission_cannot_oscillate_within_a_single_solve():
    points = wall_points()[:, [1, 2, 0]] * [1, 1, -1] + [0, 0, .04]
    points[0, 2] += .05

    class BoundaryProbe(MapSurfaceIndex):
        # Isolate the active-set failure observed at a circular pad's rim:
        # keeping one point moves the solution across its admissibility edge;
        # dropping it moves the solution back. The real kernels are tested above.
        def match(self, value, limits, **kwargs):
            match = super().match(value, limits, **kwargs)
            valid = match.valid.copy()
            valid[0] &= value[1, 2] - points[1, 2] > -.04008
            return dataclasses.replace(match, valid=valid)

    index = BoundaryProbe([box(center_x=0., center_z=-2., size_x=20., size_z=2.)])
    result = fit_map_translation(points, index)
    assert result.usable_candidate and result.retired_correspondence_count == 1
    assert result.matched_count == 119 and result.iterations <= 5
    np.testing.assert_allclose(result.correction_world_m, [0, 0, -.04], atol=1e-8)
