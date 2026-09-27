import hashlib
import json

import numpy as np
import pytest
from test_localization_observations import _publish, _scan

from dronedream_agent_core.geometry_alignment_audit import (
    analyze_geometry_capture,
    nearest_box_surfaces,
)
from dronedream_agent_core.localization_observations import GeometryObservationCapture


def _box(**changes):
    return {"center_x": 0., "center_y": 0., "center_z": 0.,
            "size_x": 2., "size_y": 4., "size_z": 6., **changes}


def test_box_interior_is_not_a_perfect_surface_match():
    distances, normals = nearest_box_surfaces([[0, 0, 0], [2, 0, 0], [1, 2, 3]], [_box()])
    np.testing.assert_allclose(distances, [1, 1, 0])
    np.testing.assert_allclose(normals, [[1, 0, 0]] * 3)


def test_box_normals_and_distance_rotate_with_map_geometry():
    distances, normals = nearest_box_surfaces([[0, 2, 0]], [_box(yaw_rad=np.pi / 2)])
    np.testing.assert_allclose(distances, [1])
    np.testing.assert_allclose(normals, [[0, 1, 0]], atol=1e-12)
    with pytest.raises(ValueError, match="INVALID_OR_TILTED"):
        nearest_box_surfaces([[0, 2, 0]], [_box(pitch_rad=.5)])


def test_actual_capture_replays_calibration_but_does_not_qualify_covariance(tmp_path):
    semantic = tmp_path / "map.json"
    semantic.write_text(json.dumps({"collision_primitives": [
        _box(center_x=2.04, size_x=.08, size_y=20., size_z=20.)]}), encoding="utf-8")
    capture = GeometryObservationCapture(tmp_path,
        map_sha256=hashlib.sha256(semantic.read_bytes()).hexdigest(), summary_publisher=_publish)
    scan, kwargs = _scan()
    assert capture.record(scan, **kwargs)
    assert capture.close()["complete"]
    world = tmp_path / 'world.sdf'
    world.write_text('<sdf><world name="w"><model name="m"><static>true</static><link name="l">'
                     '<visual name="wall"><pose>2.04 0 0 0 0 0</pose><geometry><box>'
                     '<size>.08 20 20</size></box></geometry></visual></link></model></world></sdf>')
    optical = dict(optical_world=world, expected_world_sha256=hashlib.sha256(world.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='BOUND_OPTICAL_WORLD_REQUIRED'):
        analyze_geometry_capture(capture.path, semantic, fit_translation=True)
    result = analyze_geometry_capture(capture.path, semantic, fit_translation=True,
                                       check_perturbations=True, **optical)
    assert result["frame_count"] == 1
    assert result["frames"][0]["hit_count"] == 300
    assert result["frames"][0]["translation_normal_gram_eigenvalues"][0] < 1e-9
    assert len(result["frames"][0]["translation_unobserved_directions_world_enu"]) >= 1
    assert result["pose_correction_applied"] is False
    assert result["covariance_qualification_granted"] is False
    assert result["model_control_qualification_granted"] is False
    fit = result["frames"][0]["translation_fit"]
    assert fit["usable_candidate"] and fit["observed_translation_rank"] == 1
    assert fit["covariance_qualified"] is False
    assert abs(fit["correction_world_m"][0] + .13233) < .005
    perturbations = result["frames"][0]["same_frame_translation_perturbations"]
    assert len(perturbations) == 6
    assert all(p["candidate"]["usable_candidate"] for p in perturbations)
    assert max(p["observed_shift_recovery_error_m"] for p in perturbations) < .00001
    assert max(p["unobserved_correction_change_m"] for p in perturbations) == 0
    joint = analyze_geometry_capture(capture.path, semantic, fit_joint_pose=True,
                                     check_perturbations=True, **optical)
    assert joint["solver"] == "joint-position-attitude"
    assert "translation_fit" not in joint["frames"][0]
    assert joint["frames"][0]["pose_fit"]["usable_candidate"]
    assert joint["frames"][0]["pose_fit"]["observed_pose_rank"] == 3
    assert not joint["frames"][0]["pose_fit"]["motion_permission_granted"]
    with pytest.raises(ValueError, match="SINGLE_SOLVER"):
        analyze_geometry_capture(capture.path, semantic, fit_translation=True, fit_joint_pose=True)
    semantic.write_text(semantic.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        analyze_geometry_capture(capture.path, semantic)
