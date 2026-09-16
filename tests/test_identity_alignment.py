from __future__ import annotations

import pytest

from dronedream_agent_core.contracts import ControlledVehicleIdentityEvidence, Vector3
from dronedream_agent_core.identity_alignment import (
    align_gazebo_px4_identity,
    dynamic_identity_disagreement_limit_m,
    identity_offset_innovation_m,
)


def test_identity_alignment_removes_only_bounded_along_track_latency() -> None:
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(x=46.3829, y=0.9246, z=1.4015),
        px4_position_world_enu_m=Vector3(x=46.1017, y=0.8983, z=1.3666),
        px4_velocity_ned_mps=Vector3(x=-0.0125, y=-0.7088, z=-0.0314),
    )

    assert result.raw_disagreement_m == pytest.approx(0.2846, abs=0.001)
    assert 0.35 < result.alignment_seconds < 0.45
    assert result.time_aligned_disagreement_m < 0.06
    assert result.estimator_offset_m.x == pytest.approx(0.0, abs=0.01)


def test_identity_alignment_preserves_perpendicular_disagreement() -> None:
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(x=0.0, y=0.3, z=1.0),
        px4_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        px4_velocity_ned_mps=Vector3(x=0.0, y=1.0, z=0.0),
    )

    assert result.alignment_seconds == pytest.approx(0.0)
    assert result.time_aligned_disagreement_m == pytest.approx(0.3)


def test_identity_alignment_does_not_project_a_static_sample() -> None:
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(x=0.3, y=0.0, z=1.0),
        px4_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        px4_velocity_ned_mps=Vector3(x=0.0, y=0.1, z=0.0),
    )

    assert result.alignment_seconds == pytest.approx(0.0)
    assert result.time_aligned_disagreement_m == pytest.approx(0.3)


def test_slow_precision_identity_alignment_removes_only_bounded_latency() -> None:
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(
            x=-15.354830754812939,
            y=4.733111035371373,
            z=2.985013508268464,
        ),
        px4_position_world_enu_m=Vector3(
            x=-15.095362167805433,
            y=4.712419456755743,
            z=3.050726737976074,
        ),
        px4_velocity_ned_mps=Vector3(
            x=-0.019511643797159195,
            y=0.07827882468700409,
            z=-0.007199771702289581,
        ),
        minimum_alignment_speed_mps=0.05,
    )

    assert result.raw_disagreement_m == pytest.approx(0.26846, abs=0.0001)
    assert result.alignment_seconds == pytest.approx(0.5)
    assert result.time_aligned_disagreement_m < 0.25


def test_identity_alignment_never_exceeds_freshness_window() -> None:
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(x=2.0, y=0.0, z=1.0),
        px4_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
        px4_velocity_ned_mps=Vector3(x=0.0, y=-1.0, z=0.0),
    )

    assert result.alignment_seconds == pytest.approx(0.5)
    assert result.time_aligned_disagreement_m == pytest.approx(1.5)


def test_dynamic_identity_limit_covers_observed_turning_transport_residual() -> None:
    velocity_ned = Vector3(
        x=-0.7770805954933167,
        y=0.5770831108093262,
        z=-0.515705943107605,
    )
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(
            x=-42.335130553028115,
            y=12.090984321922532,
            z=7.763146361515081,
        ),
        px4_position_world_enu_m=Vector3(
            x=-42.00681678019464,
            y=11.944106604158879,
            z=7.663686599731445,
        ),
        px4_velocity_ned_mps=velocity_ned,
        minimum_alignment_speed_mps=0.05,
    )
    limit_m = dynamic_identity_disagreement_limit_m(
        base_limit_m=0.25,
        px4_velocity_ned_mps=velocity_ned,
        alignment_seconds=result.alignment_seconds,
    )

    assert result.time_aligned_disagreement_m == pytest.approx(0.29392, abs=0.0001)
    assert limit_m == pytest.approx(0.35)
    assert result.time_aligned_disagreement_m < limit_m


def test_dynamic_identity_limit_does_not_relax_a_stationary_mismatch() -> None:
    assert dynamic_identity_disagreement_limit_m(
        base_limit_m=0.25,
        px4_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
        alignment_seconds=0.5,
    ) == pytest.approx(0.25)


def test_identity_offset_innovation_distinguishes_drift_from_entity_jump() -> None:
    slowly_drifted = Vector3(x=0.241, y=0.061, z=-0.029)

    assert identity_offset_innovation_m(
        estimator_offset_m=slowly_drifted,
        reference_offset_m=Vector3(x=0.202, y=0.084, z=-0.025),
    ) == pytest.approx(0.04545, abs=0.0001)
    assert identity_offset_innovation_m(
        estimator_offset_m=slowly_drifted,
        reference_offset_m=None,
    ) == pytest.approx(0.25028, abs=0.0001)


def test_identity_evidence_accepts_bounded_signed_alignment() -> None:
    evidence = ControlledVehicleIdentityEvidence.model_validate(
        {
            "schema_version": "dronedream.controlled-vehicle-identity.v1",
            "selected_entity_name": "base_link",
            "pose_reference": "canonical-link-relative-to-model",
            "gazebo_collision_center_world_enu_m": {"x": 0.0, "y": 0.0, "z": 1.0},
            "px4_collision_center_world_enu_m": {"x": 0.0, "y": 0.0, "z": 1.0},
            "position_disagreement_m": 0.0,
            "identity_alignment_seconds": -0.02,
            "tracking_age_seconds": 0.01,
            "consecutive_mismatch_samples": 0,
            "accepted": True,
            "updated_at_unix_ms": 1_000,
        }
    )

    assert evidence.identity_alignment_seconds == -0.02

    with pytest.raises(ValueError):
        ControlledVehicleIdentityEvidence.model_validate(
            {
                **evidence.model_dump(mode="json"),
                "identity_alignment_seconds": -0.501,
            }
        )


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_identity_alignment_rejects_invalid_time_window(value: float) -> None:
    with pytest.raises(ValueError, match="maximum alignment seconds"):
        align_gazebo_px4_identity(
            gazebo_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
            px4_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.0),
            px4_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
            maximum_alignment_seconds=value,
        )


def test_zero_speed_threshold_does_not_divide_by_stationary_speed():
    result = align_gazebo_px4_identity(
        gazebo_position_world_enu_m=Vector3(x=0., y=0., z=1.),
        px4_position_world_enu_m=Vector3(x=.3, y=0., z=1.),
        px4_velocity_ned_mps=Vector3(x=0., y=0., z=0.),
        minimum_alignment_speed_mps=0.,
    )
    assert result.alignment_seconds == 0.
    assert result.time_aligned_disagreement_m == pytest.approx(.3)
