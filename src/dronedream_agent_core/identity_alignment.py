"""Time-align PX4 and Gazebo positions before enforcing identity residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import Vector3


@dataclass(frozen=True)
class TimeAlignedIdentity:
    """A transparent raw and time-aligned comparison of one entity sample."""

    raw_disagreement_m: float
    time_aligned_disagreement_m: float
    alignment_seconds: float
    estimator_offset_m: Vector3
    aligned_gazebo_position_m: Vector3


def identity_offset_innovation_m(
    *,
    estimator_offset_m: Vector3,
    reference_offset_m: Vector3 | None,
) -> float:
    """Measure a frame-transform change without conflating it with drift.

    The first observation is still compared with the shared zero-origin
    assumption.  Once an entity/frame binding has been established, identity
    is guarded by the innovation from the last independently validated
    transform.  A slowly evolving PX4 estimator offset is therefore tracked,
    while an entity switch or coordinate-frame discontinuity remains a large
    residual and fails closed.
    """

    reference = reference_offset_m or Vector3(x=0.0, y=0.0, z=0.0)
    values = (
        estimator_offset_m.x,
        estimator_offset_m.y,
        estimator_offset_m.z,
        reference.x,
        reference.y,
        reference.z,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("identity offset innovation inputs must be finite")
    return math.dist(
        (estimator_offset_m.x, estimator_offset_m.y, estimator_offset_m.z),
        (reference.x, reference.y, reference.z),
    )


def dynamic_identity_disagreement_limit_m(
    *,
    base_limit_m: float,
    px4_velocity_ned_mps: Vector3,
    alignment_seconds: float,
    maximum_dynamic_allowance_m: float = 0.1,
) -> float:
    """Add a tightly bounded maneuver allowance to an identity residual limit.

    The linear time alignment above removes transport skew along the current
    velocity vector.  During a turn or a rapid braking transition, however,
    the older Gazebo pose lies on a curved trajectory and cannot be aligned by
    a single straight-line projection.  Reserve half of the projected travel
    distance for that unmodelled curvature, capped at ten centimetres.  A
    stationary sample receives no allowance, and the absolute identity gate
    therefore remains strict when motion cannot explain the residual.
    """

    values = (
        base_limit_m,
        px4_velocity_ned_mps.x,
        px4_velocity_ned_mps.y,
        px4_velocity_ned_mps.z,
        alignment_seconds,
        maximum_dynamic_allowance_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("identity disagreement limit inputs must be finite")
    if base_limit_m <= 0.0 or maximum_dynamic_allowance_m < 0.0:
        raise ValueError("identity disagreement limits are outside the safe range")
    speed_mps = math.hypot(
        px4_velocity_ned_mps.x, px4_velocity_ned_mps.y, px4_velocity_ned_mps.z
    )
    dynamic_allowance_m = min(
        maximum_dynamic_allowance_m,
        0.5 * speed_mps * abs(alignment_seconds),
    )
    limit = base_limit_m + dynamic_allowance_m
    if not math.isfinite(limit):
        raise ValueError("identity disagreement limit overflowed")
    return limit


def align_gazebo_px4_identity(
    *,
    gazebo_position_world_enu_m: Vector3,
    px4_position_world_enu_m: Vector3,
    px4_velocity_ned_mps: Vector3,
    maximum_alignment_seconds: float = 0.5,
    minimum_alignment_speed_mps: float = 0.2,
) -> TimeAlignedIdentity:
    """Remove only the bounded along-track component caused by source latency.

    Gazebo ``Pose_V`` does not expose a usable source timestamp in this
    runtime.  Its transport callback can therefore be current while the pose
    inside the message trails PX4 estimator telemetry.  Project the Gazebo
    position along the independently observed PX4 velocity by the bounded
    signed time that minimizes the three-dimensional residual.  Perpendicular
    disagreement remains untouched and is what the strict identity gate sees.
    """

    if not math.isfinite(maximum_alignment_seconds) or maximum_alignment_seconds < 0.0:
        raise ValueError("maximum alignment seconds must be finite and non-negative")
    if not math.isfinite(minimum_alignment_speed_mps) or minimum_alignment_speed_mps < 0.0:
        raise ValueError("minimum alignment speed must be finite and non-negative")
    gazebo = (
        gazebo_position_world_enu_m.x,
        gazebo_position_world_enu_m.y,
        gazebo_position_world_enu_m.z,
    )
    px4 = (
        px4_position_world_enu_m.x,
        px4_position_world_enu_m.y,
        px4_position_world_enu_m.z,
    )
    # NED north/east/down -> world ENU east/north/up.
    velocity = (
        px4_velocity_ned_mps.y,
        px4_velocity_ned_mps.x,
        -px4_velocity_ned_mps.z,
    )
    if not all(math.isfinite(value) for value in (*gazebo, *px4, *velocity)):
        raise ValueError("identity positions and velocity must be finite")
    delta = tuple(px4[index] - gazebo[index] for index in range(3))
    raw_disagreement = math.hypot(*delta)
    speed = math.hypot(*velocity)
    if not math.isfinite(raw_disagreement) or not math.isfinite(speed):
        raise ValueError("identity alignment distance or speed overflowed")
    alignment_seconds = 0.0
    if speed > 0 and speed >= minimum_alignment_speed_mps:
        # Normalize first to avoid overflowing v*v. A zero threshold still does
        # not authorize dividing by stationary velocity or creating fake motion.
        unconstrained = sum((delta[index] / speed) * (velocity[index] / speed)
                            for index in range(3))
        if not math.isfinite(unconstrained):
            raise ValueError("identity alignment projection overflowed")
        alignment_seconds = max(
            -maximum_alignment_seconds,
            min(maximum_alignment_seconds, unconstrained),
        )
    aligned = tuple(
        gazebo[index] + velocity[index] * alignment_seconds for index in range(3)
    )
    offset = tuple(aligned[index] - px4[index] for index in range(3))
    aligned_disagreement = math.hypot(*offset)
    if not math.isfinite(aligned_disagreement):
        raise ValueError("identity aligned position overflowed")
    return TimeAlignedIdentity(
        raw_disagreement_m=raw_disagreement,
        time_aligned_disagreement_m=aligned_disagreement,
        alignment_seconds=alignment_seconds,
        estimator_offset_m=Vector3(x=offset[0], y=offset[1], z=offset[2]),
        aligned_gazebo_position_m=Vector3(x=aligned[0], y=aligned[1], z=aligned[2]),
    )
