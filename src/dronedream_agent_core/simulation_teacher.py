"""Explicit deterministic demonstration aid, never a learned flight policy."""

import math

from .contracts import QuaternionWxyz, Vector3
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .realtime_feature_encoders import RealtimeFeatureSnapshot, world_enu_to_body


def teacher_input_deadline(features: RealtimeFeatureSnapshot | None, *, now_ms: int) -> int:
    """An explicit teacher never gets looser source freshness than the student."""
    if (
        features is None
        or not features.fresh_at(now_ms)
        or features.policy_feature_contract_sha256() != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        return 0
    return features.control_deadline_unix_ms(now_unix_ms=now_ms)


def teacher_heading_rate(
    *, orientation: QuaternionWxyz, position: Vector3, goal: Vector3, maximum_rate_dps: float
) -> float:
    """Return bounded clockwise body yaw in degrees/second for a demo target.

    Positions are world ENU metres; this proportional reference neither moves
    the aircraft nor proves that a trained policy can perform the same turn.
    """
    if not math.isfinite(maximum_rate_dps) or not 0 < maximum_rate_dps <= 45:
        raise ValueError("SIMULATION_TEACHER_YAW_LIMIT_INVALID")
    direction = world_enu_to_body(
        orientation,
        Vector3(
            x=goal.x - position.x,
            y=goal.y - position.y,
            z=0.0,
        ),
    )
    if math.hypot(direction.x, direction.y) < 0.1:
        # Near-coincident horizontal positions have no useful bearing.
        return 0.0
    # Body right is positive clockwise. A one-second proportional yaw
    # reference is only a deterministic teacher, not a fabricated neural output.
    error_deg = math.degrees(math.atan2(direction.y, direction.x))
    return max(-maximum_rate_dps, min(maximum_rate_dps, error_deg))
