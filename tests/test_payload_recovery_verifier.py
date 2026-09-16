from __future__ import annotations

from scripts.verify_payload_recovery import _safe_terminal_outcome


def test_payload_recovery_accepts_only_safe_terminal_landing_contact() -> None:
    assert _safe_terminal_outcome(
        {"landing_state": "ON_GROUND", "abort_reason": None}
    )
    assert _safe_terminal_outcome(
        {
            "landing_state": "ON_GROUND",
            "abort_reason": "LIVE_STATIC_COLLISION",
            "live_safety_event": {
                "reason": "LIVE_STATIC_COLLISION",
                "runtime_phase": "LANDING",
                "collision_primitive": "office-drone-launch-pad",
                "clearance_m": -0.021,
            },
        }
    )
    assert not _safe_terminal_outcome(
        {
            "landing_state": "ON_GROUND",
            "abort_reason": "LIVE_STATIC_COLLISION",
            "live_safety_event": {
                "reason": "LIVE_STATIC_COLLISION",
                "runtime_phase": "TRACK",
                "collision_primitive": "office-right-wall",
                "clearance_m": -0.004,
            },
        }
    )
    assert not _safe_terminal_outcome(
        {
            "landing_state": "ON_GROUND",
            "abort_reason": "LIVE_STATIC_COLLISION",
            "live_safety_event": {
                "reason": "LIVE_STATIC_COLLISION",
                "runtime_phase": "LANDING",
                "collision_primitive": "office-drone-launch-pad",
                "clearance_m": -0.031,
            },
        }
    )
