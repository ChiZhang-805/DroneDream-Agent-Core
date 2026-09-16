from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_depth_fault_recovery import verify_depth_fault_recovery


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _append(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _run(tmp_path: Path, *, unsafe_continue: bool = False) -> Path:
    gates = {
        "development_fault_injection_absent": False,
        "executor_completed": True,
        "goal_observed": True,
        "landing_confirmed": True,
    }
    _write(
        tmp_path / "qualification-summary.json",
        {
            "status": "failed",
            "gates": gates,
            "measurements": {
                "abort_reason": None,
                "landing_state": "ON_GROUND",
                "model_navigation": {
                    "provider_call_count": 12,
                    "authorized_control_applied_count": 8,
                },
            },
        },
    )
    _write(
        tmp_path / "simulation" / "development-depth-fault.json",
        {
            "development_only": True,
            "activated": True,
            "activated_at_unix_ms": 2_000,
            "drop_duration_seconds": 2.5,
            "recovered": True,
            "recovered_at_unix_ms": 4_450,
        },
    )
    _append(
        tmp_path
        / "simulation"
        / "runtime-state"
        / "local-safety-executor-history.jsonl",
        [
            {
                "recorded_at_unix_ms": 1_900,
                "status": "accepted",
                "action": "continue",
                "model_navigation_authorized": True,
            },
            {
                "recorded_at_unix_ms": 2_500,
                "status": "accepted",
                "action": "continue" if unsafe_continue else "hold",
            },
            {
                "recorded_at_unix_ms": 4_600,
                "status": "accepted",
                "action": "continue",
                "model_navigation_authorized": True,
            },
        ],
    )
    _append(
        tmp_path / "simulation" / "depth-local-safety-history.jsonl",
        [
            {"recorded_at_unix_ms": 1_950},
            {"recorded_at_unix_ms": 4_500},
        ],
    )
    return tmp_path


def test_verifies_fail_closed_depth_outage_and_resume(tmp_path: Path) -> None:
    receipt = verify_depth_fault_recovery(_run(tmp_path))

    assert receipt["status"] == "verified"
    assert receipt["flight_qualification_granted"] is False
    assert receipt["measurements"]["observed_fresh_depth_gap_ms"] == 2_550


def test_rejects_motion_authorized_during_depth_outage(tmp_path: Path) -> None:
    receipt = verify_depth_fault_recovery(
        _run(tmp_path, unsafe_continue=True)
    )

    assert receipt["status"] == "failed"
    assert receipt["gates"]["motion_not_authorized_during_depth_outage"] is False
