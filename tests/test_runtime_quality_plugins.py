from types import SimpleNamespace

from dronedream_agent_core.contracts import Px4GazeboGates
from dronedream_agent_core.execution import _active_runtime_gates_passed
from dronedream_agent_plugins.runtime_quality_plugins import _runtime_gate_integrity


def _required_gates(**optional: bool) -> Px4GazeboGates:
    return Px4GazeboGates(
        executor_completed=True,
        offboard_timing_complete=True,
        runtime_pose_samples_present=True,
        ros_observations_present=True,
        goal_observed=True,
        landing_confirmed=True,
        native_terminal_lifecycle_published=True,
        no_live_abort=True,
        px4_ulog_present=True,
        static_route_clearance_bound=True,
        **optional,
    )


def test_runtime_gate_integrity_ignores_capabilities_not_enabled_for_run() -> None:
    result = _runtime_gate_integrity(
        runtime=SimpleNamespace(status="verified", gates=_required_gates())
    )

    assert result["accepted"] is True
    assert "semantic_supervision_recorded_for_every_sample" not in result["gates"]


def test_runtime_gate_integrity_rejects_explicitly_enabled_false_gate() -> None:
    result = _runtime_gate_integrity(
        runtime=SimpleNamespace(
            status="verified",
            gates=_required_gates(
                semantic_supervision_recorded_for_every_sample=False
            ),
        )
    )

    assert result["accepted"] is False
    assert result["issue_codes"] == ["RUNTIME_GATE_INTEGRITY_REJECTED"]


def test_completion_determinism_uses_same_active_gate_envelope() -> None:
    runtime = SimpleNamespace(status="verified", gates=_required_gates())
    explicit_failure = SimpleNamespace(
        status="verified",
        gates=_required_gates(model_navigation_visual_frame_recorded=False),
    )

    assert _active_runtime_gates_passed(runtime) is True
    assert _active_runtime_gates_passed(explicit_failure) is False
