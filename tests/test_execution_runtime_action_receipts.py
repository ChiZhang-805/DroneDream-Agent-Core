from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from dronedream_agent_core.contracts import (
    Px4CoordinateContract,
    Px4Track,
    Px4TrackPoint,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionReceipt,
    RuntimeActionExecutionStep,
    WorldTrackPoint,
)
from dronedream_agent_core.execution import _load_runtime_action_receipts
from dronedream_agent_core.hashing import sha256_json


def _prepared_with_one_runtime_action():
    step = RuntimeActionExecutionStep(
        step_id="action-001",
        task_id="scan-target",
        action="delivery.scan-code",
        target_node="target",
        trigger="checkpoint",
        checkpoint_id="checkpoint-001",
        runtime_executor="native.payload.scan-code",
        adapter_id="runtime.domain.ros2-service",
        driver="ros2-service",
        required_success_evidence=["code captured"],
        max_attempts=1,
        fallback="hold",
        timeout_seconds=15.0,
        authority="actuate",
    )
    runtime_actions = RuntimeActionExecutionContract(
        contract_id="mission-test",
        task_graph_sha256="a" * 64,
        domain_action_catalog_sha256="b" * 64,
        adapter_catalog_sha256="c" * 64,
        steps=[step],
    )
    track = Px4Track(
        coordinate_contract=Px4CoordinateContract(
            model_root_world_enu_m=[0.0, 0.0, 0.0],
            collision_center_offset_model_m=[0.0, 0.0, 0.3],
        ),
        points=[
            Px4TrackPoint(x=0, y=0, z=1, phase="launch", speed_limit_mps=1),
            Px4TrackPoint(x=1, y=0, z=1, phase="transit", speed_limit_mps=1),
        ],
        source_world_points=[
            WorldTrackPoint(east_m=0, north_m=0, up_m=1),
            WorldTrackPoint(east_m=0, north_m=1, up_m=1),
        ],
        waypoint_hold_seconds=0.4,
    )
    return SimpleNamespace(px4_track=track, runtime_actions=runtime_actions)


def test_missing_runtime_action_receipt_remains_a_completion_gate_not_a_parse_error(
    tmp_path,
) -> None:
    receipts, required_step_ids = _load_runtime_action_receipts(
        _prepared_with_one_runtime_action(), tmp_path
    )

    assert receipts == []
    assert required_step_ids == {"action-001"}


def test_rejected_bound_receipt_is_preserved_for_failed_workflow_evidence(tmp_path) -> None:
    prepared = _prepared_with_one_runtime_action()
    step = prepared.runtime_actions.steps[0]
    receipt = RuntimeActionExecutionReceipt(
        execution_contract_sha256=sha256_json(prepared.runtime_actions),
        step_sha256=sha256_json(step),
        step_id=step.step_id,
        task_id=step.task_id,
        action=step.action,
        adapter_id=step.adapter_id,
        runtime_executor=step.runtime_executor,
        status="rejected",
        attempts=1,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        output={},
        observed_success_evidence=[],
        deterministic_gates={
            "driver_confirmed": False,
            "required_evidence_observed": False,
        },
        issue_codes=["ROS_2_DOMAIN_ACTION_UNAVAILABLE"],
    )
    receipt_dir = tmp_path / "runtime-actions" / "receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "action-001.receipt.json").write_text(
        receipt.model_dump_json(indent=2), encoding="utf-8"
    )

    receipts, required_step_ids = _load_runtime_action_receipts(prepared, tmp_path)

    assert receipts == [receipt]
    assert required_step_ids == {"action-001"}
