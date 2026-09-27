from __future__ import annotations

from typing import Any

import pytest

from dronedream_agent_core.contracts import (
    MissionContract,
    Px4GazeboArtifactBindings,
    Px4GazeboGates,
    Px4GazeboMeasurements,
    Px4GazeboRunEvidence,
    Px4UlogArtifact,
)
from dronedream_agent_core.execution import (
    COMPLETION_VERIFIER_INPUT_MAX_BYTES,
    _canonical_json_bytes,
    _completion_verifier_input,
)
from dronedream_agent_core.hashing import sha256_json


class _PreparedView:
    def __init__(self, contract: MissionContract) -> None:
        self.contract = contract
        self.verification_plan = None

    def model_dump(self, *, mode: str, exclude_unset: bool) -> dict[str, Any]:
        assert mode == "json"
        assert exclude_unset
        return {
            "schema_version": "dronedream.prepared-mission.v4",
            "contract": self.contract.model_dump(mode="json"),
        }


@pytest.mark.parametrize("nonvisual_unchanged", [True, False])
def test_completion_input_hash_summarizes_unbounded_visual_inventory(nonvisual_unchanged) -> None:
    contract = MissionContract(
        contract_id=f"mission-{'a' * 24}",
        conversation_id="completion-input-test",
        goal="Deliver the package and return",
        start_node="start",
        target_node="target",
        return_node="start",
        payload_action="pickup",
        map_asset_id="map.test",
        map_sha256="1" * 64,
        map_semantic_sha256="2" * 64,
        vehicle_asset_id="vehicle.test",
        vehicle_sha256="3" * 64,
        constraints=[],
        immutable_safety_rules=["remain collision free"],
    )
    huge_path_fragment = "visual-frame-path/" + ("x" * 4_000)
    frames = [
        Px4UlogArtifact(
            path=f"{huge_path_fragment}/{index}.png",
            size_bytes=100_000,
            sha256=f"{index:064x}",
        )
        for index in range(200)
    ]
    artifacts = Px4GazeboArtifactBindings(
        world_sha256="4" * 64,
        semantic_sha256="5" * 64,
        vehicle_sha256="3" * 64,
        route_sha256="6" * 64,
        track_sha256="7" * 64,
        clearance_sha256="8" * 64,
        controller_params_sha256="9" * 64,
        executor_sha256="a" * 64,
        px4_ulogs=[Px4UlogArtifact(path="flight.ulg", size_bytes=42, sha256="b" * 64)],
        ros_workspace="/opt/ros/jazzy",
        model_navigation_visual_frames=frames,
        static_render_batching={
            "nonvisual_world_unchanged": nonvisual_unchanged,
            "flight_qualification_granted": False,
            "failure": "render equivalence failed" if not nonvisual_unchanged else None,
            "batches": [
                {"mesh_sha256": f"{index:064x}", "source_visuals": ["z" * 8000]}
                for index in range(200)
            ],
        },
    )
    gates = Px4GazeboGates.model_construct(
        executor_completed=False,
        offboard_timing_complete=True,
        runtime_pose_samples_present=True,
        ros_observations_present=True,
        goal_observed=False,
        landing_confirmed=True,
        native_terminal_lifecycle_published=True,
        no_live_abort=True,
        px4_ulog_present=True,
        static_route_clearance_bound=True,
    )
    runtime = Px4GazeboRunEvidence(
        schema_version="dronedream.generic-px4-gazebo-run.v1",
        status="failed",
        world="world.test",
        vehicle="vehicle.test",
        gates=gates,
        measurements=Px4GazeboMeasurements(
            pose_sample_count=100,
            ros_observation_rows=100,
            abort_reason="waypoint stability timeout",
        ),
        artifacts=artifacts,
    )

    before = runtime.model_dump(mode="json")
    artifact = _completion_verifier_input(
        prepared=_PreparedView(contract),  # type: ignore[arg-type]
        route={"route": "bound"},  # type: ignore[arg-type]
        clearance={"clearance": "bound"},  # type: ignore[arg-type]
        track={"track": "bound"},  # type: ignore[arg-type]
        runtime=runtime,
        offboard_timing={"status": "failed", "failure": "settle timeout"},
        binding_gates={"runtime_track_file_hash_matches_prepared_file": True},
        normalized_runtime_evaluations=[
            {"plugin": index, "details": "y" * 1_000} for index in range(200)
        ],
        checkpoint_decisions=[],
        runtime_action_receipts=[],
        runtime_interruption_decisions=[],
        deterministic_success=False,
    )

    encoded = _canonical_json_bytes(artifact)
    visual_summary = artifact["runtime_evidence"]["artifact_collections"][
        "model_navigation_visual_frames"
    ]
    assert len(encoded) <= COMPLETION_VERIFIER_INPUT_MAX_BYTES
    assert artifact["input_contract"]["canonical_json_bytes"] == len(encoded)
    assert visual_summary["count"] == 200
    assert visual_summary["total_size_bytes"] == 20_000_000
    assert visual_summary["preview_truncated"] is True
    assert huge_path_fragment not in encoded.decode("utf-8")
    assert artifact["plugin_runtime_evaluations"]["items_omitted_for_input_bound"] is True
    assert runtime.model_dump(mode="json") == before
    rendered = artifact["runtime_evidence"]["artifact_bindings"]["static_render_batching"]
    original = before["artifacts"]["static_render_batching"]
    assert rendered["nonvisual_world_unchanged"] is nonvisual_unchanged
    assert rendered["flight_qualification_granted"] is False
    assert rendered["failure"] == original["failure"]
    assert rendered["batches"]["count"] == 200
    assert rendered["batches"]["collection_sha256"] == sha256_json(original["batches"])
    assert artifact["runtime_evidence"]["runtime_evidence_sha256"] == sha256_json(runtime)
    assert artifact["runtime_evidence"]["gates"]["executor_completed"] is False
    assert artifact["runtime_evidence"]["measurements"]["abort_reason"] == "waypoint stability timeout"
    assert artifact["deterministic_success"] is False
