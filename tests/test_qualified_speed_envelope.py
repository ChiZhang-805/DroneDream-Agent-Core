"""One completed-flight speed ceiling reaches planning and the actual transport boundary."""

import asyncio
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from test_default_asset_builder import _load_builder
from test_orchestrator_validation import _contract, _map
from test_px4_track import _graph, _semantic, _vehicle
from test_runtime_commands import _load_executor
from test_runtime_replan import _replan_inputs

from dronedream_agent_core.contracts import (
    GraphRoute,
    RuntimeLocalSafetyObservation,
    SemanticPlan,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.gazebo_adapter import _effective_local_speed_limit_mps
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.orchestrator import MissionPreparationBlocked, _validate_semantic_plan
from dronedream_agent_core.px4_track import route_to_px4_track
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety
from dronedream_agent_core.runtime_replan import RuntimeReplanError, build_runtime_speed_replacement


def _qualified_vehicle(limit=1.2):
    return VehicleAsset.model_validate({
        **_vehicle().model_dump(), "max_speed_mps": 6.0, "qualified_cruise_max_mps": limit,
    })


def test_design_and_qualified_speed_are_distinct_and_legacy_assets_keep_their_limit():
    vehicle = _qualified_vehicle()
    assert vehicle.max_speed_mps == 6.0
    assert vehicle.operational_max_speed_mps == 1.2
    assert _vehicle().operational_max_speed_mps == _vehicle().max_speed_mps
    with pytest.raises(ValueError, match="exceeds the design"):
        _qualified_vehicle(6.1)


def test_absent_optional_qualification_preserves_legacy_canonical_asset_hash():
    legacy = _vehicle().model_dump(mode="json")
    assert "qualified_cruise_max_mps" not in legacy
    vehicle = VehicleAsset.model_validate(legacy)
    assert sha256_json(vehicle) == sha256_json(legacy)
    assert _qualified_vehicle().model_dump()["qualified_cruise_max_mps"] == 1.2


@pytest.mark.parametrize("invalid", [0, -1, True, float("nan"), float("inf")])
def test_invalid_qualified_speed_is_not_an_execution_limit(invalid):
    with pytest.raises(ValueError):
        _qualified_vehicle(invalid)


@pytest.mark.parametrize("update", [
    {"qualified_cruise_max_mps": 0.0}, {"qualified_cruise_max_mps": float("nan")},
    {"qualified_cruise_max_mps": float("inf")}, {"qualified_cruise_max_mps": True},
    {"max_speed_mps": 0.5},
])
def test_unvalidated_model_copy_cannot_bypass_qualification(update):
    copied = _qualified_vehicle().model_copy(update=update)
    with pytest.raises(ValueError):
        _ = copied.operational_max_speed_mps


def test_qualified_speed_uses_completed_motion_not_stationary_limits_or_design(tmp_path):
    builder = _load_builder()
    track = tmp_path / "reference_track.json"
    controller = tmp_path / "controller_params.json"
    track.write_text(json.dumps({"points": [
        {"x": 0, "y": 0, "z": 1, "speed_limit_mps": 6},
        {"x": 0, "y": 0, "z": 1, "speed_limit_mps": 6},
        {"x": 1, "y": 0, "z": 1, "speed_limit_mps": 1.2},
        {"x": 2, "y": 0, "z": 1, "speed_limit_mps": 0.4},
    ]}), encoding="utf-8")
    controller.write_bytes(b'{"vel_limit": 6.0, "accel_limit": 1.5}\n')
    original = (track.read_bytes(), controller.read_bytes())
    assert builder._verified_track_speed_limit_mps(tmp_path, controller) == 1.2
    assert (track.read_bytes(), controller.read_bytes()) == original
    controller.write_text('{"vel_limit":0.8}', encoding="utf-8")
    assert builder._verified_track_speed_limit_mps(tmp_path, controller) == 0.8


@pytest.mark.parametrize("invalid", [0, -1, True, None, float("nan"), float("inf")])
def test_qualification_rejects_invalid_reference_speed(tmp_path, invalid):
    (tmp_path / "reference_track.json").write_text(json.dumps({"points": [
        {"x": 0, "y": 0, "z": 1, "speed_limit_mps": 1.2},
        {"x": 1, "y": 0, "z": 1, "speed_limit_mps": invalid},
    ]}), encoding="utf-8")
    controller = tmp_path / "params.json"
    controller.write_text('{"vel_limit":6.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="REFERENCE_SPEED_INVALID"):
        _load_builder()._verified_track_speed_limit_mps(tmp_path, controller)


def test_stationary_only_track_does_not_qualify_cruise(tmp_path):
    (tmp_path / "reference_track.json").write_text(json.dumps({"points": [
        {"x": 0, "y": 0, "z": 1, "speed_limit_mps": 1.2},
        {"x": 0, "y": 0, "z": 1, "speed_limit_mps": 1.2},
    ]}), encoding="utf-8")
    controller = tmp_path / "params.json"
    controller.write_text('{"vel_limit":6.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="REFERENCE_TRACK_DISTANCE_INVALID"):
        _load_builder()._verified_track_speed_limit_mps(tmp_path, controller)


def test_model_can_choose_sub_three_speed_but_cannot_exceed_qualified_limit():
    def plan(speed):
        return SemanticPlan(
            ordered_targets=["pickup", "office"],
            speed_preferences=[dict(target_node=target, cruise_speed_mps=speed,
                approach_speed_mps=0.4, rationale="Qualified low campus speed.")
                for target in ("pickup", "office")],
            rationale_summary="Follow qualified roads and return.",
        )
    _validate_semantic_plan(plan(1.2), _contract(), _map(), _qualified_vehicle())
    with pytest.raises(MissionPreparationBlocked, match="SPEED_OUTSIDE_ENVELOPE"):
        _validate_semantic_plan(plan(1.21), _contract(), _map(), _qualified_vehicle())


@pytest.mark.parametrize("edge_id", ["launch-to-dock", "metric-edge-1"])
def test_graph_and_metric_tracks_and_depth_worker_share_qualified_cap(tmp_path, edge_id):
    graph = _graph()
    graph.edges[0].speed_limit_mps = 5.0
    route = GraphRoute(start_node="roof-launch", goal_node="dock-a",
        node_ids=["roof-launch", "dock-a"], edge_ids=[edge_id],
        positions_m=[node.position_m for node in graph.nodes],
        route_length_m=graph.edges[0].distance_m, all_edges_flight_verified=False)
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(_semantic()), encoding="utf-8")
    vehicle = _qualified_vehicle()
    track = route_to_px4_track(route, graph, path, vehicle=vehicle)
    assert {point.speed_limit_mps for point in track.points} == {1.2}
    assert _effective_local_speed_limit_mps(
        controller_speed_limit_mps=6.0, track=track, vehicle=vehicle) == 1.2


def test_local_avoidance_cannot_recover_design_speed_from_a_larger_requested_limit():
    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=10_000,
        source="onboard", stream_healthy=True, stream_age_seconds=0.01,
        localization_covariance_m2=0.001, current_position_m=Vector3(x=0, y=0, z=1),
        current_velocity_mps=Vector3(x=0, y=0, z=0), target_position_m=Vector3(x=5, y=0, z=1))
    command = evaluate_runtime_local_safety(observation=observation,
        vehicle=_qualified_vehicle(0.15), static_primitives=[], required_clearance_m=0.35,
        generated_at_unix_ms=10_010, maximum_speed_mps=6.0)
    velocity = command.decision.selected_velocity_mps
    assert math.hypot(velocity.x, velocity.y, velocity.z) <= 0.15 + 1e-9


def test_runtime_speed_amendment_cannot_raise_the_qualified_limit(tmp_path):
    inputs = _replan_inputs(tmp_path)
    inputs.pop("catalog")
    inputs.pop("return_node")
    inputs["vehicle"] = VehicleAsset.model_validate({
        **inputs["vehicle"].model_dump(), "max_speed_mps": 6.0,
        "qualified_cruise_max_mps": 1.2,
    })
    original = inputs["decision"]
    for speed in (1.3, 1.0):
        classification = original.classification.model_copy(update={
            "requested_action": "set_speed", "parameters": {"maximum_speed_mps": speed},
        })
        inputs["decision"] = original.model_copy(update={"classification": classification})
        if speed > 1.2:
            with pytest.raises(RuntimeReplanError, match="SPEED_OUTSIDE_VEHICLE_ENVELOPE"):
                build_runtime_speed_replacement(**inputs)
        else:
            replacement = build_runtime_speed_replacement(**inputs)
            assert max(point.speed_limit_mps for point in replacement.track.points) <= speed


@pytest.mark.parametrize("method", ["set_velocity_ned", "set_model_velocity_ned",
    "set_position_velocity_ned", "set_local_position_velocity_ned"])
def test_actual_px4_transport_rejects_overspeed_without_rewriting_authorized_velocity(method):
    executor = _load_executor()

    @dataclass(frozen=True)
    class Params:
        vel_limit: float
        accel_limit: float

    original = Params(6.0, 1.5)
    effective = executor._operational_controller_params(original, _qualified_vehicle())
    assert original.vel_limit == 6.0
    assert effective.vel_limit == 1.2

    class RawClient:
        def __init__(self):
            self.commands = []

        async def set_velocity_ned(self, velocity):
            self.commands.append(velocity)

        async def set_position_velocity_ned(self, position, velocity):
            self.commands.append(velocity)

    async def exercise():
        raw = RawClient()
        client = executor.SpawnRelativeOffboardClient(raw,
            SimpleNamespace(north_m=10, east_m=20, down_m=0),
            maximum_speed_mps=effective.vel_limit)
        position = SimpleNamespace(north_m=0, east_m=0, down_m=-1, yaw_deg=0)
        velocity = SimpleNamespace(north_m_s=1.0, east_m_s=1.0, down_m_s=0, yaw_deg=0)
        sender = getattr(client, method)
        with pytest.raises(executor.UserDirectedLanding, match="operational speed envelope"):
            await sender(position, velocity) if "position" in method else await sender(velocity)
        assert raw.commands == []
        assert velocity.north_m_s == velocity.east_m_s == 1.0
        velocity.north_m_s, velocity.east_m_s = 1.2, 0.0
        await sender(position, velocity) if "position" in method else await sender(velocity)
        assert raw.commands[0].north_m_s == 1.2

    asyncio.run(exercise())
