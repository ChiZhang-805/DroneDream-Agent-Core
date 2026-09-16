from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from dronedream_agent_core.contracts import GraphRoute, Vector3
from scripts.run_school_map_depth_qualification import (
    _closed_qualification_route,
    _load_vehicle,
    _validate_training_capture,
)


def test_learner_receipts_do_not_require_a_duplicate_vision_training_corpus(tmp_path):
    _validate_training_capture(training=True, dataset=None, learner_channel=tmp_path / "port.json")
    _validate_training_capture(training=True, dataset=tmp_path / "media", learner_channel=None)
    _validate_training_capture(training=False, dataset=None, learner_channel=None)
    with pytest.raises(ValueError, match="explicit learner channel"):
        _validate_training_capture(training=True, dataset=None, learner_channel=None)


def test_repeated_closed_route_has_no_zero_length_join() -> None:
    outbound = GraphRoute.model_construct(
        start_node="launch",
        goal_node="turn",
        node_ids=["launch", "door", "turn"],
        edge_ids=["out-1", "out-2"],
        positions_m=[
            Vector3(x=0.0, y=0.0, z=1.0),
            Vector3(x=0.0, y=2.0, z=1.0),
            Vector3(x=1.0, y=3.0, z=1.0),
        ],
        route_length_m=2.0 + math.sqrt(2.0),
        all_edges_flight_verified=False,
    )

    route, graph = _closed_qualification_route(
        outbound,
        point_count=3,
        speed_limit_mps=0.3,
        round_trip_repetitions=3,
    )

    assert len(route.positions_m) == 13
    assert len(route.edge_ids) == 12
    assert len(graph.edges) == 12
    assert route.positions_m[0] == route.positions_m[4]
    assert route.positions_m[4] == route.positions_m[8]
    assert route.positions_m[8] == route.positions_m[12]
    assert all(edge.distance_m > 0.0 for edge in graph.edges)
    assert graph.nodes[0].label == "Qualification launch"
    assert "office-launch-pad" not in graph.named_entities
    assert graph.named_entities["qualification-launch"] == route.start_node
    assert route.route_length_m == pytest.approx(
        3.0 * 2.0 * (2.0 + math.sqrt(2.0))
    )


def test_depth_qualification_loads_the_flown_vehicle_contract(tmp_path: Path) -> None:
    metadata = tmp_path / "vehicle.json"
    sdf = tmp_path / "model.sdf"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.vehicle.v1",
                "asset_id": "vehicle-current",
                "name": "Current vehicle",
                "coordinate_frame": "base_link_frd",
                "dry_mass_kg": 2.0,
                "max_takeoff_mass_kg": 2.1,
                "body_radius_m": 0.3,
                "body_height_m": 0.4,
                "max_speed_mps": 1.0,
                "max_acceleration_mps2": 0.8,
                "qualified_range_m": 100.0,
                "reserve_battery_percent": 30.0,
                "max_pickup_payload_kg": 0.05,
                "sensors": ["imu", "oakd-lite-depth"],
            }
        ),
        encoding="utf-8",
    )
    sdf.write_text("<sdf><include><uri>model://x500_depth</uri></include></sdf>", encoding="utf-8")

    assert _load_vehicle(metadata, vehicle_sdf=sdf).asset_id == "vehicle-current"

    sdf.write_text("<sdf><include><uri>model://x500</uri></include></sdf>", encoding="utf-8")
    with pytest.raises(ValueError, match="current x500 depth vehicle"):
        _load_vehicle(metadata, vehicle_sdf=sdf)
