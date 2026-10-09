from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from dronedream_agent_core.contracts import GraphRoute, Vector3
from scripts.run_school_map_depth_qualification import (
    _load_vehicle,
    _qualification_route,
    _validate_training_capture,
    main,
)


def test_learner_receipts_do_not_require_a_duplicate_vision_training_corpus(tmp_path):
    _validate_training_capture(training=True, dataset=None, learner_channel=tmp_path / "port.json")
    _validate_training_capture(training=True, dataset=tmp_path / "media", learner_channel=None)
    _validate_training_capture(training=False, dataset=None, learner_channel=None)
    _validate_training_capture(training=True, dataset=None, learner_channel=None,
                               teacher_observations=True)
    with pytest.raises(ValueError, match='explicit boolean'):
        _validate_training_capture(training=True, dataset=None, learner_channel=None,
                                   teacher_observations='yes')
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

    route, graph = _qualification_route(
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


def test_preclosed_pair_route_is_not_mirrored_twice() -> None:
    source = GraphRoute.model_construct(
        start_node="launch",
        goal_node="launch",
        node_ids=["launch", "turn", "launch"],
        edge_ids=["out", "back"],
        positions_m=[
            Vector3(x=0.0, y=0.0, z=1.4),
            Vector3(x=2.0, y=0.0, z=1.4),
            Vector3(x=0.0, y=0.0, z=1.4),
        ],
        route_length_m=4.0,
        all_edges_flight_verified=False,
    )

    route, graph = _qualification_route(
        source,
        point_count=None,
        speed_limit_mps=0.3,
    )

    assert len(route.positions_m) == 3
    assert route.positions_m[0] == route.positions_m[-1]
    assert route.route_length_m == pytest.approx(4.0)
    assert len(graph.edges) == 2


def test_preclosed_pair_route_with_existing_return_leg_is_preserved() -> None:
    points = [
        Vector3(x=0.0, y=0.0, z=1.4),
        Vector3(x=1.0, y=0.0, z=2.0),
        Vector3(x=2.0, y=1.0, z=2.0),
        Vector3(x=3.0, y=1.0, z=1.4),
        Vector3(x=2.0, y=1.0, z=2.0),
        Vector3(x=1.0, y=0.0, z=2.0),
        Vector3(x=0.0, y=0.0, z=1.4),
    ]
    source = GraphRoute.model_construct(
        start_node="launch",
        goal_node="launch",
        node_ids=["launch", "a", "b", "pickup", "b", "a", "launch"],
        edge_ids=[f"edge-{index}" for index in range(6)],
        positions_m=points,
        route_length_m=sum(
            math.dist(
                (first.x, first.y, first.z),
                (second.x, second.y, second.z),
            )
            for first, second in zip(points, points[1:], strict=False)
        ),
        all_edges_flight_verified=False,
    )

    route, graph = _qualification_route(
        source,
        point_count=None,
        speed_limit_mps=0.3,
    )

    assert route.positions_m == points
    assert len(route.positions_m) == 7
    assert len(route.edge_ids) == 6
    assert route.node_ids[0] == route.node_ids[-1] == "qualification-launch"
    assert route.route_length_m == pytest.approx(source.route_length_m)
    assert len(graph.edges) == 6
    assert graph.named_entities["qualification-turn"] == route.node_ids[3]


# 功能：
#   单程采集课程保留真实终点，不暗中拼接返程或把终点声明为起点。
# 输入：
#   无。
# 输出：
#   None：核对路线、图边和长度的一致性。
def test_one_way_training_route_keeps_explicit_outbound_goal():
    source = GraphRoute(start_node='a', goal_node='b', node_ids=['a', 'b'], edge_ids=['ab'],
        positions_m=[Vector3(x=0., y=0., z=2.), Vector3(x=2., y=0., z=2.)],
        route_length_m=2., all_edges_flight_verified=False)
    route, graph = _qualification_route(source, point_count=None, speed_limit_mps=.4, return_to_launch=False)
    assert route.positions_m == source.positions_m
    assert route.start_node != route.goal_node
    assert route.goal_node == route.node_ids[-1] == graph.edges[-1].to_node
    assert route.route_length_m == 2.
    with pytest.raises(ValueError, match='cannot repeat'):
        _qualification_route(source, point_count=None, speed_limit_mps=.4,
                             return_to_launch=False, round_trip_repetitions=2)


# 功能：
#   在打开资产或启动仿真之前拒绝把单程训练开关用于产品资格或重复往返。
# 输入：
#   monkeypatch：命令行隔离器；flags：未授权的开关组合。
# 输出：
#   None：非法课程不启动任何物理运行。
@pytest.mark.parametrize('flags', [[], ['--training-data-collection', '--multimodal-dataset-root', 'unused'],
    ['--training-data-collection', '--teacher-observation-collection', '--round-trip-repetitions', '2']])
def test_one_way_training_requires_explicit_single_teacher_run(monkeypatch, flags):
    monkeypatch.setattr(sys, 'argv', ['qualification', *['unused'] * 7, '--one-way-training', *flags])
    with pytest.raises(ValueError, match='single explicit teacher'):
        main()


# 功能：产品独立路线控制不允许与训练模型或混合控制并列，避免验收时漏掉正式地图融合。
# 输入：命令行隔离器和每种冲突开关。
# 输出：在读取任何地图文件之前拒绝歧义控制权配置。
@pytest.mark.parametrize(
    "conflicting_flags",
    [
        ["--training-data-collection", "--multimodal-dataset-root", "unused",
         "--multimodal-flight-id", "flight"],
        ["--local-navigation-provider", "kimi"],
        ["--experimental-map-fusion"],
        ["--heading-policy", "measured-hold"],
    ],
)
def test_independent_route_qualification_rejects_conflicting_control_modes(
    monkeypatch, conflicting_flags
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualification",
            *["unused"] * 7,
            "--independent-route-control",
            *conflicting_flags,
        ],
    )
    with pytest.raises(
        ValueError,
        match="INDEPENDENT_ROUTE_CONTROL_REQUIRES_MODEL_FREE_PRODUCT_MODE",
    ):
        main()


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
