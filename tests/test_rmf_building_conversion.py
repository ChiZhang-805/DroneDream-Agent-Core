from __future__ import annotations

import json
import zipfile
from xml.etree import ElementTree

import pytest

from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_source_adapters import detect_asset_source, normalize_asset_source
from dronedream_agent_core.collision import planning_collision_primitives, primitive_bounds
from dronedream_agent_core.contracts import MapAsset
from dronedream_agent_core.preferred_airspace import PreferredAirspace
from dronedream_agent_core.rmf_building_conversion import (
    RmfBuildingConversionError,
    convert_rmf_building_map,
)


def _source() -> bytes:
    return b"""name: Test Building
coordinate_system: cartesian_meters
levels:
  L1:
    elevation: 0
    vertices:
      - [0, 0, 0, office]
      - [8, 0, 0, door]
      - [8, 6, 0, pickup]
      - [0, 6, 0, return]
    walls:
      - [0, 1, {}]
      - [1, 2, {}]
      - [2, 3, {}]
      - [3, 0, {}]
    floors:
      - {vertices: [0, 1, 2, 3], parameters: {}}
    lanes:
      - [0, 1, {bidirectional: [4, true]}]
      - [1, 2, {bidirectional: [4, true]}]
"""


def test_conversion_outputs_valid_gazebo_semantics_and_airspace() -> None:
    converted = convert_rmf_building_map(_source(), source_name="test.building.yaml")
    world = ElementTree.fromstring(converted.sdf).find("world")
    assert world is not None
    plugin_names = {plugin.get("name") for plugin in world.findall("plugin")}
    assert {
        "gz::sim::systems::Physics",
        "gz::sim::systems::Contact",
        "gz::sim::systems::Imu",
        "gz::sim::systems::AirPressure",
        "gz::sim::systems::AirSpeed",
        "gz::sim::systems::ApplyLinkWrench",
        "gz::sim::systems::NavSat",
        "gz::sim::systems::Magnetometer",
        "gz::sim::systems::Sensors",
    } <= plugin_names
    assert world.findtext("spherical_coordinates/world_frame_orientation") == "ENU"
    semantic = json.loads(converted.semantic)
    primitives = planning_collision_primitives(semantic)
    # Static planning geometry and the exact runtime collision readback are both
    # present, matching the existing product semantic contract.
    assert len(primitives) == 10
    assert all(item.get("name") for item in primitives)
    assert all(primitive_bounds(item) for item in primitives)
    airspace = PreferredAirspace(semantic, {"map": "test"}, radius_m=0.2, height_m=0.2)
    assert airspace.snapshot()["volumes"]
    graph = MapAsset.model_validate(json.loads(converted.topology))
    assert graph.edges[0].qualification == "geometry-derived"
    assert graph.edges[0].bidirectional is True
    assert graph.named_entities["launch"] == graph.nodes[0].node_id
    assert json.loads(converted.report)["flight_qualified"] is False


def test_launch_is_selected_from_connected_lane_not_first_decorative_vertex() -> None:
    source = (
        _source()
        .replace(
            b"      - [0, 0, 0, office]\n",
            b"      - [-2, -2, 0, decoration]\n      - [0, 0, 0, office]\n",
        )
        .replace(
            b"      - [0, 1, {bidirectional: [4, true]}]\n"
            b"      - [1, 2, {bidirectional: [4, true]}]\n",
            b"      - [1, 2, {bidirectional: [4, true]}]\n"
            b"      - [2, 3, {bidirectional: [4, true]}]\n",
        )
    )
    converted = convert_rmf_building_map(source, source_name="connected.building.yaml")
    graph = MapAsset.model_validate(json.loads(converted.topology))
    launch_id = graph.named_entities["launch"]
    assert launch_id == "l1-v1"
    assert any(edge.from_node == launch_id or edge.to_node == launch_id for edge in graph.edges)


def test_zip_normalization_produces_unqualified_gazebo_package(tmp_path) -> None:
    source = tmp_path / "test.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("map/test.building.yaml", _source())
        bundle.writestr("map/test.png", b"not interpreted by converter")
    detection = detect_asset_source(source)
    assert detection.can_normalize_locally is True
    destination = tmp_path / "test.ddpkg"
    normalize_asset_source(source, detection, destination, expected_kind="map")
    inspected = inspect_ddpkg(destination)
    assert inspected.asset_ir.simulation_targets[0].simulator == "gazebo-harmonic"
    assert inspected.asset_ir.readiness.missing_fields == []
    assert inspected.asset_ir.readiness.qualification_required is True
    assert inspected.manifest.qualification is None
    assert [item.path for item in inspected.asset_ir.files if item.role == "semantic"] == [
        "normalized/rmf/generated/map-semantic.json"
    ]
    assert (
        inspected.asset_ir.semantics.navigation_graph_path
        == "normalized/rmf/generated/navigation-semantic-graph.json"
    )


def test_unsafe_or_ambiguous_source_fails_closed(tmp_path) -> None:
    with pytest.raises(RmfBuildingConversionError, match="DUPLICATE_KEY"):
        convert_rmf_building_map(
            b"name: one\nname: two\nlevels: {L1: {vertices: [[0, 0]]}}\n",
            source_name="bad.building.yaml",
        )
    source = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("a.building.yaml", _source())
        bundle.writestr("b.building.yaml", _source())
    detection = detect_asset_source(source)
    with pytest.raises(ValueError, match="ENTRYPOINT_AMBIGUOUS"):
        normalize_asset_source(source, detection, tmp_path / "bad.ddpkg", expected_kind="map")
