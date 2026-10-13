import importlib.util
from pathlib import Path

import pytest
from shapely.geometry import Point, box

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "handoff_builder", ROOT / "scripts/build_map_handoffs.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_neighbour_between_building_and_handoff_is_rejected():
    owner = box(0, 0, 10, 10)
    assert builder.ground_relation(Point(20, 5), owner, owner, box(12, 0, 16, 10), []) is None


def test_door_connection_cannot_cut_through_its_own_building():
    owner = box(0, 0, 10, 10)
    door = {"entrance_id": "north", "position_xy_m": [5, 10]}
    opposite = builder.ground_relation(Point(5, -3), owner, owner, box(30, 30, 40, 40), [door])
    correct = builder.ground_relation(Point(5, 13), owner, owner, box(30, 30, 40, 40), [door])
    assert not opposite["entrance_connection_verified"]
    assert correct["entrance_connection_verified"]
    assert correct["entrance_distance_m"] == 3


def test_enclosed_annex_does_not_get_a_fake_exterior_target():
    annex = box(0, 0, 2, 2)
    collision = box(-10, -10, 12, 12)
    assert (
        builder.choose_location(
            annex, annex, box(30, 30, 40, 40), annex, collision, box(-50, -50, 50, 50), []
        )
        is None
    )


def test_every_building_has_an_audited_outcome_and_available_targets_are_bound(tmp_path):
    import hashlib
    import zipfile

    from dronedream_agent_core.assets import load_map_catalog, resolve_map_entity
    from dronedream_agent_core.contracts import MapAsset
    from dronedream_agent_core.place_index import _bundled_index

    with zipfile.ZipFile(
        ROOT / "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg"
    ) as z:
        raw = z.read("normalized/map/gazebo/semantic.json")
        graph = MapAsset.model_validate_json(z.read("normalized/map/navigation-graph.json"))
    semantic = tmp_path / "semantic.json"
    semantic.write_bytes(raw)
    catalog = load_map_catalog(semantic, qualified_graph=graph)
    index = _bundled_index(hashlib.sha256(raw).hexdigest())
    buildings = {b["building_id"] for b in index["buildings"]}
    outcomes = {row["place_id"]: row for row in index["handoff_locations"]}
    assert buildings <= outcomes.keys()
    assert sum(outcomes[b]["flight_target_available"] for b in buildings) == 41
    for bid in buildings:
        row = outcomes[bid]
        if row["flight_target_available"]:
            assert resolve_map_entity(bid, catalog, graph) == row["node_id"]
            assert row["route_check"]["clearance"]["accepted"]
            assert row["route_check"]["clearance"]["collision_count"] == 0
        else:
            assert row["reason"] == "no-exterior-handoff-within-current-collision-geometry"
            assert bid not in graph.named_entities
    assert "取件点" not in graph.named_entities
    assert "Väinö Auerin katu 7" not in graph.named_entities
    assert (
        outcomes["osm-facility-16790293"]["entrance_id"]
        != outcomes["osm-facility-16790295"]["entrance_id"]
    )


def test_missing_target_asks_for_another_place_instead_of_internal_failure(monkeypatch):
    from dronedream_agent_core import orchestrator
    from dronedream_agent_core.assets import AssetQualificationError

    def unresolved(*args):
        raise AssetQualificationError("UNRESOLVED_MAP_ENTITY:unknown-building")

    monkeypatch.setattr(orchestrator, "resolve_map_entity", unresolved)
    with pytest.raises(orchestrator.MissionClarificationRequired) as caught:
        orchestrator._resolve_entity("附属建筑 29466", None, None)
    assert "没有可到达的交接点" in caught.value.questions[0]["prompt"]
