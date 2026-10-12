from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from dronedream_agent_core.assets import load_map_catalog
from dronedream_agent_core.orchestrator import _intent_map_catalog_view
from dronedream_agent_core.place_index import _bundled_index, map_place_context

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg"


def source_semantic():
    with zipfile.ZipFile(PACKAGE) as archive:
        raw = archive.read("normalized/map/gazebo/semantic.json")
    return raw, hashlib.sha256(raw).hexdigest()


def test_every_shipped_building_component_has_identity_and_provenance():
    raw, digest = source_semantic()
    index = _bundled_index(digest)
    assert index is not None
    semantic = json.loads(raw)
    expected = {p["name"] for p in semantic["collision_primitives"] if p["semantic"] == "building"}
    mapped = [name for b in index["buildings"] for name in b["geometry_component_ids"]]
    assert len(mapped) == len(set(mapped)) == 42
    assert set(mapped) == expected
    assert len(index["buildings"]) == 41
    for b in index["buildings"]:
        assert b["name"] and b["source_refs"]
        assert b["handoff_candidate_ids"]
        assert b["total_entrance_count"] is None  # Public data is not a complete door survey.


def test_multiple_entrances_and_service_locations_are_preserved():
    _, digest = source_semantic()
    index = _bundled_index(digest)
    chemicum = next(b for b in index["buildings"] if b["building_id"] == "city-33270")
    assert len(chemicum["entrance_ids"]) == 5
    assert len(index["entrances"]) == len({e["entrance_id"] for e in index["entrances"]})
    assert any(e["access"] == "employees" for e in index["entrances"])
    facilities = {f["facility_id"]: f for f in index["facilities"]}
    assert "图书馆" in facilities["osm-facility-16790294"]["aliases"]
    assert "计算机楼" in facilities["osm-facility-16790295"]["aliases"]
    assert any(s["name"] == "UniCafe Chemicum" for s in index["services"])
    assert any(s["name"] == "UniCafe Physicum" for s in index["services"])
    assert all(not h["flight_target_available"] for h in index["handoff_candidates"])
    assert all(h["horizontal_building_clearance_m"] >= 2.499 for h in index["handoff_candidates"])


def test_index_reaches_intent_model_without_inventing_executable_nodes(tmp_path):
    raw, digest = source_semantic()
    path = tmp_path / "semantic.json"
    path.write_bytes(raw)
    catalog = load_map_catalog(path)
    view = _intent_map_catalog_view(catalog)
    assert len(view["place_context"]["buildings"]) == 41
    assert len(catalog.entities) == 6  # Additional public facts are not admitted route nodes.
    assert "position_xy_m" not in json.dumps(view["place_context"])
    assert not map_place_context("0" * 64)
    # A caller cannot mutate the cached source shared by other accounts/missions.
    view["place_context"]["buildings"].clear()
    assert len(map_place_context(digest)["buildings"]) == 41
