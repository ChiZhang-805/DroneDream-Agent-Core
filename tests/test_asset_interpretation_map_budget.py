"""Dense road sampling must not become unbounded model context."""
from __future__ import annotations

import json
from types import SimpleNamespace

from dronedream_agent_app import asset_interpretation
from dronedream_agent_app.interpretation_reuse import interpretation_lineage
from dronedream_agent_core.contracts import MapAsset


def dense_graph() -> MapAsset:
    return MapAsset.model_validate({
        "asset_id": "dense-road-map", "name": "Road map",
        "nodes": [{"node_id": f"road-sample-{i:04}", "label": f"Road sample {i}",
                   "position_m": {"x": float(i), "y": 0.0, "z": 2.0},
                   "semantic": "pickup" if i % 10 == 0 else "corridor"}
                  for i in range(480)],
        "edges": [{"edge_id": f"road-segment-{i:04}", "from_node": f"road-sample-{i:04}",
                   "to_node": f"road-sample-{i+1:04}", "distance_m": 1.0,
                   "minimum_clearance_m": 1.0, "speed_limit_mps": 1.0,
                   "bidirectional": False} for i in range(479)],
        "named_entities": {f"Place {i}": f"road-sample-{i:04}" for i in range(0, 480, 10)},
    })


def test_dense_source_keeps_every_named_place_without_dense_geometry(tmp_path, monkeypatch):
    graph = dense_graph()
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(graph.model_dump_json(), encoding="utf-8")
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps({
        "schema_version": "dronedream.map-semantic.v1", "coordinate_frame": "ENU",
        "entities": [], "collision_primitives": [],
        "traversability": {"exclusions": [{"id": "private-detailed-footprint",
            "polygon": [[i / 10, i / 20] for i in range(45_000)]}]},
    }), encoding="utf-8")
    selection = SimpleNamespace(root=tmp_path, graph=graph_path, semantic=semantic_path)
    monkeypatch.setattr(asset_interpretation, "resolve_versioned_map", lambda _: selection)
    monkeypatch.setattr(
        asset_interpretation, "_read_bound_json", lambda record, root, path: path.read_bytes()
    )
    store = SimpleNamespace(get_asset_version=lambda *args: {})

    source = asset_interpretation.interpretation_source(store, "map", graph.asset_id, "a" * 64)
    encoded = json.dumps(source).encode()
    assert len(json.dumps(graph.model_dump(mode="json")).encode()) > 160_000
    assert len(encoded) < 35_000
    assert set(source["required_source_ids"]) == set(graph.named_entities.values())
    included = {node["node_id"] for node in source["facts"]["nodes"]}
    assert included == set(graph.named_entities.values())
    assert b"private-detailed-footprint" not in encoded
    assert b"road-sample-0001" not in encoded
    assert source["facts"]["topology_summary"]["complete_edge_count"] == 479


def test_summary_does_not_invent_a_direct_link_or_reverse_a_one_way_road():
    facts = asset_interpretation._map_interpretation_facts(dense_graph())
    topology = facts["topology_summary"]
    assert facts["edges"] == []
    assert topology["weak_component_count"] == 1
    assert topology["named_place_degrees"][0] == {
        "node_id": "road-sample-0000", "incoming": 0, "outgoing": 1,
    }
    assert "ignores travel direction" in topology["scope"]


def test_omitted_road_geometry_still_invalidates_interpretation_dependencies():
    graph = dense_graph()
    source = {"kind": "map", "asset_id": graph.asset_id,
              "required_source_ids": sorted(set(graph.named_entities.values())),
              "facts": asset_interpretation._map_interpretation_facts(graph)}
    before = interpretation_lineage(source)
    graph.nodes[1].position_m.y = 3.0
    source["facts"] = asset_interpretation._map_interpretation_facts(graph)
    assert interpretation_lineage(source)["global"] != before["global"]
