"""Catalog integrity tests with temporary fixture bytes, not product-map qualification."""

import hashlib
import json

import pytest

import dronedream_agent_core.assets as module
from dronedream_agent_core.contracts import MapAsset


def _semantic():
    """Keep both axes and names explicit so missing data is introduced only by each test."""
    return {
        "schema_version": "dronedream.map-semantic.v1",
        "coordinate_frame": "ENU",
        "scene_id": "fixture-map",
        "entities": [
            {
                "entity_id": "start",
                "aliases": ["launch pad"],
                "position_m": [0, 0, 2],
                "semantic": "launch",
            },
            {
                "entity_id": "target",
                "aliases": ["pickup pad"],
                "position_m": [3, 0, 2],
                "semantic": "pickup",
            },
        ],
    }


def _graph():
    """Define a tiny admitted graph; no hidden campus topology enters a fixture."""
    return MapAsset.model_validate(
        {
            "asset_id": "fixture-map",
            "name": "Fixture",
            "nodes": [
                {
                    "node_id": "a",
                    "label": "Start",
                    "position_m": {"x": 0, "y": 0, "z": 2},
                    "semantic": "launch",
                },
                {
                    "node_id": "b",
                    "label": "Target",
                    "position_m": {"x": 3, "y": 0, "z": 2},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "ab",
                    "from_node": "a",
                    "to_node": "b",
                    "distance_m": 3,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                },
            ],
            "named_entities": {"start": "a", "target": "b"},
        }
    )


def _write(tmp_path, value):
    """Write only test-owned semantic bytes for the real bounded loader."""
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "position",
    [
        [1, 2],
        [1, 2, 3, 4],
        {"x": 1, "y": 2},
        {"x": 1, "y": 2, "z": 3, "altitude": 5},
        [True, 0, 2],
        ["1", 0, 2],
        [None, 0, 2],
        [10**400, 0, 2],
    ],
)
def test_invalid_anchors_cannot_gain_an_invented_altitude(tmp_path, position):
    """Missing, extra or coerced coordinates are not safe 3-D map evidence."""
    value = _semantic()
    value["entities"][0]["position_m"] = position
    with pytest.raises(module.AssetQualificationError, match="ANCHOR_INVALID"):
        module.load_map_catalog(_write(tmp_path, value))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"scene_id":"first","scene_id":"second"}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e400}',
        b"[]",
        b'{"x":' + b"[" * 70 + b"0" + b"]" * 70 + b"}",
    ],
)
def test_noncanonical_json_cannot_enter_model_context(tmp_path, raw):
    """Ambiguous keys and non-finite/deep/non-object values fail before catalog creation."""
    path = tmp_path / "semantic.json"
    path.write_bytes(raw)
    with pytest.raises(module.AssetQualificationError):
        module.load_map_catalog(path)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("scene_id", None),
        ("scene_id", True),
        ("scene_id", ""),
        ("schema_version", {}),
        ("navigation", []),
        ("navigation", {"segment_ids": [True]}),
        ("known_limits", "not a list"),
        ("known_limits", [False]),
    ],
)
def test_metadata_is_not_silently_coerced(tmp_path, field, bad):
    """Malformed metadata must not be converted into plausible-looking names or lists."""
    value = _semantic()
    value[field] = bad
    with pytest.raises(module.AssetQualificationError):
        module.load_map_catalog(_write(tmp_path, value))


def test_digest_binds_the_parsed_bytes_even_if_path_is_replaced(tmp_path, monkeypatch):
    """Changing the path after a snapshot cannot make its new hash describe old entities."""
    original = _semantic()
    path = _write(tmp_path, original)
    raw = path.read_bytes()
    read = module.read_plugin_file
    calls = []

    def replace_after_read(source, *, limit):
        """Mutate only the fixture after the real descriptor-bound read finishes."""
        calls.append(source)
        captured = read(source, limit=limit)
        replacement = _semantic()
        replacement["scene_id"] = "different-map"
        source.write_text(json.dumps(replacement), encoding="utf-8")
        return captured

    monkeypatch.setattr(module, "read_plugin_file", replace_after_read)
    catalog = module.load_map_catalog(path)
    assert calls == [path]
    assert catalog.scene_id == original["scene_id"]
    assert catalog.semantic_sha256 == hashlib.sha256(raw).hexdigest()
    assert catalog.semantic_sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


def test_conflicting_aliases_require_clarification(tmp_path):
    """List order cannot choose between two different destinations sharing a human name."""
    value = _semantic()
    for entity in value["entities"]:
        entity["aliases"] = ["shared pad"]
    catalog = module.load_map_catalog(_write(tmp_path, value))
    with pytest.raises(module.AssetQualificationError, match="AMBIGUOUS_MAP_ENTITY"):
        module.resolve_map_entity("shared pad", catalog, _graph())
    graph = _graph()
    graph.named_entities["target"] = "a"
    assert module.resolve_map_entity("shared pad", catalog, graph) == "a"


def test_graph_binding_never_imports_built_in_place_names(tmp_path):
    """A graph ID also used by a campus does not import that campus's Chinese aliases."""
    graph = _graph()
    graph.named_entities["cafeteria"] = "b"
    catalog = module.load_map_catalog(_write(tmp_path, _semantic()), qualified_graph=graph)
    entity = next(item for item in catalog.entities if item.entity_id == "cafeteria")
    assert entity.aliases == ["cafeteria"]
    assert not module.map_entity_is_mentioned("go to 餐厅", "cafeteria", catalog)
    assert not module.map_entity_is_mentioned("go to retired-place", "retired-place", catalog)


def test_mutated_graph_is_revalidated_before_binding(tmp_path):
    """Nested dict mutation cannot bypass node-reference validation at catalog admission."""
    graph = _graph()
    graph.named_entities["target"] = "missing"
    with pytest.raises(ValueError, match="unknown node"):
        module.load_map_catalog(_write(tmp_path, _semantic()), qualified_graph=graph)


def test_bound_catalog_does_not_share_graph_position_objects(tmp_path):
    """Later owner mutations cannot alter the snapshot already given to a model."""
    graph = _graph()
    graph.named_entities["extra"] = "b"
    catalog = module.load_map_catalog(_write(tmp_path, _semantic()), qualified_graph=graph)
    graph.nodes[1].position_m.x = 99
    assert next(item for item in catalog.entities if item.entity_id == "extra").position_m.x == 3


def test_aliases_are_not_silently_truncated(tmp_path):
    """Reject an overfull declaration rather than dropping the last user-visible target name."""
    value = _semantic()
    value["entities"][0]["aliases"] = [f"alias {i}" for i in range(24)]
    with pytest.raises(module.AssetQualificationError, match="ALIAS_LIMIT_EXCEEDED"):
        module.load_map_catalog(_write(tmp_path, value))
