"""Regressions for source-ground geometry and truthful qualification metadata."""

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

pytest.importorskip("shapely")
from shapely.geometry import Point, Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concave_building_keeps_courtyard_open_without_erasing_any_wall(tmp_path):
    prep = _load("prepare-ground-map")
    ring = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes/kumpula-buildings.obj").write_text(
        "g courtyard\n" + "\n".join(f"v {x} {y} {z}" for z in (0, 8) for x, y in ring)
    )
    original = {
        "name": "courtyard-collision",
        "semantic": "building",
        "center_x": 5,
        "center_y": 5,
        "center_z": 4,
        "size_x": 10,
        "size_y": 10,
        "size_z": 8,
        "yaw_rad": 0,
    }
    semantic = {"collision_primitives": [original], "fidelity": {}}
    (tmp_path / "collision-primitives.json").write_text(json.dumps({"primitives": [original]}))
    for name in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        (tmp_path / name).write_text(
            '<sdf version="1.9"><model name="kumpula_campus">'
            '<link name="building-collision-old"/></model></sdf>'
        )
    exact = prep.refine_building_collisions(tmp_path, semantic)
    covered = unary_union([Polygon(prep.footprint(p)) for p in semantic["collision_primitives"]])
    assert Polygon(ring).difference(covered).area < 1e-8
    assert covered.distance(Point(5, 7)) >= 1.5
    assert exact[0]["polygon_xy_m"] == ring
    expected = {p["name"]: p for p in semantic["collision_primitives"]}
    for name in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        root = ET.parse(tmp_path / name).getroot()
        collisions = root.findall(".//collision")
        assert len(collisions) == len(expected)
        for collision in collisions:
            primitive = expected[collision.get("name")]
            assert list(map(float, collision.find("pose").text.split())) == pytest.approx(
                [primitive["center_x"], primitive["center_y"], 4, 0, 0, 0]
            )
            assert list(
                map(float, collision.find("geometry/box/size").text.split())
            ) == pytest.approx([primitive["size_x"], primitive["size_y"], 8])


def test_candidate_never_inherits_old_flight_success(tmp_path):
    prep = _load("prepare-ground-map")
    summary = {
        "fidelity": {"qualified_routes": 1},
        "gazebo_runtime_verified": True,
        "simulation_execution_ready": True,
        "px4_mission_smoke_verified": True,
        "semantic_sha256": "obsolete",
        "qualification_evidence_sha256": "obsolete",
    }
    prep.write(tmp_path / "summary.json", summary)
    prep.write(
        tmp_path / "qualification-evidence.json", {"status": "verified", "gates": {"landed": True}}
    )
    for name in ("model.sdf", "world.physics.sdf", "world.perception.sdf", "semantic.json"):
        (tmp_path / name).write_text("new source")
    prep.update_candidate_summary(
        tmp_path, {"fidelity": {"qualified_routes": 0}, "collision_primitives": []}
    )
    result = prep.read(tmp_path / "summary.json")
    proof = prep.read(tmp_path / "qualification-evidence.json")
    assert proof["status"] == "candidate-requires-qualification"
    assert proof["gates"] == {}
    assert result["fidelity"]["qualified_routes"] == 0
    assert not result["simulation_execution_ready"]
    assert not result["gazebo_runtime_verified"]
    assert not result["px4_mission_smoke_verified"]
    assert result["semantic_sha256"] != "obsolete"


def test_release_builder_rejects_roof_shortcut_without_rewriting_graph(tmp_path):
    builder = _load("build-default-assets")
    graph = {
        "asset_id": "test-map",
        "name": "test map",
        "nodes": [
            {
                "node_id": "a",
                "label": "a",
                "semantic": "launch",
                "position_m": {"x": 0, "y": 0, "z": 3},
            },
            {
                "node_id": "b",
                "label": "b",
                "semantic": "pickup",
                "position_m": {"x": 8, "y": 0, "z": 3},
            },
        ],
        "edges": [
            {
                "edge_id": "shortcut",
                "from_node": "a",
                "to_node": "b",
                "distance_m": 8,
                "minimum_clearance_m": 0.8,
                "speed_limit_mps": 2,
            }
        ],
        "named_entities": {"a": "a", "b": "b"},
    }
    semantic = {
        "coordinate_frame": "ENU",
        "collision_primitives": [
            {
                "name": "ground",
                "semantic": "terrain",
                "center_x": 5,
                "center_y": 3,
                "center_z": -0.1,
                "size_x": 30,
                "size_y": 30,
                "size_z": 0.2,
            }
        ],
        "traversability": {
            "schema_version": "dronedream.traversability.v1",
            "required": True,
            "minimum_agl_m": 0,
            "maximum_agl_m": 12,
            "areas": [
                {
                    "id": "yard",
                    "kind": "parking",
                    "ground_z_m": 0,
                    "polygon_xy_m": [[-5, -5], [15, -5], [15, 10], [-5, 10]],
                }
            ],
            "corridors": [],
            "exclusions": [
                {
                    "id": "roof",
                    "kind": "building",
                    "polygon_xy_m": [[3, -1], [5, -1], [5, 1], [3, 1]],
                }
            ],
        },
    }
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic))
    before = json.dumps(graph, sort_keys=True)
    with pytest.raises(ValueError, match="DEFAULT_MAP_EDGE_OUTSIDE_TRAVERSABLE_GROUND:shortcut"):
        builder._validate_ground_route_graph(graph=graph, semantic_path=path)
    assert json.dumps(graph, sort_keys=True) == before
    # A genuine road around the same roof is retained as authored.
    graph["nodes"].extend(
        [
            {
                "node_id": "c",
                "label": "corner",
                "semantic": "corridor",
                "position_m": {"x": 0, "y": 4, "z": 3},
            },
            {
                "node_id": "d",
                "label": "corner",
                "semantic": "corridor",
                "position_m": {"x": 8, "y": 4, "z": 3},
            },
        ]
    )
    graph["edges"] = [
        {
            "edge_id": f"road-{i}",
            "from_node": a,
            "to_node": b,
            "distance_m": length,
            "minimum_clearance_m": 0.8,
            "speed_limit_mps": 2,
        }
        for i, (a, b, length) in enumerate([("a", "c", 4), ("c", "d", 8), ("d", "b", 4)])
    ]
    before = json.dumps(graph, sort_keys=True)
    builder._validate_ground_route_graph(graph=graph, semantic_path=path)
    assert json.dumps(graph, sort_keys=True) == before
