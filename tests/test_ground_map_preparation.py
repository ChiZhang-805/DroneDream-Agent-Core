"""Regressions for source-ground geometry and truthful qualification metadata."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

pytest.importorskip("shapely")
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observed_footway_joins_middle_of_long_street_without_inventing_ground():
    prep = _load("prepare-ground-map")
    street = LineString([(0, 0), (100, 0)])
    footway = LineString([(47, 8), (47, .2)])
    usable = unary_union([street.buffer(1), footway.buffer(1)])
    noded = prep.node_observed_corridor_joins([street, footway], usable)
    edges = [list(line.coords) for line in noded.geoms]
    assert [(0., 0.), (47., 0.)] in edges
    assert [(47., 0.), (100., 0.)] in edges
    assert [(47., .2), (47., 0.)] in edges
    assert usable.covers(noded)
    assert noded.covers(street) and noded.covers(footway)


def test_observed_joins_do_not_cross_planted_median_or_extend_beyond_bound():
    prep = _load("prepare-ground-map")
    street = LineString([(0, 0), (100, 0)])
    near = LineString([(47, 8), (47, 1)])
    far = LineString([(70, 8), (70, 3)])
    usable = unary_union([street.buffer(1), near.buffer(1), far.buffer(1)])
    usable = usable.difference(box(46, .4, 48, .6))
    noded = prep.node_observed_corridor_joins([street, near, far], usable)
    assert noded.equals(unary_union([street, near, far]))


def test_existing_handoff_revisions_reach_semantic_context_with_names_preserved():
    prep = _load("prepare-ground-map")
    entities = [
        {
            "entity_id": "exactum-delivery-zone",
            "name": "Exactum outdoor delivery zone",
            "aliases": ["Exactum handoff point", "信息科学楼交付点"],
            "semantic": "delivery",
            "position_m": {"x": -75.122, "y": -43.307, "z": 2.0},
        },
        {
            "entity_id": "physicum-delivery-zone",
            "name": "Physicum outdoor delivery zone",
            "aliases": ["Physicum handoff point", "物理楼交付点"],
            "kind": "delivery",
            "position_m": {"x": 54.629, "y": 35.294, "z": 2.0},
        },
    ]
    expected = deepcopy(entities)
    positions = [{"x": -93.68, "y": -24.274, "z": 2.0}, {"x": 22.218, "y": 68.372, "z": 2.0}]
    graph = {
        "nodes": [
            dict(node_id=e["entity_id"], position_m=p)
            for e, p in zip(entities, positions, strict=True)
        ]
    }
    semantic = {"entities": entities}
    prep.synchronize_semantic_node_positions(semantic, graph)
    for entity, position in zip(expected, positions, strict=True):
        entity["position_m"] = position
    assert semantic["entities"] == expected
    # These are separate documents; later graph work cannot silently edit the
    # already synchronized semantic description through a shared dictionary.
    assert entities[0]["position_m"] is not graph["nodes"][0]["position_m"]


def test_ground_pad_and_building_center_are_not_replaced_by_aliased_flight_nodes():
    prep = _load("prepare-ground-map")
    semantic = {
        "entities": [
            {
                "entity_id": "office-launch-pad",
                "name": "Launch pad",
                "position_m": {"x": -41, "y": -83, "z": 0},
            },
            {
                "entity_id": "physicum",
                "name": "Physicum",
                "position_m": {"x": 38.628, "y": 7.581, "z": 6.8},
            },
        ]
    }
    original = deepcopy(semantic)
    graph = {
        "nodes": [
            {"node_id": "verified-000", "position_m": {"x": -41, "y": -83, "z": 1.5}},
            {"node_id": "physicum-delivery-zone", "position_m": {"x": 22.218, "y": 68.372, "z": 2}},
        ],
        "named_entities": {
            "office-launch-pad": "verified-000",
            "physicum": "physicum-delivery-zone",
        },
    }
    prep.synchronize_semantic_node_positions(semantic, graph)
    assert semantic == original


def test_ground_support_places_explicit_start_on_physical_floor(tmp_path):
    from dronedream_agent_core.contracts import VehicleAsset
    from dronedream_agent_core.runtime_bindings import resolve_grounded_vehicle_model_root

    prep = _load("prepare-ground-map")
    sdf = ('<sdf version="1.9"><model name="kumpula_campus">'
           '<link name="ground"><collision name="ground-collision">'
           '<pose>0 0 -0.1 0 0 0</pose><geometry><box>'
           '<size>500 500 0.2</size></box></geometry></collision></link></model></sdf>')
    for name in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        (tmp_path / name).write_text(sdf)
    prep.write(tmp_path / "collision-primitives.json", {"primitives": []})
    semantic = {
        "schema_version": "dronedream.map-semantic.v1",
        "collision_primitives": [],
        "runtime_bindings": {
            "simulator": "gazebo", "vehicle_spawn": {"x": 22, "y": 68, "z": 0},
            "mission_launch_waypoint": {"x": 22, "y": 68, "z": 2},
        },
    }
    vehicle = VehicleAsset(
        asset_id="x500", name="X500", dry_mass_kg=2, max_takeoff_mass_kg=2.1,
        body_radius_m=.38, body_height_m=.43,
        collision_center_offset_model_m={"x": 0, "y": 0, "z": .228},
        max_speed_mps=6, max_acceleration_mps2=1.5, qualified_range_m=800,
        reserve_battery_percent=30, max_pickup_payload_kg=.04, sensors=["depth"],
    )
    prep.include_physical_ground_support(tmp_path, semantic)
    root = resolve_grounded_vehicle_model_root(semantic, vehicle)
    assert root.x == 22 and root.y == 68
    assert root.z + .228 - .43 / 2 == pytest.approx(0)
    assert semantic["runtime_bindings"]["mission_launch_waypoint"]["z"] == 2
    assert (prep.read(tmp_path / "collision-primitives.json")["primitives"]
            == semantic["collision_primitives"])
    # Re-running a preparation step does not grow duplicate physical surfaces.
    prep.include_physical_ground_support(tmp_path, semantic)
    assert len(semantic["collision_primitives"]) == 1
    (tmp_path / "world.physics.sdf").write_text(sdf.replace("-0.1", "0.9"))
    with pytest.raises(ValueError, match="worlds disagree"):
        prep.include_physical_ground_support(tmp_path, semantic)


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
