"""Regressions for source-ground geometry and truthful qualification metadata."""

import importlib.util
import json
import math
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


def test_official_entrance_keeps_planar_coordinates_without_becoming_a_flight_target():
    prep = _load("prepare-ground-map")
    semantic = {"entities": [], "roads": {"facility_anchors": {}}}
    source = {"planar_landmarks": [{"entity_id": "tpr-entrance-1",
        "name": "Library entrance", "kind": "entrance", "semantic": "facility-anchor",
        "position_m": [12.25, -4.75]}], "service_entrances_source": {"sha256": "source"}}
    prep.append_planar_landmarks(semantic, [(Path("observations.json"), source)])
    first = deepcopy(semantic)
    prep.append_planar_landmarks(semantic, [(Path("observations.json"), source)])
    assert semantic == first
    assert semantic["entities"][0]["position_m"] == [12.25, -4.75]
    assert semantic["entities"][0]["source_pointer"] == "/roads/facility_anchors/tpr-entrance-1"
    assert semantic["roads"]["facility_anchors"]["tpr-entrance-1"] == [12.25, -4.75]
    assert semantic["sources"]["service_entrances"] == {"sha256": "source"}


def test_surface_detour_preserves_narrow_bends_and_does_not_cross_courtyard():
    prep = _load("prepare-ground-map")
    surface = box(0, 0, 20, 20).difference(box(4, 2, 16, 18))
    path = prep.surface_detour(surface, (2., 10.), (18., 10.))
    assert path[0] == (2., 10.) and path[-1] == (18., 10.)
    assert surface.covers(LineString(path))
    disconnected = unary_union([box(0, 0, 3, 20), box(17, 0, 20, 20)])
    assert prep.surface_detour(disconnected, (2., 10.), (18., 10.)) is None


def test_ground_union_removes_internal_seam_without_creating_ground_or_joining_decks():
    from dronedream_agent_core.traversability import TraversableAirspace
    prep = _load("prepare-ground-map")
    layer = {"schema_version": "dronedream.traversability.v1", "required": True,
             "minimum_agl_m": 0., "maximum_agl_m": 5., "corridors": [], "exclusions": [],
             "areas": [{"id": str(i), "kind": "open-space", "ground_z_m": z,
                        "polygon_xy_m": list(shape.exterior.coords)[:-1]}
                       for i, (shape, z) in enumerate([(box(-5, -5, 0, 5), 0),
                                                        (box(0, -5, 5, 5), 0),
                                                        (box(8, -5, 12, 5), 10)])]}
    prep.consolidate_ground_areas(layer)
    assert len(layer["areas"]) == 2
    base = next(a for a in layer["areas"] if a["ground_z_m"] == 0)
    assert Polygon(base["polygon_xy_m"]).equals(box(-5, -5, 5, 5))
    airspace = TraversableAirspace({"traversability": layer}, [], radius_m=.38,
                                  half_height_m=.3, clearance_cap_m=1.3)
    assert airspace.clearance((0., 0., 2.5))[0] == pytest.approx(1.3)
    assert airspace.clearance((10., 0., 2.5))[0] < 0


def test_english_function_catalog_binds_existing_handoffs_without_geometry_changes():
    prep = _load("prepare-ground-map")
    graph = {
        "nodes": [
            {"node_id": "chemistry-handoff", "label": "Chemicum south handoff point",
             "position_m": {"x": 43, "y": 55, "z": 2}, "semantic": "pickup"},
            {"node_id": "library-handoff", "label": "Kumpulan kampuskirjasto outdoor handoff",
             "position_m": {"x": -66, "y": 46, "z": 2}, "semantic": "delivery"},
            {"node_id": "unknown-building", "label": "Building 123 outdoor handoff",
             "position_m": {"x": 9, "y": 8, "z": 2}, "semantic": "delivery"},
        ],
        "edges": [{"from_node": "chemistry-handoff", "to_node": "library-handoff"}],
        "named_entities": {
            "Chemicum": "chemistry-handoff", "化学楼": "chemistry-handoff",
            "Kumpula Campus Library": "library-handoff",
            "Väinö Auerin katu 12": "chemistry-handoff",
            "Building 123": "unknown-building",
        },
    }
    semantic = {
        "entities": [{"entity_id": "chemicum", "name": "Chemicum", "kind": "building",
                      "aliases": ["Chemicum", "化学楼"],
                      "position_m": {"x": 50, "y": 50, "z": 0}}],
        "traversability": {"exclusions": [{"id": "tree", "polygon_xy_m": [[1, 2]]}]},
    }
    old_graph, old_semantic = deepcopy(graph), deepcopy(semantic)
    prep.complete_english_place_catalog(semantic, graph)
    assert graph["named_entities"]["chemistry building"] == "chemistry-handoff"
    assert graph["named_entities"]["department of chemistry"] == "chemistry-handoff"
    assert graph["named_entities"]["Kumpula library"] == "library-handoff"
    assert graph["named_entities"]["Vaino Auerin katu 12"] == "chemistry-handoff"
    assert "chemistry building" in semantic["entities"][0]["aliases"]
    assert graph["nodes"][1]["label"] == "Kumpula Campus Library outdoor handoff"
    assert semantic["entities"][0]["position_m"] == {"x": 50, "y": 50, "z": 0}
    assert graph["edges"] == old_graph["edges"]
    assert [{k: v for k, v in n.items() if k != "label"} for n in graph["nodes"]] == [
        {k: v for k, v in n.items() if k != "label"} for n in old_graph["nodes"]
    ]
    assert semantic["traversability"] == old_semantic["traversability"]
    assert {k for k, v in graph["named_entities"].items() if v == "unknown-building"} == {
        "Building 123"
    }
    repeated = deepcopy((semantic, graph))
    prep.complete_english_place_catalog(semantic, graph)
    assert (semantic, graph) == repeated


def test_duplicate_source_school_keeps_one_english_function_binding():
    prep = _load("prepare-ground-map")
    position = {"x": 12.354, "y": -115.962, "z": 2.0}
    graph = {
        "nodes": [
            {"node_id": "school-city", "label": "School outdoor handoff", "position_m": position},
            {"node_id": "school-osm", "label": "自然科学高中 outdoor handoff",
             "position_m": position},
        ],
        "named_entities": {
            "Helsinki Upper Secondary School of Natural Sciences": "school-city",
            "自然科学高中": "school-osm",
        },
    }
    prep.complete_english_place_catalog({"entities": []}, graph)
    assert graph["named_entities"]["science high school"] == "school-city"
    assert {target for name, target in graph["named_entities"].items()
            if name.casefold() == "natural sciences high school"} == {"school-city"}
    assert graph["named_entities"]["自然科学高中"] == "school-osm"


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


@pytest.mark.parametrize("yaw", [0, .43])
def test_concave_building_keeps_courtyard_open_without_erasing_any_wall(
    tmp_path, monkeypatch, yaw
):
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    prep = _load("prepare-ground-map")
    ring = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
    c, s = math.cos(yaw), math.sin(yaw)
    def rotate(x, y):
        return x * c - y * s, x * s + y * c

    ring = [rotate(x, y) for x, y in ring]
    (tmp_path / "meshes").mkdir()
    original = {
        "name": "courtyard-collision",
        "semantic": "building",
        "center_x": rotate(5, 5)[0],
        "center_y": rotate(5, 5)[1],
        "center_z": 4,
        "size_x": 10,
        "size_y": 10,
        "size_z": 8,
        "yaw_rad": yaw,
    }
    from ground_map_extent_buildings import _write_mesh
    _write_mesh(tmp_path / "meshes/kumpula-buildings.obj", {
        "courtyard": {"polygon": Polygon(ring), "primitive": original, "material": "context"}
    })
    semantic = {"collision_primitives": [original], "fidelity": {}}
    (tmp_path / "collision-primitives.json").write_text(json.dumps({"primitives": [original]}))
    for name in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        (tmp_path / name).write_text(
            '<sdf version="1.9"><model name="kumpula_campus">'
            '<link name="building-collision-old"/>'
            '<link name="building-visuals"><visual name="original">'
            '<geometry><mesh><uri>meshes/kumpula-buildings.obj</uri></mesh></geometry>'
            '</visual></link></model></sdf>'
        )
    visual_before = ET.tostring(ET.parse(tmp_path / "model.sdf").find(".//visual"))
    obj_before = (tmp_path / "meshes/kumpula-buildings.obj").read_bytes()
    exact = prep.refine_building_collisions(tmp_path, semantic)
    covered = unary_union([Polygon(prep.footprint(p)) for p in semantic["collision_primitives"]])
    assert Polygon(ring).difference(covered).area < 1e-8
    assert covered.distance(Point(*rotate(5, 7))) >= 1.5
    assert Polygon(exact[0]["polygon_xy_m"]).equals(Polygon(ring))
    assert len(semantic["collision_primitives"]) > 1
    assert prep.read(tmp_path / "collision-primitives.json")["primitives"] == (
        semantic["collision_primitives"]
    )
    for name in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        root = ET.parse(tmp_path / name).getroot()
        collisions = root.findall(".//collision")
        assert len(collisions) == 1
        collision = collisions[0]
        pose = list(map(float, collision.find("pose").text.split()))
        assert pose[:5] == [0, 0, 4, 0, 0]
        def mesh_rotate(x, y):
            return x * math.cos(pose[5]) - y * math.sin(pose[5]), x * math.sin(pose[5]) + y * math.cos(pose[5])
        raw = (tmp_path / collision.find("geometry/mesh/uri").text).read_text()
        vertices, caps = [], []
        for line in raw.splitlines():
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:])))
            elif line.startswith("f "):
                points = [vertices[int(t.split("//")[0]) - 1] for t in line.split()[1:]]
                if all(p[2] == 4 for p in points):
                    caps.append(Polygon([mesh_rotate(x, y) for x, y, _ in points]))
        assert unary_union(caps).symmetric_difference(covered).area < 1e-10
        visual = root.find(".//visual")
        # Pretty-printing changes whitespace, never the original visual contents.
        for node in visual.iter():
            if node.text is not None and not node.text.strip():
                node.text = None
            node.tail = None
        assert ET.tostring(visual) == visual_before
    assert (tmp_path / "meshes/kumpula-buildings.obj").read_bytes() == obj_before
    prep.write(tmp_path / "semantic.json", semantic)
    # The geometry used by both route validation and worker static-occupancy
    # classification still sees all cells, even though SDF has only one mesh.
    geometry = KnownMapMetricPlanner(
        graph=None, semantic_path=tmp_path / "semantic.json",
        vehicle_diameter_m=.76, vehicle_height_m=.516,
    )
    assert len(geometry.primitives) == len(semantic["collision_primitives"])
    assert geometry.clearance((*rotate(1, 1), 4)) < 0
    assert geometry.clearance((*rotate(5, 7), 4)) > 0


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


def test_route_surface_preserves_claimed_body_clearance_at_obstacle_corner():
    prep = _load("prepare-ground-map")
    permitted, obstacle = box(-10, -10, 10, 10), box(-1, -1, 1, 1)
    surface = prep.route_surface(permitted, obstacle)
    assert surface.distance(obstacle) >= 1.25 - 1e-9
    assert surface.boundary.distance(permitted.boundary) >= 1.25 - 1e-9
    assert not surface.covers(LineString([(-5, 2), (5, 2)]))
    assert surface.covers(LineString([(-5, 3), (5, 3)]))


def test_preparing_again_retains_task_locations_without_old_junctions():
    prep = _load("prepare-ground-map")
    graph = {"nodes": [
        {"node_id": "launch", "semantic": "launch"},
        {"node_id": "pickup", "semantic": "pickup"},
        {"node_id": "ground-junction-0000", "semantic": "corridor"},
        {"node_id": "verified-001", "semantic": "pickup"},
    ], "named_entities": {"office": "launch"}}
    assert [n["node_id"] for n in prep.operational_nodes(graph)] == ["launch", "pickup"]


def test_missing_connector_is_recovered_by_a_ground_detour(tmp_path):
    prep = _load("prepare-ground-map")
    graph = {"asset_id": "detour", "name": "Detour", "named_entities": {"launch": "a", "delivery": "b"},
             "nodes": [{"node_id": name, "label": name, "semantic": kind,
                        "position_m": {"x": x, "y": y, "z": 3.}}
                       for name, kind, x, y in [("a", "launch", 0., 0.), ("b", "delivery", 10., 0.),
                                                ("c", "corridor", 0., 6.)]],
             "edges": [{"edge_id": "road", "from_node": "a", "to_node": "c", "distance_m": 6.,
                        "minimum_clearance_m": .8, "speed_limit_mps": 2., "bidirectional": True}]}
    semantic = {"collision_primitives": [{"name": "ground", "semantic": "terrain",
        "center_x": 5., "center_y": 3., "center_z": -.1,
        "size_x": 30., "size_y": 30., "size_z": .2}], "traversability": {
        "schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0., "maximum_agl_m": 12., "areas": [],
        "corridors": [{"id": "bent-road", "kind": "road", "ground_z_m": 0., "width_m": 4.,
                       "centerline_xy_m": [[0., 0.], [0., 6.], [10., 6.], [10., 0.]]}],
        "exclusions": [{"id": "building", "kind": "building",
                        "polygon_xy_m": [[3., -2.], [7., -2.], [7., 3.], [3., 3.]]}]}}
    path = tmp_path / "semantic.json"
    prep.write(path, semantic)
    assert prep.connect_task_locations(graph, path, "a" * 64) == []
    assert len(graph["edges"]) >= 3
    by_id = {n["node_id"]: n["position_m"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        points = [by_id[edge[k]] for k in ("from_node", "to_node")]
        assert not LineString([(p["x"], p["y"]) for p in points]).intersects(box(3, -2, 7, 3))
    assert by_id["b"] == {"x": 10., "y": 0., "z": 3.}


def test_disconnected_ground_is_reported_without_exhausting_a_three_dimensional_search(tmp_path, monkeypatch):
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner
    prep = _load("prepare-ground-map")
    graph = {"asset_id": "separated", "name": "Separated", "named_entities": {"launch": "a", "delivery": "b"},
             "nodes": [{"node_id": key, "label": key, "semantic": kind,
                        "position_m": {"x": x, "y": 0., "z": 3.}}
                       for key, kind, x in [("a", "launch", 0.), ("b", "delivery", 20.),
                                           ("c", "corridor", 1.)]],
             "edges": [{"edge_id": "source-road", "from_node": "a", "to_node": "c",
                        "distance_m": 1., "minimum_clearance_m": .8,
                        "speed_limit_mps": 2., "bidirectional": True}]}
    semantic = {"collision_primitives": [{"name": "ground", "semantic": "terrain",
        "center_x": 10., "center_y": 0., "center_z": -.1,
        "size_x": 40., "size_y": 20., "size_z": .2}], "traversability": {
        "schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0., "maximum_agl_m": 12., "corridors": [], "exclusions": [],
        "areas": [{"id": str(i), "kind": "open-space", "ground_z_m": 0.,
                   "polygon_xy_m": list(box(x-4, -4, x+4, 4).exterior.coords)[:-1]}
                  for i, x in enumerate((0., 20.))]}}
    path = tmp_path / "semantic.json"
    prep.write(path, semantic)

    def must_not_search(*args, **kwargs):
        pytest.fail("No altitude can connect these separated operating volumes")

    monkeypatch.setattr(KnownMapMetricPlanner, "plan", must_not_search)
    unresolved = prep.connect_task_locations(graph, path, "a" * 64)
    assert [item["node_id"] for item in unresolved] == ["b"]
    assert unresolved[0]["reason"] == "observed-ground-disconnected-at-vehicle-envelope-and-clearance"
    assert len(graph["edges"]) == 1
