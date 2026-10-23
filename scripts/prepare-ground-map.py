"""Prepare a reviewable ground-constrained map; never manufacture flight evidence.

The source road survey is image-registered, not a flight track. Every new edge is
geometry-derived. This output is a candidate which must pass pair qualification
before the release builder can promote it. Requires the map-preparation extra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def xy(value):
    return [float(value["x"]), float(value["y"])]


def node_observed_corridor_joins(lines, usable, maximum_join_m=2.0):
    """Join near-miss survey endpoints within already observed usable ground.

    A footway can end beside the middle of a long street segment. Split that
    segment at the actual projection; its distant existing vertices do not
    determine connectivity. The join cannot cross an excluded planted median
    or create a traversable surface outside the observed road footprints.
    """
    cuts = [{0.0, line.length} for line in lines]
    joins = []
    for index, line in enumerate(lines):
        for endpoint in (Point(line.coords[0]), Point(line.coords[-1])):
            for target_index, target in enumerate(lines):
                if target_index == index or endpoint.distance(target) > maximum_join_m:
                    continue
                distance = target.project(endpoint)
                projection = target.interpolate(distance)
                join = LineString([endpoint.coords[0], projection.coords[0]])
                if not usable.covers(join):
                    continue
                cuts[target_index].add(distance)
                if join.length > 1e-8:
                    joins.append(join)
    segments = []
    for line, distances in zip(lines, cuts, strict=True):
        ordered = sorted(distances)
        for start, end in zip(ordered, ordered[1:], strict=False):
            if end - start > 1e-8:
                segments.append(substring(line, start, end))
    return unary_union(segments + joins)


def synchronize_semantic_node_positions(semantic, graph):
    """Expose one operational position for each identity shared with the graph.

    Building centers and separately identified ground pads retain their own
    coordinates; only an exact entity / node ID denotes the same operation.
    """
    positions = {node["node_id"]: node["position_m"] for node in graph["nodes"]}
    for entity in semantic["entities"]:
        position = positions.get(entity["entity_id"])
        if position is not None:
            entity["position_m"] = dict(position)


def include_physical_ground_support(gazebo, semantic):
    """Expose the actual horizontal SDF ground box to landing and route geometry."""
    supports = []
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        root = ET.parse(gazebo / filename).getroot()
        model = root.find("model")
        if model is None:
            model = root.find("world/model[@name='kumpula_campus']")
        link = model.find("link[@name='ground']")
        collision = link.find("collision[@name='ground-collision']")
        x = y = z = yaw = 0.0
        for element in (model, link, collision):
            pose = element.find("pose")
            px, py, pz, roll, pitch, angle = map(
                float, pose.text.split() if pose is not None else [0] * 6
            )
            if roll != 0 or pitch != 0 or (pose is not None and pose.get("relative_to")):
                raise ValueError("Campus ground support must use horizontal parent-frame poses")
            x, y, z = (
                x + px * math.cos(yaw) - py * math.sin(yaw),
                y + px * math.sin(yaw) + py * math.cos(yaw),
                z + pz,
            )
            yaw += angle
        sx, sy, sz = map(float, collision.find("geometry/box/size").text.split())
        supports.append({
            "name": "ground-collision", "semantic": "terrain",
            "center_x": x, "center_y": y, "center_z": z,
            "size_x": sx, "size_y": sy, "size_z": sz, "yaw_rad": yaw,
        })
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError("Campus worlds disagree on physical ground support")
    support = supports[0]
    for key in ("collision_primitives", "runtime_collision_primitives"):
        if key in semantic:
            semantic[key] = [p for p in semantic[key] if p.get("name") != support["name"]]
            semantic[key].append(dict(support))
    collision_path = gazebo / "collision-primitives.json"
    document = read(collision_path)
    document["primitives"] = semantic["collision_primitives"]
    write(collision_path, document)


def footprint(primitive):
    cx, cy = primitive["center_x"], primitive["center_y"]
    sx, sy = primitive["size_x"] / 2, primitive["size_y"] / 2
    c, s = math.cos(primitive.get("yaw_rad", 0)), math.sin(primitive.get("yaw_rad", 0))
    return [
        [cx + x * c - y * s, cy + x * s + y * c]
        for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))
    ]


def update_candidate_summary(gazebo, semantic):
    """Invalidate inherited proof and bind the candidate's actual files."""
    (gazebo / "README.md").write_text(
        "# Kumpula Campus runtime map\n\n"
        "Registered local-ENU exterior model using City of Helsinki building data, "
        "the 2025 campus orthophoto and observed roads and paths. "
        "Adjoining coverage, when present, uses attributed HSY/city building and "
        "2023 orthophoto sources; original campus pixels and the metric origin are preserved. "
        "Source dates and geographic registration are recorded in semantic.json.\n\n"
        "Planning and all simulation worlds share footprint-derived building collisions "
        "and observed HSY 2024 tree-canopy geometry. Buildings and canopy remain excluded "
        "at every flight altitude. Routes use connected observed roads, paths and open "
        "ground, with full-aircraft clearance checked before execution.\n\n"
        "The ground is a flat registered plane. Building and canopy heights are explicitly "
        "derived exterior proxies, not measured roof or branch clearance. This model does "
        "not claim interior, doorway, power-line or dynamic-obstacle coverage. "
        "New map geometry requires a new physical qualification; old flight receipts do "
        "not qualify it.\n",
        encoding="utf-8", newline="\n",
    )
    write(
        gazebo / "qualification-evidence.json",
        {
            "schema_version": "dronedream.kumpula-campus-qualification-evidence.v1",
            "status": "candidate-requires-qualification",
            "reason": (
                "Road domain and building collision geometry changed; "
                "previous flight receipts do not qualify this candidate."
            ),
            "gates": {},
        },
    )
    summary = read(gazebo / "summary.json")
    summary["fidelity"] = semantic["fidelity"]
    if "bounds_m" in semantic:
        summary["bounds_m"] = semantic["bounds_m"]
    if "extent_enu_m" in semantic:
        summary["extent_enu_m"] = semantic["extent_enu_m"]
    summary["collision_primitive_count"] = len(semantic["collision_primitives"])
    for key in (
        "gazebo_runtime_verified",
        "px4_mission_smoke_verified",
        "simulation_execution_ready",
    ):
        summary[key] = False
    for key in list(summary):
        if key.endswith("_sha256"):
            del summary[key]
    files = {
        str(path.relative_to(gazebo)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(gazebo.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }
    summary["package_file_count"] = len(files)
    summary["package_file_sha256"] = files
    for key, filename in {
        "model_sdf_sha256": "model.sdf",
        "semantic_sha256": "semantic.json",
        "physics_world_sdf_sha256": "world.physics.sdf",
        "perception_world_sdf_sha256": "world.perception.sdf",
        "qualification_evidence_sha256": "qualification-evidence.json",
    }.items():
        summary[key] = files[filename]
    summary["package_manifest_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    write(gazebo / "summary.json", summary)


def refine_building_collisions(gazebo, semantic, maximum_cell_m=0.5):
    """Conservatively cover the actual city footprint, preserving open courtyards.

    Leaf cells intersecting the footprint are retained; no part of the city
    footprint is removed. Physics and planning receive exactly the same boxes.
    """
    groups, current = {}, None
    for line in (gazebo / "meshes/kumpula-buildings.obj").read_text("utf-8").splitlines():
        if line.startswith("g "):
            current = line[2:]
            groups[current] = []
        elif line.startswith("v ") and current is not None:
            groups[current].append(tuple(map(float, line.split()[1:4])))
    original = {
        p["name"].removesuffix("-collision"): p
        for p in semantic["collision_primitives"]
        if p.get("semantic") == "building"
    }
    refined, exact = [], []
    for name, primitive in original.items():
        vertices = groups[name]
        floor = min(v[2] for v in vertices)
        ring = [(v[0], v[1]) for v in vertices if abs(v[2] - floor) < 1e-6]
        polygon = Polygon(ring)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Invalid city footprint: {name}")
        angle = primitive.get("yaw_rad", 0.0)
        rotated = affinity.rotate(polygon, -angle, origin=(0, 0), use_radians=True)
        pending, cells = [rotated.bounds], []
        while pending:
            x0, y0, x1, y1 = pending.pop()
            rectangle = box(x0, y0, x1, y1)
            if not rotated.intersects(rectangle):
                continue
            if rotated.covers(rectangle) or max(x1 - x0, y1 - y0) <= maximum_cell_m:
                cells.append((x0, y0, x1, y1))
            elif x1 - x0 >= y1 - y0:
                mid = (x0 + x1) / 2
                pending.extend([(x0, y0, mid, y1), (mid, y0, x1, y1)])
            else:
                mid = (y0 + y1) / 2
                pending.extend([(x0, y0, x1, mid), (x0, mid, x1, y1)])
        # Adjacent dyadic cells share exact cuts. Coalesce them without growing
        # the covered region; thousands of tiny physics shapes are unnecessary.
        while True:
            before = len(cells)
            for axis in (0, 1):
                grouped = {}
                for cell in cells:
                    other = 1 - axis
                    key = (cell[other], cell[other + 2])
                    grouped.setdefault(key, []).append((cell[axis], cell[axis + 2]))
                merged = []
                for (other_low, other_high), intervals in grouped.items():
                    intervals.sort()
                    low, high = intervals[0]
                    for next_low, next_high in intervals[1:] + [(math.inf, math.inf)]:
                        if next_low <= high + 1e-10:
                            high = max(high, next_high)
                        else:
                            merged.append(
                                (low, other_low, high, other_high)
                                if axis == 0
                                else (other_low, low, other_high, high)
                            )
                            low, high = next_low, next_high
                cells = merged
            if len(cells) == before:
                break
        union = unary_union([box(*cell) for cell in cells])
        if rotated.difference(union.buffer(1e-8)).area > 1e-6:
            raise ValueError(f"Footprint coverage lost: {name}")
        c, s = math.cos(angle), math.sin(angle)
        for i, (x0, y0, x1, y1) in enumerate(cells):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            refined.append(
                {
                    **primitive,
                    "name": f"{name}-cell-{i:04d}",
                    "center_x": cx * c - cy * s,
                    "center_y": cx * s + cy * c,
                    "size_x": x1 - x0,
                    "size_y": y1 - y0,
                }
            )
        exact.append({"id": name, "kind": "building", "polygon_xy_m": ring})
    semantic["collision_primitives"] = [
        p for p in semantic["collision_primitives"] if p.get("semantic") != "building"
    ] + refined
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        path = gazebo / filename
        tree = ET.parse(path)
        model = tree.getroot().find("model")
        if model is None:
            model = tree.getroot().find("world/model[@name='kumpula_campus']")
        if model is None:
            raise ValueError(f"Missing campus physical model: {filename}")
        for link in list(model.findall("link")):
            if link.get("name", "").startswith("building-collision-"):
                model.remove(link)
        link = ET.SubElement(model, "link", {"name": "building-collision-footprints"})
        for p in refined:
            col = ET.SubElement(link, "collision", {"name": p["name"]})
            ET.SubElement(col, "pose").text = (
                f"{p['center_x']:.10f} {p['center_y']:.10f} {p['center_z']:.10f} "
                f"0 0 {p.get('yaw_rad', 0):.10f}"
            )
            geom = ET.SubElement(col, "geometry")
            shape = ET.SubElement(geom, "box")
            ET.SubElement(
                shape, "size"
            ).text = f"{p['size_x']:.10f} {p['size_y']:.10f} {p['size_z']:.10f}"
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    collision_path = gazebo / "collision-primitives.json"
    document = read(collision_path)
    document["primitives"] = semantic["collision_primitives"]
    document["building_collision_source"] = "conservative-city-footprint-cell-cover"
    document["maximum_boundary_cell_m"] = maximum_cell_m
    write(collision_path, document)
    semantic["fidelity"]["building_collisions"] = "conservative-city-footprint-cell-cover"
    return exact


def prepare(
    package: Path, survey_path: Path, observations_paths: list[Path], output: Path,
    extent_source: Path | None = None,
    building_source: Path | None = None,
):
    from ground_map_canopy import install_canopy_obstacles

    if output.exists():
        raise FileExistsError(f"Candidate destination already exists: {output}")
    survey = read(survey_path)
    observation_sources = [(path, read(path)) for path in observations_paths]
    observations = {
        key: []
        for key in (
            "corridors",
            "areas",
            "exclusions",
            "handoff_corrections",
            "unresolved_observations",
        )
    }
    removed_corridors = set()
    for _path, source in observation_sources:
        for key in observations:
            observations[key].extend(source.get(key, []))
        removed_corridors.update(source.get("remove_corridor_ids", []))
    observations["corridors"] = [
        item for item in observations["corridors"] if item["id"] not in removed_corridors
    ]
    output.mkdir(parents=True)
    # Read only the known payload, never archive-relative extraction paths.
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            if not name.startswith("normalized/map/") or name.endswith("/"):
                continue
            relative = Path(name.removeprefix("normalized/map/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source payload path")
            if relative.parts[0] == "qualification" or relative.name == "manifest.json":
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    semantic_path = output / "gazebo/semantic.json"
    semantic, graph = read(semantic_path), read(output / "navigation-graph.json")
    image_path = output / "gazebo/materials/textures/kumpula-orthophoto.jpg"
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    for path, source in observation_sources:
        if source.get("image_sha256", source.get("source_image_sha256", image_hash)) != image_hash:
            raise ValueError(f"Observation image mismatch: {path}")
    with Image.open(image_path) as image:
        width, height = image.size
    semantic["live_map"] = {
        "schema_version": "dronedream.live-map-georeference.v1",
        "image_path": "normalized/map/gazebo/materials/textures/kumpula-orthophoto.jpg",
        "width_px": width,
        "height_px": height,
        "coordinate_frame": "ENU",
        "world_to_image": [width / 500, 0, width / 2, 0, -height / 500, height / 2],
    }
    if extent_source is not None:
        from ground_map_extent import expand_registered_ground

        expand_registered_ground(output / "gazebo", semantic, extent_source)
    if building_source is not None:
        from ground_map_extent_buildings import install_extent_buildings

        install_extent_buildings(
            output / "gazebo", semantic, building_source, semantic["extent_enu_m"]
        )
    provenance = {
        "source": "registered-orthophoto-and-osm-survey",
        "image_sha256": image_hash,
        "survey_sha256": hashlib.sha256(survey_path.read_bytes()).hexdigest(),
        "observations_sha256": hashlib.sha256(
            json.dumps(observations, sort_keys=True).encode()
        ).hexdigest(),
        "observation_sources": [
            {
                "filename": p.name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                **{
                    key: source[key]
                    for key in ("source_receipts", "attribution", "license", "coverage")
                    if key in source
                },
            }
            for p, source in observation_sources
        ],
    }
    widths = {
        "residential": 6.0,
        "service": 5.0,
        "unclassified": 5.0,
        "pedestrian": 3.5,
        "footway": 2.5,
        "path": 2.5,
        "cycleway": 3.0,
    }
    road_evidence = {
        key: provenance[key]
        for key in ("source", "image_sha256", "survey_sha256", "observations_sha256")
    }
    road_evidence["source_registry"] = "traversability.provenance"
    corridors = []
    for index, segment in enumerate(survey["selected_road_segments"]):
        if segment["highway"] not in widths:
            continue
        corridors.append(
            {
                "id": f"survey-road-{index:04d}",
                "kind": "road"
                if segment["highway"] in {"residential", "service", "unclassified"}
                else "path",
                "centerline_xy_m": [xy(segment["from"]), xy(segment["to"])],
                "width_m": widths[segment["highway"]],
                "ground_z_m": 0,
                "evidence": road_evidence,
            }
        )
    # Explicitly authored observations, not nearest-node synthetic connectors.
    corridors.extend(observations["corridors"])
    areas = observations["areas"]
    exclusions = list(observations["exclusions"])
    exclusions.extend(refine_building_collisions(output / "gazebo", semantic))
    layer = {
        "schema_version": "dronedream.traversability.v1",
        "required": True,
        "minimum_agl_m": 0,
        "maximum_agl_m": 12,
        "areas": areas,
        "corridors": corridors,
        "exclusions": exclusions,
        "provenance": provenance,
    }
    semantic["traversability"] = layer
    install_canopy_obstacles(output / "gazebo", semantic)
    include_physical_ground_support(output / "gazebo", semantic)
    semantic["fidelity"]["qualified_routes"] = 0
    semantic["fidelity"]["route_policy"] = "ground-traversability-no-roof-or-canopy-overflight"
    # Correct staging locations only with explicit visual observations. Preserve
    # identity/aliases; the evidence states this is not an inferred doorway.
    by_id = {node["node_id"]: node for node in graph["nodes"]}
    for correction in observations["handoff_corrections"]:
        node = by_id[correction["node_id"]]
        if any(
            abs(node["position_m"][axis] - correction["old_position_m"][axis]) > 0.001
            for axis in ("x", "y", "z")
        ):
            raise ValueError(f"Stale handoff correction: {node['node_id']}")
        node["position_m"] = correction["new_position_m"]
        for entity in semantic["entities"]:
            if entity.get("entity_id") == node["node_id"]:
                entity["location_evidence"] = correction["evidence"]
    # The source package can already contain revised graph handoff positions.
    # Synchronize those too, not only corrections supplied in this invocation.
    synchronize_semantic_node_positions(semantic, graph)
    semantic["traversability"]["handoff_corrections"] = observations["handoff_corrections"]
    semantic["traversability"]["unresolved_observations"] = observations["unresolved_observations"]
    write(semantic_path, semantic)

    # True intersections split lines. Old demonstrated air tracks are history,
    # not operational edges; retain the named launch/pickup identities only.
    lines = [LineString(item["centerline_xy_m"]) for item in corridors]
    permitted = unary_union(
        [line.buffer(item["width_m"] / 2) for line, item in zip(lines, corridors, strict=True)]
        + [Polygon(item["polygon_xy_m"]) for item in areas]
    )
    forbidden = unary_union([Polygon(item["polygon_xy_m"]) for item in exclusions])
    # Apply the X500 planning envelope+margin once, not twice at obstacles.
    usable = permitted.difference(forbidden).buffer(-0.85)
    noded = node_observed_corridor_joins(lines, usable)
    old_nodes = [
        node
        for node in graph["nodes"]
        if node["node_id"] not in {"verified-001", "verified-002", "verified-003"}
    ]
    nodes, edges, node_by_point = list(old_nodes), [], {}

    def add_node(point):
        key = tuple(round(float(v), 4) for v in point)
        if key not in node_by_point:
            node_id = f"ground-junction-{len(node_by_point):04d}"
            node_by_point[key] = node_id
            nodes.append(
                {
                    "node_id": node_id,
                    "label": "Ground route junction",
                    "position_m": {"x": key[0], "y": key[1], "z": 3.0},
                    "semantic": "corridor",
                }
            )
        return node_by_point[key]

    def add_edge(a, b, distance):
        edges.append(
            {
                "edge_id": f"ground-edge-{len(edges):04d}",
                "from_node": a,
                "to_node": b,
                "distance_m": distance,
                "minimum_clearance_m": 0.8,
                "speed_limit_mps": 2.0,
                "bidirectional": True,
                "qualification": "geometry-derived",
                "evidence_sha256": provenance["observations_sha256"],
            }
        )

    rejected = []
    for line in getattr(noded, "geoms", [noded]):
        for start, end in zip(line.coords, list(line.coords)[1:], strict=False):
            segment = LineString([start, end])
            if segment.length < 0.001:
                continue
            if not usable.covers(segment):
                rejected.append(list(segment.coords))
                continue
            add_edge(add_node(start), add_node(end), segment.length)
    # Attachment is allowed ONLY inside already observed traversable ground.
    # No new surface is created by projection or a successful geometric join.
    unattached = []
    for node in old_nodes:
        point = Point(xy(node["position_m"]))
        targets = sorted(node_by_point, key=lambda p: point.distance(Point(p)))
        attached = False
        for target in targets:
            if point.distance(Point(target)) > 35:
                break
            segment = LineString([point.coords[0], target])
            if usable.covers(segment) and segment.length > 0.001:
                add_edge(
                    node["node_id"],
                    node_by_point[target],
                    math.hypot(segment.length, 3 - node["position_m"]["z"]),
                )
                attached = True
                break
        if not attached:
            unattached.append({"node_id": node["node_id"], "position_m": node["position_m"]})
    # Bridge survey discretization gaps only where the entire swept footprint
    # already lies on observed ground. This never adds a traversable surface.
    pairs = {frozenset((e["from_node"], e["to_node"])) for e in edges}
    points = list(node_by_point)
    for point in points:
        neighbors = sorted(
            (other for other in points if other != point), key=lambda other: math.dist(point, other)
        )[:8]
        for target in neighbors:
            distance = math.dist(point, target)
            pair = frozenset((node_by_point[point], node_by_point[target]))
            if distance > 12 or pair in pairs or not usable.covers(LineString([point, target])):
                continue
            add_edge(node_by_point[point], node_by_point[target], distance)
            pairs.add(pair)
    graph["nodes"], graph["edges"] = nodes, edges
    graph["named_entities"] = {
        key: value
        for key, value in graph["named_entities"].items()
        if value not in {"verified-001", "verified-002", "verified-003"}
    }
    # The same body/corridor proof used in production is authoritative. A
    # polygon-union prefilter alone can overestimate clearance at a junction.
    from dronedream_agent_core.contracts import MapAsset
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner

    planner = KnownMapMetricPlanner(
        graph=MapAsset.model_validate(graph),
        semantic_path=semantic_path,
        vehicle_diameter_m=0.76,
        vehicle_height_m=0.6,
    )
    positions = {
        n["node_id"]: tuple(n["position_m"][axis] for axis in ("x", "y", "z")) for n in nodes
    }
    rejected_edges = []
    safe_edges = []
    for edge in edges:
        if planner._segment_is_free(positions[edge["from_node"]], positions[edge["to_node"]]):
            safe_edges.append(edge)
        else:
            rejected_edges.append(edge)
    graph["edges"] = safe_edges
    attached_ids = {e[key] for e in safe_edges for key in ("from_node", "to_node")}
    unattached = [
        {"node_id": n["node_id"], "position_m": n["position_m"]}
        for n in old_nodes
        if n["node_id"] not in attached_ids
    ]
    write(output / "navigation-graph.json", graph)
    report = {
        "status": "candidate-requires-qualification",
        "corridors": len(corridors),
        "areas": len(areas),
        "exclusions": len(exclusions),
        "nodes": len(nodes),
        "edges": len(safe_edges),
        "unattached": unattached,
        "rejected_clearance_edges": rejected_edges,
        "rejected_survey_segments": rejected,
        "provenance": provenance,
    }
    write(output / "ground-map-review.json", report)
    update_candidate_summary(output / "gazebo", semantic)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"rejected_survey_segments", "rejected_clearance_edges", "provenance"}
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "survey", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--observations", type=Path, action="append", required=True)
    parser.add_argument("--extent-source", type=Path)
    parser.add_argument("--building-source", type=Path)
    args = parser.parse_args()
    prepare(
        args.package, args.survey, args.observations, args.output,
        args.extent_source, args.building_source,
    )
