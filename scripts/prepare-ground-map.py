"""Prepare a reviewable ground-constrained map; never manufacture flight evidence.

The source road survey is image-registered, not a flight track. Every new edge is
geometry-derived. This output is a candidate which must pass pair qualification
before the release builder can promote it. Requires the map-preparation extra.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from shapely import affinity, constrained_delaunay_triangles
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


def route_surface(permitted, forbidden, vehicle_radius_m=0.38, clearance_m=0.8):
    """Leave the aircraft envelope, continuous proof margin and rounding room."""
    inset = vehicle_radius_m + clearance_m + 0.05 + 0.02
    return permitted.buffer(-inset).difference(forbidden.buffer(inset, join_style="mitre"))


def operational_nodes(graph):
    """Retain named task locations, never junctions from a previous generation."""
    named = set(graph.get("named_entities", {}).values())
    retired = {"verified-001", "verified-002", "verified-003"}
    return [node for node in graph["nodes"]
            if node["node_id"] not in retired and
            (node["node_id"] in named or node.get("semantic") not in {"corridor", "junction"})]


def append_planar_landmarks(semantic, observation_sources):
    """Preserve sourced XY landmarks without manufacturing flight altitudes."""
    entities = {entity["entity_id"]: entity for entity in semantic["entities"]}
    anchors = semantic.setdefault("roads", {}).setdefault("facility_anchors", {})
    for _path, source in observation_sources:
        for entity in source.get("planar_landmarks", []):
            identifier = entity["entity_id"]
            position = entity["position_m"]
            if (entity.get("semantic") != "facility-anchor" or not isinstance(position, list)
                    or len(position) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in position)):
                raise ValueError("Planar landmark must retain its two measured coordinates")
            entities[identifier] = {**entity, "source_pointer": f"/roads/facility_anchors/{identifier}"}
            anchors[identifier] = position
        if source.get("service_entrances_source"):
            semantic.setdefault("sources", {})["service_entrances"] = source["service_entrances_source"]
    semantic["entities"] = list(entities.values())


def consolidate_ground_areas(layer):
    """Represent the union of touching surveyed surfaces at each elevation.

    Per-feature inset makes an artificial barrier at a road/apron boundary.
    Union before inset preserves the actual permitted footprint, including holes.
    Different elevations remain separate; their 2-D overlap is not a junction.
    """
    groups = {}
    for area in layer.get("areas", []):
        groups.setdefault(area["ground_z_m"], []).append(
            Polygon(area["polygon_xy_m"], area.get("holes_xy_m", [])))
    for corridor in layer.get("corridors", []):
        groups.setdefault(corridor["ground_z_m"], []).append(
            LineString(corridor["centerline_xy_m"]).buffer(corridor["width_m"]/2))
    areas = []
    for elevation, shapes in sorted(groups.items()):
        union = unary_union(shapes)
        polygons = [union] if isinstance(union, Polygon) else union.geoms
        for polygon in polygons:
            areas.append({"id": f"observed-ground-{len(areas):04d}", "kind": "open-space",
                          "ground_z_m": elevation,
                          "polygon_xy_m": list(polygon.exterior.coords)[:-1],
                          "holes_xy_m": [list(ring.coords)[:-1] for ring in polygon.interiors],
                          "evidence": {"source_registry": "traversability.provenance",
                                       "derivation": "union-of-same-elevation-observed-ground"}})
    layer["areas"], layer["corridors"] = areas, []


def surface_detour(surface, start, goal):
    """Find a continuous path inside observed free ground, including narrow bends.

    A fixed sampling grid can miss a real connected corridor. Triangles share
    exact edges, so their interior connections preserve the source topology.
    The caller must still prove the resulting 3-D route with the runtime planner.
    """
    parts = [surface] if isinstance(surface, Polygon) else surface.geoms
    polygon = next((p for p in parts if isinstance(p, Polygon)
                    and p.covers(Point(start)) and p.covers(Point(goal))), None)
    if polygon is None:
        return None
    if polygon.covers(LineString([start, goal])):
        return [start, goal]
    triangles = list(constrained_delaunay_triangles(polygon).geoms)
    shared, adjacency = {}, {}
    def link(a, b):
        cost = math.dist(a, b)
        adjacency.setdefault(a, []).append((b, cost))
        adjacency.setdefault(b, []).append((a, cost))
    for triangle in triangles:
        center = tuple(triangle.centroid.coords[0])
        for endpoint in (start, goal):
            if triangle.covers(Point(endpoint)):
                link(endpoint, center)
        ring = list(triangle.exterior.coords)
        for a, b in zip(ring, ring[1:]):
            edge = tuple(sorted((a, b)))
            if edge in shared:
                midpoint = ((a[0]+b[0])/2, (a[1]+b[1])/2)
                link(center, midpoint)
                link(shared[edge], midpoint)
            else:
                shared[edge] = center
    costs, previous, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            reduced, index = [start], 0
            while index < len(path)-1:
                end = len(path)-1
                while end > index+1 and not polygon.covers(LineString([path[index], path[end]])):
                    end -= 1
                reduced.append(path[end])
                index = end
            return reduced
        for neighbor, distance in adjacency.get(node, []):
            new_cost = cost + distance
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor], previous[neighbor] = new_cost, node
                heapq.heappush(queue, (new_cost, neighbor))
    return None


def connect_task_locations(graph, semantic_path, evidence_sha256, *, targets=None):
    """Recover missing topology using the actual ground-constrained planner.

    A rejected straight connector need not make a place unreachable. Search
    observed corridors for a detour, preserving the surveyed handoff position.
    Unresolved locations retain their identity and a concrete source issue.
    """
    from dronedream_agent_core.contracts import MapAsset, RouteQuery, Vector3
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy

    locations = operational_nodes(graph)
    launches = [n for n in locations if n.get("semantic") == "launch"]
    if not launches:
        raise ValueError("Map has no surveyed launch location")
    launch = launches[0]["node_id"]
    planner = KnownMapMetricPlanner(graph=MapAsset.model_validate(graph),
        semantic_path=semantic_path, vehicle_diameter_m=.76, vehicle_height_m=.6,
        policy=MetricPlannerPolicy(required_clearance_m=.8))
    semantic = read(semantic_path)
    layer = semantic["traversability"]
    permitted = unary_union([
        *[Polygon(a["polygon_xy_m"], a.get("holes_xy_m", [])) for a in layer["areas"]],
        *[LineString(c["centerline_xy_m"]).buffer(c["width_m"]/2) for c in layer["corridors"]],
    ])
    forbidden = unary_union([Polygon(e["polygon_xy_m"], e.get("holes_xy_m", []))
                             for e in layer["exclusions"]])
    # Use the planner's actual proof margin; extra arbitrary inflation can sever
    # valid narrow passages. Millimetre rounding room is checked again in 3-D.
    inset = planner.vehicle_radius_m + planner._required_clearance_m + .002
    surface = permitted.buffer(-inset).difference(forbidden.buffer(inset))
    # A disconnected horizontal operating volume cannot be joined by changing
    # altitude: these exclusions apply at every height. Use a slightly enlarged
    # free domain here so buffer approximation cannot reject a marginal passage;
    # actual route acceptance still uses the unchanged 3-D clearance proof.
    optimistic_inset = max(0., planner.vehicle_radius_m + planner._required_clearance_m - .001)
    optimistic = permitted.buffer(-optimistic_inset, quad_segs=64).difference(
        forbidden.buffer(optimistic_inset, quad_segs=64))
    optimistic_parts = [optimistic] if isinstance(optimistic, Polygon) else optimistic.geoms
    start_xy = Point(xy(launches[0]["position_m"]))
    launch_parts = [part for part in optimistic_parts if part.covers(start_xy)]
    by_point = {tuple(n["position_m"][axis] for axis in ("x", "y", "z")): n["node_id"]
                for n in graph["nodes"]}
    adjacency = {}
    pairs = set()
    def remember(edge):
        a, b = edge["from_node"], edge["to_node"]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        pairs.add(frozenset((a, b)))
    for edge in graph["edges"]:
        remember(edge)
    unresolved = []
    for location in locations:
        if targets is not None and location["node_id"] not in targets:
            continue
        seen, pending = set(), [launch]
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(adjacency.get(node, set()) - seen)
        target = location["node_id"]
        if target in seen:
            continue
        print(f"Checking ground connection: {target}", flush=True)
        begin, finish = launches[0]["position_m"], location["position_m"]
        if not any(part.covers(Point(xy(finish))) for part in launch_parts):
            unresolved.append({"node_id": target, "position_m": finish,
                "reason": "observed-ground-disconnected-at-vehicle-envelope-and-clearance"})
            continue
        planar = surface_detour(surface, (begin["x"], begin["y"]), (finish["x"], finish["y"]))
        positions = []
        if planar:
            height = max(3., begin["z"], finish["z"])
            positions = [Vector3(**begin), *[Vector3(x=x, y=y, z=height) for x, y in planar], Vector3(**finish)]
            points = [tuple(getattr(p, axis) for axis in ("x", "y", "z")) for p in positions]
            if not all(planner._segment_is_free(a, b) for a, b in zip(points, points[1:])):
                positions = []
        if not positions:
            try:
                positions = planner.plan(RouteQuery(start_node=launch, goal_node=target)).positions_m
            except ValueError as error:
                unresolved.append({"node_id": target, "position_m": location["position_m"],
                                   "reason": str(error)})
                continue
        ids = [launch]
        for position in positions[1:-1]:
            point = (position.x, position.y, position.z)
            if point not in by_point:
                identifier = f"ground-detour-{len(graph['nodes']):04d}"
                by_point[point] = identifier
                graph["nodes"].append({"node_id": identifier, "label": "Observed ground detour",
                    "position_m": position.model_dump(), "semantic": "corridor"})
            ids.append(by_point[point])
        ids.append(target)
        for index, (a, b) in enumerate(zip(ids, ids[1:])):
            if a == b or frozenset((a, b)) in pairs:
                continue
            distance = math.dist(tuple(getattr(positions[index], axis) for axis in ("x", "y", "z")),
                                 tuple(getattr(positions[index+1], axis) for axis in ("x", "y", "z")))
            edge = {"edge_id": f"ground-detour-edge-{len(graph['edges']):04d}",
                    "from_node": a, "to_node": b, "distance_m": distance,
                    "minimum_clearance_m": .8, "speed_limit_mps": 6., "bidirectional": True,
                    "qualification": "geometry-derived", "evidence_sha256": evidence_sha256}
            graph["edges"].append(edge)
            remember(edge)
    return unresolved


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


def complete_english_place_catalog(semantic, graph):
    """Translate known place functions without inferring uses from requests.

    The left-hand names already identify these facilities in the source map.
    Their translations and common short forms bind to that same operational
    node, while building centers and outdoor handoff positions stay distinct.
    """
    translations = {
        "化学楼": ("chemistry building", "chemistry department", "department of chemistry"),
        "物理楼": ("physics building", "physical sciences building"),
        "地球科学楼": ("earth sciences building",),
        "气象研究所": ("weather institute", "meteorological institute", "meteorology building"),
        "信息科学楼": ("information science building", "information sciences building"),
        "计算机科学楼": ("computer science building",),
        "数学楼": ("mathematics building",),
        "数学与统计楼": ("mathematics and statistics building",),
        "Kumpula Campus Library": ("campus library", "Kumpula library"),
        "Kumpula Sportcenter": (
            "sports centre", "sports center", "Kumpula sports centre", "Kumpula sports center",
        ),
        "HOAS Kumpula / Pietari Kalmin katu": ("HOAS Kumpula",),
        "Accelerator Laboratory": ("accelerator lab",),
        "Helsinki Upper Secondary School of Natural Sciences": (
            "science high school", "natural sciences high school",
        ),
        "Daycare Kumpupilvi": ("Kumpupilvi daycare",),
    }
    label_translations = {
        "Kumpulan liikuntakeskus": "Kumpula Sports Centre",
        "Kiihdytinlaboratorio": "Accelerator Laboratory",
        "Kumpupilvi 幼儿园": "Daycare Kumpupilvi",
        "Kumpulan kampuskirjasto": "Kumpula Campus Library",
        "自然科学高中": "Natural Sciences High School",
        "仓储附属建筑": "Storage annex",
        "附属建筑": "Annex",
    }
    named = graph["named_entities"]
    original_names = dict(named)
    additions = {}
    for source, aliases in translations.items():
        if source in original_names:
            additions.update((alias, original_names[source]) for alias in aliases)
    for source, node_id in original_names.items():
        for original, english in label_translations.items():
            if source == original or source.startswith(original + " "):
                additions[english + source[len(original):]] = node_id
                break
        ascii_accents = "".join(
            char for char in unicodedata.normalize("NFKD", source)
            if not unicodedata.combining(char)
        )
        if ascii_accents != source:
            additions[ascii_accents] = node_id
    folded_names = {name.casefold(): node_id for name, node_id in named.items()}
    positions = {node["node_id"]: node["position_m"] for node in graph["nodes"]}
    accepted_additions = {}
    for alias, node_id in additions.items():
        existing = folded_names.get(alias.casefold())
        if existing is not None and existing != node_id:
            # City and OSM records of the same school share one observed
            # handoff. Retain its first canonical English spelling instead of
            # adding a case-only alias that resolves to the duplicate record.
            if positions[existing] == positions[node_id]:
                continue
            raise ValueError(f"Conflicting source place alias: {alias}")
        named[alias] = node_id
        folded_names[alias.casefold()] = node_id
        accepted_additions[alias] = node_id
    for entity in semantic["entities"]:
        source_names = [entity["entity_id"], entity["name"], *entity.get("aliases", [])]
        bindings = {original_names[name] for name in source_names if name in original_names}
        if len(bindings) == 1:
            node_id = next(iter(bindings))
            entity["aliases"] = list(dict.fromkeys([
                *entity.get("aliases", []),
                *(name for name, target in accepted_additions.items() if target == node_id),
            ]))
    for node in graph["nodes"]:
        for original, english in label_translations.items():
            if node["label"].startswith(original + " "):
                node["label"] = english + node["label"][len(original):]
                break


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
    footprint is removed. Planning keeps the boxes; physics uses their exact
    closed union per building to avoid thousands of independent collision shapes.
    """
    from ground_map_collision_mesh import building_collision_mesh
    from ground_map_extent_buildings import _base_buildings

    original = _base_buildings(gazebo / "meshes/kumpula-buildings.obj", semantic)
    refined, exact, meshes = [], [], []
    for name, building in original.items():
        polygon, primitive = building["polygon"], building["primitive"]
        ring = list(polygon.exterior.coords)[:-1]
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
        mesh_uri = f"meshes/kumpula-building-collision-{len(meshes):03d}.obj"
        (gazebo / mesh_uri).write_bytes(
            building_collision_mesh(cells, height_m=primitive["size_z"])
        )
        meshes.append((name, mesh_uri, primitive["center_z"], angle))
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
        exact.append({"id": name, "kind": "building", "polygon_xy_m": ring,
                      "holes_xy_m": [list(r.coords)[:-1] for r in polygon.interiors]})
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
        for name, mesh_uri, center_z, angle in meshes:
            col = ET.SubElement(link, "collision", {"name": f"{name}-cell-union"})
            ET.SubElement(col, "pose").text = (
                f"0 0 {center_z:.17g} 0 0 {angle:.17g}"
            )
            geom = ET.SubElement(col, "geometry")
            shape = ET.SubElement(geom, "mesh")
            ET.SubElement(shape, "uri").text = mesh_uri
            ET.SubElement(shape, "scale").text = "1 1 1"
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
    semantic.setdefault("live_map", {
        "schema_version": "dronedream.live-map-georeference.v1",
        "image_path": "normalized/map/gazebo/materials/textures/kumpula-orthophoto.jpg",
        "width_px": width,
        "height_px": height,
        "coordinate_frame": "ENU",
        "world_to_image": [width / 500, 0, width / 2, 0, -height / 500, height / 2],
    })
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
    consolidate_ground_areas(layer)
    install_canopy_obstacles(output / "gazebo", semantic)
    include_physical_ground_support(output / "gazebo", semantic)
    semantic["fidelity"]["qualified_routes"] = 0
    semantic["fidelity"]["route_policy"] = "ground-traversability-no-roof-or-canopy-overflight"
    # Correct staging locations only with explicit visual observations. Preserve
    # identity/aliases; the evidence states this is not an inferred doorway.
    by_id = {node["node_id"]: node for node in graph["nodes"]}
    latest_corrections = {item["node_id"]: item for item in observations["handoff_corrections"]}
    current_corrections = {
        node_id for node_id, item in latest_corrections.items()
        if all(abs(by_id[node_id]["position_m"][axis] - item["new_position_m"][axis]) <= 0.001
               for axis in ("x", "y", "z"))
    }
    for correction in observations["handoff_corrections"]:
        node = by_id[correction["node_id"]]
        if node["node_id"] in current_corrections:
            continue
        already_applied = all(
            abs(node["position_m"][axis] - correction["new_position_m"][axis]) <= 0.001
            for axis in ("x", "y", "z")
        )
        if not already_applied and any(
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
    complete_english_place_catalog(semantic, graph)
    append_planar_landmarks(semantic, observation_sources)
    semantic["traversability"]["handoff_corrections"] = observations["handoff_corrections"]
    semantic["traversability"]["unresolved_observations"] = observations["unresolved_observations"]
    write(semantic_path, semantic)

    # True intersections split lines. Old demonstrated air tracks are history,
    # not operational edges; retain the named launch/pickup identities only.
    lines = [LineString(item["centerline_xy_m"]) for item in corridors]
    permitted = unary_union(
        [line.buffer(item["width_m"] / 2) for line, item in zip(lines, corridors, strict=True)]
        + [Polygon(item["polygon_xy_m"], item.get("holes_xy_m", [])) for item in areas]
    )
    forbidden = unary_union([Polygon(item["polygon_xy_m"], item.get("holes_xy_m", []))
                             for item in exclusions])
    usable = route_surface(permitted, forbidden)
    noded = node_observed_corridor_joins(lines, usable)
    old_nodes = operational_nodes(graph)
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
                "speed_limit_mps": 6.0,
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
    from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy

    planner = KnownMapMetricPlanner(
        graph=MapAsset.model_validate(graph),
        semantic_path=semantic_path,
        vehicle_diameter_m=0.76,
        vehicle_height_m=0.6,
        policy=MetricPlannerPolicy(required_clearance_m=0.8),
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
    unattached = connect_task_locations(graph, semantic_path, provenance["observations_sha256"])
    write(output / "navigation-graph.json", graph)
    report = {
        "status": "candidate-requires-qualification",
        "corridors": len(layer["corridors"]),
        "areas": len(layer["areas"]),
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
