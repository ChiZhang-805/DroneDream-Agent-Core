"""Offline building/entrance grounding and 3-D reachability for map handoffs.

Generated outdoor simulation positions are not surveyed doors or landing sites.
No geospatial dependency or repeated map audit is added to the flight loop.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from shapely import affinity
from shapely.geometry import LineString, Point, box, shape
from shapely.ops import nearest_points, unary_union

from dronedream_agent_core.collision import PREFERRED_TRANSIT_CLEARANCE_M, validate_route_clearance
from dronedream_agent_core.contracts import MapAsset, RouteQuery, VehicleAsset
from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy


def collision_footprints(semantic):
    groups = defaultdict(list)
    for primitive in semantic["collision_primitives"]:
        if primitive["semantic"] != "building":
            continue
        rectangle = box(
            -primitive["size_x"] / 2,
            -primitive["size_y"] / 2,
            primitive["size_x"] / 2,
            primitive["size_y"] / 2,
        )
        polygon = affinity.translate(
            affinity.rotate(rectangle, primitive["yaw_rad"], use_radians=True),
            primitive["center_x"],
            primitive["center_y"],
        )
        groups["city-" + primitive["name"].split("-")[1]].append(polygon)
    return {key: unary_union(value) for key, value in groups.items()}


def ground_relation(point, footprint, owner, other_buildings, entrances):
    """Measure actual facade proximity and unobstructed door-to-handoff segments."""
    facade = nearest_points(footprint, point)[0]
    facade_link = LineString([facade, point])
    if facade_link.intersection(other_buildings).length > 0.05:
        return None
    connected = []
    for entrance in entrances:
        door = Point(entrance["position_xy_m"])
        link = LineString([door, point])
        # City/OSM footprints differ by sub-metre rounding at documented doors.
        if (
            link.intersection(other_buildings).length <= 0.05
            and link.intersection(owner).length <= 1.0
        ):
            connected.append((door.distance(point), entrance["entrance_id"]))
    distance, entrance_id = min(connected) if connected else (None, None)
    return {
        "building_distance_m": round(footprint.distance(point), 3),
        "entrance_id": entrance_id,
        "entrance_distance_m": round(distance, 3) if distance is not None else None,
        "entrance_connection_verified": entrance_id is not None,
        "connection_basis": "source-door-unobstructed-in-map"
        if entrance_id
        else "exterior-facade-only",
    }


def choose_location(
    footprint,
    owner,
    other_buildings,
    collision,
    occupied,
    extent,
    entrances,
    preferred=None,
    scene_boundary_points=None,
):
    boundary = collision.buffer(2.6, join_style="mitre").boundary
    points = [boundary.interpolate(float(i)) for i in range(math.ceil(boundary.length))]
    if scene_boundary_points is not None:
        # Adjacent collision boxes can cover every sampled point on one box's
        # perimeter. Their union has the actual exterior boundary of the scene.
        points.extend(scene_boundary_points)
    if preferred is not None:
        points.insert(0, preferred)
    choices = []
    for point in points:
        if not extent.covers(point) or occupied.distance(point) < 2.499:
            continue
        relation = ground_relation(point, footprint, owner, other_buildings, entrances)
        if relation is None:
            continue
        # A shared courtyard can serve a documented door even when another facade
        # is closer. Without a door connection, never assign a neighbour's apron.
        if (
            not relation["entrance_connection_verified"]
            and footprint.distance(point) > other_buildings.distance(point) + 0.05
        ):
            continue
        if (
            preferred is not None
            and point.equals(preferred)
            and relation["entrance_connection_verified"]
        ):
            return point, relation
        rank = (
            not relation["entrance_connection_verified"],
            relation["entrance_distance_m"]
            if relation["entrance_connection_verified"]
            else relation["building_distance_m"],
        )
        choices.append((rank, point, relation))
    if not choices:
        return None
    _, point, relation = min(choices, key=lambda row: row[0])
    return point, relation


def build_handoffs(index, semantic, graph, semantic_path, vehicle):
    buildings = {item["building_id"]: item for item in index["buildings"]}
    footprints = {key: shape(value["footprint"]) for key, value in buildings.items()}
    collisions = collision_footprints(semantic)
    occupied = unary_union(list(collisions.values()))
    boundary = occupied.buffer(2.6, join_style="mitre").boundary
    scene_boundary_points = [
        boundary.interpolate(float(i)) for i in range(math.ceil(boundary.length))
    ]
    extent = box(
        -semantic["bounds_m"]["x"] / 2 + 2.5,
        -semantic["bounds_m"]["y"] / 2 + 2.5,
        semantic["bounds_m"]["x"] / 2 - 2.5,
        semantic["bounds_m"]["y"] / 2 - 2.5,
    )
    old_nodes = {node["node_id"]: node for node in graph["nodes"]}
    entries = [*index["facilities"], *index["buildings"]]
    locations = []
    nodes = dict(old_nodes)
    alias_targets = defaultdict(set)
    facility_aliases = set()
    facility_locations = {}
    for entry in entries:
        bid = entry["building_id"]
        identity = entry.get("facility_id", bid)
        is_facility = "facility_id" in entry
        primary = next(
            (
                item
                for item in index["facilities"]
                if item["building_id"] == bid and item["name"] == entry["name"]
            ),
            None,
        )
        if not is_facility and primary and primary["facility_id"] in facility_locations:
            row = dict(facility_locations[primary["facility_id"]], place_id=bid)
            locations.append(row)
        else:
            footprint = shape(entry["footprint"])
            other = unary_union([polygon for key, polygon in footprints.items() if key != bid])
            doors = [
                door
                for door in index["entrances"]
                if door["entrance_id"] in entry["entrance_ids"]
                and door["inside_scene_bounds"]
                and door["kind"] != "exit"
                and door["access"] not in {"no", "employees"}
            ]
            # Existing named targets retain stable IDs; their positions are kept
            # only when their relationship to their own entrance is valid.
            node_id = graph["named_entities"].get(entry["name"]) if is_facility else None
            node_id = node_id or identity + "-handoff"
            old = old_nodes.get(node_id)
            preferred = Point(old["position_m"]["x"], old["position_m"]["y"]) if old else None
            selection = choose_location(
                footprint,
                footprints[bid],
                other,
                collisions[bid],
                occupied,
                extent,
                doors,
                preferred,
                scene_boundary_points,
            )
            row = {
                "place_id": identity,
                "building_id": bid,
                "name": entry["name"],
                "source_refs": entry["source_refs"],
                "flight_verified": False,
            }
            if selection is None:
                row.update(
                    status="unavailable",
                    flight_target_available=False,
                    reason="no-exterior-handoff-within-current-collision-geometry",
                )
            else:
                point, relation = selection
                position = {"x": round(point.x, 3), "y": round(point.y, 3), "z": 2.0}
                node = {
                    "node_id": node_id,
                    "label": entry["name"] + " outdoor handoff",
                    "position_m": position,
                    "semantic": "delivery",
                }
                if old and position == old["position_m"]:
                    node = old
                nodes[node_id] = node
                row.update(
                    relation,
                    node_id=node_id,
                    position_m=position,
                    horizontal_building_clearance_m=round(
                        occupied.distance(Point(position["x"], position["y"])), 3
                    ),
                    status="geometry-checked",
                    flight_target_available=True,
                )
            locations.append(row)
            if is_facility:
                facility_locations[identity] = row
        row = locations[-1]
        if not row["flight_target_available"]:
            continue
        aliases = [identity, entry["name"], *entry["aliases"]]
        if is_facility:
            facility_aliases.update(aliases)
        for alias in aliases:
            if not is_facility and alias in facility_aliases:
                continue
            alias_targets[alias].add(row["node_id"])

    graph["nodes"] = list(nodes.values())
    # Ambiguous generic pickup labels must not silently select Chemicum.
    for alias in ("handoff point", "pickup point", "交接点", "取件点"):
        graph["named_entities"].pop(alias, None)
    graph["named_entities"].update(
        {
            alias: next(iter(targets))
            for alias, targets in alias_targets.items()
            if len(targets) == 1
        }
    )
    # A shared address is retained in place facts for clarification, not assigned
    # to whichever building happened to appear last during generation.
    for alias, targets in alias_targets.items():
        if len(targets) > 1:
            graph["named_entities"].pop(alias, None)

    planner = KnownMapMetricPlanner(
        graph=MapAsset.model_validate(graph),
        semantic_path=semantic_path,
        vehicle_diameter_m=vehicle.body_radius_m * 2,
        vehicle_height_m=vehicle.body_height_m,
        policy=MetricPlannerPolicy(required_clearance_m=PREFERRED_TRANSIT_CLEARANCE_M),
    )
    route_results = {}
    for row in locations:
        if not row["flight_target_available"]:
            continue
        node_id = row["node_id"]
        if node_id not in route_results:
            route = planner.plan(RouteQuery(start_node="verified-000", goal_node=node_id))
            check = validate_route_clearance(
                route,
                semantic_path,
                vehicle_diameter_m=vehicle.body_radius_m * 2,
                vehicle_height_m=vehicle.body_height_m,
            )
            if not check.accepted:
                raise ValueError(f"Handoff route intersects scene: {node_id}")
            route_results[node_id] = {
                "length_m": route.route_length_m,
                "clearance": check.model_dump(mode="json"),
            }
        row["route_check"] = route_results[node_id]
        row["status"] = "simulation-route-checked"
    index["handoff_locations"] = locations
    index["coverage"]["buildings_with_executable_handoff"] = sum(
        row["flight_target_available"] for row in locations if row["place_id"] in buildings
    )
    index["coverage"]["buildings_without_executable_handoff"] = [
        row["place_id"]
        for row in locations
        if row["place_id"] in buildings and not row["flight_target_available"]
    ]
    return index, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "semantic", "graph", "vehicle", "output_index", "output_graph"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    index, graph = build_handoffs(
        json.loads(args.index.read_bytes()),
        json.loads(args.semantic.read_bytes()),
        json.loads(args.graph.read_bytes()),
        args.semantic,
        VehicleAsset.model_validate_json(args.vehicle.read_bytes()),
    )
    for path, value in ((args.output_index, index), (args.output_graph, graph)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(index["coverage"]))


if __name__ == "__main__":
    main()
