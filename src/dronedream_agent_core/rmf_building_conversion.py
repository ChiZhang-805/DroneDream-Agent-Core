"""Bounded declarative Open-RMF building-map to Gazebo/semantic conversion.

The converter intentionally does not execute Traffic Editor, ROS, model plugins,
or arbitrary source code.  It preserves RMF topology as planning hints while
building conservative static floor/wall collision geometry for Gazebo.  The
result is still unqualified until a concrete map/aircraft pair passes runtime
acceptance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import median
from typing import Any
from xml.etree import ElementTree

import yaml

MAX_RMF_YAML_BYTES = 32 * 1024 * 1024
MAX_RMF_NODES = 500_000
MAX_RMF_LEVELS = 64
MAX_RMF_VERTICES = 100_000
MAX_RMF_SEGMENTS = 200_000


class RmfBuildingConversionError(ValueError):
    """The source does not satisfy the bounded declarative RMF contract."""


@dataclass(frozen=True)
class RmfBuildingConversion:
    name: str
    sdf: bytes
    semantic: bytes
    topology: bytes
    report: bytes


class _BoundedSafeLoader(yaml.SafeLoader):
    node_count = 0
    alias_count = 0

    def compose_node(self, parent, index):  # noqa: ANN001
        self.node_count += 1
        if self.node_count > MAX_RMF_NODES:
            raise RmfBuildingConversionError("RMF_BUILDING_NODE_LIMIT_EXCEEDED")
        if self.check_event(yaml.AliasEvent):
            self.alias_count += 1
            if self.alias_count > 256:
                raise RmfBuildingConversionError("RMF_BUILDING_ALIAS_LIMIT_EXCEEDED")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):  # noqa: ANN001
        keys = []
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=False)
            if key in keys:
                raise RmfBuildingConversionError("RMF_BUILDING_DUPLICATE_KEY")
            keys.append(key)
        return super().construct_mapping(node, deep=deep)


def _finite(value: object, code: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RmfBuildingConversionError(code)
    return float(value)


def _parameter(value: object, key: str) -> object | None:
    if not isinstance(value, dict) or key not in value:
        return None
    item = value[key]
    if isinstance(item, list) and len(item) == 2:
        return item[1]
    return item


def _slug(value: str) -> str:
    text = "".join(char.casefold() if char.isalnum() else "-" for char in value)
    return "-".join(filter(None, text.split("-")))[:120] or "rmf-building"


def _parse(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_RMF_YAML_BYTES:
        raise RmfBuildingConversionError("RMF_BUILDING_SIZE_INVALID")
    try:
        value = yaml.load(payload.decode("utf-8-sig"), Loader=_BoundedSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RmfBuildingConversionError("RMF_BUILDING_YAML_INVALID") from error
    if not isinstance(value, dict):
        raise RmfBuildingConversionError("RMF_BUILDING_OBJECT_REQUIRED")
    levels = value.get("levels")
    if not isinstance(levels, dict) or not 1 <= len(levels) <= MAX_RMF_LEVELS:
        raise RmfBuildingConversionError("RMF_BUILDING_LEVELS_INVALID")
    return value


def _level_scale(level: dict[str, Any], vertices: list[list[Any]], system: str) -> float:
    if system == "cartesian_meters":
        return 1.0
    estimates: list[float] = []
    for row in level.get("measurements", []) or []:
        if not isinstance(row, list) or len(row) < 3:
            continue
        try:
            first, second = int(row[0]), int(row[1])
            distance = _finite(_parameter(row[2], "distance"), "RMF_MEASUREMENT_INVALID")
            a, b = vertices[first], vertices[second]
            pixels = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
            if pixels > 1e-9 and distance > 0:
                estimates.append(distance / pixels)
        except (IndexError, TypeError, ValueError, RmfBuildingConversionError):
            continue
    if system == "reference_image" or estimates:
        if not estimates:
            raise RmfBuildingConversionError("RMF_BUILDING_SCALE_MISSING")
        scale = median(estimates)
        if not 1e-6 <= scale <= 1000:
            raise RmfBuildingConversionError("RMF_BUILDING_SCALE_INVALID")
        return scale
    return 1.0


def _metric_vertices(level: dict[str, Any], system: str) -> tuple[list[tuple[float, float]], float]:
    raw = level.get("vertices")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_RMF_VERTICES:
        raise RmfBuildingConversionError("RMF_BUILDING_VERTICES_INVALID")
    vertices: list[list[Any]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 2:
            raise RmfBuildingConversionError("RMF_BUILDING_VERTEX_INVALID")
        _finite(row[0], "RMF_BUILDING_VERTEX_INVALID")
        _finite(row[1], "RMF_BUILDING_VERTEX_INVALID")
        vertices.append(row)
    if system == "wgs84":
        lon0 = sum(float(row[0]) for row in vertices) / len(vertices)
        lat0 = sum(float(row[1]) for row in vertices) / len(vertices)
        latitude_scale = 111_320.0
        longitude_scale = latitude_scale * math.cos(math.radians(lat0))
        return [
            ((float(row[0]) - lon0) * longitude_scale, (float(row[1]) - lat0) * latitude_scale)
            for row in vertices
        ], 1.0
    scale = _level_scale(level, vertices, system)
    origin_x = min(float(row[0]) for row in vertices)
    origin_y = max(float(row[1]) for row in vertices)
    return [
        ((float(row[0]) - origin_x) * scale, (origin_y - float(row[1])) * scale) for row in vertices
    ], scale


def _add_box(
    model: ElementTree.Element,
    *,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float = 0.0,
) -> None:
    link = ElementTree.SubElement(model, "link", {"name": name})
    ElementTree.SubElement(
        link, "pose"
    ).text = f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 0 0 {yaw:.9f}"
    for role in ("collision", "visual"):
        node = ElementTree.SubElement(link, role, {"name": role})
        geometry = ElementTree.SubElement(node, "geometry")
        box = ElementTree.SubElement(geometry, "box")
        ElementTree.SubElement(box, "size").text = " ".join(f"{v:.6f}" for v in size)
        if role == "visual":
            material = ElementTree.SubElement(node, "material")
            ElementTree.SubElement(material, "diffuse").text = "0.72 0.75 0.80 1"


def convert_rmf_building_map(payload: bytes, *, source_name: str) -> RmfBuildingConversion:
    """Convert one declarative RMF building map without executing external tools."""
    source = _parse(payload)
    system = str(source.get("coordinate_system") or "reference_image")
    if system not in {"reference_image", "cartesian_meters", "wgs84"}:
        raise RmfBuildingConversionError("RMF_BUILDING_COORDINATE_SYSTEM_UNSUPPORTED")
    file_name = source_name.rsplit("/", 1)[-1].split(".building", 1)[0]
    declared_name = str(source.get("name") or "").strip()
    # Several upstream examples use the non-identity placeholder "building".
    # Preserve a stable per-resource identity from the compound source name.
    name = file_name if declared_name.casefold() in {"", "building"} else declared_name
    world_name = _slug(name)
    sdf = ElementTree.Element("sdf", {"version": "1.10"})
    world = ElementTree.SubElement(sdf, "world", {"name": world_name})
    ElementTree.SubElement(world, "gravity").text = "0 0 -9.81"
    ElementTree.SubElement(world, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
    ElementTree.SubElement(world, "atmosphere", {"type": "adiabatic"})
    spherical = ElementTree.SubElement(world, "spherical_coordinates")
    ElementTree.SubElement(spherical, "surface_model").text = "EARTH_WGS84"
    ElementTree.SubElement(spherical, "world_frame_orientation").text = "ENU"
    # RMF reference-image and Cartesian maps do not carry a surveyed geodetic
    # origin.  Use PX4's deterministic SITL origin for sensor simulation while
    # keeping all navigation and qualification in the map's local ENU frame.
    ElementTree.SubElement(spherical, "latitude_deg").text = "47.397971057728974"
    ElementTree.SubElement(spherical, "longitude_deg").text = "8.546163739800146"
    ElementTree.SubElement(spherical, "elevation").text = "0"
    ElementTree.SubElement(spherical, "heading_deg").text = "0"
    physics = ElementTree.SubElement(world, "physics", {"name": "1ms", "type": "ignored"})
    ElementTree.SubElement(physics, "max_step_size").text = "0.001"
    ElementTree.SubElement(physics, "real_time_factor").text = "1.0"
    systems = (
        ("gz::sim::systems::Physics", "gz-sim-physics-system"),
        ("gz::sim::systems::UserCommands", "gz-sim-user-commands-system"),
        ("gz::sim::systems::SceneBroadcaster", "gz-sim-scene-broadcaster-system"),
        ("gz::sim::systems::Contact", "gz-sim-contact-system"),
        ("gz::sim::systems::Imu", "gz-sim-imu-system"),
        ("gz::sim::systems::AirPressure", "gz-sim-air-pressure-system"),
        ("gz::sim::systems::AirSpeed", "gz-sim-air-speed-system"),
        ("gz::sim::systems::ApplyLinkWrench", "gz-sim-apply-link-wrench-system"),
        ("gz::sim::systems::NavSat", "gz-sim-navsat-system"),
        ("gz::sim::systems::Magnetometer", "gz-sim-magnetometer-system"),
        ("gz::sim::systems::Sensors", "gz-sim-sensors-system"),
    )
    for plugin_name, filename in systems:
        plugin = ElementTree.SubElement(
            world, "plugin", {"name": plugin_name, "filename": filename}
        )
        if plugin_name.endswith("Sensors"):
            ElementTree.SubElement(plugin, "render_engine").text = "ogre2"
    light = ElementTree.SubElement(world, "light", {"type": "directional", "name": "sun"})
    ElementTree.SubElement(light, "cast_shadows").text = "true"
    ElementTree.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ElementTree.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ElementTree.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
    ElementTree.SubElement(light, "direction").text = "-0.5 0.1 -0.9"
    scene = ElementTree.SubElement(world, "scene")
    ElementTree.SubElement(scene, "ambient").text = "0.7 0.7 0.7 1"
    model = ElementTree.SubElement(world, "model", {"name": f"{world_name}-static"})
    ElementTree.SubElement(model, "static").text = "true"

    semantic_levels: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    named_entities: dict[str, str] = {}
    collision_primitives: list[dict[str, Any]] = []
    segment_count = 0
    launch_node_id: str | None = None
    level_items = list(source["levels"].items())
    for level_index, (level_name, raw_level) in enumerate(level_items):
        if not isinstance(level_name, str) or not isinstance(raw_level, dict):
            raise RmfBuildingConversionError("RMF_BUILDING_LEVEL_INVALID")
        points, scale = _metric_vertices(raw_level, system)
        elevation = _finite(raw_level.get("elevation", level_index * 3.2), "RMF_ELEVATION_INVALID")
        walls = raw_level.get("walls", []) or []
        floors = raw_level.get("floors", []) or []
        lanes = raw_level.get("lanes", []) or []
        doors = raw_level.get("doors", []) or []
        if any(not isinstance(items, list) for items in (walls, floors, lanes, doors)):
            raise RmfBuildingConversionError("RMF_BUILDING_COLLECTION_INVALID")
        segment_count += len(walls) + len(lanes) + len(doors)
        if segment_count > MAX_RMF_SEGMENTS:
            raise RmfBuildingConversionError("RMF_BUILDING_SEGMENT_LIMIT_EXCEEDED")

        floor_bounds: list[list[float]] = []
        for floor_index, floor in enumerate(floors):
            indices = floor.get("vertices") if isinstance(floor, dict) else None
            if not isinstance(indices, list) or len(indices) < 3:
                continue
            try:
                polygon = [points[int(index)] for index in indices]
            except (IndexError, TypeError, ValueError) as error:
                raise RmfBuildingConversionError("RMF_FLOOR_VERTEX_INVALID") from error
            low_x, high_x = min(p[0] for p in polygon), max(p[0] for p in polygon)
            low_y, high_y = min(p[1] for p in polygon), max(p[1] for p in polygon)
            thickness = 0.12
            size = (max(0.05, high_x - low_x), max(0.05, high_y - low_y), thickness)
            center = ((low_x + high_x) / 2, (low_y + high_y) / 2, elevation - thickness / 2)
            _add_box(model, name=f"floor-{level_index}-{floor_index}", center=center, size=size)
            floor_bounds.append([low_x, low_y, elevation - thickness, high_x, high_y, elevation])
            collision_primitives.append(
                {
                    "name": f"floor-{level_index}-{floor_index}",
                    "semantic": "terrain",
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "size_x": size[0],
                    "size_y": size[1],
                    "size_z": size[2],
                    "yaw_rad": 0.0,
                }
            )

        # Outdoor/WGS84 RMF maps often contain only lanes.  Preserve them as an
        # explicitly bounded terrain pad instead of inventing building walls.
        if not floor_bounds:
            low_x, high_x = min(point[0] for point in points), max(point[0] for point in points)
            low_y, high_y = min(point[1] for point in points), max(point[1] for point in points)
            margin, thickness = 5.0, 0.12
            low_x, high_x = low_x - margin, high_x + margin
            low_y, high_y = low_y - margin, high_y + margin
            size = (max(0.05, high_x - low_x), max(0.05, high_y - low_y), thickness)
            center = (
                (low_x + high_x) / 2,
                (low_y + high_y) / 2,
                elevation - thickness / 2,
            )
            _add_box(model, name=f"terrain-{level_index}", center=center, size=size)
            floor_bounds.append([low_x, low_y, elevation - thickness, high_x, high_y, elevation])
            collision_primitives.append(
                {
                    "name": f"terrain-{level_index}",
                    "semantic": "terrain",
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "size_x": size[0],
                    "size_y": size[1],
                    "size_z": size[2],
                    "yaw_rad": 0.0,
                }
            )

        wall_records: list[dict[str, Any]] = []
        for wall_index, wall in enumerate(walls):
            if not isinstance(wall, list) or len(wall) < 2:
                raise RmfBuildingConversionError("RMF_WALL_INVALID")
            try:
                a, b = points[int(wall[0])], points[int(wall[1])]
            except (IndexError, TypeError, ValueError) as error:
                raise RmfBuildingConversionError("RMF_WALL_VERTEX_INVALID") from error
            length = math.dist(a, b)
            if length <= 1e-5:
                continue
            height, thickness = 2.8, 0.14
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, elevation + height / 2)
            size = (length, thickness, height)
            _add_box(
                model, name=f"wall-{level_index}-{wall_index}", center=center, size=size, yaw=yaw
            )
            wall_records.append(
                {"start_m": list(a), "end_m": list(b), "height_m": height, "thickness_m": thickness}
            )
            collision_primitives.append(
                {
                    "name": f"wall-{level_index}-{wall_index}",
                    "semantic": "wall",
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "size_x": size[0],
                    "size_y": size[1],
                    "size_z": size[2],
                    "yaw_rad": yaw,
                }
            )

        raw_vertices = raw_level.get("vertices", [])
        flight_height = 6.0 if system == "wgs84" else 1.4
        launch_vertex_index: int | None = None
        if launch_node_id is None:
            for lane in lanes:
                if isinstance(lane, list) and len(lane) >= 2:
                    try:
                        candidate = int(lane[0])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= candidate < len(points):
                        launch_vertex_index = candidate
                        break
            if launch_vertex_index is None:
                launch_vertex_index = 0
        for vertex_index, (point, raw_vertex) in enumerate(zip(points, raw_vertices, strict=True)):
            node_id = _slug(f"{level_name}-v{vertex_index}")
            label = raw_vertex[3] if len(raw_vertex) > 3 else ""
            label_text = label.strip() if isinstance(label, str) else ""
            lowered_label = label_text.casefold()
            node_semantic = (
                "pickup"
                if "pickup" in lowered_label or "goal" in lowered_label
                else "office"
                if "office" in lowered_label
                else "stairs"
                if "stair" in lowered_label
                else "door"
                if "door" in lowered_label
                else "outdoor"
                if system == "wgs84"
                else "corridor"
            )
            if launch_node_id is None and vertex_index == launch_vertex_index:
                node_semantic = "launch"
                launch_node_id = node_id
            graph_nodes.append(
                {
                    "node_id": node_id,
                    "label": label_text or f"{level_name} vertex {vertex_index}",
                    "position_m": {
                        "x": point[0],
                        "y": point[1],
                        "z": elevation + flight_height,
                    },
                    "semantic": node_semantic,
                }
            )
            if label_text:
                entity_id = _slug(f"{level_name}-{label_text}")
                named_entities[entity_id] = node_id
                entities.append(
                    {
                        "entity_id": entity_id,
                        "aliases": [label_text, f"{level_name} {label_text}"],
                        "position_m": [point[0], point[1], elevation + flight_height],
                        "semantic": "rmf-landmark",
                        "source_pointer": f"/levels/{level_name}/vertices/{vertex_index}",
                    }
                )
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, list) or len(lane) < 2:
                raise RmfBuildingConversionError("RMF_LANE_INVALID")
            try:
                first, second = int(lane[0]), int(lane[1])
                points[first]
                points[second]
            except (IndexError, TypeError, ValueError) as error:
                raise RmfBuildingConversionError("RMF_LANE_VERTEX_INVALID") from error
            a, b = points[first], points[second]
            distance = math.dist(a, b)
            if distance <= 1e-6:
                continue
            parameters = lane[2] if len(lane) > 2 else {}
            speed = _parameter(parameters, "speed_limit")
            speed_limit = float(speed) if type(speed) in (int, float) and float(speed) > 0 else 0.5
            graph_edges.append(
                {
                    "edge_id": _slug(f"{level_name}-lane{lane_index}"),
                    "from_node": _slug(f"{level_name}-v{first}"),
                    "to_node": _slug(f"{level_name}-v{second}"),
                    "distance_m": distance,
                    "minimum_clearance_m": 0.0,
                    "speed_limit_mps": min(3.0, speed_limit),
                    # RMF lane direction constrains ground-robot traffic flow.  It is not a
                    # physical one-way constraint for a UAV corridor; the independently
                    # computed three-dimensional clearance check decides whether either
                    # direction is usable by the selected aircraft.
                    "bidirectional": True,
                    "qualification": "geometry-derived",
                }
            )

        semantic_levels.append(
            {
                "level_id": level_name,
                "elevation_m": elevation,
                "scale_m_per_source_unit": scale,
                "floor_bounds_m": floor_bounds,
                "walls": wall_records,
                "doors": len(doors),
                "lanes": len(lanes),
            }
        )

    if not entities:
        first = graph_nodes[0]
        entities.append(
            {
                "entity_id": "map-origin",
                "aliases": ["map origin"],
                "position_m": list(first["position_m"].values()),
                "semantic": "reference",
                "source_pointer": "/levels",
            }
        )
    if launch_node_id is None:
        raise RmfBuildingConversionError("RMF_BUILDING_LAUNCH_NODE_MISSING")
    launch_node = next(node for node in graph_nodes if node["node_id"] == launch_node_id)
    named_entities.setdefault("launch", launch_node_id)
    runtime_bindings = {
        "schema_version": "dronedream.map-runtime-bindings.v1",
        "coordinate_frame": "ENU",
        "simulator": "gazebo-harmonic",
        "vehicle_spawn": {
            "x": launch_node["position_m"]["x"],
            "y": launch_node["position_m"]["y"],
            "z": semantic_levels[0]["elevation_m"] + 0.25,
        },
        "mission_launch_waypoint": launch_node["position_m"],
    }
    semantic_obj = {
        "schema_version": "dronedream.map-semantic.v1",
        "scene_id": world_name,
        "coordinate_frame": "ENU",
        "entities": entities,
        "navigation": {"segment_ids": [edge["edge_id"] for edge in graph_edges]},
        "levels": semantic_levels,
        "collision_primitives": collision_primitives,
        # The generated SDF is constructed from these exact primitives, so the
        # live depth/safety worker can use the same immutable geometry instead
        # of treating the converted world as having no runtime obstacles.
        "runtime_collision_primitives": collision_primitives,
        "runtime_bindings": runtime_bindings,
        "known_limits": [
            "RMF lanes are ground-robot topology hints and are not UAV corridors.",
            "Generated geometry is conservative and requires map-aircraft runtime qualification.",
            "Dynamic doors, lifts, referenced models and textures require optional "
            "high-fidelity conversion.",
        ],
    }
    topology_obj = {
        "schema_version": "dronedream.map-graph.v1",
        "asset_id": world_name,
        "name": name,
        "coordinate_frame": "map_enu",
        "nodes": graph_nodes,
        "edges": graph_edges,
        "named_entities": named_entities,
    }
    report_obj = {
        "schema_version": "dronedream.rmf-conversion-report.v1",
        "source_name": source_name,
        "world_name": world_name,
        "coordinate_system": system,
        "levels": len(semantic_levels),
        "entities": len(entities),
        "nodes": len(graph_nodes),
        "edges": len(graph_edges),
        "collision_primitives": len(collision_primitives),
        "flight_qualified": False,
        "requires": [
            "map-aircraft-pair-qualification",
            "gazebo-load-test",
            "route-clearance-test",
            "takeoff-hover-land-test",
        ],
    }
    ElementTree.indent(sdf, space="  ")
    sdf_bytes = ElementTree.tostring(sdf, encoding="utf-8", xml_declaration=True)

    def encode(value: object) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    return RmfBuildingConversion(
        name=name,
        sdf=sdf_bytes,
        semantic=encode(semantic_obj),
        topology=encode(topology_obj),
        report=encode(report_obj),
    )
