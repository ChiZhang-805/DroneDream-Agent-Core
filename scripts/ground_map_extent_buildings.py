"""Add complete official building footprints without discarding the base campus.

This preparation step precedes ``refine_building_collisions``. The latter derives
all four worlds' collisions and the planning exclusions from the OBJ generated
here, so the visual and physical geometry have one source.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def _base_buildings(path, semantic):
    groups, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("g "):
            current = line[2:]
            groups[current] = {"vertices": [], "material": "context"}
        elif line.startswith("usemtl ") and current is not None:
            groups[current]["material"] = line[7:]
        elif line.startswith("v ") and current is not None:
            groups[current]["vertices"].append(tuple(map(float, line.split()[1:4])))
    buildings = {}
    for primitive in semantic["collision_primitives"]:
        if primitive.get("semantic") != "building":
            continue
        name = primitive["name"].removesuffix("-collision")
        group = groups[name]
        floor = min(v[2] for v in group["vertices"])
        ring = [(v[0], v[1]) for v in group["vertices"] if abs(v[2] - floor) < 1e-8]
        buildings[name] = {
            "polygon": Polygon(ring), "primitive": dict(primitive),
            "material": group["material"], "source_ids": [], "original": True,
            "height_source": "preserved-base-model-height",
        }
    return buildings


def _primitive(name, polygon, height):
    rectangle = polygon.minimum_rotated_rectangle
    points = list(rectangle.exterior.coords)[:4]
    edges = [
        (math.dist(a, b), a, b)
        for a, b in zip(points, points[1:] + points[:1], strict=True)
    ]
    length, a, b = max(edges)
    return {
        "name": name + "-collision", "semantic": "building",
        "center_x": rectangle.centroid.x, "center_y": rectangle.centroid.y,
        "center_z": height / 2, "size_x": length,
        "size_y": min(edge[0] for edge in edges), "size_z": height,
        "yaw_rad": math.atan2(b[1] - a[1], b[0] - a[0]),
    }


def _write_mesh(path, buildings):
    lines = ["mtllib kumpula-buildings.mtl", "o KumpulaCampusBuildings"]
    index = 1
    normal_index = 1
    for name, item in buildings.items():
        polygon = orient(item["polygon"], sign=1.0)
        if polygon.interiors or not polygon.is_valid:
            raise ValueError(f"Building needs an explicit courtyard decomposition: {name}")
        points = list(polygon.exterior.coords)[:-1]
        count = len(points)
        p = item["primitive"]
        floor, roof = p["center_z"] - p["size_z"] / 2, p["center_z"] + p["size_z"] / 2
        lines.extend(["g " + name, "usemtl " + item["material"]])
        for z in (floor, roof):
            lines.extend(f"v {x:.10f} {y:.10f} {z:.10f}" for x, y in points)
        vertices = [
            tuple(float(f"{value:.10f}") for value in (x, y, z))
            for z in (floor, roof) for x, y in points
        ]
        faces = []
        vertex_ids = {point: index + offset for offset, point in enumerate(points)}
        for triangle in constrained_delaunay_triangles(polygon).geoms:
            corners = list(orient(triangle, sign=1.0).exterior.coords)[:3]
            ids = [vertex_ids[point] for point in corners]
            faces.extend([[i + count for i in ids], list(reversed(ids))])
        for offset in range(count):
            nxt = (offset + 1) % count
            faces.append([index + offset, index + nxt, index + count + nxt, index + count + offset])
        for face in faces:
            a, b, c = [vertices[i - index] for i in face[:3]]
            u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0])
            length = math.sqrt(sum(n * n for n in normal))
            if length <= 1e-14:
                raise ValueError(f"Degenerate building face: {name}")
            lines.append("vn " + " ".join(f"{n / length:.10f}" for n in normal))
            lines.append("f " + " ".join(f"{i}//{normal_index}" for i in face))
            normal_index += 1
        index += count * 2
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _install_visual_chunks(gazebo, buildings):
    chunks, current, count = [], {}, 0
    for name, item in buildings.items():
        faces = 4 * (len(item["polygon"].exterior.coords) - 1) - 4
        if faces > 8192:
            raise ValueError(f"Building exceeds per-mesh face budget: {name}")
        if current and count + faces > 4096:
            chunks.append(current)
            current, count = {}, 0
        current[name] = item
        count += faces
    if current:
        chunks.append(current)
    uris = []
    for i, chunk in enumerate(chunks):
        uri = f"meshes/kumpula-buildings-{i:03d}.obj"
        _write_mesh(gazebo / uri, chunk)
        uris.append(uri)
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        path = gazebo / filename
        tree = ET.parse(path)
        link = tree.find(".//link[@name='building-visuals']")
        if link is None:  # The physics-only world has no visual building link.
            continue
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for i, uri in enumerate(uris):
            visual = ET.SubElement(link, "visual", {"name": f"registered-buildings-{i:03d}"})
            mesh = ET.SubElement(ET.SubElement(visual, "geometry"), "mesh")
            ET.SubElement(mesh, "uri").text = uri
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return uris


def install_extent_buildings(gazebo, semantic, source_path, bounds_xy_m):
    """Preserve every old identity and footprint, then add intersecting source geometry.

    Register polygons and the 2024 land-cover polygons are both public HSY/city
    data. They are unioned into matching existing footprints, never used to
    replace or shrink the old campus. No footprint is clipped at the map edge.
    Heights remain explicitly exterior proxies, not measured roof elevations.
    """
    gazebo, source_path = Path(gazebo), Path(source_path)
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if source["origin_epsg3879"] != [25497920, 6676880]:
        raise ValueError("Building source does not use this campus coordinate origin")
    if list(bounds_xy_m) != source["bounds_xy_m"]:
        raise ValueError("Building query must cover the exact expanded map extent")
    if any(s["feature_count"] != s["totalFeatures"] for s in source["sources"]):
        raise ValueError("Incomplete official building source response")
    buildings = _base_buildings(gazebo / "meshes/kumpula-buildings.obj", semantic)
    base = {name: item["polygon"] for name, item in buildings.items()}
    extent = box(*bounds_xy_m)
    accepted = []
    for feature in source["features"]:
        polygon = Polygon(feature["polygon_xy_m"], feature.get("holes_xy_m", []))
        if not polygon.is_valid or polygon.area <= 0 or polygon.interiors:
            raise ValueError(f"Invalid building exterior: {feature['source_id']}")
        if not polygon.intersects(extent):
            continue
        source_id = feature["source_id"] + f"/{feature['part']}"
        accepted.append((source_id, polygon))
        # Identity matching only decides which volume owns the union. It does
        # not discard unmatched geometry or bridge separate building polygons.
        matches = [
            (polygon.intersection(item["polygon"]).area
             / min(polygon.area, item["polygon"].area), name)
            for name, item in buildings.items()
        ]
        score, name = max(matches, default=(0.0, ""))
        if score > 0.5:
            combined = buildings[name]["polygon"].union(polygon)
            if combined.geom_type == "Polygon" and not combined.interiors:
                buildings[name]["polygon"] = combined
                buildings[name]["source_ids"].append(source_id)
                continue
        name = "hsy-" + re.sub(r"[^a-zA-Z0-9_-]", "-", source_id)
        floors = feature.get("floors")
        known_floors = isinstance(floors, (int, float)) and 0 < floors < 100
        height = max(4.0, min(35.0, floors * 3.4)) if known_floors else 6.8
        buildings[name] = {
            "polygon": polygon, "primitive": _primitive(name, polygon, height),
            "material": "context", "source_ids": [source_id], "original": False,
            "height_source": "registered-floors-times-3.4m" if known_floors
            else "unmeasured-6.8m-exterior-proxy",
        }
    union = unary_union([item["polygon"] for item in buildings.values()])
    for name, polygon in list(base.items()) + accepted:
        if polygon.difference(union.buffer(1e-8)).area > 1e-6:
            raise ValueError(f"Building coverage lost: {name}")
    _write_mesh(gazebo / "meshes/kumpula-buildings.obj", buildings)
    visual_resources = _install_visual_chunks(gazebo, buildings)
    semantic["collision_primitives"] = [
        p for p in semantic["collision_primitives"] if p.get("semantic") != "building"
    ] + [item["primitive"] for item in buildings.values()]
    report = {
        "schema_version": "dronedream.expanded-building-evidence.v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(), "sources": source["sources"],
        "origin_epsg3879": source["origin_epsg3879"], "bounds_xy_m": list(bounds_xy_m),
        "license": source["license"], "attribution": source["attribution"],
        "height_policy": source["height_policy"], "base_building_count": len(base),
        "building_volume_count": len(buildings), "source_polygon_count": len(accepted),
        "visual_mesh_resources": visual_resources,
        "buildings": [
            {"id": name, "original": item["original"], "source_ids": item["source_ids"],
             "height_m": item["primitive"]["size_z"], "height_source": item["height_source"]}
            for name, item in buildings.items()
        ],
    }
    (gazebo / "building-extent-evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    semantic["building_count"] = len(buildings)
    semantic["sources"]["expanded_buildings_sha256"] = report["source_sha256"]
    semantic["sources"]["expanded_buildings_attribution"] = source["attribution"]
    summary_path = gazebo / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["building_count"] = len(buildings)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return report
