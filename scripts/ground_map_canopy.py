"""Materialize observed canopy exclusions as honest static simulation obstacles.

These are extruded observed footprints, not measured tree reconstructions.
Navigation retains the exact all-altitude exclusion polygons. The mesh only
adds matching physical / depth-visible obstacles to the simulation world.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

MESH_URI = "meshes/kumpula-observed-canopy.obj"
LINK_NAME = "observed-canopy-obstacles"
WORLD_FILES = ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf")
CHUNK_TARGET_TRIANGLES = 4096
CHUNK_MAX_TRIANGLES = 6144


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _height(feature: dict, operating_upper_m: float) -> tuple[float, str]:
    evidence = feature.get("evidence", {})
    layer = str(evidence.get("source_layer", ""))
    for token, height in (
        ("puusto_2_10m", 10.0),
        ("puusto_10_15m", 15.0),
        ("puusto_15_20m", 20.0),
        ("puusto_yli20m", 24.0),
    ):
        if token in layer:
            note = (
                "class-upper-bound-column-not-measured-tree-height"
                if height != 24.0
                else "greater-than-20m-class-24m-illustration-not-measured-upper-bound"
            )
            return height, note
    # Imagery-only observations provide no measured canopy altitude. A column
    # spanning the operating band conservatively prevents passing through it.
    return operating_upper_m, "unmeasured-height-column-covering-configured-operating-band"


def _observed_polygon(feature: dict) -> Polygon:
    """Remove only exact duplicate consecutive coordinates, never simplify XY."""
    if feature.get("kind") != "tree-canopy":
        raise ValueError("Canopy mesh only accepts observed canopy exclusions")
    if feature.get("holes_xy_m"):
        raise ValueError("Decompose canopy holes exactly before producing the mesh")
    points = feature.get("polygon_xy_m")
    if not isinstance(points, list) or not 3 <= len(points) <= 4096:
        raise ValueError("Invalid canopy footprint vertex budget")
    if any(
        not isinstance(p, (list, tuple))
        or len(p) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
        for p in points
    ):
        raise ValueError("Invalid canopy footprint coordinates")
    ring = []
    for point in points:
        point = tuple(point)
        if not ring or point != ring[-1]:
            ring.append(point)
    if len(ring) > 1 and ring[-1] == ring[0]:
        ring.pop()
    if len(ring) < 3:
        raise ValueError("Invalid observed canopy polygon")
    poly = Polygon(ring)
    if not poly.is_valid or poly.is_empty or poly.area <= 0:
        raise ValueError("Invalid observed canopy polygon")
    return orient(poly, sign=1.0)


def canopy_mesh(
    features: list[dict],
    *,
    operating_upper_m: float = 12.0,
    origin_xy_m: tuple[float, float] = (0.0, 0.0),
    mesh_uri: str = MESH_URI,
) -> tuple[bytes, dict]:
    """Return a closed triangle mesh with exactly the observed XY footprints."""
    if not math.isfinite(operating_upper_m) or not 0 < operating_upper_m <= 1000:
        raise ValueError("Invalid canopy operating band")
    if len(origin_xy_m) != 2 or any(not math.isfinite(v) for v in origin_xy_m):
        raise ValueError("Invalid canopy mesh origin")
    lines = [
        "# Observed footprint columns for simulation; not measured tree forms.",
        "# Source geometry is unchanged. Height classes are explanatory proxies.",
    ]
    point_count = face_count = 0
    footprints = []
    seen = set()
    for feature in features:
        identifier = feature.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Canopy IDs must be unique nonempty strings")
        seen.add(identifier)
        points = feature.get("polygon_xy_m")
        poly = _observed_polygon(feature)
        triangles = list(constrained_delaunay_triangles(poly).geoms)
        if not triangles or any(t.difference(poly).area > 1e-7 for t in triangles):
            raise ValueError("Canopy triangulation crosses the observed footprint")
        if poly.symmetric_difference(unary_union(triangles)).area > 1e-7:
            raise ValueError("Canopy triangulation lost observed coverage")
        height, height_basis = _height(feature, operating_upper_m)
        floor = 0.0
        # The map contract uses a flat ground Z=0. Do not silently consume an
        # unregistered altitude reference from image metadata.
        if feature.get("ground_z_m", 0.0) != 0.0:
            raise ValueError("Canopy ground elevation needs an explicit registered 3D source")
        vertices: list[tuple[float, float, float]] = []
        indices: dict[tuple[float, float, float], int] = {}
        faces = []

        def index(
            x: float, y: float, z: float, *, vertices=vertices, indices=indices, offset=point_count
        ) -> int:
            point = (x - origin_xy_m[0], y - origin_xy_m[1], z)
            if point not in indices:
                vertices.append(point)
                indices[point] = offset + len(vertices)
            return indices[point]

        for triangle in triangles:
            ring = list(orient(triangle, sign=1.0).exterior.coords)[:-1]
            faces.append(tuple(index(x, y, height) for x, y in ring))
            faces.append(tuple(index(x, y, floor) for x, y in reversed(ring)))
        boundary = list(poly.exterior.coords)
        for first, second in zip(boundary[:-1], boundary[1:], strict=True):
            a, b = index(*first, floor), index(*second, floor)
            c, d = index(*second, height), index(*first, height)
            faces.extend(((a, b, c), (a, c, d)))
        lines.append("g canopy-" + re.sub(r"[^a-zA-Z0-9_-]", "_", identifier))
        lines.extend("v " + " ".join(format(v, ".17g") for v in point) for point in vertices)
        lines.extend("f " + " ".join(str(v) for v in face) for face in faces)
        point_count += len(vertices)
        face_count += len(faces)
        footprints.append(
            {
                "id": identifier,
                "polygon_sha256": _digest(points),
                "source_evidence_sha256": _digest(feature.get("evidence", {})),
                "area_m2": poly.area,
                "ground_z_m": floor,
                "illustration_top_z_m": height,
                "height_basis": height_basis,
            }
        )
    content = ("\n".join(lines) + "\n").encode("utf-8")
    return content, {
        "schema_version": "dronedream.observed-canopy-mesh.v1",
        "mesh_uri": mesh_uri,
        "pose_xyz_m": [*origin_xy_m, 0.0],
        "mesh_sha256": hashlib.sha256(content).hexdigest(),
        "source_exclusions_sha256": _digest(features),
        "polygon_count": len(features),
        "vertices": point_count,
        "triangles": face_count,
        "operating_band_m": [0.0, operating_upper_m],
        "geometry_method": "exact-constrained-triangulation-and-vertical-extrusion-no-xy-buffer",
        "height_fidelity": "class-based-or-operating-band-illustration-not-measured-tree-height",
        "navigation_authority": "unchanged-traversability-tree-canopy-exclusions-at-all-altitudes",
        "coverage_limit": (
            "Observed dated footprints only; land-cover source may clip crowns over roads; "
            "no complete canopy-clearance claim."
        ),
        "footprints": footprints,
    }


def canopy_mesh_bundle(features: list[dict], *, operating_upper_m: float = 12.0):
    """Pack complete closed columns into spatially local, bounded OBJ resources.

    Split along the longest spatial axis, balancing triangle counts. No canopy
    shell is sliced: closedness, interior tests and boundary visibility survive.
    The local origin keeps each optical broad-phase bound near its real canopy.
    """
    records = [(feature, _observed_polygon(feature)) for feature in features]
    identifiers = [feature.get("id") for feature in features]
    if not records:
        raise ValueError("Canopy bundle needs observed polygons")
    if any(not isinstance(i, str) or not i for i in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Canopy IDs must be unique nonempty strings")

    def weight(record):
        return 4 * (len(record[1].exterior.coords) - 1) - 4

    def partition(items):
        total = sum(map(weight, items))
        if total <= CHUNK_TARGET_TRIANGLES or len(items) == 1:
            if total > CHUNK_MAX_TRIANGLES:
                raise ValueError("A complete canopy shell exceeds the bounded mesh budget")
            return [items]
        axis = max(
            range(2),
            key=lambda a: (
                max(p.bounds[a + 2] for _, p in items) - min(p.bounds[a] for _, p in items)
            ),
        )
        items = sorted(items, key=lambda item: (item[1].centroid.coords[0][axis], item[0]["id"]))
        subtotal, middle = 0, 1
        for position, item in enumerate(items[:-1], 1):
            middle = position
            subtotal += weight(item)
            if subtotal >= total / 2:
                break
        return partition(items[:middle]) + partition(items[middle:])

    meshes, receipts = {}, []
    for number, chunk in enumerate(partition(records)):
        bounds = [poly.bounds for _, poly in chunk]
        origin = tuple(
            (min(b[a] for b in bounds) + max(b[a + 2] for b in bounds)) / 2 for a in range(2)
        )
        uri = f"meshes/kumpula-observed-canopy-{number:03d}.obj"
        raw, receipt = canopy_mesh(
            [f for f, _ in chunk],
            operating_upper_m=operating_upper_m,
            origin_xy_m=origin,
            mesh_uri=uri,
        )
        if receipt["triangles"] > CHUNK_MAX_TRIANGLES or receipt["vertices"] > 8192:
            raise ValueError("Canopy chunk exceeds the bounded mesh budget")
        meshes[uri] = raw
        receipts.append(receipt)
    return meshes, {
        "schema_version": "dronedream.observed-canopy-mesh-bundle.v2",
        "source_exclusions_sha256": _digest(features),
        "geometry_method": "exact-closed-shell-spatial-partition-with-local-origins",
        "coordinate_cleanup": "exact-consecutive-duplicate-removal-only-no-simplification",
        "polygon_count": len(features),
        "vertices": sum(r["vertices"] for r in receipts),
        "triangles": sum(r["triangles"] for r in receipts),
        "chunk_target_triangles": CHUNK_TARGET_TRIANGLES,
        "chunk_max_triangles": CHUNK_MAX_TRIANGLES,
        "mesh_count": len(receipts),
        "meshes": receipts,
    }


def install_canopy_obstacles(gazebo: Path, semantic: dict) -> dict:
    """Install identical locally framed meshes in every world; caller saves semantic.

    Call before regenerating summary / asset hashes. Existing box and cylinder
    primitives remain unchanged: the semantic polygon layer is the geometry
    authority for these irregular obstacles, and SDF uses the exact mesh.
    """
    layer = semantic.get("traversability", {})
    features = [f for f in layer.get("exclusions", []) if f.get("kind") == "tree-canopy"]
    if not features:
        return {"status": "no-observed-canopy"}
    meshes, receipt = canopy_mesh_bundle(
        features, operating_upper_m=layer.get("maximum_agl_m", 12.0)
    )
    trees = []
    for name in WORLD_FILES:
        path = gazebo / name
        tree = ET.parse(path)
        model = tree.getroot().find("model")
        if model is None:
            model = tree.getroot().find("world/model[@name='kumpula_campus']")
        if model is None or model.findtext("static") != "true":
            raise ValueError(f"Missing static registered campus model in {name}")
        for old in list(model.findall(f"link[@name='{LINK_NAME}']")):
            model.remove(old)
        link = ET.SubElement(model, "link", {"name": LINK_NAME})
        for number, chunk in enumerate(receipt["meshes"]):
            for tag in ("collision", "visual"):
                item = ET.SubElement(link, tag, {"name": f"observed-canopy-{tag}-{number:03d}"})
                ET.SubElement(item, "pose").text = (
                    " ".join(format(v, ".17g") for v in chunk["pose_xyz_m"]) + " 0 0 0"
                )
                geom = ET.SubElement(item, "geometry")
                shape = ET.SubElement(geom, "mesh")
                ET.SubElement(shape, "uri").text = chunk["mesh_uri"]
                ET.SubElement(shape, "scale").text = "1 1 1"
                if tag == "visual":
                    material = ET.SubElement(item, "material")
                    ET.SubElement(material, "ambient").text = "0.12 0.25 0.10 1"
                    ET.SubElement(material, "diffuse").text = "0.20 0.36 0.15 1"
                    ET.SubElement(material, "specular").text = "0 0 0 1"
        trees.append((path, tree))
    collision_path = gazebo / "collision-primitives.json"
    document = json.loads(collision_path.read_text("utf-8"))
    if document.get("primitives") != semantic.get("collision_primitives"):
        raise ValueError("Physical collision metadata is not the same selected map")
    # Resolve every input and build every world before mutating the candidate.
    for uri, raw in meshes.items():
        mesh_path = gazebo / uri
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(raw)
    for path, tree in trees:
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    existing = [
        m
        for m in document.get("non_primitive_collision_meshes", [])
        if m.get("mesh_uri") != MESH_URI
        and m.get("schema_version") != "dronedream.observed-canopy-mesh-bundle.v2"
    ]
    document["non_primitive_collision_meshes"] = existing + [receipt]
    collision_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", "utf-8")
    semantic["canopy_collision_meshes"] = [receipt]
    semantic.setdefault("fidelity", {})["tree_obstacles"] = (
        "observed-footprint-static-prisms; class-height-illustration; incomplete-canopy-coverage"
    )
    return receipt
