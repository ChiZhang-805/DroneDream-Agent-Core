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


def canopy_mesh(features: list[dict], *, operating_upper_m: float = 12.0) -> tuple[bytes, dict]:
    """Return a closed triangle mesh with exactly the observed XY footprints."""
    if not math.isfinite(operating_upper_m) or not 0 < operating_upper_m <= 1000:
        raise ValueError("Invalid canopy operating band")
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
        poly = Polygon(points)
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            raise ValueError("Invalid observed canopy polygon")
        poly = orient(poly, sign=1.0)
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
            point = (x, y, z)
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
        lines.extend("v " + " ".join(format(v, ".15g") for v in point) for point in vertices)
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
        "mesh_uri": MESH_URI,
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


def install_canopy_obstacles(gazebo: Path, semantic: dict) -> dict:
    """Install one shared mesh in all world variants; caller then saves semantic.

    Call before regenerating summary / asset hashes. Existing box and cylinder
    primitives remain unchanged: the semantic polygon layer is the geometry
    authority for these irregular obstacles, and SDF uses the exact mesh.
    """
    layer = semantic.get("traversability", {})
    features = [f for f in layer.get("exclusions", []) if f.get("kind") == "tree-canopy"]
    if not features:
        return {"status": "no-observed-canopy"}
    mesh, receipt = canopy_mesh(features, operating_upper_m=layer.get("maximum_agl_m", 12.0))
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
        for tag in ("collision", "visual"):
            item = ET.SubElement(link, tag, {"name": "observed-canopy-" + tag})
            geom = ET.SubElement(item, "geometry")
            shape = ET.SubElement(geom, "mesh")
            ET.SubElement(shape, "uri").text = MESH_URI
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
    mesh_path = gazebo / MESH_URI
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_bytes(mesh)
    for path, tree in trees:
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    existing = [
        m
        for m in document.get("non_primitive_collision_meshes", [])
        if m.get("mesh_uri") != MESH_URI
    ]
    document["non_primitive_collision_meshes"] = existing + [receipt]
    collision_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", "utf-8")
    semantic["canopy_collision_meshes"] = [receipt]
    semantic.setdefault("fidelity", {})["tree_obstacles"] = (
        "observed-footprint-static-prisms; class-height-illustration; incomplete-canopy-coverage"
    )
    return receipt
