"""Expanded official footprints remain complete and usable by the real compiler."""

import importlib.util
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from dronedream_agent_core.optical_map import _obj_triangles

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ground_map_extent_buildings", ROOT / "scripts/ground_map_extent_buildings.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
SOURCE = ROOT / "scripts/data/campus-traversability/expanded-buildings.json"
BOUNDS = [-281.25, -250, 281.25, 312.5]


def _base(tmp_path):
    path = tmp_path / "gazebo"
    package = ROOT / "app/desktop/src-tauri/resources/default-assets/school-map.ddpkg"
    with zipfile.ZipFile(package) as archive:
        prefix = "normalized/map/gazebo/"
        for name in archive.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                destination = path / name[len(prefix):]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
    semantic = json.loads((path / "semantic.json").read_text(encoding="utf-8"))
    return path, semantic


def _normals(raw):
    vertices, normals = [], []
    for line in raw.decode().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("vn "):
            normals.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("f "):
            corners = [token.split("/") for token in line.split()[1:]]
            assert all(len(c) == 3 and c[1] == "" for c in corners)
            assert len({c[2] for c in corners}) == 1
            a, b, c = [np.array(vertices[int(token[0]) - 1]) for token in corners[:3]]
            normal = np.cross(b - a, c - a)
            assert math.hypot(*normal) > 0
            assert normals[int(corners[0][2]) - 1] == pytest.approx(normal / np.linalg.norm(normal))


def test_actual_source_preserves_old_footprints_and_covers_every_official_polygon(tmp_path):
    gazebo, semantic = _base(tmp_path)
    base = module._base_buildings(gazebo / "meshes/kumpula-buildings.obj", semantic)
    report = module.install_extent_buildings(gazebo, semantic, SOURCE, BOUNDS)
    new = module._base_buildings(gazebo / "meshes/kumpula-buildings.obj", semantic)
    assert len(base) == 85
    assert semantic["building_count"] == 86
    assert base.keys() <= new.keys()
    for name, old in base.items():
        assert old["polygon"].difference(new[name]["polygon"].buffer(1e-8)).area < 1e-6
        assert old["primitive"]["size_z"] == new[name]["primitive"]["size_z"]
    union = unary_union([item["polygon"] for item in new.values()])
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    for feature in source["features"]:
        polygon = Polygon(feature["polygon_xy_m"])
        assert polygon.difference(union.buffer(1e-8)).area < 1e-6
    assert len(new) == report["building_volume_count"] == semantic["building_count"]
    assert len(report["visual_mesh_resources"]) >= 2
    for uri in report["visual_mesh_resources"]:
        raw = (gazebo / uri).read_bytes()
        triangles, closed = _obj_triangles(raw, np.ones(3))
        assert closed and len(triangles) <= 8192
        _normals(raw)
    for name in ("model.sdf", "world.sdf", "world.perception.sdf"):
        tree = ET.parse(gazebo / name)
        actual = [node.text for node in tree.findall(".//link[@name='building-visuals']//uri")]
        assert actual == report["visual_mesh_resources"]


def test_concave_building_roof_does_not_cover_open_courtyard(tmp_path):
    polygon = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])
    mesh = tmp_path / "concave.obj"
    module._write_mesh(mesh, {"building": {
        "polygon": polygon, "primitive": module._primitive("building", polygon, 8),
        "material": "context",
    }})
    raw = mesh.read_bytes()
    triangles, closed = _obj_triangles(raw, np.ones(3))
    assert closed
    top = unary_union([
        Polygon([(x, y) for x, y, _ in triangle])
        for triangle in triangles if all(p[2] == 8 for p in triangle)
    ])
    assert top.symmetric_difference(polygon).area < 1e-10
    _normals(raw)


def test_enclosed_courtyard_survives_mesh_export_and_reimport(tmp_path):
    polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)],
                      [[(5, 5), (5, 15), (15, 15), (15, 5)]])
    mesh = tmp_path / "meshes" / "courtyard.obj"
    mesh.parent.mkdir()
    module._write_mesh(mesh, {"courtyard": {
        "polygon": polygon, "primitive": module._primitive("courtyard", polygon, 48),
        "material": "context",
    }})
    triangles, closed = _obj_triangles(mesh.read_bytes(), np.ones(3))
    assert closed
    result = module._base_buildings(mesh, {})["courtyard"]
    assert result["polygon"].symmetric_difference(polygon).area < 1e-10
    assert result["primitive"]["size_z"] == 48
    _normals(mesh.read_bytes())


def test_building_snapshot_cannot_silently_cover_a_different_map_extent(tmp_path):
    gazebo, semantic = _base(tmp_path)
    with pytest.raises(ValueError, match="exact expanded map extent"):
        module.install_extent_buildings(gazebo, semantic, SOURCE, [-300, -250, 300, 350])
