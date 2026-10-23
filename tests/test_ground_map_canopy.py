"""The simulation canopy must not fill road gaps absent from its source polygons."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

SPEC = importlib.util.spec_from_file_location(
    "ground_map_canopy", Path(__file__).parents[1] / "scripts/ground_map_canopy.py"
)
canopy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canopy)


def _feature(identifier, points, height_class="puusto_10_15m_2024"):
    return {
        "id": identifier,
        "kind": "tree-canopy",
        "polygon_xy_m": points,
        "evidence": {"source_layer": height_class, "source_sha256": "a" * 64},
    }


def _top_footprint(raw):
    vertices, triangles = [], []
    for line in raw.decode().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:])))
        if line.startswith("f "):
            points = [vertices[int(i) - 1] for i in line.split()[1:]]
            if len({p[2] for p in points}) == 1 and points[0][2] > 0:
                triangles.append(Polygon([(x, y) for x, y, _ in points]))
    return unary_union(triangles)


def test_concave_canopy_does_not_fill_open_road_or_bounding_box():
    points = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]
    raw, receipt = canopy.canopy_mesh([_feature("l-shaped", points)])
    footprint = _top_footprint(raw)
    assert footprint.symmetric_difference(Polygon(points)).area < 1e-10
    assert not footprint.intersects(Polygon([[3, 3], [5, 3], [5, 5], [3, 5]]))
    assert receipt["footprints"][0]["illustration_top_z_m"] == 15.0
    assert receipt["vertices"] == 12


def test_canopy_surface_is_closed_and_outward_wound_for_physical_collision():
    raw, _ = canopy.canopy_mesh(
        [_feature("concave", [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]])]
    )
    vertices, edges, volume = [], Counter(), 0.0
    for line in raw.decode().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:])))
        if line.startswith("f "):
            ids = tuple(map(int, line.split()[1:]))
            for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                edges[tuple(sorted((a, b)))] += 1
            a, b, c = [vertices[i - 1] for i in ids]
            cross = (
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            )
            volume += sum(a[i] * cross[i] for i in range(3)) / 6
    assert set(edges.values()) == {2}
    assert volume == pytest.approx(20 * 15.0)


def test_decomposed_hole_remains_open_in_physics_and_visual_mesh():
    # Four exact source pieces around a 2x2 open courtyard; none fills its hole.
    pieces = [
        ([[0, 0], [6, 0], [6, 2], [0, 2]]),
        ([[0, 4], [6, 4], [6, 6], [0, 6]]),
        ([[0, 2], [2, 2], [2, 4], [0, 4]]),
        ([[4, 2], [6, 2], [6, 4], [4, 4]]),
    ]
    raw, _ = canopy.canopy_mesh([_feature(str(i), p) for i, p in enumerate(pieces)])
    expected = Polygon([[0, 0], [6, 0], [6, 6], [0, 6]], holes=[[[2, 2], [4, 2], [4, 4], [2, 4]]])
    assert _top_footprint(raw).symmetric_difference(expected).area < 1e-10


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("puusto_2_10m_2024", 10.0),
        ("puusto_15_20m_2024", 20.0),
        ("puusto_yli20m_2024", 24.0),
        ("", 12.0),
    ],
)
def test_height_proxy_remains_explicit_not_claimed_measured(layer, expected):
    _, receipt = canopy.canopy_mesh([_feature("one", [[0, 0], [2, 0], [0, 2]], layer)])
    assert receipt["footprints"][0]["illustration_top_z_m"] == expected
    assert "not-measured" in receipt["height_fidelity"]


def test_reject_unhandled_holes_instead_of_filling_them():
    f = _feature("hole", [[0, 0], [6, 0], [6, 6], [0, 6]])
    f["holes_xy_m"] = [[[2, 2], [4, 2], [4, 4], [2, 4]]]
    with pytest.raises(ValueError, match="Decompose"):
        canopy.canopy_mesh([f])


def test_all_world_variants_share_same_mesh_without_fake_box_primitives(tmp_path):
    standalone = (
        '<sdf version="1.9"><model name="kumpula_campus"><static>true</static></model></sdf>'
    )
    world = (
        '<sdf version="1.9"><world name="campus"><model name="kumpula_campus">'
        "<static>true</static></model></world></sdf>"
    )
    for name in canopy.WORLD_FILES:
        (tmp_path / name).write_text(standalone if name == "model.sdf" else world)
    primitive = {"name": "existing-ground", "center_z": -0.1, "size_z": 0.2}
    semantic = {
        "collision_primitives": [primitive],
        "traversability": {
            "maximum_agl_m": 12.0,
            "exclusions": [_feature("one", [[0, 0], [2, 0], [0, 2]])],
        },
    }
    (tmp_path / "collision-primitives.json").write_text(json.dumps({"primitives": [primitive]}))
    receipt = canopy.install_canopy_obstacles(tmp_path, semantic)
    mesh_bytes = (tmp_path / canopy.MESH_URI).read_bytes()
    assert _top_footprint(mesh_bytes).area == 2.0
    for name in canopy.WORLD_FILES:
        tree = ET.parse(tmp_path / name)
        link = tree.find(".//link[@name='observed-canopy-obstacles']")
        assert link.find("collision/geometry/mesh/uri").text == canopy.MESH_URI
        assert link.find("visual/geometry/mesh/uri").text == canopy.MESH_URI
    doc = json.loads((tmp_path / "collision-primitives.json").read_text())
    assert doc["primitives"] == semantic["collision_primitives"] == [primitive]
    assert doc["non_primitive_collision_meshes"] == semantic["canopy_collision_meshes"] == [receipt]
    # Repeated installation replaces the same link, without duplicating trees.
    assert canopy.install_canopy_obstacles(tmp_path, semantic) == receipt
    for name in canopy.WORLD_FILES:
        matches = ET.parse(tmp_path / name).findall(".//link[@name='observed-canopy-obstacles']")
        assert len(matches) == 1
