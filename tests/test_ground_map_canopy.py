"""The simulation canopy must not fill road gaps absent from its source polygons."""

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from shapely.affinity import translate
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
    chunk = receipt["meshes"][0]
    mesh_uri = chunk["mesh_uri"]
    mesh_bytes = (tmp_path / mesh_uri).read_bytes()
    assert _top_footprint(mesh_bytes).area == 2.0
    for name in canopy.WORLD_FILES:
        tree = ET.parse(tmp_path / name)
        link = tree.find(".//link[@name='observed-canopy-obstacles']")
        assert link.find("collision/geometry/mesh/uri").text == mesh_uri
        assert link.find("visual/geometry/mesh/uri").text == mesh_uri
        assert link.findtext("collision/pose") == link.findtext("visual/pose") == "1 1 0 0 0 0"
    doc = json.loads((tmp_path / "collision-primitives.json").read_text())
    assert doc["primitives"] == semantic["collision_primitives"] == [primitive]
    assert doc["non_primitive_collision_meshes"] == semantic["canopy_collision_meshes"] == [receipt]
    # Repeated installation replaces the same link, without duplicating trees.
    assert canopy.install_canopy_obstacles(tmp_path, semantic) == receipt
    for name in canopy.WORLD_FILES:
        matches = ET.parse(tmp_path / name).findall(".//link[@name='observed-canopy-obstacles']")
        assert len(matches) == 1


def test_exact_duplicate_boundary_points_do_not_emit_zero_area_wall_faces():
    import numpy as np

    from dronedream_agent_core.optical_map import _obj_triangles

    points = [
        [-247.2, -118.9],
        [-245, -118.9],
        [-245, -117],
        [-247.2, -117],
        [-247.2, -118.9],
        [-247.2, -118.9],
    ]
    raw, receipt = canopy.canopy_mesh([_feature("duplicates", points)])
    triangles, closed = _obj_triangles(raw, np.ones(3))
    assert closed and len(triangles) == 12
    assert receipt["footprints"][0]["polygon_sha256"] == canopy._digest(points)
    assert _top_footprint(raw).symmetric_difference(Polygon(points)).area < 1e-10


def test_spatial_chunks_keep_full_shells_local_and_exact(monkeypatch):
    import numpy as np

    from dronedream_agent_core.optical_map import _obj_triangles

    monkeypatch.setattr(canopy, "CHUNK_TARGET_TRIANGLES", 24)
    features = [
        _feature(str(i), [[i * 30, 0], [i * 30 + 4, 0], [i * 30 + 4, 4], [i * 30, 4]])
        for i in range(8)
    ]
    meshes, receipt = canopy.canopy_mesh_bundle(features)
    assert len(meshes) == 4
    footprints = []
    ids = []
    for chunk in receipt["meshes"]:
        raw = meshes[chunk["mesh_uri"]]
        triangles, closed = _obj_triangles(raw, np.ones(3))
        assert closed and len(triangles) == 24
        assert np.max(np.abs(triangles[:, :, 0])) == pytest.approx(17)
        origin = chunk["pose_xyz_m"]
        footprints.append(translate(_top_footprint(raw), xoff=origin[0], yoff=origin[1]))
        ids.extend(item["id"] for item in chunk["footprints"])
    assert sorted(ids) == sorted(f["id"] for f in features)
    expected = unary_union([Polygon(f["polygon_xy_m"]) for f in features])
    assert unary_union(footprints).symmetric_difference(expected).area < 1e-9
    assert receipt["source_exclusions_sha256"] == canopy._digest(features)


def test_oversized_shell_does_not_get_cut_or_lose_its_interior():
    points = [[math.cos(i * math.tau / 1600), math.sin(i * math.tau / 1600)] for i in range(1600)]
    with pytest.raises(ValueError, match="complete canopy shell"):
        canopy.canopy_mesh_bundle([_feature("oversized", points)])


def test_actual_observed_canopy_all_chunks_pass_optical_mesh_validation():
    import numpy as np

    from dronedream_agent_core.optical_map import _obj_triangles

    source = (
        Path(__file__).parents[1] / "scripts/data/campus-traversability/canopy-observations.json"
    )
    document = json.loads(source.read_text(encoding="utf-8"))
    features = document["exclusions"]
    meshes, receipt = canopy.canopy_mesh_bundle(features)
    # Leave room in the compiler's 128 resource budget for the existing campus.
    assert receipt["mesh_count"] <= 75
    ids = []
    for chunk in receipt["meshes"]:
        triangles, closed = _obj_triangles(meshes[chunk["mesh_uri"]], np.ones(3))
        assert closed and len(triangles) <= canopy.CHUNK_MAX_TRIANGLES
        assert chunk["vertices"] <= 8192
        ids.extend(item["id"] for item in chunk["footprints"])
    assert sorted(ids) == sorted(f["id"] for f in features)
    assert receipt["source_exclusions_sha256"] == canopy._digest(features)
