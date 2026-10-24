"""Physical collision aggregation must preserve the source solid, including holes."""

import importlib.util
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "ground_map_collision_mesh", Path(__file__).parents[1] / "scripts/ground_map_collision_mesh.py"
)
mesh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mesh)


def _assert_exact_solid(raw, cells, height):
    vertices, normals, faces = [], [], []
    for line in raw.decode().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("vn "):
            normals.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("f "):
            tokens = [token.split("//") for token in line.split()[1:]]
            assert len(tokens) == 3
            assert {int(t[1]) for t in tokens} == {len(faces) + 1}
            faces.append(tuple(int(t[0]) - 1 for t in tokens))
    vertices = np.asarray(vertices)
    edges, caps, volume = Counter(), [], 0.0
    source = unary_union([box(*cell) for cell in cells])
    for i, face in enumerate(faces):
        a, b, c = vertices[list(face)]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        assert normals[i] == pytest.approx(normal, abs=1e-12)
        for j in range(3):
            edges[face[j], face[(j + 1) % 3]] += 1
        volume += np.dot(a, np.cross(b, c)) / 6
        if a[2] == b[2] == c[2] == height / 2:
            assert normal[2] == 1
            caps.append(Polygon([a[:2], b[:2], c[:2]]))
        elif normal[2] == 0:
            # At every wall (including hole walls), the solid lies inward.
            center = (a + b + c) / 3
            from shapely.geometry import Point

            assert source.covers(Point(*(center[:2] - normal[:2] * 1e-6)))
            assert not source.covers(Point(*(center[:2] + normal[:2] * 1e-6)))
    assert all(count == edges[b, a] == 1 for (a, b), count in edges.items())
    assert unary_union(caps).equals(source)
    assert volume == pytest.approx(source.area * height, rel=1e-12, abs=1e-10)
    assert vertices[:, 2].min() == -height / 2
    assert vertices[:, 2].max() == height / 2


@pytest.mark.parametrize("cells", [
    [(0, 0, 2, 2), (2, 0, 4, 2), (1, 0, 3, 2)],  # Overlap and shared edges.
    [(0, 0, 6, 2), (0, 4, 6, 6), (0, 2, 2, 4), (4, 2, 6, 4)],  # Courtyard.
    [(0, 0, 2, 2), (0, 2, 1, 4), (8, 7, 10, 9)],  # Concavity + separate shell.
    [(-2, -2, 0, 0), (0, 0, 2, 2)],  # Only touching at a point.
    # Real building33270 has this topology: a courtyard touches the outer
    # boundary at a retained cell corner. Distinct surface fans must not share
    # a four-face vertical mesh edge, nor may the opening be filled/snapped.
    [(x, y, x + 1, y + 1) for x in range(3) for y in range(3)
     if (x, y) not in {(0, 0), (1, 1)}],
])
def test_cells_become_the_same_closed_oriented_solid(cells):
    raw = mesh.building_collision_mesh(cells, height_m=7.3)
    _assert_exact_solid(raw, cells, 7.3)
    assert raw == mesh.building_collision_mesh(list(reversed(cells)), height_m=7.3)


def test_no_snapping_closes_a_real_small_gap():
    cells = [(0, 0, 1, 1), (1 + 1e-9, 0, 2, 1)]
    raw = mesh.building_collision_mesh(cells, height_m=2)
    # The complete cap still has two components; tolerances must not connect it.
    vertices, top = [], []
    for line in raw.decode().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("f "):
            points = [vertices[int(t.split("//")[0]) - 1] for t in line.split()[1:]]
            if all(p[2] == 1 for p in points):
                top.append(Polygon([p[:2] for p in points]))
    cap = unary_union(top)
    assert len(cap.geoms) == 2
    assert cap.equals(unary_union([box(*cell) for cell in cells]))


@pytest.mark.parametrize("cells,height", [([], 2), ([(0, 0, 0, 1)], 2),
                                          ([(0, 0, math.inf, 1)], 2),
                                          ([(0, 0, 1, 1)], 0)])
def test_invalid_solids_cannot_be_published(cells, height):
    with pytest.raises(ValueError):
        mesh.building_collision_mesh(cells, height_m=height)
