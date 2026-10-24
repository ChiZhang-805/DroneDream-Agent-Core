"""Extrude the exact union of a building's shared dyadic collision cells.

Only the physics representation changes. The source cells remain the planning
and occupancy contract; no visual footprint, simplification or snap is used.
"""

from __future__ import annotations

import math
from collections import Counter

from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def building_collision_mesh(
    cells: list[tuple[float, float, float, float]], *, height_m: float
) -> bytes:
    """Return a closed OBJ in the cells' original XY frame, centered at Z=0.

    The caller places it at the original common center Z and yaw. Building the
    union here, before independently rounding rotated SDF boxes, preserves the
    shared cell boundaries and courtyard holes without inventing microscopic
    gaps or using a snapping tolerance to hide them.
    """
    if not math.isfinite(height_m) or height_m <= 0 or not cells:
        raise ValueError("Building collision mesh needs cells and a positive height")
    if any(
        len(cell) != 4
        or not all(math.isfinite(value) for value in cell)
        or cell[0] >= cell[2]
        or cell[1] >= cell[3]
        for cell in cells
    ):
        raise ValueError("Invalid building collision cell")
    footprint = unary_union([box(*cell) for cell in sorted(cells)])
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Invalid building collision cell union")
    polygons = [footprint] if isinstance(footprint, Polygon) else list(footprint.geoms)
    polygons.sort(key=lambda polygon: (polygon.bounds, polygon.wkb))
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    bottom, top = -height_m / 2, height_m / 2
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        triangles = list(constrained_delaunay_triangles(polygon).geoms)
        cap = unary_union(triangles)
        if not cap.equals(polygon):
            raise ValueError("Building collision triangulation changed cell coverage")
        rings = []
        for triangle in triangles:
            ring = list(orient(triangle, sign=1.0).exterior.coords)[:-1]
            start = min(range(3), key=ring.__getitem__)
            rings.append(ring[start:] + ring[:start])
        rings.sort()
        occurrences = [point for ring in rings for point in ring]
        parents = list(range(len(occurrences)))

        def root(i: int, *, parents=parents) -> int:
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        edges = {}
        for i in range(len(rings)):
            for j in range(3):
                a, b = i * 3 + j, i * 3 + (j + 1) % 3
                key = tuple(sorted((occurrences[a], occurrences[b])))
                edges.setdefault(key, []).append((a, b))
        for adjacent in edges.values():
            if len(adjacent) > 2:
                raise ValueError("Overlapping building collision cap triangles")
            if len(adjacent) == 2:
                (a, b), (c, d) = adjacent
                if occurrences[a] != occurrences[d] or occurrences[b] != occurrences[c]:
                    raise ValueError("Inconsistent building collision cap orientation")
                parents[root(a)] = root(d)
                parents[root(b)] = root(c)
        # Join vertex occurrences only through a shared triangle edge. When a
        # courtyard touches the exterior at a single XY point, its two surface
        # fans need coincident but distinct vertices, not a four-face edge.
        indices = {}
        for i, (x, y) in enumerate(occurrences):
            fan = root(i)
            if fan not in indices:
                indices[fan] = (len(vertices) + 1, len(vertices) + 2)
                vertices.extend(((x, y, bottom), (x, y, top)))
        for i in range(len(rings)):
            paired = [indices[root(i * 3 + j)] for j in range(3)]
            faces.append(tuple(pair[1] for pair in paired))
            faces.append(tuple(pair[0] for pair in reversed(paired)))
        # Derive walls from the same cap tessellation; no independent boundary
        # subdivision can leave a T-junction. Oriented boundary edges put the
        # solid on the left, so these wall triangles point outward to the right.
        for adjacent in edges.values():
            if len(adjacent) == 1:
                first, second = adjacent[0]
                a, d = indices[root(first)]
                b, c = indices[root(second)]
                faces.extend(((a, b, c), (a, c, d)))
    directed_edges = Counter(
        (face[i], face[(i + 1) % 3]) for face in faces for i in range(3)
    )
    if any(count != 1 or directed_edges[b, a] != 1
           for (a, b), count in directed_edges.items()):
        raise ValueError("Building collision mesh is not a closed oriented solid")
    lines = ["# Exact union of retained building collision cells; no visual geometry changes."]
    lines.extend("v " + " ".join(format(v, ".17g") for v in point) for point in vertices)
    for face in faces:
        a, b, c = (vertices[index - 1] for index in face)
        u, v = tuple(b[i] - a[i] for i in range(3)), tuple(c[i] - a[i] for i in range(3))
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                  u[0] * v[1] - u[1] * v[0])
        length = math.hypot(*normal)
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Degenerate building collision triangle")
        # DART/Assimp requires explicit normals for the imported OBJ vertices.
        lines.append("vn " + " ".join(format(n / length, ".17g") for n in normal))
    lines.extend("f " + " ".join(f"{v}//{i}" for v in face)
                 for i, face in enumerate(faces, 1))
    return ("\n".join(lines) + "\n").encode("utf-8")
