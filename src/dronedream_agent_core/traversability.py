"""Hard, map-bound 2.5-D operating volumes, shared by planning and validation.

Allowed ground areas/corridors are lifted into genuine volumes. Building and
canopy footprints are excluded at *every* altitude, including above their roofs.
The distance functions are conservative 1-Lipschitz lower bounds so continuous
segment proofs and sampled clearance validation use exactly the same policy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any

XY = tuple[float, float]
Point = tuple[float, float, float]
_ALLOWED_KINDS = {"road", "path", "lawn", "parking", "open-space"}
_EXCLUDED_KINDS = {"building", "tree", "tree-canopy", "canopy"}


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not -1e9 <= value <= 1e9:
        raise ValueError(f"traversability {name} must be finite")
    return float(value)


def _points(value: Any, *, polygon: bool) -> tuple[XY, ...]:
    if not isinstance(value, list) or not (3 if polygon else 2) <= len(value) <= 4096:
        raise ValueError("traversability shape has invalid vertex count")
    points = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("traversability vertices must be XY coordinate pairs")
        points.append((_number(item[0], "x"), _number(item[1], "y")))
    if polygon and points[-1] == points[0]:
        points.pop()
    if len(set(points)) < (3 if polygon else 2):
        raise ValueError("traversability shape is degenerate")
    if polygon:
        area = sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, [*points[1:], points[0]], strict=True))
        if abs(area) < 1e-9:
            raise ValueError("traversability polygon has zero area")
    return tuple(points)


def _segment_distance(point: XY, start: XY, end: XY) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared = dx * dx + dy * dy
    t = (max(0., min(1., ((point[0] - start[0]) * dx
                         + (point[1] - start[1]) * dy) / squared)) if squared else 0.)
    return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


def _polygon_inside_distance(point: XY, polygon: tuple[XY, ...]) -> float:
    """Signed distance: positive inside, negative outside (either winding)."""
    inside = False
    distance = math.inf
    x, y = point
    for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        distance = min(distance, _segment_distance(point, start, end))
        if (start[1] > y) != (end[1] > y):
            crossing = start[0] + (y - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if x < crossing:
                inside = not inside
    return distance if inside else -distance


@dataclass(frozen=True)
class _Feature:
    identifier: str
    points: tuple[XY, ...]
    corridor_radius: float | None = None
    lower_z: float = 0.
    upper_z: float = 0.

    def inside_distance(self, point: XY) -> float:
        if self.corridor_radius is None:
            return _polygon_inside_distance(point, self.points)
        return self.corridor_radius - min(
            _segment_distance(point, a, b)
            for a, b in zip(self.points, self.points[1:], strict=False))

    @cached_property
    def bounding_box(self) -> tuple[XY, XY]:
        radius = self.corridor_radius or 0.
        return (tuple(min(p[i] for p in self.points) - radius for i in range(2)),
                tuple(max(p[i] for p in self.points) + radius for i in range(2)))

    def bounds(self) -> tuple[XY, XY]:
        return self.bounding_box

    def inside_distance_upper_bound(self, point: XY) -> float:
        """A containing rectangle bounds signed distance without replacing the shape."""
        (low_x, low_y), (high_x, high_y) = self.bounding_box
        x, y = point
        dx, dy = max(low_x - x, 0., x - high_x), max(low_y - y, 0., y - high_y)
        distance = (-math.hypot(dx, dy) if dx or dy
                    else min(x - low_x, high_x - x, y - low_y, high_y - y))
        # Keep the broad phase outward rounded, including at equal/touching
        # boundaries. The exact polygon/corridor remains the final authority.
        return distance + 1e-12 * max(1., abs(x), abs(y), abs(low_x), abs(low_y),
                                     abs(high_x), abs(high_y))


class TraversableAirspace:
    """Optional legacy compatibility; a present layer is always strictly enforced.

    A point must fit wholly inside an authored polygon or corridor. Overlapping
    regions form the permitted union; producers should overlap junction regions
    by the aircraft envelope rather than rely on touching polygon boundaries.
    No topology edge or learned proposal can relax this operating domain.
    """

    _BIN_M = 8.

    def __init__(self, semantic: dict[str, Any], primitives: list[dict[str, Any]], *,
                 radius_m: float, half_height_m: float, clearance_cap_m: float = 10.) -> None:
        for name, value in (("radius", radius_m), ("half height", half_height_m),
                            ("clearance cap", clearance_cap_m)):
            if _number(value, name) < 0 or (name == "clearance cap" and value == 0):
                raise ValueError(f"traversability {name} must be nonnegative with positive cap")
        self.active = "traversability" in semantic
        self.radius_m = radius_m
        self.half_height_m = half_height_m
        self.cap = clearance_cap_m
        self.areas: tuple[_Feature, ...] = ()
        self.exclusions: tuple[_Feature, ...] = ()
        self.bounds_m: tuple[Point, Point] | None = None
        self._areas_by_bin: dict[tuple[int, int], list[_Feature]] = defaultdict(list)
        self._exclusions_by_bin: dict[tuple[int, int], list[_Feature]] = defaultdict(list)
        if not self.active:
            return
        layer = semantic["traversability"]
        if (not isinstance(layer, dict)
                or layer.get("schema_version") != "dronedream.traversability.v1"
                or layer.get("required") is not True):
            raise ValueError("traversability layer must declare its supported required schema")
        lower = _number(layer.get("minimum_agl_m"), "minimum AGL")
        upper = _number(layer.get("maximum_agl_m"), "maximum AGL")
        if not 0 <= lower < upper <= 1000:
            raise ValueError("traversability altitude envelope invalid")
        areas: list[_Feature] = []
        exclusions: list[_Feature] = []
        vertex_count = 0
        identifiers: set[str] = set()
        for field in ("areas", "corridors", "exclusions"):
            entries = layer.get(field, [])
            if not isinstance(entries, list) or len(entries) > 10000:
                raise ValueError("traversability feature budget exceeded")
            for item in entries:
                if not isinstance(item, dict):
                    raise ValueError("traversability feature must be an object")
                identifier = item.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise ValueError("traversability feature IDs must be unique nonempty strings")
                identifiers.add(identifier)
                excluded = field == "exclusions"
                if item.get("kind") not in (_EXCLUDED_KINDS if excluded else _ALLOWED_KINDS):
                    raise ValueError("traversability feature kind is unsupported")
                corridor = field == "corridors"
                points = _points(item.get("centerline_xy_m" if corridor else "polygon_xy_m"),
                                 polygon=not corridor)
                vertex_count += len(points)
                if vertex_count > 100000:
                    raise ValueError("traversability vertex budget exceeded")
                width = _number(item.get("width_m"), "corridor width") if corridor else None
                if width is not None and not 0 < width <= 1000:
                    raise ValueError("traversability corridor width invalid")
                ground = 0. if excluded else _number(item.get("ground_z_m"), "ground elevation")
                feature = _Feature(identifier, points, width / 2 if width else None,
                                   ground + lower, ground + upper)
                (exclusions if excluded else areas).append(feature)
        if not areas:
            raise ValueError("traversability layer has no permitted operating areas")
        # Actual collision footprints remain authoritative even when the vector
        # exclusion layer accidentally omits a building or canopy.
        from .collision import _axis_endpoints, _box_half_sizes
        for index, primitive in enumerate(primitives):
            kind = str(primitive.get("semantic", "")).lower().replace("_", "-")
            if (kind not in _EXCLUDED_KINDS and kind != "roof"
                    and not kind.startswith(("building-", "tree-"))):
                continue
            cx, cy = primitive["center_x"], primitive["center_y"]
            identifier = f"collision-footprint:{primitive.get('name', index)}"
            if "size_x" in primitive:
                hx, hy, _ = _box_half_sizes(primitive)
                yaw = primitive.get("yaw_rad", 0.)
                c, s = math.cos(yaw), math.sin(yaw)
                points = tuple((cx + x * c - y * s, cy + x * s + y * c)
                               for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)))
                exclusions.append(_Feature(identifier, points))
            else:
                length = primitive.get("length_m", primitive.get("height_m", 0.))
                first, last = _axis_endpoints(primitive, length)
                exclusions.append(_Feature(identifier, (first[:2], last[:2]),
                                           primitive["radius_m"]))
        self.areas, self.exclusions = tuple(areas), tuple(exclusions)
        lows, highs = zip(*(feature.bounds() for feature in areas), strict=True)
        self.bounds_m = ((min(p[0] for p in lows), min(p[1] for p in lows),
                          min(f.lower_z for f in areas)),
                         (max(p[0] for p in highs), max(p[1] for p in highs),
                          max(f.upper_z for f in areas)))
        indexed = 0
        for features, index in ((areas, self._areas_by_bin),
                                (exclusions, self._exclusions_by_bin)):
            for feature in features:
                low, high = feature.bounds()
                expansion = self.cap + radius_m
                first = [math.floor((v - expansion) / self._BIN_M) for v in low]
                last = [math.floor((v + expansion) / self._BIN_M) for v in high]
                indexed += (last[0] - first[0] + 1) * (last[1] - first[1] + 1)
                if indexed > 2000000:
                    raise ValueError("traversability spatial index budget exceeded")
                for x in range(first[0], last[0] + 1):
                    for y in range(first[1], last[1] + 1):
                        index[x, y].append(feature)

    def clearance(self, point: Point) -> tuple[float, str]:
        if not self.active:
            return math.inf, ""
        key = tuple(math.floor(v / self._BIN_M) for v in point[:2])
        permitted = -self.cap
        for feature in self._areas_by_bin.get(key, ()):
            vertical = min(point[2] - feature.lower_z - self.half_height_m,
                           feature.upper_z - point[2] - self.half_height_m)
            if min(feature.inside_distance_upper_bound(point[:2]) - self.radius_m,
                   vertical) <= permitted:
                continue
            distance = min(feature.inside_distance(point[:2]) - self.radius_m,
                           vertical)
            permitted = max(permitted, distance)
        clearance = min(self.cap, permitted)
        reason = "traversability:permitted-ground-volume"
        for feature in self._exclusions_by_bin.get(key, ()):
            if -feature.inside_distance_upper_bound(point[:2]) - self.radius_m >= clearance:
                continue
            distance = -feature.inside_distance(point[:2]) - self.radius_m
            if distance < clearance:
                clearance, reason = distance, f"traversability:{feature.identifier}"
        return clearance, reason

    def path_clearance(self, points: list[Point], *, sample_interval_m: float = .1
                       ) -> tuple[float, str]:
        """Conservative continuous swept-volume bound, including segment interiors."""
        if not self.active:
            return math.inf, ""
        if not points or len(points) > 100000 or not 0 < sample_interval_m <= 1.:
            raise ValueError("traversability path query invalid")
        if any(len(p) != 3 or any(type(v) not in (int, float) or not -1e9 <= v <= 1e9
                                 for v in p) for p in points):
            raise ValueError("traversability path coordinates invalid")
        minimum, reason = self.clearance(points[0])
        count = 0
        for first, last in zip(points, points[1:], strict=False):
            length = math.dist(first, last)
            steps = max(1, math.ceil(length / sample_interval_m))
            count += steps + 1
            if count > 250000:
                raise ValueError("traversability path query budget exceeded")
            penalty = length / (2 * steps)
            for step in range(steps + 1):
                point = tuple(first[i] + (last[i] - first[i]) * step / steps for i in range(3))
                value, identifier = self.clearance(point)
                if value - penalty < minimum:
                    minimum, reason = value - penalty, identifier
        return minimum, reason
