"""Broad-phase bounds must preserve exact operating-domain clearance and reasons."""
import math
import pickle
import random

import pytest

from dronedream_agent_core.traversability import (
    TraversableAirspace, _Feature, _polygon_inside_distance,
)


def _airspace(radius=.38, height=.215):
    layer = {
        "schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0., "maximum_agl_m": 12.,
        "areas": [
            {"id": "square", "kind": "lawn", "ground_z_m": 0.,
             "polygon_xy_m": [[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]},
            {"id": "concave", "kind": "parking", "ground_z_m": .5,
             "polygon_xy_m": [[8., 0.], [16., 0.], [16., 8.], [14., 8.],
                              [14., 2.], [8., 2.]]},
        ],
        "corridors": [{"id": "bent-road", "kind": "road", "ground_z_m": 0.,
                       "width_m": 3., "centerline_xy_m": [[0., 0.], [0., 18.], [8., 18.]]}],
        "exclusions": [
            {"id": "concave-building", "kind": "building",
             "polygon_xy_m": [[2., -4.], [6., -4.], [6., 4.], [4., 4.],
                              [4., -2.], [2., -2.]]},
            {"id": "tree", "kind": "tree-canopy",
             "polygon_xy_m": [[-5., -1.], [-3., -1.], [-3., 1.], [-5., 1.]]},
        ],
    }
    return TraversableAirspace({"traversability": layer}, [],
                              radius_m=radius, half_height_m=height)


def _unpruned(airspace, point):
    key = tuple(math.floor(v / airspace._BIN_M) for v in point[:2])
    permitted = -airspace.cap
    for feature in airspace._areas_by_bin.get(key, ()):
        permitted = max(permitted, min(
            feature.inside_distance(point[:2]) - airspace.radius_m,
            point[2] - feature.lower_z - airspace.half_height_m,
            feature.upper_z - point[2] - airspace.half_height_m))
    clearance, reason = min(airspace.cap, permitted), "traversability:permitted-ground-volume"
    for feature in airspace._exclusions_by_bin.get(key, ()):
        distance = -feature.inside_distance(point[:2]) - airspace.radius_m
        if distance < clearance:
            clearance, reason = distance, f"traversability:{feature.identifier}"
    return clearance, reason


@pytest.mark.parametrize("radius,height", [(0., 0.), (.38, .215), (1., .5)])
def test_bounds_preserve_clearance_and_threat_at_interior_exterior_and_boundaries(radius, height):
    airspace = _airspace(radius, height)
    rng = random.Random(117)
    points = [(rng.uniform(-22., 28.), rng.uniform(-22., 30.), rng.uniform(-2., 15.))
              for _ in range(2000)]
    points += [(x, y, z) for x in (-10., -5., -3., 0., 2., 4., 6., 10., 14., 16.)
               for y in (-10., -4., -2., -1., 0., 1., 2., 4., 8., 10., 18.)
               for z in (0., height, 6., 12. - height, 12.)]
    for point in points:
        assert airspace.clearance(point) == _unpruned(airspace, point)


def test_continuous_path_penalty_and_threat_match_unpruned_queries(monkeypatch):
    airspace = _airspace()
    rng = random.Random(34)
    paths = [[(rng.uniform(-10., 20.), rng.uniform(-8., 20.), rng.uniform(-1., 13.))
              for _ in range(4)] for _ in range(20)]
    actual = [airspace.path_clearance(path) for path in paths]
    monkeypatch.setattr(airspace, "clearance", lambda point: _unpruned(airspace, point))
    assert [airspace.path_clearance(path) for path in paths] == actual


def test_distant_complex_polygon_in_same_bin_is_not_evaluated(monkeypatch):
    airspace = _airspace()
    points = tuple((4. + math.cos(i * math.tau / 166), 4. + math.sin(i * math.tau / 166))
                   for i in range(166))
    distant = _Feature("complex-building", points)
    airspace._exclusions_by_bin[(0, 0)].append(distant)
    point = (0., 0., 1.)
    expected = _unpruned(airspace, point)
    original = _Feature.inside_distance
    called = []

    def exact_distance(feature, query):
        called.append(feature.identifier)
        return original(feature, query)

    monkeypatch.setattr(_Feature, "inside_distance", exact_distance)
    assert airspace.clearance(point) == expected
    assert "complex-building" not in called


def test_bounds_are_lazy_for_preexisting_pickled_feature_instances():
    feature = _Feature("polygon", ((0., 0.), (2., 0.), (1., 2.)))
    assert "bounding_box" not in feature.__dict__
    restored = pickle.loads(pickle.dumps(feature))
    assert restored.bounds() == ((0., 0.), (2., 2.))
    assert restored.inside_distance_upper_bound((1., 1.)) >= restored.inside_distance((1., 1.))


def test_courtyard_holes_preserve_distances_and_operating_domain():
    outer = ((-10., -10.), (10., -10.), (10., 10.), (-10., 10.))
    hole = ((-3., -3.), (-3., 3.), (3., 3.), (3., -3.))
    building = _Feature("courtyard", outer, holes=(hole,))
    assert building.inside_distance((0., 0.)) == -3.
    assert building.inside_distance((5., 0.)) == 2.
    assert building.inside_distance((12., 0.)) == -2.
    layer = {"schema_version": "dronedream.traversability.v1", "required": True,
             "minimum_agl_m": 0., "maximum_agl_m": 12.,
             "areas": [{"id": "ground", "kind": "open-space", "ground_z_m": 0.,
                        "polygon_xy_m": [list(p) for p in outer]}],
             "exclusions": [{"id": "building", "kind": "building",
                             "polygon_xy_m": [list(p) for p in outer],
                             "holes_xy_m": [[list(p) for p in hole]]}]}
    airspace = TraversableAirspace({"traversability": layer}, [], radius_m=.38, half_height_m=.2)
    assert airspace.clearance((0., 0., 6.))[0] == pytest.approx(2.62)
    assert airspace.clearance((5., 0., 6.))[0] < 0.


def test_exact_clearance_reuse_does_not_quantize_coordinates_or_share_envelopes(monkeypatch):
    airspace = _airspace(0., 0.)
    original = _Feature.inside_distance
    calls = []

    def distance(feature, point):
        calls.append((feature.identifier, point))
        return original(feature, point)

    monkeypatch.setattr(_Feature, "inside_distance", distance)
    outside, inside = (-2.999999, 0., 2.), (-3.000001, 0., 2.)
    result = airspace.clearance(outside)
    count = len(calls)
    for _ in range(30):
        assert airspace.clearance(outside) == result
    assert len(calls) == count
    assert airspace.clearance(inside)[0] < 0. < result[0]
    airspace.radius_m = .5
    assert airspace.clearance(outside) == _unpruned(airspace, outside)
    other = _airspace(0., 0.)
    assert not other._exact_clearances
    assert other.clearance(outside) == result


def test_exact_clearance_reuse_is_bounded_and_eviction_does_not_change_results():
    airspace = _airspace()
    points = [(index / 1000., -8., 3.) for index in range(4100)]
    for point in points:
        assert airspace.clearance(point) == _unpruned(airspace, point)
    assert len(airspace._exact_clearances) == 4096
    assert airspace.clearance(points[0]) == _unpruned(airspace, points[0])
    assert len(airspace._exact_clearances) == 4096


@pytest.mark.parametrize("winding", [1, -1])
def test_detailed_ring_distance_matches_scalar_at_concavities_holes_and_boundaries(winding):
    def ring(count, radius, center=(0., 0.)):
        return tuple((center[0] + (radius + 2 * math.sin(i * math.tau / count * 7)) * math.cos(i * math.tau / count),
                      center[1] + (radius + 2 * math.sin(i * math.tau / count * 7)) * math.sin(i * math.tau / count))
                     for i in range(count))[::winding]

    exterior = ring(513, 30.)
    hole = ring(127, 7.)
    # Source rings can include repeated vertices and axis-aligned edges.
    exterior = (*exterior[:10], exterior[9], *exterior[10:])
    feature = _Feature("detailed-survey", exterior, holes=(hole,))
    rng = random.Random(753)
    queries = [(rng.uniform(-40, 40), rng.uniform(-40, 40)) for _ in range(400)]
    queries += list(exterior) + list(hole)
    queries += [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                for a, b in zip(exterior, (*exterior[1:], exterior[0]))]
    for point in queries:
        expected = min(_polygon_inside_distance(point, exterior),
                       -_polygon_inside_distance(point, hole))
        assert feature.inside_distance(point) == pytest.approx(expected, abs=1e-12)
    restored = pickle.loads(pickle.dumps(feature))
    assert restored.inside_distance((12., 3.)) == feature.inside_distance((12., 3.))
