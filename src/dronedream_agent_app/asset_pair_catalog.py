"""Reviewed compatibility matrix for the built-in maps and aircraft.

Compatibility is deliberately weaker than qualification.  This module records
which combinations are physically and operationally plausible, while retaining
the exact qualification identity for the combinations that have completed a
real PX4/Gazebo run.  Callers must never treat ``compatible_requires_qualification``
as permission to execute a mission.
"""

from __future__ import annotations

from copy import deepcopy

from .map_resource_catalog import map_resource_catalog
from .vehicle_resource_catalog import vehicle_resource_catalog

_SCHOOL_MAP = {
    "resource_id": "dronedream-school-map",
    "display_name": {"zh-CN": "学校地图", "en-US": "School Map"},
    "environment_class": "indoor_mixed",
}

_MAP_ENVIRONMENT_CLASSES = {
    "open-rmf-airport-terminal": "indoor_large",
    "open-rmf-battle-royale": "outdoor_open",
    "open-rmf-campus": "outdoor_large",
    "open-rmf-clinic": "indoor_multilevel",
    "open-rmf-hotel": "indoor_multilevel",
    "open-rmf-office": "indoor_tight",
    "open-rmf-triple-h": "indoor_corridor",
}

_OUTDOOR_VTOL_MAPS = frozenset(
    {"open-rmf-battle-royale", "open-rmf-campus"}
)

# These identifiers bind to receipts already shipped in default-assets/index.json.
# The qualified aircraft is the normalized built-in My Drone package, whose
# reviewed source model is PX4 x500_depth.
_BUILTIN_X500_DEPTH_QUALIFICATIONS = {
    "dronedream-school-map": "asset-qualification-9968cd42eaf394ff55ce5792",
    "open-rmf-airport-terminal": "asset-qualification-36d838de8568c92fb3708504",
    "open-rmf-battle-royale": "asset-qualification-e2cab08140791a54cfe24b44",
    "open-rmf-campus": "asset-qualification-c1e60d7740e8e5f64448db27",
    "open-rmf-clinic": "asset-qualification-106f0c4bf24b22858c9e7927",
    "open-rmf-hotel": "asset-qualification-ad8c28b4c381af95b681b0cb",
    "open-rmf-office": "asset-qualification-c4ec6a2c0137760b8f175db2",
    "open-rmf-triple-h": "asset-qualification-babfc35b4cb44740d7c89c1a",
}

# The public resource identifier names the catalog entry, while the normalized
# package identifier names the concrete map asset imported into Agent Core.
# Exposing both prevents clients from guessing package IDs (for example,
# ``open-rmf-office`` is backed by the installed asset ``office``).
_BUILTIN_MAP_ASSET_IDS = {
    "dronedream-school-map": "dronedream.school-map.v1",
    "open-rmf-airport-terminal": "airport-terminal",
    "open-rmf-battle-royale": "battle-royale",
    "open-rmf-campus": "campus",
    "open-rmf-clinic": "clinic",
    "open-rmf-hotel": "hotel",
    "open-rmf-office": "office",
    "open-rmf-triple-h": "triple-h",
}


def _maps() -> list[dict[str, object]]:
    resources = [_SCHOOL_MAP]
    for resource in map_resource_catalog()["resources"]:
        resource_id = str(resource["resource_id"])
        resources.append(
            {
                "resource_id": resource_id,
                "display_name": resource["display_name"],
                "environment_class": _MAP_ENVIRONMENT_CLASSES[resource_id],
            }
        )
    return resources


def _pair_record(
    map_resource: dict[str, object],
    vehicle_resource: dict[str, object],
) -> dict[str, object]:
    map_id = str(map_resource["resource_id"])
    vehicle_id = str(vehicle_resource["resource_id"])
    vehicle_class = str(vehicle_resource["vehicle_class"])
    environment_class = str(map_resource["environment_class"])
    qualification_id = None

    if vehicle_id == "px4-x500-depth":
        qualification_id = _BUILTIN_X500_DEPTH_QUALIFICATIONS[map_id]
        status = "qualified_builtin"
        reasons = [
            "The exact normalized My Drone/x500_depth package completed the "
            "required PX4/Gazebo gates on this map.",
        ]
        required_gates: list[str] = []
    elif vehicle_class == "vtol" and map_id not in _OUTDOOR_VTOL_MAPS:
        status = "incompatible"
        reasons = [
            "The map contains indoor, multilevel, doorway, stair, or corridor "
            "constraints outside this VTOL model's admitted operating envelope.",
        ]
        required_gates = []
    else:
        status = "compatible_requires_qualification"
        if vehicle_class == "vtol":
            reasons = [
                "The map provides an outdoor operating area suitable for VTOL "
                "launch, transition, and recovery planning.",
            ]
        else:
            reasons = [
                "The multicopter geometry can operate in this environment when "
                "its sensor limitations are respected.",
            ]
        required_gates = [
            "gazebo_dependency_resolution",
            "px4_airframe_binding",
            "route_clearance",
            "localization_observability",
            "closed_loop_takeoff_route_return_landing",
        ]

    return {
        "schema_version": "dronedream.asset-pair-compatibility.v1",
        "pair_id": f"{map_id}::{vehicle_id}",
        "map_resource_id": map_id,
        "map_display_name": deepcopy(map_resource["display_name"]),
        "environment_class": environment_class,
        "vehicle_resource_id": vehicle_id,
        "vehicle_display_name": deepcopy(vehicle_resource["display_name"]),
        "vehicle_class": vehicle_class,
        "status": status,
        "compatible": status != "incompatible",
        "qualification_id": qualification_id,
        "qualified_map_asset_id": (
            _BUILTIN_MAP_ASSET_IDS[map_id] if qualification_id is not None else None
        ),
        "qualified_vehicle_asset_id": (
            "dronedream.my-drone.v1" if qualification_id is not None else None
        ),
        "qualification_scope": (
            "exact_builtin_normalized_package" if qualification_id is not None else None
        ),
        "reasons": reasons,
        "required_gates": required_gates,
        "sensor_constraints": deepcopy(
            vehicle_resource["analysis"]["excluded_from"]
        ),
    }


def asset_pair_catalog() -> dict[str, object]:
    """Return the complete 8-map by 13-aircraft compatibility matrix."""
    maps = _maps()
    vehicles = vehicle_resource_catalog()["resources"]
    pairs = [
        _pair_record(map_resource, vehicle_resource)
        for map_resource in maps
        for vehicle_resource in vehicles
    ]
    counts = {
        "qualified_builtin": sum(
            item["status"] == "qualified_builtin" for item in pairs
        ),
        "compatible_requires_qualification": sum(
            item["status"] == "compatible_requires_qualification" for item in pairs
        ),
        "incompatible": sum(item["status"] == "incompatible" for item in pairs),
    }
    return {
        "schema_version": "dronedream.asset-pair-catalog.v1",
        "catalog_revision": "2026-09-28.1",
        "map_count": len(maps),
        "vehicle_count": len(vehicles),
        "pair_count": len(pairs),
        "counts": counts,
        "pairs": deepcopy(pairs),
    }


def get_asset_pair(map_resource_id: str, vehicle_resource_id: str) -> dict[str, object]:
    """Resolve one reviewed pair, rejecting identifiers outside the catalog."""
    pair_id = f"{map_resource_id}::{vehicle_resource_id}"
    for pair in asset_pair_catalog()["pairs"]:
        if pair["pair_id"] == pair_id:
            return pair
    raise KeyError(pair_id)
