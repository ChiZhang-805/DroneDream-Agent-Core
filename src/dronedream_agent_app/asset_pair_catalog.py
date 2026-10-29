"""Reviewed compatibility matrix for the built-in maps and aircraft.

Compatibility is deliberately weaker than qualification.  This module records
which combinations are physically and operationally plausible, while retaining
the exact qualification identity for the combinations that have completed a
real PX4/Gazebo run.  Callers must never treat ``compatible_requires_qualification``
as permission to execute a mission.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from dronedream_agent_core.asset_packages import MAX_MANIFEST_BYTES
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_plugin_sdk.protocol import decode_json

from .map_resource_catalog import map_resource_catalog
from .vehicle_resource_catalog import vehicle_resource_catalog

# The public resource identifier names the catalog entry, while the normalized
# package identifier names the concrete map asset imported into Agent Core.
# Exposing both prevents clients from guessing package IDs (for example,
# ``open-rmf-office`` is backed by the installed asset ``office``).
_BUILTIN_MAP_ASSET_IDS = {
    str(resource["resource_id"]): str(
        resource.get("analysis", {}).get("asset_id")
        or (
            "dronedream.school-map.v1" if resource["resource_id"] == "dronedream-school-map" else ""
        )
    )
    for resource in map_resource_catalog()["resources"]
}


def _bundled_qualifications(resource_root: Path | None) -> dict[str, str]:
    """Read display identities from this release; actual admission verifies the pair receipt."""
    root = (
        resource_root
        if resource_root is not None
        else (Path(__file__).resolve().parents[2] / "app" / "desktop" / "src-tauri" / "resources")
    )
    try:
        index = decode_json(
            read_plugin_file(root / "default-assets" / "index.json", limit=MAX_MANIFEST_BYTES),
            limit=MAX_MANIFEST_BYTES,
        )
        if not isinstance(index, dict) or index.get("schema_version") not in {
            "dronedream.bundled-assets.v2",
            "dronedream.bundled-assets.v3",
        }:
            return {}
        current = index.get("qualified_pair")
        pairs = (
            index.get("qualified_pairs")
            if index["schema_version"] == "dronedream.bundled-assets.v3"
            else [current]
        )
        if (
            not isinstance(pairs, list)
            or not 1 <= len(pairs) <= 64
            or not isinstance(current, dict)
            or current not in pairs
            or (
                index["schema_version"] == "dronedream.bundled-assets.v3"
                and index.get("default_qualification_id") != current.get("qualification_id")
            )
        ):
            return {}
        qualifications: dict[str, str] = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                return {}
            resource_id = pair.get("resource_id", "dronedream-school-map")
            qualification_id = pair.get("qualification_id")
            packages = pair.get("packages")
            if (
                pair.get("schema_version") != "dronedream.bundled-qualified-pair.v1"
                or resource_id not in _BUILTIN_MAP_ASSET_IDS
                or resource_id in qualifications
                or not isinstance(qualification_id, str)
                or re.fullmatch(r"asset-qualification-[0-9a-f]{24}", qualification_id) is None
                or not isinstance(packages, list)
                or len(packages) != 2
                or any(not isinstance(package, dict) for package in packages)
            ):
                return {}
            identities = {(package.get("kind"), package.get("asset_id")) for package in packages}
            if identities != {
                ("map", _BUILTIN_MAP_ASSET_IDS[resource_id]),
                ("vehicle", "dronedream.my-drone.v1"),
            }:
                return {}
            hashes = [
                pair.get("receipt_sha256"),
                *(
                    package.get(field)
                    for package in packages
                    for field in ("sha256", "content_sha256", "source_content_sha256")
                ),
            ]
            if any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes
            ):
                return {}
            qualifications[resource_id] = qualification_id
        return qualifications
    except (OSError, ValueError, TypeError):
        # A missing or damaged display index cannot resurrect a retired ID.
        return {}


def _maps() -> list[dict[str, object]]:
    """Return every reviewed map admitted to the built-in mission catalog."""

    resources = map_resource_catalog()["resources"]
    return [
        {
            "resource_id": resource["resource_id"],
            "display_name": deepcopy(resource["display_name"]),
            "environment_class": str(resource.get("category") or "real_outdoor_3d"),
        }
        for resource in resources
    ]


def _pair_record(
    map_resource: dict[str, object],
    vehicle_resource: dict[str, object],
    qualifications: dict[str, str],
) -> dict[str, object]:
    map_id = str(map_resource["resource_id"])
    vehicle_id = str(vehicle_resource["resource_id"])
    vehicle_class = str(vehicle_resource["vehicle_class"])
    environment_class = str(map_resource["environment_class"])
    qualification_id = qualifications.get(map_id) if vehicle_id == "px4-x500-depth" else None

    if qualification_id is not None:
        status = "qualified_builtin"
        reasons = [
            "The exact normalized My Drone/x500_depth package completed the "
            "required PX4/Gazebo gates on this map.",
        ]
        required_gates: list[str] = []
    else:
        status = "compatible_requires_qualification"
        if vehicle_class == "vtol":
            reasons = [
                "Outdoor map candidate only: launch, transition, turning, and recovery "
                "clearance must be established for this exact VTOL and map. "
                "A multicopter route does not establish VTOL suitability.",
            ]
        else:
            reasons = [
                "Multicopter candidate only: the exact vehicle envelope, sensing "
                "coverage, and route clearance still require qualification on this map.",
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
        "sensor_constraints": deepcopy(vehicle_resource["analysis"]["excluded_from"]),
    }


def asset_pair_catalog(*, resource_root: Path | None = None) -> dict[str, object]:
    """Return the complete reviewed-map by aircraft compatibility matrix."""
    maps = _maps()
    vehicles = vehicle_resource_catalog()["resources"]
    qualifications = _bundled_qualifications(resource_root)
    pairs = [
        _pair_record(map_resource, vehicle_resource, qualifications)
        for map_resource in maps
        for vehicle_resource in vehicles
    ]
    counts = {
        "qualified_builtin": sum(item["status"] == "qualified_builtin" for item in pairs),
        "compatible_requires_qualification": sum(
            item["status"] == "compatible_requires_qualification" for item in pairs
        ),
        "incompatible": sum(item["status"] == "incompatible" for item in pairs),
    }
    return {
        "schema_version": "dronedream.asset-pair-catalog.v1",
        "catalog_revision": "2026-10-27.1",
        "map_count": len(maps),
        "vehicle_count": len(vehicles),
        "pair_count": len(pairs),
        "counts": counts,
        "pairs": deepcopy(pairs),
    }


def get_asset_pair(
    map_resource_id: str,
    vehicle_resource_id: str,
    *,
    resource_root: Path | None = None,
) -> dict[str, object]:
    """Resolve one reviewed pair, rejecting identifiers outside the catalog."""
    pair_id = f"{map_resource_id}::{vehicle_resource_id}"
    for pair in asset_pair_catalog(resource_root=resource_root)["pairs"]:
        if pair["pair_id"] == pair_id:
            return pair
    raise KeyError(pair_id)
