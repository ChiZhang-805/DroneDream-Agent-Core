"""Reviewed first-party map resources exposed by the desktop product."""

import json
from copy import deepcopy
from pathlib import Path

_KUMPULA_CAMPUS = {
    "schema_version": "dronedream.map-resource.v1",
    "resource_id": "dronedream-school-map",
    "display_name": {
        "zh-CN": "赫尔辛基大学昆普拉校区",
        "en-US": "Kumpula Campus",
    },
    "aliases": {
        "zh-CN": ["昆普拉校区", "昆普拉科学园区", "赫尔辛基大学昆普拉校区"],
        "en-US": ["Kumpula Campus", "Kumpula Science Campus", "Kumpulan kampus"],
    },
    "description": {
        "zh-CN": "赫尔辛基大学真实校园三维仿真地图，包含建筑、正射影像、碰撞体和经验证航线。",
        "en-US": (
            "A real University of Helsinki campus simulation with buildings, "
            "orthophoto terrain, collision geometry, and a flight-verified route."
        ),
    },
    "category": "real_outdoor_3d_campus",
    "default_resource": True,
    "install_mode": "bundled",
    "review": {
        "status": "approved",
        "reviewed_at": "2026-10-11",
        "source_commit_pinned": False,
        "source_data_hashes_pinned": True,
        "license_reviewed": True,
        "executable_content_required": False,
    },
    "license": {
        "spdx_id": "LicenseRef-CC-BY-4.0-AND-ODbL-1.0",
        "name": "City of Helsinki CC BY 4.0 and OpenStreetMap ODbL 1.0",
        "commercial_use": True,
        "redistribution": True,
        "automated_analysis": True,
        "attribution_required": True,
    },
    "source": {
        "source_type": "bundled_ddpkg",
        "location": "dronedream://default-assets/school-map.ddpkg",
        "expected_sha256": ("9cfa9aec0745160625a76e11bd4b14d73c23400fc5911a5692f35e7dd2ff63e4"),
        "source_format": "dronedream-ddpkg",
        "expected_kind": "map",
        "real_world_crs": "EPSG:3879",
    },
    "analysis": {
        "status": "qualified",
        "parser": "dronedream.kumpula-campus-3d.v1",
        "building_count": 85,
        "collision_primitive_count": 5004,
        "visual_primitive_count": 86,
        "named_buildings": ["Chemicum", "Dynamicum", "Exactum", "Physicum"],
        "named_buildings_zh": [
            "化学楼",
            "气象楼",
            "信息科学楼",
            "物理楼",
        ],
        "coordinate_frame": "ENU",
        "bounds_m": {"x": 562.5, "y": 562.5, "z": 35.0},
        "resolution_m": 0.18,
        "origin": {"latitude": 60.2038, "longitude": 24.9629, "altitude_m": 0},
        "notes": [
            "Building footprints and the orthophoto are real geospatial data.",
            "The bundled route was executed end to end with PX4 and Gazebo.",
        ],
    },
    "readiness": {
        "catalog": "ready",
        "planning": "ready",
        "simulation": "ready",
        "flight": "qualified",
        "qualification_id": "asset-qualification-e6b4e5d8d596851ee549d46a",
        "required_inputs": [],
    },
}


def _generated_resources() -> tuple[dict[str, object], ...]:
    """Load build-generated Helsinki resources shipped beside this module."""

    path = Path(__file__).with_name("generated_map_resources.json")
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dronedream.generated-map-resources.v1":
        raise ValueError("GENERATED_MAP_RESOURCE_SCHEMA_INVALID")
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ValueError("GENERATED_MAP_RESOURCE_LIST_INVALID")
    return tuple(deepcopy(resource) for resource in resources)


_RESOURCES = (_KUMPULA_CAMPUS, *_generated_resources())


def map_resource_catalog() -> dict[str, object]:
    """Return detached catalog data so callers cannot mutate the registry."""

    return {
        "schema_version": "dronedream.map-resource-catalog.v1",
        "catalog_revision": "2026-10-27.1",
        "resources": deepcopy(list(_RESOURCES)),
    }


def get_map_resource(resource_id: str) -> dict[str, object]:
    """Resolve one reviewed built-in map and reject unknown identifiers."""

    for resource in _RESOURCES:
        if resource["resource_id"] == resource_id:
            return deepcopy(resource)
    raise KeyError(resource_id)
