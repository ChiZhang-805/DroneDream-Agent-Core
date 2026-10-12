"""Reviewed map resources exposed by the desktop product.

Only the bundled, flight-qualified Kumpula Campus map is a default resource.
Import adapters remain available for user-supplied maps, but retired demo maps
must not reappear in the built-in catalog or a release package.
"""

from copy import deepcopy

from dronedream_agent_core.place_index import map_place_context

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
        "expected_sha256": ("c1342f272da972827c8d8a89c2dc1112d06605d3a2d1c9c505708a837b5de27c"),
        "source_format": "dronedream-ddpkg",
        "expected_kind": "map",
        "real_world_crs": "EPSG:3879",
    },
    "analysis": {
        "status": "qualified",
        "parser": "dronedream.kumpula-campus-3d.v1",
        "building_count": 42,
        "collision_primitive_count": 44,
        "visual_primitive_count": 43,
        "named_buildings": ["Chemicum", "Dynamicum", "Exactum", "Physicum"],
        "named_buildings_zh": [
            "化学楼",
            "气象楼",
            "信息科学楼",
            "物理楼",
        ],
        "coordinate_frame": "ENU",
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
        "qualification_id": "asset-qualification-9e8f76549880788ec9e81ef0",
        "required_inputs": [],
    },
}

_KUMPULA_CAMPUS["analysis"]["place_index"] = map_place_context(
    "db46b1f1eae37f1e61df459b1a0da6ce8f9c9d37240a6f138be74cec16ca6529"
)

_RESOURCES = (_KUMPULA_CAMPUS,)


def map_resource_catalog() -> dict[str, object]:
    """Return detached catalog data so callers cannot mutate the registry."""

    return {
        "schema_version": "dronedream.map-resource-catalog.v1",
        "catalog_revision": "2026-10-11.1",
        "resources": deepcopy(list(_RESOURCES)),
    }


def get_map_resource(resource_id: str) -> dict[str, object]:
    """Resolve the sole reviewed built-in map and reject unknown identifiers."""

    for resource in _RESOURCES:
        if resource["resource_id"] == resource_id:
            return deepcopy(resource)
    raise KeyError(resource_id)
