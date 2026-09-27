"""Reviewed, content-pinned map resources shown by the desktop product.

The catalog is deliberately separate from flight qualification.  A resource can
be safe to download and useful for planning without yet being converted to a
Gazebo world or proven with a particular aircraft.  Keeping those states
separate prevents a repository license or an LLM interpretation from silently
becoming permission to fly.
"""

from __future__ import annotations

from copy import deepcopy

_OPEN_RMF_COMMIT = "7851a5792d19a037833292a3e2a823b0f9e0c111"
_OPEN_RMF_REPOSITORY = "https://github.com/open-rmf/rmf_demos.git"
_OPEN_RMF_LICENSE = {
    "spdx_id": "Apache-2.0",
    "name": "Apache License 2.0",
    "url": "https://github.com/open-rmf/rmf_demos/blob/7851a5792d19a037833292a3e2a823b0f9e0c111/LICENSE",
    "commercial_use": True,
    "redistribution": True,
    "automated_analysis": True,
}


def _rmf_resource(
    resource_id: str,
    *,
    zh_name: str,
    en_name: str,
    zh_description: str,
    en_description: str,
    levels: list[str],
    doors: int,
    lifts: int,
    models: int,
    lanes: int,
    vertices: int,
    walls: int,
    floors: int,
    source_sha256: str,
) -> dict[str, object]:
    """Build one immutable-source entry with deterministic structural analysis."""
    subpath = f"rmf_demos_maps/maps/{resource_id}/{resource_id}.building.yaml"
    return {
        "schema_version": "dronedream.map-resource.v1",
        "resource_id": f"open-rmf-{resource_id.replace('_', '-').casefold()}",
        "display_name": {"zh-CN": zh_name, "en-US": en_name},
        "description": {"zh-CN": zh_description, "en-US": en_description},
        "category": "indoor_and_site_map",
        "default_resource": True,
        "install_mode": "on_demand",
        "review": {
            "status": "approved",
            "reviewed_at": "2026-09-27",
            "source_commit_pinned": True,
            "license_reviewed": True,
            "executable_content_required": False,
        },
        "license": deepcopy(_OPEN_RMF_LICENSE),
        "source": {
            "source_type": "git",
            "location": _OPEN_RMF_REPOSITORY,
            "expected_sha256": source_sha256,
            "git_ref": _OPEN_RMF_COMMIT,
            "subpath": f"rmf_demos_maps/maps/{resource_id}",
            "source_path_at_commit": subpath,
            "source_format": "rmf-building-map-package",
            "expected_kind": "map",
        },
        "analysis": {
            "status": "preparsed",
            "parser": "dronedream.rmf-building-yaml-structure.v1",
            "source_commit": _OPEN_RMF_COMMIT,
            "level_names": levels,
            "level_count": len(levels),
            "door_count": doors,
            "lift_count": lifts,
            "model_count": models,
            "navigation_lane_count": lanes,
            "vertex_count": vertices,
            "wall_count": walls,
            "floor_polygon_count": floors,
            "notes": [
                "The navigation lanes describe ground-robot traffic and are not UAV corridors.",
                "Doors, lifts, walls and levels are planning evidence, not flight qualification.",
            ],
        },
        "readiness": {
            "catalog": "ready",
            "planning": "preparsed",
            "simulation": "builtin_conversion_available",
            "flight": "unqualified",
            "required_inputs": [
                "gazebo_dependency_resolution",
                "aircraft_pair_qualification",
            ],
        },
    }


_RESOURCES = (
    _rmf_resource(
        "airport_terminal",
        zh_name="机场航站楼",
        en_name="Airport Terminal",
        zh_description="单层大型室内航站楼，包含密集结构、门和长距离通行区域。",
        en_description=(
            "Large single-level terminal with dense structure, doors, and long traversable areas."
        ),
        levels=["L1"],
        doors=5,
        lifts=0,
        models=282,
        lanes=223,
        vertices=1297,
        walls=464,
        floors=75,
        source_sha256="b61d26f189196cd59ef000277bdf445cb04988d16e7cdc96f3e814c02b5f499a",
    ),
    _rmf_resource(
        "clinic",
        zh_name="双层诊所",
        en_name="Two-level Clinic",
        zh_description="双层医疗建筑，包含大量房间、门和四部电梯，适合室内多层规划。",
        en_description="Two-level clinical building with many rooms, doors, and four lifts.",
        levels=["L1", "L2"],
        doors=132,
        lifts=4,
        models=706,
        lanes=169,
        vertices=784,
        walls=619,
        floors=3,
        source_sha256="522cbead55f33a5a44d6882e55bfee7ff4b852344f7880c1927bbfcc09dbda2c",
    ),
    _rmf_resource(
        "hotel",
        zh_name="三层酒店",
        en_name="Three-level Hotel",
        zh_description="三层室内酒店，包含客房走廊、门和电梯等多层结构。",
        en_description="Three-level hotel with room corridors, doors, and lifts.",
        levels=["L1", "L2", "L3"],
        doors=19,
        lifts=2,
        models=175,
        lanes=122,
        vertices=353,
        walls=254,
        floors=13,
        source_sha256="e471a29b78d0d7cbc22b56e4b9b8f8adc126e83c33c73ba94fb7c122debe7dad",
    ),
    _rmf_resource(
        "office",
        zh_name="办公室",
        en_name="Office",
        zh_description="紧凑单层办公室，适合门口、走廊和室内通行验证。",
        en_description="Compact office for doorway, corridor, and indoor traversal validation.",
        levels=["L1"],
        doors=3,
        lifts=0,
        models=79,
        lanes=30,
        vertices=73,
        walls=33,
        floors=4,
        source_sha256="362e6580110e01d3a34b4680809f94dc992b12665ce3c7fba8e12023a91a6a13",
    ),
    _rmf_resource(
        "campus",
        zh_name="校园场地",
        en_name="Campus Site",
        zh_description="大尺度室外校园路线图，可用于室外路线和建筑间过渡规划。",
        en_description="Large outdoor campus route map for site routing and building transitions.",
        levels=["L1"],
        doors=0,
        lifts=0,
        models=1,
        lanes=154,
        vertices=157,
        walls=0,
        floors=0,
        source_sha256="97555099cdf04a15085c8c7dfdac0faf269cbbb7cefcb2f4b4b2ba99b29cfa58",
    ),
    _rmf_resource(
        "battle_royale",
        zh_name="开放测试场",
        en_name="Open Test Arena",
        zh_description="简单开放式布局，用于快速验证导入、规划和坐标链路。",
        en_description="Simple open layout for fast import, planning, and frame-pipeline tests.",
        levels=["L1"],
        doors=0,
        lifts=0,
        models=0,
        lanes=20,
        vertices=53,
        walls=32,
        floors=1,
        source_sha256="ba2436e4c93a334a6650f1158834078cdb150196b0d55e78620238c138c06d9a",
    ),
    _rmf_resource(
        "triple_H",
        zh_name="三联走廊测试场",
        en_name="Triple-H Corridor Test",
        zh_description="走廊型测试布局，用于通道选择和局部路径验证。",
        en_description="Corridor-style test layout for passage selection and local routing.",
        levels=["L1"],
        doors=0,
        lifts=0,
        models=0,
        lanes=22,
        vertices=51,
        walls=28,
        floors=1,
        source_sha256="1213f57cafd9300c652a86b045a82cf3aec448d804004cf16d79ba82ad46eb9c",
    ),
)


def map_resource_catalog() -> dict[str, object]:
    """Return detached catalog data so callers cannot mutate the reviewed registry."""
    return {
        "schema_version": "dronedream.map-resource-catalog.v1",
        "catalog_revision": "2026-09-27.1",
        "resources": deepcopy(list(_RESOURCES)),
    }


def get_map_resource(resource_id: str) -> dict[str, object]:
    """Resolve only a reviewed identifier; arbitrary remote locations never enter this path."""
    for resource in _RESOURCES:
        if resource["resource_id"] == resource_id:
            return deepcopy(resource)
    raise KeyError(resource_id)
