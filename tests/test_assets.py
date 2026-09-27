from __future__ import annotations

import json
from pathlib import Path

import pytest

from dronedream_agent_core.assets import (
    UNQUALIFIED_TOPOLOGY_LIMIT,
    AssetQualificationError,
    load_map_catalog,
    map_entity_is_mentioned,
    read_map_semantic_object,
    resolve_map_entity,
)
from dronedream_agent_core.contracts import MapAsset
from dronedream_agent_core.navigation_readiness import _semantic
from dronedream_agent_core.orchestrator import _intent_map_catalog_view
from dronedream_agent_core.runtime_control_io import read_runtime_object
from dronedream_plugin_sdk.protocol import MAX_JSON_NODES


# 功能：
#   将有来源的二维设施作为非飞行语义保留，同时维持办公室和取餐点的真实三维坐标。
# 输入：
#   tmp_path：独立测试语义目录。
# 输出：
#   None：模型可见平面地标，但三维目标目录不凭空生成高度。
def test_source_bound_planar_landmark_is_context_not_flight_target(tmp_path):
    path = tmp_path / "semantic.json"
    _write_semantic(path, include_pickup_binding=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["roads"] = {"facility_anchors": {"cafeteria": [30., 6.245]}}
    raw["entities"].append({"entity_id": "cafeteria", "aliases": ["食堂"],
        "semantic": "landmark", "position_m": [30., 6.245],
        "source_pointer": "/roads/facility_anchors/cafeteria"})
    path.write_text(json.dumps(raw), encoding="utf-8")
    catalog = load_map_catalog(path)
    assert len(catalog.entities) == 3
    assert all(entity.entity_id != "cafeteria" for entity in catalog.entities)
    landmark = catalog.planar_landmarks[0]
    assert (landmark.east_m, landmark.north_m) == (30., 6.245)
    assert landmark.flight_target_available is False
    assert "position_m" not in landmark.model_dump()
    view = _intent_map_catalog_view(catalog)
    assert view["non_actionable_planar_landmarks"][0]["entity_id"] == "cafeteria"
    assert all(entity["entity_id"] != "cafeteria" for entity in view["entities"])


# 功能：
#   拒绝缺失、错绑和非法平面来源，尤其不能将二维起飞点当作可执行三维位置。
# 输入：
#   mutation：需要拒绝的来源变体；tmp_path：测试目录。
# 输出：
#   None：输入在生成地图目录之前失败。
@pytest.mark.parametrize(
    "mutation", ["missing", "mismatch", "bool", "source-bool", "pointer", "launch"]
)
def test_planar_landmark_requires_exact_non_actuating_source(tmp_path, mutation):
    path = tmp_path / "semantic.json"
    _write_semantic(path, include_pickup_binding=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["roads"] = {"facility_anchors": {"facility": [1., 2.]}}
    entity = {"entity_id": "facility", "aliases": ["facility"], "semantic": "landmark",
              "position_m": [1., 2.], "source_pointer": "/roads/facility_anchors/facility"}
    if mutation == "missing":
        raw.pop("roads")
    elif mutation == "mismatch":
        raw["roads"]["facility_anchors"]["facility"] = [1., 3.]
    elif mutation == "bool":
        entity["position_m"] = [True, 2.]
    elif mutation == "source-bool":
        raw["roads"]["facility_anchors"]["facility"] = [True, 2.]
    elif mutation == "pointer":
        entity["source_pointer"] = "/unrelated/facility"
    else:
        entity["semantic"] = "launch"
    raw["entities"].append(entity)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AssetQualificationError, match="ANCHOR_INVALID|PLANAR_SOURCE_INVALID"):
        load_map_catalog(path)


# 功能：
#   验证完整地图允许有界的大型几何，但不扩大实时控制消息的默认结构预算。
# 输入：
#   tmp_path：测试语义目录。
# 输出：
#   None：目录、执行前置和能力分析读取一致，控制消息仍拒绝超量节点。
def test_large_map_uses_same_geometry_budget_across_planning_boundaries(tmp_path):
    path = tmp_path / "semantic.json"
    _write_semantic(path, include_pickup_binding=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["dense_geometry"] = [[1., 2., 3., 4., 5.] for _ in range(MAX_JSON_NODES // 6 + 1)]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert read_map_semantic_object(path) == raw
    assert _semantic(path) == raw
    assert len(load_map_catalog(path).entities) == 3
    with pytest.raises(ValueError, match="COMPLEXITY_LIMIT"):
        read_runtime_object(path, maximum_bytes=16 * 1024 * 1024)


def _write_semantic(path: Path, *, include_pickup_binding: bool) -> None:
    """Create only the selected map's explicit names and three-dimensional anchors."""
    pickup_aliases = ["外卖取餐处"] if include_pickup_binding else ["外卖点", "外卖取餐处"]
    entities: list[dict[str, object]] = [
        {
            "entity_id": "office-launch-pad",
            "aliases": ["办公室起降坪", "三楼办公室起降坪"],
            "position_m": {"x": -42.25, "y": 15.3, "z": 7.95},
            "semantic": "launch",
        },
        {
            "entity_id": "takeout-pickup",
            "aliases": pickup_aliases,
            "position_m": {"x": 48.5, "y": 1.5, "z": 1.5},
            "semantic": "facility-anchor",
        },
    ]
    if include_pickup_binding:
        entities.append(
            {
                "entity_id": "takeout-pickup-pad",
                "aliases": ["外卖点", "外卖取餐点"],
                "position_m": {"x": 48.5, "y": 1.25, "z": 1.35},
                "semantic": "pickup",
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.map-semantic.v1",
                "coordinate_frame": "ENU",
                "scene_id": "school-map",
                "entities": entities,
                "navigation": {"segment_ids": []},
                "runtime_bindings": {
                    "schema_version": "dronedream.map-runtime-bindings.v1",
                    "simulator": "gazebo-harmonic",
                    "coordinate_frame": "ENU",
                    "vehicle_spawn": {"x": -42.25, "y": 15.3, "z": 7.487},
                    "mission_launch_waypoint": {
                        "x": -42.25,
                        "y": 15.3,
                        "z": 7.95,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_retired_school_map_semantics_are_rejected(tmp_path: Path) -> None:
    """A historical schema must not silently enter the current catalog contract."""
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.autonomy.school-map-semantic.v1",
                "coordinate_frame": "ENU",
                "simulation_bindings": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetQualificationError, match="unsupported map semantic schema"):
        load_map_catalog(semantic_path)


def test_pickup_binding_owns_natural_language_pickup_alias(tmp_path: Path) -> None:
    """An explicit action-pad alias belongs to that pad, not its nearby facility anchor."""
    semantic_path = tmp_path / "semantic.json"
    _write_semantic(semantic_path, include_pickup_binding=True)

    catalog = load_map_catalog(semantic_path)
    alias_owners = [entity.entity_id for entity in catalog.entities if "外卖点" in entity.aliases]

    assert alias_owners == ["takeout-pickup-pad"]


def test_launch_binding_includes_natural_office_landing_phrases(tmp_path: Path) -> None:
    """Preserve map-authored language instead of generating it from a hard-coded map."""
    semantic_path = tmp_path / "semantic.json"
    _write_semantic(semantic_path, include_pickup_binding=True)

    catalog = load_map_catalog(semantic_path)
    launch = next(entity for entity in catalog.entities if entity.entity_id == "office-launch-pad")

    assert "办公室起降坪" in launch.aliases
    assert "三楼办公室起降坪" in launch.aliases


def test_facility_anchor_keeps_pickup_alias_without_action_binding(tmp_path: Path) -> None:
    """Do not remove a declared place name just because no separate action pad exists."""
    semantic_path = tmp_path / "semantic.json"
    _write_semantic(semantic_path, include_pickup_binding=False)

    catalog = load_map_catalog(semantic_path)
    alias_owners = [entity.entity_id for entity in catalog.entities if "外卖点" in entity.aliases]

    assert alias_owners == ["takeout-pickup"]


def test_qualified_graph_binding_exposes_topology_without_semantic_invention(
    tmp_path: Path,
) -> None:
    """Graph-only entities use qualified node data while preserving the semantic byte digest."""
    semantic_path = tmp_path / "semantic.json"
    _write_semantic(semantic_path, include_pickup_binding=True)
    graph = MapAsset.model_validate(
        {
            "asset_id": "school-map",
            "name": "Qualified School Map",
            "nodes": [
                {
                    "node_id": "launch",
                    "label": "Launch",
                    "position_m": {"x": -42.25, "y": 15.3, "z": 7.95},
                    "semantic": "launch",
                },
                {
                    "node_id": "pickup",
                    "label": "Pickup",
                    "position_m": {"x": 48.5, "y": 1.25, "z": 1.35},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "launch-pickup",
                    "from_node": "launch",
                    "to_node": "pickup",
                    "distance_m": 92.0,
                    "minimum_clearance_m": 1.5,
                    "speed_limit_mps": 0.55,
                }
            ],
            "named_entities": {
                "office-launch-pad": "launch",
                "secondary-pickup-pad": "pickup",
                "takeout-pickup-pad": "pickup",
            },
        }
    )

    semantic_only = load_map_catalog(semantic_path)
    bound = load_map_catalog(semantic_path, qualified_graph=graph)

    assert semantic_only.topology_available is False
    assert UNQUALIFIED_TOPOLOGY_LIMIT in semantic_only.known_limits
    assert bound.topology_available is True
    assert UNQUALIFIED_TOPOLOGY_LIMIT not in bound.known_limits
    assert bound.semantic_sha256 == semantic_only.semantic_sha256
    secondary_pickup = next(
        entity for entity in bound.entities if entity.entity_id == "secondary-pickup-pad"
    )
    assert secondary_pickup.position_m == graph.nodes[1].position_m
    assert secondary_pickup.semantic == "pickup"
    assert "secondary pickup pad" in secondary_pickup.aliases
    assert secondary_pickup.source_pointer == (
        "qualified-graph/named_entities/secondary-pickup-pad"
    )


def test_generic_map_semantics_are_not_tied_to_school_map(tmp_path: Path) -> None:
    """A separately authored warehouse retains its own identity, names and metre coordinates."""
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.map-semantic.v1",
                "coordinate_frame": "ENU",
                "scene_id": "warehouse-a",
                "entities": [
                    {
                        "entity_id": "dock-a",
                        "aliases": ["loading dock", "装卸区"],
                        "position_m": {"x": 12.5, "y": -4.0, "z": 1.8},
                        "semantic": "pickup",
                    },
                    {
                        "entity_id": "roof-launch",
                        "aliases": ["roof launch pad", "屋顶起飞点"],
                        "position_m": [0.0, 0.0, 8.2],
                        "semantic": "launch",
                    },
                ],
                "navigation": {"segment_ids": ["corridor-a"]},
                "runtime_bindings": {
                    "schema_version": "dronedream.map-runtime-bindings.v1",
                    "simulator": "gazebo-harmonic",
                    "coordinate_frame": "ENU",
                    "vehicle_spawn": {"x": 0.0, "y": 0.0, "z": 7.4},
                    "mission_launch_waypoint": {"x": 0.0, "y": 0.0, "z": 8.2},
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = load_map_catalog(semantic_path)

    assert catalog.scene_id == "warehouse-a"
    assert catalog.road_segment_ids == ["corridor-a"]
    assert {entity.entity_id for entity in catalog.entities} == {"dock-a", "roof-launch"}
    dock = next(entity for entity in catalog.entities if entity.entity_id == "dock-a")
    assert dock.position_m.z == 1.8


def test_entity_preflight_uses_only_the_active_catalog_and_graph(tmp_path: Path) -> None:
    """Resolve active names to graph nodes and reject a name from a retired task."""
    semantic_path = tmp_path / "semantic.json"
    _write_semantic(semantic_path, include_pickup_binding=True)
    graph = MapAsset.model_validate(
        {
            "asset_id": "active-map",
            "name": "Active map",
            "nodes": [
                {
                    "node_id": "launch",
                    "label": "Launch",
                    "position_m": {"x": 0, "y": 0, "z": 2},
                    "semantic": "launch",
                },
                {
                    "node_id": "pickup",
                    "label": "Pickup",
                    "position_m": {"x": 5, "y": 0, "z": 2},
                    "semantic": "pickup",
                },
            ],
            "edges": [
                {
                    "edge_id": "route",
                    "from_node": "launch",
                    "to_node": "pickup",
                    "distance_m": 5,
                    "minimum_clearance_m": 1,
                    "speed_limit_mps": 1,
                }
            ],
            "named_entities": {
                "office-launch-pad": "launch",
                "takeout-pickup-pad": "pickup",
            },
        }
    )
    catalog = load_map_catalog(semantic_path, qualified_graph=graph)

    assert resolve_map_entity("办公室起降坪", catalog, graph) == "launch"
    assert resolve_map_entity("takeout-pickup-pad", catalog, graph) == "pickup"
    assert map_entity_is_mentioned(
        "fly to 外卖取餐点 and return", "takeout-pickup-pad", catalog
    )
    with pytest.raises(AssetQualificationError, match="UNRESOLVED_MAP_ENTITY"):
        resolve_map_entity("retired-pickup-location", catalog, graph)
