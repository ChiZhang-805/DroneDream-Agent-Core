import json

import pytest

from dronedream_agent_core.contracts import (
    CatalogEntity,
    MapAsset,
    MapCatalog,
    MapEdge,
    MapNode,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.navigation_readiness import (
    assess_navigation_readiness,
    enforce_environment_readiness,
)


# 功能：
#   字符串、数字或列表形式的验证声明不能被提升为布尔能力。
# 输入：
#   tmp_path：测试私有目录。
#   claim：注入语义声明中的非布尔值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("claim", ["false", "true", 1, [True]])
def test_non_boolean_metadata_never_grants_runtime_capabilities(tmp_path, claim):
    graph, catalog = _graph_and_catalog()
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({
        "collision_primitives": [{"name": "wall"}],
        "navigation_layers": {"occupancy_ready": claim, "esdf_ready": claim},
        "dynamic_obstacle_tracking": {"runtime_verified": claim},
        "perception_runtime": {key: claim for key in (
            "localization_runtime_verified", "obstacle_perception_runtime_verified",
            "metric_fusion_runtime_verified", "stale_stream_watchdog_verified",
            "dynamic_tracking_runtime_verified")},
        "execution": {key: claim for key in ("simulation_execution_ready",
            "gazebo_runtime_verified", "px4_mission_smoke_verified")},
    }))
    report = assess_navigation_readiness(graph, catalog, semantic,
                                        _vehicle(["stereo-vio", "3d-lidar"]))
    for key, value in report.model_dump().items():
        if key.endswith("_ready") and not key.startswith("static_"):
            assert value is False, key


# 功能：
#   损坏编码或极深嵌套的地图语义不能产生可规划声明。
# 输入：
#   tmp_path：测试私有目录。
#   content：无法安全解析的原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", [b"\xff", b"[" * 2000 + b"]" * 2000],
                         ids=["invalid-utf8", "excessive-nesting"])
def test_corrupt_semantic_metadata_produces_no_readiness(tmp_path, content):
    graph, catalog = _graph_and_catalog()
    path = tmp_path / "semantic.json"
    path.write_bytes(content)
    report = assess_navigation_readiness(graph, catalog, path, _vehicle(["3d-lidar"]))
    assert not report.static_map_planning_ready


# 功能：
#   环境模式拼写错误应被显式拒绝，不能落入无检查分支。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_unknown_environment_mode_cannot_bypass_gate(tmp_path):
    graph, catalog = _graph_and_catalog()
    report = assess_navigation_readiness(graph, catalog, tmp_path / "missing", _vehicle(["imu"]))
    with pytest.raises(ValueError, match="ENVIRONMENT_MODE"):
        enforce_environment_readiness("unknown-indoor-environmnt", report)


# 功能：
#   即使检测与跟踪声明存在，没有度量融合和流过期检查仍不能启用动态导航。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_dynamic_navigation_requires_metric_fusion_and_stale_stream_checks(tmp_path):
    graph, catalog = _graph_and_catalog()
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps({
        "collision_primitives": [{"name": "wall"}],
        "dynamic_obstacle_tracking": {"runtime_verified": True},
        "perception_runtime": {"localization_runtime_verified": True,
            "obstacle_perception_runtime_verified": True,
            "dynamic_tracking_runtime_verified": True},
    }))
    report = assess_navigation_readiness(graph, catalog, path, _vehicle(["stereo-vio", "3d-lidar"]))
    assert report.dynamic_obstacle_tracking_ready
    assert not report.known_dynamic_map_autonomy_ready
    with pytest.raises(ValueError, match="KNOWN_DYNAMIC_MAP_AUTONOMY_NOT_READY"):
        enforce_environment_readiness("known-map-with-dynamic-obstacles", report)


# 功能：
#   构造两节点拓扑与实体目录，只用于逻辑对照，不构成地图接入证明。
# 输入：
#   无。
# 输出：
#   fixture：合成地图及目录组成的元组。
def _graph_and_catalog() -> tuple[MapAsset, MapCatalog]:
    graph = MapAsset(
        asset_id="map",
        name="Map",
        nodes=[
            MapNode(node_id="a", label="A", position_m=Vector3(x=0, y=0, z=1), semantic="launch"),
            MapNode(node_id="b", label="B", position_m=Vector3(x=1, y=0, z=1), semantic="pickup"),
        ],
        edges=[
            MapEdge(
                edge_id="a-b",
                from_node="a",
                to_node="b",
                distance_m=1,
                minimum_clearance_m=1,
                speed_limit_mps=1,
            )
        ],
        named_entities={"a": "a", "b": "b"},
    )
    catalog = MapCatalog(
        scene_id="map",
        semantic_sha256="1" * 64,
        entities=[
            CatalogEntity(
                entity_id="a",
                aliases=["a"],
                position_m=Vector3(x=0, y=0, z=1),
                semantic="launch",
                source_pointer="/a",
            )
        ],
        topology_available=True,
    )
    fixture = graph, catalog
    return fixture


# 功能：
#   建立固定机体配置，仅变化传感器声明名称，不启动真实或仿真传感器。
# 输入：
#   sensors：待检查的传感器名称列表。
# 输出：
#   vehicle：用于能力检查的合成机体。
def _vehicle(sensors: list[str]) -> VehicleAsset:
    vehicle = VehicleAsset(
        asset_id="vehicle",
        name="Vehicle",
        dry_mass_kg=1,
        max_takeoff_mass_kg=2,
        body_radius_m=0.2,
        body_height_m=0.3,
        max_speed_mps=2,
        max_acceleration_mps2=2,
        qualified_range_m=100,
        reserve_battery_percent=20,
        max_pickup_payload_kg=0,
        sensors=sensors,
    )
    return vehicle


# 功能：
#   GPS 与普通里程计不能被误认为已具备任意室内定位和避障能力。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_gps_and_odometry_do_not_claim_arbitrary_indoor_autonomy(tmp_path) -> None:
    graph, catalog = _graph_and_catalog()
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"collision_primitives": [{"name": "wall"}]}))

    report = assess_navigation_readiness(
        graph, catalog, semantic, _vehicle(["imu", "gps", "odometry"])
    )

    assert report.static_map_planning_ready is True
    assert report.indoor_localization_ready is False
    assert report.onboard_obstacle_perception_ready is False
    assert report.arbitrary_indoor_autonomy_ready is False
    with pytest.raises(ValueError, match="ARBITRARY_INDOOR_AUTONOMY_NOT_READY"):
        enforce_environment_readiness("unknown-indoor-environment", report)


# 功能：
#   检查所需声明全部满足时的组合逻辑，不将合成声明当作实际飞行验收。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_explicit_sensor_and_runtime_evidence_enables_dynamic_unknown_mode(tmp_path) -> None:
    graph, catalog = _graph_and_catalog()
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "collision_primitives": [{"name": "wall"}],
                "navigation_layers": {"occupancy_ready": True, "esdf_ready": True},
                "dynamic_obstacle_tracking": {"runtime_verified": True},
                "perception_runtime": {
                    "localization_runtime_verified": True,
                    "obstacle_perception_runtime_verified": True,
                    "metric_fusion_runtime_verified": True,
                    "dynamic_tracking_runtime_verified": True,
                    "stale_stream_watchdog_verified": True,
                },
                "execution": {
                    "simulation_execution_ready": True,
                    "gazebo_runtime_verified": True,
                    "px4_mission_smoke_verified": True,
                },
            }
        )
    )

    report = assess_navigation_readiness(
        graph, catalog, semantic, _vehicle(["imu", "stereo-vio", "3d-lidar"])
    )

    assert report.known_dynamic_map_autonomy_ready is True
    assert report.arbitrary_indoor_autonomy_ready is True
    enforce_environment_readiness("unknown-indoor-environment", report)


# 功能：
#   只有传感器名称而没有对应运行验证时，保留明确的能力缺失原因。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_sensor_names_without_runtime_evidence_do_not_enable_unknown_mode(tmp_path) -> None:
    graph, catalog = _graph_and_catalog()
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "collision_primitives": [{"name": "wall"}],
                "navigation_layers": {"occupancy_ready": True, "esdf_ready": True},
                "dynamic_obstacle_tracking": {"runtime_verified": True},
            }
        )
    )

    report = assess_navigation_readiness(
        graph, catalog, semantic, _vehicle(["stereo-vio", "3d-lidar"])
    )

    assert report.indoor_localization_ready is False
    assert report.onboard_obstacle_perception_ready is False
    assert "INDOOR_LOCALIZATION_RUNTIME_NOT_VERIFIED" in report.issue_codes
    assert "METRIC_PERCEPTION_FUSION_NOT_RUNTIME_VERIFIED" in report.issue_codes


# 功能：
#   识别 OAK-D 深度配置但不据此宣称感知链路已经实际通过运行验证。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_oakd_depth_is_recognized_as_obstacle_sensor_but_still_requires_runtime_evidence(tmp_path):
    graph, catalog = _graph_and_catalog()
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"collision_primitives": [{"name": "wall"}]}))

    report = assess_navigation_readiness(
        graph, catalog, semantic, _vehicle(["imu", "oakd-lite-depth"])
    )

    assert "ONBOARD_OBSTACLE_PERCEPTION_MISSING" not in report.issue_codes
    assert "ONBOARD_OBSTACLE_PERCEPTION_RUNTIME_NOT_VERIFIED" in report.issue_codes
    assert report.onboard_obstacle_perception_ready is False
