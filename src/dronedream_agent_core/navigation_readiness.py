"""Evidence-based capability gates for static, dynamic, and unknown-map navigation."""

from __future__ import annotations

from pathlib import Path

from .contracts import MapAsset, MapCatalog, NavigationReadinessReport, VehicleAsset
from .runtime_control_io import read_runtime_object

LOCALIZATION_SENSORS = frozenset(
    {"vio", "visual-inertial-odometry", "slam", "lidar-slam", "depth-slam", "stereo-vio"}
)
OBSTACLE_SENSORS = frozenset(
    {
        "lidar",
        "3d-lidar",
        "depth-camera",
        "oakd-lite-depth",
        "oak-d-lite-depth",
        "rgbd-camera",
        "stereo-camera",
        "radar",
        "rangefinder-array",
        "point-cloud",
    }
)


# 功能：
#   有界读取普通语义文件，读取失败、重复键或非法数值均视为证据不可用。
# 输入：
#   path：已接入资产的语义文件路径。
# 输出：
#   value：经过结构检查的语义对象；证据不可用时为空字典。
def _semantic(path: Path) -> dict[str, object]:
    try:
        value = read_runtime_object(path, maximum_bytes=16 * 1024 * 1024)
    except (OSError, ValueError, RecursionError):
        value = {}
    return value


# 功能：
#   1. 综合已接入拓扑、语义声明和传感器配置，分别给出静态、动态及未知室内能力状态。
#   2. 声明必须为真实布尔值；此汇总不能替代资产来源验证、碰撞求解或新鲜飞行前检查。
# 输入：
#   graph：已接入的地图拓扑。
#   catalog：同一地图的实体目录。
#   semantic_path：同一地图的语义声明文件。
#   vehicle：所选机体及其配置的传感器。
# 输出：
#   report：各类能力状态、传感器名称和缺失证据原因组成的报告。
def assess_navigation_readiness(graph: MapAsset, catalog: MapCatalog, semantic_path: Path,
                                vehicle: VehicleAsset) -> NavigationReadinessReport:
    if (not isinstance(graph, MapAsset) or not isinstance(catalog, MapCatalog)
            or not isinstance(vehicle, VehicleAsset)
            or type(catalog.topology_available) is not bool
            or not isinstance(vehicle.sensors, list)
            or len(vehicle.sensors) > 128
            or any(type(sensor) is not str or not 0 < len(sensor) <= 160
                   for sensor in vehicle.sensors)):
        raise ValueError("NAVIGATION_READINESS_ASSET_INPUT_INVALID")
    semantic = _semantic(semantic_path)
    collisions = semantic.get("collision_primitives")
    static_collision_ready = isinstance(collisions, list) and bool(collisions)
    sensors = {sensor.strip().lower() for sensor in vehicle.sensors}
    localization_sensor_present = bool(sensors & LOCALIZATION_SENSORS)
    obstacle_sensor_present = bool(sensors & OBSTACLE_SENSORS)

    perception_runtime = semantic.get("perception_runtime")
    if not isinstance(perception_runtime, dict):
        perception_runtime = {}
    indoor_localization = (localization_sensor_present
        and perception_runtime.get("localization_runtime_verified") is True)
    obstacle_perception = (obstacle_sensor_present
        and perception_runtime.get("obstacle_perception_runtime_verified") is True)
    metric_fusion_ready = (perception_runtime.get("metric_fusion_runtime_verified") is True
        and perception_runtime.get("stale_stream_watchdog_verified") is True)

    navigation_layers = semantic.get("navigation_layers")
    if not isinstance(navigation_layers, dict):
        navigation_layers = {}
    occupancy_ready = (
        navigation_layers.get("occupancy_ready") is True
        and navigation_layers.get("esdf_ready") is True
        and metric_fusion_ready
    )

    dynamic_tracking = semantic.get("dynamic_obstacle_tracking")
    if not isinstance(dynamic_tracking, dict):
        dynamic_tracking = {}
    dynamic_tracking_ready = (dynamic_tracking.get("runtime_verified") is True
        and perception_runtime.get("dynamic_tracking_runtime_verified") is True)

    execution = semantic.get("execution")
    if not isinstance(execution, dict):
        execution = {}
    qualified_static_simulation = all(
        execution.get(key) is True
        for key in (
            "simulation_execution_ready",
            "gazebo_runtime_verified",
            "px4_mission_smoke_verified",
        )
    )
    static_planning = catalog.topology_available and bool(graph.edges) and static_collision_ready
    dynamic_ready = (
        static_planning
        and indoor_localization
        and obstacle_perception
        and metric_fusion_ready
        and dynamic_tracking_ready
    )
    arbitrary_ready = dynamic_ready and occupancy_ready
    issues: list[str] = []
    if not static_collision_ready:
        issues.append("STATIC_COLLISION_GEOMETRY_UNAVAILABLE")
    if not occupancy_ready:
        issues.append("OCCUPANCY_ESDF_NOT_RUNTIME_VERIFIED")
    if not localization_sensor_present:
        issues.append("INDOOR_LOCALIZATION_SENSOR_MISSING")
    elif not indoor_localization:
        issues.append("INDOOR_LOCALIZATION_RUNTIME_NOT_VERIFIED")
    if not obstacle_sensor_present:
        issues.append("ONBOARD_OBSTACLE_PERCEPTION_MISSING")
    elif not obstacle_perception:
        issues.append("ONBOARD_OBSTACLE_PERCEPTION_RUNTIME_NOT_VERIFIED")
    if not metric_fusion_ready:
        issues.append("METRIC_PERCEPTION_FUSION_NOT_RUNTIME_VERIFIED")
    if not dynamic_tracking_ready:
        issues.append("DYNAMIC_OBSTACLE_TRACKING_NOT_RUNTIME_VERIFIED")
    if not qualified_static_simulation:
        issues.append("QUALIFIED_STATIC_SIMULATION_NOT_RUNTIME_VERIFIED")

    report = NavigationReadinessReport(
        static_map_planning_ready=static_planning,
        static_collision_geometry_ready=static_collision_ready,
        occupancy_esdf_ready=occupancy_ready,
        indoor_localization_ready=indoor_localization,
        onboard_obstacle_perception_ready=obstacle_perception,
        dynamic_obstacle_tracking_ready=dynamic_tracking_ready,
        qualified_static_simulation_ready=qualified_static_simulation,
        known_dynamic_map_autonomy_ready=dynamic_ready,
        arbitrary_indoor_autonomy_ready=arbitrary_ready,
        sensor_evidence=sorted(sensors),
        issue_codes=issues,
    )
    return report


# 功能：
#   1. 拒绝未知环境模式和缺少依赖能力的动态、未知室内任务。
#   2. 静态任务仍由独立资产、路线及执行前检查约束；此函数不授予实际运动权限。
# 输入：
#   environment_mode：用户任务要求的环境能力范围。
#   report：当前生成的能力汇总报告。
# 输出：
#   None：不返回业务数据。
def enforce_environment_readiness(environment_mode: str, report: NavigationReadinessReport) -> None:
    if environment_mode not in ("qualified-static-map", "known-map-with-dynamic-obstacles",
                                "unknown-indoor-environment"):
        raise ValueError("NAVIGATION_ENVIRONMENT_MODE_INVALID")
    if not isinstance(report, NavigationReadinessReport):
        raise ValueError("NAVIGATION_READINESS_REPORT_INVALID")
    # 副本也可能绕过赋值检查，不能让字符串 false 变成有效布尔声明。
    report = NavigationReadinessReport.model_validate(report.model_dump(), strict=True)
    dynamic_dependencies = (report.static_map_planning_ready
        and report.static_collision_geometry_ready and report.indoor_localization_ready
        and report.onboard_obstacle_perception_ready and report.dynamic_obstacle_tracking_ready
        and report.known_dynamic_map_autonomy_ready)
    if environment_mode == "known-map-with-dynamic-obstacles":
        if not dynamic_dependencies:
            raise ValueError("KNOWN_DYNAMIC_MAP_AUTONOMY_NOT_READY")
    elif (
        environment_mode == "unknown-indoor-environment"
        and not (report.arbitrary_indoor_autonomy_ready
                 and dynamic_dependencies and report.occupancy_esdf_ready)
    ):
        raise ValueError("ARBITRARY_INDOOR_AUTONOMY_NOT_READY")
