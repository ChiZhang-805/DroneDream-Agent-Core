"""Reviewed, immutable-source aircraft resources shown by the desktop product.

Catalog admission means that source ownership, license, format, dependencies,
and intended use were reviewed.  It does not mean that an aircraft is qualified
to fly in every world.  Simulation dependency resolution and aircraft/map pair
qualification remain explicit gates.
"""

from __future__ import annotations

from copy import deepcopy

_PX4_GAZEBO_COMMIT = "5577035667afb4b63fe1f966fb1a58bbb05d905b"
_PX4_GAZEBO_REPOSITORY = "https://github.com/PX4/PX4-gazebo-models.git"
_PX4_GAZEBO_LICENSE = {
    "spdx_id": "BSD-3-Clause",
    "name": "BSD 3-Clause License",
    "url": (
        "https://github.com/PX4/PX4-gazebo-models/blob/"
        f"{_PX4_GAZEBO_COMMIT}/LICENSE"
    ),
    "commercial_use": True,
    "redistribution": True,
    "automated_analysis": True,
}


def _px4_resource(
    model_id: str,
    *,
    zh_name: str,
    en_name: str,
    zh_description: str,
    en_description: str,
    vehicle_class: str,
    recommended_for: list[str],
    excluded_from: list[str],
    dependency_models: list[str],
    sensors: list[str],
    plugin_count: int,
    joint_count: int,
    source_files: int,
    source_bytes: int,
    source_sha256: str,
) -> dict[str, object]:
    """Create a detached-data-ready aircraft entry from reviewed facts."""
    return {
        "schema_version": "dronedream.vehicle-resource.v1",
        "resource_id": f"px4-{model_id.replace('_', '-').casefold()}",
        "display_name": {"zh-CN": zh_name, "en-US": en_name},
        "description": {"zh-CN": zh_description, "en-US": en_description},
        "category": "simulation_aircraft",
        "vehicle_class": vehicle_class,
        "default_resource": True,
        "install_mode": "on_demand",
        "review": {
            "status": "approved",
            "reviewed_at": "2026-09-27",
            "source_commit_pinned": True,
            "license_reviewed": True,
            "executable_content_required": False,
        },
        "license": deepcopy(_PX4_GAZEBO_LICENSE),
        "source": {
            "source_type": "git",
            "location": _PX4_GAZEBO_REPOSITORY,
            "expected_sha256": source_sha256,
            "git_ref": _PX4_GAZEBO_COMMIT,
            "subpath": f"models/{model_id}",
            "source_path_at_commit": f"models/{model_id}/model.sdf",
            "source_format": "gazebo-model-bundle",
            "expected_kind": "vehicle",
            "px4_sitl_model": model_id,
        },
        "analysis": {
            "status": "preparsed",
            "parser": "dronedream.px4-gazebo-sdf-structure.v1",
            "source_commit": _PX4_GAZEBO_COMMIT,
            "sdf_version": "source-declared",
            "source_file_count": source_files,
            "source_size_bytes": source_bytes,
            "dependency_models": dependency_models,
            "resolved_sensor_types": sensors,
            "runtime_plugin_count": plugin_count,
            "joint_count": joint_count,
            "recommended_for": recommended_for,
            "excluded_from": excluded_from,
            "notes": [
                "Nested model dependencies must resolve from the same pinned PX4 source revision.",
                "Runtime plugins stay disabled until the packaged runtime admits "
                "the exact plugin set.",
                "Vehicle parsing and source review do not qualify any aircraft/map "
                "pair for flight.",
            ],
        },
        "readiness": {
            "catalog": "ready",
            "planning": "preparsed",
            "simulation": "dependency_resolution_required",
            "flight": "unqualified",
            "required_inputs": [
                "gazebo_dependency_resolution",
                "px4_airframe_binding",
                "aircraft_map_pair_qualification",
            ],
        },
    }


_RESOURCES = (
    _px4_resource(
        "x500_depth",
        zh_name="X500 深度感知四旋翼",
        en_name="X500 Depth Quadcopter",
        zh_description="带 RGB 与深度相机的四旋翼，优先用于室内避障、门口和楼梯间任务。",
        en_description="Quadcopter with RGB and depth cameras for indoor avoidance and stairs.",
        vehicle_class="multicopter",
        recommended_for=["indoor_navigation", "depth_avoidance", "stairs", "doorways"],
        excluded_from=["long_range_fixed_wing"],
        dependency_models=["x500", "x500_base", "OakD-Lite"],
        sensors=["air_pressure", "camera", "depth_camera", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=6,
        source_files=8,
        source_bytes=165987,
        source_sha256="38ae8ffc3926ffbec18317ecbc36c3bb36409c3819a28f6887e4087e2dc95f07",
    ),
    _px4_resource(
        "x500_flow",
        zh_name="X500 光流定高四旋翼",
        en_name="X500 Optical-Flow Quadcopter",
        zh_description="带下视光流与测距的四旋翼，适合室内定点、低速通过和辅助降落。",
        en_description=(
            "Quadcopter with downward optical flow and ranging for indoor hold and landing."
        ),
        vehicle_class="multicopter",
        recommended_for=["gps_denied_hover", "indoor_position_hold", "precision_landing"],
        excluded_from=["long_range_fixed_wing"],
        dependency_models=["x500", "x500_base", "optical_flow", "LW20"],
        sensors=[
            "air_pressure",
            "camera",
            "custom_optical_flow",
            "gpu_lidar",
            "imu",
            "magnetometer",
            "navsat",
        ],
        plugin_count=5,
        joint_count=7,
        source_files=2,
        source_bytes=2529,
        source_sha256="3af00ba3c5ff3add7031ded1d1fc840b1169e4bba619efd226bf2d8d6bc91071",
    ),
    _px4_resource(
        "x500_lidar_down",
        zh_name="X500 下视激光雷达四旋翼",
        en_name="X500 Downward-Lidar Quadcopter",
        zh_description="带下视激光测距的四旋翼，适合离地高度保持、台阶净空和安全降落。",
        en_description=(
            "Quadcopter with downward lidar for ground clearance, altitude hold, and landing."
        ),
        vehicle_class="multicopter",
        recommended_for=["altitude_hold", "ground_clearance", "precision_landing"],
        excluded_from=["forward_obstacle_avoidance_without_additional_sensor"],
        dependency_models=["x500", "x500_base", "LW20"],
        sensors=["air_pressure", "gpu_lidar", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=6,
        source_files=2,
        source_bytes=2192,
        source_sha256="732a4412e415b5a3aaf326929fd9c56a958fd8aa97d94d3c4b6fe417d6c71602",
    ),
    _px4_resource(
        "x500_lidar_front",
        zh_name="X500 前视激光雷达四旋翼",
        en_name="X500 Forward-Lidar Quadcopter",
        zh_description="带前视激光测距的四旋翼，适合门口、走廊和目标接近阶段的近场避障。",
        en_description=(
            "Quadcopter with forward lidar for doorways, corridors, and approach avoidance."
        ),
        vehicle_class="multicopter",
        recommended_for=["forward_avoidance", "doorways", "corridor_navigation"],
        excluded_from=["full_surround_avoidance_without_additional_sensor"],
        dependency_models=["x500", "x500_base", "LW20"],
        sensors=["air_pressure", "gpu_lidar", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=6,
        source_files=2,
        source_bytes=2209,
        source_sha256="296902046675f8b3dd4f7fa383d9150ec72087e8ad36485be9df35fde4d45dd3",
    ),
    _px4_resource(
        "x500_mono_cam",
        zh_name="X500 前视单目相机四旋翼",
        en_name="X500 Forward Monocular-Camera Quadcopter",
        zh_description="带前视单目相机的四旋翼，适合目标检测、跟踪和视觉路标识别。",
        en_description=(
            "Quadcopter with a forward monocular camera for detection and visual landmarks."
        ),
        vehicle_class="multicopter",
        recommended_for=["visual_detection", "target_tracking", "landmark_navigation"],
        excluded_from=["metric_depth_without_additional_sensor"],
        dependency_models=["x500", "x500_base", "mono_cam"],
        sensors=["air_pressure", "camera", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=5,
        source_files=2,
        source_bytes=789,
        source_sha256="3b93bd613d60cba1c4f11eb04d51968566274988575ee48cb47908a3a1d1bf7c",
    ),
    _px4_resource(
        "x500_mono_cam_down",
        zh_name="X500 下视单目相机四旋翼",
        en_name="X500 Downward Monocular-Camera Quadcopter",
        zh_description="带下视单目相机的四旋翼，适合降落标记、地面纹理和视觉里程计实验。",
        en_description=(
            "Quadcopter with a downward monocular camera for landing markers and odometry."
        ),
        vehicle_class="multicopter",
        recommended_for=["precision_landing", "marker_detection", "visual_odometry"],
        excluded_from=["forward_obstacle_avoidance_without_additional_sensor"],
        dependency_models=["x500", "x500_base", "mono_cam"],
        sensors=["air_pressure", "camera", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=5,
        source_files=2,
        source_bytes=847,
        source_sha256="eb799c76fa98f9d990f55df4eb327fbbc6c8a94e2321704c8547ef2d00609b1f",
    ),
    _px4_resource(
        "x500_lidar_2d",
        zh_name="X500 二维激光雷达四旋翼",
        en_name="X500 2D-Lidar Quadcopter",
        zh_description="带水平扫描激光雷达的四旋翼，适合走廊净空和近场障碍检测。",
        en_description=(
            "Quadcopter with scanning lidar for corridor clearance and near-field obstacles."
        ),
        vehicle_class="multicopter",
        recommended_for=["corridor_navigation", "horizontal_clearance", "near_field_avoidance"],
        excluded_from=["depth_only_vertical_clearance", "long_range_fixed_wing"],
        dependency_models=["x500", "x500_base", "lidar_2d_v2"],
        sensors=["air_pressure", "gpu_lidar", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=6,
        source_files=2,
        source_bytes=833,
        source_sha256="70b8e20e07db9dea58005e08a62dcf98d9cffba5d9f90a644db3a3eeca6758c1",
    ),
    _px4_resource(
        "x500_vision",
        zh_name="X500 视觉里程计四旋翼",
        en_name="X500 Vision Quadcopter",
        zh_description="带视觉里程计接口的四旋翼，用于外部视觉定位与状态融合验证。",
        en_description=(
            "Quadcopter with visual-odometry interface for external-vision state fusion."
        ),
        vehicle_class="multicopter",
        recommended_for=["visual_odometry", "external_vision", "state_fusion"],
        excluded_from=["camera_perception_without_camera_dependency", "long_range_fixed_wing"],
        dependency_models=["x500", "x500_base"],
        sensors=["air_pressure", "imu", "magnetometer", "navsat", "odometry_publisher"],
        plugin_count=6,
        joint_count=4,
        source_files=2,
        source_bytes=629,
        source_sha256="b36b0c5c9eacafd6cf3a35671ce14e17a3fbe059467caee93b59030bb805b4fd",
    ),
    _px4_resource(
        "x500_gimbal",
        zh_name="X500 云台巡检四旋翼",
        en_name="X500 Gimbal Quadcopter",
        zh_description="带三轴云台相机的四旋翼，适合定向观察、巡检和目标持续跟踪。",
        en_description="Quadcopter with a three-axis camera gimbal for inspection and tracking.",
        vehicle_class="multicopter",
        recommended_for=["inspection", "camera_pointing", "target_tracking"],
        excluded_from=["depth_avoidance_without_additional_sensor", "long_range_fixed_wing"],
        dependency_models=["x500", "x500_base", "gimbal"],
        sensors=["air_pressure", "camera", "camera_imu", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=8,
        source_files=2,
        source_bytes=727,
        source_sha256="71102f75a1e1be68fdc16b9a6c38615b8116f192eeff995fd7fe498829f04dd9",
    ),
    _px4_resource(
        "x500",
        zh_name="X500 通用四旋翼",
        en_name="X500 General Quadcopter",
        zh_description="PX4 通用四旋翼基准模型，适合基础控制、悬停和开放场地验证。",
        en_description=(
            "PX4 reference quadcopter for basic control, hover, and open-area validation."
        ),
        vehicle_class="multicopter",
        recommended_for=["basic_control", "hover", "open_area"],
        excluded_from=["indoor_autonomy_without_perception", "long_range_fixed_wing"],
        dependency_models=["x500_base"],
        sensors=["air_pressure", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=4,
        source_files=8,
        source_bytes=169732,
        source_sha256="48b39c6957343b671ea768f459f0f24e14326daaaee85439d6c5bf9fc504c0fd",
    ),
    _px4_resource(
        "standard_vtol",
        zh_name="标准垂直起降固定翼",
        en_name="Standard VTOL",
        zh_description="可垂直起降并转入固定翼巡航，适合大尺度室外长航程任务。",
        en_description="Quadplane for vertical takeoff and efficient long-range outdoor cruise.",
        vehicle_class="vtol",
        recommended_for=["outdoor_long_range", "vertical_takeoff", "site_transit"],
        excluded_from=["indoor_corridors", "stairs", "tight_doorways"],
        dependency_models=[],
        sensors=["air_pressure", "imu", "magnetometer", "navsat"],
        plugin_count=11,
        joint_count=8,
        source_files=7,
        source_bytes=2600460,
        source_sha256="ffbed1169184b0ed94e83171f9597f4d9d079415e496b1b466b8dce18ded1d6a",
    ),
    _px4_resource(
        "tiltrotor",
        zh_name="倾转旋翼飞行器",
        en_name="Tiltrotor",
        zh_description="四发倾转旋翼实验模型，用于室外转换飞行和控制研究。",
        en_description="Quad tiltrotor research model for outdoor transition-flight control.",
        vehicle_class="vtol",
        recommended_for=["transition_flight", "outdoor_research", "site_transit"],
        excluded_from=["indoor_corridors", "stairs", "tight_doorways"],
        dependency_models=["airspeed", "standard_vtol"],
        sensors=["air_pressure", "air_speed", "imu", "magnetometer", "navsat"],
        plugin_count=13,
        joint_count=11,
        source_files=2,
        source_bytes=30056,
        source_sha256="d693a26386b438bc80ecb075af1a83c54eb78b3bb9bc1636b747708cc67294d1",
    ),
    _px4_resource(
        "quadtailsitter",
        zh_name="四旋翼尾座式飞行器",
        en_name="Quad Tailsitter",
        zh_description="尾座式垂直起降模型，用于室外过渡飞行和长距离航线研究。",
        en_description=(
            "Tailsitter VTOL model for outdoor transition and long-distance route research."
        ),
        vehicle_class="vtol",
        recommended_for=["transition_flight", "outdoor_long_range", "vtol_research"],
        excluded_from=["indoor_corridors", "stairs", "tight_doorways"],
        dependency_models=["airspeed"],
        sensors=["air_pressure", "air_speed", "imu", "magnetometer", "navsat"],
        plugin_count=5,
        joint_count=5,
        source_files=5,
        source_bytes=1349448,
        source_sha256="0a72342d9c4701e5d9fb8a3be9cb4d3dd6391978aa1bed049ef2de34f30832a9",
    ),
)


def vehicle_resource_catalog() -> dict[str, object]:
    """Return detached catalog data so request handlers cannot mutate policy."""
    return {
        "schema_version": "dronedream.vehicle-resource-catalog.v1",
        "catalog_revision": "2026-09-28.1",
        "resources": deepcopy(list(_RESOURCES)),
    }


def get_vehicle_resource(resource_id: str) -> dict[str, object]:
    """Resolve only a reviewed identifier; arbitrary sources never enter this path."""
    for resource in _RESOURCES:
        if resource["resource_id"] == resource_id:
            return deepcopy(resource)
    raise KeyError(resource_id)
