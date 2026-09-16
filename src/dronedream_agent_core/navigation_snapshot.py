"""One deployment observation compiler for control and simulation learning."""

import math
from dataclasses import dataclass

from dronedream_plugin_sdk.protocol import encode_json

from .contracts import (
    DynamicObstacleObservation,
    OnboardPerceptionFrame,
    PerceptionFusionHealth,
    Vector3,
)
from .hashing import sha256_json
from .local_world_model import MetricVoxelMap
from .navigation_context import bounded_navigation_context
from .plugin_values import plugin_json_value


@dataclass(frozen=True)
class NavigationSnapshotRequest:
    """Owned compilation inputs with metric ENU geometry and original source time."""
    # The caller must give this request a frozen local world clone and copies
    # of mutable messages. Compilation never reads a live changing world.
    world: MetricVoxelMap
    frame: OnboardPerceptionFrame
    health: PerceptionFusionHealth
    goal_position_m: Vector3
    required_clearance_m: float
    candidate_speed_mps: float
    vehicle_radius_m: float
    vehicle_height_m: float
    visual_evidence: list[dict[str, object]]
    multimodal_sensor_snapshot: dict[str, object] | None
    realtime_feature_snapshot: dict[str, object] | None
    strategic_context: dict[str, object]
    maximum_snapshot_planning_seconds: float
    include_candidate_paths: bool
    control_reference_observed_at_unix_ms: int


# 功能：
#   1. 为连续控制声明前、右、上、偏航四轴及归一化范围，不授予电机控制权限。
#   2. 保留仍被调用的候选选择兼容契约，拒绝未知输出模式。
# 输入：
#   snapshot：需要就地添加控制说明的快照。
#   strategic_context：包含本轮明确输出模式的战略上下文。
# 输出：
#   None：不返回业务数据。
def bind_navigation_control_output_contract(snapshot: dict, strategic_context: dict) -> None:
    if type(snapshot) is not dict or type(strategic_context) is not dict:
        raise ValueError("navigation authority binding requires objects")
    task = strategic_context.get("task")
    if task is not None and type(task) is not dict:
        raise ValueError("navigation task context must be an object")
    output_mode = task.get("local_navigation_output_mode") if isinstance(task, dict) else None
    if output_mode is not None and type(output_mode) is not str:
        raise ValueError("unsupported local navigation output mode")
    if output_mode is None or output_mode == "legacy-candidate-selection":
        return
    if output_mode != "normalized-body-velocity":
        raise ValueError("unsupported local navigation output mode")
    snapshot["model_authority"] = {
        "may_output": {
            "mode": output_mode,
            "axes": ["forward", "right", "up", "yaw"],
            "range": [-1.0, 1.0],
        },
        "may_not_author": [
            "free-space cells",
            "metric coordinates",
            "motor mixing",
            "unbounded attitude or thrust",
        ],
        "downstream_authority": (
            "vehicle-bound physical scaling, acceleration and jerk limiting, "
            "predictive clearance, and PX4 stabilization"
        ),
    }


# 功能：
#   1. 为训练与部署编译同一种独立输入快照，严格检查模式、参数与有限数据并绑定摘要。
#   2. 保留原始来源时间，不因编译或散列刷新观测寿命；控制权限仍由运行期门控决定。
#   3. 只复制本次编译使用的帧字段，不重复复制大地图或未使用的原始深度射线。
# 输入：
#   request：调用方已独占地图视图和冻结消息的编译请求。
# 输出：
#   snapshot：无调用方可变引用、携带原始来源时钟与完整内容摘要的导航状态。
def compile_navigation_snapshot(request: NavigationSnapshotRequest) -> dict[str, object]:
    if not isinstance(request, NavigationSnapshotRequest):
        raise ValueError("navigation snapshot request is invalid")
    if type(request.include_candidate_paths) is not bool:
        raise ValueError("navigation candidate mode requires an explicit boolean")
    reference = request.control_reference_observed_at_unix_ms
    if type(reference) is not int or not 0 <= reference <= 2**63 - 1:
        raise ValueError("navigation control reference clock is invalid")
    for value, allow_zero in (
        (request.required_clearance_m, True), (request.candidate_speed_mps, False),
        (request.vehicle_radius_m, False), (request.vehicle_height_m, False),
        (request.maximum_snapshot_planning_seconds, False),
    ):
        if (type(value) not in (int, float) or not 0 <= value <= 1_000_000
                or not math.isfinite(value) or (not allow_zero and value == 0)):
            raise ValueError("navigation snapshot physical or planning budget is invalid")
    if request.maximum_snapshot_planning_seconds > 60:
        raise ValueError("navigation snapshot planning exceeds the time budget")
    context = bounded_navigation_context(request.strategic_context)
    authority = {}
    bind_navigation_control_output_contract(authority, context)
    if authority and request.include_candidate_paths:
        raise ValueError("continuous navigation cannot include coordinate candidates")
    frame = request.frame
    if not isinstance(frame, OnboardPerceptionFrame) or not isinstance(
        request.health, PerceptionFusionHealth
    ):
        raise ValueError("navigation snapshot requires typed frame and health")
    # 实时协调器用 tuple 冻结动态目标，列表和不可变元组均是既有合法输入。
    if type(frame.dynamic_obstacles) not in (list, tuple) or len(frame.dynamic_obstacles) > 512:
        raise ValueError("navigation snapshot dynamic obstacle budget is invalid")
    geometry = plugin_json_value({
        "position": frame.localization_position_m, "velocity": frame.localization_velocity_mps,
        "goal": request.goal_position_m, "obstacles": frame.dynamic_obstacles,
    })
    position = Vector3.model_validate(geometry["position"], strict=True)
    velocity = Vector3.model_validate(geometry["velocity"], strict=True)
    goal = Vector3.model_validate(geometry["goal"], strict=True)
    obstacles = [DynamicObstacleObservation.model_validate(item, strict=True)
                 for item in geometry["obstacles"]]
    health = PerceptionFusionHealth.model_validate(plugin_json_value(request.health), strict=True)
    if (type(request.visual_evidence) is not list
            or any(type(item) is not dict for item in request.visual_evidence)
            or any(value is not None and type(value) is not dict for value in (
                request.multimodal_sensor_snapshot, request.realtime_feature_snapshot))):
        raise ValueError("navigation snapshot supplementary messages are invalid")
    # 附加快照先冻结，规划结果后追加；只做一次所有权复制，不在最终散列后保留外部引用。
    snapshot = plugin_json_value({
        "multimodal_sensor_snapshot": request.multimodal_sensor_snapshot,
        "realtime_feature_snapshot": request.realtime_feature_snapshot,
        "visual_evidence": request.visual_evidence,
    })
    snapshot = {key: value for key, value in snapshot.items() if value is not None and value != []}
    metric_snapshot = request.world.text_navigation_snapshot(
        current_position_m=position,
        current_velocity_mps=velocity,
        goal_position_m=goal,
        dynamic_obstacles=obstacles,
        required_clearance_m=request.required_clearance_m,
        candidate_speed_mps=request.candidate_speed_mps,
        vehicle_radius_m=request.vehicle_radius_m,
        vehicle_height_m=request.vehicle_height_m,
        maximum_planning_seconds=request.maximum_snapshot_planning_seconds,
        include_candidate_paths=request.include_candidate_paths,
    )
    # 地图编译器生成新容器；补充消息不能覆盖几何或模型权限字段。
    snapshot.update(metric_snapshot)
    snapshot["perception_health"] = health.model_dump(mode="json")
    if context:
        snapshot["strategic_context"] = context
    snapshot.update(authority)
    if "visual_evidence" in snapshot:
        snapshot["visual_input_supplementary"] = True
    snapshot["control_reference_observed_at_unix_ms"] = reference
    snapshot.pop("snapshot_sha256", None)
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    # 最终合并对象也要符合调用协议预算，不能让每段各自合法、拼接后却不可传输。
    encode_json(snapshot)
    return snapshot
