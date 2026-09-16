"""Generate test/fault specifications only; no scenario here is executed or marked passed."""

from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.contracts import (
    FlightPlan,
    GraphRoute,
    MissionContract,
    Px4Track,
    RouteClearanceReport,
    TaskGraph,
)
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   生成绑定任务的标称、修改和急停测试矩阵；此处只提出测试要求，不登记验收结果。
# 输入：
#   contract：当前任务合同。
#   task_graph：已分解的任务图。
#   route：待测试路线。
#   clearance：上游提供的净空报告，用于描述场景规模。
#   _：矩阵生成器不使用的扩展参数。
# 输出：
#   campaign：任务规模、可复现种子及必跑场景名称。
def _acceptance_campaign(
    *,
    contract: MissionContract,
    task_graph: TaskGraph,
    route: GraphRoute,
    clearance: RouteClearanceReport,
    **_: Any,
) -> dict[str, object]:
    campaign = {
        "campaign": "acceptance-matrix",
        "contract_id": contract.contract_id,
        "mission_shape": {
            "payload_action": contract.payload_action,
            "task_count": len(task_graph.nodes),
            "route_length_m": route.route_length_m,
            "minimum_clearance_m": clearance.minimum_clearance_m,
        },
        "seeds": [104729, 130363, 155921],
        "required_runs": [
            "nominal",
            "user-amendment-before-takeoff",
            "user-amendment-during-stable-flight",
            "emergency-stop-during-transit",
        ],
    }
    return campaign


# 功能：
#   提出遥测、能源、障碍和证据异常的扩展测试矩阵，实际结果由独立运行收集。
# 输入：
#   contract：当前任务合同。
#   task_graph：被测任务图。
#   route：被测路线。
#   _：矩阵生成器不使用的扩展参数。
# 输出：
#   campaign：压力场景名称、种子及任务规模。
def _stress_campaign(
    *, contract: MissionContract, task_graph: TaskGraph, route: GraphRoute, **_: Any
) -> dict[str, object]:
    campaign = {
        "campaign": "stress-matrix",
        "contract_id": contract.contract_id,
        "seeds": [7919, 104729, 130363, 155921, 196613, 262147, 327673, 393241],
        "required_runs": [
            "nominal",
            "low-battery-at-checkpoint",
            "tracking-error-spike",
            "telemetry-delay",
            "route-obstacle-change",
            "plugin-isolation-failure",
            "runtime-replan-adoption-timeout",
            "completion-evidence-mismatch",
        ],
        "task_count": len(task_graph.nodes),
        "route_length_m": route.route_length_m,
    }
    return campaign


# 功能：
#   描述评估专用的横向阵风及跟踪误差检查要求，不在普通飞行中注入风场。
# 输入：
#   route：被测路线，用于附带长度信息。
#   _：故障描述器不使用的扩展参数。
# 输出：
#   fault：触发比例、风速、持续时间和必查现象的故障规格。
def _wind_fault(*, route: GraphRoute, **_: Any) -> dict[str, object]:
    fault = {
        "fault_id": "wind-gust-cross-track",
        "target": "gazebo.environment.wind",
        "trigger": {"route_fraction": 0.45},
        "parameters": {"speed_mps": 3.0, "duration_s": 4.0, "direction_deg": 90},
        "required_observation": "tracking error remains inside configured runtime gate",
        "route_length_m": route.route_length_m,
    }
    return fault


# 功能：
#   定义遥测时延和抖动测试，要求过期观测不得授权继续；不在此处延迟消息。
# 输入：
#   _：固定时延描述器不使用的扩展参数。
# 输出：
#   fault：检查点触发条件、时延参数及必要观察结果。
def _telemetry_delay_fault(**_: Any) -> dict[str, object]:
    fault = {
        "fault_id": "telemetry-delay",
        "target": "ros.telemetry.bridge",
        "trigger": {"checkpoint_index": 1},
        "parameters": {"latency_ms": 350, "jitter_ms": 80, "duration_s": 5.0},
        "required_observation": "stale telemetry cannot authorize continuation",
    }
    return fault


# 功能：
#   重新验证计划后，将模拟电量骤降绑定到其第一段，避免生成不存在的段触发条件。
# 输入：
#   flight_plan：被测非空飞行计划。
#   _：电量故障描述器不使用的扩展参数。
# 输出：
#   fault：触发段、电量读数及应拒绝继续飞行的检查要求。
def _battery_fault(*, flight_plan: FlightPlan, **_: Any) -> dict[str, object]:
    flight_plan = FlightPlan.model_validate(flight_plan.model_dump(mode="python"))
    fault = {
        "fault_id": "battery-reserve-drop",
        "target": "px4.battery.telemetry",
        "trigger": {"after_segment_id": flight_plan.segments[0].segment_id},
        "parameters": {"reported_percent": 15.0},
        "required_observation": "battery reserve plugin vetoes checkpoint continuation",
    }
    return fault


# 功能：
#   按路线实际距离计算中点的动态障碍规格，拒绝零长路线；不创建真实障碍物。
# 输入：
#   px4_track：含世界 ENU 源点与执行器点位的轨迹。
#   _：障碍描述器不使用的扩展参数。
# 输出：
#   fault：世界 ENU 障碍位置、触发比例及重新验证净空的要求。
def _obstacle_fault(*, px4_track: Px4Track, **_: Any) -> dict[str, object]:
    track = Px4Track.model_validate(px4_track.model_dump(mode="python"))
    points = [(point.east_m, point.north_m, point.up_m) for point in track.source_world_points]
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:], strict=False)]
    total = sum(lengths)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("SIMULATION_OBSTACLE_FAULT_ROUTE_INVALID")
    remaining = total / 2.0
    midpoint = points[-1]
    # 不均匀采样时，索引中点不等于距离中点；跨过零长段并在目标段内部插值。
    for first, last, length in zip(points, points[1:], lengths, strict=False):
        if length > 0 and remaining <= length:
            fraction = remaining / length
            midpoint = tuple(a + (b - a) * fraction for a, b in zip(first, last, strict=True))
            break
        remaining -= length
    fault = {
        "fault_id": "dynamic-obstacle-route-change",
        "target": "gazebo.dynamic_obstacle",
        "trigger": {"route_fraction": 0.5},
        "parameters": {
            "east_m": midpoint[0],
            "north_m": midpoint[1],
            "up_m": midpoint[2],
            "radius_m": 0.6,
        },
        "required_observation": "old route freezes and replacement route requires new clearance",
    }
    return fault


# 功能：
#   注册测试矩阵与故障描述库；钩子启用仅使规格可查询，不启动后台测试或注入故障。
# 输入：
#   无。
# 输出：
#   definitions：场景生成及故障定义插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    generators = [
        (
            "simulation.campaign-acceptance",
            "闭环验收矩阵",
            "为任务生成标称、计划修改、运行修改和紧急停止测试矩阵。",
            _acceptance_campaign,
            True,
        ),
        (
            "simulation.campaign-stress",
            "鲁棒性压力矩阵",
            "生成遥测、能源、障碍、插件和证据故障的扩展压力矩阵。",
            _stress_campaign,
            False,
        ),
    ]
    for index, (plugin_id, name, description, handler, enabled) in enumerate(generators, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.generate",
                capability_kind="scenario-generator",
                capability_name=name,
                capability_description=description,
                category_id="simulation",
                category_label="仿真与测试",
                slot_id="simulation.campaign-generator",
                slot_label="仿真测试矩阵",
                activation_mode="single",
                category_order=80,
                slot_order=20,
                plugin_order=index * 10,
                hooks={"generate_campaign": handler},
                default_enabled=enabled,
                failure_mode="isolate",
            )
        )
    faults = [
        (
            "simulation.fault-wind",
            "横向阵风故障",
            "定义 Gazebo 风场阵风及跟踪误差验收要求。",
            _wind_fault,
        ),
        (
            "simulation.fault-telemetry-delay",
            "遥测时延故障",
            "定义 ROS 遥测时延和抖动场景。",
            _telemetry_delay_fault,
        ),
        (
            "simulation.fault-battery",
            "电量余量故障",
            "定义运行检查点电量跌破余量门的场景。",
            _battery_fault,
        ),
        (
            "simulation.fault-dynamic-obstacle",
            "动态障碍故障",
            "在路线中点定义动态障碍并要求重新净空验证。",
            _obstacle_fault,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(faults, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.describe",
                capability_kind="fault-injector",
                capability_name=name,
                capability_description=description,
                category_id="simulation",
                category_label="仿真与测试",
                slot_id="simulation.fault-library",
                slot_label="故障场景定义",
                activation_mode="multiple",
                category_order=80,
                slot_order=30,
                plugin_order=index * 10,
                hooks={"describe_fault": handler},
                default_enabled=True,
                failure_mode="advisory",
            )
        )
    return definitions
