"""Select observation boundaries in a bound plan, not autonomous actuator commands."""

from __future__ import annotations

import math
from typing import Any

from dronedream_agent_core.contracts import (
    FlightPlan,
    MissionContract,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
)
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin, policy_integer, policy_number


# 功能：
#   校验计划合同、唯一航段和共享端点；节点名称相同但位置不同也不能当作连续接点。
# 输入：
#   contract：检查点应绑定的任务合同。
#   flight_plan：准备导出检查点的飞行计划。
# 输出：
#   plan：结构校验且满足共享端点索引前提的计划副本。
def _validated_plan(contract: MissionContract, flight_plan: FlightPlan) -> FlightPlan:
    plan = FlightPlan.model_validate(flight_plan.model_dump(mode="python"))
    if plan.contract_id != contract.contract_id:
        raise ValueError("RUNTIME_CHECKPOINT_CONTRACT_MISMATCH")
    seen: set[str] = set()
    previous = None
    for segment in plan.segments:
        if segment.segment_id in seen:
            raise ValueError("RUNTIME_CHECKPOINT_DUPLICATE_SEGMENT")
        seen.add(segment.segment_id)
        if (
            segment.path[0].node_id != segment.from_node
            or segment.path[-1].node_id != segment.to_node
            or (previous is not None and previous != segment.path[0])
        ):
            raise ValueError("RUNTIME_CHECKPOINT_PATH_DISCONTINUITY")
        previous = segment.path[-1]
    return plan


# 功能：
#   在每个航段终点设置观测检查点，按导出时去除重复端点后的轨迹索引绑定位置。
# 输入：
#   contract：当前任务合同。
#   flight_plan：具有连续共享端点的飞行计划。
#   _：本策略不使用的扩展上下文。
# 输出：
#   checkpoint_contract：包含所有航段终点的检查点合同。
def _segment_checkpoints(
    *, contract: MissionContract, flight_plan: FlightPlan, **_: Any
) -> RuntimeCheckpointContract:
    flight_plan = _validated_plan(contract, flight_plan)
    checkpoints: list[RuntimeCheckpoint] = []
    track_index = 0
    for index, segment in enumerate(flight_plan.segments, start=1):
        track_index += len(segment.path) - 1
        checkpoints.append(
            RuntimeCheckpoint(
                checkpoint_id=f"checkpoint-{index:03d}",
                segment_id=segment.segment_id,
                task_id=segment.task_id,
                track_point_index=track_index,
                target_node=segment.to_node,
            )
        )
    checkpoint_contract = RuntimeCheckpointContract(
        contract_id=contract.contract_id, checkpoints=checkpoints
    )
    return checkpoint_contract


# 功能：
#   保留真正到达任务目标、返程节点及最后航段的检查点，不假设目标一定在第一航段。
# 输入：
#   contract：提供任务目标及返程节点的合同。
#   flight_plan：当前飞行计划。
#   _：本策略不使用的扩展上下文。
# 输出：
#   checkpoint_contract：保留关键任务边界的检查点合同。
def _mission_boundary_checkpoints(
    *, contract: MissionContract, flight_plan: FlightPlan, **_: Any
) -> RuntimeCheckpointContract:
    flight_plan = _validated_plan(contract, flight_plan)
    boundary_nodes = {contract.target_node, contract.return_node}
    checkpoints: list[RuntimeCheckpoint] = []
    track_index = 0
    for segment_index, segment in enumerate(flight_plan.segments):
        track_index += len(segment.path) - 1
        if segment.to_node not in boundary_nodes and segment_index != len(flight_plan.segments) - 1:
            continue
        checkpoints.append(
            RuntimeCheckpoint(
                checkpoint_id=f"checkpoint-{len(checkpoints) + 1:03d}",
                segment_id=segment.segment_id,
                task_id=segment.task_id,
                track_point_index=track_index,
                target_node=segment.to_node,
            )
        )
    checkpoint_contract = RuntimeCheckpointContract(
        contract_id=contract.contract_id, checkpoints=checkpoints
    )
    return checkpoint_contract


# 功能：
#   用归一化三维方向计算转弯角，拒绝非有限几何；重复点不提供有效新方向。
# 输入：
#   first：进入当前转弯前的路径点。
#   middle：当前转弯点。
#   last：转弯后的路径点。
# 输出：
#   angle_degrees：方向改变角度，单位度；任一相邻位移近零时为零。
def _turn_angle_degrees(first: Any, middle: Any, last: Any) -> float:
    incoming = (
        middle.position_m.x - first.position_m.x,
        middle.position_m.y - first.position_m.y,
        middle.position_m.z - first.position_m.z,
    )
    outgoing = (
        last.position_m.x - middle.position_m.x,
        last.position_m.y - middle.position_m.y,
        last.position_m.z - middle.position_m.z,
    )
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if not math.isfinite(incoming_length) or not math.isfinite(outgoing_length):
        raise ValueError("RUNTIME_CHECKPOINT_GEOMETRY_INVALID")
    if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
        angle_degrees = 0.0
        return angle_degrees
    cosine = sum(
        (a / incoming_length) * (b / outgoing_length)
        for a, b in zip(incoming, outgoing, strict=True)
    )
    angle_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return angle_degrees


# 功能：
#   1. 在明显转弯、长航段及狭窄航段中点增加观测边界，同时保留全部任务航段终点。
#   2. 按风险选取有限内部点后恢复轨迹顺序，超过总检查点预算时拒绝而不截掉任务尾部。
# 输入：
#   contract：当前任务合同。
#   flight_plan：待分析的飞行计划。
#   configuration：净空、点数、转弯角阈值及每段内部点预算。
#   _：本策略不使用的扩展上下文。
# 输出：
#   checkpoint_contract：按轨迹顺序绑定的风险自适应检查点合同。
def _risk_adaptive_checkpoints(
    *,
    contract: MissionContract,
    flight_plan: FlightPlan,
    configuration: dict[str, Any] | None = None,
    **_: Any,
) -> RuntimeCheckpointContract:
    flight_plan = _validated_plan(contract, flight_plan)
    configured = {} if configuration is None else configuration
    tight_clearance_m = policy_number(configured, "tight_clearance_m", 1.25, 0.3, 10.0)
    long_segment_points = policy_integer(configured, "long_segment_points", 8, 3, 100)
    minimum_turn_degrees = policy_number(configured, "minimum_turn_degrees", 40.0, 10.0, 170.0)
    maximum_internal = policy_integer(
        configured, "maximum_internal_checkpoints_per_segment", 3, 1, 8
    )
    checkpoints: list[RuntimeCheckpoint] = []
    segment_start_index = 0
    for segment in flight_plan.segments:
        traversed_points = len(segment.path) - 1
        risk_by_index: dict[int, float] = {}
        for local_index in range(1, len(segment.path) - 1):
            angle = _turn_angle_degrees(
                segment.path[local_index - 1],
                segment.path[local_index],
                segment.path[local_index + 1],
            )
            if angle >= minimum_turn_degrees:
                risk_by_index[local_index] = angle
        if (
            traversed_points >= long_segment_points
            or segment.minimum_clearance_m < tight_clearance_m
        ):
            midpoint = max(1, min(traversed_points - 1, traversed_points // 2))
            if midpoint < traversed_points:
                clearance_risk = max(0.0, tight_clearance_m - segment.minimum_clearance_m)
                risk_by_index[midpoint] = max(
                    risk_by_index.get(midpoint, 0.0),
                    minimum_turn_degrees + clearance_risk * 100.0,
                )
        selected_internal = sorted(
            sorted(risk_by_index, key=lambda index: (-risk_by_index[index], index))[
                :maximum_internal
            ]
        )
        checkpoint_indexes = [*selected_internal, traversed_points]
        for local_index in checkpoint_indexes:
            checkpoints.append(
                RuntimeCheckpoint(
                    checkpoint_id=f"checkpoint-{len(checkpoints) + 1:03d}",
                    segment_id=segment.segment_id,
                    task_id=segment.task_id,
                    track_point_index=segment_start_index + local_index,
                    target_node=segment.path[local_index].node_id,
                )
            )
        segment_start_index += traversed_points
    if len(checkpoints) > 128:
        raise ValueError("RUNTIME_RISK_ADAPTIVE_CHECKPOINT_LIMIT_EXCEEDED")
    checkpoint_contract = RuntimeCheckpointContract(
        contract_id=contract.contract_id, checkpoints=checkpoints
    )
    return checkpoint_contract


# 功能：
#   注册互斥的逐航段、任务边界和风险自适应观测策略；观测密度选择不授予飞行权限。
# 输入：
#   无。
# 输出：
#   definitions：只允许在下一任务切换的检查点策略定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "runtime.checkpoint-every-segment",
            "逐航段检查",
            "在每个飞行航段终点请求遥测与模型检查，作为默认闭环策略。",
            _segment_checkpoints,
            True,
            {},
        ),
        (
            "runtime.checkpoint-mission-boundaries",
            "任务边界检查",
            "仅检查到达任务目标和返程终点，适合低风险、低延迟仿真。",
            _mission_boundary_checkpoints,
            False,
            {},
        ),
        (
            "runtime.checkpoint-risk-adaptive",
            "风险自适应检查点",
            "在狭窄净空、长航段和明显转弯处增加内部检查点，同时保留每个任务边界。",
            _risk_adaptive_checkpoints,
            False,
            {
                "type": "object",
                "properties": {
                    "tight_clearance_m": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 10.0,
                        "default": 1.25,
                    },
                    "long_segment_points": {
                        "type": "integer",
                        "minimum": 3,
                        "maximum": 100,
                        "default": 8,
                    },
                    "minimum_turn_degrees": {
                        "type": "number",
                        "minimum": 10.0,
                        "maximum": 170.0,
                        "default": 40.0,
                    },
                    "maximum_internal_checkpoints_per_segment": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 3,
                    },
                },
                "additionalProperties": False,
            },
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.build",
            capability_kind="checkpoint-policy",
            capability_name=name,
            capability_description=description,
            category_id="runtime",
            category_label="运行时与闭环",
            slot_id="runtime.checkpoint-policy",
            slot_label="运行检查点策略",
            activation_mode="single",
            category_order=80,
            slot_order=10,
            plugin_order=index * 10,
            hooks={"build_checkpoints": handler},
            default_enabled=enabled,
            failure_mode="fail-closed",
            swap_policy="next-mission",
            configuration_schema=configuration_schema,
            metadata={
                "differentiation": (
                    "geometry-risk-adaptive"
                    if plugin_id == "runtime.checkpoint-risk-adaptive"
                    else "fixed-density"
                )
            },
        )
        for index, (
            plugin_id,
            name,
            description,
            handler,
            enabled,
            configuration_schema,
        ) in enumerate(values, start=1)
    ]
    return definitions
