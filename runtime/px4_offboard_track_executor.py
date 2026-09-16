#!/usr/bin/env python3
"""PX4 offboard trajectory executor for DroneDream real PX4/Gazebo runs.

This script is intended to run as a subprocess from local_px4_launch_wrapper.py.
It reads DroneDream reference and controller JSON files, builds an offboard
position setpoint schedule, and streams PositionNedYaw setpoints.

This standalone entry point is the explicit reference-track qualification mode,
not the model-controlled mission loop. The checkpoint executor reuses the real
MAVSDK client below, including its velocity-only transport for local policies.

Coordinate contract:
- Exported reference x/y/z means north/east/up, not raw map ENU coordinates.
- PX4 offboard local frame uses NED north/east/down.
- Mapping: north=x, east=y, down=-z.
- Reference-track setpoints are displacements from the qualified model spawn.
  They are rebased onto the estimator's measured local-NED origin before any
  arm, takeoff, hold, or track command is sent.

The executor applies only ``vel_limit`` and ``accel_limit`` to its setpoint
schedule. Selected PX4 parameters are applied and verified separately by the
launch wrapper. No other controller field is active in this process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar

MAX_REFERENCE_TRACK_POINTS = 10_000
MAX_SETPOINT_RATE_HZ = 100.0
MAX_TAKEOFF_CLIMB_RATE_M_S = 5.0
MAX_SETPOINTS = 1_000_000
MAX_INPUT_JSON_BYTES = 16 * 1024 * 1024
DEFAULT_HOVER_DURATION_SECONDS = 10.0
MIN_HOVER_DURATION_SECONDS = 1.0
MAX_HOVER_DURATION_SECONDS = 300.0
MAX_WAYPOINT_HOLD_SECONDS = 30.0
MAX_ABORT_REQUEST_BYTES = 4096
CLEANUP_COMMAND_TIMEOUT_SECONDS = 5.0
ABORT_POLL_INTERVAL_SECONDS = 0.05
MAVSDK_PREFLIGHT_CONNECT_TIMEOUT_SECONDS = 10.0
MAVSDK_PREFLIGHT_MAX_ATTEMPTS = 3
DYNAMICS_STREAM_SAMPLE_TIMEOUT_SECONDS = {
    "position_velocity": 3.0,
    "imu": 3.0,
    "attitude": 3.0,
    # Battery is a slow energy signal.  It remains recorded and monitored, but
    # must not churn its MAVSDK subscription or revoke millisecond-scale payload
    # stabilization authority merely because it updates more slowly than IMU,
    # attitude, and actuator output.
    "battery": 10.0,
    "actuator_output": 3.0,
    "odometry": 3.0,
}
DYNAMICS_TELEMETRY_REQUESTED_RATES_HZ = {
    "position_velocity": 50.0,
    "imu": 50.0,
    "attitude": 50.0,
    "battery": 2.0,
    "actuator_output": 10.0,
    "odometry": 50.0,
}
ACTUATOR_OUTPUT_ABSOLUTE_MAXIMUM_ENV = "PX4_ACTUATOR_OUTPUT_ABSOLUTE_MAXIMUM"
PX4_OFFBOARD_LOSS_TIMEOUT_SECONDS = 1.0
PX4_OFFBOARD_LOSS_HOLD_ACTION = 5
BATTERY_TRACK_START_SETTLE_TIMEOUT_SECONDS = 15.0
GAZEBO_PAYLOAD_STATE_TRANSITION_TIMEOUT_SECONDS = 15.0
GAZEBO_PAYLOAD_STATE_DISCOVERY_SETTLE_SECONDS = 0.5
GAZEBO_PAYLOAD_COMMAND_RETRY_SECONDS = 0.1
RUNTIME_EFFECT_SCHEMA_VERSION = "dronedream.scenario_runtime_effects.v1"

_T = TypeVar("_T")


class TelemetrySubscriptionCloseFailed(RuntimeError):
    """The previous subscription did not close; creating a replacement is forbidden."""


# 功能：
#   解析一条完整的挂载状态回执，拒绝重复、冲突或夹杂其他文本的状态。
# 输入：
#   raw_state：Gazebo Boolean 或 StringMsg 的文本回执。
# 输出：
#   detached：True 表示已分离，False 表示已连接。
def _parse_gazebo_payload_detached_state(raw_state: str) -> bool:
    if not isinstance(raw_state, str) or len(raw_state) > 256:
        raise RuntimeError("Gazebo payload state readback is invalid")
    match = re.fullmatch(
        r'\s*data\s*:\s*(true|false|"attached"|"detached"|\'attached\'|\'detached\')\s*',
        raw_state,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError("Gazebo payload state readback contained no recognized state")
    detached = match.group(1).strip("\"'").casefold() in {"true", "detached"}
    return detached


@dataclass(frozen=True)
class GazeboModelPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    # 功能：
    #   验证位置及单位四元数，防止非有限数、零旋转或溢出的归一化结果进入挂载计算。
    # 输入：
    #   self：待建立的 Gazebo 世界位姿。
    # 输出：
    #   None：校验失败抛出异常，不自动修正坏观测。
    def __post_init__(self) -> None:
        for value in (self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw):
            if (
                type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max
            ):
                raise ValueError("Gazebo model pose requires finite numeric components")
        if not math.isclose(math.hypot(self.qx, self.qy, self.qz, self.qw), 1.0, abs_tol=1e-6):
            raise ValueError("Gazebo model pose requires a unit quaternion")


# 功能：
#   校验遥测和控制边界的原始有限数字，不把文本或布尔值转换成有效测量。
# 输入：
#   value：原始数值字段。
#   label：错误消息中的字段名。
# 输出：
#   number：有限浮点数。
def _native_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        raise ValueError(f"{label} must be a finite native number")
    number = float(value)
    return number


# 功能：
#   校验通信及遥测等待预算，禁止非有限值导致无限等待。
# 输入：
#   timeout_seconds：调用方的秒数预算。
# 输出：
#   timeout：范围为零至一小时之间、不含零的等待秒数。
def _timeout_budget(timeout_seconds: float) -> float:
    timeout = _native_number(timeout_seconds, "timeout")
    if not 0 < timeout <= 3600:
        raise ValueError("timeout must be in (0, 3600]")
    return timeout


# 功能：
#   限制传输主题为明确的绝对 Gazebo 路径，禁止空值、空白及控制字符进入命令参数。
# 输入：
#   topic：计划绑定的命令或观察主题。
# 输出：
#   None：合法时保持原主题，不自动修正或拼接另一主题。
def _validate_gazebo_topic(topic: str) -> None:
    if (
        type(topic) is not str
        or len(topic) > 1024
        or re.fullmatch(r"/[A-Za-z0-9_.~/-]+", topic) is None
    ):
        raise ValueError("Gazebo topic must be an explicit absolute topic")


# 功能：
#   校验具名世界或模型，避免把空身份及非法名字拼入原生传输请求。
# 输入：
#   name：当前绑定的实体名，允许 Gazebo 作用域分隔符。
# 输出：
#   None：合法时保留原名。
def _validate_gazebo_entity(name: str) -> None:
    if type(name) is not str or len(name) > 512 or re.fullmatch(r"[A-Za-z0-9_.:-]+", name) is None:
        raise ValueError("Gazebo entity name is invalid")


# 功能：
#   校验 PX4 标量参数的精确名称，避免无效字段进入实际参数写入路径。
# 输入：
#   name：调用方已授权的参数名。
# 输出：
#   None：只验证名字格式，不扩大调用方的修改权限。
def _validate_parameter_name(name: str) -> None:
    if type(name) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", name) is None:
        raise ValueError("PX4 parameter name is invalid")


# 功能：
#   有界收集一条 Gazebo 主题消息，异步取消传递到子进程并等待其资源回收。
# 输入：
#   topic：已绑定的绝对主题。
#   timeout_seconds：通信总秒数预算，清理另有共享进程收集器的固定上限。
# 输出：
#   text：成功退出后得到的 UTF-8 标准输出，错误输出不冒充有效消息。
async def _capture_gazebo_topic(topic: str, timeout_seconds: float) -> str:
    from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
    from dronedream_agent_core.process_capture import capture_process

    _validate_gazebo_topic(topic)
    timeout = _timeout_budget(timeout_seconds)
    gz = shutil.which("gz")
    if gz is None:
        raise RuntimeError("Gazebo transport executable 'gz' is unavailable")
    cancel_event = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            capture_process,
            [gz, "topic", "-e", "-t", topic, "-n", "1"],
            stdin=b"",
            maximum_bytes=1_048_576,
            timeout=timeout,
            environment=dict(os.environ),
            resource_policy=PluginResourcePolicy(),
            cancel_event=cancel_event,
        )
    )
    try:
        result = await asyncio.shield(task)
        if result.returncode != 0:
            raise RuntimeError(
                "Gazebo pose read failed: " + result.stderr.decode(errors="replace")[:400]
            )
        text = result.stdout.decode("utf-8")
        return text
    finally:
        primary = sys.exception()
        cancel_event.set()
        try:
            await asyncio.shield(task)
        except (Exception, asyncio.CancelledError):
            if primary is None:
                raise


# 功能：
#   验证飞控指令中真正发送的四个字段，坏值在到达 MAVSDK 前拒绝。
# 输入：
#   command：位置或速度设定值。
# 输出：
#   None：不返回数据，不对越界输入进行静默修补。
def _validate_control_setpoint(command: Setpoint | VelocitySetpoint) -> None:
    fields = (
        ("north_m_s", "east_m_s", "down_m_s", "yaw_deg")
        if isinstance(command, VelocitySetpoint)
        else ("north_m", "east_m", "down_m", "yaw_deg")
    )
    for field in fields:
        _native_number(getattr(command, field), field)


# 功能：
#   按消息边界解析本执行器使用的 protobuf 文本子集，保留重复字段供消费者拒绝歧义。
# 输入：
#   text：不超过一 MiB 的完整消息文本，允许双引号字符串、标量及花括号子消息。
# 输出：
#   fields：字段名到值列表的树，不允许跨实体寻找缺失字段。
def _textproto_fields(text: str) -> dict[str, list[Any]]:
    if not isinstance(text, str) or len(text) > 1_048_576:
        raise RuntimeError("Gazebo text message exceeds its input budget")
    token_pattern = re.compile(
        r'\s+|#[^\n]*|"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z_0-9]*|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[{}:]'
    )
    fields: dict[str, list[Any]] = {}
    stack = [fields]
    key: str | None = None
    colon = False
    position = 0
    count = 0
    while position < len(text):
        match = token_pattern.match(text, position)
        if match is None:
            raise RuntimeError("Gazebo text message has an unsupported token")
        token = match.group()
        position = match.end()
        if token.isspace() or token.startswith("#"):
            continue
        count += 1
        if count > 200_000:
            raise RuntimeError("Gazebo text message exceeds its token budget")
        if token == "}":
            if key is not None or len(stack) == 1:
                raise RuntimeError("Gazebo text message has an unmatched closing brace")
            stack.pop()
            continue
        if key is None:
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token) is None:
                raise RuntimeError("Gazebo text message requires a field name")
            key, colon = token, False
            continue
        if token == ":":
            if colon:
                raise RuntimeError("Gazebo text message repeats a field separator")
            colon = True
            continue
        if token == "{":
            if len(stack) >= 16:
                raise RuntimeError("Gazebo text message exceeds its nesting budget")
            child: dict[str, list[Any]] = {}
            stack[-1].setdefault(key, []).append(child)
            stack.append(child)
        else:
            if not colon:
                raise RuntimeError("Gazebo scalar field omitted its separator")
            if token.startswith('"'):
                value: Any = json.loads(token)
            elif token in {"true", "false"}:
                value = token == "true"
            else:
                value = _finite_float(token, "Gazebo text scalar")
            stack[-1].setdefault(key, []).append(value)
        key, colon = None, False
    if len(stack) != 1 or key is not None:
        raise RuntimeError("Gazebo text message is incomplete")
    return fields


# 功能：
#   从一层消息读取单值字段，重复字段不能通过取第一个值掩盖冲突。
# 输入：
#   fields：当前消息的字段列表。
#   name：要读取的字段名。
#   default：字段省略时的 protobuf 默认值。
# 输出：
#   value：唯一字段值或明确指定的默认值。
def _textproto_value(fields: dict[str, list[Any]], name: str, default: Any = None) -> Any:
    values = fields.get(name, [])
    if len(values) > 1:
        raise RuntimeError(f"Gazebo text message repeats {name}")
    value = values[0] if values else default
    return value


# 功能：
#   选择唯一具名实体并校验其位姿，不借用下一个实体的坐标或默认生成姿态。
# 输入：
#   raw_pose_vector：完整 Pose_V 或 Pose 文本消息。
#   model_name：必须匹配的模型名称。
# 输出：
#   pose：该模型的有限世界位置和单位姿态。
def _parse_gazebo_model_pose(raw_pose_vector: str, model_name: str) -> GazeboModelPose:
    fields = _textproto_fields(raw_pose_vector)
    records = fields.get("pose", [fields])
    matches = [
        item
        for item in records
        if isinstance(item, dict) and _textproto_value(item, "name") == model_name
    ]
    if not matches:
        raise RuntimeError(f"Gazebo pose vector did not contain model: {model_name}")
    if len(matches) != 1:
        raise RuntimeError("Gazebo pose vector repeats the selected model")
    position = _textproto_value(matches[0], "position")
    orientation = _textproto_value(matches[0], "orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise RuntimeError("Gazebo model omitted position or orientation")
    coordinates = [_textproto_value(position, axis, 0.0) for axis in ("x", "y", "z")]
    quaternion = [_textproto_value(orientation, axis, 0.0) for axis in ("x", "y", "z", "w")]
    pose = GazeboModelPose(*coordinates, *quaternion)
    return pose


# 功能：
#   用单位姿态将机体局部向量旋转到 Gazebo ENU 世界，不添加位置平移。
# 输入：
#   pose：机体相对世界的有效姿态。
#   vector：机体坐标中的三个分量。
# 输出：
#   rotated：世界坐标中的三个旋转后分量。
def _rotate_gazebo_model_vector(
    pose: GazeboModelPose,
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a model-frame vector into Gazebo world ENU coordinates."""

    pose.__post_init__()
    vx, vy, vz = (_native_number(value, "body vector") for value in vector)
    qx, qy, qz, qw = pose.qx, pose.qy, pose.qz, pose.qw
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rotated = (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )
    for value in rotated:
        _native_number(value, "rotated body vector")
    return rotated


# 功能：
#   从世界 ENU 中的机体前向计算 NED 航向，垂直前向没有可用水平航向。
# 输入：
#   pose：具备单位四元数的真实模型位姿。
# 输出：
#   heading：从北顺时针计量并规范到正负 180 度附近的航向角。
def gazebo_body_heading_ned_deg(pose: GazeboModelPose) -> float:
    """Measure body +X heading in NED degrees from a Gazebo ENU quaternion."""

    east, north, _ = _rotate_gazebo_model_vector(pose, (1.0, 0.0, 0.0))
    if math.hypot(east, north) <= 1e-9:
        raise RuntimeError("Gazebo body forward axis has no horizontal component")
    heading = _normalized_yaw_deg(math.degrees(math.atan2(east, north)))
    return heading


# 功能：
#   延迟加载 Runtime 的 Gazebo 消息及传输绑定，让不含仿真依赖的宿主仍可解析计划。
# 输入：
#   无。
# 输出：
#   bindings：节点、位姿、位姿列表、布尔、空消息和字符串消息六种类型。
def _gazebo_transport_bindings() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load Gazebo bindings lazily so Windows planning remains importable."""

    system_packages = "/usr/lib/python3/dist-packages"
    if system_packages not in sys.path:
        sys.path.append(system_packages)
    try:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.transport13 import Node
    except ModuleNotFoundError as error:
        raise RuntimeError("Gazebo Python transport bindings are unavailable") from error
    bindings = Node, Pose, Pose_V, Boolean, Empty, StringMsg
    return bindings


class ExternalSafetyAbort(RuntimeError):
    """Bounded runner-to-executor stop request with cleanup semantics."""

    # 功能：
    #   保存外部终止原因和世界暂停状态，供退出流程选择正确的清理策略。
    # 输入：
    #   reason：非空终止原因。
    #   world_paused：是否已经由仿真监管方暂停世界。
    # 输出：
    #   None：初始化明确的终止异常。
    def __init__(self, reason: str, *, world_paused: bool) -> None:
        if type(world_paused) is not bool or not isinstance(reason, str) or not reason.strip():
            raise ValueError("external abort requires an explicit reason and pause flag")
        super().__init__(f"external safety abort requested: {reason}")
        self.reason = reason
        self.world_paused = world_paused


# 功能：
#   检查前一运行阶段必需的证据对象，不用空字典掩盖未执行的步骤。
# 输入：
#   value：前一阶段保存的结果。
#   label：该结果的业务名称。
# 输出：
#   value：确实存在的运行细节对象。
def _require_runtime_details(
    value: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"runtime scenario invariant violated: missing {label}")
    return value


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    z: float
    speed_limit_mps: float | None = None


@dataclass(frozen=True)
class ReferenceTrackPlan:
    points: list[TrackPoint]
    track_type: str | None
    hover_duration_seconds: float | None
    stop_at_waypoints: bool
    waypoint_hold_seconds: float
    waypoint_position_tolerance_m: float
    waypoint_speed_tolerance_mps: float
    waypoint_stable_window_seconds: float
    waypoint_settle_timeout_seconds: float


@dataclass(frozen=True)
class ControllerParams:
    vel_limit: float
    accel_limit: float


@dataclass(frozen=True)
class Setpoint:
    north_m: float
    east_m: float
    down_m: float
    yaw_deg: float


@dataclass(frozen=True)
class VelocitySetpoint:
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float


@dataclass(frozen=True)
class SetpointSchedulePlan:
    schedule: list[Setpoint]
    track_start_index: int
    track_end_index: int
    waypoint_arrival_indices: tuple[int, ...] = ()


# 功能：
#   把出生点相对位移平移到 PX4 实测局部原点，不把地图真值拟合成估计器位置。
# 输入：
#   schedule：出生点相对位置调度。
#   origin：解锁前实测的 NED 位置与速度。
# 输出：
#   rebased：相对当前估计器原点的独立指令列表。
def rebase_setpoint_schedule(
    schedule: list[Setpoint],
    origin: PositionVelocityNed,
) -> list[Setpoint]:
    """Translate spawn-relative setpoints into PX4's measured local-NED frame."""

    for value in (origin.north_m, origin.east_m, origin.down_m):
        _native_number(value, "origin")
    rebased = [
        Setpoint(
            north_m=origin.north_m + setpoint.north_m,
            east_m=origin.east_m + setpoint.east_m,
            down_m=origin.down_m + setpoint.down_m,
            yaw_deg=setpoint.yaw_deg,
        )
        for setpoint in schedule
    ]
    for command in rebased:
        _validate_control_setpoint(command)
    return rebased


# 功能：
#   将位置调度的航向统一为解锁前实测值，避免把路线切线误当绝对机体航向。
# 输入：
#   schedule：待保持航向的位置指令列表。
#   heading_deg：已经实测的 PX4 航向角。
# 输出：
#   held：仅替换航向后的独立调度列表。
def hold_setpoint_schedule_heading(
    schedule: list[Setpoint],
    heading_deg: float,
) -> list[Setpoint]:
    """Preserve measured heading for position-only route execution.

    A route tangent is not a safe implicit yaw command. PX4's initialized
    magnetic/local heading can differ from the Gazebo model yaw; forcing the
    tangent through ``PositionNedYaw`` can saturate the yaw controller and
    consume the thrust authority needed to hold altitude. Deliberate body or
    camera yaw must use a separately qualified runtime action.
    """

    measured_heading = _finite_float(heading_deg, "measured heading")
    held = [replace(setpoint, yaw_deg=measured_heading) for setpoint in schedule]
    return held


# 功能：
#   计算环绕角度间的最短有符号转角，对正好半圈的情况保持确定的旋转方向。
# 输入：
#   start_deg：起始航向角。
#   end_deg：目标航向角。
# 输出：
#   delta：正负 180 度以内的转角。
def _shortest_yaw_delta_deg(start_deg: float, end_deg: float) -> float:
    """Return the deterministic shortest signed turn from start to end."""

    start_deg = _native_number(start_deg, "start heading")
    end_deg = _native_number(end_deg, "end heading")
    delta = (end_deg % 360.0 - start_deg % 360.0 + 180.0) % 360.0 - 180.0
    if math.isclose(delta, -180.0, abs_tol=1e-12) and end_deg - start_deg > 0.0:
        return 180.0
    return delta


# 功能：
#   将有限航向折回一圈以内，统一半圈边界的表达方式。
# 输入：
#   yaw_deg：任意有限航向角。
# 输出：
#   normalized：规范后的角度。
def _normalized_yaw_deg(yaw_deg: float) -> float:
    yaw_deg = _native_number(yaw_deg, "heading")
    normalized = (yaw_deg + 180.0) % 360.0 - 180.0
    normalized = 180.0 if math.isclose(normalized, -180.0, abs_tol=1e-12) else normalized
    return normalized


# 功能：
#   1. 用实测机体与 PX4 航向之差对齐路线航向，限制每个采样间隔的转角。
#   2. 大转角先在前一位置转向，再继续平移，并同步更新阶段及航点下标。
# 输入：
#   plan：原调度及各阶段下标。
#   measured_px4_heading_deg：飞控实测航向。
#   measured_body_heading_ned_deg：Gazebo 机体前向换算的 NED 航向。
#   rate_hz：指令频率。
#   maximum_yaw_rate_deg_s：允许的最大每秒转角。
# 输出：
#   aligned：包含新增转向采样及重新映射下标的调度计划。
def align_setpoint_schedule_to_route_tangent(
    plan: SetpointSchedulePlan,
    *,
    measured_px4_heading_deg: float,
    measured_body_heading_ned_deg: float,
    rate_hz: float,
    maximum_yaw_rate_deg_s: float,
) -> SetpointSchedulePlan:
    """Map route-relative yaw onto measured heading and rate-limit every turn.

    The planner's first yaw is only a relative route reference; it must never
    be treated as an absolute PX4 heading. Later tangent changes are unwrapped
    using shortest turns. Any change larger than one rate-limited tick is
    completed at the previous position before translational motion resumes, so
    a fixed forward camera does not silently become a side/rear camera during
    travel.
    """

    if not plan.schedule:
        raise ValueError("setpoint schedule is empty")
    if len(plan.schedule) > MAX_SETPOINTS:
        raise ValueError("setpoint schedule exceeds its sample budget")
    for index in (plan.track_start_index, plan.track_end_index, *plan.waypoint_arrival_indices):
        if type(index) is not int or not 0 <= index < len(plan.schedule):
            raise ValueError("heading plan contains an invalid phase index")
    if plan.track_start_index > plan.track_end_index:
        raise ValueError("heading plan phases are reversed")
    for command in plan.schedule:
        _validate_control_setpoint(command)
    measured_px4_heading = _finite_float(
        measured_px4_heading_deg,
        "measured PX4 heading",
    )
    measured_body_heading = _finite_float(
        measured_body_heading_ned_deg,
        "measured body heading",
    )
    if not math.isfinite(rate_hz) or rate_hz <= 0.0 or rate_hz > MAX_SETPOINT_RATE_HZ:
        raise ValueError(f"rate_hz must be finite and in (0, {MAX_SETPOINT_RATE_HZ:g}]")
    if (
        not math.isfinite(maximum_yaw_rate_deg_s)
        or maximum_yaw_rate_deg_s < 1.0
        or maximum_yaw_rate_deg_s > 90.0
    ):
        raise ValueError("maximum yaw rate must be within [1, 90] deg/s")

    maximum_step_deg = maximum_yaw_rate_deg_s / rate_hz
    heading_frame_offset_deg = measured_px4_heading - measured_body_heading
    previous_command_yaw = measured_px4_heading
    previous_position = plan.schedule[0]
    route_heading_seen = False
    remapped_indices: dict[int, int] = {}
    transformed: list[Setpoint] = []

    for source_index, source in enumerate(plan.schedule):
        planned_yaw = _finite_float(source.yaw_deg, f"planned yaw index {source_index}")
        horizontal_translation = (
            source_index > 0
            and math.hypot(
                source.north_m - previous_position.north_m,
                source.east_m - previous_position.east_m,
            )
            > 1e-9
        )
        route_heading_seen = route_heading_seen or horizontal_translation
        desired_command_yaw = (
            planned_yaw + heading_frame_offset_deg if route_heading_seen else measured_px4_heading
        )
        command_delta = _shortest_yaw_delta_deg(
            previous_command_yaw,
            desired_command_yaw,
        )
        turn_steps = int(math.ceil(abs(command_delta) / maximum_step_deg))
        if turn_steps > 1:
            if turn_steps + len(transformed) + 1 > MAX_SETPOINTS:
                raise ValueError(f"yaw-aligned schedule exceeds the {MAX_SETPOINTS}-sample limit")
            for turn_step in range(1, turn_steps + 1):
                yaw = previous_command_yaw + command_delta * turn_step / turn_steps
                transformed.append(replace(previous_position, yaw_deg=_normalized_yaw_deg(yaw)))
        transformed.append(replace(source, yaw_deg=_normalized_yaw_deg(desired_command_yaw)))
        if len(transformed) > MAX_SETPOINTS:
            raise ValueError(f"yaw-aligned schedule exceeds the {MAX_SETPOINTS}-sample limit")
        remapped_indices[source_index] = len(transformed) - 1
        previous_command_yaw = desired_command_yaw
        previous_position = source

    aligned = SetpointSchedulePlan(
        schedule=transformed,
        track_start_index=remapped_indices[plan.track_start_index],
        track_end_index=remapped_indices[plan.track_end_index],
        waypoint_arrival_indices=tuple(
            remapped_indices[index] for index in plan.waypoint_arrival_indices
        ),
    )
    return aligned


@dataclass(frozen=True)
class TelemetryHealth:
    connected: bool
    global_position_ok: bool
    home_position_ok: bool
    local_position_ok: bool
    armable: bool


@dataclass(frozen=True)
class PositionVelocityNed:
    north_m: float
    east_m: float
    down_m: float
    north_m_s: float
    east_m_s: float
    down_m_s: float
    received_at_unix_ms: int | None = None


class OffboardClientProtocol(Protocol):
    # 功能：
    #   建立本客户端到飞控的连接并启动它拥有的遥测订阅。
    # 输入：
    #   connection_url：显式 MAVLink 连接地址。
    # 输出：
    #   None：连接失败抛出异常，不生成模拟连接。
    async def connect(self, connection_url: str) -> None: ...

    # 功能：
    #   在解锁前等待飞控报告满足本地位置飞行的健康状态。
    # 输入：
    #   timeout_seconds：最大等待秒数。
    # 输出：
    #   health：连接、估计器及解锁就绪状态。
    async def wait_until_ready(self, timeout_seconds: float) -> TelemetryHealth: ...

    # 功能：
    #   向飞控请求解锁，成功回执不等于已起飞。
    # 输入：
    #   self：已连接且通过准备检查的客户端。
    # 输出：
    #   None：命令失败以异常报告。
    async def arm(self) -> None: ...

    # 功能：
    #   发送位置控制设定值，供显式参考轨迹或安全保持使用。
    # 输入：
    #   setpoint：北、东、向下位置及航向。
    # 输出：
    #   None：不返回到达确认。
    async def set_position_ned(self, setpoint: Setpoint) -> None: ...

    # 功能：
    #   发送位置控制与速度前馈的组合，不把它当成纯速度驾驶模式。
    # 输入：
    #   setpoint：位置控制目标。
    #   velocity：同一局部坐标系中的速度前馈。
    # 输出：
    #   None：指令已提交或异常。
    async def set_position_velocity_ned(
        self, setpoint: Setpoint, velocity: VelocitySetpoint
    ) -> None: ...

    # 功能：
    #   为本地模型提供真正的速度模式传输，不附带位置追踪目标。
    # 输入：
    #   velocity：NED 三轴速度和航向角。
    # 输出：
    #   None：不把发送成功解释为避障成功。
    async def set_velocity_ned(self, velocity: VelocitySetpoint) -> None: ...

    # 功能：
    #   请求飞控切换到已经预发送指令的外部控制模式。
    # 输入：
    #   self：完成指令预发送的客户端。
    # 输出：
    #   None：外部控制请求的回执或异常。
    async def start_offboard(self) -> None: ...

    # 功能：
    #   请求退出外部控制，供正常结束及失败清理使用。
    # 输入：
    #   self：可能已进入外部控制的客户端。
    # 输出：
    #   None：退出请求回执或异常。
    async def stop_offboard(self) -> None: ...

    # 功能：
    #   发送飞控降落请求，随后仍需独立观察落地状态。
    # 输入：
    #   self：当前飞控连接。
    # 输出：
    #   None：命令回执或异常，不返回落地判定。
    async def land(self) -> None: ...

    # 功能：
    #   等待真实飞控报告 ON_GROUND，不能以等待时长替代落地证据。
    # 输入：
    #   timeout_seconds：允许的落地观察时限。
    # 输出：
    #   observation：本次明确的落地状态与确认标记。
    async def wait_until_landed(self, timeout_seconds: float) -> dict[str, Any]: ...

    # 功能：
    #   读取指定整数飞控参数供执行前后比对。
    # 输入：
    #   name：精确参数名称。
    # 输出：
    #   value：飞控返回的整数值。
    async def get_param_int(self, name: str) -> int: ...

    # 功能：
    #   写入整数飞控参数，读取确认由调用方单独执行。
    # 输入：
    #   name：被授权修改的参数名称。
    #   value：目标整数值。
    # 输出：
    #   None：写入回执或异常。
    async def set_param_int(self, name: str, value: int) -> None: ...

    # 功能：
    #   读取浮点飞控参数，不用配置文件中的期望值代替实测读回。
    # 输入：
    #   name：精确参数名称。
    # 输出：
    #   value：飞控返回的有限浮点数。
    async def get_param_float(self, name: str) -> float: ...

    # 功能：
    #   向飞控请求写入浮点参数。
    # 输入：
    #   name：被授权的参数名称。
    #   value：有限目标数值。
    # 输出：
    #   None：写入回执或异常，仍需后续读回验证。
    async def set_param_float(self, name: str, value: float) -> None: ...

    # 功能：
    #   执行计划授权的相机拍照或录像动作。
    # 输入：
    #   parameters：相机组件与动作参数。
    # 输出：
    #   receipt：相机命令回执，不冒充实际图像文件。
    async def execute_camera_command(self, parameters: dict[str, Any]) -> dict[str, Any]: ...

    # 功能：
    #   通过声明的硬件或仿真协议执行载荷连接及分离。
    # 输入：
    #   parameters：动作、设备和关节绑定。
    # 输出：
    #   receipt：该协议下实际取得的确认及诊断。
    async def execute_payload_command(self, parameters: dict[str, Any]) -> dict[str, Any]: ...

    # 功能：
    #   读取指定载荷关节的已知观察状态。
    # 输入：
    #   output_topic：本载荷的状态主题。
    #   timeout_seconds：未知状态的观察预算。
    # 输出：
    #   observation：有来源和原始时刻的分离状态。
    async def sample_payload_state(
        self, output_topic: str, timeout_seconds: float
    ) -> dict[str, Any]: ...

    # 功能：
    #   修改声明的飞控避障选项并验证读回。
    # 输入：
    #   enabled：严格布尔的开启请求。
    # 输出：
    #   receipt：参数修改前后值，不证明周围无障碍。
    async def execute_avoidance_command(self, enabled: bool) -> dict[str, Any]: ...

    # 功能：
    #   获取任务明确选择的仿真跟随目标位置。
    # 输入：
    #   parameters：目标模型身份及位姿主题。
    # 输出：
    #   pose：目标坐标和来源，不作为本机状态估计输入。
    async def sample_gazebo_pose(self, parameters: dict[str, Any]) -> dict[str, float | str]: ...

    # 功能：
    #   读取足够新鲜的电池状态供能量门控。
    # 输入：
    #   timeout_seconds：等待及新鲜度预算。
    # 输出：
    #   battery：剩余百分比与电压。
    async def sample_battery(self, timeout_seconds: float) -> dict[str, float]: ...

    # 功能：
    #   观察 GNSS 的卫星数和定位类型供降级试验验证。
    # 输入：
    #   timeout_seconds：获取样本的最长秒数。
    # 输出：
    #   gps：真实卫星数、定位类型值和名称。
    async def sample_gps_info(self, timeout_seconds: float) -> dict[str, int | str]: ...

    # 功能：
    #   从共享原生流读取本机局部位置及速度。
    # 输入：
    #   timeout_seconds：等待及新鲜度预算。
    # 输出：
    #   sample：NED 三轴位置、速度与接收时刻。
    async def sample_position_velocity_ned(self, timeout_seconds: float) -> PositionVelocityNed: ...

    # 功能：
    #   在回收原订阅后替换卡住的位置速度读取流，不重新解锁。
    # 输入：
    #   self：拥有该遥测订阅的客户端。
    # 输出：
    #   None：替换结果通过新流的实际样本体现。
    async def restart_position_velocity_ned_stream(self) -> None: ...

    # 功能：
    #   读取飞控实测航向供初始坐标对齐。
    # 输入：
    #   timeout_seconds：等待及样本年龄预算。
    # 输出：
    #   heading：NED 航向角，单位为度。
    async def sample_heading_deg(self, timeout_seconds: float) -> float: ...

    # 功能：
    #   汇总多路动力学遥测及各自年龄、错误和就绪条件。
    # 输入：
    #   max_age_seconds：快速控制允许的样本年龄上限。
    # 输出：
    #   telemetry：相互独立的来源快照及门控信息。
    def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]: ...

    # 功能：
    #   请求各来源的明确采样频率并保留请求回执。
    # 输入：
    #   self：已连接的飞控客户端。
    # 输出：
    #   evidence：逐来源速率请求结果。
    async def configure_dynamics_telemetry_rates(self) -> dict[str, Any]: ...

    # 功能：
    #   按世界和模型身份获取仿真真实姿态供限定的对齐或挂载检查。
    # 输入：
    #   world_name：仿真世界名称。
    #   model_name：精确模型名称。
    #   timeout_seconds：采样总等待秒数。
    # 输出：
    #   pose：选定模型的世界位姿。
    async def sample_gazebo_model_pose(
        self,
        *,
        world_name: str,
        model_name: str,
        timeout_seconds: float,
    ) -> GazeboModelPose: ...

    # 功能：
    #   回收本客户端拥有的订阅、后台任务及服务进程。
    # 输入：
    #   self：正在退出的连接实例。
    # 输出：
    #   None：资源无法确认回收时抛出异常。
    async def close(self) -> None: ...


# 功能：
#   向系统查询本实例的可用回环端口，避免所有嵌入式服务竞争固定端口。
# 输入：
#   无。
# 输出：
#   port：待服务绑定的端口号；探测与后续绑定之间仍可能存在系统竞争。
def _allocate_loopback_tcp_port() -> int:
    """Choose a dedicated local gRPC port for one embedded MAVSDK server.

    MAVSDK-Python otherwise starts every embedded server on TCP 50051.  A
    recently killed server or another local flight process can still own that
    port, and the C++ server aborts instead of returning a normal bind error.
    The flight process is the only consumer of this short-lived port.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    if port <= 0 or port > 65_535:
        raise RuntimeError("operating system returned an invalid MAVSDK gRPC port")
    return port


class MavsdkOffboardClient:
    # 功能：
    #   加载真实 MAVSDK 类型并建立空遥测、资源和载荷观察状态，不连接或解锁飞控。
    # 输入：
    #   self：新客户端；可选执行器量程来自明确的部署配置。
    # 输出：
    #   None：初始化资源所有权和状态容器。
    def __init__(self) -> None:
        try:
            from mavsdk import System
            from mavsdk.offboard import (
                OffboardError,
                PositionNedYaw,
                VelocityNedYaw,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("mavsdk is required for PX4 offboard execution") from exc

        self._system_cls = System
        self._position_cls = PositionNedYaw
        self._velocity_cls = VelocityNedYaw
        self._offboard_error_cls = OffboardError
        self._system: Any | None = None
        self._flight_command_requested = False
        self._mavsdk_server_port: int | None = None
        self._position_velocity_condition: asyncio.Condition | None = None
        self._position_velocity_sample: tuple[PositionVelocityNed, float] | None = None
        self._position_velocity_error: BaseException | None = None
        self._position_velocity_task: asyncio.Task[None] | None = None
        self._dynamics_samples: dict[str, tuple[dict[str, Any], float]] = {}
        self._dynamics_received_at_unix_ms: dict[str, int] = {}
        self._dynamics_errors: dict[str, str] = {}
        self._dynamics_restart_counts: dict[str, int] = {}
        self._dynamics_tasks: list[asyncio.Task[None]] = []
        raw_actuator_maximum = os.environ.get(
            ACTUATOR_OUTPUT_ABSOLUTE_MAXIMUM_ENV,
            "",
        ).strip()
        self._actuator_output_absolute_maximum: float | None = None
        if raw_actuator_maximum:
            actuator_maximum = float(raw_actuator_maximum)
            if not math.isfinite(actuator_maximum) or actuator_maximum <= 0.0:
                raise RuntimeError(
                    "PX4 actuator-output absolute maximum must be positive and finite"
                )
            self._actuator_output_absolute_maximum = actuator_maximum
        # 环境中的 attached/detached 文字不是观测；初态只能来自绑定本次运行的真实回执。
        self._payload_observers: dict[str, dict[str, Any]] = {}
        self._payload_command_active = False

    # 功能：
    #   在独占本地服务端口连接飞控，并为每个存在的遥测接口开启唯一持续订阅。
    # 输入：
    #   connection_url：本次 MAVLink 连接地址。
    # 输出：
    #   None：连接和初始化成功后返回；失败由拥有者调用 close 回收部分资源。
    async def connect(self, connection_url: str) -> None:
        if getattr(self, "_flight_command_requested", False) is not False:
            raise RuntimeError("PREFLIGHT_RECOVERY_FORBIDDEN_AFTER_MOTION_REQUEST")
        if self._system is not None:
            raise RuntimeError("PX4 offboard client is already connected")
        if any(not task.done() for task in getattr(self, "_dynamics_tasks", ())):
            raise RuntimeError("PX4 previous telemetry tasks have not closed")
        self._mavsdk_server_port = _allocate_loopback_tcp_port()
        self._system = self._system_cls(port=self._mavsdk_server_port)
        await self._system.connect(system_address=connection_url)
        await self._prime_payload_observer()
        self._position_velocity_condition = asyncio.Condition()
        self._position_velocity_sample = None
        self._position_velocity_error = None
        self._position_velocity_task = asyncio.create_task(
            self._collect_position_velocity_ned(),
            name="px4-position-velocity-telemetry",
        )
        self._dynamics_samples = {}
        self._dynamics_received_at_unix_ms = {}
        self._dynamics_errors = {}
        self._dynamics_restart_counts = {}
        self._dynamics_tasks = []
        telemetry = self._require_system().telemetry
        for stream_name, collector in (
            ("imu", self._collect_imu),
            ("attitude_euler", self._collect_attitude_euler),
            ("battery", self._collect_dynamics_battery),
            ("actuator_output_status", self._collect_actuator_output_status),
            ("odometry", self._collect_odometry),
        ):
            if callable(getattr(telemetry, stream_name, None)):
                self._dynamics_tasks.append(
                    asyncio.create_task(
                        collector(),
                        name=f"px4-{stream_name.replace('_', '-')}-telemetry",
                    )
                )

    # 功能：
    #   1. 持续接收真实位置速度，临时停流后关闭旧订阅、退避并建立唯一替代订阅。
    #   2. 恢复前撤销旧值，坏物理值终止采集；重订阅不赋予运动或重新起飞权限。
    # 输入：
    #   self：持有唯一位置速度订阅及通知条件的客户端。
    # 输出：
    #   None：采样结果写入本客户端缓存，取消时关闭该任务。
    async def _collect_position_velocity_ned(self) -> None:
        """Own one long-lived MAVSDK stream and publish its latest sample.

        Re-opening ``position_velocity_ned`` for every 20 Hz controller tick
        creates competing gRPC subscriptions.  Under a render-heavy Gazebo
        scene a single subscription timeout then kills the executor even
        though PX4 resumes publishing immediately afterwards.  One collector
        makes telemetry latest-value data and lets every safety consumer share
        the same bounded-freshness contract.
        """

        condition = self._position_velocity_condition
        if condition is None:
            raise RuntimeError("PX4 telemetry collector started without a condition")
        retry_delay = .1
        while True:
            try:
                async for telemetry in self._guarded_dynamics_stream(
                    "position_velocity", self._require_system().telemetry.position_velocity_ned()
                ):
                    position, velocity = telemetry.position, telemetry.velocity
                    sample = PositionVelocityNed(
                        north_m=_native_number(position.north_m, "north_m"),
                        east_m=_native_number(position.east_m, "east_m"),
                        down_m=_native_number(position.down_m, "down_m"),
                        north_m_s=_native_number(velocity.north_m_s, "north_m_s"),
                        east_m_s=_native_number(velocity.east_m_s, "east_m_s"),
                        down_m_s=_native_number(velocity.down_m_s, "down_m_s"),
                        received_at_unix_ms=int(time.time() * 1_000),
                    )
                    if not all(math.isfinite(value) for value in sample.__dict__.values()):
                        raise ValueError(
                            "PX4 position/velocity telemetry contains non-finite values")
                    async with condition:
                        self._position_velocity_sample = (sample, time.monotonic())
                        self._position_velocity_error = None
                        getattr(self, "_dynamics_errors", {}).pop("position_velocity", None)
                        condition.notify_all()
                    retry_delay = .1
            except asyncio.CancelledError:
                raise
            except Exception as error:
                async with condition:
                    # 旧缓存在恢复期间不可读，恢复必须取得真正的新样本。
                    self._position_velocity_error = error
                    self._position_velocity_sample = None
                    condition.notify_all()
                recoverable = not isinstance(error, TelemetrySubscriptionCloseFailed) and (
                    isinstance(error, (TimeoutError, ConnectionError))
                    or any(isinstance(item, StopAsyncIteration) for item in _exception_chain(error))
                    or _is_recoverable_mavsdk_preflight_transport_error(error))
                if not recoverable:
                    return  # 非法物理值及编程错误不能靠重订阅隐瞒。
                retry_delay = await self._restart_dynamics_stream_after_error(
                    "position_velocity", error, retry_delay)

    # 功能：
    #   按原始时刻接收独立遥测快照，重复包不续期，倒退或同刻冲突不能覆盖有效来源。
    # 输入：
    #   source：具有明确新鲜度契约的传感器名称。
    #   payload：本次传感器字段，原生时间戳保持整数微秒。
    # 输出：
    #   None：通过校验后更新本客户端的最新值缓存。
    def _store_dynamics_sample(self, source: str, payload: dict[str, Any]) -> None:
        from dronedream_plugin_sdk.protocol import copy_json

        if source not in DYNAMICS_STREAM_SAMPLE_TIMEOUT_SECONDS:
            raise ValueError("PX4 dynamics source is unsupported")
        payload = copy_json(payload, limit=32_768)
        previous = self._dynamics_samples.get(source)
        timestamp = payload.get("timestamp_us")
        if source in {"imu", "attitude", "odometry"}:
            if type(timestamp) is not int or not 0 <= timestamp < 2**63:
                raise ValueError("PX4 native timestamp must be a nonnegative integer")
            previous_timestamp = previous[0].get("timestamp_us") if previous is not None else None
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise RuntimeError("PX4 native timestamp regressed; reconnect required")
        if (
            previous is not None
            and timestamp is not None
            and previous[0].get("timestamp_us") == timestamp
        ):
            if previous[0] != payload:
                raise RuntimeError("PX4 repeated timestamp has conflicting sensor content")
            # A retransmission is not a fresh physical sample.
            return
        self._dynamics_samples[source] = (payload, time.monotonic())
        if not hasattr(self, "_dynamics_received_at_unix_ms"):
            self._dynamics_received_at_unix_ms = {}
        self._dynamics_received_at_unix_ms[source] = int(time.time() * 1000)
        self._dynamics_errors.pop(source, None)

    # 功能：
    #   记录单一来源的停流原因与重启次数，按有上限的退避延时再订阅。
    # 输入：
    #   source：失败的遥测来源。
    #   error：本次读取异常。
    #   retry_delay_seconds：当前退避秒数。
    # 输出：
    #   delay：下一次重试延时，不超过两秒。
    async def _restart_dynamics_stream_after_error(
        self,
        source: str,
        error: BaseException,
        retry_delay_seconds: float,
    ) -> float:
        """Back off before replacing one ended dynamics subscription.

        Each collector owns exactly one subscription, so the retry cannot
        create competing MAVSDK streams.  The last sample becomes stale and
        the payload expert remains unavailable until a replacement stream
        produces fresh evidence.  A bounded exponential delay prevents a
        permanently unsupported stream from spinning the event loop.
        """

        self._dynamics_errors[source] = type(error).__name__
        if isinstance(error, TelemetrySubscriptionCloseFailed):
            raise error  # 回收失败不能通过再开一个订阅来掩盖。
        counts = getattr(self, "_dynamics_restart_counts", None)
        if not isinstance(counts, dict):
            counts = {}
            self._dynamics_restart_counts = counts
        counts[source] = int(counts.get(source, 0)) + 1
        await asyncio.sleep(retry_delay_seconds)
        delay = min(2.0, retry_delay_seconds * 2.0)
        return delay

    # 功能：
    #   为一条原生异步订阅限制样本等待，并在退出时关闭该订阅。
    # 输入：
    #   source：决定停流时限的来源名称。
    #   stream：MAVSDK 提供的单个异步流。
    # 输出：
    #   sample：逐次交付的真实遥测消息，不生成缺失样本。
    async def _guarded_dynamics_stream(self, source: str, stream: Any) -> Any:
        """Yield one subscription while rejecting a connected-but-stalled stream."""

        timeout_seconds = DYNAMICS_STREAM_SAMPLE_TIMEOUT_SECONDS.get(source)
        if timeout_seconds is None:
            raise RuntimeError(f"PX4 dynamics stream has no freshness contract: {source}")
        iterator = stream.__aiter__()
        try:
            while True:
                try:
                    sample = await asyncio.wait_for(
                        anext(iterator),
                        timeout=timeout_seconds,
                    )
                except StopAsyncIteration as error:
                    raise RuntimeError(f"PX4 {source} telemetry stream ended") from error
                except TimeoutError as error:
                    raise TimeoutError(
                        f"PX4 {source} telemetry stream produced no fresh sample"
                    ) from error
                yield sample
        finally:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                try:
                    await asyncio.wait_for(close(), timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS)
                except Exception as error:
                    raise TelemetrySubscriptionCloseFailed(
                        "PX4 previous telemetry subscription did not close") from error

    # 功能：
    #   逐来源请求采样率并记录实际回执；请求成功不等于实际已经达到该频率。
    # 输入：
    #   self：已经连接的真实 MAVSDK 客户端。
    # 输出：
    #   evidence：各来源请求结果及必需来源的请求汇总。
    async def configure_dynamics_telemetry_rates(self) -> dict[str, Any]:
        """Request explicit MAVSDK rates and return non-secret preflight evidence."""

        telemetry = self._require_system().telemetry
        requests = (
            ("position_velocity", "set_rate_position_velocity_ned"),
            ("imu", "set_rate_imu"),
            ("attitude", "set_rate_attitude_euler"),
            ("battery", "set_rate_battery"),
            ("actuator_output", "set_rate_actuator_output_status"),
            ("odometry", "set_rate_odometry"),
        )
        sources: dict[str, dict[str, Any]] = {}
        for source, method_name in requests:
            rate_hz = DYNAMICS_TELEMETRY_REQUESTED_RATES_HZ[source]
            method = getattr(telemetry, method_name, None)
            if not callable(method):
                sources[source] = {
                    "requested_rate_hz": rate_hz,
                    "status": "unsupported",
                }
                continue
            try:
                await asyncio.wait_for(method(rate_hz), timeout=0.75)
            except Exception as error:
                sources[source] = {
                    "requested_rate_hz": rate_hz,
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            else:
                sources[source] = {
                    "requested_rate_hz": rate_hz,
                    "status": "requested",
                }
        required = ("position_velocity", "imu", "attitude", "odometry", "battery")
        evidence = {
            "sources": sources,
            "required_rate_requests_succeeded": all(
                sources[source]["status"] == "requested" for source in required
            ),
        }
        return evidence

    # 功能：
    #   保留里程计的原始时间与协方差，未知协方差保持空值而不是伪造零误差。
    # 输入：
    #   self：持有 MAVSDK 里程计接口的客户端。
    # 输出：
    #   None：合法观测保存到 odometry 来源，异常使该来源暂时不可用。
    async def _collect_odometry(self) -> None:
        """Preserve native packed covariance; unknown is JSON null, not zero."""
        retry_delay_seconds = 0.1
        while True:
            try:
                async for sample in self._guarded_dynamics_stream(
                    "odometry", self._require_system().telemetry.odometry()
                ):
                    from dronedream_agent_core.localization_evidence import (
                        normalize_native_pose_covariance,
                    )

                    covariance = normalize_native_pose_covariance(
                        list(sample.pose_covariance.covariance_matrix)
                    )
                    timestamp = sample.time_usec
                    if (
                        isinstance(timestamp, bool)
                        or not isinstance(timestamp, int)
                        or timestamp < 0
                    ):
                        raise RuntimeError("PX4 odometry timestamp is invalid")
                    frame_id = getattr(sample.frame_id, "name", "UNSUPPORTED")
                    self._store_dynamics_sample(
                        "odometry",
                        {
                            "timestamp_us": timestamp,
                            "frame_id": frame_id,
                            "pose_covariance_upper_m2": covariance,
                        },
                    )
                    retry_delay_seconds = 0.1
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                retry_delay_seconds = await self._restart_dynamics_stream_after_error(
                    "odometry",
                    error,
                    retry_delay_seconds,
                )

    # 功能：
    #   提取 FRD 三轴加速度和角速度，保持原始微秒时间供姿态及动力学判断。
    # 输入：
    #   self：拥有唯一 IMU 遥测订阅的客户端。
    # 输出：
    #   None：更新 imu 快照，不补造掉帧数据。
    async def _collect_imu(self) -> None:
        retry_delay_seconds = 0.1
        while True:
            try:
                async for sample in self._guarded_dynamics_stream(
                    "imu", self._require_system().telemetry.imu()
                ):
                    acceleration = sample.acceleration_frd
                    angular_velocity = sample.angular_velocity_frd
                    values = {
                        "acceleration_forward_m_s2": _native_number(
                            acceleration.forward_m_s2, "ax"
                        ),
                        "acceleration_right_m_s2": _native_number(acceleration.right_m_s2, "ay"),
                        "acceleration_down_m_s2": _native_number(acceleration.down_m_s2, "az"),
                        "angular_velocity_forward_rad_s": _native_number(
                            angular_velocity.forward_rad_s, "gx"
                        ),
                        "angular_velocity_right_rad_s": _native_number(
                            angular_velocity.right_rad_s, "gy"
                        ),
                        "angular_velocity_down_rad_s": _native_number(
                            angular_velocity.down_rad_s, "gz"
                        ),
                        "timestamp_us": sample.timestamp_us,
                    }
                    if not all(
                        math.isfinite(value)
                        for key, value in values.items()
                        if key != "timestamp_us"
                    ):
                        raise RuntimeError("PX4 IMU telemetry contains non-finite values")
                    self._store_dynamics_sample("imu", values)
                    retry_delay_seconds = 0.1
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                retry_delay_seconds = await self._restart_dynamics_stream_after_error(
                    "imu", error, retry_delay_seconds
                )

    # 功能：
    #   采集真实横滚、俯仰、偏航和原始时刻，供姿态编码及局部控制使用。
    # 输入：
    #   self：拥有姿态遥测订阅的客户端。
    # 输出：
    #   None：更新 attitude 快照，停流错误单独保留。
    async def _collect_attitude_euler(self) -> None:
        retry_delay_seconds = 0.1
        while True:
            try:
                async for sample in self._guarded_dynamics_stream(
                    "attitude", self._require_system().telemetry.attitude_euler()
                ):
                    values = {
                        "roll_deg": _native_number(sample.roll_deg, "roll_deg"),
                        "pitch_deg": _native_number(sample.pitch_deg, "pitch_deg"),
                        "yaw_deg": _native_number(sample.yaw_deg, "yaw_deg"),
                        "timestamp_us": sample.timestamp_us,
                    }
                    if not all(
                        math.isfinite(value)
                        for key, value in values.items()
                        if key != "timestamp_us"
                    ):
                        raise RuntimeError("PX4 attitude telemetry contains non-finite values")
                    self._store_dynamics_sample("attitude", values)
                    retry_delay_seconds = 0.1
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                retry_delay_seconds = await self._restart_dynamics_stream_after_error(
                    "attitude", error, retry_delay_seconds
                )

    # 功能：
    #   接收电量百分比、电压及可用电流，不把缺少电流的 NaN 当作零负载。
    # 输入：
    #   self：拥有电池遥测订阅的客户端。
    # 输出：
    #   None：更新电池快照；电流未知时省略该字段。
    async def _collect_dynamics_battery(self) -> None:
        retry_delay_seconds = 0.1
        while True:
            try:
                async for sample in self._guarded_dynamics_stream(
                    "battery", self._require_system().telemetry.battery()
                ):
                    values: dict[str, Any] = {
                        "remaining_percent": _native_number(
                            sample.remaining_percent, "remaining_percent"
                        ),
                        "voltage_v": _native_number(sample.voltage_v, "voltage_v"),
                    }
                    _validated_battery_telemetry(values, label="MAVSDK battery")
                    current = float(sample.current_battery_a)
                    if math.isfinite(current):
                        values["current_battery_a"] = current
                    if not all(
                        math.isfinite(value)
                        for value in values.values()
                        if isinstance(value, float)
                    ):
                        raise RuntimeError("PX4 battery telemetry contains non-finite values")
                    self._store_dynamics_sample("battery", values)
                    retry_delay_seconds = 0.1
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                retry_delay_seconds = await self._restart_dynamics_stream_after_error(
                    "battery", error, retry_delay_seconds
                )

    # 功能：
    #   保留原始电机输出并按明确量程归一化，量程不可知或超过量程时禁止负载模型使用。
    # 输入：
    #   self：拥有执行器遥测及部署量程的客户端。
    # 输出：
    #   None：更新原始输出、归一化值、量程来源及就绪状态。
    async def _collect_actuator_output_status(self) -> None:
        retry_delay_seconds = 0.1
        while True:
            try:
                async for sample in self._guarded_dynamics_stream(
                    "actuator_output",
                    self._require_system().telemetry.actuator_output_status(),
                ):
                    if not 1 <= len(sample.actuator) <= 32:
                        raise RuntimeError("PX4 actuator channel count is invalid")
                    raw_actuator = [
                        _native_number(value, "actuator output") for value in sample.actuator
                    ]
                    if not raw_actuator or not all(math.isfinite(value) for value in raw_actuator):
                        raise RuntimeError("PX4 actuator telemetry contains invalid values")
                    if type(sample.active) is not int or not 0 <= sample.active < 2**32:
                        raise RuntimeError("PX4 actuator active mask is invalid")
                    absolute_maximum = getattr(
                        self,
                        "_actuator_output_absolute_maximum",
                        None,
                    )
                    native_unit_interval = max(abs(value) for value in raw_actuator) <= 1.0 + 1e-6
                    if absolute_maximum is not None:
                        actuator = [value / absolute_maximum for value in raw_actuator]
                        normalization_kind = "configured-absolute-maximum"
                        normalization_ready = all(abs(value) <= 1.0 + 1e-6 for value in actuator)
                    elif native_unit_interval:
                        actuator = list(raw_actuator)
                        absolute_maximum = 1.0
                        normalization_kind = "native-unit-interval"
                        normalization_ready = True
                    else:
                        # MAVLink deliberately defines this stream as raw driver
                        # output.  A value such as 799 from PX4 SITL is not 799x
                        # saturation; without a vehicle-bound maximum it must not
                        # be fed to the payload model as a normalized feature.
                        actuator = list(raw_actuator)
                        normalization_kind = "unscaled-driver-native"
                        normalization_ready = False
                    self._store_dynamics_sample(
                        "actuator_output",
                        {
                            "active_mask": sample.active,
                            "actuator": actuator,
                            "raw_actuator": raw_actuator,
                            "normalization_kind": normalization_kind,
                            "normalization_ready": normalization_ready,
                            "normalization_absolute_maximum": absolute_maximum,
                        },
                    )
                    retry_delay_seconds = 0.1
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                retry_delay_seconds = await self._restart_dynamics_stream_after_error(
                    "actuator_output", error, retry_delay_seconds
                )

    # 功能：
    #   复制最新遥测并按来源年龄及错误门控，时钟倒退和流错误均不能授权负载推断。
    # 输入：
    #   max_age_seconds：消费者允许的正数年龄上限，单位为秒。
    # 输出：
    #   evidence：独立来源快照、年龄、阻断原因及推断就绪标记。
    def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]:
        if type(max_age_seconds) not in (int, float) or not 0 < max_age_seconds <= 3600:
            raise ValueError("dynamics telemetry freshness bound must be positive")
        now = time.monotonic()
        sources: dict[str, Any] = {}
        stale_sources: list[str] = []
        for source, (payload, recorded_at) in self._dynamics_samples.items():
            age_seconds = now - recorded_at
            sources[source] = {**copy.deepcopy(payload), "sample_age_seconds": age_seconds}
            received = getattr(self, "_dynamics_received_at_unix_ms", {}).get(source)
            if received is not None:
                sources[source]["received_at_unix_ms"] = received
            if not 0 <= age_seconds <= max_age_seconds:
                stale_sources.append(source)
        control_required_sources = {"imu", "attitude", "actuator_output"}
        advisory_sources = {"battery"}
        observed_sources = control_required_sources | advisory_sources
        fresh_sources = {
            source
            for source, payload in sources.items()
            if 0 <= payload["sample_age_seconds"] <= max_age_seconds
        }
        issues = [
            f"{source}:STREAM_ERROR:{error}"
            for source, error in sorted(self._dynamics_errors.items())
        ]
        issues.extend(f"{source}:SAMPLE_STALE" for source in sorted(stale_sources))
        issues.extend(
            f"{source}:SAMPLE_MISSING" for source in sorted(observed_sources - fresh_sources)
        )
        actuator = sources.get("actuator_output")
        if isinstance(actuator, dict) and actuator.get("normalization_ready") is not True:
            issues.append("actuator_output:NORMALIZATION_UNAVAILABLE")
        blocking_issues = [
            issue for issue in issues if issue.split(":", 1)[0] in control_required_sources
        ]
        evidence = {
            "schema_version": "dronedream.px4-dynamics-telemetry.v1",
            "collected_at_unix_ms": int(time.time() * 1_000),
            "maximum_sample_age_seconds": max_age_seconds,
            "ready_for_payload_inference": (
                control_required_sources <= fresh_sources
                and isinstance(actuator, dict)
                and actuator.get("normalization_ready") is True
                and not blocking_issues
            ),
            "sources": sources,
            "issue_codes": issues,
            "blocking_issue_codes": blocking_issues,
            "restart_counts": dict(sorted(getattr(self, "_dynamics_restart_counts", {}).items())),
        }
        return evidence

    # 功能：
    #   回收旧位置速度订阅再启动新订阅，清除旧样本，不能通过重开订阅刷新旧观测年龄。
    # 输入：
    #   self：已经连接且持有原订阅的客户端。
    # 输出：
    #   None：新任务建立后仍须等待它的真实样本。
    async def restart_position_velocity_ned_stream(self) -> None:
        """Replace a stalled gRPC telemetry subscription without reconnecting PX4.

        MAVSDK's embedded server continues publishing the last Offboard setpoint
        independently of this Python consumer.  A render-heavy, long-running
        simulation can stall one Python telemetry stream even while PX4 keeps
        producing local-position samples.  Cancel the old subscription before
        opening exactly one replacement so a transient reader failure never
        creates competing streams or authorizes motion from stale telemetry.
        """

        if self._system is None or self._position_velocity_condition is None:
            raise RuntimeError("PX4 position/velocity telemetry cannot restart before connect")
        telemetry_task = self._position_velocity_task
        self._position_velocity_task = None
        if telemetry_task is not None:
            telemetry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await telemetry_task
        async with self._position_velocity_condition:
            self._position_velocity_sample = None
            self._position_velocity_error = None
            self._position_velocity_condition.notify_all()
        self._position_velocity_task = asyncio.create_task(
            self._collect_position_velocity_ned(),
            name="px4-position-velocity-telemetry",
        )

    # 功能：
    #   取得本客户端已建立的飞控连接，禁止未连接实例发送命令。
    # 输入：
    #   self：当前客户端。
    # 输出：
    #   system：当前真实 MAVSDK System 实例。
    def _require_system(self) -> Any:
        if self._system is None:
            raise RuntimeError("PX4 offboard client is not connected")
        system = self._system
        return system

    # 功能：
    #   等待连接和本地估计器准备完成，全球定位仅作参考，解锁许可仍由 PX4 决定。
    # 输入：
    #   timeout_seconds：准备阶段总等待秒数。
    # 输出：
    #   sample：本次原生健康状态；未满足条件则超时，不放宽要求。
    async def wait_until_ready(
        self,
        timeout_seconds: float,
    ) -> TelemetryHealth:
        timeout_seconds = _timeout_budget(timeout_seconds)
        system = self._require_system()
        last_health: TelemetryHealth | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                connected = False
                async for state in system.core.connection_state():
                    if getattr(state, "is_connected", False) is True:
                        connected = True
                        break
                if not connected:
                    raise RuntimeError("PX4 connection state stream ended before connecting")

                async for health in system.telemetry.health():
                    sample = TelemetryHealth(
                        connected=True,
                        global_position_ok=getattr(health, "is_global_position_ok", False) is True,
                        home_position_ok=getattr(health, "is_home_position_ok", False) is True,
                        local_position_ok=getattr(health, "is_local_position_ok", False) is True,
                        armable=getattr(health, "is_armable", False) is True,
                    )
                    last_health = sample
                    # Every reference track is local NED, so global position is
                    # advisory. PX4 armability is still a required preflight
                    # signal: numeric local coordinates alone are not proof that
                    # the estimator considers them stable enough for flight.
                    if sample.home_position_ok and sample.local_position_ok and sample.armable:
                        return sample
                raise RuntimeError("PX4 health stream ended before the vehicle became ready")
        except TimeoutError:
            observed = _health_payload(last_health)
            details = ", ".join(f"{name}={str(value).lower()}" for name, value in observed.items())
            raise TimeoutError(
                f"PX4 readiness timeout after {timeout_seconds}s; last_health: {details}"
            ) from None

    # 功能：
    #   有界等待飞控解锁回执；回执丢失时不能推断飞控没有解锁。
    # 输入：
    #   self：经过起飞准备的客户端。
    # 输出：
    #   None：实际命令回执或异常。
    async def arm(self) -> None:
        self._flight_command_requested = True
        await asyncio.wait_for(self._require_system().action.arm(), CLEANUP_COMMAND_TIMEOUT_SECONDS)

    # 功能：
    #   从飞控实际解锁状态确认可以进行起飞前恢复，不以本地命令标志代替物理状态。
    # 输入：
    #   self：已经连接且尚未请求解锁的客户端。
    # 输出：
    #   receipt：飞控明确报告未解锁时的检查回执。
    async def verify_disarmed_before_preflight(self) -> dict:
        if getattr(self, "_flight_command_requested", False) is not False:
            raise RuntimeError("PREFLIGHT_RECOVERY_FORBIDDEN_AFTER_MOTION_REQUEST")
        stream = self._require_system().telemetry.armed()
        try:
            armed = await asyncio.wait_for(anext(stream), timeout=2.)
            if armed is not False:
                raise RuntimeError("PREFLIGHT_VEHICLE_ALREADY_ARMED_OR_STATE_INVALID")
            receipt = {"disarmed": True, "confirmed_at_unix_ms": int(time.time()*1000)}
            return receipt
        finally:
            await asyncio.wait_for(stream.aclose(), timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS)

    # 功能：
    #   校验并发送 NED 位置和航向设定值，保留飞控的位置控制模式。
    # 输入：
    #   setpoint：北、东、向下米数及航向角。
    # 输出：
    #   None：发送回执，不是实际到达证明。
    async def set_position_ned(self, setpoint: Setpoint) -> None:
        _validate_control_setpoint(setpoint)
        await self._require_system().offboard.set_position_ned(
            self._position_cls(setpoint.north_m, setpoint.east_m, setpoint.down_m, setpoint.yaw_deg)
        )

    # 功能：
    #   同时发送位置及速度前馈；本地驾驶策略需要纯速度时应调用 set_velocity_ned。
    # 输入：
    #   setpoint：有效位置设定值。
    #   velocity：有效 NED 速度前馈。
    # 输出：
    #   None：真实 MAVSDK 组合指令的发送结果。
    async def set_position_velocity_ned(
        self,
        setpoint: Setpoint,
        velocity: VelocitySetpoint,
    ) -> None:
        _validate_control_setpoint(setpoint)
        _validate_control_setpoint(velocity)
        await self._require_system().offboard.set_position_velocity_ned(
            self._position_cls(
                setpoint.north_m,
                setpoint.east_m,
                setpoint.down_m,
                setpoint.yaw_deg,
            ),
            self._velocity_cls(
                velocity.north_m_s,
                velocity.east_m_s,
                velocity.down_m_s,
                velocity.yaw_deg,
            ),
        )

    # 功能：
    #   发送不含位置目标的三轴速度指令，让本地模型通过速度幅度实际驾驶无人机。
    # 输入：
    #   velocity：NED 速度与航向，安全范围由上游仲裁后在此校验有限性。
    # 输出：
    #   None：传输回执；姿态与电机闭环仍由飞控执行。
    async def set_velocity_ned(self, velocity: VelocitySetpoint) -> None:
        """Publish a true velocity-mode Offboard setpoint.

        PX4 selects the active multicopter controller from the non-NaN fields
        in the TrajectorySetpoint.  The combined position/velocity API keeps
        the position controller active and treats velocity only as
        feed-forward.  Local joystick-like control therefore uses this
        velocity-only transport and leaves position unset inside MAVSDK.
        """

        _validate_control_setpoint(velocity)
        await self._require_system().offboard.set_velocity_ned(
            self._velocity_cls(
                velocity.north_m_s,
                velocity.east_m_s,
                velocity.down_m_s,
                velocity.yaw_deg,
            )
        )

    # 功能：
    #   进入 Offboard 模式，将飞控拒绝原因转为运行异常。
    # 输入：
    #   self：已经预发送有效设定值的连接。
    # 输出：
    #   None：模式切换回执或异常。
    async def start_offboard(self) -> None:
        self._flight_command_requested = True
        system = self._require_system()
        try:
            await asyncio.wait_for(system.offboard.start(), CLEANUP_COMMAND_TIMEOUT_SECONDS)
        except self._offboard_error_cls as exc:
            raise RuntimeError(f"offboard start failed: {exc}") from exc

    # 功能：
    #   有界请求退出 Offboard；没有建立连接时无需发送停控请求。
    # 输入：
    #   self：当前飞控连接。
    # 输出：
    #   None：模式退出回执或异常。
    async def stop_offboard(self) -> None:
        if self._system is None:
            return
        try:
            await asyncio.wait_for(self._system.offboard.stop(), CLEANUP_COMMAND_TIMEOUT_SECONDS)
        except self._offboard_error_cls as exc:
            raise RuntimeError(f"offboard stop failed: {exc}") from exc

    # 功能：
    #   有界发送降落请求，不把接收命令解释为已经触地。
    # 输入：
    #   self：可发送动作的当前飞控连接。
    # 输出：
    #   None：降落命令回执或异常。
    async def land(self) -> None:
        await asyncio.wait_for(
            self._require_system().action.land(), CLEANUP_COMMAND_TIMEOUT_SECONDS
        )

    # 功能：
    #   等待原生落地枚举明确进入 ON_GROUND，不能用命令成功或字符串模糊匹配替代。
    # 输入：
    #   timeout_seconds：原生状态观察的最大秒数。
    # 输出：
    #   observation：明确的落地枚举及确认标记。
    async def wait_until_landed(self, timeout_seconds: float) -> dict[str, Any]:
        timeout_seconds = _timeout_budget(timeout_seconds)

        # 功能：
        #   消费本次落地状态流，只有准确的 ON_GROUND 名称才能完成等待。
        # 输入：
        #   无显式参数；使用当前客户端的 landed_state 订阅。
        # 输出：
        #   observation：实际收到的落地确认。
        async def _wait() -> dict[str, Any]:
            stream = self._require_system().telemetry.landed_state()
            try:
                async for landed_state in stream:
                    raw_name = getattr(landed_state, "name", None)
                    state_name = raw_name if isinstance(raw_name, str) else landed_state
                    if state_name == "ON_GROUND":
                        observation = {"state": state_name, "confirmed": True}
                        return observation
            finally:
                await stream.aclose()
            raise RuntimeError("PX4 landed-state stream ended before ON_GROUND")

        try:
            observation = await asyncio.wait_for(_wait(), timeout=timeout_seconds)
            return observation
        except TimeoutError:
            raise TimeoutError(
                f"PX4 landing confirmation timeout after {timeout_seconds:g}s"
            ) from None

    # 功能：
    #   限时读取飞控浮点参数，拒绝非法参数名或非有限读回。
    # 输入：
    #   name：PX4 参数标识。
    # 输出：
    #   value：实际参数值。
    async def get_param_float(self, name: str) -> float:
        _validate_parameter_name(name)
        value = await asyncio.wait_for(
            self._require_system().param.get_param_float(name), CLEANUP_COMMAND_TIMEOUT_SECONDS
        )
        value = _native_number(value, name)
        return value

    # 功能：
    #   验证后限时提交浮点参数，调用成功只表示写入回执。
    # 输入：
    #   name：PX4 参数标识。
    #   value：有限数值。
    # 输出：
    #   None：写入完成，业务上的精确读回由上层继续验证。
    async def set_param_float(self, name: str, value: float) -> None:
        _validate_parameter_name(name)
        value = _native_number(value, name)
        await asyncio.wait_for(
            self._require_system().param.set_param_float(name, value),
            CLEANUP_COMMAND_TIMEOUT_SECONDS,
        )

    # 功能：
    #   限时读取整型参数并验证有符号三十二位范围。
    # 输入：
    #   name：PX4 参数标识。
    # 输出：
    #   value：实际整型参数值。
    async def get_param_int(self, name: str) -> int:
        _validate_parameter_name(name)
        value = await asyncio.wait_for(
            self._require_system().param.get_param_int(name), CLEANUP_COMMAND_TIMEOUT_SECONDS
        )
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError("PX4 integer parameter readback is invalid")
        return value

    # 功能：
    #   限时提交严格整型参数，拒绝布尔值和静默截断的浮点值。
    # 输入：
    #   name：PX4 参数标识。
    #   value：有符号三十二位整数。
    # 输出：
    #   None：写入完成，精确读回由上层验证。
    async def set_param_int(self, name: str, value: int) -> None:
        _validate_parameter_name(name)
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError("PX4 integer parameter is invalid")
        await asyncio.wait_for(
            self._require_system().param.set_param_int(name, value), CLEANUP_COMMAND_TIMEOUT_SECONDS
        )

    # 功能：
    #   向指定相机组件发送拍照或录像控制，明确区分命令确认与图像产物。
    # 输入：
    #   parameters：command 动作名称及 component_id 组件编号。
    # 输出：
    #   result：真实 MAVLink 命令回执信息，不是照片或录像完成证据。
    async def execute_camera_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        command = parameters["command"]
        component_id = parameters["component_id"]
        if type(component_id) is not int or not 0 <= component_id <= 255:
            raise ValueError("camera component identity is invalid")
        camera = self._require_system().camera
        if command == "take_photo":
            await asyncio.wait_for(camera.take_photo(component_id), CLEANUP_COMMAND_TIMEOUT_SECONDS)
        elif command == "start_video":
            await asyncio.wait_for(
                camera.start_video(component_id), CLEANUP_COMMAND_TIMEOUT_SECONDS
            )
        elif command == "stop_video":
            await asyncio.wait_for(camera.stop_video(component_id), CLEANUP_COMMAND_TIMEOUT_SECONDS)
        else:
            raise RuntimeError(f"unsupported camera command: {command}")
        result = {
            "confirmed": True,
            "transport": "mavsdk-camera",
            "component_id": component_id,
            "command": command,
            "confirmation": "MAVLink camera command acknowledged",
        }
        return result

    # 功能：
    #   复用同帧实体采样，只返回指定模型的完整位姿。
    # 输入：
    #   world_name、model_name：当前世界与实体的精确名称。
    #   timeout_seconds：最多等待秒数。
    # 输出：
    #   pose：实际模型位置和单位四元数。
    async def _sample_named_gazebo_pose(
        self,
        *,
        world_name: str,
        model_name: str,
        timeout_seconds: float = 10.0,
    ) -> GazeboModelPose:
        pose = (
            await self._sample_named_gazebo_poses(
                world_name=world_name,
                model_names=(model_name,),
                timeout_seconds=timeout_seconds,
            )
        )[model_name]
        return pose

    # 功能：
    #   向执行层提供具名原生模型位姿，用于坐标转换和挂载验证。
    # 输入：
    #   world_name、model_name：本次世界及模型名称。
    #   timeout_seconds：读取秒数预算。
    # 输出：
    #   pose：校验后的世界位置和旋转。
    async def sample_gazebo_model_pose(
        self,
        *,
        world_name: str,
        model_name: str,
        timeout_seconds: float,
    ) -> GazeboModelPose:
        pose = await self._sample_named_gazebo_pose(
            world_name=world_name,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )
        return pose

    # 功能：
    #   从同一 Pose_V 消息读取全部指定实体，失败或结束均释放本次订阅。
    # 输入：
    #   world_name：绑定世界。
    #   model_names：一到八个不重复的模型名称。
    #   timeout_seconds：采样预算。
    #   gazebo_node：可复用的原生节点，None 时建立新节点。
    # 输出：
    #   poses：实体名称到真实位姿的映射，不跨帧拼接。
    async def _sample_named_gazebo_poses(
        self,
        *,
        world_name: str,
        model_names: tuple[str, ...],
        timeout_seconds: float = 10.0,
        gazebo_node: Any | None = None,
    ) -> dict[str, GazeboModelPose]:
        from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions

        timeout_seconds = _timeout_budget(timeout_seconds)
        _validate_gazebo_entity(world_name)
        if not 1 <= len(model_names) <= 8 or len(set(model_names)) != len(model_names):
            raise ValueError("Gazebo model selection must contain unique identities")
        for model_name in model_names:
            _validate_gazebo_entity(model_name)
        GazeboNode, _, Pose_V, _, _, _ = _gazebo_transport_bindings()
        node = gazebo_node or GazeboNode()
        topic = f"/world/{world_name}/dynamic_pose/info"
        loop = asyncio.get_running_loop()
        result: asyncio.Future[dict[str, GazeboModelPose]] = loop.create_future()
        required = set(model_names)

        # 功能：
        #   在所属事件循环完成采样，迟到回调不能覆写已有结果。
        # 输入：
        #   poses：完整同帧位姿集合或解析异常。
        # 输出：
        #   None：一次性完成外层 result。
        def resolve(poses: dict[str, GazeboModelPose] | Exception) -> None:
            if not result.done():
                if isinstance(poses, Exception):
                    result.set_exception(poses)
                else:
                    result.set_result(poses)

        # 功能：
        #   在传输线程校验实体数量、重复身份和数值，再安全投递到事件循环。
        # 输入：
        #   message：原生 Pose_V 消息。
        # 输出：
        #   None：完整帧触发结果，缺失实体继续等待，损坏数据触发失败。
        def on_pose(message: Any) -> None:
            poses: dict[str, GazeboModelPose] = {}
            try:
                if len(message.pose) > 100_000:
                    raise RuntimeError("Gazebo pose vector exceeds the entity budget")
                for raw_pose in message.pose:
                    if raw_pose.name not in required:
                        continue
                    if raw_pose.name in poses:
                        raise RuntimeError("Gazebo pose vector repeats a selected model")
                    poses[raw_pose.name] = GazeboModelPose(
                        raw_pose.position.x,
                        raw_pose.position.y,
                        raw_pose.position.z,
                        raw_pose.orientation.x,
                        raw_pose.orientation.y,
                        raw_pose.orientation.z,
                        raw_pose.orientation.w,
                    )
                if required.issubset(poses) and not loop.is_closed():
                    loop.call_soon_threadsafe(resolve, poses)
            except Exception as error:
                if not loop.is_closed():
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(resolve, error)

        subscriptions = GazeboSubscriptions(node)
        try:
            subscriptions.subscribe(Pose_V, topic, on_pose)
            poses = await asyncio.wait_for(result, timeout=timeout_seconds)
            return poses
        except TimeoutError as error:
            raise RuntimeError(
                "Gazebo dynamic-pose discovery timed out for models: " + ", ".join(model_names)
            ) from error
        finally:
            primary = sys.exception()
            result.cancel()
            summary = subscriptions.close()
            if summary["complete"] is not True:
                message = "Gazebo dynamic-pose subscription cleanup failed"
                if primary is not None:
                    primary.add_note(message)
                else:
                    raise RuntimeError(message)

    # 功能：
    #   为显式模拟工作人员挂载提交载荷位姿；此接口不能当作无人机运动控制。
    # 输入：
    #   world_name、model_name：当前世界及待移动的已分离载荷名称。
    #   pose：有限位置和单位旋转。
    #   timeout_seconds：原生服务请求预算。
    #   gazebo_node：可复用节点。
    # 输出：
    #   result：服务接受回执；实际位置必须另行读回。
    async def _set_named_gazebo_pose(
        self,
        *,
        world_name: str,
        model_name: str,
        pose: GazeboModelPose,
        timeout_seconds: float = 3.0,
        gazebo_node: Any | None = None,
    ) -> dict[str, Any]:
        timeout_seconds = _timeout_budget(timeout_seconds)
        _validate_gazebo_entity(world_name)
        _validate_gazebo_entity(model_name)
        pose.__post_init__()
        GazeboNode, Pose, _, Boolean, _, _ = _gazebo_transport_bindings()
        node = gazebo_node or GazeboNode()
        request = Pose()
        request.name = model_name
        request.position.x = pose.x
        request.position.y = pose.y
        request.position.z = pose.z
        request.orientation.x = pose.qx
        request.orientation.y = pose.qy
        request.orientation.z = pose.qz
        request.orientation.w = pose.qw
        service_name = f"/world/{world_name}/set_pose"
        # Python 取消不能杀掉正在执行的原生请求；持有线程任务并等原生超时结束后再退出。
        operation = asyncio.create_task(
            asyncio.to_thread(
                node.request,
                service_name,
                request,
                Pose,
                Boolean,
                max(1, int(timeout_seconds * 1_000)),
            )
        )
        try:
            requested, response = await asyncio.wait_for(
                asyncio.shield(operation),
                timeout=timeout_seconds + 1.0,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"Gazebo set-pose service timed out for model: {model_name}"
            ) from error
        finally:
            primary = sys.exception()
            try:
                await asyncio.shield(operation)
            except (Exception, asyncio.CancelledError):
                if primary is None:
                    raise
        if requested is not True or response.data is not True:
            raise RuntimeError(f"Gazebo payload mount-pose command failed: {response}")
        result = {
            "service": service_name,
            "request_model": model_name,
            "accepted": True,
            "raw_response": str(response),
        }
        return result

    # 功能：
    #   模拟工作人员把已分离物品放到挂载点，发出连接命令后禁止再移动载荷补造成功。
    # 输入：
    #   parameters：车辆、载荷、挂点偏移、误差限及绑定摘要。
    #   gazebo_node：当前运行的原生节点。
    # 输出：
    #   alignment：设置前后位姿、实际读回偏差和挂载绑定证据。
    async def _align_payload_to_mount(
        self,
        parameters: dict[str, Any],
        *,
        gazebo_node: Any | None = None,
    ) -> dict[str, Any]:
        world_name = os.environ.get("PX4_GAZEBO_WORLD_NAME", "").strip()
        if not world_name:
            raise RuntimeError("Gazebo payload alignment has no bound world name")
        vehicle_model_name = parameters["vehicle_model_name"]
        payload_model_name = parameters["payload_model_name"]
        _validate_gazebo_entity(vehicle_model_name)
        _validate_gazebo_entity(payload_model_name)
        if vehicle_model_name == payload_model_name:
            raise ValueError("payload cannot be the flight vehicle")
        if re.fullmatch(r"[0-9a-f]{64}", parameters["payload_mount_binding_sha256"]) is None:
            raise ValueError("payload mount binding digest is invalid")
        raw_offset = parameters["payload_mount_offset_model_m"]
        if not isinstance(raw_offset, list) or len(raw_offset) != 3:
            raise RuntimeError("Gazebo payload mount offset is invalid")
        mount_offset = tuple(_native_number(value, "payload mount offset") for value in raw_offset)
        if not all(math.isfinite(value) for value in mount_offset):
            raise RuntimeError("Gazebo payload mount offset is non-finite")
        maximum_error_m = _native_number(
            parameters["payload_mount_max_alignment_error_m"], "alignment error"
        )
        if not math.isfinite(maximum_error_m) or not 0.0 < maximum_error_m <= 0.1:
            raise RuntimeError("Gazebo payload mount alignment tolerance is invalid")

        sampled_poses = await self._sample_named_gazebo_poses(
            world_name=world_name,
            model_names=(vehicle_model_name, payload_model_name),
            gazebo_node=gazebo_node,
        )
        vehicle_pose = sampled_poses[vehicle_model_name]
        payload_pose_before = sampled_poses[payload_model_name]
        target = payload_pose_before
        service: dict[str, Any] = {}
        alignment_error_m = math.inf
        observed_payload_pose = payload_pose_before
        # A successful set_pose service response only acknowledges the queued
        # request; the simulator may not have applied it yet. Before publishing
        # *any* attach command it is still safe to move the detached child, so
        # repeatedly bind the target to the newest vehicle pose and require an
        # independent dynamic-pose readback inside the declared tolerance.
        maximum_alignment_attempts = 6
        alignment_attempts_completed = 0
        for alignment_attempt in range(1, maximum_alignment_attempts + 1):
            alignment_attempts_completed = alignment_attempt
            sampled_poses = await self._sample_named_gazebo_poses(
                world_name=world_name,
                model_names=(vehicle_model_name, payload_model_name),
                gazebo_node=gazebo_node,
            )
            vehicle_pose = sampled_poses[vehicle_model_name]
            offset_world = _rotate_gazebo_model_vector(vehicle_pose, mount_offset)
            target = GazeboModelPose(
                x=vehicle_pose.x + offset_world[0],
                y=vehicle_pose.y + offset_world[1],
                z=vehicle_pose.z + offset_world[2],
                qx=vehicle_pose.qx,
                qy=vehicle_pose.qy,
                qz=vehicle_pose.qz,
                qw=vehicle_pose.qw,
            )
            service = await self._set_named_gazebo_pose(
                world_name=world_name,
                model_name=payload_model_name,
                pose=target,
                gazebo_node=gazebo_node,
            )
            readback = await self._sample_named_gazebo_poses(
                world_name=world_name,
                model_names=(vehicle_model_name, payload_model_name),
                gazebo_node=gazebo_node,
            )
            readback_vehicle = readback[vehicle_model_name]
            observed_payload_pose = readback[payload_model_name]
            readback_offset_world = _rotate_gazebo_model_vector(readback_vehicle, mount_offset)
            expected_readback_position = (
                readback_vehicle.x + readback_offset_world[0],
                readback_vehicle.y + readback_offset_world[1],
                readback_vehicle.z + readback_offset_world[2],
            )
            alignment_error_m = math.dist(
                expected_readback_position,
                (
                    observed_payload_pose.x,
                    observed_payload_pose.y,
                    observed_payload_pose.z,
                ),
            )
            if alignment_error_m <= maximum_error_m:
                break
        else:
            raise RuntimeError(
                "Gazebo payload did not reach the declared mount before attachment: "
                f"observed={alignment_error_m:.6f}m limit={maximum_error_m:.6f}m "
                f"attempts={maximum_alignment_attempts}"
            )
        return {
            "binding_sha256": str(parameters["payload_mount_binding_sha256"]),
            "vehicle_model_name": vehicle_model_name,
            "payload_model_name": payload_model_name,
            "mount_offset_model_m": list(mount_offset),
            "payload_pose_before_world_enu": {
                "x": payload_pose_before.x,
                "y": payload_pose_before.y,
                "z": payload_pose_before.z,
            },
            "target_payload_pose_world_enu": {
                "x": target.x,
                "y": target.y,
                "z": target.z,
                "qx": target.qx,
                "qy": target.qy,
                "qz": target.qz,
                "qw": target.qw,
            },
            "set_pose": service,
            "set_pose_attempts": alignment_attempts_completed,
            "pre_attachment_pose_readback": {
                "payload_position_world_enu_m": [
                    observed_payload_pose.x,
                    observed_payload_pose.y,
                    observed_payload_pose.z,
                ],
                "alignment_error_m": alignment_error_m,
                "maximum_alignment_error_m": maximum_error_m,
                "accepted": True,
            },
        }

    # 功能：
    #   为同一运行和主题保留关节事件订阅，所有事件都更新状态，坏事件撤销既有确认。
    # 输入：
    #   output_topic：当前载荷的精确状态主题。
    # 输出：
    #   observer：本事件循环持有的状态与订阅，不从普通布尔缓存恢复确认。
    def _ensure_payload_observer(self, output_topic: str) -> dict[str, Any]:
        from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions

        _validate_gazebo_topic(output_topic)
        loop = asyncio.get_running_loop()
        binding = (os.environ.get("GZ_PARTITION", ""), os.environ.get("PX4_GAZEBO_WORLD_NAME", ""))
        observers = getattr(self, "_payload_observers", None)
        if observers is None:
            observers = self._payload_observers = {}
        if output_topic in observers:
            observer = observers[output_topic]
            if (
                observer["loop"] is not loop
                or observer["binding"] != binding
                or not observer["alive"]
            ):
                raise RuntimeError("Gazebo payload observer belongs to a different runtime")
            return observer
        if len(observers) >= 8:
            raise RuntimeError("Gazebo payload observer capacity exceeded")
        GazeboNode, _, _, _, _, StringMsg = _gazebo_transport_bindings()
        node = GazeboNode()
        subscriptions = GazeboSubscriptions(node)
        observer = {
            "node": node,
            "subscriptions": subscriptions,
            "loop": loop,
            "binding": binding,
            "alive": True,
            "detached": None,
            "sequence": 0,
            "observed_at_unix_ms": None,
            "source": None,
            "error": None,
        }

        # 功能：
        #   在所属事件循环更新关节状态，关闭后排队的回调不得再改变观察记录。
        # 输入：
        #   detached：事件的明确分离状态，None 表示收到非法事件。
        #   observed_at：回调收到该事件的 UNIX 毫秒时刻。
        # 输出：
        #   None：就地更新订阅持有的最新事件与序号。
        def deliver(detached: bool | None, observed_at: int) -> None:
            if not observer["alive"]:
                return
            observer.update(
                detached=detached,
                observed_at_unix_ms=observed_at,
                source="gazebo-detachable-joint-event",
                error=None if detached is not None else "invalid-joint-state",
            )
            observer["sequence"] += 1

        # 功能：
        #   将传输线程的事件转换成独立标量，再交给所属事件循环处理。
        # 输入：
        #   message：原生 StringMsg，只有 attached 和 detached 是有效状态。
        # 输出：
        #   None：不在传输线程直接修改业务状态。
        def on_state(message: Any) -> None:
            raw = message.data
            detached = (
                (raw == "detached")
                if type(raw) is str and raw in {"attached", "detached"}
                else None
            )
            if observer["alive"] and not loop.is_closed():
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(deliver, detached, int(time.time() * 1000))

        try:
            subscriptions.subscribe(StringMsg, output_topic, on_state)
        except BaseException:
            observer["alive"] = False
            subscriptions.close()
            raise
        observers[output_topic] = observer
        return observer

    # 功能：
    #   以当前运行摘要绑定的原生起飞前回执衔接持续订阅，保留原观察时刻而不伪造新样本。
    # 输入：
    #   self：刚连接且尚未解锁的客户端；回执路径和摘要由本次仿真启动器提供。
    # 输出：
    #   None：验证通过时建立唯一主题的初始已知状态，缺少回执则保持未知。
    async def _prime_payload_observer(self) -> None:
        from dronedream_agent_core.plugin_files import read_plugin_file
        from dronedream_plugin_sdk.protocol import decode_json

        path_text = os.environ.get("PX4_GAZEBO_PAYLOAD_PREFLIGHT_PATH", "")
        expected = os.environ.get("PX4_GAZEBO_PAYLOAD_PREFLIGHT_SHA256", "")
        if not path_text and not expected:
            return
        if not path_text or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError("Gazebo payload preflight receipt binding is incomplete")
        raw = read_plugin_file(Path(path_text), limit=65_536)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("Gazebo payload preflight receipt changed")
        receipt = decode_json(raw.decode("utf-8"), limit=65_536)
        now = int(time.time() * 1000)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != "dronedream.payload-preflight-observation"
            or receipt.get("partition") != os.environ.get("GZ_PARTITION", "")
            or receipt.get("world") != os.environ.get("PX4_GAZEBO_WORLD_NAME", "")
            or not receipt.get("partition")
            or not receipt.get("world")
            or type(receipt.get("observed_at_unix_ms")) is not int
            or not 0 <= now - receipt["observed_at_unix_ms"] <= 60_000
        ):
            raise RuntimeError(
                "Gazebo payload preflight receipt is stale or belongs to another runtime"
            )
        observation = receipt.get("observation")
        if (
            not isinstance(observation, dict)
            or observation.get("confirmed") is not True
            or observation.get("detached") is not True
        ):
            raise RuntimeError("Gazebo payload preflight detached state is not confirmed")
        observer = self._ensure_payload_observer(observation.get("output_topic"))
        # 初态只衔接尚无事件的订阅；任何已经收到的新事件优先，不能被旧回执覆盖。
        if observer["sequence"] == 0:
            observer.update(
                detached=True,
                source="verified-preflight-detach-readback",
                observed_at_unix_ms=receipt["observed_at_unix_ms"],
            )

    # 功能：
    #   串行执行载荷动作，拒绝并发挂载与分离，并在所有退出路径释放动作所有权。
    # 输入：
    #   parameters：经运行计划授权的协议、主题、动作和载荷绑定。
    # 输出：
    #   result：实际指令回执与原生关节状态证据。
    async def execute_payload_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        from dronedream_plugin_sdk.protocol import copy_json

        parameters = copy_json(parameters, limit=65_536)
        if getattr(self, "_payload_command_active", False):
            raise RuntimeError("Gazebo payload command is already in progress")
        self._payload_command_active = True
        try:
            result = await self._execute_payload_command(parameters)
            return result
        finally:
            self._payload_command_active = False

    # 功能：
    #   发送已授权的执行器或关节动作；位置贴合不能替代真实关节事件，发布后不再搬动载荷。
    # 输入：
    #   parameters：由外层复制并取得串行所有权的动作参数。
    # 输出：
    #   result：本次实际确认的动作和挂载姿态检查结果。
    async def _execute_payload_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        protocol = parameters["protocol"]
        operation = parameters["operation"]
        if operation not in ("attach", "detach"):
            raise ValueError("unsupported payload operation")
        if protocol == "mavsdk-actuator":
            actuator_index = parameters["actuator_index"]
            actuator_value = _native_number(parameters["actuator_value"], "actuator_value")
            if (
                type(actuator_index) is not int
                or not 1 <= actuator_index <= 6
                or not -1 <= actuator_value <= 1
            ):
                raise ValueError("payload actuator command is outside the MAVSDK contract")
            await self._require_system().action.set_actuator(actuator_index, actuator_value)
            return {
                "confirmed": True,
                "transport": protocol,
                "operation": operation,
                "actuator_index": actuator_index,
                "actuator_value": actuator_value,
                "confirmation": "MAVLink actuator command acknowledged",
            }
        if protocol != "gazebo-transport":
            raise RuntimeError(f"unsupported payload protocol: {protocol}")
        _, _, _, _, Empty, _ = _gazebo_transport_bindings()
        topic = parameters["topic"]
        output_topic = parameters["output_topic"]
        _validate_gazebo_topic(topic)
        _validate_gazebo_topic(output_topic)
        if topic == output_topic:
            raise ValueError("payload command and state topics must differ")
        observer = self._ensure_payload_observer(output_topic)
        node = observer["node"]
        expected_detached = operation == "detach"
        loop = asyncio.get_running_loop()
        publisher = node.advertise(topic, Empty)
        connection_deadline = time.monotonic() + 3.0
        while not publisher.has_connections() and time.monotonic() < connection_deadline:
            await asyncio.sleep(0.01)
        if not publisher.has_connections():
            raise RuntimeError(f"Gazebo payload command has no subscriber: {topic}")
        # The DetachableJoint state topic is event-driven: it publishes when
        # attachment changes, not continuously.  Command discovery can finish
        # before the newly-created state subscription is matched to the
        # plugin's publisher.  Publishing in that gap can attach successfully
        # while permanently losing the only positive readback.  Wait for the
        # output publisher to be discoverable and give the reverse subscription
        # match one bounded settling interval before the first command.
        topic_info = getattr(node, "topic_info", None)
        if callable(topic_info):
            state_discovery_deadline = time.monotonic() + 3.0
            state_publishers: list[Any] = []
            while time.monotonic() < state_discovery_deadline:
                state_publishers, _ = topic_info(output_topic)
                if state_publishers:
                    break
                await asyncio.sleep(0.01)
            if not state_publishers:
                raise RuntimeError(f"Gazebo payload state publisher is unavailable: {output_topic}")
        await asyncio.sleep(GAZEBO_PAYLOAD_STATE_DISCOVERY_SETTLE_SECONDS)
        alignment: dict[str, Any] | None = None
        alignment_attempts = 0
        publish_attempts = 0
        try:
            if operation == "attach":
                if observer["detached"] is not True or observer["error"] is not None:
                    raise RuntimeError("payload must be observed detached before mount alignment")
                alignment = await asyncio.wait_for(
                    self._align_payload_to_mount(parameters, gazebo_node=node), timeout=15.0
                )
                alignment_attempts += 1
            # DetachableJoint consumes the request on Gazebo's simulation
            # update thread and only then publishes its state transition.  A
            # render-heavy world can advance that thread well below wall-clock
            # rate, and a single best-effort Empty publication made a real
            # attachment intermittently disappear.  Retain one subscription,
            # repeat the idempotent request inside a strict deadline, and still
            # require the plugin's own attached/detached state as proof.
            state_deadline = loop.time() + GAZEBO_PAYLOAD_STATE_TRANSITION_TIMEOUT_SECONDS
            initial_sequence = observer["sequence"]
            # 发布后即使回执丢失，物理动作也可能已经执行；旧分离状态不能授权再次搬动载荷。
            observer["detached"] = None
            observer["error"] = None
            while loop.time() < state_deadline:
                if publisher.publish(Empty()) is False:
                    raise RuntimeError("Gazebo payload command publication failed")
                publish_attempts += 1
                remaining = state_deadline - loop.time()
                if remaining <= 0.0:
                    break
                await asyncio.sleep(min(GAZEBO_PAYLOAD_COMMAND_RETRY_SECONDS, remaining))
                if observer["error"] is not None:
                    raise RuntimeError("Gazebo payload state event is invalid")
                if (
                    observer["sequence"] > initial_sequence
                    and observer["detached"] is expected_detached
                ):
                    break
            if (
                observer["sequence"] <= initial_sequence
                or observer["detached"] is not expected_detached
            ):
                raise TimeoutError
            observed_detached = observer["detached"]
        except TimeoutError as error:
            raise RuntimeError(
                "Gazebo payload state readback timed out after "
                f"{publish_attempts} bounded command publications within "
                f"{GAZEBO_PAYLOAD_STATE_TRANSITION_TIMEOUT_SECONDS:g} seconds: "
                f"{output_topic}; alignment_attempts={alignment_attempts}; "
                "post_publication_realignments=0; native joint event required"
            ) from error
        if observed_detached != expected_detached:
            raise RuntimeError(
                "Gazebo payload state readback did not confirm the requested attachment state"
            )
        result = {
            "confirmed": True,
            "transport": protocol,
            "operation": operation,
            "topic": topic,
            "output_topic": output_topic,
            "detached": expected_detached,
            "command_publish_attempts": publish_attempts,
            "state_transition_timeout_seconds": (GAZEBO_PAYLOAD_STATE_TRANSITION_TIMEOUT_SECONDS),
            "post_publication_realignments": 0,
            "state_readback_source": "gazebo-detachable-joint-event",
            "observed_at_unix_ms": observer["observed_at_unix_ms"],
            "raw_state": f'data: "{"detached" if observed_detached else "attached"}"',
        }
        if alignment is not None:
            alignment["alignment_attempts"] = alignment_attempts
            world_name = os.environ.get("PX4_GAZEBO_WORLD_NAME", "").strip()
            sampled_poses = await self._sample_named_gazebo_poses(
                world_name=world_name,
                model_names=(
                    str(parameters["vehicle_model_name"]),
                    str(parameters["payload_model_name"]),
                ),
                gazebo_node=node,
            )
            vehicle_pose = sampled_poses[str(parameters["vehicle_model_name"])]
            payload_pose = sampled_poses[str(parameters["payload_model_name"])]
            mount_offset = tuple(
                float(value) for value in parameters["payload_mount_offset_model_m"]
            )
            offset_world = _rotate_gazebo_model_vector(vehicle_pose, mount_offset)
            expected_position = (
                vehicle_pose.x + offset_world[0],
                vehicle_pose.y + offset_world[1],
                vehicle_pose.z + offset_world[2],
            )
            alignment_error_m = math.dist(
                expected_position,
                (payload_pose.x, payload_pose.y, payload_pose.z),
            )
            maximum_error_m = float(parameters["payload_mount_max_alignment_error_m"])
            if alignment_error_m > maximum_error_m:
                raise RuntimeError(
                    "Gazebo payload attachment pose exceeded the qualified mount tolerance: "
                    f"observed={alignment_error_m:.6f}m limit={maximum_error_m:.6f}m"
                )
            alignment["attachment_pose_readback"] = {
                "payload_position_world_enu_m": [
                    payload_pose.x,
                    payload_pose.y,
                    payload_pose.z,
                ],
                "expected_position_world_enu_m": list(expected_position),
                "alignment_error_m": alignment_error_m,
                "maximum_alignment_error_m": maximum_error_m,
                "accepted": True,
            }
            result["payload_mount_alignment"] = alignment
        return result

    # 功能：
    #   从本运行持续订阅读取最后确认的关节状态，保留事件年龄，未知状态不以环境默认值补齐。
    # 输入：
    #   output_topic：本次载荷状态的精确主题。
    #   timeout_seconds：首次未知状态的最大等待秒数。
    # 输出：
    #   result：最后原生确认、来源和原观察时刻；不是新的测量，也不是力学稳定性证明。
    async def sample_payload_state(
        self, output_topic: str, timeout_seconds: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + _timeout_budget(timeout_seconds)
        observer = self._ensure_payload_observer(output_topic)
        while observer["detached"] is None:
            if observer["error"] is not None or time.monotonic() >= deadline:
                raise RuntimeError("Gazebo payload state is unknown or invalid")
            await asyncio.sleep(min(0.02, max(0, deadline - time.monotonic())))
        topic_info = getattr(observer["node"], "topic_info", None)
        if callable(topic_info) and not topic_info(output_topic)[0]:
            observer["detached"] = None
            raise RuntimeError("Gazebo payload state publisher disappeared")
        age = int(time.time() * 1000) - observer["observed_at_unix_ms"]
        if age < 0:
            raise RuntimeError("Gazebo payload observation clock regressed")
        result = {
            "confirmed": True,
            "transport": "gazebo-transport",
            "output_topic": output_topic,
            "detached": observer["detached"],
            "state_source": observer["source"],
            "observed_at_unix_ms": observer["observed_at_unix_ms"],
            "state_age_ms": age,
            "observation_kind": "last-confirmed-event-with-live-subscription",
        }
        return result

    # 功能：
    #   设置并读回 PX4 避障接口开关；参数接受不代表外部避障算法已运行。
    # 输入：
    #   enabled：是否启用 COM_OBS_AVOID。
    # 输出：
    #   result：实际参数前值、后值及传输来源。
    async def execute_avoidance_command(self, enabled: bool) -> dict[str, Any]:
        if type(enabled) is not bool:
            raise ValueError("avoidance enable flag must be boolean")
        before = await self.get_param_int("COM_OBS_AVOID")
        requested = 1 if enabled else 0
        await self.set_param_int("COM_OBS_AVOID", requested)
        after = await self.get_param_int("COM_OBS_AVOID")
        if after != requested:
            raise RuntimeError(
                f"PX4 COM_OBS_AVOID readback mismatch: requested={requested}, observed={after}"
            )
        result = {
            "confirmed": True,
            "transport": "mavsdk-parameter",
            "parameter": "COM_OBS_AVOID",
            "before": before,
            "after": after,
            "enabled": enabled,
        }
        return result

    # 功能：
    #   从指定主题的完整消息读取精确目标实体，不借用另一模型的位置。
    # 输入：
    #   parameters：包含 target_pose_topic 和 target_model_name 的目标契约。
    # 输出：
    #   result：Gazebo 世界坐标位置及主题、实体来源。
    async def sample_gazebo_pose(self, parameters: dict[str, Any]) -> dict[str, float | str]:
        topic = parameters["target_pose_topic"]
        target_name = parameters["target_model_name"]
        _validate_gazebo_entity(target_name)
        payload = await _capture_gazebo_topic(topic, 2.0)
        pose = _parse_gazebo_model_pose(payload, target_name)
        result = {"x": pose.x, "y": pose.y, "z": pose.z, "topic": topic, "target_name": target_name}
        return result

    # 功能：
    #   等待无采集错误且年龄合法的电池样本，旧缓存不能在读取时重新获得有效期。
    # 输入：
    #   timeout_seconds：等待及最大样本年龄的秒数上限。
    # 输出：
    #   sample：经校验的百分比电量及电压。
    async def sample_battery(self, timeout_seconds: float) -> dict[str, float]:
        timeout_seconds = _timeout_budget(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            latest = self._dynamics_samples.get("battery")
            now = time.monotonic()
            if (
                latest is not None
                and 0 <= now - latest[1] <= timeout_seconds
                and "battery" not in self._dynamics_errors
            ):
                sample = _validated_battery_telemetry(
                    latest[0],
                    label="PX4 battery telemetry",
                )
                return sample
            remaining = deadline - now
            if remaining <= 0.0:
                raise TimeoutError(f"PX4 battery telemetry timeout after {timeout_seconds:g}s")
            await asyncio.sleep(min(0.05, remaining))

    # 功能：
    #   限时读取真实 GPS 卫星数和定位类型，供失效注入后的物理读回验证。
    # 输入：
    #   timeout_seconds：最大读取秒数。
    # 输出：
    #   sample：经过类型和值域检查的 GPS 状态。
    async def sample_gps_info(self, timeout_seconds: float) -> dict[str, int | str]:
        timeout_seconds = _timeout_budget(timeout_seconds)

        # 功能：
        #   取得 GPS 流首个有效样本，并在返回或取消时关闭本次独占流。
        # 输入：
        #   无显式参数；使用当前客户端的遥测连接。
        # 输出：
        #   sample：卫星数和原生定位类型。
        async def _sample() -> dict[str, int | str]:
            stream = self._require_system().telemetry.gps_info()
            try:
                async for gps_info in stream:
                    fix_type = gps_info.fix_type
                    fix_value = getattr(fix_type, "value", fix_type)
                    fix_name = str(getattr(fix_type, "name", fix_type))
                    sample = _validated_gps_telemetry(
                        {
                            "num_satellites": gps_info.num_satellites,
                            "fix_type": fix_value,
                            "fix_type_name": fix_name,
                        },
                        label="PX4 GPS telemetry",
                    )
                    return sample
            finally:
                await stream.aclose()
            raise RuntimeError("PX4 GPS info telemetry stream ended without a sample")

        try:
            sample = await asyncio.wait_for(_sample(), timeout=timeout_seconds)
            return sample
        except TimeoutError:
            raise TimeoutError(
                f"PX4 GPS info telemetry timeout after {timeout_seconds:g}s"
            ) from None

    # 功能：
    #   读取持续收集的 NED 状态，采集失败、未来时刻和过旧样本均不能继续提供有效位置。
    # 输入：
    #   timeout_seconds：等待及最大样本年龄的秒数上限。
    # 输出：
    #   sample：实际位置及速度，不通过目标设定值反推遥测。
    async def sample_position_velocity_ned(
        self,
        timeout_seconds: float,
    ) -> PositionVelocityNed:
        timeout_seconds = _timeout_budget(timeout_seconds)
        condition = self._position_velocity_condition
        if condition is None or self._position_velocity_task is None:
            raise RuntimeError("PX4 position/velocity telemetry collector is not running")
        deadline = time.monotonic() + timeout_seconds
        while True:
            async with condition:
                if self._position_velocity_error is not None:
                    raise RuntimeError(
                        "PX4 position/velocity telemetry collector failed"
                    ) from self._position_velocity_error
                latest = self._position_velocity_sample
                now = time.monotonic()
                if latest is not None and 0 <= now - latest[1] <= timeout_seconds:
                    sample = latest[0]
                    return sample
                remaining = deadline - now
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"PX4 position/velocity telemetry timeout after {timeout_seconds:g}s"
                    )
                try:
                    await asyncio.wait_for(condition.wait(), timeout=remaining)
                except TimeoutError:
                    raise TimeoutError(
                        f"PX4 position/velocity telemetry timeout after {timeout_seconds:g}s"
                    ) from None

    # 功能：
    #   从无错误的新鲜姿态流读取飞控航向，不用零度填补未知朝向。
    # 输入：
    #   timeout_seconds：等待及最大样本年龄的秒数上限。
    # 输出：
    #   heading_deg：PX4 NED 约定下以度表示的实际航向。
    async def sample_heading_deg(self, timeout_seconds: float) -> float:
        timeout_seconds = _timeout_budget(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            latest = self._dynamics_samples.get("attitude")
            now = time.monotonic()
            if (
                latest is not None
                and 0 <= now - latest[1] <= timeout_seconds
                and "attitude" not in self._dynamics_errors
            ):
                heading_deg = _native_number(
                    latest[0].get("yaw_deg"),
                    "PX4 attitude yaw_deg",
                )
                return heading_deg
            remaining = deadline - now
            if remaining <= 0.0:
                raise TimeoutError(f"PX4 attitude telemetry timeout after {timeout_seconds:g}s")
            await asyncio.sleep(min(0.02, remaining))

    # 功能：
    #   独立回收关节订阅、全部遥测任务和本客户端启动的服务；一项失败仍继续其余清理。
    # 输入：
    #   self：拥有这些资源的真实客户端，不按进程名清理其他实例。
    # 输出：
    #   None：全部完成时返回；残留任务或退出失败汇总为异常，不声称关闭成功。
    async def close(self) -> None:
        errors: list[BaseException] = []
        observers = getattr(self, "_payload_observers", {})
        for observer in observers.values():
            observer["alive"] = False
            try:
                summary = observer["subscriptions"].close()
                if summary["complete"] is not True:
                    raise RuntimeError(
                        "Gazebo payload subscriptions did not close: " + str(summary["errors"])
                    )
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
        self._payload_observers = {}
        tasks = set(getattr(self, "_dynamics_tasks", ()))
        telemetry_task = getattr(self, "_position_velocity_task", None)
        if telemetry_task is not None:
            tasks.add(telemetry_task)
        for task in tasks:
            task.cancel()
        pending = set(tasks)
        if tasks:
            try:
                done, pending = await asyncio.wait(tasks, timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS)
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        errors.append(task.exception())
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
        # 尚未退出的任务保留引用，禁止重连时把它们当作已关闭而再开并行订阅。
        self._dynamics_tasks = list(pending)
        self._position_velocity_task = telemetry_task if telemetry_task in pending else None
        if pending:
            errors.append(TimeoutError("PX4 telemetry tasks did not drain"))
        system = getattr(self, "_system", None)
        self._system = None
        self._mavsdk_server_port = None
        server_process = getattr(system, "_server_process", None)
        stop_server = getattr(system, "_stop_mavsdk_server", None)
        try:
            if callable(stop_server):
                stop_server()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        if isinstance(server_process, subprocess.Popen):
            try:
                if server_process.poll() is None:
                    server_process.terminate()
                try:
                    await asyncio.to_thread(server_process.wait, timeout=2.0)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    await asyncio.to_thread(server_process.wait, timeout=2.0)
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
        if errors:
            raise RuntimeError("PX4 client cleanup failed") from BaseExceptionGroup(
                "Owned cleanup failures", errors
            )


class FakeOffboardClient:
    """Explicit unit-test fixture, never a production transport or flight qualification source."""

    # 功能：
    #   初始化纯内存测试替身及可注入遥测，不能用于真实仿真验收。
    # 输入：
    #   self：独立的测试客户端实例。
    # 输出：
    #   None：建立测试命令记录、参数和样本队列。
    def __init__(self) -> None:
        self.connected = False
        self.armed = False
        self.offboard_started = False
        self.setpoints: list[Setpoint] = []
        self.velocity_setpoints: list[VelocitySetpoint] = []
        self.landed = False
        self.closed = False
        self.payload_detached = True
        self.int_params: dict[str, int] = {
            "SIM_GPS_USED": 10,
            "COM_OBS_AVOID": 0,
            "COM_OBL_RC_ACT": 0,
        }
        self.float_params: dict[str, float] = {
            "SIM_BAT_DRAIN": 60.0,
            "SIM_BAT_MIN_PCT": 50.0,
            "COM_OF_LOSS_T": 1.0,
        }
        self.battery_samples: list[dict[str, float]] = [
            {"remaining_percent": 100.0, "voltage_v": 16.8}
        ]
        self.gps_info_samples: list[dict[str, int | str]] = [
            {"num_satellites": 10, "fix_type": 3, "fix_type_name": "FIX_3D"}
        ]
        self.position_velocity_samples: list[PositionVelocityNed] = []
        self.heading_deg = 0.0
        self.dynamics_telemetry: dict[str, Any] = {
            "schema_version": "dronedream.px4-dynamics-telemetry.v1",
            "maximum_sample_age_seconds": 3.0,
            "ready_for_payload_inference": True,
            "sources": {
                "imu": {
                    "acceleration_forward_m_s2": 0.0,
                    "acceleration_right_m_s2": 0.0,
                    "acceleration_down_m_s2": -9.80665,
                    "angular_velocity_forward_rad_s": 0.0,
                    "angular_velocity_right_rad_s": 0.0,
                    "angular_velocity_down_rad_s": 0.0,
                    "timestamp_us": 1,
                    "sample_age_seconds": 0.0,
                },
                "attitude": {
                    "roll_deg": 0.0,
                    "pitch_deg": 0.0,
                    "yaw_deg": 0.0,
                    "timestamp_us": 1,
                    "sample_age_seconds": 0.0,
                },
                "battery": {
                    "remaining_percent": 1.0,
                    "voltage_v": 16.8,
                    "current_battery_a": 4.0,
                    "sample_age_seconds": 0.0,
                },
                "actuator_output": {
                    "active_mask": 15,
                    "actuator": [0.5, 0.5, 0.5, 0.5],
                    "sample_age_seconds": 0.0,
                },
            },
            "issue_codes": [],
        }
        self.gazebo_model_pose = GazeboModelPose(
            x=0.0,
            y=0.0,
            z=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        )
        self.gazebo_pose_samples: list[dict[str, float | str]] = [
            {"x": 2.0, "y": 2.0, "z": 1.0, "topic": "/target/pose", "target_name": "target"}
        ]

    # 功能：
    #   记录测试连接状态，不建立网络连接。
    # 输入：
    #   connection_url：仅用于接口一致性的测试地址。
    # 输出：
    #   None：connected 测试标志置真。
    async def connect(self, connection_url: str) -> None:
        _ = connection_url
        self.connected = True

    # 功能：
    #   向单元测试提供显式理想就绪夹具，不代表读取真实飞控。
    # 输入：
    #   timeout_seconds：保持接口一致的等待参数。
    # 输出：
    #   health：全部就绪的测试状态。
    async def wait_until_ready(
        self,
        timeout_seconds: float,
    ) -> TelemetryHealth:
        _ = timeout_seconds
        health = TelemetryHealth(
            connected=True,
            global_position_ok=True,
            home_position_ok=True,
            local_position_ok=True,
            armable=True,
        )
        return health

    # 功能：
    #   记录测试解锁调用，不向任何飞控发送命令。
    # 输入：
    #   self：本次测试客户端。
    # 输出：
    #   None：armed 标志置真。
    async def arm(self) -> None:
        self._flight_command_requested = True
        self.armed = True

    # 功能：
    #   保存测试位置命令，供断言检查发送顺序。
    # 输入：
    #   setpoint：本次位置和航向指令。
    # 输出：
    #   None：追加到测试命令列表。
    async def set_position_ned(self, setpoint: Setpoint) -> None:
        self.setpoints.append(setpoint)

    # 功能：
    #   同时记录位置目标和速度前馈的测试调用。
    # 输入：
    #   setpoint：测试位置目标。
    #   velocity：测试速度前馈。
    # 输出：
    #   None：两个列表分别记录本次值。
    async def set_position_velocity_ned(
        self,
        setpoint: Setpoint,
        velocity: VelocitySetpoint,
    ) -> None:
        self.setpoints.append(setpoint)
        self.velocity_setpoints.append(velocity)

    # 功能：
    #   记录纯速度测试命令，不生成位置目标。
    # 输入：
    #   velocity：测试速度及偏航值。
    # 输出：
    #   None：仅追加速度命令列表。
    async def set_velocity_ned(self, velocity: VelocitySetpoint) -> None:
        self.velocity_setpoints.append(velocity)

    # 功能：
    #   记录测试进入外部控制模式。
    # 输入：
    #   self：本次测试实例。
    # 输出：
    #   None：外部控制标志置真。
    async def start_offboard(self) -> None:
        self._flight_command_requested = True
        self.offboard_started = True

    # 功能：
    #   记录测试退出外部控制模式。
    # 输入：
    #   self：本次测试实例。
    # 输出：
    #   None：外部控制标志清除。
    async def stop_offboard(self) -> None:
        self.offboard_started = False

    # 功能：
    #   为单元测试记录降落请求，不模拟真实下降过程。
    # 输入：
    #   self：本次测试实例。
    # 输出：
    #   None：测试落地标志置真。
    async def land(self) -> None:
        self.landed = True

    # 功能：
    #   检查测试降落标志，专供执行器清理顺序测试。
    # 输入：
    #   timeout_seconds：接口占位参数。
    # 输出：
    #   observation：测试 ON_GROUND 回执，不是原生飞行证据。
    async def wait_until_landed(self, timeout_seconds: float) -> dict[str, Any]:
        _ = timeout_seconds
        if not self.landed:
            raise RuntimeError("fake PX4 has not received a land command")
        observation = {"state": "ON_GROUND", "confirmed": True}
        return observation

    # 功能：
    #   读取测试浮点参数字典，未知参数仍报错。
    # 输入：
    #   name：测试参数名。
    # 输出：
    #   value：字典中保存的值。
    async def get_param_float(self, name: str) -> float:
        value = self.float_params[name]
        return value

    # 功能：
    #   保存测试浮点参数，拒绝与真实传输不兼容的数值。
    # 输入：
    #   name：测试参数名。
    #   value：有限浮点数值。
    # 输出：
    #   None：更新测试字典。
    async def set_param_float(self, name: str, value: float) -> None:
        self.float_params[name] = _native_number(value, name)

    # 功能：
    #   读取测试整型参数字典。
    # 输入：
    #   name：测试参数名。
    # 输出：
    #   value：保存的整数。
    async def get_param_int(self, name: str) -> int:
        value = self.int_params[name]
        return value

    # 功能：
    #   保存测试整型参数，避免替身容忍真实传输拒绝的类型。
    # 输入：
    #   name：测试参数名。
    #   value：严格整数。
    # 输出：
    #   None：更新测试参数。
    async def set_param_int(self, name: str, value: int) -> None:
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError("test PX4 integer parameter is invalid")
        self.int_params[name] = value

    # 功能：
    #   返回带明确测试传输来源的相机回执。
    # 输入：
    #   parameters：待记录的测试相机参数。
    # 输出：
    #   result：测试回执，调用参数不能覆写来源标识。
    async def execute_camera_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        result = {**parameters, "confirmed": True, "transport": "fake-camera"}
        return result

    # 功能：
    #   为动作编排测试更新载荷状态，不改变仿真关节或质量。
    # 输入：
    #   parameters：attach 或 detach 测试动作。
    # 输出：
    #   result：明确标注 fake-payload 的测试状态。
    async def execute_payload_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        operation = parameters.get("operation")
        if operation == "attach":
            self.payload_detached = False
        elif operation == "detach":
            self.payload_detached = True
        else:
            raise ValueError("unsupported test payload operation")
        result = {
            **parameters,
            "confirmed": True,
            "transport": "fake-payload",
            "detached": self.payload_detached,
        }
        return result

    # 功能：
    #   返回测试载荷状态，来源始终与原生关节观测区分。
    # 输入：
    #   output_topic：测试状态主题。
    #   timeout_seconds：接口占位预算。
    # 输出：
    #   result：标记 fake-payload-state 的回执。
    async def sample_payload_state(
        self,
        output_topic: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        _ = timeout_seconds
        result = {
            "confirmed": True,
            "transport": "fake-payload-state",
            "output_topic": output_topic,
            "detached": self.payload_detached,
        }
        return result

    # 功能：
    #   更新测试避障开关，用于验证参数前后值记录。
    # 输入：
    #   enabled：严格布尔测试开关。
    # 输出：
    #   result：测试参数变化记录，不表示避障成功。
    async def execute_avoidance_command(self, enabled: bool) -> dict[str, Any]:
        if type(enabled) is not bool:
            raise ValueError("test avoidance flag must be boolean")
        before = self.int_params["COM_OBS_AVOID"]
        self.int_params["COM_OBS_AVOID"] = 1 if enabled else 0
        result = {
            "confirmed": True,
            "transport": "fake-parameter",
            "parameter": "COM_OBS_AVOID",
            "before": before,
            "after": self.int_params["COM_OBS_AVOID"],
            "enabled": enabled,
        }
        return result

    # 功能：
    #   按顺序消费测试目标位置，最后一项保持可读以支持重复采样测试。
    # 输入：
    #   parameters：接口占位的目标参数。
    # 输出：
    #   sample：本次测试位置的独立字典。
    async def sample_gazebo_pose(self, parameters: dict[str, Any]) -> dict[str, float | str]:
        _ = parameters
        if len(self.gazebo_pose_samples) > 1:
            return dict(self.gazebo_pose_samples.pop(0))
        return dict(self.gazebo_pose_samples[0])

    # 功能：
    #   消费测试电池样本，不用真实遥测连接。
    # 输入：
    #   timeout_seconds：接口占位参数。
    # 输出：
    #   sample：当前测试电池数据副本。
    async def sample_battery(self, timeout_seconds: float) -> dict[str, float]:
        _ = timeout_seconds
        if len(self.battery_samples) > 1:
            return dict(self.battery_samples.pop(0))
        return dict(self.battery_samples[0])

    # 功能：
    #   根据测试卫星参数生成 GPS 夹具，专用于失效注入的编排测试。
    # 输入：
    #   timeout_seconds：接口占位参数。
    # 输出：
    #   sample：测试卫星数和定位类型，不可作为物理效果读回。
    async def sample_gps_info(self, timeout_seconds: float) -> dict[str, int | str]:
        _ = timeout_seconds
        target_used = self.int_params["SIM_GPS_USED"]
        sample = {
            "num_satellites": target_used,
            "fix_type": 3 if target_used >= 4 else 0,
            "fix_type_name": "FIX_3D" if target_used >= 4 else "NO_GPS",
        }
        return sample

    # 功能：
    #   优先返回显式注入的测试遥测；未注入时使用理想跟随夹具，仅供单元测试。
    # 输入：
    #   timeout_seconds：接口占位参数。
    # 输出：
    #   sample：测试 NED 位置速度，不能证明任何真实飞行已到达。
    async def sample_position_velocity_ned(
        self,
        timeout_seconds: float,
    ) -> PositionVelocityNed:
        _ = timeout_seconds
        if self.position_velocity_samples:
            if len(self.position_velocity_samples) > 1:
                return self.position_velocity_samples.pop(0)
            return self.position_velocity_samples[0]
        target = self.setpoints[-1] if self.setpoints else Setpoint(0.0, 0.0, 0.0, 0.0)
        sample = PositionVelocityNed(
            north_m=target.north_m,
            east_m=target.east_m,
            down_m=target.down_m,
            north_m_s=0.0,
            east_m_s=0.0,
            down_m_s=0.0,
        )
        return sample

    # 功能：
    #   实现测试重启接口，不创建遥测任务或网络资源。
    # 输入：
    #   self：测试实例。
    # 输出：
    #   None：测试空操作结束。
    async def restart_position_velocity_ned_stream(self) -> None:
        return None

    # 功能：
    #   返回注入的有限测试航向。
    # 输入：
    #   timeout_seconds：接口占位参数。
    # 输出：
    #   heading_deg：测试航向角，单位为度。
    async def sample_heading_deg(self, timeout_seconds: float) -> float:
        _ = timeout_seconds
        heading_deg = _native_number(self.heading_deg, "fake PX4 attitude yaw_deg")
        return heading_deg

    # 功能：
    #   返回独立的动力学测试快照，消费方不能通过嵌套引用改写夹具。
    # 输入：
    #   max_age_seconds：正值年龄预算，仅校验接口参数。
    # 输出：
    #   sample：深复制的测试动力学数据。
    def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]:
        if _native_number(max_age_seconds, "test freshness") <= 0.0:
            raise ValueError("dynamics telemetry freshness bound must be positive")
        sample = copy.deepcopy(self.dynamics_telemetry)
        return sample

    # 功能：
    #   返回测试注入的模型位姿，不访问 Gazebo。
    # 输入：
    #   world_name、model_name、timeout_seconds：接口占位的实体身份和预算。
    # 输出：
    #   pose：显式注入的测试位姿。
    async def sample_gazebo_model_pose(
        self,
        *,
        world_name: str,
        model_name: str,
        timeout_seconds: float,
    ) -> GazeboModelPose:
        del world_name, model_name, timeout_seconds
        pose = self.gazebo_model_pose
        return pose

    # 功能：
    #   记录测试客户端关闭，供退出路径断言使用。
    # 输入：
    #   self：本次测试实例。
    # 输出：
    #   None：closed 标志置真。
    async def close(self) -> None:
        self.closed = True


# 功能：
#   解析环境布尔开关，未设置时使用显式默认值，未知文本不能静默变真。
# 输入：
#   raw：环境变量文本或 None。
#   default：未设置时的默认开关。
# 输出：
#   value：解析后的布尔值。
def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")


# 功能：
#   将可选环境文本解析成有限数字，空白值使用指定默认值。
# 输入：
#   raw：待解析环境变量。
#   default：未填写时的数值。
# 输出：
#   value：可用于后续单位和值域校验的数字。
def _parse_float(raw: str | None, *, default: float) -> float:
    if raw is None or not raw.strip():
        value = _native_number(default, "environment default")
    else:
        value = _finite_float(raw, "environment float")
    return value


# 功能：
#   声明显式参考轨迹运行的参数，保留环境默认值并将未知参数作为错误。
# 输入：
#   argv：命令行参数列表，None 时读取当前进程参数。
# 输出：
#   args：路径、连接、控制频率、超时及起飞门限的命名空间。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 offboard track executor")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--track", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--abort-file", type=Path)
    parser.add_argument(
        "--connection",
        default=os.environ.get("PX4_OFFBOARD_CONNECTION", "udpin://0.0.0.0:14540"),
    )
    parser.add_argument(
        "--setpoint-rate-hz",
        type=float,
        default=_parse_float(os.environ.get("PX4_OFFBOARD_SETPOINT_RATE_HZ"), default=10.0),
    )
    parser.add_argument(
        "--takeoff-timeout-seconds",
        type=float,
        default=_parse_float(os.environ.get("PX4_OFFBOARD_TAKEOFF_TIMEOUT_SECONDS"), default=30.0),
    )
    parser.add_argument(
        "--takeoff-climb-rate-m-s",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_CLIMB_RATE_M_S"),
            default=1.0,
        ),
    )
    parser.add_argument(
        "--track-timeout-seconds",
        type=float,
        default=_parse_float(os.environ.get("PX4_OFFBOARD_TRACK_TIMEOUT_SECONDS"), default=120.0),
    )
    parser.add_argument(
        "--landing-timeout-seconds",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_LANDING_TIMEOUT_SECONDS"),
            default=60.0,
        ),
    )
    parser.add_argument(
        "--takeoff-horizontal-tolerance-m",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_HORIZONTAL_TOLERANCE_M"),
            default=0.12,
        ),
    )
    parser.add_argument(
        "--takeoff-vertical-tolerance-m",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_VERTICAL_TOLERANCE_M"),
            default=0.08,
        ),
    )
    parser.add_argument(
        "--takeoff-horizontal-speed-tolerance-m-s",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_HORIZONTAL_SPEED_TOLERANCE_M_S"),
            default=0.10,
        ),
    )
    parser.add_argument(
        "--takeoff-vertical-speed-tolerance-m-s",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_VERTICAL_SPEED_TOLERANCE_M_S"),
            default=0.08,
        ),
    )
    parser.add_argument(
        "--takeoff-stable-window-seconds",
        type=float,
        default=_parse_float(
            os.environ.get("PX4_OFFBOARD_TAKEOFF_STABLE_WINDOW_SECONDS"),
            default=1.5,
        ),
    )
    parser.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="measured-hold",
    )
    parser.add_argument(
        "--maximum-yaw-rate-deg-s",
        type=float,
        default=20.0,
    )
    parser.add_argument("--gazebo-vehicle-model-name")
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args(argv)
    return args


# 功能：
#   向本次运行日志追加一行，创建其父目录而不覆盖已有日志。
# 输入：
#   path：当前任务日志路径。
#   message：需要记录的事件文字。
# 输出：
#   None：本次日志写入完成。
def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


# 功能：
#   验证原生落地回执并复制，命令成功或任意非空对象都不能替代明确的落地状态。
# 输入：
#   observation：客户端本次等待返回的原生观测。
# 输出：
#   confirmed：具有 ON_GROUND 与字面 True 标记的独立观测副本。
def _validated_landing_observation(observation: Any) -> dict[str, Any]:
    if (
        not isinstance(observation, dict)
        or observation.get("state") != "ON_GROUND"
        or observation.get("confirmed") is not True
    ):
        raise RuntimeError("PX4_NATIVE_LANDING_NOT_CONFIRMED")
    confirmed = copy.deepcopy(observation)
    return confirmed


# 功能：
#   通过有界原子写入保存时序快照，避免监视端读取半份 JSON。
# 输入：
#   path：时序证据路径。
#   payload：当前阶段的完整时序内容。
# 输出：
#   None：快照发布完成。
def _write_offboard_timing(path: Path, payload: dict[str, Any]) -> None:
    _write_json_atomic(path, payload)


# 功能：
#   校验并发布运行阶段，让独立安全监视器读取实际执行进度。
# 输入：
#   path：本次阶段通道路径。
#   phase：受支持的阶段名称。
# 输出：
#   None：完整阶段快照替换旧值。
def _write_runtime_phase(path: Path, phase: str) -> None:
    """Atomically publish the executor phase consumed by live safety monitors."""

    allowed_phases = {
        "PREFLIGHT",
        "TAKEOFF",
        "TRACK",
        "LANDING",
        "LANDED",
        "COMPLETE",
        "FAILED",
    }
    if phase not in allowed_phases:
        raise ValueError(f"unsupported runtime phase: {phase}")
    _write_json_atomic(path, {"phase": phase})


# 功能：
#   加载场景工况实现，独立部署时仅搜索约定的同工作区后端目录。
# 输入：
#   无显式参数；使用当前模块所在目录和已安装后端。
# 输出：
#   engine：实际场景工况模块，缺失时抛出异常而不伪造工况。
def _load_scenario_effect_engine() -> Any:
    try:
        from app.simulator import scenario_effects as engine
    except ModuleNotFoundError:
        backend_root = Path(__file__).resolve().parents[2] / "backend"
        if not backend_root.is_dir():
            raise RuntimeError(
                "DroneDream backend package is required for flight-timed scenario effects"
            ) from None
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))
        from app.simulator import scenario_effects as engine
    return engine


# 功能：
#   按显式请求加载和编译工况；普通鉴定未要求工况时不强制依赖后端。
# 输入：
#   无显式参数；读取 PX4_TRIAL_SCENARIO_EFFECT_REQUEST_PATH。
# 输出：
#   result：工况引擎、已校验请求与编译配置组成的三元组，无请求时三项均为 None。
def _load_runtime_effect_request() -> tuple[
    Any | None, dict[str, Any] | None, dict[str, Any] | None
]:
    raw_path = os.environ.get("PX4_TRIAL_SCENARIO_EFFECT_REQUEST_PATH", "").strip()
    if not raw_path:
        # A plain qualification flight has no timed scenario effects and must
        # remain independently runnable from the optional backend package.
        return None, None, None
    engine = _load_scenario_effect_engine()
    request = engine.load_scenario_effect_request(Path(raw_path))
    return engine, request, engine.compile_bundled_runtime_profile(request)


# 功能：
#   对有限、稳定排序的 JSON 内容计算摘要，绑定工况读回而非文件排版。
# 输入：
#   value：可序列化的实际证据对象。
# 输出：
#   digest：小写 SHA-256 摘要。
def _canonical_sha256(value: object) -> str:
    digest = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return digest


# 功能：
#   解析唯一的风速向量与启用开关，省略轴按 protobuf 零值处理，重复或非法字段拒绝。
# 输入：
#   response_text：原生 wind_info 标准输出，不拼入错误输出。
# 输出：
#   wind：世界坐标风速向量及严格布尔开关。
def _parse_gazebo_wind_info(response_text: str) -> dict[str, Any]:
    fields = _textproto_fields(response_text)
    block = _textproto_value(fields, "linear_velocity")
    if not isinstance(block, dict):
        raise RuntimeError("Gazebo wind_info response omitted linear_velocity")
    vector: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        # ``gz service`` renders protobuf text format. Proto3 omits scalar
        # fields whose value is the default zero, so a north-only wind can be
        # returned as ``linear_velocity { y: 3 }``. Treat only an omitted axis
        # as the protobuf-defined 0.0; present values remain strictly parsed
        # and compared against the requested vector below.
        value = _native_number(_textproto_value(block, axis, 0.0), "wind " + axis)
        if not math.isfinite(value):
            raise RuntimeError(f"Gazebo wind_info returned non-finite {axis}")
        vector[axis] = round(value, 12)
    enabled = _textproto_value(fields, "enable_wind", False)
    if type(enabled) is not bool:
        raise RuntimeError("Gazebo wind_info enable_wind must be boolean")
    wind = {
        "linear_velocity_mps": vector,
        "enable_wind": enabled,
    }
    return wind


# 功能：
#   验证世界名和三轴风速，并定位本机实际 Gazebo 命令。
# 输入：
#   world：本次世界名。
#   profile：包含完整 linear_velocity_mps 的编译配置。
# 输出：
#   result：风速、发布主题、读回服务和可执行文件路径。
def _validated_gazebo_wind_activation(
    world: str,
    profile: dict[str, Any],
) -> tuple[dict[str, float], str, str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", world):
        raise RuntimeError("Gazebo world name is invalid for post-hover wind activation")
    vector = profile.get("linear_velocity_mps")
    if not isinstance(vector, dict) or set(vector) != {"x", "y", "z"}:
        raise RuntimeError("compiled wind activation vector is invalid")
    requested = {
        axis: _finite_float(vector[axis], f"wind_activation.{axis}") for axis in ("x", "y", "z")
    }
    gz_cli = shutil.which("gz")
    if not gz_cli:
        raise RuntimeError("Gazebo gz CLI is unavailable for post-hover wind activation")
    return requested, f"/world/{world}/wind", f"/world/{world}/wind_info", gz_cli


# 功能：
#   保留足够数字精度构建 Gazebo 风场控制消息。
# 输入：
#   requested：已校验的世界坐标三轴风速。
# 输出：
#   message：启用风场的 protobuf 文本请求。
def _gazebo_wind_message_text(requested: dict[str, float]) -> str:
    message = (
        "linear_velocity { "
        f"x: {requested['x']:.17g} y: {requested['y']:.17g} z: {requested['z']:.17g} "
        "} enable_wind: true"
    )
    return message


# 功能：
#   执行一条授权的 Gazebo 命令，以共享进程收集器限制输出、时间及取消后的资源存活。
# 输入：
#   command：Gazebo 可执行文件和逐项参数。
#   timeout：本次命令剩余秒数预算。
#   cancel_event：异步拥有者发出的停止信号。
# 输出：
#   completed：真实退出码及有限文本输出。
def _run_gazebo_cli(
    command: list[str], timeout: float, cancel_event: threading.Event | None
) -> subprocess.CompletedProcess[str]:
    from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
    from dronedream_agent_core.process_capture import capture_process

    captured = capture_process(
        command,
        stdin=b"",
        maximum_bytes=1_048_576,
        timeout=timeout,
        environment=dict(os.environ),
        resource_policy=PluginResourcePolicy(),
        cancel_event=cancel_event,
    )
    completed = subprocess.CompletedProcess(
        command,
        captured.returncode,
        captured.stdout.decode("utf-8"),
        captured.stderr.decode("utf-8", errors="replace"),
    )
    return completed


# 功能：
#   在有限窗口内发布风场并精确读回，取消后停止后续发布，命令退出成功不能单独证明风已生效。
# 输入：
#   world：当前世界名称。
#   profile：已编译的风速向量。
#   activation_t_s：相对执行开始的激活秒数。
#   cancel_event：拥有者停止任务时的可选取消信号。
# 输出：
#   evidence：真实风场读回、来源、尝试次数及激活记录。
def _activate_gazebo_wind_profile(
    *,
    world: str,
    profile: dict[str, Any],
    activation_t_s: float,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    requested, topic, service, gz_cli = _validated_gazebo_wind_activation(world, profile)
    # Use the Runtime-owned Gazebo CLI for publication. The Python binding can
    # advertise the correct endpoint while never completing delivery to the
    # WindEffects subscriber under WSL; the official CLI shares the exact
    # transport implementation used by the simulator. Publication is still
    # not trusted on exit status alone: /wind_info exact read-back below is the
    # sole authority that permits flight to enter the track.
    attempts = int(os.environ.get("PX4_GAZEBO_WIND_READBACK_ATTEMPTS", "100"))
    publish_duration_seconds = _parse_float(
        os.environ.get("PX4_GAZEBO_WIND_PUBLISH_DURATION_SECONDS"), default=0.5
    )
    if not 1 <= attempts <= 100 or not 0.1 <= publish_duration_seconds <= 2:
        raise ValueError("Gazebo wind activation retry budget is invalid")
    if _native_number(activation_t_s, "activation time") < 0:
        raise ValueError("Gazebo wind activation time must be nonnegative")
    deadline = time.monotonic() + 15.0
    last_error = "no wind_info response"
    readback: dict[str, Any] | None = None
    publish_attempts = 0
    message_text = _gazebo_wind_message_text(requested)
    for _attempt in range(attempts):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Gazebo wind activation cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= publish_duration_seconds:
            break
        publish_attempts += 1
        publish = _run_gazebo_cli(
            [
                gz_cli,
                "topic",
                "-t",
                topic,
                "-m",
                "gz.msgs.Wind",
                "-p",
                message_text,
                "-d",
                f"{publish_duration_seconds:g}",
            ],
            timeout=min(remaining, publish_duration_seconds + 5.0),
            cancel_event=cancel_event,
        )
        if publish.returncode != 0:
            last_error = (
                f"wind publish exit={publish.returncode}, response="
                f"{(publish.stdout + publish.stderr).strip()[:400]}"
            )
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        response = _run_gazebo_cli(
            [
                gz_cli,
                "service",
                "-s",
                service,
                "--reqtype",
                "gz.msgs.Empty",
                "--reptype",
                "gz.msgs.Wind",
                "--timeout",
                str(max(1, int(min(5.0, remaining) * 1000))),
                "--req",
                " ",
            ],
            timeout=min(8.0, remaining),
            cancel_event=cancel_event,
        )
        try:
            if response.returncode != 0:
                raise RuntimeError(
                    f"exit={response.returncode}, response="
                    f"{(response.stdout + response.stderr).strip()[:400]}"
                )
            candidate = _parse_gazebo_wind_info(response.stdout)
            matches = candidate["enable_wind"] is True and all(
                math.isclose(
                    candidate["linear_velocity_mps"][axis],
                    requested[axis],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for axis in ("x", "y", "z")
            )
            if matches:
                readback = candidate
                break
            last_error = f"mismatched readback {candidate}"
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
    if readback is None:
        raise RuntimeError(
            "Gazebo post-hover wind activation was not verified after "
            f"{publish_attempts} bounded CLI publishes: {last_error}"
        )
    evidence = {
        "readback": {
            "source": service,
            "kind": "readback",
            "value": readback,
            "sha256": _canonical_sha256(readback),
        },
        "activation": {
            "source": topic,
            "kind": "acknowledgement",
            "value": {
                "phase": "after_stable_hover_before_track_entry",
                "activation_t_s": round(activation_t_s, 12),
                "readback_service": service,
                "delivery_verification": "wind_info_exact_readback",
                "publisher": "gazebo_cli_topic",
                "publish_duration_seconds": publish_duration_seconds,
                "publish_attempts": publish_attempts,
                "publisher_connections_observed": None,
                "transport_prepared_before_takeoff": False,
            },
        },
    }
    return evidence


# 功能：
#   在线程执行风场控制时保留取消通道，终止后等它停止发布并回收命令进程。
# 输入：
#   activator：实现取消信号契约的风场激活函数。
#   world：当前世界名称。
#   profile：风场参数。
#   activation_t_s：本次激活的相对时刻。
# 输出：
#   evidence：激活函数真实产生的读回证据。
async def _await_wind_activation(
    activator: Callable[..., dict[str, Any]],
    world: str,
    profile: dict[str, Any],
    activation_t_s: float,
) -> dict[str, Any]:
    cancel_event = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(
            activator,
            world=world,
            profile=profile,
            activation_t_s=activation_t_s,
            cancel_event=cancel_event,
        )
    )
    try:
        evidence = await asyncio.shield(task)
        return evidence
    finally:
        primary = sys.exception()
        cancel_event.set()
        try:
            await asyncio.shield(task)
        except (Exception, asyncio.CancelledError):
            if primary is None:
                raise


# 功能：
#   用绑定执行身份的工况算法生成可复现 GPS 失效序列，不随机改变已确认试验条件。
# 输入：
#   requested_rate：要求的失效比例。
#   tick_count：离散时间格数量。
#   execution_identity_sha256：本次执行身份摘要。
# 输出：
#   schedule：True 表示该格失效的严格布尔序列。
def compile_fixed_duty_schedule(
    *,
    requested_rate: float,
    tick_count: int,
    execution_identity_sha256: str,
) -> list[bool]:
    engine = _load_scenario_effect_engine()
    try:
        raw_schedule: object = engine.compile_bundled_gps_dropout_schedule(
            requested_rate=requested_rate,
            tick_count=tick_count,
            execution_identity_sha256=execution_identity_sha256,
        )
    except engine.ScenarioEffectContractError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(raw_schedule, list) or any(
        not isinstance(item, bool) for item in raw_schedule
    ):
        raise RuntimeError("scenario engine returned an invalid GPS dropout schedule")
    schedule = list(raw_schedule)
    if len(schedule) != tick_count:
        raise RuntimeError("scenario engine returned the wrong GPS dropout tick count")
    return schedule


# 功能：
#   先限制证据 JSON 的大小和复杂度，再通过本次独占暂存文件发布；不删除被替换的他人文件。
# 输入：
#   path：当前运行证据的目标路径。
#   payload：待发布的标准 JSON 对象。
# 输出：
#   None：发布成功后目标包含完整快照，失败保留原目标。
def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    from dronedream_agent_core.plugin_files import check_plain_plugin_path
    from dronedream_plugin_sdk.protocol import encode_json

    serialized = encode_json(payload, limit=MAX_INPUT_JSON_BYTES, node_limit=2_000_000) + "\n"
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    owned: os.stat_result | None = None
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            owned = os.fstat(handle.fileno())
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())
        check_plain_plugin_path(temp)
        current = temp.stat()
        if (
            not os.path.samestat(written, current)
            or current.st_size != written.st_size
            or current.st_mtime_ns != written.st_mtime_ns
        ):
            raise ValueError("runtime evidence temporary file changed")
        check_plain_plugin_path(path)
        temp.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            if owned is not None and os.path.samestat(owned, temp.lstat()):
                temp.unlink()


# 功能：
#   逐项关联请求与真实工况观测，区分已应用、失败和未执行，缺失证据不能标为成功。
# 输入：
#   engine：场景证据摘要实现。
#   request、profile：原始请求与编译工况。
#   observations：实际读回观测，按工况板块索引。
#   attempted_sections：已经尝试激活的板块。
#   status、error：执行结果和可选失败原因。
# 输出：
#   records：每项工况的执行状态、请求摘要及验证证据。
def _runtime_effect_records(
    engine: Any,
    request: dict[str, Any],
    profile: dict[str, Any],
    *,
    observations: dict[str, dict[str, Any]],
    attempted_sections: set[str],
    status: str,
    error: str | None,
) -> list[dict[str, Any]]:
    requested_by_id = {effect["effect_id"]: effect for effect in request["effects"]}
    records: list[dict[str, Any]] = []
    for effect_id in profile["requested_effect_ids"]:
        effect = requested_by_id[effect_id]
        if effect_id in profile.get("wind_activation", {}).get("effect_ids", []):
            section = "wind_activation"
        elif effect_id in profile.get("gps_dropout", {}).get("effect_ids", []):
            section = "gps_dropout"
        else:
            section = "battery"
        observation = observations.get(section)
        if status != "complete" and observation is None:
            reason = error or "flight-timed scenario effect did not complete"
            attempted = section in attempted_sections
            records.append(
                {
                    "effect_id": effect_id,
                    "mechanism": effect["mechanism"],
                    "status": "failed" if attempted else "skipped",
                    "capability": {
                        "status": "available",
                        "reason": (
                            reason
                            if attempted
                            else "flight terminated before this available effect was activated"
                        ),
                    },
                    "reason": (
                        reason
                        if attempted
                        else f"flight terminated before {section} activation: {reason}"
                    ),
                }
            )
            continue
        if observation is None:
            raise RuntimeError(f"runtime effect evidence omitted {section}")
        verification_observations = (
            [observation["readback"], observation["activation"]]
            if section == "wind_activation"
            else [observation]
        )
        if section == "wind_activation":
            capability_reason = (
                "Gazebo wind was enabled only after the stable-hover gate and exact readback"
            )
            method = "gazebo_wind_topic_after_stable_hover_and_exact_readback"
        elif section == "gps_dropout":
            capability_reason = "PX4 GPS availability parameter and telemetry verified the schedule"
            method = "mavsdk_sim_gps_used_plus_gps_info_telemetry_and_reset"
        else:
            capability_reason = "PX4 parameter readback and battery telemetry verified the profile"
            method = "mavsdk_parameter_readback_and_battery_telemetry"
        records.append(
            {
                "effect_id": effect_id,
                "mechanism": effect["mechanism"],
                "status": "applied",
                "capability": {
                    "status": "available",
                    "reason": capability_reason,
                },
                "evidence": {
                    "requested_value_sha256": engine.scenario_effect_value_sha256(
                        effect["requested_value"]
                    ),
                    "compiled_runtime_profile": profile,
                    "verification": {
                        "status": "verified",
                        "method": method,
                        "observations": verification_observations,
                    },
                },
            }
        )
    return records


# 功能：
#   将逐项工况结果绑定原始请求摘要并原子保存。
# 输入：
#   engine、request、profile：工况实现、已校验请求和编译配置。
#   path：本次运行证据路径。
#   observations、attempted_sections：真实观测及已尝试板块。
#   status、error：最终状态和错误信息。
# 输出：
#   None：证据文件发布完成。
def _write_runtime_effect_artifact(
    engine: Any,
    request: dict[str, Any],
    profile: dict[str, Any],
    path: Path,
    *,
    observations: dict[str, dict[str, Any]],
    attempted_sections: set[str],
    status: str,
    error: str | None = None,
) -> None:
    records = _runtime_effect_records(
        engine,
        request,
        profile,
        observations=observations,
        attempted_sections=attempted_sections,
        status=status,
        error=error,
    )
    payload = {
        "schema_version": RUNTIME_EFFECT_SCHEMA_VERSION,
        "request_sha256": request["request_sha256"],
        "compiled_runtime_profile": profile,
        "attempted_sections": sorted(attempted_sections),
        "status": status,
        "error": error,
        "records": records,
    }
    _write_json_atomic(path, payload)


# 功能：
#   将兼容数字文本的输入转换为有限浮点数，布尔值和非有限值不能冒充传感器或配置数值。
# 输入：
#   value：待转换的原始字段。
#   label：错误消息中的字段名称。
# 输出：
#   parsed：经过有限性检查的浮点数。
def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


# 功能：
#   接受整数及精确整数浮点值，拒绝布尔、数值文本和小数截断。
# 输入：
#   value：待校验数值。
#   label：错误消息中的字段名称。
# 输出：
#   parsed：无精度截断的整数。
def _finite_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"{label} must be an integer")


# 功能：
#   校验原生电池百分比和电压，禁止非有限值、负电压和比例单位混用。
# 输入：
#   value：电池遥测字典，remaining_percent 单位为百分数而非零到一比例。
#   label：本次观测名称。
# 输出：
#   sample：零到一百的 remaining_percent 及非负 voltage_v。
def _validated_battery_telemetry(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    remaining_percent = _native_number(
        value.get("remaining_percent"),
        f"{label} remaining_percent",
    )
    voltage_v = _native_number(value.get("voltage_v"), f"{label} voltage_v")
    if not 0.0 <= remaining_percent <= 100.0:
        raise ValueError(f"{label} remaining_percent must be between 0 and 100")
    if voltage_v < 0.0:
        raise ValueError(f"{label} voltage_v must be non-negative")
    sample = {"remaining_percent": remaining_percent, "voltage_v": voltage_v}
    return sample


# 功能：
#   校验 GPS 卫星数量与定位类型，保留原生枚举名称供工况读回解释。
# 输入：
#   value：含卫星数量、定位类型及可选类型名称的遥测字典。
#   label：当前观测名称。
# 输出：
#   sample：校验后的整数状态及名称。
def _validated_gps_telemetry(value: Any, *, label: str) -> dict[str, int | str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    num_satellites = _finite_integer(
        value.get("num_satellites"),
        f"{label} num_satellites",
    )
    fix_type = _finite_integer(value.get("fix_type"), f"{label} fix_type")
    if num_satellites < 0:
        raise ValueError(f"{label} num_satellites must be non-negative")
    if fix_type < 0:
        raise ValueError(f"{label} fix_type must be non-negative")
    fix_type_name = value.get("fix_type_name", str(fix_type))
    if not isinstance(fix_type_name, str) or not fix_type_name.strip():
        raise ValueError(f"{label} fix_type_name must be a non-empty string")
    sample = {
        "num_satellites": num_satellites,
        "fix_type": fix_type,
        "fix_type_name": fix_type_name,
    }
    return sample


# 功能：
#   读取一份稳定普通文件快照，拒绝链接、并发替换、重复键、非有限数和过深结构。
# 输入：
#   path：本次运行绑定的 JSON 输入路径。
#   label：用于标识读取失败的字段或文件用途。
# 输出：
#   payload：经过结构预算检查的标准 JSON 值。
def _load_bounded_json(path: Path, *, label: str) -> Any:
    from dronedream_agent_core.plugin_files import read_plugin_file
    from dronedream_plugin_sdk.protocol import decode_json

    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected: {exc}") from exc
    if stat.S_ISLNK(status.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if status.st_size > MAX_INPUT_JSON_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_INPUT_JSON_BYTES} byte limit")
    try:
        raw = read_plugin_file(path, limit=MAX_INPUT_JSON_BYTES)
    except OSError as exc:
        raise ValueError(f"{label} cannot be read: {exc}") from exc
    if len(raw) > MAX_INPUT_JSON_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_INPUT_JSON_BYTES} byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    payload = decode_json(text, limit=MAX_INPUT_JSON_BYTES, node_limit=2_000_000)
    return payload


# 功能：
#   读取参考轨迹并验证点数、坐标、速度、悬停与停稳参数，保留试验轨迹的显式执行模式。
# 输入：
#   path：受限大小的参考轨迹 JSON 文件路径。
# 输出：
#   plan：包含轨迹点和执行参数的参考计划。
def load_reference_track_plan(path: Path) -> ReferenceTrackPlan:
    payload = _load_bounded_json(path, label="reference_track.json")
    plan = parse_reference_track_plan(payload)
    return plan


# 功能：
#   验证同一份已读取轨迹快照并生成参考计划，供调度和任务摘要绑定共用，避免再次读盘混入新版本。
# 输入：
#   payload：已完成有界 JSON 读取的轨迹对象。
# 输出：
#   plan：含经过验证的轨迹点、执行模式和稳定参数的独立参考计划。
def parse_reference_track_plan(payload: object) -> ReferenceTrackPlan:
    if not isinstance(payload, dict) or not isinstance(payload.get("points"), list):
        raise ValueError("reference_track.json must be an object with points[]")
    if len(payload["points"]) > MAX_REFERENCE_TRACK_POINTS:
        raise ValueError(
            f"reference_track.json exceeds the {MAX_REFERENCE_TRACK_POINTS}-point limit"
        )
    points: list[TrackPoint] = []
    for idx, raw in enumerate(payload["points"]):
        if not isinstance(raw, dict):
            raise ValueError(f"reference point {idx} must be an object")
        points.append(
            TrackPoint(
                _finite_float(raw.get("x"), f"reference point {idx}.x"),
                _finite_float(raw.get("y"), f"reference point {idx}.y"),
                _finite_float(raw.get("z"), f"reference point {idx}.z"),
                (
                    _finite_float(
                        raw.get("speed_limit_mps"),
                        f"reference point {idx}.speed_limit_mps",
                    )
                    if raw.get("speed_limit_mps") is not None
                    else None
                ),
            )
        )
    if not points:
        raise ValueError("reference_track.json points[] cannot be empty")
    track_type_raw = payload.get("track_type")
    track_type = None if track_type_raw is None else str(track_type_raw).strip()
    if track_type not in {None, "hover", "circle", "u_turn", "lemniscate", "custom"}:
        raise ValueError("reference_track.json track_type is unsupported")
    hover_duration_seconds: float | None = None
    if track_type == "hover":
        hover_duration_seconds = _finite_float(
            payload.get("hover_duration_s", DEFAULT_HOVER_DURATION_SECONDS),
            "hover_duration_s",
        )
        if not MIN_HOVER_DURATION_SECONDS <= hover_duration_seconds <= MAX_HOVER_DURATION_SECONDS:
            raise ValueError(
                "hover_duration_s must be between "
                f"{MIN_HOVER_DURATION_SECONDS:g} and {MAX_HOVER_DURATION_SECONDS:g} seconds"
            )
    stop_at_waypoints = payload.get("stop_at_waypoints", False)
    if not isinstance(stop_at_waypoints, bool):
        raise ValueError("reference_track.json stop_at_waypoints must be a boolean")
    waypoint_hold_seconds = _finite_float(
        payload.get("waypoint_hold_seconds", 0.0),
        "waypoint_hold_seconds",
    )
    if not 0.0 <= waypoint_hold_seconds <= MAX_WAYPOINT_HOLD_SECONDS:
        raise ValueError(
            f"waypoint_hold_seconds must be between 0 and {MAX_WAYPOINT_HOLD_SECONDS:g} seconds"
        )
    waypoint_position_tolerance_m = _finite_float(
        payload.get("waypoint_position_tolerance_m", 0.2),
        "waypoint_position_tolerance_m",
    )
    waypoint_speed_tolerance_mps = _finite_float(
        payload.get("waypoint_speed_tolerance_mps", 0.15),
        "waypoint_speed_tolerance_mps",
    )
    waypoint_stable_window_seconds = _finite_float(
        payload.get("waypoint_stable_window_seconds", 0.5),
        "waypoint_stable_window_seconds",
    )
    waypoint_settle_timeout_seconds = _finite_float(
        payload.get("waypoint_settle_timeout_seconds", 12.0),
        "waypoint_settle_timeout_seconds",
    )
    bounded_positive = (
        (waypoint_position_tolerance_m, 2.0, "waypoint_position_tolerance_m"),
        (waypoint_speed_tolerance_mps, 2.0, "waypoint_speed_tolerance_mps"),
        (waypoint_stable_window_seconds, 10.0, "waypoint_stable_window_seconds"),
        (waypoint_settle_timeout_seconds, 120.0, "waypoint_settle_timeout_seconds"),
    )
    for value, maximum, label in bounded_positive:
        if not 0.0 < value <= maximum:
            raise ValueError(f"{label} must be in (0, {maximum:g}]")
    for index, point in enumerate(points):
        if point.speed_limit_mps is not None and not 0 < point.speed_limit_mps <= 10.0:
            raise ValueError(f"reference point {index}.speed_limit_mps must be in (0, 10]")
    plan = ReferenceTrackPlan(
        points=points,
        track_type=track_type,
        hover_duration_seconds=hover_duration_seconds,
        stop_at_waypoints=stop_at_waypoints,
        waypoint_hold_seconds=waypoint_hold_seconds,
        waypoint_position_tolerance_m=waypoint_position_tolerance_m,
        waypoint_speed_tolerance_mps=waypoint_speed_tolerance_mps,
        waypoint_stable_window_seconds=waypoint_stable_window_seconds,
        waypoint_settle_timeout_seconds=waypoint_settle_timeout_seconds,
    )
    return plan


# 功能：
#   读取真实终止原因与世界暂停标志，世界已暂停时不再尝试依赖仿真推进的降落。
# 输入：
#   path：当前运行的有界终止请求文件。
# 输出：
#   request：原因文本及严格布尔暂停状态组成的二元组。
def _read_external_abort_request(path: Path) -> tuple[str, bool]:
    """Read the bounded runner-to-executor abort contract.

    ``world_paused`` distinguishes a collision monitor stop, where attempting
    to land cannot make progress, from an operator stop that should retain the
    normal PX4 landing cleanup path.
    """

    from dronedream_agent_core.plugin_files import read_plugin_file
    from dronedream_plugin_sdk.protocol import decode_json

    try:
        raw = read_plugin_file(path, limit=MAX_ABORT_REQUEST_BYTES)
        payload = decode_json(raw.decode("utf-8"), limit=MAX_ABORT_REQUEST_BYTES)
    except (OSError, ValueError) as exc:
        raise RuntimeError("external safety abort request is invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("external safety abort request is invalid")
    reason = payload.get("reason")
    world_paused = payload.get("world_paused")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        raise RuntimeError("external safety abort reason is invalid")
    if not isinstance(world_paused, bool):
        raise RuntimeError("external safety abort world_paused flag is invalid")
    request = reason.strip(), world_paused
    return request


# 功能：
#   每次控制前检查外部终止请求；不存在表示未请求，非法文件不静默当作无请求。
# 输入：
#   path：当前运行可选的终止请求路径。
# 输出：
#   None：没有请求时返回，否则以明确终止或读取异常中断控制。
def _raise_if_external_abort_requested(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.lstat()
    except FileNotFoundError:
        return
    reason, world_paused = _read_external_abort_request(path)
    raise ExternalSafetyAbort(reason, world_paused=world_paused)


# 功能：
#   为仅需航点的调用方复用严格轨迹解析，不另设宽松读取路径。
# 输入：
#   path：本次参考轨迹 JSON。
# 输出：
#   points：完整解析得到的轨迹点列表。
def load_reference_track(path: Path) -> list[TrackPoint]:
    return load_reference_track_plan(path).points


# 功能：
#   读取本调度器实际使用的速度与加速度上限，拒绝缺少字段、非有限值或非正上限。
# 输入：
#   path：控制器参数 JSON 文件路径。
# 输出：
#   params：已经验证的运动约束，不包含本进程并未应用的其他飞控参数。
def load_controller_params(path: Path) -> ControllerParams:
    payload = _load_bounded_json(path, label="controller_params.json")
    if not isinstance(payload, dict):
        raise ValueError("controller_params.json must be an object")
    required = {"vel_limit", "accel_limit"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            "controller_params.json is missing active limits: " + ", ".join(sorted(missing))
        )
    params = ControllerParams(
        vel_limit=_finite_float(payload["vel_limit"], "vel_limit"),
        accel_limit=_finite_float(payload["accel_limit"], "accel_limit"),
    )
    if params.vel_limit <= 0 or params.accel_limit <= 0:
        raise ValueError("vel_limit and accel_limit must be greater than zero")
    return params


# 功能：
#   根据局部北、东位移计算路径航向；无水平位移时返回零，实际保持航向由执行层接管。
# 输入：
#   prev_point：航段起点。
#   next_point：航段终点。
# 输出：
#   yaw_deg：从北方向顺时针计量的航向角，单位为度。
def compute_yaw_from_segment(prev_point: TrackPoint, next_point: TrackPoint) -> float:
    dx = next_point.x - prev_point.x
    dy = next_point.y - prev_point.y
    yaw_deg = 0.0 if abs(dx) < 1e-9 and abs(dy) < 1e-9 else math.degrees(math.atan2(dy, dx))
    return yaw_deg


# 功能：
#   将已转换为北、东、向上位移的轨迹点改为飞控 NED 指令，仅翻转竖直方向。
# 输入：
#   point：局部轨迹点，x 为北、y 为东、z 为向上米数，不是原地图的东、北顺序。
#   yaw_deg：期望航向角，单位为度。
# 输出：
#   setpoint：北、东、向下位置及航向的飞控设定值。
def enu_point_to_ned_setpoint(point: TrackPoint, yaw_deg: float) -> Setpoint:
    setpoint = Setpoint(north_m=point.x, east_m=point.y, down_m=-point.z, yaw_deg=yaw_deg)
    return setpoint


# 功能：
#   1. 沿折线弧长生成先加速、必要时匀速、最后减速的参考位置采样。
#   2. 短路段使用三角速度曲线，长路段使用梯形曲线，并在分配前检查采样预算。
#   3. 无位移时返回空运动；该标量参考不保证转角处的矢量加速度或实际避障安全。
# 输入：
#   start：局部运动起点。
#   waypoints：后续路径点，按顺序经过。
#   params：沿路径弧长方向的速度及加速度上限。
#   rate_hz：采样频率。
#   max_samples：调用方剩余的采样预算。
# 输出：
#   result：不含起始点采样的运动位置与路径航向列表。
def _build_motion_setpoints(
    start: TrackPoint,
    waypoints: list[TrackPoint],
    params: ControllerParams,
    rate_hz: float,
    *,
    max_samples: int,
) -> list[Setpoint]:
    result: list[Setpoint] = []
    segments: list[tuple[TrackPoint, TrackPoint, float, float, float]] = []
    cumulative_distance = 0.0
    previous = start
    for waypoint in waypoints:
        distance = math.dist(
            (previous.x, previous.y, previous.z),
            (waypoint.x, waypoint.y, waypoint.z),
        )
        if distance > 1e-12:
            segments.append(
                (
                    previous,
                    waypoint,
                    distance,
                    cumulative_distance,
                    cumulative_distance + distance,
                )
            )
            cumulative_distance += distance
        previous = waypoint
    if not segments:
        return result

    acceleration_seconds = params.vel_limit / params.accel_limit
    acceleration_distance = 0.5 * params.accel_limit * acceleration_seconds**2
    if 2.0 * acceleration_distance >= cumulative_distance:
        acceleration_seconds = math.sqrt(cumulative_distance / params.accel_limit)
        peak_speed = params.accel_limit * acceleration_seconds
        acceleration_distance = 0.5 * params.accel_limit * acceleration_seconds**2
        cruise_seconds = 0.0
    else:
        peak_speed = params.vel_limit
        cruise_seconds = (cumulative_distance - 2.0 * acceleration_distance) / peak_speed
    total_seconds = 2.0 * acceleration_seconds + cruise_seconds
    sample_count = max(1, int(math.ceil(total_seconds * rate_hz)))
    if sample_count > max_samples:
        raise ValueError(f"setpoint schedule exceeds the {MAX_SETPOINTS}-sample limit")

    cruise_distance = peak_speed * cruise_seconds
    segment_index = 0
    for sample_index in range(1, sample_count + 1):
        sample_time = min(sample_index / rate_hz, total_seconds)
        if sample_time <= acceleration_seconds:
            progress = 0.5 * params.accel_limit * sample_time**2
        elif sample_time <= acceleration_seconds + cruise_seconds:
            progress = acceleration_distance + peak_speed * (sample_time - acceleration_seconds)
        else:
            deceleration_time = sample_time - acceleration_seconds - cruise_seconds
            progress = (
                acceleration_distance
                + cruise_distance
                + peak_speed * deceleration_time
                - 0.5 * params.accel_limit * deceleration_time**2
            )
        progress = min(cumulative_distance, max(0.0, progress))
        while segment_index < len(segments) - 1 and progress >= segments[segment_index][4] - 1e-12:
            segment_index += 1
        segment_start, segment_end, segment_length, segment_offset, _ = segments[segment_index]
        ratio = min(1.0, max(0.0, (progress - segment_offset) / segment_length))
        point = TrackPoint(
            x=segment_start.x + (segment_end.x - segment_start.x) * ratio,
            y=segment_start.y + (segment_end.y - segment_start.y) * ratio,
            z=segment_start.z + (segment_end.z - segment_start.z) * ratio,
        )
        result.append(
            enu_point_to_ned_setpoint(
                point,
                yaw_deg=compute_yaw_from_segment(segment_start, segment_end),
            )
        )
    return result


# 功能：
#   为不消费阶段下标的调用方返回完整位置调度，仍复用统一的起飞、运动及末端保持实现。
# 输入：
#   points：依执行顺序排列的局部轨迹点。
#   params：速度及加速度上限。
#   rate_hz：位置指令采样频率。
# 输出：
#   schedule：按时间排序的飞控位置设定值列表。
def build_setpoint_schedule(
    points: list[TrackPoint],
    params: ControllerParams,
    rate_hz: float,
) -> list[Setpoint]:
    schedule = build_setpoint_schedule_plan(points, params, rate_hz).schedule
    return schedule


# 功能：
#   1. 逐段按两端限速生成停稳式运动和等待，供初始轨迹及运行期替换共同使用。
#   2. 零距离航段仍分配独立到达采样，保留同地点不同动作的执行顺序。
#   3. 在分配前检查剩余采样预算，避免等待或重复航点绕过数量限制。
# 输入：
#   points：局部航点序列，首点是已达到的起始状态，不在本函数重复执行起飞。
#   params：控制器速度和加速度上限。
#   rate_hz：采样频率。
#   waypoint_hold_seconds：各航点到达后的保持时间。
#   initial_yaw_deg：无位移航段沿用的初始航向。
#   max_samples：调用方尚可使用的采样数量。
# 输出：
#   result：运动与保持列表、每个后续航点在该列表中的到达下标组成的二元组。
def build_stopped_waypoint_schedule(
    points: list[TrackPoint],
    params: ControllerParams,
    rate_hz: float,
    waypoint_hold_seconds: float,
    *,
    initial_yaw_deg: float,
    max_samples: int,
) -> tuple[list[Setpoint], tuple[int, ...]]:
    if not points or len(points) > MAX_REFERENCE_TRACK_POINTS:
        raise ValueError("stopped schedule point count is outside the supported range")
    if type(max_samples) is not int or not 0 <= max_samples <= MAX_SETPOINTS:
        raise ValueError("stopped schedule sample budget is invalid")
    if not math.isfinite(rate_hz) or not 0 < rate_hz <= MAX_SETPOINT_RATE_HZ:
        raise ValueError("stopped schedule rate is outside the supported range")
    if (
        not math.isfinite(waypoint_hold_seconds)
        or not 0 <= waypoint_hold_seconds <= MAX_WAYPOINT_HOLD_SECONDS
        or not math.isfinite(initial_yaw_deg)
    ):
        raise ValueError("stopped schedule hold or heading is invalid")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            params.vel_limit,
            params.accel_limit,
        )
    ):
        raise ValueError("stopped schedule controller limits must be finite and positive")
    for point in points:
        if any(not math.isfinite(value) for value in (point.x, point.y, point.z)):
            raise ValueError("stopped schedule coordinates must be finite")
        if point.speed_limit_mps is not None and (
            not math.isfinite(point.speed_limit_mps) or point.speed_limit_mps <= 0
        ):
            raise ValueError("stopped schedule point speed must be finite and positive")
    schedule: list[Setpoint] = []
    arrivals: list[int] = []
    previous, previous_yaw = points[0], initial_yaw_deg
    hold_samples = int(math.ceil(rate_hz * waypoint_hold_seconds))
    for waypoint in points[1:]:
        remaining = max_samples - len(schedule)
        # 每段至少需要一个到达采样，且先为其保持部分预留预算。
        if hold_samples + 1 > remaining:
            raise ValueError(f"setpoint schedule exceeds the {MAX_SETPOINTS}-sample limit")
        speed = min(
            value
            for value in (
                params.vel_limit,
                previous.speed_limit_mps,
                waypoint.speed_limit_mps,
            )
            if value is not None
        )
        segment = _build_motion_setpoints(
            previous,
            [waypoint],
            replace(params, vel_limit=speed),
            rate_hz,
            max_samples=remaining - hold_samples,
        )
        if not segment:
            segment = [enu_point_to_ned_setpoint(waypoint, yaw_deg=previous_yaw)]
        schedule.extend(segment)
        arrivals.append(len(schedule) - 1)
        previous_yaw = segment[-1].yaw_deg
        hold = enu_point_to_ned_setpoint(waypoint, yaw_deg=previous_yaw)
        schedule.extend(hold for _ in range(hold_samples))
        previous = waypoint
    result = schedule, tuple(arrivals)
    return result


# 功能：
#   1. 编排起飞保持、接入首点、参考轨迹及末端保持，并标出轨迹阶段的采样范围。
#   2. 停稳式轨迹保留逐航点到达下标；连续曲线和定点悬停不伪造逐航点停稳记录。
#   3. 限制采样频率、等待时长和总采样量；此调度本身不代替在线避障和遥测验收。
# 输入：
#   points：局部参考轨迹点。
#   params：运动速度和加速度上限。
#   rate_hz：采样频率。
#   hover_duration_seconds：定点悬停时长，None 表示执行路径。
#   stop_at_waypoints：是否逐航点减速停稳。
#   waypoint_hold_seconds：停稳后各航点的等待时长。
# 输出：
#   plan：位置调度、轨迹起止下标及逐航点到达记录。
def build_setpoint_schedule_plan(
    points: list[TrackPoint],
    params: ControllerParams,
    rate_hz: float,
    *,
    hover_duration_seconds: float | None = None,
    stop_at_waypoints: bool = False,
    waypoint_hold_seconds: float = 0.0,
) -> SetpointSchedulePlan:
    rate_hz = _native_number(rate_hz, "rate_hz")
    if rate_hz <= 0 or rate_hz > MAX_SETPOINT_RATE_HZ:
        raise ValueError(f"rate_hz must be finite and in (0, {MAX_SETPOINT_RATE_HZ:g}]")
    if not points or len(points) > MAX_REFERENCE_TRACK_POINTS:
        raise ValueError("points cannot be empty or exceed the reference track limit")
    for point in points:
        for axis in ("x", "y", "z"):
            _native_number(getattr(point, axis), "track point " + axis)
        if (
            point.speed_limit_mps is not None
            and _native_number(point.speed_limit_mps, "point speed") <= 0
        ):
            raise ValueError("point speed must be positive")
    for name in ("vel_limit", "accel_limit"):
        if _native_number(getattr(params, name), name) <= 0:
            raise ValueError("controller limits must be positive")
    if not isinstance(stop_at_waypoints, bool):
        raise ValueError("stop_at_waypoints must be a boolean")
    if (
        not math.isfinite(_native_number(waypoint_hold_seconds, "waypoint hold"))
        or not 0.0 <= waypoint_hold_seconds <= MAX_WAYPOINT_HOLD_SECONDS
    ):
        raise ValueError(
            f"waypoint_hold_seconds must be between 0 and {MAX_WAYPOINT_HOLD_SECONDS:g} seconds"
        )

    takeoff = TrackPoint(0.0, 0.0, max(0.5, points[0].z))
    schedule: list[Setpoint] = []

    takeoff_hold_samples = max(3, int(rate_hz * 2.0))
    if takeoff_hold_samples > MAX_SETPOINTS:
        raise ValueError(f"setpoint schedule exceeds the {MAX_SETPOINTS}-sample limit")
    for _ in range(takeoff_hold_samples):
        schedule.append(enu_point_to_ned_setpoint(takeoff, yaw_deg=0.0))

    if hover_duration_seconds is not None:
        if (
            not math.isfinite(_native_number(hover_duration_seconds, "hover duration"))
            or not MIN_HOVER_DURATION_SECONDS
            <= hover_duration_seconds
            <= MAX_HOVER_DURATION_SECONDS
        ):
            raise ValueError(
                "hover_duration_seconds must be between "
                f"{MIN_HOVER_DURATION_SECONDS:g} and {MAX_HOVER_DURATION_SECONDS:g}"
            )
        anchor = points[0]
        if any(
            math.dist(
                (point.x, point.y, point.z),
                (anchor.x, anchor.y, anchor.z),
            )
            > 1e-9
            for point in points
        ):
            raise ValueError("hover reference track must contain one stationary anchor")
        if abs(anchor.x) > 1e-9 or abs(anchor.y) > 1e-9:
            raise ValueError("hover anchor must remain at local origin x=0, y=0")
        hover_samples = max(2, int(math.ceil(rate_hz * hover_duration_seconds)) + 1)
        if hover_samples > MAX_SETPOINTS - len(schedule):
            raise ValueError(f"setpoint schedule exceeds the {MAX_SETPOINTS}-sample limit")
        hover_setpoint = enu_point_to_ned_setpoint(anchor, yaw_deg=0.0)
        track_start_index = len(schedule)
        schedule.extend(hover_setpoint for _ in range(hover_samples))
        plan = SetpointSchedulePlan(
            schedule=schedule,
            track_start_index=track_start_index,
            track_end_index=len(schedule) - 1,
            waypoint_arrival_indices=(),
        )
        return plan

    ingress = _build_motion_setpoints(
        takeoff,
        [points[0]],
        params,
        rate_hz,
        max_samples=MAX_SETPOINTS - len(schedule),
    )
    schedule.extend(ingress)
    track_start_index = len(schedule) - 1

    track_motion: list[Setpoint] = []
    waypoint_arrival_indices: list[int] = []
    if stop_at_waypoints:
        track_motion, relative_arrivals = build_stopped_waypoint_schedule(
            points,
            params,
            rate_hz,
            waypoint_hold_seconds,
            initial_yaw_deg=ingress[-1].yaw_deg if ingress else 0.0,
            max_samples=MAX_SETPOINTS - len(schedule),
        )
        waypoint_arrival_indices = [len(schedule) + index for index in relative_arrivals]
        schedule.extend(track_motion)
    else:
        track_motion = _build_motion_setpoints(
            points[0],
            points[1:],
            params,
            rate_hz,
            max_samples=MAX_SETPOINTS - len(schedule),
        )
        schedule.extend(track_motion)

    final_hold_samples = max(2, int(rate_hz * 0.5))
    if final_hold_samples > MAX_SETPOINTS - len(schedule):
        raise ValueError(f"setpoint schedule exceeds the {MAX_SETPOINTS}-sample limit")
    final_yaw = (
        track_motion[-1].yaw_deg if track_motion else ingress[-1].yaw_deg if ingress else 0.0
    )
    final_setpoint = enu_point_to_ned_setpoint(points[-1], yaw_deg=final_yaw)
    schedule.extend(final_setpoint for _ in range(final_hold_samples))

    track_end_index = len(schedule) - 1
    plan = SetpointSchedulePlan(
        schedule=schedule,
        track_start_index=track_start_index,
        track_end_index=track_end_index,
        waypoint_arrival_indices=tuple(waypoint_arrival_indices),
    )
    return plan


# 功能：
#   在起飞前拒绝非数值、非有限或不符合正值约束的稳定门限。
# 输入：
#   value：待验证的距离、速度或时间门限。
#   label：错误中对应的参数名称。
#   allow_zero：是否允许零值。
# 输出：
#   None：合法值通过，非法值抛出异常。
def _validate_takeoff_gate_limit(value: float, label: str, *, allow_zero: bool = False) -> None:
    value = _native_number(value, label)
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{label} must be finite and {comparator}")


# 功能：
#   生成严格布尔健康快照，缺失遥测和形似真值的字符串不能授权解锁。
# 输入：
#   health：飞控返回的健康状态，None 表示尚无观测。
# 输出：
#   payload：各项真实布尔就绪状态。
def _health_payload(health: TelemetryHealth | None) -> dict[str, bool]:
    payload = {
        name: getattr(health, name, None) is True
        for name in (
            "connected",
            "global_position_ok",
            "home_position_ok",
            "local_position_ok",
            "armable",
        )
    }
    return payload


# 功能：
#   从真实 NED 位置速度计算相对目标的水平、垂直误差，不把位置吻合当作动作成功。
# 输入：
#   sample：一次完整的有限数值遥测。
#   target：当前稳定保持目标。
# 输出：
#   payload：原始位置速度及以米、米每秒表示的误差和速度幅度。
def _position_velocity_payload(
    sample: PositionVelocityNed,
    target: Setpoint,
) -> dict[str, float | bool]:
    _validate_control_setpoint(target)
    for name in ("north_m", "east_m", "down_m", "north_m_s", "east_m_s", "down_m_s"):
        _native_number(getattr(sample, name), "telemetry " + name)
    horizontal_error_m = math.hypot(
        sample.north_m - target.north_m,
        sample.east_m - target.east_m,
    )
    vertical_error_m = abs(sample.down_m - target.down_m)
    horizontal_speed_m_s = math.hypot(sample.north_m_s, sample.east_m_s)
    vertical_speed_m_s = abs(sample.down_m_s)
    payload = {
        "north_m": sample.north_m,
        "east_m": sample.east_m,
        "down_m": sample.down_m,
        "north_m_s": sample.north_m_s,
        "east_m_s": sample.east_m_s,
        "down_m_s": sample.down_m_s,
        "horizontal_error_m": horizontal_error_m,
        "vertical_error_m": vertical_error_m,
        "horizontal_speed_m_s": horizontal_speed_m_s,
        "vertical_speed_m_s": vertical_speed_m_s,
    }
    for name, value in payload.items():
        _native_number(value, name)
    return payload


# 功能：
#   等待异步设备操作时持续维持控制，支持每次发送前重新仲裁目标，终止时回收未完成操作。
# 输入：
#   client：实际飞控客户端。
#   operation：需要并行等待的异步操作。
#   hold_setpoint：未提供刷新回调时的稳定目标。
#   rate_hz：控制刷新频率。
#   abort_check：每轮和返回结果前执行的终止检查。
#   setpoint_refresh：发送前的实时本地控制回调，None 保留显式固定悬停模式。
# 输出：
#   result：异步操作的实际返回值。
async def _await_with_setpoint_keepalive(
    client: OffboardClientProtocol,
    operation: Awaitable[_T],
    *,
    hold_setpoint: Setpoint,
    rate_hz: float,
    abort_check: Callable[[], None] | None = None,
    setpoint_refresh: Callable[[Setpoint], Awaitable[Setpoint]] | None = None,
) -> _T:
    """Await a bounded control operation without starving PX4 Offboard input."""

    operation_task = asyncio.ensure_future(operation)
    try:
        rate_hz = _native_number(rate_hz, "Offboard keepalive rate")
        if rate_hz <= 0:
            raise ValueError("Offboard keepalive rate must be positive")
        _validate_control_setpoint(hold_setpoint)
        while not operation_task.done():
            if abort_check is not None:
                abort_check()
            current_setpoint = (
                await setpoint_refresh(hold_setpoint)
                if setpoint_refresh is not None
                else hold_setpoint
            )
            if abort_check is not None:
                abort_check()
            await client.set_position_ned(current_setpoint)
            if operation_task.done():
                break
            await asyncio.wait({operation_task}, timeout=1.0 / rate_hz)
        if abort_check is not None:
            abort_check()
        result = operation_task.result()
        return result
    finally:
        primary = sys.exception()
        if not operation_task.done():
            operation_task.cancel()
            try:
                await operation_task
            except asyncio.CancelledError:
                pass
            except Exception as cleanup_error:
                if primary is None:
                    raise
                # 清理错误不能抹掉带 world_paused 的外部停止，否则上层可能误发降落。
                primary.add_note(
                    "Offboard operation cleanup failed: " + type(cleanup_error).__name__
                )
        elif not operation_task.cancelled():
            # 停止检查可能先于 result() 抛出；已完成失败也须领取，不能遗留异步警告。
            operation_error = operation_task.exception()
            if (
                operation_error is not None
                and primary is not None
                and operation_error is not primary
            ):
                primary.add_note(
                    "Completed Offboard operation failed: " + type(operation_error).__name__
                )


# 功能：
#   等待设备操作时周期检查停止请求，完成与停止同时到达时仍优先停止。
# 输入：
#   operation：由本函数拥有和清理的异步操作。
#   abort_check：同步停止检查函数。
#   poll_interval_seconds：两次检查之间的最长等待秒数。
# 输出：
#   result：未被停止的实际操作结果。
async def _await_with_abort_polling(
    operation: Awaitable[_T],
    *,
    abort_check: Callable[[], None],
    poll_interval_seconds: float = ABORT_POLL_INTERVAL_SECONDS,
) -> _T:
    """Await an operation while cancelling it promptly on an external stop."""

    operation_task = asyncio.ensure_future(operation)
    try:
        poll_interval_seconds = _native_number(poll_interval_seconds, "abort poll interval")
        if not 0 < poll_interval_seconds <= 1.0:
            raise ValueError("abort poll interval must be in (0, 1] seconds")
        while not operation_task.done():
            abort_check()
            await asyncio.wait({operation_task}, timeout=poll_interval_seconds)
        abort_check()
        result = operation_task.result()
        return result
    finally:
        primary = sys.exception()
        if not operation_task.done():
            operation_task.cancel()
            try:
                await operation_task
            except asyncio.CancelledError:
                pass
            except Exception as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "Awaited operation cleanup failed: " + type(cleanup_error).__name__
                )
        elif not operation_task.cancelled():
            operation_error = operation_task.exception()
            if (
                operation_error is not None
                and primary is not None
                and operation_error is not primary
            ):
                primary.add_note(
                    "Completed awaited operation failed: " + type(operation_error).__name__
                )


# 功能：
#   沿原因链收集异常，检测循环引用以避免错误诊断自身卡死。
# 输入：
#   error：最外层异常。
# 输出：
#   chain：从外到内且不重复的异常列表。
def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


# 功能：
#   仅识别解锁前可重连的 MAVSDK 传输断开，不把估计器或飞行故障归为可重试。
# 输入：
#   error：连接阶段发生的异常。
# 输出：
#   recoverable：异常链是否明确表示 gRPC 服务不可用。
def _is_recoverable_mavsdk_preflight_transport_error(error: BaseException) -> bool:
    """Recognize a dead embedded MAVSDK gRPC transport, not a flight fault."""

    for item in _exception_chain(error):
        code_method = getattr(item, "code", None)
        if callable(code_method):
            with contextlib.suppress(Exception):
                code = code_method()
                code_name = str(getattr(code, "name", code)).upper()
                if code_name == "UNAVAILABLE" or code_name.endswith(".UNAVAILABLE"):
                    return True
        type_name = type(item).__name__.casefold()
        message = str(item).casefold()
        if "rpcerror" in type_name and any(
            marker in message
            for marker in (
                "statuscode.unavailable",
                "connection reset by peer",
                "stream removed",
                "socket closed",
            )
        ):
            return True
    return False


# 功能：
#   在总超时内完成连接、未解锁确认、就绪及速率请求检查，临时失败先回收再有限重连。
# 输入：
#   client：当前飞控客户端。
#   connection：本次连接地址。
#   readiness_timeout_seconds：包括重试和遥测配置的总秒数预算。
#   abort_check：外部停止检查。
#   log_path：本次运行日志。
#   evidence：逐次记录阶段和结果的证据容器。
#   maximum_attempts：最多允许的连接次数，范围一到三。
# 输出：
#   health：真实就绪遥测；无法恢复时抛出原异常或超时。
async def connect_preflight_with_recovery(
    client: OffboardClientProtocol,
    *,
    connection: str,
    readiness_timeout_seconds: float,
    abort_check: Callable[[], None],
    log_path: Path,
    evidence: dict[str, Any],
    maximum_attempts: int = MAVSDK_PREFLIGHT_MAX_ATTEMPTS,
) -> TelemetryHealth:
    """Connect and pass readiness with bounded, pre-arm-only recovery attempts.

    This helper must be called before any arm or Offboard command.  It retries
    only an unavailable gRPC transport, a connect/rate timeout or a transient
    required-rate failure, closes the owned server, and records every attempt. Estimator
    readiness failures, safety aborts, and every in-flight transport failure
    remain fail-closed and are never retried here.
    """

    readiness_timeout_seconds = _timeout_budget(readiness_timeout_seconds)
    if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 3:
        raise ValueError("MAVSDK preflight attempts must be an integer within [1, 3]")

    from dronedream_agent_core.preflight_recovery import (
        TelemetryRateSetupPending,
        run_preflight_recovery,
    )

    # 功能：
    #   完成一个连接、飞控就绪和必需遥测配置周期，不发送解锁或飞行命令。
    # 输入：
    #   remaining：共享总期限内剩余的秒数。
    #   row：本次准备阶段的证据容器。
    # 输出：
    #   health：飞控实际报告的就绪状态。
    async def prepare(remaining: float, row: dict) -> TelemetryHealth:
        deadline = time.monotonic() + remaining
        await _await_with_abort_polling(
            asyncio.wait_for(client.connect(connection),
                             timeout=min(MAVSDK_PREFLIGHT_CONNECT_TIMEOUT_SECONDS, remaining)),
            abort_check=abort_check)
        server_port = getattr(client, "_mavsdk_server_port", None)
        if type(server_port) is int:
            row["mavsdk_server_port"] = server_port
        disarmed_check = getattr(client, "verify_disarmed_before_preflight", None)
        if callable(disarmed_check):
            row["stage"] = "disarmed-check"
            row["disarmed_check"] = await _await_with_abort_polling(
                disarmed_check(), abort_check=abort_check)
        row["stage"] = "readiness"
        health = await _await_with_abort_polling(
            client.wait_until_ready(max(.001, deadline-time.monotonic())),
            abort_check=abort_check)
        if any(getattr(health, field, None) is not True for field in
               ("connected", "home_position_ok", "local_position_ok", "armable")):
            raise RuntimeError("PREFLIGHT_FIRMWARE_NOT_READY: incomplete readiness")
        rate_configurer = getattr(client, "configure_dynamics_telemetry_rates", None)
        if callable(rate_configurer):
            row["stage"] = "dynamics-telemetry-rates"
            rates = await _await_with_abort_polling(
                asyncio.wait_for(rate_configurer(), timeout=min(5.,
                    max(.001, deadline-time.monotonic()))), abort_check=abort_check)
            row["dynamics_telemetry_rates"] = rates
            required = ("position_velocity", "imu", "attitude", "odometry", "battery")
            if not isinstance(rates, dict) or not isinstance(rates.get("sources"), dict):
                raise RuntimeError("PREFLIGHT_TELEMETRY_RATE_RECEIPT_INVALID")
            sources = rates["sources"]
            if any(not isinstance(sources.get(key), dict)
                   or sources[key].get("status") not in {"requested", "failed", "unsupported"}
                   for key in required):
                raise RuntimeError("PREFLIGHT_TELEMETRY_RATE_RECEIPT_INVALID")
            if any(sources[key]["status"] == "unsupported" for key in required):
                raise RuntimeError("PREFLIGHT_REQUIRED_TELEMETRY_UNSUPPORTED")
            if any(sources[key]["status"] == "failed" for key in required):
                raise TelemetryRateSetupPending("PREFLIGHT_TELEMETRY_RATE_REQUEST_FAILED")
            if rates.get("required_rate_requests_succeeded") is not True:
                raise RuntimeError("PREFLIGHT_TELEMETRY_RATE_RECEIPT_INVALID")
        _log(log_path, f"preflight transport ready on attempt {row['attempt']}/{maximum_attempts}")
        return health

    # 功能：
    #   区分临时连接或采样率请求故障与不可重试的安全、配置及估计器故障。
    # 输入：
    #   error：当前异常。
    #   row：发生异常的准备阶段。
    # 输出：
    #   recoverable：是否允许在解锁前清理并重连。
    def retryable(error: BaseException, row: dict) -> bool:
        recoverable = bool(
            isinstance(error, TelemetryRateSetupPending)
            or (row["stage"] in {"connect", "dynamics-telemetry-rates"}
                and isinstance(error, TimeoutError))
            or _is_recoverable_mavsdk_preflight_transport_error(error))
        return recoverable

    health = await run_preflight_recovery(
        attempt=prepare, recover=client.close, retryable=retryable,
        motion_requested=lambda: getattr(client, "_flight_command_requested", False),
        abort_check=abort_check, evidence=evidence, timeout_seconds=readiness_timeout_seconds,
        maximum_attempts=maximum_attempts)
    return health


# 功能：
#   1. 按受限竖直爬升斜坡持续发送指令，再以真实位置和速度验证连续稳定悬停。
#   2. 稳定窗口若在截止前开始，可有限完成该窗口；宽限中失稳立即失败。
# 输入：
#   client、target：飞控客户端和起飞后稳定目标。
#   takeoff_origin：实测起始位置，None 表示调用方明确选择直接目标模式。
#   climb_rate_m_s：竖直目标变化速率上限。
#   timeout_seconds、sample_rate_hz：起飞秒数预算与控制采样频率。
#   stable_window_seconds：要求的连续稳定时长。
#   horizontal_tolerance_m、vertical_tolerance_m：位置误差限。
#   horizontal_speed_tolerance_m_s、vertical_speed_tolerance_m_s：速度限。
#   evidence：保存样本、窗口重置和成败原因的容器。
#   abort_check：可选的实时停止检查。
# 输出：
#   None：真实观测满足稳定窗口时完成，失稳、断流或超时抛出异常。
async def _wait_for_takeoff_stability(
    client: OffboardClientProtocol,
    target: Setpoint,
    *,
    takeoff_origin: PositionVelocityNed | None = None,
    climb_rate_m_s: float = 1.0,
    timeout_seconds: float,
    sample_rate_hz: float,
    stable_window_seconds: float,
    horizontal_tolerance_m: float,
    vertical_tolerance_m: float,
    horizontal_speed_tolerance_m_s: float,
    vertical_speed_tolerance_m_s: float,
    evidence: dict[str, Any],
    abort_check: Callable[[], None] | None = None,
) -> None:
    for label, value, allow_zero in (
        ("timeout_seconds", timeout_seconds, False),
        ("sample_rate_hz", sample_rate_hz, False),
        ("climb_rate_m_s", climb_rate_m_s, False),
        ("stable_window_seconds", stable_window_seconds, True),
        ("horizontal_tolerance_m", horizontal_tolerance_m, False),
        ("vertical_tolerance_m", vertical_tolerance_m, False),
        (
            "horizontal_speed_tolerance_m_s",
            horizontal_speed_tolerance_m_s,
            False,
        ),
        ("vertical_speed_tolerance_m_s", vertical_speed_tolerance_m_s, False),
    ):
        _validate_takeoff_gate_limit(value, label, allow_zero=allow_zero)
    if climb_rate_m_s > MAX_TAKEOFF_CLIMB_RATE_M_S:
        raise ValueError(f"climb_rate_m_s must be no greater than {MAX_TAKEOFF_CLIMB_RATE_M_S:g}")
    _validate_control_setpoint(target)
    _timeout_budget(timeout_seconds)
    if stable_window_seconds > 3600:
        raise ValueError("takeoff stable window exceeds the supported budget")
    if takeoff_origin is not None:
        _position_velocity_payload(takeoff_origin, target)

    sample_interval_seconds = max(0.01, 1.0 / sample_rate_hz)
    telemetry_timeout_seconds = min(1.0, timeout_seconds)
    completion_sample_margin_seconds = telemetry_timeout_seconds + sample_interval_seconds
    deadline = time.monotonic() + timeout_seconds
    stability_completion_deadline: float | None = None
    ramp_started_at = time.monotonic()
    stable_since: float | None = None
    origin_down_m = target.down_m if takeoff_origin is None else takeoff_origin.down_m
    vertical_distance_m = target.down_m - origin_down_m
    evidence.update(
        {
            "schema_version": "dronedream.takeoff_gate.v4",
            "status": "waiting",
            "target_ned": {
                "north_m": target.north_m,
                "east_m": target.east_m,
                "down_m": target.down_m,
            },
            "tolerances": {
                "horizontal_position_m": horizontal_tolerance_m,
                "vertical_position_m": vertical_tolerance_m,
                "horizontal_speed_m_s": horizontal_speed_tolerance_m_s,
                "vertical_speed_m_s": vertical_speed_tolerance_m_s,
            },
            "required_stable_window_s": stable_window_seconds,
            "takeoff_profile": {
                "mode": (
                    "direct_target"
                    if takeoff_origin is None
                    else "telemetry_anchored_bounded_vertical_ramp"
                ),
                "climb_rate_limit_m_s": climb_rate_m_s,
                "origin_ned": (
                    None
                    if takeoff_origin is None
                    else {
                        "north_m": takeoff_origin.north_m,
                        "east_m": takeoff_origin.east_m,
                        "down_m": takeoff_origin.down_m,
                    }
                ),
                "vertical_distance_m": abs(vertical_distance_m),
            },
            "sample_count": 0,
            "stable_sample_count": 0,
            "reset_count": 0,
            "stability_completion_grace": {
                "granted": False,
                "used": False,
                "maximum_seconds": (stable_window_seconds + completion_sample_margin_seconds),
            },
            "observations": [],
        }
    )

    while True:
        if abort_check is not None:
            abort_check()
        now = time.monotonic()
        active_deadline = (
            deadline if stability_completion_deadline is None else stability_completion_deadline
        )
        remaining = active_deadline - now
        if remaining <= 0:
            latest = evidence.get("latest_observation")
            evidence["status"] = "failed"
            evidence["failure_reason"] = "takeoff_stability_timeout"
            raise TimeoutError(
                "takeoff did not reach a continuously stable hover within "
                f"{timeout_seconds:g}s; latest={latest}"
            )

        ramp_elapsed_seconds = max(0.0, now - ramp_started_at)
        maximum_vertical_delta_m = climb_rate_m_s * ramp_elapsed_seconds
        if abs(vertical_distance_m) <= maximum_vertical_delta_m:
            commanded_down_m = target.down_m
            ramp_fraction = 1.0
        else:
            commanded_down_m = origin_down_m + math.copysign(
                maximum_vertical_delta_m,
                vertical_distance_m,
            )
            ramp_fraction = (
                1.0
                if abs(vertical_distance_m) <= 1e-12
                else maximum_vertical_delta_m / abs(vertical_distance_m)
            )
        commanded = Setpoint(
            north_m=target.north_m,
            east_m=target.east_m,
            down_m=commanded_down_m,
            yaw_deg=target.yaw_deg,
        )
        try:
            sample = await _await_with_setpoint_keepalive(
                client,
                client.sample_position_velocity_ned(min(telemetry_timeout_seconds, remaining)),
                hold_setpoint=commanded,
                rate_hz=sample_rate_hz,
                abort_check=abort_check,
            )
        except ExternalSafetyAbort:
            evidence["status"] = "failed"
            evidence["failure_reason"] = "external_safety_abort"
            raise
        except BaseException as exc:
            if evidence["sample_count"] > 0 and time.monotonic() >= active_deadline:
                latest = evidence.get("latest_observation")
                evidence["status"] = "failed"
                evidence["failure_reason"] = "takeoff_stability_timeout"
                evidence["terminal_telemetry_error"] = f"{type(exc).__name__}: {exc}"
                raise TimeoutError(
                    "takeoff did not reach a continuously stable hover within "
                    f"{timeout_seconds:g}s; latest={latest}"
                ) from exc
            evidence["status"] = "failed"
            evidence["failure_reason"] = "position_velocity_telemetry_unavailable"
            evidence["telemetry_error"] = f"{type(exc).__name__}: {exc}"
            raise

        observed_at = time.monotonic()
        payload: dict[str, Any] = _position_velocity_payload(sample, target)
        payload["commanded_setpoint_ned"] = {
            "north_m": commanded.north_m,
            "east_m": commanded.east_m,
            "down_m": commanded.down_m,
            "yaw_deg": commanded.yaw_deg,
        }
        payload["takeoff_ramp_fraction"] = min(1.0, max(0.0, ramp_fraction))
        payload["takeoff_ramp_elapsed_s"] = ramp_elapsed_seconds
        if ramp_fraction >= 1.0 and "ramp_completed_after_s" not in evidence["takeoff_profile"]:
            evidence["takeoff_profile"]["ramp_completed_after_s"] = ramp_elapsed_seconds
        within_limits = bool(
            float(payload["horizontal_error_m"]) <= horizontal_tolerance_m
            and float(payload["vertical_error_m"]) <= vertical_tolerance_m
            and float(payload["horizontal_speed_m_s"]) <= horizontal_speed_tolerance_m_s
            and float(payload["vertical_speed_m_s"]) <= vertical_speed_tolerance_m_s
        )
        payload["within_all_limits"] = within_limits
        evidence["sample_count"] = int(evidence["sample_count"]) + 1
        if within_limits:
            evidence["stable_sample_count"] = int(evidence["stable_sample_count"]) + 1
            if stable_since is None:
                stable_since = observed_at
            required_completion_at = (
                stable_since + stable_window_seconds + completion_sample_margin_seconds
            )
            if stable_since < deadline and required_completion_at > deadline:
                stability_completion_deadline = min(
                    required_completion_at,
                    deadline + stable_window_seconds + completion_sample_margin_seconds,
                )
                grace = evidence["stability_completion_grace"]
                if isinstance(grace, dict):
                    grace["granted"] = True
                    grace["seconds"] = max(
                        0.0,
                        stability_completion_deadline - deadline,
                    )
                    grace["qualification_started_before_timeout"] = True
        else:
            if stability_completion_deadline is not None and observed_at >= deadline:
                payload["stable_duration_s"] = 0.0
                evidence["latest_observation"] = payload
                observations = evidence["observations"]
                if isinstance(observations, list):
                    observations.append(payload)
                evidence["status"] = "failed"
                evidence["failure_reason"] = "takeoff_stability_lost_during_completion_grace"
                grace = evidence["stability_completion_grace"]
                if isinstance(grace, dict):
                    grace["used"] = True
                    grace["completed"] = False
                raise TimeoutError(
                    "takeoff left the stability envelope while completing "
                    "a continuous hover window that began before the timeout"
                )
            if stable_since is not None:
                evidence["reset_count"] = int(evidence["reset_count"]) + 1
            stable_since = None
            stability_completion_deadline = None

        stable_duration = 0.0 if stable_since is None else observed_at - stable_since
        payload["stable_duration_s"] = stable_duration
        evidence["latest_observation"] = payload
        observations = evidence["observations"]
        if isinstance(observations, list):
            observations.append(payload)

        if within_limits and stable_duration >= stable_window_seconds:
            evidence["status"] = "achieved"
            evidence["achieved_stable_window_s"] = stable_duration
            grace = evidence["stability_completion_grace"]
            if isinstance(grace, dict) and bool(grace.get("granted")):
                grace["used"] = observed_at >= deadline
                grace["completed"] = True
            return

        active_deadline = (
            deadline if stability_completion_deadline is None else stability_completion_deadline
        )
        await asyncio.sleep(
            min(sample_interval_seconds, max(0.0, active_deadline - time.monotonic()))
        )


# 功能：
#   读取浮点参数前值、写入请求并再次读回，以浮点传输容差验证实际应用。
# 输入：
#   client：飞控参数客户端。
#   name：参数名称。
#   value：要求的有限数值。
# 输出：
#   record：实际前值、请求值与已验证后值。
async def _set_float_parameter_verified(
    client: OffboardClientProtocol,
    name: str,
    value: float,
) -> dict[str, float]:
    value = _native_number(value, name)
    before = _native_number(await client.get_param_float(name), name)
    await client.set_param_float(name, value)
    applied = _native_number(await client.get_param_float(name), name)
    if not math.isclose(applied, value, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            f"PX4 parameter {name} readback mismatch: requested={value:g}, applied={applied:g}"
        )
    record = {"before": before, "requested": value, "applied": applied}
    return record


# 功能：
#   依次应用一组浮点参数，每项读回通过后才继续下一项。
# 输入：
#   client：飞控参数客户端。
#   values：按顺序排列的参数名与目标值。
# 输出：
#   applied：每项参数的前值、请求值及后值记录。
async def _set_float_parameters_verified(
    client: OffboardClientProtocol,
    values: dict[str, float],
) -> dict[str, dict[str, float]]:
    applied: dict[str, dict[str, float]] = {}
    for name, value in values.items():
        applied[name] = await _set_float_parameter_verified(client, name, value)
    return applied


# 功能：
#   写入并精确读回整型参数，拒绝把参数命令确认当成应用证明。
# 输入：
#   client：飞控参数客户端。
#   name：参数名称。
#   value：严格整数目标。
# 输出：
#   record：前值、请求值及精确匹配的实际值。
async def _set_int_parameter_verified(
    client: OffboardClientProtocol,
    name: str,
    value: int,
) -> dict[str, int]:
    if type(value) is not int:
        raise ValueError("verified integer parameter requires a native integer")
    before = await client.get_param_int(name)
    await client.set_param_int(name, value)
    applied = await client.get_param_int(name)
    if type(before) is not int or type(applied) is not int or applied != value:
        raise RuntimeError(
            f"PX4 parameter {name} readback mismatch: requested={value}, applied={applied}"
        )
    record = {"before": before, "requested": value, "applied": applied}
    return record


# 功能：
#   在解锁前配置并读回飞控自身的外部控制失联保持策略，伴随计算机断流后仍由固件接管。
# 输入：
#   client：真实飞控参数客户端。
# 输出：
#   evidence：保持动作、失联时限和两项实际参数读回。
async def configure_offboard_loss_failsafe(client: OffboardClientProtocol) -> dict[str, Any]:
    """Require PX4 itself to hold if the companion Offboard stream disappears.

    A MAVSDK gRPC failure can make Python unable to send a land command even
    while PX4 is healthy.  The independent firmware failsafe therefore owns
    the last line of defence: after one second without Offboard setpoints it
    enters Hold instead of attempting an unplanned landing over stairs or a
    pickup fixture.  Every parameter is read back before arming.
    """

    evidence = {
        "schema_version": "dronedream.px4-offboard-loss-failsafe.v1",
        "action": "HOLD",
        "timeout": await _set_float_parameter_verified(
            client,
            "COM_OF_LOSS_T",
            PX4_OFFBOARD_LOSS_TIMEOUT_SECONDS,
        ),
        "mode": await _set_int_parameter_verified(
            client,
            "COM_OBL_RC_ACT",
            PX4_OFFBOARD_LOSS_HOLD_ACTION,
        ),
        "verified": True,
    }
    return evidence


# 功能：
#   注入或恢复仿真 GPS 卫星状态，同时验证参数和后续 GPS 遥测。
# 输入：
#   client：仿真飞控客户端。
#   satellites_used：目标卫星数量。
#   unavailable：期望为失效还是恢复状态。
# 输出：
#   evidence：参数读回及符合预期的实际遥测样本。
async def _set_gps_availability_verified(
    client: OffboardClientProtocol,
    *,
    satellites_used: int,
    unavailable: bool,
) -> dict[str, Any]:
    if (
        type(unavailable) is not bool
        or type(satellites_used) is not int
        or not 0 <= satellites_used <= 255
    ):
        raise ValueError("GPS availability request is invalid")
    parameter = await _set_int_parameter_verified(
        client,
        "SIM_GPS_USED",
        satellites_used,
    )
    samples: list[dict[str, int | str]] = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        sample = _validated_gps_telemetry(
            await client.sample_gps_info(min(1.0, max(0.01, deadline - time.monotonic()))),
            label="PX4 GPS availability telemetry",
        )
        samples.append(sample)
        num_satellites = sample["num_satellites"]
        fix_type = sample["fix_type"]
        if isinstance(num_satellites, bool) or not isinstance(num_satellites, int):
            raise ValueError("PX4 GPS availability num_satellites must remain an integer")
        if isinstance(fix_type, bool) or not isinstance(fix_type, int):
            raise ValueError("PX4 GPS availability fix_type must remain an integer")
        observed = (
            num_satellites < 4 and fix_type <= 1
            if unavailable
            else num_satellites >= 4 and fix_type >= 2
        )
        if observed:
            evidence = {
                "parameter_name": "SIM_GPS_USED",
                "parameter": parameter,
                "expected_availability": "unavailable" if unavailable else "available",
                "telemetry_samples": samples,
                "physical_effect_verified": True,
            }
            return evidence
        await asyncio.sleep(0.1)
    expected = "unavailable" if unavailable else "available"
    raise RuntimeError(
        f"PX4 GPS telemetry did not become {expected} after SIM_GPS_USED="
        f"{satellites_used}; samples={samples!r}"
    )


# 功能：
#   在稳定起飞后设置仿真电池下限及消耗速率，使受测轨迹从要求的电量开始。
# 输入：
#   client：仿真飞控参数客户端。
#   profile：包含轨迹起始百分比的编译配置。
#   takeoff_hold_seconds：轨迹接入前预计保持时长。
# 输出：
#   prepared：目标电量、消耗时长与参数读回。
async def _prepare_battery_profile(
    client: OffboardClientProtocol,
    profile: dict[str, Any],
    *,
    takeoff_hold_seconds: float,
) -> dict[str, Any]:
    target = _native_number(profile["target_track_start_percent"], "target battery percent")
    if not 0 <= target <= 100 or _native_number(takeoff_hold_seconds, "takeoff hold") <= 0:
        raise ValueError("battery preparation target or hold is invalid")
    if target >= 100.0 - 1e-12:
        pretrack_drain_seconds = 86400.0
    else:
        pretrack_drain_seconds = max(
            1.0,
            min(86400.0, takeoff_hold_seconds / max(1e-9, 1.0 - target / 100.0)),
        )
    parameters = {
        "SIM_BAT_MIN_PCT": await _set_float_parameter_verified(
            client,
            "SIM_BAT_MIN_PCT",
            target,
        ),
        "SIM_BAT_DRAIN": await _set_float_parameter_verified(
            client,
            "SIM_BAT_DRAIN",
            pretrack_drain_seconds,
        ),
    }
    prepared = {
        "target_track_start_percent": target,
        "takeoff_hold_seconds": takeoff_hold_seconds,
        "pretrack_drain_seconds": pretrack_drain_seconds,
        "pretrack_parameters": parameters,
    }
    return prepared


# 功能：
#   将显式电池工况的计时起点移到轨迹阶段，起飞稳定门期间保持满电仿真条件。
# 输入：
#   client：仿真飞控参数客户端。
# 输出：
#   parameters：电量下限和长消耗周期的真实读回记录。
async def _hold_battery_during_takeoff_gate(
    client: OffboardClientProtocol,
) -> dict[str, dict[str, float]]:
    parameters = {
        "SIM_BAT_MIN_PCT": await _set_float_parameter_verified(
            client,
            "SIM_BAT_MIN_PCT",
            100.0,
        ),
        "SIM_BAT_DRAIN": await _set_float_parameter_verified(
            client,
            "SIM_BAT_DRAIN",
            86400.0,
        ),
    }
    return parameters


# 功能：
#   在总截止前读取一次电池遥测，同时持续发送保持指令以免控制断流。
# 输入：
#   client：飞控客户端。
#   hold_setpoint、rate_hz：等待期间的保持目标与发送频率。
#   deadline：单调时钟上的总截止时刻。
# 输出：
#   sample：实际电池样本，开始读取前预算已耗尽则为 None。
async def _sample_battery_while_holding(
    client: OffboardClientProtocol,
    *,
    hold_setpoint: Setpoint,
    rate_hz: float,
    deadline: float,
) -> dict[str, float] | None:
    """Keep the Offboard heartbeat alive while awaiting one battery sample."""

    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return None
    sample = await _await_with_setpoint_keepalive(
        client,
        client.sample_battery(min(5.0, remaining)),
        hold_setpoint=hold_setpoint,
        rate_hz=rate_hz,
    )
    return sample


# 功能：
#   等待真实电量进入起始容差，再切换受测消耗工况；电量过冲或超时均停止进入轨迹。
# 输入：
#   client：仿真飞控客户端。
#   profile：目标电量、压降开关和消耗周期。
#   prepared：准备阶段参数及读回。
#   hold_setpoint、rate_hz：等待期间保持控制的目标和频率。
#   settle_timeout_seconds：电量稳定的总秒数预算。
# 输出：
#   result：完整采样历史、起始电量、容差和轨迹工况参数读回。
async def _transition_battery_at_track_start(
    client: OffboardClientProtocol,
    profile: dict[str, Any],
    prepared: dict[str, Any],
    *,
    hold_setpoint: Setpoint,
    rate_hz: float,
    settle_timeout_seconds: float = BATTERY_TRACK_START_SETTLE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    target = _native_number(profile["target_track_start_percent"], "battery target")
    drain_seconds = _native_number(prepared["pretrack_drain_seconds"], "battery drain seconds")
    if not 0 <= target <= 100 or drain_seconds <= 0 or type(profile["voltage_sag"]) is not bool:
        raise ValueError("battery transition profile is invalid")
    quantization_tolerance = 100.0 * 0.1 / max(1.0, drain_seconds)
    tolerance = max(5.0, quantization_tolerance + 2.0)
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("battery conditioning setpoint rate must be finite and greater than zero")
    if not math.isfinite(settle_timeout_seconds) or settle_timeout_seconds <= 0.0:
        raise ValueError("battery settle timeout must be finite and greater than zero")
    deadline = time.monotonic() + settle_timeout_seconds
    conditioning_samples: list[dict[str, float]] = []
    conditioning_sample_timeout_count = 0
    track_start_sample: dict[str, float] | None = None
    while True:
        try:
            sample = await _sample_battery_while_holding(
                client,
                hold_setpoint=hold_setpoint,
                rate_hz=rate_hz,
                deadline=deadline,
            )
        except TimeoutError:
            conditioning_sample_timeout_count += 1
            if time.monotonic() >= deadline:
                break
            continue
        if sample is None:
            break
        sample = _validated_battery_telemetry(
            sample,
            label="PX4 battery conditioning telemetry",
        )
        conditioning_samples.append(sample)
        sample_percent = sample["remaining_percent"]
        if abs(sample_percent - target) <= tolerance:
            track_start_sample = sample
            break
        if sample_percent < target - tolerance:
            raise RuntimeError(
                "PX4 battery overshot the requested track-start state: "
                f"target={target:g}%, observed={sample_percent:g}%, tolerance={tolerance:g}%"
            )
        await asyncio.sleep(min(1.0 / rate_hz, max(0.0, deadline - time.monotonic())))
    if track_start_sample is None:
        observed = (
            float(conditioning_samples[-1]["remaining_percent"])
            if conditioning_samples
            else math.nan
        )
        raise RuntimeError(
            "PX4 battery did not reach the requested track-start state before timeout: "
            f"target={target:g}%, observed={observed:g}%, tolerance={tolerance:g}%, "
            f"samples={len(conditioning_samples)}, timeout={settle_timeout_seconds:g}s"
        )
    if profile["voltage_sag"]:
        transition_values = {
            "SIM_BAT_MIN_PCT": 0.0,
            "SIM_BAT_DRAIN": float(profile["sag_drain_seconds"]),
        }
    else:
        transition_values = {
            "SIM_BAT_MIN_PCT": target,
            "SIM_BAT_DRAIN": float(profile["no_sag_hold_drain_seconds"]),
        }
    transition_parameters = await _await_with_setpoint_keepalive(
        client,
        _set_float_parameters_verified(client, transition_values),
        hold_setpoint=hold_setpoint,
        rate_hz=rate_hz,
    )
    result = {
        **prepared,
        "conditioning_sample_count": len(conditioning_samples),
        "conditioning_sample_timeout_count": conditioning_sample_timeout_count,
        "conditioning_samples": conditioning_samples,
        "conditioning_timeout_seconds": settle_timeout_seconds,
        "track_start_sample": track_start_sample,
        "track_start_tolerance_percent": tolerance,
        "track_parameters": transition_parameters,
    }
    return result


# 功能：
#   1. 执行显式参考轨迹鉴定：真实连接、起飞稳定门、限时轨迹及落地读回，不冒充本地模型闭环。
#   2. 将可选风场、GPS 失效与电池工况绑定到实际飞行阶段，并保存读回证据。
#   3. 停止或失败时独立尝试恢复工况、退出控制、确认落地及释放客户端，保留失败原因。
# 输入：
#   client、connection：实际飞控客户端及连接地址。
#   schedule、rate_hz：参考位置设定值和发送频率。
#   takeoff_timeout_seconds、takeoff_climb_rate_m_s：起飞超时与爬升速率。
#   track_timeout_seconds、landing_timeout_seconds：轨迹及降落秒数预算。
#   land_after：结束后是否请求并验证降落。
#   log_path、timing_path、runtime_phase_path：本次日志、时序和阶段证据路径。
#   track_start_index、track_end_index：调度中受测轨迹的边界。
#   scenario_engine、scenario_request、runtime_profile：可选场景请求及编译后的物理工况。
#   runtime_evidence_path：工况读回证据路径。
#   takeoff_stable_window_seconds：必须连续满足稳定条件的时长。
#   takeoff_horizontal_tolerance_m、takeoff_vertical_tolerance_m：起飞位置误差上限。
#   takeoff_horizontal_speed_tolerance_m_s、takeoff_vertical_speed_tolerance_m_s：起飞速度上限。
#   world、gazebo_vehicle_model_name：本次 Gazebo 世界及车辆精确身份。
#   abort_file：外部安全停止通道。
#   wind_activator：支持取消的风场激活器。
#   heading_policy、maximum_yaw_rate_deg_s：航向策略和转向速率上限。
# 输出：
#   None：成功时证据完成；任何必要控制或读回失败均抛出异常。
async def run_executor(
    client: OffboardClientProtocol,
    schedule: list[Setpoint],
    *,
    connection: str,
    takeoff_timeout_seconds: float,
    takeoff_climb_rate_m_s: float = 1.0,
    track_timeout_seconds: float,
    landing_timeout_seconds: float = 60.0,
    rate_hz: float,
    land_after: bool,
    log_path: Path,
    track_start_index: int = 0,
    track_end_index: int | None = None,
    timing_path: Path | None = None,
    runtime_phase_path: Path | None = None,
    scenario_engine: Any | None = None,
    scenario_request: dict[str, Any] | None = None,
    runtime_profile: dict[str, Any] | None = None,
    runtime_evidence_path: Path | None = None,
    takeoff_stable_window_seconds: float = 0.0,
    takeoff_horizontal_tolerance_m: float = 0.35,
    takeoff_vertical_tolerance_m: float = 0.25,
    takeoff_horizontal_speed_tolerance_m_s: float = 0.35,
    takeoff_vertical_speed_tolerance_m_s: float = 0.25,
    world: str = "default",
    abort_file: Path | None = None,
    wind_activator: Callable[..., dict[str, Any]] = _activate_gazebo_wind_profile,
    heading_policy: str = "measured-hold",
    maximum_yaw_rate_deg_s: float = 20.0,
    gazebo_vehicle_model_name: str | None = None,
) -> None:
    rate_hz = _native_number(rate_hz, "rate_hz")
    if rate_hz <= 0 or rate_hz > MAX_SETPOINT_RATE_HZ:
        raise ValueError(f"rate_hz must be finite and in (0, {MAX_SETPOINT_RATE_HZ:g}]")
    for label, value in (
        ("takeoff_timeout_seconds", takeoff_timeout_seconds),
        ("takeoff_climb_rate_m_s", takeoff_climb_rate_m_s),
        ("track_timeout_seconds", track_timeout_seconds),
        ("landing_timeout_seconds", landing_timeout_seconds),
    ):
        if _native_number(value, label) <= 0:
            raise ValueError(f"{label} must be finite and greater than zero")
    if takeoff_climb_rate_m_s > MAX_TAKEOFF_CLIMB_RATE_M_S:
        raise ValueError(
            f"takeoff_climb_rate_m_s must be no greater than {MAX_TAKEOFF_CLIMB_RATE_M_S:g}"
        )
    if not schedule or len(schedule) > MAX_SETPOINTS:
        raise ValueError("setpoint schedule is empty or exceeds the sample limit")
    if type(land_after) is not bool:
        raise ValueError("land_after must be boolean")
    _validate_takeoff_gate_limit(maximum_yaw_rate_deg_s, "maximum yaw rate")
    for label, value in (
        ("stable window", takeoff_stable_window_seconds),
        ("horizontal tolerance", takeoff_horizontal_tolerance_m),
        ("vertical tolerance", takeoff_vertical_tolerance_m),
        ("horizontal speed tolerance", takeoff_horizontal_speed_tolerance_m_s),
        ("vertical speed tolerance", takeoff_vertical_speed_tolerance_m_s),
    ):
        _validate_takeoff_gate_limit(value, label, allow_zero=label == "stable window")
    if heading_policy not in {"measured-hold", "route-tangent-relative"}:
        raise ValueError("unsupported heading policy")
    for index, setpoint in enumerate(schedule):
        for field_name in ("north_m", "east_m", "down_m", "yaw_deg"):
            _native_number(
                getattr(setpoint, field_name),
                f"setpoint schedule index {index}.{field_name}",
            )
    if isinstance(track_start_index, bool) or not isinstance(track_start_index, int):
        raise ValueError("track start index must be an integer")
    if track_start_index < 0 or track_start_index >= len(schedule):
        raise ValueError("track start index is outside the setpoint schedule")
    if track_end_index is not None:
        if isinstance(track_end_index, bool) or not isinstance(track_end_index, int):
            raise ValueError("track end index must be an integer")
        if track_end_index < 0 or track_end_index >= len(schedule):
            raise ValueError("track end index is outside the setpoint schedule")
        if track_end_index < track_start_index:
            raise ValueError("track end index must not precede the track start index")
    exec_start = time.monotonic()
    timing: dict[str, Any] = {
        "time_base": "executor_relative_seconds",
        "setpoint_count": len(schedule),
        "rate_hz": rate_hz,
        "takeoff_gate": {
            "status": "not_started",
        },
        "preflight_connection": {},
        "cleanup": {
            "stop_offboard": "not_needed",
            "land": "not_requested" if not land_after else "not_needed",
            "close": "pending",
        },
    }
    takeoff_gate = timing["takeoff_gate"]
    cleanup = timing["cleanup"]
    track_end = len(schedule) - 1 if track_end_index is None else track_end_index
    track_start = track_start_index
    arm_requested = False
    arm_acknowledged = False
    offboard_requested = False
    offboard_started = False
    offboard_stopped = False
    last_commanded_setpoint: Setpoint | None = None
    land_command_sent = False
    landing_confirmation_attempted = False
    runtime_failure: str | None = None
    external_abort_world_paused = False
    runtime_observations: dict[str, dict[str, Any]] = {}
    attempted_effect_sections: set[str] = set()
    gps_transitions: list[dict[str, Any]] = []
    gps_reset_verified = False
    gps_value: dict[str, Any] | None = None
    gps_control_details: dict[str, Any] | None = None
    battery_details: dict[str, Any] | None = None
    gps_profile = runtime_profile.get("gps_dropout") if runtime_profile else None
    battery_profile = runtime_profile.get("battery") if runtime_profile else None
    wind_profile = runtime_profile.get("wind_activation") if runtime_profile else None
    gps_schedule: list[bool] = []
    gps_last_tick = -1
    gps_off = False
    battery_takeoff_gate_parameters: dict[str, dict[str, float]] | None = None
    if isinstance(gps_profile, dict):
        runtime_profile_details = _require_runtime_details(
            runtime_profile,
            label="compiled runtime profile",
        )
        track_sample_count = track_end - track_start + 1
        tick_period_s = _native_number(gps_profile["tick_period_s"], "GPS tick period")
        if tick_period_s <= 0:
            raise ValueError("GPS tick period must be positive")
        tick_count = max(
            1,
            int(math.ceil(track_sample_count / rate_hz / tick_period_s)),
        )
        gps_schedule = compile_fixed_duty_schedule(
            requested_rate=float(gps_profile["requested_rate"]),
            tick_count=tick_count,
            execution_identity_sha256=str(runtime_profile_details["execution_identity_sha256"]),
        )

    # 功能：
    #   每次副作用前读取当前任务的停止通道。
    # 输入：
    #   无显式参数；使用外层的 abort_file。
    # 输出：
    #   None：未停止时通过，停止时抛出携带原因的异常。
    def check_external_abort() -> None:
        _raise_if_external_abort_requested(abort_file)

    try:
        if runtime_phase_path is not None:
            _write_runtime_phase(runtime_phase_path, "PREFLIGHT")
        check_external_abort()
        health = await connect_preflight_with_recovery(
            client,
            connection=connection,
            readiness_timeout_seconds=takeoff_timeout_seconds,
            abort_check=check_external_abort,
            log_path=log_path,
            evidence=timing["preflight_connection"],
        )
        takeoff_gate["readiness"] = _health_payload(health)
        takeoff_gate["readiness_policy"] = "local_ned_with_px4_preflight_authority"
        takeoff_gate["required_readiness"] = {
            name: takeoff_gate["readiness"][name]
            for name in ("connected", "home_position_ok", "local_position_ok", "armable")
        }
        takeoff_gate["advisory_readiness"] = {
            name: takeoff_gate["readiness"][name] for name in ("global_position_ok",)
        }
        takeoff_gate["readiness_observed"] = all(takeoff_gate["required_readiness"].values())
        if not takeoff_gate["readiness_observed"]:
            raise RuntimeError(
                "PX4 returned non-armable or incomplete readiness; sensor degradation must never "
                "bypass the preflight safety gate"
            )
        timing["offboard_loss_failsafe"] = await _await_with_abort_polling(
            configure_offboard_loss_failsafe(client),
            abort_check=check_external_abort,
        )
        _log(log_path, "PX4 offboard-loss HOLD failsafe readback verified")
        if isinstance(gps_profile, dict):
            check_external_abort()
            baseline_satellites = await _await_with_abort_polling(
                client.get_param_int("SIM_GPS_USED"),
                abort_check=check_external_abort,
            )
            if baseline_satellites < 4:
                raise RuntimeError(
                    "PX4 SIM_GPS_USED baseline must be at least 4 satellites for "
                    f"deterministic GPS recovery, got {baseline_satellites}"
                )
            gps_control_details = {
                "parameter_name": "SIM_GPS_USED",
                "before": baseline_satellites,
                "dropout_value": 0,
                "recovery_value": baseline_satellites,
            }
            _log(
                log_path,
                f"SIM_GPS_USED baseline recorded as {baseline_satellites}",
            )
        if isinstance(battery_profile, dict):
            check_external_abort()
            battery_takeoff_gate_parameters = await _await_with_abort_polling(
                _hold_battery_during_takeoff_gate(client),
                abort_check=check_external_abort,
            )
        try:
            check_external_abort()
            takeoff_origin = await _await_with_abort_polling(
                client.sample_position_velocity_ned(min(2.0, takeoff_timeout_seconds)),
                abort_check=check_external_abort,
            )
        except ExternalSafetyAbort:
            takeoff_gate["status"] = "failed"
            takeoff_gate["failure_reason"] = "external_safety_abort"
            raise
        except BaseException as exc:
            takeoff_gate["status"] = "failed"
            takeoff_gate["failure_reason"] = "initial_position_velocity_telemetry_unavailable"
            takeoff_gate["telemetry_error"] = f"{type(exc).__name__}: {exc}"
            raise
        takeoff_gate["initial_position_velocity_ned"] = {
            "north_m": takeoff_origin.north_m,
            "east_m": takeoff_origin.east_m,
            "down_m": takeoff_origin.down_m,
            "north_m_s": takeoff_origin.north_m_s,
            "east_m_s": takeoff_origin.east_m_s,
            "down_m_s": takeoff_origin.down_m_s,
        }
        try:
            check_external_abort()
            measured_heading_deg = await _await_with_abort_polling(
                client.sample_heading_deg(min(2.0, takeoff_timeout_seconds)),
                abort_check=check_external_abort,
            )
        except ExternalSafetyAbort:
            takeoff_gate["status"] = "failed"
            takeoff_gate["failure_reason"] = "external_safety_abort"
            raise
        except BaseException as exc:
            takeoff_gate["status"] = "failed"
            takeoff_gate["failure_reason"] = "initial_heading_telemetry_unavailable"
            takeoff_gate["heading_telemetry_error"] = f"{type(exc).__name__}: {exc}"
            raise
        planned_yaw_values = [setpoint.yaw_deg for setpoint in schedule]
        planned_first_setpoint = schedule[0]
        source_setpoint_count = len(schedule)
        measured_body_heading_ned_deg: float | None = None
        if heading_policy == "measured-hold":
            schedule = hold_setpoint_schedule_heading(schedule, measured_heading_deg)
        else:
            if not gazebo_vehicle_model_name:
                raise RuntimeError("route-tangent heading requires a Gazebo vehicle model identity")
            gazebo_pose = await _await_with_abort_polling(
                client.sample_gazebo_model_pose(
                    world_name=world,
                    model_name=gazebo_vehicle_model_name,
                    timeout_seconds=min(5.0, takeoff_timeout_seconds),
                ),
                abort_check=check_external_abort,
            )
            measured_body_heading_ned_deg = gazebo_body_heading_ned_deg(gazebo_pose)
            aligned_plan = align_setpoint_schedule_to_route_tangent(
                SetpointSchedulePlan(
                    schedule=schedule,
                    track_start_index=track_start,
                    track_end_index=track_end,
                ),
                measured_px4_heading_deg=measured_heading_deg,
                measured_body_heading_ned_deg=measured_body_heading_ned_deg,
                rate_hz=rate_hz,
                maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
            )
            schedule = aligned_plan.schedule
            track_start = aligned_plan.track_start_index
            track_end = aligned_plan.track_end_index
        timing["setpoint_count"] = len(schedule)
        relative_first_setpoint = schedule[0]
        takeoff_gate["heading_control"] = {
            "policy": (
                "measured_prearm_heading_hold"
                if heading_policy == "measured-hold"
                else "measured_origin_route_tangent_rate_limited"
            ),
            "measured_heading_deg": measured_heading_deg,
            "measured_gazebo_body_heading_ned_deg": measured_body_heading_ned_deg,
            "px4_minus_gazebo_heading_offset_deg": (
                None
                if measured_body_heading_ned_deg is None
                else _shortest_yaw_delta_deg(
                    measured_body_heading_ned_deg,
                    measured_heading_deg,
                )
            ),
            "planned_yaw_min_deg": min(planned_yaw_values),
            "planned_yaw_max_deg": max(planned_yaw_values),
            "planned_first_yaw_deg": planned_first_setpoint.yaw_deg,
            "commanded_yaw_deg": measured_heading_deg,
            "maximum_yaw_rate_deg_s": (
                None if heading_policy == "measured-hold" else maximum_yaw_rate_deg_s
            ),
            "source_setpoint_count": source_setpoint_count,
            "commanded_setpoint_count": len(schedule),
            "deliberate_yaw_requires_separate_qualified_action": (
                heading_policy == "measured-hold"
            ),
        }
        _log(
            log_path,
            "measured pre-arm heading "
            f"{measured_heading_deg:.3f} deg; heading policy={heading_policy}",
        )
        schedule = rebase_setpoint_schedule(schedule, takeoff_origin)
        takeoff_gate["schedule_origin_rebase"] = {
            "contract": "spawn_relative_schedule_plus_measured_px4_local_ned_origin",
            "origin_ned": {
                "north_m": takeoff_origin.north_m,
                "east_m": takeoff_origin.east_m,
                "down_m": takeoff_origin.down_m,
            },
            "relative_first_setpoint_ned": {
                "north_m": relative_first_setpoint.north_m,
                "east_m": relative_first_setpoint.east_m,
                "down_m": relative_first_setpoint.down_m,
                "yaw_deg": relative_first_setpoint.yaw_deg,
            },
            "rebased_first_setpoint_ned": {
                "north_m": schedule[0].north_m,
                "east_m": schedule[0].east_m,
                "down_m": schedule[0].down_m,
                "yaw_deg": schedule[0].yaw_deg,
            },
        }
        initial_hold = Setpoint(
            north_m=takeoff_origin.north_m,
            east_m=takeoff_origin.east_m,
            down_m=takeoff_origin.down_m,
            yaw_deg=schedule[0].yaw_deg,
        )
        check_external_abort()
        await _await_with_abort_polling(
            client.set_position_ned(initial_hold),
            abort_check=check_external_abort,
        )
        last_commanded_setpoint = initial_hold
        takeoff_gate["initial_setpoint_ned"] = {
            "north_m": initial_hold.north_m,
            "east_m": initial_hold.east_m,
            "down_m": initial_hold.down_m,
            "yaw_deg": initial_hold.yaw_deg,
        }
        check_external_abort()
        if runtime_phase_path is not None:
            _write_runtime_phase(runtime_phase_path, "TAKEOFF")
        arm_requested = True  # A missing ACK cannot prove that PX4 did not arm.
        await _await_with_abort_polling(
            client.arm(),
            abort_check=check_external_abort,
        )
        arm_acknowledged = True
        takeoff_gate["px4_arm_command"] = "accepted"
        _log(log_path, "armed")

        timing["takeoff_start_t"] = time.monotonic() - exec_start
        check_external_abort()
        offboard_requested = True
        await _await_with_abort_polling(
            client.start_offboard(),
            abort_check=check_external_abort,
        )
        offboard_started = True
        timing["offboard_start_t"] = time.monotonic() - exec_start
        _log(log_path, "offboard started")

        await _wait_for_takeoff_stability(
            client,
            schedule[0],
            takeoff_origin=takeoff_origin,
            climb_rate_m_s=takeoff_climb_rate_m_s,
            timeout_seconds=takeoff_timeout_seconds,
            sample_rate_hz=rate_hz,
            stable_window_seconds=takeoff_stable_window_seconds,
            horizontal_tolerance_m=takeoff_horizontal_tolerance_m,
            vertical_tolerance_m=takeoff_vertical_tolerance_m,
            horizontal_speed_tolerance_m_s=takeoff_horizontal_speed_tolerance_m_s,
            vertical_speed_tolerance_m_s=takeoff_vertical_speed_tolerance_m_s,
            evidence=takeoff_gate,
            abort_check=check_external_abort,
        )
        last_commanded_setpoint = schedule[0]
        timing["takeoff_stable_t"] = time.monotonic() - exec_start
        _log(log_path, "takeoff telemetry gate achieved stable hover")

        if isinstance(wind_profile, dict):
            activation_t_s = time.monotonic() - exec_start
            attempted_effect_sections.add("wind_activation")
            wind_observation = await _await_with_setpoint_keepalive(
                client,
                _await_wind_activation(wind_activator, world, wind_profile, activation_t_s),
                hold_setpoint=schedule[0],
                rate_hz=rate_hz,
                abort_check=check_external_abort,
            )
            if not isinstance(wind_observation, dict):
                raise RuntimeError("post-hover wind activator returned invalid evidence")
            runtime_observations["wind_activation"] = wind_observation
            timing["wind_activation"] = {
                "status": "verified",
                "phase": "after_stable_hover_before_track_entry",
                # Preserve the monotonic timestamp for ordering checks. The
                # request-bound evidence may round its display value, but the
                # timing trace must never appear to place activation before
                # the stable-hover sample because of decimal rounding.
                "activation_t_s": activation_t_s,
            }
            _log(log_path, "post-hover Gazebo wind activation readback verified")

        if isinstance(battery_profile, dict):
            attempted_effect_sections.add("battery")
            battery_details = await _await_with_setpoint_keepalive(
                client,
                _prepare_battery_profile(
                    client,
                    battery_profile,
                    takeoff_hold_seconds=max(1.0 / rate_hz, track_start / rate_hz),
                ),
                hold_setpoint=schedule[0],
                rate_hz=rate_hz,
                abort_check=check_external_abort,
            )
            battery_details["takeoff_gate_parameters"] = _require_runtime_details(
                battery_takeoff_gate_parameters,
                label="battery takeoff-gate control details",
            )

        dt = 1.0 / rate_hz
        event_loop = asyncio.get_running_loop()
        start = event_loop.time()
        track_deadline = start + track_timeout_seconds
        for idx, setpoint in enumerate(schedule):
            check_external_abort()
            if idx == track_start and isinstance(battery_profile, dict):
                conditioning_started = event_loop.time()
                battery_details = await _await_with_abort_polling(
                    _transition_battery_at_track_start(
                        client,
                        battery_profile,
                        _require_runtime_details(
                            battery_details,
                            label="battery control details",
                        ),
                        hold_setpoint=schedule[max(0, track_start - 1)],
                        rate_hz=rate_hz,
                    ),
                    abort_check=check_external_abort,
                )
                # Battery conditioning is a bounded pre-track safety gate, not
                # part of the trajectory execution timeout budget.
                conditioning_elapsed = event_loop.time() - conditioning_started
                start += conditioning_elapsed
                track_deadline += conditioning_elapsed
            if idx == track_start and runtime_phase_path is not None:
                _write_runtime_phase(runtime_phase_path, "TRACK")
            if event_loop.time() >= track_deadline:
                raise TimeoutError(f"track timeout after {track_timeout_seconds:g}s")
            timeout_scope = asyncio.timeout_at(track_deadline)
            try:
                async with timeout_scope:
                    if idx >= track_start and isinstance(gps_profile, dict):
                        elapsed_track_seconds = (idx - track_start) / rate_hz
                        tick_index = min(
                            len(gps_schedule) - 1,
                            int(elapsed_track_seconds / float(gps_profile["tick_period_s"])),
                        )
                        if tick_index != gps_last_tick:
                            gps_last_tick = tick_index
                            desired_off = gps_schedule[tick_index]
                            if desired_off != gps_off:
                                attempted_effect_sections.add("gps_dropout")
                                gps_control = _require_runtime_details(
                                    gps_control_details,
                                    label="GPS control details",
                                )
                                failure_type = "off" if desired_off else "ok"
                                target_satellites = (
                                    int(gps_control["dropout_value"])
                                    if desired_off
                                    else int(gps_control["recovery_value"])
                                )
                                verification = await _await_with_setpoint_keepalive(
                                    client,
                                    _set_gps_availability_verified(
                                        client,
                                        satellites_used=target_satellites,
                                        unavailable=desired_off,
                                    ),
                                    hold_setpoint=last_commanded_setpoint or schedule[0],
                                    rate_hz=rate_hz,
                                    abort_check=check_external_abort,
                                )
                                gps_off = desired_off
                                gps_transitions.append(
                                    {
                                        "tick_index": tick_index,
                                        "track_time_s": elapsed_track_seconds,
                                        "failure_type": failure_type,
                                        "physical_effect_verified": True,
                                        "verification": verification,
                                    }
                                )
                    await _await_with_abort_polling(
                        client.set_position_ned(setpoint),
                        abort_check=check_external_abort,
                    )
                    last_commanded_setpoint = setpoint
                    now_t = time.monotonic() - exec_start
                    if idx == track_start:
                        timing["track_start_t"] = now_t
                    if idx == track_end:
                        timing["track_end_t"] = now_t
                        if isinstance(battery_profile, dict):
                            current_battery_details = _require_runtime_details(
                                battery_details,
                                label="battery control details",
                            )
                            track_end_sample = _validated_battery_telemetry(
                                await _await_with_setpoint_keepalive(
                                    client,
                                    client.sample_battery(5.0),
                                    hold_setpoint=setpoint,
                                    rate_hz=rate_hz,
                                    abort_check=check_external_abort,
                                ),
                                label="PX4 track-end battery telemetry",
                            )
                            start_percent = _finite_float(
                                current_battery_details["track_start_sample"]["remaining_percent"],
                                "PX4 track-start battery remaining_percent",
                            )
                            end_percent = track_end_sample["remaining_percent"]
                            if (
                                bool(battery_profile["voltage_sag"])
                                and end_percent > start_percent + 0.5
                            ):
                                raise RuntimeError(
                                    "PX4 battery telemetry increased during requested voltage "
                                    f"sag: start={start_percent:g}%, end={end_percent:g}%"
                                )
                            current_battery_details["track_end_sample"] = track_end_sample
                            current_battery_details["observed_nonincrease"] = (
                                end_percent <= start_percent + 0.5
                            )
                    # 遥测读回或调度停顿后重新锚定发送节拍，不能瞬间补发旧采样来追赶墙钟。
                    # 总轨迹超时不因此延期；超过预算仍进入失败清理。
                    next_tick_at = max(start + (idx + 1) * dt, event_loop.time() + dt)
                    start = next_tick_at - (idx + 1) * dt
                    await _await_with_abort_polling(
                        asyncio.sleep(max(0.0, next_tick_at - event_loop.time())),
                        abort_check=check_external_abort,
                    )
            except TimeoutError:
                if timeout_scope.expired():
                    raise TimeoutError(f"track timeout after {track_timeout_seconds:g}s") from None
                raise

        if isinstance(gps_profile, dict):
            gps_control = _require_runtime_details(
                gps_control_details,
                label="GPS control details",
            )
            verification = await _await_with_setpoint_keepalive(
                client,
                _set_gps_availability_verified(
                    client,
                    satellites_used=int(gps_control["recovery_value"]),
                    unavailable=False,
                ),
                hold_setpoint=last_commanded_setpoint or schedule[0],
                rate_hz=rate_hz,
                abort_check=check_external_abort,
            )
            gps_control["restore"] = verification
            gps_control["restore_verified"] = True
            gps_off = False
            gps_reset_verified = True
            gps_transitions.append(
                {
                    "tick_index": len(gps_schedule),
                    "track_time_s": max(0.0, (track_end - track_start + 1) / rate_hz),
                    "failure_type": "ok",
                    "physical_effect_verified": True,
                    "verification": verification,
                    "final_reset": True,
                }
            )
            gps_value = {
                "schedule_algorithm": gps_profile["schedule_algorithm"],
                "tick_period_s": gps_profile["tick_period_s"],
                "schedule": gps_schedule,
                "tick_count": len(gps_schedule),
                "off_tick_count": sum(gps_schedule),
                "realized_rate": sum(gps_schedule) / len(gps_schedule),
                "transitions": gps_transitions,
                "reset_verified": gps_reset_verified,
                "control_parameter": gps_control,
            }
        if isinstance(battery_profile, dict):
            current_battery_details = _require_runtime_details(
                battery_details,
                label="battery control details",
            )
            runtime_observations["battery"] = {
                "source": "mavsdk.param+telemetry/battery",
                "kind": "readback",
                "value": current_battery_details,
                "sha256": _canonical_sha256(current_battery_details),
            }
        await client.stop_offboard()
        offboard_stopped = True
        cleanup["stop_offboard"] = "completed"
        _log(log_path, "offboard stopped")
        if land_after:
            check_external_abort()
            if runtime_phase_path is not None:
                _write_runtime_phase(runtime_phase_path, "LANDING")
            timing["land_start_t"] = time.monotonic() - exec_start
            await client.land()
            land_command_sent = True
            cleanup["land"] = "command_sent"
            _log(log_path, "land command sent")
            landing_confirmation_attempted = True
            try:
                landing_observation = await _await_with_abort_polling(
                    client.wait_until_landed(landing_timeout_seconds),
                    abort_check=check_external_abort,
                )
                landing_observation = _validated_landing_observation(landing_observation)
            except Exception as exc:
                cleanup["land"] = f"failed: {type(exc).__name__}: {exc}"
                raise
            cleanup["land"] = "confirmed_on_ground"
            cleanup["landing_observation"] = landing_observation
            timing["land_confirmed_t"] = time.monotonic() - exec_start
            _log(log_path, "landing confirmed ON_GROUND by PX4 telemetry")
    except BaseException as exc:
        if isinstance(exc, ExternalSafetyAbort):
            external_abort_world_paused = exc.world_paused
            timing["external_abort"] = {
                "reason": exc.reason,
                "world_paused": exc.world_paused,
            }
        runtime_failure = f"{type(exc).__name__}: {exc}"
        if takeoff_gate.get("status") not in {"achieved", "failed"}:
            takeoff_gate["status"] = "failed"
            takeoff_gate["failure_reason"] = "readiness_or_preflight_failure"
            takeoff_gate["preflight_error"] = runtime_failure
        timing["status"] = "failed"
        timing["failure"] = runtime_failure
        raise
    finally:
        timing["command_attempts"] = {
            "arm_requested": arm_requested,
            "arm_acknowledged": arm_acknowledged,
            "offboard_requested": offboard_requested,
            "offboard_acknowledged": offboard_started,
        }
        reset_error: BaseException | None = None
        deferred_cleanup_error: RuntimeError | None = None

        # 功能：
        #   记录退出诊断，日志失败不阻断停控、落地观察及客户端释放，也不覆盖原异常。
        # 输入：
        #   message：本次清理阶段的诊断文字。
        # 输出：
        #   None：不返回业务数据。
        def cleanup_log(message: str) -> None:
            nonlocal runtime_failure, deferred_cleanup_error
            try:
                _log(log_path, message)
            except (Exception, asyncio.CancelledError) as error:
                failure = f"{type(error).__name__}: {error}"
                cleanup.setdefault("logging_errors", []).append(failure)
                if runtime_failure is None:
                    runtime_failure = failure
                    timing["status"], timing["failure"] = "failed", failure
                    deferred_cleanup_error = RuntimeError("cleanup logging failed: " + failure)

        if (
            isinstance(gps_profile, dict)
            and gps_control_details is not None
            and not gps_reset_verified
        ):
            try:
                restore_operation = _set_gps_availability_verified(
                    client,
                    satellites_used=int(gps_control_details["recovery_value"]),
                    unavailable=False,
                )
                if offboard_started and not offboard_stopped:
                    restore = await _await_with_setpoint_keepalive(
                        client,
                        restore_operation,
                        hold_setpoint=last_commanded_setpoint or schedule[0],
                        rate_hz=rate_hz,
                    )
                else:
                    restore = await restore_operation
                gps_control_details["restore"] = restore
                gps_control_details["restore_verified"] = True
                gps_off = False
                gps_reset_verified = True
                cleanup_log("SIM_GPS_USED restored during GPS cleanup")
            except (Exception, asyncio.CancelledError) as exc:
                reset_error = exc
                gps_control_details["restore_verified"] = False
                gps_control_details["restore_error"] = f"{type(exc).__name__}: {exc}"
                cleanup_log(f"GPS availability cleanup reset failed: {exc}")
                if runtime_failure is None:
                    runtime_failure = f"{type(exc).__name__}: {exc}"
                    timing["status"] = "failed"
                    timing["failure"] = runtime_failure
        if (
            gps_value is not None
            and gps_control_details is not None
            and gps_control_details.get("restore_verified") is True
        ):
            runtime_observations["gps_dropout"] = {
                "source": "mavsdk.param+telemetry/gps_info",
                "kind": "readback",
                "value": gps_value,
                "sha256": _canonical_sha256(gps_value),
            }
        if offboard_requested and not offboard_stopped:
            try:
                await asyncio.wait_for(
                    client.stop_offboard(), timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS
                )
                offboard_stopped = True
                cleanup["stop_offboard"] = "completed_during_failure_cleanup"
                cleanup_log("offboard stopped during failure cleanup")
            except (Exception, asyncio.CancelledError) as exc:
                cleanup["stop_offboard"] = f"failed: {type(exc).__name__}: {exc}"
                cleanup_log(f"offboard failure cleanup could not stop offboard: {exc}")
        if (
            arm_requested
            and land_after
            and not landing_confirmation_attempted
            and not external_abort_world_paused
        ):
            cleanup.pop("landing_observation", None)
            try:
                timing.setdefault("land_start_t", time.monotonic() - exec_start)
                if runtime_phase_path is not None:
                    try:
                        _write_runtime_phase(runtime_phase_path, "LANDING")
                    except (Exception, asyncio.CancelledError) as exc:
                        cleanup["runtime_phase"] = (
                            f"landing_publish_failed: {type(exc).__name__}: {exc}"
                        )
                        cleanup_log(f"runtime landing phase publication failed: {exc}")
                if not land_command_sent:
                    try:
                        await asyncio.wait_for(
                            client.land(), timeout=CLEANUP_COMMAND_TIMEOUT_SECONDS
                        )
                        land_command_sent = True
                        cleanup["land_command"] = "acknowledged_during_failure_cleanup"
                        cleanup_log("land command sent during failure cleanup")
                    except (Exception, asyncio.CancelledError) as exc:
                        cleanup["land_command"] = f"failed: {type(exc).__name__}: {exc}"
                        cleanup_log(f"land acknowledgement lost during failure cleanup: {exc}")
                # 命令回执与实际落地是不同证据，丢失回执不能跳过原生状态观察。
                landing_confirmation_attempted = True
                landing_observation = await asyncio.wait_for(
                    client.wait_until_landed(landing_timeout_seconds),
                    timeout=landing_timeout_seconds,
                )
                landing_observation = _validated_landing_observation(landing_observation)
                cleanup["land"] = "confirmed_on_ground_during_failure_cleanup"
                cleanup["landing_observation"] = landing_observation
                timing["land_confirmed_t"] = time.monotonic() - exec_start
                cleanup_log("landing confirmed ON_GROUND during failure cleanup")
            except (Exception, asyncio.CancelledError) as exc:
                cleanup["land"] = f"failed: {type(exc).__name__}: {exc}"
                cleanup_log(f"offboard failure cleanup could not land: {exc}")
        elif arm_requested and land_after and external_abort_world_paused:
            cleanup["land"] = "suppressed_world_paused_external_safety_abort"
            cleanup_log("landing suppressed because the external safety monitor paused Gazebo")
        if (
            runtime_profile is not None
            and scenario_engine is not None
            and scenario_request is not None
            and runtime_evidence_path is not None
        ):
            try:
                _write_runtime_effect_artifact(
                    scenario_engine,
                    scenario_request,
                    runtime_profile,
                    runtime_evidence_path,
                    observations=runtime_observations,
                    attempted_sections=attempted_effect_sections,
                    status="complete" if runtime_failure is None else "failed",
                    error=runtime_failure,
                )
            except (Exception, asyncio.CancelledError) as exc:
                evidence_error = f"{type(exc).__name__}: {exc}"
                timing["runtime_evidence_write_error"] = evidence_error
                cleanup_log(f"runtime effect evidence write failed: {exc}")
                if runtime_failure is None:
                    runtime_failure = evidence_error
                    timing["status"] = "failed"
                    timing["failure"] = evidence_error
                    deferred_cleanup_error = RuntimeError(
                        f"runtime effect evidence write failed: {exc}"
                    )
        if runtime_failure is None:
            timing["status"] = "complete"
        try:
            await client.close()
            cleanup["close"] = "completed"
            cleanup_log("offboard client closed")
        except (Exception, asyncio.CancelledError) as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            cleanup["close"] = f"failed: {cleanup_error}"
            cleanup_log(f"offboard client cleanup failed: {exc}")
            if runtime_failure is None:
                runtime_failure = cleanup_error
                timing["status"] = "failed"
                timing["failure"] = cleanup_error
                deferred_cleanup_error = RuntimeError(f"offboard client cleanup failed: {exc}")
        if timing_path is not None:
            try:
                _write_offboard_timing(timing_path, timing)
            except (Exception, asyncio.CancelledError) as exc:
                cleanup_log(f"offboard timing evidence write failed: {exc}")
                if runtime_failure is None:
                    deferred_cleanup_error = RuntimeError(
                        f"offboard timing evidence write failed: {exc}"
                    )
        if runtime_phase_path is not None:
            execution_succeeded = runtime_failure is None and deferred_cleanup_error is None
            terminal_phase = "COMPLETE" if execution_succeeded else "FAILED"
            observation = cleanup.get("landing_observation")
            if (
                not execution_succeeded
                and isinstance(observation, dict)
                and observation.get("state") == "ON_GROUND"
                and observation.get("confirmed") is True
            ):
                terminal_phase = "LANDED"
            try:
                _write_runtime_phase(runtime_phase_path, terminal_phase)
            except (Exception, asyncio.CancelledError) as exc:
                cleanup_log(f"terminal runtime phase publication failed: {exc}")
                if runtime_failure is None:
                    deferred_cleanup_error = RuntimeError(
                        f"terminal runtime phase publication failed: {exc}"
                    )
        reset_failure_text = (
            f"{type(reset_error).__name__}: {reset_error}" if reset_error is not None else None
        )
        if reset_error is not None and runtime_failure == reset_failure_text:
            raise RuntimeError(
                f"GPS availability cleanup reset failed: {reset_error}"
            ) from reset_error
        if deferred_cleanup_error is not None:
            raise deferred_cleanup_error


# 功能：
#   加载鉴定输入并运行真实 MAVSDK 执行器；显式 dry-run 只检查调度，不能宣称飞行已完成。
# 输入：
#   argv：命令行参数，None 表示当前进程参数。
# 输出：
#   exit_code：成功或显式干运行返回零，缺少运行依赖返回二，其他异常返回一。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dry_run = _parse_bool(os.environ.get("PX4_OFFBOARD_DRY_RUN"), default=False)
    land_after = _parse_bool(os.environ.get("PX4_OFFBOARD_LAND_AFTER"), default=True)

    try:
        scenario_engine, scenario_request, runtime_profile = _load_runtime_effect_request()
        runtime_evidence_path = (
            args.run_dir / scenario_engine.RUNTIME_EVIDENCE_ARTIFACT_NAME
            if runtime_profile is not None
            else None
        )
        reference_plan = load_reference_track_plan(args.track)
        points = reference_plan.points
        params = load_controller_params(args.params)
        plan = build_setpoint_schedule_plan(
            points,
            params,
            args.setpoint_rate_hz,
            hover_duration_seconds=reference_plan.hover_duration_seconds,
            stop_at_waypoints=reference_plan.stop_at_waypoints,
            waypoint_hold_seconds=reference_plan.waypoint_hold_seconds,
        )
        _log(
            args.log,
            f"vehicle={args.vehicle} world={args.world} points={len(points)} "
            f"setpoints={len(plan.schedule)}",
        )
        _log(
            args.log,
            "offboard schedule applies vel_limit and accel_limit only; selected PX4 parameters "
            "are applied and verified by the launch wrapper",
        )

        if dry_run:
            _log(
                args.log,
                "PX4_OFFBOARD_DRY_RUN=true; executor exiting without MAVSDK command streaming",
            )
            dry_timing = {
                "execution_mode": "dry-run",
                "status": "not_observed",
                "time_base": "executor_relative_seconds",
                "setpoint_count": len(plan.schedule),
                "rate_hz": args.setpoint_rate_hz,
                "takeoff_start_t": 0.0,
                "offboard_start_t": 0.0,
                "track_start_t": plan.track_start_index / max(1e-6, args.setpoint_rate_hz),
                "track_end_t": plan.track_end_index / max(1e-6, args.setpoint_rate_hz),
                "takeoff_gate": {
                    "status": "dry_run_not_observed",
                    "reason": "dry-run does not emit PX4 position/velocity telemetry",
                },
            }
            _write_offboard_timing(args.run_dir / "offboard_timing.json", dry_timing)
            return 0

        wind_profile = runtime_profile.get("wind_activation") if runtime_profile else None
        if isinstance(wind_profile, dict):
            _validated_gazebo_wind_activation(args.world, wind_profile)
            _log(
                args.log,
                "Gazebo CLI wind publisher validated; activation remains gated until stable hover",
            )

        client = MavsdkOffboardClient()
        asyncio.run(
            run_executor(
                client,
                plan.schedule,
                connection=args.connection,
                takeoff_timeout_seconds=args.takeoff_timeout_seconds,
                takeoff_climb_rate_m_s=args.takeoff_climb_rate_m_s,
                track_timeout_seconds=args.track_timeout_seconds,
                landing_timeout_seconds=args.landing_timeout_seconds,
                rate_hz=args.setpoint_rate_hz,
                land_after=land_after,
                log_path=args.log,
                track_start_index=plan.track_start_index,
                track_end_index=plan.track_end_index,
                timing_path=args.run_dir / "offboard_timing.json",
                runtime_phase_path=args.run_dir / "runtime-phase.json",
                scenario_engine=scenario_engine,
                scenario_request=scenario_request,
                runtime_profile=runtime_profile,
                runtime_evidence_path=runtime_evidence_path,
                takeoff_stable_window_seconds=args.takeoff_stable_window_seconds,
                takeoff_horizontal_tolerance_m=args.takeoff_horizontal_tolerance_m,
                takeoff_vertical_tolerance_m=args.takeoff_vertical_tolerance_m,
                takeoff_horizontal_speed_tolerance_m_s=(
                    args.takeoff_horizontal_speed_tolerance_m_s
                ),
                takeoff_vertical_speed_tolerance_m_s=(args.takeoff_vertical_speed_tolerance_m_s),
                world=args.world,
                abort_file=args.abort_file,
                wind_activator=_activate_gazebo_wind_profile,
                heading_policy=args.heading_policy,
                maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
                gazebo_vehicle_model_name=args.gazebo_vehicle_model_name,
            )
        )
        _log(args.log, "executor completed successfully")
        return 0
    except RuntimeError as exc:
        _log(args.log, str(exc))
        if "mavsdk is required for PX4 offboard execution" in str(exc):
            print("mavsdk is required for PX4 offboard execution", file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        _log(args.log, f"executor failure: {exc}")
        print(f"px4 offboard executor failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
