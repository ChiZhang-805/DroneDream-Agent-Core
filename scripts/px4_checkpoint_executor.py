#!/usr/bin/env python3
"""Real MAVSDK Offboard executor with model-reviewed segment hover checkpoints."""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from dronedream_agent_core.collision import validate_route_clearance
from dronedream_agent_core.contracts import (
    GraphRoute,
    Px4CoordinateContract,
    Px4Track,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionReceipt,
    RuntimeActionExecutionStep,
    RuntimeAuthorizedCommand,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    RuntimeCommandAdoption,
    RuntimeControlSession,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    RuntimeOperatorControlCommand,
    RuntimeOperatorTakeoverAdoption,
    RuntimeOperatorTakeoverGrant,
    RuntimeReplacementTrack,
    RuntimeTrackProgress,
    RuntimeUserMessage,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.control_authority import (
    control_application_category,
    integrate_model_yaw,
    measured_hold_heading,
)
from dronedream_agent_core.control_cadence import ControlTickPacer
from dronedream_agent_core.control_execution_evidence import control_application_record
from dronedream_agent_core.control_timing import (
    LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
    LOCAL_TRANSPORT_BUDGET_MS,
)
from dronedream_agent_core.control_uncertainty import finite_positive_number
from dronedream_agent_core.executor_finalization import finalize_executor_resources
from dronedream_agent_core.executor_snapshots import ACTIVE_SNAPSHOTS, ExecutorSnapshots
from dronedream_agent_core.flight_command_cleanup import (
    FlightCommandAttempts,
    cleanup_flight_commands,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_lifecycle import capture_control_completion
from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.process_capture import capture_process
from dronedream_agent_core.runtime_control_io import (
    MAX_RUNTIME_REPLACEMENT_BYTES,
    publish_runtime_json,
    read_runtime_object,
    transfer_runtime_file,
)
from dronedream_agent_core.runtime_evidence import BoundedRuntimeEvidenceWriter
from dronedream_agent_core.runtime_multimodal_dataset import RuntimeMultimodalDatasetRecord
from dronedream_agent_core.runtime_progress import next_track_point_index
from dronedream_agent_core.runtime_scheduling import (
    InterpreterPauseMonitor,
    configure_sensor_thread_handoff,
    retained_interpreter_baseline,
)
from dronedream_agent_core.simulation_sensor_runtime import verify_magnetic_parameters
from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json


# 功能：
#   加载本次明确指定的飞控基础模块；初始化失败时恢复模块注册表，避免复用半初始化对象。
# 输入：
#   path：基础执行器的本地 Python 路径。
# 输出：
#   module：完成初始化的飞控基础模块。
def _load_base(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("dronedream_proven_px4_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load proven PX4 executor dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(spec.name) is module:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
        raise
    return module


# 功能：
#   1. 复用共享有限 JSON 和独占暂存发布器，不覆盖其他写入者的暂存文件。
#   2. Windows 读锁仅在明确秒数预算内重试，失败时保留上一份完整记录。
# 输入：
#   path：本次运行的明确证据输出路径。
#   payload：类型化模型或可验证的有限 JSON 数据。
#   replace_timeout_seconds：允许读锁重试的总秒数。
#   replace_retry_seconds：每次重试的期望间隔秒数。
# 输出：
#   None：不返回业务数据。
def _atomic_json(
    path: Path,
    payload: object,
    *,
    replace_timeout_seconds: float = 0.75,
    replace_retry_seconds: float = 0.01,
) -> None:
    snapshots = ACTIVE_SNAPSHOTS.get()
    if snapshots is not None and snapshots.submit(path, payload):
        return
    publish_runtime_json(
        path,
        payload,
        maximum_bytes=MAX_RUNTIME_REPLACEMENT_BYTES,
        replace_timeout_seconds=replace_timeout_seconds,
        replace_retry_seconds=replace_retry_seconds,
    )


_BOUNDED_EXECUTOR_PHASES = {"WAYPOINT_SETTLE", "CHECKPOINT", "ACTION"}
_LOCAL_CONTROL_PHASES = {
    "PERCEPTION_REFRESH_HOLD",
    "PERCEPTION_STARTUP_HOLD",
    "MODEL_AUTHORITY_HOLD",
    "LOCAL_CLEARANCE_RECOVERY",
    "HOLDING",
    "LOCAL_SLOW",
    "LOCAL_REPLAN",
}
_ENCLOSING_EXECUTOR_STATE_KEY = "enclosing_executor_state"
_MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS = 0.05


# 功能：
#   当前执行器优先读自身已提交阶段，避免诊断落盘延迟丢失局部保护的所属阶段。
# 输入：
#   path：本次阶段路径。
# 输出：
#   phase：独立阶段副本；未启用实时快照时使用原有严格文件读取。
def _read_executor_phase(path: Path) -> dict:
    snapshots = ACTIVE_SNAPSHOTS.get()
    phase = (snapshots.phase() if snapshots is not None and path == snapshots.phase_path
             else read_runtime_object(path))
    return phase


# 功能：
#   发布局部保护阶段并保存其所属的检查点、动作或稳定阶段，供有界等待和恢复共同使用。
# 输入：
#   path：执行阶段证据路径。
#   local_phase：当前局部控制阶段。
#   details：本次保护原因、控制序号等诊断字段。
# 输出：
#   None：不返回业务数据。
def _publish_local_control_phase(
    path: Path,
    *,
    local_phase: str,
    details: dict[str, object],
) -> None:
    """Publish a local-control hold without hiding its enclosing executor phase.

    Checkpoint settling and domain actions deliberately keep refreshing the
    sensor/model-authorized setpoint. A short perception or model-authority
    hold inside that refresh is a subphase of the bounded checkpoint/action,
    not a replacement for it. Keeping the enclosing phase visible lets the
    independent wall-time watchdog grant the already-bounded operation its
    finite extension instead of mistaking a healthy long mission for a hang.
    """

    published: dict[str, object] = {}
    try:
        candidate = _read_executor_phase(path)
        if isinstance(candidate, dict):
            published = candidate
    except (OSError, ValueError):
        pass
    saved_enclosing = published.get(_ENCLOSING_EXECUTOR_STATE_KEY)
    # Repeated local-control updates must retain the executor state that was
    # active before the first hold.  Otherwise a transient lease gap turns
    # TRACK into a permanently stale MODEL_AUTHORITY_HOLD observation even
    # after a new hash-bound model path owns motion again.
    enclosing = dict(saved_enclosing) if isinstance(saved_enclosing, dict) else dict(published)
    enclosing_phase = str(enclosing.get("phase", ""))
    if enclosing_phase in _BOUNDED_EXECUTOR_PHASES:
        payload = {
            **enclosing,
            **details,
            "phase": enclosing_phase,
            "local_control_phase": local_phase,
            _ENCLOSING_EXECUTOR_STATE_KEY: enclosing,
        }
    else:
        payload = {
            **details,
            "phase": local_phase,
            "local_control_phase": local_phase,
            _ENCLOSING_EXECUTOR_STATE_KEY: enclosing,
        }
    _atomic_json(path, payload)


# 功能：
#   在局部控制恢复后还原先前明确记录的执行阶段；没有有效上层阶段时保留保护状态。
# 输入：
#   path：含局部与上层阶段的运行证据路径。
# 输出：
#   None：不返回业务数据。
def _clear_local_control_phase(path: Path) -> None:
    """Restore the exact executor phase hidden by a transient local hold.

    The flight command and ``closed-loop-tracking.json`` have always recovered
    when a fresh model lease arrives.  The shared phase file also has to recover
    atomically so watchdogs, runtime-message injection, UI status, and model
    context all observe the same authoritative state instead of the last hold.
    """

    try:
        published = _read_executor_phase(path)
    except (OSError, ValueError):
        return
    if not isinstance(published, dict):
        return
    saved_enclosing = published.get(_ENCLOSING_EXECUTOR_STATE_KEY)
    if not isinstance(saved_enclosing, dict):
        return
    if not str(saved_enclosing.get("phase", "")):
        # Unit integrations and early startup may publish a hold before the
        # enclosing executor has announced any phase. There is no authoritative
        # state to restore in that case, so retaining the hold is fail-closed.
        return
    local_phase = str(published.get("local_control_phase", published.get("phase", "")))
    if local_phase not in _LOCAL_CONTROL_PHASES:
        return
    _atomic_json(path, dict(saved_enclosing))


class UserDirectedLanding(RuntimeError):
    """A user interruption intentionally superseded the prepared mission."""


class ControlInputLeaseUnavailable(UserDirectedLanding):
    """Pre-dispatch rejection only: no motion command was sent to the adapter."""


class SpawnRelativeOffboardClient:
    """Expose spawn-relative NED while the underlying PX4 client remains local-NED."""

    # 功能：
    #   建立出生点相对 NED 适配器及被动航向误差记录，不改变底层飞控的坐标定义。
    # 输入：
    #   self：本适配器实例。
    #   client：实际飞控客户端。
    #   origin：解锁前实测的本地 NED 原点。
    #   heading_hold_deg：普通路线入口保持的航向，None 表示保留输入航向。
    #   heading_policy：原始路线和替换路线共享的航向策略。
    #   maximum_yaw_rate_deg_s：路线航向变化的最大每秒角度。
    #   world_name：仿真世界身份。
    #   gazebo_vehicle_model_name：显式路线相对航向模式的仿真模型身份。
    #   heading_evidence_path：可选航向诊断输出路径。
    #   heading_evidence_flush_interval_seconds：诊断发布的最短间隔秒数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        client: Any,
        origin: Any,
        *,
        heading_hold_deg: float | None = None,
        heading_policy: str = "measured-hold",
        maximum_yaw_rate_deg_s: float = 20.0,
        world_name: str | None = None,
        gazebo_vehicle_model_name: str | None = None,
        heading_evidence_path: Path | None = None,
        heading_evidence_flush_interval_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._origin = origin
        self._heading_hold_deg = heading_hold_deg
        self._heading_policy = heading_policy
        self._maximum_yaw_rate_deg_s = maximum_yaw_rate_deg_s
        self._world_name = world_name
        self._gazebo_vehicle_model_name = gazebo_vehicle_model_name
        flush_interval = float(heading_evidence_flush_interval_seconds)
        if not math.isfinite(flush_interval) or flush_interval < 0.0:
            raise ValueError("heading evidence flush interval must be finite and non-negative")
        self._heading_evidence_path = heading_evidence_path
        self._heading_evidence_flush_interval_seconds = flush_interval
        self._last_heading_evidence_flush_monotonic = float("-inf")
        self._heading_evidence_writer_issue: str | None = None
        self._pending_heading_publication: asyncio.Task | None = None
        self._heading_tracking_errors_deg: list[float] = []
        self._heading_tracking_commanded_deg: list[float] = []
        self._heading_tracking_observed_deg: list[float] = []
        self._heading_tracking_sample_ages_seconds: list[float] = []
        self._last_heading_tracking_timestamp_us: int | None = None

    # 功能：
    #   将未被坐标适配器覆盖的接口交给底层客户端，例如连接、解锁和降落。
    # 输入：
    #   self：持有实际客户端的适配器。
    #   name：所需属性或方法名。
    # 输出：
    #   attribute：底层对应属性或绑定方法。
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    # 功能：
    #   计算跨越正负一百八十度边界时的最短航向误差。
    # 输入：
    #   commanded_deg：已发送航向，单位度。
    #   observed_deg：实测航向，单位度。
    # 输出：
    #   error_deg：非负最短角度差。
    @staticmethod
    def _absolute_heading_error_deg(commanded_deg: float, observed_deg: float) -> float:
        return abs((observed_deg - commanded_deg + 180.0) % 360.0 - 180.0)

    # 功能：
    #   对已有误差样本排序后做线性分位插值，空样本拒绝生成统计值。
    # 输入：
    #   values：有限的航向误差样本列表。
    #   percentile：零到一之间的目标分位。
    # 输出：
    #   quantile：该分位对应的插值误差。
    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("cannot compute a heading percentile without samples")
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    # 功能：
    #   从已有姿态缓存记录新鲜航向误差，不新增遥测订阅，也不授予运动权限。
    # 输入：
    #   self：保存诊断样本及上一来源序号的适配器。
    #   commanded_yaw_deg：本次实际发送给飞控的航向。
    # 输出：
    #   None：不返回业务数据。
    def _record_heading_tracking(self, commanded_yaw_deg: float) -> None:
        """Passively compare a fresh PX4 attitude sample with the sent yaw.

        Recording must never become a second telemetry subscription or a new
        motion authority.  It reads the dynamics collector's latest-value
        cache and deduplicates MAVSDK timestamps, so the evidence represents
        observed airframe attitude rather than the controller write rate.
        """

        getter = getattr(self._client, "latest_dynamics_telemetry", None)
        if not callable(getter):
            return
        try:
            dynamics = getter(1.0)
            if not isinstance(dynamics, dict):
                return
            sources = dynamics.get("sources", {})
            if not isinstance(sources, dict):
                return
            attitude = sources.get("attitude")
            if not isinstance(attitude, dict):
                return
            observed_yaw_deg = attitude["yaw_deg"]
            sample_age_seconds = attitude["sample_age_seconds"]
            timestamp_us = attitude.get("timestamp_us")
            if not all(
                type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
                for value in (commanded_yaw_deg, observed_yaw_deg, sample_age_seconds)
            ):
                return
            if sample_age_seconds < 0.0 or sample_age_seconds > 1.0:
                return
            if timestamp_us is not None:
                if type(timestamp_us) is not int or not 0 <= timestamp_us < 2**63:
                    return
                if (
                    self._last_heading_tracking_timestamp_us is not None
                    and timestamp_us <= self._last_heading_tracking_timestamp_us
                ):
                    return
                self._last_heading_tracking_timestamp_us = timestamp_us
            self._heading_tracking_commanded_deg.append(float(commanded_yaw_deg))
            self._heading_tracking_observed_deg.append(observed_yaw_deg)
            self._heading_tracking_sample_ages_seconds.append(sample_age_seconds)
            self._heading_tracking_errors_deg.append(
                self._absolute_heading_error_deg(commanded_yaw_deg, observed_yaw_deg)
            )
            self._publish_heading_tracking_evidence_if_due()
        except (KeyError, TypeError, ValueError):
            # Malformed or concurrently unavailable advisory evidence cannot
            # interrupt the flight controller.  The acceptance gate will fail
            # closed later if it cannot obtain enough valid samples.
            return

    # 功能：
    #   按间隔将被动航向证据交给单个后台任务；慢写期间不排队，不阻塞控制线程。
    # 输入：
    #   self：航向样本和证据输出配置。
    # 输出：
    #   None：不返回业务数据。
    def _publish_heading_tracking_evidence_if_due(self) -> None:
        if self._heading_evidence_path is None:
            return
        pending = self._pending_heading_publication
        if pending is not None:
            if not pending.done():
                return
            try:
                pending.result()
            except (Exception, asyncio.CancelledError) as error:
                self._heading_evidence_writer_issue = type(error).__name__
            self._pending_heading_publication = None
        now = time.monotonic()
        if (
            now - self._last_heading_evidence_flush_monotonic
            < self._heading_evidence_flush_interval_seconds
        ):
            return
        try:
            loop = asyncio.get_running_loop()
            # 先生成独立标量报告；线程不读取仍被控制周期修改的采样数组。
            payload = self.heading_tracking_evidence()
            self._pending_heading_publication = loop.create_task(
                asyncio.to_thread(_atomic_json, self._heading_evidence_path, payload),
                name="heading-diagnostic-publication")
            self._last_heading_evidence_flush_monotonic = now
        except RuntimeError as exc:
            # 无事件循环时不退回同步写盘；记录错误供最终验收拒绝。
            self._heading_evidence_writer_issue = type(exc).__name__

    # 功能：
    #   在任务收尾有限等待旧写入，再保存最终统计；超时保留在途任务及故障，不另开并发写入。
    # 输入：
    #   self：本次飞行的航向记录器。
    # 输出：
    #   evidence：包括写入错误状态的最新统计字典。
    async def flush_heading_tracking_evidence(self) -> dict[str, Any]:
        pending = self._pending_heading_publication
        if pending is not None:
            try:
                await asyncio.wait_for(asyncio.shield(pending), timeout=4.)
            except (Exception, asyncio.CancelledError) as error:
                self._heading_evidence_writer_issue = type(error).__name__
            if not pending.done():
                # 未结束写入仍属于本实例，不能另开写入覆盖它，也不能假称已经排空。
                return self.heading_tracking_evidence()
            self._pending_heading_publication = None
        if self._heading_evidence_path is not None:
            payload = self.heading_tracking_evidence()
            self._pending_heading_publication = asyncio.create_task(
                asyncio.to_thread(_atomic_json, self._heading_evidence_path, payload))
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._pending_heading_publication), timeout=4.)
            except (Exception, asyncio.CancelledError) as error:
                self._heading_evidence_writer_issue = type(error).__name__
            if self._pending_heading_publication.done():
                self._pending_heading_publication = None
        evidence = self.heading_tracking_evidence()
        return evidence

    # 功能：
    #   汇总实测误差的均值、分位、最大值及阈值内计数，无样本时明确不可用。
    # 输入：
    #   self：按来源时间去重的姿态与控制样本。
    # 输出：
    #   evidence：独立生成的航向诊断字典。
    def heading_tracking_evidence(self) -> dict[str, Any]:
        errors = self._heading_tracking_errors_deg
        evidence: dict[str, Any] = {
            "schema_version": "dronedream.px4-heading-tracking.v1",
            "policy": self._heading_policy,
            "source": "mavsdk-attitude-euler-latest-value",
            "freshness_bound_seconds": 1.0,
            "sample_count": len(errors),
            "writer_issue": self._heading_evidence_writer_issue,
        }
        if not errors:
            return {**evidence, "status": "unavailable"}
        evidence.update(
            {
                "status": "observed",
                "commanded_yaw_min_deg": min(self._heading_tracking_commanded_deg),
                "commanded_yaw_max_deg": max(self._heading_tracking_commanded_deg),
                "observed_yaw_min_deg": min(self._heading_tracking_observed_deg),
                "observed_yaw_max_deg": max(self._heading_tracking_observed_deg),
                "maximum_sample_age_seconds": max(self._heading_tracking_sample_ages_seconds),
                "absolute_error_deg": {
                    "mean": sum(errors) / len(errors),
                    "p50": self._percentile(errors, 0.50),
                    "p95": self._percentile(errors, 0.95),
                    "p99": self._percentile(errors, 0.99),
                    "maximum": max(errors),
                },
                "within_15_deg_count": sum(error <= 15.0 for error in errors),
                "within_30_deg_count": sum(error <= 30.0 for error in errors),
            }
        )
        return evidence

    # 功能：
    #   按普通路线航向策略发送相对原点的位置，不用于改写已授权模型航向。
    # 输入：
    #   self：出生点相对坐标适配器。
    #   setpoint：相对北、东、向下位置及输入航向。
    # 输出：
    #   None：不返回业务数据。
    async def set_position_ned(self, setpoint: Any) -> None:
        yaw_deg = setpoint.yaw_deg if self._heading_hold_deg is None else self._heading_hold_deg
        await self._forward_position_ned(setpoint, yaw_deg=yaw_deg)

    # 功能：
    #   发送保护位置并保留局部实测航向，不恢复为起飞前路线航向。
    # 输入：
    #   self：坐标适配器。
    #   setpoint：局部安全控制器给出的相对位置与航向。
    # 输出：
    #   None：不返回业务数据。
    async def set_local_position_ned(self, setpoint: Any) -> None:
        """Preserve a locally measured safety-hold heading after a model turn."""
        await self._forward_position_ned(setpoint, yaw_deg=setpoint.yaw_deg)

    # 功能：
    #   为位置加上实测原点后发送至实际飞控，成功发送后记录航向观测。
    # 输入：
    #   self：含本地坐标原点的适配器。
    #   setpoint：出生点相对位置。
    #   yaw_deg：上游明确选定的航向。
    # 输出：
    #   None：不返回业务数据。
    async def _forward_position_ned(self, setpoint: Any, *, yaw_deg: float) -> None:
        await self._client.set_position_ned(
            type(setpoint)(
                north_m=self._origin.north_m + setpoint.north_m,
                east_m=self._origin.east_m + setpoint.east_m,
                down_m=self._origin.down_m + setpoint.down_m,
                yaw_deg=yaw_deg,
            )
        )
        self._record_heading_tracking(yaw_deg)

    # 功能：
    #   按路线航向策略发送位置与速度前馈，只平移位置部分。
    # 输入：
    #   self：相对坐标适配器。
    #   setpoint：相对 NED 位置及航向。
    #   velocity：NED 速度前馈，单位米每秒。
    # 输出：
    #   None：不返回业务数据。
    async def set_position_velocity_ned(self, setpoint: Any, velocity: Any) -> None:
        """Rebase the position half while preserving spawn-relative velocity.

        MAVSDK's combined position/velocity command is a separate method from
        ``set_position_ned``.  Letting ``__getattr__`` forward it directly would
        silently send spawn-relative positions as estimator-local coordinates
        after takeoff, even though every observation exposed by this adapter is
        spawn-relative.  Velocity is already a frame-relative vector, so it must
        not receive the positional origin translation.
        """

        yaw_deg = setpoint.yaw_deg if self._heading_hold_deg is None else self._heading_hold_deg
        await self._forward_position_velocity_ned(setpoint, velocity, yaw_deg=yaw_deg)

    # 功能：
    #   发送安全仲裁后的位置和速度前馈，并保留仲裁后的航向。
    # 输入：
    #   self：相对坐标适配器。
    #   setpoint：安全位置目标及航向。
    #   velocity：安全速度前馈。
    # 输出：
    #   None：不返回业务数据。
    async def set_local_position_velocity_ned(self, setpoint: Any, velocity: Any) -> None:
        """Safety arbitration, not the pre-arm route policy, owns this heading."""
        await self._forward_position_velocity_ned(setpoint, velocity, yaw_deg=setpoint.yaw_deg)

    # 功能：
    #   将位置平移到飞控本地坐标，速度向量不平移，再发送组合接口并记录实际航向。
    # 输入：
    #   self：实际飞控及原点信息。
    #   setpoint：相对位置目标。
    #   velocity：NED 速度前馈。
    #   yaw_deg：统一传给位置和速度部分的航向。
    # 输出：
    #   None：不返回业务数据。
    async def _forward_position_velocity_ned(
        self,
        setpoint: Any,
        velocity: Any,
        *,
        yaw_deg: float,
    ) -> None:
        await self._client.set_position_velocity_ned(
            type(setpoint)(
                north_m=self._origin.north_m + setpoint.north_m,
                east_m=self._origin.east_m + setpoint.east_m,
                down_m=self._origin.down_m + setpoint.down_m,
                yaw_deg=yaw_deg,
            ),
            type(velocity)(
                north_m_s=velocity.north_m_s,
                east_m_s=velocity.east_m_s,
                down_m_s=velocity.down_m_s,
                yaw_deg=yaw_deg,
            ),
        )
        self._record_heading_tracking(yaw_deg)

    # 功能：
    #   按普通航向策略发送纯速度；速度是向量，不加出生点位置偏移。
    # 输入：
    #   self：坐标与航向适配器。
    #   velocity：NED 速度及普通输入航向。
    # 输出：
    #   None：不返回业务数据。
    async def set_velocity_ned(self, velocity: Any) -> None:
        """Forward a velocity-only NED request without applying the spawn origin.

        Velocity is a vector, not a position, so the spawn-relative adapter
        must never translate it.  This path is used for short-lived local
        model control; position holds continue to use ``set_position_ned``.
        """

        yaw_deg = velocity.yaw_deg if self._heading_hold_deg is None else self._heading_hold_deg
        await self._forward_velocity_ned(velocity, yaw_deg=yaw_deg)

    # 功能：
    #   原样保留已授权模型速度与航向，通过纯速度接口控制，不悄悄替换成位置目标。
    # 输入：
    #   self：实际飞控适配器。
    #   velocity：已转成世界 NED 的速度及模型授权航向。
    # 输出：
    #   None：不返回业务数据。
    async def set_model_velocity_ned(self, velocity: Any) -> None:
        """Preserve the already-authorized model yaw as well as its velocity.

        Pre-arm/route heading hold is not permission to rewrite an approved
        model command while reporting that its requested yaw was transmitted.
        Only the bounded model/safety dispatch path calls this explicit entry.
        The lower flight-controller interface still receives pure world NED
        velocity plus a heading in degrees, not heading-relative body velocity.
        """
        await self._forward_velocity_ned(velocity, yaw_deg=velocity.yaw_deg)

    # 功能：
    #   构造并发送底层纯速度指令，发送成功后只做被动航向记录。
    # 输入：
    #   self：实际飞控适配器。
    #   velocity：北、东、向下速度，单位米每秒。
    #   yaw_deg：明确授权的航向角。
    # 输出：
    #   None：不返回业务数据。
    async def _forward_velocity_ned(self, velocity: Any, *, yaw_deg: float) -> None:
        await self._client.set_velocity_ned(
            type(velocity)(
                north_m_s=velocity.north_m_s,
                east_m_s=velocity.east_m_s,
                down_m_s=velocity.down_m_s,
                yaw_deg=yaw_deg,
            )
        )
        self._record_heading_tracking(yaw_deg)

    # 功能：
    #   从实际飞控读取位置速度，只对位置减去原点，保留原始接收时刻而不刷新缓存年龄。
    # 输入：
    #   self：含出生点原点的适配器。
    #   timeout_seconds：底层采样等待预算。
    # 输出：
    #   observed_relative：相对 NED 位置、原速度和来源接收时间组成的样本。
    async def sample_position_velocity_ned(self, timeout_seconds: float) -> Any:
        observed = await self._client.sample_position_velocity_ned(timeout_seconds)
        return type(observed)(
            north_m=observed.north_m - self._origin.north_m,
            east_m=observed.east_m - self._origin.east_m,
            down_m=observed.down_m - self._origin.down_m,
            north_m_s=observed.north_m_s,
            east_m_s=observed.east_m_s,
            down_m_s=observed.down_m_s,
            **(
                {"received_at_unix_ms": observed.received_at_unix_ms}
                if hasattr(observed, "received_at_unix_ms")
                else {}
            ),
        )


class RuntimeInterruptDetected(RuntimeError):
    # 功能：
    #   携带已领取的用户改令向上中断当前执行步骤，保留消息身份与检测时间。
    # 输入：
    #   self：本异常实例。
    #   message：本次已领取的类型化用户改令。
    #   claimed_path：领取后消息文件的位置。
    #   detected_at：首次检测该消息的 UTC 时刻。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self, message: RuntimeUserMessage, claimed_path: Path, detected_at: datetime
    ) -> None:
        super().__init__(message.message_id)
        self.message = message
        self.claimed_path = claimed_path
        self.detected_at = detected_at


class RuntimeTrackReplacement(RuntimeError):
    # 功能：
    #   携带已验证的新轨迹、动作及坐标绑定向主循环转交控制，避免恢复到旧路线。
    # 输入：
    #   self：轨迹替换异常实例。
    #   message_id：触发本次替换的用户消息。
    #   schedule：新轨迹编译后的控制参考序列。
    #   waypoint_arrival_indices：源航点到参考序列的到达索引。
    #   track_sha256：被采用轨迹的摘要。
    #   replacement_sequence：当前运行内的替换次序。
    #   amendment_action：替换对应的用户意图类型。
    #   amendment_parameters：经授权的改令参数。
    #   coordinate_contract：新轨迹坐标绑定。
    #   artifact：完整的任务、动作、检查点与轨迹替换制品。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        message_id: str,
        schedule: list[Any],
        waypoint_arrival_indices: tuple[int, ...],
        track_sha256: str,
        replacement_sequence: int,
        amendment_action: str,
        amendment_parameters: dict[str, Any],
        coordinate_contract: Px4CoordinateContract,
        artifact: RuntimeReplacementTrack | None = None,
    ) -> None:
        super().__init__(message_id)
        self.message_id = message_id
        self.schedule = schedule
        self.waypoint_arrival_indices = waypoint_arrival_indices
        self.track_sha256 = track_sha256
        self.replacement_sequence = replacement_sequence
        self.amendment_action = amendment_action
        self.amendment_parameters = amendment_parameters
        self.coordinate_contract = coordinate_contract
        self.artifact = artifact


# 功能：
#   1. 从已经稳定的当前位置编译替换轨迹，不重复执行起飞流程。
#   2. 复用初始轨迹的停稳式调度，统一限速、等待和零距离航点的处理。
# 输入：
#   base：与当前发布配套的飞控执行器模块。
#   replacement：已经验证并绑定的替换轨迹制品。
#   params：当前控制器速度及加速度上限。
#   rate_hz：位置设定值采样频率。
# 输出：
#   result：替换调度列表及各后续航点到达下标组成的二元组。
def _compile_replacement_schedule(
    *,
    base: ModuleType,
    replacement: RuntimeReplacementTrack,
    params: Any,
    rate_hz: float,
) -> tuple[list[Any], tuple[int, ...]]:
    points = [
        base.TrackPoint(point.x, point.y, point.z, point.speed_limit_mps)
        for point in replacement.track.points
    ]
    if len(points) < 2:
        raise RuntimeError("runtime replacement track contains fewer than two points")
    if replacement.track.stop_at_waypoints is not True:
        raise RuntimeError("runtime replacement requires stopped waypoint execution")
    schedule = [base.enu_point_to_ned_setpoint(points[0], yaw_deg=0.0)]
    motion, relative_arrivals = base.build_stopped_waypoint_schedule(
        points,
        params,
        rate_hz,
        replacement.track.waypoint_hold_seconds,
        initial_yaw_deg=0.0,
        max_samples=base.MAX_SETPOINTS - len(schedule),
    )
    waypoint_arrival_indices = tuple(len(schedule) + index for index in relative_arrivals)
    schedule.extend(motion)
    final = base.enu_point_to_ned_setpoint(points[-1], yaw_deg=schedule[-1].yaw_deg)
    final_samples = max(2, int(rate_hz * 0.5))
    if final_samples > base.MAX_SETPOINTS - len(schedule):
        raise RuntimeError("runtime replacement schedule exceeds setpoint limit")
    schedule.extend(final for _ in range(final_samples))
    result = schedule, waypoint_arrival_indices
    return result


# 功能：
#   读取本次运行的改令会话；未启用改令目录时明确不创建会话。
# 输入：
#   control_dir：当前任务的改令控制目录。
# 输出：
#   session：有效会话对象，未配置目录时为 None。
def _runtime_session(control_dir: Path | None) -> RuntimeControlSession | None:
    if control_dir is None:
        return None
    return RuntimeControlSession.model_validate(read_runtime_object(control_dir / "session.json"))


# 功能：
#   按文件顺序领取一条改令，复核任务及执行身份并关闭旧计划副作用，拒绝串用其他会话。
# 输入：
#   control_dir：本次运行的消息收件与领取目录。
#   session：当前有效的任务、计划与执行会话。
# 输出：
#   interruption：携带已领取消息的中断对象，没有有效消息时为 None。
def _claim_runtime_message(
    control_dir: Path | None, session: RuntimeControlSession | None
) -> RuntimeInterruptDetected | None:
    if control_dir is None or session is None:
        return None
    for inbox_path in sorted((control_dir / "inbox").glob("runtime-msg-*.json")):
        claimed_path = control_dir / "claimed" / inbox_path.name
        try:
            transfer_runtime_file(inbox_path, claimed_path)
        except FileNotFoundError:
            continue
        message = RuntimeUserMessage.model_validate(read_runtime_object(claimed_path))
        gates = {
            "conversation": message.conversation_id == session.conversation_id,
            "mission": message.mission_id == session.mission_id,
            "plan_revision": message.plan_revision_id == session.plan_revision_id,
            "contract": message.contract_id == session.contract_id,
            "execution": message.execution_id == session.execution_id,
            "session_accepting": session.state == "accepting",
        }
        if not all(gates.values()):
            _atomic_json(
                control_dir / "rejected" / claimed_path.name,
                {
                    "message": message.model_dump(mode="json"),
                    "identity_gates": gates,
                    "rejected_at": datetime.now(UTC).isoformat(),
                },
            )
            raise RuntimeError("RUNTIME_MESSAGE_SESSION_BINDING_MISMATCH")
        detected_at = datetime.now(UTC)
        _atomic_json(
            control_dir / "side-effects.state.json",
            {
                "enabled": False,
                "execution_id": message.execution_id,
                "message_id": message.message_id,
                "reason": "runtime user message preempted the confirmed plan",
                "updated_at": detected_at.isoformat(),
            },
        )
        _atomic_json(
            control_dir / "detected" / claimed_path.name,
            {
                "message_sha256": sha256_json(message),
                "message_id": message.message_id,
                "detected_at": detected_at.isoformat(),
                "old_plan_advancement_inhibited": True,
                "semantic_side_effects_inhibited": True,
            },
        )
        return RuntimeInterruptDetected(message, claimed_path, detected_at)
    return None


# 功能：
#   执行一次保护悬停并检查外部终止；配置遥测时同步刷新实际身份观测。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   hold_setpoint：当前保护位置及航向。
#   abort_file：外部终止信号路径。
#   rate_hz：悬停维持频率。
#   telemetry_args：可选遥测配置与输出路径。
#   coordinate_contract：可选遥测坐标绑定。
# 输出：
#   observed：刷新所得的位置速度样本，未启用刷新时为 None。
async def _runtime_hold_tick(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    abort_file: Path,
    rate_hz: float,
    telemetry_args: argparse.Namespace | None,
    coordinate_contract: Px4CoordinateContract | None,
) -> Any | None:
    """Maintain Offboard hold and the independently monitored PX4 identity feed."""

    base._raise_if_external_abort_requested(abort_file)
    await client.set_position_ned(hold_setpoint)
    if telemetry_args is None or coordinate_contract is None:
        return None
    return await base._await_with_setpoint_keepalive(
        client,
        _refresh_px4_identity_telemetry(
            args=telemetry_args,
            client=client,
            coordinate_contract=coordinate_contract,
        ),
        hold_setpoint=hold_setpoint,
        rate_hz=rate_hz,
        abort_check=lambda: base._raise_if_external_abort_requested(abort_file),
    )


# 功能：
#   以实测位置建立连续稳定悬停，并将改令身份、执行器进度与实测状态绑定到回执。
# 输入：
#   base：底层飞控辅助模块。
#   client：飞控客户端。
#   frozen_setpoint：中断瞬间保留的设定值。
#   interruption：当前用户改令事件。
#   control_dir：本次执行控制目录。
#   phase：被中断的执行阶段。
#   schedule_index：当前调度采样下标，不作为原始航点下标。
#   abort_file：外部停止请求路径。
#   rate_hz：悬停刷新频率。
#   timeout_seconds：建立连续稳定观测的最大等待时间。
#   telemetry_args：实时感知和飞控遥测上下文。
#   coordinate_contract：当前局部坐标与世界坐标的绑定。
#   track_progress：执行器捕获的下一航点及轨迹摘要，无证据时为空。
# 输出：
#   result：已发布的悬停回执与测量位置对应的保持设定值。
# 功能：
#   在改令后以实测位置建立悬停，连续通过位置和速度门槛后发布绑定原消息的稳定回执。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   frozen_setpoint：中断前的控制参考及保持航向。
#   interruption：已领取的用户消息与检测时刻。
#   control_dir：改令回执目录。
#   phase：被中断的执行阶段。
#   schedule_index：被中断的参考序列索引。
#   abort_file：外部终止信号。
#   rate_hz：悬停控制频率。
#   timeout_seconds：建立稳定悬停的总时限。
#   telemetry_args：原生遥测发布配置。
#   coordinate_contract：遥测到地图的绑定。
#   track_progress：中断时冻结的原轨迹进度。
# 输出：
#   hold_result：稳定回执和实测悬停设定值组成的二元组。
async def _stabilize_runtime_hold(
    *,
    base: ModuleType,
    client: Any,
    frozen_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    control_dir: Path,
    phase: str,
    schedule_index: int | None,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
    track_progress: RuntimeTrackProgress | None = None,
) -> tuple[RuntimeHoldAcknowledgement, Any]:
    detected_at = interruption.detected_at
    detection_latency_ms = max(
        0,
        round((detected_at - interruption.message.submitted_at).total_seconds() * 1000),
    )
    observed = await _runtime_hold_tick(
        base=base,
        client=client,
        hold_setpoint=frozen_setpoint,
        abort_file=abort_file,
        rate_hz=rate_hz,
        telemetry_args=telemetry_args,
        coordinate_contract=coordinate_contract,
    )
    if observed is None:
        observed = await client.sample_position_velocity_ned(1.0)
    hold_setpoint = type(frozen_setpoint)(
        north_m=observed.north_m,
        east_m=observed.east_m,
        down_m=observed.down_m,
        yaw_deg=frozen_setpoint.yaw_deg,
    )
    started = time.monotonic()
    stable_since: float | None = None
    latest = observed
    position_error = math.inf
    speed = math.inf
    while time.monotonic() - started < timeout_seconds:
        latest = await _runtime_hold_tick(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            abort_file=abort_file,
            rate_hz=rate_hz,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        if latest is None:
            latest = await client.sample_position_velocity_ned(1.0)
        position_error = math.dist(
            (latest.north_m, latest.east_m, latest.down_m),
            (hold_setpoint.north_m, hold_setpoint.east_m, hold_setpoint.down_m),
        )
        speed = math.sqrt(latest.north_m_s**2 + latest.east_m_s**2 + latest.down_m_s**2)
        now = time.monotonic()
        if position_error <= 0.5 and speed <= 0.35:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= 1.0:
                break
        else:
            stable_since = None
        await asyncio.sleep(1.0 / rate_hz)
    else:
        _atomic_json(
            control_dir / "hold-failures" / f"{interruption.message.message_id}.json",
            {
                "message_sha256": sha256_json(interruption.message),
                "phase": phase,
                "position_error_m": position_error,
                "speed_mps": speed,
                "failure": "RUNTIME_HOLD_STABILITY_TIMEOUT",
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
        raise TimeoutError("runtime interruption could not establish stable hover")

    gates = {
        "telemetry_finite": all(
            math.isfinite(value)
            for value in (
                latest.north_m,
                latest.east_m,
                latest.down_m,
                latest.north_m_s,
                latest.east_m_s,
                latest.down_m_s,
                position_error,
                speed,
            )
        ),
        "position_error_within_0_50_m": position_error <= 0.5,
        "speed_within_0_35_mps": speed <= 0.35,
        "old_plan_advancement_inhibited": True,
        "semantic_side_effects_inhibited": True,
    }
    if not all(gates.values()):
        raise RuntimeError("runtime hold failed deterministic telemetry gates")
    stable_at = datetime.now(UTC)
    acknowledgement = RuntimeHoldAcknowledgement(
        message_sha256=sha256_json(interruption.message),
        message_id=interruption.message.message_id,
        execution_id=interruption.message.execution_id,
        interrupted_phase=phase,
        schedule_index=schedule_index,
        track_progress=track_progress.model_copy(deep=True) if track_progress is not None else None,
        detected_at=detected_at,
        detection_latency_ms=detection_latency_ms,
        stable_at=stable_at,
        stabilization_latency_ms=round((time.monotonic() - started) * 1000),
        frozen_command_ned_m=Vector3(
            x=frozen_setpoint.north_m,
            y=frozen_setpoint.east_m,
            z=frozen_setpoint.down_m,
        ),
        hold_command_ned_m=Vector3(
            x=hold_setpoint.north_m,
            y=hold_setpoint.east_m,
            z=hold_setpoint.down_m,
        ),
        observed_position_ned_m=Vector3(x=latest.north_m, y=latest.east_m, z=latest.down_m),
        observed_velocity_ned_mps=Vector3(x=latest.north_m_s, y=latest.east_m_s, z=latest.down_m_s),
        position_error_m=position_error,
        speed_mps=speed,
        deterministic_gates=gates,
    )
    _atomic_json(
        control_dir / "acks" / f"{interruption.message.message_id}.json",
        acknowledgement,
    )
    result = acknowledgement, hold_setpoint
    return result


# 功能：
#   维持稳定悬停等待改令决定，核对消息、悬停回执及授权门槛，超时拒绝继续。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   hold_setpoint：已稳定的保护位置。
#   interruption：当前改令消息。
#   acknowledgement：已经发布的稳定悬停回执。
#   control_dir：决定文件目录。
#   abort_file：外部终止信号。
#   rate_hz：悬停维持频率。
#   timeout_seconds：等待决定的总时限。
#   telemetry_args：遥测输出配置。
#   coordinate_contract：当前坐标绑定。
# 输出：
#   decision：身份和摘要均通过检查的改令决定。
async def _wait_runtime_decision(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    acknowledgement: RuntimeHoldAcknowledgement,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
) -> RuntimeInterruptionDecision:
    decision_path = control_dir / "decisions" / f"{interruption.message.message_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        await _runtime_hold_tick(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            abort_file=abort_file,
            rate_hz=rate_hz,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        if decision_path.is_file():
            decision = RuntimeInterruptionDecision.model_validate(
                read_runtime_object(decision_path)
            )
            if decision.message_sha256 != sha256_json(interruption.message):
                raise RuntimeError("runtime decision message hash mismatch")
            if decision.hold_ack_sha256 != sha256_json(acknowledgement):
                raise RuntimeError("runtime decision hold acknowledgement hash mismatch")
            if not all(decision.authorization_gates.values()):
                raise RuntimeError("runtime decision contains a failed authorization gate")
            return decision
        await asyncio.sleep(1.0 / rate_hz)
    raise TimeoutError("runtime interruption model decision timeout")


# 功能：
#   在悬停中等待完整替换制品，复核旧轨迹来源、新轨迹净空及起点衔接后才交还执行权。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   hold_setpoint：替换开始前的稳定位置。
#   interruption：当前改令消息。
#   acknowledgement：稳定悬停回执。
#   decision：允许替换的改令决定。
#   control_dir：替换制品与失败回执目录。
#   abort_file：外部终止信号。
#   rate_hz：保护维持频率。
#   timeout_seconds：等待替换制品的时限。
#   active_track_sha256：必须被替换的当前轨迹摘要。
#   telemetry_args：遥测输出配置。
#   coordinate_contract：当前悬停位置的坐标绑定。
# 输出：
#   replacement：全部绑定检查通过的新轨迹制品。
async def _wait_runtime_replacement(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    active_track_sha256: str,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
) -> RuntimeReplacementTrack:
    replacement_path = control_dir / "replacements" / f"{interruption.message.message_id}.json"
    failure_path = control_dir / "replan-failures" / f"{interruption.message.message_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        await _runtime_hold_tick(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            abort_file=abort_file,
            rate_hz=rate_hz,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        if failure_path.is_file():
            failure = read_runtime_object(failure_path)
            raise UserDirectedLanding(
                f"runtime replacement rejected: {failure.get('reason', 'unknown')}"
            )
        if replacement_path.is_file():
            replacement = RuntimeReplacementTrack.model_validate(
                read_runtime_object(replacement_path, maximum_bytes=MAX_RUNTIME_REPLACEMENT_BYTES)
            )
            gates = {
                "message_id": replacement.message_id == interruption.message.message_id,
                "execution_id": replacement.execution_id == interruption.message.execution_id,
                "message_hash": replacement.message_sha256 == sha256_json(interruption.message),
                "hold_hash": replacement.hold_ack_sha256 == sha256_json(acknowledgement),
                "decision_hash": replacement.decision_sha256 == sha256_json(decision),
                "prior_track_hash": replacement.prior_track_sha256 == active_track_sha256,
                "planner_gates": all(replacement.deterministic_gates.values()),
                "clearance": replacement.clearance.accepted,
                "starts_at_hold": (
                    math.dist(
                        (
                            replacement.track.points[0].x,
                            replacement.track.points[0].y,
                            -replacement.track.points[0].z,
                        ),
                        (
                            hold_setpoint.north_m,
                            hold_setpoint.east_m,
                            hold_setpoint.down_m,
                        ),
                    )
                    <= 0.75
                ),
            }
            if not all(gates.values()):
                failed = ",".join(name for name, accepted in gates.items() if not accepted)
                raise UserDirectedLanding(f"runtime replacement binding failed: {failed}")
            return replacement
        await asyncio.sleep(1.0 / rate_hz)
    raise UserDirectedLanding("runtime replan was not supplied within the bounded safe-hold window")


# 功能：
#   等待相机、载荷或避障设备命令，验证其消息、决定和悬停摘要后才允许交给驱动。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   hold_setpoint：保护位置。
#   interruption：当前改令消息。
#   acknowledgement：已通过的稳定回执。
#   decision：允许设备操作的决定。
#   control_dir：命令及失败证据目录。
#   abort_file：外部终止信号。
#   rate_hz：悬停控制频率。
#   timeout_seconds：命令等待预算。
#   telemetry_args：遥测输出配置。
#   coordinate_contract：当前坐标绑定。
# 输出：
#   command：通过绑定与授权检查的设备命令。
async def _wait_runtime_command(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
) -> RuntimeAuthorizedCommand:
    command_path = control_dir / "commands" / f"{interruption.message.message_id}.json"
    failure_path = control_dir / "command-failures" / f"{interruption.message.message_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        await _runtime_hold_tick(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            abort_file=abort_file,
            rate_hz=rate_hz,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        if failure_path.is_file():
            failure = read_runtime_object(failure_path)
            raise UserDirectedLanding(
                f"runtime command rejected: {failure.get('reason', 'unknown')}"
            )
        if command_path.is_file():
            command = RuntimeAuthorizedCommand.model_validate(read_runtime_object(command_path))
            gates = {
                "message_id": command.message_id == interruption.message.message_id,
                "execution_id": command.execution_id == interruption.message.execution_id,
                "message_hash": command.message_sha256 == sha256_json(interruption.message),
                "hold_hash": command.hold_ack_sha256 == sha256_json(acknowledgement),
                "decision_hash": command.decision_sha256 == sha256_json(decision),
                "command_gates": all(command.deterministic_gates.values()),
            }
            if not all(gates.values()):
                failed = ",".join(name for name, accepted in gates.items() if not accepted)
                raise UserDirectedLanding(f"runtime command binding failed: {failed}")
            return command
        await asyncio.sleep(1.0 / rate_hz)
    raise UserDirectedLanding(
        "runtime command was not supplied within the bounded safe-hold window"
    )


# 功能：
#   1. 在保持悬停的同时运行已授权设备操作，只有明确正向回读才登记采纳并恢复原任务。
#   2. 超时、异常和取消均收回自有设备子任务；失败不能写成成功副作用。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控与设备客户端。
#   hold_setpoint：操作期间的稳定位置。
#   interruption：当前消息与执行身份。
#   command：已经验证的设备命令。
#   control_dir：操作失败、结果和采纳证据目录。
#   abort_file：外部终止信号。
#   rate_hz：悬停控制频率。
#   timeout_seconds：设备操作的总等待预算。
#   telemetry_args：遥测发布配置。
#   coordinate_contract：坐标绑定。
# 输出：
#   outcome：设备确认成功后返回的 resume_original 状态。
async def _execute_runtime_command(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    command: RuntimeAuthorizedCommand,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
) -> str:
    if command.action == "camera_control":
        operation = client.execute_camera_command(dict(command.parameters))
    elif command.action == "payload_control":
        operation = client.execute_payload_command(dict(command.parameters))
    elif command.action == "set_avoidance":
        operation = client.execute_avoidance_command(bool(command.parameters["enabled"]))
    else:
        raise UserDirectedLanding(f"unsupported runtime command action: {command.action}")

    task = asyncio.create_task(operation)
    deadline = time.monotonic() + timeout_seconds
    try:
        while not task.done() and time.monotonic() < deadline:
            await _runtime_hold_tick(
                base=base,
                client=client,
                hold_setpoint=hold_setpoint,
                abort_file=abort_file,
                rate_hz=rate_hz,
                telemetry_args=telemetry_args,
                coordinate_contract=coordinate_contract,
            )
            await asyncio.sleep(1.0 / rate_hz)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise UserDirectedLanding("runtime command execution timed out during stable hold")
        observed_result = await task
    except BaseException as exc:
        # 必须先收回本次设备操作，再交还悬停／降落控制；取消同样走这个路径。
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        with contextlib.suppress(Exception):
            _atomic_json(
                control_dir / "command-failures" / f"{interruption.message.message_id}.json",
                {
                    "message_id": interruption.message.message_id,
                    "command_sha256": sha256_json(command),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "failed_at": datetime.now(UTC).isoformat(),
                },
            )
        raise
    if not isinstance(observed_result, dict) or observed_result.get("confirmed") is not True:
        reason = "runtime command completed without a positive device readback"
        _atomic_json(
            control_dir / "command-failures" / f"{interruption.message.message_id}.json",
            {
                "message_id": interruption.message.message_id,
                "command_sha256": sha256_json(command),
                "reason": reason,
                "observed_result": observed_result,
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
        raise UserDirectedLanding(reason)
    result = {
        "message_id": interruption.message.message_id,
        "execution_id": interruption.message.execution_id,
        "action": command.action,
        "command_sha256": sha256_json(command),
        "observed_result": observed_result,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(
        control_dir / "command-results" / f"{interruption.message.message_id}.json", result
    )
    adoption = RuntimeCommandAdoption(
        message_id=interruption.message.message_id,
        execution_id=interruption.message.execution_id,
        action=command.action,
        command_sha256=sha256_json(command),
        result_sha256=sha256_json(result),
        observed_result=observed_result,
        adopted_at=datetime.now(UTC),
    )
    _atomic_json(control_dir / "adoptions" / f"{interruption.message.message_id}.json", adoption)
    _atomic_json(
        control_dir / "side-effects.state.json",
        {
            "enabled": True,
            "execution_id": interruption.message.execution_id,
            "message_id": interruption.message.message_id,
            "reason": "hash-bound runtime command executed and device-confirmed",
            "command_sha256": sha256_json(command),
            "result_sha256": sha256_json(result),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return "resume_original"


# 功能：
#   1. 冻结旧进度并稳定悬停，根据已验证决定分派继续、降落、设备操作、人工接管或重规划。
#   2. 替换路线必须携带完整任务制品，长期暂停期间仍接受新的用户改令。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控与设备客户端。
#   frozen_setpoint：中断前最后的参考位置及航向。
#   interruption：本次已领取消息。
#   control_dir：本次改令会话目录。
#   phase：被中断阶段。
#   schedule_index：被中断参考索引。
#   abort_file：外部终止信号。
#   rate_hz：保护控制频率。
#   hold_timeout_seconds：稳定悬停时限。
#   decision_timeout_seconds：模型改令决定等待时限。
#   replan_hold_seconds：重规划制品等待时限。
#   active_track_sha256：当前路线摘要。
#   params：参考轨迹编译所需控制器限制。
#   semantic_path：净空验证所用语义地图。
#   vehicle_metadata_path：飞行器尺寸与限制。
#   coordinate_contract：原生状态和地图之间的坐标合同。
#   telemetry_args：本次遥测及当前进度上下文。
# 输出：
#   outcome：继续原任务或释放人工接管的状态；路线替换由类型化异常转交主循环。
async def _handle_runtime_interruption(
    *,
    base: ModuleType,
    client: Any,
    frozen_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    control_dir: Path,
    phase: str,
    schedule_index: int | None,
    abort_file: Path,
    rate_hz: float,
    hold_timeout_seconds: float,
    decision_timeout_seconds: float,
    replan_hold_seconds: float,
    active_track_sha256: str,
    params: Any,
    semantic_path: Path | None = None,
    vehicle_metadata_path: Path | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
    telemetry_args: argparse.Namespace | None = None,
) -> str:
    # 在首次刷新悬停前冻结进度；后续读写实时目标不能改写本次改令的路线来源。
    progress = getattr(telemetry_args, "_runtime_track_progress", None)
    if progress is not None:
        if (
            not isinstance(progress, RuntimeTrackProgress)
            or progress.track_sha256 != active_track_sha256
        ):
            raise RuntimeError("RUNTIME_TRACK_PROGRESS_BINDING_INVALID")
        progress = progress.model_copy(deep=True)
    acknowledgement, hold_setpoint = await _stabilize_runtime_hold(
        base=base,
        client=client,
        frozen_setpoint=frozen_setpoint,
        interruption=interruption,
        control_dir=control_dir,
        phase=phase,
        schedule_index=schedule_index,
        abort_file=abort_file,
        rate_hz=rate_hz,
        timeout_seconds=hold_timeout_seconds,
        telemetry_args=telemetry_args,
        coordinate_contract=coordinate_contract,
        track_progress=progress,
    )
    decision = await _wait_runtime_decision(
        base=base,
        client=client,
        hold_setpoint=hold_setpoint,
        interruption=interruption,
        acknowledgement=acknowledgement,
        control_dir=control_dir,
        abort_file=abort_file,
        rate_hz=rate_hz,
        timeout_seconds=decision_timeout_seconds,
        telemetry_args=telemetry_args,
        coordinate_contract=coordinate_contract,
    )
    processed_path = control_dir / "processed" / interruption.claimed_path.name
    transfer_runtime_file(interruption.claimed_path, processed_path)
    if decision.authorized_action == "land":
        raise UserDirectedLanding(
            f"runtime user message requested landing: {interruption.message.message_id}"
        )
    if decision.authorized_action == "apply_command":
        command = await _wait_runtime_command(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            interruption=interruption,
            acknowledgement=acknowledgement,
            decision=decision,
            control_dir=control_dir,
            abort_file=abort_file,
            rate_hz=rate_hz,
            timeout_seconds=decision_timeout_seconds,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        return await _execute_runtime_command(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            interruption=interruption,
            command=command,
            control_dir=control_dir,
            abort_file=abort_file,
            rate_hz=rate_hz,
            timeout_seconds=decision_timeout_seconds,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
    if decision.authorized_action == "hold_for_replan":
        _atomic_json(
            control_dir / "replan-required.json",
            {
                "message_id": interruption.message.message_id,
                "message_sha256": sha256_json(interruption.message),
                "hold_ack_sha256": sha256_json(acknowledgement),
                "decision_sha256": sha256_json(decision),
                "old_plan_resume_authorized": False,
                "required_artifact": "new code-validated plan revision and replacement track",
            },
        )
        replacement = await _wait_runtime_replacement(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            interruption=interruption,
            acknowledgement=acknowledgement,
            decision=decision,
            control_dir=control_dir,
            abort_file=abort_file,
            rate_hz=rate_hz,
            timeout_seconds=replan_hold_seconds,
            active_track_sha256=active_track_sha256,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        schedule, waypoint_arrival_indices = _compile_replacement_schedule(
            base=base,
            replacement=replacement,
            params=params,
            rate_hz=rate_hz,
        )
        source_schedule_setpoints = len(schedule)
        heading_policy = getattr(client, "_heading_policy", "measured-hold")
        if heading_policy == "route-tangent-relative":
            world_name = getattr(client, "_world_name", None)
            gazebo_vehicle_model_name = getattr(
                client,
                "_gazebo_vehicle_model_name",
                None,
            )
            if not world_name or not gazebo_vehicle_model_name:
                raise RuntimeError("runtime replacement heading alignment has no Gazebo identity")
            measured_heading_deg = await base._await_with_setpoint_keepalive(
                client,
                client.sample_heading_deg(min(2.0, replan_hold_seconds)),
                hold_setpoint=hold_setpoint,
                rate_hz=rate_hz,
            )
            gazebo_pose = await base._await_with_setpoint_keepalive(
                client,
                client.sample_gazebo_model_pose(
                    world_name=world_name,
                    model_name=gazebo_vehicle_model_name,
                    timeout_seconds=min(2.0, replan_hold_seconds),
                ),
                hold_setpoint=hold_setpoint,
                rate_hz=rate_hz,
            )
            aligned_plan = base.align_setpoint_schedule_to_route_tangent(
                base.SetpointSchedulePlan(
                    schedule=schedule,
                    track_start_index=0,
                    track_end_index=len(schedule) - 1,
                    waypoint_arrival_indices=waypoint_arrival_indices,
                ),
                measured_px4_heading_deg=measured_heading_deg,
                measured_body_heading_ned_deg=base.gazebo_body_heading_ned_deg(gazebo_pose),
                rate_hz=rate_hz,
                maximum_yaw_rate_deg_s=getattr(
                    client,
                    "_maximum_yaw_rate_deg_s",
                    20.0,
                ),
            )
            schedule = aligned_plan.schedule
            waypoint_arrival_indices = aligned_plan.waypoint_arrival_indices
        track_sha256 = sha256_json(replacement.track)
        adoption = {
            "message_id": interruption.message.message_id,
            "execution_id": interruption.message.execution_id,
            "replacement_sequence": replacement.replacement_sequence,
            "replacement_sha256": sha256_json(replacement),
            "track_sha256": track_sha256,
            "task_graph_sha256": (
                sha256_json(replacement.revised_task_graph)
                if replacement.revised_task_graph is not None
                else None
            ),
            "runtime_checkpoints_sha256": (
                sha256_json(replacement.runtime_checkpoints)
                if replacement.runtime_checkpoints is not None
                else None
            ),
            "runtime_actions_sha256": (
                sha256_json(replacement.runtime_actions)
                if replacement.runtime_actions is not None
                else None
            ),
            "schedule_setpoints": len(schedule),
            "source_schedule_setpoints": source_schedule_setpoints,
            "heading_policy": heading_policy,
            "adopted_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(
            control_dir / "adoptions" / f"{interruption.message.message_id}.json",
            adoption,
        )
        _atomic_json(control_dir / "active-track.json", adoption)
        _atomic_json(
            control_dir / "side-effects.state.json",
            {
                "enabled": True,
                "execution_id": interruption.message.execution_id,
                "message_id": interruption.message.message_id,
                "reason": "code-validated runtime replacement track adopted",
                "replacement_sha256": sha256_json(replacement),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        raise RuntimeTrackReplacement(
            message_id=interruption.message.message_id,
            schedule=schedule,
            waypoint_arrival_indices=waypoint_arrival_indices,
            track_sha256=track_sha256,
            replacement_sequence=replacement.replacement_sequence,
            amendment_action=replacement.amendment_action,
            amendment_parameters=dict(replacement.amendment_parameters),
            coordinate_contract=replacement.track.coordinate_contract,
            artifact=replacement,
        )
    if decision.authorized_action == "hold":
        if decision.classification.requested_action == "operator_takeover":
            if (
                semantic_path is None
                or vehicle_metadata_path is None
                or coordinate_contract is None
            ):
                raise UserDirectedLanding(
                    "operator takeover is unavailable without collision and vehicle contracts"
                )
            grant = await _wait_operator_takeover_grant(
                base=base,
                client=client,
                hold_setpoint=hold_setpoint,
                interruption=interruption,
                acknowledgement=acknowledgement,
                decision=decision,
                control_dir=control_dir,
                abort_file=abort_file,
                rate_hz=rate_hz,
                timeout_seconds=replan_hold_seconds,
                telemetry_args=telemetry_args,
                coordinate_contract=coordinate_contract,
            )
            return await _run_operator_takeover(
                base=base,
                client=client,
                hold_setpoint=hold_setpoint,
                interruption=interruption,
                grant=grant,
                control_dir=control_dir,
                abort_file=abort_file,
                rate_hz=rate_hz,
                runtime_session=_runtime_session(control_dir),
                active_track_sha256=active_track_sha256,
                params=params,
                hold_timeout_seconds=hold_timeout_seconds,
                decision_timeout_seconds=decision_timeout_seconds,
                replan_hold_seconds=replan_hold_seconds,
                semantic_path=semantic_path,
                vehicle_metadata_path=vehicle_metadata_path,
                coordinate_contract=coordinate_contract,
                telemetry_args=telemetry_args,
            )
        _atomic_json(
            control_dir / "side-effects.state.json",
            {
                "enabled": False,
                "execution_id": interruption.message.execution_id,
                "message_id": interruption.message.message_id,
                "reason": "code-authorized persistent stable hold",
                "decision_sha256": sha256_json(decision),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        session = _runtime_session(control_dir)
        hold_deadline = time.monotonic() + replan_hold_seconds
        while time.monotonic() < hold_deadline:
            await _runtime_hold_tick(
                base=base,
                client=client,
                hold_setpoint=hold_setpoint,
                abort_file=abort_file,
                rate_hz=rate_hz,
                telemetry_args=telemetry_args,
                coordinate_contract=coordinate_contract,
            )
            next_interruption = _claim_runtime_message(control_dir, session)
            if next_interruption is not None:
                return await _handle_runtime_interruption(
                    base=base,
                    client=client,
                    frozen_setpoint=hold_setpoint,
                    interruption=next_interruption,
                    control_dir=control_dir,
                    phase="PAUSED",
                    schedule_index=schedule_index,
                    abort_file=abort_file,
                    rate_hz=rate_hz,
                    hold_timeout_seconds=hold_timeout_seconds,
                    decision_timeout_seconds=decision_timeout_seconds,
                    replan_hold_seconds=replan_hold_seconds,
                    active_track_sha256=active_track_sha256,
                    params=params,
                    semantic_path=semantic_path,
                    vehicle_metadata_path=vehicle_metadata_path,
                    coordinate_contract=coordinate_contract,
                    telemetry_args=telemetry_args,
                )
            await asyncio.sleep(1.0 / rate_hz)
        raise UserDirectedLanding("runtime pause exceeded the bounded safe-hold window")
    operator_released = decision.classification.requested_action == "operator_release"
    _atomic_json(
        control_dir / "side-effects.state.json",
        {
            "enabled": True,
            "execution_id": interruption.message.execution_id,
            "message_id": interruption.message.message_id,
            "reason": (
                "operator authority released to the prepared mission"
                if operator_released
                else "code-authorized informational resume"
            ),
            "decision_sha256": sha256_json(decision),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return "release_operator_control" if operator_released else "resume_original"


# 功能：
#   在悬停中等待人工接管授权，复核消息、决定、稳定回执及有效期，不凭接管按钮直接放行。
# 输入：
#   base：飞控基础模块。
#   client：实际飞控客户端。
#   hold_setpoint：接管前稳定位置。
#   interruption：当前用户消息。
#   acknowledgement：悬停稳定回执。
#   decision：允许人工接管的决定。
#   control_dir：授权制品目录。
#   abort_file：外部终止信号。
#   rate_hz：保护维持频率。
#   timeout_seconds：等待授权时限。
#   telemetry_args：原生遥测配置。
#   coordinate_contract：坐标绑定。
# 输出：
#   grant：摘要与有效期检查通过的有界人工授权。
async def _wait_operator_takeover_grant(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    acknowledgement: RuntimeHoldAcknowledgement,
    decision: RuntimeInterruptionDecision,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    telemetry_args: argparse.Namespace | None = None,
    coordinate_contract: Px4CoordinateContract | None = None,
) -> RuntimeOperatorTakeoverGrant:
    path = control_dir / "takeover-grants" / f"{interruption.message.message_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        await _runtime_hold_tick(
            base=base,
            client=client,
            hold_setpoint=hold_setpoint,
            abort_file=abort_file,
            rate_hz=rate_hz,
            telemetry_args=telemetry_args,
            coordinate_contract=coordinate_contract,
        )
        if path.is_file():
            grant = RuntimeOperatorTakeoverGrant.model_validate(read_runtime_object(path))
            gates = {
                "message_id": grant.message_id == interruption.message.message_id,
                "execution_id": grant.execution_id == interruption.message.execution_id,
                "message_hash": grant.message_sha256 == sha256_json(interruption.message),
                "hold_hash": grant.hold_ack_sha256 == sha256_json(acknowledgement),
                "decision_hash": grant.decision_sha256 == sha256_json(decision),
                "grant_gates": all(grant.deterministic_gates.values()),
                "not_expired": datetime.now(UTC) < grant.expires_at,
            }
            if not all(gates.values()):
                failed = ",".join(name for name, accepted in gates.items() if not accepted)
                raise UserDirectedLanding(f"operator takeover grant binding failed: {failed}")
            return grant
        await asyncio.sleep(1.0 / rate_hz)
    raise UserDirectedLanding("operator takeover grant was not supplied during stable hold")


# 功能：
#   按短时人工速度与偏航命令逐步更新控制，检查序号、授权限制及地图净空，并保留实际采纳证据。
# 输入：
#   base：飞控基础类型和保护函数。
#   client：实际飞控客户端。
#   hold_setpoint：接管起始稳定位置。
#   interruption：触发接管的用户消息。
#   grant：有时限和速度限制的人工授权。
#   control_dir：人工命令、采纳和改令证据目录。
#   abort_file：外部终止信号。
#   rate_hz：人工指令插值执行频率。
#   runtime_session：当前运行会话。
#   active_track_sha256：暂停中的原路线摘要。
#   params：恢复或替换路线所用控制器限制。
#   hold_timeout_seconds：新改令的稳定悬停时限。
#   decision_timeout_seconds：新决定等待时限。
#   replan_hold_seconds：替换轨迹等待时限。
#   semantic_path：局部净空验证地图。
#   vehicle_metadata_path：飞行器尺寸与速度限制。
#   coordinate_contract：地图和控制坐标绑定。
#   telemetry_args：遥测与执行进度上下文。
# 输出：
#   outcome：释放接管后上层应采取的任务状态。
async def _run_operator_takeover(
    *,
    base: ModuleType,
    client: Any,
    hold_setpoint: Any,
    interruption: RuntimeInterruptDetected,
    grant: RuntimeOperatorTakeoverGrant,
    control_dir: Path,
    abort_file: Path,
    rate_hz: float,
    runtime_session: RuntimeControlSession | None,
    active_track_sha256: str,
    params: Any,
    hold_timeout_seconds: float,
    decision_timeout_seconds: float,
    replan_hold_seconds: float,
    semantic_path: Path,
    vehicle_metadata_path: Path,
    coordinate_contract: Px4CoordinateContract,
    telemetry_args: argparse.Namespace | None = None,
) -> str:
    vehicle = VehicleAsset.model_validate(read_runtime_object(vehicle_metadata_path))
    current_setpoint = hold_setpoint
    current_world = _setpoint_world_enu(current_setpoint, coordinate_contract)
    grant_hash = sha256_json(grant)
    next_sequence = 1
    commands: list[dict[str, Any]] = []
    evidence_path = control_dir / "takeover-evidence" / f"{grant.message_id}.json"
    evidence: dict[str, Any] = {
        "schema_version": "dronedream.runtime-operator-takeover-evidence.v1",
        "message_id": grant.message_id,
        "execution_id": grant.execution_id,
        "operator_id": grant.operator_id,
        "grant_sha256": grant_hash,
        "status": "active",
        "started_at": datetime.now(UTC).isoformat(),
        "commands": commands,
    }
    _atomic_json(evidence_path, evidence)
    _atomic_json(
        control_dir / "takeover-adoptions" / f"{grant.message_id}.json",
        RuntimeOperatorTakeoverAdoption(
            message_id=grant.message_id,
            execution_id=grant.execution_id,
            grant_sha256=grant_hash,
            adopted_at=datetime.now(UTC),
        ),
    )
    command_dir = control_dir / "operator-commands"
    processed_dir = control_dir / "processed-operator-commands"
    processed_dir.mkdir(parents=True, exist_ok=True)
    period = 1.0 / rate_hz
    try:
        while datetime.now(UTC) < grant.expires_at:
            base._raise_if_external_abort_requested(abort_file)
            new_interruption = _claim_runtime_message(control_dir, runtime_session)
            if new_interruption is not None:
                outcome = await _handle_runtime_interruption(
                    base=base,
                    client=client,
                    frozen_setpoint=current_setpoint,
                    interruption=new_interruption,
                    control_dir=control_dir,
                    phase="TRACK",
                    schedule_index=None,
                    abort_file=abort_file,
                    rate_hz=rate_hz,
                    hold_timeout_seconds=hold_timeout_seconds,
                    decision_timeout_seconds=decision_timeout_seconds,
                    replan_hold_seconds=replan_hold_seconds,
                    active_track_sha256=active_track_sha256,
                    params=params,
                    semantic_path=semantic_path,
                    vehicle_metadata_path=vehicle_metadata_path,
                    coordinate_contract=coordinate_contract,
                    telemetry_args=telemetry_args,
                )
                if outcome == "release_operator_control":
                    evidence["status"] = "released"
                    evidence["ended_at"] = datetime.now(UTC).isoformat()
                    evidence["outcome"] = "operator released control to prepared mission"
                    _atomic_json(evidence_path, evidence)
                    return "resume_original"
                if outcome != "resume_original":
                    return outcome
            command_path = command_dir / f"{next_sequence:08d}.json"
            if not command_path.is_file():
                await _runtime_hold_tick(
                    base=base,
                    client=client,
                    hold_setpoint=current_setpoint,
                    abort_file=abort_file,
                    rate_hz=rate_hz,
                    telemetry_args=telemetry_args,
                    coordinate_contract=coordinate_contract,
                )
                await asyncio.sleep(period)
                continue
            command = RuntimeOperatorControlCommand.model_validate(
                read_runtime_object(command_path)
            )
            gates = {
                "message_id": command.message_id == grant.message_id,
                "execution_id": command.execution_id == grant.execution_id,
                "grant_hash": command.grant_sha256 == grant_hash,
                "sequence": command.sequence == next_sequence,
                "fresh": 0.0 <= (datetime.now(UTC) - command.issued_at).total_seconds() <= 2.0,
                "horizontal_speed": math.hypot(
                    command.velocity_ned_mps.x, command.velocity_ned_mps.y
                )
                <= grant.maximum_horizontal_speed_mps,
                "vertical_speed": (
                    abs(command.velocity_ned_mps.z) <= grant.maximum_vertical_speed_mps
                ),
                "yaw_rate": abs(command.yaw_rate_dps) <= grant.maximum_yaw_rate_dps,
            }
            if not all(gates.values()):
                failed = ",".join(name for name, accepted in gates.items() if not accepted)
                raise UserDirectedLanding(f"operator control command rejected: {failed}")
            if command.action == "release":
                transfer_runtime_file(command_path, processed_dir / command_path.name)
                commands.append(
                    {
                        "sequence": command.sequence,
                        "action": command.action,
                        "command_sha256": sha256_json(command),
                        "outcome": "controlled_landing",
                    }
                )
                raise UserDirectedLanding("operator released takeover; controlled landing required")
            steps = max(1, math.ceil(command.duration_seconds * rate_hz))
            step_seconds = command.duration_seconds / steps
            for _ in range(steps):
                base._raise_if_external_abort_requested(abort_file)
                tick_time = datetime.now(UTC)
                if (
                    tick_time >= grant.expires_at
                    or not 0 <= (tick_time - command.issued_at).total_seconds() <= 2.0
                ):
                    raise UserDirectedLanding(
                        "operator control authorization expired before movement"
                    )
                desired_setpoint = base.Setpoint(
                    north_m=current_setpoint.north_m + command.velocity_ned_mps.x * step_seconds,
                    east_m=current_setpoint.east_m + command.velocity_ned_mps.y * step_seconds,
                    down_m=current_setpoint.down_m + command.velocity_ned_mps.z * step_seconds,
                    yaw_deg=(current_setpoint.yaw_deg + command.yaw_rate_dps * step_seconds)
                    % 360.0,
                )
                desired_world = _setpoint_world_enu(desired_setpoint, coordinate_contract)
                route = GraphRoute(
                    start_node="runtime-operator-current",
                    goal_node="runtime-operator-command",
                    node_ids=["runtime-operator-current", "runtime-operator-command"],
                    edge_ids=["runtime-operator-segment"],
                    positions_m=[current_world, desired_world],
                    route_length_m=math.dist(
                        (current_world.x, current_world.y, current_world.z),
                        (desired_world.x, desired_world.y, desired_world.z),
                    ),
                    all_edges_flight_verified=False,
                )
                clearance = validate_route_clearance(
                    route,
                    semantic_path,
                    vehicle_diameter_m=vehicle.body_radius_m * 2.0,
                    vehicle_height_m=vehicle.body_height_m,
                )
                if not clearance.accepted:
                    raise UserDirectedLanding(
                        "operator control clearance gate rejected the next command segment"
                    )
                # 净空计算也消耗实际时间；在设备 I/O 前再次核对，不能靠进入循环时的许可续飞。
                base._raise_if_external_abort_requested(abort_file)
                send_time = datetime.now(UTC)
                if (
                    send_time >= grant.expires_at
                    or not 0 <= (send_time - command.issued_at).total_seconds() <= 2.0
                ):
                    raise UserDirectedLanding(
                        "operator control authorization expired during clearance"
                    )
                current_setpoint = desired_setpoint
                current_world = desired_world
                await client.set_position_ned(current_setpoint)
                await asyncio.sleep(step_seconds)
            transfer_runtime_file(command_path, processed_dir / command_path.name)
            commands.append(
                {
                    "sequence": command.sequence,
                    "action": command.action,
                    "command_sha256": sha256_json(command),
                    "final_world_enu_m": current_world.model_dump(mode="json"),
                    "deterministic_gates": gates,
                }
            )
            next_sequence += 1
            _atomic_json(evidence_path, evidence)
        raise UserDirectedLanding("operator takeover grant expired; controlled landing required")
    except BaseException as exc:
        evidence["status"] = "released" if "released takeover" in str(exc) else "failed"
        evidence["ended_at"] = datetime.now(UTC).isoformat()
        evidence["outcome"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(evidence_path, evidence)
        raise


# 功能：
#   1. 维持控制并等待连续满足精确位置和速度门槛，以实测收敛延长停滞窗口。
#   2. 总恢复上限不随进展延长；局部模型目标不替代最终航点稳定判定。
# 输入：
#   base：基础飞控和异步控制维持工具。
#   client：实际遥测客户端。
#   setpoint：必须达到的稳定目标。
#   rate_hz：等待期间的控制频率。
#   timeout_seconds：无有效收敛时的等待预算。
#   absolute_timeout_factor：不可延长总预算相对普通预算的倍数。
#   stable_window_seconds：连续稳定时间。
#   position_tolerance_m：最大位置误差，单位米。
#   speed_tolerance_mps：最大速度，单位米每秒。
#   abort_check：外部终止检查回调。
#   runtime_interrupt_probe：可选改令探针。
#   setpoint_refresh：局部安全／模型控制刷新回调。
#   sample_observer：实际遥测发布回调。
#   target_frame_position_resolver：将实测位置换算到目标坐标的回调。
# 输出：
#   stable_result：通过连续稳定检查的样本、位置误差和速度三元组。
async def _wait_checkpoint_stable(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    rate_hz: float,
    timeout_seconds: float = 12.0,
    absolute_timeout_factor: float = 5.0,
    stable_window_seconds: float = 1.0,
    position_tolerance_m: float = 0.75,
    speed_tolerance_mps: float = 0.5,
    abort_check: Callable[[], None] | None = None,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None = None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    target_frame_position_resolver: (Callable[[Any], tuple[float, float, float]] | None) = None,
) -> tuple[Any, float, float]:
    for value, label in (
        (rate_hz, "settle control rate"),
        (timeout_seconds, "settle timeout"),
        (absolute_timeout_factor, "settle absolute timeout factor"),
        (stable_window_seconds, "settle stable window"),
        (position_tolerance_m, "settle position tolerance"),
        (speed_tolerance_mps, "settle speed tolerance"),
    ):
        if not finite_positive_number(value):
            raise ValueError(f"{label} must be finite and positive")
    if absolute_timeout_factor < 1.0:
        raise ValueError("settle absolute timeout factor must be at least one")
    started = time.monotonic()
    deadline = started + timeout_seconds
    # A model-authorized local path may deliberately move away from the signed
    # waypoint before rejoining it (for example around a roof edge or stair
    # rail). Keep a finite hard bound, but leave enough time for one such local
    # detour at the qualified recovery speed.
    absolute_deadline = started + timeout_seconds * absolute_timeout_factor
    progress_epsilon_m = 0.01
    best_position_error_m = math.inf
    best_hold_error_m = math.inf
    last_hold_target: tuple[float, float, float] | None = None
    # A sample can first satisfy every gate immediately before the signed
    # settle deadline. Give only that already-qualified sample enough bounded
    # time to prove the required continuous stable window. This does not relax
    # position or speed tolerances and cannot rescue a vehicle that never
    # enters both gates before the configured timeout.
    qualified_window_deadline = absolute_deadline + stable_window_seconds + 1.0 / rate_hz
    if not all(math.isfinite(value) for value in (started, deadline, qualified_window_deadline)):
        raise ValueError("settle deadline must be representable")
    stable_since: float | None = None
    latest: Any | None = None
    latest_error = math.inf
    latest_speed = math.inf

    # 功能：
    #   在持续采样中同时检查终止与改令，不因传入一类检查而吞掉另一类信号。
    # 输入：
    #   无显式参数；使用外层 abort_check 和 runtime_interrupt_probe。
    # 输出：
    #   None：不返回业务数据。
    def check_settle_interruption() -> None:
        if abort_check is not None:
            abort_check()
        if runtime_interrupt_probe is not None:
            interruption = runtime_interrupt_probe()
            if interruption is not None:
                raise interruption

    hold_setpoint = setpoint

    # 功能：
    #   在位置样本尚未到达时仍重新仲裁动作，并保存实际目标用于本轮跟踪误差计算。
    # 输入：
    #   planned_setpoint：基础等待器传入的检查点目标。
    # 输出：
    #   hold_setpoint：本次实际发送的本地控制目标。
    async def refresh_settle_control(planned_setpoint: Any) -> Any:
        nonlocal hold_setpoint
        hold_setpoint = await setpoint_refresh(planned_setpoint)
        return hold_setpoint

    while time.monotonic() < deadline:
        check_settle_interruption()
        latest = await base._await_with_setpoint_keepalive(
            client,
            client.sample_position_velocity_ned(1.0),
            hold_setpoint=setpoint,
            rate_hz=rate_hz,
            abort_check=check_settle_interruption,
            **(
                {"setpoint_refresh": refresh_settle_control} if setpoint_refresh is not None else {}
            ),
        )
        if sample_observer is not None:
            sample_observer(latest)
        target_frame_position = (
            target_frame_position_resolver(latest)
            if target_frame_position_resolver is not None
            else (latest.north_m, latest.east_m, latest.down_m)
        )
        if len(target_frame_position) != 3 or not all(
            math.isfinite(value) for value in target_frame_position
        ):
            raise RuntimeError("settle target-frame position is invalid")
        latest_error = math.dist(
            target_frame_position,
            (setpoint.north_m, setpoint.east_m, setpoint.down_m),
        )
        latest_speed = math.hypot(latest.north_m_s, latest.east_m_s, latest.down_m_s)
        now = time.monotonic()
        # ``timeout_seconds`` is a no-progress deadline, not a punishment for
        # a deliberately slow vertical settle.  Only a material decrease in
        # measured PX4 distance to the exact signed waypoint refreshes it.
        # The absolute deadline never moves, so noise, circling and asymptotic
        # drift cannot keep the vehicle in Offboard indefinitely.
        if latest_error <= best_position_error_m - progress_epsilon_m:
            best_position_error_m = latest_error
            deadline = min(absolute_deadline, max(deadline, now + timeout_seconds))
        hold_target = (
            hold_setpoint.north_m,
            hold_setpoint.east_m,
            hold_setpoint.down_m,
        )
        if last_hold_target is None or math.dist(last_hold_target, hold_target) > 0.02:
            # A newly authorized controller carrot starts a new, local tracking
            # comparison. It may not satisfy the waypoint gate, but following it
            # is real progress along the model-selected collision-free detour.
            best_hold_error_m = math.inf
            last_hold_target = hold_target
        hold_error_m = math.dist(
            (latest.north_m, latest.east_m, latest.down_m),
            hold_target,
        )
        if hold_error_m <= best_hold_error_m - progress_epsilon_m:
            best_hold_error_m = hold_error_m
            deadline = min(absolute_deadline, max(deadline, now + timeout_seconds))
        if latest_error <= position_tolerance_m and latest_speed <= speed_tolerance_mps:
            stable_since = now if stable_since is None else stable_since
            deadline = min(
                qualified_window_deadline,
                max(deadline, stable_since + stable_window_seconds + 1.0 / rate_hz),
            )
            if now - stable_since >= stable_window_seconds:
                return latest, latest_error, latest_speed
        else:
            stable_since = None
        await asyncio.sleep(1.0 / rate_hz)
    ended_at = time.monotonic()
    timeout_kind = "absolute" if ended_at >= absolute_deadline else "no-progress"
    raise TimeoutError(
        "waypoint stability timeout: "
        f"position_error_m={latest_error:.3f}, speed_mps={latest_speed:.3f}, "
        f"best_position_error_m={best_position_error_m:.3f}, "
        f"elapsed_seconds={ended_at - started:.3f}, timeout_kind={timeout_kind}"
    )


# 功能：
#   仅在负载接入稳定步骤完成后增加最多五厘米容差，且不额外扩大已经达到二十五厘米的配置。
# 输入：
#   configured_tolerance_m：原本的位置稳定容差。
#   runtime_action_contract：本次完整动作合同。
#   completed_action_step_ids：已经验证完成的动作步骤集合。
# 输出：
#   tolerance_m：当前负载状态允许的位置稳定容差。
def _payload_aware_waypoint_position_tolerance_m(
    *,
    configured_tolerance_m: float,
    runtime_action_contract: RuntimeActionExecutionContract | None,
    completed_action_step_ids: set[str],
) -> float:
    """Return the bounded settle radius after verified payload custody.

    A suspended payload produces small, real position oscillations near a
    waypoint even after the speed gate is satisfied.  The ordinary 0.20 m
    radius remains authoritative before pickup.  Once the post-attachment
    stability action has completed, allow at most another 0.05 m while keeping
    the independent speed and continuous-stability-window gates unchanged.
    """

    if not finite_positive_number(configured_tolerance_m):
        raise ValueError("payload settle tolerance must be finite and positive")
    if runtime_action_contract is None:
        return configured_tolerance_m

    payload_stability_completed = any(
        step.step_id in completed_action_step_ids
        and step.driver == "payload-transition"
        and step.parameters.get("operation") == "postattach-stability"
        for step in runtime_action_contract.steps
    )
    if not payload_stability_completed:
        return configured_tolerance_m
    return max(configured_tolerance_m, min(configured_tolerance_m + 0.05, 0.25))


# 功能：
#   在保持悬停与位置遥测更新的同时重试电池读取，不能用默认电量替代缺失样本。
# 输入：
#   base：异步维持控制的基础工具。
#   client：实际电池与位置遥测客户端。
#   setpoint：等待读数期间的保护目标。
#   rate_hz：保护控制频率。
#   timeout_seconds：电池读取的总时限。
#   sample_timeout_seconds：每次新读取的最大预算。
#   runtime_interrupt_probe：可选用户改令探针。
#   sample_observer：实际位置样本发布回调。
#   setpoint_refresh：每次维持控制时重新仲裁目标的回调。
#   safety_abort_check：外部终止文件等额外检查。
# 输出：
#   battery：实际读取的剩余电量和电压字典。
async def _sample_checkpoint_battery(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    rate_hz: float,
    timeout_seconds: float = 12.0,
    sample_timeout_seconds: float = 3.0,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    safety_abort_check: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Read real battery telemetry without starving the Offboard heartbeat.

    MAVSDK telemetry subscriptions can occasionally take longer than one short
    sample window to produce their first item. A checkpoint must not convert
    that transport hiccup into an uncontrolled descent, but it also must never
    invent or reuse battery state. Retry fresh, bounded subscriptions while
    continuously commanding the verified hover setpoint; fail closed if no
    real sample arrives before the overall deadline.
    """

    if not finite_positive_number(rate_hz):
        raise ValueError("checkpoint battery control rate must be finite and positive")
    if not finite_positive_number(timeout_seconds):
        raise ValueError("checkpoint battery timeout must be finite and positive")
    if not finite_positive_number(sample_timeout_seconds):
        raise ValueError("checkpoint battery sample timeout must be finite and positive")

    # 功能：
    #   电池等待期间把已检测用户改令抛回上层，不吞掉人工干预。
    # 输入：
    #   无显式参数；使用外层 runtime_interrupt_probe。
    # 输出：
    #   None：不返回业务数据。
    def abort_check() -> None:
        if safety_abort_check is not None:
            safety_abort_check()
        if runtime_interrupt_probe is None:
            return
        interruption = runtime_interrupt_probe()
        if interruption is not None:
            raise interruption

    heartbeat_stop = asyncio.Event()

    # 功能：
    #   电池读取期间持续发布真实位置样本，收到退出信号后停止自身采样。
    # 输入：
    #   无显式参数；使用外层客户端、发布回调与退出事件。
    # 输出：
    #   None：不返回业务数据。
    async def identity_heartbeat() -> None:
        """Keep the independent PX4/Gazebo identity proof fresh during battery I/O."""

        if sample_observer is None:
            return
        position_timeout = min(0.5, sample_timeout_seconds)
        while not heartbeat_stop.is_set():
            try:
                observed = await asyncio.wait_for(
                    client.sample_position_velocity_ned(position_timeout),
                    position_timeout,
                )
            except TimeoutError:
                pass
            else:
                sample_observer(observed)
            await asyncio.sleep(min(0.05, 1.0 / rate_hz))

    heartbeat_task = (
        asyncio.create_task(identity_heartbeat()) if sample_observer is not None else None
    )
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    try:
        if not math.isfinite(deadline):
            raise ValueError("checkpoint battery deadline must be representable")
        while True:
            abort_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            attempts += 1
            try:
                sample = await base._await_with_setpoint_keepalive(
                    client,
                    asyncio.wait_for(
                        client.sample_battery(min(sample_timeout_seconds, remaining)),
                        min(sample_timeout_seconds, remaining),
                    ),
                    hold_setpoint=setpoint,
                    rate_hz=rate_hz,
                    abort_check=abort_check,
                    **(
                        {"setpoint_refresh": setpoint_refresh}
                        if setpoint_refresh is not None
                        else {}
                    ),
                )
            except TimeoutError:
                continue
            if time.monotonic() >= deadline:
                break
            if not isinstance(sample, dict):
                raise ValueError("PX4 checkpoint battery telemetry must be an object")
            remaining_percent = sample.get("remaining_percent")
            voltage_v = sample.get("voltage_v")
            if (
                type(remaining_percent) not in (int, float)
                or not 0 <= remaining_percent <= 100
                or type(voltage_v) not in (int, float)
                or not 0 <= voltage_v <= 1000
            ):
                raise ValueError("PX4 checkpoint battery telemetry is invalid")
            return {
                "remaining_percent": float(remaining_percent),
                "voltage_v": float(voltage_v),
            }
    finally:
        primary_failure = sys.exc_info()[1]
        heartbeat_stop.set()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                if primary_failure is None:
                    raise
                primary_failure.add_note(f"Battery identity cleanup: {type(error).__name__}")
    raise TimeoutError(
        "PX4 checkpoint battery telemetry timeout after "
        f"{timeout_seconds:g}s across {attempts} fresh sample attempts"
    )


# 功能：
#   解析 ROS 服务实际响应中的确认、证据及错误字段，不把进程零退出码当作设备确认。
# 输入：
#   output：捕获器限长后的 ROS 命令行输出。
# 输出：
#   result：设备确认、证据列表和原始响应摘要组成的字典。
def _parse_ros2_domain_action_response(output: str) -> dict[str, Any]:
    # ros2 service call 会先回显 request。只解析独立 response 行后的构造表示，
    # 不在请求文本、证据字符串或 details_json 内搜索 success=true。
    response_marker = re.search(r"(?m)^response:\s*", output)
    if response_marker is None or len(output) > 1024 * 1024:
        raise RuntimeError("ROS 2 domain action omitted a bounded response")
    try:
        expression = ast.parse(output[response_marker.end() :].strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise RuntimeError("ROS 2 domain action response is malformed") from exc
    if (
        not isinstance(expression, ast.Call)
        or expression.args
        or not isinstance(expression.func, (ast.Name, ast.Attribute))
    ):
        raise RuntimeError("ROS 2 domain action response is not a service result")
    fields = {keyword.arg: keyword.value for keyword in expression.keywords}
    if None in fields or len(fields) != len(expression.keywords):
        raise RuntimeError("ROS 2 domain action response fields are ambiguous")
    success_value = fields.get("success")
    if isinstance(success_value, ast.Constant) and type(success_value.value) is bool:
        confirmed = success_value.value
    elif isinstance(success_value, ast.Name) and success_value.id in {"true", "false"}:
        confirmed = success_value.id == "true"
    else:
        raise RuntimeError("ROS 2 domain action response omitted the success field")
    try:
        evidence = ast.literal_eval(fields["evidence"]) if "evidence" in fields else []
        issue = ast.literal_eval(fields["issue_code"]) if "issue_code" in fields else ""
        details = ast.literal_eval(fields["details_json"]) if "details_json" in fields else "{}"
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise RuntimeError("ROS 2 domain action response contains non-literal fields") from exc
    if (
        not isinstance(evidence, list)
        or len(evidence) > 256
        or not all(isinstance(item, str) for item in evidence)
        or not isinstance(issue, str)
        or not isinstance(details, str)
    ):
        raise RuntimeError("ROS 2 domain action response has invalid evidence fields")
    result = {
        "confirmed": confirmed,
        "transport": "ros2-service",
        "evidence": evidence,
        "issue_code": issue,
        "details_json": details,
        "raw_response": output[-8_000:],
    }
    return result


# 功能：
#   保留宿主环境并补齐当前 ROS 发行版及工作空间的 Python 包路径，支持虚拟环境调用。
# 输入：
#   无显式参数；读取当前进程的 ROS 环境和 Python 版本。
# 输出：
#   env：供自有 ROS 子进程使用的独立环境字典。
def _ros2_cli_environment() -> dict[str, str]:
    """Preserve ROS 2 Python metadata when the executor runs from its venv."""

    env = os.environ.copy()
    ros_distro = env.get("ROS_DISTRO", "jazzy")
    python_path = [value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value]
    python_suffix = f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    ros_prefixes = [value for value in env.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if value]
    ros_prefixes.append(f"/opt/ros/{ros_distro}")
    for prefix in ros_prefixes:
        ros_python_packages = f"{prefix.rstrip('/')}/{python_suffix}"
        if ros_python_packages not in python_path:
            python_path.append(ros_python_packages)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


# 功能：
#   1. 以有界输出、总时限和自有进程清理调用已授权 ROS 2 服务，不通过 shell 拼接命令。
#   2. 取消时通知工作线程并等它回收进程，不能把协程取消误认为服务端动作已经撤销。
# 输入：
#   parameters：已授权服务名、类型和结构化请求。
#   timeout_seconds：本次服务调用的最长运行秒数。
# 输出：
#   result：包含实际服务确认、证据字段及有界原始输出的字典。
async def _execute_ros2_domain_action(
    parameters: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ROS 2 executable 'ros2' is unavailable")
    service_name = str(parameters["service_name"])
    service_type = str(parameters["service_type"])
    request = parameters["request"]
    if not isinstance(request, dict):
        raise RuntimeError("ROS 2 domain action request must be an object")
    # 继续兼容不支持 asyncio 子进程传输的事件循环，但线程内使用可取消的共享执行器。
    command = [
        ros2,
        "service",
        "call",
        service_name,
        service_type,
        json.dumps(
            request, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
        ),
    ]

    cancel_event = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            capture_process,
            command,
            stdin=b"",
            maximum_bytes=1024 * 1024,
            timeout=timeout_seconds,
            environment=_ros2_cli_environment(),
            resource_policy=PluginResourcePolicy(memory_limit_mb=512, process_limit=8),
            cancel_event=cancel_event,
        )
    )
    try:
        completed = await asyncio.shield(worker)
    except BaseException:
        cancel_event.set()
        # shield 防止取消仅切断 Future 而丢失工作线程；重复取消也不能跳过自有进程回收。
        while not worker.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(worker)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            worker.result()
        raise
    rendered = (completed.stdout + completed.stderr).decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(f"ROS 2 domain action service failed: {rendered[-2_000:]}")
    result = _parse_ros2_domain_action_response(rendered)
    return result


# 功能：
#   按明确驱动分派相机、载荷或 ROS 动作，装卸阶段分别核对接触前、接入后和负载稳定条件。
# 输入：
#   base：控制维持基础工具。
#   client：实际飞控和设备客户端。
#   step：包含驱动类型、参数及权限的动作步骤。
#   setpoint：动作期间保持的目标。
#   rate_hz：控制维持频率。
#   runtime_interrupt_probe：用户改令探针。
#   setpoint_refresh：局部控制刷新回调。
#   sample_observer：实际遥测观察回调。
#   target_frame_position_resolver：位置稳定判定的坐标变换。
# 输出：
#   output：驱动实际返回的设备状态和验证证据。
async def _invoke_runtime_action_driver(
    *,
    base: ModuleType,
    client: Any,
    step: RuntimeActionExecutionStep,
    setpoint: Any,
    rate_hz: float,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    target_frame_position_resolver: (Callable[[Any], tuple[float, float, float]] | None) = None,
) -> dict[str, Any]:
    if step.driver == "mavsdk-camera":
        simulation_capture = _verified_simulation_onboard_rgb_capture(step.parameters)
        if simulation_capture is not None:
            return simulation_capture
        return await client.execute_camera_command(step.parameters)
    if step.driver == "gazebo-payload":
        return await client.execute_payload_command(step.parameters)
    if step.driver == "payload-transition":
        operation = str(step.parameters["operation"])
        if operation == "precontact":
            _, position_error_m, speed_mps = await _wait_checkpoint_stable(
                base=base,
                client=client,
                setpoint=setpoint,
                rate_hz=rate_hz,
                timeout_seconds=float(step.parameters["settle_timeout_seconds"]),
                stable_window_seconds=float(step.parameters["stable_window_seconds"]),
                position_tolerance_m=float(step.parameters["position_tolerance_m"]),
                speed_tolerance_mps=float(step.parameters["speed_tolerance_mps"]),
                runtime_interrupt_probe=runtime_interrupt_probe,
                setpoint_refresh=setpoint_refresh,
                sample_observer=sample_observer,
                target_frame_position_resolver=target_frame_position_resolver,
            )
            state = await client.sample_payload_state(str(step.parameters["output_topic"]), 3.0)
            if state.get("detached") is not True:
                raise RuntimeError("pre-contact hold found payload contact or attachment")
            return {
                **state,
                "operation": operation,
                "position_error_m": position_error_m,
                "speed_mps": speed_mps,
                "stable_precontact_hover": True,
                "no_payload_contact": True,
            }
        if operation == "confirm-custody":
            state = await client.sample_payload_state(str(step.parameters["output_topic"]), 3.0)
            if state.get("detached") is not False:
                raise RuntimeError("custody confirmation found payload detached")
            physics_bound = (
                bool(step.parameters.get("payload_sdf_sha256"))
                and float(step.parameters.get("payload_mass_kg", 0.0)) > 0.0
            )
            if not physics_bound:
                raise RuntimeError("custody confirmation has no qualified payload physics binding")
            return {
                **state,
                "operation": operation,
                "payload_sdf_sha256": step.parameters["payload_sdf_sha256"],
                "payload_mass_kg": step.parameters["payload_mass_kg"],
                "payload_inertia_kg_m2": step.parameters["payload_inertia_kg_m2"],
                "payload_physics_binding_confirmed": True,
                "custody_state_accepted": True,
            }
        if operation == "postattach-stability":
            state = await client.sample_payload_state(str(step.parameters["output_topic"]), 3.0)
            if state.get("detached") is not False:
                raise RuntimeError("loaded-flight stability gate found payload detached")
            _, position_error_m, speed_mps = await _wait_checkpoint_stable(
                base=base,
                client=client,
                setpoint=setpoint,
                rate_hz=rate_hz,
                timeout_seconds=float(step.parameters["settle_timeout_seconds"]),
                stable_window_seconds=float(step.parameters["stable_window_seconds"]),
                position_tolerance_m=float(step.parameters["position_tolerance_m"]),
                speed_tolerance_mps=float(step.parameters["speed_tolerance_mps"]),
                runtime_interrupt_probe=runtime_interrupt_probe,
                setpoint_refresh=setpoint_refresh,
                sample_observer=sample_observer,
                target_frame_position_resolver=target_frame_position_resolver,
            )
            return {
                **state,
                "operation": operation,
                "position_error_m": position_error_m,
                "speed_mps": speed_mps,
                "loaded_hover_stable": True,
                "return_authorized": True,
            }
        raise RuntimeError(f"unsupported payload transition operation: {operation}")
    if step.driver == "ros2-service":
        return await _execute_ros2_domain_action(
            step.parameters, timeout_seconds=step.timeout_seconds
        )
    raise RuntimeError(f"unsupported runtime action driver: {step.driver}")


# 功能：
#   从仿真机载 RGB 数据集中检查最新完整记录、来源健康、时效及图像摘要，不使用外部示例图片。
# 输入：
#   parameters：已授权的相机操作参数。
#   now_unix_ms：本次检查的 UTC 毫秒时刻，None 表示读取当前时刻。
#   maximum_age_seconds：允许的图像来源最大年龄。
# 输出：
#   capture：通过记录和图像核对的机载采集证据。
def _verified_simulation_onboard_rgb_capture(
    parameters: dict[str, Any],
    *,
    now_unix_ms: int | None = None,
    maximum_age_seconds: float = 2.0,
) -> dict[str, Any] | None:
    """Bind a simulation camera action to a fresh onboard RGB artifact.

    Gazebo exposes the aircraft camera as a sensor topic rather than a MAVLink
    camera component.  The multimodal recorder is the trusted bridge for that
    topic: it persists each frame, its sensor health, and a hash-chained record.
    When the adapter explicitly supplies this dataset root, use the newest
    complete record instead of pretending that a nonexistent MAVSDK camera
    acknowledged the command.  Hardware runs do not set the environment value
    and continue through the real MAVSDK camera driver.
    """

    dataset_value = os.environ.get("DRONEDREAM_SIMULATION_ONBOARD_RGB_DATASET")
    if not dataset_value:
        return None
    dataset_root = Path(dataset_value)
    records_path = dataset_root / "records.jsonl"
    try:
        check_plain_plugin_path(records_path)
        record_status = records_path.stat()
        if not stat.S_ISREG(record_status.st_mode):
            raise ValueError("simulation onboard RGB record path is not a regular file")
        file_size = record_status.st_size
        if file_size <= 0:
            raise RuntimeError("simulation onboard RGB evidence has no records")
        tail_size = min(file_size, 1024 * 1024)
        with records_path.open("rb") as handle:
            if not os.path.samestat(record_status, os.fstat(handle.fileno())):
                raise ValueError("simulation onboard RGB records changed before opening")
            handle.seek(file_size - tail_size)
            tail = handle.read(tail_size)
        check_plain_plugin_path(records_path)
        if len(tail) != tail_size or not os.path.samestat(record_status, records_path.stat()):
            raise ValueError("simulation onboard RGB records changed while reading")
        lines = tail.splitlines()
        # Ignore only a genuinely in-progress final append. A complete but
        # invalid newest line is corruption and must not be bypassed by walking
        # backward to an older healthy frame.
        if tail and not tail.endswith(b"\n"):
            lines = lines[:-1]
        complete_lines = [line for line in lines if line.strip()]
        if not complete_lines:
            raise RuntimeError("simulation onboard RGB evidence has no complete typed record")
        record = RuntimeMultimodalDatasetRecord.model_validate(decode_json(complete_lines[-1]))
        record_payload = record.model_dump(mode="json")
        recorded_hash = record_payload.pop("record_sha256")
        if sha256_json(record_payload) != recorded_hash:
            raise RuntimeError("simulation onboard RGB record hash is invalid")
        observed_at = int(time.time() * 1_000) if now_unix_ms is None else now_unix_ms
        if (
            type(observed_at) is not int
            or not 0 <= observed_at < 2**63
            or not finite_positive_number(maximum_age_seconds)
        ):
            raise ValueError("simulation onboard RGB clock or age budget is invalid")
        age_seconds = (observed_at - record.recorded_at_unix_ms) / 1_000.0
        if not 0.0 <= age_seconds <= maximum_age_seconds:
            raise RuntimeError("simulation onboard RGB evidence is stale")
        rgb_statuses = [
            status for status in record.sensor_snapshot.statuses if status.modality == "rgb-camera"
        ]
        if (
            not rgb_statuses
            or any(status.health != "healthy" for status in rgb_statuses)
            or not record.sensor_snapshot.ready_for_motion
        ):
            raise RuntimeError("simulation onboard RGB sensor is not healthy")
        # 落盘是记录时间，不是曝光时间。保留 RGB 原采样年龄及快照中的传感器年龄，
        # 再加落盘后的墙钟耗时；迟到记录不能把旧图像刷新成新图像。
        rgb_age_at_record = (
            record.recorded_at_monotonic_seconds - record.rgb_sample_monotonic_seconds
        )
        snapshot_age_at_record = (
            record.recorded_at_monotonic_seconds
            - record.sensor_snapshot.captured_at_monotonic_seconds
        )
        if (
            rgb_age_at_record < 0
            or snapshot_age_at_record < 0
            or any(status.sample_age_seconds is None for status in rgb_statuses)
        ):
            raise RuntimeError("simulation onboard RGB source timing is invalid")
        age_seconds += max(
            rgb_age_at_record,
            snapshot_age_at_record + max(status.sample_age_seconds for status in rgb_statuses),
        )
        if not math.isfinite(age_seconds) or age_seconds > maximum_age_seconds:
            raise RuntimeError("simulation onboard RGB source is stale")
        frame_path = (dataset_root / record.rgb_relative_path).resolve()
        if dataset_root.resolve() not in frame_path.parents or not frame_path.is_file():
            raise RuntimeError("simulation onboard RGB artifact path is invalid")
        frame_sha256 = hashlib.sha256(
            read_plugin_file(dataset_root / record.rgb_relative_path, limit=20 * 1024 * 1024)
        ).hexdigest()
        if frame_sha256 != record.rgb_sha256:
            raise RuntimeError("simulation onboard RGB artifact hash is invalid")
        return {
            "confirmed": True,
            "transport": "gazebo-onboard-rgb-evidence",
            "command": str(parameters["command"]),
            "component_id": int(parameters["component_id"]),
            "sample_id": record.sample_id,
            "frame_sha256": frame_sha256,
            "record_sha256": record.record_sha256,
            "frame_age_seconds": age_seconds,
            "sensor_id": rgb_statuses[0].sensor_id,
        }
    except (OSError, ValueError) as error:
        # 类型化校验错误可能携带整个状态字典；外层日志仅保留错误类型。
        raise RuntimeError(
            f"simulation onboard RGB capture failed: {type(error).__name__}"
        ) from None


# 功能：
#   将各类驱动的实际确认字段映射成动作合同所需证据，没有对应回读就不声明成功。
# 输入：
#   step：决定驱动与动作类型的执行步骤。
#   output：该驱动返回的实际状态字典。
# 输出：
#   evidence：可被动作验收检查消费的证据名称列表。
def _observed_runtime_action_evidence(
    step: RuntimeActionExecutionStep, output: dict[str, Any]
) -> list[str]:
    if not isinstance(output, dict) or output.get("confirmed") is not True:
        return []
    if step.driver == "mavsdk-camera":
        return ["image captured", "pose bound", "timestamp bound"]
    if step.driver == "gazebo-payload":
        if str(output.get("operation")) == "attach" and output.get("detached") is False:
            return ["payload attached", "attachment state readback"]
        if str(output.get("operation")) == "detach" and output.get("detached") is True:
            return ["payload released", "release verified"]
        return []
    if step.driver == "payload-transition":
        if str(output.get("operation")) == "precontact":
            if (
                output.get("stable_precontact_hover") is True
                and output.get("no_payload_contact") is True
            ):
                return ["stable pre-contact hover", "no payload contact"]
            return []
        if (
            str(output.get("operation")) == "confirm-custody"
            and output.get("detached") is False
            and output.get("payload_physics_binding_confirmed") is True
            and output.get("custody_state_accepted") is True
        ):
            return [
                "payload attachment confirmed",
                "mass and inertia update confirmed",
                "custody state accepted",
            ]
        if (
            str(output.get("operation")) == "postattach-stability"
            and output.get("detached") is False
            and output.get("loaded_hover_stable") is True
            and output.get("return_authorized") is True
        ):
            return [
                "loaded hover stable",
                "post-attachment dynamics accepted",
                "return authorized",
            ]
        return []
    evidence = output.get("evidence", [])
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        return []
    return list(dict.fromkeys(item.strip() for item in evidence if item.strip()))


# 功能：
#   并行维持保护控制和一个动作，串行化局部刷新，超时或失败时取消并等待动作退出。
# 输入：
#   base：基础控制维持工具。
#   client：实际设备客户端。
#   setpoint：动作期间的稳定目标。
#   step：含时限的单个执行步骤。
#   abort_file：外部终止信号。
#   rate_hz：控制维持频率。
#   runtime_interrupt_probe：记录运行期间新改令的探针。
#   setpoint_refresh：局部安全和模型控制刷新回调。
#   sample_observer：实际遥测发布回调。
#   target_frame_position_resolver：稳定判定所需坐标变换。
# 输出：
#   action_result：实际驱动输出与期间检测到的改令组成的二元组。
async def _await_runtime_action_while_holding(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    step: RuntimeActionExecutionStep,
    abort_file: Path,
    rate_hz: float,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    target_frame_position_resolver: (Callable[[Any], tuple[float, float, float]] | None) = None,
) -> tuple[dict[str, Any], RuntimeInterruptDetected | None]:
    refresh_lock = asyncio.Lock()
    refresh_interval_seconds = max(0.02, 0.8 / rate_hz)
    last_refresh_at = float("-inf")
    latest_hold_setpoint = setpoint

    # 功能：
    #   合并近同时发起的局部控制刷新，防止动作与悬停各自推进一次控制周期。
    # 输入：
    #   planned_setpoint：调用方希望维持的目标。
    # 输出：
    #   refreshed：本轮共享的安全控制设定值。
    async def coordinated_setpoint_refresh(planned_setpoint: Any) -> Any:
        """Serialize and coalesce live safety refreshes across an action driver."""

        nonlocal last_refresh_at, latest_hold_setpoint
        if setpoint_refresh is None:
            return planned_setpoint
        async with refresh_lock:
            now = asyncio.get_running_loop().time()
            if now - last_refresh_at < refresh_interval_seconds:
                return latest_hold_setpoint
            latest_hold_setpoint = await setpoint_refresh(planned_setpoint)
            last_refresh_at = asyncio.get_running_loop().time()
            return latest_hold_setpoint

    operation = asyncio.create_task(
        _invoke_runtime_action_driver(
            base=base,
            client=client,
            step=step,
            setpoint=setpoint,
            rate_hz=rate_hz,
            runtime_interrupt_probe=runtime_interrupt_probe,
            setpoint_refresh=(
                coordinated_setpoint_refresh if setpoint_refresh is not None else None
            ),
            sample_observer=sample_observer,
            target_frame_position_resolver=target_frame_position_resolver,
        )
    )
    deadline = asyncio.get_running_loop().time() + step.timeout_seconds
    pending_interruption: RuntimeInterruptDetected | None = None
    try:
        while not operation.done():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"runtime action timed out: {step.step_id}")
            base._raise_if_external_abort_requested(abort_file)
            # Refresh every action, including payload state reads outside the
            # embedded stability gates. The coordinated callback prevents the
            # outer hold loop and an embedded gate from racing or duplicating a
            # refresh at the same control instant.
            if setpoint_refresh is not None:
                hold_setpoint = await coordinated_setpoint_refresh(setpoint)
            else:
                hold_setpoint = setpoint
            await client.set_position_ned(hold_setpoint)
            if pending_interruption is None and runtime_interrupt_probe is not None:
                pending_interruption = runtime_interrupt_probe()
            await asyncio.sleep(1.0 / rate_hz)
        return await operation, pending_interruption
    except BaseException:
        if not operation.done():
            operation.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await operation
        raise


# 功能：
#   执行并验证单步骤的真实设备回读，按合同限定重试次数；装卸副作用只尝试一次并保存回执。
# 输入：
#   base：基础控制工具。
#   client：实际飞控和设备客户端。
#   setpoint：动作期间的保护位置。
#   step：当前动作及验收要求。
#   contract：本次完整动作合同。
#   run_dir：动作回执所属运行目录。
#   abort_file：外部终止信号。
#   rate_hz：保护控制频率。
#   runtime_interrupt_probe：新改令探针。
#   setpoint_refresh：局部控制刷新。
#   sample_observer：真实遥测发布。
#   target_frame_position_resolver：目标坐标下的位置求解。
# 输出：
#   execution_result：通过验证的动作回执与待处理改令组成的二元组。
async def _execute_runtime_action_step(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    step: RuntimeActionExecutionStep,
    contract: RuntimeActionExecutionContract,
    run_dir: Path,
    abort_file: Path,
    rate_hz: float,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    target_frame_position_resolver: (Callable[[Any], tuple[float, float, float]] | None) = None,
) -> tuple[RuntimeActionExecutionReceipt, RuntimeInterruptDetected | None]:
    if runtime_interrupt_probe is not None:
        interruption = runtime_interrupt_probe()
        if interruption is not None:
            raise interruption
    contract_sha256 = sha256_json(contract)
    step_sha256 = sha256_json(step)
    receipt_path = run_dir / "runtime-actions" / "receipts" / f"{step.step_id}.receipt.json"
    started_at = datetime.now(UTC)
    output: dict[str, Any] = {}
    observed_evidence: list[str] = []
    issue_codes: list[str] = []
    pending_interruption: RuntimeInterruptDetected | None = None
    attempts = 0
    # The driver already sends bounded idempotent command publications.
    # Re-entering it after an unknown readback can realign a payload that is
    # already jointed and pull the aircraft, so the executor independently
    # enforces the current single-attempt rule.
    maximum_driver_attempts = 1 if step.driver == "gazebo-payload" else step.max_attempts
    for attempts in range(1, maximum_driver_attempts + 1):
        try:
            output, detected = await _await_runtime_action_while_holding(
                base=base,
                client=client,
                setpoint=setpoint,
                step=step,
                abort_file=abort_file,
                rate_hz=rate_hz,
                runtime_interrupt_probe=runtime_interrupt_probe,
                setpoint_refresh=setpoint_refresh,
                sample_observer=sample_observer,
                target_frame_position_resolver=target_frame_position_resolver,
            )
            pending_interruption = pending_interruption or detected
            if not isinstance(output, dict):
                # 保留可序列化的拒绝回执，不能在最终拒绝分支再次对损坏容器调用 get。
                output = {}
                raise ValueError("runtime action driver returned a non-object result")
            observed_evidence = _observed_runtime_action_evidence(step, output)
            required = {item.casefold() for item in step.required_success_evidence}
            observed = {item.casefold() for item in observed_evidence}
            deterministic_gates = {
                "driver_confirmed": output.get("confirmed") is True,
                "required_evidence_observed": required <= observed,
                "output_bound_to_adapter": (
                    isinstance(output.get("transport"), str) and bool(output["transport"].strip())
                ),
                "attempt_within_limit": attempts <= step.max_attempts,
            }
            if all(deterministic_gates.values()):
                receipt = RuntimeActionExecutionReceipt(
                    execution_contract_sha256=contract_sha256,
                    step_sha256=step_sha256,
                    step_id=step.step_id,
                    task_id=step.task_id,
                    action=step.action,
                    adapter_id=step.adapter_id,
                    runtime_executor=step.runtime_executor,
                    status="accepted",
                    attempts=attempts,
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                    output=output,
                    observed_success_evidence=observed_evidence,
                    deterministic_gates=deterministic_gates,
                )
                _atomic_json(receipt_path, receipt)
                return receipt, pending_interruption
            issue_codes = [
                name.upper() for name, accepted in deterministic_gates.items() if not accepted
            ]
        except RuntimeInterruptDetected:
            raise
        except Exception as exc:
            issue_codes = [f"{type(exc).__name__}:{str(exc)[:240]}"]
        if pending_interruption is not None:
            break
    receipt = RuntimeActionExecutionReceipt(
        execution_contract_sha256=contract_sha256,
        step_sha256=step_sha256,
        step_id=step.step_id,
        task_id=step.task_id,
        action=step.action,
        adapter_id=step.adapter_id,
        runtime_executor=step.runtime_executor,
        status="rejected",
        attempts=attempts,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        output=output,
        observed_success_evidence=observed_evidence,
        deterministic_gates={
            "driver_confirmed": output.get("confirmed") is True,
            "required_evidence_observed": False,
            "output_bound_to_adapter": (
                isinstance(output.get("transport"), str) and bool(output["transport"].strip())
            ),
            "attempt_within_limit": attempts <= step.max_attempts,
        },
        issue_codes=issue_codes or ["RUNTIME_ACTION_EXECUTION_REJECTED"],
    )
    _atomic_json(receipt_path, receipt)
    raise RuntimeError(f"runtime domain action rejected: {step.step_id}")


# 功能：
#   按触发点、检查点和任务依赖执行尚未完成的动作，成功后更新集合，避免重复装卸等副作用。
# 输入：
#   base：基础控制工具。
#   client：实际设备客户端。
#   setpoint：执行期间的保护位置。
#   contract：当前有效动作合同。
#   trigger：起飞后、检查点或重规划后等触发类型。
#   checkpoint_id：触发动作的检查点身份。
#   completed_step_ids：已完成步骤集合，由本函数成功后更新。
#   completed_task_ids：已完成任务集合，由本函数成功后更新。
#   run_dir：本次回执目录。
#   abort_file：外部终止信号。
#   rate_hz：保护控制频率。
#   runtime_interrupt_probe：新改令探针。
#   timing：累积的动作与中断记录。
#   setpoint_refresh：局部控制刷新回调。
#   sample_observer：实际遥测发布回调。
#   target_frame_position_resolver：稳定判断的位置变换。
# 输出：
#   None：不返回业务数据。
async def _execute_triggered_runtime_actions(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    contract: RuntimeActionExecutionContract | None,
    trigger: str,
    checkpoint_id: str | None,
    completed_step_ids: set[str],
    completed_task_ids: set[str],
    run_dir: Path,
    abort_file: Path,
    rate_hz: float,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None,
    timing: dict[str, Any],
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    target_frame_position_resolver: (Callable[[Any], tuple[float, float, float]] | None) = None,
) -> None:
    if contract is None:
        return
    applicable = [
        step
        for step in contract.steps
        if step.trigger == trigger
        and step.checkpoint_id == checkpoint_id
        and step.step_id not in completed_step_ids
    ]
    while applicable:
        ready = [step for step in applicable if set(step.depends_on) <= completed_task_ids]
        if not ready:
            unresolved = ",".join(
                f"{step.step_id}:{sorted(set(step.depends_on) - completed_task_ids)}"
                for step in applicable
            )
            raise RuntimeError(f"runtime action dependencies unresolved: {unresolved}")
        for step in ready:
            action_started = time.monotonic()
            receipt, interruption = await _execute_runtime_action_step(
                base=base,
                client=client,
                setpoint=setpoint,
                step=step,
                contract=contract,
                run_dir=run_dir,
                abort_file=abort_file,
                rate_hz=rate_hz,
                runtime_interrupt_probe=runtime_interrupt_probe,
                setpoint_refresh=setpoint_refresh,
                sample_observer=sample_observer,
                target_frame_position_resolver=target_frame_position_resolver,
            )
            completed_step_ids.add(step.step_id)
            completed_task_ids.add(step.task_id)
            timing["runtime_actions"].append(
                {
                    "step_id": step.step_id,
                    "task_id": step.task_id,
                    "receipt_sha256": sha256_json(receipt),
                    "duration_seconds": time.monotonic() - action_started,
                }
            )
            applicable.remove(step)
            if interruption is not None:
                raise interruption


# 功能：
#   在宣布路线任务完成前，检查合同要求的所有动作均已验证完成，不能以飞到终点替代动作成功。
# 输入：
#   contract：当前有效动作合同，None 表示未配置动作。
#   completed_step_ids：实际完成的步骤集合。
# 输出：
#   None：不返回业务数据。
def _assert_all_runtime_actions_completed(
    contract: RuntimeActionExecutionContract | None, completed_step_ids: set[str]
) -> None:
    if contract is None:
        return
    expected = {step.step_id for step in contract.steps}
    if not expected <= completed_step_ids:
        missing = ",".join(sorted(expected - completed_step_ids))
        raise RuntimeError(f"runtime actions were not completed: {missing}")


# 功能：
#   1. 将动作检查点绑定到指定航点的到达采样，区分同地点的多次访问。
#   2. 验证调度顺序、下标和坐标一致，拒绝重复检查点或静默覆盖动作身份。
# 输入：
#   base：当前飞控模块，提供一致的坐标转换。
#   schedule：已经完成航向对齐的采样序列。
#   points：与调度同源的局部轨迹点。
#   checkpoints：活动任务的检查点合同。
#   track_start_index：本轨迹阶段开始的采样下标。
#   waypoint_arrival_indices：调度编译器输出、经航向对齐同步重映射的到达下标。
# 输出：
#   found：到达采样下标至独立检查点对象的映射。
def _schedule_checkpoint_indices(
    base: ModuleType,
    schedule: list[Any],
    points: list[Any],
    checkpoints: RuntimeCheckpointContract,
    track_start_index: int,
    *,
    waypoint_arrival_indices: tuple[int, ...],
) -> dict[int, RuntimeCheckpoint]:
    next_track_point_index(waypoint_arrival_indices, 0, len(points))
    if (
        type(track_start_index) is not int
        or not 0 <= track_start_index < len(schedule)
        or any(not track_start_index < index < len(schedule) for index in waypoint_arrival_indices)
    ):
        raise ValueError("checkpoint arrival lies outside the track schedule")
    requested = {item.track_point_index: item for item in checkpoints.checkpoints}
    identities = {item.checkpoint_id for item in checkpoints.checkpoints}
    if len(requested) != len(checkpoints.checkpoints) or len(identities) != len(requested):
        raise ValueError("duplicate checkpoint point or identity")
    if any(type(index) is not int or not 1 <= index < len(points) for index in requested):
        raise ValueError("checkpoint track_point_index exceeds the reference track")
    found: dict[int, RuntimeCheckpoint] = {}
    for point_index, arrival_index in enumerate(waypoint_arrival_indices, 1):
        point = points[point_index]
        target = base.enu_point_to_ned_setpoint(point, yaw_deg=0.0)
        candidate = schedule[arrival_index]
        distance = math.dist(
            (candidate.north_m, candidate.east_m, candidate.down_m),
            (target.north_m, target.east_m, target.down_m),
        )
        if not math.isfinite(distance) or distance > 1e-8:
            raise ValueError(f"checkpoint arrival {point_index} has mismatched coordinates")
        if point_index in requested:
            found[arrival_index] = requested[point_index].model_copy(deep=True)
    return found


# 功能：
#   将出生点相对 NED 设定值转换为地图 ENU 碰撞中心，完整计入模型原点及碰撞中心偏移。
# 输入：
#   setpoint：相对北、东、向下位置。
#   coordinate_contract：模型与地图坐标绑定。
# 输出：
#   world：地图内的碰撞中心三维坐标。
def _setpoint_world_enu(
    setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
) -> Vector3:
    root_east, root_north, root_up = coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        coordinate_contract.resolved_collision_center_offset_model_m()
    )
    return Vector3(
        x=root_east + offset_east + float(setpoint.east_m),
        y=root_north + offset_north + float(setpoint.north_m),
        z=root_up + offset_up - float(setpoint.down_m),
    )


# 功能：
#   将实测位置换到当前目标所属参考系，使用本次控制实际绑定的定位偏移。
# 输入：
#   args：保存最近控制坐标绑定的上下文。
#   observed：PX4 实测位置样本。
# 输出：
#   position_ned：目标参考系内的北、东、向下位置三元组。
def _observed_route_frame_position_ned(
    *, args: argparse.Namespace, observed: Any
) -> tuple[float, float, float]:
    """Resolve PX4 telemetry into the signed route frame used by setpoints.

    Gazebo identity alignment intentionally tracks the small PX4-estimator to
    world offset.  Local safety and schedule-progress gates already apply this
    transform.  Waypoint and payload stability gates must use the same frame;
    comparing raw estimator coordinates to a route-frame setpoint otherwise
    leaves a permanent error equal to the estimator offset.
    """

    offset = getattr(
        args,
        "_last_estimator_to_world_position_offset_m",
        Vector3(x=0.0, y=0.0, z=0.0),
    )
    return (
        float(observed.north_m) + float(offset.y),
        float(observed.east_m) + float(offset.x),
        float(observed.down_m) - float(offset.z),
    )


# 功能：
#   将地图 ENU 碰撞中心逆变换成出生点相对 NED 控制位置，保持上游明确给出的航向。
# 输入：
#   base：设定值类型所属基础模块。
#   world：地图碰撞中心坐标。
#   yaw_deg：明确指定的控制航向。
#   coordinate_contract：模型原点与碰撞中心绑定。
# 输出：
#   setpoint：相对 NED 位置和航向设定值。
def _world_enu_setpoint(
    *,
    base: ModuleType,
    world: Vector3,
    yaw_deg: float,
    coordinate_contract: Px4CoordinateContract,
) -> Any:
    """Convert an ENU collision-center target to the adapter's relative NED.

    Subtract the declared spawn/model offset exactly once; this is not an
    estimator correction or a learned motion proposal. Up becomes negative down.
    """
    root_east, root_north, root_up = coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        coordinate_contract.resolved_collision_center_offset_model_m()
    )
    return base.Setpoint(
        north_m=world.y - root_north - offset_north,
        east_m=world.x - root_east - offset_east,
        down_m=-(world.z - root_up - offset_up),
        yaw_deg=yaw_deg,
    )


# 功能：
#   记录真正发送的控制及接收时刻，重新核对输入期限；迟到已发送不能被伪装成从未发送。
# 输入：
#   args：执行计数和有界证据写入器。
#   command：本次控制绑定的原始观测及模型意图合同。
#   velocity_ned_mps：实际发送的 NED 速度。
#   yaw_deg：实际发送的航向。
#   transport：实际使用的纯速度或位置速度接口。
#   accepted_at_unix_ms：适配器返回时刻，未提供时读取当前 UTC 毫秒。
#   position_ned_m：位置接口实际发送的位置，纯速度时不需要。
# 输出：
#   None：不返回业务数据。
def _record_model_control_application(
    args: argparse.Namespace,
    command: RuntimeLocalSafetyCommand,
    *,
    velocity_ned_mps: tuple[float, float, float] | None = None,
    yaw_deg: float | None = None,
    transport: str = "velocity-ned",
    accepted_at_unix_ms: int | None = None,
    position_ned_m: tuple[float, float, float] | None = None,
) -> None:
    """Account for an actual transport acceptance, never a proposed action.

    Velocity is NED in m/s, heading is degrees, and the timestamp is the actual
    adapter acceptance in UTC milliseconds. Preserve late-send evidence before
    failing; safety enforcement must not depend on a diagnostic writer existing.
    An accepted send is not proof that the aircraft reached the intended state.
    """

    accepted_at = int(time.time() * 1000) if accepted_at_unix_ms is None else accepted_at_unix_ms
    if type(accepted_at) is not int or accepted_at < 0:
        raise ValueError("CONTROL_APPLICATION_ACCEPTANCE_TIME_INVALID")
    intent = command.requested_control_intent
    category = control_application_category(
        model_authorized=command.model_navigation_authorized,
        action=command.decision.action,
        control_source=command.decision.control_source,
        heading_assisted=intent is not None and intent.yaw_control_mode == "route-heading-assist",
    )
    counts = dict(getattr(args, "_model_control_application_counts", {}))
    counts[category] = counts.get(category, 0) + 1
    args._model_control_application_counts = counts
    writer = getattr(args, "_control_application_writer", None)
    if writer is not None:
        if velocity_ned_mps is None or yaw_deg is None:
            raise ValueError("CONTROL_APPLICATION_TRANSPORT_DETAILS_REQUIRED")
        record = control_application_record(
            command,
            sequence=sum(counts.values()),
            accepted_at_unix_ms=accepted_at,
            transport=transport,
            velocity_ned_mps=velocity_ned_mps,
            yaw_heading_deg=yaw_deg,
            position_ned_m=position_ned_m,
            yaw_rate_application=(
                args._last_model_yaw_application[1]
                if getattr(args, "_last_model_yaw_application", (None,))[0] == sha256_json(command)
                else None
            ),
        )
        writer.submit(
            args.run_dir / "runtime-state" / "control-applications.jsonl",
            record.model_dump(mode="json"),
        )
    # Logging is optional; the immutable observation deadline is not. A late
    # physical send stays in the applied counts, but never counts as valid model
    # authority or resets the interrupted-input recovery window.
    if (
        command.decision.action in {"continue", "slow", "replan"}
        and accepted_at > command.valid_until_unix_ms
    ):
        raise UserDirectedLanding("CONTROL_TRANSPORT_ACCEPTANCE_EXCEEDED_INPUT_DEADLINE")
    if accepted_at <= command.valid_until_unix_ms:
        _clear_dispatch_input_gap(args)
    if command.model_navigation_authorized:
        args._model_authorized_control_applied_count = (
            getattr(args, "_model_authorized_control_applied_count", 0) + 1
        )


# 功能：
#   模型控制的保护悬停使用新鲜实测航向，并清除旧偏航积分；明确兼容模式才允许回退航向。
# 输入：
#   args：模型偏航积分与来源状态。
#   client：当前原生姿态客户端。
#   fallback_heading_deg：非模型兼容模式的备用航向。
#   model_control_required：是否必须维持模型模式的新鲜姿态约束。
# 输出：
#   heading：本次保护控制允许使用的航向。
def _local_hold_yaw(
    *,
    args: argparse.Namespace,
    client: Any,
    fallback_heading_deg: float,
    model_control_required: bool = False,
) -> float:
    """Latch native attitude on a safety handover; do not undo a model turn."""
    required = (
        model_control_required
        or bool(getattr(args, "require_model_control_authority", False))
        or bool(getattr(args, "_last_model_control_required", False))
    )
    if not required:
        # Explicit route-following compatibility does not claim model control.
        return float(fallback_heading_deg)
    try:
        telemetry = client.latest_dynamics_telemetry(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS)
        attitude = telemetry["sources"]["attitude"]
        heading = measured_hold_heading(
            yaw_deg=attitude["yaw_deg"],
            sample_age_seconds=attitude["sample_age_seconds"],
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise UserDirectedLanding("MODEL_HOLD_REQUIRES_FRESH_NATIVE_HEADING") from error
    # A safety hold interrupts yaw integration. Resumption starts here instead
    # of catching up to a pre-braking target or to the planned route bearing.
    args._model_body_control_yaw_deg = heading
    args._last_model_yaw_application = None, None
    return heading


# 功能：
#   按本地周期积分模型授权偏航速率，安全覆盖禁止继续旧转向；不以路线航向替代模型输出。
# 输入：
#   base：设定值构造模块。
#   args：周期限制及最近偏航积分状态。
#   setpoint：仅保留位置部分的当前目标。
#   command：模型意图、安全仲裁结果和来源绑定。
# 输出：
#   yaw_setpoint：使用本轮授权航向的设定值。
def _setpoint_with_model_body_yaw(
    *,
    base: ModuleType,
    args: argparse.Namespace,
    setpoint: Any,
    command: RuntimeLocalSafetyCommand,
) -> Any:
    """Integrate the approved rate, including an explicit safety yaw veto."""

    decision = command.decision
    args._last_model_yaw_application = None, None
    simulation_teacher = bool(getattr(args, "simulation_teacher_control", False))
    safety_velocity_override = (
        getattr(decision, "control_source", None) == "deterministic-safety-override"
        and getattr(command, "navigation_control_authority", None) == "model-required"
        and getattr(command, "model_navigation_authorized", False)
    )
    if safety_velocity_override and decision.selected_yaw_rate_dps != 0.0:
        raise UserDirectedLanding("SAFETY_VELOCITY_OVERRIDE_CANNOT_AUTHOR_A_TURN")
    if not simulation_teacher and (
        (
            getattr(decision, "control_source", "route-target") != "local-model-body-control"
            and not safety_velocity_override
        )
        or decision.action not in {"continue", "slow"}
        or getattr(command, "requested_control_intent", None) is None
    ):
        args._model_body_control_yaw_deg = (
            float(setpoint.yaw_deg)
            if getattr(command, "navigation_control_authority", None) == "model-required"
            else None
        )
        return setpoint
    previous = getattr(args, "_model_body_control_yaw_deg", None)
    if not isinstance(previous, int | float) or not math.isfinite(float(previous)):
        previous = float(setpoint.yaw_deg)
    rate_limit = min(
        180.0,
        float(getattr(args, "maximum_yaw_rate_deg_s", 20.0)),
    )
    requested_rate = max(
        -rate_limit,
        min(rate_limit, float(decision.selected_yaw_rate_dps)),
    )
    if (
        not safety_velocity_override
        and command.requested_control_intent is not None
        and command.requested_control_intent.yaw_control_mode == "route-heading-assist"
    ):
        # Assistance is an explicit authority choice, never inferred from the
        # magnitude of a learned command. It must remain visible in evidence.
        args._model_body_control_yaw_deg = None
        return setpoint
    next_yaw = integrate_model_yaw(
        previous_heading_deg=float(previous),
        requested_rate_dps=requested_rate,
        maximum_rate_dps=rate_limit,
        step_seconds=1.0 / float(args.setpoint_rate_hz),
    )
    args._model_body_control_yaw_deg = next_yaw
    args._last_model_yaw_application = (
        sha256_json(command),
        {
            "previous_heading_deg": float(previous),
            "clockwise_rate_dps": requested_rate,
            "integration_seconds": 1.0 / float(args.setpoint_rate_hz),
        },
    )
    return base.Setpoint(
        north_m=float(setpoint.north_m),
        east_m=float(setpoint.east_m),
        down_m=float(setpoint.down_m),
        yaw_deg=next_yaw,
    )


# 功能：
#   根据有序调度进度选取仍待到达的世界航点；目标标识保持稳定，不跟随移动设定值跳变。
# 输入：
#   track：当前活动轨迹。
#   waypoint_arrival_indices：每个后续原始航点对应的调度到达下标。
#   schedule_index：当前调度下标。
# 输出：
#   goal：世界 ENU 目标位置与稳定来源标识的二元组。
def _navigation_goal_for_schedule(
    *,
    track: Px4Track,
    waypoint_arrival_indices: tuple[int, ...],
    schedule_index: int,
) -> tuple[Vector3, str]:
    points = track.source_world_points
    if not points:
        raise ValueError("PX4 track has no source-world navigation points")
    point_index = next_track_point_index(waypoint_arrival_indices, schedule_index, len(points))
    point = points[point_index]
    goal = (
        Vector3(x=point.east_m, y=point.north_m, z=point.up_m),
        f"source-waypoint-{point_index:04d}",
    )
    return goal


# 功能：
#   发布稳定身份的语义目标、控制精度和恢复上下文，供本地控制工作器判断本次命令属于哪个目标。
# 输入：
#   path：可选目标通道文件。
#   setpoint：当前参考位置。
#   coordinate_contract：地图变换合同。
#   navigation_goal_position_m：明确语义目标，未提供时使用当前固定参考位置。
#   navigation_goal_id：目标身份，未提供时由固定内容生成。
#   control_profile：巡航或精细控制模式。
#   action_checkpoint_goal：目标是否带有必须完成的动作。
#   tracking_recovery_active：是否正在进行固定目标跟踪恢复。
#   decision_trigger：需要专家处理的触发原因。
#   recovery_episode_id：本次恢复事件身份。
# 输出：
#   None：不返回业务数据。
def _publish_local_safety_target(
    *,
    path: Path | None,
    setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
    navigation_goal_position_m: Vector3 | None = None,
    navigation_goal_id: str | None = None,
    control_profile: str = "cruise",
    action_checkpoint_goal: bool = False,
    tracking_recovery_active: bool = False,
    decision_trigger: str | None = None,
    recovery_episode_id: str | None = None,
) -> None:
    if path is None:
        return
    if control_profile not in {"cruise", "precision"}:
        raise ValueError(f"unsupported navigation control profile: {control_profile}")
    target = _setpoint_world_enu(setpoint, coordinate_contract)
    effective_navigation_goal = navigation_goal_position_m or target
    effective_navigation_goal_id = navigation_goal_id
    if effective_navigation_goal_id is None:
        # A fixed action/takeoff/landing target still needs a stable identity.
        # Never revive the former moving "current control setpoint" sentinel:
        # changing goal identity at a waypoint boundary invalidates a live model
        # path lease and can strand the aircraft just outside the settle gate.
        effective_navigation_goal_id = (
            "fixed-target-" + sha256_json(effective_navigation_goal.model_dump(mode="json"))[:24]
        )
    _atomic_json(
        path,
        {
            "schema_version": "dronedream.local-safety-target.v1",
            "target_position_m": target.model_dump(mode="json"),
            "navigation_goal_position_m": (effective_navigation_goal.model_dump(mode="json")),
            "navigation_goal_id": effective_navigation_goal_id,
            "control_profile": control_profile,
            "action_checkpoint_goal": action_checkpoint_goal,
            "tracking_recovery_active": tracking_recovery_active,
            "decision_trigger": decision_trigger,
            "recovery_episode_id": recovery_episode_id,
            "updated_at_unix_ms": int(time.time() * 1_000),
        },
    )


# 功能：
#   读取当前观测与安全命令配对，复核序号、摘要及剩余传输期限，坏配对不授予运动。
# 输入：
#   args：实时通道、持久证据路径与本次控制模式。
# 输出：
#   command：本次可消费的安全命令，未就绪、错配或过期时为 None。
def _read_local_safety_command(args: argparse.Namespace) -> RuntimeLocalSafetyCommand | None:
    receiver = getattr(args, "_local_safety_receiver", None)
    if receiver is not None:
        pair = receiver.read_latest()
        if pair is None:
            return None
        _observation, command = pair
        now_ms = int(time.time() * 1000)
        # This function runs AFTER the executor's telemetry wait, in the tick
        # that sends the command. The producer already reserved that tick.
        # Reserving another whole period here rejects an otherwise timely
        # command even when no further scheduling wait precedes transport.
        # The sender rechecks the transport budget immediately before I/O.
        reserve = (
            LOCAL_TRANSPORT_BUDGET_MS
            if command.decision.action in {"continue", "slow", "replan"}
            and (
                getattr(args, "simulation_teacher_control", False)
                or command.navigation_control_authority == "model-required"
            )
            else 0
        )
        if command.generated_at_unix_ms > now_ms or command.valid_until_unix_ms - now_ms < reserve:
            return None
        return command
    if args.local_safety_command is None or not args.local_safety_command.is_file():
        return None
    # Observation and command are separate atomic files.  Read the command on
    # both sides of the observation and accept only a stable pair; otherwise a
    # perfectly valid producer update could look like a hash violation for one
    # control tick.
    for _attempt in range(3):
        try:
            command_before = read_plugin_file(args.local_safety_command, limit=MAX_MESSAGE_BYTES)
            observation_raw = (
                read_plugin_file(args.local_safety_observation, limit=MAX_MESSAGE_BYTES)
                if args.local_safety_observation is not None
                else None
            )
            command_after = read_plugin_file(args.local_safety_command, limit=MAX_MESSAGE_BYTES)
            if command_before != command_after:
                continue
            command = RuntimeLocalSafetyCommand.model_validate(decode_json(command_after))
            now_ms = int(time.time() * 1000)
            if command.generated_at_unix_ms > now_ms or command.valid_until_unix_ms < now_ms:
                return None
            if observation_raw is not None:
                observation = RuntimeLocalSafetyObservation.model_validate(
                    decode_json(observation_raw)
                )
                if command.observation_sequence != observation.sequence:
                    continue
                if command.observation_sha256 != sha256_json(observation):
                    continue
            if (
                command.decision.action in {"continue", "slow", "replan"}
                and (
                    getattr(args, "simulation_teacher_control", False)
                    or command.navigation_control_authority == "model-required"
                )
                and command.valid_until_unix_ms - int(time.time() * 1000)
                < LOCAL_TRANSPORT_BUDGET_MS
            ):
                return None
            return command
        except (OSError, ValueError):
            continue
    return None


# 功能：
#   以原始接收时间发布真实位置速度及坐标绑定，独立发布流启动后保持单写入者所有权。
# 输入：
#   args：本次原生状态通道和输出路径。
#   coordinate_contract：地图和 PX4 坐标绑定。
#   observed：实际位置速度样本及原始接收时间。
#   dynamics_telemetry：已有动力学传感器包。
#   independent_stream：是否由当前独立发布流调用。
# 输出：
#   None：不返回业务数据。
def _publish_px4_identity_telemetry(
    *,
    args: argparse.Namespace,
    coordinate_contract: Px4CoordinateContract,
    observed: Any,
    dynamics_telemetry: dict[str, Any] | None = None,
    independent_stream: bool = False,
) -> None:
    """Publish one already-observed PX4 estimator sample for identity binding."""

    if getattr(args, "_independent_native_publisher_active", False) and not independent_stream:
        return
    values = (
        observed.north_m,
        observed.east_m,
        observed.down_m,
        observed.north_m_s,
        observed.east_m_s,
        observed.down_m_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise UserDirectedLanding("identity telemetry contained non-finite values")
    root_east, root_north, root_up = coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        coordinate_contract.resolved_collision_center_offset_model_m()
    )
    payload: dict[str, Any] = {
        "schema_version": "dronedream.px4-identity-telemetry.v1",
        "observed_position_ned_m": {
            "north_m": observed.north_m,
            "east_m": observed.east_m,
            "down_m": observed.down_m,
        },
        # Missing provenance stays missing: a cached sample cannot be stamped
        # fresh merely because this JSON document is being published again.
        "position_received_at_unix_ms": getattr(observed, "received_at_unix_ms", None),
        "map_frame_binding": {
            "source": "deployment-coordinate-contract",
            "orientation": "NED-to-ENU-fixed",
            "collision_center_origin_world_enu_m": {
                "x": root_east + offset_east,
                "y": root_north + offset_north,
                "z": root_up + offset_up,
            },
        },
        "observed_world_collision_center_m": {
            "x": root_east + offset_east + observed.east_m,
            "y": root_north + offset_north + observed.north_m,
            "z": root_up + offset_up - observed.down_m,
        },
        "observed_velocity_ned_mps": {
            "north_m_s": observed.north_m_s,
            "east_m_s": observed.east_m_s,
            "down_m_s": observed.down_m_s,
        },
        "updated_at_unix_ms": int(time.time() * 1_000),
    }
    payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    if dynamics_telemetry is not None:
        payload["dynamics"] = dynamics_telemetry
    native_channel = getattr(args, "_native_state_channel", None)
    if native_channel is not None:
        # Send the unmodified measured packet before durable filesystem I/O.
        # The independent producer is the sole writer after it starts.
        native_channel.send(payload)
    _atomic_json(
        args.run_dir / "runtime-state" / "px4-identity-telemetry.json",
        payload,
    )


# 功能：
#   从客户端已有缓存读取限定年龄的动力学包，不创建额外订阅或制造缺失数据。
# 输入：
#   client：提供最新动力学缓存的客户端。
# 输出：
#   value：可用的动力学字典，接口缺失或读取失败时为 None。
def _latest_px4_dynamics_telemetry(client: Any) -> dict[str, Any] | None:
    getter = getattr(client, "latest_dynamics_telemetry", None)
    if not callable(getter):
        return None
    try:
        value = getter(3.0)
    except (RuntimeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# 功能：
#   1. 读取实际 PX4 位置速度，首次超时后只重启一次订阅并在固定恢复期限内重采样。
#   2. 恢复期限包含订阅重启和采样等待，不接纳恢复窗口外的迟到结果。
#   3. 无法恢复时保留失败证据并交由调用方保护退出，不自行推进路线或伪造观测。
# 输入：
#   args：本次遥测时限、输出路径及原生状态通道。
#   client：实际遥测客户端。
#   coordinate_contract：遥测 NED 与地图 ENU 的绑定合同。
# 输出：
#   observed：期限内实际收到且已发布的 PX4 位置速度样本。
async def _refresh_px4_identity_telemetry(
    *,
    args: argparse.Namespace,
    client: Any,
    coordinate_contract: Px4CoordinateContract,
) -> Any:
    sample_timeout_seconds = args.tracking_telemetry_timeout_seconds
    recovery_timeout_seconds = getattr(args, "tracking_telemetry_recovery_timeout_seconds", 5.0)
    if not all(
        finite_positive_number(value)
        for value in (
            sample_timeout_seconds,
            recovery_timeout_seconds,
        )
    ):
        raise ValueError("PX4 telemetry timeouts must be finite and positive")
    loop = asyncio.get_running_loop()
    try:
        observed = await asyncio.wait_for(
            client.sample_position_velocity_ned(sample_timeout_seconds),
            sample_timeout_seconds,
        )
    except TimeoutError as initial_error:
        recovery_started = loop.time()
        recovery_deadline = recovery_started + recovery_timeout_seconds
        if not math.isfinite(recovery_deadline):
            raise ValueError("PX4 telemetry recovery deadline must be finite") from initial_error
        attempts, stream_restarted = 1, False
        _record_px4_telemetry_recovery(
            args,
            status="holding",
            duration_seconds=0.0,
            sample_attempts=attempts,
            stream_restarted=False,
            error=initial_error,
        )
        try:
            async with asyncio.timeout_at(recovery_deadline):
                restart = getattr(client, "restart_position_velocity_ned_stream", None)
                if callable(restart):
                    await restart()
                    stream_restarted = True
                while True:
                    remaining = recovery_deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError("telemetry recovery deadline expired")
                    attempts += 1
                    budget = min(sample_timeout_seconds, remaining)
                    try:
                        observed = await asyncio.wait_for(
                            client.sample_position_velocity_ned(budget),
                            budget,
                        )
                    except TimeoutError:
                        await asyncio.sleep(min(0.05, max(0.0, recovery_deadline - loop.time())))
                        continue
                    # 同步执行很久的协程可能直到返回都未让出事件循环，仍需显式复核期限。
                    if loop.time() >= recovery_deadline:
                        raise TimeoutError("telemetry sample returned after recovery deadline")
                    break
        except TimeoutError as error:
            _record_px4_telemetry_recovery(
                args,
                status="failed",
                duration_seconds=loop.time() - recovery_started,
                sample_attempts=attempts,
                stream_restarted=stream_restarted,
                error=error,
            )
            raise TimeoutError(
                "PX4 position/velocity telemetry did not recover within "
                f"{recovery_timeout_seconds:g}s after {attempts} fresh attempts"
            ) from error
        _record_px4_telemetry_recovery(
            args,
            status="recovered",
            duration_seconds=loop.time() - recovery_started,
            sample_attempts=attempts,
            stream_restarted=stream_restarted,
            error=None,
        )
    _publish_px4_identity_telemetry(
        args=args,
        coordinate_contract=coordinate_contract,
        observed=observed,
        dynamics_telemetry=_latest_px4_dynamics_telemetry(client),
    )
    return observed


# 功能：
#   按中断、恢复或失败状态更新遥测故障计数，并仅保留最近一百条事件。
# 输入：
#   args：当前运行证据路径。
#   status：本次恢复生命周期状态。
#   duration_seconds：此次中断持续时间。
#   sample_attempts：真实采样尝试次数。
#   stream_restarted：订阅重启是否已完成。
#   error：本次实际异常，恢复成功时为 None。
# 输出：
#   None：不返回业务数据。
def _record_px4_telemetry_recovery(
    args: argparse.Namespace,
    *,
    status: str,
    duration_seconds: float,
    sample_attempts: int,
    stream_restarted: bool,
    error: BaseException | None,
) -> None:
    """Publish a bounded audit trail for every in-flight telemetry outage."""

    path = args.run_dir / "runtime-state" / "px4-telemetry-recovery.json"
    try:
        payload = read_runtime_object(path)
    except (OSError, ValueError):
        payload = {
            "schema_version": "dronedream.px4-telemetry-recovery.v1",
            "outage_count": 0,
            "recovered_outage_count": 0,
            "failed_outage_count": 0,
            "events": [],
        }
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    if status == "holding":
        payload["outage_count"] = int(payload.get("outage_count", 0)) + 1
    elif status == "recovered":
        payload["recovered_outage_count"] = int(payload.get("recovered_outage_count", 0)) + 1
    elif status == "failed":
        payload["failed_outage_count"] = int(payload.get("failed_outage_count", 0)) + 1
    events.append(
        {
            "status": status,
            "duration_seconds": max(0.0, float(duration_seconds)),
            "sample_attempts": max(1, int(sample_attempts)),
            "stream_restarted": bool(stream_restarted),
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "recorded_at_unix_ms": int(time.time() * 1_000),
        }
    )
    payload["events"] = events[-100:]
    payload["last_status"] = status
    payload["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(path, payload)


# 功能：
#   将控制路径的真实数值转换为有限标量，拒绝布尔值、字符串及不可表示的巨大整数。
# 输入：
#   value：待使用的物理量或时刻。
#   label：错误信息中的字段名称。
#   minimum：可选下界，None 表示允许任意有限符号值。
# 输出：
#   scalar：已验证的浮点数。
def _control_scalar(value: object, label: str, *, minimum: float | None = None) -> float:
    if (
        type(value) not in (int, float)
        or not -sys.float_info.max <= value <= sys.float_info.max
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{label} must be a finite control number within its bounds")
    scalar = float(value)
    return scalar


# 功能：
#   核对三维控制向量的长度和每一维数值，不容许字符串拆分、缺失维度或隐式类型转换。
# 输入：
#   value：三维坐标或速度列表／元组。
#   label：错误信息中的向量名称。
# 输出：
#   vector：包含三个有限浮点数的独立元组。
def _control_vector3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-component control vector")
    vector = tuple(_control_scalar(component, label) for component in value)
    return vector


# 功能：
#   根据相邻参考点和周期提取 NED 速度并限制幅值，固定位置恢复时关闭路线速度前馈。
# 输入：
#   current_setpoint：当前参考点。
#   next_setpoint：下一参考点。
#   rate_hz：参考序列频率。
#   speed_limit_mps：允许的最大速度。
#   recovery_active：是否关闭路线切向前馈。
# 输出：
#   velocity_ned：北、东、向下速度三元组。
def _schedule_velocity_feedforward(
    *,
    current_setpoint: Any,
    next_setpoint: Any,
    rate_hz: float,
    speed_limit_mps: float,
    recovery_active: bool,
) -> tuple[float, float, float]:
    """Recover the dynamically qualified velocity encoded by a dense schedule."""

    if type(recovery_active) is not bool or not all(
        finite_positive_number(value) for value in (rate_hz, speed_limit_mps)
    ):
        raise ValueError("route velocity feedforward policy is invalid")
    if recovery_active:
        return 0.0, 0.0, 0.0
    current = _control_vector3(
        (current_setpoint.north_m, current_setpoint.east_m, current_setpoint.down_m),
        "current position",
    )
    target = _control_vector3(
        (next_setpoint.north_m, next_setpoint.east_m, next_setpoint.down_m),
        "next position",
    )
    velocity = _control_vector3(
        tuple((target[i] - current[i]) * rate_hz for i in range(3)),
        "route velocity",
    )
    speed = _control_scalar(math.hypot(*velocity), "route speed", minimum=0)
    if speed <= speed_limit_mps or speed <= 1e-12:
        return velocity
    ratio = speed_limit_mps / speed
    return tuple(component * ratio for component in velocity)


# 功能：
#   分离三维路线横向偏差和沿线时间落后，避免把正常控制响应滞后误算成障碍净空消耗。
# 输入：
#   observed_route_frame_ned：目标参考系内的实测位置。
#   planned_setpoint：当前参考位置。
#   planned_velocity_ned_mps：用于确定路线切向的参考速度。
# 输出：
#   components：总误差、横向误差、落后／超前距离及有界沿线容差。
def _route_tracking_error_components(
    *,
    observed_route_frame_ned: tuple[float, float, float],
    planned_setpoint: Any,
    planned_velocity_ned_mps: tuple[float, float, float] | None,
) -> dict[str, float | bool]:
    """Separate corridor deviation from harmless along-route time lag.

    Collision clearance is spent by motion perpendicular to the qualified
    three-dimensional route, not by remaining a few centimetres behind on
    that same route.  A separate bounded along-track lag prevents the route
    clock from running away while avoiding stop/start recovery at every dense
    reference sample.  With no usable route tangent (turn-in-place or final
    hold), the full position error remains the strict corridor error.
    """

    observed = _control_vector3(observed_route_frame_ned, "observed route position")
    planned = _control_vector3(
        (planned_setpoint.north_m, planned_setpoint.east_m, planned_setpoint.down_m),
        "planned position",
    )
    error = _control_vector3(tuple(planned[i] - observed[i] for i in range(3)), "tracking error")
    total_error_m = _control_scalar(math.hypot(*error), "tracking distance", minimum=0)
    velocity = _control_vector3(
        (0.0, 0.0, 0.0) if planned_velocity_ned_mps is None else planned_velocity_ned_mps,
        "planned velocity",
    )
    planned_speed_mps = _control_scalar(math.hypot(*velocity), "planned speed", minimum=0)
    if planned_speed_mps <= 1e-9:
        return {
            "tangent_available": False,
            "total_error_m": total_error_m,
            "cross_track_error_m": total_error_m,
            "along_track_lag_m": 0.0,
            "along_track_lead_m": 0.0,
            "along_track_lag_limit_m": 0.0,
            "along_track_lag_rejoin_limit_m": 0.0,
        }
    tangent = tuple(component / planned_speed_mps for component in velocity)
    signed_lag_m = sum(error[index] * tangent[index] for index in range(3))
    # 叉积范数直接保留横向分量，避免两个巨大平方相减吞掉厘米级真实偏差。
    cross_track_error_m = math.hypot(
        error[1] * tangent[2] - error[2] * tangent[1],
        error[2] * tangent[0] - error[0] * tangent[2],
        error[0] * tangent[1] - error[1] * tangent[0],
    )
    _control_scalar(signed_lag_m, "along-track lag")
    _control_scalar(cross_track_error_m, "cross-track error", minimum=0)
    # Bound the reference lead to at most one second of the qualified
    # velocity, with absolute floors/ceilings for very slow and fast tracks.
    # Live PX4 evidence on the 0.15 m/s School Map corridor measured a stable
    # 10.8 cm along-route lag, so a 0.75 s (11.25 cm) window sat directly on
    # normal controller variance.  One second retains a strict time bound
    # without turning harmless longitudinal response into recovery chatter.
    along_track_lag_limit_m = max(0.10, min(0.75, planned_speed_mps))
    # Once the outer lag bound is crossed, keep the fixed-target recovery
    # context active until the aircraft has regained a quarter of that time
    # budget.  Releasing at the same boundary made the worker and executor
    # alternate between ordinary and recovery proofs every few samples.
    along_track_lag_rejoin_limit_m = along_track_lag_limit_m * 0.75
    return {
        "tangent_available": True,
        "total_error_m": total_error_m,
        "cross_track_error_m": cross_track_error_m,
        "along_track_lag_m": max(0.0, signed_lag_m),
        "along_track_lead_m": max(0.0, -signed_lag_m),
        "along_track_lag_limit_m": along_track_lag_limit_m,
        "along_track_lag_rejoin_limit_m": along_track_lag_rejoin_limit_m,
    }


# 功能：
#   对固定恢复点施加有界径向速度和速度阻尼，在死区内归零，不复用会产生绕圈的切向前馈。
# 输入：
#   observed：真实位置速度，未提供时不增加前馈。
#   target_setpoint：固定恢复目标。
#   gain_s_inverse：位置误差到速度的增益。
#   velocity_damping：实测速度阻尼系数。
#   maximum_speed_mps：辅助速度上限。
#   deadband_m：不再增加前馈的位置死区。
# 输出：
#   velocity_ned：受限的北、东、向下恢复速度。
def _recovery_position_velocity_feedforward(
    *,
    observed: Any | None,
    target_setpoint: Any,
    gain_s_inverse: float = 0.8,
    velocity_damping: float = 0.5,
    maximum_speed_mps: float = 0.08,
    deadband_m: float = 0.005,
) -> tuple[float, float, float]:
    """Provide bounded radial assistance to PX4's precision position loop.

    A zero feed-forward avoids the tangential orbit produced by reusing a
    planner candidate during recovery, but PX4 can retain a centimetre-scale
    steady-state position error near hover.  This controller points only at the
    fixed absolute recovery target, decays linearly with remaining error, and
    becomes exactly zero inside a small deadband.  It therefore helps the
    aircraft rejoin a strict corridor without changing the target or relaxing
    any tracking gate.
    """

    if not finite_positive_number(maximum_speed_mps):
        raise ValueError("recovery maximum speed must be finite and positive")
    gain_s_inverse = _control_scalar(gain_s_inverse, "recovery gain", minimum=0)
    velocity_damping = _control_scalar(velocity_damping, "recovery damping", minimum=0)
    deadband_m = _control_scalar(deadband_m, "recovery deadband", minimum=0)
    if observed is None:
        return 0.0, 0.0, 0.0
    current = _control_vector3(
        (observed.north_m, observed.east_m, observed.down_m), "recovery position"
    )
    target = _control_vector3(
        (target_setpoint.north_m, target_setpoint.east_m, target_setpoint.down_m),
        "recovery target",
    )
    delta = _control_vector3(tuple(target[i] - current[i] for i in range(3)), "recovery delta")
    distance_m = _control_scalar(math.hypot(*delta), "recovery distance", minimum=0)
    if distance_m <= deadband_m or distance_m <= 1e-12:
        return 0.0, 0.0, 0.0
    observed_velocity = _control_vector3(
        (observed.north_m_s, observed.east_m_s, observed.down_m_s),
        "recovery observed velocity",
    )
    velocity = _control_vector3(
        tuple(
            delta[index] * gain_s_inverse - observed_velocity[index] * velocity_damping
            for index in range(3)
        ),
        "recovery velocity",
    )
    speed_mps = _control_scalar(math.hypot(*velocity), "recovery speed", minimum=0)
    if speed_mps <= maximum_speed_mps or speed_mps <= 1e-12:
        return velocity
    ratio = maximum_speed_mps / speed_mps
    return tuple(component * ratio for component in velocity)


# 功能：
#   只有实际位移或净空改善才刷新局部修复停滞期限，独立的总时限仍由上层约束。
# 输入：
#   now：当前单调时刻。
#   observed：实际位置速度观测。
#   minimum_clearance_m：本轮预测最小净空。
#   stall_timeout_seconds：无进展允许等待时间。
#   state：上轮修复锚点和期限，首次为 None。
#   minimum_position_progress_m：有效位移阈值。
#   minimum_clearance_progress_m：有效净空改善阈值。
# 输出：
#   state：更新后的锚点、最佳净空和停滞期限。
def _advance_local_repair_progress(
    *,
    now: float,
    observed: Any | None,
    minimum_clearance_m: float,
    stall_timeout_seconds: float,
    state: dict[str, Any] | None,
    minimum_position_progress_m: float = 0.02,
    minimum_clearance_progress_m: float = 0.005,
) -> dict[str, Any]:
    """Renew a local detour's stall window only on measurable progress.

    A safe local replan can legitimately take longer than one fixed hold
    window under low-real-time-factor simulation.  Position movement or a
    better predicted clearance renews the short stall deadline; a separate
    absolute deadline still bounds the complete detour.
    """

    now = _control_scalar(now, "local repair clock")
    minimum_clearance_m = _control_scalar(minimum_clearance_m, "local repair clearance")
    if not all(
        finite_positive_number(value)
        for value in (
            stall_timeout_seconds,
            minimum_position_progress_m,
            minimum_clearance_progress_m,
        )
    ):
        raise ValueError("local repair progress policy must be finite and positive")
    next_deadline = _control_scalar(now + stall_timeout_seconds, "local repair deadline")
    if next_deadline <= now:
        raise ValueError("local repair deadline must advance the clock")
    if state is not None and not isinstance(state, dict):
        raise ValueError("local repair state must be an object")
    position = (
        (
            _control_scalar(observed.north_m, "repair north"),
            _control_scalar(observed.east_m, "repair east"),
            _control_scalar(observed.down_m, "repair down"),
        )
        if observed is not None
        else None
    )
    if position is not None and not all(math.isfinite(value) for value in position):
        raise UserDirectedLanding("local repair telemetry contained non-finite values")
    if not math.isfinite(minimum_clearance_m):
        raise UserDirectedLanding("local repair clearance contained a non-finite value")
    if state is None:
        return {
            "position_anchor_ned_m": position,
            "best_minimum_clearance_m": minimum_clearance_m,
            "clearance_progress_anchor_m": minimum_clearance_m,
            "stall_deadline": next_deadline,
            "checked_at_monotonic": now,
            "progress_revision": 0,
            "last_progress_evidence": "repair-started",
        }

    for field in ("stall_deadline", "best_minimum_clearance_m", "clearance_progress_anchor_m"):
        _control_scalar(state.get(field), f"local repair {field}")
    if (
        type(state.get("progress_revision")) is not int
        or state["progress_revision"] < 0
        or now < _control_scalar(state.get("checked_at_monotonic", now), "repair prior clock")
    ):
        raise ValueError("local repair state has invalid progress or clock")
    if state.get("position_anchor_ned_m") is not None:
        _control_vector3(state["position_anchor_ned_m"], "repair position anchor")
    state["checked_at_monotonic"] = now
    if now >= state["stall_deadline"]:
        return state
    progressed = False
    evidence: str | None = None
    anchor = state.get("position_anchor_ned_m")
    if position is not None and anchor is not None:
        if math.dist(position, tuple(anchor)) >= minimum_position_progress_m:
            state["position_anchor_ned_m"] = position
            progressed = True
            evidence = "vehicle-moved-along-local-repair"
    elif position is not None:
        state["position_anchor_ned_m"] = position
    best_clearance = float(state["best_minimum_clearance_m"])
    state["best_minimum_clearance_m"] = max(best_clearance, minimum_clearance_m)
    # Compare directional recovery against a recent anchor rather than the
    # all-time best.  A local escape can legitimately move closer to one
    # surface before gaining substantially more clearance on the other side.
    # Requiring it to exceed the pre-manoeuvre maximum incorrectly labels that
    # recovery as stalled.  The absolute repair deadline still prevents a
    # noisy back-and-forth motion from extending the manoeuvre indefinitely.
    clearance_anchor = float(state.get("clearance_progress_anchor_m", best_clearance))
    if minimum_clearance_m >= clearance_anchor + minimum_clearance_progress_m:
        state["clearance_progress_anchor_m"] = minimum_clearance_m
        progressed = True
        evidence = "predicted-clearance-increased"
    elif minimum_clearance_m <= clearance_anchor - minimum_clearance_progress_m:
        state["clearance_progress_anchor_m"] = minimum_clearance_m
    if progressed:
        state["stall_deadline"] = next_deadline
        state["progress_revision"] = int(state["progress_revision"]) + 1
        state["last_progress_evidence"] = evidence
    return state


# 功能：
#   发送保护位置及速度前馈并保留局部航向；带期限的控制缺少组合传输时拒绝降级。
# 输入：
#   base：速度设定值类型。
#   client：实际飞控适配器。
#   setpoint：安全位置目标及航向。
#   velocity_ned_mps：北、东、向下速度前馈。
#   deadline_unix_ms：原始输入授权期限，None 表示非运动授权保护路径。
# 输出：
#   accepted_at：组合接口返回的 UTC 毫秒时刻，兼容纯位置调用时为 None。
async def _send_position_with_velocity(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    velocity_ned_mps: tuple[float, float, float] | None,
    deadline_unix_ms: int | None = None,
) -> int | None:
    """Send a local safety setpoint without restoring the pre-arm heading."""

    velocity_type = getattr(base, "VelocitySetpoint", None)
    sender = getattr(client, "set_local_position_velocity_ned", None)
    if sender is None:
        sender = getattr(client, "set_position_velocity_ned", None)
    if velocity_ned_mps is None or velocity_type is None or not callable(sender):
        if deadline_unix_ms is not None:
            raise UserDirectedLanding("bounded safety position-velocity transport is unavailable")
        position_sender = getattr(client, "set_local_position_ned", None)
        if position_sender is None:
            position_sender = client.set_position_ned
        await position_sender(setpoint)
        return
    north_m_s, east_m_s, down_m_s = velocity_ned_mps
    if not all(math.isfinite(value) for value in velocity_ned_mps):
        raise UserDirectedLanding("velocity feedforward contained non-finite values")
    if (
        deadline_unix_ms is not None
        and deadline_unix_ms - int(time.time() * 1000) < LOCAL_TRANSPORT_BUDGET_MS
    ):
        raise ControlInputLeaseUnavailable("CONTROL_TRANSPORT_HAS_INSUFFICIENT_INPUT_LEASE")
    await sender(
        setpoint,
        velocity_type(
            north_m_s=north_m_s,
            east_m_s=east_m_s,
            down_m_s=down_m_s,
            yaw_deg=float(setpoint.yaw_deg),
        ),
    )
    return int(time.time() * 1000)


# 功能：
#   仅通过纯速度接口发送模型控制，发送前复核剩余输入期限，不悄悄引入有限位置目标。
# 输入：
#   base：速度设定值构造模块。
#   client：实际飞控客户端。
#   velocity_ned_mps：仲裁后的北、东、向下速度。
#   yaw_deg：仲裁后的航向。
#   deadline_unix_ms：命令授权失效的 UTC 毫秒时刻。
# 输出：
#   accepted_at：底层发送返回时刻，用于再次验证接收回执。
async def _send_model_velocity(
    *,
    base: ModuleType,
    client: Any,
    velocity_ned_mps: tuple[float, float, float],
    yaw_deg: float,
    deadline_unix_ms: int | None = None,
) -> int:
    """Send one model-authorized motion command using velocity control only.

    A combined position/velocity message is not equivalent: PX4 interprets a
    finite position as position control and uses velocity only as feed-forward.
    Requiring the dedicated method prevents a supposedly joystick-like local
    command from silently becoming another moving world-coordinate target.
    """

    velocity_type = getattr(base, "VelocitySetpoint", None)
    # Adapters with route/prearm heading policy must expose the explicit
    # model path so that the authorized yaw is not silently rewritten.
    sender = getattr(client, "set_model_velocity_ned", None)
    if sender is None:
        sender = getattr(client, "set_velocity_ned", None)
    if velocity_type is None or not callable(sender):
        raise UserDirectedLanding("velocity-only model control transport is unavailable")
    if not all(math.isfinite(value) for value in (*velocity_ned_mps, yaw_deg)):
        raise UserDirectedLanding("model velocity command contained non-finite values")
    if (
        deadline_unix_ms is not None
        and deadline_unix_ms - int(time.time() * 1000) < LOCAL_TRANSPORT_BUDGET_MS
    ):
        raise ControlInputLeaseUnavailable("CONTROL_TRANSPORT_HAS_INSUFFICIENT_INPUT_LEASE")
    north_m_s, east_m_s, down_m_s = velocity_ned_mps
    await sender(
        velocity_type(
            north_m_s=north_m_s,
            east_m_s=east_m_s,
            down_m_s=down_m_s,
            yaw_deg=yaw_deg,
        )
    )
    return int(time.time() * 1000)


# 功能：
#   实际成功发送仍有效控制后清除输入中断计时与保护位置，收到数据本身不能清零。
# 输入：
#   args：当前控制输入中断状态。
# 输出：
#   None：不返回业务数据。
def _clear_dispatch_input_gap(args: argparse.Namespace) -> None:
    """End recovery only after a real, still-valid dispatch (including a hold).

    Receiving a packet alone cannot reset this timer: repeated packets that all
    expire before dispatch must eventually trigger the bounded failure path.
    """
    args._dispatch_input_gap_started_at = None
    args._dispatch_input_hold_setpoint = None


# 功能：
#   检查动态导航目标仍有效；静态目标不设此期限，坏时间戳或到期目标不能授权继续运动。
# 输入：
#   deadline：单调时钟的目标截止时刻，None 表示静态目标。
#   now：同一单调时钟的当前时刻。
# 输出：
#   None：不返回业务数据。
def _require_live_navigation_goal(deadline: float | None, now: float) -> None:
    if deadline is not None and (
        type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or not math.isfinite(now)
        or now >= deadline
    ):
        raise UserDirectedLanding("DYNAMIC_NAVIGATION_GOAL_EXPIRED")


# 功能：
#   1. 在发送运动指令前复核动态目标期限与原始观测期限。
#   2. 尚未发送且观测已失效时按实测位置制动，保留有界恢复窗口，不推进任务。
#   3. 已发送指令的返回时间交由接收回执复核，不能把迟到发送伪装为未发送。
# 输入：
#   args：控制上下文与恢复状态。
#   base：飞控基础类型及传输辅助模块。
#   client：实际飞控传输客户端。
#   setpoint：已仲裁的位置及航向设定值。
#   velocity_ned_mps：已仲裁的北、东、向下速度。
#   command：绑定原始观测期限的控制合同。
#   coordinate_contract：遥测、控制及世界坐标转换合同。
#   phase_path：当前控制阶段证据路径。
#   position_control：是否使用位置与速度前馈接口，否则使用纯速度接口。
#   navigation_goal_deadline_monotonic：动态目标截止时刻，静态目标为 None。
# 输出：
#   accepted_at：传输接收时刻的 UTC 毫秒数，保护性制动而未推进任务时为 None。
async def _dispatch_motion_or_brake(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    velocity_ned_mps: tuple[float, float, float],
    command: RuntimeLocalSafetyCommand,
    coordinate_contract: Px4CoordinateContract,
    phase_path: Path,
    position_control: bool = False,
    navigation_goal_deadline_monotonic: float | None = None,
) -> int | None:
    _require_live_navigation_goal(
        navigation_goal_deadline_monotonic,
        asyncio.get_running_loop().time(),
    )
    try:
        if position_control:
            accepted_at = await _send_position_with_velocity(
                base=base,
                client=client,
                setpoint=setpoint,
                velocity_ned_mps=velocity_ned_mps,
                deadline_unix_ms=command.valid_until_unix_ms,
            )
        else:
            accepted_at = await _send_model_velocity(
                base=base,
                client=client,
                velocity_ned_mps=velocity_ned_mps,
                yaw_deg=float(setpoint.yaw_deg),
                deadline_unix_ms=command.valid_until_unix_ms,
            )
    except ControlInputLeaseUnavailable:
        now = asyncio.get_running_loop().time()
        gap_started = getattr(args, "_dispatch_input_gap_started_at", None)
        if gap_started is None:
            gap_started = args._dispatch_input_gap_started_at = now
        grace = min(8.0, float(getattr(args, "local_safety_runtime_stale_grace_seconds", 8.0)))
        if not math.isfinite(grace) or grace <= 0 or now - gap_started > grace:
            raise UserDirectedLanding("CONTROL_INPUT_LEASE_GAP_EXCEEDED") from None
        observed = await _refresh_px4_identity_telemetry(
            args=args, client=client, coordinate_contract=coordinate_contract
        )
        if observed is None:
            raise UserDirectedLanding("CONTROL_INPUT_BRAKE_REQUIRES_FRESH_TELEMETRY") from None
        values = [
            getattr(observed, k, math.nan)
            for k in ("north_m", "east_m", "down_m", "north_m_s", "east_m_s", "down_m_s")
        ]
        if not all(math.isfinite(v) for v in values):
            raise UserDirectedLanding("CONTROL_INPUT_BRAKE_TELEMETRY_INVALID") from None
        speed = math.sqrt(sum(v * v for v in values[3:]))
        heading = _local_hold_yaw(
            args=args,
            client=client,
            fallback_heading_deg=float(setpoint.yaw_deg),
            model_control_required=True,
        )
        hold = getattr(args, "_dispatch_input_hold_setpoint", None)
        # Never pull a moving vehicle back toward an earlier braking position.
        if hold is None or speed > _MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS:
            hold = base.Setpoint(
                north_m=values[0], east_m=values[1], down_m=values[2], yaw_deg=heading
            )
            args._dispatch_input_hold_setpoint = (
                hold if speed <= _MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS else None
            )
        await _send_position_with_velocity(
            base=base, client=client, setpoint=hold, velocity_ned_mps=(0.0, 0.0, 0.0)
        )
        args._last_model_control_authorized = False
        _record_local_safety_executor_event(
            args,
            status="input-lease-discarded-before-dispatch",
            details={
                "command_sequence": command.observation_sequence,
                "command_valid_until_unix_ms": command.valid_until_unix_ms,
                "expired_motion_sent": False,
                "schedule_advancement_authorized": False,
                "brake_position_ned_m": [hold.north_m, hold.east_m, hold.down_m],
                "brake_velocity_ned_mps": [0.0, 0.0, 0.0],
                "brake_yaw_deg": hold.yaw_deg,
                "observed_speed_mps": speed,
                "gap_elapsed_seconds": now - gap_started,
                "maximum_gap_seconds": grace,
            },
        )
        _publish_local_control_phase(
            phase_path,
            local_phase="PERCEPTION_REFRESH_HOLD",
            details={
                "reason": "input-lease-discarded-before-dispatch",
                "schedule_advancement_authorized": False,
            },
        )
        return None
    # A late adapter return must still reach the receipt/failure check, without
    # pretending that usable input has resumed in the meantime.
    if accepted_at is not None and accepted_at <= command.valid_until_unix_ms:
        _clear_dispatch_input_gap(args)
    return accepted_at


# 功能：
#   维持统一控制周期，包含辅助函数之间的耗时，拒绝飞行期间悄悄变更控制频率。
# 输入：
#   args：本次频率配置和实际节拍器。
# 输出：
#   None：不返回业务数据。
async def _begin_local_control_tick(args: argparse.Namespace) -> None:
    """Include work between helper calls in the same control period."""
    pacer = getattr(args, "_local_control_pacer", None)
    if pacer is None:
        pacer = ControlTickPacer(args.setpoint_rate_hz)
        args._local_control_pacer = pacer
    elif pacer.rate_hz != args.setpoint_rate_hz:
        raise UserDirectedLanding("CONTROL_CADENCE_CHANGED_DURING_FLIGHT")
    await pacer.wait()


# 功能：
#   仅读取过期命令的期限用于诊断生产端中断，返回对象不能用于继续运动。
# 输入：
#   args：所需安全命令的持久证据路径。
# 输出：
#   stale：可解析的历史命令，读取失败时为 None。
def _read_stale_local_safety_command(
    args: argparse.Namespace,
) -> RuntimeLocalSafetyCommand | None:
    """Read an expired command only to recover its absolute freshness deadline.

    An expired command is never applied.  It lets the executor distinguish a
    transient producer jitter from a dead producer while holding the measured
    current PX4 position in both cases.
    """

    if args.local_safety_command is None:
        raise UserDirectedLanding("required local safety command path is unavailable")
    try:
        return RuntimeLocalSafetyCommand.model_validate(
            read_runtime_object(args.local_safety_command)
        )
    except (OSError, ValueError):
        return None


# 功能：
#   解释命令与观测为何不可用，记录解析、序号、摘要及时效差异，但不提供飞行许可。
# 输入：
#   args：当前命令与观测文件路径。
# 输出：
#   diagnostic：配对状态和错误原因字典。
def _local_safety_pair_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Explain why a command pair is unavailable without authorizing it."""

    diagnostic: dict[str, Any] = {"checked_at_unix_ms": int(time.time() * 1_000)}
    try:
        command = RuntimeLocalSafetyCommand.model_validate(
            read_runtime_object(args.local_safety_command)
        )
        diagnostic.update(
            command_parse="accepted",
            command_sequence=command.observation_sequence,
            command_valid_until_unix_ms=command.valid_until_unix_ms,
            command_age_after_expiry_seconds=max(
                0.0,
                (diagnostic["checked_at_unix_ms"] - command.valid_until_unix_ms) / 1_000.0,
            ),
            command_tracking_recovery_active=command.tracking_recovery_active,
        )
    except (OSError, ValueError) as error:
        diagnostic["command_parse"] = f"{type(error).__name__}:{error}"
        return diagnostic
    try:
        observation = RuntimeLocalSafetyObservation.model_validate(
            read_runtime_object(args.local_safety_observation)
        )
        diagnostic.update(
            observation_parse="accepted",
            observation_sequence=observation.sequence,
            sequence_matches=command.observation_sequence == observation.sequence,
            hash_matches=command.observation_sha256 == sha256_json(observation),
            observation_stream_healthy=observation.stream_healthy,
        )
    except (AttributeError, OSError, ValueError) as error:
        diagnostic["observation_parse"] = f"{type(error).__name__}:{error}"
    return diagnostic


# 功能：
#   对重复安全事件节流后入队，避免实时控制期限内同步写历史文件。
# 输入：
#   args：本次证据队列及上一事件签名。
#   status：安全事件状态。
#   details：来源序号、期限或恢复上下文。
# 输出：
#   None：不返回业务数据。
def _record_local_safety_executor_event(
    args: argparse.Namespace,
    *,
    status: str,
    details: dict[str, Any],
) -> None:
    """Queue a throttled trace without filesystem work in the live action lease."""

    run_dir = getattr(args, "run_dir", None)
    if run_dir is None:
        return
    now_unix_ms = int(time.time() * 1_000)
    signature = (
        status,
        details.get("command_sequence"),
        details.get("observation_sequence"),
        details.get("tracking_recovery_active"),
    )
    if (
        getattr(args, "_local_safety_event_signature", None) == signature
        and now_unix_ms - getattr(args, "_local_safety_event_at_unix_ms", 0) < 500
    ):
        return
    args._local_safety_event_signature = signature
    args._local_safety_event_at_unix_ms = now_unix_ms
    path = Path(run_dir) / "runtime-state" / "local-safety-executor-history.jsonl"
    writer = getattr(args, "_control_application_writer", None)
    if writer is None:
        raise RuntimeError("EXECUTOR_EVIDENCE_WRITER_NOT_STARTED")
    writer.submit(path, {"recorded_at_unix_ms": now_unix_ms, "status": status, **details})


# 功能：
#   判断当前安全命令是否适用于当前目标及恢复模式，拒绝沿用上一目标的有效期限。
# 输入：
#   command：已解析的安全命令。
#   planned_setpoint：当前参考目标。
#   coordinate_contract：地图变换合同。
#   tracking_recovery_active：当前是否处于固定目标恢复。
#   navigation_goal_id：当前语义目标身份。
# 输出：
#   compatible：命令与本轮控制上下文是否相容。
def _local_safety_command_matches_control_context(
    *,
    command: RuntimeLocalSafetyCommand,
    planned_setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
    tracking_recovery_active: bool,
    navigation_goal_id: str | None = None,
) -> bool:
    """Reject a valid lease authored for the previous controller mode/target."""

    diagnostic = _local_safety_control_context_diagnostic(
        command=command,
        planned_setpoint=planned_setpoint,
        coordinate_contract=coordinate_contract,
        tracking_recovery_active=tracking_recovery_active,
        navigation_goal_id=navigation_goal_id,
    )
    return bool(diagnostic["compatible"])


# 功能：
#   核对目标身份及模式相容性；恢复用许可不能授权更快巡航，模型路径恢复仍绑定其自身目标。
# 输入：
#   command：本轮安全许可。
#   planned_setpoint：当前参考目标。
#   coordinate_contract：参考目标到地图的变换。
#   tracking_recovery_active：执行器当前恢复模式。
#   navigation_goal_id：执行器当前目标身份。
# 输出：
#   diagnostic：是否相容、原因及目标偏差等诊断。
def _local_safety_control_context_diagnostic(
    *,
    command: RuntimeLocalSafetyCommand,
    planned_setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
    tracking_recovery_active: bool,
    navigation_goal_id: str | None = None,
) -> dict[str, Any]:
    """Explain the asymmetric controller-mode lease compatibility decision."""

    command_recovery_active = bool(getattr(command, "tracking_recovery_active", False))
    command_goal_id = getattr(command, "navigation_goal_id", None)
    if navigation_goal_id is not None and command_goal_id != navigation_goal_id:
        return {
            "compatible": False,
            "context_reason": (
                "navigation-goal-proof-missing"
                if command_goal_id is None
                else "navigation-goal-epoch-mismatch"
            ),
            "command_navigation_goal_id": command_goal_id,
            "executor_navigation_goal_id": navigation_goal_id,
            "command_tracking_recovery_active": command_recovery_active,
            "executor_tracking_recovery_active": tracking_recovery_active,
            "evaluated_target_distance_m": None,
        }
    # Recovery is a strict subset of the ordinary route envelope: the local
    # planner caps its speed at 0.2 m/s and the executor applies an even slower
    # radial rejoin command.  A command proven safe for ordinary tracking may
    # therefore be consumed while *entering* recovery when it is bound to the
    # same target.  The reverse is unsafe: a recovery-only verdict did not
    # evaluate the faster ordinary command, so leaving recovery must wait for
    # a fresh ordinary-tracking lease.
    if command_recovery_active and not tracking_recovery_active:
        return {
            "compatible": False,
            "context_reason": "recovery-proof-cannot-authorize-ordinary-tracking",
            "command_tracking_recovery_active": command_recovery_active,
            "executor_tracking_recovery_active": tracking_recovery_active,
            "evaluated_target_distance_m": None,
        }
    if not tracking_recovery_active:
        return {
            "compatible": True,
            "context_reason": "ordinary-tracking-mode-compatible",
            "command_tracking_recovery_active": command_recovery_active,
            "executor_tracking_recovery_active": tracking_recovery_active,
            "evaluated_target_distance_m": None,
        }
    if getattr(command, "navigation_control_authority", "route-fallback") == "model-required":
        # A model-required command is bound to its hash-validated model path,
        # not to the dense fallback schedule point. Recovery must keep using
        # that model-selected local target instead of silently regaining route
        # authority.
        return {
            "compatible": True,
            "context_reason": "model-authority-remains-hash-bound-during-recovery",
            "command_tracking_recovery_active": command_recovery_active,
            "executor_tracking_recovery_active": tracking_recovery_active,
            "evaluated_target_distance_m": None,
        }
    evaluated_target = getattr(command, "evaluated_target_position_m", None)
    if evaluated_target is None:
        return {
            "compatible": False,
            "context_reason": "recovery-target-proof-missing",
            "command_tracking_recovery_active": command_recovery_active,
            "executor_tracking_recovery_active": tracking_recovery_active,
            "evaluated_target_distance_m": None,
        }
    planned_world = _setpoint_world_enu(planned_setpoint, coordinate_contract)
    target_distance_m = math.dist(
        (evaluated_target.x, evaluated_target.y, evaluated_target.z),
        (planned_world.x, planned_world.y, planned_world.z),
    )
    compatible = target_distance_m <= 0.02
    return {
        "compatible": compatible,
        "context_reason": (
            "same-target-proof-compatible-with-recovery"
            if compatible
            else "recovery-target-proof-drifted"
        ),
        "command_tracking_recovery_active": command_recovery_active,
        "executor_tracking_recovery_active": tracking_recovery_active,
        "evaluated_target_distance_m": target_distance_m,
    }


# 功能：
#   在必须由模型控制的运行中拒绝普通路线回退许可，不能用兼容命令冒充模型授权。
# 输入：
#   args：本次运行的控制权限要求。
#   command：收到的安全命令。
# 输出：
#   authority_matches：命令的控制权限类型是否满足要求。
def _local_safety_command_matches_required_authority(
    *,
    args: argparse.Namespace,
    command: RuntimeLocalSafetyCommand,
) -> bool:
    """Reject compatibility commands when this run requires model-owned control."""

    return not bool(getattr(args, "require_model_control_authority", False)) or (
        command.navigation_control_authority == "model-required"
    )


# 功能：
#   1. 结合实测状态、目标身份和安全命令驱动每个控制周期，模型运动走纯速度接口。
#   2. 输入过期、权限缺失或目标错配时保护悬停，不推进任务；局部修复始终有停滞与总时限。
#   3. 只有明确配置的非模型兼容模式才允许路线位置控制，安全制动不冒充模型自主运动。
# 输入：
#   args：控制通道、频率、权限、保护时限和当前状态。
#   base：飞控类型和基础函数。
#   client：实际飞控与遥测客户端。
#   planned_setpoint：本轮参考目标。
#   coordinate_contract：地图和飞控坐标绑定。
#   phase_path：执行阶段证据路径。
#   navigation_goal_position_m：当前语义目标位置。
#   navigation_goal_id：当前目标身份。
#   control_profile：巡航或精细控制模式。
#   action_checkpoint_goal：目标是否含必须完成的动作。
#   planned_velocity_ned_mps：明确兼容模式的路线速度前馈。
#   tracking_recovery_active：是否请求固定目标恢复控制。
#   decision_trigger：专家需要处理的触发原因。
#   recovery_episode_id：当前恢复事件身份。
#   navigation_goal_deadline_monotonic：动态目标的绝对单调期限。
# 输出：
#   applied_setpoint：本轮实际应用的设定值，任务推进还需上层实测进度检查。
async def _apply_local_safety(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    planned_setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
    phase_path: Path,
    navigation_goal_position_m: Vector3 | None = None,
    navigation_goal_id: str | None = None,
    control_profile: str = "cruise",
    action_checkpoint_goal: bool = False,
    planned_velocity_ned_mps: tuple[float, float, float] | None = None,
    tracking_recovery_active: bool = False,
    decision_trigger: str | None = None,
    recovery_episode_id: str | None = None,
    navigation_goal_deadline_monotonic: float | None = None,
) -> Any:
    """Pause schedule advancement while a short-lived repair command is active."""

    _require_live_navigation_goal(
        navigation_goal_deadline_monotonic,
        asyncio.get_running_loop().time(),
    )
    _publish_local_safety_target(
        path=args.local_safety_target,
        setpoint=planned_setpoint,
        coordinate_contract=coordinate_contract,
        navigation_goal_position_m=navigation_goal_position_m,
        navigation_goal_id=navigation_goal_id,
        control_profile=control_profile,
        action_checkpoint_goal=action_checkpoint_goal,
        tracking_recovery_active=tracking_recovery_active,
        decision_trigger=decision_trigger,
        recovery_episode_id=recovery_episode_id,
    )
    if args.local_safety_command is None:
        await _send_position_with_velocity(
            base=base,
            client=client,
            setpoint=planned_setpoint,
            velocity_ned_mps=planned_velocity_ned_mps,
        )
        await asyncio.sleep(1.0 / args.setpoint_rate_hz)
        return planned_setpoint
    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.local_safety_repair_timeout_seconds
    repair_absolute_deadline = loop.time() + float(
        getattr(args, "local_safety_repair_absolute_timeout_seconds", 60.0)
    )
    repair_progress_state: dict[str, Any] | None = None
    command_startup_deadline = loop.time() + getattr(
        args, "local_safety_command_grace_seconds", 8.0
    )
    waited_for_first_command = False
    unavailable_hold_setpoint: Any | None = None
    repair_hold_setpoint: Any | None = None
    context_mismatch_started_at: float | None = None
    command_unavailable_started_at: float | None = None
    while True:
        await _begin_local_control_tick(args)
        _require_live_navigation_goal(navigation_goal_deadline_monotonic, loop.time())
        observed = await _refresh_px4_identity_telemetry(
            args=args,
            client=client,
            coordinate_contract=coordinate_contract,
        )
        _require_live_navigation_goal(navigation_goal_deadline_monotonic, loop.time())
        # _tracking_gate_tick immediately follows this function. Reuse this
        # same fresh sample instead of awaiting the next PX4 telemetry frame a
        # second time in one nominal control period.
        args._last_px4_control_observation = observed
        command = _read_local_safety_command(args)
        if command is not None and not _local_safety_command_matches_required_authority(
            args=args,
            command=command,
        ):
            if context_mismatch_started_at is None:
                context_mismatch_started_at = loop.time()
            _record_local_safety_executor_event(
                args,
                status="model-authority-contract-mismatch",
                details={
                    **_local_safety_pair_diagnostic(args),
                    "required_navigation_control_authority": "model-required",
                    "observed_navigation_control_authority": (command.navigation_control_authority),
                    "tracking_recovery_active": tracking_recovery_active,
                },
            )
            context_grace_seconds = getattr(args, "local_safety_runtime_stale_grace_seconds", 8.0)
            if loop.time() - context_mismatch_started_at > context_grace_seconds:
                raise UserDirectedLanding(
                    "required local safety command did not provide model control authority"
                )
            command = None
        elif command is not None and not _local_safety_command_matches_control_context(
            command=command,
            planned_setpoint=planned_setpoint,
            coordinate_contract=coordinate_contract,
            tracking_recovery_active=tracking_recovery_active,
            navigation_goal_id=navigation_goal_id,
        ):
            if context_mismatch_started_at is None:
                context_mismatch_started_at = loop.time()
            _record_local_safety_executor_event(
                args,
                status="context-mismatch",
                details={
                    **_local_safety_pair_diagnostic(args),
                    **_local_safety_control_context_diagnostic(
                        command=command,
                        planned_setpoint=planned_setpoint,
                        coordinate_contract=coordinate_contract,
                        tracking_recovery_active=tracking_recovery_active,
                        navigation_goal_id=navigation_goal_id,
                    ),
                    "tracking_recovery_active": tracking_recovery_active,
                },
            )
            context_grace_seconds = getattr(args, "local_safety_runtime_stale_grace_seconds", 8.0)
            if loop.time() - context_mismatch_started_at > context_grace_seconds:
                raise UserDirectedLanding(
                    "required local safety command context remained mismatched"
                )
            command = None
        elif command is not None:
            context_mismatch_started_at = None
            command_unavailable_started_at = None
            args._local_safety_command_established = True
            _record_local_safety_executor_event(
                args,
                status="accepted",
                details={
                    "command_sequence": getattr(command, "observation_sequence", None),
                    "command_valid_until_unix_ms": getattr(command, "valid_until_unix_ms", None),
                    "tracking_recovery_active": tracking_recovery_active,
                    "action": command.decision.action,
                    "navigation_control_authority": getattr(
                        command, "navigation_control_authority", "route-fallback"
                    ),
                    "model_navigation_authorized": bool(
                        getattr(command, "model_navigation_authorized", False)
                    ),
                    "model_call_id": getattr(command, "model_call_id", None),
                    "model_selected_candidate_id": getattr(
                        command, "model_selected_candidate_id", None
                    ),
                },
            )
        if command is None:
            if getattr(args, "local_safety_required", False):
                command_file_present = args.local_safety_command.is_file()
                if command_file_present:
                    if command_unavailable_started_at is None:
                        command_unavailable_started_at = loop.time()
                    runtime_grace_seconds = getattr(
                        args, "local_safety_runtime_stale_grace_seconds", 8.0
                    )
                    stale = _read_stale_local_safety_command(args)
                    pair_diagnostic = _local_safety_pair_diagnostic(args)
                    pair_diagnostic.update(
                        tracking_recovery_active=tracking_recovery_active,
                        unavailable_elapsed_seconds=(loop.time() - command_unavailable_started_at),
                        runtime_grace_seconds=runtime_grace_seconds,
                        latched_hold_active=unavailable_hold_setpoint is not None,
                    )
                    _record_local_safety_executor_event(
                        args,
                        status="command-unavailable",
                        details=pair_diagnostic,
                    )
                    if stale is None:
                        if loop.time() - command_unavailable_started_at > runtime_grace_seconds:
                            raise UserDirectedLanding(
                                "required local safety command remained unreadable"
                            )
                        if observed is None:
                            raise UserDirectedLanding(
                                "required local safety command is unreadable without PX4 telemetry"
                            )
                        if unavailable_hold_setpoint is None:
                            unavailable_hold_setpoint = base.Setpoint(
                                north_m=observed.north_m,
                                east_m=observed.east_m,
                                down_m=observed.down_m,
                                yaw_deg=_local_hold_yaw(
                                    args=args,
                                    client=client,
                                    fallback_heading_deg=planned_setpoint.yaw_deg,
                                ),
                            )
                        await _send_position_with_velocity(
                            base=base,
                            client=client,
                            setpoint=unavailable_hold_setpoint,
                            velocity_ned_mps=(0.0, 0.0, 0.0),
                        )
                        continue
                    stale_age_seconds = max(
                        0.0,
                        (int(time.time() * 1_000) - stale.valid_until_unix_ms) / 1_000.0,
                    )
                    if (
                        stale_age_seconds > runtime_grace_seconds
                        or loop.time() - command_unavailable_started_at > runtime_grace_seconds
                    ):
                        raise UserDirectedLanding(
                            "required local safety command remained stale beyond grace"
                        )
                    if observed is None:
                        raise UserDirectedLanding(
                            "required local safety command is stale without PX4 telemetry"
                        )
                    if unavailable_hold_setpoint is None:
                        unavailable_hold_setpoint = base.Setpoint(
                            north_m=observed.north_m,
                            east_m=observed.east_m,
                            down_m=observed.down_m,
                            yaw_deg=_local_hold_yaw(
                                args=args,
                                client=client,
                                fallback_heading_deg=planned_setpoint.yaw_deg,
                            ),
                        )
                    _publish_local_control_phase(
                        phase_path,
                        local_phase="PERCEPTION_REFRESH_HOLD",
                        details={
                            "local_safety_action": "hold",
                            "stale_age_seconds": stale_age_seconds,
                            "schedule_advancement_authorized": False,
                        },
                    )
                    await _send_position_with_velocity(
                        base=base,
                        client=client,
                        setpoint=unavailable_hold_setpoint,
                        velocity_ned_mps=(0.0, 0.0, 0.0),
                    )
                    continue
                if getattr(args, "_local_safety_command_established", False):
                    if command_unavailable_started_at is None:
                        command_unavailable_started_at = loop.time()
                    runtime_grace_seconds = getattr(
                        args, "local_safety_runtime_stale_grace_seconds", 8.0
                    )
                    missing_elapsed_seconds = loop.time() - command_unavailable_started_at
                    pair_diagnostic = _local_safety_pair_diagnostic(args)
                    pair_diagnostic.update(
                        tracking_recovery_active=tracking_recovery_active,
                        unavailable_elapsed_seconds=missing_elapsed_seconds,
                        runtime_grace_seconds=runtime_grace_seconds,
                        latched_hold_active=unavailable_hold_setpoint is not None,
                    )
                    _record_local_safety_executor_event(
                        args,
                        status="command-missing",
                        details=pair_diagnostic,
                    )
                    if missing_elapsed_seconds > runtime_grace_seconds:
                        raise UserDirectedLanding(
                            "required local safety command remained missing beyond grace"
                        )
                    if observed is None:
                        raise UserDirectedLanding(
                            "required local safety command is missing without PX4 telemetry"
                        )
                    observed_speed_mps = math.sqrt(
                        float(getattr(observed, "north_m_s", 0.0)) ** 2
                        + float(getattr(observed, "east_m_s", 0.0)) ** 2
                        + float(getattr(observed, "down_m_s", 0.0)) ** 2
                    )
                    braking_to_rest = (
                        unavailable_hold_setpoint is None
                        and observed_speed_mps > _MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS
                    )
                    missing_hold_setpoint = unavailable_hold_setpoint
                    if missing_hold_setpoint is None:
                        missing_hold_setpoint = base.Setpoint(
                            north_m=observed.north_m,
                            east_m=observed.east_m,
                            down_m=observed.down_m,
                            yaw_deg=_local_hold_yaw(
                                args=args,
                                client=client,
                                fallback_heading_deg=planned_setpoint.yaw_deg,
                            ),
                        )
                        if not braking_to_rest:
                            unavailable_hold_setpoint = missing_hold_setpoint
                    _publish_local_control_phase(
                        phase_path,
                        local_phase="PERCEPTION_REFRESH_HOLD",
                        details={
                            "command_file_present": False,
                            "hold_source": (
                                "px4-measured-braking-position"
                                if braking_to_rest
                                else "px4-measured-position"
                            ),
                            "hold_latched_for_command_gap": not braking_to_rest,
                            "hold_observed_speed_mps": observed_speed_mps,
                            "schedule_advancement_authorized": False,
                        },
                    )
                    await _send_position_with_velocity(
                        base=base,
                        client=client,
                        setpoint=missing_hold_setpoint,
                        velocity_ned_mps=(0.0, 0.0, 0.0),
                    )
                    continue
                if loop.time() >= command_startup_deadline:
                    raise UserDirectedLanding("required local safety command was not established")
                if observed is None:
                    raise UserDirectedLanding(
                        "required local safety command is unavailable without PX4 telemetry"
                    )
                if unavailable_hold_setpoint is None:
                    unavailable_hold_setpoint = base.Setpoint(
                        north_m=observed.north_m,
                        east_m=observed.east_m,
                        down_m=observed.down_m,
                        yaw_deg=_local_hold_yaw(
                            args=args, client=client, fallback_heading_deg=planned_setpoint.yaw_deg
                        ),
                    )
                _publish_local_control_phase(
                    phase_path,
                    local_phase="PERCEPTION_STARTUP_HOLD",
                    details={
                        "schedule_advancement_authorized": False,
                    },
                )
                await _send_position_with_velocity(
                    base=base,
                    client=client,
                    setpoint=unavailable_hold_setpoint,
                    velocity_ned_mps=(0.0, 0.0, 0.0),
                )
                continue
            if not waited_for_first_command:
                waited_for_first_command = True
                await _send_position_with_velocity(
                    base=base,
                    client=client,
                    setpoint=planned_setpoint,
                    velocity_ned_mps=planned_velocity_ned_mps,
                )
                continue
            await _send_position_with_velocity(
                base=base,
                client=client,
                setpoint=planned_setpoint,
                velocity_ned_mps=planned_velocity_ned_mps,
            )
            return planned_setpoint
        action = command.decision.action
        model_control_required = (
            getattr(command, "navigation_control_authority", "route-fallback") == "model-required"
        )
        model_control_authorized = bool(getattr(command, "model_navigation_authorized", False))
        args._last_model_control_required = model_control_required
        args._last_model_control_authorized = model_control_authorized
        args._last_model_call_id = getattr(command, "model_call_id", None)
        args._last_model_selected_candidate_id = getattr(
            command, "model_selected_candidate_id", None
        )
        args._last_model_path_sha256 = getattr(command, "model_path_sha256", None)
        if not (model_control_required and not model_control_authorized):
            # A later authorization (or non-model mode) begins a new control
            # epoch. Any future authority gap must latch the then-current PX4
            # position instead of reusing an older hold point.
            args._model_authority_hold_setpoint = None
        unavailable_hold_setpoint = None
        estimator_offset = getattr(
            command,
            "estimator_to_world_position_offset_m",
            Vector3(x=0.0, y=0.0, z=0.0),
        )
        # Persist the coordinate transform used for this exact control tick so
        # the tracking gate compares the route in the Gazebo/world frame, not
        # the deliberately shifted PX4 estimator frame.
        args._last_estimator_to_world_position_offset_m = estimator_offset
        if getattr(args, "simulation_teacher_control", False) and action in {"continue", "slow"}:
            if model_control_required or command.requested_control_intent is not None:
                raise UserDirectedLanding("simulation teacher cannot replace model authority")
            if any(abs(value) > 1e-9 for value in estimator_offset.model_dump().values()):
                raise UserDirectedLanding("simulation teacher cannot use fitted truth offsets")
            teacher_setpoint = _setpoint_with_model_body_yaw(
                base=base,
                args=args,
                setpoint=planned_setpoint,
                command=command,
            )
            selected = command.decision.selected_velocity_mps
            velocity_ned = selected.y, selected.x, -selected.z
            accepted_at = await _dispatch_motion_or_brake(
                args=args,
                base=base,
                client=client,
                setpoint=teacher_setpoint,
                velocity_ned_mps=velocity_ned,
                command=command,
                coordinate_contract=coordinate_contract,
                phase_path=phase_path,
                navigation_goal_deadline_monotonic=navigation_goal_deadline_monotonic,
            )
            if accepted_at is None:
                continue
            _record_model_control_application(
                args,
                command,
                velocity_ned_mps=velocity_ned,
                yaw_deg=float(teacher_setpoint.yaw_deg),
                transport="velocity-ned",
                accepted_at_unix_ms=accepted_at,
            )
            # Clear the temporary hold only after acceptance and its deadline
            # check. Disk writes before dispatch could consume the input lease.
            _clear_local_control_phase(phase_path)
            return teacher_setpoint
        if action == "continue" and not model_control_required:
            _clear_local_control_phase(phase_path)
            planned_world = _setpoint_world_enu(planned_setpoint, coordinate_contract)
            corrected_world = Vector3(
                x=planned_world.x - estimator_offset.x,
                y=planned_world.y - estimator_offset.y,
                z=planned_world.z - estimator_offset.z,
            )
            corrected_setpoint = _world_enu_setpoint(
                base=base,
                world=corrected_world,
                yaw_deg=float(planned_setpoint.yaw_deg),
                coordinate_contract=coordinate_contract,
            )
            # Recovery owns a fixed absolute position target.  Never reuse the
            # acceleration-limited planner candidate here: its tangential
            # component can make a fast vehicle circle the target.  A small
            # radial controller instead removes PX4's centimetre-scale hover
            # residual while decaying to zero at the target.  Safety-authored
            # velocities remain authoritative for slow/hold/replan below.
            recovery_velocity_ned_mps = (
                _recovery_position_velocity_feedforward(
                    observed=observed,
                    target_setpoint=corrected_setpoint,
                    gain_s_inverse=float(
                        getattr(args, "tracking_recovery_assist_gain_s_inverse", 0.8)
                    ),
                    velocity_damping=float(
                        getattr(args, "tracking_recovery_assist_velocity_damping", 1.0)
                    ),
                    maximum_speed_mps=float(
                        getattr(args, "tracking_recovery_assist_max_speed_mps", 0.08)
                    ),
                    deadband_m=float(getattr(args, "tracking_recovery_assist_deadband_m", 0.005)),
                )
                if tracking_recovery_active
                else planned_velocity_ned_mps
            )
            args._last_applied_velocity_feedforward_ned_mps = recovery_velocity_ned_mps
            await _send_position_with_velocity(
                base=base,
                client=client,
                setpoint=corrected_setpoint,
                velocity_ned_mps=recovery_velocity_ned_mps,
            )
            return corrected_setpoint
        corrected_command_world = Vector3(
            x=command.command_position_m.x - estimator_offset.x,
            y=command.command_position_m.y - estimator_offset.y,
            z=command.command_position_m.z - estimator_offset.z,
        )
        safe_setpoint = _world_enu_setpoint(
            base=base,
            world=corrected_command_world,
            yaw_deg=float(planned_setpoint.yaw_deg),
            coordinate_contract=coordinate_contract,
        )
        if model_control_required and not model_control_authorized:
            if observed is None:
                raise UserDirectedLanding(
                    "model authority is unavailable without live PX4 hold telemetry"
                )
            # Fail closed in the estimator frame that PX4 actually controls.
            # Reconstructing this hold from a Gazebo pose plus a continually
            # changing identity correction can turn harmless estimator jitter
            # into a staircase of position commands during a multi-second model
            # call.  A measured NED hold is invariant to that correction and
            # keeps the aircraft fixed until a revalidated model path exists.
            observed_speed_mps = math.sqrt(
                float(getattr(observed, "north_m_s", 0.0)) ** 2
                + float(getattr(observed, "east_m_s", 0.0)) ** 2
                + float(getattr(observed, "down_m_s", 0.0)) ** 2
            )
            latched_model_hold = getattr(args, "_model_authority_hold_setpoint", None)
            braking_to_rest = (
                latched_model_hold is None
                and observed_speed_mps > _MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS
            )
            if latched_model_hold is None:
                latched_model_hold = base.Setpoint(
                    north_m=observed.north_m,
                    east_m=observed.east_m,
                    down_m=observed.down_m,
                    yaw_deg=_local_hold_yaw(
                        args=args,
                        client=client,
                        fallback_heading_deg=planned_setpoint.yaw_deg,
                        model_control_required=True,
                    ),
                )
                # A fixed position captured while the aircraft still has
                # material velocity makes PX4 brake, overshoot, and then pull
                # back toward a stale point.  In a confined workspace that
                # return arc can cross furniture even though the model has
                # revoked motion.  While braking, publish the latest measured
                # estimator position with zero velocity on every control tick;
                # once speed is low, latch that stopped position for drift
                # rejection until a new model authorization begins.
                if not braking_to_rest:
                    args._model_authority_hold_setpoint = latched_model_hold
            safe_setpoint = latched_model_hold
            args._model_authority_hold_applied_count = (
                getattr(args, "_model_authority_hold_applied_count", 0) + 1
            )
            _publish_local_control_phase(
                phase_path,
                local_phase="MODEL_AUTHORITY_HOLD",
                details={
                    "model_authority_reason": getattr(
                        command, "model_authority_reason", "model-lease-unavailable"
                    ),
                    "hold_source": (
                        "px4-measured-braking-position"
                        if braking_to_rest
                        else "px4-measured-position"
                    ),
                    "hold_latched_for_authority_gap": not braking_to_rest,
                    "hold_observed_speed_mps": observed_speed_mps,
                    "hold_latch_speed_threshold_mps": (_MODEL_AUTHORITY_HOLD_LATCH_SPEED_MPS),
                    "schedule_advancement_authorized": False,
                },
            )
            await _send_position_with_velocity(
                base=base,
                client=client,
                setpoint=safe_setpoint,
                velocity_ned_mps=(0.0, 0.0, 0.0),
            )
            _record_model_control_application(
                args,
                command,
                velocity_ned_mps=(0.0, 0.0, 0.0),
                yaw_deg=float(safe_setpoint.yaw_deg),
                transport="position-velocity-ned",
                position_ned_m=(
                    float(safe_setpoint.north_m),
                    float(safe_setpoint.east_m),
                    float(safe_setpoint.down_m),
                ),
            )
            return safe_setpoint
        if model_control_required and (
            action in {"hold", "replan"}
            # A safety-authored velocity is not permission to turn toward the
            # route or to finish the interrupted model turn. Latch measured
            # heading before recording the actual zero-rate conversion.
            or getattr(command.decision, "control_source", None) == "deterministic-safety-override"
            or getattr(args, "_model_body_control_yaw_deg", None) is None
        ):
            safe_setpoint = base.Setpoint(
                north_m=safe_setpoint.north_m,
                east_m=safe_setpoint.east_m,
                down_m=safe_setpoint.down_m,
                yaw_deg=_local_hold_yaw(
                    args=args,
                    client=client,
                    fallback_heading_deg=planned_setpoint.yaw_deg,
                    model_control_required=True,
                ),
            )
        safe_setpoint = _setpoint_with_model_body_yaw(
            base=base,
            args=args,
            setpoint=safe_setpoint,
            command=command,
        )
        if model_control_required and model_control_authorized and action == "continue":
            selected_velocity = command.decision.selected_velocity_mps
            accepted_at = await _dispatch_motion_or_brake(
                args=args,
                base=base,
                client=client,
                setpoint=safe_setpoint,
                command=command,
                coordinate_contract=coordinate_contract,
                phase_path=phase_path,
                navigation_goal_deadline_monotonic=navigation_goal_deadline_monotonic,
                velocity_ned_mps=(
                    selected_velocity.y,
                    selected_velocity.x,
                    -selected_velocity.z,
                ),
            )
            if accepted_at is None:
                continue
            _record_model_control_application(
                args,
                command,
                velocity_ned_mps=(selected_velocity.y, selected_velocity.x, -selected_velocity.z),
                yaw_deg=float(safe_setpoint.yaw_deg),
                transport="velocity-ned",
                accepted_at_unix_ms=accepted_at,
            )
            # Phase-file I/O is reporting, not permission to send. It must not
            # spend the remaining source lease before the bounded transport.
            _clear_local_control_phase(phase_path)
            return safe_setpoint
        if action == "hold":
            if repair_hold_setpoint is None:
                repair_hold_setpoint = safe_setpoint
            safe_setpoint = repair_hold_setpoint
        else:
            repair_hold_setpoint = None
        clearance_recovery_active = "STATIC_CLEARANCE_RECOVERY" in set(
            getattr(command.decision, "issue_codes", [])
        )
        repair_progress_details: dict[str, Any] = {}
        if action in {"hold", "replan"} or clearance_recovery_active:
            repair_now = loop.time()
            repair_progress_state = _advance_local_repair_progress(
                now=repair_now,
                observed=observed,
                minimum_clearance_m=float(command.decision.minimum_predicted_clearance_m),
                stall_timeout_seconds=args.local_safety_repair_timeout_seconds,
                state=repair_progress_state,
            )
            deadline = float(repair_progress_state["stall_deadline"])
            repair_progress_details = {
                "repair_progress_revision": repair_progress_state["progress_revision"],
                "repair_progress_evidence": repair_progress_state["last_progress_evidence"],
                "repair_stall_seconds_remaining": max(0.0, deadline - repair_now),
                "repair_absolute_seconds_remaining": max(
                    0.0,
                    repair_absolute_deadline - repair_now,
                ),
            }
        local_control_phase = (
            "LOCAL_CLEARANCE_RECOVERY"
            if clearance_recovery_active
            else {
                "hold": "HOLDING",
                "slow": "LOCAL_SLOW",
                "replan": "LOCAL_REPLAN",
            }[action]
        )
        selected_velocity = command.decision.selected_velocity_mps
        selected_velocity_ned_mps = (
            selected_velocity.y,
            selected_velocity.x,
            -selected_velocity.z,
        )
        if action == "hold":
            selected_velocity_ned_mps = (0.0, 0.0, 0.0)
        accepted_at = None
        if model_control_required and model_control_authorized and action == "slow":
            accepted_at = await _dispatch_motion_or_brake(
                args=args,
                base=base,
                client=client,
                setpoint=safe_setpoint,
                velocity_ned_mps=selected_velocity_ned_mps,
                command=command,
                coordinate_contract=coordinate_contract,
                phase_path=phase_path,
                navigation_goal_deadline_monotonic=navigation_goal_deadline_monotonic,
            )
            if accepted_at is None:
                continue
        elif model_control_required and action == "replan":
            accepted_at = await _dispatch_motion_or_brake(
                args=args,
                base=base,
                client=client,
                setpoint=safe_setpoint,
                velocity_ned_mps=selected_velocity_ned_mps,
                command=command,
                coordinate_contract=coordinate_contract,
                phase_path=phase_path,
                position_control=True,
                navigation_goal_deadline_monotonic=navigation_goal_deadline_monotonic,
            )
            if accepted_at is None:
                continue
        else:
            accepted_at = await _send_position_with_velocity(
                base=base,
                client=client,
                setpoint=safe_setpoint,
                velocity_ned_mps=selected_velocity_ned_mps,
            )
        if model_control_required:
            _record_model_control_application(
                args,
                command,
                velocity_ned_mps=selected_velocity_ned_mps,
                yaw_deg=float(safe_setpoint.yaw_deg),
                transport=(
                    "velocity-ned"
                    if model_control_authorized and action == "slow"
                    else "position-velocity-ned"
                ),
                accepted_at_unix_ms=accepted_at,
                position_ned_m=(
                    None
                    if model_control_authorized and action == "slow"
                    else (
                        float(safe_setpoint.north_m),
                        float(safe_setpoint.east_m),
                        float(safe_setpoint.down_m),
                    )
                ),
            )
        # Publish only after the actual acceptance and its immutable receipt.
        # A disk stall here cannot turn a previously fresh model command into
        # an expired command before its first send.
        _publish_local_control_phase(
            phase_path,
            local_phase=local_control_phase,
            details={
                "local_safety_action": action,
                "static_clearance_recovery_active": clearance_recovery_active,
                "observation_sequence": command.observation_sequence,
                "threat_obstacle_id": command.decision.threat_obstacle_id,
                "minimum_predicted_clearance_m": (command.decision.minimum_predicted_clearance_m),
                **repair_progress_details,
            },
        )
        # A slowdown is itself a bounded, safe replacement setpoint. Let the
        # schedule advance after one controller tick so the next target can be
        # reassessed against fresh perception.  Only hold/replan commands stop
        # schedule time; treating "slow" as a hold made dense tracks stall.
        if action == "slow" and not clearance_recovery_active:
            return safe_setpoint
        repair_now = loop.time()
        if repair_now >= repair_absolute_deadline:
            raise UserDirectedLanding("local safety repair exceeded its absolute bounded window")
        if repair_now >= deadline:
            raise UserDirectedLanding(
                "local safety repair made no progress within its stall window"
            )
        _publish_local_safety_target(
            path=args.local_safety_target,
            setpoint=planned_setpoint,
            coordinate_contract=coordinate_contract,
            navigation_goal_position_m=navigation_goal_position_m,
            navigation_goal_id=navigation_goal_id,
            control_profile=control_profile,
            action_checkpoint_goal=action_checkpoint_goal,
            tracking_recovery_active=tracking_recovery_active,
            decision_trigger=decision_trigger,
            recovery_episode_id=recovery_episode_id,
        )


# 功能：
#   为原始路线选择绑定的分段跟踪限额，替换路线不沿用旧段配置。
# 输入：
#   args：全局跟踪限制及已验证的分段策略。
#   context：当前语义目标和路线替换身份。
# 输出：
#   limits：误差上限、恢复容差与可选分段索引。
def _tracking_limits_for_context(
    args: argparse.Namespace,
    context: dict[str, Any] | None,
) -> tuple[float, float, int | None]:
    """Select a hash-bound segment budget for the original route when available."""

    lag_limit_m = float(args.tracking_lag_limit_m)
    rejoin_tolerance_m = float(args.tracking_rejoin_tolerance_m)
    if context is None or context.get("replacement_sequence") is not None:
        return lag_limit_m, rejoin_tolerance_m, None
    navigation_goal_id = str(context.get("navigation_goal_id", ""))
    prefix = "source-waypoint-"
    if not navigation_goal_id.startswith(prefix):
        return lag_limit_m, rejoin_tolerance_m, None
    try:
        segment_index = int(navigation_goal_id[len(prefix) :]) - 1
    except ValueError:
        return lag_limit_m, rejoin_tolerance_m, None
    policies = getattr(args, "_tracking_segment_policies", ())
    if segment_index < 0 or segment_index >= len(policies):
        return lag_limit_m, rejoin_tolerance_m, None
    policy = policies[segment_index]
    return (
        float(policy["tracking_lag_limit_m"]),
        float(policy["tracking_rejoin_tolerance_m"]),
        segment_index,
    )


# 功能：
#   读取原路线当前段的巡航或精细配置，替换轨迹必须使用自己的控制权限。
# 输入：
#   args：已绑定原路线的分段策略。
#   context：当前语义目标及替换身份。
# 输出：
#   profile：本段允许使用的控制精度模式。
def _tracking_control_profile_for_context(
    args: argparse.Namespace,
    context: dict[str, Any] | None,
) -> str:
    """Return the profile signed for one original-route segment.

    Runtime replacement tracks have their own planning and safety authority;
    an original-track profile must never leak into them.
    """

    if context is None or context.get("replacement_sequence") is not None:
        return "cruise"
    navigation_goal_id = str(context.get("navigation_goal_id", ""))
    prefix = "source-waypoint-"
    if not navigation_goal_id.startswith(prefix):
        return "cruise"
    try:
        segment_index = int(navigation_goal_id[len(prefix) :]) - 1
    except ValueError:
        return "cruise"
    policies = getattr(args, "_tracking_segment_policies", ())
    if segment_index < 0 or segment_index >= len(policies):
        return "cruise"
    profile = str(policies[segment_index]["control_profile"])
    if profile not in {"cruise", "precision"}:
        raise RuntimeError("tracking corridor control profile was not validated")
    return profile


# 功能：
#   1. 执行安全仲裁控制，再以实际位置和语义目标距离决定是否推进参考序列。
#   2. 模型缺权时暂停，真实目标进展停滞时请求恢复专家，不能仅以模型小目标被触及宣告任务进展。
#   3. 记录实际控制、位置、速度和恢复证据，路线参考本身不构成继续运动的权限。
# 输入：
#   args：本轮权限、计数、跟踪限额与证据队列。
#   base：飞控基础模块。
#   client：实际控制和遥测客户端。
#   planned_setpoint：当前参考点。
#   coordinate_contract：地图和控制坐标绑定。
#   phase_path：执行阶段输出路径。
#   schedule_index：当前参考序列索引。
#   sample_now：兼容模式本轮是否需要额外遥测；模型模式每轮都检查。
#   recovery_active：上层跟踪恢复状态。
#   local_recovery_control_active：本轮局部固定目标恢复要求。
#   context：语义目标、动作及替换路线身份。
#   planned_velocity_ned_mps：路线切向速度，用于误差分解。
# 输出：
#   tracking_result：可否推进及需要恢复的误差二元组；非跟踪原因暂停时误差为 None。
async def _tracking_gate_tick(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    planned_setpoint: Any,
    coordinate_contract: Px4CoordinateContract,
    phase_path: Path,
    schedule_index: int,
    sample_now: bool,
    recovery_active: bool,
    local_recovery_control_active: bool | None = None,
    context: dict[str, Any] | None = None,
    planned_velocity_ned_mps: tuple[float, float, float] | None = None,
) -> tuple[bool, float | None]:
    """Command one control tick and gate schedule advancement on measured lag.

    The trajectory schedule describes a dynamically feasible *reference*, not
    permission to advance open-loop.  PX4 must remain close enough to the
    current reference before the executor releases the next sample.  Once lag
    exceeds the outer limit, hysteresis requires a tighter rejoin tolerance so
    the schedule cannot chatter between TRACK and TRACKING_RECOVERY.
    """

    tracking_lag_limit_m, tracking_rejoin_tolerance_m, tracking_segment_index = (
        _tracking_limits_for_context(args, context)
    )

    if local_recovery_control_active is None:
        local_recovery_control_active = recovery_active
    args._last_tracking_local_recovery_control_required = False
    args._last_px4_control_observation = None
    applied_setpoint = await _apply_local_safety(
        args=args,
        base=base,
        client=client,
        planned_setpoint=planned_setpoint,
        coordinate_contract=coordinate_contract,
        phase_path=phase_path,
        navigation_goal_position_m=(
            Vector3.model_validate(context["navigation_goal_position_m"])
            if context is not None and context.get("navigation_goal_position_m") is not None
            else None
        ),
        navigation_goal_id=(context.get("navigation_goal_id") if context is not None else None),
        control_profile=(
            str(context.get("control_profile", "cruise")) if context is not None else "cruise"
        ),
        action_checkpoint_goal=(
            bool(context.get("action_checkpoint_goal", False)) if context is not None else False
        ),
        planned_velocity_ned_mps=planned_velocity_ned_mps,
        # Precision approach is already represented by control_profile and
        # must not masquerade as fixed-target tracking recovery while the
        # dense reference is still moving. Only a real corridor/terminal
        # recovery may request the worker's recovery-bound proof.
        tracking_recovery_active=local_recovery_control_active,
        decision_trigger=(
            "progress-stalled"
            if bool(getattr(args, "_model_progress_recovery_requested", False))
            else None
        ),
        recovery_episode_id=(
            str(args._model_progress_recovery_episode_id)
            if getattr(args, "_model_progress_recovery_episode_id", None)
            else None
        ),
    )
    model_control_required = bool(getattr(args, "_last_model_control_required", False))
    model_control_authorized = bool(getattr(args, "_last_model_control_authorized", False))
    if model_control_required and not model_control_authorized:
        args._model_authority_schedule_hold_count = (
            getattr(args, "_model_authority_schedule_hold_count", 0) + 1
        )
        evidence = {
            "schema_version": "dronedream.closed-loop-tracking-state.v1",
            "schedule_index": schedule_index,
            "state": "model-authority-hold",
            "recovery_active": recovery_active,
            "model_control_authority_required": True,
            "model_navigation_authorized": False,
            "model_call_id": getattr(args, "_last_model_call_id", None),
            "model_selected_candidate_id": getattr(args, "_last_model_selected_candidate_id", None),
            "schedule_advancement_authorized": False,
            "updated_at_unix_ms": int(time.time() * 1_000),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        # _apply_local_safety has already sampled PX4 for this control tick.
        # Preserve that real observation in the latest-state register even
        # while the model lease is absent; omitting it made diagnostics and the
        # UI render a physically impossible jump to zero during every hold.
        observed = getattr(args, "_last_px4_control_observation", None)
        args._last_px4_control_observation = None
        if observed is not None:
            values = (
                observed.north_m,
                observed.east_m,
                observed.down_m,
                observed.north_m_s,
                observed.east_m_s,
                observed.down_m_s,
            )
            if not all(math.isfinite(value) for value in values):
                raise UserDirectedLanding("tracking telemetry contained non-finite values")
            estimator_offset = getattr(
                args,
                "_last_estimator_to_world_position_offset_m",
                Vector3(x=0.0, y=0.0, z=0.0),
            )
            observed_route_frame_ned = (
                observed.north_m + estimator_offset.y,
                observed.east_m + estimator_offset.x,
                observed.down_m - estimator_offset.z,
            )
            route_tracking_errors = _route_tracking_error_components(
                observed_route_frame_ned=observed_route_frame_ned,
                planned_setpoint=planned_setpoint,
                planned_velocity_ned_mps=planned_velocity_ned_mps,
            )
            root_east, root_north, root_up = coordinate_contract.model_root_world_enu_m
            offset_east, offset_north, offset_up = (
                coordinate_contract.resolved_collision_center_offset_model_m()
            )
            observed_world_collision_center = {
                "x": root_east + offset_east + observed.east_m + estimator_offset.x,
                "y": root_north + offset_north + observed.north_m + estimator_offset.y,
                "z": root_up + offset_up - observed.down_m + estimator_offset.z,
            }
            evidence.update(
                {
                    "position_error_m": math.dist(
                        (observed.north_m, observed.east_m, observed.down_m),
                        (
                            applied_setpoint.north_m,
                            applied_setpoint.east_m,
                            applied_setpoint.down_m,
                        ),
                    ),
                    "controller_position_error_m": math.dist(
                        (observed.north_m, observed.east_m, observed.down_m),
                        (
                            applied_setpoint.north_m,
                            applied_setpoint.east_m,
                            applied_setpoint.down_m,
                        ),
                    ),
                    "route_schedule_position_error_m": math.dist(
                        observed_route_frame_ned,
                        (
                            planned_setpoint.north_m,
                            planned_setpoint.east_m,
                            planned_setpoint.down_m,
                        ),
                    ),
                    "route_tangent_available": bool(route_tracking_errors["tangent_available"]),
                    "route_cross_track_error_m": float(
                        route_tracking_errors["cross_track_error_m"]
                    ),
                    "route_along_track_lag_m": float(route_tracking_errors["along_track_lag_m"]),
                    "route_along_track_lead_m": float(route_tracking_errors["along_track_lead_m"]),
                    "speed_mps": math.sqrt(
                        observed.north_m_s**2 + observed.east_m_s**2 + observed.down_m_s**2
                    ),
                    "estimator_to_world_position_offset_m": (
                        estimator_offset.model_dump(mode="json")
                    ),
                    "planned_setpoint_ned": {
                        "north_m": planned_setpoint.north_m,
                        "east_m": planned_setpoint.east_m,
                        "down_m": planned_setpoint.down_m,
                    },
                    "applied_setpoint_ned": {
                        "north_m": applied_setpoint.north_m,
                        "east_m": applied_setpoint.east_m,
                        "down_m": applied_setpoint.down_m,
                    },
                    "observed_position_ned": {
                        "north_m": observed.north_m,
                        "east_m": observed.east_m,
                        "down_m": observed.down_m,
                    },
                    "observed_velocity_ned_mps": {
                        "north_m_s": observed.north_m_s,
                        "east_m_s": observed.east_m_s,
                        "down_m_s": observed.down_m_s,
                    },
                    "observed_world_collision_center_m": (observed_world_collision_center),
                }
            )
            navigation_goal_payload = context.get("navigation_goal_position_m") if context else None
            if isinstance(navigation_goal_payload, dict):
                navigation_goal = Vector3.model_validate(navigation_goal_payload)
                evidence["model_goal_distance_m"] = math.dist(
                    tuple(observed_world_collision_center.values()),
                    (navigation_goal.x, navigation_goal.y, navigation_goal.z),
                )
                planned_world = _setpoint_world_enu(planned_setpoint, coordinate_contract)
                evidence["planned_goal_distance_m"] = math.dist(
                    (planned_world.x, planned_world.y, planned_world.z),
                    (navigation_goal.x, navigation_goal.y, navigation_goal.z),
                )
        if context:
            evidence.update(context)
        _atomic_json(args.run_dir / "runtime-state" / "closed-loop-tracking.json", evidence)
        return False, None
    if model_control_required:
        # Model-required motion is sampled on every schedule release. A dense
        # reference point may never advance merely because this happens to be
        # a lower-rate telemetry tick.
        sample_now = True
    if not sample_now:
        return True, None

    observed = getattr(args, "_last_px4_control_observation", None)
    args._last_px4_control_observation = None
    if observed is None:
        observed = await client.sample_position_velocity_ned(
            args.tracking_telemetry_timeout_seconds
        )
    values = (
        observed.north_m,
        observed.east_m,
        observed.down_m,
        observed.north_m_s,
        observed.east_m_s,
        observed.down_m_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise UserDirectedLanding("tracking telemetry contained non-finite values")
    controller_position_error_m = math.dist(
        (observed.north_m, observed.east_m, observed.down_m),
        (
            applied_setpoint.north_m,
            applied_setpoint.east_m,
            applied_setpoint.down_m,
        ),
    )
    estimator_offset = getattr(
        args,
        "_last_estimator_to_world_position_offset_m",
        Vector3(x=0.0, y=0.0, z=0.0),
    )
    observed_route_frame_ned = (
        observed.north_m + estimator_offset.y,
        observed.east_m + estimator_offset.x,
        observed.down_m - estimator_offset.z,
    )
    route_schedule_position_error_m = math.dist(
        observed_route_frame_ned,
        (
            planned_setpoint.north_m,
            planned_setpoint.east_m,
            planned_setpoint.down_m,
        ),
    )
    route_tracking_errors = _route_tracking_error_components(
        observed_route_frame_ned=observed_route_frame_ned,
        planned_setpoint=planned_setpoint,
        planned_velocity_ned_mps=planned_velocity_ned_mps,
    )
    route_cross_track_error_m = float(route_tracking_errors["cross_track_error_m"])
    route_along_track_lag_m = float(route_tracking_errors["along_track_lag_m"])
    route_along_track_lead_m = float(route_tracking_errors["along_track_lead_m"])
    route_along_track_lag_limit_m = float(route_tracking_errors["along_track_lag_limit_m"])
    route_along_track_lag_rejoin_limit_m = float(
        route_tracking_errors["along_track_lag_rejoin_limit_m"]
    )
    speed_mps = math.sqrt(observed.north_m_s**2 + observed.east_m_s**2 + observed.down_m_s**2)
    root_east, root_north, root_up = coordinate_contract.model_root_world_enu_m
    offset_east, offset_north, offset_up = (
        coordinate_contract.resolved_collision_center_offset_model_m()
    )
    observed_world_collision_center = {
        "x": root_east + offset_east + observed.east_m + estimator_offset.x,
        "y": root_north + offset_north + observed.north_m + estimator_offset.y,
        "z": root_up + offset_up - observed.down_m + estimator_offset.z,
    }
    threshold_m = tracking_rejoin_tolerance_m if recovery_active else tracking_lag_limit_m
    model_progress_slack_m = max(0.0, float(getattr(args, "model_progress_slack_m", 0.0)))
    # A short model-controller target proves that this control tick is locally
    # reachable; it does not prove progress along the dense mission schedule.
    # Bind strict-mode schedule release to the vehicle's measured distance from
    # the current semantic waypoint relative to the reference setpoint's
    # distance from that same waypoint.  Comparing cumulative route arc length
    # with straight-line goal progress can deadlock at a bend: the arc is
    # necessarily longer even when the aircraft has reached the right place.
    # This relative-lag gate still prevents centimetre-scale model targets from
    # racing the route clock, while permitting a safe locally revalidated
    # detour to make equivalent or better semantic progress.
    model_progress_hold = False
    local_recovery_control_required = False
    model_goal_distance_m: float | None = None
    planned_goal_distance_m: float | None = None
    model_goal_progress_m: float | None = None
    model_candidate_schedule_progress_m: float | None = None
    model_schedule_progress_authorized: bool | None = None
    model_controller_target_reached: bool | None = None
    model_reference_catch_up_authorized: bool | None = None
    position_error_m = controller_position_error_m
    if model_control_required:
        navigation_goal_payload = context.get("navigation_goal_position_m") if context else None
        navigation_goal_id = str(context.get("navigation_goal_id")) if context else None
        if isinstance(navigation_goal_payload, dict) and navigation_goal_id:
            navigation_goal = Vector3.model_validate(navigation_goal_payload)
            model_goal_distance_m = math.dist(
                (
                    observed_world_collision_center["x"],
                    observed_world_collision_center["y"],
                    observed_world_collision_center["z"],
                ),
                (navigation_goal.x, navigation_goal.y, navigation_goal.z),
            )
            planned_world = _setpoint_world_enu(planned_setpoint, coordinate_contract)
            planned_goal_distance_m = math.dist(
                (planned_world.x, planned_world.y, planned_world.z),
                (navigation_goal.x, navigation_goal.y, navigation_goal.z),
            )
            progress_state = getattr(args, "_model_goal_progress_state", None)
            if (
                not isinstance(progress_state, dict)
                or progress_state.get("navigation_goal_id") != navigation_goal_id
            ):
                progress_state = {
                    "navigation_goal_id": navigation_goal_id,
                    "initial_goal_distance_m": model_goal_distance_m,
                    "initial_planned_goal_distance_m": planned_goal_distance_m,
                    "accepted_schedule_distance_m": 0.0,
                }
                args._model_goal_progress_state = progress_state
            initial_planned_goal_distance_m = float(
                progress_state.get(
                    "initial_planned_goal_distance_m",
                    progress_state["initial_goal_distance_m"],
                )
            )
            model_candidate_schedule_progress_m = max(
                0.0,
                initial_planned_goal_distance_m - planned_goal_distance_m,
            )
            model_goal_progress_m = max(
                0.0,
                float(progress_state["initial_goal_distance_m"]) - model_goal_distance_m,
            )
            model_schedule_progress_authorized = (
                model_goal_distance_m
                <= planned_goal_distance_m + threshold_m + model_progress_slack_m
            )
            model_controller_target_reached = controller_position_error_m <= threshold_m
            # The independently revalidated model path can legitimately put the
            # aircraft ahead of the slower dense reference.  In that case the
            # schedule must be allowed to catch up with *measured semantic
            # progress* even while the next short-horizon model target is still
            # moving ahead.  Requiring the vehicle to settle within centimetres
            # of every refreshed local target misclassifies ordinary precision
            # flight as tracking recovery and eventually spends the recovery
            # window despite continuous physical progress.
            #
            # Do not use the normal progress slack for this catch-up path: the
            # measured aircraft must be at least as close to the semantic goal
            # as the reference sample being released.  A vehicle that is still
            # behind may advance only after actually reaching its authorized
            # controller target, preserving the anti-race invariant.
            model_reference_catch_up_authorized = model_goal_distance_m <= planned_goal_distance_m
            may_advance = model_schedule_progress_authorized and (
                model_controller_target_reached or model_reference_catch_up_authorized
            )
            if may_advance:
                progress_state["accepted_schedule_distance_m"] = max(
                    float(progress_state["accepted_schedule_distance_m"]),
                    model_candidate_schedule_progress_m,
                )
            elif controller_position_error_m <= threshold_m:
                model_progress_hold = True
                args._model_progress_schedule_hold_count = (
                    getattr(args, "_model_progress_schedule_hold_count", 0) + 1
                )
            semantic_now = time.monotonic()
            semantic_progress_state, semantic_progress_expired = (
                _advance_model_semantic_progress_window(
                    now=semantic_now,
                    recovery_after_seconds=float(
                        getattr(
                            args,
                            "semantic_progress_recovery_timeout_seconds",
                            20.0,
                        )
                    ),
                    abort_after_seconds=float(
                        getattr(
                            args,
                            "semantic_progress_abort_timeout_seconds",
                            60.0,
                        )
                    ),
                    state=getattr(
                        args,
                        "_model_semantic_progress_window",
                        None,
                    ),
                    navigation_goal_id=navigation_goal_id,
                    model_goal_distance_m=model_goal_distance_m,
                    # Dense schedule indices do not reset this window merely
                    # by existing.  They count only after the measured vehicle,
                    # model lease, and relative semantic-lag gate jointly
                    # authorize this concrete schedule advance.  A truly stuck
                    # vehicle reaches that lag bound and still times out, while
                    # an aircraft already inside the terminal settle envelope
                    # is allowed to finish the remaining reference samples.
                    authorized_schedule_advance=may_advance,
                )
            )
            args._model_semantic_progress_window = semantic_progress_state
            args._model_progress_recovery_requested = bool(
                semantic_progress_state["recovery_requested"]
            )
            args._model_progress_recovery_episode_id = semantic_progress_state.get(
                "recovery_episode_id"
            )
            _publish_model_semantic_progress_window(
                args=args,
                state=semantic_progress_state,
                now=semantic_now,
            )
            if semantic_progress_expired:
                raise UserDirectedLanding(
                    "model-authorized semantic progress stalled after recovery request"
                )
        else:
            position_error_m = max(
                controller_position_error_m,
                route_schedule_position_error_m,
            )
            may_advance = position_error_m <= threshold_m
            local_recovery_control_required = not may_advance
    else:
        if bool(route_tracking_errors["tangent_available"]):
            along_track_threshold_m = (
                route_along_track_lag_rejoin_limit_m
                if recovery_active
                else route_along_track_lag_limit_m
            )
            along_track_within_limit = route_along_track_lag_m <= along_track_threshold_m
            may_advance = route_cross_track_error_m <= threshold_m and along_track_within_limit
            # Being behind on the already-qualified route only pauses the
            # reference clock. It does not spend collision clearance and does
            # not need the worker's fixed-target radial-recovery proof. True
            # cross-track deviation still receives that stronger context.
            local_recovery_control_required = route_cross_track_error_m > threshold_m
            # Ordinary time lag inside its separate bound is diagnostic, not
            # a corridor violation.  Once it exceeds that bound, expose the
            # full lag to the bounded recovery-progress watchdog.
            position_error_m = (
                route_cross_track_error_m
                if along_track_within_limit
                else max(route_cross_track_error_m, route_along_track_lag_m)
            )
        else:
            position_error_m = route_schedule_position_error_m
            may_advance = position_error_m <= threshold_m
            local_recovery_control_required = not may_advance
    if model_control_required and not model_progress_hold:
        local_recovery_control_required = not may_advance
    args._last_tracking_local_recovery_control_required = local_recovery_control_required
    if model_control_required and may_advance:
        args._model_authorized_schedule_advance_count = (
            getattr(args, "_model_authorized_schedule_advance_count", 0) + 1
        )
    elif not model_control_required and may_advance:
        args._route_fallback_schedule_advance_count = (
            getattr(args, "_route_fallback_schedule_advance_count", 0) + 1
        )
    # Keep the public return shape stable for callers and tests while exposing
    # the time-aligned semantic progress sample to the bounded recovery window.
    args._last_tracking_model_goal_distance_m = model_goal_distance_m
    evidence: dict[str, Any] = {
        "schema_version": "dronedream.closed-loop-tracking-state.v1",
        "schedule_index": schedule_index,
        "state": (
            "tracking"
            if may_advance
            else "model-progress-hold"
            if model_progress_hold
            else "recovering"
        ),
        "recovery_active": recovery_active or (not may_advance and not model_progress_hold),
        "local_recovery_control_active": local_recovery_control_active,
        "local_recovery_control_required": local_recovery_control_required,
        "position_error_m": position_error_m,
        "controller_position_error_m": controller_position_error_m,
        "route_schedule_position_error_m": route_schedule_position_error_m,
        "route_tangent_available": bool(route_tracking_errors["tangent_available"]),
        "route_cross_track_error_m": route_cross_track_error_m,
        "route_along_track_lag_m": route_along_track_lag_m,
        "route_along_track_lead_m": route_along_track_lead_m,
        "route_along_track_lag_limit_m": route_along_track_lag_limit_m,
        "route_along_track_lag_rejoin_limit_m": (route_along_track_lag_rejoin_limit_m),
        "estimator_to_world_position_offset_m": estimator_offset.model_dump(mode="json"),
        "model_goal_distance_m": model_goal_distance_m,
        "planned_goal_distance_m": planned_goal_distance_m,
        "model_goal_progress_m": model_goal_progress_m,
        "model_candidate_schedule_progress_m": model_candidate_schedule_progress_m,
        "model_schedule_progress_authorized": model_schedule_progress_authorized,
        "model_controller_target_reached": model_controller_target_reached,
        "model_reference_catch_up_authorized": model_reference_catch_up_authorized,
        "model_progress_slack_m": model_progress_slack_m,
        "speed_mps": speed_mps,
        "advance_threshold_m": threshold_m,
        "lag_limit_m": tracking_lag_limit_m,
        "rejoin_tolerance_m": tracking_rejoin_tolerance_m,
        "tracking_corridor_segment_index": tracking_segment_index,
        "planned_setpoint_ned": {
            "north_m": planned_setpoint.north_m,
            "east_m": planned_setpoint.east_m,
            "down_m": planned_setpoint.down_m,
        },
        "planned_velocity_feedforward_ned_mps": (
            {
                "north_m_s": planned_velocity_ned_mps[0],
                "east_m_s": planned_velocity_ned_mps[1],
                "down_m_s": planned_velocity_ned_mps[2],
            }
            if planned_velocity_ned_mps is not None
            else None
        ),
        "applied_velocity_feedforward_ned_mps": (
            {
                "north_m_s": applied_velocity[0],
                "east_m_s": applied_velocity[1],
                "down_m_s": applied_velocity[2],
            }
            if (
                applied_velocity := getattr(
                    args,
                    "_last_applied_velocity_feedforward_ned_mps",
                    None,
                )
            )
            is not None
            else None
        ),
        "applied_setpoint_ned": {
            "north_m": applied_setpoint.north_m,
            "east_m": applied_setpoint.east_m,
            "down_m": applied_setpoint.down_m,
        },
        "observed_position_ned": {
            "north_m": observed.north_m,
            "east_m": observed.east_m,
            "down_m": observed.down_m,
        },
        "observed_velocity_ned_mps": {
            "north_m_s": observed.north_m_s,
            "east_m_s": observed.east_m_s,
            "down_m_s": observed.down_m_s,
        },
        "observed_world_collision_center_m": observed_world_collision_center,
        "updated_at_unix_ms": int(time.time() * 1_000),
        "updated_at": datetime.now(UTC).isoformat(),
        "model_control_authority_required": model_control_required,
        "model_navigation_authorized": model_control_authorized,
        "model_call_id": getattr(args, "_last_model_call_id", None),
        "model_selected_candidate_id": getattr(args, "_last_model_selected_candidate_id", None),
        "model_path_sha256": getattr(args, "_last_model_path_sha256", None),
        "schedule_advancement_authorized": may_advance,
    }
    if context:
        evidence.update(context)
    _atomic_json(args.run_dir / "runtime-state" / "closed-loop-tracking.json", evidence)
    if not may_advance and not model_progress_hold:
        phase = {
            "phase": "TRACKING_RECOVERY",
            "schedule_index": schedule_index,
            "position_error_m": position_error_m,
            "rejoin_tolerance_m": args.tracking_rejoin_tolerance_m,
        }
        if context:
            phase.update(context)
        _atomic_json(phase_path, phase)
    return may_advance, None if model_progress_hold else position_error_m


# 功能：
#   将配置的遥测采样频率换算成至少一个控制周期的检查间隔。
# 输入：
#   args：控制频率与跟踪采样频率。
# 输出：
#   interval：两次采样之间的控制周期数。
def _tracking_sample_interval(args: argparse.Namespace) -> int:
    return max(1, int(round(args.setpoint_rate_hz / args.tracking_sample_rate_hz)))


# 功能：
#   跨参考索引跟踪语义目标的实测进展，停滞先请求恢复专家，再在有界时间内保护退出。
# 输入：
#   now：当前单调时刻。
#   recovery_after_seconds：无有效进展多久后请求恢复。
#   abort_after_seconds：无有效进展的最终时限。
#   state：上轮语义进度窗口。
#   navigation_goal_id：稳定的语义目标身份。
#   model_goal_distance_m：实测飞行器到目标的距离。
#   authorized_schedule_advance：本轮是否已通过实测进度门槛而允许推进。
#   progress_epsilon_m：有效距离改善阈值。
# 输出：
#   progress_result：更新的语义进度状态及是否最终超时。
def _advance_model_semantic_progress_window(
    *,
    now: float,
    recovery_after_seconds: float,
    abort_after_seconds: float,
    state: dict[str, Any] | None,
    navigation_goal_id: str,
    model_goal_distance_m: float,
    authorized_schedule_advance: bool = False,
    progress_epsilon_m: float = 0.05,
) -> tuple[dict[str, Any], bool]:
    """Escalate a cross-schedule semantic stall before a bounded landing.

    Dense reference points each have their own tracking-recovery window. A
    model can nevertheless authorize centimetre-scale motion without ever
    closing distance to the semantic waypoint, which previously reset the
    per-index window indefinitely. This window follows the stable semantic
    goal across schedule indices, requests the recovery expert first, and only
    then fails closed if no material physical progress resumes.
    """

    if not isinstance(navigation_goal_id, str) or not navigation_goal_id.strip():
        raise ValueError("semantic progress requires a navigation goal id")
    now = _control_scalar(now, "semantic progress clock")
    model_goal_distance_m = _control_scalar(
        model_goal_distance_m, "semantic progress distance", minimum=0
    )
    if (
        type(authorized_schedule_advance) is not bool
        or not all(
            finite_positive_number(value)
            for value in (
                recovery_after_seconds,
                abort_after_seconds,
                progress_epsilon_m,
            )
        )
        or abort_after_seconds < recovery_after_seconds
    ):
        raise ValueError("semantic progress policy is invalid")
    recovery_deadline = _control_scalar(now + recovery_after_seconds, "semantic recovery deadline")
    abort_deadline = _control_scalar(now + abort_after_seconds, "semantic abort deadline")
    if recovery_deadline <= now or abort_deadline <= now:
        raise ValueError("semantic progress deadlines must advance the clock")
    if state is not None and not isinstance(state, dict):
        raise ValueError("semantic progress state must be an object")
    if state is None or state.get("navigation_goal_id") != navigation_goal_id:
        state = {
            "navigation_goal_id": navigation_goal_id,
            "started_at_monotonic": now,
            "last_progress_at_monotonic": now,
            "checked_at_monotonic": now,
            "recovery_deadline_monotonic": recovery_deadline,
            "abort_deadline_monotonic": abort_deadline,
            "best_model_goal_distance_m": model_goal_distance_m,
            "progress_revision": 0,
            "authorized_schedule_revision": 0,
            "recovery_requested": False,
            "recovery_episode_id": None,
            "recovery_request_count": 0,
        }
        return state, False

    for field in (
        "started_at_monotonic",
        "last_progress_at_monotonic",
        "recovery_deadline_monotonic",
        "abort_deadline_monotonic",
    ):
        _control_scalar(state.get(field), f"semantic {field}")
    for field in ("progress_revision", "authorized_schedule_revision", "recovery_request_count"):
        if type(state.get(field)) is not int or state[field] < 0:
            raise ValueError("semantic progress count is invalid")
    if type(state.get("recovery_requested")) is not bool or now < _control_scalar(
        state.get("checked_at_monotonic", now), "semantic prior clock"
    ):
        raise ValueError("semantic recovery state or clock is invalid")
    best_distance = _control_scalar(
        state.get("best_model_goal_distance_m"), "semantic best distance", minimum=0
    )
    state["checked_at_monotonic"] = now
    distance_progressed = model_goal_distance_m <= best_distance - progress_epsilon_m
    # 超过最终期限的迟到进展不能重新授权；仍继续形成恢复／超时诊断。
    if now < state["abort_deadline_monotonic"] and (
        distance_progressed or authorized_schedule_advance
    ):
        if distance_progressed:
            state["best_model_goal_distance_m"] = model_goal_distance_m
        state["last_progress_at_monotonic"] = now
        state["recovery_deadline_monotonic"] = recovery_deadline
        state["abort_deadline_monotonic"] = abort_deadline
        state["progress_revision"] = int(state["progress_revision"]) + 1
        if authorized_schedule_advance:
            state["authorized_schedule_revision"] = (
                int(state.get("authorized_schedule_revision", 0)) + 1
            )
        state["recovery_requested"] = False
        state["recovery_episode_id"] = None

    if now >= float(state["recovery_deadline_monotonic"]):
        if not bool(state["recovery_requested"]):
            request_count = int(state.get("recovery_request_count", 0)) + 1
            state["recovery_request_count"] = request_count
            state["recovery_episode_id"] = (
                "recovery-"
                + sha256_json(
                    {
                        "navigation_goal_id": navigation_goal_id,
                        "request_count": request_count,
                        "started_at_monotonic": state["started_at_monotonic"],
                        "requested_at_monotonic": now,
                    }
                )[:24]
            )
        state["recovery_requested"] = True
    return state, now >= float(state["abort_deadline_monotonic"])


# 功能：
#   发布语义进展、恢复请求及剩余期限的诊断，不延长窗口或制造额外进展。
# 输入：
#   args：当前运行目录。
#   state：已更新的语义进度窗口。
#   now：计算剩余时间的单调时刻。
# 输出：
#   None：不返回业务数据。
def _publish_model_semantic_progress_window(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    now: float,
) -> None:
    updated_at_unix_ms = int(time.time() * 1_000)
    _atomic_json(
        args.run_dir / "runtime-state" / "model-semantic-progress-window.json",
        {
            "schema_version": "dronedream.model-semantic-progress-window.v1",
            "navigation_goal_id": state["navigation_goal_id"],
            "elapsed_without_progress_seconds": max(
                0.0,
                now - float(state["last_progress_at_monotonic"]),
            ),
            "recovery_seconds_remaining": max(
                0.0,
                float(state["recovery_deadline_monotonic"]) - now,
            ),
            "abort_seconds_remaining": max(
                0.0,
                float(state["abort_deadline_monotonic"]) - now,
            ),
            "best_model_goal_distance_m": state["best_model_goal_distance_m"],
            "progress_revision": state["progress_revision"],
            "authorized_schedule_revision": state.get("authorized_schedule_revision", 0),
            "recovery_requested": state["recovery_requested"],
            "recovery_episode_id": state.get("recovery_episode_id"),
            "recovery_request_count": state.get("recovery_request_count", 0),
            "updated_at": datetime.now(UTC).isoformat(),
            "updated_at_unix_ms": updated_at_unix_ms,
        },
    )


# 功能：
#   以真实误差收敛或语义目标接近刷新短停滞期限，总恢复期限始终固定，避免振荡无限续期。
# 输入：
#   now：当前单调时刻。
#   timeout_seconds：普通无进展时限。
#   state：上轮恢复锚点和期限。
#   tracking_error_m：当前真实跟踪误差。
#   model_goal_distance_m：可选语义目标实测距离。
#   progress_epsilon_m：有效改善阈值。
#   absolute_timeout_factor：总恢复预算相对普通预算的倍数。
# 输出：
#   recovery_result：更新后的恢复状态与是否已超时。
def _advance_tracking_recovery_window(
    *,
    now: float,
    timeout_seconds: float,
    state: dict[str, Any] | None,
    tracking_error_m: float,
    model_goal_distance_m: float | None,
    progress_epsilon_m: float = 0.01,
    absolute_timeout_factor: float = 3.0,
) -> tuple[dict[str, Any], bool]:
    """Advance a finite recovery window only on measured physical progress.

    The ordinary recovery timeout is a *stall* deadline.  A model-authorized
    detour can need longer than that to pull the aircraft back inside the
    tighter hysteresis threshold, so a material decrease in either semantic
    goal distance or controller error refreshes the stall deadline.  The
    absolute deadline never moves; oscillation therefore cannot keep a flight
    alive indefinitely.
    """

    now = _control_scalar(now, "tracking recovery clock")
    tracking_error_m = _control_scalar(tracking_error_m, "tracking recovery error", minimum=0)
    if model_goal_distance_m is not None:
        model_goal_distance_m = _control_scalar(
            model_goal_distance_m, "recovery goal distance", minimum=0
        )
    if (
        not all(
            finite_positive_number(value)
            for value in (
                timeout_seconds,
                progress_epsilon_m,
                absolute_timeout_factor,
            )
        )
        or absolute_timeout_factor < 1
    ):
        raise ValueError("tracking recovery policy is invalid")
    next_deadline = _control_scalar(now + timeout_seconds, "tracking stall deadline")
    absolute_deadline = _control_scalar(
        now + timeout_seconds * absolute_timeout_factor,
        "tracking absolute deadline",
    )
    if next_deadline <= now or absolute_deadline <= now:
        raise ValueError("tracking recovery deadlines must advance the clock")
    if state is not None and not isinstance(state, dict):
        raise ValueError("tracking recovery state must be an object")
    if state is None:
        state = {
            "started_at_monotonic": now,
            "checked_at_monotonic": now,
            "stall_deadline_monotonic": next_deadline,
            "absolute_deadline_monotonic": absolute_deadline,
            "best_tracking_error_m": tracking_error_m,
            "tracking_error_progress_anchor_m": tracking_error_m,
            "best_model_goal_distance_m": model_goal_distance_m,
            "progress_revision": 0,
            "last_progress_evidence": "recovery_started",
        }
        return state, False

    for field in (
        "started_at_monotonic",
        "stall_deadline_monotonic",
        "absolute_deadline_monotonic",
    ):
        _control_scalar(state.get(field), f"recovery {field}")
    for field in ("best_tracking_error_m", "tracking_error_progress_anchor_m"):
        _control_scalar(state.get(field), f"recovery {field}", minimum=0)
    if state.get("best_model_goal_distance_m") is not None:
        _control_scalar(
            state["best_model_goal_distance_m"], "recovery best goal distance", minimum=0
        )
    if (
        type(state.get("progress_revision")) is not int
        or state["progress_revision"] < 0
        or now < _control_scalar(state.get("checked_at_monotonic", now), "recovery prior clock")
    ):
        raise ValueError("tracking recovery progress or clock is invalid")
    state["checked_at_monotonic"] = now
    # 普通停滞窗口保留恰好到期这一采样的进展判定；绝对期限到达即不可续期。
    if now > state["stall_deadline_monotonic"] or now >= state["absolute_deadline_monotonic"]:
        return state, True
    progress_evidence: str | None = None
    best_model_goal_distance_m = state.get("best_model_goal_distance_m")
    if (
        model_goal_distance_m is not None
        and math.isfinite(model_goal_distance_m)
        and (
            best_model_goal_distance_m is None
            or model_goal_distance_m <= float(best_model_goal_distance_m) - progress_epsilon_m
        )
    ):
        state["best_model_goal_distance_m"] = model_goal_distance_m
        progress_evidence = "semantic_goal_distance_decreased"

    best_tracking_error_m = float(state["best_tracking_error_m"])
    state["best_tracking_error_m"] = min(best_tracking_error_m, tracking_error_m)
    # Keep the all-time best for evidence, but measure directional convergence
    # from a recent peak.  A tightly controlled hover can overshoot by a few
    # centimetres before damping back toward the fixed target.  Comparing only
    # with the all-time trough incorrectly calls that valid convergence a
    # stall.  The immutable absolute deadline still prevents a periodic
    # oscillation from extending recovery indefinitely.
    tracking_anchor = float(state.get("tracking_error_progress_anchor_m", best_tracking_error_m))
    if tracking_error_m <= tracking_anchor - progress_epsilon_m:
        state["tracking_error_progress_anchor_m"] = tracking_error_m
        if progress_evidence is None:
            progress_evidence = "controller_error_decreased"
    elif tracking_error_m >= tracking_anchor + progress_epsilon_m:
        state["tracking_error_progress_anchor_m"] = tracking_error_m

    if progress_evidence is not None:
        absolute_deadline = float(state["absolute_deadline_monotonic"])
        state["stall_deadline_monotonic"] = min(
            absolute_deadline,
            next_deadline,
        )
        state["progress_revision"] = int(state["progress_revision"]) + 1
        state["last_progress_evidence"] = progress_evidence

    expired = now >= float(state["stall_deadline_monotonic"]) or now >= float(
        state["absolute_deadline_monotonic"]
    )
    return state, expired


# 功能：
#   发布固定参考点恢复的实际耗时、收敛证据及两种剩余期限。
# 输入：
#   args：当前运行证据目录。
#   schedule_index：正在恢复的参考索引。
#   state：恢复状态与原始绝对期限。
#   now：当前单调时刻。
#   replacement_sequence：可选新路线替换身份。
# 输出：
#   None：不返回业务数据。
def _publish_tracking_recovery_window(
    *,
    args: argparse.Namespace,
    schedule_index: int,
    state: dict[str, Any],
    now: float,
    replacement_sequence: int | None = None,
) -> None:
    payload = {
        "schema_version": "dronedream.tracking-recovery-window.v1",
        "schedule_index": schedule_index,
        "replacement_sequence": replacement_sequence,
        "elapsed_seconds": max(0.0, now - float(state["started_at_monotonic"])),
        "stall_seconds_remaining": max(
            0.0,
            float(state["stall_deadline_monotonic"]) - now,
        ),
        "absolute_seconds_remaining": max(
            0.0,
            float(state["absolute_deadline_monotonic"]) - now,
        ),
        "best_tracking_error_m": state["best_tracking_error_m"],
        "best_model_goal_distance_m": state["best_model_goal_distance_m"],
        "progress_revision": state["progress_revision"],
        "last_progress_evidence": state["last_progress_evidence"],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(
        args.run_dir / "runtime-state" / "tracking-recovery-window.json",
        payload,
    )


# 功能：
#   根据授权速度、可用减速度及稳定容差估算提前制动范围，避免到检查点才开始减速。
# 输入：
#   planned_goal_distance_m：当前参考位置到目标的距离。
#   planned_speed_mps：需要覆盖的最大控制速度。
#   maximum_acceleration_mps2：允许的加减速度限制。
#   waypoint_position_tolerance_m：终端稳定容差。
# 输出：
#   damping_required：当前是否进入提前制动范围。
def _semantic_approach_damping_required(
    *,
    planned_goal_distance_m: float,
    planned_speed_mps: float,
    maximum_acceleration_mps2: float,
    waypoint_position_tolerance_m: float,
) -> bool:
    """Begin braking early enough to enter the strict waypoint settle gate."""

    values = (
        planned_goal_distance_m,
        planned_speed_mps,
        maximum_acceleration_mps2,
        waypoint_position_tolerance_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("semantic approach damping inputs must be finite")
    if (
        planned_goal_distance_m < 0.0
        or planned_speed_mps < 0.0
        or maximum_acceleration_mps2 <= 0.0
        or waypoint_position_tolerance_m <= 0.0
    ):
        raise ValueError("semantic approach damping inputs are outside physical bounds")
    stopping_distance_m = planned_speed_mps**2 / (2.0 * maximum_acceleration_mps2)
    damping_distance_m = max(
        0.8,
        stopping_distance_m + waypoint_position_tolerance_m + 0.15,
    )
    return planned_goal_distance_m <= damping_distance_m


# 功能：
#   窄净空、终端制动或接近带动作目标时进入精细模式，其他已验证场景使用巡航模式。
# 输入：
#   planned_goal_distance_m：当前参考点距目标的米数。
#   action_checkpoint_goal：目标是否含设备动作。
#   semantic_approach_damping_active：是否已进入终端制动区。
#   tight_clearance_segment：是否是已经绑定的窄净空路段。
#   precision_approach_distance_m：带动作目标的精细接近距离。
# 输出：
#   profile：precision 或 cruise 控制模式。
def _navigation_control_profile(
    *,
    planned_goal_distance_m: float,
    action_checkpoint_goal: bool,
    semantic_approach_damping_active: bool,
    tight_clearance_segment: bool = False,
    precision_approach_distance_m: float = 4.0,
) -> str:
    """Select deliberate micro-control near action-bearing local spaces."""

    if (
        not math.isfinite(planned_goal_distance_m)
        or planned_goal_distance_m < 0.0
        or not math.isfinite(precision_approach_distance_m)
        or precision_approach_distance_m <= 0.0
    ):
        raise ValueError("navigation control profile inputs are outside physical bounds")
    if (
        tight_clearance_segment
        or semantic_approach_damping_active
        or (action_checkpoint_goal and planned_goal_distance_m <= precision_approach_distance_m)
    ):
        return "precision"
    return "cruise"


# 功能：
#   1. 从已绑定的 Gazebo 目标获取新观测，生成带净空检查和时效限制的短程跟随目标。
#   2. 已配置本地控制时复用安全／模型控制入口，要求模型的任务不允许退回直接位置控制。
#   3. 保留明确的非模型兼容模式；旧静态轨迹进度不作为动态跟随的剩余路段证据。
# 输入：
#   args：本次运行的控制通道、时序、资产及输出路径。
#   base：基础飞控执行器模块。
#   client：飞控和目标观测客户端。
#   params：供运行期改令使用的控制器参数。
#   runtime_session：当前改令会话。
#   replacement：目标身份、跟随参数和坐标绑定。
#   initial_setpoint：进入跟随时的稳定设定值。
#   phase_path：执行阶段输出路径。
#   timing：累计执行及中断证据。
# 输出：
#   current_setpoint：跟随结束时最后应用的设定值。
async def _follow_runtime_target(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    params: Any,
    runtime_session: RuntimeControlSession | None,
    replacement: RuntimeTrackReplacement,
    initial_setpoint: Any,
    phase_path: Path,
    timing: dict[str, Any],
) -> Any:
    model_required = bool(getattr(args, "require_model_control_authority", False))
    local_control = getattr(args, "local_safety_command", None) is not None
    if model_required and (
        not local_control
        or getattr(args, "local_safety_required", False) is not True
        or getattr(args, "local_safety_target", None) is None
    ):
        raise UserDirectedLanding("dynamic follow requires the model control channel")
    if getattr(args, "local_safety_required", False) and not local_control:
        raise UserDirectedLanding("dynamic follow requires the local safety channel")
    if args.semantic is None or args.vehicle_metadata is None:
        raise UserDirectedLanding("dynamic follow requires semantic and vehicle metadata artifacts")
    vehicle = VehicleAsset.model_validate(read_runtime_object(args.vehicle_metadata))
    parameters = dict(replacement.amendment_parameters)
    duration = float(parameters.get("follow_duration_seconds", 30.0))
    update_rate_hz = float(parameters.get("target_update_rate_hz", 2.0))
    standoff_m = float(parameters.get("standoff_m", 2.0))
    altitude_offset_m = float(parameters.get("altitude_offset_m", 1.0))
    maximum_speed_mps = min(
        float(parameters.get("maximum_speed_mps", 1.0)),
        vehicle.max_speed_mps,
    )
    if not (
        1.0 <= duration <= 300.0
        and 0.5 <= update_rate_hz <= 10.0
        and 0.5 <= standoff_m <= 20.0
        and -5.0 <= altitude_offset_m <= 20.0
        and 0.1 <= maximum_speed_mps <= 3.0
    ):
        raise UserDirectedLanding(
            "dynamic follow parameters are outside the bounded safety contract"
        )

    control_period = 1.0 / args.setpoint_rate_hz
    observation_period = 1.0 / update_rate_hz
    maximum_observation_age = min(2.0, max(0.25, 2.0 * observation_period))
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + duration
    next_observation_at = started
    current_setpoint = initial_setpoint
    current_world = _setpoint_world_enu(current_setpoint, replacement.coordinate_contract)
    standoff_unit: tuple[float, float] | None = None
    observation_task: asyncio.Task[dict[str, float | str]] | None = None
    observation_started_at = started
    goal_deadline: float | None = None
    goal_sequence = 0
    # 完成静态接入后不再有可用的静态航点后缀，后续改令必须重新建立有效的路线来源。
    args._runtime_track_progress = None
    observations: list[dict[str, Any]] = []
    evidence_path = args.runtime_control_dir / "follow" / f"{replacement.message_id}.evidence.json"
    evidence: dict[str, Any] = {
        "schema_version": "dronedream.runtime-follow-evidence.v1",
        "message_id": replacement.message_id,
        "replacement_sequence": replacement.replacement_sequence,
        "target_pose_topic": parameters.get("target_pose_topic"),
        "parameters": parameters,
        "control_mode": "model-required"
        if model_required
        else ("local-safety" if local_control else "position-compatibility"),
        "maximum_target_request_age_seconds": maximum_observation_age,
        "semantic_sha256": hashlib.sha256(
            read_plugin_file(args.semantic, limit=MAX_RUNTIME_REPLACEMENT_BYTES)
        ).hexdigest(),
        "vehicle_asset_id": vehicle.asset_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "observations": observations,
    }
    _atomic_json(evidence_path, evidence)
    _atomic_json(
        phase_path,
        {
            "phase": "TRACK",
            "mode": "FOLLOW_TARGET",
            "message_id": replacement.message_id,
            "replacement_sequence": replacement.replacement_sequence,
        },
    )
    try:
        while loop.time() < deadline:
            base._raise_if_external_abort_requested(args.abort_file)
            interruption = _claim_runtime_message(args.runtime_control_dir, runtime_session)
            if interruption is not None:
                started_hold = loop.time()
                await _handle_runtime_interruption(
                    base=base,
                    client=client,
                    frozen_setpoint=current_setpoint,
                    interruption=interruption,
                    control_dir=args.runtime_control_dir,
                    phase="TRACK",
                    schedule_index=None,
                    abort_file=args.abort_file,
                    rate_hz=args.setpoint_rate_hz,
                    hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                    decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                    replan_hold_seconds=args.runtime_replan_hold_seconds,
                    active_track_sha256=replacement.track_sha256,
                    params=params,
                    semantic_path=args.semantic,
                    vehicle_metadata_path=args.vehicle_metadata,
                    coordinate_contract=replacement.coordinate_contract,
                    telemetry_args=args,
                )
                hold_duration = loop.time() - started_hold
                deadline += hold_duration
                next_observation_at = loop.time()
                # 暂停期间积压的目标请求不能在恢复后被当作新观测使用。
                if observation_task is not None:
                    observation_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await observation_task
                    observation_task = None
                goal_deadline = None
                observed = await client.sample_position_velocity_ned(0.2)
                current_setpoint = base.Setpoint(
                    observed.north_m,
                    observed.east_m,
                    observed.down_m,
                    current_setpoint.yaw_deg,
                )
                current_world = _setpoint_world_enu(
                    current_setpoint,
                    replacement.coordinate_contract,
                )
                timing["runtime_interruptions"].append(
                    {
                        "message_id": interruption.message.message_id,
                        "interrupted_phase": "FOLLOW_TARGET",
                        "outcome": "resume_follow_target",
                        "duration_seconds": hold_duration,
                    }
                )

            now = loop.time()
            if observation_task is None and now >= next_observation_at:
                observation_started_at = now
                observation_task = asyncio.create_task(client.sample_gazebo_pose(parameters))
                next_observation_at = now + observation_period
            if observation_task is not None and observation_task.done():
                sample = observation_task.result()
                observation_task = None
                goal_deadline = observation_started_at + maximum_observation_age
                _require_live_navigation_goal(goal_deadline, loop.time())
                goal_sequence += 1
                target = Vector3(
                    x=float(sample["x"]),
                    y=float(sample["y"]),
                    z=float(sample["z"]),
                )
                if standoff_unit is None:
                    delta_x = current_world.x - target.x
                    delta_y = current_world.y - target.y
                    horizontal = math.hypot(delta_x, delta_y)
                    standoff_unit = (
                        (delta_x / horizontal, delta_y / horizontal)
                        if horizontal > 1e-6
                        else (-1.0, 0.0)
                    )
                desired = Vector3(
                    x=target.x + standoff_unit[0] * standoff_m,
                    y=target.y + standoff_unit[1] * standoff_m,
                    z=target.z + altitude_offset_m,
                )
                distance = math.dist(
                    (current_world.x, current_world.y, current_world.z),
                    (desired.x, desired.y, desired.z),
                )
                max_step = maximum_speed_mps * observation_period
                if distance > max_step:
                    ratio = max_step / distance
                    desired = Vector3(
                        x=current_world.x + (desired.x - current_world.x) * ratio,
                        y=current_world.y + (desired.y - current_world.y) * ratio,
                        z=current_world.z + (desired.z - current_world.z) * ratio,
                    )
                route = GraphRoute(
                    start_node="runtime-follow-current",
                    goal_node="runtime-follow-command",
                    node_ids=["runtime-follow-current", "runtime-follow-command"],
                    edge_ids=["runtime-follow-segment"],
                    positions_m=[current_world, desired],
                    route_length_m=math.dist(
                        (current_world.x, current_world.y, current_world.z),
                        (desired.x, desired.y, desired.z),
                    ),
                    all_edges_flight_verified=False,
                )
                clearance = validate_route_clearance(
                    route,
                    args.semantic,
                    vehicle_diameter_m=vehicle.body_radius_m * 2.0,
                    vehicle_height_m=vehicle.body_height_m,
                )
                if not clearance.accepted:
                    raise UserDirectedLanding(
                        "dynamic follow clearance gate rejected the next command segment"
                    )
                current_world = desired
                current_setpoint = _world_enu_setpoint(
                    base=base,
                    world=desired,
                    yaw_deg=float(current_setpoint.yaw_deg),
                    coordinate_contract=replacement.coordinate_contract,
                )
                observations.append(
                    {
                        "elapsed_seconds": loop.time() - started,
                        "target_world_enu_m": target.model_dump(mode="json"),
                        "command_world_enu_m": desired.model_dump(mode="json"),
                        "clearance_sha256": sha256_json(clearance),
                        "minimum_clearance_m": clearance.minimum_clearance_m,
                    }
                )
                if len(observations) > 3_000:
                    del observations[:-3_000]
            if goal_deadline is None:
                _require_live_navigation_goal(
                    observation_started_at + maximum_observation_age,
                    loop.time(),
                )
                # 尚无新目标时只维持进入此阶段的稳定位置，不向猜测的目标运动。
                await client.set_position_ned(current_setpoint)
                await asyncio.sleep(control_period)
            else:
                _require_live_navigation_goal(goal_deadline, loop.time())
                if local_control:
                    current_setpoint = await _apply_local_safety(
                        args=args,
                        base=base,
                        client=client,
                        planned_setpoint=current_setpoint,
                        coordinate_contract=replacement.coordinate_contract,
                        phase_path=phase_path,
                        navigation_goal_position_m=current_world,
                        navigation_goal_id=f"follow-{replacement.message_id}-{goal_sequence}",
                        navigation_goal_deadline_monotonic=goal_deadline,
                        control_profile="precision",
                    )
                else:
                    await client.set_position_ned(current_setpoint)
                    await asyncio.sleep(control_period)
        if not observations:
            raise UserDirectedLanding("dynamic follow completed without target observations")
        evidence["status"] = "complete"
        evidence["completed_at"] = datetime.now(UTC).isoformat()
        evidence["final_command_world_enu_m"] = current_world.model_dump(mode="json")
        _atomic_json(evidence_path, evidence)
        timing["runtime_interruptions"].append(
            {
                "message_id": replacement.message_id,
                "outcome": "follow_target_complete",
                "observation_count": len(observations),
                "evidence_sha256": sha256_json(evidence),
            }
        )
        return current_setpoint
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["failure"] = f"{type(exc).__name__}: {exc}"
        evidence["failed_at"] = datetime.now(UTC).isoformat()
        _atomic_json(evidence_path, evidence)
        raise
    finally:
        if observation_task is not None and not observation_task.done():
            observation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await observation_task


# 功能：
#   解析本次执行所需路径、时序、模型权限和兼容模式选项，不在参数解析期间连接飞控。
# 输入：
#   无显式参数；读取当前进程命令行。
# 输出：
#   args：待启动前校验的运行配置。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--abort-file", type=Path, required=True)
    parser.add_argument("--setpoint-rate-hz", type=float, required=True)
    parser.add_argument("--takeoff-timeout-seconds", type=float, required=True)
    parser.add_argument("--takeoff-climb-rate-m-s", type=float, required=True)
    parser.add_argument("--track-timeout-seconds", type=float, required=True)
    parser.add_argument("--landing-timeout-seconds", type=float, required=True)
    parser.add_argument("--takeoff-stable-window-seconds", type=float, required=True)
    parser.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="measured-hold",
    )
    parser.add_argument("--maximum-yaw-rate-deg-s", type=float, default=20.0)
    parser.add_argument("--gazebo-vehicle-model-name")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--connection", default="udpin://0.0.0.0:14540")
    parser.add_argument("--base-executor", type=Path, required=True)
    parser.add_argument("--checkpoint-contract", type=Path)
    parser.add_argument("--runtime-action-contract", type=Path)
    parser.add_argument("--checkpoint-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--runtime-control-dir", type=Path)
    parser.add_argument("--runtime-hold-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--runtime-decision-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--runtime-replan-hold-seconds", type=float, default=60.0)
    parser.add_argument("--local-safety-command", type=Path)
    parser.add_argument("--local-safety-observation", type=Path)
    parser.add_argument("--local-safety-channel", type=Path)
    parser.add_argument("--native-state-channel", type=Path)
    parser.add_argument("--perception-health-channel", type=Path)
    parser.add_argument("--runtime-phase-channel", type=Path, action="append", default=[])
    parser.add_argument("--local-safety-target", type=Path)
    parser.add_argument("--local-safety-required", action="store_true")
    parser.add_argument("--require-model-control-authority", action="store_true")
    parser.add_argument("--simulation-teacher-control", action="store_true")
    parser.add_argument("--local-safety-command-grace-seconds", type=float, default=8.0)
    parser.add_argument("--local-safety-runtime-stale-grace-seconds", type=float, default=8.0)
    parser.add_argument("--local-safety-repair-timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--local-safety-repair-absolute-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument("--tracking-lag-limit-m", type=float, default=0.75)
    parser.add_argument("--tracking-rejoin-tolerance-m", type=float, default=0.35)
    parser.add_argument("--tracking-corridor-policy", type=Path)
    parser.add_argument("--model-progress-slack-m", type=float, default=0.0)
    parser.add_argument("--tracking-sample-rate-hz", type=float, default=4.0)
    parser.add_argument("--tracking-telemetry-timeout-seconds", type=float, default=0.5)
    parser.add_argument(
        "--tracking-telemetry-recovery-timeout-seconds",
        type=float,
        default=5.0,
        help=("maximum fail-closed hold while replacing one stalled PX4 telemetry subscription"),
    )
    parser.add_argument("--tracking-recovery-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--tracking-recovery-assist-gain-s-inverse",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--tracking-recovery-assist-max-speed-mps",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--tracking-recovery-assist-velocity-damping",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--tracking-recovery-assist-deadband-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--semantic-progress-recovery-timeout-seconds",
        type=float,
        default=20.0,
        help=(
            "Measured no-progress interval before requesting the recovery expert; "
            "ordinary tracking recovery uses a separate timeout."
        ),
    )
    parser.add_argument(
        "--semantic-progress-abort-timeout-seconds",
        type=float,
        default=60.0,
        help=(
            "Measured no-progress interval before a controlled landing after the "
            "recovery expert has had a bounded opportunity to act."
        ),
    )
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--vehicle-metadata", type=Path)
    return parser.parse_args()


# 功能：
#   在维持稳定位置和实际遥测的同时等待模型检查点决定，验证请求摘要并接受外部中断。
# 输入：
#   base：基础控制维持工具。
#   client：实际飞控客户端。
#   setpoint：检查点稳定目标。
#   request：已发布的检查点验证请求。
#   decision_path：对应决定文件路径。
#   abort_file：外部终止信号。
#   rate_hz：控制维持频率。
#   timeout_seconds：等待模型决定时限。
#   runtime_interrupt_probe：用户改令探针。
#   sample_observer：实际位置遥测发布回调。
#   setpoint_refresh：模型审核等待期间的实时本地控制回调。
# 输出：
#   decision：绑定该请求的检查点决定，其继续授权仍由调用方检查。
async def _wait_checkpoint_decision(
    *,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    request: RuntimeCheckpointRequest,
    decision_path: Path,
    abort_file: Path,
    rate_hz: float,
    timeout_seconds: float,
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None] | None = None,
    sample_observer: Callable[[Any], None] | None = None,
    setpoint_refresh: Callable[[Any], Awaitable[Any]] | None = None,
) -> RuntimeCheckpointDecision:
    deadline = time.monotonic() + timeout_seconds
    request_hash = sha256_json(request)

    # 功能：
    #   在等待读数和最终接纳决定前继续响应终止文件与用户改令。
    # 输入：
    #   无显式参数；使用当前 abort_file、base 和 runtime_interrupt_probe。
    # 输出：
    #   None：不返回业务数据。
    def check_checkpoint_interruption() -> None:
        base._raise_if_external_abort_requested(abort_file)
        if runtime_interrupt_probe is not None:
            interruption = runtime_interrupt_probe()
            if interruption is not None:
                raise interruption

    while time.monotonic() < deadline:
        check_checkpoint_interruption()
        if decision_path.is_file():
            decision = RuntimeCheckpointDecision.model_validate(read_runtime_object(decision_path))
            if decision.request_sha256 != request_hash:
                raise RuntimeError("checkpoint decision request hash mismatch")
            return decision
        if sample_observer is None:
            hold_setpoint = (
                await setpoint_refresh(setpoint) if setpoint_refresh is not None else setpoint
            )
            check_checkpoint_interruption()
            await client.set_position_ned(hold_setpoint)
            await asyncio.sleep(1.0 / rate_hz)
        else:
            observed = await base._await_with_setpoint_keepalive(
                client,
                client.sample_position_velocity_ned(1.0),
                hold_setpoint=setpoint,
                rate_hz=rate_hz,
                abort_check=check_checkpoint_interruption,
                **({"setpoint_refresh": setpoint_refresh} if setpoint_refresh is not None else {}),
            )
            sample_observer(observed)
    raise TimeoutError(f"checkpoint decision timeout: {request.checkpoint.checkpoint_id}")


# 功能：
#   1. 在替换任务的检查点维持本地安全控制，采集位置和电池证据并执行继续门控。
#   2. 仅接受绑定本次请求的模型决定，执行该检查点动作并累计等待时间。
# 输入：
#   args：运行目录、控制频率和检查点期限。
#   base：基础飞控等待与终止工具。
#   client：实际飞控客户端。
#   setpoint：检查点目标。
#   checkpoint：本次检查点定义。
#   checkpoint_contract：替换任务的检查点契约。
#   runtime_action_contract：替换任务的设备动作契约。
#   track：冻结的替换轨迹及稳定门槛。
#   completed_action_step_ids：已完成的设备步骤集合。
#   completed_action_task_ids：已完成的任务集合。
#   runtime_interrupt_probe：用户改令探针。
#   setpoint_refresh：重新进行本地控制仲裁的回调。
#   sample_observer：实际位置证据发布回调。
#   target_frame_position_resolver：实际位置到路线参考系的转换回调。
#   timing：本次运行持续更新的检查点与耗时记录。
# 输出：
#   None：不返回业务数据。
async def _review_replacement_checkpoint(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    setpoint: Any,
    checkpoint: Any,
    checkpoint_contract: RuntimeCheckpointContract,
    runtime_action_contract: RuntimeActionExecutionContract,
    track: Px4Track,
    completed_action_step_ids: set[str],
    completed_action_task_ids: set[str],
    runtime_interrupt_probe: Callable[[], RuntimeInterruptDetected | None],
    setpoint_refresh: Callable[[Any], Awaitable[Any]],
    sample_observer: Callable[[Any], None],
    target_frame_position_resolver: Callable[[Any], tuple[float, float, float]],
    timing: dict[str, Any],
) -> None:
    # 功能：
    #   在检查点等待中检查外部终止和用户改令，及时中断当前等待。
    # 输入：
    #   无显式参数；读取 args 并调用 runtime_interrupt_probe。
    # 输出：
    #   None：不返回业务数据。
    def abort_check() -> None:
        base._raise_if_external_abort_requested(args.abort_file)
        interruption = runtime_interrupt_probe()
        if interruption is not None:
            raise interruption

    observed, position_error, speed = await _wait_checkpoint_stable(
        base=base,
        client=client,
        setpoint=setpoint,
        rate_hz=args.setpoint_rate_hz,
        timeout_seconds=track.waypoint_settle_timeout_seconds,
        stable_window_seconds=track.waypoint_stable_window_seconds,
        position_tolerance_m=_payload_aware_waypoint_position_tolerance_m(
            configured_tolerance_m=track.waypoint_position_tolerance_m,
            runtime_action_contract=runtime_action_contract,
            completed_action_step_ids=completed_action_step_ids,
        ),
        speed_tolerance_mps=track.waypoint_speed_tolerance_mps,
        abort_check=abort_check,
        setpoint_refresh=setpoint_refresh,
        sample_observer=sample_observer,
        target_frame_position_resolver=target_frame_position_resolver,
    )
    battery = await _sample_checkpoint_battery(
        base=base,
        client=client,
        setpoint=setpoint,
        rate_hz=args.setpoint_rate_hz,
        runtime_interrupt_probe=runtime_interrupt_probe,
        sample_observer=sample_observer,
        setpoint_refresh=setpoint_refresh,
        safety_abort_check=abort_check,
    )
    raw_battery = float(battery["remaining_percent"])
    battery_percent = raw_battery * 100.0 if raw_battery <= 1.0 else raw_battery
    gates = {
        "position_error_within_0_75_m": position_error <= 0.75,
        "speed_within_0_50_mps": speed <= 0.5,
        "battery_above_10_percent": battery_percent > 10.0,
        "telemetry_finite": all(
            math.isfinite(value)
            for value in (
                observed.north_m,
                observed.east_m,
                observed.down_m,
                speed,
                battery_percent,
            )
        ),
    }
    request = RuntimeCheckpointRequest(
        contract_id=checkpoint_contract.contract_id,
        checkpoint=checkpoint,
        observed_position_ned_m=Vector3(
            x=observed.north_m,
            y=observed.east_m,
            z=observed.down_m,
        ),
        observed_velocity_ned_mps=Vector3(
            x=observed.north_m_s,
            y=observed.east_m_s,
            z=observed.down_m_s,
        ),
        commanded_position_ned_m=Vector3(
            x=setpoint.north_m,
            y=setpoint.east_m,
            z=setpoint.down_m,
        ),
        position_error_m=position_error,
        speed_mps=speed,
        battery_percent=battery_percent,
        deterministic_gates=gates,
    )
    request_path = args.run_dir / "checkpoints" / f"{checkpoint.checkpoint_id}.request.json"
    decision_path = args.run_dir / "checkpoints" / f"{checkpoint.checkpoint_id}.decision.json"
    _atomic_json(request_path, request)
    if not all(gates.values()):
        raise RuntimeError(
            f"deterministic replacement checkpoint gate failed: {checkpoint.checkpoint_id}"
        )
    started = time.monotonic()
    decision = await _wait_checkpoint_decision(
        base=base,
        client=client,
        setpoint=setpoint,
        request=request,
        decision_path=decision_path,
        abort_file=args.abort_file,
        rate_hz=args.setpoint_rate_hz,
        timeout_seconds=args.checkpoint_timeout_seconds,
        runtime_interrupt_probe=runtime_interrupt_probe,
        sample_observer=sample_observer,
        setpoint_refresh=setpoint_refresh,
    )
    timing["checkpoints"].append(
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "request_sha256": sha256_json(request),
            "decision_sha256": sha256_json(decision),
            "continue_authorized": decision.continue_authorized,
            "assessment_action": decision.assessment.action,
            "runtime_revision": True,
        }
    )
    if not decision.continue_authorized or decision.assessment.action != "accept":
        raise RuntimeError(
            f"replacement checkpoint continuation rejected: {checkpoint.checkpoint_id}"
        )
    await _execute_triggered_runtime_actions(
        base=base,
        client=client,
        setpoint=setpoint,
        contract=runtime_action_contract,
        trigger="checkpoint",
        checkpoint_id=checkpoint.checkpoint_id,
        completed_step_ids=completed_action_step_ids,
        completed_task_ids=completed_action_task_ids,
        run_dir=args.run_dir,
        abort_file=args.abort_file,
        rate_hz=args.setpoint_rate_hz,
        runtime_interrupt_probe=runtime_interrupt_probe,
        timing=timing,
        setpoint_refresh=setpoint_refresh,
        sample_observer=sample_observer,
        target_frame_position_resolver=target_frame_position_resolver,
    )
    timing.setdefault("replacement_checkpoint_hold_seconds", 0.0)
    timing["replacement_checkpoint_hold_seconds"] += time.monotonic() - started


# 功能：
#   1. 接管完整的替换任务契约，以本地安全仲裁和实际遥测推进路线及设备动作。
#   2. 在途中继续处理用户改令、检查点审核和有界恢复，拒绝不完整替换任务。
# 输入：
#   args：本次执行的控制配置和证据目录。
#   base：基础飞控及调度工具。
#   client：实际飞控客户端。
#   params：控制器参数。
#   runtime_session：用户改令所属会话。
#   phase_path：执行阶段发布路径。
#   timing：持续追加的控制、检查点和中断记录。
#   initial：首次接管的完整替换任务。
#   completed_action_step_ids：跨替换保留的已执行步骤集合。
#   completed_action_task_ids：跨替换保留的已执行任务集合。
# 输出：
#   outcome：替换任务正常完成时的结果标识。
async def _fly_runtime_replacement(
    *,
    args: argparse.Namespace,
    base: ModuleType,
    client: Any,
    params: Any,
    runtime_session: RuntimeControlSession | None,
    phase_path: Path,
    timing: dict[str, Any],
    initial: RuntimeTrackReplacement,
    completed_action_step_ids: set[str],
    completed_action_task_ids: set[str],
) -> str:
    replacement = initial
    loop = asyncio.get_running_loop()
    while True:
        artifact = replacement.artifact
        if (
            artifact is None
            or artifact.revised_task_graph is None
            or artifact.runtime_checkpoints is None
            or artifact.runtime_actions is None
        ):
            raise UserDirectedLanding(
                "runtime replacement has no complete task/action/checkpoint revision"
            )
        active_track_sha256 = replacement.track_sha256
        args._runtime_track_progress = RuntimeTrackProgress(
            track_sha256=active_track_sha256,
            next_track_point_index=1,
        )
        schedule = replacement.schedule
        waypoint_arrival_indices = set(replacement.waypoint_arrival_indices)
        runtime_action_contract = artifact.runtime_actions
        checkpoint_contract = artifact.runtime_checkpoints
        action_checkpoint_ids = {
            str(step.checkpoint_id)
            for step in runtime_action_contract.steps
            if step.trigger == "checkpoint" and step.checkpoint_id is not None
        }
        action_checkpoint_point_indices = {
            item.track_point_index
            for item in checkpoint_contract.checkpoints
            if item.checkpoint_id in action_checkpoint_ids
        }
        (
            active_navigation_goal,
            active_navigation_goal_id,
        ) = _navigation_goal_for_schedule(
            track=artifact.track,
            waypoint_arrival_indices=tuple(replacement.waypoint_arrival_indices),
            schedule_index=0,
        )
        active_navigation_goal_point_index = int(active_navigation_goal_id.rsplit("-", 1)[-1])
        active_action_checkpoint_goal = (
            active_navigation_goal_point_index in action_checkpoint_point_indices
        )
        active_navigation_context: dict[str, Any] = {
            "position_m": active_navigation_goal,
            "goal_id": active_navigation_goal_id,
            "action_checkpoint_goal": active_action_checkpoint_goal,
        }
        points = [
            base.TrackPoint(point.x, point.y, point.z, point.speed_limit_mps)
            for point in artifact.track.points
        ]
        checkpoint_indices = _schedule_checkpoint_indices(
            base,
            schedule,
            points,
            checkpoint_contract,
            0,
            waypoint_arrival_indices=tuple(replacement.waypoint_arrival_indices),
        )
        timing["runtime_action_supersessions"] = [
            *timing.get("runtime_action_supersessions", []),
            {
                "replacement_sequence": replacement.replacement_sequence,
                "superseded_step_ids": artifact.superseded_runtime_action_step_ids,
                "runtime_actions_sha256": sha256_json(runtime_action_contract),
                "runtime_checkpoints_sha256": sha256_json(checkpoint_contract),
                "task_graph_sha256": sha256_json(artifact.revised_task_graph),
            },
        ]

        # 功能：
        #   从当前会话的收件目录领取用户改令。
        # 输入：
        #   无显式参数；读取 args.runtime_control_dir 和 runtime_session。
        # 输出：
        #   interruption：已领取的中断，暂无改令时为 None。
        def runtime_interrupt_probe() -> RuntimeInterruptDetected | None:
            return _claim_runtime_message(args.runtime_control_dir, runtime_session)

        # 功能：
        #   在替换路线稳定等待时响应终止文件或新的用户改令。
        # 输入：
        #   无显式参数；使用当前路线的终止配置和改令探针。
        # 输出：
        #   None：不返回业务数据。
        def settle_abort_check() -> None:
            base._raise_if_external_abort_requested(args.abort_file)
            interruption = runtime_interrupt_probe()
            if interruption is not None:
                raise interruption

        # 功能：
        #   根据当前替换目标和实时安全信息重新仲裁精细动作，避免等待时绕过本地控制。
        # 输入：
        #   planned_setpoint：本轮计划位置。
        #   coordinate_contract：该替换任务冻结的坐标约定。
        #   navigation_context：随路线推进更新的目标与检查点上下文。
        # 输出：
        #   applied_setpoint：经本地控制门控后实际下发的控制目标。
        async def live_settle_setpoint_refresh(
            planned_setpoint: Any,
            coordinate_contract: Px4CoordinateContract = replacement.coordinate_contract,
            navigation_context: dict[str, Any] = active_navigation_context,
        ) -> Any:
            return await _apply_local_safety(
                args=args,
                base=base,
                client=client,
                planned_setpoint=planned_setpoint,
                coordinate_contract=coordinate_contract,
                phase_path=phase_path,
                navigation_goal_position_m=navigation_context["position_m"],
                navigation_goal_id=str(navigation_context["goal_id"]),
                control_profile="precision",
                action_checkpoint_goal=bool(navigation_context["action_checkpoint_goal"]),
                tracking_recovery_active=True,
            )

        # 功能：
        #   将替换任务稳定等待期间的实际遥测发布到身份与动力学证据链。
        # 输入：
        #   observed：本次飞控位置和速度样本。
        #   coordinate_contract：该替换任务冻结的坐标约定。
        # 输出：
        #   None：不返回业务数据。
        def live_settle_sample_observer(
            observed: Any,
            coordinate_contract: Px4CoordinateContract = replacement.coordinate_contract,
        ) -> None:
            _publish_px4_identity_telemetry(
                args=args,
                coordinate_contract=coordinate_contract,
                observed=observed,
                dynamics_telemetry=_latest_px4_dynamics_telemetry(client),
            )

        # 功能：
        #   将实际遥测位置换算到替换轨迹使用的相对 NED 参考系。
        # 输入：
        #   observed：飞控实际位置样本。
        # 输出：
        #   position_ned：路线参考系中的北、东、下位置三元组，单位米。
        def live_settle_target_frame_position(
            observed: Any,
        ) -> tuple[float, float, float]:
            return _observed_route_frame_position_ned(args=args, observed=observed)

        timing["runtime_interruptions"].append(
            {
                "message_id": replacement.message_id,
                "outcome": "replacement_track_adopted",
                "replacement_sequence": replacement.replacement_sequence,
                "track_sha256": replacement.track_sha256,
                "setpoint_count": len(schedule),
            }
        )
        _atomic_json(
            phase_path,
            {
                "phase": "TRACK",
                "checkpoint_id": None,
                "replacement_sequence": replacement.replacement_sequence,
            },
        )
        deadline = loop.time() + args.track_timeout_seconds
        last_setpoint = schedule[0]
        try:
            while True:
                try:
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "ACTION",
                            "checkpoint_id": None,
                            "trigger": "post-replan",
                            "replacement_sequence": replacement.replacement_sequence,
                        },
                    )
                    await _execute_triggered_runtime_actions(
                        base=base,
                        client=client,
                        setpoint=last_setpoint,
                        contract=runtime_action_contract,
                        trigger="post-replan",
                        checkpoint_id=None,
                        completed_step_ids=completed_action_step_ids,
                        completed_task_ids=completed_action_task_ids,
                        run_dir=args.run_dir,
                        abort_file=args.abort_file,
                        rate_hz=args.setpoint_rate_hz,
                        runtime_interrupt_probe=runtime_interrupt_probe,
                        timing=timing,
                        setpoint_refresh=live_settle_setpoint_refresh,
                        sample_observer=live_settle_sample_observer,
                    )
                    break
                except RuntimeInterruptDetected as interruption:
                    await _handle_runtime_interruption(
                        base=base,
                        client=client,
                        frozen_setpoint=last_setpoint,
                        interruption=interruption,
                        control_dir=args.runtime_control_dir,
                        phase="ACTION",
                        schedule_index=None,
                        abort_file=args.abort_file,
                        rate_hz=args.setpoint_rate_hz,
                        hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                        decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                        replan_hold_seconds=args.runtime_replan_hold_seconds,
                        active_track_sha256=active_track_sha256,
                        params=params,
                        semantic_path=args.semantic,
                        vehicle_metadata_path=args.vehicle_metadata,
                        coordinate_contract=replacement.coordinate_contract,
                        telemetry_args=args,
                    )
            tracking_sample_interval = _tracking_sample_interval(args)
            for index, setpoint in enumerate(schedule):
                args._runtime_track_progress = RuntimeTrackProgress(
                    track_sha256=active_track_sha256,
                    next_track_point_index=next_track_point_index(
                        tuple(replacement.waypoint_arrival_indices),
                        index,
                        len(artifact.track.points),
                    ),
                )
                checkpoint = checkpoint_indices.get(index)
                recovery_started: float | None = None
                local_recovery_control_active = False
                recovery_window: dict[str, Any] | None = None
                maximum_tracking_error_m = 0.0
                while True:
                    base._raise_if_external_abort_requested(args.abort_file)
                    # Recovery is a bounded safety hold, not consumed mission
                    # motion.  Its own deadline below remains authoritative.
                    if recovery_started is None and loop.time() >= deadline:
                        raise TimeoutError("runtime replacement track timeout")
                    interruption = _claim_runtime_message(args.runtime_control_dir, runtime_session)
                    if interruption is not None:
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "HOLDING",
                                "interrupted_phase": "TRACK",
                                "message_id": interruption.message.message_id,
                                "replacement_sequence": replacement.replacement_sequence,
                            },
                        )
                        interruption_started = loop.time()
                        await _handle_runtime_interruption(
                            base=base,
                            client=client,
                            frozen_setpoint=last_setpoint,
                            interruption=interruption,
                            control_dir=args.runtime_control_dir,
                            phase="TRACK",
                            schedule_index=index,
                            abort_file=args.abort_file,
                            rate_hz=args.setpoint_rate_hz,
                            hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                            decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                            replan_hold_seconds=args.runtime_replan_hold_seconds,
                            active_track_sha256=active_track_sha256,
                            params=params,
                            semantic_path=args.semantic,
                            vehicle_metadata_path=args.vehicle_metadata,
                            coordinate_contract=replacement.coordinate_contract,
                            telemetry_args=args,
                        )
                        deadline += loop.time() - interruption_started
                        timing["runtime_interruptions"].append(
                            {
                                "message_id": interruption.message.message_id,
                                "outcome": "resume_replacement_track",
                                "replacement_sequence": replacement.replacement_sequence,
                            }
                        )
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "TRACK",
                                "checkpoint_id": None,
                                "replacement_sequence": replacement.replacement_sequence,
                            },
                        )
                    sample_now = (
                        recovery_started is not None
                        or index % tracking_sample_interval == 0
                        or index in waypoint_arrival_indices
                        or checkpoint is not None
                    )
                    navigation_goal, navigation_goal_id = _navigation_goal_for_schedule(
                        track=artifact.track,
                        waypoint_arrival_indices=tuple(replacement.waypoint_arrival_indices),
                        schedule_index=index,
                    )
                    planned_velocity_ned_mps = _schedule_velocity_feedforward(
                        current_setpoint=setpoint,
                        next_setpoint=schedule[min(index + 1, len(schedule) - 1)],
                        rate_hz=args.setpoint_rate_hz,
                        speed_limit_mps=float(params.vel_limit),
                        # Preserve the qualified route tangent for the
                        # tracking gate. Recovery actuation still replaces
                        # this with bounded radial feed-forward below.
                        recovery_active=False,
                    )
                    planned_goal_world = _setpoint_world_enu(
                        setpoint,
                        replacement.coordinate_contract,
                    )
                    planned_navigation_goal_distance_m = math.dist(
                        (
                            planned_goal_world.x,
                            planned_goal_world.y,
                            planned_goal_world.z,
                        ),
                        (navigation_goal.x, navigation_goal.y, navigation_goal.z),
                    )
                    semantic_approach_damping_active = _semantic_approach_damping_required(
                        planned_goal_distance_m=planned_navigation_goal_distance_m,
                        # The model-authorized controller may legitimately
                        # run ahead of the slower global reference.  Size
                        # the braking envelope for that full authorized
                        # speed, otherwise damping starts after the real
                        # aircraft has already entered the settle radius.
                        planned_speed_mps=max(
                            math.sqrt(sum(value * value for value in planned_velocity_ned_mps)),
                            float(params.vel_limit),
                        ),
                        maximum_acceleration_mps2=float(params.accel_limit),
                        waypoint_position_tolerance_m=(
                            artifact.track.waypoint_position_tolerance_m
                        ),
                    )
                    navigation_goal_point_index = int(navigation_goal_id.rsplit("-", 1)[-1])
                    action_checkpoint_goal = (
                        navigation_goal_point_index in action_checkpoint_point_indices
                    )
                    active_navigation_goal = navigation_goal
                    active_navigation_goal_id = navigation_goal_id
                    active_action_checkpoint_goal = action_checkpoint_goal
                    active_navigation_context.update(
                        position_m=active_navigation_goal,
                        goal_id=active_navigation_goal_id,
                        action_checkpoint_goal=active_action_checkpoint_goal,
                    )
                    control_profile = _navigation_control_profile(
                        planned_goal_distance_m=planned_navigation_goal_distance_m,
                        action_checkpoint_goal=action_checkpoint_goal,
                        semantic_approach_damping_active=(semantic_approach_damping_active),
                    )
                    may_advance, tracking_error_m = await _tracking_gate_tick(
                        args=args,
                        base=base,
                        client=client,
                        planned_setpoint=setpoint,
                        coordinate_contract=replacement.coordinate_contract,
                        phase_path=phase_path,
                        schedule_index=index,
                        sample_now=sample_now,
                        recovery_active=recovery_started is not None,
                        local_recovery_control_active=(local_recovery_control_active),
                        planned_velocity_ned_mps=planned_velocity_ned_mps,
                        context={
                            "replacement_sequence": replacement.replacement_sequence,
                            "navigation_goal_position_m": navigation_goal.model_dump(mode="json"),
                            "navigation_goal_id": navigation_goal_id,
                            "semantic_approach_damping_active": (semantic_approach_damping_active),
                            "control_profile": control_profile,
                            "action_checkpoint_goal": action_checkpoint_goal,
                        },
                    )
                    if tracking_error_m is not None:
                        maximum_tracking_error_m = max(maximum_tracking_error_m, tracking_error_m)
                    if may_advance:
                        if recovery_started is not None:
                            recovery_duration = loop.time() - recovery_started
                            deadline += recovery_duration
                            timing.setdefault("tracking_recoveries", []).append(
                                {
                                    "schedule_index": index,
                                    "replacement_sequence": replacement.replacement_sequence,
                                    "duration_seconds": recovery_duration,
                                    "maximum_position_error_m": maximum_tracking_error_m,
                                }
                            )
                            _atomic_json(
                                phase_path,
                                {
                                    "phase": "TRACK",
                                    "checkpoint_id": None,
                                    "replacement_sequence": (replacement.replacement_sequence),
                                },
                            )
                        break
                    if tracking_error_m is None:
                        # Model-required control deliberately withholds schedule
                        # authority while no live model path lease exists. This
                        # is a bounded provider wait, not a tracking error and
                        # must never switch to deterministic route recovery.
                        continue
                    local_recovery_control_active = bool(
                        getattr(
                            args,
                            "_last_tracking_local_recovery_control_required",
                            True,
                        )
                    )
                    if recovery_started is None:
                        recovery_started = loop.time()
                    recovery_now = loop.time()
                    recovery_window, recovery_expired = _advance_tracking_recovery_window(
                        now=recovery_now,
                        timeout_seconds=args.tracking_recovery_timeout_seconds,
                        state=recovery_window,
                        tracking_error_m=tracking_error_m,
                        model_goal_distance_m=getattr(
                            args,
                            "_last_tracking_model_goal_distance_m",
                            None,
                        ),
                        progress_epsilon_m=max(
                            0.003,
                            min(0.01, args.tracking_rejoin_tolerance_m * 0.10),
                        ),
                    )
                    _publish_tracking_recovery_window(
                        args=args,
                        schedule_index=index,
                        state=recovery_window,
                        now=recovery_now,
                        replacement_sequence=replacement.replacement_sequence,
                    )
                    if recovery_expired:
                        raise UserDirectedLanding(
                            "closed-loop tracking recovery exceeded its bounded window"
                        )
                last_setpoint = setpoint
                if index in waypoint_arrival_indices and checkpoint is None:
                    settle_started = loop.time()
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "WAYPOINT_SETTLE",
                            "schedule_index": index,
                            "replacement_sequence": replacement.replacement_sequence,
                        },
                    )
                    while True:
                        try:
                            _, position_error, speed = await _wait_checkpoint_stable(
                                base=base,
                                client=client,
                                setpoint=setpoint,
                                rate_hz=args.setpoint_rate_hz,
                                timeout_seconds=(artifact.track.waypoint_settle_timeout_seconds),
                                # Replacement waypoints use the same model/depth-authoritative
                                # precision controller as the original track.  Retain the
                                # ordinary no-progress deadline, but permit a longer finite
                                # hard window while measured convergence continues.
                                absolute_timeout_factor=8.0,
                                stable_window_seconds=(
                                    artifact.track.waypoint_stable_window_seconds
                                ),
                                position_tolerance_m=(
                                    _payload_aware_waypoint_position_tolerance_m(
                                        configured_tolerance_m=(
                                            artifact.track.waypoint_position_tolerance_m
                                        ),
                                        runtime_action_contract=runtime_action_contract,
                                        completed_action_step_ids=completed_action_step_ids,
                                    )
                                ),
                                speed_tolerance_mps=(artifact.track.waypoint_speed_tolerance_mps),
                                abort_check=settle_abort_check,
                                setpoint_refresh=live_settle_setpoint_refresh,
                                sample_observer=live_settle_sample_observer,
                                target_frame_position_resolver=(live_settle_target_frame_position),
                            )
                            break
                        except RuntimeInterruptDetected as interruption:
                            await _handle_runtime_interruption(
                                base=base,
                                client=client,
                                frozen_setpoint=setpoint,
                                interruption=interruption,
                                control_dir=args.runtime_control_dir,
                                phase="WAYPOINT_SETTLE",
                                schedule_index=index,
                                abort_file=args.abort_file,
                                rate_hz=args.setpoint_rate_hz,
                                hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                                decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                                replan_hold_seconds=args.runtime_replan_hold_seconds,
                                active_track_sha256=active_track_sha256,
                                params=params,
                                semantic_path=args.semantic,
                                vehicle_metadata_path=args.vehicle_metadata,
                                coordinate_contract=replacement.coordinate_contract,
                                telemetry_args=args,
                            )
                    settle_duration = loop.time() - settle_started
                    deadline += settle_duration
                    timing.setdefault("waypoint_settles", []).append(
                        {
                            "schedule_index": index,
                            "replacement_sequence": replacement.replacement_sequence,
                            "position_error_m": position_error,
                            "speed_mps": speed,
                            "duration_seconds": settle_duration,
                        }
                    )
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "TRACK",
                            "checkpoint_id": None,
                            "replacement_sequence": replacement.replacement_sequence,
                        },
                    )
                if checkpoint is not None:
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "CHECKPOINT",
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "replacement_sequence": replacement.replacement_sequence,
                        },
                    )
                    while True:
                        try:
                            await _review_replacement_checkpoint(
                                args=args,
                                base=base,
                                client=client,
                                setpoint=setpoint,
                                checkpoint=checkpoint,
                                checkpoint_contract=checkpoint_contract,
                                runtime_action_contract=runtime_action_contract,
                                track=artifact.track,
                                completed_action_step_ids=completed_action_step_ids,
                                completed_action_task_ids=completed_action_task_ids,
                                runtime_interrupt_probe=runtime_interrupt_probe,
                                setpoint_refresh=live_settle_setpoint_refresh,
                                sample_observer=live_settle_sample_observer,
                                target_frame_position_resolver=(live_settle_target_frame_position),
                                timing=timing,
                            )
                            break
                        except RuntimeInterruptDetected as interruption:
                            await _handle_runtime_interruption(
                                base=base,
                                client=client,
                                frozen_setpoint=setpoint,
                                interruption=interruption,
                                control_dir=args.runtime_control_dir,
                                phase="CHECKPOINT",
                                schedule_index=index,
                                abort_file=args.abort_file,
                                rate_hz=args.setpoint_rate_hz,
                                hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                                decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                                replan_hold_seconds=args.runtime_replan_hold_seconds,
                                active_track_sha256=active_track_sha256,
                                params=params,
                                semantic_path=args.semantic,
                                vehicle_metadata_path=args.vehicle_metadata,
                                coordinate_contract=replacement.coordinate_contract,
                                telemetry_args=args,
                            )
                # _tracking_gate_tick delegates every attempted command to
                # _apply_local_safety, which already paces the complete
                # control period (including fail-closed refresh holds).
            _assert_all_runtime_actions_completed(
                runtime_action_contract,
                completed_action_step_ids,
            )
            if replacement.amendment_action == "follow_target":
                last_setpoint = await _follow_runtime_target(
                    args=args,
                    base=base,
                    client=client,
                    params=params,
                    runtime_session=runtime_session,
                    replacement=replacement,
                    initial_setpoint=last_setpoint,
                    phase_path=phase_path,
                    timing=timing,
                )
            return active_track_sha256
        except RuntimeTrackReplacement as next_replacement:
            replacement = next_replacement


# 功能：
#   读取有界分段跟踪策略，核对轨迹摘要、段编号和净空约束后建立运行期独立配置。
# 输入：
#   path：分段策略文件；None 表示没有提供分段覆盖。
#   track：必须与策略绑定的冻结轨迹。
# 输出：
#   policies：按轨迹段顺序排列的已验证策略元组。
def _load_tracking_segment_policies(
    path: Path | None,
    *,
    track: Px4Track,
) -> tuple[dict[str, float | int | str], ...]:
    """Load per-segment tracking budgets bound to the immutable reference track."""

    if path is None:
        return ()
    payload = read_runtime_object(path, maximum_bytes=256 * 1024)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "dronedream.tracking-corridor-policy.v1"
    ):
        raise ValueError("tracking corridor policy schema is unsupported")
    if payload.get("track_sha256") != sha256_json(track):
        raise ValueError("tracking corridor policy is not bound to the reference track")
    policies = payload.get("segment_policies")
    expected_count = max(0, len(track.points) - 1)
    if not isinstance(policies, list) or len(policies) != expected_count:
        raise ValueError("tracking corridor policy does not match track segments")
    validated: list[dict[str, float | int | str]] = []
    for expected_index, raw in enumerate(policies):
        if (
            not isinstance(raw, dict)
            or type(raw.get("segment_index")) is not int
            or raw["segment_index"] != expected_index
        ):
            raise ValueError("tracking corridor segment identity is invalid")
        lag_limit_m = raw.get("tracking_lag_limit_m")
        rejoin_tolerance_m = raw.get("tracking_rejoin_tolerance_m")
        control_profile = raw.get("control_profile")
        if (
            not finite_positive_number(lag_limit_m)
            or not finite_positive_number(rejoin_tolerance_m)
            or rejoin_tolerance_m >= lag_limit_m
        ):
            raise ValueError("tracking corridor segment limits are invalid")
        if not isinstance(control_profile, str) or control_profile not in {"cruise", "precision"}:
            raise ValueError("tracking corridor segment control profile is invalid")
        measured_clearance = raw.get("minimum_route_clearance_m")
        if measured_clearance is not None and not finite_positive_number(measured_clearance):
            raise ValueError("tracking corridor measured clearance is invalid")
        validated.append(
            {
                "segment_index": expected_index,
                "tracking_lag_limit_m": float(lag_limit_m),
                "tracking_rejoin_tolerance_m": float(rejoin_tolerance_m),
                "control_profile": str(control_profile),
                **(
                    {"minimum_route_clearance_m": float(measured_clearance)}
                    if measured_clearance is not None
                    else {}
                ),
            }
        )
    return tuple(validated)


# 功能：
#   从每段已绑定的实测净空中选出最紧限制，拒绝缺失测量的原生起飞请求。
# 输入：
#   args：含冻结分段策略的执行配置。
# 输出：
#   clearance_m：所有路线段的最小实测净空，单位米。
def _native_route_preflight_clearance(args: argparse.Namespace) -> float:
    """Native motion needs the measured, track-bound route corridor, not a constant."""
    policies = getattr(args, "_tracking_segment_policies", ())
    values = [p.get("minimum_route_clearance_m") for p in policies]
    if not values or not all(finite_positive_number(v) for v in values):
        raise ValueError("NATIVE_ROUTE_MEASURED_CLEARANCE_REQUIRED_BEFORE_ARM")
    return min(values)


# 功能：
#   根据起飞前原生定位不确定性收紧全局及逐段跟踪门槛，不扩大既定飞行容差。
# 输入：
#   args：将原地更新的执行配置和逐段跟踪策略。
#   receipt：原生定位生成的可用跟踪预算回执。
# 输出：
#   adjustment：调整前后全局容差和涉及段数的记录。
def _apply_native_preflight_tracking_budget(args: argparse.Namespace, receipt: dict) -> dict:
    """Spend the live uncertainty allowance by tightening, never enlarging, tracking limits."""
    budget = receipt.get("tracking_budget")
    if not isinstance(budget, dict) or receipt.get("uncertainty_basis") != (
        "live-native-localization"
    ):
        raise ValueError("NATIVE_ROUTE_MEASURED_BUDGET_REQUIRED_BEFORE_ARM")
    lag = budget.get("tracking_lag_limit_m")
    rejoin = budget.get("tracking_rejoin_tolerance_m")
    if not all(finite_positive_number(v) for v in (lag, rejoin)) or rejoin >= lag:
        raise ValueError("NATIVE_ROUTE_MEASURED_TRACKING_LIMITS_INVALID")
    prior_lag = args.tracking_lag_limit_m
    prior_rejoin = args.tracking_rejoin_tolerance_m
    args.tracking_lag_limit_m = min(prior_lag, lag)
    args.tracking_rejoin_tolerance_m = min(prior_rejoin, rejoin)
    args._tracking_segment_policies = tuple(
        {
            **policy,
            "tracking_lag_limit_m": min(policy["tracking_lag_limit_m"], lag),
            "tracking_rejoin_tolerance_m": min(policy["tracking_rejoin_tolerance_m"], rejoin),
        }
        for policy in args._tracking_segment_policies
    )
    return {
        "prior_lag_limit_m": prior_lag,
        "prior_rejoin_tolerance_m": prior_rejoin,
        "effective_lag_limit_m": args.tracking_lag_limit_m,
        "effective_rejoin_tolerance_m": args.tracking_rejoin_tolerance_m,
        "segment_count": len(args._tracking_segment_policies),
    }


# 功能：
#   1. 冻结并核对轨迹、控制参数及动作契约，建立原生连接并完成起飞前门控。
#   2. 以本地模型指令、安全仲裁和实际遥测执行任务，处理检查点及用户改令。
#   3. 在正常结束或失败时完成有证据的控制收尾，并清理自有后台资源。
# 输入：
#   args：命令行配置、运行目录与已建立的实时通道。
#   base：当前明确指定的基础飞控执行模块。
# 输出：
#   None：不返回业务数据。
async def _run(args: argparse.Namespace, base: ModuleType) -> None:
    if getattr(args, "simulation_teacher_control", False) and (
        args.require_model_control_authority or not args.local_safety_required
    ):
        raise ValueError("simulation teacher requires independent safety and no model authority")
    tracking_limits = (
        (args.tracking_lag_limit_m, "tracking lag limit"),
        (args.tracking_rejoin_tolerance_m, "tracking rejoin tolerance"),
        (args.tracking_sample_rate_hz, "tracking sample rate"),
        (args.tracking_telemetry_timeout_seconds, "tracking telemetry timeout"),
        (
            args.tracking_telemetry_recovery_timeout_seconds,
            "tracking telemetry recovery timeout",
        ),
        (args.tracking_recovery_timeout_seconds, "tracking recovery timeout"),
        (args.local_safety_repair_timeout_seconds, "local safety repair stall timeout"),
        (
            args.local_safety_repair_absolute_timeout_seconds,
            "local safety repair absolute timeout",
        ),
        (
            args.tracking_recovery_assist_gain_s_inverse,
            "tracking recovery assist gain",
        ),
        (
            args.tracking_recovery_assist_max_speed_mps,
            "tracking recovery assist maximum speed",
        ),
        (
            args.semantic_progress_recovery_timeout_seconds,
            "semantic progress recovery timeout",
        ),
        (
            args.semantic_progress_abort_timeout_seconds,
            "semantic progress abort timeout",
        ),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value, _ in tracking_limits):
        invalid = next(
            label for value, label in tracking_limits if not math.isfinite(value) or value <= 0.0
        )
        raise ValueError(f"{invalid} must be finite and positive")
    if args.tracking_telemetry_recovery_timeout_seconds <= args.tracking_telemetry_timeout_seconds:
        raise ValueError("tracking telemetry recovery timeout must exceed the sample timeout")
    if args.tracking_rejoin_tolerance_m >= args.tracking_lag_limit_m:
        raise ValueError("tracking rejoin tolerance must be smaller than the lag limit")
    if (
        args.local_safety_repair_absolute_timeout_seconds
        <= args.local_safety_repair_timeout_seconds
    ):
        raise ValueError("local safety repair absolute timeout must exceed its stall timeout")
    if (
        not math.isfinite(args.tracking_recovery_assist_deadband_m)
        or args.tracking_recovery_assist_deadband_m < 0.0
    ):
        raise ValueError("tracking recovery assist deadband must be finite and non-negative")
    if (
        not math.isfinite(args.tracking_recovery_assist_velocity_damping)
        or args.tracking_recovery_assist_velocity_damping < 0.0
    ):
        raise ValueError(
            "tracking recovery assist velocity damping must be finite and non-negative"
        )
    if args.tracking_sample_rate_hz > args.setpoint_rate_hz:
        raise ValueError("tracking sample rate cannot exceed the setpoint rate")
    if (
        args.semantic_progress_abort_timeout_seconds
        <= args.semantic_progress_recovery_timeout_seconds
    ):
        raise ValueError("semantic progress abort timeout must exceed the recovery timeout")
    # 调度点和契约摘要必须来自同一次读取，不能在两个解析器之间再次打开可被替换的文件。
    track_payload = read_runtime_object(args.track, maximum_bytes=MAX_RUNTIME_REPLACEMENT_BYTES)
    reference = base.parse_reference_track_plan(track_payload)
    frozen_track = Px4Track.model_validate(track_payload)
    coordinate_contract = frozen_track.coordinate_contract
    active_track_sha256 = sha256_json(frozen_track)
    args._tracking_segment_policies = _load_tracking_segment_policies(
        args.tracking_corridor_policy,
        track=frozen_track,
    )
    params = base.load_controller_params(args.params)
    plan = base.build_setpoint_schedule_plan(
        reference.points,
        params,
        args.setpoint_rate_hz,
        hover_duration_seconds=reference.hover_duration_seconds,
        stop_at_waypoints=reference.stop_at_waypoints,
        waypoint_hold_seconds=reference.waypoint_hold_seconds,
    )
    checkpoint_contract = (
        RuntimeCheckpointContract.model_validate(read_runtime_object(args.checkpoint_contract))
        if args.checkpoint_contract is not None
        else RuntimeCheckpointContract.model_construct(
            contract_id="local-safety-only",
            checkpoints=[],
        )
    )
    runtime_action_contract = (
        RuntimeActionExecutionContract.model_validate(
            read_runtime_object(args.runtime_action_contract)
        )
        if args.runtime_action_contract is not None
        else None
    )
    if (
        runtime_action_contract is not None
        and runtime_action_contract.contract_id != checkpoint_contract.contract_id
    ):
        raise RuntimeError("runtime action and checkpoint contract identities differ")
    checkpoint_ids = {item.checkpoint_id for item in checkpoint_contract.checkpoints}
    if runtime_action_contract is not None and any(
        step.checkpoint_id not in checkpoint_ids
        for step in runtime_action_contract.steps
        if step.trigger == "checkpoint"
    ):
        raise RuntimeError("runtime action references an unknown checkpoint")
    action_checkpoint_ids = (
        {
            str(step.checkpoint_id)
            for step in runtime_action_contract.steps
            if step.trigger == "checkpoint" and step.checkpoint_id is not None
        }
        if runtime_action_contract is not None
        else set()
    )
    action_checkpoint_point_indices = {
        item.track_point_index
        for item in checkpoint_contract.checkpoints
        if item.checkpoint_id in action_checkpoint_ids
    }
    active_navigation_goal, active_navigation_goal_id = _navigation_goal_for_schedule(
        track=frozen_track,
        waypoint_arrival_indices=plan.waypoint_arrival_indices,
        schedule_index=0,
    )
    args._runtime_track_progress = RuntimeTrackProgress(
        track_sha256=active_track_sha256,
        next_track_point_index=1,
    )
    active_navigation_goal_point_index = int(active_navigation_goal_id.rsplit("-", 1)[-1])
    active_action_checkpoint_goal = (
        active_navigation_goal_point_index in action_checkpoint_point_indices
    )
    base._log(
        args.log,
        f"vehicle={args.vehicle} world={args.world} points={len(reference.points)} "
        f"setpoints={len(plan.schedule)} checkpoints={len(checkpoint_contract.checkpoints)}",
    )
    phase_path = args.run_dir / "runtime-phase.json"
    _atomic_json(phase_path, {"phase": "PREFLIGHT", "checkpoint_id": None})
    runtime_session = _runtime_session(args.runtime_control_dir)

    client = base.MavsdkOffboardClient()
    timing: dict[str, Any] = {
        "time_base": "executor_relative_seconds",
        "active_track_sha256": active_track_sha256,
        "setpoint_count": len(plan.schedule),
        "rate_hz": args.setpoint_rate_hz,
        "status": "running",
        "takeoff_gate": {"status": "not_started"},
        "preflight_connection": {},
        "checkpoints": [],
        "waypoint_settles": [],
        "runtime_actions": [],
        "runtime_interruptions": [],
        "cleanup": {"stop_offboard": "not_needed", "land": "not_needed", "close": "pending"},
    }
    started = time.monotonic()
    command_attempts = FlightCommandAttempts()
    offboard_stopped = False
    landed = False
    pending_interruption: RuntimeInterruptDetected | None = None
    completed_action_step_ids: set[str] = set()
    completed_action_task_ids: set[str] = set()

    # 功能：
    #   领取并缓存一个用户中断，避免多处轮询重复消费同一改令。
    # 输入：
    #   无显式参数；读取当前会话、控制目录及 pending_interruption。
    # 输出：
    #   interruption：等待处理的用户中断，暂无中断时为 None。
    def runtime_interrupt_probe() -> RuntimeInterruptDetected | None:
        nonlocal pending_interruption
        if pending_interruption is None:
            pending_interruption = _claim_runtime_message(args.runtime_control_dir, runtime_session)
        return pending_interruption

    # 功能：
    #   执行终止文件检查并缓存改令，供外层在当前飞控阶段的安全边界处理。
    # 输入：
    #   无显式参数；使用 args 和当前改令探针。
    # 输出：
    #   None：不返回业务数据。
    def abort_check() -> None:
        base._raise_if_external_abort_requested(args.abort_file)
        runtime_interrupt_probe()

    # 功能：
    #   在稳定等待中立即抛出终止或改令信号，不继续等待已作废的旧目标。
    # 输入：
    #   无显式参数；使用当前终止文件和缓存改令。
    # 输出：
    #   None：不返回业务数据。
    def settle_abort_check() -> None:
        """Abort a telemetry settle immediately for operator intervention."""

        base._raise_if_external_abort_requested(args.abort_file)
        interruption = runtime_interrupt_probe()
        if interruption is not None:
            raise interruption

    # 功能：
    #   为稳定等待中的每次控制重新执行本地模型与安全仲裁。
    # 输入：
    #   planned_setpoint：当前计划位置目标。
    # 输出：
    #   applied_setpoint：经过本地门控的实际控制目标。
    async def live_settle_setpoint_refresh(planned_setpoint: Any) -> Any:
        """Keep depth safety authoritative while a waypoint is settling."""

        return await _apply_local_safety(
            args=args,
            base=base,
            client=client,
            planned_setpoint=planned_setpoint,
            coordinate_contract=coordinate_contract,
            phase_path=phase_path,
            navigation_goal_position_m=active_navigation_goal,
            navigation_goal_id=active_navigation_goal_id,
            control_profile="precision",
            action_checkpoint_goal=active_action_checkpoint_goal,
            tracking_recovery_active=True,
        )

    # 功能：
    #   将稳定等待中取得的真实位置与动力学状态写入原生身份链。
    # 输入：
    #   observed：本次飞控位置和速度样本。
    # 输出：
    #   None：不返回业务数据。
    def live_settle_sample_observer(observed: Any) -> None:
        """Keep Gazebo/PX4 identity proof fresh during every settle sample."""

        _publish_px4_identity_telemetry(
            args=args,
            coordinate_contract=coordinate_contract,
            observed=observed,
            dynamics_telemetry=_latest_px4_dynamics_telemetry(client),
        )

    # 功能：
    #   把实际位置转换到路线参考系，供稳定误差计算使用。
    # 输入：
    #   observed：飞控实际位置样本。
    # 输出：
    #   position_ned：路线参考系中的北、东、下位置三元组，单位米。
    def live_settle_target_frame_position(
        observed: Any,
    ) -> tuple[float, float, float]:
        return _observed_route_frame_position_ned(args=args, observed=observed)

    native_publisher = None
    try:
        args._control_application_writer = BoundedRuntimeEvidenceWriter(
            args.run_dir / "runtime-state" / "control-application-writer.json",
            summary_publisher=_atomic_json,
            serializer=lambda record: json.dumps(record, allow_nan=False, sort_keys=True),
            flush_on_record_paths=(args.run_dir / "runtime-state" / "control-applications.jsonl",),
        )
        health = await base.connect_preflight_with_recovery(
            client,
            connection=args.connection,
            readiness_timeout_seconds=args.takeoff_timeout_seconds,
            abort_check=abort_check,
            log_path=args.log,
            evidence=timing["preflight_connection"],
        )
        if not health.armable or not health.home_position_ok or not health.local_position_ok:
            raise RuntimeError("PX4 readiness gate rejected checkpointed flight")
        timing["offboard_loss_failsafe"] = await base._await_with_abort_polling(
            base.configure_offboard_loss_failsafe(client),
            abort_check=abort_check,
        )
        base._log(args.log, "PX4 offboard-loss HOLD failsafe readback verified")
        sensor_deployment = args.run_dir / "native-sensors" / "sensor-deployment.json"
        if sensor_deployment.is_file():
            timing["native_magnetic_preflight"] = await base._await_with_abort_polling(
                verify_magnetic_parameters(client, sensor_deployment),
                abort_check=abort_check,
            )
        measured_origin = await base._await_with_abort_polling(
            client.sample_position_velocity_ned(min(2.0, args.takeoff_timeout_seconds)),
            abort_check=abort_check,
        )
        measured_heading_deg = await base._await_with_abort_polling(
            client.sample_heading_deg(min(2.0, args.takeoff_timeout_seconds)),
            abort_check=abort_check,
        )
        planned_yaw_values = [setpoint.yaw_deg for setpoint in plan.schedule]
        planned_first_setpoint = plan.schedule[0]
        source_setpoint_count = len(plan.schedule)
        measured_body_heading_ned_deg: float | None = None
        if args.heading_policy == "measured-hold":
            plan = replace(
                plan,
                schedule=base.hold_setpoint_schedule_heading(
                    plan.schedule,
                    measured_heading_deg,
                ),
            )
        else:
            if not args.gazebo_vehicle_model_name:
                raise RuntimeError("route-tangent heading requires a Gazebo vehicle model identity")
            gazebo_pose = await base._await_with_abort_polling(
                client.sample_gazebo_model_pose(
                    world_name=args.world,
                    model_name=args.gazebo_vehicle_model_name,
                    timeout_seconds=min(5.0, args.takeoff_timeout_seconds),
                ),
                abort_check=abort_check,
            )
            measured_body_heading_ned_deg = base.gazebo_body_heading_ned_deg(gazebo_pose)
            plan = base.align_setpoint_schedule_to_route_tangent(
                plan,
                measured_px4_heading_deg=measured_heading_deg,
                measured_body_heading_ned_deg=measured_body_heading_ned_deg,
                rate_hz=args.setpoint_rate_hz,
                maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
            )
        timing["setpoint_count"] = len(plan.schedule)
        checkpoint_indices = _schedule_checkpoint_indices(
            base,
            plan.schedule,
            reference.points,
            checkpoint_contract,
            plan.track_start_index,
            waypoint_arrival_indices=plan.waypoint_arrival_indices,
        )
        timing["takeoff_gate"]["heading_control"] = {
            "policy": (
                "measured_prearm_heading_hold"
                if args.heading_policy == "measured-hold"
                else "measured_origin_route_tangent_rate_limited"
            ),
            "measured_heading_deg": measured_heading_deg,
            "measured_gazebo_body_heading_ned_deg": measured_body_heading_ned_deg,
            "px4_minus_gazebo_heading_offset_deg": (
                None
                if measured_body_heading_ned_deg is None
                else base._shortest_yaw_delta_deg(
                    measured_body_heading_ned_deg,
                    measured_heading_deg,
                )
            ),
            "planned_yaw_min_deg": min(planned_yaw_values),
            "planned_yaw_max_deg": max(planned_yaw_values),
            "planned_first_yaw_deg": planned_first_setpoint.yaw_deg,
            "commanded_yaw_deg": measured_heading_deg,
            "maximum_yaw_rate_deg_s": (
                None if args.heading_policy == "measured-hold" else args.maximum_yaw_rate_deg_s
            ),
            "source_setpoint_count": source_setpoint_count,
            "commanded_setpoint_count": len(plan.schedule),
            "runtime_replacements_inherit_heading_policy": True,
            "deliberate_yaw_requires_separate_qualified_action": (
                args.heading_policy == "measured-hold"
            ),
        }
        base._log(
            args.log,
            "measured pre-arm heading "
            f"{measured_heading_deg:.3f} deg; heading policy={args.heading_policy}",
        )
        client = SpawnRelativeOffboardClient(
            client,
            measured_origin,
            heading_hold_deg=(
                measured_heading_deg if args.heading_policy == "measured-hold" else None
            ),
            heading_policy=args.heading_policy,
            maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
            world_name=args.world,
            gazebo_vehicle_model_name=args.gazebo_vehicle_model_name,
            heading_evidence_path=(args.run_dir / "runtime-state" / "px4-heading-tracking.json"),
        )
        from dronedream_agent_core.native_telemetry_publisher import NativeTelemetryPublisher

        # 功能：
        #   发布独立遥测流的实际状态，明确区别于控制等待中的按需采样。
        # 输入：
        #   observed：独立流的位置和速度样本。
        #   dynamics：与样本关联的动力学观测。
        # 输出：
        #   None：不返回业务数据。
        def publish_independent_native_state(observed: Any, dynamics: dict) -> None:
            _publish_px4_identity_telemetry(
                args=args,
                coordinate_contract=coordinate_contract,
                observed=observed,
                dynamics_telemetry=dynamics,
                independent_stream=True,
            )

        native_publisher = NativeTelemetryPublisher(client, publish_independent_native_state)
        args._independent_native_publisher_active = True
        native_publisher.start()
        origin = type(measured_origin)(
            north_m=0.0,
            east_m=0.0,
            down_m=0.0,
            north_m_s=measured_origin.north_m_s,
            east_m_s=measured_origin.east_m_s,
            down_m_s=measured_origin.down_m_s,
        )
        relative_first_setpoint = plan.schedule[0]
        timing["takeoff_gate"]["schedule_origin_rebase"] = {
            "contract": "spawn_relative_client_plus_measured_px4_local_ned_origin",
            "origin_ned": {
                "north_m": measured_origin.north_m,
                "east_m": measured_origin.east_m,
                "down_m": measured_origin.down_m,
            },
            "relative_first_setpoint_ned": {
                "north_m": relative_first_setpoint.north_m,
                "east_m": relative_first_setpoint.east_m,
                "down_m": relative_first_setpoint.down_m,
                "yaw_deg": relative_first_setpoint.yaw_deg,
            },
            "rebased_first_setpoint_ned": {
                "north_m": measured_origin.north_m + relative_first_setpoint.north_m,
                "east_m": measured_origin.east_m + relative_first_setpoint.east_m,
                "down_m": measured_origin.down_m + relative_first_setpoint.down_m,
                "yaw_deg": relative_first_setpoint.yaw_deg,
            },
        }
        initial_hold = base.Setpoint(
            north_m=0.0,
            east_m=0.0,
            down_m=0.0,
            yaw_deg=plan.schedule[0].yaw_deg,
        )
        # 预置设定值和落盘先完成，随后才获取短租期感知回执；不把磁盘尾延迟带入解锁窗口。
        await base._await_with_abort_polling(
            client.set_position_ned(initial_hold), abort_check=abort_check
        )
        _atomic_json(phase_path, {"phase": "TAKEOFF", "checkpoint_id": None})
        if args.local_safety_required and (
            args.require_model_control_authority or args.simulation_teacher_control
        ):
            from dronedream_agent_core.native_preflight import (
                PREFLIGHT_STABLE_WINDOW_MS,
                wait_for_native_perception,
            )

            timing["native_perception_preflight"] = await base._await_with_abort_polling(
                wait_for_native_perception(
                    args.run_dir / "depth-perception-health.json",
                    minimum_route_clearance_m=_native_route_preflight_clearance(args),
                    receiver=getattr(args, "_perception_health_receiver", None),
                    evidence=timing.setdefault("native_perception_preflight_diagnostics", {}),
                    stable_window_ms=PREFLIGHT_STABLE_WINDOW_MS,
                    require_source_timestamps=True,
                ),
                abort_check=abort_check,
            )
            timing["native_perception_preflight"]["applied_tracking_limits"] = (
                _apply_native_preflight_tracking_budget(args, timing["native_perception_preflight"])
            )
            base._log(args.log, "independent native perception verified before arm")
        if "native_perception_preflight" in timing:
            from dronedream_agent_core.native_preflight import assert_native_preflight_current

            assert_native_preflight_current(
                timing["native_perception_preflight"], now_unix_ms=int(time.time() * 1000)
            )
        await command_attempts.arm(
            client, lambda command: base._await_with_abort_polling(command, abort_check=abort_check)
        )
        base._log(args.log, "armed")
        timing["takeoff_start_t"] = time.monotonic() - started
        if "native_perception_preflight" in timing:
            # 解锁回执也可能有延迟：运动前重新取得独立感知窗口，而不是延长旧回执。
            timing["native_perception_before_offboard"] = await base._await_with_abort_polling(
                wait_for_native_perception(
                    args.run_dir / "depth-perception-health.json",
                    minimum_route_clearance_m=_native_route_preflight_clearance(args),
                    receiver=getattr(args, "_perception_health_receiver", None),
                    evidence=timing.setdefault("native_perception_offboard_diagnostics", {}),
                    stable_window_ms=PREFLIGHT_STABLE_WINDOW_MS,
                    require_source_timestamps=True,
                ),
                abort_check=abort_check,
            )
            timing["native_perception_before_offboard"]["applied_tracking_limits"] = (
                _apply_native_preflight_tracking_budget(
                    args, timing["native_perception_before_offboard"])
            )
        # 等待感知/解锁期间 SDK 可能停止预发送；在 start 前重发同一实测地面保持目标。
        await base._await_with_abort_polling(
            asyncio.wait_for(client.set_position_ned(initial_hold), timeout=5.),
            abort_check=abort_check,
        )
        if "native_perception_before_offboard" in timing:
            assert_native_preflight_current(
                timing["native_perception_before_offboard"], now_unix_ms=int(time.time() * 1000))
        await command_attempts.start_offboard(
            client, lambda command: base._await_with_abort_polling(command, abort_check=abort_check)
        )
        timing["offboard_start_t"] = time.monotonic() - started
        base._log(args.log, "offboard started")
        await base._wait_for_takeoff_stability(
            client,
            plan.schedule[0],
            takeoff_origin=origin,
            climb_rate_m_s=args.takeoff_climb_rate_m_s,
            timeout_seconds=args.takeoff_timeout_seconds,
            sample_rate_hz=args.setpoint_rate_hz,
            stable_window_seconds=args.takeoff_stable_window_seconds,
            horizontal_tolerance_m=0.12,
            vertical_tolerance_m=0.08,
            horizontal_speed_tolerance_m_s=0.10,
            vertical_speed_tolerance_m_s=0.08,
            evidence=timing["takeoff_gate"],
            abort_check=abort_check,
        )
        timing["takeoff_stable_t"] = time.monotonic() - started
        base._log(args.log, "takeoff telemetry gate achieved stable hover")
        if pending_interruption is not None:
            _atomic_json(
                phase_path,
                {
                    "phase": "HOLDING",
                    "interrupted_phase": "TAKEOFF",
                    "message_id": pending_interruption.message.message_id,
                },
            )
            interruption_started = time.monotonic()
            await _handle_runtime_interruption(
                base=base,
                client=client,
                frozen_setpoint=plan.schedule[0],
                interruption=pending_interruption,
                control_dir=args.runtime_control_dir,
                phase="TAKEOFF",
                schedule_index=None,
                abort_file=args.abort_file,
                rate_hz=args.setpoint_rate_hz,
                hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                replan_hold_seconds=args.runtime_replan_hold_seconds,
                active_track_sha256=active_track_sha256,
                params=params,
                semantic_path=args.semantic,
                vehicle_metadata_path=args.vehicle_metadata,
                coordinate_contract=coordinate_contract,
                telemetry_args=args,
            )
            timing["runtime_interruptions"].append(
                {
                    "message_id": pending_interruption.message.message_id,
                    "interrupted_phase": "TAKEOFF",
                    "outcome": "resume_original",
                    "duration_seconds": time.monotonic() - interruption_started,
                }
            )
            pending_interruption = None
        if runtime_action_contract is not None:
            _atomic_json(
                phase_path,
                {"phase": "ACTION", "checkpoint_id": None, "trigger": "post-takeoff"},
            )
            while True:
                try:
                    await _execute_triggered_runtime_actions(
                        base=base,
                        client=client,
                        setpoint=plan.schedule[0],
                        contract=runtime_action_contract,
                        trigger="post-takeoff",
                        checkpoint_id=None,
                        completed_step_ids=completed_action_step_ids,
                        completed_task_ids=completed_action_task_ids,
                        run_dir=args.run_dir,
                        abort_file=args.abort_file,
                        rate_hz=args.setpoint_rate_hz,
                        runtime_interrupt_probe=runtime_interrupt_probe,
                        timing=timing,
                        setpoint_refresh=live_settle_setpoint_refresh,
                        sample_observer=live_settle_sample_observer,
                        target_frame_position_resolver=(live_settle_target_frame_position),
                    )
                    break
                except RuntimeInterruptDetected as interruption:
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "HOLDING",
                            "interrupted_phase": "ACTION",
                            "message_id": interruption.message.message_id,
                        },
                    )
                    interruption_started = time.monotonic()
                    await _handle_runtime_interruption(
                        base=base,
                        client=client,
                        frozen_setpoint=plan.schedule[0],
                        interruption=interruption,
                        control_dir=args.runtime_control_dir,
                        phase="ACTION",
                        schedule_index=None,
                        abort_file=args.abort_file,
                        rate_hz=args.setpoint_rate_hz,
                        hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                        decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                        replan_hold_seconds=args.runtime_replan_hold_seconds,
                        active_track_sha256=active_track_sha256,
                        params=params,
                        semantic_path=args.semantic,
                        vehicle_metadata_path=args.vehicle_metadata,
                        coordinate_contract=coordinate_contract,
                        telemetry_args=args,
                    )
                    timing["runtime_interruptions"].append(
                        {
                            "message_id": interruption.message.message_id,
                            "interrupted_phase": "ACTION",
                            "outcome": "resume_prepared_actions",
                            "duration_seconds": time.monotonic() - interruption_started,
                        }
                    )
                    pending_interruption = None
        _atomic_json(phase_path, {"phase": "TRACK", "checkpoint_id": None})

        loop = asyncio.get_running_loop()
        motion_deadline = loop.time() + args.track_timeout_seconds
        last_setpoint = plan.schedule[0]
        waypoint_arrival_indices = set(plan.waypoint_arrival_indices)
        tracking_sample_interval = _tracking_sample_interval(args)
        for index, setpoint in enumerate(plan.schedule):
            args._runtime_track_progress = RuntimeTrackProgress(
                track_sha256=active_track_sha256,
                next_track_point_index=next_track_point_index(
                    tuple(plan.waypoint_arrival_indices), index, len(frozen_track.points)
                ),
            )
            checkpoint = checkpoint_indices.get(index)
            recovery_started: float | None = None
            local_recovery_control_active = False
            recovery_window: dict[str, Any] | None = None
            maximum_tracking_error_m = 0.0
            while True:
                abort_check()
                # A tracking recovery pauses schedule time and has its own
                # strict bounded window.  Do not let safe recovery consume the
                # nominal motion budget and abort an otherwise healthy flight.
                if recovery_started is None and loop.time() >= motion_deadline:
                    raise TimeoutError("track timeout")
                if pending_interruption is not None:
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "HOLDING",
                            "interrupted_phase": "TRACK",
                            "message_id": pending_interruption.message.message_id,
                        },
                    )
                    interruption_started = loop.time()
                    await _handle_runtime_interruption(
                        base=base,
                        client=client,
                        frozen_setpoint=last_setpoint,
                        interruption=pending_interruption,
                        control_dir=args.runtime_control_dir,
                        phase="TRACK",
                        schedule_index=index,
                        abort_file=args.abort_file,
                        rate_hz=args.setpoint_rate_hz,
                        hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                        decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                        replan_hold_seconds=args.runtime_replan_hold_seconds,
                        active_track_sha256=active_track_sha256,
                        params=params,
                        semantic_path=args.semantic,
                        vehicle_metadata_path=args.vehicle_metadata,
                        coordinate_contract=coordinate_contract,
                        telemetry_args=args,
                    )
                    duration = loop.time() - interruption_started
                    motion_deadline += duration
                    timing["runtime_interruptions"].append(
                        {
                            "message_id": pending_interruption.message.message_id,
                            "interrupted_phase": "TRACK",
                            "schedule_index": index,
                            "outcome": "resume_original",
                            "duration_seconds": duration,
                        }
                    )
                    pending_interruption = None
                    _atomic_json(phase_path, {"phase": "TRACK", "checkpoint_id": None})
                sample_now = (
                    recovery_started is not None
                    or index % tracking_sample_interval == 0
                    or index in waypoint_arrival_indices
                    or checkpoint is not None
                )
                navigation_goal, navigation_goal_id = _navigation_goal_for_schedule(
                    track=frozen_track,
                    waypoint_arrival_indices=plan.waypoint_arrival_indices,
                    schedule_index=index,
                )
                planned_velocity_ned_mps = _schedule_velocity_feedforward(
                    current_setpoint=setpoint,
                    next_setpoint=plan.schedule[min(index + 1, len(plan.schedule) - 1)],
                    rate_hz=args.setpoint_rate_hz,
                    speed_limit_mps=float(params.vel_limit),
                    # Preserve the qualified route tangent for cross-track
                    # decomposition; it is not used as recovery actuation.
                    recovery_active=False,
                )
                planned_goal_world = _setpoint_world_enu(setpoint, coordinate_contract)
                planned_navigation_goal_distance_m = math.dist(
                    (
                        planned_goal_world.x,
                        planned_goal_world.y,
                        planned_goal_world.z,
                    ),
                    (navigation_goal.x, navigation_goal.y, navigation_goal.z),
                )
                semantic_approach_damping_active = _semantic_approach_damping_required(
                    planned_goal_distance_m=planned_navigation_goal_distance_m,
                    # Model-required local control is bounded by vel_limit,
                    # not by the usually slower reference feed-forward.
                    planned_speed_mps=max(
                        math.sqrt(sum(value * value for value in planned_velocity_ned_mps)),
                        float(params.vel_limit),
                    ),
                    maximum_acceleration_mps2=float(params.accel_limit),
                    waypoint_position_tolerance_m=reference.waypoint_position_tolerance_m,
                )
                navigation_goal_point_index = int(navigation_goal_id.rsplit("-", 1)[-1])
                action_checkpoint_goal = (
                    navigation_goal_point_index in action_checkpoint_point_indices
                )
                active_navigation_goal = navigation_goal
                active_navigation_goal_id = navigation_goal_id
                active_action_checkpoint_goal = action_checkpoint_goal
                control_profile = _navigation_control_profile(
                    planned_goal_distance_m=planned_navigation_goal_distance_m,
                    action_checkpoint_goal=action_checkpoint_goal,
                    semantic_approach_damping_active=semantic_approach_damping_active,
                    tight_clearance_segment=(
                        _tracking_control_profile_for_context(
                            args,
                            {"navigation_goal_id": navigation_goal_id},
                        )
                        == "precision"
                    ),
                )
                may_advance, tracking_error_m = await _tracking_gate_tick(
                    args=args,
                    base=base,
                    client=client,
                    planned_setpoint=setpoint,
                    coordinate_contract=coordinate_contract,
                    phase_path=phase_path,
                    schedule_index=index,
                    sample_now=sample_now,
                    recovery_active=recovery_started is not None,
                    local_recovery_control_active=local_recovery_control_active,
                    planned_velocity_ned_mps=planned_velocity_ned_mps,
                    context={
                        "navigation_goal_position_m": navigation_goal.model_dump(mode="json"),
                        "navigation_goal_id": navigation_goal_id,
                        "semantic_approach_damping_active": (semantic_approach_damping_active),
                        "control_profile": control_profile,
                        "action_checkpoint_goal": action_checkpoint_goal,
                    },
                )
                if tracking_error_m is not None:
                    maximum_tracking_error_m = max(maximum_tracking_error_m, tracking_error_m)
                if may_advance:
                    if recovery_started is not None:
                        recovery_duration = loop.time() - recovery_started
                        motion_deadline += recovery_duration
                        timing.setdefault("tracking_recoveries", []).append(
                            {
                                "schedule_index": index,
                                "duration_seconds": recovery_duration,
                                "maximum_position_error_m": maximum_tracking_error_m,
                            }
                        )
                        _atomic_json(phase_path, {"phase": "TRACK", "checkpoint_id": None})
                    break
                if tracking_error_m is None:
                    # See the replacement-track branch above: absence of a
                    # model lease pauses motion without granting route fallback.
                    continue
                local_recovery_control_active = bool(
                    getattr(
                        args,
                        "_last_tracking_local_recovery_control_required",
                        True,
                    )
                )
                if recovery_started is None:
                    recovery_started = loop.time()
                recovery_now = loop.time()
                recovery_window, recovery_expired = _advance_tracking_recovery_window(
                    now=recovery_now,
                    timeout_seconds=args.tracking_recovery_timeout_seconds,
                    state=recovery_window,
                    tracking_error_m=tracking_error_m,
                    model_goal_distance_m=getattr(
                        args,
                        "_last_tracking_model_goal_distance_m",
                        None,
                    ),
                )
                _publish_tracking_recovery_window(
                    args=args,
                    schedule_index=index,
                    state=recovery_window,
                    now=recovery_now,
                )
                if recovery_expired:
                    raise UserDirectedLanding(
                        "closed-loop tracking recovery exceeded its bounded window"
                    )
            last_setpoint = setpoint
            if index == plan.track_start_index:
                timing["track_start_t"] = time.monotonic() - started
            if index in waypoint_arrival_indices and checkpoint is None:
                settle_started = loop.time()
                _atomic_json(
                    phase_path,
                    {"phase": "WAYPOINT_SETTLE", "schedule_index": index},
                )
                while True:
                    try:
                        _, position_error, speed = await _wait_checkpoint_stable(
                            base=base,
                            client=client,
                            setpoint=setpoint,
                            rate_hz=args.setpoint_rate_hz,
                            timeout_seconds=reference.waypoint_settle_timeout_seconds,
                            # A loaded aircraft in a tight local space can require more
                            # than five ordinary settle windows to remove the final
                            # position error.  Progress must still be measurable every
                            # ordinary window and every position/speed gate remains
                            # unchanged; this only extends the immutable hard ceiling.
                            absolute_timeout_factor=8.0,
                            stable_window_seconds=reference.waypoint_stable_window_seconds,
                            position_tolerance_m=(
                                _payload_aware_waypoint_position_tolerance_m(
                                    configured_tolerance_m=(
                                        reference.waypoint_position_tolerance_m
                                    ),
                                    runtime_action_contract=runtime_action_contract,
                                    completed_action_step_ids=completed_action_step_ids,
                                )
                            ),
                            speed_tolerance_mps=reference.waypoint_speed_tolerance_mps,
                            abort_check=settle_abort_check,
                            setpoint_refresh=live_settle_setpoint_refresh,
                            sample_observer=live_settle_sample_observer,
                            target_frame_position_resolver=(live_settle_target_frame_position),
                        )
                        break
                    except RuntimeInterruptDetected as interruption:
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "HOLDING",
                                "interrupted_phase": "WAYPOINT_SETTLE",
                                "message_id": interruption.message.message_id,
                            },
                        )
                        await _handle_runtime_interruption(
                            base=base,
                            client=client,
                            frozen_setpoint=setpoint,
                            interruption=interruption,
                            control_dir=args.runtime_control_dir,
                            phase="WAYPOINT_SETTLE",
                            schedule_index=index,
                            abort_file=args.abort_file,
                            rate_hz=args.setpoint_rate_hz,
                            hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                            decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                            replan_hold_seconds=args.runtime_replan_hold_seconds,
                            active_track_sha256=active_track_sha256,
                            params=params,
                            semantic_path=args.semantic,
                            vehicle_metadata_path=args.vehicle_metadata,
                            coordinate_contract=coordinate_contract,
                            telemetry_args=args,
                        )
                        timing["runtime_interruptions"].append(
                            {
                                "message_id": interruption.message.message_id,
                                "interrupted_phase": "WAYPOINT_SETTLE",
                                "schedule_index": index,
                                "outcome": "resume_original",
                            }
                        )
                        pending_interruption = None
                        _atomic_json(
                            phase_path,
                            {"phase": "WAYPOINT_SETTLE", "schedule_index": index},
                        )
                settle_duration = loop.time() - settle_started
                motion_deadline += settle_duration
                timing["waypoint_settles"].append(
                    {
                        "schedule_index": index,
                        "position_error_m": position_error,
                        "speed_mps": speed,
                        "duration_seconds": settle_duration,
                    }
                )
                _atomic_json(phase_path, {"phase": "TRACK", "checkpoint_id": None})
            if checkpoint is not None:
                _atomic_json(
                    phase_path,
                    {"phase": "CHECKPOINT", "checkpoint_id": checkpoint.checkpoint_id},
                )
                while True:
                    try:
                        observed, position_error, speed = await _wait_checkpoint_stable(
                            base=base,
                            client=client,
                            setpoint=setpoint,
                            rate_hz=args.setpoint_rate_hz,
                            timeout_seconds=reference.waypoint_settle_timeout_seconds,
                            absolute_timeout_factor=8.0,
                            stable_window_seconds=reference.waypoint_stable_window_seconds,
                            position_tolerance_m=(
                                _payload_aware_waypoint_position_tolerance_m(
                                    configured_tolerance_m=(
                                        reference.waypoint_position_tolerance_m
                                    ),
                                    runtime_action_contract=runtime_action_contract,
                                    completed_action_step_ids=completed_action_step_ids,
                                )
                            ),
                            speed_tolerance_mps=reference.waypoint_speed_tolerance_mps,
                            abort_check=settle_abort_check,
                            setpoint_refresh=live_settle_setpoint_refresh,
                            sample_observer=live_settle_sample_observer,
                            target_frame_position_resolver=(live_settle_target_frame_position),
                        )
                        break
                    except RuntimeInterruptDetected as interruption:
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "HOLDING",
                                "interrupted_phase": "CHECKPOINT",
                                "message_id": interruption.message.message_id,
                            },
                        )
                        await _handle_runtime_interruption(
                            base=base,
                            client=client,
                            frozen_setpoint=setpoint,
                            interruption=interruption,
                            control_dir=args.runtime_control_dir,
                            phase="CHECKPOINT",
                            schedule_index=index,
                            abort_file=args.abort_file,
                            rate_hz=args.setpoint_rate_hz,
                            hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                            decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                            replan_hold_seconds=args.runtime_replan_hold_seconds,
                            active_track_sha256=active_track_sha256,
                            params=params,
                            semantic_path=args.semantic,
                            vehicle_metadata_path=args.vehicle_metadata,
                            coordinate_contract=coordinate_contract,
                            telemetry_args=args,
                        )
                        timing["runtime_interruptions"].append(
                            {
                                "message_id": interruption.message.message_id,
                                "interrupted_phase": "CHECKPOINT",
                                "schedule_index": index,
                                "outcome": "resume_original",
                            }
                        )
                        pending_interruption = None
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "CHECKPOINT",
                                "checkpoint_id": checkpoint.checkpoint_id,
                            },
                        )
                battery = await _sample_checkpoint_battery(
                    base=base,
                    client=client,
                    setpoint=setpoint,
                    rate_hz=args.setpoint_rate_hz,
                    runtime_interrupt_probe=runtime_interrupt_probe,
                    sample_observer=live_settle_sample_observer,
                    setpoint_refresh=live_settle_setpoint_refresh,
                    safety_abort_check=abort_check,
                )
                raw_battery = float(battery["remaining_percent"])
                battery_percent = raw_battery * 100.0 if raw_battery <= 1.0 else raw_battery
                gates = {
                    "position_error_within_0_75_m": position_error <= 0.75,
                    "speed_within_0_50_mps": speed <= 0.5,
                    "battery_above_10_percent": battery_percent > 10.0,
                    "telemetry_finite": all(
                        math.isfinite(value)
                        for value in (
                            observed.north_m,
                            observed.east_m,
                            observed.down_m,
                            speed,
                            battery_percent,
                        )
                    ),
                }
                request = RuntimeCheckpointRequest(
                    contract_id=checkpoint_contract.contract_id,
                    checkpoint=checkpoint,
                    observed_position_ned_m=Vector3(
                        x=observed.north_m, y=observed.east_m, z=observed.down_m
                    ),
                    observed_velocity_ned_mps=Vector3(
                        x=observed.north_m_s,
                        y=observed.east_m_s,
                        z=observed.down_m_s,
                    ),
                    commanded_position_ned_m=Vector3(
                        x=setpoint.north_m, y=setpoint.east_m, z=setpoint.down_m
                    ),
                    position_error_m=position_error,
                    speed_mps=speed,
                    battery_percent=battery_percent,
                    deterministic_gates=gates,
                )
                request_path = (
                    args.run_dir / "checkpoints" / f"{checkpoint.checkpoint_id}.request.json"
                )
                decision_path = (
                    args.run_dir / "checkpoints" / f"{checkpoint.checkpoint_id}.decision.json"
                )
                _atomic_json(request_path, request)
                base._log(args.log, f"checkpoint requested {checkpoint.checkpoint_id}")
                if not all(gates.values()):
                    raise RuntimeError(
                        f"deterministic checkpoint gate failed: {checkpoint.checkpoint_id}"
                    )
                checkpoint_started = loop.time()
                while True:
                    try:
                        decision = await _wait_checkpoint_decision(
                            base=base,
                            client=client,
                            setpoint=setpoint,
                            request=request,
                            decision_path=decision_path,
                            abort_file=args.abort_file,
                            rate_hz=args.setpoint_rate_hz,
                            timeout_seconds=args.checkpoint_timeout_seconds,
                            runtime_interrupt_probe=runtime_interrupt_probe,
                            sample_observer=live_settle_sample_observer,
                            setpoint_refresh=live_settle_setpoint_refresh,
                        )
                        break
                    except RuntimeInterruptDetected as interruption:
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "HOLDING",
                                "interrupted_phase": "CHECKPOINT",
                                "message_id": interruption.message.message_id,
                            },
                        )
                        await _handle_runtime_interruption(
                            base=base,
                            client=client,
                            frozen_setpoint=setpoint,
                            interruption=interruption,
                            control_dir=args.runtime_control_dir,
                            phase="CHECKPOINT",
                            schedule_index=index,
                            abort_file=args.abort_file,
                            rate_hz=args.setpoint_rate_hz,
                            hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                            decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                            replan_hold_seconds=args.runtime_replan_hold_seconds,
                            active_track_sha256=active_track_sha256,
                            params=params,
                            semantic_path=args.semantic,
                            vehicle_metadata_path=args.vehicle_metadata,
                            coordinate_contract=coordinate_contract,
                            telemetry_args=args,
                        )
                        timing["runtime_interruptions"].append(
                            {
                                "message_id": interruption.message.message_id,
                                "interrupted_phase": "CHECKPOINT",
                                "schedule_index": index,
                                "outcome": "resume_original",
                            }
                        )
                        pending_interruption = None
                        _atomic_json(
                            phase_path,
                            {
                                "phase": "CHECKPOINT",
                                "checkpoint_id": checkpoint.checkpoint_id,
                            },
                        )
                motion_deadline += loop.time() - checkpoint_started
                timing["checkpoints"].append(
                    {
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "request_sha256": sha256_json(request),
                        "decision_sha256": sha256_json(decision),
                        "continue_authorized": decision.continue_authorized,
                        "assessment_action": decision.assessment.action,
                    }
                )
                if not decision.continue_authorized or decision.assessment.action != "accept":
                    raise RuntimeError(
                        f"checkpoint continuation rejected: {checkpoint.checkpoint_id}"
                    )
                base._log(args.log, f"checkpoint accepted {checkpoint.checkpoint_id}")
                if runtime_action_contract is not None:
                    action_started = loop.time()
                    _atomic_json(
                        phase_path,
                        {
                            "phase": "ACTION",
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "trigger": "checkpoint",
                        },
                    )
                    while True:
                        try:
                            await _execute_triggered_runtime_actions(
                                base=base,
                                client=client,
                                setpoint=setpoint,
                                contract=runtime_action_contract,
                                trigger="checkpoint",
                                checkpoint_id=checkpoint.checkpoint_id,
                                completed_step_ids=completed_action_step_ids,
                                completed_task_ids=completed_action_task_ids,
                                run_dir=args.run_dir,
                                abort_file=args.abort_file,
                                rate_hz=args.setpoint_rate_hz,
                                runtime_interrupt_probe=runtime_interrupt_probe,
                                timing=timing,
                                setpoint_refresh=live_settle_setpoint_refresh,
                                sample_observer=live_settle_sample_observer,
                                target_frame_position_resolver=(live_settle_target_frame_position),
                            )
                            break
                        except RuntimeInterruptDetected as interruption:
                            _atomic_json(
                                phase_path,
                                {
                                    "phase": "HOLDING",
                                    "interrupted_phase": "ACTION",
                                    "message_id": interruption.message.message_id,
                                    "checkpoint_id": checkpoint.checkpoint_id,
                                },
                            )
                            interruption_started = loop.time()
                            await _handle_runtime_interruption(
                                base=base,
                                client=client,
                                frozen_setpoint=setpoint,
                                interruption=interruption,
                                control_dir=args.runtime_control_dir,
                                phase="ACTION",
                                schedule_index=index,
                                abort_file=args.abort_file,
                                rate_hz=args.setpoint_rate_hz,
                                hold_timeout_seconds=args.runtime_hold_timeout_seconds,
                                decision_timeout_seconds=args.runtime_decision_timeout_seconds,
                                replan_hold_seconds=args.runtime_replan_hold_seconds,
                                active_track_sha256=active_track_sha256,
                                params=params,
                                semantic_path=args.semantic,
                                vehicle_metadata_path=args.vehicle_metadata,
                                coordinate_contract=coordinate_contract,
                                telemetry_args=args,
                            )
                            timing["runtime_interruptions"].append(
                                {
                                    "message_id": interruption.message.message_id,
                                    "interrupted_phase": "ACTION",
                                    "checkpoint_id": checkpoint.checkpoint_id,
                                    "outcome": "resume_prepared_actions",
                                    "duration_seconds": loop.time() - interruption_started,
                                }
                            )
                            pending_interruption = None
                    motion_deadline += loop.time() - action_started
                _atomic_json(phase_path, {"phase": "TRACK", "checkpoint_id": None})
            # _tracking_gate_tick already completed this control period.
        _assert_all_runtime_actions_completed(runtime_action_contract, completed_action_step_ids)
        timing["track_end_t"] = time.monotonic() - started
        timing["perception_control_completion"] = capture_control_completion(
            args.run_dir / "depth-perception-health.json",
            track_sha256=active_track_sha256,
            completed_at_unix_ms=int(time.time() * 1000),
            receiver=getattr(args, "_perception_health_receiver", None),
        )
        await client.stop_offboard()
        offboard_stopped = True
        timing["cleanup"]["stop_offboard"] = "completed"
        base._log(args.log, "offboard stopped")
        _atomic_json(phase_path, {"phase": "LANDING", "checkpoint_id": None})
        timing["land_start_t"] = time.monotonic() - started
        await client.land()
        observation = await client.wait_until_landed(args.landing_timeout_seconds)
        landed = True
        timing["cleanup"]["land"] = "confirmed_on_ground"
        timing["cleanup"]["landing_observation"] = observation
        timing["land_confirmed_t"] = time.monotonic() - started
        timing["status"] = "complete"
        _atomic_json(phase_path, {"phase": "COMPLETE", "checkpoint_id": None})
        base._log(args.log, "landing confirmed ON_GROUND by PX4 telemetry")
    except RuntimeTrackReplacement as replacement:
        try:
            active_track_sha256 = await _fly_runtime_replacement(
                args=args,
                base=base,
                client=client,
                params=params,
                runtime_session=runtime_session,
                phase_path=phase_path,
                timing=timing,
                initial=replacement,
                completed_action_step_ids=completed_action_step_ids,
                completed_action_task_ids=completed_action_task_ids,
            )
            timing["active_track_sha256"] = active_track_sha256
            timing["track_end_t"] = time.monotonic() - started
            timing["perception_control_completion"] = capture_control_completion(
                args.run_dir / "depth-perception-health.json",
                track_sha256=active_track_sha256,
                completed_at_unix_ms=int(time.time() * 1000),
                receiver=getattr(args, "_perception_health_receiver", None),
            )
            await client.stop_offboard()
            offboard_stopped = True
            timing["cleanup"]["stop_offboard"] = "completed"
            base._log(args.log, "replacement offboard track completed")
            _atomic_json(phase_path, {"phase": "LANDING", "checkpoint_id": None})
            timing["land_start_t"] = time.monotonic() - started
            await client.land()
            observation = await client.wait_until_landed(args.landing_timeout_seconds)
            landed = True
            timing["cleanup"]["land"] = "confirmed_on_ground"
            timing["cleanup"]["landing_observation"] = observation
            timing["land_confirmed_t"] = time.monotonic() - started
            timing["status"] = "complete"
            _atomic_json(phase_path, {"phase": "COMPLETE", "checkpoint_id": None})
            base._log(args.log, "replacement track landing confirmed ON_GROUND")
        except BaseException as exc:
            timing["status"] = "failed"
            timing["failure"] = f"{type(exc).__name__}: {exc}"
            raise
    except BaseException as exc:
        timing["status"] = "failed"
        timing["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Safety actions precede diagnostic flushing: a log failure must not
        # bypass landing, and a lost arm/start ACK does not prove non-dispatch.
        primary_failure = sys.exception() is not None
        timing["command_attempts"] = command_attempts.evidence()
        await cleanup_flight_commands(
            client,
            attempts=command_attempts,
            offboard_stopped=offboard_stopped,
            landed=landed,
            landing_timeout_seconds=args.landing_timeout_seconds,
            on_landing=lambda: _atomic_json(
                phase_path, {"phase": "LANDING", "checkpoint_id": None}
            ),
            evidence=timing["cleanup"],
        )
        timing["model_control_authority"] = {
            "control_application_counts": dict(
                getattr(args, "_model_control_application_counts", {})
            ),
            "required": bool(getattr(args, "_last_model_control_required", False)),
            "authorized_control_applied_count": int(
                getattr(args, "_model_authorized_control_applied_count", 0)
            ),
            "unauthorized_hold_applied_count": int(
                getattr(args, "_model_authority_hold_applied_count", 0)
            ),
            "authorized_schedule_advance_count": int(
                getattr(args, "_model_authorized_schedule_advance_count", 0)
            ),
            "unauthorized_schedule_hold_count": int(
                getattr(args, "_model_authority_schedule_hold_count", 0)
            ),
            "progress_schedule_hold_count": int(
                getattr(args, "_model_progress_schedule_hold_count", 0)
            ),
            "route_fallback_schedule_advance_count": int(
                getattr(args, "_route_fallback_schedule_advance_count", 0)
            ),
        }
        await finalize_executor_resources(
            client=client,
            native_publisher=native_publisher,
            application_writer=getattr(args, "_control_application_writer", None),
            control_pacer=getattr(args, "_local_control_pacer", None),
            safety_receiver=getattr(args, "_local_safety_receiver", None),
            timing=timing,
            write_terminal_phase=lambda phase: _atomic_json(
                phase_path, {"phase": phase, "checkpoint_id": None}
            ),
            write_timing=lambda value: base._write_offboard_timing(
                args.run_dir / "offboard_timing.json", value
            ),
            native_closed=lambda: setattr(args, "_independent_native_publisher_active", False),
            primary_failure=primary_failure,
            snapshots=getattr(args, "_runtime_snapshots", None),
            phase_broadcaster=getattr(args, "_phase_broadcaster", None),
        )


# 功能：
#   1. 为一次任务建立实时阶段广播与异步证据快照，并将所有权绑定当前异步上下文。
#   2. 无论飞行是否进入最终清理，退出时都排空资源，不把后台写入遗留给下一次飞行。
# 输入：
#   args：当前任务参数及有限阶段端点。
#   base：实际飞控基础模块。
# 输出：
#   None：不返回业务数据。
async def _run_with_live_snapshots(args: argparse.Namespace, base: ModuleType) -> None:
    snapshots = ExecutorSnapshots(args.run_dir)
    token = ACTIVE_SNAPSHOTS.set(snapshots)
    args._runtime_snapshots = snapshots
    broadcaster = None
    try:
        if args.runtime_phase_channel:
            from dronedream_agent_core.runtime_phase_channel import RuntimePhaseBroadcaster

            broadcaster = RuntimePhaseBroadcaster(args.runtime_phase_channel, snapshots)
            args._phase_broadcaster = broadcaster
            broadcaster.start()
        await _run(args, base)
    finally:
        primary_failure = sys.exc_info()[0] is not None
        errors = []
        try:
            if broadcaster is not None:
                await broadcaster.close()
        except BaseException as error:
            errors.append(error)
        try:
            summary = await asyncio.to_thread(snapshots.close)
            if not summary["complete"]:
                errors.append(RuntimeError("EXECUTOR_SNAPSHOTS_NOT_DRAINED"))
        finally:
            ACTIVE_SNAPSHOTS.reset(token)
        if errors and not primary_failure:
            raise RuntimeError("EXECUTOR_LIVE_STATE_CLEANUP_FAILED") from errors[0]


# 功能：
#   解析执行配置并加载指定飞控模块，建立实时通道，运行任务并返回可检查的退出状态。
# 输入：
#   无显式参数；读取命令行参数。
# 输出：
#   exit_code：任务及清理成功时为零，普通执行失败时为二。
def main() -> int:
    args = _parse_args()
    base = _load_base(args.base_executor)
    try:
        # 先加载真实 SDK 的稳定对象图，再进入专属执行进程的保留作用域。
        # 不在飞行中 collect/freeze，也不关闭新对象的循环回收。
        importlib.import_module("mavsdk")
        configure_sensor_thread_handoff()
        # 通道构造后立即登记，后续构造、运行或另一通道关闭失败均不能跳过已有资源。
        with contextlib.ExitStack() as resources:
            resources.enter_context(retained_interpreter_baseline())
            pause_monitor = InterpreterPauseMonitor()
            resources.callback(lambda: _atomic_json(
                args.run_dir / "runtime-state" / "executor-interpreter-pauses.json",
                pause_monitor.close()))
            if args.native_state_channel is not None:
                from dronedream_agent_core.native_state_channel import NativeStatePublisher

                args._native_state_channel = NativeStatePublisher(args.native_state_channel)
                resources.callback(args._native_state_channel.close)
            if args.local_safety_channel is not None:
                from dronedream_agent_core.local_safety_channel import LocalSafetyReceiver

                if args.local_safety_command is None or args.local_safety_observation is None:
                    raise ValueError("live safety channel requires durable evidence paths")
                args._local_safety_receiver = LocalSafetyReceiver(args.local_safety_channel)
                resources.callback(args._local_safety_receiver.close)
            if getattr(args, "perception_health_channel", None) is not None:
                from dronedream_agent_core.perception_health_channel import PerceptionHealthReceiver

                args._perception_health_receiver = PerceptionHealthReceiver(
                    args.perception_health_channel)
                resources.callback(args._perception_health_receiver.close)
            asyncio.run(_run_with_live_snapshots(args, base))
            base._log(args.log, "checkpoint executor completed successfully")
            return 0
    except Exception as exc:
        base._log(args.log, f"checkpoint executor failure: {exc}")
        print(f"checkpoint PX4 executor failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
