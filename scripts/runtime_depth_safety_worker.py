#!/usr/bin/env python3
"""Bridge live metric depth and native estimator state into bounded controls."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from collections import Counter, deque
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from dronedream_agent_core.assets import _load_object as _read_semantic_snapshot
from dronedream_agent_core.collision import primitive_bounds as _primitive_bounds
from dronedream_agent_core.contracts import (
    GraphRoute,
    OnboardPerceptionFrame,
    PerceptionFusionHealth,
    RawMetricRangeScan,
    RouteClearanceReport,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.control_authority import remaining_control_validity_ms
from dronedream_agent_core.control_timing import LOCAL_DISPATCH_RESERVE_MS, resolve_control_timing
from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker
from dronedream_agent_core.depth_sensor_binding import DepthSensorBinding
from dronedream_agent_core.forward_rgb_binding import ForwardRgbBinding
from dronedream_agent_core.gazebo_adapter import (
    _gazebo_image_model_payload,
    _gazebo_image_png,
    _gazebo_semantic_label_png,
    _resolve_controlled_vehicle_pose,
)
from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy
from dronedream_agent_core.learning_observation_recorder import LearningObservationRecorder
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    load_local_policy_package,
    select_local_policy,
    select_local_policy_for_simulation,
)
from dronedream_agent_core.local_policy_port import LocalPolicyPort
from dronedream_agent_core.local_safety_channel import LocalSafetyPublisher
from dronedream_agent_core.local_vision_training import load_local_vision_label_map
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.localization_observations import GeometryObservationCapture
from dronedream_agent_core.model_harness.model_port import (
    FailoverStructuredModelPort,
    StructuredModelPort,
)
from dronedream_agent_core.model_image_cache import ModelImageCache, require_model_image_runtime
from dronedream_agent_core.model_image_worker import LatestModelImageWorker, PreparedCameraSample
from dronedream_agent_core.native_pose import NativeMapPose, native_map_pose
from dronedream_agent_core.native_state_stream import NativeStateSampler
from dronedream_agent_core.navigation_context import build_navigation_context
from dronedream_agent_core.navigation_snapshot import NavigationSnapshotRequest
from dronedream_agent_core.occupancy_collision import fully_contained_box_mask
from dronedream_agent_core.perception_health_channel import PerceptionHealthPublisher
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
)
from dronedream_agent_core.pipeline_timing import (
    PhaseTimings,
    PhaseTimingSummary,
    sensor_processing_timing,
)
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.plugin_values import plugin_json_value
from dronedream_agent_core.realtime_feature_encoders import (
    body_control_intent_for_pilot_control,
    body_to_world_enu,
    encode_dynamic_targets,
    encode_metric_geometry,
    fuse_realtime_features,
    pilot_control_limits_for_profile,
    refresh_flight_state_features,
)
from dronedream_agent_core.runtime_control_io import read_runtime_object
from dronedream_agent_core.runtime_evidence import BoundedRuntimeEvidenceWriter
from dronedream_agent_core.runtime_file_reader import PinnedRuntimeObjectReader
from dronedream_agent_core.runtime_local_safety import (
    evaluate_runtime_local_safety,
    prepare_safety_publication,
    runtime_safety_query_radius_m,
)
from dronedream_agent_core.runtime_multimodal_dataset import (
    RuntimeMultimodalDatasetRecorder,
)
from dronedream_agent_core.runtime_phase import runtime_phase_context as _runtime_phase_context
from dronedream_agent_core.runtime_phase_channel import RuntimePhaseReceiver
from dronedream_agent_core.runtime_phase_observer import RuntimePhaseObserver
from dronedream_agent_core.runtime_scheduling import (
    InterpreterPauseMonitor,
    ReadyControlScheduler,
    SensorArrivalScheduler,
    configure_sensor_thread_handoff,
    local_input_cadence_enabled,
    model_input_work_allowed,
    retained_interpreter_baseline,
    sensor_input_maximum_rate_hz,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorRegistry,
    forward_camera_motion_alignment,
    forward_rgb_can_inform_control,
    oakd_lite_depth_sensor_contract,
)
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge
from dronedream_agent_core.sensor_frame_clock import (
    SensorFrameTime,
    SensorImageIngress,
    require_model_frame_time,
)
from dronedream_agent_core.simulation_camera_profile import CameraProfileReadback
from dronedream_agent_core.simulation_teacher import teacher_heading_rate, teacher_input_deadline
from dronedream_agent_core.static_geometry_index import StaticGeometryIndex
from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

Point = tuple[float, float, float]
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_MAX_NATIVE_PACKET_BYTES = 256 * 1024


# 功能：
#   拒绝布尔、隐式字符串转换及无法以有限双精度表示的物理量。
# 输入：
#   value：待检查的数值。
# 输出：
#   valid：数值类型和有限范围均有效时为真。
def _finite_number(value: object) -> bool:
    valid = type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
    return valid


# 功能：
#   将过期控制周期改为不健康观测，保留来源时间并通过契约重新校验。
# 输入：
#   observation：原始局部安全观测。
#   published_at_unix_ms：实际发布时刻，单位毫秒。
# 输出：
#   expired：禁止借过期感知继续驱动的观测。
def _expired_model_control_observation(
    observation: RuntimeLocalSafetyObservation,
    *,
    published_at_unix_ms: int,
) -> RuntimeLocalSafetyObservation:
    if (type(published_at_unix_ms) is not int
            or not observation.observed_at_unix_ms <= published_at_unix_ms <= 10**15):
        raise ValueError("EXPIRED_CONTROL_PUBLICATION_CLOCK_INVALID")
    expired = RuntimeLocalSafetyObservation.model_validate(
        {
            **observation.model_dump(mode="python"),
            "stream_healthy": False,
            "stream_age_seconds": max(
                observation.stream_age_seconds,
                (published_at_unix_ms - observation.observed_at_unix_ms) / 1_000.0,
            ),
        }
    )
    return expired


# 功能：
#   判断授权剩余时间是否不足以覆盖执行端的一个控制周期及传输预算。
# 输入：
#   published_at_unix_ms：准备发布的时刻。
#   authority_deadline_unix_ms：原始感知和模型授权共同约束的截止时刻。
# 输出：
#   unavailable：不能再安全分派此指令时为真。
def _model_control_dispatch_unavailable(*, published_at_unix_ms: int,
                                        authority_deadline_unix_ms: int) -> bool:
    if any(type(value) is not int or not 0 <= value <= 10**15
           for value in (published_at_unix_ms, authority_deadline_unix_ms)):
        raise ValueError("MODEL_CONTROL_DISPATCH_CLOCK_INVALID")
    # A positive lifetime alone is not enough: the executor reserves one
    # control period plus its transport budget, for replanning as well as axes.
    unavailable = authority_deadline_unix_ms - published_at_unix_ms < LOCAL_DISPATCH_RESERVE_MS
    return unavailable


# 功能：
#   为拒绝动作产生的悬停记录绑定对应调用，不把归因标识变成运动授权。
# 输入：
#   directive：当前生效的模型指令或空值。
#   completed_cycle：刚结束的模型周期及其拒绝原因。
# 输出：
#   call_id：悬停应归属的调用标识，没有关联调用时为空。
def _safety_hold_call_id(directive, completed_cycle):
    if (completed_cycle is not None and completed_cycle.hold_reason is not None
            and completed_cycle.model_call_id is not None
            and (directive is None or not directive.model_navigation_authorized)):
        return completed_cycle.model_call_id
    return directive.model_call_id if directive is not None else None


# 功能：
#   从有界历史中选取尚未记录的新鲜 RGB 帧及时间最接近的语义帧，仅用于训练采样。
# 输入：
#   rgb_items、semantic_items：图像与单调接收时间组成的历史序列。
#   now_monotonic_seconds：当前单调时间。
#   newest_rgb_after_monotonic_seconds：已经提交的 RGB 时间水位。
#   maximum_rgb_age_seconds、maximum_offset_seconds：帧龄与跨相机配对偏差上限。
# 输出：
#   pair：RGB 与可选语义邻帧；没有合格新帧时为空。
def _nearest_synchronized_sensor_pair(
    rgb_items: tuple[tuple[Any, float], ...],
    semantic_items: tuple[tuple[Any, float], ...],
    *,
    now_monotonic_seconds: float,
    newest_rgb_after_monotonic_seconds: float,
    maximum_rgb_age_seconds: float = 0.5,
    maximum_offset_seconds: float = 0.1,
) -> tuple[tuple[Any, float], tuple[Any, float] | None] | None:
    eligible_rgb = [
        item
        for item in rgb_items
        if item[1] > newest_rgb_after_monotonic_seconds
        and 0.0 <= now_monotonic_seconds - item[1] <= maximum_rgb_age_seconds
    ]
    for rgb_item in reversed(eligible_rgb):
        if not semantic_items:
            return rgb_item, None
        semantic_item = min(
            semantic_items,
            key=lambda candidate: abs(candidate[1] - rgb_item[1]),
        )
        if abs(semantic_item[1] - rgb_item[1]) <= maximum_offset_seconds:
            return rgb_item, semantic_item
    return None


# 功能：
#   1. 依据距离、制动距离和动作检查点收紧为精细控制，不能放宽计划限制。
#   2. 同一语义目标一旦锁存精细模式就保持，避免定位噪声使控制工况来回切换。
# 输入：
#   requested_profile：计划要求的巡航或精细工况。
#   action_checkpoint_goal：目标是否需要悬停、装卸等动作。
#   observed_goal_distance_m、observed_speed_mps：实测目标距离与速度。
#   maximum_acceleration_mps2：用于估算制动距离的加速度限制。
#   precision_latched：本目标是否已进入精细模式。
# 输出：
#   profile：当前应使用的控制工况。
def _effective_control_profile(
    *,
    requested_profile: str,
    action_checkpoint_goal: bool,
    observed_goal_distance_m: float,
    observed_speed_mps: float = 0.0,
    maximum_acceleration_mps2: float = 1.0,
    precision_latched: bool = False,
) -> str:
    if not isinstance(requested_profile, str) or requested_profile not in {"cruise", "precision"}:
        raise ValueError(f"unsupported navigation control profile: {requested_profile}")
    if type(action_checkpoint_goal) is not bool or type(precision_latched) is not bool:
        raise ValueError("navigation checkpoint and precision latch must be boolean")
    if not _finite_number(observed_goal_distance_m) or observed_goal_distance_m < 0.0:
        raise ValueError("observed navigation goal distance must be finite and non-negative")
    if not _finite_number(observed_speed_mps) or observed_speed_mps < 0.0:
        raise ValueError("observed speed must be finite and non-negative")
    if not _finite_number(maximum_acceleration_mps2) or maximum_acceleration_mps2 <= 0.0:
        raise ValueError("maximum acceleration must be finite and positive")
    braking_entry_distance_m = (
        (float(observed_speed_mps) / maximum_acceleration_mps2)
        * (float(observed_speed_mps) * 0.5) + 0.75
    )
    precision_distance_m = max(
        4.0 if action_checkpoint_goal else 0.8,
        braking_entry_distance_m,
    )
    if (
        precision_latched
        or requested_profile == "precision"
        or observed_goal_distance_m <= precision_distance_m
    ):
        return "precision"
    return "cruise"


# 功能：
#   将局部规划前视距离限制在工况及体素尺度内，不代替模型的连续速度控制输出。
# 输入：
#   profile：当前控制工况。
#   maximum_step_m：允许的最大局部前视距离。
#   world_resolution_m：局部地图的体素分辨率。
# 输出：
#   step_m：当前工况允许的前视距离。
def _controller_step_for_profile(
    *,
    profile: str,
    maximum_step_m: float,
    world_resolution_m: float,
) -> float:
    if not isinstance(profile, str) or profile not in {"cruise", "precision"}:
        raise ValueError(f"unsupported navigation control profile: {profile}")
    if (not _finite_number(maximum_step_m) or maximum_step_m <= 0.0
            or not _finite_number(world_resolution_m) or world_resolution_m <= 0.0):
        raise ValueError("navigation control profile dimensions must be positive")
    if profile == "precision":
        # Three quarters of one measured voxel keeps every micro-target local
        # and naturally caps desired speed before the strict settle gate.  The
        # PX4 still owns the low-level attitude and motor loops; this value
        # only bounds the local planner's lookahead, not motor actuation.
        return min(maximum_step_m, world_resolution_m * 0.75)
    return maximum_step_m


# 功能：
#   1. 模型控制模式只接受明确授权的模型参考目标，缺失或拒绝指令时保持当前位置。
#   2. 非模型控制的显式采集模式沿用教师路线，旁观模型不能偷偷接管教师。
# 输入：
#   route_target、current_position：教师参考位置和当前实测位置。
#   model_directive：带授权标记的模型指令。
#   require_model_control_authority：是否要求模型拥有运动授权。
# 输出：
#   target：该授权模式下可使用的导航参考位置。
def _navigation_target_for_control_authority(
    *,
    route_target: Vector3,
    current_position: Vector3,
    model_directive: object | None,
    require_model_control_authority: bool,
) -> Vector3:
    if type(require_model_control_authority) is not bool:
        raise ValueError("model control authority requirement must be boolean")
    if not require_model_control_authority:
        return route_target
    if getattr(model_directive, "model_navigation_authorized", False) is True:
        return Vector3.model_validate(model_directive.target_m)  # type: ignore[attr-defined]
    return current_position


# 功能：
#   复用飞行控制的工况限制，保证避障规划速度不突破路线及狭窄空间约束。
# 输入：
#   profile：当前工况。
#   route_speed_limit_mps：路线允许的最高速度。
# 输出：
#   speed_mps：规划器应使用的速度上限。
def _planner_speed_for_profile(*, profile: str, route_speed_limit_mps: float) -> float:
    speed_mps = pilot_control_limits_for_profile(
        profile=profile,
        route_speed_limit_mps=route_speed_limit_mps,
    )[0]
    return speed_mps


# 功能：
#   在直接控制模式禁用坐标候选搜索，避免旧的点位选择混入连续速度控制。
# 输入：
#   omitted：调用方是否显式禁用坐标候选。
#   control_output_mode：本轮输出契约。
# 输出：
#   enabled：仅显式兼容候选选择模式允许坐标候选。
def _coordinate_candidates_enabled(*, omitted: bool, control_output_mode: str) -> bool:
    if type(omitted) is not bool or control_output_mode not in (
        "normalized-body-velocity", "legacy-candidate-selection",
    ):
        raise ValueError("NAVIGATION_OUTPUT_MODE_INVALID")
    enabled = not (omitted or control_output_mode == "normalized-body-velocity")
    return enabled


# 功能：
#   1. 为直接控制绑定已批准的语义目标，不使用教师不断移动的短前视点作为模型目标。
#   2. 仅兼容候选选择模式保留坐标相等复验；直接控制通过语义目标身份撤销旧授权。
# 输入：
#   route_target：保留调用兼容的教师参考点，不赋予直接控制模型。
#   navigation_goal：已批准的阶段目标。
#   omitted_coordinate_candidates、control_output_mode：候选禁用配置与输出契约。
# 输出：
#   goal_contract：模型目标、可选坐标复验目标、实际输出模式组成的三元组。
def _model_navigation_goal_contract(
    *,
    route_target: Vector3,
    navigation_goal: Vector3,
    omitted_coordinate_candidates: bool,
    control_output_mode: str,
) -> tuple[Vector3, Vector3 | None, str]:
    direct_control = not _coordinate_candidates_enabled(
        omitted=omitted_coordinate_candidates, control_output_mode=control_output_mode)
    if direct_control:
        return navigation_goal, None, "normalized-body-velocity"
    return navigation_goal, navigation_goal, "legacy-candidate-selection"


# 功能：
#   为模型请求选择唯一可追溯的触发原因，执行端进入着陆或终止状态后不再请求运动。
# 输入：
#   model_cycle_started：是否已有初始模型请求。
#   now_monotonic、next_model_cycle_monotonic：当前时间与下一轮最早时间。
#   target_changed、dynamic_obstacle_changed、progress_recovery_requested：重规划事件。
#   executor_phase：执行端当前阶段。
# 输出：
#   trigger：本轮调用原因；无需调用时为空。
def _model_cycle_trigger(
    *,
    model_cycle_started: bool,
    now_monotonic: float,
    next_model_cycle_monotonic: float,
    target_changed: bool,
    dynamic_obstacle_changed: bool,
    progress_recovery_requested: bool,
    executor_phase: str = "UNKNOWN",
) -> str | None:
    # The executor owns the explicit landing sequence. Keep sensing and safety
    # publication alive, but do not ask actors to resume motion during teardown.
    if executor_phase in {"LANDING", "LANDED", "COMPLETE", "FAILED"}:
        return None
    if not model_cycle_started:
        return "initial"
    if now_monotonic < next_model_cycle_monotonic:
        return None
    if target_changed:
        return "goal-amended"
    if dynamic_obstacle_changed:
        return "dynamic-obstacle"
    if progress_recovery_requested:
        return "progress-stalled"
    return "periodic"


# 功能：
#   将连续遇到同一目标附近的动态障碍归为同一恢复事件，目标改变或长时间安静后重建。
# 输入：
#   state：已有恢复事件状态。
#   navigation_goal_id、dynamic_signature：目标身份和障碍集合摘要。
#   dynamic_present：本轮是否仍观测到动态障碍。
#   now_monotonic、quiet_seconds：当前时间与事件结束的安静时长。
# 输出：
#   state：续期或新建的事件；安静超时后为空。
def _advance_dynamic_recovery_episode(
    *,
    state: dict[str, object] | None,
    navigation_goal_id: str,
    dynamic_signature: str,
    dynamic_present: bool,
    now_monotonic: float,
    quiet_seconds: float = 30.0,
) -> dict[str, object] | None:
    if (not _finite_number(now_monotonic) or now_monotonic < 0
            or not _finite_number(quiet_seconds) or not 0 < quiet_seconds <= 3600
            or type(dynamic_present) is not bool
            or not isinstance(navigation_goal_id, str) or not navigation_goal_id.strip()
            or not isinstance(dynamic_signature, str) or len(dynamic_signature) != 64):
        raise ValueError("DYNAMIC_RECOVERY_INPUT_INVALID")
    if state is not None:
        last_seen = state.get("last_seen_monotonic")
        if not _finite_number(last_seen) or not 0 <= last_seen <= now_monotonic:
            raise ValueError("DYNAMIC_RECOVERY_CLOCK_INVALID")
        # 换目标即撤销旧事件，即使新目标这一帧恰好没有动态物体也不能沿用。
        if state.get("navigation_goal_id") != navigation_goal_id:
            state = None
    if not dynamic_present:
        if state is not None and now_monotonic - float(state["last_seen_monotonic"]) > (
            quiet_seconds
        ):
            return None
        return state
    if (
        state is None
        or state.get("navigation_goal_id") != navigation_goal_id
        or now_monotonic - float(state["last_seen_monotonic"]) > quiet_seconds
    ):
        state = {
            "navigation_goal_id": navigation_goal_id,
            "episode_id": "recovery-"
            + sha256_json(
                {
                    "navigation_goal_id": navigation_goal_id,
                    "initial_dynamic_signature": dynamic_signature,
                    "started_at_monotonic": now_monotonic,
                }
            )[:24],
            "started_at_monotonic": now_monotonic,
            "last_seen_monotonic": now_monotonic,
        }
    else:
        state["last_seen_monotonic"] = now_monotonic
    return state


# 功能：
#   判断显式开发故障注入是否处于丢帧区间，未配置注入时不改变正常传感器流。
# 输入：
#   first_accepted_frame_monotonic、now_monotonic：首帧接收时间与当前时间。
#   drop_after_seconds、drop_duration_seconds：开始丢帧的延迟与持续时长。
# 输出：
#   suppressed：当前帧是否应被开发注入器丢弃。
def _development_depth_frame_suppressed(
    *,
    first_accepted_frame_monotonic: float | None,
    now_monotonic: float,
    drop_after_seconds: float | None,
    drop_duration_seconds: float | None,
) -> bool:
    if (
        first_accepted_frame_monotonic is None
        or drop_after_seconds is None
        or drop_duration_seconds is None
    ):
        return False
    elapsed = now_monotonic - first_accepted_frame_monotonic
    return drop_after_seconds <= elapsed < drop_after_seconds + drop_duration_seconds


# 功能：
#   1. 注入器已知传输中断时立即置感知不健康，不再借缓存帧的剩余寿命掩盖故障。
#   2. 故障结束后仍须由正常新鲜帧链路恢复健康，不能由注入开关直接恢复授权。
# 输入：
#   health：正常感知融合的健康结果。
#   development_depth_fault_active：显式开发丢帧故障是否生效。
# 输出：
#   health：正常结果或附带丢帧原因的不健康副本。
def _fault_adjusted_perception_health(
    health: PerceptionFusionHealth,
    *,
    development_depth_fault_active: bool,
) -> PerceptionFusionHealth:
    if type(development_depth_fault_active) is not bool:
        raise ValueError("DEVELOPMENT_DEPTH_FAULT_FLAG_INVALID")
    if not development_depth_fault_active:
        return health
    issues = list(health.issue_codes)
    if "DEVELOPMENT_DEPTH_FRAME_SUPPRESSED" not in issues:
        issues.append("DEVELOPMENT_DEPTH_FRAME_SUPPRESSED")
    return health.model_copy(
        update={
            "stream_healthy": False,
            "issue_codes": issues,
        }
    )


# 功能：
#   允许两次深度帧之间结束推理，但最新获准感知必须仍健康；没有新帧不等于帧已失效。
# 输入：
#   stale_tick：本周期是否没有收到新帧，保留用于调用契约。
#   stream_healthy：融合层依据来源时间判断的当前健康状态。
# 输出：
#   stream_healthy：是否允许进入结果复验，不等同于已授予运动权限。
def _fresh_perception_can_finalize_model_cycle(
    *,
    stale_tick: bool,
    stream_healthy: bool,
) -> bool:
    if type(stale_tick) is not bool or type(stream_healthy) is not bool:
        raise ValueError("PERCEPTION_COMPLETION_FLAGS_INVALID")
    return stream_healthy


# 功能：
#   用当前健康度、语义目标和实测位姿复验异步模型结果，过期感知不进入完成流程。
# 输入：
#   coordinator、fusion：异步模型协调器及感知融合器。
#   stale_tick、goal、goal_id：新帧状态与当前目标内容及身份。
#   frame、position、velocity：最近感知帧与当前原生估计状态。
# 输出：
#   cycle：本次完成结果；感知不健康或推理尚未结束时为空。
def _poll_ready_navigation_cycle(coordinator, *, fusion, stale_tick, goal, goal_id,
                                 frame, position, velocity):
    now_ms = int(time.time() * 1000)
    current_health = fusion.health(now_unix_ms=now_ms, now_monotonic_seconds=time.monotonic())
    if not _fresh_perception_can_finalize_model_cycle(
        stale_tick=stale_tick, stream_healthy=current_health.stream_healthy,
    ):
        return None
    return coordinator.poll(
        now_unix_ms=now_ms, current_goal_position_m=goal, current_navigation_goal_id=goal_id,
        current_perception_health=current_health,
        # Poll consumes this view synchronously. Any legacy asynchronous
        # revalidation makes its own deep copy before handing it to a worker.
        current_frame=frame.model_copy(update={"localization_position_m": position,
                                                "localization_velocity_mps": velocity}),
    )




# 功能：
#   对 Windows/DrvFS 短暂读锁有限重试同一暂存文件，持久存储故障仍抛出，不能无限等待。
# 输入：
#   temporary、path：本次暂存路径与发布目标。
#   timeout_seconds、retry_seconds：总等待预算与每次退避时长。
# 输出：
#   None：不返回业务数据。
def _replace_with_bounded_retry(
    temporary: Path,
    path: Path,
    *,
    timeout_seconds: float = 0.75,
    retry_seconds: float = 0.01,
) -> None:
    if (not _finite_number(timeout_seconds) or not 0 <= timeout_seconds <= 3600
            or not _finite_number(retry_seconds) or not 0 <= retry_seconds <= 3600):
        raise ValueError("runtime publication retry budgets must be finite and non-negative")
    check_plain_plugin_path(temporary)
    original = temporary.stat()
    deadline = time.monotonic() + timeout_seconds
    if not math.isfinite(deadline):
        raise ValueError("runtime publication retry deadline overflow")
    transient_errnos = {errno.EACCES, errno.EBUSY, errno.EPERM}
    while True:
        # 短暂读锁的每次重试都重新核对身份，不能发布等待期间被替换的文件。
        check_plain_plugin_path(path)
        check_plain_plugin_path(temporary)
        current = temporary.stat()
        if (not os.path.samestat(original, current) or current.st_size != original.st_size
                or current.st_mtime_ns != original.st_mtime_ns):
            raise ValueError("RUNTIME_SNAPSHOT_TEMPORARY_REPLACED")
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            retryable = isinstance(error, PermissionError) or error.errno in transient_errnos
            remaining = deadline - time.monotonic()
            if not retryable or not math.isfinite(remaining) or remaining <= 0.0:
                raise
            time.sleep(min(max(.001, retry_seconds), remaining))
            if time.monotonic() >= deadline:
                raise


# 功能：
#   先拒绝非有限 JSON，再以完整 UTF-8 字节替换快照，不暴露半份内容。
# 输入：
#   path、payload：目标文件与 JSON 数据。
#   replace_timeout_seconds、replace_retry_seconds：发布读锁的有限重试预算。
# 输出：
#   None：不返回业务数据。
def _atomic_json(
    path: Path,
    payload: object,
    *,
    replace_timeout_seconds: float = 0.75,
    replace_retry_seconds: float = 0.01,
) -> None:
    rendered = (encode_json(payload, limit=_MAX_SNAPSHOT_BYTES) + "\n").encode("utf-8")
    _atomic_bytes(
        path, rendered, replace_timeout_seconds=replace_timeout_seconds,
        replace_retry_seconds=replace_retry_seconds,
    )


# 功能：
#   1. 独占创建有界普通暂存文件，复核身份后替换目标，失败仅清理仍归属于本次的暂存。
#   2. 拒绝静态链接；原子可见性不代表断电持久性，也不代替操作系统目录隔离。
# 输入：
#   path、payload：目标文件与不可变字节。
#   replace_timeout_seconds、replace_retry_seconds：发布读锁的有限重试预算。
# 输出：
#   None：不返回业务数据。
def _atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    replace_timeout_seconds: float = 0.75,
    replace_retry_seconds: float = 0.01,
) -> None:
    if not isinstance(payload, bytes) or len(payload) > _MAX_SNAPSHOT_BYTES:
        raise ValueError("RUNTIME_SNAPSHOT_BYTES_INVALID_OR_TOO_LARGE")
    if (not _finite_number(replace_timeout_seconds) or replace_timeout_seconds < 0.0
            or not _finite_number(replace_retry_seconds) or replace_retry_seconds < 0.0):
        raise ValueError("runtime publication retry budgets must be finite and non-negative")
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    owned = None
    try:
        # 独占创建，防止撞名时覆盖他人的暂存文件；失败仅清理本次仍拥有的文件。
        check_plain_plugin_path(temporary)
        with temporary.open("xb") as stream:
            owned = os.fstat(stream.fileno())
            stream.write(payload)
            stream.flush()
        check_plain_plugin_path(path)
        check_plain_plugin_path(temporary)
        if not os.path.samestat(owned, temporary.stat()):
            raise ValueError("RUNTIME_SNAPSHOT_TEMPORARY_REPLACED")
        _replace_with_bounded_retry(
            temporary, path, timeout_seconds=replace_timeout_seconds,
            retry_seconds=replace_retry_seconds,
        )
    finally:
        if owned is not None:
            with contextlib.suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(owned, temporary.stat()):
                    temporary.unlink()


# 功能：
#   本地端使用已认证图像字节并异步留档，云端仍同步保存其数据地址所需的 PNG 文件。
# 输入：
#   provider：模型提供方类型。
#   writer：有界后台快照写入器。
#   path、png：图像归档路径与 PNG 字节。
# 输出：
#   None：不返回业务数据。
def _persist_navigation_image(*, provider: str, writer, path: Path, png: bytes) -> None:
    if provider in {"local-policy", "simulation-training"}:
        writer.submit_bytes(path, png)
    else:
        # Cloud transports still construct their data URL from a saved file.
        _atomic_bytes(path, png)


# 功能：
#   将一份运行证据编码为单行 JSON，由共享 FIFO 写入器负责顺序与刷新。
# 输入：
#   payload：契约模型或普通 JSON 数据。
# 输出：
#   line：不带行终止符的 JSON 文本。
def _jsonl_line(payload: object) -> str:
    line = encode_json(plugin_json_value(payload, limit=_MAX_SNAPSHOT_BYTES),
                       limit=_MAX_SNAPSHOT_BYTES)
    return line


# 功能：
#   绑定脚本的发布与序列化接口，复用核心唯一 FIFO 实现，避免两套证据队列行为分叉。
# 输入：
#   summary_path：队列汇总路径。
#   maximum_pending_records、flush_interval_seconds：排队容量与批量刷新间隔。
#   flush_on_record_paths：需要逐记录刷新的关键证据路径。
# 输出：
#   writer：有界、保序的后台证据写入器。
def _BoundedRuntimeEvidenceWriter(
    summary_path: Path, *, maximum_pending_records: int = 8192,
    flush_interval_seconds: float = 0.2,
    flush_on_record_paths: tuple[Path, ...] = (),
) -> BoundedRuntimeEvidenceWriter:
    return BoundedRuntimeEvidenceWriter(
        summary_path, summary_publisher=_atomic_json, serializer=_jsonl_line,
        maximum_pending_records=maximum_pending_records,
        flush_interval_seconds=flush_interval_seconds,
        flush_on_record_paths=flush_on_record_paths,
    )


class _LatestRuntimeSnapshotWriter:
    """Coalesce owned immutable diagnostic bytes away from the control loop."""

    # 功能：
    #   创建按路径合并最新快照的有界后台队列，先发布初始状态再启动线程。
    # 输入：
    #   self：新写入器。
    #   summary_path：写入统计和错误状态文件。
    #   maximum_pending_paths：最多允许多少条不同路径同时等待写入，上限为 64。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        summary_path: Path,
        *,
        maximum_pending_paths: int = 16,
    ) -> None:
        if type(maximum_pending_paths) is not int or not 1 <= maximum_pending_paths <= 64:
            raise ValueError("maximum_pending_paths must be in [1, 64]")
        self.summary_path = summary_path
        self.maximum_pending_paths = maximum_pending_paths
        self._condition = threading.Condition()
        self._pending: dict[Path, bytes] = {}
        self._order: deque[Path] = deque()
        self._closed = False
        self._thread_finished = False
        self._issue_code: str | None = None
        self._submitted_count = 0
        self._completed_count = 0
        self._superseded_count = 0
        self._rejected_count = 0
        self._maximum_pending_observed = 0
        self._thread = threading.Thread(
            target=self._run,
            name="dronedream-runtime-snapshot-writer",
            daemon=True,
        )
        self._publish_summary()
        self._thread.start()

    # 功能：
    #   在同步锁内读取第一个故障，不因后续成功覆盖已发生的记录问题。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   issue：故障代码，没有故障时为空。
    @property
    def issue(self) -> str | None:
        with self._condition:
            return self._issue_code

    # 功能：
    #   在调用线程完成有限 JSON 序列化，后台只持有稳定字节，避免借用可变字典。
    # 输入：
    #   self：当前写入器。
    #   path、payload：快照目标和待冻结内容。
    # 输出：
    #   accepted：本次快照是否被接收入队，不代表已经落盘。
    def submit_json(self, path: Path, payload: object) -> bool:
        # Transfer stable JSON values, not a borrowed nested dictionary. Compact
        # serialization owns the bytes once and avoids a deep copy followed by
        # a second indented serialization on a competing Python thread.
        try:
            owned = (encode_json(payload, limit=_MAX_SNAPSHOT_BYTES) + "\n").encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            return self._reject_snapshot(
                f"RUNTIME_SNAPSHOT_SERIALIZATION_FAILED:{type(error).__name__}")
        return self._submit(path, payload=owned)

    # 功能：
    #   接收不可变二进制证据；不隐式转换可能继续被修改的缓冲区。
    # 输入：
    #   self：当前写入器。
    #   path、payload：目标路径和不可变字节。
    # 输出：
    #   accepted：是否成功排队。
    def submit_bytes(self, path: Path, payload: bytes) -> bool:
        if not isinstance(payload, bytes):
            return self._reject_snapshot("RUNTIME_SNAPSHOT_BYTES_REQUIRED")
        return self._submit(path, payload=payload)

    # 功能：
    #   记录拒绝次数及首个错误，唤醒等待线程，确保无效输入不会被汇总成完整记录。
    # 输入：
    #   self：当前写入器。
    #   issue：拒绝原因代码。
    # 输出：
    #   accepted：固定为假。
    def _reject_snapshot(self, issue: str) -> bool:
        with self._condition:
            if self._issue_code is None:
                self._issue_code = issue
            self._rejected_count += 1
            self._condition.notify_all()
        return False

    # 功能：
    #   1. 对同一路径仅替换尚未写出的快照，并记录被合并数量。
    #   2. 新路径超过容量或单份字节超限时拒绝，不阻塞控制线程等待磁盘。
    # 输入：
    #   self：当前写入器。
    #   path、payload：快照路径和已冻结字节。
    # 输出：
    #   accepted：是否接收入队。
    def _submit(self, path: Path, *, payload: bytes) -> bool:
        if len(payload) > _MAX_SNAPSHOT_BYTES:
            return self._reject_snapshot("RUNTIME_SNAPSHOT_BYTES_TOO_LARGE")
        normalized_path = Path(path)
        with self._condition:
            if self._closed or self._issue_code is not None:
                self._rejected_count += 1
                return False
            if normalized_path in self._pending:
                self._submitted_count += 1
                self._pending[normalized_path] = payload
                self._superseded_count += 1
                return True
            if len(self._pending) >= self.maximum_pending_paths:
                self._issue_code = "RUNTIME_SNAPSHOT_QUEUE_CAPACITY_EXCEEDED"
                self._rejected_count += 1
                self._condition.notify_all()
                return False
            self._submitted_count += 1
            self._pending[normalized_path] = payload
            self._order.append(normalized_path)
            self._maximum_pending_observed = max(
                self._maximum_pending_observed,
                len(self._pending),
            )
            self._condition.notify()
        return True

    # 功能：
    #   停止接受新快照并有界等待排空，超时如实记错，最后发布统计。
    # 输入：
    #   self：当前写入器。
    #   timeout_seconds：允许排空的最大秒数。
    # 输出：
    #   summary：包含线程和排空完整性的结束回执。
    def close(self, *, timeout_seconds: float = 4.0) -> dict[str, object]:
        if not _finite_number(timeout_seconds) or not 0 <= timeout_seconds <= threading.TIMEOUT_MAX:
            raise ValueError("RUNTIME_SNAPSHOT_CLOSE_TIMEOUT_INVALID")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout_seconds))
        with self._condition:
            if self._thread.is_alive() and self._issue_code is None:
                self._issue_code = "RUNTIME_SNAPSHOT_DRAIN_TIMEOUT"
        self._publish_summary()
        return self.summary()

    # 功能：
    #   核对提交、完成与合并数量守恒；必须关闭、线程结束且无拒绝和故障才算完整。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   summary：锁内一致的队列状态及完整性统计。
    def summary(self) -> dict[str, object]:
        with self._condition:
            pending_count = len(self._pending)
            thread_alive = self._thread.is_alive()
            complete = bool(
                self._closed
                and self._thread_finished
                and not thread_alive
                and self._issue_code is None
                and self._rejected_count == 0
                and pending_count == 0
                and self._completed_count + self._superseded_count == self._submitted_count
            )
            return {
                "schema_version": "dronedream.runtime-snapshot-writer.v1",
                "writer_mode": "bounded-latest-per-path-background",
                "maximum_pending_paths": self.maximum_pending_paths,
                "maximum_pending_observed": self._maximum_pending_observed,
                "submitted_count": self._submitted_count,
                "completed_count": self._completed_count,
                "superseded_count": self._superseded_count,
                "rejected_count": self._rejected_count,
                "pending_count": pending_count,
                "closed": self._closed,
                "thread_finished": self._thread_finished,
                "thread_alive": thread_alive,
                "issue_code": self._issue_code,
                "complete": complete,
                "updated_at_unix_ms": int(time.time() * 1_000),
            }

    # 功能：
    #   将队列一致性统计以完整文件发布，存储失败交由调用方处理。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   None：不返回业务数据。
    def _publish_summary(self) -> None:
        _atomic_json(self.summary_path, self.summary())

    # 功能：
    #   在锁外执行磁盘替换，锁内维护队列计数；任何退出均标记线程结束。
    # 输入：
    #   self：后台写入器自身。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._order and not self._closed:
                        self._condition.wait()
                    if not self._order and self._closed:
                        break
                    path = self._order.popleft()
                    payload = self._pending.pop(path)
                _atomic_bytes(path, payload)
                with self._condition:
                    self._completed_count += 1
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            with self._condition:
                if self._issue_code is None:
                    self._issue_code = (
                        f"RUNTIME_SNAPSHOT_WRITE_FAILED:{type(error).__name__}:{error}"
                    )
        finally:
            with self._condition:
                self._thread_finished = True
                self._condition.notify_all()


# 功能：
#   检查 RGB 原始来源时钟，将其写入数据集状态，不能用记录完成时间冒充拍摄时间。
# 输入：
#   image、sample_mono：图像消息与对应样本单调时间。
#   frame_time：图像来源时钟绑定。
#   state：本次采样的运行状态。
# 输出：
#   state：保留原始 RGB 时钟的状态字典。
def _dataset_clock_state(image, sample_mono: float, frame_time: SensorFrameTime | None,
                         state: dict) -> dict:
    from dataclasses import asdict
    require_model_frame_time(image, frame_time, sample_mono,
                             frame_time.source_unix_ns // 1_000_000 if frame_time else 0)
    if frame_time is None:
        return state
    return {**state, "rgb_source_clock": asdict(frame_time)}


class _LatestOnlyDatasetWriter:
    """Keep training-data I/O outside the sensor-rate safety loop.

    PNG encoding, hashing, ``fsync`` and DrvFS writes may take much longer than
    one safety tick. This worker owns one pending slot: a new frame replaces an
    older frame that has not started writing. Losing a training frame under I/O
    pressure is safe; losing the local safety lease is not.
    """

    # 功能：
    #   建立仅保留一个待写样本的训练记录线程，磁盘和 PNG 编码不能拖住安全控制。
    # 输入：
    #   self：新写入器。
    #   recorder：多模态数据集记录器。
    #   semantic_label_map_sha256、semantic_label_class_ids：语义监督标注身份和合法类别。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        recorder: RuntimeMultimodalDatasetRecorder,
        *,
        semantic_label_map_sha256: str | None,
        semantic_label_class_ids: frozenset[int],
    ) -> None:
        self.recorder = recorder
        self.semantic_label_map_sha256 = semantic_label_map_sha256
        self.semantic_label_class_ids = semantic_label_class_ids
        self._condition = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._closed = False
        self._thread_finished = False
        self._issue: str | None = None
        self._submitted_count = 0
        self._completed_count = 0
        self._dropped_pending_count = 0
        self._rejected_count = 0
        self._last_completed_rgb_monotonic_seconds: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="dronedream-multimodal-dataset-writer",
            daemon=True,
        )
        self._publish_summary()
        self._thread.start()

    # 功能：
    #   读取后台记录失败原因，供采集主循环及时停止把数据集当成健康状态。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   issue：首个记录问题，没有问题时为空。
    @property
    def issue(self) -> str | None:
        with self._condition:
            return self._issue

    # 功能：
    #   读取由于最新样本覆盖待写样本而被主动舍弃的数量。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   count：被合并舍弃的待写样本数。
    @property
    def dropped_pending_count(self) -> int:
        with self._condition:
            return self._dropped_pending_count

    # 功能：
    #   交接一次采样到后台单槽；仅覆盖尚未开始写出的样本，并保留原始来源时钟。
    # 输入：
    #   self：当前写入器。
    #   rgb_image、semantic_image：由回调独立持有且交接后不再修改的图像消息。
    #   frame、sensor_snapshot：同一采样的感知帧与传感器快照。
    #   recorded_at_unix_ms、recorded_at_monotonic_seconds：记录发起时间。
    #   rgb_sample_monotonic_seconds、semantic_sample_monotonic_seconds：各图像样本时间。
    #   state、frame_time：采样状态与原始 RGB 时钟绑定。
    # 输出：
    #   accepted：是否接收此次样本，不代表该样本一定落盘。
    def submit(
        self,
        *,
        rgb_image: Any,
        frame: OnboardPerceptionFrame,
        sensor_snapshot: Any,
        recorded_at_unix_ms: int,
        recorded_at_monotonic_seconds: float,
        rgb_sample_monotonic_seconds: float,
        semantic_image: Any | None,
        semantic_sample_monotonic_seconds: float | None,
        state: dict[str, Any],
        frame_time: SensorFrameTime | None = None,
    ) -> bool:
        job = {
            "rgb_image": rgb_image,
            "frame": frame,
            "sensor_snapshot": sensor_snapshot,
            "recorded_at_unix_ms": recorded_at_unix_ms,
            "recorded_at_monotonic_seconds": recorded_at_monotonic_seconds,
            "rgb_sample_monotonic_seconds": rgb_sample_monotonic_seconds,
            "semantic_image": semantic_image,
            "semantic_sample_monotonic_seconds": semantic_sample_monotonic_seconds,
            "state": _dataset_clock_state(rgb_image, rgb_sample_monotonic_seconds,
                                           frame_time, state),
        }
        with self._condition:
            if self._closed or self._issue is not None:
                self._rejected_count += 1
                return False
            self._submitted_count += 1
            if self._pending is not None:
                self._dropped_pending_count += 1
            self._pending = job
            self._condition.notify()
        return True

    # 功能：
    #   禁止新样本并有界排空最后一个样本，超时不伪装为完整记录。
    # 输入：
    #   self：当前写入器。
    #   timeout_seconds：允许排空的最大秒数。
    # 输出：
    #   summary：记录统计和线程完成状态。
    def close(self, *, timeout_seconds: float = 5.0) -> dict[str, object]:
        if not _finite_number(timeout_seconds) or not 0 <= timeout_seconds <= threading.TIMEOUT_MAX:
            raise ValueError("DATASET_WRITER_CLOSE_TIMEOUT_INVALID")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout_seconds))
        with self._condition:
            if self._thread.is_alive() and self._issue is None:
                self._issue = "DATASET_WRITER_DRAIN_TIMEOUT"
        self._publish_summary()
        return self._summary()

    # 功能：
    #   合并数据集统计与写入线程计数，区分成功写入、主动合并和失败拒绝。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   summary：含 writer_complete 完整性判定的统计字典。
    def _summary(self) -> dict[str, object]:
        with self._condition:
            writer = {
                "writer_mode": "bounded-latest-only-background",
                "writer_submitted_count": self._submitted_count,
                "writer_completed_count": self._completed_count,
                "writer_dropped_pending_count": self._dropped_pending_count,
                "writer_rejected_count": self._rejected_count,
                "writer_pending": self._pending is not None,
                "writer_thread_finished": self._thread_finished,
                "writer_thread_alive": self._thread.is_alive(),
                "writer_complete": (
                    self._closed and self._thread_finished and not self._thread.is_alive()
                    and self._issue is None and self._rejected_count == 0
                    and self._pending is None
                    and self._completed_count + self._dropped_pending_count == self._submitted_count
                ),
                "writer_issue": self._issue,
                "writer_last_completed_rgb_monotonic_seconds": (
                    self._last_completed_rgb_monotonic_seconds
                ),
            }
        return {**self.recorder.summary(), **writer}

    # 功能：
    #   刷新低优先级数据集统计；失败保留错误状态，不抛入高优先级安全控制线程。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   None：不返回业务数据。
    def _publish_summary(self) -> None:
        # Dataset evidence has lower authority than local safety. A failed
        # summary refresh must never block or crash the control caller.
        try:
            _atomic_json(self.recorder.root / "summary.json", self._summary())
        except (OSError, TypeError, ValueError, OverflowError, RecursionError) as error:
            with self._condition:
                if self._issue is None:
                    self._issue = f"DATASET_SUMMARY_WRITE_FAILED:{type(error).__name__}"

    # 功能：
    #   编码 RGB 和语义监督图、检查纵横比并提交样本，记录失败后关闭接收并保留错误。
    # 输入：
    #   self：后台写入器自身。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    self._thread_finished = True
                    self._condition.notify_all()
                    return
                job = self._pending
                self._pending = None
            assert job is not None
            try:
                semantic_image = job["semantic_image"]
                semantic_mask_png = None
                if semantic_image is not None:
                    rgb_width = int(job["rgb_image"].width)
                    rgb_height = int(job["rgb_image"].height)
                    semantic_width = int(semantic_image.width)
                    semantic_height = int(semantic_image.height)
                    if rgb_width * semantic_height != semantic_width * rgb_height:
                        raise ValueError("RGB_SEMANTIC_ASPECT_RATIO_MISMATCH")
                    semantic_mask_png = _gazebo_semantic_label_png(
                        semantic_image,
                        allowed_class_ids=self.semantic_label_class_ids,
                    )
                self.recorder.record(
                    rgb_png=_gazebo_image_png(job["rgb_image"]),
                    frame=job["frame"],
                    sensor_snapshot=job["sensor_snapshot"],
                    recorded_at_unix_ms=job["recorded_at_unix_ms"],
                    recorded_at_monotonic_seconds=job["recorded_at_monotonic_seconds"],
                    rgb_sample_monotonic_seconds=job["rgb_sample_monotonic_seconds"],
                    semantic_mask_png=semantic_mask_png,
                    semantic_label_map_sha256=(
                        self.semantic_label_map_sha256 if semantic_mask_png is not None else None
                    ),
                    semantic_sample_monotonic_seconds=job["semantic_sample_monotonic_seconds"],
                    state=job["state"],
                )
                with self._condition:
                    self._completed_count += 1
                    self._last_completed_rgb_monotonic_seconds = job["rgb_sample_monotonic_seconds"]
            except (
                OSError, RuntimeError, TypeError, ValueError, OverflowError, RecursionError
            ) as error:
                with self._condition:
                    self._issue = f"{type(error).__name__}:{error}"
                    self._pending = None
                    self._closed = True
            self._publish_summary()


# 功能：
#   汇总非空碰撞图元的空间范围并留出余量，作为未知空间保守占据地图的边界。
# 输入：
#   primitives：静态碰撞图元列表。
#   padding_m：各方向额外扩展的米数。
# 输出：
#   bounds：地图最小与最大 ENU 向量。
def _map_bounds(
    primitives: list[dict[str, Any]], padding_m: float = 25.0
) -> tuple[Vector3, Vector3]:
    if (not isinstance(primitives, list) or not 1 <= len(primitives) <= 100_000
            or not _finite_number(padding_m) or not 0 <= padding_m <= 1000):
        raise ValueError("RUNTIME_MAP_BOUNDS_INPUT_INVALID")
    bounds = [_primitive_bounds(primitive) for primitive in primitives]
    return (
        Vector3(
            x=min(low[0] for low, _high in bounds) - padding_m,
            y=min(low[1] for low, _high in bounds) - padding_m,
            z=min(low[2] for low, _high in bounds) - padding_m,
        ),
        Vector3(
            x=max(high[0] for _low, high in bounds) + padding_m,
            y=max(high[1] for _low, high in bounds) + padding_m,
            z=max(high[2] for _low, high in bounds) + padding_m,
        ),
    )


# 功能：
#   摘要地图范围、主要语义类别和合格路线数据，为战略推理提供有界宏观背景。
# 输入：
#   semantic、primitives：已加载语义地图及碰撞图元。
#   route、clearance：通过独立净空校验的路线及报告。
# 输出：
#   context：地图和路线背景字典，不取代实时感知或控制授权。
def _strategic_map_context(
    *,
    semantic: dict[str, Any],
    primitives: list[dict[str, Any]],
    route: GraphRoute,
    clearance: RouteClearanceReport,
) -> dict[str, object]:
    minimum, maximum = _map_bounds(primitives, padding_m=0.0)
    semantic_counts = Counter(
        str(primitive.get("semantic", "unspecified")) for primitive in primitives
    )
    return {
        "schema": str(semantic.get("schema_version", "semantic-map")),
        "coordinate_frame": "Gazebo world ENU collision-envelope center",
        "bounds_min_m": minimum.model_dump(mode="json"),
        "bounds_max_m": maximum.model_dump(mode="json"),
        "collision_primitive_count": len(primitives),
        "major_semantic_counts": dict(
            sorted(
                semantic_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:12]
        ),
        "qualified_route_start": route.start_node,
        "qualified_route_goal": route.goal_node,
        "qualified_route_point_count": len(route.positions_m),
        "qualified_route_length_m": route.route_length_m,
        "qualified_minimum_clearance_m": clearance.minimum_clearance_m,
    }


# 功能：
#   按模型提交快照内的目标身份记录异步结果，避免完成时的新目标冒充旧调用的目标。
# 输入：
#   snapshot：提交模型时冻结的输入快照。
#   fallback_goal_id：无战略任务身份的非运行期调用所用兼容默认值。
# 输出：
#   goal_id：快照实际归属的语义目标身份。
def _submitted_snapshot_goal_id(
    snapshot: dict[str, object], *, fallback_goal_id: str
) -> str:
    strategic = snapshot.get("strategic_context")
    task = strategic.get("task") if isinstance(strategic, dict) else None
    goal_id = task.get("navigation_goal_id") if isinstance(task, dict) else None
    if isinstance(goal_id, str) and goal_id.strip():
        return goal_id.strip()
    return fallback_goal_id


# 功能：
#   将传感器注册表压成浅层、有界的战略摘要；本地专家仍使用单独附带的完整类型化快照。
# 输入：
#   snapshot：本轮多模态传感器注册与健康快照。
# 输出：
#   context：传感器身份、运动就绪状态及有界错误摘要。
def _strategic_sensor_context(
    snapshot: RuntimeMultimodalSensorSnapshot,
) -> dict[str, object]:
    return {
        "contract_set_sha256": snapshot.contract_set_sha256,
        "ready_for_motion": snapshot.ready_for_motion,
        "active_sensor_ids": list(snapshot.active_sensor_ids),
        "registry_issue_codes": list(snapshot.issue_codes[:16]),
        "sensor_statuses": [
            {
                "sensor_id": status.sensor_id,
                "modality": status.modality,
                "required_for_motion": status.required_for_motion,
                "health": status.health,
            }
            for status in snapshot.statuses[:16]
        ],
    }


# 功能：
#   1. 从执行回执恢复装卸状态及实测负载，不将缺少质量证据当成零负载。
#   2. 提取原生动力学来源、归一化执行器和时效条件，为负载专家提供输入。
#   3. 损坏回执拒绝本轮控制；上游就绪标记不能绕过独立的新鲜度和来源检查。
# 输入：
#   receipts_dir：当前任务按执行顺序命名的动作回执目录。
#   vehicle：经核对的机体与负载上限元数据。
#   identity_telemetry_path：未提供当前内存包时才使用的原生遥测文件。
#   identity_telemetry_payload：本控制周期已接入的原生遥测包，优先复用。
# 输出：
#   context：载荷状态、质量限制及可选的动力学摘要。
def _payload_context(
    receipts_dir: Path,
    vehicle: VehicleAsset,
    *,
    identity_telemetry_path: Path | None = None,
    identity_telemetry_payload: dict | None = None,
) -> dict[str, object]:
    state = "no-runtime-payload-evidence"
    payload_mass_kg: float | None = None
    payload_mass_invalid = False
    accepted_steps: list[str] = []
    if receipts_dir.is_dir():
        for path in sorted(receipts_dir.glob("*.receipt.json")):
            try:
                receipt = read_runtime_object(path)
            except (OSError, ValueError) as error:
                # 不能跳过损坏的装卸记录后继续假设无负载；上层会撤回本轮控制。
                raise ValueError("RUNTIME_PAYLOAD_RECEIPT_UNREADABLE") from error
            if receipt.get("status") != "accepted":
                continue
            accepted_steps.append(str(receipt.get("task_id", path.stem)))
            output = receipt.get("output")
            if not isinstance(output, dict):
                continue
            if output.get("detached") is True:
                state = "detached"
            if output.get("detached") is False:
                state = "attached"
            if output.get("payload_physics_binding_confirmed") is True:
                state = "custody-confirmed"
            if output.get("loaded_hover_stable") is True:
                state = "loaded-stable"
            if "payload_mass_kg" in output:
                raw_payload_mass = output.get("payload_mass_kg")
                if (
                    _finite_number(raw_payload_mass) and raw_payload_mass > 0.0
                ):
                    payload_mass_kg = float(raw_payload_mass)
                    payload_mass_invalid = False
                else:
                    payload_mass_kg = None
                    payload_mass_invalid = True
    context: dict[str, object] = {
        "state": state,
        "accepted_transition_steps": accepted_steps[-8:],
        "observed_payload_mass_kg": payload_mass_kg,
        "maximum_payload_kg": vehicle.max_pickup_payload_kg,
        "within_declared_payload_limit": (
            False
            if payload_mass_invalid
            else (
                payload_mass_kg <= vehicle.max_pickup_payload_kg + 1e-9
                if payload_mass_kg is not None
                else None
            )
        ),
    }
    if identity_telemetry_payload is not None or (
        identity_telemetry_path is not None and identity_telemetry_path.is_file()
    ):
        identity_telemetry = identity_telemetry_payload
        if identity_telemetry is None:
            try:
                identity_telemetry = read_runtime_object(
                    identity_telemetry_path, maximum_bytes=_MAX_NATIVE_PACKET_BYTES)
            except (OSError, ValueError) as error:
                raise ValueError("RUNTIME_PAYLOAD_TELEMETRY_UNREADABLE") from error
        if isinstance(identity_telemetry, dict):
            dynamics = identity_telemetry.get("dynamics")
            if (
                isinstance(dynamics, dict)
                and dynamics.get("schema_version") == "dronedream.px4-dynamics-telemetry.v1"
                and isinstance(dynamics.get("sources"), dict)
            ):
                sources = dynamics["sources"]
                imu = sources.get("imu") if isinstance(sources.get("imu"), dict) else {}
                attitude = (
                    sources.get("attitude") if isinstance(sources.get("attitude"), dict) else {}
                )
                battery = sources.get("battery") if isinstance(sources.get("battery"), dict) else {}
                actuator = (
                    sources.get("actuator_output")
                    if isinstance(sources.get("actuator_output"), dict)
                    else {}
                )
                actuator_values = actuator.get("actuator")
                finite_actuator_values = (
                    [
                        float(value)
                        for value in actuator_values
                        if _finite_number(value)
                    ]
                    if isinstance(actuator_values, list)
                    else []
                )
                actuator_normalization_ready = bool(actuator.get("normalization_ready") is True)
                active_actuator_values = [
                    value for value in finite_actuator_values if abs(value) > 1e-9
                ]
                maximum_sample_age_seconds = dynamics.get("maximum_sample_age_seconds")
                source_names = (
                    "imu",
                    "attitude",
                    "battery",
                    "actuator_output",
                )
                control_required_source_names = (
                    "imu",
                    "attitude",
                    "actuator_output",
                )
                source_sample_age_seconds = {
                    source_name: source.get("sample_age_seconds")
                    for source_name in source_names
                    if isinstance((source := sources.get(source_name)), dict)
                }
                freshness_contract_valid = bool(
                    _finite_number(maximum_sample_age_seconds)
                    and 0.0 < float(maximum_sample_age_seconds) <= 3.0
                    and all(
                        _finite_number(source_sample_age_seconds.get(source_name))
                        and 0.0
                        <= float(source_sample_age_seconds[source_name])
                        <= float(maximum_sample_age_seconds)
                        for source_name in control_required_source_names
                    )
                )
                issue_codes = dynamics.get("issue_codes")
                blocking_issue_codes = dynamics.get("blocking_issue_codes")
                if isinstance(issue_codes, list) and not isinstance(blocking_issue_codes, list):
                    # Recorded inputs from before the split remain replayable,
                    # but only at this ingestion boundary.  Reconstruct the
                    # stricter fast-control veto set; never treat all telemetry
                    # issues (notably a slow battery stream) as equivalent.
                    blocking_issue_codes = [
                        issue
                        for issue in issue_codes
                        if isinstance(issue, str)
                        and issue.split(":", 1)[0] in set(control_required_source_names)
                    ]
                telemetry_ready = bool(
                    dynamics.get("ready_for_payload_inference") is True
                    and freshness_contract_valid
                    and actuator_normalization_ready
                    and isinstance(issue_codes, list)
                    and isinstance(blocking_issue_codes, list)
                    and not blocking_issue_codes
                )
                context["dynamics"] = {
                    "available": bool(sources),
                    "ready": telemetry_ready,
                    "telemetry_schema_version": dynamics.get("schema_version"),
                    "telemetry_payload_sha256": sha256_json(dynamics),
                    "maximum_sample_age_seconds": maximum_sample_age_seconds,
                    "source_sample_age_seconds": source_sample_age_seconds,
                    "issue_codes": issue_codes if isinstance(issue_codes, list) else [],
                    "blocking_issue_codes": (
                        blocking_issue_codes if isinstance(blocking_issue_codes, list) else []
                    ),
                    "restart_counts": (
                        dynamics.get("restart_counts")
                        if isinstance(dynamics.get("restart_counts"), dict)
                        else {}
                    ),
                    "acceleration_forward_m_s2": imu.get("acceleration_forward_m_s2"),
                    "acceleration_right_m_s2": imu.get("acceleration_right_m_s2"),
                    "acceleration_down_m_s2": imu.get("acceleration_down_m_s2"),
                    "angular_velocity_forward_rad_s": imu.get("angular_velocity_forward_rad_s"),
                    "angular_velocity_right_rad_s": imu.get("angular_velocity_right_rad_s"),
                    "angular_velocity_down_rad_s": imu.get("angular_velocity_down_rad_s"),
                    "roll_deg": attitude.get("roll_deg"),
                    "pitch_deg": attitude.get("pitch_deg"),
                    "current_battery_a": battery.get("current_battery_a"),
                    "voltage_v": battery.get("voltage_v"),
                    "actuator_normalization_ready": actuator_normalization_ready,
                    "actuator_normalization_kind": actuator.get("normalization_kind"),
                    "actuator_normalization_absolute_maximum": actuator.get(
                        "normalization_absolute_maximum"
                    ),
                    "actuator_mean": (
                        sum(active_actuator_values) / len(active_actuator_values)
                        if active_actuator_values
                        else None
                    ),
                    "actuator_max_abs": (
                        max(abs(value) for value in active_actuator_values)
                        if active_actuator_values
                        else None
                    ),
                }
    return context


# 功能：
#   将点投影到有限线段计算最短距离；极短线段退化为点距离。
# 输入：
#   point、start、end：同一米制坐标系下的查询点及线段端点。
# 输出：
#   distance_m：点到线段的欧氏距离。
def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    delta = tuple(end[index] - start[index] for index in range(3))
    length_squared = sum(component * component for component in delta)
    if length_squared <= 1e-12:
        return math.dist(point, start)
    ratio = max(
        0.0,
        min(
            1.0,
            sum((point[index] - start[index]) * delta[index] for index in range(3))
            / length_squared,
        ),
    )
    projection = tuple(start[index] + delta[index] * ratio for index in range(3))
    return math.dist(point, projection)


# 功能：
#   1. 核对静态地图、路线和净空报告的内容绑定，仅栅格化合格路线附近的保守先验。
#   2. 在遍历前限制计算规模、遍历中限制体素数，全部分类完成后再提交地图。
# 输入：
#   world：待初始化的局部体素地图。
#   semantic_path、route_path、clearance_path：静态几何、路线及独立净空报告路径。
#   vehicle_radius_m、vehicle_height_m：完整机体碰撞包络尺寸。
#   required_clearance_m：必须保留的操作净空。
#   expected_semantic_sha256：启动时已读取地图的摘要，防止主循环和先验使用不同地图。
#   qualified_route、qualified_clearance：主入口已固定的路线和报告；同时省略才在此读取。
# 输出：
#   counts：地图接收的已知空闲与占据体素统计。
def _seed_qualified_static_map(
    *,
    world: MetricVoxelMap,
    semantic_path: Path,
    route_path: Path,
    clearance_path: Path,
    vehicle_radius_m: float,
    vehicle_height_m: float,
    required_clearance_m: float,
    expected_semantic_sha256: str | None = None,
    qualified_route: GraphRoute | None = None,
    qualified_clearance: RouteClearanceReport | None = None,
) -> tuple[int, int]:
    if (qualified_route is None) != (qualified_clearance is None):
        raise ValueError("qualified route and clearance must be supplied together")
    route = (qualified_route if qualified_route is not None
             else _read_worker_contract(route_path, GraphRoute))
    clearance = (qualified_clearance if qualified_clearance is not None
                 else _read_worker_contract(clearance_path, RouteClearanceReport))
    if not clearance.accepted:
        raise ValueError("qualified static map route clearance is not accepted")
    if clearance.route_sha256 != sha256_json(route):
        raise ValueError("qualified static map route hash mismatch")
    if (expected_semantic_sha256 is not None
            and clearance.semantic_sha256 != expected_semantic_sha256):
        raise ValueError("qualified static map semantic hash mismatch")
    if clearance.minimum_clearance_m < required_clearance_m:
        raise ValueError("qualified static map route lacks required clearance")

    geometry = KnownMapMetricPlanner(
        graph=None,
        semantic_path=semantic_path,
        expected_semantic_sha256=clearance.semantic_sha256,
        vehicle_diameter_m=vehicle_radius_m * 2.0,
        vehicle_height_m=vehicle_height_m,
        policy=MetricPlannerPolicy(
            resolution_m=world.resolution_m,
            required_clearance_m=required_clearance_m,
        ),
    )
    semantic_sha256 = geometry.semantic_sha256
    padding_m = max(2.0, vehicle_radius_m + required_clearance_m + world.resolution_m)
    route_segments = [
        ((start.x, start.y, start.z), (end.x, end.y, end.z))
        for start, end in zip(route.positions_m, route.positions_m[1:], strict=False)
    ]
    corridor_keys: set[tuple[int, int, int]] = set()
    inspected_voxels = 0
    for start, end in route_segments:
        low = Vector3(
            x=max(world.minimum_bound_m.x, min(start[0], end[0]) - padding_m),
            y=max(world.minimum_bound_m.y, min(start[1], end[1]) - padding_m),
            z=max(world.minimum_bound_m.z, min(start[2], end[2]) - padding_m),
        )
        high = Vector3(
            x=min(
                world.maximum_bound_m.x - world.resolution_m * 0.01,
                max(start[0], end[0]) + padding_m,
            ),
            y=min(
                world.maximum_bound_m.y - world.resolution_m * 0.01,
                max(start[1], end[1]) + padding_m,
            ),
            z=min(
                world.maximum_bound_m.z - world.resolution_m * 0.01,
                max(start[2], end[2]) + padding_m,
            ),
        )
        first = world.key_for(low)
        last = world.key_for(high)
        # 写入体素数不能限制遍历成本；细网格或长斜线必须在三重循环前检查计算预算。
        inspected_voxels += math.prod(max(0, last[i] - first[i] + 1) for i in range(3))
        if inspected_voxels > 8_000_000:
            raise ValueError("qualified static route corridor exceeds the traversal safety budget")
        for x in range(first[0], last[0] + 1):
            for y in range(first[1], last[1] + 1):
                for z in range(first[2], last[2] + 1):
                    key = (x, y, z)
                    center = world.center_for(key)
                    if (
                        _point_segment_distance(
                            (center.x, center.y, center.z),
                            start,
                            end,
                        )
                        <= padding_m
                    ):
                        corridor_keys.add(key)
                        if len(corridor_keys) > 1_000_000:
                            raise ValueError("qualified static route corridor exceeds voxel budget")
    classifications = []
    for key in sorted(corridor_keys):
        center = world.center_for(key)
        classifications.append(
            (
                center,
                geometry.clearance((center.x, center.y, center.z)) < 0.0,
            )
        )
    counts = world.seed_known_static_region(
        classifications,
        source_sha256=semantic_sha256,
    )
    world.bind_qualified_route(
        route.positions_m,
        route_sha256=clearance.route_sha256,
        minimum_clearance_m=clearance.minimum_clearance_m,
    )
    return counts


# 功能：
#   只消除完全包含于已知静态碰撞体的感知体素；仅重叠不能证明未知突出物不存在。
# 输入：
#   perceived、known_static：感知新增盒体与已知静态图元。
# 输出：
#   novel：仍须保留的感知障碍。
#   removed_count：被完整包含规则消除的体素数。
def _remove_known_static_perception_duplicates(
    perceived: list[dict[str, Any]],
    known_static: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    # An overlapping voxel may contain an unmodelled object protruding from
    # a known wall. Only its full containment proves that removing it cannot
    # reduce the collision union. Coarse overlap was not a valid proof.
    mask = fully_contained_box_mask(perceived, known_static)
    novel = [candidate for candidate, removed in zip(perceived, mask, strict=True) if not removed]
    return novel, len(perceived) - len(novel)


class _NativeIdentityTracker:
    """Pin deployment binding; simulator truth is diagnostics only, never a correction."""

    # 功能：
    #   初始化飞行期间不可切换的地图绑定和原生估计时间基线，不创建真值校正通路。
    # 输入：
    #   self：新身份跟踪器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self.binding_sha256: str | None = None
        self.last_validated_payload: dict[str, object] | None = None
        self.last_pose: NativeMapPose | None = None
        self.raw_disagreement_m: float | None = None

    # 功能：
    #   有界读取普通原生遥测文件，拒绝重复键与损坏 JSON 后进行身份接入。
    # 输入：
    #   self：当前跟踪器。
    #   path：原生身份遥测路径。
    #   now_unix_ms：当前真实接收时间。
    # 输出：
    #   pose：与固定地图绑定一致、通过时效校验的原生位姿。
    def evaluate(self, path: Path, *, now_unix_ms: int) -> NativeMapPose:
        payload = read_runtime_object(path, maximum_bytes=_MAX_NATIVE_PACKET_BYTES)
        return self.admit_payload(payload, now_unix_ms=now_unix_ms)

    # 功能：
    #   持有遥测独立副本，校验位姿、绑定和来源时钟不倒退；失败不推进已接入状态。
    # 输入：
    #   self：当前跟踪器。
    #   payload：原生遥测包。
    #   now_unix_ms：当前时间，用于验证未来值及过期状态。
    # 输出：
    #   pose：可供控制使用的原生地图坐标位姿。
    def admit_payload(self, payload: dict[str, object], *, now_unix_ms: int) -> NativeMapPose:
        if not isinstance(payload, dict):
            raise ValueError("NATIVE_IDENTITY_NOT_AN_OBJECT")
        payload = copy_json(payload, limit=_MAX_NATIVE_PACKET_BYTES)
        pose = native_map_pose(payload, now_unix_ms=now_unix_ms)
        if self.binding_sha256 is not None and pose.binding_sha256 != self.binding_sha256:
            raise ValueError("NATIVE_MAP_BINDING_CHANGED_DURING_EXECUTION")
        if (self.last_pose is not None
                and pose.observed_at_unix_ms < self.last_pose.observed_at_unix_ms):
            raise ValueError("NATIVE_POSE_CLOCK_REGRESSED")
        self.binding_sha256 = pose.binding_sha256
        self.last_validated_payload = payload
        self.last_pose = pose
        return pose


# 功能：
#   同时检查单调钟与学习参考 UNIX 钟，避免循环内取钟偏差使过期图像进入训练历史。
# 输入：
#   source_monotonic、source_unix_ms：原始图像的两个来源时刻。
#   reference_monotonic、reference_unix_ms：本次观测参考时刻。
# 输出：
#   ready：两个来源年龄均在原有两百毫秒范围内时为 True。
def _learning_visual_current(
    *, source_monotonic, source_unix_ms, reference_monotonic, reference_unix_ms,
) -> bool:
    ready = (type(source_unix_ms) is int and type(reference_unix_ms) is int
             and 0 <= source_unix_ms <= reference_unix_ms < 2**63
             and reference_unix_ms - source_unix_ms <= 200
             and _finite_number(source_monotonic) and source_monotonic >= 0
             and _finite_number(reference_monotonic)
             and 0 <= reference_monotonic - source_monotonic <= .2)
    return ready


# 功能：
#   复用同一来源、同一尺寸的工作器编码，避免学习线程再次解码、缩放及压缩同一帧。
# 输入：
#   image：本次学习引用的原始消息对象。
#   output_size：缓存要求的模型像素宽高。
#   sample：同一回调消息对应的已完成编码。
# 输出：
#   encoded：原始 PNG 与 RGB 不可变字节二元组。
def _reuse_learning_image(image, *, output_size, sample: PreparedCameraSample):
    if image is not sample.message or tuple(output_size) != sample.image.size:
        raise ValueError("LEARNING_PREPARED_IMAGE_SOURCE_MISMATCH")
    encoded = sample.image.png, sample.image.rgb
    return encoded


# 功能：
#   1. 编码并缓存带原始时钟的学习视觉输入，只保存新图像，生成可追溯的内容摘要和引用。
#   2. 同源同尺寸时复用已完成编码；尺寸不同时重新编码，来源或时钟不一致时拒绝。
# 输入：
#   cache、image：图像缓存与由接收方持有的消息。
#   received_monotonic、received_utc_ms：此样本对应的单调和 UNIX 时间。
#   size、directory：学习图像尺寸与归档目录。
#   frame_time：摄像头来源时钟绑定。
#   prepared_sample：可选的同一消息编码结果，不提供时按原路径编码。
# 输出：
#   references：学习记录使用的图像路径、摘要、尺寸及来源时间列表。
def _prepare_learning_visual(
    *, cache: ModelImageCache, image: Any, received_monotonic: float,
    received_utc_ms: int, size: tuple[int, int], directory: Path,
    frame_time: SensorFrameTime | None = None,
    prepared_sample: PreparedCameraSample | None = None,
) -> list[dict]:
    decoder = _gazebo_image_model_payload
    if prepared_sample is not None:
        if (not isinstance(prepared_sample, PreparedCameraSample)
                or prepared_sample.message is not image
                or prepared_sample.image.received_monotonic_seconds != received_monotonic
                or prepared_sample.image.received_at_unix_ms != received_utc_ms
                or prepared_sample.image.frame_time != frame_time):
            raise ValueError("LEARNING_PREPARED_IMAGE_SOURCE_MISMATCH")
        if prepared_sample.image.size == size:
            decoder = partial(_reuse_learning_image, sample=prepared_sample)
    prepared, new_frame = cache.prepare(
        image, received_monotonic_seconds=received_monotonic,
        received_at_unix_ms=received_utc_ms, size=size, decoder=decoder,
        frame_time=frame_time,
    )
    payload = prepared.multimodal(directory)
    if new_frame:
        _atomic_bytes(Path(payload["path"]), prepared.png)
    return [{"kind": "forward-rgb-camera", "sha256": payload["content_sha256"],
             "model_rgb_sha256": payload["model_rgb_sha256"], "content_type": "image/png",
             "path": payload["path"],
             "relative_path": str(Path("learning-observation-frames") / Path(payload["path"]).name),
             "observed_at_unix_ms": received_utc_ms,
             **{key: value for key, value in payload.items() if key == "timestamp_basis"
                or key.startswith("scene_") or key == "host_received_unix_ns"},
             "width": size[0], "height": size[1]}]


# 功能：
#   启动低优先级阶段文件观察器并注册关闭回执，控制时序内不做此类挂载盘读取。
# 输入：
#   directory：当前运行证据目录。
#   cleanup：本次工作器的退出栈。
#   receiver：可选当前执行器上下文通道，不借用磁盘状态续期。
# 输出：
#   observer：只提供阶段背景、不授予目标或运动权限的观察器。
def _start_phase_observer(
    directory: Path, cleanup: contextlib.ExitStack, *, receiver=None,
) -> RuntimePhaseObserver:
    observer = RuntimePhaseObserver(directory / "runtime-phase.json",
        reader=receiver.read_context if receiver is not None else _runtime_phase_context)
    cleanup.callback(lambda: _atomic_json(
        directory / "sensor-phase-observer-receipt.json", observer.close()))
    return observer


# 功能：
#   注册资源的异常退出清理，并使正常结束和退出栈共享一次成功关闭及其结果。
# 输入：
#   cleanup：当前工作线程入口拥有的退出栈。
#   resource：具有 close 方法的自有资源。
#   close_options：该资源支持的有界关闭选项。
# 输出：
#   close_once：可由正常收尾调用的关闭函数；成功关闭后不会重复关闭。
def _register_resource_close(cleanup: contextlib.ExitStack, resource: Any, **close_options: Any):
    closed = False
    result = None

    # 功能：
    #   关闭资源并缓存成功结果；失败继续抛出，退出栈仍可清理其他资源。
    # 输入：
    #   无；使用外层绑定的资源与关闭选项。
    # 输出：
    #   result：资源关闭回执或空值。
    def close_once():
        nonlocal closed, result
        if not closed:
            result = resource.close(**close_options)
            closed = True
        return result

    cleanup.callback(close_once)
    return close_once


# 功能：
#   建立整个原生工作器的资源生命周期，初始化或处理抛错也执行已注册清理。
# 输入：
#   无；命令行参数由内部入口读取。
# 输出：
#   exit_code：正常运行与证据排空的退出状态。
def main() -> int:
    # Native callbacks must be released even if initialization or processing
    # raises before the normal evidence-drain path.
    with contextlib.ExitStack() as cleanup:
        return _run_worker(cleanup)


# 功能：
#   有界读取本次运行的完整制品，拒绝重复键、非有限值及隐式类型转换，不复用旧文件兜底。
# 输入：
#   path：制品普通文件路径。
#   model：所需的类型化合同类。
# 输出：
#   contract：从同一份已校验字节构建的独立合同。
def _read_worker_contract(path: Path, model):
    raw = read_plugin_file(path, limit=64 * 1024 * 1024)
    payload = decode_json(raw, limit=64 * 1024 * 1024, node_limit=2_000_000)
    contract = model.model_validate_json(
        encode_json(payload, limit=64 * 1024 * 1024, node_limit=2_000_000), strict=True)
    return contract


# 功能：
#   核对当前机体真实元数据和本次限制，不以固定质量、载荷或相机列表冒充未知机体。
# 输入：
#   path：必需的当前机体元数据文件。
#   radius_m、height_m：本次碰撞包络的半径与高度。
#   speed_mps、acceleration_mps2：本次允许的速度及用于制动计算的加速度。
# 输出：
#   vehicle：与本次限制相符的独立机体合同。
def _load_worker_vehicle(path: Path, *, radius_m: float, height_m: float,
                         speed_mps: float, acceleration_mps2: float) -> VehicleAsset:
    vehicle = _read_worker_contract(path, VehicleAsset)
    if any(not _finite_number(value) or value <= 0
           for value in (radius_m, height_m, speed_mps, acceleration_mps2)):
        raise ValueError("WORKER_VEHICLE_LIMIT_INVALID")
    if abs(vehicle.body_radius_m - radius_m) > 1e-6:
        raise ValueError("vehicle radius does not match runtime clearance envelope")
    if abs(vehicle.body_height_m - height_m) > 1e-6:
        raise ValueError("vehicle height does not match runtime clearance envelope")
    if (speed_mps > vehicle.max_speed_mps + 1e-6
            or acceleration_mps2 > vehicle.max_acceleration_mps2 + 1e-6):
        raise ValueError("vehicle motion limits do not match the admitted metadata")
    return vehicle


# 功能：
#   校验执行器发布的目标和控制标志；目标可持续有效，但损坏、未来时间和字符串布尔不授予控制。
# 输入：
#   path：当前运行的目标文件。
#   now_unix_ms：当前本机时间，用于排除来自未来的目标更新。
#   reader：可选的本回合固定目录读取器，不得指向其他目标路径。
#   receiver：可选当前执行器上下文通道；提供后禁止回退旧目标文件。
# 输出：
#   target：经严格验证、独立持有的目标数据。
def _read_control_target(path: Path, *, now_unix_ms: int, reader=None, receiver=None) -> dict:
    if receiver is not None and reader is not None:
        raise ValueError("LOCAL_SAFETY_TARGET_SOURCE_AMBIGUOUS")
    if reader is not None and reader.path != path.absolute():
        raise ValueError("LOCAL_SAFETY_TARGET_READER_PATH_CHANGED")
    target = (receiver.read_target() if receiver is not None else
              reader.read() if reader is not None else
              read_runtime_object(path, maximum_bytes=128 * 1024))
    if target.get("schema_version") != "dronedream.local-safety-target.v1":
        raise ValueError("LOCAL_SAFETY_TARGET_SCHEMA_INVALID")
    for name in ("target_position_m", "navigation_goal_position_m"):
        Vector3.model_validate(target.get(name), strict=True)
    for name in ("action_checkpoint_goal", "tracking_recovery_active"):
        if type(target.get(name)) is not bool:
            raise ValueError("LOCAL_SAFETY_TARGET_FLAG_INVALID")
    goal_id = target.get("navigation_goal_id")
    if not isinstance(goal_id, str) or not goal_id.strip() or len(goal_id) > 512:
        raise ValueError("LOCAL_SAFETY_TARGET_ID_INVALID")
    if target.get("control_profile") not in ("cruise", "precision"):
        raise ValueError("LOCAL_SAFETY_TARGET_PROFILE_INVALID")
    timestamp = target.get("updated_at_unix_ms")
    if (type(now_unix_ms) is not int or type(timestamp) is not int
            or not 0 <= timestamp <= now_unix_ms <= 10**15):
        raise ValueError("LOCAL_SAFETY_TARGET_CLOCK_INVALID")
    for name in ("decision_trigger", "recovery_episode_id"):
        value = target.get(name)
        if value is not None and (not isinstance(value, str) or not value or len(value) > 512):
            raise ValueError("LOCAL_SAFETY_TARGET_CONTEXT_INVALID")
    return target


# 功能：
#   排空已发生的真实模型调用回执，包括被拒绝或超时的结果；不依赖本轮是否产生可执行动作。
# 输入：
#   coordinator：持有有界调用回执队列的模型协调器。
#   writer、path：负责实际保存的 FIFO 及本次调用证据路径。
# 输出：
#   count：本次已交给记录队列的调用条数。
def _drain_model_calls(coordinator, writer, path: Path) -> int:
    count = 0
    while count < 65:
        record = coordinator.pop_model_call_record()
        if record is None:
            return count
        if writer.submit(path, record) is not True:
            raise ValueError("MODEL_CALL_EVIDENCE_NOT_ACCEPTED")
        count += 1
    raise ValueError("MODEL_CALL_EVIDENCE_QUEUE_EXCEEDS_BUDGET")


# 功能：
#   1. 绑定地图、机体、相机与原生估计来源，装载明确选择且获准运行的模型或教师模式。
#   2. 优先处理新感知及已就绪模型控制，独立复核时效和授权后发布有界连续控制。
#   3. 将视觉留档、模型背景和学习记录移出控制截止路径，停止时排空证据并检查完整性。
# 输入：
#   cleanup：主入口拥有的资源退出栈。
#   sys.argv：本次运行的显式配置，包括主题、路径、模型包、机体限制与开发开关。
# 输出：
#   exit_code：证据正常排空为 0，关闭不完整为 2；初始化错误继续交给主入口清理。
def _run_worker(cleanup: contextlib.ExitStack) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--vehicle-metadata", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--qualified-route", type=Path)
    parser.add_argument("--qualified-clearance", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--local-safety-channel", type=Path)
    parser.add_argument("--native-state-channel", type=Path)
    parser.add_argument("--perception-health-channel", type=Path)
    parser.add_argument("--runtime-phase-channel", type=Path)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--identity-telemetry", type=Path, required=True)
    parser.add_argument("--collision-offset", nargs=3, type=float, required=True)
    parser.add_argument("--vehicle-radius", type=float, required=True)
    parser.add_argument("--vehicle-height", type=float, required=True)
    parser.add_argument("--max-speed", type=float, required=True)
    parser.add_argument("--max-acceleration", type=float, required=True)
    parser.add_argument("--required-clearance", type=float, required=True)
    parser.add_argument("--depth-topic", default="/depth_camera")
    parser.add_argument("--camera-profile-receipt", type=Path)
    parser.add_argument("--rate-hz", type=float, default=8.0)
    parser.add_argument("--live-evidence-retention-radius-m", type=float, default=12.0)
    parser.add_argument("--live-evidence-maximum-age-seconds", type=float, default=30.0)
    parser.add_argument("--live-evidence-prune-period-seconds", type=float, default=5.0)
    parser.add_argument("--local-navigation-provider")
    parser.add_argument("--simulation-training-channel", type=Path)
    parser.add_argument("--record-learning-observations", action="store_true",
                        help="Record deployment inputs without requiring any policy weights.")
    parser.add_argument("--learning-image-size", nargs=2, type=int, default=(224, 128),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--simulation-teacher-control", action="store_true")
    parser.add_argument("--local-navigation-fallback-provider")
    parser.add_argument("--local-policy-package", type=Path, action="append", default=[])
    parser.add_argument("--local-policy-qualification", type=Path, action="append", default=[])
    parser.add_argument(
        "--local-policy-simulation-admission", type=Path, action="append", default=[]
    )
    parser.add_argument("--local-navigation-model-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--local-navigation-fallback-model-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument("--local-navigation-period-seconds", type=float, default=3.0)
    parser.add_argument("--local-navigation-controller-step-m", type=float)
    parser.add_argument(
        "--local-navigation-cpu-ids",
        nargs="+",
        type=int,
        default=[],
        help="Linux CPU IDs reserved for the asynchronous local-model worker thread.",
    )
    parser.add_argument("--local-navigation-context-id")
    parser.add_argument("--require-model-control-authority", action="store_true")
    parser.add_argument(
        "--omit-coordinate-candidates",
        action="store_true",
        help=(
            "Build continuous-control observations without the superseded "
            "coordinate-candidate search. Intended for direct pilot-control "
            "packages and deterministic-teacher collection."
        ),
    )
    parser.add_argument("--model-navigation-evidence", type=Path)
    parser.add_argument("--model-navigation-call-evidence", type=Path)
    parser.add_argument("--model-navigation-snapshot-evidence", type=Path)
    parser.add_argument("--rgb-topic")
    parser.add_argument("--render-scene-epoch", help="Pinned isolated-render source identity")
    parser.add_argument("--model-navigation-frame-dir", type=Path)
    parser.add_argument("--multimodal-dataset-root", type=Path)
    parser.add_argument("--multimodal-flight-id")
    parser.add_argument("--semantic-label-topic")
    parser.add_argument("--semantic-label-map", type=Path)
    parser.add_argument("--multimodal-dataset-maximum-mib", type=int, default=5_120)
    parser.add_argument("--multimodal-record-period-seconds", type=float, default=0.1)
    parser.add_argument("--development-depth-drop-after-seconds", type=float)
    parser.add_argument("--development-depth-drop-duration-seconds", type=float)
    parser.add_argument("--development-fault-evidence", type=Path)
    parser.add_argument(
        "--development-payload-collection",
        action="store_true",
        help=(
            "Permit fresh-telemetry, reduced-step payload motion solely to collect "
            "a first dynamics corpus before a qualified payload adapter exists."
        ),
    )
    args = parser.parse_args()
    # argparse 的 float 接受 NaN/Infinity；进入任何文件或线程初始化前集中拒绝。
    for name, value in vars(args).items():
        numbers = value if isinstance(value, (tuple, list)) else (value,)
        if any(type(number) is float and not _finite_number(number) for number in numbers):
            parser.error(f"--{name.replace('_', '-')} must contain only finite numbers")
    if args.require_model_control_authority and args.local_navigation_provider not in {
        "local-policy", "simulation-training",
    }:
        parser.error("continuous model-required control requires an explicit local actor")
    if (args.qualified_route is None) != (args.qualified_clearance is None):
        parser.error("qualified route and clearance must be supplied together")
    if not 0 <= args.required_clearance <= 1000 or not 0 < args.rate_hz <= 1000:
        parser.error("clearance or sensor rate is outside the runtime budget")
    vehicle = _load_worker_vehicle(args.vehicle_metadata, radius_m=args.vehicle_radius,
        height_m=args.vehicle_height, speed_mps=args.max_speed,
        acceleration_mps2=args.max_acceleration)
    # 机体资产用于资格摘要；控制侧只收紧本次限制，不反向修改物理资产身份。
    control_vehicle = VehicleAsset.model_validate({**vehicle.model_dump(mode="python"),
        "max_speed_mps": min(args.max_speed, vehicle.max_speed_mps),
        "max_acceleration_mps2": min(args.max_acceleration, vehicle.max_acceleration_mps2)},
        strict=True)
    from dronedream_agent_core.simulation_sensor_frames import load_simulation_frame_contract

    observer_frames = load_simulation_frame_contract(
        args.command.with_name("simulation-sensor-frames.json"), vehicle_name=args.vehicle,
        collision_center_model_m=args.collision_offset)
    image_ingress = SensorImageIngress(expected_scene_epoch=args.render_scene_epoch)
    if args.render_scene_epoch is not None and args.semantic_label_topic:
        parser.error("isolated-render semantic-label source is not yet supported")
    cleanup.callback(lambda: _atomic_json(
        args.command.with_name("camera-source-clock.json"), image_ingress.summary()))
    camera_profile_readback = (CameraProfileReadback(args.camera_profile_receipt)
                               if args.camera_profile_receipt is not None else None)
    if camera_profile_readback is not None and not args.rgb_topic:
        parser.error("camera profile readback requires both native RGB and depth streams")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")
    if args.live_evidence_retention_radius_m <= 10.0:
        parser.error("--live-evidence-retention-radius-m must exceed 10 metres")
    if args.live_evidence_maximum_age_seconds <= 2.0:
        parser.error("--live-evidence-maximum-age-seconds must exceed two seconds")
    if args.live_evidence_prune_period_seconds <= 0.0:
        parser.error("--live-evidence-prune-period-seconds must be positive")
    if args.local_navigation_model_timeout_seconds <= 0.0:
        parser.error("--local-navigation-model-timeout-seconds must be positive")
    if args.local_navigation_fallback_model_timeout_seconds <= 0.0:
        parser.error("--local-navigation-fallback-model-timeout-seconds must be positive")
    if args.local_navigation_period_seconds <= 0.0:
        parser.error("--local-navigation-period-seconds must be positive")
    if any(cpu_id < 0 for cpu_id in args.local_navigation_cpu_ids):
        parser.error("--local-navigation-cpu-ids must be non-negative")
    if len(set(args.local_navigation_cpu_ids)) != len(args.local_navigation_cpu_ids):
        parser.error("--local-navigation-cpu-ids must be unique")
    if (args.multimodal_dataset_root is None) != (args.multimodal_flight_id is None):
        parser.error("multimodal recording requires both dataset root and flight identity")
    if args.multimodal_dataset_root is not None and not args.rgb_topic:
        parser.error("multimodal recording requires the onboard --rgb-topic")
    if (args.semantic_label_topic is None) != (args.semantic_label_map is None):
        parser.error("semantic supervision requires both topic and label map")
    if args.semantic_label_topic is not None and args.multimodal_dataset_root is None:
        parser.error("semantic supervision is only valid with multimodal recording")
    if not 1 <= args.multimodal_dataset_maximum_mib <= 20 * 1024:
        parser.error("--multimodal-dataset-maximum-mib is outside the safe range")
    if (
        args.local_navigation_controller_step_m is not None
        and args.local_navigation_controller_step_m <= 0.0
    ):
        parser.error("--local-navigation-controller-step-m must be positive")
    if any(not 64 <= value <= 1024 for value in args.learning_image_size):
        parser.error("learning image dimensions must be in [64, 1024]")
    if args.simulation_teacher_control and (
        not args.record_learning_observations or args.local_navigation_provider
        or args.require_model_control_authority
    ):
        parser.error("simulation teacher requires observation recording and no model authority")
    if (args.local_navigation_provider or args.record_learning_observations) and (
        args.qualified_route is None or args.qualified_clearance is None
    ):
        parser.error("model navigation requires --qualified-route and --qualified-clearance")
    if args.require_model_control_authority and not args.local_navigation_provider:
        parser.error("model control authority requires --local-navigation-provider")
    if (args.local_navigation_provider or args.record_learning_observations) and (
        args.vehicle_metadata is None
    ):
        parser.error("model navigation requires --vehicle-metadata")
    if args.local_navigation_provider == "local-policy":
        if not args.local_policy_package or not (
            args.local_policy_qualification or args.local_policy_simulation_admission
        ):
            parser.error(
                "local-policy navigation requires packages and qualification "
                "or simulation admission receipts"
            )
    elif (
        args.local_policy_package
        or args.local_policy_qualification
        or args.local_policy_simulation_admission
    ):
        parser.error("local policy artifacts require --local-navigation-provider local-policy")
    if args.local_navigation_fallback_provider == "local-policy":
        parser.error("local-policy is supported as the primary local navigation provider")
    development_fault_values = (
        args.development_depth_drop_after_seconds,
        args.development_depth_drop_duration_seconds,
        args.development_fault_evidence,
    )
    if any(value is not None for value in development_fault_values) and not all(
        value is not None for value in development_fault_values
    ):
        parser.error("development depth-drop injection requires after, duration, and evidence path")
    if (
        args.development_depth_drop_after_seconds is not None
        and args.development_depth_drop_after_seconds < 0.0
    ):
        parser.error("--development-depth-drop-after-seconds must be non-negative")
    if (
        args.development_depth_drop_duration_seconds is not None
        and args.development_depth_drop_duration_seconds <= 0.0
    ):
        parser.error("--development-depth-drop-duration-seconds must be positive")
    if args.development_payload_collection:
        if args.local_navigation_provider != "local-policy":
            parser.error("development payload collection requires the local-policy provider")
        if not args.local_policy_simulation_admission or args.local_policy_qualification:
            parser.error("development payload collection requires simulation admission only")
        if args.multimodal_dataset_root is None:
            parser.error("development payload collection requires multimodal recording")

    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node

    semantic, semantic_bytes = _read_semantic_snapshot(args.semantic)
    active_semantic_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    if not isinstance(primitives, list) or not primitives:
        raise SystemExit("semantic artifact has no runtime collision primitives")
    static_geometry_index = StaticGeometryIndex(primitives, bounds=_primitive_bounds)
    minimum, maximum = _map_bounds(primitives)
    world = MetricVoxelMap(
        resolution_m=0.25,
        minimum_bound_m=minimum,
        maximum_bound_m=maximum,
        unknown_is_occupied=True,
    )
    strategic_map_context: dict[str, object] = {}
    if args.local_navigation_provider or args.record_learning_observations:
        # 三处使用同一已解析的路线/报告，禁止先验和模型背景各自重读后混入替换文件。
        qualified_route = _read_worker_contract(args.qualified_route, GraphRoute)
        qualified_clearance = _read_worker_contract(args.qualified_clearance, RouteClearanceReport)
        _seed_qualified_static_map(
            world=world,
            semantic_path=args.semantic,
            route_path=args.qualified_route,
            clearance_path=args.qualified_clearance,
            vehicle_radius_m=args.vehicle_radius,
            vehicle_height_m=args.vehicle_height,
            required_clearance_m=args.required_clearance,
            expected_semantic_sha256=active_semantic_sha256,
            qualified_route=qualified_route,
            qualified_clearance=qualified_clearance,
        )
        strategic_map_context = _strategic_map_context(
            semantic=semantic,
            primitives=primitives,
            route=qualified_route,
            clearance=qualified_clearance,
        )
    mount = oakd_lite_depth_sensor_contract()
    sensor_registry = RuntimeSensorRegistry()
    depth_binding = DepthSensorBinding(sensor_registry, vehicle_id=vehicle.asset_id)
    fusion = RuntimePerceptionFusion(
        world=world,
        accepted_sensor_ids={"oakd-lite-depth"},
        maximum_stream_age_seconds=0.35,
        minimum_rays_per_frame=8,
        sensor_registry=sensor_registry,
    )
    if bool(args.simulation_training_channel) != (
        args.local_navigation_provider == "simulation-training"
    ):
        raise ValueError("simulation training provider requires its explicit local channel")
    if args.simulation_training_channel and (
        not args.require_model_control_authority or args.local_navigation_fallback_provider
        or args.local_policy_package or args.local_policy_qualification
        or args.local_policy_simulation_admission or args.simulation_teacher_control
    ):
        raise ValueError("simulation learner cannot mix teacher, package or cloud authority")
    model_navigation = None
    close_model_navigation = None
    local_navigation_output_mode = "legacy-candidate-selection"
    if args.record_learning_observations:
        if args.rgb_topic:
            require_model_image_runtime()
        local_navigation_output_mode = "normalized-body-velocity"
        args.omit_coordinate_candidates = True
        args.rate_hz = max(20., args.rate_hz)
    model_navigation_visual_size: tuple[int, int] | None = None
    if args.local_navigation_provider:
        # 协调器尚未接管时，端口初始化之后的任何配置错误也必须能释放已建资源。
        model_port_cleanup = contextlib.ExitStack()
        cleanup.callback(model_port_cleanup.close)
        if args.local_navigation_provider == "simulation-training":
            from dronedream_agent_core.training.policy_port import SimulationTrainingPolicyPort

            model_port = SimulationTrainingPolicyPort(args.simulation_training_channel)
            local_navigation_output_mode = "normalized-body-velocity"
            args.omit_coordinate_candidates = True
            model_navigation_visual_size = (
                tuple(args.learning_image_size) if args.rgb_topic else None
            )
        elif args.local_navigation_provider == "local-policy":
            packages = [load_local_policy_package(path) for path in args.local_policy_package]
            qualifications = [
                _read_worker_contract(path, LocalPolicyQualificationReceipt)
                for path in args.local_policy_qualification
            ]
            simulation_admissions = [
                _read_worker_contract(path, LocalPolicySimulationAdmissionReceipt)
                for path in args.local_policy_simulation_admission
            ]
            active_map_sha256 = active_semantic_sha256
            if simulation_admissions:
                selection = select_local_policy_for_simulation(
                    packages=packages,
                    admissions=simulation_admissions,
                    qualification_receipts=qualifications,
                    map_sha256=active_map_sha256,
                    vehicle_sha256=sha256_json(vehicle),
                    sensor_contract_sha256=sha256_json(mount),
                )
            else:
                selection = select_local_policy(
                    packages=packages,
                    receipts=qualifications,
                    map_sha256=active_map_sha256,
                    vehicle_sha256=sha256_json(vehicle),
                    sensor_contract_sha256=sha256_json(mount),
                )
            selected_package = next(
                package
                for package in packages
                if package.package_sha256 == selection.package_sha256
            )
            local_navigation_output_mode = (
                selected_package.manifest.pilot_control_mode
                or "legacy-candidate-selection"
            )
            if args.require_model_control_authority and (
                selected_package.manifest.realtime_feature_count is None
                or selected_package.manifest.pilot_control_mode
                != "normalized-body-velocity"
            ):
                raise SystemExit(
                    "model-required flight needs a realtime-feature policy with a "
                    "qualified continuous pilot-control head"
                )
            has_visual_encoder = "perception-encoder" in selected_package.artifact_paths
            if has_visual_encoder != bool(args.rgb_topic):
                raise SystemExit(
                    "selected local policy and forward-RGB runtime configuration do not match"
                )
            if has_visual_encoder:
                visual_width = selected_package.manifest.visual_width
                visual_height = selected_package.manifest.visual_height
                if visual_width is None or visual_height is None:
                    raise SystemExit("selected local visual policy has no frame dimensions")
                model_navigation_visual_size = (visual_width, visual_height)
            _atomic_json(
                args.command.with_name("local-policy-selection.json"),
                {
                    **selection.model_dump(mode="json"),
                    "map_sha256": active_map_sha256,
                    "vehicle_sha256": sha256_json(vehicle),
                    "sensor_contract_sha256": sha256_json(mount),
                    "development_payload_collection": bool(args.development_payload_collection),
                    "flight_qualification_granted": False
                    if args.development_payload_collection
                    else None,
                },
            )
            model_port: Any = LocalPolicyPort(
                selected_package,
                development_payload_collection=args.development_payload_collection,
                visual_worker_cpu_ids=tuple(args.local_navigation_cpu_ids),
                scheduling_jitter_grace_ms=0.0,
            )
        else:
            model_port = StructuredModelPort(
                args.local_navigation_provider,
                max_attempts=1,
                timeout_seconds=args.local_navigation_model_timeout_seconds,
            )
        if callable(getattr(model_port, "close", None)):
            model_port_cleanup.callback(model_port.close)
        if args.local_navigation_fallback_provider and (
            local_navigation_output_mode != "normalized-body-velocity"
        ):
            fallback_port = StructuredModelPort(
                args.local_navigation_fallback_provider,
                max_attempts=1,
                timeout_seconds=args.local_navigation_fallback_model_timeout_seconds)
            if callable(getattr(fallback_port, "close", None)):
                model_port_cleanup.callback(fallback_port.close)
            model_port = FailoverStructuredModelPort(
                model_port,
                fallback_port,
                fallback_accepts_multimodal=False,
                # Local experts own the real-time loop.  A completed local
                # result that narrowly misses its admitted latency or output
                # contract causes a one-cycle hold and immediate local retry;
                # it must not move every 200 ms decision onto a multi-second
                # cloud provider.  Only a true outer timeout opens the
                # fallback circuit, and one successful recovery call is enough
                # before the local primary is re-probed.
                primary_probe_after_fallback_successes=(
                    1 if args.local_navigation_provider == "local-policy" else 3
                ),
                retry_primary_after_returned_failure=(
                    args.local_navigation_provider == "local-policy"
                ),
            )
        control_timing = resolve_control_timing(
            output_mode=local_navigation_output_mode,
            sensor_tick_hz=args.rate_hz,
            decision_period_seconds=args.local_navigation_period_seconds,
            provider_timeout_seconds=args.local_navigation_model_timeout_seconds,
        )
        args.rate_hz = control_timing.sensor_tick_hz
        args.local_navigation_period_seconds = control_timing.decision_period_seconds
        _atomic_json(args.command.with_name("local-control-timing.json"), {
            "control_output_mode": local_navigation_output_mode,
            "effective": control_timing.as_dict(),
            "input_arrival_maximum_hz": sensor_input_maximum_rate_hz(
                maintenance_rate_hz=args.rate_hz,
                continuous_local_control=(
                    args.local_navigation_provider in {"local-policy", "simulation-training"}
                    and local_navigation_output_mode == "normalized-body-velocity")),
            "cloud_fallback_controls_motion": bool(
                args.local_navigation_fallback_provider
                and local_navigation_output_mode != "normalized-body-velocity"
            ),
        })
        model_navigation = EventDrivenIndoorNavigationCoordinator(
            fusion=fusion,
            port=model_port,
            required_clearance_m=args.required_clearance,
            candidate_speed_mps=(
                args.max_speed
                if local_navigation_output_mode == "normalized-body-velocity"
                else min(0.75, max(0.2, args.max_speed * 0.25))
            ),
            vehicle_radius_m=args.vehicle_radius,
            vehicle_height_m=args.vehicle_height,
            maximum_decision_age_seconds=control_timing.maximum_decision_age_seconds,
            control_lease_seconds=control_timing.control_lease_seconds,
            maximum_controller_step_m=args.local_navigation_controller_step_m,
            model_worker_cpu_ids=tuple(args.local_navigation_cpu_ids),
            control_output_mode=local_navigation_output_mode,
            # Shadow collection distils only the deterministic command that
            # actually flew.  It does not need, and must not learn from, the
            # superseded coordinate-candidate search used by the admitted
            # legacy observer package.
            include_candidate_paths=_coordinate_candidates_enabled(
                omitted=args.omit_coordinate_candidates,
                control_output_mode=local_navigation_output_mode,
            ),
        )
        close_model_navigation = _register_resource_close(cleanup, model_navigation)
        # 协调器 close 已拥有端口；取消临时所有权，不再由两个生命周期重复关闭。
        model_port_cleanup.pop_all()
    model_navigation_evidence = args.model_navigation_evidence or args.command.with_name(
        "model-navigation-cycles.jsonl"
    )
    model_navigation_call_evidence = args.model_navigation_call_evidence or args.command.with_name(
        "model-navigation-model-calls.jsonl"
    )
    model_navigation_snapshot_evidence = (
        args.model_navigation_snapshot_evidence
        or args.command.with_name("model-navigation-snapshots.jsonl")
    )
    model_navigation_frame_dir = (
        args.model_navigation_frame_dir or args.command.parent / "model-navigation-frames"
    )
    semantic_label_map_sha256: str | None = None
    semantic_label_class_ids: frozenset[int] = frozenset()
    if args.semantic_label_map is not None:
        semantic_label_map_sha256, semantic_label_class_ids = load_local_vision_label_map(
            args.semantic_label_map
        )
    multimodal_dataset = (
        RuntimeMultimodalDatasetRecorder(
            args.multimodal_dataset_root,
            flight_id=args.multimodal_flight_id,
            map_sha256=active_semantic_sha256,
            maximum_bytes=args.multimodal_dataset_maximum_mib * 1024 * 1024,
            minimum_period_seconds=args.multimodal_record_period_seconds,
        )
        if args.multimodal_dataset_root is not None
        else None
    )
    multimodal_dataset_writer = (
        _LatestOnlyDatasetWriter(
            multimodal_dataset,
            semantic_label_map_sha256=semantic_label_map_sha256,
            semantic_label_class_ids=semantic_label_class_ids,
        )
        if multimodal_dataset is not None
        else None
    )
    close_dataset_writer = (_register_resource_close(
        cleanup, multimodal_dataset_writer, timeout_seconds=8.0,
    ) if multimodal_dataset_writer is not None else None)
    local_safety_history = args.history or args.command.with_name(
        "depth-local-safety-history.jsonl"
    )
    runtime_evidence_writer = _BoundedRuntimeEvidenceWriter(
        args.command.with_name("runtime-evidence-writer-summary.json"),
        flush_on_record_paths=(local_safety_history,)
        if args.local_navigation_provider == "simulation-training" else (),
    )
    close_evidence_writer = _register_resource_close(
        cleanup, runtime_evidence_writer, timeout_seconds=8.0)
    runtime_snapshot_writer = _LatestRuntimeSnapshotWriter(
        args.command.with_name("runtime-snapshot-writer-summary.json")
    )
    close_snapshot_writer = _register_resource_close(
        cleanup, runtime_snapshot_writer, timeout_seconds=4.0)
    safety_publisher = (LocalSafetyPublisher(args.local_safety_channel)
                        if args.local_safety_channel is not None else None)
    close_safety_publisher = (_register_resource_close(cleanup, safety_publisher)
                              if safety_publisher is not None else None)
    health_publisher = (PerceptionHealthPublisher(args.perception_health_channel)
                        if args.perception_health_channel is not None else None)
    close_health_publisher = (_register_resource_close(cleanup, health_publisher)
                             if health_publisher is not None else None)
    learning_recorder = (
        LearningObservationRecorder(lambda record: runtime_evidence_writer.submit(
            args.command.with_name("learning-observations.jsonl"), record,
        )) if args.record_learning_observations else None
    )
    close_learning_recorder = (_register_resource_close(cleanup, learning_recorder)
                               if learning_recorder is not None else None)
    geometry_capture = (GeometryObservationCapture(args.command.parent,
        map_sha256=active_semantic_sha256,
        summary_publisher=_atomic_json) if args.record_learning_observations else None)
    if geometry_capture is not None:
        cleanup.callback(geometry_capture.close)
    bridge = MetricRangeSensorBridge(mount)
    tracker = DepthMotionTracker(known_static_primitives=primitives)
    # Native state progresses independently of depth rendering and inference.
    native_state_sampler = NativeStateSampler(
        args.identity_telemetry, channel_path=args.native_state_channel,
    )
    close_native_sampler = _register_resource_close(cleanup, native_state_sampler)
    native_state_sampler.start()
    phase_receiver = (RuntimePhaseReceiver(args.runtime_phase_channel)
                      if args.runtime_phase_channel is not None else None)
    if phase_receiver is not None:
        cleanup.callback(phase_receiver.close)
    target_reader = (None if phase_receiver is not None else
                     PinnedRuntimeObjectReader(args.target, maximum_bytes=128 * 1024))
    if target_reader is not None:
        cleanup.callback(target_reader.close)
    lock = threading.Lock()
    latest_pose: tuple[Any, float] | None = None
    latest_image: tuple[Any, float, int] | None = None
    depth_history: deque[tuple[Any, float, int]] = deque(maxlen=16)
    latest_rgb: tuple[Any, float] | None = None
    latest_rgb_received_at_unix_ms: int | None = None
    model_image_cache = ModelImageCache()
    prepared_image_size = (model_navigation_visual_size or
                           (tuple(args.learning_image_size)
                            if args.record_learning_observations else None))
    model_image_worker = (
        LatestModelImageWorker(size=prepared_image_size,
                               decoder=_gazebo_image_model_payload, prepare_sensor_quality=True)
        if args.rgb_topic and prepared_image_size is not None
        and (args.local_navigation_provider in {"local-policy", "simulation-training"}
             or args.record_learning_observations)
        else None
    )
    close_model_image_worker = (_register_resource_close(cleanup, model_image_worker)
                                if model_image_worker is not None else None)
    # 退出栈逆序清理：先停止并排空可能投递图像的订阅，再关闭其消费线程。
    node = Node()
    subscriptions = GazeboSubscriptions(node)
    cleanup.callback(subscriptions.close)
    last_model_image_time = -1.
    learning_image_cache = ModelImageCache()
    latest_semantic_label: tuple[Any, float] | None = None
    rgb_history: deque[tuple[Any, float]] = deque(maxlen=12)
    semantic_label_history: deque[tuple[Any, float]] = deque(maxlen=12)
    rgb_binding = ForwardRgbBinding(sensor_registry, vehicle_id=vehicle.asset_id)
    last_dataset_rgb_received_at = -1.0
    camera_profile_recorded = False

    # 功能：
    #   独立持有仿真位姿消息用于诊断，不将其替代原生估计状态或用于纠偏。
    # 输入：
    #   message：Gazebo 位姿数组消息。
    # 输出：
    #   None：不返回业务数据。
    def on_pose(message: Any) -> None:
        nonlocal latest_pose
        copied = Pose_V()
        copied.CopyFrom(message)
        with lock:
            latest_pose = copied, time.monotonic()

    # 功能：
    #   校验深度来源时钟、复制消息并更新有界接收历史，不在原生回调内投影或写盘。
    # 输入：
    #   message：深度相机原始消息。
    # 输出：
    #   None：不返回业务数据。
    def on_image(message: Any) -> None:
        nonlocal latest_image
        frame_time = image_ingress.admit("depth", message, received_unix_ns=time.time_ns(),
                                          received_monotonic_seconds=time.monotonic())
        if frame_time is None:
            return
        copied = Image()
        copied.CopyFrom(message)
        with lock:
            if camera_profile_readback is not None:
                camera_profile_readback.observe("depth", message.width, message.height)
            latest_image = (copied, frame_time.sample_monotonic_seconds,
                            frame_time.source_unix_ns // 1_000_000)
            depth_history.append(latest_image)

    # 功能：
    #   接入带时钟绑定的 RGB，核对实际相机尺寸后将自有消息送往后台模型图像编码器。
    # 输入：
    #   message：前视 RGB 相机消息。
    # 输出：
    #   None：不返回业务数据。
    def on_rgb(message: Any) -> None:
        nonlocal latest_rgb, latest_rgb_received_at_unix_ms
        frame_time = image_ingress.admit("rgb", message, received_unix_ns=time.time_ns(),
                                          received_monotonic_seconds=time.monotonic())
        if frame_time is None:
            return
        copied = Image()
        copied.CopyFrom(message)
        with lock:
            shape_accepted = (camera_profile_readback is None or camera_profile_readback.observe(
                "rgb", message.width, message.height))
            latest_rgb = copied, frame_time.sample_monotonic_seconds
            latest_rgb_received_at_unix_ms = frame_time.source_unix_ns // 1_000_000
            rgb_history.append(latest_rgb)
            if model_image_worker is not None and shape_accepted:
                # Conversion starts on receipt, not at the end of a safety tick.
                # The callback's owned protobuf copy is never mutated afterwards.
                model_image_worker.submit(copied,
                    received_monotonic_seconds=latest_rgb[1],
                    received_at_unix_ms=latest_rgb_received_at_unix_ms, frame_time=frame_time)

    # 功能：
    #   保存训练监督专用语义图的独立副本和接收时间，不把真值标签提供给控制模型。
    # 输入：
    #   message：仿真语义标签图像消息。
    # 输出：
    #   None：不返回业务数据。
    def on_semantic_label(message: Any) -> None:
        nonlocal latest_semantic_label
        copied = Image()
        copied.CopyFrom(message)
        with lock:
            latest_semantic_label = copied, time.monotonic()
            semantic_label_history.append(latest_semantic_label)

    subscriptions.subscribe(Pose_V, f"/world/{args.world}/dynamic_pose/info", on_pose)
    subscriptions.subscribe(Image, args.depth_topic, on_image)
    if args.rgb_topic:
        subscriptions.subscribe(Image, args.rgb_topic, on_rgb)
    if args.semantic_label_topic:
        subscriptions.subscribe(Image, args.semantic_label_topic, on_semantic_label)
    sequence = 0
    identity_alignment = _NativeIdentityTracker()
    control_session_id = uuid4().hex
    previous_pose: tuple[float, Vector3] | None = None
    last_image_received_at = -1.0
    last_depth_processing_failed = False
    logged_error_kinds: set[str] = set()
    first_accepted_frame_monotonic: float | None = None
    stream_unhealthy_since_monotonic: float | None = None
    development_fault_activated_at_unix_ms: int | None = None
    development_fault_recovered_at_unix_ms: int | None = None
    next_tick = time.monotonic()
    next_model_cycle = 0.0
    model_cycle_started = False
    last_model_target_sha256: str | None = None
    last_dynamic_signature: str | None = None
    dynamic_recovery_episode: dict[str, object] | None = None
    last_decision_compute_seconds: float | None = None
    precision_latched_goal_id: str | None = None
    last_live_evidence_prune_at = -math.inf
    latest_realtime_feature_snapshot = None
    if args.development_fault_evidence is not None:
        _atomic_json(
            args.development_fault_evidence,
            {
                "schema_version": "dronedream.development-depth-fault.v1",
                "development_only": True,
                "fault_kind": "depth-frame-drop",
                "drop_after_first_accepted_frame_seconds": (
                    args.development_depth_drop_after_seconds
                ),
                "drop_duration_seconds": args.development_depth_drop_duration_seconds,
                "activated": False,
                "recovered": False,
                "activated_at_unix_ms": None,
                "recovered_at_unix_ms": None,
            },
        )
    stop_requested = threading.Event()
    local_input_wake_enabled = local_input_cadence_enabled(
        args.local_navigation_provider, local_navigation_output_mode,
        simulation_teacher_control=args.simulation_teacher_control)
    if local_input_wake_enabled:
        # Native receive, camera encoding, inference and durable writers share
        # this interpreter. Do not let each ready thread retain a full 5 ms
        # quantum in a control path whose entire sensor lease is only 250 ms.
        configure_sensor_thread_handoff()

    # 功能：
    #   信号处理器只发出停止事件，由主循环在正常上下文排空证据和释放原生资源。
    # 输入：
    #   _signum、_frame：系统传入的信号编号及当前执行帧，不用于控制计算。
    # 输出：
    #   None：不返回业务数据。
    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    ready_control_scheduler = ReadyControlScheduler()
    # Stage metadata must not perform mounted-filesystem I/O in the input or
    # reply deadline. This observer grants no target/action authority and its
    # missing/stale state never skips a control tick or renews a sensor source.
    phase_observer = _start_phase_observer(args.command.parent, cleanup, receiver=phase_receiver)
    sensor_arrival_scheduler = SensorArrivalScheduler(rate_hz=sensor_input_maximum_rate_hz(
        maintenance_rate_hz=args.rate_hz,
        continuous_local_control=local_input_wake_enabled))
    interpreter_pause_monitor = InterpreterPauseMonitor()
    close_pause_monitor = _register_resource_close(cleanup, interpreter_pause_monitor)
    previous_tick_timing = None
    perception_timing_summary = PhaseTimingSummary()
    while not stop_requested.is_set():
        if runtime_evidence_writer.issue is not None or runtime_snapshot_writer.issue is not None:
            raise RuntimeError("RUNTIME_CONTROL_RECORDING_FAILED")
        now = time.monotonic()
        tick_ms = int(time.time() * 1000)
        fault_active = _development_depth_frame_suppressed(
            first_accepted_frame_monotonic=first_accepted_frame_monotonic,
            now_monotonic=now,
            drop_after_seconds=args.development_depth_drop_after_seconds,
            drop_duration_seconds=args.development_depth_drop_duration_seconds,
        )
        result_ready = bool(model_navigation is not None
                            and model_navigation.continuous_result_ready)
        local_pending = bool(model_navigation is not None
            and args.local_navigation_provider in {"local-policy", "simulation-training"}
            and model_navigation.continuous_request_pending)
        control_perception_healthy = bool((result_ready or local_pending) and fusion.has_frame
            and fusion.health(now_unix_ms=tick_ms, now_monotonic_seconds=now).stream_healthy)
        # Feature retention (e.g. 350/500 ms geometry/tracks) is not a model
        # action lease. Waiting for an already non-dispatchable reply would
        # delay the independent new depth frame that the next action needs.
        dispatch_budget_ready = bool(latest_realtime_feature_snapshot is not None
            and latest_realtime_feature_snapshot.control_deadline_unix_ms(
                now_unix_ms=tick_ms) - tick_ms >= LOCAL_DISPATCH_RESERVE_MS
            and model_navigation is not None
            and model_navigation.continuous_request_remaining_ms(
                now_unix_ms=tick_ms) >= LOCAL_DISPATCH_RESERVE_MS)
        prioritize_ready_control = ready_control_scheduler.eligible(
            depth_sequence=sequence, result_ready=result_ready,
            perception_healthy=control_perception_healthy,
            features_retain_dispatch_budget=dispatch_budget_ready,
            depth_processing_failed=last_depth_processing_failed, fault_active=fault_active,
        )
        with lock:
            arrival_sample = latest_image[1] if latest_image is not None else None
        sensor_arrival_due = bool(local_input_wake_enabled
            and not (local_pending and dispatch_budget_ready)
            and sensor_arrival_scheduler.eligible(
                now_monotonic=now, sample_monotonic=arrival_sample,
                processed_sample_monotonic=last_image_received_at, fault_active=fault_active))
        if now >= next_tick and not prioritize_ready_control and (
            ready_control_scheduler.await_pending_result(
                now_monotonic=now, depth_sequence=sequence, continuous_pending=local_pending,
                result_ready=result_ready, perception_healthy=control_perception_healthy,
                features_retain_dispatch_budget=dispatch_budget_ready,
                depth_processing_failed=last_depth_processing_failed, fault_active=fault_active)
        ):
            time.sleep(.002)
            continue
        if now < next_tick and not prioritize_ready_control and not sensor_arrival_due:
            time.sleep(min(.002 if local_input_wake_enabled or (model_navigation is not None
                           and model_navigation.request_pending) else .02, next_tick - now))
            continue
        if not prioritize_ready_control:
            next_tick = now + 1.0 / args.rate_hz
            if sensor_arrival_due:
                sensor_arrival_scheduler.record_attempt(now_monotonic=now)
        else:
            ready_control_scheduler.consume(sequence)
        cycle_timing = PhaseTimings()
        with lock:
            pose_item = latest_pose
            image_item = latest_image
            depth_items = tuple(depth_history)
            rgb_item = latest_rgb
            rgb_received_at_unix_ms = latest_rgb_received_at_unix_ms
            semantic_label_item = latest_semantic_label
            rgb_items = tuple(rgb_history)
            semantic_label_items = tuple(semantic_label_history)
        if image_item is None:
            continue
        selected_image_time = None if prioritize_ready_control else (
            native_state_sampler.buffer.select_image_time(
            tuple(item[2] for item in depth_items if item[1] > last_image_received_at),
            now_unix_ms=int(time.time() * 1000),
            maximum_range_m=mount.maximum_range_m,
            maximum_acceleration_mps2=args.max_acceleration,
            maximum_alignment_variance_m2=fusion.maximum_localization_covariance_m2,
        ))
        if selected_image_time is not None:
            image_item = next(item for item in reversed(depth_items)
                              if item[2] == selected_image_time)
        image, image_received_at, image_received_at_unix_ms = image_item
        new_depth_frame = (selected_image_time is not None
                           and image_received_at > last_image_received_at)
        if new_depth_frame and not fault_active:
            # Normal timer ticks and early arrival ticks share the same input
            # rate bound. A maintenance-only pass cannot postpone fresh input.
            sensor_arrival_scheduler.record_attempt(now_monotonic=now)
        stale_tick = fault_active or not new_depth_frame
        depth_input_timing = None
        if new_depth_frame and not fault_active:
            input_clock = image_ingress.lookup("depth", image_received_at)
            if input_clock is not None:
                depth_input_timing = sensor_processing_timing(
                    input_clock, processing_monotonic=time.monotonic())
        # A ready action may use the existing still-fresh world before spending
        # another scan period on projection/tracking. The pending depth frame
        # is not marked consumed; the next ordinary tick must process it.
        latest_fused_frame = fusion.frozen_frame if stale_tick else None
        if stale_tick and last_depth_processing_failed:
            # A frame may already have advanced the tracker before a later
            # fusion/encoder check failed. Do not replay that source, or pair
            # a partially updated world with encodings from the previous frame.
            # Only an independent next frame can recover; old leases expire.
            continue
        if stale_tick and latest_fused_frame is None:
            continue
        wall_clock_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        # Truth is an optional external diagnostic. Losing this stream cannot
        # replace the estimator or change a learned action's frame transform.
        truth_position = None
        if pose_item is not None:
            pose_message, pose_received_at = pose_item
            resolved = _resolve_controlled_vehicle_pose(
                pose_message.pose, vehicle_name=args.vehicle,
                collision_center_offset_model_m=tuple(args.collision_offset),
                frames=observer_frames,
            )
            if resolved is not None:
                root = resolved[0]
                truth_position = Vector3(**dict(zip(("x", "y", "z"),
                    (root[i] + args.collision_offset[i] for i in range(3)), strict=True)))
                if previous_pose is not None and pose_received_at - previous_pose[0] > 1e-4:
                    elapsed = pose_received_at - previous_pose[0]
                    wall_clock_velocity = Vector3(
                        x=(truth_position.x - previous_pose[1].x) / elapsed,
                        y=(truth_position.y - previous_pose[1].y) / elapsed,
                        z=(truth_position.z - previous_pose[1].z) / elapsed,
                    )
                previous_pose = pose_received_at, truth_position
        now_unix_ms = int(time.time() * 1_000)
        cycle_outcome = "rejected"
        try:
            cycle_timing.mark("image_selection")
            if camera_profile_readback is not None:
                with lock:
                    profile_proof = camera_profile_readback.require_ready()
                if not camera_profile_recorded:
                    _atomic_json(args.command.with_name("camera-profile-readback.json"),
                                 profile_proof)
                    camera_profile_recorded = True
            if model_navigation is not None:
                _drain_model_calls(model_navigation, runtime_evidence_writer,
                                   model_navigation_call_evidence)
            prepared_rgb_sample = None
            if model_image_worker is not None:
                prepared_rgb_sample = model_image_worker.latest(
                    now_monotonic_seconds=now, maximum_age_seconds=.2,
                )
                # Do not attach the newest raw source's timestamp or registry
                # quality to an older converted frame. Both use the exact source.
                rgb_item = ((prepared_rgb_sample.message,
                             prepared_rgb_sample.image.received_monotonic_seconds)
                            if prepared_rgb_sample is not None else None)
                rgb_received_at_unix_ms = (
                    prepared_rgb_sample.image.received_at_unix_ms
                    if prepared_rgb_sample is not None else None)
            if args.rgb_topic and rgb_item is not None:
                rgb_binding.register(
                    rgb_item[0], source_monotonic=rgb_item[1],
                    source_unix_ms=rgb_received_at_unix_ms,
                    frame_time=(prepared_rgb_sample.image.frame_time
                                if prepared_rgb_sample is not None
                                else image_ingress.lookup("rgb", rgb_item[1])),
                    now_monotonic=now,
                    prepared_measurement=(prepared_rgb_sample.measurement
                                          if prepared_rgb_sample is not None else None),
                )
            native_observation = native_state_sampler.buffer.latest(now_unix_ms=now_unix_ms)
            native_pose = identity_alignment.admit_payload(
                native_observation.payload, now_unix_ms=now_unix_ms,
            )
            identity_ok, identity_issues = True, []
            estimator_offset = Vector3(x=0, y=0, z=0)
            position = native_pose.position_world_enu_m
            orientation = native_pose.orientation_world_from_body
            velocity = native_pose.velocity_world_enu_mps
            identity_alignment.raw_disagreement_m = (
                math.dist(tuple(position.model_dump().values()),
                          tuple(truth_position.model_dump().values()))
                if truth_position is not None else None
            )
            native_sample = native_observation.sample
            latest_flight_state_encoding = native_observation.encoding
            # State history advances on native evidence, including ticks with
            # no new depth image. Re-reading a packet never renews its lease.
            # Native IMU acceleration is in physical m/s². Differentiating
            # simulator velocity using host-loop wall time changes its units
            # when Gazebo's real-time factor changes, so never do that here.
            acceleration_world_enu_mps2 = (
                body_to_world_enu(orientation, native_sample.acceleration_body_mps2)
                if native_sample.acceleration_body_mps2 is not None else None
            )
            visual_motion_alignment = forward_camera_motion_alignment(
                body_orientation_world_from_body=orientation,
                body_velocity_world_enu_mps=velocity,
                mount=mount,
            )
            cycle_timing.mark("native_state_and_rgb_registration")
            if stale_tick:
                frame = latest_fused_frame
                if frame is None:
                    continue
                dynamic = list(frame.dynamic_obstacles)
                health = fusion.health(now_unix_ms=now_unix_ms)
            else:
                sequence += 1
                aligned_native = native_state_sampler.buffer.align_image(
                    image_received_at_unix_ms=image_received_at_unix_ms,
                    now_unix_ms=now_unix_ms,
                    maximum_range_m=mount.maximum_range_m,
                    maximum_acceleration_mps2=args.max_acceleration,
                )
                projection = depth_binding.project(image)
                scan = RawMetricRangeScan(
                    sensor_id="oakd-lite-depth",
                    sequence=sequence,
                    # Track motion uses the depth source clock. Native state
                    # keeps its own, possibly older deadline in the fusion.
                    observed_at_unix_ms=image_received_at_unix_ms,
                    observed_at_monotonic_seconds=image_received_at,
                    body_position_world_enu_m=aligned_native.pose.position_world_enu_m,
                    body_orientation_world_from_body=aligned_native.pose.orientation_world_from_body,
                    body_velocity_world_enu_mps=aligned_native.pose.velocity_world_enu_mps,
                    localization_covariance_m2=aligned_native.conservative_variance_m2,
                    samples=list(projection.samples),
                    source_coverage=projection.source_coverage,
                )
                frame = bridge.assemble(scan)
                cycle_timing.mark("depth_projection")
                if geometry_capture is not None:
                    geometry_capture.record(scan, mount=mount,
                        calibration_sha256=projection.calibration_sha256,
                        native_pose_binding_sha256=aligned_native.pose.binding_sha256,
                        source_clock=depth_input_timing)
                    cycle_timing.mark("localization_capture_enqueue")
                last_image_received_at = image_received_at
                last_depth_processing_failed = True
                prepared_tracks = tracker.prepare(
                    frame,
                    observed_at_monotonic_seconds=image_received_at,
                )
                # This frame was just assembled locally and has not been
                # shared. Assignment validates the dynamic field; fusion then
                # validates and freezes the complete graph at its boundary.
                # Rebuilding every measured ray here adds no source evidence.
                frame.dynamic_obstacles = list(prepared_tracks.observations)
                health = fusion.ingest(
                    frame,
                    now_unix_ms=now_unix_ms,
                    now_monotonic_seconds=now,
                )
                tracker.commit(prepared_tracks)
                frame = fusion.frozen_frame
                if frame is None:
                    raise ValueError("PERCEPTION_FRAME_NOT_RECEIVED")
                dynamic = list(frame.dynamic_obstacles)
                cycle_timing.mark("tracking_and_metric_fusion")
                geometry_encoding = encode_metric_geometry(
                    scan,
                    sensor_mount=mount,
                    expected_horizontal_fov_rad=1.274,
                    expected_vertical_fov_rad=2 * math.atan(
                        math.tan(1.274 / 2) * int(image.height) / int(image.width)
                    ),
                    encoded_at_unix_ms=now_unix_ms,
                )
                dynamic_encoding = encode_dynamic_targets(
                    dynamic,
                    body_position_world_enu_m=position,
                    body_orientation_world_from_body=orientation,
                    body_velocity_world_enu_mps=velocity,
                    observed_at_unix_ms=frame.observed_at_unix_ms,
                    encoded_at_unix_ms=now_unix_ms,
                )
                if now - last_live_evidence_prune_at >= args.live_evidence_prune_period_seconds:
                    world.prune_live_evidence(
                        center_m=position,
                        radius_m=args.live_evidence_retention_radius_m,
                        now_monotonic_seconds=now,
                        maximum_age_seconds=args.live_evidence_maximum_age_seconds,
                    )
                    last_live_evidence_prune_at = now
                last_depth_processing_failed = False
                if first_accepted_frame_monotonic is None:
                    first_accepted_frame_monotonic = image_received_at
            if latest_realtime_feature_snapshot is not None or not stale_tick:
                if local_input_wake_enabled:
                    # 投影/融合可能耗去一个原生状态周期。几何来源和射线坐标不变，
                    # 但控制位置、速度、姿态与状态编码须共同取实际已经到达的更新，
                    # 不能只给旧编码改时间或只更新模型输入而仍用旧姿态映射速度。
                    control_reference_ms = int(time.time() * 1000)
                    native_observation = native_state_sampler.buffer.latest_after(
                        native_observation, now_unix_ms=control_reference_ms)
                    native_pose = identity_alignment.admit_payload(
                        native_observation.payload, now_unix_ms=control_reference_ms)
                    now_unix_ms, now = control_reference_ms, time.monotonic()
                    native_sample = native_observation.sample
                    latest_flight_state_encoding = native_observation.encoding
                    position = native_pose.position_world_enu_m
                    orientation = native_pose.orientation_world_from_body
                    velocity = native_pose.velocity_world_enu_mps
                    acceleration_world_enu_mps2 = (
                        body_to_world_enu(orientation, native_sample.acceleration_body_mps2)
                        if native_sample.acceleration_body_mps2 is not None else None)
                    visual_motion_alignment = forward_camera_motion_alignment(
                        body_orientation_world_from_body=orientation,
                        body_velocity_world_enu_mps=velocity, mount=mount)
                    # 必须重新计算原深度的年龄，不能让新的控制参考钟给旧几何续期。
                    health = fusion.health(now_unix_ms=now_unix_ms, now_monotonic_seconds=now)
                if stale_tick:
                    # Preserve geometry/track source times while accepting a
                    # newer state encoding. Missing images still age normally.
                    previous_encodings = {
                        encoding.encoder_role: encoding
                        for encoding in latest_realtime_feature_snapshot.encodings
                    }
                    geometry_encoding = previous_encodings["metric-geometry-encoder"]
                    dynamic_encoding = previous_encodings["dynamic-target-encoder"]
                if local_input_wake_enabled:
                    dynamic_encoding = encode_dynamic_targets(
                        dynamic, body_position_world_enu_m=position,
                        body_orientation_world_from_body=orientation,
                        body_velocity_world_enu_mps=velocity,
                        observed_at_unix_ms=frame.observed_at_unix_ms,
                        encoded_at_unix_ms=now_unix_ms)
                latest_realtime_feature_snapshot = fuse_realtime_features(
                    (geometry_encoding, dynamic_encoding, latest_flight_state_encoding),
                    captured_at_unix_ms=now_unix_ms,
                )
            health = _fault_adjusted_perception_health(
                health,
                development_depth_fault_active=fault_active,
            )
            if health.stream_healthy:
                stream_unhealthy_since_monotonic = None
            elif stream_unhealthy_since_monotonic is None:
                stream_unhealthy_since_monotonic = now
            if fault_active and development_fault_activated_at_unix_ms is None:
                development_fault_activated_at_unix_ms = now_unix_ms
            if (
                development_fault_activated_at_unix_ms is not None
                and not fault_active
                and development_fault_recovered_at_unix_ms is None
            ):
                development_fault_recovered_at_unix_ms = now_unix_ms
            if args.development_fault_evidence is not None:
                runtime_snapshot_writer.submit_json(
                    args.development_fault_evidence,
                    {
                        "schema_version": "dronedream.development-depth-fault.v1",
                        "development_only": True,
                        "fault_kind": "depth-frame-drop",
                        "drop_after_first_accepted_frame_seconds": (
                            args.development_depth_drop_after_seconds
                        ),
                        "drop_duration_seconds": (args.development_depth_drop_duration_seconds),
                        "activated": development_fault_activated_at_unix_ms is not None,
                        "recovered": development_fault_recovered_at_unix_ms is not None,
                        "activated_at_unix_ms": development_fault_activated_at_unix_ms,
                        "recovered_at_unix_ms": development_fault_recovered_at_unix_ms,
                    },
                )
            health_payload = {
                    **health.model_dump(mode="json"),
                    "identity_accepted": identity_ok,
                    "identity_issue_codes": identity_issues,
                    "identity_raw_position_disagreement_m": (identity_alignment.raw_disagreement_m),
                    "map_frame_binding_sha256": identity_alignment.binding_sha256,
                    "pose_source": "native-estimator-fixed-deployment-binding",
                    "truth_correction_applied": False,
                    "localization_covariance_m2": native_sample.localization_covariance_m2,
                    "localization_observed_at_unix_ms": native_sample.observed_at_unix_ms,
                    "perception_observed_at_unix_ms": frame.observed_at_unix_ms,
                    "realtime_features_ready": bool(
                        latest_realtime_feature_snapshot is not None
                        and latest_realtime_feature_snapshot.fresh_at(now_unix_ms)
                    ),
                    "physical_velocity_source": (
                        "px4-identity-telemetry" if identity_ok else "unavailable"
                    ),
                    "gazebo_wall_clock_velocity_mps": (wall_clock_velocity.model_dump(mode="json")),
                    "dynamic_track_count": len(dynamic),
                    "dynamic_clearance_receipts": fusion.last_dynamic_clearances,
                    "development_depth_fault_active": fault_active,
                    "development_depth_fault_configured": (
                        args.development_fault_evidence is not None
                    ),
                    "text_model_input_mode": (
                        "structured-metric-state-plus-forward-rgb"
                        if args.rgb_topic
                        else "structured-metric-state-no-image-required"
                    ),
                    "local_navigation_model_enabled": model_navigation is not None,
                    "local_navigation_control_authority_required": (
                        args.require_model_control_authority
                    ),
                    "local_navigation_controller_step_m": (
                        model_navigation.maximum_controller_step_m
                        if model_navigation is not None
                        else None
                    ),
                    "local_navigation_visual_enabled": bool(args.rgb_topic),
                    "local_navigation_visual_frame_age_seconds": (
                        max(0.0, now - rgb_item[1]) if rgb_item is not None else None
                    ),
                    "local_navigation_visual_motion_alignment": (
                        visual_motion_alignment.model_dump(mode="json")
                    ),
                    "local_navigation_model_pending": (
                        model_navigation.request_pending if model_navigation is not None else False
                    ),
                    "multimodal_dataset_recording_enabled": (multimodal_dataset_writer is not None),
                    "multimodal_dataset_recording_issue": (
                        multimodal_dataset_writer.issue
                        if multimodal_dataset_writer is not None
                        else None
                    ),
                    "semantic_supervision_enabled": bool(args.semantic_label_topic),
                    "semantic_supervision_frame_age_seconds": (
                        max(0.0, now - semantic_label_item[1])
                        if semantic_label_item is not None
                        else None
                    ),
                    "last_local_safety_decision_compute_seconds": (last_decision_compute_seconds),
                    "updated_at_unix_ms": now_unix_ms,
                }
            if health_publisher is not None:
                health_publisher.send(health_payload)
            runtime_snapshot_writer.submit_json(args.health, health_payload)
            cycle_timing.mark("feature_encoding_and_health")
            try:
                # Read once, without a separate mounted-filesystem stat and
                # its check/read race. A missing target still prohibits work.
                target_payload = _read_control_target(
                    args.target, now_unix_ms=int(time.time() * 1000), reader=target_reader,
                    receiver=phase_receiver)
            except FileNotFoundError:
                cycle_outcome = "awaiting-target"
                continue
            cycle_timing.mark("target_read")
            route_target = Vector3.model_validate(target_payload["target_position_m"], strict=True)
            navigation_goal = Vector3.model_validate(
                target_payload["navigation_goal_position_m"], strict=True)
            navigation_goal_id = target_payload.get("navigation_goal_id")
            if not isinstance(navigation_goal_id, str) or not navigation_goal_id.strip():
                raise RuntimeError("local safety target has no stable navigation goal identity")
            navigation_goal_id = navigation_goal_id.strip()
            requested_control_profile = target_payload.get("control_profile")
            if requested_control_profile not in {"cruise", "precision"}:
                raise RuntimeError("local safety target control profile is invalid")
            action_checkpoint_goal = target_payload["action_checkpoint_goal"]
            observed_goal_distance_m = math.dist(
                (position.x, position.y, position.z),
                (navigation_goal.x, navigation_goal.y, navigation_goal.z),
            )
            observed_speed_mps = math.sqrt(
                velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
            )
            control_profile = _effective_control_profile(
                requested_profile=requested_control_profile,
                action_checkpoint_goal=action_checkpoint_goal,
                observed_goal_distance_m=observed_goal_distance_m,
                observed_speed_mps=observed_speed_mps,
                maximum_acceleration_mps2=control_vehicle.max_acceleration_mps2,
                precision_latched=(precision_latched_goal_id == navigation_goal_id),
            )
            if control_profile == "precision":
                precision_latched_goal_id = navigation_goal_id
            active_planner_speed_mps = _planner_speed_for_profile(
                profile=control_profile,
                route_speed_limit_mps=args.max_speed,
            )
            (
                active_horizontal_speed_limit_mps,
                active_vertical_speed_limit_mps,
                active_yaw_rate_limit_dps,
            ) = pilot_control_limits_for_profile(
                profile=control_profile,
                route_speed_limit_mps=args.max_speed,
            )
            tracking_recovery_active = target_payload["tracking_recovery_active"]
            progress_recovery_requested = (
                target_payload.get("decision_trigger") == "progress-stalled"
            )
            recovery_episode_id = target_payload.get("recovery_episode_id")
            if not isinstance(recovery_episode_id, str) or not recovery_episode_id:
                recovery_episode_id = None
            target = route_target
            (
                model_snapshot_goal,
                model_goal_revalidation,
                model_snapshot_output_mode,
            ) = _model_navigation_goal_contract(
                route_target=route_target,
                navigation_goal=navigation_goal,
                omitted_coordinate_candidates=args.omit_coordinate_candidates,
                control_output_mode=local_navigation_output_mode,
            )
            model_directive = None
            common_navigation_task = {
                "control_session_id": control_session_id,
                "navigation_goal_id": navigation_goal_id,
                "control_profile": control_profile,
                "local_navigation_output_mode": model_snapshot_output_mode,
                "normalized_pilot_control_limits": {
                    "horizontal_speed_mps": active_planner_speed_mps,
                    "vertical_speed_mps": active_vertical_speed_limit_mps,
                    "yaw_rate_dps": active_yaw_rate_limit_dps,
                },
                "action_checkpoint_goal": action_checkpoint_goal,
                "observed_goal_distance_m": round(observed_goal_distance_m, 3),
                "control_profile_contract": (
                    "continuous body axes with precision speed bounds"
                    if control_profile == "precision"
                    else "bounded route progress with full local revalidation"
                ),
            }
            active_controller_step_m = None
            completed_model_cycle = None
            early_model_poll = False
            if model_navigation is not None:
                active_controller_step_m = _controller_step_for_profile(
                    profile=control_profile,
                    maximum_step_m=model_navigation.maximum_controller_step_m,
                    world_resolution_m=world.resolution_m,
                )
                if local_navigation_output_mode == "normalized-body-velocity":
                    # Continuous completion only consumes an already-done
                    # future; it never waits for inference or runs path search.
                    # Admit it before this tick's safety evaluation instead of
                    # wasting another complete sensor period after it finishes.
                    completed_model_cycle = _poll_ready_navigation_cycle(
                        model_navigation, fusion=fusion, stale_tick=stale_tick,
                        goal=model_goal_revalidation, goal_id=navigation_goal_id,
                        frame=frame, position=position, velocity=velocity,
                    )
                    early_model_poll = True
                # RGB preparation, new invocation and evidence I/O stay after
                # safety publication. Legacy candidate revalidation stays there
                # as well; it has a different, potentially expensive workload.
                model_directive = model_navigation.controller_directive(
                    current_position_m=position,
                    fallback_target_m=route_target,
                    now_unix_ms=now_unix_ms,
                    navigation_goal_id=navigation_goal_id,
                    maximum_step_m=active_controller_step_m,
                )
                target = _navigation_target_for_control_authority(
                    route_target=route_target,
                    current_position=position,
                    model_directive=model_directive,
                    require_model_control_authority=args.require_model_control_authority,
                )
                # Without model-required authority this port is shadow-only:
                # collect its decision and latency evidence, but never let an
                # old candidate/hold lease alter the deterministic route
                # teacher's target.  Earlier behavior mislabeled such flights
                # as route-fallback while still steering toward model output.
            observation = fusion.local_safety_observation(
                target_position_m=target,
                now_unix_ms=now_unix_ms,
                body_orientation_world_from_body=orientation,
                current_acceleration_world_enu_mps2=(
                    acceleration_world_enu_mps2
                ),
            )
            if stale_tick:
                # Pose/velocity identity telemetry remains independently live
                # while the depth frame is stale. Bind the braking command to
                # that current physical state, but retain the stale sensor age
                # and unhealthy verdict from the last accepted depth frame.
                observation = RuntimeLocalSafetyObservation.model_validate(
                    {
                        **observation.model_dump(mode="python"),
                        "current_position_m": position,
                        "current_velocity_mps": velocity,
                    }
                )
            if not identity_ok or not health.stream_healthy:
                observation = observation.model_copy(update={"stream_healthy": False})
            realtime_control_features_ready = bool(
                latest_realtime_feature_snapshot is not None
                and latest_realtime_feature_snapshot.fresh_at(now_unix_ms)
            )
            if (
                args.require_model_control_authority
                and model_directive is not None
                and model_directive.model_navigation_authorized
                and not realtime_control_features_ready
            ):
                observation = observation.model_copy(update={"stream_healthy": False})
            query_center = observation.current_position_m
            query_radius_m = runtime_safety_query_radius_m(
                observation=observation, vehicle=control_vehicle,
                required_clearance_m=args.required_clearance,
                maximum_speed_mps=active_planner_speed_mps,
            )
            perceived = world.local_occupied_box_primitives(
                center_m=query_center,
                radius_m=query_radius_m,
                now_monotonic_seconds=now,
                maximum_age_seconds=0.5,
                limit=96,
            )
            nearby_static = static_geometry_index.nearby(
                (query_center.x, query_center.y, query_center.z), radius_m=query_radius_m
            )
            perceived, duplicate_perception_count = _remove_known_static_perception_duplicates(
                perceived, nearby_static
            )
            static = [*nearby_static, *perceived]
            control_validity_ms = (
                remaining_control_validity_ms(
                    now_unix_ms=now_unix_ms,
                    authority_deadline_unix_ms=model_directive.valid_until_unix_ms,
                    requested_validity_ms=600,
                )
                if model_directive is not None and model_directive.model_navigation_authorized
                else None
            )
            live_model_authority = bool(
                args.require_model_control_authority
                and model_directive is not None
                and model_directive.model_navigation_authorized
                and model_directive.navigation_snapshot_sha256 is not None
                and model_directive.model_call_id is not None
                and model_directive.path_sha256 is not None
                and model_directive.pilot_control is not None
                and model_directive.source_expert in {
                    "local-navigation-policy", "precision-maneuver-policy", "recovery-policy",
                }
                and realtime_control_features_ready
                and control_validity_ms is not None
            )
            if args.require_model_control_authority and not live_model_authority:
                observation = observation.model_copy(update={"stream_healthy": False})
            # 负载回执损坏应在本轮发布运动之前拒绝，而不是先飞一拍再报告错误。
            # 复用本周期同一原生包；后续模型与训练背景不另开遥测文件拼接不同时刻。
            tick_payload_context = _payload_context(
                args.command.parent / "runtime-actions" / "receipts", vehicle,
                identity_telemetry_payload=native_observation.payload)
            command = None
            requested_control_intent = None
            if live_model_authority:
                requested_control_intent = body_control_intent_for_pilot_control(
                    source_expert=model_directive.source_expert,
                    model_call_id=model_directive.model_call_id,
                    navigation_snapshot_sha256=model_directive.navigation_snapshot_sha256,
                    task_reference_sha256=model_directive.path_sha256,
                    pilot_control=model_directive.pilot_control,
                    harness_control_scale=model_directive.controller_step_scale,
                    generated_at_unix_ms=now_unix_ms,
                    maximum_horizontal_speed_mps=active_planner_speed_mps,
                    maximum_vertical_speed_mps=active_vertical_speed_limit_mps,
                    maximum_yaw_rate_dps=active_yaw_rate_limit_dps,
                    maximum_acceleration_mps2=control_vehicle.max_acceleration_mps2,
                    maximum_jerk_mps3=max(
                        1.0,
                        control_vehicle.max_acceleration_mps2 * 5.0,
                    ),
                    validity_milliseconds=control_validity_ms,
                )
            decision_compute_seconds = None
            cycle_timing.mark("context_and_collision_inputs")
            stale_hold_elapsed_seconds = (
                now - stream_unhealthy_since_monotonic
                if stream_unhealthy_since_monotonic is not None
                else 0.0
            )
            if identity_ok and (health.stream_healthy or stale_hold_elapsed_seconds <= 2.0):
                decision_started_at = time.monotonic()
                evaluated = evaluate_runtime_local_safety(
                    observation=observation,
                    vehicle=control_vehicle,
                    static_primitives=static,
                    required_clearance_m=args.required_clearance,
                    generated_at_unix_ms=now_unix_ms,
                    command_horizon_seconds=0.2,
                    # Keep authority short-lived even when Gazebo renders the
                    # full School Map below real time.  Any lease gap causes a
                    # fixed-position hold and then a bounded landing.
                    validity_milliseconds=600,
                    maximum_speed_mps=active_planner_speed_mps,
                    tracking_recovery_active=tracking_recovery_active,
                    navigation_goal_id=navigation_goal_id,
                    navigation_control_authority=(
                        "model-required"
                        if args.require_model_control_authority
                        else "route-fallback"
                    ),
                    model_navigation_authorized=live_model_authority,
                    model_navigation_snapshot_sha256=(
                        model_directive.navigation_snapshot_sha256
                        if live_model_authority else None
                    ),
                    model_call_id=(
                        _safety_hold_call_id(model_directive, completed_model_cycle)
                        if args.require_model_control_authority
                        else None
                    ),
                    model_selected_candidate_id=(
                        model_directive.selected_candidate_id
                        if args.require_model_control_authority
                        and model_directive is not None
                        and model_directive.model_navigation_authorized
                        else None
                    ),
                    model_path_sha256=(
                        model_directive.path_sha256
                        if args.require_model_control_authority
                        and model_directive is not None
                        and model_directive.model_navigation_authorized
                        else None
                    ),
                    model_authority_reason=(
                        model_directive.reason
                        if model_directive is not None
                        else "model-navigation-disabled"
                    ),
                    requested_control_intent=requested_control_intent,
                    route_yaw_rate_dps=(teacher_heading_rate(
                        orientation=orientation, position=position, goal=navigation_goal,
                        maximum_rate_dps=min(45., active_yaw_rate_limit_dps),
                    ) if args.simulation_teacher_control else 0.),
                )
                decision_compute_seconds = time.monotonic() - decision_started_at
                last_decision_compute_seconds = decision_compute_seconds
                published_at_unix_ms = int(time.time() * 1_000)
                if (
                    requested_control_intent is not None
                    and _model_control_dispatch_unavailable(
                        published_at_unix_ms=published_at_unix_ms,
                        authority_deadline_unix_ms=requested_control_intent.valid_until_unix_ms)
                ):
                    # A model/body request that loses its dispatch budget while
                    # evaluation is running must produce a braking hold, not a
                    # zero-lifetime/unusable motion command. The second
                    # evaluation takes the planner's constant-time unhealthy
                    # path and preserves the expired lease in evidence without
                    # granting it motion authority.
                    expired_observation = _expired_model_control_observation(
                        observation,
                        published_at_unix_ms=published_at_unix_ms,
                    )
                    evaluated = evaluate_runtime_local_safety(
                        observation=expired_observation,
                        vehicle=control_vehicle,
                        static_primitives=static,
                        required_clearance_m=args.required_clearance,
                        generated_at_unix_ms=published_at_unix_ms,
                        command_horizon_seconds=0.2,
                        validity_milliseconds=600,
                        maximum_speed_mps=active_planner_speed_mps,
                        tracking_recovery_active=tracking_recovery_active,
                        navigation_goal_id=navigation_goal_id,
                        navigation_control_authority="model-required",
                        model_navigation_authorized=False,
                        model_call_id=model_directive.model_call_id,
                        model_selected_candidate_id=(
                            model_directive.selected_candidate_id
                        ),
                        model_path_sha256=model_directive.path_sha256,
                        model_authority_reason=(
                            "control-intent-dispatch-budget-exhausted"
                        ),
                        requested_control_intent=None,
                    )
                    observation = expired_observation
                # Revalidate instead of model_copy(update=...).  Pydantic's
                # model_copy deliberately skips validators, which previously
                # allowed an estimator/world correction above the 1 m contract
                # limit to be serialized after an identity divergence.
                command = prepare_safety_publication(
                    evaluated, published_at_unix_ms, estimator_offset,
                )
                if args.simulation_teacher_control:
                    # Unlike the historical route collector, a continuous
                    # teacher cannot renew old sensor data after slow planning.
                    deadline = teacher_input_deadline(
                        latest_realtime_feature_snapshot, now_ms=published_at_unix_ms,
                    )
                    if command is None or deadline <= published_at_unix_ms:
                        command = None
                    else:
                        command = RuntimeLocalSafetyCommand.model_validate({
                            **command.model_dump(mode="json"),
                            "valid_until_unix_ms": min(deadline, command.valid_until_unix_ms),
                        })
            # Compute and validate the command before publishing either side
            # of the hash-bound pair.  Publishing the observation first and
            # then running candidate evaluation left a long mismatch window
            # under full School Map load, so the 20 Hz executor could miss
            # every otherwise-valid lease.  The executor's double-read makes
            # this short adjacent-write window safe.
            if safety_publisher is not None:
                # Publish the hash-bound pair in one bounded local datagram.
                # Durable evidence is asynchronous; it never grants authority.
                if command is not None:
                    safety_publisher.send(observation, command)
                runtime_snapshot_writer.submit_json(args.observation,
                                                    observation.model_dump(mode="json"))
                if command is not None:
                    runtime_snapshot_writer.submit_json(
                        args.command, command.model_dump(mode="json"),
                    )
            else:
                _atomic_json(args.observation, observation.model_dump(mode="json"))
                if command is not None:
                    _atomic_json(args.command, command.model_dump(mode="json"))
            cycle_timing.mark("safety_evaluation_and_publication")
            # One native packet and one set of attachment receipts per tick.
            # Data capture and model context must not reopen a mutable telemetry
            # file and silently describe a different moment than the encoders.
            try:
                # Model orchestration stays below independent safety publication,
                # but precedes optional recording and diagnostic serialization.
                # Although real provider
                # calls run in their own executor, completion processing, RGB PNG
                # encoding and evidence I/O can still take hundreds of
                # milliseconds on the full School Map. Keeping them after command
                # publication prevents a recovery-mode transition from consuming
                # the fixed-position stale-command grace window.
                if model_navigation is not None:
                    # Keep a completed response pending while depth is stale. The
                    # active path and command leases still expire and cause a hold;
                    # only completion consumption waits for a fresh frame so the
                    # second path validation is based on current metric evidence.
                    receipt = (
                        completed_model_cycle if early_model_poll else _poll_ready_navigation_cycle(
                            model_navigation, fusion=fusion, stale_tick=stale_tick,
                            goal=model_goal_revalidation, goal_id=navigation_goal_id,
                            frame=frame, position=position, velocity=velocity,
                        )
                    )
                    if receipt is not None:
                        runtime_evidence_writer.submit(
                            model_navigation_evidence,
                            {
                                **receipt.model_dump(mode="json"),
                                "recorded_at_unix_ms": now_unix_ms,
                            },
                        )
                        submitted_snapshot = model_navigation.pop_submitted_snapshot()
                        if submitted_snapshot is not None:
                            runtime_evidence_writer.submit(
                                model_navigation_snapshot_evidence,
                                {
                                    "recorded_at_unix_ms": now_unix_ms,
                                    "navigation_goal_id": _submitted_snapshot_goal_id(
                                        submitted_snapshot,
                                        fallback_goal_id=navigation_goal_id,
                                    ),
                                    "snapshot": submitted_snapshot,
                                },
                            )
                    _drain_model_calls(model_navigation, runtime_evidence_writer,
                                       model_navigation_call_evidence)
                    target_sha256 = sha256_json(
                        {
                            "navigation_goal_id": navigation_goal_id,
                            "navigation_goal_position_m": navigation_goal,
                            "control_profile": control_profile,
                            "action_checkpoint_goal": action_checkpoint_goal,
                        }
                    )
                    dynamic_signature = sha256_json(
                        [
                            {
                                "obstacle_id": obstacle.obstacle_id,
                                "position_m": obstacle.position_m,
                                "velocity_mps": obstacle.velocity_mps,
                            }
                            for obstacle in dynamic
                        ]
                    )
                    dynamic_recovery_episode = _advance_dynamic_recovery_episode(
                        state=dynamic_recovery_episode,
                        navigation_goal_id=navigation_goal_id,
                        dynamic_signature=dynamic_signature,
                        dynamic_present=bool(dynamic),
                        now_monotonic=now,
                    )
                    phase_context = phase_observer.latest()
                    # Projection/fusion can occupy most of a sensor period. Check
                    # the invocation timer here, not with the old loop-entry clock.
                    model_now_monotonic = time.monotonic()
                    trigger = _model_cycle_trigger(
                        model_cycle_started=model_cycle_started,
                        now_monotonic=model_now_monotonic,
                        next_model_cycle_monotonic=next_model_cycle,
                        target_changed=target_sha256 != last_model_target_sha256,
                        dynamic_obstacle_changed=(
                            dynamic_signature != last_dynamic_signature and bool(dynamic)
                        ),
                        progress_recovery_requested=progress_recovery_requested,
                        executor_phase=str(phase_context["executor_phase"]),
                    )
                    with lock:
                        waiting_image_time = latest_image[1] if latest_image is not None else None
                    input_work_allowed = model_input_work_allowed(
                        continuous_control=(local_navigation_output_mode
                                            == "normalized-body-velocity"),
                        handoff_tick=prioritize_ready_control,
                        new_depth_frame=not stale_tick,
                        newer_depth_waiting=bool(waiting_image_time is not None
                            and waiting_image_time > last_image_received_at),
                    )
                    if (trigger is not None and input_work_allowed
                            and not model_navigation.request_pending) and (
                        not stale_tick or local_navigation_output_mode == "normalized-body-velocity"
                        and health.stream_healthy
                    ):
                        model_timing = PhaseTimings()
                        active_recovery_episode_id = None
                        if trigger == "progress-stalled":
                            active_recovery_episode_id = recovery_episode_id
                        elif trigger == "dynamic-obstacle":
                            active_recovery_episode_id = (
                                str(dynamic_recovery_episode["episode_id"])
                                if dynamic_recovery_episode is not None
                                else None
                            )
                        multimodal = None
                        visual_ready = not args.rgb_topic
                        visual_now = time.monotonic()
                        if model_image_worker is not None:
                            # Encoding may finish while metric perception runs.
                            # Select again without waiting, and bind registry
                            # quality/clock to precisely these completed pixels.
                            prepared_rgb_sample = model_image_worker.latest(
                                now_monotonic_seconds=visual_now, maximum_age_seconds=.2)
                            rgb_item = ((prepared_rgb_sample.message,
                                         prepared_rgb_sample.image.received_monotonic_seconds)
                                        if prepared_rgb_sample is not None else None)
                            rgb_received_at_unix_ms = (
                                prepared_rgb_sample.image.received_at_unix_ms
                                if prepared_rgb_sample is not None else None)
                            if prepared_rgb_sample is not None:
                                rgb_binding.register(
                                    rgb_item[0], source_monotonic=rgb_item[1],
                                    source_unix_ms=rgb_received_at_unix_ms,
                                    frame_time=prepared_rgb_sample.image.frame_time,
                                    now_monotonic=visual_now)
                        if (
                            args.rgb_topic
                            and rgb_item is not None
                            and 0 <= visual_now - rgb_item[1] <= (
                                0.2 if local_navigation_output_mode == "normalized-body-velocity"
                                else 0.5
                            )
                            and forward_rgb_can_inform_control(
                                visual_motion_alignment,
                                continuous_control=(local_navigation_output_mode
                                                    == "normalized-body-velocity"),
                            )
                        ):
                            try:
                                image_size = model_navigation_visual_size
                                if image_size is None:
                                    # Cloud planner/observer imagery has no local
                                    # encoder dimensions. Preserve aspect ratio.
                                    width, height = int(rgb_item[0].width), int(rgb_item[0].height)
                                    scale = min(1., 384. / max(1, width, height))
                                    image_size = (max(1, round(width * scale)),
                                                  max(1, round(height * scale)))
                                if model_image_worker is not None:
                                    if prepared_rgb_sample is None:
                                        raise ValueError("MODEL_RGB_PREPARED_SOURCE_MISSING")
                                    prepared = prepared_rgb_sample.image
                                    new_image = (prepared.received_monotonic_seconds
                                                 != last_model_image_time)
                                else:
                                    prepared, new_image = model_image_cache.prepare(
                                        rgb_item[0], received_monotonic_seconds=rgb_item[1],
                                        received_at_unix_ms=rgb_received_at_unix_ms,
                                        size=image_size, decoder=_gazebo_image_model_payload,
                                        frame_time=image_ingress.lookup("rgb", rgb_item[1]),
                                    )
                                image_payload = prepared.multimodal(model_navigation_frame_dir)
                                frame_path = Path(image_payload["path"])
                                if new_image:
                                    _persist_navigation_image(
                                        provider=args.local_navigation_provider,
                                        writer=runtime_snapshot_writer,
                                        path=frame_path, png=prepared.png,
                                    )
                                    last_model_image_time = prepared.received_monotonic_seconds
                                multimodal = [image_payload]
                                # schedule() validates source age BEFORE starting
                                # the visual encoder. Do not prefetch around that gate.
                                visual_ready = True
                            except (OSError, ValueError):
                                visual_ready = False
                        if visual_ready:
                            model_timing.mark("visual_preparation")
                            strategic_sensor_snapshot = fusion.multimodal_sensor_snapshot(
                                now_monotonic_seconds=visual_now
                            )
                            if strategic_sensor_snapshot is None:
                                raise RuntimeError("MULTIMODAL_SENSOR_REGISTRY_UNAVAILABLE")
                            payload_context = tick_payload_context
                            strategic_context = build_navigation_context(
                                task={**phase_context, **common_navigation_task,
                                      "decision_trigger": trigger,
                                      "recovery_episode_id": active_recovery_episode_id},
                                map_context=strategic_map_context,
                                sensor_context=_strategic_sensor_context(strategic_sensor_snapshot),
                                vehicle=vehicle, payload=payload_context,
                                rgb_enabled=bool(args.rgb_topic),
                            )
                            model_timing.mark("strategic_context")
                            schedule_now_unix_ms = now_unix_ms
                            schedule_features = latest_realtime_feature_snapshot
                            if (local_navigation_output_mode == "normalized-body-velocity"
                                    and schedule_features is not None):
                                # Depth projection/fusion and diagnostics may have
                                # consumed much of the first state sample's lease.
                                # Read an actual newer native encoding, not a newer
                                # timestamp on old state or geometry. Safety above
                                # retains its own original observation and deadline.
                                schedule_now_unix_ms = int(time.time() * 1000)
                                scheduled_native = native_state_sampler.buffer.latest(
                                    now_unix_ms=schedule_now_unix_ms)
                                schedule_features = refresh_flight_state_features(
                                    schedule_features, scheduled_native.encoding,
                                    captured_at_unix_ms=schedule_now_unix_ms,
                                )
                            immediate = model_navigation.schedule(
                                goal_position_m=model_snapshot_goal,
                                navigation_goal_id=navigation_goal_id,
                                now_unix_ms=schedule_now_unix_ms,
                                trigger=trigger,
                                context_id=args.local_navigation_context_id,
                                multimodal=multimodal,
                                strategic_context=strategic_context,
                                realtime_feature_snapshot=(
                                    schedule_features.model_dump(mode="json")
                                    if schedule_features is not None
                                    else None
                                ),
                            )
                            model_timing.mark("schedule_and_freeze")
                            runtime_evidence_writer.submit(
                                args.command.with_name("model-navigation-timing.jsonl"),
                                {"recorded_at_unix_ms": now_unix_ms,
                                 "scheduled_at_unix_ms": int(time.time() * 1000),
                                 "phase_ms": model_timing.snapshot()},
                            )
                            if immediate is not None:
                                runtime_evidence_writer.submit(
                                    model_navigation_evidence,
                                    {
                                        **immediate.model_dump(mode="json"),
                                        "recorded_at_unix_ms": now_unix_ms,
                                    },
                                )
                                recorded_snapshot = model_navigation.pop_submitted_snapshot()
                                if recorded_snapshot is not None:
                                    runtime_evidence_writer.submit(
                                        model_navigation_snapshot_evidence,
                                        {
                                            "recorded_at_unix_ms": now_unix_ms,
                                            "navigation_goal_id": _submitted_snapshot_goal_id(
                                                recorded_snapshot,
                                                fallback_goal_id=navigation_goal_id,
                                            ),
                                            "snapshot": recorded_snapshot,
                                        },
                                    )
                            model_cycle_started = True
                            next_model_cycle = (
                                model_now_monotonic + args.local_navigation_period_seconds)
                            last_model_target_sha256 = target_sha256
                            last_dynamic_signature = dynamic_signature
            finally:
                # Record the already-published control even if model
                # preparation fails. No frame, timestamp or authority is
                # refreshed by moving non-control work after submission.
                cycle_timing.mark("model_scheduling")
                try:
                    if learning_recorder is not None:
                        learning_recorder.poll()
                        learning_sensors = fusion.multimodal_sensor_snapshot(
                            now_monotonic_seconds=now)
                        learning_visual_ready = not args.rgb_topic or (
                            rgb_item is not None and rgb_received_at_unix_ms is not None
                            and _learning_visual_current(source_monotonic=rgb_item[1],
                                source_unix_ms=rgb_received_at_unix_ms,
                                reference_monotonic=now, reference_unix_ms=now_unix_ms)
                        )
                        if (learning_recorder.ready_for_submission
                                and command is not None
                                and latest_realtime_feature_snapshot is not None
                                and health.stream_healthy and learning_sensors is not None
                                and learning_visual_ready):
                            learning_context = build_navigation_context(
                                task={**phase_observer.latest(),
                                      **common_navigation_task,
                                      "decision_trigger": target_payload.get(
                                          "decision_trigger", "periodic"),
                                      "recovery_episode_id": recovery_episode_id},
                                map_context=strategic_map_context,
                                sensor_context=_strategic_sensor_context(learning_sensors),
                                vehicle=vehicle, rgb_enabled=bool(args.rgb_topic),
                                payload=tick_payload_context,
                            )
                            learning_recorder.submit(NavigationSnapshotRequest(
                                world=world.navigation_clone(center_m=position, radius_m=12.),
                                frame=frame.model_copy(update={
                                    "localization_position_m": position,
                                    "localization_velocity_mps": velocity}, deep=True),
                                health=health, goal_position_m=navigation_goal,
                                required_clearance_m=args.required_clearance,
                                candidate_speed_mps=active_planner_speed_mps,
                                vehicle_radius_m=vehicle.body_radius_m,
                                vehicle_height_m=vehicle.body_height_m,
                                visual_evidence=[],
                                multimodal_sensor_snapshot=learning_sensors.model_dump(mode="json"),
                                realtime_feature_snapshot=latest_realtime_feature_snapshot.model_dump(
                                    mode="json"),
                                strategic_context=learning_context,
                                    maximum_snapshot_planning_seconds=.05,
                                    include_candidate_paths=False,
                                control_reference_observed_at_unix_ms=now_unix_ms,
                            ), command, prepare_visual=(partial(
                                _prepare_learning_visual, cache=learning_image_cache,
                                image=rgb_item[0], received_monotonic=rgb_item[1],
                                received_utc_ms=rgb_received_at_unix_ms,
                                frame_time=(prepared_rgb_sample.image.frame_time
                                    if prepared_rgb_sample is not None
                                    else image_ingress.lookup("rgb", rgb_item[1])),
                                prepared_sample=prepared_rgb_sample,
                                size=tuple(args.learning_image_size),
                                directory=args.command.parent / "learning-observation-frames",
                            ) if args.rgb_topic and rgb_item is not None else None))
                    synchronized_sensor_pair = (
                        None
                        if args.semantic_label_topic and not semantic_label_items
                        else _nearest_synchronized_sensor_pair(
                            rgb_items,
                            semantic_label_items if args.semantic_label_topic else (),
                            now_monotonic_seconds=now,
                            newest_rgb_after_monotonic_seconds=(last_dataset_rgb_received_at),
                        )
                    )
                    if (
                        multimodal_dataset_writer is not None
                        and multimodal_dataset_writer.issue is None
                        and synchronized_sensor_pair is not None
                    ):
                        dataset_rgb_item, dataset_semantic_item = synchronized_sensor_pair
                        sensor_snapshot = fusion.multimodal_sensor_snapshot(
                            now_monotonic_seconds=now)
                        # Bind the latest validated PX4 dynamics to the same record as
                        # RGB, depth, pose, velocity, and the active navigation goal.
                        # Keeping this summary in the hash-chained record avoids a
                        # later training job having to join an unrelated, mutable
                        # telemetry side file by timestamp.  _payload_context applies
                        # the same freshness and finite-value checks used by the local
                        # payload expert; unavailable or stale telemetry remains
                        # explicit instead of being fabricated.
                        dataset_payload_context = tick_payload_context
                        if sensor_snapshot is not None and multimodal_dataset_writer.submit(
                            rgb_image=dataset_rgb_item[0],
                            frame=frame,
                            sensor_snapshot=sensor_snapshot,
                            recorded_at_unix_ms=now_unix_ms,
                            recorded_at_monotonic_seconds=now,
                            rgb_sample_monotonic_seconds=dataset_rgb_item[1],
                            frame_time=image_ingress.lookup("rgb", dataset_rgb_item[1]),
                            semantic_image=(
                                dataset_semantic_item[0]
                                if dataset_semantic_item is not None else None
                            ),
                            semantic_sample_monotonic_seconds=(
                                dataset_semantic_item[1]
                                if dataset_semantic_item is not None else None
                            ),
                            state={
                                "current_position_m": position.model_dump(mode="json"),
                                "current_velocity_mps": velocity.model_dump(mode="json"),
                                "navigation_goal_id": navigation_goal_id,
                                "navigation_goal_m": navigation_goal.model_dump(mode="json"),
                                "control_profile": control_profile,
                                "dynamic_obstacles": [
                                    obstacle.model_dump(mode="json") for obstacle in dynamic
                                ],
                                "identity_accepted": identity_ok,
                                "forward_camera_motion_alignment": (
                                    visual_motion_alignment.model_dump(mode="json")
                                ),
                                "realtime_feature_snapshot": (
                                    latest_realtime_feature_snapshot.model_dump(mode="json")
                                    if latest_realtime_feature_snapshot is not None
                                    else None
                                ),
                                "payload": dataset_payload_context,
                            },
                        ):
                            last_dataset_rgb_received_at = dataset_rgb_item[1]
                    cycle_timing.mark("learning_and_media_submission")
                finally:
                    runtime_evidence_writer.submit(
                        local_safety_history,
                        {
                            "recorded_at_unix_ms": now_unix_ms,
                            "identity_accepted": identity_ok,
                            "identity_issue_codes": identity_issues,
                            "tracking_recovery_active": tracking_recovery_active,
                            "decision_compute_seconds": decision_compute_seconds,
                            "pipeline_phase_ms": cycle_timing.snapshot(),
                            "depth_input_timing": depth_input_timing,
                            "previous_completed_tick_timing": previous_tick_timing,
                            "prioritized_ready_control": prioritize_ready_control,
                            "sensor_arrival_wake": (
                                sensor_arrival_due and not prioritize_ready_control),
                            "route_target_m": route_target.model_dump(mode="json"),
                            "effective_controller_target_m": target.model_dump(mode="json"),
                            "observation": observation.model_dump(mode="json"),
                            "command": (
                                command.model_dump(mode="json") if command is not None else None),
                            "realtime_feature_snapshot": (
                                latest_realtime_feature_snapshot.model_dump(mode="json")
                                if latest_realtime_feature_snapshot is not None
                                else None
                            ),
                        },
                    )
            runtime_snapshot_writer.submit_json(
                args.command.with_name("metric-local-world-summary.json"),
                {
                    # This is emitted on the sensor-rate safety path, so it
                    # must stay O(1).  Full frontier extraction is reserved for
                    # explicit planning/model cycles via text_map_summary().
                    **world.runtime_evidence_summary(),
                    "local_perception_primitive_count": len(perceived),
                    "filtered_static_perception_duplicate_count": (duplicate_perception_count),
                    "local_semantic_primitive_count": len(static) - len(perceived),
                    "observation_sha256": sha256_json(observation),
                    "route_target_m": route_target.model_dump(mode="json"),
                    "navigation_goal_m": navigation_goal.model_dump(mode="json"),
                    "navigation_goal_id": navigation_goal_id,
                    "control_profile": control_profile,
                    "requested_control_profile": requested_control_profile,
                    "precision_latched": precision_latched_goal_id == navigation_goal_id,
                    "action_checkpoint_goal": action_checkpoint_goal,
                    "observed_goal_distance_m": observed_goal_distance_m,
                    "observed_speed_mps": observed_speed_mps,
                    "effective_controller_target_m": target.model_dump(mode="json"),
                    "local_navigation_model_enabled": model_navigation is not None,
                    "local_navigation_controller_step_m": (active_controller_step_m),
                    "local_navigation_visual_enabled": bool(args.rgb_topic),
                    "local_navigation_model_pending": (
                        model_navigation.request_pending if model_navigation is not None else False
                    ),
                    "multimodal_sensor_snapshot": (
                        fusion.multimodal_sensor_snapshot(now_monotonic_seconds=now).model_dump(
                            mode="json"
                        )
                    ),
                    "realtime_feature_snapshot": (
                        latest_realtime_feature_snapshot.model_dump(mode="json")
                        if latest_realtime_feature_snapshot is not None
                        else None
                    ),
                    "live_evidence_retention_radius_m": (args.live_evidence_retention_radius_m),
                    "live_evidence_maximum_age_seconds": (args.live_evidence_maximum_age_seconds),
                    "live_evidence_prune_period_seconds": (args.live_evidence_prune_period_seconds),
                },
            )
            cycle_timing.mark("control_evidence_and_world_summary")
            previous_tick_timing = {
                "recorded_at_unix_ms": now_unix_ms,
                "depth_sequence": sequence,
                "new_depth_frame": new_depth_frame,
                "phase_ms": cycle_timing.snapshot(),
            }
            cycle_outcome = "control"
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            error_kind = f"{type(error).__name__}:{error}"
            log_key = error_kind.splitlines()[0]
            if log_key not in logged_error_kinds and len(logged_error_kinds) < 32:
                # Keep the original failure location, not just the last
                # downstream symptom, with bounded per-process log volume.
                logged_error_kinds.add(log_key)
                traceback.print_exception(type(error), error, error.__traceback__)
            failed_health = {
                    "schema_version": "dronedream.perception-fusion-health.v1",
                    "stream_healthy": False,
                    "issue_codes": [error_kind],
                    "updated_at_unix_ms": now_unix_ms,
                }
            if health_publisher is not None:
                health_publisher.send(failed_health)
            # 异常期间也不得让挂载盘写入阻塞采样恢复，证据仍由有界写入器落盘。
            runtime_snapshot_writer.submit_json(args.health, failed_health)
        finally:
            # 起飞前无任务目标或校验拒绝也必须计时，不能只留下成功控制周期的耗时。
            # 常量空间内累计，磁盘发布推迟至关闭，不在关键路径增加同步 I/O。
            cycle_timing.mark("cycle_tail")
            perception_timing_summary.record(cycle_timing, outcome=cycle_outcome)
    subscription_summary = subscriptions.close()
    _atomic_json(args.command.with_name("perception-phase-timing-summary.json"),
                 perception_timing_summary.snapshot())
    _atomic_json(args.command.with_name("sensor-subscription-shutdown.json"), subscription_summary)
    _atomic_json(args.command.with_name("sensor-interpreter-pauses.json"),
                 close_pause_monitor())
    close_native_sampler()
    _atomic_json(args.command.with_name("native-state-sampler-summary.json"),
                 native_state_sampler.summary())
    geometry_summary = geometry_capture.close() if geometry_capture is not None else None
    if close_model_image_worker is not None:
        _atomic_json(args.command.with_name("model-rgb-worker-summary.json"),
                     close_model_image_worker())
    if close_safety_publisher is not None:
        close_safety_publisher()
    if close_health_publisher is not None:
        close_health_publisher()
    if close_model_navigation is not None:
        close_model_navigation()
        _drain_model_calls(model_navigation, runtime_evidence_writer,
                           model_navigation_call_evidence)
    model_shutdown_complete = model_navigation is None or not model_navigation.request_pending
    _atomic_json(args.command.with_name("model-navigation-shutdown.json"),
                 {"complete": model_shutdown_complete,
                  "pending_call": not model_shutdown_complete})
    learning_summary = None
    if close_learning_recorder is not None:
        learning_summary = close_learning_recorder()
        _atomic_json(args.command.with_name("learning-observation-summary.json"),
                     learning_summary)
    runtime_snapshot_summary = close_snapshot_writer()
    runtime_evidence_summary = close_evidence_writer()
    dataset_summary = None
    if close_dataset_writer is not None:
        # The parent runtime ends this long-lived worker with SIGTERM. Drain a
        # bounded final sample and publish the final summary before exiting so
        # records.jsonl and summary.json cannot disagree by one committed row.
        dataset_summary = close_dataset_writer()
    return (
        0
        if subscription_summary["complete"] is True
        and model_shutdown_complete
        and runtime_evidence_summary.get("complete") is True
        and runtime_snapshot_summary.get("complete") is True
        and (learning_summary is None or learning_summary.get("complete") is True)
        and (geometry_summary is None or geometry_summary.get("complete") is True)
        and (dataset_summary is None or dataset_summary.get("writer_complete") is True)
        else 2
    )


if __name__ == "__main__":
    # This is an exec-created process. Retain only imported startup objects;
    # all sensor/runtime objects are created inside main and remain collectible.
    with retained_interpreter_baseline():
        raise SystemExit(main())
