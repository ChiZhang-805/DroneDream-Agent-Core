"""Real PX4 SITL + Gazebo runner for an arbitrary validated track contract."""

from __future__ import annotations

import binascii
import csv
import heapq
import json
import math
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from contextlib import suppress
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .collision import (
    PREFERRED_TRANSIT_CLEARANCE_M,
    _clearance,  # same conservative envelope used by static gate
    _validated_primitive,
    build_tracking_corridor_budget,
    primitive_bounds,
)
from .contracts import (
    DynamicObstacleObservation,
    GraphRoute,
    LocalPlannerRequest,
    Px4GazeboModelControlAuthorityEvidence,
    Px4GazeboRunEvidence,
    Px4RuntimeAbortRequest,
    Px4Track,
    RouteClearanceReport,
    RuntimeActionExecutionContract,
    RuntimeCheckpointContract,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from .control_execution_evidence import verify_control_applications
from .control_timing import (
    LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
    continuous_control_cadence_bounded,
    continuous_control_evidence_required,
    continuous_timing_is_bounded,
)
from .dynamic_safety import predictive_safety_decision
from .gazebo_subscriptions import GazeboSubscriptions, subscription_shutdown_is_complete
from .hashing import sha256_json
from .identity_alignment import (
    align_gazebo_px4_identity,
    dynamic_identity_disagreement_limit_m,
    identity_offset_innovation_m,
)
from .localization_truth_capture import LocalizationTruthCapture
from .native_process_diagnostics import NativeSourceStallProbe
from .perception_lifecycle import verify_control_completion
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .preflight_render_warmup import warmup_receipt_ready
from .preview_worker import LatestPreviewWorker
from .runtime_bindings import load_map_runtime_bindings, resolve_vehicle_collision_center_offset
from .runtime_control_io import publish_runtime_json, read_runtime_object
from .runtime_scheduling import InterpreterPauseMonitor, SimulationSourceIntervals
from .simulation_camera_profile import prepare_camera_profile, validate_camera_profile_choice
from .simulation_dynamic_obstacles import (
    dynamic_observation,
    dynamic_positions,
    load_dynamic_geometry,
    simulation_velocity,
    witness_age,
)
from .simulation_graphics_evidence import graphics_request_evidence
from .simulation_graphics_lifetime import prepare_render_process_environment
from .simulation_phase_monitor import SimulationPhaseMonitor
from .simulation_render_cache import (
    finalize_render_cache,
    prepare_render_cache,
    verify_render_preparation,
)
from .simulation_render_replica import prepare_render_replica
from .simulation_sensor_frames import (
    collision_center_from_canonical,
    inspect_simulation_sensor_frames,
    select_canonical_poses,
    simulation_pose_time_ns,
)
from .simulation_sensor_runtime import prepare_sensor_runtime
from .static_render_batching import prepare_static_render_world
from .training.gazebo_witness import GazeboOutcomeWitness
from .training.outcome_channel import OutcomePublisher
from .training.outcome_channel import descriptor_path as outcome_descriptor_path


class SimulationRuntimeError(RuntimeError):
    """Real runtime could not satisfy an execution or evidence gate."""


SIMULATION_TAKEOFF_STABILITY_TIMEOUT_SECONDS = 180.0
_RUN_LOCK = threading.Lock()


# 功能：
#   拒绝布尔、非有限和越界数字，避免延迟或距离配置绕过比较门控。
# 输入：
#   value：配置值；name：诊断字段名；minimum、maximum：允许范围。
# 输出：
#   value：通过验证的原数值。
def _runtime_number(value, name: str, minimum=0.0, maximum=1e9):
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"runtime {name} is outside its finite numeric range")
    return value


# 功能：
#   严格解析当前本地运行合同，不接受重复键、非有限数或类型强转。
# 输入：
#   path：合同文件；model：预期合同类型。
# 输出：
#   artifact：重新验证的合同。
def _runtime_contract(path: Path, model):
    raw = decode_json(
        read_plugin_file(path, limit=64 * 1024**2), limit=64 * 1024**2, node_limit=2_000_000
    )
    artifact = model.model_validate_json(
        encode_json(raw, limit=64 * 1024**2, node_limit=2_000_000), strict=True
    )
    return artifact


# 功能：
#   有界读取完整证据流，尾部损坏不能留下可被误用的成功前缀。
# 输入：
#   path：当前运行所有的 JSONL 文件。
# 输出：
#   records：完整对象列表；从未创建的可选流为空列表。
def _runtime_rows(path: Path) -> list[dict[str, Any]]:
    check_plain_plugin_path(path)
    try:
        before = path.stat()
    except FileNotFoundError:
        return []
    if not stat.S_ISREG(before.st_mode) or before.st_size > 512 * 1024**2:
        raise ValueError("RUNTIME_EVIDENCE_FILE_INVALID")
    records = []
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not os.path.samestat(before, opened):
            raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
        total = 0
        while line := stream.readline(16 * 1024**2 + 1):
            total += len(line)
            if total > opened.st_size or len(records) >= 1_000_000 or not line.endswith(b"\n"):
                raise ValueError("RUNTIME_EVIDENCE_TRUNCATED_OR_OVERSIZED")
            record = decode_json(line, limit=16 * 1024**2)
            if not isinstance(record, dict):
                raise ValueError("RUNTIME_EVIDENCE_RECORD_NOT_OBJECT")
            records.append(record)
        after = os.fstat(stream.fileno())
    check_plain_plugin_path(path)
    current = path.stat()
    if (
        not os.path.samestat(after, current)
        or total != opened.st_size
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or current.st_size != after.st_size
        or current.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
    return records


# 功能：
#   将一次仿真的传输环境变更限定在调用期，并拒绝同解释器并发实例串线。
# 输入：
#   function：真实运行入口。
# 输出：
#   isolated：带环境恢复的入口函数。
def _isolate_transport_environment(function):
    # 功能：
    #   即使导入 Gazebo 或构造节点失败也恢复原环境，互斥锁不代表跨进程隔离。
    # 输入：
    #   args、kwargs：原入口参数。
    # 输出：
    #   result：原入口的运行结果。
    @wraps(function)
    def isolated(*args, **kwargs):
        if not _RUN_LOCK.acquire(blocking=False):
            raise SimulationRuntimeError("SIMULATION_ALREADY_RUNNING_IN_THIS_PROCESS")
        previous = {key: os.environ.get(key) for key in ("GZ_PARTITION", "GZ_IP")}
        try:
            result = function(*args, **kwargs)
            return result
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            _RUN_LOCK.release()

    return isolated


# 功能：
#   独立尝试一个清理动作，记录异常以保证后续资源仍得到回收尝试。
# 输入：
#   errors：本轮清理异常列表；function、args、kwargs：清理回调及参数。
# 输出：
#   result：清理返回值；失败时为 None。
def _cleanup_runtime_resource(errors, function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
        return result
    except BaseException as error:
        errors.append(error)
        return None


# 功能：
#   等待位姿消费者退出，超时不能被当成后台资源已正常排空。
# 输入：
#   worker：本次运行的位姿线程。
# 输出：
#   无。
def _join_pose_worker(worker):
    worker.join(timeout=5.0)
    if worker.is_alive():
        raise SimulationRuntimeError("POSE_WORKER_DID_NOT_STOP")


# 功能：
#   将清理失败附到原始异常；没有原始异常时拒绝发布成功结果。
# 输入：
#   errors：全部清理异常；primary：进入清理阶段前的异常。
# 输出：
#   无。
def _finish_runtime_cleanup(errors, primary):
    if not errors:
        return
    details = ",".join(type(error).__name__ for error in errors)
    if primary is not None:
        primary.add_note("Runtime cleanup failures: " + details)
        return
    raise SimulationRuntimeError("RUNTIME_CLEANUP_FAILED: " + details) from errors[0]


# 功能：
#   核对本轮输入资产始终未被替换，禁止用运行后的新文件为此前的飞行背书。
# 输入：
#   bindings：启动前已固定的路径与文件摘要。
# 输出：
#   无。
def _verify_runtime_inputs(bindings):
    if any(_sha256(path) != digest for path, digest in bindings.items()):
        raise SimulationRuntimeError("RUNTIME_INPUT_ASSET_CHANGED")


# 功能：
#   发布前按下游实际使用的严格合同核验全部证据字段，保留未提供字段的缺省状态。
# 输入：
#   evidence：适配器生成的运行证据。
# 输出：
#   validated：与执行层兼容的独立证据字典。
def _validated_runtime_evidence(evidence):
    contract = Px4GazeboRunEvidence.model_validate_json(
        encode_json(evidence, limit=64 * 1024**2), strict=True
    )
    validated = contract.model_dump(mode="json", exclude_unset=True)
    return validated


# 功能：
#   1. 在并入任务证据前校验停止请求，未知字段或损坏内容仍明确表示停止而非通过。
#   2. 原请求文件保持不变；非法请求只生成固定失败说明，不能使整份失败证据无法落盘。
# 输入：
#   path：本次运行的外部停止请求文件。
# 输出：
#   request：合法停止证据或明确的非法请求说明；没有文件时为 None。
def _read_external_abort_evidence(path: Path) -> dict | None:
    try:
        payload = read_runtime_object(path, maximum_bytes=64 * 1024)
        contract = Px4RuntimeAbortRequest.model_validate_json(
            encode_json(payload, limit=64 * 1024), strict=True)
        request = contract.model_dump(mode="json", exclude_unset=True)
    except FileNotFoundError:
        request = None
    except (OSError, ValueError, UnicodeError, RecursionError):
        request = {"reason": "EXTERNAL_ABORT_REQUEST_INVALID", "world_paused": False,
                   "issue_codes": ["EXTERNAL_ABORT_REQUEST_CONTRACT_REJECTED"]}
    return request


# 功能：
#   扩展参数只允许声明过的启动和恢复调节项，禁止覆盖已冻结的轨迹、输出目录及控制权限。
# 输入：
#   arguments：可选的扩展参数列表，采用名称和值成对表示。
# 输出：
#   options：验证后的名称到值字典，保留原参数字符串。
def _validated_executor_options(arguments):
    if arguments is None:
        return {}
    if (
        not isinstance(arguments, list)
        or len(arguments) > 16
        or len(arguments) % 2
        or any(not isinstance(value, str) or not value or "\0" in value for value in arguments)
    ):
        raise ValueError("EXECUTOR_EXTENSION_ARGUMENTS_INVALID")
    numeric = {
        "--takeoff-timeout-seconds",
        "--local-safety-command-grace-seconds",
        "--local-safety-runtime-stale-grace-seconds",
        "--semantic-progress-recovery-timeout-seconds",
        "--semantic-progress-abort-timeout-seconds",
        "--maximum-yaw-rate-deg-s",
    }
    options = {}
    for key, value in zip(arguments[::2], arguments[1::2], strict=True):
        if key in options or key not in numeric | {"--base-executor", "--heading-policy"}:
            raise ValueError("EXECUTOR_EXTENSION_OVERRIDE_FORBIDDEN")
        if key in numeric:
            _runtime_number(float(value), key, 0.001, 3600)
        if key == "--heading-policy" and value not in {"measured-hold", "route-tangent-relative"}:
            raise ValueError("EXECUTOR_HEADING_POLICY_INVALID")
        options[key] = value
    return options


# 功能：
#   从本轮 SDF 顶层声明读取真实世界或机型名，不补用学校地图或旧机型名称。
# 输入：
#   path：当前资产；tag：world 或 model；expected：可选的显式名称约束。
# 输出：
#   name：唯一且符合传输标识约束的名称。
def _sdf_entity_name(path: Path, tag: str, expected: str | None) -> str:
    payload = read_plugin_file(path, limit=64 * 1024**2)
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise SimulationRuntimeError("SDF_ENTITY_DECLARATIONS_FORBIDDEN")
    root = ElementTree.fromstring(payload)
    nodes = [node for node in root if node.tag.rsplit("}", 1)[-1] == tag]
    if len(nodes) != 1:
        raise SimulationRuntimeError("SDF_TOP_LEVEL_ENTITY_AMBIGUOUS")
    name = nodes[0].get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}", name):
        raise SimulationRuntimeError("SDF_TOP_LEVEL_ENTITY_NAME_INVALID")
    if expected is not None and expected != name:
        raise SimulationRuntimeError("SDF_TOP_LEVEL_ENTITY_NAME_MISMATCH")
    return name


# 功能：
#   按所选 SITL 模型查找唯一启动配置，不能对其他机型继续硬用 x500 的编号。
# 输入：
#   rootfs：本轮已复制的 PX4 根目录；model：明确的 SITL 模型名。
# 输出：
#   autostart_id：所选机型脚本中的配置编号。
def _px4_autostart_id(rootfs: Path, model: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("PX4 SITL model identifier is invalid")
    directory = rootfs / "etc" / "init.d-posix" / "airframes"
    scripts = list(directory.glob(f"*_gz_{model}"))
    if len(scripts) != 1 or not scripts[0].is_file():
        raise SimulationRuntimeError("PX4_MODEL_AUTOSTART_UNAVAILABLE_OR_AMBIGUOUS")
    match = re.fullmatch(r"([0-9]+)_gz_" + re.escape(model), scripts[0].name)
    if match is None:
        raise SimulationRuntimeError("PX4_MODEL_AUTOSTART_INVALID")
    autostart_id = match.group(1)
    return autostart_id


# 功能：
#   从系统允许的逻辑 CPU 中划分互不重叠的仿真与模型集合；不推断物理核心拓扑。
# 输入：
#   allowed_cpu_ids：可用逻辑 CPU；local_policy_enabled：是否启用本地策略。
# 输出：
#   general、model：普通进程和模型工作进程的 CPU 元组。
def _partition_runtime_cpus(
    allowed_cpu_ids: tuple[int, ...], *, local_policy_enabled: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(local_policy_enabled) is not bool or any(
        type(item) is not int or not 0 <= item <= 1_000_000 for item in allowed_cpu_ids
    ):
        raise ValueError("runtime CPU-affinity inputs are invalid")
    cpu_ids = tuple(sorted(set(allowed_cpu_ids)))
    if not local_policy_enabled or len(cpu_ids) < 4:
        general, model = cpu_ids, ()
        return general, model
    if len(cpu_ids) >= 16:
        reserved_count = 4
    elif len(cpu_ids) >= 8:
        reserved_count = 2
    else:
        reserved_count = 1
    general, model = cpu_ids[:-reserved_count], cpu_ids[-reserved_count:]
    return general, model


# 功能：
#   根据当前解释器真正可用的 CPU 和 taskset 生成隔离计划，不虚构硬件能力。
# 输入：
#   local_policy_enabled：是否启用本地策略。
# 输出：
#   plan：启用状态、CPU 集合或无法分配的原因。
def _runtime_cpu_affinity_plan(*, local_policy_enabled: bool) -> dict[str, object]:
    """Build a portable taskset plan without inventing unavailable CPUs."""

    if type(local_policy_enabled) is not bool:
        raise ValueError("local policy mode must be boolean")
    if not local_policy_enabled:
        plan = {"enabled": False, "reason": "local-policy-disabled"}
        return plan
    taskset = shutil.which("taskset")
    if taskset is None or not hasattr(os, "sched_getaffinity"):
        plan = {"enabled": False, "reason": "taskset-or-affinity-unavailable"}
        return plan
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    general, model = _partition_runtime_cpus(
        allowed,
        local_policy_enabled=True,
    )
    if not model:
        plan = {
            "enabled": False,
            "reason": "fewer-than-four-runtime-cpus",
            "allowed_cpu_ids": list(allowed),
        }
        return plan
    plan = {
        "enabled": True,
        "reason": "dedicated-local-model-cpu-set",
        "taskset_path": taskset,
        "allowed_cpu_ids": list(allowed),
        "general_cpu_ids": list(general),
        "local_model_cpu_ids": list(model),
    }
    return plan


# 功能：
#   在已验证的命令前添加逻辑 CPU 亲和性参数，不修改调用方命令列表。
# 输入：
#   command：启动参数；plan：分配计划；local_model：是否选择模型 CPU 集合。
# 输出：
#   argv：本次子进程的实际启动参数。
def _with_runtime_cpu_affinity(
    command: list[str], *, plan: dict[str, object], local_model: bool
) -> list[str]:
    if type(plan.get("enabled")) is not bool or type(local_model) is not bool:
        raise SimulationRuntimeError("runtime CPU-affinity mode must be boolean")
    if not plan["enabled"]:
        argv = list(command)
        return argv
    key = "local_model_cpu_ids" if local_model else "general_cpu_ids"
    cpu_ids = plan.get(key)
    taskset = plan.get("taskset_path")
    if (
        not isinstance(cpu_ids, list)
        or not cpu_ids
        or not isinstance(taskset, str)
        or any(type(cpu) is not int or not 0 <= cpu <= 1_000_000 for cpu in cpu_ids)
    ):
        raise SimulationRuntimeError("runtime CPU-affinity plan is incomplete")
    cpu_list = ",".join(str(int(cpu_id)) for cpu_id in cpu_ids)
    argv = [taskset, "--cpu-list", cpu_list, *command]
    return argv


# 功能：
#   将已扣除机身包络的最小净空分给局部安全余量与跟踪误差，避免固定容差穿越窄通道。
# 输入：
#   minimum_route_clearance_m：路线验收得到的最小净空，单位米。
# 输出：
#   policy：共享几何模块计算的跟踪与安全预算。
def _tracking_corridor_policy(minimum_route_clearance_m: float) -> dict[str, float]:
    """Bind controller lag to the route's measured free-space corridor.

    The clearance report already includes the vehicle collision envelope.  A
    fixed tracking tolerance can therefore be larger than a narrow corridor
    and allow a physically colliding deviation.  Reserve part of the measured
    clearance as a hard local-safety margin and spend only the remaining
    budget on tracking error.
    """

    try:
        policy = build_tracking_corridor_budget(minimum_route_clearance_m)
        return policy
    except ValueError as error:
        raise SimulationRuntimeError(str(error)) from error


# 功能：
#   由允许速度和体素尺寸计算有限前视距离，后续安全层仍须重新验证每次控制。
# 输入：
#   maximum_speed_mps：速度上限；world_resolution_m：局部地图分辨率。
# 输出：
#   lookahead_m：不超过四个体素的前视距离。
def _model_navigation_controller_lookahead_m(
    *, maximum_speed_mps: float, world_resolution_m: float = 0.25
) -> float:
    """Size a responsive target on an already revalidated model path.

    One-voxel targets made the predictive planner interpret a safe path as a
    request to crawl at roughly 0.1 m/s. A lookahead of about 0.75 seconds of
    configured speed lets the velocity controller reach useful cruise speed,
    while the four-voxel ceiling remains inside the coordinator's bounded
    local path and every command is still rechecked by sensor-rate safety.
    """

    if (
        type(maximum_speed_mps) not in (int, float)
        or type(world_resolution_m) not in (int, float)
        or not 0 < maximum_speed_mps <= 1e6
        or not 0 < world_resolution_m <= 1e6
    ):
        raise SimulationRuntimeError("model navigation lookahead inputs must be positive")
    lookahead_m = min(
        world_resolution_m * 4.0,
        max(world_resolution_m, maximum_speed_mps * 0.75),
    )
    return lookahead_m


# 功能：
#   求控制器与整条轨迹共同允许的全局速度上界；每个局部段仍使用自己的更严限制。
# 输入：
#   controller_speed_limit_mps：控制器速度上限；track：当前轨迹合同。
# 输出：
#   speed_limit_mps：有效全局速度上界。
def _effective_local_speed_limit_mps(
    *, controller_speed_limit_mps: float, track: Px4Track
) -> float:
    """Resolve the maximum speed authorized by both controller and track."""

    if (
        type(controller_speed_limit_mps) not in (int, float)
        or not 0 < controller_speed_limit_mps <= 1e6
    ):
        raise SimulationRuntimeError("controller speed limit must be finite and positive")
    track_limits = [
        float(point.speed_limit_mps) for point in track.points if point.speed_limit_mps is not None
    ]
    if any(not math.isfinite(limit) or limit <= 0.0 for limit in track_limits):
        raise SimulationRuntimeError("track speed limits must be finite and positive")
    speed_limit_mps = min(
        controller_speed_limit_mps, max(track_limits, default=controller_speed_limit_mps)
    )
    return speed_limit_mps


# 功能：
#   只读取运行链路实际使用的速度与加速度限制，不将旧增益字段当成 PX4 参数注入。
# 输入：
#   path：本轮已绑定的控制器参数文件。
# 输出：
#   speed_limit、acceleration_limit：有限正值的速度和加速度上限。
def _load_controller_limits(path: Path) -> tuple[float, float]:
    """Load only the limits that the runtime actually consumes.

    PX4 owns its low-level gains. Historical gain-shaped fields can remain in
    preserved imported artifacts, but they have no route into runtime control.
    """

    try:
        payload = read_runtime_object(path)
    except (OSError, ValueError) as error:
        raise SimulationRuntimeError("controller parameters are invalid") from error
    if not isinstance(payload, dict):
        raise SimulationRuntimeError("controller parameters must be a JSON object")
    limits: list[float] = []
    for key in ("vel_limit", "accel_limit"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SimulationRuntimeError(f"controller {key} must be numeric")
        if not 0 < value <= 1e6:
            raise SimulationRuntimeError(f"controller {key} must be finite and positive")
        limit = float(value)
        limits.append(limit)
    speed_limit, acceleration_limit = limits
    return speed_limit, acceleration_limit


# 功能：
#   限制 PX4 与世界坐标原点校正的单步创新量，物理碰撞净空由独立监视器检查。
# 输入：
#   minimum_route_clearance_m：验证为正的路线净空。
# 输出：
#   correction_limit_m：允许的 0.25 米单步校正上限。
def _identity_correction_limit_m(minimum_route_clearance_m: float) -> float:
    """Bound estimator/world transform innovation independently of tracking.

    PX4's local estimator origin can differ from the Gazebo world frame by
    more than a narrow route's remaining physical corridor.  That transform
    does not itself move the aircraft or spend collision clearance.  Keep a
    conservative 25 cm initial/step innovation bound and the consecutive
    mismatch gate, while a separate one-metre command-contract ceiling limits
    cumulative correction.  Physical clearance remains independently enforced
    against the Gazebo collision centre.
    """

    if (
        type(minimum_route_clearance_m) not in (int, float)
        or not 0 < minimum_route_clearance_m <= 1e6
    ):
        raise SimulationRuntimeError("identity correction requires positive route clearance")
    correction_limit_m = 0.25
    return correction_limit_m


# 功能：
#   将着陆阶段、表面类别和微小接触深度联合判断，不能据此认定飞行器已安全着陆。
# 输入：
#   phase：执行阶段；primitive_name：接触体名称；clearance_m：带符号净空。
# 输出：
#   tolerated：是否符合着陆接触数值容差。
def _is_tolerated_landing_contact(
    *, phase: str | None, primitive_name: str, clearance_m: float
) -> bool:
    if phase not in {"LANDING", "LANDED", "COMPLETE"}:
        return False
    if not -0.02 <= clearance_m < -0.001:
        return False
    tolerated = any(
        token in primitive_name.casefold() for token in ("floor", "ground", "road", "pad")
    )
    return tolerated


# 功能：
#   核验接触发生在水平箱体上表面且完整机身投影受支撑，侧撞不能因物体叫 floor 而获豁免。
# 输入：
#   primitive：已验证接触基元；center：机身中心；radius、half_height：保守机身包络。
# 输出：
#   supported：是否符合从上方接触支撑面的几何条件。
def _landing_contact_from_above(primitive, center, radius, half_height):
    primitive = _validated_primitive(primitive)
    if (
        "size_x" not in primitive
        or abs(primitive.get("roll_rad", 0.0)) > 1e-8
        or abs(primitive.get("pitch_rad", 0.0)) > 1e-8
    ):
        return False
    _runtime_number(radius, "landing radius", 0.001, 100)
    _runtime_number(half_height, "landing half height", 0.001, 100)
    top = primitive["center_z"] + primitive["size_z"] / 2
    gap = center[2] - half_height - top
    yaw = primitive.get("yaw_rad", 0.0)
    dx, dy = center[0] - primitive["center_x"], center[1] - primitive["center_y"]
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    supported = (
        -0.020001 <= gap <= 0.001
        and abs(local_x) + radius <= primitive["size_x"] / 2
        and abs(local_y) + radius <= primitive["size_y"] / 2
    )
    return supported


# 功能：
#   在主动飞行阶段要求独立定位身份持续更新，结束后改由着陆状态和碰撞监视器收尾。
# 输入：
#   phase：当前执行阶段，未知阶段按需要定位身份处理。
# 输出：
#   required：是否必须检查高频定位身份。
def _tracking_identity_required(phase: str | None) -> bool:
    """Keep the PX4/Gazebo identity heartbeat strict only while flight is active.

    The checkpoint executor stops publishing its high-rate tracking artifact
    after the airborne schedule has completed. Landing is then proved by the
    PX4 landed-state and Gazebo collision monitors, so treating that expected
    publisher shutdown as an in-flight identity loss creates a false abort.
    """

    required = phase not in {"PREFLIGHT", "LANDING", "LANDED", "COMPLETE", "FAILED"}
    return required


# 功能：
#   只给明确的结束阶段一次有界收尾窗口，不给未知阶段无限延期。
# 输入：
#   phase：执行阶段；already_used：是否已使用收尾宽限。
# 输出：
#   allowed：是否允许本次宽限。
def _terminal_completion_grace_allowed(phase: str | None, *, already_used: bool) -> bool:
    """Allow one finite cleanup window after the airborne schedule closes."""

    allowed = already_used is False and phase in {"LANDING", "LANDED", "COMPLETE", "FAILED"}
    return allowed


# 功能：
#   决定是否使用粗参考路线作偏航约束；模型掌握控制权时允许独立避障但不关闭碰撞检查。
# 输入：
#   phase：当前阶段；goal_observed：是否已观测到目标。
#   model_control_authority_required：模型主控标志。
# 输出：
#   enforced：是否执行参考折线偏离限制。
def _reference_route_deviation_enforced(
    *,
    phase: str | None,
    goal_observed: bool,
    model_control_authority_required: bool,
) -> bool:
    """Use the qualified reference-route corridor while it remains authoritative.

    Under required model control, the local policy continuously requests a
    bounded body-frame velocity and yaw-rate intent. Sensor-rate deterministic
    safety may alter that intent to avoid an obstacle, so the resulting motion
    can intentionally leave the coarse global reference polyline. Static
    collision monitoring, semantic progress gating, identity checks, and the
    finite no-progress watchdog remain active; applying the one-metre teacher
    corridor as an additional abort would incorrectly forbid that autonomy.
    """

    enforced = (
        phase not in {"LANDING", "LANDED", "FAILED", "COMPLETE"}
        and not goal_observed
        and not model_control_authority_required
    )
    return enforced


# 功能：
#   检查导航周期是否建立了经过复验的运动租约；实际下发证据还须通过独立控制应用验收。
# 输入：
#   cycle：导航周期记录；call_by_id：按调用编号索引的模型记录。
# 输出：
#   applied：是否具有候选路径或推杆运动租约的记录。
def _model_navigation_cycle_is_applied(
    cycle: dict[str, Any],
    *,
    call_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Prove that a model decision established a controller motion lease."""

    if not isinstance(cycle.get("controller_target_m"), dict):
        return False
    action = cycle.get("model_action")
    if action == "select-candidate":
        applied = bool(
            cycle.get("selected_candidate_id") is not None
            and cycle.get("deterministic_metric_path_revalidated") is True
            and cycle.get("dynamic_path_revalidated") is True
        )
        return applied
    if action != "pilot-control":
        return False
    call = call_by_id.get(str(cycle.get("model_call_id", "")))
    trace = call.get("local_expert_trace") if isinstance(call, dict) else None
    applied = bool(
        cycle.get("selected_candidate_id") is None
        and isinstance(cycle.get("selected_metric_path_sha256"), str)
        and isinstance(trace, dict)
        and trace.get("navigation_action") == "pilot-control"
        and isinstance(trace.get("pilot_control"), dict)
    )
    return applied


# 功能：
#   为低实时率仿真分配不超过六小时的延期预算，停滞监视器仍独立限制无进展等待。
# 输入：
#   estimated_flight_seconds：估算飞行时长；track_timeout_seconds：轨迹基础超时。
# 输出：
#   extension_seconds：有上下界的延期秒数。
def _progress_extension_budget_seconds(
    *, estimated_flight_seconds: float, track_timeout_seconds: float
) -> float:
    """Return a finite wall-time budget sized for real low-RTF depth simulation.

    Even a physically short route can contain thousands of 10 Hz setpoints and
    run at a small fraction of real time once RGB, semantic and depth cameras
    are active.  Scale continuously from the physical route estimate instead
    of introducing a cliff at an arbitrary route duration.  The budget remains
    capped at six hours, while the independent 180 second no-progress watchdog
    and the executor's bounded recovery windows reject hangs inside that cap.
    """

    _runtime_number(estimated_flight_seconds, "estimated flight seconds")
    _runtime_number(track_timeout_seconds, "track timeout", 0.001)
    budget = max(track_timeout_seconds * 2.0, estimated_flight_seconds * 75.0)
    extension_seconds = max(180.0, min(21_600.0, budget))
    return extension_seconds


# 功能：
#   合并基础与进展延期预算，形成子执行器绝对截止时长。
# 输入：
#   estimated_flight_seconds：飞行估算；track_timeout_seconds：基础轨迹超时。
# 输出：
#   timeout_seconds：不超过六小时的子执行器时限。
def _executor_track_timeout_seconds(
    *, estimated_flight_seconds: float, track_timeout_seconds: float
) -> float:
    """Keep the child executor alive while the adapter's progress watchdog is authoritative."""

    extension = _progress_extension_budget_seconds(
        estimated_flight_seconds=estimated_flight_seconds,
        track_timeout_seconds=track_timeout_seconds,
    )
    timeout_seconds = min(21_600.0, track_timeout_seconds + extension)
    return timeout_seconds


# 功能：
#   将父进程绝对时限放在子执行器时限之外，为有界收尾留出时间。
# 输入：
#   executor_track_timeout_seconds：子执行器的轨迹截止时长。
# 输出：
#   timeout_seconds：父进程墙钟时限。
def _executor_wall_timeout_seconds(*, executor_track_timeout_seconds: float) -> float:
    """Keep the parent watchdog outside the child executor's finite deadline.

    The child deadline is deliberately expanded for camera-heavy simulations
    that run below real time.  Runtime replanning can also replace the original
    schedule with a longer one.  If the parent remains tied to the original
    physical-route estimate, it can abort a healthy executor at the newly
    selected goal.  The independent no-progress watchdog still fails a stalled
    run after 180 seconds; this is only the absolute outer process ceiling.
    """

    _runtime_number(executor_track_timeout_seconds, "executor timeout", 0.001)
    timeout_seconds = max(450.0, executor_track_timeout_seconds + 270.0)
    return timeout_seconds


# 功能：
#   将带时区的模型调用时间换算为 Unix 毫秒，拒绝依赖本机时区解释无时区时间。
# 输入：
#   record：包含 created_at 的调用记录。
# 输出：
#   timestamp_ms：有效时间戳；无效时为 None。
def _model_call_unix_ms(record: dict[str, Any]) -> int | None:
    value = record.get("created_at")
    if not isinstance(value, str) or not value:
        return None
    with suppress(ValueError, OverflowError, OSError):
        created = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if created.utcoffset() is not None:
            timestamp_ms = round(created.timestamp() * 1_000)
            return timestamp_ms
    return None


# 功能：
#   在控制命令实际生产区间内计算模型调用的最大间隙，包含首尾没有调用的时间。
# 输入：
#   active_command_timestamps_ms：命令时间戳；model_call_records：模型调用记录。
# 输出：
#   maximum_gap_seconds：最大间隙；证据不足时为 None。
def _maximum_model_participation_gap_seconds(
    *,
    active_command_timestamps_ms: list[int],
    model_call_records: list[dict[str, Any]],
) -> float | None:
    """Measure the largest successful-model gap across active command production."""

    if not active_command_timestamps_ms or not model_call_records:
        return None
    active_start = min(active_command_timestamps_ms)
    active_end = max(active_command_timestamps_ms)
    calls = sorted(
        timestamp
        for record in model_call_records
        if (timestamp := _model_call_unix_ms(record)) is not None
        and active_start <= timestamp <= active_end
    )
    if not calls:
        maximum_gap_seconds = (active_end - active_start) / 1_000.0
        return maximum_gap_seconds
    timeline = [active_start, *calls, active_end]
    maximum_gap_seconds = max(
        (later - earlier) / 1_000.0 for earlier, later in zip(timeline, timeline[1:], strict=False)
    )
    return maximum_gap_seconds


# 功能：
#   将写入器排空声明与独立读取的实际记录数交叉核对，拒绝只凭 complete 标志通过。
# 输入：
#   summary：写入汇总；record_count：实际总记录数；artifact_counts：逐文件计数。
# 输出：
#   complete：全部数量与生命周期条件是否一致。
def _runtime_evidence_writer_complete(
    summary: dict[str, Any] | None,
    *,
    record_count: int,
    artifact_counts: dict[str, int],
) -> bool:
    """Require a clean, drained writer and an independently counted corpus."""
    from .runtime_evidence import runtime_evidence_inventory_complete

    complete = runtime_evidence_inventory_complete(
        summary, artifact_counts=artifact_counts, record_count=record_count
    )
    return complete


# 功能：
#   核对每次快照提交均已写入或被新值取代，并确认后台线程退出、没有未处理或拒绝项。
# 输入：
#   summary：快照写入器的生命周期和五项计数。
# 输出：
#   complete：快照是否完整排空。
def _runtime_snapshot_writer_complete(summary: dict[str, Any] | None) -> bool:
    """Require every mutable snapshot submission to be written or superseded."""

    count_fields = (
        "rejected_count",
        "pending_count",
        "submitted_count",
        "completed_count",
        "superseded_count",
    )
    complete = bool(
        isinstance(summary, dict)
        and all(type(summary.get(key)) is int and summary[key] >= 0 for key in count_fields)
        and summary.get("schema_version") == "dronedream.runtime-snapshot-writer.v1"
        and summary.get("complete") is True
        and summary.get("issue_code") is None
        and summary.get("rejected_count") == 0
        and summary.get("pending_count") == 0
        and summary.get("thread_finished") is True
        and summary.get("thread_alive") is False
        and summary.get("submitted_count")
        == summary.get("completed_count", 0) + summary.get("superseded_count", 0)
    )
    return complete


# 功能：
#   从新鲜且类型有效的执行遥测提取进展，不把未来时间、过期记录或字符串授权当作有效状态。
# 输入：
#   path：跟踪快照；now_unix_ms：当前墙钟；maximum_age_seconds：最大允许年龄。
# 输出：
#   progress：进度索引、目标、距离、位置与模型授权的五元组；无效时为 None。
def _fresh_tracking_progress_state(
    path: Path,
    *,
    now_unix_ms: int,
    maximum_age_seconds: float = 5.0,
) -> (
    tuple[
        int,
        str | None,
        float | None,
        tuple[float, float, float] | None,
        bool,
    ]
    | None
):
    """Return fresh route and semantic progress from closed-loop telemetry.

    Gazebo can run well below real time when the depth camera and collision
    world are active.  A fixed wall deadline must still stop a hung executor,
    but it must not abort a process that is publishing fresh physical progress
    toward its current semantic waypoint.  The caller remains responsible for
    requiring a meaningful improvement and for enforcing finite budgets.
    """

    try:
        _runtime_number(now_unix_ms, "wall clock", 0, 10**16)
        _runtime_number(maximum_age_seconds, "progress age", 0, 3600)
        payload = read_runtime_object(path)
        updated_at_unix_ms = payload["updated_at_unix_ms"]
        schedule_index = payload["schedule_index"]
        state = payload["state"]
        if (
            type(updated_at_unix_ms) is not int
            or type(schedule_index) is not int
            or not isinstance(state, str)
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    age_seconds = (now_unix_ms - updated_at_unix_ms) / 1_000.0
    if not 0.0 <= age_seconds <= maximum_age_seconds:
        return None
    if schedule_index < 0 or state not in {
        "tracking",
        "recovering",
        "model-progress-hold",
    }:
        return None
    navigation_goal_id_value = payload.get("navigation_goal_id")
    navigation_goal_id = navigation_goal_id_value
    if navigation_goal_id is not None and (
        not isinstance(navigation_goal_id, str) or not 1 <= len(navigation_goal_id) <= 512
    ):
        return None
    model_goal_distance_m: float | None = None
    with suppress(TypeError, ValueError):
        candidate = payload.get("model_goal_distance_m")
        model_goal_distance_m = _runtime_number(candidate, "goal distance")
    observed_position_m: tuple[float, float, float] | None = None
    observed_payload = payload.get("observed_world_collision_center_m")
    if isinstance(observed_payload, dict):
        with suppress(KeyError, TypeError, ValueError):
            candidate_position = (
                _runtime_number(observed_payload["x"], "position x", -1e9),
                _runtime_number(observed_payload["y"], "position y", -1e9),
                _runtime_number(observed_payload["z"], "position z", -1e9),
            )
            if all(math.isfinite(value) for value in candidate_position):
                observed_position_m = candidate_position
    model_navigation_authorized = payload.get("model_navigation_authorized", False)
    if type(model_navigation_authorized) is not bool:
        return None
    progress = (
        schedule_index,
        navigation_goal_id,
        model_goal_distance_m,
        observed_position_m,
        model_navigation_authorized,
    )
    return progress


# 功能：
#   复用完整进展验证，仅向需要调度索引的调用方返回索引。
# 输入：
#   path：进展文件；now_unix_ms：当前时间；maximum_age_seconds：新鲜度界限。
# 输出：
#   schedule_index：已验证的索引；无效时为 None。
def _fresh_tracking_progress(
    path: Path,
    *,
    now_unix_ms: int,
    maximum_age_seconds: float = 5.0,
) -> int | None:
    """Return the fresh schedule index used by bounded wall-time extension."""

    progress = _fresh_tracking_progress_state(
        path,
        now_unix_ms=now_unix_ms,
        maximum_age_seconds=maximum_age_seconds,
    )
    schedule_index = progress[0] if progress is not None else None
    return schedule_index


# 功能：
#   读取短暂等待模型期间仍保留的语义进展窗口，只用于有界延期，不授予飞行控制权。
# 输入：
#   path：语义进展窗口；now_unix_ms：当前时间；maximum_age_seconds：年龄上限。
# 输出：
#   window：目标、最佳距离、物理进展修订及授权调度修订；无效时为 None。
def _fresh_semantic_progress_window(
    path: Path,
    *,
    now_unix_ms: int,
    maximum_age_seconds: float = 5.0,
) -> tuple[str, float, int, int] | None:
    """Read durable semantic progress that survives brief authority-hold files.

    ``closed-loop-tracking.json`` is a latest-state register and can be
    overwritten by a short model-authority hold before the adapter's 2 Hz
    watchdog samples it.  The semantic window retains monotonic physical and
    model-authorized schedule revisions, so it is the durable companion signal
    for wall-time extension.  It never authorizes flight; it only prevents a
    progressing low-RTF simulation from being mistaken for a hung executor.
    """

    try:
        _runtime_number(now_unix_ms, "wall clock", 0, 10**16)
        _runtime_number(maximum_age_seconds, "progress age", 0, 3600)
        payload = read_runtime_object(path)
        updated_at_unix_ms = payload["updated_at_unix_ms"]
        navigation_goal_id = payload["navigation_goal_id"]
        best_distance_m = _runtime_number(payload["best_model_goal_distance_m"], "best distance")
        progress_revision = payload["progress_revision"]
        schedule_revision = payload.get("authorized_schedule_revision", 0)
        if (
            any(
                type(value) is not int
                for value in (updated_at_unix_ms, progress_revision, schedule_revision)
            )
            or not isinstance(navigation_goal_id, str)
            or len(navigation_goal_id) > 512
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    age_seconds = (now_unix_ms - updated_at_unix_ms) / 1_000.0
    if (
        not 0.0 <= age_seconds <= maximum_age_seconds
        or not navigation_goal_id
        or not math.isfinite(best_distance_m)
        or best_distance_m < 0.0
        or progress_revision < 0
        or schedule_revision < 0
    ):
        return None
    window = (
        navigation_goal_id,
        best_distance_m,
        progress_revision,
        schedule_revision,
    )
    return window


# 功能：
#   依据调度推进、目标变更、距离改善或已授权物理移动更新停滞检测锚点。
# 输入：
#   progress_state：验证后的进展；last_schedule_index、last_goal_id：上次调度和目标。
#   semantic_anchor_m、physical_anchor_m：上次语义距离和物理位置。
# 输出：
#   anchors：是否推进、原因、新调度索引、目标、语义锚点及物理锚点。
def _advance_closed_loop_progress_anchors(
    progress_state: tuple[
        int,
        str | None,
        float | None,
        tuple[float, float, float] | None,
        bool,
    ],
    *,
    last_schedule_index: int,
    last_goal_id: str | None,
    semantic_anchor_m: float | None,
    physical_anchor_m: tuple[float, float, float] | None,
) -> tuple[
    bool,
    str | None,
    int,
    str | None,
    float | None,
    tuple[float, float, float] | None,
]:
    """Advance finite-timeout anchors from measured closed-loop progress.

    A model-authorized local leg can keep one reference schedule index while
    flying many metres toward its semantic waypoint. Treat its measured
    position or decreasing goal distance as progress just as the stall
    watchdog does, so the wall-time extension cannot contradict that watchdog.
    """

    (
        schedule_index,
        navigation_goal_id,
        model_goal_distance_m,
        observed_position_m,
        model_navigation_authorized,
    ) = progress_state
    if schedule_index > last_schedule_index:
        anchors = (
            True,
            "schedule_index_advanced",
            schedule_index,
            navigation_goal_id,
            model_goal_distance_m,
            observed_position_m,
        )
        return anchors
    if navigation_goal_id != last_goal_id:
        anchors = (
            True,
            "navigation_goal_changed",
            last_schedule_index,
            navigation_goal_id,
            model_goal_distance_m,
            observed_position_m,
        )
        return anchors
    if model_goal_distance_m is not None and (
        semantic_anchor_m is None or model_goal_distance_m <= semantic_anchor_m - 0.01
    ):
        anchors = (
            True,
            "semantic_goal_distance_decreased",
            last_schedule_index,
            last_goal_id,
            model_goal_distance_m,
            observed_position_m,
        )
        return anchors
    if (
        model_navigation_authorized
        and observed_position_m is not None
        and (physical_anchor_m is None or math.dist(physical_anchor_m, observed_position_m) >= 0.05)
    ):
        anchors = (
            True,
            "model_authorized_physical_displacement",
            last_schedule_index,
            last_goal_id,
            semantic_anchor_m,
            observed_position_m,
        )
        return anchors
    anchors = (
        False,
        None,
        last_schedule_index,
        last_goal_id,
        semantic_anchor_m,
        physical_anchor_m,
    )
    return anchors


# 功能：
#   每份新进展证据最多消费一次延期资格，并为短暂模型等待保留有限有效窗口。
# 输入：
#   deadline_progress_state：本次进展；progress_evidence_revision：当前修订。
#   last_extended_progress_revision：已消费修订；last_progress_observed_at：最近进展时间。
#   now：当前单调时间；maximum_transient_hold_seconds：临时等待上限。
# 输出：
#   allowed：是否可以使用这份进展延长墙钟预算。
def _recent_progress_extension_allowed(
    *,
    deadline_progress_state: object | None,
    progress_evidence_revision: int,
    last_extended_progress_revision: int,
    last_progress_observed_at: float | None,
    now: float,
    maximum_transient_hold_seconds: float = 60.0,
) -> bool:
    """Keep a wall extension valid across a brief model-authority hold.

    ``closed-loop-tracking.json`` deliberately enters ``model-authority-hold``
    while the next model decision is pending. That state is not itself proof of
    motion, but it must not erase meaningful motion observed moments earlier.
    The revision must still be new and the cached evidence is accepted for no
    longer than the semantic navigation abort window. A crashed or stalled
    executor therefore receives at most one additional wall-time slice, while a
    legitimate precision turn is not aborted earlier than its own recovery
    contract.
    """

    if progress_evidence_revision <= last_extended_progress_revision:
        return False
    if deadline_progress_state is not None:
        return True
    allowed = (
        last_progress_observed_at is not None
        and 0.0 <= now - last_progress_observed_at <= maximum_transient_hold_seconds
    )
    return allowed


# 功能：
#   识别自身具有有限截止时间的悬停、检查点和动作阶段，外层停滞限制仍然有效。
# 输入：
#   phase：当前执行阶段。
# 输出：
#   allowed：是否属于允许一次外层等待的有界子阶段。
def _bounded_executor_subphase_extension_allowed(phase: str | None) -> bool:
    """Identify executor phases with their own finite completion deadline.

    These phases intentionally stop advancing the high-rate route schedule
    while they settle telemetry, ask the checkpoint model, or execute a domain
    action. The adapter's 180-second no-progress watchdog remains authoritative
    over hangs, so a wall-time slice may follow the active bounded subphase
    without converting the overall run into an unbounded wait.
    """

    allowed = phase in {"WAYPOINT_SETTLE", "CHECKPOINT", "ACTION"}
    return allowed


# 功能：
#   在普通文件和身份不变约束下分块计算摘要，不一次载入大型日志。
# 输入：
#   path：本轮资产或证据文件。
# 输出：
#   digest：实际读取字节的 SHA-256。
def _sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=8 * 1024**3)
    return digest


# 功能：
#   有界、原子地发布有限 JSON 快照，读者不能看到一半写入的数据。
# 输入：
#   path：快照目标；payload：将要写入的对象。
# 输出：
#   无。
def _write_json(path: Path, payload: object) -> None:
    publish_runtime_json(path, payload, maximum_bytes=64 * 1024**2)


# 功能：
#   用本次调用独占的临时文件原子替换二进制快照，清理只针对仍属于自己的临时文件。
# 输入：
#   path：快照目标；payload：不超过 64 MiB 的字节。
# 输出：
#   无。
def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or len(payload) > 64 * 1024**2:
        raise ValueError("RUNTIME_BINARY_SNAPSHOT_TOO_LARGE")
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".snapshot-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            owned = os.fstat(stream.fileno())
            stream.write(payload)
            stream.flush()
        check_plain_plugin_path(path)
        check_plain_plugin_path(temporary)
        if not os.path.samestat(owned, temporary.stat()):
            raise ValueError("RUNTIME_BINARY_SNAPSHOT_REPLACED")
        temporary.replace(path)
    finally:
        if temporary is not None and owned is not None:
            with suppress(FileNotFoundError):
                if os.path.samestat(owned, temporary.lstat()):
                    temporary.unlink()


# 功能：
#   构造包含长度、类型、内容和 CRC 的标准 PNG 数据块。
# 输入：
#   name：四字节块类型；payload：块内容。
# 输出：
#   chunk：完整 PNG 数据块字节。
def _png_chunk(name: bytes, payload: bytes) -> bytes:
    chunk = (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", binascii.crc32(name + payload) & 0xFFFFFFFF)
    )
    return chunk


# 功能：
#   按已验证的像素布局编码真实 Gazebo 图像，正确处理颜色顺序与行填充。
# 输入：
#   message：原生 RGB/RGBA/BGR/BGRA 消息；output_size：可选输出尺寸。
# 输出：
#   png：原帧或缩放后的 PNG 字节。
def _gazebo_image_png(
    message: Any,
    *,
    output_size: tuple[int, int] | None = None,
) -> bytes:
    """Encode a bounded Gazebo RGB/RGBA frame without an image dependency."""

    from .rgb_input_quality import freeze_gazebo_rgb

    message = freeze_gazebo_rgb(message)
    width, height = message.width, message.height
    formats = {
        3: (3, 2, False),  # RGB_INT8
        4: (4, 6, False),  # RGBA_INT8
        8: (3, 2, True),  # BGR_INT8
        5: (4, 6, True),  # BGRA_INT8
    }
    channels, color_type, swap_red_blue = formats.get(int(message.pixel_format_type), (0, 0, False))
    if channels == 0:
        raise ValueError("Gazebo live frame format is not supported")
    row_bytes = width * channels
    step = message.step
    source = bytes(message.data)
    if len(source) < step * height:
        raise ValueError("Gazebo live frame payload is incomplete")
    if output_size is not None:
        png, _rgb8 = _gazebo_image_model_payload(
            message,
            output_size=output_size,
        )
        return png
    rows: list[bytes] = []
    for row_index in range(height):
        row = bytearray(source[row_index * step : row_index * step + row_bytes])
        if swap_red_blue:
            for offset in range(0, len(row), channels):
                row[offset], row[offset + 2] = row[offset + 2], row[offset]
        rows.append(b"\x00" + bytes(row))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=3))
        + _png_chunk(b"IEND", b"")
    )
    return png


# 功能：
#   同一次缩放生成模型直接使用的 RGB8 字节和可回放 PNG，及时关闭临时图像。
# 输入：
#   message：真实图像消息；output_size：明确的输出宽高。
# 输出：
#   png、rgb8：同一幅图像的证据编码与紧密排列三通道像素。
def _gazebo_image_model_payload(
    message: Any,
    *,
    output_size: tuple[int, int],
) -> tuple[bytes, bytes]:
    """Return one resized frame as evidence PNG and packed RGB8 model input.

    The online visual expert consumes the packed bytes directly.  Keeping the
    PNG beside them preserves replayable evidence without forcing the model
    worker to decode the image that the sensor worker has just encoded.
    """

    output_width, output_height = output_size
    if (
        type(output_width) is not int
        or type(output_height) is not int
        or output_width <= 0
        or output_height <= 0
        or output_width > 4096
        or output_height > 2160
    ):
        raise ValueError("Gazebo model frame dimensions are invalid")
    import io

    from .rgb_input_quality import decode_gazebo_rgb

    with decode_gazebo_rgb(message) as decoded, decoded.resize(output_size) as resized:
        rgb8 = resized.tobytes()
        with io.BytesIO() as buffer:
            resized.save(buffer, format="PNG", compress_level=1)
            png = buffer.getvalue()
    return png, rgb8


# 功能：
#   按仿真语义相机的末通道类别协议生成单通道标签图，禁止灰度转换改变类别编号。
# 输入：
#   message：RGB 标签消息；allowed_class_ids：允许的类别编号集合。
# 输出：
#   png：保持原始类别编号的标签 PNG。
def _gazebo_semantic_label_png(
    message: Any,
    *,
    allowed_class_ids: frozenset[int] = frozenset(range(8)),
) -> bytes:
    """Encode Gazebo's RGB label map as a normalized one-channel PNG.

    Gazebo's semantic-camera contract places the class identifier in the last
    channel of each RGB pixel.  Keeping only that channel prevents image
    libraries from turning a label map into weighted grayscale values.
    """

    from .rgb_input_quality import freeze_gazebo_rgb

    message = freeze_gazebo_rgb(message)
    width, height = message.width, message.height
    if (
        not isinstance(allowed_class_ids, frozenset)
        or not allowed_class_ids
        or any(type(value) is not int or not 0 <= value <= 255 for value in allowed_class_ids)
    ):
        raise ValueError("Gazebo semantic class identifiers are invalid")
    if message.pixel_format_type != 3:
        raise ValueError("Gazebo semantic frame must use RGB_INT8")
    row_bytes = width * 3
    step = message.step
    source = bytes(message.data)
    if len(source) < step * height:
        raise ValueError("Gazebo semantic frame payload is incomplete")
    rows: list[bytes] = []
    observed_class_ids: set[int] = set()
    for row_index in range(height):
        source_row = source[row_index * step : row_index * step + row_bytes]
        labels = source_row[2::3]
        observed_class_ids.update(labels)
        rows.append(b"\x00" + labels)
    unsupported = observed_class_ids - allowed_class_ids
    if unsupported:
        raise ValueError(
            "Gazebo semantic frame contains unsupported class identifiers: "
            + ",".join(str(value) for value in sorted(unsupported))
        )
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=3))
        + _png_chunk(b"IEND", b"")
    )
    return png


# 功能：
#   生成俯瞰当前路线的真实场景相机；此观察视角不冒充无人机前视传感器。
# 输入：
#   route_points：当前任务的世界坐标路线点。
# 输出：
#   model、camera：观察相机 SDF 与实际生成位置。
def _live_camera_sdf(
    route_points: list[tuple[float, float, float]],
) -> tuple[str, tuple[float, float, float]]:
    """Build a real Gazebo camera that frames the selected mission route."""

    xs = [point[0] for point in route_points]
    ys = [point[1] for point in route_points]
    zs = [point[2] for point in route_points]
    target = (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    )
    span = max(max(xs) - min(xs), max(ys) - min(ys), 6.0)
    camera = (
        target[0] - span * 0.9,
        target[1] - span * 0.9,
        max(zs) + span * 0.7 + 3.0,
    )
    dx, dy, dz = (target[index] - camera[index] for index in range(3))
    yaw = math.atan2(dy, dx)
    # Gazebo cameras look along sensor +X. Positive pitch rotates that axis
    # downward, toward the route below this camera.
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    model = f"""<?xml version="1.0"?>
<sdf version="1.10">
  <model name="dronedream_live_camera">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="camera_link">
      <!-- Keep orientation on the link: the create-service spawn pose sets
           the model's world position and otherwise replaces model pose RPY. -->
      <pose relative_to="__model__">0 0 0 0 {pitch:.12g} {yaw:.12g}</pose>
      <sensor name="live_camera" type="camera">
        <always_on>true</always_on>
        <update_rate>12</update_rate>
        <topic>/dronedream/live/camera</topic>
        <camera>
          <horizontal_fov>1.0472</horizontal_fov>
          <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>5000</far></clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>
"""
    return model, camera


# 功能：
#   无 shell 地运行短时外部探测命令，显式限定等待时间并保留退出码。
# 输入：
#   argv：命令及参数；env：子进程环境；timeout：等待秒数。
# 输出：
#   result：退出码、标准输出和标准错误。
def _run(
    argv: list[str], *, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result


# 功能：
#   在启动仿真前检查同一解释器能否加载 ONNX Runtime，不能缺模型后端仍继续飞行。
# 输入：
#   无。
# 输出：
#   无。
def _require_local_policy_runtime() -> None:
    """Refuse to start Gazebo when this interpreter cannot execute ONNX policy files."""

    probe = _run(
        [
            sys.executable,
            "-c",
            "import onnxruntime; print(onnxruntime.__version__)",
        ],
        env=os.environ.copy(),
        timeout=15,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        suffix = f":{detail[-1]}" if detail else ""
        raise SimulationRuntimeError(f"LOCAL_POLICY_RUNTIME_UNAVAILABLE{suffix}")


# 功能：
#   要求本地导航时必须具备当前深度传感器工作链路，缺失时禁止静默跳过模型控制。
# 输入：
#   local_navigation_provider：指定导航供应方；depth_safety_supported：原生深度链路可用性。
# 输出：
#   无。
def _require_local_navigation_sensor_runtime(
    *,
    local_navigation_provider: str | None,
    depth_safety_supported: bool,
) -> None:
    """Reject a requested local model loop before it can silently disappear.

    The current model worker is also the authority that projects fresh metric
    depth and publishes hash-bound local candidates. Running without it would
    make a model-required acceptance mission look alive while no expert ever
    receives an observation or authorizes motion.
    """

    if local_navigation_provider is not None and not depth_safety_supported:
        raise SimulationRuntimeError(
            "LOCAL_NAVIGATION_SENSOR_RUNTIME_UNAVAILABLE: requested local navigation "
            "requires a depth-capable vehicle and the runtime depth safety worker"
        )


# 功能：
#   有界等待所选世界的场景服务就绪，不把其他世界服务当成本轮启动成功。
# 输入：
#   gz_binary：Gazebo 程序；world_name：精确世界名；env：隔离环境；timeout：总等待秒数。
# 输出：
#   无。
def _wait_for_world(gz_binary: str, world_name: str, env: dict[str, str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    service = f"/world/{world_name}/scene/info"
    while time.monotonic() < deadline:
        result = _run([gz_binary, "service", "-i", "--service", service], env=env, timeout=5)
        if result.returncode == 0 and "Service providers" in result.stdout:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Gazebo world did not expose {service}")


# 功能：
#   等待精确机型名称对应的话题，不接受另一个名称仅包含该字符串的飞行器。
# 输入：
#   gz_binary：Gazebo 程序；vehicle_name：精确模型名；env：隔离环境；timeout：等待秒数。
# 输出：
#   无。
def _wait_for_vehicle(
    gz_binary: str, vehicle_name: str, env: dict[str, str], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _run([gz_binary, "topic", "-l"], env=env, timeout=5)
        if result.returncode == 0 and any(
            "/model/" + vehicle_name + "/" in "/" + line.lstrip("/")
            for line in result.stdout.splitlines()
        ):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Gazebo did not expose vehicle {vehicle_name}")


# 功能：
#   在本轮世界按当前 SDF 生成实体，转义协议字符串并确认资产没有在生成过程中被替换。
# 输入：
#   gz_binary：Gazebo 程序；world_name、entity_name：世界与实体名。
#   sdf_path：实体资产；pose：世界坐标；env：本轮隔离环境。
# 输出：
#   evidence：生成回执、源文件摘要和位置。
def _spawn_entity(
    gz_binary: str,
    *,
    world_name: str,
    entity_name: str,
    sdf_path: Path,
    pose: tuple[float, float, float],
    env: dict[str, str],
) -> dict[str, object]:
    if len(pose) != 3:
        raise ValueError("spawn pose requires three coordinates")
    for value in pose:
        _runtime_number(value, "spawn coordinate", -1e6, 1e6)
    source_digest = _sha256(sdf_path)
    request = (
        f"sdf_filename: {json.dumps(str(sdf_path), ensure_ascii=False)} "
        f"name: {json.dumps(entity_name, ensure_ascii=False)} "
        f"pose {{ position {{ x: {pose[0]:.12g} y: {pose[1]:.12g} z: {pose[2]:.12g} }} }} "
        "allow_renaming: false"
    )
    result = _run(
        [
            gz_binary,
            "service",
            "-s",
            f"/world/{world_name}/create",
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            request,
        ],
        env=env,
        timeout=10,
    )
    if _sha256(sdf_path) != source_digest:
        raise SimulationRuntimeError("SDF_CHANGED_DURING_SPAWN")
    accepted = result.returncode == 0 and bool(re.fullmatch(r"\s*data:\s*true\s*", result.stdout))
    evidence = {
        "accepted": accepted,
        "entity_name": entity_name,
        "sdf_path": str(sdf_path),
        "sdf_sha256": source_digest,
        "pose_enu_m": pose,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip()[:1000],
        "stderr": result.stderr.strip()[:1000],
    }
    if not accepted:
        raise SimulationRuntimeError(f"Gazebo rejected entity spawn: {evidence}")
    return evidence


# 功能：
#   解析可拆卸挂载点的原生布尔或文字状态，未知格式不猜测已经脱离。
# 输入：
#   raw_state：Gazebo 状态消息文本。
# 输出：
#   detached：是否脱离；没有明确状态时为 None。
def _parse_payload_detached_state(raw_state: str) -> bool | None:
    boolean = re.search(r"data:\s*(true|false)\b", raw_state, flags=re.IGNORECASE)
    if boolean is not None:
        detached = boolean.group(1).casefold() == "true"
        return detached
    named = re.search(
        r"data:\s*[\"']?(detached|attached)[\"']?",
        raw_state,
        flags=re.IGNORECASE,
    )
    if named is not None:
        detached = named.group(1).casefold() == "detached"
        return detached
    return None


# 功能：
#   起飞前请求卸载并等待真实状态消息确认，防止预先挂载载荷冒充途中取件。
# 输入：
#   gz_binary：Gazebo 程序；detach_topic、output_topic：卸载命令和独立状态话题。
#   env：运行环境；timeout：有限的阶段等待秒数。
# 输出：
#   evidence：卸载请求与独立状态回执。
def _detach_payload_before_flight(
    gz_binary: str,
    *,
    detach_topic: str,
    output_topic: str,
    env: dict[str, str],
    timeout: float = 15.0,
) -> dict[str, object]:
    """Fail closed unless Gazebo confirms that the payload starts detached.

    Gazebo Harmonic's DetachableJoint system starts in the attached state.  The
    mission payload is spawned at its pickup checkpoint, so leaving that
    default in place creates a fixed joint from the vehicle to a remote object
    before PX4 starts.  Publish a preflight detach and require the plugin's
    state topic to confirm ``data: true`` (the plugin's detached-state value).
    """

    _runtime_number(timeout, "payload detach timeout", 0.001, 300)
    deadline = time.monotonic() + timeout
    required_topics = {detach_topic, output_topic}
    observed_topics: set[str] = set()
    while time.monotonic() < deadline:
        result = _run([gz_binary, "topic", "-l"], env=env, timeout=5)
        observed_topics = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        if result.returncode == 0 and required_topics <= observed_topics:
            break
        time.sleep(0.2)
    else:
        missing = sorted(required_topics - observed_topics)
        raise SimulationRuntimeError(f"payload preflight detach topics were not ready: {missing}")

    command_deadline = time.monotonic() + max(9.0, timeout)
    attempts: list[dict[str, object]] = []
    for attempt in range(1, 4):
        remaining = command_deadline - time.monotonic()
        if remaining <= 0:
            break
        listener = subprocess.Popen(
            [gz_binary, "topic", "-e", "-t", output_topic, "-n", "1"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        publisher: subprocess.CompletedProcess[str] | None = None
        stdout = ""
        stderr = ""
        try:
            time.sleep(0.08)
            try:
                publisher = _run(
                    [
                        gz_binary,
                        "topic",
                        "-t",
                        detach_topic,
                        "-m",
                        "gz.msgs.Empty",
                        "-p",
                        "",
                    ],
                    env=env,
                    timeout=min(3.0, max(0.1, remaining)),
                )
            except subprocess.TimeoutExpired as error:
                publisher = subprocess.CompletedProcess(
                    list(error.cmd),
                    124,
                    (error.stdout or "") if isinstance(error.stdout, str) else "",
                    (error.stderr or "") if isinstance(error.stderr, str) else "",
                )
            if publisher.returncode == 0:
                try:
                    stdout, stderr = listener.communicate(
                        timeout=min(3.0, max(0.1, command_deadline - time.monotonic()))
                    )
                except subprocess.TimeoutExpired:
                    _terminate(listener)
                    stdout, stderr = listener.communicate(timeout=3.0)
            else:
                _terminate(listener)
                stdout, stderr = listener.communicate(timeout=3.0)
        finally:
            primary = sys.exception()
            errors = []
            _cleanup_runtime_resource(errors, _terminate, listener)
            _cleanup_runtime_resource(errors, listener.communicate, timeout=3.0)
            _finish_runtime_cleanup(errors, primary)

        raw_state = stdout.strip()
        detached_state = _parse_payload_detached_state(raw_state)
        confirmed = (
            publisher is not None
            and publisher.returncode == 0
            and listener.returncode == 0
            and detached_state is True
        )
        attempt_evidence = {
            "attempt": attempt,
            "confirmed_detached": confirmed,
            "publisher_exit_code": None if publisher is None else publisher.returncode,
            "publisher_stdout": "" if publisher is None else publisher.stdout.strip()[:1000],
            "publisher_stderr": "" if publisher is None else publisher.stderr.strip()[:1000],
            "listener_exit_code": listener.returncode,
            "raw_state": raw_state[:1000],
            "listener_stderr": stderr.strip()[:1000],
        }
        attempts.append(attempt_evidence)
        if confirmed:
            return {
                "confirmed": True,
                "detached": True,
                "detach_topic": detach_topic,
                "output_topic": output_topic,
                "attempts": attempts,
            }
        time.sleep(0.15)

    raise SimulationRuntimeError(
        f"Gazebo did not confirm a detached payload before PX4 startup: {attempts}"
    )


# 功能：
#   从当前动作合同、检查点和所选机型挂载偏移解析载荷生成位置，不沿用旧地图中的负载参数。
# 输入：
#   vehicle_sdf：所选机型；runtime_action_contract_path：动作合同。
#   checkpoint_contract_path：检查点合同。
#   track：本次已批准的轨迹。
# 输出：
#   specification：载荷实体、资产、位置和通信话题；没有取件动作时为 None。
def _payload_spawn_spec(
    *,
    vehicle_sdf: Path,
    runtime_action_contract_path: Path | None,
    checkpoint_contract_path: Path | None,
    track: Px4Track,
) -> dict[str, object] | None:
    """Resolve a physical payload spawn from the bound runtime-action checkpoint.

    A DetachableJoint declaration only describes how two existing Gazebo models
    connect; it does not create the child model. Spawn only when an accepted
    runtime-action contract actually contains an attach operation, and bind the
    location to that action's checkpoint rather than a free-form coordinate.
    """

    if runtime_action_contract_path is None:
        return None
    if checkpoint_contract_path is None:
        raise SimulationRuntimeError("payload runtime action lacks a checkpoint contract")
    actions = _runtime_contract(runtime_action_contract_path, RuntimeActionExecutionContract)
    attach_steps = [
        step
        for step in actions.steps
        if step.driver == "gazebo-payload" and str(step.parameters.get("operation")) == "attach"
    ]
    if not attach_steps:
        return None
    step = attach_steps[0]
    if step.checkpoint_id is None:
        raise SimulationRuntimeError("Gazebo payload attach action is not checkpoint-bound")
    checkpoints = _runtime_contract(checkpoint_contract_path, RuntimeCheckpointContract)
    if checkpoints.contract_id != actions.contract_id:
        raise SimulationRuntimeError("payload action and checkpoint contract IDs differ")
    checkpoint = next(
        (item for item in checkpoints.checkpoints if item.checkpoint_id == step.checkpoint_id),
        None,
    )
    if checkpoint is None or checkpoint.track_point_index >= len(track.source_world_points):
        raise SimulationRuntimeError("payload action checkpoint is absent from the PX4 track")

    try:
        vehicle_root = ElementTree.parse(vehicle_sdf).getroot()
    except (ElementTree.ParseError, OSError) as error:
        raise SimulationRuntimeError("vehicle SDF is invalid while resolving payload") from error
    detachable = next(
        (
            item
            for item in vehicle_root.iter()
            if item.tag.rsplit("}", 1)[-1] == "plugin"
            and "detachable"
            in f"{item.attrib.get('name', '')} {item.attrib.get('filename', '')}".casefold()
        ),
        None,
    )
    if detachable is None:
        raise SimulationRuntimeError("payload action requires a DetachableJoint plugin")
    plugin_values = {
        item.tag.rsplit("}", 1)[-1]: (item.text or "").strip()
        for item in detachable.iter()
        if (item.text or "").strip()
    }
    child_model = plugin_values.get("child_model", "")
    if not child_model:
        raise SimulationRuntimeError("DetachableJoint child_model is missing")
    topics = {
        key: plugin_values.get(key, "") for key in ("attach_topic", "detach_topic", "output_topic")
    }
    missing_topics = [key for key, value in topics.items() if not value]
    if missing_topics:
        raise SimulationRuntimeError(
            f"DetachableJoint payload topics are missing: {missing_topics}"
        )

    payload_candidates: list[Path] = []
    for candidate in sorted(vehicle_sdf.parent.glob("*.sdf")):
        try:
            candidate_root = ElementTree.parse(candidate).getroot()
        except (ElementTree.ParseError, OSError):
            continue
        model_names = [
            item.attrib.get("name", "")
            for item in candidate_root.iter()
            if item.tag.rsplit("}", 1)[-1] == "model"
        ]
        if child_model in model_names:
            payload_candidates.append(candidate)
    if len(payload_candidates) != 1:
        raise SimulationRuntimeError(
            f"payload model {child_model!r} resolved to {len(payload_candidates)} SDF files"
        )

    # Payload geometry belongs to the selected vehicle/payload package, not to
    # the map.  The action contract is frozen from the current, hash-pinned
    # vehicle summary during planning.  Reading the historical map-side payload
    # binding here made an old value capable of silently overriding a newer
    # vehicle package whenever those two values diverged.
    mount_offset = step.parameters.get("payload_mount_offset_model_m")
    if (
        not isinstance(mount_offset, list)
        or len(mount_offset) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in mount_offset
        )
    ):
        raise SimulationRuntimeError("payload action mount offset is invalid")
    mount_east, mount_north, mount_up = (
        float(mount_offset[0]),
        float(mount_offset[1]),
        float(mount_offset[2]),
    )
    offset_east, offset_north, offset_up = (
        track.coordinate_contract.resolved_collision_center_offset_model_m()
    )
    target = track.source_world_points[checkpoint.track_point_index]
    pose = (
        target.east_m - offset_east + mount_east,
        target.north_m - offset_north + mount_north,
        target.up_m - offset_up + mount_up,
    )
    specification = {
        "entity_name": child_model,
        "sdf_path": payload_candidates[0],
        "pose": pose,
        "checkpoint_id": checkpoint.checkpoint_id,
        "track_point_index": checkpoint.track_point_index,
        "runtime_action_step_id": step.step_id,
        **topics,
    }
    return specification


# 功能：
#   为本次仿真复制独立 PX4 运行根目录，排除旧参数与日志，拒绝逃出所选 PX4 的符号链接。
# 输入：
#   px4_root：已安装的 PX4 源根；run_dir：本轮专属目录。
# 输出：
#   destination、executable：隔离根目录和实际 PX4 可执行文件。
def _prepare_rootfs(px4_root: Path, run_dir: Path) -> tuple[Path, Path]:
    build_root = px4_root / "build/px4_sitl_default"
    source_rootfs = build_root / "rootfs"
    source_etc = build_root / "etc"
    executable = build_root / "bin/px4"
    if not source_rootfs.is_dir() or not source_etc.is_dir() or not executable.is_file():
        raise FileNotFoundError("PX4 SITL build is missing rootfs, etc, or bin/px4")
    for source_tree in (source_rootfs, source_etc):
        for source in source_tree.rglob("*"):
            if not source.is_symlink():
                continue
            try:
                source.resolve(strict=True).relative_to(px4_root)
            except (OSError, ValueError) as exc:
                raise SimulationRuntimeError(f"unsafe PX4 rootfs symlink: {source}") from exc
    destination = run_dir / "px4_rootfs"
    shutil.copytree(
        source_rootfs,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "dataman", "eeprom", "log", "parameters.bson", "parameters_backup.bson"
        ),
    )
    shutil.copytree(source_etc, destination / "etc", symlinks=False, dirs_exist_ok=True)
    return destination, executable


# 功能：
#   从当前 SITL 机型读取唯一转子速度量程，不把驱动原生输出直接解释成通用推力。
# 输入：
#   px4_root：PX4 安装位置；px4_sitl_model：所选仿真机型。
# 输出：
#   reference：一致的正量程；缺失、非法或多量程混用时为 None。
def _px4_sitl_actuator_output_absolute_maximum(
    px4_root: Path,
    px4_sitl_model: str,
) -> float | None:
    """Read one unambiguous motor-output scale from the active SITL model.

    MAVLink ACTUATOR_OUTPUT_STATUS carries driver-native values without a
    universal unit.  Gazebo multicopters publish rotor angular-velocity
    commands, so payload inference may use them only after binding the values
    to the model's declared maxRotVelocity.  Mixed-propulsion models with
    different maxima deliberately return no scale and therefore fail closed.
    """

    model_sdf = px4_root / "Tools" / "simulation" / "gz" / "models" / px4_sitl_model / "model.sdf"
    if not model_sdf.is_file():
        return None
    try:
        root = ElementTree.parse(model_sdf).getroot()
    except (ElementTree.ParseError, OSError):
        return None
    try:
        maxima = [
            float(element.text or "nan")
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "maxRotVelocity"
        ]
    except (ValueError, OverflowError):
        return None
    if not maxima or any(not math.isfinite(value) or value <= 0.0 for value in maxima):
        return None
    reference = maxima[0]
    if any(not math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-6) for value in maxima):
        return None
    return reference


# 功能：
#   有界停止本轮拥有的独立进程组，必要时强制结束；该操作仅适用于仿真进程。
# 输入：
#   process：使用独立会话启动的子进程；未启动时为 None。
# 输出：
#   无。
def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    # 组长已退出不等于孙进程退出；所有调用方均用 start_new_session 创建独占组。
    try:
        os.kill(-process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    # 组长退出后同组进程仍可能持有传感器或日志管道，必须完成整组清理。
    with suppress(ProcessLookupError):
        os.kill(-process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    process.wait(timeout=5)


# 功能：
#   同时要求落地收尾状态和原生 ON_GROUND 观察，降落命令回执不能代替实际落地。
# 输入：
#   timing：执行器的本轮收尾与观察记录。
# 输出：
#   confirmed：是否具备落地状态证据。
def _landing_confirmed(timing: dict[str, Any]) -> bool:
    cleanup = timing.get("cleanup", {})
    confirmed = (
        isinstance(cleanup, dict)
        and str(cleanup.get("land", "")).startswith("confirmed_on_ground")
        and isinstance(cleanup.get("landing_observation"), dict)
        and cleanup["landing_observation"].get("state") == "ON_GROUND"
    )
    return confirmed


# 功能：
#   加载当前 ROS 工作区的真实构建环境，以空字节分隔读取变量，路径不拼入 shell 程序文本。
# 输入：
#   ros_workspace：本轮工作区；env：传入的隔离环境。
# 输出：
#   sourced：包含必要 ROS 库路径的完整子进程环境。
def _ros_overlay_environment(*, ros_workspace: Path, env: dict[str, str]) -> dict[str, str]:
    """Source the built ROS overlay and return its complete child environment."""

    setup = ros_workspace / "install" / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(f"ROS workspace setup is missing: {setup}")
    completed = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1" && env -0',
            "dronedream-ros-overlay",
            str(setup),
        ],
        env=env,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[:400]
        raise SimulationRuntimeError(f"ROS workspace setup failed: {detail}")
    sourced: dict[str, str] = {}
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        sourced[key.decode("utf-8")] = value.decode("utf-8")
    for required in ("AMENT_PREFIX_PATH", "LD_LIBRARY_PATH"):
        if not sourced.get(required):
            raise SimulationRuntimeError(
                f"ROS workspace setup did not define required environment: {required}"
            )
    return sourced


# 功能：
#   在调用方已经确认原生落地后发布绑定任务的终止通知，任务失败也可以安全收尾。
# 输入：
#   contract_id：任务合同；executor_return_code：执行器退出码；env：本轮 ROS 环境。
# 输出：
#   evidence：发布内容和 ROS 发布进程回执。
def _publish_native_terminal_lifecycle(
    *, contract_id: str, executor_return_code: int, env: dict[str, str]
) -> dict[str, object]:
    payload = {
        "contract_id": contract_id,
        "terminal_state": "ON_GROUND",
        "executor_return_code": executor_return_code,
        "landing_confirmed": True,
        "safe_to_stop_watchdog": True,
    }
    result = _run(
        [
            "ros2",
            "topic",
            "pub",
            "--once",
            "--wait-matching-subscriptions",
            "0",
            "/dronedream/mission_lifecycle",
            "dronedream_agent_msgs/msg/MissionLifecycle",
            json.dumps(payload, separators=(",", ":")),
        ],
        env=env,
        timeout=10,
    )
    if result.returncode != 0:
        raise SimulationRuntimeError(
            f"native terminal lifecycle publication failed: {result.stderr.strip()[:400]}"
        )
    evidence = {
        **payload,
        "publisher_exit_code": result.returncode,
        "publisher_stdout": result.stdout.strip()[:400],
    }
    return evidence


# 功能：
#   对每个线段作夹紧投影，求观测点到参考折线的最短欧氏距离。
# 输入：
#   point：当前世界位置；route：参考折线。
# 输出：
#   minimum：最短距离；无有效线段时为无穷大。
def _distance_to_polyline(
    point: tuple[float, float, float], route: list[tuple[float, float, float]]
) -> float:
    minimum = math.inf
    for start, end in zip(route, route[1:], strict=False):
        delta = tuple(end[index] - start[index] for index in range(3))
        squared = sum(value * value for value in delta)
        ratio = (
            0.0
            if squared <= 1e-18
            else max(
                0.0,
                min(
                    1.0,
                    sum((point[index] - start[index]) * delta[index] for index in range(3))
                    / squared,
                ),
            )
        )
        nearest = tuple(start[index] + ratio * delta[index] for index in range(3))
        minimum = min(minimum, math.dist(point, nearest))
    return minimum


# 功能：
#   起飞前约束在出生点，爬升时约束在起飞段，巡航时采用任务路线，避免把地面位置算作航路偏离。
# 输入：
#   phase：当前阶段；spawn_center：真实生成的碰撞中心；route：空中任务路线。
# 输出：
#   reference：与当前阶段匹配的监视折线。
def _phase_reference_polyline(
    *,
    phase: str | None,
    spawn_center: tuple[float, float, float],
    route: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Monitor the ground pose and climb separately from the airborne route.

    A two-metre takeoff is not a two-metre deviation while still on the ground.
    Do not disable monitoring: preflight remains bounded to the actual spawn,
    and takeoff to its segment to the declared launch point. Collision checks
    still inspect every pose against every primitive in every phase.
    """
    if not route:
        raise ValueError("reference route is empty")
    if phase is None or phase == "PREFLIGHT":
        reference = [spawn_center, spawn_center]
    elif phase == "TAKEOFF":
        reference = [spawn_center, route[0]]
    else:
        reference = list(route)
    return reference


# 功能：
#   只在任务飞行阶段累计离开起点与抵达目标的实测证据，闭合返程不能起飞前就判定完成。
# 输入：
#   center：实测碰撞中心；start、goal：任务起终点；departure_observed：既有离开证据。
#   phase：执行阶段；departure_distance_m：离开距离阈值；goal_tolerance_m：到达容差。
# 输出：
#   departure_observed、goal_observed：更新后的离开与到达标志。
def _update_goal_progress(
    *,
    center: tuple[float, float, float],
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    departure_observed: bool,
    departure_distance_m: float = 0.75,
    goal_tolerance_m: float = 0.6,
    phase: str | None = "TRACK",
) -> tuple[bool, bool]:
    """Advance the measured departure/arrival state for open or closed routes.

    A route that returns to its start must not satisfy its final-goal gate from
    the initial pose.  Requiring a measured departure before any arrival also
    gives open routes a single, explicit state machine instead of relying on a
    minimum-ever distance that can accidentally pass before flight begins.
    """

    if phase not in {"TRACK", "WAYPOINT_SETTLE", "CHECKPOINT", "ACTION", "HOLDING"}:
        return departure_observed, False
    if not departure_observed and math.dist(center, start) > departure_distance_m:
        departure_observed = True
    goal_observed = departure_observed and math.dist(center, goal) <= goal_tolerance_m
    return departure_observed, goal_observed


# 功能：
#   从当前机型声明的规范链接解析真实模型姿态和碰撞中心，不用任意包含机型名的实体替代。
# 输入：
#   poses：Gazebo 位姿列表；vehicle_name：当前机型；collision_center_offset_model_m：模型内偏移。
#   frames：已解析的所选机型坐标声明。
# 输出：
#   resolved：扣除声明偏移后的碰撞中心位置、实体名、坐标约定和候选名；缺规范链接时为 None。
def _resolve_controlled_vehicle_pose(
    poses: Any,
    *,
    vehicle_name: str,
    collision_center_offset_model_m: tuple[float, float, float],
    frames: dict,
) -> tuple[tuple[float, float, float], str, str, list[str]] | None:
    """Exact composed centre expressed in the existing evidence convention.

    Consumers add the declared offset to obtain the actual collision centre.
    The returned offset-adjusted position is NOT the physical wrapper origin.
    A missing canonical link cannot fall back to a potentially static wrapper.
    """
    if (
        frames["vehicle_model_name"] != vehicle_name
        or tuple(frames["collision_center_model_m"]) != collision_center_offset_model_m
    ):
        raise ValueError("SIMULATION_FRAME_OBSERVER_BINDING_MISMATCH")
    selected = select_canonical_poses(
        poses, vehicle_name=vehicle_name, canonical_link_name=frames["canonical_link_name"]
    )
    if selected is None:
        return None
    model, canonical, selected_name, candidate_names = selected
    center = collision_center_from_canonical(
        model_world=model,
        canonical_in_model=canonical,
        canonical_at_rest=frames["canonical_at_rest"],
        collision_center_model_m=collision_center_offset_model_m,
    )
    adjusted = tuple(center[i] - collision_center_offset_model_m[i] for i in range(3))
    resolved = (
        adjusted,
        selected_name,
        "calibrated-collision-center-offset-adjusted",
        candidate_names,
    )
    return resolved


# 功能：
#   明确选择当前开发源码或已安装工作进程，开发训练不得静默退回旧安装包。
# 输入：
#   packaged_worker：安装包入口；source_worker：当前源码入口；source_training：开发训练标志。
# 输出：
#   worker：应执行的深度工作进程路径。
def _select_depth_worker(
    *,
    packaged_worker: Path,
    source_worker: Path,
    source_training: bool,
) -> Path:
    """A source checkout must use the worker paired with its imported core.

    This includes non-training previews: a bundled worker may call removed
    private interfaces in the newer source package. An installed product with
    no checkout uses its bundle; explicit source training never falls back.
    """
    if source_worker.is_file():
        worker = source_worker
        return worker
    if source_training:
        raise ValueError("source training requires the current source sensor worker")
    worker = packaged_worker if packaged_worker.is_file() else source_worker
    return worker


# 功能：
#   将明确的机头保持或相对路线转向策略和有限角速度上限转换为执行器参数。
# 输入：
#   heading_policy：机头方向策略；maximum_yaw_rate_deg_s：每秒最大转向角度。
# 输出：
#   arguments：策略和角速度上限的命令行参数。
def _heading_executor_arguments(
    heading_policy: str,
    maximum_yaw_rate_deg_s: float,
) -> list[str]:
    if heading_policy not in {"measured-hold", "route-tangent-relative"}:
        raise ValueError("unsupported PX4 heading policy")
    if (
        type(maximum_yaw_rate_deg_s) not in (int, float)
        or not 1 <= maximum_yaw_rate_deg_s <= 90
        or not math.isfinite(maximum_yaw_rate_deg_s)
        or maximum_yaw_rate_deg_s < 1.0
        or maximum_yaw_rate_deg_s > 90.0
    ):
        raise ValueError("maximum yaw rate must be between 1 and 90 deg/s")
    arguments = [
        "--heading-policy",
        heading_policy,
        "--maximum-yaw-rate-deg-s",
        f"{maximum_yaw_rate_deg_s:g}",
    ]
    return arguments


# 功能：
#   1. 绑定当前地图、机型、轨迹、控制脚本与模型包，启动隔离的 Gazebo、PX4 和 ROS 链路。
#   2. 将原生传感器交给本地专家及安全控制层，独立观察位姿、碰撞、进展和真实落地。
#   3. 有界回收本轮资源，以真实控制、观察和文件证据形成验收结果，开发教师采集不授予产品资格。
# 输入：
#   world_sdf、semantic_path、vehicle_sdf、vehicle_metadata_path：本轮所选环境与机型资产。
#   route_path、track_path、clearance_path、controller_params_path：已批准路线、轨迹、净空和限制。
#   executor_path、px4_root、ros_workspace、run_dir：当前运行组件与独占输出目录。
#   checkpoint_contract_path、runtime_action_contract_path、contract_id：任务检查点和动作绑定。
#   local_navigation_*、local_policy_*：模型供应方、权限、期限及当前模型包和资格凭据。
#   simulation_*、native_sensor_runtime、render_*：原生传感器、渲染和显式开发训练配置。
#   multimodal_*、semantic_label_*、record_learning_observations：真实观察记录及离线监督配置。
#   development_*：故障或载荷开发实验开关，启用时不能获得正式运行资格。
#   其他关键字参数：显式机型名、观察视角、资源隔离和有界超时配置。
# 输出：
#   evidence：严格校验的真实运行门控、观测和文件摘要。
@_isolate_transport_environment
def run_px4_gazebo_track(
    *,
    run_dir: Path,
    world_sdf: Path,
    semantic_path: Path,
    vehicle_sdf: Path,
    route_path: Path,
    track_path: Path,
    clearance_path: Path,
    controller_params_path: Path,
    px4_root: Path,
    executor_path: Path,
    ros_workspace: Path,
    vehicle_metadata_path: Path,
    contract_id: str = "runtime-contract",
    world_name: str | None = None,
    vehicle_name: str | None = None,
    px4_sitl_model: str = "x500",
    executor_extra_args: list[str] | None = None,
    checkpoint_contract_path: Path | None = None,
    runtime_action_contract_path: Path | None = None,
    live_camera_enabled: bool = True,
    local_navigation_provider: str | None = None,
    local_navigation_fallback_provider: str | None = None,
    local_navigation_model_timeout_seconds: float = 10.0,
    local_navigation_fallback_model_timeout_seconds: float = 10.0,
    local_navigation_period_seconds: float = 3.0,
    local_navigation_context_id: str | None = None,
    local_navigation_visual_enabled: bool = False,
    local_navigation_control_authority_required: bool = False,
    local_navigation_omit_coordinate_candidates: bool = False,
    heading_policy: str = "measured-hold",
    maximum_yaw_rate_deg_s: float = 20.0,
    multimodal_dataset_root: Path | None = None,
    multimodal_flight_id: str | None = None,
    semantic_label_topic: str | None = None,
    semantic_label_map_path: Path | None = None,
    multimodal_dataset_maximum_mib: int = 5_120,
    multimodal_record_period_seconds: float = 0.1,
    local_policy_package_paths: tuple[Path, ...] = (),
    local_policy_qualification_paths: tuple[Path, ...] = (),
    local_policy_simulation_admission_paths: tuple[Path, ...] = (),
    development_depth_drop_after_seconds: float | None = None,
    development_depth_drop_duration_seconds: float | None = None,
    development_payload_collection: bool = False,
    record_learning_observations: bool = False,
    learning_image_size: tuple[int, int] = (224, 128),
    simulation_teacher_control: bool = False,
    simulation_training_channel: Path | None = None,
    batch_static_world_visuals: bool = False,
    simulation_camera_profile: str = "native",
    camera_source_model_sha256: str | None = None,
    preflight_render_warmup: bool = False,
    preflight_depth_warmup: bool = False,
    render_preparation_runtime: Path | None = None,
    render_cache_bundle: Path | None = None,
    render_replica_runtime: Path | None = None,
    native_sensor_runtime: Path | None = None,
) -> dict[str, object]:
    """Execute one route; success requires runtime state and evidence, not process exit alone."""

    executor_options = _validated_executor_options(executor_extra_args)
    heading_policy = executor_options.get("--heading-policy", heading_policy)
    if "--maximum-yaw-rate-deg-s" in executor_options:
        maximum_yaw_rate_deg_s = float(executor_options["--maximum-yaw-rate-deg-s"])
    for name, value in {
        "live_camera_enabled": live_camera_enabled,
        "local_navigation_visual_enabled": local_navigation_visual_enabled,
        "local_navigation_control_authority_required": local_navigation_control_authority_required,
        "local_navigation_omit_coordinate_candidates": local_navigation_omit_coordinate_candidates,
        "development_payload_collection": development_payload_collection,
        "record_learning_observations": record_learning_observations,
        "simulation_teacher_control": simulation_teacher_control,
        "batch_static_world_visuals": batch_static_world_visuals,
        "preflight_render_warmup": preflight_render_warmup,
        "preflight_depth_warmup": preflight_depth_warmup,
    }.items():
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    for value in (
        local_navigation_model_timeout_seconds,
        local_navigation_fallback_model_timeout_seconds,
        local_navigation_period_seconds,
    ):
        _runtime_number(value, "local navigation timeout/period", 0.000001, 3600)
    if development_depth_drop_after_seconds is not None:
        _runtime_number(development_depth_drop_after_seconds, "depth drop delay", 0, 86400)
    if development_depth_drop_duration_seconds is not None:
        _runtime_number(
            development_depth_drop_duration_seconds, "depth drop duration", 0.001, 86400
        )
    if type(multimodal_dataset_maximum_mib) is not int:
        raise ValueError("multimodal dataset quota must be an integer")
    _runtime_number(multimodal_record_period_seconds, "recording period", 0.05, 10)
    check_plain_plugin_path(run_dir)
    validate_camera_profile_choice(simulation_camera_profile, camera_source_model_sha256)
    if render_replica_runtime is not None and (
        simulation_training_channel is None
        or not local_navigation_visual_enabled
        or simulation_camera_profile == "native"
        or live_camera_enabled
        or preflight_render_warmup
        or preflight_depth_warmup
        or render_preparation_runtime is not None
        or render_cache_bundle is not None
        or semantic_label_topic is not None
    ):
        raise ValueError(
            "isolated rendering requires exclusive source-bound onboard visual training"
        )
    if render_cache_bundle is not None and render_preparation_runtime is None:
        raise ValueError("render cache input requires its verified native runtime")
    if render_preparation_runtime is not None and not preflight_render_warmup:
        raise ValueError("render cache requires explicit preflight preparation")
    if preflight_depth_warmup and not preflight_render_warmup:
        raise ValueError("depth warmup requires explicit preflight render warmup")
    if preflight_render_warmup and (
        simulation_camera_profile == "native" or simulation_training_channel is None
    ):
        raise ValueError("render warmup requires explicit source-bound visual training")
    if simulation_camera_profile != "native" and (
        simulation_training_channel is None or not local_navigation_visual_enabled
    ):
        raise ValueError(
            "camera stream profile is restricted to explicit visual simulation training"
        )

    if not px4_sitl_model or not px4_sitl_model.replace("_", "").replace("-", "").isalnum():
        raise ValueError("PX4 SITL model identifier is invalid")
    if batch_static_world_visuals and semantic_label_topic is not None:
        raise ValueError("static render batching cannot remap semantic instance labels")
    if local_navigation_model_timeout_seconds <= 0.0:
        raise ValueError("local navigation model timeout must be positive")
    if local_navigation_fallback_model_timeout_seconds <= 0.0:
        raise ValueError("local navigation fallback model timeout must be positive")
    if local_navigation_period_seconds <= 0.0:
        raise ValueError("local navigation period must be positive")
    heading_executor_arguments = _heading_executor_arguments(
        heading_policy,
        maximum_yaw_rate_deg_s,
    )
    if (development_depth_drop_after_seconds is None) != (
        development_depth_drop_duration_seconds is None
    ):
        raise ValueError("development depth-drop injection requires after and duration")
    if (
        development_depth_drop_after_seconds is not None
        and development_depth_drop_after_seconds < 0.0
    ):
        raise ValueError("development depth-drop delay must be non-negative")
    if (
        development_depth_drop_duration_seconds is not None
        and development_depth_drop_duration_seconds <= 0.0
    ):
        raise ValueError("development depth-drop duration must be positive")
    if bool(simulation_training_channel) != (local_navigation_provider == "simulation-training"):
        raise ValueError("simulation learner requires an explicit run-bound policy channel")
    if simulation_training_channel is not None and (
        not local_navigation_control_authority_required
        or local_navigation_fallback_provider
        or local_policy_package_paths
        or local_policy_qualification_paths
        or local_policy_simulation_admission_paths
        or simulation_teacher_control
    ):
        raise ValueError("simulation learner authority must not mix with other providers")
    if record_learning_observations and vehicle_metadata_path is None:
        raise ValueError("learning observation collection requires the actual vehicle metadata")
    if simulation_teacher_control and (
        not record_learning_observations
        or local_navigation_provider is not None
        or local_navigation_control_authority_required
    ):
        raise ValueError(
            "simulation teacher requires observation recording without model authority"
        )
    if len(learning_image_size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 64 <= value <= 1024
        for value in learning_image_size
    ):
        raise ValueError("learning image dimensions must be in [64, 1024]")
    if (
        local_navigation_visual_enabled
        and local_navigation_provider is None
        and not record_learning_observations
    ):
        raise ValueError("visual local navigation requires a model provider")
    if local_navigation_visual_enabled:
        from .model_image_cache import require_model_image_runtime

        require_model_image_runtime()
    if (
        local_navigation_visual_enabled
        and heading_policy != "route-tangent-relative"
        and not (
            local_navigation_provider == "local-policy"
            and local_navigation_control_authority_required
        )
    ):
        raise ValueError(
            "forward visual navigation requires route-tangent-relative PX4 heading control"
        )
    if local_navigation_control_authority_required and local_navigation_provider is None:
        raise ValueError("model control authority requires a model provider")
    if (
        local_navigation_omit_coordinate_candidates
        and local_navigation_provider is None
        and not record_learning_observations
    ):
        raise ValueError("omitting coordinate candidates requires a model provider")
    if (multimodal_dataset_root is None) != (multimodal_flight_id is None):
        raise ValueError("multimodal recording requires dataset root and flight identity")
    if (
        multimodal_flight_id is not None
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", multimodal_flight_id) is None
    ):
        raise ValueError("multimodal flight identity must be normalized lowercase")
    if (semantic_label_topic is None) != (semantic_label_map_path is None):
        raise ValueError("semantic supervision requires topic and label map")
    if semantic_label_topic is not None and multimodal_dataset_root is None:
        raise ValueError("semantic supervision requires multimodal recording")
    if not 1 <= multimodal_dataset_maximum_mib <= 20 * 1024:
        raise ValueError("multimodal dataset quota is outside the safe range")
    if not 0.05 <= multimodal_record_period_seconds <= 10.0:
        raise ValueError("multimodal recording period is outside the safe range")
    if (
        local_navigation_fallback_provider is not None
        and local_navigation_fallback_provider == local_navigation_provider
    ):
        raise ValueError("local navigation fallback provider must differ from primary")
    if local_navigation_provider == "local-policy":
        if not local_policy_package_paths or not (
            local_policy_qualification_paths or local_policy_simulation_admission_paths
        ):
            raise ValueError(
                "local policy navigation requires packages and qualification "
                "or simulation admission receipts"
            )
    elif (
        local_policy_package_paths
        or local_policy_qualification_paths
        or local_policy_simulation_admission_paths
    ):
        raise ValueError("local policy artifacts require the local-policy provider")
    if development_payload_collection:
        if local_navigation_provider != "local-policy":
            raise ValueError("development payload collection requires local-policy")
        if not local_policy_simulation_admission_paths or local_policy_qualification_paths:
            raise ValueError("development payload collection requires simulation admission only")
        if multimodal_dataset_root is None:
            raise ValueError("development payload collection requires multimodal recording")
    runtime_cpu_affinity = _runtime_cpu_affinity_plan(
        local_policy_enabled=local_navigation_provider == "local-policy",
    )

    if run_dir.exists():
        unexpected = [path for path in run_dir.iterdir() if path.name != "runtime-control"]
        if unexpected:
            raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    native_preflight_marker = run_dir / "native-runtime-preflight-ready.json"
    native_preflight_marker.unlink(missing_ok=True)
    for required in (
        world_sdf,
        semantic_path,
        vehicle_sdf,
        route_path,
        track_path,
        clearance_path,
        controller_params_path,
        executor_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if checkpoint_contract_path is not None and not checkpoint_contract_path.is_file():
        raise FileNotFoundError(checkpoint_contract_path)
    if runtime_action_contract_path is not None and not runtime_action_contract_path.is_file():
        raise FileNotFoundError(runtime_action_contract_path)
    if multimodal_dataset_root is not None and multimodal_dataset_root.exists():
        raise FileExistsError(multimodal_dataset_root)
    if semantic_label_map_path is not None and not semantic_label_map_path.is_file():
        raise FileNotFoundError(semantic_label_map_path)
    if not vehicle_metadata_path.is_file():
        raise FileNotFoundError(vehicle_metadata_path)
    for local_policy_path in (
        *local_policy_package_paths,
        *local_policy_qualification_paths,
        *local_policy_simulation_admission_paths,
    ):
        if not local_policy_path.exists():
            raise FileNotFoundError(local_policy_path)
    if local_navigation_provider == "local-policy":
        _require_local_policy_runtime()

    executor_help = _run(
        [sys.executable, str(executor_path), "--help"],
        env=os.environ.copy(),
        timeout=15,
    )
    required_executor_flags = {
        "--abort-file",
        "--landing-timeout-seconds",
        "--takeoff-climb-rate-m-s",
        "--takeoff-stable-window-seconds",
    }
    if local_navigation_control_authority_required:
        required_executor_flags.update(
            {"--model-progress-slack-m", "--require-model-control-authority"}
        )
    missing_flags = sorted(
        flag for flag in required_executor_flags if flag not in executor_help.stdout
    )
    if executor_help.returncode != 0 or missing_flags:
        raise SimulationRuntimeError(
            f"PX4 executor lacks required closed-loop flags: {missing_flags}"
        )

    input_paths = [
        world_sdf,
        semantic_path,
        vehicle_sdf,
        route_path,
        track_path,
        clearance_path,
        controller_params_path,
        executor_path,
        vehicle_metadata_path,
    ]
    if "--base-executor" in executor_options:
        input_paths.append(Path(executor_options["--base-executor"]))
    if checkpoint_contract_path is not None:
        input_paths.append(checkpoint_contract_path)
    if runtime_action_contract_path is not None:
        input_paths.append(runtime_action_contract_path)
    input_hashes = {path: _sha256(path) for path in input_paths}
    world_name = _sdf_entity_name(world_sdf, "world", world_name)
    vehicle_name = _sdf_entity_name(vehicle_sdf, "model", vehicle_name)
    route = _runtime_contract(route_path, GraphRoute)
    track = _runtime_contract(track_path, Px4Track)
    clearance = _runtime_contract(clearance_path, RouteClearanceReport)
    if not clearance.accepted or clearance.route_sha256 != sha256_json(route):
        raise SimulationRuntimeError("route is not bound to an accepted static-clearance report")
    if len(route.positions_m) != len(track.source_world_points):
        raise SimulationRuntimeError("route and PX4 track point counts differ")
    if any(
        (point.x, point.y, point.z) != (world.east_m, world.north_m, world.up_m)
        for point, world in zip(route.positions_m, track.source_world_points, strict=True)
    ):
        raise SimulationRuntimeError("route and PX4 world points differ")
    if clearance.semantic_sha256 != input_hashes[semantic_path]:
        raise SimulationRuntimeError("clearance references a different semantic asset")

    semantic = decode_json(
        read_plugin_file(semantic_path, limit=64 * 1024**2),
        limit=64 * 1024**2,
        node_limit=2_000_000,
    )
    bindings = load_map_runtime_bindings(semantic)
    primitives = semantic["collision_primitives"]
    if not isinstance(primitives, list) or not primitives or len(primitives) > 100_000:
        raise SimulationRuntimeError("STATIC_COLLISION_GEOMETRY_INVALID")
    primitives = [_validated_primitive(item) for item in primitives]
    try:
        selected_vehicle = _runtime_contract(vehicle_metadata_path, VehicleAsset)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SimulationRuntimeError("selected vehicle metadata is invalid") from error
    diameter = selected_vehicle.body_radius_m * 2.0
    height = selected_vehicle.body_height_m
    offset_east, offset_north, offset_up = (
        track.coordinate_contract.resolved_collision_center_offset_model_m()
    )
    model_root = tuple(track.coordinate_contract.model_root_world_enu_m)
    if len(model_root) != 3:
        raise SimulationRuntimeError("PX4 model-root contract is invalid")
    spawn = bindings.vehicle_spawn
    collision_offset = resolve_vehicle_collision_center_offset(selected_vehicle)
    if model_root != (spawn.x, spawn.y, spawn.z) or (offset_east, offset_north, offset_up) != (
        collision_offset.x,
        collision_offset.y,
        collision_offset.z,
    ):
        raise SimulationRuntimeError(
            "runtime spawn or collision offset differs from selected assets"
        )
    payload_spawn = _payload_spawn_spec(
        vehicle_sdf=vehicle_sdf,
        runtime_action_contract_path=runtime_action_contract_path,
        checkpoint_contract_path=checkpoint_contract_path,
        track=track,
    )

    trial_rootfs, px4_executable = _prepare_rootfs(px4_root, run_dir)
    copied_track = run_dir / "reference_track.json"
    copied_params = run_dir / "controller_params.json"
    hash_plugin_file(track_path, limit=64 * 1024**2, destination=copied_track)
    hash_plugin_file(controller_params_path, limit=64 * 1024**2, destination=copied_params)
    if (
        _sha256(copied_track) != input_hashes[track_path]
        or _sha256(copied_params) != input_hashes[controller_params_path]
    ):
        raise SimulationRuntimeError("RUNTIME_STAGED_ASSET_CHANGED")
    controller_max_speed_mps, local_max_acceleration_mps2 = _load_controller_limits(copied_params)
    local_max_speed_mps = _effective_local_speed_limit_mps(
        controller_speed_limit_mps=controller_max_speed_mps,
        track=track,
    )
    local_safety_supported = "--local-safety-command" in executor_help.stdout
    tracking_identity_supported = "--tracking-lag-limit-m" in executor_help.stdout
    local_safety_target_path = run_dir / "local-safety-target.json"
    local_safety_observation_path = run_dir / "local-safety-observation.json"
    local_safety_command_path = run_dir / "local-safety-command.json"
    depth_safety_observation_path = run_dir / "depth-local-safety-observation.json"
    depth_safety_command_path = run_dir / "depth-local-safety-command.json"
    depth_safety_history_path = run_dir / "depth-local-safety-history.jsonl"
    depth_perception_health_path = run_dir / "depth-perception-health.json"
    runtime_evidence_writer_summary_path = run_dir / "runtime-evidence-writer-summary.json"
    runtime_snapshot_writer_summary_path = run_dir / "runtime-snapshot-writer-summary.json"
    development_depth_fault_path = run_dir / "development-depth-fault.json"
    packaged_depth_worker_script = executor_path.with_name("runtime_depth_safety_worker.py")
    source_depth_worker_script = (
        Path(__file__).resolve().parents[2] / "scripts" / ("runtime_depth_safety_worker.py")
    )
    source_training = simulation_training_channel is not None or simulation_teacher_control
    depth_worker_script = _select_depth_worker(
        packaged_worker=packaged_depth_worker_script,
        source_worker=source_depth_worker_script,
        source_training=source_training,
    )
    _write_json(
        run_dir / "depth-worker-source.json",
        {
            "path": str(depth_worker_script.resolve()),
            "sha256": _sha256(depth_worker_script) if depth_worker_script.is_file() else None,
            "source_training": source_training,
        },
    )
    depth_safety_supported = bool(
        local_safety_supported
        and depth_worker_script.is_file()
        and "model://x500_depth" in vehicle_sdf.read_text(encoding="utf-8")
    )
    if record_learning_observations and not depth_safety_supported:
        raise ValueError("learning observations require the native depth safety worker")
    if depth_safety_supported and "--local-safety-channel" not in executor_help.stdout:
        raise ValueError(
            "native sensor control requires the current atomic safety channel executor"
        )
    if depth_safety_supported and "--native-state-channel" not in executor_help.stdout:
        raise ValueError("native sensor control requires the current native state channel executor")
    if depth_safety_supported and "--perception-health-channel" not in executor_help.stdout:
        raise ValueError(
            "native sensor control requires the current perception health channel executor")
    if "--runtime-phase-channel" not in executor_help.stdout:
        raise ValueError("simulation requires the current live phase channel executor")
    _require_local_navigation_sensor_runtime(
        local_navigation_provider=local_navigation_provider,
        depth_safety_supported=depth_safety_supported,
    )
    active_local_safety_observation_path = (
        depth_safety_observation_path if depth_safety_supported else local_safety_observation_path
    )
    active_local_safety_command_path = (
        depth_safety_command_path if depth_safety_supported else local_safety_command_path
    )
    tracking_corridor_policy = _tracking_corridor_policy(clearance.minimum_clearance_m)
    segment_clearances = list(clearance.segment_minimum_clearances_m)
    if segment_clearances and len(segment_clearances) != len(route.positions_m) - 1:
        raise SimulationRuntimeError(
            "route clearance segment budgets do not match the route topology"
        )
    segment_tracking_policies = [
        {
            "segment_index": segment_index,
            "control_profile": (
                "precision" if segment_clearance_m < PREFERRED_TRANSIT_CLEARANCE_M else "cruise"
            ),
            **_tracking_corridor_policy(segment_clearance_m),
        }
        for segment_index, segment_clearance_m in enumerate(segment_clearances)
    ]
    local_required_clearance_m = tracking_corridor_policy["required_local_clearance_m"]
    model_navigation_controller_step_m = _model_navigation_controller_lookahead_m(
        maximum_speed_mps=local_max_speed_mps,
    )
    identity_correction_limit_m = _identity_correction_limit_m(clearance.minimum_clearance_m)
    copied_checkpoint_contract: Path | None = None
    if checkpoint_contract_path is not None:
        copied_checkpoint_contract = run_dir / "runtime-checkpoints.json"
        hash_plugin_file(
            checkpoint_contract_path, limit=64 * 1024**2, destination=copied_checkpoint_contract
        )
        if _sha256(copied_checkpoint_contract) != input_hashes[checkpoint_contract_path]:
            raise SimulationRuntimeError("RUNTIME_STAGED_CHECKPOINT_CHANGED")

    env = os.environ.copy()
    partition = f"dronedream_agent_{os.getpid()}_{int(time.time())}"
    state = run_dir / "runtime-state"
    for name in ("cache", "config", "data"):
        (state / name).mkdir(parents=True, exist_ok=True)
    _write_json(
        state / "tracking-corridor-policy.json",
        {
            "schema_version": "dronedream.tracking-corridor-policy.v1",
            "track_sha256": sha256_json(track),
            "route_sha256": sha256_json(route),
            "segment_policies": segment_tracking_policies,
            **tracking_corridor_policy,
        },
    )
    px4_plugins = px4_root / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
    px4_server_config = px4_root / "src/modules/simulation/gz_bridge/server.config"
    route_content = read_plugin_file(route_path, limit=64 * 1024**2)
    if GraphRoute.model_validate_json(route_content) != route:
        raise SimulationRuntimeError("mission route changed before simulator launch")
    with (run_dir / "mission-route.json").open("xb") as route_copy:
        route_copy.write(route_content)
    render_world = world_sdf
    render_batching = None
    camera_profile = None
    if simulation_camera_profile != "native":
        camera_profile = prepare_camera_profile(
            source_models=px4_root / "Tools/simulation/gz/models",
            output=run_dir / "camera-profile",
            expected_source_sha256=camera_source_model_sha256,
            profile=simulation_camera_profile,
        )
    if batch_static_world_visuals:
        render_world, render_batching = prepare_static_render_world(
            world_sdf, run_dir / "static-render-world"
        )
    resource_roots = [
        *([Path(camera_profile["model_resource_root"])] if camera_profile else []),
        render_world.parent,
        world_sdf.parent,
        vehicle_sdf.parent / "models",
        vehicle_sdf.parent.parent,
        px4_root / "Tools/simulation/gz/models",
        px4_root / "Tools/simulation/gz/worlds",
    ]
    sensor_frames = inspect_simulation_sensor_frames(
        vehicle_sdf,
        [p for p in resource_roots if p.is_dir()],
        collision_center_model_m=[offset_east, offset_north, offset_up],
        require_depth=depth_safety_supported,
    )
    _write_json(run_dir / "simulation-sensor-frames.json", sensor_frames)
    env.update(
        {
            "GZ_PARTITION": partition,
            "GZ_CONFIG_PATH": f"/usr/share/gz:{env.get('GZ_CONFIG_PATH', '')}",
            "GZ_SIM_RESOURCE_PATH": ":".join(
                (
                    *((camera_profile["model_resource_root"],) if camera_profile else ()),
                    str(render_world.parent),
                    str(world_sdf.parent),
                    str(vehicle_sdf.parent / "models"),
                    str(vehicle_sdf.parent.parent),
                    str(px4_root / "Tools/simulation/gz/models"),
                    str(px4_root / "Tools/simulation/gz/worlds"),
                )
            ),
            "GZ_SIM_SYSTEM_PLUGIN_PATH": str(px4_plugins),
            "GZ_SIM_SERVER_CONFIG_PATH": str(px4_server_config),
            "HEADLESS": "1",
            "PX4_GZ_STANDALONE": "1",
            "PX4_GZ_MODEL_NAME": vehicle_name,
            "PX4_GAZEBO_WORLD_NAME": world_name,
            "PX4_SYS_AUTOSTART": _px4_autostart_id(trial_rootfs, px4_sitl_model),
            "GZ_IP": "127.0.0.1",
            "PYTHONUNBUFFERED": "1",
            "XDG_CACHE_HOME": str(state / "cache"),
            "XDG_CONFIG_HOME": str(state / "config"),
            "XDG_DATA_HOME": str(state / "data"),
            "ROS_DOMAIN_ID": env.get("ROS_DOMAIN_ID", "74"),
            "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
            "ROS2_DISABLE_DAEMON": "1",
            "CYCLONEDDS_URI": (
                '<CycloneDDS><Domain Id="any"><General><Interfaces>'
                '<NetworkInterface address="127.0.0.1"/></Interfaces>'
                "<AllowMulticast>false</AllowMulticast></General><Discovery>"
                "<ParticipantIndex>auto</ParticipantIndex><Peers>"
                '<Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
            ),
        }
    )
    sensor_deployment = None
    if depth_safety_supported and (
        simulation_teacher_control or local_navigation_control_authority_required
    ):
        sensor_root = native_sensor_runtime or Path(
            env.get(
                "DRONEDREAM_NATIVE_SENSOR_RUNTIME", str(executor_path.parent / "native-sensors")
            )
        )
        sensor_deployment = prepare_sensor_runtime(
            world_sdf=world_sdf,
            px4_root=px4_root,
            runtime_root=sensor_root,
            output=run_dir / "native-sensors",
            source_root=(
                Path(__file__).resolve().parents[2] / "native/gazebo_sensors"
                if source_training
                else None
            ),
        )
        env.update(sensor_deployment["environment"])
    render_cache_deployment = None
    render_replica_deployment = None
    if render_replica_runtime is not None:
        try:
            render_replica_deployment = prepare_render_replica(
                runtime_root=render_replica_runtime,
                server_config=Path(env["GZ_SIM_SERVER_CONFIG_PATH"]),
                world_sdf=render_world,
                vehicle_sdf=vehicle_sdf,
                camera_sdf=Path(camera_profile["model_resource_root"]) / "OakD-Lite/model.sdf",
                resource_paths=tuple(
                    Path(p) for p in env["GZ_SIM_RESOURCE_PATH"].split(os.pathsep)
                ),
                world_name=world_name,
                vehicle_name=vehicle_name,
                output=run_dir / "render-replica",
            )
        except Exception as error:
            # No Gazebo/PX4/airframe/renderer has been launched at this point.
            # Preserve that fact on preparation errors so training cleanup does
            # not demand a landing receipt for an aircraft that never existed.
            _write_json(
                run_dir / "simulation-preflight-stop.json",
                {
                    "run_directory": str(run_dir.resolve()),
                    "flight_vehicle_spawn_attempted": False,
                    "all_started_processes_exited": True,
                    "terminal_state": "NOT_STARTED",
                    "qualification_granted": False,
                    "preparation_error": str(error)[:512],
                },
            )
            raise
        env.update(render_replica_deployment["environment"])
    if render_preparation_runtime is not None:
        render_cache_deployment = prepare_render_cache(
            runtime_root=render_preparation_runtime,
            input_bundle=render_cache_bundle,
            output=run_dir / "render-cache",
            server_config=Path(env["GZ_SIM_SERVER_CONFIG_PATH"]),
            resources={
                "world": world_sdf.parent,
                "vehicle": vehicle_sdf.parent,
                "px4_models": px4_root / "Tools/simulation/gz/models",
                "ogre_media": Path("/usr/share/gz/gz-rendering8/ogre2/media"),
            },
            assets={"camera": camera_profile["receipt"], "render_batching": render_batching},
            environment=env,
        )
        env.update(render_cache_deployment["environment"])
    actuator_output_absolute_maximum = _px4_sitl_actuator_output_absolute_maximum(
        px4_root,
        px4_sitl_model,
    )
    if actuator_output_absolute_maximum is not None:
        env["PX4_ACTUATOR_OUTPUT_ABSOLUTE_MAXIMUM"] = f"{actuator_output_absolute_maximum:.12g}"
    # ROS command-line tools use the system interpreter.  The mission planner
    # and the depth worker use the signed Runtime virtual environment instead.
    # Reusing the latter's PYTHONPATH for ``ros2`` can shadow ros2cli (or its
    # protobuf build) and silently remove every mission observation.  Give ROS
    # an explicit, isolated module path while retaining all transport/domain
    # variables from the common environment.
    ros_environment = _ros_overlay_environment(ros_workspace=ros_workspace, env=env)
    ros_python_paths = [
        ros_workspace / "build" / "dronedream_agent_ros",
        ros_workspace / "install" / "lib/python3.12/site-packages",
        ros_workspace / "install" / "dronedream_agent_ros" / "lib/python3.12/site-packages",
        ros_workspace / "install" / "dronedream_agent_msgs" / "lib/python3.12/site-packages",
        Path("/opt/ros/jazzy/lib/python3.12/site-packages"),
    ]
    ros_environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in ros_python_paths if path.is_dir()
    )
    ros_environment.pop("PYTHONHOME", None)
    gz_binary = "/usr/bin/gz"
    # Gazebo Transport bindings in this process read the real environment,
    # while child processes receive the explicit copy above.
    os.environ["GZ_PARTITION"] = partition
    os.environ["GZ_IP"] = "127.0.0.1"
    route_points = [(point.x, point.y, point.z) for point in route.positions_m]
    estimated_seconds = sum(
        math.dist(first, second)
        / min(track.points[index].speed_limit_mps, track.points[index + 1].speed_limit_mps)
        for index, (first, second) in enumerate(zip(route_points, route_points[1:], strict=False))
    ) + track.waypoint_hold_seconds * (len(route_points) - 1)
    track_timeout = max(180.0, estimated_seconds * 1.8 + 60.0)
    executor_track_timeout = _executor_track_timeout_seconds(
        estimated_flight_seconds=estimated_seconds,
        track_timeout_seconds=track_timeout,
    )
    progress_extension_budget_seconds = _progress_extension_budget_seconds(
        estimated_flight_seconds=estimated_seconds,
        track_timeout_seconds=track_timeout,
    )

    processes: list[subprocess.Popen[Any] | None] = []
    depth_worker_process: subprocess.Popen[Any] | None = None
    render_replica_process: subprocess.Popen[Any] | None = None
    samples: list[tuple[float, float, float, float]] = []
    sample_lock = threading.Lock()
    frame_lock = threading.Lock()
    last_frame_at = 0.0
    abort_reason: str | None = None
    executor_return_code: int | None = None
    native_terminal_lifecycle: dict[str, object] | None = None
    flight_vehicle_spawn_attempted = False
    goal_observed_runtime = False
    closed_route = math.dist(route_points[0], route_points[-1]) <= 0.6
    goal_departure_observed = not closed_route
    goal_tracking_started_at: float | None = None
    phase_channels = [run_dir / "runtime-control" / f"phase-{name}-channel.json"
                      for name in ("independent", "identity", "control")]
    independent_phase_monitor = SimulationPhaseMonitor(channel=phase_channels[0])
    live_safety_event: dict[str, object] | None = None
    tolerated_landing_contacts = 0
    minimum_tolerated_landing_clearance = math.inf
    started = time.monotonic()

    # Lazy imports preserve Windows-side planning and tests while using the real
    # Gazebo Python bindings inside DroneDreamRuntime.
    system_packages = "/usr/lib/python3/dist-packages"
    if system_packages not in sys.path:
        sys.path.append(system_packages)
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GazeboNode

    gazebo_node = GazeboNode()
    gazebo_subscriptions = GazeboSubscriptions(gazebo_node)
    pose_lock = threading.Lock()
    pose_mailbox_lock = threading.Lock()
    pose_available = threading.Event()
    pose_worker_stop = threading.Event()
    latest_pose_message: tuple[Any, float, int] | None = None
    pose_processing_longest: list[tuple[float, int, dict]] = []
    pose_source_intervals = SimulationSourceIntervals()
    source_stall_probe = None
    pose_worker: threading.Thread | None = None
    pose_messages_received = 0
    pose_messages_overwritten = 0
    pose_messages_processed = 0
    previous_vehicle_pose: tuple[float, tuple[float, float, float]] | None = None
    dynamic_pose_history: dict[str, tuple[float, tuple[float, float, float]]] = {}
    local_safety_sequence = 0
    last_local_safety_at = 0.0
    last_local_safety_history_at = 0.0
    controlled_entity_name: str | None = None
    identity_mismatch_samples = 0
    identity_stale_samples = 0
    last_estimator_offset = Vector3(x=0.0, y=0.0, z=0.0)
    last_raw_estimator_offset = Vector3(x=0.0, y=0.0, z=0.0)
    has_validated_estimator_offset = False
    maximum_absolute_identity_correction_m = 1.0
    estimator_offset_filter_alpha = 0.2
    last_identity_validated_at_unix_ms: int | None = None
    last_identity_tracking_sample_unix_ms: int | None = None
    last_identity_debug_at = 0.0
    runtime_primitives = semantic.get("runtime_collision_primitives", primitives)
    if not isinstance(runtime_primitives, list) or len(runtime_primitives) > 100_000:
        raise SimulationRuntimeError("RUNTIME_COLLISION_GEOMETRY_INVALID")
    runtime_primitives = [_validated_primitive(item) for item in runtime_primitives]
    runtime_bounds = [(primitive, primitive_bounds(primitive)) for primitive in runtime_primitives]
    # 两条仿真见证通道共享当前世界的碰撞包络，不再猜测所有物体都是同尺寸行人。
    dynamic_geometry = load_dynamic_geometry(render_world)
    identity_phase_monitor = SimulationPhaseMonitor(channel=phase_channels[1])
    training_witness: GazeboOutcomeWitness | None = None
    localization_truth: LocalizationTruthCapture | None = None

    # 功能：
    #   按当前速度确定有限搜索范围，筛选与无人机相邻的已验证世界碰撞基元。
    # 输入：
    #   position：当前世界位置。
    # 输出：
    #   selected：附近基元列表。
    def nearby_primitives(position: tuple[float, float, float]) -> list[dict[str, Any]]:
        search_radius = max(5.0, local_max_speed_mps * 3.0 + 2.0)
        selected: list[dict[str, Any]] = []
        for primitive, (lower, upper) in runtime_bounds:
            if all(
                lower[index] - search_radius <= position[index] <= upper[index] + search_radius
                for index in range(3)
            ):
                selected.append(primitive)
        return selected

    # 功能：
    #   解析独立 Gazebo 观察，核对 PX4 定位身份并发布真实见证；原生深度主控时不另建真值导航器。
    # 输入：
    #   message：位姿帧；received_at：接收单调时间；received_at_unix_ms：接收墙钟时间。
    # 输出：
    #   无。
    def process_pose(message: Any, received_at: float, received_at_unix_ms: int) -> None:
        nonlocal abort_reason, controlled_entity_name, identity_mismatch_samples
        nonlocal identity_stale_samples, last_estimator_offset
        nonlocal last_raw_estimator_offset, has_validated_estimator_offset
        nonlocal last_identity_validated_at_unix_ms
        nonlocal last_identity_tracking_sample_unix_ms
        nonlocal last_identity_debug_at, last_local_safety_at, live_safety_event
        nonlocal last_local_safety_history_at, local_safety_sequence
        nonlocal pose_messages_processed
        nonlocal previous_vehicle_pose
        nonlocal dynamic_pose_history
        if abort_reason is not None:
            return
        # Keep the actual transport-receive clock through queueing and disk
        # work. Processing a delayed witness cannot make its pose newly true.
        elapsed = received_at - started
        source_queue_age_seconds = max(0.0, time.monotonic() - received_at)
        pose_messages_processed += 1
        scene_time_ns = simulation_pose_time_ns(message)
        if localization_truth is not None:
            localization_truth.record(
                message, received_monotonic=received_at, received_unix_ms=received_at_unix_ms
            )
        dynamic_poses = dynamic_positions(message.pose)
        resolved_vehicle = _resolve_controlled_vehicle_pose(
            message.pose,
            vehicle_name=vehicle_name,
            collision_center_offset_model_m=(offset_east, offset_north, offset_up),
            frames=sensor_frames,
        )
        if resolved_vehicle is None:
            return
        vehicle_pose, selected_entity_name, pose_reference, pose_candidates = resolved_vehicle
        if elapsed - last_identity_debug_at >= 0.5:
            raw_candidates = []
            for candidate in message.pose:
                candidate_name = str(candidate.name)
                if candidate_name not in pose_candidates:
                    continue
                raw_candidates.append(
                    {
                        "entity_id": int(candidate.id),
                        "name": candidate_name,
                        "header": str(candidate.header)[:1_000],
                        "position": {
                            "x": float(candidate.position.x),
                            "y": float(candidate.position.y),
                            "z": float(candidate.position.z),
                        },
                    }
                )
            _write_json(
                run_dir / "runtime-state" / "controlled-vehicle-pose-components.json",
                {
                    "schema_version": "dronedream.controlled-vehicle-pose-components.v1",
                    "elapsed_s": elapsed,
                    "selected_entity_name": selected_entity_name,
                    "pose_reference": pose_reference,
                    "raw_candidates": raw_candidates,
                    "resolved_model_root_world_enu_m": {
                        "east_m": vehicle_pose[0],
                        "north_m": vehicle_pose[1],
                        "up_m": vehicle_pose[2],
                    },
                    "source_queue_age_seconds": source_queue_age_seconds,
                },
            )
            last_identity_debug_at = elapsed
        if selected_entity_name != controlled_entity_name:
            controlled_entity_name = selected_entity_name
            _write_json(
                run_dir / "controlled-vehicle-entity.json",
                {
                    "schema_version": "dronedream.controlled-vehicle-entity.v1",
                    "vehicle_model_name": vehicle_name,
                    "selected_entity_name": selected_entity_name,
                    "pose_reference": pose_reference,
                    "candidate_entity_names": pose_candidates,
                    "collision_center_offset_model_m": [
                        offset_east,
                        offset_north,
                        offset_up,
                    ],
                    "selected_at_elapsed_s": elapsed,
                },
            )
        east, north, up = vehicle_pose
        with sample_lock:
            samples.append((elapsed, east, north, up))
        now_unix_ms = int(time.time() * 1_000)
        try:
            rendered = json.dumps(
                {
                    "schema_version": "dronedream.live-telemetry.v1",
                    "mode": "simulation",
                    "coordinate_frame": "gazebo-enu",
                    "elapsed_s": elapsed,
                    "east_m": east,
                    "north_m": north,
                    "up_m": up,
                    "updated_at_unix_ms": now_unix_ms,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            _write_bytes_atomic(run_dir / "live-telemetry.json", rendered)
        except OSError:
            pass
        if not local_safety_supported or not local_safety_target_path.is_file():
            previous_vehicle_pose = scene_time_ns, vehicle_pose
            return
        with pose_lock:
            # Native sensor control needs an independent outcome witness, not
            # a second truth-driven navigation controller. Keep its observation
            # cadence inside the learner's unchanged 100 ms boundary budget.
            witness_period = 0.04 if depth_safety_supported else 0.08
            if elapsed - last_local_safety_at < witness_period:
                return
            previous = previous_vehicle_pose
            previous_vehicle_pose = scene_time_ns, vehicle_pose
            velocity = simulation_velocity(vehicle_pose, previous, scene_time_ns)
            obstacles: list[DynamicObstacleObservation] = []
            for name, root_position in sorted(dynamic_poses.items()):
                history = dynamic_pose_history.get(name)
                obstacle_velocity = simulation_velocity(root_position, history, scene_time_ns)
                dynamic_pose_history[name] = scene_time_ns, root_position
                obstacles.append(
                    dynamic_observation(
                        name,
                        root_position,
                        obstacle_velocity,
                        dynamic_geometry,
                        velocity_known=history is not None,
                    )
                )
            # 已消失后重新出现的实体不能复用旧速度样本。
            dynamic_pose_history = {
                name: (scene_time_ns, position) for name, position in dynamic_poses.items()
            }
            try:
                target_payload = read_runtime_object(local_safety_target_path)
                target = Vector3.model_validate(target_payload["target_position_m"], strict=True)
                current = Vector3(
                    x=east + offset_east,
                    y=north + offset_north,
                    z=up + offset_up,
                )
                identity_phase_read = identity_phase_monitor.read(run_dir / "runtime-phase.json")
                runtime_phase = identity_phase_read["phase"]
                identity_required = _tracking_identity_required(runtime_phase)
                identity_telemetry_path = run_dir / "runtime-state" / "px4-identity-telemetry.json"
                tracking_path = (
                    identity_telemetry_path
                    if identity_telemetry_path.is_file()
                    else run_dir / "runtime-state" / "closed-loop-tracking.json"
                )
                identity_payload: dict[str, Any] | None = None
                if tracking_path.is_file():
                    tracking = read_runtime_object(tracking_path)
                    px4_center = Vector3.model_validate(
                        tracking["observed_world_collision_center_m"], strict=True
                    )
                    velocity_payload = tracking["observed_velocity_ned_mps"]
                    px4_velocity_ned = Vector3(
                        x=_runtime_number(
                            velocity_payload["north_m_s"], "north speed", -1000, 1000
                        ),
                        y=_runtime_number(velocity_payload["east_m_s"], "east speed", -1000, 1000),
                        z=_runtime_number(velocity_payload["down_m_s"], "down speed", -1000, 1000),
                    )
                    if type(tracking.get("updated_at_unix_ms")) is not int:
                        raise ValueError("identity sample timestamp must be an integer")
                    tracking_age_seconds = (now_unix_ms - tracking["updated_at_unix_ms"]) / 1000.0
                    identity_alignment = align_gazebo_px4_identity(
                        gazebo_position_world_enu_m=current,
                        px4_position_world_enu_m=px4_center,
                        px4_velocity_ned_mps=px4_velocity_ned,
                        minimum_alignment_speed_mps=0.05,
                    )
                    raw_identity_error_m = identity_alignment.raw_disagreement_m
                    identity_error_m = identity_alignment.time_aligned_disagreement_m
                    effective_identity_limit_m = dynamic_identity_disagreement_limit_m(
                        base_limit_m=identity_correction_limit_m,
                        px4_velocity_ned_mps=px4_velocity_ned,
                        alignment_seconds=identity_alignment.alignment_seconds,
                    )
                    offset_innovation_m = identity_offset_innovation_m(
                        estimator_offset_m=identity_alignment.estimator_offset_m,
                        reference_offset_m=(
                            last_raw_estimator_offset if has_validated_estimator_offset else None
                        ),
                    )
                    identity_transform_consistent = bool(
                        identity_error_m <= maximum_absolute_identity_correction_m
                        and offset_innovation_m <= effective_identity_limit_m
                    )
                    identity_fresh = 0.0 <= tracking_age_seconds <= 0.5
                    correction_source = "unavailable"
                    tracking_sample_unix_ms = int(tracking["updated_at_unix_ms"])
                    is_new_identity_sample = (
                        tracking_sample_unix_ms != last_identity_tracking_sample_unix_ms
                    )
                    if is_new_identity_sample:
                        last_identity_tracking_sample_unix_ms = tracking_sample_unix_ms
                        if identity_fresh and identity_transform_consistent:
                            identity_mismatch_samples = 0
                            identity_stale_samples = 0
                            raw_offset = identity_alignment.estimator_offset_m
                            if not has_validated_estimator_offset:
                                last_estimator_offset = raw_offset
                                has_validated_estimator_offset = True
                            else:
                                alpha = estimator_offset_filter_alpha
                                last_estimator_offset = Vector3(
                                    x=(
                                        last_estimator_offset.x
                                        + alpha * (raw_offset.x - last_estimator_offset.x)
                                    ),
                                    y=(
                                        last_estimator_offset.y
                                        + alpha * (raw_offset.y - last_estimator_offset.y)
                                    ),
                                    z=(
                                        last_estimator_offset.z
                                        + alpha * (raw_offset.z - last_estimator_offset.z)
                                    ),
                                )
                            last_raw_estimator_offset = raw_offset
                            last_identity_validated_at_unix_ms = now_unix_ms
                            correction_source = "live"
                        elif identity_fresh:
                            # Gazebo pose callbacks run much faster than the PX4
                            # telemetry publisher. Count independent estimator
                            # samples, not repeated comparisons against the same
                            # sample, before declaring an identity break.
                            identity_mismatch_samples += 1
                            identity_stale_samples = 0
                        else:
                            identity_stale_samples += 1
                    elif not identity_fresh:
                        identity_stale_samples += 1
                    correction_age_seconds = (
                        (now_unix_ms - last_identity_validated_at_unix_ms) / 1_000.0
                        if last_identity_validated_at_unix_ms is not None
                        else None
                    )
                    identity_accepted = correction_source == "live" or (
                        correction_age_seconds is not None
                        and 0.0 <= correction_age_seconds <= 2.0
                        and identity_mismatch_samples < 5
                        and identity_stale_samples < 20
                    )
                    if not identity_required:
                        identity_mismatch_samples = 0
                        identity_stale_samples = 0
                        identity_accepted = last_identity_validated_at_unix_ms is not None
                        if identity_accepted:
                            correction_source = "validated-before-terminal-phase"
                    if identity_accepted and correction_source != "live":
                        correction_source = correction_source if not identity_required else "cached"
                    identity_payload = {
                        "schema_version": "dronedream.controlled-vehicle-identity.v1",
                        "selected_entity_name": selected_entity_name,
                        "pose_reference": pose_reference,
                        "gazebo_collision_center_world_enu_m": current.model_dump(mode="json"),
                        "px4_collision_center_world_enu_m": px4_center.model_dump(mode="json"),
                        "position_disagreement_m": raw_identity_error_m,
                        "time_aligned_position_disagreement_m": identity_error_m,
                        "identity_offset_innovation_m": offset_innovation_m,
                        "maximum_offset_innovation_m": effective_identity_limit_m,
                        "maximum_absolute_correction_m": (maximum_absolute_identity_correction_m),
                        "identity_alignment_seconds": (identity_alignment.alignment_seconds),
                        "aligned_gazebo_collision_center_world_enu_m": (
                            identity_alignment.aligned_gazebo_position_m.model_dump(mode="json")
                        ),
                        "tracking_age_seconds": tracking_age_seconds,
                        "tracking_fresh": identity_fresh,
                        "runtime_phase": runtime_phase,
                        "identity_required": identity_required,
                        "validated_during_active_flight": (
                            last_identity_validated_at_unix_ms is not None
                        ),
                        "correction_source": correction_source,
                        "last_correction_age_seconds": correction_age_seconds,
                        "consecutive_mismatch_samples": identity_mismatch_samples,
                        "maximum_correction_m": maximum_absolute_identity_correction_m,
                        "accepted": identity_accepted and identity_mismatch_samples < 5,
                        "gazebo_pose_queue_age_seconds": source_queue_age_seconds,
                        "updated_at_unix_ms": now_unix_ms,
                    }
                    _write_json(
                        run_dir / "runtime-state" / "controlled-vehicle-identity.json",
                        identity_payload,
                    )
                    identity_failure_reason: str | None = None
                    if identity_required and identity_mismatch_samples >= 5:
                        identity_failure_reason = "CONTROLLED_ENTITY_IDENTITY_MISMATCH"
                    elif (
                        identity_required and identity_stale_samples >= 20 and not identity_accepted
                    ):
                        identity_failure_reason = "PX4_TRACKING_TELEMETRY_STALE"
                    if identity_failure_reason is not None and abort_reason is None:
                        abort_reason = identity_failure_reason
                        live_safety_event = {
                            "schema_version": "dronedream.live-safety-event.v1",
                            "reason": abort_reason,
                            "elapsed_s": elapsed,
                            "vehicle_collision_center_world_enu_m": {
                                "east_m": current.x,
                                "north_m": current.y,
                                "up_m": current.z,
                            },
                            "px4_collision_center_world_enu_m": (
                                px4_center.model_dump(mode="json")
                            ),
                            "position_disagreement_m": raw_identity_error_m,
                            "time_aligned_position_disagreement_m": identity_error_m,
                            "identity_offset_innovation_m": offset_innovation_m,
                            "maximum_offset_innovation_m": effective_identity_limit_m,
                            "maximum_absolute_correction_m": (
                                maximum_absolute_identity_correction_m
                            ),
                            "identity_alignment_seconds": (identity_alignment.alignment_seconds),
                            "selected_entity_name": selected_entity_name,
                            "pose_reference": pose_reference,
                            "tracking_age_seconds": tracking_age_seconds,
                        }
                        _write_json(run_dir / "live-safety-event.json", live_safety_event)
                local_safety_sequence += 1
                observation_age_seconds = witness_age(received_at, time.monotonic())
                observation = RuntimeLocalSafetyObservation(
                    sequence=local_safety_sequence,
                    observed_at_unix_ms=received_at_unix_ms,
                    source="simulation-ground-truth",
                    stream_healthy=(
                        observation_age_seconds <= 0.1
                        and previous is not None
                        and all(item.confidence == 1 for item in obstacles)
                    ),
                    stream_age_seconds=observation_age_seconds,
                    localization_covariance_m2=0.0,
                    current_position_m=current,
                    current_velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
                    target_position_m=target,
                    dynamic_obstacles=obstacles,
                )
                if depth_safety_supported:
                    # This file is label/verification-only. The actuator already
                    # consumes the native depth channel selected above; never
                    # spend its observer thread planning unused truth commands.
                    # Collision/entity monitoring elsewhere remains unchanged.
                    _write_bytes_atomic(
                        local_safety_observation_path,
                        observation.model_dump_json(indent=2).encode("utf-8"),
                    )
                    last_local_safety_at = elapsed
                    return
                request = LocalPlannerRequest(
                    current_position_m=current,
                    current_velocity_mps=observation.current_velocity_mps,
                    target_position_m=target,
                    dynamic_obstacles=obstacles,
                    vehicle_radius_m=diameter / 2.0,
                    vehicle_height_m=height,
                    max_speed_mps=local_max_speed_mps,
                    max_acceleration_mps2=local_max_acceleration_mps2,
                    required_clearance_m=local_required_clearance_m,
                    prediction_horizon_seconds=3.0,
                    prediction_step_seconds=0.2,
                )
                decision = predictive_safety_decision(
                    request,
                    nearby_primitives((current.x, current.y, current.z)),
                )
                estimator_offset = Vector3(x=0.0, y=0.0, z=0.0)
                if identity_payload is not None and bool(identity_payload.get("accepted")):
                    estimator_offset = last_estimator_offset
                command = RuntimeLocalSafetyCommand(
                    observation_sha256=sha256_json(observation),
                    observation_sequence=observation.sequence,
                    generated_at_unix_ms=now_unix_ms,
                    valid_until_unix_ms=now_unix_ms + 500,
                    source=observation.source,
                    estimator_to_world_position_offset_m=estimator_offset,
                    command_position_m=Vector3(
                        x=current.x + decision.selected_velocity_mps.x * 0.2,
                        y=current.y + decision.selected_velocity_mps.y * 0.2,
                        z=current.z + decision.selected_velocity_mps.z * 0.2,
                    ),
                    decision=decision,
                )
                _write_bytes_atomic(
                    local_safety_observation_path,
                    observation.model_dump_json(indent=2).encode("utf-8"),
                )
                _write_bytes_atomic(
                    local_safety_command_path,
                    command.model_dump_json(indent=2).encode("utf-8"),
                )
                if decision.action != "continue" or elapsed - last_local_safety_history_at >= 1.0:
                    history_entry = {
                        "elapsed_s": elapsed,
                        "route_target_m": target.model_dump(mode="json"),
                        "observation": observation.model_dump(mode="json"),
                        "command": command.model_dump(mode="json"),
                    }
                    with (run_dir / "local-safety-history.jsonl").open(
                        "a", encoding="utf-8"
                    ) as history:
                        history.write(json.dumps(history_entry, sort_keys=True) + "\n")
                    last_local_safety_history_at = elapsed
                last_local_safety_at = elapsed
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                return

    # 功能：
    #   在传输回调中仅保存最新帧和接收时间，覆盖旧帧并计数，不让几何处理阻塞消息接收。
    # 输入：
    #   message：本次原生位姿帧。
    # 输出：
    #   无。
    def on_pose(message: Any) -> None:
        """Keep only the newest Gazebo pose frame.

        Gazebo can publish the dynamic-pose stream substantially faster than
        collision and predictive-avoidance evaluation can run.  Performing
        those operations in this transport callback creates an unbounded
        message backlog: PX4 telemetry is current while the supposedly live
        Gazebo pose is old.  Copying one frame into a latest-value mailbox
        makes stale frames disposable and leaves all safety work to one
        bounded worker.
        """

        nonlocal latest_pose_message, pose_messages_received, pose_messages_overwritten
        nonlocal abort_reason
        received_at = time.monotonic()
        received_at_unix_ms = int(time.time() * 1000)
        if source_stall_probe is not None:
            source_stall_probe.source_received(received_at)
        try:
            scene_time_ns = simulation_pose_time_ns(message)
        except ValueError:
            abort_reason = "SIMULATION_POSE_SOURCE_TIME_INVALID"
            return
        pose_source_intervals.observe(
            received_at,
            received_at_unix_ms,
            scene_time_ns,
        )
        if training_witness is not None:
            training_witness.receive(
                message.pose, received_at, received_at_unix_ms, simulation_time_ns=scene_time_ns
            )
            if training_witness.error is not None and abort_reason is None:
                abort_reason = "INDEPENDENT_TRAINING_WITNESS_FAILED"
        copied = Pose_V()
        copied.CopyFrom(message)
        with pose_mailbox_lock:
            pose_messages_received += 1
            if latest_pose_message is not None:
                pose_messages_overwritten += 1
            latest_pose_message = copied, received_at, received_at_unix_ms
        pose_available.set()

    # 功能：
    #   消费最新位姿邮箱，记录真实处理耗时，处理异常时发布中止而非继续使用旧判断。
    # 输入：
    #   无。
    # 输出：
    #   无。
    def pose_worker_loop() -> None:
        nonlocal abort_reason, latest_pose_message, live_safety_event
        while not pose_worker_stop.is_set():
            pose_available.wait(0.1)
            pose_available.clear()
            with pose_mailbox_lock:
                pending = latest_pose_message
                latest_pose_message = None
            if pending is None:
                continue
            message, received_at, received_at_unix_ms = pending
            processing_started = time.monotonic()
            try:
                process_pose(message, received_at, received_at_unix_ms)
            except Exception as error:  # fail closed across the transport thread boundary
                if abort_reason is None:
                    abort_reason = "GAZEBO_POSE_PROCESSING_FAILED"
                    live_safety_event = {
                        "schema_version": "dronedream.live-safety-event.v1",
                        "reason": abort_reason,
                        "error_type": type(error).__name__,
                    }
                    _write_json(run_dir / "live-safety-event.json", live_safety_event)
                return
            finally:
                duration_ms = (time.monotonic() - processing_started) * 1000
                entry = (
                    duration_ms,
                    pose_messages_processed,
                    {
                        "received_at_unix_ms": received_at_unix_ms,
                        "source_queue_ms": (processing_started - received_at) * 1000,
                        "processing_ms": duration_ms,
                        "processed_sequence": pose_messages_processed,
                    },
                )
                if len(pose_processing_longest) < 20:
                    heapq.heappush(pose_processing_longest, entry)
                else:
                    heapq.heappushpop(pose_processing_longest, entry)

    # 功能：
    #   限速复制真实预览帧给专属写入线程，不在 Gazebo 回调中做 PNG 编码和磁盘写入。
    # 输入：
    #   message：原生观察相机消息。
    # 输出：
    #   无。
    def on_live_frame(message: Any) -> None:
        nonlocal last_frame_at
        now = time.monotonic()
        with frame_lock:
            if now - last_frame_at < 0.08:
                return
            last_frame_at = now
        # Gazebo callbacks share transport resources. PNG encoding/filesystem
        # writes must not block delivery of independent pose messages.
        copied = Image()
        copied.CopyFrom(message)
        preview_worker.submit(copied)

    observer_pause_monitor = None
    preview_worker = None
    try:
        independent_phase_monitor.open()
        identity_phase_monitor.open()
        _verify_runtime_inputs(input_hashes)
        if simulation_training_channel is not None:
            source_stall_probe = NativeSourceStallProbe()
        observer_pause_monitor = InterpreterPauseMonitor()
        preview_worker = LatestPreviewWorker(
            lambda message: _write_bytes_atomic(
                run_dir / "live-frame.png", _gazebo_image_png(message)
            )
        )
        if source_training and depth_safety_supported:
            localization_truth = LocalizationTruthCapture(
                run_dir, frames=sensor_frames, summary_publisher=_write_json
            )
        if simulation_training_channel is not None:
            training_witness = GazeboOutcomeWitness(
                publisher=OutcomePublisher(outcome_descriptor_path(simulation_training_channel)),
                resolve_pose=partial(
                    _resolve_controlled_vehicle_pose,
                    vehicle_name=vehicle_name,
                    collision_center_offset_model_m=(offset_east, offset_north, offset_up),
                    frames=sensor_frames,
                ),
                collision_offset=(offset_east, offset_north, offset_up),
                target=route.positions_m[-1],
                dynamic_geometry=dynamic_geometry,
            )
        with (
            (run_dir / "gazebo.log").open("w", encoding="utf-8") as gazebo_log,
            (run_dir / "px4.log").open("w", encoding="utf-8") as px4_log,
            (run_dir / "executor.stdout.log").open("w", encoding="utf-8") as executor_out,
            (run_dir / "executor.stderr.log").open("w", encoding="utf-8") as executor_err,
            (run_dir / "ros-observer.log").open("w", encoding="utf-8") as observer_log,
            (run_dir / "domain-action-server.log").open("w", encoding="utf-8") as domain_action_log,
            (run_dir / "ros-observations.csv").open("w", encoding="utf-8") as ros_csv,
            (run_dir / "depth-safety-worker.log").open("w", encoding="utf-8") as depth_worker_log,
            (run_dir / "render-replica.log").open("w", encoding="utf-8") as render_replica_log,
        ):
            gazebo_env, graphics_lifetime = prepare_render_process_environment(env)
            _write_json(
                run_dir / "simulation-graphics-request.json",
                {**graphics_request_evidence(gazebo_env), "driver_lifetime": graphics_lifetime},
            )
            gazebo = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        gz_binary,
                        "sim",
                        "-r",
                        "-s",
                        "--headless-rendering",
                        str(render_world),
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=gazebo_env,
                stdout=gazebo_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(gazebo)
            if source_stall_probe is not None:
                source_stall_probe.register("gazebo", gazebo.pid)
            _wait_for_world(gz_binary, world_name, env, 90)
            if render_replica_deployment is not None:
                render_replica_process = subprocess.Popen(
                    _with_runtime_cpu_affinity(
                        render_replica_deployment["command"],
                        plan=runtime_cpu_affinity,
                        local_model=False,
                    ),
                    env=gazebo_env,
                    stdout=render_replica_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                processes.append(render_replica_process)
                if source_stall_probe is not None:
                    source_stall_probe.register("render-replica", render_replica_process.pid)
            if preflight_render_warmup:
                # A separate bounded process owns diagnostic subscriptions/RPC.
                # Airframe and PX4 wait for verified preparation and subscription
                # shutdown. Explicit depth preparation keeps its sensor-only rig
                # in this same process until Gazebo exits, avoiding destroy/recreate.
                warmup_camera = Path(camera_profile["model_resource_root"]) / "OakD-Lite/model.sdf"
                warmup_command = [
                    sys.executable,
                    "-m",
                    "dronedream_agent_core.preflight_render_warmup",
                    "--gz",
                    gz_binary,
                    "--world",
                    world_name,
                    "--flight-vehicle",
                    vehicle_name,
                    "--camera",
                    str(warmup_camera),
                    "--source-sha256",
                    camera_profile["receipt"]["profiled_model_sha256"],
                    "--position",
                    str(float(model_root[0])),
                    str(float(model_root[1])),
                    str(float(model_root[2]) + 1.0),
                    "--output",
                    str(run_dir / "render-warmup"),
                ]
                if preflight_depth_warmup:
                    warmup_command.append("--include-depth")
                warmup_result = _run(warmup_command, env=env, timeout=120)
                _write_json(
                    run_dir / "render-warmup-process.json",
                    {
                        "exit_code": warmup_result.returncode,
                        "stdout": warmup_result.stdout[-4000:],
                        "stderr": warmup_result.stderr[-4000:],
                        "flight_vehicle_spawned": False,
                    },
                )
                warmup_receipt_path = run_dir / "render-warmup/receipt.json"
                warmup_receipt = (
                    read_runtime_object(warmup_receipt_path)
                    if warmup_receipt_path.is_file()
                    else {}
                )
                if warmup_result.returncode != 0 or not warmup_receipt_ready(
                    warmup_receipt,
                    source_sha256=camera_profile["receipt"]["profiled_model_sha256"],
                    include_depth=preflight_depth_warmup,
                ):
                    raise SimulationRuntimeError("PREFLIGHT_RENDER_WARMUP_NOT_CONFIRMED")
            if render_cache_deployment is not None:
                _write_json(
                    run_dir / "render-cache-readback.json",
                    verify_render_preparation(run_dir / "render-cache"),
                )
            if payload_spawn is not None:
                payload_evidence = _spawn_entity(
                    gz_binary,
                    world_name=world_name,
                    entity_name=str(payload_spawn["entity_name"]),
                    sdf_path=Path(payload_spawn["sdf_path"]),
                    pose=tuple(payload_spawn["pose"]),
                    env=env,
                )
                payload_evidence.update(
                    {
                        "checkpoint_id": payload_spawn["checkpoint_id"],
                        "track_point_index": payload_spawn["track_point_index"],
                        "runtime_action_step_id": payload_spawn["runtime_action_step_id"],
                        "attach_topic": payload_spawn["attach_topic"],
                        "detach_topic": payload_spawn["detach_topic"],
                        "output_topic": payload_spawn["output_topic"],
                    }
                )
                _write_json(run_dir / "payload_spawn.json", payload_evidence)
            flight_vehicle_spawn_attempted = True  # Even a lost spawn reply is an attempt.
            spawn_evidence = _spawn_entity(
                gz_binary,
                world_name=world_name,
                entity_name=vehicle_name,
                sdf_path=vehicle_sdf,
                pose=(float(model_root[0]), float(model_root[1]), float(model_root[2])),
                env=env,
            )
            _write_json(run_dir / "vehicle_spawn.json", spawn_evidence)
            if payload_spawn is not None:
                preflight_detach = _detach_payload_before_flight(
                    gz_binary,
                    detach_topic=str(payload_spawn["detach_topic"]),
                    output_topic=str(payload_spawn["output_topic"]),
                    env=env,
                )
                payload_evidence.update(
                    {
                        "initial_attachment_policy": (
                            "gazebo-harmonic-start-attached-then-verified-preflight-detach"
                        ),
                        "preflight_detach": preflight_detach,
                    }
                )
                _write_json(run_dir / "payload_spawn.json", payload_evidence)
                # 跨进程传递真实分离回执，不传可直接冒充物理状态的 attached/detached 常量。
                payload_preflight_path = run_dir / "payload-preflight-observation.json"
                _write_json(payload_preflight_path, {
                    "schema_version": "dronedream.payload-preflight-observation",
                    "partition": env.get("GZ_PARTITION", ""), "world": world_name,
                    "observed_at_unix_ms": int(time.time() * 1000),
                    "observation": preflight_detach,
                })
                env["PX4_GAZEBO_PAYLOAD_PREFLIGHT_PATH"] = str(payload_preflight_path)
                env["PX4_GAZEBO_PAYLOAD_PREFLIGHT_SHA256"] = _sha256(payload_preflight_path)
            camera_ready = False
            if live_camera_enabled:
                camera_sdf, camera_pose = _live_camera_sdf(route_points)
                camera_sdf_path = run_dir / "live-camera.sdf"
                camera_sdf_path.write_text(camera_sdf, encoding="utf-8")
                try:
                    camera_spawn = _spawn_entity(
                        gz_binary,
                        world_name=world_name,
                        entity_name="dronedream_live_camera",
                        sdf_path=camera_sdf_path,
                        pose=camera_pose,
                        env=env,
                    )
                    camera_ready = True
                    _write_json(run_dir / "live-camera-spawn.json", camera_spawn)
                except SimulationRuntimeError as error:
                    _write_json(
                        run_dir / "live-camera-spawn.json",
                        {"accepted": False, "issue": str(error)},
                    )
            else:
                _write_json(
                    run_dir / "live-camera-spawn.json",
                    {
                        "accepted": False,
                        "status": "disabled-for-onboard-perception-qualification",
                    },
                )
            pose_worker = threading.Thread(
                target=pose_worker_loop,
                name="dronedream-gazebo-pose-worker",
                daemon=True,
            )
            pose_worker.start()
            gazebo_subscriptions.subscribe(
                Pose_V, f"/world/{world_name}/dynamic_pose/info", on_pose
            )
            if camera_ready:
                gazebo_subscriptions.subscribe(Image, "/dronedream/live/camera", on_live_frame)
            if depth_safety_supported:
                if render_replica_deployment is not None:
                    # PX4 and the executor have not started yet. Camera readiness
                    # cannot arm or authorize a control command.
                    readiness = _run(
                        [
                            sys.executable,
                            "-m",
                            "dronedream_agent_core.simulation_render_replica",
                            "--deployment",
                            str(run_dir / "render-replica/deployment.json"),
                            "--receipt",
                            str(run_dir / "render-replica/readiness.json"),
                        ],
                        env=env,
                        timeout=105,
                    )
                    _write_json(
                        run_dir / "render-replica/readiness-process.json",
                        {
                            "exit_code": readiness.returncode,
                            "stdout": readiness.stdout[-2000:],
                            "stderr": readiness.stderr[-2000:],
                            "px4_started": False,
                            "qualification_granted": False,
                        },
                    )
                    if readiness.returncode != 0 or render_replica_process.poll() is not None:
                        raise SimulationRuntimeError("ISOLATED_RENDERER_PREFLIGHT_NOT_READY")
                worker_environment = dict(env)
                source_root = str(Path(__file__).resolve().parents[1])
                worker_environment["PYTHONPATH"] = os.pathsep.join(
                    value
                    for value in (
                        source_root,
                        system_packages,
                        worker_environment.get("PYTHONPATH", ""),
                    )
                    if value
                )
                depth_worker_command = [
                    sys.executable,
                    str(depth_worker_script),
                    "--world",
                    world_name,
                    "--vehicle",
                    vehicle_name,
                    "--vehicle-metadata",
                    str(vehicle_metadata_path),
                    "--semantic",
                    str(semantic_path),
                    "--target",
                    str(local_safety_target_path),
                    "--observation",
                    str(depth_safety_observation_path),
                    "--local-safety-channel",
                    str(run_dir / "runtime-control" / "local-safety-channel.json"),
                    "--native-state-channel",
                    str(run_dir / "runtime-control" / "native-state-channel.json"),
                    "--perception-health-channel",
                    str(run_dir / "runtime-control" / "perception-health-channel.json"),
                    "--runtime-phase-channel",
                    str(phase_channels[2]),
                    "--command",
                    str(depth_safety_command_path),
                    "--history",
                    str(depth_safety_history_path),
                    "--health",
                    str(depth_perception_health_path),
                    "--identity-telemetry",
                    str(run_dir / "runtime-state" / "px4-identity-telemetry.json"),
                    "--collision-offset",
                    f"{offset_east:.12g}",
                    f"{offset_north:.12g}",
                    f"{offset_up:.12g}",
                    "--vehicle-radius",
                    f"{diameter / 2.0:.12g}",
                    "--vehicle-height",
                    f"{height:.12g}",
                    "--max-speed",
                    f"{local_max_speed_mps:.12g}",
                    "--max-acceleration",
                    f"{local_max_acceleration_mps2:.12g}",
                    "--required-clearance",
                    f"{local_required_clearance_m:.12g}",
                ]
                onboard_rgb_topic = (
                    f"/world/{world_name}/model/{vehicle_name}/link/camera_link/sensor/IMX214/image"
                )
                if local_navigation_visual_enabled or multimodal_dataset_root is not None:
                    depth_worker_command.extend(["--rgb-topic", onboard_rgb_topic])
                if render_replica_deployment is not None:
                    depth_worker_command.extend(
                        [
                            "--render-scene-epoch",
                            render_replica_deployment["epoch"],
                            "--depth-topic",
                            render_replica_deployment["depth_topic"],
                        ]
                    )
                if (
                    runtime_cpu_affinity.get("enabled") is True
                    and local_navigation_provider is not None
                ):
                    model_cpu_ids = runtime_cpu_affinity.get("local_model_cpu_ids")
                    if isinstance(model_cpu_ids, list) and model_cpu_ids:
                        depth_worker_command.extend(
                            [
                                "--local-navigation-cpu-ids",
                                *(str(cpu_id) for cpu_id in model_cpu_ids),
                            ]
                        )
                if multimodal_dataset_root is not None:
                    assert multimodal_flight_id is not None
                    depth_worker_command.extend(
                        [
                            "--multimodal-dataset-root",
                            str(multimodal_dataset_root),
                            "--multimodal-flight-id",
                            multimodal_flight_id,
                            "--multimodal-dataset-maximum-mib",
                            str(multimodal_dataset_maximum_mib),
                            "--multimodal-record-period-seconds",
                            f"{multimodal_record_period_seconds:g}",
                        ]
                    )
                if semantic_label_topic is not None:
                    assert semantic_label_map_path is not None
                    depth_worker_command.extend(
                        [
                            "--semantic-label-topic",
                            semantic_label_topic,
                            "--semantic-label-map",
                            str(semantic_label_map_path),
                        ]
                    )
                if development_depth_drop_after_seconds is not None:
                    assert development_depth_drop_duration_seconds is not None
                    depth_worker_command.extend(
                        [
                            "--development-depth-drop-after-seconds",
                            f"{development_depth_drop_after_seconds:g}",
                            "--development-depth-drop-duration-seconds",
                            f"{development_depth_drop_duration_seconds:g}",
                            "--development-fault-evidence",
                            str(development_depth_fault_path),
                        ]
                    )
                if local_navigation_provider is not None:
                    depth_worker_command.extend(
                        [
                            "--qualified-route",
                            str(route_path),
                            "--qualified-clearance",
                            str(clearance_path),
                            "--local-navigation-provider",
                            local_navigation_provider,
                            "--local-navigation-model-timeout-seconds",
                            f"{local_navigation_model_timeout_seconds:g}",
                            "--local-navigation-period-seconds",
                            f"{local_navigation_period_seconds:g}",
                            "--local-navigation-controller-step-m",
                            f"{model_navigation_controller_step_m:.12g}",
                            "--model-navigation-evidence",
                            str(run_dir / "model-navigation-cycles.jsonl"),
                            "--model-navigation-call-evidence",
                            str(run_dir / "model-navigation-model-calls.jsonl"),
                            "--model-navigation-snapshot-evidence",
                            str(run_dir / "model-navigation-snapshots.jsonl"),
                        ]
                    )
                    if local_navigation_control_authority_required:
                        depth_worker_command.append("--require-model-control-authority")
                    if simulation_training_channel is not None:
                        depth_worker_command.extend(
                            ["--simulation-training-channel", str(simulation_training_channel)]
                        )
                    if local_navigation_omit_coordinate_candidates:
                        depth_worker_command.append("--omit-coordinate-candidates")
                    if local_navigation_fallback_provider is not None:
                        depth_worker_command.extend(
                            [
                                "--local-navigation-fallback-provider",
                                local_navigation_fallback_provider,
                                "--local-navigation-fallback-model-timeout-seconds",
                                f"{local_navigation_fallback_model_timeout_seconds:g}",
                            ]
                        )
                    for package_path in local_policy_package_paths:
                        depth_worker_command.extend(["--local-policy-package", str(package_path)])
                    for qualification_path in local_policy_qualification_paths:
                        depth_worker_command.extend(
                            ["--local-policy-qualification", str(qualification_path)]
                        )
                    for admission_path in local_policy_simulation_admission_paths:
                        depth_worker_command.extend(
                            ["--local-policy-simulation-admission", str(admission_path)]
                        )
                    if development_payload_collection:
                        depth_worker_command.append("--development-payload-collection")
                if camera_profile is not None:
                    depth_worker_command.extend(
                        [
                            "--camera-profile-receipt",
                            str(run_dir / "camera-profile/camera-profile.json"),
                        ]
                    )
                if record_learning_observations:
                    depth_worker_command.extend(
                        [
                            "--record-learning-observations",
                            "--learning-image-size",
                            str(learning_image_size[0]),
                            str(learning_image_size[1]),
                        ]
                    )
                    if simulation_teacher_control:
                        depth_worker_command.append("--simulation-teacher-control")
                    if local_navigation_provider is None:
                        depth_worker_command.extend(
                            [
                                "--qualified-route",
                                str(route_path),
                                "--qualified-clearance",
                                str(clearance_path),
                            ]
                        )
                if local_navigation_provider is not None:
                    if local_navigation_context_id is not None:
                        depth_worker_command.extend(
                            ["--local-navigation-context-id", local_navigation_context_id]
                        )
                    if local_navigation_visual_enabled:
                        depth_worker_command.extend(
                            [
                                "--model-navigation-frame-dir",
                                str(run_dir / "model-navigation-frames"),
                            ]
                        )
                depth_worker_process = subprocess.Popen(
                    _with_runtime_cpu_affinity(
                        depth_worker_command,
                        plan=runtime_cpu_affinity,
                        # The process owns the sensor-rate depth projection,
                        # world model, safety decisions, and evidence I/O as
                        # well as one asynchronous model thread.  Keep the
                        # process on the general set; the executor initializer
                        # above pins only the model thread to the reserved set.
                        local_model=False,
                    ),
                    env=worker_environment,
                    stdout=depth_worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                processes.append(depth_worker_process)
            _write_json(
                native_preflight_marker,
                {
                    "schema_version": "dronedream.native-runtime-preflight.v1",
                    "contract_id": contract_id,
                    "status": "ready",
                    "readiness_scope": "runtime-processes-only",
                    "flight_ready": False,
                    "vehicle_spawn_confirmed": bool(spawn_evidence.get("accepted")),
                    "payload_preflight": (
                        payload_evidence.get("preflight_detach")
                        if payload_spawn is not None
                        else {"status": "not-applicable"}
                    ),
                    "live_camera_ready": camera_ready,
                    "depth_safety_worker_started": depth_worker_process is not None,
                    "local_navigation_model_enabled": (
                        depth_worker_process is not None and local_navigation_provider is not None
                    ),
                    "local_navigation_visual_enabled": (
                        depth_worker_process is not None
                        and local_navigation_provider is not None
                        and local_navigation_visual_enabled
                    ),
                    "multimodal_dataset_recording_enabled": (
                        depth_worker_process is not None and multimodal_dataset_root is not None
                    ),
                    "semantic_supervision_enabled": (
                        depth_worker_process is not None and semantic_label_topic is not None
                    ),
                    "runtime_cpu_affinity": runtime_cpu_affinity,
                },
            )
            abort_file = run_dir / "live_abort.request.json"

            bridge = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        "ros2",
                        "run",
                        "ros_gz_bridge",
                        "parameter_bridge",
                        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=ros_environment,
                stdout=observer_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(bridge)
            observer = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        "ros2",
                        "run",
                        "dronedream_agent_ros",
                        "gazebo_pose_observer",
                        "--ros-args",
                        "-p",
                        "use_sim_time:=true",
                        "-p",
                        f"entity_name:={vehicle_name}",
                        "-p",
                        f"gazebo_pose_topic:=/world/{world_name}/dynamic_pose/info",
                        "-p",
                        f"contract_id:={contract_id}",
                        "-p",
                        "segment_id:=runtime-segment",
                        "-p",
                        f"runtime_phase_path:={run_dir / 'runtime-phase.json'}",
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=ros_environment,
                stdout=observer_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(observer)
            if copied_checkpoint_contract is not None:
                domain_action_server = subprocess.Popen(
                    _with_runtime_cpu_affinity(
                        [
                            "ros2",
                            "run",
                            "dronedream_agent_ros",
                            "domain_action_server",
                            "--ros-args",
                            "-p",
                            f"contract_id:={contract_id}",
                            "-p",
                            f"track_path:={copied_track}",
                            "-p",
                            f"checkpoint_contract_path:={copied_checkpoint_contract}",
                            "-p",
                            "maximum_target_distance_m:=1.0",
                            "-p",
                            "maximum_observation_age_seconds:=2.0",
                        ],
                        plan=runtime_cpu_affinity,
                        local_model=False,
                    ),
                    env=ros_environment,
                    stdout=domain_action_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                processes.append(domain_action_server)
            safety_guard = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        "ros2",
                        "run",
                        "dronedream_agent_ros",
                        "safety_event_guard",
                        "--ros-args",
                        "-p",
                        f"contract_id:={contract_id}",
                        "-p",
                        f"abort_file:={abort_file}",
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=ros_environment,
                stdout=observer_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(safety_guard)
            ros_recorder = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        "ros2",
                        "topic",
                        "echo",
                        "--no-daemon",
                        "--csv",
                        "/dronedream/mission_observation",
                        "dronedream_agent_msgs/msg/MissionObservation",
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=ros_environment,
                stdout=ros_csv,
                stderr=observer_log,
                text=True,
                start_new_session=True,
            )
            processes.append(ros_recorder)

            px4 = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        str(px4_executable),
                        "-d",
                        "-w",
                        str(trial_rootfs),
                        str(trial_rootfs),
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                cwd=trial_rootfs,
                env=env,
                stdout=px4_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(px4)
            if source_stall_probe is not None:
                source_stall_probe.register("px4", px4.pid)
            _wait_for_vehicle(gz_binary, vehicle_name, env, 120)
            executor_environment = dict(env)
            executor_environment["PYTHONPATH"] = os.pathsep.join(
                entry
                for entry in executor_environment.get("PYTHONPATH", "").split(os.pathsep)
                if entry and entry != system_packages and not entry.startswith("/opt/ros/")
            )
            if multimodal_dataset_root is not None:
                executor_environment["DRONEDREAM_SIMULATION_ONBOARD_RGB_DATASET"] = str(
                    multimodal_dataset_root
                )
            executor = subprocess.Popen(
                _with_runtime_cpu_affinity(
                    [
                        sys.executable,
                        str(executor_path),
                        "--run-dir",
                        str(run_dir),
                        "--track",
                        str(copied_track),
                        "--params",
                        str(copied_params),
                        "--vehicle",
                        px4_sitl_model,
                        "--world",
                        world_name,
                        "--gazebo-vehicle-model-name",
                        vehicle_name,
                        "--abort-file",
                        str(abort_file),
                        "--setpoint-rate-hz",
                        "20",
                        *(["--simulation-teacher-control"] if simulation_teacher_control else []),
                        "--takeoff-timeout-seconds",
                        f"{SIMULATION_TAKEOFF_STABILITY_TIMEOUT_SECONDS:g}",
                        "--takeoff-climb-rate-m-s",
                        "0.7",
                        "--track-timeout-seconds",
                        f"{executor_track_timeout:g}",
                        "--landing-timeout-seconds",
                        "90",
                        "--takeoff-stable-window-seconds",
                        "5",
                        "--tracking-lag-limit-m",
                        f"{tracking_corridor_policy['tracking_lag_limit_m']:.9g}",
                        "--tracking-rejoin-tolerance-m",
                        f"{tracking_corridor_policy['tracking_rejoin_tolerance_m']:.9g}",
                        *(
                            [
                                "--tracking-corridor-policy",
                                str(state / "tracking-corridor-policy.json"),
                            ]
                            if "--tracking-corridor-policy" in executor_help.stdout
                            else []
                        ),
                        "--model-progress-slack-m",
                        f"{track.waypoint_position_tolerance_m:.9g}",
                        "--tracking-sample-rate-hz",
                        f"{tracking_corridor_policy['tracking_sample_rate_hz']:.9g}",
                        "--tracking-telemetry-recovery-timeout-seconds",
                        "5",
                        *heading_executor_arguments,
                        *(argument for path in phase_channels
                          for argument in ("--runtime-phase-channel", str(path))),
                        "--log",
                        str(run_dir / "offboard_executor.log"),
                        *(
                            [
                                "--local-safety-command",
                                str(active_local_safety_command_path),
                                "--local-safety-observation",
                                str(active_local_safety_observation_path),
                                *(
                                    [
                                        "--local-safety-channel",
                                        str(
                                            run_dir
                                            / "runtime-control"
                                            / "local-safety-channel.json"
                                        ),
                                        "--native-state-channel",
                                        str(
                                            run_dir
                                            / "runtime-control"
                                            / "native-state-channel.json"
                                        ),
                                        "--perception-health-channel",
                                        str(run_dir / "runtime-control"
                                            / "perception-health-channel.json"),
                                    ]
                                    if depth_safety_supported
                                    else []
                                ),
                                "--local-safety-target",
                                str(local_safety_target_path),
                                *(["--local-safety-required"] if depth_safety_supported else []),
                                *(
                                    ["--require-model-control-authority"]
                                    if local_navigation_control_authority_required
                                    else []
                                ),
                            ]
                            if local_safety_supported
                            else []
                        ),
                        *(value for pair in executor_options.items() for value in pair),
                    ],
                    plan=runtime_cpu_affinity,
                    local_model=False,
                ),
                env=executor_environment,
                stdout=executor_out,
                stderr=executor_err,
                text=True,
                start_new_session=True,
            )
            processes.append(executor)
            deadline = time.monotonic() + _executor_wall_timeout_seconds(
                executor_track_timeout_seconds=executor_track_timeout
            )
            progress_evidence_revision = 0
            last_extended_progress_revision = -1
            latest_progress_evidence: str | None = None
            last_progress_observed_at: float | None = None
            terminal_completion_grace_used = False
            last_observed_schedule_index = -1
            last_progress_goal_id: str | None = None
            semantic_progress_anchor_m: float | None = None
            physical_progress_anchor_m: tuple[float, float, float] | None = None
            last_semantic_window_goal_id: str | None = None
            last_semantic_window_progress_revision = -1
            last_semantic_window_schedule_revision = -1
            progress_stall_deadline: float | None = None
            last_progress_check_at = 0.0
            consumed_samples = 0
            while executor.poll() is None:
                loop_now = time.monotonic()
                if loop_now - last_progress_check_at >= 0.5:
                    fresh_progress_state = _fresh_tracking_progress_state(
                        run_dir / "runtime-state" / "closed-loop-tracking.json",
                        now_unix_ms=int(time.time() * 1_000),
                    )
                    last_progress_check_at = loop_now
                    if fresh_progress_state is not None:
                        (
                            progress_observed,
                            progress_kind,
                            last_observed_schedule_index,
                            last_progress_goal_id,
                            semantic_progress_anchor_m,
                            physical_progress_anchor_m,
                        ) = _advance_closed_loop_progress_anchors(
                            fresh_progress_state,
                            last_schedule_index=last_observed_schedule_index,
                            last_goal_id=last_progress_goal_id,
                            semantic_anchor_m=semantic_progress_anchor_m,
                            physical_anchor_m=physical_progress_anchor_m,
                        )
                        if progress_observed:
                            progress_evidence_revision += 1
                            latest_progress_evidence = progress_kind
                            last_progress_observed_at = loop_now
                            progress_stall_deadline = loop_now + 180.0
                    semantic_window = _fresh_semantic_progress_window(
                        run_dir / "runtime-state" / "model-semantic-progress-window.json",
                        now_unix_ms=int(time.time() * 1_000),
                    )
                    if semantic_window is not None:
                        (
                            semantic_goal_id,
                            _semantic_best_distance_m,
                            semantic_revision,
                            authorized_schedule_revision,
                        ) = semantic_window
                        semantic_window_progressed = (
                            semantic_goal_id != last_semantic_window_goal_id
                            or semantic_revision > last_semantic_window_progress_revision
                            or authorized_schedule_revision > last_semantic_window_schedule_revision
                        )
                        if semantic_window_progressed:
                            progress_evidence_revision += 1
                            latest_progress_evidence = (
                                "semantic_goal_changed"
                                if semantic_goal_id != last_semantic_window_goal_id
                                else "semantic_progress_revision_advanced"
                                if semantic_revision > last_semantic_window_progress_revision
                                else "authorized_schedule_revision_advanced"
                            )
                            last_progress_observed_at = loop_now
                            progress_stall_deadline = loop_now + 180.0
                        last_semantic_window_goal_id = semantic_goal_id
                        last_semantic_window_progress_revision = semantic_revision
                        last_semantic_window_schedule_revision = authorized_schedule_revision
                if (
                    progress_stall_deadline is not None
                    and loop_now >= progress_stall_deadline
                    and abort_reason is None
                ):
                    stall_phase = independent_phase_monitor.read(run_dir / "runtime-phase.json")[
                        "phase"
                    ]
                    if _tracking_identity_required(stall_phase):
                        abort_reason = "EXECUTOR_PROGRESS_STALLED"
                    else:
                        progress_stall_deadline = None
                if loop_now >= deadline and abort_reason is None:
                    deadline_phase = independent_phase_monitor.read(run_dir / "runtime-phase.json")[
                        "phase"
                    ]
                    deadline_progress_state = _fresh_tracking_progress_state(
                        run_dir / "runtime-state" / "closed-loop-tracking.json",
                        now_unix_ms=int(time.time() * 1_000),
                    )
                    progress = (
                        deadline_progress_state[0] if deadline_progress_state is not None else None
                    )
                    if _terminal_completion_grace_allowed(
                        deadline_phase,
                        already_used=terminal_completion_grace_used,
                    ):
                        extension = 120.0
                        deadline = time.monotonic() + extension
                        terminal_completion_grace_used = True
                        _write_json(
                            run_dir / "executor-wall-time-extension.json",
                            {
                                "schema_version": ("dronedream.executor-wall-time-extension.v1"),
                                "latest_schedule_index": progress,
                                "extension_seconds": extension,
                                "remaining_budget_seconds": (progress_extension_budget_seconds),
                                "runtime_phase": deadline_phase,
                                "reason": "bounded_terminal_landing_and_cleanup",
                                "updated_at_unix_ms": int(time.time() * 1_000),
                            },
                        )
                    elif (
                        _bounded_executor_subphase_extension_allowed(deadline_phase)
                        and progress_extension_budget_seconds > 0.0
                    ):
                        extension = min(60.0, progress_extension_budget_seconds)
                        deadline = time.monotonic() + extension
                        progress_extension_budget_seconds -= extension
                        _write_json(
                            run_dir / "executor-wall-time-extension.json",
                            {
                                "schema_version": ("dronedream.executor-wall-time-extension.v1"),
                                "latest_schedule_index": progress,
                                "extension_seconds": extension,
                                "remaining_budget_seconds": (progress_extension_budget_seconds),
                                "runtime_phase": deadline_phase,
                                "reason": "bounded_executor_subphase_in_progress",
                                "updated_at_unix_ms": int(time.time() * 1_000),
                            },
                        )
                    elif (
                        _recent_progress_extension_allowed(
                            deadline_progress_state=deadline_progress_state,
                            progress_evidence_revision=progress_evidence_revision,
                            last_extended_progress_revision=(last_extended_progress_revision),
                            last_progress_observed_at=last_progress_observed_at,
                            now=loop_now,
                        )
                        and progress_extension_budget_seconds > 0.0
                    ):
                        extension = min(60.0, progress_extension_budget_seconds)
                        deadline = time.monotonic() + extension
                        progress_extension_budget_seconds -= extension
                        last_extended_progress_revision = progress_evidence_revision
                        _write_json(
                            run_dir / "executor-wall-time-extension.json",
                            {
                                "schema_version": ("dronedream.executor-wall-time-extension.v1"),
                                "latest_schedule_index": progress,
                                "progress_evidence": latest_progress_evidence,
                                "progress_evidence_revision": progress_evidence_revision,
                                "progress_evidence_age_seconds": (
                                    None
                                    if last_progress_observed_at is None
                                    else loop_now - last_progress_observed_at
                                ),
                                "extension_seconds": extension,
                                "remaining_budget_seconds": (progress_extension_budget_seconds),
                                "reason": "fresh_closed_loop_progress_under_slow_simulation",
                                "updated_at_unix_ms": int(time.time() * 1_000),
                            },
                        )
                    else:
                        abort_reason = "EXECUTOR_WALL_TIMEOUT"
                if (
                    depth_safety_supported
                    and depth_worker_process is not None
                    and depth_worker_process.poll() is not None
                    and abort_reason is None
                ):
                    abort_reason = "DEPTH_SAFETY_WORKER_EXITED"
                if (
                    render_replica_process is not None
                    and render_replica_process.poll() is not None
                    and abort_reason is None
                ):
                    abort_reason = "ISOLATED_RENDERER_EXITED"
                with sample_lock:
                    fresh_samples = samples[consumed_samples:]
                    consumed_samples = len(samples)
                for elapsed_s, x, y, z in fresh_samples:
                    center = (x + offset_east, y + offset_north, z + offset_up)
                    phase_path = run_dir / "runtime-phase.json"
                    phase_read = independent_phase_monitor.read(phase_path)
                    phase = phase_read["phase"]
                    if phase_read["issue"] and abort_reason is None:
                        abort_reason = str(phase_read["issue"])
                        live_safety_event = {
                            "schema_version": "dronedream.live-safety-event.v1",
                            "reason": abort_reason,
                            "elapsed_s": elapsed_s,
                            "phase_read": phase_read,
                        }
                        _write_json(run_dir / "live-safety-event.json", live_safety_event)
                    if phase == "TRACK" and goal_tracking_started_at is None:
                        # Buffered ground/takeoff poses must not be relabelled as
                        # task progress when the phase file first switches to TRACK.
                        goal_tracking_started_at = time.monotonic() - started
                    if (
                        goal_tracking_started_at is not None
                        and elapsed_s >= goal_tracking_started_at
                    ):
                        goal_departure_observed, observed_now = _update_goal_progress(
                            center=center,
                            start=route_points[0],
                            goal=route_points[-1],
                            departure_observed=goal_departure_observed,
                            phase=phase,
                        )
                        goal_observed_runtime = goal_observed_runtime or observed_now
                    minimum, minimum_index = min(
                        (
                            _clearance(
                                center,
                                primitive,
                                radius_m=diameter / 2,
                                half_height_m=height / 2,
                            ),
                            index,
                        )
                        for index, primitive in enumerate(primitives)
                    )
                    minimum_primitive = primitives[minimum_index]
                    minimum_name = str(minimum_primitive.get("name", "unknown"))
                    monitored_route = _phase_reference_polyline(
                        phase=phase,
                        spawn_center=tuple(
                            model_root[i] + (offset_east, offset_north, offset_up)[i]
                            for i in range(3)
                        ),
                        route=route_points,
                    )
                    route_deviation = _distance_to_polyline(center, monitored_route)
                    if minimum < -0.001:
                        if _is_tolerated_landing_contact(
                            phase=phase,
                            primitive_name=minimum_name,
                            clearance_m=minimum,
                        ) and _landing_contact_from_above(
                            minimum_primitive, center, diameter / 2, height / 2
                        ):
                            tolerated_landing_contacts += 1
                            minimum_tolerated_landing_clearance = min(
                                minimum_tolerated_landing_clearance, minimum
                            )
                        else:
                            if abort_reason is None:
                                abort_reason = "LIVE_STATIC_COLLISION"
                                live_safety_event = {
                                    "schema_version": "dronedream.live-safety-event.v1",
                                    "reason": abort_reason,
                                    "elapsed_s": elapsed_s,
                                    "vehicle_collision_center_world_enu_m": {
                                        "east_m": center[0],
                                        "north_m": center[1],
                                        "up_m": center[2],
                                    },
                                    "clearance_m": minimum,
                                    "collision_primitive": minimum_name,
                                    "route_deviation_m": route_deviation,
                                    "runtime_phase": phase,
                                }
                                _write_json(run_dir / "live-safety-event.json", live_safety_event)
                    # Landing intentionally leaves the airborne route in the vertical
                    # direction.  Keep collision monitoring active, but close the
                    # route-following phase once the final airborne goal is observed.
                    if (
                        _reference_route_deviation_enforced(
                            phase=phase,
                            goal_observed=goal_observed_runtime,
                            model_control_authority_required=(
                                local_navigation_control_authority_required
                            ),
                        )
                        and route_deviation > 1.0
                        and abort_reason is None
                    ):
                        abort_reason = "LIVE_ROUTE_DEVIATION"
                        live_safety_event = {
                            "schema_version": "dronedream.live-safety-event.v1",
                            "reason": abort_reason,
                            "elapsed_s": elapsed_s,
                            "vehicle_collision_center_world_enu_m": {
                                "east_m": center[0],
                                "north_m": center[1],
                                "up_m": center[2],
                            },
                            "clearance_m": minimum,
                            "nearest_collision_primitive": minimum_name,
                            "route_deviation_m": route_deviation,
                            "goal_departure_observed": goal_departure_observed,
                            "goal_observed": goal_observed_runtime,
                            "runtime_phase": phase,
                            "phase_read": phase_read,
                            "reference_polyline_world_enu_m": monitored_route,
                        }
                        _write_json(run_dir / "live-safety-event.json", live_safety_event)
                if abort_reason and not abort_file.exists():
                    abort_payload: dict[str, object] = {
                        "reason": abort_reason,
                        "world_paused": False,
                    }
                    if live_safety_event is not None:
                        abort_payload["safety_event"] = live_safety_event
                    _write_json(abort_file, abort_payload)
                time.sleep(0.05)
            executor_return_code = executor.wait(timeout=10)
            terminal_timing_path = run_dir / "offboard_timing.json"
            terminal_timing = (
                read_runtime_object(terminal_timing_path) if terminal_timing_path.is_file() else {}
            )
            # A non-zero executor result describes the mission outcome, not the
            # aircraft's physical state.  Once independent PX4 telemetry has
            # confirmed ON_GROUND, the native watchdog must receive the
            # contract-bound terminal lifecycle event before this adapter
            # tears down the observation publisher.  Otherwise a safely landed
            # failure is misreported as a second in-flight observation outage
            # during orderly cleanup.
            if _landing_confirmed(terminal_timing):
                try:
                    native_terminal_lifecycle = _publish_native_terminal_lifecycle(
                        contract_id=contract_id,
                        executor_return_code=executor_return_code,
                        env=ros_environment,
                    )
                    _write_json(
                        run_dir / "native-terminal-lifecycle.json",
                        native_terminal_lifecycle,
                    )
                except SimulationRuntimeError:
                    abort_reason = "NATIVE_TERMINAL_LIFECYCLE_FAILED"
                    raise
    finally:
        primary_error = sys.exception()
        cleanup_errors = []
        subscription_summary = _cleanup_runtime_resource(cleanup_errors, gazebo_subscriptions.close)
        training_witness_summary = (
            _cleanup_runtime_resource(cleanup_errors, training_witness.close)
            if training_witness
            else None
        )
        pose_worker_stop.set()
        pose_available.set()
        if pose_worker is not None:
            _cleanup_runtime_resource(cleanup_errors, _join_pose_worker, pose_worker)
        localization_truth_summary = (
            _cleanup_runtime_resource(cleanup_errors, localization_truth.close)
            if localization_truth
            else None
        )
        # Stop sampling before deliberate process shutdown; otherwise shutdown
        # waits overwrite the finite buffer containing the actual live stall.
        stall_probe_summary = (
            _cleanup_runtime_resource(cleanup_errors, source_stall_probe.close)
            if source_stall_probe
            else None
        )
        for process in reversed(processes):
            _cleanup_runtime_resource(cleanup_errors, _terminate, process)
        preview_summary = (
            _cleanup_runtime_resource(cleanup_errors, preview_worker.close)
            if preview_worker
            else None
        )
        interpreter_summary = (
            _cleanup_runtime_resource(cleanup_errors, observer_pause_monitor.close)
            if observer_pause_monitor
            else None
        )
        if render_cache_deployment is not None:
            try:
                cache_finalization = finalize_render_cache(
                    run_dir / "render-cache",
                    all_owned_processes_exited=all(p.poll() is not None for p in processes),
                    flight_vehicle_spawn_attempted=flight_vehicle_spawn_attempted,
                    timing_path=run_dir / "offboard_timing.json",
                )
            except Exception as cache_error:
                # Keep the actual flight outcome; no partial cache can be reused.
                cache_finalization = {
                    "error": str(cache_error),
                    "complete": False,
                    "qualification_granted": False,
                }
            _cleanup_runtime_resource(
                cleanup_errors,
                _write_json,
                run_dir / "render-cache-finalization.json",
                cache_finalization,
            )
        if not flight_vehicle_spawn_attempted and all(p.poll() is not None for p in processes):
            _cleanup_runtime_resource(
                cleanup_errors,
                _write_json,
                run_dir / "simulation-preflight-stop.json",
                {
                    "run_directory": str(run_dir.resolve()),
                    "flight_vehicle_spawn_attempted": False,
                    "all_started_processes_exited": True,
                    "terminal_state": "NOT_STARTED",
                    "qualification_granted": False,
                },
            )
        _cleanup_runtime_resource(
            cleanup_errors,
            _write_json,
            run_dir / "independent-observer-timing.json",
            {
                "subscription_shutdown": subscription_summary,
                "training_witness": training_witness_summary,
                "localization_truth": localization_truth_summary,
                "interpreter": interpreter_summary,
                "preview": preview_summary,
                "pose_source_intervals": pose_source_intervals.summary(),
                "source_stall_probe": stall_probe_summary,
                "slowest_pose_processing": [
                    row for _, _, row in sorted(pose_processing_longest, reverse=True)
                ],
                "source_clock": "gazebo-transport-receive-wall-clock",
                "qualification_granted": False,
            },
        )
        _cleanup_runtime_resource(cleanup_errors, independent_phase_monitor.close)
        _cleanup_runtime_resource(cleanup_errors, identity_phase_monitor.close)
        _finish_runtime_cleanup(cleanup_errors, primary_error)

    _write_json(
        run_dir / "runtime-state" / "gazebo-pose-stream-health.json",
        {
            "schema_version": "dronedream.gazebo-pose-stream-health.v1",
            "received_frames": pose_messages_received,
            "processed_frames": pose_messages_processed,
            "overwritten_stale_frames": pose_messages_overwritten,
            "latest_value_mailbox": True,
            "accepted": pose_messages_processed > 0,
        },
    )

    with (run_dir / "gazebo_pose_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("elapsed_s", "east_m", "north_m", "model_root_up_m"))
        writer.writerows(samples)

    centers = [(x + offset_east, y + offset_north, z + offset_up) for _, x, y, z in samples]
    goal = route_points[-1]
    minimum_goal_distance = min((math.dist(point, goal) for point in centers), default=math.inf)
    timing_path = run_dir / "offboard_timing.json"
    timing = read_runtime_object(timing_path) if timing_path.is_file() else {}
    model_control_authority = Px4GazeboModelControlAuthorityEvidence.model_validate_json(
        encode_json(timing.get("model_control_authority", {})),
        strict=True,
    ).model_dump(mode="json", exclude_unset=True)
    ros_rows = 0
    ros_path = run_dir / "ros-observations.csv"
    if ros_path.is_file():
        with ros_path.open(encoding="utf-8", errors="replace") as handle:
            ros_rows = sum(1 for line in handle if line.strip())
    px4_ulogs = [
        {
            "path": str(path.relative_to(run_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted((run_dir / "px4_rootfs/log").rglob("*.ulg"))
    ]
    external_abort_request = _read_external_abort_evidence(abort_file)
    if external_abort_request is not None:
        if abort_reason is None:
            abort_reason = external_abort_request["reason"]
        candidate_safety_event = external_abort_request.get("safety_event")
        if live_safety_event is None and isinstance(candidate_safety_event, dict):
            live_safety_event = candidate_safety_event
    controlled_entity_path = run_dir / "controlled-vehicle-entity.json"
    controlled_identity_path = run_dir / "runtime-state" / "controlled-vehicle-identity.json"
    controlled_entity = (
        read_runtime_object(controlled_entity_path) if controlled_entity_path.is_file() else None
    )
    controlled_identity = (
        read_runtime_object(controlled_identity_path)
        if controlled_identity_path.is_file()
        else None
    )
    depth_perception_health: dict[str, Any] | None = None
    if depth_perception_health_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            depth_perception_health = read_runtime_object(depth_perception_health_path)
    runtime_evidence_writer_summary: dict[str, Any] | None = None
    if runtime_evidence_writer_summary_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            candidate_writer_summary = read_runtime_object(runtime_evidence_writer_summary_path)
            if isinstance(candidate_writer_summary, dict):
                runtime_evidence_writer_summary = candidate_writer_summary
    runtime_snapshot_writer_summary: dict[str, Any] | None = None
    if runtime_snapshot_writer_summary_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            candidate_snapshot_summary = read_runtime_object(runtime_snapshot_writer_summary_path)
            if isinstance(candidate_snapshot_summary, dict):
                runtime_snapshot_writer_summary = candidate_snapshot_summary
    multimodal_dataset_summary: dict[str, Any] | None = None
    multimodal_dataset_record_count = 0
    multimodal_semantic_record_count = 0
    multimodal_dataset_records_path: Path | None = None
    if multimodal_dataset_root is not None:
        multimodal_dataset_summary_path = multimodal_dataset_root / "summary.json"
        multimodal_dataset_records_path = multimodal_dataset_root / "records.jsonl"
        if multimodal_dataset_summary_path.is_file():
            with suppress(OSError, json.JSONDecodeError):
                candidate_summary = read_runtime_object(multimodal_dataset_summary_path)
                if isinstance(candidate_summary, dict):
                    multimodal_dataset_summary = candidate_summary
        if multimodal_dataset_records_path.is_file():
            for record in _runtime_rows(multimodal_dataset_records_path):
                multimodal_dataset_record_count += 1
                if all(
                    record.get(field) is not None
                    for field in (
                        "semantic_mask_relative_path",
                        "semantic_mask_sha256",
                        "semantic_label_map_sha256",
                        "semantic_sample_monotonic_seconds",
                        "rgb_semantic_time_offset_seconds",
                    )
                ):
                    multimodal_semantic_record_count += 1
    development_fault_injection: dict[str, Any] | None = None
    if development_depth_fault_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            candidate_fault = read_runtime_object(development_depth_fault_path)
            if isinstance(candidate_fault, dict):
                development_fault_injection = candidate_fault
    model_navigation_cycle_path = run_dir / "model-navigation-cycles.jsonl"
    model_navigation_call_path = run_dir / "model-navigation-model-calls.jsonl"
    model_navigation_snapshot_path = run_dir / "model-navigation-snapshots.jsonl"
    model_navigation_timing_path = run_dir / "model-navigation-timing.jsonl"
    local_policy_selection_path = run_dir / "local-policy-selection.json"
    local_policy_selection: dict[str, Any] | None = None
    local_control_timing_path = run_dir / "local-control-timing.json"
    local_control_timing: dict[str, Any] | None = None
    if local_control_timing_path.is_file():
        with suppress(OSError, ValueError):
            timing_payload = read_runtime_object(local_control_timing_path)
            if isinstance(timing_payload, dict):
                local_control_timing = timing_payload
    if local_policy_selection_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            local_policy_selection = read_runtime_object(local_policy_selection_path)
    local_safety_executor_history_path = (
        run_dir / "runtime-state" / "local-safety-executor-history.jsonl"
    )
    control_applications_path = run_dir / "runtime-state" / "control-applications.jsonl"
    learning_observations_path = run_dir / "learning-observations.jsonl"
    learning_observation_summary_path = run_dir / "learning-observation-summary.json"
    learning_observation_record_count = 0
    if learning_observations_path.is_file():
        try:
            for record in _runtime_rows(learning_observations_path):
                if not isinstance(record.get("snapshot"), dict):
                    raise ValueError("learning observation record is invalid")
                learning_observation_record_count += 1
        except (OSError, ValueError):
            learning_observation_record_count = -1
    learning_observation_summary = {}
    if learning_observation_summary_path.is_file():
        with suppress(OSError, ValueError):
            learning_observation_summary = read_runtime_object(learning_observation_summary_path)
    model_navigation_visual_frames = sorted(
        (run_dir / "model-navigation-frames").glob("forward-rgb-*.png")
    )
    depth_safety_history = _runtime_rows(depth_safety_history_path)
    depth_safety_history_count = len(depth_safety_history)
    local_safety_executor_history_count = len(_runtime_rows(local_safety_executor_history_path))
    model_navigation_cycles = _runtime_rows(model_navigation_cycle_path)
    model_navigation_calls = _runtime_rows(model_navigation_call_path)
    model_navigation_snapshots = _runtime_rows(model_navigation_snapshot_path)
    control_applications = []
    control_application_read_error = None
    if control_applications_path.is_file():
        try:
            control_applications = _runtime_rows(control_applications_path)
        except (OSError, ValueError) as error:
            control_application_read_error = type(error).__name__
    application_counts = model_control_authority.get("control_application_counts", {})
    application_count = (
        sum(application_counts.values())
        if isinstance(application_counts, dict)
        and all(type(v) is int and v >= 0 for v in application_counts.values())
        else -1
    )
    from .runtime_evidence import control_evidence_inventory

    control_writer_inventory = control_evidence_inventory(run_dir)
    control_application_verification = verify_control_applications(
        control_applications,
        navigation_snapshots=model_navigation_snapshots,
        command_records=depth_safety_history,
        expected_count=application_count,
        writer_summary=timing.get("control_application_writer"),
        writer_artifact_counts=control_writer_inventory,
        application_artifact_path=str(control_applications_path),
    )
    if control_application_read_error:
        control_application_verification["accepted"] = False
        control_application_verification["issue_codes"].append("CONTROL_APPLICATION_READ_FAILED")
    from .runtime_evidence import navigation_evidence_inventory

    runtime_evidence_artifact_counts = navigation_evidence_inventory(run_dir)
    runtime_evidence_record_count = sum(runtime_evidence_artifact_counts.values())
    model_navigation_call_by_id = {
        str(call.get("call_id")): call
        for call in model_navigation_calls
        if isinstance(call, dict) and isinstance(call.get("call_id"), str)
    }
    applied_model_navigation_cycles = [
        cycle
        for cycle in model_navigation_cycles
        if isinstance(cycle, dict)
        and _model_navigation_cycle_is_applied(
            cycle,
            call_by_id=model_navigation_call_by_id,
        )
    ]
    model_cycle_timestamps_ms = sorted(
        int(cycle["recorded_at_unix_ms"])
        for cycle in model_navigation_cycles
        if isinstance(cycle.get("recorded_at_unix_ms"), int)
    )
    maximum_model_cycle_gap_seconds = (
        max(
            (later - earlier) / 1_000.0
            for earlier, later in zip(
                model_cycle_timestamps_ms,
                model_cycle_timestamps_ms[1:],
                strict=False,
            )
        )
        if len(model_cycle_timestamps_ms) >= 2
        else None
    )
    active_command_timestamps_ms = [
        int(record["recorded_at_unix_ms"])
        for record in depth_safety_history
        if record.get("identity_accepted") is True
        and isinstance(record.get("command"), dict)
        and isinstance(record.get("recorded_at_unix_ms"), int)
    ]
    maximum_model_participation_gap_seconds = _maximum_model_participation_gap_seconds(
        active_command_timestamps_ms=active_command_timestamps_ms,
        model_call_records=model_navigation_calls,
    )
    model_participation_gap_limit_seconds = max(
        60.0,
        local_navigation_model_timeout_seconds * 3.0 + local_navigation_period_seconds * 3.0,
    )
    continuous_model_authority = continuous_control_evidence_required(
        local_navigation_provider, local_navigation_control_authority_required
    )
    if continuous_model_authority:
        model_participation_gap_limit_seconds = LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
    active_local_safety_sequence = 0
    if active_local_safety_observation_path.is_file():
        with suppress(OSError, ValueError):
            active_local_safety_sequence = _runtime_contract(
                active_local_safety_observation_path, RuntimeLocalSafetyObservation
            ).sequence
    sensor_subscription_summary = None
    sensor_subscription_path = run_dir / "sensor-subscription-shutdown.json"
    if sensor_subscription_path.is_file():
        with suppress(OSError, ValueError):
            sensor_subscription_summary = read_runtime_object(sensor_subscription_path)
    model_shutdown = None
    model_shutdown_path = run_dir / "model-navigation-shutdown.json"
    if model_shutdown_path.is_file():
        with suppress(OSError, ValueError):
            model_shutdown = read_runtime_object(model_shutdown_path)
    gates = {
        "native_observer_subscriptions_closed": subscription_shutdown_is_complete(
            subscription_summary
        ),
        "executor_completed": executor_return_code == 0,
        "offboard_timing_complete": timing.get("status") == "complete",
        "runtime_pose_samples_present": len(samples) >= 10,
        "ros_observations_present": ros_rows >= 1,
        "goal_observed": goal_observed_runtime,
        "landing_confirmed": (_landing_confirmed(timing)),
        "native_terminal_lifecycle_published": native_terminal_lifecycle is not None,
        "no_live_abort": abort_reason is None and external_abort_request is None,
        "px4_ulog_present": bool(px4_ulogs),
        "static_route_clearance_bound": clearance.accepted,
        # Deliberate fault campaigns are valuable training evidence, but must
        # never be mistaken for an operational qualification run.
        "development_fault_injection_absent": (
            development_depth_drop_after_seconds is None and development_fault_injection is None
        ),
        "development_payload_collection_absent": not development_payload_collection,
        "offline_training_authority_absent": simulation_training_channel is None,
    }
    if local_safety_supported:
        if record_learning_observations:
            gates["learning_observation_recording_complete"] = bool(
                isinstance(learning_observation_summary, dict)
                and learning_observation_summary.get("complete") is True
                and learning_observation_summary.get("completed", 0) > 0
                and learning_observation_summary.get("completed")
                == learning_observation_record_count
                and learning_observations_path.is_file()
            )
        if simulation_teacher_control:
            gates["teacher_execution_recording_complete"] = control_application_verification.get(
                "record_count", 0
            ) > 0 and set(control_application_verification.get("issue_codes", [])) == {
                "CONTROL_APPLICATION_HAS_NO_MODEL_MOTION"
            }
        gates.update(
            {
                "local_safety_observation_present": (
                    active_local_safety_observation_path.is_file()
                ),
                "local_safety_command_present": active_local_safety_command_path.is_file(),
                "local_safety_observation_sequence_advanced": (active_local_safety_sequence >= 1),
            }
        )
    if depth_safety_supported:
        gates.update(
            {
                "native_sensor_subscriptions_closed": subscription_shutdown_is_complete(
                    sensor_subscription_summary
                ),
                "live_depth_perception_healthy": verify_control_completion(timing),
                "live_depth_metric_map_present": (
                    (run_dir / "metric-local-world-summary.json").is_file()
                ),
                "live_depth_safety_history_present": depth_safety_history_count >= 1,
                "runtime_evidence_writer_complete": (
                    _runtime_evidence_writer_complete(
                        runtime_evidence_writer_summary,
                        record_count=runtime_evidence_record_count,
                        artifact_counts=runtime_evidence_artifact_counts,
                    )
                ),
                "runtime_snapshot_writer_complete": (
                    _runtime_snapshot_writer_complete(runtime_snapshot_writer_summary)
                ),
            }
        )
    if multimodal_dataset_root is not None:
        gates.update(
            {
                "multimodal_dataset_summary_present": (multimodal_dataset_summary is not None),
                "multimodal_dataset_records_present": (multimodal_dataset_record_count >= 1),
                "multimodal_dataset_summary_consistent": bool(
                    multimodal_dataset_summary is not None
                    and multimodal_dataset_summary.get("record_count")
                    == multimodal_dataset_record_count
                    and multimodal_dataset_summary.get("flight_id") == multimodal_flight_id
                    and multimodal_dataset_summary.get("map_sha256") == _sha256(semantic_path)
                    and multimodal_dataset_summary.get("issue_code") is None
                    and multimodal_dataset_summary.get("qualification_granted") is False
                ),
            }
        )
        if semantic_label_topic is not None:
            gates["semantic_supervision_recorded_for_every_sample"] = bool(
                multimodal_dataset_record_count >= 1
                and multimodal_semantic_record_count == multimodal_dataset_record_count
            )
    if local_navigation_provider is not None:
        gates["model_navigation_shutdown_complete"] = bool(
            isinstance(model_shutdown, dict) and model_shutdown.get("complete") is True
            and model_shutdown.get("pending_call") is False
        )
        if continuous_model_authority:
            gates["actual_model_control_input_deadlines_bounded"] = (
                control_application_verification["accepted"]
            )
            gates["continuous_control_effective_timing_bounded"] = continuous_timing_is_bounded(
                local_control_timing,
            )
            native_publication = timing.get("native_state_publication")
            gates["independent_native_state_publication_drained"] = bool(
                isinstance(native_publication, dict)
                and native_publication.get("drained") is True
                and native_publication.get("unique_source_count", 0) >= 16
                and native_publication.get("last_issue") is None
            )
        shadow_observation_only = not local_navigation_control_authority_required
        gates.update(
            {
                "model_navigation_cycle_recorded": bool(model_navigation_cycles),
                "model_navigation_provider_call_recorded": bool(model_navigation_calls),
                "model_navigation_primary_provider_call_recorded": any(
                    call.get("provider") == local_navigation_provider
                    for call in model_navigation_calls
                ),
                "model_navigation_snapshot_recorded": bool(model_navigation_snapshots),
                "model_navigation_provider_cadence_bounded": bool(
                    continuous_control_cadence_bounded(
                        control_application_verification, maximum_model_participation_gap_seconds
                    )
                    if continuous_model_authority
                    else (
                        maximum_model_participation_gap_seconds is not None
                        and maximum_model_participation_gap_seconds
                        <= model_participation_gap_limit_seconds
                    )
                ),
                "model_navigation_invocation_failure_absent": not any(
                    cycle.get("hold_reason")
                    in {
                        "MODEL_NAVIGATION_INVOCATION_FAILED",
                        "MODEL_NAVIGATION_INVOCATION_TIMEOUT",
                    }
                    for cycle in model_navigation_cycles
                ),
                "model_navigation_fresh_revalidation_stable": not any(
                    cycle.get("hold_reason") == "PERCEPTION_CHANGED_TO_UNHEALTHY_DURING_MODEL_CALL"
                    for cycle in model_navigation_cycles
                ),
            }
        )
        if shadow_observation_only:
            gates["model_navigation_shadow_only"] = bool(
                int(model_control_authority.get("authorized_control_applied_count", 0)) == 0
                and int(model_control_authority.get("authorized_schedule_advance_count", 0)) == 0
                and int(model_control_authority.get("route_fallback_schedule_advance_count", 0)) > 0
            )
        else:
            gates["model_navigation_authorized_control_applied"] = bool(
                applied_model_navigation_cycles
                and int(model_control_authority.get("authorized_control_applied_count", 0)) > 0
            )
        if local_navigation_visual_enabled:
            gates["model_navigation_visual_frame_recorded"] = bool(model_navigation_visual_frames)
        if local_navigation_provider == "local-policy":
            local_policy_selection_recorded = bool(
                local_policy_selection is not None
                and local_policy_selection.get("qualification_receipt_id")
                and local_policy_selection.get("package_sha256")
            )
            if local_policy_simulation_admission_paths:
                gates["local_policy_simulation_admission_recorded"] = bool(
                    local_policy_selection_recorded
                    and local_policy_selection is not None
                    and local_policy_selection.get("simulation_only") is True
                )
            else:
                gates["local_policy_qualified_selection_recorded"] = bool(
                    local_policy_selection_recorded
                    and local_policy_selection is not None
                    and local_policy_selection.get("simulation_only") is not True
                )
        if local_navigation_control_authority_required:
            gates.update(
                {
                    "model_navigation_control_authority_required": (
                        model_control_authority.get("required") is True
                    ),
                    "model_navigation_authorized_control_applied": (
                        gates["model_navigation_authorized_control_applied"]
                    ),
                    "model_navigation_authorized_schedule_advance_recorded": int(
                        model_control_authority.get("authorized_schedule_advance_count", 0)
                    )
                    > 0,
                    "model_navigation_route_fallback_absent": int(
                        model_control_authority.get("route_fallback_schedule_advance_count", 0)
                    )
                    == 0,
                }
            )
    if tracking_identity_supported:
        gates.update(
            {
                "controlled_vehicle_entity_selected": controlled_entity is not None,
                "controlled_vehicle_identity_confirmed": bool(
                    controlled_identity is not None and controlled_identity.get("accepted")
                ),
            }
        )
    evidence = {
        "schema_version": "dronedream.generic-px4-gazebo-run.v1",
        "status": "verified" if all(gates.values()) else "failed",
        "world": world_name,
        "vehicle": vehicle_name,
        "gates": gates,
        "measurements": {
            "native_sensor_deployment": sensor_deployment,
            "pose_sample_count": len(samples),
            "ros_observation_rows": ros_rows,
            "minimum_goal_distance_m": (
                minimum_goal_distance if math.isfinite(minimum_goal_distance) else None
            ),
            "goal_departure_observed": goal_departure_observed,
            "goal_observed_runtime": goal_observed_runtime,
            "live_safety_event": live_safety_event,
            "external_abort_request": external_abort_request,
            "landing_state": (
                timing.get("cleanup", {}).get("landing_observation", {}).get("state")
            ),
            "abort_reason": abort_reason,
            "executor_return_code": executor_return_code,
            "tolerated_landing_contact_samples": tolerated_landing_contacts,
            "minimum_tolerated_landing_clearance_m": (
                minimum_tolerated_landing_clearance
                if math.isfinite(minimum_tolerated_landing_clearance)
                else None
            ),
            "local_safety": {
                "supported_by_executor": local_safety_supported,
                "observation_sequence": active_local_safety_sequence,
                "required_clearance_m": local_required_clearance_m,
                "identity_correction_limit_m": identity_correction_limit_m,
                "authority_source": (
                    "live-depth-metric-fusion"
                    if depth_safety_supported
                    else "simulation-ground-truth"
                ),
                "observation_artifact": (
                    active_local_safety_observation_path.name
                    if active_local_safety_observation_path.is_file()
                    else None
                ),
                "command_artifact": (
                    active_local_safety_command_path.name
                    if active_local_safety_command_path.is_file()
                    else None
                ),
                "depth_perception_health": depth_perception_health,
                "perception_control_completion": timing.get("perception_control_completion"),
                "depth_safety_history_count": depth_safety_history_count,
                "runtime_evidence_writer_summary": runtime_evidence_writer_summary,
                "runtime_evidence_artifact_counts": runtime_evidence_artifact_counts,
                "runtime_snapshot_writer_summary": runtime_snapshot_writer_summary,
                "executor_history_artifact": (
                    str(local_safety_executor_history_path.relative_to(run_dir))
                    if local_safety_executor_history_path.is_file()
                    else None
                ),
                "executor_history_count": local_safety_executor_history_count,
            },
            "model_navigation": {
                "enabled": local_navigation_provider is not None,
                "provider": local_navigation_provider,
                "fallback_provider": local_navigation_fallback_provider,
                "primary_provider_call_count": sum(
                    1
                    for call in model_navigation_calls
                    if call.get("provider") == local_navigation_provider
                ),
                "fallback_provider_call_count": sum(
                    1
                    for call in model_navigation_calls
                    if call.get("provider") == local_navigation_fallback_provider
                ),
                "cycle_count": len(model_navigation_cycles),
                "provider_call_count": len(model_navigation_calls),
                "snapshot_count": len(model_navigation_snapshots),
                "motion_proposal_count": len(applied_model_navigation_cycles),
                "authorized_control_applied_count": int(
                    model_control_authority.get("authorized_control_applied_count", 0)
                ),
                "model_timeout_seconds": local_navigation_model_timeout_seconds,
                "fallback_model_timeout_seconds": (
                    local_navigation_fallback_model_timeout_seconds
                    if local_navigation_fallback_provider is not None
                    else None
                ),
                "cycle_period_seconds": (
                    local_control_timing["effective"]["decision_period_seconds"]
                    if continuous_timing_is_bounded(local_control_timing)
                    else local_navigation_period_seconds
                ),
                "control_authority": "deterministic-local-safety",
                "model_control_authority_required": (local_navigation_control_authority_required),
                "model_control_authority_evidence": model_control_authority,
                "controller_step_m": model_navigation_controller_step_m,
                "visual_enabled": local_navigation_visual_enabled,
                "visual_frame_count": len(model_navigation_visual_frames),
                "local_policy_selection": local_policy_selection,
                "maximum_cycle_gap_seconds": maximum_model_cycle_gap_seconds,
                "maximum_provider_participation_gap_seconds": (
                    maximum_model_participation_gap_seconds
                ),
                "provider_participation_gap_limit_seconds": (model_participation_gap_limit_seconds),
                "invocation_failure_count": sum(
                    1
                    for cycle in model_navigation_cycles
                    if cycle.get("hold_reason") == "MODEL_NAVIGATION_INVOCATION_FAILED"
                ),
                "invocation_timeout_count": sum(
                    1
                    for cycle in model_navigation_cycles
                    if cycle.get("hold_reason") == "MODEL_NAVIGATION_INVOCATION_TIMEOUT"
                ),
                "revalidation_timeout_count": sum(
                    1
                    for cycle in model_navigation_cycles
                    if cycle.get("hold_reason") == "MODEL_NAVIGATION_REVALIDATION_TIMEOUT"
                ),
                "perception_changed_during_model_call_count": sum(
                    1
                    for cycle in model_navigation_cycles
                    if cycle.get("hold_reason")
                    == "PERCEPTION_CHANGED_TO_UNHEALTHY_DURING_MODEL_CALL"
                ),
            },
            "multimodal_dataset": {
                "enabled": multimodal_dataset_root is not None,
                "flight_id": multimodal_flight_id,
                "record_count": multimodal_dataset_record_count,
                "semantic_record_count": multimodal_semantic_record_count,
                "record_period_seconds": (
                    multimodal_record_period_seconds
                    if multimodal_dataset_root is not None
                    else None
                ),
                "maximum_mib": (
                    multimodal_dataset_maximum_mib if multimodal_dataset_root is not None else None
                ),
                "semantic_topic": semantic_label_topic,
                "summary": multimodal_dataset_summary,
            },
            "controlled_vehicle_entity": controlled_entity,
            "simulation_learning": {
                "observations_recorded": record_learning_observations,
                "deterministic_teacher_control": simulation_teacher_control,
                "model_control_qualification_granted": False,
                "observation_summary": learning_observation_summary,
            },
            "controlled_vehicle_identity": controlled_identity,
            "development_fault_injection": development_fault_injection,
            "development_payload_collection": {
                "enabled": development_payload_collection,
                "development_only": development_payload_collection,
                "flight_qualification_granted": False if development_payload_collection else None,
                "maximum_controller_step_scale": (0.2 if development_payload_collection else None),
                "fresh_px4_dynamics_required": development_payload_collection,
            },
        },
        "artifacts": {
            "world_sha256": _sha256(world_sdf),
            "render_world_sha256": _sha256(render_world),
            "static_render_batching": render_batching,
            "semantic_sha256": _sha256(semantic_path),
            "vehicle_sha256": _sha256(vehicle_sdf),
            "route_sha256": _sha256(route_path),
            "track_sha256": _sha256(track_path),
            "clearance_sha256": _sha256(clearance_path),
            "controller_params_sha256": _sha256(controller_params_path),
            "executor_sha256": _sha256(executor_path),
            "px4_ulogs": px4_ulogs,
            "ros_workspace": str(ros_workspace),
            "model_navigation_cycles_sha256": (
                _sha256(model_navigation_cycle_path)
                if model_navigation_cycle_path.is_file()
                else None
            ),
            "model_navigation_calls_sha256": (
                _sha256(model_navigation_call_path)
                if model_navigation_call_path.is_file()
                else None
            ),
            "model_navigation_snapshots_sha256": (
                _sha256(model_navigation_snapshot_path)
                if model_navigation_snapshot_path.is_file()
                else None
            ),
            "model_navigation_timing_sha256": (
                _sha256(model_navigation_timing_path)
                if model_navigation_timing_path.is_file()
                else None
            ),
            "local_policy_selection_sha256": (
                _sha256(local_policy_selection_path)
                if local_policy_selection_path.is_file()
                else None
            ),
            "depth_safety_history_sha256": (
                _sha256(depth_safety_history_path) if depth_safety_history_path.is_file() else None
            ),
            "runtime_evidence_writer_summary_sha256": (
                _sha256(runtime_evidence_writer_summary_path)
                if runtime_evidence_writer_summary_path.is_file()
                else None
            ),
            "runtime_snapshot_writer_summary_sha256": (
                _sha256(runtime_snapshot_writer_summary_path)
                if runtime_snapshot_writer_summary_path.is_file()
                else None
            ),
            "local_safety_executor_history_sha256": (
                _sha256(local_safety_executor_history_path)
                if local_safety_executor_history_path.is_file()
                else None
            ),
            "control_applications_sha256": (
                _sha256(control_applications_path) if control_applications_path.is_file() else None
            ),
            "learning_observations_sha256": (
                _sha256(learning_observations_path)
                if learning_observations_path.is_file()
                else None
            ),
            "learning_observation_summary_sha256": (
                _sha256(learning_observation_summary_path)
                if learning_observation_summary_path.is_file()
                else None
            ),
            "control_application_verification": control_application_verification,
            "local_control_timing_sha256": (
                _sha256(local_control_timing_path) if local_control_timing_path.is_file() else None
            ),
            "development_fault_injection_sha256": (
                _sha256(development_depth_fault_path)
                if development_depth_fault_path.is_file()
                else None
            ),
            "multimodal_dataset_root": (
                str(multimodal_dataset_root.resolve())
                if multimodal_dataset_root is not None
                else None
            ),
            "multimodal_dataset_records_sha256": (
                _sha256(multimodal_dataset_records_path)
                if multimodal_dataset_records_path is not None
                and multimodal_dataset_records_path.is_file()
                else None
            ),
            "semantic_label_map_sha256": (
                _sha256(semantic_label_map_path) if semantic_label_map_path is not None else None
            ),
            "model_navigation_visual_frames": [
                {
                    "path": str(path.relative_to(run_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in model_navigation_visual_frames
            ],
        },
    }
    _verify_runtime_inputs(input_hashes)
    evidence = _validated_runtime_evidence(evidence)
    _write_json(run_dir / "mission_evidence.json", evidence)
    return evidence
