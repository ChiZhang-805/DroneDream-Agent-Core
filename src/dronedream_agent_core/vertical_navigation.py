"""实时竖直运动约束：米制净空、门檐过渡、覆盖证据与负载资格；不生成模型动作。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .collision import primitive_bounds
from .contracts import LocalPlannerRequest, StrictModel, VehicleAsset
from .hashing import sha256_json
from .plugin_files import read_plugin_file


class FlightDynamicsEnvelope(StrictModel):
    """已测量的机型／负载范围；模型推测、额定最大加速度不能充当制动下界。"""

    schema_version: Literal["dronedream.flight-dynamics-envelope.v1"] = (
        "dronedream.flight-dynamics-envelope.v1"
    )
    vehicle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    domain: Literal["simulation", "hardware"]
    minimum_total_mass_kg: float = Field(gt=0, le=70)
    maximum_total_mass_kg: float = Field(gt=0, le=70)
    maximum_climb_mps: float = Field(gt=0, le=20)
    maximum_descent_mps: float = Field(gt=0, le=20)
    maximum_horizontal_mps: float = Field(gt=0, le=20)
    maximum_command_acceleration_mps2: float = Field(gt=0, le=30)
    braking_deceleration_lower_bound_mps2: float = Field(gt=0, le=30)
    response_delay_upper_bound_seconds: float = Field(gt=0, le=2)
    minimum_battery_voltage_v: float = Field(gt=0, le=200)
    maximum_tilt_deg: float = Field(gt=0, le=60)
    maximum_actuator_fraction: float = Field(gt=0, le=1)
    collision_radius_m: float = Field(gt=0, le=3)
    collision_height_m: float = Field(gt=0, le=3)

    # 功能：
    #   拒绝颠倒的质量范围及自相矛盾的加速度资格。
    # 输入：
    #   self：声明了测量证据与有效运行条件的包络。
    # 输出：
    #   self：内部约束一致的包络，不代表证据已通过独立验收。
    @model_validator(mode="after")
    def validate_limits(self):
        if self.minimum_total_mass_kg > self.maximum_total_mass_kg:
            raise ValueError("DYNAMICS_MASS_INTERVAL_EMPTY")
        if self.braking_deceleration_lower_bound_mps2 > self.maximum_command_acceleration_mps2:
            raise ValueError("DYNAMICS_BRAKING_EXCEEDS_ACCELERATION_LIMIT")
        return self


# 功能：
#   1. 读取独立的测量验收文件，并逐文件核验原始轨迹摘要，禁止只填一个“合格”标志。
#   2. 逐字段比对验收的机型、适用环境和物理限制，禁止使用其他负载的测量记录。
# 输入：
#   path：包络 JSON；旁边的 evidence/<摘要>.json 引用同目录下的原始测量文件。
# 输出：
#   envelope：文件完整性和已验收限制一致的包络，不由此函数自动签发物理资格。
def load_flight_dynamics_envelope(path: Path) -> FlightDynamicsEnvelope:
    path = Path(path)
    envelope = FlightDynamicsEnvelope.model_validate_json(
        read_plugin_file(path, limit=65_536), strict=True)
    evidence_dir = path.parent / "evidence"
    raw = read_plugin_file(evidence_dir / (envelope.evidence_sha256 + ".json"), limit=1_048_576)
    if hashlib.sha256(raw).hexdigest() != envelope.evidence_sha256:
        raise ValueError("FLIGHT_DYNAMICS_EVIDENCE_HASH_MISMATCH")
    report = json.loads(raw)
    expected = envelope.model_dump(mode="json", exclude={"evidence_sha256"})
    if (not isinstance(report, dict)
            or report.get("schema_version") != "dronedream.flight-dynamics-qualification.v1"
            or report.get("status") != "accepted"
            or sha256_json(report.get("limits")) != sha256_json(expected)
            or report.get("independent_validation_passed") is not True):
        raise ValueError("FLIGHT_DYNAMICS_QUALIFICATION_MISMATCH")
    sources = report.get("measurement_sources")
    if not isinstance(sources, list) or not 2 <= len(sources) <= 256:
        raise ValueError("FLIGHT_DYNAMICS_MEASUREMENTS_MISSING")
    splits, seen, total_bytes = set(), set(), 0
    for item in sources:
        if (not isinstance(item, dict) or set(item) != {"file", "sha256", "split"}
                or item["split"] not in {"calibration", "validation"}
                or not isinstance(item["file"], str)
                or Path(item["file"]).name != item["file"]
                or any(char in item["file"] for char in "\\/:\x00")
                or item["file"] in {".", "..", ""}):
            raise ValueError("FLIGHT_DYNAMICS_MEASUREMENT_REFERENCE_INVALID")
        data = read_plugin_file(evidence_dir / item["file"], limit=16 * 1024**2)
        total_bytes += len(data)
        if total_bytes > 128 * 1024**2:
            raise ValueError("FLIGHT_DYNAMICS_MEASUREMENT_BUDGET_EXCEEDED")
        digest = hashlib.sha256(data).hexdigest()
        if not data or digest != item["sha256"] or digest in seen:
            raise ValueError("FLIGHT_DYNAMICS_MEASUREMENT_HASH_OR_SPLIT_INVALID")
        splits.add(item["split"])
        seen.add(digest)
    if splits != {"calibration", "validation"}:
        raise ValueError("FLIGHT_DYNAMICS_INDEPENDENT_VALIDATION_MISSING")
    return envelope


@dataclass(frozen=True)
class HeightInterval:
    """世界 ENU 中机体碰撞中心的允许区间；None 表示未由本批几何建立边界。"""

    minimum_center_z_m: float | None
    maximum_center_z_m: float | None
    floor_z_m: float | None
    ceiling_z_m: float | None


# 功能：
#   拒绝非有限物理量与布尔伪数值，避免坏输入变成放行依据。
# 输入：
#   value：待核对的物理量。
# 输出：
#   valid：值是否为有限实数。
def _finite(value) -> bool:
    try:
        valid = type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   1. 按整个机身水平占用范围查询上下边界，不只看中心点下方。
#   2. 使用包含旋转的保守包围盒；侧向墙体留给完整三维碰撞检查。
# 输入：
#   position：机体碰撞中心的世界 ENU 米制坐标。
#   bounds：经过验证的静态／感知图元包围盒。
#   radius、half_height、margin：机体半径、半高及额外安全余量，单位米。
# 输出：
#   interval：考虑机体尺寸后的高度区间及原始地面／顶面高度。
def height_interval(position, bounds, radius, half_height, margin) -> HeightInterval:
    values = (*position, radius, half_height, margin)
    if (not all(_finite(v) for v in values) or radius <= 0 or half_height <= 0 or margin < 0):
        raise ValueError("HEIGHT_INTERVAL_INPUT_INVALID")
    floor, ceiling = None, None
    reach = radius + margin
    for low, high in bounds:
        if (high[0] < position[0] - reach or low[0] > position[0] + reach
                or high[1] < position[1] - reach or low[1] > position[1] + reach):
            continue
        if high[2] <= position[2]:
            floor = high[2] if floor is None else max(floor, high[2])
        elif low[2] >= position[2]:
            ceiling = low[2] if ceiling is None else min(ceiling, low[2])
    interval = HeightInterval(
        None if floor is None else floor + half_height + margin,
        None if ceiling is None else ceiling - half_height - margin,
        floor, ceiling,
    )
    return interval


class CeilingTransition:
    """门檐退出迟滞：收紧立即生效，放宽需要连续稳定；不靠 indoor/outdoor 标签切换。"""

    # 功能：
    #   建立仅归属当前运行实例的顶面迟滞状态，不跨任务保存旧地图限制。
    # 输入：
    #   stable_seconds：顶部空间持续变宽后允许解除旧限制的秒数。
    # 输出：
    #   None：初始化内部状态。
    def __init__(self, stable_seconds: float = 0.4):
        if not _finite(stable_seconds) or not 0.1 <= stable_seconds <= 2:
            raise ValueError("CEILING_TRANSITION_WINDOW_INVALID")
        self.stable_seconds = stable_seconds
        self.ceiling = None
        self.pending_since = None
        self.last_time = None

    # 功能：
    #   顶部限制收紧时立即采用；只有更新的连续观测才累计退出门檐的稳定时间。
    # 输入：
    #   upper_z：当前全机身已离开门檐后测得的中心高度上限，未知时为 None。
    #   source_seconds：原始观测的单调时刻，不使用重复处理的当前时间续期。
    # 输出：
    #   ceiling：本周期保守保留的中心高度上限。
    def update(self, upper_z: float | None, source_seconds: float) -> float | None:
        if (not _finite(source_seconds) or source_seconds < 0
                or upper_z is not None and not _finite(upper_z)):
            raise ValueError("CEILING_TRANSITION_OBSERVATION_INVALID")
        if self.last_time is not None and source_seconds < self.last_time:
            raise ValueError("CEILING_TRANSITION_CLOCK_REGRESSED")
        if self.last_time == source_seconds:
            # 重复观测可以收紧，但不能推进解除限制的时钟。
            if upper_z is not None:
                tightened = upper_z if self.ceiling is None else min(self.ceiling, upper_z)
                if tightened != self.ceiling:
                    self.pending_since = None
                self.ceiling = tightened
            return self.ceiling
        if self.last_time is not None and source_seconds - self.last_time > self.stable_seconds:
            self.pending_since = None
        self.last_time = source_seconds
        if self.ceiling is None or upper_z is not None and upper_z <= self.ceiling:
            self.ceiling, self.pending_since = upper_z, None
        else:
            if self.pending_since is None:
                self.pending_since = source_seconds
            if source_seconds - self.pending_since >= self.stable_seconds:
                self.ceiling, self.pending_since = upper_z, None
        return self.ceiling


# 功能：
#   1. 直接核对实测负载、机型总重和动力学适用范围，不让专家缩杆覆盖硬限制。
#   2. 已卸载仍受提供的机型包络约束；缺少负载质量、低电压或姿态超界均拒绝授权。
# 输入：
#   vehicle、payload：原始机型及同周期装卸／遥测背景。
#   envelope：可选的独立测量包络；domain：本次仿真或真机环境。
#   measurement_only：显式仿真采集权限，不授予正式飞行资格。
# 输出：
#   issues：本周期不能放行的具体原因列表。
def payload_motion_issues(vehicle, payload, envelope, domain, measurement_only=False) -> list[str]:
    if type(measurement_only) is not bool or measurement_only and domain != "simulation":
        raise ValueError("FLIGHT_DYNAMICS_MEASUREMENT_DOMAIN_INVALID")
    loaded = payload.get("state") in {"attached", "custody-confirmed", "loaded-stable"}
    mass = payload.get("observed_payload_mass_kg") if loaded else 0.0
    issues = []
    if domain == "hardware":
        if envelope is None:
            issues.append("HARDWARE_MOTION_DYNAMICS_UNQUALIFIED")
        if payload.get("state") not in {"detached", "loaded-stable"}:
            issues.append("HARDWARE_PAYLOAD_STATE_UNKNOWN")
    if loaded and payload.get("state") != "loaded-stable":
        issues.append("PAYLOAD_TRANSITION_UNSETTLED")
    if loaded and (not _finite(mass) or mass <= 0):
        return ["PAYLOAD_MASS_UNKNOWN"]
    if (mass > vehicle.max_pickup_payload_kg
            or vehicle.dry_mass_kg + mass > vehicle.max_takeoff_mass_kg):
        issues.append("PAYLOAD_EXCEEDS_VEHICLE_LIMIT")
    if loaded and envelope is None and not measurement_only:
        issues.append("LOADED_MOTION_DYNAMICS_UNQUALIFIED")
    if measurement_only and envelope is None:
        # 首批数据可以在隔离仿真中低速采集；此分支不产生任何合格包络或模型准入。
        dynamics = payload.get("dynamics")
        if (not isinstance(dynamics, dict) or dynamics.get("ready") is not True
                or dynamics.get("actuator_normalization_ready") is not True
                or not _finite(dynamics.get("actuator_max_abs"))
                or not 0 <= dynamics["actuator_max_abs"] < 1):
            issues.append("MEASUREMENT_DYNAMICS_TELEMETRY_UNREADY")
    if envelope is not None:
        total = vehicle.dry_mass_kg + mass
        if envelope.vehicle_sha256 != sha256_json(vehicle) or envelope.domain != domain:
            issues.append("FLIGHT_DYNAMICS_BINDING_MISMATCH")
        if not envelope.minimum_total_mass_kg <= total <= envelope.maximum_total_mass_kg:
            issues.append("FLIGHT_DYNAMICS_MASS_OUTSIDE_QUALIFICATION")
        dynamics = payload.get("dynamics")
        if not isinstance(dynamics, dict) or dynamics.get("ready") is not True:
            issues.append("FLIGHT_DYNAMICS_TELEMETRY_UNREADY")
        else:
            actuator = dynamics.get("actuator_max_abs")
            if (not _finite(actuator) or not 0 <= actuator <= envelope.maximum_actuator_fraction
                    or dynamics.get("actuator_normalization_ready") is not True):
                issues.append("FLIGHT_DYNAMICS_ACTUATOR_OUTSIDE_QUALIFICATION")
            voltage, ages = dynamics.get("voltage_v"), dynamics.get("source_sample_age_seconds", {})
            battery_age = ages.get("battery") if isinstance(ages, dict) else None
            if (not _finite(voltage) or voltage < envelope.minimum_battery_voltage_v
                    or not _finite(battery_age) or not 0 <= battery_age <= 1):
                issues.append("FLIGHT_DYNAMICS_BATTERY_OUTSIDE_QUALIFICATION")
            for axis in ("roll_deg", "pitch_deg"):
                angle = dynamics.get(axis)
                if not _finite(angle) or abs(angle) > envelope.maximum_tilt_deg:
                    issues.append("FLIGHT_DYNAMICS_ATTITUDE_OUTSIDE_QUALIFICATION")
                    break
    return issues


class VerticalMotionGuard:
    """一个控制周期的固定背景；检查模型速度而非代替模型生成坐标或确定安全。"""

    # 功能：
    #   固定本周期的几何、负载和顶部过渡边界，并生成可绑定到观测的摘要。
    # 输入：
    #   vehicle、payload、primitives：当前机型、同周期负载和保守局部几何。
    #   position、source_seconds、transition：碰撞中心、源时钟及运行级迟滞器。
    #   clearance、coverage_check：净空米数与对整条扫掠路径的证据检查函数。
    #   envelope、domain：测量得到的动力学包络及适用环境；coverage_identity：地图观测来源版本。
    #   measurement_only：仅限上游已审定的仿真采集，状态写入每次观测背景。
    # 输出：
    #   None：建立只用于当前控制周期的保护器。
    def __init__(self, *, vehicle, payload, primitives, position, source_seconds, transition,
                 clearance, coverage_check: Callable, envelope=None, domain="simulation",
                 coverage_identity=None, measurement_only=False):
        self.vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
        self.payload = deepcopy(payload)
        self.envelope = (FlightDynamicsEnvelope.model_validate(
            envelope.model_dump(mode="python"), strict=True) if envelope is not None else None)
        if domain not in {"simulation", "hardware"} or not callable(coverage_check):
            raise ValueError("VERTICAL_GUARD_BINDING_INVALID")
        self.bounds = [primitive_bounds(p) for p in primitives]
        self.coverage_check = coverage_check
        self.clearance = clearance
        self.measurement_only = measurement_only
        self.radius = max(vehicle.body_radius_m,
                          self.envelope.collision_radius_m if self.envelope else 0.)
        self.height = max(vehicle.body_height_m,
                          self.envelope.collision_height_m if self.envelope else 0.)
        self.interval = height_interval(position, self.bounds, self.radius,
                                        self.height / 2, clearance)
        self.ceiling = transition.update(self.interval.maximum_center_z_m, source_seconds)
        self.issues = payload_motion_issues(self.vehicle, self.payload, self.envelope, domain,
                                            measurement_only)
        self.context = {
            "schema": "dronedream.vertical-motion-context.v1",
            "coordinate_frame": "world-enu-collision-center",
            "source_seconds": source_seconds,
            "position_m": list(position),
            "vehicle_sha256": sha256_json(self.vehicle),
            "geometry_sha256": sha256_json(primitives),
            "coverage_sha256": sha256_json(coverage_identity),
            "payload_sha256": sha256_json(self.payload),
            "dynamics_sha256": sha256_json(self.envelope) if self.envelope else None,
            "measurement_only": measurement_only,
            "minimum_center_z_m": self.interval.minimum_center_z_m,
            "maximum_center_z_m": self.ceiling,
            "floor_z_m": self.interval.floor_z_m,
            "ceiling_z_m": self.interval.ceiling_z_m,
            "ground_height_m": (position[2] - self.interval.floor_z_m
                                if self.interval.floor_z_m is not None else None),
            "coverage_policy": "qualified-static-corridor-or-observed-whole-body",
            "collision_radius_m": self.radius,
            "collision_height_m": self.height,
            "issue_codes": list(self.issues),
        }
        self.sha256 = sha256_json(self.context)

    # 功能：
    #   用机体整个停止扫掠盒对比静态障碍及同期移动目标的扫掠盒，宁可拒绝而不漏过中间碰撞。
    # 输入：
    #   start、stop：停止扫掠两端；seconds：反应加制动的持续秒数；request：当前障碍与机体状态。
    #   extra_margin：延迟期间可能出现的侧向漂移米数。
    # 输出：
    #   conflict：是否与已知障碍扫掠范围重叠。
    def stopping_conflict(self, start, stop, seconds, request, extra_margin) -> bool:
        extent = (request.vehicle_radius_m, request.vehicle_radius_m, request.vehicle_height_m / 2)
        padding = self.clearance + extra_margin
        low = tuple(min(start[i], stop[i]) - extent[i] - padding for i in range(3))
        high = tuple(max(start[i], stop[i]) + extent[i] + padding for i in range(3))
        bounds = list(self.bounds)
        for obstacle in request.dynamic_obstacles:
            center = (obstacle.position_m.x, obstacle.position_m.y, obstacle.position_m.z)
            velocity = (obstacle.velocity_mps.x, obstacle.velocity_mps.y, obstacle.velocity_mps.z)
            first = tuple(center[i] + velocity[i] * obstacle.age_seconds for i in range(3))
            last = tuple(first[i] + velocity[i] * seconds for i in range(3))
            uncertainty = ((1. - obstacle.confidence) * .5
                           + obstacle.age_seconds * math.hypot(*velocity))
            size = (obstacle.radius_m, obstacle.radius_m, obstacle.height_m / 2)
            bounds.append((tuple(min(first[i], last[i]) - size[i] - uncertainty for i in range(3)),
                           tuple(max(first[i], last[i]) + size[i] + uncertainty for i in range(3))))
        conflict = any(all(low[i] <= other_high[i] and high[i] >= other_low[i] for i in range(3))
                       for other_low, other_high in bounds)
        return conflict

    # 功能：
    #   将经过测量的指令加速度上限传入预测器；没有包络时不虚构额外物理能力。
    # 输入：
    #   maximum：原始载具／任务允许的加速度上限。
    # 输出：
    #   limit：不超过任何已知上限的加速度。
    def acceleration_limit(self, maximum: float) -> float:
        limit = (min(maximum, self.envelope.maximum_command_acceleration_mps2)
                 if self.envelope is not None else maximum)
        if self.measurement_only:
            limit = min(limit, .2)
        return limit

    # 功能：
    #   1. 检查整段运动的上下净空与观测覆盖，不能靠一个向上测距点证明全机身可通过。
    #   2. 将实测反应延迟和制动下界换算成额外可停止路径，低能力负载需要更大余量。
    #   3. 返回拒绝原因，不修改模型意图；调用者必须对获选速度再次验收。
    # 输入：
    #   request：当前预测请求；velocity：实际候选 ENU 速度；path：从当前位置起的完整点列。
    # 输出：
    #   issues：该候选的约束违反列表，空列表仅表示通过本层检查。
    def check(self, request: LocalPlannerRequest, velocity, path) -> list[str]:
        if self.issues:
            return list(self.issues)
        points = list(path)
        if not points or len(points) > 1001:
            raise ValueError("VERTICAL_GUARD_PATH_INVALID")
        if self.measurement_only and math.hypot(*velocity) > .2 + 1e-9:
            return ["MEASUREMENT_SPEED_LIMIT"]
        envelope = self.envelope
        if envelope is not None:
            if (velocity[2] > envelope.maximum_climb_mps
                    or -velocity[2] > envelope.maximum_descent_mps
                    or math.hypot(*velocity[:2]) > envelope.maximum_horizontal_mps):
                return ["FLIGHT_DYNAMICS_SPEED_OUTSIDE_QUALIFICATION"]
            # 分别保留当前惯性与候选速度的停止扫掠，不能用减速后的下一拍速度掩盖当前动量。
            current = request.current_velocity_mps
            for speed in (velocity, (current.x, current.y, current.z)):
                magnitude = math.hypot(*speed)
                delay = (envelope.response_delay_upper_bound_seconds
                         + request.perception_stream_age_seconds)
                # 保守计入延迟期间沿行进方向继续加速；加速度上限不是制动下界。
                acceleration = request.max_acceleration_mps2
                after_delay = magnitude + acceleration * delay
                distance = (magnitude * delay + .5 * acceleration * delay**2
                            + after_delay**2 / (2 * envelope.braking_deceleration_lower_bound_mps2))
                if magnitude > 1e-9:
                    stop = tuple(points[0][i] + speed[i] / magnitude * distance for i in range(3))
                    seconds = delay + after_delay / envelope.braking_deceleration_lower_bound_mps2
                    if seconds > 15.:
                        return ["BRAKING_PREDICTION_HORIZON_EXCEEDED"]
                    drift = .5 * acceleration * delay**2
                    if self.stopping_conflict(points[0], stop, seconds, request, drift):
                        return ["BRAKING_SWEPT_SPACE_CONFLICT"]
                    if not self.coverage_check([points[0], stop], self.clearance + drift):
                        return ["BRAKING_SWEPT_SPACE_UNOBSERVED_OR_BLOCKED"]
                    # 顶面迟滞同样约束停止过程中仍会爬升的距离。
                    if self.ceiling is not None and stop[2] > self.ceiling + 1e-9:
                        return ["VERTICAL_BRAKING_HEADROOM_INSUFFICIENT"]
        if self.ceiling is not None and max(p[2] for p in points) > self.ceiling + 1e-9:
            return ["CEILING_OR_EXIT_TRANSITION_LIMIT"]
        if not self.coverage_check(points, self.clearance):
            return ["SWEPT_SPACE_UNOBSERVED_OR_BLOCKED"]
        return []
