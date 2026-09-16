"""Canonical, identity-bound sensor contracts for the local autonomy runtime.

The metric depth mount predates the multimodal runtime and remains the source
of truth for range-to-body calibration.  The additional contracts in this
module do not turn sensor readings into learned authority.  They provide the
shared timing, identity, unit, calibration, covariance, and freshness envelope
that deterministic fusion and local experts must agree on before a sample can
influence motion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from threading import RLock
from typing import Literal

from pydantic import Field, model_validator

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    QuaternionWxyz,
    StrictModel,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json

from .depth_projection import DepthProjectionCalibration
from .quaternion_geometry import rotate_vector as _rotate_vector
from .rgb_input_quality import RGB_QUALITY_SEMANTICS

Sha256 = str
SensorModality = Literal[
    "rgb-camera",
    "depth-camera",
    "lidar",
    "radar",
    "rangefinder",
    "imu",
    "magnetometer",
    "barometer",
    "gnss",
    "odometry",
    "optical-flow",
    "state-estimate",
    "powertrain",
    "payload",
]
SensorHealthState = Literal["healthy", "degraded", "unavailable"]


class RuntimeSensorContract(StrictModel):
    """Immutable admission rules for one runtime sensor stream."""

    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=120)
    vehicle_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=160)
    modality: SensorModality
    coordinate_frame: str = Field(min_length=1, max_length=160)
    unit: str = Field(min_length=1, max_length=80)
    calibration_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    intrinsics_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    extrinsics_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    maximum_sample_age_seconds: float = Field(gt=0.0, le=30.0)
    maximum_transport_latency_seconds: float = Field(gt=0.0, le=10.0)
    minimum_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    required_for_motion: bool = False


class RuntimeSensorEnvelope(StrictModel):
    """Content-bound metadata for one sensor sample kept outside raw payloads."""

    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=120)
    vehicle_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=160)
    modality: SensorModality
    sequence: int = Field(ge=1)
    sample_monotonic_seconds: float = Field(ge=0.0)
    received_monotonic_seconds: float = Field(ge=0.0)
    coordinate_frame: str = Field(min_length=1, max_length=160)
    unit: str = Field(min_length=1, max_length=80)
    calibration_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    intrinsics_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    extrinsics_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    covariance_diagonal: list[float] = Field(default_factory=list, max_length=36)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    payload_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: Literal["raw", "derived", "model-feature"] = "raw"

    # 功能：
    #   检查协方差对角项的有限性与非负性；跨时钟关系由注册表结合允许偏差核对。
    # 输入：
    #   self：完成字段解析的传感器元数据。
    # 输出：
    #   self：协方差检查通过的元数据。
    @model_validator(mode="after")
    def validate_timing_and_covariance(self) -> RuntimeSensorEnvelope:
        if any(value < 0.0 or not math.isfinite(value) for value in self.covariance_diagonal):
            raise ValueError("sensor covariance diagonal must be finite and non-negative")
        return self


class RuntimeSensorStatus(StrictModel):
    """Current fail-closed status of one admitted sensor stream."""

    sensor_id: str
    modality: SensorModality
    required_for_motion: bool
    latest_sequence: int = Field(ge=0)
    sample_age_seconds: float | None = Field(default=None, ge=0.0)
    transport_latency_seconds: float | None = Field(default=None, ge=0.0)
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    health: SensorHealthState
    issue_codes: list[str] = Field(default_factory=list, max_length=16)
    payload_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RuntimeMultimodalSensorSnapshot(StrictModel):
    """Auditable latest-value view consumed by fusion and Harness routing."""

    captured_at_monotonic_seconds: float = Field(ge=0.0)
    contract_set_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    ready_for_motion: bool
    statuses: list[RuntimeSensorStatus] = Field(max_length=64)
    active_sensor_ids: list[str] = Field(max_length=64)
    issue_codes: list[str] = Field(default_factory=list, max_length=64)


class ForwardCameraMotionAlignment(StrictModel):
    """Auditable relation between the calibrated camera axis and horizontal motion."""

    schema_version: Literal["dronedream.forward-camera-motion-alignment.v1"] = (
        "dronedream.forward-camera-motion-alignment.v1"
    )
    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=120)
    camera_forward_world_enu: Vector3
    horizontal_speed_mps: float = Field(ge=0.0)
    minimum_alignment_speed_mps: float = Field(gt=0.0, le=10.0)
    alignment_required: bool
    alignment_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    minimum_alignment_cosine: float = Field(ge=-1.0, le=1.0)
    aligned_for_forward_navigation: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=8)


# 功能：
#   1. 将校准相机前向轴转到世界坐标，与水平运动方向比较；悬停及纯升降不要求水平对齐。
#   2. 仅判断视野朝向关系，不证明图像新鲜、障碍可见或当前动作安全。
# 输入：
#   body_orientation_world_from_body：机体到世界坐标的姿态。
#   body_velocity_world_enu_mps：世界 ENU 速度。
#   mount：可选相机安装校准，缺省使用当前 OAK-D Lite 安装。
#   minimum_alignment_speed_mps：需要检查水平运动朝向的最低速度。
#   minimum_alignment_cosine：允许的最小方向余弦。
# 输出：
#   result：相机世界朝向、速度及对齐判定。
def forward_camera_motion_alignment(
    *,
    body_orientation_world_from_body: QuaternionWxyz,
    body_velocity_world_enu_mps: Vector3,
    mount: CalibratedRangeSensorMount | None = None,
    minimum_alignment_speed_mps: float = 0.03,
    minimum_alignment_cosine: float = math.cos(math.radians(45.0)),
) -> ForwardCameraMotionAlignment:
    if (type(minimum_alignment_speed_mps) not in (int, float)
            or not 0.01 <= minimum_alignment_speed_mps <= 10.0):
        raise ValueError("camera alignment speed threshold is outside the safe range")
    if (type(minimum_alignment_cosine) not in (int, float)
            or not -1.0 <= minimum_alignment_cosine <= 1.0):
        raise ValueError("camera alignment cosine threshold is invalid")
    resolved_mount = mount or oakd_lite_depth_sensor_contract()
    forward_body = _rotate_vector(
        resolved_mount.orientation_body_from_sensor,
        (1.0, 0.0, 0.0),
    )
    forward_world = _rotate_vector(body_orientation_world_from_body, forward_body)
    horizontal_forward_norm = math.hypot(forward_world[0], forward_world[1])
    horizontal_speed = math.hypot(
        body_velocity_world_enu_mps.x,
        body_velocity_world_enu_mps.y,
    )
    alignment_required = horizontal_speed >= minimum_alignment_speed_mps
    alignment_cosine = None
    aligned = True
    issues: list[str] = []
    if alignment_required:
        if horizontal_forward_norm <= 1e-9:
            aligned = False
            issues.append("FORWARD_CAMERA_AXIS_VERTICAL")
        else:
            alignment_cosine = (
                forward_world[0] * body_velocity_world_enu_mps.x
                + forward_world[1] * body_velocity_world_enu_mps.y
            ) / (horizontal_forward_norm * horizontal_speed)
            alignment_cosine = max(-1.0, min(1.0, alignment_cosine))
            aligned = alignment_cosine >= minimum_alignment_cosine
            if not aligned:
                issues.append("FORWARD_CAMERA_NOT_MOTION_ALIGNED")
    result = ForwardCameraMotionAlignment(
        sensor_id="oakd-lite-forward-rgb",
        camera_forward_world_enu=Vector3(
            x=forward_world[0],
            y=forward_world[1],
            z=forward_world[2],
        ),
        horizontal_speed_mps=horizontal_speed,
        minimum_alignment_speed_mps=minimum_alignment_speed_mps,
        alignment_required=alignment_required,
        alignment_cosine=alignment_cosine,
        minimum_alignment_cosine=minimum_alignment_cosine,
        aligned_for_forward_navigation=aligned,
        issue_codes=issues,
    )
    return result


# 功能：
#   决定前视图像是否可提供信息；连续控制允许侧滑等方向，路径观察模式要求视野与运动对齐。
# 输入：
#   alignment：相机和运动方向的对齐判定。
#   continuous_control：是否为连续控制模式，必须是布尔值。
# 输出：
#   usable：图像可作为信息输入时为 True，不代表获得运动授权。
def forward_rgb_can_inform_control(
    alignment: ForwardCameraMotionAlignment, *, continuous_control: bool,
) -> bool:
    if type(continuous_control) is not bool:
        raise ValueError("continuous control mode must be boolean")
    usable = continuous_control or alignment.aligned_for_forward_navigation
    return usable


# 功能：
#   在注册表所有读写入口统一拒绝非数值、非有限或负时钟，不将布尔值当秒数。
# 输入：
#   value：调用方提供的单调时刻。
# 输出：
#   None：不返回业务数据。
def _validate_registry_clock(value: float) -> None:
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("sensor registry time must be finite and non-negative")
    except OverflowError as error:
        raise ValueError("sensor registry time must be finite and non-negative") from error


class RuntimeSensorRegistry:
    """Thread-safe, depth-one mailbox for heterogeneous sensor streams.

    New samples replace older samples only after every contract check passes.
    Slow consumers therefore observe the freshest accepted sample instead of
    building an unbounded queue whose results would already be unsafe to use.
    """

    # 功能：
    #   创建受锁保护的最新值注册表，冻结初始契约并限制允许的跨来源时钟偏差。
    # 输入：
    #   self：当前注册表实例。
    #   contracts：需要登记的初始传感器契约。
    #   maximum_clock_skew_seconds：允许的时钟小幅偏差，单位秒。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, contracts: Iterable[RuntimeSensorContract] = (), *,
                 maximum_clock_skew_seconds: float = 0.01) -> None:
        if (type(maximum_clock_skew_seconds) not in (int, float)
                or not 0 <= maximum_clock_skew_seconds <= 1):
            raise ValueError("sensor clock skew allowance is outside the safe range")
        self.maximum_clock_skew_seconds = maximum_clock_skew_seconds
        self._contracts: dict[str, RuntimeSensorContract] = {}
        self._latest: dict[str, RuntimeSensorEnvelope] = {}
        self._lock = RLock()
        for contract in contracts:
            self.register_contract(contract)

    # 功能：
    #   在同一把锁下按传感器标识排序并计算完整契约集合摘要。
    # 输入：
    #   self：当前注册表实例。
    # 输出：
    #   digest：顺序无关、内容绑定的契约集合摘要。
    @property
    def contract_set_sha256(self) -> str:
        with self._lock:
            digest = sha256_json(
                [
                    self._contracts[sensor_id].model_dump(mode="json")
                    for sensor_id in sorted(self._contracts)
                ]
            )
            return digest

    # 功能：
    #   重新校验并冻结单个契约，拒绝同名内容变化或超过快照容量的新标识。
    # 输入：
    #   self：当前注册表实例。
    #   contract：待登记的传感器契约。
    # 输出：
    #   None：不返回业务数据。
    def register_contract(self, contract: RuntimeSensorContract) -> None:
        contract = RuntimeSensorContract.model_validate(contract.model_dump(mode="python"))
        with self._lock:
            existing = self._contracts.get(contract.sensor_id)
            if existing is not None and existing != contract:
                raise ValueError("SENSOR_CONTRACT_IDENTITY_CHANGED")
            if existing is None and len(self._contracts) >= 64:
                raise ValueError("SENSOR_CONTRACT_LIMIT_EXCEEDED")
            self._contracts[contract.sensor_id] = contract

    # 功能：
    #   返回指定传感器的独立校准契约副本，未知标识不能继承其它传感器配置。
    # 输入：
    #   self：当前注册表实例。
    #   sensor_id：传感器标识。
    # 输出：
    #   result：独立的传感器契约副本。
    def contract(self, sensor_id: str) -> RuntimeSensorContract:
        with self._lock:
            try:
                result = self._contracts[sensor_id].model_copy(deep=True)
                return result
            except KeyError as error:
                raise ValueError("SENSOR_CONTRACT_NOT_REGISTERED") from error

    # 功能：
    #   获取最新已接收元数据的副本；此读取不证明它在当前时刻仍有效。
    # 输入：
    #   self：当前注册表实例。
    #   sensor_id：传感器标识。
    # 输出：
    #   result：最新元数据副本，尚未接收时为 None。
    def latest(self, sensor_id: str) -> RuntimeSensorEnvelope | None:
        with self._lock:
            value = self._latest.get(sensor_id)
            result = value.model_copy(deep=True) if value is not None else None
            return result

    # 功能：
    #   1. 核对设备、坐标、单位、校准、时钟、传输延迟及顺序，只在全部检查后替换最新样本。
    #   2. 质量不足的合法样本仍留作诊断；是否满足运动需求由状态读取时的契约判定。
    # 输入：
    #   self：当前注册表实例。
    #   envelope：待接收的内容绑定元数据。
    #   now_monotonic_seconds：接收校验时的主机单调时刻。
    # 输出：
    #   None：不返回业务数据。
    def ingest(self, envelope: RuntimeSensorEnvelope, *, now_monotonic_seconds: float) -> None:
        _validate_registry_clock(now_monotonic_seconds)
        envelope = RuntimeSensorEnvelope.model_validate(envelope.model_dump(mode="python"))
        with self._lock:
            contract = self._contracts.get(envelope.sensor_id)
            if contract is None:
                raise ValueError("SENSOR_CONTRACT_NOT_REGISTERED")
            identity_fields = (
                "vehicle_id",
                "modality",
                "coordinate_frame",
                "unit",
                "calibration_sha256",
                "intrinsics_sha256",
                "extrinsics_sha256",
            )
            if any(getattr(envelope, name) != getattr(contract, name) for name in identity_fields):
                raise ValueError("SENSOR_SAMPLE_CONTRACT_MISMATCH")
            if envelope.received_monotonic_seconds > (
                now_monotonic_seconds + self.maximum_clock_skew_seconds
            ):
                raise ValueError("SENSOR_RECEIVE_TIMESTAMP_IN_FUTURE")
            transport_latency = (
                envelope.received_monotonic_seconds - envelope.sample_monotonic_seconds
            )
            if transport_latency < -self.maximum_clock_skew_seconds:
                raise ValueError("SENSOR_SAMPLE_TIMESTAMP_IN_FUTURE")
            if transport_latency > contract.maximum_transport_latency_seconds:
                raise ValueError("SENSOR_TRANSPORT_LATENCY_EXCEEDED")
            sample_age = now_monotonic_seconds - envelope.sample_monotonic_seconds
            if sample_age < -self.maximum_clock_skew_seconds:
                raise ValueError("SENSOR_SAMPLE_TIMESTAMP_IN_FUTURE")
            if sample_age > contract.maximum_sample_age_seconds:
                raise ValueError("SENSOR_SAMPLE_STALE_AT_INGEST")
            previous = self._latest.get(envelope.sensor_id)
            if previous is not None and envelope.sequence <= previous.sequence:
                raise ValueError("SENSOR_SEQUENCE_NOT_MONOTONIC")
            if previous is not None and envelope.sample_monotonic_seconds < (
                previous.sample_monotonic_seconds - self.maximum_clock_skew_seconds
            ):
                raise ValueError("SENSOR_SAMPLE_TIMESTAMP_REGRESSION")
            if previous is not None and envelope.received_monotonic_seconds < (
                previous.received_monotonic_seconds - self.maximum_clock_skew_seconds
            ):
                raise ValueError("SENSOR_RECEIVE_TIMESTAMP_REGRESSION")
            self._latest[envelope.sensor_id] = envelope.model_copy(deep=True)

    # 功能：
    #   结合当前时刻重新判断来源状态，明确区分未接收、过期、低质量及健康样本。
    # 输入：
    #   self：当前注册表实例。
    #   sensor_id：已登记的传感器标识。
    #   now_monotonic_seconds：本次状态查询的单调时刻。
    # 输出：
    #   result：包含全部该传感器问题代码的状态。
    def status(self, sensor_id: str, *, now_monotonic_seconds: float) -> RuntimeSensorStatus:
        _validate_registry_clock(now_monotonic_seconds)
        with self._lock:
            contract = self._contracts.get(sensor_id)
            if contract is None:
                raise ValueError("SENSOR_CONTRACT_NOT_REGISTERED")
            envelope = self._latest.get(sensor_id)
            if envelope is None:
                result = RuntimeSensorStatus(
                    sensor_id=sensor_id,
                    modality=contract.modality,
                    required_for_motion=contract.required_for_motion,
                    latest_sequence=0,
                    health="unavailable",
                    issue_codes=["SENSOR_SAMPLE_NOT_RECEIVED"],
                )
                return result
            age = max(0.0, now_monotonic_seconds - envelope.sample_monotonic_seconds)
            transport_latency = max(
                0.0,
                envelope.received_monotonic_seconds - envelope.sample_monotonic_seconds,
            )
            issues: list[str] = []
            if max(envelope.sample_monotonic_seconds, envelope.received_monotonic_seconds) > (
                now_monotonic_seconds + self.maximum_clock_skew_seconds
            ):
                issues.append("SENSOR_TIMESTAMP_IN_FUTURE")
            if age > contract.maximum_sample_age_seconds:
                issues.append("SENSOR_SAMPLE_STALE")
            if transport_latency > contract.maximum_transport_latency_seconds:
                issues.append("SENSOR_TRANSPORT_LATENCY_EXCEEDED")
            if envelope.quality < contract.minimum_quality:
                issues.append("SENSOR_QUALITY_BELOW_CONTRACT")
            if envelope.coverage < contract.minimum_coverage:
                issues.append("SENSOR_COVERAGE_BELOW_CONTRACT")
            health: SensorHealthState = "healthy" if not issues else "degraded"
            result = RuntimeSensorStatus(
                sensor_id=sensor_id,
                modality=contract.modality,
                required_for_motion=contract.required_for_motion,
                latest_sequence=envelope.sequence,
                sample_age_seconds=age,
                transport_latency_seconds=transport_latency,
                quality=envelope.quality,
                coverage=envelope.coverage,
                health=health,
                issue_codes=issues,
                payload_sha256=envelope.payload_sha256,
            )
            return result

    # 功能：
    #   1. 在同一锁域读取契约摘要与各来源状态，必需来源未就绪时拒绝报告可用于运动。
    #   2. 顶层摘要超量时显式标记截断，保留所有传感器完整诊断，不因问题过多失去快照。
    # 输入：
    #   self：当前注册表实例。
    #   now_monotonic_seconds：所有来源统一使用的查询时刻。
    # 输出：
    #   result：元数据级最新值快照，不替代模型、碰撞及实际飞控的运动授权。
    def snapshot(self, *, now_monotonic_seconds: float) -> RuntimeMultimodalSensorSnapshot:
        _validate_registry_clock(now_monotonic_seconds)
        # status 和摘要属性会再次获取锁，因此必须保留可重入锁以保证整组读取一致。
        with self._lock:
            statuses = [
                self.status(sensor_id, now_monotonic_seconds=now_monotonic_seconds)
                for sensor_id in sorted(self._contracts)
            ]
            contract_set_sha256 = self.contract_set_sha256
        required_unhealthy = [
            status
            for status in statuses
            if status.required_for_motion and status.health != "healthy"
        ]
        issue_codes = sorted(
            {
                f"{status.sensor_id}:{issue}"
                for status in statuses
                for issue in status.issue_codes
            }
        )
        if not any(status.required_for_motion for status in statuses):
            issue_codes.append("SENSOR_REQUIRED_CONTRACTS_MISSING")
        if len(issue_codes) > 64:
            issue_codes = [*issue_codes[:63], "SENSOR_ISSUE_CODES_TRUNCATED"]
        result = RuntimeMultimodalSensorSnapshot(
            captured_at_monotonic_seconds=now_monotonic_seconds,
            contract_set_sha256=contract_set_sha256,
            ready_for_motion=any(status.required_for_motion for status in statuses)
            and not required_unhealthy,
            statuses=statuses,
            active_sensor_ids=[
                status.sensor_id for status in statuses if status.latest_sequence > 0
            ],
            issue_codes=issue_codes,
        )
        return result


# 功能：
#   返回当前内置 OAK-D Lite 的米制安装校准，每次创建独立对象，不借用旧任务缓存。
# 输入：
#   无。
# 输出：
#   mount：机体安装偏移、单位姿态及校准量程。
def oakd_lite_depth_sensor_contract() -> CalibratedRangeSensorMount:
    mount = CalibratedRangeSensorMount(
        sensor_id="oakd-lite-depth",
        translation_body_m=Vector3(x=0.13233, y=0.0, z=0.03278),
        orientation_body_from_sensor=QuaternionWxyz(
            w=1.0,
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        minimum_range_m=0.2,
        maximum_range_m=19.1,
    )
    return mount


# 功能：
#   将当前深度投影内参与米制安装外参绑定为单一来源契约，拒绝投影和安装量程不一致。
# 输入：
#   vehicle_id：所属飞行器标识。
#   calibration：已验证的深度投影校准。
#   required_for_motion：该来源是否为运动信息的必需项。
#   maximum_sample_age_seconds：允许的观测年龄上限。
# 输出：
#   contract：含内外参摘要、量纲、时限及最低质量的深度来源契约。
def oakd_lite_depth_runtime_contract(
    *,
    vehicle_id: str,
    calibration: DepthProjectionCalibration,
    required_for_motion: bool = True,
    maximum_sample_age_seconds: float = 0.35,
) -> RuntimeSensorContract:
    if type(required_for_motion) is not bool:
        raise ValueError("required for motion must be boolean")
    mount = oakd_lite_depth_sensor_contract()
    if not isinstance(calibration, DepthProjectionCalibration):
        raise ValueError("DEPTH_PROJECTION_CALIBRATION_REQUIRED")
    if (calibration.minimum_depth_m != mount.minimum_range_m
            or calibration.maximum_depth_m != mount.maximum_range_m):
        raise ValueError("DEPTH_PROJECTION_MOUNT_RANGE_MISMATCH")
    projection = calibration.identity_payload
    extrinsics = {
        "translation_body_m": mount.translation_body_m,
        "orientation_body_from_sensor": mount.orientation_body_from_sensor,
    }
    contract = RuntimeSensorContract(
        sensor_id=mount.sensor_id,
        vehicle_id=vehicle_id,
        modality="depth-camera",
        coordinate_frame="world-enu",
        unit="meter-ray",
        calibration_sha256=sha256_json(
            {"mount": mount.model_dump(mode="json"), "projection": projection}
        ),
        intrinsics_sha256=sha256_json(projection),
        extrinsics_sha256=sha256_json(extrinsics),
        maximum_sample_age_seconds=maximum_sample_age_seconds,
        maximum_transport_latency_seconds=0.25,
        minimum_quality=0.05,
        minimum_coverage=0.05,
        required_for_motion=required_for_motion,
    )
    return contract


# 功能：
#   按实际首帧整数尺寸和视场角构造当前前视 RGB 来源契约，绑定质量语义与安装外参。
# 输入：
#   vehicle_id：所属飞行器标识。
#   width：图像宽度，单位像素。
#   height：图像高度，单位像素。
#   horizontal_fov_rad：水平视场角，单位弧度。
#   required_for_motion：该来源是否为必需项。
#   maximum_sample_age_seconds：允许的观测年龄上限。
# 输出：
#   contract：具有精确尺寸、校准摘要、单位及质量门限的 RGB 来源契约。
def oakd_lite_forward_rgb_runtime_contract(
    *,
    vehicle_id: str,
    width: int,
    height: int,
    horizontal_fov_rad: float = 1.204,
    required_for_motion: bool = True,
    maximum_sample_age_seconds: float = 0.5,
) -> RuntimeSensorContract:
    if (type(width) is not int or type(height) is not int
            or width < 32 or width > 8_192 or height < 32 or height > 8_192):
        raise ValueError("forward RGB dimensions are outside the supported range")
    if (type(horizontal_fov_rad) not in (int, float)
            or not 0.1 <= horizontal_fov_rad < math.pi):
        raise ValueError("forward RGB field of view is invalid")
    if type(required_for_motion) is not bool:
        raise ValueError("required for motion must be boolean")
    mount = oakd_lite_depth_sensor_contract()
    intrinsics = {
        "sensor_model": "IMX214",
        "width": width,
        "height": height,
        "horizontal_fov_rad": horizontal_fov_rad,
        "pixel_layout": "runtime-message-rgb",
        "quality_semantics": RGB_QUALITY_SEMANTICS,
    }
    extrinsics = {
        "translation_body_m": mount.translation_body_m,
        "orientation_body_from_sensor": mount.orientation_body_from_sensor,
    }
    contract = RuntimeSensorContract(
        sensor_id="oakd-lite-forward-rgb",
        vehicle_id=vehicle_id,
        modality="rgb-camera",
        coordinate_frame="camera-optical",
        unit="uint8-rgb",
        calibration_sha256=sha256_json(
            {"intrinsics": intrinsics, "extrinsics": extrinsics}
        ),
        intrinsics_sha256=sha256_json(intrinsics),
        extrinsics_sha256=sha256_json(extrinsics),
        maximum_sample_age_seconds=maximum_sample_age_seconds,
        maximum_transport_latency_seconds=0.25,
        minimum_quality=0.01,
        minimum_coverage=1.0,
        required_for_motion=required_for_motion,
    )
    return contract
