"""Metric, time-bound feature encoders for local flight control.

These encoders deliberately operate on calibrated physical observations rather
than on UI summaries.  They are the deterministic front end for learned local
experts: each output is content-bound, mask-aware and explicit about age and
uncertainty.  A learned ONNX head may consume these tensors, but it may not
invent a missing sensor or turn a stale sample into an observed zero.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import islice
from typing import Literal, TypeVar

from pydantic import Field, model_validator

from .contracts import (
    BodyFrameControlIntent,
    CalibratedRangeSensorMount,
    DynamicObstacleObservation,
    NormalizedPilotControl,
    QuaternionWxyz,
    RawMetricRangeScan,
    StrictModel,
    Vector3,
)
from .control_feature_contract import (
    DYNAMIC_TARGET_FEATURE_COUNT,
    FLIGHT_STATE_FEATURE_COUNT,
    FLIGHT_STATE_HISTORY_LENGTH,
    FLIGHT_STATE_MAXIMUM_GAP_MS,
    GEOMETRY_FEATURE_COUNT,
    SENSOR_FEATURE_COUNT,
    SPATIAL_CELL_COUNT,
    encoder_contract_sha256,
    policy_feature_contract_sha256,
)
from .control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
from .hashing import sha256_json
from .pilot_control_mapping import PilotControlLimits, physical_pilot_request
from .quaternion_geometry import rotate_vector as _rotate
from .relative_motion import closest_center_distance_m, cylinder_contact_seconds

FeatureEncoderRole = Literal[
    "metric-geometry-encoder",
    "dynamic-target-encoder",
    "flight-state-encoder",
]
REALTIME_CONTROL_FEATURE_COUNT = SENSOR_FEATURE_COUNT
# Sensor encoders intentionally stay independent of mission intent.  The local
# policy receives an additional, egocentric control reference derived from the
# active coarse route and the synchronized flight-state quaternion.  This is
# the neural equivalent of telling a pilot where the next corridor lies; it is
# not an actuator command and it does not replace collision prediction.
POLICY_CONTROL_REFERENCE_FEATURE_COUNT = 10
POLICY_REALTIME_FEATURE_COUNT = (
    REALTIME_CONTROL_FEATURE_COUNT + POLICY_CONTROL_REFERENCE_FEATURE_COUNT
)
PILOT_CONTROL_PRECISION_SPEED_CAP_MPS = 0.2
PILOT_CONTROL_VERTICAL_SPEED_CAP_MPS = 0.75
PILOT_CONTROL_YAW_RATE_CAP_DPS = 20.0
_ENCODER_ROLE_ORDER: tuple[FeatureEncoderRole, ...] = (
    "metric-geometry-encoder",
    "dynamic-target-encoder",
    "flight-state-encoder",
)

_SECTOR_COUNT = 8
_SECTOR_WIDTH_RAD = 2.0 * math.pi / _SECTOR_COUNT
_ENCODER_WIDTHS = dict(
    zip(
        _ENCODER_ROLE_ORDER,
        (
            GEOMETRY_FEATURE_COUNT,
            DYNAMIC_TARGET_FEATURE_COUNT,
            FLIGHT_STATE_FEATURE_COUNT,
        ),
        strict=True,
    )
)
_BLOCKING_ENCODER_ISSUES = frozenset(
    {
        "GEOMETRY_ANGULAR_COVERAGE_LOW",
        "GEOMETRY_SAMPLE_COUNT_LOW",
        "GEOMETRY_RANGE_OUTSIDE_CALIBRATION",
        "GEOMETRY_SOURCE_COVERAGE_LOW",
        "FLIGHT_STATE_HISTORY_WARMING",
        "FLIGHT_STATE_SAMPLE_GAP",
        "FLIGHT_STATE_SOURCE_CHANGED",
        "FLIGHT_STATE_IMU_CLOCK_RESET",
        "FLIGHT_STATE_COVARIANCE_UNAVAILABLE",
        "FLIGHT_STATE_LOCALIZATION_UNCERTAIN",
        "FLIGHT_STATE_IMU_UNAVAILABLE",
        "DYNAMIC_TARGET_STALE",
    }
)
EncoderModel = TypeVar("EncoderModel", bound=StrictModel)


# 功能：
#   按训练和运行共用规则定义满杆物理幅度，不代替飞控的加速度、跃度与碰撞限制。
# 输入：
#   profile：巡航或精细控制场景。
#   route_speed_limit_mps：当前路线限速，必须是正的有限数且不超过控制契约的 20 m/s。
# 输出：
#   limits：水平速度、垂直速度和偏航角速度上限三元组。
def pilot_control_limits_for_profile(
    *, profile: Literal["cruise", "precision"], route_speed_limit_mps: float
) -> tuple[float, float, float]:
    if profile not in {"cruise", "precision"}:
        raise ValueError(f"unsupported navigation control profile: {profile}")
    route_speed_limit_mps = _positive_limit(route_speed_limit_mps, maximum=20.0, name="ROUTE_SPEED")
    horizontal_speed_mps = (
        min(route_speed_limit_mps, PILOT_CONTROL_PRECISION_SPEED_CAP_MPS)
        if profile == "precision"
        else route_speed_limit_mps
    )
    limits = (
        horizontal_speed_mps,
        min(horizontal_speed_mps, PILOT_CONTROL_VERTICAL_SPEED_CAP_MPS),
        PILOT_CONTROL_YAW_RATE_CAP_DPS,
    )
    return limits


# 功能：
#   在归一化之前拒绝非法物理幅度，不能把负上限用于反向缩放控制量。
# 输入：
#   value：原始数值。
#   maximum：该物理量在公共控制契约中的上界。
#   name：错误定位标识。
# 输出：
#   bounded：通过检查的浮点上限。
def _positive_limit(value: float, *, maximum: float, name: str) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(
                f"ENCODER_{name}_LIMIT_INVALID: physical limits must be finite and in range"
            )
    except OverflowError as error:
        raise ValueError(
            f"ENCODER_{name}_LIMIT_INVALID: physical limits must be finite and in range"
        ) from error
    bounded = float(value)
    return bounded


# 功能：
#   按当前模型严格重验独立副本，不接受构造后绕过验证的嵌套值；保持 Python 实时路径避免 JSON 中转。
# 输入：
#   value：应有明确模型类型的输入。
#   contract：预期的模型类。
# 输出：
#   snapshot：脱离调用方可变容器的严格快照。
def _strict_copy(value: EncoderModel, contract: type[EncoderModel]) -> EncoderModel:
    if not isinstance(value, contract):
        raise ValueError("ENCODER_INPUT_CONTRACT_INVALID")
    snapshot = contract.model_validate(value.model_dump(mode="python"), strict=True)
    return snapshot


# 功能：
#   有界消费调用方的流，输入过长或无限时及时拒绝，不先把整个生成器转成列表。
# 输入：
#   values：模型或角色迭代器。
#   maximum：允许条目数。
# 输出：
#   items：最多指定数量的列表，超出容量抛出错误。
def _bounded_items(values: Iterable, *, maximum: int) -> list:
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("ENCODER_COLLECTION_INVALID")
    try:
        items = list(islice(iter(values), maximum + 1))
    except TypeError as error:
        raise ValueError("ENCODER_COLLECTION_INVALID") from error
    if len(items) > maximum:
        raise ValueError("ENCODER_COLLECTION_LIMIT_EXCEEDED")
    return items


class RealtimeFeatureEncoding(StrictModel):
    """One fixed-width, auditable encoder result.

    ``valid_mask`` is kept separate from values so absence cannot be confused
    with a measured zero.  Features are normalized only by declared physical
    limits and remain finite, which makes the same contract usable by native
    code, ONNX Runtime and replay qualification.
    """

    schema_version: Literal["dronedream.realtime-feature-encoding.v1"] = (
        "dronedream.realtime-feature-encoding.v1"
    )
    encoder_role: FeatureEncoderRole
    source_ids: list[str] = Field(min_length=1, max_length=32)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observed_at_unix_ms: int = Field(ge=0)
    encoded_at_unix_ms: int = Field(ge=0)
    maximum_age_milliseconds: int = Field(gt=0, le=30_000)
    quality: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    features: list[float] = Field(min_length=1, max_length=512)
    valid_mask: list[float] = Field(min_length=1, max_length=512)
    inference_latency_ms: float = Field(ge=0.0, le=30_000.0)
    issue_codes: list[str] = Field(default_factory=list, max_length=24)

    # 功能：
    #   校验角色固定宽度、掩码与采集时序；张量形状不能替代模型包的特征语义契约。
    # 输入：
    #   self：构造中的单传感器编码。
    # 输出：
    #   self：结构一致的编码，不授予运动权限。
    @model_validator(mode="after")
    def validate_encoding(self) -> RealtimeFeatureEncoding:
        if len(self.features) != len(self.valid_mask):
            raise ValueError("feature values and validity mask must have equal width")
        if len(self.features) != _ENCODER_WIDTHS[self.encoder_role]:
            raise ValueError("feature width does not match its encoder role")
        if any(not math.isfinite(value) for value in self.features):
            raise ValueError("encoded features must be finite")
        if any(value not in {0.0, 1.0} for value in self.valid_mask):
            raise ValueError("feature validity mask must be binary")
        if self.encoded_at_unix_ms < self.observed_at_unix_ms:
            raise ValueError("feature encoding cannot predate its observation")
        return self

    # 功能：
    #   依据原始采集时间判断编码保留期，不用后续编码时间刷新旧观测寿命。
    # 输入：
    #   self：单角色编码。
    #   unix_ms：当前绝对毫秒时间。
    # 输出：
    #   fresh：时钟、编码结构和保留期均有效时为真。
    def fresh_at(self, unix_ms: int) -> bool:
        if type(unix_ms) is not int or unix_ms < 0:
            return False
        try:
            encoding = _strict_copy(self, RealtimeFeatureEncoding)
        except (ValueError, TypeError, OverflowError):
            return False
        fresh = (
            encoding.encoded_at_unix_ms <= unix_ms
            and unix_ms - encoding.observed_at_unix_ms <= encoding.maximum_age_milliseconds
        )
        return fresh


class RealtimeFeatureSnapshot(StrictModel):
    """Synchronized encoder set admitted to one local-control tick."""

    schema_version: Literal["dronedream.realtime-feature-snapshot.v1"] = (
        "dronedream.realtime-feature-snapshot.v1"
    )
    captured_at_unix_ms: int = Field(ge=0)
    required_roles: list[FeatureEncoderRole] = Field(
        default_factory=lambda: list(_ENCODER_ROLE_ORDER), min_length=1, max_length=3
    )
    encodings: list[RealtimeFeatureEncoding] = Field(min_length=1, max_length=16)
    ready_for_control: bool
    fused_features: list[float] = Field(min_length=1, max_length=2_048)
    fused_valid_mask: list[float] = Field(min_length=1, max_length=2_048)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_codes: list[str] = Field(default_factory=list, max_length=64)

    # 功能：
    #   重算角色顺序、融合张量、就绪判断与内容散列，禁止借用其他观测的成功标志。
    # 输入：
    #   self：构造中的融合快照。
    # 输出：
    #   self：与原始编码逐项一致的快照。
    @model_validator(mode="after")
    def validate_snapshot(self) -> RealtimeFeatureSnapshot:
        if len(self.fused_features) != len(self.fused_valid_mask):
            raise ValueError("fused values and validity mask must have equal width")
        roles = [encoding.encoder_role for encoding in self.encodings]
        if len(set(roles)) != len(roles):
            raise ValueError("feature snapshot cannot contain duplicate encoder roles")
        if roles != [role for role in _ENCODER_ROLE_ORDER if role in roles]:
            raise ValueError("feature snapshot roles must follow the canonical tensor order")
        if self.fused_features != [v for e in self.encodings for v in e.features]:
            raise ValueError("fused features differ from their source encodings")
        if self.fused_valid_mask != [v for e in self.encodings for v in e.valid_mask]:
            raise ValueError("fused mask differs from its source encodings")
        issues = _snapshot_readiness_issues(
            self.encodings, self.required_roles, self.captured_at_unix_ms
        )
        if self.issue_codes != issues or self.ready_for_control != (not issues):
            raise ValueError("feature snapshot readiness differs from sensor evidence")
        payload = _snapshot_hash_payload(
            self.encodings, self.required_roles, self.captured_at_unix_ms, issues
        )
        if self.snapshot_sha256 != sha256_json(payload):
            raise ValueError("feature snapshot content hash does not match")
        return self

    # 功能：
    #   重验融合散列并逐个检查必需传感器寿命，修改 ready 或 required_roles 不能绕过检查。
    # 输入：
    #   self：当前融合快照。
    #   unix_ms：检查时刻的绝对毫秒时间。
    # 输出：
    #   fresh：所有必需编码仍有效且内容未被改写时为真。
    def fresh_at(self, unix_ms: int) -> bool:
        if type(unix_ms) is not int or unix_ms < 0:
            return False
        try:
            snapshot = _strict_copy(self, RealtimeFeatureSnapshot)
        except (ValueError, TypeError, OverflowError):
            return False
        by_role = {encoding.encoder_role: encoding for encoding in snapshot.encodings}
        fresh = (
            snapshot.ready_for_control
            and unix_ms >= snapshot.captured_at_unix_ms
            and all(
                role in by_role and by_role[role].fresh_at(unix_ms)
                for role in snapshot.required_roles
            )
        )
        return fresh

    # 功能：
    #   组合三个显式特征契约，不根据向量宽度猜测模型输入含义。
    # 输入：
    #   self：当前融合快照。
    # 输出：
    #   digest：完整策略特征契约摘要，缺少对应身份时为 None。
    def policy_feature_contract_sha256(self) -> str | None:
        digest = policy_feature_contract_sha256(
            {
                encoding.encoder_role: encoding.feature_contract_sha256
                for encoding in self.encodings
                if encoding.feature_contract_sha256 is not None
            }
        )
        return digest

    # 功能：
    #   由最旧必需观测和旋转速度缩短动作期限，保留为上下文不等于还能下发动作。
    # 输入：
    #   self：融合快照。
    #   now_unix_ms：准备提交动作的当前时刻。
    # 输出：
    #   deadline：动作可用的最晚毫秒时间，零表示无可用窗口且不授予控制权。
    def control_deadline_unix_ms(self, *, now_unix_ms: int) -> int:
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("CONTROL_DEADLINE_CLOCK_INVALID")
        if not self.fresh_at(now_unix_ms):
            return 0
        maximum_age_ms = round(LOCAL_CONTROL_MAXIMUM_AGE_SECONDS * 1000)
        deadline = min(
            encoding.observed_at_unix_ms
            + min(
                maximum_age_ms,
                encoding.maximum_age_milliseconds,
            )
            for encoding in self.encodings
            if encoding.encoder_role in self.required_roles
        )
        state = next((e for e in self.encodings if e.encoder_role == "flight-state-encoder"), None)
        if state is not None:
            # Canonical gyro channels are 11:14, divided by 5 rad/s. A saturated
            # or masked channel cannot bound rotational aging. This is a
            # freshness veto only; it is not reconstructed attitude/optical flow.
            gyro = state.features[11:14]
            if (
                len(gyro) != 3
                or any(m != 1 for m in state.valid_mask[11:14])
                or any(abs(v) >= 4 for v in gyro)
            ):
                return 0
            rate = 5 * math.hypot(*gyro)
            if rate:
                oldest = min(
                    e.observed_at_unix_ms
                    for e in self.encodings
                    if e.encoder_role in self.required_roles
                )
                deadline = min(deadline, oldest + math.floor(math.radians(10) / rate * 1000))
        # Retained features can remain useful at this instant, but an action
        # needs positive remaining time. Equality cannot authorize dispatch.
        return deadline if deadline > now_unix_ms else 0


class FlightStateSample(StrictModel):
    """Pose/IMU/EKF sample in explicit ENU and body conventions."""

    source_id: str = Field(min_length=1, max_length=120)
    source_sample_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    imu_timestamp_us: int | None = Field(default=None, ge=0)
    observed_at_unix_ms: int = Field(ge=0)
    orientation_world_from_body: QuaternionWxyz
    velocity_world_enu_mps: Vector3
    acceleration_body_mps2: Vector3 | None = None
    # Angular velocity is axial: use right-handed FLU, not reflected FRU.
    angular_velocity_body_rad_s: Vector3 | None = None
    localization_covariance_m2: float | None = Field(default=None, ge=0.0, le=10_000.0)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)


# 功能：
#   将已检查的特征缩放结果压入训练契约区间，保留无效掩码由调用方负责。
# 输入：
#   value：归一化数值，可用正无穷表示无预计接触的时间。
#   minimum：区间下界。
#   maximum：区间上界。
# 输出：
#   clipped：区间内数值。
def _clip(value: float, minimum: float, maximum: float) -> float:
    if math.isnan(value):
        raise ValueError("ENCODER_FEATURE_NAN")
    clipped = min(maximum, max(minimum, value))
    return clipped


# 功能：
#   选择明确的编码时刻，回放由调用方提供时间，拒绝未来采样或非法类型。
# 输入：
#   observed_at_unix_ms：观测采集的绝对毫秒时间。
#   requested：回放或运行方指定的编码时间，未指定时读取系统时间。
# 输出：
#   now_ms：不早于采集时刻的合法整数时间。
def _encoding_clock(observed_at_unix_ms: int, requested: int | None) -> int:
    now_ms = int(time.time() * 1_000) if requested is None else requested
    if type(now_ms) is not int or now_ms < 0:
        raise ValueError("encoder clock must be a nonnegative integer")
    if type(observed_at_unix_ms) is not int or observed_at_unix_ms < 0:
        raise ValueError("observation clock must be a nonnegative integer")
    if now_ms < observed_at_unix_ms:
        raise ValueError("encoder observation is in the future")
    return now_ms


# 功能：
#   用缩放范数避免直接平方溢出，仍超出可表示范围时拒绝而非产生零方向。
# 输入：
#   vector：有限三维向量。
# 输出：
#   magnitude：保留输入单位的有限非负模长。
def _magnitude(vector: Vector3) -> float:
    magnitude = math.hypot(vector.x, vector.y, vector.z)
    if not math.isfinite(magnitude):
        raise ValueError("ENCODER_VECTOR_MAGNITUDE_INVALID")
    return magnitude


# 功能：
#   将公共手柄式前右上向量转换到世界 ENU；右轴只在本边界反射一次，再按机体姿态旋转。
# 输入：
#   orientation_world_from_body：机体前左上坐标到世界坐标的姿态。
#   body：公共前右上控制向量，单位由上层控制契约规定。
# 输出：
#   result：保留输入单位和幅度的世界 ENU 向量。
def body_to_world_enu(orientation_world_from_body: QuaternionWxyz, body: Vector3) -> Vector3:
    rotated = _rotate(orientation_world_from_body, (body.x, -body.y, body.z))
    result = Vector3(x=rotated[0], y=rotated[1], z=rotated[2])
    return result


# 功能：
#   先逆旋转世界 ENU 向量，再将前左上反射为公共前右上，保持与正向转换互逆。
# 输入：
#   orientation_world_from_body：机体到世界坐标的姿态。
#   world：世界 ENU 向量。
# 输出：
#   result：同单位、同幅度的公共前右上向量。
def world_enu_to_body(orientation_world_from_body: QuaternionWxyz, world: Vector3) -> Vector3:
    inverse = QuaternionWxyz(
        w=orientation_world_from_body.w,
        x=-orientation_world_from_body.x,
        y=-orientation_world_from_body.y,
        z=-orientation_world_from_body.z,
    )
    rotated = _rotate(inverse, (world.x, world.y, world.z))
    result = Vector3(x=rotated[0], y=-rotated[1], z=rotated[2])
    return result


# 功能：
#   将机体前右方向映射到顺时针八个方位格，前向位于零号格中心。
# 输入：
#   x_forward：向前分量。
#   y_right：向右分量。
# 输出：
#   sector：零至七的方位索引。
def _sector_index(x_forward: float, y_right: float) -> int:
    angle = math.atan2(y_right, x_forward)
    sector = int(math.floor((angle + _SECTOR_WIDTH_RAD / 2.0) / _SECTOR_WIDTH_RAD)) % 8
    return sector


# 功能：
#   根据仰角选择水平、上环、下环或极区，保持模型训练使用的固定 26 格顺序。
# 输入：
#   x_forward：向前分量。
#   y_right：向右分量。
#   z_up：向上分量。
# 输出：
#   cell：零至二十五的空间格索引。
def _spatial_index(x_forward: float, y_right: float, z_up: float) -> int:
    elevation = math.atan2(z_up, math.hypot(x_forward, y_right))
    if elevation >= math.pi / 3:
        return 25
    if elevation <= -math.pi / 3:
        return 24
    ring = 8 if elevation > math.pi / 12 else 16 if elevation < -math.pi / 12 else 0
    return ring + _sector_index(x_forward, y_right)


# 功能：
#   根据安装姿态和标定视场估算可覆盖的空间格，平面雷达不宣称看到了上下方。
# 输入：
#   mount：传感器到机体的标定。
#   horizontal_fov：水平视场弧度。
#   vertical_fov：垂直视场弧度，零代表平面扫描。
# 输出：
#   cells：预计可覆盖的不可变格集合，不是已经观测到的自由空间。
def _expected_spatial_cells(
    mount: CalibratedRangeSensorMount, horizontal_fov: float, vertical_fov: float
) -> frozenset[int]:
    q = mount.orientation_body_from_sensor
    cells = _calibrated_spatial_support((q.w, q.x, q.y, q.z), horizontal_fov, vertical_fov)
    return cells


# 功能：
#   以安装旋转与视场数值缓存角度采样结果，不缓存观测、时钟或控制许可。
# 输入：
#   quaternion：传感器到机体的单位四元数。
#   horizontal_fov：水平视场弧度。
#   vertical_fov：垂直视场弧度。
# 输出：
#   cells：预计角度支持的不可变索引集合。
@lru_cache(maxsize=64)
def _calibrated_spatial_support(
    quaternion: tuple[float, float, float, float],
    horizontal_fov: float,
    vertical_fov: float,
) -> frozenset[int]:
    # Only calibration-dependent angular support is cached, never measured
    # ranges, occupied/free cells, confidence, sample clocks or motion authority.
    # Values, not sensor names or mutable model objects, identify the calibration.
    orientation = QuaternionWxyz(**dict(zip(("w", "x", "y", "z"), quaternion, strict=True)))
    result = set()
    for azimuth_index in range(33):
        azimuth = horizontal_fov * (azimuth_index / 32 - 0.5)
        for elevation_index in range(17 if vertical_fov else 1):
            elevation = vertical_fov * (elevation_index / 16 - 0.5)
            direction = _rotate(
                orientation,
                (
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ),
            )
            result.add(_spatial_index(direction[0], -direction[1], direction[2]))
    cells = frozenset(result)
    return cells


# 功能：
#   将标定距离样本编码为机体 26 格的最近命中、探测距离、命中率、置信度和覆盖特征。
#   未观测格保留零掩码，稀疏射线不能证明格间或偏置传感器后方无障碍。
# 输入：
#   scan：带采集时间、定位不确定性及距离样本的扫描。
#   sensor_mount：传感器与机体的标定变换和量程。
#   expected_horizontal_fov_rad：标定水平视场弧度。
#   expected_vertical_fov_rad：标定垂直视场弧度。
#   maximum_age_milliseconds：特征保留期，不代表动作允许期限。
#   encoded_at_unix_ms：可选的显式编码时钟。
# 输出：
#   encoding：有掩码、来源散列和覆盖诊断的固定宽度编码。
def encode_metric_geometry(
    scan: RawMetricRangeScan,
    *,
    sensor_mount: CalibratedRangeSensorMount,
    expected_horizontal_fov_rad: float = 2.0 * math.pi,
    expected_vertical_fov_rad: float = 0.0,
    maximum_age_milliseconds: int = 350,
    encoded_at_unix_ms: int | None = None,
) -> RealtimeFeatureEncoding:
    started_ns = time.perf_counter_ns()
    now_ms = _encoding_clock(scan.observed_at_unix_ms, encoded_at_unix_ms)
    # This encoder is also called directly by replay/training consumers. Do not
    # rely on a prior bridge call or a mutable Pydantic object's initial check.
    scan = _strict_copy(scan, RawMetricRangeScan)
    sensor_mount = _strict_copy(sensor_mount, CalibratedRangeSensorMount)
    if sensor_mount.sensor_id != scan.sensor_id:
        raise ValueError("range scan and sensor calibration identify different sensors")
    if (
        type(expected_horizontal_fov_rad) not in (int, float)
        or not 0.05 <= expected_horizontal_fov_rad <= 2.0 * math.pi
    ):
        raise ValueError("expected horizontal field of view is outside (0, 2*pi]")
    if (
        type(expected_vertical_fov_rad) not in (int, float)
        or not 0.0 <= expected_vertical_fov_rad <= math.pi
    ):
        raise ValueError("expected vertical field of view is outside [0, pi]")
    if type(maximum_age_milliseconds) is not int or not 0 < maximum_age_milliseconds <= 30_000:
        raise ValueError("geometry maximum age must be in (0, 30000] ms")
    expected_sectors = _expected_spatial_cells(
        sensor_mount,
        expected_horizontal_fov_rad,
        expected_vertical_fov_rad,
    )
    sector_ranges: list[list[float]] = [[] for _ in range(SPATIAL_CELL_COUNT)]
    sector_free: list[list[float]] = [[] for _ in range(SPATIAL_CELL_COUNT)]
    sector_confidence: list[list[float]] = [[] for _ in range(SPATIAL_CELL_COUNT)]
    sector_hits = [0] * SPATIAL_CELL_COUNT
    rejected_range_samples = 0
    maximum_range = sensor_mount.maximum_range_m
    for sample in scan.samples:
        direction_norm = _magnitude(sample.direction_sensor)
        if direction_norm <= 1e-9 or sample.confidence <= 0.0:
            continue
        if not sensor_mount.minimum_range_m <= sample.range_m <= maximum_range:
            rejected_range_samples += 1
            continue
        direction_body = _rotate(
            sensor_mount.orientation_body_from_sensor,
            (
                sample.direction_sensor.x / direction_norm,
                sample.direction_sensor.y / direction_norm,
                sample.direction_sensor.z / direction_norm,
            ),
        )
        endpoint = (
            sensor_mount.translation_body_m.x + direction_body[0] * sample.range_m,
            -(sensor_mount.translation_body_m.y + direction_body[1] * sample.range_m),
            sensor_mount.translation_body_m.z + direction_body[2] * sample.range_m,
        )
        sector = _spatial_index(*endpoint)
        # Distances are from the vehicle origin after applying the mount offset.
        # These compressed features guide a policy; they do not certify a swept
        # collision-free volume between sparse rays or behind an offset sensor.
        body_distance = math.hypot(*endpoint)
        if not math.isfinite(body_distance):
            raise ValueError("ENCODER_GEOMETRY_DISTANCE_INVALID")
        sector_free[sector].append(body_distance)
        sector_confidence[sector].append(sample.confidence)
        if sample.hit:
            sector_ranges[sector].append(body_distance)
            sector_hits[sector] += 1
    features: list[float] = []
    mask: list[float] = []
    accepted_samples = 0
    for index in range(SPATIAL_CELL_COUNT):
        ranges = sector_ranges[index]
        free = sector_free[index]
        confidence = sector_confidence[index]
        present = bool(free)
        accepted_samples += len(free)
        nearest_hit = min(ranges) if ranges else maximum_range
        free_reach = min(free) if free else 0.0
        hit_ratio = sector_hits[index] / len(free) if free else 0.0
        mean_confidence = sum(confidence) / len(confidence) if confidence else 0.0
        angular_coverage = min(1.0, len(free) / max(1.0, len(scan.samples) / len(expected_sectors)))
        features.extend(
            (
                _clip(nearest_hit / maximum_range, 0.0, 1.0),
                _clip(free_reach / maximum_range, 0.0, 1.0),
                hit_ratio,
                mean_confidence,
                angular_coverage,
            )
        )
        mask.extend((1.0 if present else 0.0,) * 5)
    issues: list[str] = []
    populated_expected = sum(bool(sector_free[index]) for index in expected_sectors)
    coverage = populated_expected / len(expected_sectors)
    if coverage < 0.5:
        issues.append("GEOMETRY_ANGULAR_COVERAGE_LOW")
    if accepted_samples < 32:
        issues.append("GEOMETRY_SAMPLE_COUNT_LOW")
    if rejected_range_samples:
        issues.append("GEOMETRY_RANGE_OUTSIDE_CALIBRATION")
    if scan.source_coverage is not None and scan.source_coverage < 0.05:
        issues.append("GEOMETRY_SOURCE_COVERAGE_LOW")
    source = {
        "sensor_id": scan.sensor_id,
        "sequence": scan.sequence,
        "observed_at_unix_ms": scan.observed_at_unix_ms,
        "sensor_mount": sensor_mount.model_dump(mode="json"),
        "expected_horizontal_fov_rad": expected_horizontal_fov_rad,
        "expected_vertical_fov_rad": expected_vertical_fov_rad,
        "source_coverage": scan.source_coverage,
        "localization_covariance_m2": scan.localization_covariance_m2,
        "samples": [sample.model_dump(mode="json") for sample in scan.samples],
    }
    encoding = RealtimeFeatureEncoding(
        encoder_role="metric-geometry-encoder",
        feature_contract_sha256=encoder_contract_sha256("metric-geometry-encoder"),
        source_ids=[scan.sensor_id],
        source_sha256=sha256_json(source),
        observed_at_unix_ms=scan.observed_at_unix_ms,
        encoded_at_unix_ms=now_ms,
        maximum_age_milliseconds=maximum_age_milliseconds,
        quality=(sum(sample.confidence for sample in scan.samples) / len(scan.samples))
        * (scan.source_coverage if scan.source_coverage is not None else 1.0),
        coverage=coverage,
        uncertainty=_clip(scan.localization_covariance_m2 / 0.25 + (1.0 - coverage), 0.0, 1.0),
        features=features,
        valid_mask=mask,
        inference_latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
        issue_codes=issues,
    )
    return encoding


# 功能：
#   将动态轨迹外推到编码时刻，按预计接触紧迫性选择各格代表目标，保留最旧采集期限。
#   接触时间只计算目标几何与相对运动，实际飞机扫掠体积由下游安全层核验。
# 输入：
#   obstacles：最多 512 个带年龄和速度的动态观测。
#   body_position_world_enu_m：观测时刻的机体 ENU 位置。
#   body_orientation_world_from_body：机体到世界姿态。
#   body_velocity_world_enu_mps：机体世界速度。
#   observed_at_unix_ms：参考采集时刻。
#   maximum_range_m：编码参考量程。
#   maximum_age_milliseconds：目标最大保留期。
#   encoded_at_unix_ms：显式编码时钟。
# 输出：
#   encoding：固定顺序的相对距离、接近速度、接触时间、尺寸和运动特征。
def encode_dynamic_targets(
    obstacles: Sequence[DynamicObstacleObservation],
    *,
    body_position_world_enu_m: Vector3,
    body_orientation_world_from_body: QuaternionWxyz,
    body_velocity_world_enu_mps: Vector3,
    observed_at_unix_ms: int,
    maximum_range_m: float = 30.0,
    maximum_age_milliseconds: int = 500,
    encoded_at_unix_ms: int | None = None,
) -> RealtimeFeatureEncoding:
    started_ns = time.perf_counter_ns()
    maximum_range_m = _positive_limit(maximum_range_m, maximum=10_000.0, name="DYNAMIC_RANGE")
    if type(maximum_age_milliseconds) is not int or not 0 < maximum_age_milliseconds <= 30_000:
        raise ValueError("dynamic target maximum age must be in (0, 30000] ms")
    now_ms = _encoding_clock(observed_at_unix_ms, encoded_at_unix_ms)
    obstacles = tuple(
        _strict_copy(item, DynamicObstacleObservation)
        for item in _bounded_items(obstacles, maximum=512)
    )
    body_position_world_enu_m = _strict_copy(body_position_world_enu_m, Vector3)
    body_velocity_world_enu_mps = _strict_copy(body_velocity_world_enu_mps, Vector3)
    body_orientation_world_from_body = _strict_copy(
        body_orientation_world_from_body, QuaternionWxyz
    )
    additional_age_ms = max(0, now_ms - observed_at_unix_ms)
    ego_age_seconds = additional_age_ms / 1_000
    stale_target_count = 0
    nearest: list[tuple[float, ...] | None] = [None] * SPATIAL_CELL_COUNT
    priorities: list[tuple[float, float, str] | None] = [None] * SPATIAL_CELL_COUNT
    oldest_source_time = observed_at_unix_ms
    for obstacle in obstacles:
        if obstacle.age_seconds * 1_000.0 + additional_age_ms > maximum_age_milliseconds:
            stale_target_count += 1
            continue
        if obstacle.confidence <= 0.0:
            continue
        if math.ceil(obstacle.age_seconds * 1_000) > observed_at_unix_ms:
            raise ValueError("DYNAMIC_TARGET_SOURCE_CLOCK_INVALID")
        age_seconds = obstacle.age_seconds + additional_age_ms / 1_000
        oldest_source_time = min(
            oldest_source_time, observed_at_unix_ms - math.ceil(obstacle.age_seconds * 1_000)
        )
        # Position is the last measured center. Extrapolation is explicit and
        # does not refresh the source deadline. Ego pose is at the frame time.
        relative_world = Vector3(
            x=(
                obstacle.position_m.x
                + obstacle.velocity_mps.x * age_seconds
                - body_position_world_enu_m.x
                - body_velocity_world_enu_mps.x * ego_age_seconds
            ),
            y=(
                obstacle.position_m.y
                + obstacle.velocity_mps.y * age_seconds
                - body_position_world_enu_m.y
                - body_velocity_world_enu_mps.y * ego_age_seconds
            ),
            z=(
                obstacle.position_m.z
                + obstacle.velocity_mps.z * age_seconds
                - body_position_world_enu_m.z
                - body_velocity_world_enu_mps.z * ego_age_seconds
            ),
        )
        relative_body = world_enu_to_body(body_orientation_world_from_body, relative_world)
        distance = _magnitude(relative_body)
        if distance > maximum_range_m:
            continue
        relative_velocity_world = Vector3(
            x=obstacle.velocity_mps.x - body_velocity_world_enu_mps.x,
            y=obstacle.velocity_mps.y - body_velocity_world_enu_mps.y,
            z=obstacle.velocity_mps.z - body_velocity_world_enu_mps.z,
        )
        velocity_body = world_enu_to_body(body_orientation_world_from_body, relative_velocity_world)
        radial_velocity = (
            relative_body.x * velocity_body.x
            + relative_body.y * velocity_body.y
            + relative_body.z * velocity_body.z
        ) / max(distance, 1e-6)
        closing_speed = -radial_velocity
        if not math.isfinite(closing_speed):
            raise ValueError("DYNAMIC_TARGET_RELATIVE_SPEED_INVALID")
        time_to_contact = cylinder_contact_seconds(
            relative_world,
            relative_velocity_world,
            radius_m=obstacle.radius_m,
            height_m=obstacle.height_m,
        )
        sector = _spatial_index(relative_body.x, relative_body.y, relative_body.z)
        candidate = (
            distance,
            closing_speed,
            time_to_contact,
            obstacle.confidence,
            age_seconds,
            velocity_body.x,
            velocity_body.y,
            velocity_body.z,
            obstacle.radius_m,
            obstacle.height_m,
            closest_center_distance_m(relative_world, relative_velocity_world),
        )
        priority = (time_to_contact, distance, obstacle.obstacle_id)
        if priorities[sector] is None or priority < priorities[sector]:
            priorities[sector] = priority
            nearest[sector] = candidate
    features: list[float] = []
    mask: list[float] = []
    populated = 0
    for item in nearest:
        present = item is not None
        populated += int(present)
        if item is None:
            # No retained track is not an observed empty sector. Geometry carries
            # coverage independently, and stale dropped tracks block readiness.
            features.extend([0.0] * 11)
            mask.extend([0.0] * 11)
            continue
        (
            distance,
            closing_speed,
            time_to_contact,
            confidence,
            age_seconds,
            vx,
            vy,
            vz,
            radius,
            height,
            closest,
        ) = item
        features.extend(
            (
                _clip(distance / maximum_range_m, 0.0, 1.0),
                _clip(closing_speed / 10.0, -1.0, 1.0),
                _clip(time_to_contact / 30.0, 0.0, 1.0),
                confidence,
                _clip(age_seconds / (maximum_age_milliseconds / 1_000.0), 0.0, 1.0),
                _clip(vx / 10.0, -1.0, 1.0),
                _clip(vy / 10.0, -1.0, 1.0),
                _clip(vz / 10.0, -1.0, 1.0),
                _clip(radius / maximum_range_m, 0.0, 1.0),
                _clip(height / maximum_range_m, 0.0, 1.0),
                _clip(closest / maximum_range_m, 0.0, 1.0),
            )
        )
        mask.extend([1.0] * 11)
    source = {
        "position": body_position_world_enu_m.model_dump(mode="json"),
        "orientation": body_orientation_world_from_body.model_dump(mode="json"),
        "body_velocity": body_velocity_world_enu_mps.model_dump(mode="json"),
        "observed_at_unix_ms": observed_at_unix_ms,
        "encoded_at_unix_ms": now_ms,
        "maximum_range_m": maximum_range_m,
        "obstacles": [obstacle.model_dump(mode="json") for obstacle in obstacles],
    }
    quality = (
        sum(obstacle.confidence for obstacle in obstacles) / len(obstacles) if obstacles else 1.0
    )
    encoding = RealtimeFeatureEncoding(
        encoder_role="dynamic-target-encoder",
        feature_contract_sha256=encoder_contract_sha256("dynamic-target-encoder"),
        source_ids=["fused-dynamic-tracks"],
        source_sha256=sha256_json(source),
        observed_at_unix_ms=oldest_source_time,
        encoded_at_unix_ms=now_ms,
        maximum_age_milliseconds=maximum_age_milliseconds,
        quality=quality,
        coverage=populated / SPATIAL_CELL_COUNT,
        uncertainty=1.0 - quality,
        features=features,
        valid_mask=mask,
        inference_latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
        issue_codes=["DYNAMIC_TARGET_STALE"] if stale_target_count else [],
    )
    return encoding


class FlightStateEncoder:
    """Temporal state encoder with bounded history and timing validation."""

    # 功能：
    #   初始化有界单流历史，窗口或间隔变化会改变特征契约；实例由单一采样调用方拥有。
    # 输入：
    #   history_length：四至 256 个样本的历史长度。
    #   maximum_gap_ms：允许的相邻采样间隔。
    # 输出：
    #   None：创建空历史和无上次编码的实例。
    def __init__(
        self,
        *,
        history_length: int = FLIGHT_STATE_HISTORY_LENGTH,
        maximum_gap_ms: int = FLIGHT_STATE_MAXIMUM_GAP_MS,
    ) -> None:
        if type(history_length) is not int or not 4 <= history_length <= 256:
            raise ValueError("flight state history length must be in [4, 256]")
        if type(maximum_gap_ms) is not int or not 10 <= maximum_gap_ms <= 5_000:
            raise ValueError("flight state maximum gap must be in [10, 5000] ms")
        self._samples: deque[FlightStateSample] = deque(maxlen=history_length)
        self._last_encoding: RealtimeFeatureEncoding | None = None
        self.maximum_gap_ms = maximum_gap_ms

    # 功能：
    #   根据姿态、速度、IMU 和协方差生成时间窗口特征，输出成功后才提交候选历史。
    #   换源、时钟重置和采样断档重新预热；重复 IMU 包返回原编码，不延长观测寿命。
    # 输入：
    #   sample：当前物理状态样本。
    #   encoded_at_unix_ms：可选的显式编码时间。
    # 输出：
    #   encoding：带缺失掩码、时序诊断和来源契约的状态编码。
    def encode(
        self,
        sample: FlightStateSample,
        *,
        encoded_at_unix_ms: int | None = None,
    ) -> RealtimeFeatureEncoding:
        started_ns = time.perf_counter_ns()
        now_ms = _encoding_clock(sample.observed_at_unix_ms, encoded_at_unix_ms)
        # Revalidate the entire incoming graph, including model_copy updates.
        # Build a candidate history; a failed encoding must not consume a tick.
        sample = _strict_copy(sample, FlightStateSample)
        if type(self.maximum_gap_ms) is not int or not 10 <= self.maximum_gap_ms <= 5_000:
            raise ValueError("flight state maximum gap must be in [10, 5000] ms")
        if self._samples and sample.observed_at_unix_ms <= self._samples[-1].observed_at_unix_ms:
            raise ValueError("flight state timestamps must increase strictly")
        reset_issue = None
        if self._samples:
            previous = self._samples[-1]
            if previous.source_id != sample.source_id:
                reset_issue = "FLIGHT_STATE_SOURCE_CHANGED"
            elif sample.imu_timestamp_us is not None and previous.imu_timestamp_us is not None:
                if sample.imu_timestamp_us < previous.imu_timestamp_us:
                    reset_issue = "FLIGHT_STATE_IMU_CLOCK_RESET"
                elif sample.imu_timestamp_us == previous.imu_timestamp_us:
                    # Transport can repeat a sensor packet with a new receive
                    # time. It is still one physical observation, not another
                    # training/history row and not a renewed control deadline.
                    assert self._last_encoding is not None
                    return self._last_encoding.model_copy(deep=True)
            if reset_issue is None and (
                sample.observed_at_unix_ms - previous.observed_at_unix_ms > self.maximum_gap_ms
            ):
                reset_issue = "FLIGHT_STATE_SAMPLE_GAP"
        pending = deque(
            () if reset_issue is not None else self._samples, maxlen=self._samples.maxlen
        )
        pending.append(sample)
        samples = list(pending)
        gaps = [
            samples[index].observed_at_unix_ms - samples[index - 1].observed_at_unix_ms
            for index in range(1, len(samples))
        ]
        current_velocity_body = world_enu_to_body(
            sample.orientation_world_from_body, sample.velocity_world_enu_mps
        )
        values = [
            sample.orientation_world_from_body.w,
            sample.orientation_world_from_body.x,
            sample.orientation_world_from_body.y,
            sample.orientation_world_from_body.z,
            _clip(current_velocity_body.x / 20.0, -1.0, 1.0),
            _clip(current_velocity_body.y / 20.0, -1.0, 1.0),
            _clip(current_velocity_body.z / 20.0, -1.0, 1.0),
            (
                _clip(sample.localization_covariance_m2 / 0.25, 0.0, 4.0)
                if sample.localization_covariance_m2 is not None
                else 0.0
            ),
        ]
        mask = [1.0] * len(values)
        mask[7] = float(sample.localization_covariance_m2 is not None)
        for vector, scale in (
            (sample.acceleration_body_mps2, 9.80665),
            (sample.angular_velocity_body_rad_s, 5.0),
        ):
            if vector is None:
                values.extend((0.0, 0.0, 0.0))
                mask.extend((0.0, 0.0, 0.0))
            else:
                values.extend(
                    _clip(component / scale, -4.0, 4.0)
                    for component in (vector.x, vector.y, vector.z)
                )
                mask.extend((1.0, 1.0, 1.0))
        velocity_series = [
            world_enu_to_body(item.orientation_world_from_body, item.velocity_world_enu_mps)
            for item in samples
        ]
        for axis in ("x", "y", "z"):
            # Each sample uses its contemporaneous attitude. These are temporal
            # body-relative velocity statistics, not inertial acceleration.
            axis_values = [float(getattr(item, axis)) for item in velocity_series]
            mean = sum(axis_values) / len(axis_values)
            variance = sum((value - mean) ** 2 for value in axis_values) / len(axis_values)
            values.extend(
                (
                    _clip(mean / 20.0, -1.0, 1.0),
                    _clip(math.sqrt(variance) / 5.0, 0.0, 2.0),
                )
            )
            mask.extend((1.0, 1.0))
        issues: list[str] = []
        if sample.localization_covariance_m2 is None:
            issues.append("FLIGHT_STATE_COVARIANCE_UNAVAILABLE")
        elif sample.localization_covariance_m2 > 0.25:
            issues.append("FLIGHT_STATE_LOCALIZATION_UNCERTAIN")
        if sample.acceleration_body_mps2 is None or sample.angular_velocity_body_rad_s is None:
            issues.append("FLIGHT_STATE_IMU_UNAVAILABLE")
        if reset_issue is not None:
            issues.append(reset_issue)
        if gaps and max(gaps) > self.maximum_gap_ms:
            issues.append("FLIGHT_STATE_SAMPLE_GAP")
        if len(samples) < self._samples.maxlen:
            issues.append("FLIGHT_STATE_HISTORY_WARMING")
        source_sha256 = sha256_json([item.model_dump(mode="json") for item in samples])
        coverage = len(samples) / int(self._samples.maxlen or 1)
        encoding = RealtimeFeatureEncoding(
            encoder_role="flight-state-encoder",
            feature_contract_sha256=(
                encoder_contract_sha256(
                    "flight-state-encoder",
                    history_length=int(self._samples.maxlen or 1),
                    maximum_gap_ms=self.maximum_gap_ms,
                )
                if all(item.source_sample_sha256 is not None for item in samples)
                else None
            ),
            source_ids=[sample.source_id],
            source_sha256=source_sha256,
            observed_at_unix_ms=sample.observed_at_unix_ms,
            encoded_at_unix_ms=now_ms,
            maximum_age_milliseconds=self.maximum_gap_ms,
            quality=sample.quality,
            coverage=coverage,
            uncertainty=(
                _clip(sample.localization_covariance_m2 / 0.25, 0.0, 1.0)
                if sample.localization_covariance_m2 is not None
                else 1.0
            ),
            features=values,
            valid_mask=mask,
            inference_latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            issue_codes=issues,
        )
        self._samples = pending
        self._last_encoding = encoding.model_copy(deep=True)
        return encoding


# 功能：
#   从必需角色的时效、质量和阻断诊断推导就绪状态，可选上下文不能冒充缺失的传感器。
# 输入：
#   encodings：已校验的角色编码。
#   required_roles：唯一且有效的必需角色序列。
#   captured_at_unix_ms：融合时刻。
# 输出：
#   issues：按角色顺序生成的阻断原因。
def _snapshot_readiness_issues(
    encodings: Sequence[RealtimeFeatureEncoding],
    required_roles: Sequence[FeatureEncoderRole],
    captured_at_unix_ms: int,
) -> list[str]:
    if not required_roles or len(set(required_roles)) != len(required_roles):
        raise ValueError("required encoder roles must be nonempty and unique")
    by_role = {encoding.encoder_role: encoding for encoding in encodings}
    issues: list[str] = []
    for role in required_roles:
        if role not in _ENCODER_ROLE_ORDER:
            raise ValueError(f"unsupported required encoder role: {role}")
        encoding = by_role.get(role)
        if encoding is None:
            issues.append(f"REQUIRED_ENCODER_MISSING:{role}")
        elif not encoding.fresh_at(captured_at_unix_ms):
            issues.append(f"REQUIRED_ENCODER_STALE:{role}")
        elif encoding.quality < 0.35:
            issues.append(f"REQUIRED_ENCODER_LOW_QUALITY:{role}")
        else:
            issues.extend(
                f"REQUIRED_ENCODER_NOT_READY:{role}:{issue}"
                for issue in encoding.issue_codes
                if issue in _BLOCKING_ENCODER_ISSUES
            )
    return issues


# 功能：
#   绑定有序张量、掩码、来源时钟和诊断，不仅散列形状或角色名字。
# 输入：
#   encodings：当前有序编码。
#   required_roles：必需角色。
#   captured_at_unix_ms：融合时刻。
#   issues：由编码推导的就绪问题。
# 输出：
#   payload：用于快照身份校验的独立标准对象。
def _snapshot_hash_payload(
    encodings: Sequence[RealtimeFeatureEncoding],
    required_roles: Sequence[FeatureEncoderRole],
    captured_at_unix_ms: int,
    issues: list[str],
) -> dict[str, object]:
    payload = {
        "captured_at_unix_ms": captured_at_unix_ms,
        "required_roles": list(required_roles),
        "encodings": [
            {
                key: value
                for key, value in encoding.model_dump(mode="json").items()
                if key != "feature_contract_sha256" or value is not None
            }
            for encoding in encodings
        ],
        "issues": issues,
    }
    return payload


# 功能：
#   独立复制编码、按固定角色顺序融合并重算就绪状态；融合不创造传感器证据或更新其采集时间。
# 输入：
#   encodings：最多三个不同角色的编码。
#   captured_at_unix_ms：当前融合毫秒时间。
#   required_roles：当前调用必须存在的传感器角色。
# 输出：
#   snapshot：带真实阻断原因和内容散列的融合输入。
def fuse_realtime_features(
    encodings: Iterable[RealtimeFeatureEncoding],
    *,
    captured_at_unix_ms: int,
    required_roles: Iterable[FeatureEncoderRole] = _ENCODER_ROLE_ORDER,
) -> RealtimeFeatureSnapshot:
    if type(captured_at_unix_ms) is not int or captured_at_unix_ms < 0:
        raise ValueError("ENCODER_FUSION_CLOCK_INVALID")
    required = tuple(_bounded_items(required_roles, maximum=3))
    # Callers retain their original encoding objects. A fused input must own
    # the exact graph its hash describes, and reject mutated nested values.
    supplied = [
        _strict_copy(item, RealtimeFeatureEncoding) for item in _bounded_items(encodings, maximum=3)
    ]
    if not supplied:
        raise ValueError("at least one realtime feature encoding is required")
    by_role = {encoding.encoder_role: encoding for encoding in supplied}
    if len(by_role) != len(supplied):
        raise ValueError("feature snapshot cannot contain duplicate encoder roles")
    ordered = [by_role[role] for role in _ENCODER_ROLE_ORDER if role in by_role]
    issues = _snapshot_readiness_issues(ordered, required, captured_at_unix_ms)
    values = [value for encoding in ordered for value in encoding.features]
    mask = [value for encoding in ordered for value in encoding.valid_mask]
    if (
        set(required) == set(_ENCODER_ROLE_ORDER)
        and not issues
        and len(values) != REALTIME_CONTROL_FEATURE_COUNT
    ):
        raise ValueError("realtime control feature contract width drifted")
    payload = _snapshot_hash_payload(ordered, required, captured_at_unix_ms, issues)
    snapshot = RealtimeFeatureSnapshot(
        captured_at_unix_ms=captured_at_unix_ms,
        required_roles=list(required),
        encodings=ordered,
        ready_for_control=not issues,
        fused_features=values,
        fused_valid_mask=mask,
        snapshot_sha256=sha256_json(payload),
        issue_codes=issues,
    )
    return snapshot


# 功能：
#   在深度处理后合入较新的同源飞行状态，几何和动态轨迹仍保留原采集期限与坐标。
#   新快照只属于下一次模型请求，不回写已发布安全回执中的输入。
# 输入：
#   snapshot：之前的有效融合快照。
#   flight_state：较新的同源、同契约状态编码。
#   captured_at_unix_ms：新融合时刻。
# 输出：
#   refreshed：重新散列并检查就绪状态的快照。
def refresh_flight_state_features(
    snapshot: RealtimeFeatureSnapshot,
    flight_state: RealtimeFeatureEncoding,
    *,
    captured_at_unix_ms: int,
) -> RealtimeFeatureSnapshot:
    snapshot = _strict_copy(snapshot, RealtimeFeatureSnapshot)
    flight_state = _strict_copy(flight_state, RealtimeFeatureEncoding)
    if type(captured_at_unix_ms) is not int or captured_at_unix_ms < 0:
        raise ValueError("CONTROL_REFRESH_NATIVE_CLOCK_INVALID")
    previous = next(
        (row for row in snapshot.encodings if row.encoder_role == "flight-state-encoder"), None
    )
    if (
        previous is None
        or flight_state.encoder_role != "flight-state-encoder"
        or flight_state.source_ids != previous.source_ids
        or flight_state.feature_contract_sha256 != previous.feature_contract_sha256
    ):
        raise ValueError("CONTROL_REFRESH_NATIVE_SOURCE_OR_CONTRACT_CHANGED")
    if (
        flight_state.observed_at_unix_ms < previous.observed_at_unix_ms
        or captured_at_unix_ms < snapshot.captured_at_unix_ms
        or flight_state.encoded_at_unix_ms > captured_at_unix_ms
    ):
        raise ValueError("CONTROL_REFRESH_NATIVE_CLOCK_INVALID")
    refreshed = fuse_realtime_features(
        [
            flight_state.model_copy(deep=True)
            if row.encoder_role == "flight-state-encoder"
            else row.model_copy(deep=True)
            for row in snapshot.encodings
        ],
        captured_at_unix_ms=captured_at_unix_ms,
        required_roles=snapshot.required_roles,
    )
    return refreshed


# 功能：
#   将已明确选择的候选路径接近点换算为前右上速度请求，保留 candidate-path-derived 来源标签。
#   此兼容接口不是神经网络的连续杆量输出，不能冒充学习控制或绕过时效与安全仲裁。
# 输入：
#   source_expert：提交候选的本地专家角色。
#   model_call_id：对应模型调用标识。
#   navigation_snapshot_sha256：该调用使用的导航快照。
#   task_reference_sha256：当前宏观任务参考的独立散列。
#   current_position_world_enu_m：当前机体世界位置。
#   target_position_world_enu_m：已选候选的接近点。
#   body_orientation_world_from_body：当前机体到世界的姿态。
#   generated_at_unix_ms：请求生成时刻。
#   maximum_speed_mps：合速度上限。
#   maximum_acceleration_mps2：后续控制层执行的加速度约束。
#   maximum_jerk_mps3：后续控制层执行的跃度约束。
#   horizon_seconds：把相对位移换算为速度的参考时间窗。
#   validity_milliseconds：请求保留期，最终受原始传感器期限进一步约束。
#   maximum_yaw_rate_dps：顺时针偏航角速度幅度上限。
# 输出：
#   intent：有物理单位和来源标签的控制请求，不是已执行的动作。
def body_control_intent_for_target(
    *,
    source_expert: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ],
    model_call_id: str,
    navigation_snapshot_sha256: str,
    task_reference_sha256: str,
    current_position_world_enu_m: Vector3,
    target_position_world_enu_m: Vector3,
    body_orientation_world_from_body: QuaternionWxyz,
    generated_at_unix_ms: int,
    maximum_speed_mps: float,
    maximum_acceleration_mps2: float,
    maximum_jerk_mps3: float,
    horizon_seconds: float = 0.2,
    validity_milliseconds: int = 500,
    maximum_yaw_rate_dps: float = 45.0,
) -> BodyFrameControlIntent:
    maximum_speed_mps = _positive_limit(maximum_speed_mps, maximum=20.0, name="SPEED")
    maximum_yaw_rate_dps = _positive_limit(maximum_yaw_rate_dps, maximum=180.0, name="YAW_RATE")
    maximum_acceleration_mps2 = _positive_limit(
        maximum_acceleration_mps2, maximum=30.0, name="ACCELERATION"
    )
    maximum_jerk_mps3 = _positive_limit(maximum_jerk_mps3, maximum=100.0, name="JERK")
    _control_clock(generated_at_unix_ms, validity_milliseconds)
    current_position_world_enu_m = _strict_copy(current_position_world_enu_m, Vector3)
    target_position_world_enu_m = _strict_copy(target_position_world_enu_m, Vector3)
    body_orientation_world_from_body = _strict_copy(
        body_orientation_world_from_body, QuaternionWxyz
    )
    if type(horizon_seconds) not in (int, float) or not 0.05 <= horizon_seconds <= 1.0:
        raise ValueError("body control horizon must be in [0.05, 1.0] seconds")
    delta_world = Vector3(
        x=target_position_world_enu_m.x - current_position_world_enu_m.x,
        y=target_position_world_enu_m.y - current_position_world_enu_m.y,
        z=target_position_world_enu_m.z - current_position_world_enu_m.z,
    )
    requested_world = Vector3(
        x=delta_world.x / horizon_seconds,
        y=delta_world.y / horizon_seconds,
        z=delta_world.z / horizon_seconds,
    )
    speed = _magnitude(requested_world)
    if speed > maximum_speed_mps and speed > 1e-9:
        scale = maximum_speed_mps / speed
        requested_world = Vector3(
            x=requested_world.x * scale,
            y=requested_world.y * scale,
            z=requested_world.z * scale,
        )
    requested_body = world_enu_to_body(body_orientation_world_from_body, requested_world)
    horizontal_bearing_deg = math.degrees(math.atan2(requested_body.y, requested_body.x))
    yaw_rate = _clip(
        horizontal_bearing_deg / horizon_seconds,
        -maximum_yaw_rate_dps,
        maximum_yaw_rate_dps,
    )
    if math.hypot(requested_body.x, requested_body.y) < 0.02:
        yaw_rate = 0.0
    intent = BodyFrameControlIntent(
        source_expert=source_expert,
        control_origin="candidate-path-derived",
        model_call_id=model_call_id,
        navigation_snapshot_sha256=navigation_snapshot_sha256,
        task_reference_sha256=task_reference_sha256,
        generated_at_unix_ms=generated_at_unix_ms,
        valid_until_unix_ms=generated_at_unix_ms + validity_milliseconds,
        forward_velocity_mps=requested_body.x,
        right_velocity_mps=requested_body.y,
        up_velocity_mps=requested_body.z,
        yaw_rate_dps=yaw_rate,
        maximum_acceleration_mps2=maximum_acceleration_mps2,
        maximum_jerk_mps3=maximum_jerk_mps3,
    )
    return intent


# 功能：
#   把本地模型输出的连续归一化四轴映射成物理速度和偏航角速度，保留正负方向与幅度。
#   模型不掌握电机混控，下游根据实测状态继续约束加速度、跃度、碰撞与期限。
# 输入：
#   source_expert：本地连续控制专家角色。
#   model_call_id：本次推理标识。
#   navigation_snapshot_sha256：本次观测输入散列。
#   task_reference_sha256：当前任务参考散列。
#   pilot_control：模型输出的前、右、上、偏航四个归一化轴。
#   generated_at_unix_ms：请求生成时刻。
#   maximum_horizontal_speed_mps：满水平杆对应的合速度上限。
#   maximum_vertical_speed_mps：满垂直杆对应的速度上限。
#   maximum_yaw_rate_dps：满偏航杆对应的角速度上限。
#   maximum_acceleration_mps2：传给安全控制层的加速度约束。
#   maximum_jerk_mps3：传给安全控制层的跃度约束。
#   validity_milliseconds：请求保留期，不能刷新传感器期限。
#   harness_control_scale：只能减小幅度的 Harness 比例。
# 输出：
#   intent：标为 continuous-model-output 的请求，仍需仲裁和飞控执行回执。
def body_control_intent_for_pilot_control(
    *,
    source_expert: Literal[
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ],
    model_call_id: str,
    navigation_snapshot_sha256: str,
    task_reference_sha256: str,
    pilot_control: NormalizedPilotControl,
    generated_at_unix_ms: int,
    maximum_horizontal_speed_mps: float,
    maximum_vertical_speed_mps: float,
    maximum_yaw_rate_dps: float,
    maximum_acceleration_mps2: float,
    maximum_jerk_mps3: float,
    validity_milliseconds: int = 500,
    harness_control_scale: float = 1.0,
) -> BodyFrameControlIntent:
    maximum_horizontal_speed_mps = _positive_limit(
        maximum_horizontal_speed_mps, maximum=20.0, name="HORIZONTAL_SPEED"
    )
    maximum_vertical_speed_mps = _positive_limit(
        maximum_vertical_speed_mps, maximum=20.0, name="VERTICAL_SPEED"
    )
    maximum_yaw_rate_dps = _positive_limit(maximum_yaw_rate_dps, maximum=180.0, name="YAW_RATE")
    maximum_acceleration_mps2 = _positive_limit(
        maximum_acceleration_mps2, maximum=30.0, name="ACCELERATION"
    )
    maximum_jerk_mps3 = _positive_limit(maximum_jerk_mps3, maximum=100.0, name="JERK")
    _control_clock(generated_at_unix_ms, validity_milliseconds)
    pilot_control = _strict_copy(pilot_control, NormalizedPilotControl)
    if type(harness_control_scale) not in (int, float) or not 0.1 <= harness_control_scale <= 1.0:
        raise ValueError("Harness control scale must be in [0.1, 1]")
    forward, right, up, yaw = physical_pilot_request(
        pilot_control,
        PilotControlLimits(
            maximum_horizontal_speed_mps, maximum_vertical_speed_mps, maximum_yaw_rate_dps
        ),
        harness_scale=harness_control_scale,
    )
    intent = BodyFrameControlIntent(
        source_expert=source_expert,
        control_origin="continuous-model-output",
        model_call_id=model_call_id,
        navigation_snapshot_sha256=navigation_snapshot_sha256,
        task_reference_sha256=task_reference_sha256,
        generated_at_unix_ms=generated_at_unix_ms,
        valid_until_unix_ms=generated_at_unix_ms + validity_milliseconds,
        harness_control_scale=harness_control_scale,
        forward_velocity_mps=forward,
        right_velocity_mps=right,
        up_velocity_mps=up,
        yaw_rate_dps=yaw,
        maximum_acceleration_mps2=maximum_acceleration_mps2,
        maximum_jerk_mps3=maximum_jerk_mps3,
    )
    return intent


# 功能：
#   在计算控制请求到期时间之前检查整数时间与契约范围，布尔值不是一毫秒或一秒。
# 输入：
#   generated_at_unix_ms：生成时刻。
#   validity_milliseconds：20 至 2000 毫秒的请求窗口。
# 输出：
#   None：合法时继续，否则拒绝生成控制请求。
def _control_clock(generated_at_unix_ms: int, validity_milliseconds: int) -> None:
    if type(generated_at_unix_ms) is not int or generated_at_unix_ms < 0:
        raise ValueError("CONTROL_INTENT_CLOCK_INVALID")
    if type(validity_milliseconds) is not int or not 20 <= validity_milliseconds <= 2_000:
        raise ValueError(
            "CONTROL_INTENT_VALIDITY_INVALID: body-frame control horizon must be in [20, 2000] ms"
        )
