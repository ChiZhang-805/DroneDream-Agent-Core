"""Lossless frame identifiers and bounded source-clock odometry pairing.

This boundary deliberately does not infer clock domains from similar numbers,
reinterpret MAVSDK's legacy frame enum, or authorize a flight. Callers must bind
both streams to the same independently established clock domain. In particular,
a simulator timestamp is not automatically a physical vehicle's boot clock.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, replace

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import QuaternionWxyz, Vector3
from .hashing import sha256_json
from .native_pose import _multiply, finite_number


# 功能：校验协议整数的精确类型和范围，拒绝布尔值及浮点数冒充时钟或实体 ID。
# 输入：value：协议值；minimum、maximum：闭区间。
# 输出：原始整数；非法值抛出统一协议错误。
def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("NATIVE_ODOMETRY_INTEGER_INVALID")
    return value


# 功能：校验有限物理数值，不自动将文本、未知值或溢出整数转为零。
# 输入：value：协议标量。
# 输出：有限浮点数。
def _real(value):
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        raise ValueError("NATIVE_ODOMETRY_VALUE_INVALID")
    return float(value)


# 功能：拒绝 JSON 重复键，防止不同读者对时间戳、坐标系或重置次数产生不同解释。
# 输入：pairs：JSON 解码器保留的有序键值对。
# 输出：无重复键的字典。
def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("NATIVE_ODOMETRY_DUPLICATE_FIELD")
        result[key] = value
    return result


# 功能：保留 MAVLink 协方差未知项，不将未报告的交叉相关项解释成零。
# 输入：value：21 项上三角数组；JSON null 表示 SDK 转换的原始 NaN。
# 输出：不可变协方差元组；首项未知仍保留其余原始字段供诊断。
def _covariance(value):
    if type(value) is not list or len(value) != 21:
        raise ValueError("NATIVE_ODOMETRY_COVARIANCE_SHAPE_INVALID")
    output = tuple(None if item is None else _real(item) for item in value)
    if any(output[index] is not None and output[index] < 0 for index in (0, 6, 11, 15, 18, 20)):
        raise ValueError("NATIVE_ODOMETRY_VARIANCE_NEGATIVE")
    return output


@dataclass(frozen=True)
class SourceOdometry:
    system_id: int
    component_id: int
    timestamp_us: int
    received_monotonic_seconds: float
    frame_id: int
    child_frame_id: int
    reset_counter: int
    estimator_type: int
    quality: int
    position_frame_m: tuple[float, float, float]
    orientation_frame_from_body_wxyz: tuple[float, float, float, float]
    velocity_child_frame_m_s: tuple[float, float, float]
    angular_velocity_child_frame_rad_s: tuple[float, float, float]
    pose_covariance_upper: tuple[float | None, ...]
    velocity_covariance_upper: tuple[float | None, ...]

    # 功能：阻止直接构造不可变对象绕过协议检查，所有入口遵守同一数值与容量边界。
    # 输入：self：待构造的消息。
    # 输出：无；畸形字段立即抛错。
    def __post_init__(self):
        _integer(self.system_id, 1, 255)
        _integer(self.component_id, 1, 255)
        _integer(self.timestamp_us, 0, 2**64 - 1)
        for value in (self.frame_id, self.child_frame_id, self.reset_counter, self.estimator_type):
            _integer(value, 0, 255)
        _integer(self.quality, -1, 100)
        if _real(self.received_monotonic_seconds) < 0:
            raise ValueError("NATIVE_ODOMETRY_RECEIPT_INVALID")
        for values, length in (
            (self.position_frame_m, 3),
            (self.orientation_frame_from_body_wxyz, 4),
            (self.velocity_child_frame_m_s, 3),
            (self.angular_velocity_child_frame_rad_s, 3),
        ):
            if type(values) is not tuple or len(values) != length:
                raise ValueError("NATIVE_ODOMETRY_VECTOR_INVALID")
            for value in values:
                _real(value)
        if abs(math.hypot(*self.orientation_frame_from_body_wxyz) - 1.0) > 1e-4:
            raise ValueError("NATIVE_ODOMETRY_QUATERNION_INVALID")
        for value in (self.pose_covariance_upper, self.velocity_covariance_upper):
            if type(value) is not tuple:
                raise ValueError("NATIVE_ODOMETRY_COVARIANCE_SHAPE_INVALID")
            _covariance(list(value))


# 功能：读取原始 ODOMETRY 的真实协议坐标编号、来源时刻及重置计数，不使用 SDK 的旧枚举。
# 输入：fields_json：有界消息 JSON；system_id、component_id：实际发送实体。
#   received_monotonic：接收钟。
# 输出：不可变原始消息；没有地图变换、协方差缩小或飞行授权。
def decode_source_odometry(fields_json, *, system_id, component_id, received_monotonic):
    if type(fields_json) is not str or len(fields_json.encode("utf-8")) > 8192:
        raise ValueError("NATIVE_ODOMETRY_MESSAGE_TOO_LARGE")
    try:
        data = json.loads(fields_json, object_pairs_hook=_unique)
    except (RecursionError, json.JSONDecodeError) as error:
        raise ValueError("NATIVE_ODOMETRY_MESSAGE_INVALID") from error
    if (
        type(data) is not dict
        or data.get("message_name") != "ODOMETRY"
        or type(data.get("message_id")) is not int
        or data["message_id"] != 331
    ):
        raise ValueError("NATIVE_ODOMETRY_MESSAGE_INVALID")
    try:
        quaternion = data["q"]
        if type(quaternion) is not list or len(quaternion) != 4:
            raise ValueError("NATIVE_ODOMETRY_QUATERNION_INVALID")
        quaternion = tuple(_real(value) for value in quaternion)
        norm = math.hypot(*quaternion)
        if abs(norm - 1.0) > 1e-4:
            raise ValueError("NATIVE_ODOMETRY_QUATERNION_INVALID")
        received = _real(received_monotonic)
        if received < 0:
            raise ValueError("NATIVE_ODOMETRY_RECEIPT_INVALID")
        # Raw JSON transport rounds floats; preserve the reported quaternion.
        # Any later geometric normalization must remain explicit at conversion.
        return SourceOdometry(
            _integer(system_id, 1, 255),
            _integer(component_id, 1, 255),
            _integer(data["time_usec"], 0, 2**64 - 1),
            received,
            _integer(data["frame_id"], 0, 255),
            _integer(data["child_frame_id"], 0, 255),
            _integer(data["reset_counter"], 0, 255),
            _integer(data["estimator_type"], 0, 255),
            _integer(data["quality"], -1, 100),
            tuple(_real(data[axis]) for axis in ("x", "y", "z")),
            quaternion,
            tuple(_real(data[axis]) for axis in ("vx", "vy", "vz")),
            tuple(_real(data[axis]) for axis in ("rollspeed", "pitchspeed", "yawspeed")),
            _covariance(data["pose_covariance"]),
            _covariance(data["velocity_covariance"]),
        )
    except KeyError as error:
        raise ValueError("NATIVE_ODOMETRY_FIELD_MISSING") from error


# 功能：从同一原始包生成控制与图像消费者共享的遥测字段，避免两条链路解释不同。
# 输入：packet：已解码的原始 ODOMETRY。
# 输出：独立的规范字段；未知协方差保留为空，不修改数值或授予定位资格。
def canonical_odometry_fields(packet):
    if (type(packet) is not SourceOdometry or packet.frame_id not in (1, 20)
            or packet.estimator_type != 8 or packet.quality == -1):
        raise ValueError("NATIVE_ODOMETRY_AUTOPILOT_FRAME_UNSUPPORTED")
    return {
        "timestamp_us": packet.timestamp_us,
        "frame_id": {1: "LOCAL_NED", 20: "LOCAL_FRD"}[packet.frame_id],
        "child_frame_id": {1: "LOCAL_NED", 12: "BODY_FRD", 20: "LOCAL_FRD"}.get(
            packet.child_frame_id, "UNSUPPORTED"),
        "pose_covariance_upper_m2": (list(packet.pose_covariance_upper)
            if packet.pose_covariance_upper[0] is not None else None),
        "twist_covariance_upper": (list(packet.velocity_covariance_upper)
            if packet.velocity_covariance_upper[0] is not None else None),
        "velocity_child_frame_m_s": list(packet.velocity_child_frame_m_s),
        "reset_counter": packet.reset_counter,
        "estimator_type": packet.estimator_type,
        "quality": packet.quality,
    }


# 功能：将已明确为原始 MAV_FRAME_LOCAL_NED 的位姿转换到固定地图 ENU/FLU。
# 输入：packet：原始飞控消息；binding、binding_sha256：既有部署原点及摘要。
# 输出：位置、姿态、速度三元组；不修改消息协方差，不证明时间同步或精度。
def source_pose_in_map(packet, *, binding, binding_sha256):
    if (
        type(packet) is not SourceOdometry
        or packet.frame_id != 1
        or packet.child_frame_id not in (1, 12)
        or packet.estimator_type != 8
        or packet.quality == -1
    ):
        raise ValueError("NATIVE_ODOMETRY_MAP_FRAME_UNSUPPORTED")
    binding = copy_json(binding, limit=16384)
    if (
        type(binding) is not dict
        or sha256_json(binding) != binding_sha256
        or binding.get("orientation") != "NED-to-ENU-fixed"
        or binding.get("source") != "deployment-coordinate-contract"
    ):
        raise ValueError("NATIVE_ODOMETRY_MAP_BINDING_INVALID")
    origin = binding.get("collision_center_origin_world_enu_m")
    if type(origin) is not dict or set(origin) != {"x", "y", "z"}:
        raise ValueError("NATIVE_ODOMETRY_MAP_ORIGIN_INVALID")
    origin = [finite_number(origin, key) for key in "xyz"]
    norm = math.hypot(*packet.orientation_frame_from_body_wxyz)
    q = tuple(value / norm for value in packet.orientation_frame_from_body_wxyz)
    world_q = _multiply(_multiply((0, math.sqrt(0.5), math.sqrt(0.5), 0), q), (0, 1, 0, 0))
    if world_q[0] < 0:
        world_q = tuple(-value for value in world_q)
    velocity = packet.velocity_child_frame_m_s
    if packet.child_frame_id == 12:  # Rotate BODY_FRD before the world basis change.
        rotated = _multiply(_multiply(q, (0, *velocity)), (q[0], -q[1], -q[2], -q[3]))
        velocity = rotated[1:]
    north, east, down = packet.position_frame_m
    return (
        Vector3(x=origin[0] + east, y=origin[1] + north, z=origin[2] - down),
        QuaternionWxyz(w=world_q[0], x=world_q[1], y=world_q[2], z=world_q[3]),
        Vector3(x=velocity[1], y=velocity[0], z=-velocity[2]),
    )


class SourceOdometryHistory:
    """Single-owner, no interpolation, no extrapolation, no receipt-time matching."""

    # 功能：绑定本次连接的实体与时钟域；时钟域必须来自调用方已核验的运行配置。
    # 输入：clock_domain：运行独有时钟身份；system_id、component_id：期望飞控实体；capacity：容量。
    # 输出：空历史；不自行推断硬件钟与仿真钟相同。
    def __init__(self, *, clock_domain, system_id, component_id, capacity=256):
        if type(clock_domain) is not str or not 1 <= len(clock_domain) <= 160:
            raise ValueError("NATIVE_ODOMETRY_CLOCK_DOMAIN_INVALID")
        self.clock_domain = clock_domain
        self.identity = (_integer(system_id, 1, 255), _integer(component_id, 1, 255))
        _integer(capacity, 2, 512)
        self._history = deque(maxlen=capacity)
        self._available = {}
        self.reset_count = 0
        self._invalidated = False

    # 功能：维护真实来源历史；重置、坐标变化和异常清空旧样本，重复包不更新接收时刻。
    # 输入：packet：经严格解码的完整原始消息；available_at_monotonic：本消费者首次可用时刻。
    # 输出：是否新增独立观测；源时间回退或冲突抛错并清空。
    def ingest(self, packet, *, available_at_monotonic=None):
        if self._invalidated:
            raise ValueError("NATIVE_ODOMETRY_HISTORY_INVALIDATED")
        if (
            type(packet) is not SourceOdometry
            or (packet.system_id, packet.component_id) != self.identity
        ):
            self._history.clear()
            self._available.clear()
            self._invalidated = True
            raise ValueError("NATIVE_ODOMETRY_SOURCE_CHANGED")
        available = (
            packet.received_monotonic_seconds
            if available_at_monotonic is None
            else _real(available_at_monotonic)
        )
        if available < packet.received_monotonic_seconds:
            raise ValueError("NATIVE_ODOMETRY_AVAILABILITY_BEFORE_RECEIPT")
        if self._history:
            previous = self._history[-1]
            if (
                packet.timestamp_us < previous.timestamp_us
                or packet.received_monotonic_seconds < previous.received_monotonic_seconds
            ):
                self._history.clear()
                self._available.clear()
                self._invalidated = True
                raise ValueError("NATIVE_ODOMETRY_CLOCK_REGRESSED")
            if packet.timestamp_us == previous.timestamp_us:
                if (
                    replace(packet, received_monotonic_seconds=previous.received_monotonic_seconds)
                    != previous
                ):
                    self._history.clear()
                    self._available.clear()
                    self._invalidated = True
                    raise ValueError("NATIVE_ODOMETRY_SOURCE_CONFLICT")
                return False
            if (
                packet.reset_counter,
                packet.frame_id,
                packet.child_frame_id,
                packet.estimator_type,
            ) != (
                previous.reset_counter,
                previous.frame_id,
                previous.child_frame_id,
                previous.estimator_type,
            ):
                self._history.clear()
                self._available.clear()
                self.reset_count += 1
        if len(self._history) == self._history.maxlen:
            self._available.pop(self._history[0].timestamp_us)
        self._history.append(packet)
        self._available[packet.timestamp_us] = available
        return True

    # 功能：按同一明确时钟域的原始曝光时刻配对，不以相近接收时间替代拍摄时间。
    # 输入：timestamp_us：图像源时刻；clock_domain：图像时钟身份。
    #   available_at_monotonic：消费时刻；maximum_*：配对预算。
    # 输出：已实际到达的最近完整消息；跨重置、过期、无效质量或异域均不能配对。
    def align(
        self,
        *,
        timestamp_us,
        clock_domain,
        available_at_monotonic,
        maximum_skew_us=20_000,
        maximum_receive_age_seconds=0.25,
    ):
        _integer(timestamp_us, 0, 2**64 - 1)
        _integer(maximum_skew_us, 0, 50_000)
        now = _real(available_at_monotonic)
        age = _real(maximum_receive_age_seconds)
        if now < 0 or not 0 < age <= 0.25 or clock_domain != self.clock_domain:
            raise ValueError("NATIVE_ODOMETRY_ALIGNMENT_CONTRACT_INVALID")
        # A reset empties the preceding segment. Do not extrapolate a new
        # segment backwards across its first source timestamp.
        # A caller can inspect an earlier consumption instant. Future packets
        # must not establish the upper bracket, even if the chosen packet is old.
        visible = [
            packet for packet in self._history if self._available[packet.timestamp_us] <= now
        ]
        if (
            not visible
            or visible[-1].quality == -1
            or not visible[0].timestamp_us <= timestamp_us <= visible[-1].timestamp_us
        ):
            raise ValueError("NATIVE_ODOMETRY_SOURCE_TIME_NOT_PAIRED")
        candidates = [
            packet
            for packet in visible
            if 0 <= now - packet.received_monotonic_seconds <= age
            and abs(packet.timestamp_us - timestamp_us) <= maximum_skew_us
        ]
        if not candidates:
            raise ValueError("NATIVE_ODOMETRY_SOURCE_TIME_NOT_PAIRED")
        selected = min(
            candidates,
            key=lambda packet: (abs(packet.timestamp_us - timestamp_us), packet.timestamp_us),
        )
        if selected.quality == -1:
            raise ValueError("NATIVE_ODOMETRY_SOURCE_TIME_NOT_PAIRED")
        return selected
