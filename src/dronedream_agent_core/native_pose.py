"""Estimator-only map pose with a fixed, explicitly declared frame binding.

The map origin is deployment configuration, never a transform fitted against
the simulator's current ground truth. MAVSDK attitude is NED-from-body-FRD;
consumers use ENU-from-body-FLU. Receive times are preserved, not capture times.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import QuaternionWxyz, Vector3
from .hashing import sha256_json
from .source_clock import source_received_at_unix_ms


# 功能：
#   读取有限数值字段，拒绝布尔值、字符串以及转换为浮点数会溢出的整数。
# 输入：
#   payload：含遥测字段的对象。
#   key：要读取的字段名。
# 输出：
#   number：通过检查的浮点数。
def finite_number(payload: Mapping[str, object], key: str) -> float:
    if not isinstance(payload, Mapping):
        raise ValueError("NATIVE_STATE_OBJECT_INVALID")
    value = payload.get(key)
    if (type(value) not in (int, float)
            or not -sys.float_info.max <= value <= sys.float_info.max):
        raise ValueError(f"native state requires a finite {key}")
    number = float(value)
    return number


# 功能：
#   读取非负有符号 64 位来源时刻，保留原字段单位，不以当前时刻替代缺失值。
# 输入：
#   payload：含来源时刻的对象。
#   key：明确单位的时间字段名。
# 输出：
#   value：原始整数时刻。
def source_timestamp(payload: Mapping[str, object], key: str) -> int:
    if not isinstance(payload, Mapping):
        raise ValueError("NATIVE_STATE_OBJECT_INVALID")
    value = payload.get(key)
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"native state requires a nonnegative integer {key}")
    return value


# 功能：
#   提取必需的遥测子对象，缺失或损坏时拒绝，不自动生成空对象。
# 输入：
#   payload：父级遥测对象。
#   key：子对象字段名。
# 输出：
#   value：校验后的子对象引用。
def required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("NATIVE_STATE_OBJECT_INVALID")
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"native state is missing {key}")
    return value


# 功能：
#   按顺序合成两个标量在前的四元数，不混淆世界基变换与机体系变换。
# 输入：
#   a：左乘旋转的 w、x、y、z 分量。
#   b：右乘旋转的 w、x、y、z 分量。
# 输出：
#   product：有序 Hamilton 四元数乘积。
def _multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    w, x, y, z = a
    v, i, j, k = b
    product = (
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    )
    return product


# 功能：
#   将 NED 世界、FRD 机体系的 ZYX 欧拉姿态转为 ENU 世界、FLU 机体系的姿态。
# 输入：
#   roll_deg：原生横滚角，单位为度。
#   pitch_deg：原生俯仰角，单位为度。
#   yaw_deg：原生偏航角，单位为度。
# 输出：
#   orientation：从 FLU 机体系到 ENU 世界系的单位四元数。
def native_attitude_enu_flu(roll_deg: float, pitch_deg: float, yaw_deg: float) -> QuaternionWxyz:
    if not all(type(v) in (float, int) and -sys.float_info.max <= v <= sys.float_info.max
               for v in (roll_deg, pitch_deg, yaw_deg)):
        raise ValueError("NATIVE_ATTITUDE_NONFINITE")
    if abs(roll_deg) > 180 or abs(pitch_deg) > 90 or abs(yaw_deg) > 360:
        raise ValueError("NATIVE_ATTITUDE_OUT_OF_RANGE")
    r, p, y = (math.radians(v) / 2 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    ned_from_frd = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    q = _multiply(_multiply((0, math.sqrt(0.5), math.sqrt(0.5), 0), ned_from_frd), (0, 1, 0, 0))
    # 左侧改变世界坐标基，右侧改变机体坐标基，缺少任一侧都会改变物理方向。
    # 统一标量符号避免 q 与 -q 两种等价写法造成虚假的姿态跳变。
    if q[0] < 0:
        q = tuple(-v for v in q)
    orientation = QuaternionWxyz(w=q[0], x=q[1], y=q[2], z=q[3])
    return orientation


@dataclass(frozen=True)
class NativeMapPose:
    """Estimator-derived ENU/FLU state with original receive times and fixed-frame identity."""
    position_world_enu_m: Vector3
    orientation_world_from_body: QuaternionWxyz
    velocity_world_enu_mps: Vector3
    observed_at_unix_ms: int
    binding_sha256: str
    position_received_at_unix_ms: int
    attitude_received_at_unix_ms: int


# 功能：
#   1. 应用固定部署原点及 NED 到 ENU 的变换，不消费仿真真值位置或姿态。
#   2. 按最旧来源执行年龄门控，分别保留位置与姿态接收时刻供后续对齐。
# 输入：
#   identity：包含部署绑定、原生位置速度和姿态的遥测对象。
#   now_unix_ms：当前消费 UNIX 毫秒时刻。
#   maximum_age_ms：允许的最大年龄，不能超过 250 毫秒。
# 输出：
#   pose：包含地图位姿、速度、来源时刻及绑定摘要的原生状态。
def native_map_pose(
    identity: Mapping[str, object], *, now_unix_ms: int, maximum_age_ms: int = 250
) -> NativeMapPose:
    if type(maximum_age_ms) is not int or not 0 < maximum_age_ms <= 250:
        raise ValueError("NATIVE_POSE_AGE_LIMIT_INVALID")
    if type(now_unix_ms) is not int or not 0 <= now_unix_ms < 2**63:
        raise ValueError("NATIVE_POSE_CONSUMER_TIME_INVALID")
    # 使用同一小型 JSON 快照校验和计算摘要，拒绝键字符串化及循环结构。
    binding = copy_json(dict(required_mapping(identity, "map_frame_binding")), limit=16_384)
    if (
        binding.get("orientation") != "NED-to-ENU-fixed"
        or binding.get("source") != "deployment-coordinate-contract"
    ):
        raise ValueError("NATIVE_MAP_FRAME_BINDING_UNSUPPORTED")
    # 任意偏航局部系不能冒充北向 NED；部署坐标也不能依赖 Pydantic 的宽松转换。
    raw_origin = required_mapping(binding, "collision_center_origin_world_enu_m")
    if set(raw_origin) != {"x", "y", "z"}:
        raise ValueError("NATIVE_MAP_ORIGIN_INVALID")
    origin = Vector3(**{key: finite_number(raw_origin, key) for key in ("x", "y", "z")})
    binding_hash = sha256_json(binding)
    if identity.get("map_frame_binding_sha256") != binding_hash:
        raise ValueError("NATIVE_MAP_FRAME_BINDING_HASH_MISMATCH")
    position = required_mapping(identity, "observed_position_ned_m")
    velocity = required_mapping(identity, "observed_velocity_ned_mps")
    dynamics = required_mapping(identity, "dynamics")
    attitude = required_mapping(required_mapping(dynamics, "sources"), "attitude")
    age = finite_number(attitude, "sample_age_seconds")
    if age < 0 or age > maximum_age_ms / 1000:
        raise ValueError("NATIVE_ATTITUDE_EXPIRED")
    source_timestamp(attitude, "timestamp_us")
    collected = source_timestamp(dynamics, "collected_at_unix_ms")
    attitude_time = source_received_at_unix_ms(
        attitude, collected_at_unix_ms=collected, now_unix_ms=now_unix_ms,
        maximum_age_ms=maximum_age_ms,
    )
    times = (
        source_timestamp(identity, "updated_at_unix_ms"),
        collected,
        source_timestamp(identity, "position_received_at_unix_ms"),
        attitude_time,
    )
    if min(times) < 0 or max(times) > now_unix_ms:
        raise ValueError("native pose has a future or invalid source time")
    if now_unix_ms - min(times) > maximum_age_ms:
        raise ValueError("native pose source has expired")
    pose = NativeMapPose(
        position_world_enu_m=Vector3(
            x=origin.x + finite_number(position, "east_m"),
            y=origin.y + finite_number(position, "north_m"),
            z=origin.z - finite_number(position, "down_m"),
        ),
        orientation_world_from_body=native_attitude_enu_flu(
            *(finite_number(attitude, key) for key in ("roll_deg", "pitch_deg", "yaw_deg"))
        ),
        velocity_world_enu_mps=Vector3(
            x=finite_number(velocity, "east_m_s"),
            y=finite_number(velocity, "north_m_s"),
            z=-finite_number(velocity, "down_m_s"),
        ),
        observed_at_unix_ms=min(times),
        binding_sha256=binding_hash,
        position_received_at_unix_ms=source_timestamp(identity, "position_received_at_unix_ms"),
        attitude_received_at_unix_ms=attitude_time,
    )
    return pose
