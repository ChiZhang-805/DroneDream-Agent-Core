"""Validate native PX4 state and translate sensor units at one boundary.

No file I/O and no controller commands live here. The caller must first validate
the PX4/Gazebo identity binding. Receive timestamps are not sensor capture times:
PX4 boot-clock timestamps are retained as provenance, not interpreted as UTC.
"""

from collections.abc import Mapping

from .contracts import Vector3
from .control_feature_contract import (
    FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS,
    FLIGHT_STATE_MAXIMUM_GAP_MS,
    GRAVITY_MPS2,
)
from .hashing import sha256_json
from .localization_evidence import native_localization_evidence
from .native_pose import (
    finite_number as _number,
)
from .native_pose import (
    native_map_pose,
)
from .native_pose import (
    required_mapping as _mapping,
)
from .native_pose import (
    source_timestamp as _timestamp,
)
from .realtime_feature_encoders import FlightStateSample, world_enu_to_body
from .source_clock import source_received_at_unix_ms


# 功能：
#   1. 汇集真实位姿、IMU 和定位不确定性，核对各来源有效期及位置姿态与 IMU 的时差。
#   2. 平移量使用公共前右上向量，角速度保持右手前左上；按当前姿态扣除重力。
#   3. 保留原始设备整数时间和内容身份，缓存重新发布不能制造新的观测。
# 输入：
#   identity：已完成设备与部署身份绑定检查的原生遥测对象。
#   now_unix_ms：当前消费 UNIX 毫秒时刻。
#   maximum_age_ms：最大允许年龄毫秒数，受实时控制契约限制。
# 输出：
#   sample：带来源摘要、速度、线加速度、角速度和可选协方差的状态样本。
def native_flight_state_sample(
    identity: Mapping[str, object], *, now_unix_ms: int,
    maximum_age_ms: int = FLIGHT_STATE_MAXIMUM_GAP_MS,
) -> FlightStateSample:
    if (type(maximum_age_ms) is not int
            or not 10 <= maximum_age_ms <= FLIGHT_STATE_MAXIMUM_GAP_MS):
        raise ValueError("native state age limit exceeds the control contract")
    if not isinstance(identity, Mapping):
        raise ValueError("native state identity must be an object")
    if identity.get("schema_version") != "dronedream.px4-identity-telemetry.v1":
        raise ValueError("native state identity schema is missing or incompatible")
    native_pose = native_map_pose(identity, now_unix_ms=now_unix_ms, maximum_age_ms=maximum_age_ms)
    orientation_world_from_body = native_pose.orientation_world_from_body
    identity_time = _timestamp(identity, "updated_at_unix_ms")
    dynamics = _mapping(identity, "dynamics")
    if dynamics.get("schema_version") != "dronedream.px4-dynamics-telemetry.v1":
        raise ValueError("native dynamics schema is missing or incompatible")
    # 年龄锚定缓存汇集时刻，不是稍后重写身份文件的时刻。
    dynamics_time = _timestamp(dynamics, "collected_at_unix_ms")
    imu = _mapping(_mapping(dynamics, "sources"), "imu")
    age_seconds = _number(imu, "sample_age_seconds")
    if age_seconds < 0:
        raise ValueError("native IMU sample age cannot be negative")
    if age_seconds > maximum_age_ms / 1_000:
        # 换算毫秒前先限制范围，有限但畸大的浮点数也可能在乘法后溢出。
        raise ValueError("native IMU source has expired")
    imu_time = source_received_at_unix_ms(
        imu, collected_at_unix_ms=dynamics_time, now_unix_ms=now_unix_ms,
        maximum_age_ms=maximum_age_ms,
    )
    source_times = (identity_time, dynamics_time, native_pose.observed_at_unix_ms, imu_time)
    if any(timestamp > now_unix_ms or timestamp < 0 for timestamp in source_times):
        raise ValueError("native state contains a future or invalid source timestamp")
    observed_at = min(source_times)
    if now_unix_ms - observed_at > maximum_age_ms:
        raise ValueError("native state source has expired")
    measurement_times = (native_pose.position_received_at_unix_ms,
                         native_pose.attitude_received_at_unix_ms, imu_time)
    if max(measurement_times) - min(measurement_times) > FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS:
        raise ValueError("NATIVE_POSE_IMU_NOT_SYNCHRONIZED")
    covariance, localization_time = native_localization_evidence(
        identity, now_unix_ms=now_unix_ms, maximum_age_ms=maximum_age_ms,
    )
    if localization_time is not None:
        observed_at = min(observed_at, localization_time)
    boot_timestamp_us = _timestamp(imu, "timestamp_us")
    # 设备序列时间不经 float64 转换，避免 2**53 以上不同包被舍入成同一身份。
    velocity = _mapping(identity, "observed_velocity_ned_mps")
    gravity_body = world_enu_to_body(
        orientation_world_from_body, Vector3(x=0.0, y=0.0, z=-GRAVITY_MPS2)
    )
    acceleration = Vector3(
        x=_number(imu, "acceleration_forward_m_s2") + gravity_body.x,
        y=_number(imu, "acceleration_right_m_s2") + gravity_body.y,
        z=-_number(imu, "acceleration_down_m_s2") + gravity_body.z,
    )
    gyro = Vector3(
        x=_number(imu, "angular_velocity_forward_rad_s"),
        y=-_number(imu, "angular_velocity_right_rad_s"),
        z=-_number(imu, "angular_velocity_down_rad_s"),
    )
    sample = FlightStateSample(
        source_id=f"px4-native-estimator-{native_pose.binding_sha256}",
        imu_timestamp_us=boot_timestamp_us,
        observed_at_unix_ms=observed_at,
        orientation_world_from_body=orientation_world_from_body,
        velocity_world_enu_mps=Vector3(
            x=_number(velocity, "east_m_s"), y=_number(velocity, "north_m_s"),
            z=-_number(velocity, "down_m_s"),
        ),
        acceleration_body_mps2=acceleration,
        angular_velocity_body_rad_s=gyro,
        localization_covariance_m2=covariance,
        source_sample_sha256=sha256_json({
            "map_binding": native_pose.binding_sha256,
            "position": native_pose.position_world_enu_m.model_dump(mode="json"),
            "velocity": native_pose.velocity_world_enu_mps.model_dump(mode="json"),
            "pose": orientation_world_from_body.model_dump(mode="json"),
            "pose_received_at_unix_ms": native_pose.observed_at_unix_ms,
            "imu_received_at_unix_ms": imu_time,
            "imu": {key: value for key, value in imu.items()
                    if key not in {"sample_age_seconds", "received_at_unix_ms"}},
            "localization": {"variance_bound_m2": covariance, "received_at": localization_time},
            "imu_boot_timestamp_us": boot_timestamp_us,
        }),
    )
    return sample
