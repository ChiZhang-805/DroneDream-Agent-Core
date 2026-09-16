"""Synthetic unit observations; these tests are not flight qualification."""

import math

import pytest
from control_fixtures import complete_feature_snapshot

from dronedream_agent_core.contracts import QuaternionWxyz
from dronedream_agent_core.control_feature_contract import (
    CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    GRAVITY_MPS2,
    encoder_contract_sha256,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.native_flight_state import native_flight_state_sample
from dronedream_agent_core.realtime_feature_encoders import FlightStateEncoder


# 功能：
#   创建固定地图绑定、位置速度和静止 IMU 的独立原生遥测夹具。
# 输入：
#   timestamp：身份及缓存汇集的 UNIX 毫秒时刻。
# 输出：
#   payload：包含原生设备时间及源年龄的遥测字典。
def _identity(timestamp: int = 1_000) -> dict:
    payload = {
        "schema_version": "dronedream.px4-identity-telemetry.v1",
        "updated_at_unix_ms": timestamp,
        "position_received_at_unix_ms": timestamp,
        "observed_position_ned_m": {"north_m": 0.0, "east_m": 0.0, "down_m": 0.0},
        "map_frame_binding": {
            "source": "deployment-coordinate-contract",
            "orientation": "NED-to-ENU-fixed",
            "collision_center_origin_world_enu_m": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "observed_velocity_ned_mps": {"north_m_s": 1.0, "east_m_s": 2.0, "down_m_s": -0.5},
        "dynamics": {
            "schema_version": "dronedream.px4-dynamics-telemetry.v1",
            "collected_at_unix_ms": timestamp,
            "sources": {
                "imu": {
                    "timestamp_us": 300_000,
                    "sample_age_seconds": 0.01,
                    "acceleration_forward_m_s2": 0.0,
                    "acceleration_right_m_s2": 0.0,
                    "acceleration_down_m_s2": -GRAVITY_MPS2,
                    "angular_velocity_forward_rad_s": 0.1,
                    "angular_velocity_right_rad_s": 0.2,
                    "angular_velocity_down_rad_s": 0.3,
                }
            },
        },
    }
    payload["dynamics"]["sources"]["attitude"] = {
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 90.0,
        "timestamp_us": 300_000,
        "sample_age_seconds": 0.0,
    }
    payload["map_frame_binding_sha256"] = sha256_json(payload["map_frame_binding"])
    return payload


# 功能：
#   为测试设定位置接收时间和可选极限俯仰，然后调用真实状态转换入口。
# 输入：
#   payload：测试遥测字典，为 None 时新建夹具。
#   now：消费 UNIX 毫秒时刻。
#   pose_time：位置观测接收时刻。
#   orientation：非空时启用本组测试约定的正 90 度 FLU 俯仰。
# 输出：
#   sample：生产转换函数生成的状态样本。
def _sample(
    payload: dict | None = None, *, now: int = 1_020, pose_time: int = 1_005,
    orientation: QuaternionWxyz | None = None,
):
    payload = _identity() if payload is None else payload
    payload["position_received_at_unix_ms"] = pose_time
    if orientation is not None:
        # This test's +90 degree FLU pitch is -90 degrees in native NED/FRD.
        payload["dynamics"]["sources"]["attitude"]["pitch_deg"] = -90.0
    sample = native_flight_state_sample(
        payload,
        now_unix_ms=now,
    )
    return sample


# 功能：
#   检查速度轴变换、重力扣除、角速度符号及未知协方差的有效掩码。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_imu_translation_rotation_gravity_and_missing_covariance() -> None:
    sample = _sample()
    assert sample.velocity_world_enu_mps.model_dump() == {"x": 2.0, "y": 1.0, "z": 0.5}
    assert sample.acceleration_body_mps2.model_dump() == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert sample.angular_velocity_body_rad_s.model_dump() == {"x": 0.1, "y": -0.2, "z": -0.3}
    assert sample.observed_at_unix_ms == 990  # Oldest input, not time of reading the file.
    assert sample.localization_covariance_m2 is None
    assert sample.source_sample_sha256 is not None
    result = FlightStateEncoder().encode(sample, encoded_at_unix_ms=1_020)
    assert result.valid_mask[7] == 0.0
    assert result.features[7] == 0.0
    assert result.valid_mask[8:14] == [1.0] * 6
    assert result.uncertainty == 1.0
    assert "FLIGHT_STATE_COVARIANCE_UNAVAILABLE" in result.issue_codes


# 功能：
#   倾斜机体的重力必须按姿态旋转扣除，不能固定只减垂直分量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_gravity_compensation_uses_tilt_not_a_fixed_vertical_subtraction() -> None:
    payload = _identity()
    imu = payload["dynamics"]["sources"]["imu"]
    imu["acceleration_forward_m_s2"] = -GRAVITY_MPS2
    imu["acceleration_down_m_s2"] = 0.0
    pitch = QuaternionWxyz(w=math.sqrt(0.5), x=0, y=math.sqrt(0.5), z=0)
    sample = _sample(payload, orientation=pitch)
    assert list(sample.acceleration_body_mps2.model_dump().values()) == pytest.approx([0, 0, 0])


# 功能：
#   原生协方差进入特征且保留自己的期限，不能借较新 IMU 延长年龄。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_estimator_covariance_is_used_with_its_original_deadline() -> None:
    from test_localization_evidence import identity

    payload = _identity()
    payload["dynamics"]["sources"]["odometry"] = identity()["dynamics"]["sources"]["odometry"]
    sample = _sample(payload)
    assert sample.localization_covariance_m2 == 0.03
    assert sample.observed_at_unix_ms == 950
    result = FlightStateEncoder().encode(sample, encoded_at_unix_ms=1020)
    assert result.valid_mask[7] == 1
    assert result.features[7] == pytest.approx(0.03 / 0.25)
    payload["dynamics"]["sources"]["odometry"]["sample_age_seconds"] = 0.5
    with pytest.raises(ValueError, match="LOCALIZATION_SAMPLE_EXPIRED"):
        _sample(payload)


# 功能：
#   非法 IMU 字段明确拒绝，不用零或默认值伪造有效测量。
# 输入：
#   field：被修改的 IMU 字段。
#   value：非法测量内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_age_seconds", -0.1),
        ("sample_age_seconds", 0.251),
        ("sample_age_seconds", float("nan")),
        ("timestamp_us", True),
        ("sample_age_seconds", 1e308),
        ("timestamp_us", 1.5),
        ("angular_velocity_down_rad_s", float("inf")),
        ("acceleration_forward_m_s2", None),
    ],
)
def test_invalid_native_sensor_values_are_not_replaced_by_zero(field, value) -> None:
    payload = _identity()
    payload["dynamics"]["sources"]["imu"][field] = value
    with pytest.raises(ValueError):
        _sample(payload)


# 功能：
#   身份、动力学和位置时间分别检查，不容许过期或未来来源。
# 输入：
#   source：被修改时间的来源类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["identity", "dynamics", "pose"])
def test_each_source_has_its_own_deadline_and_cannot_be_future_dated(source) -> None:
    for timestamp in (700, 1_021):
        payload = _identity()
        pose_time = 1_005
        if source == "identity":
            payload["updated_at_unix_ms"] = timestamp
        elif source == "dynamics":
            payload["dynamics"]["collected_at_unix_ms"] = timestamp
        else:
            pose_time = timestamp
        with pytest.raises(ValueError, match="(?i)expired|future"):
            _sample(payload, pose_time=pose_time)


# 功能：
#   身份文件和位置重新发布，不能掩盖原始 IMU 观测已过期。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_republishing_identity_does_not_rejuvenate_imu_and_pose() -> None:
    payload = _identity()
    payload["updated_at_unix_ms"] = 1_400
    with pytest.raises(ValueError, match="(?i)expired"):
        _sample(payload, now=1_410, pose_time=1_400)


# 功能：
#   必需传感器或速度字段缺失时停止转换，不使用旧缓存或常数补齐。
# 输入：
#   missing：删除的必需字段名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("missing", ["imu", "collected_at_unix_ms", "east_m_s"])
def test_missing_native_sources_fail_closed(missing) -> None:
    payload = _identity()
    if missing == "imu":
        del payload["dynamics"]["sources"][missing]
    elif missing == "collected_at_unix_ms":
        del payload["dynamics"][missing]
    else:
        del payload["observed_velocity_ned_mps"][missing]
    with pytest.raises(ValueError):
        _sample(payload)


# 功能：
#   历史时序配置改变必须改变特征契约摘要，缺少来源身份不能冒充规范输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_history_contract_is_explicit_and_changes_with_timing() -> None:
    sample = _sample()
    canonical = FlightStateEncoder().encode(sample, encoded_at_unix_ms=1_020)
    relaxed = FlightStateEncoder(history_length=4, maximum_gap_ms=5_000).encode(
        sample,
        encoded_at_unix_ms=1_020,
    )
    assert canonical.feature_contract_sha256 == encoder_contract_sha256("flight-state-encoder")
    assert canonical.feature_contract_sha256 != relaxed.feature_contract_sha256
    legacy = sample.model_copy(update={"source_sample_sha256": None})
    assert FlightStateEncoder().encode(legacy).feature_contract_sha256 is None
    assert complete_feature_snapshot().policy_feature_contract_sha256() == (
        CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    )


# 功能：
#   同长度但不同的原生测量必须产生不同内容身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_source_content_hash_changes_for_same_length_different_observations() -> None:
    original = _sample()
    payload = _identity()
    payload["dynamics"]["sources"]["imu"]["angular_velocity_down_rad_s"] = -0.3
    assert _sample(payload).source_sample_sha256 != original.source_sample_sha256


# 功能：
#   重复设备包不推进编码历史或期限，调用方修改返回特征不污染缓存。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_repeated_imu_packet_does_not_create_history_or_renew_deadline() -> None:
    encoder = FlightStateEncoder()
    first = encoder.encode(_sample(), encoded_at_unix_ms=1_020)
    # Caller mutation must not alter cached sensor evidence either.
    first.features[8] = 99.0
    repeated = encoder.encode(
        _sample(_identity(1_100), now=1_120, pose_time=1_105),
        encoded_at_unix_ms=1_120,
    )
    assert repeated.observed_at_unix_ms == 990
    assert repeated.encoded_at_unix_ms == 1_020
    assert repeated.features[8] == 0.0
    assert repeated.coverage == 1 / 16
    assert not repeated.fresh_at(1_241)


# 功能：
#   IMU 启动时钟回退后重新预热时序历史，不把两次启动拼成连续轨迹。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_imu_boot_clock_reset_starts_a_new_temporal_history() -> None:
    encoder = FlightStateEncoder()
    encoder.encode(_sample(), encoded_at_unix_ms=1_020)
    payload = _identity(1_100)
    payload["dynamics"]["sources"]["imu"]["timestamp_us"] = 5
    restarted = encoder.encode(
        _sample(payload, now=1_120, pose_time=1_105), encoded_at_unix_ms=1_120
    )
    assert restarted.coverage == 1 / 16
    assert "FLIGHT_STATE_IMU_CLOCK_RESET" in restarted.issue_codes
    assert "FLIGHT_STATE_HISTORY_WARMING" in restarted.issue_codes


# 功能：
#   各自未过期但彼此不同步的姿态和 IMU 不能合成虚假加速度。
# 输入：
#   source：人为偏移接收时间的来源。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source", ["imu", "attitude", "position"])
def test_individually_fresh_but_mismatched_state_cannot_generate_false_acceleration(source):
    payload = _identity()
    if source == "position":
        payload["position_received_at_unix_ms"] = 900
    else:
        payload["dynamics"]["sources"][source]["sample_age_seconds"] = .1
    with pytest.raises(ValueError, match="NOT_SYNCHRONIZED"):
        native_flight_state_sample(payload, now_unix_ms=1020)


# 功能：
#   用独立比力公式验证多个静止倾斜姿态，经重力补偿后都没有虚构线加速度。
# 输入：
#   roll：横滚角度。
#   pitch：俯仰角度。
#   yaw：偏航角度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("roll,pitch,yaw", [
    (0, 0, 0), (20, 30, 170), (-45, -60, -150), (90, 0, 90), (0, 89, 45),
])
def test_stationary_tilted_imu_has_no_invented_linear_acceleration(roll, pitch, yaw):
    # Independent NED/FRD specific-force formula, not a round trip through
    # the production quaternion conversion.
    r, p = math.radians(roll), math.radians(pitch)
    payload = _identity()
    payload["dynamics"]["sources"]["attitude"].update(
        roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
    payload["dynamics"]["sources"]["imu"].update(
        acceleration_forward_m_s2=GRAVITY_MPS2*math.sin(p),
        acceleration_right_m_s2=-GRAVITY_MPS2*math.sin(r)*math.cos(p),
        acceleration_down_m_s2=-GRAVITY_MPS2*math.cos(r)*math.cos(p),
    )
    result = native_flight_state_sample(payload, now_unix_ms=1020)
    assert list(result.acceleration_body_mps2.model_dump().values()) == pytest.approx([0]*3)


# 功能：
#   顶层状态或年龄参数损坏时，以明确领域错误拒绝，不能泄漏属性及比较异常。
# 输入：
#   payload：原生状态包或非法类型。
#   maximum_age：候选最大年龄毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload,maximum_age", [(None, 250), ([], 250), ({}, None)])
def test_native_sample_boundary_rejects_invalid_objects_and_age(payload, maximum_age):
    with pytest.raises(ValueError):
        native_flight_state_sample(payload, now_unix_ms=1020, maximum_age_ms=maximum_age)
