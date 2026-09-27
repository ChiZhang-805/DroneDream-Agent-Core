"""Raw source-frame and reset boundaries; no simulated flight qualification."""

import json
from dataclasses import replace

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.native_odometry_source import (
    SourceOdometryHistory,
    decode_source_odometry,
    source_pose_in_map,
)


# 功能：构造协议形状的测试消息，保留未知协方差交叉项。
# 输入：changes：仅用于各边界测试的字段变更。
# 输出：完整原始 ODOMETRY JSON 对象。
def fields(**changes):
    covariance = [None] * 21
    for index in (0, 6, 11, 15, 18, 20):
        covariance[index] = 0.01
    return (
        dict(
            message_id=331,
            message_name="ODOMETRY",
            frame_id=1,
            child_frame_id=1,
            time_usec=1_000_000,
            reset_counter=2,
            estimator_type=8,
            quality=0,
            x=1.0,
            y=2.0,
            z=-3.0,
            vx=0.1,
            vy=0.2,
            vz=0.3,
            rollspeed=0.01,
            pitchspeed=0.02,
            yawspeed=0.03,
            q=[1.0, 0.0, 0.0, 0.0],
            pose_covariance=covariance,
            velocity_covariance=covariance,
        )
        | changes
    )


# 功能：使用正式协议解析器生成测试消息，不绕过字段校验。
# 输入：received：模拟接收钟；changes：协议字段变更。
# 输出：已校验不可变消息。
def packet(received=10.0, **changes):
    return decode_source_odometry(
        json.dumps(fields(**changes)), system_id=1, component_id=1, received_monotonic=received
    )


# 功能：建立运行专属历史，避免测试默认为所有时间戳共享一个时钟。
# 输入：无。
# 输出：空历史对象。
def history():
    return SourceOdometryHistory(clock_domain="verified-test-run", system_id=1, component_id=1)


# 功能：验证原始协议编号和完整未知协方差被保留，不产生精度或同步声明。
# 输入：无。
# 输出：无；比较不可变字段与测试消息。
def test_raw_frame_id_is_not_the_legacy_sdk_enum():
    value = packet()
    assert value.frame_id == value.child_frame_id == 1  # MAV_FRAME_LOCAL_NED, not BODY_NED.
    assert value.reset_counter == 2 and value.timestamp_us == 1_000_000
    assert value.position_frame_m == (1.0, 2.0, -3.0)
    assert value.pose_covariance_upper[1] is None
    assert value.quality == 0  # Unknown quality must not turn into "good".


# 功能：覆盖时钟、实体、数值、协方差和姿态的畸形输入，防止宽松类型转换。
# 输入：changes：单个畸形协议字段。
# 输出：无；全部被协议解析拒绝。
@pytest.mark.parametrize(
    "changes",
    [
        {"time_usec": True},
        {"time_usec": -1},
        {"time_usec": 2**64},
        {"time_usec": 1.2},
        {"frame_id": "1"},
        {"frame_id": False},
        {"reset_counter": 256},
        {"quality": -2},
        {"quality": 101},
        {"x": "1"},
        {"x": float("nan")},
        {"z": float("inf")},
        {"q": [0.0, 0.0, 0.0, 0.0]},
        {"q": [1.0, 0.0, 0.0]},
        {"pose_covariance": [0.0] * 20},
        {"velocity_covariance": [-1.0] * 21},
        {"pose_covariance": [True] * 21},
    ],
)
def test_malformed_packets_are_rejected(changes):
    with pytest.raises(ValueError, match="NATIVE_ODOMETRY"):
        packet(**changes)


# 功能：拒绝重复键、错误消息及缺失扩展字段，不虚构重置计数。
# 输入：无。
# 输出：无；每种原始 JSON 错误均明确拒绝。
def test_json_envelope_is_bounded_and_unambiguous():
    for raw in ('{"x":1,"x":2}', "[]", "x" * 8193, json.dumps(fields(message_id=True))):
        with pytest.raises(ValueError, match="NATIVE_ODOMETRY"):
            decode_source_odometry(raw, system_id=1, component_id=1, received_monotonic=10.0)
    data = fields()
    del data["reset_counter"]
    with pytest.raises(ValueError, match="FIELD_MISSING"):
        decode_source_odometry(
            json.dumps(data), system_id=1, component_id=1, received_monotonic=10.0
        )


# 功能：验证接收较晚但源时刻较近的包被正确配对，不按到达时刻选择错误位姿。
# 输入：无。
# 输出：无；配对源时间为一秒，而不是接收时间最靠近的一点零二秒。
def test_alignment_uses_source_clock_not_receipt_order():
    buffer = history()
    first, second = packet(10.0), packet(10.02, time_usec=1_020_000, x=9.0)
    buffer.ingest(first)
    buffer.ingest(second)
    actual = buffer.align(
        timestamp_us=1_001_000, clock_domain="verified-test-run", available_at_monotonic=10.18
    )
    assert actual == first
    with pytest.raises(ValueError, match="CONTRACT"):
        buffer.align(
            timestamp_us=1_001_000, clock_domain="unrelated-clock", available_at_monotonic=10.18
        )
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_001_000, clock_domain="verified-test-run", available_at_monotonic=10.5
        )


# 功能：防止同一物理样本重发刷新接收期限；同时间戳内容变化使整个历史失效。
# 输入：无。
# 输出：无；旧包不续龄，冲突后必须建立新的运行历史。
def test_duplicate_does_not_refresh_and_conflict_invalidates():
    buffer = history()
    assert buffer.ingest(packet())
    assert not buffer.ingest(packet(received=10.1))
    assert buffer._history[0].received_monotonic_seconds == 10.0
    with pytest.raises(ValueError, match="CONFLICT"):
        buffer.ingest(packet(received=10.1, x=2.0))
    with pytest.raises(ValueError, match="INVALIDATED"):
        buffer.ingest(packet(received=10.2, time_usec=1_010_000))


# 功能：重置及坐标系切换清空旧段，包括八位重置计数回绕；禁止跨段配对。
# 输入：changes：重置或坐标变更。
# 输出：无；变更后的首帧之前没有可用配对。
@pytest.mark.parametrize(
    "changes",
    [
        dict(reset_counter=3),
        dict(reset_counter=0),
        dict(frame_id=20),
        dict(child_frame_id=12),
        dict(estimator_type=2),
    ],
)
def test_reset_or_frame_change_cannot_reuse_old_segment(changes):
    buffer = history()
    buffer.ingest(packet())
    buffer.ingest(packet(10.02, time_usec=1_020_000, **changes))
    assert buffer.reset_count == 1 and len(buffer._history) == 1
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_010_000, clock_domain="verified-test-run", available_at_monotonic=10.03
        )


# 功能：回退时钟或不同实体不能悄悄进入同一历史；直接构造消息也不能绕过校验。
# 输入：无。
# 输出：无；源错误永久撤销历史，非法替换立即失败。
def test_clock_and_identity_changes_fail_closed():
    for value in (
        packet(10.02, time_usec=999_999),
        packet(9.9, time_usec=1_010_000),
        replace(packet(10.02, time_usec=1_010_000), system_id=2),
    ):
        buffer = history()
        buffer.ingest(packet())
        with pytest.raises(ValueError, match="REGRESSED|SOURCE_CHANGED"):
            buffer.ingest(value)
        assert not buffer._history
    with pytest.raises(ValueError, match="INTEGER"):
        replace(packet(), timestamp_us=True)


# 功能：验证机体系速度必须先旋转到 NED，再转地图 ENU；位置只应用固定部署原点。
# 输入：无。
# 输出：无；相同物理速度的 NED 与 FRD 表示产生相同结果。
def test_explicit_world_and_body_velocity_frames_convert_consistently():
    binding = {
        "orientation": "NED-to-ENU-fixed",
        "source": "deployment-coordinate-contract",
        "collision_center_origin_world_enu_m": {"x": 10.0, "y": 20.0, "z": 30.0},
    }
    quaternion = [2**-0.5, 0.0, 0.0, 2**-0.5]  # Body forward points east.
    ned = packet(q=quaternion, vx=0.0, vy=1.0, vz=0.0)
    body = packet(q=quaternion, child_frame_id=12, vx=1.0, vy=0.0, vz=0.0)
    first = source_pose_in_map(ned, binding=binding, binding_sha256=sha256_json(binding))
    second = source_pose_in_map(body, binding=binding, binding_sha256=sha256_json(binding))
    assert first[0].model_dump() == {"x": 12.0, "y": 21.0, "z": 33.0}
    assert first[2].model_dump() == pytest.approx(second[2].model_dump(), abs=1e-12)
    assert first[1].model_dump() == pytest.approx({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    for invalid in (
        replace(ned, frame_id=20),
        replace(ned, frame_id=8),
        replace(ned, quality=-1),
        replace(ned, estimator_type=2),
    ):
        with pytest.raises(ValueError, match="FRAME_UNSUPPORTED"):
            source_pose_in_map(invalid, binding=binding, binding_sha256=sha256_json(binding))
    with pytest.raises(ValueError, match="BINDING_INVALID"):
        source_pose_in_map(ned, binding=binding, binding_sha256="0" * 64)


# 功能：未来未到达或明确质量失败的样本不得参与配对，容量不能无限增长。
# 输入：无。
# 输出：无；边界返回错误且历史保持上限。
def test_future_invalid_quality_and_capacity():
    buffer = SourceOdometryHistory(
        clock_domain="verified-test-run", system_id=1, component_id=1, capacity=2
    )
    for index in range(4):
        buffer.ingest(packet(10.0 + index * 0.02, time_usec=1_000_000 + index * 20_000, quality=-1))
    assert len(buffer._history) == 2
    for now in (9.0, 10.07):
        with pytest.raises(ValueError, match="NOT_PAIRED"):
            buffer.align(
                timestamp_us=1_040_000, clock_domain="verified-test-run", available_at_monotonic=now
            )


# 功能：未来消息不能建立配对上界，最近的故障质量消息也不能被较旧好消息遮盖。
# 输入：无；构造延迟到达和质量失败的真实协议形状数据。
# 输出：无；两种错误均被拒绝，不外推源位姿。
def test_unreceived_upper_bracket_and_failed_nearest_packet_are_rejected():
    buffer = history()
    buffer.ingest(packet())
    buffer.ingest(packet(10.1, time_usec=1_020_000))
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_010_000, clock_domain="verified-test-run", available_at_monotonic=10.05
        )
    buffer.ingest(packet(10.12, time_usec=1_040_000, quality=-1))
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_020_000, clock_domain="verified-test-run", available_at_monotonic=10.13
        )
    buffer.ingest(packet(10.14, time_usec=1_060_000))
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_039_000, clock_domain="verified-test-run", available_at_monotonic=10.15
        )


# 功能：区分传输收到和消费者读到，迟发布的上界不能参与过去的决策，重发也不续龄。
# 输入：无。
# 输出：无；首次可用时刻和原始接收时刻分别受检查。
def test_consumer_availability_is_not_transport_receipt_time():
    buffer = history()
    first = packet()
    buffer.ingest(first, available_at_monotonic=10.02)
    buffer.ingest(packet(10.04, time_usec=1_020_000), available_at_monotonic=10.10)
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        buffer.align(
            timestamp_us=1_010_000, clock_domain="verified-test-run", available_at_monotonic=10.08
        )
    assert (
        buffer.align(
            timestamp_us=1_010_000, clock_domain="verified-test-run", available_at_monotonic=10.11
        )
        == first
    )
    assert not buffer.ingest(packet(10.14, time_usec=1_020_000), available_at_monotonic=10.15)
    assert buffer._available[1_020_000] == 10.10
    with pytest.raises(ValueError, match="AVAILABILITY_BEFORE_RECEIPT"):
        history().ingest(first, available_at_monotonic=9.0)
