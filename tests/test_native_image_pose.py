import json

import pytest
from test_native_odometry_source import fields

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.localization_evidence import position_variance_bound_m2
from dronedream_agent_core.native_image_pose import NativeImagePoseBuffer
from dronedream_agent_core.native_odometry_source import (
    canonical_odometry_fields,
    decode_source_odometry,
)


# 功能：构造不同设备时钟和主机时钟的完整遥测，避免测试仅靠时间数值巧合通过。
# 输入：stamp：设备微秒；received：主机单调秒；milliseconds：主机 UNIX 毫秒；changes：原始字段。
# 输出：独立遥测对象。
def payload(stamp=1_000_000, received=10.0, milliseconds=1000, **changes):
    binding = {
        "orientation": "NED-to-ENU-fixed",
        "source": "deployment-coordinate-contract",
        "collision_center_origin_world_enu_m": {"x": 10.0, "y": 20.0, "z": 30.0},
    }
    wire = fields(time_usec=stamp, **changes)
    return {
        "map_frame_binding": binding,
        "map_frame_binding_sha256": sha256_json(binding),
        "dynamics": {
            "sources": {
                "odometry": {
                    **canonical_odometry_fields(decode_source_odometry(json.dumps(wire),
                        system_id=1, component_id=1, received_monotonic=received)),
                    "received_at_unix_ms": milliseconds,
                    "mavlink_source": {
                        "fields_json": json.dumps(wire),
                        "system_id": 1,
                        "component_id": 1,
                        "received_monotonic_seconds": received,
                    },
                }
            }
        },
    }


# 功能：建立含前后源时间边界的真实形状缓存。
# 输入：无。
# 输出：源时钟域已指定的缓存。
def buffer():
    result = NativeImagePoseBuffer(clock_domain="same-run")
    result.ingest(payload(), available_at_monotonic=10.01)
    result.ingest(payload(1_020_000, 10.02, 1020, x=9.0), available_at_monotonic=10.03)
    return result


# 功能：用延迟到达的图像进行配对，不将接收时间当作曝光时间。
# 输入：state：缓存；changes：待验证的边界参数。
# 输出：对齐结果，或者明确的拒绝。
def align(state, **changes):
    arguments = dict(
        image_timestamp_ns=1_001_000_000,
        clock_domain="same-run",
        image_received_unix_ms=1150,
        now_unix_ms=1160,
        now_monotonic=10.16,
        maximum_range_m=10.0,
        maximum_acceleration_mps2=1.0,
    )
    arguments.update(changes)
    return state.align(**arguments)


# 功能：位姿和协方差来自同一原始消息，保留接收期限，源时差只能增加方差。
# 输入：无。
# 输出：无；验证没有使用较新的错误位置，没有修改来源或降低原始不确定性。
def test_single_packet_pose_with_source_clock_and_original_deadline():
    result = align(buffer())
    assert result.pose.position_world_enu_m.model_dump() == {"x": 12.0, "y": 21.0, "z": 33.0}
    assert result.pose.observed_at_unix_ms == 1000
    assert result.pose.position_received_at_unix_ms == result.pose.attitude_received_at_unix_ms
    assert result.source_skew_us == -1000
    assert result.conservative_variance_m2 >= position_variance_bound_m2(
        fields()["pose_covariance"]
    )
    assert result.source_evidence["qualification_granted"] is False
    assert result.source_evidence["pose_extrapolated"] is False
    assert result.source_evidence["received_monotonic_seconds"] == 10.0
    assert json.loads(result.source_evidence["fields_json"])["time_usec"] == 1_000_000


# 功能：规范字段与原始消息不一致时停止使用缓存，不能让控制和地图定位各用一份证据。
# 输入：key、value：单个错配字段；包含布尔值冒充合法数值的情况。
# 输出：无；错配被拒绝，后续对齐也不能使用之前的好消息。
@pytest.mark.parametrize("key,value", [
    ("frame_id", "LOCAL_FRD"), ("child_frame_id", "BODY_FRD"),
    ("estimator_type", 7), ("quality", True),
    ("pose_covariance_upper_m2", [0.0] * 21),
    ("twist_covariance_upper", None),
    ("velocity_child_frame_m_s", [True, 0.0, 0.0]),
])
def test_canonical_and_source_must_agree(key, value):
    state = buffer()
    current = payload(1_040_000, 10.04, 1040)
    current["dynamics"]["sources"]["odometry"][key] = value
    with pytest.raises(ValueError, match="CANONICAL_FIELDS_MISMATCH"):
        state.ingest(current, available_at_monotonic=10.05)
    with pytest.raises(ValueError, match="HISTORY_UNAVAILABLE"):
        align(state)


# 功能：异域、过期、未来、错误数值和跨微秒的源时间不能静默退回旧的接收时间配对。
# 输入：changes：单个非法边界条件。
# 输出：无；所有候选明确拒绝。
@pytest.mark.parametrize(
    "changes",
    [
        {"clock_domain": "another-run"},
        {"image_timestamp_ns": True},
        {"image_timestamp_ns": 1_001_000_001},
        {"image_received_unix_ms": 900},
        {"image_received_unix_ms": 1170},
        {"now_unix_ms": 1300},
        {"now_monotonic": 10.3},
        {"maximum_range_m": float("inf")},
        {"maximum_acceleration_mps2": -1},
        {"image_timestamp_ns": 1_021_000_000},
    ],
)
def test_no_fallback_on_invalid_source_alignment(changes):
    with pytest.raises(ValueError):
        align(buffer(), **changes)


# 功能：定位重置隔离历史，禁止跨重置图像借用旧位姿。
# 输入：无。
# 输出：无；旧图像拒绝，重置后新图像仍可正常使用。
def test_reset_clears_pose_and_receipt_history():
    state = buffer()
    state.ingest(payload(1_040_000, 10.04, 1040, reset_counter=3), available_at_monotonic=10.05)
    assert len(state._records) == 1
    with pytest.raises(ValueError, match="NOT_PAIRED"):
        align(state)
    state.ingest(payload(1_060_000, 10.06, 1060, reset_counter=3), available_at_monotonic=10.07)
    assert align(state, image_timestamp_ns=1_041_000_000).source_evidence["reset_counter"] == 3


# 功能：新地图绑定或同包标识矛盾使缓存失效，不复用错误前的良好历史。
# 输入：kind：故障类型。
# 输出：无；接入和之后配对均拒绝。
@pytest.mark.parametrize("kind", ["binding", "timestamp", "reset", "missing", "frame"])
def test_broken_lineage_invalidates_cache(kind):
    state, value = buffer(), payload(1_040_000, 10.04, 1040)
    row = value["dynamics"]["sources"]["odometry"]
    if kind == "binding":
        value["map_frame_binding"]["collision_center_origin_world_enu_m"]["x"] = 9.0
        value["map_frame_binding_sha256"] = sha256_json(value["map_frame_binding"])
    elif kind == "timestamp":
        row["timestamp_us"] += 1
    elif kind == "reset":
        row["reset_counter"] += 1
    elif kind == "missing":
        del row["mavlink_source"]["received_monotonic_seconds"]
    else:
        raw = json.loads(row["mavlink_source"]["fields_json"])
        raw["frame_id"] = 20
        row["mavlink_source"]["fields_json"] = json.dumps(raw)
    with pytest.raises((ValueError, KeyError)):
        state.ingest(value, available_at_monotonic=10.05)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        align(state)


# 功能：重复消息不能刷新首次接收和可用时间；历史容量严格有界。
# 输入：无。
# 输出：无；维持旧截止时间且只保留最近 128 条独立消息。
def test_republication_does_not_refresh_and_storage_is_bounded():
    state = buffer()
    state.ingest(payload(1_020_000, 10.02, 1150, x=9.0), available_at_monotonic=10.15)
    assert state._records[1_020_000][0] == 1020
    for index in range(2, 140):
        state.ingest(
            payload(1_000_000 + index * 20_000, 10.0 + index * 0.02, 1000 + index * 20),
            available_at_monotonic=10.01 + index * 0.02,
        )
    assert len(state._records) == 128
