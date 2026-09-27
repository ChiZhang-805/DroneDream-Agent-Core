"""Source-bound native scan entry, including clock and frame fault injection."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_local_pose_alignment import scene
from test_native_odometry_source import packet

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    QuaternionWxyz,
    RawMetricRangeScan,
    RawRangeSample,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.map_pose_transport import map_pose_transport_candidate
from dronedream_agent_core.native_odometry_source import source_pose_in_map
from dronedream_agent_core.temporal_map_pose import TemporalMapPoseTracker


# 功能：建立原始 MAVLink→地图系扫描的完整解析夹具，不给定位器独立真值。
# 输入：无。
# 输出：连续定位器及所有来源一致的入参。
def inputs():
    native = packet(x=0., y=0., z=0., q=[math.sqrt(.5), 0., 0., math.sqrt(.5)])
    binding = dict(orientation="NED-to-ENU-fixed", source="deployment-coordinate-contract",
                   collision_center_origin_world_enu_m=dict(x=.04, y=-.03, z=.02))
    digest = sha256_json(binding)
    position, orientation, velocity = source_pose_in_map(
        native, binding=binding, binding_sha256=digest)
    mount = CalibratedRangeSensorMount(sensor_id="depth",
        translation_body_m=Vector3(x=0., y=0., z=0.),
        orientation_body_from_sensor=QuaternionWxyz(w=1., x=0., y=0., z=0.),
        minimum_range_m=.1, maximum_range_m=20.)
    points, index = scene()
    samples = [RawRangeSample(direction_sensor=Vector3(**dict(zip("xyz", point, strict=True))),
                              range_m=float(np.linalg.norm(point)), hit=True, confidence=1.)
               for point in points]
    scan = RawMetricRangeScan(sensor_id="depth", sequence=1, observed_at_unix_ms=10000,
        observed_at_monotonic_seconds=10., body_position_world_enu_m=position,
        body_orientation_world_from_body=orientation, body_velocity_world_enu_mps=velocity,
        localization_covariance_m2=.03, samples=samples)
    tracker = TemporalMapPoseTracker(map_sha256="a" * 64, binding_sha256=digest,
                                     clock_domain="test-native")
    return tracker, dict(packet=native, scan=scan, mount=mount, index=index,
        binding=binding, binding_sha256=digest, image_timestamp_ns=1_000_000_000,
        now_monotonic=10.05, map_sha256="a" * 64, clock_domain="test-native")


# 功能：验证原始扫描接入连续配准后仍可通过统一 NED/FRD 编码器，且不改原始协方差。
# 输入：真实协议形状的解析夹具。
# 输出：正确地图修正及保留原始采样时刻的传输候选，无飞行权限。
def test_native_scan_to_temporal_to_transport():
    tracker, args = inputs()
    before = args["scan"].model_dump()
    result = tracker.update_native(**args)
    assert result.fit.usable_candidate
    np.testing.assert_allclose(result.fit.correction_world_m, [-.04, .03, -.02], atol=1e-6)
    transported = map_pose_transport_candidate(packet=args["packet"], fit=result.fit,
        binding=args["binding"], binding_sha256=args["binding_sha256"],
        image_timestamp_ns=args["image_timestamp_ns"])
    assert transported.source_timestamp_us == 1_000_000
    assert not transported.covariance_qualified
    assert args["scan"].model_dump() == before


# 功能：对真实入口逐项注入来源、时间、位置、姿态、速度错误，不允许错配扫描参与历史。
# 输入：变体名称。
# 输出：当前输入被拒绝且已有历史失效，不能悄悄忽略损坏继续发布。
@pytest.mark.parametrize("case", ["future", "skew", "fractional", "bool",
                                  "position", "orientation", "velocity", "binding",
                                  "sample-budget"])
def test_native_input_faults_invalidate_tracker(case):
    tracker, args = inputs()
    if case == "future":
        args["now_monotonic"] = 9.9
    elif case == "skew":
        args["image_timestamp_ns"] += 21_000_000
    elif case == "fractional":
        args["image_timestamp_ns"] += 1
    elif case == "bool":
        args["image_timestamp_ns"] = True
    elif case == "position":
        args["scan"].body_position_world_enu_m.x += .001
    elif case == "orientation":
        args["scan"].body_orientation_world_from_body = QuaternionWxyz(w=0., x=0., y=0., z=1.)
    elif case == "velocity":
        args["scan"].body_velocity_world_enu_mps.x += .001
    elif case == "binding":
        args["binding"]["collision_center_origin_world_enu_m"]["x"] += 1.
    else:
        args["scan"].samples = args["scan"].samples * 3
    with pytest.raises(ValueError):
        tracker.update_native(**args)
    _, valid = inputs()
    with pytest.raises(ValueError, match="TRACKER_INVALIDATED"):
        tracker.update_native(**valid)


# 功能：迟到观测只退役旧修正，后续新帧可以恢复；不续期、不沿用过期位姿。
# 输入：正常帧、迟到帧、时间递增的新帧。
# 输出：迟到帧拒绝，新帧独立冷启动，原有精度和身份检查不变。
def test_expired_native_frame_retires_history_but_new_frame_recovers():
    tracker, args = inputs()
    tracker.update_native(**args)
    args["now_monotonic"] = 10.251
    with pytest.raises(ValueError, match="NATIVE_SOURCE_EXPIRED"):
        tracker.update_native(**args)
    args["packet"] = replace(args["packet"], timestamp_us=1_050_000,
                             received_monotonic_seconds=10.3)
    args["scan"].observed_at_monotonic_seconds = 10.3
    args["image_timestamp_ns"] = 1_050_000_000
    args["now_monotonic"] = 10.35
    result = tracker.update_native(**args)
    assert result.fit.usable_candidate
    assert not result.history_used
    assert result.previous_correction_timestamp_ns is None
    assert result.history_retired_reason == "NATIVE_SOURCE_EXPIRED"


# 功能：避免通过迟到时间绕过身份检查，损坏输入仍使整个来源实例停用。
# 输入：同时过期且地图身份变化的记录。
# 输出：身份错误优先，后续有效帧不能恢复损坏实例。
def test_expired_frame_cannot_hide_changed_map_identity():
    tracker, args = inputs()
    args["now_monotonic"] = 10.251
    args["map_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        tracker.update_native(**args)
    _, valid = inputs()
    with pytest.raises(ValueError, match="TRACKER_INVALIDATED"):
        tracker.update_native(**valid)


# 功能：拒绝同一时钟下切换发布实体或相机标定，以免继承另一无人机的修正历史。
# 输入：改变实体编号或相机外参。
# 输出：身份改变报错，不作为估计器重置重新接受。
@pytest.mark.parametrize("case", ["entity", "mount"])
def test_native_identity_cannot_change_inside_history(case):
    tracker, args = inputs()
    tracker.update_native(**args)
    args["packet"] = replace(args["packet"], timestamp_us=1_050_000)
    args["image_timestamp_ns"] = 1_050_000_000
    if case == "entity":
        args["packet"] = replace(args["packet"], system_id=2)
    else:
        args["mount"].translation_body_m.x = .1
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        tracker.update_native(**args)
