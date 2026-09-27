"""Offline moving-camera registration test, NOT native estimator qualification.

Input positions are deliberately corrupted independent fixture truth. This
isolates geometry/attitude/time sensitivity; it does not measure PX4 error or
authorize tighter covariance. Failed fits and unmatched images remain visible.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dronedream_plugin_sdk.protocol import decode_json

from .depth_projection import DepthProjectionCalibration, project_metric_depth_frame
from .geometry_motion_fixture import (
    bytes_digest,
    euler_quaternion,
    fixture_sensor_mount,
    sensor_hits_world,
)
from .local_map_alignment import fit_map_translation
from .local_pose_alignment import fit_map_pose
from .optical_map import compile_optical_map
from .plugin_files import read_plugin_file
from .simulation_sensor_frames import _rotation
from .temporal_map_pose import TemporalMapPoseTracker


# 功能：
#   读取有界普通文件并复核身份，拒绝链接、空文件及读取期间的替换或增长。
# 输入：
#   path：待读取的采集或地图文件。
#   maximum：允许的最大字节数。
# 输出：
#   data：通过边界检查的原始字节。
def _bounded(path, maximum):
    data = read_plugin_file(path, limit=maximum)
    if not data or len(data) > maximum:
        raise ValueError("MOTION_AUDIT_FILE_BUDGET_INVALID")
    return data


# 功能：
#   汇总调用方明确选择的有效误差或耗时样本，空集合保留为无统计值。
# 输入：
#   values：有限数值列表。
# 输出：
#   result：p50、p95、p99、max 对象；无样本时为 None。
def _percentiles(values):
    result = dict(zip(("p50", "p95", "p99", "max"),
        np.quantile(values, [.5, .95, .99, 1.]).tolist(), strict=True)) if values else None
    return result


# 功能：
#   按 wxyz 顺序合成姿态，用于向独立夹具真值施加已知局部姿态扰动。
# 输入：
#   a：世界从机体的四元数。
#   b：准备右乘的局部扰动四元数。
# 输出：
#   quaternion：先执行 b 再执行 a 的组合姿态。
def _quaternion_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    quaternion = [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                  w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]
    return quaternion


# 功能：
#   在预先校验的递增时间轴上二分查找最近姿态，拒绝外推及超过配对间隔的观测。
# 输入：
#   poses：与时间数组一一对应的已验证姿态序列。
#   times：按来源时间严格递增的一维整数数组，由批量入口一次校验。
#   target：目标来源时间，单位纳秒，负时间不外推。
#   maximum_gap_ns：允许的最大时间差，默认八毫秒。
# 输出：
#   result：匹配到的原姿态对象；没有匹配时为 None。
def nearest_source_pose(poses, times, target, *, maximum_gap_ns=8_000_000):
    if (type(target) is not int or type(maximum_gap_ns) is not int
            or not 0 <= maximum_gap_ns <= (1 << 63)-1 or len(poses) != len(times)):
        raise ValueError("MOTION_AUDIT_SOURCE_PAIR_INVALID")
    result = None
    if not len(times) or target < times[0] or target > times[-1]:
        return result
    offset = int(np.searchsorted(times, target))
    candidates = [i for i in (offset-1, offset) if 0 <= i < len(times)]
    chosen = min(candidates, key=lambda i: abs(int(times[i])-target))
    if abs(int(times[chosen])-target) > maximum_gap_ns:
        return result
    result = poses[chosen]
    return result


# 功能：
#   1. 核对采集摘要、原始像素布局与递增源时间，拒绝矛盾或含糊的采集内容。
#   2. 比较正确姿态、人工姿态偏差及延迟姿态下的几何校正，保留未匹配及无命中帧。
#   3. 统计可观测子空间误差、退化方向和离线计算耗时，不授予定位协方差或飞行资格。
# 输入：
#   directory：夹具采集证据目录。
#   semantic_path：与采集绑定的地图碰撞语义文件。
#   allow_incomplete_capture：是否允许显式分析未完成采集，同时保留失败状态。
#   joint_pose：是否联合求解平移和姿态，而非仅求解条件平移。
#   temporal_pose：是否以真实输入位移连续传播上一帧修正；各扰动条件拥有独立历史。
# 输出：
#   report：含原始证据摘要、逐帧比较与条件统计的离线审计报告。
def compare_moving_fixture(directory: Path, semantic_path: Path, *,
                           allow_incomplete_capture: bool = False,
                           joint_pose: bool = False, optical_world: Path | None = None,
                           temporal_pose: bool = False) -> dict:
    if (any(type(value) is not bool for value in
            (joint_pose, allow_incomplete_capture, temporal_pose))
            or (temporal_pose and not joint_pose)):
        raise ValueError("MOTION_AUDIT_MODE_INVALID")
    raw_report = _bounded(directory / "capture.json", 16*1024*1024)
    capture = decode_json(raw_report, limit=16*1024*1024, node_limit=1_000_000)
    if (not isinstance(capture, dict)
            or capture.get("schema") != "dronedream.camera-motion-calibration"
            or type(capture.get("complete")) is not bool
            or (capture.get("complete") is not True and not allow_incomplete_capture)
            or capture.get("measured_motion") is not True
            or any(capture.get(k) is not False for k in ("native_estimator_qualification",
                "covariance_qualified", "model_control_qualification", "actuator_commands_sent"))):
        raise ValueError("MOTION_AUDIT_REQUIRES_COMPLETE_CAMERA_ONLY_CAPTURE")
    data = {}
    for name in ("poses.json", "frames.json", "commands.json"):
        raw = _bounded(directory / name, 32*1024*1024)
        if bytes_digest(raw) != capture["files"][name]:
            raise ValueError("MOTION_AUDIT_CAPTURE_DIGEST_MISMATCH")
        data[name] = decode_json(raw, limit=32*1024*1024, node_limit=1_000_000)
    poses, frames = data["poses.json"], data["frames.json"]
    if (not isinstance(poses, list) or not isinstance(frames, list)
            or not isinstance(data["commands.json"], list)
            or type(capture.get("pose_count")) is not int
            or type(capture.get("frame_count")) is not int
            or not 2 <= len(poses) <= 25000 or not 1 <= len(frames) <= 256
            or len(poses) != capture["pose_count"] or len(frames) != capture["frame_count"]):
        raise ValueError("MOTION_AUDIT_CAPTURE_COUNT_INVALID")
    for pose in poses:
        if not isinstance(pose, dict):
            raise ValueError("MOTION_AUDIT_POSE_INVALID")
        stamp = pose.get("simulation_time_ns")
        if type(stamp) is not int or not 0 <= stamp <= (1 << 63)-1:
            raise ValueError("MOTION_AUDIT_POSE_TIME_INVALID")
        _rotation(pose["orientation_wxyz"])
        position = pose.get("position_m")
        if (not isinstance(position, list) or len(position) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or abs(value) > 1e5 for value in position)):
            raise ValueError("MOTION_AUDIT_POSE_INVALID")
    times = np.asarray([p["simulation_time_ns"] for p in poses], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("MOTION_AUDIT_POSE_TIME_NOT_PROGRESSING")
    semantic_raw = _bounded(semantic_path, 16*1024*1024)
    if bytes_digest(semantic_raw) != capture["sources"]["semantic"]["sha256"]:
        raise ValueError("MOTION_AUDIT_MAP_DIGEST_MISMATCH")
    semantic = decode_json(semantic_raw, limit=16*1024*1024, node_limit=1_000_000)
    if not isinstance(semantic, dict):
        raise ValueError("MOTION_AUDIT_MAP_INVALID")
    source_world = capture.get("sources", {}).get("world", {})
    if not source_world.get("path") or not source_world.get("sha256"):
        raise ValueError("MOTION_AUDIT_BOUND_OPTICAL_WORLD_REQUIRED")
    index, _, optical_receipt = compile_optical_map(
        optical_world or Path(source_world["path"]), expected_world_sha256=source_world["sha256"],
        expected_resources=capture.get('render_receipt', {}).get('preserved_relative_resources', {})
    )
    calibration = DepthProjectionCalibration(**capture["calibration"])
    if calibration.sha256 != capture["calibration_sha256"]:
        raise ValueError("MOTION_AUDIT_CALIBRATION_MISMATCH")
    fixture_sensor_mount(calibration)
    bias = np.array([.04, -.03, .02])
    attitude_error = euler_quaternion(np.radians([.3, -.2, .5]))
    rows = []
    temporal_identity = dict(map_sha256=bytes_digest(semantic_raw),
                             binding_sha256=bytes_digest(raw_report),
                             clock_domain="camera-only-fixture:" + bytes_digest(raw_report))
    trackers = {condition: TemporalMapPoseTracker(**temporal_identity)
                for condition in ("exact_attitude", "biased_attitude", "delayed_pose_100ms")}
    last_time = -1
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("MOTION_AUDIT_FRAME_IDENTITY_INVALID")
        path = frame.get("path")
        if not isinstance(path, str) or re.fullmatch(r"raw/[0-9]{4}\.depth\.f32", path) is None:
            raise ValueError("MOTION_AUDIT_FRAME_PATH_INVALID")
        resolved = (directory / path).resolve()
        if not resolved.is_relative_to(directory.resolve()):
            raise ValueError("MOTION_AUDIT_FRAME_PATH_ESCAPES_CAPTURE")
        # 保留原路径交给普通文件检查，不能先 resolve 再把内部链接伪装成普通文件。
        raw = _bounded(directory / path, 640*480*4)
        stamp = frame.get("simulation_time_ns")
        if type(stamp) is not int or stamp <= last_time or stamp > (1 << 63)-1:
            raise ValueError("MOTION_AUDIT_FRAME_TIME_NOT_PROGRESSING")
        last_time = stamp
        if (bytes_digest(raw) != frame["sha256"] or len(raw) != frame["bytes"]
                or any(type(frame.get(key)) is not int
                       for key in ("width", "height", "step", "bytes"))
                or frame["step"] != calibration.width*4
                or len(raw) != calibration.width*calibration.height*4
                or frame.get("width") != calibration.width
                or frame.get("height") != calibration.height
                or frame.get("pixel_format") != "R_FLOAT32"):
            raise ValueError("MOTION_AUDIT_FRAME_IDENTITY_INVALID")
        row = {"path": path, "simulation_time_ns": stamp, "conditions": {}}
        rows.append(row)
        reference = nearest_source_pose(poses, times, stamp)
        if reference is None:
            row["issue"] = "NO_SOURCE_TIME_POSE_WITHIN_8_MS"
            continue
        row["source_pair_gap_ms"] = abs(reference["simulation_time_ns"]-stamp)/1e6
        row["reference_position_m"] = reference["position_m"]
        began = time.perf_counter_ns()
        try:
            projected = project_metric_depth_frame(data=raw, row_step_bytes=frame["step"],
                                                   calibration=calibration)
        except ValueError as error:
            # 无有效深度是该帧的失败结果；其他契约或编程错误仍抛出，不吞掉未知问题。
            if str(error) != "depth image contains no calibrated metric samples":
                raise
            row["projection_wall_ms"] = (time.perf_counter_ns()-began)/1e6
            row["issue"] = "NO_VALID_METRIC_PIXELS"
            continue
        projection_ms = (time.perf_counter_ns()-began)/1e6
        row.update({"projection_wall_ms": projection_ms,
                    "source_coverage": projected.source_coverage,
                    "ray_count": len(projected.samples)})
        if not any(sample.hit for sample in projected.samples):
            row["issue"] = "NO_METRIC_HITS"
            continue
        delayed = nearest_source_pose(poses, times, stamp-100_000_000)
        for condition, observation in (("exact_attitude", reference),
                ("biased_attitude", reference), ("delayed_pose_100ms", delayed)):
            if observation is None:
                row["conditions"][condition] = {"issue": "DELAYED_POSE_NOT_AVAILABLE"}
                continue
            estimated = {"position_m": (np.asarray(observation["position_m"])+bias).tolist(),
                         "orientation_wxyz": observation["orientation_wxyz"]}
            if condition == "biased_attitude":
                estimated["orientation_wxyz"] = _quaternion_product(
                    estimated["orientation_wxyz"], attitude_error)
            started = time.perf_counter_ns()
            points, origin = sensor_hits_world(projected.samples, estimated)
            solve_start = time.perf_counter_ns()
            temporal_result = None
            if temporal_pose:
                temporal_result = trackers[condition].update(points, index,
                    sensor_origins_world_m=origin,
                    reference_position_world_m=estimated["position_m"],
                    source_timestamp_ns=stamp, reset_counter=0, **temporal_identity)
                fit = temporal_result.fit
            else:
                fit = (fit_map_pose(points, index, sensor_origins_world_m=origin,
                                   reference_position_world_m=estimated["position_m"])
                       if joint_pose else fit_map_translation(points, index,
                                                             sensor_origins_world_m=origin))
            ended = time.perf_counter_ns()
            initial_error = np.asarray(estimated["position_m"])-reference["position_m"]
            corrected_error = initial_error+fit.correction_world_m
            null = np.asarray(fit.unobserved_directions_world).reshape((-1, 3))
            observed_error = corrected_error-null.T @ (null @ corrected_error)
            attitude_residual = ((np.asarray(fit.rotation_world_from_input) if joint_pose
                                  else np.eye(3)) @ _rotation(estimated["orientation_wxyz"])
                                 @ _rotation(reference["orientation_wxyz"]).T)
            angular_error = float(np.arccos(np.clip((np.trace(attitude_residual)-1)/2, -1, 1)))
            row["conditions"][condition] = {**asdict(fit),
                "temporal_history": ({key: value for key, value in asdict(temporal_result).items()
                                      if key != "fit"} if temporal_result else None),
                "input_error_world_m": initial_error.tolist(),
                "corrected_error_world_m": corrected_error.tolist(),
                "corrected_attitude_error_rad": angular_error if fit.usable_candidate else None,
                "observed_subspace_error_norm_m":
                    float(np.linalg.norm(observed_error)) if fit.usable_candidate else None,
                "fit_wall_ms": (ended-solve_start)/1e6,
                "transform_plus_fit_wall_ms": (ended-started)/1e6,
                "projection_transform_fit_wall_ms": projection_ms+(ended-started)/1e6}
    summary = {}
    for condition in ("exact_attitude", "biased_attitude", "delayed_pose_100ms"):
        values = [r["conditions"][condition] for r in rows if condition in r["conditions"]]
        valid = [v for v in values if v.get("usable_candidate") is True]
        summary[condition] = {"evaluated": len(values), "usable_candidates": len(valid),
            "issues": dict(Counter(v["issue"] for v in values if v.get("issue"))),
            "observed_translation_ranks": dict(Counter(v["observed_translation_rank"]
                                                         for v in valid)),
            "observed_subspace_error_norm_m": _percentiles(
                [v["observed_subspace_error_norm_m"] for v in valid]),
            "corrected_absolute_error_xyz_p95_m": np.quantile(
                np.abs([v["corrected_error_world_m"] for v in valid]), .95, axis=0).tolist()
                if valid else None,
            "corrected_attitude_error_rad": _percentiles(
                [v["corrected_attitude_error_rad"] for v in valid]),
            "observed_pose_ranks": dict(Counter(v["observed_pose_rank"] for v in valid))
                if joint_pose else None,
            "fit_wall_ms_all_attempts": _percentiles([v["fit_wall_ms"] for v in values
                                                       if "fit_wall_ms" in v]),
            "projection_transform_fit_wall_ms_all_attempts": _percentiles(
                [v["projection_transform_fit_wall_ms"] for v in values
                 if "projection_transform_fit_wall_ms" in v])}
    report = {"schema": "dronedream.camera-motion-geometry-audit",
        "solver": "joint-position-attitude" if joint_pose else "conditional-translation",
        "temporal_pose": temporal_pose,
        "temporal_implementation_sha256": (bytes_digest((Path(__file__).parent /
            "temporal_map_pose.py").read_bytes()) if temporal_pose else None),
        "solver_implementation_sha256": bytes_digest((Path(__file__).parent /
            ("local_pose_alignment.py" if joint_pose else "local_map_alignment.py")).read_bytes()),
        "capture_complete": capture["complete"], "capture_issue": capture.get("issue"),
        "capture_close_errors": capture.get("close_errors", []), "capture_sha256":
        bytes_digest(raw_report), "capture_files": capture["files"],
        "map_sha256": bytes_digest(semantic_raw), "calibration_sha256": calibration.sha256,
        "optical_map": optical_receipt,
        "implementation_sha256": bytes_digest(Path(__file__).read_bytes()),
        "frame_count": len(rows), "unmatched_frames": sum(
            r.get("issue") == "NO_SOURCE_TIME_POSE_WITHIN_8_MS" for r in rows),
        "invalid_geometry_frames": sum(r.get("issue") in {
            "NO_METRIC_HITS", "NO_VALID_METRIC_PIXELS"} for r in rows),
        "measured_position_span_m": capture["measured_position_span_m"],
        "source_pair_gap_ms": _percentiles([r["source_pair_gap_ms"] for r in rows
                                            if "source_pair_gap_ms" in r]),
        "deliberate_position_bias_m": bias.tolist(), "attitude_bias_rpy_deg": [.3, -.2, .5],
        "summary": summary, "frames": rows, "covariance_qualified": False,
        "native_estimator_qualification": False, "model_control_qualification": False,
        "truth_used_to_construct_corrupted_fixture_input": True,
        "limitation": "Kinematic camera-only calibration test. Truth intentionally supplies "
          "a corrupted test pose, NOT native estimation. Conditional error statistics exclude "
          "rejected fits listed separately. Timing is offline compute, not end-to-end control."}
    return report
