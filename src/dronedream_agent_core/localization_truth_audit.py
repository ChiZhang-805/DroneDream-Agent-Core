"""Independent, timestamp-qualified offline comparison of geometric candidates.

No alignment is fitted to truth. Clock matching reports its actual gap; a
receive-time match is not promoted to source-clock synchronization or accuracy
qualification. This module is not imported by control or model inference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import RawMetricRangeScan
from .hashing import sha256_json
from .localization_observations import MAXIMUM_CAPTURE_RECORDS, MAXIMUM_RECORD_BYTES
from .localization_truth_capture import MAXIMUM_TRUTH_RECORD_BYTES, MAXIMUM_TRUTH_RECORDS
from .simulation_sensor_frames import _rotation, collision_center_from_canonical


def _real(value, issue):
    """Reject coercion and overflow before NumPy can hide malformed JSON numbers."""
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        raise ValueError(issue)
    return float(value)


def _numeric_array(value, shape, issue):
    """Validate the small fixed candidate shape before converting elements to float."""
    try:
        raw = np.asarray(value, dtype=object)
        if raw.shape != shape:
            raise ValueError(issue)
        return np.array([_real(item, issue) for item in raw.flat]).reshape(shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(issue) from error


def _witness_motion(centers, orientations, source_times):
    """Offline motion strata, never a velocity input to the controller.

    Central differences use source simulation time, not receipt speed. Endpoints
    and gaps over 100 ms remain unclassified. Include rotation-only motion so
    apparently stationary positions cannot hide moving-camera errors.
    """
    output = []
    for index in range(len(centers)):
        item = {"state": "unclassified", "linear_speed_mps": None,
                "angular_speed_radps": None}
        if 0 < index < len(centers)-1:
            dt = source_times[index+1] - source_times[index-1]
            if 0 < dt <= .1:
                speed = float(np.linalg.norm(np.asarray(centers[index+1])
                    - centers[index-1]) / dt)
                rotation = orientations[index+1] @ orientations[index-1].T
                angular = float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))/dt)
                state = ("moving" if speed >= .05 or angular >= .05 else
                         "stationary" if speed <= .02 and angular <= .02 else "slow")
                item.update(state=state, linear_speed_mps=speed, angular_speed_radps=angular)
        output.append(item)
    return output


def _records(path, *, limit, maximum_bytes):
    """Read a finite complete JSONL clip and verify each row plus the exact file digest."""
    digest, records = hashlib.sha256(), []
    with path.open("rb") as stream:
        while line := stream.readline(maximum_bytes + 1):
            if len(line) > maximum_bytes or not line.endswith(b"\n") or len(records) >= limit:
                raise ValueError("TRUTH_AUDIT_CAPTURE_BOUND_EXCEEDED")
            digest.update(line)
            try:
                row = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise ValueError("TRUTH_AUDIT_RECORD_INVALID") from error
            if not isinstance(row, dict):
                raise ValueError("TRUTH_AUDIT_RECORD_INVALID")
            stored = row.pop("record_sha256", None)
            if stored != sha256_json(row):
                raise ValueError("TRUTH_AUDIT_RECORD_HASH_MISMATCH")
            records.append(row)
    if not records:
        raise ValueError("TRUTH_AUDIT_EMPTY_CAPTURE")
    return records, digest.hexdigest()


def compare_geometry_truth(capture_path: Path, report: dict, truth_path: Path,
                           frames_path: Path) -> dict:
    """Compare an offline candidate against independent, same-run timestamped truth.

    Matching uses no fitted transform and never writes a control input. Metadata
    hashes establish consistency, not trust; error quantiles do not grant flight
    qualification or turn host-receive timestamps into exposure synchronization.
    """
    if len({p.parent.resolve() for p in (capture_path, truth_path, frames_path)}) != 1:
        raise ValueError("TRUTH_AUDIT_REQUIRES_SAME_RUN")
    with frames_path.open("rb") as stream:
        raw = stream.read(131073)
    if len(raw) > 131072:
        raise ValueError("TRUTH_AUDIT_FRAME_CONTRACT_TOO_LARGE")
    frames = json.loads(raw)
    if not isinstance(frames, dict):
        raise ValueError("TRUTH_AUDIT_FRAME_CONTRACT_INVALID")
    digest = frames.pop("record_sha256", None)
    if (digest != sha256_json(frames) or frames.get("depth_mount_verified") is not True
            or frames.get("truth_used_as_control_input") is not False):
        raise ValueError("TRUTH_AUDIT_FRAME_CONTRACT_INVALID")
    scans, capture_hash = _records(capture_path, limit=MAXIMUM_CAPTURE_RECORDS,
                                   maximum_bytes=MAXIMUM_RECORD_BYTES)
    if not isinstance(report, dict) or not isinstance(report.get("frames"), list):
        raise ValueError("TRUTH_AUDIT_CANDIDATE_CAPTURE_MISMATCH")
    if capture_hash != report.get("capture_sha256") or len(scans) != len(report["frames"]):
        raise ValueError("TRUTH_AUDIT_CANDIDATE_CAPTURE_MISMATCH")
    truth, truth_hash = _records(truth_path, limit=MAXIMUM_TRUTH_RECORDS,
                                 maximum_bytes=MAXIMUM_TRUTH_RECORD_BYTES)
    centers, orientations, previous = [], [], None
    for row in truth:
        if (row.get("frames_sha256") != digest or row.get("qualification_granted") is not False
                or row.get("truth_used_as_control_input") is not False
                or row.get("schema_version") != "dronedream.localization-truth-observation.v1"):
            raise ValueError("TRUTH_AUDIT_WITNESS_BINDING_INVALID")
        times = [row.get(k) for k in ("sequence", "received_monotonic_seconds",
                  "received_at_unix_ms", "publisher_simulation_time_ns")]
        if (any(type(times[index]) is not int or not 0 <= times[index] < 2**64
                for index in (0, 2, 3)) or times[0] < 1
                or _real(times[1], "TRUTH_AUDIT_CLOCK_INVALID") < 0):
            raise ValueError("TRUTH_AUDIT_CLOCK_INVALID")
        if previous is not None and any(a <= b for a, b in zip(times, previous, strict=True)):
            raise ValueError("TRUTH_AUDIT_CLOCK_NOT_PROGRESSING")
        previous = times
        center = collision_center_from_canonical(model_world=row["model_world"],
            canonical_in_model=row["canonical_in_model"],
            canonical_at_rest=frames["canonical_at_rest"],
            collision_center_model_m=frames["collision_center_model_m"])
        stored_center = _numeric_array(row.get("collision_center_world_enu_m"), (3,),
                                       "TRUTH_AUDIT_CENTER_REFERENCE_MISMATCH")
        if not np.allclose(center, stored_center, rtol=0, atol=1e-10):
            raise ValueError("TRUTH_AUDIT_CENTER_REFERENCE_MISMATCH")
        centers.append(center)
        orientations.append(_rotation(row["model_world"]["orientation_wxyz"])
            @ _rotation(row["canonical_in_model"]["orientation_wxyz"])
            @ _rotation(frames["canonical_at_rest"]["orientation_wxyz"]).T)
    scene_times = []
    for row in scans:
        # A recomputed row hash does not make malformed sensor state meaningful.
        row["scan"] = RawMetricRangeScan.model_validate(row.get("scan"), strict=True).model_dump()
        clock = row.get("source_clock")
        if clock is not None and not isinstance(clock, dict):
            raise ValueError("TRUTH_AUDIT_SENSOR_CLOCK_INVALID")
        stamp = (clock or {}).get("publisher_simulation_time_ns")
        if stamp is not None and (type(stamp) is not int or not 0 <= stamp < 2**64):
            raise ValueError("TRUTH_AUDIT_SENSOR_CLOCK_INVALID")
        scene_times.append(stamp)
    has_scene = all(type(t) is int and t >= 0 for t in scene_times)
    # Subtract an exact integer origin before conversion; large source clocks
    # must not lose small matching intervals through float rounding.
    origin_ns = truth[0]["publisher_simulation_time_ns"]
    scan_times = (np.array([t-origin_ns for t in scene_times], dtype=float) / 1e9
                  if has_scene else np.array([
                      r["scan"]["observed_at_monotonic_seconds"] for r in scans]))
    if not np.isfinite(scan_times).all() or np.any(np.diff(scan_times) <= 0):
        raise ValueError("TRUTH_AUDIT_SENSOR_CLOCK_NOT_PROGRESSING")
    truth_times = np.array([(r["publisher_simulation_time_ns"]-origin_ns) / 1e9 if has_scene
                           else r["received_monotonic_seconds"] for r in truth])
    witness_motion = _witness_motion(centers, orientations,
        np.array([(r["publisher_simulation_time_ns"]-origin_ns) / 1e9 for r in truth]))
    time_limit = .02 if has_scene else .05
    truth_source_ns = [row["publisher_simulation_time_ns"] for row in truth]
    results, before, after, angular_before, angular_after = [], [], [], [], []
    for source, candidate, source_time in zip(scans, report["frames"], scan_times, strict=True):
        if (not isinstance(candidate, dict) or type(candidate.get("sequence")) is not int
                or candidate["sequence"] != source["scan"]["sequence"]):
            raise ValueError("TRUTH_AUDIT_CANDIDATE_SEQUENCE_MISMATCH")
        nearest = int(np.argmin(np.abs(truth_times - source_time)))
        gap = abs(truth_times[nearest] - source_time)
        if has_scene:
            # Compare source nanoseconds as integers, including exact ties.
            # Floating seconds can choose a different witness on either side
            # of the same midpoint and consequently change its receive gap.
            stamp = source["source_clock"]["publisher_simulation_time_ns"]
            right = bisect_left(truth_source_ns, stamp)
            choices = [i for i in (right-1, right) if 0 <= i < len(truth_source_ns)]
            nearest = min(choices, key=lambda i: (abs(truth_source_ns[i]-stamp), i))
            gap = abs(truth_source_ns[nearest]-stamp)/1e9
        joint = report.get("solver") == "joint-position-attitude"
        fit = candidate.get("pose_fit" if joint else "translation_fit") or {}
        host_gap = abs(truth[nearest]["received_monotonic_seconds"]
                       - source["scan"]["observed_at_monotonic_seconds"])
        item = {"sequence": source["scan"]["sequence"], "nearest_witness_time_gap_ms": gap * 1000,
                "nearest_witness_receive_gap_ms": host_gap * 1000,
                "fit_usable": fit.get("usable_candidate") is True, "compared": False,
                "witness_motion_state": "unclassified"}
        # 功能：逐项报告源时间与接收时间的失败原因；不能用较小的源时间差
        # 掩盖排队延迟，也不能把未比较的样本混作零误差。
        # 输入：本帧原始来源钟、独立见证钟及固定匹配预算。
        # 输出：有界原因列表；仅增强离线诊断，不放宽原有准入条件。
        rejection_reasons = []
        if not truth_times[0] <= source_time <= truth_times[-1]:
            rejection_reasons.append("outside_witness_time_span")
        if gap > time_limit:
            rejection_reasons.append("source_time_gap_exceeded")
        if host_gap > .1:
            rejection_reasons.append("receive_time_gap_exceeded")
        time_matched = not rejection_reasons
        if not item["fit_usable"]:
            rejection_reasons.append("fit_unusable")
        item["comparison_rejection_reasons"] = rejection_reasons
        if time_matched:
            motion = witness_motion[nearest]
            item.update(witness_motion_state=motion["state"],
                witness_linear_speed_mps=motion["linear_speed_mps"],
                witness_angular_speed_radps=motion["angular_speed_radps"])
        if time_matched and item["fit_usable"]:
            position = np.array([source["scan"]["body_position_world_enu_m"][a] for a in "xyz"])
            baseline = position - centers[nearest]
            correction = _numeric_array(fit.get("correction_world_m"), (3,),
                                        "TRUTH_AUDIT_CANDIDATE_INVALID")
            if (correction.shape != (3,) or not np.isfinite(correction).all()
                    or np.linalg.norm(correction) > 1.):
                raise ValueError("TRUTH_AUDIT_CANDIDATE_INVALID")
            corrected = baseline + correction
            item.update(compared=True, baseline_error_world_enu_m=baseline.tolist(),
                        fitted_error_world_enu_m=corrected.tolist())
            if joint:
                delta_rotation = _numeric_array(fit.get("rotation_world_from_input"), (3, 3),
                                                "TRUTH_AUDIT_CANDIDATE_ROTATION_INVALID")
                if (delta_rotation.shape != (3, 3) or not np.isfinite(delta_rotation).all()
                        or not np.allclose(delta_rotation.T @ delta_rotation, np.eye(3),
                                           atol=1e-8, rtol=0.)
                        or abs(np.linalg.det(delta_rotation)-1.) > 1e-8):
                    raise ValueError("TRUTH_AUDIT_CANDIDATE_ROTATION_INVALID")
                q = source["scan"]["body_orientation_world_from_body"]
                native_rotation = _rotation([q[k] for k in ("w", "x", "y", "z")])

                def error_angle(rotation, reference_rotation=orientations[nearest]):
                    """Principal SO(3) angle in radians; clamp only floating-point roundoff."""
                    difference = rotation @ reference_rotation.T
                    return float(np.arccos(np.clip((np.trace(difference)-1)/2, -1, 1)))

                angular_before.append(error_angle(native_rotation))
                angular_after.append(error_angle(delta_rotation @ native_rotation))
                item.update(baseline_attitude_error_rad=angular_before[-1],
                            fitted_attitude_error_rad=angular_after[-1])
            before.append(np.abs(baseline))
            after.append(np.abs(corrected))
        results.append(item)

    def quantiles(values):
        """Keep empty motion strata unmeasured instead of fabricating zero error."""
        return ({name: np.quantile(values, q, axis=0).tolist()
                 for name, q in (("p50", .5), ("p95", .95), ("maximum", 1.))} if values else None)

    strata = {}
    for label in ("stationary", "slow", "moving", "unclassified"):
        rows = [r for r in results if r["witness_motion_state"] == label]
        scored = [r for r in rows if r["compared"]]
        strata[label] = {"frame_count": len(rows), "compared_count": len(scored),
            "unusable_fit_count": sum(not r["fit_usable"] for r in rows),
            "baseline_absolute_error_xyz_m": quantiles([
                np.abs(r["baseline_error_world_enu_m"]) for r in scored]),
            "fitted_absolute_error_xyz_m": quantiles([
                np.abs(r["fitted_error_world_enu_m"]) for r in scored]),
            "baseline_attitude_error_rad": quantiles([
                r["baseline_attitude_error_rad"] for r in scored
                if "baseline_attitude_error_rad" in r]),
            "fitted_attitude_error_rad": quantiles([
                r["fitted_attitude_error_rad"] for r in scored
                if "fitted_attitude_error_rad" in r])}

    return {"frames_sha256": digest, "truth_capture_sha256": truth_hash,
            "capture_sha256": capture_hash, "source_frame_count": len(scans),
            "truth_frame_count": len(truth), "compared_count": len(after),
            "comparison_rejection_counts": dict(Counter(
                reason for item in results for reason in item["comparison_rejection_reasons"])),
            "nearest_witness_receive_gap_ms": quantiles([
                item["nearest_witness_receive_gap_ms"] for item in results]),
            "clock_comparison": "publisher-simulation-time" if has_scene else "host-receipt-time",
            "maximum_allowed_match_gap_ms": time_limit * 1000,
            "baseline_absolute_error_xyz_m": quantiles(before),
            "fitted_absolute_error_xyz_m": quantiles(after),
            "baseline_attitude_error_rad": quantiles(angular_before),
            "fitted_attitude_error_rad": quantiles(angular_after),
            "witness_position_span_xyz_m": np.ptp(centers, axis=0).tolist(),
            "motion_strata": strata,
            "motion_classification": {"time_base": "publisher-simulation-time",
                "maximum_central_difference_interval_seconds": .1,
                "moving_linear_speed_mps": .05, "moving_angular_speed_radps": .05,
                "stationary_maximum_linear_speed_mps": .02,
                "stationary_maximum_angular_speed_radps": .02,
                "purpose": "offline-error-stratification-not-flight-phase-or-authority"},
            "qualification_granted": False, "truth_used_as_control_input": False,
            "limitations": ["nearest timestamp, not interpolated/exposure-synchronized truth",
                            "stationary captures do not qualify moving-flight localization",
                            "no transform was fitted against the witness"], "frames": results}
