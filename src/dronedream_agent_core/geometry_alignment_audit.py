"""Offline map/scan consistency and translation observability diagnostics.

Nearest surfaces alone do not establish correct correspondences or covariance.
Conditional translation or joint pose candidates are computed offline, never applied.
Nearest-box diagnostics are explicitly separate from full primitive fitting.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import CalibratedRangeSensorMount, RawMetricRangeScan
from .hashing import sha256_json
from .local_map_alignment import fit_map_translation
from .local_pose_alignment import fit_map_pose
from .localization_observations import MAXIMUM_CAPTURE_RECORDS, MAXIMUM_RECORD_BYTES
from .optical_map import compile_optical_map
from .sensor_bridge import MetricRangeSensorBridge


def nearest_box_surfaces(points, primitives) -> tuple[np.ndarray, np.ndarray]:
    """Return true unsigned box-surface distance and outward world normal.

    Points inside a solid are NOT zero-distance inliers. Edge/corner normals
    follow the closest surface direction. No control-envelope inflation applies.
    """
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not primitives):
        raise ValueError("GEOMETRY_AUDIT_POINTS_OR_BOXES_INVALID")
    minimum = np.full(len(points), np.inf)
    normals = np.zeros_like(points)
    for box in primitives:
        center = np.asarray([box[f"center_{axis}"] for axis in "xyz"], dtype=np.float64)
        half = np.asarray([box[f"size_{axis}"] / 2 for axis in "xyz"], dtype=np.float64)
        yaw = float(box.get("yaw_rad", 0.))
        tilt = [float(box.get(f"{axis}_rad", 0.)) for axis in ("roll", "pitch")]
        if (not np.isfinite([*center, *half, yaw, *tilt]).all() or np.min(half) <= 0
                or any(abs(value) > 1e-12 for value in tilt)):
            raise ValueError("GEOMETRY_AUDIT_BOX_INVALID_OR_TILTED")
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        local = (points - center) @ rotation
        delta = local - np.clip(local, -half, half)
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1e-15)
        inside = distance <= 1e-15
        if np.any(inside):
            gaps = half - np.abs(local[inside])
            axis = np.argmin(gaps, axis=1)
            indices = np.arange(len(axis))
            distance[inside] = gaps[indices, axis]
            inner_normal = np.zeros((len(axis), 3))
            inner_normal[indices, axis] = np.where(local[inside][indices, axis] >= 0, 1., -1.)
            normal[inside] = inner_normal
        better = distance < minimum
        minimum[better] = distance[better]
        normals[better] = (normal @ rotation.T)[better]
    return minimum, normals


def analyze_geometry_capture(capture_path: Path, semantic_path: Path, *,
                             fit_translation: bool = False,
                             check_perturbations: bool = False,
                             fit_joint_pose: bool = False,
                             optical_world: Path | None = None,
                             expected_world_sha256: str | None = None,
                             render_receipt: Path | None = None) -> dict:
    if (any(type(flag) is not bool for flag in
            (fit_translation, fit_joint_pose, check_perturbations))
            or (fit_translation and fit_joint_pose)):
        raise ValueError("GEOMETRY_AUDIT_EXPLICIT_SINGLE_SOLVER_REQUIRED")
    fitting = fit_translation or fit_joint_pose
    if check_perturbations and not fitting:
        raise ValueError("GEOMETRY_AUDIT_PERTURBATIONS_REQUIRE_FIT")
    with semantic_path.open("rb") as stream:
        semantic_bytes = stream.read(16*1024*1024+1)
    if len(semantic_bytes) > 16*1024*1024:
        raise ValueError("GEOMETRY_AUDIT_MAP_TOO_LARGE")
    map_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
    semantic = json.loads(semantic_bytes)
    if not isinstance(semantic, dict):
        raise ValueError("GEOMETRY_AUDIT_MAP_NOT_OBJECT")
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    if not isinstance(primitives, list) or not primitives:
        raise ValueError("GEOMETRY_AUDIT_MAP_PRIMITIVES_MISSING")
    optical_receipt = None
    translation_index = None
    if fitting:
        if optical_world is None or expected_world_sha256 is None:
            raise ValueError("GEOMETRY_AUDIT_BOUND_OPTICAL_WORLD_REQUIRED")
        resources = {}
        if render_receipt is not None:
            from .plugin_files import read_plugin_file
            receipt = json.loads(read_plugin_file(render_receipt, limit=8*1024*1024))
            if not isinstance(receipt, dict) or receipt.get('source_world_sha256') != expected_world_sha256:
                raise ValueError("GEOMETRY_AUDIT_RENDER_WORLD_MISMATCH")
            resources = receipt.get('preserved_relative_resources', {})
        translation_index, primitives, optical_receipt = compile_optical_map(
            optical_world, expected_world_sha256=expected_world_sha256, expected_resources=resources)
    boxes = [p for p in primitives if all(f"size_{axis}" in p for axis in "xyz")
             and p.get("registration_eligible", True)
             and all(abs(float(p.get(f"{axis}_rad", 0.))) <= 1e-12 for axis in ("roll", "pitch"))]
    if not boxes and not fitting:
        raise ValueError("GEOMETRY_AUDIT_NO_SUPPORTED_SURFACES")
    fit_key = "pose_fit" if fit_joint_pose else "translation_fit"

    def solve(hits, origins, reference):
        return (fit_map_pose(hits, translation_index, sensor_origins_world_m=origins,
                             reference_position_world_m=reference)
                if fit_joint_pose else fit_map_translation(hits, translation_index,
                                                            sensor_origins_world_m=origins))
    frames = []
    capture_digest = hashlib.sha256()
    last_time = -1.
    last_sequence = -1
    binding = None
    with capture_path.open("rb") as stream:
        while line := stream.readline(MAXIMUM_RECORD_BYTES + 1):
            if (len(line) > MAXIMUM_RECORD_BYTES or not line.endswith(b"\n")
                    or len(frames) >= MAXIMUM_CAPTURE_RECORDS):
                raise ValueError("GEOMETRY_AUDIT_CAPTURE_LIMIT_OR_TRUNCATION")
            capture_digest.update(line)
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("GEOMETRY_AUDIT_RECORD_NOT_OBJECT")
            digest = record.pop("record_sha256", None)
            if digest != sha256_json(record) or record.get("map_sha256") != map_sha256:
                raise ValueError("GEOMETRY_AUDIT_CAPTURE_OR_MAP_HASH_MISMATCH")
            if (record.get("schema_version") != "dronedream.native-geometry-observation.v1"
                    or record.get("truth_correction_applied") is not False
                    or record.get("motion_permission_granted") is not False
                    or record.get("model_control_qualification_granted") is not False):
                raise ValueError("GEOMETRY_AUDIT_CAPTURE_NOT_NATIVE_DIAGNOSTIC")
            current_binding = tuple(record.get(key) for key in (
                "native_pose_binding_sha256", "calibration_sha256"))
            if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in current_binding):
                raise ValueError("GEOMETRY_AUDIT_SOURCE_BINDING_MISSING")
            if binding is not None and current_binding != binding:
                raise ValueError("GEOMETRY_AUDIT_SOURCE_BINDING_CHANGED")
            binding = current_binding
            scan = RawMetricRangeScan.model_validate(record["scan"])
            if scan.sequence <= last_sequence or scan.observed_at_monotonic_seconds <= last_time:
                raise ValueError("GEOMETRY_AUDIT_SOURCE_NOT_PROGRESSING")
            last_time, last_sequence = scan.observed_at_monotonic_seconds, scan.sequence
            mount = CalibratedRangeSensorMount.model_validate(record["mount"])
            frame = MetricRangeSensorBridge(mount).assemble(scan)
            hits = [(ray.endpoint_m.x, ray.endpoint_m.y, ray.endpoint_m.z)
                    for ray in frame.range_rays if ray.hit]
            origins = [(ray.origin_m.x, ray.origin_m.y, ray.origin_m.z)
                       for ray in frame.range_rays if ray.hit]
            summary = {"sequence": scan.sequence, "observed_at_unix_ms": scan.observed_at_unix_ms,
                "hit_count": len(hits), "source_coverage": scan.source_coverage,
                "localization_variance_bound_m2": scan.localization_covariance_m2,
                "source_clock_kind": (record.get("source_clock") or {}).get(
                    "clock_kind", "unknown")}
            if hits and boxes:
                distance, normals = nearest_box_surfaces(hits, boxes)
                inliers = distance <= .25  # Diagnostic bin, not a safety margin or confidence gate.
                gram = normals[inliers].T @ normals[inliers] / max(1, np.count_nonzero(inliers))
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                summary.update({"surface_residual_p50_m": float(np.median(distance)),
                    "surface_residual_p95_m": float(np.quantile(distance, .95)),
                    "fraction_within_025m_of_supported_surface": float(np.mean(inliers)),
                    "translation_normal_gram_eigenvalues": eigenvalues.tolist(),
                    "translation_unobserved_directions_world_enu":
                        eigenvectors[:, eigenvalues < 1e-6].T.tolist()})
            if hits and translation_index is not None:
                reference = np.array([getattr(scan.body_position_world_enu_m, a) for a in "xyz"])
                started = time.perf_counter()
                fit = solve(hits, origins, reference)
                summary[fit_key + "_wall_ms"] = (time.perf_counter() - started) * 1000
                summary[fit_key] = asdict(fit)
                if check_perturbations and fit.usable_candidate:
                    # Known input perturbations on the SAME captured frame
                    # test solver consistency, not absolute accuracy or a
                    # new independent flight. Unknown map error remains.
                    null = np.asarray(fit.unobserved_directions_world).reshape(-1, 3)
                    observed = np.eye(3) - null.T @ null
                    perturbations = []
                    for shift in np.concatenate((np.eye(3), -np.eye(3))) * .04:
                        shifted = solve(np.asarray(hits)+shift, np.asarray(origins)+shift,
                                        reference+shift)
                        record = {"input_shift_world_m": shift.tolist(),
                                  "candidate": asdict(shifted)}
                        if shifted.usable_candidate:
                            difference = (np.asarray(shifted.correction_world_m)
                                          - fit.correction_world_m)
                            record["observed_shift_recovery_error_m"] = float(
                                np.linalg.norm(observed @ (difference + shift)))
                            record["unobserved_correction_change_m"] = float(
                                np.linalg.norm(null @ difference))
                        perturbations.append(record)
                    summary["same_frame_translation_perturbations"] = perturbations
            frames.append(summary)
    if not frames:
        raise ValueError("GEOMETRY_AUDIT_EMPTY_CAPTURE")
    return {"schema_version": "dronedream.native-geometry-audit.v1",
        "solver": ("joint-position-attitude" if fit_joint_pose else
                   "conditional-translation" if fit_translation else "diagnostic-only"),
        "solver_implementation_sha256": (hashlib.sha256((Path(__file__).parent /
            ("local_pose_alignment.py" if fit_joint_pose else "local_map_alignment.py"))
            .read_bytes()).hexdigest() if fitting else None),
        "map_sha256": map_sha256, "capture_sha256": capture_digest.hexdigest(),
        "optical_map": optical_receipt,
        "frame_count": len(frames), "nearest_box_diagnostic_supported_count": len(boxes),
        "nearest_box_diagnostic_unassessed_count": len(primitives) - len(boxes), "frames": frames,
        "pose_correction_applied": False, "covariance_qualification_granted": False,
        "model_control_qualification_granted": False,
        "translation_fit_primitive_counts": (
            translation_index.primitive_counts if translation_index is not None else None),
        "limitations": ["nearest surfaces do not prove correct correspondence",
                        "normal Gram matrix is not a pose covariance",
                        "nearest-box diagnostic excludes other shapes",
                        "visible-surface fit rejects unsupported shapes",
                        "unmapped geometry and dynamic objects are not localized",
                        ("joint pose fit marginalizes unknown rotation for translation rank"
                         if fit_joint_pose else
                         "translation fit is conditional on a fixed input attitude"),
                        "same-frame perturbations are not absolute ground-truth accuracy"]}
