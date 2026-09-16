"""Measure synthetic depth -> map -> feature latency, without control or network I/O.

GC callbacks provide attribution only. Collection remains enabled and its time
remains in every reported wall-clock duration. No measurements are discarded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import struct
import time
from pathlib import Path
from types import SimpleNamespace

from dronedream_agent_core.contracts import QuaternionWxyz, RawMetricRangeScan, Vector3
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker
from dronedream_agent_core.depth_sensor_binding import DepthSensorBinding
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion
from dronedream_agent_core.realtime_feature_encoders import (
    FlightStateEncoder,
    FlightStateSample,
    encode_dynamic_targets,
    encode_metric_geometry,
    fuse_realtime_features,
)
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeSensorRegistry,
    oakd_lite_depth_sensor_contract,
)
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99_ms": ordered[math.ceil(.99 * len(ordered)) - 1], "max_ms": max(ordered)}


def run(repetitions: int, *, include_tracking_state: bool = False) -> dict:
    mount = oakd_lite_depth_sensor_contract()
    registry = RuntimeSensorRegistry()
    binding = DepthSensorBinding(registry, vehicle_id="analytic-fixture")
    bridge = MetricRangeSensorBridge(mount)
    world = MetricVoxelMap(resolution_m=.25,
        minimum_bound_m=Vector3(x=-25, y=-25, z=-25),
        maximum_bound_m=Vector3(x=25, y=25, z=25))
    fusion = RuntimePerceptionFusion(world=world, accepted_sensor_ids={mount.sensor_id},
                                     sensor_registry=registry)
    tracker = DepthMotionTracker()
    state_encoder = FlightStateEncoder()
    payload = struct.pack("<19200f", *[2. + (i % 41) * .1 for i in range(19200)])
    message = SimpleNamespace(width=160, height=120, step=640, pixel_format_type=13, data=payload)
    zero = Vector3(x=0, y=0, z=0)
    orientation = QuaternionWxyz(w=1, x=0, y=0, z=0)
    gc_started = 0
    gc_total_ns = 0
    gc_counts = [0, 0, 0]

    def gc_callback(phase, info):
        nonlocal gc_started, gc_total_ns
        if phase == "start":
            gc_started = time.perf_counter_ns()
        else:
            gc_total_ns += time.perf_counter_ns() - gc_started
            gc_counts[info["generation"]] += 1

    samples = []
    gc.callbacks.append(gc_callback)
    try:
        for index in range(20 + repetitions):
            captured = time.monotonic()
            unix_ms = int(time.time() * 1000)
            initial_gc = gc_total_ns
            initial_gc_counts = list(gc_counts)
            start = time.perf_counter_ns()
            projection = binding.project(message)
            projected = time.perf_counter_ns()
            scan = RawMetricRangeScan(sensor_id=mount.sensor_id, sequence=index + 1,
                observed_at_unix_ms=unix_ms, observed_at_monotonic_seconds=captured,
                body_position_world_enu_m=zero, body_velocity_world_enu_mps=zero,
                body_orientation_world_from_body=orientation, localization_covariance_m2=.01,
                source_coverage=projection.source_coverage, samples=list(projection.samples))
            frame = bridge.assemble(scan)
            assembled = time.perf_counter_ns()
            if include_tracking_state:
                prepared_tracks = tracker.prepare(frame, observed_at_monotonic_seconds=captured)
                frame.dynamic_obstacles = list(prepared_tracks.observations)
            tracked = time.perf_counter_ns()
            health = fusion.ingest(frame, now_unix_ms=int(time.time() * 1000),
                                   now_monotonic_seconds=time.monotonic())
            if include_tracking_state:
                tracker.commit(prepared_tracks)
            fused = time.perf_counter_ns()
            encoded = encode_metric_geometry(scan, sensor_mount=mount,
                expected_horizontal_fov_rad=1.274,
                expected_vertical_fov_rad=2 * math.atan(math.tan(1.274 / 2) * .75),
                encoded_at_unix_ms=int(time.time() * 1000))
            geometry_ready = time.perf_counter_ns()
            ready = True
            if include_tracking_state:
                state = state_encoder.encode(FlightStateSample(
                    source_id="synthetic-state-not-native-telemetry",
                    observed_at_unix_ms=unix_ms, orientation_world_from_body=orientation,
                    velocity_world_enu_mps=zero, acceleration_body_mps2=zero,
                    angular_velocity_body_rad_s=zero, localization_covariance_m2=.01,
                ), encoded_at_unix_ms=int(time.time()*1000))
                dynamic = encode_dynamic_targets(frame.dynamic_obstacles,
                    body_position_world_enu_m=zero, body_orientation_world_from_body=orientation,
                    body_velocity_world_enu_mps=zero, observed_at_unix_ms=unix_ms,
                    encoded_at_unix_ms=int(time.time()*1000))
                snapshot = fuse_realtime_features([encoded, dynamic, state],
                    captured_at_unix_ms=int(time.time()*1000))
                ready = snapshot.ready_for_control
            end = time.perf_counter_ns()
            if index >= 20:
                samples.append({"sequence": index + 1,
                    "projection_ms": (projected - start) / 1e6,
                    "bridge_ms": (assembled - projected) / 1e6,
                    "tracking_ms": (tracked - assembled) / 1e6,
                    "fusion_ms": (fused - tracked) / 1e6,
                    "geometry_features_ms": (geometry_ready - fused) / 1e6,
                    "state_dynamic_snapshot_ms": (end - geometry_ready) / 1e6,
                    "total_ms": (end - start) / 1e6,
                    "gc_ms_included": (gc_total_ns - initial_gc) / 1e6,
                    "gc_collections": [a - b for a, b in zip(gc_counts, initial_gc_counts,
                                                            strict=True)],
                    "stream_healthy": health.stream_healthy,
                    "features_ready": ready,
                    "feature_quality": encoded.quality})
    finally:
        gc.callbacks.remove(gc_callback)
    root = Path(__file__).resolve().parents[1]
    source_names = ("depth_projection.py", "depth_sensor_binding.py", "sensor_bridge.py",
                    "perception_runtime.py", "local_world_model.py", "realtime_feature_encoders.py",
                    "depth_obstacle_tracker.py", "control_feature_contract.py")
    return {"kind": "analytic-sensor-pipeline-benchmark", "python": platform.python_version(),
        "platform": platform.platform(), "repetitions": repetitions,
        "input_kind": "synthetic-static-depth-and-exact-pose-not-flight",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "sources_sha256": {name: hashlib.sha256(
            (root / "src/dronedream_agent_core" / name).read_bytes()).hexdigest()
            for name in source_names},
        "gc_enabled": gc.isenabled(), "gc_thresholds": gc.get_threshold(),
        "includes_tracking_and_state_encoding": include_tracking_state,
        "timing": {key: summarize([sample[key] for sample in samples]) for key in (
            "projection_ms", "bridge_ms", "tracking_ms", "fusion_ms", "geometry_features_ms",
            "state_dynamic_snapshot_ms", "total_ms")},
        "functional_checks_passed": all(s["stream_healthy"] and s["features_ready"]
                                        and s["feature_quality"] >= .35
                                        for s in samples),
        "projection_p99_below_20ms": (
            summarize([s["projection_ms"] for s in samples])["p99_ms"] < 20),
        "cycle_p99_below_50ms": summarize([s["total_ms"] for s in samples])["p99_ms"] < 50,
        "includes_camera_capture_transport_or_model_inference": False,
        "flight_qualification_granted": False, "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--include-tracking-state", action="store_true")
    args = parser.parse_args()
    if not 100 <= args.repetitions <= 2000:
        parser.error("repetitions must be between 100 and 2000")
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    result = run(args.repetitions, include_tracking_state=args.include_tracking_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))
    return 0 if result["functional_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
