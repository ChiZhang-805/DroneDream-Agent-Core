#!/usr/bin/env python3
"""Benchmark the calibrated real-time feature path with reproducible inputs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    DynamicObstacleObservation,
    QuaternionWxyz,
    RawMetricRangeScan,
    RawRangeSample,
    Vector3,
)
from dronedream_agent_core.realtime_feature_encoders import (
    FlightStateEncoder,
    FlightStateSample,
    encode_dynamic_targets,
    encode_metric_geometry,
    fuse_realtime_features,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "maximum_ms": max(values),
    }


def _mount() -> CalibratedRangeSensorMount:
    return CalibratedRangeSensorMount(
        sensor_id="benchmark-lidar",
        translation_body_m=Vector3(x=0.12, y=0.0, z=0.03),
        orientation_body_from_sensor=QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0),
        minimum_range_m=0.2,
        maximum_range_m=30.0,
    )


def _samples() -> list[RawRangeSample]:
    samples: list[RawRangeSample] = []
    for index in range(320):
        angle = 2.0 * math.pi * index / 320.0
        samples.append(
            RawRangeSample(
                direction_sensor=Vector3(
                    x=math.cos(angle),
                    y=math.sin(angle),
                    z=0.05 * math.sin(angle * 3.0),
                ),
                range_m=4.0 + 2.0 * (1.0 + math.sin(angle * 2.0)),
                hit=index % 5 != 0,
                confidence=0.94,
            )
        )
    return samples


def _obstacles() -> list[DynamicObstacleObservation]:
    return [
        DynamicObstacleObservation(
            obstacle_id=f"track-{index}",
            position_m=Vector3(
                x=3.0 + index * 0.4,
                y=-3.0 + (index % 8) * 0.8,
                z=1.2,
            ),
            velocity_mps=Vector3(
                x=-0.2 - index * 0.01,
                y=0.15 if index % 2 else -0.15,
                z=0.0,
            ),
            radius_m=0.35,
            height_m=1.7,
            confidence=0.9,
            age_seconds=0.03,
        )
        for index in range(16)
    ]


def main() -> int:
    args = _arguments()
    if args.iterations < 100:
        raise SystemExit("--iterations must be at least 100")
    if not 16 <= args.warmup < args.iterations:
        raise SystemExit("--warmup must be at least 16 and below --iterations")
    mount = _mount()
    samples = _samples()
    obstacles = _obstacles()
    orientation = QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0)
    position = Vector3(x=0.0, y=0.0, z=1.2)
    velocity = Vector3(x=0.8, y=0.05, z=0.0)
    encoder = FlightStateEncoder(history_length=16)
    timings: dict[str, list[float]] = {
        "metric_geometry": [],
        "dynamic_targets": [],
        "flight_state": [],
        "feature_fusion": [],
        "combined": [],
    }
    base_unix_ms = int(time.time() * 1_000)
    final_snapshot = None
    for iteration in range(args.iterations):
        observed_at = base_unix_ms + iteration * 10
        scan = RawMetricRangeScan(
            sensor_id=mount.sensor_id,
            sequence=iteration + 1,
            observed_at_unix_ms=observed_at,
            observed_at_monotonic_seconds=iteration * 0.01,
            body_position_world_enu_m=position,
            body_orientation_world_from_body=orientation,
            body_velocity_world_enu_mps=velocity,
            localization_covariance_m2=0.01,
            samples=samples,
        )
        combined_started = time.perf_counter_ns()
        started = time.perf_counter_ns()
        geometry = encode_metric_geometry(
            scan,
            sensor_mount=mount,
            encoded_at_unix_ms=observed_at,
        )
        geometry_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        started = time.perf_counter_ns()
        dynamic = encode_dynamic_targets(
            obstacles,
            body_position_world_enu_m=position,
            body_orientation_world_from_body=orientation,
            body_velocity_world_enu_mps=velocity,
            observed_at_unix_ms=observed_at,
            encoded_at_unix_ms=observed_at,
        )
        dynamic_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        started = time.perf_counter_ns()
        state = encoder.encode(
            FlightStateSample(
                source_id="benchmark-flight-state",
                observed_at_unix_ms=observed_at,
                orientation_world_from_body=orientation,
                velocity_world_enu_mps=velocity,
                acceleration_body_mps2=Vector3(x=0.1, y=0.0, z=0.0),
                angular_velocity_body_rad_s=Vector3(x=0.0, y=0.0, z=0.05),
                localization_covariance_m2=0.01,
            ),
            encoded_at_unix_ms=observed_at,
        )
        state_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        started = time.perf_counter_ns()
        final_snapshot = fuse_realtime_features(
            (geometry, dynamic, state),
            captured_at_unix_ms=observed_at,
        )
        fusion_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        combined_ms = (time.perf_counter_ns() - combined_started) / 1_000_000.0
        if iteration >= args.warmup:
            timings["metric_geometry"].append(geometry_ms)
            timings["dynamic_targets"].append(dynamic_ms)
            timings["flight_state"].append(state_ms)
            timings["feature_fusion"].append(fusion_ms)
            timings["combined"].append(combined_ms)
    if final_snapshot is None or not final_snapshot.ready_for_control:
        raise SystemExit("realtime feature pipeline did not reach control readiness")
    metrics = {name: _summary(values) for name, values in timings.items()}
    qualification = {
        "combined_p99_limit_ms": 10.0,
        "combined_maximum_limit_ms": 100.0,
        "passed": (
            metrics["combined"]["p99_ms"] <= 10.0
            and metrics["combined"]["maximum_ms"] <= 100.0
        ),
    }
    receipt = {
        "schema_version": "dronedream.realtime-feature-benchmark.v1",
        "recorded_at_unix_ms": int(time.time() * 1_000),
        "python": sys.version,
        "platform": platform.platform(),
        "measured_iterations": args.iterations - args.warmup,
        "warmup_iterations": args.warmup,
        "input_contract": {
            "range_samples_per_scan": len(samples),
            "dynamic_tracks_per_tick": len(obstacles),
            "flight_state_history": 16,
            "fused_feature_width": len(final_snapshot.fused_features),
            "fused_valid_mask_width": len(final_snapshot.fused_valid_mask),
        },
        "latency": metrics,
        "qualification": qualification,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if qualification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
