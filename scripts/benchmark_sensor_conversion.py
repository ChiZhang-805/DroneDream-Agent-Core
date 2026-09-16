"""Reproducible analytic sensor benchmark; no simulator or actuator access.

The historical point lattice exists only as a mathematical comparison here.
It is never an alternate runtime implementation or a flight success metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import struct
import time
from dataclasses import asdict
from pathlib import Path

from dronedream_agent_core.depth_projection import (
    DepthProjectionCalibration,
    metric_depth_sample_stride,
    project_metric_depth_frame,
)


def run(repetitions: int = 200) -> dict:
    results = []
    for width, height in ((160, 120), (320, 240), (640, 480)):
        calibration = DepthProjectionCalibration(width=width, height=height,
            horizontal_fov_rad=1.274, minimum_depth_m=.2, maximum_depth_m=19.1,
            sample_stride_pixels=metric_depth_sample_stride(width=width, height=height),
            no_return_mode="gazebo-far-clip")
        # Fixed heterogeneous input: valid finite depth, clipping and missing
        # data. This measures conversion, not inference, transport or rendering.
        payload = b"".join(struct.pack("<f", value) for value in (
            2. + (i % 41)*.1 if i % 29 else (math.inf if i % 2 else math.nan)
            for i in range(width*height)))
        for _ in range(20):
            project_metric_depth_frame(data=payload, row_step_bytes=width*4,
                                       calibration=calibration)
        elapsed = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            projected = project_metric_depth_frame(data=payload, row_step_bytes=width*4,
                                                   calibration=calibration)
            elapsed.append((time.perf_counter_ns() - started)/1e6)
        ordered = sorted(elapsed)
        results.append({"width": width, "height": height, "repetitions": repetitions,
            "calibration": asdict(calibration), "calibration_sha256": calibration.sha256,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "source_coverage": projected.source_coverage, "rays": len(projected.samples),
            "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(.95*len(ordered))-1],
            "p99_ms": ordered[math.ceil(.99*len(ordered))-1], "max_ms": max(ordered)})
    phases, detected, lattice_detected, max_error = 0, 0, 0, 0.
    calibration = DepthProjectionCalibration(width=160, height=120, horizontal_fov_rad=1.274,
        minimum_depth_m=.2, maximum_depth_m=19.1, sample_stride_pixels=8,
        no_return_mode="gazebo-far-clip")
    focal = 80 / math.tan(1.274/2)
    for row in range(40, 48):
        for col in range(64, 72):
            values = [8.] * 19_200
            values[row*160+col] = 1.25
            projected = project_metric_depth_frame(data=struct.pack("<19200f", *values),
                row_step_bytes=640, calibration=calibration)
            near = [s for s in projected.samples if s.hit and s.range_m < 2.]
            phases += 1
            detected += int(len(near) == 1)
            lattice_detected += int(row % 8 == 4 and col % 8 == 4)
            expected = (1.25, (79.5-col)*1.25/focal, (59.5-row)*1.25/focal)
            if near:
                sample = near[0]
                direction = tuple(sample.direction_sensor.model_dump().values())
                norm = math.sqrt(sum(v*v for v in direction))
                actual = tuple(sample.range_m*v/norm for v in direction)
                max_error = max(max_error, math.dist(expected, actual))
    passed = detected == phases and max_error < 1e-5 and results[0]["p99_ms"] < 20.
    return {"kind": "analytic-sensor-conversion-benchmark", "python": platform.python_version(),
        "platform": platform.platform(), "input_kind": "synthetic-known-depth-not-flight",
        "projection_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1]
            / "src/dronedream_agent_core/depth_projection.py").read_bytes()).hexdigest(),
        "depth_profiles": results, "thin_obstacle_phases": phases,
        "thin_obstacle_retained": detected, "point_lattice_comparison_retained": lattice_detected,
        "maximum_point_error_m": max_error, "acceptance_passed": passed,
        "flight_qualification_granted": False, "hardware_accuracy_measured": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if not 100 <= args.repetitions <= 2000:
        parser.error("repetitions must be between 100 and 2000")
    if args.output.exists():
        parser.error("output exists; preserve the previous measurement")
    result = run(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
