"""Measure adaptive deadline arithmetic only, not camera-to-actuator latency."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

from dronedream_agent_core import observation_validity as module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30_000)
    args = parser.parse_args()
    if not 1000 <= args.samples <= 1_000_000:
        parser.error("samples must be in [1000, 1000000]")
    base = dict(source_observed_at_unix_ms=1000, now_unix_ms=1020,
        inherited_deadline_unix_ms=1250, clearance_margin_m=1., ego_speed_bound_mps=.5,
        obstacle_speed_bound_mps=.2, acceleration_bound_mps2=1., uncertainty_margin_m=.05,
        downstream_reserve_ms=70)
    scenarios = {
        "open-space": {},
        "near-obstacle": dict(clearance_margin_m=.18, ego_speed_bound_mps=1.),
        "crossing-fast": dict(obstacle_speed_bound_mps=4.),
        "fast-turn": dict(angular_speed_bound_rad_s=3.14),
        "qualified-braking-input": dict(margin_kind="swept-free-distance",
            minimum_braking_deceleration_mps2=1.),
        "unknown-braking": dict(margin_kind="swept-free-distance"),
    }
    results = {}
    for name, changes in scenarios.items():
        inputs = {**base, **changes}
        durations = []
        dispositions = Counter()
        for _ in range(200):
            module.observation_validity(**inputs)
        for _ in range(args.samples):
            started = time.perf_counter_ns()
            result = module.observation_validity(**inputs)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
            dispositions[result.disposition] += 1
        durations.sort()
        results[name] = dict(samples=args.samples, p50_ms=durations[len(durations)//2],
            p99_ms=durations[(len(durations)*99+99)//100-1], maximum_ms=durations[-1],
            dispositions=dict(dispositions))
    sources = [Path(__file__), Path(module.__file__)]
    payload = dict(python=platform.python_version(), platform=platform.platform(),
        gc_enabled=gc.isenabled(), scope="deadline arithmetic, allocation and Python call only",
        exclusions=["sensor conversion", "geometry query", "tracking", "model inference",
                    "transport", "flight-controller response", "physical qualification"],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scenarios=results, all_p99_below_one_ms=all(r["p99_ms"] < 1 for r in results.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
