"""Extract causal decision candidates from real logs; no automatic expert labels."""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

from dronedream_agent_core.decision_dataset import file_digest
from dronedream_agent_core.decision_runtime_adapter import state_from_navigation_snapshot
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：按真实日志时间降采样并保留同目标历史，统计所有拒绝原因；不制造正确动作。
# 输入：导航快照 JSONL、新目录、标定摘要与明确制动参数；输出：待标注候选和覆盖报告。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--braking-acceleration", type=float, required=True)
    parser.add_argument("--scene", choices=("office", "corridor", "door", "stairs",
                                          "transition", "outdoor", "pickup"), required=True)
    args = parser.parse_args()
    source_hash = file_digest(args.snapshots)
    calibration_hash = file_digest(args.calibration_file)
    args.output.mkdir(parents=True, exist_ok=False)
    counts, issues = Counter(), Counter()
    history = deque(maxlen=10)
    previous_identity = None
    last_time = -1
    with args.snapshots.open("r", encoding="utf-8") as source, \
            (args.output / "candidates.jsonl").open("x", encoding="utf-8") as output:
        while line := source.readline(4 * 1024**2 + 1):
            counts["source_rows"] += 1
            if len(line) > 4 * 1024**2:
                raise ValueError("DECISION_SNAPSHOT_LINE_TOO_LARGE")
            record = json.loads(line)
            snapshot = record["snapshot"]
            now = snapshot["control_reference_observed_at_unix_ms"]
            if now < last_time:
                raise ValueError("DECISION_SOURCE_CLOCK_REVERSED")
            if last_time >= 0 and now - last_time < 200:
                counts["temporal_downsampled"] += 1
                continue
            identity = (snapshot["strategic_context"]["task"]["navigation_goal_id"],
                        snapshot["known_static_map"]["qualified_route_sha256"])
            if identity != previous_identity:
                history.clear()
            while history and now - history[0].observed_at_ms > 2500:
                history.popleft()
            try:
                state, coverage = state_from_navigation_snapshot(snapshot,
                    mission_id="replay-" + source_hash[:16], sequence=counts["source_rows"],
                    calibration_sha256=calibration_hash,
                    braking_acceleration_mps2=args.braking_acceleration, scene=args.scene,
                    history=tuple(history))
            except (ValueError, KeyError, TypeError) as error:
                counts["rejected"] += 1
                issues[type(error).__name__ + ":" + str(error)[:180]] += 1
                continue
            output.write(json.dumps({"schema_version": "dronedream.decision-candidate.v2",
                "sample_id": decision_digest(state.model_dump(mode="json")),
                "state": state.model_dump(mode="json"), "source_sha256": source_hash,
                "source_line": counts["source_rows"], "formal_training_eligible": False,
                "label_status": "awaiting-independent-outcome", "coverage_issues": coverage},
                allow_nan=False) + "\n")
            counts["candidate_windows"] += 1
            issues.update(coverage)
            history.append(state.frame)
            last_time, previous_identity = now, identity
    report = {"schema_version": "dronedream.decision-candidate-inventory.v2",
              "counts": dict(counts), "issues": dict(issues), "formal_decision_windows": 0,
              "source_sha256": source_hash, "gpu_ready": False,
              "calibration_file_sha256": calibration_hash,
              "braking_acceleration_mps2": args.braking_acceleration,
              "parameter_status": "configured-not-measured-braking",
              "calibration_scope": "provided-file-only-not-complete-sensor-qualification"}
    with (args.output / "inventory.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
