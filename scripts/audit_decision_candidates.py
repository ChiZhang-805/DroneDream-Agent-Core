"""Measure actual candidate coverage before spending on training; no generated labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.decision_dataset import file_digest, read_records
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest


# 功能：逐条统计真实候选的缺失、时效和覆盖；不把字段 unknown 变成可行行为。
# 输入：已提取 JSONL、新报告；输出：重算数量/短缺，正式决策样本数始终零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_records(args.candidates)
    counts, sources, histories, flags = Counter(), Counter(), Counter(), Counter()
    identities = set()
    mission_ids, goals, maps = set(), set(), set()
    for row in rows:
        if row.get("schema_version") != "dronedream.decision-candidate.v2":
            raise ValueError("DECISION_CANDIDATE_SCHEMA_INVALID")
        state = DecisionStateV2.model_validate_json(json.dumps(row["state"]))
        identity = decision_digest(state.model_dump(mode="json"))
        if identity != row["sample_id"] or identity in identities:
            raise ValueError("DECISION_CANDIDATE_IDENTITY_OR_DUPLICATE")
        identities.add(identity)
        mission_ids.add(state.mission_id)
        goals.add((state.mission_id, state.goal_id))
        maps.add(state.map_sha256)
        counts.update(row["coverage_issues"])
        histories[len(state.history)] += 1
        for name in ("pose_source", "geometry_source", "route_source"):
            source = getattr(state.frame, name)
            quality = "missing" if source is None else (
                "fresh" if source.fresh(state.frame.observed_at_ms) else "expired")
            sources[name+":"+quality] += 1
        for name in ("local_route_verified", "crossing_obstacle", "persistent_blockage",
                     "caution_required", "can_hold_position", "can_brake", "payload_attached"):
            flags[name+":"+str(getattr(state.frame, name))] += 1
    report = {"schema_version": "dronedream.decision-candidate-quality.v1",
        "source_sha256": file_digest(args.candidates), "unique_candidates": len(identities),
        "mission_count": len(mission_ids), "goal_count": len(goals), "map_count": len(maps),
        "source_quality": dict(sources), "unknown_or_known_flags": dict(flags),
        "history_lengths": dict(histories), "coverage_issues": dict(counts),
        "formal_decision_windows": 0, "gpu_ready": False,
        "required_collection": ["native behavior application and independently verified outcome",
            "observed local route feasibility, not planned-route presence",
            "causal crossing/blockage observations and recovery evidence",
            "independent layouts and locked deployment-distribution evaluation"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
