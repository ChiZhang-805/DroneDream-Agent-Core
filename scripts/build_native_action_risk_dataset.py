#!/usr/bin/env python3
"""Generate offline action-risk probes; never opens a flight control channel."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.action_risk_dataset import (
    bounded_action_probes,
    counterfactual_risk_samples,
)
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.dagger_artifacts import write_rows
from dronedream_agent_core.training.mission_groups import MissionGroupManifest
from dronedream_agent_core.training.stream_episode import load_grounded_imitation_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--maximum-observations-per-episode", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.maximum_observations_per_episode <= 256:
        parser.error("observation bound must be 1..256")
    if len({str(path.resolve()) for path in args.episode}) != len(args.episode):
        parser.error("episode repeated")
    if len(args.episode) * args.maximum_observations_per_episode > 128:
        parser.error("one dataset invocation retains at most 128 sampled observations")
    teacher_config = CounterfactualConfig.model_validate_json(args.teacher_config.read_bytes())
    actions = bounded_action_probes()
    args.output.mkdir(parents=True, exist_ok=False)
    labels, records, groups, source_receipts, counterfactuals, observations = [], [], {}, [], [], []
    for path in args.episode:
        episode = load_grounded_imitation_episode(path, teacher_config)
        available = episode.visits
        count = min(args.maximum_observations_per_episode, len(available))
        indices = (
            sorted({round(i * (len(available) - 1) / (count - 1)) for i in range(count)})
            if count > 1
            else [0]
        )
        for index in indices:
            observation = available[index].observation
            observations.append(observation)
            stream = observation.sample.temporal_evidence.stream_id
            previous = groups.setdefault(stream, episode.mission_group_sha256)
            if previous != episode.mission_group_sha256:
                raise ValueError("ACTION_RISK_STREAM_GROUP_CONFLICT")
            for label, record in counterfactual_risk_samples(
                observation, actions, episode.oracle.risk
            ):
                labels.append(label)
                records.append(record)
        counterfactuals.extend(episode.oracle.receipts)
        episode.verify_sources(path)
        source_receipts.append(
            {
                "episode_id": path.name,
                "native_record_kind": episode.native_record_kind,
                "mission_group_sha256": episode.mission_group_sha256,
                "mission_split": episode.mission_split.model_dump(mode="json"),
                "asset_sha256": episode.config.asset_sha256,
                "teacher_geometry_sha256": episode.oracle.teacher.geometry_sha256,
                "vehicle_envelope_sha256": sha256_json(asdict(episode.oracle.teacher.envelope)),
                "source_files_sha256": episode.source_files_sha256,
                "selected_sequences": [available[i].observation.sequence for i in indices],
            }
        )
    hashes = {
        "action-risk": write_rows(args.output / "action-risk.jsonl", labels),
        "action-records": write_rows(args.output / "action-records.jsonl", records),
        "counterfactual-receipts": write_rows(
            args.output / "counterfactual-receipts.jsonl", counterfactuals
        ),
        "stream-groups": write_rows(args.output / "stream-groups.json", [MissionGroupManifest(
            groups=groups, evidence=[source["mission_split"] for source in source_receipts])]),
        "source-observations": write_rows(args.output / "source-observations.jsonl", observations),
    }
    receipt = {
        "purpose": "grounded-native-action-risk-supervision",
        "teacher_config": teacher_config.model_dump(),
        "probe_actions_sha256": sha256_json(actions),
        "source_receipts": source_receipts,
        "file_sha256": hashes,
        "sample_count": len(labels),
        "probe_count_per_observation": len(actions),
        "class_counts": dict(
            Counter("unsafe" if row.risk_target >= 0.5 else "safe" for row in labels)
        ),
        "counterfactual_source": "nominal-swept-geometry-not-physical-replay",
        "qualified_for_flight": False,
        "behavior_cloning_dataset": False,
    }
    write_rows(args.output / "dataset-receipt.jsonl", [receipt])
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
