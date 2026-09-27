#!/usr/bin/env python3
"""Generate offline action-risk probes; never opens a flight control channel."""

import argparse
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.training.action_risk_dataset import (
    bounded_action_probes,
    counterfactual_risk_samples,
)
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.dagger_artifacts import write_rows
from dronedream_agent_core.training.mission_groups import MissionGroupManifest
from dronedream_agent_core.training.stream_episode import load_grounded_imitation_episode
from dronedream_agent_core.training.risk_clearance_contract import OBSERVED_CLEARANCE_LABELS
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   从完整仿真观测构造带动作条件的几何风险标签；这些标签不作为动作模仿答案或飞行证明。
# 输入：
#   命令行参数：有界仿真段、教师配置、每段观测上限和新的输出目录。
# 输出：
#   exit_code：全部标签及来源回执写入成功时为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--maximum-observations-per-episode", type=int, default=16)
    parser.add_argument("--exclude-expired-safety-holds", action="store_true")
    parser.add_argument("--reproduce-legacy-labels", action="store_true",
                        help="Explicit archival reproduction only; do not mix with current labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.maximum_observations_per_episode <= 256:
        parser.error("observation bound must be 1..256")
    if len({str(path.resolve()) for path in args.episode}) != len(args.episode):
        parser.error("episode repeated")
    if len(args.episode) * args.maximum_observations_per_episode > 128:
        parser.error("one dataset invocation retains at most 128 sampled observations")
    teacher_config = CounterfactualConfig.model_validate(
        decode_json(read_plugin_file(args.teacher_config, limit=4 * 1024**2),
                    limit=4 * 1024**2)
    )
    # 新建数据必须由配置显式声明与实际净空条件一致的标签；旧资料仍可读取审计，不能混用。
    if (not args.reproduce_legacy_labels
            and teacher_config.risk_label_semantics != OBSERVED_CLEARANCE_LABELS):
        parser.error("teacher config must declare risk_label_semantics=observation-clearance-v2")
    actions = bounded_action_probes()
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    labels, records, groups, source_receipts, counterfactuals, observations = [], [], {}, [], [], []
    for path in args.episode:
        episode = load_grounded_imitation_episode(path, teacher_config)
        available, excluded = [], []
        # 在任何标签生成之前按物理延迟契约核验全部观测；不依据风险高低挑选样本。
        for visit in episode.visits:
            issue = (episode.oracle.risk_context_issue(visit.observation)
                     if episode.native_record_kind == "stream-action" else None)
            if issue is not None:
                if (not args.exclude_expired_safety_holds or not visit.safety_intervened
                        or visit.applied_action.mode != "hold"):
                    raise ValueError(issue)
                excluded.append({"sequence": visit.observation.sequence,
                    "observation_sha256": sha256_json(visit.observation), "reason": issue,
                    "applied_mode": "hold", "safety_intervened": True})
            else:
                available.append(visit)
        if not available:
            raise ValueError("ACTION_RISK_NO_ELIGIBLE_OBSERVATIONS")
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
            batch = (episode.oracle.fixed_risk_context(observation)
                     if episode.native_record_kind == "stream-action"
                     else nullcontext(episode.oracle.risk))
            with batch as assess:
                evaluated = list(counterfactual_risk_samples(observation, actions, assess))
            for label, record in evaluated:
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
                "excluded_observations": excluded,
                "selection_policy": "uniform-after-latency-contract-not-risk-score",
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
