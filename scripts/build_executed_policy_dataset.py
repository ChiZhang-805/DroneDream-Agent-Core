#!/usr/bin/env python3
"""Build whole-mission BC splits from acknowledged simulation teacher controls."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dronedream_agent_core.asset_package_storage import publish_asset_directory
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.dagger_artifacts import write_rows
from dronedream_agent_core.training.demonstrations import (
    collect_demonstrations,
    validate_demonstration_splits,
)
from dronedream_agent_core.training.mission_groups import MissionGroupManifest


# 功能：
#   在独占暂存目录写入两套样本、历史和空间组，摘要直接取实际写入字节。
# 输入：
#   staging：本次私有暂存目录。
#   train：已接纳训练语料。
#   validation：已接纳独立验证语料。
#   visual_required：本次构建是否要求图像证据。
# 输出：
#   receipt：完整数据集回执；视觉编码、风险标签和飞行资格仍明确为否。
def _write_dataset(staging, train, validation, *, visual_required):
    group_manifest = MissionGroupManifest(
        groups={**train.stream_groups, **validation.stream_groups},
        evidence=[row["mission_split"] for row in [*train.receipts, *validation.receipts]],
    )
    outputs = {}
    for name, corpus in (("training", train), ("validation", validation)):
        sample_digest = write_rows(staging / f"{name}.jsonl", corpus.samples)
        history_digest = write_rows(staging / f"{name}-observations.jsonl", corpus.observations)
        outputs[name] = {
            "sha256": sample_digest,
            "observation_history_sha256": history_digest,
            "observation_count": len(corpus.observations),
            "counts": corpus.counts,
            "sources": corpus.receipts,
        }
    group_digest = write_rows(staging / "stream-groups.json", [group_manifest])
    receipt = {
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "label_source": "verified-executed-simulation-teacher-control",
        "visual_required": visual_required,
        "visual_features_encoded": False,
        "counterfactual_action_risk_labels": False,
        "qualified_for_flight": False,
        "split_method": "map-spatial-route",
        "stream_groups_sha256": group_digest,
        "outputs": outputs,
    }
    write_rows(staging / "dataset-receipt.json", [receipt])
    return receipt


# 功能：
#   校验训练和留出来源，全部文件及回执写完后无覆盖发布，失败不留下半成品目标目录。
# 输入：
#   training_runs：训练运行目录列表。
#   validation_runs：独立验证运行目录列表。
#   output：尚不存在的最终输出目录。
#   require_visual：是否必须有原始相机证据。
# 输出：
#   counts：训练和验证集的实际样本计数。
def build_dataset(training_runs, validation_runs, output, *, require_visual):
    output = Path(output).absolute()
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    train = collect_demonstrations(training_runs, require_visual=require_visual)
    validation = collect_demonstrations(validation_runs, require_visual=require_visual)
    validate_demonstration_splits(train, validation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.staging-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        _write_dataset(staging, train, validation, visual_required=require_visual)
        publish_asset_directory(staging, output)
    counts = {
        name: corpus.counts for name, corpus in (("training", train), ("validation", validation))
    }
    return counts


# 功能：
#   解析明确的教师运行和视觉模式，发布完整数据集后输出计数并正常退出。
# 输入：
#   无；读取命令行参数。
# 输出：
#   exit_code：成功构建后的退出码。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, action="append", required=True)
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Explicit non-visual experiment, not a visual policy dataset",
    )
    args = parser.parse_args()
    counts = build_dataset(
        args.training_run, args.validation_run, args.output, require_visual=not args.geometry_only
    )
    print(json.dumps(counts))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
