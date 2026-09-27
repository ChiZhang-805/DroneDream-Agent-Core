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
from dronedream_agent_core.training.mission_groups import MissionGroupManifest, merge_mission_group_manifests
from dronedream_agent_core.training.demonstration_readiness import demonstration_readiness


# 功能：
#   在独占暂存目录写入训练、验证及可选测试样本，禁止合并时覆盖流的路线身份。
# 输入：
#   staging：本次私有暂存目录。
#   train：已接纳训练语料。
#   validation：已接纳独立验证语料。
#   test：可选独立测试语料；不参与训练或超参数选择。
#   visual_required：本次构建是否要求图像证据。
#   allow_nonvisual_history：是否明确允许无图的数值历史。
# 输出：
#   receipt：完整数据集回执；视觉编码、风险标签和飞行资格仍明确为否。
def _write_dataset(staging, train, validation, *, test=None, visual_required, allow_nonvisual_history=False):
    corpora = [("training", train), ("validation", validation)]
    if test is not None:
        corpora.append(("test", test))
    group_manifest = merge_mission_group_manifests(
        *(MissionGroupManifest(groups=corpus.stream_groups,
                               evidence=[row["mission_split"] for row in corpus.receipts])
          for _, corpus in corpora)
    )
    outputs = {}
    for name, corpus in corpora:
        sample_digest = write_rows(staging / f"{name}.jsonl", corpus.samples)
        history_digest = write_rows(staging / f"{name}-observations.jsonl", corpus.observations)
        outputs[name] = {
            "sha256": sample_digest,
            "observation_history_sha256": history_digest,
            "observation_count": len(corpus.observations),
            "counts": corpus.counts,
            "causal_coverage": demonstration_readiness(corpus),
            "sources": corpus.receipts,
        }
        if corpus.heading_sources:
            heading_digest = write_rows(
                staging / f"{name}-heading-sources.jsonl", corpus.heading_sources
            )
            outputs[name]["heading_observations_sha256"] = heading_digest
            outputs[name]["heading_observation_count"] = len(corpus.heading_sources)
    group_digest = write_rows(staging / "stream-groups.json", [group_manifest])
    receipt = {
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "label_source": "verified-executed-simulation-teacher-control",
        "visual_required": visual_required,
        "allow_nonvisual_history": allow_nonvisual_history,
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
#   逐对检查训练、验证及可选测试来源，全部文件写完后无覆盖发布，失败不留下半成品。
# 输入：
#   training_runs：训练运行目录列表。
#   validation_runs：独立验证运行目录列表。
#   output：尚不存在的最终输出目录。
#   test_runs：可选独立测试运行列表；显式空列表不是有效测试集。
#   require_visual：是否必须有原始相机证据。
#   allow_nonvisual_history：显式保留缺图历史，缺图帧仍不进入视觉监督。
#   include_heading_sources：显式保存精细及巡航控制分支的原始几何，缺少当前输入时拒绝导出。
# 输出：
#   counts：各实际发布划分的样本计数。
def build_dataset(training_runs, validation_runs, output, *, require_visual, allow_nonvisual_history=False, test_runs=None, include_heading_sources=False):
    output = Path(output).absolute()
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    train = collect_demonstrations(training_runs, require_visual=require_visual,
                                  allow_nonvisual_history=allow_nonvisual_history,
                                  include_heading_sources=include_heading_sources)
    validation = collect_demonstrations(validation_runs, require_visual=require_visual,
                                       allow_nonvisual_history=allow_nonvisual_history,
                                       include_heading_sources=include_heading_sources)
    validate_demonstration_splits(train, validation)
    test = None
    if test_runs is not None:
        if type(test_runs) not in (list, tuple) or not 1 <= len(test_runs) <= 128:
            raise ValueError("DEMONSTRATION_TEST_ROOTS_INVALID")
        test = collect_demonstrations(test_runs, require_visual=require_visual,
                                      allow_nonvisual_history=allow_nonvisual_history,
                                      include_heading_sources=include_heading_sources)
        # 测试与训练不重叠仍不足够：与调参验证集也必须独立。
        validate_demonstration_splits(train, test)
        validate_demonstration_splits(validation, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.staging-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        _write_dataset(staging, train, validation, test=test, visual_required=require_visual,
                       allow_nonvisual_history=allow_nonvisual_history)
        publish_asset_directory(staging, output)
    counts = {
        name: corpus.counts for name, corpus in (("training", train), ("validation", validation))
    }
    if test is not None:
        counts["test"] = test.counts
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
    parser.add_argument("--test-run", type=Path, action="append",
                        help="Optional independent final-evaluation run; never used for training/tuning")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-heading-sources", action="store_true",
                        help="Preserve verified raw navigation and precision observations for composed-heading evaluation")
    parser.add_argument("--allow-nonvisual-history", action="store_true",
                        help="Keep authentic state history without RGB; never invent visual action labels")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Explicit non-visual experiment, not a visual policy dataset",
    )
    args = parser.parse_args()
    counts = build_dataset(
        args.training_run, args.validation_run, args.output, require_visual=not args.geometry_only,
        allow_nonvisual_history=args.allow_nonvisual_history,
        test_runs=args.test_run,
        include_heading_sources=args.include_heading_sources,
    )
    print(json.dumps(counts))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
