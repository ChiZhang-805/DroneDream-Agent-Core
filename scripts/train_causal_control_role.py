#!/usr/bin/env python3
"""Train one fresh control expert for simulation development, without a fake package.

The ensemble assembler separately requires all qualified roles. This command
cannot promote a single learned expert into a complete aircraft model pack.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_expert_harness import NAVIGATION_EXPERT_ROLES
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.causal_policy import (
    causal_examples,
    export_causal_policy,
    load_causal_checkpoint,
    save_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    bind_source_group_metrics,
    export_replay_bundle,
    require_unseen_validation,
    validate_split_sources,
)
from dronedream_agent_core.training.causal_training_inputs import (
    SOURCE_TO_REPLAY,
    read_causal_training_inputs,
    training_cpu_threads,
)
from dronedream_agent_core.training.dagger_artifacts import (
    load_dagger_training_artifacts,
)
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.mission_groups import (
    merge_mission_group_manifests,
)
from dronedream_agent_core.training.visual_lineage import (
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
    verify_visual_training_inputs,
)
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   解析单专家训练参数，在限定 CPU 线程作用域内执行训练，结束后恢复调用方设置。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：完成时为零，非法配置或训练失败直接抛出错误。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "validation",
        "training-observations",
        "validation-observations",
        "stream-groups",
        "config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expert-role", choices=NAVIGATION_EXPERT_ROLES, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--training-visual-receipt", type=Path)
    parser.add_argument("--validation-visual-receipt", type=Path)
    parser.add_argument("--base-training-receipt", type=Path)
    parser.add_argument(
        "--base-policy",
        type=Path,
        help="Refine these current causal weights without modifying the base",
    )
    parser.add_argument(
        "--dagger-dataset",
        type=Path,
        action="append",
        default=[],
        help="Grounded, content-bound corrections to aggregate with base replay",
    )
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 8:
        parser.error("cpu threads must be in [1, 8]")
    with training_cpu_threads(args.cpu_threads):
        exit_code = train_role(args)
    return exit_code


# 功能：
#   1. 验证当前因果样本、历史、视觉和热启动来源，禁止留出泄漏或重复监督。
#   2. 训练单个操纵专家，独占导出权重、回放和完整回执，不授予飞行或整包资格。
# 输入：
#   args：由命令行解析器验证过的单专家训练选项。
# 输出：
#   exit_code：全部产物和回执写入成功时为零。
def train_role(args) -> int:
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.dagger_dataset) > 128:
        raise ValueError("DAGGER_DATASET_LIST_TOO_LARGE")
    # Parse, train and export replay from one immutable read of each input.
    # A file edited during training cannot silently change the receipt/replay.
    sources, config, decoded, group_manifest, _ = read_causal_training_inputs(
        {name: getattr(args, name) for name in (*SOURCE_TO_REPLAY, "config")}
    )
    receipt_contents = [
        read_plugin_file(path, limit=4 * 1024 * 1024)
        for path in (args.training_visual_receipt, args.validation_visual_receipt)
        if path is not None
    ]
    visual_input_contract = verify_visual_training_inputs(
        training_content=sources["train"],
        validation_content=sources["validation"],
        receipt_contents=receipt_contents,
        feature_count=config.visual_feature_count,
    )
    groups = group_manifest.groups
    labels_by_split = [decoded[name] for name in ("training_replay", "validation_replay")]
    history_by_split = [
        decoded[name] for name in ("training_observations", "validation_observations")
    ]
    examples = [
        causal_examples(
            labels,
            navigation_role=args.expert_role,
            stream_groups=groups,
            history_length=config.history_length,
            history_observations=history,
        )
        for labels, history in zip(labels_by_split, history_by_split, strict=True)
    ]
    base_content = (
        read_plugin_file(args.base_policy, limit=256 * 1024 * 1024) if args.base_policy else None
    )
    initial = load_causal_checkpoint(base_content) if base_content is not None else None
    base_sha = hashlib.sha256(base_content).hexdigest() if base_content is not None else None
    base_receipt_sha = None
    if base_content is not None:
        if args.base_training_receipt is None:
            raise ValueError("CAUSAL_WARMSTART_REQUIRES_BOUND_TRAINING_RECEIPT")
        raw_receipt = read_plugin_file(args.base_training_receipt, limit=4 * 1024 * 1024)
        prior_visual = verify_causal_checkpoint_receipt(
            base_content, raw_receipt, expert_role=args.expert_role
        )
        require_matching_visual_input(prior_visual, visual_input_contract)
        base_receipt_sha = hashlib.sha256(raw_receipt).hexdigest()
    elif args.base_training_receipt is not None:
        raise ValueError("CAUSAL_WARMSTART_RECEIPT_WITHOUT_CHECKPOINT")
    correction_receipts, additional_labels, additional_history = {}, [], []
    for directory in args.dagger_dataset:
        directory = directory.absolute()
        check_plain_plugin_path(directory)
        identity = str(directory.resolve())
        if identity in correction_receipts:
            raise ValueError("DAGGER_DATASET_LISTED_TWICE")
        labels, history, new_manifest, receipt_sha = load_dagger_training_artifacts(
            directory, expected_visual_input=visual_input_contract
        )
        group_manifest = merge_mission_group_manifests(group_manifest, new_manifest)
        groups = group_manifest.groups
        correction_receipts[identity] = receipt_sha
        examples[0].extend(
            causal_examples(
                labels,
                navigation_role=args.expert_role,
                stream_groups=groups,
                history_length=config.history_length,
                history_observations=history,
            )
        )
        additional_labels.extend(s for s in labels if s.navigation_expert_role == args.expert_role)
        additional_history.extend(history)
    split_groups = validate_split_sources(
        [*labels_by_split[0], *history_by_split[0], *additional_labels, *additional_history],
        [*labels_by_split[1], *history_by_split[1]],
        group_manifest,
    )
    if base_content is not None:
        require_unseen_validation(decode_json(raw_receipt, limit=4 * 1024 * 1024), split_groups[1])
    source_ids = [
        (e.sample.temporal_evidence.stream_id, e.sample.temporal_evidence.sample_sha256)
        for e in examples[0]
    ]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("CAUSAL_TRAINING_AGGREGATION_DUPLICATE_LABEL")
    args.output.mkdir(parents=True, exist_ok=False)
    model, metrics = train_causal_policy(*examples, config, initial_policy=initial)
    metrics = bind_source_group_metrics(metrics, split_groups)
    path = args.output / (args.expert_role + ".onnx")
    digest = export_causal_policy(model, path)
    save_causal_checkpoint(model, path.with_suffix(".pt"))
    replay_hashes = export_replay_bundle(
        args.output,
        training=[s for s in labels_by_split[0] if s.navigation_expert_role == args.expert_role]
        + additional_labels,
        validation=[s for s in labels_by_split[1] if s.navigation_expert_role == args.expert_role],
        training_history=[*history_by_split[0], *additional_history],
        validation_history=history_by_split[1],
        manifest=group_manifest,
    )
    receipt = {
        "architecture": "causal-gru-control",
        "expert_role": args.expert_role,
        "artifact_sha256": digest,
        "config": config.model_dump(),
        "metrics": metrics,
        "split_contract": CAUSAL_SPLIT_CONTRACT,
        "replay_artifact_sha256": replay_hashes,
        "checkpoint_sha256": hash_plugin_file(path.with_suffix(".pt"), limit=256 * 1024 * 1024),
        "base_training_receipt_sha256": base_receipt_sha,
        "visual_input_contract": visual_input_contract,
        "visual_encoding_receipts_sha256": [
            hashlib.sha256(c).hexdigest() for c in receipt_contents
        ],
        "initial_policy_sha256": base_sha,
        "dagger_receipt_sha256": correction_receipts,
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "input_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in sources.items()
        },
        "qualified_for_flight": False,
        "complete_ensemble": False,
    }
    write_evidence_object(args.output / "training-receipt.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
