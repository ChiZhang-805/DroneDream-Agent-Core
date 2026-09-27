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
from dronedream_agent_core.training.causal_device import causal_training_device
from dronedream_agent_core.training.causal_policy import (
    causal_examples,
    export_causal_policy,
    load_causal_checkpoint,
    save_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_regularization import (
    CausalRegularization,
    validate_regularization,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    bind_source_group_metrics,
    export_replay_bundle,
    protected_validation_groups,
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Explicit optimization device; unavailable CUDA fails without CPU fallback")
    parser.add_argument("--training-visual-receipt", type=Path)
    parser.add_argument("--validation-visual-receipt", type=Path)
    parser.add_argument("--regularization-config", type=Path)
    parser.add_argument("--base-training-receipt", type=Path)
    parser.add_argument("--encoder-policy", type=Path,
                        help="Transfer only sensor encoder/history layers; initialize all control heads afresh")
    parser.add_argument("--encoder-training-receipt", type=Path)
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
#   绑定跨专家编码器的实际权重、原角色、视觉契约及历史训练来源，不把源控制头当成目标专家。
# 输入：
#   policy_path、receipt_path：明确的源检查点与其训练回执。
#   expert_role、config：本次目标专家及编码结构。
#   visual_input_contract：本次视觉预处理和权重身份。
#   validation_groups：本次不得被源模型训练接触的留出组。
# 输出：
#   encoder、provenance、receipt：经验证的源模型、迁移来源记录与原始回执对象。
def load_encoder_transfer(policy_path, receipt_path, *, expert_role, config, visual_input_contract, validation_groups):
    if policy_path is None or receipt_path is None:
        raise ValueError("CAUSAL_ENCODER_TRANSFER_REQUIRES_POLICY_AND_RECEIPT")
    content = read_plugin_file(policy_path, limit=256 * 1024 * 1024)
    raw_receipt = read_plugin_file(receipt_path, limit=4 * 1024 * 1024)
    receipt = decode_json(raw_receipt, limit=4 * 1024 * 1024)
    source_role = receipt.get("expert_role") if isinstance(receipt, dict) else None
    if (expert_role not in NAVIGATION_EXPERT_ROLES or source_role not in NAVIGATION_EXPERT_ROLES
            or source_role == expert_role):
        raise ValueError("CAUSAL_ENCODER_TRANSFER_REQUIRES_DISTINCT_EXPERT")
    visual = verify_causal_checkpoint_receipt(content, raw_receipt, expert_role=source_role)
    require_matching_visual_input(visual, visual_input_contract)
    require_unseen_validation(receipt, validation_groups)
    protected_validation_groups(receipt)
    encoder = load_causal_checkpoint(content)
    if receipt.get("config") != encoder.config.model_dump():
        raise ValueError("CAUSAL_ENCODER_TRANSFER_RECEIPT_CONFIG_MISMATCH")
    fields = ("history_length", "encoder_width", "recurrent_width", "visual_feature_count")
    if any(getattr(config, field) != getattr(encoder.config, field) for field in fields):
        raise ValueError("CAUSAL_ENCODER_TRANSFER_ARCHITECTURE_MISMATCH")
    provenance = dict(checkpoint_sha256=hashlib.sha256(content).hexdigest(),
        training_receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),
        source_expert_role=source_role, target_expert_role=expert_role,
        output_heads_copied=False, qualified_for_flight=False)
    return encoder, provenance, receipt


# 功能：
#   累积热启动或编码器迁移曾接触的训练和调参组，防止把祖先路线误当未见最终测试。
# 输入：
#   metrics：本轮已绑定来源的独立指标字典。
#   previous_receipts：经内容绑定的祖先回执列表。
#   validation_groups：当前独立留出组。
# 输出：
#   metrics：保留当前指标并合并历史训练组、单独保存历史调参组的字典。
def inherit_training_provenance(metrics, previous_receipts, validation_groups):
    groups = set(metrics["training_groups"])
    historical_validation = set()
    for receipt in previous_receipts:
        require_unseen_validation(receipt, validation_groups)
        groups.update(receipt["metrics"]["training_groups"])
        historical_validation.update(protected_validation_groups(receipt))
    metrics = {**metrics, "training_groups": sorted(groups),
               "historical_validation_groups": sorted(historical_validation)}
    return metrics


# 功能：
#   1. 验证当前因果样本、历史、视觉和热启动来源，禁止留出泄漏或重复监督。
#   2. 训练单个操纵专家，独占导出权重、回放和完整回执，不授予飞行或整包资格。
# 输入：
#   args：由命令行解析器验证过的单专家训练选项。
# 输出：
#   exit_code：全部产物和回执写入成功时为零。
def train_role(args) -> int:
    # 在读取大型回放或建立输出目录前拒绝不可用设备。
    device = getattr(args, "device", "cpu")
    causal_training_device(device)
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.dagger_dataset) > 128:
        raise ValueError("DAGGER_DATASET_LIST_TOO_LARGE")
    encoder_path = getattr(args, "encoder_policy", None)
    encoder_receipt_path = getattr(args, "encoder_training_receipt", None)
    if (encoder_path is None) != (encoder_receipt_path is None):
        raise ValueError("CAUSAL_ENCODER_TRANSFER_REQUIRES_POLICY_AND_RECEIPT")
    if encoder_path is not None and (args.base_policy is not None or args.base_training_receipt is not None):
        raise ValueError("CAUSAL_INITIALIZATION_MODES_CONFLICT")
    # Parse, train and export replay from one immutable read of each input.
    # A file edited during training cannot silently change the receipt/replay.
    sources, config, decoded, group_manifest, _ = read_causal_training_inputs(
        {name: getattr(args, name) for name in (*SOURCE_TO_REPLAY, "config")}
    )
    regularization = None
    regularization_path = getattr(args, "regularization_config", None)
    if regularization_path is not None:
        raw_regularization = read_plugin_file(regularization_path, limit=64 * 1024)
        regularization = CausalRegularization.model_validate(
            decode_json(raw_regularization, limit=64 * 1024), strict=True)
        sources["regularization_config"] = raw_regularization
    regularization = validate_regularization(regularization, config.visual_feature_count)
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
    previous_receipts = []
    if base_content is not None:
        prior_receipt = decode_json(raw_receipt, limit=4 * 1024 * 1024)
        require_unseen_validation(prior_receipt, split_groups[1])
        protected_validation_groups(prior_receipt)
        previous_receipts.append(prior_receipt)
    encoder, encoder_provenance = None, None
    if encoder_path is not None:
        encoder, encoder_provenance, encoder_receipt = load_encoder_transfer(
            encoder_path, encoder_receipt_path, expert_role=args.expert_role, config=config,
            visual_input_contract=visual_input_contract, validation_groups=split_groups[1])
        previous_receipts.append(encoder_receipt)
    for previous in previous_receipts:
        if protected_validation_groups(previous) & split_groups[0]:
            raise ValueError("CAUSAL_REFINEMENT_TRAINS_ON_PROTECTED_HOLDOUT")
    source_ids = [
        (e.sample.temporal_evidence.stream_id, e.sample.temporal_evidence.sample_sha256)
        for e in examples[0]
    ]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("CAUSAL_TRAINING_AGGREGATION_DUPLICATE_LABEL")
    args.output.mkdir(parents=True, exist_ok=False)
    model, metrics = train_causal_policy(
        *examples, config, initial_policy=initial, initial_encoder=encoder,
        regularization=regularization, device=device)
    metrics = bind_source_group_metrics(metrics, split_groups)
    metrics = inherit_training_provenance(metrics, previous_receipts, split_groups[1])
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
        "encoder_initialization": encoder_provenance,
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
