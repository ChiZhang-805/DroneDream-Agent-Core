#!/usr/bin/env python3
"""Train and assemble all three temporal control roles; never auto-qualify."""

import argparse
import hashlib
import json
from pathlib import Path

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import NAVIGATION_EXPERT_ROLES
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.training.causal_policy import (
    CausalPolicyConfig,
    causal_examples,
    export_causal_policy,
    save_causal_checkpoint,
    train_causal_policy,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    bind_source_group_metrics,
    export_replay_bundle,
)
from dronedream_agent_core.training.causal_training_inputs import (
    SOURCE_TO_REPLAY,
    read_causal_training_inputs,
    training_cpu_threads,
)
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.policy_packaging import (
    assemble_causal_package,
    validate_causal_base,
)
from dronedream_agent_core.training.visual_lineage import (
    VisualInputContract,
    require_matching_visual_input,
    verify_visual_training_inputs,
)


# 功能：
#   解析三个操纵专家的集合训练参数，在限定 CPU 线程作用域内执行且恢复原设置。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：训练与候选包生成成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--training-observations", type=Path, required=True)
    parser.add_argument("--validation-observations", type=Path, required=True)
    parser.add_argument(
        "--stream-groups",
        type=Path,
        required=True,
        help="Content-bound spatial mission group manifest",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--training-visual-receipt", type=Path)
    parser.add_argument("--validation-visual-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 8:
        parser.error("cpu threads must be in [1, 8]")
    with training_cpu_threads(args.cpu_threads):
        exit_code = train_ensemble(args)
    return exit_code


# 功能：
#   1. 整批验证当前来源、视觉、保留专家及三套训练配置，再开始顺序训练。
#   2. 导出三种操纵权重、来源回放和候选集合包，缺少完整回执不能视为成功。
# 输入：
#   args：命令行解析器提供的集合训练选项。
# 输出：
#   exit_code：全部候选产物及回执成功写入时为零，不代表飞行验收。
def train_ensemble(args) -> int:
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    content, config, decoded, manifest, split_groups = read_causal_training_inputs(
        {name: getattr(args, name) for name in (*SOURCE_TO_REPLAY, "config")}
    )
    # 后一个专家种子越界也必须在第一个专家开始优化之前被发现。
    if config.seed + len(NAVIGATION_EXPERT_ROLES) - 1 > 2**63 - 1:
        raise ValueError("CAUSAL_ENSEMBLE_SEED_INVALID")
    role_configs = {
        role: CausalPolicyConfig.model_validate(
            {**config.model_dump(), "seed": config.seed + index}
        )
        for index, role in enumerate(NAVIGATION_EXPERT_ROLES)
    }
    base = validate_causal_base(args.base_package, visual_feature_count=config.visual_feature_count)
    from dronedream_agent_core.training.ensemble_lineage import load_base_lineage

    load_base_lineage(base)  # Reject unproven retained experts before any optimization.
    receipt_contents = [
        read_plugin_file(path, limit=4 * 1024 * 1024)
        for path in (args.training_visual_receipt, args.validation_visual_receipt)
        if path is not None
    ]
    visual_input = verify_visual_training_inputs(
        training_content=content["train"],
        validation_content=content["validation"],
        receipt_contents=receipt_contents,
        feature_count=config.visual_feature_count,
    )
    if visual_input is not None:
        encoder_sha = next(
            a.sha256 for a in base.manifest.artifacts if a.role == "perception-encoder"
        )
        require_matching_visual_input(
            visual_input, VisualInputContract.from_manifest(base.manifest, encoder_sha).model_dump()
        )
    groups = manifest.groups
    train, validation = [decoded[name] for name in ("training_replay", "validation_replay")]
    training_observations, validation_observations = [
        decoded[name] for name in ("training_observations", "validation_observations")
    ]
    # Validate every role before creating output or starting a long training job.
    examples = {
        role: (
            causal_examples(
                train,
                navigation_role=role,
                stream_groups=groups,
                history_length=config.history_length,
                history_observations=training_observations,
            ),
            causal_examples(
                validation,
                navigation_role=role,
                stream_groups=groups,
                history_length=config.history_length,
                history_observations=validation_observations,
            ),
        )
        for role in NAVIGATION_EXPERT_ROLES
    }
    args.output.mkdir(parents=True, exist_ok=False)
    metrics, paths, receipts = {}, {}, {}
    for role in NAVIGATION_EXPERT_ROLES:
        role_config = role_configs[role]
        model, metrics[role] = train_causal_policy(*examples[role], role_config)
        metrics[role] = bind_source_group_metrics(metrics[role], split_groups)
        path = args.output / "weights" / f"{role}.onnx"
        digest = export_causal_policy(model, path)
        paths[role] = path
        save_causal_checkpoint(model, path.with_suffix(".pt"))
        replay_root = args.output / "replay" / role
        replay_root.mkdir(parents=True, exist_ok=False)
        replay_hashes = export_replay_bundle(
            replay_root,
            training=[s for s in train if s.navigation_expert_role == role],
            validation=[s for s in validation if s.navigation_expert_role == role],
            training_history=training_observations,
            validation_history=validation_observations,
            manifest=manifest,
        )
        receipts[role] = {
            "architecture": "causal-gru-control",
            "expert_role": role,
            "artifact_sha256": digest,
            "checkpoint_sha256": hash_plugin_file(path.with_suffix(".pt"), limit=256 * 1024 * 1024),
            "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "config": role_config.model_dump(),
            "metrics": metrics[role],
            "split_contract": CAUSAL_SPLIT_CONTRACT,
            "replay_artifact_sha256": replay_hashes,
            "input_sha256": {
                name: hashlib.sha256(value).hexdigest() for name, value in content.items()
            },
            "visual_input_contract": visual_input,
            "visual_encoding_receipts_sha256": [
                hashlib.sha256(value).hexdigest() for value in receipt_contents
            ],
            "qualified_for_flight": False,
        }
        write_evidence_object(replay_root / "training-receipt.json", receipts[role])
        print(json.dumps({"role": role, **metrics[role]}, allow_nan=False), flush=True)
    package = assemble_causal_package(
        base_root=args.base_package,
        navigation_models=paths,
        output_root=args.output / "package",
        package_id=args.package_id,
        history_length=config.history_length,
        navigation_receipts=receipts,
    )
    write_evidence_object(
        args.output / "training-receipt.json",
        {
            "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "package_sha256": package.package_sha256,
            "config": config.model_dump(),
            "stream_groups_sha256": sha256_json(groups),
            "metrics": metrics,
            "qualified_for_flight": False,
        },
    )
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
