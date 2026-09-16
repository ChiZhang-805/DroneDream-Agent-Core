#!/usr/bin/env python3
"""Offline simulation refinement; creates new candidate evidence, never promotion.

An explicit adapter factory is required. It owns simulator lifecycle, independent
sensor/command/verifier receipts and quiescing the flight between rollouts. No
dummy environment or real-aircraft fallback is selected automatically.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.runtime_scheduling import (
    InterpreterPauseMonitor,
    retained_interpreter_baseline,
)
from dronedream_agent_core.training.causal_policy import (
    FrozenCausalEvaluation,
    causal_examples,
    export_causal_policy,
    load_causal_checkpoint,
    save_causal_checkpoint,
)
from dronedream_agent_core.training.causal_ppo import CausalPPOTrainer
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    REPLAY_FILES,
    decode_replay,
    prepare_rollout_config,
    protected_validation_groups,
    read_bound_replay,
    require_unseen_validation,
)
from dronedream_agent_core.training.evidence_files import (
    MAX_TRAINING_ROW_BYTES,
    training_json_value,
)
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)
from dronedream_agent_core.training.executed_control import NATIVE_ACTION_COORDINATE_FRAME
from dronedream_agent_core.training.flight_environment import RewardConfig
from dronedream_agent_core.training.mission_groups import MissionGroupEvidence
from dronedream_agent_core.training.ppo import PPOConfig
from dronedream_agent_core.training.visual_lineage import (
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
)
from dronedream_plugin_sdk.protocol import decode_json, encode_json


# 功能：
#   1. 严查基座、视觉、回放及留出来源，构造冻结编码器的因果 PPO 训练器。
#   2. 大型解码对象只在此函数内存在，返回后仅保留张量和原始字节，降低实时 GC 扫描量。
# 输入：
#   args：当前架构及五项回放路径的配置。
#   config：PPO 优化参数。
#   reward_config：独立证据奖励参数。
#   base_content：基座检查点的不可变字节。
#   receipt_content：绑定该检查点的训练回执字节。
# 输出：
#   trainer：已编译回放、冻结留出评价和完整来源身份的训练器。
def build_trainer(args, config, reward_config, *, base_content, receipt_content):
    if args.architecture != "causal-gru-control":
        raise ValueError("NATIVE_PPO_REQUIRES_CURRENT_CAUSAL_ARCHITECTURE")
    base = load_causal_checkpoint(base_content)
    receipt = decode_json(receipt_content, limit=4 * 1024 * 1024)
    if type(receipt) is not dict:
        raise ValueError("CAUSAL_REFINEMENT_RECEIPT_NOT_OBJECT")
    visual = verify_causal_checkpoint_receipt(
        base_content, receipt_content, expert_role=receipt.get("expert_role")
    )
    if receipt["config"] != base.config.model_dump():
        raise ValueError("CAUSAL_REFINEMENT_CHECKPOINT_CONFIG_CHANGED")
    contents = read_bound_replay({name: getattr(args, name) for name in REPLAY_FILES}, receipt)
    decoded, manifest, groups = decode_replay(contents)
    require_unseen_validation(receipt, groups[1])
    examples = [
        causal_examples(
            decoded[split + "_replay"],
            navigation_role=receipt["expert_role"],
            stream_groups=manifest.groups,
            history_length=base.config.history_length,
            history_observations=decoded[split + "_observations"],
        )
        for split in ("training", "validation")
    ]
    trainer = CausalPPOTrainer(base, examples[0], config, reward_config)
    trainer.evaluation = FrozenCausalEvaluation.compile(examples[1], base.config)
    trainer.baseline_evaluation = trainer.evaluation.evaluate(base)
    trainer.held_out_groups = groups[1] | protected_validation_groups(receipt)
    trainer.training_groups = set(receipt["metrics"]["training_groups"]) | groups[0]
    trainer.visual_input_contract = visual
    # Bytes are not GC-traversed Python observation graphs. main persists them
    # before starting physics, then releases them; only compiled tensors remain.
    trainer.frozen_replay_contents = contents
    trainer.replay_artifact_sha256 = receipt["replay_artifact_sha256"]
    return trainer


# 功能：
#   把已完成采集的有界记录写入二进制 JSONL，先验单行和全文件预算并拒绝短写。
# 输入：
#   evidence：本任务独占打开的二进制证据流。
#   records：本次已确认完成的暖机或训练步骤记录。
# 输出：
#   None：不返回业务数据。
def append_rollout_records(evidence, records) -> None:
    for record in records:
        if type(record) is not dict:
            raise ValueError("PPO_ROLLOUT_EVIDENCE_NOT_OBJECT")
        content = (
            encode_json(
                training_json_value(record), limit=MAX_TRAINING_ROW_BYTES - 1, node_limit=1_000_000
            )
            + "\n"
        ).encode("utf-8")
        if evidence.tell() + len(content) > 512 * 1024 * 1024:
            raise ValueError("PPO_ROLLOUT_EVIDENCE_BYTE_LIMIT")
        if evidence.write(content) != len(content):
            raise OSError("PPO_ROLLOUT_EVIDENCE_SHORT_WRITE")


# 功能：
#   解析明确的仿真精调参数，限制更新次数后调用无自动晋升的训练入口。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：全部精调产物、关闭流程和最终回执成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--base-training-receipt", type=Path, required=True)
    parser.add_argument("--architecture", required=True, choices=("causal-gru-control",))
    parser.add_argument("--training-replay", type=Path, required=True)
    parser.add_argument("--validation-replay", type=Path, required=True)
    parser.add_argument("--training-observations", type=Path, required=True)
    parser.add_argument("--validation-observations", type=Path, required=True)
    parser.add_argument("--stream-groups", type=Path, required=True)
    parser.add_argument("--environment-factory", required=True, help="Importable module:function")
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.updates <= 100_000:
        parser.error("updates must be in [1, 100000]")
    exit_code = refine(args)
    return exit_code


# 功能：
#   1. 读取固定来源、建立仿真环境，按采集、确认停止、优化和保存检查点的顺序精调。
#   2. 最终关闭成功后才发布训练回执，清理失败不掩盖首个训练或落地异常。
# 输入：
#   args：已通过命令行更新次数限制的精调参数。
# 输出：
#   exit_code：完整候选结果成功写入时为零，候选不自动获得飞行资格。
def refine(args) -> int:
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    factory_parts = args.environment_factory.split(":")
    if (
        len(factory_parts) != 2
        or not factory_parts[1].isidentifier()
        or not all(part.isidentifier() for part in factory_parts[0].split("."))
    ):
        raise ValueError("PPO_ENVIRONMENT_FACTORY_INVALID")
    config = PPOConfig.model_validate(
        decode_json(read_plugin_file(args.config, limit=4 * 1024 * 1024), limit=4 * 1024 * 1024)
    )
    reward_config = (
        RewardConfig.model_validate(
            decode_json(
                read_plugin_file(args.reward_config, limit=4 * 1024 * 1024), limit=4 * 1024 * 1024
            )
        )
        if args.reward_config
        else RewardConfig()
    )
    base_content = read_plugin_file(args.base_policy, limit=256 * 1024 * 1024)
    base_receipt_content = read_plugin_file(args.base_training_receipt, limit=4 * 1024 * 1024)
    trainer = build_trainer(
        args, config, reward_config, base_content=base_content, receipt_content=base_receipt_content
    )
    visual_contract = trainer.visual_input_contract
    environment_content = read_plugin_file(args.environment_config, limit=4 * 1024 * 1024)
    env_config, mission_split = prepare_rollout_config(
        decode_json(environment_content, limit=4 * 1024 * 1024), trainer.held_out_groups
    )
    trainer.refinement_context_sha256 = sha256_json(
        {
            "base_receipt_sha256": hashlib.sha256(base_receipt_content).hexdigest(),
            "replay_artifact_sha256": trainer.replay_artifact_sha256,
            "environment_config": env_config,
            "mission_split": mission_split.model_dump(),
            "environment_factory": args.environment_factory,
            "applied_action_coordinate_frame": NATIVE_ACTION_COORDINATE_FRAME,
        }
    )
    if args.resume:
        trainer.restore_checkpoint(args.resume)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in trainer.frozen_replay_contents.items():
        publish_evidence_bytes(args.output / REPLAY_FILES[name], content, limit=256 * 1024 * 1024)
    del trainer.frozen_replay_contents
    gc.collect()  # No simulation or motion authority exists at this point.
    module, name = factory_parts
    factory = getattr(importlib.import_module(module), name)
    env = factory(env_config)
    primary_error = None
    try:
        if env.simulation_only is not True or env.evidence_kind != "px4-gazebo":
            raise ValueError("PPO_COMMAND_REQUIRES_PHYSICAL_SIMULATION_ADAPTER")
        if not callable(getattr(env, "quiesce", None)):
            raise ValueError("PPO_ENVIRONMENT_MUST_QUIESCE_BETWEEN_ROLLOUTS")
        actual_split = MissionGroupEvidence.model_validate(env.mission_split.model_dump())
        if actual_split != mission_split:
            raise ValueError("PPO_ENVIRONMENT_SPATIAL_ROUTE_CHANGED")
        require_matching_visual_input(
            visual_contract, env.visual.input_contract if env.visual else None
        )
        with (args.output / "rollouts.jsonl").open("xb") as evidence:
            for _ in range(args.updates):
                # The library returns only after its mandatory quiesce.
                pause_monitor = InterpreterPauseMonitor()
                collection_error = None
                try:
                    rollout = trainer.collect(env)
                except BaseException as error:
                    collection_error = error
                    raise
                finally:
                    # collect has attempted its mandatory quiesce, including
                    # on failure. Only the native receipt can confirm landing;
                    # this diagnostic is not a flight or qualification receipt.
                    try:
                        pauses = pause_monitor.close()
                        write_evidence_object(
                            args.output / f"learner-pauses-{trainer.updates:06d}.json", pauses
                        )
                    except BaseException as cleanup_error:
                        if collection_error is None:
                            raise
                        collection_error.add_note(
                            "PPO pause diagnostics failed: " + repr(cleanup_error)
                        )
                append_rollout_records(evidence, getattr(trainer, "warmup_records", []))
                append_rollout_records(evidence, rollout.records)
                evidence.flush()
                os.fsync(evidence.fileno())
                metrics = trainer.update(rollout)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                trainer.save_checkpoint(args.output / f"checkpoint-{trainer.updates:06d}.pt")
        model = trainer.actor.deployment_model()
        validation = trainer.evaluation.evaluate(model)
        path = args.output / f"{trainer.expert_role}.onnx"
        digest = export_causal_policy(model, path)
        save_causal_checkpoint(model, path.with_suffix(".pt"))
        receipt = {
            "artifact_sha256": digest,
            "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "base_sha256": hashlib.sha256(base_content).hexdigest(),
            "checkpoint_sha256": hash_plugin_file(path.with_suffix(".pt"), limit=256 * 1024 * 1024),
            "base_training_receipt_sha256": hashlib.sha256(base_receipt_content).hexdigest(),
            "visual_input_contract": visual_contract,
            "split_contract": CAUSAL_SPLIT_CONTRACT,
            "replay_artifact_sha256": trainer.replay_artifact_sha256,
            "refinement_context_sha256": trainer.refinement_context_sha256,
            "baseline_validation": trainer.baseline_evaluation,
            "metrics": {
                **validation,
                "validation_window_groups": validation["validation_groups"],
                "validation_groups": sorted(trainer.held_out_groups),
                "training_window_count": len(trainer.replay.inputs),
                "training_groups": sorted(trainer.training_groups | {mission_split.group_sha256}),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            "input_sha256": {
                **trainer.replay_artifact_sha256,
                "environment_config": hashlib.sha256(environment_content).hexdigest(),
                "rollout_evidence": hash_plugin_file(
                    args.output / "rollouts.jsonl", limit=512 * 1024 * 1024
                ),
            },
            "rollout_mission_split": mission_split.model_dump(),
            "replay_hash": trainer.replay_hash,
            "config": model.config.model_dump(),
            "ppo_config": config.model_dump(),
            "reward_config": trainer.reward_config.model_dump(),
            "updates": trainer.updates,
            "evidence_kind": env.evidence_kind,
            "qualified_for_flight": False,
            "expert_role": trainer.expert_role,
            "architecture": args.architecture,
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            env.close()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note("PPO environment cleanup failed: " + repr(cleanup_error))
    write_evidence_object(args.output / "training-receipt.json", receipt)
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    with retained_interpreter_baseline():
        raise SystemExit(main())
