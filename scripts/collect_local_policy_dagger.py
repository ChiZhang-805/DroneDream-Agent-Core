#!/usr/bin/env python3
"""Run a bounded native student flight, ground it, then produce corrections.

No optimizer, teacher search or counterfactual evaluation runs while airborne.
The explicit output directory is new; no model is promoted or installed here.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_scheduling import (
    InterpreterPauseMonitor,
    retained_interpreter_baseline,
)
from dronedream_agent_core.training.causal_policy import load_causal_checkpoint
from dronedream_agent_core.training.causal_replay import (
    prepare_rollout_config,
    protected_validation_groups,
)
from dronedream_agent_core.training.causal_student import OnnxCausalStudent
from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads
from dronedream_agent_core.training.counterfactual_teacher import (
    CounterfactualConfig,
)
from dronedream_agent_core.training.dagger_artifacts import write_rows, write_training_artifacts
from dronedream_agent_core.training.evidence_files import read_evidence_object
from dronedream_agent_core.training.px4_environment import (
    Px4GazeboTrainingEnvironment,
    Px4TrainingConfig,
    require_visual_control_source,
)
from dronedream_agent_core.training.stream_collection import (
    collect_native_control_stream,
    label_stream_visits,
)
from dronedream_agent_core.training.stream_episode import load_grounded_stream_episode
from dronedream_agent_core.training.visual_lineage import (
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
)
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   解析有界学生采集参数，在单线程 CPU 作用域内运行，结束后恢复调用方线程设置。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：采集、落地确认、离线标注及关闭均成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument(
        "--policy-onnx",
        type=Path,
        required=True,
        help="Actual deployment graph bound by the base training receipt",
    )
    parser.add_argument("--base-training-receipt", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=805)
    args = parser.parse_args()
    if not 1 <= args.steps <= 256 or not 0 <= args.seed <= 2**63 - 1:
        parser.error("steps must be 1..256; seed must be in [0, 2**63-1]")
    with training_cpu_threads(1):
        exit_code = collect_and_annotate(args)
    return exit_code


# 功能：
#   1. 在启动环境前绑定当前配置、留出路线、检查点及实际 ONNX 学生策略。
#   2. 只在已落地采集返回后运行教师，环境关闭成功后才写出完成回执。
# 输入：
#   args：经命令行验证的采集参数与新输出目录。
# 输出：
#   exit_code：整条采集标注链路完成时为零，不授予产品飞行资格。
def collect_and_annotate(args) -> int:
    args.output = args.output.absolute()
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    environment_payload, environment_sha = read_evidence_object(
        args.environment_config, limit=4 * 1024 * 1024
    )
    teacher_payload, teacher_sha = read_evidence_object(args.teacher_config, limit=4 * 1024 * 1024)
    config = Px4TrainingConfig.model_validate(environment_payload)
    require_visual_control_source(config)
    teacher_config = CounterfactualConfig.model_validate(teacher_payload)
    policy_bytes = read_plugin_file(args.base_policy, limit=256 * 1024 * 1024)
    training_receipt = read_plugin_file(args.base_training_receipt, limit=4 * 1024 * 1024)
    visual_contract = verify_causal_checkpoint_receipt(
        policy_bytes, training_receipt, expert_role=config.expert_role
    )
    bound_config, split = prepare_rollout_config(
        config.model_dump(mode="json"),
        protected_validation_groups(decode_json(training_receipt, limit=4 * 1024 * 1024)),
    )
    config = Px4TrainingConfig.model_validate(bound_config)
    model = load_causal_checkpoint(policy_bytes)
    student = OnnxCausalStudent(
        model,
        expert_role=config.expert_role,
        onnx_content=read_plugin_file(args.policy_onnx, limit=256 * 1024 * 1024),
        receipt_content=training_receipt,
    )
    policy_sha = student.artifact_sha256
    args.output.mkdir(parents=True, exist_ok=False)
    env = Px4GazeboTrainingEnvironment(config)
    primary_error = None
    try:
        require_matching_visual_input(
            visual_contract, env.visual.input_contract if env.visual else None
        )
        if model.config.visual_feature_count != (
            env.visual.manifest.visual_feature_count if env.visual else 0
        ):
            raise ValueError("DAGGER_STUDENT_VISUAL_ENCODER_WIDTH_MISMATCH")
        gc.collect()
        visits = collect_visits(env, student, config, args)
        # The collection API returns only after independent native landing.
        # Preserve raw visits before annotation, including if no correction is
        # feasible. A failed teacher must not erase actual student evidence.
        visit_sha = write_rows(
            args.output / "student-visits.jsonl",
            (
                {
                    "observation": v.observation.model_dump(mode="json"),
                    "proposal": v.proposal.model_dump(mode="json"),
                    "applied_action": v.applied_action.model_dump(mode="json"),
                    "application_sha256": v.application_sha256,
                    "not_a_reward_transition": True,
                }
                for v in visits
            ),
        )
        episode = load_grounded_stream_episode(env.episode_path, teacher_config)
        if list(episode.visits) != visits:
            raise ValueError("STREAM_COLLECTION_GROUNDED_READBACK_MISMATCH")
        if episode.mission_split != split:
            raise ValueError("STREAM_COLLECTION_ROUTE_CHANGED")
        result = annotate_visits(visits, episode.oracle, config, args.output)
        episode.verify_sources(env.episode_path)
        hashes = {
            "student-visits": visit_sha,
            **write_training_artifacts(
                args.output,
                visits,
                result,
                episode_groups={visit.observation.episode_id: split for visit in visits},
            ),
        }
        receipt = {
            "purpose": "grounded-native-stream-annotation",
            "policy_sha256": policy_sha,
            "base_checkpoint_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "student_inference_backend": "onnxruntime-cpu",
            "startup_onnx_maximum_absolute_error": student.startup_maximum_absolute_error,
            "base_training_receipt_sha256": hashlib.sha256(training_receipt).hexdigest(),
            "expert_role": config.expert_role,
            "evidence_kind": env.evidence_kind,
            "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "environment_config_sha256": sha256_json(config),
            "environment_input_sha256": environment_sha,
            "teacher_config_sha256": teacher_sha,
            "teacher_config": teacher_config.model_dump(mode="json"),
            "visual_encoder_sha256": env.visual.sha256 if env.visual else None,
            "visual_input_contract": env.visual.input_contract if env.visual else None,
            "file_sha256": hashes,
            "student_steps": len(visits),
            "collection_schedule": "source-driven; acceptance joined after landing",
            "not_a_reward_transition": True,
            "source_files_sha256": episode.source_files_sha256,
            "applied_motion_steps": sum(v.applied_action.mode == "pilot-control" for v in visits),
            "safety_interventions": sum(v.safety_intervened for v in visits),
            "qualified_for_flight": False,
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
            primary_error.add_note("DAgger environment cleanup failed: " + repr(cleanup_error))
    write_rows(args.output / "collection-receipt.jsonl", [receipt])
    print(json.dumps(receipt, allow_nan=False), flush=True)
    exit_code = 0
    return exit_code


# 功能：
#   运行来源驱动的有界采集并保存暂停诊断，诊断错误不能盖住原采集或落地失败。
# 输入：
#   env：具有停稳和真实执行回执接口的原生仿真环境。
#   student：绑定实际图摘要的当前 ONNX 操纵策略。
#   config：已核对来源和留出隔离的环境配置。
#   args：采集步数、种子及证据目录。
# 输出：
#   visits：底层确认落地后返回的实际访问记录。
def collect_visits(env, student, config, args):
    monitor = InterpreterPauseMonitor()
    primary_error = None
    try:
        visits = collect_native_control_stream(
            env,
            student,
            seed=args.seed,
            steps=args.steps,
            held_out_missions=set(config.held_out_missions),
            student_policy_sha256=student.artifact_sha256,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            write_rows(args.output / "learner-pauses.jsonl", [monitor.close()])
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note("DAgger pause diagnostics failed: " + repr(cleanup_error))
    return visits


# 功能：
#   落地后生成教师纠正及原提案风险标签，保留教师诊断且不掩盖主要标注异常。
# 输入：
#   visits：与实际执行结果绑定的学生访问。
#   oracle：仅用于离线计算的当前回合教师。
#   config：含留出任务集合的采集配置。
#   output：新建且独占使用的输出目录。
# 输出：
#   result：行为纠正、风险样本和纠正记录。
def annotate_visits(visits, oracle, config, output):
    primary_error = None
    try:
        result = label_stream_visits(
            visits,
            oracle.correction,
            oracle.risk,
            held_out_missions=set(config.held_out_missions),
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            write_rows(output / "counterfactual-receipts.jsonl", oracle.receipts)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note("DAgger teacher evidence failed: " + repr(cleanup_error))
    return result


if __name__ == "__main__":
    # Retain only the finite imported startup graph, before any aircraft,
    # observations or episode history exists. New cyclic garbage still collects.
    with retained_interpreter_baseline():
        raise SystemExit(main())
