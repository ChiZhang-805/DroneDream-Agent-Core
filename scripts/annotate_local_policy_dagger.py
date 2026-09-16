#!/usr/bin/env python3
"""Annotate completed native transitions after confirmed landing; never fly.

A safely aborted collection may contain valid earlier transitions. Preserve
their actual outcomes and clearly label collection completion separately.
"""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.counterfactual_teacher import CounterfactualConfig
from dronedream_agent_core.training.dagger_artifacts import write_rows, write_training_artifacts
from dronedream_agent_core.training.evidence_files import read_evidence_object
from dronedream_agent_core.training.stream_collection import label_stream_visits
from dronedream_agent_core.training.stream_episode import load_grounded_imitation_episode
from dronedream_agent_core.training.student_collection import label_student_visits


# 功能：
#   解析只允许落地后处理的离线标注参数，不提供飞行或模型自动晋升入口。
# 输入：
#   无：参数来自当前命令行。
# 输出：
#   exit_code：标注及证据保存全部成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exit_code = annotate(args)
    return exit_code


# 功能：
#   1. 严格读取教师配置及已落地回合，按读取器确认的记录类型选择监督方式。
#   2. 原始来源复核成功后才保存完成回执，诊断失败不覆盖首个标注异常。
# 输入：
#   args：包含回合、教师配置及新输出目录的已解析参数。
# 输出：
#   exit_code：来源绑定的标注产物全部完成时为零，不代表飞行验收。
def annotate(args) -> int:
    args.episode, args.output = args.episode.absolute(), args.output.absolute()
    check_plain_plugin_path(args.episode)
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    teacher_payload, teacher_sha = read_evidence_object(args.teacher_config, limit=4 * 1024 * 1024)
    teacher_config = CounterfactualConfig.model_validate(teacher_payload)
    episode = load_grounded_imitation_episode(args.episode, teacher_config)
    kind = episode.native_record_kind
    if kind not in {"stream-action", "reward-step"}:
        raise ValueError("DAGGER_ANNOTATION_RECORD_KIND_INVALID")
    # 不重新猜测目录标志文件；类型必须来自已经核验完整来源的读取器。
    streaming = kind == "stream-action"
    reset, reset_sha = read_evidence_object(args.episode / "reset.json", limit=4 * 1024 * 1024)
    if reset_sha != episode.source_files_sha256["reset.json"]:
        raise ValueError("DAGGER_NATIVE_RESET_CHANGED")
    visits, oracle = list(episode.visits), episode.oracle
    args.output.mkdir(parents=True, exist_ok=False)
    primary_error = None
    try:
        kwargs = {"held_out_missions": set(episode.config.held_out_missions)}
        if not streaming:
            kwargs["evidence_kind"] = "px4-gazebo"
        result = (label_stream_visits if streaming else label_student_visits)(
            visits, oracle.correction, oracle.risk, **kwargs
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            write_rows(args.output / "counterfactual-receipts.jsonl", oracle.receipts)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note("DAgger teacher evidence failed: " + repr(cleanup_error))
    episode.verify_sources(args.episode)
    hashes = write_training_artifacts(
        args.output, visits, result, episode_groups={args.episode.name: episode.mission_split}
    )
    receipt = {
        "purpose": (
            "grounded-native-stream-annotation"
            if streaming
            else "grounded-native-transition-annotation"
        ),
        "not_a_reward_transition": streaming,
        "episode": args.episode.name,
        "verified_transition_count": len(visits),
        "file_sha256": hashes,
        "source_files_sha256": episode.source_files_sha256,
        "visual_input_contract": reset.get("visual_input_contract"),
        "teacher_config_sha256": teacher_sha,
        "teacher_config": teacher_config.model_dump(mode="json"),
        "collection_completion_not_asserted": True,
        "qualified_for_flight": False,
    }
    write_rows(args.output / "annotation-receipt.jsonl", [receipt])
    print(json.dumps(receipt, allow_nan=False), flush=True)
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
