"""Audit original stage implementations offline; never add formal labels or alter a run."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.decision_label_evidence import (
    behavior_application_matches,
    verify_native_label,
)
from dronedream_agent_core.decision_stage_control import verify_stage_control
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.stream_capture import (
    grounded_control_index,
    require_grounded_stream,
)
from scripts.collect_native_decisions import records


# 功能：固定时长非重叠选择窗口，不搜索有利片段；原控制账本中的所有介入均参与审计。
# 输入：已安全收尾原生回合、新报告路径；输出：混合来源审计与原始摘要，不授予标签。
def audit(episode, output, *, export_root=None, window_ms=3000):
    if type(window_ms) is not int or not 3000 <= window_ms <= 10000:
        raise ValueError("DECISION_STAGE_PARTITION_DURATION_INVALID")
    require_grounded_stream(episode)
    index = decode_object((episode / "decision-selections.json").read_text("utf-8"))
    selections = []
    for reference in index["selections"]:
        path = (episode / reference["path"]).resolve()
        if not path.is_relative_to(episode.resolve()) or file_digest(path) != reference["sha256"]:
            raise ValueError("DECISION_STAGE_SELECTION_SOURCE_INVALID")
        selections.append(decode_object(path.read_text("utf-8")))
    sim = episode / "flight/simulation"
    call_ids = {r["submission"]["call_id"] for r in selections if r.get("submission")}
    grounded_control_index(sim, call_ids)
    if export_root is not None:
        export_root.mkdir(parents=True, exist_ok=False)
    exports = []
    commands = {
        decision_digest(r["command"]): r["command"]
        for r in records(sim / "depth-local-safety-history.jsonl")
        if r.get("command")
    }
    actuals = list(records(sim / "runtime-state/control-applications.jsonl"))
    # 每条命令/执行回执只做一次类型校验；后续按行为分组不反复解码整轮所有复杂对象。
    parsed_commands = {
        digest: RuntimeLocalSafetyCommand.model_validate(c) for digest, c in commands.items()
    }
    parsed_actuals = [(a, ControlApplicationRecord.model_validate(a)) for a in actuals]
    brake_path = sim / "runtime-state/executor-brake-applications.jsonl"
    brakes = list(records(brake_path)) if brake_path.exists() else []
    witnesses = decode_object((episode / "decision-independent-witnesses.json").read_text("utf-8"))[
        "observations"
    ]
    groups, current = [], []
    for selection in selections:
        identity = tuple(selection["state"][k] for k in ("goal_id", "route_sha256")) + (
            selection["action"],
        )
        if current:
            prior = tuple(current[0]["state"][k] for k in ("goal_id", "route_sha256")) + (
                current[0]["action"],
            )
            if (
                identity != prior
                or selection["selected_at_ms"] - current[0]["selected_at_ms"] >= window_ms
            ):
                groups.append(current)
                current = []
        current.append(selection)
    if current:
        groups.append(current)
    results, rejected = [], Counter()
    continuations = []
    for number, group in reversed(list(enumerate(groups))):
        calls = {r["submission"]["call_id"]: r for r in group if r.get("submission")}
        model = [
            a
            for a, actual in parsed_actuals
            if behavior_application_matches(
                parsed_commands[a["command_sha256"]],
                actual,
                group[0]["action"],
            )
            and commands.get(a["command_sha256"], {}).get("model_call_id") in calls
            and group[0]["selected_at_ms"]
            <= a["accepted_at_unix_ms"]
            <= group[-1]["selected_at_ms"] + 250
        ]
        record = dict(
            first_selection=group[0]["state"]["sequence"],
            last_selection=group[-1]["state"]["sequence"],
            verified=False,
        )
        try:
            if not model:
                raise ValueError("DECISION_STAGE_NO_ACTUAL_MODEL_EXECUTION")
            start, end = model[0]["accepted_at_unix_ms"], model[-1]["accepted_at_unix_ms"]
            first = calls[commands[model[0]["command_sha256"]]["model_call_id"]]
            full = [a for a in actuals if start <= a["accepted_at_unix_ms"] <= end]
            if any(
                (
                    a["model_authorized"]
                    or commands[a["command_sha256"]].get("model_authority_reason")
                    == "model-requested-hold"
                )
                and commands[a["command_sha256"]]["model_call_id"] not in calls
                for a in full
            ):
                raise ValueError("DECISION_STAGE_BEHAVIOR_CHANGED")
            selected_commands = [commands[a["command_sha256"]] for a in full]
            document = dict(
                behavior_application=dict(
                    action=first["action"],
                    start_ms=start,
                    end_ms=end,
                    controller_mode="model-with-bounded-hybrid",
                ),
                commands=selected_commands,
                control_applications=full,
                observations=witnesses,
            )
            if brake_path.exists():
                document["executor_brake_applications"] = [
                    a for a in brakes if start <= a["accepted_at_unix_ms"] <= end
                ]
            if first["action"] == "wait":
                matching = [
                    entry
                    for entry in continuations
                    if all(
                        entry["row"]["state"][k] == first["state"][k]
                        for k in ("mission_id", "map_sha256", "goal_id", "route_sha256")
                    )
                    and end <= entry["row"]["state"]["frame"]["observed_at_ms"] <= end + 30000
                    and entry["row"]["state"]["frame"]["crossing_obstacle"] is False
                ]
                document["continuation"] = min(
                    matching,
                    key=lambda entry: entry["row"]["state"]["frame"]["observed_at_ms"],
                    default=None,
                )
            if export_root is not None:
                from scripts.export_decision_stage_candidate import export_candidate

                entry = export_candidate(
                    episode, export_root, number, group, first, document, all_selections=selections
                )
                exports.append(entry)
                # 只有实际物理验收通过的后续推进才可支持等待；失败导出仍保留在清单中。
                if first["action"] in {"follow_route", "slow_down"}:
                    outcome = decode_object(
                        (export_root / entry["outcomes"][0]["path"]).read_text("utf-8")
                    )
                    resumed = dict(
                        state=first["state"],
                        episode_id=episode.name,
                        parent_group=entry["parent_group"],
                        split=entry["split"],
                    )
                    try:
                        verify_native_label(resumed, outcome)
                    except (ValueError, KeyError, TypeError):
                        pass
                    else:
                        candidate = decode_object(
                            (export_root / entry["candidate"]["path"]).read_text("utf-8")
                        )
                        proof = decode_object(
                            (export_root / candidate["input_evidence"]["path"]).read_text("utf-8")
                        )
                        continuations.append(
                            dict(row=resumed, outcome=outcome, input_evidence=proof)
                        )
            record.update(
                verify_stage_control(first, document), verified=True, start_ms=start, end_ms=end
            )
        except (ValueError, KeyError, TypeError) as error:
            record["rejection"] = str(error)[:250]
            rejected[record["rejection"]] += 1
        results.append(record)
    report = dict(
        schema_version="dronedream.decision-stage-audit.v1",
        episode=str(episode),
        source_selection_index_sha256=file_digest(episode / "decision-selections.json"),
        partition_window_ms=window_ms,
        control_only_verified=sum(r["verified"] for r in results),
        windows=sorted(results, key=lambda r: r["first_selection"]),
        rejections=dict(rejected),
        formal_training_additions=0,
        qualification_granted=False,
        requires_physics_and_input_audit=True,
    )
    write_evidence_object(output, report)
    if export_root is not None:
        write_evidence_object(
            export_root / "manifest.json",
            dict(schema_version="dronedream.decision-collection-manifest.v1", entries=exports),
        )
    return report


# 功能：只输出新离线报告，源回合保持原样；输入：显式路径；输出：来源审计数量。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument("--window-ms", type=int, default=3000)
    args = parser.parse_args()
    report = audit(
        args.episode, args.output, export_root=args.export_root, window_ms=args.window_ms
    )
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}))


if __name__ == "__main__":
    main()
