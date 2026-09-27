"""Read-only collection diagnostics; zero changes to authorization or dataset counts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object, file_digest, read_records
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：计算实测分位数，空数据不补零；输入：毫秒数值；输出：数量和中位/P95/最大值。
def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


# 功能：关联原提案、全部执行记录和真实时钟，找出连续区间而非仅汇报平均成功率。
# 输入：已结束回合；输出：诊断统计，不产生标签或删掉中断/接管记录。
def analyze(episode):
    index = decode_object((episode / "decision-selections.json").read_text("utf-8"))
    selections = []
    for ref in index["selections"]:
        path = episode / ref["path"]
        if (
            not path.resolve().is_relative_to(episode.resolve())
            or file_digest(path) != ref["sha256"]
        ):
            raise ValueError("DECISION_DIAGNOSTIC_SELECTION_CHANGED")
        selections.append(decode_object(path.read_text("utf-8")))
    by_call = {r["submission"]["call_id"]: r for r in selections if r["submission"]}
    simulation = episode / "flight/simulation"
    command_rows = read_records(simulation / "depth-local-safety-history.jsonl")
    commands = {
        decision_digest(r["command"]): r["command"] for r in command_rows if r.get("command")
    }
    applications = read_records(simulation / "runtime-state/control-applications.jsonl")
    reasons, segments, active, known = Counter(), [], [], set()
    for application in applications:
        command = commands.get(application["command_sha256"])
        if command is None:
            reason = "command-missing"
        else:
            call = command.get("model_call_id")
            row = by_call.get(call)
            if row is None:
                reason = "outside-selected-behavior"
            elif not application["model_authorized"]:
                reason = "selected-call-without-model-authority"
            elif (
                command.get("model_navigation_snapshot_sha256")
                != row["snapshot"]["snapshot_sha256"]
            ):
                reason = "selected-input-identity-mismatch"
            else:
                reason = "selected-model-applied"
                known.add(call)
        reasons[reason] += 1
        stamp = application["accepted_at_unix_ms"]
        if active and (reason != "selected-model-applied" or not 0 < stamp - active[-1] <= 250):
            segments.append(
                {
                    "start_ms": active[0],
                    "end_ms": active[-1],
                    "duration_ms": active[-1] - active[0],
                    "applications": len(active),
                }
            )
            active = []
        if reason == "selected-model-applied":
            active.append(stamp)
    if active:
        segments.append(
            {
                "start_ms": active[0],
                "end_ms": active[-1],
                "duration_ms": active[-1] - active[0],
                "applications": len(active),
            }
        )
    return {
        "schema_version": "dronedream.decision-collection-diagnostic.v1",
        "selections": len(selections),
        "submitted_calls": len(by_call),
        "applied_calls": len(known),
        "application_reasons": dict(reasons),
        "uninterrupted_model_segments": segments,
        "at_least_two_second_segments": sum(r["duration_ms"] >= 2000 for r in segments),
        "proposal_preparation_ms": distribution(
            [r["selected_at_ms"] - r["state"]["frame"]["observed_at_ms"] for r in selections]
        ),
        "context_ms": distribution(
            [r["preparation_ms"]["context"] for r in selections if "preparation_ms" in r]
        ),
        "teacher_ms": distribution(
            [r["preparation_ms"]["teacher"] for r in selections if "preparation_ms" in r]
        ),
        "formal_training_additions": 0,
        "qualification_granted": False,
    }


# 功能：只读诊断入口；输入：回合路径和新报告路径；输出：有界概要与完整报告。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.episode)
    write_evidence_object(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "uninterrupted_model_segments"}
        )
    )


if __name__ == "__main__":
    main()
