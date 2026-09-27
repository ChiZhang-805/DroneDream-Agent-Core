"""Read-only, bounded aggregation of native control timing; never labels training rows."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from dronedream_agent_core.decision_dataset import file_digest
from dronedream_agent_core.decision_label_evidence import read_bound
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from scripts.collect_native_decisions import records


# 功能：汇总有限数量的毫秒测量，保留负数以揭示过期，不截断为看似成功的零。
# 输入：实测数值列表；输出：数量、最小/最大/均值和中位数，不授予运动或训练资格。
def statistics(values):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("DECISION_LATENCY_MEASUREMENT_INVALID")
    values = sorted(values)
    if not values:
        return {"count": 0}
    return dict(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        median=values[len(values) // 2],
        p95=values[math.ceil(len(values) * 0.95) - 1],
    )


# 功能：累积同一种实际耗时，不用缺失字段补零；允许负余量揭示已过期输入。
# 输入：分组容器、字段字典及可选前缀；输出：原容器更新，无控制副作用。
def accumulate(target, values, prefix=""):
    for key, value in (values or {}).items():
        if type(value) in (int, float):
            statistics([value])
            target.setdefault(prefix + key, []).append(value)


# 功能：有界读取可选诊断文件；缺失明确记为不可用，不将空日志当成零延迟。
# 输入：文件路径；输出：对象或None。路径固定在原回合，不读取任意外部引用。
def optional_document(path):
    if not path.exists():
        return None
    return read_bound(path.parent, {"path": path.name, "sha256": file_digest(path)})


# 功能：将生产端剩余租期、调度选择与执行回执分别统计，不把发布当成执行。
# 输入：已结束回合目录；输出：有界诊断报告。来源UNIX毫秒不修改，无训练新增。
def audit(episode):
    simulation = episode / "flight/simulation"
    commands, applications, brakes, handoffs = Counter(), Counter(), Counter(), Counter()
    spans = {
        key: []
        for key in (
            "model_remaining_at_record_ms",
            "model_dispatch_elapsed_ms",
            "model_remaining_at_accept_ms",
            "bridge_remaining_at_record_ms",
        )
    }
    missing, reasons, authority = Counter(), Counter(), Counter()
    phase_groups, input_timing, scheduling, interface = {}, {}, {}, {}
    published_model_calls, accepted_model_calls = set(), set()
    command_calls = {}
    unmatched_model_applications = 0
    for row in records(simulation / "depth-local-safety-history.jsonl"):
        command = row.get("command")
        if not command:
            continue
        kind = (
            "model"
            if command.get("model_navigation_authorized")
            else "bridge"
            if command.get("hybrid_lease")
            else "hold"
        )
        commands[kind] += 1
        if kind == "model":
            published_model_calls.add(command.get("model_call_id"))
            command_calls[decision_digest(command)] = command.get("model_call_id")
        authority[command.get("model_authority_reason", "absent")] += 1
        group = "ready-control" if row.get("prioritized_ready_control") else "sensor-tick"
        accumulate(phase_groups.setdefault(group, {}), row.get("pipeline_phase_ms"))
        accumulate(input_timing, row.get("depth_input_timing"), "depth.")
        if kind in {"model", "bridge"}:
            spans[kind + "_remaining_at_record_ms"].append(
                command["valid_until_unix_ms"] - row["recorded_at_unix_ms"]
            )
        reasons[(row.get("bounded_hybrid_diagnostic") or {}).get("reason", "absent")] += 1
        for key, value in (row.get("bounded_hybrid_input_checks") or {}).items():
            if value is False:
                missing[key] += 1
        handoff = row.get("handoff_scheduling") or {}
        handoffs[
            str(
                (
                    handoff.get("result_ready"),
                    handoff.get("local_pending"),
                    handoff.get("dispatch_budget_ready"),
                    row.get("prioritized_ready_control"),
                )
            )
        ] += 1
    for row in records(simulation / "runtime-state/control-applications.jsonl"):
        applications[row["control_source"]] += 1
        if row["model_authorized"]:
            call_id = command_calls.get(row["command_sha256"])
            if call_id is None:
                unmatched_model_applications += 1
            else:
                accepted_model_calls.add(call_id)
            spans["model_dispatch_elapsed_ms"].append(
                row["accepted_at_unix_ms"] - row["command_generated_at_unix_ms"]
            )
            spans["model_remaining_at_accept_ms"].append(
                row["command_valid_until_unix_ms"] - row["accepted_at_unix_ms"]
            )
    for row in records(simulation / "runtime-state/executor-brake-applications.jsonl"):
        brakes[row["reason"]] += 1
    model_path = simulation / "model-navigation-timing.jsonl"
    if model_path.exists():
        for row in records(model_path):
            accumulate(scheduling, row.get("phase_ms"), "phase.")
            accumulate(scheduling, row.get("input_preparation_ms"), "input.")
    interface_doc = optional_document(episode / "training-interface-timing.json")
    for row in (interface_doc or {}).get("rows", []):
        stage = row.get("stage", "unknown")
        accumulate(
            interface,
            {k: v for k, v in row.items() if k.endswith("_ms") and not k.endswith("unix_ms")},
            stage + ".",
        )
        accumulate(
            interface,
            {
                "runtime_preparation_ms": (row.get("runtime_timing") or {}).get(
                    "input_preparation_ms"
                )
            },
            stage + ".",
        )
    # 原始选择索引逐项摘要核验，禁止从另一次实验借用较好教师耗时。
    selection_doc = optional_document(episode / "decision-selections.json")
    teacher_timing = {}
    for ref in (selection_doc or {}).get("selections", []):
        selection = read_bound(episode, ref)
        result = "submitted" if selection.get("submission") else "not-submitted"
        accumulate(teacher_timing, selection.get("preparation_ms"), result + ".")
    # 同一次调用可能发布/应用多条命令；按调用ID对照，不混淆发布计数和执行次数。
    published_model_calls.discard(None)
    accepted_model_calls.discard(None)
    return dict(
        schema_version="dronedream.decision-control-latency.v1",
        episode=str(episode),
        published_commands=dict(commands),
        accepted_applications=dict(applications),
        executor_brakes=dict(brakes),
        model_authority_reasons=dict(authority),
        call_join={
            "published": len(published_model_calls),
            "accepted": len(accepted_model_calls),
            "published_without_acceptance": len(published_model_calls - accepted_model_calls),
            # 未找到原命令时不知道调用ID，不能把None去掉后报告为零个遗漏调用。
            "unmatched_model_applications": unmatched_model_applications,
        },
        hybrid_reasons=dict(reasons),
        hybrid_missing_inputs=dict(missing),
        handoff_flags=dict(handoffs),
        timing_ms={k: statistics(v) for k, v in spans.items()},
        pipeline_phase_ms={
            group: {k: statistics(v) for k, v in values.items()}
            for group, values in phase_groups.items()
        },
        depth_timing_ms={k: statistics(v) for k, v in input_timing.items() if k.endswith("_ms")},
        model_scheduling_ms={k: statistics(v) for k, v in scheduling.items()},
        interface_timing_ms={k: statistics(v) for k, v in interface.items()},
        teacher_timing_ms={k: statistics(v) for k, v in teacher_timing.items()},
        interface_scope=None
        if interface_doc is None
        else {
            k: interface_doc.get(k)
            for k in (
                "retention",
                "requests_received",
                "input_rejection_counts",
                "proposal_preparation_expirations",
                "input_admission_budget_ms",
                "post_preparation_dispatch_reserve_ms",
            )
        },
        formal_training_additions=0,
        qualification_granted=False,
    )


# 功能：在显式新文件输出诊断；输入：回合及目标路径；输出：短汇总，源账本只读。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("DECISION_LATENCY_REPORT_ALREADY_EXISTS")
    report = audit(args.episode)
    write_evidence_object(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
