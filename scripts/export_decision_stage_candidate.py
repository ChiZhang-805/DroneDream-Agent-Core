"""Export unqualified mixed-controller candidates with original source and outcome records."""

import copy

from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.decision_layout_lineage import collection_split
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：保留窗口及桥接起点前/交还后边界见证，避免每条样本重复整回合数据。
# 输入：完整原始序列、起止UNIX毫秒和原命令；输出：连续原始切片，不抽帧、不修补缺口。
def bounded_displacement_witnesses(witnesses, commands, start, end):
    starts = [start]
    for command in commands:
        lease = command.get("hybrid_lease")
        if lease is not None:
            starts.append(lease["started_at_unix_ms"])
    lower = min(starts)
    before = [row for row in witnesses if row["observed_at_unix_ms"] <= lower]
    after = [row for row in witnesses if row["observed_at_unix_ms"] >= end]
    return (
        before[-1:]
        + [row for row in witnesses if lower < row["observed_at_unix_ms"] < end]
        + after[:1]
    )


# 功能：导出完整窗口及输入因果前缀，混合控制另用明确schema；失败原样进入待审清单。
# 输入：已排空回合、原始选择组、原命令/回执/见证、新目录；输出：相对路径清单条目。
def export_candidate(episode, output, number, group, first, document, *, all_selections=None):
    prototypes = sorted((episode / "decision-windows").glob("*-outcome.json"))
    if not prototypes:
        raise ValueError("DECISION_STAGE_BOUND_PHYSICAL_CONTEXT_MISSING")
    prototype = decode_object(prototypes[0].read_text("utf-8"))
    keys = (
        "parent_group",
        "episode_id",
        "map_sha256",
        "geometry_sha256",
        "envelope",
        "static_primitives",
    )
    source = decode_object((episode / "decision-input-source.json").read_text("utf-8"))
    state = first["state"]
    digest = decision_digest(state)
    if prototype["map_sha256"] != state["map_sha256"] or prototype["episode_id"] != episode.name:
        raise ValueError("DECISION_STAGE_CONTEXT_IDENTITY_INVALID")
    proof = copy.deepcopy(source)
    proof["snapshots"] = proof["snapshots"][: state["sequence"] + 1]
    if not proof["snapshots"] or proof["snapshots"][-1] != first["snapshot"]:
        raise ValueError("DECISION_STAGE_SOURCE_PREFIX_INVALID")
    outcome = copy.deepcopy(document)
    # 命令可被执行器重复刷新，命令内容只存一次；实际执行账本逐条完整保留。
    outcome["commands"] = list({decision_digest(c): c for c in outcome["commands"]}.values())
    outcome.update({k: prototype[k] for k in keys})
    outcome.update(
        schema_version="dronedream.decision-stage-outcome.v1",
        state_sha256=digest,
        goal_world_enu_m=first["snapshot"]["goal_position_m"],
        behavior_selections=group,
        executor_phase_summary=decode_object(
            (episode / "flight/simulation/offboard_timing.json").read_text("utf-8")
        )["snapshots"],
    )
    outcome["behavior_application"].update(
        purpose="native-stage-application",
        source="executor",
        applied=True,
        state_sha256=digest,
        control_application_hashes=[decision_digest(a) for a in outcome["control_applications"]],
        requested_speed_limit_mps=first["reason"].get("requested_speed_limit_mps"),
        nominal_speed_limit_mps=first["reason"].get("nominal_speed_limit_mps"),
    )
    if "executor_brake_applications" in outcome:
        outcome["behavior_application"]["executor_brake_hashes"] = [
            decision_digest(a) for a in outcome["executor_brake_applications"]
        ]
    # 独立见证保留起点前一帧到终点；不把整个回合的无关帧混入结果，也不填补间隙。
    start, end = (
        outcome["behavior_application"]["start_ms"],
        outcome["behavior_application"]["end_ms"],
    )
    if first["action"] == "request_observation":
        from dronedream_agent_core.decision_observation_recovery import capture_observation_recovery

        outcome.update(
            capture_observation_recovery(
                source, all_selections or group, state, start_ms=start, end_ms=end
            )
        )
    before = [o for o in document["observations"] if o["observed_at_unix_ms"] <= start]
    # 桥接位移审计需要交还后的见证；原生结果端点允许终点前100ms，保持两份明确用途。
    outcome["hybrid_displacement_observations"] = bounded_displacement_witnesses(
        document["observations"], outcome["commands"], start, end
    )
    outcome["observations"] = before[-1:] + [
        o for o in document["observations"] if start < o["observed_at_unix_ms"] <= end
    ]
    outcome["source_receipts"] = {
        str(p.relative_to(episode)): file_digest(p)
        for p in (
            episode / "decision-selections.json",
            episode / "decision-input-source.json",
            episode / "decision-independent-witnesses.json",
            episode / "flight/simulation/offboard_timing.json",
            episode / "flight/simulation/runtime-state/control-applications.jsonl",
        )
    }
    brake_path = episode / "flight/simulation/runtime-state/executor-brake-applications.jsonl"
    if brake_path.exists():
        outcome["source_receipts"][str(brake_path.relative_to(episode))] = file_digest(brake_path)
    context_path = episode / "decision-collection-context.json"
    if context_path.exists():
        outcome["source_receipts"][context_path.name] = file_digest(context_path)

    # 功能：仅独占写入新证据目录；输入：后缀和对象；输出：可移动的内容摘要引用。
    def publish(suffix, data):
        path = output / f"{number:04d}-{suffix}.json"
        # 输入前缀最多512帧；与read_bound的16MiB读取合同一致，仍保持显式有界。
        write_evidence_object(path, data, limit=16 * 1024**2, node_limit=2_000_000)
        return dict(path=path.name, sha256=file_digest(path))

    proof_ref = publish("input", proof)
    candidate = dict(
        schema_version="dronedream.decision-candidate.v2",
        sample_id=digest,
        state=state,
        input_evidence=proof_ref,
        source_sha256=proof_ref["sha256"],
        coverage_issues=first["issues"],
        formal_training_eligible=False,
        label_status="awaiting-independent-outcome",
    )
    return dict(
        candidate=publish("candidate", candidate),
        outcomes=[publish("outcome", outcome)],
        parent_group=outcome["parent_group"],
        episode_id=episode.name,
        split=collection_split(episode),
        license="proprietary-user-owned",
    )
