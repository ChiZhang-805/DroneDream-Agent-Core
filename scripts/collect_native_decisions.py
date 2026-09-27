"""Prospective stage decisions through the native PX4 training outlet; simulation only."""

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.decision_collection_teacher import CollectionTeacher
from dronedream_agent_core.decision_dataset import SPLITS, decode_object, file_digest
from dronedream_agent_core.decision_input_evidence import input_algorithm_identity
from dronedream_agent_core.decision_label_evidence import behavior_application_matches
from dronedream_agent_core.decision_native_capture import NativeDecisionCapture
from dronedream_agent_core.decision_phase_evidence import verify_window_phases
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.flight_environment import ControlPreparationExpired, PilotAction
from dronedream_agent_core.training.px4_environment import (
    Px4GazeboTrainingEnvironment,
    Px4TrainingConfig,
)
from dronedream_agent_core.training.stream_capture import (
    grounded_control_index,
    require_grounded_stream,
)


# 功能：有界读取原执行账本，不能把未换行尾记录或过大文件视为完整证据。
# 输入：只读账本路径；输出：原始JSON行，损坏时抛错。
def records(path):
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("DECISION_COLLECTION_LEDGER_TOO_LARGE")
    with path.open(encoding="utf-8") as stream:
        while line := stream.readline(4 * 1024**2 + 1):
            if len(line) > 4 * 1024**2 or not line.endswith("\n"):
                raise ValueError("DECISION_COLLECTION_LEDGER_INCOMPLETE")
            yield decode_object(line)


# 功能：单份归档失败仍保留其它证据及小型失败报告；不因磁盘/大小异常抹掉整轮诊断。
# 输入：独占文件、对象、运行报告及显式容量；输出：写入是否成功，失败记录且禁止正式组装。
def persist_collection_evidence(path, value, report, **budgets):
    try:
        write_evidence_object(path, value, **budgets)
        return True
    except (ValueError, OSError, TypeError, RecursionError) as error:
        failures = report.setdefault("archive_errors", [])
        if len(failures) < 32:
            failures.append(
                {"file": path.name, "error": type(error).__name__ + ":" + str(error)[:300]}
            )
        report["archive_complete"] = False
        return False


# 功能：短时无可用新帧时继续等待，不重发旧动作；真实执行器退出或契约错误仍上抛。
# 输入：仿真环境、整个采集的绝对单调秒截止、诊断计数；输出：新观测或预算结束的None。
def await_fresh_observation(env, deadline, counters, on_wait=None):
    while time.monotonic() < deadline:
        try:
            return env.next_stream_observation(deadline=min(deadline, time.monotonic() + 1.9))
        except TimeoutError as error:
            if str(error) != "PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT":
                raise
            counters["observation_wait_timeouts"] += 1
            if on_wait is not None:
                on_wait()
    return None


# 功能：把等待期间的独立见证分块归档；未通过的缺帧段记入隔离原因，不补帧或改时钟。
# 输入：观测端口、UNIX毫秒起止、标签专用容器；输出：已收到的原始源时刻游标。
# 不能把请求截止当作收到帧的水位；否则晚到但顺序正确的帧会在下次轮询中被漏过。
def drain_label_witnesses(env, start_ms, end_ms, witnesses, errors):
    if not 0 <= end_ms - start_ms <= 30000:
        errors["DECISION_WITNESS_POLL_GAP"] += 1
        return end_ms
    while start_ms < end_ms:
        stop = min(end_ms, start_ms + 1500)
        try:
            batch = env.label_witness_window(start_ms, stop)
            for witness in batch:
                prior = witnesses.get(witness.sequence)
                if prior is not None and prior != witness:
                    raise ValueError("DECISION_WITNESS_SEQUENCE_CHANGED")
                witnesses[witness.sequence] = witness
            latest = max((w.observed_at_unix_ms for w in batch), default=start_ms)
            if latest <= start_ms:
                break
            start_ms = latest
            if stop == end_ms:
                break
        except Exception as error:
            # 标签附件失败不应中断控制或伪造“飞行失败”；正式准入会检查原始连续性。
            errors[type(error).__name__ + ":" + str(error)[:160]] += 1
            # 尚未收到不是确定缺失；保留原游标，下次再读，不伪造完整覆盖。
            if "OUTCOME_WINDOW_NOT_YET_OBSERVED" in str(error):
                break
            start_ms = stop
    return start_ms


# 功能：刚观测到横穿后要求200毫秒连续清空证据再恢复运动，抑制边缘检测闪烁。
# 输入：同一路线的因果历史、当前提案；输出：原提案或短暂等待，不读取未来/仿真真值。
# 新障碍立即等待；缺观测仍走补观测分支。未知和超过250毫秒的间隔不算清空证据。
def stabilize_crossing_release(state, proposal):
    action, pilot, reason = proposal
    frame = state.frame
    if action not in {"follow_route", "slow_down"} or frame.can_hold_position is not True:
        return proposal
    frames = [*state.history, frame]
    latest_crossing = next(
        (i for i in range(len(frames) - 1, -1, -1) if frames[i].crossing_obstacle is True), None
    )
    if latest_crossing is None:
        return proposal
    clear_since = None
    previous = frames[latest_crossing]
    for observed in frames[latest_crossing + 1 :]:
        if not 0 < observed.observed_at_ms - previous.observed_at_ms <= 250:
            clear_since = None
        if observed.crossing_obstacle is not False:
            clear_since = None
        elif clear_since is None:
            clear_since = observed.observed_at_ms
        previous = observed
    if clear_since is not None and frame.observed_at_ms - clear_since >= 200:
        return proposal
    return (
        "wait",
        PilotAction(mode="hold", axes=[0.0] * 4),
        {
            "reason": "crossing-clearance-not-yet-continuous",
            "required_clear_duration_ms": 200,
            "original_proposal": action,
        },
    )


# 功能：净空临界值抖动时立即降速，但只有连续200毫秒明确解除风险才恢复名义速度。
# 输入：同目标因果历史、原四轴提案和米/秒控制尺度；输出：仅缩小平移轴的提案。
# 不修改观测中的caution字段、不延长租期；等待、补观测和终止始终优先。
def stabilize_caution_release(state, proposal, limits):
    action, pilot, reason = proposal
    if action != "follow_route":
        return proposal
    frames = [*state.history, state.frame]
    last = next(
        (i for i in range(len(frames) - 1, -1, -1) if frames[i].caution_required is True), None
    )
    if last is None:
        return proposal
    clear_since = None
    previous = frames[last]
    for frame in frames[last + 1 :]:
        if not 0 < frame.observed_at_ms - previous.observed_at_ms <= 250:
            clear_since = None
        if frame.caution_required is not False:
            clear_since = None
        elif clear_since is None:
            clear_since = frame.observed_at_ms
        previous = frame
    if clear_since is not None and state.frame.observed_at_ms - clear_since >= 200:
        return proposal
    nominal = reason["nominal_speed_limit_mps"]
    cap = min(nominal / 2, 0.15)
    speed = math.hypot(
        pilot.axes[0] * limits.horizontal_speed_mps,
        pilot.axes[1] * limits.horizontal_speed_mps,
        pilot.axes[2] * limits.vertical_speed_mps,
    )
    scale = min(1.0, cap / max(speed, 1e-12))
    slowed = PilotAction(
        mode="pilot-control", axes=[*(a * scale for a in pilot.axes[:3]), pilot.axes[3]]
    )
    return (
        "slow_down",
        slowed,
        dict(
            reason,
            reason="caution-clearance-not-yet-continuous",
            requested_speed_limit_mps=cap,
            required_clear_duration_ms=200,
            original_proposal=action,
        ),
    )


# 功能：选择同一回合、路线和目标的已验收恢复段，不能用另一段飞行替等待补成绩。
# 输入：等待状态、实际结束UNIX毫秒及完整恢复结果；输出：最近合格恢复附件或None。
def find_wait_continuation(state, end_ms, continuations):
    matches = [
        entry
        for entry in continuations
        if all(
            entry["row"]["state"][key] == getattr(state, key)
            for key in ("mission_id", "map_sha256", "goal_id", "route_sha256")
        )
        and end_ms <= entry["row"]["state"]["frame"]["observed_at_ms"] <= end_ms + 30000
        and entry["row"]["state"]["frame"]["crossing_obstacle"] is False
    ]
    return min(
        matches, key=lambda entry: entry["row"]["state"]["frame"]["observed_at_ms"], default=None
    )


# 功能：落地后把事先记录的行为窗口与全部真实回执相交，不把未执行提案计为标签。
# 输入：回合、选择记录、独立真值及地图；输出：原始结果/隔离报告；不宣称数据集已合格。
def finalize(env, selections, witnesses, *, split="train"):
    require_grounded_stream(env.episode_path)
    call_ids = {r["submission"]["call_id"] for r in selections if r.get("submission")}
    if not call_ids:
        return {"verified_native_windows": 0, "rejections": {"NO_SUBMITTED_BEHAVIOR": 1}}
    # 先利用既有完整性审计，禁止从半写入账本抽取有利片段。
    grounded_control_index(env.simulation, call_ids)
    commands = {}
    for row in records(env.simulation / "depth-local-safety-history.jsonl"):
        if row.get("command"):
            command = RuntimeLocalSafetyCommand.model_validate(row["command"])
            commands[decision_digest(command.model_dump(mode="json"))] = command
    applications = [
        ControlApplicationRecord.model_validate(r)
        for r in records(env.simulation / "runtime-state/control-applications.jsonl")
    ]
    brake_path = env.simulation / "runtime-state/executor-brake-applications.jsonl"
    brakes = list(records(brake_path)) if brake_path.exists() else []
    sources = {r["submission"]["call_id"]: r["snapshot"] for r in selections if r.get("submission")}
    phase_summary = decode_object((env.simulation / "offboard_timing.json").read_text("utf-8"))[
        "snapshots"
    ]
    groups, current = [], []
    for row in selections:
        identity = (row["action"], row["state"].goal_id, row["state"].route_sha256)
        previous = current[-1] if current else None
        if current and (
            identity
            != (previous["action"], previous["state"].goal_id, previous["state"].route_sha256)
            or row["selected_at_ms"] - current[0]["selected_at_ms"] >= 3000
        ):
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    counts, rejected = Counter(), Counter()
    native = sorted(witnesses.values(), key=lambda o: o.observed_at_unix_ms)
    output = env.episode_path / "decision-windows"
    output.mkdir(exist_ok=False)
    continuations = []
    # 先复核后发生的运动，再为前面的等待绑定真实恢复；原文件仍按原窗口编号保存。
    for index, group in reversed(list(enumerate(groups))):
        first, last = group[0], group[-1]
        identifiers = {r["submission"]["call_id"] for r in group if r.get("submission")}
        selected = [
            a
            for a in applications
            if a.command_sha256 in commands
            and behavior_application_matches(commands[a.command_sha256], a, first["action"])
            and commands[a.command_sha256].model_call_id in identifiers
            and first["selected_at_ms"] <= a.accepted_at_unix_ms <= last["selected_at_ms"] + 250
        ]
        if (
            not selected
            or selected[-1].accepted_at_unix_ms - selected[0].accepted_at_unix_ms < 2000
        ):
            rejected["INSUFFICIENT_CONTINUOUS_EXECUTION"] += 1
            continue
        start, end = selected[0].accepted_at_unix_ms, selected[-1].accepted_at_unix_ms
        # 原窗口第一个提案可能根本没有执行；标签输入必须匹配首条真实模型执行的原提案。
        # 原始全部选择及实际制动仍保留，不能删除窗口内不利控制来凑连续成功。
        first_call = commands[selected[0].command_sha256].model_call_id
        first = next(
            r for r in group if r.get("submission") and r["submission"]["call_id"] == first_call
        )
        state = first["state"]
        try:
            phases = verify_window_phases(phase_summary, start, end)
            capture = NativeDecisionCapture(
                state=state,
                action=first["action"],
                selected_at_ms=start,
                episode_id=env.episode_path.name,
                parent_group=env.mission_split.group_sha256,
                split=split,
                requested_speed_limit_mps=first["reason"].get("requested_speed_limit_mps"),
                nominal_speed_limit_mps=first["reason"].get("nominal_speed_limit_mps"),
                label_context={
                    "map_sha256": state.map_sha256,
                    "geometry_sha256": env.verifier.geometry_sha256,
                    "static_primitives": env.verifier.primitives,
                    "goal_world_enu_m": first["snapshot"]["goal_position_m"],
                    "envelope": vars(env.verifier.envelope),
                },
            )
            overridden = False
            if any(start <= brake["accepted_at_unix_ms"] <= end for brake in brakes):
                capture.reject("DECISION_PURE_WINDOW_CONTAINS_EXECUTOR_BRAKE")
            # 全时间区间内任何未知来源/换行为命令都算介入，不能只保留期望调用。
            for actual in applications:
                if start <= actual.accepted_at_unix_ms <= end:
                    command = commands.get(actual.command_sha256)
                    if command is None:
                        capture.reject("DECISION_EXECUTED_COMMAND_MISSING")
                        continue
                    overridden |= command.model_call_id not in identifiers
                    overridden |= not behavior_application_matches(command, actual, first["action"])
                    capture.accepted(command, actual, snapshot=sources.get(command.model_call_id))
            before = [o for o in native if o.observed_at_unix_ms <= start]
            for witness in ([before[-1]] if before else []) + [
                o for o in native if start < o.observed_at_unix_ms <= end
            ]:
                capture.witness(witness)
            extra = None
            if first["action"] == "request_observation":
                from dronedream_agent_core.decision_observation_recovery import (
                    capture_observation_recovery,
                )

                source = decode_object(
                    (env.episode_path / "decision-input-source.json").read_text("utf-8")
                )
                extra = capture_observation_recovery(
                    source, selections, state.model_dump(mode="json"), start_ms=start, end_ms=end
                )
            if first["action"] == "wait":
                extra = {"continuation": find_wait_continuation(state, end, continuations)}
            document, report = capture.finish(
                end_ms=end,
                overridden=overridden,
                premature_landing=phases["premature_landing"],
                action_evidence=extra,
            )
            document["phase_evidence"] = phases
            # 完整回合确实落地，行为窗口在关闭请求前；终止事实仍独立保存在运行证据中。
            write_evidence_object(output / f"{index:04d}-outcome.json", document)
            write_evidence_object(output / f"{index:04d}-audit.json", report)
            write_evidence_object(
                output / f"{index:04d}-candidate.json",
                {
                    "schema_version": "dronedream.decision-candidate.v2",
                    "sample_id": decision_digest(state.model_dump(mode="json")),
                    "state": state.model_dump(mode="json"),
                    "formal_training_eligible": False,
                    "coverage_issues": first["issues"],
                    "label_status": "requires-corpus-assembly",
                },
            )
            if report["verified"]:
                counts[first["action"]] += 1
                if first["action"] in {"follow_route", "slow_down"}:
                    continuations.append({"row": capture.row, "outcome": document})
            else:
                rejected[report["rejection"]] += 1
        except (ValueError, KeyError, TypeError) as error:
            rejected[str(error)[:220]] += 1
    return {
        "verified_native_windows": sum(counts.values()),
        "verified_by_action": dict(counts),
        "rejections": dict(rejected),
        "formal_decision_windows": 0,
    }


# 功能：事先选择行为、通过既有端口提交，独立采集真值；结束时先安全落地再离线检查。
# 输入：源摘要冻结的配置、最多120秒持续时间、场景族；输出：真实采集报告，失败不伪装成功。
def collect(config, seconds, scene, *, split="train", route_guidance=False):
    if split not in SPLITS:
        raise ValueError("DECISION_COLLECTION_SPLIT_INVALID")
    env = Px4GazeboTrainingEnvironment(config)
    from dronedream_agent_core.decision_route_teacher import RouteCollectionTeacher

    route = None
    teacher_class = RouteCollectionTeacher if route_guidance else CollectionTeacher
    policy = decision_digest(
        {
            "script_sha256": file_digest(Path(__file__)),
            "teacher_sha256": file_digest(
                Path(__file__).parents[1]
                / "src/dronedream_agent_core/decision_collection_teacher.py"
            ),
            "config": config.model_dump(mode="json"),
            "seconds": seconds,
            "scene": scene,
            "split": split,
            "route_guidance": route_guidance,
        }
    )
    env.bind_policy_identity(policy)
    # 保存输入重建代码而不仅是摘要；未来升级后仍能找回原版本审核旧记录。
    input_version = 2 if route_guidance else 1
    input_identity = input_algorithm_identity(input_version)
    source_root = Path(__file__).parents[1] / "src/dronedream_agent_core"
    input_sources = {
        name: (source_root / name).read_bytes().decode("utf-8") for name in input_identity
    }
    semantic_text = config.semantic.read_bytes().decode("utf-8")
    selections, witnesses = [], {}
    history = deque(maxlen=10)
    report = {
        "simulation_only": True,
        "qualification_granted": False,
        "policy_sha256": policy,
        "expired_proposals": 0,
        "observation_wait_timeouts": 0,
        "label_capture_errors": {},
        "error": None,
        "safely_closed": False,
    }
    errors = Counter()
    last_identity = None
    try:
        observation = env.reset(seed=250925)
        calibration = file_digest(env.simulation / "camera-profile/camera-profile.json")
        if input_algorithm_identity(input_version) != input_identity:
            raise ValueError("DECISION_INPUT_CODE_CHANGED_DURING_STARTUP")
        teacher_parameters = dict(
            body_radius_m=env.verifier.envelope.body_radius_m,
            body_height_m=env.verifier.envelope.body_height_m,
            nominal_speed_mps=config.speed_limit_mps,
            clearance_m=config.required_clearance_m,
        )
        # 启动器会将单程输入构造成往返路线；必须绑定执行中的路线，而不是配置模板。
        if route_guidance:
            route = decode_object((env.simulation / "mission-route.json").read_text("utf-8"))
        teacher = teacher_class(
            map_sha256=config.asset_sha256["semantic"],
            primitives=env.verifier.primitives,
            **teacher_parameters,
            **({"qualified_route": route} if route_guidance else {}),
        )
        deadline = time.monotonic() + seconds
        last_witness_ms = int(time.time() * 1000) - 100

        # 功能：等待新观测时只归档标签通道；输出：更新标签游标，不访问/改写控制提案。
        def drain():
            nonlocal last_witness_ms
            last_witness_ms = drain_label_witnesses(
                env, last_witness_ms, int(time.time() * 1000) - 30, witnesses, errors
            )

        while len(selections) < config.episode_steps and time.monotonic() < deadline:
            cycle_started = time.perf_counter()
            snapshot = env.current_navigation_snapshot()
            context_finished = time.perf_counter()
            now = snapshot["control_reference_observed_at_unix_ms"]
            goal = snapshot["strategic_context"]["task"]["navigation_goal_id"]
            identity = (goal, snapshot["known_static_map"]["qualified_route_sha256"])
            if identity != last_identity:
                history.clear()
            while history and not 0 < now - history[0].observed_at_ms <= 2500:
                history.popleft()
            state, proof, issues = teacher.observe(
                snapshot,
                mission_id=config.mission_id,
                sequence=len(selections),
                calibration_sha256=calibration,
                braking_acceleration_mps2=0.8,
                scene=scene,
                history=tuple(history),
            )
            proposal = teacher.propose(state, snapshot, observation.sample.pilot_control_limits)
            proposal = stabilize_caution_release(
                state, proposal, observation.sample.pilot_control_limits
            )
            action, pilot, reason = stabilize_crossing_release(state, proposal)
            selected_ms = int(time.time() * 1000)
            row = {
                "state": state,
                "snapshot": snapshot,
                "action": action,
                "selected_at_ms": selected_ms,
                "reason": reason,
                "map_check": proof,
                "issues": issues,
                "submission": None,
                "preparation_ms": {
                    "context": (context_finished - cycle_started) * 1000,
                    "teacher": (time.perf_counter() - context_finished) * 1000,
                },
            }
            selections.append(row)  # 原行为先于提交记录，后续只补提交身份而不改动作。
            try:
                row["submission"] = env.submit_stream_action(pilot)
            except ControlPreparationExpired:
                report["expired_proposals"] += 1
            history.append(state.frame)
            last_identity = identity
            # 真值只到标签容器；失败不传递给飞控、不成为策略输入。
            drain()
            if time.monotonic() >= deadline:
                break
            observation = await_fresh_observation(env, deadline, report, on_wait=drain)
            if observation is None:
                break
        drain()
    except Exception as error:
        report["error"] = type(error).__name__ + ":" + str(error)[:500]
    finally:
        try:
            env.close()
            report["safely_closed"] = True
        except Exception as error:
            report["close_error"] = type(error).__name__ + ":" + str(error)[:500]
        report["episode_path"] = str(env.episode_path) if env.episode_path else None
        report["selections"] = len(selections)
        report["submitted"] = sum(r["submission"] is not None for r in selections)
        report["label_capture_errors"] = dict(errors)
        if env.episode_path is not None:
            persist_collection_evidence(
                env.episode_path / "decision-collection-context.json",
                {
                    "schema_version": "dronedream.decision-collection-context.v1",
                    "split": split,
                    "input_version": input_version,
                    "policy_sha256": policy,
                },
                report,
            )
            persist_collection_evidence(
                env.episode_path / "decision-input-code.json",
                {
                    "algorithm": input_identity,
                    "source_text": input_sources,
                    "qualification_granted": False,
                },
                report,
            )
            if selections:
                persist_collection_evidence(
                    env.episode_path / "decision-input-source.json",
                    {
                        "schema_version": f"dronedream.decision-input-evidence.v{input_version}",
                        **({"qualified_route": route} if route_guidance else {}),
                        "algorithm": input_identity,
                        "semantic_text": semantic_text,
                        "teacher": teacher_parameters,
                        "identity": dict(
                            mission_id=config.mission_id,
                            calibration_sha256=calibration,
                            braking_acceleration_mps2=0.8,
                            scene=scene,
                        ),
                        "snapshots": [r["snapshot"] for r in selections],
                    },
                    report,
                    limit=16 * 1024**2,
                    node_limit=2_000_000,
                )
            selection_root = env.episode_path / "decision-selections"
            selection_root.mkdir(exist_ok=False)
            references = []
            for index, row in enumerate(selections):
                path = selection_root / f"{index:06d}.json"
                if persist_collection_evidence(
                    path, {**row, "state": row["state"].model_dump(mode="json")}, report
                ):
                    references.append(
                        {
                            "path": str(path.relative_to(env.episode_path)),
                            "sha256": file_digest(path),
                        }
                    )
            persist_collection_evidence(
                env.episode_path / "decision-selections.json",
                {
                    "schema_version": "dronedream.decision-selection-index.v1",
                    "selections": references,
                },
                report,
            )
            persist_collection_evidence(
                env.episode_path / "decision-independent-witnesses.json",
                {
                    "scope": "label-only",
                    "observations": [
                        o.model_dump(mode="json")
                        for o in sorted(witnesses.values(), key=lambda r: r.observed_at_unix_ms)
                    ],
                },
                report,
                limit=16 * 1024**2,
                node_limit=2_000_000,
            )
        if input_algorithm_identity(input_version) != input_identity:
            report["error"] = "DECISION_INPUT_CODE_CHANGED_DURING_COLLECTION"
        if (
            report["safely_closed"]
            and selections
            and report["error"] is None
            and not report.get("archive_errors")
        ):
            try:
                report.update(finalize(env, selections, witnesses, split=split))
            except Exception as error:
                report["finalize_error"] = type(error).__name__ + ":" + str(error)[:500]
        write_evidence_object(config.output_root / "collection-report.json", report)
    return report


# 功能：拒绝无界采集或复用输出目录，实际运行仅在已安装仿真环境中执行。
# 输入：命令行冻结配置和有界秒数；输出：完整报告及非零失败退出码。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--route-guidance", action="store_true")
    parser.add_argument("--output-root", type=Path,
                        help="New per-attempt directory; never overwrite the previous run.")
    parser.add_argument(
        "--scene",
        choices=("office", "corridor", "door", "stairs", "transition", "outdoor", "pickup"),
        required=True,
    )
    args = parser.parse_args()
    config = Px4TrainingConfig.model_validate(decode_object(args.config.read_text("utf-8")))
    if args.output_root is not None:
        config = Px4TrainingConfig.model_validate(
            {**config.model_dump(mode="python"), "output_root": args.output_root})
    split = "train"
    receipt_path = args.config.parent / "scene-receipt.json"
    if receipt_path.exists():
        receipt = decode_object(receipt_path.read_text("utf-8"))
        if receipt.get("scene_family") == "room-tree-generator-v1":
            from dronedream_agent_core.decision_scene_topology import verified_scene_family

            verified_scene_family(receipt, config.asset_sha256["semantic"])
            split = receipt.get("split")
            if split not in SPLITS or not args.route_guidance:
                raise ValueError("DECISION_TOPOLOGY_SPLIT_AND_ROUTE_GUIDANCE_REQUIRED")
    if (
        not 3 <= args.seconds <= 120
        or config.initial_collection_mode != "stream-imitation"
        or config.episode_steps > 480
    ):
        raise ValueError("DECISION_COLLECTION_BUDGET_INVALID")
    config.output_root.mkdir(parents=True, exist_ok=False)
    try:
        report = collect(
            config, args.seconds, args.scene, split=split, route_guidance=args.route_guidance
        )
    except Exception as error:
        report = {
            "error": type(error).__name__ + ":" + str(error)[:500],
            "qualification_granted": False,
            "formal_decision_windows": 0,
        }
        # 保留已有运行报告；构造阶段失败没有启动飞行，不能写成已安全落地。
        write_evidence_object(config.output_root / "collection-failure.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(
        bool(
            report["error"]
            or report.get("close_error")
            or report.get("finalize_error")
            or report.get("archive_errors")
            or not report.get("verified_native_windows")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
