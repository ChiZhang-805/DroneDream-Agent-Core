"""Read-only audit joining actual transport receipts to immutable safety commands."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from dronedream_agent_core.control_execution_evidence import verify_control_applications
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand


# 功能：有界读取完整证据流，损坏/超长行不被静默跳过。
# 输入：JSONL 文件；输出：按原顺序保留的记录。
def rows(path):
    result = []
    with path.open(encoding="utf-8") as stream:
        while line := stream.readline(4 * 1024 * 1024 + 1):
            if len(line) > 4 * 1024 * 1024:
                raise ValueError("EVIDENCE_LINE_TOO_LARGE")
            result.append(json.loads(line))
    return result


# 功能：分块计算原始证据摘要并关闭文件；输入：证据路径；输出：SHA256。
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 功能：核对真实发送和衔接授权，保留整体任务失败，不把诊断结束改写成任务通过。
# 输入：独立仿真目录、新报告文件；输出：回执计数、时速界限及证据摘要。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.run
    names = ["runtime-state/control-applications.jsonl", "runtime-state/local-safety-executor-history.jsonl",
             "depth-local-safety-history.jsonl", "model-navigation-snapshots.jsonl"]
    loaded = {name: rows(root / name) for name in names}
    writer = json.loads((root / "runtime-state/control-application-writer.json").read_text())
    counts, application_path = {}, None
    # 写入器持有 WSL 原路径；仅匹配已知文件名，不访问任意清单绝对路径。
    for artifact in writer["artifacts"]:
        name = "runtime-state/" + artifact["path"].replace("\\", "/").rsplit("/", 1)[-1]
        if name not in loaded or artifact["path"] in counts:
            raise ValueError("WRITER_ARTIFACT_UNEXPECTED")
        counts[artifact["path"]] = len(loaded[name])
        if name == names[0]:
            application_path = artifact["path"]
    records, history = loaded[names[0]], loaded[names[2]]
    verification = verify_control_applications(records, navigation_snapshots=loaded[names[3]],
        command_records=history, expected_count=len(records), writer_summary=writer,
        writer_artifact_counts=counts, application_artifact_path=application_path)
    commands = {}
    for row in history:
        if row.get("command") is not None:
            command = RuntimeLocalSafetyCommand.model_validate(row["command"])
            commands[sha256_json(command)] = command
    classified, speeds, elapsed, episodes = Counter(), [], [], set()
    for record in records:
        command = commands.get(record["command_sha256"])
        if command is None:
            classified["unbound"] += 1
            continue
        authority = command.navigation_control_authority
        classified[authority + ":" + record["safety_action"]] += 1
        if authority == "bounded-hybrid" and record["safety_action"] in {"continue", "slow"}:
            lease = command.hybrid_lease
            speeds.append(math.sqrt(sum(v*v for v in record["velocity_ned_mps"])))
            elapsed.append(record["accepted_at_unix_ms"] - lease.started_at_unix_ms)
            episodes.add((lease.route_sha256, lease.navigation_goal_id, lease.episode))
    workflow_path = root / "workflow-result.json"
    if workflow_path.is_file():
        workflow = json.loads(workflow_path.read_text())
        runtime = workflow["runtime_evidence"]
        landing = runtime["gates"]["landing_confirmed"]
        abort_reason = runtime["measurements"].get("abort_reason")
        workflow_status = workflow["status"]
    else:
        # 采集运行没有产品工作流报告，必须读真实终止回执，不能用“没报错”替代落地。
        terminal = json.loads((root / "native-terminal-lifecycle.json").read_text())
        landing = (terminal.get("landing_confirmed") is True
                   and terminal.get("terminal_state") == "ON_GROUND"
                   and terminal.get("safe_to_stop_watchdog") is True)
        abort_reason = None
        workflow_status = "training-collection-not-product-qualification"
    report = {"transport_verification": verification, "actual_classification": dict(classified),
        "hybrid_episodes_with_motion": len(episodes), "hybrid_maximum_sent_speed_mps": max(speeds, default=None),
        "hybrid_maximum_acceptance_since_episode_start_ms": max(elapsed, default=None),
        "landing_confirmed": landing,
        "abort_reason": abort_reason, "workflow_status": workflow_status,
        "full_mission_accepted": False, "formal_training_additions": 0,
        "limitation": "Transport acceptance is not proof of achieved motion or collision-free full-route flight; displacement budget requires native telemetry.",
        "source_sha256": {name: digest(root / name) for name in names}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}))


if __name__ == "__main__":
    main()
