"""Audit label-only native motion against bridge limits without promoting dataset rows."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object, file_digest, read_records
from dronedream_agent_core.decision_hybrid_evidence import audit_hybrid_displacement
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.stream_capture import require_grounded_stream


# 功能：只在已落地回合读取完整账本及独立真值，失败仍保存摘要与原因。
# 输入：独立回合路径和新报告路径；输出：实测审核报告，绝不增加正式样本。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_grounded_stream(args.episode)
    simulation = args.episode / "flight/simulation"
    sources = {
        "commands": simulation / "depth-local-safety-history.jsonl",
        "applications": simulation / "runtime-state/control-applications.jsonl",
        "witnesses": args.episode / "decision-independent-witnesses.json",
    }
    try:
        result = audit_hybrid_displacement(
            read_records(sources["commands"]),
            read_records(sources["applications"]),
            decode_object(sources["witnesses"].read_text("utf-8"))["observations"],
        )
        result["displacement_check_completed"] = True
    except ValueError as error:
        result = {
            "displacement_check_completed": False,
            "reason": str(error),
            "qualification_granted": False,
            "formal_training_additions": 0,
        }
    result["source_sha256"] = {key: file_digest(path) for key, path in sources.items()}
    write_evidence_object(args.output, result)
    print(json.dumps(result))
    return 0 if result["displacement_check_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
