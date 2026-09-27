"""Bind stopped native collection windows to replayable inputs for corpus audit."""

import argparse
import copy
import json
from pathlib import Path

from dronedream_agent_core.decision_dataset import SPLITS, decode_object, file_digest
from dronedream_agent_core.decision_input_evidence import verify_input_evidence
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.stream_capture import require_grounded_stream


# 功能：落地后绑定完整因果前缀和独立结果，失败窗口也进入待审清单而非被悄悄删除。
# 输入：已结束原生回合、新目录、明确集合；输出：可搬迁清单，不授予正式资格。
def prepare(episode, output, split):
    if split not in SPLITS:
        raise ValueError("DECISION_MANIFEST_SPLIT_INVALID")
    require_grounded_stream(episode)
    source = decode_object((episode / "decision-input-source.json").read_text("utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    entries, quarantine = [], []

    # 功能：独占写入新证据并计算实际内容摘要；输入：文件名/对象；输出：包内相对引用。
    def publish(name, document):
        write_evidence_object(output / name, document)
        return {"path": name, "sha256": file_digest(output / name)}

    for candidate_path in sorted((episode / "decision-windows").glob("*-candidate.json")):
        stem = candidate_path.name.removesuffix("-candidate.json")
        candidate = decode_object(candidate_path.read_text("utf-8"))
        try:
            proof = copy.deepcopy(source)
            sequence = candidate["state"]["sequence"]
            if type(sequence) is not int or not 0 <= sequence < len(proof["snapshots"]):
                raise ValueError("DECISION_INPUT_PREFIX_SEQUENCE_INVALID")
            proof["snapshots"] = proof["snapshots"][: sequence + 1]
            state, issues = verify_input_evidence(candidate["state"], proof)
            outcome_path = candidate_path.with_name(stem + "-outcome.json")
            outcome = decode_object(outcome_path.read_text("utf-8"))
            if outcome["state_sha256"] != decision_digest(state.model_dump(mode="json")):
                raise ValueError("DECISION_MANIFEST_OUTCOME_STATE_MISMATCH")
            continuation = outcome.get("continuation")
            if continuation is not None:
                resumed = continuation["row"]["state"]
                recovery_proof = copy.deepcopy(source)
                recovery_proof["snapshots"] = recovery_proof["snapshots"][: resumed["sequence"] + 1]
                verify_input_evidence(resumed, recovery_proof)
                continuation["input_evidence"] = recovery_proof
            proof_ref = publish(stem + "-input.json", proof)
            candidate.update(
                input_evidence=proof_ref,
                source_sha256=proof_ref["sha256"],
                coverage_issues=issues,
                label_status="awaiting-independent-outcome",
            )
            entries.append(
                {
                    "candidate": publish(stem + "-candidate.json", candidate),
                    "outcomes": [publish(stem + "-outcome.json", outcome)],
                    "parent_group": outcome["parent_group"],
                    "episode_id": outcome["episode_id"],
                    "split": split,
                    "license": "proprietary-user-owned",
                }
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            quarantine.append(
                {"window": stem, "reason": type(error).__name__ + ":" + str(error)[:300]}
            )
    publish(
        "manifest.json",
        {"schema_version": "dronedream.decision-collection-manifest.v1", "entries": entries},
    )
    receipt = {
        "source_episode": str(episode),
        "source_input_sha256": file_digest(episode / "decision-input-source.json"),
        "bound_windows": len(entries),
        "input_quarantine": quarantine,
        "formal_decision_windows": 0,
        "gpu_ready": False,
        "requires_independent_outcome_audit": True,
    }
    publish("binding-report.json", receipt)
    return receipt


# 功能：生成待审清单而不改变源实验；输入：显式路径/集合；输出：绑定数量，空清单返回非零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    result = prepare(args.episode, args.output, args.split)
    print(json.dumps(result))
    return 0 if result["bound_windows"] and not result["input_quarantine"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
