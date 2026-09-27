"""Assemble independently verified native decisions; quarantine gaps, never invent labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.decision_dataset import SPLITS, audit_records, decode_object, file_digest
from dronedream_agent_core.decision_input_evidence import (
    verify_input_evidence,
    verify_input_outcome_geometry,
)
from dronedream_agent_core.decision_label_evidence import read_bound, verify_native_label
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest


# 功能：独占写入规范化证据，返回内容摘要；输入：新文件/对象；输出：SHA256。
def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return file_digest(path)


# 功能：把已采集候选与原始物理结果绑定成监督记录；缺少证据直接拒绝，不调用模型补答案。
# 输入：采集清单条目和根目录；输出：经过独立结果验证的样本及结果原文。
def verified_entry(entry, root):
    if set(entry) != {"candidate", "outcomes", "parent_group", "episode_id", "split", "license"}:
        raise ValueError("DECISION_COLLECTION_ENTRY_INVALID")
    if entry["split"] not in SPLITS:
        raise ValueError("DECISION_COLLECTION_SPLIT_INVALID")
    candidate = read_bound(root, entry["candidate"])
    if (
        candidate.get("schema_version") != "dronedream.decision-candidate.v2"
        or type(candidate.get("coverage_issues")) is not list
        or not candidate.get("source_sha256")
        or candidate.get("label_status") != "awaiting-independent-outcome"
    ):
        raise ValueError("DECISION_CANDIDATE_STATE_NOT_QUALIFIED")
    # 缺失信息本身是重要训练场景；是否允许某个标签由动作专属结果验证决定，
    # 不因为存在 unknown 而一律丢弃。未知故障类型仍需先完善转换器，不能静默忽略。
    understood_gaps = {
        "GOAL_ALIGNED_SECTORS_NOT_BODY_CLEARANCES",
        "LOCAL_ROUTE_STATUS_NOT_WITNESSED",
        "BEHAVIOR_OUTCOME_LABELS_REQUIRED",
        "BODY_ORIENTATION_UNKNOWN",
        # 尚无双帧跟踪是显式未知，不等于“前方无人”；仍须原始输入重建和独立执行结果。
        # 格式错误、重复跟踪及运动不一致不在此列表，不能借兼容新字段放过坏来源。
        "DYNAMIC_TWO_FRAME_TRACK_REQUIRED",
        "DYNAMIC_HISTORY_GAP",
        "DYNAMIC_CONTEXT_CHANGED",
        "DYNAMIC_HORIZONTAL_ROUTE_UNDEFINED",
    }
    if any(gap not in understood_gaps for gap in candidate["coverage_issues"]):
        raise ValueError("DECISION_CANDIDATE_UNSUPPORTED_GAP")
    state = DecisionStateV2.model_validate_json(json.dumps(candidate["state"]))
    state_hash = decision_digest(state.model_dump(mode="json"))
    if candidate.get("sample_id") != state_hash:
        raise ValueError("DECISION_CANDIDATE_IDENTITY_INVALID")
    if "input_evidence" not in candidate:
        raise ValueError("DECISION_CANDIDATE_INPUT_EVIDENCE_REQUIRED")
    proof = read_bound(root, candidate["input_evidence"])
    if candidate["source_sha256"] != candidate["input_evidence"]["sha256"]:
        raise ValueError("DECISION_CANDIDATE_INPUT_SOURCE_MISMATCH")
    _, rebuilt_issues = verify_input_evidence(state.model_dump(mode="json"), proof)
    if set(rebuilt_issues) != set(candidate["coverage_issues"]):
        raise ValueError("DECISION_CANDIDATE_COVERAGE_MISMATCH")
    if not isinstance(entry["outcomes"], list) or not 1 <= len(entry["outcomes"]) <= 5:
        raise ValueError("DECISION_NATIVE_OUTCOMES_REQUIRED")
    row = {
        "sample_id": state_hash,
        "state": state.model_dump(mode="json"),
        "parent_group": entry["parent_group"],
        "episode_id": entry["episode_id"],
        "split": entry["split"],
        "license": entry["license"],
        "evidence": [],
        "label_source": "verified-simulation-outcome",
        "formal_training_eligible": True,
        "acceptable_actions": [],
    }
    documents = [read_bound(root, ref) for ref in entry["outcomes"]]
    for document in documents:
        verify_input_outcome_geometry(proof, document)
    actions = [verify_native_label(row, document) for document in documents]
    if len(set(actions)) != len(actions):
        raise ValueError("DECISION_DUPLICATE_OUTCOME_ACTION")
    row["acceptable_actions"] = [a for a in ACTIONS if a in actions]
    return row, documents, proof


# 功能：为原生行为采集结果建立可迁移语料，逐条隔离失败；全局泄漏检查失败则整包不准入。
# 输入：只读来源清单/新输出目录；输出：正式语料、隔离原因、内容清单和计算出的准入结果。
def assemble(manifest_path: Path, output: Path):
    if manifest_path.stat().st_size > 32 * 1024**2:
        raise ValueError("DECISION_COLLECTION_MANIFEST_TOO_LARGE")
    manifest = decode_object(manifest_path.read_text("utf-8"))
    if (
        set(manifest) != {"schema_version", "entries"}
        or manifest["schema_version"] != "dronedream.decision-collection-manifest.v1"
        or not isinstance(manifest["entries"], list)
        or len(manifest["entries"]) > 150000
    ):
        raise ValueError("DECISION_COLLECTION_MANIFEST_INVALID")
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    rows, rejected, reasons = [], [], Counter()
    for index, entry in enumerate(manifest["entries"]):
        try:
            row, documents, proof = verified_entry(entry, manifest_path.parent)
            references = []
            for number, document in enumerate(documents):
                name = f"outcome-{index:06d}-{number}.json"
                sha = write_json(evidence / name, document)
                references.append({"path": "evidence/" + name, "sha256": sha})
            proof_name = f"input-{index:06d}.json"
            proof_reference = {
                "path": "evidence/" + proof_name,
                "sha256": write_json(evidence / proof_name, proof),
            }
            certificate = {
                "schema_version": "dronedream.decision-label-certificate.v1",
                "state_sha256": decision_digest(row["state"]),
                "parent_group": row["parent_group"],
                "episode_id": row["episode_id"],
                "label_source": row["label_source"],
                "acceptable_actions": row["acceptable_actions"],
                "verified": True,
                "verifier_version": "native-independent-geometry-v1",
                "outcome_references": references,
                "input_reference": proof_reference,
                "input_candidate_sha256": entry["candidate"]["sha256"],
            }
            name = f"certificate-{index:06d}.json"
            row["evidence"] = [
                {"path": "evidence/" + name, "sha256": write_json(evidence / name, certificate)}
            ]
            # 单条资格先核对；跨集合/布局冲突由完整集合检查，不能靠逐条写入绕过。
            audit_records([row], output)
            rows.append(row)
        except (ValueError, KeyError, TypeError, OSError) as error:
            reason = type(error).__name__ + ":" + str(error)[:300]
            reasons[reason] += 1
            rejected.append({"entry_index": index, "reason": reason})
    report = {
        "source_manifest_sha256": file_digest(manifest_path),
        "source_entries": len(manifest["entries"]),
        "verified_candidates": len(rows),
        "rejected_entries": len(rejected),
        "rejection_reasons": dict(reasons),
        "formal_decision_windows": 0,
        "gpu_ready": False,
        "execution_authority": False,
    }
    try:
        if rows:
            audit = audit_records(rows, output)
            report.update(
                audit=audit,
                formal_decision_windows=audit["formal_windows"],
                gpu_ready=audit["gpu_data_ready"] and not rejected,
            )
            with (output / "decisions.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        else:
            report["remaining"] = ["NO_INDEPENDENTLY_VERIFIED_DECISION_LABELS"]
    except ValueError as error:
        # 保留证据供修复分组，不输出看似可训练的语料，也不自动重新划分测试集。
        report["remaining"] = ["CORPUS_GLOBAL_INTEGRITY_FAILED:" + str(error)]
    write_json(output / "quarantine.json", rejected)
    write_json(output / "assembly-report.json", report)
    return report


# 功能：CLI 组装入口；输入：清单与独占目录；输出：已验证数量，未齐套返回非零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["gpu_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
