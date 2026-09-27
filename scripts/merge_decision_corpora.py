"""Merge verified simulation corpora with bound layout lineage; no relabeling or resplitting."""

import argparse
import copy
import json
from pathlib import Path

from dronedream_agent_core.decision_dataset import audit_records, file_digest, read_records
from dronedream_agent_core.decision_label_evidence import read_bound
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：内容寻址归档已核验JSON，避免同名文件互相覆盖；输入：独占根/对象；输出：相对引用。
def store_document(root, document):
    digest = decision_digest(document)
    relative = f"evidence/{digest}.json"
    path = root / relative
    payload = json.dumps(
        document, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    if path.exists():
        if path.read_text("utf-8") != payload:
            raise ValueError("DECISION_MERGE_CONTENT_CONFLICT")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    return {"path": relative, "sha256": file_digest(path)}


# 功能：重新验证每份语料及其结果，再迁移证据引用；保留原样本、标签、集合和父组身份。
# 输入：原语料JSONL列表、每份对应的原生成器收据、新目录；输出：全局审计，非GPU许可。
# 不复制运行目录、账户信息或整个地图目录；只迁移证书实际引用的JSON内容。
def merge(corpora, scene_receipts, output):
    if not corpora or len(corpora) != len(scene_receipts) or len(corpora) > 10000:
        raise ValueError("DECISION_MERGE_SOURCES_INVALID")
    inputs = []
    for corpus, receipt in zip(corpora, scene_receipts, strict=True):
        rows = read_records(corpus)
        audit_records(rows, corpus.parent)
        if any(row["label_source"] != "verified-simulation-outcome" for row in rows):
            raise ValueError("DECISION_MERGE_SIMULATION_ONLY")
        # 收据也经过严格JSON/大小/摘要核对；内容与实际地图的绑定由总审计验证。
        receipt_doc = read_bound(
            receipt.parent, {"path": receipt.name, "sha256": file_digest(receipt)}
        )
        inputs.append((corpus, rows, receipt_doc))
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    combined, lineage = [], {}
    for corpus, rows, receipt in inputs:
        receipt_ref = store_document(output, receipt)
        for original in rows:
            row = copy.deepcopy(original)
            refs = []
            for reference in row["evidence"]:
                certificate = read_bound(corpus.parent, reference)
                if certificate.get("schema_version") != "dronedream.decision-label-certificate.v1":
                    raise ValueError("DECISION_MERGE_CERTIFICATE_ONLY")
                certificate["input_reference"] = store_document(
                    output, read_bound(corpus.parent, certificate["input_reference"])
                )
                certificate["outcome_references"] = [
                    store_document(output, read_bound(corpus.parent, ref))
                    for ref in certificate["outcome_references"]
                ]
                refs.append(store_document(output, certificate))
            row["evidence"] = refs
            binding = {
                "parent_group": row["parent_group"],
                "map_sha256": row["state"]["map_sha256"],
                "receipt": receipt_ref,
            }
            previous = lineage.setdefault(row["parent_group"], binding)
            if previous["map_sha256"] != binding["map_sha256"]:
                raise ValueError("DECISION_MERGE_PARENT_MAP_CHANGED")
            combined.append(row)
    with (output / "layout-lineage.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": "dronedream.decision-layout-lineage.v1",
                "entries": list(lineage.values()),
            },
            stream,
            allow_nan=False,
        )
    # 无全局通过结果就不发布decisions.jsonl；不能按单份通过推断合并后无重复/泄漏。
    report = audit_records(combined, output)
    report["source_corpora"] = [file_digest(corpus) for corpus in corpora]
    with (output / "merge-report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
    with (output / "decisions.jsonl").open("x", encoding="utf-8") as stream:
        for row in combined:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return report


# 功能：显式成对提供原语料/场景收据，输出不可覆盖；输出：实际合并数量和准入缺口。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--scene-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.corpus, args.scene_receipt, args.output)))


if __name__ == "__main__":
    main()
