"""Portable tiny decision baseline with group isolation and held-out calibration."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from dataclasses import asdict

from dronedream_agent_core.decision_shadow import ACTIONS, StageState, numeric_features


# 功能：验证决策训练语料，禁止无标签回放、重复样本和同任务跨集合泄漏。
# 输入：JSON 样本数组，显式烟测开关；输出：经验证语料，不推断缺失标签。
def validate_corpus(rows, *, smoke=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError("CORPUS_EMPTY")
    groups, identities = {}, set()
    for row in rows:
        if row["id"] in identities:
            raise ValueError("CORPUS_DUPLICATE_ID")
        identities.add(row["id"])
        group, split = row["group"], row["split"]
        if not isinstance(group, str) or not group or split not in {"train", "calibration", "test"}:
            raise ValueError("CORPUS_GROUP_INVALID")
        if groups.setdefault(group, split) != split:
            raise ValueError("CORPUS_GROUP_LEAKAGE")
        StageState(**row["state"])
        if row["reference"] not in ACTIONS:
            raise ValueError("CORPUS_LABEL_MISSING")
        if smoke:
            if row.get("label_source") != "synthetic-rule-contract":
                raise ValueError("SMOKE_CORPUS_SOURCE_INVALID")
        elif (row.get("label_source") not in {"reviewed-expert", "verified-simulation-outcome"}
              or row.get("formal_training_eligible") is not True
              or not isinstance(row.get("label_evidence_sha256"), str)
              or len(row["label_evidence_sha256"]) != 64
              or any(c not in "0123456789abcdef" for c in row["label_evidence_sha256"])
              or not isinstance(row.get("label_evidence_path"), str)
              or row.get("license") not in {"MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0", "proprietary-user-owned"}):
            raise ValueError("CORPUS_LABEL_EVIDENCE_REQUIRED")
    for split in ("train", "calibration", "test"):
        coverage = set(r["reference"] for r in rows if r["split"] == split)
        if not coverage or (not smoke or split == "train") and coverage != set(ACTIONS):
            raise ValueError("CORPUS_ACTION_COVERAGE_INCOMPLETE:" + split)
    return rows


# 功能：仅用训练集学习小分类器、校准集确定温度，测试集只用于最终报告。
# 输入：通过标签来源校验的语料及新目录；输出：可审计 JSON 权重/概率指标，不授予控制权限。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000:
        parser.error("epochs must be 1..1000")
    if args.corpus.stat().st_size > 100 * 1024**2:
        raise ValueError("CORPUS_TOO_LARGE")
    raw = args.corpus.read_bytes()
    rows = validate_corpus(json.loads(raw), smoke=args.smoke)
    if not args.smoke:
        from audit_tartanair_sample import digest, portable_path
        corpus_root = args.corpus.parent.resolve()
        verified = {}
        for row in rows:
            evidence = (corpus_root / portable_path(row["label_evidence_path"])).resolve()
            if not evidence.is_relative_to(corpus_root) or not evidence.is_file():
                raise ValueError("CORPUS_EVIDENCE_PATH_INVALID")
            if evidence not in verified:
                verified[evidence] = digest(evidence)
            if verified[evidence] != row["label_evidence_sha256"]:
                raise ValueError("CORPUS_EVIDENCE_HASH_MISMATCH")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(24)
    random.seed(24)
    partitions = {split: [r for r in rows if r["split"] == split] for split in ("train", "calibration", "test")}
    tensors = {split: (torch.tensor([numeric_features(StageState(**r["state"])) for r in part], dtype=torch.float32),
                      torch.tensor([ACTIONS.index(r["reference"]) for r in part])) for split, part in partitions.items()}
    width = tensors["train"][0].shape[1]
    model = torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.Tanh(), torch.nn.Linear(32, len(ACTIONS)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    counts = torch.bincount(tensors["train"][1], minlength=len(ACTIONS)).float()
    weights = counts.sum()/(len(ACTIONS)*counts)
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tensors["train"][0]), tensors["train"][1], weight=weights)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(tensors["calibration"][0])
        candidates = torch.logspace(math.log10(.5), math.log10(5), 41)
        losses = [torch.nn.functional.cross_entropy(logits/t, tensors["calibration"][1]).item() for t in candidates]
        temperature = float(candidates[min(range(len(losses)), key=losses.__getitem__)])
        test_logits = model(tensors["test"][0])
        probabilities = torch.softmax(test_logits/temperature, dim=1)
        actual = tensors["test"][1]
        predicted = probabilities.argmax(dim=1)
        report = {"corpus_sha256": hashlib.sha256(raw).hexdigest(), "smoke_only": args.smoke,
            "label_source_counts": dict(Counter(r["label_source"] for r in rows)),
            "partition_sizes": {k: len(v) for k, v in partitions.items()},
            "temperature": temperature, "test_accuracy": float((predicted == actual).float().mean()),
            "test_nll_uncalibrated": float(torch.nn.functional.cross_entropy(test_logits, actual)),
            "test_nll_calibrated": float(torch.nn.functional.cross_entropy(test_logits/temperature, actual)),
            "per_class_recall": {action: (float((predicted[actual == i] == i).float().mean()) if bool((actual == i).any()) else None) for i, action in enumerate(ACTIONS)},
            "partition_label_counts": {split: dict(Counter(r["reference"] for r in part)) for split, part in partitions.items()},
            "execution_authority": False, "flight_acceptance": False,
            "formal_training_additions": 0, "torch_version": torch.__version__,
            "warning": "Calibrated classifier probabilities are not safety probabilities. Synthetic smoke scores do not establish drone capability."}
    artifact = {"format": "dronedream.stage-mlp.shadow.v1", "actions": ACTIONS,
        "feature_fields": list(asdict(StageState(None, None, None, None, None, None))),
        "feature_encoding": "value/(1+abs(value)), known-mask; None=0,0",
        "activation": "tanh", "temperature": temperature, "execution_authority": False,
        "smoke_only": args.smoke, "weights": {k: v.tolist() for k, v in model.state_dict().items()}}
    for name, value in (("report.json", report), ("shadow-weights.json", artifact)):
        with (args.output/name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
