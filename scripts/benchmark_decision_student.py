"""Measure the full CPU feature-plus-student path on real replay inputs, not quality."""

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.decision_dataset import file_digest, read_records
from dronedream_agent_core.decision_state_adapter import DecisionStateV2
from dronedream_agent_core.uav_decision_student import FEATURE_SHA256, make_student, numeric_state


# 功能：校验导出身份后实测状态解析/特征/分类总耗时；没有标签时绝不报告正确率。
# 输入：模型目录、真实候选、新输出文件；输出：样本数/分位耗时与 smoke 身份。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.model / "report.json").read_text("utf-8"))
    weights = args.model / "student.safetensors"
    if (report["feature_sha256"] != FEATURE_SHA256
            or file_digest(weights) != report["weights_sha256"]):
        raise ValueError("DECISION_STUDENT_PACKAGE_MISMATCH")
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    model = make_student(report["input_width"], report["hidden_width"])
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval()
    rows = read_records(args.candidates)
    elapsed = []
    with torch.inference_mode():
        for index in range(5+len(rows)):
            raw = rows[max(0, index-5)]["state"]
            tick = time.perf_counter()
            state = DecisionStateV2.model_validate_json(json.dumps(raw))
            logits = model(torch.tensor([numeric_state(state)], dtype=torch.float32))
            probabilities = torch.softmax(logits.float()/report["calibration"]["temperature"],
                                          dim=1).tolist()
            if len(probabilities[0]) != 5:
                raise ValueError("DECISION_STUDENT_OUTPUT_INVALID")
            if index >= 5:
                elapsed.append((time.perf_counter()-tick)*1000)
    ordered = sorted(elapsed)
    result = {"samples": len(elapsed), "warmups": 5, "threads": 2,
        "p50_ms": ordered[int((len(ordered)-1)*.5)],
        "p95_ms": ordered[int((len(ordered)-1)*.95)],
        "p99_ms": ordered[int((len(ordered)-1)*.99)], "max_ms": max(ordered),
        "scope": "JSON validation + causal numeric features + CPU MLP + calibrated softmax",
        "concurrent_perception_load": False, "includes_IPC": False,
        "weights_sha256": file_digest(weights), "candidate_sha256": file_digest(args.candidates),
        "smoke_only": report["smoke_only"], "accuracy": None, "execution_authority": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
