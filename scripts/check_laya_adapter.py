"""Offline sanity and option-order checks; never grants flight authority."""
import argparse
import json
import os
from pathlib import Path
import time


# 功能：独立复现官方基础任务和无人机边界任务，检查选项顺序敏感性。
# 输入：已固定版本的本地权重和新报告文件；输出：原始回答与耗时，不产生控制量。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import laya
    torch.set_num_threads(4)
    agent = laya.load(str(args.model.resolve()), device="cpu")
    tasks = [
        ("billing", "We were billed twice for March. Please refund the duplicate today.",
         "Which department should handle this request?",
         {"billing": "invoices, payments, refunds", "technical": "bugs and software problems", "sales": "new purchases"}),
        ("technical", "The software crashes with an error every time I open a file. Please fix this bug.",
         "Which department should handle this request?",
         {"billing": "invoices, payments, refunds", "technical": "bugs and software problems", "sales": "new purchases"}),
        ("wait", "A pedestrian is crossing directly in front of the drone. The drone can safely stop and hover. Wait until the pedestrian passes.",
         "Select the next safe behavior.",
         {"follow_route": "continue flying forward", "slow_down": "continue at reduced speed", "wait": "stop and wait", "replan": "find another route", "request_observation": "request missing sensor information"}),
    ]
    results = []
    for label, state, instructions, options in tasks:
        for reverse in (False, True):
            criteria = dict(reversed(list(options.items()))) if reverse else options
            start = time.perf_counter()
            result = agent.predict(state, {"decision": {"type": "choice", "instructions": instructions, "criteria": criteria}})
            results.append({"reference": label, "state": state, "criteria": criteria,
                "answer": result, "latency_ms": (time.perf_counter() - start) * 1000})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"model": str(args.model), "execution_authority": False, "results": results}, stream, indent=2)
    print(json.dumps([{"expected": r["reference"], "order": list(r["criteria"]), "answer": r["answer"]["answers"]} for r in results]))


if __name__ == "__main__":
    main()
