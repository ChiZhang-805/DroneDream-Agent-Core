"""Measure actual local inference; does not grant control authority or rent a GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

from dronedream_agent_core.decision_dataset import read_records
from dronedream_agent_core.decision_state_adapter import DecisionStateV2
from dronedream_agent_core.laya_uav_training import encode_decision, load_laya, model_batch


# 功能：在固定真实权重上比较线程配置；输入：候选文件与新报告；输出：端到端耗时/峰值差异。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize-encoder", action="store_true")
    args = parser.parse_args()
    import torch

    rows = read_records(args.candidates)
    states = [DecisionStateV2.model_validate_json(json.dumps(r["state"]))
              for r in rows[:5]]
    torch.set_num_threads(2)
    agent = load_laya(args.model, "cpu")
    agent.model.eval()
    reports, reference = [], None
    with torch.inference_mode():
        settings = ((4, False), (4, True), (8, True)) if args.quantize_encoder else (
            (1, False), (2, False), (4, False), (8, False))
        quantized = False
        for threads, quantize in settings:
            if quantize and not quantized:
                agent.model.encoder = torch.ao.quantization.quantize_dynamic(
                    agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
                quantized = True
            torch.set_num_threads(threads)
            times, outputs, tokens = [], [], []
            for state in states:
                tick = time.perf_counter()
                item = encode_decision(agent.tok, state)
                logits, _ = agent.model(**model_batch([item], agent.tok, agent.device))
                outputs.append(logits.detach().clone())
                tokens.append(len(item["ids"]))
                times.append((time.perf_counter() - tick)*1000)
            outputs = torch.cat(outputs)
            reference = outputs if reference is None else reference
            reports.append({"threads": threads, "quantized_encoder": quantized,
                            "samples": len(times), "milliseconds": times,
                            "median_ms": statistics.median(times), "worst_ms": max(times),
                            "max_logit_difference": float((outputs-reference).abs().max()),
                            "encoded_tokens": tokens})
    result = {"schema_version": "dronedream.laya-cpu-benchmark.v1", "measurements": reports,
              "model": args.model.name, "formal_training_additions": 0,
              "flight_authority": False,
              "warning": "five samples per setting; not a p95/p99 qualification"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
