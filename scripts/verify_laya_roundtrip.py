"""Recalibrate an existing smoke checkpoint, then verify actual export/inference."""

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.decision_dataset import audit_records, file_digest, read_records
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2
from dronedream_agent_core.laya_uav_training import (
    acceptable_mask,
    encode_decision,
    export_laya,
    fit_temperature,
    infer_logits,
    load_laya,
    verify_export,
)
from dronedream_agent_core.local_stage_decision_port import LocalStageDecisionPort


# 功能：对真实烟测权重重新校准并回载，量测本机异步时效；不授权飞行。
# 输入：烟测语料/权重/新输出目录；输出：校准、数值一致性和实际通道结果。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    rows = read_records(args.corpus)
    audit_records(rows, args.corpus.parent, smoke=True)
    receipt = json.loads((args.model / "uav-training-receipt.json").read_text("utf-8"))
    if receipt.get("smoke_only") is not True:
        raise ValueError("SMOKE_CHECKPOINT_REQUIRED")
    agent = load_laya(args.model, "cpu")
    part = [r for r in rows if r["split"] == "calibration"]
    items = [encode_decision(agent.tok, DecisionStateV2.model_validate_json(
        json.dumps(r["state"]))) for r in part]
    accepted = torch.tensor([acceptable_mask(r["acceptable_actions"], list(ACTIONS))
                             for r in part], dtype=torch.bool)
    calibration = fit_temperature(infer_logits(agent, items), accepted)
    # 旧温度下的指标不能继续挂在新校准上；明确记录这是补充烟测，不是重新训练。
    receipt.pop("metrics", None)
    receipt.update(calibration=calibration,
                   recalibrated_from_weights_sha256=file_digest(args.model / "model.safetensors"))
    args.output.mkdir(parents=True, exist_ok=False)
    export_laya(agent, args.output / "model", receipt, calibration["temperature"])
    parity = verify_export(agent, args.output / "model", items[:2])
    del agent
    port = LocalStageDecisionPort(args.output / "model")
    results = []
    try:
        end = time.monotonic() + 65
        while not port.ready and not port.closed and time.monotonic() < end:
            port.poll()
            time.sleep(.01)
        if not port.ready:
            raise ValueError("LAYA_PORT_START_FAILED:" + port.last_reason)
        for index in range(3):
            state = DecisionStateV2.model_validate_json(json.dumps(part[index]["state"]))
            port.submit(state)
            tick = time.monotonic()
            result = port.poll()
            while port.inflight is not None and not port.closed:
                result = port.poll()
                if result is not None:
                    break
                time.sleep(.005)
            results.append({"elapsed_ms": (time.monotonic() - tick) * 1000,
                            "reason": port.last_reason, "advice": result})
    finally:
        port.close()
    report = {"parity": parity, "channel_results": results,
              "formal_training_additions": 0, "flight_authority": False}
    with (args.output / "roundtrip.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
