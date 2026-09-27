"""Train the real Laya encoder/head on audited UAV decisions; never enables flight."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from importlib.metadata import version
from pathlib import Path

from dronedream_agent_core.decision_dataset import audit_records, file_digest, read_records
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest
from dronedream_agent_core.laya_uav_training import (
    acceptable_mask,
    decision_metrics,
    encode_decision,
    export_laya,
    fit_temperature,
    infer_logits,
    load_laya,
    model_batch,
    set_cross_entropy,
    training_option_order,
    verify_export,
)


# 功能：独占写入 JSON 证据，避免覆盖之前的运行结果。
# 输入：新文件和 JSON 对象；输出：落盘文件，写入异常向调用者传播。
def write_report(path: Path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


# 功能：严格解析训练参数，正式训练必须通过数据准入；烟测不能生成正式包。
# 输入：CLI 参数；输出：真实 Laya 训练/校准/导出和资源测量结果。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--option-order-augmentation", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--probe-steps", type=int, default=0,
                        help="Bounded backward/optimizer smoke, not a trained artifact")
    args = parser.parse_args()
    if (not 1 <= args.epochs <= 8 or not 1 <= args.micro_batch <= args.effective_batch <= 128
            or args.effective_batch % args.micro_batch or not 1 <= args.threads <= 16
            or not 0 <= args.seed < 2**32 or not 0 <= args.probe_steps <= 10
            or args.probe_steps and not args.smoke):
        parser.error("invalid bounded training configuration")
    rows = read_records(args.corpus)
    audit = audit_records(rows, args.corpus.parent, smoke=args.smoke)
    if not args.smoke and not audit["gpu_data_ready"]:
        raise ValueError("LAYA_FORMAL_DATA_NOT_READY:" + ",".join(audit["readiness_issues"]))
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    agent = load_laya(args.model, args.device, args.max_len)
    agent.model.encoder.gradient_checkpointing_enable()
    agent.model.head_checkpointing = True
    partitions, targets = {}, {}
    started = time.perf_counter()
    for split in ("train", "development", "calibration", "test", "stress"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            raise ValueError("LAYA_EMPTY_SPLIT:" + split)
        partitions[split] = [encode_decision(agent.tok,
            DecisionStateV2.model_validate_json(json.dumps(r["state"])), max_len=args.max_len)
            for r in part]
        targets[split] = torch.tensor([acceptable_mask(r["acceptable_actions"], list(ACTIONS))
                                      for r in part], dtype=torch.bool)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"output", "resume", "corpus", "model"}}
    identity = {"config": config, "corpus_sha256": file_digest(args.corpus),
                "base_weights_sha256": file_digest(args.model / "model.safetensors"),
                "base_config_sha256": file_digest(args.model / "rl_agent_config.json"),
                "tokenizer_sha256": file_digest(args.model / "tokenizer/tokenizer.json"),
                "code_sha256": {p.name: file_digest(p) for p in (
                    Path(__file__), *sorted((Path(__file__).resolve().parents[1] /
                    "src/dronedream_agent_core").glob("decision_*.py")),
                    Path(__file__).resolve().parents[1] /
                    "src/dronedream_agent_core/laya_uav_training.py")},
                "packages": {name: version(name) for name in
                             ("torch", "laya", "transformers", "numpy", "pydantic")}}
    run_hash = decision_digest(identity)
    write_report(args.output / "input-audit.json", audit)
    write_report(args.output / "training-identity.json", {**identity, "run_sha256": run_hash})
    head = [p for name, p in agent.model.named_parameters()
            if not name.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": list(agent.model.encoder.parameters()), "lr": 1e-5},
        {"params": head, "lr": 5e-5}], weight_decay=.01)
    total_steps = args.epochs * math.ceil(len(partitions["train"]) / args.effective_batch)
    warmup = max(1, int(total_steps * .05))

    # 功能：按实际优化步线性预热并衰减，恢复时由 scheduler 状态保持连续。
    # 输入：步号；输出：学习率倍数，不修改标签或时钟。
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_amp = args.device == "cuda" and torch.cuda.is_bf16_supported()
    initial_epoch, best_loss, steps = 0, float("inf"), 0
    best_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["run_sha256"] != run_hash:
            raise ValueError("LAYA_RESUME_IDENTITY_MISMATCH")
        agent.model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        initial_epoch, best_loss, steps = (checkpoint["next_epoch"], checkpoint["best_loss"],
                                          checkpoint["steps"])
        best_state = checkpoint["best_model"]
        if initial_epoch >= args.epochs:
            raise ValueError("LAYA_RESUME_ALREADY_FINISHED")
    step_times = []
    for epoch in range(initial_epoch, args.epochs):
        # 换序只作用于训练集；标签同序重映射，校准和评价始终使用标准行为顺序。
        training_items, training_targets = partitions["train"], targets["train"]
        if args.option_order_augmentation:
            train_rows = [r for r in rows if r["split"] == "train"]
            orders = [training_option_order(r["sample_id"], args.seed, epoch) for r in train_rows]
            training_items = [encode_decision(agent.tok,
                DecisionStateV2.model_validate_json(json.dumps(row["state"])),
                max_len=args.max_len, order=order)
                for row, order in zip(train_rows, orders, strict=True)]
            training_targets = torch.tensor([
                acceptable_mask(row["acceptable_actions"], list(order))
                for row, order in zip(train_rows, orders, strict=True)], dtype=torch.bool)
        agent.model.train()
        # 首轮适配决策头；probe 从一开始测全量反传，不能低估正式显存需求。
        for parameter in agent.model.encoder.parameters():
            parameter.requires_grad_(bool(epoch or args.probe_steps))
        indices = list(range(len(partitions["train"])))
        random.shuffle(indices)
        for start in range(0, len(indices), args.effective_batch):
            group = indices[start:start + args.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            tick = time.perf_counter()
            if args.device == "cuda":
                torch.cuda.synchronize()
                tick = time.perf_counter()
            for micro in range(0, len(group), args.micro_batch):
                chosen = group[micro:micro + args.micro_batch]
                batch = model_batch([training_items[i] for i in chosen], agent.tok,
                                    agent.device)
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    logits, _ = agent.model(**batch)
                    loss = set_cross_entropy(logits, training_targets[chosen].to(agent.device))
                    # 最后不足一个累积批次时按真实样本数归一化，不能少算梯度。
                    loss = loss.sum() / len(group)
                loss.backward()
                group_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            steps += 1
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            step_times.append(elapsed)
            print(json.dumps({"epoch": epoch, "step": steps, "seconds": elapsed,
                              "loss": group_loss}), flush=True)
            if args.probe_steps and steps >= args.probe_steps:
                report = {"smoke_only": True, "formal_training_additions": 0,
                          "actual_laya_backward": True, "encoder_trainable": True,
                          "optimizer_steps": steps, "step_seconds": step_times,
                          "effective_batch": args.effective_batch,
                          "max_encoded_tokens": max(len(i["ids"]) for p in partitions.values()
                                                    for i in p),
                          "device": str(agent.device), "flight_authority": False,
                          "run_sha256": run_hash,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                          if args.device == "cuda" else None}
                write_report(args.output / "probe-report.json", report)
                return
        dev_logits = infer_logits(agent, partitions["development"], args.micro_batch)
        dev_loss = float(set_cross_entropy(dev_logits, targets["development"]).mean())
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_state = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()}
        # 每个 epoch 独立 checkpoint，包含最佳模型，恢复不借用可能变动的外部文件。
        torch.save({"run_sha256": run_hash, "model": agent.model.state_dict(),
                    "best_model": best_state, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                    "best_loss": best_loss, "steps": steps, "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else []},
                   args.output / f"checkpoint-epoch-{epoch + 1}.pt")
        write_report(args.output / f"development-epoch-{epoch + 1}.json",
                     decision_metrics(dev_logits, targets["development"]))
    agent.model.load_state_dict(best_state, strict=True)
    calibration = fit_temperature(infer_logits(agent, partitions["calibration"], args.micro_batch),
                                  targets["calibration"])
    metrics = {}
    for split in ("test", "stress"):
        logits = infer_logits(agent, partitions[split], args.micro_batch)
        metrics[split] = {"raw": decision_metrics(logits, targets[split]),
                          "calibrated": decision_metrics(logits, targets[split],
                                                         calibration["temperature"])}
    receipt = {"schema_version": "dronedream.laya-uav-training.v1", "run_sha256": run_hash,
               "smoke_only": args.smoke, "flight_authority": False,
               "state_schema": "dronedream.decision-state.v2", "actions": list(ACTIONS),
               "calibration": calibration, "metrics": metrics, "best_development_loss": best_loss,
               "optimizer_steps": steps, "seconds": time.perf_counter() - started,
               "device": str(agent.device), "torch_version": torch.__version__}
    export_laya(agent, args.output / "model", receipt, calibration["temperature"])
    parity = verify_export(agent, args.output / "model", partitions["test"][:2])
    write_report(args.output / "export-parity.json", parity)
    write_report(args.output / "training-report.json", receipt)


if __name__ == "__main__":
    main()
