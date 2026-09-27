"""Train and calibrate a CPU baseline on audited labels, not model-invented answers."""

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.decision_dataset import audit_records, file_digest, read_records
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest
from dronedream_agent_core.laya_uav_training import (
    acceptable_mask,
    decision_metrics,
    encode_decision,
    fit_temperature,
    infer_logits,
    load_laya,
    set_cross_entropy,
    verify_package,
)
from dronedream_agent_core.uav_decision_student import FEATURE_SHA256, make_student, numeric_state


# 功能：独占保存机器可读报告；输入：新路径及 JSON 内容；输出：文件，不覆盖旧实验。
def save_json(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)


# 功能：真实集合标签为主要监督，教师仅作可关闭辅助；不回传梯度到教师目标。
# 输入：学生 logits、已验证标签掩码、可选训练集教师概率；输出：有限标量损失。
def student_loss(logits, accepted, teacher_probabilities=None):
    import torch

    supervised = set_cross_entropy(logits, accepted).mean()
    if teacher_probabilities is None:
        return supervised
    if (teacher_probabilities.shape != logits.shape
            or not bool(torch.isfinite(teacher_probabilities).all())
            or not bool((teacher_probabilities >= 0).all())
            or not torch.allclose(teacher_probabilities.sum(dim=1),
                                  torch.ones(len(logits), device=logits.device), atol=1e-5)):
        raise ValueError("DECISION_TEACHER_PROBABILITIES_INVALID")
    auxiliary = torch.nn.functional.kl_div(torch.log_softmax(logits / 2., dim=1),
        teacher_probabilities.detach(), reduction="batchmean") * 4.
    return .8 * supervised + .2 * auxiliary


# 功能：训练轻量对照、只用开发集选择、独立校准后导出回载；无飞控副作用。
# 输入：严格语料、新目录、有限训练预算；输出：非部署模型及完整分集合成绩。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--teacher-package", type=Path)
    parser.add_argument("--teacher-identity", type=Path)
    parser.add_argument("--teacher-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or not 0 <= args.seed < 2**32:
        parser.error("invalid bounded student configuration")
    if bool(args.teacher_package) != bool(args.teacher_identity):
        parser.error("teacher package and training identity must be supplied together")
    rows = read_records(args.corpus)
    audit = audit_records(rows, args.corpus.parent, smoke=args.smoke)
    if not args.smoke and not audit["gpu_data_ready"]:
        raise ValueError("DECISION_STUDENT_FORMAL_DATA_NOT_READY")
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tensors, masks = {}, {}
    for split in ("train", "development", "calibration", "test", "stress"):
        part = [row for row in rows if row["split"] == split]
        if not part:
            raise ValueError("DECISION_STUDENT_EMPTY_SPLIT:" + split)
        tensors[split] = torch.tensor([numeric_state(DecisionStateV2.model_validate_json(
            json.dumps(row["state"]))) for row in part], dtype=torch.float32)
        masks[split] = torch.tensor([acceptable_mask(row["acceptable_actions"], list(ACTIONS))
                                    for row in part], dtype=torch.bool)
    args.output.mkdir(parents=True, exist_ok=False)
    width = tensors["train"].shape[1]
    teacher_probabilities, teacher_report = None, None
    if args.teacher_package:
        teacher_started = time.perf_counter()
        teacher_receipt = verify_package(args.teacher_package)
        teacher_identity = json.loads(args.teacher_identity.read_text("utf-8"))
        identity_hash = teacher_identity.pop("run_sha256")
        if (decision_digest(teacher_identity) != identity_hash
                or teacher_receipt["run_sha256"] != identity_hash
                or teacher_identity["corpus_sha256"] != file_digest(args.corpus)
                or teacher_receipt["smoke_only"] != args.smoke
                or teacher_receipt["actions"] != list(ACTIONS)
                or teacher_receipt["state_schema"] != "dronedream.decision-state.v2"):
            raise ValueError("DECISION_DISTILLATION_TEACHER_IDENTITY_MISMATCH")
        teacher = load_laya(args.teacher_package, args.teacher_device)
        training_rows = [r for r in rows if r["split"] == "train"]
        encoded = [encode_decision(teacher.tok, DecisionStateV2.model_validate_json(
            json.dumps(row["state"]))) for row in training_rows]
        logits = infer_logits(teacher, encoded)
        temperature = teacher_receipt["calibration"]["temperature"]
        teacher_probabilities = torch.softmax(logits / temperature / 2., dim=1)
        target_file = args.output / "teacher-train-probabilities.safetensors"
        save_file({"probabilities": teacher_probabilities}, str(target_file))
        predicted = logits.argmax(dim=1)
        disagreements = int((~masks["train"].gather(1, predicted[:, None])).sum())
        teacher_report = {"run_sha256": identity_hash,
            "weights_sha256": file_digest(args.teacher_package / "model.safetensors"),
            "soft_target_split": "train", "rows": len(training_rows),
            "disagreements_with_verified_labels": disagreements,
            "auxiliary_weight": .2, "distillation_temperature": 2.,
            "teacher_calibration_temperature": temperature,
            "soft_targets_sha256": file_digest(target_file),
            "soft_target_seconds": time.perf_counter()-teacher_started}
        # 教师仅提供训练集辅助目标；不覆盖真实标签，也不将测试集软目标送进训练。
        del teacher
        if args.teacher_device == "cuda":
            torch.cuda.empty_cache()
    model = make_student(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, best_loss, losses = None, float("inf"), []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(tensors["train"])).split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensors["train"][indices])
            loss = student_loss(logits, masks["train"][indices],
                teacher_probabilities[indices] if teacher_probabilities is not None else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = float(set_cross_entropy(model(tensors["development"]),
                                               masks["development"]).mean())
        losses.append({"epoch": epoch, "development_loss": dev_loss})
        if dev_loss < best_loss:
            best_loss = dev_loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best, strict=True)
    model.eval()
    with torch.inference_mode():
        calibration = fit_temperature(model(tensors["calibration"]), masks["calibration"])
        metrics = {split: decision_metrics(model(tensors[split]), masks[split],
                                          calibration["temperature"])
                   for split in ("test", "stress")}
    weights = args.output / "student.safetensors"
    save_file(model.state_dict(), str(weights))
    reloaded = make_student(width)
    reloaded.load_state_dict(load_file(str(weights)), strict=True)
    reloaded.eval()
    with torch.inference_mode():
        difference = float((model(tensors["test"])-reloaded(tensors["test"])).abs().max())
    if difference > 1e-6:
        raise ValueError("DECISION_STUDENT_EXPORT_MISMATCH")
    report = {"schema_version": "dronedream.decision-student-training.v1",
        "feature_sha256": FEATURE_SHA256, "input_width": width, "hidden_width": 64,
        "actions": list(ACTIONS), "seed": args.seed, "epochs": args.epochs,
        "corpus_sha256": file_digest(args.corpus), "weights_sha256": file_digest(weights),
        "audit": audit, "calibration": calibration, "metrics": metrics,
        "development_curve": losses, "reload_max_absolute_error": difference,
        "seconds": time.perf_counter()-started, "smoke_only": args.smoke,
        "model_kind": "supervised-MLP-with-Laya-distillation" if teacher_report else
                      "supervised-MLP-baseline-not-Laya-not-distilled",
        "teacher": teacher_report,
        "execution_authority": False, "formal_training_additions": 0}
    save_json(args.output / "report.json", report)
    print(json.dumps({"seconds": report["seconds"], "reload_error": difference,
                      "smoke_only": args.smoke, "formal_windows": audit["formal_windows"]}))


if __name__ == "__main__":
    main()
