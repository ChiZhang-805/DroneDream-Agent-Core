"""Actual Laya supervised fine-tuning primitives, shared with offline inference.

Torch/Laya are imported lazily: installing the app must not load a training runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .decision_dataset import file_digest
from .decision_shadow import ACTIONS
from .decision_state_adapter import DecisionStateV2, decision_question, model_state


# 功能：用同一模板打包训练/推理输入；只删最早历史，绝不截断当前风险或选项。
# 输入：tokenizer、v2 状态、模型支持的长度与选项排列；输出：完整 token/标记及截减记录。
def encode_decision(tokenizer, state: DecisionStateV2, *, max_len: int = 1024,
                    head_max_len: int = 256, order: tuple[str, ...] = ACTIONS) -> dict:
    from laya.common import QTYPES, build_sequence, render_options

    if not 256 <= max_len <= 8192 or not 64 <= head_max_len < max_len:
        raise ValueError("LAYA_SEQUENCE_BUDGET_INVALID")
    question = decision_question(order)
    # 上游会静默裁剪 instructions/options；先验证本任务固定头部全部装得下。
    ins_ids = tokenizer("choice question: " + question["ins"],
                        add_special_tokens=False)["input_ids"]
    options = [tokenizer(" " + option, add_special_tokens=False)["input_ids"]
               for option in render_options(question)]
    if (any(len(o) > 48 for o in options)
            or len(ins_ids) + sum(len(o) + 1 for o in options) > head_max_len):
        raise ValueError("LAYA_QUESTION_WOULD_TRUNCATE")
    for count in range(len(state.history), -1, -1):
        text = model_state(state, history_limit=count)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        empty, _ = build_sequence(tokenizer, "", question, max_len, head_max_len, state_ids=[])
        if len(empty) + len(tokens) > max_len:
            continue
        ids, markers = build_sequence(tokenizer, text, question, max_len, head_max_len,
                                      state_ids=tokens)
        if len(markers) != len(ACTIONS) or len(ids) != len(empty) + len(tokens):
            raise ValueError("LAYA_SEQUENCE_TRUNCATION_DETECTED")
        return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                "history_retained": count, "option_order": list(order)}
    raise ValueError("LAYA_CURRENT_STATE_EXCEEDS_CONTEXT")


# 功能：把合格动作集合映射到当前选项顺序，防止换序增强产生错标签。
# 输入：每条样本的合格动作和编码结果；输出：布尔集合监督，不凭空编概率。
def acceptable_mask(labels: list[str], order: list[str]) -> list[bool]:
    if (not labels or len(set(labels)) != len(labels) or not set(labels) <= set(ACTIONS)
            or len(order) != len(ACTIONS) or set(order) != set(ACTIONS)):
        raise ValueError("LAYA_LABEL_OR_ORDER_INVALID")
    return [action in labels for action in order]


# 功能：以样本身份/种子/轮次产生可复现选项排列，不依赖全局 RNG 或标签。
# 输入：非空样本摘要、非负种子/轮次；输出：五个完整行为 ID 排列。
def training_option_order(sample_id: str, seed: int, epoch: int) -> tuple[str, ...]:
    import hashlib

    if (not sample_id or type(seed) is not int or seed < 0
            or type(epoch) is not int or epoch < 0):
        raise ValueError("LAYA_OPTION_AUGMENTATION_IDENTITY_INVALID")
    return tuple(sorted(ACTIONS, key=lambda action: hashlib.sha256(
        f"{sample_id}:{seed}:{epoch}:{action}".encode()).digest()))


# 功能：最大化合格集合的总概率，唯一标签时等价交叉熵；不惩罚另一合理动作。
# 输入：B×5 logits 和同形布尔集合；输出：每条样本损失，拒绝空集合和非法形状。
def set_cross_entropy(logits, accepted):
    import torch

    if (logits.ndim != 2 or logits.shape[1] != len(ACTIONS)
            or accepted.shape != logits.shape or accepted.dtype != torch.bool
            or not bool(accepted.any(dim=1).all()) or not bool(torch.isfinite(logits).all())):
        raise ValueError("LAYA_SET_LOSS_INPUT_INVALID")
    return torch.logsumexp(logits.float(), dim=1) - torch.logsumexp(
        logits.float().masked_fill(~accepted, float("-inf")), dim=1)


# 功能：将上游 collate 输出缩减为真实 forward 参数，不把 target/meta 传入模型。
# 输入：编码样本和目标设备；输出：模型张量字典。
def model_batch(items: list[dict], tokenizer, device):
    from laya.common import collate_items

    batch = collate_items([[item] for item in items], tokenizer.pad_token_id)
    return {key: batch[key].to(device) for key in
            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


# 功能：严格加载本地真实 Laya；拒绝 CUDA 请求被上游静默退回 CPU。
# 输入：已固定权重目录、设备、最大长度；输出：实际编码器/决策头与 tokenizer。
def load_laya(model_path: Path, device: str, max_len: int = 1024):
    import torch
    from laya import Agent

    if not model_path.is_dir() or not (model_path / "model.safetensors").is_file():
        raise ValueError("LAYA_LOCAL_CHECKPOINT_REQUIRED")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("LAYA_CUDA_REQUIRED_BUT_UNAVAILABLE")
    agent = Agent(str(model_path.resolve()), device=device)
    supported = getattr(agent.model.encoder.config, "max_position_embeddings", 0)
    if supported < max_len:
        raise ValueError("LAYA_ENCODER_CONTEXT_UNSUPPORTED")
    agent.cfg["max_len"] = max_len
    agent.cfg["head_max_len"] = 256
    agent.cfg["temperature_by_options"] = {}
    agent.temperature_by_options = {}
    agent.temperature = [1.0, 1.0, 1.0]
    # act_head 的预训练二分类不是飞行许可；本方案不使用也不训练它。
    for parameter in agent.model.act_head.parameters():
        parameter.requires_grad_(False)
    return agent


# 功能：只在给定集合上测 logits；切换 eval 禁止 dropout 污染评估。
# 输入：真实模型和编码集合；输出：CPU logits，不附加安全含义。
def infer_logits(agent, items: list[dict], batch_size: int = 1):
    import torch

    if not items or batch_size < 1:
        raise ValueError("LAYA_INFERENCE_BATCH_INVALID")
    agent.model.eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            logits, _ = agent.model(**model_batch(items[offset:offset + batch_size],
                                                 agent.tok, agent.device))
            output.append(logits.detach().float().cpu())
    return torch.cat(output)


# 功能：仅用唯一标签校准温度，避免集合标签让 ECE/NLL 定义含糊。
# 输入：独立校准 logits 与合格集合；输出：温度和有效样本量，无有效标签则失败。
def fit_temperature(logits, accepted) -> dict:
    import torch

    unique = accepted.sum(dim=1) == 1
    if not bool(unique.any()):
        raise ValueError("LAYA_UNIQUE_CALIBRATION_LABELS_REQUIRED")
    z, y = logits[unique], accepted[unique].long().argmax(dim=1)
    # SDK 只接受 [0.5, 5]；超出范围会在回载时被静默裁剪，破坏校准一致性。
    candidates = torch.cat((torch.logspace(math.log10(.5), math.log10(5), 81),
                            torch.tensor([.5, 1.0, 5.0]))).clamp(.5, 5)
    losses = [torch.nn.functional.cross_entropy(z / t, y).item() for t in candidates]
    index = min(range(len(losses)), key=losses.__getitem__)
    return {"temperature": float(candidates[index]), "unique_rows": int(unique.sum()),
            "calibration_nll": losses[index], "method": "held-out-temperature-v1"}


# 功能：区分集合准确率和唯一标签分类指标，保留零召回与缺失类别。
# 输入：logits、合格集合、独立温度；输出：指标，不把模型分数当安全概率。
def decision_metrics(logits, accepted, temperature: float = 1.0) -> dict:
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("LAYA_TEMPERATURE_INVALID")
    set_cross_entropy(logits, accepted)
    p = torch.softmax(logits.float() / temperature, dim=1)
    pred = p.argmax(dim=1)
    success = accepted.gather(1, pred[:, None]).squeeze(1)
    unique = accepted.sum(dim=1) == 1
    result = {"rows": len(logits), "acceptable_accuracy": float(success.float().mean()),
              "unique_rows": int(unique.sum()), "ambiguous_rows": int((~unique).sum()),
              "temperature": temperature, "probabilities_are_safety_probabilities": False}
    if not bool(unique.any()):
        return {**result, "macro_f1": None, "nll": None, "brier": None, "ece": None}
    probabilities, predicted = p[unique], pred[unique]
    truth = accepted[unique].long().argmax(dim=1)
    per_class, scores = {}, []
    for index, action in enumerate(ACTIONS):
        tp = int(((predicted == index) & (truth == index)).sum())
        total = int((truth == index).sum())
        predicted_total = int((predicted == index).sum())
        f1 = 2 * tp / (total + predicted_total) if total + predicted_total else None
        per_class[action] = {"support": total, "recall": tp / total if total else None,
                             "f1": f1}
        if total:
            scores.append(f1)
    confidence = probabilities.max(dim=1).values
    correct = (predicted == truth).float()
    ece = 0.0
    for bin_index in range(15):
        mask = ((confidence >= bin_index / 15) &
                (confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1))
        if bool(mask.any()):
            ece += float(mask.float().mean() *
                         (confidence[mask].mean() - correct[mask].mean()).abs())
    onehot = torch.nn.functional.one_hot(truth, len(ACTIONS)).float()
    return {**result, "per_class": per_class,
            "macro_f1": sum(scores) / len(scores) if len(scores) == len(ACTIONS) else None,
            "nll": float(torch.nn.functional.cross_entropy(logits[unique] / temperature, truth)),
            "brier": float(((probabilities - onehot)**2).sum(dim=1).mean()),
            "ece": ece, "ece_bins": 15}


# 功能：导出 SDK 兼容的真实权重与校准，清除旧按选项数温度覆盖。
# 输入：已训练模型、新目录、训练回执；输出：可重新加载但尚无飞行资格的模型包。
def export_laya(agent, destination: Path, receipt: dict, temperature: float):
    from laya.agent import _fix_tokenizer_config
    from safetensors.torch import save_file

    if not math.isfinite(temperature) or not .5 <= temperature <= 5:
        raise ValueError("LAYA_EXPORT_TEMPERATURE_INVALID")
    destination.mkdir(parents=True, exist_ok=False)
    agent.tok.save_pretrained(destination / "tokenizer")
    # 固定 SDK 在首次加载时会规范化 tokenizer 配置；必须在冻结哈希之前完成。
    # 此处仅修改刚导出的新包，不改基础检查点或已安装的软件。
    _fix_tokenizer_config(str(destination))
    agent.model.encoder.config.save_pretrained(destination / "encoder")
    tensors = {k: v.detach().cpu().contiguous().clone()
               for k, v in agent.model.state_dict().items()}
    save_file(tensors, str(destination / "model.safetensors"))
    config = dict(agent.cfg)
    config["temperature"] = [temperature, 1.0, 1.0]
    config["temperature_by_options"] = {}
    config["training"] = receipt
    for name, data in (("rl_agent_config.json", config), ("uav-training-receipt.json", receipt)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    files = {p.relative_to(destination).as_posix(): file_digest(p)
             for p in destination.rglob("*") if p.is_file()}
    with (destination / "uav-package-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "dronedream.laya-package.v1", "files": files,
                   "flight_authority": False}, stream, indent=2)


# 功能：核对部署包全部内容；摘要只证明完整性，不代替来源信任或飞行验收。
# 输入：本地模型目录；输出：训练回执；遗漏/篡改文件一律拒绝。
def verify_package(destination: Path) -> dict:
    from .decision_dataset import evidence_path

    manifest = json.loads((destination / "uav-package-manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != "dronedream.laya-package.v1":
        raise ValueError("LAYA_PACKAGE_MANIFEST_INVALID")
    expected = manifest["files"]
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
              if p.is_file() and p.name != "uav-package-manifest.json"}
    required = {"model.safetensors", "rl_agent_config.json", "uav-training-receipt.json",
                "encoder/config.json", "tokenizer/tokenizer.json"}
    if not required <= set(expected) or set(expected) != actual:
        raise ValueError("LAYA_PACKAGE_FILES_MISMATCH")
    for name, digest in expected.items():
        if file_digest(evidence_path(destination, name)) != digest:
            raise ValueError("LAYA_PACKAGE_HASH_MISMATCH:" + name)
    return json.loads((destination / "uav-training-receipt.json").read_text("utf-8"))


# 功能：实际回载 SDK 权重并比较 logits/温度；不是只检查文件能打开。
# 输入：内存模型、已导出目录、至少一条相同输入；输出：数值一致性报告。
def verify_export(agent, destination: Path, items: list[dict]) -> dict:
    import torch

    receipt = verify_package(destination)
    expected = infer_logits(agent, items)
    restored = load_laya(destination, str(agent.device), agent.cfg["max_len"])
    actual = infer_logits(restored, items)
    verify_package(destination)
    config = json.loads((destination / "rl_agent_config.json").read_text("utf-8"))
    temperature = receipt["calibration"]["temperature"]
    # load_laya 专供 raw logits，因此单独验证 SDK 将使用的配置范围与覆盖项。
    if (config["temperature"] != [temperature, 1.0, 1.0]
            or config.get("temperature_by_options") or not .5 <= temperature <= 5):
        raise ValueError("LAYA_EXPORT_CALIBRATION_MISMATCH")
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise ValueError("LAYA_EXPORT_LOGIT_MISMATCH")
    return {"passed": True, "rows": len(items), "temperature": temperature,
            "max_logit_error": float((expected - actual).abs().max()),
            "smoke_only": receipt["smoke_only"], "flight_authority": False}
