"""Snapshot-bound plugin helpers shared by the live execution sidecars."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, copy_json

from .contracts import PreparedMission
from .extensions import ExtensionRegistry
from .hashing import sha256_json
from .plugin_api import build_discovered_extension_registry
from .plugin_contracts import PluginHookReceipt
from .plugin_files import check_plain_plugin_path
from .plugin_values import plugin_json_value

_RECEIPT_LOCK = threading.Lock()


# 功能：
#   要求非空门控集合中的每一项明确为 True，缺失证据不能借空集合真值通过。
# 输入：
#   gates：待汇总的门控映射。
# 输出：
#   accepted：是否所有实际提供的门控均为布尔真值。
def all_required_gates_passed(gates: object) -> bool:
    accepted = isinstance(gates, dict) and bool(gates) and all(v is True for v in gates.values())
    return accepted


# 功能：
#   从准备阶段冻结的插件快照重建运行注册表，不动态借用当前目录中的替代版本。
# 输入：
#   prepared：已冻结插件快照的任务。
# 输出：
#   registry：与该任务快照绑定的运行扩展注册表。
def runtime_extension_registry(prepared: PreparedMission) -> ExtensionRegistry:
    registry = build_discovered_extension_registry(prepared.plugin_snapshot)
    return registry


# 功能：
#   按冻结顺序应用角色提示词增强器，拒绝空白或非文本结果。
# 输入：
#   registry：当前任务的扩展注册表。
#   role：运行阶段模型角色。
#   instructions：核心提供的初始提示词。
# 输出：
#   result：增强后的提示词与实际钩子回执列表。
def augment_runtime_prompt(
    registry: ExtensionRegistry, *, role: str, instructions: str
) -> tuple[str, list[PluginHookReceipt]]:
    value, receipts = registry.invoke_pipeline(
        "models.prompt-packs",
        "augment_prompt",
        instructions,
        role=role,
    )
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("RUNTIME_PROMPT_PIPELINE_INVALID")
    result = (value, receipts)
    return result


# 功能：
#   1. 先检查 Python 原始模型值，防止非有限数被 JSON 序列化成 null 后绕过检查。
#   2. 应用冻结的结构化输出守卫，按预先冻结的摘要检查信封不可被改写。
# 输入：
#   registry：当前任务的扩展注册表。
#   role：预期模型角色。
#   expected_schema：预期输出结构名称。
#   artifact：模型端口生成的类型化制品。
#   record：绑定该制品的模型调用记录。
# 输出：
#   receipts：实际输出守卫的调用回执列表。
def validate_runtime_model_output(
    registry: ExtensionRegistry,
    *,
    role: str,
    expected_schema: str,
    artifact: object,
    record: object,
) -> list[PluginHookReceipt]:
    # 仅用转换结果检查有限性和规模；线上信封仍保留模型原有日期等 JSON 表示。
    plugin_json_value({"artifact": artifact, "record": record})
    artifact_value = artifact.model_dump(mode="json")  # type: ignore[attr-defined]
    record_value = record.model_dump(mode="json")  # type: ignore[attr-defined]
    envelope = copy_json({"artifact": artifact_value, "record": record_value})
    expected_digest = sha256_json(envelope)
    guarded, receipts = registry.invoke_pipeline(
        "models.structured-output-guards",
        "validate_output",
        envelope,
        role=role,
        expected_schema=expected_schema,
    )
    # 字典 == 会把 True 与 1 当作相等；先冻结摘要也能发现原地修改同一对象的情况。
    if sha256_json(copy_json(guarded)) != expected_digest:
        raise RuntimeError("PLUGIN_MODEL_OUTPUT_MUTATION_FORBIDDEN")
    return receipts


# 功能：
#   1. 写入前严格复核回执并检查整批编码预算，非法回执不能留下部分记录或空日志。
#   2. 在进程内锁下追加并 flush；这不是跨进程锁，也不承诺 fsync 级持久性。
# 输入：
#   path：本次运行的钩子回执追加日志。
#   receipts：需要保存的完整回执列表，空列表不创建文件。
# 输出：
#   None：不返回业务数据。
def append_hook_receipts(path: Path, receipts: list[PluginHookReceipt]) -> None:
    if not receipts:
        return
    validated = [
        PluginHookReceipt.model_validate(receipt.model_dump(mode="python"), strict=True)
        for receipt in receipts
    ]
    payloads = copy_json([receipt.model_dump(mode="json") for receipt in validated])
    rendered = "".join(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in payloads
    )
    if len(rendered.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("RUNTIME_HOOK_RECEIPT_BATCH_TOO_LARGE")
    with _RECEIPT_LOCK:
        check_plain_plugin_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()


# 功能：
#   1. 将独立插件结论映射为稳定且不覆盖同名项的门控，非对象输出明确否决。
#   2. 严格复制有限 JSON 证据，拒绝汇总通过却附带失败分项的矛盾结果。
#   3. 无插件结果时保持空输出，是否要求某能力存在由核心任务政策决定。
# 输入：
#   outputs：各插件实际返回的评估结果。
#   gate_prefix：当前阶段的门控命名前缀。
# 输出：
#   result：确定性门控映射与独立归一化评估列表组成的元组。
def require_plugin_acceptance(
    outputs: list[Any], *, gate_prefix: str
) -> tuple[dict[str, bool], list[dict[str, object]]]:
    gates: dict[str, bool] = {}
    normalized: list[dict[str, object]] = []
    for index, output in enumerate(outputs, start=1):
        if not isinstance(output, dict):
            gates[f"{gate_prefix}_{index:02d}"] = False
            normalized.append(
                {
                    "accepted": False,
                    "issue_codes": ["PLUGIN_VERDICT_NOT_OBJECT"],
                }
            )
            continue
        output = copy_json(output)
        accepted = output.get("accepted") is True
        if accepted and "gates" in output and not all_required_gates_passed(output["gates"]):
            raise ValueError("PLUGIN_VERDICT_GATES_INCONSISTENT")
        identity = str(
            output.get("detector")
            or output.get("validator")
            or output.get("evaluation")
            or f"{index:02d}"
        )
        safe_identity = "".join(
            character if character.isalnum() else "_" for character in identity.casefold()
        ).strip("_")
        suffix_value = safe_identity or f"{index:02d}"
        key = f"{gate_prefix}_{suffix_value}"
        suffix = 2
        original = key
        while key in gates:
            key = f"{original}_{suffix}"
            suffix += 1
        gates[key] = accepted
        normalized.append(output)
    result = (gates, normalized)
    return result
