"""Ordered, failure-aware dispatch for non-tool Harness plugin capabilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import MAX_JSON_DEPTH, MAX_JSON_NODES

from .hashing import canonical_json, sha256_json
from .plugin_contracts import (
    PluginActivationMode,
    PluginFailureMode,
    PluginHookReceipt,
    PluginSwapPolicy,
)


class ExtensionExecutionError(RuntimeError):
    """A fail-closed Harness extension rejected or failed an invocation."""

    # 功能：
    #   将失败关闭的钩子回执附到异常上，使上层能够保存失败身份而不继续派发。
    # 输入：
    #   receipt：实际失败钩子的回执。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, receipt: PluginHookReceipt) -> None:
        issue = receipt.issue_codes[0] if receipt.issue_codes else "PLUGIN_HOOK_FAILED"
        super().__init__(issue)
        self.receipt = receipt


@dataclass(frozen=True)
class ExtensionPlugin:
    """Bind hook callables to one installed capability and its ordering policy."""

    plugin_id: str
    version: str
    package_sha256: str
    capability_id: str
    slot_id: str
    activation_mode: PluginActivationMode
    failure_mode: PluginFailureMode
    swap_policy: PluginSwapPolicy
    pipeline_order: int
    runs_after: tuple[str, ...]
    runs_before: tuple[str, ...]
    hooks: dict[str, Callable[..., Any]]


# 功能：
#   1. 在深度与访问预算内构造有限数据回执，拒绝非字符串键并稳定排序集合。
#   2. 保持合法模型的既有 JSON 表示，不透明宿主服务只记录类型，不读取其状态。
# 输入：
#   value：钩子输入或输出中的一个值。
# 输出：
#   represented：回执使用的数据表示，不透明对象的类型标记不代表完整对象身份。
def _jsonable(value: Any) -> Any:
    remaining = MAX_JSON_NODES

    # 功能：
    #   转换当前数据节点，共享访问预算以限制循环输入、过深数据及模型重复展开。
    # 输入：
    #   item：当前节点，可能包含明确借用的宿主服务。
    #   depth：从根节点开始的嵌套深度。
    # 输出：
    #   converted：当前节点的有限、无键转换歧义的回执表示。
    def convert(item: Any, depth: int) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            raise ValueError("PLUGIN_HOOK_VALUE_TOO_COMPLEX")
        if isinstance(item, BaseModel):
            # 先检查 Python 值，避免 JSON 序列化把非法键或非有限数改写后掩盖问题。
            # 正式摘要仍沿用模型 JSON 模式的日期等表示，不能改变既有合法回执字节。
            # 两种表示是同一棵模型树：各自受剩余预算限制，但不能重复扣减输出节点额度。
            # 否则合法的约五万节点任务加上运行证据会被误算成超过十万节点。
            model_budget = remaining
            convert(item.model_dump(mode="python"), depth + 1)
            remaining = model_budget
            converted = convert(item.model_dump(mode="json"), depth + 1)
        elif isinstance(item, Path):
            converted = str(item)
        elif isinstance(item, dict):
            if len(item) > remaining:
                raise ValueError("PLUGIN_HOOK_VALUE_TOO_COMPLEX")
            if any(type(key) is not str for key in item):
                raise ValueError("PLUGIN_HOOK_VALUE_NON_STRING_KEY")
            converted = {key: convert(child, depth + 1) for key, child in item.items()}
        elif isinstance(item, (list, tuple, set, frozenset)):
            if len(item) > remaining:
                raise ValueError("PLUGIN_HOOK_VALUE_TOO_COMPLEX")
            children = [convert(child, depth + 1) for child in item]
            converted = (
                sorted(children, key=canonical_json)
                if isinstance(item, (set, frozenset))
                else children
            )
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("PLUGIN_HOOK_VALUE_NON_FINITE")
            converted = item
        elif isinstance(item, (str, int, bool)) or item is None:
            converted = item
        else:
            # 服务仍然借用；类型标记既不是完整身份，也不是服务副作用的安全证明。
            converted = {"python_type": type(item).__name__}
        return converted

    try:
        represented = convert(value, 0)
    except RecursionError as error:
        raise ValueError("PLUGIN_HOOK_VALUE_TOO_COMPLEX") from error
    return represented


# 功能：
#   深复制普通数据容器和模型，不透明宿主服务保留借用引用；数据隔离不限制服务自身的副作用。
# 输入：
#   value：准备交给钩子的业务值或服务对象。
# 输出：
#   detached：普通数据独立、服务引用保持不变的输入值。
def _detach_hook_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        detached = value.model_copy(deep=True)
    elif isinstance(value, dict):
        detached = {key: _detach_hook_data(item) for key, item in value.items()}
    elif isinstance(value, list):
        detached = [_detach_hook_data(item) for item in value]
    elif isinstance(value, tuple):
        detached = tuple(_detach_hook_data(item) for item in value)
    elif isinstance(value, (set, frozenset)):
        detached = type(value)(_detach_hook_data(item) for item in value)
    else:
        detached = value
    return detached


class ExtensionRegistry:
    """Resolve exclusive, fan-out, and ordered-pipeline Harness extension slots."""

    # 功能：
    #   创建不包含任何隐式默认选择的扩展目录，插件由上游明确注册。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self._plugins: dict[tuple[str, str], ExtensionPlugin] = {}

    # 功能：
    #   独立登记钩子映射和顺序声明，拒绝重复能力；插槽或拓扑验证失败时撤回本次登记。
    # 输入：
    #   plugin：准备登记的插件能力定义。
    # 输出：
    #   None：不返回业务数据。
    def register(self, plugin: ExtensionPlugin) -> None:
        key = (plugin.plugin_id, plugin.capability_id)
        if key in self._plugins:
            raise ValueError(
                f"duplicate extension plugin capability: {plugin.plugin_id}:{plugin.capability_id}"
            )
        self._plugins[key] = replace(
            plugin,
            hooks=dict(plugin.hooks),
            runs_after=tuple(plugin.runs_after),
            runs_before=tuple(plugin.runs_before),
        )
        try:
            self._validate_slot(plugin.slot_id)
        except Exception:
            self._plugins.pop(key)
            raise

    # 功能：
    #   导出按身份排序的路由元数据与钩子名称，查询时不执行任何钩子。
    # 输入：
    #   无。
    # 输出：
    #   catalog：供上游选择和展示的独立目录列表。
    def catalog(self) -> list[dict[str, object]]:
        catalog = [
            {
                "plugin_id": item.plugin_id,
                "version": item.version,
                "package_sha256": item.package_sha256,
                "capability_id": item.capability_id,
                "slot_id": item.slot_id,
                "activation_mode": item.activation_mode,
                "failure_mode": item.failure_mode,
                "swap_policy": item.swap_policy,
                "pipeline_order": item.pipeline_order,
                "hooks": sorted(item.hooks),
            }
            for item in sorted(
                self._plugins.values(),
                key=lambda value: (value.plugin_id, value.capability_id),
            )
        ]
        return catalog

    # 功能：
    #   筛选插槽和钩子，流水线按依赖拓扑排序；返回钩子映射副本，防止查询者替换内部处理器。
    # 输入：
    #   slot_id：目标插槽。
    #   hook：可选的钩子名筛选。
    # 输出：
    #   selected：按生效顺序排列的插件能力列表。
    def plugins_for_slot(self, slot_id: str, hook: str | None = None) -> list[ExtensionPlugin]:
        values = [
            replace(item, hooks=dict(item.hooks))
            for item in self._plugins.values()
            if item.slot_id == slot_id
        ]
        if hook is not None:
            values = [item for item in values if hook in item.hooks]
        if not values:
            selected = []
            return selected
        mode = values[0].activation_mode
        if mode == "pipeline":
            selected = self._ordered_pipeline(values)
            return selected
        selected = sorted(
            values,
            key=lambda item: (item.pipeline_order, item.plugin_id, item.capability_id),
        )
        return selected

    # 功能：
    #   拒绝同插槽混用激活方式、单选插槽多个实现以及流水线的顺序环。
    # 输入：
    #   slot_id：需要重新检查的插槽。
    # 输出：
    #   None：不返回业务数据。
    def _validate_slot(self, slot_id: str) -> None:
        values = [item for item in self._plugins.values() if item.slot_id == slot_id]
        modes = {item.activation_mode for item in values}
        if len(modes) > 1:
            raise ValueError(f"mixed activation modes in extension slot: {slot_id}")
        if values and values[0].activation_mode == "single" and len(values) > 1:
            raise ValueError(f"multiple implementations in single extension slot: {slot_id}")
        if values and values[0].activation_mode == "pipeline":
            self._ordered_pipeline(values)

    # 功能：
    #   按已选择能力构建拓扑顺序，用显式数值顺序与身份稳定打破并列，发现依赖环时拒绝。
    # 输入：
    #   values：同一流水线插槽中准备参与排序的能力。
    # 输出：
    #   ordered：满足前后依赖约束的稳定顺序。
    @staticmethod
    def _ordered_pipeline(values: list[ExtensionPlugin]) -> list[ExtensionPlugin]:
        # 功能：
        #   将插件和能力两级身份组合，区分同一包内的多个钩子能力。
        # 输入：
        #   item：当前能力定义。
        # 输出：
        #   instance_id：本次排序图的唯一节点标识。
        def key(item: ExtensionPlugin) -> str:
            instance_id = f"{item.plugin_id}#{item.capability_id}"
            return instance_id

        by_id = {key(item): item for item in values}
        plugin_keys: dict[str, list[str]] = {}
        for instance_id, item in by_id.items():
            plugin_keys.setdefault(item.plugin_id, []).append(instance_id)
            plugin_keys.setdefault(item.capability_id, []).append(instance_id)
        edges: dict[str, set[str]] = {instance_id: set() for instance_id in by_id}
        indegree: dict[str, int] = {instance_id: 0 for instance_id in by_id}
        for item in values:
            item_key = key(item)
            for predecessor in item.runs_after:
                for predecessor_key in plugin_keys.get(predecessor, []):
                    if item_key not in edges[predecessor_key]:
                        edges[predecessor_key].add(item_key)
                        indegree[item_key] += 1
            for successor in item.runs_before:
                for successor_key in plugin_keys.get(successor, []):
                    if successor_key not in edges[item_key]:
                        edges[item_key].add(successor_key)
                        indegree[successor_key] += 1
        ready = sorted(
            (by_id[plugin_id] for plugin_id, degree in indegree.items() if degree == 0),
            key=lambda item: (item.pipeline_order, item.plugin_id, item.capability_id),
        )
        ordered: list[ExtensionPlugin] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for successor in sorted(edges[key(current)]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(by_id[successor])
                    ready.sort(
                        key=lambda item: (
                            item.pipeline_order,
                            item.plugin_id,
                            item.capability_id,
                        )
                    )
        if len(ordered) != len(values):
            raise ValueError("plugin pipeline ordering cycle")
        return ordered

    # 功能：
    #   为一次钩子调用记录能力身份、结果状态和输入输出的数据表示摘要。
    # 输入：
    #   plugin：实际执行的能力定义。
    #   hook：实际调用的钩子名。
    #   outcome：接受或失败状态。
    #   input_payload：执行前保存的输入表示。
    #   output_payload：实际输出或失败时的空对象。
    #   issue_codes：可选失败标识列表。
    # 输出：
    #   receipt：本次钩子的摘要绑定回执。
    @staticmethod
    def _receipt(
        plugin: ExtensionPlugin,
        hook: str,
        *,
        outcome: str,
        input_payload: Any,
        output_payload: Any,
        issue_codes: list[str] | None = None,
    ) -> PluginHookReceipt:
        receipt = PluginHookReceipt(
            invocation_id=f"hook-{uuid4().hex[:24]}",
            plugin_id=plugin.plugin_id,
            plugin_version=plugin.version,
            plugin_package_sha256=plugin.package_sha256,
            capability_id=plugin.capability_id,
            slot_id=plugin.slot_id,
            hook=hook,
            outcome=outcome,  # type: ignore[arg-type]
            input_sha256=sha256_json(_jsonable(input_payload)),
            output_sha256=sha256_json(_jsonable(output_payload)),
            issue_codes=issue_codes or [],
            created_at=datetime.now(UTC),
        )
        return receipt

    # 功能：
    #   执行前冻结输入表示，向处理器传递独立数据；处理或输出回执失败都遵守声明的失败策略。
    # 输入：
    #   plugin：当前选定的插件能力。
    #   hook：实际钩子名。
    #   kwargs：本次业务输入与明确借用的宿主服务。
    # 输出：
    #   result：处理器输出与回执组成的二元组，隔离失败时输出为 None。
    def _invoke(
        self,
        plugin: ExtensionPlugin,
        hook: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, PluginHookReceipt]:
        handler = plugin.hooks[hook]
        input_payload = _jsonable(kwargs)
        try:
            output = handler(**_detach_hook_data(kwargs))
            # 输出无法形成有效回执同样属于执行失败，必须进入隔离或失败关闭分支。
            receipt = self._receipt(
                plugin,
                hook,
                outcome="accepted",
                input_payload=input_payload,
                output_payload=output,
            )
        except Exception as error:
            receipt = self._receipt(
                plugin,
                hook,
                outcome="failed",
                input_payload=input_payload,
                output_payload={},
                issue_codes=[f"PLUGIN_HOOK_FAILED:{type(error).__name__}"],
            )
            if plugin.failure_mode == "fail-closed":
                raise ExtensionExecutionError(receipt) from error
            result = (None, receipt)
            return result
        result = (output, receipt)
        return result

    # 功能：
    #   调用唯一匹配的钩子；必需能力缺失或匹配歧义时拒绝，不任意挑选实现。
    # 输入：
    #   slot_id：目标插槽。
    #   hook：钩子名。
    #   required：是否必须存在对应能力。
    #   kwargs：业务输入和宿主服务。
    # 输出：
    #   result：输出和回执列表；可选能力缺失时为 None 与空列表。
    def invoke_single(
        self,
        slot_id: str,
        hook: str,
        *,
        required: bool = False,
        **kwargs: Any,
    ) -> tuple[Any | None, list[PluginHookReceipt]]:
        plugins = self.plugins_for_slot(slot_id, hook)
        if not plugins:
            if required:
                raise KeyError(f"required extension slot missing: {slot_id}:{hook}")
            result = (None, [])
            return result
        if len(plugins) != 1:
            raise ValueError(f"extension single slot has {len(plugins)} handlers: {slot_id}")
        output, receipt = self._invoke(plugins[0], hook, kwargs)
        result = (output, [receipt])
        return result

    # 功能：
    #   逐个独立调用匹配能力，只收集接受的输出但保留隔离失败的回执，失败关闭异常向上抛出。
    # 输入：
    #   slot_id：多选插槽。
    #   hook：钩子名。
    #   kwargs：每个能力独立获得的业务输入。
    # 输出：
    #   result：成功输出列表与全部已返回回执列表的二元组。
    def invoke_multiple(
        self, slot_id: str, hook: str, **kwargs: Any
    ) -> tuple[list[Any], list[PluginHookReceipt]]:
        outputs: list[Any] = []
        receipts: list[PluginHookReceipt] = []
        for plugin in self.plugins_for_slot(slot_id, hook):
            output, receipt = self._invoke(plugin, hook, kwargs)
            receipts.append(receipt)
            if receipt.outcome == "accepted":
                outputs.append(output)
        result = (outputs, receipts)
        return result

    # 功能：
    #   按顺序传递接受且非 None 的结果，隔离失败保留上一阶段数据，失败关闭立即终止。
    # 输入：
    #   slot_id：流水线插槽。
    #   hook：钩子名。
    #   value：起始业务值。
    #   kwargs：每一步共同需要的附加业务输入或服务。
    # 输出：
    #   result：最终业务值与按执行顺序保存的回执列表。
    def invoke_pipeline(
        self,
        slot_id: str,
        hook: str,
        value: Any,
        **kwargs: Any,
    ) -> tuple[Any, list[PluginHookReceipt]]:
        current = value
        receipts: list[PluginHookReceipt] = []
        for plugin in self.plugins_for_slot(slot_id, hook):
            output, receipt = self._invoke(plugin, hook, {"value": current, **kwargs})
            receipts.append(receipt)
            if receipt.outcome == "accepted":
                if output is None:
                    invalid = self._receipt(
                        plugin,
                        hook,
                        outcome="failed",
                        input_payload={"value": current, **kwargs},
                        output_payload={},
                        issue_codes=["PLUGIN_PIPELINE_RETURNED_NONE"],
                    )
                    receipts[-1] = invalid
                    if plugin.failure_mode == "fail-closed":
                        raise ExtensionExecutionError(invalid)
                    continue
                current = output
        result = (current, receipts)
        return result
