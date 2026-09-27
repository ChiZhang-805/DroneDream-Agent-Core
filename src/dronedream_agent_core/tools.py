"""Typed plugin registry with explicit authority and hash-bound receipts."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

import jsonschema
from pydantic import BaseModel

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import ToolReceipt
from .extensions import ExtensionExecutionError, ExtensionRegistry
from .hashing import canonical_json, sha256_json
from .plugin_contracts import PluginHookReceipt

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
ToolAuthority = Literal["read", "plan", "simulate", "actuate"]


class ToolExecutionError(RuntimeError):
    """A plugin call failed with a hash-bound receipt safe to persist as evidence."""

    # 功能：
    #   将拒绝或失败的工具回执附到异常上，异常消息使用问题标识而不伪造成功输出。
    # 输入：
    #   receipt：本次工具调用的失败或拒绝回执。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, receipt: ToolReceipt) -> None:
        super().__init__(receipt.issue_codes[0] if receipt.issue_codes else "TOOL_CALL_FAILED")
        self.receipt = receipt


@dataclass(frozen=True)
class ToolPlugin(Generic[InputT, OutputT]):
    """Declare typed or schema-based I/O and the handler's required authority."""

    tool_id: str
    version: str
    authority: ToolAuthority
    input_type: type[InputT] | None
    output_type: type[OutputT] | None
    handler: Callable[[Any], Any]
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    plugin_id: str | None = None
    plugin_package_sha256: str | None = None
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    slot_id: str | None = None

    # 功能：
    #   要求工具同时声明输入和输出边界，外部纯 JSON 工具也不能省略任意一端的 Schema。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        if self.input_type is None and self.input_schema is None:
            raise ValueError("tool plugin requires input type or schema")
        if self.output_type is None and self.output_schema is None:
            raise ValueError("tool plugin requires output type or schema")

    # 功能：
    #   优先导出实际 Pydantic 输入契约，否则返回声明式 Schema 的独立副本。
    # 输入：
    #   无。
    # 输出：
    #   schema：调用方可读取且不共享原嵌套字典的输入约束。
    def resolved_input_schema(self) -> dict[str, Any]:
        if self.input_type is not None:
            schema = self.input_type.model_json_schema()
            return schema
        assert self.input_schema is not None
        schema = copy.deepcopy(self.input_schema)
        return schema

    # 功能：
    #   导出实际输出类型的 Schema，或隔离声明式输出约束，避免目录读者修改执行规则。
    # 输入：
    #   无。
    # 输出：
    #   schema：输出结果必须满足的结构约束。
    def resolved_output_schema(self) -> dict[str, Any]:
        if self.output_type is not None:
            schema = self.output_type.model_json_schema()
            return schema
        assert self.output_schema is not None
        schema = copy.deepcopy(self.output_schema)
        return schema


class ToolRegistry:
    """Enforce per-mission tool authority, guarded results and bounded attempts."""

    # 功能：
    #   建立本任务的权限集合、工具目录、调用预算和缓存，不在账户或任务间共享结果。
    # 输入：
    #   allowed_authorities：调用方明确允许的工具权限集合。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, allowed_authorities: set[ToolAuthority]) -> None:
        self._allowed = frozenset(allowed_authorities)
        self._plugins: dict[str, ToolPlugin] = {}
        self._extensions: ExtensionRegistry | None = None
        self._hook_receipt_sink: Callable[[list[PluginHookReceipt]], None] | None = None
        self._cache: dict[str, dict[str, object]] = {}
        self._cache_lock = threading.RLock()
        self._runtime_lock = threading.Lock()
        self._active_calls = 0
        self._runtime_total_call_count = 0
        self._runtime_budgeted_call_count = 0
        self._runtime_total_call_limit = 256
        self._runtime_budgeted_call_limit = 256
        self._runtime_budget_exempt_slots: frozenset[str] = frozenset()
        self._runtime_timeout_seconds = 60.0

    # 功能：
    #   1. 校验并重置任务内可选调用预算、独立总预算及单次处理器等待时限。
    #   2. 为必需插槽保留可选预算豁免，但不豁免总上限；新任务上下文清除原结果缓存。
    #   3. 调用尚未退出时拒绝重配，避免旧调用占用新预算或把旧结果写入新任务缓存。
    # 输入：
    #   maximum_calls：可选工具调用上限，实际限制至 256。
    #   timeout_seconds：处理器及其重试共享的等待时限，单位秒。
    #   maximum_total_calls：包括保留插槽在内的总调用上限，实际限制至 1024。
    #   budget_exempt_slot_ids：不计入可选预算但仍计入总预算的插槽集合。
    # 输出：
    #   None：不返回业务数据。
    def configure_runtime_limits(
        self,
        *,
        maximum_calls: int,
        timeout_seconds: float,
        maximum_total_calls: int = 256,
        budget_exempt_slot_ids: set[str] | frozenset[str] | None = None,
    ) -> None:
        if (
            type(maximum_calls) is not int
            or maximum_calls < 0
            or type(maximum_total_calls) is not int
            or maximum_total_calls < 1
            or type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 300
            or not math.isfinite(timeout_seconds)
        ):
            raise ValueError("TOOL_RUNTIME_LIMITS_INVALID")
        with self._runtime_lock:
            if self._active_calls:
                raise RuntimeError("TOOL_RUNTIME_CONFIGURATION_IN_USE")
            self._runtime_total_call_count = 0
            self._runtime_budgeted_call_count = 0
            self._runtime_total_call_limit = max(1, min(1024, maximum_total_calls))
            self._runtime_budgeted_call_limit = max(0, min(256, maximum_calls))
            self._runtime_budget_exempt_slots = frozenset(budget_exempt_slot_ids or ())
            self._runtime_timeout_seconds = timeout_seconds
            # 配置与缓存一起切换，新调用不能落在两者更新之间。
            with self._cache_lock:
                self._cache.clear()

    # 功能：
    #   原子占用一次执行尝试，先检查总预算，再检查非保留插槽的可选预算。
    # 输入：
    #   plugin：本次准备调用的已注册工具。
    # 输出：
    #   None：不返回业务数据。
    def _claim_runtime_call(self, plugin: ToolPlugin[Any, Any]) -> None:
        with self._runtime_lock:
            if self._runtime_total_call_count >= self._runtime_total_call_limit:
                raise RuntimeError("HARNESS_TOTAL_TOOL_CALL_BUDGET_EXCEEDED")
            budget_exempt = plugin.slot_id in self._runtime_budget_exempt_slots
            if (
                not budget_exempt
                and self._runtime_budgeted_call_count >= self._runtime_budgeted_call_limit
            ):
                raise RuntimeError("HARNESS_OPTIONAL_TOOL_CALL_BUDGET_EXCEEDED")
            self._runtime_total_call_count += 1
            if not budget_exempt:
                self._runtime_budgeted_call_count += 1

    # 功能：
    #   仅在没有活动调用时安装中间件与调度扩展，同时清除在原校验规则下接受的缓存。
    # 输入：
    #   registry：新的扩展注册表。
    #   receipt_sink：接收钩子执行回执的可选记录回调。
    # 输出：
    #   None：不返回业务数据。
    def configure_extensions(
        self,
        registry: ExtensionRegistry,
        *,
        receipt_sink: Callable[[list[PluginHookReceipt]], None] | None = None,
    ) -> None:
        with self._runtime_lock:
            if self._active_calls:
                raise RuntimeError("TOOL_RUNTIME_CONFIGURATION_IN_USE")
            self._extensions = registry
            self._hook_receipt_sink = receipt_sink
            with self._cache_lock:
                self._cache.clear()

    # 功能：
    #   执行有序工具中间件并上报回执，保留失败关闭的钩子回执供工具调用方拒绝结果。
    # 输入：
    #   hook：调用前或调用后的钩子名称。
    #   value：当前阶段的业务值。
    #   plugin：本次工具身份和权限信息。
    # 输出：
    #   result：转换结果与失败回执组成的二元组；正常阶段的失败回执为 None。
    def _middleware(
        self, hook: str, value: Any, *, plugin: ToolPlugin
    ) -> tuple[Any, PluginHookReceipt | None]:
        if self._extensions is None:
            result = (value, None)
            return result
        try:
            output, receipts = self._extensions.invoke_pipeline(
                "tools.middleware",
                hook,
                value,
                tool={
                    "tool_id": plugin.tool_id,
                    "plugin_id": plugin.plugin_id,
                    "authority": plugin.authority,
                    "slot_id": plugin.slot_id,
                },
            )
        except ExtensionExecutionError as error:
            if self._hook_receipt_sink is not None:
                self._hook_receipt_sink([error.receipt])
            result = (None, error.receipt)
            return result
        if self._hook_receipt_sink is not None and receipts:
            self._hook_receipt_sink(receipts)
        result = (output, None)
        return result

    # 功能：
    #   拒绝重复工具标识，登记处理器和类型，并独立保存可变 Schema 与路由元数据。
    # 输入：
    #   plugin：准备注册的工具定义。
    # 输出：
    #   None：不返回业务数据。
    def register(self, plugin: ToolPlugin) -> None:
        if plugin.tool_id in self._plugins:
            raise ValueError(f"duplicate tool id: {plugin.tool_id}")
        # frozen 数据类只能禁止属性赋值，不能阻止调用者修改其嵌套字典。
        self._plugins[plugin.tool_id] = replace(
            plugin,
            input_schema=copy.deepcopy(plugin.input_schema),
            output_schema=copy.deepcopy(plugin.output_schema),
            routing_metadata=copy.deepcopy(plugin.routing_metadata),
        )

    # 功能：
    #   解析工具调度扩展并记录其回执，无扩展时默认单次串行且不缓存；调度不能授予权限。
    # 输入：
    #   plugin：待调度工具的身份与权限元数据。
    # 输出：
    #   policy：调度扩展返回或宿主默认的策略对象。
    def _execution_policy(self, plugin: ToolPlugin) -> dict[str, object]:
        if self._extensions is None:
            policy = {"maximum_attempts": 1, "cache": False, "parallelism": 1}
            return policy
        try:
            output, receipts = self._extensions.invoke_single(
                "tools.execution-policy",
                "resolve_tool_execution",
                required=True,
                tool={
                    "tool_id": plugin.tool_id,
                    "plugin_id": plugin.plugin_id,
                    "authority": plugin.authority,
                    "slot_id": plugin.slot_id,
                },
            )
        except ExtensionExecutionError as error:
            if self._hook_receipt_sink is not None:
                self._hook_receipt_sink([error.receipt])
            raise RuntimeError("TOOL_EXECUTION_POLICY_FAILED") from error
        if self._hook_receipt_sink is not None and receipts:
            self._hook_receipt_sink(receipts)
        if not isinstance(output, dict):
            raise RuntimeError("TOOL_EXECUTION_POLICY_INVALID")
        policy = output
        return policy

    # 功能：
    #   从策略解析到结果收集固定整个批次的配置，异常退出也释放占用。
    # 输入：
    #   calls：按顺序排列的工具标识与输入二元组列表。
    # 输出：
    #   results：各调用的输出及回执列表；任一调用失败会抛出对应异常。
    def call_batch(
        self, calls: list[tuple[str, BaseModel | dict[str, object]]]
    ) -> list[tuple[BaseModel | dict[str, object], ToolReceipt]]:
        with self._runtime_lock:
            self._active_calls += 1
        try:
            results = self._call_batch_in_current_context(calls)
            return results
        finally:
            with self._runtime_lock:
                self._active_calls -= 1

    # 功能：
    #   在已占用的上下文内遵守所有工具的并行上限，最多八个线程，结果保持请求顺序。
    # 输入：
    #   calls：按顺序排列的工具标识与输入二元组列表。
    # 输出：
    #   results：各调用的输出及回执列表；任一调用失败会抛出对应异常。
    def _call_batch_in_current_context(
        self, calls: list[tuple[str, BaseModel | dict[str, object]]]
    ) -> list[tuple[BaseModel | dict[str, object], ToolReceipt]]:
        if not calls:
            results = []
            return results
        maximum = min(len(calls), 8)
        for tool_id, _value in calls:
            plugin = self._plugins.get(tool_id)
            if plugin is None:
                raise KeyError(f"unknown tool: {tool_id}")
            parallelism = self._execution_policy(plugin).get("parallelism", 1)
            if type(parallelism) is not int or parallelism < 1:
                raise RuntimeError("TOOL_EXECUTION_PARALLELISM_INVALID")
            # 任一成员要求串行，整个共享批次就不能用另一成员的宽松上限覆盖它。
            maximum = min(maximum, parallelism)
        with ThreadPoolExecutor(max_workers=maximum) as executor:
            # 每个线程独立复制请求上下文，保证进度仍归属于原账户与请求。
            futures = [executor.submit(copy_context().run, self.call, tool_id, value) for tool_id, value in calls]
            results = [future.result() for future in futures]
            return results

    # 功能：
    #   要求插槽恰有一个实现，多于一个或完全缺失时拒绝，不用列表首项消除歧义。
    # 输入：
    #   slot_id：单选插槽标识。
    # 输出：
    #   tool_id：该插槽唯一工具标识。
    def tool_for_slot(self, slot_id: str) -> str:
        matches = self.tool_ids_for_slot(slot_id)
        if len(matches) != 1:
            raise KeyError(f"plugin slot {slot_id} has {len(matches)} active implementations")
        tool_id = matches[0]
        return tool_id

    # 功能：
    #   仅查询并排序插槽内工具标识，不执行任何处理器。
    # 输入：
    #   slot_id：目标插槽标识。
    # 输出：
    #   tool_ids：排序后的工具标识列表。
    def tool_ids_for_slot(self, slot_id: str) -> list[str]:
        tool_ids = sorted(
            plugin.tool_id for plugin in self._plugins.values() if plugin.slot_id == slot_id
        )
        return tool_ids

    # 功能：
    #   解析插槽唯一实现后，沿正常工具入口执行权限、预算、输入输出与中间件检查。
    # 输入：
    #   slot_id：目标单选插槽。
    #   value：工具输入。
    # 输出：
    #   result：实际工具输出与回执的二元组。
    def call_slot(
        self, slot_id: str, value: BaseModel | dict[str, object]
    ) -> tuple[BaseModel | dict[str, object], ToolReceipt]:
        result = self.call(self.tool_for_slot(slot_id), value)
        return result

    # 功能：
    #   导出工具身份、权限和输入输出约束，隔离所有可变目录字段；查询目录不授予执行权。
    # 输入：
    #   无。
    # 输出：
    #   catalog：按工具标识排序的独立目录对象列表。
    def catalog(self) -> list[dict[str, object]]:
        catalog = [
            {
                "tool_id": plugin.tool_id,
                "version": plugin.version,
                "authority": plugin.authority,
                "plugin_id": plugin.plugin_id,
                "plugin_package_sha256": plugin.plugin_package_sha256,
                "routing_metadata": copy.deepcopy(plugin.routing_metadata),
                "slot_id": plugin.slot_id,
                "input_schema": plugin.resolved_input_schema(),
                "output_schema": plugin.resolved_output_schema(),
            }
            for plugin in sorted(self._plugins.values(), key=lambda value: value.tool_id)
        ]
        return catalog

    # 功能：
    #   1. 在整个逻辑调用期间固定运行配置，禁止并发重配使中间件、预算和缓存跨任务混用。
    #   2. 无论成功、异常或命中缓存都释放占用；退出不代表已强制终止超时的 Python 线程。
    # 输入：
    #   tool_id：已注册工具标识。
    #   value：类型化或字典形式的业务输入。
    # 输出：
    #   result：通过验证的工具输出和本次调用回执组成的二元组。
    def call(
        self, tool_id: str, value: BaseModel | dict[str, object]
    ) -> tuple[BaseModel | dict[str, object], ToolReceipt]:
        with self._runtime_lock:
            self._active_calls += 1
        try:
            result = self._call_in_current_context(tool_id, value)
            return result
        finally:
            with self._runtime_lock:
                self._active_calls -= 1

    # 功能：
    #   1. 在已占用的配置上下文中检查预算、权限和输入，执行共享等待期限内的受限重试。
    #   2. 在计算摘要和访问缓存前拒绝不合法的 JSON；输出与回执均验证后才缓存结果。
    #   3. 同步可信中间件不受强制超时限制；超时处理器可能仍在运行，因此不重叠重试。
    # 输入：
    #   tool_id：已注册工具标识。
    #   value：类型化或字典形式的业务输入。
    # 输出：
    #   result：通过验证的工具输出和本次调用回执组成的二元组。
    def _call_in_current_context(
        self, tool_id: str, value: BaseModel | dict[str, object]
    ) -> tuple[BaseModel | dict[str, object], ToolReceipt]:
        plugin = self._plugins.get(tool_id)
        if plugin is None:
            raise KeyError(f"unknown tool: {tool_id}")
        self._claim_runtime_call(plugin)
        raw_input = copy_json(
            value.model_dump(mode="json") if isinstance(value, BaseModel) else value,
            limit=256 * 1024,
        )
        policy = self._execution_policy(plugin)
        raw_input, middleware_failure = self._middleware(
            "before_tool_call", raw_input, plugin=plugin
        )
        if middleware_failure is not None:
            input_hash = middleware_failure.input_sha256
            raise ToolExecutionError(
                ToolReceipt(
                    call_id=f"tool-{uuid4().hex[:24]}",
                    tool_id=plugin.tool_id,
                    tool_version=plugin.version,
                    plugin_id=plugin.plugin_id,
                    plugin_package_sha256=plugin.plugin_package_sha256,
                    outcome="rejected",
                    input_sha256=input_hash,
                    output_sha256=sha256_json({}),
                    output={},
                    issue_codes=middleware_failure.issue_codes,
                )
            )
        # 历史摘要会把字典键转为字符串；先校验，防止 1 与 "1" 冒用同一缓存条目。
        raw_input = copy_json(raw_input, limit=256 * 1024)
        input_hash = sha256_json(raw_input)
        cache_key = sha256_json(
            {
                "tool_id": plugin.tool_id,
                "version": plugin.version,
                "input": raw_input,
                "plugin_package_sha256": plugin.plugin_package_sha256,
            }
        )
        if policy.get("cache") is True and plugin.authority in {"read", "plan"}:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                output_payload = copy.deepcopy(cached)
                output = (
                    plugin.output_type.model_validate(output_payload)
                    if plugin.output_type is not None
                    else output_payload
                )
                receipt = ToolReceipt(
                    call_id=f"tool-{uuid4().hex[:24]}",
                    tool_id=plugin.tool_id,
                    tool_version=plugin.version,
                    plugin_id=plugin.plugin_id,
                    plugin_package_sha256=plugin.plugin_package_sha256,
                    outcome="accepted",
                    input_sha256=input_hash,
                    output_sha256=sha256_json(output_payload),
                    output=output_payload,
                    issue_codes=["CACHE_HIT"],
                )
                result = (output, receipt)
                return result

        # 功能：
        #   生成绑定本次工具和实际输入摘要的失败异常，不把未知输出伪装成成功结果。
        # 输入：
        #   issue_code：拒绝或执行失败的问题标识。
        #   outcome：失败类型，默认为 failed。
        # 输出：
        #   error：携带可持久化失败回执的工具异常。
        def failed(issue_code: str, *, outcome: str = "failed") -> ToolExecutionError:
            error = ToolExecutionError(
                ToolReceipt(
                    call_id=f"tool-{uuid4().hex[:24]}",
                    tool_id=plugin.tool_id,
                    tool_version=plugin.version,
                    plugin_id=plugin.plugin_id,
                    plugin_package_sha256=plugin.plugin_package_sha256,
                    outcome=outcome,
                    input_sha256=input_hash,
                    output_sha256=sha256_json({}),
                    output={},
                    issue_codes=[issue_code],
                )
            )
            return error

        if plugin.authority not in self._allowed:
            raise failed("AUTHORITY_NOT_GRANTED", outcome="rejected")
        try:
            if len(canonical_json(raw_input).encode("utf-8")) > 256 * 1024:
                raise ValueError("TOOL_INPUT_TOO_LARGE")
            if plugin.input_type is not None:
                validated_input: BaseModel | dict[str, object] = plugin.input_type.model_validate(
                    raw_input
                )
            else:
                jsonschema.validate(raw_input, plugin.resolved_input_schema())
                if not isinstance(raw_input, dict):
                    raise ValueError("tool input schema must describe an object")
                validated_input = raw_input
        except Exception as error:
            raise failed(f"TOOL_INPUT_INVALID:{type(error).__name__}"[:96]) from error
        try:
            maximum_attempts = max(1, min(3, int(policy.get("maximum_attempts", 1))))
            # 丢失执行器确认不等于动作未发生，调度策略不能重复这个副作用。
            if plugin.authority not in {"read", "plan", "simulate"}:
                maximum_attempts = 1
            last_error: Exception | None = None
            raw_output: Any = None
            deadline = time.monotonic() + self._runtime_timeout_seconds
            for _attempt in range(maximum_attempts):
                timed_out_inflight = False
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("TOOL_ATTEMPT_DEADLINE_EXPIRED")
                    if _attempt:
                        self._claim_runtime_call(plugin)
                    executor = ThreadPoolExecutor(max_workers=1)
                    try:
                        # 输入复制或 submit 本身也会失败，线程池从创建起就必须受清理保护。
                        future = executor.submit(copy_context().run, plugin.handler, copy.deepcopy(validated_input))
                        try:
                            raw_output = future.result(timeout=remaining)
                        except TimeoutError:
                            timed_out_inflight = not future.done()
                            raise
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
                    if timed_out_inflight or time.monotonic() >= deadline:
                        break
            if last_error is not None:
                raise last_error
            if plugin.output_type is not None:
                output: BaseModel | dict[str, object] = plugin.output_type.model_validate(
                    raw_output
                )
                payload = output.model_dump(mode="json")
            else:
                payload = (
                    raw_output.model_dump(mode="json")
                    if isinstance(raw_output, BaseModel)
                    else raw_output
                )
                jsonschema.validate(payload, plugin.resolved_output_schema())
                if not isinstance(payload, dict):
                    raise ValueError("tool output schema must describe an object")
                output = payload
            payload = copy_json(payload, limit=1024 * 1024)
            if len(canonical_json(payload).encode("utf-8")) > 1024 * 1024:
                raise ValueError("TOOL_OUTPUT_TOO_LARGE")
            payload, middleware_failure = self._middleware(
                "after_tool_call", payload, plugin=plugin
            )
            if middleware_failure is not None:
                raise ValueError(middleware_failure.issue_codes[0])
            payload = copy_json(payload, limit=1024 * 1024)
            if plugin.output_type is not None:
                output = plugin.output_type.model_validate(payload)
                payload = output.model_dump(mode="json")
            else:
                jsonschema.validate(payload, plugin.resolved_output_schema())
                output = payload
            # 中间件可能改写结果，只有最终校验且未超预算的形式才能进入缓存。
            if len(canonical_json(payload).encode("utf-8")) > 1024 * 1024:
                raise ValueError("TOOL_OUTPUT_TOO_LARGE")
            receipt = ToolReceipt(
                call_id=f"tool-{uuid4().hex[:24]}",
                tool_id=plugin.tool_id,
                tool_version=plugin.version,
                plugin_id=plugin.plugin_id,
                plugin_package_sha256=plugin.plugin_package_sha256,
                outcome="accepted",
                input_sha256=input_hash,
                output_sha256=sha256_json(output),
                output=payload,
            )
            if policy.get("cache") is True and plugin.authority in {"read", "plan"}:
                with self._cache_lock:
                    self._cache[cache_key] = copy.deepcopy(payload)
        except Exception as error:
            raise failed(f"TOOL_EXECUTION_FAILED:{type(error).__name__}"[:96]) from error
        result = (output, receipt)
        return result
