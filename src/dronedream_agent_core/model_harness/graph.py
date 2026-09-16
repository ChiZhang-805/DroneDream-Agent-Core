"""Typed, bounded execution graph for composable DroneDream Harness workflows."""

from __future__ import annotations

import inspect
import math
import re
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronedream_plugin_sdk.protocol import copy_json

from ..hashing import sha256_json

HarnessNodeKind = Literal["core", "plugin", "barrier"]
HarnessNodeFailureMode = Literal["fail-closed", "isolate", "fallback"]
HarnessNodeState = Literal["accepted", "failed", "isolated", "fallback", "cached"]

_MISSION_STAGE_HANDLERS = {
    "mission.request-ingest": ("core.request-ingest", "core"),
    "mission.context-prepare": ("core.context-prepare", "plugin"),
    "mission.intent-parse": ("core.intent-parse", "core"),
    "mission.intent-consensus": ("core.intent-consensus", "barrier"),
    "mission.contract-freeze": ("core.contract-freeze", "barrier"),
    "mission.tool-advice": ("core.tool-advice", "plugin"),
    "mission.task-decompose": ("core.task-decompose", "core"),
    "mission.semantic-plan": ("core.semantic-plan", "core"),
    "mission.route-resolve": ("core.route-resolve", "core"),
    "mission.clearance-gate": ("core.clearance-gate", "barrier"),
    "mission.track-export": ("core.track-export", "core"),
    "mission.plan-evaluation": ("core.plan-evaluation", "plugin"),
    "mission.plan-review": ("core.plan-review", "core"),
    "mission.runtime-checkpoints": ("core.runtime-checkpoints", "core"),
    "mission.verification-plan": ("core.verification-plan", "barrier"),
    "mission.evidence-finalize": ("core.evidence-finalize", "barrier"),
}
_REQUIRED_MISSION_STAGES = frozenset(
    stage_id
    for stage_id in _MISSION_STAGE_HANDLERS
    if stage_id not in {"mission.tool-advice", "mission.plan-evaluation"}
)
_INTENT_REVIEW_PATTERN = re.compile(r"^mission\.intent-review-([1-9][0-9]*)$")
_MISSION_STAGE_EXECUTION_RANK = {
    "mission.request-ingest": (0, 0),
    "mission.context-prepare": (1, 0),
    "mission.intent-parse": (2, 0),
    "mission.intent-consensus": (4, 0),
    "mission.contract-freeze": (5, 0),
    "mission.tool-advice": (6, 0),
    "mission.task-decompose": (7, 0),
    "mission.semantic-plan": (8, 0),
    "mission.route-resolve": (9, 0),
    "mission.clearance-gate": (10, 0),
    "mission.track-export": (11, 0),
    "mission.plan-evaluation": (12, 0),
    "mission.plan-review": (13, 0),
    "mission.runtime-checkpoints": (14, 0),
    "mission.verification-plan": (15, 0),
    "mission.evidence-finalize": (16, 0),
}


# 功能：
#   为固定任务阶段生成执行排序键，独立意图审查共享阶段并按明确序号排序。
# 输入：
#   stage_id：已经识别的任务阶段标识。
# 输出：
#   rank：阶段序位与同阶段次序组成的排序键。
def _mission_stage_execution_rank(stage_id: str) -> tuple[int, int]:
    review_match = _INTENT_REVIEW_PATTERN.fullmatch(stage_id)
    if review_match:
        rank = (3, int(review_match.group(1)))
    else:
        rank = _MISSION_STAGE_EXECUTION_RANK[stage_id]
    return rank


class HarnessGraphModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True, allow_inf_nan=False
    )


class HarnessNodeSpec(HarnessGraphModel):
    node_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    node_kind: HarnessNodeKind = "plugin"
    handler_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,159}$")
    depends_on: list[str] = Field(default_factory=list, max_length=32)
    required_inputs: list[str] = Field(default_factory=list, max_length=64)
    output_key: str | None = Field(default=None, max_length=120)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    retry_limit: int = Field(default=0, ge=0, le=5)
    failure_mode: HarnessNodeFailureMode = "fail-closed"
    fallback_handler_id: str | None = Field(default=None, max_length=160)
    cacheable: bool = False
    authority: Literal["read", "plan", "simulate"] = "plan"

    # 功能：
    #   回退必须绑定处理器，核心及屏障节点必须失败关闭，不能被隔离后继续执行。
    # 输入：
    #   self：待校验的节点执行配置。
    # 输出：
    #   self：满足失败处理约束的节点配置。
    @model_validator(mode="after")
    def validate_fallback(self) -> HarnessNodeSpec:
        if self.failure_mode == "fallback" and not self.fallback_handler_id:
            raise ValueError("fallback nodes require fallback_handler_id")
        if self.node_kind in {"core", "barrier"} and self.failure_mode != "fail-closed":
            raise ValueError("core and barrier nodes must fail closed")
        return self


class HarnessTopology(HarnessGraphModel):
    schema_version: Literal["dronedream.harness-topology.v1"] = "dronedream.harness-topology.v1"
    topology_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{2,119}$")
    name: str = Field(min_length=1, max_length=120)
    nodes: list[HarnessNodeSpec] = Field(min_length=1, max_length=256)
    maximum_parallelism: int = Field(default=4, ge=1, le=16)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # 功能：
    #   在任何处理器启动前拒绝重复身份、未知依赖、自依赖、环以及同层输出键碰撞。
    # 输入：
    #   self：待校验的运行拓扑。
    # 输出：
    #   self：具备无环依赖且并行输出不会相互覆盖的拓扑。
    @model_validator(mode="after")
    def validate_graph(self) -> HarnessTopology:
        identifiers = [item.node_id for item in self.nodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate Harness node id")
        known = set(identifiers)
        for item in self.nodes:
            unknown = set(item.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Harness node {item.node_id} has unknown dependencies: {sorted(unknown)}"
                )
            if item.node_id in item.depends_on:
                raise ValueError(f"Harness node {item.node_id} depends on itself")
        for layer in _topological_layers(self.nodes):
            keys = [node.output_key for node in layer if node.output_key is not None]
            if len(keys) != len(set(keys)):
                raise ValueError("HARNESS_PARALLEL_OUTPUT_COLLISION")
        return self


# 功能：
#   重新验证运行图并核对必要任务阶段、处理器、节点类型和审查顺序，拒绝冒名实现。
# 输入：
#   topology：准备交给任务流水线的运行拓扑。
# 输出：
#   None：不返回业务数据。
def validate_protected_mission_topology(topology: HarnessTopology) -> None:
    topology = _topology_snapshot(topology)
    observed = {node.node_id for node in topology.nodes}
    missing = sorted(_REQUIRED_MISSION_STAGES - observed)
    if missing:
        raise ValueError("HARNESS_REQUIRED_STAGE_MISSING:" + ",".join(missing))
    review_indices: list[int] = []
    for node in topology.nodes:
        match = _INTENT_REVIEW_PATTERN.fullmatch(node.node_id)
        if match:
            review_indices.append(int(match.group(1)))
            expected = ("core.intent-review", "core")
        else:
            expected = _MISSION_STAGE_HANDLERS.get(node.node_id)
        if expected is None:
            raise ValueError(f"HARNESS_STAGE_IMPLEMENTATION_MISSING:{node.node_id}")
        expected_handler, expected_kind = expected
        if node.handler_id != expected_handler:
            raise ValueError(f"HARNESS_STAGE_HANDLER_MISMATCH:{node.node_id}")
        if node.node_kind != expected_kind:
            raise ValueError(f"HARNESS_STAGE_KIND_MISMATCH:{node.node_id}")
    if not review_indices:
        raise ValueError("HARNESS_INTENT_REVIEW_STAGE_MISSING")
    # 只按实际审查数量构造连续序列，不按外部提供的巨大序号分配内存。
    if sorted(review_indices) != list(range(1, len(review_indices) + 1)):
        raise ValueError("HARNESS_INTENT_REVIEW_SEQUENCE_INVALID")
    for node in topology.nodes:
        node_rank = _mission_stage_execution_rank(node.node_id)
        for dependency in node.depends_on:
            if _mission_stage_execution_rank(dependency) > node_rank:
                raise ValueError(
                    f"HARNESS_STAGE_EXECUTION_ORDER_MISMATCH:{node.node_id}:{dependency}"
                )


class HarnessBudget(HarnessGraphModel):
    maximum_nodes: int = Field(default=128, ge=1, le=512)
    maximum_elapsed_seconds: float = Field(default=600.0, gt=0.0, le=7200.0)
    maximum_retries: int = Field(default=32, ge=0, le=256)
    maximum_parallelism: int = Field(default=4, ge=1, le=16)


class HarnessRuntimePolicy(HarnessGraphModel):
    """Core-validated projection of selected Harness policy plugins."""

    topology: HarnessTopology
    budget: HarnessBudget
    maximum_model_calls: int = Field(ge=8, le=48)
    maximum_tool_calls: int = Field(ge=0, le=64)
    provider_attempts: int = Field(ge=1, le=5)
    model_timeout_seconds: float = Field(gt=0.0, le=600.0)
    tool_timeout_seconds: float = Field(gt=0.0, le=300.0)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


# 功能：
#   复制并重新验证完整拓扑，拒绝构造后的非法改写，防止运行对象借用可变配置。
# 输入：
#   topology：待冻结的运行拓扑模型。
# 输出：
#   snapshot：通过严格类型、标准 JSON、8 MiB 预算和依赖校验的独立拓扑。
def _topology_snapshot(topology: HarnessTopology) -> HarnessTopology:
    if not isinstance(topology, HarnessTopology):
        raise ValueError("HARNESS_TOPOLOGY_INVALID")
    value = copy_json(topology.model_dump(mode="python", warnings="error"), limit=8 * 1024 * 1024)
    snapshot = HarnessTopology.model_validate(value)
    return snapshot


# 功能：
#   检查预算原值的数字类型与有限性，拒绝布尔、文本和隐式整数截断。
# 输入：
#   value：未进行夹紧或转换的策略值。
#   integer：是否只允许整数。
# 输出：
#   value：通过类型检查的原始数字，后续由核心边界约束范围。
def _policy_number(value: object, *, integer: bool) -> int | float:
    allowed = (int,) if integer else (int, float)
    if type(value) not in allowed or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("HARNESS_POLICY_NUMBER_INVALID")
    return value


# 功能：
#   1. 严格核对六类策略及核心上限，再投影当前运行时实际支持的并行度、重试、超时和缓存字段。
#   2. 插件不能放宽核心预算；停止策略可收紧可选节点失败处理，隔离策略不能放宽原节点约束。
#   3. 未投影的插件描述字段仅进入策略身份摘要，不宣称已经实现对应运行机制。
# 输入：
#   topology：选择的运行拓扑。
#   policies：六类已选择插件提供的策略字典。
#   maximum_model_calls：核心允许的模型调用次数上限。
#   maximum_tool_calls：核心允许的工具调用次数上限。
#   model_timeout_seconds：核心允许的单次模型调用时限。
# 输出：
#   policy：独立的运行拓扑、核心约束后的预算及策略身份摘要。
def resolve_harness_runtime_policy(
    topology: HarnessTopology,
    policies: Mapping[str, Mapping[str, object]],
    *,
    maximum_model_calls: int,
    maximum_tool_calls: int,
    model_timeout_seconds: float,
) -> HarnessRuntimePolicy:
    topology = _topology_snapshot(topology)
    if any(node.node_id.startswith("mission.") for node in topology.nodes):
        validate_protected_mission_topology(topology)
    required = {"scheduler", "retry", "timeout", "budget", "fallback", "cache"}
    if not isinstance(policies, Mapping) or set(policies) != required:
        raise ValueError("HARNESS_POLICY_SET_INCOMPLETE")
    if any(not isinstance(value, Mapping) for value in policies.values()):
        raise ValueError("HARNESS_POLICY_SET_INVALID")
    policies = copy_json({key: dict(value) for key, value in policies.items()})
    numeric_fields = {
        "scheduler": {"maximum_parallelism": True},
        "retry": {"maximum_retries": True, "provider_attempts": True},
        "timeout": {
            "local_stage_seconds": False,
            "mission_seconds": False,
            "model_seconds": False,
            "tool_seconds": False,
        },
        "budget": {
            "maximum_nodes": True,
            "maximum_retries": True,
            "maximum_parallelism": True,
            "maximum_model_calls": True,
            "maximum_tool_calls": True,
        },
    }
    for group, fields in numeric_fields.items():
        for key, integer in fields.items():
            if key in policies[group]:
                _policy_number(policies[group][key], integer=integer)
    if type(policies["cache"].get("cache_read_only", False)) is not bool:
        raise ValueError("HARNESS_CACHE_POLICY_INVALID")
    if not 8 <= _policy_number(maximum_model_calls, integer=True) <= 48:
        raise ValueError("HARNESS_MODEL_CALL_CEILING_INVALID")
    if not 0 <= _policy_number(maximum_tool_calls, integer=True) <= 64:
        raise ValueError("HARNESS_TOOL_CALL_CEILING_INVALID")
    if not 0 < _policy_number(model_timeout_seconds, integer=False) <= 600:
        raise ValueError("HARNESS_MODEL_TIMEOUT_CEILING_INVALID")
    scheduler = policies["scheduler"]
    retry = policies["retry"]
    timeout = policies["timeout"]
    budget_value = policies["budget"]
    fallback = policies["fallback"]
    cache = policies["cache"]
    strategy = str(scheduler.get("strategy", ""))
    if strategy not in {"parallel-ready", "sequential"}:
        raise ValueError("HARNESS_SCHEDULER_POLICY_INVALID")
    parallelism = (
        1
        if strategy == "sequential"
        else int(scheduler.get("maximum_parallelism", topology.maximum_parallelism))
    )
    parallelism = max(1, min(16, topology.maximum_parallelism, parallelism))
    local_timeout = max(0.1, min(600.0, float(timeout.get("local_stage_seconds", 30.0))))
    retry_cap = max(0, min(256, int(retry.get("maximum_retries", 0))))
    cache_read_only = cache.get("cache_read_only") is True
    optional_failure = str(fallback.get("optional_failure", "stop"))
    if optional_failure not in {"stop", "isolate"}:
        raise ValueError("HARNESS_FALLBACK_POLICY_INVALID")
    nodes: list[HarnessNodeSpec] = []
    for node in topology.nodes:
        nodes.append(
            node.model_copy(
                update={
                    "timeout_seconds": min(node.timeout_seconds, local_timeout),
                    "retry_limit": min(node.retry_limit, retry_cap, 5),
                    "failure_mode": (
                        "fail-closed" if optional_failure == "stop" else node.failure_mode
                    ),
                    "cacheable": bool(
                        node.cacheable
                        and cache_read_only
                        and node.node_kind == "plugin"
                        and node.authority == "read"
                    ),
                }
            )
        )
    resolved_topology = topology.model_copy(
        update={
            "nodes": nodes,
            "maximum_parallelism": parallelism,
            "metadata": {
                **topology.metadata,
                "scheduler_strategy": strategy,
            },
        }
    )
    resolved_budget = HarnessBudget(
        maximum_nodes=max(1, min(512, int(budget_value.get("maximum_nodes", 128)))),
        maximum_elapsed_seconds=max(
            1.0,
            min(7200.0, float(timeout.get("mission_seconds", 600.0))),
        ),
        maximum_retries=min(
            retry_cap,
            max(0, min(256, int(budget_value.get("maximum_retries", retry_cap)))),
        ),
        maximum_parallelism=min(
            parallelism,
            max(1, min(16, int(budget_value.get("maximum_parallelism", parallelism)))),
        ),
    )
    if len(resolved_topology.nodes) > resolved_budget.maximum_nodes:
        raise ValueError("HARNESS_NODE_BUDGET_EXCEEDED")
    projected = {
        "topology": resolved_topology.model_dump(mode="json"),
        "budget": resolved_budget.model_dump(mode="json"),
        "maximum_model_calls": min(
            48,
            maximum_model_calls,
            max(8, int(budget_value.get("maximum_model_calls", maximum_model_calls))),
        ),
        "maximum_tool_calls": min(
            maximum_tool_calls,
            max(0, int(budget_value.get("maximum_tool_calls", maximum_tool_calls))),
        ),
        "provider_attempts": max(1, min(5, int(retry.get("provider_attempts", 1)))),
        "model_timeout_seconds": min(
            model_timeout_seconds,
            max(1.0, float(timeout.get("model_seconds", model_timeout_seconds))),
        ),
        "tool_timeout_seconds": max(
            0.1,
            min(300.0, float(timeout.get("tool_seconds", 60.0))),
        ),
    }
    policy = HarnessRuntimePolicy(
        **projected,
        policy_sha256=sha256_json({"selected": policies, "projected": projected}),
    )
    return policy


class HarnessNodeReceipt(HarnessGraphModel):
    invocation_id: str
    topology_id: str
    node_id: str
    handler_id: str
    state: HarnessNodeState
    attempt_count: int = Field(ge=1, le=6)
    elapsed_ms: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_codes: list[str] = Field(default_factory=list, max_length=32)
    created_at: datetime


class HarnessGraphResult(HarnessGraphModel):
    run_id: str
    topology_id: str
    outputs: dict[str, Any]
    receipts: list[HarnessNodeReceipt]
    elapsed_ms: int = Field(ge=0)
    created_at: datetime


class HarnessStageReceipt(HarnessGraphModel):
    topology_id: str
    node_id: str
    node_kind: HarnessNodeKind
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime


class HarnessGraphError(RuntimeError):
    """A bounded graph run could not safely complete."""

    # 功能：
    #   保存稳定失败码和已经产生的诊断回执副本，不借用调用者的可变列表。
    # 输入：
    #   self：待初始化的异常实例。
    #   code：运行图失败码。
    #   receipts：失败之前已经收集的可选回执。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, code: str, receipts: list[HarnessNodeReceipt] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.receipts = deepcopy(receipts) if receipts is not None else []


class HarnessStageRuntime:
    """Enforce a selected topology around protected mission-stage implementations."""

    # 功能：
    #   冻结经过重新验证的阶段拓扑；该账本负责顺序与回执，不替调用方执行处理器。
    # 输入：
    #   self：待初始化的阶段账本。
    #   topology：任务实际选择的阶段拓扑。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, topology: HarnessTopology) -> None:
        self._topology = _topology_snapshot(topology)
        if any(node.node_id.startswith("mission.") for node in self._topology.nodes):
            validate_protected_mission_topology(self._topology)
        self._nodes = {item.node_id: item for item in self._topology.nodes}
        self._completed: dict[str, HarnessStageReceipt] = {}

    # 功能：
    #   提供当前冻结拓扑的独立查看副本，不把内部依赖对象暴露给调用方修改。
    # 输入：
    #   self：当前阶段账本。
    # 输出：
    #   topology：用于查看的拓扑副本。
    @property
    def topology(self) -> HarnessTopology:
        topology = self._topology.model_copy(deep=True)
        return topology

    # 功能：
    #   查询可选阶段是否存在于本任务的冻结配置中，不从活动配置动态增删阶段。
    # 输入：
    #   self：当前阶段账本。
    #   node_id：待查询的阶段标识。
    # 输出：
    #   selected：阶段是否被本任务选择。
    def contains(self, node_id: str) -> bool:
        selected = node_id in self._nodes
        return selected

    # 功能：
    #   在必需输入和前序回执存在后记录一次阶段完成，拒绝重复完成及非有限摘要数据。
    #   回执只证明调用方提交的阶段数据，不自行证明飞行已完成。
    # 输入：
    #   self：按任务执行顺序调用的阶段账本。
    #   node_id：已执行的阶段标识。
    #   inputs：阶段实际输入映射。
    #   output：阶段实际输出。
    # 输出：
    #   receipt：不与内部账本共享可变字段的完成回执。
    def complete(self, node_id: str, *, inputs: Any, output: Any) -> HarnessStageReceipt:
        node = self._nodes.get(node_id)
        if node is None:
            raise HarnessGraphError(f"HARNESS_STAGE_NOT_IN_TOPOLOGY:{node_id}")
        if node_id in self._completed:
            raise HarnessGraphError(f"HARNESS_STAGE_ALREADY_COMPLETED:{node_id}")
        if not isinstance(inputs, Mapping):
            raise HarnessGraphError(f"HARNESS_STAGE_INPUT_INVALID:{node_id}")
        missing_inputs = [item for item in node.required_inputs if item not in inputs]
        if missing_inputs:
            raise HarnessGraphError(
                f"HARNESS_STAGE_REQUIRED_INPUT_MISSING:{node_id}:" + ",".join(missing_inputs)
            )
        missing = [item for item in node.depends_on if item not in self._completed]
        if missing:
            raise HarnessGraphError(
                f"HARNESS_STAGE_DEPENDENCY_INCOMPLETE:{node_id}:{','.join(missing)}"
            )
        receipt = HarnessStageReceipt(
            topology_id=self._topology.topology_id,
            node_id=node_id,
            node_kind=node.node_kind,
            input_sha256=sha256_json(inputs),
            output_sha256=sha256_json(output),
            completed_at=datetime.now(UTC),
        )
        self._completed[node_id] = receipt.model_copy(deep=True)
        return receipt

    # 功能：
    #   要求全部选中阶段完成，按稳定依赖顺序返回独立回执，不暴露内部账本。
    # 输入：
    #   self：当前阶段账本。
    # 输出：
    #   receipts：所有阶段的完成回执副本。
    def finish(self) -> list[HarnessStageReceipt]:
        missing = sorted(set(self._nodes) - set(self._completed))
        if missing:
            raise HarnessGraphError(f"HARNESS_TOPOLOGY_INCOMPLETE:{','.join(missing)}")
        receipts = [
            self._completed[item.node_id].model_copy(deep=True)
            for layer in _topological_layers(self._topology.nodes)
            for item in layer
        ]
        return receipts


HarnessHandler = Callable[[dict[str, Any]], Any]
HarnessObserver = Callable[[str, Mapping[str, Any]], None]


class _InvocationTimeout(TimeoutError):
    """等待到期而处理器可能仍在运行，不等同于处理器主动抛出的 TimeoutError。"""


# 功能：
#   丢弃超时之后才返回的结果；若结果是未等待协程则关闭，不执行它或产生完成回执。
# 输入：
#   future：已经结束或取消的处理器调用。
# 输出：
#   None：不返回业务数据。
def _discard_late_result(future: Future[Any]) -> None:
    if not future.cancelled() and future.exception() is None:
        value = future.result()
        if inspect.iscoroutine(value):
            value.close()


# 功能：
#   在独立线程中有界调用同步处理器，区分等待到期与处理器已结束的超时异常。
#   结果立即分离；未等待的协程关闭后拒绝，超时线程不被假装终止或与新调用重叠。
# 输入：
#   handler：已选择的同步处理器。
#   inputs：本次调用的输入快照。
#   timeout_seconds：节点与总预算共同允许的等待秒数。
# 输出：
#   result：与处理器返回容器分离的同步结果。
def _invoke_handler(handler: HarnessHandler, inputs: dict[str, Any], timeout_seconds: float) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    try:
        future = pool.submit(handler, deepcopy(inputs))
        try:
            value = future.result(timeout=timeout_seconds)
        except FutureTimeoutError as error:
            if future.done() and future.exception() is error:
                raise
            future.cancel()
            future.add_done_callback(_discard_late_result)
            raise _InvocationTimeout() from error
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise TypeError("HARNESS_SYNC_HANDLER_REQUIRED")
        result = deepcopy(value)
        return result
    finally:
        if future is not None:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)


@dataclass
class HarnessCircuitBreaker:
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    _failures: dict[str, int] = field(default_factory=dict)
    _opened_at: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # 功能：
    #   拒绝错误阈值和非有限冷却参数，零秒冷却作为明确的立即恢复设置保留。
    # 输入：
    #   self：尚未接收调用结果的连续失败计数器。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        if type(self.failure_threshold) is not int or self.failure_threshold < 1:
            raise ValueError("HARNESS_CIRCUIT_THRESHOLD_INVALID")
        if _policy_number(self.recovery_seconds, integer=False) < 0:
            raise ValueError("HARNESS_CIRCUIT_RECOVERY_INVALID")

    # 功能：
    #   在连续失败冷却期间拒绝新调用，冷却到期后重置失败计数并允许再次尝试。
    #   此计数器不限制恢复后的并发探测数量，不能替代调用并行度限制。
    # 输入：
    #   self：共享的处理器失败计数器。
    #   handler_id：被请求的处理器标识。
    # 输出：
    #   allowed：当前是否允许提交调用。
    def allow(self, handler_id: str) -> bool:
        with self._lock:
            opened = self._opened_at.get(handler_id)
            if opened is None:
                allowed = True
            elif time.monotonic() - opened >= self.recovery_seconds:
                self._opened_at.pop(handler_id, None)
                self._failures[handler_id] = 0
                allowed = True
            else:
                allowed = False
            return allowed

    # 功能：
    #   在结果被有效接收后清除该处理器的连续失败和冷却记录。
    # 输入：
    #   self：共享失败计数器。
    #   handler_id：成功返回的处理器标识。
    # 输出：
    #   None：不返回业务数据。
    def success(self, handler_id: str) -> None:
        with self._lock:
            self._failures[handler_id] = 0
            self._opened_at.pop(handler_id, None)

    # 功能：
    #   增加连续失败计数，达到阈值后记录单调冷却起点。
    # 输入：
    #   self：共享失败计数器。
    #   handler_id：失败的处理器标识。
    # 输出：
    #   None：不返回业务数据。
    def failure(self, handler_id: str) -> None:
        with self._lock:
            failures = self._failures.get(handler_id, 0) + 1
            self._failures[handler_id] = failures
            if failures >= self.failure_threshold:
                self._opened_at[handler_id] = time.monotonic()


# 功能：
#   按拓扑依赖生成稳定并行层，同一依赖重复声明只计一次；无法遍历全部节点即拒绝。
# 输入：
#   nodes：已经核对身份与依赖引用的节点列表。
# 输出：
#   layers：每层内部按标识排序、层间满足依赖关系的节点列表。
def _topological_layers(nodes: list[HarnessNodeSpec]) -> list[list[HarnessNodeSpec]]:
    by_id = {item.node_id: item for item in nodes}
    indegree = {item.node_id: len(set(item.depends_on)) for item in nodes}
    successors: dict[str, set[str]] = defaultdict(set)
    for item in nodes:
        for dependency in set(item.depends_on):
            successors[dependency].add(item.node_id)
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    layers: list[list[HarnessNodeSpec]] = []
    visited = 0
    while ready:
        current_ids = ready
        ready = []
        layer = [by_id[node_id] for node_id in current_ids]
        layers.append(layer)
        visited += len(layer)
        for node_id in current_ids:
            for successor in sorted(successors[node_id]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        ready.sort()
    if visited != len(nodes):
        raise ValueError("Harness topology contains a dependency cycle")
    return layers


@dataclass
class _RunBudget:
    """Shared per-run admission state, never reused across graphs or accounts."""

    deadline: float
    retries_remaining: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # 功能：
    #   在提交工作前原子检查取消及总时限，重试必须先消费本次运行的共享重试额度。
    # 输入：
    #   self：只属于本次运行的预算状态。
    #   retry：本次准入是否属于追加重试。
    # 输出：
    #   remaining：总时限内尚余的单调时间秒数。
    def admit(self, *, retry: bool = False) -> float:
        with self.lock:
            remaining = self.deadline - time.monotonic()
            if self.cancelled.is_set() or remaining <= 0:
                raise HarnessGraphError("HARNESS_ELAPSED_BUDGET_EXCEEDED")
            if retry:
                if self.retries_remaining <= 0:
                    raise HarnessGraphError("HARNESS_RETRY_BUDGET_EXCEEDED")
                self.retries_remaining -= 1
            return remaining


class HarnessGraphExecutor:
    """Execute typed Harness nodes with budgets, isolation, fallback, and receipts."""

    # 功能：
    #   建立处理器目录、观察器和只读结果缓存；核心飞行阶段仍由独立阶段账本约束。
    # 输入：
    #   self：待初始化的运行图执行器。
    #   handlers：主处理器标识到可调用对象的映射。
    #   fallback_handlers：可选回退处理器映射。
    #   budget：本地执行预算，未提供时使用核心默认值。
    #   observers：可信、非阻塞的观察回调列表。
    #   circuit_breaker：可选共享连续失败计数器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        handlers: Mapping[str, HarnessHandler],
        fallback_handlers: Mapping[str, HarnessHandler] | None = None,
        budget: HarnessBudget | None = None,
        observers: list[HarnessObserver] | None = None,
        circuit_breaker: HarnessCircuitBreaker | None = None,
    ) -> None:
        self.handlers = dict(handlers)
        self.fallback_handlers = dict(fallback_handlers or {})
        self.budget = (
            HarnessBudget.model_validate(budget.model_dump())
            if budget is not None
            else HarnessBudget()
        )
        self.observers = list(observers or [])
        self.circuit_breaker = circuit_breaker or HarnessCircuitBreaker()
        self._cache: OrderedDict[str, tuple[HarnessHandler, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()

    # 功能：
    #   1. 重新验证拓扑并冻结预算和输入，按依赖层并行执行节点，避免同层输出互相覆盖。
    #   2. 所有节点共享本次时限和重试额度；超时后撤销后续准入，不等待不可终止的外部线程。
    #   3. 汇总被接受的输出及诊断回执；这不是飞行成功或进程隔离证明。
    # 输入：
    #   self：当前运行图执行器。
    #   topology：本次选择的完整拓扑。
    #   inputs：处理器可见的初始数据映射。
    # 输出：
    #   result：本次运行输出、节点回执及总耗时。
    def run(self, topology: HarnessTopology, inputs: dict[str, Any]) -> HarnessGraphResult:
        topology = _topology_snapshot(topology)
        budget = HarnessBudget.model_validate(self.budget.model_dump())
        if not isinstance(inputs, dict):
            raise ValueError("HARNESS_INPUTS_INVALID")
        if len(topology.nodes) > budget.maximum_nodes:
            raise HarnessGraphError("HARNESS_NODE_BUDGET_EXCEEDED")
        started = time.monotonic()
        outputs = deepcopy(inputs)
        run_budget = _RunBudget(
            started + budget.maximum_elapsed_seconds,
            budget.maximum_retries,
        )
        receipts: list[HarnessNodeReceipt] = []
        retry_count = 0
        layers = _topological_layers(topology.nodes)
        parallelism = min(
            topology.maximum_parallelism,
            budget.maximum_parallelism,
        )
        for layer in layers:
            self._check_elapsed(started, receipts, budget.maximum_elapsed_seconds)
            pool = ThreadPoolExecutor(max_workers=min(parallelism, len(layer)))
            futures = {}
            try:
                futures = {
                    node.node_id: pool.submit(
                        self._run_node,
                        topology,
                        node,
                        deepcopy(outputs),
                        run_budget,
                    )
                    for node in layer
                }
                layer_results: dict[str, tuple[str | None, Any, HarnessNodeReceipt]] = {}
                for node in layer:
                    remaining = run_budget.admit()
                    attempts = node.retry_limit + 1 + int(node.failure_mode == "fallback")
                    node_wait_limit = node.timeout_seconds * attempts + 0.25
                    try:
                        result = futures[node.node_id].result(
                            timeout=min(remaining, node_wait_limit)
                        )
                    except FutureTimeoutError as error:
                        # 按实际限制等待的预算报告超时，避免平台时钟粒度造成错误归因。
                        if remaining <= node_wait_limit:
                            raise HarnessGraphError(
                                "HARNESS_ELAPSED_BUDGET_EXCEEDED", receipts
                            ) from error
                        self._check_elapsed(started, receipts, budget.maximum_elapsed_seconds)
                        raise HarnessGraphError(
                            f"HARNESS_NODE_TIMEOUT:{node.node_id}", receipts
                        ) from error
                    layer_results[node.node_id] = result
                run_budget.admit()
            except BaseException:
                # Python 不能杀死第三方运行线程；撤销准入阻止迟到结果提交及启动重试。
                run_budget.cancelled.set()
                raise
            finally:
                for future in futures.values():
                    future.cancel()
                # 非阻塞关闭只保证调用方可退出，不宣称第三方线程已经停止。
                pool.shutdown(wait=False, cancel_futures=True)
            for node in sorted(layer, key=lambda value: value.node_id):
                output_key, value, receipt = layer_results[node.node_id]
                receipts.append(receipt)
                retry_count += receipt.attempt_count - 1
                if retry_count > budget.maximum_retries:
                    raise HarnessGraphError("HARNESS_RETRY_BUDGET_EXCEEDED", receipts)
                if output_key is not None and receipt.state not in {"failed", "isolated"}:
                    outputs[output_key] = deepcopy(value)
                if receipt.state == "failed":
                    raise HarnessGraphError(
                        receipt.issue_codes[0] if receipt.issue_codes else "HARNESS_NODE_FAILED",
                        receipts,
                    )
        self._check_elapsed(started, receipts, budget.maximum_elapsed_seconds)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        result = HarnessGraphResult(
            run_id=f"harness-run-{uuid4().hex[:24]}",
            topology_id=topology.topology_id,
            outputs=outputs,
            receipts=receipts,
            elapsed_ms=elapsed_ms,
            created_at=datetime.now(UTC),
        )
        self._notify("graph.completed", result.model_dump(mode="json"))
        return result

    # 功能：
    #   检查输入、处理器和预算后调用节点；只有及时且可生成回执的只读结果可进入缓存。
    #   缓存绑定完整拓扑、输入与处理器对象，保留最多 128 项，不复用旧配置或替换前处理器的结果。
    # 输入：
    #   self：当前执行器。
    #   topology：本次冻结拓扑。
    #   node：要执行的节点。
    #   outputs：本层之前的独立数据快照。
    #   run_budget：本次运行共享的时限、取消和重试状态。
    # 输出：
    #   result：输出键、输出值和该节点的回执。
    def _run_node(
        self,
        topology: HarnessTopology,
        node: HarnessNodeSpec,
        outputs: dict[str, Any],
        run_budget: _RunBudget,
    ) -> tuple[str | None, Any, HarnessNodeReceipt]:
        run_budget.admit()
        missing = [key for key in node.required_inputs if key not in outputs]
        if missing:
            receipt = self._receipt(
                topology,
                node,
                state="failed",
                attempts=1,
                started=time.monotonic(),
                inputs=outputs,
                output={},
                issues=[f"HARNESS_REQUIRED_INPUT_MISSING:{','.join(missing)}"],
            )
            result = node.output_key, None, receipt
            return result
        handler = self.handlers.get(node.handler_id)
        if handler is None:
            receipt = self._receipt(
                topology,
                node,
                state="failed",
                attempts=1,
                started=time.monotonic(),
                inputs=outputs,
                output={},
                issues=[f"HARNESS_HANDLER_MISSING:{node.handler_id}"],
            )
            result = node.output_key, None, receipt
            return result
        if not self.circuit_breaker.allow(node.handler_id):
            receipt = self._receipt(
                topology,
                node,
                state="failed",
                attempts=1,
                started=time.monotonic(),
                inputs=outputs,
                output={},
                issues=[f"HARNESS_CIRCUIT_OPEN:{node.handler_id}"],
            )
            result = node.output_key, None, receipt
            return result
        cache_key = sha256_json(
            {
                "topology": topology.model_dump(mode="json"),
                "node_id": node.node_id,
                "handler_id": node.handler_id,
                "inputs": outputs,
            }
        )
        cacheable = node.cacheable and node.node_kind == "plugin" and node.authority == "read"
        if cacheable:
            with self._cache_lock:
                entry = self._cache.get(cache_key)
                if entry is not None and entry[0] is handler:
                    cached = deepcopy(entry[1])
                    receipt = self._receipt(
                        topology,
                        node,
                        state="cached",
                        attempts=1,
                        started=time.monotonic(),
                        inputs=outputs,
                        output=cached,
                    )
                    run_budget.admit()
                    self._cache.move_to_end(cache_key)
                    result = node.output_key, cached, receipt
                    return result
                if entry is not None:
                    del self._cache[cache_key]
        started = time.monotonic()
        self._notify(
            "node.started",
            {"topology_id": topology.topology_id, "node_id": node.node_id},
        )
        last_error: BaseException | None = None
        timed_out = False
        attempts = 0
        value: Any = None
        for attempt in range(1, node.retry_limit + 2):
            remaining = run_budget.admit(retry=attempt > 1)
            attempts = attempt
            try:
                value = _invoke_handler(handler, outputs, min(node.timeout_seconds, remaining))
                run_budget.admit()
                receipt = self._receipt(
                    topology,
                    node,
                    state="accepted",
                    attempts=attempts,
                    started=started,
                    inputs=outputs,
                    output=value,
                )
                # 先验证输出并形成回执，再发布缓存，避免无效结果毒化下一次调用。
                run_budget.admit()
                if cacheable:
                    with self._cache_lock:
                        self._cache[cache_key] = (handler, deepcopy(value))
                        self._cache.move_to_end(cache_key)
                        while len(self._cache) > 128:
                            self._cache.popitem(last=False)
                self.circuit_breaker.success(node.handler_id)
                self._notify("node.completed", receipt.model_dump(mode="json"))
                result = node.output_key, value, receipt
                return result
            except _InvocationTimeout as error:
                if remaining <= node.timeout_seconds:
                    raise HarnessGraphError("HARNESS_ELAPSED_BUDGET_EXCEEDED") from error
                last_error = FutureTimeoutError()
                timed_out = True
                # 已运行的处理器不能强制取消；不与其重叠启动重试或回退。
                break
            except HarnessGraphError:
                raise
            except Exception as error:  # plugin boundary deliberately catches third-party errors
                last_error = error
        self.circuit_breaker.failure(node.handler_id)
        issue = f"HARNESS_NODE_FAILED:{type(last_error).__name__}"
        if node.failure_mode == "fallback" and node.fallback_handler_id and not timed_out:
            fallback = self.fallback_handlers.get(node.fallback_handler_id)
            if fallback is not None:
                try:
                    remaining = run_budget.admit()
                    value = _invoke_handler(fallback, outputs, min(node.timeout_seconds, remaining))
                    run_budget.admit()
                    receipt = self._receipt(
                        topology,
                        node,
                        state="fallback",
                        attempts=attempts,
                        started=started,
                        inputs=outputs,
                        output=value,
                        issues=[issue],
                    )
                    self._notify("node.fallback", receipt.model_dump(mode="json"))
                    result = node.output_key, value, receipt
                    return result
                except HarnessGraphError:
                    raise
                except _InvocationTimeout:
                    issue = "HARNESS_FALLBACK_FAILED:TimeoutError"
                except Exception as error:  # noqa: PERF203 - explicit boundary evidence
                    issue = f"HARNESS_FALLBACK_FAILED:{type(error).__name__}"
        state: HarnessNodeState = "isolated" if node.failure_mode == "isolate" else "failed"
        receipt = self._receipt(
            topology,
            node,
            state=state,
            attempts=attempts,
            started=started,
            inputs=outputs,
            output={},
            issues=[issue],
        )
        self._notify("node.failed", receipt.model_dump(mode="json"))
        result = node.output_key, None, receipt
        return result

    # 功能：
    #   在接收结果前检查本次冻结时限，最后一层同样不能跳过总预算检查。
    # 输入：
    #   self：当前执行器。
    #   started：本次运行的单调起点。
    #   receipts：已经收集的回执。
    #   maximum_seconds：本次开始时冻结的总秒数上限。
    # 输出：
    #   None：不返回业务数据。
    def _check_elapsed(
        self, started: float, receipts: list[HarnessNodeReceipt], maximum_seconds: float
    ) -> None:
        if time.monotonic() - started >= maximum_seconds:
            raise HarnessGraphError("HARNESS_ELAPSED_BUDGET_EXCEEDED", receipts)

    # 功能：
    #   向可信非阻塞观察器分别提供独立事件副本，失败只隔离观察器，不改变执行权限。
    # 输入：
    #   self：当前执行器。
    #   event：事件名。
    #   payload：对应事件的结构化资料。
    # 输出：
    #   None：不返回业务数据。
    def _notify(self, event: str, payload: Mapping[str, Any]) -> None:
        for observer in self.observers:
            try:
                observer(event, deepcopy(payload))
            except Exception:
                continue

    # 功能：
    #   将实际输入输出摘要、尝试次数和耗时绑定到节点回执，不宣称完成真实飞行。
    # 输入：
    #   topology：本次冻结拓扑。
    #   node：当前节点。
    #   state：节点结果状态。
    #   attempts：主处理器实际尝试次数。
    #   started：节点开始的单调时刻。
    #   inputs：本次节点输入快照。
    #   output：本次节点输出快照。
    #   issues：可选的稳定诊断码列表。
    # 输出：
    #   receipt：绑定数据身份与运行耗时的节点回执。
    @staticmethod
    def _receipt(
        topology: HarnessTopology,
        node: HarnessNodeSpec,
        *,
        state: HarnessNodeState,
        attempts: int,
        started: float,
        inputs: Any,
        output: Any,
        issues: list[str] | None = None,
    ) -> HarnessNodeReceipt:
        receipt = HarnessNodeReceipt(
            invocation_id=f"harness-node-{uuid4().hex[:24]}",
            topology_id=topology.topology_id,
            node_id=node.node_id,
            handler_id=node.handler_id,
            state=state,
            attempt_count=attempts,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            input_sha256=sha256_json(inputs),
            output_sha256=sha256_json(output),
            issue_codes=issues or [],
            created_at=datetime.now(UTC),
        )
        return receipt
