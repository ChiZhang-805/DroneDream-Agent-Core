from __future__ import annotations

import time

import pytest

from dronedream_agent_core.model_harness.graph import (
    HarnessBudget,
    HarnessGraphError,
    HarnessGraphExecutor,
    HarnessNodeSpec,
    HarnessRuntimePolicy,
    HarnessStageRuntime,
    HarnessTopology,
    resolve_harness_runtime_policy,
    validate_protected_mission_topology,
)
from dronedream_agent_plugins.workflow_topologies import (
    official_topology_templates,
    plugin_definitions,
)


# 功能：
#   为测试节点创建通过依赖校验的独立拓扑，不执行节点处理器。
# 输入：
#   nodes：待组成拓扑的节点。
#   parallelism：拓扑允许的最大并行度。
# 输出：
#   topology：供执行器或阶段账本使用的测试拓扑。
def _topology(*nodes: HarnessNodeSpec, parallelism: int = 4) -> HarnessTopology:
    topology = HarnessTopology(
        topology_id="harness.test-topology",
        name="Test topology",
        nodes=list(nodes),
        maximum_parallelism=parallelism,
    )
    return topology


# 功能：
#   验证前序完成后才调度两个并行审查，实际输出和回执均符合依赖关系。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_executes_dependencies_and_parallel_nodes() -> None:
    calls: list[str] = []

    # 功能：
    #   记录首阶段调用并构造后续两条分支使用的测试意图。
    # 输入：
    #   values：包含 request 的节点数据快照。
    # 输出：
    #   intent：供后续审查拼接的测试文本。
    def first(values: dict[str, object]) -> str:
        calls.append("first")
        intent = f"{values['request']}:intent"
        return intent

    # 功能：
    #   记录左侧审查调用并消费首阶段产生的意图。
    # 输入：
    #   values：包含 intent 的节点数据快照。
    # 输出：
    #   review：左侧审查的测试结果。
    def left(values: dict[str, object]) -> str:
        calls.append("left")
        review = f"{values['intent']}:left"
        return review

    # 功能：
    #   记录右侧审查调用并消费首阶段产生的意图。
    # 输入：
    #   values：包含 intent 的节点数据快照。
    # 输出：
    #   review：右侧审查的测试结果。
    def right(values: dict[str, object]) -> str:
        calls.append("right")
        review = f"{values['intent']}:right"
        return review

    executor = HarnessGraphExecutor(handlers={"first": first, "left": left, "right": right})
    result = executor.run(
        _topology(
            HarnessNodeSpec(
                node_id="stage.intent",
                handler_id="first",
                required_inputs=["request"],
                output_key="intent",
            ),
            HarnessNodeSpec(
                node_id="stage.left-review",
                handler_id="left",
                depends_on=["stage.intent"],
                required_inputs=["intent"],
                output_key="left_review",
            ),
            HarnessNodeSpec(
                node_id="stage.right-review",
                handler_id="right",
                depends_on=["stage.intent"],
                required_inputs=["intent"],
                output_key="right_review",
            ),
        ),
        {"request": "mission"},
    )

    assert calls[0] == "first"
    assert set(calls[1:]) == {"left", "right"}
    assert result.outputs["left_review"] == "mission:intent:left"
    assert result.outputs["right_review"] == "mission:intent:right"
    assert [receipt.state for receipt in result.receipts] == [
        "accepted",
        "accepted",
        "accepted",
    ]


# 功能：
#   验证主调用按限额重试后回退，计划权限的回退结果不进入只读缓存，观察器收到事件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_retry_fallback_cache_and_observer() -> None:
    attempts = 0
    fallback_calls = 0
    events: list[str] = []

    # 功能：
    #   累加主调用次数并模拟一次已经结束的提供方故障。
    # 输入：
    #   _：本用例不读取的节点数据。
    # 输出：
    #   None：不返回业务数据。
    def broken(_: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("unavailable")

    # 功能：
    #   累加回退次数并生成与原始请求绑定的替代结果，不代表飞行安全证明。
    # 输入：
    #   values：包含原始 request 的独立快照。
    # 输出：
    #   result：本用例的回退数据。
    def fallback(values: dict[str, object]) -> dict[str, object]:
        nonlocal fallback_calls
        fallback_calls += 1
        result = {"safe": True, "request": values["request"]}
        return result

    executor = HarnessGraphExecutor(
        handlers={"primary": broken},
        fallback_handlers={"safe": fallback},
        observers=[lambda event, _payload: events.append(event)],
    )
    topology = _topology(
        HarnessNodeSpec(
            node_id="stage.resolver",
            handler_id="primary",
            fallback_handler_id="safe",
            failure_mode="fallback",
            required_inputs=["request"],
            output_key="result",
            retry_limit=2,
            cacheable=True,
        )
    )

    first = executor.run(topology, {"request": "deliver"})
    second = executor.run(topology, {"request": "deliver"})

    assert attempts == 6
    assert fallback_calls == 2
    assert first.outputs["result"] == {"safe": True, "request": "deliver"}
    assert second.outputs["result"] == first.outputs["result"]
    assert first.receipts[0].state == "fallback"
    assert "node.fallback" in events


# 功能：
#   验证明确可隔离的建议节点失败可形成隔离回执，核心节点同样失败必须中止。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_isolates_advisory_node_but_fails_closed_core() -> None:
    # 功能：
    #   对两种节点权限注入相同故障，以比较失败处理边界。
    # 输入：
    #   _：本用例不读取的节点数据。
    # 输出：
    #   None：不返回业务数据。
    def broken(_: dict[str, object]) -> object:
        raise RuntimeError("broken")

    isolated = HarnessGraphExecutor(handlers={"broken": broken}).run(
        _topology(
            HarnessNodeSpec(
                node_id="stage.advisory",
                handler_id="broken",
                failure_mode="isolate",
            )
        ),
        {"request": "mission"},
    )
    assert isolated.receipts[0].state == "isolated"

    with pytest.raises(HarnessGraphError, match="HARNESS_NODE_FAILED"):
        HarnessGraphExecutor(handlers={"broken": broken}).run(
            _topology(
                HarnessNodeSpec(
                    node_id="stage.core-gate",
                    handler_id="broken",
                    node_kind="core",
                )
            ),
            {"request": "mission"},
        )


# 功能：
#   验证依赖环、必需输入缺失和节点数量超预算均不能形成成功运行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_rejects_cycle_missing_input_and_budget_overrun() -> None:
    with pytest.raises(ValueError, match="cycle"):
        _topology(
            HarnessNodeSpec(node_id="stage.one", handler_id="one", depends_on=["stage.two"]),
            HarnessNodeSpec(node_id="stage.two", handler_id="two", depends_on=["stage.one"]),
        )

    with pytest.raises(HarnessGraphError, match="HARNESS_REQUIRED_INPUT_MISSING"):
        HarnessGraphExecutor(handlers={"one": lambda values: values}).run(
            _topology(
                HarnessNodeSpec(
                    node_id="stage.one",
                    handler_id="one",
                    required_inputs=["required"],
                )
            ),
            {},
        )

    with pytest.raises(HarnessGraphError, match="HARNESS_NODE_BUDGET_EXCEEDED"):
        HarnessGraphExecutor(
            handlers={"one": lambda values: values},
            budget=HarnessBudget(maximum_nodes=1),
        ).run(
            _topology(
                HarnessNodeSpec(node_id="stage.one", handler_id="one"),
                HarnessNodeSpec(node_id="stage.two", handler_id="one"),
            ),
            {},
        )


# 功能：
#   节点处理器超过自身等待时限后返回失败，不能接纳迟到结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_enforces_node_timeout() -> None:
    # 功能：
    #   用受控短暂阻塞模拟超过节点时限的外部调用。
    # 输入：
    #   _：未使用的节点数据。
    # 输出：
    #   result：时限之后才会产生的空测试结果。
    def slow(_: dict[str, object]) -> object:
        time.sleep(0.04)
        result = {}
        return result

    with pytest.raises(HarnessGraphError, match="HARNESS_NODE_FAILED:TimeoutError"):
        HarnessGraphExecutor(handlers={"slow": slow}).run(
            _topology(
                HarnessNodeSpec(
                    node_id="stage.slow",
                    handler_id="slow",
                    timeout_seconds=0.01,
                )
            ),
            {},
        )


# 功能：
#   共享重试额度为零时只允许首次尝试，额外调用必须在提交前被阻止。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_retry_budget_is_enforced_before_extra_handler_calls() -> None:
    calls = []

    # 功能：
    #   记录实际进入处理器的次数并返回失败，便于核对额度是否提前执行。
    # 输入：
    #   values：本用例不读取的输入快照。
    # 输出：
    #   None：不返回业务数据。
    def broken(values):
        calls.append(1)
        raise RuntimeError("offline")

    executor = HarnessGraphExecutor(
        handlers={"broken": broken}, budget=HarnessBudget(maximum_retries=0)
    )
    with pytest.raises(HarnessGraphError, match="RETRY_BUDGET_EXCEEDED"):
        executor.run(
            _topology(HarnessNodeSpec(node_id="stage.test", handler_id="broken", retry_limit=3)), {}
        )
    assert len(calls) == 1


# 功能：
#   即使最后一层自身没有很短时限，总任务到期后仍不能接纳其结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_graph_checks_total_deadline_before_accepting_final_layer() -> None:
    # 功能：
    #   模拟超过整个图时限但未超过节点独立时限的调用。
    # 输入：
    #   values：未使用的输入快照。
    # 输出：
    #   result：总时限之后才产生的测试文本。
    def slow(values):
        time.sleep(0.04)
        result = "late"
        return result

    executor = HarnessGraphExecutor(
        handlers={"slow": slow}, budget=HarnessBudget(maximum_elapsed_seconds=0.01)
    )
    with pytest.raises(HarnessGraphError, match="ELAPSED_BUDGET_EXCEEDED"):
        executor.run(_topology(HarnessNodeSpec(node_id="stage.test", handler_id="slow")), {})


# 功能：
#   用粗粒度测试时钟复现等待与读钟边界，超时仍归因于真正限制等待的总预算。
# 输入：
#   monkeypatch：替换图内部计时器的夹具。
# 输出：
#   None：不返回业务数据。
def test_graph_timeout_keeps_budget_owner_despite_clock_resolution(monkeypatch) -> None:
    from types import SimpleNamespace

    from dronedream_agent_core.model_harness import graph

    # 等待先到期而注入时钟还未前进，也不能归因于较大的节点时限。
    monkeypatch.setattr(graph, "time", SimpleNamespace(monotonic=lambda: 5.0))
    executor = HarnessGraphExecutor(
        handlers={"slow": lambda _: time.sleep(0.04)},
        budget=HarnessBudget(maximum_elapsed_seconds=0.01),
    )
    with pytest.raises(HarnessGraphError, match="ELAPSED_BUDGET_EXCEEDED"):
        executor.run(_topology(HarnessNodeSpec(node_id="stage.test", handler_id="slow")), {})


# 功能：
#   处理器修改输入、调用者修改返回结果均不污染来源或下一次命中的只读缓存。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_handler_cannot_mutate_nested_input_or_poison_cached_output() -> None:
    # 功能：
    #   故意改写收到的嵌套输入，并返回可变结果以检验两端隔离。
    # 输入：
    #   values：执行器提供的独立数据快照。
    # 输出：
    #   result：包含可变列表的测试结果。
    def handler(values):
        values["request"]["private"] = "mutated"
        result = {"items": ["original"]}
        return result

    executor = HarnessGraphExecutor(handlers={"reader": handler})
    topology = _topology(
        HarnessNodeSpec(
            node_id="stage.test",
            handler_id="reader",
            cacheable=True,
            authority="read",
            output_key="result",
        )
    )
    inputs = {"request": {"private": "unchanged"}}
    first = executor.run(topology, inputs)
    assert inputs["request"]["private"] == "unchanged"
    first.outputs["result"]["items"].append("poison")
    second = executor.run(topology, inputs)
    assert second.receipts[0].state == "cached"
    assert second.outputs["result"] == {"items": ["original"]}


# 功能：
#   验证调用方不因线程池隐式等待而一直等到超时处理器真正返回。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_harness_graph_timeout_returns_without_waiting_for_stuck_handler() -> None:
    # 功能：
    #   在较长但有限的阻塞后返回，用于观察调用方是否被强制等待。
    # 输入：
    #   _：未使用的输入快照。
    # 输出：
    #   result：迟到的空测试结果。
    def slow(_: dict[str, object]) -> object:
        time.sleep(0.2)
        result = {}
        return result

    started = time.monotonic()
    with pytest.raises(HarnessGraphError, match="HARNESS_NODE_FAILED:TimeoutError"):
        HarnessGraphExecutor(handlers={"slow": slow}).run(
            _topology(
                HarnessNodeSpec(
                    node_id="stage.stuck",
                    handler_id="slow",
                    timeout_seconds=0.01,
                )
            ),
            {},
        )

    assert time.monotonic() - started < 0.1


# 功能：
#   验证策略的并行、预算和超时值受核心上限约束，核心节点不能取得只读缓存捷径。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plugin_policies_are_projected_into_runtime_without_relaxing_core_limits() -> None:
    topology = _topology(
        HarnessNodeSpec(
            node_id="stage.core-gate",
            handler_id="core",
            node_kind="core",
            retry_limit=3,
            cacheable=True,
            authority="read",
        ),
        HarnessNodeSpec(
            node_id="stage.optional-read",
            handler_id="plugin",
            node_kind="plugin",
            retry_limit=3,
            cacheable=True,
            authority="read",
        ),
        parallelism=8,
    )
    selected = {
        "scheduler": {"strategy": "sequential", "maximum_parallelism": 16},
        "retry": {"maximum_retries": 2, "provider_attempts": 4},
        "timeout": {
            "model_seconds": 999,
            "tool_seconds": 12,
            "local_stage_seconds": 9,
            "mission_seconds": 120,
        },
        "budget": {
            "maximum_model_calls": 100,
            "maximum_tool_calls": 30,
            "maximum_nodes": 20,
            "maximum_retries": 2,
            "maximum_parallelism": 8,
        },
        "fallback": {"optional_failure": "isolate"},
        "cache": {"cache_read_only": True},
    }

    policy = resolve_harness_runtime_policy(
        topology,
        selected,
        maximum_model_calls=48,
        maximum_tool_calls=16,
        model_timeout_seconds=180,
    )

    assert isinstance(policy, HarnessRuntimePolicy)
    assert policy.topology.maximum_parallelism == 1
    assert policy.maximum_model_calls == 48
    assert policy.maximum_tool_calls == 16
    assert policy.model_timeout_seconds == 180
    assert policy.tool_timeout_seconds == 12
    assert policy.topology.nodes[0].failure_mode == "fail-closed"
    assert policy.topology.nodes[0].cacheable is False
    assert policy.topology.nodes[1].cacheable is True
    assert all(node.retry_limit == 2 for node in policy.topology.nodes)


# 功能：
#   逐个检查官方拓扑声明的阶段都绑定已有任务实现，不仅验证节点名称存在。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_first_party_mission_topologies_bind_every_declared_stage_to_real_implementation() -> None:
    for plugin in plugin_definitions():
        resolved = plugin.hooks["resolve_topology"]()
        validate_protected_mission_topology(HarnessTopology.model_validate(resolved))


# 功能：
#   验证不存在的阶段及被改绑的净空处理器都无法冒充受保护任务流水线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mission_topology_cannot_claim_an_unimplemented_or_rebound_stage() -> None:
    topology = HarnessTopology.model_validate(plugin_definitions()[0].hooks["resolve_topology"]())
    unknown = topology.model_copy(
        update={
            "nodes": [
                *topology.nodes,
                HarnessNodeSpec(
                    node_id="mission.magic-stage",
                    node_kind="plugin",
                    handler_id="plugin.magic-stage",
                ),
            ]
        }
    )
    with pytest.raises(ValueError, match="HARNESS_STAGE_IMPLEMENTATION_MISSING"):
        validate_protected_mission_topology(unknown)

    rebound = topology.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"handler_id": "plugin.fake-clearance"})
                if node.node_id == "mission.clearance-gate"
                else node
                for node in topology.nodes
            ]
        }
    )
    with pytest.raises(ValueError, match="HARNESS_STAGE_HANDLER_MISMATCH"):
        validate_protected_mission_topology(rebound)


# 功能：
#   拒绝让较早审查阶段依赖实际执行顺序中较晚阶段，防止声明与任务调用顺序不一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_protected_topology_rejects_dependency_after_runtime_execution_order() -> None:
    topology = official_topology_templates()["topology.committee-closed-loop"]
    out_of_order = topology.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"depends_on": ["mission.intent-review-3"]})
                if node.node_id == "mission.intent-review-1"
                else node
                for node in topology.nodes
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "HARNESS_STAGE_EXECUTION_ORDER_MISMATCH:mission.intent-review-1:mission.intent-review-3"
        ),
    ):
        validate_protected_mission_topology(out_of_order)


# 功能：
#   巨大审查序号必须被有界拒绝，不能按该序号大小创建检查数组。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_protected_review_index_validation_does_not_allocate_an_unbounded_range() -> None:
    topology = official_topology_templates()["topology.committee-closed-loop"].model_copy(deep=True)
    topology.nodes.append(
        HarnessNodeSpec(
            node_id="mission.intent-review-" + "9" * 30,
            node_kind="core",
            handler_id="core.intent-review",
        )
    )
    with pytest.raises(ValueError, match="REVIEW_SEQUENCE_INVALID"):
        validate_protected_mission_topology(topology)


# 功能：
#   阶段运行器创建后修改原拓扑，不能删掉运行器已经冻结的前序要求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stage_runtime_owns_a_snapshot_of_required_dependencies() -> None:
    topology = _topology(
        HarnessNodeSpec(node_id="stage.first", handler_id="first"),
        HarnessNodeSpec(node_id="stage.second", handler_id="second", depends_on=["stage.first"]),
    )
    runtime = HarnessStageRuntime(topology)
    topology.nodes[1].depends_on.clear()
    exposed = runtime.topology
    exposed.nodes[1].depends_on.clear()
    with pytest.raises(HarnessGraphError, match="DEPENDENCY_INCOMPLETE"):
        runtime.complete("stage.second", inputs={}, output={})


# 功能：
#   等待超时但主处理器尚未结束时，既不启动重试，也不重叠启动回退。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_timed_out_handler_is_not_overlapped_with_retry_or_fallback() -> None:
    calls = []

    # 功能：
    #   记录主处理器启动并持续到节点时限之后。
    # 输入：
    #   values：未使用的输入快照。
    # 输出：
    #   None：不返回业务数据。
    def slow(values):
        calls.append("primary")
        time.sleep(0.04)

    # 功能：
    #   记录本不应该在主处理器仍运行时启动的回退调用。
    # 输入：
    #   values：未使用的输入快照。
    # 输出：
    #   None：不返回业务数据。
    def fallback(values):
        calls.append("fallback")

    executor = HarnessGraphExecutor(handlers={"slow": slow}, fallback_handlers={"safe": fallback})
    with pytest.raises(HarnessGraphError, match="HARNESS_NODE_FAILED:TimeoutError"):
        executor.run(
            _topology(
                HarnessNodeSpec(
                    node_id="stage.test",
                    handler_id="slow",
                    retry_limit=3,
                    timeout_seconds=0.01,
                    failure_mode="fallback",
                    fallback_handler_id="safe",
                )
            ),
            {},
        )
    assert calls == ["primary"]
