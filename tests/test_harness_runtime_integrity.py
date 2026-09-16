"""运行图预算、缓存身份及阶段回执的隔离边界。"""

import inspect
import threading
from unittest.mock import Mock

import pytest

from dronedream_agent_core.model_harness import graph as graph_module
from dronedream_agent_core.model_harness.graph import (
    HarnessCircuitBreaker,
    HarnessGraphError,
    HarnessGraphExecutor,
    HarnessNodeSpec,
    HarnessStageRuntime,
    HarnessTopology,
    resolve_harness_runtime_policy,
)


# 功能：
#   构造单个只读节点的拓扑，便于隔离缓存和调用预算的行为。
# 输入：
#   无。
# 输出：
#   topology：没有真实外部调用的测试拓扑。
def _topology():
    topology = HarnessTopology(
        topology_id="test.runtime",
        name="Runtime test",
        nodes=[
            HarnessNodeSpec(
                node_id="stage.reader",
                handler_id="reader",
                authority="read",
                cacheable=True,
                output_key="result",
            ),
        ],
    )
    return topology


# 功能：
#   返回每次独立的最小合法策略集合，未指定预算继续使用核心默认值。
# 输入：
#   无。
# 输出：
#   policies：六类运行策略的测试字典。
def _policies():
    policies = {
        "scheduler": {"strategy": "parallel-ready"},
        "retry": {},
        "timeout": {},
        "budget": {},
        "fallback": {"optional_failure": "stop"},
        "cache": {},
    }
    return policies


# 功能：
#   同名拓扑更新配置后不能命中旧缓存，摘要必须绑定完整运行配置而不只是名字。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_binds_current_topology_configuration():
    handler = Mock(side_effect=["first", "second"])
    executor = HarnessGraphExecutor(handlers={"reader": handler})
    topology = _topology()
    assert executor.run(topology, {}).outputs["result"] == "first"
    topology.metadata = {"map_selection": "changed"}
    assert executor.run(topology, {}).outputs["result"] == "second"
    assert handler.call_count == 2


# 功能：
#   替换同名处理器后旧缓存失效，不能把旧实现的结果归给新实现。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_does_not_survive_handler_replacement():
    first = Mock(return_value="first")
    second = Mock(return_value="second")
    executor = HarnessGraphExecutor(handlers={"reader": first})
    topology = _topology()
    executor.run(topology, {})
    executor.handlers["reader"] = second
    assert executor.run(topology, {}).outputs["result"] == "second"
    second.assert_called_once()


# 功能：
#   不能为非有限结果生成有效回执，也不能先把非法结果放进缓存再宣告失败。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_invalid_output_cannot_poison_cache():
    executor = HarnessGraphExecutor(handlers={"reader": Mock(return_value={"value": float("nan")})})
    with pytest.raises(HarnessGraphError):
        executor.run(_topology(), {})
    assert not executor._cache


# 功能：
#   调用者修改一次返回的阶段回执后，内部完成账本和后续 finish 结果保持原值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stage_receipts_are_not_mutable_authority_references():
    runtime = HarnessStageRuntime(_topology())
    receipt = runtime.complete("stage.reader", inputs={}, output={"read": True})
    receipt.node_id = "stage.other"
    completed = runtime.finish()
    assert completed[0].node_id == "stage.reader"
    completed[0].output_sha256 = "0" * 64
    assert runtime.finish()[0].output_sha256 != "0" * 64


# 功能：
#   阶段运行器在接收被原地改写的非法拓扑时立即拒绝，不建立看似有效的完成账本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stage_runtime_revalidates_modified_topology():
    topology = _topology()
    topology.nodes[0].depends_on = ["missing"]
    with pytest.raises(ValueError):
        HarnessStageRuntime(topology)


# 功能：
#   拒绝布尔、文本和非有限策略数值；不能通过截断或夹紧把错误输入变成正常预算。
# 输入：
#   group：策略所属分类。
#   key：待破坏的策略字段。
#   value：非法数值或开关类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("group", "key", "value"),
    [
        ("scheduler", "maximum_parallelism", True),
        ("scheduler", "maximum_parallelism", "4"),
        ("retry", "provider_attempts", 1.5),
        ("timeout", "local_stage_seconds", float("nan")),
        ("timeout", "mission_seconds", float("inf")),
        ("budget", "maximum_nodes", "128"),
        ("cache", "cache_read_only", "false"),
    ],
)
def test_runtime_policy_rejects_invalid_numeric_types(group, key, value):
    policies = _policies()
    policies[group][key] = value
    with pytest.raises(ValueError):
        resolve_harness_runtime_policy(
            _topology(),
            policies,
            maximum_model_calls=48,
            maximum_tool_calls=16,
            model_timeout_seconds=180,
        )


# 功能：
#   选择停止策略后可选节点不能继续沿用隔离失败策略，同时不放宽必要节点边界。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_stop_policy_tightens_optional_node_failure_mode():
    topology = _topology()
    topology.nodes[0].failure_mode = "isolate"
    result = resolve_harness_runtime_policy(
        topology,
        _policies(),
        maximum_model_calls=48,
        maximum_tool_calls=16,
        model_timeout_seconds=180,
    )
    assert result.topology.nodes[0].failure_mode == "fail-closed"


# 功能：
#   同层节点写入相同输出键时，在任何处理器执行前拒绝，不允许排序决定谁覆盖谁。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_parallel_output_collision_is_rejected_before_calls():
    handler = Mock(return_value="value")
    topology = _topology()
    topology.nodes.append(topology.nodes[0].model_copy(update={"node_id": "stage.second"}))
    with pytest.raises((ValueError, HarnessGraphError)):
        HarnessGraphExecutor(handlers={"reader": handler}).run(topology, {})
    handler.assert_not_called()


# 功能：
#   主处理器自己抛出的 TimeoutError 表示调用已结束，不能误认为后台线程仍在执行而禁用回退。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_completed_handler_timeout_allows_bounded_fallback():
    fallback = Mock(return_value="recovered")
    topology = _topology()
    topology.nodes[0].failure_mode = "fallback"
    topology.nodes[0].fallback_handler_id = "safe"
    executor = HarnessGraphExecutor(
        handlers={"reader": Mock(side_effect=TimeoutError("remote"))},
        fallback_handlers={"safe": fallback},
    )
    result = executor.run(topology, {})
    assert result.outputs["result"] == "recovered"
    fallback.assert_called_once()


# 功能：
#   同步执行器拒绝未等待的协程结果，并关闭其帧，不能缓存它或遗留未等待警告。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_async_result_is_closed_and_rejected():
    # 功能：
    #   提供一个不应被同步图实际执行的协程。
    # 输入：
    #   无。
    # 输出：
    #   result：仅在协程被等待时才会产生的值。
    async def deferred():
        result = "not executed"
        return result

    pending = deferred()
    try:
        executor = HarnessGraphExecutor(handlers={"reader": Mock(return_value=pending)})
        with pytest.raises(HarnessGraphError):
            executor.run(_topology(), {})
        assert inspect.getcoroutinestate(pending) == inspect.CORO_CLOSED
        assert not executor._cache
    finally:
        pending.close()


# 功能：
#   验证实际等待超时后，处理器迟到返回的协程仍被关闭；不执行或缓存迟到结果。
# 输入：
#   monkeypatch：只替换当前测试的完成回调，用事件等待回收而不依赖固定休眠。
# 输出：
#   None：不返回业务数据。
def test_late_async_result_is_discarded_after_timeout(monkeypatch):
    released = threading.Event()
    disposed = threading.Event()
    original = graph_module._discard_late_result

    # 功能：
    #   提供不允许由超时同步调用执行的异步结果。
    # 输入：
    #   无。
    # 输出：
    #   result：只有等待协程后才会产生的测试值。
    async def deferred():
        result = "late"
        return result

    pending = deferred()

    # 功能：
    #   等待测试释放事件，模拟超时之后才完成的外部处理器。
    # 输入：
    #   inputs：未被本处理器使用的独立输入映射。
    # 输出：
    #   pending：未执行的测试协程。
    def delayed(inputs):
        assert released.wait(5)
        return pending

    # 功能：
    #   使用真实回收逻辑并通知测试线程，保证断言发生在回收结束之后。
    # 输入：
    #   future：已经完成的迟到调用。
    # 输出：
    #   None：不返回业务数据。
    def discard(future):
        try:
            original(future)
        finally:
            disposed.set()

    monkeypatch.setattr(graph_module, "_discard_late_result", discard)
    try:
        with pytest.raises(graph_module._InvocationTimeout):
            graph_module._invoke_handler(delayed, {}, 0.01)
        released.set()
        assert disposed.wait(5)
        assert inspect.getcoroutinestate(pending) == inspect.CORO_CLOSED
    finally:
        released.set()
        pending.close()


# 功能：
#   拒绝布尔、零阈值及非法冷却数值，不用隐式转换改变失败计数的意义。
# 输入：
#   settings：单次测试的非法阈值或冷却配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "settings",
    [
        {"failure_threshold": True},
        {"failure_threshold": 0},
        {"failure_threshold": 1.5},
        {"recovery_seconds": float("nan")},
        {"recovery_seconds": float("inf")},
        {"recovery_seconds": -1},
        {"recovery_seconds": "30"},
    ],
)
def test_breaker_rejects_invalid_configuration(settings):
    with pytest.raises(ValueError):
        HarnessCircuitBreaker(**settings)


# 功能：
#   使用可控单调时钟验证失败阈值、冷却到期及成功重置，避免依赖真实等待。
# 输入：
#   monkeypatch：仅替换本测试模块使用的单调时钟。
# 输出：
#   None：不返回业务数据。
def test_breaker_recovers_at_deadline_and_success_clears_failures(monkeypatch):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(graph_module.time, "monotonic", clock)
    breaker = HarnessCircuitBreaker(failure_threshold=2, recovery_seconds=30.0)
    breaker.failure("reader")
    assert breaker.allow("reader")
    breaker.failure("reader")
    assert not breaker.allow("reader")
    clock.return_value = 129.99
    assert not breaker.allow("reader")
    clock.return_value = 130.0
    assert breaker.allow("reader")
    breaker.failure("reader")
    breaker.success("reader")
    breaker.failure("reader")
    assert breaker.allow("reader")


# 功能：
#   超过只读缓存条数上限后淘汰最久未使用项，最近命中项仍可复用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cache_is_bounded_and_retains_recently_used_entry():
    handler = Mock(return_value="value")
    executor = HarnessGraphExecutor(handlers={"reader": handler})
    topology = _topology()
    for index in range(128):
        executor.run(topology, {"index": index})
    executor.run(topology, {"index": 0})
    assert handler.call_count == 128
    executor.run(topology, {"index": 128})
    assert len(executor._cache) == 128
    executor.run(topology, {"index": 0})
    assert handler.call_count == 129
    executor.run(topology, {"index": 1})
    assert handler.call_count == 130
    assert len(executor._cache) == 128


# 功能：
#   一个观察器修改事件副本时不污染下一个观察器或最终结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_observers_do_not_share_mutable_event_payloads():
    received = []

    # 功能：
    #   清空收到的事件映射，模拟有缺陷但无执行权限的观察器。
    # 输入：
    #   event：本次事件名。
    #   payload：观察器自己的资料副本。
    # 输出：
    #   None：不返回业务数据。
    def mutate(event, payload):
        payload.clear()

    # 功能：
    #   收集独立事件供测试核对，不修改事件资料。
    # 输入：
    #   event：本次事件名。
    #   payload：本观察器独立收到的资料。
    # 输出：
    #   None：不返回业务数据。
    def capture(event, payload):
        received.append((event, payload))

    executor = HarnessGraphExecutor(
        handlers={"reader": Mock(return_value="value")}, observers=[mutate, capture]
    )
    result = executor.run(_topology(), {})
    assert result.outputs["result"] == "value"
    assert all(payload for _, payload in received)
    assert received[-1][1]["outputs"] == result.outputs
