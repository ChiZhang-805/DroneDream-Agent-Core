"""Offline role-call tests: strict quorum, bounded fallbacks and per-response usage receipts."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import ModelCallRecord, PlanCritique
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.model_port import (
    ModelInvocationError,
    StructuredCallResult,
)
from dronedream_agent_core.orchestrator import MissionOrchestrator, MissionPreparationBlocked
from dronedream_agent_plugins.model_runtime_plugins import _usage_meter


class _Evidence:
    """Capture emitted event order without accessing an account or persistent evidence store."""

    # 功能：
    #   创建只存内存的事件列表，避免测试访问账户或持久化证据。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.events = []

    # 功能：
    #   按到达顺序记录事件，供断言计量与共识检查的先后关系。
    # 输入：
    #   name：事件名称。
    #   value：对应事件内容。
    # 输出：
    #   None：不返回业务数据。
    def append(self, name, value):
        self.events.append((name, value))


class _Port:
    """Return one deterministic fixture or a bounded invocation failure; never open a transport."""

    # 功能：
    #   设置不联网的模型端口夹具，允许模拟接受、异议和调用失败。
    # 输入：
    #   name：供应商名称夹具。
    #   identifier：用于组成调用身份的测试字符。
    #   accepted：返回的计划审查是否接受。
    #   fail：是否在返回材料前报告一次物理调用失败。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, name, identifier, *, accepted=True, fail=False):
        self.settings = SimpleNamespace(name=name, model="fixture-model")
        self.identifier = identifier
        self.accepted = accepted
        self.fail = fail
        self.supports_image_input = False
        self.supports_provider_context = False
        self.calls = []

    # 功能：
    #   保存调用参数，返回摘要与计数绑定的结构结果，或模拟消耗一次尝试的异常。
    # 输入：
    #   kwargs：真实编排器交给端口的结构化调用参数。
    # 输出：
    #   result：计划审查夹具及其调用记录。
    def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ModelInvocationError("test failure", attempts_used=1)
        artifact = PlanCritique(
            accepted=self.accepted,
            issue_codes=[] if self.accepted else ["UNSAFE"],
            repair_instructions=[],
        )
        record = ModelCallRecord(
            call_id="model-" + self.identifier * 24,
            role="plan_critic",
            attempt=1,
            input_sha256="a" * 64,
            output_sha256=sha256_json(artifact),
            output_schema="PlanCritique",
            provider=self.settings.name,
            model=self.settings.model,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            created_at=datetime.now(UTC),
        )
        result = StructuredCallResult(artifact=artifact, record=record)
        return result


# 功能：
#   为真实编排器调用方法接入内存端口、路由策略和证据，不初始化外部依赖。
# 输入：
#   ports：端口名称到测试实例的映射。
#   candidates：路由器应返回的候选名称序列。
#   policy：共识钩子应返回的配置。
#   budget：允许消耗的物理尝试总量。
# 输出：
#   harness：编排器实例、上下文记录列表和证据接收器组成的三元组。
def _harness(ports, candidates, policy, *, budget=10):
    instance = MissionOrchestrator.__new__(MissionOrchestrator)
    instance.model_ports = ports
    instance.primary = ports["primary"]
    instance.critic = ports.get("critic", object())
    instance.config = SimpleNamespace(persisted_task_context=False)
    instance._model_media = []
    instance._model_call_budget = budget
    instance._model_call_count = 0
    stored = []
    instance.context_store = SimpleNamespace(append=lambda *args, **kwargs: stored.append(kwargs))

    # 功能：
    #   按真实模型插槽名提供本例选择结果，未声明插槽立即使测试失败。
    # 输入：
    #   slot：宿主正在调用的策略插槽。
    #   hook：宿主请求的钩子名。
    #   kwargs：测试策略不使用的其余调用参数。
    # 输出：
    #   selection：对应插槽的路由、共识配置或空角色覆盖。
    def select(slot, hook, **kwargs):
        if slot == "models.role-policy":
            selection = None
        elif slot == "models.runtime-router":
            selection = {"candidates": candidates}
        elif slot == "models.consensus-policy":
            selection = policy
        else:
            raise AssertionError(slot)
        return selection

    instance._invoke_single_extension = select
    instance._invoke_extension_pipeline = lambda slot, hook, value, **kwargs: value
    instance._invoke_multiple_extensions = lambda slot, hook, **kwargs: [
        _usage_meter(record=kwargs["record"])
    ]
    harness = (instance, stored, _Evidence())
    return harness


# 功能：
#   用固定计划审查请求进入生产编排方法，不在测试中重写共识逻辑。
# 输入：
#   instance：已连接测试端口与策略的编排器。
#   evidence：本次调用使用的内存证据接收器。
# 输出：
#   result：真实编排逻辑选出的结构化调用结果。
def _call(instance, evidence):
    result = instance._call(
        port=instance.primary,
        role="plan_critic",
        output_type=PlanCritique,
        instructions="Review this plan.",
        input_artifact={"plan": "fixture"},
        conversation_id="test",
        evidence=evidence,
    )
    return result


# 功能：
#   验证首端口失败后回退，每个成功响应均留用量证据，只有选中材料进入对话上下文。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_peer_falls_through_and_every_returned_peer_has_usage():
    ports = {
        "primary": _Port("first", "1", fail=True),
        "critic": _Port("second", "2"),
        "safety": _Port("third", "3"),
    }
    instance, stored, evidence = _harness(
        ports, list(ports), {"minimum_responses": 2, "maximum_responses": 2}
    )
    result = _call(instance, evidence)
    assert instance._model_call_count == 3
    assert result.record.provider == "second"
    assert len(result.supporting_records) == 1
    usages = [value for name, value in evidence.events if name.endswith("response-usage")]
    assert [value["port"] for value in usages] == ["critic", "safety"]
    assert sum(value["metering"][0]["total_tokens"] for value in usages) == 30
    assert len(stored) == 1  # Only the selected artifact enters conversational context.
    consensus = next(value for name, value in evidence.events if name.endswith("consensus"))
    assert consensus["candidate_ports"] == list(ports)
    assert consensus["response_records"][1]["input_tokens"] == 10


# 功能：
#   验证重复名称或同一端口对象的别名不能凑足双响应要求，拒绝发生在调用前。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_names_and_alias_objects_cannot_create_quorum():
    port = _Port("same", "1")
    instance, _, evidence = _harness(
        {"primary": port, "critic": port},
        ["primary", "primary", "critic"],
        {"minimum_responses": 2, "maximum_responses": 2},
    )
    with pytest.raises(MissionPreparationBlocked, match="INSUFFICIENT_RESPONSES"):
        _call(instance, evidence)
    assert not port.calls


# 功能：
#   验证共识数量和开关拒绝布尔、文本与小数的隐式转换，且不消耗模型调用。
# 输入：
#   update：覆盖正常共识配置的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update",
    [
        {"minimum_responses": True},
        {"maximum_responses": "2"},
        {"maximum_responses": 1.5},
        {"require_identical": "false"},
        {"record_dissent": 1},
    ],
)
def test_policy_scalars_are_not_coerced_to_quorum_or_consent(update):
    port = _Port("first", "1")
    instance, _, evidence = _harness(
        {"primary": port}, ["primary"], {"minimum_responses": 1, "maximum_responses": 1} | update
    )
    with pytest.raises(MissionPreparationBlocked, match="BOUNDS_INVALID"):
        _call(instance, evidence)
    assert not port.calls


# 功能：
#   验证严格共识遭遇异议时拒绝结果，但仍保留每个已返回响应的用量证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_dissent_keeps_usage_evidence_even_when_no_result_is_admitted():
    ports = {"primary": _Port("first", "1"), "critic": _Port("second", "2", accepted=False)}
    instance, stored, evidence = _harness(
        ports,
        list(ports),
        {"minimum_responses": 2, "maximum_responses": 2, "require_identical": True},
    )
    with pytest.raises(MissionPreparationBlocked, match="CONSENSUS_DISSENT"):
        _call(instance, evidence)
    assert len([name for name, _ in evidence.events if name.endswith("response-usage")]) == 2
    assert not stored


# 功能：
#   验证首端口失败耗尽物理预算后不调用回退端口，不把失败尝试排除在计数之外。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_fallback_never_exceeds_the_physical_attempt_budget():
    ports = {"primary": _Port("first", "1", fail=True), "critic": _Port("second", "2")}
    instance, _, evidence = _harness(
        ports, list(ports), {"minimum_responses": 1, "maximum_responses": 1}, budget=1
    )
    with pytest.raises(MissionPreparationBlocked, match="BUDGET_EXCEEDED"):
        _call(instance, evidence)
    assert instance._model_call_count == 1
    assert not ports["critic"].calls


# 功能：
#   验证单响应策略在首个成功结果后结束，不因存在其他端口而额外消耗调用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_single_response_policy_does_not_call_extra_peers():
    ports = {"primary": _Port("first", "1"), "critic": _Port("second", "2")}
    instance, _, evidence = _harness(
        ports, list(ports), {"minimum_responses": 1, "maximum_responses": 1}
    )
    _call(instance, evidence)
    assert not ports["critic"].calls
