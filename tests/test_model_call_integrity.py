"""离线复现调用输入身份、亚秒截止时间及备用通道撤销；不调用真实模型。"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_model_port_attempts import _port

from dronedream_agent_core.contracts import IntentArtifact
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness import model_port
from dronedream_agent_core.model_harness.model_port import (
    FailoverStructuredModelPort,
    ModelInvocationError,
)


# 功能：
#   为调用身份测试构造独立的合法意图，不能当作真实模型或飞行结果。
# 输入：
#   无。
# 输出：
#   artifact：供离线传输替身返回的意图对象。
def _intent():
    artifact = IntentArtifact(
        goal="inspect gate",
        start_entity="office",
        target_entity="gate",
        return_entity="office",
        payload_action="none",
    )
    return artifact


# 功能：
#   提供没有网络连接的模型端口，并等待其异步清理结束。
# 输入：
#   monkeypatch：pytest 替换器。
# 输出：
#   port：只用于当前测试的模型调用端口。
@pytest.fixture
def offline_port(monkeypatch):
    monkeypatch.setattr(model_port, "OpenAI", lambda **kwargs: SimpleNamespace(close=lambda: None))
    port = _port()
    try:
        yield port
    finally:
        port.close()
        with port._transport_changed:
            assert port._transport_changed.wait_for(lambda: port._transport_worker is None, 2.0)


# 功能：
#   验证任务可以把较长构造上限收紧到亚秒级，而不被隐式抬高到一秒。
# 输入：
#   offline_port：无网络端口。
# 输出：
#   None：不返回业务数据。
def test_mission_timeout_preserves_requested_subsecond_bound(offline_port):
    offline_port.configure_execution_policy(maximum_attempts=1, timeout_seconds=0.05)
    assert offline_port.invocation_timeout_seconds == 0.05


# 功能：
#   验证调用回执绑定实际发送的输入快照，而不是调用结束时的可变原对象。
# 输入：
#   offline_port：无网络端口。
#   monkeypatch：pytest 替换器。
# 输出：
#   None：不返回业务数据。
def test_record_hash_binds_sent_input_not_later_caller_mutation(offline_port, monkeypatch):
    request = {"message": "inspect gate", "parameters": {"speed": 1.0}}
    expected_hash = sha256_json(request)
    sent = []

    # 功能：
    #   捕获实际输入后改写调用方对象，模拟其他任务并发修改草稿。
    # 输入：
    #   kwargs：端口发给传输适配器的参数。
    # 输出：
    #   response：合法意图、响应标识及输入输出 token 数。
    def reply(**kwargs):
        sent.append(kwargs["input_json"])
        request["parameters"]["speed"] = 100.0
        response = (_intent(), "response-current", 10, 5)
        return response

    monkeypatch.setattr(offline_port, "_responses_call", reply)
    result = offline_port.call(
        role="intent_parser",
        output_type=IntentArtifact,
        instructions="Return intent",
        input_artifact=request,
    )
    assert '"speed":1.0' in sent[0]
    assert result.record.input_sha256 == expected_hash


# 功能：
#   验证最大尝试次数无效时先拒绝输入，不等待正在恢复的传输。
# 输入：
#   offline_port：无网络端口。
#   monkeypatch：pytest 替换器。
# 输出：
#   None：不返回业务数据。
def test_invalid_attempt_budget_is_rejected_before_transport_wait(offline_port, monkeypatch):
    # 功能：
    #   使不应发生的传输等待立即暴露为测试失败。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def forbidden_snapshot():
        raise AssertionError("invalid request must not wait for transport")

    monkeypatch.setattr(offline_port, "_transport_snapshot", forbidden_snapshot)
    with pytest.raises(ValueError, match="maximum_physical_attempts"):
        offline_port.call(
            role="intent_parser",
            output_type=IntentArtifact,
            instructions="Return intent",
            input_artifact={},
            maximum_physical_attempts=True,
        )


class Peer:
    # 功能：
    #   建立可控的主备通道替身，用于单独测量生命周期而非模型推理。
    # 输入：
    #   name：通道名称。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, name):
        self.settings = SimpleNamespace(name=name)
        self.closed = 0
        self.resets = 0
        self.calls = 0
        self.invocation_timeout_seconds = 1.0

    # 功能：
    #   记录一次替身调用并返回合法形状的离线结果。
    # 输入：
    #   kwargs：透传请求。
    # 输出：
    #   result：携带实际尝试次数的测试结果。
    def call(self, **kwargs):
        self.calls += 1
        result = SimpleNamespace(record=SimpleNamespace(attempt=2))
        return result

    # 功能：
    #   记录恢复动作供通道切换断言使用。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def reset_transport(self):
        self.resets += 1

    # 功能：
    #   记录关闭动作，验证两端资源都得到清理。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        self.closed += 1


# 功能：
#   复现任务切换或关闭后，旧通道的成功回复仍被外层返回的问题。
# 输入：
#   monkeypatch：pytest 替换器。
#   close：是否以关闭代替通道切换。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("close", [False, True])
def test_failover_rejects_success_from_revoked_invocation(monkeypatch, close):
    primary, fallback = Peer("primary"), Peer("fallback")
    port = FailoverStructuredModelPort(primary, fallback)
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   暂停主通道回包，确保撤销发生在结果返回之前。
    # 输入：
    #   kwargs：透传请求。
    # 输出：
    #   result：两次物理尝试后的离线结果。
    def delayed(**kwargs):
        entered.set()
        assert release.wait(2.0)
        result = SimpleNamespace(record=SimpleNamespace(attempt=2))
        return result

    monkeypatch.setattr(primary, "call", delayed)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(port.call)
            try:
                assert entered.wait(1.0)
                port.close() if close else port.reset_transport()
            finally:
                release.set()
            with pytest.raises(ModelInvocationError) as failure:
                future.result(timeout=1.0)
            assert failure.value.reason_code == "MODEL_TRANSPORT_RETIRED"
            assert failure.value.attempts_used == 2
    finally:
        port.close()


# 功能：
#   验证一端关闭失败不妨碍另一端清理，并且关闭后的包装器拒绝新请求。
# 输入：
#   monkeypatch：pytest 替换器。
# 输出：
#   None：不返回业务数据。
def test_failover_close_revokes_admission_and_cleans_both_peers(monkeypatch):
    primary, fallback = Peer("primary"), Peer("fallback")
    port = FailoverStructuredModelPort(primary, fallback)

    # 功能：
    #   模拟主通道第三方关闭实现异常，检验异常后的另一端清理。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def broken_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(primary, "close", broken_close)
    with pytest.raises(RuntimeError, match="close failed"):
        port.close()
    assert fallback.closed == 1
    with pytest.raises(TimeoutError, match="UNAVAILABLE"):
        port.call()
    assert primary.calls == fallback.calls == 0
    port.reset_transport()
    port.advance_after_failure()
    assert primary.resets == fallback.resets == 0


# 功能：
#   防止文本真值、布尔次数等隐式类型转换改变主备调度或视觉权限。
# 输入：
#   kwargs：待拒绝的主备策略。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"primary_probe_after_fallback_successes": True},
        {"primary_probe_after_fallback_successes": 1.5},
        {"fallback_accepts_multimodal": "false"},
        {"retry_primary_after_returned_failure": "false"},
    ],
)
def test_failover_policy_rejects_ambiguous_types(kwargs):
    with pytest.raises(ValueError):
        FailoverStructuredModelPort(Peer("primary"), Peer("fallback"), **kwargs)


# 功能：
#   防止错误次数中的布尔值或小数污染上层实际调用预算。
# 输入：
#   attempts：无效的尝试次数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("attempts", [True, 1.5, "1"])
def test_invocation_error_requires_integer_attempts(attempts):
    with pytest.raises(ValueError, match="attempts_used"):
        ModelInvocationError("failed", attempts_used=attempts)
