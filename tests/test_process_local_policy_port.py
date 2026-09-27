"""Exercise real spawned IPC; no simulator, credentials or actuator access."""

import os
import time
from types import SimpleNamespace

import pytest

from dronedream_agent_core.process_local_policy_port import ProcessLocalPolicyPort, _freeze
from dronedream_agent_core.model_harness.model_port import ModelInvocationError


# 功能：提供可跨进程启动的确定性模型替身；输入：包和配置；输出：因果历史及受控故障。
class FakePolicy:
    def __init__(self, package, **options):
        self.history = []
        self.frames = []
        if options.get('fail_start'):
            raise ValueError('startup secret must not leak')

    def observe_navigation_snapshot(self, snapshot):
        self.history.append(snapshot)

    def prime_multimodal(self, value):
        self.frames.append(value)

    def call(self, **kwargs):
        if kwargs.get('crash'):
            os._exit(2)
        if kwargs.get('fail'):
            raise ValueError('private message must not leak')
        time.sleep(kwargs.get('delay', 0))
        return dict(history=self.history, frames=self.frames, value=kwargs.get('value'), pid=os.getpid())

    def close(self):
        pass


# 功能：创建最小可序列化连续控制清单；输入：无；输出：只供生命周期验证的包。
def package():
    return SimpleNamespace(manifest=SimpleNamespace(pilot_control_mode='normalized-body-velocity',
        package_id='test.process', maximum_inference_latency_ms=20))


@pytest.fixture
def port():
    instance = ProcessLocalPolicyPort(package(), _factory=FakePolicy)
    yield instance
    instance.close()
    assert not instance._process.is_alive()


# 功能：验证顺序、隔离和投递时冻结；输入：独立端口；输出：子进程看不到父进程后续改写。
def test_owned_input_and_causal_order(port):
    snapshot = {'values': [1]}
    assert port.observe_navigation_snapshot(snapshot)
    snapshot['values'][0] = 99
    port.prime_multimodal([{'frame': 'one'}])
    result = port.call(value='first', control_deadline_monotonic=time.monotonic()+.24)
    assert result['history'] == [{'values': [1]}]
    assert result['frames'] == [[{'frame': 'one'}]]
    assert result['pid'] != os.getpid()
    assert port.call(value='second')['value'] == 'second'


# 功能：迟到响应必须排空且不能冒充下一次响应；输入：端口；输出：两次独立返回。
def test_late_response_not_reused(port):
    start = time.monotonic()
    assert port.call(value='late', delay=.1, control_deadline_monotonic=start+.06)['value'] == 'late'
    assert time.monotonic() > start+.06
    assert port.call(value='next')['value'] == 'next'


# 功能：故障只暴露有界原因码，不能把模型输入或异常秘密交给前端；输入：端口；输出：可恢复失败。
def test_failure_and_next_call(port):
    with pytest.raises(ModelInvocationError) as caught:
        port.call(fail=True)
    assert caught.value.reason_code == 'LOCAL_POLICY_PROCESS_VALUEERROR'
    assert 'private message' not in str(caught.value)
    assert port.call(value='after failure')['value'] == 'after failure'


@pytest.mark.parametrize('deadline', [0, float('nan'), float('inf'), True])
def test_invalid_deadline_rejected_before_dispatch(port, deadline):
    with pytest.raises(ValueError, match='CONTROL_DEADLINE_INVALID'):
        port.call(control_deadline_monotonic=deadline)


def test_crashed_child_is_not_reused(port):
    with pytest.raises((RuntimeError, EOFError)):
        port.call(crash=True)
    with pytest.raises(RuntimeError, match='UNAVAILABLE'):
        port.call()


def test_startup_failure_and_payload_bound():
    with pytest.raises(RuntimeError, match='STARTUP_FAILED'):
        ProcessLocalPolicyPort(package(), _factory=FakePolicy, fail_start=True)
    with pytest.raises(ValueError, match='MESSAGE_TOO_LARGE'):
        _freeze(b'x' * (8*1024*1024))


def test_idempotent_close(port):
    port.close()
    port.close()
    assert not port._process.is_alive()
    with pytest.raises(RuntimeError, match='UNAVAILABLE'):
        port.call()


# 功能：在实际子进程中执行原有输入/输出校验，模型采用确定性后端；不连接飞控。
def real_port_factory(package, **options):
    from test_local_policy_packages import _PilotControlBackend
    from dronedream_agent_core.local_policy_port import LocalPolicyPort
    return LocalPolicyPort(package, backend=_PilotControlBackend(), **options)


def test_navigation_validation_runs_in_child(tmp_path):
    from test_local_policy_packages import _write_package
    from test_policy_prepared_input import snapshot
    from dronedream_agent_core.perception_runtime import request_text_navigation_decision
    model = _write_package(tmp_path/'model', package_id='test.process.real', continuous_control=True)
    port = ProcessLocalPolicyPort(model, _factory=real_port_factory)
    try:
        value = snapshot()
        port.observe_navigation_snapshot(value)
        result, selected = request_text_navigation_decision(port=port, snapshot=value)
        assert result.artifact.action == 'pilot-control'
        assert selected is None  # Continuous axes are not a discrete path candidate.
        value['goal_distance_m'] = 999
        with pytest.raises(ModelInvocationError):
            request_text_navigation_decision(port=port, snapshot=value)
    finally:
        port.close()


# 功能：公开错误白名单必须精确匹配，不能把附带私有内容的异常正文透传。
def test_failure_code_allowlist_is_exact():
    from dronedream_agent_core.process_local_policy_port import _failure
    assert _failure(ValueError('LOCAL_POLICY_CONTROL_DEADLINE_INVALID'))[1] == 'LOCAL_POLICY_CONTROL_DEADLINE_INVALID'
    assert _failure(ValueError('LOCAL_POLICY_CONTROL_DEADLINE_INVALID private message'))[1] == 'LOCAL_POLICY_PROCESS_VALUEERROR'


# 功能：跨进程诊断只保留受信源码行号，并能穿过协调器的数字指标过滤；不泄露错误正文。
# 输入：带私密正文的测试异常；输出：公开诊断包含合法行号且不含私密文本。
def test_source_locations_survive_public_diagnostic_filter():
    from dronedream_agent_core.process_local_policy_port import _failure
    from dronedream_agent_core.perception_runtime import _failure_diagnostic
    namespace = {}
    exec(compile("def fail():\n    raise ValueError('private input')\n",
                 "local_policy_port.py", "exec"), namespace)
    try:
        namespace['fail']()
    except ValueError as error:
        response = _failure(error)
    public = ModelInvocationError('Local process failed', attempts_used=1,
                                  reason_code=response[1])
    public.diagnostic_metrics = response[2]
    diagnostic = _failure_diagnostic(public, stage='model-invocation')
    assert diagnostic['failure_diagnostic_metrics']['source-line-local-policy-port'] == 2
    assert 'private input' not in str(diagnostic)


# 功能：队列背压属于零次调用的准入拒绝，不能构造非法调用异常覆盖真实原因。
# 输入：隔离端口、满队列替身；输出：原因码与次数准确，下一次仍可调用。
def test_queue_backpressure_is_zero_attempt_admission(port, monkeypatch):
    from dronedream_agent_core.process_local_policy_port import LocalPolicyAdmissionError
    with monkeypatch.context() as patch:
        patch.setattr(port, '_enqueue', lambda *args: False)
        with pytest.raises(LocalPolicyAdmissionError) as caught:
            port.call(value='not submitted')
        assert caught.value.reason_code == 'LOCAL_POLICY_PROCESS_BACKPRESSURE'
        assert caught.value.attempts_used == 0
    assert port.call(value='next')['value'] == 'next'


# 功能：子进程收到已过期请求不运行模型，回传原始准入原因而不是二次 ValueError。
# 输入：故意延后投递的端口；输出：零次调用、无模型回执，随后请求仍可执行。
def test_child_expiration_preserves_zero_attempt_reason(port, monkeypatch):
    from dronedream_agent_core.process_local_policy_port import LocalPolicyAdmissionError
    enqueue = port._enqueue
    def delayed_enqueue(operation, value):
        time.sleep(.08)
        return enqueue(operation, value)
    with monkeypatch.context() as patch:
        patch.setattr(port, '_enqueue', delayed_enqueue)
        with pytest.raises(LocalPolicyAdmissionError) as caught:
            port.call(value='expired', control_deadline_monotonic=time.monotonic()+.04)
        assert caught.value.reason_code == 'LOCAL_POLICY_PROCESS_INPUT_EXPIRED'
        assert caught.value.attempts_used == 0
        assert not hasattr(caught.value, 'model_call_record')
    assert port.call(value='next')['value'] == 'next'
