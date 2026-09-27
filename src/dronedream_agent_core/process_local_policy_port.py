"""Run local experts outside the sensor interpreter; no actuator access here.

The bounded IPC channel is private to two owned processes, never a network or
plugin deserialization interface. Original monotonic deadlines cross unchanged.
"""

import math
import multiprocessing as mp
import pickle
from pathlib import Path
import queue
import threading
import time

from .local_policy_port import LocalPolicyPort
from .model_harness.model_port import ModelInvocationError, ProviderSettings
from .contracts import ModelCallRecord

_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_PUBLIC_FAILURE_CODES = frozenset({
    'LOCAL_POLICY_CONTROL_DEADLINE_INVALID', 'LOCAL_POLICY_SNAPSHOT_HASH_INVALID',
    'TEXT_NAVIGATION_INPUT_INVALID', 'TEXT_NAVIGATION_SNAPSHOT_HASH_INVALID',
    'TEXT_NAVIGATION_DECISION_SNAPSHOT_MISMATCH', 'TEXT_NAVIGATION_UNHEALTHY_STREAM_REQUIRES_HOLD',
    'TEXT_NAVIGATION_CANDIDATE_SET_MISSING', 'TEXT_NAVIGATION_CANDIDATE_IDENTITY_INVALID',
    'TEXT_NAVIGATION_CANDIDATE_NOT_AUTHORIZED', 'TEXT_NAVIGATION_CANDIDATE_NOT_METRIC_VALIDATED',
})


class LocalPolicyAdmissionError(RuntimeError):
    """No inference attempt was made; an expired/full-queue request is not a model failure."""

    # 功能：记录入队或执行前拒绝，明确零次推理，不破坏 ModelInvocationError 的正数约束。
    # 输入：内部固定原因码；输出：可由协调器记录的无动作准入异常。
    def __init__(self, reason_code):
        if reason_code not in {'LOCAL_POLICY_PROCESS_INPUT_EXPIRED',
                               'LOCAL_POLICY_PROCESS_BACKPRESSURE'}:
            raise ValueError('LOCAL_POLICY_ADMISSION_REASON_INVALID')
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.attempts_used = 0


# 功能：在进入异步队列前冻结进程内部消息，避免后台序列化期间调用者修改内容。
# 输入：本进程创建的消息；输出：有大小上限的私有 IPC 字节，不能用于读取外部 pickle。
def _freeze(message):
    data = pickle.dumps(message, protocol=5)
    if len(data) > _MAX_MESSAGE_BYTES:
        raise ValueError('LOCAL_POLICY_PROCESS_MESSAGE_TOO_LARGE')
    return data


# 功能：把有限错误诊断交回父进程，不传递原始输入、图像或可能带秘密的异常正文。
# 输入：本地调用异常；输出：有限类型、原因码和已有数值诊断。
def _failure(error):
    code = getattr(error, 'reason_code', None)
    if code is None and type(error) is ValueError and str(error) in _PUBLIC_FAILURE_CODES:
        code = str(error)
    if not isinstance(code, str) or len(code) > 128:
        code = 'LOCAL_POLICY_PROCESS_' + type(error).__name__.upper()
    metrics = getattr(error, 'diagnostic_metrics', {})
    metrics = {k:v for k,v in metrics.items() if isinstance(k, str) and len(k) <= 128
               and type(v) in (int, float) and -1e12 <= v <= 1e12} if isinstance(metrics, dict) else {}
    # 只报告本包允许列表中的源文件行号，不输出异常正文、输入值或用户路径。
    traceback = error.__traceback__
    while traceback is not None:
        filename = Path(traceback.tb_frame.f_code.co_filename).name
        if filename in {"local_policy_port.py", "perception_runtime.py", "local_policy_features.py",
                         "process_local_policy_port.py", "causal_control.py"}:
            metrics[f"source-line-{filename[:-3].replace('_', '-')}"] = traceback.tb_lineno
        traceback = traceback.tb_next
    record = getattr(error, 'model_call_record', None)
    return ('error', code, dict(list(metrics.items())[:32]),
            record if isinstance(record, ModelCallRecord) else None)


# 功能：唯一子进程串行维护因果历史和模型，按原始截止时间推理，不连接任何飞控接口。
# 输入：父进程私有有界队列、响应管道、已选择模型包及构造参数。
# 输出：就绪或完整调用结果；观察、视觉预取失败会使进程退出，不能隐瞒损坏继续控制。
def _serve(inbox, responses, package, options, factory):
    port = None
    try:
        port = factory(package, **options)
        responses.send_bytes(_freeze(('ready',)))
        while True:
            data = inbox.get()
            if len(data) > _MAX_MESSAGE_BYTES:
                raise ValueError('LOCAL_POLICY_PROCESS_MESSAGE_TOO_LARGE')
            operation, value = pickle.loads(data)
            if operation == 'close':
                break
            if operation == 'observe':
                port.observe_navigation_snapshot(value)
            elif operation == 'prime':
                port.prime_multimodal(value)
            elif operation in ('call', 'navigation'):
                try:
                    deadline = value.get('control_deadline_monotonic')
                    if deadline is not None and time.monotonic() >= deadline:
                        raise LocalPolicyAdmissionError('LOCAL_POLICY_PROCESS_INPUT_EXPIRED')
                    if operation == 'navigation':
                        from .perception_runtime import request_text_navigation_decision
                        result = request_text_navigation_decision(port=port, **value)
                    else:
                        result = port.call(**value)
                    response = ('result', result)
                except Exception as error:
                    response = _failure(error)
                responses.send_bytes(_freeze(response))
            else:
                raise ValueError('LOCAL_POLICY_PROCESS_OPERATION_INVALID')
    except BaseException as error:
        try:
            responses.send_bytes(_freeze(_failure(error)))
        except (OSError, EOFError):
            pass
    finally:
        if port is not None:
            port.close()
        responses.close()


class ProcessLocalPolicyPort:
    """Single-inference, bounded parent facade; sensor callbacks never wait for ONNX."""

    supports_observation_history = True
    supports_control_deadline = True
    supports_provider_context = False

    # 功能：启动独立模型进程并核对就绪，在解锁飞机前完成权重加载；不复制感知进程线程状态。
    # 输入：连续控制包、原有模型选项；_factory 仅供隔离生命周期测试注入替身。
    # 输出：只接受一个在途推理的端口；启动异常完整回收自己创建的进程。
    def __init__(self, package, *, _factory=LocalPolicyPort, **options):
        if package.manifest.pilot_control_mode != 'normalized-body-velocity':
            raise ValueError('LOCAL_POLICY_PROCESS_REQUIRES_CONTINUOUS_MODE')
        self.package = package
        self.settings = ProviderSettings(name='local-policy', model=package.manifest.package_id,
            api_key_env='', base_url=None, api_style='chat-completions')
        self.invocation_timeout_seconds = package.manifest.maximum_inference_latency_ms / 1000.
        context = mp.get_context('spawn')
        self._inbox = context.Queue(maxsize=16)
        self._responses, child = context.Pipe(duplex=False)
        self._call_lock = threading.Lock()
        self._closed = False
        self._process = context.Process(target=_serve,
            args=(self._inbox, child, package, options, _factory), name='local-policy-inference')
        try:
            self._process.start()
            child.close()
            if not self._responses.poll(30.):
                raise TimeoutError('LOCAL_POLICY_PROCESS_STARTUP_TIMEOUT')
            result = pickle.loads(self._responses.recv_bytes(_MAX_MESSAGE_BYTES))
            if result != ('ready',):
                raise RuntimeError('LOCAL_POLICY_PROCESS_STARTUP_FAILED')
        except BaseException:
            child.close()
            self.close()
            raise

    # 功能：立即冻结并投递有界消息，不等待传感器线程之外的消费者；满队列明确报告背压。
    # 输入：操作与本进程拥有的内容；输出：是否入队，绝不覆盖已经排队的因果历史。
    def _enqueue(self, operation, value):
        if self._closed or not self._process.is_alive():
            raise RuntimeError('LOCAL_POLICY_PROCESS_UNAVAILABLE')
        try:
            self._inbox.put_nowait(_freeze((operation, value)))
            return True
        except queue.Full:
            return False

    # 功能：投递真实数值历史，不在感知线程重复编译张量；输入：完整快照；输出：是否入队。
    def observe_navigation_snapshot(self, snapshot):
        return self._enqueue('observe', snapshot)

    # 功能：预取已由协调器核对来源的图像；输入：图像消息；输出：无动作，背压时稍后同步推理再处理。
    def prime_multimodal(self, multimodal):
        self._enqueue('prime', multimodal)

    # 功能：等待唯一在途推理，跨进程保留原截止时间；迟到结果只交回协调器判定，不能给它续期。
    # 输入：统一模型端口调用参数；输出：原始结构化决策及完整模型回执。
    def call(self, **kwargs):
        return self._request('call', kwargs)

    # 功能：把输入副本及输出绑定核验一同放入独立进程，避免感知解释器承担重复 JSON 拷贝。
    # 输入：原导航请求；输出：由原校验函数核对的决策和候选，不省略任何输出检查。
    def request_navigation(self, **kwargs):
        return self._request('navigation', kwargs)

    # 功能：统一串行调用通道；输入：内部操作及参数；输出：一一对应的完整响应。
    def _request(self, operation, kwargs):
        deadline = kwargs.get('control_deadline_monotonic')
        if deadline is not None and (type(deadline) not in (int, float)
                or not math.isfinite(deadline) or not time.monotonic() < deadline <= time.monotonic()+.25):
            raise ValueError('LOCAL_POLICY_CONTROL_DEADLINE_INVALID')
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('LOCAL_POLICY_CONCURRENT_CALL_NOT_ALLOWED')
        try:
            if not self._enqueue(operation, kwargs):
                raise LocalPolicyAdmissionError('LOCAL_POLICY_PROCESS_BACKPRESSURE')
            # Do not release the in-flight slot on the action deadline: the
            # coordinator already quarantines expired calls. Drain its reply so
            # a late result can never be confused with the following request.
            drain_until = time.monotonic() + 5.
            while not self._responses.poll(.005):
                if self._closed or not self._process.is_alive():
                    raise RuntimeError('LOCAL_POLICY_PROCESS_UNAVAILABLE')
                if time.monotonic() >= drain_until:
                    self.close()
                    raise TimeoutError('LOCAL_POLICY_PROCESS_DRAIN_TIMEOUT')
            result = pickle.loads(self._responses.recv_bytes(_MAX_MESSAGE_BYTES))
            if result[0] == 'error':
                error = (LocalPolicyAdmissionError(result[1])
                    if result[1] in {'LOCAL_POLICY_PROCESS_INPUT_EXPIRED',
                                     'LOCAL_POLICY_PROCESS_BACKPRESSURE'}
                    else ModelInvocationError('Local inference process rejected the request',
                        attempts_used=1, reason_code=result[1]))
                error.diagnostic_metrics = result[2]
                if result[3] is not None:
                    error.model_call_record = result[3]
                raise error
            if result[0] != 'result':
                raise RuntimeError('LOCAL_POLICY_PROCESS_RESPONSE_INVALID')
            return result[1]
        except (OSError, EOFError) as error:
            # The pipe can close before is_alive() observes process exit.
            # Retire that endpoint immediately; never submit another request.
            self.close()
            raise RuntimeError('LOCAL_POLICY_PROCESS_UNAVAILABLE') from error
        finally:
            self._call_lock.release()

    # 功能：符合原有端口重置协议，不重启模型、不抹除历史、不产生第二个并发推理。
    # 输入：无；输出：无。
    def reset_transport(self):
        return None

    # 功能：保持选定模型不变，不在失败后静默切换提供者；输入：无；输出：无。
    def advance_after_failure(self):
        return None

    # 功能：幂等回收自己创建的模型进程；无飞控连接，不终止仿真或其他用户进程。
    # 输入：无；输出：无后台模型残留，未发送的输入不会在下一次运行复活。
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._inbox.put_nowait(_freeze(('close', None)))
        except queue.Full:
            pass
        if self._process.pid is not None:
            self._process.join(timeout=2.)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.)
        self._inbox.cancel_join_thread()
        self._inbox.close()
        self._responses.close()
