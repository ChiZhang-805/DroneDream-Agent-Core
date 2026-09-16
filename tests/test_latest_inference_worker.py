"""Bounded compute scheduling with explicit synchronization, not flight evidence."""

import threading
from concurrent.futures import CancelledError

import pytest

from dronedream_agent_core.latest_inference_worker import LatestInferenceWorker


# 功能：
#   千次待处理提交只保留最新一项，不形成陈旧输入的回放队列。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_only_running_and_latest_pending_inputs_execute_with_no_replay_queue():
    entered, release, seen = threading.Event(), threading.Event(), []

    # 功能：
    #   用显式事件阻塞首项，记录实际执行项并计算简单结果。
    # 输入：
    #   value：当前测试整数。
    # 输出：
    #   result：输入的两倍。
    def encode(value):
        seen.append(value)
        if value == 0:
            entered.set()
            assert release.wait(2.)
        result = value * 2
        return result

    worker = LatestInferenceWorker(encode, name="bounded-unit-inference")
    try:
        first = worker.submit("0", 0)
        assert entered.wait(1.)
        pending = [worker.submit(str(i), i) for i in range(1, 1001)]
        assert worker.submit("1000", 1000) is pending[-1]
        assert all(future.cancelled() for future in pending[:-1])
        assert seen == [0]
        release.set()
        assert first.result(1.) == 0 and pending[-1].result(1.) == 2000
        assert seen == [0, 1000]
        assert worker.cached("1000").result() == 2000
        assert worker.cached("0") is None
    finally:
        release.set()
        receipt = worker.close()
    assert receipt["coalesced"] == 999 and receipt["completed"] == 2
    assert receipt["thread_stopped"] and receipt["maximum_pending_inputs"] == 1


# 功能：
#   被合并请求的取消回调可以重入缓存查询，不持有工作器锁调用回调。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancel_callbacks_can_reenter_without_deadlock():
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   保持首项运行，提供合并与回调重入的确定同步点。
    # 输入：
    #   value：输入标识。
    # 输出：
    #   value：原输入标识。
    def encode(value):
        if value == "running":
            entered.set()
            assert release.wait(2.)
        return value

    worker = LatestInferenceWorker(encode, name="callback-unit-inference")
    try:
        first = worker.submit("running", "running")
        assert entered.wait(1.)
        old = worker.submit("old", "old")
        called = []
        old.add_done_callback(lambda _: called.append(worker.cached("running")))
        # Returning to a currently active input cancels the displaced pending one.
        assert worker.submit("running", "running") is first
        assert old.cancelled() and called == [first]
        release.set()
        assert first.result(1.) == "running"
    finally:
        release.set()
        worker.close()


# 功能：
#   普通编码错误不回退旧成功结果，下一独立来源仍能正常运行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_source_never_returns_an_older_success_but_next_source_can_recover():
    # 功能：
    #   对指定坏输入抛出普通错误，其余输入原样返回。
    # 输入：
    #   value：当前来源标识。
    # 输出：
    #   value：合法来源标识。
    def encode(value):
        if value == "bad":
            raise ValueError("encoder error")
        return value

    worker = LatestInferenceWorker(encode, name="failure-unit-inference")
    try:
        assert worker.submit("good", "good").result(1.) == "good"
        failed = worker.submit("bad", "bad")
        with pytest.raises(ValueError, match="encoder error"):
            failed.result(1.)
        with pytest.raises(ValueError, match="encoder error"):
            worker.cached("bad").result(1.)
        assert worker.cached("good") is None
        assert worker.submit("next", "next").result(1.) == "next"
    finally:
        receipt = worker.close()
    assert receipt["failed"] == 1


# 功能：
#   被调用方取消的同键请求重新提交时，必须获得新的可完成 Future。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cancelled_pending_input_can_be_resubmitted_without_reusing_cancelled_future():
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   暂停活动来源，让测试在等待槽取消并重新提交输入。
    # 输入：
    #   value：来源标识。
    # 输出：
    #   value：来源标识。
    def encode(value):
        if value == "active":
            entered.set()
            assert release.wait(2.)
        return value

    worker = LatestInferenceWorker(encode, name="resubmit-unit-inference")
    try:
        active = worker.submit("active", "active")
        assert entered.wait(1.)
        cancelled = worker.submit("pending", "pending")
        assert cancelled.cancel()
        assert worker.cached("pending") is None
        replacement = worker.submit("pending", "pending")
        assert replacement is not cancelled
        release.set()
        assert active.result(1.) == "active"
        assert replacement.result(1.) == "pending"
    finally:
        release.set()
        worker.close()


# 功能：
#   线程启动失败后不接受后续请求，不把未启动线程当作存活工作器。
# 输入：
#   monkeypatch：线程启动故障注入工具。
# 输出：
#   None：不返回业务数据。
def test_thread_start_failure_is_not_reported_as_an_active_worker(monkeypatch):
    # 功能：
    #   模拟线程创建资源不足。
    # 输入：
    #   _thread：待启动线程。
    # 输出：
    #   None：不返回业务数据。
    def fail_start(_thread):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    worker = LatestInferenceWorker(lambda value: value, name="start-failure-unit")
    with pytest.raises(RuntimeError, match="thread unavailable"):
        worker.submit("input", "data")
    with pytest.raises(RuntimeError, match="WORKER_FAILED"):
        worker.submit("next", "data")
    assert worker.close()["thread_stopped"]


# 功能：
#   专属初始化失败通过首个 Future 暴露，并终止后续接收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_initializer_failure_is_visible_and_cannot_queue_more_work():
    # 功能：
    #   模拟线程亲和性初始化失败。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def broken():
        raise ValueError("affinity failed")

    worker = LatestInferenceWorker(lambda value: value, name="init-unit-inference",
                                    initializer=broken)
    try:
        with pytest.raises(ValueError, match="affinity failed"):
            worker.submit("key", "data").result(1.)
        with pytest.raises(RuntimeError, match="WORKER_FAILED"):
            worker.submit("new", "data")
    finally:
        assert worker.close()["thread_stopped"]


# 功能：
#   关闭超时明确失败，等待项取消但活动推理只能在自身返回后回收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_close_cancels_pending_but_never_claims_a_running_encoder_stopped():
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   在释放事件到达前保持活动推理运行。
    # 输入：
    #   value：来源标识。
    # 输出：
    #   value：释放后返回的来源标识。
    def encode(value):
        entered.set()
        assert release.wait(2.)
        return value

    worker = LatestInferenceWorker(encode, name="close-unit-inference")
    try:
        first = worker.submit("running", "running")
        assert entered.wait(1.)
        pending = worker.submit("pending", "pending")
        with pytest.raises(RuntimeError, match="DID_NOT_STOP"):
            worker.close(timeout_seconds=.001)
        with pytest.raises(CancelledError):
            pending.result()
        with pytest.raises(RuntimeError, match="WORKER_CLOSED"):
            worker.cached("running")
        with pytest.raises(RuntimeError, match="WORKER_CLOSED"):
            worker.submit("next", "next")
        release.set()
        assert first.result(1.) == "running"
    finally:
        release.set()
        assert worker.close()["thread_stopped"]


# 功能：
#   非法关闭预算在创建线程前拒绝，不影响随后合法关闭。
# 输入：
#   timeout：非法等待值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, False, 0, -1., float("nan"), float("inf"), 2.01, "1"])
def test_close_requires_a_finite_bounded_join(timeout):
    worker = LatestInferenceWorker(lambda value: value, name="unused-unit-inference")
    with pytest.raises(ValueError, match="CLOSE_TIMEOUT_INVALID"):
        worker.close(timeout_seconds=timeout)
    assert worker._thread is None
    assert worker.close()["thread_stopped"]


# 功能：
#   编码函数、初始化函数和线程名称必须在首次提交前通过配置检查。
# 输入：
#   options：非法配置字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("options", [{"encode": None}, {"initializer": 1}, {"name": " "}])
def test_invalid_worker_configuration_is_rejected(options):
    settings = {"encode": lambda value: value, "name": "validation-test", **options}
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        LatestInferenceWorker(**settings)


# 功能：
#   编码或完成回调的终止异常不能遗留永远等待的下一项 Future。
# 输入：
#   mode：终止异常发生在编码还是回调。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["encode", "callback"])
def test_fatal_worker_exit_completes_pending_future(mode):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   在受控同步点触发终止类异常，区别于可恢复的普通输入错误。
    # 输入：
    #   value：本次测试输入。
    # 输出：
    #   value：未触发编码故障时的输入。
    def encode(value):
        entered.set()
        assert release.wait(2.)
        if mode == "encode":
            raise SystemExit("injected termination")
        return value

    # 功能：
    #   模拟 Future 用户回调中的终止异常。
    # 输入：
    #   future：已经完成的首个 Future。
    # 输出：
    #   None：不返回业务数据。
    def terminate_callback(future):
        raise SystemExit("injected callback termination")

    worker = LatestInferenceWorker(encode, name="fatal-test")
    try:
        active = worker.submit("active", "active")
        assert entered.wait(1.)
        pending = worker.submit("next", "next")
        if mode == "callback":
            active.add_done_callback(terminate_callback)
        release.set()
        with pytest.raises(SystemExit):
            pending.result(1.)
        with pytest.raises(RuntimeError, match="WORKER_FAILED"):
            worker.submit("last", "last")
    finally:
        release.set()
        assert worker.close()["thread_stopped"]
