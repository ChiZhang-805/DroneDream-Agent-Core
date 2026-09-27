"""Actual feedback is additional admission evidence, never a substitute for safety."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_bounded_hybrid_control import arm, tick

from dronedream_agent_core.executor_snapshots import ExecutorSnapshots
from dronedream_agent_core.hybrid_control import BoundedHybridArbiter
from dronedream_agent_core.runtime_phase_channel import (
    RuntimePhaseBroadcaster,
    RuntimePhaseReceiver,
)


# 功能：固定真实执行形式的三条不同回执；输入：时间偏移；输出：只供测试的数据。
def receipts(offset=0):
    return [(str(i), "office", 1000 + offset + t) for i, t in enumerate((10, 160, 320))]


# 功能：仅收到推理不能启动接管，实际回执到达后才可以；输入：原始回执；输出：有界许可。
def test_bridge_waits_for_actual_transport_not_proposals():
    arbiter = BoundedHybridArbiter(require_execution_feedback=True)
    arm(arbiter)
    assert tick(arbiter, 0.4) is None
    assert arbiter.snapshot()["reason"] == "actual-model-execution-not-established"
    arbiter.observe_execution(receipts())
    assert tick(arbiter, 0.42).maximum_speed_mps == 0.25


# 功能：过期、异目标和未来回执不能许可；输入：错配三元组；输出：不开始桥接。
@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("x", "office", 0)],
        receipts(1000),
        [(str(i), "other", t) for i, (_, _, t) in enumerate(receipts())],
    ],
)
def test_invalid_context_cannot_bootstrap(rows):
    arbiter = BoundedHybridArbiter(require_execution_feedback=True)
    arm(arbiter)
    arbiter.observe_execution(rows)
    assert tick(arbiter, 0.4) is None


# 功能：硬故障之后不能复用之前的实际执行；输入：新推理配旧执行；输出：仍拒绝。
def test_fault_requires_execution_after_recovery_epoch():
    arbiter = BoundedHybridArbiter(require_execution_feedback=True)
    arm(arbiter)
    arbiter.observe_execution(receipts())
    arbiter.invalidate("test-fault")
    arm(arbiter, 0.5)
    assert tick(arbiter, 0.9) is None
    arbiter.observe_execution(receipts(500))
    assert tick(arbiter, 0.91)


# 功能：同一执行回执不能反复启动新桥接；输入：第二次推理；输出：直到新执行才接管。
def test_bridge_rearm_requires_new_executions():
    arbiter = BoundedHybridArbiter(require_execution_feedback=True)
    arm(arbiter)
    arbiter.observe_execution(receipts())
    assert tick(arbiter, 0.4)
    arm(arbiter, 0.5)
    assert tick(arbiter, 0.9) is None
    arbiter.observe_execution(receipts(500))
    assert tick(arbiter, 0.91).episode == 2


# 功能：同一桥接租约不因中间少量回复而改变速度参数；输入：高速度或舍入扰动；输出：不续租。
@pytest.mark.parametrize("speed", [.4, .15 - 1e-17])
def test_active_bridge_speed_is_immutable(speed):
    arbiter = BoundedHybridArbiter()
    for index, stamp in enumerate((0., .15, .31)):
        tick(arbiter, stamp, model_live=True, model_call_id=str(index), model_speed_mps=.15)
    first = tick(arbiter, .4)
    tick(arbiter, .5, model_live=True, model_call_id="new", model_speed_mps=speed)
    assert tick(arbiter, .6) == first


# 功能：真正降速撤销旧桥接，不让旧较快速度继续生效；输入：新更低上限；输出：等待重新武装。
def test_new_lower_speed_revokes_active_bridge():
    arbiter = BoundedHybridArbiter()
    arm(arbiter)
    assert tick(arbiter, .4)
    tick(arbiter, .5, model_live=True, model_call_id="slow", model_speed_mps=.1)
    assert tick(arbiter, .6) is None


# 功能：坏结构、重复、乱序和伪布尔时钟清空缓存；输入：非法回执；输出：明确拒绝。
@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [("1", "office", True)],
        receipts() * 2,
        [("1", "office", 1), ("1", "office", 2)],
        [("1", "office", 2), ("2", "office", 1)],
        [("", "office", 1)],
    ],
)
def test_bad_feedback_clears_prior_evidence(rows):
    arbiter = BoundedHybridArbiter(require_execution_feedback=True)
    arm(arbiter)
    arbiter.observe_execution(receipts())
    with pytest.raises(ValueError, match="FEEDBACK_INVALID"):
        arbiter.observe_execution(rows)
    assert tick(arbiter, 0.4) is None


# 功能：模拟已鉴权端点内存，避免测试真实端口；输入：发送毫秒；输出：无运动权接收器。
def receiver(stamp=1000):
    obj = object.__new__(RuntimePhaseReceiver)
    obj._lock, obj._stop, obj._failure = threading.Lock(), threading.Event(), None
    obj._latest = dict(updated_at_unix_ms=stamp, model_applications=[list(r) for r in receipts()])
    return obj


# 功能：新心跳不重写旧回执时钟，返回副本不污染原缓存；输入：当前时钟；输出：原始执行时刻。
def test_receiver_never_redates_or_lends_receipts():
    obj = receiver()
    with patch("dronedream_agent_core.runtime_phase_channel.time.time", return_value=1.1):
        rows = obj.read_model_applications()
        assert rows[0][2] == 1010
        rows[0][2] = 1099
        assert obj.read_model_applications()[0][2] == 1010
    with patch("dronedream_agent_core.runtime_phase_channel.time.time", return_value=2.0):
        assert obj.read_model_applications() == []
    obj._stop.set()
    assert obj.read_model_applications() == []


# 功能：消费者拒绝接管时给出实际历史不足，不再只有泛化过期；输入：无执行历史；输出：原因。
def test_executor_classifies_bridge_rejection():
    from test_bounded_hybrid_control import bridge_command, executor_args
    from test_runtime_commands import _load_executor

    args = executor_args()
    args._hybrid_model_receipts = ()
    assert not _load_executor()._hybrid_transport_admissible(
        args,
        SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        bridge_command(),
        now=1.0,
        now_unix_ms=1000,
    )
    assert args._hybrid_rejection_reason == "hybrid-actual-execution-history-insufficient"


# 功能：实际本地广播贯通到接收器且保留接收原钟；输入：隔离端点；输出：相同三条回执。
def test_actual_feedback_channel_roundtrip(tmp_path):
    snapshots = ExecutorSnapshots(tmp_path)
    channel = RuntimePhaseReceiver(tmp_path / "feedback.json")
    broadcaster = RuntimePhaseBroadcaster([channel.path], snapshots, model_applications=receipts)
    try:
        snapshots.submit(snapshots.phase_path, {"phase": "TRACK"})
        broadcaster.start()
        deadline = time.monotonic() + 1.
        found = []
        while time.monotonic() < deadline:
            found = channel.read_model_applications()
            if found:
                break
            time.sleep(.005)
        assert found == [list(r) for r in receipts()]
    finally:
        asyncio.run(broadcaster.close())
        snapshots.close()
        channel.close()
