"""Real spawned-process tests of timing/identity; no model or flight client."""

import json
import time
from pathlib import Path

import pytest

from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2
from dronedream_agent_core.local_stage_decision_port import LocalStageDecisionPort
from scripts.prepare_decision_contract_smoke import synthetic_state


# 功能：测试子进程按真实队列回复；输入：队列和延迟配置；输出：测试概率或非法分布。
def fake_worker(requests, results, mode):
    if mode == "startup-delay":
        time.sleep(2)
    results.put({"kind": "ready", "smoke_only": True})
    try:
        while True:
            request_id, _ = requests.get()
            if mode == "delay":
                time.sleep(.15)
            values = dict.fromkeys(ACTIONS, .2)
            if mode == "invalid":
                values["wait"] = float("nan")
            results.put({"kind": "advice", "request_id": request_id,
                             "raw": values, "calibrated": values})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        requests.close()
        results.close()


# 功能：仅测试使用的有界轮询等待；输入：通道和条件；输出：最后建议或测试超时。
def wait_for(port, condition, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = port.poll()
        if condition(result):
            return result
        time.sleep(.005)
    raise AssertionError(port.last_reason)


# 功能：创建严格 v2 测试状态；输入：序号和可选目标；输出：因果接口实例。
def state(index=1, goal="goal-1"):
    value = synthetic_state(1, "follow_route")
    value["sequence"] = index
    value["goal_id"] = goal
    return DecisionStateV2.model_validate_json(json.dumps(value))


# 功能：真实进程初始化、建议和幂等回收；输入：无；输出：建议无控制权限。
def test_advice_and_cleanup():
    port = LocalStageDecisionPort(Path("fast"), worker=fake_worker)
    try:
        wait_for(port, lambda _: port.ready)
        port.submit(state())
        result = wait_for(port, lambda result: result is not None)
        assert result["execution_authority"] is False
        assert result["sequence"] == 1
        assert result["smoke_only"] is True
        assert result["goal_id"] == "goal-1"
        assert result["clock_domain"] == "synthetic-only"
        assert result["input_observed_at_ms"] == 10000
        assert result["expires_at_host_monotonic_s"] > time.monotonic()
        assert 0 <= result["valid_for_ms"] <= 750
        with pytest.raises(ValueError, match="SEQUENCE"):
            port.submit(state())
    finally:
        port.close()
        port.close()
    assert not port.process.is_alive()


# 功能：慢结果不能续期或沿旧目标执行；输入：过期/目标改变模式；输出：无有效建议。
@pytest.mark.parametrize("mode", ["expired", "new-goal", "invalid"])
def test_stale_or_invalid_result(mode):
    port = LocalStageDecisionPort(Path("invalid" if mode == "invalid" else "delay"),
                                 ttl_ms=50 if mode == "expired" else 750, worker=fake_worker)
    try:
        wait_for(port, lambda _: port.ready)
        port.submit(state())
        port.poll()
        if mode == "new-goal":
            port.submit(state(2, "other-goal"))
        expected = {"expired": "advice-expired", "new-goal": "goal-or-contract-changed",
                    "invalid": "model-channel-invalid"}[mode]
        result = wait_for(port, lambda _: port.last_reason == expected)
        assert result is None
    finally:
        port.close()


# 功能：加载时间超限时 poll 不等待 join；输入：无；输出：快速撤销并可正常清理。
def test_poll_is_nonblocking_on_timeout():
    port = LocalStageDecisionPort(Path("startup-delay"), startup_timeout_s=.00001,
                                 worker=fake_worker)
    try:
        port.started -= 1
        tick = time.monotonic()
        port.poll()
        assert time.monotonic() - tick < .1
        assert port.closed
    finally:
        port.close()


# 功能：旧状态重新提交不能得到新的750毫秒；输入：UNIX源钟旧帧或未来帧；输出：不排队。
@pytest.mark.parametrize("offset", [-1000, 1000])
def test_input_source_time_is_not_renewed(offset):
    port = LocalStageDecisionPort(Path("fast"), worker=fake_worker)
    try:
        value = state().model_dump(mode="json")
        old = value["frame"]["observed_at_ms"]
        stamp = int(time.time() * 1000) + offset
        value["clock_domain"] = "unix-ms"
        value["frame"]["observed_at_ms"] = stamp
        for key in ("pose_source", "geometry_source", "route_source"):
            if value["frame"][key]:
                value["frame"][key]["observed_at_ms"] += stamp - old
        port.submit(DecisionStateV2.model_validate_json(json.dumps(value)))
        assert port.pending is None
        assert port.last_reason == "input-frame-expired-or-future"
    finally:
        port.close()


# 功能：同目标出现新风险或A→B→A切换不能让旧建议复活；输入：真实队列；输出：旧建议拒收。
@pytest.mark.parametrize("change", ["risk", "aba"])
def test_changed_context_invalidates_inflight_advice(change):
    port = LocalStageDecisionPort(Path("delay"), worker=fake_worker)
    try:
        wait_for(port, lambda _: port.ready)
        port.submit(state())
        port.poll()
        if change == "aba":
            port.submit(state(2, "goal-B"))
            port.submit(state(3))
        else:
            value = state(2).model_dump(mode="json")
            value["frame"]["crossing_obstacle"] = True
            port.submit(DecisionStateV2.model_validate_json(json.dumps(value)))
        result = wait_for(port, lambda _: port.last_reason == "decision-context-superseded")
        assert result is None
    finally:
        port.close()
