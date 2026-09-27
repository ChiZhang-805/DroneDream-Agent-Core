"""Checks for fresh parcel readback, including lost one-shot state events."""

import json
import subprocess

import pytest

from dronedream_agent_core import gazebo_adapter as adapter
from dronedream_agent_core import payload_state_query as query

TOKEN = "a" * 32


# 功能：核验明确状态与请求、迭代号绑定；输入：两种状态；输出：解析结果断言。
@pytest.mark.parametrize("state,detached", [("attached", False), ("detached", True)])
def test_explicit_snapshot(state, detached):
    result = query.parse_attachment_snapshot(f'data: "{TOKEN}|{state}|123"', TOKEN)
    assert result["detached"] is detached
    assert result["iteration"] == 123


# 功能：未知、旧请求、畸形回执不能通过；输入：坏响应；输出：拒绝断言。
@pytest.mark.parametrize(
    "raw",
    [
        "",
        'data: "detached"',
        "data: true",
        f'data: "{TOKEN}|unknown|12"',
        f'data: "{TOKEN}|detached|0"',
        f'data: "{TOKEN}|detached|-1"',
        f'data: "{TOKEN}|detached|18446744073709551616"',
        f'data: "{"b" * 32}|detached|12"',
    ],
)
def test_unknown_is_not_detached(raw):
    with pytest.raises(RuntimeError):
        query.parse_attachment_snapshot(raw, TOKEN)


# 功能：两次独立查询必须在真实时步推进后才能确认；输入：含重复和挂载态的快照；输出：有限重试证据。
def test_detach_requires_advancing_independent_snapshots(monkeypatch):
    samples = iter([(False, 10), (True, 11), (True, 11), (False, 12), (True, 13), (True, 14)])

    def read(*args, **kwargs):
        detached, iteration = next(samples)
        return dict(detached=detached, iteration=iteration, source="gazebo-ecm-joint-query")

    monkeypatch.setattr(query, "query_attachment_state", read)
    monkeypatch.setattr(adapter, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    result = adapter._detach_payload_with_snapshot(
        "gz",
        detach_topic="/detach",
        output_topic="/state",
        state_service="/world/w/model/p/attachment_state",
        env={},
        timeout=3,
    )
    assert result["confirmed"] is True
    assert len(result["attempts"]) == 6


# 功能：连续未知不能转为成功；输入：服务超时；输出：带失败证据的拒绝。
def test_detach_unknown_fails_closed(monkeypatch):
    def read(*args, **kwargs):
        raise RuntimeError("PAYLOAD_STATE_QUERY_UNCONFIRMED")

    monkeypatch.setattr(query, "query_attachment_state", read)
    monkeypatch.setattr(adapter, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    with pytest.raises(adapter.SimulationRuntimeError, match="UNCONFIRMED"):
        adapter._detach_payload_with_snapshot(
            "gz",
            detach_topic="/detach",
            output_topic="/state",
            state_service="/world/w/model/p/attachment_state",
            env={},
            timeout=0.05,
        )


# 功能：验证生产查询实际使用当前随机标识，且命令成功但空响应仍失败。
# 输入：模拟 CLI 响应；输出：请求参数与结果断言。
def test_query_binds_request_not_cached_event(monkeypatch):
    def run(argv, **kwargs):
        token = json.loads(argv[-1].split(":", 1)[1])
        assert argv[1] == "service"
        return subprocess.CompletedProcess(argv, 0, f'data: "{token}|detached|99"', "")

    monkeypatch.setattr(query.subprocess, "run", run)
    result = query.query_attachment_state("gz", service="/world/w/model/p/attachment_state", env={})
    assert result["iteration"] == 99 and result["source"] == "gazebo-ecm-joint-query"


# 功能：证明相机冷启动无回读时先等待物理就绪，不先耗尽解除命令重试。
# 输入：先 unknown、随后 attached 的服务；输出：初始化证据与最终分离。
def test_preflight_waits_for_physics_before_command_budget(monkeypatch):
    calls = []
    clock = [0.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def read(*args, **kwargs):
        calls.append("query")
        if 2 <= len(calls) <= 4:
            raise RuntimeError("PAYLOAD_STATE_QUERY_UNCONFIRMED")
        return dict(detached=False, iteration=100 + len(calls), source="gazebo-ecm-joint-query")

    def detach(*args, **kwargs):
        assert len(calls) >= 15
        assert kwargs["timeout"] == 20
        return dict(confirmed=True, detached=True)

    monkeypatch.setattr(query, "query_attachment_state", read)
    monkeypatch.setattr(adapter, "_detach_payload_with_snapshot", detach)
    monkeypatch.setattr(
        adapter, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "/detach\n/state", "")
    )
    result = adapter._detach_payload_before_flight(
        "gz",
        detach_topic="/detach",
        output_topic="/state",
        state_service="/world/w/model/p/attachment_state",
        env={},
        readiness_timeout=3,
        timeout=20,
    )
    assert result["readiness_attempts"] >= 15
    assert result["initialization_snapshot"]["detached"] is False


# 功能：拒绝服务可用但物理迭代不推进的假就绪；输入：重复快照；输出：不发送分离命令。
def test_repeated_physics_snapshot_is_not_ready(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(query, "query_attachment_state", lambda *a, **k: dict(detached=True, iteration=42))
    monkeypatch.setattr(adapter, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "/detach\n/state", ""))
    monkeypatch.setattr(adapter, "_detach_payload_with_snapshot", lambda *a, **k: pytest.fail("not ready"))
    with pytest.raises(adapter.SimulationRuntimeError, match="PAYLOAD_PHYSICS_NOT_READY"):
        adapter._detach_payload_before_flight("gz", detach_topic="/detach", output_topic="/state",
            state_service="/world/w/model/p/attachment_state", env={}, readiness_timeout=2)


# 功能：物理线程完全停滞时拒绝启动，不能把扩大启动等待时间变成绕过确认。
# 输入：持续无回读；输出：明确的就绪失败，不执行卸载流程。
def test_stalled_physics_fails_without_detach_success(monkeypatch):
    def read(*args, **kwargs):
        raise RuntimeError("PAYLOAD_STATE_QUERY_UNCONFIRMED")

    monkeypatch.setattr(query, "query_attachment_state", read)
    monkeypatch.setattr(
        adapter, "_run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "/detach\n/state", "")
    )
    with pytest.raises(adapter.SimulationRuntimeError, match="PAYLOAD_PHYSICS_NOT_READY"):
        adapter._detach_payload_before_flight(
            "gz",
            detach_topic="/detach",
            output_topic="/state",
            state_service="/world/w/model/p/attachment_state",
            env={},
            readiness_timeout=0.01,
        )
