"""Bounded IPC clock handoff must neither extrapolate nor renew expired samples."""

import threading
from types import SimpleNamespace

import pytest

from dronedream_agent_core.isolated_map_measurement import _read_fresh_clock


# 功能：用确定的单调时钟覆盖暂时迟到、永久过期、明确断钟和未来接收时间，不依赖机器快慢。
# 输入：不同邮箱状态。输出：只接纳真实刷新且未过期的时钟，等待始终有界。
@pytest.mark.parametrize("mode", ["fresh", "refresh", "stale", "missing", "future", "disconnect"])
def test_clock_mailbox_waits_for_real_refresh_only(mode):
    clock = SimpleNamespace(value=1_050_000_000)
    received = SimpleNamespace(value=0.94 if mode != "fresh" else 0.99)
    wall = [1.0]
    sleeps = []
    if mode == "missing":
        clock.value = 0
    if mode == "future":
        received.value = 1.01

    def sleep(seconds):
        sleeps.append(seconds)
        wall[0] += seconds
        if mode == "refresh" and len(sleeps) == 3:
            clock.value = 1_060_000_000
            received.value = wall[0]
        elif mode == "disconnect":
            clock.value = 0

    value = _read_fresh_clock(threading.Lock(), clock, received, now=lambda: wall[0], sleep=sleep)
    assert value == {"fresh": 1_050_000_000, "refresh": 1_060_000_000}.get(mode)
    assert wall[0] <= 1.030000001
    if mode in ("fresh", "missing", "future"):
        assert not sleeps
    if mode == "stale":
        assert clock.value == 1_050_000_000 and received.value == 0.94


# 功能：序列化延迟后才读取当前源钟，防止发出时邮箱已经过期。
# 输入：真实进程和人为延迟的父侧序列化器。输出：顺序被验证，原始源时间不被改写。
def test_dispatch_samples_clock_after_serialization(monkeypatch):
    import time

    from test_isolated_map_measurement import process_fixture, receive_now

    import dronedream_agent_core.isolated_map_measurement as module
    from dronedream_agent_core.live_map_measurement import MapMeasurementPending

    worker, record = process_fixture()
    encode = module.encode_json
    events = []

    def delayed_encode(*args, **kwargs):
        result = encode(*args, **kwargs)
        time.sleep(0.04)
        events.append("encoded")
        return result

    def source_clock():
        events.append("clock")
        return 1_050_000_000

    monkeypatch.setattr(module, "encode_json", delayed_encode)
    try:
        receive_now(record)
        try:
            result = worker.measure(
                record, source_now_ns=source_clock, monotonic_now=time.monotonic
            )
        except MapMeasurementPending:
            # 在严重调度迟到时允许真实新鲜度拒绝；本测试断言的是采钟发生顺序。
            pass
        else:
            assert result["source_timestamp_ns"] == 1_000_000_000
        assert events[:2] == ["encoded", "clock"]
    finally:
        worker.close()
