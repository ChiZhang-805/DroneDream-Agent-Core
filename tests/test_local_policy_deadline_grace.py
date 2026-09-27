"""Scheduling jitter may consume existing source budget, never renew it."""

from types import SimpleNamespace

import pytest
from clock_fixtures import isolate_monotonic

import dronedream_agent_core.local_policy_port as policy


# 功能：以固定时钟验证宽限只消费剩余来源预算；输入：隔离端口；输出：不改包资格的断言。
def test_deadline_grace_preserves_source_expiry_and_fifty_ms_cap(monkeypatch):
    port = object.__new__(policy.LocalPolicyPort)
    port.package = SimpleNamespace(manifest=SimpleNamespace(maximum_inference_latency_ms=20.))
    port.scheduling_jitter_grace_ms = 0.
    clock = [100.025]
    isolate_monotonic(monkeypatch, policy, lambda: clock[0])
    with pytest.raises(TimeoutError, match="INPUT_PREPARATION"):
        port._check_call_budget(100.)
    assert port._check_call_budget(100., control_deadline_monotonic=100.04) == pytest.approx(25.)
    clock[0] = 100.04
    with pytest.raises(TimeoutError, match="CONTROL_DEADLINE_EXPIRED"):
        port._check_call_budget(100., control_deadline_monotonic=100.04)
    clock[0] = 100.071
    with pytest.raises(TimeoutError, match="INPUT_PREPARATION"):
        port._check_call_budget(100., control_deadline_monotonic=100.2)
    assert port.package.manifest.maximum_inference_latency_ms == 20.
    assert port.scheduling_jitter_grace_ms == 0.


# 功能：不足标称推理时间时仍服从更早的真实来源期限；输入：短余量；输出：拒绝过期调用。
def test_earlier_control_deadline_is_stricter_than_package_latency(monkeypatch):
    port = object.__new__(policy.LocalPolicyPort)
    port.package = SimpleNamespace(manifest=SimpleNamespace(maximum_inference_latency_ms=20.))
    port.scheduling_jitter_grace_ms = 50.
    isolate_monotonic(monkeypatch, policy, lambda: 100.012)
    with pytest.raises(TimeoutError, match="CONTROL_DEADLINE_EXPIRED"):
        port._check_call_budget(100., control_deadline_monotonic=100.01)
