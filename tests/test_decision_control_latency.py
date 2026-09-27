"""Timing diagnostics are measurements, never retroactive training labels."""

import json

import pytest

from dronedream_agent_core.decision_state_adapter import decision_digest
from scripts.audit_decision_control_latency import audit, statistics


# 功能：验证真实毫秒分位数及负余量不被抹平；输入：固定测量；输出：严格统计断言。
def test_statistics_preserve_expiry_and_reject_nonfinite():
    assert statistics([]) == {"count": 0}
    assert statistics([-3, 0, 9])["minimum"] == -3
    assert statistics(list(range(1, 101)))["p95"] == 95
    for value in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="MEASUREMENT_INVALID"):
            statistics([value])


# 功能：用微型固定账本检验发布与实际应用通过完整命令摘要相连；不会给合成数据资格。
# 输入：隔离临时目录；输出：调用去重、可选文件缺失和分组耗时的断言。
def test_audit_joins_actual_calls_and_does_not_invent_missing_diagnostics(tmp_path):
    simulation = tmp_path / "flight/simulation"
    (simulation / "runtime-state").mkdir(parents=True)
    command = {
        "model_navigation_authorized": True,
        "model_call_id": "call-1",
        "valid_until_unix_ms": 1200,
    }
    rows = [
        {
            "recorded_at_unix_ms": 1100,
            "command": command,
            "pipeline_phase_ms": {"encoding": 12},
            "prioritized_ready_control": True,
        }
    ]
    actual = {
        "command_sha256": decision_digest(command),
        "model_authorized": True,
        "control_source": "model",
        "accepted_at_unix_ms": 1150,
        "command_generated_at_unix_ms": 1090,
        "command_valid_until_unix_ms": 1200,
    }
    for name, values in (
        ("depth-local-safety-history.jsonl", rows),
        ("runtime-state/control-applications.jsonl", [actual, actual]),
        ("runtime-state/executor-brake-applications.jsonl", []),
    ):
        (simulation / name).write_text("".join(json.dumps(v) + "\n" for v in values), "utf-8")
    result = audit(tmp_path)
    assert result["call_join"]["published"] == result["call_join"]["accepted"] == 1
    assert result["accepted_applications"] == {"model": 2}
    assert result["interface_scope"] is None
    assert result["model_scheduling_ms"] == {}
    assert result["pipeline_phase_ms"]["ready-control"]["encoding"]["p95"] == 12
    assert not result["qualification_granted"]
    assert result["formal_training_additions"] == 0
    assert result["call_join"]["unmatched_model_applications"] == 0
    actual["command_sha256"] = "f" * 64
    (simulation / "runtime-state/control-applications.jsonl").write_text(
        json.dumps(actual) + "\n", "utf-8"
    )
    unmatched = audit(tmp_path)["call_join"]
    assert unmatched["accepted"] == 0
    assert unmatched["unmatched_model_applications"] == 1
