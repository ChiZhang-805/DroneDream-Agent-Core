"""Lifecycle evidence is never inferred from a successful collector shutdown."""

import pytest

from dronedream_agent_core.decision_phase_evidence import verify_window_phases
from dronedream_agent_core.executor_snapshots import ExecutorSnapshots


# 功能：构造明确标记的合成阶段转换；输入：无；输出：不代表真实飞行的单元夹具。
def fixture():
    return dict(
        complete=True,
        phase_history_complete=True,
        phase_events=[
            dict(sequence=i + 1, at_unix_ms=t, phase=p, executor_phase=p)
            for i, (t, p) in enumerate(((1000, "TRACK"), (4000, "LANDING"), (5000, "COMPLETE")))
        ],
    )


# 功能：窗口内降落与采集结束后的正常降落分开；输入：两种时间窗；输出：不同事实。
def test_landing_after_window_is_not_early_but_landing_inside_is():
    assert not verify_window_phases(fixture(), 1100, 3500)["premature_landing"]
    assert verify_window_phases(fixture(), 1100, 4100)["premature_landing"]


# 功能：未知、倒序、溢出和缺少前后覆盖不得默认通过；输入：破坏夹具；输出：拒绝。
@pytest.mark.parametrize("fault", ["missing", "overflow", "order", "unknown", "start", "end"])
def test_missing_phase_proof_never_becomes_no_premature_landing(fault):
    summary = fixture()
    start, end = 1100, 3500
    if fault == "missing":
        summary.pop("phase_events")
    elif fault == "overflow":
        summary["phase_history_complete"] = False
    elif fault == "order":
        summary["phase_events"][1]["at_unix_ms"] = 500
    elif fault == "unknown":
        summary["phase_events"][0]["executor_phase"] = "UNKNOWN"
    elif fault == "start":
        start = 500
    else:
        end = 6000
    with pytest.raises(ValueError):
        verify_window_phases(summary, start, end)


# 功能：内存记录必须在诊断快照合并前保留转换且不与调用方共享可变引用。
# 输入：快速发布三个阶段；输出：所有转换保留，重复相同阶段不膨胀历史。
def test_executor_records_all_transitions_before_file_coalescing(tmp_path):
    def writer(*args, **kwargs):
        pass

    snapshots = ExecutorSnapshots(tmp_path, writer=writer)
    payload = {"phase": "TRACK"}
    snapshots.submit(snapshots.phase_path, payload)
    snapshots.submit(snapshots.phase_path, payload)
    payload["phase"] = "LANDING"
    snapshots.submit(snapshots.phase_path, payload)
    snapshots.submit(snapshots.phase_path, {"phase": "COMPLETE"})
    result = snapshots.close()
    assert result["complete"] and result["phase_history_complete"]
    assert [e["phase"] for e in result["phase_events"]] == ["TRACK", "LANDING", "COMPLETE"]


# 功能：超过容量时只让证据失去完整资格，不能让日志停止控制线程。
# 输入：已满的合成阶段历史；输出：继续接收快照且完整性明确为假。
def test_phase_history_overflow_is_evidence_failure_not_flight_exception(tmp_path):
    snapshots = ExecutorSnapshots(tmp_path, writer=lambda *args, **kwargs: None)
    with snapshots._condition:
        snapshots._phase_events = [dict(phase="TRACK", executor_phase="TRACK")] * 8192
    assert snapshots.submit(snapshots.phase_path, {"phase": "LANDING"})
    assert not snapshots.close()["phase_history_complete"]
