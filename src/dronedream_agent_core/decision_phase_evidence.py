"""Offline lifecycle audit: a closed run alone cannot prove its earlier windows flew."""

from .runtime_phase import ENDING_PHASES


# 功能：使用执行器未合并的阶段转换判断窗口是否提前结束，未知不能默认成没有降落。
# 输入：已排空执行器快照回执和UNIX毫秒窗口；输出：提前终止事实及覆盖该窗口的原事件。
def verify_window_phases(summary, start_ms, end_ms):
    if (
        summary.get("complete") is not True
        or summary.get("phase_history_complete") is not True
        or type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms >= end_ms
    ):
        raise ValueError("DECISION_PHASE_HISTORY_INCOMPLETE")
    events = summary.get("phase_events")
    if type(events) is not list or not 2 <= len(events) <= 8192:
        raise ValueError("DECISION_PHASE_HISTORY_MISSING")
    previous = None
    for i, event in enumerate(events):
        if (
            type(event) is not dict
            or event.get("sequence") != i + 1
            or type(event.get("at_unix_ms")) is not int
            or previous is not None
            and event["at_unix_ms"] < previous
        ):
            raise ValueError("DECISION_PHASE_HISTORY_ORDER_INVALID")
        previous = event["at_unix_ms"]
    before = [e for e in events if e["at_unix_ms"] <= start_ms]
    after = [e for e in events if e["at_unix_ms"] > end_ms]
    if not before or not after:
        raise ValueError("DECISION_PHASE_HISTORY_COVERAGE_MISSING")
    covered = [before[-1]] + [e for e in events if start_ms < e["at_unix_ms"] <= end_ms]
    active = {
        "TRACK",
        "WAYPOINT_SETTLE",
        "CHECKPOINT",
        "ACTION",
        "HOLDING",
        "PAUSED",
        "TRACKING_RECOVERY",
    }
    if any(e.get("executor_phase") not in active | ENDING_PHASES for e in covered):
        raise ValueError("DECISION_PHASE_NOT_IN_FLIGHT_WINDOW")
    return {
        "premature_landing": any(e["executor_phase"] in ENDING_PHASES for e in covered),
        "events": [dict(e) for e in covered + [after[0]]],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "physical_landing_proof": False,
    }
