"""Collector failure and provenance contracts; fixtures are never formal data."""

from collections import Counter
from types import SimpleNamespace

import pytest
from decision_goal_fixture import local_target_fixture
from test_decision_label_evidence import native_fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.decision_label_evidence import verify_command_goal
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment
from scripts import collect_native_decisions as collector


# 功能：单个大附件失败不能导致其它原始记录或失败报告丢失；输出：保留文件且资格阻断。
def test_archive_failure_keeps_other_evidence(tmp_path):
    report = {}
    assert not collector.persist_collection_evidence(
        tmp_path / "bad.json", {"nonfinite": float("inf")}, report
    )
    assert collector.persist_collection_evidence(tmp_path / "good.json", {"original": True}, report)
    assert report["archive_complete"] is False and report["archive_errors"][0]["file"] == "bad.json"
    assert (tmp_path / "good.json").exists() and not (tmp_path / "bad.json").exists()


# 功能：晚到但源时间有序的见证不得被请求截止水位跳过；输入：两次轮询；输出：帧全保留。
def test_witness_cursor_tracks_received_source_time():
    first = SimpleNamespace(sequence=1, observed_at_unix_ms=1000)
    late = SimpleNamespace(sequence=2, observed_at_unix_ms=1020)
    final = SimpleNamespace(sequence=3, observed_at_unix_ms=1080)
    calls = []

    # 功能：模拟第一轮仅收到旧帧、第二轮补到原始新帧；输出：不修改任何时间戳。
    def window(start, end):
        calls.append((start, end))
        return [first] if len(calls) == 1 else [first, late, final]

    env = SimpleNamespace(label_witness_window=window)
    witnesses, errors = {}, Counter()
    cursor = collector.drain_label_witnesses(env, 990, 1050, witnesses, errors)
    assert cursor == 1000
    cursor = collector.drain_label_witnesses(env, cursor, 1100, witnesses, errors)
    assert calls == [(990, 1050), (1000, 1100)]
    assert cursor == 1080 and set(witnesses) == {1, 2, 3} and not errors


# 功能：未收到不等于丢失，不前移游标掩盖延迟；输入：暂无帧异常；输出：原游标和诊断。
def test_witness_not_yet_received_does_not_advance():
    def window(start, end):
        raise ValueError("OUTCOME_WINDOW_NOT_YET_OBSERVED")

    errors = Counter()
    assert collector.drain_label_witnesses(
        SimpleNamespace(label_witness_window=window), 1000, 1100, {}, errors
    ) == 1000
    assert sum(errors.values()) == 1


# 功能：临界净空不引起快慢反复，恢复须连续且不能抹去原始未知状态；输入：合成序列；输出：限速。
@pytest.mark.parametrize(
    "times,cautions,expected",
    [
        ([0, 100], [True, False], "slow_down"),
        ([0, 100, 200, 300], [True, False, False, False], "follow_route"),
        ([0, 100, 200, 300], [True, False, None, False], "slow_down"),
        ([0, 100, 900], [True, False, False], "slow_down"),
        ([0, 100], [False, False], "follow_route"),
    ],
)
def test_caution_release_is_causal_and_speed_bounded(times, cautions, expected):
    frames = [
        SimpleNamespace(observed_at_ms=10000 + t, caution_required=c)
        for t, c in zip(times, cautions, strict=True)
    ]
    state = SimpleNamespace(frame=frames[-1], history=frames[:-1])
    pilot = collector.PilotAction(mode="pilot-control", axes=[0.6, 0.8, 0.0, 0.2])
    proposal = ("follow_route", pilot, {"nominal_speed_limit_mps": 0.4})
    result = collector.stabilize_caution_release(
        state, proposal, SimpleNamespace(horizontal_speed_mps=0.4, vertical_speed_mps=0.2)
    )
    assert result[0] == expected
    if expected == "follow_route":
        assert result is proposal
    else:
        assert collector.math.hypot(*result[1].axes[:2]) * 0.4 == pytest.approx(0.15)
        assert result[1].axes[3] == 0.2
        assert result[1].axes[0] / result[1].axes[1] == pytest.approx(0.75)
    assert pilot.axes == [0.6, 0.8, 0.0, 0.2]
    assert state.frame.caution_required is cautions[-1]


# 功能：紧急/补观测/等待不被慢行滞回覆盖；输入：无需读取的状态和高优先级提案；输出：原样。
@pytest.mark.parametrize("action", ["wait", "request_observation", "replan", "slow_down"])
def test_caution_release_preserves_noncruise_actions(action):
    proposal = (action, object(), {})
    assert collector.stabilize_caution_release(None, proposal, None) is proposal


# 功能：对横穿解除采用有界连续证据，不把一个清空帧、未知帧或断流当作恢复。
# 输入：合成历史时间序列；输出：200毫秒确认后恢复，之前等待且不消除原始观测。
@pytest.mark.parametrize(
    "times,clear,expected",
    [
        ([0, 100], [True, False], "wait"),
        ([0, 100, 200, 300], [True, False, False, False], "follow_route"),
        ([0, 100, 200, 300], [True, False, None, False], "wait"),
        ([0, 100, 900], [True, False, False], "wait"),
        ([0, 100, 200], [False, False, False], "follow_route"),
    ],
)
def test_crossing_release_requires_continuous_clear_frames(times, clear, expected):
    frames = [
        SimpleNamespace(observed_at_ms=10000 + t, crossing_obstacle=c, can_hold_position=True)
        for t, c in zip(times, clear, strict=True)
    ]
    state = SimpleNamespace(frame=frames[-1], history=frames[:-1])
    proposal = ("follow_route", object(), {"reason": "verified"})
    result = collector.stabilize_crossing_release(state, proposal)
    assert result[0] == expected
    if expected == "follow_route":
        assert result is proposal
    else:
        assert result[1].mode == "hold" and not any(result[1].axes)


# 功能：补观测优先于恢复滞回，避免用旧位置强行保持悬停。
# 输入：缺观测提案；输出：原样保留，而非改成等待。
def test_crossing_release_preserves_request_observation():
    state = SimpleNamespace(frame=SimpleNamespace(can_hold_position=True))
    proposal = ("request_observation", object(), {})
    assert collector.stabilize_crossing_release(state, proposal) is proposal


# 功能：等待只能关联当前任务同一路线的后续清空恢复，不能把其他路线或更早结果拼接。
# 输入：合成恢复附件；输出：最近合法匹配或None。
@pytest.mark.parametrize("fault", [None, "route", "time", "unknown"])
def test_wait_continuation_identity(fault):
    row, outcome = native_fixture()
    state = SimpleNamespace(**row["state"])
    row["state"]["frame"].update(observed_at_ms=12500, crossing_obstacle=False)
    if fault == "route":
        row["state"]["route_sha256"] = "f" * 64
    elif fault == "time":
        row["state"]["frame"]["observed_at_ms"] = 11000
    elif fault == "unknown":
        row["state"]["frame"]["crossing_obstacle"] = None
    entry = {"row": row, "outcome": outcome}
    result = collector.find_wait_continuation(state, 12000, [entry])
    assert (result is entry) == (fault is None)


# 功能：长等待按有界窗口归档且去重，不跳过间隔；输入：可控见证端口；输出：连续查询。
def test_label_drain_chunks_waits_and_deduplicates():
    calls, witnesses, errors = [], {}, Counter()
    frames = [SimpleNamespace(sequence=i+1, observed_at_unix_ms=t)
              for i, t in enumerate((100, 1600, 3100, 4100))]

    class Environment:
        def label_witness_window(self, start, end):
            calls.append((start, end))
            return [frame for frame in frames if start <= frame.observed_at_unix_ms <= end]

    assert collector.drain_label_witnesses(Environment(), 100, 4100, witnesses, errors) == 4100
    assert calls == [(100, 1600), (1600, 3100), (3100, 4100)]
    assert witnesses == {f.sequence: f for f in frames} and not errors


# 功能：标签接口失败不传导到控制，缺口仍显式隔离；输入：失败端口；输出：诊断计数。
def test_label_drain_records_failure_without_aborting_control():
    errors = Counter()

    class Environment:
        def label_witness_window(self, start, end):
            raise ValueError("MISSING_WITNESS")

    assert collector.drain_label_witnesses(Environment(), 0, 3100, {}, errors) == 3100
    assert errors["ValueError:MISSING_WITNESS"] == 3
    collector.drain_label_witnesses(None, 0, 31001, {}, errors)
    assert errors["DECISION_WITNESS_POLL_GAP"] == 1


# 功能：只需输入时不读无关回执账本，副本不能污染待执行请求；输出：无I/O且身份仍受校验。
def test_navigation_snapshot_does_not_scan_control_receipts():
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._pending = {"snapshot": {"snapshot_sha256": "a" * 64, "nested": {"x": 1}}}
    env._observation = SimpleNamespace(sample=SimpleNamespace(source_snapshot_sha256="a" * 64))
    env._quiesced = env._closed = False
    snapshot = env.current_navigation_snapshot()  # 故意不提供回执监视器。
    snapshot["nested"]["x"] = 9
    assert env._pending["snapshot"]["nested"]["x"] == 1
    env._pending["snapshot"]["snapshot_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        env.current_navigation_snapshot()


# 功能：短超时后只等待新观测，不重复发送过期动作；输入：可控时钟；输出：恢复的新对象。
def test_transient_wait_preserves_overall_collection_budget(monkeypatch):
    ticks = iter([0.0, 0.0, 1.9, 1.9])
    monkeypatch.setattr(collector.time, "monotonic", lambda: next(ticks))
    calls = []

    class Environment:
        def next_stream_observation(self, *, deadline):
            calls.append(deadline)
            if len(calls) == 1:
                raise TimeoutError("PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT")
            return "fresh-observation"

    counters = {"observation_wait_timeouts": 0}
    assert collector.await_fresh_observation(Environment(), 5.0, counters) == "fresh-observation"
    assert calls == [1.9, 3.8] and counters["observation_wait_timeouts"] == 1


# 功能：总预算已结束不继续调用，非新帧等待类超时不能被吞掉；输出：None或原异常。
def test_wait_budget_and_unrelated_failure(monkeypatch):
    monkeypatch.setattr(collector.time, "monotonic", lambda: 5.0)
    assert collector.await_fresh_observation(None, 5.0, {}) is None

    class Environment:
        def next_stream_observation(self, **kwargs):
            raise TimeoutError("EXECUTOR_CONNECTION_LOST")

    with pytest.raises(TimeoutError, match="CONNECTION_LOST"):
        collector.await_fresh_observation(Environment(), 6.0, {})


# 功能：只在原始输入仍绑定同一任务时接受滚动目标；输出：通过或准确拒绝缺证据。
def test_local_target_requires_source_snapshot():
    state, command, snapshot, goal = local_target_fixture()
    verify_command_goal(command, state, goal, snapshot)
    with pytest.raises(ValueError, match="TARGET_CHANGED"):
        verify_command_goal(command, state, goal)


# 功能：即使重签快照也不能偷换任务、地图、路线或目的地；输入：逐项突变；输出：拒绝。
@pytest.mark.parametrize("fault", ["goal", "route", "map", "position", "hash"])
def test_source_binding_rejects_changed_goal(fault):
    state, command, snapshot, goal = local_target_fixture()
    if fault == "goal":
        snapshot["strategic_context"]["task"]["navigation_goal_id"] = "another"
    elif fault in {"route", "map"}:
        snapshot["known_static_map"][
            "qualified_route_sha256" if fault == "route" else "source_sha256"
        ] = "f" * 64
    else:
        snapshot["goal_position_m"]["x"] += 1
    if fault != "hash":
        snapshot.pop("snapshot_sha256")
        snapshot["snapshot_sha256"] = decision_digest(snapshot)
        raw = command.model_dump(mode="json")
        raw["model_navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
        raw["requested_control_intent"]["navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
        command = RuntimeLocalSafetyCommand.model_validate(raw)
    with pytest.raises(ValueError, match="SOURCE_"):
        verify_command_goal(command, state, goal, snapshot)
