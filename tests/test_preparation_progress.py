from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from dronedream_agent_app.preparation_progress import PreparationProgress
from dronedream_agent_core.model_harness.progress import progress_sink, report_progress, report_model_progress
from dronedream_agent_core.model_harness.progress import report_artifact_progress
from dronedream_agent_core.contracts import SemanticPlan
from dronedream_agent_app.identity import VerifiedIdentity
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore

pytestmark = pytest.mark.usefixtures("isolated_server_credentials")


def test_request_account_isolation_bounded_events_and_late_callbacks():
    mailbox = PreparationProgress()
    key = ("owner", None, None, "thread", "request")
    late = []
    def operation():
        callback = progress_sink.get()
        late.append(callback)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(report_model_progress, callback, "intent_parser", 1, "started").result()
        for index in range(150): report_progress("route", str(index), str(index))
        return "result"
    assert mailbox.run(key, operation) == "result"
    snapshot = mailbox.read(key, 0)
    assert snapshot["state"] == "completed" and len(snapshot["events"]) == 128
    assert mailbox.read(("other", None, None, "thread", "request"), 0)["events"] == []
    cursor = snapshot["events"][-1]["sequence"]
    late[0]("route", "迟到", "late")
    assert mailbox.read(key, cursor)["events"] == []
    assert progress_sink.get() is None


def test_failure_preserves_original_exception_and_cleans_context():
    mailbox = PreparationProgress()
    def failed(): raise ValueError("provider failure")
    with pytest.raises(ValueError, match="provider failure"):
        mailbox.run(("key",), failed)
    assert mailbox.read(("key",), 0)["state"] == "failed"
    assert progress_sink.get() is None


def test_observer_failure_does_not_change_model_business_result():
    def broken(*args): raise RuntimeError("UI observer failed")
    token = progress_sink.set(broken)
    try:
        report_progress("assets", "地图", "map")
        report_model_progress(broken, "intent_parser", 1, "completed")
    finally:
        progress_sink.reset(token)


def test_model_progress_explains_scope_and_reports_measured_time():
    events = []
    sink = lambda *event: events.append(event)
    report_model_progress(sink, "task_decomposer", 1, "started")
    assert len(events) == 2 and len(events[1][1]) > 100
    assert "固定四步模板" in events[1][1]
    report_model_progress(sink, "task_decomposer", 1, "completed", 17093)
    assert "17.1 秒" in events[-1][1]


def test_candidate_summary_excludes_private_or_unneeded_reasoning():
    events = []
    report_artifact_progress(lambda *args: events.append(args), SemanticPlan(ordered_targets=["office", "pickup", "office"], rationale_summary="not-for-progress"))
    assert "office → pickup → office" in events[0][1]
    assert "尚不是飞行许可" in events[0][1]
    assert "not-for-progress" not in str(events)


def test_progress_endpoint_requires_login_and_accepts_safe_cursor(tmp_path):
    store = AppStore(tmp_path)
    identity = VerifiedIdentity("account-test", None, None, "https://example.supabase.co/auth/v1", 2_000_000_000)
    headers = {"Authorization": "Bearer " + "a" * 64, "X-DroneDream-Identity-Token": "test-session"}
    with TestClient(create_app(store=store, token="a" * 64, identity_verifier=lambda _: identity)) as client:
        url = "/v1/threads/thread-test/preparation-progress?request_id=aaaaaaaaaaaaaaaa"
        assert client.get(url).status_code in (401, 403)
        assert client.get(url, headers=headers).json() == {"state": "pending", "events": []}
        assert client.get(url + "&after=-1", headers=headers).status_code == 422
    store.close()
