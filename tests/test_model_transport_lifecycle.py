"""Deterministic ownership faults, no live credentials or provider requests."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_model_port_attempts import _port

from dronedream_agent_core.contracts import IntentArtifact
from dronedream_agent_core.model_harness import model_port
from dronedream_agent_core.model_harness.model_port import ModelInvocationError


class Client:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


def drain(port):
    with port._transport_changed:
        assert port._transport_changed.wait_for(lambda: port._transport_worker is None, 2.)


def invoke(port):
    return port.call(role="intent_parser", output_type=IntentArtifact,
        instructions="Return intent", input_artifact={"message": "inspect gate"},
        context_id="mission")


def artifact():
    return IntentArtifact(goal="inspect gate", start_entity="office", target_entity="gate",
        return_entity="office", payload_action="none")


def test_constructor_failure_never_republishes_retired_client_and_can_recover(monkeypatch):
    clients, calls = [], []

    def factory(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("secret constructor details must not be returned")
        clients.append(Client())
        return clients[-1]

    monkeypatch.setattr(model_port, "OpenAI", factory)
    port = _port()
    port.reset_transport()
    drain(port)
    assert port._client is None and not port._transport_ready.is_set()
    assert clients[0].closes == 1
    with pytest.raises(TimeoutError, match="^MODEL_TRANSPORT_UNAVAILABLE$"):
        invoke(port)
    port.reset_transport()
    drain(port)
    assert port._client is clients[1] and port._transport_ready.is_set()
    assert all(call["max_retries"] == 0 for call in calls)
    port.close()
    drain(port)
    assert [client.closes for client in clients] == [1, 1]


def test_reset_storm_has_one_worker_and_one_pending_replacement_while_close_blocks(monkeypatch):
    clients = []
    entered, release = threading.Event(), threading.Event()

    class SlowClose(Client):
        def close(self):
            super().close()
            entered.set()
            assert release.wait(2.)

    def factory(**kwargs):
        clients.append(SlowClose() if not clients else Client())
        return clients[-1]

    monkeypatch.setattr(model_port, "OpenAI", factory)
    port = _port()
    port.reset_transport()
    try:
        assert entered.wait(1.)
        worker = port._transport_worker
        assert port._transport_ready.is_set() and len(clients) == 2
        for _ in range(500):
            port.reset_transport()
        assert port._transport_worker is worker and len(clients) == 2
        assert port._transport_reset_requested and not port._transport_ready.is_set()
        assert port._transport_generation == 2  # The other 499 requests are coalesced.
    finally:
        release.set()
    drain(port)
    assert len(clients) == 3 and port._client is clients[-1]
    assert [client.closes for client in clients] == [1, 1, 0]
    port.close()
    drain(port)
    port.close()
    port.reset_transport()
    assert len(clients) == 3 and [client.closes for client in clients] == [1, 1, 1]
    with pytest.raises(TimeoutError, match="UNAVAILABLE"):
        invoke(port)


@pytest.mark.parametrize("stop", [False, True])
def test_constructor_after_revocation_cannot_publish_an_obsolete_client(monkeypatch, stop):
    clients = []
    entered, release = threading.Event(), threading.Event()

    def factory(**kwargs):
        client = Client()
        clients.append(client)
        if len(clients) == 2:
            entered.set()
            assert release.wait(2.)
        return client

    monkeypatch.setattr(model_port, "OpenAI", factory)
    port = _port()
    port.reset_transport()
    try:
        assert entered.wait(1.)
        worker = port._transport_worker
        for _ in range(100):
            port.close() if stop else port.reset_transport()
        assert port._transport_worker is worker and len(clients) == 2
        assert not port._transport_ready.is_set()
    finally:
        release.set()
    drain(port)
    assert clients[0].closes == 1 and clients[1].closes == 1
    if stop:
        assert len(clients) == 2 and port._client is None
        with pytest.raises(TimeoutError, match="UNAVAILABLE"):
            invoke(port)
    else:
        assert len(clients) == 3 and port._client is clients[2]
        assert port._transport_ready.is_set()
        port.close()
        drain(port)


@pytest.mark.parametrize("late_failure", [False, True])
def test_retired_call_cannot_retry_or_overwrite_new_context(monkeypatch, late_failure):
    monkeypatch.setattr(model_port, "OpenAI", lambda **kwargs: Client())
    port = _port()
    entered, release = threading.Event(), threading.Event()
    count = 0

    def response(**kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(2.)
            if late_failure:
                raise RuntimeError("late provider error")
            return artifact(), "obsolete-response", 20, 10
        return artifact(), "current-response", 30, 15

    monkeypatch.setattr(port, "_responses_call", response)
    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(invoke, port)
        try:
            assert entered.wait(1.)
            port.reset_transport()
            drain(port)
            result = invoke(port)
            assert result.record.response_id == "current-response"
        finally:
            release.set()
        with pytest.raises(ModelInvocationError) as failed:
            stale.result(timeout=1.)
    assert failed.value.reason_code == "MODEL_TRANSPORT_RETIRED"
    assert failed.value.attempts_used == 1 and count == 2
    assert port._previous_response_by_context == {"mission": "current-response"}
    port.close()
    drain(port)


def test_close_wakes_call_waiting_for_replacement_without_waiting_for_sdk(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    count = 0

    def factory(**kwargs):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            assert release.wait(2.)
        return Client()

    monkeypatch.setattr(model_port, "OpenAI", factory)
    port = _port()
    port.reset_transport()
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(invoke, port)
        try:
            assert entered.wait(1.)
            port.close()
            with pytest.raises(TimeoutError, match="UNAVAILABLE"):
                waiting.result(timeout=.5)
            assert port._transport_worker is not None
        finally:
            release.set()
    drain(port)


@pytest.mark.parametrize("closing", [False, True])
def test_thread_start_failure_keeps_transport_closed_to_calls_and_recoverable(monkeypatch, closing):
    clients = []

    def factory(**kwargs):
        clients.append(Client())
        return clients[-1]

    class BrokenThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(model_port, "OpenAI", factory)
    port = _port()
    with monkeypatch.context() as patch:
        patch.setattr(model_port.threading, "Thread", BrokenThread)
        with pytest.raises(RuntimeError, match="thread unavailable"):
            port.close() if closing else port.reset_transport()
    with pytest.raises(TimeoutError, match="UNAVAILABLE"):
        invoke(port)
    assert port._client is clients[0] and port._transport_worker is None
    port.close() if closing else port.reset_transport()
    drain(port)
    assert len(clients) == (1 if closing else 2) and clients[0].closes == 1
    port.close()
    drain(port)


def test_retired_generation_is_rejected_before_sdk_request_admission(monkeypatch):
    monkeypatch.setattr(model_port, "OpenAI", lambda **kwargs: Client())
    port = _port()
    generation = port._transport_snapshot()
    port.reset_transport()
    drain(port)
    # Fake client has no with_options: any SDK call would fail the test.
    with pytest.raises(TimeoutError, match="RETIRED"):
        port._request_client(generation)
    port.close()
    drain(port)
