from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dronedream_agent_app.asset_interpretation import (
    AssetInterpretationService,
    AssetUnderstanding,
    interpret_mission_assets,
)
from dronedream_agent_app.custom_models import ModelConnection
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.contracts import ModelCallRecord

pytestmark = pytest.mark.usefixtures("isolated_server_credentials")


# 功能：
#   构造明确的测试模型端口，记录输入并返回带实际测试调用次数的结构化回包。
# 输入：
#   无。
# 输出：
#   port：仅用于单元测试的替身。
class FakePort:
    def __init__(self):
        self.calls = []
        self.invalid = False

    def call(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_artifact"]["required_source_ids"]
        artifact = AssetUnderstanding(
            summary="解析测试",
            items=[
                {"source_id": "invented" if self.invalid else item, "explanation": "源条目的用途"}
                for item in ids
            ],
            limitations=["实时状态需要另行读取"],
        )
        return SimpleNamespace(
            artifact=artifact,
            record=ModelCallRecord(
                call_id="model-" + "a" * 24,
                role="context_summarizer",
                attempt=1,
                input_sha256="a" * 64,
                output_sha256="b" * 64,
                output_schema="AssetUnderstanding",
                provider="kimi",
                model="kimi-k2.6",
                input_tokens=10,
                output_tokens=5,
                latency_ms=1,
                created_at=datetime.now(UTC),
            ),
        )

    def close(self):
        self.closed = True


@pytest.fixture
def setup(tmp_path):
    store = AppStore(tmp_path)
    service = AssetInterpretationService(store)
    connection = ModelConnection(
        selection_id="kimi-k2.6",
        provider="kimi",
        model_id="kimi-k2.6",
        api_key="test-secret-not-persisted",
        base_url="https://example.supabase.co/functions/v1/model-gateway",
        api_style="chat-completions",
        capability_id="model.kimi",
        source="default",
    )
    source = {
        "asset_id": "map-test",
        "content_sha256": "a" * 64,
        "kind": "map",
        "facts": {"nodes": [{"node_id": "pickup", "semantic": "pickup"}]},
        "required_source_ids": ["pickup"],
    }
    options = {
        "scope": {"owner_account_id": "account-test", "source_edition": "autonomy"},
        "source": source,
        "connection": connection,
        "locale": "zh-CN",
        "port": FakePort(),
    }
    yield service, options
    store.close()


def test_cache_reuses_model_result_without_new_requests_and_survives_restart(setup):
    service, options = setup
    first = service.interpret(**options)
    second = AssetInterpretationService(service.store).interpret(**options)
    assert not first["cached"] and second["cached"]
    assert len(first["model_calls"]) == 1 and second["model_calls"] == []
    assert len(options["port"].calls) == 1
    assert options["port"].calls[0]["maximum_physical_attempts"] == 1
    with sqlite3.connect(service.path) as db:
        raw = db.execute("SELECT record_json FROM interpretations").fetchone()[0]
    assert options["connection"].api_key not in raw


@pytest.mark.parametrize(
    "change", ["owner", "edition", "hash", "facts", "model", "endpoint", "locale"]
)
def test_cache_invalidates_for_identity_asset_model_and_prompt_inputs(setup, change):
    service, options = setup
    first = service.interpret(**options)
    if change == "owner":
        options["scope"] = {**options["scope"], "owner_account_id": "another-account"}
    elif change == "edition":
        options["scope"] = {**options["scope"], "source_edition": "sim"}
    elif change == "hash":
        options["source"]["content_sha256"] = "c" * 64
    elif change == "facts":
        options["source"]["facts"]["updated"] = True
    elif change == "model":
        options["connection"] = replace(options["connection"], model_id="other-model")
    elif change == "endpoint":
        options["connection"] = replace(
            options["connection"], base_url="https://changed.example/v1"
        )
    else:
        options["locale"] = "en-US"
    second = service.interpret(**options)
    if change == "hash":
        assert second["cached"] and second["reuse_mode"] == "unchanged_dependencies"
        assert len(options["port"].calls) == 1
        return
    assert not second["cached"] and first["cache_key"] != second["cache_key"]
    assert len(options["port"].calls) == 2


def test_unknown_source_rejected_and_failed_reinterpretation_preserves_good_cache(setup):
    service, options = setup
    first = service.interpret(**options)
    options["port"].invalid = True
    with pytest.raises(ValueError, match="REFERENCES_INVALID"):
        service.interpret(**options, force=True)
    assert service.interpret(**options)["understanding"] == first["understanding"]


def test_cache_corruption_is_not_model_understanding(setup):
    service, options = setup
    result = service.interpret(**options)
    with sqlite3.connect(service.path) as db:
        record = json.loads(db.execute("SELECT record_json FROM interpretations").fetchone()[0])
        record["understanding"]["summary"] = "corrupted"
        db.execute(
            "UPDATE interpretations SET record_json=? WHERE cache_key=?",
            (json.dumps(record), result["cache_key"]),
        )
    repaired = service.interpret(**options)
    assert not repaired["cached"]
    assert repaired["understanding"]["summary"] != "corrupted"
    assert len(options["port"].calls) == 2


def test_full_cache_fails_before_spending_model_tokens(setup):
    service, options = setup
    with sqlite3.connect(service.path) as db:
        db.executemany(
            "INSERT INTO interpretations VALUES (?, ?)", [(str(i), "{}") for i in range(1024)]
        )
    with pytest.raises(ValueError, match="CACHE_FULL"):
        service.interpret(**options)
    assert options["port"].calls == []


def test_pair_preparation_reuses_manual_cache_and_shares_call_budget(setup):
    service, options = setup
    manual = service.interpret(**options)
    vehicle_source = {**options["source"], "kind": "vehicle", "asset_id": "vehicle-test"}
    options = {key: value for key, value in options.items() if key != "source"}
    sources = (manual["source"], vehicle_source)
    prepared = interpret_mission_assets(
        service=service, sources=sources, maximum_model_calls=10, **options
    )
    assert prepared["interpretations"]["map"]["cached"]
    assert not prepared["interpretations"]["vehicle"]["cached"]
    assert len(prepared["model_calls"]) == 1
    assert prepared["remaining_model_calls"] == 9
    reused = interpret_mission_assets(
        service=service, sources=sources, maximum_model_calls=8, **options
    )
    assert reused["remaining_model_calls"] == 8
    assert reused["model_calls"] == []
    assert len(options["port"].calls) == 2
    assert all(item["authority"] == "advisory-only" for item in reused["interpretations"].values())


def test_no_new_interpretation_can_consume_reserved_planning_calls(setup):
    service, options = setup
    source = options.pop("source")
    sources = (source, {**source, "kind": "vehicle", "asset_id": "vehicle-test"})
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        interpret_mission_assets(service=service, sources=sources, maximum_model_calls=8, **options)
    assert options["port"].calls == []


def test_map_without_named_entities_is_not_forced_to_invent_a_place(setup):
    service, options = setup
    options["source"]["required_source_ids"] = []
    interpreted = service.interpret(**options)
    assert interpreted["understanding"]["items"] == []
    assert interpreted["understanding"]["limitations"]


@pytest.mark.parametrize(
    "url",
    [
        "https://other.supabase.co/functions/v1/model-gateway",
        "https://example.supabase.co:1234/functions/v1/model-gateway",
        "https://example.supabase.co/extra/functions/v1/model-gateway",
    ],
)
def test_platform_grants_cannot_be_routed_to_another_project_or_endpoint(url):
    from dronedream_agent_app.mission_service import _validate_gateway

    with pytest.raises(ValueError):
        _validate_gateway(url, "https://example.supabase.co/auth/v1")


def test_concurrent_interpretation_does_not_launch_duplicate_model_request(setup):
    service, options = setup
    service._lock.acquire()
    try:
        with pytest.raises(ValueError, match="BUSY"):
            service.interpret(**options)
        assert not options["port"].calls
    finally:
        service._lock.release()


def test_interpret_api_requires_identity_and_rejects_wrong_owner(tmp_path):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.server import create_app

    store = AppStore(tmp_path)
    identity = VerifiedIdentity(
        owner_account_id="account-test",
        tenant_id=None,
        organization_id=None,
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    with TestClient(
        create_app(store=store, token="a" * 64, identity_verifier=lambda token: identity)
    ) as client:
        body = {
            "expected_owner_account_id": "wrong-owner",
            "source_edition": "autonomy",
            "kind": "map",
            "asset_id": "map-test",
            "content_sha256": "a" * 64,
            "model_id": "kimi-k2.6",
            "model_grant": "ddg_" + "b" * 24,
        }
        url = "/v1/threads/thread-test/interpret"
        headers = {"Authorization": "Bearer " + "a" * 64}
        assert client.post(url, json=body, headers=headers).status_code == 401
        headers["X-DroneDream-Identity-Token"] = "test-session"
        assert client.post(url, json=body, headers=headers).status_code == 403
        body["force"] = "false"
        assert client.post(url, json=body, headers=headers).status_code == 422
    store.close()


def test_interpret_api_uses_selected_model_caches_and_cannot_make_a_plan(setup, monkeypatch):
    from fastapi.testclient import TestClient

    from dronedream_agent_app import asset_interpretation
    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.server import create_app

    service, options = setup
    store = service.store
    identity = VerifiedIdentity(
        "account-test", None, None, "https://example.supabase.co/auth/v1", 2_000_000_000
    )
    # 只替换源文件和外部付费传输；真实 API 身份绑定、插件检查、缓存及存储均运行。
    monkeypatch.setattr(
        asset_interpretation, "interpretation_source", lambda *args: options["source"]
    )
    connections = []

    def port_factory(*args, **kwargs):
        connections.append(kwargs)
        return options["port"]

    monkeypatch.setattr(asset_interpretation, "StructuredModelPort", port_factory)
    thread = store.create_thread("offline interpretation", "kimi-k2.6")
    headers = {
        "Authorization": "Bearer " + "a" * 64,
        "X-DroneDream-Identity-Token": "offline-signed-session",
    }
    body = {
        "expected_owner_account_id": "account-test",
        "source_edition": "autonomy",
        "kind": "map",
        "asset_id": "map-test",
        "content_sha256": "a" * 64,
        "model_id": "kimi-k2.6",
        "model_grant": "ddg_" + "b" * 24,
        "gateway_base_url": "https://example.supabase.co/functions/v1/model-gateway",
    }
    with TestClient(
        create_app(store=store, token="a" * 64, identity_verifier=lambda _: identity)
    ) as client:
        url = f"/v1/threads/{thread['thread_id']}/interpret"
        first = client.post(url, json=body, headers=headers)
        assert first.status_code == 200, first.text
        assert not first.json()["cached"]
        second = client.post(url, json=body, headers=headers)
        assert second.status_code == 200 and second.json()["cached"]
        assert len(options["port"].calls) == 1
        assert connections[0]["settings"].model == "kimi-k2.6"
        assert connections[0]["api_key"] == body["model_grant"]
        assert options["port"].closed
        current = store.get_thread(thread["thread_id"])
        assert current["state"] == "planning"
        assert all(message["kind"] == "status" for message in current["messages"])
        assert not any(
            message["metadata"].get("plan_revision_id") for message in current["messages"]
        )


def test_clarification_api_never_leaves_an_old_plan_confirmable(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.mission_service import MissionService
    from dronedream_agent_app.server import create_app
    from dronedream_agent_core.orchestrator import MissionClarificationRequired

    store = AppStore(tmp_path)
    identity = VerifiedIdentity(
        "account-test", None, None, "https://example.supabase.co/auth/v1", 2_000_000_000
    )
    thread = store.create_thread("offline clarification", "kimi-k2.6")
    store.set_thread_state(thread["thread_id"], "awaiting_confirmation")

    def clarify(*args, **kwargs):
        raise MissionClarificationRequired(["东门还是西门的取件处？"])

    monkeypatch.setattr(MissionService, "prepare", clarify)
    body = {
        "expected_owner_account_id": "account-test",
        "source_edition": "autonomy",
        "message": "拿下快递",
        "map_id": "map-test",
        "map_content_sha256": "a" * 64,
        "vehicle_id": "vehicle-test",
        "vehicle_content_sha256": "b" * 64,
        "model_id": "kimi-k2.6",
        "model_grant": "ddg_" + "c" * 24,
        "gateway_base_url": "https://example.supabase.co/functions/v1/model-gateway",
    }
    headers = {
        "Authorization": "Bearer " + "a" * 64,
        "X-DroneDream-Identity-Token": "offline-signed-session",
    }
    with TestClient(
        create_app(store=store, token="a" * 64, identity_verifier=lambda _: identity)
    ) as client:
        response = client.post(
            f"/v1/threads/{thread['thread_id']}/prepare", json=body, headers=headers
        )
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == {
            "code": "MISSION_CLARIFICATION_REQUIRED",
            "fields": ["东门还是西门的取件处？"],
        }
        current = store.get_thread(thread["thread_id"])
        assert current["state"] == "planning"
        assert current["messages"][-1]["kind"] == "text"
        assert not current["messages"][-1]["metadata"]["actuator_authority"]
    store.close()
