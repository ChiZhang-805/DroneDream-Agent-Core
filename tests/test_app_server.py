from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dronedream_agent_app.identity import VerifiedIdentity
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.model_harness.memory import AccountMemoryStore, MemoryOwnerScope

pytestmark = pytest.mark.usefixtures("isolated_server_credentials")


class _MemoryVault:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def put(self, name: str, secret: str) -> None:
        self.values[name] = secret

    def get(self, name: str) -> str:
        return self.values[name]

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def _ui_plugin_bundle(version: str) -> bytes:
    panel = b'{"title":"Mission audit"}\n'
    manifest = {
        "schema_version": "dronedream.plugin-manifest.v1",
        "plugin_id": "example.mission-audit",
        "name": "Mission Audit",
        "version": version,
        "description": "Inspect immutable mission evidence.",
        "publisher": "Example",
        "api_version": "1.0",
        "minimum_app_version": "0.1.0",
        "runtime": {"kind": "ui-declarative"},
        "capabilities": [
            {
                "capability_id": "example.mission-audit.panel",
                "kind": "ui-panel",
                "name": "Mission Audit",
                "description": "Show mission evidence.",
                "authority": "read",
                "metadata": {"entrypoint": "ui/panel.json"},
            }
        ],
        "permissions": ["mission.read", "ui.panel"],
        "file_sha256": {"ui/panel.json": hashlib.sha256(panel).hexdigest()},
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr("ui/panel.json", panel)
    return output.getvalue()


def test_loopback_api_requires_random_session_token(tmp_path):
    token = "a" * 64
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))

    assert client.get("/health").status_code == 200
    assert client.get("/v1/bootstrap").status_code == 401
    response = client.get("/v1/bootstrap", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert len(response.json()["models"]) == 7


def test_account_memory_governance_api_is_local_owner_and_thread_bound(tmp_path):
    token = "m" * 64
    headers = {
        "Authorization": f"Bearer {token}",
        "X-DroneDream-Identity-Token": "signed-supabase-session",
    }
    app_store = AppStore(tmp_path)
    verified = VerifiedIdentity(
        owner_account_id="account-001",
        tenant_id=None,
        organization_id=None,
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    client = TestClient(
        create_app(
            store=app_store,
            token=token,
            identity_verifier=lambda supplied: (
                verified
                if supplied == "signed-supabase-session"
                else (_ for _ in ()).throw(ValueError("IDENTITY_TOKEN_INVALID"))
            ),
        )
    )
    thread_ids = [
        client.post(
            "/v1/threads",
            headers=headers,
            json={"title": f"Memory {index}", "selected_model": "gpt-5.4"},
        ).json()["thread_id"]
        for index in range(3)
    ]
    scope = MemoryOwnerScope(
        owner_account_id="account-001",
        source_edition="autonomy",
    )
    memory = AccountMemoryStore(tmp_path / "account-memory.sqlite3")
    memory.upsert(
        scope,
        kind="preference",
        memory_key="preference.task",
        payload={"locale": "zh-CN"},
        source_conversation_id=thread_ids[0],
    )
    conflict = memory.record_candidate(
        MemoryOwnerScope(owner_account_id="account-001", source_edition="sim"),
        kind="preference",
        memory_key="preference.task",
        payload={"locale": "en-US"},
        source_conversation_id=thread_ids[1],
        confidence=0.8,
        ttl_days=365,
    )
    scope_body = {
        "expected_owner_account_id": "account-001",
        "source_edition": "field",
        "thread_id": thread_ids[1],
    }

    assert (
        client.post(
            "/v1/account-memory/candidates/list",
            headers={"Authorization": f"Bearer {token}"},
            json={**scope_body, "limit": 100},
        ).status_code
        == 401
    )
    listed = client.post(
        "/v1/account-memory/candidates/list",
        headers=headers,
        json={**scope_body, "limit": 100},
    )
    assert listed.status_code == 200
    conflict_receipt = next(
        item
        for item in listed.json()["candidates"]
        if item["candidate_id"] == conflict.candidate_id
    )
    assert conflict_receipt["status"] == "conflict"
    assert conflict_receipt["storage_status"] == "pending"
    assert listed.json()["owner_bound"] is True
    assert listed.json()["thread_bound"] is True

    wrong_owner = client.post(
        "/v1/account-memory/candidates/list",
        headers=headers,
        json={**scope_body, "expected_owner_account_id": "account-002", "limit": 100},
    )
    assert wrong_owner.status_code == 403
    unbound = client.post(
        "/v1/account-memory/candidates/list",
        headers=headers,
        json={**scope_body, "thread_id": thread_ids[2], "limit": 100},
    )
    assert unbound.status_code == 409
    assert unbound.json()["detail"] == "ACCOUNT_MEMORY_THREAD_NOT_BOUND"

    resolved = client.post(
        f"/v1/account-memory/candidates/{conflict.candidate_id}/resolve",
        headers=headers,
        json=scope_body,
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "consolidated"
    assert resolved.json()["memory"]["payload"] == {"locale": "en-US"}

    rejected_candidate = memory.record_candidate(
        scope,
        kind="preference",
        memory_key="preference.task",
        payload={"locale": "fr-FR"},
        source_conversation_id=thread_ids[0],
        confidence=0.8,
        ttl_days=365,
    )
    rejected = client.post(
        f"/v1/account-memory/candidates/{rejected_candidate.candidate_id}/reject",
        headers=headers,
        json=scope_body,
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    forgotten = client.post(
        "/v1/account-memory/forget",
        headers=headers,
        json={
            **scope_body,
            "memory_key": "preference.task",
            "mode": "permanent",
            "reason": "test_user_request",
        },
    )
    assert forgotten.status_code == 200
    assert forgotten.json()["status"] == "forgotten"
    assert forgotten.json()["active_deleted"] == 1
    assert forgotten.json()["candidates_deleted"] == 3


def test_execution_api_requires_verified_bound_owner_and_current_plan_revision(
    tmp_path, monkeypatch
):
    token = "e" * 64
    verified = VerifiedIdentity(
        owner_account_id="account-executor",
        tenant_id=None,
        organization_id=None,
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    captured: dict[str, object] = {}

    def execute(_self, **values):
        captured.update(values)
        return {"execution_id": "execution-" + "a" * 32, "state": "executing"}

    monkeypatch.setattr("dronedream_agent_app.runtime_manager.RuntimeManager.execute", execute)
    memory_path = tmp_path / "account-memory.sqlite3"
    client = TestClient(
        create_app(
            store=AppStore(tmp_path / "app"),
            token=token,
            account_memory_path=memory_path,
            identity_verifier=lambda supplied: (
                verified
                if supplied == "signed-executor"
                else (_ for _ in ()).throw(ValueError("IDENTITY_TOKEN_INVALID"))
            ),
        )
    )
    local_headers = {"Authorization": f"Bearer {token}"}
    headers = {
        **local_headers,
        "X-DroneDream-Identity-Token": "signed-executor",
    }
    thread_id = client.post(
        "/v1/threads",
        headers=local_headers,
        json={"title": "Execute", "selected_model": "gpt-5.4"},
    ).json()["thread_id"]
    payload = {
        "expected_owner_account_id": "account-executor",
        "source_edition": "autonomy",
        "plan_revision_id": "plan-" + "b" * 32,
        "model_id": "gpt-5.4",
        "model_grant": "ddg_abcdefghijklmnopqrstuvwxyz",
        "gateway_base_url": "https://example.supabase.co/functions/v1/model-gateway",
    }

    assert (
        client.post(
            f"/v1/threads/{thread_id}/execute", headers=local_headers, json=payload
        ).status_code
        == 401
    )
    wrong_owner = client.post(
        f"/v1/threads/{thread_id}/execute",
        headers=headers,
        json={**payload, "expected_owner_account_id": "another-account"},
    )
    assert wrong_owner.status_code == 403
    unbound = client.post(f"/v1/threads/{thread_id}/execute", headers=headers, json=payload)
    assert unbound.status_code == 409
    assert unbound.json()["detail"] == "EXECUTION_THREAD_NOT_BOUND"

    memory = AccountMemoryStore(memory_path)
    memory.bind_thread(
        MemoryOwnerScope(owner_account_id="account-executor", source_edition="autonomy"),
        thread_id,
    )
    accepted = client.post(f"/v1/threads/{thread_id}/execute", headers=headers, json=payload)
    assert accepted.status_code == 202
    assert captured["thread_id"] == thread_id
    assert captured["plan_revision_id"] == payload["plan_revision_id"]
    assert captured["owner_scope"].owner_account_id == "account-executor"


def test_runtime_control_surface_requires_verified_thread_owner_and_operator_identity(
    tmp_path, monkeypatch
):
    token = "r" * 64
    owner = VerifiedIdentity(
        owner_account_id="account-operator",
        tenant_id="tenant-a",
        organization_id="organization-a",
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    wrong_tenant = VerifiedIdentity(
        owner_account_id="account-operator",
        tenant_id="tenant-b",
        organization_id="organization-a",
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.RuntimeManager.execution_evidence",
        lambda _self, thread_id: {"thread_id": thread_id, "status": "executing"},
    )
    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.RuntimeManager.submit_message",
        lambda _self, thread_id, text: {
            "accepted": True,
            "thread_id": thread_id,
            "text": text,
        },
    )
    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.RuntimeManager.issue_takeover_grant",
        lambda _self, thread_id, **values: {
            "accepted": True,
            "thread_id": thread_id,
            **values,
        },
    )

    def submit_operator_control(_self, thread_id, **values):
        observed.update(values)
        return {"accepted": True, "thread_id": thread_id}

    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.RuntimeManager.submit_operator_control",
        submit_operator_control,
    )

    def verify(supplied: str) -> VerifiedIdentity:
        if supplied == "signed-owner":
            return owner
        if supplied == "signed-wrong-tenant":
            return wrong_tenant
        raise ValueError("IDENTITY_TOKEN_INVALID")

    memory_path = tmp_path / "account-memory.sqlite3"
    app_store = AppStore(tmp_path / "app")
    client = TestClient(
        create_app(
            store=app_store,
            token=token,
            account_memory_path=memory_path,
            identity_verifier=verify,
        )
    )
    local_headers = {"Authorization": f"Bearer {token}"}
    owner_headers = {
        **local_headers,
        "X-DroneDream-Identity-Token": "signed-owner",
    }
    wrong_tenant_headers = {
        **local_headers,
        "X-DroneDream-Identity-Token": "signed-wrong-tenant",
    }
    thread_id = client.post(
        "/v1/threads",
        headers=local_headers,
        json={"title": "Runtime owner", "selected_model": "gpt-5.4"},
    ).json()["thread_id"]
    message_id = "runtime-msg-" + "a" * 32
    endpoints = (
        ("get", f"/v1/threads/{thread_id}/execution-evidence", None),
        ("post", f"/v1/threads/{thread_id}/runtime-message", {"text": "hold"}),
        (
            "post",
            f"/v1/threads/{thread_id}/operator-takeover-grant",
            {
                "message_id": message_id,
                "operator_id": "account-operator",
                "duration_seconds": 60,
            },
        ),
        (
            "post",
            f"/v1/threads/{thread_id}/operator-control",
            {
                "message_id": message_id,
                "grant_token": "g" * 32,
                "action": "release",
            },
        ),
    )
    for method, path, body in endpoints:
        response = client.request(method, path, headers=local_headers, json=body)
        assert response.status_code == 401

    unbound = client.get(f"/v1/threads/{thread_id}/execution-evidence", headers=owner_headers)
    assert unbound.status_code == 409
    assert unbound.json()["detail"] == "EXECUTION_THREAD_NOT_BOUND"

    memory = AccountMemoryStore(memory_path)
    memory.bind_thread(
        MemoryOwnerScope(
            owner_account_id=owner.owner_account_id,
            tenant_id=owner.tenant_id,
            organization_id=owner.organization_id,
            source_edition="autonomy",
        ),
        thread_id,
    )
    wrong_scope = client.get(
        f"/v1/threads/{thread_id}/execution-evidence", headers=wrong_tenant_headers
    )
    assert wrong_scope.status_code == 403
    assert wrong_scope.json()["detail"] == "EXECUTION_THREAD_OWNER_MISMATCH"
    wrong_operator = client.post(
        f"/v1/threads/{thread_id}/operator-takeover-grant",
        headers=owner_headers,
        json={
            "message_id": message_id,
            "operator_id": "another-operator",
            "duration_seconds": 60,
        },
    )
    assert wrong_operator.status_code == 403
    assert wrong_operator.json()["detail"] == "OPERATOR_IDENTITY_MISMATCH"

    for method, path, body in endpoints:
        response = client.request(method, path, headers=owner_headers, json=body)
        assert response.status_code in {200, 201, 202}
    assert observed["operator_id"] == owner.owner_account_id


def test_visual_harness_api_exposes_real_revision_and_rejects_stale_edits(tmp_path):
    token = "c" * 64
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_app(store=AppStore(tmp_path), token=token)) as client:
        current = client.get("/v1/harness/topologies/current", headers=headers)
        assert current.status_code == 200
        assert current.json()["active"]["validation"]["valid"] is True

        moved = client.patch(
            "/v1/harness/topologies/current",
            headers=headers,
            json={
                "schema_version": "dronedream.harness-edit-operation.v1",
                "client_operation_id": "move-api-operation-0001",
                "base_revision": 1,
                "operation": "move_node",
                "payload": {"node_id": "mission.intent-parse", "x": 700, "y": 120},
            },
        )
        assert moved.status_code == 200
        assert moved.json()["revision"]["revision"] == 2

        stale = client.patch(
            "/v1/harness/topologies/current",
            headers=headers,
            json={
                "schema_version": "dronedream.harness-edit-operation.v1",
                "client_operation_id": "move-api-operation-0002",
                "base_revision": 1,
                "operation": "move_node",
                "payload": {"node_id": "mission.intent-parse", "x": 710, "y": 130},
            },
        )
        assert stale.status_code == 409
        assert stale.json()["detail"] == "HARNESS_REVISION_CONFLICT:2"

        dry_run = client.post("/v1/harness/topologies/dry-run", headers=headers)
        assert dry_run.status_code == 200
        assert dry_run.json()["external_calls_executed"] == 0

        switched = client.post(
            "/v1/plugins/harness.topology-committee/enable",
            headers=headers,
        )
        assert switched.status_code == 200
        linked = client.get("/v1/harness/topologies/current", headers=headers).json()
        assert linked["current"]["revision"] == 3
        assert linked["current"]["candidate"]["topology_id"] == ("topology.committee-closed-loop")
        assert linked["active"]["revision"] == 3


def test_attachment_is_bound_to_its_task_thread(tmp_path):
    token = "b" * 64
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))
    headers = {"Authorization": f"Bearer {token}"}
    thread = client.post(
        "/v1/threads",
        headers=headers,
        json={"title": "附件任务", "selected_model": "gpt-5.4"},
    ).json()

    response = client.post(
        f"/v1/threads/{thread['thread_id']}/attachments",
        headers=headers,
        files={"attachment": ("requirements.md", "到保安亭取外卖", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json()["thread_id"] == thread["thread_id"]
    assert response.json()["extracted_text"] == "到保安亭取外卖"


def test_connector_credential_api_never_returns_secret(tmp_path):
    token = "d" * 64
    headers = {"Authorization": f"Bearer {token}"}
    vault = _MemoryVault()
    client = TestClient(
        create_app(
            store=AppStore(tmp_path),
            token=token,
            connector_credential_vault=vault,
        )
    )

    response = client.post(
        "/v1/connector-credentials",
        headers=headers,
        json={
            "display_name": "PagerDuty",
            "secret": "pd-secret-value",
            "allowed_plugin_ids": ["connector.alerts.pagerduty"],
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert "pd-secret-value" not in response.text
    assert vault.values[created["reference"]] == "pd-secret-value"
    bootstrap = client.get("/v1/bootstrap", headers=headers)
    assert "pd-secret-value" not in bootstrap.text
    assert bootstrap.json()["connector_credentials"][0]["allowed_plugin_ids"] == [
        "connector.alerts.pagerduty"
    ]
    removed = client.delete(f"/v1/connector-credentials/{created['reference']}", headers=headers)
    assert removed.status_code == 200
    assert created["reference"] not in vault.values


def test_plugin_api_is_authenticated_transactional_and_versioned(tmp_path):
    token = "c" * 64
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))

    assert client.get("/v1/plugins", headers=headers).status_code == 200
    protected = client.post("/v1/plugins/runtime.safe-hold/disable", headers=headers)
    assert protected.status_code == 409

    disabled = client.post("/v1/plugins/model.kimi/disable", headers=headers)
    assert disabled.status_code == 200
    assert len(client.get("/v1/bootstrap", headers=headers).json()["models"]) == 5

    imported = client.post(
        "/v1/plugins/import",
        headers=headers,
        files={"bundle": ("mission-audit.zip", _ui_plugin_bundle("1.0.0"), "application/zip")},
    )
    assert imported.status_code == 201
    assert imported.json()["version"] == "1.0.0"
    assert (
        client.post(
            "/v1/plugins/example.mission-audit/trust-local-package", headers=headers
        ).status_code
        == 200
    )
    assert (
        client.post("/v1/plugins/example.mission-audit/enable", headers=headers).status_code == 200
    )
    panel = client.get("/v1/plugins/example.mission-audit/panel", headers=headers)
    assert panel.status_code == 200
    assert panel.json()["title"] == "Mission audit"

    staged = client.post(
        "/v1/plugins/import",
        headers=headers,
        files={"bundle": ("mission-audit.zip", _ui_plugin_bundle("1.1.0"), "application/zip")},
    )
    assert staged.status_code == 201
    assert staged.json()["staged_version"] == "1.1.0"

    activated = client.post(
        "/v1/plugins/example.mission-audit/activate",
        headers=headers,
        json={"version": "1.1.0"},
    )
    assert activated.status_code == 200
    detail = client.get("/v1/plugins/example.mission-audit", headers=headers).json()
    assert detail["version"] == "1.1.0"
    assert len(detail["versions"]) == 2
    assert detail["events"]

    removed = client.delete("/v1/plugins/example.mission-audit", headers=headers)
    assert removed.status_code == 200
    assert removed.json()["status"] == "uninstalled"


def test_asset_import_job_api_persists_quarantine_failures(tmp_path):
    token = "e" * 64
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))

    created = client.post(
        "/v1/asset-import-jobs",
        headers=headers,
        data={"source_format": "ddpkg"},
        files={"bundle": ("broken.ddpkg", b"not-a-zip", "application/zip")},
    )

    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert created.json()["state"] == "parsing"
    assert client.get("/v1/asset-import-jobs").status_code == 401
    assert client.get("/v1/asset-import-jobs", headers=headers).json()[0]["job_id"] == job_id

    processed = client.post(
        f"/v1/asset-import-jobs/{job_id}/process",
        headers=headers,
    )
    assert processed.status_code == 422
    failed = client.get(f"/v1/asset-import-jobs/{job_id}", headers=headers).json()
    assert failed["state"] == "failed"
    assert failed["issue_codes"] == ["DDPKG_NOT_ZIP"]
    assert client.post(f"/v1/asset-import-jobs/{job_id}/cancel", headers=headers).status_code == 409
    bootstrap = client.get("/v1/bootstrap", headers=headers).json()
    assert bootstrap["asset_import_jobs"][0]["job_id"] == job_id


def test_companion_result_api_requires_the_exact_quarantined_source_binding(tmp_path):
    token = "7" * 64
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))
    created = client.post(
        "/v1/asset-import-jobs",
        headers=headers,
        data={"source_format": "auto", "expected_kind": "map"},
        files={"bundle": ("campus.blend", b"native-blender-project", "application/octet-stream")},
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    waiting = client.post(f"/v1/asset-import-jobs/{job_id}/process", headers=headers)
    assert waiting.status_code == 200
    assert waiting.json()["required_inputs"] == ["plugin_adapter:blender.phobos"]

    rejected = client.post(
        f"/v1/asset-import-jobs/{job_id}/companion-result",
        headers=headers,
        data={"source_package_sha256": "f" * 64, "adapter_id": "blender.phobos"},
        files={"result": ("campus.ddpkg", b"not-trusted", "application/zip")},
    )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "ASSET_COMPANION_SOURCE_HASH_MISMATCH"


def test_current_bundled_pair_is_seeded_into_content_addressed_api(tmp_path):
    token = "f" * 64
    headers = {"Authorization": f"Bearer {token}"}
    repository_resources = Path(__file__).parents[1] / "app" / "desktop" / "src-tauri" / "resources"
    resource_root = tmp_path / "resources"
    shutil.copytree(repository_resources / "default-assets", resource_root / "default-assets")
    official_plugins = resource_root / "official-plugins"
    official_plugins.mkdir(parents=True)
    (official_plugins / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.official-plugin-index.v1",
                "plugins": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            store=AppStore(tmp_path / "store"),
            token=token,
            resource_root=resource_root,
        )
    )

    bootstrap = client.get("/v1/bootstrap", headers=headers).json()
    assert "maps" not in bootstrap
    assert "vehicles" not in bootstrap
    versions = bootstrap["asset_versions"]
    bundled_index = json.loads(
        (resource_root / "default-assets" / "index.json").read_text(encoding="utf-8")
    )
    package_keys = {
        (entry["asset_id"], entry["content_sha256"])
        for pair in bundled_index["qualified_pairs"]
        for entry in pair["packages"]
    }
    assert len(versions) == len(package_keys)
    assert {(entry["asset_id"], entry["content_sha256"]) for entry in versions} == package_keys
    assert {entry["maturity"] for entry in versions} == {"qualified"}
    default_packages = {
        entry["kind"]: entry for entry in bundled_index["qualified_pair"]["packages"]
    }
    map_version = next(
        entry
        for entry in versions
        if entry["asset_id"] == default_packages["map"]["asset_id"]
        and entry["content_sha256"] == default_packages["map"]["content_sha256"]
    )
    vehicle_version = next(
        entry
        for entry in versions
        if entry["asset_id"] == default_packages["vehicle"]["asset_id"]
        and entry["content_sha256"] == default_packages["vehicle"]["content_sha256"]
    )

    created = client.post(
        "/v1/asset-qualification-jobs",
        headers=headers,
        json={
            "map_asset_id": map_version["asset_id"],
            "map_content_sha256": map_version["content_sha256"],
            "vehicle_asset_id": vehicle_version["asset_id"],
            "vehicle_content_sha256": vehicle_version["content_sha256"],
        },
    )

    assert created.status_code == 201
    job = created.json()
    assert job["state"] == "created"
    assert (
        client.get(f"/v1/asset-qualification-jobs/{job['job_id']}", headers=headers).json()[
            "job_id"
        ]
        == job["job_id"]
    )
    assert (
        client.get("/v1/asset-qualification-jobs", headers=headers).json()[0]["job_id"]
        == job["job_id"]
    )
