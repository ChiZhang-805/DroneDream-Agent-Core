from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dronedream_agent_app import conversation_title
from dronedream_agent_app.conversation_title import ConversationTitle
from dronedream_agent_app.identity import VerifiedIdentity
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore

pytestmark = pytest.mark.usefixtures("isolated_server_credentials")


@pytest.mark.parametrize("value", ["", " ", "a\nb", "x" * 33, "a\x00b"])
def test_invalid_model_titles_are_rejected(value):
    with pytest.raises(ValidationError):
        ConversationTitle(title=value)


# 功能：
#   验证命名实际经过授权、模型调用和会话保存，且不会创建计划或执行动作。
# 输入：
#   tmp_path：隔离目录；monkeypatch：测试传输替身。
# 输出：
#   无返回值。
def test_title_endpoint_identity_model_boundary_and_saved_result(tmp_path, monkeypatch):
    store = AppStore(tmp_path)
    thread = store.create_thread("用户原文", "kimi-k2.6")
    identity = VerifiedIdentity("account-test", None, None, "https://example.supabase.co/auth/v1", 2_000_000_000)
    calls = []
    closed = []

    class Port:
        def call(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(artifact=ConversationTitle(title="去外卖点取餐"), record=SimpleNamespace(model_dump=lambda **_: {"provider": "kimi", "model": "kimi-k2.6"}))

        def close(self):
            closed.append(True)

    monkeypatch.setattr(conversation_title, "StructuredModelPort", lambda *args, **kwargs: Port())
    body = {"expected_owner_account_id": "account-test", "source_edition": "autonomy", "model_id": "kimi-k2.6", "model_grant": "ddg_" + "b" * 24, "gateway_base_url": "https://example.supabase.co/functions/v1/model-gateway", "message": "去外卖点拿餐再回办公室"}
    headers = {"Authorization": "Bearer " + "a" * 64, "X-DroneDream-Identity-Token": "test-session"}
    with TestClient(create_app(store=store, token="a" * 64, identity_verifier=lambda _: identity)) as client:
        url = f"/v1/threads/{thread['thread_id']}/title"
        assert client.post(url, json=body).status_code in (401, 403)
        assert client.post(url, json={**body, "expected_owner_account_id": "other"}, headers=headers).status_code == 403
        assert client.post(url, json={**body, "model_id": "wrong-model"}, headers=headers).status_code == 409
        assert client.post(url, json={**body, "gateway_base_url": "https://attacker.example/v1"}, headers=headers).status_code == 409
        assert not calls
        result = client.post(url, json=body, headers=headers)
        assert result.status_code == 200, result.text
        assert result.json()["title"] == "去外卖点取餐"
        assert result.json()["actuator_authority"] is False
    assert len(calls) == len(closed) == 1
    assert calls[0]["maximum_physical_attempts"] == 1
    assert calls[0]["input_artifact"]["message"] == body["message"]
    saved = store.get_thread(thread["thread_id"])
    assert saved["title"] == "去外卖点取餐"
    assert saved["messages"] == []
    assert saved["state"] == "planning"
