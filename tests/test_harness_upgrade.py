"""Installed-user upgrade regression, without network calls or flight authority."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from dronedream_agent_app.harness_design_service import (
    HarnessDesignService, HarnessRevision, _candidate_from_topology,
    official_topology_templates, validate_and_compile_harness,
)
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_app.server import create_app


# 功能：
#   在临时用户目录中保存旧版官方流程，保留旧摘要和版本历史。
# 输入：
#   root：用例目录；custom：是否修改执行策略以模拟自定义流程。
# 输出：
#   store、manager、path、original：测试存储、插件替身、旧记录路径与原始字节。
def legacy_store(root, custom=False):
    store = AppStore(root)
    manager = Mock(spec=PluginManager)
    service = HarnessDesignService(store, manager)
    template = official_topology_templates()["topology.balanced-closed-loop"]
    nodes = [n.model_copy(update={"depends_on": ["mission.runtime-checkpoints"],
                                 "required_inputs": ["checkpoints"]})
             if n.node_id == "mission.evidence-finalize" else n
             for n in template.nodes if n.node_id != "mission.verification-plan"]
    candidate = _candidate_from_topology(template.model_copy(update={"nodes": nodes}),
                                        profile_id="harness.profile-balanced", base_revision=0)
    if custom:
        candidate.nodes[0].policy.timeout_seconds += 1
    validation = validate_and_compile_harness(candidate).model_copy(update={"valid": True, "issues": []})
    record = HarnessRevision(revision=1, state="active", candidate=candidate,
                             validation=validation, created_at="2026-08-27T08:02:21Z")
    path = service._revision_path(1)
    service._write_json(path, record.model_dump(mode="json"))
    original = path.read_bytes()
    return store, manager, path, original


# 功能：
#   确认旧官方流程升级后具备必需阶段，旧文件不变，多次重启不重复生成版本。
# 输入：
#   tmp_path：独立用户目录。
# 输出：
#   None：断言升级、冻结、历史保留和幂等性。
def test_legacy_upgrade_preserves_history_and_is_idempotent(tmp_path):
    store, manager, path, original = legacy_store(tmp_path)
    try:
        service = HarnessDesignService(store, manager)
        frozen = service.freeze_active_for_task()
        assert frozen["revision"] == 2
        assert "mission.verification-plan" in {n["node_id"] for n in frozen["topology"]["nodes"]}
        assert path.read_bytes() == original
        assert HarnessDesignService(store, manager).freeze_active_for_task()["revision"] == 2
        manager.apply_profile.assert_not_called()
    finally:
        store.close()


# 功能：
#   用固定的实际旧版官方配置回归升级，不随当前模板变化而重新生成历史样本。
# 输入：
#   tmp_path：独立应用目录。
# 输出：
#   None：断言历史样本可升级且语义校验通过。
def test_frozen_august_configuration_upgrades(tmp_path):
    store, manager, path, _ = legacy_store(tmp_path)
    original = (Path(__file__).parent/'fixtures/harness-balanced-pre-verification.json').read_bytes()
    path.write_bytes(original)
    try:
        service = HarnessDesignService(store, manager)
        assert service.freeze_active_for_task()['revision'] == 2
        assert path.read_bytes() == original
    finally:
        store.close()


# 功能：
#   验证自定义旧配置不被官方模板覆盖，防止升级丢失用户执行策略。
# 输入：
#   tmp_path：独立用户目录。
# 输出：
#   None：断言原始数据和索引保持不变。
def test_custom_legacy_is_not_silently_replaced(tmp_path):
    store, manager, path, original = legacy_store(tmp_path, custom=True)
    try:
        service = HarnessDesignService(store, manager)
        assert service._read_index()["active_revision"] == 1
        assert path.read_bytes() == original
    finally:
        store.close()


# 功能：
#   模拟最后发布索引失败，验证旧配置不丢失，重试能完成升级。
# 输入：
#   tmp_path：独立用户目录；monkeypatch：注入索引写入失败。
# 输出：
#   None：断言原记录保留且重启恢复。
def test_interrupted_upgrade_can_retry(tmp_path, monkeypatch):
    store, manager, path, original = legacy_store(tmp_path)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(HarnessDesignService, "_save_index", Mock(side_effect=OSError("disk full")))
            with pytest.raises(OSError):
                HarnessDesignService(store, manager)
        assert path.read_bytes() == original
        assert HarnessDesignService(store, manager).freeze_active_for_task()["revision"] == 3
    finally:
        store.close()


# 功能：
#   让真实 HTTP 路由发生配置或未知异常，验证错误编号与脱敏日志可关联。
# 输入：
#   tmp_path：独立应用目录；monkeypatch：替换只读配置入口；known：是否为配置异常。
# 输出：
#   None：断言状态码、错误编号和日志无机密内容。
@pytest.mark.usefixtures("isolated_server_credentials")
@pytest.mark.parametrize("known", [True, False])
def test_http_errors_have_private_safe_evidence(tmp_path, monkeypatch, known):
    from dronedream_agent_app.harness_design_service import HarnessDesignServiceError
    store = AppStore(tmp_path)
    app = create_app(store=store, token="upgrade-token")
    error = HarnessDesignServiceError("secret-token") if known else RuntimeError("secret-token")
    monkeypatch.setattr(HarnessDesignService, "current", Mock(side_effect=error))
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/v1/harness/topologies/current", headers={"Authorization": "Bearer upgrade-token"})
        assert response.status_code == (409 if known else 500)
        entry = json.loads((tmp_path / "logs/backend-errors.jsonl").read_text(encoding="utf-8"))
        assert entry["error_id"] == response.json()["detail"]["error_id"]
        assert "secret-token" not in json.dumps(entry)
        assert "upgrade-token" not in json.dumps(entry)
    finally:
        store.close()


# 功能：
#   验证模型错误能保留供应商状态码诊断，不泄露响应正文、令牌和供应商错误消息。
# 输入：
#   tmp_path：独立目录；monkeypatch：注入模型调用失败。
# 输出：
#   None：断言错误编号、状态码和脱敏证据正确。
@pytest.mark.usefixtures("isolated_server_credentials")
def test_model_errors_preserve_status_without_secrets(tmp_path, monkeypatch):
    import httpx
    from openai import BadRequestError
    from dronedream_agent_core.model_harness.model_port import ModelInvocationError
    store = AppStore(tmp_path)
    app = create_app(store=store, token='diagnostic-token')
    cause = BadRequestError('secret-provider-text', response=httpx.Response(400, request=httpx.Request('POST', 'https://example.com')), body={'secret': 'secret-response'})
    error = ModelInvocationError('private-message', attempts_used=1)
    error.__cause__ = cause
    monkeypatch.setattr(HarnessDesignService, 'current', Mock(side_effect=error))
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get('/v1/harness/topologies/current', headers={'Authorization': 'Bearer diagnostic-token'})
        assert response.status_code == 409
        assert response.json()['detail']['code'] == 'MODEL_INVOCATION_FAILED'
        raw = (tmp_path/'logs/backend-errors.jsonl').read_text(encoding='utf-8')
        assert 'status=400' in raw
        assert not any(secret in raw for secret in ('secret-provider-text', 'secret-response', 'private-message', 'diagnostic-token'))
    finally:
        store.close()
