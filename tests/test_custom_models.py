from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from dronedream_agent_app.custom_models import (
    CustomModelService,
    WindowsCredentialVault,
    detect_provider,
    validate_custom_base_url,
)
from dronedream_agent_app.storage import AppStore


class MemoryVault:
    # 功能：
    #   建立隔离的内存秘密表，不使用真实用户凭证库。
    # 输入：
    #   self：测试凭证库。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    # 功能：
    #   保存测试秘密供模型服务后续取用。
    # 输入：
    #   self：测试凭证库。
    #   name：测试凭证标识。
    #   secret：测试秘密原文。
    # 输出：
    #   None：不返回业务数据。
    def put(self, name: str, secret: str) -> None:
        self.values[name] = secret

    # 功能：
    #   精确读取测试秘密，缺失时保持 KeyError 语义。
    # 输入：
    #   self：测试凭证库。
    #   name：测试凭证标识。
    # 输出：
    #   secret：已保存的测试秘密。
    def get(self, name: str) -> str:
        secret = self.values[name]
        return secret

    # 功能：
    #   幂等撤销内存秘密，用于检查模型删除后的凭证回收。
    # 输入：
    #   self：测试凭证库。
    #   name：要移除的标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, name: str) -> None:
        self.values.pop(name, None)


# 功能：
#   提供独立模型数据库并负责关闭，失败测试同样不遗留 SQLite 连接。
# 输入：
#   tmp_path：当前测试独占目录。
# 输出：
#   store：测试使用的本机存储。
@pytest.fixture
def store(tmp_path):
    store = AppStore(tmp_path)
    try:
        yield store
    finally:
        store.close()


# 功能：
#   验证品牌识别优先按端点，未知端点依次按密钥前缀、模型名称和通用品牌兜底。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_provider_detection_prefers_endpoint_and_has_safe_fallback():
    assert detect_provider("https://api.openai.com/v1")["provider"] == "openai"
    assert detect_provider("https://openrouter.ai/api/v1")["provider"] == "openrouter"
    assert (
        detect_provider("https://generativelanguage.googleapis.com/v1beta/openai")["provider"]
        == "gemini"
    )
    assert detect_provider("https://api.cerebras.ai/v1")["provider"] == "cerebras"
    assert detect_provider("https://open.bigmodel.cn/api/paas/v4")["provider"] == "zhipu"
    assert detect_provider("https://dashscope.aliyuncs.com/compatible-mode/v1")["icon"] == "qwen"
    assert detect_provider("https://api.x.ai/v1")["provider"] == "xai"
    assert detect_provider("https://api.anthropic.com/v1")["icon"] == "claude"
    assert detect_provider("https://bedrock-runtime.us-east-1.amazonaws.com")["icon"] == "bedrock"
    assert detect_provider("https://models.github.ai/inference")["provider"] == "github-models"
    assert detect_provider("https://private.example/v1", "gsk_example")["provider"] == "groq"
    assert detect_provider("https://private.example/v1", model_id="claude-sonnet-4") == {
        "provider": "anthropic",
        "icon": "claude",
    }
    assert detect_provider("https://private.example/v1", model_id="qwen3-max")["provider"] == "qwen"
    assert detect_provider("https://private.example/v1", model_id="grok-4")["icon"] == "grok"
    assert detect_provider("https://private.example/v1")["provider"] == "openai-compatible"


# 功能：
#   回环模型允许 HTTP，远端明文地址及查询中夹带秘密的地址必须拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_custom_endpoint_requires_https_except_for_loopback():
    assert validate_custom_base_url("http://127.0.0.1:11434/v1").startswith("http://")
    with pytest.raises(ValueError, match="CUSTOM_MODEL_BASE_URL_INVALID"):
        validate_custom_base_url("http://models.example/v1")
    with pytest.raises(ValueError, match="CUSTOM_MODEL_BASE_URL_INVALID"):
        validate_custom_base_url("https://models.example/v1?token=secret")


# 功能：
#   目录与持久化元数据不暴露秘密，短期授权仅能由绑定任务与模型消费一次。
# 输入：
#   store：隔离模型与任务存储。
# 输出：
#   None：不返回业务数据。
def test_custom_catalog_and_grant_never_return_long_lived_key(store):
    vault = MemoryVault()
    service = CustomModelService(store, vault)
    profile = service.create(
        display_name="Private Planner",
        base_url="https://models.example/v1",
        model_id="planner-72b",
        api_key="secret-customer-key",
        api_style="chat-completions",
    )
    profile_id = str(profile["profile_id"])
    catalog = service.catalog()
    assert catalog == [
        {
            "id": f"custom:{profile_id}",
            "model": "planner-72b",
            "label": "Private Planner",
            "provider": "openai-compatible",
            "icon": "generic",
            "source": "custom",
            "profile_id": profile_id,
        }
    ]
    assert "secret-customer-key" not in str(profile)
    assert "secret-customer-key" not in str(store.list_custom_models())

    thread = store.create_thread("custom", f"custom:{profile_id}")
    grant = service.issue_grant(profile_id, str(thread["thread_id"]))
    assert str(grant["grant"]).startswith("ddc_")
    connection = service.consume_grant(
        str(grant["grant"]), str(thread["thread_id"]), f"custom:{profile_id}"
    )
    assert connection.api_key == "secret-customer-key"
    with pytest.raises(ValueError, match="CUSTOM_MODEL_GRANT_INVALID"):
        service.consume_grant(str(grant["grant"]), str(thread["thread_id"]), f"custom:{profile_id}")


# 功能：
#   删除或禁用模型后，先前签发的短期授权不能继续调用该模型。
# 输入：
#   store：隔离模型与任务存储。
#   action：模拟的删除或禁用操作。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("action", ["delete", "disable"])
def test_profile_revocation_invalidates_already_issued_grants(store, action):
    service = CustomModelService(store, MemoryVault())
    profile = service.create(
        display_name="Private",
        base_url="https://models.example/v1",
        model_id="planner",
        api_key="unit-test-key",
        api_style="chat-completions",
    )
    profile_id = profile["profile_id"]
    thread = store.create_thread("custom", f"custom:{profile_id}")
    issued = service.issue_grant(profile_id, thread["thread_id"])
    if action == "delete":
        service.delete(profile_id)
    else:
        with store.transaction() as connection:
            connection.execute(
                "UPDATE custom_models SET enabled=0 WHERE profile_id=?",
                (profile_id,),
            )
    with pytest.raises(ValueError, match="CUSTOM_MODEL_(GRANT_INVALID|DISABLED)"):
        service.consume_grant(issued["grant"], thread["thread_id"], f"custom:{profile_id}")


# 功能：
#   模型目录查询成功和网络异常都关闭独立客户端，不触碰真实模型 API。
# 输入：
#   store：隔离的本机存储。
#   monkeypatch：替换模型客户端的工具。
#   fails：是否模拟目录网络失败。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fails", [False, True])
def test_model_discovery_closes_its_client_on_all_paths(store, monkeypatch, fails):
    closed = []

    # 功能：
    #   按外层故障标志返回合成目录或抛出连接失败，替代实际网络请求。
    # 输入：
    #   无；使用外层 fails 标志。
    # 输出：
    #   result：仅包含测试模型的目录响应。
    def model_list():
        if fails:
            raise ConnectionError("unit-test offline")
        result = SimpleNamespace(data=[SimpleNamespace(id="planner")])
        return result

    client = SimpleNamespace(
        models=SimpleNamespace(list=model_list),
        close=lambda: closed.append(1),
    )
    monkeypatch.setattr("dronedream_agent_app.custom_models.OpenAI", lambda **kwargs: client)
    service = CustomModelService(store, MemoryVault())
    if fails:
        with pytest.raises(ConnectionError):
            service.discover(base_url="https://models.example/v1", api_key="unit-test-key")
    else:
        assert service.discover(base_url="https://models.example/v1", api_key="unit-test-key")[
            "models"
        ] == ["planner"]
    assert closed == [1]


# 功能：
#   在独占测试目录验证真实 Windows DPAPI 保存、读取和删除，磁盘内容不得含测试明文。
# 输入：
#   tmp_path：测试独占凭证目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
def test_windows_vault_round_trip_is_encrypted_at_rest(tmp_path):
    vault = WindowsCredentialVault(tmp_path)
    secret = "sk-local-test-value-that-must-not-be-plaintext"
    vault.put("cmp-12345678", secret)
    blob = (tmp_path / "cmp-12345678.dpapi").read_bytes()
    assert secret.encode() not in blob
    assert vault.get("cmp-12345678") == secret
    vault.delete("cmp-12345678")
    assert not (tmp_path / "cmp-12345678.dpapi").exists()
