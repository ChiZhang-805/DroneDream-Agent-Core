"""Empty inherited variables must not route another provider's key to OpenAI."""
import pytest
from dronedream_agent_core.model_harness.model_port import ProviderSettings


@pytest.mark.parametrize("provider,prefix,url", [
    ("kimi", "KIMI", "https://api.moonshot.ai/v1"),
    ("deepseek", "DEEPSEEK", "https://api.deepseek.com"),
])
@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_empty_inherited_endpoint_uses_matching_provider(monkeypatch, provider, prefix, url, value):
    """功能：空变量不可串到 SDK 默认服务；输入：WSL/容器继承空值；输出：对应服务地址。"""
    monkeypatch.setenv(prefix + "_BASE_URL", value)
    assert ProviderSettings.from_env(provider).base_url == url


def test_nonempty_custom_provider_endpoint_is_preserved(monkeypatch):
    """功能：尊重已有自定义网关；输入：明确 Kimi 地址；输出：原指定地址。"""
    monkeypatch.setenv("KIMI_BASE_URL", "https://example.org/v1")
    assert ProviderSettings.from_env("kimi").base_url == "https://example.org/v1"
