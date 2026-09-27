"""Read-only provider authorization check; never prints credentials or response bodies."""
from dronedream_agent_core.model_harness.model_port import ProviderSettings, StructuredModelPort


# 功能：通过当前配置的模型列表接口确认授权，避免启动仿真后才发现认证失败。
# 输入：当前进程 Kimi 配置；输出：非秘密的服务地址和状态码，不调用生成接口。
def main():
    settings = ProviderSettings.from_env("kimi")
    port = StructuredModelPort("kimi", timeout_seconds=15, settings=settings)
    print("provider_endpoint", settings.base_url, flush=True)
    try:
        result = port._client.models.list()
        print("authorization accepted; model_count", len(result.data), flush=True)
    except Exception as error:
        print("authorization_error", type(error).__name__, getattr(error, "status_code", None), flush=True)
        raise SystemExit(1)
    finally:
        port.close()


if __name__ == "__main__":
    main()
