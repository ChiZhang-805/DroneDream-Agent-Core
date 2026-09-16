from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   声明产品网关的 OpenAI 模型目录及图像输入标记，不创建在线调用会话。
# 输入：
#   无。
# 输出：
#   definition：包含模型目录、权限和展示位置的提供方插件定义。
def plugin_definition() -> PluginDefinition:
    # 此处是产品支持清单；图像标记不替代实际端口对输入材料的准入验证。
    models = [
        {
            "id": "gpt-4.1",
            "label": "GPT 4.1",
            "provider": "openai",
            "supports_image_input": True,
        },
        {
            "id": "gpt-5.1",
            "label": "GPT 5.1",
            "provider": "openai",
            "supports_image_input": True,
        },
        {
            "id": "gpt-5.4",
            "label": "GPT 5.4",
            "provider": "openai",
            "supports_image_input": True,
        },
    ]
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="model.openai",
            name="OpenAI 模型",
            version="1.0.0",
            description="通过 DroneDream 托管网关提供结构化 OpenAI 模型调用。",
            publisher="DroneDream",
            runtime=PluginRuntime(kind="model-provider"),
            capabilities=[
                PluginCapability(
                    capability_id="model.openai.structured",
                    kind="model-provider",
                    name="OpenAI 结构化模型",
                    description="提供意图、计划、检查点与完成验收模型角色。",
                    metadata={"provider": "openai", "models": models},
                )
            ],
            permissions=["network.model-gateway"],
            default_enabled=True,
            placement=PluginPlacement(
                category_id="models",
                category_label="模型与推理",
                slot_id="models.providers",
                slot_label="模型供应商",
                activation_mode="multiple",
                scope="general",
                category_order=50,
                slot_order=10,
                plugin_order=10,
            ),
        )
    )
    return definition
