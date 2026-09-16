from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   声明产品网关的 DeepSeek 文本模型目录，不执行供应商能力探测。
# 输入：
#   无。
# 输出：
#   definition：包含模型目录、权限和展示位置的提供方插件定义。
def plugin_definition() -> PluginDefinition:
    models = [
        {
            "id": "deepseek-v4-flash",
            "label": "DeepSeek V4 Flash",
            "provider": "deepseek",
            "supports_image_input": False,
        },
        {
            "id": "deepseek-v4-pro",
            "label": "DeepSeek V4 Pro",
            "provider": "deepseek",
            "supports_image_input": False,
        },
    ]
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="model.deepseek",
            name="DeepSeek 模型",
            version="1.0.0",
            description="通过 DroneDream 托管网关提供结构化 DeepSeek 模型调用。",
            publisher="DroneDream",
            runtime=PluginRuntime(kind="model-provider"),
            capabilities=[
                PluginCapability(
                    capability_id="model.deepseek.structured",
                    kind="model-provider",
                    name="DeepSeek 结构化模型",
                    description="提供意图、计划、检查点与完成验收模型角色。",
                    metadata={"provider": "deepseek", "models": models},
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
                plugin_order=20,
            ),
        )
    )
    return definition
