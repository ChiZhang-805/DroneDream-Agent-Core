from __future__ import annotations

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)


# 功能：
#   声明产品网关的 Kimi 模型目录及当前文本输入配置，不执行供应商能力探测。
# 输入：
#   无。
# 输出：
#   definition：包含模型目录、权限和展示位置的提供方插件定义。
def plugin_definition() -> PluginDefinition:
    # 标记对应本产品的接入配置，不据此推断供应商所有模型的视觉能力。
    models = [
        {
            "id": "kimi-k2.6",
            "label": "Kimi K2.6",
            "provider": "kimi",
            "supports_image_input": False,
        },
        {
            "id": "kimi-k3",
            "label": "Kimi K3",
            "provider": "kimi",
            "supports_image_input": False,
        },
    ]
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="model.kimi",
            name="Kimi 模型",
            version="1.0.0",
            description="通过 DroneDream 托管网关提供结构化 Kimi 模型调用。",
            publisher="DroneDream",
            runtime=PluginRuntime(kind="model-provider"),
            capabilities=[
                PluginCapability(
                    capability_id="model.kimi.structured",
                    kind="model-provider",
                    name="Kimi 结构化模型",
                    description="提供意图、计划、检查点与完成验收模型角色。",
                    metadata={"provider": "kimi", "models": models},
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
                plugin_order=30,
            ),
        )
    )
    return definition
