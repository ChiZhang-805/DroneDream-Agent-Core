"""Cloud/local port policies and invocation receipts; these are not flight-control experts."""

from __future__ import annotations

from typing import Any

from dronedream_agent_core.contracts import AttachmentArtifact
from dronedream_agent_core.plugin_api import PluginDefinition

from ._helpers import hook_plugin


# 功能：
#   检查端口列表类型、数量和名称长度，按首次出现顺序去重。
# 输入：
#   available_ports：当前配置中可供路由的端口名称列表。
# 输出：
#   ports：通过格式检查且不重复的端口列表。
def _ports(available_ports: list[str]) -> list[str]:
    if (
        not isinstance(available_ports, list)
        or len(available_ports) > 64
        or any(
            not isinstance(port, str) or not port.strip() or len(port) > 160
            for port in available_ports
        )
    ):
        raise ValueError("MODEL_ROUTER_PORTS_INVALID")
    ports = list(dict.fromkeys(available_ports))
    return ports


# 功能：
#   按请求端口及角色端口顺序生成回退候选；没有可用端口时返回空候选供宿主拒绝。
# 输入：
#   requested_port：角色策略选中的首选端口。
#   available_ports：当前可用端口名称列表。
#   role：调用角色，用于标记监控和验收的关闭式失败要求。
#   _：本路由器不使用的扩展参数。
# 输出：
#   routing：角色、去重候选序列与失败要求。
def _resilient_router(
    *, requested_port: str, available_ports: list[str], role: str, **_: Any
) -> dict[str, object]:
    available_ports = _ports(available_ports)
    preferred = requested_port if requested_port in available_ports else "primary"
    fallbacks = [
        value
        for value in (preferred, "critic", "safety", "perception", "primary")
        if value in available_ports
    ]
    routing = {
        "role": role,
        "candidates": list(dict.fromkeys(fallbacks)),
        "fail_closed": role in {"execution_monitor", "completion_verifier"},
    }
    return routing


# 功能：
#   优先选名称以 local 开头的端口，无此端口时保留一个已配置的远端候选。
# 输入：
#   requested_port：没有本地候选时尝试使用的端口。
#   available_ports：当前可用端口名称列表。
#   role：本次调用角色。
#   _：本路由器不使用的扩展参数。
# 输出：
#   routing：角色、有序候选和关闭式失败标记。
def _privacy_first_router(
    *, requested_port: str, available_ports: list[str], role: str, **_: Any
) -> dict[str, object]:
    available_ports = _ports(available_ports)
    # 名称只是软偏好，不证明执行地点或隐私授权；实际联网准入由端口层负责。
    local = [value for value in available_ports if value.startswith("local")]
    candidates = local or (
        [requested_port]
        if requested_port in available_ports
        else (["primary"] if "primary" in available_ports else [])
    )
    routing = {"role": role, "candidates": candidates, "fail_closed": True}
    return routing


# 功能：
#   要求一个通过结构验证的响应，不要求多端口一致性。
# 输入：
#   _：单响应策略不使用的调度参数。
# 输出：
#   consensus：最少、最多响应数及一致性要求。
def _single_consensus(**_: Any) -> dict[str, object]:
    consensus = {"minimum_responses": 1, "require_identical": False, "maximum_responses": 1}
    return consensus


# 功能：
#   对审查、监控和验收要求双响应，其他角色使用单响应，并要求记录分歧。
# 输入：
#   role：用于确定响应数量的调用角色。
#   _：本策略不使用的扩展参数。
# 输出：
#   consensus：响应数量、一致性及分歧记录要求。
def _dual_consensus(*, role: str, **_: Any) -> dict[str, object]:
    critical = role.endswith("critic") or role in {
        "execution_monitor",
        "completion_verifier",
    }
    consensus = {
        "minimum_responses": 2 if critical else 1,
        "maximum_responses": 2 if critical else 1,
        "require_identical": False,
        "record_dissent": True,
    }
    return consensus


# 功能：
#   要求至少两个且最多三个结构化响应完全一致，由宿主执行数量与内容校验。
# 输入：
#   _：严格一致策略不使用的调度参数。
# 输出：
#   consensus：响应数量、一致性及分歧记录要求。
def _strict_consensus(**_: Any) -> dict[str, object]:
    consensus = {
        "minimum_responses": 2,
        "maximum_responses": 3,
        "require_identical": True,
        "record_dissent": True,
    }
    return consensus


# 功能：
#   校验供应商报告的词元计数，保留未报告状态，生成计量信息而不直接扣减账户额度。
# 输入：
#   record：带输入输出计数、供应商及模型标识的调用记录。
#   _：计量器不使用的扩展参数。
# 输出：
#   usage：输入、输出、可确定的总词元数及供应商和模型标识。
def _usage_meter(*, record: Any, **_: Any) -> dict[str, object]:
    input_tokens = record.input_tokens
    output_tokens = record.output_tokens
    for count in (input_tokens, output_tokens):
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("MODEL_USAGE_TOKENS_INVALID")
    # 未报告不能记为零；只有两个分量都存在时才计算总量，避免部分计数伪装完整用量。
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        ),
        "provider": record.provider,
        "model": record.model,
    }
    return usage


# 功能：
#   复核解码附件并提取本地图像引用；文件内容和摘要由后续模型传输层验证。
# 输入：
#   attachments：经过附件解码的材料列表。
#   _：预处理器不使用的扩展参数。
# 输出：
#   prepared：图像文件引用列表及其数量。
def _image_preprocessor(*, attachments: list[Any], **_: Any) -> dict[str, object]:
    if not isinstance(attachments, list) or len(attachments) > 32:
        raise ValueError("MODEL_IMAGE_ATTACHMENTS_INVALID")
    media: list[dict[str, object]] = []
    for attachment in attachments:
        attachment = AttachmentArtifact.model_validate(
            attachment.model_dump(mode="python"), strict=True
        )
        model_input = attachment.model_input
        if not isinstance(model_input, dict) or model_input.get("type") != "input_image_reference":
            continue
        path = model_input.get("source_path")
        if (
            attachment.decoded_kind != "image"
            or not attachment.content_type.startswith("image/")
            or not isinstance(path, str)
            or not path.strip()
            or "\x00" in path
            or len(path) > 32767
        ):
            raise ValueError("MODEL_IMAGE_REFERENCE_INVALID")
        media.append(
            {
                "kind": "image-file",
                "path": path,
                "content_type": attachment.content_type,
                "sha256": attachment.source_sha256,
            }
        )
    # 这里只整理引用，没有打开图片、提取视觉特征或证明文件仍与解码时相同。
    prepared = {"media": media, "count": len(media)}
    return prepared


# 功能：
#   注册互斥的路由及共识策略，并添加图像引用预处理和用量计量钩子。
# 输入：
#   无。
# 输出：
#   definitions：不含飞控执行权限的模型运行扩展定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    for order, (plugin_id, name, description, handler, enabled) in enumerate(
        [
            (
                "models.router-resilient",
                "弹性角色路由",
                "按角色选择独立端口，并在端口故障时按冻结顺序回退。",
                _resilient_router,
                True,
            ),
            (
                "models.router-privacy-first",
                "隐私优先路由",
                "优先选择本地端口；没有本地端口时保持单一受控连接。",
                _privacy_first_router,
                False,
            ),
        ],
        start=1,
    ):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.route",
                capability_kind="model-router",
                capability_name=name,
                capability_description=description,
                category_id="models",
                category_label="模型与推理",
                slot_id="models.runtime-router",
                slot_label="运行时模型路由",
                activation_mode="single",
                category_order=20,
                slot_order=30,
                plugin_order=order * 10,
                hooks={"route_model": handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
            )
        )
    for order, (plugin_id, name, description, handler, enabled) in enumerate(
        [
            (
                "models.consensus-single",
                "单响应",
                "使用一个经结构化验证的响应。",
                _single_consensus,
                True,
            ),
            (
                "models.consensus-dual-review",
                "双模型审查",
                "关键审查角色要求两个独立端口并记录分歧。",
                _dual_consensus,
                False,
            ),
            (
                "models.consensus-strict",
                "严格一致共识",
                "要求至少两个结构化结果完全一致，否则阻断。",
                _strict_consensus,
                False,
            ),
        ],
        start=1,
    ):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.select",
                capability_kind="consensus-policy",
                capability_name=name,
                capability_description=description,
                category_id="models",
                category_label="模型与推理",
                slot_id="models.consensus-policy",
                slot_label="模型共识策略",
                activation_mode="single",
                category_order=20,
                slot_order=40,
                plugin_order=order * 10,
                hooks={"select_consensus": handler},
                default_enabled=enabled,
                failure_mode="fail-closed",
            )
        )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="models.multimodal-images",
            name="图像多模态输入",
            description="把已解码图像作为受限视觉输入交给支持视觉的模型端口。",
            capability_id="models.multimodal-images.preprocess",
            capability_kind="multimodal-preprocessor",
            capability_name="图像多模态输入",
            capability_description="只接受经过附件边界验证的本地图像。",
            category_id="models",
            category_label="模型与推理",
            slot_id="models.multimodal-preprocessors",
            slot_label="多模态预处理",
            activation_mode="pipeline",
            category_order=20,
            slot_order=50,
            plugin_order=10,
            pipeline_order=10,
            hooks={"preprocess_multimodal": _image_preprocessor},
            default_enabled=True,
            failure_mode="isolate",
            permissions=["attachment.read"],
        )
    )
    definitions.append(
        hook_plugin(
            module_name=__name__,
            plugin_id="models.token-meter",
            name="模型用量计量",
            description="按真实供应商响应记录输入、输出与总 token。",
            capability_id="models.token-meter.measure",
            capability_kind="token-meter",
            capability_name="模型用量计量",
            capability_description="生成可审计的模型调用用量收据。",
            category_id="models",
            category_label="模型与推理",
            slot_id="models.token-meters",
            slot_label="用量计量",
            activation_mode="multiple",
            category_order=20,
            slot_order=60,
            plugin_order=10,
            hooks={"measure_tokens": _usage_meter},
            default_enabled=True,
            failure_mode="isolate",
        )
    )
    return definitions
