"""Offline checks of declared model routing and prompt boundaries, without API calls."""

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.model_port import ProviderSettings
from dronedream_agent_plugins import model_custom, model_deepseek, model_kimi, model_openai
from dronedream_agent_plugins.model_policies import _adversarial, _specialist, _unified
from dronedream_agent_plugins.model_runtime_plugins import (
    _dual_consensus,
    _ports,
    _single_consensus,
    _strict_consensus,
)
from dronedream_agent_plugins.prompt_packs import _append
from dronedream_agent_plugins.structured_output_plugins import _provenance_guard


# 功能：
#   验证产品目录的图像准入标记与端口配置一致，不访问真实端点或查询供应商最新能力。
# 输入：
#   module：当前被检查的提供方声明模块。
#   monkeypatch：隔离并恢复模型选择环境变量的测试夹具。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("module", [model_openai, model_deepseek, model_kimi])
def test_provider_catalog_agrees_with_product_transport_flags(module, monkeypatch):
    metadata = module.plugin_definition().manifest.capabilities[0].metadata
    provider = metadata["provider"]
    monkeypatch.setenv(f"{provider.upper()}_API_STYLE", "chat-completions")
    for model in metadata["models"]:
        monkeypatch.setenv(f"{provider.upper()}_MODEL", model["id"])
        settings = ProviderSettings.from_env(provider)
        assert settings.model == model["id"]
        assert settings.supports_image_input is model["supports_image_input"]


# 功能：
#   验证查询方修改一个模型目录不会改变后续查询，也不会触发密钥读取或在线调用。
# 输入：
#   module：被测提供方声明模块。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("module", [model_openai, model_deepseek, model_kimi, model_custom])
def test_provider_definitions_have_independent_metadata(module):
    definition = module.plugin_definition()
    metadata = definition.manifest.capabilities[0].metadata
    expected = [dict(item) for item in metadata["models"]]
    metadata["models"].append({"id": "not-configured"})
    assert module.plugin_definition().manifest.capabilities[0].metadata["models"] == expected


# 功能：
#   验证专用角色分配先匹配安全，其次感知和审查，其余任务交给主端口。
# 输入：
#   role：本例的调用角色。
#   expected：该角色应选择的端口名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "role,expected",
    [
        ("execution_monitor", "safety"),
        ("completion_verifier", "safety"),
        ("runtime_amendment_validator", "safety"),
        ("runtime_message_classifier", "safety"),
        ("scene_interpreter", "perception"),
        ("attachment_interpreter", "perception"),
        ("target_verifier", "perception"),
        ("intent_critic", "critic"),
        ("plan_critic", "critic"),
        ("global_planner", "primary"),
    ],
)
def test_specialist_roles_preserve_safety_precedence(role, expected):
    assert _specialist(role=role, requested_port="ignored") == {"port": expected}


# 功能：
#   验证统一策略只统一端口，对抗策略扩大审查角色但保留非审查角色原端口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unified_and_adversarial_policies_preserve_their_declared_boundaries():
    assert _unified(role="plan_critic") == {"port": "primary"}
    assert _adversarial(role="runtime_message_classifier", requested_port="safety") == {
        "port": "critic"
    }
    assert _adversarial(role="scene_interpreter", requested_port="perception") == {
        "port": "perception"
    }


# 功能：
#   验证共识策略返回具体响应要求，双响应仅对指定关键角色启用，不在钩子内调用模型。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_consensus_policies_request_bounded_role_specific_counts():
    assert _single_consensus() == {
        "minimum_responses": 1,
        "maximum_responses": 1,
        "require_identical": False,
    }
    assert _dual_consensus(role="global_planner")["minimum_responses"] == 1
    for role in ("plan_critic", "completion_verifier", "execution_monitor"):
        assert _dual_consensus(role=role)["minimum_responses"] == 2
        assert _dual_consensus(role=role)["record_dissent"] is True
    assert _strict_consensus() == {
        "minimum_responses": 2,
        "maximum_responses": 3,
        "require_identical": True,
        "record_dissent": True,
    }


# 功能：
#   验证端口列表拒绝错误容器、非法名称和过大目录，防止异常路由输入扩散。
# 输入：
#   ports：本例的非法端口列表或伪列表值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ports", [("primary",), [None], [" "], ["p" * 161], ["p"] * 65])
def test_router_rejects_invalid_port_declarations(ports):
    with pytest.raises(ValueError, match="PORTS_INVALID"):
        _ports(ports)


# 功能：
#   验证调用方修改原角色集合不会扩大已构造提示钩子的范围，空集合也不代表全部角色。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_prompt_role_capture_cannot_be_broadened_by_editor_mutation():
    roles = {"plan_critic"}
    augment = _append("Review constraints.", roles=roles)
    roles.add("global_planner")
    assert augment(value="Plan. ", role="global_planner") == "Plan. "
    assert "Review constraints." in augment(value="Review.", role="plan_critic")
    assert _append("Unused.", roles=set())(value="Original.", role="plan_critic") == "Original."


# 功能：
#   验证结构化输出门分别检查角色、Schema 名称及内容摘要，不接受错位回执。
# 输入：
#   update：覆盖正确回执的错误字段。
#   code：该字段应触发的拒绝标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update,code",
    [
        ({"role": "global_planner"}, "ROLE_MISMATCH"),
        ({"output_schema": "OtherSchema"}, "SCHEMA_BINDING_MISMATCH"),
        ({"output_sha256": "0" * 64}, "HASH_MISMATCH"),
    ],
)
def test_output_guard_checks_actual_role_schema_and_hash(update, code):
    artifact = {"accepted": True}
    envelope = {
        "artifact": artifact,
        "record": {
            "role": "plan_critic",
            "output_schema": "PlanCritique",
            "output_sha256": sha256_json(artifact),
        },
    }
    assert (
        _provenance_guard(value=envelope, role="plan_critic", expected_schema="PlanCritique")
        is envelope
    )
    envelope["record"].update(update)
    with pytest.raises(ValueError, match=code):
        _provenance_guard(value=envelope, role="plan_critic", expected_schema="PlanCritique")
