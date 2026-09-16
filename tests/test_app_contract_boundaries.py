"""HTTP validation must reject ambiguous consent and control values before storage."""

import pytest
from pydantic import ValidationError

from dronedream_agent_app.models import (
    AccountMemoryCandidateDecisionRequest,
    ConnectorCredentialCreateRequest,
    CustomModelDiscoverRequest,
    HarnessRevisionActionRequest,
    MessageCreate,
    MissionPrepareRequest,
    ModelRoleConnectionRequest,
    OperatorControlRequest,
    PluginConfigurationRequest,
    PluginMarketplaceInstallRequest,
    PluginRollbackRequest,
    SettingsPatch,
    ThreadPatch,
)


# 功能：
#   验证 HTTP 记忆开关只接受布尔，不把文本、整数或空值当成同意。
# 输入：
#   field：记忆开关名。
#   value：非法开关值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["false", 1, None])
@pytest.mark.parametrize(
    "field", ["memory_enabled", "remember_asset_choices", "remember_task_preferences"]
)
def test_http_memory_consent_is_a_real_boolean(field, value):
    with pytest.raises(ValidationError):
        SettingsPatch.model_validate({field: value})


# 功能：
#   验证任务必填字段在请求阶段拒绝显式清空，避免进入 SQL 后才失败。
# 输入：
#   field：不可为空的任务字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["pinned", "archived", "title", "selected_model", "locale"])
def test_nonnullable_thread_fields_cannot_be_cleared(field):
    with pytest.raises(ValidationError):
        ThreadPatch.model_validate({field: None})


# 功能：
#   验证严格空值规则不阻止清除可选资产，且未传入字段仍保持未更新语义。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_asset_selection_can_still_be_explicitly_cleared():
    patch = ThreadPatch.model_validate({"selected_map_id": None})
    assert patch.model_dump(exclude_unset=True) == {"selected_map_id": None}
    assert SettingsPatch().model_dump(exclude_unset=True) == {}


# 功能：
#   验证任意嵌套元数据中的非有限数不能绕过普通数值字段的校验。
# 输入：
#   bad：非有限浮点数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_arbitrary_metadata_does_not_bypass_finite_json(bad):
    with pytest.raises(ValidationError):
        MessageCreate(content="test", metadata={"nested": [bad]})


# 功能：
#   构造深层字典验证深度限制，防止解析后结构继续造成无限递归处理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_metadata_depth_is_bounded():
    metadata = {}
    for _ in range(64):
        metadata = {"next": metadata}
    with pytest.raises(ValidationError):
        MessageCreate(content="test", metadata=metadata)


# 功能：
#   验证操纵速度不从布尔或数字字符串转换，控制请求必须使用明确数值。
# 输入：
#   bad：非法速度值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [True, "0.5"])
def test_control_velocity_is_numeric_not_coerced(bad):
    with pytest.raises(ValidationError):
        OperatorControlRequest(
            message_id="runtime-msg-" + "a" * 32, grant_token="b" * 32, north_mps=bad
        )


# 功能：
#   验证 Harness 修订号不能利用 Python 布尔属于整数的关系绕过类型检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_revision_cannot_use_boolean_as_integer():
    with pytest.raises(ValidationError):
        HarnessRevisionActionRequest(base_revision=True)


# 功能：
#   验证记忆重新同意不接受文本 true，不能由模糊字符串扩大账户权限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reconsent_cannot_be_the_string_true():
    with pytest.raises(ValidationError):
        AccountMemoryCandidateDecisionRequest(
            source_edition="autonomy", thread_id="thread-" + "a" * 32, explicit_reconsent="true"
        )


# 功能：
#   验证插件版本按完整身份匹配，不能携带额外尾部或目录穿越内容。
# 输入：
#   version：非法版本字符串。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("version", ["1.2.3garbage", "1.2.3/../other"])
def test_plugin_version_must_match_the_entire_identity(version):
    with pytest.raises(ValidationError):
        PluginRollbackRequest(version=version)
    with pytest.raises(ValidationError):
        PluginMarketplaceInstallRequest(source_id="source", plugin_id="plugin", version=version)


# 功能：
#   验证完整匹配仍保留产品支持的预发布与构建后缀，避免只允许三个数字的回归。
# 输入：
#   version：受支持的版本字符串。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("version", ["1.2.3", "1.2.3-preview.1", "1.2.3+local"])
def test_supported_plugin_version_suffixes_are_retained(version):
    assert PluginRollbackRequest(version=version).version == version


# 功能：
#   验证两份同角色连接在规划请求中明确报错，不发生隐式覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_preparation_cannot_silently_replace_a_duplicate_role():
    connection = {"role_port": "critic", "model_id": "local", "model_grant": "ddg_" + "a" * 32}
    with pytest.raises(ValidationError, match="MODEL_ROLE_DUPLICATE"):
        MissionPrepareRequest(
            source_edition="autonomy",
            message="Inspect the lobby",
            map_id="map",
            vehicle_id="vehicle",
            model_id="local",
            model_grant="ddg_" + "b" * 32,
            role_models=[connection, connection],
        )


# 功能：
#   验证任意元数据不能携带 JSON 无法表达的对象，避免 HTTP 校验成功而存储失败。
# 输入：
#   bad：非标准 JSON 对象或会发生键合并的映射。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [object(), {"a"}, (1, 2), {1: "a", "1": "b"}])
def test_metadata_rejects_non_json_values(bad):
    with pytest.raises(ValueError):
        MessageCreate(content="fixture", metadata={"nested": bad})
    with pytest.raises(ValueError):
        PluginConfigurationRequest(configuration={"nested": bad})


# 功能：
#   验证大字符串同样受 UTF-8 字节预算限制，不能靠节点数少绕过请求容量检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_metadata_string_bytes_are_bounded():
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        MessageCreate(content="fixture", metadata={"blob": "x" * (4 * 1024 * 1024)})


# 功能：
#   验证凭证字符原样保留，通用文本去空格规则不能改变用户提供的实际秘密。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_credentials_are_not_silently_trimmed():
    connector = ConnectorCredentialCreateRequest(
        display_name=" fixture ", secret=" secret with spaces ", allowed_plugin_ids=["model.openai"]
    )
    discovery = CustomModelDiscoverRequest(
        base_url="https://example.invalid/v1", api_key=" padded-key "
    )
    assert connector.display_name == "fixture"
    assert connector.secret == " secret with spaces "
    assert discovery.api_key == " padded-key "
    assert connector.secret not in repr(connector)
    assert discovery.api_key not in repr(discovery)


# 功能：
#   验证标准 JSON 元数据与已验证的角色子模型仍能使用，严格编码不误拒合法内部调用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_valid_metadata_and_typed_role_models_remain_supported():
    metadata = {"note": "地图入口", "observations": [None, True, 2, 0.5]}
    assert MessageCreate(content="fixture", metadata=metadata).metadata == metadata
    role = ModelRoleConnectionRequest(
        role_port="critic", model_id="fixture", model_grant="ddg_" + "a" * 32
    )
    request = MissionPrepareRequest(
        source_edition="autonomy",
        message="Inspect the lobby",
        map_id="map",
        vehicle_id="vehicle",
        model_id="fixture",
        model_grant="ddg_" + "b" * 32,
        role_models=[role],
        input_metadata=metadata,
    )
    assert request.role_models[0].role_port == "critic"
    assert request.input_metadata == metadata


# 功能：
#   验证任务输入通道的自由元数据也受到相同 JSON 类型检查，不只覆盖普通消息。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mission_input_metadata_uses_same_json_boundary():
    with pytest.raises(ValueError, match="TYPE_INVALID"):
        MissionPrepareRequest(
            source_edition="autonomy",
            message="Inspect the lobby",
            map_id="map",
            vehicle_id="vehicle",
            model_id="fixture",
            model_grant="ddg_" + "b" * 32,
            input_metadata={"custom_object": object()},
        )
