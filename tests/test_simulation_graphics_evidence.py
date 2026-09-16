import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.simulation_graphics_evidence import graphics_request_evidence


# 功能：
#   白名单只记录请求，不泄露其他凭证，也不宣称实际显卡或飞行资格已验证。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_requested_adapter_does_not_claim_actual_gpu_or_leak_other_environment():
    env = {"GALLIUM_DRIVER": "llvmpipe", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
           "GALLIVM_PERF": "nopt", "API_KEY": "not-for-logs", "AUTH_TOKEN": "private"}
    original = dict(env)
    evidence = graphics_request_evidence(env)
    requested = evidence["requested_environment"]
    assert env == original
    assert requested["GALLIUM_DRIVER"] == "llvmpipe"
    assert requested["GALLIVM_PERF"] == "nopt"
    assert requested["MESA_SHADER_CACHE_DISABLE"] is None
    assert "API_KEY" not in requested and "AUTH_TOKEN" not in requested
    assert evidence["requested_environment_sha256"] == sha256_json(requested)
    assert not evidence["actual_driver_verified"]
    assert not evidence["image_equivalence_claimed"]
    assert not evidence["flight_qualification_granted"]


# 功能：
#   编译配置变化需改变摘要；旧证据保持独立，空字符串与未设置分别计入身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_changed_compiler_settings_change_identity_without_mutating_existing_evidence():
    env = {"GALLIVM_PERF": "nopt"}
    first = graphics_request_evidence(env)
    env["GALLIVM_PERF"] = ""
    second = graphics_request_evidence(env)
    assert first["requested_environment"]["GALLIVM_PERF"] == "nopt"
    assert first["requested_environment_sha256"] != second["requested_environment_sha256"]
    assert second["requested_environment_sha256"] != graphics_request_evidence({})[
        "requested_environment_sha256"]


# 功能：
#   拒绝非文本与超长图形参数，防止不合法设置进入启动证据。
# 输入：
#   value：故意构造的非法配置值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [1, "x" * 2049], ids=["not-text", "oversized"])
def test_invalid_graphics_value_rejected_before_start(value):
    with pytest.raises(ValueError, match="SIMULATION_GRAPHICS_REQUEST_INVALID"):
        graphics_request_evidence({"GALLIVM_PERF": value})
