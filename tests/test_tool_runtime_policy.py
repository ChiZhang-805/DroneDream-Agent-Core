from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from dronedream_agent_core.tools import ToolExecutionError, ToolPlugin, ToolRegistry


class _Payload(BaseModel):
    value: int


# 功能：
#   为预算与超时测试创建具有整数输入输出的只读工具，不连接外部系统。
# 输入：
#   handler：本例的本地处理器。
# 输出：
#   registry：只包含测试工具的注册表。
def _registry(handler) -> ToolRegistry:
    registry = ToolRegistry(allowed_authorities={"read"})
    registry.register(
        ToolPlugin(
            tool_id="test.policy-tool",
            version="1.0.0",
            authority="read",
            input_type=_Payload,
            output_type=_Payload,
            handler=handler,
        )
    )
    return registry


# 功能：
#   验证第一次成功调用会消耗任务预算，后续调用不能因为输入不同而绕过额度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tool_registry_enforces_per_mission_call_budget() -> None:
    registry = _registry(lambda value: value)
    registry.configure_runtime_limits(maximum_calls=1, timeout_seconds=1)

    result, _receipt = registry.call("test.policy-tool", {"value": 1})

    assert result == _Payload(value=1)
    with pytest.raises(RuntimeError, match="HARNESS_OPTIONAL_TOOL_CALL_BUDGET_EXCEEDED"):
        registry.call("test.policy-tool", {"value": 2})


# 功能：
#   验证处理器超过等待时限后返回明确失败回执，不标记为工具成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tool_registry_enforces_runtime_timeout() -> None:
    # 功能：
    #   等待超过本例配置的工具时限，再返回输入，模拟迟到的本地计算结果。
    # 输入：
    #   value：待原样返回的整数模型。
    # 输出：
    #   value：等待后的原模型。
    def slow(value: _Payload) -> _Payload:
        time.sleep(0.15)
        return value

    registry = _registry(slow)
    registry.configure_runtime_limits(maximum_calls=2, timeout_seconds=0.1)

    with pytest.raises(ToolExecutionError) as raised:
        registry.call("test.policy-tool", {"value": 1})

    assert raised.value.receipt.issue_codes == ["TOOL_EXECUTION_FAILED:TimeoutError"]
