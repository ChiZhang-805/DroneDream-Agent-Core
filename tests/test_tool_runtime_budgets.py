import pytest
from pydantic import BaseModel

from dronedream_agent_core.tools import ToolPlugin, ToolRegistry


class _Value(BaseModel):
    value: int


# 功能：
#   创建绑定指定插槽的惰性规划工具，使测试仅观察调用预算而不执行路线计算。
# 输入：
#   tool_id：测试工具标识。
#   slot_id：用于判定预算豁免的插槽标识。
# 输出：
#   plugin：原样返回输入的工具定义。
def _tool(tool_id: str, slot_id: str) -> ToolPlugin[_Value, _Value]:
    plugin = ToolPlugin(
        tool_id=tool_id,
        version="1.0.0",
        authority="plan",
        input_type=_Value,
        output_type=_Value,
        handler=lambda value: value,
        slot_id=slot_id,
    )
    return plugin


# 功能：
#   验证可选建议工具耗尽自身预算后，保留的路线工具仍可使用独立总预算继续执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_optional_budget_cannot_starve_reserved_planning_tools() -> None:
    registry = ToolRegistry(allowed_authorities={"plan"})
    registry.register(_tool("optional", "mission.advice"))
    registry.register(_tool("route", "planning.route-strategy"))
    registry.configure_runtime_limits(
        maximum_calls=1,
        maximum_total_calls=4,
        timeout_seconds=1,
        budget_exempt_slot_ids={"planning.route-strategy"},
    )

    registry.call("optional", _Value(value=1))
    with pytest.raises(RuntimeError, match="HARNESS_OPTIONAL_TOOL_CALL_BUDGET_EXCEEDED"):
        registry.call("optional", _Value(value=2))

    output, _receipt = registry.call("route", _Value(value=3))
    assert output == _Value(value=3)


# 功能：
#   验证保留插槽不能无限调用，即使完全豁免可选预算也受任务总次数限制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reserved_tools_still_obey_independent_total_ceiling() -> None:
    registry = ToolRegistry(allowed_authorities={"plan"})
    registry.register(_tool("route", "planning.route-strategy"))
    registry.configure_runtime_limits(
        maximum_calls=0,
        maximum_total_calls=1,
        timeout_seconds=1,
        budget_exempt_slot_ids={"planning.route-strategy"},
    )

    registry.call("route", _Value(value=1))
    with pytest.raises(RuntimeError, match="HARNESS_TOTAL_TOOL_CALL_BUDGET_EXCEEDED"):
        registry.call("route", _Value(value=2))
