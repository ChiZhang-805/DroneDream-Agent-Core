"""Structured model context must retain field identity and valid JSON numbers."""

import pytest
from test_runtime_depth_context import _vehicle

from dronedream_agent_core.navigation_context import (
    bounded_navigation_context,
    build_navigation_context,
)


# 功能：
#   拒绝战略上下文中的非有限量，不把未知质量或状态改写成零。
# 输入：
#   value：非法浮点值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_context_numbers_are_rejected_at_the_boundary(value):
    with pytest.raises(ValueError, match="non-finite number"):
        bounded_navigation_context({"payload": {"dynamics": {"mass_kg": value}}})


# 功能：
#   字段名不能经强制转换或截断后相互覆盖。
# 输入：
#   fields：冲突或非法字段名集合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fields", [
    {"x" * 64 + "a": 1, "x" * 64 + "b": 2},
    {1: "numeric key", "1": "text key"},
    {"": 1},
])
def test_field_names_cannot_be_coerced_or_truncated_into_collisions(fields):
    with pytest.raises(ValueError, match="invalid field name"):
        bounded_navigation_context({"task": fields})


# 功能：
#   保留合法边界字段、零、缺失和布尔语义，并与调用方可变容器隔离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_exact_boundary_fields_and_valid_zero_missing_values_are_preserved():
    key = "x" * 64
    source = {"task": {key: [0, None, False, 1.25], "description": " a  b "}}
    result = bounded_navigation_context(source)
    assert result == {"task": {key: [0, None, False, 1.25], "description": "a b"}}
    source["task"][key].append(3)
    assert result["task"][key] == [0, None, False, 1.25]


# 功能：
#   顶层非对象必须产生可归类的值错误，不抛随机类型异常或被当空上下文接受。
# 输入：
#   value：不符合上下文契约的容器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [[], ["task"], 1, "task"])
def test_invalid_top_level_context(value):
    with pytest.raises(ValueError):
        bounded_navigation_context(value)


# 功能：
#   在正规化描述文字前限制输入规模，避免巨大整数和字符串拖慢控制准备。
# 输入：
#   value：超过局部上下文预算的数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [1 << 14000, "x" * 100_000], ids=["integer", "text"])
def test_context_work_is_bounded_before_normalization(value):
    with pytest.raises(ValueError):
        bounded_navigation_context({"task": {"value": value}})


# 功能：
#   深度局部合法但总节点过多的上下文不能无限展开。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_context_uses_a_shared_node_budget():
    wide = {str(index): [0] * 32 for index in range(24)}
    with pytest.raises(ValueError):
        bounded_navigation_context({"task": {str(index): wide for index in range(24)}})


# 功能：
#   运行注册信息不能覆盖固定的未知空间、坐标或视觉权限边界。
# 输入：
#   field、value：企图覆盖的固定传感器字段及不一致值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "value"), [
    ("unknown_space_policy", "free"), ("coordinate_frame", "NED"),
    ("rgb_role", "metric authority"),
])
def test_sensor_context_cannot_override_authority(field, value):
    with pytest.raises(ValueError):
        build_navigation_context(task={}, map_context={}, sensor_context={field: value},
            vehicle=_vehicle(), payload={}, rgb_enabled=True)


# 功能：
#   声明配置的上下文独立保存输入，不把配置名当成实测就绪证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_built_context_is_detached_and_preserves_sensor_health():
    task = {"phase": "TRACK"}
    result = build_navigation_context(task=task, map_context={},
        sensor_context={"ready_for_motion": False}, vehicle=_vehicle(), payload={},
        rgb_enabled=True)
    task["phase"] = "DONE"
    assert result["task"]["phase"] == "TRACK"
    assert result["sensor_contract"]["ready_for_motion"] is False


# 功能：
#   是否启用 RGB 只能接受布尔值，不能因字符串 false 非空而打开视觉声明。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rgb_configuration_requires_boolean():
    with pytest.raises(ValueError):
        build_navigation_context(task={}, map_context={}, sensor_context={}, vehicle=_vehicle(),
                                 payload={}, rgb_enabled="false")
