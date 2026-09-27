"""Desktop projection regressions: domain actions, budgets and immutable bytes."""
from types import SimpleNamespace
import pytest
from dronedream_agent_app.mission_service import _public_planner_artifact


# 功能：
#   验证实际轮数与配置预算分离，取餐插件动作保持原样。
# 输入：
#   attempts、maximum：参数化轮数和上限。
# 输出：
#   无返回值。
@pytest.mark.parametrize("attempts,maximum", [(1, 1), (1, 5), (3, 5), (5, 5)])
def test_projection_preserves_budget_and_plugin_actions(attempts, maximum):
    prepared = SimpleNamespace(
        planning_attempts=attempts,
        task_graph=SimpleNamespace(nodes=[SimpleNamespace(task_id="hold", action="delivery.precontact-hold", target_node="verified-036", depends_on=[], success_evidence=["stable hover"])]),
        intent=SimpleNamespace(goal="取餐", start_entity="office", target_entity="pickup", return_entity="office", assumptions=[]),
        tool_receipts=[],
    )
    artifact = _public_planner_artifact(prepared=prepared, input_metadata={"public_harness_context_sha256": "a"*64}, public_map_id="map", public_map_version=1, public_aircraft_id="vehicle", public_aircraft_version=1, maximum_planning_rounds=maximum)
    assert artifact["repair"]["attempt"] == attempts
    assert artifact["repair"]["max_attempts"] == maximum
    assert artifact["task_graph"]["nodes"][0]["action"] == "delivery.precontact-hold"
    assert artifact["safety_policy"]["actuator_authority"] is False


# 功能：
#   拒绝无法解释的预算，不能将超预算任务投影为有效计划。
# 输入：
#   attempts、maximum：非法轮数和上限。
# 输出：
#   无返回值。
@pytest.mark.parametrize("attempts,maximum", [(0, 5), (6, 5), (1, 0), (1, 6)])
def test_projection_rejects_invalid_budget(attempts, maximum):
    with pytest.raises(ValueError, match="ATTEMPT_BUDGET"):
        _public_planner_artifact(prepared=SimpleNamespace(planning_attempts=attempts), input_metadata={}, public_map_id="map", public_map_version=1, public_aircraft_id="vehicle", public_aircraft_version=1, maximum_planning_rounds=maximum)
