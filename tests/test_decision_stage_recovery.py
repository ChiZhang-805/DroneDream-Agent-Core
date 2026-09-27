"""All five stage behaviors share execution audit but retain distinct outcome requirements."""

import pytest
from test_decision_stage_wait import wait_fixture

from dronedream_agent_core.decision_label_evidence import verify_native_label
from dronedream_agent_core.decision_stage_control import verify_stage_control


# 功能：补观测和重规划可以记录真实保持，但保持本身不意味着恢复/重规划已经成功。
# 输入：只具有真实格式保持账本的合成夹具；输出：控制审计通过、完整标签仍拒绝。
@pytest.mark.parametrize("action", ["request_observation", "replan"])
def test_stationary_stage_requires_its_actual_result(action):
    row, document = wait_fixture()
    document.pop("continuation")
    document["behavior_application"]["action"] = action
    for selected in document["behavior_selections"]:
        selected["action"] = action
    control = verify_stage_control(row, document)
    assert control["selected_hold"] > 0
    assert control["model"] == 0
    with pytest.raises((ValueError, KeyError)):
        verify_native_label(row, document)


# 功能：五类动作之外的新名字不能隐式取得阶段执行资格；输入：错误动作名；输出：拒绝。
def test_unknown_stage_action_stays_rejected():
    row, document = wait_fixture()
    document["behavior_application"]["action"] = "ignore_obstacle"
    with pytest.raises(ValueError, match="ACTION_NOT_SUPPORTED"):
        verify_stage_control(row, document)
