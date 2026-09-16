"""Context/fault tests only; these labels cannot confirm a landed aircraft."""

import json

import pytest

from dronedream_agent_core.runtime_phase import (
    ENDING_PHASES,
    MAXIMUM_PHASE_BYTES,
    phase_context,
    runtime_phase_context,
)


# 功能：
#   非法阶段容器、超长文字和错误类型均变为未知，不猜测活跃状态或控制权限。
# 输入：
#   payload：候选阶段对象。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload", [None, [], "FAILED", 3, {"phase": ["FAILED"]},
    {"phase": "x" * 129}, {"enclosing_executor_state": {"phase": False}}])
def test_malformed_context_is_unknown_not_permission(payload):
    assert phase_context(payload) == {
        "phase": "UNKNOWN", "executor_phase": "UNKNOWN", "checkpoint_id": None}


# 功能：
#   外层终止标签优先于旧的内部活跃标签，不能因暂停上下文而重新激活任务。
# 输入：
#   ending：每一种终止阶段标签。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ending", sorted(ENDING_PHASES))
def test_ending_label_wins_over_contradictory_enclosing_active_state(ending):
    assert phase_context({"phase": ending,
        "enclosing_executor_state": {"phase": "TRACK"}})["executor_phase"] == ending


# 功能：
#   只提取阶段与检查点，不复制输入里自报的落地或推进权限。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_context_does_not_copy_control_or_ground_permissions(tmp_path):
    payload = {"phase": "MODEL_AUTHORITY_HOLD", "checkpoint_id": "pickup",
        "enclosing_executor_state": {"phase": "TRACK"},
        "landing_confirmed": True, "schedule_advancement_authorized": True}
    path = tmp_path / "phase.json"
    path.write_text(json.dumps(payload))
    assert runtime_phase_context(path) == {
        "phase": "MODEL_AUTHORITY_HOLD", "executor_phase": "TRACK", "checkpoint_id": "pickup"}


# 功能：
#   文件缺失、损坏、过深或超量时不能合成任务完成状态。
# 输入：
#   tmp_path：测试私有目录。
#   content：待写入的错误内容；None 表示文件缺失。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", [None, b"invalid", b"\xff", b"[]", b"{" * 2000,
    b'{"phase":"COMPLETE","extra":"' + b"x" * MAXIMUM_PHASE_BYTES + b'"}'],
    ids=["missing", "invalid", "non-unicode", "array", "nested", "oversize"])
def test_missing_invalid_or_oversize_phase_cannot_synthesize_completion(tmp_path, content):
    path = tmp_path / "phase.json"
    if content is not None:
        path.write_bytes(content)
    assert runtime_phase_context(path) == phase_context(None)
