"""Synthetic disk receipts prove revision selection without running a drone or cloud model."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_mission_verification import _artifacts

from dronedream_agent_core.contracts import RuntimeReplacementTrack
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_interrupt import _atomic_json
from dronedream_agent_core.runtime_revision import RuntimeRevisionError, load_active_replacement

EXECUTION = "execution-" + "a" * 32


# 功能：
#   构造几何不变但执行身份明确的新计划，验证换计划不能仅比较坐标。
# 输入：
#   无。
# 输出：
#   replacement：包含任务图、检查点、动作合同和确定性门控的完整替换制品。
def _replacement():
    values = _artifacts()
    replacement = RuntimeReplacementTrack(
        message_id="runtime-msg-" + "b" * 32, execution_id=EXECUTION, replacement_sequence=1,
        message_sha256="c" * 64, hold_ack_sha256="d" * 64, decision_sha256="e" * 64,
        prior_track_sha256="f" * 64, target_node="target", return_node="office",
        route=values[5], clearance=values[6], track=values[7], revised_task_graph=values[2],
        runtime_checkpoints=values[8], runtime_actions=values[9],
        deterministic_gates={"bound": True}, generated_at=datetime.now(UTC),
    )
    return replacement


# 功能：
#   写入替换制品及匹配其全部合同摘要的模拟执行器采纳回执，不以模型提案代替采纳。
# 输入：
#   control_dir：隔离测试的控制目录。
#   replacement：本次模拟采纳的完整替换制品。
# 输出：
#   adoption：刚写入的绑定回执字典。
def _publish(control_dir, replacement):
    adoption = {
        "execution_id": replacement.execution_id, "message_id": replacement.message_id,
        "replacement_sequence": replacement.replacement_sequence,
        "replacement_sha256": sha256_json(replacement),
        "track_sha256": sha256_json(replacement.track),
        "task_graph_sha256": sha256_json(replacement.revised_task_graph),
        "runtime_checkpoints_sha256": sha256_json(replacement.runtime_checkpoints),
        "runtime_actions_sha256": sha256_json(replacement.runtime_actions),
    }
    _atomic_json(control_dir / "replacements" / f"{replacement.message_id}.json", replacement)
    _atomic_json(control_dir / "active-track.json", adoption)
    return adoption


# 功能：
#   验证仅有替换建议文件而没有执行器采纳回执时，原计划仍是活动计划。
# 输入：
#   tmp_path：测试控制目录。
# 输出：
#   None：不返回业务数据。
def test_a_proposal_without_adoption_is_not_active(tmp_path):
    replacement = _replacement()
    _atomic_json(tmp_path / "replacements" / f"{replacement.message_id}.json", replacement)
    assert load_active_replacement(tmp_path, execution_id=EXECUTION) is None


# 功能：
#   验证完整合法采纳回执返回它准确绑定的替换制品。
# 输入：
#   tmp_path：测试控制目录。
# 输出：
#   None：不返回业务数据。
def test_valid_adoption_selects_exact_revision(tmp_path):
    replacement = _replacement()
    _publish(tmp_path, replacement)
    assert load_active_replacement(tmp_path, execution_id=EXECUTION) == replacement


# 功能：
#   验证即使轨迹摘要相同，混入不同身份、序号或子合同摘要也不能通过采纳绑定。
# 输入：
#   tmp_path：测试控制目录。
#   update：对合法采纳回执的单项破坏。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("update", [
    {"execution_id": "execution-" + "f" * 32}, {"message_id": "../outside"},
    {"replacement_sequence": True}, {"replacement_sequence": 2},
    {"replacement_sha256": "0" * 64}, {"runtime_actions_sha256": "0" * 64},
    {"runtime_checkpoints_sha256": "0" * 64}, {"task_graph_sha256": "0" * 64},
])
def test_matching_track_alone_does_not_accept_a_mixed_revision(tmp_path, update):
    adoption = _publish(tmp_path, _replacement())
    _atomic_json(tmp_path / "active-track.json", {**adoption, **update})
    with pytest.raises(RuntimeRevisionError):
        load_active_replacement(tmp_path, execution_id=EXECUTION)


# 功能：
#   验证已存在的无效采纳回执必须报错，而不是退回原计划掩盖证据损坏。
# 输入：
#   tmp_path：测试控制目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_present_active_receipt_does_not_fall_back_to_original(tmp_path):
    _atomic_json(tmp_path / "active-track.json", [])
    with pytest.raises(RuntimeRevisionError):
        load_active_replacement(tmp_path, execution_id=EXECUTION)


# 功能：
#   验证未采纳替换中的新检查点不能触发模型调用或生成继续决定。
# 输入：
#   tmp_path：测试运行目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_cannot_select_a_replacement_that_was_only_proposed(tmp_path):
    import threading

    from test_checkpointing import _request

    from dronedream_agent_core.checkpointing import CheckpointCoordinator

    values, replacement = _artifacts(), _replacement()
    checkpoint = replacement.runtime_checkpoints.checkpoints[0]
    checkpoint.checkpoint_id = "checkpoint-100001"
    _atomic_json(tmp_path / "runtime-control" / "replacements" /
                 f"{replacement.message_id}.json", replacement)
    coordinator = object.__new__(CheckpointCoordinator)
    coordinator.prepared = SimpleNamespace(contract=values[0], plan=values[4],
                                           task_graph=values[2], px4_track=values[7])
    coordinator.contract, coordinator.run_dir = values[8], tmp_path
    coordinator.abort_file, coordinator.receipt_path = tmp_path / "abort.json", tmp_path / "hooks"
    coordinator._stop, coordinator.error, coordinator.decisions = threading.Event(), None, []
    coordinator.extensions = SimpleNamespace(
        invoke_multiple=lambda *a, **kw: ([], []),
        invoke_pipeline=lambda *args, **kw: (args[2], []),
    )
    calls = []

    # 功能：
    #   记录不应发生的模型调用，并立即停止错误执行的测试循环。
    # 输入：
    #   kwargs：协调器拟发送的模型参数。
    # 输出：
    #   result：不能被当成合法模型响应的哨兵对象。
    def forbidden_call(**kwargs):
        calls.append(kwargs)
        coordinator._stop.set()
        result = object()
        return result
    coordinator.port = SimpleNamespace(call=forbidden_call)
    request = _request().model_copy(update={"contract_id": values[0].contract_id,
                                           "checkpoint": checkpoint})
    _atomic_json(tmp_path / "checkpoints" / f"{checkpoint.checkpoint_id}.request.json", request)
    coordinator._run()
    assert not calls
    assert coordinator.error is not None
    assert not coordinator.decisions


# 功能：
#   验证活动采纳记录不能用重复身份键或非法额外值掩盖错误，即使最后字段能匹配合法方案。
# 输入：
#   tmp_path：完整替换计划和采纳回执所在目录。
#   extra：注入原始文档的歧义或非有限 JSON 字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("extra", [
    '"execution_id":"different-execution",',
    '"diagnostic":NaN,',
    '"diagnostic":1e999,',
    '"diagnostic":' + '[' * 65 + '0' + ']' * 65 + ',',
])
def test_active_revision_rejects_ambiguous_raw_adoption(tmp_path, extra):
    adoption = _publish(tmp_path, _replacement())
    raw = "{" + extra + json.dumps(adoption)[1:]
    (tmp_path / "active-track.json").write_text(raw, encoding="utf-8")
    with pytest.raises(RuntimeRevisionError):
        load_active_replacement(tmp_path, execution_id=EXECUTION)
