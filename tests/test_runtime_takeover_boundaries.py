"""Desktop takeover boundary tests; no flight, WSL process or model API is started."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dronedream_agent_app.runtime_manager import RuntimeBridgeError
from dronedream_agent_core.contracts import (
    ModelCallRecord,
    RuntimeControlSession,
    RuntimeInterruptionDecision,
    RuntimeMessageClassification,
)
from dronedream_agent_core.hashing import sha256_json
from tests.test_runtime_interrupt import _ack, _message
from tests.test_runtime_manager import _manager


# 功能：
#   创建身份相互匹配的接管材料和受控活动进程，不调用真实运行环境。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   fixture：管理器、任务标识、控制目录、消息、悬停回执及分类决策。
def _takeover_fixture(tmp_path):
    manager = _manager(tmp_path / "store", tmp_path / "resources")
    thread_id = manager.store.create_thread("takeover boundary", "gpt-5.4")["thread_id"]
    message = _message("请求人工接管").model_copy(update={"conversation_id": thread_id})
    acknowledgement = _ack(message)
    now = datetime.now(UTC)
    session = RuntimeControlSession(
        conversation_id=thread_id,
        mission_id=message.mission_id,
        plan_revision_id=message.plan_revision_id,
        contract_id=message.contract_id,
        execution_id=message.execution_id,
        prepared_mission_sha256="f" * 64,
        created_at=now,
    )
    classification = RuntimeMessageClassification(
        message_kind="motion_adjustment",
        requested_action="operator_takeover",
        requires_plan_revision=False,
        summary="Request a separate operator grant.",
    )
    decision = RuntimeInterruptionDecision(
        message_sha256=sha256_json(message),
        hold_ack_sha256=sha256_json(acknowledgement),
        classification=classification,
        model_call=ModelCallRecord(
            call_id="model-" + "e" * 24,
            role="runtime_message_classifier",
            attempt=1,
            input_sha256="a" * 64,
            output_sha256=sha256_json(classification),
            output_schema="RuntimeMessageClassification",
            provider="openai",
            model="fixture-only",
            latency_ms=10,
            created_at=now,
        ),
        authorized_action="hold",
        authorization_gates={"stable": True},
        decision_reason="Separate authenticated grant required.",
    )
    control_dir = tmp_path / "run" / "runtime-control"
    manager._active_runs[thread_id] = SimpleNamespace(
        execution_id="execution-" + "1" * 32,
        run_dir=control_dir.parent,
        process=SimpleNamespace(poll=lambda: None),
    )
    manager._atomic_json(control_dir / "session.json", session)
    manager._atomic_json(control_dir / "processed" / f"{message.message_id}.json", message)
    manager._atomic_json(control_dir / "acks" / f"{message.message_id}.json", acknowledgement)
    manager._atomic_json(control_dir / "decisions" / f"{message.message_id}.json", decision)
    fixture = manager, thread_id, control_dir, message, acknowledgement, decision
    return fixture


# 功能：
#   为同一测试消息申请短期授权并组装一条合法速度指令的输入参数。
# 输入：
#   fixture：接管测试的完整材料。
#   operator_id：申请接管的用户标识。
# 输出：
#   arguments：提交控制指令使用的具名参数。
def _control_arguments(fixture, operator_id="owner"):
    manager, thread_id, _directory, message, _acknowledgement, _decision = fixture
    grant = manager.issue_takeover_grant(
        thread_id, message_id=message.message_id, operator_id=operator_id, duration_seconds=60
    )
    arguments = dict(
        thread_id=thread_id,
        operator_id=operator_id,
        message_id=message.message_id,
        grant_token=grant["grant_token"],
        action="velocity",
        north_mps=0.5,
        east_mps=0.0,
        down_mps=0.0,
        yaw_rate_dps=0.0,
        duration_seconds=0.25,
    )
    return arguments


# 功能：
#   验证会话身份、悬停消息和决策门禁任一不匹配时都不能发放接管授权。
# 输入：
#   tmp_path：隔离测试目录。
#   defect：要注入的单个证据缺陷。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "defect", ["closed", "conversation", "execution", "ack_id", "ack_hash", "decision_gate"]
)
def test_takeover_rejects_mismatched_evidence(tmp_path, defect):
    manager, thread_id, directory, message, acknowledgement, decision = _takeover_fixture(tmp_path)
    if defect in {"closed", "conversation", "execution"}:
        path = directory / "session.json"
        session = RuntimeControlSession.model_validate_json(path.read_bytes())
        changes = {
            "closed": {"state": "closed"},
            "conversation": {"conversation_id": "different-task"},
            "execution": {"execution_id": "execution-" + "2" * 32},
        }
        manager._atomic_json(path, session.model_copy(update=changes[defect]))
    elif defect in {"ack_id", "ack_hash"}:
        changes = (
            {"message_id": "runtime-msg-" + "2" * 32}
            if defect == "ack_id"
            else {"message_sha256": "2" * 64}
        )
        acknowledgement = acknowledgement.model_copy(update=changes)
        manager._atomic_json(directory / "acks" / f"{message.message_id}.json", acknowledgement)
        # 决策引用被替换回执的新哈希，避免把缺陷退化为单纯哈希不匹配。
        decision = decision.model_copy(update={"hold_ack_sha256": sha256_json(acknowledgement)})
    else:
        decision = decision.model_copy(update={"authorization_gates": {"stable": False}})
    manager._atomic_json(directory / "decisions" / f"{message.message_id}.json", decision)
    with pytest.raises(RuntimeBridgeError, match="TAKEOVER"):
        manager.issue_takeover_grant(
            thread_id, message_id=message.message_id, operator_id="owner", duration_seconds=60
        )
    assert not manager._takeover_grants
    assert not (directory / "takeover-grants" / f"{message.message_id}.json").exists()


# 功能：
#   验证同一授权支持中文用户标识，并保持以速度而非位置为控制输出。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_takeover_supports_unicode_operator_identity(tmp_path):
    fixture = _takeover_fixture(tmp_path)
    arguments = _control_arguments(fixture, operator_id="开发者账户")
    accepted = fixture[0].submit_operator_control(**arguments)
    assert accepted == {"accepted": True, "sequence": 1, "action": "velocity"}


# 功能：
#   验证发布失败不消耗序号，恢复后的首条控制指令仍可被执行器按序读取。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：注入一次文件发布失败的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_command_write_does_not_skip_sequence(tmp_path, monkeypatch):
    fixture = _takeover_fixture(tmp_path)
    manager, thread_id = fixture[:2]
    arguments = _control_arguments(fixture)
    original = manager._atomic_json

    # 功能：
    #   在指令写入前模拟磁盘故障。
    # 输入：
    #   args：位置参数。
    #   kwargs：具名参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_write(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(manager, "_atomic_json", fail_write)
    with pytest.raises(OSError):
        manager.submit_operator_control(**arguments)
    assert manager._takeover_grants[thread_id].next_sequence == 1
    monkeypatch.setattr(manager, "_atomic_json", original)
    assert manager.submit_operator_control(**arguments)["sequence"] == 1


# 功能：
#   验证发放授权后会话关闭，不再接受后续速度指令。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_operator_command_rechecks_current_session(tmp_path):
    fixture = _takeover_fixture(tmp_path)
    manager, thread_id, directory = fixture[:3]
    arguments = _control_arguments(fixture)
    session = RuntimeControlSession.model_validate_json((directory / "session.json").read_bytes())
    manager._atomic_json(directory / "session.json", session.model_copy(update={"state": "closed"}))
    with pytest.raises(RuntimeBridgeError, match="TAKEOVER"):
        manager.submit_operator_control(**arguments)
    assert manager._takeover_grants[thread_id].next_sequence == 1


# 功能：
#   验证同一接管消息不能被重复发放另一份令牌而覆盖已发布授权。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_takeover_grant_is_one_time(tmp_path):
    fixture = _takeover_fixture(tmp_path)
    arguments = _control_arguments(fixture)
    manager, thread_id, directory, message = fixture[:4]
    path = directory / "takeover-grants" / f"{message.message_id}.json"
    previous = path.read_bytes()
    with pytest.raises(RuntimeBridgeError, match="TAKEOVER"):
        manager.issue_takeover_grant(
            thread_id, message_id=message.message_id, operator_id="owner", duration_seconds=60
        )
    assert path.read_bytes() == previous
    assert manager.submit_operator_control(**arguments)["sequence"] == 1


# 功能：
#   验证非法控制数值被拒绝且不消耗序号，不将 NaN 或超大整数送入控制契约。
# 输入：
#   tmp_path：隔离测试目录。
#   value：非法北向速度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), 10**1000],
    ids=["bool", "nan", "infinite", "overflow"],
)
def test_operator_command_rejects_invalid_numbers(tmp_path, value):
    fixture = _takeover_fixture(tmp_path)
    arguments = _control_arguments(fixture)
    arguments["north_mps"] = value
    with pytest.raises(RuntimeBridgeError, match="NONFINITE_OR_INVALID"):
        fixture[0].submit_operator_control(**arguments)
    assert fixture[0]._takeover_grants[fixture[1]].next_sequence == 1


# 功能：
#   验证控制材料的重复键和同名不同内容副本不能通过目录优先级掩盖。
# 输入：
#   tmp_path：隔离测试目录。
#   defect：重复会话键或冲突消息副本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("defect", ["duplicate_key", "conflicting_copy"])
def test_takeover_rejects_ambiguous_files(tmp_path, defect):
    manager, thread_id, directory, message, _acknowledgement, _decision = _takeover_fixture(
        tmp_path
    )
    if defect == "duplicate_key":
        path = directory / "session.json"
        path.write_bytes(path.read_bytes().replace(b"{", b'{"state":"closed",', 1))
    else:
        manager._atomic_json(
            directory / "inbox" / f"{message.message_id}.json",
            message.model_copy(update={"text": "different request"}),
        )
    with pytest.raises(RuntimeBridgeError, match="TAKEOVER_GRANT_REJECTED"):
        manager.issue_takeover_grant(
            thread_id, message_id=message.message_id, operator_id="owner", duration_seconds=60
        )


# 功能：
#   验证越界消息标识在构造控制文件路径前即被拒绝。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_takeover_rejects_path_in_message_id(tmp_path):
    fixture = _takeover_fixture(tmp_path)
    with pytest.raises(RuntimeBridgeError, match="MESSAGE_ID_INVALID"):
        fixture[0].issue_takeover_grant(
            fixture[1], message_id="../../outside", operator_id="owner", duration_seconds=60
        )
