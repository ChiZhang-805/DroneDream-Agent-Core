"""Bounded reads and identity binding for desktop-issued runtime control authority."""

import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from dronedream_agent_core.contract_json import decode_contract_json
from dronedream_agent_core.contracts import (
    RuntimeControlSession,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeUserMessage,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import read_plugin_file

Record = TypeVar("Record", bound=BaseModel)


class RuntimeControlRecordError(ValueError):
    """Control evidence is absent, ambiguous or bound to another task."""


# 功能：
#   有界读取控制契约，拒绝链接、重复 JSON 键、非有限数和不符合契约的数据。
# 输入：
#   path：当前运行控制目录中的契约文件路径。
#   contract：期望的 Pydantic 契约类型。
# 输出：
#   record：经解析及类型校验的控制记录。
def read_control_record(path: Path, contract: type[Record]) -> Record:
    try:
        raw = read_plugin_file(path, limit=1024 * 1024)
        record = decode_contract_json(raw, contract, limit=1024 * 1024, node_limit=100_000)
    except (OSError, ValueError) as error:
        raise RuntimeControlRecordError("RUNTIME_CONTROL_RECORD_INVALID") from error
    return record


# 功能：
#   核对控制会话仍接受输入且属于指定任务；可进一步限定执行器会话标识。
# 输入：
#   control_dir：当前运行的控制目录。
#   thread_id：桌面任务标识。
#   execution_id：可选的执行器会话标识，与桌面宿主进程标识不是同一字段。
# 输出：
#   session：身份匹配且处于 accepting 状态的控制会话。
def read_active_session(control_dir: Path, thread_id: str, execution_id: str | None = None):
    session = read_control_record(control_dir / "session.json", RuntimeControlSession)
    if (
        session.state != "accepting"
        or session.conversation_id != thread_id
        or (execution_id is not None and session.execution_id != execution_id)
    ):
        raise RuntimeControlRecordError("RUNTIME_CONTROL_SESSION_MISMATCH")
    return session


# 功能：
#   读取指定消息及其悬停、决策记录，核对完整身份链；不替代执行器当前悬停检查。
# 输入：
#   control_dir：当前任务的运行控制目录。
#   thread_id：申请接管的桌面任务标识。
#   message_id：申请接管的运行消息标识。
# 输出：
#   evidence：会话、消息、悬停回执、分类决策和全部通过的授权门禁。
def read_takeover_evidence(control_dir: Path, thread_id: str, message_id: str):
    if not isinstance(message_id, str) or not re.fullmatch(r"runtime-msg-[0-9a-f]{32}", message_id):
        raise RuntimeControlRecordError("RUNTIME_CONTROL_MESSAGE_ID_INVALID")
    session = read_active_session(control_dir, thread_id)
    messages = []
    for folder in ("processed", "claimed", "inbox"):
        path = control_dir / folder / f"{message_id}.json"
        if path.exists():
            messages.append(read_control_record(path, RuntimeUserMessage))
    if not messages:
        raise RuntimeControlRecordError("RUNTIME_CONTROL_MESSAGE_NOT_FOUND")
    message = messages[0]
    message_hash = sha256_json(message)
    # 生命周期迁移期间可以短暂存在相同副本，但不允许按目录优先级挑选冲突消息。
    if any(sha256_json(item) != message_hash for item in messages[1:]):
        raise RuntimeControlRecordError("RUNTIME_CONTROL_MESSAGE_AMBIGUOUS")
    acknowledgement = read_control_record(
        control_dir / "acks" / f"{message_id}.json", RuntimeHoldAcknowledgement
    )
    decision = read_control_record(
        control_dir / "decisions" / f"{message_id}.json", RuntimeInterruptionDecision
    )
    gates = {
        "message_id_matches": message.message_id == message_id,
        "conversation_matches": message.conversation_id == session.conversation_id,
        "mission_matches": message.mission_id == session.mission_id,
        "plan_matches": message.plan_revision_id == session.plan_revision_id,
        "contract_matches": message.contract_id == session.contract_id,
        "session_execution_matches": message.execution_id == session.execution_id,
        "message_action_is_takeover": decision.classification.requested_action
        == "operator_takeover",
        "executor_is_holding": decision.authorized_action == "hold",
        "message_hash_matches": decision.message_sha256 == message_hash,
        "hold_hash_matches": decision.hold_ack_sha256 == sha256_json(acknowledgement),
        "hold_message_id_matches": acknowledgement.message_id == message_id,
        "hold_message_hash_matches": acknowledgement.message_sha256 == message_hash,
        "hold_gates_passed": all(acknowledgement.deterministic_gates.values()),
        "decision_gates_passed": bool(decision.authorization_gates)
        and all(decision.authorization_gates.values()),
        "execution_matches": message.execution_id == acknowledgement.execution_id,
    }
    if not all(gates.values()):
        failed = ",".join(name for name, passed in gates.items() if not passed)
        raise RuntimeControlRecordError(f"RUNTIME_CONTROL_BINDING_REJECTED:{failed}")
    evidence = session, message, acknowledgement, decision, gates
    return evidence
