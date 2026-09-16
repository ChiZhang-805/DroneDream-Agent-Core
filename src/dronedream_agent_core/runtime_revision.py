"""Resolve the executor's adopted revision; a merely proposed replacement is not active."""

from __future__ import annotations

import re
from pathlib import Path

from .contracts import RuntimeControlSession, RuntimeReplacementTrack
from .hashing import sha256_json
from .plugin_files import check_plain_plugin_path
from .runtime_control_io import read_runtime_object
from .runtime_plugins import all_required_gates_passed


class RuntimeRevisionError(RuntimeError):
    """Active execution evidence is missing, malformed or internally inconsistent."""


# 功能：
#   有界读取普通证据文件并拒绝链接、替换、歧义键、非有限值和过深结构，统一报告损坏回执。
# 输入：
#   path：当前任务的采纳记录或替换制品路径。
#   maximum_bytes：该类证据允许读取的最大字节数。
# 输出：
#   value：通过文件身份、大小和 JSON 结构检查的字典。
def _read_object(path: Path, *, maximum_bytes: int) -> dict:
    try:
        value = read_runtime_object(path, maximum_bytes=maximum_bytes)
        return value
    except (OSError, ValueError, RecursionError) as error:
        raise RuntimeRevisionError("RUNTIME_ACTIVE_REVISION_INVALID") from error


# 功能：
#   逐项绑定执行身份、替换序号及完整制品摘要；轨迹不变也不能代替任务图和动作合同绑定。
# 输入：
#   adoption：执行器实际发布的采纳记录。
#   replacement：待证明已采纳的替换制品。
#   execution_id：当前运行的唯一执行身份。
# 输出：
#   gates：执行身份、序号、摘要及核心安全门控的检查结果。
def replacement_adoption_gates(
    *, adoption: dict[str, object], replacement: RuntimeReplacementTrack, execution_id: str,
) -> dict[str, bool]:
    gates = {
        "execution": adoption.get("execution_id") == execution_id,
        "message": adoption.get("message_id") == replacement.message_id,
        "replacement_sequence": (
            type(adoption.get("replacement_sequence")) is int
            and adoption.get("replacement_sequence") == replacement.replacement_sequence
        ),
        "replacement_hash": adoption.get("replacement_sha256") == sha256_json(replacement),
        "track_hash": adoption.get("track_sha256") == sha256_json(replacement.track),
        "replacement_execution": replacement.execution_id == execution_id,
        "replacement_gates": all_required_gates_passed(replacement.deterministic_gates),
    }
    if replacement.revised_task_graph is not None:
        gates.update({
            "task_graph_hash": adoption.get("task_graph_sha256")
            == sha256_json(replacement.revised_task_graph),
            "checkpoint_contract_hash": adoption.get("runtime_checkpoints_sha256")
            == sha256_json(replacement.runtime_checkpoints),
            "runtime_action_contract_hash": adoption.get("runtime_actions_sha256")
            == sha256_json(replacement.runtime_actions),
        })
    return gates


# 功能：
#   1. 只加载执行器当前已采纳的替换制品，不扫描历史建议，也不按相同坐标猜测当前计划。
#   2. 缺少采纳记录才代表沿用原确认计划；存在但损坏、链接或绑定失败均明确拒绝。
# 输入：
#   control_dir：本轮执行的控制目录。
#   execution_id：可选的预期执行身份；未提供时从有界会话记录中读取。
# 输出：
#   replacement：完整绑定的已采纳替换制品；尚未采纳新计划时为 None。
def load_active_replacement(
    control_dir: Path, *, execution_id: str | None = None,
) -> RuntimeReplacementTrack | None:
    active_path = control_dir / "active-track.json"
    try:
        check_plain_plugin_path(active_path)
    except (OSError, ValueError) as error:
        raise RuntimeRevisionError("RUNTIME_ACTIVE_REVISION_INVALID") from error
    if not active_path.exists():
        replacement = None
        return replacement
    adoption = _read_object(active_path, maximum_bytes=65_536)
    if execution_id is None:
        try:
            session = RuntimeControlSession.model_validate(
                _read_object(control_dir / "session.json", maximum_bytes=65_536)
            )
        except ValueError as error:
            raise RuntimeRevisionError("RUNTIME_ACTIVE_REVISION_SESSION_INVALID") from error
        execution_id = session.execution_id
    # Validate identity before it becomes a relative filename. Even a matching
    # track hash must not permit path traversal or a previous execution's receipt.
    message_id = adoption.get("message_id")
    if not isinstance(message_id, str) or not re.fullmatch(r"runtime-msg-[0-9a-f]{32}", message_id):
        raise RuntimeRevisionError("RUNTIME_ACTIVE_REVISION_MESSAGE_INVALID")
    try:
        replacement = RuntimeReplacementTrack.model_validate(_read_object(
            control_dir / "replacements" / f"{message_id}.json", maximum_bytes=16 * 1024 * 1024,
        ))
    except ValueError as error:
        raise RuntimeRevisionError("RUNTIME_ACTIVE_REVISION_ARTIFACT_INVALID") from error
    gates = replacement_adoption_gates(
        adoption=adoption, replacement=replacement, execution_id=execution_id,
    )
    if not all(gates.values()):
        failed = ",".join(name for name, accepted in gates.items() if not accepted)
        raise RuntimeRevisionError(f"RUNTIME_ACTIVE_REVISION_REJECTED:{failed}")
    return replacement
