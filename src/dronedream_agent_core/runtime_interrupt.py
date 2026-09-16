"""Runtime message ingress, structured classification, and fail-closed authorization."""

from __future__ import annotations

import math
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .contracts import (
    MapAsset,
    MapCatalog,
    PreparedMission,
    RuntimeActionExecutionReceipt,
    RuntimeAmendmentDirective,
    RuntimeAuthorizedCommand,
    RuntimeCommandAdoption,
    RuntimeControlSession,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeMessageClassification,
    RuntimeOperatorTakeoverAdoption,
    RuntimeOperatorTakeoverGrant,
    RuntimeReplacementTrack,
    RuntimeUserMessage,
    VehicleAsset,
)
from .extensions import ExtensionExecutionError
from .hashing import sha256_json
from .model_harness.model_port import ProviderName, StructuredModelPort
from .prompts import RUNTIME_MESSAGE_CLASSIFIER
from .runtime_commands import RuntimeCommandError, build_runtime_command
from .runtime_control_io import (
    MAX_RUNTIME_REPLACEMENT_BYTES,
    publish_runtime_json,
    read_runtime_object,
)
from .runtime_language import affirmative_phrase_present
from .runtime_plugins import (
    all_required_gates_passed,
    append_hook_receipts,
    augment_runtime_prompt,
    runtime_extension_registry,
    validate_runtime_model_output,
)
from .runtime_replan import (
    RuntimeReplanError,
    build_runtime_coverage_replacement,
    build_runtime_replacement,
    build_runtime_speed_replacement,
)
from .runtime_revision import load_active_replacement, replacement_adoption_gates


class RuntimeMessageRejected(RuntimeError):
    """Runtime ingress rejected an unbound or no-longer-actionable message."""


# 功能：
#   复用当前运行的替换采纳绑定规则，轨迹相同也必须匹配执行身份、序号和所有修订合同。
# 输入：
#   adoption：执行器发布的替换采纳回执。
#   replacement：对应替换制品。
#   session：当前运行会话。
# 输出：
#   gates：各项身份及制品摘要检查结果。
def _runtime_adoption_gates(
    *,
    adoption: dict[str, object],
    replacement: RuntimeReplacementTrack,
    session: RuntimeControlSession,
) -> dict[str, bool]:
    gates = replacement_adoption_gates(
        adoption=adoption, replacement=replacement, execution_id=session.execution_id,
    )
    return gates


# 功能：
#   从当前运行目录的结构化动作回执收集已接受步骤，拒绝错误状态、问题码或不全真的门控。
# 输入：
#   run_dir：调用方确定的本次运行证据根目录。
# 输出：
#   accepted：实际读取并校验的已接受步骤标识集合。
def _accepted_runtime_action_step_ids(run_dir: Path) -> set[str]:
    accepted: set[str] = set()
    receipt_dir = run_dir / "runtime-actions" / "receipts"
    for path in sorted(receipt_dir.glob("*.receipt.json")):
        try:
            receipt = RuntimeActionExecutionReceipt.model_validate(
                read_runtime_object(path)
            )
        except (OSError, ValueError) as error:
            raise RuntimeMessageRejected(f"RUNTIME_ACTION_RECEIPT_INVALID:{path.name}") from error
        if (receipt.status != "accepted" or receipt.issue_codes
                or not all_required_gates_passed(receipt.deterministic_gates)):
            raise RuntimeMessageRejected(f"RUNTIME_ACTION_RECEIPT_REJECTED:{receipt.step_id}")
        accepted.add(receipt.step_id)
    return accepted


# 功能：
#   绑定外围命令的执行身份、动作、摘要和成功读回，不能把命令发布当成执行完成。
# 输入：
#   adoption：执行器命令采纳及读回记录。
#   command：已授权的有界命令。
#   session：当前运行会话。
# 输出：
#   gates：身份、内容及成功读回的分项检查结果。
def _runtime_command_adoption_gates(
    *,
    adoption: RuntimeCommandAdoption,
    command: RuntimeAuthorizedCommand,
    session: RuntimeControlSession,
) -> dict[str, bool]:
    gates = {
        "execution": adoption.execution_id == session.execution_id,
        "message": adoption.message_id == command.message_id,
        "action": adoption.action == command.action,
        "command_hash": adoption.command_sha256 == sha256_json(command),
        "command_execution": command.execution_id == session.execution_id,
        "command_gates": all_required_gates_passed(command.deterministic_gates),
        "execution_success": adoption.success is True,
    }
    return gates


# 功能：
#   复用统一运行证据发布器；仅类型化替换轨迹使用较大预算，普通命令保持默认消息上限。
# 输入：
#   path：改令决定、命令或替换制品的目标路径。
#   payload：需要发布的结构化运行证据。
#   replace_existing：是否允许更新当前目标，独占模式用于已有请求优先的场景。
# 输出：
#   None：不返回业务数据。
def _atomic_json(path: Path, payload: object, *, replace_existing: bool = True) -> None:
    if isinstance(payload, RuntimeReplacementTrack):
        publish_runtime_json(
            path, payload, replace_existing=replace_existing,
            maximum_bytes=MAX_RUNTIME_REPLACEMENT_BYTES,
        )
    else:
        publish_runtime_json(path, payload, replace_existing=replace_existing)


# 功能：
#   建立当前执行独占的改令会话和证据子目录，保留已有会话，不覆盖它继续使用。
# 输入：
#   control_dir：本次运行的控制目录。
#   conversation_id：用户对话身份。
#   mission_id：任务身份。
#   plan_revision_id：用户确认的计划身份。
#   contract_id：不可变任务合同身份。
#   execution_id：当前实际运行身份。
#   prepared_mission_sha256：已确认任务包的摘要。
# 输出：
#   session：已写入控制目录的执行绑定会话。
def create_runtime_control_session(
    *,
    control_dir: Path,
    conversation_id: str,
    mission_id: str,
    plan_revision_id: str,
    contract_id: str,
    execution_id: str,
    prepared_mission_sha256: str,
) -> RuntimeControlSession:
    session_path = control_dir / "session.json"
    if session_path.exists():
        raise FileExistsError(f"runtime control session already exists: {session_path}")
    session = RuntimeControlSession(
        conversation_id=conversation_id,
        mission_id=mission_id,
        plan_revision_id=plan_revision_id,
        contract_id=contract_id,
        execution_id=execution_id,
        prepared_mission_sha256=prepared_mission_sha256,
        created_at=datetime.now(UTC),
    )
    for name in (
        "inbox",
        "claimed",
        "acks",
        "decisions",
        "processed",
        "replacements",
        "replan-failures",
        "commands",
        "command-results",
        "command-failures",
        "adoptions",
        "takeover-grants",
        "takeover-adoptions",
        "operator-commands",
        "processed-operator-commands",
        "takeover-evidence",
        "follow",
    ):
        (control_dir / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(session_path, session, replace_existing=False)
    _atomic_json(
        control_dir / "side-effects.state.json",
        {
            "enabled": True,
            "execution_id": execution_id,
            "reason": "active confirmed plan",
        },
    )
    return session


# 功能：
#   将现有会话改为关闭以拒绝新消息；关闭入口并不证明无人机已落地。
# 输入：
#   control_dir：需要关闭的当前运行控制目录。
# 输出：
#   closed：已发布关闭状态的会话。
def close_runtime_control_session(control_dir: Path) -> RuntimeControlSession:
    path = control_dir / "session.json"
    session = RuntimeControlSession.model_validate(read_runtime_object(path, maximum_bytes=65_536))
    closed = session.model_copy(update={"state": "closed"})
    _atomic_json(path, closed)
    return closed


# 功能：
#   读取当前接受消息的会话，拒绝终止飞行阶段的新改令，再独占发布绑定本次执行的消息。
# 输入：
#   control_dir：本次运行控制目录。
#   text：用户的自然语言改令。
# 输出：
#   message：已写入收件箱、仍需执行器稳定悬停及核心授权的消息。
def submit_runtime_message(*, control_dir: Path, text: str) -> RuntimeUserMessage:
    session_path = control_dir / "session.json"
    if not session_path.is_file():
        raise RuntimeMessageRejected("RUNTIME_CONTROL_SESSION_NOT_FOUND")
    session = RuntimeControlSession.model_validate(
        read_runtime_object(session_path, maximum_bytes=65_536)
    )
    if session.state != "accepting":
        raise RuntimeMessageRejected("RUNTIME_CONTROL_SESSION_CLOSED")
    phase_path = control_dir.parent / "runtime-phase.json"
    if phase_path.is_file():
        phase = read_runtime_object(phase_path, maximum_bytes=65_536).get("phase")
        if phase in {"LANDING", "LANDED", "COMPLETE", "FAILED"}:
            raise RuntimeMessageRejected(f"RUNTIME_MESSAGE_TOO_LATE:{phase}")
    message = RuntimeUserMessage(
        message_id=f"runtime-msg-{uuid4().hex}",
        conversation_id=session.conversation_id,
        mission_id=session.mission_id,
        plan_revision_id=session.plan_revision_id,
        contract_id=session.contract_id,
        execution_id=session.execution_id,
        text=text,
        submitted_at=datetime.now(UTC),
    )
    _atomic_json(
        control_dir / "inbox" / f"{message.message_id}.json", message, replace_existing=False,
    )
    return message


_EXACT_EMERGENCY_MESSAGES = {
    "停止",
    "停下",
    "降落",
    "abort",
    "emergency",
    "land",
    "stop",
}
_IMMEDIATE_EMERGENCY_PHRASES = (
    "紧急",
    "停止当前",
    "停止任务",
    "立即停止",
    "马上停止",
    "立刻停止",
    "现在停止",
    "立即停下",
    "马上停下",
    "立刻停下",
    "立即降落",
    "马上降落",
    "立刻降落",
    "现在降落",
    "原地降落",
    "就地降落",
    "abort mission",
    "emergency stop",
    "stop now",
    "land now",
)
_AMENDMENT_TOKENS = (
    "改到",
    "改去",
    "改道",
    "改变目的地",
    "不是",
    "不要继续去",
    "换成",
    "换到",
    "另一个",
    "change destination",
    "instead",
    "reroute",
)


# 功能：
#   在云端分类前核对本消息的稳定悬停回执、执行身份及副作用冻结状态。
# 输入：
#   message：当前运行消息。
#   acknowledgement：执行器返回的稳定悬停证据。
# 输出：
#   gates：消息摘要、身份和非空全真悬停门控的检查结果。
def _hold_acknowledgement_gates(message, acknowledgement) -> dict[str, bool]:
    gates = {
        "message_hash_matches_ack": acknowledgement.message_sha256 == sha256_json(message),
        "message_id_matches_ack": message.message_id == acknowledgement.message_id,
        "execution_id_matches_ack": message.execution_id == acknowledgement.execution_id,
        "side_effects_inhibited": acknowledgement.side_effects_inhibited is True,
        "hold_deterministic_gates_passed": all_required_gates_passed(
            acknowledgement.deterministic_gates),
    }
    return gates


# 功能：
#   改令参数被拒绝时保留悬停，但已经要求的安全降落不能被参数错误改回悬停。
# 输入：
#   action：核心初步授权动作。
#   reason：初步授权理由。
#   directive：插件给出的结构化参数检查结果。
# 输出：
#   result：最终动作与理由组成的二元组。
def _apply_directive_validation(action: str, reason: str, directive) -> tuple[str, str]:
    if directive.issue_codes and action != "land":
        result = "hold", "Runtime amendment parameters failed deterministic validation."
    else:
        result = action, reason
    return result


# 功能：
#   识别明确立即停止／降落请求，保留问句及否定限制，不把“返回后降落”误当成立即落地。
# 输入：
#   text：用户的原始自然语言文本。
# 输出：
#   requested：精确紧急命令或明确肯定即时短语成立时为真。
def _emergency_override_requested(text: str) -> bool:
    # Preserve question marks: "land now?" is not an exact emergency command.
    normalized = " ".join(text.casefold().split()).strip("。.!！,，;；:：")
    requested = normalized in _EXACT_EMERGENCY_MESSAGES or any(
        affirmative_phrase_present(normalized, phrase, allow_deferred=False)
        for phrase in _IMMEDIATE_EMERGENCY_PHRASES
    )
    return requested


# 功能：
#   1. 以稳定悬停与消息身份门控为前提，核心代码将模型分类映射到可审查的安全状态。
#   2. 紧急和安全降落保留强制约束；改目的地必须重规划，外围控制需独立命令及读回。
#   3. 此处的授权结果不是飞控执行成功；人工接管仍需独立认证授权。
# 输入：
#   message：执行绑定的用户消息。
#   acknowledgement：执行器冻结旧计划后的悬停回执。
#   classification：模型分类与请求的动作。
# 输出：
#   result：动作、分项门控和授权理由组成的三元组。
def authorize_runtime_action(
    *,
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    classification: RuntimeMessageClassification,
) -> tuple[str, dict[str, bool], str]:
    normalized = message.text.casefold()
    emergency_override = _emergency_override_requested(normalized)
    amendment_override = any(token in normalized for token in _AMENDMENT_TOKENS)
    gates = _hold_acknowledgement_gates(message, acknowledgement)
    if not all(gates.values()):
        result = "land", gates, "Hold acknowledgement failed a deterministic safety gate."
        return result
    if emergency_override or classification.message_kind == "emergency_stop":
        result = "land", gates, "Emergency wording or classification forces controlled landing."
        return result
    if classification.requested_action in {"land", "safe_land"}:
        result = "land", gates, "A safe-land amendment is authorized only as controlled landing."
        return result
    if classification.requested_action == "operator_takeover":
        result = "hold", gates, "Operator takeover requires a separate authenticated control grant."
        return result
    if classification.requested_action == "operator_release":
        result = (
            "resume_original",
            gates,
            "Authenticated operator control is released back to the prepared mission.",
        )
        return result
    if classification.requested_action in {
        "camera_control",
        "payload_control",
        "set_avoidance",
    }:
        result = (
            "apply_command",
            gates,
            "A bounded peripheral or flight-policy command requires code validation and readback.",
        )
        return result
    if classification.requested_action == "pause" and not amendment_override:
        result = "hold", gates, "Pause keeps the aircraft in deterministic stable hold."
        return result
    if (
        amendment_override
        or classification.requires_plan_revision
        or classification.message_kind in {"mission_amendment", "motion_adjustment"}
        or classification.requested_action in {"replan", "adjust_motion"}
    ):
        result = (
            "hold_for_replan",
            gates,
            "The old plan is superseded; continuation requires a new code-validated revision.",
        )
        return result
    if (
        classification.message_kind == "informational"
        and classification.requested_action == "resume"
    ):
        result = "resume_original", gates, "Informational message does not alter the plan."
        return result
    result = "hold_for_replan", gates, "Ambiguous runtime intent cannot resume the old plan."
    return result


# 功能：
#   从整体等待预算预留本地处理时间，剩余时间最多分给两次模型尝试，不让重试各花一整份预算。
# 输入：
#   total_timeout_seconds：有限正数的端到端等待预算，秒。
# 输出：
#   policy：最大尝试次数与每次调用超时秒数组成的二元组。
def _runtime_model_attempt_policy(total_timeout_seconds: float) -> tuple[int, float]:
    if type(total_timeout_seconds) not in (int, float):
        raise ValueError("runtime model timeout must be finite and positive")
    try:
        total_timeout_seconds = float(total_timeout_seconds)
    except OverflowError as error:
        raise ValueError("runtime model timeout must be finite and positive") from error
    if not math.isfinite(total_timeout_seconds) or total_timeout_seconds <= 0.0:
        raise ValueError("runtime model timeout must be finite and positive")
    attempts = 2 if total_timeout_seconds >= 4.0 else 1
    reserve_seconds = min(2.0, total_timeout_seconds * 0.1)
    per_attempt_seconds = (total_timeout_seconds - reserve_seconds) / attempts
    policy = attempts, per_attempt_seconds
    return policy


class RuntimeInterruptionCoordinator:
    """Waits for stable-hold evidence before making any real model call."""

    # 功能：
    #   冻结本轮任务、地图、车辆和会话副本，配置模型总预算与插件快照，但不开始处理消息。
    # 输入：
    #   self：运行改令协调器。
    #   prepared：当前确认的任务包。
    #   session：本轮执行的消息会话。
    #   control_dir：消息与采纳证据目录。
    #   provider：运行分类模型提供方。
    #   abort_file：执行器中止请求路径。
    #   lifecycle_db_path：任务生命周期数据库路径。
    #   model_timeout_seconds：分类尝试的整体预算，秒。
    #   map_graph：合格地图的图结构。
    #   map_catalog：地图语义实体目录。
    #   semantic_path：当前地图语义文件路径。
    #   vehicle：已绑定的无人机资产。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        prepared: PreparedMission,
        session: RuntimeControlSession,
        control_dir: Path,
        provider: ProviderName,
        abort_file: Path,
        lifecycle_db_path: Path,
        model_timeout_seconds: float,
        map_graph: MapAsset,
        map_catalog: MapCatalog,
        semantic_path: Path,
        vehicle: VehicleAsset,
    ) -> None:
        self.prepared = prepared.model_copy(deep=True)
        self.session = session.model_copy(deep=True)
        self.control_dir = control_dir
        self.abort_file = abort_file
        self.lifecycle_db_path = lifecycle_db_path
        self.map_graph = map_graph.model_copy(deep=True)
        self.map_catalog = map_catalog.model_copy(deep=True)
        self.semantic_path = semantic_path
        self.vehicle = vehicle.model_copy(deep=True)
        model_attempts, per_attempt_timeout_seconds = _runtime_model_attempt_policy(
            model_timeout_seconds
        )
        self.port = StructuredModelPort(
            provider,
            max_attempts=model_attempts,
            timeout_seconds=per_attempt_timeout_seconds,
        )
        self.extensions = runtime_extension_registry(self.prepared)
        self.receipt_path = control_dir.parent / "plugin-hook-receipts.jsonl"
        self.adoption_timeout_seconds = min(model_timeout_seconds, 30.0)
        self.decisions: list[RuntimeInterruptionDecision] = []
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._publication_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="runtime-interruption-coordinator", daemon=True
        )

    # 功能：
    #   启动运行消息协调线程，拒绝关闭后的重新启动；实际运动仍归执行器负责。
    # 输入：
    #   self：尚未启动且未停止的协调器。
    # 输出：
    #   None：不返回业务数据。
    def start(self) -> None:
        with self._publication_lock:
            if self._stop.is_set():
                raise RuntimeError("runtime interruption coordinator already stopped")
            self._thread.start()

    # 功能：
    #   发出停止信号，并在共享总预算内等待在途发布及线程退出；不宣称远端请求被强制取消。
    # 输入：
    #   self：当前协调器。
    #   timeout_seconds：等待发布同步和线程结束的总秒数。
    # 输出：
    #   None：不返回业务数据。
    def stop(self, timeout_seconds: float) -> None:
        if (type(timeout_seconds) not in (int, float)
                or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("runtime interruption stop timeout invalid")
        deadline = time.monotonic() + timeout_seconds
        self._stop.set()
        if not self._publication_lock.acquire(timeout=timeout_seconds):
            raise TimeoutError("runtime interruption publication did not stop")
        try:
            started = self._thread.ident is not None
        finally:
            self._publication_lock.release()
        if not started:
            return
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("runtime interruption coordinator did not stop")

    # 功能：
    #   独占写入受控停止请求，不覆盖操作员更早到达的中止证据。
    # 输入：
    #   self：当前运行改令协调器。
    #   reason：中止原因。
    # 输出：
    #   None：不返回业务数据。
    def _request_abort(self, reason: str) -> None:
        with suppress(FileExistsError):
            _atomic_json(self.abort_file,
                         {"reason": reason, "requested_at": datetime.now(UTC).isoformat()},
                         replace_existing=False)

    # 功能：
    #   复核停止、会话及活动计划后独占发布运行决定或制品，只有决定成功写入才加入历史。
    # 输入：
    #   self：当前运行改令协调器。
    #   path：决定、命令或替换制品目标。
    #   payload：已经完成业务校验的发布内容。
    #   expected_replacement：此次模型输入所绑定的活动替换；原计划时为 None。
    # 输出：
    #   published：已成功发布时为真；本协调器或对应会话已停止时为假。
    def _publish_if_active(
        self, path: Path, payload: object, *, expected_replacement: RuntimeReplacementTrack | None,
    ) -> bool:
        published = False
        with self._publication_lock:
            if self._stop.is_set():
                return published
            session = RuntimeControlSession.model_validate(
                read_runtime_object(self.control_dir / "session.json", maximum_bytes=65_536)
            )
            if session.state == "closed":
                return published
            if session != self.session:
                raise RuntimeMessageRejected("RUNTIME_CONTROL_SESSION_CHANGED")
            if load_active_replacement(
                self.control_dir, execution_id=self.session.execution_id,
            ) != expected_replacement:
                raise RuntimeMessageRejected("RUNTIME_ACTIVE_REVISION_CHANGED")
            # 这个锁协调本进程的停止与发布，不替代执行器对跨进程身份和采纳摘要的检查。
            _atomic_json(path, payload, replace_existing=False)
            if isinstance(payload, RuntimeInterruptionDecision):
                self.decisions.append(payload)
            published = True
        return published

    # 功能：
    #   1. 校验消息与稳定悬停证据，以当前已采纳任务上下文调用模型、分类插件和改令策略。
    #   2. 核心授权后发布决定，再按动作构建外围命令、替换轨迹或等待认证接管的采纳记录。
    #   3. 外围命令、替换与接管须等执行器采纳；信息性继续和降落则按核心授权更新生命周期。
    #   4. 发布前复核停止和计划身份，存储初始化或关闭失败同样记错并中止。
    # 输入：
    #   self：持有本次任务、模型、插件、资产与执行证据路径的协调器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        from .context import ContextStore

        processed: set[str] = set()
        resumed_adoptions: set[str] = set()
        lifecycle_context = None
        try:
            lifecycle_context = ContextStore(self.lifecycle_db_path)
            while not self._stop.wait(0.05):
                for adoption_path in sorted((self.control_dir / "adoptions").glob("*.json")):
                    message_id = adoption_path.stem
                    if message_id in resumed_adoptions:
                        continue
                    replacement_path = self.control_dir / "replacements" / f"{message_id}.json"
                    command_path = self.control_dir / "commands" / f"{message_id}.json"
                    if replacement_path.is_file():
                        adoption_value = read_runtime_object(adoption_path, maximum_bytes=65_536)
                        replacement = RuntimeReplacementTrack.model_validate(
                            read_runtime_object(
                                replacement_path, maximum_bytes=MAX_RUNTIME_REPLACEMENT_BYTES,
                            )
                        )
                        adoption_gates = _runtime_adoption_gates(
                            adoption=adoption_value,
                            replacement=replacement,
                            session=self.session,
                        )
                    elif command_path.is_file():
                        adoption = RuntimeCommandAdoption.model_validate(
                            read_runtime_object(adoption_path, maximum_bytes=65_536)
                        )
                        command = RuntimeAuthorizedCommand.model_validate(
                            read_runtime_object(command_path)
                        )
                        adoption_gates = _runtime_command_adoption_gates(
                            adoption=adoption,
                            command=command,
                            session=self.session,
                        )
                    else:
                        continue
                    if not all(adoption_gates.values()):
                        failed = ",".join(
                            name for name, accepted in adoption_gates.items() if not accepted
                        )
                        raise RuntimeMessageRejected(f"RUNTIME_ADOPTION_REJECTED:{failed}")
                    lifecycle_context.lifecycle.set_execution_state(
                        conversation_id=self.session.conversation_id,
                        execution_id=self.session.execution_id,
                        state="executing",
                    )
                    resumed_adoptions.add(message_id)
                for ack_path in sorted((self.control_dir / "acks").glob("*.json")):
                    message_id = ack_path.stem
                    if message_id in processed:
                        continue
                    message_path = self.control_dir / "claimed" / f"{message_id}.json"
                    if not message_path.is_file():
                        continue
                    message = RuntimeUserMessage.model_validate(
                        read_runtime_object(message_path)
                    )
                    acknowledgement = RuntimeHoldAcknowledgement.model_validate(
                        read_runtime_object(ack_path)
                    )
                    identity_gates = {
                        "message_file": message.message_id == message_id,
                        "conversation": message.conversation_id == self.session.conversation_id,
                        "mission": message.mission_id == self.session.mission_id,
                        "plan_revision": (
                            message.plan_revision_id == self.session.plan_revision_id
                        ),
                        "contract": message.contract_id == self.session.contract_id,
                        "execution": message.execution_id == self.session.execution_id,
                    }
                    if not all(identity_gates.values()):
                        raise RuntimeMessageRejected("RUNTIME_MESSAGE_SESSION_BINDING_MISMATCH")
                    if not all(_hold_acknowledgement_gates(message, acknowledgement).values()):
                        raise RuntimeMessageRejected("RUNTIME_STABLE_HOLD_ACKNOWLEDGEMENT_INVALID")
                    active_replacement = load_active_replacement(
                        self.control_dir, execution_id=self.session.execution_id,
                    )
                    current_task_graph = (
                        active_replacement.revised_task_graph
                        if active_replacement is not None
                        and active_replacement.revised_task_graph is not None
                        else self.prepared.task_graph
                    )
                    current_track = (
                        active_replacement.track if active_replacement is not None
                        else self.prepared.px4_track
                    )
                    lifecycle_context.lifecycle.set_execution_state(
                        conversation_id=self.session.conversation_id,
                        execution_id=self.session.execution_id,
                        state="holding",
                    )
                    try:
                        instructions, prompt_receipts = augment_runtime_prompt(
                            self.extensions,
                            role="runtime_message_classifier",
                            instructions=RUNTIME_MESSAGE_CLASSIFIER,
                        )
                    except ExtensionExecutionError as error:
                        append_hook_receipts(self.receipt_path, [error.receipt])
                        raise
                    append_hook_receipts(self.receipt_path, prompt_receipts)
                    result = self.port.call(
                        role="runtime_message_classifier",
                        output_type=RuntimeMessageClassification,
                        instructions=instructions,
                        input_artifact={
                            "runtime_user_message": message.model_dump(mode="json"),
                            "stable_hold_acknowledgement": acknowledgement.model_dump(mode="json"),
                            "mission_contract": self.prepared.contract.model_dump(mode="json"),
                            "current_task_graph": current_task_graph.model_dump(mode="json"),
                            "prepared_plan_revision": self.prepared.plan.revision,
                            "current_replacement_sequence": (
                                active_replacement.replacement_sequence
                                if active_replacement is not None else 0
                            ),
                            "current_execution_track_sha256": sha256_json(current_track),
                            "current_target_node": (
                                active_replacement.target_node if active_replacement is not None
                                else self.prepared.contract.target_node
                            ),
                            "current_return_node": (
                                active_replacement.return_node if active_replacement is not None
                                else self.prepared.contract.return_node
                            ),
                            "immutable_rules": [
                                "The old plan is frozen before this model call.",
                                "The model has no actuator or continuation authority.",
                                "A destination or motion change requires a new plan revision.",
                            ],
                        },
                        context_id=(f"{self.session.conversation_id}::runtime_message_classifier"),
                    )
                    # The flight can terminate while a synchronous provider request is in
                    # progress.  Once shutdown is requested, discard the late classification
                    # instead of spending more time building a replacement for a closed
                    # execution or waiting for an adoption that can no longer arrive.
                    if self._stop.is_set():
                        return
                    try:
                        output_guard_receipts = validate_runtime_model_output(
                            self.extensions,
                            role="runtime_message_classifier",
                            expected_schema=RuntimeMessageClassification.__name__,
                            artifact=result.artifact,
                            record=result.record,
                        )
                    except ExtensionExecutionError as error:
                        append_hook_receipts(self.receipt_path, [error.receipt])
                        raise
                    append_hook_receipts(self.receipt_path, output_guard_receipts)
                    try:
                        classified_value, classification_receipts = self.extensions.invoke_pipeline(
                            "runtime.amendment-classifier",
                            "classify_amendment",
                            result.artifact.model_dump(mode="json"),
                            message=message,
                            prepared=self.prepared,
                        )
                        classification = RuntimeMessageClassification.model_validate(
                            classified_value
                        )
                        directive_value, directive_receipts = self.extensions.invoke_single(
                            "runtime.amendment-policy",
                            "apply_amendment",
                            required=True,
                            classification=classification,
                            message=message,
                            acknowledgement=acknowledgement,
                            prepared=self.prepared,
                        )
                        directive = RuntimeAmendmentDirective.model_validate(directive_value)
                    except ExtensionExecutionError as error:
                        append_hook_receipts(self.receipt_path, [error.receipt])
                        raise
                    append_hook_receipts(
                        self.receipt_path, [*classification_receipts, *directive_receipts]
                    )
                    action, authorization_gates, reason = authorize_runtime_action(
                        message=message,
                        acknowledgement=acknowledgement,
                        classification=classification,
                    )
                    action, reason = _apply_directive_validation(action, reason, directive)
                    decision = RuntimeInterruptionDecision(
                        message_sha256=sha256_json(message),
                        hold_ack_sha256=sha256_json(acknowledgement),
                        classification=classification,
                        model_call=result.record,
                        authorized_action=action,
                        authorization_gates={**identity_gates, **authorization_gates},
                        decision_reason=reason,
                        plugin_hook_receipts=[
                            *prompt_receipts,
                            *output_guard_receipts,
                            *classification_receipts,
                            *directive_receipts,
                        ],
                        amendment_directive=directive,
                    )
                    if not self._publish_if_active(
                        self.control_dir / "decisions" / f"{message_id}.json", decision,
                        expected_replacement=active_replacement,
                    ):
                        return
                    if decision.authorized_action == "apply_command":
                        try:
                            command = build_runtime_command(
                                message=message,
                                acknowledgement=acknowledgement,
                                decision=decision,
                                prepared=self.prepared,
                            )
                            if not self._publish_if_active(
                                self.control_dir / "commands" / f"{message_id}.json",
                                command, expected_replacement=active_replacement,
                            ):
                                return
                            adoption_path = self.control_dir / "adoptions" / f"{message_id}.json"
                            adoption_deadline = time.monotonic() + self.adoption_timeout_seconds
                            while not self._stop.wait(0.05):
                                if adoption_path.is_file():
                                    adoption = RuntimeCommandAdoption.model_validate(
                                        read_runtime_object(adoption_path, maximum_bytes=65_536)
                                    )
                                    adoption_gates = _runtime_command_adoption_gates(
                                        adoption=adoption,
                                        command=command,
                                        session=self.session,
                                    )
                                    if not all(adoption_gates.values()):
                                        failed = ",".join(
                                            name
                                            for name, accepted in adoption_gates.items()
                                            if not accepted
                                        )
                                        raise RuntimeMessageRejected(
                                            f"RUNTIME_COMMAND_ADOPTION_REJECTED:{failed}"
                                        )
                                    lifecycle_context.lifecycle.set_execution_state(
                                        conversation_id=self.session.conversation_id,
                                        execution_id=self.session.execution_id,
                                        state="executing",
                                    )
                                    resumed_adoptions.add(message_id)
                                    break
                                if time.monotonic() >= adoption_deadline:
                                    raise RuntimeMessageRejected("RUNTIME_COMMAND_ADOPTION_TIMEOUT")
                        except RuntimeCommandError as exc:
                            _atomic_json(
                                self.control_dir / "command-failures" / f"{message_id}.json",
                                {
                                    "message_id": message_id,
                                    "decision_sha256": sha256_json(decision),
                                    "reason": str(exc),
                                    "failed_at": datetime.now(UTC).isoformat(),
                                },
                            )
                    if decision.authorized_action == "hold_for_replan":
                        prior_track = self.prepared.px4_track
                        active_target_node = self.prepared.contract.target_node
                        active_return_node = self.prepared.contract.return_node
                        active_task_graph = self.prepared.task_graph
                        active_runtime_actions = self.prepared.runtime_actions
                        prior_track_sha256 = sha256_json(prior_track)
                        replacement_sequence = 1
                        previous = load_active_replacement(
                            self.control_dir, execution_id=self.session.execution_id,
                        )
                        if previous is not None:
                            prior_track_sha256 = sha256_json(previous.track)
                            replacement_sequence = previous.replacement_sequence + 1
                            prior_track = previous.track
                            active_target_node = previous.target_node
                            active_return_node = previous.return_node
                            if previous.revised_task_graph is not None:
                                active_task_graph = previous.revised_task_graph
                            if previous.runtime_actions is not None:
                                active_runtime_actions = previous.runtime_actions
                        try:
                            if decision.classification.requested_action == "set_speed":
                                builder = build_runtime_speed_replacement
                            elif decision.classification.requested_action == "set_coverage":
                                builder = build_runtime_coverage_replacement
                            else:
                                builder = build_runtime_replacement
                            common = dict(
                                message=message,
                                acknowledgement=acknowledgement,
                                decision=decision,
                                replacement_sequence=replacement_sequence,
                                prior_track_sha256=prior_track_sha256,
                                prior_track=prior_track,
                                graph=self.map_graph,
                                semantic_path=self.semantic_path,
                                vehicle=self.vehicle,
                                expected_map_sha256=self.prepared.contract.map_sha256,
                                expected_semantic_sha256=(
                                    self.prepared.contract.map_semantic_sha256
                                ),
                                expected_vehicle_asset_id=(self.prepared.contract.vehicle_asset_id),
                                active_target_node=active_target_node,
                                active_task_graph=active_task_graph,
                                active_runtime_actions=active_runtime_actions,
                                completed_runtime_action_step_ids=(
                                    _accepted_runtime_action_step_ids(self.control_dir.parent)
                                ),
                                plugin_snapshot=self.prepared.plugin_snapshot,
                            )
                            if builder in {
                                build_runtime_replacement,
                                build_runtime_coverage_replacement,
                            }:
                                common.update(
                                    catalog=self.map_catalog,
                                    return_node=active_return_node,
                                )
                            if builder is build_runtime_speed_replacement:
                                common["active_return_node"] = active_return_node
                            replacement = builder(**common)
                            if not self._publish_if_active(
                                self.control_dir / "replacements" / f"{message_id}.json",
                                replacement, expected_replacement=active_replacement,
                            ):
                                return
                            # The executor adopts replacement tracks asynchronously.  Do not
                            # leave the durable task lifecycle in ``holding`` until mission
                            # completion: wait for the executor's hash-bound adoption receipt,
                            # validate it, and resume the lifecycle in this same message flow.
                            # The outer adoption watcher remains as a recovery path for receipts
                            # that predate coordinator startup.
                            adoption_path = self.control_dir / "adoptions" / f"{message_id}.json"
                            adoption_deadline = time.monotonic() + self.adoption_timeout_seconds
                            while not self._stop.wait(0.05):
                                if adoption_path.is_file():
                                    adoption = read_runtime_object(
                                        adoption_path, maximum_bytes=65_536,
                                    )
                                    adoption_gates = _runtime_adoption_gates(
                                        adoption=adoption,
                                        replacement=replacement,
                                        session=self.session,
                                    )
                                    if not all(adoption_gates.values()):
                                        failed = ",".join(
                                            name
                                            for name, accepted in adoption_gates.items()
                                            if not accepted
                                        )
                                        raise RuntimeMessageRejected(
                                            f"RUNTIME_ADOPTION_REJECTED:{failed}"
                                        )
                                    lifecycle_context.lifecycle.set_execution_state(
                                        conversation_id=self.session.conversation_id,
                                        execution_id=self.session.execution_id,
                                        state="executing",
                                    )
                                    resumed_adoptions.add(message_id)
                                    break
                                if time.monotonic() >= adoption_deadline:
                                    raise RuntimeMessageRejected("RUNTIME_ADOPTION_TIMEOUT")
                        except RuntimeReplanError as exc:
                            _atomic_json(
                                self.control_dir / "replan-failures" / f"{message_id}.json",
                                {
                                    "message_id": message_id,
                                    "decision_sha256": sha256_json(decision),
                                    "reason": str(exc),
                                    "failed_at": datetime.now(UTC).isoformat(),
                                },
                            )
                    if (
                        decision.authorized_action == "hold"
                        and decision.classification.requested_action == "operator_takeover"
                    ):
                        adoption_path = (
                            self.control_dir / "takeover-adoptions" / f"{message_id}.json"
                        )
                        adoption_deadline = time.monotonic() + self.adoption_timeout_seconds
                        while not self._stop.wait(0.05):
                            if adoption_path.is_file():
                                adoption = RuntimeOperatorTakeoverAdoption.model_validate(
                                    read_runtime_object(adoption_path, maximum_bytes=65_536)
                                )
                                grant_path = (
                                    self.control_dir / "takeover-grants" / f"{message_id}.json"
                                )
                                if not grant_path.is_file():
                                    raise RuntimeMessageRejected(
                                        "RUNTIME_TAKEOVER_GRANT_ARTIFACT_MISSING"
                                    )
                                grant = RuntimeOperatorTakeoverGrant.model_validate(
                                    read_runtime_object(grant_path)
                                )
                                gates = {
                                    "message_id": adoption.message_id == message_id,
                                    "execution_id": (
                                        adoption.execution_id == self.session.execution_id
                                    ),
                                    "grant_hash": (adoption.grant_sha256 == sha256_json(grant)),
                                }
                                if not all(gates.values()):
                                    failed = ",".join(
                                        name for name, accepted in gates.items() if not accepted
                                    )
                                    raise RuntimeMessageRejected(
                                        f"RUNTIME_TAKEOVER_ADOPTION_REJECTED:{failed}"
                                    )
                                lifecycle_context.lifecycle.set_execution_state(
                                    conversation_id=self.session.conversation_id,
                                    execution_id=self.session.execution_id,
                                    state="executing",
                                )
                                resumed_adoptions.add(message_id)
                                break
                            if time.monotonic() >= adoption_deadline:
                                raise RuntimeMessageRejected("RUNTIME_TAKEOVER_ADOPTION_TIMEOUT")
                    if decision.authorized_action == "resume_original":
                        lifecycle_context.lifecycle.set_execution_state(
                            conversation_id=self.session.conversation_id,
                            execution_id=self.session.execution_id,
                            state="executing",
                        )
                    elif decision.authorized_action == "land":
                        lifecycle_context.lifecycle.set_execution_state(
                            conversation_id=self.session.conversation_id,
                            execution_id=self.session.execution_id,
                            state="landing",
                        )
                    processed.add(message_id)
        except BaseException as exc:
            self.error = exc
            self._request_abort(f"RUNTIME_INTERRUPTION_COORDINATOR_FAILURE_{type(exc).__name__}")
        finally:
            if lifecycle_context is not None:
                try:
                    lifecycle_context.close()
                except BaseException as exc:
                    if self.error is None:
                        self.error = exc
                    self._request_abort(f"RUNTIME_LIFECYCLE_CLOSE_FAILURE_{type(exc).__name__}")
            time.sleep(0.05)
