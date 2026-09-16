"""Transactional task-thread, plan-revision, and execution identity state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

from .contracts import MissionLifecycleBinding, PlanRevisionRecord, TaskThread


class LifecycleTransitionError(RuntimeError):
    """A request attempted an invalid task lifecycle transition."""


# 功能：
#   取得持久化生命周期使用的 UTC 时间，不混用仿真时间。
# 输入：
#   无。
# 输出：
#   timestamp：带 UTC 时区的当前时间。
def _now() -> datetime:
    timestamp = datetime.now(UTC)
    return timestamp


# 功能：
#   核对线程、当前计划及执行状态的成对关系，拒绝跨任务引用和不一致的执行资格。
# 输入：
#   thread：持久化或待导入的任务记录。
#   revision：该任务声称正在使用的计划记录。
# 输出：
#   None：校验成功不返回数据，关系不成立则抛出生命周期异常。
def _require_binding(thread: TaskThread, revision: PlanRevisionRecord) -> None:
    if (
        thread.conversation_id != revision.conversation_id
        or thread.mission_id != revision.mission_id
        or thread.current_plan_revision_id != revision.plan_revision_id
    ):
        raise LifecycleTransitionError("TASK_PLAN_BINDING_IDENTITY_MISMATCH")
    statuses = {
        "awaiting_confirmation": {"proposed", "confirmed"},
        "executing": {"executing"},
        "holding": {"executing"},
        "landing": {"executing"},
        "completed": {"completed"},
        "failed": {"failed"},
    }
    if revision.status not in statuses.get(thread.state, set()):
        raise LifecycleTransitionError("TASK_PLAN_BINDING_STATE_MISMATCH")
    if thread.state in {"executing", "holding", "landing"} and thread.active_execution_id is None:
        raise LifecycleTransitionError("TASK_PLAN_BINDING_EXECUTION_MISSING")
    if thread.state == "awaiting_confirmation" and thread.active_execution_id is not None:
        raise LifecycleTransitionError("TASK_PLAN_BINDING_EXECUTION_UNEXPECTED")


class MissionLifecycleStore:
    """Uses the ContextStore database connection but keeps lifecycle writes atomic."""

    # 功能：
    #   在上下文所有者提供的线程专属连接上初始化表，不提交所有者尚未完成的事务。
    # 输入：
    #   self：待初始化的存储。
    #   connection：由调用方负责关闭、当前不在事务中的 SQLite 连接。
    # 输出：
    #   None：初始化实例，不返回业务数据。
    def __init__(self, connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            raise LifecycleTransitionError("LIFECYCLE_INIT_REQUIRES_IDLE_TRANSACTION")
        self._connection = connection
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS task_threads (
              conversation_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL UNIQUE,
              thread_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plan_revisions (
              plan_revision_id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE (conversation_id, revision),
              FOREIGN KEY (conversation_id) REFERENCES task_threads(conversation_id)
            );
            """
        )
        self._connection.commit()

    # 功能：
    #   先取得写锁再读取转换依据，使确认、修订和结束执行不会交错覆盖。
    # 输入：
    #   self：持有当前连接的存储。
    # 输出：
    #   None：成功开启写事务，不返回业务数据。
    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    # 功能：
    #   按会话读取并校验任务，拒绝索引身份与 JSON 内容不一致的持久化记录。
    # 输入：
    #   self：当前存储。
    #   conversation_id：精确会话标识。
    # 输出：
    #   thread：独立任务对象；无该记录时为 None。
    def _thread(self, conversation_id: str) -> TaskThread | None:
        row = self._connection.execute(
            "SELECT thread_json, conversation_id, mission_id "
            "FROM task_threads WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            thread = TaskThread.model_validate_json(row[0], strict=True)
        except ValueError as error:
            raise LifecycleTransitionError("TASK_THREAD_RECORD_INVALID") from error
        if (thread.conversation_id, thread.mission_id) != (row[1], row[2]):
            raise LifecycleTransitionError("TASK_THREAD_RECORD_IDENTITY_MISMATCH")
        return thread

    # 功能：
    #   按精确计划标识读取并验证索引与内容，不回退到名称相近或时间最新的计划。
    # 输入：
    #   self：当前存储。
    #   plan_revision_id：计划记录标识。
    # 输出：
    #   revision：独立计划对象；无该记录时为 None。
    def _revision(self, plan_revision_id: str) -> PlanRevisionRecord | None:
        row = self._connection.execute(
            "SELECT record_json, plan_revision_id, conversation_id, revision "
            "FROM plan_revisions WHERE plan_revision_id = ?",
            (plan_revision_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            revision = PlanRevisionRecord.model_validate_json(row[0], strict=True)
        except ValueError as error:
            raise LifecycleTransitionError("PLAN_REVISION_RECORD_INVALID") from error
        if (revision.plan_revision_id, revision.conversation_id, revision.revision) != tuple(
            row[1:]
        ):
            raise LifecycleTransitionError("PLAN_REVISION_RECORD_IDENTITY_MISMATCH")
        return revision

    # 功能：
    #   取得线程指向的当前计划并验证关系；调用方事务负责固定数据库读取快照。
    # 输入：
    #   self：当前存储。
    #   thread：从当前事务读取的线程。
    # 输出：
    #   revision：属于此线程且状态相符的当前计划。
    def _current_revision(self, thread: TaskThread) -> PlanRevisionRecord:
        if thread.current_plan_revision_id is None:
            raise LifecycleTransitionError("TASK_HAS_NO_PLAN_REVISION")
        revision = self._revision(thread.current_plan_revision_id)
        if revision is None:
            raise LifecycleTransitionError("TASK_PLAN_REVISION_MISSING")
        _require_binding(thread, revision)
        return revision

    # 功能：
    #   提供单条任务快照，不暴露数据库连接或赋予执行权限。
    # 输入：
    #   self：当前存储。
    #   conversation_id：目标会话标识。
    # 输出：
    #   thread：独立任务记录，缺失时为 None。
    def get_thread(self, conversation_id: str) -> TaskThread | None:
        thread = self._thread(conversation_id)
        return thread

    # 功能：
    #   提供精确计划的只读快照；读到计划不代表用户已确认执行。
    # 输入：
    #   self：当前存储。
    #   plan_revision_id：目标计划标识。
    # 输出：
    #   revision：独立计划记录，缺失时为 None。
    def get_revision(self, plan_revision_id: str) -> PlanRevisionRecord | None:
        revision = self._revision(plan_revision_id)
        return revision

    # 功能：
    #   原子地复用或创建会话的稳定任务身份，失败时回滚，不重置已有执行状态。
    # 输入：
    #   self：当前存储。
    #   conversation_id：用户任务所在会话。
    # 输出：
    #   thread：已有或新创建的独立任务快照。
    def ensure_thread(self, conversation_id: str) -> TaskThread:
        self._begin()
        try:
            existing = self._thread(conversation_id)
            if existing is not None:
                self._connection.commit()
                return existing
            now = _now()
            thread = TaskThread(
                conversation_id=conversation_id,
                mission_id=f"mission-{uuid4().hex}",
                state="planning",
                created_at=now,
                updated_at=now,
            )
            self._connection.execute(
                "INSERT INTO task_threads(conversation_id, mission_id, thread_json) "
                "VALUES (?, ?, ?)",
                (conversation_id, thread.mission_id, thread.model_dump_json()),
            )
            self._connection.commit()
            return thread
        except BaseException:
            self._connection.rollback()
            raise

    # 功能：
    #   1. 为未执行的任务发布新提案，并原子替代旧的待确认提案。
    #   2. 保留任务身份和历史关系；活动飞行、缺失父计划或跨任务指针一律拒绝。
    # 输入：
    #   self：当前存储。
    #   conversation_id：提案所属会话。
    #   contract_id：结构化任务合同标识。
    #   prepared_mission_sha256：本次冻结任务的摘要。
    #   source_message_sha256：产生本次计划的用户消息摘要。
    # 输出：
    #   binding：已持久化且等待用户确认的新任务与计划绑定。
    def record_plan_revision(
        self,
        *,
        conversation_id: str,
        contract_id: str,
        prepared_mission_sha256: str,
        source_message_sha256: str,
    ) -> MissionLifecycleBinding:
        self._begin()
        try:
            thread = self._thread(conversation_id)
            if thread is None:
                now = _now()
                thread = TaskThread(
                    conversation_id=conversation_id,
                    mission_id=f"mission-{uuid4().hex}",
                    state="planning",
                    created_at=now,
                    updated_at=now,
                )
                self._connection.execute(
                    "INSERT INTO task_threads(conversation_id, mission_id, thread_json) "
                    "VALUES (?, ?, ?)",
                    (conversation_id, thread.mission_id, thread.model_dump_json()),
                )
            if thread.state in {"executing", "holding", "landing"}:
                raise LifecycleTransitionError("ACTIVE_EXECUTION_REJECTS_PREFLIGHT_REPLAN")

            parent_id = thread.current_plan_revision_id
            parent = self._current_revision(thread) if parent_id else None
            if parent is None and (
                thread.state != "planning" or thread.active_execution_id is not None
            ):
                raise LifecycleTransitionError("TASK_PLAN_REVISION_MISSING")
            row = self._connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM plan_revisions "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            revision_number = int(row[0])
            if parent is not None and parent.status in {"proposed", "confirmed"}:
                superseded = parent.model_copy(update={"status": "superseded"})
                self._connection.execute(
                    "UPDATE plan_revisions SET record_json = ? WHERE plan_revision_id = ?",
                    (superseded.model_dump_json(), superseded.plan_revision_id),
                )

            record = PlanRevisionRecord(
                plan_revision_id=f"plan-{uuid4().hex}",
                conversation_id=conversation_id,
                mission_id=thread.mission_id,
                revision=revision_number,
                parent_plan_revision_id=parent_id,
                status="proposed",
                contract_id=contract_id,
                prepared_mission_sha256=prepared_mission_sha256,
                source_message_sha256=source_message_sha256,
                created_at=_now(),
            )
            self._connection.execute(
                "INSERT INTO plan_revisions(plan_revision_id, conversation_id, revision, "
                "record_json) VALUES (?, ?, ?, ?)",
                (
                    record.plan_revision_id,
                    conversation_id,
                    record.revision,
                    record.model_dump_json(),
                ),
            )
            updated_thread = thread.model_copy(
                update={
                    "state": "awaiting_confirmation",
                    "current_plan_revision_id": record.plan_revision_id,
                    "active_execution_id": None,
                    "updated_at": _now(),
                }
            )
            self._connection.execute(
                "UPDATE task_threads SET thread_json = ? WHERE conversation_id = ?",
                (updated_thread.model_dump_json(), conversation_id),
            )
            self._connection.commit()
            binding = MissionLifecycleBinding(thread=updated_thread, plan_revision=record)
            return binding
        except BaseException:
            self._connection.rollback()
            raise

    # 功能：
    #   一次性消费当前提案的确认，核对任务、合同及冻结摘要后原子签发执行身份。
    # 输入：
    #   self：当前存储。
    #   conversation_id：确认所属会话。
    #   plan_revision_id：用户确认的精确计划。
    #   contract_id：已展示并确认的合同标识。
    #   prepared_mission_sha256：即将执行的冻结任务摘要。
    # 输出：
    #   execution_binding：执行中线程、执行中计划及新执行标识组成的三元组。
    def confirm_execution(
        self,
        *,
        conversation_id: str,
        plan_revision_id: str,
        contract_id: str,
        prepared_mission_sha256: str,
    ) -> tuple[TaskThread, PlanRevisionRecord, str]:
        self._begin()
        try:
            thread = self._thread(conversation_id)
            record = self._revision(plan_revision_id)
            if thread is None or record is None:
                raise LifecycleTransitionError("UNKNOWN_TASK_OR_PLAN_REVISION")
            gates = {
                "thread_awaiting_confirmation": thread.state == "awaiting_confirmation",
                "no_existing_execution": thread.active_execution_id is None,
                "current_revision_matches": (thread.current_plan_revision_id == plan_revision_id),
                "revision_is_proposed": record.status == "proposed",
                "conversation_matches": record.conversation_id == conversation_id,
                "mission_matches": record.mission_id == thread.mission_id,
                "contract_matches": record.contract_id == contract_id,
                "prepared_hash_matches": (
                    record.prepared_mission_sha256 == prepared_mission_sha256
                ),
            }
            if not all(gates.values()):
                failed = ",".join(name for name, passed in gates.items() if not passed)
                raise LifecycleTransitionError(f"EXECUTION_CONFIRMATION_REJECTED:{failed}")
            execution_id = f"execution-{uuid4().hex}"
            executing_record = record.model_copy(update={"status": "executing"})
            executing_thread = thread.model_copy(
                update={
                    "state": "executing",
                    "active_execution_id": execution_id,
                    "updated_at": _now(),
                }
            )
            self._connection.execute(
                "UPDATE plan_revisions SET record_json = ? WHERE plan_revision_id = ?",
                (executing_record.model_dump_json(), plan_revision_id),
            )
            self._connection.execute(
                "UPDATE task_threads SET thread_json = ? WHERE conversation_id = ?",
                (executing_thread.model_dump_json(), conversation_id),
            )
            self._connection.commit()
            execution_binding = executing_thread, executing_record, execution_id
            return execution_binding
        except BaseException:
            self._connection.rollback()
            raise

    # 功能：
    #   转换当前执行状态；结束任务与结束对应计划必须同一事务完成，终态不能恢复飞行。
    # 输入：
    #   self：当前存储。
    #   conversation_id：目标会话。
    #   execution_id：已签发且仍匹配当前任务的执行身份。
    #   state：请求进入的执行、悬停、降落、完成或失败状态。
    # 输出：
    #   updated：持久化成功后的任务快照。
    def set_execution_state(
        self,
        *,
        conversation_id: str,
        execution_id: str,
        state: str,
    ) -> TaskThread:
        if state not in {"executing", "holding", "landing", "completed", "failed"}:
            raise ValueError(f"unsupported execution state: {state}")
        self._begin()
        try:
            thread = self._thread(conversation_id)
            if thread is None or thread.active_execution_id != execution_id:
                raise LifecycleTransitionError("EXECUTION_ID_NOT_ACTIVE_FOR_TASK")
            allowed = {
                "executing": {"executing", "holding", "landing", "completed", "failed"},
                "holding": {"holding", "executing", "landing", "failed"},
                "landing": {"landing", "completed", "failed"},
            }
            if thread.state not in allowed or state not in allowed[thread.state]:
                raise LifecycleTransitionError(
                    f"INVALID_EXECUTION_TRANSITION:{thread.state}->{state}"
                )
            revision = self._current_revision(thread)
            updated = thread.model_copy(update={"state": state, "updated_at": _now()})
            if state in {"completed", "failed"}:
                # Finalize task and revision together. A crash must not leave a
                # completed task paired with an independently executable proposal.
                final_revision = revision.model_copy(update={"status": state})
                self._connection.execute(
                    "UPDATE plan_revisions SET record_json = ? WHERE plan_revision_id = ?",
                    (final_revision.model_dump_json(), final_revision.plan_revision_id),
                )
            self._connection.execute(
                "UPDATE task_threads SET thread_json = ? WHERE conversation_id = ?",
                (updated.model_dump_json(), conversation_id),
            )
            self._connection.commit()
            return updated
        except BaseException:
            self._connection.rollback()
            raise

    # 功能：
    #   在同一读取快照中组装任务及精确当前计划，拒绝缺失或不一致的绑定。
    # 输入：
    #   self：当前存储。
    #   conversation_id：目标会话。
    # 输出：
    #   binding：校验过的独立任务与计划快照。
    def binding(self, conversation_id: str) -> MissionLifecycleBinding:
        # 保存点可嵌套在所有者事务中，既固定两次读取的快照，也不提交所有者事务。
        self._connection.execute("SAVEPOINT lifecycle_binding_read")
        try:
            thread = self._thread(conversation_id)
            if thread is None:
                raise LifecycleTransitionError("TASK_HAS_NO_PLAN_REVISION")
            revision = self._current_revision(thread)
            binding = MissionLifecycleBinding(thread=thread, plan_revision=revision)
        except BaseException:
            self._connection.execute("ROLLBACK TO lifecycle_binding_read")
            raise
        finally:
            self._connection.execute("RELEASE lifecycle_binding_read")
        return binding

    # 功能：
    #   1. 重验证并隔离准备任务边车，核对内部身份、状态及已有持久化记录。
    #   2. 只允许原子插入缺失记录或复用相同记录；旧边车不得覆盖活动或已替代任务。
    # 输入：
    #   self：目标存储。
    #   binding：待恢复的任务与计划边车。
    # 输出：
    #   imported：导入成功的独立绑定快照。
    def import_binding(self, binding: MissionLifecycleBinding) -> MissionLifecycleBinding:
        try:
            imported = MissionLifecycleBinding.model_validate(
                binding.model_dump(mode="python"), strict=True
            )
        except ValueError as error:
            raise LifecycleTransitionError("TASK_PLAN_BINDING_INVALID") from error
        _require_binding(imported.thread, imported.plan_revision)
        binding = imported
        self._begin()
        try:
            existing_thread = self._thread(binding.thread.conversation_id)
            existing_revision = self._revision(binding.plan_revision.plan_revision_id)
            if existing_thread is None:
                self._connection.execute(
                    "INSERT INTO task_threads(conversation_id, mission_id, thread_json) "
                    "VALUES (?, ?, ?)",
                    (
                        binding.thread.conversation_id,
                        binding.thread.mission_id,
                        binding.thread.model_dump_json(),
                    ),
                )
            elif existing_thread != binding.thread:
                raise LifecycleTransitionError("TASK_THREAD_SIDECAR_DIVERGENCE")
            if existing_revision is None:
                self._connection.execute(
                    "INSERT INTO plan_revisions(plan_revision_id, conversation_id, revision, "
                    "record_json) VALUES (?, ?, ?, ?)",
                    (
                        binding.plan_revision.plan_revision_id,
                        binding.plan_revision.conversation_id,
                        binding.plan_revision.revision,
                        binding.plan_revision.model_dump_json(),
                    ),
                )
            elif existing_revision != binding.plan_revision:
                raise LifecycleTransitionError("PLAN_REVISION_SIDECAR_DIVERGENCE")
            self._connection.commit()
            return imported
        except BaseException:
            self._connection.rollback()
            raise

    # 功能：
    #   输出经过成对校验的可读生命周期记录，不附加执行权限或数据库句柄。
    # 输入：
    #   self：当前存储。
    #   conversation_id：目标会话。
    # 输出：
    #   exported：任务与当前计划的 JSON 文本。
    def export_debug_json(self, conversation_id: str) -> str:
        exported = json.dumps(
            self.binding(conversation_id).model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return exported
