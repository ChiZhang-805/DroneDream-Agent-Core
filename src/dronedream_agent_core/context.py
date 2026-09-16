"""Durable bounded context storage; model-side response IDs are not the sole memory."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import ConversationEvent, ConversationWindow
from .lifecycle import MissionLifecycleStore


class ContextStore:
    """One thread-affine connection; separate processes share the durable WAL database."""

    # 功能：
    #   创建本地上下文连接及表结构，不依赖模型供应商保存对话；初始化失败时回收连接。
    # 输入：
    #   path：持久化 SQLite 文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        try:
            self._initialize_tables()
        except BaseException:
            self._connection.close()
            raise

    # 功能：
    #   在当前连接建立事件、摘要和供应商续接表，启用 WAL 与外键并接入任务生命周期存储。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _initialize_tables(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              conversation_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              event_json TEXT NOT NULL,
              PRIMARY KEY (conversation_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS summaries (
              conversation_id TEXT PRIMARY KEY,
              summary TEXT NOT NULL,
              through_sequence INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provider_context (
              conversation_id TEXT NOT NULL,
              provider TEXT NOT NULL,
              response_id TEXT NOT NULL,
              PRIMARY KEY (conversation_id, provider)
            );
            """
        )
        self._connection.commit()
        self.lifecycle = MissionLifecycleStore(self._connection)

    # 功能：
    #   关闭当前实例持有的连接，保留磁盘事件供后续会话继续读取。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._connection.close()

    # 功能：
    #   1. 校验并隔离事件载荷，拒绝自定义对象和非有限数值。
    #   2. 在写事务内分配对话序号并落库，防止独立连接并发写入时取得相同序号。
    # 输入：
    #   conversation_id：事件所属对话标识。
    #   role：事件发起角色。
    #   event_type：业务事件类型。
    #   payload：符合 JSON 约束的事件载荷。
    # 输出：
    #   event：已持久化且带唯一标识、序号和时间的事件模型。
    def append(
        self,
        conversation_id: str,
        *,
        role: str,
        event_type: str,
        payload: dict[str, object],
    ) -> ConversationEvent:
        # 先校验再进入事务，既避免非法值被变成 null，也减少持有数据库写锁的时间。
        payload = copy_json(payload)
        with self._connection:
            # 必须在读取 MAX 之前获得写锁；只在 INSERT 时加锁无法防止重复取号。
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            event = ConversationEvent(
                event_id=f"event-{uuid4().hex[:24]}",
                conversation_id=conversation_id,
                sequence=int(row[0]),
                role=role,
                event_type=event_type,
                payload=payload,
                created_at=datetime.now(UTC),
            )
            self._connection.execute(
                "INSERT INTO events(conversation_id, sequence, event_json) VALUES (?, ?, ?)",
                (conversation_id, event.sequence, event.model_dump_json()),
            )
        return event

    # 功能：
    #   保存覆盖已持久化事件的摘要，拒绝超过实际事件的游标和落后于当前摘要的迟到结果。
    # 输入：
    #   conversation_id：摘要所属对话标识。
    #   summary：本次生成的摘要文本。
    #   through_sequence：摘要已覆盖的最后一个事件序号。
    # 输出：
    #   None：不返回业务数据。
    def set_summary(self, conversation_id: str, summary: str, through_sequence: int) -> None:
        if type(through_sequence) is not int or not 1 <= through_sequence <= 2**63 - 1:
            raise ValueError("through_sequence must be a positive SQLite integer")
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            maximum = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if through_sequence > maximum:
                raise ValueError("summary cannot cover events that have not been persisted")
            previous = self._connection.execute(
                "SELECT through_sequence FROM summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if previous is not None and through_sequence < previous[0]:
                raise ValueError("summary is older than the current persisted summary")
            self._connection.execute(
                """INSERT INTO summaries(conversation_id, summary, through_sequence)
                   VALUES (?, ?, ?)
                   ON CONFLICT(conversation_id) DO UPDATE SET
                     summary=excluded.summary, through_sequence=excluded.through_sequence""",
                (conversation_id, summary, through_sequence),
            )

    # 功能：
    #   按对话和供应商保存续接标识；该标识只用于供应商续接，不替代本地事件与摘要。
    # 输入：
    #   conversation_id：所属对话标识。
    #   provider：模型供应商标识。
    #   response_id：供应商返回的续接响应标识。
    # 输出：
    #   None：不返回业务数据。
    def set_response_id(self, conversation_id: str, provider: str, response_id: str) -> None:
        self._connection.execute(
            """INSERT INTO provider_context(conversation_id, provider, response_id)
               VALUES (?, ?, ?)
               ON CONFLICT(conversation_id, provider) DO UPDATE SET
                 response_id=excluded.response_id""",
            (conversation_id, provider, response_id),
        )
        self._connection.commit()

    # 功能：
    #   读取最新一页未摘要事件并按时间顺序返回，适用于当前模型调用的近期上下文。
    # 输入：
    #   conversation_id：待读取的对话标识。
    #   max_recent_events：最多返回的事件数量。
    # 输出：
    #   window：包含已有摘要、近期事件与供应商续接标识的窗口。
    def window(self, conversation_id: str, *, max_recent_events: int = 24) -> ConversationWindow:
        window = self._window(conversation_id, max_recent_events=max_recent_events, latest=True)
        return window

    # 功能：
    #   从摘要游标之后的最早事件开始分页，防止长对话因读取最新页而跳过尚未摘要的中间页。
    # 输入：
    #   conversation_id：待摘要的对话标识。
    #   max_recent_events：本次最多消费的事件数量。
    # 输出：
    #   window：按游标连续向前推进的待摘要窗口。
    def summary_window(
        self, conversation_id: str, *, max_recent_events: int = 200
    ) -> ConversationWindow:
        window = self._window(conversation_id, max_recent_events=max_recent_events, latest=False)
        return window

    # 功能：
    #   查询有限页事件并组合摘要和续接标识；排序方向只由代码选择，业务标识通过参数绑定。
    # 输入：
    #   conversation_id：待读取的对话标识。
    #   max_recent_events：一至两百之间的事件数上限。
    #   latest：为 True 时取最新页，否则取最早的未摘要页。
    # 输出：
    #   window：按正序排列事件的上下文窗口。
    def _window(
        self, conversation_id: str, *, max_recent_events: int, latest: bool
    ) -> ConversationWindow:
        if type(max_recent_events) is not int or not 1 <= max_recent_events <= 200:
            raise ValueError("max_recent_events must be between 1 and 200")
        summary_row = self._connection.execute(
            "SELECT summary, through_sequence FROM summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        through_sequence = int(summary_row[1]) if summary_row else 0
        order = "DESC" if latest else "ASC"
        rows = self._connection.execute(
            f"""SELECT event_json FROM events
               WHERE conversation_id = ? AND sequence > ?
               ORDER BY sequence {order} LIMIT ?""",
            (conversation_id, through_sequence, max_recent_events),
        ).fetchall()
        ordered_rows = reversed(rows) if latest else rows
        events = [ConversationEvent.model_validate(json.loads(row[0])) for row in ordered_rows]
        response_rows = self._connection.execute(
            "SELECT provider, response_id FROM provider_context WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        window = ConversationWindow(
            conversation_id=conversation_id,
            summary=summary_row[0] if summary_row else None,
            recent_events=events,
            previous_response_ids={str(provider): str(value) for provider, value in response_rows},
        )
        return window

    # 功能：
    #   按调用方明确选择的保留数量删除本对话最早事件，保留最新序号锚点，不触及其他对话。
    # 输入：
    #   conversation_id：应用保留规则的对话标识。
    #   maximum_events：希望保留的最近事件数，必须介于二十四和十万之间。
    # 输出：
    #   deleted_count：本次实际删除的事件行数。
    def apply_retention(self, conversation_id: str, *, maximum_events: int) -> int:
        if type(maximum_events) is not int or not 24 <= maximum_events <= 100_000:
            raise ValueError("maximum_events must be between 24 and 100000")
        threshold_row = self._connection.execute(
            """SELECT sequence FROM events WHERE conversation_id = ?
               ORDER BY sequence DESC LIMIT 1 OFFSET ?""",
            (conversation_id, maximum_events - 1),
        ).fetchone()
        if threshold_row is None:
            deleted_count = 0
            return deleted_count
        threshold = int(threshold_row[0])
        cursor = self._connection.execute(
            "DELETE FROM events WHERE conversation_id = ? AND sequence < ?",
            (conversation_id, threshold),
        )
        self._connection.commit()
        deleted_count = int(cursor.rowcount)
        return deleted_count
