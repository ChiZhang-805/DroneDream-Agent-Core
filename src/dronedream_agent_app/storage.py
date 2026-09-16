"""Durable local application state, separate from immutable mission evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Concatenate, Literal, ParamSpec, TypeVar
from uuid import uuid4

from dronedream_agent_core.asset_packages import (
    AssetImportJob,
    DDPkgManifest,
    InspectedDDPkg,
)
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationJob,
    AssetPairQualificationReceipt,
)
from dronedream_agent_core.contracts import (
    MissionAssetPairQualificationBinding,
)
from dronedream_agent_core.model_harness.boundary import ModelHarnessExecutionAuthority
from dronedream_agent_core.plugin_contracts import (
    PluginGovernancePolicy,
    PluginManifest,
    PluginSnapshot,
)
from dronedream_plugin_sdk.protocol import decode_json, encode_json


# 功能：
#   生成持久化记录使用的 UTC 时间；不用于测量传感器新鲜度或控制耗时。
# 输入：
#   无。
# 输出：
#   timestamp：包含时区的 ISO 时间字符串。
def utc_now() -> str:
    timestamp = datetime.now(UTC).isoformat()
    return timestamp


class AssetImportError(ValueError):
    """An imported asset bundle failed validation."""


_P = ParamSpec("_P")
_T = TypeVar("_T")
_STATE_JSON_LIMIT = 64 * 1024 * 1024
_STATE_JSON_NODES = 2_000_000


# 功能：
#   1. 用可重入锁串行化共享连接的读、检查、写及返回值构造，防止读到其他线程的未提交数据。
#   2. 锁不合并独立事务；包含文件验收的存储调用可能阻塞，不应放在实时控制回调内。
# 输入：
#   operation：需要共享存储锁的方法。
# 输出：
#   locked：保留原方法元数据及参数类型的包装方法。
def _serialized(
    operation: Callable[Concatenate[AppStore, _P], _T],
) -> Callable[Concatenate[AppStore, _P], _T]:
    # 功能：
    #   在当前存储锁内调用原方法，异常原样交还调用方。
    # 输入：
    #   self：持有共享连接与可重入锁的存储实例。
    #   args：原方法的位置参数。
    #   kwargs：原方法的关键字参数。
    # 输出：
    #   result：原方法的返回值。
    @wraps(operation)
    def locked(self: AppStore, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        with self._lock:
            result = operation(self, *args, **kwargs)
            return result

    return locked


# 功能：
#   严格校验调用方或 JSON 中的布尔值，拒绝文本 false、整数和空值冒充开关。
# 输入：
#   value：待校验的开关值。
# 输出：
#   value：原始布尔值。
def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("APP_STATE_BOOLEAN_REQUIRED")
    return value


# 功能：
#   按声明长度加一读取资格回执，限制增长文件的读取量，并拒绝截短或增长后的内容。
#   后续摘要及语义检查由调用者负责。
# 输入：
#   path：资格回执文件路径。
#   expected_size：清单声明的字节数，必须为 1 至 8 MiB 的整数。
# 输出：
#   payload：长度符合声明的回执字节。
def _qualification_payload(path: Path, expected_size: int) -> bytes:
    if type(expected_size) is not int or not 0 < expected_size <= 8 * 1024 * 1024:
        raise ValueError("qualification evidence size invalid")
    with path.open("rb") as stream:
        payload = stream.read(expected_size + 1)
    if len(payload) != expected_size:
        raise ValueError("qualification evidence size changed")
    return payload


class AppStore:
    """Local mutable application state; neither cloud authority nor flight evidence."""

    # 功能：
    #   1. 创建本机可变状态目录及 SQLite 表，开启 WAL、外键与共享连接锁。
    #   2. 只添加缺失字段，保留旧用户记录；这里不保存或授予云端订阅、计费权限。
    # 输入：
    #   self：待初始化的存储实例。
    #   root：应用管理的本机数据根目录。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_root = self.root / "assets"
        self.assets_root.mkdir(exist_ok=True)
        self.asset_quarantine_root = self.root / "asset-quarantine"
        self.asset_quarantine_root.mkdir(exist_ok=True)
        self.asset_versions_root = self.root / "asset-versions"
        self.asset_versions_root.mkdir(exist_ok=True)
        self.asset_qualification_root = self.root / "asset-qualifications"
        self.asset_qualification_root.mkdir(exist_ok=True)
        self.missions_root = self.root / "missions"
        self.missions_root.mkdir(exist_ok=True)
        self.attachments_root = self.root / "attachments"
        self.attachments_root.mkdir(exist_ok=True)
        self.plugin_staging_root = self.root / "plugin-staging"
        self.plugin_staging_root.mkdir(exist_ok=True)
        self.plugins_root = self.root / "plugins"
        self.plugins_root.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._plugin_batch_active = False
        self._connection = sqlite3.connect(self.root / "autonomy.sqlite3", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
              thread_id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              state TEXT NOT NULL,
              selected_model TEXT NOT NULL,
              selected_map_id TEXT,
              selected_map_content_sha256 TEXT,
              selected_vehicle_id TEXT,
              selected_vehicle_content_sha256 TEXT,
              locale TEXT NOT NULL DEFAULT 'zh-CN',
              pinned INTEGER NOT NULL DEFAULT 0,
              archived INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              role TEXT NOT NULL,
              kind TEXT NOT NULL,
              content TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(thread_id, sequence),
              FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS assets (
              asset_id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              name TEXT NOT NULL,
              status TEXT NOT NULL,
              bundle_root TEXT NOT NULL,
              manifest_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              CHECK(kind IN ('map', 'vehicle'))
            );
            CREATE TABLE IF NOT EXISTS asset_import_jobs (
              job_id TEXT PRIMARY KEY,
              job_json TEXT NOT NULL,
              source_format TEXT NOT NULL,
              source_path TEXT NOT NULL,
              package_sha256 TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS asset_versions (
              asset_id TEXT NOT NULL,
              content_sha256 TEXT NOT NULL,
              kind TEXT NOT NULL,
              maturity TEXT NOT NULL,
              bundle_root TEXT NOT NULL,
              manifest_json TEXT NOT NULL,
              asset_ir_json TEXT NOT NULL,
              imported_at TEXT NOT NULL,
              PRIMARY KEY(asset_id, content_sha256),
              CHECK(kind IN ('map', 'world', 'vehicle'))
            );
            CREATE TABLE IF NOT EXISTS asset_pair_qualification_jobs (
              job_id TEXT PRIMARY KEY,
              job_json TEXT NOT NULL,
              map_asset_id TEXT NOT NULL,
              map_content_sha256 TEXT NOT NULL,
              vehicle_asset_id TEXT NOT NULL,
              vehicle_content_sha256 TEXT NOT NULL,
              workspace_root TEXT NOT NULL,
              map_bundle_root TEXT NOT NULL,
              vehicle_bundle_root TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attachments (
              attachment_id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              display_name TEXT NOT NULL,
              content_type TEXT NOT NULL,
              byte_size INTEGER NOT NULL,
              local_path TEXT NOT NULL,
              extracted_text TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS plugins (
              plugin_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              authority TEXT NOT NULL,
              enabled INTEGER NOT NULL,
              builtin INTEGER NOT NULL,
              description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plugin_versions (
              plugin_id TEXT NOT NULL,
              version TEXT NOT NULL,
              package_sha256 TEXT NOT NULL,
              bundle_root TEXT NOT NULL,
              manifest_json TEXT NOT NULL,
              installed_at TEXT NOT NULL,
              PRIMARY KEY(plugin_id, version)
            );
            CREATE TABLE IF NOT EXISTS plugin_events (
              receipt_id TEXT PRIMARY KEY,
              plugin_id TEXT NOT NULL,
              operation TEXT NOT NULL,
              accepted INTEGER NOT NULL,
              receipt_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plugin_governance_events (
              decision_id TEXT PRIMARY KEY,
              plugin_id TEXT NOT NULL,
              operation TEXT NOT NULL,
              accepted INTEGER NOT NULL,
              decision_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plugin_usage_events (
              invocation_id TEXT PRIMARY KEY,
              plugin_id TEXT NOT NULL,
              plugin_version TEXT NOT NULL,
              capability_id TEXT NOT NULL,
              slot_id TEXT NOT NULL,
              invocation_kind TEXT NOT NULL,
              outcome TEXT NOT NULL,
              duration_ms REAL NOT NULL,
              input_bytes INTEGER NOT NULL,
              output_bytes INTEGER NOT NULL,
              issue_code TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS plugin_usage_plugin_created_idx
              ON plugin_usage_events(plugin_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS plugin_configurations (
              plugin_id TEXT PRIMARY KEY,
              configuration_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(plugin_id) REFERENCES plugins(plugin_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS task_plugin_snapshots (
              snapshot_id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              catalog_sha256 TEXT NOT NULL,
              snapshot_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS model_harness_execution_authorities (
              authority_id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              plan_revision_id TEXT NOT NULL,
              authority_sha256 TEXT NOT NULL,
              authority_json TEXT NOT NULL,
              status TEXT NOT NULL,
              issued_at TEXT NOT NULL,
              consumed_at TEXT,
              execution_id TEXT,
              FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
              CHECK(status IN ('issued', 'consumed', 'superseded'))
            );
            CREATE INDEX IF NOT EXISTS execution_authority_thread_status_idx
              ON model_harness_execution_authorities(thread_id, status, issued_at DESC);
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS custom_models (
              profile_id TEXT PRIMARY KEY,
              display_name TEXT NOT NULL,
              provider TEXT NOT NULL,
              icon TEXT NOT NULL,
              base_url TEXT NOT NULL,
              api_style TEXT NOT NULL,
              model_id TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connector_credentials (
              reference TEXT PRIMARY KEY,
              display_name TEXT NOT NULL,
              allowed_plugin_ids_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        self._migrate_plugin_columns()
        self._migrate_plugin_version_columns()
        self._migrate_thread_asset_version_columns()
        self._connection.commit()

    # 功能：
    #   等待已进入的存储操作释放锁后关闭连接；上层应先停止继续提交新操作。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # 功能：
    #   1. 独占一个写事务，成功提交，异常或取消时回滚。
    #   2. 普通嵌套事务在回滚处理外拒绝；显式插件批次内用保存点隔离子操作失败。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   connection：上下文期间独占的 SQLite 连接。
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection
        with self._lock:
            # Test before entering this transaction's rollback handler. A nested
            # BEGIN used to fail there and accidentally discard the outer write.
            if self._connection.in_transaction:
                if not self._plugin_batch_active:
                    raise RuntimeError("APP_STORE_NESTED_TRANSACTION")
                # Only an explicit lifecycle batch permits composing public
                # store methods. Savepoints preserve a method's rollback even
                # if the caller catches an error and continues the batch.
                savepoint = "plugin_" + uuid4().hex
                self._connection.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield connection
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                except BaseException:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
                return
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    # 功能：
    #   在一个数据库事务内发布插件元数据、选择来源和回执；失败全部回滚。
    #   只用于短暂数据库发布阶段，不包含进程排空、等待或文件系统事务。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   None：不返回业务数据。
    @contextmanager
    def atomic_plugin_changes(self) -> Iterator[None]:
        with self._lock:
            if self._plugin_batch_active or self._connection.in_transaction:
                raise RuntimeError("APP_STORE_NESTED_PLUGIN_BATCH")
            with self.transaction():
                self._plugin_batch_active = True
                try:
                    yield
                finally:
                    self._plugin_batch_active = False

    # 功能：
    #   补齐插件目录字段；首次迁移选择来源时只区分内置默认与外部显式选择，不伪造旧回执。
    # 输入：
    #   self：初始化中的存储实例。
    # 输出：
    #   None：不返回业务数据。
    def _migrate_plugin_columns(self) -> None:
        existing = {str(row[1]) for row in self._connection.execute("PRAGMA table_info(plugins)")}
        selection_columns_missing = (
            not {
                "selection_source",
                "selected_by",
                "selection_receipt_sha256",
                "harness_revision_sha256",
            }
            <= existing
        )
        additions = {
            "publisher": "TEXT NOT NULL DEFAULT ''",
            "runtime_kind": "TEXT NOT NULL DEFAULT 'builtin-python'",
            "status": "TEXT NOT NULL DEFAULT 'installed'",
            "health": "TEXT NOT NULL DEFAULT 'unknown'",
            "removable": "INTEGER NOT NULL DEFAULT 1",
            "package_sha256": "TEXT NOT NULL DEFAULT ''",
            "bundle_root": "TEXT NOT NULL DEFAULT ''",
            "manifest_json": "TEXT NOT NULL DEFAULT '{}'",
            "last_error": "TEXT",
            "installed_at": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
            "trust_status": "TEXT NOT NULL DEFAULT 'verified'",
            "trust_decision_json": "TEXT NOT NULL DEFAULT '{}'",
            "update_ring": "TEXT NOT NULL DEFAULT 'stable'",
            "selection_source": "TEXT NOT NULL DEFAULT 'explicit'",
            "selected_by": "TEXT NOT NULL DEFAULT 'account_configurable'",
            "selection_receipt_sha256": "TEXT",
            "harness_revision_sha256": "TEXT",
        }
        for name, definition in additions.items():
            if name not in existing:
                self._connection.execute(f"ALTER TABLE plugins ADD COLUMN {name} {definition}")
        if selection_columns_missing:
            # Existing external selections can only be attributed to the local
            # account surface. Built-in defaults are product-managed until an
            # explicit enable/profile action records a stronger provenance.
            self._connection.execute(
                "UPDATE plugins SET "
                "selection_source=CASE WHEN builtin=1 THEN 'product_managed_default' "
                "ELSE 'explicit' END,"
                "selected_by=CASE WHEN builtin=1 THEN 'product_managed' "
                "ELSE 'account_configurable' END,"
                "selection_receipt_sha256=NULL,harness_revision_sha256=NULL"
            )

    # 功能：
    #   为暂存版本添加独立信任字段；已有版本没有验签证据时默认未验证，不继承活动版本信任。
    # 输入：
    #   self：初始化中的存储实例。
    # 输出：
    #   None：不返回业务数据。
    def _migrate_plugin_version_columns(self) -> None:
        existing = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(plugin_versions)")
        }
        additions = {
            "trust_status": "TEXT NOT NULL DEFAULT 'unverified'",
            "trust_decision_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in additions.items():
            if name not in existing:
                self._connection.execute(
                    f"ALTER TABLE plugin_versions ADD COLUMN {name} {definition}"
                )

    # 功能：
    #   给旧任务表添加资产内容摘要和语言字段，保留原任务；缺失摘要保持空值而非猜测最新版。
    # 输入：
    #   self：初始化中的存储实例。
    # 输出：
    #   None：不返回业务数据。
    def _migrate_thread_asset_version_columns(self) -> None:
        existing = {str(row[1]) for row in self._connection.execute("PRAGMA table_info(threads)")}
        for name in (
            "selected_map_content_sha256",
            "selected_vehicle_content_sha256",
        ):
            if name not in existing:
                self._connection.execute(f"ALTER TABLE threads ADD COLUMN {name} TEXT")
        if "locale" not in existing:
            self._connection.execute(
                "ALTER TABLE threads ADD COLUMN locale TEXT NOT NULL DEFAULT 'zh-CN'"
            )

    # 功能：
    #   将查询行复制成独立字典，严格解码 SQLite 开关及有界 JSON，拒绝损坏或歧义内容。
    # 输入：
    #   row：本次查询返回的 SQLite 行。
    # 输出：
    #   value：去掉 JSON 后缀并解码后的记录副本。
    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, object]:
        value = dict(row)
        for key in ("pinned", "archived", "enabled", "builtin", "removable", "accepted"):
            if key in value:
                # SQLite 布尔列的合法存储形式只有整数 0/1；文本及其他整数不是有效授权。
                if type(value[key]) is not int or value[key] not in (0, 1):
                    raise ValueError("APP_STATE_STORED_BOOLEAN_INVALID")
                value[key] = bool(value[key])
        for key in (
            "metadata_json",
            "manifest_json",
            "value_json",
            "configuration_json",
            "snapshot_json",
            "receipt_json",
            "trust_decision_json",
            "allowed_plugin_ids_json",
            "decision_json",
            "asset_ir_json",
        ):
            if key in value:
                value[key.removesuffix("_json")] = decode_json(
                    value.pop(key), limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
                )
        return value

    # 功能：
    #   新建本地规划任务；任务存在本身不代表用户已确认执行。
    # 输入：
    #   self：当前存储实例。
    #   title：任务标题。
    #   selected_model：所选模型标识。
    #   locale：任务语言。
    # 输出：
    #   thread：已保存的任务及初始消息列表。
    @_serialized
    def create_thread(
        self,
        title: str,
        selected_model: str,
        locale: Literal["zh-CN", "en-US"] = "zh-CN",
    ) -> dict[str, object]:
        now = utc_now()
        thread_id = f"thread-{uuid4().hex}"
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO threads("
                "thread_id,title,state,selected_model,locale,created_at,updated_at"
                ") "
                "VALUES(?,?,'planning',?,?,?,?)",
                (thread_id, title, selected_model, locale, now, now),
            )
        thread = self.get_thread(thread_id)
        return thread

    # 功能：
    #   返回按置顶及更新时间排序的本地任务摘要，默认不显示已归档任务。
    # 输入：
    #   self：当前存储实例。
    #   include_archived：是否同时列出归档任务，必须为布尔值。
    # 输出：
    #   threads：已解码的任务摘要列表。
    @_serialized
    def list_threads(self, include_archived: bool = False) -> list[dict[str, object]]:
        where = "" if _boolean(include_archived) else "WHERE archived = 0"
        rows = self._connection.execute(
            f"SELECT * FROM threads {where} ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
        threads = [self._row(row) for row in rows]
        return threads

    # 功能：
    #   在同一存储锁下读取指定任务及按序消息，找不到任务则报错。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：待读取的任务标识。
    # 输出：
    #   thread：包含消息的独立任务字典。
    @_serialized
    def get_thread(self, thread_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        thread = self._row(row)
        message_rows = self._connection.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY sequence", (thread_id,)
        ).fetchall()
        thread["messages"] = [self._row(item) for item in message_rows]
        return thread

    # 功能：
    #   1. 仅修改允许的任务偏好；替换资产 ID 但未提供摘要时清空旧摘要。
    #   2. 本机记忆设置允许时另存最后选择；两次提交不是同一个原子事务。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：待修改的任务标识。
    #   changes：字段更新字典。
    # 输出：
    #   thread：修改后重新读取的任务。
    @_serialized
    def patch_thread(self, thread_id: str, changes: dict[str, object]) -> dict[str, object]:
        allowed = {
            "title",
            "selected_model",
            "selected_map_id",
            "selected_map_content_sha256",
            "selected_vehicle_id",
            "selected_vehicle_content_sha256",
            "locale",
            "pinned",
            "archived",
        }
        updates = {key: value for key, value in changes.items() if key in allowed}
        for key in ("pinned", "archived"):
            if key in updates:
                _boolean(updates[key])
        if "selected_map_id" in updates and "selected_map_content_sha256" not in updates:
            updates["selected_map_content_sha256"] = None
        if "selected_vehicle_id" in updates and "selected_vehicle_content_sha256" not in updates:
            updates["selected_vehicle_content_sha256"] = None
        if not updates:
            thread = self.get_thread(thread_id)
            return thread
        # 先检查存储中的开关，损坏值不能在任务已经提交后才被发现并用于记忆写入。
        settings = self.get_settings()
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [int(value) if isinstance(value, bool) else value for value in updates.values()]
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE threads SET {assignments} WHERE thread_id = ?", (*values, thread_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(thread_id)
        if settings["memory_enabled"] and settings["remember_asset_choices"]:
            remembered: dict[str, object] = {}
            if "selected_map_id" in updates:
                remembered["last_map_id"] = updates["selected_map_id"]
            if "selected_map_content_sha256" in updates:
                remembered["last_map_content_sha256"] = updates["selected_map_content_sha256"]
            if "selected_vehicle_id" in updates:
                remembered["last_vehicle_id"] = updates["selected_vehicle_id"]
            if "selected_vehicle_content_sha256" in updates:
                remembered["last_vehicle_content_sha256"] = updates[
                    "selected_vehicle_content_sha256"
                ]
            if remembered:
                self.patch_settings(remembered)
        thread = self.get_thread(thread_id)
        return thread

    # 功能：
    #   保存已知任务状态标签；状态转换是否合法及能否执行由执行器负责。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：目标任务标识。
    #   state：受支持的生命周期标签。
    # 输出：
    #   thread：更新后的任务。
    @_serialized
    def set_thread_state(self, thread_id: str, state: str) -> dict[str, object]:
        if state not in {
            "planning",
            "awaiting_confirmation",
            "executing",
            "holding",
            "landing",
            "completed",
            "failed",
        }:
            raise ValueError("THREAD_STATE_INVALID")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE threads SET state = ?, updated_at = ? WHERE thread_id = ?",
                (state, utc_now(), thread_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(thread_id)
        thread = self.get_thread(thread_id)
        return thread

    # 功能：
    #   在一个事务内分配任务消息序号、保存消息并更新任务时间，失败不留下半条消息。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：所属任务标识。
    #   role：消息角色。
    #   kind：消息类型。
    #   content：消息正文。
    #   metadata：可选的标准 JSON 元数据对象。
    # 输出：
    #   message：带持久化身份和序号的消息。
    @_serialized
    def append_message(
        self,
        thread_id: str,
        *,
        role: str,
        kind: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if metadata is not None and type(metadata) is not dict:
            raise ValueError("APP_MESSAGE_METADATA_OBJECT_REQUIRED")
        metadata_json = encode_json(
            metadata if metadata is not None else {},
            limit=_STATE_JSON_LIMIT,
            node_limit=_STATE_JSON_NODES,
        )
        now = utc_now()
        message_id = f"message-{uuid4().hex}"
        with self.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(thread_id)
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
                (
                    message_id,
                    thread_id,
                    sequence,
                    role,
                    kind,
                    content,
                    metadata_json,
                    now,
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
            )
        row = self._connection.execute(
            "SELECT * FROM messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        assert row is not None
        message = self._row(row)
        return message

    # 功能：
    #   将选中的本地文件复制到任务附件目录并保存文本预览；上传字节上限由入口先检查。
    #   文件复制和数据库提交不是一个事务，正文始终作为不可信输入而不是指令执行。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：所属任务标识。
    #   display_name：展示文件名。
    #   content_type：入口识别的媒体类型。
    #   source：入口已接收的本地源文件。
    # 输出：
    #   attachment：包含本地路径、大小及可选预览的记录。
    @_serialized
    def save_attachment(
        self,
        thread_id: str,
        *,
        display_name: str,
        content_type: str,
        source: Path,
    ) -> dict[str, object]:
        if (
            self._connection.execute(
                "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            is None
        ):
            raise KeyError(thread_id)
        attachment_id = f"attachment-{uuid4().hex}"
        thread_root = self.attachments_root / thread_id
        thread_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(display_name).suffix.lower()
        target = thread_root / f"{attachment_id}{suffix}"
        shutil.copy2(source, target)
        extractable = {
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".tsv",
            ".yaml",
            ".yml",
            ".py",
            ".toml",
        }
        extracted: str | None = None
        if suffix in extractable and target.stat().st_size <= 1024 * 1024:
            extracted = target.read_text(encoding="utf-8", errors="replace")[:12_000]
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO attachments VALUES(?,?,?,?,?,?,?,?)",
                (
                    attachment_id,
                    thread_id,
                    display_name,
                    content_type,
                    target.stat().st_size,
                    str(target),
                    extracted,
                    now,
                ),
            )
        attachment = self.get_attachment(attachment_id, thread_id)
        return attachment

    # 功能：
    #   同时按任务与附件标识读取记录，拒绝从另一任务取用附件。
    # 输入：
    #   self：当前存储实例。
    #   attachment_id：附件标识。
    #   thread_id：请求所属任务标识。
    # 输出：
    #   attachment：附件记录副本。
    @_serialized
    def get_attachment(self, attachment_id: str, thread_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM attachments WHERE attachment_id = ? AND thread_id = ?",
            (attachment_id, thread_id),
        ).fetchone()
        if row is None:
            raise KeyError(attachment_id)
        attachment = dict(row)
        return attachment

    # 功能：
    #   按调用方指定顺序拼接任务内附件预览；单份预览在保存时截断，总上下文预算由上层控制。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：请求所属任务标识。
    #   attachment_ids：需要包含的附件标识列表。
    # 输出：
    #   context：带文件名的预览文本；二进制附件只提供提示。
    @_serialized
    def attachment_context(self, thread_id: str, attachment_ids: list[str]) -> str:
        blocks: list[str] = []
        for attachment_id in attachment_ids:
            value = self.get_attachment(attachment_id, thread_id)
            extracted = value.get("extracted_text")
            if extracted:
                blocks.append(f"FILE {value['display_name']}:\n{extracted}")
            else:
                blocks.append(
                    f"FILE {value['display_name']}: binary attachment; no text extraction available"
                )
        context = "\n\n".join(blocks)
        return context

    # 功能：
    #   读取最新计划消息的元数据，拒绝非对象及歧义 JSON；计划不等于执行确认。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：目标任务标识。
    # 输出：
    #   value：最新计划的元数据对象。
    @_serialized
    def latest_plan(self, thread_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT metadata_json FROM messages WHERE thread_id = ? AND kind = 'plan' "
            "ORDER BY sequence DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise KeyError("PLAN_NOT_FOUND")
        value = decode_json(
            row["metadata_json"], limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
        )
        if not isinstance(value, dict):
            raise ValueError("PLAN_METADATA_INVALID")
        return value

    # 功能：
    #   在同一事务内保存经过重验的一次性执行绑定，并作废同任务尚未使用的旧绑定。
    # 输入：
    #   self：当前存储实例。
    #   authority：上层完成用户确认及内核校验后签发的执行绑定。
    # 输出：
    #   authority：重新校验后保存的绑定副本。
    @_serialized
    def issue_execution_authority(
        self, authority: ModelHarnessExecutionAuthority
    ) -> ModelHarnessExecutionAuthority:
        authority = ModelHarnessExecutionAuthority.model_validate_json(authority.model_dump_json())
        with self.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM threads WHERE thread_id = ?", (authority.thread_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(authority.thread_id)
            connection.execute(
                "UPDATE model_harness_execution_authorities SET status='superseded' "
                "WHERE thread_id=? AND status='issued'",
                (authority.thread_id,),
            )
            connection.execute(
                "INSERT INTO model_harness_execution_authorities("
                "authority_id,thread_id,plan_revision_id,authority_sha256,authority_json,"
                "status,issued_at,consumed_at,execution_id) VALUES(?,?,?,?,?,'issued',?,?,?)",
                (
                    authority.authority_id,
                    authority.thread_id,
                    authority.plan_revision_id,
                    authority.authority_sha256,
                    authority.model_dump_json(),
                    authority.issued_at.isoformat(),
                    None,
                    None,
                ),
            )
        return authority

    # 功能：
    #   读取执行绑定并核对独立索引身份、摘要和状态，损坏记录不能作为有效执行凭据。
    # 输入：
    #   self：当前存储实例。
    #   authority_id：指定的一次性绑定标识。
    # 输出：
    #   result：校验后的绑定对象与当前消耗状态。
    @_serialized
    def execution_authority(self, authority_id: str) -> tuple[ModelHarnessExecutionAuthority, str]:
        row = self._connection.execute(
            "SELECT * FROM model_harness_execution_authorities WHERE authority_id=?",
            (authority_id,),
        ).fetchone()
        if row is None:
            raise KeyError(authority_id)
        authority = ModelHarnessExecutionAuthority.model_validate_json(str(row["authority_json"]))
        if authority.authority_id != authority_id or any(
            getattr(authority, key) != row[key]
            for key in ("thread_id", "plan_revision_id", "authority_sha256")
        ):
            raise ValueError("EXECUTION_AUTHORITY_BINDING_MISMATCH")
        status = str(row["status"])
        if status not in {"issued", "consumed", "superseded"}:
            raise ValueError("EXECUTION_AUTHORITY_STATUS_INVALID")
        result = (authority, status)
        return result

    # 功能：
    #   原子消费完全一致且仍有效的执行绑定，拒绝重放、索引漂移及空执行标识。
    # 输入：
    #   self：当前存储实例。
    #   expected：执行器期望消费的完整绑定。
    #   execution_id：本次执行的非空标识。
    # 输出：
    #   stored：已成功消费的绑定。
    @_serialized
    def consume_execution_authority(
        self,
        expected: ModelHarnessExecutionAuthority,
        *,
        execution_id: str,
    ) -> ModelHarnessExecutionAuthority:
        if type(execution_id) is not str or not execution_id.strip():
            raise ValueError("EXECUTION_AUTHORITY_EXECUTION_ID_INVALID")
        with self.transaction() as connection:
            stored, status = self.execution_authority(expected.authority_id)
            if stored != expected:
                raise ValueError("EXECUTION_AUTHORITY_BINDING_MISMATCH")
            if status != "issued":
                raise ValueError("EXECUTION_AUTHORITY_NOT_ACTIVE")
            cursor = connection.execute(
                "UPDATE model_harness_execution_authorities SET status='consumed',"
                "consumed_at=?,execution_id=? WHERE authority_id=? AND status='issued'",
                (utc_now(), execution_id, expected.authority_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("EXECUTION_AUTHORITY_NOT_ACTIVE")
        return stored

    # 功能：
    #   保存导入进度并保留首次登记的隔离源身份；更新不能换源文件、格式或包摘要。
    # 输入：
    #   self：当前存储实例。
    #   job：待保存的导入任务。
    #   source_format：首次登记必填的源格式，更新可省略。
    #   source_path：首次登记必填的隔离目录内源文件。
    #   package_sha256：首次登记必填的源文件摘要。
    # 输出：
    #   result：保存后重读并验证的任务字典。
    @_serialized
    def save_asset_import_job(
        self,
        job: AssetImportJob,
        *,
        source_format: str | None = None,
        source_path: Path | None = None,
        package_sha256: str | None = None,
    ) -> dict[str, object]:
        job = AssetImportJob.model_validate_json(job.model_dump_json())
        existing = self._connection.execute(
            "SELECT source_format,source_path,package_sha256 FROM asset_import_jobs "
            "WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        if existing is None and None in (source_format, source_path, package_sha256):
            raise ValueError("ASSET_IMPORT_SOURCE_METADATA_MISSING")
        effective_format = (
            source_format if source_format is not None else str(existing["source_format"])
        )
        effective_path = (
            source_path if source_path is not None else Path(str(existing["source_path"]))
        )
        effective_sha256 = (
            package_sha256 if package_sha256 is not None else str(existing["package_sha256"])
        )
        if job.source_format != effective_format or job.package_sha256 != effective_sha256:
            raise ValueError("ASSET_IMPORT_SOURCE_METADATA_MISMATCH")
        quarantine = self.asset_quarantine_root.resolve()
        resolved_source = effective_path.resolve()
        if quarantine not in resolved_source.parents or not resolved_source.is_file():
            raise ValueError("ASSET_IMPORT_SOURCE_PATH_INVALID")
        if existing is not None and (
            effective_format != existing["source_format"]
            or effective_sha256 != existing["package_sha256"]
            or resolved_source != Path(str(existing["source_path"])).resolve()
        ):
            raise ValueError("ASSET_IMPORT_SOURCE_IDENTITY_CHANGED")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO asset_import_jobs VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET job_json=excluded.job_json,"
                "source_format=excluded.source_format,source_path=excluded.source_path,"
                "package_sha256=excluded.package_sha256,updated_at=excluded.updated_at",
                (
                    job.job_id,
                    job.model_dump_json(),
                    effective_format,
                    str(resolved_source),
                    effective_sha256,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
        result = self.get_asset_import_job(job.job_id)
        return result

    # 功能：
    #   重验导入任务的字段及源身份，并确认 JSON 任务标识与请求键一致。
    # 输入：
    #   self：当前存储实例。
    #   job_id：指定导入任务标识。
    # 输出：
    #   result：校验后的 JSON 兼容任务字典。
    @_serialized
    def get_asset_import_job(self, job_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT job_json,source_format,package_sha256 FROM asset_import_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        job = AssetImportJob.model_validate_json(str(row["job_json"]))
        if (
            job.job_id != job_id
            or job.source_format != str(row["source_format"])
            or job.package_sha256 != str(row["package_sha256"])
        ):
            raise AssetImportError("ASSET_IMPORT_SOURCE_METADATA_MISMATCH")
        result = job.model_dump(mode="json")
        return result

    # 功能：
    #   复用公开任务读取的身份校验，不让类型化读取绕过源一致性检查。
    # 输入：
    #   self：当前存储实例。
    #   job_id：指定导入任务标识。
    # 输出：
    #   job：通过验证的导入任务对象。
    @_serialized
    def load_asset_import_job(self, job_id: str) -> AssetImportJob:
        job = AssetImportJob.model_validate(self.get_asset_import_job(job_id))
        return job

    # 功能：
    #   每次重新解析源文件位置，拒绝离开隔离目录或不存在的文件；内容摘要由导入器重验。
    # 输入：
    #   self：当前存储实例。
    #   job_id：指定导入任务标识。
    # 输出：
    #   source：通过目录归属检查的绝对文件路径。
    @_serialized
    def get_asset_import_source(self, job_id: str) -> Path:
        self.get_asset_import_job(job_id)
        row = self._connection.execute(
            "SELECT source_path FROM asset_import_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        source = Path(str(row["source_path"])).resolve()
        if self.asset_quarantine_root.resolve() not in source.parents or not source.is_file():
            raise AssetImportError("ASSET_IMPORT_SOURCE_PATH_INVALID")
        return source

    # 功能：
    #   按更新时间倒序列出任务，每个成员都通过相同的身份重验入口。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   jobs：已校验的导入任务列表。
    @_serialized
    def list_asset_import_jobs(self) -> list[dict[str, object]]:
        rows = self._connection.execute(
            "SELECT job_id FROM asset_import_jobs ORDER BY updated_at DESC"
        ).fetchall()
        jobs = [self.get_asset_import_job(str(row["job_id"])) for row in rows]
        return jobs

    # 功能：
    #   通过统一批量入口登记一个已经完成包检查的资产，不在此重新进行仿真验收。
    # 输入：
    #   self：当前存储实例。
    #   inspected：包检查器生成的清单与资产 IR。
    #   bundle_root：资产版本目录内的包路径。
    # 输出：
    #   version：已登记的精确版本记录。
    @_serialized
    def record_asset_version(
        self,
        inspected: InspectedDDPkg,
        *,
        bundle_root: Path,
    ) -> dict[str, object]:
        version = self.record_asset_versions([(inspected, bundle_root)])[0]
        return version

    # 功能：
    #   在同一事务中登记一组已检查包的版本索引；路径必须位于资产版本目录下。
    # 输入：
    #   self：当前存储实例。
    #   versions：已检查包及对应目录的列表。
    # 输出：
    #   records：全部成功登记后重读的版本记录。
    @_serialized
    def record_asset_versions(
        self,
        versions: list[tuple[InspectedDDPkg, Path]],
    ) -> list[dict[str, object]]:
        if not versions:
            raise AssetImportError("ASSET_VERSIONS_EMPTY")
        prepared: list[tuple[InspectedDDPkg, Path, str, str]] = []
        versions_root = self.asset_versions_root.resolve()
        for inspected, bundle_root in versions:
            qualification = inspected.manifest.qualification
            maturity = qualification.maturity if qualification is not None else "visual_only"
            resolved_root = bundle_root.resolve()
            if versions_root not in resolved_root.parents:
                raise AssetImportError("ASSET_VERSION_PATH_INVALID")
            prepared.append((inspected, resolved_root, maturity, utc_now()))
        with self.transaction() as connection:
            for inspected, resolved_root, maturity, now in prepared:
                connection.execute(
                    "INSERT INTO asset_versions VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(asset_id,content_sha256) DO UPDATE SET "
                    "maturity=excluded.maturity,bundle_root=excluded.bundle_root,"
                    "manifest_json=excluded.manifest_json,asset_ir_json=excluded.asset_ir_json,"
                    "imported_at=excluded.imported_at",
                    (
                        inspected.manifest.asset_id,
                        inspected.manifest.content_sha256,
                        inspected.manifest.asset_kind,
                        maturity,
                        str(resolved_root),
                        inspected.manifest.model_dump_json(),
                        inspected.asset_ir.model_dump_json(),
                        now,
                    ),
                )
        records = [
            self.get_asset_version(
                inspected.manifest.asset_id,
                inspected.manifest.content_sha256,
            )
            for inspected, _resolved_root, _maturity, _now in prepared
        ]
        return records

    # 功能：
    #   列出已保留的资产版本索引；较新时间戳本身不代表具有飞行资格。
    # 输入：
    #   self：当前存储实例。
    #   asset_id：可选的资产筛选标识。
    # 输出：
    #   versions：按导入时间倒序排列的版本列表。
    @_serialized
    def list_asset_versions(self, asset_id: str | None = None) -> list[dict[str, object]]:
        rows = self._connection.execute(
            "SELECT * FROM asset_versions WHERE asset_id = ? ORDER BY imported_at DESC"
            if asset_id
            else "SELECT * FROM asset_versions ORDER BY imported_at DESC",
            (asset_id,) if asset_id else (),
        ).fetchall()
        versions = [self._row(row) for row in rows]
        return versions

    # 功能：
    #   只读取指定资产与摘要的记录，缺失时不自动替换成其他版本。
    # 输入：
    #   self：当前存储实例。
    #   asset_id：资产标识。
    #   content_sha256：要求使用的内容摘要。
    # 输出：
    #   version：精确匹配的版本记录。
    @_serialized
    def get_asset_version(self, asset_id: str, content_sha256: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM asset_versions WHERE asset_id = ? AND content_sha256 = ?",
            (asset_id, content_sha256),
        ).fetchone()
        if row is None:
            raise KeyError(f"{asset_id}@{content_sha256}")
        version = self._row(row)
        return version

    # 功能：
    #   1. 将明确指定的内置旧资产选择迁移到已合格的新摘要，不凭时间或名字选择版本。
    #   2. 拒绝仍被任务引用的源包；先暂存目录、提交索引，再清理目录，提交失败则恢复目录。
    #   3. 本入口只用于确认被替代的发布内置包；删除后的包内容须从原发布制品恢复。
    # 输入：
    #   self：当前存储实例。
    #   asset_id：源与替代包共同的资产标识。
    #   source_content_sha256：明确退役的源摘要。
    #   replacement_content_sha256：已安装并合格的替代摘要。
    # 输出：
    #   migrated：本次是否找到并移除了源版本索引。
    @_serialized
    def migrate_bundled_asset_version(
        self,
        *,
        asset_id: str,
        source_content_sha256: str,
        replacement_content_sha256: str,
    ) -> bool:
        if not asset_id or not all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (source_content_sha256, replacement_content_sha256)
        ):
            raise AssetImportError("BUNDLED_ASSET_MIGRATION_IDENTITY_INVALID")
        if source_content_sha256 == replacement_content_sha256:
            raise AssetImportError("BUNDLED_ASSET_MIGRATION_IDENTITY_INVALID")

        replacement = self.get_asset_version(asset_id, replacement_content_sha256)
        if replacement["maturity"] != "qualified":
            raise AssetImportError("BUNDLED_ASSET_MIGRATION_REPLACEMENT_NOT_QUALIFIED")

        source_row = self._connection.execute(
            "SELECT * FROM asset_versions WHERE asset_id = ? AND content_sha256 = ?",
            (asset_id, source_content_sha256),
        ).fetchone()
        source = self._row(source_row) if source_row is not None else None
        if source is not None:
            if (
                source["maturity"] not in {"visual_only", "qualified"}
                or source["kind"] != replacement["kind"]
            ):
                raise AssetImportError("BUNDLED_ASSET_MIGRATION_SOURCE_NOT_TRANSITIONAL")
            for row in self._connection.execute(
                "SELECT job_id FROM asset_pair_qualification_jobs"
            ).fetchall():
                job = self.load_asset_pair_qualification_job(str(row["job_id"]))
                source_is_active_input = job.state not in {"qualified", "failed", "cancelled"} and (
                    (
                        job.map_asset_id == asset_id
                        and job.map_content_sha256 == source_content_sha256
                    )
                    or (
                        job.vehicle_asset_id == asset_id
                        and job.vehicle_content_sha256 == source_content_sha256
                    )
                )
                source_is_qualified_result = (
                    job.map_asset_id == asset_id
                    and job.result_map_content_sha256 == source_content_sha256
                ) or (
                    job.vehicle_asset_id == asset_id
                    and job.result_vehicle_content_sha256 == source_content_sha256
                )
                if source_is_active_input or source_is_qualified_result:
                    raise AssetImportError("BUNDLED_ASSET_MIGRATION_SOURCE_IN_USE")

        versions_root = self.asset_versions_root.resolve()
        tombstone: Path | None = None
        source_root: Path | None = None
        if source is not None:
            source_root = Path(str(source["bundle_root"])).resolve()
            expected_root = (
                versions_root / str(source["kind"]) / asset_id / source_content_sha256
            ).resolve()
            if source_root != expected_root or versions_root not in source_root.parents:
                raise AssetImportError("BUNDLED_ASSET_MIGRATION_PATH_INVALID")
            if source_root.exists() and not source_root.is_dir():
                raise AssetImportError("BUNDLED_ASSET_MIGRATION_PATH_INVALID")
            if source_root.is_dir():
                tombstone = (versions_root / f".pruning-{uuid4().hex}").resolve()
                if versions_root not in tombstone.parents:
                    raise AssetImportError("BUNDLED_ASSET_MIGRATION_PATH_INVALID")
                source_root.rename(tombstone)

        kind = str(replacement["kind"])
        if kind in {"map", "world"}:
            id_column = "selected_map_id"
            hash_column = "selected_map_content_sha256"
            setting_id = "last_map_id"
            setting_hash = "last_map_content_sha256"
        else:
            id_column = "selected_vehicle_id"
            hash_column = "selected_vehicle_content_sha256"
            setting_id = "last_vehicle_id"
            setting_hash = "last_vehicle_content_sha256"

        try:
            with self.transaction() as connection:
                connection.execute(
                    f"UPDATE threads SET {hash_column} = ?, updated_at = ? "
                    f"WHERE {id_column} = ? AND {hash_column} = ?",
                    (replacement_content_sha256, utc_now(), asset_id, source_content_sha256),
                )
                remembered = {
                    str(row["key"]): json.loads(str(row["value_json"]))
                    for row in connection.execute(
                        "SELECT key,value_json FROM settings WHERE key IN (?,?)",
                        (setting_id, setting_hash),
                    ).fetchall()
                }
                if (
                    remembered.get(setting_id) == asset_id
                    and remembered.get(setting_hash) == source_content_sha256
                ):
                    connection.execute(
                        "INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE "
                        "SET value_json=excluded.value_json",
                        (setting_hash, json.dumps(replacement_content_sha256)),
                    )
                connection.execute(
                    "DELETE FROM asset_versions WHERE asset_id = ? AND content_sha256 = ?",
                    (asset_id, source_content_sha256),
                )
        except BaseException:
            if tombstone is not None and tombstone.is_dir() and source_root is not None:
                source_root.parent.mkdir(parents=True, exist_ok=True)
                tombstone.rename(source_root)
            raise

        if tombstone is not None and tombstone.is_dir():
            shutil.rmtree(tombstone)
        migrated = source is not None
        return migrated

    # 功能：
    #   退役明确指定且结果摘要匹配的内置资格任务及其工作目录；数据库失败时恢复暂存目录。
    #   不用于清理用户实验；成功删除后的证据须从对应发布制品或备份恢复。
    # 输入：
    #   self：当前存储实例。
    #   qualification_id：明确被替代的资格标识。
    #   map_content_sha256：预期结果地图摘要。
    #   vehicle_content_sha256：预期结果飞机摘要。
    # 输出：
    #   retired：是否找到并退役了指定任务。
    @_serialized
    def retire_bundled_asset_pair_qualification(
        self,
        *,
        qualification_id: str,
        map_content_sha256: str,
        vehicle_content_sha256: str,
    ) -> bool:
        if not qualification_id or not all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (map_content_sha256, vehicle_content_sha256)
        ):
            raise AssetImportError("BUNDLED_PAIR_RETIREMENT_IDENTITY_INVALID")

        matches: list[tuple[sqlite3.Row, AssetPairQualificationJob]] = []
        for row in self._connection.execute(
            "SELECT * FROM asset_pair_qualification_jobs"
        ).fetchall():
            job = self.load_asset_pair_qualification_job(str(row["job_id"]))
            if job.qualification_id == qualification_id:
                matches.append((row, job))
        if not matches:
            retired = False
            return retired
        if len(matches) != 1:
            raise AssetImportError("BUNDLED_PAIR_RETIREMENT_IDENTITY_AMBIGUOUS")

        row, job = matches[0]
        if (
            job.state != "qualified"
            or job.result_map_content_sha256 != map_content_sha256
            or job.result_vehicle_content_sha256 != vehicle_content_sha256
        ):
            raise AssetImportError("BUNDLED_PAIR_RETIREMENT_IDENTITY_MISMATCH")

        qualification_root = self.asset_qualification_root.resolve()
        workspace = Path(str(row["workspace_root"])).resolve()
        # 包含于根目录仍可能属于另一个任务。删除只接受该任务的规范目录，
        # 预期路径不 resolve，避免任务目录符号链接把“预期”一同指向别人的目录。
        if workspace != qualification_root / job.job_id:
            raise AssetImportError("BUNDLED_PAIR_RETIREMENT_PATH_INVALID")
        if workspace.exists() and not workspace.is_dir():
            raise AssetImportError("BUNDLED_PAIR_RETIREMENT_PATH_INVALID")

        tombstone: Path | None = None
        if workspace.is_dir():
            tombstone = (qualification_root / f".retiring-{uuid4().hex}").resolve()
            if qualification_root not in tombstone.parents:
                raise AssetImportError("BUNDLED_PAIR_RETIREMENT_PATH_INVALID")
            workspace.rename(tombstone)
        try:
            with self.transaction() as connection:
                connection.execute(
                    "DELETE FROM asset_pair_qualification_jobs WHERE job_id = ?",
                    (str(row["job_id"]),),
                )
        except BaseException:
            if tombstone is not None and tombstone.is_dir():
                workspace.parent.mkdir(parents=True, exist_ok=True)
                tombstone.rename(workspace)
            raise
        if tombstone is not None and tombstone.is_dir():
            shutil.rmtree(tombstone)
        retired = True
        return retired

    # 功能：
    #   保存资格任务进度并绑定原地图、飞机及工作路径；后续保存不能静默更换或忽略路径。
    # 输入：
    #   self：当前存储实例。
    #   job：待保存的资格任务。
    #   workspace_root：首次登记必填的资格工作目录。
    #   map_bundle_root：首次登记必填的版本目录内地图包路径。
    #   vehicle_bundle_root：首次登记必填的版本目录内飞机包路径。
    # 输出：
    #   result：保存后重新校验的任务字典。
    @_serialized
    def save_asset_pair_qualification_job(
        self,
        job: AssetPairQualificationJob,
        *,
        workspace_root: Path | None = None,
        map_bundle_root: Path | None = None,
        vehicle_bundle_root: Path | None = None,
    ) -> dict[str, object]:
        job = AssetPairQualificationJob.model_validate_json(job.model_dump_json())
        existing = self._connection.execute(
            "SELECT * FROM asset_pair_qualification_jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        if existing is None and None in (workspace_root, map_bundle_root, vehicle_bundle_root):
            raise ValueError("ASSET_QUALIFICATION_JOB_PATHS_MISSING")
        effective_workspace = (
            workspace_root.resolve()
            if workspace_root is not None
            else Path(str(existing["workspace_root"])).resolve()
        )
        effective_map = (
            map_bundle_root.resolve()
            if map_bundle_root is not None
            else Path(str(existing["map_bundle_root"])).resolve()
        )
        effective_vehicle = (
            vehicle_bundle_root.resolve()
            if vehicle_bundle_root is not None
            else Path(str(existing["vehicle_bundle_root"])).resolve()
        )
        if self.asset_qualification_root.resolve() not in effective_workspace.parents:
            raise ValueError("ASSET_QUALIFICATION_WORKSPACE_PATH_INVALID")
        versions_root = self.asset_versions_root.resolve()
        if (
            versions_root not in effective_map.parents
            or versions_root not in effective_vehicle.parents
            or not effective_map.is_dir()
            or not effective_vehicle.is_dir()
        ):
            raise ValueError("ASSET_QUALIFICATION_VERSION_PATH_INVALID")
        if existing is not None:
            immutable = {
                "map_asset_id": job.map_asset_id,
                "map_content_sha256": job.map_content_sha256,
                "vehicle_asset_id": job.vehicle_asset_id,
                "vehicle_content_sha256": job.vehicle_content_sha256,
            }
            if any(str(existing[key]) != value for key, value in immutable.items()):
                raise ValueError("ASSET_QUALIFICATION_JOB_IDENTITY_CHANGED")
            if any(
                Path(str(existing[key])).resolve() != path
                for key, path in (
                    ("workspace_root", effective_workspace),
                    ("map_bundle_root", effective_map),
                    ("vehicle_bundle_root", effective_vehicle),
                )
            ):
                raise ValueError("ASSET_QUALIFICATION_JOB_IDENTITY_CHANGED")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO asset_pair_qualification_jobs("
                "job_id,job_json,map_asset_id,map_content_sha256,vehicle_asset_id,"
                "vehicle_content_sha256,workspace_root,map_bundle_root,vehicle_bundle_root,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET job_json=excluded.job_json,"
                "updated_at=excluded.updated_at",
                (
                    job.job_id,
                    job.model_dump_json(),
                    job.map_asset_id,
                    job.map_content_sha256,
                    job.vehicle_asset_id,
                    job.vehicle_content_sha256,
                    str(effective_workspace),
                    str(effective_map),
                    str(effective_vehicle),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
        result = self.get_asset_pair_qualification_job(job.job_id)
        return result

    # 功能：
    #   重验资格任务内容，并核对查询键和四个独立源身份列，拒绝索引与正文错配。
    # 输入：
    #   self：当前存储实例。
    #   job_id：需要精确读取的资格任务标识。
    # 输出：
    #   job：已通过字段及索引身份检查的资格任务。
    @_serialized
    def load_asset_pair_qualification_job(self, job_id: str) -> AssetPairQualificationJob:
        row = self._connection.execute(
            "SELECT * FROM asset_pair_qualification_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        job = AssetPairQualificationJob.model_validate_json(str(row["job_json"]))
        if job.job_id != job_id or any(
            getattr(job, key) != row[key]
            for key in (
                "map_asset_id",
                "map_content_sha256",
                "vehicle_asset_id",
                "vehicle_content_sha256",
            )
        ):
            raise AssetImportError("ASSET_QUALIFICATION_JOB_IDENTITY_MISMATCH")
        return job

    # 功能：
    #   把已重验的资格任务转换为独立 JSON 字典，不向调用者暴露内部模型引用。
    # 输入：
    #   self：当前存储实例。
    #   job_id：资格任务标识。
    # 输出：
    #   result：JSON 兼容的任务字典。
    @_serialized
    def get_asset_pair_qualification_job(self, job_id: str) -> dict[str, object]:
        result = self.load_asset_pair_qualification_job(job_id).model_dump(mode="json")
        return result

    # 功能：
    #   按更新时间倒序列出资格任务，各记录使用统一的身份重验入口。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   jobs：已校验的资格任务字典列表。
    @_serialized
    def list_asset_pair_qualification_jobs(self) -> list[dict[str, object]]:
        rows = self._connection.execute(
            "SELECT job_id FROM asset_pair_qualification_jobs ORDER BY updated_at DESC"
        ).fetchall()
        jobs = [self.get_asset_pair_qualification_job(str(row["job_id"])) for row in rows]
        return jobs

    # 功能：
    #   查找指定结果资产对最新且回执校验通过的资格任务；不把数据库 qualified 标签当成证据。
    # 输入：
    #   self：当前存储实例。
    #   map_asset_id：地图资产标识。
    #   map_content_sha256：验收后地图包摘要。
    #   vehicle_asset_id：飞机资产标识。
    #   vehicle_content_sha256：验收后飞机包摘要。
    # 输出：
    #   job：通过回执校验的资格任务；无有效匹配时为 None。
    @_serialized
    def qualified_asset_pair(
        self,
        *,
        map_asset_id: str,
        map_content_sha256: str,
        vehicle_asset_id: str,
        vehicle_content_sha256: str,
    ) -> AssetPairQualificationJob | None:
        rows = self._connection.execute(
            "SELECT job_id FROM asset_pair_qualification_jobs "
            "WHERE map_asset_id = ? AND vehicle_asset_id = ? ORDER BY updated_at DESC",
            (map_asset_id, vehicle_asset_id),
        ).fetchall()
        for row in rows:
            job = self.load_asset_pair_qualification_job(str(row["job_id"]))
            if (
                job.state == "qualified"
                and job.qualification_id is not None
                and job.result_map_content_sha256 == map_content_sha256
                and job.result_vehicle_content_sha256 == vehicle_content_sha256
            ):
                try:
                    self.verified_asset_pair_receipt(job)
                except (AssetImportError, KeyError, OSError, TypeError, ValueError):
                    continue
                return job
        job = None
        return job

    # 功能：
    #   1. 读取两个结果包内声明的资格回执，核对路径、长度、摘要、源身份及两份回执一致性。
    #   2. 此处验证回执与清单绑定，不重跑仿真、不重新检查整包全部成员，也不代替发布者签名。
    # 输入：
    #   self：当前存储实例。
    #   job：具有结果摘要的合格资格任务。
    # 输出：
    #   result：共同的资格回执对象及原始回执字节的 SHA-256。
    @_serialized
    def verified_asset_pair_receipt(
        self,
        job: AssetPairQualificationJob,
    ) -> tuple[AssetPairQualificationReceipt, str]:
        if (
            job.state != "qualified"
            or job.qualification_id is None
            or job.result_map_content_sha256 is None
            or job.result_vehicle_content_sha256 is None
        ):
            raise AssetImportError("ASSET_QUALIFICATION_EVIDENCE_NOT_READY")

        identities = (
            (
                job.map_asset_id,
                job.map_content_sha256,
                job.result_map_content_sha256,
                {"map", "world"},
            ),
            (
                job.vehicle_asset_id,
                job.vehicle_content_sha256,
                job.result_vehicle_content_sha256,
                {"vehicle"},
            ),
        )
        receipts: list[AssetPairQualificationReceipt] = []
        payload_hashes: list[str] = []
        versions_root = self.asset_versions_root.resolve()
        for asset_id, source_hash, result_hash, allowed_kinds in identities:
            version = self.get_asset_version(asset_id, result_hash)
            try:
                manifest = DDPkgManifest.model_validate(version["manifest"])
                qualification = manifest.qualification
                if (
                    manifest.asset_id != asset_id
                    or manifest.asset_kind not in allowed_kinds
                    or manifest.content_sha256 != result_hash
                    or qualification is None
                    or qualification.maturity != "qualified"
                    or qualification.content_sha256 != result_hash
                ):
                    raise ValueError("qualified package identity changed")
                evidence_path = next(
                    path
                    for path in qualification.evidence_paths
                    if path.endswith(f"/{job.qualification_id}.json")
                    or path == f"qualification/{job.qualification_id}.json"
                )
                declaration = next(entry for entry in manifest.files if entry.path == evidence_path)
                root = Path(str(version["bundle_root"])).resolve()
                if versions_root not in root.parents:
                    raise ValueError("version root escaped")
                evidence_file = (root / Path(*PurePosixPath(evidence_path).parts)).resolve()
                if root not in evidence_file.parents or not evidence_file.is_file():
                    raise ValueError("qualification evidence missing")
                if evidence_file.stat().st_size != declaration.size_bytes:
                    raise ValueError("qualification evidence size changed")
                payload = _qualification_payload(evidence_file, declaration.size_bytes)
                payload_hash = hashlib.sha256(payload).hexdigest()
                if payload_hash != declaration.sha256:
                    raise ValueError("qualification evidence hash changed")
                receipt = AssetPairQualificationReceipt.model_validate_json(payload)
                expected_source_hash = (
                    receipt.map_content_sha256
                    if asset_id == job.map_asset_id
                    else receipt.vehicle_content_sha256
                )
                if expected_source_hash != source_hash:
                    raise ValueError("qualification source identity changed")
            except (KeyError, StopIteration, TypeError, ValueError) as error:
                raise AssetImportError("ASSET_QUALIFICATION_EVIDENCE_INVALID") from error
            receipts.append(receipt)
            payload_hashes.append(payload_hash)

        map_receipt, vehicle_receipt = receipts
        if (
            payload_hashes[0] != payload_hashes[1]
            or map_receipt != vehicle_receipt
            or map_receipt.qualification_id != job.qualification_id
            or map_receipt.map_asset_id != job.map_asset_id
            or map_receipt.map_content_sha256 != job.map_content_sha256
            or map_receipt.vehicle_asset_id != job.vehicle_asset_id
            or map_receipt.vehicle_content_sha256 != job.vehicle_content_sha256
        ):
            raise AssetImportError("ASSET_QUALIFICATION_PAIR_EVIDENCE_MISMATCH")
        result = (map_receipt, payload_hashes[0])
        return result

    # 功能：
    #   从已重验回执构造不含本机路径的执行证据绑定，供规划结果传递到执行器时复核。
    # 输入：
    #   self：当前存储实例。
    #   job：已完成资格验收的任务。
    # 输出：
    #   binding：包含源与结果资产摘要、环境、插件及证据摘要的绑定。
    @_serialized
    def verified_asset_pair_execution_binding(
        self,
        job: AssetPairQualificationJob,
    ) -> MissionAssetPairQualificationBinding:
        receipt, receipt_sha256 = self.verified_asset_pair_receipt(job)
        if (
            job.qualification_id is None
            or job.result_map_content_sha256 is None
            or job.result_vehicle_content_sha256 is None
        ):
            raise AssetImportError("ASSET_QUALIFICATION_EVIDENCE_NOT_READY")
        binding = MissionAssetPairQualificationBinding(
            qualification_id=job.qualification_id,
            receipt_sha256=receipt_sha256,
            runtime_evidence_sha256=receipt.runtime_evidence_sha256,
            map_asset_id=receipt.map_asset_id,
            map_source_content_sha256=receipt.map_content_sha256,
            map_qualified_content_sha256=job.result_map_content_sha256,
            vehicle_asset_id=receipt.vehicle_asset_id,
            vehicle_source_content_sha256=receipt.vehicle_content_sha256,
            vehicle_qualified_content_sha256=job.result_vehicle_content_sha256,
            environment_versions=receipt.environment_versions,
            plugin_snapshot_sha256=receipt.plugin_snapshot_sha256,
            qualified_at=receipt.qualified_at,
        )
        return binding

    # 功能：
    #   重验任务身份后解析工作目录及两个源包目录，拒绝越过应用所管辖的目录边界。
    # 输入：
    #   self：当前存储实例。
    #   job_id：资格任务标识。
    # 输出：
    #   paths：工作目录、地图包目录、飞机包目录组成的路径元组。
    @_serialized
    def asset_pair_qualification_paths(self, job_id: str) -> tuple[Path, Path, Path]:
        self.load_asset_pair_qualification_job(job_id)
        row = self._connection.execute(
            "SELECT workspace_root,map_bundle_root,vehicle_bundle_root "
            "FROM asset_pair_qualification_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        workspace = Path(str(row["workspace_root"])).resolve()
        map_root = Path(str(row["map_bundle_root"])).resolve()
        vehicle_root = Path(str(row["vehicle_bundle_root"])).resolve()
        if self.asset_qualification_root.resolve() not in workspace.parents:
            raise AssetImportError("ASSET_QUALIFICATION_WORKSPACE_PATH_INVALID")
        versions_root = self.asset_versions_root.resolve()
        if (
            versions_root not in map_root.parents
            or versions_root not in vehicle_root.parents
            or not map_root.is_dir()
            or not vehicle_root.is_dir()
        ):
            raise AssetImportError("ASSET_QUALIFICATION_VERSION_PATH_INVALID")
        paths = (workspace, map_root, vehicle_root)
        return paths

    # 功能：
    #   列出未卸载插件的目录记录，保留禁用或不健康条目供诊断，不代表执行准入。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   plugins：按内置优先和名称排序的目录列表。
    @_serialized
    def list_plugins(self) -> list[dict[str, object]]:
        plugins = [
            self._row(row)
            for row in self._connection.execute(
                "SELECT * FROM plugins WHERE status != 'uninstalled' ORDER BY builtin DESC, name"
            )
        ]
        return plugins

    # 功能：
    #   读取指定插件的当前目录项；目录存在不等于获准执行。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    # 输出：
    #   plugin：目录记录副本。
    @_serialized
    def get_plugin(self, plugin_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM plugins WHERE plugin_id = ?", (plugin_id,)
        ).fetchone()
        if row is None:
            raise KeyError(plugin_id)
        plugin = self._row(row)
        return plugin

    # 功能：
    #   同一事务更新活动目录及版本索引；首次初始化选择来源，后续保留已有选择回执。
    # 输入：
    #   self：当前存储实例。
    #   manifest：安装器检查过的清单。
    #   package_sha256：安装包摘要。
    #   bundle_root：包目录，内置实现可为空。
    #   builtin：是否产品内置。
    #   enabled：明确的启用标记。
    #   status：生命周期状态。
    #   health：健康状态。
    #   last_error：最近错误信息，空值清除旧错误。
    #   trust_status：安装器判定的信任状态。
    #   trust_decision：信任回执。
    # 输出：
    #   plugin：更新后的活动插件记录。
    @_serialized
    def upsert_plugin(
        self,
        *,
        manifest: PluginManifest,
        package_sha256: str,
        bundle_root: Path | None,
        builtin: bool,
        enabled: bool,
        status: str,
        health: str,
        last_error: str | None = None,
        trust_status: str = "verified",
        trust_decision: dict[str, object] | None = None,
    ) -> dict[str, object]:
        now = utc_now()
        _boolean(builtin)
        _boolean(enabled)
        authorities = [item.authority for item in manifest.capabilities]
        authority_order = {"read": 0, "plan": 1, "simulate": 2, "control": 3, "actuate": 4}
        authority = max(authorities, key=authority_order.__getitem__)
        manifest_json = manifest.model_dump_json()
        root_value = str(bundle_root.resolve()) if bundle_root is not None else ""
        trust_json = encode_json(
            trust_decision if trust_decision is not None else {},
            limit=_STATE_JSON_LIMIT,
            node_limit=_STATE_JSON_NODES,
        )
        selection_source = "product_managed_default" if builtin else "explicit"
        selected_by = "product_managed" if builtin else "account_configurable"
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugins("
                "plugin_id,name,version,authority,enabled,builtin,description,publisher,"
                "runtime_kind,status,health,removable,package_sha256,bundle_root,manifest_json,"
                "last_error,installed_at,updated_at,trust_status,trust_decision_json,update_ring,"
                "selection_source,selected_by,selection_receipt_sha256,"
                "harness_revision_sha256) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(plugin_id) DO UPDATE SET name=excluded.name,version=excluded.version,"
                "authority=excluded.authority,enabled=excluded.enabled,builtin=excluded.builtin,"
                "description=excluded.description,publisher=excluded.publisher,"
                "runtime_kind=excluded.runtime_kind,status=excluded.status,health=excluded.health,"
                "removable=excluded.removable,package_sha256=excluded.package_sha256,"
                "bundle_root=excluded.bundle_root,manifest_json=excluded.manifest_json,"
                "last_error=excluded.last_error,updated_at=excluded.updated_at,"
                "trust_status=excluded.trust_status,trust_decision_json=excluded.trust_decision_json,"
                "update_ring=excluded.update_ring",
                (
                    manifest.plugin_id,
                    manifest.name,
                    manifest.version,
                    authority,
                    int(enabled),
                    int(builtin),
                    manifest.description,
                    manifest.publisher,
                    manifest.runtime.kind,
                    status,
                    health,
                    int(manifest.removable),
                    package_sha256,
                    root_value,
                    manifest_json,
                    last_error,
                    now,
                    now,
                    trust_status,
                    trust_json,
                    manifest.provenance.update_ring,
                    selection_source,
                    selected_by,
                    None,
                    None,
                ),
            )
            connection.execute(
                "INSERT INTO plugin_versions("
                "plugin_id,version,package_sha256,bundle_root,manifest_json,installed_at,"
                "trust_status,trust_decision_json) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(plugin_id,version) DO UPDATE SET "
                "package_sha256=excluded.package_sha256,bundle_root=excluded.bundle_root,"
                "manifest_json=excluded.manifest_json,trust_status=excluded.trust_status,"
                "trust_decision_json=excluded.trust_decision_json",
                (
                    manifest.plugin_id,
                    manifest.version,
                    package_sha256,
                    root_value,
                    manifest_json,
                    now,
                    trust_status,
                    trust_json,
                ),
            )
        plugin = self.get_plugin(manifest.plugin_id)
        return plugin

    # 功能：
    #   保存单插件生命周期结果；省略的状态保持不变，last_error 空值清除旧错误。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：目标插件标识。
    #   enabled：可选的严格布尔启用值。
    #   status：可选的新生命周期状态。
    #   health：可选的新健康状态。
    #   last_error：新的错误说明或空值。
    # 输出：
    #   plugin：更新后的插件记录。
    @_serialized
    def set_plugin_lifecycle(
        self,
        plugin_id: str,
        *,
        enabled: bool | None = None,
        status: str | None = None,
        health: str | None = None,
        last_error: str | None = None,
    ) -> dict[str, object]:
        updates: dict[str, object] = {"updated_at": utc_now()}
        if enabled is not None:
            updates["enabled"] = int(_boolean(enabled))
        if status is not None:
            updates["status"] = status
        if health is not None:
            updates["health"] = health
        updates["last_error"] = last_error
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE plugins SET {assignments} WHERE plugin_id = ?",
                (*updates.values(), plugin_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(plugin_id)
        plugin = self.get_plugin(plugin_id)
        return plugin

    # 功能：
    #   核对选择者与回执、Harness 摘要的组合，防止混用产品默认和显式选择来源。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：目标插件标识。
    #   selection_source：显式选择或产品默认选择。
    #   selected_by：执行选择的产品、账户界面或 Harness 设计者。
    #   selection_receipt_sha256：可选的选择回执摘要。
    #   harness_revision_sha256：Harness 设计者选择必须绑定的内容摘要。
    # 输出：
    #   plugin：带新来源的目录记录。
    @_serialized
    def set_plugin_selection_provenance(
        self,
        plugin_id: str,
        *,
        selection_source: Literal["explicit", "product_managed_default"],
        selected_by: Literal["product_managed", "account_configurable", "agent_harness_designer"],
        selection_receipt_sha256: str | None = None,
        harness_revision_sha256: str | None = None,
    ) -> dict[str, object]:
        if selection_source not in ("explicit", "product_managed_default") or selected_by not in (
            "product_managed",
            "account_configurable",
            "agent_harness_designer",
        ):
            raise ValueError("PLUGIN_SELECTION_PROVENANCE_INVALID")
        if selection_source == "product_managed_default":
            if selected_by != "product_managed":
                raise ValueError("PLUGIN_SELECTION_PROVENANCE_INVALID")
            if selection_receipt_sha256 is not None or harness_revision_sha256 is not None:
                raise ValueError("PLUGIN_MANAGED_SELECTION_RECEIPT_INVALID")
        elif selected_by == "product_managed":
            raise ValueError("PLUGIN_EXPLICIT_SELECTION_SURFACE_REQUIRED")
        if selected_by == "agent_harness_designer" and harness_revision_sha256 is None:
            raise ValueError("PLUGIN_HARNESS_REVISION_BINDING_REQUIRED")
        if selected_by != "agent_harness_designer" and harness_revision_sha256 is not None:
            raise ValueError("PLUGIN_HARNESS_REVISION_BINDING_INVALID")
        for digest in (selection_receipt_sha256, harness_revision_sha256):
            if digest is not None and (
                not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("PLUGIN_SELECTION_HASH_INVALID")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE plugins SET selection_source=?,selected_by=?,"
                "selection_receipt_sha256=?,harness_revision_sha256=?,updated_at=? "
                "WHERE plugin_id=?",
                (
                    selection_source,
                    selected_by,
                    selection_receipt_sha256,
                    harness_revision_sha256,
                    utc_now(),
                    plugin_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(plugin_id)
        plugin = self.get_plugin(plugin_id)
        return plugin

    # 功能：
    #   同时更新指定版本和活动目录的信任结果；指定版本不活动时整体回滚。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    #   version：必须正在使用的版本。
    #   trust_status：信任验证结果。
    #   trust_decision：对应回执。
    # 输出：
    #   plugin：更新后的活动记录。
    @_serialized
    def set_plugin_trust(
        self,
        plugin_id: str,
        *,
        version: str,
        trust_status: str,
        trust_decision: dict[str, object],
    ) -> dict[str, object]:
        decision_json = encode_json(
            trust_decision, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
        )
        with self.transaction() as connection:
            version_cursor = connection.execute(
                "UPDATE plugin_versions SET trust_status = ?,trust_decision_json = ? "
                "WHERE plugin_id = ? AND version = ?",
                (trust_status, decision_json, plugin_id, version),
            )
            if version_cursor.rowcount != 1:
                raise KeyError(f"{plugin_id}@{version}")
            current_cursor = connection.execute(
                "UPDATE plugins SET trust_status = ?,trust_decision_json = ?,updated_at = ? "
                "WHERE plugin_id = ? AND version = ?",
                (trust_status, decision_json, utc_now(), plugin_id, version),
            )
            if current_cursor.rowcount != 1:
                raise KeyError(plugin_id)
        plugin = self.get_plugin(plugin_id)
        return plugin

    # 功能：
    #   更新暂存版本信任；若恰为活动版本则同步目录，但不切换版本。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    #   version：待更新的已安装版本。
    #   trust_status：信任验证结果。
    #   trust_decision：对应回执。
    # 输出：
    #   record：更新后的版本记录。
    @_serialized
    def set_plugin_version_trust(
        self,
        plugin_id: str,
        *,
        version: str,
        trust_status: str,
        trust_decision: dict[str, object],
    ) -> dict[str, object]:
        decision_json = encode_json(
            trust_decision, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
        )
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE plugin_versions SET trust_status = ?,trust_decision_json = ? "
                "WHERE plugin_id = ? AND version = ?",
                (trust_status, decision_json, plugin_id, version),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"{plugin_id}@{version}")
            connection.execute(
                "UPDATE plugins SET trust_status = ?,trust_decision_json = ?,updated_at = ? "
                "WHERE plugin_id = ? AND version = ?",
                (trust_status, decision_json, utc_now(), plugin_id, version),
            )
        record = self.get_plugin_version(plugin_id, version)
        return record

    # 功能：
    #   原子应用整组生命周期变更，任一条目非法或缺失时全部回滚。
    # 输入：
    #   self：当前存储实例。
    #   updates：插件标识到字段更新的映射。
    # 输出：
    #   plugins：按输入顺序返回的更新后记录。
    @_serialized
    def set_plugin_lifecycles(
        self, updates: dict[str, dict[str, object]]
    ) -> list[dict[str, object]]:
        if not updates:
            plugins = []
            return plugins
        now = utc_now()
        with self.transaction() as connection:
            for plugin_id, patch in updates.items():
                unsupported = set(patch) - {"enabled", "status", "health", "last_error"}
                if unsupported:
                    raise ValueError(
                        "PLUGIN_LIFECYCLE_BATCH_FIELD_INVALID:" + ",".join(sorted(unsupported))
                    )
                values = {"updated_at": now, **patch}
                if "enabled" in values:
                    values["enabled"] = int(_boolean(values["enabled"]))
                assignments = ", ".join(f"{key} = ?" for key in values)
                cursor = connection.execute(
                    f"UPDATE plugins SET {assignments} WHERE plugin_id = ?",
                    (*values.values(), plugin_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(plugin_id)
        plugins = [self.get_plugin(plugin_id) for plugin_id in updates]
        return plugins

    # 功能：
    #   切换到精确的已安装版本且保持禁用；清单必须匹配查询键，防止覆盖另一插件。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：目标插件标识。
    #   version：所选已安装版本。
    # 输出：
    #   plugin：切换后的禁用目录记录。
    @_serialized
    def activate_plugin_version(self, plugin_id: str, version: str) -> dict[str, object]:
        version_row = self._connection.execute(
            "SELECT * FROM plugin_versions WHERE plugin_id = ? AND version = ?",
            (plugin_id, version),
        ).fetchone()
        if version_row is None:
            raise KeyError(f"{plugin_id}@{version}")
        manifest = PluginManifest.model_validate_json(str(version_row["manifest_json"]))
        if manifest.plugin_id != plugin_id or manifest.version != version:
            raise ValueError("PLUGIN_VERSION_IDENTITY_MISMATCH")
        plugin = self.upsert_plugin(
            manifest=manifest,
            package_sha256=str(version_row["package_sha256"]),
            bundle_root=Path(str(version_row["bundle_root"]))
            if str(version_row["bundle_root"])
            else None,
            builtin=bool(self.get_plugin(plugin_id)["builtin"]),
            enabled=False,
            status="disabled",
            health="unknown",
            trust_status=str(version_row["trust_status"]),
            trust_decision=decode_json(
                version_row["trust_decision_json"],
                limit=_STATE_JSON_LIMIT,
                node_limit=_STATE_JSON_NODES,
            ),
        )
        return plugin

    # 功能：
    #   登记新的已检查版本，不替换活动目录；重复版本由数据库唯一约束拒绝。
    # 输入：
    #   self：当前存储实例。
    #   manifest：安装器验证过的清单。
    #   package_sha256：对应安装包摘要。
    #   bundle_root：已安装包目录。
    #   trust_status：信任结果，默认未验证。
    #   trust_decision：可选信任回执。
    # 输出：
    #   record：暂存版本记录。
    @_serialized
    def install_plugin_version(
        self,
        *,
        manifest: PluginManifest,
        package_sha256: str,
        bundle_root: Path,
        trust_status: str = "unverified",
        trust_decision: dict[str, object] | None = None,
    ) -> dict[str, object]:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugin_versions("
                "plugin_id,version,package_sha256,bundle_root,manifest_json,installed_at,"
                "trust_status,trust_decision_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    manifest.plugin_id,
                    manifest.version,
                    package_sha256,
                    str(bundle_root.resolve()),
                    manifest.model_dump_json(),
                    now,
                    trust_status,
                    encode_json(
                        trust_decision if trust_decision is not None else {},
                        limit=_STATE_JSON_LIMIT,
                        node_limit=_STATE_JSON_NODES,
                    ),
                ),
            )
        record = self.get_plugin_version(manifest.plugin_id, manifest.version)
        return record

    # 功能：
    #   读取精确的已安装版本，不以最新版或活动版本代替缺失条目。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    #   version：版本标识。
    # 输出：
    #   record：匹配的版本记录副本。
    @_serialized
    def get_plugin_version(self, plugin_id: str, version: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM plugin_versions WHERE plugin_id = ? AND version = ?",
            (plugin_id, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"{plugin_id}@{version}")
        record = self._row(row)
        return record

    # 功能：
    #   列出保留版本供管理器明确选择，不自动激活其中任何版本。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    # 输出：
    #   versions：按安装时间倒序的版本列表。
    @_serialized
    def list_plugin_versions(self, plugin_id: str) -> list[dict[str, object]]:
        versions = [
            self._row(row)
            for row in self._connection.execute(
                "SELECT * FROM plugin_versions WHERE plugin_id = ? ORDER BY installed_at DESC",
                (plugin_id,),
            )
        ]
        return versions

    # 功能：
    #   倒序读取指定插件的操作回执，用于本机诊断。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    # 输出：
    #   events：历史操作回执列表。
    @_serialized
    def list_plugin_events(self, plugin_id: str) -> list[dict[str, object]]:
        events = [
            self._row(row)
            for row in self._connection.execute(
                "SELECT * FROM plugin_events WHERE plugin_id = ? ORDER BY created_at DESC",
                (plugin_id,),
            )
        ]
        return events

    # 功能：
    #   保存实际治理决策，严格区分布尔接受与拒绝，不把文本 false 转成接受。
    # 输入：
    #   self：当前存储实例。
    #   decision：含决策标识、插件、操作、接受结果与时间的 JSON 回执。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def record_plugin_governance_decision(self, decision: dict[str, object]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugin_governance_events("
                "decision_id,plugin_id,operation,accepted,decision_json,created_at"
                ") VALUES(?,?,?,?,?,?)",
                (
                    decision["decision_id"],
                    decision["plugin_id"],
                    decision["operation"],
                    int(_boolean(decision["accepted"])),
                    encode_json(decision, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES),
                    decision["created_at"],
                ),
            )

    # 功能：
    #   返回最多一千条治理记录用于诊断，历史记录不作为新调用的准入缓存。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：可选的插件筛选标识。
    #   limit：请求条数，裁剪到 1—1000。
    # 输出：
    #   decisions：按创建时间倒序排列的决策列表。
    @_serialized
    def list_plugin_governance_decisions(
        self, plugin_id: str | None = None, *, limit: int = 100
    ) -> list[dict[str, object]]:
        safe_limit = min(max(limit, 1), 1_000)
        if plugin_id is None:
            rows = self._connection.execute(
                "SELECT * FROM plugin_governance_events ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM plugin_governance_events WHERE plugin_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (plugin_id, safe_limit),
            )
        decisions = [self._row(row) for row in rows]
        return decisions

    # 功能：
    #   保存插件调用耗时、流量及结果，拒绝非法计量值；该表不是云端模型 token 账本。
    # 输入：
    #   self：当前存储实例。
    #   event：含调用身份、结果、毫秒耗时、字节计数与时间的回执。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def record_plugin_usage(self, event: dict[str, object]) -> None:
        duration = event["duration_ms"]
        if type(duration) not in (int, float):
            raise ValueError("APP_USAGE_METRIC_INVALID")
        try:
            valid_duration = math.isfinite(duration) and duration >= 0
        except OverflowError:
            valid_duration = False
        if not valid_duration or any(
            type(event[key]) is not int or not 0 <= event[key] <= 2**63 - 1
            for key in ("input_bytes", "output_bytes")
        ):
            raise ValueError("APP_USAGE_METRIC_INVALID")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugin_usage_events("
                "invocation_id,plugin_id,plugin_version,capability_id,slot_id,"
                "invocation_kind,outcome,duration_ms,input_bytes,output_bytes,issue_code,created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event["invocation_id"],
                    event["plugin_id"],
                    event["plugin_version"],
                    event["capability_id"],
                    event["slot_id"],
                    event["invocation_kind"],
                    event["outcome"],
                    event["duration_ms"],
                    event["input_bytes"],
                    event["output_bytes"],
                    event.get("issue_code"),
                    event["created_at"],
                ),
            )

    # 功能：
    #   读取有条数上限的本地插件调用历史，可限定到单个插件。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：可选的插件筛选标识。
    #   limit：请求条数，裁剪到 1—1000。
    # 输出：
    #   events：按创建时间倒序的调用记录列表。
    @_serialized
    def list_plugin_usage(
        self, plugin_id: str | None = None, *, limit: int = 100
    ) -> list[dict[str, object]]:
        safe_limit = min(max(limit, 1), 1_000)
        if plugin_id is None:
            rows = self._connection.execute(
                "SELECT * FROM plugin_usage_events ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM plugin_usage_events WHERE plugin_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (plugin_id, safe_limit),
            )
        events = [self._row(row) for row in rows]
        return events

    # 功能：
    #   汇总调用次数、结果、耗时与流量；无调用时计数为零，最后调用时间为空。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    # 输出：
    #   summary：调用统计字典。
    @_serialized
    def summarize_plugin_usage(self, plugin_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT COUNT(*) AS calls,"
            "COALESCE(SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END),0) AS successes,"
            "COALESCE(SUM(CASE WHEN outcome='error' THEN 1 ELSE 0 END),0) AS errors,"
            "COALESCE(SUM(duration_ms),0) AS duration_ms,"
            "COALESCE(AVG(duration_ms),0) AS average_duration_ms,"
            "COALESCE(SUM(input_bytes),0) AS input_bytes,"
            "COALESCE(SUM(output_bytes),0) AS output_bytes,"
            "MAX(created_at) AS last_called_at "
            "FROM plugin_usage_events WHERE plugin_id = ?",
            (plugin_id,),
        ).fetchone()
        summary = self._row(row) if row is not None else {}
        return summary

    # 功能：
    #   保存已有插件的配置；凭证秘密由凭证库管理，不应写入此 JSON 配置。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    #   configuration：已通过插件配置检查的 JSON 对象。
    # 输出：
    #   saved：保存后的配置记录。
    @_serialized
    def save_plugin_configuration(
        self, plugin_id: str, configuration: dict[str, object]
    ) -> dict[str, object]:
        self.get_plugin(plugin_id)
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugin_configurations VALUES(?,?,?) "
                "ON CONFLICT(plugin_id) DO UPDATE SET "
                "configuration_json=excluded.configuration_json,updated_at=excluded.updated_at",
                (
                    plugin_id,
                    encode_json(
                        configuration, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
                    ),
                    now,
                ),
            )
        saved = self.get_plugin_configuration(plugin_id)
        return saved

    # 功能：
    #   返回独立配置副本，尚未配置时使用显式空对象和空更新时间。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：插件标识。
    # 输出：
    #   configuration：带插件标识及更新时间的配置记录。
    @_serialized
    def get_plugin_configuration(self, plugin_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM plugin_configurations WHERE plugin_id = ?", (plugin_id,)
        ).fetchone()
        if row is None:
            configuration = {"plugin_id": plugin_id, "configuration": {}, "updated_at": None}
            return configuration
        value = self._row(row)
        configuration = {
            "plugin_id": plugin_id,
            "configuration": value["configuration"],
            "updated_at": value["updated_at"],
        }
        return configuration

    # 功能：
    #   保存插件操作的接受或拒绝回执，不把真值性误当成布尔批准。
    # 输入：
    #   self：当前存储实例。
    #   receipt：含操作身份、插件、布尔结果及时间的回执。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def record_plugin_event(self, receipt: dict[str, object]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO plugin_events("
                "receipt_id,plugin_id,operation,accepted,receipt_json,created_at"
                ") VALUES(?,?,?,?,?,?)",
                (
                    receipt["receipt_id"],
                    receipt["plugin_id"],
                    receipt["operation"],
                    int(_boolean(receipt["accepted"])),
                    encode_json(receipt, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES),
                    receipt["created_at"],
                ),
            )

    # 功能：
    #   追加任务使用的插件选择快照，不覆盖旧快照；快照不自动授予执行权限。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：所属任务标识。
    #   snapshot：执行链路创建的插件目录快照。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def save_plugin_snapshot(self, thread_id: str, snapshot: PluginSnapshot) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO task_plugin_snapshots VALUES(?,?,?,?,?)",
                (
                    snapshot.snapshot_id,
                    thread_id,
                    snapshot.catalog_sha256,
                    snapshot.model_dump_json(),
                    snapshot.created_at.isoformat(),
                ),
            )

    # 功能：
    #   读取任务最新快照；执行前仍需校验其内容与当前任务绑定。
    # 输入：
    #   self：当前存储实例。
    #   thread_id：所属任务标识。
    # 输出：
    #   snapshot：已解码的快照记录。
    @_serialized
    def latest_plugin_snapshot(self, thread_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM task_plugin_snapshots WHERE thread_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            raise KeyError("PLUGIN_SNAPSHOT_NOT_FOUND")
        snapshot = self._row(row)
        return snapshot

    # 功能：
    #   将插件标为已卸载并禁用，同时保留历史；实际文件回收由安装管理器负责。
    # 输入：
    #   self：当前存储实例。
    #   plugin_id：要卸载的精确插件标识。
    # 输出：
    #   plugin：已卸载且禁用的记录。
    @_serialized
    def mark_plugin_uninstalled(self, plugin_id: str) -> dict[str, object]:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE plugins SET enabled = 0,status = 'uninstalled',health = 'unknown',"
                "updated_at = ? WHERE plugin_id = ?",
                (utc_now(), plugin_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(plugin_id)
        plugin = self.get_plugin(plugin_id)
        return plugin

    # 功能：
    #   合并本机默认偏好与已保存设置，严格校验 JSON 和记忆开关；不替代云端账户授权。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   values：校验后的本机设置字典。
    @_serialized
    def get_settings(self) -> dict[str, object]:
        values = {
            "locale": "zh-CN",
            "theme": "system",
            "update_channel": "stable",
            "default_model_id": "gpt-5.4",
            "memory_enabled": True,
            "remember_task_preferences": True,
            "remember_asset_choices": True,
            "last_map_id": None,
            "last_map_content_sha256": None,
            "last_vehicle_id": None,
            "last_vehicle_content_sha256": None,
            "plugin_update_ring": "stable",
            "plugin_governance": PluginGovernancePolicy().model_dump(mode="json"),
            "plugin_marketplace_sources": [],
        }
        for row in self._connection.execute("SELECT * FROM settings"):
            values[str(row["key"])] = decode_json(
                row["value_json"], limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES
            )
        for key in ("memory_enabled", "remember_asset_choices", "remember_task_preferences"):
            _boolean(values[key])
        return values

    # 功能：
    #   原子保存偏好；关闭本机记忆或资产记忆时同时清空记住的资产 ID 和摘要。
    # 输入：
    #   self：当前存储实例。
    #   changes：标准 JSON 更新，记忆开关必须为布尔值。
    # 输出：
    #   settings：提交后重新读取的有效设置。
    @_serialized
    def patch_settings(self, changes: dict[str, object]) -> dict[str, object]:
        # Validate the whole patch before writing or clearing remembered assets.
        # 同时校验顶层键，SQLite 不得把整数键转换成与字符串键相同的设置名。
        encode_json(changes, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES)
        for key in ("memory_enabled", "remember_asset_choices", "remember_task_preferences"):
            if key in changes:
                _boolean(changes[key])
        current = self.get_settings()
        next_memory_enabled = _boolean(changes.get("memory_enabled", current["memory_enabled"]))
        next_remember_assets = _boolean(
            changes.get("remember_asset_choices", current["remember_asset_choices"])
        )
        if not next_memory_enabled or not next_remember_assets:
            changes = {
                **changes,
                "last_map_id": None,
                "last_map_content_sha256": None,
                "last_vehicle_id": None,
                "last_vehicle_content_sha256": None,
            }
        with self.transaction() as connection:
            for key, value in changes.items():
                if value is None and key not in {
                    "last_map_id",
                    "last_map_content_sha256",
                    "last_vehicle_id",
                    "last_vehicle_content_sha256",
                }:
                    continue
                connection.execute(
                    "INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE "
                    "SET value_json=excluded.value_json",
                    (
                        key,
                        encode_json(value, limit=_STATE_JSON_LIMIT, node_limit=_STATE_JSON_NODES),
                    ),
                )
        settings = self.get_settings()
        return settings

    # 功能：
    #   列出连接器凭证元数据，不读取或返回秘密。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   credentials：按创建时间排列的引用记录。
    @_serialized
    def list_connector_credentials(self) -> list[dict[str, object]]:
        credentials = [
            self._row(row)
            for row in self._connection.execute(
                "SELECT * FROM connector_credentials ORDER BY created_at ASC"
            )
        ]
        return credentials

    # 功能：
    #   查找精确引用，不访问操作系统凭证库。
    # 输入：
    #   self：当前存储实例。
    #   reference：凭证引用标识。
    # 输出：
    #   credential：元数据副本。
    @_serialized
    def get_connector_credential(self, reference: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM connector_credentials WHERE reference = ?", (reference,)
        ).fetchone()
        if row is None:
            raise KeyError(reference)
        credential = self._row(row)
        return credential

    # 功能：
    #   保存凭证显示名称及允许插件列表，不接收 API Key 明文；同一事务内完成资料回读。
    # 输入：
    #   self：当前存储实例。
    #   value：包含 reference、display_name 和 allowed_plugin_ids 的元数据。
    # 输出：
    #   credential：新保存的引用记录。
    @_serialized
    def save_connector_credential(self, value: dict[str, object]) -> dict[str, object]:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO connector_credentials(reference,display_name,"
                "allowed_plugin_ids_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (
                    value["reference"],
                    value["display_name"],
                    encode_json(
                        value["allowed_plugin_ids"],
                        limit=_STATE_JSON_LIMIT,
                        node_limit=_STATE_JSON_NODES,
                    ),
                    now,
                    now,
                ),
            )
            # 回读失败必须回滚插入，否则服务补偿删除秘密后会留下无法使用的引用。
            credential = self.get_connector_credential(str(value["reference"]))
        return credential

    # 功能：
    #   删除精确引用元数据，凭证库秘密撤销由连接器管理器负责。
    # 输入：
    #   self：当前存储实例。
    #   reference：要移除的引用。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def delete_connector_credential(self, reference: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM connector_credentials WHERE reference = ?", (reference,)
            )
            if cursor.rowcount != 1:
                raise KeyError(reference)

    # 功能：
    #   读取已保存的模型连接资料，严格解码启用值，不返回凭证。
    # 输入：
    #   self：当前存储实例。
    # 输出：
    #   profiles：按创建时间排列的资料列表。
    @_serialized
    def list_custom_models(self) -> list[dict[str, object]]:
        rows = self._connection.execute(
            "SELECT * FROM custom_models ORDER BY created_at ASC"
        ).fetchall()
        profiles = [self._row(row) for row in rows]
        return profiles

    # 功能：
    #   精确读取连接资料及明确的启用状态。
    # 输入：
    #   self：当前存储实例。
    #   profile_id：连接资料标识。
    # 输出：
    #   profile：连接资料副本。
    @_serialized
    def get_custom_model(self, profile_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM custom_models WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(profile_id)
        profile = self._row(row)
        return profile

    # 功能：
    #   1. 新建模型连接元数据，启用值只允许布尔；秘密由模型管理器保存到独立凭证库。
    #   2. 同一事务内回读资料，回读失败时不得提交已启用但没有秘密的空壳模型。
    # 输入：
    #   self：当前存储实例。
    #   profile：包含提供商、地址、协议、模型标识及启用值的资料。
    # 输出：
    #   saved：新保存的连接资料。
    @_serialized
    def save_custom_model(self, profile: dict[str, object]) -> dict[str, object]:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO custom_models(profile_id,display_name,provider,icon,base_url,"
                "api_style,model_id,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    profile["profile_id"],
                    profile["display_name"],
                    profile["provider"],
                    profile["icon"],
                    profile["base_url"],
                    profile["api_style"],
                    profile["model_id"],
                    int(_boolean(profile.get("enabled", True))),
                    now,
                    now,
                ),
            )
            saved = self.get_custom_model(str(profile["profile_id"]))
        return saved

    # 功能：
    #   删除指定连接资料，不影响其他资料；凭证撤销由模型管理器负责。
    # 输入：
    #   self：当前存储实例。
    #   profile_id：要删除的资料标识。
    # 输出：
    #   None：不返回业务数据。
    @_serialized
    def delete_custom_model(self, profile_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM custom_models WHERE profile_id = ?", (profile_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(profile_id)
