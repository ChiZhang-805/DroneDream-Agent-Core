"""Bounded complete evidence FIFO shared by sensing and control execution."""

from __future__ import annotations

import os
import re
import stat
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TextIO

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .plugin_files import check_plain_plugin_path
from .plugin_values import plugin_json_value

MAXIMUM_EVIDENCE_RECORD_BYTES = 16 * 1024 * 1024
MAXIMUM_EVIDENCE_FILE_BYTES = 512 * 1024 * 1024
MAXIMUM_EVIDENCE_FILE_RECORDS = 1_000_000
MAXIMUM_PENDING_EVIDENCE_BYTES = 64 * 1024 * 1024
MAXIMUM_EVIDENCE_STREAMS = 64

# Explicit consumer inventory: never follow arbitrary paths claimed by a writer
# receipt. Timing records belong to this FIFO just as decisions and snapshots do.
NAVIGATION_EVIDENCE_FILENAMES = (
    "depth-local-safety-history.jsonl", "learning-observations.jsonl",
    "model-navigation-cycles.jsonl", "model-navigation-model-calls.jsonl",
    "model-navigation-snapshots.jsonl", "model-navigation-timing.jsonl",
)
CONTROL_EVIDENCE_FILENAMES = (
    "control-applications.jsonl", "local-safety-executor-history.jsonl",
    "executor-brake-applications.jsonl",
)


# 功能：
#   1. 检查全部普通路径后独占领取固定文件，不覆盖另一所有者的既有证据。
#   2. 后续领取失败时保留先前空文件作为中断现场，不擅自删除已对外可见的状态。
# 输入：
#   paths：调用方私有运行目录中的有限固定路径集合。
# 输出：
#   None：不返回业务数据。
def reserve_new_evidence_files(*paths: Path) -> None:
    if not 1 <= len(paths) <= MAXIMUM_EVIDENCE_STREAMS:
        raise ValueError("RUNTIME_EVIDENCE_CLAIM_COUNT_INVALID")
    for path in paths:
        check_plain_plugin_path(path)
    for path in paths:
        with path.open("xb"):
            pass


# 功能：
#   在生产者关闭后，按固定导航文件清单重新计数，不信任回执中自报的路径。
# 输入：
#   run_dir：当前运行的独立目录。
# 输出：
#   counts：固定文件路径到实际记录数量的映射，非法流为 -1。
def navigation_evidence_inventory(run_dir: Path) -> dict[str, int]:
    counts = _evidence_inventory(tuple(run_dir / name for name in NAVIGATION_EVIDENCE_FILENAMES))
    return counts


# 功能：
#   在执行器关闭后重数固定控制流，不将任意回执路径扩展为可读取文件。
# 输入：
#   run_dir：当前运行目录。
# 输出：
#   counts：固定控制、诊断和独立制动证据路径的实际记录数量。
def control_evidence_inventory(run_dir: Path) -> dict[str, int]:
    counts = _evidence_inventory(tuple(
        run_dir / "runtime-state" / name for name in CONTROL_EVIDENCE_FILENAMES))
    return counts


# 功能：
#   1. 有界逐行复核普通文件中的严格 JSON 对象，检测截断、重复键及读取期间变化。
#   2. 真正缺失的未使用流为零，读到部分有效前缀后发生任何错误仍将整流标为 -1。
# 输入：
#   paths：宿主固定清单中的证据文件路径。
# 输出：
#   counts：每份文件的完整记录数或失败标记，不代表记录语义或实际飞行合格。
def _evidence_inventory(paths: tuple[Path, ...]) -> dict[str, int]:
    counts = {}
    for path in paths:
        count = 0
        found = False
        try:
            check_plain_plugin_path(path)
            before = path.stat()
            found = True
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAXIMUM_EVIDENCE_FILE_BYTES:
                raise ValueError("RUNTIME_EVIDENCE_FILE_INVALID")
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if not os.path.samestat(before, opened):
                    raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
                total = 0
                while line := stream.readline(MAXIMUM_EVIDENCE_RECORD_BYTES + 1):
                    total += len(line)
                    if (len(line) > MAXIMUM_EVIDENCE_RECORD_BYTES or not line.endswith(b"\n")
                            or total > MAXIMUM_EVIDENCE_FILE_BYTES
                            or count >= MAXIMUM_EVIDENCE_FILE_RECORDS):
                        raise ValueError("RUNTIME_EVIDENCE_RECORD_TRUNCATED_OR_OVERSIZED")
                    value = decode_json(line, limit=MAXIMUM_EVIDENCE_RECORD_BYTES)
                    if not isinstance(value, dict):
                        raise ValueError("RUNTIME_EVIDENCE_RECORD_NOT_OBJECT")
                    count += 1
                after = os.fstat(stream.fileno())
            check_plain_plugin_path(path)
            current = path.stat()
            if (not os.path.samestat(after, current) or total != opened.st_size
                    or after.st_size != opened.st_size or current.st_size != after.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or current.st_mtime_ns != after.st_mtime_ns):
                raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
        except FileNotFoundError:
            if found:
                count = -1  # 读取中消失不是一条从未使用过的流。
        except (OSError, ValueError, RecursionError):
            count = -1
        counts[str(path)] = count
    return counts


# 功能：统一同一盘符路径的 Windows/WSL 表达，不按文件名或任意后缀猜测归属。
# 输入：仅用于比对的路径字符串；输出：稳定标识，不访问回执声明的任何文件。
# 保留目录大小写，拒绝折叠点段；非标准挂载位置必须保持原路径精确匹配。
def evidence_path_identity(value: str) -> str:
    path = value.replace("\\", "/")
    if any(part in {".", "..", ""} for part in path.split("/")[1:]):
        return value
    windows = re.fullmatch(r"([A-Za-z]):/(.+)", path)
    mounted = re.fullmatch(r"/mnt/([a-z])/(.+)", path)
    if match := windows or mounted:
        return "drive:" + match[1].lower() + ":/" + match[2]
    return value


# 功能：逐文件对照关闭回执与固定清单独立计数，拒绝线程、队列、路径或数量不一致。
# 输入：后台关闭回执、独立计数及总记录数；输出：完整性是否通过，不替代控制语义验证。
def runtime_evidence_inventory_complete(summary, *, artifact_counts, record_count) -> bool:
    if not isinstance(summary, dict) or not isinstance(artifact_counts, dict):
        return False
    if (type(record_count) is not int or record_count < 0
            or any(not isinstance(path, str) or not path or type(count) is not int or count < 0
                   for path, count in artifact_counts.items())
            or sum(artifact_counts.values()) != record_count
            or summary.get("schema_version") != "dronedream.runtime-evidence-writer.v1"
            or summary.get("complete") is not True or summary.get("closed") is not True
            or summary.get("issue_code") is not None
            or summary.get("thread_finished") is not True
            or summary.get("thread_alive") is not False):
        return False
    for field, expected in (("submitted_count", record_count), ("completed_count", record_count),
                            ("rejected_count", 0), ("pending_count", 0)):
        if type(summary.get(field)) is not int or summary[field] != expected:
            return False
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    # 只统一已由调用方固定清单读取的路径；不增加可访问路径，也不能合并别名重复项。
    normalized = {evidence_path_identity(path): count for path, count in artifact_counts.items()}
    if len(normalized) != len(artifact_counts):
        return False
    seen = set()
    for row in artifacts:
        if not isinstance(row, dict):
            return False
        path = row.get("path")
        if not isinstance(path, str):
            return False
        path = evidence_path_identity(path)
        if path not in normalized or path in seen:
            return False
        seen.add(path)
        for field in ("submitted_count", "completed_count"):
            if type(row.get(field)) is not int or row[field] != normalized[path]:
                return False
    complete = all(count == 0 or path in seen for path, count in normalized.items())
    return complete


class BoundedRuntimeEvidenceWriter:
    """Persist complete audit JSONL streams outside the real-time control path.

    Unlike the separate latest-only training-data writer, runtime audit records
    may never be superseded or silently discarded.  The producer therefore
    uses a bounded FIFO.  Capacity exhaustion or any writer error is durable
    qualification evidence and makes ``complete`` false, while flight safety
    continues to be governed by the adjacent observation/command files.
    """

    # 功能：
    #   1. 建立有限容量、有限待处理字节的完整证据 FIFO，固定当前运行目录。
    #   2. 首次回执失败即锁定失败，记录不合并、不替换；控制安全仍由独立控制链路负责。
    # 输入：
    #   self：待初始化的后台写入器。
    #   summary_path：当前运行目录中的汇总路径。
    #   summary_publisher：发布汇总对象的回调。
    #   serializer：在后台将记录编码为单行 JSON 的回调。
    #   maximum_pending_records：排队记录数上限。
    #   flush_interval_seconds：普通流的批量刷新间隔秒数。
    #   flush_on_record_paths：必须每条写入后立即刷新的关键流。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        summary_path: Path,
        *,
        summary_publisher: Callable[[Path, object], None],
        serializer: Callable[[object], str],
        maximum_pending_records: int = 8192,
        flush_interval_seconds: float = 0.2,
        flush_on_record_paths: Iterable[Path] = (),
    ) -> None:
        if not callable(summary_publisher) or not callable(serializer):
            raise ValueError("RUNTIME_EVIDENCE_CALLBACK_INVALID")
        if type(maximum_pending_records) is not int or not 1 <= maximum_pending_records <= 65_536:
            raise ValueError("maximum_pending_records must be positive")
        if (type(flush_interval_seconds) not in (int, float)
                or not 0.0 < flush_interval_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("flush_interval_seconds must be finite and positive")
        self.summary_path = Path(os.path.abspath(summary_path))
        self._root = self.summary_path.parent
        check_plain_plugin_path(self.summary_path)
        self._summary_publisher = summary_publisher
        self._serializer = serializer
        self.maximum_pending_records = maximum_pending_records
        self.flush_interval_seconds = flush_interval_seconds
        # 学习器消费的执行回执需要立即可见，其余流仍批量刷新；磁盘操作只在后台执行。
        flush_paths = []
        for path in flush_on_record_paths:
            if len(flush_paths) >= MAXIMUM_EVIDENCE_STREAMS:
                raise ValueError("RUNTIME_EVIDENCE_STREAM_LIMIT")
            flush_paths.append(self._destination(path))
        self._flush_on_record_paths = frozenset(flush_paths)
        self._condition = threading.Condition()
        self._pending: deque[tuple[Path, object, int]] = deque()
        self._pending_bytes = 0
        self._owned_files: dict[Path, os.stat_result] = {}
        self._written_bytes: Counter[Path] = Counter()
        self._closed = False
        self._thread_finished = False
        self._issue_code: str | None = None
        self._submitted_count = 0
        self._completed_count = 0
        self._rejected_count = 0
        self._maximum_pending_observed = 0
        self._submitted_by_path: Counter[str] = Counter()
        self._completed_by_path: Counter[str] = Counter()
        self._thread = threading.Thread(
            target=self._run,
            name="dronedream-runtime-evidence-writer",
            daemon=True,
        )
        self._publish_summary()
        try:
            self._thread.start()
        except BaseException:
            self._closed = self._thread_finished = True
            self._issue_code = "RUNTIME_EVIDENCE_THREAD_START_FAILED"
            self._publish_summary()
            raise

    # 功能：
    #   仅用路径正规化限定运行目录，不在高频提交路径进行文件系统解析。
    # 输入：
    #   self：固定根目录的写入器。
    #   path：候选流路径。
    # 输出：
    #   normalized：运行目录内且不同于汇总路径的绝对路径。
    def _destination(self, path: Path) -> Path:
        normalized = Path(os.path.abspath(path))
        if (not normalized.is_relative_to(self._root) or normalized == self._root
                or normalized == self.summary_path):
            raise ValueError("RUNTIME_EVIDENCE_DESTINATION_INVALID")
        return normalized

    # 功能：
    #   读取已锁定的失败类别，不向诊断输出回调异常原文或记录内容。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   issue：失败码；尚未失败时为 None。
    @property
    def issue(self) -> str | None:
        with self._condition:
            issue = self._issue_code
        return issue

    # 功能：
    #   验证目标及有限载荷并复制独立 JSON 快照，容量不足锁定失败，提交不做磁盘写入。
    # 输入：
    #   self：当前写入器。
    #   path：当前运行内的明确流路径。
    #   payload：有限 JSON 或支持的契约模型。
    # 输出：
    #   accepted：排队成功时为 True，不表示磁盘写入完成。
    def submit(self, path: Path, payload: object) -> bool:
        with self._condition:
            if self._closed or self._issue_code is not None:
                self._rejected_count += 1
                return False
        try:
            normalized_path = self._destination(path)
            owned_payload = plugin_json_value(payload, limit=MAXIMUM_EVIDENCE_RECORD_BYTES - 1)
            if type(owned_payload) is not dict:
                raise ValueError("RUNTIME_EVIDENCE_RECORD_NOT_OBJECT")
            payload_bytes = len(encode_json(owned_payload,
                limit=MAXIMUM_EVIDENCE_RECORD_BYTES - 1).encode("utf-8")) + 1
        except Exception as error:
            with self._condition:
                if self._issue_code is None:
                    self._issue_code = f"RUNTIME_EVIDENCE_SNAPSHOT_FAILED:{type(error).__name__}"
                self._rejected_count += 1
            return False
        with self._condition:
            if self._closed or self._issue_code is not None:
                self._rejected_count += 1
                return False
            if (len(self._pending) >= self.maximum_pending_records
                    or self._pending_bytes + payload_bytes > MAXIMUM_PENDING_EVIDENCE_BYTES
                    or (str(normalized_path) not in self._submitted_by_path
                        and len(self._submitted_by_path) >= MAXIMUM_EVIDENCE_STREAMS)):
                self._issue_code = "RUNTIME_EVIDENCE_QUEUE_CAPACITY_EXCEEDED"
                self._rejected_count += 1
                self._condition.notify_all()
                return False
            self._pending.append((normalized_path, owned_payload, payload_bytes))
            self._pending_bytes += payload_bytes
            self._submitted_count += 1
            self._submitted_by_path[str(normalized_path)] += 1
            self._maximum_pending_observed = max(
                self._maximum_pending_observed,
                len(self._pending),
            )
            self._condition.notify()
        accepted = True
        return accepted

    # 功能：
    #   停止接收并限时排空，超时或发布失败保留为不完整；禁止工作线程等待自身退出。
    # 输入：
    #   self：当前写入器。
    #   timeout_seconds：允许等待后台排空的秒数。
    # 输出：
    #   summary：真实队列、线程、拒绝及写入状态快照。
    def close(self, *, timeout_seconds: float = 8.0) -> dict[str, object]:
        if (type(timeout_seconds) not in (int, float)
                or not 0.0 <= timeout_seconds <= threading.TIMEOUT_MAX):
            raise ValueError("runtime evidence drain timeout must be finite and nonnegative")
        if threading.current_thread() is self._thread:
            raise ValueError("RUNTIME_EVIDENCE_SELF_CLOSE_FORBIDDEN")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout_seconds)
        with self._condition:
            if self._thread.is_alive() and self._issue_code is None:
                self._issue_code = "RUNTIME_EVIDENCE_DRAIN_TIMEOUT"
        self._publish_summary()
        summary = self.summary()
        return summary

    # 功能：
    #   在锁内生成独立状态快照，区分提交、写入、排队、拒绝及后台终止。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   summary：仅表示流持久化完整性的诊断对象，不授予飞行或模型资格。
    def summary(self) -> dict[str, object]:
        with self._condition:
            pending_count = len(self._pending)
            thread_alive = self._thread.is_alive()
            complete = bool(
                self._closed
                and self._thread_finished
                and not thread_alive
                and self._issue_code is None
                and self._rejected_count == 0
                and pending_count == 0
                and self._completed_count == self._submitted_count
            )
            paths = sorted(set(self._submitted_by_path) | set(self._completed_by_path))
            summary = {
                "schema_version": "dronedream.runtime-evidence-writer.v1",
                "writer_mode": "bounded-complete-background-fifo",
                "maximum_pending_records": self.maximum_pending_records,
                "flush_on_record_paths": sorted(str(path) for path in self._flush_on_record_paths),
                "maximum_pending_observed": self._maximum_pending_observed,
                "submitted_count": self._submitted_count,
                "completed_count": self._completed_count,
                "rejected_count": self._rejected_count,
                "pending_count": pending_count,
                "pending_bytes": self._pending_bytes,
                "maximum_pending_bytes": MAXIMUM_PENDING_EVIDENCE_BYTES,
                "closed": self._closed,
                "thread_finished": self._thread_finished,
                "thread_alive": thread_alive,
                "issue_code": self._issue_code,
                "complete": complete,
                "artifacts": [
                    {
                        "path": path,
                        "submitted_count": self._submitted_by_path[path],
                        "completed_count": self._completed_by_path[path],
                    }
                    for path in paths
                ],
                "updated_at_unix_ms": int(time.time() * 1_000),
            }
        return summary

    # 功能：
    #   在锁外调用汇总发布器，发布失败锁定诊断而不打断控制器的其他清理。
    # 输入：
    #   self：当前写入器。
    # 输出：
    #   None：不返回业务数据。
    def _publish_summary(self) -> None:
        try:
            self._summary_publisher(self.summary_path, self.summary())
        except Exception as error:
            with self._condition:
                if self._issue_code is None:
                    self._issue_code = f"RUNTIME_EVIDENCE_SUMMARY_FAILED:{type(error).__name__}"

    # 功能：
    #   检查打开句柄与当前路径仍对应同一普通文件，拒绝静态链接及文件替换。
    # 输入：
    #   self：记录初始文件身份的写入器。
    #   path：当前输出路径。
    #   handle：写入器拥有的打开句柄。
    # 输出：
    #   metadata：本次验证后的文件元数据。
    def _verify_handle(self, path: Path, handle: TextIO) -> os.stat_result:
        check_plain_plugin_path(path)
        metadata = os.fstat(handle.fileno())
        if (not stat.S_ISREG(metadata.st_mode)
                or not os.path.samestat(metadata, path.stat())
                or not os.path.samestat(metadata, self._owned_files[path])):
            raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
        return metadata

    # 功能：
    #   尝试刷新所有自有句柄，逐一复核文件身份和总字节，即使一项失败也继续刷新其他流。
    # 输入：
    #   self：当前写入器。
    #   handles：路径到自有打开句柄的映射。
    # 输出：
    #   None：不返回业务数据。
    def _flush_handles(self, handles: dict[Path, TextIO]) -> None:
        failure = None
        for path, handle in handles.items():
            try:
                self._verify_handle(path, handle)
                handle.flush()
                if self._verify_handle(path, handle).st_size != self._written_bytes[path]:
                    raise ValueError("RUNTIME_EVIDENCE_FILE_SIZE_CHANGED")
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise failure

    # 功能：
    #   1. 按 FIFO 顺序验证单行 JSON 后写入自有新流，不追加旧非空文件或跟随替换路径。
    #   2. 关键流逐条刷新，普通流批量刷新；任何写入、刷新、关闭或线程异常使回执不完整。
    # 输入：
    #   self：后台线程独占文件句柄的写入器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        handles: dict[Path, TextIO] = {}
        try:
            last_flush = time.monotonic()
            while True:
                job: tuple[Path, object, int] | None = None
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait(timeout=self.flush_interval_seconds)
                        break
                    if self._pending:
                        job = self._pending.popleft()
                        self._pending_bytes -= job[2]
                    elif self._closed:
                        break
                if job is not None:
                    path, payload, _ = job
                    text = self._serializer(payload)
                    if (type(text) is not str or "\n" in text or "\r" in text
                            or len(text) >= MAXIMUM_EVIDENCE_RECORD_BYTES
                            or not isinstance(decode_json(text,
                                limit=MAXIMUM_EVIDENCE_RECORD_BYTES - 1), dict)):
                        raise ValueError("RUNTIME_EVIDENCE_SERIALIZED_RECORD_INVALID")
                    line = text + "\n"
                    byte_count = len(line.encode("utf-8"))
                    handle = handles.get(path)
                    if handle is None:
                        check_plain_plugin_path(path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        check_plain_plugin_path(path)
                        try:
                            before = path.stat()
                        except FileNotFoundError:
                            created = path.open("x", encoding="utf-8", newline="\n")
                            try:
                                before = os.fstat(created.fileno())
                            finally:
                                created.close()
                        if not stat.S_ISREG(before.st_mode) or before.st_size != 0:
                            raise ValueError("RUNTIME_EVIDENCE_STREAM_ALREADY_HAS_DATA")
                        # 允许采集器预先独占领取的空文件；从不截断既有文件。
                        # 新建后也使用 append 语义，避免缓冲在旧偏移处覆盖外部追加的证据。
                        handle = path.open("a", encoding="utf-8", newline="\n",
                                           buffering=256 * 1024)
                        handles[path] = handle
                        opened = os.fstat(handle.fileno())
                        self._owned_files[path] = opened
                        if opened.st_size != 0 or not os.path.samestat(before, opened):
                            raise ValueError("RUNTIME_EVIDENCE_FILE_CHANGED")
                    self._verify_handle(path, handle)
                    if (self._written_bytes[path] + byte_count > MAXIMUM_EVIDENCE_FILE_BYTES
                            or self._completed_by_path[str(path)] >= MAXIMUM_EVIDENCE_FILE_RECORDS):
                        raise ValueError("RUNTIME_EVIDENCE_STREAM_BUDGET_EXCEEDED")
                    if handle.write(line) != len(line):
                        raise OSError("RUNTIME_EVIDENCE_SHORT_WRITE")
                    self._written_bytes[path] += byte_count
                    if path in self._flush_on_record_paths:
                        self._flush_handles({path: handle})
                    with self._condition:
                        self._completed_count += 1
                        self._completed_by_path[str(path)] += 1
                current = time.monotonic()
                if current - last_flush >= self.flush_interval_seconds:
                    self._flush_handles(handles)
                    last_flush = current
        except BaseException as error:
            with self._condition:
                if self._issue_code is None:
                    self._issue_code = (
                        f"RUNTIME_EVIDENCE_WRITE_FAILED:{type(error).__name__}"
                    )
        finally:
            try:
                self._flush_handles(handles)
            except BaseException as error:
                with self._condition:
                    if self._issue_code is None:
                        self._issue_code = (
                            f"RUNTIME_EVIDENCE_FINAL_FLUSH_FAILED:{type(error).__name__}"
                        )
            for handle in handles.values():
                try:
                    handle.close()
                except BaseException as error:
                    with self._condition:
                        if self._issue_code is None:
                            self._issue_code = (
                                f"RUNTIME_EVIDENCE_CLOSE_FAILED:{type(error).__name__}"
                            )
            with self._condition:
                self._thread_finished = True
                self._condition.notify_all()
