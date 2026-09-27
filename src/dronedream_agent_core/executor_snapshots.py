"""Run-owned latest snapshots; diagnostic storage cannot pace native telemetry."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .runtime_control_io import publish_runtime_json
from .runtime_phase import phase_context

ACTIVE_SNAPSHOTS: ContextVar[ExecutorSnapshots | None] = ContextVar(
    "executor_snapshots", default=None)
MAXIMUM_BYTES = 256 * 1024


class ExecutorSnapshots:
    """Only three explicit diagnostic paths; no queue of historical commands."""

    # 功能：
    #   为当前执行器的阶段、目标和遥测建立最多三条待写快照，其他文件仍使用原发布路径。
    # 输入：
    #   directory：本次运行目录。
    #   writer：在后台执行的原子 JSON 写入器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, directory: Path, *, writer=publish_runtime_json):
        self.phase_path = directory / "runtime-phase.json"
        self.target_path = directory / "local-safety-target.json"
        self.paths = frozenset((self.phase_path, self.target_path,
                               directory / "runtime-state" / "px4-identity-telemetry.json"))
        self._writer = writer
        self._condition = threading.Condition()
        self._pending: dict[Path, bytes] = {}
        self._phase: bytes | None = None
        self._target: bytes | None = None
        self._closed = False
        self._error = None
        self._submitted = self._written = self._coalesced = 0
        self._phase_events = []
        self._phase_history_overflow = False
        self._thread = threading.Thread(target=self._run, name="executor-snapshots", daemon=True)
        self._thread.start()

    # 功能：
    #   接收白名单快照并冻结有限 JSON；同一路径合并待写版本，失败时明确拒绝。
    # 输入：
    #   path：候选输出路径。
    #   payload：尚未借给后台线程的数据。
    # 输出：
    #   handled：是否由本实例接管；为假时调用方应走原同步发布路径。
    def submit(self, path: Path, payload: object) -> bool:
        if path not in self.paths:
            return False
        raw = encode_json(payload, limit=MAXIMUM_BYTES).encode("utf-8")
        if not isinstance(decode_json(raw, limit=MAXIMUM_BYTES), dict):
            raise ValueError("EXECUTOR_SNAPSHOT_OBJECT_REQUIRED")
        with self._condition:
            if self._closed or self._error is not None:
                raise RuntimeError("EXECUTOR_SNAPSHOT_WRITER_UNAVAILABLE")
            self._submitted += 1
            if path in self._pending:
                self._coalesced += 1
            self._pending[path] = raw
            if path == self.phase_path:
                context = phase_context(decode_json(raw, limit=MAXIMUM_BYTES))
                identity = (context["phase"], context["executor_phase"])
                previous = self._phase_events[-1] if self._phase_events else None
                if previous is None or identity != (previous["phase"], previous["executor_phase"]):
                    # 在磁盘快照合并前保留阶段转换；只记录标签，不存控制量或传感器数据。
                    # 容量溢出不得停止飞行，但后续数据审核必须拒绝“完整阶段证据”。
                    if len(self._phase_events) < 8192:
                        self._phase_events.append(dict(
                            sequence=len(self._phase_events) + 1,
                            at_unix_ms=time.time_ns() // 1_000_000,
                            phase=identity[0], executor_phase=identity[1],
                        ))
                    else:
                        self._phase_history_overflow = True
                self._phase = raw
            elif path == self.target_path:
                self._target = raw
            self._condition.notify()
        return True

    # 功能：
    #   返回当前执行器已提交的阶段副本，不读磁盘、不补传感器或运动权限。
    # 输入：
    #   无。
    # 输出：
    #   phase：独立阶段字典，首次发布前为空。
    def phase(self) -> dict:
        with self._condition:
            raw = self._phase
        phase = decode_json(raw, limit=MAXIMUM_BYTES) if raw is not None else {}
        return phase

    # 功能：
    #   读取执行器实际发布的目标副本；广播不能改变目标身份、来源时间或赋予运动权限。
    # 输入：
    #   无。
    # 输出：
    #   target：已提交的目标；首次发布前为 None。
    def control_target(self) -> dict | None:
        with self._condition:
            raw = self._target
        target = decode_json(raw, limit=MAXIMUM_BYTES) if raw is not None else None
        return target

    # 功能：
    #   串行写出最新快照，文件慢时不阻塞生产者，首个写入错误保留到收尾。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or bool(self._pending))
                    if not self._pending:
                        return
                    path = next(iter(self._pending))
                    raw = self._pending.pop(path)
                # Windows/WSL 的短暂读锁只能在后台有界等待，不能一次冲突
                # 就永久停掉遥测发布；持续故障仍由原发布器抛出并锁定失败。
                self._writer(path, decode_json(raw, limit=MAXIMUM_BYTES),
                             maximum_bytes=MAXIMUM_BYTES,
                             replace_timeout_seconds=0.25,
                             replace_retry_seconds=0.01)
                with self._condition:
                    self._written += 1
        except BaseException as error:
            with self._condition:
                self._error = f"{type(error).__name__}:{error}"

    # 功能：
    #   停止收新快照并等待真实写出；超时、丢写、拒写均不能报告完整。
    # 输入：
    #   timeout_seconds：有限排空等待秒数。
    # 输出：
    #   summary：写出、合并计数及收尾完整性。
    def close(self, timeout_seconds: float = 4.) -> dict:
        if type(timeout_seconds) not in (float, int) or not 0 <= timeout_seconds <= 30:
            raise ValueError("EXECUTOR_SNAPSHOT_TIMEOUT_INVALID")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout_seconds)
        with self._condition:
            complete = (not self._thread.is_alive() and self._error is None and not self._pending
                        and self._written + self._coalesced == self._submitted)
            summary = {"complete": complete, "submitted": self._submitted,
                       "written": self._written, "coalesced": self._coalesced,
                       "issue": self._error, "thread_alive": self._thread.is_alive()}
            summary["phase_history_complete"] = complete and not self._phase_history_overflow
            summary["phase_events"] = [dict(event) for event in self._phase_events]
        return summary
