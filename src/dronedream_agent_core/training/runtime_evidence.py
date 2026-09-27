"""Read-only live evidence cursors; simulator truth never enters policy inputs."""

from __future__ import annotations

import os
import stat as file_stat
import threading
from collections import OrderedDict, deque
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from ..control_execution_evidence import ControlApplicationRecord, validate_application_binding
from ..hashing import sha256_json
from ..plugin_files import check_plain_plugin_path, read_plugin_file
from .outcome_channel import OutcomeReceiver


class JsonlCursor:
    """Incremental bounded reader of complete durable records, including split writes."""

    # 功能：
    #   初始化单文件增量游标；保留读取偏移与未写完的行，不提前创建证据文件。
    # 输入：
    #   self：当前游标。
    #   path：只读跟踪的 JSONL 文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path):
        self.path = Path(path).absolute()
        check_plain_plugin_path(self.path)
        self._offset = 0
        self._pending = b""
        self._identity = None
        self._error = None

    # 功能：
    #   1. 每次最多读取 4 MiB，只返回完整且不超过 2 MiB 的有限、无歧义 JSON 对象。
    #   2. 保留半行，拒绝已打开文件被替换、缩短或删除；游标自身锁存错误，不允许跳过坏行。
    # 输入：
    #   self：保存文件身份、读取偏移与未完成字节的游标。
    # 输出：
    #   rows：本次新增完整记录；文件尚未创建或尚无完整行时为空列表。
    def read(self) -> list[dict]:
        if self._error is not None:
            raise ValueError("TRAINING_EVIDENCE_CURSOR_FAILED") from self._error
        try:
            rows = self._read()
        except (ValueError, OSError) as error:
            self._error = error
            raise
        return rows

    # 功能：
    #   1. 读取已固定路径的普通追加文件，复核打开与读取后的身份，保留未完成的末行。
    #   2. 不证明已读历史字节不会被外部进程原地篡改，生产者权限隔离仍有必要。
    # 输入：
    #   self：保存偏移与半行的游标；上层负责锁存本方法产生的异常。
    # 输出：
    #   rows：本次完整记录。
    def _read(self) -> list[dict]:
        rows = []
        try:
            check_plain_plugin_path(self.path)
            before = self.path.stat()
            if not file_stat.S_ISREG(before.st_mode):
                raise ValueError("TRAINING_EVIDENCE_NOT_REGULAR_FILE")
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                if not os.path.samestat(before, stat) or not file_stat.S_ISREG(stat.st_mode):
                    raise ValueError("TRAINING_EVIDENCE_FILE_REPLACED_OR_TRUNCATED")
                identity = stat.st_dev, stat.st_ino
                if (
                    self._identity is not None and identity != self._identity
                ) or stat.st_size < self._offset:
                    raise ValueError("TRAINING_EVIDENCE_FILE_REPLACED_OR_TRUNCATED")
                self._identity = identity
                stream.seek(self._offset)
                expected = min(4 * 1024 * 1024, stat.st_size - self._offset)
                data = stream.read(expected)
                after = os.fstat(stream.fileno())
                named_after = self.path.stat()
                check_plain_plugin_path(self.path)
                if (len(data) != expected or after.st_size < stat.st_size
                        or not os.path.samestat(stat, named_after)
                        or (after.st_size == stat.st_size
                            and after.st_mtime_ns != stat.st_mtime_ns)):
                    raise ValueError("TRAINING_EVIDENCE_FILE_REPLACED_OR_TRUNCATED")
                # Consumers stop their monitor on any parse error. The cursor
                # does not offer recovery by replaying or skipping bad records.
                self._offset += len(data)
        except FileNotFoundError:
            if self._identity is not None:
                raise ValueError("TRAINING_EVIDENCE_FILE_DISAPPEARED") from None
            return rows
        lines = (self._pending + data).split(b"\n")
        self._pending = lines.pop()
        if len(self._pending) > 2 * 1024 * 1024:
            raise ValueError("TRAINING_EVIDENCE_RECORD_TOO_LARGE")
        for line in lines:
            if not line.strip():
                raise ValueError("TRAINING_EVIDENCE_EMPTY_RECORD")
            if len(line) > 2 * 1024 * 1024:
                raise ValueError("TRAINING_EVIDENCE_RECORD_TOO_LARGE")
            row = decode_json(line)
            if not isinstance(row, dict):
                raise ValueError("TRAINING_EVIDENCE_RECORD_NOT_OBJECT")
            rows.append(row)
        return rows


# 功能：
#   校验独立证据时间入口的 Unix 毫秒量纲，拒绝布尔、浮点、负值和超范围整数。
# 输入：
#   value：候选源时间。
# 输出：
#   stamp：通过校验的原始整数毫秒值。
def _source_timestamp(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("OUTCOME_SOURCE_TIMESTAMP_INVALID")
    stamp = value
    return stamp


class OutcomeWindowError(ValueError):
    """A rejected independent window with bounded, label-only diagnostics."""

    # 功能：
    #   汇总窗口缺口和独立观测诊断；这些原始真值只供离线评分，不进入策略输入。
    # 输入：
    #   self：窗口拒绝异常。
    #   code：稳定的失败原因。
    #   start_ms：请求窗口起始时间，毫秒。
    #   end_ms：请求窗口结束时间，毫秒。
    #   selected：已选择的窗口观测。
    #   available：有界缓存中的可用观测。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, code, *, start_ms, end_ms, selected, available):
        super().__init__(code)
        self.diagnostics = {
            "code": code,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "selected_count": len(selected),
            "available_count": len(available),
            "start_gap_ms": start_ms - selected[0].observed_at_unix_ms if selected else None,
            "end_gap_ms": end_ms - selected[-1].observed_at_unix_ms if selected else None,
            "maximum_internal_gap_ms": max(
                (
                    b.observed_at_unix_ms - a.observed_at_unix_ms
                    for a, b in zip(selected, selected[1:], strict=False)
                ),
                default=0,
            ),
            "selected_observations": [row.model_dump(mode="json") for row in selected],
            "latest_available_observations": [
                row.model_dump(mode="json") for row in available[-4:]
            ],
            "source": "independent-simulation-witness; never actor input",
            "qualification_granted": False,
        }


class IndependentPoseMonitor:
    """Bounded observation-only channel, separate from diagnostic snapshot I/O."""

    # 功能：
    #   创建最多 512 帧的独立真值缓存并启动接收线程；启动失败时释放接收通道。
    # 输入：
    #   self：当前独立观测监视器。
    #   path：本轮独立真值通道的描述文件路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, path: Path):
        self.path = path
        self._rows = deque(maxlen=512)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="training-outcome-observer"
        )
        self._receiver = OutcomeReceiver(path)
        try:
            self._thread.start()
        except BaseException:
            self._receiver.close()
            raise

    # 功能：
    #   连续接收原始真值并拒绝源时钟倒退；锁存错误后停止，不用主机到达时间刷新旧帧。
    # 输入：
    #   self：持有接收器、缓存、停止事件和错误状态的监视器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        while not self._stop.is_set():
            try:
                for observation in self._receiver.read():
                    with self._lock:
                        previous = self._rows[-1] if self._rows else None
                        if previous is not None and (
                            observation.observed_at_unix_ms <= previous.observed_at_unix_ms
                        ):
                            raise ValueError("OUTCOME_POSE_CLOCK_REGRESSED")
                        self._rows.append(observation)
            except (ValueError, OSError) as error:
                with self._lock:
                    self._error = type(error).__name__ + ":" + str(error)
                return
            self._stop.wait(0.005)

    # 功能：
    #   1. 选择起点前最近一帧及窗口内原始帧，检查边缘时差、内部缺口和每帧健康状态。
    #   2. 起始基线不等于完整窗口；不借结束时刻之后的帧插值或补造缺失真值。
    # 输入：
    #   self：独立观测缓存及错误状态。
    #   start_ms：动作窗口起始毫秒。
    #   end_ms：严格晚于起点的窗口结束毫秒。
    # 输出：
    #   witnesses：通过覆盖与健康检查的独立观测深副本列表。
    def window(self, start_ms: int, end_ms: int) -> list[RuntimeLocalSafetyObservation]:
        start_ms, end_ms = _source_timestamp(start_ms), _source_timestamp(end_ms)
        if not start_ms < end_ms:
            raise ValueError("OUTCOME_WINDOW_INVALID")
        with self._lock:
            if self._error:
                raise ValueError("OUTCOME_MONITOR_FAILED:" + self._error)
            rows = list(self._rows)
        before = [r for r in rows if r.observed_at_unix_ms <= start_ms]
        inside = [r for r in rows if start_ms < r.observed_at_unix_ms <= end_ms]
        if not before or not inside:
            raise OutcomeWindowError(
                "OUTCOME_WINDOW_NOT_YET_OBSERVED",
                start_ms=start_ms,
                end_ms=end_ms,
                selected=([before[-1]] if before else []) + inside,
                available=rows,
            )
        selected = [before[-1], *inside]
        if (
            start_ms - selected[0].observed_at_unix_ms > 100
            or end_ms - selected[-1].observed_at_unix_ms > 100
            or any(not r.stream_healthy or r.stream_age_seconds > 0.1 for r in selected)
            or any(
                b.observed_at_unix_ms - a.observed_at_unix_ms > 250
                for a, b in zip(selected, selected[1:], strict=False)
            )
        ):
            raise OutcomeWindowError(
                "OUTCOME_WINDOW_HAS_UNVERIFIED_GAP",
                start_ms=start_ms,
                end_ms=end_ms,
                selected=selected,
                available=rows,
            )
        witnesses = [r.model_copy(deep=True) for r in selected]
        return witnesses

    # 功能：
    #   判断动作起点之前是否已有新鲜健康真值，只返回时间覆盖状态，不把真值位置交给策略。
    # 输入：
    #   self：独立观测缓存及错误状态。
    #   start_ms：拟执行动作的起点毫秒。
    # 输出：
    #   available：起点前最近真值不超过 100 毫秒且源数据健康时为真。
    def has_start_baseline(self, start_ms: int) -> bool:
        start_ms = _source_timestamp(start_ms)
        available = False
        with self._lock:
            if self._error:
                raise ValueError("OUTCOME_MONITOR_FAILED:" + self._error)
            for row in reversed(self._rows):
                if row.observed_at_unix_ms <= start_ms:
                    available = (
                        start_ms - row.observed_at_unix_ms <= 100
                        and row.stream_healthy
                        and row.stream_age_seconds <= 0.1
                    )
                    break
        return available

    # 功能：
    #   为流式模仿标签取得输入时刻之前的健康独立真值，不等待动作窗口或借用未来帧。
    # 输入：
    #   self：独立真值监视器。
    #   source_ms：策略输入的源时间毫秒。
    # 输出：
    #   witness：最近且时差不超过 100 毫秒的仿真真值深副本。
    def initial_witness(self, source_ms: int) -> RuntimeLocalSafetyObservation:
        source_ms = _source_timestamp(source_ms)
        with self._lock:
            if self._error:
                raise ValueError("OUTCOME_MONITOR_FAILED:" + self._error)
            for row in reversed(self._rows):
                if row.observed_at_unix_ms <= source_ms:
                    if (source_ms - row.observed_at_unix_ms <= 100
                            and row.source == "simulation-ground-truth"
                            and row.stream_healthy and row.stream_age_seconds <= .1):
                        witness = row.model_copy(deep=True)
                        return witness
                    break
        raise ValueError("STREAM_COLLECTION_INITIAL_WITNESS_MISSING")

    # 功能：
    #   先停止并等待接收线程，再关闭通道；超时明确报错，不假装线程已回收。
    # 输入：
    #   self：当前独立观测监视器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        if self._thread.is_alive():
            raise RuntimeError("OUTCOME_MONITOR_DID_NOT_STOP")
        self._receiver.close()


class ControlReceiptMonitor:
    """Tail actual command/acceptance evidence without blocking learner inputs.

    The two independent writers may flush in either order. Joining their hashes
    is required; neither a proposal nor a command publication is an acceptance.
    """

    # 功能：
    #   独立跟踪模型控制命令与实际接受回执，用有界缓存等待两端完成摘要匹配。
    # 输入：
    #   self：控制执行回执监视器。
    #   command_path：控制命令 JSONL 路径。
    #   application_path：执行接受回执 JSONL 路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, command_path: Path, application_path: Path):
        self._commands = JsonlCursor(command_path)
        self._applications = JsonlCursor(application_path)
        self._known_commands: OrderedDict[str, RuntimeLocalSafetyCommand] = OrderedDict()
        self._known_applications: OrderedDict[int, ControlApplicationRecord] = OrderedDict()
        self._last_sequence, self._last_time = 0, -1
        self._error: Exception | None = None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="training-control-receipts", daemon=True
        )
        self._thread.start()

    # 功能：
    #   校验命令与连续接受回执，按命令摘要连接；每端最多保留 4096 项，不将发布视为接受。
    # 输入：
    #   self：命令、回执缓存及最后接受序号和时间。
    #   commands：本次新增命令记录。
    #   applications：本次新增执行接受记录。
    # 输出：
    #   None：不返回业务数据。
    def _ingest(self, commands, applications):
        with self._lock:
            for row in commands:
                if row.get("command") is not None:
                    command = RuntimeLocalSafetyCommand.model_validate(row["command"])
                    self._known_commands[sha256_json(command)] = command
            for row in applications:
                application = ControlApplicationRecord.model_validate(row)
                if (
                    application.sequence != self._last_sequence + 1
                    or application.accepted_at_unix_ms < self._last_time
                ):
                    raise ValueError("PX4_TRAINING_EXECUTION_RECEIPTS_REORDERED_OR_MISSING")
                self._last_sequence = application.sequence
                self._last_time = application.accepted_at_unix_ms
                self._known_applications[application.sequence] = application
            for cache in (self._known_commands, self._known_applications):
                while len(cache) > 4096:
                    cache.popitem(last=False)

    # 功能：
    #   周期性读取两端记录，任何读取或校验错误均锁存，防止随后仍返回旧成功回执。
    # 输入：
    #   self：当前控制回执监视器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self):
        try:
            while not self._stop.is_set():
                self._ingest(self._commands.read(), self._applications.read())
                self._stop.wait(0.01)
        except Exception as error:
            with self._lock:
                self._error = error

    # 功能：
    #   查找同时具有命令和匹配摘要接受回执的调用；返回副本，禁止调用方改变监视器证据。
    # 输入：
    #   self：当前控制回执监视器。
    #   call_id：待匹配的模型调用标识。
    # 输出：
    #   result：匹配的命令与接受回执二元组；任一端尚未齐全时为 None。
    def poll(self, call_id: str):
        result = None
        with self._lock:
            if self._error is not None:
                raise RuntimeError("TRAINING_CONTROL_RECEIPT_MONITOR_FAILED") from self._error
            for application in self._known_applications.values():
                command = self._known_commands.get(application.command_sha256)
                if command is not None and command.model_call_id == call_id:
                    result = command.model_copy(deep=True), application.model_copy(deep=True)
                    break
        return result

    # 功能：
    #   读取严格早于当前观测的最新真实回执，核对命令绑定；不以旧回执替代尚未齐全的新回执。
    # 输入：
    #   self：当前回合监视器；source_ms：原始观测 Unix 毫秒时间。
    # 输出：
    #   application：独立的已绑定回执，尚未齐全时为 None。
    def latest_before(self, source_ms: int) -> ControlApplicationRecord | None:
        if type(source_ms) is not int or not 0 <= source_ms < 2**63:
            raise ValueError('TRAINING_RECEIPT_SOURCE_CLOCK_INVALID')
        with self._lock:
            if self._error is not None:
                raise RuntimeError('TRAINING_CONTROL_RECEIPT_MONITOR_FAILED') from self._error
            for row in reversed(self._known_applications.values()):
                if row.accepted_at_unix_ms >= source_ms:
                    continue
                command = self._known_commands.get(row.command_sha256)
                if command is None:
                    return None
                validate_application_binding(command, row)
                application = row.model_copy(deep=True)
                return application
        return None

    # 功能：
    #   等待接收线程结束并报告后台错误，避免关闭操作掩盖此前的证据流失败。
    # 输入：
    #   self：当前控制回执监视器。
    # 输出：
    #   None：不返回业务数据。
    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("TRAINING_CONTROL_RECEIPT_MONITOR_DID_NOT_STOP")
        with self._lock:
            if self._error is not None:
                raise RuntimeError("TRAINING_CONTROL_RECEIPT_MONITOR_FAILED") from self._error


# 功能：
#   限量读取单个证据对象并拒绝歧义键、非有限值和过深结构；文件名本身不能证明来源。
# 输入：
#   path：需要读取的 UTF-8 JSON 证据文件。
# 输出：
#   value：结构检查通过且不超过 2 MiB 的字典。
def read_object(path: Path) -> dict:
    raw = read_plugin_file(Path(path).absolute(), limit=2 * 1024 * 1024)
    value = decode_json(raw)
    if not isinstance(value, dict):
        raise ValueError("TRAINING_EVIDENCE_NOT_OBJECT")
    return value
