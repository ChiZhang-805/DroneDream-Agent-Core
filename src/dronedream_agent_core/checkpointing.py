"""Sidecar model coordinator for non-blocking PX4 segment checkpoints."""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, decode_json

from .contracts import (
    PreparedMission,
    RuntimeAssessment,
    RuntimeCheckpointContract,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    RuntimeReplacementTrack,
)
from .extensions import ExtensionExecutionError
from .hashing import sha256_json
from .model_harness.model_port import ProviderName, StructuredModelPort
from .plugin_files import read_plugin_file
from .prompts import EXECUTION_MONITOR
from .runtime_control_io import publish_runtime_json as _atomic_json
from .runtime_plugins import (
    all_required_gates_passed,
    append_hook_receipts,
    augment_runtime_prompt,
    require_plugin_acceptance,
    runtime_extension_registry,
    validate_runtime_model_output,
)
from .runtime_revision import load_active_replacement


# 功能：
#   只读取已确认任务中编译的检查点合同；缺失时拒绝旧路径推导，并隔离可变对象。
# 输入：
#   prepared：包含冻结检查点合同的任务。
# 输出：
#   contract：重新校验、任务身份一致且没有共享可变列表的检查点合同。
def checkpoint_contract_for(prepared: PreparedMission) -> RuntimeCheckpointContract:
    if prepared.runtime_checkpoints is None:
        raise ValueError("RUNTIME_CHECKPOINT_CONTRACT_REQUIRED")
    contract = RuntimeCheckpointContract.model_validate(
        prepared.runtime_checkpoints.model_dump(mode="python"),
        strict=True,
    )
    if contract.contract_id != prepared.contract.contract_id:
        raise ValueError("RUNTIME_CHECKPOINT_CONTRACT_MISMATCH")
    if len({item.checkpoint_id for item in contract.checkpoints}) != len(contract.checkpoints):
        raise ValueError("RUNTIME_CHECKPOINT_ID_DUPLICATED")
    return contract


# 功能：
#   汇总模型明确接受与两组非空全真门控；此函数不采集遥测，也不单独证明物理安全。
# 输入：
#   request：带执行器确定性门控的检查点请求。
#   assessment：结构化模型评估。
#   binding_gates：核心计算的任务、轨迹和当前计划绑定结果。
# 输出：
#   authorized：三项授权条件同时成立时为真。
def checkpoint_continue_authorized(
    *,
    request: RuntimeCheckpointRequest,
    assessment: RuntimeAssessment,
    binding_gates: dict[str, bool],
) -> bool:
    authorized = (
        assessment.action == "accept"
        and all_required_gates_passed(request.deterministic_gates)
        and all_required_gates_passed(binding_gates)
    )
    return authorized


class CheckpointCoordinator:
    """Call the real execution-monitor model while the executor keeps hovering."""

    # 功能：
    #   绑定当前任务副本、编译检查点及冻结插件注册表，创建但不启动旁路模型协调线程。
    # 输入：
    #   self：当前协调器。
    #   prepared：已确认执行的准备任务。
    #   run_dir：本轮执行证据目录。
    #   provider：执行监视模型供应商。
    #   abort_file：当前执行器监听的中止文件。
    #   model_timeout_seconds：每次模型调用允许等待的秒数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        *,
        prepared: PreparedMission,
        run_dir: Path,
        provider: ProviderName,
        abort_file: Path,
        model_timeout_seconds: float,
    ) -> None:
        self.prepared = prepared.model_copy(deep=True)
        self.contract = checkpoint_contract_for(self.prepared)
        self.run_dir = run_dir
        self.abort_file = abort_file
        self.extensions = runtime_extension_registry(self.prepared)
        self.receipt_path = run_dir / "plugin-hook-receipts.jsonl"
        self.decisions: list[RuntimeCheckpointDecision] = []
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._publication_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="dronedream-checkpoint-coordinator", daemon=True
        )
        # 先完成可能失败的插件绑定及线程结构初始化，最后才分配需要回收的供应商连接。
        self.port = StructuredModelPort(
            provider, max_attempts=3, timeout_seconds=model_timeout_seconds
        )

    # 功能：
    #   启动模型监视旁路；已停止的协调器不能重新启动，悬停与低延迟安全仍由执行器负责。
    # 输入：
    #   self：未启动且未停止的协调器。
    # 输出：
    #   None：不返回业务数据。
    def start(self) -> None:
        with self._publication_lock:
            if self._stop.is_set():
                raise RuntimeError("checkpoint coordinator already stopped")
            self._thread.start()

    # 功能：
    #   1. 立即发出停止信号，再有界等待正在发布的决定和线程退出，成功返回后不再发布决定。
    #   2. 未启动时也撤销模型连接资格；超时不宣称远端调用已取消或线程已退出。
    # 输入：
    #   self：当前协调器。
    #   timeout_seconds：发布同步及线程回收共享的总等待秒数。
    # 输出：
    #   None：不返回业务数据。
    def stop(self, *, timeout_seconds: float = 10.0) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= threading.TIMEOUT_MAX
        ):
            raise ValueError("checkpoint coordinator stop timeout invalid")
        deadline = time.monotonic() + timeout_seconds
        self._stop.set()
        self.port.close()
        if not self._publication_lock.acquire(timeout=timeout_seconds):
            raise TimeoutError("checkpoint publication did not stop")
        try:
            started = self._thread.ident is not None
        finally:
            self._publication_lock.release()
        if not started:
            return
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("checkpoint coordinator did not stop")

    # 功能：
    #   独占发布有界中止原因，已有操作员或安全中止记录优先保留。
    # 输入：
    #   self：当前协调器。
    #   reason：需要截断为最多 240 字符的中止原因。
    # 输出：
    #   None：不返回业务数据。
    def _request_abort(self, reason: str) -> None:
        with suppress(FileExistsError):
            _atomic_json(
                self.abort_file,
                {"reason": reason[:240], "world_paused": False},
                replace_existing=False,
            )

    # 功能：
    #   在停止同步锁内复核活动计划并发布决定，只有写入成功才记录历史；磁盘换计划仍由执行器复核。
    # 输入：
    #   self：当前协调器。
    #   path：当前请求对应的决定路径。
    #   decision：完成输出校验和授权汇总的决定。
    #   control_dir：执行器活动计划记录目录。
    #   replacement：模型输入所绑定的已采纳替换计划；原计划时为 None。
    # 输出：
    #   published：决定已独占发布时为真；已请求停止时为假。
    def _publish_decision(
        self,
        path: Path,
        decision: RuntimeCheckpointDecision,
        *,
        control_dir: Path,
        replacement: RuntimeReplacementTrack | None,
    ) -> bool:
        published = False
        with self._publication_lock:
            if self._stop.is_set():
                return published
            if load_active_replacement(control_dir) != replacement:
                raise RuntimeError("checkpoint active revision changed before publication")
            # 决定文件属于单个请求。重复协调器不能覆盖已发布的不同决定。
            _atomic_json(path, decision, replace_existing=False)
            self.decisions.append(decision)
            published = True
        return published

    # 功能：
    #   1. 读取有界检查点请求，绑定当前已采纳计划、全局轨迹点和 PX4 局部坐标。
    #   2. 调用异常检测、提示增强与结构化模型，再校验输出；停止或换计划使迟到结果失效。
    #   3. 核心门控和模型接受共同决定是否继续；失败写中止证据，退出回收连接。
    #   4. 模型不直接驱动执行器；严格拒绝把整数或文本当作安全门控布尔值。
    # 输入：
    #   self：持有任务、插件、模型端口及执行目录的协调器。
    # 输出：
    #   None：不返回业务数据。
    def _run(self) -> None:
        try:
            segment_by_id = {segment.segment_id: segment for segment in self.prepared.plan.segments}
            segment_start_index_by_id: dict[str, int] = {}
            next_segment_start_index = 0
            for segment in self.prepared.plan.segments:
                segment_start_index_by_id[segment.segment_id] = next_segment_start_index
                next_segment_start_index += len(segment.path) - 1
            original_checkpoints = {item.checkpoint_id: item for item in self.contract.checkpoints}
            processed: set[str] = set()
            while not self._stop.wait(0.05):
                request_paths = sorted((self.run_dir / "checkpoints").glob("*.request.json"))
                request_path = next(
                    (
                        path
                        for path in request_paths
                        if path.stem.removesuffix(".request") not in processed
                    ),
                    None,
                )
                if request_path is None:
                    continue
                checkpoint_id = request_path.stem.removesuffix(".request")
                decision_path = self.run_dir / "checkpoints" / f"{checkpoint_id}.decision.json"
                request = RuntimeCheckpointRequest.model_validate(
                    decode_json(read_plugin_file(request_path, limit=MAX_MESSAGE_BYTES)),
                    strict=True,
                )
                if request.contract_id != self.prepared.contract.contract_id:
                    raise RuntimeError("checkpoint request contract mismatch")
                control_dir = self.run_dir / "runtime-control"
                replacement = load_active_replacement(control_dir)
                if replacement is None:
                    checkpoint = original_checkpoints.get(checkpoint_id)
                    active_task_graph = self.prepared.task_graph
                    active_track = self.prepared.px4_track
                else:
                    # Do not search historical proposals or prefer original IDs.
                    # Only the executor's adopted contract can define this checkpoint.
                    active_task_graph = replacement.revised_task_graph
                    active_track = replacement.track
                    checkpoint = None
                    if replacement.runtime_checkpoints is not None:
                        checkpoint = next(
                            (
                                item
                                for item in replacement.runtime_checkpoints.checkpoints
                                if item.checkpoint_id == checkpoint_id
                            ),
                            None,
                        )
                if checkpoint is None or active_task_graph is None:
                    raise RuntimeError(
                        f"checkpoint has no hash-bound execution artifact: {checkpoint_id}"
                    )
                if request.checkpoint != checkpoint:
                    raise RuntimeError("checkpoint request artifact mismatch")
                if replacement is None:
                    segment = segment_by_id[checkpoint.segment_id]
                else:
                    segment = SimpleNamespace(
                        segment_id=checkpoint.segment_id,
                        task_id=checkpoint.task_id,
                        from_node=replacement.route.start_node,
                        to_node=checkpoint.target_node,
                        path=replacement.track.source_world_points,
                    )
                point_index = checkpoint.track_point_index
                index_valid = type(point_index) is int and 0 <= point_index < len(
                    active_track.points
                )
                planned_point = active_track.points[point_index] if index_valid else None
                expected_ned = (
                    {
                        "north_m": planned_point.x,
                        "east_m": planned_point.y,
                        "down_m": -planned_point.z,
                    }
                    if planned_point is not None
                    else None
                )
                commanded = request.commanded_position_ned_m
                command_matches = expected_ned is not None and all(
                    abs(observed - expected) <= 1e-8
                    for observed, expected in zip(
                        (commanded.x, commanded.y, commanded.z),
                        (
                            expected_ned["north_m"],
                            expected_ned["east_m"],
                            expected_ned["down_m"],
                        ),
                        strict=True,
                    )
                )
                if replacement is None:
                    segment_start_index = segment_start_index_by_id[checkpoint.segment_id]
                    local_point_index = checkpoint.track_point_index - segment_start_index
                    checkpoint_plan_point_valid = 0 <= local_point_index < len(segment.path)
                    checkpoint_target_matches_plan_point = (
                        checkpoint_plan_point_valid
                        and checkpoint.target_node == segment.path[local_point_index].node_id
                    )
                else:
                    # Adoption was checked above. The replacement checkpoint's
                    # target belongs to that revision, not to an old plan segment.
                    checkpoint_plan_point_valid = index_valid
                    checkpoint_target_matches_plan_point = checkpoint.target_node == segment.to_node
                binding_gates = {
                    "global_track_point_index_in_range": index_valid,
                    "checkpoint_maps_to_hash_bound_plan_point": checkpoint_plan_point_valid,
                    "checkpoint_target_matches_hash_bound_plan_point": (
                        checkpoint_target_matches_plan_point
                    ),
                    "checkpoint_task_matches_segment": checkpoint.task_id == segment.task_id,
                    "px4_local_ned_command_matches_global_track_point": command_matches,
                }
                if replacement is not None:
                    binding_gates.update(
                        {
                            "replacement_gates_passed": all_required_gates_passed(
                                replacement.deterministic_gates
                            ),
                            "replacement_task_graph_hash_bound": (
                                replacement.runtime_actions is not None
                                and replacement.runtime_actions.task_graph_sha256
                                == sha256_json(active_task_graph)
                            ),
                            "replacement_checkpoint_contract_bound": (
                                replacement.runtime_checkpoints is not None
                                and checkpoint in replacement.runtime_checkpoints.checkpoints
                            ),
                        }
                    )
                checkpoint_receipts = []
                try:
                    anomaly_outputs, anomaly_receipts = self.extensions.invoke_multiple(
                        "runtime.anomaly-detectors",
                        "evaluate_checkpoint",
                        request=request,
                        checkpoint=checkpoint,
                        segment=segment,
                        prepared=self.prepared,
                    )
                    checkpoint_receipts.extend(anomaly_receipts)
                    anomaly_gates, anomaly_evaluations = require_plugin_acceptance(
                        anomaly_outputs,
                        gate_prefix="plugin_anomaly",
                    )
                    binding_gates.update(anomaly_gates)
                    instructions, prompt_receipts = augment_runtime_prompt(
                        self.extensions,
                        role="execution_monitor",
                        instructions=EXECUTION_MONITOR,
                    )
                    checkpoint_receipts.extend(prompt_receipts)
                except ExtensionExecutionError as error:
                    append_hook_receipts(self.receipt_path, [error.receipt])
                    raise
                append_hook_receipts(self.receipt_path, checkpoint_receipts)
                result = self.port.call(
                    role="execution_monitor",
                    output_type=RuntimeAssessment,
                    instructions=instructions,
                    input_artifact={
                        "mission_contract": self.prepared.contract.model_dump(mode="json"),
                        "task_graph": active_task_graph.model_dump(mode="json"),
                        "segment_summary": {
                            "segment_id": segment.segment_id,
                            "task_id": segment.task_id,
                            "from_node": segment.from_node,
                            "to_node": segment.to_node,
                            "path_point_count": len(segment.path),
                            "path_sha256": sha256_json(segment.path),
                            "coordinate_frame": "qualified map world ENU",
                        },
                        "checkpoint_request": request.model_dump(mode="json"),
                        "checkpoint_index_semantics": (
                            "global zero-based index within complete px4_track.points"
                        ),
                        "px4_coordinate_contract": (
                            active_track.coordinate_contract.model_dump(mode="json")
                        ),
                        "expected_commanded_position_ned_m": expected_ned,
                        "code_computed_binding_gates": binding_gates,
                        "plugin_anomaly_evaluations": anomaly_evaluations,
                        "immutable_rule": (
                            "all deterministic gates must be true; model output has no "
                            "actuator authority"
                        ),
                    },
                    context_id=(f"{self.prepared.contract.conversation_id}::execution_monitor"),
                )
                # Provider latency can outlive this episode. A late acceptance
                # must not write a new continuation decision after shutdown.
                if self._stop.is_set():
                    return
                # A valid response to the previous revision is still stale.
                # Compare full artifacts, since unchanged coordinates do not
                # imply unchanged task, payload or checkpoint obligations.
                if load_active_replacement(control_dir) != replacement:
                    raise RuntimeError("checkpoint active revision changed during model call")
                try:
                    output_guard_receipts = validate_runtime_model_output(
                        self.extensions,
                        role="execution_monitor",
                        expected_schema=RuntimeAssessment.__name__,
                        artifact=result.artifact,
                        record=result.record,
                    )
                except ExtensionExecutionError as error:
                    append_hook_receipts(self.receipt_path, [error.receipt])
                    raise
                checkpoint_receipts.extend(output_guard_receipts)
                append_hook_receipts(self.receipt_path, output_guard_receipts)
                decision = RuntimeCheckpointDecision(
                    request_sha256=sha256_json(request),
                    assessment=result.artifact,
                    model_call=result.record,
                    continue_authorized=checkpoint_continue_authorized(
                        request=request,
                        assessment=result.artifact,
                        binding_gates=binding_gates,
                    ),
                    plugin_hook_receipts=checkpoint_receipts,
                )
                if not self._publish_decision(
                    decision_path,
                    decision,
                    control_dir=control_dir,
                    replacement=replacement,
                ):
                    return
                processed.add(checkpoint_id)
                if not decision.continue_authorized:
                    self._request_abort(
                        f"MODEL_CHECKPOINT_{checkpoint.checkpoint_id}_"
                        f"{result.artifact.action.upper()}"
                    )
                    return
        except Exception as exc:
            self.error = exc
            self._request_abort(f"MODEL_CHECKPOINT_COORDINATOR_FAILURE_{type(exc).__name__}")
        finally:
            try:
                self.port.close()
            except Exception as error:
                if self.error is None:
                    self.error = error
                self._request_abort("MODEL_CHECKPOINT_TRANSPORT_CLEANUP_FAILURE")
            # Let filesystem observers consume the last atomic decision before exit.
            time.sleep(0.05)
