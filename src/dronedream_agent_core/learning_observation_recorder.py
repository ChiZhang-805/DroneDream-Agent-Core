"""Bounded simulation observation recording, without invoking a policy.

The recorder cannot publish controls. It uses the deployment snapshot compiler
and keeps the original command/observation binding for later execution joins.
Dropping work under load is recorded; it never creates artificial history rows.
"""

import copy
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace

from .contracts import RuntimeLocalSafetyCommand
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .hashing import sha256_json
from .navigation_snapshot import NavigationSnapshotRequest, compile_navigation_snapshot
from .plugin_values import plugin_json_value
from .realtime_feature_encoders import RealtimeFeatureSnapshot


class LearningObservationRecorder:
    """One coordinator owns submit/poll/close; the worker never calls the sink."""

    # 功能：
    #   建立单任务观测编译线程与提交计数，记录器本身不调用策略模型或发布控制。
    # 输入：
    #   self：新观测记录器。
    #   sink：由协调线程调用、只有返回 True 才算接受记录的写入函数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, sink: Callable[[dict], bool]) -> None:
        if not callable(sink):
            raise ValueError("LEARNING_OBSERVATION_SINK_INVALID")
        self._sink = sink
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="learning-observation"
        )
        self._pending: Future | None = None
        self._closed = False
        self._source: tuple[str, int] | None = None
        self.submitted = self.completed = self.busy_skipped = self.duplicate_skipped = 0
        self.issue: str | None = None

    # 功能：
    #   先结算已完成任务，再判断单一编译槽是否可用；不积压观测队列。
    # 输入：
    #   self：协调线程持有的记录器。
    # 输出：
    #   ready：记录器未关闭、未失败且没有待处理任务时为 True。
    @property
    def ready_for_submission(self) -> bool:
        self.poll()
        ready = not self._closed and self.issue is None and self._pending is None
        return ready

    # 功能：
    #   1. 验证当前特征契约、原始来源时钟及连续控制模式，冻结一份可学习观测。
    #   2. 忙碌和重复样本计数后跳过；成功排入线程后才推进提交计数及来源基线。
    # 输入：
    #   self：由单一协调线程调用的记录器。
    #   request：持有独立地图克隆的编译请求，其余可变消息在本函数复制。
    #   command：用于绑定执行回执的类型化控制指令，不由本函数发送。
    #   prepare_visual：可选视觉准备函数，在工作线程内运行。
    # 输出：
    #   accepted：成功排入编译时为 True，关闭／失败／忙碌／过期／重复时为 False。
    def submit(self, request: NavigationSnapshotRequest, command: RuntimeLocalSafetyCommand,
               *, prepare_visual: Callable[[], list[dict]] | None = None) -> bool:
        self.poll()
        if self._closed or self.issue is not None:
            return False
        if self._pending is not None:
            self.busy_skipped += 1
            return False
        if not isinstance(request, NavigationSnapshotRequest):
            raise ValueError("LEARNING_OBSERVATION_REQUEST_INVALID")
        if prepare_visual is not None and not callable(prepare_visual):
            raise ValueError("LEARNING_OBSERVATION_VISUAL_CALLBACK_INVALID")
        if request.include_candidate_paths is not False:
            raise ValueError("LEARNING_OBSERVER_CANNOT_RECORD_COORDINATE_CANDIDATES")
        features = RealtimeFeatureSnapshot.model_validate(
            plugin_json_value(request.realtime_feature_snapshot)
        )
        if features.policy_feature_contract_sha256() != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
            raise ValueError("LEARNING_OBSERVER_REQUIRES_CURRENT_FEATURES")
        flight = next((e for e in features.encodings
                       if e.encoder_role == "flight-state-encoder"), None)
        if flight is None:
            raise ValueError("LEARNING_OBSERVER_REQUIRES_FLIGHT_STATE")
        source = flight.source_sha256, flight.observed_at_unix_ms
        if source == self._source:
            self.duplicate_skipped += 1
            return False
        if self._source is not None and (
            source[0] == self._source[0] or source[1] <= self._source[1]
        ):
            raise ValueError("LEARNING_OBSERVER_SOURCE_CLOCK_CONFLICT")
        if not features.fresh_at(request.control_reference_observed_at_unix_ms):
            return False
        # 地图由调用者以 navigation_clone 转移所有权；不在实时提交路径重复复制大地图。
        frozen = replace(
            request,
            frame=request.frame.model_copy(deep=True),
            health=request.health.model_copy(deep=True),
            goal_position_m=request.goal_position_m.model_copy(deep=True),
            visual_evidence=copy.deepcopy(request.visual_evidence),
            realtime_feature_snapshot=features.model_dump(mode="json"),
            multimodal_sensor_snapshot=copy.deepcopy(request.multimodal_sensor_snapshot),
            strategic_context=copy.deepcopy(request.strategic_context),
        )
        command_snapshot = RuntimeLocalSafetyCommand.model_validate(plugin_json_value(command))
        pending = self._executor.submit(
            self._compile, frozen, sha256_json(command_snapshot), prepare_visual
        )
        self._pending = pending
        self._source = source
        self.submitted += 1
        return True

    # 功能：
    #   在工作线程复用部署编译器，保留观测时间与指令摘要，仅生成无控制权限的证据。
    # 输入：
    #   request：已冻结且独占地图视图的请求。
    #   command_sha256：提交时绑定的指令摘要。
    #   prepare_visual：可选的视觉编码及持久化回调。
    # 输出：
    #   record：编译快照、分阶段耗时及明确不授予控制权的记录。
    @staticmethod
    def _compile(request: NavigationSnapshotRequest, command_sha256: str,
                 prepare_visual: Callable[[], list[dict]] | None) -> dict:
        started = time.perf_counter_ns()
        if prepare_visual is not None:
            request = replace(request, visual_evidence=prepare_visual())
        visual_finished = time.perf_counter_ns()
        snapshot = compile_navigation_snapshot(request)
        compile_finished = time.perf_counter_ns()
        record = {
            "evidence_kind": "simulation-observation-only",
            "policy_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "model_invoked": False,
            "control_authority_granted": False,
            "recorded_at_unix_ms": request.control_reference_observed_at_unix_ms,
            "evaluated_command_sha256": command_sha256,
            "snapshot": snapshot,
            "snapshot_compile_ms": (compile_finished - started) / 1_000_000,
            "snapshot_phase_ms": {
                "visual_prepare_and_persist": (visual_finished - started) / 1_000_000,
                "shared_snapshot_compile": (compile_finished - visual_finished) / 1_000_000,
            },
        }
        return record

    # 功能：
    #   由协调线程提交已完成结果，只有写入端确认才增加完成计数；失败锁定且不泄露原文。
    # 输入：
    #   self：持有单个编译 Future 的记录器。
    # 输出：
    #   None：不返回业务数据。
    def poll(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        pending, self._pending = self._pending, None
        try:
            record = pending.result()
            if self._sink(record) is not True:
                self.issue = "LEARNING_OBSERVATION_SINK_REJECTED"
                return
            self.completed += 1
        except BaseException as error:
            self.issue = f"LEARNING_OBSERVATION_FAILED:{type(error).__name__}"
            if not isinstance(error, Exception):
                raise

    # 功能：
    #   有界等待一个在途任务后关闭提交入口，超时结果不再写入接收端，不强杀工作线程。
    # 输入：
    #   self：需要关闭的记录器。
    #   timeout_seconds：允许等待编译完成的有限非负秒数。
    # 输出：
    #   summary：关闭后的提交、完成和丢弃计数及失败状态。
    def close(self, *, timeout_seconds: float = 2.0) -> dict:
        if (type(timeout_seconds) not in (int, float)
                or not 0 <= timeout_seconds <= 3600):
            raise ValueError("LEARNING_OBSERVATION_CLOSE_BUDGET_INVALID")
        self._closed = True
        try:
            if self._pending is not None:
                try:
                    self._pending.result(timeout=timeout_seconds)
                except TimeoutError:
                    if not self._pending.done():
                        self.issue = "LEARNING_OBSERVATION_DRAIN_TIMEOUT"
                        # 丢弃迟到结果，不在关闭接收端后补写；工作线程可能仍需自行退出。
                        self._pending.cancel()
                        self._pending = None
                except Exception:
                    pass  # 由 poll 统一记录工作本身的失败。
                self.poll()
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
        summary = self.summary()
        return summary

    # 功能：
    #   汇总记录完整性，关闭且全部已提交任务成功写入才算完整，不代替模型或任务验收。
    # 输入：
    #   self：待查询的记录器。
    # 输出：
    #   summary：提交／完成／跳过次数、问题代码与完整性标志。
    def summary(self) -> dict:
        summary = {
            "submitted": self.submitted,
            "completed": self.completed,
            "busy_skipped": self.busy_skipped,
            "duplicate_skipped": self.duplicate_skipped,
            "issue_code": self.issue,
            "complete": (self._closed and self.issue is None and self._pending is None
                         and self.submitted == self.completed),
        }
        return summary
