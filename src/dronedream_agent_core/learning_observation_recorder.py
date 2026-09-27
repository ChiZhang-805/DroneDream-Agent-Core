"""Bounded simulation observation recording, without invoking a policy.

The recorder cannot publish controls. It uses the deployment snapshot compiler
and keeps the original command/observation binding for later execution joins.
Dropping work under load is recorded; it never creates artificial history rows.
"""

import copy
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace

from .contracts import RuntimeLocalSafetyCommand
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .hashing import sha256_json
from .navigation_snapshot import NavigationSnapshotRequest, compile_navigation_snapshot, freeze_navigation_frame
from .plugin_values import plugin_json_value
from .realtime_feature_encoders import RealtimeFeatureSnapshot


# 功能：
#   仅用有身份的停滞事件或本轮确认的动态威胁为教师观测标注恢复上下文。
#   不把普通跟踪阶段、远处动态物体或旧目标事件当作恢复示范，不授予执行权限。
# 输入：
#   target：执行器本轮目标上下文。
#   navigation_goal_id：本轮导航目标身份。
#   dynamic_episode：运行器维护的动态恢复事件。
#   dynamic_threat_confirmed：原动作确因本轮新鲜动态障碍违反净空约束，而非仅存在最近物体。
# 输出：
#   context：本轮决策触发原因和可选恢复事件身份。
def learning_recovery_task_context(target: dict, navigation_goal_id: str, dynamic_episode: dict | None, dynamic_threat_confirmed: bool) -> dict:
    if type(target) is not dict or type(dynamic_threat_confirmed) is not bool:
        raise ValueError('LEARNING_RECOVERY_CONTEXT_INVALID')
    if not isinstance(navigation_goal_id, str) or not navigation_goal_id.strip():
        raise ValueError('LEARNING_RECOVERY_GOAL_INVALID')
    context = {'decision_trigger': 'periodic', 'recovery_episode_id': None}
    episode_id = target.get('recovery_episode_id')
    if (target.get('decision_trigger') == 'progress-stalled'
            and isinstance(episode_id, str) and episode_id.strip()):
        return {'decision_trigger': 'progress-stalled', 'recovery_episode_id': episode_id}
    if dynamic_episode is not None and type(dynamic_episode) is not dict:
        raise ValueError('LEARNING_RECOVERY_EPISODE_INVALID')
    if dynamic_threat_confirmed and dynamic_episode is not None:
        episode_id = dynamic_episode.get('episode_id')
        if (dynamic_episode.get('navigation_goal_id') == navigation_goal_id
                and isinstance(episode_id, str) and episode_id.strip()):
            context = {'decision_trigger': 'dynamic-obstacle', 'recovery_episode_id': episode_id}
    return context


class LearningObservationRecorder:
    """One coordinator owns submit/poll/close; the worker never calls the sink."""

    # 功能：
    #   建立单槽后台或显式同线程观测编译器；两者都不调用策略模型或发布控制。
    # 输入：
    #   self：新观测记录器。
    #   sink：由协调线程调用、只有返回 True 才算接受记录的写入函数。
    #   synchronous：为 True 时在拥有活地图的协调线程编译，不创建后台编译线程。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, sink: Callable[[dict], bool], *, synchronous: bool = False) -> None:
        if not callable(sink):
            raise ValueError("LEARNING_OBSERVATION_SINK_INVALID")
        if type(synchronous) is not bool:
            raise ValueError("LEARNING_OBSERVATION_COMPILER_MODE_INVALID")
        self._sink = sink
        self._synchronous = synchronous
        self._owner_thread_id = threading.get_ident()
        self._executor = None if synchronous else ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="learning-observation"
        )
        self._pending: Future | None = None
        self._closed = False
        self._source: tuple[str, int, int] | None = None
        self.submitted = self.completed = self.busy_skipped = self.duplicate_skipped = 0
        self.expired_skipped = 0
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
    #   2. 忙碌和重复样本跳过；保留同槽修订号，成功排入后才推进来源基线。
    #   3. 无获准指令时仍保存有效感知历史，明确标为无动作，绝不补造命令摘要。
    # 输入：
    #   self：由单一协调线程调用的记录器。
    #   request：后台模式必须持有独立地图克隆；同步模式允许当前协调线程独占的活地图。
    #   command：用于绑定执行回执的类型化控制指令；None 表示本轮无获准指令。
    #   prepare_visual：可选视觉准备函数，在配置的同步或后台编译线程内运行。
    # 输出：
    #   accepted：成功交付本轮编译时为 True，关闭／失败／忙碌／过期／重复时为 False。
    def submit(self, request: NavigationSnapshotRequest, command: RuntimeLocalSafetyCommand | None,
               *, prepare_visual: Callable[[], list[dict]] | None = None) -> bool:
        if self._synchronous and threading.get_ident() != self._owner_thread_id:
            raise ValueError("LEARNING_OBSERVATION_WRONG_OWNER_THREAD")
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
        source = flight.source_sha256, flight.observed_at_unix_ms, flight.history_slot_revision
        if source == self._source:
            self.duplicate_skipped += 1
            return False
        if self._source is not None and (
            source[0] == self._source[0] or source[1] < self._source[1]
            or (source[1] == self._source[1] and source[2] <= self._source[2])
        ):
            raise ValueError("LEARNING_OBSERVER_SOURCE_CLOCK_CONFLICT")
        if not features.fresh_at(request.control_reference_observed_at_unix_ms):
            self.expired_skipped += 1
            return False
        # 后台地图必须由调用者克隆；同步模式在返回前完成读取，之后只保留编译结果，
        # 不让活地图逃逸到后台，也不触发下一帧的整片写时复制。
        frozen = replace(
            request,
            frame=freeze_navigation_frame(request.frame),
            health=request.health.model_copy(deep=True),
            goal_position_m=request.goal_position_m.model_copy(deep=True),
            visual_evidence=copy.deepcopy(request.visual_evidence),
            realtime_feature_snapshot=features.model_dump(mode="json"),
            multimodal_sensor_snapshot=copy.deepcopy(request.multimodal_sensor_snapshot),
            strategic_context=copy.deepcopy(request.strategic_context),
        )
        command_digest = None
        if command is not None:
            command_snapshot = RuntimeLocalSafetyCommand.model_validate(plugin_json_value(command))
            command_digest = sha256_json(command_snapshot)
        if self._synchronous:
            # 复用同一结果与失败结算路径；不是人为完成一个传感器观测或执行动作。
            pending = Future()
        else:
            pending = self._executor.submit(self._compile, frozen, command_digest, prepare_visual)
        self._pending = pending
        self._source = source
        self.submitted += 1
        if self._synchronous:
            try:
                pending.set_result(self._compile(frozen, command_digest, prepare_visual))
            except BaseException as error:
                pending.set_exception(error)
            self.poll()
        return True

    # 功能：
    #   复用部署编译器，保留观测时间与指令摘要，仅生成无控制权限的证据。
    # 输入：
    #   request：已冻结且独占地图视图的请求。
    #   command_sha256：提交时绑定的指令摘要；None 仅允许保留观测历史。
    #   prepare_visual：可选的视觉编码及持久化回调。
    # 输出：
    #   record：编译快照、分阶段耗时及明确不授予控制权的记录。
    @staticmethod
    def _compile(request: NavigationSnapshotRequest, command_sha256: str | None,
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
            "control_evaluation_status": "no-command" if command_sha256 is None else "command-bound",
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
            if self._executor is not None:
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
            "compiler_mode": "same-thread" if self._synchronous else "background-thread",
            "submitted": self.submitted,
            "completed": self.completed,
            "busy_skipped": self.busy_skipped,
            "duplicate_skipped": self.duplicate_skipped,
            "expired_skipped": self.expired_skipped,
            "issue_code": self.issue,
            "complete": (self._closed and self.issue is None and self._pending is None
                         and self.submitted == self.completed),
        }
        return summary
