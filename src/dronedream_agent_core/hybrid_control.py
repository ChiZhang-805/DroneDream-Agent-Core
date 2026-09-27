"""Bounded model-delay arbitration; geometry and transport remain independent gates."""

from __future__ import annotations

import math
import re

from .contracts import HybridControlLease


# 功能：仅允许本地模型或显式仿真训练端口使用有界衔接，拒绝云端、无授权和观察教师。
# 输入：提供者、模型权限、教师模式及训练端口；输出：无，非法组合明确拒绝。
def require_hybrid_provider(*, provider, model_authority, teacher_control, training_channel):
    supported = provider == "local-policy" or (
        provider == "simulation-training" and training_channel is not None
    )
    if not model_authority or teacher_control or not supported:
        raise ValueError("HYBRID_REQUIRES_LOCAL_MODEL_CONTROL")


class BoundedHybridArbiter:
    """单任务有状态仲裁器；硬故障与目标变化必须等待新模型恢复。"""

    # 功能：创建未授权的接管状态，不允许程序启动即沿路线替代模型。
    # 输入：无；上限固定，避免配置错误扩大接管权限。
    # 输出：新仲裁器，尚无任何运动许可。
    def __init__(self, *, require_execution_feedback=False):
        if type(require_execution_feedback) is not bool:
            raise ValueError("HYBRID_FEEDBACK_CONFIGURATION_INVALID")
        self._require_execution_feedback = require_execution_feedback
        self._execution_receipts = ()
        self._execution_epoch_wall = None
        self._last_bridge_started_wall = None
        self._identity = None
        self._last_clock = None
        self._last_wall = None
        self._good_since = None
        self._calls = {}
        self._armed = False
        self._episode = 0
        self._start = None
        self._start_wall = None
        self._last_position = None
        self._distance = 0.0
        self._reason = "model-history-not-established"
        self._model_speed_cap = None
        self._bridge_speed_cap = None

    # 功能：公开只读诊断，不暴露或修改授权内部状态。
    # 输入：无；输出：有界统计，不能作为飞行授权。
    def snapshot(self):
        return {
            "reason": self._reason,
            "armed": self._armed,
            "recent_model_responses": len(self._calls),
            "episode": self._episode,
            "episode_distance_m": self._distance,
            "execution_feedback_required": self._require_execution_feedback,
        }

    # 功能：接纳同回合执行端广播的原始回执，坏数据清空，不把重复心跳当作新的执行。
    # 输入：最多三条(call_id, goal_id, 实际接收UNIX毫秒)；输出：无，不单独授予接管。
    def observe_execution(self, receipts):
        self._execution_receipts = ()
        if type(receipts) not in (list, tuple) or len(receipts) > 3:
            raise ValueError("HYBRID_EXECUTION_FEEDBACK_INVALID")
        parsed = []
        for row in receipts:
            if (type(row) not in (list, tuple) or len(row) != 3
                    or any(type(v) is not str or not 1 <= len(v) <= 160 for v in row[:2])
                    or type(row[2]) is not int or not 0 <= row[2] < 2**63
                    or (parsed and row[2] < parsed[-1][2])
                    or any(prior[0] == row[0] for prior in parsed)):
                raise ValueError("HYBRID_EXECUTION_FEEDBACK_INVALID")
            parsed.append(tuple(row))
        self._execution_receipts = tuple(parsed)

    # 功能：与飞控端使用相同的实际回执门槛，避免把仅完成推理的响应当作已执行预热。
    # 输入：当前目标及UNIX毫秒；输出：三条不同的新鲜执行已到位，不延长任何来源寿命。
    def _execution_ready(self, goal_id, now_unix_ms):
        receipts = self._execution_receipts
        return (len(receipts) == 3
                and all(r[1] == goal_id and 0 <= now_unix_ms - r[2] <= 1500 for r in receipts)
                and self._execution_epoch_wall is not None
                and receipts[0][2] >= self._execution_epoch_wall
                and (self._last_bridge_started_wall is None
                     or receipts[0][2] > self._last_bridge_started_wall))

    # 功能：由调用方明确撤销发生异常的接管历史，不能保留旧模型预热凭证。
    # 输入：诊断原因；输出：无。
    def invalidate(self, reason):
        self._revoke()
        self._reason = reason

    # 功能：区分真实延迟与模型合同/推理故障，后者立即撤销历史接管条件。
    # 输入：完成周期的失败代码；输出：本周期是否存在硬故障，不改写其错误记录。
    def observe_model_failure(self, code):
        delays = {
            "CONTROL_SOURCE_EXPIRED_DURING_PREPARATION",
            "CONTROL_SOURCE_EXPIRED_DURING_INFERENCE",
            "LOCAL_POLICY_CONTROL_DEADLINE_EXPIRED",
            "LOCAL_POLICY_PROCESS_INPUT_EXPIRED",
            "TRAINING_POLICY_PROPOSAL_DEFERRED",
            "CONTINUOUS_FEATURE_DISPATCH_BUDGET_INSUFFICIENT",
        }
        if code is None or code in delays:
            return False
        self.invalidate("model-cycle-failure:" + str(code))
        return True

    # 功能：撤销当前接管并要求持续的新模型响应重新武装。
    # 输入：无；不清除事件序号，防止同一任务复用旧许可。
    # 输出：无。
    def _revoke(self):
        self._armed = False
        self._good_since = None
        self._calls.clear()
        self._start = None
        self._start_wall = None
        self._last_position = None
        self._distance = 0.0
        self._model_speed_cap = None
        self._bridge_speed_cap = None
        self._execution_epoch_wall = None

    # 功能：仅在模型短时迟到且独立传感器、路线覆盖正常时授予低速衔接；
    #       路程按逐帧累计，绕圈、间歇响应和时钟回退都不能刷新接管预算。
    # 输入：双时钟、当前实测位置、路线/目标身份、新鲜安全状态、模型授权和原因。
    # 输出：独立接管许可或 None；不是已通过碰撞检查的飞行命令。
    def update(
        self,
        *,
        now: float,
        now_unix_ms: int,
        position: tuple[float, float, float],
        route_sha256: str,
        goal_id: str,
        sensors_and_route_ready: bool,
        model_live: bool,
        model_call_id: str | None,
        model_reason: str,
        model_speed_mps: float | None = None,
    ):
        if (
            not math.isfinite(now)
            or now < 0
            or type(now_unix_ms) is not int
            or now_unix_ms < 0
            or len(position) != 3
            or not all(math.isfinite(v) for v in position)
            or type(sensors_and_route_ready) is not bool
            or type(model_live) is not bool
            or not isinstance(route_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", route_sha256)
            or not isinstance(goal_id, str)
            or not 1 <= len(goal_id) <= 160
        ):
            self.invalidate("input-invalid")
            raise ValueError("HYBRID_INPUT_INVALID")
        if model_live and (
            type(model_speed_mps) not in (float, int)
            or not 0 <= model_speed_mps <= 20
            or not math.isfinite(model_speed_mps)
        ):
            self.invalidate("model-speed-evidence-missing")
            raise ValueError("HYBRID_MODEL_SPEED_REQUIRED")
        identity = (route_sha256, goal_id)
        clock_bad = self._last_clock is not None and (
            now < self._last_clock
            or now_unix_ms < self._last_wall
            or abs((now - self._last_clock) * 1000 - (now_unix_ms - self._last_wall)) > 100
        )
        self._last_clock, self._last_wall = now, now_unix_ms
        if identity != self._identity or clock_bad:
            self._revoke()
            self._identity = identity
        if clock_bad or not sensors_and_route_ready:
            self.invalidate("clock-inconsistent" if clock_bad else "independent-safety-not-ready")
            return None
        if self._start is not None:
            if self._last_position is not None:
                self._distance += math.dist(self._last_position, position)
            self._last_position = position
            if now - self._start >= 1.5 or self._distance >= 0.35:
                self.invalidate("bridge-budget-exhausted")
        # 时间窗内的不同真实响应可跨越短延迟，不能要求本来就在修复的连续性先完美。
        # 但超过窗口、重复回执或硬故障不能建立新许可。
        self._calls = {key: stamp for key, stamp in self._calls.items() if now - stamp <= 1.5}
        if not self._calls:
            self._good_since = None
            if self._start is None and self._armed:
                self.invalidate("model-response-window-expired")
        if model_live and model_call_id:
            # 阶段降速及接近目标时的低速不能因模型短暂迟到被默认0.25米/秒覆盖。
            # 只收缩独立衔接速度；不续期原模型动作、方向或传感器。零速度不能授权路线运动。
            if model_speed_mps <= 1e-6:
                self.invalidate("model-requested-zero-motion")
                return None
            # 一个接管周期的租约参数必须固定。新模型要求实质降速时撤销旧周期；
            # 数值舍入的1e-9米/秒误差不视为新命令，不把单次高速度回复写入旧租约。
            if (self._start is not None and self._bridge_speed_cap is not None
                    and model_speed_mps + 1e-9 < self._bridge_speed_cap):
                self.invalidate("model-speed-reduced-during-bridge")
            self._model_speed_cap = min(0.25, model_speed_mps)
            if self._good_since is None:
                self._good_since = now
                self._execution_epoch_wall = now_unix_ms
            if model_call_id not in self._calls:
                self._calls[model_call_id] = now
                if len(self._calls) > 3:
                    del self._calls[next(iter(self._calls))]
            if len(self._calls) >= 3 and now - self._good_since >= 0.3:
                self._armed = True
                self._start = None
                self._last_position = None
                self._distance = 0.0
            self._reason = "model-active-bridge-armed" if self._armed else "model-history-warming"
            return None
        if model_reason not in {"model-lease-expired", "model-lease-dispatch-delay"}:
            self.invalidate("not-a-model-delay:" + str(model_reason))
            return None
        if not self._armed or self._model_speed_cap is None:
            self._reason = "model-history-not-established"
            return None
        if self._start is None:
            if self._require_execution_feedback and not self._execution_ready(goal_id, now_unix_ms):
                self._reason = "actual-model-execution-not-established"
                return None
            self._start, self._start_wall = now, now_unix_ms
            self._last_bridge_started_wall = now_unix_ms
            self._bridge_speed_cap = self._model_speed_cap
            self._episode += 1
            self._calls.clear()
            self._good_since = None
        self._last_position = position
        if now - self._start >= 1.5 or self._distance >= 0.35:
            self.invalidate("bridge-budget-exhausted")
            return None
        self._reason = "bounded-delay-bridge"
        return HybridControlLease(
            route_sha256=route_sha256,
            navigation_goal_id=goal_id,
            episode=self._episode,
            started_at_unix_ms=self._start_wall,
            expires_at_unix_ms=self._start_wall + 1500,
            maximum_speed_mps=self._bridge_speed_cap,
            maximum_distance_m=0.35,
        )
