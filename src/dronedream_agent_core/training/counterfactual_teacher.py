"""Offline velocity teacher and action-conditioned swept-geometry labels.

This is an explicitly nominal, acceleration-limited counterfactual, not a
learned pilot, a physics rollout, or a calibrated collision probability. It
never publishes commands. Actual student actions and independently observed
outcomes remain separate evidence. A dynamics mismatch needs physical replay,
not a claim that this geometric label proves the action safe.
"""

import math
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Literal

from pydantic import Field, model_serializer

from ..contracts import (
    DynamicObstacleObservation,
    NormalizedPilotControl,
    QuaternionWxyz,
    StrictModel,
    Vector3,
)
from ..control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
from ..hashing import sha256_json
from ..pilot_control_mapping import PilotControlLimits, physical_pilot_request
from ..realtime_feature_encoders import body_to_world_enu, world_enu_to_body
from ..simulation_teacher import teacher_heading_rate
from .dagger import ProposedActionRisk, TeacherCorrection
from .evidence_snapshot import detach_evidence
from .flight_environment import FlightObservation, PilotAction
from .geometry_inputs import finite_metric, metric_vector
from .outcome_verifier import OutcomeEnvelope, swept_clearance_bound
from .risk_clearance_contract import (
    LEGACY_CLEARANCE_LABELS,
    OBSERVED_CLEARANCE_LABELS,
    observation_clearance_label,
)
from .risk_latency_contract import decision_latency_contract
from .swept_geometry import SweptMapGeometry


class CounterfactualConfig(StrictModel):
    """Bounded offline dynamics assumptions; these are not measured aircraft guarantees."""

    acceleration_mps2: float = Field(gt=0, le=10)
    braking_acceleration_mps2: float = Field(gt=0, le=10)
    latency_seconds: float = Field(default=0.07, ge=0, le=0.25)
    command_seconds: float = Field(default=0.25, gt=0, le=0.25)
    integration_seconds: float = Field(default=0.02, ge=0.005, le=0.02)
    required_clearance_m: float = Field(default=0.25, gt=0, le=2)
    position_uncertainty_m: float = Field(default=0.1, ge=0, le=2)
    dynamic_acceleration_bound_mps2: float = Field(default=2.0, ge=0, le=10)
    maximum_horizon_seconds: float = Field(default=5.0, gt=0, le=10)
    correction_refinement_passes: int = Field(default=4, ge=1, le=8, strict=True)
    risk_label_semantics: Literal["configured-clearance-v1", "observation-clearance-v2"] = (
        LEGACY_CLEARANCE_LABELS
    )
    decision_latency_policy: Literal["measured-actuation-v1", "bounded-source-age-v1"] = (
        "measured-actuation-v1"
    )

    # 功能：保留旧教师配置摘要；新语义显式写入身份，禁止把旧标签静默解释成新版。
    # 输入：handler：标准序列化器。
    # 输出：config：旧格式或带显式版本的新配置。
    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        config = handler(self)
        if self.risk_label_semantics == LEGACY_CLEARANCE_LABELS:
            config.pop("risk_label_semantics", None)
        if self.decision_latency_policy == "measured-actuation-v1":
            config.pop("decision_latency_policy", None)
        return config


@dataclass(frozen=True)
class CounterfactualState:
    """Source-aligned world state and independent witness identity for one hypothetical action."""

    position: Vector3
    velocity: Vector3
    orientation: QuaternionWxyz
    goal: Vector3
    dynamic_obstacles: tuple[DynamicObstacleObservation, ...]
    # An explicit, content-bound sensor/simulation witness is required even
    # when the observed dynamic list is empty. Missing is never "no obstacles".
    context_sha256: str
    additional_position_uncertainty_m: float = 0.0
    # Offline measured source-state -> actual actuator acceptance. The nominal
    # configured transport delay must not make a measured slower chain instant.
    observed_action_latency_seconds: float = 0.0


# 功能：
#   按固定 ENU 的 x、y、z 顺序提取已经验证的向量，不依赖模型字段遍历顺序。
# 输入：
#   value：三维契约向量。
# 输出：
#   components：保持坐标顺序的三元组。
def _vector(value: Vector3):
    components = value.x, value.y, value.z
    return components


# 功能：
#   按欧氏范数限制整个速度变化量，避免逐轴裁剪使合速度加速度越界。
# 输入：
#   current：当前三维速度。
#   target：目标三维速度。
#   maximum_change：本积分步允许的非负速度增量。
# 输出：
#   velocity：不超过整体增量预算的新速度。
def _approach(current, target, maximum_change):
    current, target = metric_vector(current), metric_vector(target)
    if not finite_metric(maximum_change) or maximum_change < 0:
        raise ValueError("COUNTERFACTUAL_NEGATIVE_VELOCITY_CHANGE")
    delta = tuple(b - a for a, b in zip(current, target, strict=True))
    size = math.hypot(*delta)
    if not math.isfinite(size):
        raise ValueError("COUNTERFACTUAL_VELOCITY_DELTA_OVERFLOW")
    ratio = min(1.0, maximum_change / max(size, 1e-12))
    velocity = metric_vector(tuple(a + d * ratio for a, d in zip(current, delta, strict=True)))
    return velocity


# 功能：
#   重新验证并复制源状态及有界动态物体，拒绝缺少见证身份和重复障碍标识。
# 输入：
#   state：带上下文摘要的反事实源状态。
# 输出：
#   snapshot：不共享向量、姿态和动态物体模型的状态快照。
def _state_snapshot(state: CounterfactualState) -> CounterfactualState:
    if not isinstance(state, CounterfactualState):
        raise ValueError("COUNTERFACTUAL_STATE_INVALID")
    if (
        type(state.context_sha256) is not str
        or len(state.context_sha256) != 64
        or any(c not in "0123456789abcdef" for c in state.context_sha256)
    ):
        raise ValueError("COUNTERFACTUAL_CONTEXT_IDENTITY_REQUIRED")
    if type(state.dynamic_obstacles) not in (list, tuple) or len(state.dynamic_obstacles) > 256:
        raise ValueError("COUNTERFACTUAL_DYNAMIC_LIST_INVALID")
    obstacles = tuple(
        DynamicObstacleObservation.model_validate(o.model_dump(), strict=True)
        for o in state.dynamic_obstacles
    )
    if len({o.obstacle_id for o in obstacles}) != len(obstacles):
        raise ValueError("COUNTERFACTUAL_DYNAMIC_IDENTITY_DUPLICATE")
    snapshot = replace(
        state,
        position=Vector3.model_validate(state.position.model_dump(), strict=True),
        velocity=Vector3.model_validate(state.velocity.model_dump(), strict=True),
        goal=Vector3.model_validate(state.goal.model_dump(), strict=True),
        orientation=QuaternionWxyz.model_validate(state.orientation.model_dump(), strict=True),
        dynamic_obstacles=obstacles,
    )
    return snapshot


class CounterfactualTeacher:
    """Generate offline corrections and nominal risk labels without publishing control."""

    # 功能：
    #   固定普通几何、飞机包络和名义动力学参数，建立不发布任何控制命令的离线教师。
    # 输入：
    #   self：离线反事实教师。
    #   primitives：有效地图几何原语。
    #   envelope：飞机尺寸及围栏。
    #   config：显式名义动力学假设。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self, primitives: list[dict], envelope: OutcomeEnvelope, config: CounterfactualConfig
    ):
        if not primitives:
            raise ValueError("COUNTERFACTUAL_BOUND_GEOMETRY_REQUIRED")
        primitives = detach_evidence(primitives)
        envelope = replace(envelope)
        config = CounterfactualConfig.model_validate(config.model_dump(), strict=True)
        self.geometry = SweptMapGeometry(
            primitives, radius_m=envelope.body_radius_m, half_height_m=envelope.body_height_m / 2
        )
        self.geometry_sha256 = sha256_json(primitives)
        self.envelope, self.config = envelope, config

    # 功能：
    #   1. 冻结源状态、控制及限额，积分延迟、速度指令和制动阶段。
    #   2. 合并静态几何、动态加速度不确定性和围栏余量，生成名义风险而非实飞安全证明。
    # 输入：
    #   self：已绑定地图和飞机包络的教师。
    #   state：具有独立上下文身份的观测状态。
    #   action：连续速度与转向提案。
    #   limits：归一化四轴对应的物理限额。
    # 输出：
    #   receipt：实际参与计算的输入、积分轨迹、余量和名义风险回执。
    def evaluate(
        self, state: CounterfactualState, action: PilotAction, limits: PilotControlLimits
    ) -> dict:
        state = _state_snapshot(state)
        action = PilotAction.model_validate(action.model_dump(), strict=True)
        limits = replace(limits)
        if action.mode != "pilot-control":
            raise ValueError("COUNTERFACTUAL_REQUIRES_CONTINUOUS_VELOCITY")
        if (
            not isinstance(state.context_sha256, str)
            or len(state.context_sha256) != 64
            or any(c not in "0123456789abcdef" for c in state.context_sha256)
        ):
            raise ValueError("COUNTERFACTUAL_CONTEXT_IDENTITY_REQUIRED")
        q = state.orientation
        if abs(math.hypot(q.w, q.x, q.y, q.z) - 1.0) > 1e-3:
            raise ValueError("COUNTERFACTUAL_ORIENTATION_NOT_UNIT")
        if (
            type(state.additional_position_uncertainty_m) not in (float, int)
            or not 0 <= state.additional_position_uncertainty_m <= 2.0
        ):
            raise ValueError("COUNTERFACTUAL_POSITION_UNCERTAINTY_INVALID")
        config = CounterfactualConfig.model_validate(self.config.model_dump(), strict=True)
        if (
            type(state.observed_action_latency_seconds) not in (float, int)
            or not 0 <= state.observed_action_latency_seconds <= 0.25
        ):
            raise ValueError("COUNTERFACTUAL_OBSERVED_LATENCY_INVALID")
        # Bounded comparisons also reject NaN/inf without converting huge Python ints to float.
        bounded_latency = config.decision_latency_policy == "bounded-source-age-v1"
        latency = (
            LOCAL_CONTROL_MAXIMUM_AGE_SECONDS
            if bounded_latency
            else max(config.latency_seconds, state.observed_action_latency_seconds)
        )
        control = NormalizedPilotControl(
            forward_axis=action.axes[0],
            right_axis=action.axes[1],
            up_axis=action.axes[2],
            yaw_axis=action.axes[3],
        )
        forward, right, up, yaw = physical_pilot_request(control, limits, harness_scale=1.0)
        speed = max(math.hypot(*_vector(state.velocity)), math.hypot(forward, right, up))
        horizon = (
            latency
            + config.command_seconds
            + (speed / config.braking_acceleration_mps2)
            + config.integration_seconds
        )
        if horizon > config.maximum_horizon_seconds:
            raise ValueError("COUNTERFACTUAL_BRAKING_EXCEEDS_VALIDATED_HORIZON")
        if any(o.age_seconds > 0.1 or o.confidence < 0.5 for o in state.dynamic_obstacles):
            raise ValueError("COUNTERFACTUAL_DYNAMIC_WITNESS_UNCERTAIN")
        count = math.ceil(horizon / config.integration_seconds)
        dt = horizon / count
        positions, times = [_vector(state.position)], [0.0]
        velocity = _vector(state.velocity)
        elapsed = 0.0
        for _ in range(count):
            middle = elapsed + dt / 2
            if middle < latency:
                following_velocity = velocity
            else:
                turning_time = min(config.command_seconds, max(0.0, middle - latency))
                heading = math.radians(yaw) * turning_time
                # Positive FRU yaw turns right: the instantaneous body axes
                # rotate in the original body plane before the world transform.
                body = Vector3(
                    x=forward * math.cos(heading) - right * math.sin(heading),
                    y=forward * math.sin(heading) + right * math.cos(heading),
                    z=up,
                )
                target = _vector(body_to_world_enu(q, body))
                active = middle < latency + config.command_seconds
                acceleration = (
                    config.acceleration_mps2 if active else config.braking_acceleration_mps2
                )
                following_velocity = _approach(
                    velocity, target if active else (0.0, 0.0, 0.0), acceleration * dt
                )
            positions.append(
                tuple(
                    p + (v + w) * dt / 2
                    for p, v, w in zip(positions[-1], velocity, following_velocity, strict=True)
                )
            )
            elapsed += dt
            times.append(elapsed)
            velocity = following_velocity
        # Account for integration curvature; static segment queries already
        # include their spatial sampling error and the full vehicle envelope.
        integration_margin = (
            max(config.acceleration_mps2, config.braking_acceleration_mps2) * dt * dt
        )
        # 固定源状态后，学生事前不知道本次执行最终耗时。新版只使用共同源年龄上界，
        # 不把已执行后的延迟当作可观测特征。名义恒速等待期间，不同延迟的启动位置
        # 相差至多 |v0|*预算；动态障碍也按其速度覆盖这段相对时间差。
        # 较晚执行不一定更危险（例如先远离障碍），所以不能只取最大延迟而不扩张。
        # 额外保留离散切换相位的两个时间格位移；仍只是名义包络，不是真机安全证明。
        latency_uncertainty = 0.0
        latency_contract = None
        if bounded_latency:
            obstacle_speed = max(
                (math.hypot(*_vector(o.velocity_mps)) for o in state.dynamic_obstacles), default=0.0
            )
            latency_contract = decision_latency_contract(
                math.hypot(*_vector(state.velocity)),
                math.hypot(forward, right, up),
                obstacle_speed,
                config.integration_seconds,
            )
            latency_uncertainty = latency_contract["additional_swept_uncertainty_m"]
        uncertainty = (
            config.position_uncertainty_m
            + integration_margin
            + state.additional_position_uncertainty_m
            + latency_uncertainty
        )
        minimum = self.geometry.clearance(positions) - uncertainty
        for obstacle in state.dynamic_obstacles:
            center, motion = _vector(obstacle.position_m), _vector(obstacle.velocity_mps)
            # Expand possible unmodelled dynamic acceleration across the whole
            # horizon. It is intentionally not a zero-acceleration guarantee.
            age = obstacle.age_seconds + horizon
            inflation = 0.5 * config.dynamic_acceleration_bound_mps2 * age * age
            primitive = {
                "shape": "cylinder",
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "radius_m": obstacle.radius_m + inflation,
                "height_m": obstacle.height_m + 2 * inflation,
            }
            relative = [
                tuple(
                    p - c - v * (t + obstacle.age_seconds)
                    for p, c, v in zip(point, center, motion, strict=True)
                )
                for point, t in zip(positions, times, strict=True)
            ]
            for a, b in zip(relative, relative[1:], strict=False):
                minimum = min(
                    minimum,
                    swept_clearance_bound(
                        a,
                        b,
                        [primitive],
                        radius_m=self.envelope.body_radius_m,
                        half_height_m=self.envelope.body_height_m / 2,
                    )
                    - uncertainty,
                )
        extents = (
            self.envelope.body_radius_m,
            self.envelope.body_radius_m,
            self.envelope.body_height_m / 2,
        )
        fence_margin = min(
            min(v - lo - e, hi - v - e)
            for point in positions
            for v, lo, hi, e in zip(
                point,
                self.envelope.minimum_enu_m,
                self.envelope.maximum_enu_m,
                extents,
                strict=True,
            )
        )
        clearance = min(minimum, fence_margin - uncertainty)
        risk = max(0.0, min(1.0, 1.0 - clearance / config.required_clearance_m))
        goal_progress = math.dist(positions[0], _vector(state.goal)) - math.dist(
            positions[-1], _vector(state.goal)
        )
        if not math.isfinite(goal_progress) or not math.isfinite(clearance):
            raise ValueError("COUNTERFACTUAL_RESULT_NONFINITE")
        receipt = {
            "purpose": "offline-nominal-swept-geometry",
            "risk_target": risk,
            "not_a_calibrated_collision_probability": True,
            "geometry_sha256": self.geometry_sha256,
            "context_sha256": state.context_sha256,
            "config": config.model_dump(mode="json"),
            "effective_latency_seconds": latency,
            "vehicle_envelope": asdict(self.envelope),
            "state": {
                "position": state.position.model_dump(),
                "velocity": state.velocity.model_dump(),
                "orientation": state.orientation.model_dump(),
                "goal": state.goal.model_dump(),
                "dynamic_obstacles": [o.model_dump() for o in state.dynamic_obstacles],
                "additional_position_uncertainty_m": state.additional_position_uncertainty_m,
                "observed_action_latency_seconds": state.observed_action_latency_seconds,
            },
            "action": action.model_dump(),
            "physical_request": [forward, right, up, yaw],
            "positions_m": positions,
            "times_seconds": times,
            "clearance_lower_bound_m": clearance,
            "final_speed_mps": math.hypot(*velocity),
            "goal_progress_m": goal_progress,
            "qualification_granted": False,
        }
        if bounded_latency:
            receipt["decision_latency_contract"] = latency_contract
        return receipt

    # 功能：
    #   从有界方向种子搜索并逐步精调连续轴幅度，每个候选单独通过名义零风险检查才可选择。
    # 输入：
    #   self：离线教师。
    #   observation：需要绑定标签的原始观测。
    #   state：该观测对应的世界状态。
    # 输出：
    #   correction：与冻结观测绑定的教师控制纠正。
    #   receipt：所选候选的计算与搜索证据。
    def correction(
        self, observation: FlightObservation, state: CounterfactualState
    ) -> tuple[TeacherCorrection, dict]:
        observation = FlightObservation.model_validate(observation.model_dump())
        state = _state_snapshot(state)
        limits = observation.sample.pilot_control_limits
        if limits is None:
            raise ValueError("COUNTERFACTUAL_PHYSICAL_LIMITS_REQUIRED")
        goal = world_enu_to_body(
            state.orientation,
            Vector3(
                x=state.goal.x - state.position.x,
                y=state.goal.y - state.position.y,
                z=state.goal.z - state.position.z,
            ),
        )
        distance = math.hypot(goal.x, goal.y, goal.z)
        scale = min(
            limits.horizontal_speed_mps,
            math.sqrt(2 * self.config.braking_acceleration_mps2 * max(0.0, distance - 0.1)),
        )
        desired = [
            goal.x / max(distance, 0.1) * scale / limits.horizontal_speed_mps,
            goal.y / max(distance, 0.1) * scale / limits.horizontal_speed_mps,
            max(-1.0, min(1.0, goal.z / max(distance, 0.1) * scale / limits.vertical_speed_mps)),
        ]
        yaw = teacher_heading_rate(
            orientation=state.orientation,
            position=state.position,
            goal=state.goal,
            maximum_rate_dps=limits.yaw_rate_dps,
        )
        # A coarse seed locates feasible directions. It is not the final label:
        # polish the actual continuous velocity amplitudes below, otherwise
        # nearby student states receive the same half-stick correction.
        candidates = [tuple(desired)] + list(product((-0.5, 0.0, 0.5), repeat=3))
        evaluated, seen = [], set()

        # 功能：
        #   对裁剪后的候选去重并独立评估，只保存通过名义风险检查的候选和得分。
        # 输入：
        #   axes：候选三维平移轴幅度。
        # 输出：
        #   result：得分、动作和回执；重复或不安全候选为 None。
        def consider(axes):
            axes = tuple(max(-1.0, min(1.0, value)) for value in axes)
            if axes in seen:
                return None
            seen.add(axes)
            action = PilotAction(mode="pilot-control", axes=[*axes, yaw / limits.yaw_rate_dps])
            receipt = self.evaluate(_state_snapshot(state), action.model_copy(deep=True), limits)
            if receipt["risk_target"] == 0:
                score = receipt["goal_progress_m"] - 0.005 * sum(a * a for a in axes)
                result = score, action, receipt
                evaluated.append(result)
                return result
            return None

        for axes in candidates:
            consider(axes)
        if not evaluated:
            raise ValueError("COUNTERFACTUAL_NO_VERIFIED_CORRECTION")
        best = max(evaluated, key=lambda row: row[0])
        seed_score = best[0]
        # Deterministic bounded coordinate search, with every intermediate
        # proposal independently rechecked. Smaller magnitude is not assumed
        # safe, and an unsafe point never supplies the next search centre.
        for refinement in range(self.config.correction_refinement_passes):
            step = 0.25 / 2**refinement
            centre = best[1].axes[:3]
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = list(centre)
                    candidate[axis] += sign * step
                    result = consider(candidate)
                    if result is not None and result[0] > best[0] + 1e-12:
                        best = result
        score, action, receipt = best
        receipt["correction_search"] = {
            "method": "bounded-continuous-velocity-refinement",
            "refinement_passes": self.config.correction_refinement_passes,
            "evaluated_actions": len(seen),
            "safe_actions": len(evaluated),
            "maximum_evaluations": 28 + 6 * self.config.correction_refinement_passes,
            "seed_score": seed_score,
            "selected_score": score,
            "score_definition": "goal_progress_m - 0.005 * sum(translation_axes_squared)",
            "online_controller": False,
        }
        correction = TeacherCorrection(
            observation_sha256=sha256_json(observation),
            action=action,
            verified_action_risk=receipt["risk_target"],
            verifier_receipt_sha256=sha256_json(receipt),
        )
        return correction, receipt

    # 功能：
    #   将学生原始提案的风险绑定到冻结观测，与教师纠正标签分离；非速度模式不伪造速度评分。
    # 输入：
    #   self：当前教师。
    #   observation：学生提案对应的观测。
    #   state：同一观测对应的世界状态。
    #   action：学生提出的动作。
    # 输出：
    #   risk：动作条件风险，非速度模式为 None。
    #   receipt：风险计算证据，非速度模式为 None。
    def risk(self, observation: FlightObservation, state: CounterfactualState, action: PilotAction):
        action = PilotAction.model_validate(action.model_dump(), strict=True)
        if action.mode != "pilot-control":
            return None, None
        observation = FlightObservation.model_validate(observation.model_dump())
        state = _state_snapshot(state)
        if observation.sample.pilot_control_limits is None:
            raise ValueError("COUNTERFACTUAL_PHYSICAL_LIMITS_REQUIRED")
        receipt = self.evaluate(
            state, action.model_copy(deep=True), observation.sample.pilot_control_limits
        )
        if self.config.risk_label_semantics == OBSERVED_CLEARANCE_LABELS:
            # 只替换风险监督的定义，几何积分、原始输入、不确定度和行为纠正均不改写。
            score, contract = observation_clearance_label(
                observation.sample.state_features, receipt["clearance_lower_bound_m"]
            )
            receipt["risk_target"] = score
            receipt["risk_label_contract"] = contract
        risk = ProposedActionRisk(
            observation_sha256=sha256_json(observation),
            proposed_action_sha256=sha256_json(action),
            risk=receipt["risk_target"],
            source="swept-geometry",
            verifier_receipt_sha256=sha256_json(receipt),
        )
        return risk, receipt
